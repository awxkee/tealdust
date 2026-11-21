/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! GDF error preparation of one restoration unit stripe (up to 64x64).
//!
//! The 13 source rows around the current one are kept as 10-bit normalized
//! u16 rows (8-bit samples `<< 2`, 12-bit samples `>> 2`), so one kernel
//! serves every bit depth and the taps are plain 16-bit loads.

use std::arch::x86_64::*;

use crate::gdf_tables::{GDF_ALPHA, GDF_BIAS, GDF_INTER_ERROR, GDF_INTRA_ERROR, GDF_WEIGHT};
use crate::gdf_unit::{
    GRAD_CELLS, GdfPixel, ROW_ORIGIN, ROW_STRIDE, TAP_D, TAP_H, TAP_K, TAP_V, backup_row,
};
use crate::looprestoration::{
    LR_HAVE_BOTTOM, LR_HAVE_BOTTOM_INTEGRATED, LR_HAVE_TOP, LR_HAVE_TOP_INTEGRATED,
};
use crate::rowops_dispatch::GDF_PREP_COORDS;

/// 2x2 gradient sums of the four directions for the cells of rows
/// `row_off - 1` and `row_off`; cell `i` covers columns `2 * i - 1` and `2 * i`.
///
/// # Safety
///
/// Every row pointer must address column `ROW_ORIGIN` of a `ROW_STRIDE` row.
#[inline(always)]
unsafe fn gradient_row(
    dst: &mut [[u16; 4]; GRAD_CELLS],
    rows: &[*const u16; 13],
    row_off: usize,
    w: usize,
) {
    let cells = (w + 3) >> 1;
    for (blk, dst) in dst.as_chunks_mut::<8>().0[..cells.div_ceil(8)]
        .iter_mut()
        .enumerate()
    {
        let col = 16 * blk as isize - 1;
        unsafe {
            let mut g = [_mm256_setzero_si256(); 4];
            for r in [row_off - 1, row_off] {
                // Columns -2..=80 of the origin are read at most.
                let ld =
                    |row: usize, dx: isize| _mm256_loadu_si256(rows[row].offset(col + dx).cast());
                let b2 = _mm256_slli_epi16::<1>(ld(r, 0));
                let ad = |a: __m256i, c: __m256i| {
                    _mm256_abs_epi16(_mm256_sub_epi16(_mm256_sub_epi16(b2, a), c))
                };
                g[0] = _mm256_add_epi16(g[0], ad(ld(r - 1, 0), ld(r + 1, 0)));
                g[1] = _mm256_add_epi16(g[1], ad(ld(r, -1), ld(r, 1)));
                g[2] = _mm256_add_epi16(g[2], ad(ld(r - 1, -1), ld(r + 1, 1)));
                g[3] = _mm256_add_epi16(g[3], ad(ld(r + 1, -1), ld(r - 1, 1)));
            }
            // Sum each cell's two columns, then interleave the directions.
            let g = g.map(|v| _mm256_madd_epi16(v, _mm256_set1_epi16(1)));
            let g01 = _mm256_or_si256(g[0], _mm256_slli_epi32::<16>(g[1]));
            let g23 = _mm256_or_si256(g[2], _mm256_slli_epi32::<16>(g[3]));
            let lo = _mm256_unpacklo_epi32(g01, g23);
            let hi = _mm256_unpackhi_epi32(g01, g23);
            let out: *mut __m256i = dst.as_mut_ptr().cast();
            _mm256_storeu_si256(out, _mm256_permute2x128_si256::<0x20>(lo, hi));
            _mm256_storeu_si256(out.add(1), _mm256_permute2x128_si256::<0x31>(lo, hi));
        }
    }
}

/// Exchanges both neighbors of two taps in the lanes where `keep` is clear.
#[inline(always)]
fn swap_taps(a: &mut [__m256i; 2], b: &mut [__m256i; 2], keep: __m256i) {
    for (a, b) in a.iter_mut().zip(b.iter_mut()) {
        let (va, vb) = (*a, *b);
        *a = unsafe { _mm256_blendv_epi8(vb, va, keep) };
        *b = unsafe { _mm256_blendv_epi8(va, vb, keep) };
    }
}

#[inline(always)]
fn tap_sum(d: [__m256i; 2], alpha: u16) -> __m256i {
    unsafe {
        let alpha = _mm256_set1_epi16(alpha as i16);
        let neg_alpha = _mm256_sub_epi16(_mm256_setzero_si256(), alpha);
        let a = _mm256_min_epi16(_mm256_max_epi16(d[0], neg_alpha), alpha);
        let b = _mm256_min_epi16(_mm256_max_epi16(d[1], neg_alpha), alpha);
        _mm256_min_epi16(
            _mm256_max_epi16(_mm256_add_epi16(a, b), _mm256_set1_epi16(-512)),
            _mm256_set1_epi16(511),
        )
    }
}

/// `acc += s0 * w0 + s1 * w1` per pixel. Unpacks are lane-local: `acc[0]`
/// holds pixels 0..4 and 8..12, `acc[1]` pixels 4..8 and 12..16.
#[inline(always)]
fn madd_pair(acc: &mut [__m256i; 2], s0: __m256i, s1: __m256i, w0: i16, w1: i16) {
    unsafe {
        let w = _mm256_set1_epi32(((w1 as u16 as u32) << 16 | w0 as u16 as u32) as i32);
        acc[0] = _mm256_add_epi32(acc[0], _mm256_madd_epi16(_mm256_unpacklo_epi16(s0, s1), w));
        acc[1] = _mm256_add_epi32(acc[1], _mm256_madd_epi16(_mm256_unpackhi_epi16(s0, s1), w));
    }
}

/// Rounds and clamps one accumulator to its `0..2 * scale` sub-index, as words
/// in pixel order.
///
/// Saturating the accumulator to i16 first is exact: for `scale <= 16384` any
/// `|v| >= 32767` already rounds to at least `scale`, which the clamp absorbs.
#[inline(always)]
fn sub_idx(acc: [__m256i; 2], scale: i16) -> __m256i {
    unsafe {
        let v = _mm256_max_epi16(
            _mm256_packs_epi32(acc[0], acc[1]),
            _mm256_set1_epi16(-32767),
        );
        let rounded = _mm256_mulhrs_epi16(_mm256_abs_epi16(v), _mm256_set1_epi16(scale));
        let signed = _mm256_sign_epi16(rounded, v);
        let clamped = _mm256_min_epi16(
            _mm256_max_epi16(signed, _mm256_set1_epi16(-scale)),
            _mm256_set1_epi16(scale - 1),
        );
        _mm256_add_epi16(clamped, _mm256_set1_epi16(scale))
    }
}

/// Class masks and third-output start values of one 16-pixel block; both hold
/// for the two rows sharing a gradient pair.
#[derive(Clone, Copy)]
struct BlockClass {
    /// Per-pixel word masks, set where class bit 0 / bit 1 is clear.
    keep: [__m256i; 2],
    /// Bias plus gradient terms, in `madd_pair` accumulator order.
    acc2: [__m256i; 2],
}

/// GDF error values of a `w`x`h` stripe (both at most 64) into `dst`.
///
/// Same contract as `looprestoration::gdf_prep_8bpc` / `gdf_prep_hbd`:
/// `down_shift` brings the samples to 10 bits (8-bit samples are scaled up
/// instead).
///
/// The four filter classes only permute taps: bit 0 transposes the kernel
/// (`[dy, dx]` <-> `[dx, dy]`) and bit 1 mirrors it (`dx` -> `-dx`). So the
/// neighbor differences are swapped per pixel and every pixel then uses the
/// class-0 alphas and weights, of which only 30 of the 54 tap weights are
/// ever non-zero. The gradient terms do not depend on the class and only
/// feed the third output. `gdf_table_structure` in `rowops_dispatch` checks
/// all of this against the tables.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn gdf_prep_unit_avx2<P: GdfPixel>(
    dst: &mut [i8],
    dst_stride: usize,
    p: &[P],
    p_off: usize,
    stride: usize,
    left: &[[P; 6]],
    lpf: &[P],
    lpf_off: usize,
    lpf_bottom: &[P],
    lpf_bottom_off: usize,
    w: usize,
    h: usize,
    ref_dst_idx: usize,
    qp_idx: usize,
    edges: u8,
    down_shift: u32,
) {
    assert!((1..=64).contains(&w));
    let mut bufs = [[0u16; ROW_STRIDE]; 13];
    let mut ptrs = [0usize; 13];

    backup_row(&mut bufs[6], p, p_off, Some(&left[0]), w, edges, down_shift);
    ptrs[6] = 6;
    if edges & LR_HAVE_TOP_INTEGRATED != 0 {
        for n in 0..6 {
            backup_row(
                &mut bufs[n],
                lpf,
                lpf_off + n * stride,
                None,
                w,
                edges,
                down_shift,
            );
            ptrs[n] = n;
        }
    } else if edges & LR_HAVE_TOP != 0 {
        backup_row(&mut bufs[4], lpf, lpf_off, None, w, edges, down_shift);
        backup_row(
            &mut bufs[5],
            lpf,
            lpf_off + stride,
            None,
            w,
            edges,
            down_shift,
        );
        ptrs[..6].copy_from_slice(&[4, 4, 4, 4, 4, 5]);
    } else {
        ptrs[..6].fill(6);
    }
    let mut bak_idx = 7usize;
    for y in 1..6 {
        let src_o = p_off + y * stride;
        backup_row(
            &mut bufs[bak_idx],
            p,
            src_o,
            Some(&left[y]),
            w,
            edges,
            down_shift,
        );
        ptrs[bak_idx] = bak_idx;
        bak_idx += 1;
    }

    // Class-0 alpha per tap and gradient, class-0 weight per output and tap.
    let set = ref_dst_idx * 6 + qp_idx;
    let alpha_set: &[u16; 88] = GDF_ALPHA[set * 88..][..88].try_into().unwrap();
    let weight_set: &[i16; 264] = GDF_WEIGHT[set * 264..][..264].try_into().unwrap();
    let alpha: [u16; 22] = std::array::from_fn(|k| alpha_set[k * 4]);
    let weight: [[i16; 22]; 3] =
        std::array::from_fn(|idx| std::array::from_fn(|k| weight_set[idx * 88 + k * 4]));
    let bias = GDF_BIAS[set];
    let (error_lut, scale): (&[i8], i16) = if ref_dst_idx == 0 {
        (&GDF_INTRA_ERROR[qp_idx * 4096..], 8)
    } else {
        (
            &GDF_INTER_ERROR[(ref_dst_idx - 1) * 6000 + qp_idx * 1000..],
            5,
        )
    };
    let scale2 = _mm256_set1_epi16(2 * scale);
    let ga: [i32; 4] = std::array::from_fn(|d| alpha[18 + d] as i32);
    let gw: [i32; 4] = std::array::from_fn(|d| weight[2][18 + d] as i32);
    let grad_alpha = _mm256_setr_epi32(ga[0], ga[1], ga[2], ga[3], ga[0], ga[1], ga[2], ga[3]);
    let grad_weight = _mm256_setr_epi32(gw[0], gw[1], gw[2], gw[3], gw[0], gw[1], gw[2], gw[3]);
    let bias2 = _mm256_set1_epi32(bias[2] as i32);
    let mask_order = _mm256_setr_epi32(0, 4, 2, 6, 0, 4, 2, 6);

    let row_ptrs = |bufs: &[[u16; ROW_STRIDE]; 13], ptrs: &[usize; 13]| -> [*const u16; 13] {
        std::array::from_fn(|i| bufs[ptrs[i]][ROW_ORIGIN..].as_ptr())
    };
    let mut grad = [[[0u16; 4]; GRAD_CELLS]; 2];
    // SAFETY (here and below): the pointers address column ROW_ORIGIN of
    // ROW_STRIDE-wide rows, and w <= 64 bounds every block that is read.
    unsafe { gradient_row(&mut grad[0], &row_ptrs(&bufs, &ptrs), 6, w) };
    let mut grad_bit = 1usize;
    let blocks = w.div_ceil(16);
    let zero = _mm256_setzero_si256();
    let mut class = [BlockClass {
        keep: [zero; 2],
        acc2: [zero; 2],
    }; 4];

    for y in 0..h {
        let src_o = p_off + (y + 6) * stride;
        if y + 6 < h {
            backup_row(
                &mut bufs[bak_idx],
                p,
                src_o,
                Some(&left[y + 6]),
                w,
                edges,
                down_shift,
            );
            ptrs[12] = bak_idx;
        } else if edges & LR_HAVE_BOTTOM_INTEGRATED != 0 {
            backup_row(&mut bufs[bak_idx], p, src_o, None, w, edges, down_shift);
            ptrs[12] = bak_idx;
        } else if y + 4 < h && edges & LR_HAVE_BOTTOM != 0 {
            let src_o = lpf_bottom_off + (y + 6 - h) * stride;
            backup_row(
                &mut bufs[bak_idx],
                lpf_bottom,
                src_o,
                None,
                w,
                edges,
                down_shift,
            );
            ptrs[12] = bak_idx;
        } else {
            ptrs[12] = ptrs[11];
        }
        bak_idx = (bak_idx + 1) % 13;
        let rows = row_ptrs(&bufs, &ptrs);

        if y & 1 == 0 {
            unsafe { gradient_row(&mut grad[grad_bit], &rows, 8, w) };
            grad_bit ^= 1;
            for (blk, class) in class[..blocks].iter_mut().enumerate() {
                // Eight pixel pairs; each sums its own and the next cell of
                // both gradient rows.
                let mut shared = [_mm_setzero_si128(); 4];
                let mut masks = [_mm_setzero_si128(); 4];
                for j in 0..4 {
                    let cell = blk * 8 + 2 * j;
                    let l = |g: &[[u16; 4]; GRAD_CELLS], i: usize| {
                        let pair: &[[u16; 4]; 2] = g[i..i + 2].try_into().unwrap();
                        _mm256_cvtepu16_epi32(unsafe { _mm_loadu_si128(pair.as_ptr().cast()) })
                    };
                    let sum = _mm256_add_epi32(
                        _mm256_add_epi32(l(&grad[0], cell), l(&grad[0], cell + 1)),
                        _mm256_add_epi32(l(&grad[1], cell), l(&grad[1], cell + 1)),
                    );
                    // dword 0: sum0 > sum1 (class bit 0 clear), dword 2:
                    // sum2 > sum3 (class bit 1 clear).
                    let gt = _mm256_cmpgt_epi32(sum, _mm256_shuffle_epi32::<0b10_11_00_01>(sum));
                    masks[j] = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(gt, mask_order));
                    let v = _mm256_min_epi32(_mm256_srli_epi32::<4>(sum), grad_alpha);
                    let d = _mm256_madd_epi16(v, grad_weight);
                    let d = _mm256_add_epi32(d, _mm256_shuffle_epi32::<0b01_00_11_10>(d));
                    let d = _mm256_add_epi32(d, _mm256_shuffle_epi32::<0b00_01_00_01>(d));
                    let d = _mm256_add_epi32(d, bias2);
                    shared[j] =
                        _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b00_00_10_00>(d));
                }
                class.keep = [
                    _mm256_set_m128i(
                        _mm_unpacklo_epi64(masks[2], masks[3]),
                        _mm_unpacklo_epi64(masks[0], masks[1]),
                    ),
                    _mm256_set_m128i(
                        _mm_unpackhi_epi64(masks[2], masks[3]),
                        _mm_unpackhi_epi64(masks[0], masks[1]),
                    ),
                ];
                class.acc2 = [
                    _mm256_set_m128i(shared[2], shared[0]),
                    _mm256_set_m128i(shared[3], shared[1]),
                ];
            }
        }

        let mut out = [0i8; 64];
        for (blk, (class, out)) in class[..blocks]
            .iter()
            .zip(out.as_chunks_mut::<16>().0)
            .enumerate()
        {
            let x = blk as isize * 16;
            let [keep1, keep2] = class.keep;
            let mut acc0 = [_mm256_set1_epi32(bias[0] as i32); 2];
            let mut acc1 = [_mm256_set1_epi32(bias[1] as i32); 2];
            let mut acc2 = class.acc2;

            let m = unsafe { _mm256_loadu_si256(rows[6].offset(x).cast()) };
            // Both neighbors of tap `k` minus the centre pixel.
            let diffs = |k: usize| -> [__m256i; 2] {
                let [dy, dx] = GDF_PREP_COORDS[k];
                let (dy, dx) = (dy as usize, dx as isize);
                unsafe {
                    let a = _mm256_loadu_si256(rows[6 - dy].offset(x - dx).cast());
                    let b = _mm256_loadu_si256(rows[6 + dy].offset(x + dx).cast());
                    [_mm256_sub_epi16(a, m), _mm256_sub_epi16(b, m)]
                }
            };
            // Vertical / horizontal sums at distance d = i + 1.
            let axis = |i: usize| {
                let (mut v, mut h) = (diffs(TAP_V[i]), diffs(TAP_H[i]));
                swap_taps(&mut v, &mut h, keep1);
                [tap_sum(v, alpha[TAP_V[i]]), tap_sum(h, alpha[TAP_H[i]])]
            };
            for i in 0..2 {
                let (kv, kh) = (TAP_V[i], TAP_H[i]);
                let [v, h] = axis(i);
                madd_pair(&mut acc0, v, h, weight[0][kv], weight[0][kh]);
                madd_pair(&mut acc1, v, h, weight[1][kv], weight[1][kh]);
                madd_pair(&mut acc2, v, h, weight[2][kv], weight[2][kh]);
            }
            for i in [2, 4] {
                let ([v0, h0], [v1, h1]) = (axis(i), axis(i + 1));
                if i == 2 {
                    madd_pair(&mut acc2, v0, h0, weight[2][TAP_V[2]], weight[2][TAP_H[2]]);
                }
                madd_pair(
                    &mut acc0,
                    v0,
                    v1,
                    weight[0][TAP_V[i]],
                    weight[0][TAP_V[i + 1]],
                );
                madd_pair(
                    &mut acc1,
                    h0,
                    h1,
                    weight[1][TAP_H[i]],
                    weight[1][TAP_H[i + 1]],
                );
            }

            let (mut d0, mut d1) = (diffs(TAP_D[0]), diffs(TAP_D[1]));
            swap_taps(&mut d0, &mut d1, keep2);
            let (s0, s1) = (tap_sum(d0, alpha[TAP_D[0]]), tap_sum(d1, alpha[TAP_D[1]]));
            madd_pair(&mut acc0, s0, s1, weight[0][TAP_D[0]], weight[0][TAP_D[1]]);
            madd_pair(&mut acc1, s0, s1, weight[1][TAP_D[0]], weight[1][TAP_D[1]]);

            let mut k: [[__m256i; 2]; 4] = std::array::from_fn(|i| diffs(TAP_K[i]));
            {
                let [k0, k1, k2, k3] = &mut k;
                swap_taps(k0, k2, keep1);
                swap_taps(k1, k3, keep1);
                swap_taps(k0, k1, keep2);
                swap_taps(k2, k3, keep2);
            }
            let ks: [__m256i; 4] = std::array::from_fn(|i| tap_sum(k[i], alpha[TAP_K[i]]));
            madd_pair(
                &mut acc0,
                ks[0],
                ks[1],
                weight[0][TAP_K[0]],
                weight[0][TAP_K[1]],
            );
            madd_pair(
                &mut acc1,
                ks[2],
                ks[3],
                weight[1][TAP_K[2]],
                weight[1][TAP_K[3]],
            );

            let full = _mm256_add_epi16(
                _mm256_mullo_epi16(
                    _mm256_add_epi16(
                        _mm256_mullo_epi16(sub_idx(acc0, scale), scale2),
                        sub_idx(acc1, scale),
                    ),
                    scale2,
                ),
                sub_idx(acc2, scale),
            );
            let mut lut_indices = [0u16; 16];
            unsafe { _mm256_storeu_si256(lut_indices.as_mut_ptr().cast(), full) };
            for (out, index) in out.iter_mut().zip(lut_indices) {
                *out = error_lut[index as usize];
            }
        }
        dst[y * dst_stride..][..w].copy_from_slice(&out[..w]);

        ptrs.copy_within(1.., 0);
    }
}
