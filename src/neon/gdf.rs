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
//! NEON counterpart of `avx::gdf`; see there and `gdf_unit` for the layout.

use std::arch::aarch64::*;

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
#[inline]
#[target_feature(enable = "neon")]
unsafe fn gradient_row(
    dst: &mut [[u16; 4]; GRAD_CELLS],
    rows: &[*const u16; 13],
    row_off: usize,
    w: usize,
) {
    let cells = (w + 3) >> 1;
    for (blk, dst) in dst.as_chunks_mut::<4>().0[..cells.div_ceil(4)]
        .iter_mut()
        .enumerate()
    {
        let col = 8 * blk as isize - 1;
        unsafe {
            let mut g = [vdupq_n_u16(0); 4];
            for r in [row_off - 1, row_off] {
                // Columns -2..=72 of the origin are read at most.
                let ld = |row: usize, dx: isize| {
                    vreinterpretq_s16_u16(vld1q_u16(rows[row].offset(col + dx)))
                };
                let b2 = vshlq_n_s16::<1>(ld(r, 0));
                let ad = |a: int16x8_t, c: int16x8_t| {
                    vreinterpretq_u16_s16(vabsq_s16(vsubq_s16(vsubq_s16(b2, a), c)))
                };
                g[0] = vaddq_u16(g[0], ad(ld(r - 1, 0), ld(r + 1, 0)));
                g[1] = vaddq_u16(g[1], ad(ld(r, -1), ld(r, 1)));
                g[2] = vaddq_u16(g[2], ad(ld(r - 1, -1), ld(r + 1, 1)));
                g[3] = vaddq_u16(g[3], ad(ld(r + 1, -1), ld(r - 1, 1)));
            }
            // Sum each cell's two columns, then interleave the directions.
            let d = g.map(|v| vmovn_u32(vpaddlq_u16(v)));
            vst4_u16(
                dst.as_mut_ptr().cast(),
                uint16x4x4_t(d[0], d[1], d[2], d[3]),
            );
        }
    }
}

/// Exchanges both neighbors of two taps in the lanes where `keep` is clear.
#[inline]
#[target_feature(enable = "neon")]
fn swap_taps(a: &mut [int16x8_t; 2], b: &mut [int16x8_t; 2], keep: uint16x8_t) {
    for (a, b) in a.iter_mut().zip(b.iter_mut()) {
        let (va, vb) = (*a, *b);
        *a = vbslq_s16(keep, va, vb);
        *b = vbslq_s16(keep, vb, va);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tap_sum(d: [int16x8_t; 2], alpha: u16) -> int16x8_t {
    let alpha = vdupq_n_s16(alpha as i16);
    let neg_alpha = vnegq_s16(alpha);
    let a = vminq_s16(vmaxq_s16(d[0], neg_alpha), alpha);
    let b = vminq_s16(vmaxq_s16(d[1], neg_alpha), alpha);
    vminq_s16(
        vmaxq_s16(vaddq_s16(a, b), vdupq_n_s16(-512)),
        vdupq_n_s16(511),
    )
}

/// `acc += s * w` per pixel; `acc[0]` holds pixels 0..4, `acc[1]` pixels 4..8.
#[inline]
#[target_feature(enable = "neon")]
fn mla(acc: &mut [int32x4_t; 2], s: int16x8_t, w: i16) {
    acc[0] = vmlal_n_s16(acc[0], vget_low_s16(s), w);
    acc[1] = vmlal_high_n_s16(acc[1], s, w);
}

/// Rounds and clamps one accumulator to its `0..2 * scale` sub-index.
///
/// Saturating the accumulator to i16 first is exact: for `scale <= 16384` any
/// `|v| >= 32767` already rounds to at least `scale`, which the clamp absorbs.
#[inline]
#[target_feature(enable = "neon")]
fn sub_idx(acc: [int32x4_t; 2], scale: i16) -> int16x8_t {
    let v = vmaxq_s16(
        vcombine_s16(vqmovn_s32(acc[0]), vqmovn_s32(acc[1])),
        vdupq_n_s16(-32767),
    );
    // (|v| * scale + (1 << 14)) >> 15
    let rounded = vqrdmulhq_n_s16(vabsq_s16(v), scale);
    let signed = vbslq_s16(vcltzq_s16(v), vnegq_s16(rounded), rounded);
    let clamped = vminq_s16(
        vmaxq_s16(signed, vdupq_n_s16(-scale)),
        vdupq_n_s16(scale - 1),
    );
    vaddq_s16(clamped, vdupq_n_s16(scale))
}

/// Class masks and third-output start values of one 8-pixel block; both hold
/// for the two rows sharing a gradient pair.
#[derive(Clone, Copy)]
struct BlockClass {
    /// Per-pixel masks, set where class bit 0 / bit 1 is clear.
    keep: [uint16x8_t; 2],
    /// Bias plus gradient terms.
    acc2: [int32x4_t; 2],
}

/// GDF error values of a `w`x`h` stripe (both at most 64) into `dst`.
///
/// Same contract and method as `avx::gdf::gdf_prep_unit_avx2`, eight pixels
/// per step: the filter classes are applied by swapping neighbor differences
/// per pixel, so every pixel uses the class-0 alphas and the 30 class-0
/// weights that can be non-zero. `gdf_table_structure` in `rowops_dispatch`
/// checks those table properties.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_prep_unit_neon<P: GdfPixel>(
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
            let src_o = lpf_off + n * stride;
            backup_row(&mut bufs[n], lpf, src_o, None, w, edges, down_shift);
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
    let alpha_set: &[u16; 88] = GDF_ALPHA[set * 88..].first_chunk::<88>().unwrap();
    let weight_set: &[i16; 264] = GDF_WEIGHT[set * 264..].first_chunk::<264>().unwrap();
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
    let ga: [u32; 4] = std::array::from_fn(|d| alpha[18 + d] as u32);
    let gw: [i32; 4] = std::array::from_fn(|d| weight[2][18 + d] as i32);
    let (grad_alpha, grad_weight) = unsafe { (vld1q_u32(ga.as_ptr()), vld1q_s32(gw.as_ptr())) };

    let row_ptrs = |bufs: &[[u16; ROW_STRIDE]; 13], ptrs: &[usize; 13]| -> [*const u16; 13] {
        std::array::from_fn(|i| bufs[ptrs[i]][ROW_ORIGIN..].as_ptr())
    };
    let mut grad = [[[0u16; 4]; GRAD_CELLS]; 2];
    // SAFETY (here and below): the pointers address column ROW_ORIGIN of
    // ROW_STRIDE-wide rows, and w <= 64 bounds every block that is read.
    unsafe { gradient_row(&mut grad[0], &row_ptrs(&bufs, &ptrs), 6, w) };
    let mut grad_bit = 1usize;
    let blocks = w.div_ceil(8);
    let mut class = [BlockClass {
        keep: [vdupq_n_u16(0); 2],
        acc2: [vdupq_n_s32(0); 2],
    }; 8];

    for y in 0..h {
        let src_o = p_off + (y + 6) * stride;
        if y + 6 < h {
            let left = Some(&left[y + 6]);
            backup_row(&mut bufs[bak_idx], p, src_o, left, w, edges, down_shift);
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
                // Four pixel pairs; each sums its own and the next cell of
                // both gradient rows.
                let mut keep = [[0u16; 8]; 2];
                let mut shared = [0i32; 8];
                for j in 0..4 {
                    let cell = blk * 4 + j;
                    let l =
                        |g: &[[u16; 4]; GRAD_CELLS], i: usize| unsafe { vld1_u16(g[i].as_ptr()) };
                    let sum = vaddq_u32(
                        vaddl_u16(l(&grad[0], cell), l(&grad[0], cell + 1)),
                        vaddl_u16(l(&grad[1], cell), l(&grad[1], cell + 1)),
                    );
                    // lane 0: sum0 > sum1 (class bit 0 clear), lane 2:
                    // sum2 > sum3 (class bit 1 clear).
                    let gt = vcgtq_u32(sum, vrev64q_u32(sum));
                    let v = vminq_u32(vshrq_n_u32::<4>(sum), grad_alpha);
                    let total = (bias[2] as i32)
                        .wrapping_add(vaddvq_s32(vmulq_s32(vreinterpretq_s32_u32(v), grad_weight)));
                    for px in [2 * j, 2 * j + 1] {
                        keep[0][px] = vgetq_lane_u32::<0>(gt) as u16;
                        keep[1][px] = vgetq_lane_u32::<2>(gt) as u16;
                        shared[px] = total;
                    }
                }
                unsafe {
                    class.keep = [vld1q_u16(keep[0].as_ptr()), vld1q_u16(keep[1].as_ptr())];
                    class.acc2 = [vld1q_s32(shared.as_ptr()), vld1q_s32(shared[4..].as_ptr())];
                }
            }
        }

        let mut out = [0i8; 64];
        for (blk, (class, out)) in class[..blocks]
            .iter()
            .zip(out.as_chunks_mut::<8>().0)
            .enumerate()
        {
            let x = blk as isize * 8;
            let [keep1, keep2] = class.keep;
            let mut acc0 = [vdupq_n_s32(bias[0] as i32); 2];
            let mut acc1 = [vdupq_n_s32(bias[1] as i32); 2];
            let mut acc2 = class.acc2;

            let m = unsafe { vld1q_s16(rows[6].offset(x).cast()) };
            // Both neighbors of tap `k` minus the center pixel.
            let diffs = |k: usize| -> [int16x8_t; 2] {
                let [dy, dx] = GDF_PREP_COORDS[k];
                let (dy, dx) = (dy as usize, dx as isize);
                unsafe {
                    let a = vld1q_s16(rows[6 - dy].offset(x - dx).cast());
                    let b = vld1q_s16(rows[6 + dy].offset(x + dx).cast());
                    [vsubq_s16(a, m), vsubq_s16(b, m)]
                }
            };
            // Vertical / horizontal sums at distance d = i + 1.
            let axis = |i: usize| {
                let (mut v, mut h) = (diffs(TAP_V[i]), diffs(TAP_H[i]));
                swap_taps(&mut v, &mut h, keep1);
                [tap_sum(v, alpha[TAP_V[i]]), tap_sum(h, alpha[TAP_H[i]])]
            };
            for i in 0..6 {
                let (kv, kh) = (TAP_V[i], TAP_H[i]);
                let [v, h] = axis(i);
                mla(&mut acc0, v, weight[0][kv]);
                mla(&mut acc1, h, weight[1][kh]);
                if i < 2 {
                    mla(&mut acc0, h, weight[0][kh]);
                    mla(&mut acc1, v, weight[1][kv]);
                }
                if i < 3 {
                    mla(&mut acc2, v, weight[2][kv]);
                    mla(&mut acc2, h, weight[2][kh]);
                }
            }

            let (mut d0, mut d1) = (diffs(TAP_D[0]), diffs(TAP_D[1]));
            swap_taps(&mut d0, &mut d1, keep2);
            for (d, k) in [(d0, TAP_D[0]), (d1, TAP_D[1])] {
                let s = tap_sum(d, alpha[k]);
                mla(&mut acc0, s, weight[0][k]);
                mla(&mut acc1, s, weight[1][k]);
            }

            let mut k: [[int16x8_t; 2]; 4] = std::array::from_fn(|i| diffs(TAP_K[i]));
            {
                let [k0, k1, k2, k3] = &mut k;
                swap_taps(k0, k2, keep1);
                swap_taps(k1, k3, keep1);
                swap_taps(k0, k1, keep2);
                swap_taps(k2, k3, keep2);
            }
            for i in 0..4 {
                let s = tap_sum(k[i], alpha[TAP_K[i]]);
                let (acc, idx) = if i < 2 {
                    (&mut acc0, 0)
                } else {
                    (&mut acc1, 1)
                };
                mla(acc, s, weight[idx][TAP_K[i]]);
            }

            let full = vmlaq_n_s16(
                sub_idx(acc2, scale),
                vmlaq_n_s16(sub_idx(acc1, scale), sub_idx(acc0, scale), 2 * scale),
                2 * scale,
            );
            let mut lut_indices = [0u16; 8];
            unsafe { vst1q_u16(lut_indices.as_mut_ptr(), vreinterpretq_u16_s16(full)) };
            for (out, index) in out.iter_mut().zip(lut_indices) {
                *out = error_lut[index as usize];
            }
        }
        dst[y * dst_stride..][..w].copy_from_slice(&out[..w]);

        ptrs.copy_within(1.., 0);
    }
}
