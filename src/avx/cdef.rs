/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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

use crate::avx::{_mm_hsumv_epi32, _mm256_hsum_epi32};
use crate::cdef::{CDEF_HAVE_BOTTOM, CDEF_HAVE_LEFT, CDEF_HAVE_RIGHT, CDEF_HAVE_TOP};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn cdef_fill_i16_avx2(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    let sentinel = _mm_set1_epi16(i16::MIN);
    for row in tmp.chunks_exact_mut(stride).take(h) {
        if w >= 8 {
            unsafe { _mm_storeu_si128(row.as_mut_ptr().cast(), sentinel) };
            for v in &mut row[8..w] {
                *v = i16::MIN;
            }
        } else {
            row[..w].fill(i16::MIN);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn copy_u8_to_i16_avx2(dst: &mut [i16], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = src.len();
    if n >= 8 {
        unsafe {
            let v = _mm_loadl_epi64(src.as_ptr().cast());
            _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_cvtepu8_epi16(v));
            if n > 8 {
                let s = src.as_ptr().add(n - 8);
                let d = dst.as_mut_ptr().add(n - 8);
                let v = _mm_loadl_epi64(s.cast());
                _mm_storeu_si128(d.cast(), _mm_cvtepu8_epi16(v));
            }
        }
    } else {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = s as i16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn cdef_padding_8bpc_avx2_full<const W: usize, const H: usize>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    left: &[[u8; 2]],
    top: &[u8],
    top_off: usize,
    bottom: &[u8],
    bottom_off: usize,
    bottom_stride: usize,
) {
    debug_assert!(W == 4 || W == 8);
    debug_assert!(H == 4 || H == 8);
    debug_assert!(top_off >= 2);
    debug_assert!(bottom_off >= 2);

    let o = 2 * tmp_stride + 2;
    let top_src = top_off - 2;
    let top_dst = o - 2 - 2 * tmp_stride;
    copy_u8_to_i16_avx2(
        &mut tmp[top_dst..top_dst + W + 4],
        &top[top_src..top_src + W + 4],
    );
    copy_u8_to_i16_avx2(
        &mut tmp[top_dst + tmp_stride..top_dst + tmp_stride + W + 4],
        &top[top_src + src_stride..top_src + src_stride + W + 4],
    );

    let mut soff = src_off;
    for y in 0..H {
        let ti = o + y * tmp_stride;
        tmp[ti - 2] = left[y][0] as i16;
        tmp[ti - 1] = left[y][1] as i16;
        copy_u8_to_i16_avx2(&mut tmp[ti..ti + W + 2], &src[soff..soff + W + 2]);
        soff += src_stride;
    }

    let bottom_src = bottom_off - 2;
    let bottom_dst = o - 2 + H * tmp_stride;
    copy_u8_to_i16_avx2(
        &mut tmp[bottom_dst..bottom_dst + W + 4],
        &bottom[bottom_src..bottom_src + W + 4],
    );
    copy_u8_to_i16_avx2(
        &mut tmp[bottom_dst + tmp_stride..bottom_dst + tmp_stride + W + 4],
        &bottom[bottom_src + bottom_stride..bottom_src + bottom_stride + W + 4],
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn cdef_padding_8bpc_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    left: &[[u8; 2]],
    top: &[u8],
    top_off: usize,
    bottom: &[u8],
    bottom_off: usize,
    bottom_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    const CDEF_HAVE_ALL: u8 = CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT | CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM;
    if edges == CDEF_HAVE_ALL {
        match (w, h) {
            (8, 8) => {
                cdef_padding_8bpc_avx2_full::<8, 8>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            (8, 4) => {
                cdef_padding_8bpc_avx2_full::<8, 4>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            (4, 8) => {
                cdef_padding_8bpc_avx2_full::<4, 8>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            (4, 4) => {
                cdef_padding_8bpc_avx2_full::<4, 4>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            _ => {}
        }
    }

    let o = 2 * tmp_stride + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    if edges & CDEF_HAVE_TOP == 0 {
        let base = o.wrapping_sub(2).wrapping_sub(2 * tmp_stride);
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut toff = top_off;
    for y in y_start..0 {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (toff as i32 + x_start) as usize;
        copy_u8_to_i16_avx2(&mut tmp[ti..ti + copy_w], &top[si..si + copy_w]);
        toff += src_stride;
    }

    for y in 0..h as i32 {
        let ti = (o as i32 + y * tmp_stride as i32 - 2) as usize;
        for x in x_start..0 {
            tmp[ti + (x + 2) as usize] = left[y as usize][(x + 2) as usize] as i16;
        }
    }

    let copy_w = x_end as usize;
    let mut soff = src_off;
    for y in 0..h as i32 {
        let ti = (o as i32 + y * tmp_stride as i32) as usize;
        copy_u8_to_i16_avx2(&mut tmp[ti..ti + copy_w], &src[soff..soff + copy_w]);
        soff += src_stride;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut boff = bottom_off;
    for y in h as i32..y_end {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (boff as i32 + x_start) as usize;
        copy_u8_to_i16_avx2(&mut tmp[ti..ti + copy_w], &bottom[si..si + copy_w]);
        boff += bottom_stride;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn square_weighted_sym15_avx2(p: &[i32; 15]) -> u32 {
    let a = _mm256_setr_epi32(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    let b = _mm256_setr_epi32(p[14], p[13], p[12], p[11], p[10], p[9], p[8], 0);
    let w = _mm256_setr_epi32(840, 420, 280, 210, 168, 140, 120, 105);
    let sq = _mm256_add_epi32(_mm256_mullo_epi32(a, a), _mm256_mullo_epi32(b, b));
    _mm256_hsum_epi32(_mm256_mullo_epi32(sq, w)) as u32
}

#[inline]
#[target_feature(enable = "avx2")]
fn square_weighted_alt11_avx2(p: &[i32; 11]) -> u32 {
    let a = _mm256_setr_epi32(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    let b = _mm256_setr_epi32(p[10], p[9], p[8], 0, 0, 0, 0, 0);
    let w = _mm256_setr_epi32(420, 210, 140, 105, 105, 105, 105, 105);
    let sq = _mm256_add_epi32(_mm256_mullo_epi32(a, a), _mm256_mullo_epi32(b, b));
    _mm256_hsum_epi32(_mm256_mullo_epi32(sq, w)) as u32
}

#[inline]
#[target_feature(enable = "avx2")]
fn finish_cdef_dir_avx2(
    partial_sum_hv: &[[i32; 8]; 2],
    partial_sum_diag: &[[i32; 15]; 2],
    partial_sum_alt: &[[i32; 11]; 4],
    var: &mut u32,
) -> i32 {
    let hv0 = unsafe { _mm256_loadu_si256(partial_sum_hv[0].as_ptr().cast()) };
    let hv1 = unsafe { _mm256_loadu_si256(partial_sum_hv[1].as_ptr().cast()) };
    let mut cost = [0u32; 8];
    cost[2] = _mm256_hsum_epi32(_mm256_mullo_epi32(hv0, hv0)) as u32 * 105;
    cost[6] = _mm256_hsum_epi32(_mm256_mullo_epi32(hv1, hv1)) as u32 * 105;
    cost[0] = square_weighted_sym15_avx2(&partial_sum_diag[0]);
    cost[4] = square_weighted_sym15_avx2(&partial_sum_diag[1]);
    cost[1] = square_weighted_alt11_avx2(&partial_sum_alt[0]);
    cost[3] = square_weighted_alt11_avx2(&partial_sum_alt[1]);
    cost[5] = square_weighted_alt11_avx2(&partial_sum_alt[2]);
    cost[7] = square_weighted_alt11_avx2(&partial_sum_alt[3]);

    let mut best_dir = 0i32;
    let mut best_cost = cost[0];
    for (n, &c) in cost.iter().enumerate().skip(1) {
        if c > best_cost {
            best_cost = c;
            best_dir = n as i32;
        }
    }

    *var = (best_cost - cost[(best_dir ^ 4) as usize]) >> 10;
    best_dir
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum_i16x8(v: __m128i) -> i32 {
    _mm_cvtsi128_si32(_mm_hsumv_epi32(_mm_madd_epi16(v, _mm_set1_epi16(1)))) as i32
}

#[inline]
#[target_feature(enable = "avx2")]
fn reverse_i16x8(v: __m128i) -> __m128i {
    let shuf = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    _mm_shuffle_epi8(v, shuf)
}

#[inline]
#[target_feature(enable = "avx2")]
fn reverse_i16x4_low(v: __m128i) -> __m128i {
    let shuf = _mm_setr_epi8(6, 7, 4, 5, 2, 3, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1);
    _mm_shuffle_epi8(v, shuf)
}

#[inline]
#[target_feature(enable = "avx2")]
fn pair_sum_i16x8(v: __m128i) -> __m128i {
    _mm_hadd_epi16(v, _mm_setzero_si128())
}

#[inline]
#[target_feature(enable = "avx2")]
fn shl_words_i16x8(v: __m128i, n: usize) -> __m128i {
    match n {
        0 => v,
        1 => _mm_slli_si128::<2>(v),
        2 => _mm_slli_si128::<4>(v),
        3 => _mm_slli_si128::<6>(v),
        4 => _mm_slli_si128::<8>(v),
        5 => _mm_slli_si128::<10>(v),
        6 => _mm_slli_si128::<12>(v),
        7 => _mm_slli_si128::<14>(v),
        _ => _mm_setzero_si128(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn overflow_words_i16x8(v: __m128i, n: usize) -> __m128i {
    match n {
        0 => _mm_setzero_si128(),
        1 => _mm_srli_si128::<14>(v),
        2 => _mm_srli_si128::<12>(v),
        3 => _mm_srli_si128::<10>(v),
        4 => _mm_srli_si128::<8>(v),
        5 => _mm_srli_si128::<6>(v),
        6 => _mm_srli_si128::<4>(v),
        7 => _mm_srli_si128::<2>(v),
        _ => v,
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_shifted_i16x8(lo: &mut __m128i, hi: &mut __m128i, v: __m128i, n: usize) {
    *lo = _mm_add_epi16(*lo, shl_words_i16x8(v, n));
    *hi = _mm_add_epi16(*hi, overflow_words_i16x8(v, n));
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x4_prefix(dst: &mut [i32], v: __m128i, n: usize) {
    match n {
        0 => {}
        1 => unsafe { _mm_store_ss(dst.as_mut_ptr().cast(), _mm_castsi128_ps(v)) },
        2 => unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast(), v) },
        3 => {
            unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast(), v) };
            dst[2] = _mm_extract_epi32::<2>(v);
        }
        _ => unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), v) },
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8_to_i32(dst: &mut [i32], v: __m128i, n: usize) {
    let lo = _mm_cvtepi16_epi32(v);
    let hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v));
    let lo_n = n.min(4);

    store_i32x4_prefix(dst, lo, lo_n);
    if n > 4 {
        store_i32x4_prefix(&mut dst[4..], hi, n - 4);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn cdef_find_dir_from_rows_avx2(rows: &[__m128i; 8], var: &mut u32) -> i32 {
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    let zero = _mm_setzero_si128();
    let mut col_sum = zero;
    let mut diag0_lo = zero;
    let mut diag0_hi = zero;
    let mut diag1_lo = zero;
    let mut diag1_hi = zero;
    let mut alt0_lo = zero;
    let mut alt0_hi = zero;
    let mut alt1_lo = zero;
    let mut alt1_hi = zero;
    let mut alt2_lo = zero;
    let mut alt2_hi = zero;
    let mut alt3_lo = zero;
    let mut alt3_hi = zero;

    for y in 0..8usize {
        let row = rows[y];
        let rev = reverse_i16x8(row);
        let pair = pair_sum_i16x8(row);
        let pair_rev = reverse_i16x4_low(pair);

        partial_sum_hv[0][y] = hsum_i16x8(row);
        col_sum = _mm_add_epi16(col_sum, row);

        add_shifted_i16x8(&mut diag0_lo, &mut diag0_hi, row, y);
        add_shifted_i16x8(&mut diag1_lo, &mut diag1_hi, rev, y);
        add_shifted_i16x8(&mut alt0_lo, &mut alt0_hi, pair, y);
        add_shifted_i16x8(&mut alt1_lo, &mut alt1_hi, pair_rev, y);

        let half_y = y >> 1;
        add_shifted_i16x8(
            &mut alt2_lo,
            &mut alt2_hi,
            row,
            3usize.saturating_sub(half_y),
        );
        add_shifted_i16x8(&mut alt3_lo, &mut alt3_hi, row, half_y);
    }

    store_i16x8_to_i32(&mut partial_sum_hv[1], col_sum, 8);

    store_i16x8_to_i32(&mut partial_sum_diag[0][..8], diag0_lo, 8);
    store_i16x8_to_i32(&mut partial_sum_diag[0][8..], diag0_hi, 7);
    store_i16x8_to_i32(&mut partial_sum_diag[1][..8], diag1_lo, 8);
    store_i16x8_to_i32(&mut partial_sum_diag[1][8..], diag1_hi, 7);
    store_i16x8_to_i32(&mut partial_sum_alt[0][..8], alt0_lo, 8);
    store_i16x8_to_i32(&mut partial_sum_alt[0][8..], alt0_hi, 3);
    store_i16x8_to_i32(&mut partial_sum_alt[1][..8], alt1_lo, 8);
    store_i16x8_to_i32(&mut partial_sum_alt[1][8..], alt1_hi, 3);
    store_i16x8_to_i32(&mut partial_sum_alt[2][..8], alt2_lo, 8);
    store_i16x8_to_i32(&mut partial_sum_alt[2][8..], alt2_hi, 3);
    store_i16x8_to_i32(&mut partial_sum_alt[3][..8], alt3_lo, 8);
    store_i16x8_to_i32(&mut partial_sum_alt[3][8..], alt3_hi, 3);

    finish_cdef_dir_avx2(&partial_sum_hv, &partial_sum_diag, &partial_sum_alt, var)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x16_2rows(tmp: &[i16], p0: isize, p1: isize, off: isize) -> __m256i {
    let lo = unsafe { _mm_loadu_si128(tmp.as_ptr().offset(p0 + off).cast()) };
    let hi = unsafe { _mm_loadu_si128(tmp.as_ptr().offset(p1 + off).cast()) };
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16xw_2rows<const W: usize>(tmp: &[i16], p0: isize, p1: isize, off: isize) -> __m256i {
    debug_assert!(W == 4 || W == 8);
    unsafe {
        let lo = if W == 8 {
            _mm_loadu_si128(tmp.as_ptr().offset(p0 + off).cast())
        } else {
            _mm_loadl_epi64(tmp.as_ptr().offset(p0 + off).cast())
        };
        let hi = if W == 8 {
            _mm_loadu_si128(tmp.as_ptr().offset(p1 + off).cast())
        } else {
            _mm_loadl_epi64(tmp.as_ptr().offset(p1 + off).cast())
        };
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn cdef_min_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epu16(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
fn constrain_i16(diff: __m256i, threshold: __m256i, shc: __m128i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let adiff = _mm256_abs_epi16(diff);
    let t = _mm256_max_epi16(
        zero,
        _mm256_sub_epi16(threshold, _mm256_srl_epi16(adiff, shc)),
    );
    let m = _mm256_min_epu16(adiff, t);
    _mm256_blendv_epi8(m, _mm256_sub_epi16(zero, m), _mm256_cmpgt_epi16(zero, diff))
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_tap_i16(v: __m256i, tap: i32) -> __m256i {
    match tap {
        1 => v,
        2 => _mm256_add_epi16(v, v),
        3 => _mm256_add_epi16(_mm256_add_epi16(v, v), v),
        4 => _mm256_slli_epi16::<2>(v),
        _ => _mm256_mullo_epi16(v, _mm256_set1_epi16(tap as i16)),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_i16(sum: __m256i, v: __m256i, tap: i32) -> __m256i {
    _mm256_add_epi16(sum, add_tap_i16(v, tap))
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x8_2rows(dst: &mut [u8], p0: usize, p1: usize, v: __m256i) {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_packus_epi16(lo, hi);
    unsafe {
        _mm_storel_epi64(dst.as_mut_ptr().add(p0).cast(), p);
        _mm_storel_epi64(dst.as_mut_ptr().add(p1).cast(), _mm_srli_si128::<8>(p));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8xw_2rows<const W: usize>(dst: &mut [u8], p0: usize, p1: usize, v: __m256i) {
    debug_assert!(W == 4 || W == 8);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_packus_epi16(lo, hi);
    unsafe {
        if W == 8 {
            _mm_storel_epi64(dst.as_mut_ptr().add(p0).cast(), p);
            _mm_storel_epi64(dst.as_mut_ptr().add(p1).cast(), _mm_srli_si128::<8>(p));
        } else {
            _mm_store_ss(dst.as_mut_ptr().add(p0).cast(), _mm_castsi128_ps(p));
            _mm_store_ss(
                dst.as_mut_ptr().add(p1).cast(),
                _mm_castsi128_ps(_mm_srli_si128::<8>(p)),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn cdef_filter_block_8bpc_avx2_shape<
    const W: usize,
    const H: usize,
    const HAS_PRI: bool,
    const HAS_SEC: bool,
>(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    debug_assert!(W == 4 || W == 8);
    debug_assert!(H == 4 || H == 8);
    let clip = HAS_PRI && HAS_SEC;
    let pri_s = _mm256_set1_epi16(pri_strength as i16);
    let sec_s = _mm256_set1_epi16(sec_strength as i16);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm256_setzero_si256();
    let eight = _mm256_set1_epi16(8);
    let lowmask = _mm256_set1_epi16(0xff);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let mut y = 0usize;

    while y < H {
        let t0 = (o + y * tmp_stride) as isize;
        let t1 = t0 + tmp_stride as isize;
        let load = |off: isize| load_i16xw_2rows::<W>(tmp, t0, t1, off);
        let px = load(0);
        let mut sum = zero;
        let mut min_v = px;
        let mut max_v = px;

        if HAS_PRI {
            let mut ptap = pri_tap;
            for k in 0..2 {
                let off = dirs[dir + 2][k] as isize;
                let p0 = load(off);
                let p1 = load(-off);
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(p0, px), pri_s, pri_shc),
                    ptap,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(p1, px), pri_s, pri_shc),
                    ptap,
                );
                ptap = (ptap & 3) | 2;
                if clip {
                    min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                    max_v = _mm256_max_epi16(max_v, _mm256_max_epi16(p0, p1));
                }
                if HAS_SEC {
                    let off2 = dirs[dir + 4][k] as isize;
                    let off3 = dirs[dir][k] as isize;
                    let s0 = load(off2);
                    let s1 = load(-off2);
                    let s2 = load(off3);
                    let s3 = load(-off3);
                    let st = 2 - k as i32;
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                        st,
                    );
                    min_v = cdef_min_i16(
                        min_v,
                        cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                    );
                    max_v = _mm256_max_epi16(
                        max_v,
                        _mm256_max_epi16(_mm256_max_epi16(s0, s1), _mm256_max_epi16(s2, s3)),
                    );
                }
            }
        } else if HAS_SEC {
            for k in 0..2 {
                let off1 = dirs[dir + 4][k] as isize;
                let off2 = dirs[dir][k] as isize;
                let s0 = load(off1);
                let s1 = load(-off1);
                let s2 = load(off2);
                let s3 = load(-off2);
                let st = 2 - k as i32;
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                    st,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                    st,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                    st,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                    st,
                );
            }
        }

        let mask = _mm256_cmpgt_epi16(zero, sum);
        let delta = _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(sum, mask), eight));
        let mut res = _mm256_add_epi16(px, delta);
        if clip {
            res = _mm256_min_epi16(_mm256_max_epi16(res, min_v), max_v);
        }
        res = _mm256_and_si256(res, lowmask);
        let d0 = dst_off + y * dst_stride;
        let d1 = d0 + dst_stride;
        store_u8xw_2rows::<W>(dst, d0, d1, res);
        y += 2;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn cdef_filter_block_8bpc_avx2_shape_dispatch<const W: usize, const H: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    match (pri_strength != 0, sec_strength != 0) {
        (true, true) => cdef_filter_block_8bpc_avx2_shape::<W, H, true, true>(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
        ),
        (true, false) => cdef_filter_block_8bpc_avx2_shape::<W, H, true, false>(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
        ),
        (false, true) => cdef_filter_block_8bpc_avx2_shape::<W, H, false, true>(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
        ),
        (false, false) => (),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_8x8_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_avx2_shape_dispatch::<8, 8>(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_8x4_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_avx2_shape_dispatch::<8, 4>(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_4x8_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_avx2_shape_dispatch::<4, 8>(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_4x4_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_avx2_shape_dispatch::<4, 4>(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    match (w, h) {
        (8, 8) => {
            cdef_filter_block_8x8_8bpc_avx2(
                dst,
                dst_stride,
                dst_off,
                tmp,
                tmp_stride,
                o,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
            );
            return;
        }
        (8, 4) => {
            cdef_filter_block_8bpc_avx2_shape_dispatch::<8, 4>(
                dst,
                dst_stride,
                dst_off,
                tmp,
                tmp_stride,
                o,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
            );
            return;
        }
        (4, 8) => {
            cdef_filter_block_4x8_8bpc_avx2(
                dst,
                dst_stride,
                dst_off,
                tmp,
                tmp_stride,
                o,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
            );
            return;
        }
        (4, 4) => {
            cdef_filter_block_4x4_8bpc_avx2(
                dst,
                dst_stride,
                dst_off,
                tmp,
                tmp_stride,
                o,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
            );
            return;
        }
        _ => {}
    }

    if w < 8 {
        crate::cdef_dispatch::cdef_filter_block_8bpc_scalar(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        );
        return;
    }

    let has_pri = pri_strength != 0;
    let has_sec = sec_strength != 0;
    let clip = has_pri && has_sec;
    let pri_s = _mm256_set1_epi16(pri_strength as i16);
    let sec_s = _mm256_set1_epi16(sec_strength as i16);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm256_setzero_si256();
    let eight = _mm256_set1_epi16(8);
    let lowmask = _mm256_set1_epi16(0xff);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 8;
    let mut y = 0usize;

    while y < h {
        let paired = y + 1 < h;
        for g in 0..groups {
            let bx = g * 8;
            let t0 = (o + y * tmp_stride + bx) as isize;
            let t1 = if paired { t0 + tmp_stride as isize } else { t0 };
            let load = |off: isize| load_i16x16_2rows(tmp, t0, t1, off);
            let px = load(0);
            let mut sum = zero;
            let mut min_v = px;
            let mut max_v = px;

            if has_pri {
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off = dirs[dir + 2][k] as isize;
                    let p0 = load(off);
                    let p1 = load(-off);
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(p0, px), pri_s, pri_shc),
                        ptap,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(p1, px), pri_s, pri_shc),
                        ptap,
                    );
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                        max_v = _mm256_max_epi16(max_v, _mm256_max_epi16(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = 2 - k as i32;
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                            st,
                        );
                        min_v = cdef_min_i16(
                            min_v,
                            cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                        );
                        max_v = _mm256_max_epi16(
                            max_v,
                            _mm256_max_epi16(_mm256_max_epi16(s0, s1), _mm256_max_epi16(s2, s3)),
                        );
                    }
                }
            } else {
                for k in 0..2 {
                    let off1 = dirs[dir + 4][k] as isize;
                    let off2 = dirs[dir][k] as isize;
                    let s0 = load(off1);
                    let s1 = load(-off1);
                    let s2 = load(off2);
                    let s3 = load(-off2);
                    let st = 2 - k as i32;
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                        st,
                    );
                }
            }

            let mask = _mm256_cmpgt_epi16(zero, sum);
            let delta =
                _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(sum, mask), eight));
            let mut res = _mm256_add_epi16(px, delta);
            if clip {
                res = _mm256_min_epi16(_mm256_max_epi16(res, min_v), max_v);
            }
            res = _mm256_and_si256(res, lowmask);
            let d0 = dst_off + y * dst_stride + bx;
            let d1 = if paired { d0 + dst_stride } else { d0 };
            store_u8x8_2rows(dst, d0, d1, res);
        }
        y += if paired { 2 } else { 1 };
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_dir_8bpc_pair(img: &[u8], stride: usize, y: usize) -> __m256i {
    let lo = unsafe { _mm_loadl_epi64(img.as_ptr().add(y * stride).cast()) };
    let hi = unsafe { _mm_loadl_epi64(img.as_ptr().add((y + 4) * stride).cast()) };
    let bytes = _mm_unpacklo_epi64(lo, hi);
    _mm256_sub_epi16(_mm256_cvtepu8_epi16(bytes), _mm256_set1_epi16(128))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_find_dir_8bpc_avx2(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    let z = _mm_setzero_si128();
    let mut rows = [z; 8];
    let r04 = load_dir_8bpc_pair(img, stride, 0);
    let r15 = load_dir_8bpc_pair(img, stride, 1);
    let r26 = load_dir_8bpc_pair(img, stride, 2);
    let r37 = load_dir_8bpc_pair(img, stride, 3);
    rows[0] = _mm256_castsi256_si128(r04);
    rows[4] = _mm256_extracti128_si256::<1>(r04);
    rows[1] = _mm256_castsi256_si128(r15);
    rows[5] = _mm256_extracti128_si256::<1>(r15);
    rows[2] = _mm256_castsi256_si128(r26);
    rows[6] = _mm256_extracti128_si256::<1>(r26);
    rows[3] = _mm256_castsi256_si128(r37);
    rows[7] = _mm256_extracti128_si256::<1>(r37);
    cdef_find_dir_from_rows_avx2(&rows, var)
}

#[cfg(test)]
mod tests {
    use crate::cdef_dispatch::cdef_filter_block_8bpc_scalar;

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    #[test]
    fn cdef_filter_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        const TMP_STRIDE: usize = 12;
        const O: usize = 2 * TMP_STRIDE + 2;
        const DST_STRIDE: usize = 16;
        let mut rng = R(0xd1b54a32d192ed03);
        for _ in 0..40_000 {
            let mut tmp = [0i16; 144];
            for t in tmp.iter_mut() {
                *t = rng.range(0, 255) as i16;
            }
            for _ in 0..18 {
                let i = rng.range(0, tmp.len() as i32 - 1) as usize;
                tmp[i] = i16::MIN;
            }
            for y in 0..8 {
                for x in 0..8 {
                    tmp[O + y * TMP_STRIDE + x] = rng.range(0, 255) as i16;
                }
            }

            // choose variant: ensure at least one strength nonzero
            let variant = rng.range(0, 2); // 0 pri+sec, 1 pri-only, 2 sec-only
            let pri_strength = if variant == 2 { 0 } else { rng.range(1, 63) };
            let sec_strength = if variant == 1 { 0 } else { rng.range(1, 63) };
            let pri_shift = rng.range(0, 7);
            let sec_shift = rng.range(0, 7);
            let pri_tap = if pri_strength != 0 {
                4 - (pri_strength & 1)
            } else {
                0
            };
            let dir = rng.range(0, 7) as usize;
            let w = if rng.range(0, 1) == 0 { 4 } else { 8 };
            let h = if rng.range(0, 1) == 0 { 4 } else { 8 };

            let mut a = [0u8; DST_STRIDE * 8];
            let mut b = [0u8; DST_STRIDE * 8];
            cdef_filter_block_8bpc_scalar(
                &mut a,
                DST_STRIDE,
                0,
                &tmp,
                TMP_STRIDE,
                O,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
                w,
                h,
            );
            unsafe {
                super::cdef_filter_block_8bpc_avx2(
                    &mut b,
                    DST_STRIDE,
                    0,
                    &tmp,
                    TMP_STRIDE,
                    O,
                    pri_strength,
                    sec_strength,
                    pri_shift,
                    sec_shift,
                    pri_tap,
                    dir,
                    w,
                    h,
                );
            }
            assert_eq!(
                a, b,
                "mismatch variant={variant} dir={dir} w={w} h={h} pri={pri_strength} sec={sec_strength}"
            );
        }
    }
}
