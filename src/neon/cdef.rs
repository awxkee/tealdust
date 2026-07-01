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

use crate::cdef::{CDEF_HAVE_BOTTOM, CDEF_HAVE_LEFT, CDEF_HAVE_RIGHT, CDEF_HAVE_TOP};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn cdef_fill_i16_neon(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    let sentinel = vdupq_n_s16(i16::MIN);
    for row in tmp.chunks_exact_mut(stride).take(h) {
        if w >= 8 {
            unsafe { vst1q_s16(row.as_mut_ptr(), sentinel) };
            for v in &mut row[8..w] {
                *v = i16::MIN;
            }
        } else {
            row[..w].fill(i16::MIN);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn copy_u8_to_i16_neon(dst: &mut [i16], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = src.len();
    if n >= 8 {
        unsafe {
            let v = vld1_u8(src.as_ptr());
            vst1q_s16(dst.as_mut_ptr(), vreinterpretq_s16_u16(vmovl_u8(v)));
            if n > 8 {
                let s = src.as_ptr().add(n - 8);
                let d = dst.as_mut_ptr().add(n - 8);
                let v = vld1_u8(s);
                vst1q_s16(d, vreinterpretq_s16_u16(vmovl_u8(v)));
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
#[target_feature(enable = "neon")]
fn cdef_padding_8bpc_neon_full<const W: usize, const H: usize>(
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
    copy_u8_to_i16_neon(
        &mut tmp[top_dst..top_dst + W + 4],
        &top[top_src..top_src + W + 4],
    );
    copy_u8_to_i16_neon(
        &mut tmp[top_dst + tmp_stride..top_dst + tmp_stride + W + 4],
        &top[top_src + src_stride..top_src + src_stride + W + 4],
    );

    let mut soff = src_off;
    for y in 0..H {
        let ti = o + y * tmp_stride;
        tmp[ti - 2] = left[y][0] as i16;
        tmp[ti - 1] = left[y][1] as i16;
        copy_u8_to_i16_neon(&mut tmp[ti..ti + W + 2], &src[soff..soff + W + 2]);
        soff += src_stride;
    }

    let bottom_src = bottom_off - 2;
    let bottom_dst = o - 2 + H * tmp_stride;
    copy_u8_to_i16_neon(
        &mut tmp[bottom_dst..bottom_dst + W + 4],
        &bottom[bottom_src..bottom_src + W + 4],
    );
    copy_u8_to_i16_neon(
        &mut tmp[bottom_dst + tmp_stride..bottom_dst + tmp_stride + W + 4],
        &bottom[bottom_src + bottom_stride..bottom_src + bottom_stride + W + 4],
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn cdef_padding_8bpc_neon(
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
                cdef_padding_8bpc_neon_full::<8, 8>(
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
                cdef_padding_8bpc_neon_full::<8, 4>(
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
                cdef_padding_8bpc_neon_full::<4, 8>(
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
                cdef_padding_8bpc_neon_full::<4, 4>(
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
        cdef_fill_i16_neon(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        cdef_fill_i16_neon(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        cdef_fill_i16_neon(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        cdef_fill_i16_neon(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut toff = top_off;
    for y in y_start..0 {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (toff as i32 + x_start) as usize;
        copy_u8_to_i16_neon(&mut tmp[ti..ti + copy_w], &top[si..si + copy_w]);
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
        copy_u8_to_i16_neon(&mut tmp[ti..ti + copy_w], &src[soff..soff + copy_w]);
        soff += src_stride;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut boff = bottom_off;
    for y in h as i32..y_end {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (boff as i32 + x_start) as usize;
        copy_u8_to_i16_neon(&mut tmp[ti..ti + copy_w], &bottom[si..si + copy_w]);
        boff += bottom_stride;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn square_weighted4(vals: [i32; 4], mirror: [i32; 4], weights: [i32; 4]) -> u32 {
    let a = unsafe { vld1q_s32(vals.as_ptr()) };
    let b = unsafe { vld1q_s32(mirror.as_ptr()) };
    let w = unsafe { vld1q_s32(weights.as_ptr()) };
    let sq = vaddq_s32(vmulq_s32(a, a), vmulq_s32(b, b));
    vaddvq_s32(vmulq_s32(sq, w)) as u32
}

#[inline]
#[target_feature(enable = "neon")]
fn square_weighted_sym15_neon(p: &[i32; 15]) -> u32 {
    square_weighted4(
        [p[0], p[1], p[2], p[3]],
        [p[14], p[13], p[12], p[11]],
        [840, 420, 280, 210],
    ) + square_weighted4(
        [p[4], p[5], p[6], p[7]],
        [p[10], p[9], p[8], 0],
        [168, 140, 120, 105],
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn square_weighted_alt11_neon(p: &[i32; 11]) -> u32 {
    square_weighted4(
        [p[0], p[1], p[2], p[3]],
        [p[10], p[9], p[8], 0],
        [420, 210, 140, 105],
    ) + square_weighted4([p[4], p[5], p[6], p[7]], [0, 0, 0, 0], [105, 105, 105, 105])
}

#[inline]
#[target_feature(enable = "neon")]
fn finish_cdef_dir_neon(
    partial_sum_hv: &[[i32; 8]; 2],
    partial_sum_diag: &[[i32; 15]; 2],
    partial_sum_alt: &[[i32; 11]; 4],
    var: &mut u32,
) -> i32 {
    let hv0a = unsafe { vld1q_s32(partial_sum_hv[0].as_ptr()) };
    let hv0b = unsafe { vld1q_s32(partial_sum_hv[0].as_ptr().add(4)) };
    let hv1a = unsafe { vld1q_s32(partial_sum_hv[1].as_ptr()) };
    let hv1b = unsafe { vld1q_s32(partial_sum_hv[1].as_ptr().add(4)) };
    let mut cost = [0u32; 8];
    cost[2] =
        (vaddvq_s32(vmulq_s32(hv0a, hv0a)) as u32 + vaddvq_s32(vmulq_s32(hv0b, hv0b)) as u32) * 105;
    cost[6] =
        (vaddvq_s32(vmulq_s32(hv1a, hv1a)) as u32 + vaddvq_s32(vmulq_s32(hv1b, hv1b)) as u32) * 105;
    cost[0] = square_weighted_sym15_neon(&partial_sum_diag[0]);
    cost[4] = square_weighted_sym15_neon(&partial_sum_diag[1]);
    cost[1] = square_weighted_alt11_neon(&partial_sum_alt[0]);
    cost[3] = square_weighted_alt11_neon(&partial_sum_alt[1]);
    cost[5] = square_weighted_alt11_neon(&partial_sum_alt[2]);
    cost[7] = square_weighted_alt11_neon(&partial_sum_alt[3]);

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
#[target_feature(enable = "neon")]
fn hsum_i16x8(v: int16x8_t) -> i32 {
    let pair = vpaddlq_s16(v);
    vaddvq_s32(pair)
}

#[inline]
#[target_feature(enable = "neon")]
fn reverse_i16x8(v: int16x8_t) -> int16x8_t {
    let rev64 = vrev64q_s16(v);
    vextq_s16::<4>(rev64, rev64)
}

#[inline]
#[target_feature(enable = "neon")]
fn reverse_i16x4_low(v: int16x8_t) -> int16x8_t {
    vcombine_s16(vrev64_s16(vget_low_s16(v)), vdup_n_s16(0))
}

#[inline]
#[target_feature(enable = "neon")]
fn pair_sum_i16x8(v: int16x8_t) -> int16x8_t {
    vpaddq_s16(v, vdupq_n_s16(0))
}

#[inline]
#[target_feature(enable = "neon")]
fn shl_words_i16x8(v: int16x8_t, n: usize) -> int16x8_t {
    let z = vdupq_n_s16(0);
    match n {
        0 => v,
        1 => vextq_s16::<7>(z, v),
        2 => vextq_s16::<6>(z, v),
        3 => vextq_s16::<5>(z, v),
        4 => vextq_s16::<4>(z, v),
        5 => vextq_s16::<3>(z, v),
        6 => vextq_s16::<2>(z, v),
        7 => vextq_s16::<1>(z, v),
        _ => z,
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn overflow_words_i16x8(v: int16x8_t, n: usize) -> int16x8_t {
    let z = vdupq_n_s16(0);
    match n {
        0 => z,
        1 => vextq_s16::<7>(v, z),
        2 => vextq_s16::<6>(v, z),
        3 => vextq_s16::<5>(v, z),
        4 => vextq_s16::<4>(v, z),
        5 => vextq_s16::<3>(v, z),
        6 => vextq_s16::<2>(v, z),
        7 => vextq_s16::<1>(v, z),
        _ => v,
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn add_shifted_i16x8(lo: &mut int16x8_t, hi: &mut int16x8_t, v: int16x8_t, n: usize) {
    *lo = vaddq_s16(*lo, shl_words_i16x8(v, n));
    *hi = vaddq_s16(*hi, overflow_words_i16x8(v, n));
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4_prefix(dst: &mut [i32], v: int32x4_t, n: usize) {
    match n {
        0 => {}
        1 => dst[0] = vgetq_lane_s32::<0>(v),
        2 => unsafe { vst1_s32(dst.as_mut_ptr(), vget_low_s32(v)) },
        3 => {
            unsafe { vst1_s32(dst.as_mut_ptr(), vget_low_s32(v)) };
            dst[2] = vgetq_lane_s32::<2>(v);
        }
        _ => unsafe { vst1q_s32(dst.as_mut_ptr(), v) },
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8_to_i32(dst: &mut [i32], v: int16x8_t, n: usize) {
    let lo = vmovl_s16(vget_low_s16(v));
    let hi = vmovl_s16(vget_high_s16(v));
    let lo_n = n.min(4);

    store_i32x4_prefix(dst, lo, lo_n);
    if n > 4 {
        store_i32x4_prefix(&mut dst[4..], hi, n - 4);
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(super) fn cdef_find_dir_from_rows_neon(rows: &[int16x8_t; 8], var: &mut u32) -> i32 {
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    let zero = vdupq_n_s16(0);
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
        col_sum = vaddq_s16(col_sum, row);

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

    finish_cdef_dir_neon(&partial_sum_hv, &partial_sum_diag, &partial_sum_alt, var)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x4(tmp: &[i16], p: isize, off: isize) -> int16x4_t {
    unsafe { vld1_s16(tmp.as_ptr().offset(p + off)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8(tmp: &[i16], p: isize, off: isize) -> int16x8_t {
    unsafe { vld1q_s16(tmp.as_ptr().offset(p + off)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn cdef_min_i16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    vreinterpret_s16_u16(vmin_u16(vreinterpret_u16_s16(a), vreinterpret_u16_s16(b)))
}

#[inline]
#[target_feature(enable = "neon")]
fn cdef_min_i16q(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vreinterpretq_s16_u16(vminq_u16(
        vreinterpretq_u16_s16(a),
        vreinterpretq_u16_s16(b),
    ))
}

#[inline]
#[target_feature(enable = "neon")]
fn add_tap_i16(v: int16x4_t, tap: i32) -> int16x4_t {
    match tap {
        1 => v,
        2 => vadd_s16(v, v),
        3 => vadd_s16(vadd_s16(v, v), v),
        4 => {
            let t = vadd_s16(v, v);
            vadd_s16(t, t)
        }
        _ => vmul_n_s16(v, tap as i16),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_tap_i16(sum: int16x4_t, v: int16x4_t, tap: i32) -> int16x4_t {
    vadd_s16(sum, add_tap_i16(v, tap))
}

#[inline]
#[target_feature(enable = "neon")]
fn add_tap_i16q(v: int16x8_t, tap: i32) -> int16x8_t {
    match tap {
        1 => v,
        2 => vaddq_s16(v, v),
        3 => vaddq_s16(vaddq_s16(v, v), v),
        4 => {
            let t = vaddq_s16(v, v);
            vaddq_s16(t, t)
        }
        _ => vmulq_n_s16(v, tap as i16),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_tap_i16q(sum: int16x8_t, v: int16x8_t, tap: i32) -> int16x8_t {
    vaddq_s16(sum, add_tap_i16q(v, tap))
}

#[inline]
#[target_feature(enable = "neon")]
fn constrain_i16(diff: int16x4_t, threshold: int16x4_t, nsh: int16x4_t) -> int16x4_t {
    let zero = vdup_n_s16(0);
    let adiff = vabs_s16(diff);
    let t = vmax_s16(zero, vsub_s16(threshold, vshl_s16(adiff, nsh)));
    let m = vreinterpret_s16_u16(vmin_u16(
        vreinterpret_u16_s16(adiff),
        vreinterpret_u16_s16(t),
    ));
    vbsl_s16(vclt_s16(diff, zero), vsub_s16(zero, m), m)
}

#[inline]
#[target_feature(enable = "neon")]
fn cnst_i16q(diff: int16x8_t, threshold: int16x8_t, nsh: int16x8_t) -> int16x8_t {
    let zero = vdupq_n_s16(0);
    let adiff = vabsq_s16(diff);
    let t = vmaxq_s16(zero, vsubq_s16(threshold, vshlq_s16(adiff, nsh)));
    let m = vreinterpretq_s16_u16(vminq_u16(
        vreinterpretq_u16_s16(adiff),
        vreinterpretq_u16_s16(t),
    ));
    vbslq_s16(vcltq_s16(diff, zero), vsubq_s16(zero, m), m)
}

#[inline]
#[target_feature(enable = "neon")]
fn mask_u8(v: int16x4_t) -> int16x4_t {
    vreinterpret_s16_u16(vand_u16(vreinterpret_u16_s16(v), vdup_n_u16(0xff)))
}

#[inline]
#[target_feature(enable = "neon")]
fn mask_u8q(v: int16x8_t) -> int16x8_t {
    vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_s16(v), vdupq_n_u16(0xff)))
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x4_u8(dst: &mut [u8], p: usize, v: int16x4_t) {
    let packed = vqmovun_s16(vcombine_s16(v, vdup_n_s16(0)));
    unsafe {
        vst1_lane_u32::<0>(dst.as_mut_ptr().add(p).cast(), vreinterpret_u32_u8(packed));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8_u8(dst: &mut [u8], p: usize, v: int16x8_t) {
    let packed = vqmovun_s16(v);
    unsafe { vst1_u8(dst.as_mut_ptr().add(p), packed) };
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn cdef_filter_block_4w_8bpc_neon_shape<
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
    debug_assert!(H == 4 || H == 8);
    let clip = HAS_PRI && HAS_SEC;
    let pri_s = vdup_n_s16(pri_strength as i16);
    let sec_s = vdup_n_s16(sec_strength as i16);
    let pri_nsh = vdup_n_s16(-(pri_shift as i16));
    let sec_nsh = vdup_n_s16(-(sec_shift as i16));
    let zero = vdup_n_s16(0);
    let eight = vdup_n_s16(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let mut dp = dst_off;
    let mut tp = o;

    for _ in 0..H {
        let tpx = tp as isize;
        let load = |off: isize| load_i16x4(tmp, tpx, off);
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
                sum = madd_tap_i16(sum, constrain_i16(vsub_s16(p0, px), pri_s, pri_nsh), ptap);
                sum = madd_tap_i16(sum, constrain_i16(vsub_s16(p1, px), pri_s, pri_nsh), ptap);
                ptap = (ptap & 3) | 2;
                if clip {
                    min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                    max_v = vmax_s16(max_v, vmax_s16(p0, p1));
                }
                if HAS_SEC {
                    let off2 = dirs[dir + 4][k] as isize;
                    let off3 = dirs[dir][k] as isize;
                    let s0 = load(off2);
                    let s1 = load(-off2);
                    let s2 = load(off3);
                    let s3 = load(-off3);
                    let st = 2 - k as i32;
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s0, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s1, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s2, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s3, px), sec_s, sec_nsh), st);
                    min_v = cdef_min_i16(
                        min_v,
                        cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                    );
                    max_v = vmax_s16(max_v, vmax_s16(vmax_s16(s0, s1), vmax_s16(s2, s3)));
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
                sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s0, px), sec_s, sec_nsh), st);
                sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s1, px), sec_s, sec_nsh), st);
                sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s2, px), sec_s, sec_nsh), st);
                sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s3, px), sec_s, sec_nsh), st);
            }
        }

        let mask = vreinterpret_s16_u16(vclt_s16(sum, zero));
        let delta = vshr_n_s16::<4>(vadd_s16(vadd_s16(sum, mask), eight));
        let mut res = vadd_s16(px, delta);
        if clip {
            res = vmin_s16(vmax_s16(res, min_v), max_v);
        }
        store_i16x4_u8(dst, dp, mask_u8(res));
        dp += dst_stride;
        tp += tmp_stride;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn cdef_filter_block_8w_8bpc_neon_shape<
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
    debug_assert!(H == 4 || H == 8);
    let clip = HAS_PRI && HAS_SEC;
    let pri_s = vdupq_n_s16(pri_strength as i16);
    let sec_s = vdupq_n_s16(sec_strength as i16);
    let pri_nsh = vdupq_n_s16(-(pri_shift as i16));
    let sec_nsh = vdupq_n_s16(-(sec_shift as i16));
    let zero = vdupq_n_s16(0);
    let eight = vdupq_n_s16(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let mut dp = dst_off;
    let mut tp = o;

    for _ in 0..H {
        let tpx = tp as isize;
        let load = |off: isize| load_i16x8(tmp, tpx, off);
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
                sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(p0, px), pri_s, pri_nsh), ptap);
                sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(p1, px), pri_s, pri_nsh), ptap);
                ptap = (ptap & 3) | 2;
                if clip {
                    min_v = cdef_min_i16q(min_v, cdef_min_i16q(p0, p1));
                    max_v = vmaxq_s16(max_v, vmaxq_s16(p0, p1));
                }
                if HAS_SEC {
                    let off2 = dirs[dir + 4][k] as isize;
                    let off3 = dirs[dir][k] as isize;
                    let s0 = load(off2);
                    let s1 = load(-off2);
                    let s2 = load(off3);
                    let s3 = load(-off3);
                    let st = 2 - k as i32;
                    sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s0, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s1, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s2, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s3, px), sec_s, sec_nsh), st);
                    min_v = cdef_min_i16q(
                        min_v,
                        cdef_min_i16q(cdef_min_i16q(s0, s1), cdef_min_i16q(s2, s3)),
                    );
                    max_v = vmaxq_s16(max_v, vmaxq_s16(vmaxq_s16(s0, s1), vmaxq_s16(s2, s3)));
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
                sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s0, px), sec_s, sec_nsh), st);
                sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s1, px), sec_s, sec_nsh), st);
                sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s2, px), sec_s, sec_nsh), st);
                sum = madd_tap_i16q(sum, cnst_i16q(vsubq_s16(s3, px), sec_s, sec_nsh), st);
            }
        }

        let mask = vreinterpretq_s16_u16(vcltq_s16(sum, zero));
        let delta = vshrq_n_s16::<4>(vaddq_s16(vaddq_s16(sum, mask), eight));
        let mut res = vaddq_s16(px, delta);
        if clip {
            res = vminq_s16(vmaxq_s16(res, min_v), max_v);
        }
        store_i16x8_u8(dst, dp, mask_u8q(res));
        dp += dst_stride;
        tp += tmp_stride;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn cdef_filter_block_8w_8bpc_neon_shape_dispatch<const H: usize>(
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
        (true, true) => cdef_filter_block_8w_8bpc_neon_shape::<H, true, true>(
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
        (true, false) => cdef_filter_block_8w_8bpc_neon_shape::<H, true, false>(
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
        (false, true) => cdef_filter_block_8w_8bpc_neon_shape::<H, false, true>(
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
#[target_feature(enable = "neon")]
fn cdef_filter_block_4w_8bpc_neon_shape_dispatch<const H: usize>(
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
        (true, true) => cdef_filter_block_4w_8bpc_neon_shape::<H, true, true>(
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
        (true, false) => cdef_filter_block_4w_8bpc_neon_shape::<H, true, false>(
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
        (false, true) => cdef_filter_block_4w_8bpc_neon_shape::<H, false, true>(
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
#[target_feature(enable = "neon")]
pub(crate) fn cdef_filter_block_8x8_8bpc_neon(
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
    cdef_filter_block_8w_8bpc_neon_shape_dispatch::<8>(
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
#[target_feature(enable = "neon")]
pub(crate) fn cdef_filter_block_8x4_8bpc_neon(
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
    cdef_filter_block_8w_8bpc_neon_shape_dispatch::<4>(
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
#[target_feature(enable = "neon")]
pub(crate) fn cdef_filter_block_4x8_8bpc_neon(
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
    cdef_filter_block_4w_8bpc_neon_shape_dispatch::<8>(
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
#[target_feature(enable = "neon")]
pub(crate) fn cdef_filter_block_4x4_8bpc_neon(
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
    cdef_filter_block_4w_8bpc_neon_shape_dispatch::<4>(
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
#[target_feature(enable = "neon")]
pub(crate) fn cdef_filter_block_8bpc_neon(
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
            cdef_filter_block_8x8_8bpc_neon(
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
            cdef_filter_block_8w_8bpc_neon_shape_dispatch::<4>(
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
            cdef_filter_block_4x8_8bpc_neon(
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
            cdef_filter_block_4x4_8bpc_neon(
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

    let has_pri = pri_strength != 0;
    let has_sec = sec_strength != 0;
    let clip = has_pri && has_sec;
    let pri_s = vdup_n_s16(pri_strength as i16);
    let sec_s = vdup_n_s16(sec_strength as i16);
    let pri_nsh = vdup_n_s16(-(pri_shift as i16));
    let sec_nsh = vdup_n_s16(-(sec_shift as i16));
    let zero = vdup_n_s16(0);
    let eight = vdup_n_s16(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 4;
    let mut dp = dst_off;
    let mut tp = o;

    for _ in 0..h {
        for g in 0..groups {
            let bx = g * 4;
            let tpx = (tp + bx) as isize;
            let load = |off: isize| load_i16x4(tmp, tpx, off);
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
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(p0, px), pri_s, pri_nsh), ptap);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(p1, px), pri_s, pri_nsh), ptap);
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                        max_v = vmax_s16(max_v, vmax_s16(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = 2 - k as i32;
                        sum =
                            madd_tap_i16(sum, constrain_i16(vsub_s16(s0, px), sec_s, sec_nsh), st);
                        sum =
                            madd_tap_i16(sum, constrain_i16(vsub_s16(s1, px), sec_s, sec_nsh), st);
                        sum =
                            madd_tap_i16(sum, constrain_i16(vsub_s16(s2, px), sec_s, sec_nsh), st);
                        sum =
                            madd_tap_i16(sum, constrain_i16(vsub_s16(s3, px), sec_s, sec_nsh), st);
                        min_v = cdef_min_i16(
                            min_v,
                            cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                        );
                        max_v = vmax_s16(max_v, vmax_s16(vmax_s16(s0, s1), vmax_s16(s2, s3)));
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
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s0, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s1, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s2, px), sec_s, sec_nsh), st);
                    sum = madd_tap_i16(sum, constrain_i16(vsub_s16(s3, px), sec_s, sec_nsh), st);
                }
            }

            let mask = vreinterpret_s16_u16(vclt_s16(sum, zero));
            let delta = vshr_n_s16::<4>(vadd_s16(vadd_s16(sum, mask), eight));
            let mut res = vadd_s16(px, delta);
            if clip {
                res = vmin_s16(vmax_s16(res, min_v), max_v);
            }
            store_i16x4_u8(dst, dp + bx, mask_u8(res));
        }
        dp += dst_stride;
        tp += tmp_stride;
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn cdef_find_dir_8bpc_neon(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    let mut rows = [vdupq_n_s16(0); 8];
    let bias = vdupq_n_s16(128);
    for y in 0..8usize {
        let src = unsafe { img.as_ptr().add(y * stride) };
        let pix = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(src) }));
        rows[y] = vsubq_s16(pix, bias);
    }
    cdef_find_dir_from_rows_neon(&rows, var)
}
