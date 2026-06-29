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

use core::arch::aarch64::*;

use crate::cfl_dispatch::{CflAlphaAccum8, CflApply8, CflMhccpPred8};
const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

#[inline(always)]
fn predict_one(dc: i32, alpha: i32, ac: i32) -> u8 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, 255) as u8
}

#[inline(always)]
fn load_u8x8(a: &[u8; 8]) -> uint8x8_t {
    unsafe { vld1_u8(a.as_ptr()) }
}

#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> uint8x16_t {
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline(always)]
fn load_u8x16_tail8(src: &[u8]) -> uint8x16_t {
    debug_assert!(src.len() >= 8);
    unsafe { vcombine_u8(vld1_u8(src.as_ptr().cast()), vdup_n_u8(0)) }
}

#[inline(always)]
fn load_u8x8_tail4(src: &[u8]) -> uint8x8_t {
    debug_assert!(src.len() >= 4);
    unsafe { vreinterpret_u8_u32(vld1_lane_u32::<0>(src.as_ptr().cast(), vdup_n_u32(0))) }
}

#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: uint8x8_t) {
    unsafe { vst1_u8(a.as_mut_ptr(), v) };
}

#[inline(always)]
fn store_u8x4(a: &mut [u8], v: uint8x8_t) {
    debug_assert!(a.len() >= 4);
    let mut tmp = [0u8; 8];
    unsafe { vst1_u8(tmp.as_mut_ptr(), v) };
    a[..4].copy_from_slice(&tmp[..4]);
}

#[inline(always)]
fn store_u8x16(a: &mut [u8; 16], v: uint8x16_t) {
    unsafe { vst1q_u8(a.as_mut_ptr(), v) };
}

#[inline(always)]
fn pad_bottom(plane: &mut [u8], row0: usize, stride: usize, w: usize, h: usize, ylim: usize) {
    debug_assert_ne!(ylim, 0);
    let src = row0 + (ylim - 1) * stride;
    for yy in ylim..h {
        let dst = row0 + yy * stride;
        plane.copy_within(src..src + w, dst);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn even_u8x8(src: uint8x16_t) -> uint8x8_t {
    vget_low_u8(vuzp1q_u8(src, src))
}

#[inline]
#[target_feature(enable = "neon")]
fn odd_u8x8(src: uint8x16_t) -> uint8x8_t {
    vget_low_u8(vuzp2q_u8(src, src))
}

#[inline]
#[target_feature(enable = "neon")]
fn left_u8x8(src: uint8x16_t, prev: u8) -> uint8x8_t {
    let shifted = vextq_u8::<15>(vdupq_n_u8(prev), src);
    even_u8x8(shifted)
}

#[inline]
#[target_feature(enable = "neon")]
fn ac8_420_i16(top: uint8x16_t, bot: uint8x16_t, dc0v: int16x8_t) -> int16x8_t {
    let top_pairs = vpaddlq_u8(top);
    let bot_pairs = vpaddlq_u8(bot);

    let sum2x2 = vaddq_u16(top_pairs, bot_pairs); // <= 1020
    let sum2x2_x2 = vshlq_n_u16::<1>(sum2x2); // <= 2040

    vsubq_s16(vreinterpretq_s16_u16(sum2x2_x2), dc0v)
}

#[inline]
#[target_feature(enable = "neon")]
fn ac8_420_filter_i16<const FILTER: u32>(
    cur: uint8x16_t,
    top: uint8x16_t,
    bot: uint8x16_t,
    prev_cur: u8,
    prev_bot: u8,
    dc0v: int16x8_t,
) -> int16x8_t {
    if FILTER == CFL_FLT_TYPE_VSTRIP {
        let left_cur = vmovl_u8(left_u8x8(cur, prev_cur));
        let center_cur = vmovl_u8(even_u8x8(cur));
        let right_cur = vmovl_u8(odd_u8x8(cur));
        let left_bot = vmovl_u8(left_u8x8(bot, prev_bot));
        let center_bot = vmovl_u8(even_u8x8(bot));
        let right_bot = vmovl_u8(odd_u8x8(bot));

        let cur_sum = vaddq_u16(vaddq_u16(left_cur, vshlq_n_u16::<1>(center_cur)), right_cur);
        let bot_sum = vaddq_u16(vaddq_u16(left_bot, vshlq_n_u16::<1>(center_bot)), right_bot);
        vsubq_s16(vreinterpretq_s16_u16(vaddq_u16(cur_sum, bot_sum)), dc0v)
    } else if FILTER == CFL_FLT_TYPE_GAUSS {
        let left = vmovl_u8(left_u8x8(cur, prev_cur));
        let center = vmovl_u8(even_u8x8(cur));
        let right = vmovl_u8(odd_u8x8(cur));
        let top = vmovl_u8(even_u8x8(top));
        let bot = vmovl_u8(even_u8x8(bot));

        let center4 = vshlq_n_u16::<2>(center);
        let sum = vaddq_u16(
            vaddq_u16(vaddq_u16(left, center4), right),
            vaddq_u16(top, bot),
        );
        vsubq_s16(vreinterpretq_s16_u16(sum), dc0v)
    } else {
        ac8_420_i16(cur, bot, dc0v)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ac8_444_i16(src: uint8x8_t, dc0v: int16x8_t) -> int16x8_t {
    vsubq_s16(vreinterpretq_s16_u16(vshll_n_u8::<3>(src)), dc0v)
}

#[inline]
#[target_feature(enable = "neon")]
fn ac8_422_uniform_i16(src: uint8x16_t, dc0v: int16x8_t) -> int16x8_t {
    let sum = vpaddlq_u8(src);
    vsubq_s16(vreinterpretq_s16_u16(vshlq_n_u16::<2>(sum)), dc0v)
}

#[inline]
#[target_feature(enable = "neon")]
fn ac8_422_gauss_i16(src: uint8x16_t, dc0v: int16x8_t) -> int16x8_t {
    ac8_444_i16(even_u8x8(src), dc0v)
}

/// Apply alpha to 8 i16 AC lanes.
///
/// Only this function widens to i32, because `alpha * ac` may need i32.
/// Everything before this stays i16.
#[inline]
#[target_feature(enable = "neon")]
fn apply8_i16_ac_wide(
    ac: int16x8_t,
    _alpha: i16,
    alpha_v: int16x4_t,
    _dc: i32,
    dc_v: int32x4_t,
    round_v: int32x4_t,
    zero_v: int32x4_t,
) -> uint8x8_t {
    let ac_lo = vget_low_s16(ac);
    let ac_hi = vget_high_s16(ac);

    // i16 * i16 -> i32. This is the only widening part.
    let diff_lo = vmull_s16(ac_lo, alpha_v);
    let diff_hi = vmull_s16(ac_hi, alpha_v);

    let mag_lo = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff_lo), round_v));
    let mag_hi = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff_hi), round_v));

    let signed_lo = vbslq_s32(vcltq_s32(diff_lo, zero_v), vnegq_s32(mag_lo), mag_lo);
    let signed_hi = vbslq_s32(vcltq_s32(diff_hi, zero_v), vnegq_s32(mag_hi), mag_hi);

    let val_lo = vaddq_s32(dc_v, signed_lo);
    let val_hi = vaddq_s32(dc_v, signed_hi);

    vqmovn_u16(vcombine_u16(vqmovun_s32(val_lo), vqmovun_s32(val_hi)))
}

macro_rules! cfl8_apply_wide {
    ($ac:expr, $alpha:expr, $alpha_v:expr, $dc:expr, $dc_v:expr, $round_v:expr, $zero_v:expr) => {
        apply8_i16_ac_wide($ac, $alpha, $alpha_v, $dc, $dc_v, $round_v, $zero_v)
    };
}

macro_rules! cfl8_apply_rdm {
    ($ac:expr, $alpha:expr, $_alpha_v:expr, $dc:expr, $_dc_v:expr, $_round_v:expr, $_zero_v:expr) => {{
        let zero = vdupq_n_s16(0);
        let alpha8 = vdupq_n_s16($alpha);
        let ac_abs = vabsq_s16($ac);
        let alpha_abs = vabsq_s16(alpha8);
        let mag = vqrdmulhq_s16(vshlq_n_s16::<4>(ac_abs), alpha_abs);
        let neg_mask = veorq_u16(vcltq_s16($ac, zero), vcltq_s16(alpha8, zero));
        let signed = vbslq_s16(neg_mask, vnegq_s16(mag), mag);
        vqmovun_s16(vaddq_s16(vdupq_n_s16($dc as i16), signed))
    }};
}

macro_rules! define_cfl8_neon_impl {
    ($mod_name:ident, $apply8_i16_ac_macro:ident, $(#[$target_attr:meta])*) => {
        mod $mod_name {
            use super::*;

            #[inline]
            $(#[$target_attr])*
            fn apply8_i16_ac_fn(
                ac: int16x8_t,
                alpha: i16,
                _alpha_v: int16x4_t,
                dc: i32,
                _dc_v: int32x4_t,
                _round_v: int32x4_t,
                _zero_v: int32x4_t,
            ) -> uint8x8_t {
                $apply8_i16_ac_macro!(ac, alpha, _alpha_v, dc, _dc_v, _round_v, _zero_v)
            }

            #[inline]
            $(#[$target_attr])*
            fn apply16_444_i16_ac_fn(
                src: uint8x16_t,
                dc0v: int16x8_t,
                alpha: i16,
                alpha_v: int16x4_t,
                dc: i32,
                dc_v: int32x4_t,
                round_v: int32x4_t,
                zero_v: int32x4_t,
            ) -> uint8x16_t {
                let lo = apply8_i16_ac_fn(
                    ac8_444_i16(vget_low_u8(src), dc0v),
                    alpha,
                    alpha_v,
                    dc,
                    dc_v,
                    round_v,
                    zero_v,
                );
                let hi = apply8_i16_ac_fn(
                    ac8_444_i16(vget_high_u8(src), dc0v),
                    alpha,
                    alpha_v,
                    dc,
                    dc_v,
                    round_v,
                    zero_v,
                );
                vcombine_u8(lo, hi)
            }

$(#[$target_attr])*
fn cfl_apply_420_8bpc_impl<const FILTER: u32>(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;

    if !do_u && !do_v {
        return;
    }

    assert_ne!(xlim, 0);
    assert_ne!(ylim, 0);

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha1));

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    let dc0v = vdupq_n_s16(dc0 as i16);

    let alpha0v = vdup_n_s16(alpha0 as i16);
    let alpha1v = vdup_n_s16(alpha1 as i16);

    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);

    let round_v = vdupq_n_s32(1024);
    let zero_v = vdupq_n_s32(0);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;

    for cy in 0..ylim {
        let cur = y[yrow..yrow + lfull].as_chunks::<16>().0;
        let top_row = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
            yrow - ystride
        } else {
            yrow
        };
        let top = y[top_row..top_row + lfull].as_chunks::<16>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<16>()
            .0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (i, (((du, dv), c), (t, b))) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let luma_x = i * 16;
                    let prev_cur = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let prev_bot = if (luma_x & 63) == 0 {
                        y[yrow + ystride + luma_x]
                    } else {
                        y[yrow + ystride + luma_x - 1]
                    };
                    let ac = ac8_420_filter_i16::<FILTER>(
                        load_u8x16(c),
                        load_u8x16(t),
                        load_u8x16(b),
                        prev_cur,
                        prev_bot,
                        dc0v,
                    );

                    store_u8x8(
                        du,
                        apply8_i16_ac_fn(
                            ac,
                            alpha0 as i16,
                            alpha0v,
                            dc1,
                            dc1v,
                            round_v,
                            zero_v,
                        ),
                    );
                    store_u8x8(
                        dv,
                        apply8_i16_ac_fn(
                            ac,
                            alpha1 as i16,
                            alpha1v,
                            dc2,
                            dc2v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;

                for (i, ((du, c), (t, b))) in u_chunks
                    .iter_mut()
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let luma_x = i * 16;
                    let prev_cur = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let prev_bot = if (luma_x & 63) == 0 {
                        y[yrow + ystride + luma_x]
                    } else {
                        y[yrow + ystride + luma_x - 1]
                    };
                    let ac = ac8_420_filter_i16::<FILTER>(
                        load_u8x16(c),
                        load_u8x16(t),
                        load_u8x16(b),
                        prev_cur,
                        prev_bot,
                        dc0v,
                    );

                    store_u8x8(
                        du,
                        apply8_i16_ac_fn(
                            ac,
                            alpha0 as i16,
                            alpha0v,
                            dc1,
                            dc1v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (i, ((dv, c), (t, b))) in v_chunks
                    .iter_mut()
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let luma_x = i * 16;
                    let prev_cur = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let prev_bot = if (luma_x & 63) == 0 {
                        y[yrow + ystride + luma_x]
                    } else {
                        y[yrow + ystride + luma_x - 1]
                    };
                    let ac = ac8_420_filter_i16::<FILTER>(
                        load_u8x16(c),
                        load_u8x16(t),
                        load_u8x16(b),
                        prev_cur,
                        prev_bot,
                        dc0v,
                    );

                    store_u8x8(
                        dv,
                        apply8_i16_ac_fn(
                            ac,
                            alpha1 as i16,
                            alpha1v,
                            dc2,
                            dc2v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (false, false) => unreachable!(),
        }

        let x4full = xfull + ((xlim - xfull) / 4) * 4;
        let mut x = xfull;
        while x < x4full {
            let luma_x = x << 1;
            let top_row = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
                yrow - ystride
            } else {
                yrow
            };
            let prev_cur = if (luma_x & 63) == 0 {
                y[yrow + luma_x]
            } else {
                y[yrow + luma_x - 1]
            };
            let prev_bot = if (luma_x & 63) == 0 {
                y[yrow + ystride + luma_x]
            } else {
                y[yrow + ystride + luma_x - 1]
            };
            let ac = ac8_420_filter_i16::<FILTER>(
                load_u8x16_tail8(&y[yrow + luma_x..]),
                load_u8x16_tail8(&y[top_row + luma_x..]),
                load_u8x16_tail8(&y[yrow + ystride + luma_x..]),
                prev_cur,
                prev_bot,
                dc0v,
            );
            if do_u {
                store_u8x4(
                    &mut u[urow + x..urow + x + 4],
                    apply8_i16_ac_fn(
                        ac,
                        alpha0 as i16,
                        alpha0v,
                        dc1,
                        dc1v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            if do_v {
                store_u8x4(
                    &mut v[vrow + x..vrow + x + 4],
                    apply8_i16_ac_fn(
                        ac,
                        alpha1 as i16,
                        alpha1v,
                        dc2,
                        dc2v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            x += 4;
        }

        for x in x4full..xlim {
            let ac = cfl_ac_420_scalar_filter::<FILTER>(y, yrow, ystride, cy, x, dc0);
            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }

        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }

        yrow += ystride << 1;
        urow += cstride;
        vrow += cstride;
    }

    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }

    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ac8_422_i16<const FILTER: u32>(src: uint8x16_t, prev: u8, dc0v: int16x8_t) -> int16x8_t {
    if FILTER == CFL_FLT_TYPE_GAUSS {
        ac8_422_gauss_i16(src, dc0v)
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        let left = vmovl_u8(left_u8x8(src, prev));
        let center = vmovl_u8(even_u8x8(src));
        let right = vmovl_u8(odd_u8x8(src));
        let sum = vshlq_n_u16::<1>(vaddq_u16(vaddq_u16(left, vshlq_n_u16::<1>(center)), right));
        vsubq_s16(vreinterpretq_s16_u16(sum), dc0v)
    } else {
        ac8_422_uniform_i16(src, dc0v)
    }
}

#[inline(always)]
fn cfl_ac_420_scalar_filter<const FILTER: u32>(
    y: &[u8],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: i32,
) -> i32 {
    let xl = x << 1;
    let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
    if FILTER == CFL_FLT_TYPE_GAUSS {
        let top = if (cy & 31) == 0 {
            yrow + xl
        } else {
            yrow + xl - ystride
        };
        y[yrow + left] as i32
            + 4 * y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[top] as i32
            + y[yrow + xl + ystride] as i32
            - dc0
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        y[yrow + left] as i32
            + 2 * y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[yrow + left + ystride] as i32
            + 2 * y[yrow + xl + ystride] as i32
            + y[yrow + xl + ystride + 1] as i32
            - dc0
    } else {
        ((y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[yrow + xl + ystride] as i32
            + y[yrow + xl + ystride + 1] as i32)
            << 1)
            - dc0
    }
}

#[inline(always)]
fn cfl_ac_422_scalar_filter<const FILTER: u32>(y: &[u8], yrow: usize, x: usize, dc0: i32) -> i32 {
    let xl = x << 1;
    if FILTER == CFL_FLT_TYPE_GAUSS {
        ((y[yrow + xl] as i32) << 3) - dc0
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
        (y[yrow + left] as i32 + 2 * y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) * 2 - dc0
    } else {
        ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
    }
}

$(#[$target_attr])*
fn cfl_apply_422_8bpc_impl<const FILTER: u32>(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;

    if !do_u && !do_v {
        return;
    }

    assert_ne!(xlim, 0);
    assert_ne!(ylim, 0);

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha1));

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    let dc0v = vdupq_n_s16(dc0 as i16);

    let alpha0v = vdup_n_s16(alpha0 as i16);
    let alpha1v = vdup_n_s16(alpha1 as i16);

    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);

    let round_v = vdupq_n_s32(1024);
    let zero_v = vdupq_n_s32(0);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;

    for _y in 0..ylim {
        let row = y[yrow..yrow + lfull].as_chunks::<16>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (i, ((du, dv), yy)) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(row.iter())
                    .enumerate()
                {
                    let luma_x = i * 16;
                    let prev = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let ac = ac8_422_i16::<FILTER>(load_u8x16(yy), prev, dc0v);

                    store_u8x8(
                        du,
                        apply8_i16_ac_fn(
                            ac,
                            alpha0 as i16,
                            alpha0v,
                            dc1,
                            dc1v,
                            round_v,
                            zero_v,
                        ),
                    );
                    store_u8x8(
                        dv,
                        apply8_i16_ac_fn(
                            ac,
                            alpha1 as i16,
                            alpha1v,
                            dc2,
                            dc2v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;

                for (i, (du, yy)) in u_chunks.iter_mut().zip(row.iter()).enumerate() {
                    let luma_x = i * 16;
                    let prev = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let ac = ac8_422_i16::<FILTER>(load_u8x16(yy), prev, dc0v);

                    store_u8x8(
                        du,
                        apply8_i16_ac_fn(
                            ac,
                            alpha0 as i16,
                            alpha0v,
                            dc1,
                            dc1v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (i, (dv, yy)) in v_chunks.iter_mut().zip(row.iter()).enumerate() {
                    let luma_x = i * 16;
                    let prev = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let ac = ac8_422_i16::<FILTER>(load_u8x16(yy), prev, dc0v);

                    store_u8x8(
                        dv,
                        apply8_i16_ac_fn(
                            ac,
                            alpha1 as i16,
                            alpha1v,
                            dc2,
                            dc2v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (false, false) => unreachable!(),
        }

        let x4full = xfull + ((xlim - xfull) / 4) * 4;
        let mut x = xfull;
        while x < x4full {
            let luma_x = x << 1;
            let prev = if (luma_x & 63) == 0 {
                y[yrow + luma_x]
            } else {
                y[yrow + luma_x - 1]
            };
            let ac = ac8_422_i16::<FILTER>(load_u8x16_tail8(&y[yrow + luma_x..]), prev, dc0v);
            if do_u {
                store_u8x4(
                    &mut u[urow + x..urow + x + 4],
                    apply8_i16_ac_fn(
                        ac,
                        alpha0 as i16,
                        alpha0v,
                        dc1,
                        dc1v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            if do_v {
                store_u8x4(
                    &mut v[vrow + x..vrow + x + 4],
                    apply8_i16_ac_fn(
                        ac,
                        alpha1 as i16,
                        alpha1v,
                        dc2,
                        dc2v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            x += 4;
        }

        for x in x4full..xlim {
            let ac = cfl_ac_422_scalar_filter::<FILTER>(y, yrow, x, dc0);

            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }

        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }

        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }

        yrow += ystride;
        urow += cstride;
        vrow += cstride;
    }

    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }

    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

$(#[$target_attr])*
fn cfl_apply_444_8bpc_impl(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;

    if !do_u && !do_v {
        return;
    }

    assert_ne!(xlim, 0);
    assert_ne!(ylim, 0);

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha1));

    let nfull = xlim / 16;
    let xfull = nfull * 16;

    let dc0v = vdupq_n_s16(dc0 as i16);

    let alpha0v = vdup_n_s16(alpha0 as i16);
    let alpha1v = vdup_n_s16(alpha1 as i16);

    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);

    let round_v = vdupq_n_s32(1024);
    let zero_v = vdupq_n_s32(0);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;

    for _y in 0..ylim {
        let row = y[yrow..yrow + xfull].as_chunks::<16>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for ((du, dv), yy) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(row.iter()) {
                    let yy = load_u8x16(yy);
                    store_u8x16(
                        du,
                        apply16_444_i16_ac_fn(
                            yy,
                            dc0v,
                            alpha0 as i16,
                            alpha0v,
                            dc1,
                            dc1v,
                            round_v,
                            zero_v,
                        ),
                    );
                    store_u8x16(
                        dv,
                        apply16_444_i16_ac_fn(
                            yy,
                            dc0v,
                            alpha1 as i16,
                            alpha1v,
                            dc2,
                            dc2v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;

                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    store_u8x16(
                        du,
                        apply16_444_i16_ac_fn(
                            load_u8x16(yy),
                            dc0v,
                            alpha0 as i16,
                            alpha0v,
                            dc1,
                            dc1v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    store_u8x16(
                        dv,
                        apply16_444_i16_ac_fn(
                            load_u8x16(yy),
                            dc0v,
                            alpha1 as i16,
                            alpha1v,
                            dc2,
                            dc2v,
                            round_v,
                            zero_v,
                        ),
                    );
                }
            }

            (false, false) => unreachable!(),
        }

        let x8full = xfull + ((xlim - xfull) / 8) * 8;
        let mut x = xfull;
        while x < x8full {
            let yy_chunks = y[yrow + x..yrow + x + 8].as_chunks::<8>().0;
            let ac = ac8_444_i16(load_u8x8(&yy_chunks[0]), dc0v);
            if do_u {
                let u_chunks = u[urow + x..urow + x + 8].as_chunks_mut::<8>().0;
                store_u8x8(
                    &mut u_chunks[0],
                    apply8_i16_ac_fn(
                        ac,
                        alpha0 as i16,
                        alpha0v,
                        dc1,
                        dc1v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            if do_v {
                let v_chunks = v[vrow + x..vrow + x + 8].as_chunks_mut::<8>().0;
                store_u8x8(
                    &mut v_chunks[0],
                    apply8_i16_ac_fn(
                        ac,
                        alpha1 as i16,
                        alpha1v,
                        dc2,
                        dc2v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            x += 8;
        }

        let x4full = x8full + ((xlim - x8full) / 4) * 4;
        while x < x4full {
            let ac = ac8_444_i16(load_u8x8_tail4(&y[yrow + x..]), dc0v);
            if do_u {
                store_u8x4(
                    &mut u[urow + x..urow + x + 4],
                    apply8_i16_ac_fn(
                        ac,
                        alpha0 as i16,
                        alpha0v,
                        dc1,
                        dc1v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            if do_v {
                store_u8x4(
                    &mut v[vrow + x..vrow + x + 4],
                    apply8_i16_ac_fn(
                        ac,
                        alpha1 as i16,
                        alpha1v,
                        dc2,
                        dc2v,
                        round_v,
                        zero_v,
                    ),
                );
            }
            x += 4;
        }

        for x in x4full..xlim {
            let ac = ((y[yrow + x] as i32) << 3) - dc0;

            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }

        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }

        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }

        yrow += ystride;
        urow += cstride;
        vrow += cstride;
    }

    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }

    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

            pub(super) fn cfl_apply_420_8bpc(args: CflApply8<'_>) {
                match args.params.filter_type {
                    CFL_FLT_TYPE_VSTRIP => unsafe {
                        cfl_apply_420_8bpc_impl::<CFL_FLT_TYPE_VSTRIP>(args)
                    },
                    CFL_FLT_TYPE_GAUSS => unsafe {
                        cfl_apply_420_8bpc_impl::<CFL_FLT_TYPE_GAUSS>(args)
                    },
                    _ => unsafe { cfl_apply_420_8bpc_impl::<0>(args) },
                }
            }

            pub(super) fn cfl_apply_422_8bpc(args: CflApply8<'_>) {
                match args.params.filter_type {
                    CFL_FLT_TYPE_VSTRIP => unsafe {
                        cfl_apply_422_8bpc_impl::<CFL_FLT_TYPE_VSTRIP>(args)
                    },
                    CFL_FLT_TYPE_GAUSS => unsafe {
                        cfl_apply_422_8bpc_impl::<CFL_FLT_TYPE_GAUSS>(args)
                    },
                    _ => unsafe { cfl_apply_422_8bpc_impl::<0>(args) },
                }
            }

            pub(super) fn cfl_apply_444_8bpc(args: CflApply8<'_>) {
                unsafe { cfl_apply_444_8bpc_impl(args) }
            }
        }
    };
}

define_cfl8_neon_impl!(cfl8_neon_base, cfl8_apply_wide, #[target_feature(enable = "neon")]);
define_cfl8_neon_impl!(cfl8_neon_rdm, cfl8_apply_rdm,#[target_feature(enable = "rdm")]);

#[target_feature(enable = "neon")]
pub(crate) fn cfl_apply_420_8bpc_neon(args: CflApply8<'_>) {
    cfl8_neon_base::cfl_apply_420_8bpc(args)
}

#[target_feature(enable = "rdm")]
pub(crate) fn cfl_apply_420_8bpc_neon_rdm(args: CflApply8<'_>) {
    cfl8_neon_rdm::cfl_apply_420_8bpc(args)
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_apply_422_8bpc_neon(args: CflApply8<'_>) {
    cfl8_neon_base::cfl_apply_422_8bpc(args)
}

#[target_feature(enable = "rdm")]
pub(crate) fn cfl_apply_422_8bpc_neon_rdm(args: CflApply8<'_>) {
    cfl8_neon_rdm::cfl_apply_422_8bpc(args)
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_apply_444_8bpc_neon(args: CflApply8<'_>) {
    cfl8_neon_base::cfl_apply_444_8bpc(args)
}

#[target_feature(enable = "rdm")]
pub(crate) fn cfl_apply_444_8bpc_neon_rdm(args: CflApply8<'_>) {
    cfl8_neon_rdm::cfl_apply_444_8bpc(args)
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_mul32_neon(v: int32x4_t, alpha: i32) -> int32x4_t {
    let prod = vmulq_n_s32(v, alpha);
    let mag = vshrq_n_s32::<16>(vaddq_s32(vabsq_s32(prod), vdupq_n_s32(1 << 15)));
    vbslq_s32(vcltq_s32(prod, vdupq_n_s32(0)), vnegq_s32(mag), mag)
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_sqrnd8_neon(v: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<8>(vaddq_s32(vmulq_s32(v, v), vdupq_n_s32(128)))
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_pred4_neon(v0: int32x4_t, v1: int32x4_t, alpha: [i32; 3], a2v2: int32x4_t) -> int32x4_t {
    vaddq_s32(
        vaddq_s32(
            mhccp_mul32_neon(v0, alpha[0]),
            mhccp_mul32_neon(mhccp_sqrnd8_neon(v1), alpha[1]),
        ),
        a2v2,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_load_u8x8_i32_halves(src: &[u8]) -> (int32x4_t, int32x4_t) {
    debug_assert!(src.len() >= 8);
    let v = unsafe { vld1_u8(src.as_ptr()) };
    let v16 = vmovl_u8(v);
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v16))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_load_u8x16_i32_quads(src: &[u8; 16]) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let v = load_u8x16(src);
    let lo16 = vmovl_u8(vget_low_u8(v));
    let hi16 = vmovl_u8(vget_high_u8(v));
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_store_u8x8_neon(dst: &mut [u8; 8], lo: int32x4_t, hi: int32x4_t) {
    let v16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
    unsafe { vst1_u8(dst.as_mut_ptr(), vqmovn_u16(v16)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_store_u8x16_neon(
    dst: &mut [u8; 16],
    a: int32x4_t,
    b: int32x4_t,
    c: int32x4_t,
    d: int32x4_t,
) {
    let ab = vcombine_u16(vqmovun_s32(a), vqmovun_s32(b));
    let cd = vcombine_u16(vqmovun_s32(c), vqmovun_s32(d));
    unsafe {
        vst1q_u8(
            dst.as_mut_ptr(),
            vcombine_u8(vqmovn_u16(ab), vqmovn_u16(cd)),
        )
    };
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_pred16_store_neon(
    dst: &mut [u8; 16],
    v0: &[u8; 16],
    v1: &[u8; 16],
    alpha: [i32; 3],
    a2v2: int32x4_t,
) {
    let (v0_0, v0_1, v0_2, v0_3) = mhccp_load_u8x16_i32_quads(v0);
    let (v1_0, v1_1, v1_2, v1_3) = mhccp_load_u8x16_i32_quads(v1);
    mhccp_store_u8x16_neon(
        dst,
        mhccp_pred4_neon(v0_0, v1_0, alpha, a2v2),
        mhccp_pred4_neon(v0_1, v1_1, alpha, a2v2),
        mhccp_pred4_neon(v0_2, v1_2, alpha, a2v2),
        mhccp_pred4_neon(v0_3, v1_3, alpha, a2v2),
    );
}

#[inline(always)]
fn accum_alpha_u16x8(acc: uint32x4_t, v: uint16x8_t, m: uint16x8_t) -> uint32x4_t {
    unsafe {
        let acc = vmlal_u16(acc, vget_low_u16(v), vget_low_u16(m));
        vmlal_u16(acc, vget_high_u16(v), vget_high_u16(m))
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_alpha_accum_8bpc_neon(args: CflAlphaAccum8<'_>) {
    if args.sample_stride != 1 || args.len < 16 {
        crate::cfl_dispatch::cfl_alpha_accum_8bpc_scalar(args);
        return;
    }

    let CflAlphaAccum8 {
        alpha,
        samples,
        sample_off,
        sample_stride: _,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    } = args;

    let mut acc0 = vdupq_n_u32(0);
    let mut acc1 = vdupq_n_u32(0);
    let mut acc2 = vdupq_n_u32(0);
    let mut processed = 0usize;

    let (sample_chunks, sample_rem) = samples[sample_off..sample_off + len].as_chunks::<16>();
    for (chunk_idx, s) in sample_chunks.iter().enumerate() {
        let i = imat_off + chunk_idx * 16;
        let v = load_u8x16(s);
        let vlo = vmovl_u8(vget_low_u8(v));
        let vhi = vmovl_u8(vget_high_u8(v));
        let m0lo = unsafe { vld1q_u16(imat0[i..].as_ptr()) };
        let m0hi = unsafe { vld1q_u16(imat0[i + 8..].as_ptr()) };
        let m1lo = unsafe { vld1q_u16(imat1[i..].as_ptr()) };
        let m1hi = unsafe { vld1q_u16(imat1[i + 8..].as_ptr()) };
        acc0 = accum_alpha_u16x8(acc0, vlo, m0lo);
        acc0 = accum_alpha_u16x8(acc0, vhi, m0hi);
        acc1 = accum_alpha_u16x8(acc1, vlo, m1lo);
        acc1 = accum_alpha_u16x8(acc1, vhi, m1hi);
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_low_u16(vlo)));
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_high_u16(vlo)));
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_low_u16(vhi)));
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_high_u16(vhi)));
    }
    processed += sample_chunks.len() * 16;

    alpha[0] += vaddvq_u32(acc0) as i32;
    alpha[1] += vaddvq_u32(acc1) as i32;
    alpha[2] += (vaddvq_u32(acc2) as i32) << a2sh;

    if !sample_rem.is_empty() {
        crate::cfl_dispatch::cfl_alpha_accum_8bpc_scalar(CflAlphaAccum8 {
            alpha,
            samples,
            sample_off: sample_off + processed,
            sample_stride: 1,
            imat0,
            imat1,
            imat_off: imat_off + processed,
            len: sample_rem.len(),
            a2sh,
        });
    }
}

#[inline(always)]
fn mhccp_pred_one_8_neon(alpha: &[i32; 3], a2v2: i32, v0: i32, v1: i32) -> u8 {
    let sq = (v1 * v1 + 128) >> 8;
    (crate::ipred::mul32(alpha[0], v0, 16) + crate::ipred::mul32(alpha[1], sq, 16) + a2v2)
        .clamp(0, 255) as u8
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_mhccp_pred_8bpc_neon(args: CflMhccpPred8<'_>) {
    if !crate::cfl_dispatch::cfl_mhccp_coeffs_fit_fast_mul(&args.alpha) || args.w < 8 {
        crate::cfl_dispatch::cfl_mhccp_pred_8bpc_scalar(args);
        return;
    }

    let CflMhccpPred8 {
        dst,
        dst_stride,
        src,
        src_off,
        src_top_stride,
        w,
        h,
        alpha,
        edge_flags,
        dir,
    } = args;
    let has_t = edge_flags & (1 << 2) != 0;
    let has_l = edge_flags & (1 << 3) != 0;
    let dir_t = dir == crate::levels::CflMhDir::Top;
    let dir_l = dir == crate::levels::CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let a2v2_scalar = crate::ipred::mul32(alpha[2], 128, 16);
    let a2v2 = vdupq_n_s32(a2v2_scalar);

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && y < h {
        let dst_row = &mut dst[..w];
        let (dst16, r16) = dst_row.as_chunks_mut::<16>();
        let prev = sp - src_top_stride;
        for (i, chunk) in dst16.iter_mut().enumerate() {
            let x = i * 16;
            mhccp_pred16_store_neon(
                chunk,
                (&src[prev + x..prev + x + 16]).try_into().unwrap(),
                (&src[sp + x..sp + x + 16]).try_into().unwrap(),
                alpha,
                a2v2,
            );
        }
        let done16 = dst16.len() * 16;
        let (dst8, dst_tail) = r16.as_chunks_mut::<8>();
        for (i, chunk) in dst8.iter_mut().enumerate() {
            let x = done16 + i * 8;
            let (v0_lo, v0_hi) = mhccp_load_u8x8_i32_halves(&src[prev + x..]);
            let (v1_lo, v1_hi) = mhccp_load_u8x8_i32_halves(&src[sp + x..]);
            mhccp_store_u8x8_neon(
                chunk,
                mhccp_pred4_neon(v0_lo, v1_lo, alpha, a2v2),
                mhccp_pred4_neon(v0_hi, v1_hi, alpha, a2v2),
            );
        }
        let done = done16 + dst8.len() * 8;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            *d = mhccp_pred_one_8_neon(
                &alpha,
                a2v2_scalar,
                src[prev + x] as i32,
                src[sp + x] as i32,
            );
        }
        sp += w;
        y = 1;
    }

    for (row_y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate().skip(y) {
        let dst_row = &mut dst_row[..w];
        let mut x0 = 0usize;
        if dir_l {
            let v0 = if has_l {
                src[left_off + row_y * n_left + 1] as i32
            } else {
                src[sp] as i32
            };
            dst_row[0] = mhccp_pred_one_8_neon(&alpha, a2v2_scalar, v0, src[sp] as i32);
            x0 = 1;
        }
        let (dst16, r16) = dst_row[x0..].as_chunks_mut::<16>();
        for (i, chunk) in dst16.iter_mut().enumerate() {
            let x = x0 + i * 16;
            let v0_off = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x - 1
            } else {
                sp + x
            };
            mhccp_pred16_store_neon(
                chunk,
                (&src[v0_off..v0_off + 16]).try_into().unwrap(),
                (&src[sp + x..sp + x + 16]).try_into().unwrap(),
                alpha,
                a2v2,
            );
        }
        let done16 = x0 + dst16.len() * 16;
        let (dst8, dst_tail) = r16.as_chunks_mut::<8>();
        for (i, chunk) in dst8.iter_mut().enumerate() {
            let x = done16 + i * 8;
            let v0_off = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x - 1
            } else {
                sp + x
            };
            let (v0_lo, v0_hi) = mhccp_load_u8x8_i32_halves(&src[v0_off..]);
            let (v1_lo, v1_hi) = mhccp_load_u8x8_i32_halves(&src[sp + x..]);
            mhccp_store_u8x8_neon(
                chunk,
                mhccp_pred4_neon(v0_lo, v1_lo, alpha, a2v2),
                mhccp_pred4_neon(v0_hi, v1_hi, alpha, a2v2),
            );
        }
        let done = done16 + dst8.len() * 8;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            let v0_idx = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x.saturating_sub(1)
            } else {
                sp + x
            };
            *d = mhccp_pred_one_8_neon(&alpha, a2v2_scalar, src[v0_idx] as i32, src[sp + x] as i32);
        }
        sp += w;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn pair_madd_u8x16(v: uint8x16_t, weights: uint8x8_t) -> uint16x8_t {
    let lo = vmull_u8(vget_low_u8(v), weights);
    let hi = vmull_u8(vget_high_u8(v), weights);
    vcombine_u16(vmovn_u32(vpaddlq_u16(lo)), vmovn_u32(vpaddlq_u16(hi)))
}

#[inline]
#[target_feature(enable = "neon")]
fn gen_y8_u8x8<const FILTER: i32>(
    src: &[u8],
    src_off: usize,
    top: &[u8],
    top_off: usize,
    bottom_offset: usize,
    x: usize,
) -> uint8x8_t {
    let xl = x << 1;
    if FILTER == 1 {
        let left_w = vcreate_u8(0x0001_0001_0001_0001);
        let center_right_w = vcreate_u8(0x0102_0102_0102_0102);
        let cur_left = load_u8x16(
            src[src_off + xl - 1..src_off + xl - 1 + 16]
                .try_into()
                .unwrap(),
        );
        let cur_center = load_u8x16(src[src_off + xl..src_off + xl + 16].try_into().unwrap());
        let bot_left = load_u8x16(
            src[src_off + bottom_offset + xl - 1..src_off + bottom_offset + xl - 1 + 16]
                .try_into()
                .unwrap(),
        );
        let bot_center = load_u8x16(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 16]
                .try_into()
                .unwrap(),
        );
        let cur = vaddq_u16(
            pair_madd_u8x16(cur_left, left_w),
            pair_madd_u8x16(cur_center, center_right_w),
        );
        let bot = vaddq_u16(
            pair_madd_u8x16(bot_left, left_w),
            pair_madd_u8x16(bot_center, center_right_w),
        );
        vshrn_n_u16::<3>(vaddq_u16(cur, bot))
    } else if FILTER == 2 {
        let left_w = vcreate_u8(0x0001_0001_0001_0001);
        let center_right_w = vcreate_u8(0x0104_0104_0104_0104);
        let center_w = left_w;
        let cur_left = load_u8x16(
            src[src_off + xl - 1..src_off + xl - 1 + 16]
                .try_into()
                .unwrap(),
        );
        let cur_center = load_u8x16(src[src_off + xl..src_off + xl + 16].try_into().unwrap());
        let top_c = load_u8x16(top[top_off + xl..top_off + xl + 16].try_into().unwrap());
        let bot_c = load_u8x16(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 16]
                .try_into()
                .unwrap(),
        );
        let cur = vaddq_u16(
            pair_madd_u8x16(cur_left, left_w),
            pair_madd_u8x16(cur_center, center_right_w),
        );
        let tb = vaddq_u16(
            pair_madd_u8x16(top_c, center_w),
            pair_madd_u8x16(bot_c, center_w),
        );
        vshrn_n_u16::<3>(vaddq_u16(cur, tb))
    } else {
        let cur = load_u8x16(src[src_off + xl..src_off + xl + 16].try_into().unwrap());
        let bot = load_u8x16(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 16]
                .try_into()
                .unwrap(),
        );
        let sum = vaddq_u16(vpaddlq_u8(cur), vpaddlq_u8(bot));
        vshrn_n_u16::<2>(sum)
    }
}

#[target_feature(enable = "neon")]
fn cfl_gen_y_row_8bpc_neon_impl<const FILTER: i32>(args: crate::cfl_dispatch::CflGenYRow8<'_>) {
    let crate::cfl_dispatch::CflGenYRow8 {
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type: _,
    } = args;

    let mut processed = 0usize;
    if FILTER != 0 && n_left == 0 && !dst.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_8bpc_scalar(crate::cfl_dispatch::CflGenYRow8 {
            dst: &mut dst[..1],
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left,
            filter_type: FILTER,
        });
        processed = 1;
    }

    let (chunks, rem) = dst[processed..].as_chunks_mut::<8>();
    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let x = n_left + processed + chunk_idx * 8;
        store_u8x8(
            chunk,
            gen_y8_u8x8::<FILTER>(src, src_off, top, top_off, bottom_offset, x),
        );
    }
    processed += chunks.len() * 8;

    if !rem.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_8bpc_scalar(crate::cfl_dispatch::CflGenYRow8 {
            dst: rem,
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left: n_left + processed,
            filter_type: FILTER,
        });
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_gen_y_row_8bpc_neon(args: crate::cfl_dispatch::CflGenYRow8<'_>) {
    match args.filter_type {
        1 => cfl_gen_y_row_8bpc_neon_impl::<1>(args),
        2 => cfl_gen_y_row_8bpc_neon_impl::<2>(args),
        _ => cfl_gen_y_row_8bpc_neon_impl::<0>(args),
    }
}
