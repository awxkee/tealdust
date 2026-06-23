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

use crate::cfl_dispatch::CflApply8;
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
fn load_u8x16(a: &[u8; 16]) -> uint8x16_t {
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: uint8x8_t) {
    unsafe { vst1_u8(a.as_mut_ptr(), v) };
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
fn ac8_420_i16(top: uint8x16_t, bot: uint8x16_t, dc0v: int16x8_t) -> int16x8_t {
    let top_pairs = vpaddlq_u8(top);
    let bot_pairs = vpaddlq_u8(bot);

    let sum2x2 = vaddq_u16(top_pairs, bot_pairs); // <= 1020
    let sum2x2_x2 = vshlq_n_u16::<1>(sum2x2); // <= 2040

    vsubq_s16(vreinterpretq_s16_u16(sum2x2_x2), dc0v)
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
    ac8_444_i16(vget_low_u8(vuzp1q_u8(src, src)), dc0v)
}

/// Apply alpha to 8 i16 AC lanes.
///
/// Only this function widens to i32, because `alpha * ac` may need i32.
/// Everything before this stays i16.
#[inline]
#[target_feature(enable = "neon")]
fn apply8_i16_ac(
    ac: int16x8_t,
    alpha_v: int16x4_t,
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

#[inline]
#[target_feature(enable = "neon")]
fn apply16_444_i16_ac(
    src: uint8x16_t,
    dc0v: int16x8_t,
    alpha_v: int16x4_t,
    dc_v: int32x4_t,
    round_v: int32x4_t,
    zero_v: int32x4_t,
) -> uint8x16_t {
    let lo = apply8_i16_ac(
        ac8_444_i16(vget_low_u8(src), dc0v),
        alpha_v,
        dc_v,
        round_v,
        zero_v,
    );
    let hi = apply8_i16_ac(
        ac8_444_i16(vget_high_u8(src), dc0v),
        alpha_v,
        dc_v,
        round_v,
        zero_v,
    );
    vcombine_u8(lo, hi)
}

#[target_feature(enable = "neon")]
fn cfl_apply_420_8bpc_neon_impl(args: CflApply8<'_>) {
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
        let top = y[yrow..yrow + lfull].as_chunks::<16>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<16>()
            .0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (((du, dv), t), b) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), dc0v);

                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, round_v, zero_v));
                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, round_v, zero_v));
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;

                for ((du, t), b) in u_chunks.iter_mut().zip(top.iter()).zip(bot.iter()) {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), dc0v);

                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, round_v, zero_v));
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for ((dv, t), b) in v_chunks.iter_mut().zip(top.iter()).zip(bot.iter()) {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), dc0v);

                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, round_v, zero_v));
                }
            }

            (false, false) => unreachable!(),
        }

        for x in xfull..xlim {
            let xl = x << 1;

            let ac = ((y[yrow + xl] as i32
                + y[yrow + xl + 1] as i32
                + y[yrow + xl + ystride] as i32
                + y[yrow + xl + ystride + 1] as i32)
                << 1)
                - dc0;

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
fn ac8_422_i16<const GAUSS: bool>(src: uint8x16_t, dc0v: int16x8_t) -> int16x8_t {
    if GAUSS {
        ac8_422_gauss_i16(src, dc0v)
    } else {
        ac8_422_uniform_i16(src, dc0v)
    }
}

#[inline(always)]
fn cfl_ac_422_scalar_filter<const GAUSS: bool>(y: &[u8], yrow: usize, x: usize, dc0: i32) -> i32 {
    let xl = x << 1;
    if GAUSS {
        ((y[yrow + xl] as i32) << 3) - dc0
    } else {
        ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
    }
}

#[target_feature(enable = "neon")]
fn cfl_apply_422_8bpc_neon_impl<const GAUSS: bool>(args: CflApply8<'_>) {
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

                for ((du, dv), yy) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(row.iter()) {
                    let ac = ac8_422_i16::<GAUSS>(load_u8x16(yy), dc0v);

                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, round_v, zero_v));
                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, round_v, zero_v));
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;

                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    let ac = ac8_422_i16::<GAUSS>(load_u8x16(yy), dc0v);

                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, round_v, zero_v));
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    let ac = ac8_422_i16::<GAUSS>(load_u8x16(yy), dc0v);

                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, round_v, zero_v));
                }
            }

            (false, false) => unreachable!(),
        }

        for x in xfull..xlim {
            let ac = cfl_ac_422_scalar_filter::<GAUSS>(y, yrow, x, dc0);

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

#[target_feature(enable = "neon")]
fn cfl_apply_444_8bpc_neon_impl(args: CflApply8<'_>) {
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
                        apply16_444_i16_ac(yy, dc0v, alpha0v, dc1v, round_v, zero_v),
                    );
                    store_u8x16(
                        dv,
                        apply16_444_i16_ac(yy, dc0v, alpha1v, dc2v, round_v, zero_v),
                    );
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;

                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    store_u8x16(
                        du,
                        apply16_444_i16_ac(load_u8x16(yy), dc0v, alpha0v, dc1v, round_v, zero_v),
                    );
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    store_u8x16(
                        dv,
                        apply16_444_i16_ac(load_u8x16(yy), dc0v, alpha1v, dc2v, round_v, zero_v),
                    );
                }
            }

            (false, false) => unreachable!(),
        }

        for x in xfull..xlim {
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

pub(crate) fn cfl_apply_420_8bpc_neon(args: CflApply8<'_>) {
    unsafe { cfl_apply_420_8bpc_neon_impl(args) }
}

pub(crate) fn cfl_apply_422_8bpc_neon(args: CflApply8<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => crate::cfl_dispatch::cfl_apply_422_8bpc_scalar(args),
        CFL_FLT_TYPE_GAUSS => unsafe { cfl_apply_422_8bpc_neon_impl::<true>(args) },
        _ => unsafe { cfl_apply_422_8bpc_neon_impl::<false>(args) },
    }
}

pub(crate) fn cfl_apply_444_8bpc_neon(args: CflApply8<'_>) {
    unsafe { cfl_apply_444_8bpc_neon_impl(args) }
}
