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

use crate::cfl_dispatch::CflApplyHbd;
const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

#[inline(always)]
fn pad_bottom(plane: &mut [u16], row0: usize, stride: usize, w: usize, h: usize, ylim: usize) {
    debug_assert_ne!(ylim, 0);
    let src = row0 + (ylim - 1) * stride;
    for yy in ylim..h {
        let dst = row0 + yy * stride;
        plane.copy_within(src..src + w, dst);
    }
}

#[inline(always)]
fn load_u16x8(a: &[u16; 8]) -> uint16x8_t {
    unsafe { vld1q_u16(a.as_ptr()) }
}

#[inline(always)]
fn load_u16x4(a: &[u16; 4]) -> uint16x4_t {
    unsafe { vld1_u16(a.as_ptr()) }
}

#[inline(always)]
fn store_u16x4(a: &mut [u16; 4], v: uint16x4_t) {
    unsafe { vst1_u16(a.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_420_i32(top: uint16x8_t, bot: uint16x8_t, dc0v: int32x4_t) -> int32x4_t {
    let top = vpaddlq_u16(top);
    let bot = vpaddlq_u16(bot);
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<1>(vaddq_u32(top, bot))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_422_uniform_i32(src: uint16x8_t, dc0v: int32x4_t) -> int32x4_t {
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<2>(vpaddlq_u16(src))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_422_gauss_i32(src: uint16x8_t, dc0v: int32x4_t) -> int32x4_t {
    let even = vget_low_u16(vuzp1q_u16(src, src));
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<3>(vmovl_u16(even))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_444_i32(src: uint16x4_t, dc0v: int32x4_t) -> int32x4_t {
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<3>(vmovl_u16(src))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn apply4_i32_ac(ac: int32x4_t, alpha: i32, dc_v: int32x4_t, max_v: int32x4_t) -> uint16x4_t {
    let diff = vmull_n_s16(vqmovn_s32(ac), alpha as i16);
    let mag = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff), vdupq_n_s32(1024)));
    let signed = vbslq_s32(vcltq_s32(diff, vdupq_n_s32(0)), vnegq_s32(mag), mag);
    let val = vminq_s32(vmaxq_s32(vaddq_s32(dc_v, signed), vdupq_n_s32(0)), max_v);
    vqmovun_s32(val)
}

#[target_feature(enable = "neon")]
fn cfl_apply_420_hbd_neon_impl(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
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

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let lfull = nfull * 8;

    let dc0v = vdupq_n_s32(dc0);
    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let top = y[yrow..yrow + lfull].as_chunks::<8>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (((du, dv), t), b) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(top)
                    .zip(bot)
                {
                    let ac = ac4_420_i32(load_u16x8(t), load_u16x8(b), dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for ((du, t), b) in u_chunks.iter_mut().zip(top).zip(bot) {
                    let ac = ac4_420_i32(load_u16x8(t), load_u16x8(b), dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((dv, t), b) in v_chunks.iter_mut().zip(top).zip(bot) {
                    let ac = ac4_420_i32(load_u16x8(t), load_u16x8(b), dc0v);
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
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
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
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

#[target_feature(enable = "neon")]
fn cfl_apply_422_hbd_neon_impl<const GAUSS: bool>(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
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

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let lfull = nfull * 8;

    let dc0v = vdupq_n_s32(dc0);
    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + lfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let s = load_u16x8(s);
                    let ac = if GAUSS {
                        ac4_422_gauss_i32(s, dc0v)
                    } else {
                        ac4_422_uniform_i32(s, dc0v)
                    };
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let s = load_u16x8(s);
                    let ac = if GAUSS {
                        ac4_422_gauss_i32(s, dc0v)
                    } else {
                        ac4_422_uniform_i32(s, dc0v)
                    };
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let s = load_u16x8(s);
                    let ac = if GAUSS {
                        ac4_422_gauss_i32(s, dc0v)
                    } else {
                        ac4_422_uniform_i32(s, dc0v)
                    };
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (false, false) => unreachable!(),
        }
        for x in xfull..xlim {
            let ac = crate::cfl_dispatch::cfl_ac_422_hbd_scalar(
                y,
                yrow,
                x,
                dc0,
                if GAUSS { CFL_FLT_TYPE_GAUSS } else { 0 },
            );
            if do_u {
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
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
fn cfl_apply_444_hbd_neon_impl(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
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

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let dc0v = vdupq_n_s32(dc0);
    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + xfull].as_chunks::<4>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let ac = ac4_444_i32(load_u16x4(s), dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let ac = ac4_444_i32(load_u16x4(s), dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let ac = ac4_444_i32(load_u16x4(s), dc0v);
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (false, false) => unreachable!(),
        }
        for x in xfull..xlim {
            let ac = ((y[yrow + x] as i32) << 3) - dc0;
            if do_u {
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
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

pub(crate) fn cfl_apply_420_hbd_neon(args: CflApplyHbd<'_>) {
    unsafe { cfl_apply_420_hbd_neon_impl(args) }
}

pub(crate) fn cfl_apply_422_hbd_neon(args: CflApplyHbd<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => crate::cfl_dispatch::cfl_apply_422_hbd_scalar(args),
        CFL_FLT_TYPE_GAUSS => unsafe { cfl_apply_422_hbd_neon_impl::<true>(args) },
        _ => unsafe { cfl_apply_422_hbd_neon_impl::<false>(args) },
    }
}

pub(crate) fn cfl_apply_444_hbd_neon(args: CflApplyHbd<'_>) {
    unsafe { cfl_apply_444_hbd_neon_impl(args) }
}
