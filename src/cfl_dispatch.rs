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
use std::sync::OnceLock;

use crate::levels::CflMhDir;

const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

#[cfg(target_arch = "aarch64")]
const ENABLE_NEON_CFL_RDM_8BPC: bool = true;

#[derive(Clone, Copy)]
pub(crate) struct CflLayout {
    pub(crate) yrow0: usize,
    pub(crate) urow0: usize,
    pub(crate) vrow0: usize,
    pub(crate) ystride: usize,
    pub(crate) cstride: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct CflArea {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) xlim: usize,
    pub(crate) ylim: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct CflParams {
    pub(crate) dc0: i32,
    pub(crate) dc1: i32,
    pub(crate) dc2: i32,
    pub(crate) alpha0: i32,
    pub(crate) alpha1: i32,
    pub(crate) filter_type: u32,
}

pub(crate) struct CflApply8<'a> {
    pub(crate) y: &'a [u8],
    pub(crate) u: &'a mut [u8],
    pub(crate) v: &'a mut [u8],
    pub(crate) layout: CflLayout,
    pub(crate) area: CflArea,
    pub(crate) params: CflParams,
}

pub(crate) struct CflApplyHbd<'a> {
    pub(crate) y: &'a [u16],
    pub(crate) u: &'a mut [u16],
    pub(crate) v: &'a mut [u16],
    pub(crate) layout: CflLayout,
    pub(crate) area: CflArea,
    pub(crate) params: CflParams,
    pub(crate) bitdepth_max: i32,
}

pub(crate) type CflApplyFn = for<'a> unsafe fn(CflApply8<'a>);
pub(crate) type CflApplyHbdFn = for<'a> unsafe fn(CflApplyHbd<'a>);

pub(crate) struct CflMhccpPred8<'a> {
    pub(crate) dst: &'a mut [u8],
    pub(crate) dst_stride: usize,
    pub(crate) src: &'a [u8],
    pub(crate) src_off: usize,
    pub(crate) src_top_stride: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) alpha: [i32; 3],
    pub(crate) edge_flags: i32,
    pub(crate) dir: CflMhDir,
}

pub(crate) struct CflMhccpPredHbd<'a> {
    pub(crate) dst: &'a mut [u16],
    pub(crate) dst_stride: usize,
    pub(crate) src: &'a [u16],
    pub(crate) src_off: usize,
    pub(crate) src_top_stride: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) alpha: [i32; 3],
    pub(crate) edge_flags: i32,
    pub(crate) dir: CflMhDir,
    pub(crate) bitdepth: i32,
    pub(crate) bitdepth_max: i32,
}

pub(crate) struct CflGenYRow8<'a> {
    pub(crate) dst: &'a mut [u8],
    pub(crate) src: &'a [u8],
    pub(crate) src_off: usize,
    pub(crate) top: &'a [u8],
    pub(crate) top_off: usize,
    pub(crate) bottom_offset: usize,
    pub(crate) n_left: usize,
    pub(crate) filter_type: i32,
}

pub(crate) struct CflGenYRowHbd<'a> {
    pub(crate) dst: &'a mut [u16],
    pub(crate) src: &'a [u16],
    pub(crate) src_off: usize,
    pub(crate) top: &'a [u16],
    pub(crate) top_off: usize,
    pub(crate) bottom_offset: usize,
    pub(crate) n_left: usize,
    pub(crate) filter_type: i32,
}

pub(crate) struct CflAlphaAccum8<'a> {
    pub(crate) alpha: &'a mut [i32; 3],
    pub(crate) samples: &'a [u8],
    pub(crate) sample_off: usize,
    pub(crate) sample_stride: usize,
    pub(crate) imat0: &'a [u16; crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES],
    pub(crate) imat1: &'a [u16; crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES],
    pub(crate) imat_off: usize,
    pub(crate) len: usize,
    pub(crate) a2sh: i32,
}

pub(crate) struct CflAlphaAccumHbd<'a> {
    pub(crate) alpha: &'a mut [i32; 3],
    pub(crate) samples: &'a [u16],
    pub(crate) sample_off: usize,
    pub(crate) sample_stride: usize,
    pub(crate) imat0: &'a [u16; crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES],
    pub(crate) imat1: &'a [u16; crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES],
    pub(crate) imat_off: usize,
    pub(crate) len: usize,
    pub(crate) a2sh: i32,
}

pub(crate) type CflGenYRow8Fn = for<'a> unsafe fn(CflGenYRow8<'a>);
pub(crate) type CflGenYRowHbdFn = for<'a> unsafe fn(CflGenYRowHbd<'a>);

pub(crate) type CflAlphaAccum8Fn = for<'a> unsafe fn(CflAlphaAccum8<'a>);
pub(crate) type CflAlphaAccumHbdFn = for<'a> unsafe fn(CflAlphaAccumHbd<'a>);

pub(crate) type CflMhccpPred8Fn = for<'a> unsafe fn(CflMhccpPred8<'a>);
pub(crate) type CflMhccpPredHbdFn = for<'a> unsafe fn(CflMhccpPredHbd<'a>);

#[inline(always)]
pub(crate) fn cfl_mhccp_coeffs_fit_fast_mul(alpha: &[i32; 3]) -> bool {
    // For AV2 MHCCP predictors v0/sqrnd(v1) are bounded by the pixel max
    // (255 for 8bpc, <=4095 for current HBD).  With |alpha| <= 65535 the
    // scalar mul32(a, b, 16) never enters its operand-dropping path, so the SIMD
    // `(a * b + sign-round) >> 16` path is exact.
    alpha.iter().all(|&a| a.unsigned_abs() <= 65_535)
}

#[inline(always)]
fn predict_one(dc: i32, alpha: i32, ac: i32) -> u8 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, 255) as u8
}

#[inline(always)]
pub(crate) fn predict_one_hbd(dc: i32, alpha: i32, ac: i32, bitdepth_max: i32) -> u16 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, bitdepth_max) as u16
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

#[inline(always)]
fn cfl_ac_420_scalar(
    y: &[u8],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: i32,
    filter_type: u32,
) -> i32 {
    let xl = x << 1;
    let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
    if filter_type == CFL_FLT_TYPE_GAUSS {
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
    } else if filter_type == CFL_FLT_TYPE_VSTRIP {
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
fn cfl_ac_420_hbd_scalar(
    y: &[u16],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: i32,
    filter_type: u32,
) -> i32 {
    let xl = x << 1;
    let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
    if filter_type == CFL_FLT_TYPE_GAUSS {
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
    } else if filter_type == CFL_FLT_TYPE_VSTRIP {
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

pub(crate) fn cfl_apply_420_8bpc_scalar(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let CflArea { w, h, xlim, ylim } = area;
    let CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for cy in 0..ylim {
        for x in 0..xlim {
            let ac = cfl_ac_420_scalar(y, yrow, ystride, cy, x, dc0, filter_type);
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

/// Bit-exact scalar apply path for 4:4:4 CFL. The luma plane is already at
/// chroma resolution, so the AC term is just `y << 3` minus the q3 DC.
pub(crate) fn cfl_apply_444_8bpc_scalar(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let CflArea { w, h, xlim, ylim } = area;
    let CflParams {
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

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let ysrc = &y[yrow..yrow + xlim];

        match (do_u, do_v) {
            (true, true) => {
                let udst = &mut u[urow..urow + xlim];
                let vdst = &mut v[vrow..vrow + xlim];
                for ((&yy, du), dv) in ysrc.iter().zip(udst.iter_mut()).zip(vdst.iter_mut()) {
                    let ac = ((yy as i32) << 3) - dc0;
                    *du = predict_one(dc1, alpha0, ac);
                    *dv = predict_one(dc2, alpha1, ac);
                }
            }
            (true, false) => {
                let udst = &mut u[urow..urow + xlim];
                for (&yy, du) in ysrc.iter().zip(udst.iter_mut()) {
                    let ac = ((yy as i32) << 3) - dc0;
                    *du = predict_one(dc1, alpha0, ac);
                }
            }
            (false, true) => {
                let vdst = &mut v[vrow..vrow + xlim];
                for (&yy, dv) in ysrc.iter().zip(vdst.iter_mut()) {
                    let ac = ((yy as i32) << 3) - dc0;
                    *dv = predict_one(dc2, alpha1, ac);
                }
            }
            (false, false) => unreachable!(),
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

#[inline(always)]
fn cfl_ac_422_scalar(y: &[u8], yrow: usize, x: usize, dc0: i32, filter_type: u32) -> i32 {
    let xl = x << 1;
    if filter_type == CFL_FLT_TYPE_GAUSS {
        ((y[yrow + xl] as i32) << 3) - dc0
    } else if filter_type == CFL_FLT_TYPE_VSTRIP {
        let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
        (y[yrow + left] as i32 + 2 * y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) * 2 - dc0
    } else {
        ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
    }
}

/// Bit-exact scalar apply path for 4:2:2 CFL. This covers the horizontal CFL
/// downsampling filters while keeping full vertical chroma resolution.
pub(crate) fn cfl_apply_422_8bpc_scalar(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let CflArea { w, h, xlim, ylim } = area;
    let CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        if filter_type == CFL_FLT_TYPE_GAUSS {
            let ysrc = &y[yrow..yrow + (xlim << 1)];
            match (do_u, do_v) {
                (true, true) => {
                    let udst = &mut u[urow..urow + xlim];
                    let vdst = &mut v[vrow..vrow + xlim];
                    for ((pair, du), dv) in ysrc
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(udst.iter_mut())
                        .zip(vdst.iter_mut())
                    {
                        let ac = ((pair[0] as i32) << 3) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                        *dv = predict_one(dc2, alpha1, ac);
                    }
                }
                (true, false) => {
                    let udst = &mut u[urow..urow + xlim];
                    for (pair, du) in ysrc.as_chunks::<2>().0.iter().zip(udst.iter_mut()) {
                        let ac = ((pair[0] as i32) << 3) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                    }
                }
                (false, true) => {
                    let vdst = &mut v[vrow..vrow + xlim];
                    for (pair, dv) in ysrc.as_chunks::<2>().0.iter().zip(vdst.iter_mut()) {
                        let ac = ((pair[0] as i32) << 3) - dc0;
                        *dv = predict_one(dc2, alpha1, ac);
                    }
                }
                (false, false) => unreachable!(),
            }
        } else if filter_type != CFL_FLT_TYPE_VSTRIP {
            let ysrc = &y[yrow..yrow + (xlim << 1)];
            match (do_u, do_v) {
                (true, true) => {
                    let udst = &mut u[urow..urow + xlim];
                    let vdst = &mut v[vrow..vrow + xlim];
                    for ((pair, du), dv) in ysrc
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(udst.iter_mut())
                        .zip(vdst.iter_mut())
                    {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                        *dv = predict_one(dc2, alpha1, ac);
                    }
                }
                (true, false) => {
                    let udst = &mut u[urow..urow + xlim];
                    for (pair, du) in ysrc.as_chunks::<2>().0.iter().zip(udst.iter_mut()) {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                    }
                }
                (false, true) => {
                    let vdst = &mut v[vrow..vrow + xlim];
                    for (pair, dv) in ysrc.as_chunks::<2>().0.iter().zip(vdst.iter_mut()) {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *dv = predict_one(dc2, alpha1, ac);
                    }
                }
                (false, false) => unreachable!(),
            }
        } else {
            for x in 0..xlim {
                let ac = cfl_ac_422_scalar(y, yrow, x, dc0, filter_type);
                if do_u {
                    u[urow + x] = predict_one(dc1, alpha0, ac);
                }
                if do_v {
                    v[vrow + x] = predict_one(dc2, alpha1, ac);
                }
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

/// Bit-exact high-bit-depth scalar reference for the 4:2:0 CFL fast path.
pub(crate) fn cfl_apply_420_hbd_scalar(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    let CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let CflArea { w, h, xlim, ylim } = area;
    let CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for cy in 0..ylim {
        for x in 0..xlim {
            let ac = cfl_ac_420_hbd_scalar(y, yrow, ystride, cy, x, dc0, filter_type);
            if do_u {
                u[urow + x] = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
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
        pad_bottom_hbd(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom_hbd(v, vrow0, cstride, w, h, ylim);
    }
}

#[inline(always)]
fn pad_bottom_hbd(plane: &mut [u16], row0: usize, stride: usize, w: usize, h: usize, ylim: usize) {
    debug_assert_ne!(ylim, 0);
    let src = row0 + (ylim - 1) * stride;
    for yy in ylim..h {
        let dst = row0 + yy * stride;
        plane.copy_within(src..src + w, dst);
    }
}

/// Bit-exact high-bit-depth scalar reference for 4:4:4 CFL.
pub(crate) fn cfl_apply_444_hbd_scalar(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    let CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let CflArea { w, h, xlim, ylim } = area;
    let CflParams {
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

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let ysrc = &y[yrow..yrow + xlim];

        match (do_u, do_v) {
            (true, true) => {
                let udst = &mut u[urow..urow + xlim];
                let vdst = &mut v[vrow..vrow + xlim];
                for ((&yy, du), dv) in ysrc.iter().zip(udst.iter_mut()).zip(vdst.iter_mut()) {
                    let ac = ((yy as i32) << 3) - dc0;
                    *du = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                    *dv = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                }
            }
            (true, false) => {
                let udst = &mut u[urow..urow + xlim];
                for (&yy, du) in ysrc.iter().zip(udst.iter_mut()) {
                    let ac = ((yy as i32) << 3) - dc0;
                    *du = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                }
            }
            (false, true) => {
                let vdst = &mut v[vrow..vrow + xlim];
                for (&yy, dv) in ysrc.iter().zip(vdst.iter_mut()) {
                    let ac = ((yy as i32) << 3) - dc0;
                    *dv = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                }
            }
            (false, false) => unreachable!(),
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
        pad_bottom_hbd(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom_hbd(v, vrow0, cstride, w, h, ylim);
    }
}

#[inline(always)]
pub(crate) fn cfl_ac_422_hbd_scalar(
    y: &[u16],
    yrow: usize,
    x: usize,
    dc0: i32,
    filter_type: u32,
) -> i32 {
    let xl = x << 1;
    if filter_type == CFL_FLT_TYPE_GAUSS {
        ((y[yrow + xl] as i32) << 3) - dc0
    } else if filter_type == CFL_FLT_TYPE_VSTRIP {
        let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
        (y[yrow + left] as i32 + 2 * y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) * 2 - dc0
    } else {
        ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
    }
}

/// Bit-exact high-bit-depth scalar reference for 4:2:2 CFL.
pub(crate) fn cfl_apply_422_hbd_scalar(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    let CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let CflArea { w, h, xlim, ylim } = area;
    let CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        if filter_type == CFL_FLT_TYPE_GAUSS {
            let ysrc = &y[yrow..yrow + (xlim << 1)];
            match (do_u, do_v) {
                (true, true) => {
                    let udst = &mut u[urow..urow + xlim];
                    let vdst = &mut v[vrow..vrow + xlim];
                    for ((pair, du), dv) in ysrc
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(udst.iter_mut())
                        .zip(vdst.iter_mut())
                    {
                        let ac = ((pair[0] as i32) << 3) - dc0;
                        *du = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                        *dv = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                    }
                }
                (true, false) => {
                    let udst = &mut u[urow..urow + xlim];
                    for (pair, du) in ysrc.as_chunks::<2>().0.iter().zip(udst.iter_mut()) {
                        let ac = ((pair[0] as i32) << 3) - dc0;
                        *du = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                    }
                }
                (false, true) => {
                    let vdst = &mut v[vrow..vrow + xlim];
                    for (pair, dv) in ysrc.as_chunks::<2>().0.iter().zip(vdst.iter_mut()) {
                        let ac = ((pair[0] as i32) << 3) - dc0;
                        *dv = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                    }
                }
                (false, false) => unreachable!(),
            }
        } else if filter_type != CFL_FLT_TYPE_VSTRIP {
            let ysrc = &y[yrow..yrow + (xlim << 1)];
            match (do_u, do_v) {
                (true, true) => {
                    let udst = &mut u[urow..urow + xlim];
                    let vdst = &mut v[vrow..vrow + xlim];
                    for ((pair, du), dv) in ysrc
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(udst.iter_mut())
                        .zip(vdst.iter_mut())
                    {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *du = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                        *dv = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                    }
                }
                (true, false) => {
                    let udst = &mut u[urow..urow + xlim];
                    for (pair, du) in ysrc.as_chunks::<2>().0.iter().zip(udst.iter_mut()) {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *du = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                    }
                }
                (false, true) => {
                    let vdst = &mut v[vrow..vrow + xlim];
                    for (pair, dv) in ysrc.as_chunks::<2>().0.iter().zip(vdst.iter_mut()) {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *dv = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                    }
                }
                (false, false) => unreachable!(),
            }
        } else {
            for x in 0..xlim {
                let ac = cfl_ac_422_hbd_scalar(y, yrow, x, dc0, filter_type);
                if do_u {
                    u[urow + x] = predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
                }
                if do_v {
                    v[vrow + x] = predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
                }
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
        pad_bottom_hbd(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom_hbd(v, vrow0, cstride, w, h, ylim);
    }
}

#[inline(always)]
fn gen_y_filter_8bpc(
    src: &[u8],
    src_off: usize,
    top: &[u8],
    top_off: usize,
    bottom_offset: usize,
    n_left: usize,
    x: usize,
    filter_type: i32,
) -> u8 {
    let c = x << 1;
    let r = c + 1;
    let l_idx = if n_left > 0 {
        c - 1
    } else {
        (c as i32 - 1).max(0) as usize
    };
    match filter_type {
        1 => {
            ((src[src_off + l_idx] as u32
                + 2 * src[src_off + c] as u32
                + src[src_off + r] as u32
                + src[src_off + bottom_offset + l_idx] as u32
                + 2 * src[src_off + bottom_offset + c] as u32
                + src[src_off + bottom_offset + r] as u32)
                >> 3) as u8
        }
        2 => {
            ((src[src_off + l_idx] as u32
                + 4 * src[src_off + c] as u32
                + src[src_off + r] as u32
                + top[top_off + c] as u32
                + src[src_off + bottom_offset + c] as u32)
                >> 3) as u8
        }
        _ => {
            ((src[src_off + c] as u32
                + src[src_off + r] as u32
                + src[src_off + bottom_offset + c] as u32
                + src[src_off + bottom_offset + r] as u32)
                >> 2) as u8
        }
    }
}

#[inline(always)]
fn gen_y_filter_hbd(
    src: &[u16],
    src_off: usize,
    top: &[u16],
    top_off: usize,
    bottom_offset: usize,
    n_left: usize,
    x: usize,
    filter_type: i32,
) -> u16 {
    let c = x << 1;
    let r = c + 1;
    let l_idx = if n_left > 0 {
        c - 1
    } else {
        (c as i32 - 1).max(0) as usize
    };
    match filter_type {
        1 => {
            ((src[src_off + l_idx] as u32
                + 2 * src[src_off + c] as u32
                + src[src_off + r] as u32
                + src[src_off + bottom_offset + l_idx] as u32
                + 2 * src[src_off + bottom_offset + c] as u32
                + src[src_off + bottom_offset + r] as u32)
                >> 3) as u16
        }
        2 => {
            ((src[src_off + l_idx] as u32
                + 4 * src[src_off + c] as u32
                + src[src_off + r] as u32
                + top[top_off + c] as u32
                + src[src_off + bottom_offset + c] as u32)
                >> 3) as u16
        }
        _ => {
            ((src[src_off + c] as u32
                + src[src_off + r] as u32
                + src[src_off + bottom_offset + c] as u32
                + src[src_off + bottom_offset + r] as u32)
                >> 2) as u16
        }
    }
}

pub(crate) fn cfl_gen_y_row_8bpc_scalar(args: CflGenYRow8<'_>) {
    let CflGenYRow8 {
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type,
    } = args;

    for (rel_x, dst_px) in dst.iter_mut().enumerate() {
        *dst_px = gen_y_filter_8bpc(
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left,
            n_left + rel_x,
            filter_type,
        );
    }
}

pub(crate) fn cfl_gen_y_row_hbd_scalar(args: CflGenYRowHbd<'_>) {
    let CflGenYRowHbd {
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type,
    } = args;

    for (rel_x, dst_px) in dst.iter_mut().enumerate() {
        *dst_px = gen_y_filter_hbd(
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left,
            n_left + rel_x,
            filter_type,
        );
    }
}

pub(crate) fn cfl_alpha_accum_8bpc_scalar(args: CflAlphaAccum8<'_>) {
    let CflAlphaAccum8 {
        alpha,
        samples,
        sample_off,
        sample_stride,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    } = args;

    for i in 0..len {
        let v = samples[sample_off + i * sample_stride] as i32;
        alpha[0] += imat0[imat_off + i] as i32 * v;
        alpha[1] += imat1[imat_off + i] as i32 * v;
        alpha[2] += v << a2sh;
    }
}

pub(crate) fn cfl_alpha_accum_hbd_scalar(args: CflAlphaAccumHbd<'_>) {
    let CflAlphaAccumHbd {
        alpha,
        samples,
        sample_off,
        sample_stride,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    } = args;

    for i in 0..len {
        let v = samples[sample_off + i * sample_stride] as i32;
        alpha[0] += imat0[imat_off + i] as i32 * v;
        alpha[1] += imat1[imat_off + i] as i32 * v;
        alpha[2] += v << a2sh;
    }
}

const CFL_MHCCP_HAS_TOP: i32 = 1 << 2;
const CFL_MHCCP_HAS_LEFT: i32 = 1 << 3;

#[inline(always)]
fn mhccp_mul32(a: i32, b: i32) -> i32 {
    crate::ipred::mul32(a, b, 16)
}

#[inline(always)]
fn mhccp_sqrnd_8(v: i32) -> i32 {
    (v * v + 128) >> 8
}

#[inline(always)]
fn mhccp_sqrnd_hbd(v: i32, bitdepth: i32) -> i32 {
    (v * v + (1 << (bitdepth - 1))) >> bitdepth
}

#[inline(always)]
fn mhccp_pred_one_8(alpha: &[i32; 3], a2v2: i32, v0: i32, v1: i32) -> u8 {
    (mhccp_mul32(alpha[0], v0) + mhccp_mul32(alpha[1], mhccp_sqrnd_8(v1)) + a2v2).clamp(0, 255)
        as u8
}

#[inline(always)]
fn mhccp_pred_one_hbd(
    alpha: &[i32; 3],
    a2v2: i32,
    v0: i32,
    v1: i32,
    bitdepth: i32,
    bitdepth_max: i32,
) -> u16 {
    (mhccp_mul32(alpha[0], v0) + mhccp_mul32(alpha[1], mhccp_sqrnd_hbd(v1, bitdepth)) + a2v2)
        .clamp(0, bitdepth_max) as u16
}

pub(crate) fn cfl_mhccp_pred_8bpc_scalar(args: CflMhccpPred8<'_>) {
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

    let has_t = edge_flags & CFL_MHCCP_HAS_TOP != 0;
    let has_l = edge_flags & CFL_MHCCP_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let a2v2 = mhccp_mul32(alpha[2], 128);

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && y < h {
        let dst_row = &mut dst[..w];
        let prev_row = sp - src_top_stride;
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            *dst_px = mhccp_pred_one_8(&alpha, a2v2, src[prev_row + x] as i32, src[sp + x] as i32);
        }
        sp += w;
        y = 1;
    }

    for (row_y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate().skip(y) {
        let dst_row = &mut dst_row[..w];
        let mut x = 0usize;
        if dir_l && has_l && x < w {
            let v0 = src[left_off + row_y * n_left + 1] as i32;
            let v1 = src[sp] as i32;
            dst_row[0] = mhccp_pred_one_8(&alpha, a2v2, v0, v1);
            x = 1;
        }
        for (rel_x, dst_px) in dst_row[x..].iter_mut().enumerate() {
            let x = x + rel_x;
            let v0_idx = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x.saturating_sub(1)
            } else {
                sp + x
            };
            *dst_px = mhccp_pred_one_8(&alpha, a2v2, src[v0_idx] as i32, src[sp + x] as i32);
        }
        sp += w;
    }
}

pub(crate) fn cfl_mhccp_pred_hbd_scalar(args: CflMhccpPredHbd<'_>) {
    let CflMhccpPredHbd {
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
        bitdepth,
        bitdepth_max,
    } = args;

    let has_t = edge_flags & CFL_MHCCP_HAS_TOP != 0;
    let has_l = edge_flags & CFL_MHCCP_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let mid = 1 << (bitdepth - 1);
    let a2v2 = mhccp_mul32(alpha[2], mid);

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && y < h {
        let dst_row = &mut dst[..w];
        let prev_row = sp - src_top_stride;
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            *dst_px = mhccp_pred_one_hbd(
                &alpha,
                a2v2,
                src[prev_row + x] as i32,
                src[sp + x] as i32,
                bitdepth,
                bitdepth_max,
            );
        }
        sp += w;
        y = 1;
    }

    for (row_y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate().skip(y) {
        let dst_row = &mut dst_row[..w];
        let mut x = 0usize;
        if dir_l && has_l && x < w {
            let v0 = src[left_off + row_y * n_left + 1] as i32;
            let v1 = src[sp] as i32;
            dst_row[0] = mhccp_pred_one_hbd(&alpha, a2v2, v0, v1, bitdepth, bitdepth_max);
            x = 1;
        }
        for (rel_x, dst_px) in dst_row[x..].iter_mut().enumerate() {
            let x = x + rel_x;
            let v0_idx = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x.saturating_sub(1)
            } else {
                sp + x
            };
            *dst_px = mhccp_pred_one_hbd(
                &alpha,
                a2v2,
                src[v0_idx] as i32,
                src[sp + x] as i32,
                bitdepth,
                bitdepth_max,
            );
        }
        sp += w;
    }
}

static CFL_ALPHA_ACCUM_8BPC: OnceLock<CflAlphaAccum8Fn> = OnceLock::new();
static CFL_ALPHA_ACCUM_HBD: OnceLock<CflAlphaAccumHbdFn> = OnceLock::new();

#[inline]
fn resolve_cfl_alpha_accum_8bpc() -> CflAlphaAccum8Fn {
    *CFL_ALPHA_ACCUM_8BPC.get_or_init(|| {
        let mut _f: CflAlphaAccum8Fn = cfl_alpha_accum_8bpc_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_alpha_accum_8bpc_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_alpha_accum_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_alpha_accum_hbd() -> CflAlphaAccumHbdFn {
    *CFL_ALPHA_ACCUM_HBD.get_or_init(|| {
        let mut _f: CflAlphaAccumHbdFn = cfl_alpha_accum_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_alpha_accum_hbd_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_alpha_accum_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn cfl_alpha_accum_8bpc(args: CflAlphaAccum8<'_>) {
    unsafe { resolve_cfl_alpha_accum_8bpc()(args) };
}

#[inline]
pub(crate) fn cfl_alpha_accum_hbd(args: CflAlphaAccumHbd<'_>) {
    unsafe { resolve_cfl_alpha_accum_hbd()(args) };
}

static CFL_GEN_Y_ROW_8BPC: OnceLock<CflGenYRow8Fn> = OnceLock::new();
static CFL_GEN_Y_ROW_HBD: OnceLock<CflGenYRowHbdFn> = OnceLock::new();

#[inline]
fn resolve_cfl_gen_y_row_8bpc() -> CflGenYRow8Fn {
    *CFL_GEN_Y_ROW_8BPC.get_or_init(|| {
        let mut _f: CflGenYRow8Fn = cfl_gen_y_row_8bpc_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_gen_y_row_8bpc_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_gen_y_row_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_gen_y_row_hbd() -> CflGenYRowHbdFn {
    *CFL_GEN_Y_ROW_HBD.get_or_init(|| {
        let mut _f: CflGenYRowHbdFn = cfl_gen_y_row_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_gen_y_row_hbd_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_gen_y_row_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn cfl_gen_y_row_8bpc(args: CflGenYRow8<'_>) {
    unsafe { resolve_cfl_gen_y_row_8bpc()(args) };
}

#[inline]
pub(crate) fn cfl_gen_y_row_hbd(args: CflGenYRowHbd<'_>) {
    unsafe { resolve_cfl_gen_y_row_hbd()(args) };
}

static CFL_MHCCP_PRED_8BPC: OnceLock<CflMhccpPred8Fn> = OnceLock::new();
static CFL_MHCCP_PRED_HBD: OnceLock<CflMhccpPredHbdFn> = OnceLock::new();

#[inline]
fn resolve_cfl_mhccp_pred_8bpc() -> CflMhccpPred8Fn {
    *CFL_MHCCP_PRED_8BPC.get_or_init(|| {
        let mut _f: CflMhccpPred8Fn = cfl_mhccp_pred_8bpc_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_mhccp_pred_8bpc_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_mhccp_pred_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_mhccp_pred_hbd() -> CflMhccpPredHbdFn {
    *CFL_MHCCP_PRED_HBD.get_or_init(|| {
        let mut _f: CflMhccpPredHbdFn = cfl_mhccp_pred_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_mhccp_pred_hbd_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_mhccp_pred_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn cfl_mhccp_pred_8bpc(args: CflMhccpPred8<'_>) {
    unsafe { resolve_cfl_mhccp_pred_8bpc()(args) };
}

#[inline]
pub(crate) fn cfl_mhccp_pred_hbd(args: CflMhccpPredHbd<'_>) {
    unsafe { resolve_cfl_mhccp_pred_hbd()(args) };
}

static CFL_APPLY_420_8BPC: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_420_8BPC_FILTERED: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_422_8BPC: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_444_8BPC: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_420_HBD: OnceLock<CflApplyHbdFn> = OnceLock::new();
static CFL_APPLY_420_HBD_FILTERED: OnceLock<CflApplyHbdFn> = OnceLock::new();
static CFL_APPLY_422_HBD: OnceLock<CflApplyHbdFn> = OnceLock::new();
static CFL_APPLY_444_HBD: OnceLock<CflApplyHbdFn> = OnceLock::new();

#[inline]
fn resolve_cfl_apply_420() -> CflApplyFn {
    *CFL_APPLY_420_8BPC.get_or_init(|| {
        let mut _f: CflApplyFn = cfl_apply_420_8bpc_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_420_8bpc_neon;
            if ENABLE_NEON_CFL_RDM_8BPC && std::arch::is_aarch64_feature_detected!("rdm") {
                _f = crate::neon::cfl_apply_420_8bpc_neon_rdm;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cfl_apply_420_8bpc_sse41;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_420_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_apply_420_filtered() -> CflApplyFn {
    *CFL_APPLY_420_8BPC_FILTERED.get_or_init(|| {
        let mut _f: CflApplyFn = cfl_apply_420_8bpc_scalar;
        // Filtered 4:2:0 variants (VSTRIP/GAUSS) have NEON and AVX2 entries.
        // Keep SSE on the scalar fallback until its 4:2:0 entry grows the same
        // filter coverage.
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_420_8bpc_neon;
            if ENABLE_NEON_CFL_RDM_8BPC && std::arch::is_aarch64_feature_detected!("rdm") {
                _f = crate::neon::cfl_apply_420_8bpc_neon_rdm;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_420_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_apply_422() -> CflApplyFn {
    *CFL_APPLY_422_8BPC.get_or_init(|| {
        let mut _f: CflApplyFn = cfl_apply_422_8bpc_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_422_8bpc_neon;
            if ENABLE_NEON_CFL_RDM_8BPC && std::arch::is_aarch64_feature_detected!("rdm") {
                _f = crate::neon::cfl_apply_422_8bpc_neon_rdm;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cfl_apply_422_8bpc_sse41;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_422_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_apply_444() -> CflApplyFn {
    *CFL_APPLY_444_8BPC.get_or_init(|| {
        let mut _f: CflApplyFn = cfl_apply_444_8bpc_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_444_8bpc_neon;
            if ENABLE_NEON_CFL_RDM_8BPC && std::arch::is_aarch64_feature_detected!("rdm") {
                _f = crate::neon::cfl_apply_444_8bpc_neon_rdm;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cfl_apply_444_8bpc_sse41;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_444_8bpc_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn cfl_apply_420_8bpc(args: CflApply8<'_>) {
    let f = if args.params.filter_type == CFL_FLT_TYPE_VSTRIP
        || args.params.filter_type == CFL_FLT_TYPE_GAUSS
    {
        resolve_cfl_apply_420_filtered()
    } else {
        resolve_cfl_apply_420()
    };

    // SAFETY: the resolvers only install target-feature entries after the
    // matching runtime CPU feature check succeeds; scalar fallback is always valid.
    unsafe { f(args) };
}

#[inline]
pub(crate) fn cfl_apply_422_8bpc(args: CflApply8<'_>) {
    // SAFETY: the resolver only installs a target-feature entry after the
    // matching runtime CPU feature check succeeds; scalar fallback is always valid.
    unsafe { resolve_cfl_apply_422()(args) };
}

#[inline]
pub(crate) fn cfl_apply_444_8bpc(args: CflApply8<'_>) {
    // SAFETY: the resolver only installs a target-feature entry after the
    // matching runtime CPU feature check succeeds; scalar fallback is always valid.
    unsafe { resolve_cfl_apply_444()(args) };
}

#[inline]
fn resolve_cfl_apply_420_hbd() -> CflApplyHbdFn {
    *CFL_APPLY_420_HBD.get_or_init(|| {
        let mut _f: CflApplyHbdFn = cfl_apply_420_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_420_hbd_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cfl_apply_420_hbd_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_420_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_apply_420_hbd_filtered() -> CflApplyHbdFn {
    *CFL_APPLY_420_HBD_FILTERED.get_or_init(|| {
        let mut _f: CflApplyHbdFn = cfl_apply_420_hbd_scalar;
        // Filtered HBD 4:2:0 has NEON and AVX2 implementations; keep SSE on
        // the scalar fallback until its 4:2:0 entry grows the same filter coverage.
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_420_hbd_neon;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_420_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_apply_422_hbd() -> CflApplyHbdFn {
    *CFL_APPLY_422_HBD.get_or_init(|| {
        let mut _f: CflApplyHbdFn = cfl_apply_422_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_422_hbd_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cfl_apply_422_hbd_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_422_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_apply_444_hbd() -> CflApplyHbdFn {
    *CFL_APPLY_444_HBD.get_or_init(|| {
        let mut _f: CflApplyHbdFn = cfl_apply_444_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cfl_apply_444_hbd_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cfl_apply_444_hbd_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cfl_apply_444_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn cfl_apply_420_hbd(args: CflApplyHbd<'_>) {
    let f = if args.params.filter_type == CFL_FLT_TYPE_VSTRIP
        || args.params.filter_type == CFL_FLT_TYPE_GAUSS
    {
        resolve_cfl_apply_420_hbd_filtered()
    } else {
        resolve_cfl_apply_420_hbd()
    };

    // SAFETY: the resolvers only install target-feature entries after the
    // matching runtime CPU feature check succeeds; scalar fallback is always valid.
    unsafe { f(args) };
}

#[inline]
pub(crate) fn cfl_apply_422_hbd(args: CflApplyHbd<'_>) {
    // SAFETY: the resolver only installs a target-feature entry after the
    // matching runtime CPU feature check succeeds; scalar fallback is always valid.
    unsafe { resolve_cfl_apply_422_hbd()(args) };
}

#[inline]
pub(crate) fn cfl_apply_444_hbd(args: CflApplyHbd<'_>) {
    // SAFETY: the resolver only installs a target-feature entry after the
    // matching runtime CPU feature check succeeds; scalar fallback is always valid.
    unsafe { resolve_cfl_apply_444_hbd()(args) };
}
