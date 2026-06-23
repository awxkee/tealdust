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

const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

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

pub(crate) type CflApplyFn = for<'a> fn(CflApply8<'a>);
pub(crate) type CflApplyHbdFn = for<'a> fn(CflApplyHbd<'a>);

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
        for x in 0..xlim {
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
                    for ((&yy, du), dv) in ysrc
                        .iter()
                        .step_by(2)
                        .take(xlim)
                        .zip(udst.iter_mut())
                        .zip(vdst.iter_mut())
                    {
                        let ac = ((yy as i32) << 3) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                        *dv = predict_one(dc2, alpha1, ac);
                    }
                }
                (true, false) => {
                    let udst = &mut u[urow..urow + xlim];
                    for (&yy, du) in ysrc.iter().step_by(2).take(xlim).zip(udst.iter_mut()) {
                        let ac = ((yy as i32) << 3) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                    }
                }
                (false, true) => {
                    let vdst = &mut v[vrow..vrow + xlim];
                    for (&yy, dv) in ysrc.iter().step_by(2).take(xlim).zip(vdst.iter_mut()) {
                        let ac = ((yy as i32) << 3) - dc0;
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
                        .chunks_exact(2)
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
                    for (pair, du) in ysrc.chunks_exact(2).zip(udst.iter_mut()) {
                        let ac = ((pair[0] as i32 + pair[1] as i32) << 2) - dc0;
                        *du = predict_one(dc1, alpha0, ac);
                    }
                }
                (false, true) => {
                    let vdst = &mut v[vrow..vrow + xlim];
                    for (pair, dv) in ysrc.chunks_exact(2).zip(vdst.iter_mut()) {
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
        for x in 0..xlim {
            let xl = x << 1;
            let ac = ((y[yrow + xl] as i32
                + y[yrow + xl + 1] as i32
                + y[yrow + xl + ystride] as i32
                + y[yrow + xl + ystride + 1] as i32)
                << 1)
                - dc0;
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
        for x in 0..xlim {
            let ac = ((y[yrow + x] as i32) << 3) - dc0;
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
        for x in 0..xlim {
            let ac = cfl_ac_422_hbd_scalar(y, yrow, x, dc0, filter_type);
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

static CFL_APPLY_420_8BPC: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_422_8BPC: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_444_8BPC: OnceLock<CflApplyFn> = OnceLock::new();
static CFL_APPLY_420_HBD: OnceLock<CflApplyHbdFn> = OnceLock::new();
static CFL_APPLY_422_HBD: OnceLock<CflApplyHbdFn> = OnceLock::new();
static CFL_APPLY_444_HBD: OnceLock<CflApplyHbdFn> = OnceLock::new();

#[inline]
fn resolve_cfl_apply_420() -> CflApplyFn {
    *CFL_APPLY_420_8BPC.get_or_init(|| {
        let mut f = cfl_apply_420_8bpc_scalar as CflApplyFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_420_8bpc_neon as CflApplyFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_420_8bpc_sse41 as CflApplyFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::cfl_apply_420_8bpc_avx2 as CflApplyFn;
            }
        }
        f
    })
}

#[inline]
fn resolve_cfl_apply_422() -> CflApplyFn {
    *CFL_APPLY_422_8BPC.get_or_init(|| {
        let mut f = cfl_apply_422_8bpc_scalar as CflApplyFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_422_8bpc_neon as CflApplyFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_422_8bpc_sse41 as CflApplyFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::cfl_apply_422_8bpc_avx2 as CflApplyFn;
            }
        }
        f
    })
}

#[inline]
fn resolve_cfl_apply_444() -> CflApplyFn {
    *CFL_APPLY_444_8BPC.get_or_init(|| {
        let mut f = cfl_apply_444_8bpc_scalar as CflApplyFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_444_8bpc_neon as CflApplyFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_444_8bpc_sse41 as CflApplyFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::cfl_apply_444_8bpc_avx2 as CflApplyFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn cfl_apply_420_8bpc(args: CflApply8<'_>) {
    resolve_cfl_apply_420()(args);
}

#[inline]
pub(crate) fn cfl_apply_422_8bpc(args: CflApply8<'_>) {
    resolve_cfl_apply_422()(args);
}

#[inline]
pub(crate) fn cfl_apply_444_8bpc(args: CflApply8<'_>) {
    resolve_cfl_apply_444()(args);
}

#[inline]
fn resolve_cfl_apply_420_hbd() -> CflApplyHbdFn {
    *CFL_APPLY_420_HBD.get_or_init(|| {
        let mut f = cfl_apply_420_hbd_scalar as CflApplyHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_420_hbd_neon as CflApplyHbdFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_420_hbd_sse41 as CflApplyHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::cfl_apply_420_hbd_avx2 as CflApplyHbdFn;
            }
        }
        f
    })
}

#[inline]
fn resolve_cfl_apply_422_hbd() -> CflApplyHbdFn {
    *CFL_APPLY_422_HBD.get_or_init(|| {
        let mut f = cfl_apply_422_hbd_scalar as CflApplyHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_422_hbd_neon as CflApplyHbdFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_422_hbd_sse41 as CflApplyHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::cfl_apply_422_hbd_avx2 as CflApplyHbdFn;
            }
        }
        f
    })
}

#[inline]
fn resolve_cfl_apply_444_hbd() -> CflApplyHbdFn {
    *CFL_APPLY_444_HBD.get_or_init(|| {
        let mut f = cfl_apply_444_hbd_scalar as CflApplyHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_444_hbd_neon as CflApplyHbdFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_444_hbd_sse41 as CflApplyHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::cfl_apply_444_hbd_avx2 as CflApplyHbdFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn cfl_apply_420_hbd(args: CflApplyHbd<'_>) {
    resolve_cfl_apply_420_hbd()(args);
}

#[inline]
pub(crate) fn cfl_apply_422_hbd(args: CflApplyHbd<'_>) {
    resolve_cfl_apply_422_hbd()(args);
}

#[inline]
pub(crate) fn cfl_apply_444_hbd(args: CflApplyHbd<'_>) {
    resolve_cfl_apply_444_hbd()(args);
}
