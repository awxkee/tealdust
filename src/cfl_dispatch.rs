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

pub(crate) type CflApply420Fn = fn(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
);

#[inline(always)]
fn predict_one(dc: i32, alpha: i32, ac: i32) -> u8 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, 255) as u8
}

/// Bit-exact scalar reference. Mirrors the 420 / uniform branch of the
/// `cfl_pred_raw` apply loop, including the right/bottom padding.
pub(crate) fn cfl_apply_420_8bpc_scalar(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
) {
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
            if alpha0 != 0 {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if alpha1 != 0 {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }
        if alpha0 != 0 {
            let last = u[urow + xlim - 1];
            for xpad in xlim..w {
                u[urow + xpad] = last;
            }
        }
        if alpha1 != 0 {
            let last = v[vrow + xlim - 1];
            for xpad in xlim..w {
                v[vrow + xpad] = last;
            }
        }
        yrow += ystride << 1;
        urow += cstride;
        vrow += cstride;
    }
    if alpha0 != 0 {
        let src = urow0 + (ylim - 1) * cstride;
        for yy in ylim..h {
            let dst = urow0 + yy * cstride;
            u.copy_within(src..src + w, dst);
        }
    }
    if alpha1 != 0 {
        let src = vrow0 + (ylim - 1) * cstride;
        for yy in ylim..h {
            let dst = vrow0 + yy * cstride;
            v.copy_within(src..src + w, dst);
        }
    }
}

static CFL_APPLY_420_8BPC: OnceLock<CflApply420Fn> = OnceLock::new();

#[inline]
fn resolve_cfl_apply_420() -> CflApply420Fn {
    *CFL_APPLY_420_8BPC.get_or_init(|| {
        let mut f = cfl_apply_420_8bpc_scalar as CflApply420Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cfl_apply_420_8bpc_neon as CflApply420Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cfl_apply_420_8bpc_sse41 as CflApply420Fn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cfl_apply_420_8bpc(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
) {
    resolve_cfl_apply_420()(
        y, u, v, yrow0, urow0, vrow0, ystride, cstride, w, h, xlim, ylim, dc0, dc1, dc2, alpha0,
        alpha1,
    );
}
