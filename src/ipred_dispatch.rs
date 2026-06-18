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

pub(crate) type IntraPred8Fn =
    fn(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, width: usize, height: usize, angle: i32);

pub(crate) type SmoothPred8Fn =
    fn(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, width: usize, height: usize);

pub(crate) trait IntraPred8Backend {
    fn ipred_v(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
        angle: i32,
    );

    fn ipred_h(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
        angle: i32,
    );

    fn ipred_smooth(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
    );

    fn ipred_smooth_v(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
    );

    fn ipred_smooth_h(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
    );
}

pub(crate) struct ScalarIntraPred8;

impl IntraPred8Backend for ScalarIntraPred8 {
    #[inline(always)]
    fn ipred_v(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
        angle: i32,
    ) {
        crate::ipred::ipred_v_8bpc(dst, stride, tl, o, width, height, angle);
    }

    #[inline(always)]
    fn ipred_h(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
        angle: i32,
    ) {
        crate::ipred::ipred_h_8bpc(dst, stride, tl, o, width, height, angle);
    }

    #[inline(always)]
    fn ipred_smooth(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
    ) {
        crate::ipred::ipred_smooth_8bpc(dst, stride, tl, o, width, height);
    }

    #[inline(always)]
    fn ipred_smooth_v(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
    ) {
        crate::ipred::ipred_smooth_v_8bpc(dst, stride, tl, o, width, height);
    }

    #[inline(always)]
    fn ipred_smooth_h(
        dst: &mut [u8],
        stride: usize,
        tl: &[u8],
        o: usize,
        width: usize,
        height: usize,
    ) {
        crate::ipred::ipred_smooth_h_8bpc(dst, stride, tl, o, width, height);
    }
}

#[inline]
pub(crate) fn ipred_v_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ScalarIntraPred8::ipred_v(dst, stride, tl, o, width, height, angle);
}

#[inline]
pub(crate) fn ipred_h_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ScalarIntraPred8::ipred_h(dst, stride, tl, o, width, height, angle);
}

#[inline]
pub(crate) fn ipred_smooth_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    ScalarIntraPred8::ipred_smooth(dst, stride, tl, o, width, height);
}

#[inline]
pub(crate) fn ipred_smooth_v_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    ScalarIntraPred8::ipred_smooth_v(dst, stride, tl, o, width, height);
}

#[inline]
pub(crate) fn ipred_smooth_h_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    ScalarIntraPred8::ipred_smooth_h(dst, stride, tl, o, width, height);
}

static IPRED_V_8BPC: OnceLock<IntraPred8Fn> = OnceLock::new();
static IPRED_H_8BPC: OnceLock<IntraPred8Fn> = OnceLock::new();
static IPRED_SMOOTH_8BPC: OnceLock<SmoothPred8Fn> = OnceLock::new();
static IPRED_SMOOTH_V_8BPC: OnceLock<SmoothPred8Fn> = OnceLock::new();
static IPRED_SMOOTH_H_8BPC: OnceLock<SmoothPred8Fn> = OnceLock::new();

#[inline]
fn resolve_ipred_v() -> IntraPred8Fn {
    *IPRED_V_8BPC.get_or_init(|| {
        let mut f = ipred_v_scalar as IntraPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_v_8bpc_neon as IntraPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_v_8bpc_sse41 as IntraPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_h() -> IntraPred8Fn {
    *IPRED_H_8BPC.get_or_init(|| {
        let mut f = ipred_h_scalar as IntraPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_h_8bpc_neon as IntraPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_h_8bpc_sse41 as IntraPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_smooth() -> SmoothPred8Fn {
    *IPRED_SMOOTH_8BPC.get_or_init(|| {
        let mut f = ipred_smooth_scalar as SmoothPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_smooth_8bpc_neon as SmoothPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_smooth_8bpc_sse41 as SmoothPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_smooth_v() -> SmoothPred8Fn {
    *IPRED_SMOOTH_V_8BPC.get_or_init(|| {
        let mut f = ipred_smooth_v_scalar as SmoothPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_smooth_v_8bpc_neon as SmoothPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_smooth_v_8bpc_sse41 as SmoothPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_smooth_h() -> SmoothPred8Fn {
    *IPRED_SMOOTH_H_8BPC.get_or_init(|| {
        let mut f = ipred_smooth_h_scalar as SmoothPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_smooth_h_8bpc_neon as SmoothPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_smooth_h_8bpc_sse41 as SmoothPred8Fn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_v(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    resolve_ipred_v()(dst, stride, tl, o, width, height, angle);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_h(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    resolve_ipred_h()(dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_smooth(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    resolve_ipred_smooth()(dst, stride, tl, o, w, h);
}

pub(crate) fn ipred_smooth_v(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    resolve_ipred_smooth_v()(dst, stride, tl, o, w, h);
}

pub(crate) fn ipred_smooth_h(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    resolve_ipred_smooth_h()(dst, stride, tl, o, w, h);
}

// ---------------------------------------------------------------------------
// DC family dispatch. dc/dc_top/dc_left share IntraPred8Fn (they take `angle`);
// dc_128 has no top-left edge or angle.
// ---------------------------------------------------------------------------

pub(crate) type DcPred128Fn = fn(dst: &mut [u8], stride: usize, width: usize, height: usize);

#[inline]
pub(crate) fn ipred_dc_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    crate::ipred::ipred_dc_8bpc(dst, stride, tl, o, width, height, angle);
}

#[inline]
pub(crate) fn ipred_dc_top_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    crate::ipred::ipred_dc_top_8bpc(dst, stride, tl, o, width, height, angle);
}

#[inline]
pub(crate) fn ipred_dc_left_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    crate::ipred::ipred_dc_left_8bpc(dst, stride, tl, o, width, height, angle);
}

#[inline]
pub(crate) fn ipred_dc_128_scalar(dst: &mut [u8], stride: usize, width: usize, height: usize) {
    crate::ipred::ipred_dc_128_8bpc(dst, stride, width, height);
}

static IPRED_DC_8BPC: OnceLock<IntraPred8Fn> = OnceLock::new();
static IPRED_DC_TOP_8BPC: OnceLock<IntraPred8Fn> = OnceLock::new();
static IPRED_DC_LEFT_8BPC: OnceLock<IntraPred8Fn> = OnceLock::new();
static IPRED_DC_128_8BPC: OnceLock<DcPred128Fn> = OnceLock::new();

#[inline]
fn resolve_ipred_dc() -> IntraPred8Fn {
    *IPRED_DC_8BPC.get_or_init(|| {
        let mut f = ipred_dc_scalar as IntraPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_dc_8bpc_neon as IntraPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_dc_8bpc_sse41 as IntraPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_dc_top() -> IntraPred8Fn {
    *IPRED_DC_TOP_8BPC.get_or_init(|| {
        let mut f = ipred_dc_top_scalar as IntraPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_dc_top_8bpc_neon as IntraPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_dc_top_8bpc_sse41 as IntraPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_dc_left() -> IntraPred8Fn {
    *IPRED_DC_LEFT_8BPC.get_or_init(|| {
        let mut f = ipred_dc_left_scalar as IntraPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_dc_left_8bpc_neon as IntraPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_dc_left_8bpc_sse41 as IntraPred8Fn;
            }
        }
        f
    })
}

#[inline]
fn resolve_ipred_dc_128() -> DcPred128Fn {
    *IPRED_DC_128_8BPC.get_or_init(|| {
        let mut f = ipred_dc_128_scalar as DcPred128Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_dc_128_8bpc_neon as DcPred128Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_dc_128_8bpc_sse41 as DcPred128Fn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_dc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    resolve_ipred_dc()(dst, stride, tl, o, width, height, angle);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_dc_top(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    resolve_ipred_dc_top()(dst, stride, tl, o, width, height, angle);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_dc_left(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    resolve_ipred_dc_left()(dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_dc_128(dst: &mut [u8], stride: usize, width: usize, height: usize) {
    resolve_ipred_dc_128()(dst, stride, width, height);
}

// --------------------------------- Paeth -----------------------------------

#[inline]
pub(crate) fn ipred_paeth_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    crate::ipred::ipred_paeth_8bpc(dst, stride, tl, o, w, h);
}

static IPRED_PAETH_8BPC: OnceLock<SmoothPred8Fn> = OnceLock::new();

#[inline]
fn resolve_ipred_paeth() -> SmoothPred8Fn {
    *IPRED_PAETH_8BPC.get_or_init(|| {
        let mut f = ipred_paeth_scalar as SmoothPred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_paeth_8bpc_neon as SmoothPred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_paeth_8bpc_sse41 as SmoothPred8Fn;
            }
        }
        f
    })
}

pub(crate) fn ipred_paeth(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    resolve_ipred_paeth()(dst, stride, tl, o, w, h);
}
