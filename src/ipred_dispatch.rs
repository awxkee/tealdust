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

use crate::pixel::BitDepth;

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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_v_8bpc_avx2 as IntraPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_h_8bpc_avx2 as IntraPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_smooth_8bpc_avx2 as SmoothPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_smooth_v_8bpc_avx2 as SmoothPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_smooth_h_8bpc_avx2 as SmoothPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_dc_8bpc_avx2 as IntraPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_dc_top_8bpc_avx2 as IntraPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_dc_left_8bpc_avx2 as IntraPred8Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_dc_128_8bpc_avx2 as DcPred128Fn;
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

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_paeth_8bpc_avx2 as SmoothPred8Fn;
            }
        }
        f
    })
}

pub(crate) fn ipred_paeth(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    resolve_ipred_paeth()(dst, stride, tl, o, w, h);
}

pub(crate) type Z1Pred8Fn = fn(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
);

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn ipred_z1_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    crate::ipred::ipred_z1_8bpc(
        dst,
        stride,
        tl,
        o,
        w,
        h,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

static IPRED_Z1_8BPC: OnceLock<Z1Pred8Fn> = OnceLock::new();

#[inline]
fn resolve_ipred_z1() -> Z1Pred8Fn {
    *IPRED_Z1_8BPC.get_or_init(|| {
        #[allow(unused_mut)]
        let mut f = ipred_z1_scalar as Z1Pred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_z1_8bpc_neon as Z1Pred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_z1_8bpc_sse41 as Z1Pred8Fn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_z1_8bpc_avx2 as Z1Pred8Fn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    resolve_ipred_z1()(
        dst,
        stride,
        tl,
        o,
        w,
        h,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn ipred_z3_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    crate::ipred::ipred_z3_8bpc(
        dst,
        stride,
        tl,
        o,
        w,
        h,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

static IPRED_Z3_8BPC: OnceLock<Z1Pred8Fn> = OnceLock::new();

#[inline]
fn resolve_ipred_z3() -> Z1Pred8Fn {
    *IPRED_Z3_8BPC.get_or_init(|| {
        #[allow(unused_mut)]
        let mut f = ipred_z3_scalar as Z1Pred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_z3_8bpc_neon as Z1Pred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_z3_8bpc_sse41 as Z1Pred8Fn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_z3_8bpc_avx2 as Z1Pred8Fn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    resolve_ipred_z3()(
        dst,
        stride,
        tl,
        o,
        w,
        h,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

pub(crate) type Z2Pred8Fn = fn(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
);

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn ipred_z2_scalar(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    crate::ipred::ipred_z2_8bpc(dst, stride, tl, o, w, h, angle, max_width, max_height);
}

static IPRED_Z2_8BPC: OnceLock<Z2Pred8Fn> = OnceLock::new();

#[inline]
fn resolve_ipred_z2() -> Z2Pred8Fn {
    *IPRED_Z2_8BPC.get_or_init(|| {
        #[allow(unused_mut)]
        let mut f = ipred_z2_scalar as Z2Pred8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ipred_z2_8bpc_neon as Z2Pred8Fn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ipred_z2_8bpc_sse41 as Z2Pred8Fn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::ipred_z2_8bpc_avx2 as Z2Pred8Fn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    resolve_ipred_z2()(dst, stride, tl, o, w, h, angle, max_width, max_height);
}

pub(crate) type IntraPredHbdFn = fn(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: u16,
);

pub(crate) type SmoothPredHbdFn = fn(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
);

pub(crate) type DcPred128HbdFn =
    fn(dst: &mut [u16], stride: usize, width: usize, height: usize, bitdepth_max: u16);

pub(crate) type Z1PredHbdFn = fn(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
);

pub(crate) type Z2PredHbdFn = fn(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: u16,
);

#[inline(always)]
fn hbd_from_max(bitdepth_max: u16) -> crate::pixel::BitDepth16 {
    crate::pixel::BitDepth16::new(if bitdepth_max <= 1023 { 10 } else { 12 })
}

#[inline]
pub(crate) fn ipred_v_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_v(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
    );
}

#[inline]
pub(crate) fn ipred_h_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_h(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
    );
}

#[inline]
pub(crate) fn ipred_dc_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_dc(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
    );
}

#[inline]
pub(crate) fn ipred_dc_top_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_dc_top(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
    );
}

#[inline]
pub(crate) fn ipred_dc_left_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_dc_left(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
    );
}

#[inline]
pub(crate) fn ipred_dc_128_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_dc_128(hbd_from_max(bitdepth_max), dst, stride, width, height);
}

#[inline]
pub(crate) fn ipred_paeth_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_paeth(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
    );
}

#[inline]
pub(crate) fn ipred_smooth_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_smooth(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
    );
}

#[inline]
pub(crate) fn ipred_smooth_v_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_smooth_v(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
    );
}

#[inline]
pub(crate) fn ipred_smooth_h_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_smooth_h(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn ipred_z1_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    crate::ipred::ipred_z1(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn ipred_z3_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    crate::ipred::ipred_z3(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn ipred_z2_hbd_scalar(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: u16,
) {
    crate::ipred::ipred_z2(
        hbd_from_max(bitdepth_max),
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
        max_width,
        max_height,
    );
}

static IPRED_V_HBD: OnceLock<IntraPredHbdFn> = OnceLock::new();
static IPRED_H_HBD: OnceLock<IntraPredHbdFn> = OnceLock::new();
static IPRED_DC_HBD: OnceLock<IntraPredHbdFn> = OnceLock::new();
static IPRED_DC_TOP_HBD: OnceLock<IntraPredHbdFn> = OnceLock::new();
static IPRED_DC_LEFT_HBD: OnceLock<IntraPredHbdFn> = OnceLock::new();
static IPRED_DC_128_HBD: OnceLock<DcPred128HbdFn> = OnceLock::new();
static IPRED_PAETH_HBD: OnceLock<SmoothPredHbdFn> = OnceLock::new();
static IPRED_SMOOTH_HBD: OnceLock<SmoothPredHbdFn> = OnceLock::new();
static IPRED_SMOOTH_V_HBD: OnceLock<SmoothPredHbdFn> = OnceLock::new();
static IPRED_SMOOTH_H_HBD: OnceLock<SmoothPredHbdFn> = OnceLock::new();
static IPRED_Z1_HBD: OnceLock<Z1PredHbdFn> = OnceLock::new();
static IPRED_Z2_HBD: OnceLock<Z2PredHbdFn> = OnceLock::new();
static IPRED_Z3_HBD: OnceLock<Z1PredHbdFn> = OnceLock::new();

macro_rules! resolve_hbd_ipred {
    ($lock:ident, $scalar:path, $sse:path, $avx:path, $neon:path, $ty:ty) => {{
        *$lock.get_or_init(|| {
            let mut f = $scalar as $ty;
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    f = $neon as $ty;
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("sse4.1") {
                    f = $sse as $ty;
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    f = $avx as $ty;
                }
            }
            f
        })
    }};
}

#[inline]
fn resolve_ipred_v_hbd() -> IntraPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_V_HBD,
        ipred_v_hbd_scalar,
        crate::sse::ipred_v_hbd_sse41,
        crate::avx::ipred_v_hbd_avx2,
        crate::neon::ipred_v_hbd_neon,
        IntraPredHbdFn
    )
}

#[inline]
fn resolve_ipred_h_hbd() -> IntraPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_H_HBD,
        ipred_h_hbd_scalar,
        crate::sse::ipred_h_hbd_sse41,
        crate::avx::ipred_h_hbd_avx2,
        crate::neon::ipred_h_hbd_neon,
        IntraPredHbdFn
    )
}

#[inline]
fn resolve_ipred_dc_hbd() -> IntraPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_DC_HBD,
        ipred_dc_hbd_scalar,
        crate::sse::ipred_dc_hbd_sse41,
        crate::avx::ipred_dc_hbd_avx2,
        crate::neon::ipred_dc_hbd_neon,
        IntraPredHbdFn
    )
}

#[inline]
fn resolve_ipred_dc_top_hbd() -> IntraPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_DC_TOP_HBD,
        ipred_dc_top_hbd_scalar,
        crate::sse::ipred_dc_top_hbd_sse41,
        crate::avx::ipred_dc_top_hbd_avx2,
        crate::neon::ipred_dc_top_hbd_neon,
        IntraPredHbdFn
    )
}

#[inline]
fn resolve_ipred_dc_left_hbd() -> IntraPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_DC_LEFT_HBD,
        ipred_dc_left_hbd_scalar,
        crate::sse::ipred_dc_left_hbd_sse41,
        crate::avx::ipred_dc_left_hbd_avx2,
        crate::neon::ipred_dc_left_hbd_neon,
        IntraPredHbdFn
    )
}

#[inline]
fn resolve_ipred_dc_128_hbd() -> DcPred128HbdFn {
    resolve_hbd_ipred!(
        IPRED_DC_128_HBD,
        ipred_dc_128_hbd_scalar,
        crate::sse::ipred_dc_128_hbd_sse41,
        crate::avx::ipred_dc_128_hbd_avx2,
        crate::neon::ipred_dc_128_hbd_neon,
        DcPred128HbdFn
    )
}

#[inline]
fn resolve_ipred_paeth_hbd() -> SmoothPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_PAETH_HBD,
        ipred_paeth_hbd_scalar,
        crate::sse::ipred_paeth_hbd_sse41,
        crate::avx::ipred_paeth_hbd_avx2,
        crate::neon::ipred_paeth_hbd_neon,
        SmoothPredHbdFn
    )
}

#[inline]
fn resolve_ipred_smooth_hbd() -> SmoothPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_SMOOTH_HBD,
        ipred_smooth_hbd_scalar,
        crate::sse::ipred_smooth_hbd_sse41,
        crate::avx::ipred_smooth_hbd_avx2,
        crate::neon::ipred_smooth_hbd_neon,
        SmoothPredHbdFn
    )
}

#[inline]
fn resolve_ipred_smooth_v_hbd() -> SmoothPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_SMOOTH_V_HBD,
        ipred_smooth_v_hbd_scalar,
        crate::sse::ipred_smooth_v_hbd_sse41,
        crate::avx::ipred_smooth_v_hbd_avx2,
        crate::neon::ipred_smooth_v_hbd_neon,
        SmoothPredHbdFn
    )
}

#[inline]
fn resolve_ipred_smooth_h_hbd() -> SmoothPredHbdFn {
    resolve_hbd_ipred!(
        IPRED_SMOOTH_H_HBD,
        ipred_smooth_h_hbd_scalar,
        crate::sse::ipred_smooth_h_hbd_sse41,
        crate::avx::ipred_smooth_h_hbd_avx2,
        crate::neon::ipred_smooth_h_hbd_neon,
        SmoothPredHbdFn
    )
}

#[inline]
fn resolve_ipred_z1_hbd() -> Z1PredHbdFn {
    resolve_hbd_ipred!(
        IPRED_Z1_HBD,
        ipred_z1_hbd_scalar,
        crate::sse::ipred_z1_hbd_sse41,
        crate::avx::ipred_z1_hbd_avx2,
        crate::neon::ipred_z1_hbd_neon,
        Z1PredHbdFn
    )
}

#[inline]
fn resolve_ipred_z2_hbd() -> Z2PredHbdFn {
    resolve_hbd_ipred!(
        IPRED_Z2_HBD,
        ipred_z2_hbd_scalar,
        crate::sse::ipred_z2_hbd_sse41,
        crate::avx::ipred_z2_hbd_avx2,
        crate::neon::ipred_z2_hbd_neon,
        Z2PredHbdFn
    )
}

#[inline]
fn resolve_ipred_z3_hbd() -> Z1PredHbdFn {
    resolve_hbd_ipred!(
        IPRED_Z3_HBD,
        ipred_z3_hbd_scalar,
        crate::sse::ipred_z3_hbd_sse41,
        crate::avx::ipred_z3_hbd_avx2,
        crate::neon::ipred_z3_hbd_neon,
        Z1PredHbdFn
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_ipred_hbd(
    m: u8,
    bitdepth: u8,
    bitdepth_max: u16,
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    edge: &[u16],
    edge_o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_w: i32,
    max_h: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let d = &mut dst[dst_off..];
    match m {
        0 /* DcPred */ => resolve_ipred_dc_hbd()(d, stride, edge, edge_o, w, h, angle, bitdepth_max),
        _ if m == DC_128_PRED => resolve_ipred_dc_128_hbd()(d, stride, w, h, bitdepth_max),
        _ if m == TOP_DC_PRED => resolve_ipred_dc_top_hbd()(d, stride, edge, edge_o, w, h, angle, bitdepth_max),
        _ if m == LEFT_DC_PRED => resolve_ipred_dc_left_hbd()(d, stride, edge, edge_o, w, h, angle, bitdepth_max),
        2 /* HorPred */ => resolve_ipred_h_hbd()(d, stride, edge, edge_o, w, h, angle, bitdepth_max),
        1 /* VertPred */ => resolve_ipred_v_hbd()(d, stride, edge, edge_o, w, h, angle, bitdepth_max),
        12 /* PaethPred */ => resolve_ipred_paeth_hbd()(d, stride, edge, edge_o, w, h, bitdepth_max),
        9 /* SmoothPred */ => resolve_ipred_smooth_hbd()(d, stride, edge, edge_o, w, h, bitdepth_max),
        10 /* SmoothVPred */ => resolve_ipred_smooth_v_hbd()(d, stride, edge, edge_o, w, h, bitdepth_max),
        11 /* SmoothHPred */ => resolve_ipred_smooth_h_hbd()(d, stride, edge, edge_o, w, h, bitdepth_max),
        _ if m == Z1_PRED => resolve_ipred_z1_hbd()(d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights, bitdepth_max),
        _ if m == Z2_PRED => resolve_ipred_z2_hbd()(d, stride, edge, edge_o, w, h, angle, max_w, max_h, bitdepth_max),
        _ if m == Z3_PRED => resolve_ipred_z3_hbd()(d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights, bitdepth_max),
        _ if m == DIP_PRED => crate::ipred::ipred_dip(crate::pixel::BitDepth16::new(bitdepth), d, stride, edge, edge_o, w, h, angle),
        _ => resolve_ipred_dc_128_hbd()(d, stride, w, h, bitdepth_max),
    }
}
