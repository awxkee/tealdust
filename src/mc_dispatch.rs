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

#[inline]
pub(crate) fn compound_tmp_len(w: usize, h: usize) -> usize {
    (w * h).next_multiple_of(16)
}

#[inline]
pub(crate) fn inter_bilin_8bpc_tmp_len(w: usize, h: usize, mx: i32, my: i32) -> usize {
    if mx != 0 && my != 0 {
        w.next_multiple_of(16).max(64) * (h + 1)
    } else {
        0
    }
}

#[inline]
pub(crate) fn inter_8tap_8bpc_tmp_len(
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) -> usize {
    if crate::mc::get_h_filter(mx, filter_type, w).is_some()
        && crate::mc::get_v_filter(my, filter_type, h).is_some()
    {
        w.next_multiple_of(8).max(64) * (h + 7)
    } else {
        0
    }
}

#[inline]
pub(crate) fn inter_bilin_hbd_tmp_len(h: usize, mx: i32, my: i32) -> usize {
    if mx != 0 && my != 0 { 64 * (h + 1) } else { 0 }
}

#[inline]
pub(crate) fn inter_8tap_hbd_tmp_len(
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) -> usize {
    if crate::mc::get_h_filter(mx, filter_type, w).is_some()
        && crate::mc::get_v_filter(my, filter_type, h).is_some()
    {
        64 * (h + 7)
    } else {
        0
    }
}

#[inline]
fn inter_tmp(scratch: &mut Vec<i16>, len: usize) -> &mut [i16] {
    if scratch.len() < len {
        scratch.resize(len, 0);
    }
    &mut scratch[..len]
}

pub(crate) type PrepHbdFn = unsafe fn(&mut [i16], usize, &[u16], usize, usize, usize, u8);

pub(crate) fn prep_hbd_scalar(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    bitdepth: u8,
) {
    crate::mc::prep_scalar(
        <crate::pixel::BitDepth16 as crate::pixel::BitDepth>::new(bitdepth),
        tmp,
        tmp_stride,
        src,
        src_stride,
        w,
        h,
    );
}

static PREP_HBD: OnceLock<PrepHbdFn> = OnceLock::new();

#[inline]
fn resolve_prep_hbd() -> PrepHbdFn {
    *PREP_HBD.get_or_init(|| {
        let mut _f = prep_hbd_scalar as PrepHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::prep_hbd_neon as PrepHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::prep_hbd_sse41 as PrepHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::prep_hbd_avx2 as PrepHbdFn;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn prep_hbd(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    bitdepth: u8,
) {
    // SAFETY: resolver only installs kernels when the required CPU feature is present.
    unsafe { resolve_prep_hbd()(tmp, tmp_stride, src, src_stride, w, h, bitdepth) }
}

pub(crate) type PutBilinHbdFn =
    unsafe fn(&mut [u16], usize, &[u16], usize, usize, usize, i32, i32, u8, &mut [i16]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
    _mid_scratch: &mut [i16],
) {
    crate::mc::put_bilin_scalar(
        <crate::pixel::BitDepth16 as crate::pixel::BitDepth>::new(bitdepth),
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
    );
}

static PUT_BILIN_HBD: OnceLock<PutBilinHbdFn> = OnceLock::new();

#[inline]
fn resolve_put_bilin_hbd() -> PutBilinHbdFn {
    *PUT_BILIN_HBD.get_or_init(|| {
        let mut _f = put_bilin_hbd_scalar as PutBilinHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::put_bilin_hbd_neon as PutBilinHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::put_bilin_hbd_sse41 as PutBilinHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::put_bilin_hbd_avx2 as PutBilinHbdFn;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn put_bilin_hbd_with_scratch(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_bilin_hbd_tmp_len(h, mx, my));
    unsafe {
        resolve_put_bilin_hbd()(
            dst, dst_stride, src, src_stride, w, h, mx, my, bitdepth, mid,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn put_bilin_hbd(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
) {
    let mut scratch = Vec::new();
    put_bilin_hbd_with_scratch(
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth,
        &mut scratch,
    );
}

pub(crate) type PrepBilinHbdFn =
    unsafe fn(&mut [i16], usize, &[u16], usize, usize, usize, i32, i32, u8, &mut [i16]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin_hbd_scalar(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
    _mid_scratch: &mut [i16],
) {
    crate::mc::prep_bilin_scalar(
        <crate::pixel::BitDepth16 as crate::pixel::BitDepth>::new(bitdepth),
        tmp,
        tmp_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
    );
}

static PREP_BILIN_HBD: OnceLock<PrepBilinHbdFn> = OnceLock::new();

#[inline]
fn resolve_prep_bilin_hbd() -> PrepBilinHbdFn {
    *PREP_BILIN_HBD.get_or_init(|| {
        let mut _f = prep_bilin_hbd_scalar as PrepBilinHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::prep_bilin_hbd_neon as PrepBilinHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::prep_bilin_hbd_sse41 as PrepBilinHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::prep_bilin_hbd_avx2 as PrepBilinHbdFn;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn prep_bilin_hbd_with_scratch(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_bilin_hbd_tmp_len(h, mx, my));
    unsafe {
        resolve_prep_bilin_hbd()(
            tmp, tmp_stride, src, src_stride, w, h, mx, my, bitdepth, mid,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn prep_bilin_hbd(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
) {
    let mut scratch = Vec::new();
    prep_bilin_hbd_with_scratch(
        tmp,
        tmp_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        bitdepth,
        &mut scratch,
    );
}

pub(crate) type Put8tapHbdFn =
    unsafe fn(&mut [u16], usize, &[u16], usize, usize, usize, usize, i32, i32, i32, u8, &mut [i16]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
    _mid_scratch: &mut [i16],
) {
    crate::mc::put_8tap_scalar(
        <crate::pixel::BitDepth16 as crate::pixel::BitDepth>::new(bitdepth),
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

static PUT_8TAP_HBD: OnceLock<Put8tapHbdFn> = OnceLock::new();

#[inline]
fn resolve_put_8tap_hbd() -> Put8tapHbdFn {
    *PUT_8TAP_HBD.get_or_init(|| {
        let mut _f = put_8tap_hbd_scalar as Put8tapHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::put_8tap_hbd_neon as Put8tapHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::put_8tap_hbd_sse41 as Put8tapHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::put_8tap_hbd_avx2 as Put8tapHbdFn;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn put_8tap_hbd_with_scratch(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_8tap_hbd_tmp_len(w, h, mx, my, filter_type));
    unsafe {
        resolve_put_8tap_hbd()(
            dst,
            dst_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter_type,
            bitdepth,
            mid,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn put_8tap_hbd(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
) {
    let mut scratch = Vec::new();
    put_8tap_hbd_with_scratch(
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
        bitdepth,
        &mut scratch,
    );
}

pub(crate) type Prep8tapHbdFn =
    unsafe fn(&mut [i16], usize, &[u16], usize, usize, usize, usize, i32, i32, i32, u8, &mut [i16]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap_hbd_scalar(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
    _mid_scratch: &mut [i16],
) {
    crate::mc::prep_8tap_scalar(
        <crate::pixel::BitDepth16 as crate::pixel::BitDepth>::new(bitdepth),
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

static PREP_8TAP_HBD: OnceLock<Prep8tapHbdFn> = OnceLock::new();

#[inline]
fn resolve_prep_8tap_hbd() -> Prep8tapHbdFn {
    *PREP_8TAP_HBD.get_or_init(|| {
        let mut _f = prep_8tap_hbd_scalar as Prep8tapHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::prep_8tap_hbd_neon as Prep8tapHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::prep_8tap_hbd_sse41 as Prep8tapHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::prep_8tap_hbd_avx2 as Prep8tapHbdFn;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn prep_8tap_hbd_with_scratch(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_8tap_hbd_tmp_len(w, h, mx, my, filter_type));
    unsafe {
        resolve_prep_8tap_hbd()(
            tmp,
            tmp_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter_type,
            bitdepth,
            mid,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn prep_8tap_hbd(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
) {
    let mut scratch = Vec::new();
    prep_8tap_hbd_with_scratch(
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
        bitdepth,
        &mut scratch,
    );
}

pub(crate) type PutBilin8bpcFn =
    unsafe fn(&mut [u8], usize, &[u8], usize, usize, usize, i32, i32, &mut [i16]);
pub(crate) type PrepBilin8bpcFn =
    unsafe fn(&mut [i16], usize, &[u8], usize, usize, usize, i32, i32, &mut [i16]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_8bpc_scalar_dispatch(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    _mid_scratch: &mut [i16],
) {
    crate::mc::put_bilin_8bpc(dst, dst_stride, src, src_stride, w, h, mx, my);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin_8bpc_scalar_dispatch(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    _mid_scratch: &mut [i16],
) {
    crate::mc::prep_bilin_8bpc(tmp, tmp_stride, src, src_stride, w, h, mx, my);
}

static PUT_BILIN_8BPC: OnceLock<PutBilin8bpcFn> = OnceLock::new();
static PREP_BILIN_8BPC: OnceLock<PrepBilin8bpcFn> = OnceLock::new();

#[inline]
fn resolve_put_bilin_8bpc() -> PutBilin8bpcFn {
    *PUT_BILIN_8BPC.get_or_init(|| {
        let mut _f = put_bilin_8bpc_scalar_dispatch as PutBilin8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::put_bilin_8bpc_neon as PutBilin8bpcFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = put_bilin_8bpc_scalar_dispatch as PutBilin8bpcFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::put_bilin_8bpc_avx2 as PutBilin8bpcFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_prep_bilin_8bpc() -> PrepBilin8bpcFn {
    *PREP_BILIN_8BPC.get_or_init(|| {
        let mut _f = prep_bilin_8bpc_scalar_dispatch as PrepBilin8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::prep_bilin_8bpc_neon as PrepBilin8bpcFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = prep_bilin_8bpc_scalar_dispatch as PrepBilin8bpcFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::prep_bilin_8bpc_avx2 as PrepBilin8bpcFn;
            }
        }
        _f
    })
}

pub(crate) type Put8tap8bpcFn =
    unsafe fn(&mut [u8], usize, &[u8], usize, usize, usize, usize, i32, i32, i32, &mut [i16]);
pub(crate) type Prep8tap8bpcFn =
    unsafe fn(&mut [i16], usize, &[u8], usize, usize, usize, usize, i32, i32, i32, &mut [i16]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_8bpc_scalar_dispatch(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    _mid_scratch: &mut [i16],
) {
    crate::mc::put_8tap_8bpc(
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap_8bpc_scalar_dispatch(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    _mid_scratch: &mut [i16],
) {
    crate::mc::prep_8tap_8bpc(
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

static PUT_8TAP_8BPC: OnceLock<Put8tap8bpcFn> = OnceLock::new();
static PREP_8TAP_8BPC: OnceLock<Prep8tap8bpcFn> = OnceLock::new();

#[inline]
fn resolve_put_8tap_8bpc() -> Put8tap8bpcFn {
    *PUT_8TAP_8BPC.get_or_init(|| {
        let mut _f = put_8tap_8bpc_scalar_dispatch as Put8tap8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::put_8tap_8bpc_neon as Put8tap8bpcFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = put_8tap_8bpc_scalar_dispatch as Put8tap8bpcFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::put_8tap_8bpc_avx2 as Put8tap8bpcFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_prep_8tap_8bpc() -> Prep8tap8bpcFn {
    *PREP_8TAP_8BPC.get_or_init(|| {
        let mut _f = prep_8tap_8bpc_scalar_dispatch as Prep8tap8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::prep_8tap_8bpc_neon as Prep8tap8bpcFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = prep_8tap_8bpc_scalar_dispatch as Prep8tap8bpcFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::prep_8tap_8bpc_avx2 as Prep8tap8bpcFn;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn put_8tap_8bpc_with_scratch(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_8tap_8bpc_tmp_len(w, h, mx, my, filter_type));
    unsafe {
        resolve_put_8tap_8bpc()(
            dst,
            dst_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter_type,
            mid,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn prep_8tap_8bpc_with_scratch(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_8tap_8bpc_tmp_len(w, h, mx, my, filter_type));
    unsafe {
        resolve_prep_8tap_8bpc()(
            tmp,
            tmp_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter_type,
            mid,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn put_bilin_8bpc_with_scratch(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_bilin_8bpc_tmp_len(w, h, mx, my));
    unsafe { resolve_put_bilin_8bpc()(dst, dst_stride, src, src_stride, w, h, mx, my, mid) }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn prep_bilin_8bpc_with_scratch(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    scratch: &mut Vec<i16>,
) {
    let mid = inter_tmp(scratch, inter_bilin_8bpc_tmp_len(w, h, mx, my));
    unsafe { resolve_prep_bilin_8bpc()(tmp, tmp_stride, src, src_stride, w, h, mx, my, mid) }
}

pub(crate) fn avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    crate::mc::avg_8bpc(dst, dst_stride, tmp1, tmp2, w, h);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    crate::mc::w_avg_8bpc(dst, dst_stride, tmp1, tmp2, w, h, weight);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &[u8],
) {
    crate::mc::mask_8bpc(dst, dst_stride, tmp1, tmp2, w, h, m);
}

pub(crate) fn blend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    m: &[u8],
) {
    crate::mc::blend_8bpc(dst, dst_stride, tmp, w, h, m);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    // 444 (no ss), 422 (h-only), 420 (h+v). (false, true) does not occur in
    // AV2 chroma layouts; callers must not route that case here.
    crate::mc::w_mask_8bpc(
        dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        m,
        mask_stride,
        sign,
        ss_hor,
        ss_ver,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    crate::mc::warp_affine_8x8_8bpc(dst, dst_stride, src, src_stride, src_off, abcd, mx, my);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8t_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    crate::mc::warp_affine_8x8t_8bpc(tmp, tmp_stride, src, src_stride, src_off, abcd, mx, my);
}
