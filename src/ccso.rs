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
use crate::intops::iclip;
use crate::pixel::{BitDepth, Pixel};
use std::sync::OnceLock;

pub(crate) static CCSO_POS: [[i8; 2]; 7] = [
    [-1, 0],
    [0, -1],
    [-1, -1],
    [-1, 1],
    [-1, -2],
    [1, -2],
    [0, 2],
];

#[inline(always)]
pub(crate) fn ccso_score(diff: i32, quant_step: i32, edge_classifier: u32) -> u32 {
    if diff > quant_step && edge_classifier == 0 {
        return 2;
    }
    if diff < -quant_step {
        return 0;
    }
    1
}

#[inline(always)]
pub(crate) fn ccso_offset(i: u8, offset_idxs: &[u8], offset_lut: &[i8]) -> i8 {
    let byte_idx = (i >> 1) as usize;
    let half_idx = (i & 1) as usize;
    let offset_idx = (7 & (offset_idxs[byte_idx] >> (4 * half_idx))) as usize;
    offset_lut[offset_idx]
}

#[inline(always)]
pub(crate) fn ccso_build_offset_map(offset_idxs: &[u8], offset_lut: &[i8]) -> [i8; 256] {
    let mut map = [0i8; 256];
    let mut i = 0usize;
    while i < 128 {
        map[i] = ccso_offset(i as u8, offset_idxs, offset_lut);
        i += 1;
    }
    while i < 256 {
        map[i] = map[i - 128];
        i += 1;
    }
    map
}

#[allow(clippy::too_many_arguments)]
fn ccso_padding<P: Pixel>(
    tmp: &mut [P],
    tmp_stride: usize,
    o: usize,
    src: &[P],
    src_stride: usize,
    src_off: usize,
    left: &[[P; 2]],
    top: &[P],
    top_off: usize,
    bottom: &[P],
    bottom_off: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    let x_min: i32 = if edges & CDEF_HAVE_LEFT != 0 { -2 } else { 0 };
    let x_max: i32 = w as i32 - 1 + if edges & CDEF_HAVE_RIGHT != 0 { 2 } else { 0 };
    let y_min: i32 = if edges & CDEF_HAVE_TOP != 0 { -2 } else { 0 };
    let y_max: i32 = h as i32 - 1 + if edges & CDEF_HAVE_BOTTOM != 0 { 2 } else { 0 };

    for y in -2i32..h as i32 + 2 {
        let src_y = iclip(y, y_min, y_max);
        for x in -2i32..w as i32 + 2 {
            let src_x = iclip(x, x_min, x_max);
            let v = if src_y < 0 {
                top[(top_off as i32 + src_x + (2 + src_y) * src_stride as i32) as usize]
            } else if src_y >= h as i32 {
                bottom
                    [(bottom_off as i32 + src_x + (src_y - h as i32) * src_stride as i32) as usize]
            } else if src_x < 0 {
                left[src_y as usize][(2 + src_x) as usize]
            } else {
                src[(src_off as i32 + src_x + src_y * src_stride as i32) as usize]
            };
            tmp[(o as i32 + x + y * tmp_stride as i32) as usize] = v;
        }
    }
}

pub(crate) struct CcsoPrepSrc<'a, P: Pixel> {
    pub(crate) src: &'a [P],
    pub(crate) stride: usize,
    pub(crate) off: usize,
    pub(crate) left: &'a [[P; 2]],
    pub(crate) top: &'a [P],
    pub(crate) top_off: usize,
    pub(crate) bottom: &'a [P],
    pub(crate) bottom_off: usize,
}

pub(crate) struct CcsoPrepCfg {
    pub(crate) max_band_log2: u32,
    pub(crate) ext_filter: usize,
    pub(crate) quant_step: i32,
    pub(crate) edge_clf: u32,
    pub(crate) bo_only: bool,
}

pub(crate) struct CcsoPrepArea {
    pub(crate) edges: u8,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) ss_hor: usize,
    pub(crate) ss_ver: usize,
}

pub(crate) struct CcsoPrepCtx<'a, P: Pixel> {
    pub(crate) dst: &'a mut [u8],
    pub(crate) dst_stride: usize,
    pub(crate) src: CcsoPrepSrc<'a, P>,
    pub(crate) tmp_buf: &'a mut Vec<P>,
    pub(crate) cfg: CcsoPrepCfg,
    pub(crate) area: CcsoPrepArea,
}

#[allow(clippy::too_many_arguments)]
pub(crate) type CcsoPrep8bpcFn = unsafe fn(
    &mut [u8],
    usize,
    &[u8],
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    u32,
    isize,
    i32,
    u32,
    bool,
);

#[allow(clippy::too_many_arguments)]
pub(crate) type CcsoPrepHbdFn = unsafe fn(
    &mut [u8],
    usize,
    &[u16],
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    u32,
    isize,
    i32,
    u32,
    bool,
);

pub(crate) type CcsoAdd8bpcFn =
    unsafe fn(&mut [u8], usize, &[u8], usize, &[u8], &[i8], usize, usize, &[[u16; 4]]);

pub(crate) type CcsoAddHbdFn =
    unsafe fn(&mut [u16], usize, &[u8], usize, &[u8], &[i8], usize, usize, &[[u16; 4]], i32);

#[allow(clippy::too_many_arguments)]
pub(crate) fn ccso_prep_lut_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    tmp_stride: usize,
    o: usize,
    w: usize,
    h: usize,
    ss_hor: usize,
    ss_ver: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
    bo_only: bool,
) {
    for (y, dst) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
        let row = o + (y << ss_ver) * tmp_stride;
        for (x, out) in dst[..w].iter_mut().enumerate() {
            let ti = row + (x << ss_hor);
            let c = tmp[ti] as i32;
            let band = (c as u32 >> shift) as u8;
            if bo_only {
                *out = band;
            } else {
                let cls0 = ccso_score(
                    tmp[(ti as isize + luma_offset) as usize] as i32 - c,
                    quant_step,
                    edge_clf,
                );
                let cls1 = ccso_score(
                    tmp[(ti as isize - luma_offset) as usize] as i32 - c,
                    quant_step,
                    edge_clf,
                );
                *out = ((cls0 << 5) | (cls1 << 3)) as u8 | band;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ccso_prep_lut_hbd_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u16],
    tmp_stride: usize,
    o: usize,
    w: usize,
    h: usize,
    ss_hor: usize,
    ss_ver: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
    bo_only: bool,
) {
    for (y, dst) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
        let row = o + (y << ss_ver) * tmp_stride;
        for (x, out) in dst[..w].iter_mut().enumerate() {
            let ti = row + (x << ss_hor);
            let c = tmp[ti] as i32;
            let band = (c as u32 >> shift) as u8;
            if bo_only {
                *out = band;
            } else {
                let cls0 = ccso_score(
                    tmp[(ti as isize + luma_offset) as usize] as i32 - c,
                    quant_step,
                    edge_clf,
                );
                let cls1 = ccso_score(
                    tmp[(ti as isize - luma_offset) as usize] as i32 - c,
                    quant_step,
                    edge_clf,
                );
                *out = ((cls0 << 5) | (cls1 << 3)) as u8 | band;
            }
        }
    }
}

static CCSO_PREP_8BPC: OnceLock<CcsoPrep8bpcFn> = OnceLock::new();
static CCSO_PREP_HBD: OnceLock<CcsoPrepHbdFn> = OnceLock::new();
static CCSO_ADD_8BPC: OnceLock<CcsoAdd8bpcFn> = OnceLock::new();
static CCSO_ADD_HBD: OnceLock<CcsoAddHbdFn> = OnceLock::new();

#[inline]
fn resolve_ccso_prep_8bpc() -> CcsoPrep8bpcFn {
    *CCSO_PREP_8BPC.get_or_init(|| {
        let mut _f = ccso_prep_lut_8bpc_scalar as CcsoPrep8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ccso_prep_lut_8bpc_neon as CcsoPrep8bpcFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ccso_prep_lut_8bpc_sse41 as CcsoPrep8bpcFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ccso_prep_lut_8bpc_avx2 as CcsoPrep8bpcFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_ccso_prep_hbd() -> CcsoPrepHbdFn {
    *CCSO_PREP_HBD.get_or_init(|| {
        let mut _f = ccso_prep_lut_hbd_scalar as CcsoPrepHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ccso_prep_lut_hbd_neon as CcsoPrepHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ccso_prep_lut_hbd_sse41 as CcsoPrepHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ccso_prep_lut_hbd_avx2 as CcsoPrepHbdFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_ccso_add_8bpc() -> CcsoAdd8bpcFn {
    *CCSO_ADD_8BPC.get_or_init(|| {
        let mut _f = ccso_add_8bpc_scalar as CcsoAdd8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ccso_add_8bpc_neon as CcsoAdd8bpcFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ccso_add_8bpc_sse41 as CcsoAdd8bpcFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ccso_add_8bpc_avx2 as CcsoAdd8bpcFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_ccso_add_hbd() -> CcsoAddHbdFn {
    *CCSO_ADD_HBD.get_or_init(|| {
        let mut _f = ccso_add_hbd_scalar as CcsoAddHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ccso_add_hbd_neon as CcsoAddHbdFn;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ccso_add_hbd_sse41 as CcsoAddHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ccso_add_hbd_avx2 as CcsoAddHbdFn;
            }
        }
        _f
    })
}

pub(crate) fn ccso_prep<BD: BitDepth>(bd: BD, ctx: CcsoPrepCtx<'_, BD::Pixel>) {
    let CcsoPrepCtx {
        dst,
        dst_stride,
        src,
        tmp_buf,
        cfg,
        area,
    } = ctx;
    let CcsoPrepSrc {
        src,
        stride: src_stride,
        off: src_off,
        left,
        top,
        top_off,
        bottom,
        bottom_off,
    } = src;
    let CcsoPrepCfg {
        max_band_log2,
        ext_filter,
        quant_step,
        edge_clf,
        bo_only,
    } = cfg;
    let CcsoPrepArea {
        edges,
        w,
        h,
        ss_hor,
        ss_ver,
    } = area;

    let shift = bd.bitdepth() as u32 - max_band_log2;

    if bo_only {
        if let Some(src8) = <BD::Pixel as Pixel>::try_as_u8_slice(src) {
            unsafe {
                resolve_ccso_prep_8bpc()(
                    dst, dst_stride, src8, src_stride, src_off, w, h, ss_hor, ss_ver, shift, 0,
                    quant_step, edge_clf, true,
                )
            };
            return;
        }
        if let Some(src16) = <BD::Pixel as Pixel>::try_as_u16_slice(src) {
            unsafe {
                resolve_ccso_prep_hbd()(
                    dst, dst_stride, src16, src_stride, src_off, w, h, ss_hor, ss_ver, shift, 0,
                    quant_step, edge_clf, true,
                )
            };
            return;
        }
    }

    let dy = CCSO_POS[ext_filter][0] as isize;
    let dx = CCSO_POS[ext_filter][1] as isize;
    let tmp_stride: usize = 68;
    let luma_offset = dx + dy * tmp_stride as isize;
    let tmp_need = tmp_stride * (h.max(8) * (1 << ss_ver) + 4 + 4);
    if tmp_buf.len() < tmp_need {
        tmp_buf.resize(tmp_need, BD::Pixel::default());
    }
    let tmp_buf = &mut tmp_buf[..tmp_need];
    let o = 2 * tmp_stride + 2;

    ccso_padding(
        tmp_buf,
        tmp_stride,
        o,
        src,
        src_stride,
        src_off,
        left,
        top,
        top_off,
        bottom,
        bottom_off,
        w << ss_hor,
        h << ss_ver,
        edges,
    );

    if let Some(tmp8) = <BD::Pixel as Pixel>::try_as_u8_slice(tmp_buf) {
        // SAFETY: resolver returns an 8-bit kernel only after feature detection;
        // the scalar default is always sound and has the same argument layout.
        unsafe {
            resolve_ccso_prep_8bpc()(
                dst,
                dst_stride,
                tmp8,
                tmp_stride,
                o,
                w,
                h,
                ss_hor,
                ss_ver,
                shift,
                luma_offset,
                quant_step,
                edge_clf,
                bo_only,
            )
        };
        return;
    }
    if let Some(tmp16) = <BD::Pixel as Pixel>::try_as_u16_slice(tmp_buf) {
        // SAFETY: same as the 8-bit dispatch, with native-endian u16 storage.
        unsafe {
            resolve_ccso_prep_hbd()(
                dst,
                dst_stride,
                tmp16,
                tmp_stride,
                o,
                w,
                h,
                ss_hor,
                ss_ver,
                shift,
                luma_offset,
                quant_step,
                edge_clf,
                bo_only,
            )
        };
        return;
    }

    unreachable!("unsupported CCSO pixel storage");
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ccso_add_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
) {
    let offset_map = ccso_build_offset_map(offset_idxs, offset_lut);
    let n_blocks = (h + 3) >> 2;
    let dst_block_len = dst_stride * 4 * n_blocks;
    let idx_block_len = idx_stride * 4 * n_blocks;
    for ((dst_rows, idx_rows), mask) in dst[..dst_block_len]
        .chunks_exact_mut(dst_stride * 4)
        .zip(idx_buf[..idx_block_len].chunks_exact(idx_stride * 4))
        .zip(ll_mask[..n_blocks].iter())
    {
        let row_mask = mask[0];
        for (bx, xx) in (0..w).step_by(4).enumerate() {
            if row_mask & (1 << bx) == 0 {
                for (dst_row, idx_row) in dst_rows
                    .chunks_exact_mut(dst_stride)
                    .zip(idx_rows.chunks_exact(idx_stride))
                {
                    let dst4 = &mut dst_row[xx..xx + 4].as_chunks_mut::<4>().0[0];
                    let idx4 = &idx_row[xx..xx + 4].as_chunks::<4>().0[0];
                    for (dst, &idx) in dst4.iter_mut().zip(idx4.iter()) {
                        let off = offset_map[idx as usize];
                        let cur = *dst as i32;
                        *dst = (cur + off as i32).clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ccso_add_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
    bitdepth_max: i32,
) {
    let offset_map = ccso_build_offset_map(offset_idxs, offset_lut);
    let n_blocks = (h + 3) >> 2;
    let dst_block_len = dst_stride * 4 * n_blocks;
    let idx_block_len = idx_stride * 4 * n_blocks;
    for ((dst_rows, idx_rows), mask) in dst[..dst_block_len]
        .chunks_exact_mut(dst_stride * 4)
        .zip(idx_buf[..idx_block_len].chunks_exact(idx_stride * 4))
        .zip(ll_mask[..n_blocks].iter())
    {
        let row_mask = mask[0];
        for (bx, xx) in (0..w).step_by(4).enumerate() {
            if row_mask & (1 << bx) == 0 {
                for (dst_row, idx_row) in dst_rows
                    .chunks_exact_mut(dst_stride)
                    .zip(idx_rows.chunks_exact(idx_stride))
                {
                    let dst4 = &mut dst_row[xx..xx + 4].as_chunks_mut::<4>().0[0];
                    let idx4 = &idx_row[xx..xx + 4].as_chunks::<4>().0[0];
                    for (dst, &idx) in dst4.iter_mut().zip(idx4.iter()) {
                        let off = offset_map[idx as usize];
                        let cur = *dst as i32;
                        *dst = (cur + off as i32).clamp(0, bitdepth_max) as u16;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ccso_add<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
) {
    if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
        // SAFETY: dispatch is guarded by runtime feature detection; scalar default is sound.
        unsafe {
            resolve_ccso_add_8bpc()(
                d8,
                dst_stride,
                idx_buf,
                idx_stride,
                offset_idxs,
                offset_lut,
                w,
                h,
                ll_mask,
            )
        };
        return;
    }
    if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
        // SAFETY: dispatch is guarded by runtime feature detection; scalar default is sound.
        unsafe {
            resolve_ccso_add_hbd()(
                d16,
                dst_stride,
                idx_buf,
                idx_stride,
                offset_idxs,
                offset_lut,
                w,
                h,
                ll_mask,
                bd.bitdepth_max(),
            )
        };
        return;
    }

    unreachable!("unsupported CCSO pixel storage");
}
