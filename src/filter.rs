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

use crate::pixel::{BitDepth, Pixel};

/// `avg` row: `dst[x] = clip((tmp1[x] + tmp2[x] + rnd) >> sh)` for `x in 0..n`.
#[inline]
pub(crate) fn avg_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::rowops_dispatch::avg_row_8bpc(d8, tmp1, tmp2, n, rnd, sh);
            return;
        }
    } else if BD::BPC == 16 {
        if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
            crate::rowops_dispatch::avg_row_hbd(d16, tmp1, tmp2, n, rnd, sh, bd.bitdepth_max());
            return;
        }
    }
    let (dc, dr) = dst[..n].as_chunks_mut::<8>();
    let (t1c, t1r) = tmp1[..n].as_chunks::<8>();
    let (t2c, t2r) = tmp2[..n].as_chunks::<8>();
    for ((d, a), b) in dc.iter_mut().zip(t1c).zip(t2c) {
        for k in 0..8 {
            d[k] = bd.pixel_clip((a[k] as i32 + b[k] as i32 + rnd) >> sh);
        }
    }
    for ((d, &a), &b) in dr.iter_mut().zip(t1r).zip(t2r) {
        *d = bd.pixel_clip((a as i32 + b as i32 + rnd) >> sh);
    }
}

/// `w_avg` row: `dst[x] = clip((tmp1[x]*weight + tmp2[x]*(16-weight) + rnd) >> sh)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn w_avg_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::rowops_dispatch::w_avg_row_8bpc(d8, tmp1, tmp2, n, weight, rnd, sh);
            return;
        }
    } else if BD::BPC == 16 {
        if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
            crate::rowops_dispatch::w_avg_row_hbd(
                d16,
                tmp1,
                tmp2,
                n,
                weight,
                rnd,
                sh,
                bd.bitdepth_max(),
            );
            return;
        }
    }
    let (dc, dr) = dst[..n].as_chunks_mut::<8>();
    let (t1c, t1r) = tmp1[..n].as_chunks::<8>();
    let (t2c, t2r) = tmp2[..n].as_chunks::<8>();
    for ((d, a), b) in dc.iter_mut().zip(t1c).zip(t2c) {
        for k in 0..8 {
            d[k] = bd.pixel_clip((a[k] as i32 * weight + b[k] as i32 * (16 - weight) + rnd) >> sh);
        }
    }
    for ((d, &a), &b) in dr.iter_mut().zip(t1r).zip(t2r) {
        *d = bd.pixel_clip((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh);
    }
}

/// `mask` row: `dst[x] = clip((tmp1[x]*m + tmp2[x]*(64-m) + rnd) >> sh)`, `m = mask[x]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn mask_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::rowops_dispatch::mask_row_8bpc(d8, tmp1, tmp2, mask, n, rnd, sh);
            return;
        }
    } else if BD::BPC == 16 {
        if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
            crate::rowops_dispatch::mask_row_hbd(
                d16,
                tmp1,
                tmp2,
                mask,
                n,
                rnd,
                sh,
                bd.bitdepth_max(),
            );
            return;
        }
    }
    let (dc, dr) = dst[..n].as_chunks_mut::<8>();
    let (t1c, t1r) = tmp1[..n].as_chunks::<8>();
    let (t2c, t2r) = tmp2[..n].as_chunks::<8>();
    let (mc, mr) = mask[..n].as_chunks::<8>();
    for (((d, a), b), m) in dc.iter_mut().zip(t1c).zip(t2c).zip(mc) {
        for k in 0..8 {
            let mk = m[k] as i32;
            d[k] = bd.pixel_clip((a[k] as i32 * mk + b[k] as i32 * (64 - mk) + rnd) >> sh);
        }
    }
    for (((d, &a), &b), &m) in dr.iter_mut().zip(t1r).zip(t2r).zip(mr) {
        let mk = m as i32;
        *d = bd.pixel_clip((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh);
    }
}

/// `blend` row: `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6` (truncate, no clamp).
#[inline]
pub(crate) fn blend_row<P: Pixel>(dst: &mut [P], tmp: &[P], mask: &[u8], n: usize) {
    if let (Some(t8), Some(d8)) = (P::try_as_u8_slice(tmp), P::try_as_u8_slice_mut(dst)) {
        crate::rowops_dispatch::blend_row_8bpc(d8, t8, mask, n);
        return;
    }
    if let (Some(t16), Some(d16)) = (P::try_as_u16_slice(tmp), P::try_as_u16_slice_mut(dst)) {
        crate::rowops_dispatch::blend_row_hbd(d16, t16, mask, n);
        return;
    }
    let (dc, dr) = dst[..n].as_chunks_mut::<8>();
    let (tc, tr) = tmp[..n].as_chunks::<8>();
    let (mc, mr) = mask[..n].as_chunks::<8>();
    for ((d, t), m) in dc.iter_mut().zip(tc).zip(mc) {
        for k in 0..8 {
            let mk = m[k] as i32;
            let dv: i32 = d[k].into();
            let tv: i32 = t[k].into();
            d[k] = P::from_i32((dv * (64 - mk) + tv * mk + 32) >> 6);
        }
    }
    for ((d, &t), &m) in dr.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        let dv: i32 = (*d).into();
        let tv: i32 = t.into();
        *d = P::from_i32((dv * (64 - mk) + tv * mk + 32) >> 6);
    }
}

/// `morph` row: `dst[x] = clip((alpha*dst[x] + beta) >> 8)`.
pub(crate) fn morph_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    alpha: i32,
    beta: i32,
    n: usize,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::rowops_dispatch::morph_row_8bpc(d8, alpha, beta, n);
            return;
        }
    } else if BD::BPC == 16 {
        if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
            crate::rowops_dispatch::morph_row_hbd(d16, alpha, beta, n, bd.bitdepth_max());
            return;
        }
    }
    let (dc, dr) = dst[..n].as_chunks_mut::<8>();
    for d in dc.iter_mut() {
        for k in 0..8 {
            let dv: i32 = d[k].into();
            d[k] = bd.pixel_clip((alpha * dv + beta) >> 8);
        }
    }
    for d in dr.iter_mut() {
        let dv: i32 = (*d).into();
        *d = bd.pixel_clip((alpha * dv + beta) >> 8);
    }
}

/// itx DC-only row: `dst[x] = clip(dst[x] + dc)` for `x in 0..n`.
#[inline]
pub(crate) fn dc_add_row<BD: BitDepth>(bd: BD, dst: &mut [BD::Pixel], dc: i32, n: usize) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::rowops_dispatch::dc_add_row_8bpc(d8, dc, n);
            return;
        }
    } else if BD::BPC == 16 {
        if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
            crate::rowops_dispatch::dc_add_row_hbd(d16, dc, n, bd.bitdepth_max());
            return;
        }
    }

    let (dchunks, dr) = dst[..n].as_chunks_mut::<8>();
    for d in dchunks.iter_mut() {
        for k in 0..8 {
            let p: i32 = d[k].into();
            d[k] = bd.pixel_clip(p + dc);
        }
    }
    for d in dr.iter_mut() {
        let p: i32 = (*d).into();
        *d = bd.pixel_clip(p + dc);
    }
}

/// itx row-clip pass: `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)` in place.
#[inline]
pub(crate) fn row_clip(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    crate::rowops_dispatch::row_clip(tmp, n, rnd, shift, min, max);
}

/// itx plain residual-add row: `dst[x] = clip(dst[x] + ((c[x]+rnd)>>shift))`.
#[inline]
pub(crate) fn residual_add_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    // 8-bit fast path: OnceLock-dispatched real SIMD (NEON/SSE) over u8.
    // BPC==8 ⇒ Pixel == u8, so the reinterpret is a no-op `Some`.
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::rowops_dispatch::residual_add_row_8bpc(d8, c, n, rnd, shift);
            return;
        }
    } else if BD::BPC == 16 {
        if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
            crate::rowops_dispatch::residual_add_row_hbd(d16, c, n, rnd, shift, bd.bitdepth_max());
            return;
        }
    }

    let (dc, dr) = dst[..n].as_chunks_mut::<8>();
    let (cc, cr) = c[..n].as_chunks::<8>();
    for (d, cv) in dc.iter_mut().zip(cc) {
        for k in 0..8 {
            let p: i32 = d[k].into();
            d[k] = bd.pixel_clip(p + ((cv[k] + rnd) >> shift));
        }
    }
    for (d, &cv) in dr.iter_mut().zip(cr) {
        let p: i32 = (*d).into();
        *d = bd.pixel_clip(p + ((cv + rnd) >> shift));
    }
}

/// `cctx` row: cross-component-transform rotate + clip over two i32 planes.
/// `u'[i] = iclip((u*cosa - v*sina + 128 - (a<0)) >> 8, min, max)`,
/// `v'[i] = iclip((u*sina + v*cosa + 128 - (b<0)) >> 8, min, max)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cctx_row(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    crate::rowops_dispatch::cctx_row(u, v, sina, cosa, sz, min, max);
}

/// One symmetric FIR tap: `a` is read from `row_p` at `+dx`, `b` from `row_m`
/// at `-dx` (relative to the per-pixel column `o + x`).
pub(crate) struct WienerTap<'a> {
    pub row_p: &'a [u8],
    pub row_m: &'a [u8],
    pub dx: i32,
    pub coef: i32,
}

type NsWienerFirFn = unsafe fn(&mut [u8], &[u8], usize, &[WienerTap<'_>], usize);
type PcWienerFirFn = unsafe fn(&mut [u8], &[u8], i32, usize, &[WienerTap<'_>], usize);

static NS_WIENER_FIR: std::sync::OnceLock<NsWienerFirFn> = std::sync::OnceLock::new();
static PC_WIENER_FIR: std::sync::OnceLock<PcWienerFirFn> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn ns_wiener_fir_run() -> NsWienerFirFn {
    *NS_WIENER_FIR.get_or_init(|| {
        let mut _f: NsWienerFirFn = ns_wiener_fir_run_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ns_wiener_fir_run_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ns_wiener_fir_run_sse41;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ns_wiener_fir_run_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn pc_wiener_fir_run() -> PcWienerFirFn {
    *PC_WIENER_FIR.get_or_init(|| {
        let mut _f: PcWienerFirFn = pc_wiener_fir_run_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::pc_wiener_fir_run_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::pc_wiener_fir_run_sse41;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::pc_wiener_fir_run_avx2;
            }
        }
        _f
    })
}

/// Pure-scalar "NS" Wiener FIR — the reference implementation and the fallback
/// for targets without a hand-written SIMD kernel.
pub(crate) fn ns_wiener_fir_run_scalar(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    for x in 0..n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m << 7;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b - 2 * m) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
    }
}

pub(crate) struct UvLumaTap<'a> {
    pub(crate) row: &'a [u8],
    pub(crate) ldx: i32,
    pub(crate) coef: i32,
}

type NsWienerUvFirFn = unsafe fn(
    &mut [u8],
    &[u8],
    usize,
    &[WienerTap<'_>],
    &[u8],
    usize,
    &[UvLumaTap<'_>],
    usize,
    usize,
);

static NS_WIENER_UV_FIR: std::sync::OnceLock<NsWienerUvFirFn> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn ns_wiener_uv_fir_run() -> NsWienerUvFirFn {
    *NS_WIENER_UV_FIR.get_or_init(|| {
        let mut _f: NsWienerUvFirFn = ns_wiener_uv_fir_run_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ns_wiener_uv_fir_run_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ns_wiener_uv_fir_run_sse41;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ns_wiener_uv_fir_run_avx2;
            }
        }
        _f
    })
}

/// Pure-scalar chroma NS-Wiener FIR: 6 symmetric chroma taps (center = chroma
/// `m`) plus 12 asymmetric luma cross-taps (center = luma `lc`, subsampled by
/// `lstep = 1 << ss_hor`). Reference and fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ns_wiener_uv_fir_run_scalar(
    dst: &mut [u8],
    c_center: &[u8],
    co: usize,
    ctaps: &[WienerTap],
    l_center: &[u8],
    lo: usize,
    ltaps: &[UvLumaTap],
    lstep: usize,
    n: usize,
) {
    for x in 0..n {
        let cc = co + x;
        let m = c_center[cc] as i32;
        let mut s = m << 7;
        for t in ctaps {
            let a = t.row_p[(cc as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(cc as i32 - t.dx) as usize] as i32;
            s += (a + b - 2 * m) * t.coef;
        }
        let lcx = lo + x * lstep;
        let lc = l_center[lcx] as i32;
        for t in ltaps {
            let lv = t.row[(lcx as i32 + t.ldx) as usize] as i32;
            s += (lv - lc) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
    }
}

/// Pure-scalar "PC" Wiener FIR.
pub(crate) fn pc_wiener_fir_run_scalar(
    dst: &mut [u8],
    center: &[u8],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    for x in 0..n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m * center_coef;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
    }
}

/// High-bit-depth symmetric Wiener FIR tap. Layout mirrors [`WienerTap`], but
/// samples are native `u16` pixels and the output clamp is supplied by the
/// caller (`1023` for 10bpc, `4095` for 12bpc).
pub(crate) struct WienerTapHbd<'a> {
    pub row_p: &'a [u16],
    pub row_m: &'a [u16],
    pub dx: i32,
    pub coef: i32,
}

type NsWienerFirHbdFn = unsafe fn(&mut [u16], &[u16], usize, &[WienerTapHbd<'_>], usize, i32);
type PcWienerFirHbdFn = unsafe fn(&mut [u16], &[u16], i32, usize, &[WienerTapHbd<'_>], usize, i32);

static NS_WIENER_FIR_HBD: std::sync::OnceLock<NsWienerFirHbdFn> = std::sync::OnceLock::new();
static PC_WIENER_FIR_HBD: std::sync::OnceLock<PcWienerFirHbdFn> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn ns_wiener_fir_run_hbd() -> NsWienerFirHbdFn {
    *NS_WIENER_FIR_HBD.get_or_init(|| {
        let mut _f: NsWienerFirHbdFn = ns_wiener_fir_run_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ns_wiener_fir_run_hbd_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ns_wiener_fir_run_hbd_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ns_wiener_fir_run_hbd_avx2;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn pc_wiener_fir_run_hbd() -> PcWienerFirHbdFn {
    *PC_WIENER_FIR_HBD.get_or_init(|| {
        let mut _f: PcWienerFirHbdFn = pc_wiener_fir_run_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::pc_wiener_fir_run_hbd_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::pc_wiener_fir_run_hbd_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::pc_wiener_fir_run_hbd_avx2;
            }
        }
        _f
    })
}

pub(crate) fn ns_wiener_fir_run_hbd_scalar(
    dst: &mut [u16],
    center: &[u16],
    col0: usize,
    taps: &[WienerTapHbd],
    n: usize,
    bitdepth_max: i32,
) {
    for x in 0..n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m << 7;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b - 2 * m) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
    }
}

pub(crate) fn pc_wiener_fir_run_hbd_scalar(
    dst: &mut [u16],
    center: &[u16],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTapHbd],
    n: usize,
    bitdepth_max: i32,
) {
    for x in 0..n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m * center_coef;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
    }
}

pub(crate) struct UvLumaTapHbd<'a> {
    pub(crate) row: &'a [u16],
    pub(crate) ldx: i32,
    pub(crate) coef: i32,
}

type NsWienerUvFirHbdFn = unsafe fn(
    &mut [u16],
    &[u16],
    usize,
    &[WienerTapHbd<'_>],
    &[u16],
    usize,
    &[UvLumaTapHbd<'_>],
    usize,
    usize,
    i32,
);

static NS_WIENER_UV_FIR_HBD: std::sync::OnceLock<NsWienerUvFirHbdFn> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn ns_wiener_uv_fir_run_hbd() -> NsWienerUvFirHbdFn {
    *NS_WIENER_UV_FIR_HBD.get_or_init(|| {
        let mut _f: NsWienerUvFirHbdFn = ns_wiener_uv_fir_run_hbd_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::ns_wiener_uv_fir_run_hbd_neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::ns_wiener_uv_fir_run_hbd_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::ns_wiener_uv_fir_run_hbd_avx2;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ns_wiener_uv_fir_run_hbd_scalar(
    dst: &mut [u16],
    c_center: &[u16],
    co: usize,
    ctaps: &[WienerTapHbd],
    l_center: &[u16],
    lo: usize,
    ltaps: &[UvLumaTapHbd],
    lstep: usize,
    n: usize,
    bitdepth_max: i32,
) {
    for x in 0..n {
        let cc = co + x;
        let m = c_center[cc] as i32;
        let mut s = m << 7;
        for t in ctaps {
            let a = t.row_p[(cc as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(cc as i32 - t.dx) as usize] as i32;
            s += (a + b - 2 * m) * t.coef;
        }
        let lcx = lo + x * lstep;
        let lc = l_center[lcx] as i32;
        for t in ltaps {
            let lv = t.row[(lcx as i32 + t.ldx) as usize] as i32;
            s += (lv - lc) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
    }
}

/// GDF residual add over a run of `n` consecutive pixels.
#[inline]
pub(crate) fn gdf_add_run(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    crate::rowops_dispatch::gdf_add_run_8bpc(dst, err, scale, n);
}

/// GDF gradient: accumulate per-column gradient into 8 lanes, then pair-reduce.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn gdf_gradient_group(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    center_rows: [&[u8]; 2],
    a_rows: [&[u8]; 2],
    c_rows: [&[u8]; 2],
    col0: usize,
    dx: i32,
    shift: u32,
) {
    crate::rowops_dispatch::gdf_gradient_group(
        dst,
        d,
        base_cell,
        ncells,
        center_rows,
        a_rows,
        c_rows,
        col0,
        dx,
        shift,
    );
}

#[cfg(test)]
mod wiener_scalar_proof {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn u8(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn buf(rng: &mut Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.u8()).collect()
    }

    #[test]
    fn ns_wiener_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15A);
        let f = ns_wiener_fir_run();
        for _ in 0..400 {
            let len = 256usize;
            let center = buf(&mut rng, len);
            let n_taps = rng.range(1, 8) as usize;
            let rows: Vec<(Vec<u8>, Vec<u8>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        buf(&mut rng, len),
                        buf(&mut rng, len),
                        rng.range(1, 16),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let taps: Vec<WienerTap> = rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTap {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let col0 = 64usize;
            let n = rng.range(1, 100) as usize;
            let mut d_ref = vec![0u8; n];
            let mut d_dsp = vec![0u8; n];
            ns_wiener_fir_run_scalar(&mut d_ref, &center, col0, &taps, n);
            // SAFETY: resolver selected this function after runtime CPU feature detection.
            unsafe { f(&mut d_dsp, &center, col0, &taps, n) };
            assert_eq!(d_ref, d_dsp, "ns dispatch mismatch n={} taps={}", n, n_taps);
        }
    }

    #[test]
    fn pc_wiener_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15B);
        let f = pc_wiener_fir_run();
        for _ in 0..400 {
            let len = 256usize;
            let center = buf(&mut rng, len);
            let center_coef = rng.range(-128, 128);
            let n_taps = rng.range(1, 6) as usize;
            let rows: Vec<(Vec<u8>, Vec<u8>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        buf(&mut rng, len),
                        buf(&mut rng, len),
                        rng.range(1, 16),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let taps: Vec<WienerTap> = rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTap {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let col0 = 64usize;
            let n = rng.range(1, 100) as usize;
            let mut d_ref = vec![0u8; n];
            let mut d_dsp = vec![0u8; n];
            pc_wiener_fir_run_scalar(&mut d_ref, &center, center_coef, col0, &taps, n);
            // SAFETY: resolver selected this function after runtime CPU feature detection.
            unsafe { f(&mut d_dsp, &center, center_coef, col0, &taps, n) };
            assert_eq!(d_ref, d_dsp, "pc dispatch mismatch n={} taps={}", n, n_taps);
        }
    }
    fn u16_buf(rng: &mut Rng, len: usize, max: u16) -> Vec<u16> {
        (0..len).map(|_| rng.range(0, max as i32) as u16).collect()
    }

    #[test]
    fn ns_wiener_uv_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15C);
        let f = ns_wiener_uv_fir_run();
        for _ in 0..400 {
            let len = 320usize;
            let c_center = buf(&mut rng, len);
            let l_center = buf(&mut rng, len);
            let lstep = if rng.range(0, 1) == 0 { 1usize } else { 2usize };
            let n_taps = rng.range(1, 8) as usize;
            let c_rows: Vec<(Vec<u8>, Vec<u8>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        buf(&mut rng, len),
                        buf(&mut rng, len),
                        rng.range(-4, 4),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let ctaps: Vec<WienerTap> = c_rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTap {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let n_luma_taps = rng.range(1, 12) as usize;
            let l_rows: Vec<(Vec<u8>, i32, i32)> = (0..n_luma_taps)
                .map(|_| {
                    (
                        buf(&mut rng, len),
                        rng.range(-4, 4) * lstep as i32,
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let ltaps: Vec<UvLumaTap> = l_rows
                .iter()
                .map(|(row, ldx, coef)| UvLumaTap {
                    row,
                    ldx: *ldx,
                    coef: *coef,
                })
                .collect();
            let co = 64usize;
            let lo = 32usize;
            let n = rng.range(1, 96) as usize;
            let mut d_ref = vec![0u8; n];
            let mut d_dsp = vec![0u8; n];
            ns_wiener_uv_fir_run_scalar(
                &mut d_ref, &c_center, co, &ctaps, &l_center, lo, &ltaps, lstep, n,
            );
            // SAFETY: resolver selected this function after runtime CPU feature detection.
            unsafe {
                f(
                    &mut d_dsp, &c_center, co, &ctaps, &l_center, lo, &ltaps, lstep, n,
                )
            };
            assert_eq!(
                d_ref, d_dsp,
                "uv dispatch mismatch n={} ctaps={} ltaps={} lstep={}",
                n, n_taps, n_luma_taps, lstep
            );
        }
    }

    #[test]
    fn ns_wiener_hbd_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15D);
        let f = ns_wiener_fir_run_hbd();
        for _ in 0..400 {
            let max = if rng.range(0, 1) == 0 {
                1023u16
            } else {
                4095u16
            };
            let len = 320usize;
            let center = u16_buf(&mut rng, len, max);
            let n_taps = rng.range(1, 8) as usize;
            let rows: Vec<(Vec<u16>, Vec<u16>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        u16_buf(&mut rng, len, max),
                        u16_buf(&mut rng, len, max),
                        rng.range(-4, 4),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let taps: Vec<WienerTapHbd> = rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTapHbd {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let col0 = 64usize;
            let n = rng.range(1, 128) as usize;
            let mut d_ref = vec![0u16; n];
            let mut d_dsp = vec![0u16; n];
            ns_wiener_fir_run_hbd_scalar(&mut d_ref, &center, col0, &taps, n, max as i32);
            // SAFETY: resolver selected this function after runtime CPU feature detection.
            unsafe { f(&mut d_dsp, &center, col0, &taps, n, max as i32) };
            assert_eq!(
                d_ref, d_dsp,
                "hbd ns dispatch mismatch n={} taps={}",
                n, n_taps
            );
        }
    }

    #[test]
    fn pc_wiener_hbd_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15E);
        let f = pc_wiener_fir_run_hbd();
        for _ in 0..400 {
            let max = if rng.range(0, 1) == 0 {
                1023u16
            } else {
                4095u16
            };
            let len = 320usize;
            let center = u16_buf(&mut rng, len, max);
            let center_coef = rng.range(-256, 256);
            let n_taps = rng.range(1, 12) as usize;
            let rows: Vec<(Vec<u16>, Vec<u16>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        u16_buf(&mut rng, len, max),
                        u16_buf(&mut rng, len, max),
                        rng.range(-4, 4),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let taps: Vec<WienerTapHbd> = rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTapHbd {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let col0 = 64usize;
            let n = rng.range(1, 128) as usize;
            let mut d_ref = vec![0u16; n];
            let mut d_dsp = vec![0u16; n];
            pc_wiener_fir_run_hbd_scalar(
                &mut d_ref,
                &center,
                center_coef,
                col0,
                &taps,
                n,
                max as i32,
            );
            // SAFETY: resolver selected this function after runtime CPU feature detection.
            unsafe { f(&mut d_dsp, &center, center_coef, col0, &taps, n, max as i32) };
            assert_eq!(
                d_ref, d_dsp,
                "hbd pc dispatch mismatch n={} taps={}",
                n, n_taps
            );
        }
    }

    #[test]
    fn ns_wiener_uv_hbd_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15F);
        let f = ns_wiener_uv_fir_run_hbd();
        for _ in 0..400 {
            let max = if rng.range(0, 1) == 0 {
                1023u16
            } else {
                4095u16
            };
            let len = 360usize;
            let c_center = u16_buf(&mut rng, len, max);
            let l_center = u16_buf(&mut rng, len, max);
            let lstep = if rng.range(0, 1) == 0 { 1usize } else { 2usize };
            let n_taps = rng.range(1, 8) as usize;
            let c_rows: Vec<(Vec<u16>, Vec<u16>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        u16_buf(&mut rng, len, max),
                        u16_buf(&mut rng, len, max),
                        rng.range(-4, 4),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let ctaps: Vec<WienerTapHbd> = c_rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTapHbd {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let n_luma_taps = rng.range(1, 12) as usize;
            let l_rows: Vec<(Vec<u16>, i32, i32)> = (0..n_luma_taps)
                .map(|_| {
                    (
                        u16_buf(&mut rng, len, max),
                        rng.range(-4, 4) * lstep as i32,
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let ltaps: Vec<UvLumaTapHbd> = l_rows
                .iter()
                .map(|(row, ldx, coef)| UvLumaTapHbd {
                    row,
                    ldx: *ldx,
                    coef: *coef,
                })
                .collect();
            let co = 64usize;
            let lo = 32usize;
            let n = rng.range(1, 128) as usize;
            let mut d_ref = vec![0u16; n];
            let mut d_dsp = vec![0u16; n];
            ns_wiener_uv_fir_run_hbd_scalar(
                &mut d_ref, &c_center, co, &ctaps, &l_center, lo, &ltaps, lstep, n, max as i32,
            );
            // SAFETY: resolver selected this function after runtime CPU feature detection.
            unsafe {
                f(
                    &mut d_dsp, &c_center, co, &ctaps, &l_center, lo, &ltaps, lstep, n, max as i32,
                )
            };
            assert_eq!(
                d_ref, d_dsp,
                "hbd uv dispatch mismatch n={} ctaps={} ltaps={} lstep={}",
                n, n_taps, n_luma_taps, lstep
            );
        }
    }
}
