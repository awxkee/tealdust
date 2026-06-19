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

pub(crate) type ResidualAddFn = unsafe fn(&mut [u8], &[i32], usize, i32, i32);

pub(crate) fn residual_add_row_8bpc_scalar(
    dst: &mut [u8],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    for i in 0..n {
        let p = dst[i] as i32;
        dst[i] = (p + ((c[i] + rnd) >> shift)).clamp(0, 255) as u8;
    }
}

static RESIDUAL_ADD: OnceLock<ResidualAddFn> = OnceLock::new();

#[inline]
fn resolve_residual_add() -> ResidualAddFn {
    *RESIDUAL_ADD.get_or_init(|| {
        let mut f = residual_add_row_8bpc_scalar as ResidualAddFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::residual_add_row_8bpc_neon as ResidualAddFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::residual_add_row_8bpc_sse41 as ResidualAddFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn residual_add_row_8bpc(dst: &mut [u8], c: &[i32], n: usize, rnd: i32, shift: i32) {
    // SAFETY: `resolve_residual_add` only returns the SSE/NEON kernel when the
    // corresponding feature was detected; the scalar default is always sound.
    unsafe { resolve_residual_add()(dst, c, n, rnd, shift) };
}

pub(crate) type DcAddFn = unsafe fn(&mut [u8], i32, usize);

pub(crate) fn dc_add_row_8bpc_scalar(dst: &mut [u8], dc: i32, n: usize) {
    for d in dst[..n].iter_mut() {
        *d = (*d as i32 + dc).clamp(0, 255) as u8;
    }
}

static DC_ADD: OnceLock<DcAddFn> = OnceLock::new();

#[inline]
fn resolve_dc_add() -> DcAddFn {
    *DC_ADD.get_or_init(|| {
        let mut f = dc_add_row_8bpc_scalar as DcAddFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::dc_add_row_8bpc_neon as DcAddFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::dc_add_row_8bpc_sse41 as DcAddFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn dc_add_row_8bpc(dst: &mut [u8], dc: i32, n: usize) {
    // SAFETY: see `residual_add_row_8bpc`.
    unsafe { resolve_dc_add()(dst, dc, n) };
}

pub(crate) type RowClipFn = unsafe fn(&mut [i32], usize, i32, i32, i32, i32);

pub(crate) fn row_clip_scalar(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    for t in tmp[..n].iter_mut() {
        *t = ((*t + rnd) >> shift).max(min).min(max);
    }
}

static ROW_CLIP: OnceLock<RowClipFn> = OnceLock::new();

#[inline]
fn resolve_row_clip() -> RowClipFn {
    *ROW_CLIP.get_or_init(|| {
        let mut f = row_clip_scalar as RowClipFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::row_clip_neon as RowClipFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::row_clip_sse41 as RowClipFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn row_clip(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    // SAFETY: resolved kernel matches a detected feature; scalar default is sound.
    unsafe { resolve_row_clip()(tmp, n, rnd, shift, min, max) };
}

pub(crate) type CctxFn = unsafe fn(&mut [i32], &mut [i32], i32, i32, usize, i32, i32);

pub(crate) fn cctx_row_scalar(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    for i in 0..sz {
        let a = u[i] * cosa - v[i] * sina;
        let b = u[i] * sina + v[i] * cosa;
        u[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        v[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
    }
}

static CCTX: OnceLock<CctxFn> = OnceLock::new();

#[inline]
fn resolve_cctx() -> CctxFn {
    *CCTX.get_or_init(|| {
        let mut f = cctx_row_scalar as CctxFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cctx_row_neon as CctxFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cctx_row_sse41 as CctxFn;
            }
        }
        f
    })
}

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
    // SAFETY: see `row_clip`.
    unsafe { resolve_cctx()(u, v, sina, cosa, sz, min, max) };
}

pub(crate) type AvgFn = unsafe fn(&mut [u8], &[i16], &[i16], usize, i32, i32);

pub(crate) fn avg_row_8bpc_scalar(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    for x in 0..n {
        dst[x] = ((t1[x] as i32 + t2[x] as i32 + rnd) >> sh).clamp(0, 255) as u8;
    }
}

static AVG: OnceLock<AvgFn> = OnceLock::new();

#[inline]
fn resolve_avg() -> AvgFn {
    *AVG.get_or_init(|| {
        let mut f = avg_row_8bpc_scalar as AvgFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::avg_row_8bpc_neon as AvgFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::avg_row_8bpc_sse41 as AvgFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn avg_row_8bpc(dst: &mut [u8], t1: &[i16], t2: &[i16], n: usize, rnd: i32, sh: i32) {
    // SAFETY: resolved kernel matches a detected feature; scalar default sound.
    unsafe { resolve_avg()(dst, t1, t2, n, rnd, sh) };
}

// ---------------------------------------------------------------------------
// w_avg_row (8bpc)
// ---------------------------------------------------------------------------

pub(crate) type WAvgFn = unsafe fn(&mut [u8], &[i16], &[i16], usize, i32, i32, i32);

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_avg_row_8bpc_scalar(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    for x in 0..n {
        dst[x] = ((t1[x] as i32 * weight + t2[x] as i32 * (16 - weight) + rnd) >> sh).clamp(0, 255)
            as u8;
    }
}

static W_AVG: OnceLock<WAvgFn> = OnceLock::new();

#[inline]
fn resolve_w_avg() -> WAvgFn {
    *W_AVG.get_or_init(|| {
        let mut f = w_avg_row_8bpc_scalar as WAvgFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::w_avg_row_8bpc_neon as WAvgFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::w_avg_row_8bpc_sse41 as WAvgFn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn w_avg_row_8bpc(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    // SAFETY: see `avg_row_8bpc`.
    unsafe { resolve_w_avg()(dst, t1, t2, n, weight, rnd, sh) };
}

// ---------------------------------------------------------------------------
// mask_row (8bpc)
// ---------------------------------------------------------------------------

pub(crate) type MaskFn = unsafe fn(&mut [u8], &[i16], &[i16], &[u8], usize, i32, i32);

#[allow(clippy::too_many_arguments)]
pub(crate) fn mask_row_8bpc_scalar(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    for x in 0..n {
        let m = mask[x] as i32;
        dst[x] = ((t1[x] as i32 * m + t2[x] as i32 * (64 - m) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

static MASK: OnceLock<MaskFn> = OnceLock::new();

#[inline]
fn resolve_mask() -> MaskFn {
    *MASK.get_or_init(|| {
        let mut f = mask_row_8bpc_scalar as MaskFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::mask_row_8bpc_neon as MaskFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::mask_row_8bpc_sse41 as MaskFn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn mask_row_8bpc(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    // SAFETY: see `avg_row_8bpc`.
    unsafe { resolve_mask()(dst, t1, t2, mask, n, rnd, sh) };
}

pub(crate) type BlendFn = unsafe fn(&mut [u8], &[u8], &[u8], usize);

pub(crate) fn blend_row_8bpc_scalar(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    for x in 0..n {
        let m = mask[x] as i32;
        let d = dst[x] as i32;
        let t = tmp[x] as i32;
        dst[x] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
    }
}

static BLEND: OnceLock<BlendFn> = OnceLock::new();

#[inline]
fn resolve_blend() -> BlendFn {
    *BLEND.get_or_init(|| {
        let mut f = blend_row_8bpc_scalar as BlendFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::blend_row_8bpc_neon as BlendFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::blend_row_8bpc_sse41 as BlendFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn blend_row_8bpc(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    // SAFETY: see `avg_row_8bpc`.
    unsafe { resolve_blend()(dst, tmp, mask, n) };
}

pub(crate) type MorphFn = unsafe fn(&mut [u8], i32, i32, usize);

pub(crate) fn morph_row_8bpc_scalar(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    for x in 0..n {
        dst[x] = ((alpha * dst[x] as i32 + beta) >> 8).clamp(0, 255) as u8;
    }
}

static MORPH: OnceLock<MorphFn> = OnceLock::new();

#[inline]
fn resolve_morph() -> MorphFn {
    *MORPH.get_or_init(|| {
        let mut f = morph_row_8bpc_scalar as MorphFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::morph_row_8bpc_neon as MorphFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::morph_row_8bpc_sse41 as MorphFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn morph_row_8bpc(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    // SAFETY: see `avg_row_8bpc`.
    unsafe { resolve_morph()(dst, alpha, beta, n) };
}

pub(crate) type GdfAddFn = unsafe fn(&mut [u8], &[i8], i32, usize);

pub(crate) fn gdf_add_run_8bpc_scalar(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    for x in 0..n {
        let diff = err[x] as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let adj = if diff < 0 { -mag } else { mag };
        dst[x] = (dst[x] as i32 + adj).clamp(0, 255) as u8;
    }
}

static GDF_ADD: OnceLock<GdfAddFn> = OnceLock::new();

#[inline]
fn resolve_gdf_add() -> GdfAddFn {
    *GDF_ADD.get_or_init(|| {
        let mut f = gdf_add_run_8bpc_scalar as GdfAddFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::gdf_add_run_8bpc_neon as GdfAddFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::gdf_add_run_8bpc_sse41 as GdfAddFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn gdf_add_run_8bpc(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    // SAFETY: see `avg_row_8bpc`.
    unsafe { resolve_gdf_add()(dst, err, scale, n) };
}

pub(crate) type GdfGradFn = unsafe fn(
    &mut [[u16; 4]],
    usize,
    usize,
    usize,
    [&[u8]; 2],
    [&[u8]; 2],
    [&[u8]; 2],
    usize,
    i32,
    u32,
);

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdf_gradient_group_scalar(
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
    let mut acc = [0i32; 8];
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let brow: &[u8; 8] = center_rows[y][bcol..bcol + 8].try_into().unwrap();
        let arow: &[u8; 8] = a_rows[y][acol..acol + 8].try_into().unwrap();
        let crow: &[u8; 8] = c_rows[y][ccol..ccol + 8].try_into().unwrap();
        for k in 0..8 {
            let b = (brow[k] as i32) >> shift;
            let a = (arow[k] as i32) >> shift;
            let c = (crow[k] as i32) >> shift;
            acc[k] += (b + b - a - c).abs();
        }
    }
    for k in 0..ncells {
        dst[base_cell + k][d] = (acc[2 * k] + acc[2 * k + 1]) as u16;
    }
}

static GDF_GRAD: OnceLock<GdfGradFn> = OnceLock::new();

#[inline]
fn resolve_gdf_grad() -> GdfGradFn {
    *GDF_GRAD.get_or_init(|| {
        let mut f = gdf_gradient_group_scalar as GdfGradFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::gdf_gradient_group_neon as GdfGradFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::gdf_gradient_group_sse41 as GdfGradFn;
            }
        }
        f
    })
}

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
    // SAFETY: see `avg_row_8bpc`.
    unsafe {
        resolve_gdf_grad()(
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
        )
    };
}
