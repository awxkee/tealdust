/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! SSE4.1 kernels for the CfL / MHCCP stages that the 4-lane path did not
//! cover: filtered 4:2:0 apply (VSTRIP / GAUSS), luma downsampling for the
//! MHCCP reference, the MHCCP matrix / alpha accumulation and prediction.
//! Every kernel works on 4 x i32 lanes and is bit-exact with the scalar
//! reference in `cfl_dispatch`; partial groups use the scalar per-pixel code.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cfl_dispatch::{
    CflAlphaAccum8, CflAlphaAccumHbd, CflApply8, CflApplyHbd, CflGenMat8, CflGenMatHbd,
    CflGenYRow8, CflGenYRowHbd, CflMhccpPred8, CflMhccpPredHbd,
};
use crate::levels::CflMhDir;

const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;
const CFL_MHCCP_HAS_TOP: i32 = 1 << 2;
const CFL_MHCCP_HAS_LEFT: i32 = 1 << 3;

trait Px: Copy + Into<i32> {
    /// Eight consecutive samples widened to u16 lanes.
    fn load8(s: &[Self]) -> __m128i;
    fn from_i32(v: i32) -> Self;
}

impl Px for u8 {
    #[inline(always)]
    fn load8(s: &[u8]) -> __m128i {
        let s = &s[..8];
        unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(s.as_ptr().cast())) }
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
}

impl Px for u16 {
    #[inline(always)]
    fn load8(s: &[u16]) -> __m128i {
        let s = &s[..8];
        unsafe { _mm_loadu_si128(s.as_ptr().cast()) }
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
}

/// Even and odd samples of `s[off..off + 8]` as i32 lanes.
#[inline]
#[target_feature(enable = "sse4.1")]
fn even_odd<P: Px>(s: &[P], off: usize) -> (__m128i, __m128i) {
    let v = P::load8(&s[off..]);
    (
        _mm_and_si128(v, _mm_set1_epi32(0xffff)),
        _mm_srli_epi32::<16>(v),
    )
}

/// Left taps for outputs at even positions `c, c+2, c+4, c+6`, given the odd
/// samples `r` (`c+1, c+3, ..`): lane 0 is `first`, lane k is `r[k - 1]`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn left_taps(r: __m128i, first: i32) -> __m128i {
    _mm_insert_epi32::<0>(_mm_slli_si128::<4>(r), first)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store4<P: Px>(dst: &mut [P], v: __m128i) {
    let mut lanes = [0i32; 4];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), v) };
    for (d, &l) in dst[..4].iter_mut().zip(&lanes) {
        *d = P::from_i32(l);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum(v: __m128i) -> i32 {
    let s = _mm_add_epi32(v, _mm_shuffle_epi32::<0b01_00_11_10>(v));
    _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s)))
}

// --- filtered 4:2:0 apply ------------------------------------------------

#[inline(always)]
fn predict_scalar(dc: i32, alpha: i32, ac: i32, max: i32) -> i32 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    (dc + if diff < 0 { -mag } else { mag }).clamp(0, max)
}

/// `clamp(dc + round(alpha * ac / 2048), 0, max)` with the sign applied to
/// the rounded magnitude, as `predict_one`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn predict4(dc: __m128i, alpha: __m128i, ac: __m128i, max: __m128i) -> __m128i {
    let diff = _mm_mullo_epi32(alpha, ac);
    let mag = _mm_srai_epi32::<11>(_mm_add_epi32(_mm_abs_epi32(diff), _mm_set1_epi32(1024)));
    let v = _mm_add_epi32(dc, _mm_sign_epi32(mag, diff));
    _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), max)
}

#[inline(always)]
fn ac_420_scalar<P: Px>(
    y: &[P],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: i32,
    filter: u32,
) -> i32 {
    let px = |i: usize| -> i32 { y[i].into() };
    let xl = x << 1;
    let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
    if filter == CFL_FLT_TYPE_GAUSS {
        let top = if (cy & 31) == 0 {
            yrow + xl
        } else {
            yrow + xl - ystride
        };
        px(yrow + left) + 4 * px(yrow + xl) + px(yrow + xl + 1) + px(top) + px(yrow + xl + ystride)
            - dc0
    } else {
        px(yrow + left)
            + 2 * px(yrow + xl)
            + px(yrow + xl + 1)
            + px(yrow + left + ystride)
            + 2 * px(yrow + xl + ystride)
            + px(yrow + xl + ystride + 1)
            - dc0
    }
}

/// AC terms for chroma columns `x..x + 4` of a filtered 4:2:0 row.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn ac_420_filtered4<P: Px>(
    y: &[P],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: __m128i,
    filter: u32,
) -> __m128i {
    let xl = x << 1;
    // The left tap clamps at each 64-luma-column boundary.
    let first_left = |row: usize| -> i32 {
        if xl & 63 == 0 {
            y[row + xl].into()
        } else {
            y[row + xl - 1].into()
        }
    };
    let (c, r) = even_odd(y, yrow + xl);
    let l = left_taps(r, first_left(yrow));
    let (cb, rb) = even_odd(y, yrow + xl + ystride);
    let sum = if filter == CFL_FLT_TYPE_GAUSS {
        let top_row = if (cy & 31) == 0 { yrow } else { yrow - ystride };
        let (t, _) = even_odd(y, top_row + xl);
        _mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(l, _mm_slli_epi32::<2>(c)),
                _mm_add_epi32(r, t),
            ),
            cb,
        )
    } else {
        let lb = left_taps(rb, first_left(yrow + ystride));
        _mm_add_epi32(
            _mm_add_epi32(_mm_add_epi32(l, _mm_slli_epi32::<1>(c)), r),
            _mm_add_epi32(_mm_add_epi32(lb, _mm_slli_epi32::<1>(cb)), rb),
        )
    };
    _mm_sub_epi32(sum, dc0)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply_420_filtered<P: Px>(
    y: &[P],
    u: &mut [P],
    v: &mut [P],
    layout: crate::cfl_dispatch::CflLayout,
    area: crate::cfl_dispatch::CflArea,
    params: crate::cfl_dispatch::CflParams,
    max: i32,
) {
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let p = params;
    let do_u = p.alpha0 != 0;
    let do_v = p.alpha1 != 0;
    if !do_u && !do_v {
        return;
    }
    let (dc0v, maxv) = (_mm_set1_epi32(p.dc0), _mm_set1_epi32(max));
    let (dc1v, dc2v) = (_mm_set1_epi32(p.dc1), _mm_set1_epi32(p.dc2));
    let (a0v, a1v) = (_mm_set1_epi32(p.alpha0), _mm_set1_epi32(p.alpha1));

    let (mut yrow, mut urow, mut vrow) = (yrow0, urow0, vrow0);
    for cy in 0..ylim {
        let mut x = 0usize;
        // Full 8-sample loads must stay inside the luma slice.
        let fits = |x: usize| yrow + ystride + (x << 1) + 8 <= y.len();
        while x + 4 <= xlim && fits(x) {
            let ac = ac_420_filtered4(y, yrow, ystride, cy, x, dc0v, p.filter_type);
            if do_u {
                store4(&mut u[urow + x..], predict4(dc1v, a0v, ac, maxv));
            }
            if do_v {
                store4(&mut v[vrow + x..], predict4(dc2v, a1v, ac, maxv));
            }
            x += 4;
        }
        for x in x..xlim {
            let ac = ac_420_scalar(y, yrow, ystride, cy, x, p.dc0, p.filter_type);
            if do_u {
                u[urow + x] = P::from_i32(predict_scalar(p.dc1, p.alpha0, ac, max));
            }
            if do_v {
                v[vrow + x] = P::from_i32(predict_scalar(p.dc2, p.alpha1, ac, max));
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
    for (plane, row0, on) in [(&mut *u, urow0, do_u), (&mut *v, vrow0, do_v)] {
        if on {
            let src = row0 + (ylim - 1) * cstride;
            for yy in ylim..h {
                plane.copy_within(src..src + w, row0 + yy * cstride);
            }
        }
    }
}

/// 4:2:0 CfL apply for every luma filter type; the plain 2x2 average keeps
/// using the dedicated kernel.
#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_apply_420_filtered_8bpc_sse41(args: CflApply8<'_>) {
    if args.params.filter_type != CFL_FLT_TYPE_VSTRIP
        && args.params.filter_type != CFL_FLT_TYPE_GAUSS
    {
        return crate::sse::cfl_apply_420_8bpc_sse41(args);
    }
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    apply_420_filtered(y, u, v, layout, area, params, 255);
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_apply_420_filtered_hbd_sse41(args: CflApplyHbd<'_>) {
    if args.params.filter_type != CFL_FLT_TYPE_VSTRIP
        && args.params.filter_type != CFL_FLT_TYPE_GAUSS
    {
        return crate::sse::cfl_apply_420_hbd_sse41(args);
    }
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    apply_420_filtered(y, u, v, layout, area, params, bitdepth_max);
}

// --- MHCCP luma downsampling -----------------------------------------------

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn gen_y_scalar<P: Px>(
    src: &[P],
    src_off: usize,
    top: &[P],
    top_off: usize,
    bottom_offset: usize,
    n_left: usize,
    x: usize,
    filter: i32,
) -> i32 {
    let s = |i: usize| -> i32 { src[src_off + i].into() };
    let b = |i: usize| -> i32 { src[src_off + bottom_offset + i].into() };
    let c = x << 1;
    let l = if n_left > 0 {
        c - 1
    } else {
        c.saturating_sub(1)
    };
    match filter {
        1 => (s(l) + 2 * s(c) + s(c + 1) + b(l) + 2 * b(c) + b(c + 1)) >> 3,
        2 => (s(l) + 4 * s(c) + s(c + 1) + Into::<i32>::into(top[top_off + c]) + b(c)) >> 3,
        _ => (s(c) + s(c + 1) + b(c) + b(c + 1)) >> 2,
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn gen_y_row<P: Px>(
    dst: &mut [P],
    src: &[P],
    src_off: usize,
    top: &[P],
    top_off: usize,
    bottom_offset: usize,
    n_left: usize,
    filter: i32,
) {
    let n = dst.len();
    let mut i = 0usize;
    let fits = |i: usize| {
        let c = (n_left + i) << 1;
        src_off + bottom_offset + c + 8 <= src.len()
            && src_off + c + 8 <= src.len()
            && (filter != 2 || top_off + c + 8 <= top.len())
    };
    while i + 4 <= n && fits(i) {
        let x = n_left + i;
        let c = x << 1;
        let (sc, sr) = even_odd(src, src_off + c);
        let (bc, br) = even_odd(src, src_off + bottom_offset + c);
        let first_l = |row: usize| -> i32 {
            let l = if x == 0 { 0 } else { c - 1 };
            src[row + l].into()
        };
        let v = match filter {
            1 => {
                let sl = left_taps(sr, first_l(src_off));
                let bl = left_taps(br, first_l(src_off + bottom_offset));
                let s = _mm_add_epi32(_mm_add_epi32(sl, _mm_slli_epi32::<1>(sc)), sr);
                let b = _mm_add_epi32(_mm_add_epi32(bl, _mm_slli_epi32::<1>(bc)), br);
                _mm_srli_epi32::<3>(_mm_add_epi32(s, b))
            }
            2 => {
                let sl = left_taps(sr, first_l(src_off));
                let (tc, _) = even_odd(top, top_off + c);
                let s = _mm_add_epi32(_mm_add_epi32(sl, _mm_slli_epi32::<2>(sc)), sr);
                _mm_srli_epi32::<3>(_mm_add_epi32(_mm_add_epi32(s, tc), bc))
            }
            _ => _mm_srli_epi32::<2>(_mm_add_epi32(_mm_add_epi32(sc, sr), _mm_add_epi32(bc, br))),
        };
        store4(&mut dst[i..], v);
        i += 4;
    }
    for i in i..n {
        let v = gen_y_scalar(
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left,
            n_left + i,
            filter,
        );
        dst[i] = P::from_i32(v);
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_gen_y_row_8bpc_sse41(args: CflGenYRow8<'_>) {
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
    gen_y_row(
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type,
    );
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_gen_y_row_hbd_sse41(args: CflGenYRowHbd<'_>) {
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
    gen_y_row(
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type,
    );
}

// --- MHCCP matrix and alpha accumulation -------------------------------------

#[inline(always)]
fn strided4<P: Px>(s: &[P], off: usize, stride: usize) -> __m128i {
    let g = |k: usize| -> i32 { s[off + k * stride].into() };
    unsafe { _mm_setr_epi32(g(0), g(1), g(2), g(3)) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x4_i32(s: &[u16]) -> __m128i {
    let s = &s[..4];
    unsafe { _mm_cvtepu16_epi32(_mm_loadl_epi64(s.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x4_u16(d: &mut [u16], v: __m128i) {
    let d = &mut d[..4];
    unsafe { _mm_storel_epi64(d.as_mut_ptr().cast(), _mm_packus_epi32(v, v)) };
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn gen_mat<P: Px>(
    sums: &mut crate::cfl_dispatch::CflGenMatSums,
    imat0: &mut [u16],
    imat1: &mut [u16],
    imat_off: usize,
    y: &[P],
    v0_off: usize,
    v0_stride: usize,
    v1_off: usize,
    v1_stride: usize,
    len: usize,
    bitdepth: i32,
) {
    let rnd = _mm_set1_epi32(1 << (bitdepth - 1));
    let sh = _mm_cvtsi32_si128(bitdepth);
    let z = _mm_setzero_si128();
    let (mut m00, mut m01, mut s0, mut m11, mut s1) = (z, z, z, z, z);
    let mut i = 0usize;
    while i + 4 <= len {
        let v0 = strided4(y, v0_off + i * v0_stride, v0_stride);
        let raw1 = strided4(y, v1_off + i * v1_stride, v1_stride);
        let v1 = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(raw1, raw1), rnd), sh);
        store_i32x4_u16(&mut imat0[imat_off + i..], v0);
        store_i32x4_u16(&mut imat1[imat_off + i..], v1);
        m00 = _mm_add_epi32(m00, _mm_mullo_epi32(v0, v0));
        m01 = _mm_add_epi32(m01, _mm_mullo_epi32(v0, v1));
        m11 = _mm_add_epi32(m11, _mm_mullo_epi32(v1, v1));
        s0 = _mm_add_epi32(s0, v0);
        s1 = _mm_add_epi32(s1, v1);
        i += 4;
    }
    // Lane sums wrap like the scalar running sums, so order does not matter.
    sums.m00 = sums.m00.wrapping_add(hsum(m00));
    sums.m01 = sums.m01.wrapping_add(hsum(m01));
    sums.m11 = sums.m11.wrapping_add(hsum(m11));
    sums.sum0 = sums.sum0.wrapping_add(hsum(s0));
    sums.sum1 = sums.sum1.wrapping_add(hsum(s1));
    for i in i..len {
        let v0: i32 = y[v0_off + i * v0_stride].into();
        let r: i32 = y[v1_off + i * v1_stride].into();
        let v1 = (r * r + (1 << (bitdepth - 1))) >> bitdepth;
        imat0[imat_off + i] = v0 as u16;
        imat1[imat_off + i] = v1 as u16;
        sums.add_pair(v0, v1);
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_gen_mat_8bpc_sse41(args: CflGenMat8<'_>) {
    let CflGenMat8 {
        sums,
        imat0,
        imat1,
        imat_off,
        y,
        v0_off,
        v0_stride,
        v1_off,
        v1_stride,
        len,
    } = args;
    gen_mat(
        sums, imat0, imat1, imat_off, y, v0_off, v0_stride, v1_off, v1_stride, len, 8,
    );
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_gen_mat_hbd_sse41(args: CflGenMatHbd<'_>) {
    let CflGenMatHbd {
        sums,
        imat0,
        imat1,
        imat_off,
        y,
        v0_off,
        v0_stride,
        v1_off,
        v1_stride,
        len,
        bitdepth,
    } = args;
    gen_mat(
        sums, imat0, imat1, imat_off, y, v0_off, v0_stride, v1_off, v1_stride, len, bitdepth,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn alpha_accum<P: Px>(
    alpha: &mut [i32; 3],
    samples: &[P],
    sample_off: usize,
    sample_stride: usize,
    imat0: &[u16],
    imat1: &[u16],
    imat_off: usize,
    len: usize,
    a2sh: i32,
) {
    let z = _mm_setzero_si128();
    let (mut a0, mut a1, mut sv) = (z, z, z);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = strided4(samples, sample_off + i * sample_stride, sample_stride);
        a0 = _mm_add_epi32(
            a0,
            _mm_mullo_epi32(load_u16x4_i32(&imat0[imat_off + i..]), v),
        );
        a1 = _mm_add_epi32(
            a1,
            _mm_mullo_epi32(load_u16x4_i32(&imat1[imat_off + i..]), v),
        );
        sv = _mm_add_epi32(sv, v);
        i += 4;
    }
    alpha[0] = alpha[0].wrapping_add(hsum(a0));
    alpha[1] = alpha[1].wrapping_add(hsum(a1));
    alpha[2] = alpha[2].wrapping_add(hsum(sv).wrapping_shl(a2sh as u32));
    for i in i..len {
        let v: i32 = samples[sample_off + i * sample_stride].into();
        alpha[0] += imat0[imat_off + i] as i32 * v;
        alpha[1] += imat1[imat_off + i] as i32 * v;
        alpha[2] += v << a2sh;
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_alpha_accum_8bpc_sse41(args: CflAlphaAccum8<'_>) {
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
    alpha_accum(
        alpha,
        samples,
        sample_off,
        sample_stride,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    );
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_alpha_accum_hbd_sse41(args: CflAlphaAccumHbd<'_>) {
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
    alpha_accum(
        alpha,
        samples,
        sample_off,
        sample_stride,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    );
}

// --- MHCCP prediction ------------------------------------------------------

/// `mul32(v, alpha, 16)` for operands that never need the drop path
/// (`cfl_mhccp_coeffs_fit_fast_mul`): round the magnitude, restore the sign.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mul32_q16(v: __m128i, alpha: __m128i) -> __m128i {
    let m = _mm_mullo_epi32(v, alpha);
    let mag = _mm_srli_epi32::<16>(_mm_add_epi32(_mm_abs_epi32(m), _mm_set1_epi32(1 << 15)));
    _mm_sign_epi32(mag, m)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn mhccp_pred<P: Lo4>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_off: usize,
    src_top_stride: usize,
    w: usize,
    h: usize,
    alpha: [i32; 3],
    edge_flags: i32,
    dir: CflMhDir,
    bitdepth: i32,
    max: i32,
) {
    let has_t = edge_flags & CFL_MHCCP_HAS_TOP != 0;
    let has_l = edge_flags & CFL_MHCCP_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let a2v2 = crate::ipred::mul32(alpha[2], 1 << (bitdepth - 1), 16);

    let (al0, al1) = (_mm_set1_epi32(alpha[0]), _mm_set1_epi32(alpha[1]));
    let (a2v, maxv) = (_mm_set1_epi32(a2v2), _mm_set1_epi32(max));
    let rnd = _mm_set1_epi32(1 << (bitdepth - 1));
    let sh = _mm_cvtsi32_si128(bitdepth);
    let px = |i: usize| -> i32 { src[i].into() };
    let pred1 = |v0: i32, v1: i32| -> P {
        let sq = (v1 * v1 + (1 << (bitdepth - 1))) >> bitdepth;
        let mul = |a: i32, b: i32| crate::ipred::mul32(a, b, 16);
        P::from_i32((mul(alpha[0], v0) + mul(alpha[1], sq) + a2v2).clamp(0, max))
    };
    // Predicts `row[x0..w]` from `v0` samples at `v0_base + x` and `v1`
    // samples at `sp + x`.
    let row_run = |row: &mut [P], x0: usize, v0_base: usize, sp: usize| {
        let mut x = x0;
        while x + 4 <= w {
            let v0 = _mm_cvtepu16_epi32(P::load8_lo4(&src[v0_base + x..]));
            let v1 = _mm_cvtepu16_epi32(P::load8_lo4(&src[sp + x..]));
            let sq = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(v1, v1), rnd), sh);
            let p = _mm_add_epi32(_mm_add_epi32(mul32_q16(v0, al0), mul32_q16(sq, al1)), a2v);
            store4(
                &mut row[x..],
                _mm_min_epi32(_mm_max_epi32(p, _mm_setzero_si128()), maxv),
            );
            x += 4;
        }
        for x in x..w {
            row[x] = pred1(px(v0_base + x), px(sp + x));
        }
    };

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && h > 0 {
        row_run(&mut dst[..w], 0, sp - src_top_stride, sp);
        sp += w;
        y = 1;
    }
    for (row_y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate().skip(y) {
        let dst_row = &mut dst_row[..w];
        if dir_t {
            let back = (((row_y > 0) as usize) | has_t as usize) * w;
            row_run(dst_row, 0, sp - back, sp);
        } else if dir_l {
            if has_l {
                dst_row[0] = pred1(px(left_off + row_y * n_left + 1), px(sp));
            } else {
                dst_row[0] = pred1(px(sp), px(sp));
            }
            row_run(dst_row, 1, sp - 1, sp);
        } else {
            row_run(dst_row, 0, sp, sp);
        }
        sp += w;
    }
}

trait Lo4: Px {
    /// Four consecutive samples widened to u16 lanes (low half).
    fn load8_lo4(s: &[Self]) -> __m128i;
}

impl Lo4 for u8 {
    #[inline(always)]
    fn load8_lo4(s: &[u8]) -> __m128i {
        let s = &s[..4];
        unsafe {
            _mm_cvtepu8_epi16(_mm_cvtsi32_si128(i32::from_le_bytes([
                s[0], s[1], s[2], s[3],
            ])))
        }
    }
}

impl Lo4 for u16 {
    #[inline(always)]
    fn load8_lo4(s: &[u16]) -> __m128i {
        let s = &s[..4];
        unsafe { _mm_loadl_epi64(s.as_ptr().cast()) }
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_mhccp_pred_8bpc_sse41(args: CflMhccpPred8<'_>) {
    if !crate::cfl_dispatch::cfl_mhccp_coeffs_fit_fast_mul(&args.alpha) {
        return crate::cfl_dispatch::cfl_mhccp_pred_8bpc_scalar(args);
    }
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
    mhccp_pred(
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
        8,
        255,
    );
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn cfl_mhccp_pred_hbd_sse41(args: CflMhccpPredHbd<'_>) {
    if !crate::cfl_dispatch::cfl_mhccp_coeffs_fit_fast_mul(&args.alpha) {
        return crate::cfl_dispatch::cfl_mhccp_pred_hbd_scalar(args);
    }
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
    mhccp_pred(
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
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfl_dispatch::*;
    use crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES as N;

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    fn rand_vec(seed: &mut u64, n: usize, max: u32) -> Vec<u16> {
        (0..n).map(|_| (lcg(seed) % (max + 1)) as u16).collect()
    }

    fn have() -> bool {
        std::is_x86_feature_detected!("sse4.1")
    }

    #[test]
    fn apply_420_filtered_matches_scalar() {
        if !have() {
            return;
        }
        let mut seed = 11u64;
        for &(w, h) in &[
            (4usize, 4usize),
            (8, 8),
            (16, 8),
            (32, 32),
            (8, 32),
            (32, 16),
        ] {
            for &bits in &[8u32, 10, 12] {
                for filter in 0..3u32 {
                    let max = (1u32 << bits) - 1;
                    let ystride = 2 * w + 12;
                    let y = rand_vec(&mut seed, ystride * (2 * h + 4), max);
                    let cstride = w + 5;
                    let xlim = 1 + lcg(&mut seed) as usize % w;
                    let ylim = 1 + lcg(&mut seed) as usize % h;
                    let alpha0 = lcg(&mut seed) as i32 % 129 - 64;
                    let alpha1 = lcg(&mut seed) as i32 % 129 - 64;
                    let params = CflParams {
                        dc0: (lcg(&mut seed) % (8 * max)) as i32,
                        dc1: (lcg(&mut seed) % max) as i32,
                        dc2: (lcg(&mut seed) % max) as i32,
                        alpha0,
                        alpha1,
                        filter_type: filter,
                    };
                    let layout = CflLayout {
                        yrow0: 2 * ystride,
                        urow0: 0,
                        vrow0: 0,
                        ystride,
                        cstride,
                    };
                    let area = CflArea { w, h, xlim, ylim };
                    let tag = format!("{w}x{h} {bits}b f{filter} xlim{xlim} ylim{ylim}");
                    if bits == 8 {
                        let y8: Vec<u8> = y.iter().map(|&v| v as u8).collect();
                        let run = |sse: bool| {
                            let (mut u, mut v) = (vec![0u8; cstride * h], vec![0u8; cstride * h]);
                            let args = CflApply8 {
                                y: &y8,
                                u: &mut u,
                                v: &mut v,
                                layout,
                                area,
                                params,
                            };
                            if sse {
                                unsafe { cfl_apply_420_filtered_8bpc_sse41(args) }
                            } else {
                                cfl_apply_420_8bpc_scalar(args)
                            }
                            (u, v)
                        };
                        assert_eq!(run(true), run(false), "{tag}");
                    } else {
                        let run = |sse: bool| {
                            let (mut u, mut v) = (vec![0u16; cstride * h], vec![0u16; cstride * h]);
                            let args = CflApplyHbd {
                                y: &y,
                                u: &mut u,
                                v: &mut v,
                                layout,
                                area,
                                params,
                                bitdepth_max: max as i32,
                            };
                            if sse {
                                unsafe { cfl_apply_420_filtered_hbd_sse41(args) }
                            } else {
                                cfl_apply_420_hbd_scalar(args)
                            }
                            (u, v)
                        };
                        assert_eq!(run(true), run(false), "{tag}");
                    }
                }
            }
        }
    }

    #[test]
    fn gen_y_row_matches_scalar() {
        if !have() {
            return;
        }
        let mut seed = 5u64;
        for n in 1..=40usize {
            for n_left in 0..3usize {
                for filter in 0..3i32 {
                    for &bits in &[8u32, 10] {
                        let max = (1u32 << bits) - 1;
                        let bottom_offset = 2 * (n + n_left) + 6;
                        let src = rand_vec(&mut seed, 3 * bottom_offset, max);
                        let top = rand_vec(&mut seed, 2 * (n + n_left) + 2, max);
                        let tag = format!("n{n} left{n_left} f{filter} {bits}b");
                        if bits == 8 {
                            let s8: Vec<u8> = src.iter().map(|&v| v as u8).collect();
                            let t8: Vec<u8> = top.iter().map(|&v| v as u8).collect();
                            let run = |sse: bool| {
                                let mut dst = vec![0u8; n];
                                let a = CflGenYRow8 {
                                    dst: &mut dst,
                                    src: &s8,
                                    src_off: 1,
                                    top: &t8,
                                    top_off: 0,
                                    bottom_offset,
                                    n_left,
                                    filter_type: filter,
                                };
                                if sse {
                                    unsafe { cfl_gen_y_row_8bpc_sse41(a) }
                                } else {
                                    cfl_gen_y_row_8bpc_scalar(a)
                                }
                                dst
                            };
                            assert_eq!(run(true), run(false), "{tag}");
                        } else {
                            let run = |sse: bool| {
                                let mut dst = vec![0u16; n];
                                let a = CflGenYRowHbd {
                                    dst: &mut dst,
                                    src: &src,
                                    src_off: 1,
                                    top: &top,
                                    top_off: 0,
                                    bottom_offset,
                                    n_left,
                                    filter_type: filter,
                                };
                                if sse {
                                    unsafe { cfl_gen_y_row_hbd_sse41(a) }
                                } else {
                                    cfl_gen_y_row_hbd_scalar(a)
                                }
                                dst
                            };
                            assert_eq!(run(true), run(false), "{tag}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gen_mat_and_alpha_accum_match_scalar() {
        if !have() {
            return;
        }
        let mut seed = 9u64;
        for len in [0usize, 1, 3, 4, 7, 16, 33, 128] {
            for &stride in &[1usize, 3] {
                for &bits in &[8u32, 10, 12] {
                    let max = (1u32 << bits) - 1;
                    let y = rand_vec(&mut seed, 4 + len * stride * 2, max);
                    let y8: Vec<u8> = y.iter().map(|&v| v as u8).collect();
                    let gen_mat_run = |sse: bool| {
                        let mut sums = CflGenMatSums {
                            m00: 5,
                            m01: -3,
                            sum0: 1,
                            m11: 2,
                            sum1: 0,
                        };
                        let (mut i0, mut i1) = (Box::new([0u16; N]), Box::new([0u16; N]));
                        if bits == 8 {
                            let a = CflGenMat8 {
                                sums: &mut sums,
                                imat0: &mut i0,
                                imat1: &mut i1,
                                imat_off: 2,
                                y: &y8,
                                v0_off: 1,
                                v0_stride: stride,
                                v1_off: 2,
                                v1_stride: stride,
                                len,
                            };
                            if sse {
                                unsafe { cfl_gen_mat_8bpc_sse41(a) }
                            } else {
                                cfl_gen_mat_8bpc_scalar(a)
                            }
                        } else {
                            let a = CflGenMatHbd {
                                sums: &mut sums,
                                imat0: &mut i0,
                                imat1: &mut i1,
                                imat_off: 2,
                                y: &y,
                                v0_off: 1,
                                v0_stride: stride,
                                v1_off: 2,
                                v1_stride: stride,
                                len,
                                bitdepth: bits as i32,
                            };
                            if sse {
                                unsafe { cfl_gen_mat_hbd_sse41(a) }
                            } else {
                                cfl_gen_mat_hbd_scalar(a)
                            }
                        }
                        ((sums.m00, sums.m01, sums.sum0, sums.m11, sums.sum1), i0, i1)
                    };
                    let (ra, i0, i1) = gen_mat_run(true);
                    let (rb, j0, j1) = gen_mat_run(false);
                    assert_eq!(ra, rb, "gen_mat len{len} s{stride} {bits}b");
                    assert!(i0[..] == j0[..] && i1[..] == j1[..], "imat len{len}");

                    let acc = |sse: bool| {
                        let mut alpha = [7, -2, 3];
                        if bits == 8 {
                            let a = CflAlphaAccum8 {
                                alpha: &mut alpha,
                                samples: &y8,
                                sample_off: 3,
                                sample_stride: stride,
                                imat0: &i0,
                                imat1: &i1,
                                imat_off: 2,
                                len,
                                a2sh: 3,
                            };
                            if sse {
                                unsafe { cfl_alpha_accum_8bpc_sse41(a) }
                            } else {
                                cfl_alpha_accum_8bpc_scalar(a)
                            }
                        } else {
                            let a = CflAlphaAccumHbd {
                                alpha: &mut alpha,
                                samples: &y,
                                sample_off: 3,
                                sample_stride: stride,
                                imat0: &i0,
                                imat1: &i1,
                                imat_off: 2,
                                len,
                                a2sh: 5,
                            };
                            if sse {
                                unsafe { cfl_alpha_accum_hbd_sse41(a) }
                            } else {
                                cfl_alpha_accum_hbd_scalar(a)
                            }
                        }
                        alpha
                    };
                    assert_eq!(acc(true), acc(false), "alpha len{len} s{stride} {bits}b");
                }
            }
        }
    }

    #[test]
    fn mhccp_pred_matches_scalar() {
        if !have() {
            return;
        }
        let mut seed = 13u64;
        let dirs = [
            CflMhDir::Center,
            CflMhDir::Top,
            CflMhDir::Left,
            CflMhDir::All,
        ];
        for &(w, h) in &[(4usize, 4usize), (8, 4), (16, 16), (5, 7), (32, 8)] {
            for dir in dirs {
                for edge_flags in [
                    0,
                    CFL_MHCCP_HAS_TOP,
                    CFL_MHCCP_HAS_LEFT,
                    CFL_MHCCP_HAS_TOP | CFL_MHCCP_HAS_LEFT,
                ] {
                    for &bits in &[8u32, 10, 12] {
                        let max = (1u32 << bits) - 1;
                        let top_stride = w;
                        let src_off = 2 * w + 4;
                        let src = rand_vec(&mut seed, src_off + 64 * 64 + 4 * 64 + 16, max);
                        let src8: Vec<u8> = src.iter().map(|&v| v as u8).collect();
                        let big = lcg(&mut seed) % 4 == 0;
                        let r = |s: &mut u64| {
                            let lim = if big { 200_000 } else { 65_536 };
                            (lcg(s) % (2 * lim)) as i32 - lim as i32
                        };
                        let alpha = [r(&mut seed), r(&mut seed), r(&mut seed)];
                        let stride = w + 3;
                        let tag = format!("{w}x{h} {dir:?} e{edge_flags} {bits}b a{alpha:?}");
                        if bits == 8 {
                            let run = |sse: bool| {
                                let mut dst = vec![0u8; stride * h];
                                let a = CflMhccpPred8 {
                                    dst: &mut dst,
                                    dst_stride: stride,
                                    src: &src8,
                                    src_off,
                                    src_top_stride: top_stride,
                                    w,
                                    h,
                                    alpha,
                                    edge_flags,
                                    dir,
                                };
                                if sse {
                                    unsafe { cfl_mhccp_pred_8bpc_sse41(a) }
                                } else {
                                    cfl_mhccp_pred_8bpc_scalar(a)
                                }
                                dst
                            };
                            assert_eq!(run(true), run(false), "{tag}");
                        } else {
                            let run = |sse: bool| {
                                let mut dst = vec![0u16; stride * h];
                                let a = CflMhccpPredHbd {
                                    dst: &mut dst,
                                    dst_stride: stride,
                                    src: &src,
                                    src_off,
                                    src_top_stride: top_stride,
                                    w,
                                    h,
                                    alpha,
                                    edge_flags,
                                    dir,
                                    bitdepth: bits as i32,
                                    bitdepth_max: max as i32,
                                };
                                if sse {
                                    unsafe { cfl_mhccp_pred_hbd_sse41(a) }
                                } else {
                                    cfl_mhccp_pred_hbd_scalar(a)
                                }
                                dst
                            };
                            assert_eq!(run(true), run(false), "{tag}");
                        }
                    }
                }
            }
        }
    }
}
