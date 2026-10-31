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

//! GDF prep rows for SSE4.1: eight pixels (four filter-class pairs) per step,
//! following the NEON row kernel; the `< 8` tail uses the scalar row.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::gdf_tables::{GDF_ALPHA, GDF_INTER_ERROR, GDF_INTRA_ERROR, GDF_WEIGHT};
use crate::rowops_dispatch::{GdfPrepRow, gdf_prep_class};

const GDF_PREP_COORDS: [[i8; 2]; 18] = [
    [6, 0],
    [5, 0],
    [4, 0],
    [3, 0],
    [2, 1],
    [2, 0],
    [2, -1],
    [1, 2],
    [1, 1],
    [1, 0],
    [1, -1],
    [1, -2],
    [0, 6],
    [0, 5],
    [0, 4],
    [0, 3],
    [0, 2],
    [0, 1],
];

/// Eight pixels as `i16` with the per-bitdepth normalization applied
/// (`<< 2` at 8 bpc, `>> down_shift` at high bitdepth).
trait GdfPx: Copy {
    fn load8(row: &[Self], col: usize, sh: __m128i) -> __m128i;
}

impl GdfPx for u8 {
    #[inline(always)]
    fn load8(row: &[u8], col: usize, sh: __m128i) -> __m128i {
        let v = &row[col..col + 8];
        unsafe { _mm_sll_epi16(_mm_cvtepu8_epi16(_mm_loadl_epi64(v.as_ptr().cast())), sh) }
    }
}

impl GdfPx for u16 {
    #[inline(always)]
    fn load8(row: &[u16], col: usize, sh: __m128i) -> __m128i {
        let v = &row[col..col + 8];
        unsafe { _mm_srl_epi16(_mm_loadu_si128(v.as_ptr().cast()), sh) }
    }
}

/// Per-lane pick from a 4-entry (one per filter class) 16-bit table row.
#[inline]
#[target_feature(enable = "sse4.1")]
fn class_lookup<T>(table: &[T], tidx: __m128i) -> __m128i {
    debug_assert!(core::mem::size_of::<T>() == 2);
    let t = &table[..4];
    unsafe { _mm_shuffle_epi8(_mm_loadl_epi64(t.as_ptr().cast()), tidx) }
}

/// `clamp(apply_sign(v * scale), -scale, scale - 1) + scale` per lane.
#[inline]
#[target_feature(enable = "sse4.1")]
fn sub_idx(v: __m128i, scale: i32) -> __m128i {
    let v = _mm_mullo_epi32(v, _mm_set1_epi32(scale));
    let r = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_abs_epi32(v), _mm_set1_epi32(1 << 14)));
    let r = _mm_sign_epi32(r, v);
    let r = _mm_min_epi32(
        _mm_max_epi32(r, _mm_set1_epi32(-scale)),
        _mm_set1_epi32(scale - 1),
    );
    _mm_add_epi32(r, _mm_set1_epi32(scale))
}

#[inline]
fn lookup_error(ref_dst_idx: usize, error_lut_base: usize, full_idx: usize) -> i8 {
    if ref_dst_idx == 0 {
        GDF_INTRA_ERROR[error_lut_base + full_idx]
    } else {
        GDF_INTER_ERROR[error_lut_base + full_idx]
    }
}

/// GDF prep for 8 consecutive pixels starting at `col`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn prep_8px<P: GdfPx>(
    rows: [&[P]; 13],
    col: usize,
    cls: [usize; 4],
    shared: [[i32; 3]; 4],
    p: &GdfPrepRow,
    sh: __m128i,
    dst: &mut [i8],
) {
    // u16 lane i reads table entry cls[i / 2]: bytes 2c and 2c + 1.
    let mut tb = [0u8; 16];
    for (i, pair) in tb.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let c = 2 * cls[i >> 1] as u8;
        *pair = [c, c + 1];
    }
    let tidx = unsafe { _mm_loadu_si128(tb.as_ptr().cast()) };

    let mut acc = [[_mm_setzero_si128(); 2]; 3];
    for (idx, acc) in acc.iter_mut().enumerate() {
        let s = |j: usize| shared[j][idx];
        *acc = [
            _mm_setr_epi32(s(0), s(0), s(1), s(1)),
            _mm_setr_epi32(s(2), s(2), s(3), s(3)),
        ];
    }

    let m = P::load8(rows[6], col, sh);
    let (v_lo, v_hi) = (_mm_set1_epi16(-512), _mm_set1_epi16(511));
    for (k, &[dy, dx]) in GDF_PREP_COORDS.iter().enumerate() {
        let alpha = class_lookup(&GDF_ALPHA[p.alpha_base + k * 4..], tidx);
        let neg_alpha = _mm_sub_epi16(_mm_setzero_si128(), alpha);
        let a = P::load8(
            rows[(6 - dy) as usize],
            (col as isize - dx as isize) as usize,
            sh,
        );
        let b = P::load8(
            rows[(6 + dy) as usize],
            (col as isize + dx as isize) as usize,
            sh,
        );
        let above = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(a, m), neg_alpha), alpha);
        let below = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(b, m), neg_alpha), alpha);
        let v = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(above, below), v_lo), v_hi);
        for (idx, acc) in acc.iter_mut().enumerate() {
            let wv = class_lookup(&GDF_WEIGHT[p.weight_base + idx * 88 + k * 4..], tidx);
            let (lo, hi) = (_mm_mullo_epi16(v, wv), _mm_mulhi_epi16(v, wv));
            acc[0] = _mm_add_epi32(acc[0], _mm_unpacklo_epi16(lo, hi));
            acc[1] = _mm_add_epi32(acc[1], _mm_unpackhi_epi16(lo, hi));
        }
    }

    let scale2 = _mm_set1_epi32(2 * p.scale);
    let mut full = [0i32; 8];
    for half in 0..2 {
        let s0 = sub_idx(acc[0][half], p.scale);
        let s1 = sub_idx(acc[1][half], p.scale);
        let s2 = sub_idx(acc[2][half], p.scale);
        let f = _mm_add_epi32(
            _mm_mullo_epi32(_mm_add_epi32(_mm_mullo_epi32(s0, scale2), s1), scale2),
            s2,
        );
        unsafe { _mm_storeu_si128(full[half * 4..].as_mut_ptr().cast(), f) };
    }
    for (d, &f) in dst[..8].iter_mut().zip(full.iter()) {
        *d = lookup_error(p.ref_dst_idx, p.error_lut_base, f as usize);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn prep_row_full<P: GdfPx>(
    rows: [&[P]; 13],
    o: usize,
    full: usize,
    grad: [&[[u16; 4]]; 2],
    dst: &mut [i8],
    p: &GdfPrepRow,
    sh: __m128i,
) {
    let mut x1 = 0usize;
    while x1 < full {
        let c = [0, 1, 2, 3].map(|j| gdf_prep_class(grad, (x1 >> 1) + j, p));
        prep_8px(
            rows,
            o + x1,
            c.map(|c| c.0),
            c.map(|c| c.1),
            p,
            sh,
            &mut dst[x1..],
        );
        x1 += 8;
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_prep_row_8bpc_sse41(
    rows: [&[u8]; 13],
    o: usize,
    w: usize,
    grad: [&[[u16; 4]]; 2],
    dst: &mut [i8],
    p: &GdfPrepRow,
) {
    let full = w & !7;
    prep_row_full(rows, o, full, grad, dst, p, _mm_cvtsi32_si128(2));
    if full < w {
        crate::rowops_dispatch::gdf_prep_row_8bpc_scalar(
            rows.map(|r| &r[full..]),
            o,
            w - full,
            [&grad[0][full >> 1..], &grad[1][full >> 1..]],
            &mut dst[full..],
            p,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_prep_row_hbd_sse41(
    rows: [&[u16]; 13],
    o: usize,
    w: usize,
    grad: [&[[u16; 4]]; 2],
    dst: &mut [i8],
    p: &GdfPrepRow,
    down_shift: u32,
    up_scale: i32,
) {
    // High bitdepth never up-scales; the pre-shift is then exactly `>> down_shift`.
    let full = if up_scale == 1 { w & !7 } else { 0 };
    prep_row_full(
        rows,
        o,
        full,
        grad,
        dst,
        p,
        _mm_cvtsi32_si128(down_shift as i32),
    );
    if full < w {
        crate::rowops_dispatch::gdf_prep_row_hbd_scalar(
            rows.map(|r| &r[full..]),
            o,
            w - full,
            [&grad[0][full >> 1..], &grad[1][full >> 1..]],
            &mut dst[full..],
            p,
            down_shift,
            up_scale,
        );
    }
}
