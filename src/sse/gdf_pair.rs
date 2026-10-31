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

//! GDF prep pixel pairs for SSE4.1 (128-bit port of the AVX2 kernel).

use crate::gdf_tables::{GDF_ALPHA, GDF_INTER_ERROR, GDF_INTRA_ERROR, GDF_WEIGHT};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

static GDF_PREP_COORDS_8BPC: [[i8; 2]; 18] = [
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

#[inline]
fn gdf_prep_apply_sign(v: i32) -> i32 {
    if v < 0 {
        -((v.wrapping_neg() + (1 << 14)) >> 15)
    } else {
        (v + (1 << 14)) >> 15
    }
}

#[inline]
fn gdf_prep_lookup_error(ref_dst_idx: usize, error_lut_base: usize, full_idx: usize) -> i8 {
    if ref_dst_idx == 0 {
        GDF_INTRA_ERROR[error_lut_base + full_idx]
    } else {
        GDF_INTER_ERROR[error_lut_base + full_idx]
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn gdf_load_pair_u8_i32(row: &[u8], col: usize) -> __m128i {
    let raw = unsafe { _mm_loadu_si16(row[col..].as_ptr().cast()) };
    _mm_cvtepu8_epi32(raw)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn gdf_clip_i32x4(v: __m128i, lo: __m128i, hi: __m128i) -> __m128i {
    _mm_min_epi32(_mm_max_epi32(v, lo), hi)
}

#[inline(always)]
fn gdf_prep_full_idx(v0: i32, v1: i32, v2: i32, scale: i32) -> usize {
    let scale2 = scale as usize * 2;
    let v0 = gdf_prep_apply_sign(v0 * scale);
    let v1 = gdf_prep_apply_sign(v1 * scale);
    let v2 = gdf_prep_apply_sign(v2 * scale);
    let s0 = (v0.clamp(-scale, scale - 1) + scale) as usize;
    let s1 = (v1.clamp(-scale, scale - 1) + scale) as usize;
    let s2 = (v2.clamp(-scale, scale - 1) + scale) as usize;
    ((s0 * scale2) + s1) * scale2 + s2
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_prep_pair_8bpc_sse41(
    rows: [&[u8]; 13],
    col: usize,
    cls: usize,
    shared_vals: [i32; 3],
    alpha_base: usize,
    weight_base: usize,
    error_lut_base: usize,
    scale: i32,
    ref_dst_idx: usize,
) -> [i8; 2] {
    let m = gdf_load_pair_u8_i32(rows[6], col);
    let v_lo = _mm_set1_epi32(-512);
    let v_hi = _mm_set1_epi32(511);
    let mut acc0 = _mm_set1_epi32(shared_vals[0]);
    let mut acc1 = _mm_set1_epi32(shared_vals[1]);
    let mut acc2 = _mm_set1_epi32(shared_vals[2]);

    for (k, &[dy, dx]) in GDF_PREP_COORDS_8BPC.iter().enumerate() {
        let dy = dy as i32;
        let dx = dx as i32;
        let alpha = GDF_ALPHA[alpha_base + k * 4 + cls] as i32;
        let alpha_v = _mm_set1_epi32(alpha);
        let neg_alpha_v = _mm_set1_epi32(-alpha);
        let a_col = (col as i32 - dx) as usize;
        let b_col = (col as i32 + dx) as usize;
        let a = gdf_load_pair_u8_i32(rows[(6 - dy) as usize], a_col);
        let b = gdf_load_pair_u8_i32(rows[(6 + dy) as usize], b_col);
        let above = gdf_clip_i32x4(
            _mm_slli_epi32::<2>(_mm_sub_epi32(a, m)),
            neg_alpha_v,
            alpha_v,
        );
        let below = gdf_clip_i32x4(
            _mm_slli_epi32::<2>(_mm_sub_epi32(b, m)),
            neg_alpha_v,
            alpha_v,
        );
        let v = gdf_clip_i32x4(_mm_add_epi32(above, below), v_lo, v_hi);
        acc0 = _mm_add_epi32(
            acc0,
            _mm_mullo_epi32(
                v,
                _mm_set1_epi32(GDF_WEIGHT[weight_base + k * 4 + cls] as i32),
            ),
        );
        acc1 = _mm_add_epi32(
            acc1,
            _mm_mullo_epi32(
                v,
                _mm_set1_epi32(GDF_WEIGHT[weight_base + 88 + k * 4 + cls] as i32),
            ),
        );
        acc2 = _mm_add_epi32(
            acc2,
            _mm_mullo_epi32(
                v,
                _mm_set1_epi32(GDF_WEIGHT[weight_base + 176 + k * 4 + cls] as i32),
            ),
        );
    }

    [
        gdf_prep_lookup_error(
            ref_dst_idx,
            error_lut_base,
            gdf_prep_full_idx(
                _mm_cvtsi128_si32(acc0),
                _mm_cvtsi128_si32(acc1),
                _mm_cvtsi128_si32(acc2),
                scale,
            ),
        ),
        gdf_prep_lookup_error(
            ref_dst_idx,
            error_lut_base,
            gdf_prep_full_idx(
                _mm_extract_epi32::<1>(acc0),
                _mm_extract_epi32::<1>(acc1),
                _mm_extract_epi32::<1>(acc2),
                scale,
            ),
        ),
    ]
}
