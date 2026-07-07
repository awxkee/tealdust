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

use std::arch::x86_64::*;

use crate::gdf_tables::{GDF_ALPHA, GDF_INTER_ERROR, GDF_INTRA_ERROR, GDF_WEIGHT};

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
    [0, 3],
    [0, 2],
    [0, 1],
    [0, -1],
    [0, -2],
    [0, -3],
];

#[inline(always)]
fn load_i32x8(a: &[i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn load_i16x16(a: &[i16; 16]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline(always)]
fn load_i16x8(a: &[i16; 8]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}
#[inline(always)]
fn load_u16x16(a: &[u16; 16]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline(always)]
fn load_u16x8(a: &[u16; 8]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline(always)]
fn m128x2_to_m256(lo: __m128i, hi: __m128i) -> __m256i {
    unsafe { _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi) }
}

#[inline(always)]
fn load_u8x16_i16(a: &[u8; 16]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(a.as_ptr().cast())) }
}

#[inline(always)]
fn load_u8x8_i16(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr().cast())) }
}

#[inline(always)]
fn madd_i16x16_const(a: __m256i, b: __m256i, coeff: __m256i) -> (__m256i, __m256i) {
    unsafe {
        let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), coeff);
        let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), coeff);
        (
            _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm256_castsi256_si128(lo)),
                _mm256_castsi256_si128(hi),
            ),
            _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(_mm256_extracti128_si256::<1>(lo)),
                _mm256_extracti128_si256::<1>(hi),
            ),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_i16x16(a: __m256i, b: __m256i, w1: __m256i, w2: __m256i) -> (__m256i, __m256i) {
    let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), _mm256_unpacklo_epi16(w1, w2));
    let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), _mm256_unpackhi_epi16(w1, w2));
    (
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm256_castsi256_si128(lo)),
            _mm256_castsi256_si128(hi),
        ),
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm256_extracti128_si256::<1>(lo)),
            _mm256_extracti128_si256::<1>(hi),
        ),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn madd_i16x8_const(a: __m128i, b: __m128i, coeff: __m128i) -> (__m128i, __m128i) {
    (
        _mm_madd_epi16(_mm_unpacklo_epi16(a, b), coeff),
        _mm_madd_epi16(_mm_unpackhi_epi16(a, b), coeff),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn madd_i16x8(a: __m128i, b: __m128i, w1: __m128i, w2: __m128i) -> (__m128i, __m128i) {
    (
        _mm_madd_epi16(_mm_unpacklo_epi16(a, b), _mm_unpacklo_epi16(w1, w2)),
        _mm_madd_epi16(_mm_unpackhi_epi16(a, b), _mm_unpackhi_epi16(w1, w2)),
    )
}

#[inline(always)]
fn load_i16x8_i32(a: &[i16; 8]) -> __m256i {
    unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn load_i16x16_i32x2(a: &[i16; 16]) -> (__m256i, __m256i) {
    unsafe {
        let v = _mm256_loadu_si256(a.as_ptr().cast());
        (
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)),
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v)),
        )
    }
}

#[inline(always)]
fn load_u16x8_i32(a: &[u16; 8]) -> __m256i {
    unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn load_u16x16_i32x2(a: &[u16; 16]) -> (__m256i, __m256i) {
    unsafe {
        let v = _mm256_loadu_si256(a.as_ptr().cast());
        (
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v)),
            _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v)),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8_u16_clip(a: &mut [u16; 8], v: __m256i, max_v: __m256i) {
    let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_packus_epi32(lo, hi);
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, p) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x16_u16_clip(a: &mut [u16; 16], lo: __m256i, hi: __m256i, max_v: __m256i) {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_min_epi32(_mm256_max_epi32(lo, zero), max_v);
    let hi = _mm256_min_epi32(_mm256_max_epi32(hi, zero), max_v);
    let p = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(lo, hi));
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), p) };
}

#[target_feature(enable = "avx2")]
pub(crate) fn residual_add_row_hbd_avx2(
    dst: &mut [u16],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
    bitdepth_max: i32,
) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(shift);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let f = |cv: __m256i| _mm256_sra_epi32(_mm256_add_epi32(cv, rnd_v), shc);

    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (c16, _) = c[..n].as_chunks::<16>();
    for (d, cv) in d16.iter_mut().zip(c16) {
        let (d0, d1) = load_u16x16_i32x2(&*d);
        let c0 = f(load_i32x8((&cv[..8]).try_into().unwrap()));
        let c1 = f(load_i32x8((&cv[8..]).try_into().unwrap()));
        store_i32x16_u16_clip(d, _mm256_add_epi32(d0, c0), _mm256_add_epi32(d1, c1), max_v);
    }
    let done = d16.len() * 16;
    let (d8, r8) = r16.as_chunks_mut::<8>();
    let (c8, cr) = c[done..n].as_chunks::<8>();
    for (d, cv) in d8.iter_mut().zip(c8) {
        let dv = load_u16x8_i32(&*d);
        let cf = f(load_i32x8(cv));
        store_i32x8_u16_clip(d, _mm256_add_epi32(dv, cf), max_v);
    }
    for (d, &cv) in r8.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dc_add_row_hbd_avx2(dst: &mut [u16], dc: i32, n: usize, bitdepth_max: i32) {
    if dc == 0 {
        return;
    }
    let dc_v = _mm256_set1_epi32(dc);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let (d64, r64) = dst[..n].as_chunks_mut::<64>();
    for d in d64.iter_mut() {
        let (d01, d23) = d.split_at_mut(32);
        let (d0, d1) = d01.split_at_mut(16);
        let (d2, d3) = d23.split_at_mut(16);
        let d0: &mut [u16; 16] = d0.try_into().unwrap();
        let d1: &mut [u16; 16] = d1.try_into().unwrap();
        let d2: &mut [u16; 16] = d2.try_into().unwrap();
        let d3: &mut [u16; 16] = d3.try_into().unwrap();

        let (d00, d01) = load_u16x16_i32x2(&*d0);
        let (d10, d11) = load_u16x16_i32x2(&*d1);
        let (d20, d21) = load_u16x16_i32x2(&*d2);
        let (d30, d31) = load_u16x16_i32x2(&*d3);
        store_i32x16_u16_clip(
            d0,
            _mm256_add_epi32(d00, dc_v),
            _mm256_add_epi32(d01, dc_v),
            max_v,
        );
        store_i32x16_u16_clip(
            d1,
            _mm256_add_epi32(d10, dc_v),
            _mm256_add_epi32(d11, dc_v),
            max_v,
        );
        store_i32x16_u16_clip(
            d2,
            _mm256_add_epi32(d20, dc_v),
            _mm256_add_epi32(d21, dc_v),
            max_v,
        );
        store_i32x16_u16_clip(
            d3,
            _mm256_add_epi32(d30, dc_v),
            _mm256_add_epi32(d31, dc_v),
            max_v,
        );
    }
    let (d16, r16) = r64.as_chunks_mut::<16>();
    for d in d16.iter_mut() {
        let (d0, d1) = load_u16x16_i32x2(&*d);
        store_i32x16_u16_clip(
            d,
            _mm256_add_epi32(d0, dc_v),
            _mm256_add_epi32(d1, dc_v),
            max_v,
        );
    }
    let (d8, r8) = r16.as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        store_i32x8_u16_clip(d, _mm256_add_epi32(load_u16x8_i32(&*d), dc_v), max_v);
    }
    for d in r8.iter_mut() {
        *d = ((*d as i32) + dc).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avg_row_hbd_avx2(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let f = |a: __m256i, b: __m256i| {
        _mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(a, b), rnd_v), shc)
    };
    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in d16.iter_mut().zip(a16).zip(b16) {
        let (a0, a1) = load_i16x16_i32x2(a);
        let (b0, b1) = load_i16x16_i32x2(b);
        store_i32x16_u16_clip(d, f(a0, b0), f(a1, b1), max_v);
    }
    let done = d16.len() * 16;
    let (d8, r8) = r16.as_chunks_mut::<8>();
    let (a8, ar) = t1[done..n].as_chunks::<8>();
    let (b8, br) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in d8.iter_mut().zip(a8).zip(b8) {
        let v = f(load_i16x8_i32(a), load_i16x8_i32(b));
        store_i32x8_u16_clip(d, v, max_v);
    }
    for ((d, &a), &b) in r8.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 + b as i32 + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn w_avg_row_hbd_avx2(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let coeff =
        _mm256_set1_epi32((((16 - weight) as u16 as u32) << 16 | (weight as u16 as u32)) as i32);
    let coeff128 =
        _mm_set1_epi32((((16 - weight) as u16 as u32) << 16 | (weight as u16 as u32)) as i32);
    let rnd_v = _mm256_set1_epi32(rnd);
    let rnd128_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let f = |v: __m256i| _mm256_sra_epi32(_mm256_add_epi32(v, rnd_v), shc);
    let f128 = |v: __m128i| _mm_sra_epi32(_mm_add_epi32(v, rnd128_v), shc);

    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in d16.iter_mut().zip(a16).zip(b16) {
        let (s0, s1) = madd_i16x16_const(load_i16x16(a), load_i16x16(b), coeff);
        store_i32x16_u16_clip(d, f(s0), f(s1), max_v);
    }
    let done = d16.len() * 16;
    let (d8, r8) = r16.as_chunks_mut::<8>();
    let (a8, ar) = t1[done..n].as_chunks::<8>();
    let (b8, br) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in d8.iter_mut().zip(a8).zip(b8) {
        let (s0, s1) = madd_i16x8_const(load_i16x8(a), load_i16x8(b), coeff128);
        let out = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(f128(s0)), f128(s1));
        store_i32x8_u16_clip(d, out, max_v);
    }
    for ((d, &a), &b) in r8.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, bitdepth_max)
            as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn mask_row_hbd_avx2(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let rnd128_v = _mm_set1_epi32(rnd);
    let c64 = _mm256_set1_epi16(64);
    let c64_128 = _mm_set1_epi16(64);
    let shc = _mm_cvtsi32_si128(sh);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let f = |v: __m256i| _mm256_sra_epi32(_mm256_add_epi32(v, rnd_v), shc);
    let f128 = |v: __m128i| _mm_sra_epi32(_mm_add_epi32(v, rnd128_v), shc);

    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for (((d, a), b), m) in d16.iter_mut().zip(a16).zip(b16).zip(m16) {
        let m0 = load_u8x16_i16(m);
        let (s0, s1) = madd_i16x16(
            load_i16x16(a),
            load_i16x16(b),
            m0,
            _mm256_sub_epi16(c64, m0),
        );
        store_i32x16_u16_clip(d, f(s0), f(s1), max_v);
    }
    let done = d16.len() * 16;
    let (d8, r8) = r16.as_chunks_mut::<8>();
    let (a8, ar) = t1[done..n].as_chunks::<8>();
    let (b8, br) = t2[done..n].as_chunks::<8>();
    let (m8, mr) = mask[done..n].as_chunks::<8>();
    for (((d, a), b), m) in d8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let mv = load_u8x8_i16(m);
        let (s0, s1) = madd_i16x8(load_i16x8(a), load_i16x8(b), mv, _mm_sub_epi16(c64_128, mv));
        let out = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(f128(s0)), f128(s1));
        store_i32x8_u16_clip(d, out, max_v);
    }
    for (((d, &a), &b), &m) in r8.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn blend_row_hbd_avx2(dst: &mut [u16], tmp: &[u16], mask: &[u8], n: usize) {
    // Exact `(dst*(64-m) + tmp*m + 32) >> 6`.  dav2d's HBD path keeps
    // the result in i32, but uses paired 16-bit products (`pmaddwd`) rather
    // than two full 32-bit multiplies.
    let c64 = _mm256_set1_epi16(64);
    let c64_128 = _mm_set1_epi16(64);
    let rnd_v = _mm256_set1_epi32(32);
    let max_v = _mm256_set1_epi32(0xffff);
    let f = |s: __m256i| _mm256_srai_epi32::<6>(_mm256_add_epi32(s, rnd_v));

    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (t16, _) = tmp[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for ((d, t), m) in d16.iter_mut().zip(t16).zip(m16) {
        let mv = load_u8x16_i16(m);
        let (s0, s1) = madd_i16x16(
            load_u16x16(&*d),
            load_u16x16(t),
            _mm256_sub_epi16(c64, mv),
            mv,
        );
        store_i32x16_u16_clip(d, f(s0), f(s1), max_v);
    }
    let done = d16.len() * 16;
    let (d8, r8) = r16.as_chunks_mut::<8>();
    let (t8, tr) = tmp[done..n].as_chunks::<8>();
    let (m8, mr) = mask[done..n].as_chunks::<8>();
    for ((d, t), m) in d8.iter_mut().zip(t8).zip(m8) {
        let mv = load_u8x8_i16(m);
        let (s0, s1) = madd_i16x8(
            load_u16x8(&*d),
            load_u16x8(t),
            _mm_sub_epi16(c64_128, mv),
            mv,
        );
        store_i32x8_u16_clip(d, f(m128x2_to_m256(s0, s1)), max_v);
    }
    for ((d, &t), &m) in r8.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn morph_row_hbd_avx2(
    dst: &mut [u16],
    alpha: i32,
    beta: i32,
    n: usize,
    bitdepth_max: i32,
) {
    if !(i16::MIN as i32..=i16::MAX as i32).contains(&alpha) {
        for d in dst[..n].iter_mut() {
            *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, bitdepth_max) as u16;
        }
        return;
    }

    // Exact `(alpha*dst + beta) >> 8`, keeping the i32 result but computing
    // products as `[pixel, 0] dot `[alpha, 0]` with `pmaddwd`.
    let coeff = _mm256_set1_epi32(alpha & 0xffff);
    let coeff128 = _mm_set1_epi32(alpha & 0xffff);
    let beta_v = _mm256_set1_epi32(beta);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let f = |s: __m256i| _mm256_srai_epi32::<8>(_mm256_add_epi32(s, beta_v));
    let zero = _mm256_setzero_si256();
    let zero128 = _mm_setzero_si128();

    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    for d in d16.iter_mut() {
        let (s0, s1) = madd_i16x16_const(load_u16x16(&*d), zero, coeff);
        store_i32x16_u16_clip(d, f(s0), f(s1), max_v);
    }
    let (d8, r8) = r16.as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        let (s0, s1) = madd_i16x8_const(load_u16x8(&*d), zero128, coeff128);
        store_i32x8_u16_clip(d, f(m128x2_to_m256(s0, s1)), max_v);
    }
    for d in r8.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, bitdepth_max) as u16;
    }
}

#[inline]
fn gdf_bitdepth_from_max(bitdepth_max: i32) -> i32 {
    if bitdepth_max <= 0xff {
        8
    } else if bitdepth_max <= 0x3ff {
        10
    } else {
        12
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gdf_adj_i16x16(e: __m256i, scale: __m256i, rnd: __m256i, sh: __m128i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let diff = _mm256_mullo_epi16(e, scale);
    let mag = _mm256_sra_epi16(_mm256_add_epi16(_mm256_abs_epi16(diff), rnd), sh);
    _mm256_blendv_epi8(
        mag,
        _mm256_sub_epi16(zero, mag),
        _mm256_cmpgt_epi16(zero, diff),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn i16x16_to_i32x8x2(v: __m256i) -> (__m256i, __m256i) {
    (
        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)),
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v)),
    )
}

/// High-bit-depth GDF residual add.
///
/// Matches dav2d/AVM scalar scaling: 10-bit residuals are shifted by 2 on add,
/// while 12-bit residuals are used at full precision.
#[target_feature(enable = "avx2")]
pub(crate) fn gdf_add_run_hbd_avx2(
    dst: &mut [u16],
    err: &[i8],
    scale: i32,
    n: usize,
    bitdepth_max: i32,
) {
    let bitdepth = gdf_bitdepth_from_max(bitdepth_max);
    let shift = 12 - bitdepth;
    let sc = _mm256_set1_epi16(scale as i16);
    let rnd = _mm256_set1_epi16(if shift == 0 { 0 } else { 1 << (shift - 1) });
    let sh = _mm_cvtsi32_si128(shift);
    let max_v = _mm256_set1_epi32(bitdepth_max);

    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (e16, er16) = err[..n].as_chunks::<16>();
    for (d, e) in d16.iter_mut().zip(e16) {
        let e = unsafe { _mm_loadu_si128(e.as_ptr().cast()) };
        let adj = gdf_adj_i16x16(_mm256_cvtepi8_epi16(e), sc, rnd, sh);
        let (a0, a1) = i16x16_to_i32x8x2(adj);
        let (d0, d1) = load_u16x16_i32x2(&*d);
        store_i32x16_u16_clip(d, _mm256_add_epi32(d0, a0), _mm256_add_epi32(d1, a1), max_v);
    }

    let (d8, tail_dst) = r16.as_chunks_mut::<8>();
    let (e8, tail_err) = er16.as_chunks::<8>();
    for (d, e) in d8.iter_mut().zip(e8) {
        let e = unsafe { _mm_loadl_epi64(e.as_ptr().cast()) };
        let adj = gdf_adj_i16x16(_mm256_cvtepi8_epi16(e), sc, rnd, sh);
        let a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(adj));
        let d0 = load_u16x8_i32(&*d);
        store_i32x8_u16_clip(d, _mm256_add_epi32(d0, a), max_v);
    }

    let rnd_scalar = if shift == 0 { 0 } else { 1 << (shift - 1) };
    for (d, &e) in tail_dst.iter_mut().zip(tail_err) {
        let diff = e as i32 * scale;
        let mag = (diff.abs() + rnd_scalar) >> shift;
        let adj = if diff < 0 { -mag } else { mag };
        *d = (*d as i32 + adj).clamp(0, bitdepth_max) as u16;
    }
}

/// HBD GDF gradient: per-column `|2*b - a - c|` summed over two rows,
/// then pair-reduced to up to four 2x2 output cells.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn gdf_gradient_group_hbd_avx2(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    center_rows: [&[u16]; 2],
    a_rows: [&[u16]; 2],
    c_rows: [&[u16]; 2],
    col0: usize,
    dx: i32,
    shift: u32,
) {
    let mut acc = _mm_setzero_si128();
    let sh = _mm_cvtsi32_si128(shift as i32);
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let b = unsafe { _mm_loadu_si128(center_rows[y].as_ptr().add(bcol).cast()) };
        let a = unsafe { _mm_loadu_si128(a_rows[y].as_ptr().add(acol).cast()) };
        let c = unsafe { _mm_loadu_si128(c_rows[y].as_ptr().add(ccol).cast()) };
        let b = _mm_srl_epi16(b, sh);
        let a = _mm_srl_epi16(a, sh);
        let c = _mm_srl_epi16(c, sh);
        let t = _mm_sub_epi16(_mm_sub_epi16(_mm_add_epi16(b, b), a), c);
        acc = _mm_add_epi16(acc, _mm_abs_epi16(t));
    }
    let pair = _mm_madd_epi16(acc, _mm_set1_epi16(1));
    store_gdf_gradient_cells_i32x4(dst, d, base_cell, ncells, pair);
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_gdf_gradient_cells_i32x4(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    v: __m128i,
) {
    if ncells > 0 {
        dst[base_cell][d] = _mm_cvtsi128_si32(v) as u16;
    }
    if ncells > 1 {
        dst[base_cell + 1][d] = _mm_extract_epi32::<1>(v) as u16;
    }
    if ncells > 2 {
        dst[base_cell + 2][d] = _mm_extract_epi32::<2>(v) as u16;
    }
    if ncells > 3 {
        dst[base_cell + 3][d] = _mm_extract_epi32::<3>(v) as u16;
    }
}

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
#[target_feature(enable = "avx2")]
fn gdf_load_pair_i32(row: &[u16], col: usize, sh: __m128i) -> __m128i {
    let raw = unsafe { core::ptr::read_unaligned(row.as_ptr().add(col).cast::<u32>()) } as i32;
    _mm_srl_epi32(_mm_cvtepu16_epi32(_mm_cvtsi32_si128(raw)), sh)
}

#[inline]
#[target_feature(enable = "avx2")]
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

/// HBD GDF prep inner 2-pixel pair.
///
/// The pair shares `cls` and gradient-derived `shared_vals`, so AVX2 is used
/// across the two x samples while the final LUT indexing remains scalar.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn gdf_prep_pair_hbd_avx2(
    rows: [&[u16]; 13],
    col: usize,
    cls: usize,
    shared_vals: [i32; 3],
    alpha_base: usize,
    weight_base: usize,
    error_lut_base: usize,
    scale: i32,
    down_shift: u32,
    up_scale: i32,
    ref_dst_idx: usize,
) -> [i8; 2] {
    let sh = _mm_cvtsi32_si128(down_shift as i32);
    let m = gdf_load_pair_i32(rows[6], col, sh);
    let up_scale_v = _mm_set1_epi32(up_scale);
    let v_lo = _mm_set1_epi32(-512);
    let v_hi = _mm_set1_epi32(511);
    let mut acc0 = _mm_set1_epi32(shared_vals[0]);
    let mut acc1 = _mm_set1_epi32(shared_vals[1]);
    let mut acc2 = _mm_set1_epi32(shared_vals[2]);

    for (k, &[dy, dx]) in GDF_PREP_COORDS.iter().enumerate() {
        let dy = dy as i32;
        let dx = dx as i32;
        let alpha = GDF_ALPHA[alpha_base + k * 4 + cls] as i32;
        let alpha_v = _mm_set1_epi32(alpha);
        let neg_alpha_v = _mm_set1_epi32(-alpha);
        let a_col = (col as i32 - dx) as usize;
        let b_col = (col as i32 + dx) as usize;
        let a = gdf_load_pair_i32(rows[(6 - dy) as usize], a_col, sh);
        let b = gdf_load_pair_i32(rows[(6 + dy) as usize], b_col, sh);
        let above = gdf_clip_i32x4(
            _mm_mullo_epi32(_mm_sub_epi32(a, m), up_scale_v),
            neg_alpha_v,
            alpha_v,
        );
        let below = gdf_clip_i32x4(
            _mm_mullo_epi32(_mm_sub_epi32(b, m), up_scale_v),
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
