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

#[inline(always)]
fn load_i32x8(a: &[i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn load_i16x8_i32(a: &[i16; 8]) -> __m256i {
    unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn load_u16x8_i32(a: &[u16; 8]) -> __m256i {
    unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn load_u8x8_i32(a: &[u8; 8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i32x8_u16_clip(a: &mut [u16; 8], v: __m256i, max_v: __m256i) {
    let v = unsafe { _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v) };
    let lo = unsafe { _mm256_castsi256_si128(v) };
    let hi = unsafe { _mm256_extracti128_si256::<1>(v) };
    let p = unsafe { _mm_packus_epi32(lo, hi) };
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, p) };
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn residual_add_row_hbd_avx2(
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
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (c8, cr) = c[..n].as_chunks::<8>();
    for (d, cv) in d8.iter_mut().zip(c8) {
        let dv = load_u16x8_i32(&*d);
        let cf = _mm256_sra_epi32(_mm256_add_epi32(load_i32x8(cv), rnd_v), shc);
        store_i32x8_u16_clip(d, _mm256_add_epi32(dv, cf), max_v);
    }
    for (d, &cv) in r8.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dc_add_row_hbd_avx2(dst: &mut [u16], dc: i32, n: usize, bitdepth_max: i32) {
    if dc == 0 {
        return;
    }
    let dc_v = _mm256_set1_epi32(dc);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        store_i32x8_u16_clip(d, _mm256_add_epi32(load_u16x8_i32(&*d), dc_v), max_v);
    }
    for d in r8.iter_mut() {
        *d = ((*d as i32) + dc).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avg_row_hbd_avx2(
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
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, ar) = t1[..n].as_chunks::<8>();
    let (b8, br) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in d8.iter_mut().zip(a8).zip(b8) {
        let v = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(load_i16x8_i32(a), load_i16x8_i32(b)),
                rnd_v,
            ),
            shc,
        );
        store_i32x8_u16_clip(d, v, max_v);
    }
    for ((d, &a), &b) in r8.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 + b as i32 + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn w_avg_row_hbd_avx2(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let w1 = _mm256_set1_epi32(weight);
    let w2 = _mm256_set1_epi32(16 - weight);
    let rnd_v = _mm256_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, ar) = t1[..n].as_chunks::<8>();
    let (b8, br) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in d8.iter_mut().zip(a8).zip(b8) {
        let v = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(load_i16x8_i32(a), w1),
                    _mm256_mullo_epi32(load_i16x8_i32(b), w2),
                ),
                rnd_v,
            ),
            shc,
        );
        store_i32x8_u16_clip(d, v, max_v);
    }
    for ((d, &a), &b) in r8.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, bitdepth_max)
            as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn mask_row_hbd_avx2(
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
    let c64 = _mm256_set1_epi32(64);
    let shc = _mm_cvtsi32_si128(sh);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, ar) = t1[..n].as_chunks::<8>();
    let (b8, br) = t2[..n].as_chunks::<8>();
    let (m8, mr) = mask[..n].as_chunks::<8>();
    for (((d, a), b), m) in d8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let m = load_u8x8_i32(m);
        let v = _mm256_sra_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(load_i16x8_i32(a), m),
                    _mm256_mullo_epi32(load_i16x8_i32(b), _mm256_sub_epi32(c64, m)),
                ),
                rnd_v,
            ),
            shc,
        );
        store_i32x8_u16_clip(d, v, max_v);
    }
    for (((d, &a), &b), &m) in r8.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn blend_row_hbd_avx2(dst: &mut [u16], tmp: &[u16], mask: &[u8], n: usize) {
    let c64 = _mm256_set1_epi32(64);
    let rnd_v = _mm256_set1_epi32(32);
    let max_v = _mm256_set1_epi32(0xffff);
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (t8, tr) = tmp[..n].as_chunks::<8>();
    let (m8, mr) = mask[..n].as_chunks::<8>();
    for ((d, t), m) in d8.iter_mut().zip(t8).zip(m8) {
        let m = load_u8x8_i32(m);
        let v = _mm256_srai_epi32::<6>(_mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(load_u16x8_i32(&*d), _mm256_sub_epi32(c64, m)),
                _mm256_mullo_epi32(load_u16x8_i32(t), m),
            ),
            rnd_v,
        ));
        store_i32x8_u16_clip(d, v, max_v);
    }
    for ((d, &t), &m) in r8.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u16;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn morph_row_hbd_avx2(
    dst: &mut [u16],
    alpha: i32,
    beta: i32,
    n: usize,
    bitdepth_max: i32,
) {
    let a_v = _mm256_set1_epi32(alpha);
    let b_v = _mm256_set1_epi32(beta);
    let max_v = _mm256_set1_epi32(bitdepth_max);
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        let v = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(load_u16x8_i32(&*d), a_v),
            b_v,
        ));
        store_i32x8_u16_clip(d, v, max_v);
    }
    for d in r8.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, bitdepth_max) as u16;
    }
}
