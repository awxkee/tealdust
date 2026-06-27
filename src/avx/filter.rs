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
fn store_i32x8(a: &mut [i32; 8], v: __m256i) {
    unsafe { _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v) };
}

#[inline(always)]
fn load_i32x4(a: &[i32; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_i32x4(a: &mut [i32; 4], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline(always)]
fn load_u8x8_i32(a: &[u8; 8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x8_i32(a: &[i16; 8]) -> __m256i {
    unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x16_i16(a: &[u8; 16]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x32(a: &[u8; 32]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x32(a: &mut [u8; 32], v: __m256i) {
    unsafe { _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8_u8(a: &mut [u8; 8], v: __m256i) {
    unsafe {
        let z = _mm256_setzero_si256();
        let p16 = _mm256_packs_epi32(v, z);
        let p8 = _mm256_packus_epi16(p16, z);
        let lo = _mm256_castsi256_si128(p8);
        let hi = _mm256_extracti128_si256::<1>(p8);
        let out = _mm_unpacklo_epi32(lo, hi);
        _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x16_u8(a: &mut [u8; 16], lo: __m256i, hi: __m256i) {
    unsafe {
        let p16 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(lo, hi));
        let p8 = _mm256_packus_epi16(p16, p16);
        let l = _mm256_castsi256_si128(p8);
        let h = _mm256_extracti128_si256::<1>(p8);
        let out = _mm_unpacklo_epi64(l, h);
        _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16_u8(a: &mut [u8; 16], v: __m256i) {
    unsafe {
        let p8 = _mm256_packus_epi16(v, v);
        let l = _mm256_castsi256_si128(p8);
        let h = _mm256_extracti128_si256::<1>(p8);
        let out = _mm_unpacklo_epi64(l, h);
        _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn sra_i32(v: __m256i, shift: i32) -> __m256i {
    _mm256_sra_epi32(v, _mm_cvtsi32_si128(shift))
}

#[target_feature(enable = "avx2")]
pub(crate) fn residual_add_row_8bpc_avx2(
    dst: &mut [u8],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (c16, _) = c[..n].as_chunks::<16>();
    for (d, cv) in d16.iter_mut().zip(c16) {
        let cf_lo = sra_i32(
            _mm256_add_epi32(load_i32x8((&cv[..8]).try_into().unwrap()), rnd_v),
            shift,
        );
        let cf_hi = sra_i32(
            _mm256_add_epi32(load_i32x8((&cv[8..]).try_into().unwrap()), rnd_v),
            shift,
        );
        let d_lo = load_u8x8_i32((&d[..8]).try_into().unwrap());
        let d_hi = load_u8x8_i32((&d[8..]).try_into().unwrap());
        store_i32x16_u8(
            d,
            _mm256_add_epi32(d_lo, cf_lo),
            _mm256_add_epi32(d_hi, cf_hi),
        );
    }
    let done = d16.len() * 16;
    let (d8, r8) = r16.as_chunks_mut::<8>();
    let (c8, cr) = c[done..n].as_chunks::<8>();
    for (d, cv) in d8.iter_mut().zip(c8) {
        let cf = sra_i32(_mm256_add_epi32(load_i32x8(cv), rnd_v), shift);
        let dv = load_u8x8_i32(d);
        store_i32x8_u8(d, _mm256_add_epi32(dv, cf));
    }
    for (d, &cv) in r8.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dc_add_row_8bpc_avx2(dst: &mut [u8], dc: i32, n: usize) {
    if dc == 0 {
        return;
    }
    let amt = if dc > 0 {
        dc.min(255) as u8
    } else {
        dc.saturating_neg().min(255) as u8
    };
    let (c32, r32) = dst[..n].as_chunks_mut::<32>();
    let amt_v = _mm256_set1_epi8(amt as i8);
    if dc > 0 {
        for d in c32.iter_mut() {
            store_u8x32(d, _mm256_adds_epu8(load_u8x32(&*d), amt_v));
        }
        for d in r32.iter_mut() {
            *d = d.saturating_add(amt);
        }
    } else {
        for d in c32.iter_mut() {
            store_u8x32(d, _mm256_subs_epu8(load_u8x32(&*d), amt_v));
        }
        for d in r32.iter_mut() {
            *d = d.saturating_sub(amt);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn row_clip_avx2(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let min_v = _mm256_set1_epi32(min);
    let max_v = _mm256_set1_epi32(max);
    let clip = |v: __m256i| {
        _mm256_min_epi32(
            _mm256_max_epi32(sra_i32(_mm256_add_epi32(v, rnd_v), shift), min_v),
            max_v,
        )
    };
    let (c8, r8) = tmp[..n].as_chunks_mut::<8>();
    for ch in c8.iter_mut() {
        store_i32x8(ch, clip(load_i32x8(&*ch)));
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let rnd_s = _mm_set1_epi32(rnd);
    let sh_s = _mm_cvtsi32_si128(shift);
    let min_s = _mm_set1_epi32(min);
    let max_s = _mm_set1_epi32(max);
    for ch in c4.iter_mut() {
        let r = _mm_min_epi32(
            _mm_max_epi32(
                _mm_sra_epi32(_mm_add_epi32(load_i32x4(&*ch), rnd_s), sh_s),
                min_s,
            ),
            max_s,
        );
        store_i32x4(ch, r);
    }
    for t in r4.iter_mut() {
        *t = ((*t + rnd) >> shift).max(min).min(max);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn cctx_row_avx2(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    let sina_v = _mm256_set1_epi32(sina);
    let cosa_v = _mm256_set1_epi32(cosa);
    let c128 = _mm256_set1_epi32(128);
    let zero = _mm256_setzero_si256();
    let min_v = _mm256_set1_epi32(min);
    let max_v = _mm256_set1_epi32(max);
    let rot = |uu: __m256i, vv: __m256i| -> (__m256i, __m256i) {
        let a = _mm256_sub_epi32(
            _mm256_mullo_epi32(uu, cosa_v),
            _mm256_mullo_epi32(vv, sina_v),
        );
        let b = _mm256_add_epi32(
            _mm256_mullo_epi32(uu, sina_v),
            _mm256_mullo_epi32(vv, cosa_v),
        );
        let ra = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_add_epi32(a, c128),
            _mm256_cmpgt_epi32(zero, a),
        ));
        let rb = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_add_epi32(b, c128),
            _mm256_cmpgt_epi32(zero, b),
        ));
        (
            _mm256_min_epi32(_mm256_max_epi32(ra, min_v), max_v),
            _mm256_min_epi32(_mm256_max_epi32(rb, min_v), max_v),
        )
    };
    let (uc8, ur8) = u[..sz].as_chunks_mut::<8>();
    let (vc8, vr8) = v[..sz].as_chunks_mut::<8>();
    for (uch, vch) in uc8.iter_mut().zip(vc8.iter_mut()) {
        let (ra, rb) = rot(load_i32x8(&*uch), load_i32x8(&*vch));
        store_i32x8(uch, ra);
        store_i32x8(vch, rb);
    }
    for (uu, vv) in ur8.iter_mut().zip(vr8.iter_mut()) {
        let a = *uu * cosa - *vv * sina;
        let b = *uu * sina + *vv * cosa;
        *uu = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        *vv = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn avg_row_8bpc_avx2(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let f = |a: __m256i, b: __m256i| sra_i32(_mm256_add_epi32(_mm256_add_epi32(a, b), rnd_v), sh);
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in c16.iter_mut().zip(a16).zip(b16) {
        let lo = f(
            load_i16x8_i32((&a[..8]).try_into().unwrap()),
            load_i16x8_i32((&b[..8]).try_into().unwrap()),
        );
        let hi = f(
            load_i16x8_i32((&a[8..]).try_into().unwrap()),
            load_i16x8_i32((&b[8..]).try_into().unwrap()),
        );
        store_i32x16_u8(d, lo, hi);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, ar) = t1[done..n].as_chunks::<8>();
    let (b8, br) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        store_i32x8_u8(d, f(load_i16x8_i32(a), load_i16x8_i32(b)));
    }
    for ((d, &a), &b) in r8.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 + b as i32 + rnd) >> sh).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn w_avg_row_8bpc_avx2(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    let w1 = _mm256_set1_epi32(weight);
    let w2 = _mm256_set1_epi32(16 - weight);
    let rnd_v = _mm256_set1_epi32(rnd);
    let f = |a: __m256i, b: __m256i| {
        sra_i32(
            _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(a, w1), _mm256_mullo_epi32(b, w2)),
                rnd_v,
            ),
            sh,
        )
    };
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in c16.iter_mut().zip(a16).zip(b16) {
        let lo = f(
            load_i16x8_i32((&a[..8]).try_into().unwrap()),
            load_i16x8_i32((&b[..8]).try_into().unwrap()),
        );
        let hi = f(
            load_i16x8_i32((&a[8..]).try_into().unwrap()),
            load_i16x8_i32((&b[8..]).try_into().unwrap()),
        );
        store_i32x16_u8(d, lo, hi);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, ar) = t1[done..n].as_chunks::<8>();
    let (b8, br) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        store_i32x8_u8(d, f(load_i16x8_i32(a), load_i16x8_i32(b)));
    }
    for ((d, &a), &b) in r8.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn mask_row_8bpc_avx2(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = _mm256_set1_epi32(rnd);
    let c64 = _mm256_set1_epi32(64);
    let f = |a: __m256i, b: __m256i, m: __m256i| {
        sra_i32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(a, m),
                    _mm256_mullo_epi32(b, _mm256_sub_epi32(c64, m)),
                ),
                rnd_v,
            ),
            sh,
        )
    };
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for (((d, a), b), m) in c16.iter_mut().zip(a16).zip(b16).zip(m16) {
        let lo = f(
            load_i16x8_i32((&a[..8]).try_into().unwrap()),
            load_i16x8_i32((&b[..8]).try_into().unwrap()),
            load_u8x8_i32((&m[..8]).try_into().unwrap()),
        );
        let hi = f(
            load_i16x8_i32((&a[8..]).try_into().unwrap()),
            load_i16x8_i32((&b[8..]).try_into().unwrap()),
            load_u8x8_i32((&m[8..]).try_into().unwrap()),
        );
        store_i32x16_u8(d, lo, hi);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, ar) = t1[done..n].as_chunks::<8>();
    let (b8, br) = t2[done..n].as_chunks::<8>();
    let (m8, mr) = mask[done..n].as_chunks::<8>();
    for (((d, a), b), m) in c8.iter_mut().zip(a8).zip(b8).zip(m8) {
        store_i32x8_u8(d, f(load_i16x8_i32(a), load_i16x8_i32(b), load_u8x8_i32(m)));
    }
    for (((d, &a), &b), &m) in r8.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn blend_row_8bpc_avx2(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    let c64 = _mm256_set1_epi16(64);
    let rnd_v = _mm256_set1_epi16(32);
    let f = |d: __m256i, t: __m256i, m: __m256i| {
        _mm256_srai_epi16::<6>(_mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mullo_epi16(d, _mm256_sub_epi16(c64, m)),
                _mm256_mullo_epi16(t, m),
            ),
            rnd_v,
        ))
    };
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (t16, _) = tmp[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for ((d, t), m) in c16.iter_mut().zip(t16).zip(m16) {
        store_i16x16_u8(
            d,
            f(load_u8x16_i16(&*d), load_u8x16_i16(t), load_u8x16_i16(m)),
        );
    }
    for ((d, &t), &m) in r16
        .iter_mut()
        .zip(&tmp[c16.len() * 16..n])
        .zip(&mask[c16.len() * 16..n])
    {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn morph_row_8bpc_avx2(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    let a_v = _mm256_set1_epi32(alpha);
    let b_v = _mm256_set1_epi32(beta);
    let f = |d: __m256i| _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_mullo_epi32(d, a_v), b_v));
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    for d in c16.iter_mut() {
        let lo = f(load_u8x8_i32((&d[..8]).try_into().unwrap()));
        let hi = f(load_u8x8_i32((&d[8..]).try_into().unwrap()));
        store_i32x16_u8(d, lo, hi);
    }
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for d in c8.iter_mut() {
        store_i32x8_u8(d, f(load_u8x8_i32(&*d)));
    }
    for d in r8.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, 255) as u8;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i8x16_i16(a: &[i8]) -> __m256i {
    unsafe { _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x16_i16_slice(a: &[u8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(a.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x16_from_i16(a: &mut [u8], v: __m256i) {
    unsafe {
        let p8 = _mm256_packus_epi16(v, v);
        let lo = _mm256_castsi256_si128(p8);
        let hi = _mm256_extracti128_si256::<1>(p8);
        _mm_storeu_si128(a.as_mut_ptr().cast(), _mm_unpacklo_epi64(lo, hi));
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn gdf_add_run_8bpc_avx2(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let sc = _mm256_set1_epi16(scale as i16);
    let rnd = _mm256_set1_epi16(8);
    let zero = _mm256_setzero_si256();
    let adj = |e: __m256i| {
        let diff = _mm256_mullo_epi16(e, sc);
        let mag = _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_abs_epi16(diff), rnd));
        _mm256_blendv_epi8(
            mag,
            _mm256_sub_epi16(zero, mag),
            _mm256_cmpgt_epi16(zero, diff),
        )
    };

    let (dst16, dst_rem16) = dst[..n].as_chunks_mut::<16>();
    let (err16, err_rem16) = err[..n].as_chunks::<16>();
    for (d, e) in dst16.iter_mut().zip(err16) {
        let a = adj(load_i8x16_i16(e));
        let d0 = load_u8x16_i16_slice(&*d);
        store_u8x16_from_i16(d, _mm256_add_epi16(d0, a));
    }

    let (dst8, dst_tail) = dst_rem16.as_chunks_mut::<8>();
    let (err8, err_tail) = err_rem16.as_chunks::<8>();
    for (d, e) in dst8.iter_mut().zip(err8) {
        let a = adj(_mm256_cvtepi8_epi16(unsafe {
            _mm_loadl_epi64(e.as_ptr().cast())
        }));
        let d0 = _mm256_cvtepu8_epi16(unsafe { _mm_loadl_epi64(d.as_ptr().cast()) });
        let out = _mm256_add_epi16(d0, a);
        unsafe {
            let p8 = _mm256_packus_epi16(out, out);
            _mm_storel_epi64(d.as_mut_ptr().cast(), _mm256_castsi256_si128(p8));
        }
    }

    for (d, &e) in dst_tail.iter_mut().zip(err_tail) {
        let diff = e as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let a = if diff < 0 { -mag } else { mag };
        *d = (*d as i32 + a).clamp(0, 255) as u8;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn gdf_gradient_group_avx2(
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
    let mut acc = _mm_setzero_si128();
    let sh = _mm_cvtsi32_si128(shift as i32);
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let b = unsafe { _mm_loadl_epi64(center_rows[y].as_ptr().add(bcol).cast()) };
        let a = unsafe { _mm_loadl_epi64(a_rows[y].as_ptr().add(acol).cast()) };
        let c = unsafe { _mm_loadl_epi64(c_rows[y].as_ptr().add(ccol).cast()) };
        let b = _mm_srl_epi16(_mm_cvtepu8_epi16(b), sh);
        let a = _mm_srl_epi16(_mm_cvtepu8_epi16(a), sh);
        let c = _mm_srl_epi16(_mm_cvtepu8_epi16(c), sh);
        let t = _mm_sub_epi16(_mm_sub_epi16(_mm_add_epi16(b, b), a), c);
        acc = _mm_add_epi16(acc, _mm_abs_epi16(t));
    }
    let pair = _mm_madd_epi16(acc, _mm_set1_epi16(1));
    let mut out = [0i32; 4];
    unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), pair) };
    for k in 0..ncells {
        dst[base_cell + k][d] = out[k] as u16;
    }
}

/// cctx rotate+clip over two i16 coefficient planes, widening only inside the SIMD arithmetic.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cctx_row_i16_avx2(
    u: &mut [i16],
    v: &mut [i16],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let a_pair = ((cosa as i16 as u16) as u32 | (((-sina) as i16 as u16) as u32) << 16) as i32;
        let b_pair = ((sina as i16 as u16) as u32 | ((cosa as i16 as u16) as u32) << 16) as i32;
        let a_pair_v = _mm256_set1_epi32(a_pair);
        let b_pair_v = _mm256_set1_epi32(b_pair);
        let c128 = _mm256_set1_epi32(128);
        let zero = _mm256_setzero_si256();
        let min_v = _mm256_set1_epi32(min);
        let max_v = _mm256_set1_epi32(max);
        let (u_chunks, ur) = u[..sz].as_chunks_mut::<8>();
        let (v_chunks, vr) = v[..sz].as_chunks_mut::<8>();
        for (uch, vch) in u_chunks.iter_mut().zip(v_chunks.iter_mut()) {
            let uu16 = _mm_loadu_si128(uch.as_ptr() as *const __m128i);
            let vv16 = _mm_loadu_si128(vch.as_ptr() as *const __m128i);
            let uv_lo = _mm_unpacklo_epi16(uu16, vv16);
            let uv_hi = _mm_unpackhi_epi16(uu16, vv16);
            let uv = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(uv_lo), uv_hi);
            let a = _mm256_madd_epi16(uv, a_pair_v);
            let b = _mm256_madd_epi16(uv, b_pair_v);
            let ru = _mm256_min_epi32(
                _mm256_max_epi32(
                    _mm256_srai_epi32::<8>(_mm256_add_epi32(
                        _mm256_add_epi32(a, c128),
                        _mm256_cmpgt_epi32(zero, a),
                    )),
                    min_v,
                ),
                max_v,
            );
            let rv = _mm256_min_epi32(
                _mm256_max_epi32(
                    _mm256_srai_epi32::<8>(_mm256_add_epi32(
                        _mm256_add_epi32(b, c128),
                        _mm256_cmpgt_epi32(zero, b),
                    )),
                    min_v,
                ),
                max_v,
            );
            let pu =
                _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(ru, _mm256_setzero_si256()));
            let pv =
                _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(rv, _mm256_setzero_si256()));
            _mm_storeu_si128(uch.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(pu));
            _mm_storeu_si128(vch.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(pv));
        }
        for (uu, vv) in ur.iter_mut().zip(vr.iter_mut()) {
            let ui = *uu as i32;
            let vi = *vv as i32;
            let a = ui * cosa - vi * sina;
            let b = ui * sina + vi * cosa;
            *uu = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max) as i16;
            *vv = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max) as i16;
        }
    }
}
