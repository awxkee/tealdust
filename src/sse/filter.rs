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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
fn load_i16x4_i32(a: &[i16; 4]) -> __m128i {
    unsafe { _mm_cvtepi16_epi32(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u8x4_i32(a: &[u8; 4]) -> __m128i {
    _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(*a)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i8x4_i32(a: &[i8; 4]) -> __m128i {
    let bytes = [a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8];
    _mm_cvtepi8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(bytes)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u8x8_i32x2(a: &[u8; 8]) -> (__m128i, __m128i) {
    let v = unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) };
    (
        _mm_cvtepu8_epi32(v),
        _mm_cvtepu8_epi32(_mm_srli_si128(v, 4)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u8x16_i32x4(a: &[u8; 16]) -> (__m128i, __m128i, __m128i, __m128i) {
    unsafe {
        let v = _mm_loadu_si128(a.as_ptr().cast());
        (
            _mm_cvtepu8_epi32(v),
            _mm_cvtepu8_epi32(_mm_srli_si128(v, 4)),
            _mm_cvtepu8_epi32(_mm_srli_si128(v, 8)),
            _mm_cvtepu8_epi32(_mm_srli_si128(v, 12)),
        )
    }
}
#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i8x8_i32x2(a: &[i8; 8]) -> (__m128i, __m128i) {
    let v = unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) };
    (
        _mm_cvtepi8_epi32(v),
        _mm_cvtepi8_epi32(_mm_srli_si128(v, 4)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i8x16_i32x4(a: &[i8; 16]) -> (__m128i, __m128i, __m128i, __m128i) {
    unsafe {
        let v = _mm_loadu_si128(a.as_ptr().cast());
        (
            _mm_cvtepi8_epi32(v),
            _mm_cvtepi8_epi32(_mm_srli_si128(v, 4)),
            _mm_cvtepi8_epi32(_mm_srli_si128(v, 8)),
            _mm_cvtepi8_epi32(_mm_srli_si128(v, 12)),
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x8_i32x2(a: &[i16; 8]) -> (__m128i, __m128i) {
    let v = unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) };
    (
        _mm_cvtepi16_epi32(v),
        _mm_cvtepi16_epi32(_mm_srli_si128(v, 8)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x16_i32x4(a: &[i16; 16]) -> (__m128i, __m128i, __m128i, __m128i) {
    unsafe {
        let lo = _mm_loadu_si128(a.as_ptr().cast());
        let hi = _mm_loadu_si128(a.as_ptr().add(8).cast());
        (
            _mm_cvtepi16_epi32(lo),
            _mm_cvtepi16_epi32(_mm_srli_si128(lo, 8)),
            _mm_cvtepi16_epi32(hi),
            _mm_cvtepi16_epi32(_mm_srli_si128(hi, 8)),
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x8(a: &[i16; 8]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_i16x4(a: &[i16; 4]) -> __m128i {
    unsafe { _mm_loadl_epi64(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u8x4_i16(a: &[u8; 4]) -> __m128i {
    _mm_cvtepu8_epi16(_mm_cvtsi32_si128(i32::from_le_bytes(*a)))
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
fn madd_i16x4_const(a: __m128i, b: __m128i, coeff: __m128i) -> __m128i {
    _mm_madd_epi16(_mm_unpacklo_epi16(a, b), coeff)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn madd_i16x8(a: __m128i, b: __m128i, w1: __m128i, w2: __m128i) -> (__m128i, __m128i) {
    (
        _mm_madd_epi16(_mm_unpacklo_epi16(a, b), _mm_unpacklo_epi16(w1, w2)),
        _mm_madd_epi16(_mm_unpackhi_epi16(a, b), _mm_unpackhi_epi16(w1, w2)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn madd_i16x4(a: __m128i, b: __m128i, w1: __m128i, w2: __m128i) -> __m128i {
    _mm_madd_epi16(_mm_unpacklo_epi16(a, b), _mm_unpacklo_epi16(w1, w2))
}

#[inline(always)]
fn load_i32x4(a: &[i32; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_i32x4(a: &mut [i32; 4], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x4_u8(a: &mut [u8; 4], v: __m128i) {
    let p16 = _mm_packs_epi32(v, v);
    let p8 = _mm_packus_epi16(p16, p16);
    *a = (_mm_cvtsi128_si32(p8) as u32).to_le_bytes();
}

/// Pack two i32x4 lanes (lo, hi) to 8 clamped u8 and store.
#[inline(always)]
fn store_i32x8_u8(a: &mut [u8; 8], lo: __m128i, hi: __m128i) {
    let p16 = unsafe { _mm_packs_epi32(lo, hi) };
    let p8 = unsafe { _mm_packus_epi16(p16, p16) };
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, p8) };
}

#[inline(always)]
fn pack_i32x8_u8(lo: __m128i, hi: __m128i) -> __m128i {
    let p16 = unsafe { _mm_packs_epi32(lo, hi) };
    unsafe { _mm_packus_epi16(p16, p16) }
}

#[inline(always)]
fn store_i32x16_u8(a: &mut [u8; 16], v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) {
    let lo = pack_i32x8_u8(v0, v1);
    let hi = pack_i32x8_u8(v2, v3);
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, _mm_unpacklo_epi64(lo, hi)) };
}

#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_u8x16(a: &mut [u8; 16], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline(always)]
fn load_u8x8(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: __m128i) {
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline(always)]
fn load_u8x8_i16(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

/// One 16-byte load split into two i16x8 lanes (low 8, high 8), replacing two
/// adjacent 8-byte loads of a contiguous `[u8; 16]` chunk.
#[inline(always)]
fn load_u8x16_i16x2(a: &[u8; 16]) -> (__m128i, __m128i) {
    let v = unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) };
    unsafe {
        (
            _mm_cvtepu8_epi16(v),
            _mm_cvtepu8_epi16(_mm_srli_si128(v, 8)),
        )
    }
}

#[inline(always)]
fn store_i16x8_u8(a: &mut [u8; 8], v: __m128i) {
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, _mm_packus_epi16(v, v)) };
}

#[inline(always)]
fn store_i16x8x2_u8(a: &mut [u8; 16], lo: __m128i, hi: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, _mm_packus_epi16(lo, hi)) };
}

/// 8-bit residual add: `dst[i] = clip(dst[i] + ((c[i] + rnd) >> shift), 0, 255)`.
/// i32 lanes, 2x-unrolled to 8 px/iter (two i32x4), then a 4-px tail and scalar.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn residual_add_row_8bpc_sse41(
    dst: &mut [u8],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(shift);
    let f = |cv: __m128i| _mm_sra_epi32(_mm_add_epi32(cv, rnd_v), shc);
    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (cc16, _) = c[..n].as_chunks::<16>();
    for (d, cv) in d16.iter_mut().zip(cc16) {
        let c0 = f(load_i32x4((&cv[0..4]).try_into().unwrap()));
        let c1 = f(load_i32x4((&cv[4..8]).try_into().unwrap()));
        let c2 = f(load_i32x4((&cv[8..12]).try_into().unwrap()));
        let c3 = f(load_i32x4((&cv[12..16]).try_into().unwrap()));
        let (d0, d1, d2, d3) = load_u8x16_i32x4(&*d);
        store_i32x16_u8(
            d,
            _mm_add_epi32(d0, c0),
            _mm_add_epi32(d1, c1),
            _mm_add_epi32(d2, c2),
            _mm_add_epi32(d3, c3),
        );
    }
    let done = d16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (cc8, _) = c[done..n].as_chunks::<8>();
    for (d, cv) in c8.iter_mut().zip(cc8) {
        let cf_lo = f(load_i32x4((&cv[..4]).try_into().unwrap()));
        let cf_hi = f(load_i32x4((&cv[4..]).try_into().unwrap()));
        let (d_lo, d_hi) = load_u8x8_i32x2(&*d);
        store_i32x8_u8(d, _mm_add_epi32(d_lo, cf_lo), _mm_add_epi32(d_hi, cf_hi));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (cc4, cr) = c[done..n].as_chunks::<4>();
    for (d, cv) in c4.iter_mut().zip(cc4) {
        let cf = f(load_i32x4(cv));
        let dv = load_u8x4_i32(d);
        store_i32x4_u8(d, _mm_add_epi32(dv, cf));
    }
    for (d, &cv) in r4.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, 255) as u8;
    }
}

/// `dst[i] = clip(dst[i] + dc, 0, 255)`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn dc_add_row_8bpc_sse41(dst: &mut [u8], dc: i32, n: usize) {
    if dc == 0 {
        return;
    }

    let amt = if dc > 0 {
        dc.min(255) as u8
    } else {
        dc.saturating_neg().min(255) as u8
    };

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (c8, r8) = r16.as_chunks_mut::<8>();

    if dc > 0 {
        let amt_v = _mm_set1_epi8(amt as i8);
        for d in c16.iter_mut() {
            store_u8x16(d, _mm_adds_epu8(load_u8x16(&*d), amt_v));
        }
        for d in c8.iter_mut() {
            store_u8x8(d, _mm_adds_epu8(load_u8x8(&*d), amt_v));
        }
        for d in r8.iter_mut() {
            *d = d.saturating_add(amt);
        }
    } else {
        let amt_v = _mm_set1_epi8(amt as i8);
        for d in c16.iter_mut() {
            store_u8x16(d, _mm_subs_epu8(load_u8x16(&*d), amt_v));
        }
        for d in c8.iter_mut() {
            store_u8x8(d, _mm_subs_epu8(load_u8x8(&*d), amt_v));
        }
        for d in r8.iter_mut() {
            *d = d.saturating_sub(amt);
        }
    }
}

/// itx row-clip: `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)` (i32 in/out).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn row_clip_sse41(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(shift);
    let min_v = _mm_set1_epi32(min);
    let max_v = _mm_set1_epi32(max);
    let clip = |v: __m128i| {
        _mm_min_epi32(
            _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(v, rnd_v), shc), min_v),
            max_v,
        )
    };
    let (c8, r8) = tmp[..n].as_chunks_mut::<8>();
    for ch in c8.iter_mut() {
        let r_lo = clip(load_i32x4((&ch[..4]).try_into().unwrap()));
        let r_hi = clip(load_i32x4((&ch[4..]).try_into().unwrap()));
        store_i32x4((&mut ch[..4]).try_into().unwrap(), r_lo);
        store_i32x4((&mut ch[4..]).try_into().unwrap(), r_hi);
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for ch in c4.iter_mut() {
        let r = clip(load_i32x4(ch));
        store_i32x4(ch, r);
    }
    for t in r4.iter_mut() {
        *t = ((*t + rnd) >> shift).max(min).min(max);
    }
}

/// cctx rotate+clip over two i32 planes. `cmpgt(0, a)` is the `-1` mask where
/// `a < 0`, so `a + 128 + mask == a + 128 - (a < 0)`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cctx_row_sse41(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    let sina_v = _mm_set1_epi32(sina);
    let cosa_v = _mm_set1_epi32(cosa);
    let c128 = _mm_set1_epi32(128);
    let zero = _mm_setzero_si128();
    let min_v = _mm_set1_epi32(min);
    let max_v = _mm_set1_epi32(max);
    let rot = |uu: __m128i, vv: __m128i| -> (__m128i, __m128i) {
        {
            let a = _mm_sub_epi32(_mm_mullo_epi32(uu, cosa_v), _mm_mullo_epi32(vv, sina_v));
            let b = _mm_add_epi32(_mm_mullo_epi32(uu, sina_v), _mm_mullo_epi32(vv, cosa_v));
            let ra = _mm_srai_epi32::<8>(_mm_add_epi32(
                _mm_add_epi32(a, c128),
                _mm_cmpgt_epi32(zero, a),
            ));
            let rb = _mm_srai_epi32::<8>(_mm_add_epi32(
                _mm_add_epi32(b, c128),
                _mm_cmpgt_epi32(zero, b),
            ));
            (
                _mm_min_epi32(_mm_max_epi32(ra, min_v), max_v),
                _mm_min_epi32(_mm_max_epi32(rb, min_v), max_v),
            )
        }
    };
    let (uc8, ur8) = u[..sz].as_chunks_mut::<8>();
    let (vc8, vr8) = v[..sz].as_chunks_mut::<8>();
    for (uch, vch) in uc8.iter_mut().zip(vc8.iter_mut()) {
        let u_lo = load_i32x4((&uch[..4]).try_into().unwrap());
        let u_hi = load_i32x4((&uch[4..]).try_into().unwrap());
        let v_lo = load_i32x4((&vch[..4]).try_into().unwrap());
        let v_hi = load_i32x4((&vch[4..]).try_into().unwrap());
        let (ra_lo, rb_lo) = rot(u_lo, v_lo);
        let (ra_hi, rb_hi) = rot(u_hi, v_hi);
        store_i32x4((&mut uch[..4]).try_into().unwrap(), ra_lo);
        store_i32x4((&mut uch[4..]).try_into().unwrap(), ra_hi);
        store_i32x4((&mut vch[..4]).try_into().unwrap(), rb_lo);
        store_i32x4((&mut vch[4..]).try_into().unwrap(), rb_hi);
    }
    let (uc4, ur4) = ur8.as_chunks_mut::<4>();
    let (vc4, vr4) = vr8.as_chunks_mut::<4>();
    for (uch, vch) in uc4.iter_mut().zip(vc4.iter_mut()) {
        let (ra, rb) = rot(load_i32x4(uch), load_i32x4(vch));
        store_i32x4(uch, ra);
        store_i32x4(vch, rb);
    }
    for (uu, vv) in ur4.iter_mut().zip(vr4.iter_mut()) {
        let a = *uu * cosa - *vv * sina;
        let b = *uu * sina + *vv * cosa;
        *uu = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        *vv = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
    }
}

/// `dst[x] = clip((t1[x] + t2[x] + rnd) >> sh, 0, 255)`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn avg_row_8bpc_sse41(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let f = |a: __m128i, b: __m128i| _mm_sra_epi32(_mm_add_epi32(_mm_add_epi32(a, b), rnd_v), shc);
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in c16.iter_mut().zip(a16).zip(b16) {
        let (a0, a1, a2, a3) = load_i16x16_i32x4(a);
        let (b0, b1, b2, b3) = load_i16x16_i32x4(b);
        store_i32x16_u8(d, f(a0, b0), f(a1, b1), f(a2, b2), f(a3, b3));
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, _) = t1[done..n].as_chunks::<8>();
    let (b8, _) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        store_i32x8_u8(d, f(a0, b0), f(a1, b1));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    for ((d, a), b) in c4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u8(d, f(load_i16x4_i32(a), load_i16x4_i32(b)));
    }
    for ((d, &a), &b) in r4.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 + b as i32 + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = clip((t1[x]*weight + t2[x]*(16-weight) + rnd) >> sh, 0, 255)`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn w_avg_row_8bpc_sse41(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    // Exact `(a*weight + b*(16-weight) + rnd) >> sh` with one `pmaddwd`
    // per 4 pixels instead of widening both terms and using two `pmulld`s.
    let coeff = _mm_set1_epi32(((16 - weight) << 16) | (weight & 0xffff));
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let f = |s: __m128i| _mm_sra_epi32(_mm_add_epi32(s, rnd_v), shc);

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in c16.iter_mut().zip(a16).zip(b16) {
        let (s0, s1) = madd_i16x8_const(
            load_i16x8((&a[..8]).try_into().unwrap()),
            load_i16x8((&b[..8]).try_into().unwrap()),
            coeff,
        );
        let (s2, s3) = madd_i16x8_const(
            load_i16x8((&a[8..]).try_into().unwrap()),
            load_i16x8((&b[8..]).try_into().unwrap()),
            coeff,
        );
        store_i32x16_u8(d, f(s0), f(s1), f(s2), f(s3));
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, _) = t1[done..n].as_chunks::<8>();
    let (b8, _) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let (s0, s1) = madd_i16x8_const(load_i16x8(a), load_i16x8(b), coeff);
        store_i32x8_u8(d, f(s0), f(s1));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    for ((d, a), b) in c4.iter_mut().zip(a4).zip(b4) {
        let s = madd_i16x4_const(load_i16x4(a), load_i16x4(b), coeff);
        store_i32x4_u8(d, f(s));
    }
    for ((d, &a), &b) in r4.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = clip((t1[x]*m + t2[x]*(64-m) + rnd) >> sh, 0, 255)`, `m = mask[x]`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn mask_row_8bpc_sse41(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    // Exact `(a*m + b*(64-m) + rnd) >> sh` with `pmaddwd`, matching
    // dav2d's paired-product shape and avoiding 32-bit `pmulld`.
    let rnd_v = _mm_set1_epi32(rnd);
    let c64 = _mm_set1_epi16(64);
    let shc = _mm_cvtsi32_si128(sh);
    let f = |s: __m128i| _mm_sra_epi32(_mm_add_epi32(s, rnd_v), shc);

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for (((d, a), b), m) in c16.iter_mut().zip(a16).zip(b16).zip(m16) {
        let (m0, m1) = load_u8x16_i16x2(m);
        let (s0, s1) = madd_i16x8(
            load_i16x8((&a[..8]).try_into().unwrap()),
            load_i16x8((&b[..8]).try_into().unwrap()),
            m0,
            _mm_sub_epi16(c64, m0),
        );
        let (s2, s3) = madd_i16x8(
            load_i16x8((&a[8..]).try_into().unwrap()),
            load_i16x8((&b[8..]).try_into().unwrap()),
            m1,
            _mm_sub_epi16(c64, m1),
        );
        store_i32x16_u8(d, f(s0), f(s1), f(s2), f(s3));
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, _) = t1[done..n].as_chunks::<8>();
    let (b8, _) = t2[done..n].as_chunks::<8>();
    let (m8, _) = mask[done..n].as_chunks::<8>();
    for (((d, a), b), m) in c8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let mv = load_u8x8_i16(m);
        let (s0, s1) = madd_i16x8(load_i16x8(a), load_i16x8(b), mv, _mm_sub_epi16(c64, mv));
        store_i32x8_u8(d, f(s0), f(s1));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    for (((d, a), b), m) in c4.iter_mut().zip(a4).zip(b4).zip(m4) {
        let mv = load_u8x4_i16(m);
        let s = madd_i16x4(load_i16x4(a), load_i16x4(b), mv, _mm_sub_epi16(c64, mv));
        store_i32x4_u8(d, f(s));
    }
    for (((d, &a), &b), &m) in r4.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6`, `m = mask[x]`. The weighted
/// average stays in [0,255] so it fits i16 lanes: 2x-unrolled to 16 px/iter.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn blend_row_8bpc_sse41(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    let c64 = _mm_set1_epi16(64);
    let rnd_v = _mm_set1_epi16(32);
    let f = |d: __m128i, t: __m128i, m: __m128i| {
        _mm_srai_epi16::<6>(_mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(d, _mm_sub_epi16(c64, m)),
                _mm_mullo_epi16(t, m),
            ),
            rnd_v,
        ))
    };
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (t16, _) = tmp[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for ((d, t), m) in c16.iter_mut().zip(t16).zip(m16) {
        let (d0, d1) = load_u8x16_i16x2(&*d);
        let (t0, t1) = load_u8x16_i16x2(t);
        let (m0, m1) = load_u8x16_i16x2(m);
        let o0 = f(d0, t0, m0);
        let o1 = f(d1, t1, m1);
        store_i16x8x2_u8(d, o0, o1);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (t8, tr) = tmp[done..n].as_chunks::<8>();
    let (m8, mr) = mask[done..n].as_chunks::<8>();
    for ((d, t), m) in c8.iter_mut().zip(t8).zip(m8) {
        let o = f(load_u8x8_i16(d), load_u8x8_i16(t), load_u8x8_i16(m));
        store_i16x8_u8(d, o);
    }
    for ((d, &t), &m) in r8.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u8;
    }
}

/// `dst[x] = clip((alpha*dst[x] + beta) >> 8, 0, 255)`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn morph_row_8bpc_sse41(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    if !(i16::MIN as i32..=i16::MAX as i32).contains(&alpha) {
        for d in dst[..n].iter_mut() {
            *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, 255) as u8;
        }
        return;
    }

    // Exact scalar formula using `pmaddwd` with `[pixel, 0]` pairs.
    let coeff = _mm_set1_epi32(alpha & 0xffff);
    let beta_v = _mm_set1_epi32(beta);
    let f = |s: __m128i| _mm_srai_epi32::<8>(_mm_add_epi32(s, beta_v));
    let zero = _mm_setzero_si128();

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    for d in c16.iter_mut() {
        let (d0, d1) = load_u8x16_i16x2(&*d);
        let (s0, s1) = madd_i16x8_const(d0, zero, coeff);
        let (s2, s3) = madd_i16x8_const(d1, zero, coeff);
        store_i32x16_u8(d, f(s0), f(s1), f(s2), f(s3));
    }
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for d in c8.iter_mut() {
        let (s0, s1) = madd_i16x8_const(load_u8x8_i16(&*d), zero, coeff);
        store_i32x8_u8(d, f(s0), f(s1));
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for d in c4.iter_mut() {
        let s = madd_i16x4_const(load_u8x4_i16(d), zero, coeff);
        store_i32x4_u8(d, f(s));
    }
    for d in r4.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, 255) as u8;
    }
}

/// GDF residual add: `dst[x] = clip(dst[x] + sign(e)*((|e|+8)>>4), 0, 255)`,
/// `e = err[x]*scale`. `cmpgt(0, e)` selects the negated magnitude.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_add_run_8bpc_sse41(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let sc = _mm_set1_epi32(scale);
    let rnd = _mm_set1_epi32(8);
    let zero = _mm_setzero_si128();
    let adj = |e: __m128i| {
        let diff = _mm_mullo_epi32(e, sc);
        let mag = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_abs_epi32(diff), rnd));
        _mm_blendv_epi8(mag, _mm_sub_epi32(zero, mag), _mm_cmpgt_epi32(zero, diff))
    };
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (e16, _) = err[..n].as_chunks::<16>();
    for (d, e) in c16.iter_mut().zip(e16) {
        let (e0, e1, e2, e3) = load_i8x16_i32x4(e);
        let a0 = adj(e0);
        let a1 = adj(e1);
        let a2 = adj(e2);
        let a3 = adj(e3);
        let (d0, d1, d2, d3) = load_u8x16_i32x4(&*d);
        store_i32x16_u8(
            d,
            _mm_add_epi32(d0, a0),
            _mm_add_epi32(d1, a1),
            _mm_add_epi32(d2, a2),
            _mm_add_epi32(d3, a3),
        );
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (e8, _) = err[done..n].as_chunks::<8>();
    for (d, e) in c8.iter_mut().zip(e8) {
        let (e0, e1) = load_i8x8_i32x2(e);
        let a_lo = adj(e0);
        let a_hi = adj(e1);
        let (d_lo, d_hi) = load_u8x8_i32x2(&*d);
        store_i32x8_u8(d, _mm_add_epi32(d_lo, a_lo), _mm_add_epi32(d_hi, a_hi));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (e4, er) = err[done..n].as_chunks::<4>();
    for (d, e) in c4.iter_mut().zip(e4) {
        let a = adj(load_i8x4_i32(e));
        let dv = load_u8x4_i32(d);
        store_i32x4_u8(d, _mm_add_epi32(dv, a));
    }
    for (d, &e) in r4.iter_mut().zip(er) {
        let diff = e as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let a = if diff < 0 { -mag } else { mag };
        *d = ((*d as i32) + a).clamp(0, 255) as u8;
    }
}

/// GDF gradient: per-column `|2*b - a - c|` (each `>> shift`) summed over the 2
/// rows into 8 lanes, then pair-reduced to `ncells` cells via `hadd`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_gradient_group_sse41(
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
    let shc = _mm_cvtsi32_si128(shift as i32);
    let mut acc_lo = _mm_setzero_si128();
    let mut acc_hi = _mm_setzero_si128();
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let brow: &[u8; 8] = center_rows[y][bcol..bcol + 8].try_into().unwrap();
        let arow: &[u8; 8] = a_rows[y][acol..acol + 8].try_into().unwrap();
        let crow: &[u8; 8] = c_rows[y][ccol..ccol + 8].try_into().unwrap();
        let sh = |a: &[u8; 4]| _mm_srl_epi32(load_u8x4_i32(a), shc);
        let b_lo = sh((&brow[..4]).try_into().unwrap());
        let b_hi = sh((&brow[4..]).try_into().unwrap());
        let a_lo = sh((&arow[..4]).try_into().unwrap());
        let a_hi = sh((&arow[4..]).try_into().unwrap());
        let c_lo = sh((&crow[..4]).try_into().unwrap());
        let c_hi = sh((&crow[4..]).try_into().unwrap());
        let t_lo = _mm_sub_epi32(_mm_sub_epi32(_mm_add_epi32(b_lo, b_lo), a_lo), c_lo);
        let t_hi = _mm_sub_epi32(_mm_sub_epi32(_mm_add_epi32(b_hi, b_hi), a_hi), c_hi);
        acc_lo = _mm_add_epi32(acc_lo, _mm_abs_epi32(t_lo));
        acc_hi = _mm_add_epi32(acc_hi, _mm_abs_epi32(t_hi));
    }
    let pair = _mm_hadd_epi32(acc_lo, acc_hi);
    let mut out = [0i32; 4];
    store_i32x4(&mut out, pair);
    for k in 0..ncells {
        dst[base_cell + k][d] = out[k] as u16;
    }
}

#[cfg(test)]
mod residual_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn scalar(dst: &mut [u8], c: &[i32], n: usize, rnd: i32, shift: i32) {
        for i in 0..n {
            dst[i] = ((dst[i] as i32) + ((c[i] + rnd) >> shift)).clamp(0, 255) as u8;
        }
    }

    #[test]
    fn residual_8bpc_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..30000 {
            let n = rng.range(1, 40) as usize; // exercises 8-wide body + scalar tail
            let shift = rng.range(0, 13); // includes shift==0 (WHT path)
            let rnd = if shift == 0 {
                0
            } else {
                rng.range(0, 1 << shift)
            };
            let mut c = vec![0i32; n + 8];
            for v in c.iter_mut() {
                // mix ranges: small, mid, and large enough to overflow i16 at the
                // pack (so clip-to-[0,255] via double-saturate is exercised)
                *v = match rng.next() % 4 {
                    0 => rng.range(-400, 400),
                    1 => rng.range(-100_000, 100_000),
                    2 => rng.range(-(1 << 24), 1 << 24),
                    _ => 0,
                };
            }
            let mut base = vec![0u8; n + 8];
            for v in base.iter_mut() {
                *v = rng.range(0, 255) as u8;
            }
            let mut a = base.clone();
            let mut b = base.clone();
            scalar(&mut a, &c, n, rnd, shift);
            unsafe { residual_add_row_8bpc_sse41(&mut b, &c, n, rnd, shift) };
            assert_eq!(a, b, "n={n} shift={shift} rnd={rnd}");
        }
    }

    #[test]
    fn dc_8bpc_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0xfeed_face_0bad_c0de);
        for _ in 0..30000 {
            let n = rng.range(1, 40) as usize;
            let dc = match rng.next() % 4 {
                0 => rng.range(-400, 400),
                1 => rng.range(-100_000, 100_000),
                2 => rng.range(-(1 << 24), 1 << 24),
                _ => 0,
            };
            let mut base = vec![0u8; n + 8];
            for v in base.iter_mut() {
                *v = rng.range(0, 255) as u8;
            }
            let mut a = base.clone();
            let mut b = base.clone();
            for i in 0..n {
                a[i] = ((a[i] as i32) + dc).clamp(0, 255) as u8;
            }
            unsafe { dc_add_row_8bpc_sse41(&mut b, dc, n) };
            assert_eq!(a, b, "n={n} dc={dc}");
        }
    }
}

#[cfg(test)]
mod rowops_sse_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
        fn i16(&mut self) -> i16 {
            self.range(-32768, 32767) as i16
        }
        fn u8(&mut self) -> u8 {
            self.range(0, 255) as u8
        }
    }

    #[test]
    fn avg_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a01);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let sh = rng.range(1, 14);
            let rnd = rng.range(0, 1 << sh);
            let t1: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let t2: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                a[x] = ((t1[x] as i32 + t2[x] as i32 + rnd) >> sh).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { avg_row_8bpc_sse41(&mut b, &t1, &t2, n, rnd, sh) };
            assert_eq!(a, b, "n={n} sh={sh} rnd={rnd}");
        }
    }

    #[test]
    fn w_avg_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a02);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let sh = rng.range(1, 14);
            let rnd = rng.range(0, 1 << sh);
            let weight = rng.range(0, 16);
            let t1: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let t2: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                a[x] = ((t1[x] as i32 * weight + t2[x] as i32 * (16 - weight) + rnd) >> sh)
                    .clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { w_avg_row_8bpc_sse41(&mut b, &t1, &t2, n, weight, rnd, sh) };
            assert_eq!(a, b, "n={n} sh={sh} w={weight}");
        }
    }

    #[test]
    fn mask_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a03);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let sh = rng.range(1, 14);
            let rnd = rng.range(0, 1 << sh);
            let t1: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let t2: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let mask: Vec<u8> = (0..n + 8).map(|_| rng.range(0, 64) as u8).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                let m = mask[x] as i32;
                a[x] =
                    ((t1[x] as i32 * m + t2[x] as i32 * (64 - m) + rnd) >> sh).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { mask_row_8bpc_sse41(&mut b, &t1, &t2, &mask, n, rnd, sh) };
            assert_eq!(a, b, "n={n} sh={sh}");
        }
    }

    #[test]
    fn blend_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a04);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let tmp: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mask: Vec<u8> = (0..n + 8).map(|_| rng.range(0, 64) as u8).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                let m = mask[x] as i32;
                let d = a[x] as i32;
                let t = tmp[x] as i32;
                a[x] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
            }
            let mut b = base.clone();
            unsafe { blend_row_8bpc_sse41(&mut b, &tmp, &mask, n) };
            assert_eq!(a, b, "n={n}");
        }
    }

    #[test]
    fn morph_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a05);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let alpha = rng.range(-100_000, 100_000);
            let beta = rng.range(-(1 << 20), 1 << 20);
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                a[x] = ((alpha * a[x] as i32 + beta) >> 8).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { morph_row_8bpc_sse41(&mut b, alpha, beta, n) };
            assert_eq!(a, b, "n={n} alpha={alpha} beta={beta}");
        }
    }

    #[test]
    fn gdf_add_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a06);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let scale = rng.range(0, 100_000);
            let err: Vec<i8> = (0..n + 8).map(|_| rng.range(-128, 127) as i8).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                let diff = err[x] as i32 * scale;
                let mag = (diff.abs() + 8) >> 4;
                let adj = if diff < 0 { -mag } else { mag };
                a[x] = (a[x] as i32 + adj).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { gdf_add_run_8bpc_sse41(&mut b, &err, scale, n) };
            assert_eq!(a, b, "n={n} scale={scale}");
        }
    }

    #[test]
    fn row_clip_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a07);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let shift = rng.range(0, 14);
            let rnd = if shift == 0 {
                0
            } else {
                rng.range(0, 1 << shift)
            };
            let min = rng.range(-(1 << 20), 0);
            let max = rng.range(0, 1 << 20);
            let src: Vec<i32> = (0..n + 8)
                .map(|_| match rng.next() % 3 {
                    0 => rng.range(-1000, 1000),
                    1 => rng.range(-(1 << 28), 1 << 28),
                    _ => 0,
                })
                .collect();
            let mut a = src.clone();
            for x in 0..n {
                a[x] = ((a[x] + rnd) >> shift).max(min).min(max);
            }
            let mut b = src.clone();
            unsafe { row_clip_sse41(&mut b, n, rnd, shift, min, max) };
            assert_eq!(a, b, "n={n} shift={shift}");
        }
    }

    #[test]
    fn cctx_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a08);
        for _ in 0..20000 {
            let sz = rng.range(1, 40) as usize;
            // bound so u*cosa - v*sina cannot overflow i32 (matches decoder ranges).
            let sina = rng.range(-256, 256);
            let cosa = rng.range(-256, 256);
            let min = rng.range(-(1 << 20), 0);
            let max = rng.range(0, 1 << 20);
            let u0: Vec<i32> = (0..sz + 8)
                .map(|_| rng.range(-(1 << 22), 1 << 22))
                .collect();
            let v0: Vec<i32> = (0..sz + 8)
                .map(|_| rng.range(-(1 << 22), 1 << 22))
                .collect();
            let (mut ua, mut va) = (u0.clone(), v0.clone());
            for i in 0..sz {
                let a = ua[i] * cosa - va[i] * sina;
                let b = ua[i] * sina + va[i] * cosa;
                ua[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
                va[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
            }
            let (mut ub, mut vb) = (u0.clone(), v0.clone());
            unsafe { cctx_row_sse41(&mut ub, &mut vb, sina, cosa, sz, min, max) };
            assert_eq!(ua, ub, "u sz={sz}");
            assert_eq!(va, vb, "v sz={sz}");
        }
    }
}

#[cfg(test)]
mod gdf_gradient_sse_test {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn scalar(
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
            for k in 0..8 {
                let b = (center_rows[y][bcol + k] as i32) >> shift;
                let a = (a_rows[y][acol + k] as i32) >> shift;
                let c = (c_rows[y][ccol + k] as i32) >> shift;
                acc[k] += (b + b - a - c).abs();
            }
        }
        for k in 0..ncells {
            dst[base_cell + k][d] = (acc[2 * k] + acc[2 * k + 1]) as u16;
        }
    }

    #[test]
    fn gdf_gradient_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let l = 64usize;
        let mut rng = Rng(0x0a09);
        for _ in 0..20000 {
            let dx = rng.range(0, 16);
            let bcol = rng.range(dx, (l as i32) - 8 - dx) as usize;
            let col0 = bcol + 1;
            let ncells = rng.range(1, 4) as usize;
            let d = rng.range(0, 3) as usize;
            let shift = rng.range(0, 7) as u32;
            let mk =
                |rng: &mut Rng| -> Vec<u8> { (0..l).map(|_| rng.range(0, 255) as u8).collect() };
            let cr = [mk(&mut rng), mk(&mut rng)];
            let ar = [mk(&mut rng), mk(&mut rng)];
            let crr = [mk(&mut rng), mk(&mut rng)];
            let mut da = [[0u16; 4]; 4];
            let mut db = [[0u16; 4]; 4];
            scalar(
                &mut da,
                d,
                0,
                ncells,
                [&cr[0], &cr[1]],
                [&ar[0], &ar[1]],
                [&crr[0], &crr[1]],
                col0,
                dx,
                shift,
            );
            unsafe {
                gdf_gradient_group_sse41(
                    &mut db,
                    d,
                    0,
                    ncells,
                    [&cr[0], &cr[1]],
                    [&ar[0], &ar[1]],
                    [&crr[0], &crr[1]],
                    col0,
                    dx,
                    shift,
                );
            }
            assert_eq!(
                da, db,
                "dx={dx} col0={col0} ncells={ncells} d={d} shift={shift}"
            );
        }
    }
}

/// cctx rotate+clip over two i16 coefficient planes, widening only inside the SIMD arithmetic.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cctx_row_i16_sse41(
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
        let a_pair_v = _mm_set1_epi32(a_pair);
        let b_pair_v = _mm_set1_epi32(b_pair);
        let c128 = _mm_set1_epi32(128);
        let zero = _mm_setzero_si128();
        let min_v = _mm_set1_epi32(min);
        let max_v = _mm_set1_epi32(max);
        let (u_chunks, ur) = u[..sz].as_chunks_mut::<4>();
        let (v_chunks, vr) = v[..sz].as_chunks_mut::<4>();
        for (uch, vch) in u_chunks.iter_mut().zip(v_chunks.iter_mut()) {
            let uu16 = _mm_loadl_epi64(uch.as_ptr() as *const __m128i);
            let vv16 = _mm_loadl_epi64(vch.as_ptr() as *const __m128i);
            let uv = _mm_unpacklo_epi16(uu16, vv16);
            let a = _mm_madd_epi16(uv, a_pair_v);
            let b = _mm_madd_epi16(uv, b_pair_v);
            let ru = _mm_min_epi32(
                _mm_max_epi32(
                    _mm_srai_epi32::<8>(_mm_add_epi32(
                        _mm_add_epi32(a, c128),
                        _mm_cmpgt_epi32(zero, a),
                    )),
                    min_v,
                ),
                max_v,
            );
            let rv = _mm_min_epi32(
                _mm_max_epi32(
                    _mm_srai_epi32::<8>(_mm_add_epi32(
                        _mm_add_epi32(b, c128),
                        _mm_cmpgt_epi32(zero, b),
                    )),
                    min_v,
                ),
                max_v,
            );
            _mm_storel_epi64(uch.as_mut_ptr() as *mut __m128i, _mm_packs_epi32(ru, zero));
            _mm_storel_epi64(vch.as_mut_ptr() as *mut __m128i, _mm_packs_epi32(rv, zero));
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
