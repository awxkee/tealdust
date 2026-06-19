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

// ---------------------------------------------------------------------------
// Fixed-array SIMD load/store helpers. Each takes/returns a `&[T; N]` so the
// bounds are proven at the call site (via `as_chunks` + `try_into`) and the
// body needs no raw-pointer arithmetic. `#[inline(always)]` folds them into the
// `#[target_feature]` kernels below, where the SSE4.1 widening/pack intrinsics
// are valid.
// ---------------------------------------------------------------------------

#[inline(always)]
fn load_i16x4_i32(a: &[i16; 4]) -> __m128i {
    unsafe { _mm_cvtepi16_epi32(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn load_u8x4_i32(a: &[u8; 4]) -> __m128i {
    unsafe { _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(*a))) }
}

#[inline(always)]
fn load_i8x4_i32(a: &[i8; 4]) -> __m128i {
    let bytes = [a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8];
    unsafe { _mm_cvtepi8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(bytes))) }
}

#[inline(always)]
fn load_i32x4(a: &[i32; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_i32x4(a: &mut [i32; 4], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

/// Pack one i32x4 to u8 (signed-sat to i16, then unsigned-sat to u8 ==
/// `clamp(.,0,255)`) and write the low 4 bytes.
#[inline(always)]
fn store_i32x4_u8(a: &mut [u8; 4], v: __m128i) {
    let p16 = unsafe { _mm_packs_epi32(v, v) };
    let p8 = unsafe { _mm_packus_epi16(p16, p16) };
    *a = (unsafe { _mm_cvtsi128_si32(p8) } as u32).to_le_bytes();
}

/// Pack two i32x4 lanes (lo, hi) to 8 clamped u8 and store.
#[inline(always)]
fn store_i32x8_u8(a: &mut [u8; 8], lo: __m128i, hi: __m128i) {
    let p16 = unsafe { _mm_packs_epi32(lo, hi) };
    let p8 = unsafe { _mm_packus_epi16(p16, p16) };
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, p8) };
}

#[inline(always)]
fn load_u8x8_i16(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i16x8_u8(a: &mut [u8; 8], v: __m128i) {
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, _mm_packus_epi16(v, v)) };
}

#[inline(always)]
fn store_i16x8x2_u8(a: &mut [u8; 16], lo: __m128i, hi: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, _mm_packus_epi16(lo, hi)) };
}

// ---------------------------------------------------------------------------

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
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (cc8, _) = c[..n].as_chunks::<8>();
    for (d, cv) in c8.iter_mut().zip(cc8) {
        let cf_lo = _mm_sra_epi32(
            _mm_add_epi32(load_i32x4((&cv[..4]).try_into().unwrap()), rnd_v),
            shc,
        );
        let cf_hi = _mm_sra_epi32(
            _mm_add_epi32(load_i32x4((&cv[4..]).try_into().unwrap()), rnd_v),
            shc,
        );
        let d_lo = load_u8x4_i32((&d[..4]).try_into().unwrap());
        let d_hi = load_u8x4_i32((&d[4..]).try_into().unwrap());
        store_i32x8_u8(d, _mm_add_epi32(d_lo, cf_lo), _mm_add_epi32(d_hi, cf_hi));
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (cc4, cr) = c[done..n].as_chunks::<4>();
    for (d, cv) in c4.iter_mut().zip(cc4) {
        let cf = _mm_sra_epi32(_mm_add_epi32(load_i32x4(cv), rnd_v), shc);
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
    let dc_v = _mm_set1_epi32(dc);
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in c8.iter_mut() {
        let d_lo = load_u8x4_i32((&d[..4]).try_into().unwrap());
        let d_hi = load_u8x4_i32((&d[4..]).try_into().unwrap());
        store_i32x8_u8(d, _mm_add_epi32(d_lo, dc_v), _mm_add_epi32(d_hi, dc_v));
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for d in c4.iter_mut() {
        let dv = load_u8x4_i32(d);
        store_i32x4_u8(d, _mm_add_epi32(dv, dc_v));
    }
    for d in r4.iter_mut() {
        *d = ((*d as i32) + dc).clamp(0, 255) as u8;
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
    let sh8 = _mm_cvtsi32_si128(8);
    let rot = |uu: __m128i, vv: __m128i| -> (__m128i, __m128i) {
        {
            let a = _mm_sub_epi32(_mm_mullo_epi32(uu, cosa_v), _mm_mullo_epi32(vv, sina_v));
            let b = _mm_add_epi32(_mm_mullo_epi32(uu, sina_v), _mm_mullo_epi32(vv, cosa_v));
            let ra = _mm_sra_epi32(
                _mm_add_epi32(_mm_add_epi32(a, c128), _mm_cmpgt_epi32(zero, a)),
                sh8,
            );
            let rb = _mm_sra_epi32(
                _mm_add_epi32(_mm_add_epi32(b, c128), _mm_cmpgt_epi32(zero, b)),
                sh8,
            );
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
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let lo = f(
            load_i16x4_i32((&a[..4]).try_into().unwrap()),
            load_i16x4_i32((&b[..4]).try_into().unwrap()),
        );
        let hi = f(
            load_i16x4_i32((&a[4..]).try_into().unwrap()),
            load_i16x4_i32((&b[4..]).try_into().unwrap()),
        );
        store_i32x8_u8(d, lo, hi);
    }
    let done = c8.len() * 8;
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
    let w1 = _mm_set1_epi32(weight);
    let w2 = _mm_set1_epi32(16 - weight);
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let f = |a: __m128i, b: __m128i| {
        _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a, w1), _mm_mullo_epi32(b, w2)),
                rnd_v,
            ),
            shc,
        )
    };
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let lo = f(
            load_i16x4_i32((&a[..4]).try_into().unwrap()),
            load_i16x4_i32((&b[..4]).try_into().unwrap()),
        );
        let hi = f(
            load_i16x4_i32((&a[4..]).try_into().unwrap()),
            load_i16x4_i32((&b[4..]).try_into().unwrap()),
        );
        store_i32x8_u8(d, lo, hi);
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    for ((d, a), b) in c4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u8(d, f(load_i16x4_i32(a), load_i16x4_i32(b)));
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
    let rnd_v = _mm_set1_epi32(rnd);
    let c64 = _mm_set1_epi32(64);
    let shc = _mm_cvtsi32_si128(sh);
    let f = |a: __m128i, b: __m128i, m: __m128i| {
        _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(a, m),
                    _mm_mullo_epi32(b, _mm_sub_epi32(c64, m)),
                ),
                rnd_v,
            ),
            shc,
        )
    };
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    let (m8, _) = mask[..n].as_chunks::<8>();
    for (((d, a), b), m) in c8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let lo = f(
            load_i16x4_i32((&a[..4]).try_into().unwrap()),
            load_i16x4_i32((&b[..4]).try_into().unwrap()),
            load_u8x4_i32((&m[..4]).try_into().unwrap()),
        );
        let hi = f(
            load_i16x4_i32((&a[4..]).try_into().unwrap()),
            load_i16x4_i32((&b[4..]).try_into().unwrap()),
            load_u8x4_i32((&m[4..]).try_into().unwrap()),
        );
        store_i32x8_u8(d, lo, hi);
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    for (((d, a), b), m) in c4.iter_mut().zip(a4).zip(b4).zip(m4) {
        store_i32x4_u8(d, f(load_i16x4_i32(a), load_i16x4_i32(b), load_u8x4_i32(m)));
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
        let o0 = f(
            load_u8x8_i16((&d[..8]).try_into().unwrap()),
            load_u8x8_i16((&t[..8]).try_into().unwrap()),
            load_u8x8_i16((&m[..8]).try_into().unwrap()),
        );
        let o1 = f(
            load_u8x8_i16((&d[8..]).try_into().unwrap()),
            load_u8x8_i16((&t[8..]).try_into().unwrap()),
            load_u8x8_i16((&m[8..]).try_into().unwrap()),
        );
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
    let a_v = _mm_set1_epi32(alpha);
    let b_v = _mm_set1_epi32(beta);
    let sh8 = _mm_cvtsi32_si128(8);
    let f = |d: __m128i| _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(d, a_v), b_v), sh8);
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in c8.iter_mut() {
        let lo = f(load_u8x4_i32((&d[..4]).try_into().unwrap()));
        let hi = f(load_u8x4_i32((&d[4..]).try_into().unwrap()));
        store_i32x8_u8(d, lo, hi);
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for d in c4.iter_mut() {
        let r = f(load_u8x4_i32(d));
        store_i32x4_u8(d, r);
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
    let sh4 = _mm_cvtsi32_si128(4);
    let zero = _mm_setzero_si128();
    let adj = |e: __m128i| {
        let diff = _mm_mullo_epi32(e, sc);
        let mag = _mm_sra_epi32(_mm_add_epi32(_mm_abs_epi32(diff), rnd), sh4);
        _mm_blendv_epi8(mag, _mm_sub_epi32(zero, mag), _mm_cmpgt_epi32(zero, diff))
    };
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (e8, _) = err[..n].as_chunks::<8>();
    for (d, e) in c8.iter_mut().zip(e8) {
        let a_lo = adj(load_i8x4_i32((&e[..4]).try_into().unwrap()));
        let a_hi = adj(load_i8x4_i32((&e[4..]).try_into().unwrap()));
        let d_lo = load_u8x4_i32((&d[..4]).try_into().unwrap());
        let d_hi = load_u8x4_i32((&d[4..]).try_into().unwrap());
        store_i32x8_u8(d, _mm_add_epi32(d_lo, a_lo), _mm_add_epi32(d_hi, a_hi));
    }
    let done = c8.len() * 8;
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
