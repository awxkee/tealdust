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

use crate::itx_2d::ITX_TMP_PIXELS;

use core::cell::RefCell;

thread_local! {
    static SSE41_ITX_I16_SCRATCH: RefCell<[i16; ITX_TMP_PIXELS]> = const { RefCell::new([0i16; ITX_TMP_PIXELS]) };
}

#[inline(always)]
fn with_sse41_itx_i16_scratch<R>(len: usize, f: impl FnOnce(&mut [i16]) -> R) -> R {
    assert!(len <= ITX_TMP_PIXELS);
    SSE41_ITX_I16_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        f(&mut scratch[..len])
    })
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct16_i32x4_impl(s: &[__m128i; 16]) -> [__m128i; 16] {
    let z = _mm_setzero_si128();
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            let k = _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m]);
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(s[j], k));
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_adst16_i32x4_impl(s: &[__m128i; 16], flip: bool) -> [__m128i; 16] {
    let rows = if flip {
        &crate::itx_1d::FLIPADST16_KERNEL_ROWS
    } else {
        &crate::itx_1d::ADST16_KERNEL_ROWS
    };
    let z = _mm_setzero_si128();
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let row = &rows[m];
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            let k = _mm_set1_epi32(row[j] as i32);
            acc = _mm_add_epi32(acc, _mm_mullo_epi32(s[j], k));
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_tx16_i32x4_impl(s: &[__m128i; 16], kind: usize) -> [__m128i; 16] {
    match kind {
        crate::itx_2d::TX_KIND_DCT => sse41_dct16_i32x4_impl(s),
        crate::itx_2d::TX_KIND_ADST => sse41_adst16_i32x4_impl(s, false),
        crate::itx_2d::TX_KIND_FLIPADST => sse41_adst16_i32x4_impl(s, true),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn iadst_dequant_16x16_sse41_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        iadst_dequant_16x16_sse41_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        iadst_dequant_16x16_sse41_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn iadst_dequant_16x16_sse41_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 256);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut s = [_mm_setzero_si128(); 16];
            let mut j = 0usize;
            while j < 16 {
                let mut v = _mm_loadu_si128(coeff.as_ptr().add(y + j * 16) as *const __m128i);
                if IS_RECT2 {
                    v = _mm_srai_epi32::<8>(_mm_add_epi32(
                        _mm_mullo_epi32(v, _mm_set1_epi32(181)),
                        _mm_set1_epi32(128),
                    ));
                }
                s[j] = v;
                j += 1;
            }
            let out = sse41_tx16_i32x4_impl(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                sse41_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
                x += 4;
            }
            y += 4;
        }
        while y < 16 {
            tmp[y * 32..y * 32 + 16].fill(0);
            y += 1;
        }
        coeff[..256].fill(0);
        let mut x = 0usize;
        while x < 16 {
            let mut s = [_mm_setzero_si128(); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = _mm_loadu_si128(tmp.as_ptr().add(x + j * 32) as *const __m128i);
                j += 1;
            }
            let out = sse41_tx16_i32x4_impl(&s, second_kind);
            j = 0;
            while j < 16 {
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + j * 32) as *mut __m128i, out[j]);
                j += 1;
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_store4x4_i32_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    v: &[__m128i; 4],
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) {
    unsafe {
        macro_rules! clip {
            ($x:expr) => {{
                _mm_min_epi32(
                    _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32($x, rnd), sh), minv),
                    maxv,
                )
            }};
        }
        let c0 = clip!(v[0]);
        let c1 = clip!(v[1]);
        let c2 = clip!(v[2]);
        let c3 = clip!(v[3]);
        let t0 = _mm_unpacklo_epi32(c0, c1);
        let t1 = _mm_unpackhi_epi32(c0, c1);
        let t2 = _mm_unpacklo_epi32(c2, c3);
        let t3 = _mm_unpackhi_epi32(c2, c3);
        let r0 = _mm_unpacklo_epi64(t0, t2);
        let r1 = _mm_unpackhi_epi64(t0, t2);
        let r2 = _mm_unpacklo_epi64(t1, t3);
        let r3 = _mm_unpackhi_epi64(t1, t3);
        _mm_storeu_si128(tmp.as_mut_ptr().add(off) as *mut __m128i, r0);
        _mm_storeu_si128(tmp.as_mut_ptr().add(off + 32) as *mut __m128i, r1);
        _mm_storeu_si128(tmp.as_mut_ptr().add(off + 64) as *mut __m128i, r2);
        _mm_storeu_si128(tmp.as_mut_ptr().add(off + 96) as *mut __m128i, r3);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_store4x4_i16_clip<const STRIDE: usize>(
    scratch: &mut [i16],
    off: usize,
    v0: __m128i,
    v1: __m128i,
    v2: __m128i,
    v3: __m128i,
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) {
    unsafe {
        debug_assert!(STRIDE == 4 || STRIDE == 8 || STRIDE == 16 || STRIDE == 32);
        debug_assert!(off + 3 * STRIDE + 4 <= scratch.len());
        macro_rules! clip {
            ($x:expr) => {{
                _mm_min_epi32(
                    _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32($x, rnd), sh), minv),
                    maxv,
                )
            }};
        }
        let c0 = clip!(v0);
        let c1 = clip!(v1);
        let c2 = clip!(v2);
        let c3 = clip!(v3);
        let t0 = _mm_unpacklo_epi32(c0, c1);
        let t1 = _mm_unpackhi_epi32(c0, c1);
        let t2 = _mm_unpacklo_epi32(c2, c3);
        let t3 = _mm_unpackhi_epi32(c2, c3);
        let r0 = _mm_unpacklo_epi64(t0, t2);
        let r1 = _mm_unpackhi_epi64(t0, t2);
        let r2 = _mm_unpacklo_epi64(t1, t3);
        let r3 = _mm_unpackhi_epi64(t1, t3);
        _mm_storel_epi64(
            scratch.as_mut_ptr().add(off) as *mut __m128i,
            _mm_packs_epi32(r0, _mm_setzero_si128()),
        );
        _mm_storel_epi64(
            scratch.as_mut_ptr().add(off + STRIDE) as *mut __m128i,
            _mm_packs_epi32(r1, _mm_setzero_si128()),
        );
        _mm_storel_epi64(
            scratch.as_mut_ptr().add(off + 2 * STRIDE) as *mut __m128i,
            _mm_packs_epi32(r2, _mm_setzero_si128()),
        );
        _mm_storel_epi64(
            scratch.as_mut_ptr().add(off + 3 * STRIDE) as *mut __m128i,
            _mm_packs_epi32(r3, _mm_setzero_si128()),
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_store8x8_i16_clip<const STRIDE: usize>(
    scratch: &mut [i16],
    off: usize,
    v0lo: __m128i,
    v0hi: __m128i,
    v1lo: __m128i,
    v1hi: __m128i,
    v2lo: __m128i,
    v2hi: __m128i,
    v3lo: __m128i,
    v3hi: __m128i,
    v4lo: __m128i,
    v4hi: __m128i,
    v5lo: __m128i,
    v5hi: __m128i,
    v6lo: __m128i,
    v6hi: __m128i,
    v7lo: __m128i,
    v7hi: __m128i,
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) {
    unsafe {
        debug_assert!(STRIDE == 8 || STRIDE == 16 || STRIDE == 32);
        debug_assert!(off + 7 * STRIDE + 8 <= scratch.len());
        macro_rules! clip {
            ($x:expr) => {{
                _mm_min_epi32(
                    _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32($x, rnd), sh), minv),
                    maxv,
                )
            }};
        }

        let r0 = _mm_packs_epi32(clip!(v0lo), clip!(v0hi));
        let r1 = _mm_packs_epi32(clip!(v1lo), clip!(v1hi));
        let r2 = _mm_packs_epi32(clip!(v2lo), clip!(v2hi));
        let r3 = _mm_packs_epi32(clip!(v3lo), clip!(v3hi));
        let r4 = _mm_packs_epi32(clip!(v4lo), clip!(v4hi));
        let r5 = _mm_packs_epi32(clip!(v5lo), clip!(v5hi));
        let r6 = _mm_packs_epi32(clip!(v6lo), clip!(v6hi));
        let r7 = _mm_packs_epi32(clip!(v7lo), clip!(v7hi));

        let t0 = _mm_unpacklo_epi16(r0, r1);
        let t1 = _mm_unpackhi_epi16(r0, r1);
        let t2 = _mm_unpacklo_epi16(r2, r3);
        let t3 = _mm_unpackhi_epi16(r2, r3);
        let t4 = _mm_unpacklo_epi16(r4, r5);
        let t5 = _mm_unpackhi_epi16(r4, r5);
        let t6 = _mm_unpacklo_epi16(r6, r7);
        let t7 = _mm_unpackhi_epi16(r6, r7);

        let u0 = _mm_unpacklo_epi32(t0, t2);
        let u1 = _mm_unpackhi_epi32(t0, t2);
        let u2 = _mm_unpacklo_epi32(t1, t3);
        let u3 = _mm_unpackhi_epi32(t1, t3);
        let u4 = _mm_unpacklo_epi32(t4, t6);
        let u5 = _mm_unpackhi_epi32(t4, t6);
        let u6 = _mm_unpacklo_epi32(t5, t7);
        let u7 = _mm_unpackhi_epi32(t5, t7);

        let o0 = _mm_unpacklo_epi64(u0, u4);
        let o1 = _mm_unpackhi_epi64(u0, u4);
        let o2 = _mm_unpacklo_epi64(u1, u5);
        let o3 = _mm_unpackhi_epi64(u1, u5);
        let o4 = _mm_unpacklo_epi64(u2, u6);
        let o5 = _mm_unpackhi_epi64(u2, u6);
        let o6 = _mm_unpacklo_epi64(u3, u7);
        let o7 = _mm_unpackhi_epi64(u3, u7);

        _mm_storeu_si128(scratch.as_mut_ptr().add(off) as *mut __m128i, o0);
        _mm_storeu_si128(scratch.as_mut_ptr().add(off + STRIDE) as *mut __m128i, o1);
        _mm_storeu_si128(
            scratch.as_mut_ptr().add(off + 2 * STRIDE) as *mut __m128i,
            o2,
        );
        _mm_storeu_si128(
            scratch.as_mut_ptr().add(off + 3 * STRIDE) as *mut __m128i,
            o3,
        );
        _mm_storeu_si128(
            scratch.as_mut_ptr().add(off + 4 * STRIDE) as *mut __m128i,
            o4,
        );
        _mm_storeu_si128(
            scratch.as_mut_ptr().add(off + 5 * STRIDE) as *mut __m128i,
            o5,
        );
        _mm_storeu_si128(
            scratch.as_mut_ptr().add(off + 6 * STRIDE) as *mut __m128i,
            o6,
        );
        _mm_storeu_si128(
            scratch.as_mut_ptr().add(off + 7 * STRIDE) as *mut __m128i,
            o7,
        );
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_load4_i16_scratch(src: &[i16], off: usize) -> __m128i {
    debug_assert!(off + 4 <= src.len());
    unsafe { _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    m: usize,
) -> [__m128i; 4] {
    unsafe {
        let z = _mm_setzero_si128();
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let mut j = 0usize;
        while j < 32 {
            let mut v = _mm_loadu_si128(coeff.as_ptr().add(base + j * 32) as *const __m128i);
            if IS_RECT2 {
                v = _mm_srai_epi32::<8>(_mm_add_epi32(_mm_mullo_epi32(v, rect_mul), rect_rnd));
            }
            a0 = _mm_add_epi32(
                a0,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]),
                ),
            );
            a1 = _mm_add_epi32(
                a1,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]),
                ),
            );
            a2 = _mm_add_epi32(
                a2,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]),
                ),
            );
            a3 = _mm_add_epi32(
                a3,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]),
                ),
            );
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i32x4_from_tmp4(tmp: &[i32; ITX_TMP_PIXELS], base: usize, m: usize) -> [__m128i; 4] {
    unsafe {
        let z = _mm_setzero_si128();
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 32 {
            let v = _mm_loadu_si128(tmp.as_ptr().add(base + j * 32) as *const __m128i);
            a0 = _mm_add_epi32(
                a0,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]),
                ),
            );
            a1 = _mm_add_epi32(
                a1,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]),
                ),
            );
            a2 = _mm_add_epi32(
                a2,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]),
                ),
            );
            a3 = _mm_add_epi32(
                a3,
                _mm_mullo_epi32(
                    v,
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]),
                ),
            );
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_tx8_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    kind: usize,
    m: usize,
) -> [__m128i; 4] {
    unsafe {
        let z = _mm_setzero_si128();
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let mut j = 0usize;
        while j < 8 {
            let mut v = _mm_loadu_si128(coeff.as_ptr().add(base + j * 8) as *const __m128i);
            if IS_RECT2 {
                v = _mm_srai_epi32::<8>(_mm_add_epi32(_mm_mullo_epi32(v, rect_mul), rect_rnd));
            }
            a0 = _mm_add_epi32(
                a0,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m, j))),
            );
            a1 = _mm_add_epi32(
                a1,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m + 1, j))),
            );
            a2 = _mm_add_epi32(
                a2,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m + 2, j))),
            );
            a3 = _mm_add_epi32(
                a3,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m + 3, j))),
            );
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_tx8_i32x4_from_tmp4(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    kind: usize,
    m: usize,
) -> [__m128i; 4] {
    unsafe {
        let z = _mm_setzero_si128();
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 8 {
            let v = _mm_loadu_si128(tmp.as_ptr().add(base + j * 32) as *const __m128i);
            a0 = _mm_add_epi32(
                a0,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m, j))),
            );
            a1 = _mm_add_epi32(
                a1,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m + 1, j))),
            );
            a2 = _mm_add_epi32(
                a2,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m + 2, j))),
            );
            a3 = _mm_add_epi32(
                a3,
                _mm_mullo_epi32(v, _mm_set1_epi32(tx8_coeff(kind, m + 3, j))),
            );
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
fn tx8_coeff(kind: usize, out: usize, input: usize) -> i32 {
    match kind {
        crate::itx_2d::TX_KIND_DCT => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_ADST => crate::itx_2d::ADST8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_FLIPADST => crate::itx_2d::ADST8_KW[(7 - out) * 8 + input] as i32,
        _ => unreachable!(),
    }
}

#[inline]
fn sse41_identity_coeff(n: usize, out: usize, input: usize) -> i32 {
    if out != input {
        0
    } else {
        match n {
            4 => 128,
            8 => 181,
            16 => 256,
            32 => 362,
            _ => unreachable!(),
        }
    }
}

#[inline]
fn sse41_identity_scale(n: usize) -> i32 {
    match n {
        4 => 128,
        8 => 181,
        16 => 256,
        32 => 362,
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn sse41_identity_i16x4_coeff_to_i32<const IS_RECT2: bool>(
    coeff: &[i16],
    off: usize,
    scale: __m128i,
) -> __m128i {
    let v = sse41_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, off);
    _mm_mullo_epi32(_mm_cvtepi16_epi32(v), scale)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_identity_i16x4_scratch_to_i32(scratch: &[i16], off: usize, scale: __m128i) -> __m128i {
    let v = sse41_load4_i16_scratch(scratch, off);
    _mm_mullo_epi32(_mm_cvtepi16_epi32(v), scale)
}

#[inline]
fn sse41_tx_dense_coeff(kind: usize, n: usize, out: usize, input: usize) -> i32 {
    match (kind, n) {
        (crate::itx_2d::TX_KIND_DCT, 4) => crate::itx_2d::DCT4_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 8) => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 16) => crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out],
        (crate::itx_2d::TX_KIND_DCT, 32) => crate::itx_2d::DCT32_DENSE_KERNEL[input * 32 + out],
        (crate::itx_2d::TX_KIND_IDENTITY, 4)
        | (crate::itx_2d::TX_KIND_IDENTITY, 8)
        | (crate::itx_2d::TX_KIND_IDENTITY, 16)
        | (crate::itx_2d::TX_KIND_IDENTITY, 32) => sse41_identity_coeff(n, out, input),
        (crate::itx_2d::TX_KIND_ADST, 4) => crate::itx_1d::ADST4_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_ADST, 8) => crate::itx_1d::ADST8_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_ADST, 16) => crate::itx_1d::ADST16_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_FLIPADST, 4) => {
            crate::itx_1d::FLIPADST4_KERNEL_ROWS[out][input] as i32
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 8) => {
            crate::itx_1d::ADST8_KERNEL_ROWS[7 - out][input] as i32
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 16) => {
            crate::itx_1d::FLIPADST16_KERNEL_ROWS[out][input] as i32
        }
        _ => unreachable!(),
    }
}
#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_tx_dense_coeff_pair(kind: usize, n: usize, out: usize, input: usize) -> __m128i {
    debug_assert_eq!(input & 1, 0);
    let (table, idx): (&[i32], usize) = match (kind, n) {
        (crate::itx_2d::TX_KIND_DCT, 4) => (&crate::itx_2d::DCT4_KP_X4, out * 4 + (input >> 1)),
        (crate::itx_2d::TX_KIND_DCT, 8) => (&crate::itx_2d::DCT8_KP_X4, out * 4 + (input >> 1)),
        (crate::itx_2d::TX_KIND_DCT, 16) => {
            (&crate::itx_2d::DCT16_DENSE_PAIR_X4, out * 8 + (input >> 1))
        }
        (crate::itx_2d::TX_KIND_DCT, 32) => {
            (&crate::itx_2d::DCT32_DENSE_PAIR_X4, out * 16 + (input >> 1))
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 4)
        | (crate::itx_2d::TX_KIND_IDENTITY, 8)
        | (crate::itx_2d::TX_KIND_IDENTITY, 16)
        | (crate::itx_2d::TX_KIND_IDENTITY, 32) => {
            let k0 = sse41_identity_coeff(n, out, input) as i16;
            let k1 = sse41_identity_coeff(n, out, input + 1) as i16;
            return sse41_coeff_pair_from_scalars_i16(k0, k1);
        }
        (crate::itx_2d::TX_KIND_ADST, 4) => (&crate::itx_2d::ADST4_KP_X4, out * 4 + (input >> 1)),
        (crate::itx_2d::TX_KIND_ADST, 8) => (&crate::itx_2d::ADST8_KP_X4, out * 4 + (input >> 1)),
        (crate::itx_2d::TX_KIND_ADST, 16) => (&crate::itx_2d::ADST16_KP_X4, out * 8 + (input >> 1)),
        (crate::itx_2d::TX_KIND_FLIPADST, 4) => {
            (&crate::itx_2d::FLIPADST4_KP_X4, out * 4 + (input >> 1))
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 8) => {
            (&crate::itx_2d::ADST8_KP_X4, (7 - out) * 4 + (input >> 1))
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 16) => {
            (&crate::itx_2d::FLIPADST16_KP_X4, out * 8 + (input >> 1))
        }
        _ => unreachable!(),
    };
    sse41_coeff_pair_i16(table, idx)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_load4_i16_coeff_packed_const<const IS_RECT2: bool>(src: &[i16], off: usize) -> __m128i {
    debug_assert!(off + 4 <= src.len());
    let mut v = unsafe { _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i) };
    if IS_RECT2 {
        // Low-bit-depth i16 path: keep rect2 normalization packed so the
        // transform can accumulate with pmaddwd instead of widening first.
        v = _mm_mulhrs_epi16(v, _mm_set1_epi16(0x5a80));
    }
    v
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_coeff_pair_from_scalars_i16(k0: i16, k1: i16) -> __m128i {
    _mm_set1_epi32(((k1 as u16 as i32) << 16) | (k0 as u16 as i32))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_coeff_pair_i16(table: &[i32], idx: usize) -> __m128i {
    debug_assert!(idx * 4 + 4 <= table.len());
    unsafe { _mm_loadu_si128(table.as_ptr().add(idx * 4) as *const __m128i) }
}

macro_rules! sse41_dct16_i16x4_all_body {
    () => {{
        let z = _mm_setzero_si128();
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(1), load!(3)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, base >> 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(5), load!(7)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(9), load!(11)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 2),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(13), load!(15)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 3),
                ),
            );
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(2), load!(6)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, base >> 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(10), load!(14)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, (base >> 1) + 1),
                ),
            );
            d[m] = acc;
            m += 1;
        }
        let f0 = _mm_add_epi32(
            _mm_madd_epi16(
                _mm_unpacklo_epi16(load!(4), load!(12)),
                sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 0),
            ),
            z,
        );
        let f1 = _mm_add_epi32(
            _mm_madd_epi16(
                _mm_unpacklo_epi16(load!(4), load!(12)),
                sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 1),
            ),
            z,
        );
        let g0 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(8)),
            sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 0),
        );
        let g1 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(8)),
            sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 1),
        );
        let cc = [
            _mm_add_epi32(g0, f0),
            _mm_add_epi32(g1, f1),
            _mm_sub_epi32(g1, f1),
            _mm_sub_epi32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = _mm_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = _mm_sub_epi32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = _mm_add_epi32(a[k], b[k]);
            out[k + 8] = _mm_sub_epi32(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! sse41_dct32_i16x4_all_body {
    () => {{
        let z = _mm_setzero_si128();
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut p = 0usize;
            while p < 16 {
                let cb = base + p;
                let i0 = 2 * p + 1;
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(i0), load!(i0 + 2)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KBP_X4, cb >> 1),
                    ),
                );
                p += 2;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            let mut p = 0usize;
            while p < 8 {
                let i0 = 4 * p + 2;
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(i0), load!(i0 + 4)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KDP_X4, (base + p) >> 1),
                    ),
                );
                p += 2;
            }
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(4), load!(12)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, base >> 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(20), load!(28)),
                    sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, (base >> 1) + 1),
                ),
            );
            f[m] = acc;
            m += 1;
        }
        let h0 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(8), load!(24)),
            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 0),
        );
        let h1 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(8), load!(24)),
            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 1),
        );
        let g0 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(16)),
            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 0),
        );
        let g1 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(16)),
            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 1),
        );
        let e = [
            _mm_add_epi32(g0, h0),
            _mm_add_epi32(g1, h1),
            _mm_sub_epi32(g1, h1),
            _mm_sub_epi32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = _mm_add_epi32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = _mm_sub_epi32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = _mm_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = _mm_sub_epi32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = _mm_add_epi32(a[k], b[k]);
            out[k + 16] = _mm_sub_epi32(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_residual_add_u8x4(dst: *mut u8, v: __m128i, rnd: __m128i, sh: __m128i) {
    unsafe {
        let r = _mm_sra_epi32(_mm_add_epi32(v, rnd), sh);
        let r16 = _mm_packs_epi32(r, _mm_setzero_si128());
        let p4 = _mm_castps_si128(_mm_load_ss(dst.cast()));
        let p16 = _mm_cvtepu8_epi16(p4);
        let sum = _mm_adds_epi16(p16, r16);
        let out = _mm_packus_epi16(sum, _mm_setzero_si128());
        _mm_store_ss(dst.cast(), _mm_castsi128_ps(out));
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct16_i16x4_scratch4_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    unsafe {
        debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
        debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
        let z = _mm_setzero_si128();
        macro_rules! load {
            ($idx:expr) => {
                if ($idx) < ACTIVE {
                    sse41_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
                } else {
                    z
                }
            };
        }
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 1 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(1), load!(3)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 5 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(5), load!(7)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            if ACTIVE > 9 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(9), load!(11)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 2),
                    ),
                );
            }
            if ACTIVE > 13 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(13), load!(15)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 3),
                    ),
                );
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 2 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(2), load!(6)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 10 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(10), load!(14)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            d[m] = acc;
            m += 1;
        }
        let x412 = _mm_unpacklo_epi16(load!(4), load!(12));
        let f0 = if ACTIVE > 4 {
            _mm_madd_epi16(x412, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 0))
        } else {
            z
        };
        let f1 = if ACTIVE > 4 {
            _mm_madd_epi16(x412, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 1))
        } else {
            z
        };
        let x08 = _mm_unpacklo_epi16(load!(0), load!(8));
        let g0 = _mm_madd_epi16(x08, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 0));
        let g1 = _mm_madd_epi16(x08, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 1));
        let cc = [
            _mm_add_epi32(g0, f0),
            _mm_add_epi32(g1, f1),
            _mm_sub_epi32(g1, f1),
            _mm_sub_epi32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = _mm_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = _mm_sub_epi32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut k = 0usize;
        while k < 8 {
            _mm_storeu_si128(
                tmp.as_mut_ptr().add(base + k * 32) as *mut __m128i,
                _mm_add_epi32(a[k], b[k]),
            );
            _mm_storeu_si128(
                tmp.as_mut_ptr().add(base + (k + 8) * 32) as *mut __m128i,
                _mm_sub_epi32(a[7 - k], b[7 - k]),
            );
            k += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct16_i16x4_scratch4_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    unsafe {
        debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
        debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
        let z = _mm_setzero_si128();
        macro_rules! load {
            ($idx:expr) => {
                if ($idx) < ACTIVE {
                    sse41_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
                } else {
                    z
                }
            };
        }
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 1 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(1), load!(3)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 5 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(5), load!(7)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            if ACTIVE > 9 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(9), load!(11)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 2),
                    ),
                );
            }
            if ACTIVE > 13 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(13), load!(15)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 3),
                    ),
                );
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 2 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(2), load!(6)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 10 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(10), load!(14)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            d[m] = acc;
            m += 1;
        }
        let x412 = _mm_unpacklo_epi16(load!(4), load!(12));
        let f0 = if ACTIVE > 4 {
            _mm_madd_epi16(x412, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 0))
        } else {
            z
        };
        let f1 = if ACTIVE > 4 {
            _mm_madd_epi16(x412, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 1))
        } else {
            z
        };
        let x08 = _mm_unpacklo_epi16(load!(0), load!(8));
        let g0 = _mm_madd_epi16(x08, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 0));
        let g1 = _mm_madd_epi16(x08, sse41_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 1));
        let cc = [
            _mm_add_epi32(g0, f0),
            _mm_add_epi32(g1, f1),
            _mm_sub_epi32(g1, f1),
            _mm_sub_epi32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = _mm_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = _mm_sub_epi32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut k = 0usize;
        while k < 8 {
            sse41_residual_add_u8x4(
                dst.as_mut_ptr().add(dst_off + k * dst_stride + base),
                _mm_add_epi32(a[k], b[k]),
                rnd1,
                sh1,
            );
            sse41_residual_add_u8x4(
                dst.as_mut_ptr().add(dst_off + (k + 8) * dst_stride + base),
                _mm_sub_epi32(a[7 - k], b[7 - k]),
                rnd1,
                sh1,
            );
            k += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i16x4_scratch4_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    unsafe {
        debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
        debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
        let z = _mm_setzero_si128();
        macro_rules! load {
            ($idx:expr) => {
                if ($idx) < ACTIVE {
                    sse41_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
                } else {
                    z
                }
            };
        }
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let kbase = m * 16;
            let mut acc = z;
            let mut pair = 0usize;
            while pair < 8 {
                let i0 = 4 * pair + 1;
                if ACTIVE > i0 {
                    acc = _mm_add_epi32(
                        acc,
                        _mm_madd_epi16(
                            _mm_unpacklo_epi16(load!(i0), load!(i0 + 2)),
                            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KBP_X4, (kbase >> 1) + pair),
                        ),
                    );
                }
                pair += 1;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let kbase = m * 8;
            let mut acc = z;
            let mut pair = 0usize;
            while pair < 4 {
                let i0 = 8 * pair + 2;
                if ACTIVE > i0 {
                    acc = _mm_add_epi32(
                        acc,
                        _mm_madd_epi16(
                            _mm_unpacklo_epi16(load!(i0), load!(i0 + 4)),
                            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KDP_X4, (kbase >> 1) + pair),
                        ),
                    );
                }
                pair += 1;
            }
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 4 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(4), load!(12)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 20 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(20), load!(28)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            f[m] = acc;
            m += 1;
        }
        let x824 = _mm_unpacklo_epi16(load!(8), load!(24));
        let h0 = if ACTIVE > 8 {
            _mm_madd_epi16(x824, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 0))
        } else {
            z
        };
        let h1 = if ACTIVE > 8 {
            _mm_madd_epi16(x824, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 1))
        } else {
            z
        };
        let x016 = _mm_unpacklo_epi16(load!(0), load!(16));
        let g0 = _mm_madd_epi16(x016, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 0));
        let g1 = _mm_madd_epi16(x016, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 1));
        let e = [
            _mm_add_epi32(g0, h0),
            _mm_add_epi32(g1, h1),
            _mm_sub_epi32(g1, h1),
            _mm_sub_epi32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = _mm_add_epi32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = _mm_sub_epi32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = _mm_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = _mm_sub_epi32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut k = 0usize;
        while k < 16 {
            _mm_storeu_si128(
                tmp.as_mut_ptr().add(base + k * 32) as *mut __m128i,
                _mm_add_epi32(a[k], b[k]),
            );
            _mm_storeu_si128(
                tmp.as_mut_ptr().add(base + (k + 16) * 32) as *mut __m128i,
                _mm_sub_epi32(a[15 - k], b[15 - k]),
            );
            k += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i16x4_scratch4_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    unsafe {
        debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
        debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
        let z = _mm_setzero_si128();
        macro_rules! load {
            ($idx:expr) => {
                if ($idx) < ACTIVE {
                    sse41_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
                } else {
                    z
                }
            };
        }
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let kbase = m * 16;
            let mut acc = z;
            let mut pair = 0usize;
            while pair < 8 {
                let i0 = 4 * pair + 1;
                if ACTIVE > i0 {
                    acc = _mm_add_epi32(
                        acc,
                        _mm_madd_epi16(
                            _mm_unpacklo_epi16(load!(i0), load!(i0 + 2)),
                            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KBP_X4, (kbase >> 1) + pair),
                        ),
                    );
                }
                pair += 1;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let kbase = m * 8;
            let mut acc = z;
            let mut pair = 0usize;
            while pair < 4 {
                let i0 = 8 * pair + 2;
                if ACTIVE > i0 {
                    acc = _mm_add_epi32(
                        acc,
                        _mm_madd_epi16(
                            _mm_unpacklo_epi16(load!(i0), load!(i0 + 4)),
                            sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KDP_X4, (kbase >> 1) + pair),
                        ),
                    );
                }
                pair += 1;
            }
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 4 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(4), load!(12)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 20 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(20), load!(28)),
                        sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            f[m] = acc;
            m += 1;
        }
        let x824 = _mm_unpacklo_epi16(load!(8), load!(24));
        let h0 = if ACTIVE > 8 {
            _mm_madd_epi16(x824, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 0))
        } else {
            z
        };
        let h1 = if ACTIVE > 8 {
            _mm_madd_epi16(x824, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 1))
        } else {
            z
        };
        let x016 = _mm_unpacklo_epi16(load!(0), load!(16));
        let g0 = _mm_madd_epi16(x016, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 0));
        let g1 = _mm_madd_epi16(x016, sse41_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 1));
        let e = [
            _mm_add_epi32(g0, h0),
            _mm_add_epi32(g1, h1),
            _mm_sub_epi32(g1, h1),
            _mm_sub_epi32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = _mm_add_epi32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = _mm_sub_epi32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = _mm_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = _mm_sub_epi32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut k = 0usize;
        while k < 16 {
            sse41_residual_add_u8x4(
                dst.as_mut_ptr().add(dst_off + k * dst_stride + base),
                _mm_add_epi32(a[k], b[k]),
                rnd1,
                sh1,
            );
            sse41_residual_add_u8x4(
                dst.as_mut_ptr().add(dst_off + (k + 16) * dst_stride + base),
                _mm_sub_epi32(a[15 - k], b[15 - k]),
                rnd1,
                sh1,
            );
            k += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct16_i16x4_scratch4_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        sse41_dct16_i16x4_scratch4_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        sse41_dct16_i16x4_scratch4_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else {
        sse41_dct16_i16x4_scratch4_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i16x4_scratch4_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        sse41_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        sse41_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else if active <= 16 {
        sse41_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    } else {
        sse41_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 32>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct16_i16x4_scratch4_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    if active <= 4 {
        sse41_dct16_i16x4_scratch4_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    } else if active <= 8 {
        sse41_dct16_i16x4_scratch4_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    } else {
        sse41_dct16_i16x4_scratch4_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i16x4_scratch4_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    if active <= 4 {
        sse41_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    } else if active <= 8 {
        sse41_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    } else if active <= 16 {
        sse41_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    } else {
        sse41_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 32>(
            scratch, base, dst, dst_off, dst_stride, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct16_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [__m128i; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            sse41_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    sse41_dct16_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn sse41_dct32_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [__m128i; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            sse41_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    sse41_dct32_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_dct_i16_sse41_impl<const N: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_dct_i16_sse41_impl_const::<N, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_dct_i16_sse41_impl_const::<N, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_dct_i16_sse41_impl_const<const N: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(N == 16 || N == 32);
    debug_assert!(coeff.len() >= N * N);
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < N / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let ncols = ngrp * 4;
    let rnd = _mm_set1_epi32((1 << shift0) >> 1);
    let sh = _mm_cvtsi32_si128(shift0);
    let minv = _mm_set1_epi32(row_clip_min);
    let maxv = _mm_set1_epi32(row_clip_max);

    with_sse41_itx_i16_scratch(ITX_TMP_PIXELS, |scratch| {
        scratch.fill(0);
        let mut y = 0usize;
        while y + 8 <= ncols {
            if N == 16 {
                let lo = sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y);
                let hi =
                    sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y + 4);
                let mut x = 0usize;
                while x < 16 {
                    sse41_store8x8_i16_clip::<16>(
                        scratch,
                        y * 16 + x,
                        lo[x],
                        hi[x],
                        lo[x + 1],
                        hi[x + 1],
                        lo[x + 2],
                        hi[x + 2],
                        lo[x + 3],
                        hi[x + 3],
                        lo[x + 4],
                        hi[x + 4],
                        lo[x + 5],
                        hi[x + 5],
                        lo[x + 6],
                        hi[x + 6],
                        lo[x + 7],
                        hi[x + 7],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 8;
                }
            } else {
                let lo = sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y);
                let hi =
                    sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y + 4);
                let mut x = 0usize;
                while x < 32 {
                    sse41_store8x8_i16_clip::<32>(
                        scratch,
                        y * 32 + x,
                        lo[x],
                        hi[x],
                        lo[x + 1],
                        hi[x + 1],
                        lo[x + 2],
                        hi[x + 2],
                        lo[x + 3],
                        hi[x + 3],
                        lo[x + 4],
                        hi[x + 4],
                        lo[x + 5],
                        hi[x + 5],
                        lo[x + 6],
                        hi[x + 6],
                        lo[x + 7],
                        hi[x + 7],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= ncols {
            if N == 16 {
                let out = sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y);
                let mut x = 0usize;
                while x < 16 {
                    sse41_store4x4_i16_clip::<16>(
                        scratch,
                        y * 16 + x,
                        out[x],
                        out[x + 1],
                        out[x + 2],
                        out[x + 3],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 4;
                }
            } else {
                let out = sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y);
                let mut x = 0usize;
                while x < 32 {
                    sse41_store4x4_i16_clip::<32>(
                        scratch,
                        y * 32 + x,
                        out[x],
                        out[x + 1],
                        out[x + 2],
                        out[x + 3],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 4;
                }
            }
            y += 4;
        }
        coeff[..N * N].fill(0);

        let mut x = 0usize;
        while x < N {
            if N == 16 {
                sse41_dct16_i16x4_scratch4_stride_eob_store::<16>(scratch, x, ncols, tmp);
            } else {
                sse41_dct32_i16x4_scratch4_stride_eob_store::<32>(scratch, x, ncols, tmp);
            }
            x += 4;
        }
    });
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_dct_i16_sse41_fused_8bpc_impl_const<const N: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    debug_assert!(N == 16 || N == 32);
    debug_assert!(coeff.len() >= N * N);
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < N / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let ncols = ngrp * 4;
    let rnd = _mm_set1_epi32((1 << shift0) >> 1);
    let sh = _mm_cvtsi32_si128(shift0);
    let minv = _mm_set1_epi32(row_clip_min);
    let maxv = _mm_set1_epi32(row_clip_max);

    with_sse41_itx_i16_scratch(ITX_TMP_PIXELS, |scratch| {
        scratch.fill(0);
        let mut y = 0usize;
        while y + 8 <= ncols {
            if N == 16 {
                let lo = sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y);
                let hi =
                    sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y + 4);
                let mut x = 0usize;
                while x < 16 {
                    sse41_store8x8_i16_clip::<16>(
                        scratch,
                        y * 16 + x,
                        lo[x],
                        hi[x],
                        lo[x + 1],
                        hi[x + 1],
                        lo[x + 2],
                        hi[x + 2],
                        lo[x + 3],
                        hi[x + 3],
                        lo[x + 4],
                        hi[x + 4],
                        lo[x + 5],
                        hi[x + 5],
                        lo[x + 6],
                        hi[x + 6],
                        lo[x + 7],
                        hi[x + 7],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 8;
                }
            } else {
                let lo = sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y);
                let hi =
                    sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y + 4);
                let mut x = 0usize;
                while x < 32 {
                    sse41_store8x8_i16_clip::<32>(
                        scratch,
                        y * 32 + x,
                        lo[x],
                        hi[x],
                        lo[x + 1],
                        hi[x + 1],
                        lo[x + 2],
                        hi[x + 2],
                        lo[x + 3],
                        hi[x + 3],
                        lo[x + 4],
                        hi[x + 4],
                        lo[x + 5],
                        hi[x + 5],
                        lo[x + 6],
                        hi[x + 6],
                        lo[x + 7],
                        hi[x + 7],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= ncols {
            if N == 16 {
                let out = sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y);
                let mut x = 0usize;
                while x < 16 {
                    sse41_store4x4_i16_clip::<16>(
                        scratch,
                        y * 16 + x,
                        out[x],
                        out[x + 1],
                        out[x + 2],
                        out[x + 3],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 4;
                }
            } else {
                let out = sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y);
                let mut x = 0usize;
                while x < 32 {
                    sse41_store4x4_i16_clip::<32>(
                        scratch,
                        y * 32 + x,
                        out[x],
                        out[x + 1],
                        out[x + 2],
                        out[x + 3],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    x += 4;
                }
            }
            y += 4;
        }
        coeff[..N * N].fill(0);

        let rnd1 = _mm_set1_epi32((1 << shift1) >> 1);
        let sh1 = _mm_cvtsi32_si128(shift1);
        let mut x = 0usize;
        while x < N {
            if N == 16 {
                sse41_dct16_i16x4_scratch4_stride_eob_add_u8::<16>(
                    scratch, x, ncols, dst, dst_off, dst_stride, rnd1, sh1,
                );
            } else {
                sse41_dct32_i16x4_scratch4_stride_eob_add_u8::<32>(
                    scratch, x, ncols, dst, dst_off, dst_stride, rnd1, sh1,
                );
            }
            x += 4;
        }
    });
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tx_dequant_dense_sse41_i32_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_sse41_i32_impl_const::<N, W, H, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_sse41_i32_impl_const::<N, W, H, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tx_dequant_dense_sse41_i32_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = _mm_setzero_si128();
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);

        let mut y = 0usize;
        while y + 4 <= nrows {
            let mut m = 0usize;
            while m < W {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < W {
                    let mut v = _mm_loadu_si128(coeff.as_ptr().add(y + j * H) as *const __m128i);
                    if IS_RECT2 {
                        v = _mm_srai_epi32::<8>(_mm_add_epi32(
                            _mm_mullo_epi32(v, rect_mul),
                            rect_rnd,
                        ));
                    }
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(first_kind, W, m, j)),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(first_kind, W, m + 1, j)),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(first_kind, W, m + 2, j)),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(first_kind, W, m + 3, j)),
                        ),
                    );
                    j += 1;
                }
                let g = [a0, a1, a2, a3];
                sse41_store4x4_i32_clip(tmp, y * 32 + m, &g, rnd, sh, minv, maxv);
                m += 4;
            }
            y += 4;
        }
        while y < H {
            tmp[y * 32..y * 32 + W].fill(0);
            y += 1;
        }
        coeff[..W * H].fill(0);

        let mut x = 0usize;
        while x < W {
            let mut vin = [z; H];
            {
                let mut j = 0usize;
                while j < H {
                    vin[j] = _mm_loadu_si128(tmp.as_ptr().add(x + j * 32) as *const __m128i);
                    j += 1;
                }
            }
            let mut m = 0usize;
            while m < H {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < H {
                    let v = vin[j];
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(second_kind, H, m, j)),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(second_kind, H, m + 1, j)),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(second_kind, H, m + 2, j)),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(sse41_tx_dense_coeff(second_kind, H, m + 3, j)),
                        ),
                    );
                    j += 1;
                }
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, a0);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 1) * 32) as *mut __m128i, a1);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 2) * 32) as *mut __m128i, a2);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 3) * 32) as *mut __m128i, a3);
                m += 4;
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tx_dequant_dense_sse41_i16_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_sse41_i16_impl_const::<N, W, H, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_sse41_i16_impl_const::<N, W, H, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tx_dequant_dense_sse41_i16_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = _mm_setzero_si128();
    let rnd = _mm_set1_epi32((1 << shift0) >> 1);
    let sh = _mm_cvtsi32_si128(shift0);
    let minv = _mm_set1_epi32(row_clip_min);
    let maxv = _mm_set1_epi32(row_clip_max);

    with_sse41_itx_i16_scratch(N, |scratch| unsafe {
        scratch.fill(0);
        let mut y = 0usize;

        if first_kind == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = _mm_set1_epi32(sse41_identity_scale(W));
            while y + 4 <= nrows {
                let mut m = 0usize;
                while m < W {
                    let a0 = sse41_identity_i16x4_coeff_to_i32::<IS_RECT2>(
                        coeff,
                        y + (m + 0) * H,
                        scale,
                    );
                    let a1 = sse41_identity_i16x4_coeff_to_i32::<IS_RECT2>(
                        coeff,
                        y + (m + 1) * H,
                        scale,
                    );
                    let a2 = sse41_identity_i16x4_coeff_to_i32::<IS_RECT2>(
                        coeff,
                        y + (m + 2) * H,
                        scale,
                    );
                    let a3 = sse41_identity_i16x4_coeff_to_i32::<IS_RECT2>(
                        coeff,
                        y + (m + 3) * H,
                        scale,
                    );
                    sse41_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
                y += 4;
            }
        }

        while y + 8 <= nrows && first_kind == crate::itx_2d::TX_KIND_DCT && (W == 16 || W == 32) {
            if W == 16 {
                let lo = sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let hi =
                    sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 16 {
                    sse41_store8x8_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            } else {
                let lo = sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let hi =
                    sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 32 {
                    sse41_store8x8_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= nrows {
            if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
                let out = sse41_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 16 {
                    sse41_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
                let out = sse41_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 32 {
                    sse41_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 = sse41_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 =
                            sse41_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                        let x01 = _mm_unpacklo_epi16(x0, x1);
                        a0 = _mm_add_epi32(
                            a0,
                            _mm_madd_epi16(x01, sse41_tx_dense_coeff_pair(first_kind, W, m, j)),
                        );
                        a1 = _mm_add_epi32(
                            a1,
                            _mm_madd_epi16(x01, sse41_tx_dense_coeff_pair(first_kind, W, m + 1, j)),
                        );
                        a2 = _mm_add_epi32(
                            a2,
                            _mm_madd_epi16(x01, sse41_tx_dense_coeff_pair(first_kind, W, m + 2, j)),
                        );
                        a3 = _mm_add_epi32(
                            a3,
                            _mm_madd_epi16(x01, sse41_tx_dense_coeff_pair(first_kind, W, m + 3, j)),
                        );
                        j += 2;
                    }
                    sse41_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        sh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        coeff[..W * H].fill(0);

        let mut x = 0usize;
        if second_kind == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = _mm_set1_epi32(sse41_identity_scale(H));
            while x < W {
                let mut m = 0usize;
                while m < H {
                    let a = sse41_identity_i16x4_scratch_to_i32(scratch, x + m * W, scale);
                    _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, a);
                    m += 1;
                }
                x += 4;
            }
        }
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                sse41_dct16_i16x4_scratch4_stride_eob_store::<W>(scratch, x, nrows, tmp);
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                sse41_dct32_i16x4_scratch4_stride_eob_store::<W>(scratch, x, nrows, tmp);
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = sse41_load4_i16_scratch(scratch, x + j * W);
                        let x1 = sse41_load4_i16_scratch(scratch, x + (j + 1) * W);
                        let x01 = _mm_unpacklo_epi16(x0, x1);
                        a0 = _mm_add_epi32(
                            a0,
                            _mm_madd_epi16(x01, sse41_tx_dense_coeff_pair(second_kind, H, m, j)),
                        );
                        a1 = _mm_add_epi32(
                            a1,
                            _mm_madd_epi16(
                                x01,
                                sse41_tx_dense_coeff_pair(second_kind, H, m + 1, j),
                            ),
                        );
                        a2 = _mm_add_epi32(
                            a2,
                            _mm_madd_epi16(
                                x01,
                                sse41_tx_dense_coeff_pair(second_kind, H, m + 2, j),
                            ),
                        );
                        a3 = _mm_add_epi32(
                            a3,
                            _mm_madd_epi16(
                                x01,
                                sse41_tx_dense_coeff_pair(second_kind, H, m + 3, j),
                            ),
                        );
                        j += 2;
                    }
                    _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, a0);
                    _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 1) * 32) as *mut __m128i, a1);
                    _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 2) * 32) as *mut __m128i, a2);
                    _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 3) * 32) as *mut __m128i, a3);
                    m += 4;
                }
            }
            x += 4;
        }
    });
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tx_dequant_8x8_sse41_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_8x8_sse41_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_8x8_sse41_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn tx_dequant_8x8_sse41_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 64);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 2 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 8 {
                let g = sse41_tx8_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, first_kind, x);
                sse41_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
                x += 4;
            }
            y += 4;
        }
        while y < 8 {
            tmp[y * 32..y * 32 + 8].fill(0);
            y += 1;
        }
        coeff[..64].fill(0);
        let mut x = 0usize;
        while x < 8 {
            let g_lo = sse41_tx8_i32x4_from_tmp4(tmp, x, second_kind, 0);
            let g_hi = sse41_tx8_i32x4_from_tmp4(tmp, x, second_kind, 4);
            for (m, g) in [(0usize, &g_lo), (4, &g_hi)] {
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, g[0]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 1) * 32) as *mut __m128i, g[1]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 2) * 32) as *mut __m128i, g[2]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 3) * 32) as *mut __m128i, g[3]);
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_16x16_sse41_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_16x16_sse41_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_16x16_sse41_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_16x16_sse41_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 256);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let z = _mm_setzero_si128();
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);

        macro_rules! dct16x4_coeff {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let mut v =
                        _mm_loadu_si128(coeff.as_ptr().add($base + j * 16) as *const __m128i);
                    if IS_RECT2 {
                        v = _mm_srai_epi32::<8>(_mm_add_epi32(
                            _mm_mullo_epi32(v, rect_mul),
                            rect_rnd,
                        ));
                    }
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]),
                        ),
                    );
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }
        macro_rules! dct16x4_tmp {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let v = _mm_loadu_si128(tmp.as_ptr().add($base + j * 32) as *const __m128i);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]),
                        ),
                    );
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }

        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 16 {
                let g = dct16x4_coeff!(y, x);
                sse41_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
                x += 4;
            }
            y += 4;
        }
        while y < 16 {
            tmp[y * 32..y * 32 + 16].fill(0);
            y += 1;
        }
        coeff[..256].fill(0);

        let mut x = 0usize;
        while x < 16 {
            let g0 = dct16x4_tmp!(x, 0);
            let g4 = dct16x4_tmp!(x, 4);
            let g8 = dct16x4_tmp!(x, 8);
            let g12 = dct16x4_tmp!(x, 12);
            for (m, g) in [(0usize, &g0), (4, &g4), (8, &g8), (12, &g12)] {
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, g[0]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 1) * 32) as *mut __m128i, g[1]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 2) * 32) as *mut __m128i, g[2]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 3) * 32) as *mut __m128i, g[3]);
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_32x32_sse41_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_32x32_sse41_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_32x32_sse41_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_32x32_sse41_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 1024);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 8 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 32 {
                let g = sse41_dct32_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, x);
                sse41_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
                x += 4;
            }
            y += 4;
        }
        while y < 32 {
            tmp[y * 32..y * 32 + 32].fill(0);
            y += 1;
        }
        coeff[..1024].fill(0);
        let mut x = 0usize;
        while x < 32 {
            let groups = [
                sse41_dct32_i32x4_from_tmp4(tmp, x, 0),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 4),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 8),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 12),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 16),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 20),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 24),
                sse41_dct32_i32x4_from_tmp4(tmp, x, 28),
            ];
            let mut m = 0usize;
            while m < 32 {
                let g = &groups[m / 4];
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, g[0]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 1) * 32) as *mut __m128i, g[1]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 2) * 32) as *mut __m128i, g[2]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 3) * 32) as *mut __m128i, g[3]);
                m += 4;
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_4x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<16, 4, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_8x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_8x8_sse41_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_16x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_16x16_sse41_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_32x32_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_sse41_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_64x64_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<1024, 32, 32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_4x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<16, 4, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_8x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_8x8_sse41_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_16x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    iadst_dequant_16x16_sse41_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_4x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<32, 4, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_8x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<32, 8, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_8x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<128, 8, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_16x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<128, 16, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_16x32_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<512, 16, 32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_32x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<512, 32, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_4x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<64, 4, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_16x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<64, 16, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_8x32_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<256, 8, 32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_32x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<256, 32, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_4x32_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<128, 4, 32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_32x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i32_impl::<128, 32, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_4x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<32, 4, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_8x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<32, 8, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_8x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<128, 8, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_16x8_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<128, 16, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_4x16_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<64, 4, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_16x4_sse41(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i32_impl::<64, 16, 4>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

// Low-bit-depth i16 coefficient entry points.

macro_rules! idct_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $s:expr) => {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            tx_dequant_dense_sse41_i16_impl::<{ $n }, { $s }, { $s }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                crate::itx_2d::TX_KIND_DCT,
                crate::itx_2d::TX_KIND_DCT,
            )
        }
    };
}
macro_rules! iadst_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $s:expr) => {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            tx_dequant_dense_sse41_i16_impl::<{ $n }, { $s }, { $s }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                first_kind,
                second_kind,
            )
        }
    };
}
macro_rules! idct_rect_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $w:expr, $h:expr) => {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            tx_dequant_dense_sse41_i16_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                crate::itx_2d::TX_KIND_DCT,
                crate::itx_2d::TX_KIND_DCT,
            )
        }
    };
}
macro_rules! iadst_rect_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $w:expr, $h:expr) => {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            tx_dequant_dense_sse41_i16_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                first_kind,
                second_kind,
            )
        }
    };
}
idct_i16_fn!(
    idct_dequant_4x4_i16_sse41,
    idct_dequant_4x4_i16_sse41_impl,
    16,
    4
);
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_8x8_i16_sse41(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_sse41_i16_impl::<64, 8, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_dequant_dct_i16_sse41_fused_8bpc_impl<const N: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        idct_dequant_dct_i16_sse41_fused_8bpc_impl_const::<N, true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        idct_dequant_dct_i16_sse41_fused_8bpc_impl_const::<N, false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_16x16_i16_sse41_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    idct_dequant_dct_i16_sse41_fused_8bpc_impl::<16>(
        coeff,
        dst,
        dst_off,
        dst_stride,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_32x32_i16_sse41_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    idct_dequant_dct_i16_sse41_fused_8bpc_impl::<32>(
        coeff,
        dst,
        dst_off,
        dst_stride,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_16x16_i16_sse41(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_sse41_impl::<16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_32x32_i16_sse41(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_sse41_impl::<32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

idct_i16_fn!(
    idct_dequant_64x64_i16_sse41,
    idct_dequant_64x64_i16_sse41_impl,
    1024,
    32
);
iadst_i16_fn!(
    iadst_dequant_4x4_i16_sse41,
    iadst_dequant_4x4_i16_sse41_impl,
    16,
    4
);
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_8x8_i16_sse41(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i16_impl::<64, 8, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn iadst_dequant_16x16_i16_sse41(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_sse41_i16_impl::<256, 16, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

idct_rect_i16_fn!(
    idct_dequant_4x8_i16_sse41,
    idct_dequant_4x8_i16_sse41_impl,
    32,
    4,
    8
);
idct_rect_i16_fn!(
    idct_dequant_8x4_i16_sse41,
    idct_dequant_8x4_i16_sse41_impl,
    32,
    8,
    4
);
idct_rect_i16_fn!(
    idct_dequant_8x16_i16_sse41,
    idct_dequant_8x16_i16_sse41_impl,
    128,
    8,
    16
);
idct_rect_i16_fn!(
    idct_dequant_16x8_i16_sse41,
    idct_dequant_16x8_i16_sse41_impl,
    128,
    16,
    8
);
idct_rect_i16_fn!(
    idct_dequant_16x32_i16_sse41,
    idct_dequant_16x32_i16_sse41_impl,
    512,
    16,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x16_i16_sse41,
    idct_dequant_32x16_i16_sse41_impl,
    512,
    32,
    16
);
idct_rect_i16_fn!(
    idct_dequant_4x16_i16_sse41,
    idct_dequant_4x16_i16_sse41_impl,
    64,
    4,
    16
);
idct_rect_i16_fn!(
    idct_dequant_16x4_i16_sse41,
    idct_dequant_16x4_i16_sse41_impl,
    64,
    16,
    4
);
idct_rect_i16_fn!(
    idct_dequant_8x32_i16_sse41,
    idct_dequant_8x32_i16_sse41_impl,
    256,
    8,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x8_i16_sse41,
    idct_dequant_32x8_i16_sse41_impl,
    256,
    32,
    8
);
idct_rect_i16_fn!(
    idct_dequant_4x32_i16_sse41,
    idct_dequant_4x32_i16_sse41_impl,
    128,
    4,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x4_i16_sse41,
    idct_dequant_32x4_i16_sse41_impl,
    128,
    32,
    4
);
iadst_rect_i16_fn!(
    iadst_dequant_4x8_i16_sse41,
    iadst_dequant_4x8_i16_sse41_impl,
    32,
    4,
    8
);
iadst_rect_i16_fn!(
    iadst_dequant_8x4_i16_sse41,
    iadst_dequant_8x4_i16_sse41_impl,
    32,
    8,
    4
);
iadst_rect_i16_fn!(
    iadst_dequant_8x16_i16_sse41,
    iadst_dequant_8x16_i16_sse41_impl,
    128,
    8,
    16
);
iadst_rect_i16_fn!(
    iadst_dequant_16x8_i16_sse41,
    iadst_dequant_16x8_i16_sse41_impl,
    128,
    16,
    8
);
iadst_rect_i16_fn!(
    iadst_dequant_4x16_i16_sse41,
    iadst_dequant_4x16_i16_sse41_impl,
    64,
    4,
    16
);
iadst_rect_i16_fn!(
    iadst_dequant_16x4_i16_sse41,
    iadst_dequant_16x4_i16_sse41_impl,
    64,
    16,
    4
);
