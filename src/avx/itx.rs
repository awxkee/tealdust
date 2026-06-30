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

use crate::itx_2d::ITX_TMP_PIXELS;

use core::cell::RefCell;

thread_local! {
    static AVX2_ITX_I16_SCRATCH: RefCell<[i16; ITX_TMP_PIXELS]> = const { RefCell::new([0i16; ITX_TMP_PIXELS]) };
}

#[inline(always)]
fn with_avx2_itx_i16_scratch<R>(len: usize, f: impl FnOnce(&mut [i16]) -> R) -> R {
    assert!(len <= ITX_TMP_PIXELS);
    AVX2_ITX_I16_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        f(&mut scratch[..len])
    })
}

// Concrete 32x32 DCT kernels.  These are intentionally backend-local and do not
// pass through DctSimd4/DctWide or any generic 1-D transform wrapper.

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i32x4_impl(s: &[__m128i; 16]) -> [__m128i; 16] {
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
#[target_feature(enable = "avx2")]
fn avx2_adst16_i32x4_impl(s: &[__m128i; 16], flip: bool) -> [__m128i; 16] {
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
#[target_feature(enable = "avx2")]
fn avx2_tx16_i32x4_impl(s: &[__m128i; 16], kind: usize) -> [__m128i; 16] {
    match kind {
        crate::itx_2d::TX_KIND_DCT => avx2_dct16_i32x4_impl(s),
        crate::itx_2d::TX_KIND_ADST => avx2_adst16_i32x4_impl(s, false),
        crate::itx_2d::TX_KIND_FLIPADST => avx2_adst16_i32x4_impl(s, true),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn iadst_dequant_16x16_avx2_i32_impl(
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
        iadst_dequant_16x16_avx2_i32_impl_const::<true>(
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
        iadst_dequant_16x16_avx2_i32_impl_const::<false>(
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
#[target_feature(enable = "avx2")]
fn iadst_dequant_16x16_avx2_i32_impl_const<const IS_RECT2: bool>(
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
    let rnd256 = _mm256_set1_epi32((1 << shift0) >> 1);
    let minv256 = _mm256_set1_epi32(row_clip_min);
    let maxv256 = _mm256_set1_epi32(row_clip_max);

    let mut y = 0usize;
    while y + 8 <= ncols {
        let out = avx2_tx16_i32x8_from_coeff8_const::<IS_RECT2>(coeff, y, 16, first_kind);
        avx2_store8x8_i32_clip(
            tmp,
            y * 32,
            array_ref8_i32x8(&out, 0),
            rnd256,
            sh,
            minv256,
            maxv256,
        );
        avx2_store8x8_i32_clip(
            tmp,
            y * 32 + 8,
            array_ref8_i32x8(&out, 8),
            rnd256,
            sh,
            minv256,
            maxv256,
        );
        y += 8;
    }
    if y + 4 <= ncols {
        let mut s = [_mm_setzero_si128(); 16];
        let mut j = 0usize;
        while j < 16 {
            let mut v =
                unsafe { _mm_loadu_si128(coeff.as_ptr().add(y + j * 16) as *const __m128i) };
            if IS_RECT2 {
                v = _mm_srai_epi32::<8>(_mm_add_epi32(
                    _mm_mullo_epi32(v, _mm_set1_epi32(181)),
                    _mm_set1_epi32(128),
                ));
            }
            s[j] = v;
            j += 1;
        }
        let out = avx2_tx16_i32x4_impl(&s, first_kind);
        let mut x = 0usize;
        while x < 16 {
            let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
            avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
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
    while x + 8 <= 16 {
        let out = avx2_tx16_i32x8_from_tmp8(tmp, x, second_kind);
        let mut m = 0usize;
        while m < 16 {
            unsafe {
                _mm256_storeu_si256(tmp.as_mut_ptr().add(x + m * 32) as *mut __m256i, out[m]);
            }
            m += 1;
        }
        x += 8;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store4x4_i32_clip(
    dst: &mut [i32; ITX_TMP_PIXELS],
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
        _mm_storeu_si128(tmp_ptr(dst, off), r0);
        _mm_storeu_si128(tmp_ptr(dst, off + 32), r1);
        _mm_storeu_si128(tmp_ptr(dst, off + 64), r2);
        _mm_storeu_si128(tmp_ptr(dst, off + 96), r3);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store8x8_i32_clip(
    dst: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    v: &[__m256i; 8],
    rnd: __m256i,
    sh: __m128i,
    minv: __m256i,
    maxv: __m256i,
) {
    unsafe {
        debug_assert!(off + 7 * 32 + 8 <= dst.len());
        macro_rules! clip {
            ($x:expr) => {{
                _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32($x, rnd), sh), minv),
                    maxv,
                )
            }};
        }
        let r0 = clip!(v[0]);
        let r1 = clip!(v[1]);
        let r2 = clip!(v[2]);
        let r3 = clip!(v[3]);
        let r4 = clip!(v[4]);
        let r5 = clip!(v[5]);
        let r6 = clip!(v[6]);
        let r7 = clip!(v[7]);

        let t0 = _mm256_unpacklo_epi32(r0, r1);
        let t1 = _mm256_unpackhi_epi32(r0, r1);
        let t2 = _mm256_unpacklo_epi32(r2, r3);
        let t3 = _mm256_unpackhi_epi32(r2, r3);
        let t4 = _mm256_unpacklo_epi32(r4, r5);
        let t5 = _mm256_unpackhi_epi32(r4, r5);
        let t6 = _mm256_unpacklo_epi32(r6, r7);
        let t7 = _mm256_unpackhi_epi32(r6, r7);

        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);

        let ptr = dst.as_mut_ptr().add(off) as *mut __m256i;
        _mm256_storeu_si256(ptr, _mm256_permute2x128_si256::<0x20>(u0, u4));
        _mm256_storeu_si256(ptr.add(32 / 8), _mm256_permute2x128_si256::<0x20>(u1, u5));
        _mm256_storeu_si256(
            ptr.add((2 * 32) / 8),
            _mm256_permute2x128_si256::<0x20>(u2, u6),
        );
        _mm256_storeu_si256(
            ptr.add((3 * 32) / 8),
            _mm256_permute2x128_si256::<0x20>(u3, u7),
        );
        _mm256_storeu_si256(
            ptr.add((4 * 32) / 8),
            _mm256_permute2x128_si256::<0x31>(u0, u4),
        );
        _mm256_storeu_si256(
            ptr.add((5 * 32) / 8),
            _mm256_permute2x128_si256::<0x31>(u1, u5),
        );
        _mm256_storeu_si256(
            ptr.add((6 * 32) / 8),
            _mm256_permute2x128_si256::<0x31>(u2, u6),
        );
        _mm256_storeu_si256(
            ptr.add((7 * 32) / 8),
            _mm256_permute2x128_si256::<0x31>(u3, u7),
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store_i32x8_rows(tmp: &mut [i32; ITX_TMP_PIXELS], base: usize, out: &[__m256i; 8]) {
    unsafe {
        debug_assert!(base + 7 * 32 + 8 <= tmp.len());
        let ptr = tmp.as_mut_ptr();
        _mm256_storeu_si256(ptr.add(base) as *mut __m256i, out[0]);
        _mm256_storeu_si256(ptr.add(base + 32) as *mut __m256i, out[1]);
        _mm256_storeu_si256(ptr.add(base + 2 * 32) as *mut __m256i, out[2]);
        _mm256_storeu_si256(ptr.add(base + 3 * 32) as *mut __m256i, out[3]);
        _mm256_storeu_si256(ptr.add(base + 4 * 32) as *mut __m256i, out[4]);
        _mm256_storeu_si256(ptr.add(base + 5 * 32) as *mut __m256i, out[5]);
        _mm256_storeu_si256(ptr.add(base + 6 * 32) as *mut __m256i, out[6]);
        _mm256_storeu_si256(ptr.add(base + 7 * 32) as *mut __m256i, out[7]);
    }
}

#[inline(always)]
fn array_ref8_i32x8(src: &[__m256i], off: usize) -> &[__m256i; 8] {
    debug_assert!(off + 8 <= src.len());
    unsafe { &*(src.as_ptr().add(off) as *const [__m256i; 8]) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store4x4_i16_clip<const STRIDE: usize>(
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
#[target_feature(enable = "avx2")]
fn avx2_load4_i16_scratch(src: &[i16], off: usize) -> __m128i {
    debug_assert!(off + 4 <= src.len());
    unsafe { _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_load8_i16_scratch(src: &[i16], off: usize) -> __m128i {
    debug_assert!(off + 8 <= src.len());
    unsafe { _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_load16_i16_scratch(src: &[i16], off: usize) -> __m256i {
    debug_assert!(off + 16 <= src.len());
    unsafe { _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_pair8_i16(a: __m128i, b: __m128i) -> __m256i {
    let lo = _mm_unpacklo_epi16(a, b);
    let hi = _mm_unpackhi_epi16(a, b);
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[inline]
fn tmp_ptr(dst: &mut [i32; ITX_TMP_PIXELS], off: usize) -> *mut __m128i {
    unsafe { dst.as_mut_ptr().add(off) as *mut __m128i }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_load8_i32_coeff_const<const IS_RECT2: bool>(src: &[i32], off: usize) -> __m256i {
    debug_assert!(off + 8 <= src.len());
    let mut v = unsafe { _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i) };
    if IS_RECT2 {
        v = _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(v, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ));
    }
    v
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i32x8_impl(s: &[__m256i; 16]) -> [__m256i; 16] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            let k = _mm256_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m]);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s[j], k));
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_adst16_i32x8_impl(s: &[__m256i; 16], flip: bool) -> [__m256i; 16] {
    let rows = if flip {
        &crate::itx_1d::FLIPADST16_KERNEL_ROWS
    } else {
        &crate::itx_1d::ADST16_KERNEL_ROWS
    };
    let z = _mm256_setzero_si256();
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let row = &rows[m];
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            let k = _mm256_set1_epi32(row[j] as i32);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s[j], k));
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx16_i32x8_impl(s: &[__m256i; 16], kind: usize) -> [__m256i; 16] {
    match kind {
        crate::itx_2d::TX_KIND_DCT => avx2_dct16_i32x8_impl(s),
        crate::itx_2d::TX_KIND_ADST => avx2_adst16_i32x8_impl(s, false),
        crate::itx_2d::TX_KIND_FLIPADST => avx2_adst16_i32x8_impl(s, true),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx16_i32x8_from_coeff8_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    stride: usize,
    kind: usize,
) -> [__m256i; 16] {
    let z = _mm256_setzero_si256();
    let mut s = [z; 16];
    let mut j = 0usize;
    while j < 16 {
        s[j] = avx2_load8_i32_coeff_const::<IS_RECT2>(coeff, base + j * stride);
        j += 1;
    }
    avx2_tx16_i32x8_impl(&s, kind)
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx16_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    stride: usize,
    kind: usize,
) -> [__m128i; 16] {
    let z = _mm_setzero_si128();
    let mut s = [z; 16];
    let rect_mul = _mm_set1_epi32(181);
    let rect_rnd = _mm_set1_epi32(128);
    let mut j = 0usize;
    while j < 16 {
        let mut v =
            unsafe { _mm_loadu_si128(coeff.as_ptr().add(base + j * stride) as *const __m128i) };
        if IS_RECT2 {
            v = _mm_srai_epi32::<8>(_mm_add_epi32(_mm_mullo_epi32(v, rect_mul), rect_rnd));
        }
        s[j] = v;
        j += 1;
    }
    avx2_tx16_i32x4_impl(&s, kind)
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx16_i32x8_from_tmp8(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    kind: usize,
) -> [__m256i; 16] {
    let z = _mm256_setzero_si256();
    let mut s = [z; 16];
    let mut j = 0usize;
    while j < 16 {
        unsafe {
            s[j] = _mm256_loadu_si256(tmp.as_ptr().add(base + j * 32) as *const __m256i);
        }
        j += 1;
    }
    avx2_tx16_i32x8_impl(&s, kind)
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i32x8_from_coeff8_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    m: usize,
) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 8];
    let mut j = 0usize;
    while j < 32 {
        let v = avx2_load8_i32_coeff_const::<IS_RECT2>(coeff, base + j * 32);
        let mut lane = 0usize;
        while lane < 8 {
            let k = _mm256_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + lane]);
            out[lane] = _mm256_add_epi32(out[lane], _mm256_mullo_epi32(v, k));
            lane += 1;
        }
        j += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i32x8_from_tmp8(tmp: &[i32; ITX_TMP_PIXELS], base: usize, m: usize) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 8];
    let mut j = 0usize;
    while j < 32 {
        let v = unsafe { _mm256_loadu_si256(tmp.as_ptr().add(base + j * 32) as *const __m256i) };
        let mut lane = 0usize;
        while lane < 8 {
            let k = _mm256_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + lane]);
            out[lane] = _mm256_add_epi32(out[lane], _mm256_mullo_epi32(v, k));
            lane += 1;
        }
        j += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx8_i32x8_from_coeff8_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    kind: usize,
) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 8];
    let mut j = 0usize;
    while j < 8 {
        let v = avx2_load8_i32_coeff_const::<IS_RECT2>(coeff, base + j * 8);
        let mut lane = 0usize;
        while lane < 8 {
            let k = _mm256_set1_epi32(tx8_coeff(kind, lane, j));
            out[lane] = _mm256_add_epi32(out[lane], _mm256_mullo_epi32(v, k));
            lane += 1;
        }
        j += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx8_i32x8_from_tmp8(tmp: &[i32; ITX_TMP_PIXELS], base: usize, kind: usize) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 8];
    let mut j = 0usize;
    while j < 8 {
        let v = unsafe { _mm256_loadu_si256(tmp.as_ptr().add(base + j * 32) as *const __m256i) };
        let mut lane = 0usize;
        while lane < 8 {
            let k = _mm256_set1_epi32(tx8_coeff(kind, lane, j));
            out[lane] = _mm256_add_epi32(out[lane], _mm256_mullo_epi32(v, k));
            lane += 1;
        }
        j += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx_dense_i32x8_from_coeff8_const<const IS_RECT2: bool, const W: usize, const H: usize>(
    coeff: &[i32],
    base: usize,
    kind: usize,
    m: usize,
) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 8];
    let mut j = 0usize;
    while j < W {
        let v = avx2_load8_i32_coeff_const::<IS_RECT2>(coeff, base + j * H);
        let mut lane = 0usize;
        while lane < 8 {
            let k = _mm256_set1_epi32(avx2_tx_dense_coeff(kind, W, m + lane, j));
            out[lane] = _mm256_add_epi32(out[lane], _mm256_mullo_epi32(v, k));
            lane += 1;
        }
        j += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_tx_dense_i32x8_from_tmp8<const H: usize>(
    vin: &[__m256i; H],
    kind: usize,
    m: usize,
) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut out = [z; 8];
    let mut j = 0usize;
    while j < H {
        let v = vin[j];
        let mut lane = 0usize;
        while lane < 8 {
            let k = _mm256_set1_epi32(avx2_tx_dense_coeff(kind, H, m + lane, j));
            out[lane] = _mm256_add_epi32(out[lane], _mm256_mullo_epi32(v, k));
            lane += 1;
        }
        j += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i32x4_from_coeff4_const<const IS_RECT2: bool>(
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
#[target_feature(enable = "avx2")]
fn avx2_tx8_i32x4_from_coeff4_const<const IS_RECT2: bool>(
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
fn tx8_coeff(kind: usize, out: usize, input: usize) -> i32 {
    match kind {
        crate::itx_2d::TX_KIND_DCT => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_ADST => crate::itx_2d::ADST8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_FLIPADST => crate::itx_2d::ADST8_KW[(7 - out) * 8 + input] as i32,
        _ => unreachable!(),
    }
}

#[inline]
fn avx2_identity_coeff(n: usize, out: usize, input: usize) -> i32 {
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
fn avx2_identity_scale(n: usize) -> i32 {
    match n {
        4 => 128,
        8 => 181,
        16 => 256,
        32 => 362,
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_identity_i16x4_coeff_to_i32<const IS_RECT2: bool>(
    coeff: &[i16],
    off: usize,
    scale: __m128i,
) -> __m128i {
    let v = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, off);
    _mm_mullo_epi32(_mm_cvtepi16_epi32(v), scale)
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_identity_i16x4_scratch_to_i32(scratch: &[i16], off: usize, scale: __m128i) -> __m128i {
    let v = avx2_load4_i16_scratch(scratch, off);
    _mm_mullo_epi32(_mm_cvtepi16_epi32(v), scale)
}

#[inline]
fn avx2_tx_dense_coeff(kind: usize, n: usize, out: usize, input: usize) -> i32 {
    match (kind, n) {
        (crate::itx_2d::TX_KIND_DCT, 4) => crate::itx_2d::DCT4_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 8) => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 16) => crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out],
        (crate::itx_2d::TX_KIND_DCT, 32) => crate::itx_2d::DCT32_DENSE_KERNEL[input * 32 + out],
        (crate::itx_2d::TX_KIND_IDENTITY, 4)
        | (crate::itx_2d::TX_KIND_IDENTITY, 8)
        | (crate::itx_2d::TX_KIND_IDENTITY, 16)
        | (crate::itx_2d::TX_KIND_IDENTITY, 32) => avx2_identity_coeff(n, out, input),
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
#[target_feature(enable = "avx2")]
fn avx2_tx_dense_coeff_pair(kind: usize, n: usize, out: usize, input: usize) -> __m128i {
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
            let k0 = avx2_identity_coeff(n, out, input) as i16;
            let k1 = avx2_identity_coeff(n, out, input + 1) as i16;
            return avx2_coeff_pair_from_scalars_i16(k0, k1);
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
    avx2_coeff_pair_i16(table, idx)
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store16x8_i16_clip256<const STRIDE: usize>(
    scratch: &mut [i16],
    off: usize,
    v0: __m256i,
    v1: __m256i,
    v2: __m256i,
    v3: __m256i,
    v4: __m256i,
    v5: __m256i,
    v6: __m256i,
    v7: __m256i,
    v8: __m256i,
    v9: __m256i,
    v10: __m256i,
    v11: __m256i,
    v12: __m256i,
    v13: __m256i,
    v14: __m256i,
    v15: __m256i,
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) {
    unsafe {
        debug_assert!(STRIDE == 16 || STRIDE == 32);
        debug_assert!(off + 7 * STRIDE + 16 <= scratch.len());

        let rnd256 = _mm256_broadcastsi128_si256(rnd);
        let min256 = _mm256_broadcastsi128_si256(minv);
        let max256 = _mm256_broadcastsi128_si256(maxv);
        macro_rules! clip_pack {
            ($x:expr) => {{
                let c = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32($x, rnd256), sh), min256),
                    max256,
                );
                _mm_packs_epi32(_mm256_castsi256_si128(c), _mm256_extracti128_si256::<1>(c))
            }};
        }

        let r0 = clip_pack!(v0);
        let r1 = clip_pack!(v1);
        let r2 = clip_pack!(v2);
        let r3 = clip_pack!(v3);
        let r4 = clip_pack!(v4);
        let r5 = clip_pack!(v5);
        let r6 = clip_pack!(v6);
        let r7 = clip_pack!(v7);
        let r8 = clip_pack!(v8);
        let r9 = clip_pack!(v9);
        let r10 = clip_pack!(v10);
        let r11 = clip_pack!(v11);
        let r12 = clip_pack!(v12);
        let r13 = clip_pack!(v13);
        let r14 = clip_pack!(v14);
        let r15 = clip_pack!(v15);

        macro_rules! transpose8_named {
            (
                $a0:expr, $a1:expr, $a2:expr, $a3:expr,
                $a4:expr, $a5:expr, $a6:expr, $a7:expr =>
                $o0:ident, $o1:ident, $o2:ident, $o3:ident,
                $o4:ident, $o5:ident, $o6:ident, $o7:ident
            ) => {
                let t0 = _mm_unpacklo_epi16($a0, $a1);
                let t1 = _mm_unpackhi_epi16($a0, $a1);
                let t2 = _mm_unpacklo_epi16($a2, $a3);
                let t3 = _mm_unpackhi_epi16($a2, $a3);
                let t4 = _mm_unpacklo_epi16($a4, $a5);
                let t5 = _mm_unpackhi_epi16($a4, $a5);
                let t6 = _mm_unpacklo_epi16($a6, $a7);
                let t7 = _mm_unpackhi_epi16($a6, $a7);

                let u0 = _mm_unpacklo_epi32(t0, t2);
                let u1 = _mm_unpackhi_epi32(t0, t2);
                let u2 = _mm_unpacklo_epi32(t1, t3);
                let u3 = _mm_unpackhi_epi32(t1, t3);
                let u4 = _mm_unpacklo_epi32(t4, t6);
                let u5 = _mm_unpackhi_epi32(t4, t6);
                let u6 = _mm_unpacklo_epi32(t5, t7);
                let u7 = _mm_unpackhi_epi32(t5, t7);

                let $o0 = _mm_unpacklo_epi64(u0, u4);
                let $o1 = _mm_unpackhi_epi64(u0, u4);
                let $o2 = _mm_unpacklo_epi64(u1, u5);
                let $o3 = _mm_unpackhi_epi64(u1, u5);
                let $o4 = _mm_unpacklo_epi64(u2, u6);
                let $o5 = _mm_unpackhi_epi64(u2, u6);
                let $o6 = _mm_unpacklo_epi64(u3, u7);
                let $o7 = _mm_unpackhi_epi64(u3, u7);
            };
        }

        transpose8_named!(r0, r1, r2, r3, r4, r5, r6, r7 => l0, l1, l2, l3, l4, l5, l6, l7);
        transpose8_named!(r8, r9, r10, r11, r12, r13, r14, r15 => h0, h1, h2, h3, h4, h5, h6, h7);
        let ptr = scratch.as_mut_ptr();
        _mm_storeu_si128(ptr.add(off) as *mut __m128i, l0);
        _mm_storeu_si128(ptr.add(off + 8) as *mut __m128i, h0);
        _mm_storeu_si128(ptr.add(off + STRIDE) as *mut __m128i, l1);
        _mm_storeu_si128(ptr.add(off + STRIDE + 8) as *mut __m128i, h1);
        _mm_storeu_si128(ptr.add(off + 2 * STRIDE) as *mut __m128i, l2);
        _mm_storeu_si128(ptr.add(off + 2 * STRIDE + 8) as *mut __m128i, h2);
        _mm_storeu_si128(ptr.add(off + 3 * STRIDE) as *mut __m128i, l3);
        _mm_storeu_si128(ptr.add(off + 3 * STRIDE + 8) as *mut __m128i, h3);
        _mm_storeu_si128(ptr.add(off + 4 * STRIDE) as *mut __m128i, l4);
        _mm_storeu_si128(ptr.add(off + 4 * STRIDE + 8) as *mut __m128i, h4);
        _mm_storeu_si128(ptr.add(off + 5 * STRIDE) as *mut __m128i, l5);
        _mm_storeu_si128(ptr.add(off + 5 * STRIDE + 8) as *mut __m128i, h5);
        _mm_storeu_si128(ptr.add(off + 6 * STRIDE) as *mut __m128i, l6);
        _mm_storeu_si128(ptr.add(off + 6 * STRIDE + 8) as *mut __m128i, h6);
        _mm_storeu_si128(ptr.add(off + 7 * STRIDE) as *mut __m128i, l7);
        _mm_storeu_si128(ptr.add(off + 7 * STRIDE + 8) as *mut __m128i, h7);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store16x16_i16_clip256<const STRIDE: usize, const N: usize, const M: usize>(
    scratch: &mut [i16],
    off: usize,
    out: &[Avx2I32x16; N],
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) {
    unsafe {
        debug_assert!(STRIDE == 16 || STRIDE == 32);
        debug_assert!(M + 16 <= N);
        debug_assert!(off + 15 * STRIDE + 16 <= scratch.len());

        let rnd256 = _mm256_broadcastsi128_si256(rnd);
        let min256 = _mm256_broadcastsi128_si256(minv);
        let max256 = _mm256_broadcastsi128_si256(maxv);
        macro_rules! clip_pack16 {
            ($x:expr) => {{
                let x = $x;
                let l = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(x.lo, rnd256), sh), min256),
                    max256,
                );
                let h = _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(x.hi, rnd256), sh), min256),
                    max256,
                );
                let lo = _mm256_permute2x128_si256::<0x20>(l, h);
                let hi = _mm256_permute2x128_si256::<0x31>(l, h);
                _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(lo, hi))
            }};
        }

        let r0 = clip_pack16!(out[M]);
        let r1 = clip_pack16!(out[M + 1]);
        let r2 = clip_pack16!(out[M + 2]);
        let r3 = clip_pack16!(out[M + 3]);
        let r4 = clip_pack16!(out[M + 4]);
        let r5 = clip_pack16!(out[M + 5]);
        let r6 = clip_pack16!(out[M + 6]);
        let r7 = clip_pack16!(out[M + 7]);
        let r8 = clip_pack16!(out[M + 8]);
        let r9 = clip_pack16!(out[M + 9]);
        let r10 = clip_pack16!(out[M + 10]);
        let r11 = clip_pack16!(out[M + 11]);
        let r12 = clip_pack16!(out[M + 12]);
        let r13 = clip_pack16!(out[M + 13]);
        let r14 = clip_pack16!(out[M + 14]);
        let r15 = clip_pack16!(out[M + 15]);

        macro_rules! transpose8x16_named {
            (
                $a0:expr, $a1:expr, $a2:expr, $a3:expr,
                $a4:expr, $a5:expr, $a6:expr, $a7:expr =>
                $o0:ident, $o1:ident, $o2:ident, $o3:ident,
                $o4:ident, $o5:ident, $o6:ident, $o7:ident
            ) => {
                let t0 = _mm256_unpacklo_epi16($a0, $a1);
                let t1 = _mm256_unpackhi_epi16($a0, $a1);
                let t2 = _mm256_unpacklo_epi16($a2, $a3);
                let t3 = _mm256_unpackhi_epi16($a2, $a3);
                let t4 = _mm256_unpacklo_epi16($a4, $a5);
                let t5 = _mm256_unpackhi_epi16($a4, $a5);
                let t6 = _mm256_unpacklo_epi16($a6, $a7);
                let t7 = _mm256_unpackhi_epi16($a6, $a7);

                let u0 = _mm256_unpacklo_epi32(t0, t2);
                let u1 = _mm256_unpackhi_epi32(t0, t2);
                let u2 = _mm256_unpacklo_epi32(t1, t3);
                let u3 = _mm256_unpackhi_epi32(t1, t3);
                let u4 = _mm256_unpacklo_epi32(t4, t6);
                let u5 = _mm256_unpackhi_epi32(t4, t6);
                let u6 = _mm256_unpacklo_epi32(t5, t7);
                let u7 = _mm256_unpackhi_epi32(t5, t7);

                let $o0 = _mm256_unpacklo_epi64(u0, u4);
                let $o1 = _mm256_unpackhi_epi64(u0, u4);
                let $o2 = _mm256_unpacklo_epi64(u1, u5);
                let $o3 = _mm256_unpackhi_epi64(u1, u5);
                let $o4 = _mm256_unpacklo_epi64(u2, u6);
                let $o5 = _mm256_unpackhi_epi64(u2, u6);
                let $o6 = _mm256_unpacklo_epi64(u3, u7);
                let $o7 = _mm256_unpackhi_epi64(u3, u7);
            };
        }

        transpose8x16_named!(r0, r1, r2, r3, r4, r5, r6, r7 => a0, a1, a2, a3, a4, a5, a6, a7);
        transpose8x16_named!(r8, r9, r10, r11, r12, r13, r14, r15 => b0, b1, b2, b3, b4, b5, b6, b7);

        macro_rules! row_lo {
            ($a:expr, $b:expr) => {
                _mm256_permute2x128_si256::<0x20>($a, $b)
            };
        }
        macro_rules! row_hi {
            ($a:expr, $b:expr) => {
                _mm256_permute2x128_si256::<0x31>($a, $b)
            };
        }

        let ptr = scratch.as_mut_ptr();
        _mm256_storeu_si256(ptr.add(off) as *mut __m256i, row_lo!(a0, b0));
        _mm256_storeu_si256(ptr.add(off + STRIDE) as *mut __m256i, row_lo!(a1, b1));
        _mm256_storeu_si256(ptr.add(off + 2 * STRIDE) as *mut __m256i, row_lo!(a2, b2));
        _mm256_storeu_si256(ptr.add(off + 3 * STRIDE) as *mut __m256i, row_lo!(a3, b3));
        _mm256_storeu_si256(ptr.add(off + 4 * STRIDE) as *mut __m256i, row_lo!(a4, b4));
        _mm256_storeu_si256(ptr.add(off + 5 * STRIDE) as *mut __m256i, row_lo!(a5, b5));
        _mm256_storeu_si256(ptr.add(off + 6 * STRIDE) as *mut __m256i, row_lo!(a6, b6));
        _mm256_storeu_si256(ptr.add(off + 7 * STRIDE) as *mut __m256i, row_lo!(a7, b7));
        _mm256_storeu_si256(ptr.add(off + 8 * STRIDE) as *mut __m256i, row_hi!(a0, b0));
        _mm256_storeu_si256(ptr.add(off + 9 * STRIDE) as *mut __m256i, row_hi!(a1, b1));
        _mm256_storeu_si256(ptr.add(off + 10 * STRIDE) as *mut __m256i, row_hi!(a2, b2));
        _mm256_storeu_si256(ptr.add(off + 11 * STRIDE) as *mut __m256i, row_hi!(a3, b3));
        _mm256_storeu_si256(ptr.add(off + 12 * STRIDE) as *mut __m256i, row_hi!(a4, b4));
        _mm256_storeu_si256(ptr.add(off + 13 * STRIDE) as *mut __m256i, row_hi!(a5, b5));
        _mm256_storeu_si256(ptr.add(off + 14 * STRIDE) as *mut __m256i, row_hi!(a6, b6));
        _mm256_storeu_si256(ptr.add(off + 15 * STRIDE) as *mut __m256i, row_hi!(a7, b7));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_load4_i16_coeff_packed_const<const IS_RECT2: bool>(src: &[i16], off: usize) -> __m128i {
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
#[target_feature(enable = "avx2")]
fn avx2_load8_i16_coeff_packed_const<const IS_RECT2: bool>(src: &[i16], off: usize) -> __m128i {
    debug_assert!(off + 8 <= src.len());
    let mut v = unsafe { _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i) };
    if IS_RECT2 {
        // Same rect2 normalization as the x4 path, but keep all eight lanes
        // live so the first pass consumes an 8-column AVX chunk at a time.
        v = _mm_mulhrs_epi16(v, _mm_set1_epi16(0x5a80));
    }
    v
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_load16_i16_coeff_packed_const<const IS_RECT2: bool>(src: &[i16], off: usize) -> __m256i {
    debug_assert!(off + 16 <= src.len());
    let mut v = unsafe { _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i) };
    if IS_RECT2 {
        v = _mm256_mulhrs_epi16(v, _mm256_set1_epi16(0x5a80));
    }
    v
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_pair8_i16_from_rows(a: __m128i, b: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(_mm_unpacklo_epi16(a, b)),
        _mm_unpackhi_epi16(a, b),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_coeff_pair_from_scalars_i16(k0: i16, k1: i16) -> __m128i {
    _mm_set1_epi32(((k1 as u16 as i32) << 16) | (k0 as u16 as i32))
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_coeff_pair_i16(table: &[i32], idx: usize) -> __m128i {
    debug_assert!(idx * 4 + 4 <= table.len());
    unsafe { _mm_loadu_si128(table.as_ptr().add(idx * 4) as *const __m128i) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_coeff_pair_i16x8(table: &[i32], idx: usize) -> __m256i {
    _mm256_broadcastsi128_si256(avx2_coeff_pair_i16(table, idx))
}

#[derive(Clone, Copy)]
struct Avx2I32x16 {
    /// Columns 0..3 and 8..11 in the two AVX2 128-bit lanes.
    lo: __m256i,
    /// Columns 4..7 and 12..15 in the two AVX2 128-bit lanes.
    hi: __m256i,
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_i32x16_zero() -> Avx2I32x16 {
    let z = _mm256_setzero_si256();
    Avx2I32x16 { lo: z, hi: z }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_i32x16_add(a: Avx2I32x16, b: Avx2I32x16) -> Avx2I32x16 {
    Avx2I32x16 {
        lo: _mm256_add_epi32(a.lo, b.lo),
        hi: _mm256_add_epi32(a.hi, b.hi),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_i32x16_sub(a: Avx2I32x16, b: Avx2I32x16) -> Avx2I32x16 {
    Avx2I32x16 {
        lo: _mm256_sub_epi32(a.lo, b.lo),
        hi: _mm256_sub_epi32(a.hi, b.hi),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_madd_i16x16_pair(a: __m256i, b: __m256i, k: __m256i) -> Avx2I32x16 {
    Avx2I32x16 {
        lo: _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), k),
        hi: _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), k),
    }
}

macro_rules! avx2_dct16_i16x4_all_body {
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
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, base >> 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(5), load!(7)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(9), load!(11)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 2),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(13), load!(15)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 3),
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
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, base >> 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(10), load!(14)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, (base >> 1) + 1),
                ),
            );
            d[m] = acc;
            m += 1;
        }
        let f0 = _mm_add_epi32(
            _mm_madd_epi16(
                _mm_unpacklo_epi16(load!(4), load!(12)),
                avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 0),
            ),
            z,
        );
        let f1 = _mm_add_epi32(
            _mm_madd_epi16(
                _mm_unpacklo_epi16(load!(4), load!(12)),
                avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 1),
            ),
            z,
        );
        let g0 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(8)),
            avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 0),
        );
        let g1 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(8)),
            avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 1),
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

macro_rules! avx2_dct32_i16x4_all_body {
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KBP_X4, cb >> 1),
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KDP_X4, (base + p) >> 1),
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
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, base >> 1),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(20), load!(28)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, (base >> 1) + 1),
                ),
            );
            f[m] = acc;
            m += 1;
        }
        let h0 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(8), load!(24)),
            avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 0),
        );
        let h1 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(8), load!(24)),
            avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 1),
        );
        let g0 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(16)),
            avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 0),
        );
        let g1 = _mm_madd_epi16(
            _mm_unpacklo_epi16(load!(0), load!(16)),
            avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 1),
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

macro_rules! avx2_dct16_i16x8_all_body {
    () => {{
        let z = _mm256_setzero_si256();
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(1), load!(3)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, base >> 1),
                ),
            );
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(5), load!(7)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 1),
                ),
            );
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(9), load!(11)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 2),
                ),
            );
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(13), load!(15)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 3),
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
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(2), load!(6)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KDP_X4, base >> 1),
                ),
            );
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(10), load!(14)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KDP_X4, (base >> 1) + 1),
                ),
            );
            d[m] = acc;
            m += 1;
        }
        let f0 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(4), load!(12)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KFP_X4, 0),
        );
        let f1 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(4), load!(12)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KFP_X4, 1),
        );
        let g0 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(0), load!(8)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KGP_X4, 0),
        );
        let g1 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(0), load!(8)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KGP_X4, 1),
        );
        let cc = [
            _mm256_add_epi32(g0, f0),
            _mm256_add_epi32(g1, f1),
            _mm256_sub_epi32(g1, f1),
            _mm256_sub_epi32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = _mm256_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = _mm256_sub_epi32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = _mm256_add_epi32(a[k], b[k]);
            out[k + 8] = _mm256_sub_epi32(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! avx2_dct32_i16x8_all_body {
    () => {{
        let z = _mm256_setzero_si256();
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut p = 0usize;
            while p < 16 {
                let cb = base + p;
                let i0 = 2 * p + 1;
                acc = _mm256_add_epi32(
                    acc,
                    _mm256_madd_epi16(
                        avx2_pair8_i16_from_rows(load!(i0), load!(i0 + 2)),
                        avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KBP_X4, cb >> 1),
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
                acc = _mm256_add_epi32(
                    acc,
                    _mm256_madd_epi16(
                        avx2_pair8_i16_from_rows(load!(i0), load!(i0 + 4)),
                        avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KDP_X4, (base + p) >> 1),
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
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(4), load!(12)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KFP_X4, base >> 1),
                ),
            );
            acc = _mm256_add_epi32(
                acc,
                _mm256_madd_epi16(
                    avx2_pair8_i16_from_rows(load!(20), load!(28)),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KFP_X4, (base >> 1) + 1),
                ),
            );
            f[m] = acc;
            m += 1;
        }
        let h0 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(8), load!(24)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KHP_X4, 0),
        );
        let h1 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(8), load!(24)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KHP_X4, 1),
        );
        let g0 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(0), load!(16)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KGP_X4, 0),
        );
        let g1 = _mm256_madd_epi16(
            avx2_pair8_i16_from_rows(load!(0), load!(16)),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KGP_X4, 1),
        );
        let e = [
            _mm256_add_epi32(g0, h0),
            _mm256_add_epi32(g1, h1),
            _mm256_sub_epi32(g1, h1),
            _mm256_sub_epi32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = _mm256_add_epi32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = _mm256_sub_epi32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = _mm256_add_epi32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = _mm256_sub_epi32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = _mm256_add_epi32(a[k], b[k]);
            out[k + 16] = _mm256_sub_epi32(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_residual_add_u8x8(dst: *mut u8, v: __m256i, rnd: __m256i, sh: __m128i) {
    unsafe {
        let r = _mm256_sra_epi32(_mm256_add_epi32(v, rnd), sh);
        let r16 = _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
        let p8 = _mm_loadl_epi64(dst as *const __m128i);
        let p16 = _mm_cvtepu8_epi16(p8);
        let sum = _mm_adds_epi16(p16, r16);
        let out = _mm_packus_epi16(sum, _mm_setzero_si128());
        _mm_storel_epi64(dst as *mut __m128i, out);
    }
}

macro_rules! avx2_dct16_i16x16_all_body {
    () => {{
        let z = avx2_i32x16_zero();
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(1),
                    load!(3),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, base >> 1),
                ),
            );
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(5),
                    load!(7),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 1),
                ),
            );
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(9),
                    load!(11),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 2),
                ),
            );
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(13),
                    load!(15),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 3),
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
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(2),
                    load!(6),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KDP_X4, base >> 1),
                ),
            );
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(10),
                    load!(14),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KDP_X4, (base >> 1) + 1),
                ),
            );
            d[m] = acc;
            m += 1;
        }
        let f0 = avx2_madd_i16x16_pair(
            load!(4),
            load!(12),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KFP_X4, 0),
        );
        let f1 = avx2_madd_i16x16_pair(
            load!(4),
            load!(12),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KFP_X4, 1),
        );
        let g0 = avx2_madd_i16x16_pair(
            load!(0),
            load!(8),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KGP_X4, 0),
        );
        let g1 = avx2_madd_i16x16_pair(
            load!(0),
            load!(8),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT16_KGP_X4, 1),
        );
        let cc = [
            avx2_i32x16_add(g0, f0),
            avx2_i32x16_add(g1, f1),
            avx2_i32x16_sub(g1, f1),
            avx2_i32x16_sub(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = avx2_i32x16_add(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = avx2_i32x16_sub(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = avx2_i32x16_add(a[k], b[k]);
            out[k + 8] = avx2_i32x16_sub(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! avx2_dct32_i16x16_all_body {
    () => {{
        let z = avx2_i32x16_zero();
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut p = 0usize;
            while p < 16 {
                let cb = base + p;
                let i0 = 2 * p + 1;
                acc = avx2_i32x16_add(
                    acc,
                    avx2_madd_i16x16_pair(
                        load!(i0),
                        load!(i0 + 2),
                        avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KBP_X4, cb >> 1),
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
                acc = avx2_i32x16_add(
                    acc,
                    avx2_madd_i16x16_pair(
                        load!(i0),
                        load!(i0 + 4),
                        avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KDP_X4, (base + p) >> 1),
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
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(4),
                    load!(12),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KFP_X4, base >> 1),
                ),
            );
            acc = avx2_i32x16_add(
                acc,
                avx2_madd_i16x16_pair(
                    load!(20),
                    load!(28),
                    avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KFP_X4, (base >> 1) + 1),
                ),
            );
            f[m] = acc;
            m += 1;
        }
        let h0 = avx2_madd_i16x16_pair(
            load!(8),
            load!(24),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KHP_X4, 0),
        );
        let h1 = avx2_madd_i16x16_pair(
            load!(8),
            load!(24),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KHP_X4, 1),
        );
        let g0 = avx2_madd_i16x16_pair(
            load!(0),
            load!(16),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KGP_X4, 0),
        );
        let g1 = avx2_madd_i16x16_pair(
            load!(0),
            load!(16),
            avx2_coeff_pair_i16x8(&crate::itx_2d::DCT32_KGP_X4, 1),
        );
        let e = [
            avx2_i32x16_add(g0, h0),
            avx2_i32x16_add(g1, h1),
            avx2_i32x16_sub(g1, h1),
            avx2_i32x16_sub(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = avx2_i32x16_add(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = avx2_i32x16_sub(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = avx2_i32x16_add(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = avx2_i32x16_sub(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = avx2_i32x16_add(a[k], b[k]);
            out[k + 16] = avx2_i32x16_sub(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_residual_add_u8x8_expand_x2(dst: *mut u8, v: __m256i, rnd: __m256i, sh: __m128i) {
    unsafe {
        let r = _mm256_sra_epi32(_mm256_add_epi32(v, rnd), sh);
        let r16 = _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
        let rlo = _mm_unpacklo_epi16(r16, r16);
        let rhi = _mm_unpackhi_epi16(r16, r16);
        let p8 = _mm_loadu_si128(dst as *const __m128i);
        let plo = _mm_cvtepu8_epi16(p8);
        let phi = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(p8));
        let slo = _mm_adds_epi16(plo, rlo);
        let shi = _mm_adds_epi16(phi, rhi);
        let out = _mm_packus_epi16(slo, shi);
        _mm_storeu_si128(dst as *mut __m128i, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_residual_add_u8x4(dst: &mut [u8], off: usize, v: __m128i, rnd: __m128i, sh: __m128i) {
    unsafe {
        debug_assert!(off + 4 <= dst.len());
        let r = _mm_sra_epi32(_mm_add_epi32(v, rnd), sh);
        let r16 = _mm_packs_epi32(r, _mm_setzero_si128());
        let p8 = _mm_castps_si128(_mm_load_ss(dst.as_ptr().add(off).cast()));
        let p16 = _mm_unpacklo_epi8(p8, _mm_setzero_si128());
        let sum = _mm_adds_epi16(p16, r16);
        let out = _mm_packus_epi16(sum, _mm_setzero_si128());
        _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(out));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_residual_add_u8x4_expand_x2(
    dst: &mut [u8],
    off: usize,
    v: __m128i,
    rnd: __m128i,
    sh: __m128i,
) {
    unsafe {
        debug_assert!(off + 8 <= dst.len());
        let r = _mm_sra_epi32(_mm_add_epi32(v, rnd), sh);
        let r16 = _mm_packs_epi32(r, _mm_setzero_si128());
        // Duplicate each residual lane horizontally: [a,b,c,d] ->
        // [a,a,b,b,c,c,d,d].  This removes the old scalar extract/clamp
        // expansion path used by rectangular/64-expanded fused AVX writes.
        let r16x2 = _mm_unpacklo_epi16(r16, r16);
        let p = dst.as_mut_ptr().add(off);
        let d8 = _mm_loadl_epi64(p as *const __m128i);
        let d16 = _mm_unpacklo_epi8(d8, _mm_setzero_si128());
        let sum = _mm_adds_epi16(d16, r16x2);
        let out = _mm_packus_epi16(sum, _mm_setzero_si128());
        _mm_storel_epi64(p as *mut __m128i, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_writeback8_i32_u8<const W: usize, const H: usize>(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    x: usize,
    y: usize,
    v: __m256i,
    rnd: __m256i,
    sh: __m128i,
) {
    debug_assert!(x + 8 <= W);
    debug_assert!(y < H);
    unsafe {
        if out_w > W {
            let ox = x * 2;
            let oy = if out_h > H { y * 2 } else { y };
            let off0 = dst_off + oy * dst_stride + ox;
            avx2_residual_add_u8x8_expand_x2(dst.as_mut_ptr().add(off0), v, rnd, sh);
            if out_h > H {
                avx2_residual_add_u8x8_expand_x2(
                    dst.as_mut_ptr().add(off0 + dst_stride),
                    v,
                    rnd,
                    sh,
                );
            }
        } else {
            let ox = x;
            let oy = if out_h > H { y * 2 } else { y };
            let off0 = dst_off + oy * dst_stride + ox;
            avx2_residual_add_u8x8(dst.as_mut_ptr().add(off0), v, rnd, sh);
            if out_h > H {
                avx2_residual_add_u8x8(dst.as_mut_ptr().add(off0 + dst_stride), v, rnd, sh);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_i32x16_contiguous(v: Avx2I32x16) -> (__m256i, __m256i) {
    (
        _mm256_permute2x128_si256::<0x20>(v.lo, v.hi),
        _mm256_permute2x128_si256::<0x31>(v.lo, v.hi),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_store_i32x16_row(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Avx2I32x16) {
    unsafe {
        debug_assert!(off + 16 <= tmp.len());
        let (lo, hi) = avx2_i32x16_contiguous(v);
        let ptr = tmp.as_mut_ptr().add(off) as *mut __m256i;
        _mm256_storeu_si256(ptr, lo);
        _mm256_storeu_si256(ptr.add(1), hi);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_residual_add_u8x16(dst: *mut u8, v: Avx2I32x16, rnd: __m256i, sh: __m128i) {
    unsafe {
        let (lo, hi) = avx2_i32x16_contiguous(v);
        let rlo = _mm256_sra_epi32(_mm256_add_epi32(lo, rnd), sh);
        let rhi = _mm256_sra_epi32(_mm256_add_epi32(hi, rnd), sh);
        let r16 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(rlo, rhi));
        let p8 = _mm_loadu_si128(dst as *const __m128i);
        let p16 = _mm256_cvtepu8_epi16(p8);
        let sum = _mm256_adds_epi16(p16, r16);
        let out = _mm_packus_epi16(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256::<1>(sum),
        );
        _mm_storeu_si128(dst as *mut __m128i, out);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_residual_add_u8x16_expand_x2(dst: *mut u8, v: Avx2I32x16, rnd: __m256i, sh: __m128i) {
    let (lo, hi) = avx2_i32x16_contiguous(v);
    unsafe {
        avx2_residual_add_u8x8_expand_x2(dst, lo, rnd, sh);
        avx2_residual_add_u8x8_expand_x2(dst.add(16), hi, rnd, sh);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_writeback16_i32_u8<const W: usize, const H: usize>(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    x: usize,
    y: usize,
    v: Avx2I32x16,
    rnd: __m256i,
    sh: __m128i,
) {
    debug_assert!(x + 16 <= W);
    debug_assert!(y < H);
    unsafe {
        if out_w > W {
            let ox = x * 2;
            let oy = if out_h > H { y * 2 } else { y };
            let off0 = dst_off + oy * dst_stride + ox;
            avx2_residual_add_u8x16_expand_x2(dst.as_mut_ptr().add(off0), v, rnd, sh);
            if out_h > H {
                avx2_residual_add_u8x16_expand_x2(
                    dst.as_mut_ptr().add(off0 + dst_stride),
                    v,
                    rnd,
                    sh,
                );
            }
        } else {
            let ox = x;
            let oy = if out_h > H { y * 2 } else { y };
            let off0 = dst_off + oy * dst_stride + ox;
            avx2_residual_add_u8x16(dst.as_mut_ptr().add(off0), v, rnd, sh);
            if out_h > H {
                avx2_residual_add_u8x16(dst.as_mut_ptr().add(off0 + dst_stride), v, rnd, sh);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_writeback4_i32_u8<const W: usize, const H: usize>(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    x: usize,
    y: usize,
    v: __m128i,
    rnd: __m128i,
    sh: __m128i,
) {
    debug_assert!(x + 4 <= W);
    debug_assert!(y < H);
    if out_w > W {
        let ox = x * 2;
        let oy = if out_h > H { y * 2 } else { y };
        let off0 = dst_off + oy * dst_stride + ox;
        avx2_residual_add_u8x4_expand_x2(dst, off0, v, rnd, sh);
        if out_h > H {
            let off1 = off0 + dst_stride;
            avx2_residual_add_u8x4_expand_x2(dst, off1, v, rnd, sh);
        }
    } else {
        let ox = x;
        let oy = if out_h > H { y * 2 } else { y };
        let off0 = dst_off + oy * dst_stride + ox;
        avx2_residual_add_u8x4(dst, off0, v, rnd, sh);
        if out_h > H {
            let off1 = off0 + dst_stride;
            avx2_residual_add_u8x4(dst, off1, v, rnd, sh);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x16_scratch16_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
    debug_assert!(base + 16 <= STRIDE);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 16 <= scratch.len());
    let zero_lane = _mm256_setzero_si256();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load16_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zero_lane
            }
        };
    }
    let out = avx2_dct16_i16x16_all_body!();
    let mut k = 0usize;
    while k < 16 {
        avx2_store_i32x16_row(tmp, base + k * 32, out[k]);
        k += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x16_scratch16_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
    debug_assert!(base + 16 <= STRIDE);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 16 <= scratch.len());
    let zero_lane = _mm256_setzero_si256();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load16_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zero_lane
            }
        };
    }
    let out = avx2_dct32_i16x16_all_body!();
    let mut k = 0usize;
    while k < 32 {
        avx2_store_i32x16_row(tmp, base + k * 32, out[k]);
        k += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x16_scratch16_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
    debug_assert!(base + 16 <= STRIDE);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 16 <= scratch.len());
    let zero_lane = _mm256_setzero_si256();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load16_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zero_lane
            }
        };
    }
    let out = avx2_dct16_i16x16_all_body!();
    let mut k = 0usize;
    while k < 16 {
        avx2_writeback16_i32_u8::<STRIDE, 16>(
            dst, dst_off, dst_stride, out_w, out_h, base, k, out[k], rnd1, sh1,
        );
        k += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x16_scratch16_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
    debug_assert!(base + 16 <= STRIDE);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 16 <= scratch.len());
    let zero_lane = _mm256_setzero_si256();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load16_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zero_lane
            }
        };
    }
    let out = avx2_dct32_i16x16_all_body!();
    let mut k = 0usize;
    while k < 32 {
        avx2_writeback16_i32_u8::<STRIDE, 32>(
            dst, dst_off, dst_stride, out_w, out_h, base, k, out[k], rnd1, sh1,
        );
        k += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x8_scratch8_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    unsafe {
        debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
        debug_assert!(base + 8 <= STRIDE);
        debug_assert!(base + (ACTIVE - 1) * STRIDE + 8 <= scratch.len());
        let z128 = _mm_setzero_si128();
        let z = _mm256_setzero_si256();
        macro_rules! load {
            ($idx:expr) => {
                if ($idx) < ACTIVE {
                    avx2_load8_i16_scratch(scratch, base + ($idx) * STRIDE)
                } else {
                    z128
                }
            };
        }
        macro_rules! madd_pair {
            ($i0:expr, $i1:expr, $tbl:expr, $idx:expr) => {
                _mm256_madd_epi16(
                    avx2_pair8_i16(load!($i0), load!($i1)),
                    avx2_coeff_pair_i16x8($tbl, $idx),
                )
            };
        }
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let kbase = m * 8;
            let mut acc = z;
            if ACTIVE > 1 {
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(1, 3, &crate::itx_2d::DCT16_KBP_X4, kbase >> 1),
                );
            }
            if ACTIVE > 5 {
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(5, 7, &crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 1),
                );
            }
            if ACTIVE > 9 {
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(9, 11, &crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 2),
                );
            }
            if ACTIVE > 13 {
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(13, 15, &crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 3),
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
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(2, 6, &crate::itx_2d::DCT16_KDP_X4, kbase >> 1),
                );
            }
            if ACTIVE > 10 {
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(10, 14, &crate::itx_2d::DCT16_KDP_X4, (kbase >> 1) + 1),
                );
            }
            d[m] = acc;
            m += 1;
        }
        let f0 = if ACTIVE > 4 {
            madd_pair!(4, 12, &crate::itx_2d::DCT16_KFP_X4, 0)
        } else {
            z
        };
        let f1 = if ACTIVE > 4 {
            madd_pair!(4, 12, &crate::itx_2d::DCT16_KFP_X4, 1)
        } else {
            z
        };
        let g0 = madd_pair!(0, 8, &crate::itx_2d::DCT16_KGP_X4, 0);
        let g1 = madd_pair!(0, 8, &crate::itx_2d::DCT16_KGP_X4, 1);
        let cc0 = _mm256_add_epi32(g0, f0);
        let cc1 = _mm256_add_epi32(g1, f1);
        let cc2 = _mm256_sub_epi32(g1, f1);
        let cc3 = _mm256_sub_epi32(g0, f0);
        let a0 = _mm256_add_epi32(cc0, d[0]);
        let a1 = _mm256_add_epi32(cc1, d[1]);
        let a2 = _mm256_add_epi32(cc2, d[2]);
        let a3 = _mm256_add_epi32(cc3, d[3]);
        let a4 = _mm256_sub_epi32(cc3, d[3]);
        let a5 = _mm256_sub_epi32(cc2, d[2]);
        let a6 = _mm256_sub_epi32(cc1, d[1]);
        let a7 = _mm256_sub_epi32(cc0, d[0]);
        let a = [a0, a1, a2, a3, a4, a5, a6, a7];
        let mut k = 0usize;
        while k < 8 {
            _mm256_storeu_si256(
                tmp.as_mut_ptr().add(base + k * 32) as *mut __m256i,
                _mm256_add_epi32(a[k], b[k]),
            );
            _mm256_storeu_si256(
                tmp.as_mut_ptr().add(base + (k + 8) * 32) as *mut __m256i,
                _mm256_sub_epi32(a[7 - k], b[7 - k]),
            );
            k += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x8_scratch8_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
    debug_assert!(base + 8 <= STRIDE);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 8 <= scratch.len());
    let z128 = _mm_setzero_si128();
    let z = _mm256_setzero_si256();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load8_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                z128
            }
        };
    }
    macro_rules! madd_pair {
        ($i0:expr, $i1:expr, $tbl:expr, $idx:expr) => {
            _mm256_madd_epi16(
                avx2_pair8_i16(load!($i0), load!($i1)),
                avx2_coeff_pair_i16x8($tbl, $idx),
            )
        };
    }
    let mut b = [z; 8];
    let mut m = 0usize;
    while m < 8 {
        let kbase = m * 8;
        let mut acc = z;
        if ACTIVE > 1 {
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(1, 3, &crate::itx_2d::DCT16_KBP_X4, kbase >> 1),
            );
        }
        if ACTIVE > 5 {
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(5, 7, &crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 1),
            );
        }
        if ACTIVE > 9 {
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(9, 11, &crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 2),
            );
        }
        if ACTIVE > 13 {
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(13, 15, &crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 3),
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
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(2, 6, &crate::itx_2d::DCT16_KDP_X4, kbase >> 1),
            );
        }
        if ACTIVE > 10 {
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(10, 14, &crate::itx_2d::DCT16_KDP_X4, (kbase >> 1) + 1),
            );
        }
        d[m] = acc;
        m += 1;
    }
    let f0 = if ACTIVE > 4 {
        madd_pair!(4, 12, &crate::itx_2d::DCT16_KFP_X4, 0)
    } else {
        z
    };
    let f1 = if ACTIVE > 4 {
        madd_pair!(4, 12, &crate::itx_2d::DCT16_KFP_X4, 1)
    } else {
        z
    };
    let g0 = madd_pair!(0, 8, &crate::itx_2d::DCT16_KGP_X4, 0);
    let g1 = madd_pair!(0, 8, &crate::itx_2d::DCT16_KGP_X4, 1);
    let cc0 = _mm256_add_epi32(g0, f0);
    let cc1 = _mm256_add_epi32(g1, f1);
    let cc2 = _mm256_sub_epi32(g1, f1);
    let cc3 = _mm256_sub_epi32(g0, f0);
    let a0 = _mm256_add_epi32(cc0, d[0]);
    let a1 = _mm256_add_epi32(cc1, d[1]);
    let a2 = _mm256_add_epi32(cc2, d[2]);
    let a3 = _mm256_add_epi32(cc3, d[3]);
    let a4 = _mm256_sub_epi32(cc3, d[3]);
    let a5 = _mm256_sub_epi32(cc2, d[2]);
    let a6 = _mm256_sub_epi32(cc1, d[1]);
    let a7 = _mm256_sub_epi32(cc0, d[0]);
    macro_rules! write_row {
        ($row:expr, $v:expr) => {
            avx2_writeback8_i32_u8::<STRIDE, 16>(
                dst, dst_off, dst_stride, out_w, out_h, base, $row, $v, rnd1, sh1,
            );
        };
    }
    write_row!(0, _mm256_add_epi32(a0, b[0]));
    write_row!(1, _mm256_add_epi32(a1, b[1]));
    write_row!(2, _mm256_add_epi32(a2, b[2]));
    write_row!(3, _mm256_add_epi32(a3, b[3]));
    write_row!(4, _mm256_add_epi32(a4, b[4]));
    write_row!(5, _mm256_add_epi32(a5, b[5]));
    write_row!(6, _mm256_add_epi32(a6, b[6]));
    write_row!(7, _mm256_add_epi32(a7, b[7]));
    write_row!(8, _mm256_sub_epi32(a7, b[7]));
    write_row!(9, _mm256_sub_epi32(a6, b[6]));
    write_row!(10, _mm256_sub_epi32(a5, b[5]));
    write_row!(11, _mm256_sub_epi32(a4, b[4]));
    write_row!(12, _mm256_sub_epi32(a3, b[3]));
    write_row!(13, _mm256_sub_epi32(a2, b[2]));
    write_row!(14, _mm256_sub_epi32(a1, b[1]));
    write_row!(15, _mm256_sub_epi32(a0, b[0]));
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x8_scratch8_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    unsafe {
        debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
        debug_assert!(base + 8 <= STRIDE);
        debug_assert!(base + (ACTIVE - 1) * STRIDE + 8 <= scratch.len());
        let z128 = _mm_setzero_si128();
        let z = _mm256_setzero_si256();
        macro_rules! load {
            ($idx:expr) => {
                if ($idx) < ACTIVE {
                    avx2_load8_i16_scratch(scratch, base + ($idx) * STRIDE)
                } else {
                    z128
                }
            };
        }
        macro_rules! madd_pair {
            ($i0:expr, $i1:expr, $tbl:expr, $idx:expr) => {
                _mm256_madd_epi16(
                    avx2_pair8_i16(load!($i0), load!($i1)),
                    avx2_coeff_pair_i16x8($tbl, $idx),
                )
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
                    acc = _mm256_add_epi32(
                        acc,
                        madd_pair!(
                            i0,
                            i0 + 2,
                            &crate::itx_2d::DCT32_KBP_X4,
                            (kbase >> 1) + pair
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
                    acc = _mm256_add_epi32(
                        acc,
                        madd_pair!(
                            i0,
                            i0 + 4,
                            &crate::itx_2d::DCT32_KDP_X4,
                            (kbase >> 1) + pair
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
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(4, 12, &crate::itx_2d::DCT32_KFP_X4, kbase >> 1),
                );
            }
            if ACTIVE > 20 {
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(20, 28, &crate::itx_2d::DCT32_KFP_X4, (kbase >> 1) + 1),
                );
            }
            f[m] = acc;
            m += 1;
        }
        let h0 = if ACTIVE > 8 {
            madd_pair!(8, 24, &crate::itx_2d::DCT32_KHP_X4, 0)
        } else {
            z
        };
        let h1 = if ACTIVE > 8 {
            madd_pair!(8, 24, &crate::itx_2d::DCT32_KHP_X4, 1)
        } else {
            z
        };
        let g0 = madd_pair!(0, 16, &crate::itx_2d::DCT32_KGP_X4, 0);
        let g1 = madd_pair!(0, 16, &crate::itx_2d::DCT32_KGP_X4, 1);
        let e0 = _mm256_add_epi32(g0, h0);
        let e1 = _mm256_add_epi32(g1, h1);
        let e2 = _mm256_sub_epi32(g1, h1);
        let e3 = _mm256_sub_epi32(g0, h0);
        let cc0 = _mm256_add_epi32(e0, f[0]);
        let cc1 = _mm256_add_epi32(e1, f[1]);
        let cc2 = _mm256_add_epi32(e2, f[2]);
        let cc3 = _mm256_add_epi32(e3, f[3]);
        let cc4 = _mm256_sub_epi32(e3, f[3]);
        let cc5 = _mm256_sub_epi32(e2, f[2]);
        let cc6 = _mm256_sub_epi32(e1, f[1]);
        let cc7 = _mm256_sub_epi32(e0, f[0]);
        let cc = [cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7];
        let a0 = _mm256_add_epi32(cc[0], d[0]);
        let a1 = _mm256_add_epi32(cc[1], d[1]);
        let a2 = _mm256_add_epi32(cc[2], d[2]);
        let a3 = _mm256_add_epi32(cc[3], d[3]);
        let a4 = _mm256_add_epi32(cc[4], d[4]);
        let a5 = _mm256_add_epi32(cc[5], d[5]);
        let a6 = _mm256_add_epi32(cc[6], d[6]);
        let a7 = _mm256_add_epi32(cc[7], d[7]);
        let a8 = _mm256_sub_epi32(cc[7], d[7]);
        let a9 = _mm256_sub_epi32(cc[6], d[6]);
        let a10 = _mm256_sub_epi32(cc[5], d[5]);
        let a11 = _mm256_sub_epi32(cc[4], d[4]);
        let a12 = _mm256_sub_epi32(cc[3], d[3]);
        let a13 = _mm256_sub_epi32(cc[2], d[2]);
        let a14 = _mm256_sub_epi32(cc[1], d[1]);
        let a15 = _mm256_sub_epi32(cc[0], d[0]);
        let a = [
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        ];
        let mut k = 0usize;
        while k < 16 {
            _mm256_storeu_si256(
                tmp.as_mut_ptr().add(base + k * 32) as *mut __m256i,
                _mm256_add_epi32(a[k], b[k]),
            );
            _mm256_storeu_si256(
                tmp.as_mut_ptr().add(base + (k + 16) * 32) as *mut __m256i,
                _mm256_sub_epi32(a[15 - k], b[15 - k]),
            );
            k += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x8_scratch8_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
    debug_assert!(base + 8 <= STRIDE);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 8 <= scratch.len());
    let z128 = _mm_setzero_si128();
    let z = _mm256_setzero_si256();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load8_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                z128
            }
        };
    }
    macro_rules! madd_pair {
        ($i0:expr, $i1:expr, $tbl:expr, $idx:expr) => {
            _mm256_madd_epi16(
                avx2_pair8_i16(load!($i0), load!($i1)),
                avx2_coeff_pair_i16x8($tbl, $idx),
            )
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
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(
                        i0,
                        i0 + 2,
                        &crate::itx_2d::DCT32_KBP_X4,
                        (kbase >> 1) + pair
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
                acc = _mm256_add_epi32(
                    acc,
                    madd_pair!(
                        i0,
                        i0 + 4,
                        &crate::itx_2d::DCT32_KDP_X4,
                        (kbase >> 1) + pair
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
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(4, 12, &crate::itx_2d::DCT32_KFP_X4, kbase >> 1),
            );
        }
        if ACTIVE > 20 {
            acc = _mm256_add_epi32(
                acc,
                madd_pair!(20, 28, &crate::itx_2d::DCT32_KFP_X4, (kbase >> 1) + 1),
            );
        }
        f[m] = acc;
        m += 1;
    }
    let h0 = if ACTIVE > 8 {
        madd_pair!(8, 24, &crate::itx_2d::DCT32_KHP_X4, 0)
    } else {
        z
    };
    let h1 = if ACTIVE > 8 {
        madd_pair!(8, 24, &crate::itx_2d::DCT32_KHP_X4, 1)
    } else {
        z
    };
    let g0 = madd_pair!(0, 16, &crate::itx_2d::DCT32_KGP_X4, 0);
    let g1 = madd_pair!(0, 16, &crate::itx_2d::DCT32_KGP_X4, 1);
    let e0 = _mm256_add_epi32(g0, h0);
    let e1 = _mm256_add_epi32(g1, h1);
    let e2 = _mm256_sub_epi32(g1, h1);
    let e3 = _mm256_sub_epi32(g0, h0);
    let cc0 = _mm256_add_epi32(e0, f[0]);
    let cc1 = _mm256_add_epi32(e1, f[1]);
    let cc2 = _mm256_add_epi32(e2, f[2]);
    let cc3 = _mm256_add_epi32(e3, f[3]);
    let cc4 = _mm256_sub_epi32(e3, f[3]);
    let cc5 = _mm256_sub_epi32(e2, f[2]);
    let cc6 = _mm256_sub_epi32(e1, f[1]);
    let cc7 = _mm256_sub_epi32(e0, f[0]);
    let a0 = _mm256_add_epi32(cc0, d[0]);
    let a1 = _mm256_add_epi32(cc1, d[1]);
    let a2 = _mm256_add_epi32(cc2, d[2]);
    let a3 = _mm256_add_epi32(cc3, d[3]);
    let a4 = _mm256_add_epi32(cc4, d[4]);
    let a5 = _mm256_add_epi32(cc5, d[5]);
    let a6 = _mm256_add_epi32(cc6, d[6]);
    let a7 = _mm256_add_epi32(cc7, d[7]);
    let a8 = _mm256_sub_epi32(cc7, d[7]);
    let a9 = _mm256_sub_epi32(cc6, d[6]);
    let a10 = _mm256_sub_epi32(cc5, d[5]);
    let a11 = _mm256_sub_epi32(cc4, d[4]);
    let a12 = _mm256_sub_epi32(cc3, d[3]);
    let a13 = _mm256_sub_epi32(cc2, d[2]);
    let a14 = _mm256_sub_epi32(cc1, d[1]);
    let a15 = _mm256_sub_epi32(cc0, d[0]);
    macro_rules! write_row {
        ($row:expr, $v:expr) => {
            avx2_writeback8_i32_u8::<STRIDE, 32>(
                dst, dst_off, dst_stride, out_w, out_h, base, $row, $v, rnd1, sh1,
            );
        };
    }
    write_row!(0, _mm256_add_epi32(a0, b[0]));
    write_row!(1, _mm256_add_epi32(a1, b[1]));
    write_row!(2, _mm256_add_epi32(a2, b[2]));
    write_row!(3, _mm256_add_epi32(a3, b[3]));
    write_row!(4, _mm256_add_epi32(a4, b[4]));
    write_row!(5, _mm256_add_epi32(a5, b[5]));
    write_row!(6, _mm256_add_epi32(a6, b[6]));
    write_row!(7, _mm256_add_epi32(a7, b[7]));
    write_row!(8, _mm256_add_epi32(a8, b[8]));
    write_row!(9, _mm256_add_epi32(a9, b[9]));
    write_row!(10, _mm256_add_epi32(a10, b[10]));
    write_row!(11, _mm256_add_epi32(a11, b[11]));
    write_row!(12, _mm256_add_epi32(a12, b[12]));
    write_row!(13, _mm256_add_epi32(a13, b[13]));
    write_row!(14, _mm256_add_epi32(a14, b[14]));
    write_row!(15, _mm256_add_epi32(a15, b[15]));
    write_row!(16, _mm256_sub_epi32(a15, b[15]));
    write_row!(17, _mm256_sub_epi32(a14, b[14]));
    write_row!(18, _mm256_sub_epi32(a13, b[13]));
    write_row!(19, _mm256_sub_epi32(a12, b[12]));
    write_row!(20, _mm256_sub_epi32(a11, b[11]));
    write_row!(21, _mm256_sub_epi32(a10, b[10]));
    write_row!(22, _mm256_sub_epi32(a9, b[9]));
    write_row!(23, _mm256_sub_epi32(a8, b[8]));
    write_row!(24, _mm256_sub_epi32(a7, b[7]));
    write_row!(25, _mm256_sub_epi32(a6, b[6]));
    write_row!(26, _mm256_sub_epi32(a5, b[5]));
    write_row!(27, _mm256_sub_epi32(a4, b[4]));
    write_row!(28, _mm256_sub_epi32(a3, b[3]));
    write_row!(29, _mm256_sub_epi32(a2, b[2]));
    write_row!(30, _mm256_sub_epi32(a1, b[1]));
    write_row!(31, _mm256_sub_epi32(a0, b[0]));
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x16_scratch16_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        avx2_dct16_i16x16_scratch16_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        avx2_dct16_i16x16_scratch16_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else {
        avx2_dct16_i16x16_scratch16_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x16_scratch16_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        avx2_dct32_i16x16_scratch16_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        avx2_dct32_i16x16_scratch16_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else if active <= 16 {
        avx2_dct32_i16x16_scratch16_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    } else {
        avx2_dct32_i16x16_scratch16_stride_active_store::<STRIDE, 32>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x16_scratch16_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    if active <= 4 {
        avx2_dct16_i16x16_scratch16_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 8 {
        avx2_dct16_i16x16_scratch16_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else {
        avx2_dct16_i16x16_scratch16_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x16_scratch16_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    if active <= 4 {
        avx2_dct32_i16x16_scratch16_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 8 {
        avx2_dct32_i16x16_scratch16_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 16 {
        avx2_dct32_i16x16_scratch16_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else {
        avx2_dct32_i16x16_scratch16_stride_active_add_u8::<STRIDE, 32>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x8_scratch8_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        avx2_dct16_i16x8_scratch8_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        avx2_dct16_i16x8_scratch8_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else {
        avx2_dct16_i16x8_scratch8_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x8_scratch8_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        avx2_dct32_i16x8_scratch8_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        avx2_dct32_i16x8_scratch8_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else if active <= 16 {
        avx2_dct32_i16x8_scratch8_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    } else {
        avx2_dct32_i16x8_scratch8_stride_active_store::<STRIDE, 32>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x8_scratch8_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    if active <= 4 {
        avx2_dct16_i16x8_scratch8_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 8 {
        avx2_dct16_i16x8_scratch8_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else {
        avx2_dct16_i16x8_scratch8_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x8_scratch8_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m256i,
    sh1: __m128i,
) {
    if active <= 4 {
        avx2_dct32_i16x8_scratch8_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 8 {
        avx2_dct32_i16x8_scratch8_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 16 {
        avx2_dct32_i16x8_scratch8_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else {
        avx2_dct32_i16x8_scratch8_stride_active_add_u8::<STRIDE, 32>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x4_scratch4_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
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
                    avx2_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 5 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(5), load!(7)),
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            if ACTIVE > 9 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(9), load!(11)),
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 2),
                    ),
                );
            }
            if ACTIVE > 13 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(13), load!(15)),
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 3),
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 10 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(10), load!(14)),
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            d[m] = acc;
            m += 1;
        }
        let x412 = _mm_unpacklo_epi16(load!(4), load!(12));
        let f0 = if ACTIVE > 4 {
            _mm_madd_epi16(x412, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 0))
        } else {
            z
        };
        let f1 = if ACTIVE > 4 {
            _mm_madd_epi16(x412, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 1))
        } else {
            z
        };
        let x08 = _mm_unpacklo_epi16(load!(0), load!(8));
        let g0 = _mm_madd_epi16(x08, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 0));
        let g1 = _mm_madd_epi16(x08, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 1));
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
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x4_scratch4_stride_active_store<const STRIDE: usize, const ACTIVE: usize>(
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
                    avx2_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
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
                            avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KBP_X4, (kbase >> 1) + pair),
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
                            avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KDP_X4, (kbase >> 1) + pair),
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, kbase >> 1),
                    ),
                );
            }
            if ACTIVE > 20 {
                acc = _mm_add_epi32(
                    acc,
                    _mm_madd_epi16(
                        _mm_unpacklo_epi16(load!(20), load!(28)),
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, (kbase >> 1) + 1),
                    ),
                );
            }
            f[m] = acc;
            m += 1;
        }
        let x824 = _mm_unpacklo_epi16(load!(8), load!(24));
        let h0 = if ACTIVE > 8 {
            _mm_madd_epi16(x824, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 0))
        } else {
            z
        };
        let h1 = if ACTIVE > 8 {
            _mm_madd_epi16(x824, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 1))
        } else {
            z
        };
        let x016 = _mm_unpacklo_epi16(load!(0), load!(16));
        let g0 = _mm_madd_epi16(x016, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 0));
        let g1 = _mm_madd_epi16(x016, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 1));
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
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x4_scratch4_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        avx2_dct16_i16x4_scratch4_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        avx2_dct16_i16x4_scratch4_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else {
        avx2_dct16_i16x4_scratch4_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x4_scratch4_stride_eob_store<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    if active <= 4 {
        avx2_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 4>(scratch, base, tmp)
    } else if active <= 8 {
        avx2_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 8>(scratch, base, tmp)
    } else if active <= 16 {
        avx2_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 16>(scratch, base, tmp)
    } else {
        avx2_dct32_i16x4_scratch4_stride_active_store::<STRIDE, 32>(scratch, base, tmp)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x4_scratch4_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
    let z = _mm_setzero_si128();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
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
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, kbase >> 1),
                ),
            );
        }
        if ACTIVE > 5 {
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(5), load!(7)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 1),
                ),
            );
        }
        if ACTIVE > 9 {
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(9), load!(11)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 2),
                ),
            );
        }
        if ACTIVE > 13 {
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(13), load!(15)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KBP_X4, (kbase >> 1) + 3),
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
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, kbase >> 1),
                ),
            );
        }
        if ACTIVE > 10 {
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(10), load!(14)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KDP_X4, (kbase >> 1) + 1),
                ),
            );
        }
        d[m] = acc;
        m += 1;
    }
    let x412 = _mm_unpacklo_epi16(load!(4), load!(12));
    let f0 = if ACTIVE > 4 {
        _mm_madd_epi16(x412, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 0))
    } else {
        z
    };
    let f1 = if ACTIVE > 4 {
        _mm_madd_epi16(x412, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KFP_X4, 1))
    } else {
        z
    };
    let x08 = _mm_unpacklo_epi16(load!(0), load!(8));
    let g0 = _mm_madd_epi16(x08, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 0));
    let g1 = _mm_madd_epi16(x08, avx2_coeff_pair_i16(&crate::itx_2d::DCT16_KGP_X4, 1));
    let cc0 = _mm_add_epi32(g0, f0);
    let cc1 = _mm_add_epi32(g1, f1);
    let cc2 = _mm_sub_epi32(g1, f1);
    let cc3 = _mm_sub_epi32(g0, f0);
    let a0 = _mm_add_epi32(cc0, d[0]);
    let a1 = _mm_add_epi32(cc1, d[1]);
    let a2 = _mm_add_epi32(cc2, d[2]);
    let a3 = _mm_add_epi32(cc3, d[3]);
    let a4 = _mm_sub_epi32(cc3, d[3]);
    let a5 = _mm_sub_epi32(cc2, d[2]);
    let a6 = _mm_sub_epi32(cc1, d[1]);
    let a7 = _mm_sub_epi32(cc0, d[0]);
    macro_rules! write_row {
        ($row:expr, $v:expr) => {
            avx2_writeback4_i32_u8::<STRIDE, 16>(
                dst, dst_off, dst_stride, out_w, out_h, base, $row, $v, rnd1, sh1,
            );
        };
    }
    write_row!(0, _mm_add_epi32(a0, b[0]));
    write_row!(1, _mm_add_epi32(a1, b[1]));
    write_row!(2, _mm_add_epi32(a2, b[2]));
    write_row!(3, _mm_add_epi32(a3, b[3]));
    write_row!(4, _mm_add_epi32(a4, b[4]));
    write_row!(5, _mm_add_epi32(a5, b[5]));
    write_row!(6, _mm_add_epi32(a6, b[6]));
    write_row!(7, _mm_add_epi32(a7, b[7]));
    write_row!(8, _mm_sub_epi32(a7, b[7]));
    write_row!(9, _mm_sub_epi32(a6, b[6]));
    write_row!(10, _mm_sub_epi32(a5, b[5]));
    write_row!(11, _mm_sub_epi32(a4, b[4]));
    write_row!(12, _mm_sub_epi32(a3, b[3]));
    write_row!(13, _mm_sub_epi32(a2, b[2]));
    write_row!(14, _mm_sub_epi32(a1, b[1]));
    write_row!(15, _mm_sub_epi32(a0, b[0]));
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x4_scratch4_stride_active_add_u8<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
    let z = _mm_setzero_si128();
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                avx2_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KBP_X4, (kbase >> 1) + pair),
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
                        avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KDP_X4, (kbase >> 1) + pair),
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
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, kbase >> 1),
                ),
            );
        }
        if ACTIVE > 20 {
            acc = _mm_add_epi32(
                acc,
                _mm_madd_epi16(
                    _mm_unpacklo_epi16(load!(20), load!(28)),
                    avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KFP_X4, (kbase >> 1) + 1),
                ),
            );
        }
        f[m] = acc;
        m += 1;
    }
    let x824 = _mm_unpacklo_epi16(load!(8), load!(24));
    let h0 = if ACTIVE > 8 {
        _mm_madd_epi16(x824, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 0))
    } else {
        z
    };
    let h1 = if ACTIVE > 8 {
        _mm_madd_epi16(x824, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KHP_X4, 1))
    } else {
        z
    };
    let x016 = _mm_unpacklo_epi16(load!(0), load!(16));
    let g0 = _mm_madd_epi16(x016, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 0));
    let g1 = _mm_madd_epi16(x016, avx2_coeff_pair_i16(&crate::itx_2d::DCT32_KGP_X4, 1));
    let e0 = _mm_add_epi32(g0, h0);
    let e1 = _mm_add_epi32(g1, h1);
    let e2 = _mm_sub_epi32(g1, h1);
    let e3 = _mm_sub_epi32(g0, h0);
    let cc0 = _mm_add_epi32(e0, f[0]);
    let cc1 = _mm_add_epi32(e1, f[1]);
    let cc2 = _mm_add_epi32(e2, f[2]);
    let cc3 = _mm_add_epi32(e3, f[3]);
    let cc4 = _mm_sub_epi32(e3, f[3]);
    let cc5 = _mm_sub_epi32(e2, f[2]);
    let cc6 = _mm_sub_epi32(e1, f[1]);
    let cc7 = _mm_sub_epi32(e0, f[0]);
    let a0 = _mm_add_epi32(cc0, d[0]);
    let a1 = _mm_add_epi32(cc1, d[1]);
    let a2 = _mm_add_epi32(cc2, d[2]);
    let a3 = _mm_add_epi32(cc3, d[3]);
    let a4 = _mm_add_epi32(cc4, d[4]);
    let a5 = _mm_add_epi32(cc5, d[5]);
    let a6 = _mm_add_epi32(cc6, d[6]);
    let a7 = _mm_add_epi32(cc7, d[7]);
    let a8 = _mm_sub_epi32(cc7, d[7]);
    let a9 = _mm_sub_epi32(cc6, d[6]);
    let a10 = _mm_sub_epi32(cc5, d[5]);
    let a11 = _mm_sub_epi32(cc4, d[4]);
    let a12 = _mm_sub_epi32(cc3, d[3]);
    let a13 = _mm_sub_epi32(cc2, d[2]);
    let a14 = _mm_sub_epi32(cc1, d[1]);
    let a15 = _mm_sub_epi32(cc0, d[0]);
    macro_rules! write_row {
        ($row:expr, $v:expr) => {
            avx2_writeback4_i32_u8::<STRIDE, 32>(
                dst, dst_off, dst_stride, out_w, out_h, base, $row, $v, rnd1, sh1,
            );
        };
    }
    write_row!(0, _mm_add_epi32(a0, b[0]));
    write_row!(1, _mm_add_epi32(a1, b[1]));
    write_row!(2, _mm_add_epi32(a2, b[2]));
    write_row!(3, _mm_add_epi32(a3, b[3]));
    write_row!(4, _mm_add_epi32(a4, b[4]));
    write_row!(5, _mm_add_epi32(a5, b[5]));
    write_row!(6, _mm_add_epi32(a6, b[6]));
    write_row!(7, _mm_add_epi32(a7, b[7]));
    write_row!(8, _mm_add_epi32(a8, b[8]));
    write_row!(9, _mm_add_epi32(a9, b[9]));
    write_row!(10, _mm_add_epi32(a10, b[10]));
    write_row!(11, _mm_add_epi32(a11, b[11]));
    write_row!(12, _mm_add_epi32(a12, b[12]));
    write_row!(13, _mm_add_epi32(a13, b[13]));
    write_row!(14, _mm_add_epi32(a14, b[14]));
    write_row!(15, _mm_add_epi32(a15, b[15]));
    write_row!(16, _mm_sub_epi32(a15, b[15]));
    write_row!(17, _mm_sub_epi32(a14, b[14]));
    write_row!(18, _mm_sub_epi32(a13, b[13]));
    write_row!(19, _mm_sub_epi32(a12, b[12]));
    write_row!(20, _mm_sub_epi32(a11, b[11]));
    write_row!(21, _mm_sub_epi32(a10, b[10]));
    write_row!(22, _mm_sub_epi32(a9, b[9]));
    write_row!(23, _mm_sub_epi32(a8, b[8]));
    write_row!(24, _mm_sub_epi32(a7, b[7]));
    write_row!(25, _mm_sub_epi32(a6, b[6]));
    write_row!(26, _mm_sub_epi32(a5, b[5]));
    write_row!(27, _mm_sub_epi32(a4, b[4]));
    write_row!(28, _mm_sub_epi32(a3, b[3]));
    write_row!(29, _mm_sub_epi32(a2, b[2]));
    write_row!(30, _mm_sub_epi32(a1, b[1]));
    write_row!(31, _mm_sub_epi32(a0, b[0]));
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x4_scratch4_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    if active <= 4 {
        avx2_dct16_i16x4_scratch4_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 8 {
        avx2_dct16_i16x4_scratch4_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else {
        avx2_dct16_i16x4_scratch4_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x4_scratch4_stride_eob_add_u8<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: __m128i,
    sh1: __m128i,
) {
    if active <= 4 {
        avx2_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 4>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 8 {
        avx2_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 8>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else if active <= 16 {
        avx2_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 16>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    } else {
        avx2_dct32_i16x4_scratch4_stride_active_add_u8::<STRIDE, 32>(
            scratch, base, dst, dst_off, dst_stride, out_w, out_h, rnd1, sh1,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [__m128i; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    avx2_dct16_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [__m128i; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    avx2_dct32_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x8_all_from_coeff8_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [__m256i; 16] {
    debug_assert!(base + 15 * STRIDE + 8 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            avx2_load8_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    avx2_dct16_i16x8_all_body!()
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x8_all_from_coeff8_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [__m256i; 32] {
    debug_assert!(base + 31 * STRIDE + 8 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            avx2_load8_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    avx2_dct32_i16x8_all_body!()
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x16_all_from_coeff16_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [Avx2I32x16; 16] {
    debug_assert!(base + 15 * STRIDE + 16 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            avx2_load16_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    avx2_dct16_i16x16_all_body!()
}

#[inline]
#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x16_all_from_coeff16_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [Avx2I32x16; 32] {
    debug_assert!(base + 31 * STRIDE + 16 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            avx2_load16_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    avx2_dct32_i16x16_all_body!()
}

#[target_feature(enable = "avx2")]
fn avx2_dct16_i16x4_coeff_rows_to_scratch<const IS_RECT2: bool, const COEFF_STRIDE: usize>(
    coeff: &[i16],
    scratch: &mut [i16],
    mut y: usize,
    nrows: usize,
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) -> usize {
    while y + 16 <= nrows {
        let out =
            avx2_dct16_i16x16_all_from_coeff16_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 16;
        avx2_store16x16_i16_clip256::<16, 16, 0>(scratch, row_base, &out, rnd, sh, minv, maxv);
        y += 16;
    }
    while y + 8 <= nrows {
        let out = avx2_dct16_i16x8_all_from_coeff8_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 16;
        avx2_store16x8_i16_clip256::<16>(
            scratch, row_base, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
            out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15], rnd, sh, minv,
            maxv,
        );
        y += 8;
    }
    if y + 4 <= nrows {
        let out = avx2_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 16;
        let mut m = 0usize;
        while m < 16 {
            avx2_store4x4_i16_clip::<16>(
                scratch,
                row_base + m,
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
        y += 4;
    }
    y
}

#[target_feature(enable = "avx2")]
fn avx2_dct32_i16x4_coeff_rows_to_scratch<const IS_RECT2: bool, const COEFF_STRIDE: usize>(
    coeff: &[i16],
    scratch: &mut [i16],
    mut y: usize,
    nrows: usize,
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) -> usize {
    while y + 16 <= nrows {
        let out =
            avx2_dct32_i16x16_all_from_coeff16_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 32;
        avx2_store16x16_i16_clip256::<32, 32, 0>(scratch, row_base, &out, rnd, sh, minv, maxv);
        avx2_store16x16_i16_clip256::<32, 32, 16>(
            scratch,
            row_base + 16,
            &out,
            rnd,
            sh,
            minv,
            maxv,
        );
        y += 16;
    }
    while y + 8 <= nrows {
        let out = avx2_dct32_i16x8_all_from_coeff8_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 32;
        let mut m = 0usize;
        while m < 32 {
            avx2_store16x8_i16_clip256::<32>(
                scratch,
                row_base + m,
                out[m],
                out[m + 1],
                out[m + 2],
                out[m + 3],
                out[m + 4],
                out[m + 5],
                out[m + 6],
                out[m + 7],
                out[m + 8],
                out[m + 9],
                out[m + 10],
                out[m + 11],
                out[m + 12],
                out[m + 13],
                out[m + 14],
                out[m + 15],
                rnd,
                sh,
                minv,
                maxv,
            );
            m += 16;
        }
        y += 8;
    }
    if y + 4 <= nrows {
        let out = avx2_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 32;
        let mut m = 0usize;
        while m < 32 {
            avx2_store4x4_i16_clip::<32>(
                scratch,
                row_base + m,
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
        y += 4;
    }
    y
}

#[inline]
#[target_feature(enable = "avx2")]
fn idct_dequant_dct_i16_avx2_impl<const N: usize>(
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
        idct_dequant_dct_i16_avx2_impl_const::<N, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_dct_i16_avx2_impl_const::<N, false>(
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
#[target_feature(enable = "avx2")]
fn idct_dequant_dct_i16_avx2_impl_const<const N: usize, const IS_RECT2: bool>(
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

    with_avx2_itx_i16_scratch(ITX_TMP_PIXELS, |scratch| {
        scratch.fill(0);
        let mut y = 0usize;
        if N == 16 {
            y = avx2_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, 16>(
                coeff, scratch, y, ncols, rnd, sh, minv, maxv,
            );
        } else {
            y = avx2_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, 32>(
                coeff, scratch, y, ncols, rnd, sh, minv, maxv,
            );
        }
        debug_assert_eq!(y, ncols);
        coeff[..N * N].fill(0);

        let mut x = 0usize;
        while x + 16 <= N {
            if N == 16 {
                avx2_dct16_i16x16_scratch16_stride_eob_store::<16>(scratch, x, ncols, tmp);
            } else {
                avx2_dct32_i16x16_scratch16_stride_eob_store::<32>(scratch, x, ncols, tmp);
            }
            x += 16;
        }
        while x + 8 <= N {
            if N == 16 {
                avx2_dct16_i16x8_scratch8_stride_eob_store::<16>(scratch, x, ncols, tmp);
            } else {
                avx2_dct32_i16x8_scratch8_stride_eob_store::<32>(scratch, x, ncols, tmp);
            }
            x += 8;
        }
        while x < N {
            if N == 16 {
                avx2_dct16_i16x4_scratch4_stride_eob_store::<16>(scratch, x, ncols, tmp);
            } else {
                avx2_dct32_i16x4_scratch4_stride_eob_store::<32>(scratch, x, ncols, tmp);
            }
            x += 4;
        }
    });
}

#[inline]
#[target_feature(enable = "avx2")]
fn idct_dequant_dct_i16_avx2_fused_8bpc_impl_const<const N: usize, const IS_RECT2: bool>(
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

    with_avx2_itx_i16_scratch(ITX_TMP_PIXELS, |scratch| {
        scratch.fill(0);
        let mut y = 0usize;
        if N == 16 {
            y = avx2_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, 16>(
                coeff, scratch, y, ncols, rnd, sh, minv, maxv,
            );
        } else {
            y = avx2_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, 32>(
                coeff, scratch, y, ncols, rnd, sh, minv, maxv,
            );
        }
        debug_assert_eq!(y, ncols);
        coeff[..N * N].fill(0);

        let rnd1 = _mm256_set1_epi32((1 << shift1) >> 1);
        let sh1 = _mm_cvtsi32_si128(shift1);
        let mut x = 0usize;
        while x + 16 <= N {
            if N == 16 {
                avx2_dct16_i16x16_scratch16_stride_eob_add_u8::<16>(
                    scratch, x, ncols, dst, dst_off, dst_stride, 16, 16, rnd1, sh1,
                );
            } else {
                avx2_dct32_i16x16_scratch16_stride_eob_add_u8::<32>(
                    scratch, x, ncols, dst, dst_off, dst_stride, 32, 32, rnd1, sh1,
                );
            }
            x += 16;
        }
        while x + 8 <= N {
            if N == 16 {
                avx2_dct16_i16x8_scratch8_stride_eob_add_u8::<16>(
                    scratch, x, ncols, dst, dst_off, dst_stride, 16, 16, rnd1, sh1,
                );
            } else {
                avx2_dct32_i16x8_scratch8_stride_eob_add_u8::<32>(
                    scratch, x, ncols, dst, dst_off, dst_stride, 32, 32, rnd1, sh1,
                );
            }
            x += 8;
        }
        debug_assert_eq!(x, N);
    });
}

#[inline]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i32_impl<const N: usize, const W: usize, const H: usize>(
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
        tx_dequant_dense_avx2_i32_impl_const::<N, W, H, true>(
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
        tx_dequant_dense_avx2_i32_impl_const::<N, W, H, false>(
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
#[target_feature(enable = "avx2")]
fn tx_dequant_4x4_avx2_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    _eob: i32,
    _tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_4x4_avx2_i32_impl_const::<true>(
            coeff,
            tmp,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_4x4_avx2_i32_impl_const::<false>(
            coeff,
            tmp,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn tx_dequant_4x4_avx2_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 16);
        let z = _mm_setzero_si128();
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);

        macro_rules! load_col {
            ($j:expr) => {{
                let mut v = _mm_loadu_si128(coeff.as_ptr().add(($j) * 4) as *const __m128i);
                if IS_RECT2 {
                    v = _mm_srai_epi32::<8>(_mm_add_epi32(_mm_mullo_epi32(v, rect_mul), rect_rnd));
                }
                v
            }};
        }

        let c0 = load_col!(0);
        let c1 = load_col!(1);
        let c2 = load_col!(2);
        let c3 = load_col!(3);

        macro_rules! row_pass {
            ($m:expr) => {{
                let mut a = z;
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        c0,
                        _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, 4, $m, 0)),
                    ),
                );
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        c1,
                        _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, 4, $m, 1)),
                    ),
                );
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        c2,
                        _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, 4, $m, 2)),
                    ),
                );
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        c3,
                        _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, 4, $m, 3)),
                    ),
                );
                a
            }};
        }

        let row = [row_pass!(0), row_pass!(1), row_pass!(2), row_pass!(3)];
        avx2_store4x4_i32_clip(tmp, 0, &row, rnd, sh, minv, maxv);
        coeff[..16].fill(0);

        // Snapshot the row pass before writing back in-place.
        let r0 = _mm_loadu_si128(tmp.as_ptr() as *const __m128i);
        let r1 = _mm_loadu_si128(tmp.as_ptr().add(32) as *const __m128i);
        let r2 = _mm_loadu_si128(tmp.as_ptr().add(64) as *const __m128i);
        let r3 = _mm_loadu_si128(tmp.as_ptr().add(96) as *const __m128i);

        macro_rules! col_pass {
            ($m:expr) => {{
                let mut a = z;
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        r0,
                        _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, 4, $m, 0)),
                    ),
                );
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        r1,
                        _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, 4, $m, 1)),
                    ),
                );
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        r2,
                        _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, 4, $m, 2)),
                    ),
                );
                a = _mm_add_epi32(
                    a,
                    _mm_mullo_epi32(
                        r3,
                        _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, 4, $m, 3)),
                    ),
                );
                a
            }};
        }

        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, col_pass!(0));
        _mm_storeu_si128(tmp.as_mut_ptr().add(32) as *mut __m128i, col_pass!(1));
        _mm_storeu_si128(tmp.as_mut_ptr().add(64) as *mut __m128i, col_pass!(2));
        _mm_storeu_si128(tmp.as_mut_ptr().add(96) as *mut __m128i, col_pass!(3));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i32_impl_const<
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
        let z256 = _mm256_setzero_si256();
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);
        let rnd256 = _mm256_set1_epi32((1 << shift0) >> 1);
        let minv256 = _mm256_set1_epi32(row_clip_min);
        let maxv256 = _mm256_set1_epi32(row_clip_max);

        let mut y = 0usize;
        if W >= 8 {
            while y + 8 <= nrows {
                let mut m = 0usize;
                while m + 8 <= W {
                    let g = avx2_tx_dense_i32x8_from_coeff8_const::<IS_RECT2, W, H>(
                        coeff, y, first_kind, m,
                    );
                    avx2_store8x8_i32_clip(tmp, y * 32 + m, &g, rnd256, sh, minv256, maxv256);
                    m += 8;
                }
                y += 8;
            }
        }
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
                            _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, W, m, j)),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, W, m + 1, j)),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, W, m + 2, j)),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(avx2_tx_dense_coeff(first_kind, W, m + 3, j)),
                        ),
                    );
                    j += 1;
                }
                let g = [a0, a1, a2, a3];
                avx2_store4x4_i32_clip(tmp, y * 32 + m, &g, rnd, sh, minv, maxv);
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
        if H >= 8 {
            while x + 8 <= W {
                let mut vin = [z256; H];
                let mut j = 0usize;
                while j < H {
                    vin[j] = _mm256_loadu_si256(tmp.as_ptr().add(x + j * 32) as *const __m256i);
                    j += 1;
                }
                let mut m = 0usize;
                while m + 8 <= H {
                    let g = avx2_tx_dense_i32x8_from_tmp8::<H>(&vin, second_kind, m);
                    avx2_store_i32x8_rows(tmp, x + m * 32, &g);
                    m += 8;
                }
                x += 8;
            }
        }
        while x < W {
            // Snapshot the H input rows for this column group before computing
            // outputs: the loop below stores results back into tmp, which would
            // otherwise corrupt rows that later output groups still need to read.
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
                            _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, H, m, j)),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, H, m + 1, j)),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, H, m + 2, j)),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(avx2_tx_dense_coeff(second_kind, H, m + 3, j)),
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
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_impl<const N: usize, const W: usize, const H: usize>(
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
        tx_dequant_dense_avx2_i16_impl_const::<N, W, H, true>(
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
        tx_dequant_dense_avx2_i16_impl_const::<N, W, H, false>(
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

#[inline(never)]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_impl_const<
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

    with_avx2_itx_i16_scratch(N, |scratch| unsafe {
        scratch.fill(0);
        let mut y = 0usize;
        if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = avx2_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, sh, minv, maxv,
            );
        } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = avx2_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, sh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            let mut m = 0usize;
            while m < W {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < W {
                    let x0 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                    let x1 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                    let x01 = _mm_unpacklo_epi16(x0, x1);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m, j)),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m + 1, j)),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m + 2, j)),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m + 3, j)),
                    );
                    j += 2;
                }
                avx2_store4x4_i16_clip::<W>(
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
        coeff[..W * H].fill(0);

        let mut x = 0usize;
        while x + 16 <= W && second_kind == crate::itx_2d::TX_KIND_DCT && (H == 16 || H == 32) {
            if H == 16 {
                avx2_dct16_i16x16_scratch16_stride_eob_store::<W>(scratch, x, nrows, tmp);
            } else {
                avx2_dct32_i16x16_scratch16_stride_eob_store::<W>(scratch, x, nrows, tmp);
            }
            x += 16;
        }
        while x + 8 <= W && second_kind == crate::itx_2d::TX_KIND_DCT && (H == 16 || H == 32) {
            if H == 16 {
                avx2_dct16_i16x8_scratch8_stride_eob_store::<W>(scratch, x, nrows, tmp);
            } else {
                avx2_dct32_i16x8_scratch8_stride_eob_store::<W>(scratch, x, nrows, tmp);
            }
            x += 8;
        }
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                avx2_dct16_i16x4_scratch4_stride_eob_store::<W>(scratch, x, nrows, tmp);
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                avx2_dct32_i16x4_scratch4_stride_eob_store::<W>(scratch, x, nrows, tmp);
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = avx2_load4_i16_scratch(scratch, x + j * W);
                        let x1 = avx2_load4_i16_scratch(scratch, x + (j + 1) * W);
                        let x01 = _mm_unpacklo_epi16(x0, x1);
                        a0 = _mm_add_epi32(
                            a0,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m, j)),
                        );
                        a1 = _mm_add_epi32(
                            a1,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m + 1, j)),
                        );
                        a2 = _mm_add_epi32(
                            a2,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m + 2, j)),
                        );
                        a3 = _mm_add_epi32(
                            a3,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m + 3, j)),
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

#[inline(never)]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_fused_8bpc_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
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

    with_avx2_itx_i16_scratch(N, |scratch| {
        scratch.fill(0);
        let mut y = 0usize;

        if first_kind == crate::itx_2d::TX_KIND_IDENTITY {
            y = fused_identity_pass::<W, H, IS_RECT2>(
                coeff, nrows, rnd, sh, minv, maxv, scratch, y,
            );
        }

        if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = avx2_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, sh, minv, maxv,
            );
        } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = avx2_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, sh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            let mut m = 0usize;
            while m < W {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < W {
                    let x0 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                    let x1 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                    let x01 = _mm_unpacklo_epi16(x0, x1);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m, j)),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m + 1, j)),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m + 2, j)),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(first_kind, W, m + 3, j)),
                    );
                    j += 2;
                }
                avx2_store4x4_i16_clip::<W>(
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
        coeff[..W * H].fill(0);

        let rnd1_4 = _mm_set1_epi32((1 << shift1) >> 1);
        let rnd1_8 = _mm256_set1_epi32((1 << shift1) >> 1);
        let sh1 = _mm_cvtsi32_si128(shift1);

        let mut x = 0usize;

        // True identity second-pass: write scaled scratch values directly.
        // This removes the dense loop over H with zero coefficient pairs for
        // IDTX and H/V transforms.
        if second_kind == crate::itx_2d::TX_KIND_IDENTITY {
            x = fused_identity_second_pass::<W, H>(
                scratch, dst, dst_off, dst_stride, out_w, out_h, rnd1_4, rnd1_8, sh1, x,
            );
        }

        while x + 16 <= W && second_kind == crate::itx_2d::TX_KIND_DCT && (H == 16 || H == 32) {
            if H == 16 {
                avx2_dct16_i16x16_scratch16_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            } else {
                avx2_dct32_i16x16_scratch16_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            }
            x += 16;
        }
        while x + 8 <= W && second_kind == crate::itx_2d::TX_KIND_DCT && (H == 16 || H == 32) {
            if H == 16 {
                avx2_dct16_i16x8_scratch8_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            } else {
                avx2_dct32_i16x8_scratch8_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            }
            x += 8;
        }
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                avx2_dct16_i16x4_scratch4_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_4, sh1,
                );
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                avx2_dct32_i16x4_scratch4_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_4, sh1,
                );
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = avx2_load4_i16_scratch(scratch, x + j * W);
                        let x1 = avx2_load4_i16_scratch(scratch, x + (j + 1) * W);
                        let x01 = _mm_unpacklo_epi16(x0, x1);
                        a0 = _mm_add_epi32(
                            a0,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m, j)),
                        );
                        a1 = _mm_add_epi32(
                            a1,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m + 1, j)),
                        );
                        a2 = _mm_add_epi32(
                            a2,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m + 2, j)),
                        );
                        a3 = _mm_add_epi32(
                            a3,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(second_kind, H, m + 3, j)),
                        );
                        j += 2;
                    }
                    avx2_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1_4, sh1,
                    );
                    avx2_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1_4,
                        sh1,
                    );
                    avx2_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1_4,
                        sh1,
                    );
                    avx2_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1_4,
                        sh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
    });
}

// Hot fused path: keep kind pairs as const generics only for the small
// curated set used by the dispatcher below. The broad fallback remains
// runtime-kind SIMD to avoid rebuilding the full shape × kind-pair grid.
#[inline]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_fused_8bpc_hot_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
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

    with_avx2_itx_i16_scratch(N, |scratch| {
        scratch.fill(0);
        let mut y = 0usize;

        if FIRST_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            y = fused_identity_pass::<W, H, IS_RECT2>(
                coeff, nrows, rnd, sh, minv, maxv, scratch, y,
            );
        }

        if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = avx2_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, sh, minv, maxv,
            );
        } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = avx2_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, sh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            let mut m = 0usize;
            while m < W {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < W {
                    let x0 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                    let x1 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                    let x01 = _mm_unpacklo_epi16(x0, x1);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(FIRST_KIND, W, m, j)),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(FIRST_KIND, W, m + 1, j)),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(FIRST_KIND, W, m + 2, j)),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(FIRST_KIND, W, m + 3, j)),
                    );
                    j += 2;
                }
                avx2_store4x4_i16_clip::<W>(
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
        coeff[..W * H].fill(0);

        let rnd1_4 = _mm_set1_epi32((1 << shift1) >> 1);
        let rnd1_8 = _mm256_set1_epi32((1 << shift1) >> 1);
        let sh1 = _mm_cvtsi32_si128(shift1);

        let mut x = 0usize;

        // True identity second-pass: write scaled scratch values directly.
        // This removes the dense loop over H with zero coefficient pairs for
        // IDTX and H/V transforms.
        if SECOND_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            x = fused_identity_second_pass::<W, H>(
                scratch, dst, dst_off, dst_stride, out_w, out_h, rnd1_4, rnd1_8, sh1, x,
            );
        }

        while x + 16 <= W && SECOND_KIND == crate::itx_2d::TX_KIND_DCT && (H == 16 || H == 32) {
            if H == 16 {
                avx2_dct16_i16x16_scratch16_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            } else {
                avx2_dct32_i16x16_scratch16_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            }
            x += 16;
        }
        while x + 8 <= W && SECOND_KIND == crate::itx_2d::TX_KIND_DCT && (H == 16 || H == 32) {
            if H == 16 {
                avx2_dct16_i16x8_scratch8_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            } else {
                avx2_dct32_i16x8_scratch8_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_8, sh1,
                );
            }
            x += 8;
        }
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                avx2_dct16_i16x4_scratch4_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_4, sh1,
                );
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                avx2_dct32_i16x4_scratch4_stride_eob_add_u8::<W>(
                    scratch, x, nrows, dst, dst_off, dst_stride, out_w, out_h, rnd1_4, sh1,
                );
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = avx2_load4_i16_scratch(scratch, x + j * W);
                        let x1 = avx2_load4_i16_scratch(scratch, x + (j + 1) * W);
                        let x01 = _mm_unpacklo_epi16(x0, x1);
                        a0 = _mm_add_epi32(
                            a0,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(SECOND_KIND, H, m, j)),
                        );
                        a1 = _mm_add_epi32(
                            a1,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(SECOND_KIND, H, m + 1, j)),
                        );
                        a2 = _mm_add_epi32(
                            a2,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(SECOND_KIND, H, m + 2, j)),
                        );
                        a3 = _mm_add_epi32(
                            a3,
                            _mm_madd_epi16(x01, avx2_tx_dense_coeff_pair(SECOND_KIND, H, m + 3, j)),
                        );
                        j += 2;
                    }
                    avx2_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1_4, sh1,
                    );
                    avx2_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1_4,
                        sh1,
                    );
                    avx2_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1_4,
                        sh1,
                    );
                    avx2_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1_4,
                        sh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
    });
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn fused_identity_pass<const W: usize, const H: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    nrows: usize,
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
    scratch: &mut [i16],
    y: usize,
) -> usize {
    let scale = _mm_set1_epi32(avx2_identity_scale(W));
    let mut y = y;
    while y + 4 <= nrows {
        let mut m = 0usize;
        while m < W {
            let a0 = avx2_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 0) * H, scale);
            let a1 = avx2_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 1) * H, scale);
            let a2 = avx2_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 2) * H, scale);
            let a3 = avx2_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 3) * H, scale);
            avx2_store4x4_i16_clip::<W>(scratch, y * W + m, a0, a1, a2, a3, rnd, sh, minv, maxv);
            m += 4;
        }
        y += 4;
    }
    y
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn fused_identity_second_pass<const W: usize, const H: usize>(
    scratch: &[i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1_4: __m128i,
    rnd1_8: __m256i,
    sh1: __m128i,
    mut x: usize,
) -> usize {
    let scale = _mm_set1_epi32(avx2_identity_scale(H));
    while x + 8 <= W {
        let mut m = 0usize;
        while m < H {
            let lo = avx2_identity_i16x4_scratch_to_i32(scratch, x + m * W, scale);
            let hi = avx2_identity_i16x4_scratch_to_i32(scratch, x + 4 + m * W, scale);
            let v = _mm256_set_m128i(hi, lo);
            avx2_writeback8_i32_u8::<W, H>(
                dst, dst_off, dst_stride, out_w, out_h, x, m, v, rnd1_8, sh1,
            );
            m += 1;
        }
        x += 8;
    }
    while x < W {
        let mut m = 0usize;
        while m < H {
            let a = avx2_identity_i16x4_scratch_to_i32(scratch, x + m * W, scale);
            avx2_writeback4_i32_u8::<W, H>(
                dst, dst_off, dst_stride, out_w, out_h, x, m, a, rnd1_4, sh1,
            );
            m += 1;
        }
        x += 4;
    }
    x
}

#[inline]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_fused_4x4_impl(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_avx2_i16_fused_4x4_const::<true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_avx2_i16_fused_4x4_const::<false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_fused_4x4_const<const IS_RECT2: bool>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(coeff.len() >= 16);
    let z = _mm_setzero_si128();
    let rnd = _mm_set1_epi32((1 << shift0) >> 1);
    let sh = _mm_cvtsi32_si128(shift0);
    let minv = _mm_set1_epi32(row_clip_min);
    let maxv = _mm_set1_epi32(row_clip_max);

    let c0 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 0);
    let c1 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 4);
    let c2 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 8);
    let c3 = avx2_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 12);
    let c01 = _mm_unpacklo_epi16(c0, c1);
    let c23 = _mm_unpacklo_epi16(c2, c3);

    macro_rules! row_pass {
        ($m:expr) => {{
            _mm_add_epi32(
                _mm_madd_epi16(c01, avx2_tx_dense_coeff_pair(first_kind, 4, $m, 0)),
                _mm_madd_epi16(c23, avx2_tx_dense_coeff_pair(first_kind, 4, $m, 2)),
            )
        }};
    }
    macro_rules! clip {
        ($x:expr) => {{
            _mm_min_epi32(
                _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32($x, rnd), sh), minv),
                maxv,
            )
        }};
    }
    let r0 = clip!(row_pass!(0));
    let r1 = clip!(row_pass!(1));
    let r2 = clip!(row_pass!(2));
    let r3 = clip!(row_pass!(3));

    // Transpose exactly like avx2_store4x4_i16_clip::<4>, but keep the
    // four packed rows in registers for the column pass.
    let t0 = _mm_unpacklo_epi32(r0, r1);
    let t1 = _mm_unpackhi_epi32(r0, r1);
    let t2 = _mm_unpacklo_epi32(r2, r3);
    let t3 = _mm_unpackhi_epi32(r2, r3);
    let s0 = _mm_packs_epi32(_mm_unpacklo_epi64(t0, t2), z);
    let s1 = _mm_packs_epi32(_mm_unpackhi_epi64(t0, t2), z);
    let s2 = _mm_packs_epi32(_mm_unpacklo_epi64(t1, t3), z);
    let s3 = _mm_packs_epi32(_mm_unpackhi_epi64(t1, t3), z);

    let s01 = _mm_unpacklo_epi16(s0, s1);
    let s23 = _mm_unpacklo_epi16(s2, s3);
    macro_rules! col_pass {
        ($m:expr) => {{
            _mm_add_epi32(
                _mm_madd_epi16(s01, avx2_tx_dense_coeff_pair(second_kind, 4, $m, 0)),
                _mm_madd_epi16(s23, avx2_tx_dense_coeff_pair(second_kind, 4, $m, 2)),
            )
        }};
    }

    let rnd1 = _mm_set1_epi32((1 << shift1) >> 1);
    let sh1 = _mm_cvtsi32_si128(shift1);
    avx2_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        0,
        col_pass!(0),
        rnd1,
        sh1,
    );
    avx2_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        1,
        col_pass!(1),
        rnd1,
        sh1,
    );
    avx2_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        2,
        col_pass!(2),
        rnd1,
        sh1,
    );
    avx2_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        3,
        col_pass!(3),
        rnd1,
        sh1,
    );
    coeff[..16].fill(0);
}

#[inline]
#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_fused_hot_square<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    debug_assert_eq!(W, H);
    macro_rules! call_pair {
        ($first:expr, $second:expr) => {{
            tx_dequant_dense_avx2_i16_fused_8bpc_hot_impl_const::<
                N,
                W,
                H,
                false,
                { $first },
                { $second },
            >(
                coeff,
                dst,
                dst_off,
                dst_stride,
                out_w,
                out_h,
                eob,
                tx,
                shift0,
                row_clip_min,
                row_clip_max,
                shift1,
            );
            true
        }};
    }

    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => false,
    }
}

#[target_feature(enable = "avx2")]
fn tx_dequant_dense_avx2_i16_fused_8bpc_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_avx2_i16_fused_8bpc_impl_const::<N, W, H, true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_avx2_i16_fused_8bpc_impl_const::<N, W, H, false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    }
}

macro_rules! avx2_fused_match_body {
    ($call:ident, $coeff:ident, $dst:ident, $dst_off:ident, $dst_stride:ident, $out_w:ident, $out_h:ident, $eob:ident, $tx:ident, $is_rect2:ident, $shift0:ident, $row_clip_min:ident, $row_clip_max:ident, $shift1:ident, $first_kind:ident, $second_kind:ident) => {{
        match $tx {
            crate::levels::txsz::TX_8X8 => $call::<64, 8, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_16X16 => $call::<256, 16, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_32X32 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_64X64 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X8 => $call::<32, 4, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X4 => $call::<32, 8, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X16 => $call::<128, 8, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X8 => $call::<128, 16, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X32 => $call::<512, 16, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X16 => $call::<512, 32, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X64 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X32 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X16 => $call::<64, 4, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X4 => $call::<64, 16, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X32 => $call::<256, 8, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X8 => $call::<256, 32, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X64 => $call::<512, 16, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X16 => $call::<512, 32, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X32 => $call::<128, 4, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X4 => $call::<128, 32, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X64 => $call::<256, 8, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X8 => $call::<256, 32, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X64 => $call::<128, 4, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X4 => $call::<128, 32, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            _ => return false,
        }
        true
    }};
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn itx_dequant_i16_avx2_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    if tx == crate::levels::txsz::TX_4X4 {
        tx_dequant_dense_avx2_i16_fused_4x4_impl(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        );
        return true;
    }

    // Recover the hot 8x8/16x16 DCT/ADST/FLIPADST pairs with const-kind
    // coefficient tables. Cold/rectangular/identity pairs continue through the
    // runtime-kind SIMD body below, so we do not resurrect the full kind grid.
    if !is_rect2 {
        let handled_hot = match tx {
            crate::levels::txsz::TX_8X8 => tx_dequant_dense_avx2_i16_fused_hot_square::<64, 8, 8>(
                coeff,
                dst,
                dst_off,
                dst_stride,
                out_w,
                out_h,
                eob,
                tx,
                shift0,
                row_clip_min,
                row_clip_max,
                shift1,
                first_kind,
                second_kind,
            ),
            crate::levels::txsz::TX_16X16 => {
                tx_dequant_dense_avx2_i16_fused_hot_square::<256, 16, 16>(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                    first_kind,
                    second_kind,
                )
            }
            _ => false,
        };
        if handled_hot {
            return true;
        }
    }

    avx2_fused_match_body!(
        tx_dequant_dense_avx2_i16_fused_8bpc_impl,
        coeff,
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
        first_kind,
        second_kind
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn tx_dequant_8x8_avx2_i32_impl(
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
        tx_dequant_8x8_avx2_i32_impl_const::<true>(
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
        tx_dequant_8x8_avx2_i32_impl_const::<false>(
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
#[target_feature(enable = "avx2")]
fn tx_dequant_8x8_avx2_i32_impl_const<const IS_RECT2: bool>(
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
    let rnd256 = _mm256_set1_epi32((1 << shift0) >> 1);
    let minv256 = _mm256_set1_epi32(row_clip_min);
    let maxv256 = _mm256_set1_epi32(row_clip_max);

    let mut y = 0usize;
    while y + 8 <= ncols {
        let out = avx2_tx8_i32x8_from_coeff8_const::<IS_RECT2>(coeff, y, first_kind);
        avx2_store8x8_i32_clip(tmp, y * 32, &out, rnd256, sh, minv256, maxv256);
        y += 8;
    }
    if y + 4 <= ncols {
        let mut x = 0usize;
        while x < 8 {
            let g = avx2_tx8_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, first_kind, x);
            avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
            x += 4;
        }
        y += 4;
    }
    while y < 8 {
        tmp[y * 32..y * 32 + 8].fill(0);
        y += 1;
    }
    coeff[..64].fill(0);

    let out = avx2_tx8_i32x8_from_tmp8(tmp, 0, second_kind);
    avx2_store_i32x8_rows(tmp, 0, &out);
}

#[inline]
#[target_feature(enable = "avx2")]
fn idct_dequant_16x16_avx2_i32_impl(
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
        idct_dequant_16x16_avx2_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_16x16_avx2_i32_impl_const::<false>(
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
#[target_feature(enable = "avx2")]
fn idct_dequant_16x16_avx2_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
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
    let rnd256 = _mm256_set1_epi32((1 << shift0) >> 1);
    let minv256 = _mm256_set1_epi32(row_clip_min);
    let maxv256 = _mm256_set1_epi32(row_clip_max);

    let mut y = 0usize;
    while y + 8 <= ncols {
        let out =
            avx2_tx16_i32x8_from_coeff8_const::<IS_RECT2>(coeff, y, 16, crate::itx_2d::TX_KIND_DCT);
        avx2_store8x8_i32_clip(
            tmp,
            y * 32,
            array_ref8_i32x8(&out, 0),
            rnd256,
            sh,
            minv256,
            maxv256,
        );
        avx2_store8x8_i32_clip(
            tmp,
            y * 32 + 8,
            array_ref8_i32x8(&out, 8),
            rnd256,
            sh,
            minv256,
            maxv256,
        );
        y += 8;
    }
    if y + 4 <= ncols {
        let out =
            avx2_tx16_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, 16, crate::itx_2d::TX_KIND_DCT);
        let mut x = 0usize;
        while x < 16 {
            let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
            avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
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
    while x + 8 <= 16 {
        let out = avx2_tx16_i32x8_from_tmp8(tmp, x, crate::itx_2d::TX_KIND_DCT);
        let mut m = 0usize;
        while m < 16 {
            unsafe {
                _mm256_storeu_si256(tmp.as_mut_ptr().add(x + m * 32) as *mut __m256i, out[m]);
            }
            m += 1;
        }
        x += 8;
    }
}

#[target_feature(enable = "avx2")]
fn idct_dequant_32x32_avx2_i32_impl(
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
        idct_dequant_32x32_avx2_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_32x32_avx2_i32_impl_const::<false>(
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

#[target_feature(enable = "avx2")]
fn idct_dequant_32x32_avx2_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
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
    let rnd256 = _mm256_set1_epi32((1 << shift0) >> 1);
    let minv256 = _mm256_set1_epi32(row_clip_min);
    let maxv256 = _mm256_set1_epi32(row_clip_max);

    let mut y = 0usize;
    while y + 8 <= ncols {
        let mut x = 0usize;
        while x < 32 {
            let g = avx2_dct32_i32x8_from_coeff8_const::<IS_RECT2>(coeff, y, x);
            avx2_store8x8_i32_clip(tmp, y * 32 + x, &g, rnd256, sh, minv256, maxv256);
            x += 8;
        }
        y += 8;
    }
    if y + 4 <= ncols {
        let mut x = 0usize;
        while x < 32 {
            let g = avx2_dct32_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, x);
            avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
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
    while x + 8 <= 32 {
        // Compute all output-row groups from pristine row-pass result before
        // storing any (in-place aliasing).
        let groups = [
            avx2_dct32_i32x8_from_tmp8(tmp, x, 0),
            avx2_dct32_i32x8_from_tmp8(tmp, x, 8),
            avx2_dct32_i32x8_from_tmp8(tmp, x, 16),
            avx2_dct32_i32x8_from_tmp8(tmp, x, 24),
        ];
        let mut m = 0usize;
        while m < 32 {
            let g = &groups[m / 8];
            let mut lane = 0usize;
            while lane < 8 {
                unsafe {
                    _mm256_storeu_si256(
                        tmp.as_mut_ptr().add(x + (m + lane) * 32) as *mut __m256i,
                        g[lane],
                    );
                }
                lane += 1;
            }
            m += 8;
        }
        x += 8;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x4_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_4x4_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x8_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_8x8_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_16x16_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_64x64_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<1024, 32, 32>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_4x4_avx2(
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
    tx_dequant_4x4_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x8_avx2(
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
    tx_dequant_8x8_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x16_avx2(
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
    iadst_dequant_16x16_avx2_i32_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x8_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<32, 4, 8>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x4_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<32, 8, 4>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x16_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<128, 8, 16>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x8_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<128, 16, 8>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x32_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<512, 16, 32>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x16_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<512, 32, 16>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x16_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<64, 4, 16>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x4_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<64, 16, 4>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x32_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<256, 8, 32>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x8_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<256, 32, 8>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x32_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<128, 4, 32>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x4_avx2(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i32_impl::<128, 32, 4>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_4x8_avx2(
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
    tx_dequant_dense_avx2_i32_impl::<32, 4, 8>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x4_avx2(
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
    tx_dequant_dense_avx2_i32_impl::<32, 8, 4>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x16_avx2(
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
    tx_dequant_dense_avx2_i32_impl::<128, 8, 16>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x8_avx2(
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
    tx_dequant_dense_avx2_i32_impl::<128, 16, 8>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_4x16_avx2(
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
    tx_dequant_dense_avx2_i32_impl::<64, 4, 16>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x4_avx2(
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
    tx_dequant_dense_avx2_i32_impl::<64, 16, 4>(
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
        #[target_feature(enable = "avx2")]
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
            tx_dequant_dense_avx2_i16_impl::<{ $n }, { $s }, { $s }>(
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
        #[target_feature(enable = "avx2")]
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
            tx_dequant_dense_avx2_i16_impl::<{ $n }, { $s }, { $s }>(
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
        #[target_feature(enable = "avx2")]
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
            tx_dequant_dense_avx2_i16_impl::<{ $n }, { $w }, { $h }>(
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
        #[target_feature(enable = "avx2")]
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
            tx_dequant_dense_avx2_i16_impl::<{ $n }, { $w }, { $h }>(
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
    idct_dequant_4x4_i16_avx2,
    idct_dequant_4x4_i16_avx2_impl,
    16,
    4
);
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x8_i16_avx2(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_avx2_i16_impl::<64, 8, 8>(
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
#[target_feature(enable = "avx2")]
fn idct_dequant_dct_i16_avx2_fused_8bpc_impl<const N: usize>(
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
        idct_dequant_dct_i16_avx2_fused_8bpc_impl_const::<N, true>(
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
        idct_dequant_dct_i16_avx2_fused_8bpc_impl_const::<N, false>(
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

// Keep the very large 32-point i16 DCT bodies in fixed, non-generic call
// targets. The public dispatch wrappers stay tiny, while this isolates the
// expensive monomorphized transform body behind one call. That call is far
// cheaper than the 32x32 transform itself and avoids cloning the 32-point graph
// into every wrapper that reaches it.
#[target_feature(enable = "avx2")]
fn idct_dequant_32x32_i16_avx2_fixed_fused_8bpc_impl(
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
    idct_dequant_dct_i16_avx2_fused_8bpc_impl::<32>(
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

#[target_feature(enable = "avx2")]
fn idct_dequant_32x32_i16_avx2_fixed_impl(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_avx2_impl::<32>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_i16_avx2_fused_8bpc(
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
    idct_dequant_dct_i16_avx2_fused_8bpc_impl::<16>(
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

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_i16_avx2_fused_8bpc(
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
    idct_dequant_32x32_i16_avx2_fixed_fused_8bpc_impl(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_i16_avx2(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_avx2_impl::<16>(
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

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_i16_avx2(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_i16_avx2_fixed_impl(
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
    idct_dequant_64x64_i16_avx2,
    idct_dequant_64x64_i16_avx2_impl,
    1024,
    32
);
iadst_i16_fn!(
    iadst_dequant_4x4_i16_avx2,
    iadst_dequant_4x4_i16_avx2_impl,
    16,
    4
);
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x8_i16_avx2(
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
    tx_dequant_dense_avx2_i16_impl::<64, 8, 8>(
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
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x16_i16_avx2(
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
    tx_dequant_dense_avx2_i16_impl::<256, 16, 16>(
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
    idct_dequant_4x8_i16_avx2,
    idct_dequant_4x8_i16_avx2_impl,
    32,
    4,
    8
);
idct_rect_i16_fn!(
    idct_dequant_8x4_i16_avx2,
    idct_dequant_8x4_i16_avx2_impl,
    32,
    8,
    4
);
idct_rect_i16_fn!(
    idct_dequant_8x16_i16_avx2,
    idct_dequant_8x16_i16_avx2_impl,
    128,
    8,
    16
);
idct_rect_i16_fn!(
    idct_dequant_16x8_i16_avx2,
    idct_dequant_16x8_i16_avx2_impl,
    128,
    16,
    8
);
idct_rect_i16_fn!(
    idct_dequant_16x32_i16_avx2,
    idct_dequant_16x32_i16_avx2_impl,
    512,
    16,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x16_i16_avx2,
    idct_dequant_32x16_i16_avx2_impl,
    512,
    32,
    16
);
idct_rect_i16_fn!(
    idct_dequant_4x16_i16_avx2,
    idct_dequant_4x16_i16_avx2_impl,
    64,
    4,
    16
);
idct_rect_i16_fn!(
    idct_dequant_16x4_i16_avx2,
    idct_dequant_16x4_i16_avx2_impl,
    64,
    16,
    4
);
idct_rect_i16_fn!(
    idct_dequant_8x32_i16_avx2,
    idct_dequant_8x32_i16_avx2_impl,
    256,
    8,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x8_i16_avx2,
    idct_dequant_32x8_i16_avx2_impl,
    256,
    32,
    8
);
idct_rect_i16_fn!(
    idct_dequant_4x32_i16_avx2,
    idct_dequant_4x32_i16_avx2_impl,
    128,
    4,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x4_i16_avx2,
    idct_dequant_32x4_i16_avx2_impl,
    128,
    32,
    4
);
iadst_rect_i16_fn!(
    iadst_dequant_4x8_i16_avx2,
    iadst_dequant_4x8_i16_avx2_impl,
    32,
    4,
    8
);
iadst_rect_i16_fn!(
    iadst_dequant_8x4_i16_avx2,
    iadst_dequant_8x4_i16_avx2_impl,
    32,
    8,
    4
);
iadst_rect_i16_fn!(
    iadst_dequant_8x16_i16_avx2,
    iadst_dequant_8x16_i16_avx2_impl,
    128,
    8,
    16
);
iadst_rect_i16_fn!(
    iadst_dequant_16x8_i16_avx2,
    iadst_dequant_16x8_i16_avx2_impl,
    128,
    16,
    8
);
iadst_rect_i16_fn!(
    iadst_dequant_4x16_i16_avx2,
    iadst_dequant_4x16_i16_avx2_impl,
    64,
    4,
    16
);
iadst_rect_i16_fn!(
    iadst_dequant_16x4_i16_avx2,
    iadst_dequant_16x4_i16_avx2_impl,
    64,
    16,
    4
);
