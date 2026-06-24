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

use crate::itx_2d::{
    Adst2dBackend, Dct2dBackend, DctSimd4, ITX_TMP_PIXELS, idct_dequant_simd4_core,
    itx_dequant_simd4_core,
};

#[derive(Clone, Copy)]
pub(crate) struct SseI32x4(__m128i);

impl crate::itx_1d::DctLane for SseI32x4 {
    #[inline(always)]
    fn zero() -> Self {
        SseI32x4(unsafe { _mm_setzero_si128() })
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        SseI32x4(unsafe { _mm_add_epi32(self.0, o.0) })
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        SseI32x4(unsafe { _mm_sub_epi32(self.0, o.0) })
    }
    #[inline(always)]
    fn mul(self, k: Self) -> Self {
        SseI32x4(unsafe { _mm_mullo_epi32(self.0, k.0) })
    }
    #[inline(always)]
    fn dup_load(table: &[i32], idx: usize) -> Self {
        // SAFETY: callers index within the kernel tables.
        SseI32x4(unsafe { _mm_set1_epi32(*table.get_unchecked(idx)) })
    }
    #[inline(always)]
    fn mul_add(self, x: Self, k: Self) -> Self {
        // SSE has no integer FMA: multiply-low then add.
        SseI32x4(unsafe { _mm_add_epi32(self.0, _mm_mullo_epi32(x.0, k.0)) })
    }
    type Coeffs = __m128i;
    #[inline(always)]
    fn load_coeffs(table: &[i32], idx: usize) -> __m128i {
        // SAFETY: callers index a 4-wide group within the kernel tables.
        unsafe { _mm_loadu_si128(table.as_ptr().add(idx) as *const __m128i) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, x: Self, c: __m128i) -> Self {
        // SSE has no by-lane multiply: broadcast lane LANE, then mul-add.
        let bc = unsafe {
            match LANE {
                0 => _mm_shuffle_epi32(c, 0x00),
                1 => _mm_shuffle_epi32(c, 0x55),
                2 => _mm_shuffle_epi32(c, 0xAA),
                _ => _mm_shuffle_epi32(c, 0xFF),
            }
        };
        SseI32x4(unsafe { _mm_add_epi32(self.0, _mm_mullo_epi32(x.0, bc)) })
    }
}

pub(crate) struct SseWide;

impl crate::itx_1d::DctWide for SseWide {
    type In = __m128i;
    type Acc = (__m128i, __m128i);
    type Coeffs = __m128i;
    type Clip = (__m128i, __m128i, __m128i, __m128i);
    #[inline(always)]
    fn zero() -> Self::Acc {
        unsafe { (_mm_setzero_si128(), _mm_setzero_si128()) }
    }
    #[inline(always)]
    fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { (_mm_add_epi32(a.0, b.0), _mm_add_epi32(a.1, b.1)) }
    }
    #[inline(always)]
    fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { (_mm_sub_epi32(a.0, b.0), _mm_sub_epi32(a.1, b.1)) }
    }
    #[inline(always)]
    fn load_coeffs(table: &[i16], idx: usize) -> __m128i {
        unsafe { _mm_loadu_si128(table.as_ptr().add(idx) as *const __m128i) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(acc: Self::Acc, x: __m128i, c: __m128i) -> Self::Acc {
        unsafe {
            // Fallback single-tap widening MAC. The paired path below is used by
            // the hot ITX kernels; keep this for the generic default and odd
            // future callers.
            let raw = match LANE {
                0 => _mm_extract_epi16(c, 0),
                1 => _mm_extract_epi16(c, 1),
                2 => _mm_extract_epi16(c, 2),
                3 => _mm_extract_epi16(c, 3),
                4 => _mm_extract_epi16(c, 4),
                5 => _mm_extract_epi16(c, 5),
                6 => _mm_extract_epi16(c, 6),
                _ => _mm_extract_epi16(c, 7),
            };
            let k = _mm_set1_epi32((raw as i16) as i32);
            let xlo = _mm_unpacklo_epi16(x, _mm_setzero_si128());
            let xhi = _mm_unpackhi_epi16(x, _mm_setzero_si128());
            (
                _mm_add_epi32(acc.0, _mm_madd_epi16(xlo, k)),
                _mm_add_epi32(acc.1, _mm_madd_epi16(xhi, k)),
            )
        }
    }
    #[inline(always)]
    fn mul_add_pair<const LANE0: i32, const LANE1: i32>(
        acc: Self::Acc,
        x0: __m128i,
        x1: __m128i,
        c: __m128i,
    ) -> Self::Acc {
        unsafe {
            let _ = LANE1;
            debug_assert_eq!(LANE1, LANE0 + 1);
            debug_assert_eq!(LANE0 & 1, 0);

            // c is eight i16 coefficients. Replicate an adjacent i16 pair as
            // [c0, c1, c0, c1, ...], then use madd_epi16 on interleaved source
            // pairs: x0[i] * c0 + x1[i] * c1. This halves the number of madds
            // and avoids the extract+broadcast sequence in the single-lane path.
            let k01 = match LANE0 {
                0 => _mm_shuffle_epi32(c, 0x00),
                2 => _mm_shuffle_epi32(c, 0x55),
                4 => _mm_shuffle_epi32(c, 0xaa),
                _ => _mm_shuffle_epi32(c, 0xff),
            };
            let xlo = _mm_unpacklo_epi16(x0, x1);
            let xhi = _mm_unpackhi_epi16(x0, x1);
            (
                _mm_add_epi32(acc.0, _mm_madd_epi16(xlo, k01)),
                _mm_add_epi32(acc.1, _mm_madd_epi16(xhi, k01)),
            )
        }
    }
    #[inline(always)]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> __m128i {
        unsafe {
            let lo = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            let hi = _mm_loadu_si128(src.as_ptr().add(off + 4) as *const __m128i);
            _mm_packs_epi32(lo, hi)
        }
    }
    #[inline(always)]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> __m128i {
        unsafe {
            let lo = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            let hi = _mm_loadu_si128(src.as_ptr().add(off + 4) as *const __m128i);
            _mm_mulhrs_epi16(_mm_packs_epi32(lo, hi), _mm_set1_epi16(0x5a80))
        }
    }
    #[inline(always)]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> __m128i {
        unsafe {
            let lo = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            _mm_packs_epi32(lo, _mm_setzero_si128())
        }
    }
    #[inline(always)]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> __m128i {
        unsafe { _mm_mulhrs_epi16(Self::load4_narrow(src, off), _mm_set1_epi16(0x5a80)) }
    }
    #[inline(always)]
    unsafe fn load8_i16(src: &[i16], off: usize) -> __m128i {
        debug_assert!(off + 8 <= src.len());
        unsafe { _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i) }
    }
    #[inline(always)]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> __m128i {
        unsafe { _mm_mulhrs_epi16(Self::load8_i16(src, off), _mm_set1_epi16(0x5a80)) }
    }
    #[inline(always)]
    unsafe fn load4_i16(src: &[i16], off: usize) -> __m128i {
        debug_assert!(off + 4 <= src.len());
        unsafe { _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i) }
    }
    #[inline(always)]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> __m128i {
        unsafe { _mm_mulhrs_epi16(Self::load4_i16(src, off), _mm_set1_epi16(0x5a80)) }
    }
    #[inline(always)]
    fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        unsafe {
            (
                _mm_set1_epi32(rnd),
                _mm_cvtsi32_si128(shift),
                _mm_set1_epi32(min),
                _mm_set1_epi32(max),
            )
        }
    }
    #[inline(always)]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, sh, minv, maxv) = clip;

            let lo = _mm_min_epi32(
                _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(acc.0, rnd), sh), minv),
                maxv,
            );
            let hi = _mm_min_epi32(
                _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(acc.1, rnd), sh), minv),
                maxv,
            );

            #[inline(always)]
            fn store_lane0(dst: &mut [i32], off: usize, v: __m128i) {
                unsafe {
                    _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(v));
                }
            }

            store_lane0(dst, off, lo);
            store_lane0(dst, off + 1 * stride, _mm_shuffle_epi32::<0x55>(lo));
            store_lane0(dst, off + 2 * stride, _mm_shuffle_epi32::<0xaa>(lo));
            store_lane0(dst, off + 3 * stride, _mm_shuffle_epi32::<0xff>(lo));

            store_lane0(dst, off + 4 * stride, hi);
            store_lane0(dst, off + 5 * stride, _mm_shuffle_epi32::<0x55>(hi));
            store_lane0(dst, off + 6 * stride, _mm_shuffle_epi32::<0xaa>(hi));
            store_lane0(dst, off + 7 * stride, _mm_shuffle_epi32::<0xff>(hi));
        }
    }

    #[inline(always)]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, sh, minv, maxv) = clip;
            let lo = _mm_min_epi32(
                _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(acc.0, rnd), sh), minv),
                maxv,
            );
            #[inline(always)]
            fn store_lane0(dst: &mut [i32], off: usize, v: __m128i) {
                unsafe {
                    _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(v));
                }
            }

            store_lane0(dst, off, lo);
            store_lane0(dst, off + 1 * stride, _mm_shuffle_epi32::<0x55>(lo));
            store_lane0(dst, off + 2 * stride, _mm_shuffle_epi32::<0xaa>(lo));
            store_lane0(dst, off + 3 * stride, _mm_shuffle_epi32::<0xff>(lo));
        }
    }

    #[inline(always)]
    unsafe fn store4x4_strided_clip<const HIGH: bool>(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 4],
        clip: Self::Clip,
    ) {
        unsafe {
            #[inline(always)]
            unsafe fn clip_vec(
                v: __m128i,
                rnd: __m128i,
                sh: __m128i,
                minv: __m128i,
                maxv: __m128i,
            ) -> __m128i {
                unsafe {
                    _mm_min_epi32(
                        _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(v, rnd), sh), minv),
                        maxv,
                    )
                }
            }
            let (rnd, sh, minv, maxv) = clip;
            let c0 = clip_vec(if HIGH { acc[0].1 } else { acc[0].0 }, rnd, sh, minv, maxv);
            let c1 = clip_vec(if HIGH { acc[1].1 } else { acc[1].0 }, rnd, sh, minv, maxv);
            let c2 = clip_vec(if HIGH { acc[2].1 } else { acc[2].0 }, rnd, sh, minv, maxv);
            let c3 = clip_vec(if HIGH { acc[3].1 } else { acc[3].0 }, rnd, sh, minv, maxv);

            let t0 = _mm_unpacklo_epi32(c0, c1);
            let t1 = _mm_unpackhi_epi32(c0, c1);
            let t2 = _mm_unpacklo_epi32(c2, c3);
            let t3 = _mm_unpackhi_epi32(c2, c3);
            let r0 = _mm_unpacklo_epi64(t0, t2);
            let r1 = _mm_unpackhi_epi64(t0, t2);
            let r2 = _mm_unpacklo_epi64(t1, t3);
            let r3 = _mm_unpackhi_epi64(t1, t3);

            _mm_storeu_si128(dst.as_mut_ptr().add(off) as *mut __m128i, r0);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + stride) as *mut __m128i, r1);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + 2 * stride) as *mut __m128i, r2);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + 3 * stride) as *mut __m128i, r3);
        }
    }

    #[inline(always)]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            _mm_storeu_si128(dst.as_mut_ptr().add(off) as *mut __m128i, acc.0);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + 4) as *mut __m128i, acc.1);
        }
    }

    #[inline(always)]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            _mm_storeu_si128(dst.as_mut_ptr().add(off) as *mut __m128i, acc.0);
        }
    }
}

pub(crate) struct SseDct2d;

impl DctSimd4 for SseDct2d {
    type V = SseI32x4;
    type Wide = SseWide;

    #[inline(always)]
    unsafe fn zero() -> Self::V {
        SseI32x4(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    unsafe fn splat(v: i32) -> Self::V {
        SseI32x4(unsafe { _mm_set1_epi32(v) })
    }

    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        SseI32x4(unsafe { _mm_add_epi32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        SseI32x4(unsafe { _mm_sub_epi32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        SseI32x4(unsafe { _mm_mullo_epi32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe {
            let scaled = _mm_add_epi32(
                _mm_mullo_epi32(a.0, _mm_set1_epi32(181)),
                _mm_set1_epi32(128),
            );
            SseI32x4(_mm_srai_epi32::<8>(scaled))
        }
    }

    #[inline(always)]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_ptr().add(off) as *const __m128i };
        SseI32x4(unsafe { _mm_loadu_si128(p) })
    }

    #[inline(always)]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_mut_ptr().add(off) as *mut __m128i };
        unsafe { _mm_storeu_si128(p, v.0) };
    }

    #[inline(always)]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) as *const __m128i };
        SseI32x4(unsafe { _mm_loadu_si128(p) })
    }

    #[inline(always)]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) as *const __m128i };
        SseI32x4(unsafe { _mm_cvtepi16_epi32(_mm_loadl_epi64(p)) })
    }

    #[inline(always)]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        let mut out = [0i32; 4];
        let p = out.as_mut_ptr() as *mut __m128i;
        unsafe { _mm_storeu_si128(p, v.0) };
        out
    }

    #[inline(always)]
    unsafe fn store4x4_clip(
        tmp: &mut [i32; ITX_TMP_PIXELS],
        off: usize,
        stride: usize,
        v: [Self::V; 4],
        rnd: i32,
        shift: i32,
        min: i32,
        max: i32,
    ) {
        debug_assert!(off + 3 + 3 * stride < ITX_TMP_PIXELS);
        unsafe {
            #[inline(always)]
            unsafe fn clip_vec(
                v: __m128i,
                rnd: __m128i,
                sh: __m128i,
                minv: __m128i,
                maxv: __m128i,
            ) -> __m128i {
                unsafe {
                    _mm_min_epi32(
                        _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(v, rnd), sh), minv),
                        maxv,
                    )
                }
            }

            let rnd = _mm_set1_epi32(rnd);
            let sh = _mm_cvtsi32_si128(shift);
            let minv = _mm_set1_epi32(min);
            let maxv = _mm_set1_epi32(max);

            let c0 = clip_vec(v[0].0, rnd, sh, minv, maxv);
            let c1 = clip_vec(v[1].0, rnd, sh, minv, maxv);
            let c2 = clip_vec(v[2].0, rnd, sh, minv, maxv);
            let c3 = clip_vec(v[3].0, rnd, sh, minv, maxv);

            // Transpose columns-as-lanes into four row vectors:
            // cN = [r0cN, r1cN, r2cN, r3cN].
            let t0 = _mm_unpacklo_epi32(c0, c1);
            let t1 = _mm_unpackhi_epi32(c0, c1);
            let t2 = _mm_unpacklo_epi32(c2, c3);
            let t3 = _mm_unpackhi_epi32(c2, c3);
            let r0 = _mm_unpacklo_epi64(t0, t2);
            let r1 = _mm_unpackhi_epi64(t0, t2);
            let r2 = _mm_unpacklo_epi64(t1, t3);
            let r3 = _mm_unpackhi_epi64(t1, t3);

            let p = tmp.as_mut_ptr().add(off) as *mut __m128i;
            _mm_storeu_si128(p, r0);
            _mm_storeu_si128(tmp.as_mut_ptr().add(off + stride) as *mut __m128i, r1);
            _mm_storeu_si128(tmp.as_mut_ptr().add(off + 2 * stride) as *mut __m128i, r2);
            _mm_storeu_si128(tmp.as_mut_ptr().add(off + 3 * stride) as *mut __m128i, r3);
        }
    }
}

impl Dct2dBackend for SseDct2d {
    #[inline(always)]
    fn idct_dequant_4x4(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 16, 4, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_8x8(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 64, 8, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_16x16(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 256, 16, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_32x32(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 1024, 32, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_64x64(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 1024, 32, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }
}

impl Adst2dBackend for SseDct2d {
    #[inline(always)]
    fn iadst_dequant_4x4(
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
        itx_dequant_simd4_core::<Self, 16, 4, i32>(
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
        );
    }

    #[inline(always)]
    fn iadst_dequant_8x8(
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
        itx_dequant_simd4_core::<Self, 64, 8, i32>(
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
        );
    }

    #[inline(always)]
    fn iadst_dequant_16x16(
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
        itx_dequant_simd4_core::<Self, 256, 16, i32>(
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
        );
    }
}

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
    SseDct2d::idct_dequant_4x4(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    SseDct2d::idct_dequant_8x8(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    SseDct2d::idct_dequant_16x16(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    SseDct2d::idct_dequant_32x32(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    SseDct2d::idct_dequant_64x64(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    SseDct2d::iadst_dequant_4x4(
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
    );
}

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
    SseDct2d::iadst_dequant_8x8(
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
    );
}

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
    SseDct2d::iadst_dequant_16x16(
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
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 32, 4, 8, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 32, 8, 4, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 128, 8, 16, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 128, 16, 8, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 512, 16, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 512, 32, 16, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 64, 4, 16, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 64, 16, 4, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 256, 8, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 256, 32, 8, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 128, 4, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, 128, 32, 4, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, 32, 4, 8, i32>(
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
    );
}

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
    crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, 32, 8, 4, i32>(
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
    );
}

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
    crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, 128, 8, 16, i32>(
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
    );
}

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
    crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, 128, 16, 8, i32>(
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
    );
}

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
    crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, 64, 4, 16, i32>(
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
    );
}

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
    crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, 64, 16, 4, i32>(
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
    );
}

// Low-bit-depth i16 coefficient entry points.

macro_rules! idct_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $s:expr) => {
        #[target_feature(enable = "sse4.1")]
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
            idct_dequant_simd4_core::<SseDct2d, { $n }, { $s }, i16>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            );
        }
    };
}
macro_rules! iadst_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $s:expr) => {
        #[target_feature(enable = "sse4.1")]
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
            itx_dequant_simd4_core::<SseDct2d, { $n }, { $s }, i16>(
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
            );
        }
    };
}
macro_rules! idct_rect_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $w:expr, $h:expr) => {
        #[target_feature(enable = "sse4.1")]
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
            crate::itx_2d::idct_dequant_rect_simd4_core::<SseDct2d, { $n }, { $w }, { $h }, i16>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            );
        }
    };
}
macro_rules! iadst_rect_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $w:expr, $h:expr) => {
        #[target_feature(enable = "sse4.1")]
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
            crate::itx_2d::itx_dequant_rect_simd4_core::<SseDct2d, { $n }, { $w }, { $h }, i16>(
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
            );
        }
    };
}
idct_i16_fn!(
    idct_dequant_4x4_i16_sse41,
    idct_dequant_4x4_i16_sse41_impl,
    16,
    4
);
idct_i16_fn!(
    idct_dequant_8x8_i16_sse41,
    idct_dequant_8x8_i16_sse41_impl,
    64,
    8
);
idct_i16_fn!(
    idct_dequant_16x16_i16_sse41,
    idct_dequant_16x16_i16_sse41_impl,
    256,
    16
);
idct_i16_fn!(
    idct_dequant_32x32_i16_sse41,
    idct_dequant_32x32_i16_sse41_impl,
    1024,
    32
);
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
iadst_i16_fn!(
    iadst_dequant_8x8_i16_sse41,
    iadst_dequant_8x8_i16_sse41_impl,
    64,
    8
);
iadst_i16_fn!(
    iadst_dequant_16x16_i16_sse41,
    iadst_dequant_16x16_i16_sse41_impl,
    256,
    16
);
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
