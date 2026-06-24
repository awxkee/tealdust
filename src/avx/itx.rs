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

use crate::itx_2d::{
    DctSimd4, ITX_TMP_PIXELS, idct_dequant_simd4_core_avx2, itx_dequant_simd4_core_avx2,
};

#[derive(Clone, Copy)]
pub(crate) struct AvxI32x4(__m128i);

impl crate::itx_1d::DctLane for AvxI32x4 {
    #[inline(always)]
    fn zero() -> Self {
        AvxI32x4(unsafe { _mm_setzero_si128() })
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        AvxI32x4(unsafe { _mm_add_epi32(self.0, o.0) })
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        AvxI32x4(unsafe { _mm_sub_epi32(self.0, o.0) })
    }
    #[inline(always)]
    fn mul(self, k: Self) -> Self {
        AvxI32x4(unsafe { _mm_mullo_epi32(self.0, k.0) })
    }
    #[inline(always)]
    fn dup_load(table: &[i32], idx: usize) -> Self {
        // SAFETY: callers index within the kernel tables.
        AvxI32x4(unsafe { _mm_set1_epi32(*table.get_unchecked(idx)) })
    }
    #[inline(always)]
    fn mul_add(self, x: Self, k: Self) -> Self {
        AvxI32x4(unsafe { _mm_add_epi32(self.0, _mm_mullo_epi32(x.0, k.0)) })
    }
    type Coeffs = __m128i;
    #[inline(always)]
    fn load_coeffs(table: &[i32], idx: usize) -> __m128i {
        // SAFETY: callers index a 4-wide group within the kernel tables.
        unsafe { _mm_loadu_si128(table.as_ptr().add(idx) as *const __m128i) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, x: Self, c: __m128i) -> Self {
        let bc = unsafe {
            match LANE {
                0 => _mm_shuffle_epi32(c, 0x00),
                1 => _mm_shuffle_epi32(c, 0x55),
                2 => _mm_shuffle_epi32(c, 0xAA),
                _ => _mm_shuffle_epi32(c, 0xFF),
            }
        };
        AvxI32x4(unsafe { _mm_add_epi32(self.0, _mm_mullo_epi32(x.0, bc)) })
    }
}

pub(crate) struct AvxWide;

impl crate::itx_1d::DctWide for AvxWide {
    type In = __m256i;
    type Acc = __m256i;
    type Coeffs = __m256i;
    type Clip = (__m256i, __m128i, __m256i, __m256i);

    #[inline(always)]
    fn zero() -> Self::Acc {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { _mm256_add_epi32(a, b) }
    }

    #[inline(always)]
    fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { _mm256_sub_epi32(a, b) }
    }

    #[inline(always)]
    fn load_coeffs(table: &[i16], idx: usize) -> __m256i {
        unsafe {
            let c = _mm_loadu_si128(table.as_ptr().add(idx) as *const __m128i);
            _mm256_broadcastsi128_si256(c)
        }
    }

    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(acc: Self::Acc, x: __m256i, c: __m256i) -> Self::Acc {
        unsafe {
            let raw = match LANE {
                0 => _mm_extract_epi16(_mm256_castsi256_si128(c), 0),
                1 => _mm_extract_epi16(_mm256_castsi256_si128(c), 1),
                2 => _mm_extract_epi16(_mm256_castsi256_si128(c), 2),
                3 => _mm_extract_epi16(_mm256_castsi256_si128(c), 3),
                4 => _mm_extract_epi16(_mm256_castsi256_si128(c), 4),
                5 => _mm_extract_epi16(_mm256_castsi256_si128(c), 5),
                6 => _mm_extract_epi16(_mm256_castsi256_si128(c), 6),
                _ => _mm_extract_epi16(_mm256_castsi256_si128(c), 7),
            };
            let xlo16 = _mm256_unpacklo_epi16(x, _mm256_setzero_si256());
            let xhi16 = _mm256_unpackhi_epi16(x, _mm256_setzero_si256());
            let lo = _mm256_madd_epi16(xlo16, _mm256_set1_epi16(raw as i16));
            let hi = _mm256_madd_epi16(xhi16, _mm256_set1_epi16(raw as i16));
            // lo contains rows 0..3 in its low 128-bit lane; hi contains rows
            // 4..7 in its low 128-bit lane. Combine those halves explicitly.
            let sum8 = _mm256_permute2x128_si256::<0x20>(lo, hi);
            _mm256_add_epi32(acc, sum8)
        }
    }

    #[inline(always)]
    fn mul_add_pair<const LANE0: i32, const LANE1: i32>(
        acc: Self::Acc,
        x0: __m256i,
        x1: __m256i,
        c: __m256i,
    ) -> Self::Acc {
        unsafe {
            let _ = LANE1;
            debug_assert_eq!(LANE1, LANE0 + 1);
            debug_assert_eq!(LANE0 & 1, 0);

            let k01 = match LANE0 {
                0 => _mm256_shuffle_epi32::<0x00>(c),
                2 => _mm256_shuffle_epi32::<0x55>(c),
                4 => _mm256_shuffle_epi32::<0xaa>(c),
                _ => _mm256_shuffle_epi32::<0xff>(c),
            };
            let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(x0, x1), k01);
            let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(x0, x1), k01);
            let sum8 = _mm256_permute2x128_si256::<0x20>(lo, hi);
            _mm256_add_epi32(acc, sum8)
        }
    }

    #[inline(always)]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe {
            let v = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
            let p = _mm256_packs_epi32(v, _mm256_setzero_si256());
            // packs_epi32 is lane-local: [0..3, z, 4..7, z] -> [0..7, z].
            _mm256_permute4x64_epi64::<0xd8>(p)
        }
    }

    #[inline(always)]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe {
            let x = Self::load8_narrow(src, off);
            _mm256_mulhrs_epi16(x, _mm256_set1_epi16(0x5a80))
        }
    }

    #[inline(always)]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe {
            let lo = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            let p = _mm_packs_epi32(lo, _mm_setzero_si128());
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), p)
        }
    }

    #[inline(always)]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe { _mm256_mulhrs_epi16(Self::load4_narrow(src, off), _mm256_set1_epi16(0x5a80)) }
    }
    #[inline(always)]
    unsafe fn load8_i16(src: &[i16], off: usize) -> __m256i {
        debug_assert!(off + 8 <= src.len());
        unsafe {
            let x = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), x)
        }
    }
    #[inline(always)]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> __m256i {
        unsafe { _mm256_mulhrs_epi16(Self::load8_i16(src, off), _mm256_set1_epi16(0x5a80)) }
    }
    #[inline(always)]
    unsafe fn load4_i16(src: &[i16], off: usize) -> __m256i {
        debug_assert!(off + 4 <= src.len());
        unsafe {
            let x = _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i);
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), x)
        }
    }
    #[inline(always)]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> __m256i {
        unsafe { _mm256_mulhrs_epi16(Self::load4_i16(src, off), _mm256_set1_epi16(0x5a80)) }
    }

    #[inline(always)]
    fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        unsafe {
            (
                _mm256_set1_epi32(rnd),
                _mm_cvtsi32_si128(shift),
                _mm256_set1_epi32(min),
                _mm256_set1_epi32(max),
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
            let v = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(acc, rnd), sh), minv),
                maxv,
            );
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            #[inline(always)]
            unsafe fn store_lane0(dst: &mut [i32], off: usize, v: __m128i) {
                unsafe { _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(v)) };
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
            let v = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(acc, rnd), sh), minv),
                maxv,
            );
            let lo = _mm256_castsi256_si128(v);
            #[inline(always)]
            unsafe fn store_lane0(dst: &mut [i32], off: usize, v: __m128i) {
                unsafe { _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(v)) };
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
            unsafe fn clip_lane<const HIGH: bool>(
                v: __m256i,
                rnd: __m256i,
                sh: __m128i,
                minv: __m256i,
                maxv: __m256i,
            ) -> __m128i {
                unsafe {
                    let v = _mm256_min_epi32(
                        _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(v, rnd), sh), minv),
                        maxv,
                    );
                    if HIGH {
                        _mm256_extracti128_si256::<1>(v)
                    } else {
                        _mm256_castsi256_si128(v)
                    }
                }
            }
            let (rnd, sh, minv, maxv) = clip;
            let c0 = clip_lane::<HIGH>(acc[0], rnd, sh, minv, maxv);
            let c1 = clip_lane::<HIGH>(acc[1], rnd, sh, minv, maxv);
            let c2 = clip_lane::<HIGH>(acc[2], rnd, sh, minv, maxv);
            let c3 = clip_lane::<HIGH>(acc[3], rnd, sh, minv, maxv);

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
    unsafe fn store8x8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 8],
        clip: Self::Clip,
    ) {
        debug_assert!(off + 7 + 7 * stride < dst.len());
        unsafe {
            #[inline(always)]
            unsafe fn clip_vec(
                v: __m256i,
                rnd: __m256i,
                sh: __m128i,
                minv: __m256i,
                maxv: __m256i,
            ) -> __m256i {
                unsafe {
                    _mm256_min_epi32(
                        _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(v, rnd), sh), minv),
                        maxv,
                    )
                }
            }

            #[inline(always)]
            unsafe fn store_row(dst: &mut [i32], off: usize, v: __m256i) {
                unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, v) };
            }

            let (rnd, sh, minv, maxv) = clip;
            let c0 = clip_vec(acc[0], rnd, sh, minv, maxv);
            let c1 = clip_vec(acc[1], rnd, sh, minv, maxv);
            let c2 = clip_vec(acc[2], rnd, sh, minv, maxv);
            let c3 = clip_vec(acc[3], rnd, sh, minv, maxv);
            let c4 = clip_vec(acc[4], rnd, sh, minv, maxv);
            let c5 = clip_vec(acc[5], rnd, sh, minv, maxv);
            let c6 = clip_vec(acc[6], rnd, sh, minv, maxv);
            let c7 = clip_vec(acc[7], rnd, sh, minv, maxv);

            // cN is one output column with lanes [r0cN..r7cN]. First build
            // 4-column row fragments in each 128-bit lane, then join low/high
            // halves from columns 0..3 and 4..7 into eight full rows.
            let t0 = _mm256_unpacklo_epi32(c0, c1);
            let t1 = _mm256_unpackhi_epi32(c0, c1);
            let t2 = _mm256_unpacklo_epi32(c2, c3);
            let t3 = _mm256_unpackhi_epi32(c2, c3);
            let t4 = _mm256_unpacklo_epi32(c4, c5);
            let t5 = _mm256_unpackhi_epi32(c4, c5);
            let t6 = _mm256_unpacklo_epi32(c6, c7);
            let t7 = _mm256_unpackhi_epi32(c6, c7);

            let a0 = _mm256_unpacklo_epi64(t0, t2);
            let a1 = _mm256_unpackhi_epi64(t0, t2);
            let a2 = _mm256_unpacklo_epi64(t1, t3);
            let a3 = _mm256_unpackhi_epi64(t1, t3);
            let a4 = _mm256_unpacklo_epi64(t4, t6);
            let a5 = _mm256_unpackhi_epi64(t4, t6);
            let a6 = _mm256_unpacklo_epi64(t5, t7);
            let a7 = _mm256_unpackhi_epi64(t5, t7);

            let r0 = _mm256_permute2x128_si256::<0x20>(a0, a4);
            let r1 = _mm256_permute2x128_si256::<0x20>(a1, a5);
            let r2 = _mm256_permute2x128_si256::<0x20>(a2, a6);
            let r3 = _mm256_permute2x128_si256::<0x20>(a3, a7);
            let r4 = _mm256_permute2x128_si256::<0x31>(a0, a4);
            let r5 = _mm256_permute2x128_si256::<0x31>(a1, a5);
            let r6 = _mm256_permute2x128_si256::<0x31>(a2, a6);
            let r7 = _mm256_permute2x128_si256::<0x31>(a3, a7);

            store_row(dst, off, r0);
            store_row(dst, off + stride, r1);
            store_row(dst, off + 2 * stride, r2);
            store_row(dst, off + 3 * stride, r3);
            store_row(dst, off + 4 * stride, r4);
            store_row(dst, off + 5 * stride, r5);
            store_row(dst, off + 6 * stride, r6);
            store_row(dst, off + 7 * stride, r7);
        }
    }

    #[inline(always)]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, acc) };
    }

    #[inline(always)]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            _mm_storeu_si128(
                dst.as_mut_ptr().add(off) as *mut __m128i,
                _mm256_castsi256_si128(acc),
            )
        };
    }
}

pub(crate) struct AvxDct2d;

impl DctSimd4 for AvxDct2d {
    type V = AvxI32x4;
    type Wide = AvxWide;

    #[inline(always)]
    unsafe fn zero() -> Self::V {
        AvxI32x4(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    unsafe fn splat(v: i32) -> Self::V {
        AvxI32x4(unsafe { _mm_set1_epi32(v) })
    }

    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        AvxI32x4(unsafe { _mm_add_epi32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        AvxI32x4(unsafe { _mm_sub_epi32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        AvxI32x4(unsafe { _mm_mullo_epi32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe {
            let scaled = _mm_add_epi32(
                _mm_mullo_epi32(a.0, _mm_set1_epi32(181)),
                _mm_set1_epi32(128),
            );
            AvxI32x4(_mm_srai_epi32::<8>(scaled))
        }
    }

    #[inline(always)]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_ptr().add(off) as *const __m128i };
        AvxI32x4(unsafe { _mm_loadu_si128(p) })
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
        AvxI32x4(unsafe { _mm_loadu_si128(p) })
    }

    #[inline(always)]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) as *const __m128i };
        AvxI32x4(unsafe { _mm_cvtepi16_epi32(_mm_loadl_epi64(p)) })
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
    crate::itx_2d::idct_dequant_simd4_core_avx2::<AvxDct2d, 16, 4, i32>(
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
    crate::itx_2d::idct_dequant_simd4_core_avx2::<AvxDct2d, 64, 8, i32>(
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
    crate::itx_2d::idct_dequant_simd4_core_avx2::<AvxDct2d, 256, 16, i32>(
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
    crate::itx_2d::idct_dequant_simd4_core_avx2::<AvxDct2d, 1024, 32, i32>(
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
    crate::itx_2d::idct_dequant_simd4_core_avx2::<AvxDct2d, 1024, 32, i32>(
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
    crate::itx_2d::itx_dequant_simd4_core_avx2::<AvxDct2d, 16, 4, i32>(
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
    crate::itx_2d::itx_dequant_simd4_core_avx2::<AvxDct2d, 64, 8, i32>(
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
    crate::itx_2d::itx_dequant_simd4_core_avx2::<AvxDct2d, 256, 16, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 32, 4, 8, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 32, 8, 4, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 128, 8, 16, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 128, 16, 8, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 512, 16, 32, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 512, 32, 16, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 64, 4, 16, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 64, 16, 4, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 256, 8, 32, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 256, 32, 8, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 128, 4, 32, i32>(
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
    crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, 128, 32, 4, i32>(
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
    crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, 32, 4, 8, i32>(
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
    crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, 32, 8, 4, i32>(
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
    crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, 128, 8, 16, i32>(
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
    crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, 128, 16, 8, i32>(
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
    crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, 64, 4, 16, i32>(
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
    crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, 64, 16, 4, i32>(
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
        #[target_feature(enable = "avx2")]
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
            idct_dequant_simd4_core_avx2::<AvxDct2d, { $n }, { $s }, i16>(
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
        #[target_feature(enable = "avx2")]
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
            itx_dequant_simd4_core_avx2::<AvxDct2d, { $n }, { $s }, i16>(
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
        #[target_feature(enable = "avx2")]
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
            crate::itx_2d::idct_dequant_rect_simd4_core_avx2::<AvxDct2d, { $n }, { $w }, { $h }, i16>(
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
        #[target_feature(enable = "avx2")]
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
            crate::itx_2d::itx_dequant_rect_simd4_core_avx2::<AvxDct2d, { $n }, { $w }, { $h }, i16>(
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
    idct_dequant_4x4_i16_avx2,
    idct_dequant_4x4_i16_avx2_impl,
    16,
    4
);
idct_i16_fn!(
    idct_dequant_8x8_i16_avx2,
    idct_dequant_8x8_i16_avx2_impl,
    64,
    8
);
idct_i16_fn!(
    idct_dequant_16x16_i16_avx2,
    idct_dequant_16x16_i16_avx2_impl,
    256,
    16
);
idct_i16_fn!(
    idct_dequant_32x32_i16_avx2,
    idct_dequant_32x32_i16_avx2_impl,
    1024,
    32
);
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
iadst_i16_fn!(
    iadst_dequant_8x8_i16_avx2,
    iadst_dequant_8x8_i16_avx2_impl,
    64,
    8
);
iadst_i16_fn!(
    iadst_dequant_16x16_i16_avx2,
    iadst_dequant_16x16_i16_avx2_impl,
    256,
    16
);
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
