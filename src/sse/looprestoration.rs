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

use crate::filter::WienerTap;

/// Load 8 consecutive `u8` and zero-extend into two `__m128i` (4×i32) halves.
#[inline(always)]
unsafe fn load8(p: *const u8) -> (__m128i, __m128i) {
    unsafe {
        let v = _mm_loadl_epi64(p as *const __m128i); // 8 bytes in low 64 bits
        let lo = _mm_cvtepu8_epi32(v); // bytes 0..3 -> i32x4
        let hi = _mm_cvtepu8_epi32(_mm_srli_si128(v, 4)); // bytes 4..7 -> i32x4
        (lo, hi)
    }
}

/// `(s + 64) >> 7`, clamped to `[0, 255]`, narrowed to 8 packed `u8`, stored.
#[inline(always)]
unsafe fn finish(dst: *mut u8, slo: __m128i, shi: __m128i) {
    unsafe {
        let rnd = _mm_set1_epi32(64);
        let zero = _mm_setzero_si128();
        let max = _mm_set1_epi32(255);
        // _mm_srai_epi32 is an arithmetic shift, matching `>> 7` on i32.
        let vlo = _mm_min_epi32(
            _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(slo, rnd), 7), zero),
            max,
        );
        let vhi = _mm_min_epi32(
            _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(shi, rnd), 7), zero),
            max,
        );
        // Values are in [0, 255], so the saturating packs are exact.
        let packed16 = _mm_packus_epi32(vlo, vhi); // 8 x u16
        let packed8 = _mm_packus_epi16(packed16, packed16); // low 8 bytes = result
        _mm_storel_epi64(dst as *mut __m128i, packed8);
    }
}

#[target_feature(enable = "sse4.1")]
fn ns_wiener_fir_run_sse41_impl(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    unsafe {
        let mut x = 0;
        while x + 8 <= n {
            let c = col0 + x;
            debug_assert!(c + 8 <= center.len());
            let (mlo, mhi) = load8(center[c..c + 8].as_ptr());
            let mut slo = _mm_slli_epi32(mlo, 7);
            let mut shi = _mm_slli_epi32(mhi, 7);
            let two_mlo = _mm_add_epi32(mlo, mlo);
            let two_mhi = _mm_add_epi32(mhi, mhi);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8(t.row_p[cp..cp + 8].as_ptr());
                let (blo, bhi) = load8(t.row_m[cm..cm + 8].as_ptr());
                let coef = _mm_set1_epi32(t.coef);
                // (a + b - 2*m) * coef
                slo = _mm_add_epi32(
                    slo,
                    _mm_mullo_epi32(_mm_sub_epi32(_mm_add_epi32(alo, blo), two_mlo), coef),
                );
                shi = _mm_add_epi32(
                    shi,
                    _mm_mullo_epi32(_mm_sub_epi32(_mm_add_epi32(ahi, bhi), two_mhi), coef),
                );
            }
            finish(dst[x..x + 8].as_mut_ptr(), slo, shi);
            x += 8;
        }
        while x < n {
            let c = col0 + x;
            let m = center[c] as i32;
            let mut s = m << 7;
            for t in taps {
                let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
                let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
                s += (a + b - 2 * m) * t.coef;
            }
            dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
            x += 1;
        }
    }
}

#[target_feature(enable = "sse4.1")]
fn pc_wiener_fir_run_sse41_impl(
    dst: &mut [u8],
    center: &[u8],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    unsafe {
        let mut x = 0;
        while x + 8 <= n {
            let c = col0 + x;
            debug_assert!(c + 8 <= center.len());
            let (mlo, mhi) = load8(center[c..c + 8].as_ptr());
            let cc = _mm_set1_epi32(center_coef);
            let mut slo = _mm_mullo_epi32(mlo, cc);
            let mut shi = _mm_mullo_epi32(mhi, cc);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8(t.row_p[cp..cp + 8].as_ptr());
                let (blo, bhi) = load8(t.row_m[cm..cm + 8].as_ptr());
                let coef = _mm_set1_epi32(t.coef);
                // (a + b) * coef
                slo = _mm_add_epi32(slo, _mm_mullo_epi32(_mm_add_epi32(alo, blo), coef));
                shi = _mm_add_epi32(shi, _mm_mullo_epi32(_mm_add_epi32(ahi, bhi), coef));
            }
            finish(dst[x..x + 8].as_mut_ptr(), slo, shi);
            x += 8;
        }
        while x < n {
            let c = col0 + x;
            let m = center[c] as i32;
            let mut s = m * center_coef;
            for t in taps {
                let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
                let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
                s += (a + b) * t.coef;
            }
            dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
            x += 1;
        }
    }
}

/// Safe entry point. Only assigned in the dispatcher under an
/// `is_x86_feature_detected!("sse4.1")` guard, so the feature is present.
pub(crate) fn ns_wiener_fir_run_sse41(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    unsafe { ns_wiener_fir_run_sse41_impl(dst, center, col0, taps, n) }
}

/// Safe entry point. See [`ns_wiener_fir_run_sse41`].
pub(crate) fn pc_wiener_fir_run_sse41(
    dst: &mut [u8],
    center: &[u8],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    unsafe { pc_wiener_fir_run_sse41_impl(dst, center, center_coef, col0, taps, n) }
}
