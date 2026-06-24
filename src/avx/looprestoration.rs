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
fn load8(p: &[u8]) -> (__m128i, __m128i) {
    unsafe {
        let v = _mm_loadl_epi64(p.as_ptr().cast());
        let lo = _mm_cvtepu8_epi32(v);
        let hi = _mm_cvtepu8_epi32(_mm_srli_si128(v, 4));
        (lo, hi)
    }
}

/// `(s + 64) >> 7`, clamped to `[0, 255]`, narrowed to 8 packed `u8`, stored.
#[inline(always)]
fn finish(dst: &mut [u8], slo: __m128i, shi: __m128i) {
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
        _mm_storel_epi64(dst.as_mut_ptr().cast(), packed8);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ns_wiener_fir_run_avx2(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    let mut x = 0;
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let (mlo, mhi) = load8(&center[c..]);
        let mut slo = _mm_slli_epi32(mlo, 7);
        let mut shi = _mm_slli_epi32(mhi, 7);
        let two_mlo = _mm_add_epi32(mlo, mlo);
        let two_mhi = _mm_add_epi32(mhi, mhi);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let (alo, ahi) = load8(&t.row_p[cp..]);
            let (blo, bhi) = load8(&t.row_m[cm..]);
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
        finish(&mut dst[x..], slo, shi);
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

#[target_feature(enable = "avx2")]
pub(crate) fn pc_wiener_fir_run_avx2(
    dst: &mut [u8],
    center: &[u8],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    let mut x = 0;
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let (mlo, mhi) = load8(&center[c..]);
        let cc = _mm_set1_epi32(center_coef);
        let mut slo = _mm_mullo_epi32(mlo, cc);
        let mut shi = _mm_mullo_epi32(mhi, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let (alo, ahi) = load8(&t.row_p[cp..]);
            let (blo, bhi) = load8(&t.row_m[cm..]);
            let coef = _mm_set1_epi32(t.coef);
            // (a + b) * coef
            slo = _mm_add_epi32(slo, _mm_mullo_epi32(_mm_add_epi32(alo, blo), coef));
            shi = _mm_add_epi32(shi, _mm_mullo_epi32(_mm_add_epi32(ahi, bhi), coef));
        }
        finish(&mut dst[x..], slo, shi);
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

// Safe entry point. Only assigned in the dispatcher under an
// `is_x86_feature_detected!("avx2")` guard, so the feature is present.
use crate::filter::UvLumaTap;

/// 4 consecutive `u8` -> i32x4.
#[inline(always)]
unsafe fn load4u8(row: &[u8], idx: usize) -> __m128i {
    let arr: [u8; 4] = row[idx..idx + 4].try_into().unwrap();
    unsafe { _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(arr))) }
}

/// Gather 4 `u8` at `idx, idx+step, ..` -> i32x4 (contiguous fast path for step 1).
#[inline(always)]
unsafe fn gather4u8(row: &[u8], idx: usize, step: usize) -> __m128i {
    let arr: [u8; 4] = if step == 1 {
        row[idx..idx + 4].try_into().unwrap()
    } else {
        [
            row[idx],
            row[idx + step],
            row[idx + 2 * step],
            row[idx + 3 * step],
        ]
    };
    unsafe { _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes(arr))) }
}

/// `(s + 64) >> 7` clamped to `[0,255]`, narrowed to 4 `u8`, stored at `dst[x..x+4]`.
#[inline(always)]
unsafe fn finish4(dst: &mut [u8], x: usize, s: __m128i) {
    unsafe {
        let v = _mm_min_epi32(
            _mm_max_epi32(
                _mm_srai_epi32(_mm_add_epi32(s, _mm_set1_epi32(64)), 7),
                _mm_setzero_si128(),
            ),
            _mm_set1_epi32(255),
        );
        let p8 = _mm_packus_epi16(_mm_packus_epi32(v, v), _mm_packus_epi32(v, v));
        let bytes = (_mm_cvtsi128_si32(p8) as u32).to_le_bytes();
        dst[x..x + 4].copy_from_slice(&bytes);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ns_wiener_uv_fir_run_avx2(
    dst: &mut [u8],
    c_center: &[u8],
    co: usize,
    ctaps: &[WienerTap],
    l_center: &[u8],
    lo: usize,
    ltaps: &[UvLumaTap],
    lstep: usize,
    n: usize,
) {
    unsafe {
        let mut x = 0;
        while x + 4 <= n {
            let cb = co + x;
            let m = load4u8(c_center, cb);
            let two_m = _mm_add_epi32(m, m);
            let mut s = _mm_slli_epi32::<7>(m);
            for t in ctaps {
                let a = load4u8(t.row_p, (cb as i32 + t.dx) as usize);
                let b = load4u8(t.row_m, (cb as i32 - t.dx) as usize);
                let coef = _mm_set1_epi32(t.coef);
                s = _mm_add_epi32(
                    s,
                    _mm_mullo_epi32(_mm_sub_epi32(_mm_add_epi32(a, b), two_m), coef),
                );
            }
            let lb = lo + x * lstep;
            let lc = gather4u8(l_center, lb, lstep);
            for t in ltaps {
                let lv = gather4u8(t.row, (lb as i32 + t.ldx) as usize, lstep);
                let coef = _mm_set1_epi32(t.coef);
                s = _mm_add_epi32(s, _mm_mullo_epi32(_mm_sub_epi32(lv, lc), coef));
            }
            finish4(dst, x, s);
            x += 4;
        }
        while x < n {
            let cc = co + x;
            let m = c_center[cc] as i32;
            let mut s = m << 7;
            for t in ctaps {
                let a = t.row_p[(cc as i32 + t.dx) as usize] as i32;
                let b = t.row_m[(cc as i32 - t.dx) as usize] as i32;
                s += (a + b - 2 * m) * t.coef;
            }
            let lcx = lo + x * lstep;
            let lc = l_center[lcx] as i32;
            for t in ltaps {
                let lv = t.row[(lcx as i32 + t.ldx) as usize] as i32;
                s += (lv - lc) * t.coef;
            }
            dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
            x += 1;
        }
    }
}

/// Safe entry point. See [`ns_wiener_fir_run_avx2`].

#[cfg(test)]
mod uv_fir_sse_tests {
    use crate::filter::{UvLumaTap, WienerTap, ns_wiener_uv_fir_run_scalar};

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    #[test]
    fn ns_wiener_uv_fir_sse_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        const CW: usize = 96;
        const LW: usize = 160;
        let mut rng = R(0x243f6a8885a308d3);
        for _ in 0..40_000 {
            let c_rows: Vec<Vec<u8>> = (0..5)
                .map(|_| (0..CW).map(|_| rng.range(0, 255) as u8).collect())
                .collect();
            let l_rows: Vec<Vec<u8>> = (0..5)
                .map(|_| (0..LW).map(|_| rng.range(0, 255) as u8).collect())
                .collect();

            let lstep = if rng.range(0, 1) == 0 { 1usize } else { 2 };
            let ctaps: Vec<WienerTap> = (0..6)
                .map(|_| WienerTap {
                    row_p: &c_rows[rng.range(0, 4) as usize],
                    row_m: &c_rows[rng.range(0, 4) as usize],
                    dx: rng.range(-2, 2),
                    coef: rng.range(-128, 127),
                })
                .collect();
            let ltaps: Vec<UvLumaTap> = (0..12)
                .map(|_| UvLumaTap {
                    row: &l_rows[rng.range(0, 4) as usize],
                    ldx: rng.range(-2, 2) * lstep as i32,
                    coef: rng.range(-128, 127),
                })
                .collect();

            let co = 8usize;
            let lo = 8usize;
            let n = (rng.range(1, 4) as usize) * 4; // 4,8,12,16

            let mut a = vec![0u8; n];
            let mut b = vec![0u8; n];
            ns_wiener_uv_fir_run_scalar(
                &mut a, &c_rows[2], co, &ctaps, &l_rows[2], lo, &ltaps, lstep, n,
            );
            unsafe {
                super::ns_wiener_uv_fir_run_avx2(
                    &mut b, &c_rows[2], co, &ctaps, &l_rows[2], lo, &ltaps, lstep, n,
                );
            }
            assert_eq!(a, b, "mismatch lstep={lstep} n={n}");
        }
    }
}
