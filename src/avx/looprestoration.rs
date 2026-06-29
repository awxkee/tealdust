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

use crate::filter::{UvLumaTap, WienerTap};

// Precision note: AVM/dav2d LR FIRs accumulate in signed int and only round at
// the final `+64 >> 7` stage.  Keep the SIMD products and sums in i32 here;
// using 16-bit accumulators would overflow for HBD/PC-Wiener ranges.

#[inline]
#[target_feature(enable = "avx2")]
fn load8_u8_i32(p: &[u8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(p.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load16_u8_i32x2(p: &[u8]) -> (__m256i, __m256i) {
    unsafe {
        let v = _mm_loadu_si128(p.as_ptr().cast());
        let lo = _mm256_cvtepu8_epi32(v);
        let hi = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(v));
        (lo, hi)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gather8_u8_i32(row: &[u8], idx: usize, step: usize) -> __m256i {
    if step == 1 {
        load8_u8_i32(&row[idx..])
    } else if step == 2 && idx + 16 <= row.len() {
        // AV2 chroma LR samples the luma plane with lstep=2 for 4:2:x.
        // Avoid the old scalar gather-through-stack: bytes [0,2,..14] are the
        // low bytes of eight little-endian u16 lanes.
        unsafe {
            let v = _mm_loadu_si128(row.as_ptr().add(idx).cast());
            let even = _mm_and_si128(v, _mm_set1_epi16(0x00ff));
            _mm256_cvtepu16_epi32(even)
        }
    } else {
        let arr = [
            row[idx],
            row[idx + step],
            row[idx + 2 * step],
            row[idx + 3 * step],
            row[idx + 4 * step],
            row[idx + 5 * step],
            row[idx + 6 * step],
            row[idx + 7 * step],
        ];
        unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(arr.as_ptr().cast())) }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gather16_u8_i32x2(row: &[u8], idx: usize, step: usize) -> (__m256i, __m256i) {
    if step == 1 {
        load16_u8_i32x2(&row[idx..])
    } else if step == 2 && idx + 32 <= row.len() {
        unsafe {
            let v = _mm256_loadu_si256(row.as_ptr().add(idx).cast());
            let even = _mm256_and_si256(v, _mm256_set1_epi16(0x00ff));
            let lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(even));
            let hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(even));
            (lo, hi)
        }
    } else {
        (
            gather8_u8_i32(row, idx, step),
            gather8_u8_i32(row, idx + 8 * step, step),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gather32_u8_i32x4(row: &[u8], idx: usize, step: usize) -> (__m256i, __m256i, __m256i, __m256i) {
    let (a0, a1) = gather16_u8_i32x2(row, idx, step);
    let (a2, a3) = gather16_u8_i32x2(row, idx + 16 * step, step);
    (a0, a1, a2, a3)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clip8_i32(v: __m256i) -> __m256i {
    _mm256_min_epi32(
        _mm256_max_epi32(
            _mm256_srai_epi32::<7>(_mm256_add_epi32(v, _mm256_set1_epi32(64))),
            _mm256_setzero_si256(),
        ),
        _mm256_set1_epi32(255),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn finish8(dst: &mut [u8], s: __m256i) {
    let v = clip8_i32(s);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let u16x8 = _mm_packus_epi32(lo, hi);
    let u8x16 = _mm_packus_epi16(u16x8, u16x8);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast(), u8x16) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn finish16(dst: &mut [u8], slo: __m256i, shi: __m256i) {
    let vlo = clip8_i32(slo);
    let vhi = clip8_i32(shi);
    // packus_epi32 is lane-local: [lo0..3 hi0..3 | lo4..7 hi4..7].
    // Swap 64-bit lanes 1 and 2 so the subsequent 128-bit pack sees pixels
    // in strictly increasing order.
    let u16x16 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(vlo, vhi));
    let lo16 = _mm256_castsi256_si128(u16x16);
    let hi16 = _mm256_extracti128_si256::<1>(u16x16);
    let u8x16 = _mm_packus_epi16(lo16, hi16);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), u8x16) };
}

#[target_feature(enable = "avx2")]
pub(crate) fn ns_wiener_fir_run_avx2(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    let mut x = 0usize;
    while x + 32 <= n {
        let c = col0 + x;
        debug_assert!(c + 32 <= center.len());
        let (m0, m1) = load16_u8_i32x2(&center[c..]);
        let (m2, m3) = load16_u8_i32x2(&center[c + 16..]);
        let mut s0 = _mm256_slli_epi32::<7>(m0);
        let mut s1 = _mm256_slli_epi32::<7>(m1);
        let mut s2 = _mm256_slli_epi32::<7>(m2);
        let mut s3 = _mm256_slli_epi32::<7>(m3);
        let two_m0 = _mm256_add_epi32(m0, m0);
        let two_m1 = _mm256_add_epi32(m1, m1);
        let two_m2 = _mm256_add_epi32(m2, m2);
        let two_m3 = _mm256_add_epi32(m3, m3);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 32 <= t.row_p.len() && cm + 32 <= t.row_m.len());
            let coef = _mm256_set1_epi32(t.coef);
            let (a0, a1) = load16_u8_i32x2(&t.row_p[cp..]);
            let (b0, b1) = load16_u8_i32x2(&t.row_m[cm..]);
            s0 = _mm256_add_epi32(
                s0,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a0, b0), two_m0), coef),
            );
            s1 = _mm256_add_epi32(
                s1,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a1, b1), two_m1), coef),
            );
            let (a2, a3) = load16_u8_i32x2(&t.row_p[cp + 16..]);
            let (b2, b3) = load16_u8_i32x2(&t.row_m[cm + 16..]);
            s2 = _mm256_add_epi32(
                s2,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a2, b2), two_m2), coef),
            );
            s3 = _mm256_add_epi32(
                s3,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a3, b3), two_m3), coef),
            );
        }
        finish16(&mut dst[x..], s0, s1);
        finish16(&mut dst[x + 16..], s2, s3);
        x += 32;
    }
    while x + 16 <= n {
        let c = col0 + x;
        debug_assert!(c + 16 <= center.len());
        let (mlo, mhi) = load16_u8_i32x2(&center[c..]);
        let mut slo = _mm256_slli_epi32::<7>(mlo);
        let mut shi = _mm256_slli_epi32::<7>(mhi);
        let two_mlo = _mm256_add_epi32(mlo, mlo);
        let two_mhi = _mm256_add_epi32(mhi, mhi);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
            let (alo, ahi) = load16_u8_i32x2(&t.row_p[cp..]);
            let (blo, bhi) = load16_u8_i32x2(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            slo = _mm256_add_epi32(
                slo,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(alo, blo), two_mlo), coef),
            );
            shi = _mm256_add_epi32(
                shi,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(ahi, bhi), two_mhi), coef),
            );
        }
        finish16(&mut dst[x..], slo, shi);
        x += 16;
    }
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let m = load8_u8_i32(&center[c..]);
        let mut s = _mm256_slli_epi32::<7>(m);
        let two_m = _mm256_add_epi32(m, m);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let a = load8_u8_i32(&t.row_p[cp..]);
            let b = load8_u8_i32(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(
                s,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a, b), two_m), coef),
            );
        }
        finish8(&mut dst[x..], s);
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
    let mut x = 0usize;
    while x + 32 <= n {
        let c = col0 + x;
        debug_assert!(c + 32 <= center.len());
        let (m0, m1) = load16_u8_i32x2(&center[c..]);
        let (m2, m3) = load16_u8_i32x2(&center[c + 16..]);
        let cc = _mm256_set1_epi32(center_coef);
        let mut s0 = _mm256_mullo_epi32(m0, cc);
        let mut s1 = _mm256_mullo_epi32(m1, cc);
        let mut s2 = _mm256_mullo_epi32(m2, cc);
        let mut s3 = _mm256_mullo_epi32(m3, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 32 <= t.row_p.len() && cm + 32 <= t.row_m.len());
            let coef = _mm256_set1_epi32(t.coef);
            let (a0, a1) = load16_u8_i32x2(&t.row_p[cp..]);
            let (b0, b1) = load16_u8_i32x2(&t.row_m[cm..]);
            s0 = _mm256_add_epi32(s0, _mm256_mullo_epi32(_mm256_add_epi32(a0, b0), coef));
            s1 = _mm256_add_epi32(s1, _mm256_mullo_epi32(_mm256_add_epi32(a1, b1), coef));
            let (a2, a3) = load16_u8_i32x2(&t.row_p[cp + 16..]);
            let (b2, b3) = load16_u8_i32x2(&t.row_m[cm + 16..]);
            s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(_mm256_add_epi32(a2, b2), coef));
            s3 = _mm256_add_epi32(s3, _mm256_mullo_epi32(_mm256_add_epi32(a3, b3), coef));
        }
        finish16(&mut dst[x..], s0, s1);
        finish16(&mut dst[x + 16..], s2, s3);
        x += 32;
    }
    while x + 16 <= n {
        let c = col0 + x;
        debug_assert!(c + 16 <= center.len());
        let (mlo, mhi) = load16_u8_i32x2(&center[c..]);
        let cc = _mm256_set1_epi32(center_coef);
        let mut slo = _mm256_mullo_epi32(mlo, cc);
        let mut shi = _mm256_mullo_epi32(mhi, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
            let (alo, ahi) = load16_u8_i32x2(&t.row_p[cp..]);
            let (blo, bhi) = load16_u8_i32x2(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            slo = _mm256_add_epi32(slo, _mm256_mullo_epi32(_mm256_add_epi32(alo, blo), coef));
            shi = _mm256_add_epi32(shi, _mm256_mullo_epi32(_mm256_add_epi32(ahi, bhi), coef));
        }
        finish16(&mut dst[x..], slo, shi);
        x += 16;
    }
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let m = load8_u8_i32(&center[c..]);
        let cc = _mm256_set1_epi32(center_coef);
        let mut s = _mm256_mullo_epi32(m, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let a = load8_u8_i32(&t.row_p[cp..]);
            let b = load8_u8_i32(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(s, _mm256_mullo_epi32(_mm256_add_epi32(a, b), coef));
        }
        finish8(&mut dst[x..], s);
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

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
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
    let mut x = 0usize;
    while x + 32 <= n {
        let cb = co + x;
        let (m0, m1) = load16_u8_i32x2(&c_center[cb..]);
        let (m2, m3) = load16_u8_i32x2(&c_center[cb + 16..]);
        let two_m0 = _mm256_add_epi32(m0, m0);
        let two_m1 = _mm256_add_epi32(m1, m1);
        let two_m2 = _mm256_add_epi32(m2, m2);
        let two_m3 = _mm256_add_epi32(m3, m3);
        let mut s0 = _mm256_slli_epi32::<7>(m0);
        let mut s1 = _mm256_slli_epi32::<7>(m1);
        let mut s2 = _mm256_slli_epi32::<7>(m2);
        let mut s3 = _mm256_slli_epi32::<7>(m3);
        for t in ctaps {
            let cp = (cb as i32 + t.dx) as usize;
            let cm = (cb as i32 - t.dx) as usize;
            let (a0, a1) = load16_u8_i32x2(&t.row_p[cp..]);
            let (b0, b1) = load16_u8_i32x2(&t.row_m[cm..]);
            let (a2, a3) = load16_u8_i32x2(&t.row_p[cp + 16..]);
            let (b2, b3) = load16_u8_i32x2(&t.row_m[cm + 16..]);
            let coef = _mm256_set1_epi32(t.coef);
            s0 = _mm256_add_epi32(
                s0,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a0, b0), two_m0), coef),
            );
            s1 = _mm256_add_epi32(
                s1,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a1, b1), two_m1), coef),
            );
            s2 = _mm256_add_epi32(
                s2,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a2, b2), two_m2), coef),
            );
            s3 = _mm256_add_epi32(
                s3,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a3, b3), two_m3), coef),
            );
        }
        let lb = lo + x * lstep;
        let (lc0, lc1, lc2, lc3) = gather32_u8_i32x4(l_center, lb, lstep);
        for t in ltaps {
            let li = (lb as i32 + t.ldx) as usize;
            let (lv0, lv1, lv2, lv3) = gather32_u8_i32x4(t.row, li, lstep);
            let coef = _mm256_set1_epi32(t.coef);
            s0 = _mm256_add_epi32(s0, _mm256_mullo_epi32(_mm256_sub_epi32(lv0, lc0), coef));
            s1 = _mm256_add_epi32(s1, _mm256_mullo_epi32(_mm256_sub_epi32(lv1, lc1), coef));
            s2 = _mm256_add_epi32(s2, _mm256_mullo_epi32(_mm256_sub_epi32(lv2, lc2), coef));
            s3 = _mm256_add_epi32(s3, _mm256_mullo_epi32(_mm256_sub_epi32(lv3, lc3), coef));
        }
        finish16(&mut dst[x..], s0, s1);
        finish16(&mut dst[x + 16..], s2, s3);
        x += 32;
    }
    while x + 16 <= n {
        let cb = co + x;
        let (mlo, mhi) = load16_u8_i32x2(&c_center[cb..]);
        let two_mlo = _mm256_add_epi32(mlo, mlo);
        let two_mhi = _mm256_add_epi32(mhi, mhi);
        let mut slo = _mm256_slli_epi32::<7>(mlo);
        let mut shi = _mm256_slli_epi32::<7>(mhi);
        for t in ctaps {
            let cp = (cb as i32 + t.dx) as usize;
            let cm = (cb as i32 - t.dx) as usize;
            let (alo, ahi) = load16_u8_i32x2(&t.row_p[cp..]);
            let (blo, bhi) = load16_u8_i32x2(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            slo = _mm256_add_epi32(
                slo,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(alo, blo), two_mlo), coef),
            );
            shi = _mm256_add_epi32(
                shi,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(ahi, bhi), two_mhi), coef),
            );
        }
        let lb = lo + x * lstep;
        let (lclo, lchi) = gather16_u8_i32x2(l_center, lb, lstep);
        for t in ltaps {
            let li = (lb as i32 + t.ldx) as usize;
            let (lvlo, lvhi) = gather16_u8_i32x2(t.row, li, lstep);
            let coef = _mm256_set1_epi32(t.coef);
            slo = _mm256_add_epi32(slo, _mm256_mullo_epi32(_mm256_sub_epi32(lvlo, lclo), coef));
            shi = _mm256_add_epi32(shi, _mm256_mullo_epi32(_mm256_sub_epi32(lvhi, lchi), coef));
        }
        finish16(&mut dst[x..], slo, shi);
        x += 16;
    }
    while x + 8 <= n {
        let cb = co + x;
        let m = load8_u8_i32(&c_center[cb..]);
        let two_m = _mm256_add_epi32(m, m);
        let mut s = _mm256_slli_epi32::<7>(m);
        for t in ctaps {
            let a = load8_u8_i32(&t.row_p[(cb as i32 + t.dx) as usize..]);
            let b = load8_u8_i32(&t.row_m[(cb as i32 - t.dx) as usize..]);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(
                s,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a, b), two_m), coef),
            );
        }
        let lb = lo + x * lstep;
        let lc = gather8_u8_i32(l_center, lb, lstep);
        for t in ltaps {
            let lv = gather8_u8_i32(t.row, (lb as i32 + t.ldx) as usize, lstep);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(s, _mm256_mullo_epi32(_mm256_sub_epi32(lv, lc), coef));
        }
        finish8(&mut dst[x..], s);
        x += 8;
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

#[cfg(test)]
mod uv_fir_avx2_tests {
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
    fn ns_wiener_uv_fir_avx2_matches_scalar() {
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
            let n = rng.range(1, 48) as usize;

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
