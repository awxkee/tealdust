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

use crate::filter::{UvLumaTapHbd, WienerTapHbd};

#[inline]
#[target_feature(enable = "avx2")]
fn load8_u16_i32(p: &[u16]) -> __m256i {
    unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(p.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn finish8_u16(dst: &mut [u16], s: __m256i, bitdepth_max: i32) {
    let rnd = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi32(bitdepth_max);
    let v = _mm256_min_epi32(
        _mm256_max_epi32(_mm256_srai_epi32::<7>(_mm256_add_epi32(s, rnd)), zero),
        max,
    );
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let packed = _mm_packus_epi32(lo, hi);
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), packed);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load4_u16_i32(row: &[u16], idx: usize) -> __m128i {
    unsafe { _mm_cvtepu16_epi32(_mm_loadl_epi64(row[idx..].as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gather4_u16_i32(row: &[u16], idx: usize, step: usize) -> __m128i {
    if step == 1 {
        load4_u16_i32(row, idx)
    } else {
        let arr = [
            row[idx],
            row[idx + step],
            row[idx + 2 * step],
            row[idx + 3 * step],
        ];
        unsafe { _mm_cvtepu16_epi32(_mm_loadl_epi64(arr.as_ptr() as *const __m128i)) }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn finish4_u16(dst: &mut [u16], x: usize, s: __m128i, bitdepth_max: i32) {
    let v = _mm_min_epi32(
        _mm_max_epi32(
            _mm_srai_epi32(_mm_add_epi32(s, _mm_set1_epi32(64)), 7),
            _mm_setzero_si128(),
        ),
        _mm_set1_epi32(bitdepth_max),
    );
    let packed = _mm_packus_epi32(v, v);
    unsafe {
        _mm_storel_epi64(dst[x..].as_mut_ptr().cast(), packed);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ns_wiener_fir_run_hbd_avx2(
    dst: &mut [u16],
    center: &[u16],
    col0: usize,
    taps: &[WienerTapHbd],
    n: usize,
    bitdepth_max: i32,
) {
    let mut x = 0usize;
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let m = load8_u16_i32(&center[c..]);
        let mut s = _mm256_slli_epi32::<7>(m);
        let two_m = _mm256_add_epi32(m, m);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let a = load8_u16_i32(&t.row_p[cp..]);
            let b = load8_u16_i32(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(
                s,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a, b), two_m), coef),
            );
        }
        finish8_u16(&mut dst[x..], s, bitdepth_max);
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
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
        x += 1;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn pc_wiener_fir_run_hbd_avx2(
    dst: &mut [u16],
    center: &[u16],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTapHbd],
    n: usize,
    bitdepth_max: i32,
) {
    let mut x = 0usize;
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let m = load8_u16_i32(&center[c..]);
        let cc = _mm256_set1_epi32(center_coef);
        let mut s = _mm256_mullo_epi32(m, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let a = load8_u16_i32(&t.row_p[cp..]);
            let b = load8_u16_i32(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(s, _mm256_mullo_epi32(_mm256_add_epi32(a, b), coef));
        }
        finish8_u16(&mut dst[x..], s, bitdepth_max);
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
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
        x += 1;
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ns_wiener_uv_fir_run_hbd_avx2(
    dst: &mut [u16],
    c_center: &[u16],
    co: usize,
    ctaps: &[WienerTapHbd],
    l_center: &[u16],
    lo: usize,
    ltaps: &[UvLumaTapHbd],
    lstep: usize,
    n: usize,
    bitdepth_max: i32,
) {
    let mut x = 0usize;
    while x + 4 <= n {
        let cb = co + x;
        let m = load4_u16_i32(c_center, cb);
        let two_m = _mm_add_epi32(m, m);
        let mut s = _mm_slli_epi32(m, 7);
        for t in ctaps {
            let a = load4_u16_i32(t.row_p, (cb as i32 + t.dx) as usize);
            let b = load4_u16_i32(t.row_m, (cb as i32 - t.dx) as usize);
            let coef = _mm_set1_epi32(t.coef);
            s = _mm_add_epi32(
                s,
                _mm_mullo_epi32(_mm_sub_epi32(_mm_add_epi32(a, b), two_m), coef),
            );
        }
        let lb = lo + x * lstep;
        let lc = gather4_u16_i32(l_center, lb, lstep);
        for t in ltaps {
            let lv = gather4_u16_i32(t.row, (lb as i32 + t.ldx) as usize, lstep);
            let coef = _mm_set1_epi32(t.coef);
            s = _mm_add_epi32(s, _mm_mullo_epi32(_mm_sub_epi32(lv, lc), coef));
        }
        finish4_u16(dst, x, s, bitdepth_max);
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
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
        x += 1;
    }
}
