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
fn load16_u16_i32x2(p: &[u16]) -> (__m256i, __m256i) {
    unsafe {
        let lo = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.as_ptr().cast()));
        let hi = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.as_ptr().add(8).cast()));
        (lo, hi)
    }
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
fn clip8_u16_i32(s: __m256i, bitdepth_max: i32) -> __m256i {
    _mm256_min_epi32(
        _mm256_max_epi32(
            _mm256_srai_epi32::<7>(_mm256_add_epi32(s, _mm256_set1_epi32(64))),
            _mm256_setzero_si256(),
        ),
        _mm256_set1_epi32(bitdepth_max),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn finish16_u16(dst: &mut [u16], slo: __m256i, shi: __m256i, bitdepth_max: i32) {
    let vlo = clip8_u16_i32(slo, bitdepth_max);
    let vhi = clip8_u16_i32(shi, bitdepth_max);
    let packed = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(vlo, vhi));
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), packed) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn gather8_u16_i32(row: &[u16], idx: usize, step: usize) -> __m256i {
    if step == 1 {
        load8_u16_i32(&row[idx..])
    } else if step == 2 && idx + 16 <= row.len() {
        unsafe {
            // Select u16 lanes [0,2,4,6] and [8,10,12,14] without a scalar
            // stack gather. pshufb is lane-local, so join the two 64-bit packs.
            let v = _mm256_loadu_si256(row.as_ptr().add(idx).cast());
            let mask = _mm256_setr_epi8(
                0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13,
                -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let packed = _mm256_shuffle_epi8(v, mask);
            let lo = _mm256_castsi256_si128(packed);
            let hi = _mm256_extracti128_si256::<1>(packed);
            let even = _mm_unpacklo_epi64(lo, hi);
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
        unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(arr.as_ptr().cast())) }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gather16_u16_i32x2(row: &[u16], idx: usize, step: usize) -> (__m256i, __m256i) {
    if step == 1 {
        load16_u16_i32x2(&row[idx..])
    } else {
        (
            gather8_u16_i32(row, idx, step),
            gather8_u16_i32(row, idx + 8 * step, step),
        )
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
    while x + 16 <= n {
        let c = col0 + x;
        debug_assert!(c + 16 <= center.len());
        let (mlo, mhi) = load16_u16_i32x2(&center[c..]);
        let mut slo = _mm256_slli_epi32::<7>(mlo);
        let mut shi = _mm256_slli_epi32::<7>(mhi);
        let two_mlo = _mm256_add_epi32(mlo, mlo);
        let two_mhi = _mm256_add_epi32(mhi, mhi);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
            let (alo, ahi) = load16_u16_i32x2(&t.row_p[cp..]);
            let (blo, bhi) = load16_u16_i32x2(&t.row_m[cm..]);
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
        finish16_u16(&mut dst[x..], slo, shi, bitdepth_max);
        x += 16;
    }
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
    while x + 16 <= n {
        let c = col0 + x;
        debug_assert!(c + 16 <= center.len());
        let (mlo, mhi) = load16_u16_i32x2(&center[c..]);
        let cc = _mm256_set1_epi32(center_coef);
        let mut slo = _mm256_mullo_epi32(mlo, cc);
        let mut shi = _mm256_mullo_epi32(mhi, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
            let (alo, ahi) = load16_u16_i32x2(&t.row_p[cp..]);
            let (blo, bhi) = load16_u16_i32x2(&t.row_m[cm..]);
            let coef = _mm256_set1_epi32(t.coef);
            slo = _mm256_add_epi32(slo, _mm256_mullo_epi32(_mm256_add_epi32(alo, blo), coef));
            shi = _mm256_add_epi32(shi, _mm256_mullo_epi32(_mm256_add_epi32(ahi, bhi), coef));
        }
        finish16_u16(&mut dst[x..], slo, shi, bitdepth_max);
        x += 16;
    }
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
    while x + 16 <= n {
        let cb = co + x;
        let (mlo, mhi) = load16_u16_i32x2(&c_center[cb..]);
        let two_mlo = _mm256_add_epi32(mlo, mlo);
        let two_mhi = _mm256_add_epi32(mhi, mhi);
        let mut slo = _mm256_slli_epi32::<7>(mlo);
        let mut shi = _mm256_slli_epi32::<7>(mhi);
        for t in ctaps {
            let cp = (cb as i32 + t.dx) as usize;
            let cm = (cb as i32 - t.dx) as usize;
            let (alo, ahi) = load16_u16_i32x2(&t.row_p[cp..]);
            let (blo, bhi) = load16_u16_i32x2(&t.row_m[cm..]);
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
        let (lclo, lchi) = gather16_u16_i32x2(l_center, lb, lstep);
        for t in ltaps {
            let li = (lb as i32 + t.ldx) as usize;
            let (lvlo, lvhi) = gather16_u16_i32x2(t.row, li, lstep);
            let coef = _mm256_set1_epi32(t.coef);
            slo = _mm256_add_epi32(slo, _mm256_mullo_epi32(_mm256_sub_epi32(lvlo, lclo), coef));
            shi = _mm256_add_epi32(shi, _mm256_mullo_epi32(_mm256_sub_epi32(lvhi, lchi), coef));
        }
        finish16_u16(&mut dst[x..], slo, shi, bitdepth_max);
        x += 16;
    }
    while x + 8 <= n {
        let cb = co + x;
        let m = load8_u16_i32(&c_center[cb..]);
        let two_m = _mm256_add_epi32(m, m);
        let mut s = _mm256_slli_epi32::<7>(m);
        for t in ctaps {
            let a = load8_u16_i32(&t.row_p[(cb as i32 + t.dx) as usize..]);
            let b = load8_u16_i32(&t.row_m[(cb as i32 - t.dx) as usize..]);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(
                s,
                _mm256_mullo_epi32(_mm256_sub_epi32(_mm256_add_epi32(a, b), two_m), coef),
            );
        }
        let lb = lo + x * lstep;
        let lc = gather8_u16_i32(l_center, lb, lstep);
        for t in ltaps {
            let lv = gather8_u16_i32(t.row, (lb as i32 + t.ldx) as usize, lstep);
            let coef = _mm256_set1_epi32(t.coef);
            s = _mm256_add_epi32(s, _mm256_mullo_epi32(_mm256_sub_epi32(lv, lc), coef));
        }
        finish8_u16(&mut dst[x..], s, bitdepth_max);
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
        dst[x] = ((s + 64) >> 7).clamp(0, bitdepth_max) as u16;
        x += 1;
    }
}
