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

use crate::filter::WienerTap;
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn load8_u8_i32(p: *const u8) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1_u8(p) }; // 8 x u8
    let w = vmovl_u8(v); // 8 x u16
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w)));
    (lo, hi)
}

/// `(s + 64) >> 7`, clamped to `[0, 255]`, then narrow two `int32x4_t` halves
/// to 8 packed `u8` and store at `dst`.
#[inline]
#[target_feature(enable = "neon")]
fn finish_store(dst: *mut u8, slo: int32x4_t, shi: int32x4_t) {
    let rnd = vdupq_n_s32(64);
    let zero = vdupq_n_s32(0);
    let max = vdupq_n_s32(255);
    // (s + 64) >> 7  (arithmetic shift, matching `sra` on i32)
    let vlo = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(slo, rnd)), zero), max);
    let vhi = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(shi, rnd)), zero), max);
    // Values are in [0, 255], so plain (non-saturating) narrowing is exact.
    let u16lo = vmovn_u32(vreinterpretq_u32_s32(vlo));
    let u16hi = vmovn_u32(vreinterpretq_u32_s32(vhi));
    let packed = vmovn_u16(vcombine_u16(u16lo, u16hi));
    unsafe { vst1_u8(dst, packed) };
}

/// NEON "NS" Wiener FIR. Mirrors `crate::simd::ns_wiener_fir_run_simd`.
pub(crate) fn ns_wiener_fir_run_neon(
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
        unsafe {
            let (mlo, mhi) = load8_u8_i32(center[c..c + 8].as_ptr());
            let mut slo = vshlq_n_s32::<7>(mlo);
            let mut shi = vshlq_n_s32::<7>(mhi);
            let two_mlo = vaddq_s32(mlo, mlo);
            let two_mhi = vaddq_s32(mhi, mhi);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8_u8_i32(t.row_p[cp..cp + 8].as_ptr());
                let (blo, bhi) = load8_u8_i32(t.row_m[cm..cm + 8].as_ptr());
                let coef = vdupq_n_s32(t.coef);
                // (a + b - 2*m) * coef
                slo = vaddq_s32(
                    slo,
                    vmulq_s32(vsubq_s32(vaddq_s32(alo, blo), two_mlo), coef),
                );
                shi = vaddq_s32(
                    shi,
                    vmulq_s32(vsubq_s32(vaddq_s32(ahi, bhi), two_mhi), coef),
                );
            }
            finish_store(dst[x..x + 8].as_mut_ptr(), slo, shi);
        }
        x += 8;
    }
    // Scalar tail (identical arithmetic to the vector body).
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

/// NEON "PC" Wiener FIR. Mirrors `crate::simd::pc_wiener_fir_run_simd`.
pub(crate) fn pc_wiener_fir_run_neon(
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
        unsafe {
            let (mlo, mhi) = load8_u8_i32(center[c..c + 8].as_ptr());
            let cc = vdupq_n_s32(center_coef);
            let mut slo = vmulq_s32(mlo, cc);
            let mut shi = vmulq_s32(mhi, cc);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8_u8_i32(t.row_p[cp..cp + 8].as_ptr());
                let (blo, bhi) = load8_u8_i32(t.row_m[cm..cm + 8].as_ptr());
                let coef = vdupq_n_s32(t.coef);
                // (a + b) * coef
                slo = vaddq_s32(slo, vmulq_s32(vaddq_s32(alo, blo), coef));
                shi = vaddq_s32(shi, vmulq_s32(vaddq_s32(ahi, bhi), coef));
            }
            finish_store(dst[x..x + 8].as_mut_ptr(), slo, shi);
        }
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

use crate::filter::UvLumaTap;

#[inline]
#[target_feature(enable = "neon")]
fn widen4_u8_i32(arr: [u8; 4]) -> int32x4_t {
    let dup = vreinterpret_u8_u32(vdup_n_u32(u32::from_le_bytes(arr)));
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(dup))))
}

#[inline]
#[target_feature(enable = "neon")]
fn load4u8(row: &[u8], idx: usize) -> int32x4_t {
    widen4_u8_i32(row[idx..idx + 4].try_into().unwrap())
}

#[inline]
#[target_feature(enable = "neon")]
fn gather4u8(row: &[u8], idx: usize, step: usize) -> int32x4_t {
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
    widen4_u8_i32(arr)
}

#[inline]
#[target_feature(enable = "neon")]
fn finish4(dst: &mut [u8], x: usize, s: int32x4_t) {
    let v = vminq_s32(
        vmaxq_s32(
            vshrq_n_s32::<7>(vaddq_s32(s, vdupq_n_s32(64))),
            vdupq_n_s32(0),
        ),
        vdupq_n_s32(255),
    );
    let u16x4 = vmovn_u32(vreinterpretq_u32_s32(v));
    let u8x8 = vmovn_u16(vcombine_u16(u16x4, u16x4));
    let lane = vget_lane_u32::<0>(vreinterpret_u32_u8(u8x8));
    dst[x..x + 4].copy_from_slice(&lane.to_le_bytes());
}

/// NEON chroma NS-Wiener FIR. Mirror of `ns_wiener_uv_fir_run_sse41`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ns_wiener_uv_fir_run_neon(
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
    let mut x = 0;
    while x + 4 <= n {
        let cb = co + x;
        unsafe {
            let m = load4u8(c_center, cb);
            let two_m = vaddq_s32(m, m);
            let mut s = vshlq_n_s32::<7>(m);
            for t in ctaps {
                let a = load4u8(t.row_p, (cb as i32 + t.dx) as usize);
                let b = load4u8(t.row_m, (cb as i32 - t.dx) as usize);
                let coef = vdupq_n_s32(t.coef);
                s = vaddq_s32(s, vmulq_s32(vsubq_s32(vaddq_s32(a, b), two_m), coef));
            }
            let lb = lo + x * lstep;
            let lc = gather4u8(l_center, lb, lstep);
            for t in ltaps {
                let lv = gather4u8(t.row, (lb as i32 + t.ldx) as usize, lstep);
                let coef = vdupq_n_s32(t.coef);
                s = vaddq_s32(s, vmulq_s32(vsubq_s32(lv, lc), coef));
            }
            finish4(dst, x, s);
        }
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
