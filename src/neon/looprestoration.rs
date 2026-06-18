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

//! Hand-written NEON loop-restoration FIR kernels.
//!
//! These are a 1:1 translation of the portable `I32x8` kernels in
//! `crate::simd` (`ns_wiener_fir_run_simd` / `pc_wiener_fir_run_simd`), which
//! are proven bit-exact against the pure-scalar reference by the
//! `wiener_scalar_proof` tests. Eight pixels are processed per iteration as two
//! `int32x4_t` halves (`lo`/`hi`); the scalar tail handles the remainder.
//!
//! NEON is part of the aarch64 baseline, so no `#[target_feature]` is required
//! (matching `crate::neon::itx`); the intrinsics are invoked inside `unsafe`.

use crate::simd::WienerTap;
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
            let (mlo, mhi) = load8_u8_i32(center.as_ptr().add(c));
            let mut slo = vshlq_n_s32::<7>(mlo);
            let mut shi = vshlq_n_s32::<7>(mhi);
            let two_mlo = vaddq_s32(mlo, mlo);
            let two_mhi = vaddq_s32(mhi, mhi);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8_u8_i32(t.row_p.as_ptr().add(cp));
                let (blo, bhi) = load8_u8_i32(t.row_m.as_ptr().add(cm));
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
            finish_store(dst.as_mut_ptr().add(x), slo, shi);
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
            let (mlo, mhi) = load8_u8_i32(center.as_ptr().add(c));
            let cc = vdupq_n_s32(center_coef);
            let mut slo = vmulq_s32(mlo, cc);
            let mut shi = vmulq_s32(mhi, cc);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8_u8_i32(t.row_p.as_ptr().add(cp));
                let (blo, bhi) = load8_u8_i32(t.row_m.as_ptr().add(cm));
                let coef = vdupq_n_s32(t.coef);
                // (a + b) * coef
                slo = vaddq_s32(slo, vmulq_s32(vaddq_s32(alo, blo), coef));
                shi = vaddq_s32(shi, vmulq_s32(vaddq_s32(ahi, bhi), coef));
            }
            finish_store(dst.as_mut_ptr().add(x), slo, shi);
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
