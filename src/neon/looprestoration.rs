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
fn u8x8_to_i32x2(v: uint8x8_t) -> (int32x4_t, int32x4_t) {
    let w = vmovl_u8(v); // 8 x u16
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w)));
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn load8_u8_i32(p: &[u8]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1_u8(p.as_ptr()) }; // 8 x u8
    u8x8_to_i32x2(v)
}

#[inline]
#[target_feature(enable = "neon")]
fn u8x16_to_i32x4(v: uint8x16_t) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let lo16 = vmovl_u8(vget_low_u8(v));
    let hi16 = vmovl_u8(vget_high_u8(v));
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load16_u8_i32x4(p: &[u8]) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u8(p.as_ptr()) };
    u8x16_to_i32x4(v)
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::type_complexity)]
fn load32_u8_i32x8(
    p: &[u8],
) -> (
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
) {
    let (a0, a1, a2, a3) = load16_u8_i32x4(p);
    let (a4, a5, a6, a7) = load16_u8_i32x4(&p[16..]);
    (a0, a1, a2, a3, a4, a5, a6, a7)
}

/// `(s + 64) >> 7`, clamped to `[0, 255]`, then narrow two `int32x4_t` halves
/// to 8 packed `u8` and store at `dst`.
#[inline]
#[target_feature(enable = "neon")]
fn finish_store(dst: &mut [u8], slo: int32x4_t, shi: int32x4_t) {
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
    unsafe { vst1_u8(dst.as_mut_ptr(), packed) };
}

#[inline]
#[target_feature(enable = "neon")]
fn finish_store16(dst: &mut [u8], s0: int32x4_t, s1: int32x4_t, s2: int32x4_t, s3: int32x4_t) {
    let rnd = vdupq_n_s32(64);
    let zero = vdupq_n_s32(0);
    let max = vdupq_n_s32(255);
    let v0 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s0, rnd)), zero), max);
    let v1 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s1, rnd)), zero), max);
    let v2 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s2, rnd)), zero), max);
    let v3 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s3, rnd)), zero), max);
    let u16a = vcombine_u16(
        vmovn_u32(vreinterpretq_u32_s32(v0)),
        vmovn_u32(vreinterpretq_u32_s32(v1)),
    );
    let u16b = vcombine_u16(
        vmovn_u32(vreinterpretq_u32_s32(v2)),
        vmovn_u32(vreinterpretq_u32_s32(v3)),
    );
    let packed = vcombine_u8(vmovn_u16(u16a), vmovn_u16(u16b));
    unsafe { vst1q_u8(dst.as_mut_ptr(), packed) };
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn finish_store32(
    dst: &mut [u8],
    s0: int32x4_t,
    s1: int32x4_t,
    s2: int32x4_t,
    s3: int32x4_t,
    s4: int32x4_t,
    s5: int32x4_t,
    s6: int32x4_t,
    s7: int32x4_t,
) {
    finish_store16(dst, s0, s1, s2, s3);
    finish_store16(&mut dst[16..], s4, s5, s6, s7);
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
    while x + 32 <= n {
        let c = col0 + x;
        debug_assert!(c + 32 <= center.len());
        unsafe {
            let (m0, m1, m2, m3, m4, m5, m6, m7) = load32_u8_i32x8(&center[c..]);
            let mut s0 = vshlq_n_s32::<7>(m0);
            let mut s1 = vshlq_n_s32::<7>(m1);
            let mut s2 = vshlq_n_s32::<7>(m2);
            let mut s3 = vshlq_n_s32::<7>(m3);
            let mut s4 = vshlq_n_s32::<7>(m4);
            let mut s5 = vshlq_n_s32::<7>(m5);
            let mut s6 = vshlq_n_s32::<7>(m6);
            let mut s7 = vshlq_n_s32::<7>(m7);
            let two_m0 = vaddq_s32(m0, m0);
            let two_m1 = vaddq_s32(m1, m1);
            let two_m2 = vaddq_s32(m2, m2);
            let two_m3 = vaddq_s32(m3, m3);
            let two_m4 = vaddq_s32(m4, m4);
            let two_m5 = vaddq_s32(m5, m5);
            let two_m6 = vaddq_s32(m6, m6);
            let two_m7 = vaddq_s32(m7, m7);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 32 <= t.row_p.len() && cm + 32 <= t.row_m.len());
                let (a0, a1, a2, a3, a4, a5, a6, a7) = load32_u8_i32x8(&t.row_p[cp..]);
                let (b0, b1, b2, b3, b4, b5, b6, b7) = load32_u8_i32x8(&t.row_m[cm..]);
                let coef = vdupq_n_s32(t.coef);
                s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(vaddq_s32(a0, b0), two_m0), coef));
                s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(vaddq_s32(a1, b1), two_m1), coef));
                s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(vaddq_s32(a2, b2), two_m2), coef));
                s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(vaddq_s32(a3, b3), two_m3), coef));
                s4 = vaddq_s32(s4, vmulq_s32(vsubq_s32(vaddq_s32(a4, b4), two_m4), coef));
                s5 = vaddq_s32(s5, vmulq_s32(vsubq_s32(vaddq_s32(a5, b5), two_m5), coef));
                s6 = vaddq_s32(s6, vmulq_s32(vsubq_s32(vaddq_s32(a6, b6), two_m6), coef));
                s7 = vaddq_s32(s7, vmulq_s32(vsubq_s32(vaddq_s32(a7, b7), two_m7), coef));
            }
            finish_store32(&mut dst[x..], s0, s1, s2, s3, s4, s5, s6, s7);
        }
        x += 32;
    }
    while x + 16 <= n {
        let c = col0 + x;
        debug_assert!(c + 16 <= center.len());
        unsafe {
            let (m0, m1, m2, m3) = load16_u8_i32x4(&center[c..]);
            let mut s0 = vshlq_n_s32::<7>(m0);
            let mut s1 = vshlq_n_s32::<7>(m1);
            let mut s2 = vshlq_n_s32::<7>(m2);
            let mut s3 = vshlq_n_s32::<7>(m3);
            let two_m0 = vaddq_s32(m0, m0);
            let two_m1 = vaddq_s32(m1, m1);
            let two_m2 = vaddq_s32(m2, m2);
            let two_m3 = vaddq_s32(m3, m3);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
                let (a0, a1, a2, a3) = load16_u8_i32x4(&t.row_p[cp..]);
                let (b0, b1, b2, b3) = load16_u8_i32x4(&t.row_m[cm..]);
                let coef = vdupq_n_s32(t.coef);
                s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(vaddq_s32(a0, b0), two_m0), coef));
                s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(vaddq_s32(a1, b1), two_m1), coef));
                s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(vaddq_s32(a2, b2), two_m2), coef));
                s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(vaddq_s32(a3, b3), two_m3), coef));
            }
            finish_store16(&mut dst[x..], s0, s1, s2, s3);
        }
        x += 16;
    }
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        unsafe {
            let (mlo, mhi) = load8_u8_i32(&center[c..]);
            let mut slo = vshlq_n_s32::<7>(mlo);
            let mut shi = vshlq_n_s32::<7>(mhi);
            let two_mlo = vaddq_s32(mlo, mlo);
            let two_mhi = vaddq_s32(mhi, mhi);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8_u8_i32(&t.row_p[cp..]);
                let (blo, bhi) = load8_u8_i32(&t.row_m[cm..]);
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
            finish_store(&mut dst[x..], slo, shi);
        }
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
    while x + 32 <= n {
        let c = col0 + x;
        debug_assert!(c + 32 <= center.len());
        unsafe {
            let (m0, m1, m2, m3, m4, m5, m6, m7) = load32_u8_i32x8(&center[c..]);
            let cc = vdupq_n_s32(center_coef);
            let mut s0 = vmulq_s32(m0, cc);
            let mut s1 = vmulq_s32(m1, cc);
            let mut s2 = vmulq_s32(m2, cc);
            let mut s3 = vmulq_s32(m3, cc);
            let mut s4 = vmulq_s32(m4, cc);
            let mut s5 = vmulq_s32(m5, cc);
            let mut s6 = vmulq_s32(m6, cc);
            let mut s7 = vmulq_s32(m7, cc);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 32 <= t.row_p.len() && cm + 32 <= t.row_m.len());
                let (a0, a1, a2, a3, a4, a5, a6, a7) = load32_u8_i32x8(&t.row_p[cp..]);
                let (b0, b1, b2, b3, b4, b5, b6, b7) = load32_u8_i32x8(&t.row_m[cm..]);
                let coef = vdupq_n_s32(t.coef);
                s0 = vaddq_s32(s0, vmulq_s32(vaddq_s32(a0, b0), coef));
                s1 = vaddq_s32(s1, vmulq_s32(vaddq_s32(a1, b1), coef));
                s2 = vaddq_s32(s2, vmulq_s32(vaddq_s32(a2, b2), coef));
                s3 = vaddq_s32(s3, vmulq_s32(vaddq_s32(a3, b3), coef));
                s4 = vaddq_s32(s4, vmulq_s32(vaddq_s32(a4, b4), coef));
                s5 = vaddq_s32(s5, vmulq_s32(vaddq_s32(a5, b5), coef));
                s6 = vaddq_s32(s6, vmulq_s32(vaddq_s32(a6, b6), coef));
                s7 = vaddq_s32(s7, vmulq_s32(vaddq_s32(a7, b7), coef));
            }
            finish_store32(&mut dst[x..], s0, s1, s2, s3, s4, s5, s6, s7);
        }
        x += 32;
    }
    while x + 16 <= n {
        let c = col0 + x;
        debug_assert!(c + 16 <= center.len());
        unsafe {
            let (m0, m1, m2, m3) = load16_u8_i32x4(&center[c..]);
            let cc = vdupq_n_s32(center_coef);
            let mut s0 = vmulq_s32(m0, cc);
            let mut s1 = vmulq_s32(m1, cc);
            let mut s2 = vmulq_s32(m2, cc);
            let mut s3 = vmulq_s32(m3, cc);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
                let (a0, a1, a2, a3) = load16_u8_i32x4(&t.row_p[cp..]);
                let (b0, b1, b2, b3) = load16_u8_i32x4(&t.row_m[cm..]);
                let coef = vdupq_n_s32(t.coef);
                s0 = vaddq_s32(s0, vmulq_s32(vaddq_s32(a0, b0), coef));
                s1 = vaddq_s32(s1, vmulq_s32(vaddq_s32(a1, b1), coef));
                s2 = vaddq_s32(s2, vmulq_s32(vaddq_s32(a2, b2), coef));
                s3 = vaddq_s32(s3, vmulq_s32(vaddq_s32(a3, b3), coef));
            }
            finish_store16(&mut dst[x..], s0, s1, s2, s3);
        }
        x += 16;
    }
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        unsafe {
            let (mlo, mhi) = load8_u8_i32(&center[c..]);
            let cc = vdupq_n_s32(center_coef);
            let mut slo = vmulq_s32(mlo, cc);
            let mut shi = vmulq_s32(mhi, cc);
            for t in taps {
                let cp = (c as i32 + t.dx) as usize;
                let cm = (c as i32 - t.dx) as usize;
                debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
                let (alo, ahi) = load8_u8_i32(&t.row_p[cp..]);
                let (blo, bhi) = load8_u8_i32(&t.row_m[cm..]);
                let coef = vdupq_n_s32(t.coef);
                // (a + b) * coef
                slo = vaddq_s32(slo, vmulq_s32(vaddq_s32(alo, blo), coef));
                shi = vaddq_s32(shi, vmulq_s32(vaddq_s32(ahi, bhi), coef));
            }
            finish_store(&mut dst[x..], slo, shi);
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
fn gather8u8(row: &[u8], idx: usize, step: usize) -> (int32x4_t, int32x4_t) {
    if step == 1 {
        load8_u8_i32(&row[idx..])
    } else if step == 2 && idx + 16 <= row.len() {
        unsafe {
            let v = vld1q_u8(row.as_ptr().add(idx));
            let even = vget_low_u8(vuzp1q_u8(v, v));
            u8x8_to_i32x2(even)
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
        load8_u8_i32(&arr)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn gather16u8(row: &[u8], idx: usize, step: usize) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    if step == 1 {
        load16_u8_i32x4(&row[idx..])
    } else if step == 2 && idx + 32 <= row.len() {
        unsafe {
            let lo = vld1q_u8(row.as_ptr().add(idx));
            let hi = vld1q_u8(row.as_ptr().add(idx + 16));
            u8x16_to_i32x4(vuzp1q_u8(lo, hi))
        }
    } else {
        let (a0, a1) = gather8u8(row, idx, step);
        let (a2, a3) = gather8u8(row, idx + 8 * step, step);
        (a0, a1, a2, a3)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn finish8(dst: &mut [u8], slo: int32x4_t, shi: int32x4_t) {
    finish_store(dst, slo, shi);
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
    while x + 16 <= n {
        let cb = co + x;
        unsafe {
            let (m0, m1, m2, m3) = load16_u8_i32x4(&c_center[cb..]);
            let two_m0 = vaddq_s32(m0, m0);
            let two_m1 = vaddq_s32(m1, m1);
            let two_m2 = vaddq_s32(m2, m2);
            let two_m3 = vaddq_s32(m3, m3);
            let mut s0 = vshlq_n_s32::<7>(m0);
            let mut s1 = vshlq_n_s32::<7>(m1);
            let mut s2 = vshlq_n_s32::<7>(m2);
            let mut s3 = vshlq_n_s32::<7>(m3);
            for t in ctaps {
                let cp = (cb as i32 + t.dx) as usize;
                let cm = (cb as i32 - t.dx) as usize;
                let (a0, a1, a2, a3) = load16_u8_i32x4(&t.row_p[cp..]);
                let (b0, b1, b2, b3) = load16_u8_i32x4(&t.row_m[cm..]);
                let coef = vdupq_n_s32(t.coef);
                s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(vaddq_s32(a0, b0), two_m0), coef));
                s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(vaddq_s32(a1, b1), two_m1), coef));
                s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(vaddq_s32(a2, b2), two_m2), coef));
                s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(vaddq_s32(a3, b3), two_m3), coef));
            }
            let lb = lo + x * lstep;
            let (lc0, lc1, lc2, lc3) = gather16u8(l_center, lb, lstep);
            for t in ltaps {
                let li = (lb as i32 + t.ldx) as usize;
                let (lv0, lv1, lv2, lv3) = gather16u8(t.row, li, lstep);
                let coef = vdupq_n_s32(t.coef);
                s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(lv0, lc0), coef));
                s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(lv1, lc1), coef));
                s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(lv2, lc2), coef));
                s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(lv3, lc3), coef));
            }
            finish_store16(&mut dst[x..], s0, s1, s2, s3);
        }
        x += 16;
    }
    while x + 8 <= n {
        let cb = co + x;
        unsafe {
            let (mlo, mhi) = load8_u8_i32(&c_center[cb..]);
            let two_mlo = vaddq_s32(mlo, mlo);
            let two_mhi = vaddq_s32(mhi, mhi);
            let mut slo = vshlq_n_s32::<7>(mlo);
            let mut shi = vshlq_n_s32::<7>(mhi);
            for t in ctaps {
                let (alo, ahi) = load8_u8_i32(&t.row_p[(cb as i32 + t.dx) as usize..]);
                let (blo, bhi) = load8_u8_i32(&t.row_m[(cb as i32 - t.dx) as usize..]);
                let coef = vdupq_n_s32(t.coef);
                slo = vaddq_s32(
                    slo,
                    vmulq_s32(vsubq_s32(vaddq_s32(alo, blo), two_mlo), coef),
                );
                shi = vaddq_s32(
                    shi,
                    vmulq_s32(vsubq_s32(vaddq_s32(ahi, bhi), two_mhi), coef),
                );
            }
            let lb = lo + x * lstep;
            let (lclo, lchi) = gather8u8(l_center, lb, lstep);
            for t in ltaps {
                let (lvlo, lvhi) = gather8u8(t.row, (lb as i32 + t.ldx) as usize, lstep);
                let coef = vdupq_n_s32(t.coef);
                slo = vaddq_s32(slo, vmulq_s32(vsubq_s32(lvlo, lclo), coef));
                shi = vaddq_s32(shi, vmulq_s32(vsubq_s32(lvhi, lchi), coef));
            }
            finish8(&mut dst[x..], slo, shi);
        }
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
