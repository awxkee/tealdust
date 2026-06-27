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

use crate::filter::{UvLumaTapHbd, WienerTapHbd};
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn u16x8_to_i32x2(v: uint16x8_t) -> (int32x4_t, int32x4_t) {
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v)));
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn load8_u16_i32(p: &[u16]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u16(p.as_ptr()) };
    u16x8_to_i32x2(v)
}

#[inline]
#[target_feature(enable = "neon")]
fn load16_u16_i32x4(p: &[u16]) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    unsafe {
        let lo = vld1q_u16(p.as_ptr());
        let hi = vld1q_u16(p.as_ptr().add(8));
        let (a0, a1) = u16x8_to_i32x2(lo);
        let (a2, a3) = u16x8_to_i32x2(hi);
        (a0, a1, a2, a3)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn finish8_u16(dst: &mut [u16], slo: int32x4_t, shi: int32x4_t, bitdepth_max: i32) {
    let rnd = vdupq_n_s32(64);
    let zero = vdupq_n_s32(0);
    let max = vdupq_n_s32(bitdepth_max);
    let vlo = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(slo, rnd)), zero), max);
    let vhi = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(shi, rnd)), zero), max);
    let u16lo = vmovn_u32(vreinterpretq_u32_s32(vlo));
    let u16hi = vmovn_u32(vreinterpretq_u32_s32(vhi));
    unsafe { vst1q_u16(dst.as_mut_ptr(), vcombine_u16(u16lo, u16hi)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn finish16_u16(
    dst: &mut [u16],
    s0: int32x4_t,
    s1: int32x4_t,
    s2: int32x4_t,
    s3: int32x4_t,
    bitdepth_max: i32,
) {
    let rnd = vdupq_n_s32(64);
    let zero = vdupq_n_s32(0);
    let max = vdupq_n_s32(bitdepth_max);
    let v0 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s0, rnd)), zero), max);
    let v1 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s1, rnd)), zero), max);
    let v2 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s2, rnd)), zero), max);
    let v3 = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vaddq_s32(s3, rnd)), zero), max);
    unsafe {
        vst1q_u16(
            dst.as_mut_ptr(),
            vcombine_u16(
                vmovn_u32(vreinterpretq_u32_s32(v0)),
                vmovn_u32(vreinterpretq_u32_s32(v1)),
            ),
        );
        vst1q_u16(
            dst.as_mut_ptr().add(8),
            vcombine_u16(
                vmovn_u32(vreinterpretq_u32_s32(v2)),
                vmovn_u32(vreinterpretq_u32_s32(v3)),
            ),
        );
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn gather8_u16_i32(row: &[u16], idx: usize, step: usize) -> (int32x4_t, int32x4_t) {
    if step == 1 {
        load8_u16_i32(&row[idx..])
    } else if step == 2 && idx + 16 <= row.len() {
        unsafe {
            let lo = vld1q_u16(row.as_ptr().add(idx));
            let hi = vld1q_u16(row.as_ptr().add(idx + 8));
            u16x8_to_i32x2(vuzp1q_u16(lo, hi))
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
        load8_u16_i32(&arr)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn gather16_u16_i32(
    row: &[u16],
    idx: usize,
    step: usize,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    if step == 1 {
        load16_u16_i32x4(&row[idx..])
    } else {
        let (a0, a1) = gather8_u16_i32(row, idx, step);
        let (a2, a3) = gather8_u16_i32(row, idx + 8 * step, step);
        (a0, a1, a2, a3)
    }
}

#[target_feature(enable = "neon")]
fn ns_wiener_fir_run_hbd_neon_impl(
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
        let (m0, m1, m2, m3) = load16_u16_i32x4(&center[c..]);
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
            let (a0, a1, a2, a3) = load16_u16_i32x4(&t.row_p[cp..]);
            let (b0, b1, b2, b3) = load16_u16_i32x4(&t.row_m[cm..]);
            let coef = vdupq_n_s32(t.coef);
            s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(vaddq_s32(a0, b0), two_m0), coef));
            s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(vaddq_s32(a1, b1), two_m1), coef));
            s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(vaddq_s32(a2, b2), two_m2), coef));
            s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(vaddq_s32(a3, b3), two_m3), coef));
        }
        finish16_u16(&mut dst[x..], s0, s1, s2, s3, bitdepth_max);
        x += 16;
    }
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let (mlo, mhi) = load8_u16_i32(&center[c..]);
        let mut slo = vshlq_n_s32::<7>(mlo);
        let mut shi = vshlq_n_s32::<7>(mhi);
        let two_mlo = vaddq_s32(mlo, mlo);
        let two_mhi = vaddq_s32(mhi, mhi);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let (alo, ahi) = load8_u16_i32(&t.row_p[cp..]);
            let (blo, bhi) = load8_u16_i32(&t.row_m[cm..]);
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
        finish8_u16(&mut dst[x..], slo, shi, bitdepth_max);
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

#[target_feature(enable = "neon")]
fn pc_wiener_fir_run_hbd_neon_impl(
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
        let (m0, m1, m2, m3) = load16_u16_i32x4(&center[c..]);
        let cc = vdupq_n_s32(center_coef);
        let mut s0 = vmulq_s32(m0, cc);
        let mut s1 = vmulq_s32(m1, cc);
        let mut s2 = vmulq_s32(m2, cc);
        let mut s3 = vmulq_s32(m3, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 16 <= t.row_p.len() && cm + 16 <= t.row_m.len());
            let (a0, a1, a2, a3) = load16_u16_i32x4(&t.row_p[cp..]);
            let (b0, b1, b2, b3) = load16_u16_i32x4(&t.row_m[cm..]);
            let coef = vdupq_n_s32(t.coef);
            s0 = vaddq_s32(s0, vmulq_s32(vaddq_s32(a0, b0), coef));
            s1 = vaddq_s32(s1, vmulq_s32(vaddq_s32(a1, b1), coef));
            s2 = vaddq_s32(s2, vmulq_s32(vaddq_s32(a2, b2), coef));
            s3 = vaddq_s32(s3, vmulq_s32(vaddq_s32(a3, b3), coef));
        }
        finish16_u16(&mut dst[x..], s0, s1, s2, s3, bitdepth_max);
        x += 16;
    }
    while x + 8 <= n {
        let c = col0 + x;
        debug_assert!(c + 8 <= center.len());
        let (mlo, mhi) = load8_u16_i32(&center[c..]);
        let cc = vdupq_n_s32(center_coef);
        let mut slo = vmulq_s32(mlo, cc);
        let mut shi = vmulq_s32(mhi, cc);
        for t in taps {
            let cp = (c as i32 + t.dx) as usize;
            let cm = (c as i32 - t.dx) as usize;
            debug_assert!(cp + 8 <= t.row_p.len() && cm + 8 <= t.row_m.len());
            let (alo, ahi) = load8_u16_i32(&t.row_p[cp..]);
            let (blo, bhi) = load8_u16_i32(&t.row_m[cm..]);
            let coef = vdupq_n_s32(t.coef);
            slo = vaddq_s32(slo, vmulq_s32(vaddq_s32(alo, blo), coef));
            shi = vaddq_s32(shi, vmulq_s32(vaddq_s32(ahi, bhi), coef));
        }
        finish8_u16(&mut dst[x..], slo, shi, bitdepth_max);
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

pub(crate) fn ns_wiener_fir_run_hbd_neon(
    dst: &mut [u16],
    center: &[u16],
    col0: usize,
    taps: &[WienerTapHbd],
    n: usize,
    bitdepth_max: i32,
) {
    unsafe { ns_wiener_fir_run_hbd_neon_impl(dst, center, col0, taps, n, bitdepth_max) }
}

pub(crate) fn pc_wiener_fir_run_hbd_neon(
    dst: &mut [u16],
    center: &[u16],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTapHbd],
    n: usize,
    bitdepth_max: i32,
) {
    unsafe {
        pc_wiener_fir_run_hbd_neon_impl(dst, center, center_coef, col0, taps, n, bitdepth_max)
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn ns_wiener_uv_fir_run_hbd_neon_impl(
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
        let (m0, m1, m2, m3) = load16_u16_i32x4(&c_center[cb..]);
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
            let (a0, a1, a2, a3) = load16_u16_i32x4(&t.row_p[cp..]);
            let (b0, b1, b2, b3) = load16_u16_i32x4(&t.row_m[cm..]);
            let coef = vdupq_n_s32(t.coef);
            s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(vaddq_s32(a0, b0), two_m0), coef));
            s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(vaddq_s32(a1, b1), two_m1), coef));
            s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(vaddq_s32(a2, b2), two_m2), coef));
            s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(vaddq_s32(a3, b3), two_m3), coef));
        }
        let lb = lo + x * lstep;
        let (lc0, lc1, lc2, lc3) = gather16_u16_i32(l_center, lb, lstep);
        for t in ltaps {
            let li = (lb as i32 + t.ldx) as usize;
            let (lv0, lv1, lv2, lv3) = gather16_u16_i32(t.row, li, lstep);
            let coef = vdupq_n_s32(t.coef);
            s0 = vaddq_s32(s0, vmulq_s32(vsubq_s32(lv0, lc0), coef));
            s1 = vaddq_s32(s1, vmulq_s32(vsubq_s32(lv1, lc1), coef));
            s2 = vaddq_s32(s2, vmulq_s32(vsubq_s32(lv2, lc2), coef));
            s3 = vaddq_s32(s3, vmulq_s32(vsubq_s32(lv3, lc3), coef));
        }
        finish16_u16(&mut dst[x..], s0, s1, s2, s3, bitdepth_max);
        x += 16;
    }
    while x + 8 <= n {
        let cb = co + x;
        let (mlo, mhi) = load8_u16_i32(&c_center[cb..]);
        let two_mlo = vaddq_s32(mlo, mlo);
        let two_mhi = vaddq_s32(mhi, mhi);
        let mut slo = vshlq_n_s32::<7>(mlo);
        let mut shi = vshlq_n_s32::<7>(mhi);
        for t in ctaps {
            let (alo, ahi) = load8_u16_i32(&t.row_p[(cb as i32 + t.dx) as usize..]);
            let (blo, bhi) = load8_u16_i32(&t.row_m[(cb as i32 - t.dx) as usize..]);
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
        let (lclo, lchi) = gather8_u16_i32(l_center, lb, lstep);
        for t in ltaps {
            let (lvlo, lvhi) = gather8_u16_i32(t.row, (lb as i32 + t.ldx) as usize, lstep);
            let coef = vdupq_n_s32(t.coef);
            slo = vaddq_s32(slo, vmulq_s32(vsubq_s32(lvlo, lclo), coef));
            shi = vaddq_s32(shi, vmulq_s32(vsubq_s32(lvhi, lchi), coef));
        }
        finish8_u16(&mut dst[x..], slo, shi, bitdepth_max);
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn ns_wiener_uv_fir_run_hbd_neon(
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
    unsafe {
        ns_wiener_uv_fir_run_hbd_neon_impl(
            dst,
            c_center,
            co,
            ctaps,
            l_center,
            lo,
            ltaps,
            lstep,
            n,
            bitdepth_max,
        )
    }
}
