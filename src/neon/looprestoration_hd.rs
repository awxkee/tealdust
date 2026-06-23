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
fn load8_u16_i32(p: &[u16]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u16(p.as_ptr()) };
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v)));
    (lo, hi)
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
fn load4_u16_i32(row: &[u16], idx: usize) -> int32x4_t {
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(row[idx..].as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn gather4_u16_i32(row: &[u16], idx: usize, step: usize) -> int32x4_t {
    if step == 1 {
        load4_u16_i32(row, idx)
    } else {
        let arr = [
            row[idx],
            row[idx + step],
            row[idx + 2 * step],
            row[idx + 3 * step],
        ];
        unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(arr.as_ptr()))) }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn finish4_u16(dst: &mut [u16], x: usize, s: int32x4_t, bitdepth_max: i32) {
    let v = vminq_s32(
        vmaxq_s32(
            vshrq_n_s32::<7>(vaddq_s32(s, vdupq_n_s32(64))),
            vdupq_n_s32(0),
        ),
        vdupq_n_s32(bitdepth_max),
    );
    let u16x4 = vmovn_u32(vreinterpretq_u32_s32(v));
    unsafe { vst1_u16(dst[x..].as_mut_ptr(), u16x4) };
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
    while x + 4 <= n {
        let cb = co + x;
        let m = load4_u16_i32(c_center, cb);
        let two_m = vaddq_s32(m, m);
        let mut s = vshlq_n_s32::<7>(m);
        for t in ctaps {
            let a = load4_u16_i32(t.row_p, (cb as i32 + t.dx) as usize);
            let b = load4_u16_i32(t.row_m, (cb as i32 - t.dx) as usize);
            let coef = vdupq_n_s32(t.coef);
            s = vaddq_s32(s, vmulq_s32(vsubq_s32(vaddq_s32(a, b), two_m), coef));
        }
        let lb = lo + x * lstep;
        let lc = gather4_u16_i32(l_center, lb, lstep);
        for t in ltaps {
            let lv = gather4_u16_i32(t.row, (lb as i32 + t.ldx) as usize, lstep);
            let coef = vdupq_n_s32(t.coef);
            s = vaddq_s32(s, vmulq_s32(vsubq_s32(lv, lc), coef));
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
