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

use std::arch::aarch64::*;

#[inline(always)]
fn load4_u8_i32(dst: &[u8], base: isize, stride_line: isize) -> int32x4_t {
    let arr: [u8; 4] = if stride_line == 1 {
        dst[base as usize..base as usize + 4].try_into().unwrap()
    } else {
        [
            dst[base as usize],
            dst[(base + stride_line) as usize],
            dst[(base + 2 * stride_line) as usize],
            dst[(base + 3 * stride_line) as usize],
        ]
    };
    let dup = unsafe { vreinterpret_u8_u32(vdup_n_u32(u32::from_le_bytes(arr))) };
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(dup)))) }
}

#[inline(always)]
fn store4_clip_u8(dst: &mut [u8], base: isize, stride_line: isize, v: int32x4_t) {
    if stride_line == 1 {
        let u16x4 = unsafe { vqmovun_s32(v) };
        let u8x8 = unsafe { vqmovn_u16(vcombine_u16(u16x4, u16x4)) };
        let lane = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(u8x8)) };
        dst[base as usize..base as usize + 4].copy_from_slice(&lane.to_le_bytes());
    } else {
        let mut arr = [0i32; 4];
        unsafe { vst1q_s32(arr.as_mut_ptr(), v) };
        dst[base as usize] = arr[0] as u8;
        dst[(base + stride_line) as usize] = arr[1] as u8;
        dst[(base + 2 * stride_line) as usize] = arr[2] as u8;
        dst[(base + 3 * stride_line) as usize] = arr[3] as u8;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn deblock_apply_8bpc_neon(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    let qc = vdupq_n_s32(q_thr_clamp);
    let nqc = vdupq_n_s32(-q_thr_clamp);
    let rnd = vdupq_n_s32(1 << 10);
    let zero = vdupq_n_s32(0);
    let v255 = vdupq_n_s32(255);
    let three = vdupq_n_s32(3);
    let four = vdupq_n_s32(4);
    let nsh = vdupq_n_s32(-11);

    let d0 = load4_u8_i32(dst, off, stride_line);
    let dm1 = load4_u8_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u8_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u8_i32(dst, off - 2 * stride_tap, stride_line);
    // delta_m2 = clip(4*(3*(d0-dm1) - (dp1-dm2)), -qc, qc)
    let inner = vsubq_s32(vmulq_s32(three, vsubq_s32(d0, dm1)), vsubq_s32(dp1, dm2));
    let delta = vminq_s32(vmaxq_s32(vmulq_s32(four, inner), nqc), qc);

    if !neg_lossless {
        let dn = vmulq_s32(
            delta,
            vdupq_n_s32(crate::deblock::W_MULT[(width_neg - 1) as usize] as i32),
        );
        for j in 0..width_neg {
            let diff = vshlq_s32(
                vaddq_s32(vmulq_s32(dn, vdupq_n_s32(width_neg - j)), rnd),
                nsh,
            );
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u8_i32(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vaddq_s32(cur, diff), zero), v255);
            store4_clip_u8(dst, base, stride_line, res);
        }
    }

    if !pos_lossless {
        let dpv = vmulq_s32(
            delta,
            vdupq_n_s32(crate::deblock::W_MULT[(width_pos - 1) as usize] as i32),
        );
        for j in 0..width_pos {
            let diff = vshlq_s32(
                vaddq_s32(vmulq_s32(dpv, vdupq_n_s32(width_pos - j)), rnd),
                nsh,
            );
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u8_i32(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vsubq_s32(cur, diff), zero), v255);
            store4_clip_u8(dst, base, stride_line, res);
        }
    }
}
