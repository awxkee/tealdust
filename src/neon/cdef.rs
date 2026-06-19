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
fn load_i16x4_i32(a: &[i16; 4]) -> int32x4_t {
    unsafe { vmovl_s16(vld1_s16(a.as_ptr())) }
}

#[inline(always)]
fn store_i32x4_u8(a: &mut [u8; 4], v: int32x4_t) {
    let u16x4 = unsafe { vqmovun_s32(v) };
    let u8x8 = unsafe { vqmovn_u16(vcombine_u16(u16x4, u16x4)) };
    let lane = unsafe { vget_lane_u32::<0>(vreinterpret_u32_u8(u8x8)) };
    *a = lane.to_le_bytes();
}

#[inline(always)]
fn constrain_v(diff: int32x4_t, threshold: int32x4_t, nsh: int32x4_t) -> int32x4_t {
    unsafe {
        let adiff = vabsq_s32(diff);
        let t = vmaxq_s32(vdupq_n_s32(0), vsubq_s32(threshold, vshlq_s32(adiff, nsh)));
        let m = vminq_s32(adiff, t);
        let neg = vsubq_s32(vdupq_n_s32(0), m);
        vbslq_s32(vcltq_s32(diff, vdupq_n_s32(0)), neg, m)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn cdef_filter_block_8bpc_neon(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    let has_pri = pri_strength != 0;
    let has_sec = sec_strength != 0;
    let clip = has_pri && has_sec;
    let pri_s = vdupq_n_s32(pri_strength);
    let sec_s = vdupq_n_s32(sec_strength);
    let pri_nsh = vdupq_n_s32(-pri_shift);
    let sec_nsh = vdupq_n_s32(-sec_shift);
    let zero = vdupq_n_s32(0);
    let eight = vdupq_n_s32(8);
    let nsh4 = vdupq_n_s32(-4);
    let lowmask = vdupq_n_s32(0xFF);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 4;
    let mut dp = dst_off;
    let mut tp = o;

    for _y in 0..h {
        for g in 0..groups {
            let bx = g * 4;
            let tpx = (tp + bx) as isize;
            let load = |off: isize| {
                load_i16x4_i32((&tmp[(tpx + off) as usize..][..4]).try_into().unwrap())
            };
            let px = load(0);
            let mut sum = zero;
            let mut min_v = px;
            let mut max_v = px;

            if has_pri {
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off1 = dirs[dir + 2][k] as isize;
                    let p0 = load(off1);
                    let p1 = load(-off1);
                    let pt = vdupq_n_s32(ptap);
                    sum = vaddq_s32(
                        sum,
                        vmulq_s32(pt, constrain_v(vsubq_s32(p0, px), pri_s, pri_nsh)),
                    );
                    sum = vaddq_s32(
                        sum,
                        vmulq_s32(pt, constrain_v(vsubq_s32(p1, px), pri_s, pri_nsh)),
                    );
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = vminq_s32(min_v, vminq_s32(p0, p1));
                        max_v = vmaxq_s32(max_v, vmaxq_s32(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = vdupq_n_s32(2 - k as i32);
                        sum = vaddq_s32(
                            sum,
                            vmulq_s32(st, constrain_v(vsubq_s32(s0, px), sec_s, sec_nsh)),
                        );
                        sum = vaddq_s32(
                            sum,
                            vmulq_s32(st, constrain_v(vsubq_s32(s1, px), sec_s, sec_nsh)),
                        );
                        sum = vaddq_s32(
                            sum,
                            vmulq_s32(st, constrain_v(vsubq_s32(s2, px), sec_s, sec_nsh)),
                        );
                        sum = vaddq_s32(
                            sum,
                            vmulq_s32(st, constrain_v(vsubq_s32(s3, px), sec_s, sec_nsh)),
                        );
                        min_v = vminq_s32(min_v, vminq_s32(vminq_s32(s0, s1), vminq_s32(s2, s3)));
                        max_v = vmaxq_s32(max_v, vmaxq_s32(vmaxq_s32(s0, s1), vmaxq_s32(s2, s3)));
                    }
                }
            } else {
                for k in 0..2 {
                    let off1 = dirs[dir + 4][k] as isize;
                    let off2 = dirs[dir][k] as isize;
                    let s0 = load(off1);
                    let s1 = load(-off1);
                    let s2 = load(off2);
                    let s3 = load(-off2);
                    let st = vdupq_n_s32(2 - k as i32);
                    sum = vaddq_s32(
                        sum,
                        vmulq_s32(st, constrain_v(vsubq_s32(s0, px), sec_s, sec_nsh)),
                    );
                    sum = vaddq_s32(
                        sum,
                        vmulq_s32(st, constrain_v(vsubq_s32(s1, px), sec_s, sec_nsh)),
                    );
                    sum = vaddq_s32(
                        sum,
                        vmulq_s32(st, constrain_v(vsubq_s32(s2, px), sec_s, sec_nsh)),
                    );
                    sum = vaddq_s32(
                        sum,
                        vmulq_s32(st, constrain_v(vsubq_s32(s3, px), sec_s, sec_nsh)),
                    );
                }
            }

            // delta = (sum - (sum < 0) + 8) >> 4 ; mask = -1 where sum<0, so sum + mask
            let mask = vreinterpretq_s32_u32(vcltq_s32(sum, zero));
            let delta = vshlq_s32(vaddq_s32(vaddq_s32(sum, mask), eight), nsh4);
            let mut res = vaddq_s32(px, delta);
            if clip {
                res = vminq_s32(vmaxq_s32(res, min_v), max_v);
            }
            res = vandq_s32(res, lowmask);
            store_i32x4_u8((&mut dst[dp + bx..dp + bx + 4]).try_into().unwrap(), res);
        }
        dp += dst_stride;
        tp += tmp_stride;
    }
}
