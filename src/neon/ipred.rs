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

use crate::intops::ulog2;
use crate::levels::ANGLE_MULTI_MRL_FLAG;
use crate::tables::SM_WEIGHTS;

#[inline(always)]
fn load_u8x8_i16(ptr: *const u8) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr))) }
}

#[inline(always)]
fn load_u8x8_i16_fixed(ptr: &[u8; 8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr.as_ptr()))) }
}

#[inline(always)]
fn store_i16x8_u8(ptr: *mut u8, v: int16x8_t) {
    unsafe { vst1_u8(ptr, vqmovun_s16(v)) };
}

#[inline(always)]
fn store_i16x8_u8_fixed(ptr: &mut [u8; 8], v: int16x8_t) {
    unsafe { vst1_u8(ptr.as_mut_ptr(), vqmovun_s16(v)) };
}

#[inline(always)]
fn sra_i16(v: int16x8_t, shift: i32) -> int16x8_t {
    unsafe { vshlq_s16(v, vdupq_n_s16(-(shift as i16))) }
}

#[inline(always)]
fn dist8(base: i16) -> int16x8_t {
    let a = [
        base,
        base - 1,
        base - 2,
        base - 3,
        base - 4,
        base - 5,
        base - 6,
        base - 7,
    ];
    unsafe { vld1q_s16(a.as_ptr()) }
}

pub(crate) fn ipred_v_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if width < 16 {
        crate::ipred_dispatch::ipred_v_scalar(dst, stride, tl, o, width, height, angle);
        return;
    }

    unsafe {
        if angle & ANGLE_MULTI_MRL_FLAG != 0 {
            let e_stride = (width + height) * 2 + 1;
            let mut x = 0usize;
            while x + 16 <= width {
                let a = vld1q_u8(tl.as_ptr().add(o + 1 + x));
                let b = vld1q_u8(tl.as_ptr().add(o + 1 + e_stride + x));
                let v = vrhaddq_u8(a, b);
                vst1q_u8(dst.as_mut_ptr().add(x), v);
                x += 16;
            }
            while x < width {
                let top = tl[o + 1 + x] as u16;
                let top2 = tl[o + 1 + e_stride + x] as u16;
                dst[x] = ((top + top2 + 1) >> 1) as u8;
                x += 1;
            }
        } else {
            let top = &tl[o + 1..o + 1 + width];
            dst.copy_from_slice(top);
        }
    }

    let mut off = stride;
    for _ in 1..height {
        dst.copy_within(0..width, off);
        off += stride;
    }
}

pub(crate) fn ipred_h_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if width < 16 {
        crate::ipred_dispatch::ipred_h_scalar(dst, stride, tl, o, width, height, angle);
        return;
    }

    let e_stride = (width + height) * 2 + 1;
    let mrl = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let mut off = 0usize;
    for y in 0..height {
        let v = if mrl {
            let left = tl[o - 1 - y] as u16;
            let left2 = tl[o + e_stride - 1 - y] as u16;
            ((left + left2 + 1) >> 1) as u8
        } else {
            tl[o - 1 - y]
        };
        let row = &mut dst[off..off + width];
        row.fill(v);
        off += stride;
    }
}

pub(crate) fn ipred_smooth_v_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_v_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bhl2 = ulog2(h as u32) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom = tl[o - h - 1] as i16;

    unsafe {
        let bottom_v = vdupq_n_s16(bottom);
        let rnd = vdupq_n_s16((h >> 1) as i16);
        let add32 = vdupq_n_s16(32);
        let mut off = 0usize;
        for y in 0..h {
            let off_y = vdupq_n_s16((h - 1 - y) as i16);
            let w_ver = vdupq_n_s16(weights[y] as i16);
            let row = &mut dst[off..off + w];
            let tl_src = &tl[o + 1..];
            for (dst, tl) in row
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(tl_src.as_chunks::<8>().0.iter())
            {
                let above = load_u8x8_i16_fixed(tl);
                let mul = vmulq_s16(vsubq_s16(above, bottom_v), off_y);
                let pred = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul, rnd), bhl2));
                let adj = sra_i16(
                    vaddq_s16(vmulq_s16(vsubq_s16(above, pred), w_ver), add32),
                    6,
                );
                store_i16x8_u8_fixed(dst, vaddq_s16(pred, adj));
            }

            let rem_dst = row.as_chunks_mut::<8>().1;
            let rem_src = tl_src.as_chunks::<8>().1;

            for (dst, &above_s) in rem_dst.iter_mut().zip(rem_src.iter()) {
                let above = above_s as i32;
                let mul = (above - bottom as i32) * (h as i32 - 1 - y as i32);
                let pred = bottom as i32 + ((mul + (h >> 1) as i32) >> bhl2);
                *dst = (pred + (((above - pred) * weights[y] as i32 + 32) >> 6)) as u8;
            }
            off += stride;
        }
    }
}

pub(crate) fn ipred_smooth_h_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_h_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;

    unsafe {
        let right_v = vdupq_n_s16(right);
        let rnd = vdupq_n_s16((w >> 1) as i16);
        let add32 = vdupq_n_s16(32);
        let mut off = 0usize;
        for y in 0..h {
            let left = tl[o - 1 - y] as i16;
            let left_v = vdupq_n_s16(left);
            let diff = vdupq_n_s16(left - right);
            let row = &mut dst[off..off + w];
            let mut x = 0usize;
            while x + 8 <= w {
                let d = dist8((w - 1 - x) as i16);
                let wx = load_u8x8_i16(weights.as_ptr().add(x));
                let pred = vaddq_s16(right_v, sra_i16(vaddq_s16(vmulq_s16(diff, d), rnd), bwl2));
                let adj = sra_i16(vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred), wx), add32), 6);
                store_i16x8_u8(row.as_mut_ptr().add(x), vaddq_s16(pred, adj));
                x += 8;
            }
            while x < w {
                let mul = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
                let pred = right as i32 + ((mul + (w >> 1) as i32) >> bwl2);
                row[x] = (pred + (((left as i32 - pred) * weights[x] as i32 + 32) >> 6)) as u8;
                x += 1;
            }
            off += stride;
        }
    }
}

pub(crate) fn ipred_smooth_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let bhl2 = ulog2(h as u32) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let bottom = tl[o - h - 1] as i16;

    unsafe {
        let right_v = vdupq_n_s16(right);
        let bottom_v = vdupq_n_s16(bottom);
        let rnd_ver = vdupq_n_s16((h >> 1) as i16);
        let rnd_hor = vdupq_n_s16((w >> 1) as i16);
        let add32 = vdupq_n_s16(32);
        let one = vdupq_n_s16(1);
        let mut off = 0usize;
        for y in 0..h {
            let left = tl[o - 1 - y] as i16;
            let left_v = vdupq_n_s16(left);
            let diff_hor = vdupq_n_s16(left - right);
            let off_ver = vdupq_n_s16((h - 1 - y) as i16);
            let w_ver = vdupq_n_s16(weights[y] as i16);
            let row = &mut dst[off..off + w];
            let mut x = 0usize;
            while x + 8 <= w {
                let above = load_u8x8_i16(tl.as_ptr().add(o + 1 + x));
                let wx = load_u8x8_i16(weights.as_ptr().add(x));
                let d = dist8((w - 1 - x) as i16);

                let mut pred_ver = vaddq_s16(
                    bottom_v,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(above, bottom_v), off_ver), rnd_ver),
                        bhl2,
                    ),
                );
                let mut pred_hor = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff_hor, d), rnd_hor), bwl2),
                );
                pred_ver = vaddq_s16(
                    pred_ver,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(above, pred_ver), w_ver), add32),
                        6,
                    ),
                );
                pred_hor = vaddq_s16(
                    pred_hor,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred_hor), wx), add32),
                        6,
                    ),
                );
                let out = sra_i16(vaddq_s16(vaddq_s16(pred_ver, pred_hor), one), 1);
                store_i16x8_u8(row.as_mut_ptr().add(x), out);
                x += 8;
            }
            while x < w {
                let above = tl[o + 1 + x] as i32;
                let mul_ver = (above - bottom as i32) * (h as i32 - 1 - y as i32);
                let mul_hor = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
                let mut pred_ver = bottom as i32 + ((mul_ver + (h >> 1) as i32) >> bhl2);
                let mut pred_hor = right as i32 + ((mul_hor + (w >> 1) as i32) >> bwl2);
                pred_ver += ((above - pred_ver) * weights[y] as i32 + 32) >> 6;
                pred_hor += ((left as i32 - pred_hor) * weights[x] as i32 + 32) >> 6;
                row[x] = ((pred_ver + pred_hor + 1) >> 1) as u8;
                x += 1;
            }
            off += stride;
        }
    }
}
