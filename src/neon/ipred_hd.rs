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

use crate::dip_tables::DIP_WEIGHTS;
use crate::intops::ulog2;
use crate::levels::{ANGLE_IBP_FLAG, ANGLE_MULTI_MRL_FLAG};
use crate::tables::SM_WEIGHTS;

#[inline]
#[target_feature(enable = "neon")]
fn avg_pred_hbd_neon(dst: &mut [u16], stride: usize, tmp: &[u16], w: usize, h: usize) {
    for y in 0..h {
        let dst_row = &mut dst[y * stride..y * stride + w];
        let tmp_row = &tmp[y * 64..y * 64 + w];
        let (d8, drem) = dst_row.as_chunks_mut::<8>();
        for (d, t) in d8.iter_mut().zip(tmp_row.as_chunks::<8>().0.iter()) {
            let a = unsafe { vld1q_u16(d.as_ptr()) };
            let b = unsafe { vld1q_u16(t.as_ptr()) };
            unsafe { vst1q_u16(d.as_mut_ptr(), vrhaddq_u16(a, b)) };
        }
        let base = d8.len() * 8;
        for (i, d) in drem.iter_mut().enumerate() {
            *d = ((*d as u32 + tmp_row[base + i] as u32 + 1) >> 1) as u16;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ibp_blend_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tmp: &[u16],
    w: usize,
    h: usize,
    inv: bool,
    weights: &[[u8; 16]; 16],
    bitdepth_max: u16,
) {
    let x_shift = w >> 5;
    let y_shift = h >> 5;
    let c128 = vdupq_n_u32(128);
    let c64 = vdupq_n_u32(64);
    let maxv = vdupq_n_u16(bitdepth_max);
    let mut wrow = [0u16; 128];
    for y in 0..h {
        let wy = y >> y_shift;
        for x in 0..w {
            let wx = x >> x_shift;
            wrow[x] = (if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            }) as u16;
        }
        let dst_row = &mut dst[y * stride..y * stride + w];
        let tmp_row = &tmp[y * 64..y * 64 + w];
        let mut x = 0usize;
        while x + 8 <= w {
            let wv = unsafe { vld1q_u16(wrow[x..].as_ptr()) };
            let dv = unsafe { vld1q_u16(dst_row[x..].as_ptr()) };
            let tv = unsafe { vld1q_u16(tmp_row[x..].as_ptr()) };
            let wl = vmovl_u16(vget_low_u16(wv));
            let wh = vmovl_u16(vget_high_u16(wv));
            let dl = vmovl_u16(vget_low_u16(dv));
            let dh = vmovl_u16(vget_high_u16(dv));
            let tl = vmovl_u16(vget_low_u16(tv));
            let th = vmovl_u16(vget_high_u16(tv));
            let rl = vshrq_n_u32::<7>(vaddq_u32(
                vaddq_u32(vmulq_u32(tl, vsubq_u32(c128, wl)), vmulq_u32(dl, wl)),
                c64,
            ));
            let rh = vshrq_n_u32::<7>(vaddq_u32(
                vaddq_u32(vmulq_u32(th, vsubq_u32(c128, wh)), vmulq_u32(dh, wh)),
                c64,
            ));
            let packed = vminq_u16(vcombine_u16(vqmovn_u32(rl), vqmovn_u32(rh)), maxv);
            unsafe { vst1q_u16(dst_row[x..].as_mut_ptr(), packed) };
            x += 8;
        }
        while x < w {
            let wx = x >> x_shift;
            let weight = (if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            }) as u32;
            let t = tmp_row[x] as u32;
            let d = dst_row[x] as u32;
            dst_row[x] =
                ((t * (128 - weight) + d * weight + 64) >> 7).min(bitdepth_max as u32) as u16;
            x += 1;
        }
    }
}

#[inline(always)]
fn load_u16x8(a: &[u16; 8]) -> uint16x8_t {
    unsafe { vld1q_u16(a.as_ptr()) }
}

#[inline(always)]
fn store_u16x8(a: &mut [u16; 8], v: uint16x8_t) {
    unsafe { vst1q_u16(a.as_mut_ptr(), v) };
}

#[target_feature(enable = "neon")]
fn sum_u16_neon(s: &[u16]) -> u32 {
    let mut acc = vdupq_n_u32(0);
    let (chunks, rem) = s.as_chunks::<8>();
    for c in chunks.iter() {
        let v = load_u16x8(c);
        acc = vaddq_u32(acc, vaddl_u16(vget_low_u16(v), vget_high_u16(v)));
    }
    let mut total = vaddvq_u32(acc);
    for &v in rem {
        total += v as u32;
    }
    total
}

#[target_feature(enable = "neon")]
fn splat_fill_neon(dst: &mut [u16], stride: usize, off: usize, w: usize, h: usize, dc: u16) {
    let v = vdupq_n_u16(dc);
    let mut p = off;
    for _ in 0..h {
        let (chunks, rem) = dst[p..p + w].as_chunks_mut::<8>();
        for c in chunks.iter_mut() {
            store_u16x8(c, v);
        }
        rem.fill(dc);
        p += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_v_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_v_hbd_scalar(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (w + h) * 2 + 1;
        let top1 = &tl[o + 1..o + 1 + w];
        let top2 = &tl[o + 1 + e_stride..o + 1 + e_stride + w];
        let (chunks, rem) = dst[..w].as_chunks_mut::<8>();
        for ((d, a), b) in chunks
            .iter_mut()
            .zip(top1.as_chunks::<8>().0.iter())
            .zip(top2.as_chunks::<8>().0.iter())
        {
            store_u16x8(d, vrhaddq_u16(load_u16x8(a), load_u16x8(b)));
        }
        let base = chunks.len() * 8;
        for (i, d) in rem.iter_mut().enumerate() {
            let x = base + i;
            *d = ((top1[x] as u32 + top2[x] as u32 + 1) >> 1) as u16;
        }
    } else {
        dst[..w].copy_from_slice(&tl[o + 1..o + 1 + w]);
    }
    let mut off = stride;
    for _ in 1..h {
        dst.copy_within(0..w, off);
        off += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_h_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_h_hbd_scalar(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    let e_stride = (w + h) * 2 + 1;
    let mrl = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let mut off = 0usize;
    for y in 0..h {
        let v = if mrl {
            ((tl[o - 1 - y] as u32 + tl[o + e_stride - 1 - y] as u32 + 1) >> 1) as u16
        } else {
            tl[o - 1 - y]
        };
        let vv = vdupq_n_u16(v);
        let row = &mut dst[off..off + w];
        let (chunks, rem) = row.as_chunks_mut::<8>();
        for c in chunks.iter_mut() {
            store_u16x8(c, vv);
        }
        rem.fill(v);
        off += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_dc_128_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    splat_fill_neon(dst, stride, 0, w, h, (bitdepth_max + 1) >> 1);
}

#[target_feature(enable = "neon")]
fn ipred_dc_top_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        crate::ipred_dispatch::ipred_dc_top_hbd_scalar(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            bitdepth_max,
        );
        return;
    }
    let dc = (((w >> 1) as u32 + sum_u16_neon(&tl[o + 1..o + 1 + w]))
        >> (w as u32).trailing_zeros()) as u16;
    splat_fill_neon(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "neon")]
fn ipred_dc_left_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        crate::ipred_dispatch::ipred_dc_left_hbd_scalar(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            bitdepth_max,
        );
        return;
    }
    let dc =
        (((h >> 1) as u32 + sum_u16_neon(&tl[o - h..o])) >> (h as u32).trailing_zeros()) as u16;
    splat_fill_neon(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "neon")]
fn ipred_dc_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        crate::ipred_dispatch::ipred_dc_hbd_scalar(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    let n = (w + h) as u32;
    let sum = sum_u16_neon(&tl[o + 1..o + 1 + w]) + sum_u16_neon(&tl[o - h..o]);
    let dc = if n & (n - 1) == 0 {
        (sum + w as u32) >> n.trailing_zeros()
    } else {
        crate::ipred::fast_div32_dc(sum, n).min(bitdepth_max as u32)
    } as u16;
    splat_fill_neon(dst, stride, 0, w, h, dc);
}

#[inline]
#[target_feature(enable = "neon")]
fn sra_i32_neon(v: int32x4_t, shift: i32) -> int32x4_t {
    vshlq_s32(v, vdupq_n_s32(-shift))
}

#[inline]
#[target_feature(enable = "neon")]
fn dist4_hbd_neon(base: i32) -> int32x4_t {
    vsubq_s32(vdupq_n_s32(base), setr_i32x4_neon(0, 1, 2, 3))
}

#[inline]
#[target_feature(enable = "neon")]
fn weights4_hbd_neon(w: &[u8]) -> int32x4_t {
    debug_assert!(w.len() >= 4);
    setr_i32x4_neon(w[0] as i32, w[1] as i32, w[2] as i32, w[3] as i32)
}

#[target_feature(enable = "neon")]
fn ipred_smooth_v_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    if w < 4 {
        crate::ipred_dispatch::ipred_smooth_v_hbd_scalar(dst, stride, tl, o, w, h, bitdepth_max);
        return;
    }

    let bhl2 = ulog2(h as u32) as i32;
    let rnd = vdupq_n_s32((h >> 1) as i32);
    let scale = (w * h >= 64) as usize + (w * h > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom_s = tl[o - h - 1] as i32;
    let bottom = vdupq_n_s32(bottom_s);
    let add32 = vdupq_n_s32(32);
    let mut off = 0usize;
    for y in 0..h {
        let off_y = vdupq_n_s32((h - 1 - y) as i32);
        let w_ver = vdupq_n_s32(weights[y] as i32);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let (chunks, rem) = row.as_chunks_mut::<4>();
        for (d, t) in chunks.iter_mut().zip(top_src.as_chunks::<4>().0.iter()) {
            let above = load_u16x4_i32_neon(t);
            let pred = vaddq_s32(
                bottom,
                sra_i32_neon(
                    vaddq_s32(vmulq_s32(vsubq_s32(above, bottom), off_y), rnd),
                    bhl2,
                ),
            );
            let out = vaddq_s32(
                pred,
                vshrq_n_s32::<6>(vaddq_s32(vmulq_s32(vsubq_s32(above, pred), w_ver), add32)),
            );
            store_i32x4_u16_max_neon(d, out, bitdepth_max);
        }
        let base = chunks.len() * 4;
        for (i, d) in rem.iter_mut().enumerate() {
            let x = base + i;
            let above = tl[o + 1 + x] as i32;
            let pred = bottom_s
                + (((above - bottom_s) * (h as i32 - 1 - y as i32) + (h >> 1) as i32) >> bhl2);
            *d = (pred + (((above - pred) * weights[y] as i32 + 32) >> 6))
                .clamp(0, bitdepth_max as i32) as u16;
        }
        off += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_smooth_h_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    if w < 4 {
        crate::ipred_dispatch::ipred_smooth_h_hbd_scalar(dst, stride, tl, o, w, h, bitdepth_max);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let rnd = vdupq_n_s32((w >> 1) as i32);
    let scale = (w * h >= 64) as usize + (w * h > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right_s = tl[o + w + 1] as i32;
    let right = vdupq_n_s32(right_s);
    let add32 = vdupq_n_s32(32);
    let mut off = 0usize;
    for y in 0..h {
        let left_s = tl[o - 1 - y] as i32;
        let left = vdupq_n_s32(left_s);
        let diff = vdupq_n_s32(left_s - right_s);
        let row = &mut dst[off..off + w];
        let (chunks, rem) = row.as_chunks_mut::<4>();
        for (ci, (d, wx)) in chunks
            .iter_mut()
            .zip(weights[..w].as_chunks::<4>().0.iter())
            .enumerate()
        {
            let x0 = (w - 1 - ci * 4) as i32;
            let pred = vaddq_s32(
                right,
                sra_i32_neon(vaddq_s32(vmulq_s32(diff, dist4_hbd_neon(x0)), rnd), bwl2),
            );
            let out = vaddq_s32(
                pred,
                vshrq_n_s32::<6>(vaddq_s32(
                    vmulq_s32(vsubq_s32(left, pred), weights4_hbd_neon(wx)),
                    add32,
                )),
            );
            store_i32x4_u16_max_neon(d, out, bitdepth_max);
        }
        let base = chunks.len() * 4;
        for (i, d) in rem.iter_mut().enumerate() {
            let x = base + i;
            let pred = right_s
                + (((left_s - right_s) * (w as i32 - 1 - x as i32) + (w >> 1) as i32) >> bwl2);
            *d = (pred + (((left_s - pred) * weights[x] as i32 + 32) >> 6))
                .clamp(0, bitdepth_max as i32) as u16;
        }
        off += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_smooth_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    if w < 4 {
        crate::ipred_dispatch::ipred_smooth_hbd_scalar(dst, stride, tl, o, w, h, bitdepth_max);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let bhl2 = ulog2(h as u32) as i32;
    let rnd_ver = vdupq_n_s32((h >> 1) as i32);
    let rnd_hor = vdupq_n_s32((w >> 1) as i32);
    let scale = (w * h >= 64) as usize + (w * h > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right_s = tl[o + w + 1] as i32;
    let bottom_s = tl[o - h - 1] as i32;
    let right = vdupq_n_s32(right_s);
    let bottom = vdupq_n_s32(bottom_s);
    let add32 = vdupq_n_s32(32);
    let one = vdupq_n_s32(1);
    let mut off = 0usize;
    for y in 0..h {
        let left_s = tl[o - 1 - y] as i32;
        let left = vdupq_n_s32(left_s);
        let diff_hor = vdupq_n_s32(left_s - right_s);
        let off_ver = vdupq_n_s32((h - 1 - y) as i32);
        let w_ver = vdupq_n_s32(weights[y] as i32);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let (chunks, rem) = row.as_chunks_mut::<4>();
        for (ci, ((d, t), wx)) in chunks
            .iter_mut()
            .zip(top_src.as_chunks::<4>().0.iter())
            .zip(weights[..w].as_chunks::<4>().0.iter())
            .enumerate()
        {
            let above = load_u16x4_i32_neon(t);
            let pv = vaddq_s32(
                bottom,
                sra_i32_neon(
                    vaddq_s32(vmulq_s32(vsubq_s32(above, bottom), off_ver), rnd_ver),
                    bhl2,
                ),
            );
            let x0 = (w - 1 - ci * 4) as i32;
            let ph = vaddq_s32(
                right,
                sra_i32_neon(
                    vaddq_s32(vmulq_s32(diff_hor, dist4_hbd_neon(x0)), rnd_hor),
                    bwl2,
                ),
            );
            let pv = vaddq_s32(
                pv,
                vshrq_n_s32::<6>(vaddq_s32(vmulq_s32(vsubq_s32(above, pv), w_ver), add32)),
            );
            let ph = vaddq_s32(
                ph,
                vshrq_n_s32::<6>(vaddq_s32(
                    vmulq_s32(vsubq_s32(left, ph), weights4_hbd_neon(wx)),
                    add32,
                )),
            );
            store_i32x4_u16_max_neon(
                d,
                vshrq_n_s32::<1>(vaddq_s32(vaddq_s32(pv, ph), one)),
                bitdepth_max,
            );
        }
        let base = chunks.len() * 4;
        for (i, d) in rem.iter_mut().enumerate() {
            let x = base + i;
            let above = tl[o + 1 + x] as i32;
            let mut pv = bottom_s
                + (((above - bottom_s) * (h as i32 - 1 - y as i32) + (h >> 1) as i32) >> bhl2);
            let mut ph = right_s
                + (((left_s - right_s) * (w as i32 - 1 - x as i32) + (w >> 1) as i32) >> bwl2);
            pv += ((above - pv) * weights[y] as i32 + 32) >> 6;
            ph += ((left_s - ph) * weights[x] as i32 + 32) >> 6;
            *d = ((pv + ph + 1) >> 1).clamp(0, bitdepth_max as i32) as u16;
        }
        off += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_paeth_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    if w < 4 {
        crate::ipred_dispatch::ipred_paeth_hbd_scalar(dst, stride, tl, o, w, h, bitdepth_max);
        return;
    }

    let topleft_s = tl[o] as i32;
    let topleft = vdupq_n_s32(topleft_s);
    let mut off = 0usize;
    for y in 0..h {
        let left_s = tl[o - 1 - y] as i32;
        let left = vdupq_n_s32(left_s);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let (chunks, rem) = row.as_chunks_mut::<4>();
        for (d, t) in chunks.iter_mut().zip(top_src.as_chunks::<4>().0.iter()) {
            let top = load_u16x4_i32_neon(t);
            let base = vsubq_s32(vaddq_s32(left, top), topleft);
            let ld = vabsq_s32(vsubq_s32(left, base));
            let td = vabsq_s32(vsubq_s32(top, base));
            let tld = vabsq_s32(vsubq_s32(topleft, base));
            let left_mask = vandq_u32(vcleq_s32(ld, td), vcleq_s32(ld, tld));
            let top_mask = vandq_u32(vmvnq_u32(left_mask), vcleq_s32(td, tld));
            let inner = vbslq_s32(top_mask, top, topleft);
            store_i32x4_u16_max_neon(d, vbslq_s32(left_mask, left, inner), bitdepth_max);
        }
        let base_x = chunks.len() * 4;
        for (i, d) in rem.iter_mut().enumerate() {
            let top_s = tl[o + 1 + base_x + i] as i32;
            let base = left_s + top_s - topleft_s;
            let ld = (left_s - base).abs();
            let td = (top_s - base).abs();
            let tld = (topleft_s - base).abs();
            *d = if ld <= td && ld <= tld {
                left_s
            } else if td <= tld {
                top_s
            } else {
                topleft_s
            } as u16;
        }
        off += stride;
    }
}

pub(crate) fn ipred_v_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_v_hbd_neon_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_h_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_h_hbd_neon_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_hbd_neon_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_top_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_top_hbd_neon_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_left_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_left_hbd_neon_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_128_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_128_hbd_neon_impl(dst, stride, w, h, bitdepth_max) }
}

pub(crate) fn ipred_paeth_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    unsafe { ipred_paeth_hbd_neon_impl(dst, stride, tl, o, w, h, bitdepth_max) }
}
pub(crate) fn ipred_smooth_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    unsafe { ipred_smooth_hbd_neon_impl(dst, stride, tl, o, w, h, bitdepth_max) }
}
pub(crate) fn ipred_smooth_v_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    unsafe { ipred_smooth_v_hbd_neon_impl(dst, stride, tl, o, w, h, bitdepth_max) }
}
pub(crate) fn ipred_smooth_h_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    unsafe { ipred_smooth_h_hbd_neon_impl(dst, stride, tl, o, w, h, bitdepth_max) }
}

#[inline(always)]
fn load_u16x4_i32_neon(s: &[u16]) -> int32x4_t {
    debug_assert!(s.len() >= 4);
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(s.as_ptr()))) }
}

#[inline(always)]
fn setr_i32x4_neon(a: i32, b: i32, c: i32, d: i32) -> int32x4_t {
    let v = [a, b, c, d];
    unsafe { vld1q_s32(v.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4_u16_max_neon(a: &mut [u16], v: int32x4_t, bitdepth_max: u16) {
    debug_assert!(a.len() >= 4);
    let v = vminq_s32(
        vmaxq_s32(v, vdupq_n_s32(0)),
        vdupq_n_s32(bitdepth_max as i32),
    );
    unsafe { vst1_u16(a.as_mut_ptr(), vqmovun_s32(v)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn dr_filter4_hbd_neon(
    f: &crate::ipred::DrFilter4Tap,
    bitdepth_max: u16,
    a0: int32x4_t,
    a1: int32x4_t,
    a2: int32x4_t,
    a3: int32x4_t,
) -> int32x4_t {
    let acc = vaddq_s32(
        vaddq_s32(vmulq_n_s32(a0, f.a as i32), vmulq_n_s32(a1, f.b as i32)),
        vaddq_s32(vmulq_n_s32(a2, f.c as i32), vmulq_n_s32(a3, f.d as i32)),
    );
    let v = vshrq_n_s32::<7>(vaddq_s32(acc, vdupq_n_s32(64)));
    vminq_s32(
        vmaxq_s32(v, vdupq_n_s32(0)),
        vdupq_n_s32(bitdepth_max as i32),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn z1_luma_row_hbd_neon(
    filt: &[u16],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u16,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u16],
    w: usize,
    bitdepth_max: u16,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let base_const = (top_off as i32 + base0) as usize;
    let mut x = 0usize;
    while x + 4 <= n_filter {
        let bi = base_const + x;
        let v = dr_filter4_hbd_neon(
            f,
            bitdepth_max,
            load_u16x4_i32_neon(&filt[bi - 1..]),
            load_u16x4_i32_neon(&filt[bi..]),
            load_u16x4_i32_neon(&filt[bi + 1..]),
            load_u16x4_i32_neon(&filt[bi + 2..]),
        );
        store_i32x4_u16_max_neon(&mut dst_row[x..], v, bitdepth_max);
        x += 4;
    }
    while x < n_filter {
        let bi = base_const + x;
        let v = f.a as i32 * filt[bi - 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi + 1] as i32
            + f.d as i32 * filt[bi + 2] as i32;
        dst_row[x] = ((v + 64) >> 7).clamp(0, bitdepth_max as i32) as u16;
        x += 1;
    }
    dst_row[n_filter..w].fill(fill);
}

#[inline]
#[target_feature(enable = "neon")]
fn z1_chroma_row_hbd_neon(
    filt: &[u16],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u16,
    shift: usize,
    dst_row: &mut [u16],
    w: usize,
    bitdepth_max: u16,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let iw = vdupq_n_s32((32 - shift) as i32);
    let sw = vdupq_n_s32(shift as i32);
    let rnd = vdupq_n_s32(16);
    let base_const = (top_off as i32 + base0) as usize;
    let mut x = 0usize;
    while x + 4 <= n_filter {
        let bi = base_const + x;
        let a = load_u16x4_i32_neon(&filt[bi..]);
        let b = load_u16x4_i32_neon(&filt[bi + 1..]);
        let v = vshrq_n_s32::<5>(vaddq_s32(
            vaddq_s32(vmulq_s32(a, iw), vmulq_s32(b, sw)),
            rnd,
        ));
        store_i32x4_u16_max_neon(&mut dst_row[x..], v, bitdepth_max);
        x += 4;
    }
    while x < n_filter {
        let bi = base_const + x;
        let v = (32 - shift as i32) * filt[bi] as i32 + shift as i32 * filt[bi + 1] as i32;
        dst_row[x] = ((v + 16) >> 5).clamp(0, bitdepth_max as i32) as u16;
        x += 1;
    }
    dst_row[n_filter..w].fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn ipred_z1_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![0u16; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z1_hbd_neon_impl(
            &mut tmp,
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        ipred_z1_hbd_neon_impl(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        avg_pred_hbd_neon(dst, stride, &tmp, w, h);
        return;
    }
    if enable_ibp {
        let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
        let mode_idx = (10 - (a >> 3)).min(6) as usize;
        let mut tmp = vec![0u16; 64 * 64];
        ipred_z1_hbd_neon_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle & !ANGLE_IBP_FLAG,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        ipred_z3_hbd_neon_impl(
            &mut tmp,
            64,
            tl,
            o,
            w,
            h,
            (180 + a) | angle_flags,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        ibp_blend_hbd_neon(
            dst,
            stride,
            &tmp,
            w,
            h,
            false,
            &ibp_weights[mode_idx],
            bitdepth_max,
        );
        return;
    }
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[a as usize] as i32;
    let max_base_x = (w + h) as i32 - 1 + (mrl_idx as i32 * 2);
    let mut filt = [0u16; 141];
    let top_off = 2usize + mrl_idx;
    let sz = 1 + mrl_idx + w + h + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, 90 - a, is_sm_t)
    } else {
        0
    };
    if str > 0 {
        crate::ipred::filter_edge(
            &mut filt[1..],
            sz,
            1,
            sz as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz + 1];
    let mut ypos = dx * (1 + mrl_idx as i32);
    for y in 0..h {
        let base0 = ypos >> 6;
        let fill = filt[top_off + max_base_x as usize];
        if base0 > max_base_x {
            for row in dst.chunks_mut(stride).take(h).skip(y) {
                row[..w].fill(fill);
            }
            break;
        }
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let dst_row = &mut dst[y * stride..y * stride + w];
        if is_luma {
            z1_luma_row_hbd_neon(
                &filt,
                top_off,
                base0,
                max_base_x,
                fill,
                f,
                dst_row,
                w,
                bitdepth_max,
            );
        } else {
            z1_chroma_row_hbd_neon(
                &filt,
                top_off,
                base0,
                max_base_x,
                fill,
                shift,
                dst_row,
                w,
                bitdepth_max,
            );
        }
        ypos += dx;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn z3_luma_col_hbd_neon(
    filt: &[u16],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u16,
    f: &crate::ipred::DrFilter4Tap,
    col: &mut [u16],
    h: usize,
    bitdepth_max: u16,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let lob = left_off as i32 - base0;
    let mut y = 0usize;
    while y + 4 <= n_filter {
        let bi = lob - y as i32;
        let v = dr_filter4_hbd_neon(
            f,
            bitdepth_max,
            setr_i32x4_neon(
                filt[(bi + 1) as usize] as i32,
                filt[bi as usize] as i32,
                filt[(bi - 1) as usize] as i32,
                filt[(bi - 2) as usize] as i32,
            ),
            setr_i32x4_neon(
                filt[bi as usize] as i32,
                filt[(bi - 1) as usize] as i32,
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
            ),
            setr_i32x4_neon(
                filt[(bi - 1) as usize] as i32,
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
                filt[(bi - 4) as usize] as i32,
            ),
            setr_i32x4_neon(
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
                filt[(bi - 4) as usize] as i32,
                filt[(bi - 5) as usize] as i32,
            ),
        );
        store_i32x4_u16_max_neon(&mut col[y..], v, bitdepth_max);
        y += 4;
    }
    while y < n_filter {
        let bi = (lob - y as i32) as usize;
        let v = f.a as i32 * filt[bi + 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi - 1] as i32
            + f.d as i32 * filt[bi - 2] as i32;
        col[y] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
        y += 1;
    }
    col[n_filter..h].fill(fill);
}

#[inline]
#[target_feature(enable = "neon")]
fn z3_chroma_col_hbd_neon(
    filt: &[u16],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u16,
    shift: usize,
    col: &mut [u16],
    h: usize,
    bitdepth_max: u16,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let iw = vdupq_n_s32((32 - shift) as i32);
    let sw = vdupq_n_s32(shift as i32);
    let rnd = vdupq_n_s32(16);
    let lob = left_off as i32 - base0;
    let mut y = 0usize;
    while y + 4 <= n_filter {
        let bi = lob - y as i32;
        let a = setr_i32x4_neon(
            filt[bi as usize] as i32,
            filt[(bi - 1) as usize] as i32,
            filt[(bi - 2) as usize] as i32,
            filt[(bi - 3) as usize] as i32,
        );
        let b = setr_i32x4_neon(
            filt[(bi - 1) as usize] as i32,
            filt[(bi - 2) as usize] as i32,
            filt[(bi - 3) as usize] as i32,
            filt[(bi - 4) as usize] as i32,
        );
        let v = vshrq_n_s32::<5>(vaddq_s32(
            vaddq_s32(vmulq_s32(a, iw), vmulq_s32(b, sw)),
            rnd,
        ));
        store_i32x4_u16_max_neon(&mut col[y..], v, bitdepth_max);
        y += 4;
    }
    while y < n_filter {
        let bi = (lob - y as i32) as usize;
        let v = (32 - shift as i32) * filt[bi] as i32 + shift as i32 * filt[bi - 1] as i32;
        col[y] = ((v + 16) >> 5).clamp(0, bitdepth_max as i32) as u16;
        y += 1;
    }
    col[n_filter..h].fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn ipred_z3_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![0u16; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z3_hbd_neon_impl(
            &mut tmp,
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        ipred_z3_hbd_neon_impl(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        avg_pred_hbd_neon(dst, stride, &tmp, w, h);
        return;
    }
    if enable_ibp {
        if h > 64 {
            return crate::ipred_dispatch::ipred_z3_hbd_scalar(
                dst,
                stride,
                tl,
                o,
                w,
                h,
                angle,
                max_width,
                max_height,
                ibp_weights,
                bitdepth_max,
            );
        }
        let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
        let mode_idx = ((a - 183) >> 3).min(6) as usize;
        let mut tmp = vec![0u16; 64 * 64];
        ipred_z3_hbd_neon_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle & !ANGLE_IBP_FLAG,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        ipred_z1_hbd_neon_impl(
            &mut tmp,
            64,
            tl,
            o,
            w,
            h,
            (a - 180) | angle_flags,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
        ibp_blend_hbd_neon(
            dst,
            stride,
            &tmp,
            w,
            h,
            true,
            &ibp_weights[mode_idx],
            bitdepth_max,
        );
        return;
    }
    if h > 64 {
        return crate::ipred_dispatch::ipred_z3_hbd_scalar(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        );
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let dy = crate::tables::DR_INTRA_DERIVATIVE[(270 - a) as usize] as i32;
    let max_base_y = (w + h) as i32 - 1 + (mrl_idx as i32 * 2);
    let mut filt = [0u16; 141];
    let left_off = 1 + w + h + mrl_idx * 2;
    let sz = 1 + mrl_idx + w + h + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, a - 180, is_sm_l)
    } else {
        0
    };
    if str > 0 {
        crate::ipred::filter_edge(
            &mut filt[2..],
            sz,
            h as i32 - max_height,
            sz as i32 - 1,
            &tl[o + 1 - sz..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[2..2 + sz].copy_from_slice(&tl[o + 1 - sz..o + 1]);
    }
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];
    let mut col = [0u16; 128];
    let mut ypos = dy * (1 + mrl_idx as i32);
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let base0 = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        if is_luma {
            z3_luma_col_hbd_neon(
                &filt,
                left_off,
                base0,
                max_base_y,
                fill,
                f,
                &mut col,
                h,
                bitdepth_max,
            );
        } else {
            z3_chroma_col_hbd_neon(
                &filt,
                left_off,
                base0,
                max_base_y,
                fill,
                shift,
                &mut col,
                h,
                bitdepth_max,
            );
        }
        for (y, &c) in col[..h].iter().enumerate() {
            dst[y * stride + x] = c;
        }
        ypos += dy;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn z2_top_span_hbd_neon(
    filt: &[u16],
    top_off: usize,
    mut xpos: i32,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u16],
    x_start: usize,
    w: usize,
    bitdepth_max: u16,
) {
    let mut x = x_start;
    while x + 4 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 1 < 0 || ti0 + 8 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 1) as usize;
        let v = dr_filter4_hbd_neon(
            f,
            bitdepth_max,
            load_u16x4_i32_neon(&filt[sa..]),
            load_u16x4_i32_neon(&filt[sa + 1..]),
            load_u16x4_i32_neon(&filt[sa + 2..]),
            load_u16x4_i32_neon(&filt[sa + 3..]),
        );
        store_i32x4_u16_max_neon(&mut dst_row[x..], v, bitdepth_max);
        x += 4;
        xpos += 64 * 4;
    }
    while x < w {
        let base_x = xpos >> 6;
        let ti = top_off as i32 + base_x;
        let v = f.a as i32 * filt[(ti + 1) as usize] as i32
            + f.b as i32 * filt[(ti + 2) as usize] as i32
            + f.c as i32 * filt[(ti + 3) as usize] as i32
            + f.d as i32 * filt[(ti + 4) as usize] as i32;
        dst_row[x] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
        x += 1;
        xpos += 64;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn z2_top_span_chroma_hbd_neon(
    filt: &[u16],
    top_off: usize,
    mut xpos: i32,
    shift: usize,
    dst_row: &mut [u16],
    x_start: usize,
    w: usize,
    bitdepth_max: u16,
) {
    let iw = vdupq_n_s32((32 - shift) as i32);
    let sw = vdupq_n_s32(shift as i32);
    let rnd = vdupq_n_s32(16);
    let mut x = x_start;
    while x + 4 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 2 < 0 || ti0 + 7 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 2) as usize;
        let a = load_u16x4_i32_neon(&filt[sa..]);
        let b = load_u16x4_i32_neon(&filt[sa + 1..]);
        let v = vshrq_n_s32::<5>(vaddq_s32(
            vaddq_s32(vmulq_s32(a, iw), vmulq_s32(b, sw)),
            rnd,
        ));
        store_i32x4_u16_max_neon(&mut dst_row[x..], v, bitdepth_max);
        x += 4;
        xpos += 64 * 4;
    }
    while x < w {
        let base_x = xpos >> 6;
        let ti = top_off as i32 + base_x;
        let v = (32 - shift as i32) * filt[(ti + 2) as usize] as i32
            + shift as i32 * filt[(ti + 3) as usize] as i32;
        dst_row[x] = ((v + 16) >> 5).clamp(0, bitdepth_max as i32) as u16;
        x += 1;
        xpos += 64;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn ipred_z2_hbd_neon_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: u16,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![0u16; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z2_hbd_neon_impl(
            &mut tmp,
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
            bitdepth_max,
        );
        ipred_z2_hbd_neon_impl(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
            bitdepth_max,
        );
        avg_pred_hbd_neon(dst, stride, &tmp, w, h);
        return;
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let dy = crate::tables::DR_INTRA_DERIVATIVE[(a - 90) as usize] as i32;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[(180 - a) as usize] as i32;
    let mut filt = [0u16; 72];
    let top_off = mrl_idx;
    let sz_t = 1 + w + mrl_idx;
    let str_t = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, a - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        crate::ipred::filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz_t as i32,
            str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&tl[o..o + sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];
    let mut filt2 = [0u16; 72];
    let left_off: usize = h + 2;
    let sz_l = 1 + h + mrl_idx;
    let str_l = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, 180 - a, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        crate::ipred::filter_edge(
            &mut filt2[1..],
            sz_l,
            h as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - (h + mrl_idx)..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - (h + mrl_idx)..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];
    for y in 0..h {
        let ypos = (y + 1) as i32;
        let mut xpos = -(ypos + mrl_idx as i32) * dx;
        let mut x = 0usize;
        let dst_row = &mut dst[y * stride..y * stride + w];
        while x < w && xpos < -(64 * (1 + mrl_idx as i32)) {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - (xpos_l + mrl_idx as i32) * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            if is_luma {
                let f = &crate::ipred::DR_INTERP_FILTER[shift];
                let v = f.a as i32 * filt2[bi - 1] as i32
                    + f.b as i32 * filt2[bi - 2] as i32
                    + f.c as i32 * filt2[bi - 3] as i32
                    + f.d as i32 * filt2[bi - 4] as i32;
                dst_row[x] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
            } else {
                let v = (32 - shift as i32) * filt2[bi - 2] as i32
                    + shift as i32 * filt2[bi - 3] as i32;
                dst_row[x] = ((v + 16) >> 5).clamp(0, bitdepth_max as i32) as u16;
            }
            x += 1;
            xpos += 64;
        }
        if x < w {
            let shift = ((xpos & 0x3F) >> 1) as usize;
            if is_luma {
                let f = &crate::ipred::DR_INTERP_FILTER[shift];
                z2_top_span_hbd_neon(&filt, top_off, xpos, f, dst_row, x, w, bitdepth_max);
            } else {
                z2_top_span_chroma_hbd_neon(
                    &filt,
                    top_off,
                    xpos,
                    shift,
                    dst_row,
                    x,
                    w,
                    bitdepth_max,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    unsafe {
        ipred_z1_hbd_neon_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    unsafe {
        ipred_z3_hbd_neon_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
            bitdepth_max,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: u16,
) {
    unsafe {
        ipred_z2_hbd_neon_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            bitdepth_max,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn ipred_dip_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
    bitdepth_max: u16,
) {
    let trans = (mode & 16) != 0;
    let wd = width >> 2;
    let hd = height >> 2;
    let wl2 = ulog2(wd as u32);
    let hl2 = ulog2(hd as u32);
    let wrnd = width >> 3;
    let hrnd = height >> 3;
    let i_t: usize = if trans { 5 } else { 1 };
    let i_l: usize = if trans { 1 } else { 5 };
    let mut inp = [0i32; 11];
    inp[0] = tl[o] as i32;
    let mut in_sum = inp[0];

    let mut ti = o + 1;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..wd {
            sum += tl[ti] as i32;
            ti += 1;
        }
        inp[i_t + i] = (sum + wrnd as i32) >> wl2;
        in_sum += inp[i_t + i];
    }

    let mut li = o;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..hd {
            li -= 1;
            sum += tl[li] as i32;
        }
        inp[i_l + i] = (sum + hrnd as i32) >> hl2;
        in_sum += inp[i_l + i];
    }

    let mut sum = 0i32;
    for x in 0..wd {
        sum += tl[o + x + width + 1] as i32;
    }
    let idx_tr = if trans { 10 } else { 9 };
    inp[idx_tr] = (sum + wrnd as i32) >> wl2;
    in_sum += inp[idx_tr];

    sum = 0;
    for y in 0..hd {
        sum += tl[o - (y + height + 1)] as i32;
    }
    let idx_bl = if trans { 9 } else { 10 };
    inp[idx_bl] = (sum + hrnd as i32) >> hl2;
    in_sum += inp[idx_bl];

    let m = (mode & 7) as usize;
    let mut uwl2 = wl2 - 1;
    let mut dwl2 = 0i32;
    if uwl2 < 0 {
        dwl2 = -uwl2;
        uwl2 = 0;
    }
    let step_x = 1usize << uwl2;
    let dw = 1usize << dwl2;
    let mut uhl2 = hl2 - 1;
    let mut dhl2 = 0i32;
    if uhl2 < 0 {
        dhl2 = -uhl2;
        uhl2 = 0;
    }
    let step_y = 1usize << uhl2;
    let dh = 1usize << dhl2;
    let grid_h = 8usize >> dhl2;
    let grid_w = 8usize >> dwl2;

    let mut y = step_y - 1;
    for gy in 0..grid_h {
        let iy = gy * dh;
        let mut x = step_x - 1;
        let dst_row = &mut dst[y * stride..y * stride + width];
        for gx in 0..grid_w {
            let ix = gx * dw;
            let idx = if trans { ix * 8 + iy } else { iy * 8 + ix };
            let mut s = 0i32;
            let weights = &DIP_WEIGHTS[m][idx];
            for i in 0..11 {
                s += weights[i] as i32 * inp[i];
            }
            dst_row[x] = (((s + 2048) >> 12) - in_sum).clamp(0, bitdepth_max as i32) as u16;
            x += step_x;
        }
        y += step_y;
    }

    if step_x > 1 {
        y = step_y - 1;
        for _gy in 0..grid_h {
            let mut p1 = tl[o - (y + 1)] as i32;
            let mut x = 0usize;
            let dst_row = &mut dst[y * stride..y * stride + width];
            for _gx in 0..grid_w {
                let p0 = p1;
                p1 = dst_row[x + step_x - 1] as i32;
                for z in 0..step_x - 1 {
                    let z1 = (z + 1) as i32;
                    dst_row[x + z] = ((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2) as u16;
                }
                x += step_x;
            }
            y += step_y;
        }
    }

    if step_y > 1 {
        let mut p0_buf = [0u16; 128];
        let mut p1_buf = [0u16; 128];
        for gy in 0..grid_h {
            let base_y = gy * step_y;
            let sparse_y = base_y + step_y - 1;
            if gy == 0 {
                p0_buf[..width].copy_from_slice(&tl[o + 1..o + 1 + width]);
            } else {
                let prev = (base_y - 1) * stride;
                p0_buf[..width].copy_from_slice(&dst[prev..prev + width]);
            }
            let p1_off = sparse_y * stride;
            p1_buf[..width].copy_from_slice(&dst[p1_off..p1_off + width]);
            for z in 0..step_y - 1 {
                let z1 = (z + 1) as i32;
                let row_off = (base_y + z) * stride;
                let row = &mut dst[row_off..row_off + width];
                for x in 0..width {
                    row[x] = ((p0_buf[x] as i32 * (step_y as i32 - z1) + p1_buf[x] as i32 * z1)
                        >> uhl2) as u16;
                }
            }
        }
    }
}
