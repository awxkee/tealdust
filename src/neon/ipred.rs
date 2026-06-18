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
fn load_u8x8_i16_fixed(ptr: &[u8; 8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr.as_ptr()))) }
}

#[inline(always)]
fn store_i16x8_u8_fixed(ptr: &mut [u8; 8], v: int16x8_t) {
    unsafe { vst1_u8(ptr.as_mut_ptr(), vqmovun_s16(v)) };
}

/// Pack two `int16x8_t` lanes (saturating to u8) and store 16 bytes at once.
#[inline(always)]
fn store_i16x8x2_u8_fixed(ptr: &mut [u8; 16], lo: int16x8_t, hi: int16x8_t) {
    unsafe {
        vst1q_u8(
            ptr.as_mut_ptr(),
            vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi)),
        )
    };
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

    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        let top1 = &tl[o + 1..o + 1 + width];
        let top2 = &tl[o + 1 + e_stride..o + 1 + e_stride + width];
        let (dc, drem) = dst[..width].as_chunks_mut::<16>();
        for ((d, a), b) in dc
            .iter_mut()
            .zip(top1.as_chunks::<16>().0.iter())
            .zip(top2.as_chunks::<16>().0.iter())
        {
            store_u8x16_fixed(d, unsafe {
                vrhaddq_u8(load_u8x16_fixed(a), load_u8x16_fixed(b))
            });
        }
        let base_x = (width / 16) * 16;
        for (xi, d) in drem.iter_mut().enumerate() {
            let x = base_x + xi;
            let t1 = tl[o + 1 + x] as u16;
            let t2 = tl[o + 1 + e_stride + x] as u16;
            *d = ((t1 + t2 + 1) >> 1) as u8;
        }
    } else {
        let top = &tl[o + 1..o + 1 + width];
        dst[..width].copy_from_slice(top);
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
            let (c16, r16) = row.as_chunks_mut::<16>();
            for (dst, tl) in c16.iter_mut().zip(tl_src.as_chunks::<16>().0.iter()) {
                let above0 = load_u8x8_i16_fixed((&tl[..8]).try_into().unwrap());
                let above1 = load_u8x8_i16_fixed((&tl[8..]).try_into().unwrap());
                let mul0 = vmulq_s16(vsubq_s16(above0, bottom_v), off_y);
                let pred0 = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul0, rnd), bhl2));
                let adj0 = sra_i16(
                    vaddq_s16(vmulq_s16(vsubq_s16(above0, pred0), w_ver), add32),
                    6,
                );
                let mul1 = vmulq_s16(vsubq_s16(above1, bottom_v), off_y);
                let pred1 = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul1, rnd), bhl2));
                let adj1 = sra_i16(
                    vaddq_s16(vmulq_s16(vsubq_s16(above1, pred1), w_ver), add32),
                    6,
                );
                store_i16x8x2_u8_fixed(dst, vaddq_s16(pred0, adj0), vaddq_s16(pred1, adj1));
            }

            let done = c16.len() * 16;
            let (c8, r8) = r16.as_chunks_mut::<8>();
            for (dst, tl) in c8.iter_mut().zip(tl_src[done..].as_chunks::<8>().0.iter()) {
                let above = load_u8x8_i16_fixed(tl);
                let mul = vmulq_s16(vsubq_s16(above, bottom_v), off_y);
                let pred = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul, rnd), bhl2));
                let adj = sra_i16(
                    vaddq_s16(vmulq_s16(vsubq_s16(above, pred), w_ver), add32),
                    6,
                );
                store_i16x8_u8_fixed(dst, vaddq_s16(pred, adj));
            }

            let base_x = done + c8.len() * 8;
            for (xi, dst) in r8.iter_mut().enumerate() {
                let above = tl_src[base_x + xi] as i32;
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
            let (c16, r16) = row.as_chunks_mut::<16>();
            for (ci, (oc, wxc)) in c16
                .iter_mut()
                .zip(weights[..w].as_chunks::<16>().0.iter())
                .enumerate()
            {
                let x = ci * 16;
                let d_lo = dist8((w - 1 - x) as i16);
                let d_hi = dist8((w - 1 - x - 8) as i16);
                let wx_lo = load_u8x8_i16_fixed((&wxc[..8]).try_into().unwrap());
                let wx_hi = load_u8x8_i16_fixed((&wxc[8..]).try_into().unwrap());
                let pred_lo = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff, d_lo), rnd), bwl2),
                );
                let adj_lo = sra_i16(
                    vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred_lo), wx_lo), add32),
                    6,
                );
                let pred_hi = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff, d_hi), rnd), bwl2),
                );
                let adj_hi = sra_i16(
                    vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred_hi), wx_hi), add32),
                    6,
                );
                store_i16x8x2_u8_fixed(oc, vaddq_s16(pred_lo, adj_lo), vaddq_s16(pred_hi, adj_hi));
            }

            let done = c16.len() * 16;
            let (c8, r8) = r16.as_chunks_mut::<8>();
            for (ci, (oc, wxc)) in c8
                .iter_mut()
                .zip(weights[done..w].as_chunks::<8>().0.iter())
                .enumerate()
            {
                let x = done + ci * 8;
                let dvec = dist8((w - 1 - x) as i16);
                let wx = load_u8x8_i16_fixed(wxc);
                let pred = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff, dvec), rnd), bwl2),
                );
                let adj = sra_i16(vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred), wx), add32), 6);
                store_i16x8_u8_fixed(oc, vaddq_s16(pred, adj));
            }
            let base_x = done + c8.len() * 8;
            for (xi, oc) in r8.iter_mut().enumerate() {
                let x = base_x + xi;
                let mul = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
                let pred = right as i32 + ((mul + (w >> 1) as i32) >> bwl2);
                *oc = (pred + (((left as i32 - pred) * weights[x] as i32 + 32) >> 6)) as u8;
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
            let top_src = &tl[o + 1..o + 1 + w];
            let (c16, r16) = row.as_chunks_mut::<16>();
            for (ci, ((oc, t), wxc)) in c16
                .iter_mut()
                .zip(top_src.as_chunks::<16>().0.iter())
                .zip(weights[..w].as_chunks::<16>().0.iter())
                .enumerate()
            {
                let x = ci * 16;
                let above0 = load_u8x8_i16_fixed((&t[..8]).try_into().unwrap());
                let above1 = load_u8x8_i16_fixed((&t[8..]).try_into().unwrap());
                let wx0 = load_u8x8_i16_fixed((&wxc[..8]).try_into().unwrap());
                let wx1 = load_u8x8_i16_fixed((&wxc[8..]).try_into().unwrap());
                let d0 = dist8((w - 1 - x) as i16);
                let d1 = dist8((w - 1 - x - 8) as i16);

                let mut pv0 = vaddq_s16(
                    bottom_v,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(above0, bottom_v), off_ver), rnd_ver),
                        bhl2,
                    ),
                );
                let mut ph0 = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff_hor, d0), rnd_hor), bwl2),
                );
                pv0 = vaddq_s16(
                    pv0,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(above0, pv0), w_ver), add32),
                        6,
                    ),
                );
                ph0 = vaddq_s16(
                    ph0,
                    sra_i16(vaddq_s16(vmulq_s16(vsubq_s16(left_v, ph0), wx0), add32), 6),
                );
                let out0 = sra_i16(vaddq_s16(vaddq_s16(pv0, ph0), one), 1);

                let mut pv1 = vaddq_s16(
                    bottom_v,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(above1, bottom_v), off_ver), rnd_ver),
                        bhl2,
                    ),
                );
                let mut ph1 = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff_hor, d1), rnd_hor), bwl2),
                );
                pv1 = vaddq_s16(
                    pv1,
                    sra_i16(
                        vaddq_s16(vmulq_s16(vsubq_s16(above1, pv1), w_ver), add32),
                        6,
                    ),
                );
                ph1 = vaddq_s16(
                    ph1,
                    sra_i16(vaddq_s16(vmulq_s16(vsubq_s16(left_v, ph1), wx1), add32), 6),
                );
                let out1 = sra_i16(vaddq_s16(vaddq_s16(pv1, ph1), one), 1);

                store_i16x8x2_u8_fixed(oc, out0, out1);
            }
            let done = c16.len() * 16;
            let (c8, r8) = r16.as_chunks_mut::<8>();
            for (ci, ((oc, t), wxc)) in c8
                .iter_mut()
                .zip(top_src[done..].as_chunks::<8>().0.iter())
                .zip(weights[done..w].as_chunks::<8>().0.iter())
                .enumerate()
            {
                let above = load_u8x8_i16_fixed(t);
                let wx = load_u8x8_i16_fixed(wxc);
                let d = dist8((w - 1 - done - ci * 8) as i16);

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
                store_i16x8_u8_fixed(oc, out);
            }
            let base_x = done + c8.len() * 8;
            for (xi, oc) in r8.iter_mut().enumerate() {
                let x = base_x + xi;
                let above = tl[o + 1 + x] as i32;
                let mul_ver = (above - bottom as i32) * (h as i32 - 1 - y as i32);
                let mul_hor = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
                let mut pred_ver = bottom as i32 + ((mul_ver + (h >> 1) as i32) >> bhl2);
                let mut pred_hor = right as i32 + ((mul_hor + (w >> 1) as i32) >> bwl2);
                pred_ver += ((above - pred_ver) * weights[y] as i32 + 32) >> 6;
                pred_hor += ((left as i32 - pred_hor) * weights[x] as i32 + 32) >> 6;
                *oc = ((pred_ver + pred_hor + 1) >> 1) as u8;
            }
            off += stride;
        }
    }
}

use crate::levels::ANGLE_IBP_FLAG;

#[inline(always)]
fn load_u8x16_fixed(a: &[u8; 16]) -> uint8x16_t {
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline(always)]
fn store_u8x16_fixed(a: &mut [u8; 16], v: uint8x16_t) {
    unsafe { vst1q_u8(a.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn sum_u8_neon(s: &[u8]) -> u32 {
    let mut acc = vdupq_n_u32(0);
    let (chunks, rem) = s.as_chunks::<16>();
    for c in chunks.iter() {
        acc = vpadalq_u16(acc, vpaddlq_u8(load_u8x16_fixed(c)));
    }
    let mut total = vaddvq_u32(acc);
    for &b in rem {
        total += b as u32;
    }
    total
}

#[inline]
#[target_feature(enable = "neon")]
fn splat_fill_neon(dst: &mut [u8], stride: usize, off: usize, w: usize, h: usize, dc: u8) {
    let v = vdupq_n_u8(dc);
    let mut p = off;
    for _ in 0..h {
        let (chunks, rem) = dst[p..p + w].as_chunks_mut::<16>();
        for c in chunks.iter_mut() {
            store_u8x16_fixed(c, v);
        }
        rem.fill(dc);
        p += stride;
    }
}

#[target_feature(enable = "neon")]
fn ipred_dc_128_8bpc_neon_impl(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    splat_fill_neon(dst, stride, 0, w, h, 128);
}

#[target_feature(enable = "neon")]
fn ipred_dc_top_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_top_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let sum = sum_u8_neon(&tl[o + 1..o + 1 + w]);
    let dc = (((w >> 1) as u32 + sum) >> (w as u32).trailing_zeros()) as u8;
    splat_fill_neon(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "neon")]
fn ipred_dc_left_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_left_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let sum = sum_u8_neon(&tl[o - h..o]);
    let dc = (((h >> 1) as u32 + sum) >> (h as u32).trailing_zeros()) as u8;
    splat_fill_neon(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "neon")]
fn ipred_dc_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let n_pel = (w + h) as u32;
    let sum = sum_u8_neon(&tl[o + 1..o + 1 + w]) + sum_u8_neon(&tl[o - h..o]);
    let dc = if n_pel & (n_pel - 1) == 0 {
        (sum + w as u32) >> n_pel.trailing_zeros()
    } else {
        crate::ipred::fast_div32_dc(sum, n_pel).min(255)
    } as u8;
    splat_fill_neon(dst, stride, 0, w, h, dc);
}

pub(crate) fn ipred_dc_128_8bpc_neon(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    unsafe { ipred_dc_128_8bpc_neon_impl(dst, stride, w, h) }
}

pub(crate) fn ipred_dc_top_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_top_8bpc_neon_impl(dst, stride, tl, o, w, h, angle) }
}

pub(crate) fn ipred_dc_left_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_left_8bpc_neon_impl(dst, stride, tl, o, w, h, angle) }
}

pub(crate) fn ipred_dc_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_8bpc_neon_impl(dst, stride, tl, o, w, h, angle) }
}

#[target_feature(enable = "neon")]
fn ipred_paeth_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    let topleft = tl[o] as i32;
    if w < 8 {
        return crate::ipred::ipred_paeth_8bpc(dst, stride, tl, o, w, h);
    }
    let tl_v = vdupq_n_s16(topleft as i16);
    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let left_v = vdupq_n_s16(left as i16);
        let top_src = &tl[o + 1..o + 1 + w];
        let (c16, r16) = dst[off..off + w].as_chunks_mut::<16>();
        for (d, t) in c16.iter_mut().zip(top_src.as_chunks::<16>().0.iter()) {
            let top0 = load_u8x8_i16_fixed((&t[..8]).try_into().unwrap());
            let top1 = load_u8x8_i16_fixed((&t[8..]).try_into().unwrap());
            let base0 = vsubq_s16(vaddq_s16(left_v, top0), tl_v);
            let cond_l0 = vandq_u16(
                vceqq_s16(
                    vabsq_s16(vsubq_s16(left_v, base0)),
                    vminq_s16(
                        vabsq_s16(vsubq_s16(left_v, base0)),
                        vabsq_s16(vsubq_s16(top0, base0)),
                    ),
                ),
                vceqq_s16(
                    vabsq_s16(vsubq_s16(left_v, base0)),
                    vminq_s16(
                        vabsq_s16(vsubq_s16(left_v, base0)),
                        vabsq_s16(vsubq_s16(tl_v, base0)),
                    ),
                ),
            );
            let cond_t0 = vceqq_s16(
                vabsq_s16(vsubq_s16(top0, base0)),
                vminq_s16(
                    vabsq_s16(vsubq_s16(top0, base0)),
                    vabsq_s16(vsubq_s16(tl_v, base0)),
                ),
            );
            let res0 = vbslq_s16(cond_l0, left_v, vbslq_s16(cond_t0, top0, tl_v));
            let base1 = vsubq_s16(vaddq_s16(left_v, top1), tl_v);
            let cond_l1 = vandq_u16(
                vceqq_s16(
                    vabsq_s16(vsubq_s16(left_v, base1)),
                    vminq_s16(
                        vabsq_s16(vsubq_s16(left_v, base1)),
                        vabsq_s16(vsubq_s16(top1, base1)),
                    ),
                ),
                vceqq_s16(
                    vabsq_s16(vsubq_s16(left_v, base1)),
                    vminq_s16(
                        vabsq_s16(vsubq_s16(left_v, base1)),
                        vabsq_s16(vsubq_s16(tl_v, base1)),
                    ),
                ),
            );
            let cond_t1 = vceqq_s16(
                vabsq_s16(vsubq_s16(top1, base1)),
                vminq_s16(
                    vabsq_s16(vsubq_s16(top1, base1)),
                    vabsq_s16(vsubq_s16(tl_v, base1)),
                ),
            );
            let res1 = vbslq_s16(cond_l1, left_v, vbslq_s16(cond_t1, top1, tl_v));
            store_i16x8x2_u8_fixed(d, res0, res1);
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (d, t) in c8.iter_mut().zip(top_src[done..].as_chunks::<8>().0.iter()) {
            let top_v = load_u8x8_i16_fixed(t);
            let base = vsubq_s16(vaddq_s16(left_v, top_v), tl_v);
            let ld = vabsq_s16(vsubq_s16(left_v, base));
            let td = vabsq_s16(vsubq_s16(top_v, base));
            let tld = vabsq_s16(vsubq_s16(tl_v, base));
            let cond_l = vandq_u16(
                vceqq_s16(ld, vminq_s16(ld, td)),
                vceqq_s16(ld, vminq_s16(ld, tld)),
            );
            let cond_t = vceqq_s16(td, vminq_s16(td, tld));
            let inner = vbslq_s16(cond_t, top_v, tl_v);
            let res = vbslq_s16(cond_l, left_v, inner);
            store_i16x8_u8_fixed(d, res);
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let top = tl[o + 1 + base_x + xi] as i32;
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();
            *d = if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            } as u8;
        }
        off += stride;
    }
}

pub(crate) fn ipred_paeth_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    unsafe { ipred_paeth_8bpc_neon_impl(dst, stride, tl, o, w, h) }
}

/// 8 bytes -> two i32x4 lanes (ascending): lane k == a[k].
#[inline(always)]
fn load8_u8_i32(a: &[u8; 8]) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vmovl_u8(vld1_u8(a.as_ptr())) };
    unsafe {
        (
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))),
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w))),
        )
    }
}

/// 8 bytes -> two i32x4 lanes (reversed): lane k == a[7 - k].
#[inline(always)]
fn load8_u8_i32_rev(a: &[u8; 8]) -> (int32x4_t, int32x4_t) {
    let r = unsafe { vrev64_u8(vld1_u8(a.as_ptr())) };
    let w = unsafe { vmovl_u8(r) };
    unsafe {
        (
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))),
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w))),
        )
    }
}

/// `clamp((a*w0 + b*w1 + c*w2 + d*w3 + 64) >> 7, 0, 255)` packed to 8 u8.
#[inline(always)]
fn tap4_pack_neon(
    a: int32x4_t,
    b: int32x4_t,
    c: int32x4_t,
    d: int32x4_t,
    rnd: int32x4_t,
    w0: (int32x4_t, int32x4_t),
    w1: (int32x4_t, int32x4_t),
    w2: (int32x4_t, int32x4_t),
    w3: (int32x4_t, int32x4_t),
) -> uint8x8_t {
    unsafe {
        let acc_lo = vaddq_s32(
            vaddq_s32(vmulq_s32(a, w0.0), vmulq_s32(b, w1.0)),
            vaddq_s32(vmulq_s32(c, w2.0), vmulq_s32(d, w3.0)),
        );
        let acc_hi = vaddq_s32(
            vaddq_s32(vmulq_s32(a, w0.1), vmulq_s32(b, w1.1)),
            vaddq_s32(vmulq_s32(c, w2.1), vmulq_s32(d, w3.1)),
        );
        let res_lo = vshrq_n_s32::<7>(vaddq_s32(acc_lo, rnd));
        let res_hi = vshrq_n_s32::<7>(vaddq_s32(acc_hi, rnd));
        // saturating narrows == clamp(.,0,255)
        vqmovn_u16(vcombine_u16(vqmovun_s32(res_lo), vqmovun_s32(res_hi)))
    }
}

#[inline(always)]
fn store_u8x8_fixed(a: &mut [u8; 8], v: uint8x8_t) {
    unsafe { vst1_u8(a.as_mut_ptr(), v) };
}

#[inline(always)]
fn store_u8x8x2_fixed(a: &mut [u8; 16], lo: uint8x8_t, hi: uint8x8_t) {
    unsafe { vst1q_u8(a.as_mut_ptr(), vcombine_u8(lo, hi)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn z1_luma_row_neon(
    filt: &[u8],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u8,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u8],
    w: usize,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let av = vdupq_n_s32(f.a as i32);
    let bv = vdupq_n_s32(f.b as i32);
    let cv = vdupq_n_s32(f.c as i32);
    let dv = vdupq_n_s32(f.d as i32);
    let rnd = vdupq_n_s32(64);

    let base_const = (top_off as i32 + base0) as usize;
    let (body, fill_tail) = dst_row.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    for (ci, d) in c16.iter_mut().enumerate() {
        let bi = base_const + ci * 16;
        let pa = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32((&filt[bi - 1..bi - 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bi..bi + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bi + 1..bi + 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bi + 2..bi + 2 + 8]).try_into().unwrap()),
        );
        let bb = bi + 8;
        let pb = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32((&filt[bb - 1..bb - 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bb..bb + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bb + 1..bb + 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bb + 2..bb + 2 + 8]).try_into().unwrap()),
        );
        store_u8x8x2_fixed(d, pa, pb);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for (ci, d) in c8.iter_mut().enumerate() {
        let bi = base_const + done + ci * 8;
        let p = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32((&filt[bi - 1..bi - 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bi..bi + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bi + 1..bi + 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[bi + 2..bi + 2 + 8]).try_into().unwrap()),
        );
        store_u8x8_fixed(d, p);
    }
    let base_x = done + c8.len() * 8;
    for (xi, d) in r8.iter_mut().enumerate() {
        let bi = base_const + base_x + xi;
        let v = f.a as i32 * filt[bi - 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi + 1] as i32
            + f.d as i32 * filt[bi + 2] as i32;
        *d = ((v + 64) >> 7).clamp(0, 255) as u8;
    }
    fill_tail.fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn ipred_z1_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || enable_ibp || !is_luma || mrl_idx != 0 || w < 8 {
        return crate::ipred::ipred_z1_8bpc(
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
        );
    }
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let a = angle & 511;

    let dx = crate::tables::DR_INTRA_DERIVATIVE[a as usize] as i32;
    let max_base_x = (w + h) as i32 - 1;
    let mut filt = [0u8; 141];
    let top_off = 2usize;
    let sz = 1 + w + h;
    let str = if enable_intra_edge_filter && have_top {
        crate::ipred::get_filter_strength((w + h) as i32, 90 - a, is_sm_t)
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

    let mut ypos = dx;
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
        z1_luma_row_neon(&filt, top_off, base0, max_base_x, fill, f, dst_row, w);
        ypos += dx;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    unsafe {
        ipred_z1_8bpc_neon_impl(
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
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn z3_luma_col_neon(
    filt: &[u8],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u8,
    f: &crate::ipred::DrFilter4Tap,
    col: &mut [u8],
    h: usize,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let av = vdupq_n_s32(f.a as i32);
    let bv = vdupq_n_s32(f.b as i32);
    let cv = vdupq_n_s32(f.c as i32);
    let dv = vdupq_n_s32(f.d as i32);
    let rnd = vdupq_n_s32(64);

    let lob = left_off as i32 - base0; // bi_j at y == 0
    let (body, fill_tail) = col.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    for (ci, d) in c16.iter_mut().enumerate() {
        // group A covers col[y0..y0+8], group B col[y0+8..y0+16]; bi_j drops by
        // 8 between them. Reversed windows start at bi_j-6 .. bi_j-9.
        let bij = lob - (ci * 16) as i32;
        let (sa, sb, sc, sd) = (
            (bij - 6) as usize,
            (bij - 7) as usize,
            (bij - 8) as usize,
            (bij - 9) as usize,
        );
        let pa = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32_rev((&filt[sa..sa + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sb..sb + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sc..sc + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sd..sd + 8]).try_into().unwrap()),
        );
        let bij2 = bij - 8;
        let (sa2, sb2, sc2, sd2) = (
            (bij2 - 6) as usize,
            (bij2 - 7) as usize,
            (bij2 - 8) as usize,
            (bij2 - 9) as usize,
        );
        let pb = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32_rev((&filt[sa2..sa2 + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sb2..sb2 + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sc2..sc2 + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sd2..sd2 + 8]).try_into().unwrap()),
        );
        store_u8x8x2_fixed(d, pa, pb);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for (ci, d) in c8.iter_mut().enumerate() {
        let bij = lob - (done + ci * 8) as i32;
        let (sa, sb, sc, sd) = (
            (bij - 6) as usize,
            (bij - 7) as usize,
            (bij - 8) as usize,
            (bij - 9) as usize,
        );
        let p = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32_rev((&filt[sa..sa + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sb..sb + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sc..sc + 8]).try_into().unwrap()),
            load8_u8_i32_rev((&filt[sd..sd + 8]).try_into().unwrap()),
        );
        store_u8x8_fixed(d, p);
    }
    let base_y = done + c8.len() * 8;
    for (yi, d) in r8.iter_mut().enumerate() {
        let bi = (lob - (base_y + yi) as i32) as usize;
        let v = f.a as i32 * filt[bi + 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi - 1] as i32
            + f.d as i32 * filt[bi - 2] as i32;
        *d = ((v + 64) >> 7).clamp(0, 255) as u8;
    }
    fill_tail.fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn ipred_z3_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || enable_ibp || !is_luma || mrl_idx != 0 || h > 64 {
        return crate::ipred::ipred_z3_8bpc(
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
        );
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let a = angle & 511;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(270 - a) as usize] as i32;
    let max_base_y = (w + h) as i32 - 1;
    let mut filt = [0u8; 141];
    let left_off = 1 + w + h;
    let sz = 1 + w + h;
    let str = if enable_intra_edge_filter && have_left {
        crate::ipred::get_filter_strength((w + h) as i32, a - 180, is_sm_l)
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

    let mut col = [0u8; 64];
    let mut ypos = dy;
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let base0 = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        z3_luma_col_neon(&filt, left_off, base0, max_base_y, fill, f, &mut col, h);
        for (y, &c) in col[..h].iter().enumerate() {
            dst[y * stride + x] = c;
        }
        ypos += dy;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    unsafe {
        ipred_z3_8bpc_neon_impl(
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
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn z2_top_span_neon(
    filt: &[u8],
    top_off: usize,
    mut xpos: i32,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u8],
    x_start: usize,
    w: usize,
) {
    let av = vdupq_n_s32(f.a as i32);
    let bv = vdupq_n_s32(f.b as i32);
    let cv = vdupq_n_s32(f.c as i32);
    let dv = vdupq_n_s32(f.d as i32);
    let rnd = vdupq_n_s32(64);

    let mut x = x_start;
    while x + 8 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 1 < 0 || ti0 + 12 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 1) as usize;
        let packed = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            load8_u8_i32((&filt[sa..sa + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[sa + 1..sa + 1 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[sa + 2..sa + 2 + 8]).try_into().unwrap()),
            load8_u8_i32((&filt[sa + 3..sa + 3 + 8]).try_into().unwrap()),
        );
        store_u8x8_fixed((&mut dst_row[x..x + 8]).try_into().unwrap(), packed);
        x += 8;
        xpos += 64 * 8;
    }
    while x < w {
        let base_x = xpos >> 6;
        // Keep `ti` signed: at the left/top boundary `base_x` is -1, so `ti` is
        // -1 and `(ti + 1)` is 0. Casting to usize before the +1 would wrap and
        // overflow (matches the scalar reference and the 8-wide path above).
        let ti = top_off as i32 + base_x;
        let v = f.a as i32 * filt[(ti + 1) as usize] as i32
            + f.b as i32 * filt[(ti + 2) as usize] as i32
            + f.c as i32 * filt[(ti + 3) as usize] as i32
            + f.d as i32 * filt[(ti + 4) as usize] as i32;
        dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
        x += 1;
        xpos += 64;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn ipred_z2_8bpc_neon_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || !is_luma || mrl_idx != 0 {
        return crate::ipred::ipred_z2_8bpc(dst, stride, tl, o, w, h, angle, max_width, max_height);
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let a = angle & 511;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(a - 90) as usize] as i32;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[(180 - a) as usize] as i32;

    let mut filt = [0u8; 72];
    let top_off = 0usize;
    let sz_t = 1 + w;
    let str_t = if enable_intra_edge_filter && have_top {
        crate::ipred::get_filter_strength((w + h) as i32, a - 90, is_sm_t)
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

    let mut filt2 = [0u8; 72];
    let left_off: usize = h + 2;
    let sz_l = 1 + h;
    let str_l = if enable_intra_edge_filter && have_left {
        crate::ipred::get_filter_strength((w + h) as i32, 180 - a, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        crate::ipred::filter_edge(
            &mut filt2[1..],
            sz_l,
            h as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - h..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - h..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..h {
        let ypos = (y + 1) as i32;
        let mut xpos = -ypos * dx;
        let mut x = 0usize;
        let dst_row = &mut dst[y * stride..y * stride + w];

        while x < w && xpos < -64 {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - xpos_l * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            let v = f.a as i32 * filt2[bi - 1] as i32
                + f.b as i32 * filt2[bi - 2] as i32
                + f.c as i32 * filt2[bi - 3] as i32
                + f.d as i32 * filt2[bi - 4] as i32;
            dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
            x += 1;
            xpos += 64;
        }

        if x < w {
            let shift = ((xpos & 0x3F) >> 1) as usize;
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z2_top_span_neon(&filt, top_off, xpos, f, dst_row, x, w);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    unsafe { ipred_z2_8bpc_neon_impl(dst, stride, tl, o, w, h, angle, max_width, max_height) }
}
