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
            store_u8x16_fixed(d, unsafe { vrhaddq_u8(load_u8x16_fixed(a), load_u8x16_fixed(b)) });
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
            let (rc, rrem) = row.as_chunks_mut::<8>();
            for (ci, (oc, wxc)) in rc
                .iter_mut()
                .zip(weights[..w].as_chunks::<8>().0.iter())
                .enumerate()
            {
                let dvec = dist8((w - 1 - ci * 8) as i16);
                let wx = load_u8x8_i16_fixed(wxc);
                let pred = vaddq_s16(right_v, sra_i16(vaddq_s16(vmulq_s16(diff, dvec), rnd), bwl2));
                let adj = sra_i16(vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred), wx), add32), 6);
                store_i16x8_u8_fixed(oc, vaddq_s16(pred, adj));
            }
            let base_x = (w / 8) * 8;
            for (xi, oc) in rrem.iter_mut().enumerate() {
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
            let (rc, rrem) = row.as_chunks_mut::<8>();
            for (ci, ((oc, t), wxc)) in rc
                .iter_mut()
                .zip(top_src.as_chunks::<8>().0.iter())
                .zip(weights[..w].as_chunks::<8>().0.iter())
                .enumerate()
            {
                let above = load_u8x8_i16_fixed(t);
                let wx = load_u8x8_i16_fixed(wxc);
                let d = dist8((w - 1 - ci * 8) as i16);

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
            let base_x = (w / 8) * 8;
            for (xi, oc) in rrem.iter_mut().enumerate() {
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

// ---------------------------------------------------------------------------
// DC family (dc / dc_top / dc_left / dc_128), 8bpc. SIMD edge reduction + fill;
// IBP-flagged blocks fall back to scalar. Bit-exact with the scalar path.
// ---------------------------------------------------------------------------

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
    let mut acc = unsafe { vdupq_n_u32(0) };
    let (chunks, rem) = s.as_chunks::<16>();
    for c in chunks.iter() {
        acc = unsafe { vpadalq_u16(acc, vpaddlq_u8(load_u8x16_fixed(c))) };
    }
    let mut total = unsafe { vaddvq_u32(acc) };
    for &b in rem {
        total += b as u32;
    }
    total
}

#[inline]
#[target_feature(enable = "neon")]
fn splat_fill_neon(dst: &mut [u8], stride: usize, off: usize, w: usize, h: usize, dc: u8) {
    let v = unsafe { vdupq_n_u8(dc) };
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

// ---------------------------------------------------------------------------
// Paeth predictor, 8bpc. Mirrors the SSE implementation and the scalar path.
// ---------------------------------------------------------------------------

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
    let tl_v = unsafe { vdupq_n_s16(topleft as i16) };
    let base_x = (w / 8) * 8;
    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let left_v = unsafe { vdupq_n_s16(left as i16) };
        let top_src = &tl[o + 1..o + 1 + w];
        let (rc, rrem) = dst[off..off + w].as_chunks_mut::<8>();
        for (d, t) in rc.iter_mut().zip(top_src.as_chunks::<8>().0.iter()) {
            let top_v = load_u8x8_i16_fixed(t);
            unsafe {
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
        }
        for (x, d) in rrem.iter_mut().enumerate() {
            let top = tl[o + 1 + base_x + x] as i32;
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
