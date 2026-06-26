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
use crate::levels::ANGLE_MULTI_MRL_FLAG;
use crate::tables::SM_WEIGHTS;

#[target_feature(enable = "neon")]
pub(crate) fn pal_pred_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    if w < 16 {
        crate::ipred::pal_pred(dst, stride, pal, idx, w, h);
        return;
    }

    let mut pal_buf = [0u8; 16];
    pal_buf[..8].copy_from_slice(&pal[..8]);
    let pal_v = unsafe { vld1q_u8(pal_buf.as_ptr()) };
    let mask = vdupq_n_u8(7);
    let zero = vdup_n_u8(0);
    let mut idx_off = 0usize;
    let mut dst_off = 0usize;
    for _ in 0..h {
        let row = &mut dst[dst_off..dst_off + w];
        let row_idx = &idx[idx_off..idx_off + (w >> 1)];
        let (idx16, rem16) = row_idx.as_chunks::<16>();
        let mut x = 0usize;
        for c in idx16.iter() {
            let packed = unsafe { vld1q_u8(c.as_ptr()) };
            let lo_idx = vandq_u8(packed, mask);
            let hi_idx = vandq_u8(vshrq_n_u8::<4>(packed), mask);
            let lo = vqtbl1q_u8(pal_v, lo_idx);
            let hi = vqtbl1q_u8(pal_v, hi_idx);
            unsafe {
                vst1q_u8(row[x..].as_mut_ptr(), vzip1q_u8(lo, hi));
                vst1q_u8(row[x + 16..].as_mut_ptr(), vzip2q_u8(lo, hi));
            }
            x += 32;
        }

        let (idx8, rem) = rem16.as_chunks::<8>();
        for c in idx8.iter() {
            let packed = unsafe { vcombine_u8(vld1_u8(c.as_ptr()), zero) };
            let lo_idx = vandq_u8(packed, mask);
            let hi_idx = vandq_u8(vshrq_n_u8::<4>(packed), mask);
            let lo = vqtbl1q_u8(pal_v, lo_idx);
            let hi = vqtbl1q_u8(pal_v, hi_idx);
            unsafe { vst1q_u8(row[x..].as_mut_ptr(), vzip1q_u8(lo, hi)) };
            x += 16;
        }

        for &i in rem {
            row[x] = pal[(i & 7) as usize];
            row[x + 1] = pal[(i >> 4) as usize];
            x += 2;
        }
        idx_off += w >> 1;
        dst_off += stride;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn avg_pred_8bpc_neon(dst: &mut [u8], stride: usize, tmp: &[u8], w: usize, h: usize) {
    for y in 0..h {
        let dst_row = &mut dst[y * stride..y * stride + w];
        let tmp_row = &tmp[y * 64..y * 64 + w];
        let (d16, drem) = dst_row.as_chunks_mut::<16>();
        for (d, t) in d16.iter_mut().zip(tmp_row.as_chunks::<16>().0.iter()) {
            store_u8x16_fixed(d, vrhaddq_u8(load_u8x16_fixed(d), load_u8x16_fixed(t)));
        }
        let base = d16.len() * 16;
        for (i, d) in drem.iter_mut().enumerate() {
            *d = ((*d as u16 + tmp_row[base + i] as u16 + 1) >> 1) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ibp_blend_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    inv: bool,
    weights: &[[u8; 16]; 16],
) {
    let x_shift = w >> 5;
    let y_shift = h >> 5;
    let c128 = vdupq_n_u16(128);
    let c64 = vdupq_n_u16(64);
    let mut wrow = [0u8; 128];
    for y in 0..h {
        let wy = y >> y_shift;
        for x in 0..w {
            let wx = x >> x_shift;
            wrow[x] = if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            };
        }
        let dst_row = &mut dst[y * stride..y * stride + w];
        let tmp_row = &tmp[y * 64..y * 64 + w];
        let mut x = 0usize;
        while x + 16 <= w {
            let wv8 = unsafe { vld1q_u8(wrow[x..].as_ptr()) };
            let dv8 = unsafe { vld1q_u8(dst_row[x..].as_ptr()) };
            let tv8 = unsafe { vld1q_u8(tmp_row[x..].as_ptr()) };
            let wl = vmovl_u8(vget_low_u8(wv8));
            let wh = vmovl_u8(vget_high_u8(wv8));
            let dl = vmovl_u8(vget_low_u8(dv8));
            let dh = vmovl_u8(vget_high_u8(dv8));
            let tl = vmovl_u8(vget_low_u8(tv8));
            let th = vmovl_u8(vget_high_u8(tv8));
            let rl = vshrq_n_u16::<7>(vaddq_u16(
                vaddq_u16(vmulq_u16(tl, vsubq_u16(c128, wl)), vmulq_u16(dl, wl)),
                c64,
            ));
            let rh = vshrq_n_u16::<7>(vaddq_u16(
                vaddq_u16(vmulq_u16(th, vsubq_u16(c128, wh)), vmulq_u16(dh, wh)),
                c64,
            ));
            unsafe {
                vst1q_u8(
                    dst_row[x..].as_mut_ptr(),
                    vcombine_u8(vqmovn_u16(rl), vqmovn_u16(rh)),
                )
            };
            x += 16;
        }
        while x + 8 <= w {
            let wv = vmovl_u8(unsafe { vld1_u8(wrow[x..].as_ptr()) });
            let dv = vmovl_u8(unsafe { vld1_u8(dst_row[x..].as_ptr()) });
            let tv = vmovl_u8(unsafe { vld1_u8(tmp_row[x..].as_ptr()) });
            let r = vshrq_n_u16::<7>(vaddq_u16(
                vaddq_u16(vmulq_u16(tv, vsubq_u16(c128, wv)), vmulq_u16(dv, wv)),
                c64,
            ));
            unsafe { vst1_u8(dst_row[x..].as_mut_ptr(), vqmovn_u16(r)) };
            x += 8;
        }
        while x < w {
            let wx = x >> x_shift;
            let weight = (if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            }) as u16;
            let t = tmp_row[x] as u16;
            let d = dst_row[x] as u16;
            dst_row[x] = ((t * (128 - weight) + d * weight + 64) >> 7) as u8;
            x += 1;
        }
    }
}

#[inline(always)]
fn load_u8x8_i16_fixed(ptr: &[u8; 8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr.as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16_i16x2_neon(a: &[u8; 16]) -> (int16x8_t, int16x8_t) {
    let v = unsafe { vld1q_u8(a.as_ptr()) };
    (
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))),
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))),
    )
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

#[inline]
#[target_feature(enable = "neon")]
fn sra_i16(v: int16x8_t, shift: i32) -> int16x8_t {
    vshlq_s16(v, vdupq_n_s16(-(shift as i16)))
}

#[inline]
#[target_feature(enable = "neon")]
fn dist8(base: i16) -> int16x8_t {
    static OFFSETS: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let offsets = unsafe { vld1q_s16(OFFSETS.as_ptr()) };
    vsubq_s16(vdupq_n_s16(base), offsets)
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
                let (above0, above1) = load_u8x16_i16x2_neon(tl);
                let mul0 = vmulq_s16(vsubq_s16(above0, bottom_v), off_y);
                let pred0 = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul0, rnd), bhl2));
                let adj0 =
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(above0, pred0), w_ver), add32));
                let mul1 = vmulq_s16(vsubq_s16(above1, bottom_v), off_y);
                let pred1 = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul1, rnd), bhl2));
                let adj1 =
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(above1, pred1), w_ver), add32));
                store_i16x8x2_u8_fixed(dst, vaddq_s16(pred0, adj0), vaddq_s16(pred1, adj1));
            }

            let done = c16.len() * 16;
            let (c8, r8) = r16.as_chunks_mut::<8>();
            for (dst, tl) in c8.iter_mut().zip(tl_src[done..].as_chunks::<8>().0.iter()) {
                let above = load_u8x8_i16_fixed(tl);
                let mul = vmulq_s16(vsubq_s16(above, bottom_v), off_y);
                let pred = vaddq_s16(bottom_v, sra_i16(vaddq_s16(mul, rnd), bhl2));
                let adj =
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(above, pred), w_ver), add32));
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
                let (wx_lo, wx_hi) = load_u8x16_i16x2_neon(wxc);
                let pred_lo = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff, d_lo), rnd), bwl2),
                );
                let adj_lo = vshrq_n_s16::<6>(vaddq_s16(
                    vmulq_s16(vsubq_s16(left_v, pred_lo), wx_lo),
                    add32,
                ));
                let pred_hi = vaddq_s16(
                    right_v,
                    sra_i16(vaddq_s16(vmulq_s16(diff, d_hi), rnd), bwl2),
                );
                let adj_hi = vshrq_n_s16::<6>(vaddq_s16(
                    vmulq_s16(vsubq_s16(left_v, pred_hi), wx_hi),
                    add32,
                ));
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
                let adj =
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred), wx), add32));
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
                let (above0, above1) = load_u8x16_i16x2_neon(t);
                let (wx0, wx1) = load_u8x16_i16x2_neon(wxc);
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
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(above0, pv0), w_ver), add32)),
                );
                ph0 = vaddq_s16(
                    ph0,
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(left_v, ph0), wx0), add32)),
                );
                let out0 = vshrq_n_s16::<1>(vaddq_s16(vaddq_s16(pv0, ph0), one));

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
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(above1, pv1), w_ver), add32)),
                );
                ph1 = vaddq_s16(
                    ph1,
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(left_v, ph1), wx1), add32)),
                );
                let out1 = vshrq_n_s16::<1>(vaddq_s16(vaddq_s16(pv1, ph1), one));

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
                    vshrq_n_s16::<6>(vaddq_s16(
                        vmulq_s16(vsubq_s16(above, pred_ver), w_ver),
                        add32,
                    )),
                );
                pred_hor = vaddq_s16(
                    pred_hor,
                    vshrq_n_s16::<6>(vaddq_s16(vmulq_s16(vsubq_s16(left_v, pred_hor), wx), add32)),
                );
                let out = vshrq_n_s16::<1>(vaddq_s16(vaddq_s16(pred_ver, pred_hor), one));
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
            let (top0, top1) = load_u8x16_i16x2_neon(t);
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

#[inline(always)]
fn widen8_at_neon<const OFF: i32>(v: uint8x16_t) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vmovl_u8(vget_low_u8(vextq_u8::<OFF>(v, v))) };
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
        let va = unsafe { vld1q_u8(filt[bi - 1..].as_ptr()) };
        let pa = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            widen8_at_neon::<0>(va),
            widen8_at_neon::<1>(va),
            widen8_at_neon::<2>(va),
            widen8_at_neon::<3>(va),
        );
        // group B taps bi+7..bi+10 -> byte-offsets 5..8 of a load at bi+2.
        let vb = unsafe { vld1q_u8(filt[bi + 2..].as_ptr()) };
        let pb = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            widen8_at_neon::<5>(vb),
            widen8_at_neon::<6>(vb),
            widen8_at_neon::<7>(vb),
            widen8_at_neon::<8>(vb),
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

#[inline]
#[target_feature(enable = "neon")]
fn z1_chroma_row_neon(
    filt: &[u8],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u8,
    shift: usize,
    dst_row: &mut [u8],
    w: usize,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let iw = vdupq_n_u16((32 - shift) as u16);
    let sw = vdupq_n_u16(shift as u16);
    let rnd = vdupq_n_u16(16);
    let base_const = (top_off as i32 + base0) as usize;
    let (body, fill_tail) = dst_row.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    for (ci, d) in c16.iter_mut().enumerate() {
        let bi = base_const + ci * 16;
        let a = unsafe { vld1q_u8(filt[bi..].as_ptr()) };
        let b = unsafe { vld1q_u8(filt[bi + 1..].as_ptr()) };
        let al = vmovl_u8(vget_low_u8(a));
        let ah = vmovl_u8(vget_high_u8(a));
        let bl = vmovl_u8(vget_low_u8(b));
        let bh = vmovl_u8(vget_high_u8(b));
        let rl = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(al, iw), vmulq_u16(bl, sw)),
            rnd,
        ));
        let rh = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(ah, iw), vmulq_u16(bh, sw)),
            rnd,
        ));
        unsafe { vst1q_u8(d.as_mut_ptr(), vcombine_u8(vqmovn_u16(rl), vqmovn_u16(rh))) };
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for (ci, d) in c8.iter_mut().enumerate() {
        let bi = base_const + done + ci * 8;
        let a = vmovl_u8(unsafe { vld1_u8(filt[bi..].as_ptr()) });
        let b = vmovl_u8(unsafe { vld1_u8(filt[bi + 1..].as_ptr()) });
        let r = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(a, iw), vmulq_u16(b, sw)),
            rnd,
        ));
        unsafe { vst1_u8(d.as_mut_ptr(), vqmovn_u16(r)) };
    }
    let base_x = done + c8.len() * 8;
    for (xi, d) in r8.iter_mut().enumerate() {
        let bi = base_const + base_x + xi;
        let v = (32 - shift as i32) * filt[bi] as i32 + shift as i32 * filt[bi + 1] as i32;
        *d = ((v + 16) >> 5).clamp(0, 255) as u8;
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
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![0u8; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z1_8bpc_neon_impl(
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
        );
        ipred_z1_8bpc_neon_impl(
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
        );
        avg_pred_8bpc_neon(dst, stride, &tmp, w, h);
        return;
    }
    if enable_ibp {
        let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
        let mode_idx = (10 - (a >> 3)).min(6) as usize;
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z1_8bpc_neon_impl(
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
        );
        ipred_z3_8bpc_neon_impl(
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
        );
        ibp_blend_8bpc_neon(dst, stride, &tmp, w, h, false, &ibp_weights[mode_idx]);
        return;
    }
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;

    let dx = crate::tables::DR_INTRA_DERIVATIVE[a as usize] as i32;
    let max_base_x = (w + h) as i32 - 1 + (mrl_idx as i32 * 2);
    let mut filt = [0u8; 141];
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
        let dst_row = &mut dst[y * stride..y * stride + w];
        if is_luma {
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z1_luma_row_neon(&filt, top_off, base0, max_base_x, fill, f, dst_row, w);
        } else {
            z1_chroma_row_neon(&filt, top_off, base0, max_base_x, fill, shift, dst_row, w);
        }
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
    // Full byte-reverse table for vqtbl1q_u8: out[i] = in[15 - i].
    let rev16 =
        unsafe { vld1q_u8([15u8, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0].as_ptr()) };
    for (ci, d) in c16.iter_mut().enumerate() {
        let bij = lob - (ci * 16) as i32;
        let ra = unsafe { vqtbl1q_u8(vld1q_u8(filt[(bij - 14) as usize..].as_ptr()), rev16) };
        let pa = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            widen8_at_neon::<0>(ra),
            widen8_at_neon::<1>(ra),
            widen8_at_neon::<2>(ra),
            widen8_at_neon::<3>(ra),
        );
        // rb[i] = filt[bij-2 - i]; group B windows = byte-offsets 5..8.
        let rb = unsafe { vqtbl1q_u8(vld1q_u8(filt[(bij - 17) as usize..].as_ptr()), rev16) };
        let pb = tap4_pack_neon(
            av,
            bv,
            cv,
            dv,
            rnd,
            widen8_at_neon::<5>(rb),
            widen8_at_neon::<6>(rb),
            widen8_at_neon::<7>(rb),
            widen8_at_neon::<8>(rb),
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

#[inline]
#[target_feature(enable = "neon")]
fn z3_chroma_col_neon(
    filt: &[u8],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u8,
    shift: usize,
    col: &mut [u8],
    h: usize,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let iw = vdupq_n_u16((32 - shift) as u16);
    let sw = vdupq_n_u16(shift as u16);
    let rnd = vdupq_n_u16(16);
    let lob = left_off as i32 - base0;
    let (body, fill_tail) = col.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    let rev16 =
        unsafe { vld1q_u8([15u8, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0].as_ptr()) };
    for (ci, d) in c16.iter_mut().enumerate() {
        let bij = lob - (ci * 16) as i32;
        let a = unsafe { vqtbl1q_u8(vld1q_u8(filt[(bij - 15) as usize..].as_ptr()), rev16) };
        let b = unsafe { vqtbl1q_u8(vld1q_u8(filt[(bij - 16) as usize..].as_ptr()), rev16) };
        let al = vmovl_u8(vget_low_u8(a));
        let ah = vmovl_u8(vget_high_u8(a));
        let bl = vmovl_u8(vget_low_u8(b));
        let bh = vmovl_u8(vget_high_u8(b));
        let rl = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(al, iw), vmulq_u16(bl, sw)),
            rnd,
        ));
        let rh = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(ah, iw), vmulq_u16(bh, sw)),
            rnd,
        ));
        unsafe { vst1q_u8(d.as_mut_ptr(), vcombine_u8(vqmovn_u16(rl), vqmovn_u16(rh))) };
    }
    let base_y = c16.len() * 16;
    for (yi, d) in r16.iter_mut().enumerate() {
        let bi = (lob - (base_y + yi) as i32) as usize;
        let v = (32 - shift as i32) * filt[bi] as i32 + shift as i32 * filt[bi - 1] as i32;
        *d = ((v + 16) >> 5).clamp(0, 255) as u8;
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
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![0u8; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z3_8bpc_neon_impl(
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
        );
        ipred_z3_8bpc_neon_impl(
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
        );
        avg_pred_8bpc_neon(dst, stride, &tmp, w, h);
        return;
    }
    if enable_ibp {
        if h > 64 {
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
        let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
        let mode_idx = ((a - 183) >> 3).min(6) as usize;
        let mut tmp = vec![0u8; 64 * 64];
        ipred_z3_8bpc_neon_impl(
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
        );
        ipred_z1_8bpc_neon_impl(
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
        );
        ibp_blend_8bpc_neon(dst, stride, &tmp, w, h, true, &ibp_weights[mode_idx]);
        return;
    }
    if h > 64 {
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

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(270 - a) as usize] as i32;
    let max_base_y = (w + h) as i32 - 1 + (mrl_idx as i32 * 2);
    let mut filt = [0u8; 141];
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

    let mut col = [0u8; 128];
    let mut ypos = dy * (1 + mrl_idx as i32);
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let base0 = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        if is_luma {
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z3_luma_col_neon(&filt, left_off, base0, max_base_y, fill, f, &mut col, h);
        } else {
            z3_chroma_col_neon(&filt, left_off, base0, max_base_y, fill, shift, &mut col, h);
        }
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

#[inline]
#[target_feature(enable = "neon")]
fn z2_top_span_chroma_neon(
    filt: &[u8],
    top_off: usize,
    mut xpos: i32,
    shift: usize,
    dst_row: &mut [u8],
    x_start: usize,
    w: usize,
) {
    let iw = vdupq_n_u16((32 - shift) as u16);
    let sw = vdupq_n_u16(shift as u16);
    let rnd = vdupq_n_u16(16);
    let mut x = x_start;
    while x + 16 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 2 < 0 || ti0 + 19 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 2) as usize;
        let a = unsafe { vld1q_u8(filt[sa..].as_ptr()) };
        let b = unsafe { vld1q_u8(filt[sa + 1..].as_ptr()) };
        let al = vmovl_u8(vget_low_u8(a));
        let ah = vmovl_u8(vget_high_u8(a));
        let bl = vmovl_u8(vget_low_u8(b));
        let bh = vmovl_u8(vget_high_u8(b));
        let rl = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(al, iw), vmulq_u16(bl, sw)),
            rnd,
        ));
        let rh = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(ah, iw), vmulq_u16(bh, sw)),
            rnd,
        ));
        unsafe {
            vst1q_u8(
                dst_row[x..].as_mut_ptr(),
                vcombine_u8(vqmovn_u16(rl), vqmovn_u16(rh)),
            )
        };
        x += 16;
        xpos += 64 * 16;
    }
    while x + 8 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 2 < 0 || ti0 + 11 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 2) as usize;
        let a = vmovl_u8(unsafe { vld1_u8(filt[sa..].as_ptr()) });
        let b = vmovl_u8(unsafe { vld1_u8(filt[sa + 1..].as_ptr()) });
        let r = vshrq_n_u16::<5>(vaddq_u16(
            vaddq_u16(vmulq_u16(a, iw), vmulq_u16(b, sw)),
            rnd,
        ));
        unsafe { vst1_u8(dst_row[x..].as_mut_ptr(), vqmovn_u16(r)) };
        x += 8;
        xpos += 64 * 8;
    }
    while x < w {
        let base_x = xpos >> 6;
        let ti = top_off as i32 + base_x;
        let v = (32 - shift as i32) * filt[(ti + 2) as usize] as i32
            + shift as i32 * filt[(ti + 3) as usize] as i32;
        dst_row[x] = ((v + 16) >> 5).clamp(0, 255) as u8;
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
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = vec![0u8; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z2_8bpc_neon_impl(
            &mut tmp,
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
        );
        ipred_z2_8bpc_neon_impl(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
        );
        avg_pred_8bpc_neon(dst, stride, &tmp, w, h);
        return;
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(a - 90) as usize] as i32;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[(180 - a) as usize] as i32;

    let mut filt = [0u8; 72];
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

    let mut filt2 = [0u8; 72];
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
                dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
            } else {
                let v = (32 - shift as i32) * filt2[bi - 2] as i32
                    + shift as i32 * filt2[bi - 3] as i32;
                dst_row[x] = ((v + 16) >> 5).clamp(0, 255) as u8;
            }
            x += 1;
            xpos += 64;
        }

        if x < w {
            let shift = ((xpos & 0x3F) >> 1) as usize;
            if is_luma {
                let f = &crate::ipred::DR_INTERP_FILTER[shift];
                z2_top_span_neon(&filt, top_off, xpos, f, dst_row, x, w);
            } else {
                z2_top_span_chroma_neon(&filt, top_off, xpos, shift, dst_row, x, w);
            }
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

#[target_feature(enable = "neon")]
pub(crate) fn ipred_dip_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
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
            dst_row[x] = (((s + 2048) >> 12) - in_sum).clamp(0, 255) as u8;
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
                    dst_row[x + z] = ((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2) as u8;
                }
                x += step_x;
            }
            y += step_y;
        }
    }

    if step_y > 1 {
        let mut p0_buf = [0u8; 128];
        let mut p1_buf = [0u8; 128];
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
                        >> uhl2) as u8;
                }
            }
        }
    }
}
