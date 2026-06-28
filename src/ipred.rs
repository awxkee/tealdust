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

use crate::dip_tables::DIP_WEIGHTS;
use crate::intops::{apply_sign, iclip, imax, imin, ulog2};
use crate::levels::CflMhDir;
use crate::levels::{
    ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, ANGLE_IS_LUMA, ANGLE_MRL_IDX_MASK,
    ANGLE_MRL_IDX_SHIFT, ANGLE_MULTI_MRL_FLAG, ANGLE_SMOOTH_LEFT_EDGE_FLAG,
    ANGLE_SMOOTH_TOP_EDGE_FLAG, ANGLE_USE_EDGE_FILTER_FLAG,
};
use crate::pixel::{BitDepth, BitDepth8, Pixel};
use crate::recon::derive_alpha;
use crate::tables::{
    DC_IBP_WEIGHTS, DIV_RECIP, DIV_SCALE_SH_BIAS, DIV_SCALE_SH_COEFW, DIV_SCALE_SH_OFFSET,
    DR_INTRA_DERIVATIVE, SM_WEIGHTS,
};

#[derive(Clone, Copy)]
pub(crate) struct DrFilter4Tap {
    pub(crate) a: i8,
    pub(crate) b: u8,
    pub(crate) c: u8,
    pub(crate) d: i8,
}

pub(crate) static DR_INTERP_FILTER: [DrFilter4Tap; 32] = [
    DrFilter4Tap {
        a: 0,
        b: 128,
        c: 0,
        d: 0,
    },
    DrFilter4Tap {
        a: -2,
        b: 127,
        c: 4,
        d: -1,
    },
    DrFilter4Tap {
        a: -3,
        b: 125,
        c: 8,
        d: -2,
    },
    DrFilter4Tap {
        a: -5,
        b: 123,
        c: 13,
        d: -3,
    },
    DrFilter4Tap {
        a: -6,
        b: 121,
        c: 17,
        d: -4,
    },
    DrFilter4Tap {
        a: -7,
        b: 118,
        c: 22,
        d: -5,
    },
    DrFilter4Tap {
        a: -9,
        b: 116,
        c: 27,
        d: -6,
    },
    DrFilter4Tap {
        a: -9,
        b: 112,
        c: 32,
        d: -7,
    },
    DrFilter4Tap {
        a: -10,
        b: 109,
        c: 37,
        d: -8,
    },
    DrFilter4Tap {
        a: -11,
        b: 106,
        c: 41,
        d: -8,
    },
    DrFilter4Tap {
        a: -11,
        b: 102,
        c: 46,
        d: -9,
    },
    DrFilter4Tap {
        a: -12,
        b: 98,
        c: 52,
        d: -10,
    },
    DrFilter4Tap {
        a: -12,
        b: 94,
        c: 56,
        d: -10,
    },
    DrFilter4Tap {
        a: -12,
        b: 90,
        c: 61,
        d: -11,
    },
    DrFilter4Tap {
        a: -12,
        b: 85,
        c: 66,
        d: -11,
    },
    DrFilter4Tap {
        a: -12,
        b: 81,
        c: 71,
        d: -12,
    },
    DrFilter4Tap {
        a: -12,
        b: 76,
        c: 76,
        d: -12,
    },
    DrFilter4Tap {
        a: -12,
        b: 71,
        c: 81,
        d: -12,
    },
    DrFilter4Tap {
        a: -11,
        b: 66,
        c: 85,
        d: -12,
    },
    DrFilter4Tap {
        a: -11,
        b: 61,
        c: 90,
        d: -12,
    },
    DrFilter4Tap {
        a: -10,
        b: 56,
        c: 94,
        d: -12,
    },
    DrFilter4Tap {
        a: -10,
        b: 52,
        c: 98,
        d: -12,
    },
    DrFilter4Tap {
        a: -9,
        b: 46,
        c: 102,
        d: -11,
    },
    DrFilter4Tap {
        a: -8,
        b: 41,
        c: 106,
        d: -11,
    },
    DrFilter4Tap {
        a: -8,
        b: 37,
        c: 109,
        d: -10,
    },
    DrFilter4Tap {
        a: -7,
        b: 32,
        c: 112,
        d: -9,
    },
    DrFilter4Tap {
        a: -6,
        b: 27,
        c: 116,
        d: -9,
    },
    DrFilter4Tap {
        a: -5,
        b: 22,
        c: 118,
        d: -7,
    },
    DrFilter4Tap {
        a: -4,
        b: 17,
        c: 121,
        d: -6,
    },
    DrFilter4Tap {
        a: -3,
        b: 13,
        c: 123,
        d: -5,
    },
    DrFilter4Tap {
        a: -2,
        b: 8,
        c: 125,
        d: -3,
    },
    DrFilter4Tap {
        a: -1,
        b: 4,
        c: 127,
        d: -2,
    },
];

pub(crate) fn filter_strength(wh: i32, angle: i32, is_sm: bool) -> i32 {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else {
        if wh <= 8 {
            if angle >= 56 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 32 {
                return 3;
            }
            if angle >= 16 {
                return 2;
            }
            if angle >= 8 {
                return 1;
            }
        } else if wh <= 32 {
            if angle >= 32 {
                return 3;
            }
            if angle >= 4 {
                return 2;
            }
            return 1;
        } else {
            return 3;
        }
    }
    0
}

pub(crate) fn filter_edge<P: Pixel>(
    out: &mut [P],
    sz: usize,
    lim_from: i32,
    lim_to: i32,
    inp: &[P],
    from: i32,
    to: i32,
    strength: usize,
) {
    static KERNEL: [[u8; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

    debug_assert!(strength > 0);
    // NB: lim_from / lim_to may be negative (C uses signed `int`); compare in
    // i32 space so a negative bound yields an empty loop instead of wrapping.
    let mut i: i32 = 0;
    while i < imin(sz as i32, lim_from) {
        out[i as usize] = inp[iclip(i, from, to - 1) as usize];
        i += 1;
    }
    while i < imin(lim_to, sz as i32) {
        let mut s = 0i32;
        for j in 0..5 {
            s += inp[iclip(i - 2 + j, from, to - 1) as usize].into()
                * KERNEL[strength - 1][j as usize] as i32;
        }
        out[i as usize] = P::from_i32((s + 8) >> 4);
        i += 1;
    }
    while i < sz as i32 {
        out[i as usize] = inp[iclip(i, from, to - 1) as usize];
        i += 1;
    }
}

fn splat_dc<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    off: usize,
    width: usize,
    mut height: usize,
    dc: P,
) {
    let mut p = off;
    while height > 0 {
        dst[p..p + width].fill(dc);
        p += stride;
        height -= 1;
    }
}

fn dc_gen_top<P: Pixel>(tl: &[P], o: usize, width: usize) -> u32 {
    let mut dc = (width >> 1) as u32;
    for &px in &tl[o + 1..o + 1 + width] {
        dc += px.as_u16() as u32;
    }
    dc >> (width as u32).trailing_zeros()
}

fn dc_gen_left<P: Pixel>(tl: &[P], o: usize, height: usize) -> u32 {
    let mut dc = (height >> 1) as u32;
    for i in 0..height {
        dc += tl[o - 1 - i].as_u16() as u32;
    }
    dc >> (height as u32).trailing_zeros()
}

pub(crate) fn fast_div32_dc(num: u32, den: u32) -> u32 {
    debug_assert!(den > 0 && den <= 255);
    let mut shift = ulog2(den);
    let rem = den as i32 - (1 << shift);
    let idx = (rem << (7 - shift)) as usize;
    debug_assert!(idx <= 128);
    shift += 9;
    ((num as u64 * DIV_RECIP[idx] as u64) as u32 + ((1u32 << shift) >> 1)) >> shift
}

fn dc_gen<BD: BitDepth>(bd: BD, tl: &[BD::Pixel], o: usize, width: usize, height: usize) -> u32 {
    let n_pel = width + height;
    let mut dc = 0u32;
    for &px in &tl[o + 1..o + 1 + width] {
        dc += px.as_u16() as u32;
    }
    for i in 0..height {
        dc += tl[o - 1 - i].as_u16() as u32;
    }
    if n_pel & (n_pel - 1) == 0 {
        return (dc + width as u32) >> (n_pel as u32).trailing_zeros();
    }
    (fast_div32_dc(dc, n_pel as u32)).min(bd.bitdepth_max() as u32)
}

pub(crate) fn ipred_dc_128_8bpc(dst: &mut [u8], stride: usize, width: usize, height: usize) {
    ipred_dc_128(BitDepth8, dst, stride, width, height);
}

pub(crate) fn ipred_dc_128<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    width: usize,
    height: usize,
) {
    let dc = BD::Pixel::from_i32((bd.bitdepth_max() + 1) >> 1);
    splat_dc(dst, stride, 0, width, height, dc);
}

pub(crate) fn ipred_dc_top_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_dc_top(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_dc_top<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    mut height: usize,
    angle: i32,
) {
    let dc = dc_gen_top(tl, o, width);
    let mut off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let h = height >> 2;
        let w_y = &DC_IBP_WEIGHTS[h..];
        for y in 0..h {
            let wy = 128 - w_y[y] as u32;
            let dc_wy = dc * w_y[y] as u32;
            let dst_row = &mut dst[off..off + width];
            for (x, dst_px) in dst_row.iter_mut().enumerate() {
                *dst_px = BD::Pixel::from_i32(
                    ((tl[o + 1 + x].as_u16() as u32 * wy + dc_wy + 64) >> 7) as i32,
                );
            }
            off += stride;
        }
        height -= h;
    }

    splat_dc(
        dst,
        stride,
        off,
        width,
        height,
        BD::Pixel::from_i32(dc as i32),
    );
}

pub(crate) fn ipred_dc_left_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_dc_left(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_dc_left<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    mut width: usize,
    height: usize,
    angle: i32,
) {
    let dc = dc_gen_left(tl, o, height);
    let mut off = 0;
    let mut x_off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let w = width >> 2;
        let w_x = &DC_IBP_WEIGHTS[w..];
        for y in 0..height {
            let left = tl[o - 1 - y].as_u16() as u32;
            let dst_row = &mut dst[off..off + w];
            for (x, dst_px) in dst_row.iter_mut().enumerate() {
                *dst_px = BD::Pixel::from_i32(
                    ((left * (128 - w_x[x] as u32) + dc * w_x[x] as u32 + 64) >> 7) as i32,
                );
            }
            off += stride;
        }
        off = 0;
        x_off = w;
        width -= w;
    }

    let dc_p = BD::Pixel::from_i32(dc as i32);
    let mut p = off;
    for _ in 0..height {
        dst[p + x_off..p + x_off + width].fill(dc_p);
        p += stride;
    }
}

pub(crate) fn ipred_dc_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_dc(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_dc<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    mut width: usize,
    mut height: usize,
    angle: i32,
) {
    let dc = dc_gen(bd, tl, o, width, height);
    let mut off = 0;
    let mut x_off = 0;

    if angle & ANGLE_IBP_FLAG != 0 {
        let h = height >> 2;
        let w = width >> 2;
        let x_start = if width < height { w } else { 0 };
        let w_y = &DC_IBP_WEIGHTS[h..];
        for y in 0..h {
            let wy = 128 - w_y[y] as u32;
            let dc_wy = dc * w_y[y] as u32;
            let dst_row = &mut dst[off + x_start..off + width];
            for (x, dst_px) in dst_row.iter_mut().enumerate() {
                let x = x_start + x;
                *dst_px = BD::Pixel::from_i32(
                    ((tl[o + 1 + x].as_u16() as u32 * wy + dc_wy + 64) >> 7) as i32,
                );
            }
            off += stride;
        }

        let y_start = if width >= height { h } else { 0 };
        off = y_start * stride;
        let w_x = &DC_IBP_WEIGHTS[w..];
        for y in y_start..height {
            let left = tl[o - 1 - y].as_u16() as u32;
            let dst_row = &mut dst[off..off + w];
            for (x, dst_px) in dst_row.iter_mut().enumerate() {
                *dst_px = BD::Pixel::from_i32(
                    ((left * (128 - w_x[x] as u32) + dc * w_x[x] as u32 + 64) >> 7) as i32,
                );
            }
            off += stride;
        }
        off = h * stride + w;
        x_off = 0;
        width -= w;
        height -= h;
    }

    let dc_p = BD::Pixel::from_i32(dc as i32);
    let mut p = off;
    for _ in 0..height {
        dst[p + x_off..p + x_off + width].fill(dc_p);
        p += stride;
    }
}

pub(crate) fn ipred_v_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_v(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_v<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        for (x, dst_px) in dst[..width].iter_mut().enumerate() {
            let top: i32 = tl[o + 1 + x].into();
            let top2: i32 = tl[o + 1 + e_stride + x].into();
            *dst_px = BD::Pixel::from_i32((top + top2 + 1) >> 1);
        }
        let mut off = stride;
        for _ in 1..height {
            dst.copy_within(0..width, off);
            off += stride;
        }
        return;
    }
    let mut off = 0;
    for _ in 0..height {
        dst[off..off + width].copy_from_slice(&tl[o + 1..o + 1 + width]);
        off += stride;
    }
}

pub(crate) fn ipred_h_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    ipred_h(BitDepth8, dst, stride, tl, o, width, height, angle);
}

pub(crate) fn ipred_h<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        let mut off = 0;
        for y in 0..height {
            let left: i32 = tl[o - 1 - y].into();
            let left2: i32 = tl[o + e_stride - 1 - y].into();
            let v = BD::Pixel::from_i32((left + left2 + 1) >> 1);
            dst[off..off + width].fill(v);
            off += stride;
        }
        return;
    }
    let mut off = 0;
    for y in 0..height {
        let v = tl[o - 1 - y];
        dst[off..off + width].fill(v);
        off += stride;
    }
}

pub(crate) fn ipred_paeth_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    ipred_paeth(BitDepth8, dst, stride, tl, o, w, h);
}

pub(crate) fn ipred_paeth<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let topleft: i32 = tl[o].into();
    let mut off = 0;
    for y in 0..h {
        let left: i32 = tl[o - 1 - y].into();
        let dst_row = &mut dst[off..off + w];
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            let top: i32 = tl[o + 1 + x].into();
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();
            *dst_px = BD::Pixel::from_i32(if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            });
        }
        off += stride;
    }
}

pub(crate) fn ipred_smooth_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    ipred_smooth(BitDepth8, dst, stride, tl, o, w, h);
}

pub(crate) fn ipred_smooth<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let bwl2 = ulog2(w as u32);
    let bhl2 = ulog2(h as u32);
    let rnd_ver = (h >> 1) as i32;
    let rnd_hor = (w >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right: i32 = tl[o + w + 1].into();
    let bottom: i32 = tl[o - h - 1].into();

    let mut off = 0;
    for y in 0..h {
        let left: i32 = tl[o - 1 - y].into();
        let diff_hor = left - right;
        let off_ver = h as i32 - 1 - y as i32;
        let w_ver = weights[y] as i32;
        let dst_row = &mut dst[off..off + w];
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            let above: i32 = tl[o + 1 + x].into();
            let mul_ver = (above - bottom) * off_ver;
            let mul_hor = diff_hor * (w as i32 - 1 - x as i32);
            let mut pred_ver = bottom + ((mul_ver + rnd_ver) >> bhl2);
            let mut pred_hor = right + ((mul_hor + rnd_hor) >> bwl2);
            pred_ver += ((above - pred_ver) * w_ver + 32) >> 6;
            pred_hor += ((left - pred_hor) * weights[x] as i32 + 32) >> 6;
            *dst_px = BD::Pixel::from_i32((pred_ver + pred_hor + 1) >> 1);
        }
        off += stride;
    }
}

pub(crate) fn ipred_smooth_v_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    ipred_smooth_v(BitDepth8, dst, stride, tl, o, w, h);
}

pub(crate) fn ipred_smooth_v<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let bhl2 = ulog2(h as u32);
    let rnd = (h >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom: i32 = tl[o - h - 1].into();

    let mut off = 0;
    for y in 0..h {
        let off_y = h as i32 - 1 - y as i32;
        let w_ver = weights[y] as i32;
        let dst_row = &mut dst[off..off + w];
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            let above: i32 = tl[o + 1 + x].into();
            let mul = (above - bottom) * off_y;
            let pred = bottom + ((mul + rnd) >> bhl2);
            *dst_px = BD::Pixel::from_i32(pred + (((above - pred) * w_ver + 32) >> 6));
        }
        off += stride;
    }
}

pub(crate) fn ipred_smooth_h_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    ipred_smooth_h(BitDepth8, dst, stride, tl, o, w, h);
}

pub(crate) fn ipred_smooth_h<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    w: usize,
    h: usize,
) {
    let bwl2 = ulog2(w as u32);
    let rnd = (w >> 1) as i32;
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right_val: i32 = tl[o + w + 1].into();

    let mut off = 0;
    for y in 0..h {
        let left: i32 = tl[o - 1 - y].into();
        let diff = left - right_val;
        let dst_row = &mut dst[off..off + w];
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            let mul = diff * (w as i32 - 1 - x as i32);
            let pred = right_val + ((mul + rnd) >> bwl2);
            *dst_px = BD::Pixel::from_i32(pred + (((left - pred) * weights[x] as i32 + 32) >> 6));
        }
        off += stride;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    ipred_z1(
        BitDepth8,
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mut angle: i32,
    max_width: i32,
    _max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    angle &= 511;

    if mrl_mul {
        let e_stride = (width + height) * 2 + mrl_idx * 3 + 1;
        let mut tmp = [BD::Pixel::default(); 64 * 64];
        ipred_z1(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width,
            _max_height,
            ibp_weights,
        );
        ipred_z1(
            bd,
            dst,
            stride,
            tl,
            o + e_stride,
            width,
            height,
            angle | ANGLE_IS_LUMA,
            max_width,
            _max_height,
            ibp_weights,
        );
        for (y, dst_row) in dst.chunks_mut(stride).take(height).enumerate() {
            let tmp_row = &tmp[y * 64..y * 64 + width];
            for (tmp_px, dst_px) in tmp_row.iter().zip(dst_row[..width].iter_mut()) {
                let a: i32 = (*tmp_px).into();
                let b: i32 = (*dst_px).into();
                *dst_px = BD::Pixel::from_i32((a + b + 1) >> 1);
            }
        }
        return;
    }

    let dx = DR_INTRA_DERIVATIVE[angle as usize] as i32;
    let max_base_x = (width + height) as i32 - 1 + (mrl_idx as i32 * 2);

    // C: pixel filt[1 + 1 + 3 + 64 + 64 + 2 * 3 + 2] (= 141).
    let mut filt = [BD::Pixel::default(); 141];
    let top_off = 2 + mrl_idx;
    let sz = 1 + mrl_idx + width + height + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        filter_strength((width + height) as i32, 90 - angle, is_sm_t)
    } else {
        0
    };
    if str > 0 {
        filter_edge(
            &mut filt[1..],
            sz,
            1,
            sz as i32 + max_width - width as i32,
            &tl[o..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    // C: `filt[sz + 2] = filt[sz + 1] = filt[sz]` (right-associative), so both
    // sz+1 and sz+2 take filt[sz]. The assignment order matters: set sz+1 from
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz + 1];

    let mut ypos = dx * (1 + mrl_idx as i32);
    for y in 0..height {
        let mut base = ypos >> 6;
        let fill = filt[top_off + max_base_x as usize];
        if base > max_base_x {
            for dst_row in dst.chunks_mut(stride).take(height).skip(y) {
                dst_row[..width].fill(fill);
            }
            break;
        }
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &DR_INTERP_FILTER[shift];
        let dst_row = &mut dst[y * stride..y * stride + width];
        let mut row_iter = dst_row.iter_mut().enumerate();
        while let Some((_x, dst_px)) = row_iter.next() {
            if base > max_base_x {
                *dst_px = fill;
                for (_, dst_px) in row_iter {
                    *dst_px = fill;
                }
                break;
            }
            let bi = top_off as i32 + base;
            *dst_px = if is_luma {
                let v = f.a as i32 * Into::<i32>::into(filt[(bi - 1) as usize])
                    + f.b as i32 * Into::<i32>::into(filt[bi as usize])
                    + f.c as i32 * Into::<i32>::into(filt[(bi + 1) as usize])
                    + f.d as i32 * Into::<i32>::into(filt[(bi + 2) as usize]);
                bd.pixel_clip((v + 64) >> 7)
            } else {
                let v = (32 - shift as i32) * Into::<i32>::into(filt[bi as usize])
                    + shift as i32 * Into::<i32>::into(filt[(bi + 1) as usize]);
                bd.pixel_clip((v + 16) >> 5)
            };
            base += 1;
        }
        ypos += dx;
    }

    if enable_ibp {
        let mode_idx = imin(10 - (angle >> 3), 6) as usize;
        let mut tmp = [BD::Pixel::default(); 64 * 64];
        ipred_z3(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            (180 + angle) | angle_flags,
            max_width,
            _max_height,
            ibp_weights,
        );
        ibp_blend(
            bd,
            dst,
            stride,
            &tmp,
            width,
            height,
            false,
            &ibp_weights[mode_idx],
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    ipred_z3(
        BitDepth8,
        dst,
        stride,
        tl,
        o,
        width,
        height,
        angle,
        max_width,
        max_height,
        ibp_weights,
    );
}

#[allow(clippy::explicit_counter_loop)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mut angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    angle &= 511;

    if mrl_mul {
        let e_stride = (width + height) * 2 + mrl_idx * 3 + 1;
        let mut tmp = [BD::Pixel::default(); 64 * 64];
        ipred_z3(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width,
            max_height,
            ibp_weights,
        );
        ipred_z3(
            bd,
            dst,
            stride,
            tl,
            o + e_stride,
            width,
            height,
            angle | ANGLE_IS_LUMA,
            max_width,
            max_height,
            ibp_weights,
        );
        for (y, dst_row) in dst.chunks_mut(stride).take(height).enumerate() {
            let tmp_row = &tmp[y * 64..y * 64 + width];
            for (tmp_px, dst_px) in tmp_row.iter().zip(dst_row[..width].iter_mut()) {
                let a: i32 = (*tmp_px).into();
                let b: i32 = (*dst_px).into();
                *dst_px = BD::Pixel::from_i32((a + b + 1) >> 1);
            }
        }
        return;
    }

    let dy = DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32;
    let max_base_y = (width + height) as i32 - 1 + (mrl_idx as i32 * 2);

    // C: pixel filt[1 + 1 + 3 + 64 + 64 + 2 * 3 + 2] (= 141).
    let mut filt = [BD::Pixel::default(); 141];
    let left_off = 1 + width + height + mrl_idx * 2;
    let sz = 1 + mrl_idx + width + height + mrl_idx * 2;

    let str = if enable_intra_edge_filter && mrl_idx == 0 && have_left {
        filter_strength((width + height) as i32, angle - 180, is_sm_l)
    } else {
        0
    };

    if str > 0 {
        filter_edge(
            &mut filt[2..],
            sz,
            height as i32 - max_height,
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

    let mut ypos = dy * (1 + mrl_idx as i32);
    for x in 0..width {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &DR_INTERP_FILTER[shift];
        let mut base = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        let mut rows = dst.chunks_mut(stride).take(height);
        while let Some(dst_row) = rows.next() {
            if base <= max_base_y {
                let bi = left_off as i32 - base;
                dst_row[x] = if is_luma {
                    let v = f.a as i32 * Into::<i32>::into(filt[(bi + 1) as usize])
                        + f.b as i32 * Into::<i32>::into(filt[bi as usize])
                        + f.c as i32 * Into::<i32>::into(filt[(bi - 1) as usize])
                        + f.d as i32 * Into::<i32>::into(filt[(bi - 2) as usize]);
                    bd.pixel_clip((v + 64) >> 7)
                } else {
                    let v = (32 - shift as i32) * Into::<i32>::into(filt[bi as usize])
                        + shift as i32 * Into::<i32>::into(filt[(bi - 1) as usize]);
                    bd.pixel_clip((v + 16) >> 5)
                };
                base += 1;
            } else {
                dst_row[x] = fill;
                for dst_row in rows {
                    dst_row[x] = fill;
                }
                break;
            }
        }
        ypos += dy;
    }

    if enable_ibp {
        let mode_idx = imin((angle - 183) >> 3, 6) as usize;
        let mut tmp = [BD::Pixel::default(); 64 * 64];
        ipred_z1(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            (angle - 180) | angle_flags,
            max_width,
            max_height,
            ibp_weights,
        );
        ibp_blend(
            bd,
            dst,
            stride,
            &tmp,
            width,
            height,
            true,
            &ibp_weights[mode_idx],
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    ipred_z2(
        BitDepth8, dst, stride, tl, o, width, height, angle, max_width, max_height,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
    o: usize,
    width: usize,
    height: usize,
    mut angle: i32,
    max_width: i32,
    max_height: i32,
) {
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    angle &= 511;

    if mrl_mul {
        let e_stride = (width + height) * 2 + mrl_idx * 3 + 1;
        let mut tmp = [BD::Pixel::default(); 64 * 64];
        ipred_z2(
            bd,
            &mut tmp,
            64,
            tl,
            o,
            width,
            height,
            angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT) | ANGLE_IS_LUMA,
            max_width,
            max_height,
        );
        ipred_z2(
            bd,
            dst,
            stride,
            tl,
            o + e_stride,
            width,
            height,
            angle | ANGLE_IS_LUMA,
            max_width,
            max_height,
        );
        for (y, dst_row) in dst.chunks_mut(stride).take(height).enumerate() {
            let tmp_row = &tmp[y * 64..y * 64 + width];
            for (tmp_px, dst_px) in tmp_row.iter().zip(dst_row[..width].iter_mut()) {
                let a: i32 = (*tmp_px).into();
                let b: i32 = (*dst_px).into();
                *dst_px = BD::Pixel::from_i32((a + b + 1) >> 1);
            }
        }
        return;
    }

    let dy = DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32;
    let dx = DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32;

    // Top filter buffer
    let mut filt = [BD::Pixel::default(); 72];
    let top_off = mrl_idx;
    let sz_t = 1 + width + mrl_idx;
    let str_t = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        filter_strength((width + height) as i32, angle - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - width as i32,
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

    // Left filter buffer
    let mut filt2 = [BD::Pixel::default(); 72];
    let left_off: usize = height + 2;
    let sz_l = 1 + height + mrl_idx;
    let str_l = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        filter_strength((width + height) as i32, 180 - angle, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        filter_edge(
            &mut filt2[1..],
            sz_l,
            height as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - (height + mrl_idx)..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - (height + mrl_idx)..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..height {
        let ypos = (y + 1) as i32;
        let mut xpos = -(ypos + mrl_idx as i32) * dx;
        let mut x = 0usize;
        let dst_row = &mut dst[y * stride..y * stride + width];

        // Left reference loop
        while x < width && xpos < -(64 * (1 + mrl_idx as i32)) {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - (xpos_l + mrl_idx as i32) * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            dst_row[x] = if is_luma {
                let f = &DR_INTERP_FILTER[shift];
                let v = f.a as i32 * Into::<i32>::into(filt2[bi - 1])
                    + f.b as i32 * Into::<i32>::into(filt2[bi - 2])
                    + f.c as i32 * Into::<i32>::into(filt2[bi - 3])
                    + f.d as i32 * Into::<i32>::into(filt2[bi - 4]);
                bd.pixel_clip((v + 64) >> 7)
            } else {
                let v = (32 - shift as i32) * Into::<i32>::into(filt2[bi - 2])
                    + shift as i32 * Into::<i32>::into(filt2[bi - 3]);
                bd.pixel_clip((v + 16) >> 5)
            };
            x += 1;
            xpos += 64;
        }

        // Top reference loop
        for dst_px in dst_row[x..].iter_mut() {
            let base_x = xpos >> 6;
            let shift = ((xpos & 0x3F) >> 1) as usize;
            let ti = top_off as i32 + base_x;
            *dst_px = if is_luma {
                let f = &DR_INTERP_FILTER[shift];
                let v = f.a as i32 * Into::<i32>::into(filt[(ti + 1) as usize])
                    + f.b as i32 * Into::<i32>::into(filt[(ti + 2) as usize])
                    + f.c as i32 * Into::<i32>::into(filt[(ti + 3) as usize])
                    + f.d as i32 * Into::<i32>::into(filt[(ti + 4) as usize]);
                bd.pixel_clip((v + 64) >> 7)
            } else {
                let v = (32 - shift as i32) * Into::<i32>::into(filt[(ti + 2) as usize])
                    + shift as i32 * Into::<i32>::into(filt[(ti + 3) as usize]);
                bd.pixel_clip((v + 16) >> 5)
            };
            xpos += 64;
        }
    }
}

pub(crate) fn ibp_blend<BD: BitDepth>(
    _bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tmp: &[BD::Pixel],
    width: usize,
    height: usize,
    inv: bool,
    weights: &[[u8; 16]; 16],
) {
    let x_shift = width >> (4 + 1);
    let y_shift = height >> (4 + 1);

    for (y, dst_row) in dst.chunks_mut(stride).take(height).enumerate() {
        let wy = y >> y_shift;
        let tmp_row = &tmp[y * 64..y * 64 + width];
        for (x, (tmp_px, dst_px)) in tmp_row.iter().zip(dst_row[..width].iter_mut()).enumerate() {
            let wx = x >> x_shift;
            let weight = if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            } as i32;
            let t: i32 = (*tmp_px).into();
            let d: i32 = (*dst_px).into();
            *dst_px = BD::Pixel::from_i32((t * (128 - weight) + d * weight + 64) >> 7);
        }
    }
}

pub(crate) fn get_div_scale_sh(d: i32) -> (i32, i32) {
    let d = imax(1, d.abs());
    let sh = ulog2(d as u32);
    let nsh = sh - 14;
    let d = if nsh >= 0 {
        let rnd = if nsh > 0 { 1 << (nsh - 1) } else { 0 };
        (d + rnd) >> nsh
    } else {
        d << (-nsh)
    };
    let d = iclip(d, 1, 0x7fff);
    let d = d & ((1 << 14) - 1);

    let idx = (d >> 11) as usize;
    let coefw = DIV_SCALE_SH_COEFW[idx] as i32;
    let bias = DIV_SCALE_SH_BIAS[idx] as i32;
    let d = d - DIV_SCALE_SH_OFFSET[idx] as i32;
    let scale = (((coefw * ((d * d) >> 14)) >> 8) - (d >> 1) + bias) << 2;
    (scale, sh)
}

pub(crate) fn mul32(a: i32, b: i32, sh: i32) -> i32 {
    let a2 = ulog2((a.abs() | 1) as u32) + 1;
    let b2 = ulog2((b.abs() | 1) as u32) + 1;
    let drop = if a2 + b2 > 29 { a2 + b2 - 29 } else { 0 };
    let ash = drop >> 1;
    let bsh = drop - ash;
    let adj = sh - (ash + bsh);
    let mul = (a >> ash) * (b >> bsh);
    if adj <= 0 {
        return mul;
    }
    debug_assert!(adj <= 29);
    let bias = 1u32 << (adj as u32 - 1);
    if mul >= 0 {
        ((mul as u32).wrapping_add(bias) >> adj as u32) as i32
    } else {
        -((((-mul) as u32).wrapping_add(bias) >> adj as u32) as i32)
    }
}

pub(crate) fn ipred_dip_8bpc(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
) {
    ipred_dip(BitDepth8, dst, stride, tl, o, width, height, mode);
}

pub(crate) fn ipred_dip<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    tl: &[BD::Pixel],
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
    inp[0] = tl[o].into();
    let mut in_sum = inp[0];

    let mut ti = o + 1;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..wd {
            sum += Into::<i32>::into(tl[ti]);
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
            sum += Into::<i32>::into(tl[li]);
        }
        inp[i_l + i] = (sum + hrnd as i32) >> hl2;
        in_sum += inp[i_l + i];
    }

    let mut sum = 0i32;
    for x in 0..wd {
        sum += Into::<i32>::into(tl[o + x + width + 1]);
    }
    let idx_tr = if trans { 10 } else { 9 };
    inp[idx_tr] = (sum + wrnd as i32) >> wl2;
    in_sum += inp[idx_tr];

    sum = 0;
    for y in 0..hd {
        sum += Into::<i32>::into(tl[o - (y + height + 1)]);
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
            for i in 0..11 {
                s += DIP_WEIGHTS[m][idx][i] as i32 * inp[i];
            }
            dst_row[x] = bd.pixel_clip(((s + 2048) >> 12) - in_sum);
            x += step_x;
        }
        y += step_y;
    }

    if step_x > 1 {
        y = step_y - 1;
        for _gy in 0..grid_h {
            let mut p1: i32 = tl[o - (y + 1)].into();
            let mut x = 0usize;
            let dst_row = &mut dst[y * stride..y * stride + width];
            for _gx in 0..grid_w {
                let p0 = p1;
                p1 = dst_row[x + step_x - 1].into();
                for (z, dst_px) in dst_row[x..x + step_x - 1].iter_mut().enumerate() {
                    let z1 = (z + 1) as i32;
                    *dst_px = BD::Pixel::from_i32((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2);
                }
                x += step_x;
            }
            y += step_y;
        }
    }

    if step_y > 1 {
        for x in 0..width {
            let mut p1: i32 = tl[o + x + 1].into();
            y = 0;
            for _gy in 0..grid_h {
                let p0 = p1;
                p1 = dst[(y + step_y - 1) * stride + x].into();
                for z in 0..step_y - 1 {
                    let z1 = (z + 1) as i32;
                    dst[(y + z) * stride + x] =
                        BD::Pixel::from_i32((p0 * (step_y as i32 - z1) + p1 * z1) >> uhl2);
                }
                y += step_y;
            }
        }
    }
}

pub(crate) fn pal_pred<P: Pixel>(
    dst: &mut [P],
    stride: usize,
    pal: &[P],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let mut idx_iter = idx.iter();
    for dst_row in dst.chunks_mut(stride).take(h) {
        for pair in dst_row[..w].as_chunks_mut::<2>().0.iter_mut() {
            let i = *idx_iter.next().expect("palette index buffer too small");
            pair[0] = pal[(i & 7) as usize];
            pair[1] = pal[(i >> 4) as usize];
        }
    }
}

pub(crate) const CFL_FLT_TYPE_UNIFORM: i32 = 0;
pub(crate) const CFL_FLT_TYPE_VSTRIP: i32 = 1;
pub(crate) const CFL_FLT_TYPE_GAUSS: i32 = 2;
pub(crate) const CFL_HAS_TOP: i32 = 1 << 2;
pub(crate) const CFL_HAS_LEFT: i32 = 1 << 3;
pub(crate) const CFL_DIR_ALL: i32 = CflMhDir::All as i32;
pub(crate) const CFL_DIR_LEFT: i32 = CflMhDir::Left as i32;
pub(crate) const CFL_DIR_TOP: i32 = CflMhDir::Top as i32;
pub(crate) const CFL_IS_TOP_SB_EDGE: u32 = 1 << 4;
pub(crate) const CFL_ALPHA_LOG2: u32 = 5;
pub(crate) const CFL_ALPHA_U_SHIFT: u32 = 16 - CFL_ALPHA_LOG2;
pub(crate) const CFL_ALPHA_V_SHIFT: u32 = 32 - CFL_ALPHA_LOG2;
pub(crate) const CFL_ALPHA_U_MASK: u32 = ((1 << CFL_ALPHA_LOG2) - 1) << CFL_ALPHA_U_SHIFT;
pub(crate) const CFL_ALPHA_V_MASK: u32 = ((1 << CFL_ALPHA_LOG2) - 1) << CFL_ALPHA_V_SHIFT;

#[inline(always)]
fn cfl_filter<P: Pixel>(
    src: &[P],
    c: usize,
    l: usize,
    r: usize,
    b: usize,
    top: &[P],
    tc: usize,
    filter_type: i32,
) -> P {
    let s = |i: usize| -> i32 { src[i].into() };
    let t = |i: usize| -> i32 { top[i].into() };
    match filter_type {
        CFL_FLT_TYPE_UNIFORM => P::from_i32((s(c) + s(r) + s(b + c) + s(b + r)) >> 2),
        CFL_FLT_TYPE_VSTRIP => {
            P::from_i32((s(l) + 2 * s(c) + s(r) + s(b + l) + 2 * s(b + c) + s(b + r)) >> 3)
        }
        _ => P::from_i32((s(l) + 4 * s(c) + s(r) + t(tc) + s(b + c)) >> 3),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_gen_y_420<P: Pixel>(
    dst: &mut [P],
    dst_top_stride: usize,
    src: &[P],
    src_off: usize,
    top_sb_edge: Option<(&[P], usize)>,
    src_stride: usize,
    refw: usize,
    refh: usize,
    tw: usize,
    th: usize,
    flags: i32,
    filter_type: i32,
    ss_hor: usize,
    ss_ver: usize,
) {
    let has_t = flags & CFL_HAS_TOP != 0;
    let has_l = flags & CFL_HAS_LEFT != 0;
    let dir = flags & CFL_DIR_ALL;
    let n_left: usize = if has_l {
        1 + (dir == CFL_DIR_LEFT) as usize
    } else {
        0
    };
    let n_top: usize = if has_t {
        1 + (dir == CFL_DIR_TOP) as usize
    } else {
        0
    };
    let dst_left_base = n_top * dst_top_stride + 64 * 64;

    // I444 (ss_hor==ss_ver==0): the MHCCP luma reference is the luma plane copied
    // 1:1 with NO spatial downsampling (AVM: output = input[i], i.e. its q3 value
    // is `input<<3` while cfl_filter here returns luma-scale, so no scaling).
    // The dst layout is identical to the 4:2:0 case (it is at chroma resolution =
    // luma resolution here); only the source sampling differs.
    if ss_hor == 0 && ss_ver == 0 {
        let ss = n_left;
        let mut dst_p = 0usize;
        let mut dst_lp = dst_left_base;
        if has_t {
            let has_tsb = top_sb_edge.is_some();
            let (tsb, tsb_off) = top_sb_edge.unwrap_or((src, src_off));
            let mut top_sp: usize = if has_tsb {
                tsb_off - ss
            } else {
                (src_off - ss) - n_top * src_stride
            };
            let top_buf: &[P] = if has_tsb { tsb } else { src };
            for _y in 0..n_top {
                dst[dst_lp..(n_left + dst_lp)].copy_from_slice(&top_buf[top_sp..(n_left + top_sp)]);
                for x in n_left..refw {
                    dst[dst_p + x - n_left] = top_buf[top_sp + x];
                }
                if !has_tsb {
                    top_sp += src_stride;
                }
                dst_lp += n_left;
                dst_p += dst_top_stride;
            }
        }
        let mut sp = src_off - ss;
        for _y in 0..th {
            dst[dst_lp..(n_left + dst_lp)].copy_from_slice(&src[sp..(n_left + sp)]);
            for x in n_left..n_left + tw {
                dst[dst_p + x - n_left] = src[sp + x];
            }
            sp += src_stride;
            dst_lp += n_left;
            dst_p += tw;
        }
        let n_bl = refh.saturating_sub(th);
        for _y in 0..n_bl {
            dst[dst_lp..(n_left + dst_lp)].copy_from_slice(&src[sp..(n_left + sp)]);
            sp += src_stride;
            dst_lp += n_left;
        }
        return;
    }

    let ss = n_left << 1;

    // Vertical handling is subsampling-aware: I420 (ss_ver=1) averages two luma
    // rows per chroma row (bottom tap = src_stride, 2-row stride); I422 (ss_ver=0)
    // keeps full vertical resolution (bottom tap = 0 so cfl_filter degenerates to a
    // horizontal-only average, 1-row stride), matching AVM's
    // cfl_adaptive_luma_subsampling_422.
    let vstep = (src_stride as isize) << ss_ver;
    let b_v: isize = if ss_ver == 1 { src_stride as isize } else { 0 };

    let mut dst_p = 0usize;
    let mut dst_lp = dst_left_base;

    // tl+t+tr: top reference rows
    if has_t {
        let has_tsb = top_sb_edge.is_some();
        let (tsb, tsb_off) = top_sb_edge.unwrap_or((src, src_off));
        let mut top_sp: usize;
        let top_buf: &[P];
        let b: isize;
        let mut t: isize;

        if has_tsb {
            top_sp = tsb_off - ss;
            top_buf = tsb;
            b = 0;
            t = 0;
        } else {
            top_sp = (src_off - ss) - (n_top as isize * vstep) as usize;
            top_buf = src;
            b = b_v;
            t = if n_top == 1 {
                -(src_stride as isize)
            } else {
                0
            };
        }

        for _y in 0..n_top {
            for x in 0..n_left {
                let c = x * 2;
                let r = c + 1;
                // For odd n_left the left tap is NOT clamped at column 0 (reads the
                // pixel one column left, i.e. relative -1); only the even case clamps.
                let l_off: isize = if n_left & 1 != 0 {
                    c as isize - 1
                } else {
                    imax(c as i32 - 1, 0) as isize
                };
                dst[dst_lp + x] = cfl_filter(
                    top_buf,
                    top_sp + c,
                    (top_sp as isize + l_off) as usize,
                    top_sp + r,
                    b as usize,
                    top_buf,
                    (top_sp as isize + t) as usize + c,
                    filter_type,
                );
            }
            for x in n_left..refw {
                let c = x * 2;
                let r = c + 1;
                let l_idx = if n_left > 0 {
                    c - 1
                } else {
                    imax(c as i32 - 1, 0) as usize
                };
                dst[dst_p + x - n_left] = cfl_filter(
                    top_buf,
                    top_sp + c,
                    top_sp + l_idx,
                    top_sp + r,
                    b as usize,
                    top_buf,
                    (top_sp as isize + t) as usize + c,
                    filter_type,
                );
            }
            if !has_tsb {
                top_sp += vstep as usize;
                t = -(src_stride as isize);
            }
            dst_lp += n_left;
            dst_p += dst_top_stride;
        }
    }

    // l+blk: main block rows
    let b = b_v;
    let mut sp = src_off - ss;
    let first_top: (&[P], usize) = if has_t {
        if let Some((tsb, tsb_off)) = top_sb_edge {
            (tsb, tsb_off - ss)
        } else {
            (src, src_off - ss - src_stride)
        }
    } else {
        (src, src_off - ss)
    };

    for y in 0..th {
        let (tb, tp) = if y == 0 {
            first_top
        } else {
            (src, sp - src_stride)
        };

        for x in 0..n_left {
            let c = x * 2;
            let r = c + 1;
            let l_off: isize = if n_left & 1 != 0 {
                c as isize - 1
            } else {
                imax(c as i32 - 1, 0) as isize
            };
            dst[dst_lp + x] = cfl_filter(
                src,
                sp + c,
                (sp as isize + l_off) as usize,
                sp + r,
                b as usize,
                tb,
                tp + c,
                filter_type,
            );
        }
        for x in n_left..n_left + tw {
            let c = x * 2;
            let r = c + 1;
            let l_idx = if n_left > 0 {
                c - 1
            } else {
                imax(c as i32 - 1, 0) as usize
            };
            dst[dst_p + x - n_left] = cfl_filter(
                src,
                sp + c,
                sp + l_idx,
                sp + r,
                b as usize,
                tb,
                tp + c,
                filter_type,
            );
        }
        sp += vstep as usize;
        dst_lp += n_left;
        dst_p += tw;
    }

    // bl: bottom-left extension rows. For valid streams refh >= th; a malformed
    // stream can derive an inconsistent CfL geometry where refh < th, so use a
    // saturating subtraction (yielding no extension rows) instead of a usize
    // underflow. No-op for valid input.
    let n_bl = refh.saturating_sub(th);
    for _y in 0..n_bl {
        let top_sp_bl = sp - src_stride;
        for x in 0..n_left {
            let c = x * 2;
            let r = c + 1;
            let l_off: isize = if n_left & 1 != 0 {
                c as isize - 1
            } else {
                imax(c as i32 - 1, 0) as isize
            };
            dst[dst_lp + x] = cfl_filter(
                src,
                sp + c,
                (sp as isize + l_off) as usize,
                sp + r,
                b as usize,
                src,
                top_sp_bl + c,
                filter_type,
            );
        }
        sp += vstep as usize;
        dst_lp += n_left;
    }
}

pub(crate) const CFL_MHCCP_MAX_EDGE_SAMPLES: usize = 386;
pub(crate) const CFL_MHCCP_MAX_LUMA_SIZE: usize = 4736;

#[inline(always)]
fn sqrnd<BD: BitDepth>(bd: BD, v: i32) -> i32 {
    let b = bd.bitdepth() as i32;
    let mid = 1 << (b - 1);
    (v * v + mid) >> b
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_gen_mat<BD: BitDepth>(
    bd: BD,
    mat: &mut [[i32; 3]; 3],
    imat: &mut [[u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2],
    y: &[BD::Pixel],
    y_off: usize,
    y_top_stride: usize,
    refw: usize,
    refh: usize,
    edge_flags: i32,
    dir: CflMhDir,
) {
    let bd_bits = bd.bitdepth() as i32;
    let has_t = edge_flags & CFL_HAS_TOP != 0;
    let has_l = edge_flags & CFL_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = y_off + n_top * y_top_stride + 64 * 64;

    for r in mat.iter_mut() {
        r.fill(0);
    }

    let mut n: usize = 0;

    let mat2sh = bd_bits - 1;
    if has_t {
        for i in 0..n_left {
            let v0: i32 = y[left_off + i].into();
            let neighbor: i32 = if i == 0 {
                y[left_off + i + (dir_t as usize | dir_l as usize)].into()
            } else {
                y[y_off].into()
            };
            let v1 = sqrnd(bd, neighbor);
            imat[0][n] = v0 as u16;
            imat[1][n] = v1 as u16;
            mat[0][0] += v0 * v0;
            mat[0][1] += v0 * v1;
            mat[0][2] += v0 << mat2sh;
            mat[1][1] += v1 * v1;
            mat[1][2] += v1 << mat2sh;
            n += 1;
        }
        let start: usize = if !dir_l && !has_l { 1 } else { 0 };
        let end = imax(
            start as i32,
            refw as i32 - n_left as i32 - 1 - (start == 0) as i32,
        ) as usize;
        for i in start..end {
            let v0: i32 = y[y_off + i].into();
            let yi = y_off + dir_t as usize * y_top_stride + i + dir_l as usize;
            let v1 = sqrnd(bd, y[yi].into());
            imat[0][n] = v0 as u16;
            imat[1][n] = v1 as u16;
            mat[0][0] += v0 * v0;
            mat[0][1] += v0 * v1;
            mat[0][2] += v0 << mat2sh;
            mat[1][1] += v1 * v1;
            mat[1][2] += v1 << mat2sh;
            n += 1;
        }
    }

    if has_l {
        //   for (i = 1 - start; i < refh - start - 1; i++)
        let start = (dir_t && !has_t) as i32;
        let begin = (1 - start) as usize;
        let end = imax(begin as i32, refh as i32 - start - 1) as usize;
        for i in begin..end {
            let v0: i32 = y[left_off + i * n_left].into();
            let ni = left_off + (i + dir_t as usize) * n_left + dir_l as usize;
            let v1 = sqrnd(bd, y[ni].into());
            imat[0][n] = v0 as u16;
            imat[1][n] = v1 as u16;
            mat[0][0] += v0 * v0;
            mat[0][1] += v0 * v1;
            mat[0][2] += v0 << mat2sh;
            mat[1][1] += v1 * v1;
            mat[1][2] += v1 << mat2sh;
            n += 1;
        }
    }

    mat[2][2] = (n as i32) << ((bd_bits - 1) << 1);

    if n > 0 {
        let nl2 = 31 - (n as u32).leading_zeros() as i32;
        let mat_sh = 22 - 2 * bd_bits - nl2 - (n as i32 & ((1 << nl2) - 1) != 0) as i32;
        if mat_sh > 0 {
            for i in 0..3 {
                for j in i..3 {
                    mat[i][j] <<= mat_sh;
                }
            }
        } else if mat_sh < 0 {
            for i in 0..3 {
                for j in i..3 {
                    mat[i][j] >>= -mat_sh;
                }
            }
        }
    }

    let add = 2 << (bd_bits - 8);
    mat[0][0] += add;
    mat[1][1] += add;
    mat[2][2] += add;
    mat[1][0] = mat[0][1];
    mat[2][0] = mat[0][2];
    mat[2][1] = mat[1][2];
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_calc_alphas<BD: BitDepth>(
    bd: BD,
    alpha: &mut [i32; 3],
    c: &[BD::Pixel],
    c_off: usize,
    top_sb_edge: Option<(&[BD::Pixel], usize)>,
    stride: usize,
    refw: usize,
    refh: usize,
    mat: &mut [[i32; 3]; 3],
    imat: &[[u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2],
    edge_flags: i32,
) {
    let bd_bits = bd.bitdepth() as i32;
    let has_t = edge_flags & CFL_HAS_TOP != 0;
    let has_l = edge_flags & CFL_HAS_LEFT != 0;
    let a2sh = bd_bits - 1;

    let mut n: usize = 0;
    if has_t {
        let (top, top_off) = if let Some((tsb, tsb_off)) = top_sb_edge {
            (tsb, tsb_off - has_l as usize)
        } else {
            (c, c_off - stride - has_l as usize)
        };
        let start: usize = if !has_l { 1 } else { 0 };
        let end = imax(start as i32, refw as i32 - 1 - (start == 0) as i32) as usize;
        for i in start..end {
            let v: i32 = top[top_off + i].into();
            alpha[0] += imat[0][n] as i32 * v;
            alpha[1] += imat[1][n] as i32 * v;
            alpha[2] += v << a2sh;
            n += 1;
        }
    }
    if has_l {
        let start = if has_t { 0 } else { 1 }; // = !has_t
        let end = if has_t { refh - 2 } else { refh - 1 };
        for i in start..end {
            let v: i32 = c[c_off + i * stride - 1].into();
            alpha[0] += imat[0][n] as i32 * v;
            alpha[1] += imat[1][n] as i32 * v;
            alpha[2] += v << a2sh;
            n += 1;
        }
    }

    if n > 0 {
        let nl2 = 31 - (n as u32).leading_zeros() as i32;
        let mat_sh = 22 - 2 * bd_bits - nl2 - (n as i32 & ((1 << nl2) - 1) != 0) as i32;
        if mat_sh > 0 {
            for a in alpha.iter_mut() {
                *a <<= mat_sh;
            }
        } else if mat_sh < 0 {
            for a in alpha.iter_mut() {
                *a >>= -mat_sh;
            }
        }
    }

    let mut tmp = [[0i32; 2]; 3];
    let (mut scale, mut sh) = get_div_scale_sh(mat[0][0]);
    tmp[0][0] = mul32(mat[0][1], scale, sh);
    tmp[0][1] = mul32(mat[0][2], scale, sh);
    alpha[0] = mul32(alpha[0], scale, sh);
    tmp[1][0] = mat[1][1] - mul32(mat[1][0], tmp[0][0], 16);
    tmp[1][1] = mat[1][2] - mul32(mat[1][0], tmp[0][1], 16);
    alpha[1] -= mul32(mat[1][0], alpha[0], 16);
    tmp[2][0] = mat[2][1] - mul32(mat[2][0], tmp[0][0], 16);
    tmp[2][1] = mat[2][2] - mul32(mat[2][0], tmp[0][1], 16);
    alpha[2] -= mul32(mat[2][0], alpha[0], 16);

    (scale, sh) = get_div_scale_sh(tmp[1][0]);
    tmp[1][1] = mul32(tmp[1][1], scale, sh);
    alpha[1] = mul32(alpha[1], scale, sh);
    tmp[2][1] -= mul32(tmp[2][0], tmp[1][1], 16);
    alpha[2] -= mul32(tmp[2][0], alpha[1], 16);

    (scale, sh) = get_div_scale_sh(tmp[2][1]);
    alpha[2] = mul32(alpha[2], scale, sh);
    alpha[1] -= mul32(tmp[1][1], alpha[2], 16);
    alpha[0] -= mul32(tmp[0][0], alpha[1], 16) + mul32(tmp[0][1], alpha[2], 16);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_mhccp_pred<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_off: usize,
    src_top_stride: usize,
    w: usize,
    h: usize,
    alpha: &[i32; 3],
    edge_flags: i32,
    dir: CflMhDir,
) {
    let has_t = edge_flags & CFL_HAS_TOP != 0;
    let has_l = edge_flags & CFL_HAS_LEFT != 0;
    let dir_t = dir == CflMhDir::Top;
    let dir_l = dir == CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;

    let mid = 1 << (bd.bitdepth() as i32 - 1);
    let a2v2 = mul32(alpha[2], mid, 16);
    let mut sp = src_off;
    let mut dp = 0usize;
    let mut y = 0usize;

    while y < dir_t as usize && has_t {
        let dst_row = &mut dst[dp..dp + w];
        for (x, dst_px) in dst_row.iter_mut().enumerate() {
            let v0: i32 = src[sp + x - src_top_stride].into();
            let v1 = sqrnd(bd, src[sp + x].into());
            *dst_px = bd.pixel_clip(mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v2);
        }
        sp += w;
        dp += dst_stride;
        y += 1;
    }

    while y < h {
        let mut x = 0usize;
        let dst_row = &mut dst[dp..dp + w];
        while x < dir_l as usize && has_l {
            let v0: i32 = src[left_off + y * n_left + dir_l as usize].into();
            let v1 = sqrnd(bd, src[sp].into());
            dst_row[0] = bd.pixel_clip(mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v2);
            x += 1;
        }
        for (rel_x, dst_px) in dst_row[x..].iter_mut().enumerate() {
            let x = x + rel_x;
            let v0_idx = if dir_t {
                sp + x - (((y > 0) as usize) | has_t as usize) * w
            } else if dir_l {
                sp + imax(x as i32 - 1, 0) as usize
            } else {
                sp + x
            };
            let v0: i32 = src[v0_idx].into();
            let v1 = sqrnd(bd, src[sp + x].into());
            *dst_px = bd.pixel_clip(mul32(alpha[0], v0, 16) + mul32(alpha[1], v1, 16) + a2v2);
        }
        sp += w;
        dp += dst_stride;
        y += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_pred_raw<BD: BitDepth>(
    bd: BD,
    y_plane: &[BD::Pixel],
    u_plane: &mut [BD::Pixel],
    v_plane: &mut [BD::Pixel],
    ytop_off: usize,
    utop_off: usize,
    vtop_off: usize,
    ysrc_off: usize,
    usrc_off: usize,
    vsrc_off: usize,
    ystride: isize,
    cstride: isize,
    wpad: usize,
    hpad: usize,
    w: usize,
    h: usize,
    flags: u32,
    implicit: bool,
    ss_hor: usize,
    ss_ver: usize,
) -> Result<(), ()> {
    #[inline(always)]
    fn add_off(base: usize, off: isize) -> usize {
        if off >= 0 {
            base.checked_add(off as usize)
                .expect("CfL reference offset overflow")
        } else {
            base.checked_sub((-off) as usize)
                .expect("CfL reference offset before plane start")
        }
    }

    #[inline(always)]
    fn rp<P: Pixel>(plane: &[P], base: usize, off: isize) -> i32 {
        plane[add_off(base, off)].into()
    }

    #[inline(always)]
    fn px<P: Pixel>(plane: &[P], base: usize, off: isize) -> P {
        plane[add_off(base, off)]
    }

    #[inline(always)]
    fn wp<P: Pixel>(plane: &mut [P], base: usize, off: isize, v: P) {
        let idx = add_off(base, off);
        plane[idx] = v;
    }

    if w == 0 || h == 0 || 4 * wpad > w || 4 * hpad > h || ystride <= 0 || cstride <= 0 {
        return Err(());
    }

    let has_t = flags & CFL_HAS_TOP as u32 != 0;
    let has_l = flags & CFL_HAS_LEFT as u32 != 0;
    let xlim = w - 4 * wpad;
    let ylim = h - 4 * hpad;
    let skiph = w == 64;
    let skipv = h == 64;
    let flt = flags & 3;

    {
        let ystr = ystride as i64;
        let cstr = cstride as i64;
        let yv = ystr << ss_ver; // luma row advance per chroma row
        let xh = 1i64 << ss_hor; // luma column step
        let ylim_i = ylim as i64;
        let xlim_i = xlim as i64;
        let ylen = y_plane.len() as i64;
        let ulen = u_plane.len() as i64;
        let vlen = v_plane.len() as i64;
        let subs = (ss_hor | ss_ver) != 0;
        let gauss = flt == CFL_FLT_TYPE_GAUSS as u32;
        let vstrip = flt == CFL_FLT_TYPE_VSTRIP as u32;

        let v_dn = if ss_ver == 1 { ystr } else { 0 };
        let fwd_c = if subs { 1 } else { 0 };
        let back_l = if subs && (gauss || vstrip) { 1 } else { 0 };

        let mut y_lo = i64::MAX;
        let mut y_hi = i64::MIN;
        if xlim > 0 && ylim > 0 {
            // apply-loop downsample, base ysrc_off (left column clamped >= 0).
            // The gaussian vertical tap reads the row above only for local rows
            // cy&31 != 0; the block's first row (cy=0) clamps `top` to the current
            // row, and every interior row's above-read is yrow-ystride >= ysrc_off.
            // So the apply loop never reads below ysrc_off — do not extend the low
            // bound by `up` (that falsely rejected top-edge gaussian CfL blocks).
            y_lo = y_lo.min(ysrc_off as i64);
            y_hi = y_hi.max(ysrc_off as i64 + (ylim_i - 1) * yv + (xlim_i - 1) * xh + fwd_c + v_dn);
        }
        if has_l && ylim > 0 {
            // left-edge downsample, base ysrc_off - (1 + ss_hor). The gaussian
            // above-tap is clamped to the current row on the first row (y==0), so
            // the column never reads below `base`; only `back_l` (horizontal reach)
            // extends the low bound, not `up`.
            let base = ysrc_off as i64 - (1 + ss_hor as i64);
            y_lo = y_lo.min(base - back_l);
            y_hi = y_hi.max(base + (ylim_i - 1) * yv + fwd_c + v_dn);
        }
        if has_t && xlim > 0 {
            // top-edge downsample, base ytop_off; vertical reach 0 at the SB edge
            let bottom = if flags & CFL_IS_TOP_SB_EDGE != 0 {
                0
            } else {
                v_dn
            };
            let up_t = if gauss { bottom } else { 0 };
            y_lo = y_lo.min(ytop_off as i64 - up_t);
            y_hi = y_hi.max(ytop_off as i64 + (xlim_i - 1) * xh + fwd_c + bottom);
        }
        if y_hi >= y_lo && (y_lo < 0 || y_hi >= ylen) {
            return Err(());
        }

        // The full w x h destination block is always written (DC fill, or apply
        // plus the bottom-row replication), regardless of alpha.
        let mut u_lo = usrc_off as i64;
        let mut u_hi = usrc_off as i64 + (h as i64 - 1) * cstr + w as i64 - 1;
        let mut v_lo = vsrc_off as i64;
        let mut v_hi = vsrc_off as i64 + (h as i64 - 1) * cstr + w as i64 - 1;
        if has_l {
            // left column at usrc_off-1; the y>=ylim tail reads one column back,
            // which for ylim==0 lands one stride before usrc_off-1.
            let extra = if ylim == 0 { cstr } else { 0 };
            u_lo = u_lo.min(usrc_off as i64 - 1 - extra);
            v_lo = v_lo.min(vsrc_off as i64 - 1 - extra);
            if ylim > 0 {
                u_hi = u_hi.max(usrc_off as i64 - 1 + (ylim_i - 1) * cstr);
                v_hi = v_hi.max(vsrc_off as i64 - 1 + (ylim_i - 1) * cstr);
            }
        }
        if has_t {
            // top row utop_off..utop_off+xlim-1; the xlim==0 tail reads utop_off-1.
            let j = xlim_i - 1;
            u_lo = u_lo.min(utop_off as i64 + j.min(0));
            u_hi = u_hi.max(utop_off as i64 + j.max(0));
            v_lo = v_lo.min(vtop_off as i64 + j.min(0));
            v_hi = v_hi.max(vtop_off as i64 + j.max(0));
        }
        if u_lo < 0 || u_hi >= ulen || v_lo < 0 || v_hi >= vlen {
            return Err(());
        }
    }

    let mut dc = [0i32; 3];
    let mut n_top = 0usize;
    let mut n_left = 0usize;
    let mut edge = [[BD::Pixel::default(); 8]; 3];
    let mut edge_i = 0usize;
    let mut sum_x = 0i32;
    let mut sum_xx = 0i32;
    let mut sum_y = [0i32; 2];
    let mut sum_xy = [0i32; 2];

    if implicit {
        if has_t && has_l {
            if w > h * 2 {
                n_top = 8;
            } else if h > w * 2 {
                n_left = 8;
            } else {
                n_top = 4;
                n_left = 4;
            }
        } else {
            n_top = if has_t { imin(8, w as i32) as usize } else { 0 };
            n_left = if has_l { imin(8, h as i32) as usize } else { 0 };
        }
    }

    if has_l {
        let mut yleft = add_off(ysrc_off, -((1 + ss_hor) as isize));
        let mut uleft = add_off(usrc_off, -1);
        let mut vleft = add_off(vsrc_off, -1);
        let step = if n_left != 0 {
            h >> (n_left as u32).trailing_zeros()
        } else {
            0
        };
        let mut l = 0i32;
        for y in 0..ylim {
            l = if (ss_hor | ss_ver) == 0 {
                rp(y_plane, yleft, 0) << 3
            } else if ss_ver == 0 {
                if flt == CFL_FLT_TYPE_GAUSS as u32 {
                    rp(y_plane, yleft, 0) << 3
                } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
                    (rp(y_plane, yleft, -1) + 2 * rp(y_plane, yleft, 0) + rp(y_plane, yleft, 1))
                        << 1
                } else {
                    (rp(y_plane, yleft, 0) + rp(y_plane, yleft, 1)) << 2
                }
            } else if flt == CFL_FLT_TYPE_GAUSS as u32 {
                rp(y_plane, yleft, -1)
                    + 4 * rp(y_plane, yleft, 0)
                    + rp(y_plane, yleft, 1)
                    + rp(y_plane, yleft, if y != 0 { -ystride } else { 0 })
                    + rp(y_plane, yleft, ystride)
            } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
                rp(y_plane, yleft, -1)
                    + 2 * rp(y_plane, yleft, 0)
                    + rp(y_plane, yleft, 1)
                    + rp(y_plane, yleft, -1 + ystride)
                    + 2 * rp(y_plane, yleft, ystride)
                    + rp(y_plane, yleft, 1 + ystride)
            } else {
                (rp(y_plane, yleft, 0)
                    + rp(y_plane, yleft, 1)
                    + rp(y_plane, yleft, ystride)
                    + rp(y_plane, yleft, 1 + ystride))
                    << 1
            };
            if !skipv || (y & 1) == 0 {
                dc[0] += l;
                dc[1] += u_plane[uleft].into();
                dc[2] += v_plane[vleft].into();
            }
            if implicit && n_left != 0 && (((y & (step - 1)) ^ (step >> 1)) == 0) {
                edge[0][edge_i] = BD::Pixel::from_i32(l >> 3);
                edge[1][edge_i] = u_plane[uleft];
                edge[2][edge_i] = v_plane[vleft];
                edge_i += 1;
            }
            yleft = add_off(yleft, ystride << ss_ver);
            uleft = add_off(uleft, cstride);
            vleft = add_off(vleft, cstride);
        }
        for y in ylim..h {
            if !skipv || (y & 1) == 0 {
                dc[0] += l;
                dc[1] += rp(u_plane, uleft, -cstride);
                dc[2] += rp(v_plane, vleft, -cstride);
            }
            if implicit && n_left != 0 && (((y & (step - 1)) ^ (step >> 1)) == 0) {
                edge[0][edge_i] = BD::Pixel::from_i32(l >> 3);
                edge[1][edge_i] = px(u_plane, uleft, -cstride);
                edge[2][edge_i] = px(v_plane, vleft, -cstride);
                edge_i += 1;
            }
        }
    }

    if has_t {
        let step = if n_top != 0 {
            w >> (n_top as u32).trailing_zeros()
        } else {
            0
        };
        let mut l = 0i32;
        for x in 0..xlim {
            let xl = (x << ss_hor) as isize;
            l = if (ss_hor | ss_ver) == 0 {
                rp(y_plane, ytop_off, xl) << 3
            } else if ss_ver == 0 {
                if flt == CFL_FLT_TYPE_GAUSS as u32 {
                    rp(y_plane, ytop_off, xl) << 3
                } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
                    let left = imax(0, xl as i32 - 1) as isize;
                    (rp(y_plane, ytop_off, left)
                        + 2 * rp(y_plane, ytop_off, xl)
                        + rp(y_plane, ytop_off, xl + 1))
                        << 1
                } else {
                    (rp(y_plane, ytop_off, xl) + rp(y_plane, ytop_off, xl + 1)) << 2
                }
            } else {
                let bottom = if flags & CFL_IS_TOP_SB_EDGE != 0 {
                    0
                } else {
                    ystride
                };
                if flt == CFL_FLT_TYPE_GAUSS as u32 {
                    let left = imax(0, xl as i32 - 1) as isize;
                    rp(y_plane, ytop_off, left)
                        + 4 * rp(y_plane, ytop_off, xl)
                        + rp(y_plane, ytop_off, xl + 1)
                        + rp(y_plane, ytop_off, xl - bottom)
                        + rp(y_plane, ytop_off, xl + bottom)
                } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
                    let left = imax(0, xl as i32 - 1) as isize;
                    rp(y_plane, ytop_off, left)
                        + 2 * rp(y_plane, ytop_off, xl)
                        + rp(y_plane, ytop_off, xl + 1)
                        + rp(y_plane, ytop_off, left + bottom)
                        + 2 * rp(y_plane, ytop_off, xl + bottom)
                        + rp(y_plane, ytop_off, xl + bottom + 1)
                } else {
                    (rp(y_plane, ytop_off, xl)
                        + rp(y_plane, ytop_off, xl + 1)
                        + rp(y_plane, ytop_off, xl + bottom)
                        + rp(y_plane, ytop_off, xl + bottom + 1))
                        << 1
                }
            };
            if !skiph || (x & 1) == 0 {
                dc[0] += l;
                dc[1] += rp(u_plane, utop_off, x as isize);
                dc[2] += rp(v_plane, vtop_off, x as isize);
            }
            if implicit && n_top != 0 && (((x & (step - 1)) ^ (step >> 1)) == 0) {
                edge[0][edge_i] = BD::Pixel::from_i32(l >> 3);
                edge[1][edge_i] = px(u_plane, utop_off, x as isize);
                edge[2][edge_i] = px(v_plane, vtop_off, x as isize);
                edge_i += 1;
            }
        }
        for x in xlim..w {
            if !skiph || (x & 1) == 0 {
                dc[0] += l;
                dc[1] += rp(u_plane, utop_off, xlim as isize - 1);
                dc[2] += rp(v_plane, vtop_off, xlim as isize - 1);
            }
            if implicit && n_top != 0 && (((x & (step - 1)) ^ (step >> 1)) == 0) {
                edge[0][edge_i] = BD::Pixel::from_i32(l >> 3);
                edge[1][edge_i] = px(u_plane, utop_off, xlim as isize - 1);
                edge[2][edge_i] = px(v_plane, vtop_off, xlim as isize - 1);
                edge_i += 1;
            }
        }
    }

    if !has_t && !has_l {
        dc[0] = 4 << bd.bitdepth();
        dc[1] = (bd.bitdepth_max() + 1) >> 1;
        dc[2] = (bd.bitdepth_max() + 1) >> 1;
    } else {
        let npx = (if has_t { w >> skiph as usize } else { 0 })
            + (if has_l { h >> skipv as usize } else { 0 });
        if npx & (npx - 1) == 0 {
            dc[0] = (dc[0] + (npx as i32 >> 1)) >> (npx as u32).trailing_zeros();
            dc[1] = (dc[1] + (npx as i32 >> 1)) >> (npx as u32).trailing_zeros();
            dc[2] = (dc[2] + (npx as i32 >> 1)) >> (npx as u32).trailing_zeros();
        } else {
            dc[0] = fast_div32_dc(dc[0] as u32, npx as u32) as i32;
            dc[1] = fast_div32_dc(dc[1] as u32, npx as u32) as i32;
            dc[2] = fast_div32_dc(dc[2] as u32, npx as u32) as i32;
        }
    }

    let mut alpha = [0i32; 2];
    if implicit {
        debug_assert_eq!(edge_i, n_top + n_left);
        for j in 0..n_top + n_left {
            let e0: i32 = edge[0][j].into();
            let e1: i32 = edge[1][j].into();
            let e2: i32 = edge[2][j].into();
            sum_x += e0;
            sum_y[0] += e1;
            sum_y[1] += e2;
            sum_xx += e0 * e0;
            sum_xy[0] += e0 * e1;
            sum_xy[1] += e0 * e2;
        }
        let count_l2 = if n_top + n_left > 0 {
            (n_top as u32 + n_left as u32).trailing_zeros()
        } else {
            0
        };
        let den = sum_xx - (((sum_x as i64 * sum_x as i64) >> count_l2) as i32);
        for pl in 0..2 {
            let num = sum_xy[pl] - (((sum_x as i64 * sum_y[pl] as i64) >> count_l2) as i32);
            alpha[pl] = derive_alpha(num, den, 0);
        }
    } else {
        let shu = CFL_ALPHA_U_SHIFT - 5;
        let shv = CFL_ALPHA_V_SHIFT - 5;
        alpha[0] = ((flags & CFL_ALPHA_U_MASK) as i16 as i32) >> shu;
        alpha[1] = ((flags & CFL_ALPHA_V_MASK) as i32) >> shv;
    }

    if alpha[0] == 0 {
        let val = bd.pixel_clip(dc[1]);
        for y in 0..h {
            let row = add_off(usrc_off, y as isize * cstride);
            for x in 0..w {
                wp(u_plane, row, x as isize, val);
            }
        }
    }
    if alpha[1] == 0 {
        let val = bd.pixel_clip(dc[2]);
        for y in 0..h {
            let row = add_off(vsrc_off, y as isize * cstride);
            for x in 0..w {
                wp(v_plane, row, x as isize, val);
            }
        }
    }

    if let (Some(y_u8), Some(u_u8), Some(v_u8)) = (
        BD::Pixel::try_as_u8_slice(y_plane),
        BD::Pixel::try_as_u8_slice_mut(u_plane),
        BD::Pixel::try_as_u8_slice_mut(v_plane),
    ) {
        let cfl_layout = crate::cfl_dispatch::CflLayout {
            yrow0: ysrc_off,
            urow0: usrc_off,
            vrow0: vsrc_off,
            ystride: ystride as usize,
            cstride: cstride as usize,
        };
        let cfl_area = crate::cfl_dispatch::CflArea { w, h, xlim, ylim };
        let cfl_params = crate::cfl_dispatch::CflParams {
            dc0: dc[0],
            dc1: dc[1],
            dc2: dc[2],
            alpha0: alpha[0],
            alpha1: alpha[1],
            filter_type: flt,
        };

        match (ss_hor, ss_ver) {
            (0, 0) => {
                crate::cfl_dispatch::cfl_apply_444_8bpc(crate::cfl_dispatch::CflApply8 {
                    y: y_u8,
                    u: u_u8,
                    v: v_u8,
                    layout: cfl_layout,
                    area: cfl_area,
                    params: cfl_params,
                });
                return Ok(());
            }
            (1, 0) => {
                crate::cfl_dispatch::cfl_apply_422_8bpc(crate::cfl_dispatch::CflApply8 {
                    y: y_u8,
                    u: u_u8,
                    v: v_u8,
                    layout: cfl_layout,
                    area: cfl_area,
                    params: cfl_params,
                });
                return Ok(());
            }
            (1, 1) => {
                crate::cfl_dispatch::cfl_apply_420_8bpc(crate::cfl_dispatch::CflApply8 {
                    y: y_u8,
                    u: u_u8,
                    v: v_u8,
                    layout: cfl_layout,
                    area: cfl_area,
                    params: cfl_params,
                });
                return Ok(());
            }
            _ => {}
        }
    }

    if let (Some(y_u16), Some(u_u16), Some(v_u16)) = (
        BD::Pixel::try_as_u16_slice(y_plane),
        BD::Pixel::try_as_u16_slice_mut(u_plane),
        BD::Pixel::try_as_u16_slice_mut(v_plane),
    ) {
        let bitdepth_max = bd.bitdepth_max();
        let cfl_layout = crate::cfl_dispatch::CflLayout {
            yrow0: ysrc_off,
            urow0: usrc_off,
            vrow0: vsrc_off,
            ystride: ystride as usize,
            cstride: cstride as usize,
        };
        let cfl_area = crate::cfl_dispatch::CflArea { w, h, xlim, ylim };
        let cfl_params = crate::cfl_dispatch::CflParams {
            dc0: dc[0],
            dc1: dc[1],
            dc2: dc[2],
            alpha0: alpha[0],
            alpha1: alpha[1],
            filter_type: flt,
        };

        match (ss_hor, ss_ver) {
            (0, 0) => {
                crate::cfl_dispatch::cfl_apply_444_hbd(crate::cfl_dispatch::CflApplyHbd {
                    y: y_u16,
                    u: u_u16,
                    v: v_u16,
                    layout: cfl_layout,
                    area: cfl_area,
                    params: cfl_params,
                    bitdepth_max,
                });
                return Ok(());
            }
            (1, 0) => {
                crate::cfl_dispatch::cfl_apply_422_hbd(crate::cfl_dispatch::CflApplyHbd {
                    y: y_u16,
                    u: u_u16,
                    v: v_u16,
                    layout: cfl_layout,
                    area: cfl_area,
                    params: cfl_params,
                    bitdepth_max,
                });
                return Ok(());
            }
            (1, 1) => {
                crate::cfl_dispatch::cfl_apply_420_hbd(crate::cfl_dispatch::CflApplyHbd {
                    y: y_u16,
                    u: u_u16,
                    v: v_u16,
                    layout: cfl_layout,
                    area: cfl_area,
                    params: cfl_params,
                    bitdepth_max,
                });
                return Ok(());
            }
            _ => {}
        }
    }

    let mut yrow = ysrc_off;
    let mut urow = usrc_off;
    let mut vrow = vsrc_off;
    for y in 0..ylim {
        for x in 0..xlim {
            let xl = (x << ss_hor) as isize;
            let left = imax((xl as i32) & -64, xl as i32 - 1) as isize;
            let ac = if (ss_hor | ss_ver) == 0 {
                rp(y_plane, yrow, x as isize) << 3
            } else if ss_ver == 0 {
                if flt == CFL_FLT_TYPE_GAUSS as u32 {
                    rp(y_plane, yrow, xl) << 3
                } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
                    (rp(y_plane, yrow, left)
                        + 2 * rp(y_plane, yrow, xl)
                        + rp(y_plane, yrow, xl + 1))
                        << 1
                } else {
                    (rp(y_plane, yrow, xl) + rp(y_plane, yrow, xl + 1)) << 2
                }
            } else if flt == CFL_FLT_TYPE_GAUSS as u32 {
                let top = if (y & 31) == 0 { xl } else { xl - ystride };
                rp(y_plane, yrow, left)
                    + 4 * rp(y_plane, yrow, xl)
                    + rp(y_plane, yrow, xl + 1)
                    + rp(y_plane, yrow, top)
                    + rp(y_plane, yrow, xl + ystride)
            } else if flt == CFL_FLT_TYPE_VSTRIP as u32 {
                rp(y_plane, yrow, left)
                    + 2 * rp(y_plane, yrow, xl)
                    + rp(y_plane, yrow, xl + 1)
                    + rp(y_plane, yrow, left + ystride)
                    + 2 * rp(y_plane, yrow, xl + ystride)
                    + rp(y_plane, yrow, xl + ystride + 1)
            } else {
                (rp(y_plane, yrow, xl)
                    + rp(y_plane, yrow, xl + 1)
                    + rp(y_plane, yrow, xl + ystride)
                    + rp(y_plane, yrow, xl + ystride + 1))
                    << 1
            } - dc[0];

            if alpha[0] != 0 {
                let diff = alpha[0] * ac;
                let val = dc[1] + apply_sign((diff.abs() + 1024) >> 11, diff);
                wp(u_plane, urow, x as isize, bd.pixel_clip(val));
            }
            if alpha[1] != 0 {
                let diff = alpha[1] * ac;
                let val = dc[2] + apply_sign((diff.abs() + 1024) >> 11, diff);
                wp(v_plane, vrow, x as isize, bd.pixel_clip(val));
            }
        }
        if alpha[0] != 0 {
            let last = px(u_plane, urow, xlim as isize - 1);
            for xpad in xlim..w {
                wp(u_plane, urow, xpad as isize, last);
            }
        }
        if alpha[1] != 0 {
            let last = px(v_plane, vrow, xlim as isize - 1);
            for xpad in xlim..w {
                wp(v_plane, vrow, xpad as isize, last);
            }
        }
        yrow = add_off(yrow, ystride << ss_ver);
        urow = add_off(urow, cstride);
        vrow = add_off(vrow, cstride);
    }

    if alpha[0] != 0 {
        let src_row = add_off(usrc_off, (ylim as isize - 1) * cstride);
        for y in ylim..h {
            let dst_row = add_off(usrc_off, y as isize * cstride);
            u_plane.copy_within(src_row..src_row + w, dst_row);
        }
    }
    if alpha[1] != 0 {
        let src_row = add_off(vsrc_off, (ylim as isize - 1) * cstride);
        for y in ylim..h {
            let dst_row = add_off(vsrc_off, y as isize * cstride);
            v_plane.copy_within(src_row..src_row + w, dst_row);
        }
    }

    Ok(())
}
