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

use crate::intops::{iclip, imax, imin};
use crate::pixel::{BitDepth, BitDepth8, Pixel};
use crate::tables::{EXT_WARP_FILTER, MC_SUBPEL_FILTERS, MC_WARP_FILTER};

/// HBD (10-bit -> 4, 12-bit -> 2).
#[inline(always)]
pub(crate) fn intermediate_bits<BD: BitDepth>(bd: BD) -> i32 {
    if BD::BPC == 8 {
        4
    } else {
        14 - bd.bitdepth() as i32
    }
}

/// stays `int16_t` for both; the bias keeps HBD prepped values in i16 range.
#[inline(always)]
fn prep_bias<BD: BitDepth>(_bd: BD) -> i32 {
    if BD::BPC == 8 { 0 } else { 8192 }
}

pub(crate) fn put<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    src: &[P],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for (dst_row, src_row) in dst
        .chunks_mut(dst_stride)
        .zip(src.chunks(src_stride))
        .take(h)
    {
        dst_row[..w].copy_from_slice(&src_row[..w]);
    }
}

pub(crate) fn prep<BD: BitDepth>(
    bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    let ib = intermediate_bits(bd);
    let bias = prep_bias(bd);
    for (tmp_row, src_row) in tmp
        .chunks_mut(tmp_stride)
        .zip(src.chunks(src_stride))
        .take(h)
    {
        for (d, &s) in tmp_row[..w].iter_mut().zip(&src_row[..w]) {
            let s: i32 = s.into();
            *d = ((s << ib) - bias) as i16;
        }
    }
}

pub(crate) fn avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    avg(BitDepth8, dst, dst_stride, tmp1, tmp2, w, h);
}

pub(crate) fn avg<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    let ib = intermediate_bits(bd);
    let sh = ib + 1;
    let rnd = (1 << ib) + prep_bias(bd) * 2;
    for y in 0..h {
        let row = y * dst_stride;
        if row >= dst.len() {
            break;
        }
        let yw = y * w;
        let d = &mut dst[row..];
        let t1 = &tmp1[yw.min(tmp1.len())..];
        let t2 = &tmp2[yw.min(tmp2.len())..];
        let n = w.min(d.len()).min(t1.len()).min(t2.len());
        crate::filter::avg_row(bd, d, t1, t2, n, rnd, sh);
    }
}

pub(crate) fn w_avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    w_avg(BitDepth8, dst, dst_stride, tmp1, tmp2, w, h, weight);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_avg<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    let ib = intermediate_bits(bd);
    let sh = ib + 4;
    let rnd = (8 << ib) + prep_bias(bd) * 16;
    for y in 0..h {
        let row = y * dst_stride;
        if row >= dst.len() {
            break;
        }
        let yw = y * w;
        let d = &mut dst[row..];
        let t1 = &tmp1[yw.min(tmp1.len())..];
        let t2 = &tmp2[yw.min(tmp2.len())..];
        let n = w.min(d.len()).min(t1.len()).min(t2.len());
        crate::filter::w_avg_row(bd, d, t1, t2, n, weight, rnd, sh);
    }
}

pub(crate) fn mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    mask_fn(BitDepth8, dst, dst_stride, tmp1, tmp2, w, h, mask);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mask_fn<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    let ib = intermediate_bits(bd);
    let sh = ib + 6;
    let rnd = (32 << ib) + prep_bias(bd) * 64;
    for y in 0..h {
        let row = y * dst_stride;
        if row >= dst.len() {
            break;
        }
        let yw = y * w;
        let d = &mut dst[row..];
        let t1 = &tmp1[yw.min(tmp1.len())..];
        let t2 = &tmp2[yw.min(tmp2.len())..];
        let mk = &mask[yw.min(mask.len())..];
        let n = w.min(d.len()).min(t1.len()).min(t2.len()).min(mk.len());
        crate::filter::mask_row(bd, d, t1, t2, mk, n, rnd, sh);
    }
}

pub(crate) fn blend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    blend(dst, dst_stride, tmp, w, h, mask);
}

pub(crate) fn blend<P: Pixel>(
    dst: &mut [P],
    dst_stride: usize,
    tmp: &[P],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    for y in 0..h {
        let row = y * dst_stride;
        if row >= dst.len() {
            break;
        }
        let yw = y * w;
        let d = &mut dst[row..];
        let t = &tmp[yw.min(tmp.len())..];
        let mk = &mask[yw.min(mask.len())..];
        let n = w.min(d.len()).min(t.len()).min(mk.len());
        crate::filter::blend_row(d, t, mk, n);
    }
}

pub(crate) fn morph<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    alpha: i32,
    beta: i32,
    w: usize,
    h: usize,
) {
    for y in 0..h {
        let row = y * dst_stride;
        if row >= dst.len() {
            break;
        }
        let d = &mut dst[row..];
        let n = w.min(d.len());
        crate::filter::morph_row(bd, d, alpha, beta, n);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emu_edge<P: Pixel>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: isize,
    y: isize,
    dst: &mut [P],
    dst_stride: usize,
    r: &[P],
    ref_stride: usize,
) {
    let ref_y = iclip(y as i32, 0, ih as i32 - 1) as usize;
    let ref_x = iclip(x as i32, 0, iw as i32 - 1) as usize;
    let ref_off = ref_y * ref_stride + ref_x;

    let left_ext = iclip(-x as i32, 0, bw as i32 - 1) as usize;
    let right_ext = iclip((x + bw as isize - iw as isize) as i32, 0, bw as i32 - 1) as usize;
    let top_ext = iclip(-y as i32, 0, bh as i32 - 1) as usize;
    let bottom_ext = iclip((y + bh as isize - ih as isize) as i32, 0, bh as i32 - 1) as usize;

    let center_w = bw - left_ext - right_ext;
    let center_h = bh - top_ext - bottom_ext;

    let mut roff = ref_off;
    for cy in 0..center_h {
        let blk_y = top_ext + cy;
        let blk_off = blk_y * dst_stride;
        // The clamped source offset/width stay within the reference plane for
        // valid streams (iw/ih == the reference buffer size). Clamp both the row
        // offset and the read width against the actual buffer length so a
        // malformed reference whose declared dimensions (iw/ih) exceed its real
        // allocation extends the last available pixel instead of reading out of
        // bounds. No-op for valid input.
        let rstart = roff.min(r.len());
        let avail = r.len().saturating_sub(rstart).min(center_w);
        dst[blk_off + left_ext..blk_off + left_ext + avail]
            .copy_from_slice(&r[rstart..rstart + avail]);
        if avail < center_w {
            let fill = if avail > 0 {
                dst[blk_off + left_ext + avail - 1]
            } else if blk_off + left_ext > 0 {
                dst[blk_off + left_ext - 1]
            } else {
                P::default()
            };
            dst[blk_off + left_ext + avail..blk_off + left_ext + center_w].fill(fill);
        }
        if left_ext > 0 {
            let fill = dst[blk_off + left_ext];
            dst[blk_off..blk_off + left_ext].fill(fill);
        }
        if right_ext > 0 {
            let fill = dst[blk_off + left_ext + center_w - 1];
            dst[blk_off + left_ext + center_w..blk_off + bw].fill(fill);
        }
        roff += ref_stride;
    }

    let first_row_off = top_ext * dst_stride;
    for ty in 0..top_ext {
        dst.copy_within(first_row_off..first_row_off + bw, ty * dst_stride);
    }

    if bottom_ext > 0 {
        let last_row_start = (top_ext + center_h - 1) * dst_stride;
        for by in 0..bottom_ext {
            let dst_start = (top_ext + center_h + by) * dst_stride;
            dst.copy_within(last_row_start..last_row_start + bw, dst_start);
        }
    }
}

pub(crate) fn sad_nxn<P: Pixel>(
    p0: &[P],
    p0_stride: usize,
    p1: &[P],
    p1_stride: usize,
    w: usize,
    h: usize,
    bd_min8: i32,
) -> i32 {
    let mut sad = 0i32;
    let mut o0 = 0;
    let mut o1 = 0;
    let mut y = 0;
    while y < h {
        for x in 0..w {
            let a: i32 = p0[o0 + x].into();
            let b: i32 = p1[o1 + x].into();
            sad += (a - b).abs();
        }
        o0 += p0_stride * 2;
        o1 += p1_stride * 2;
        y += 2;
    }
    sad >> bd_min8
}

fn get_h_filter(mx: i32, filter_type: i32, w: usize) -> Option<[i8; 8]> {
    if mx == 0 {
        return None;
    }
    let f = if filter_type == -1 {
        EXT_WARP_FILTER[(mx - 1) as usize]
    } else if w > 4 {
        MC_SUBPEL_FILTERS[filter_type as usize][(mx - 1) as usize]
    } else {
        MC_SUBPEL_FILTERS[(3 + (filter_type & 1)) as usize][(mx - 1) as usize]
    };
    Some(f)
}

fn get_v_filter(my: i32, filter_type: i32, h: usize) -> Option<[i8; 8]> {
    if my == 0 {
        return None;
    }
    let f = if filter_type == -1 {
        EXT_WARP_FILTER[(my - 1) as usize]
    } else if h > 4 {
        MC_SUBPEL_FILTERS[filter_type as usize][(my - 1) as usize]
    } else {
        MC_SUBPEL_FILTERS[(3 + (filter_type & 1)) as usize][(my - 1) as usize]
    };
    Some(f)
}

#[inline(always)]
fn filter_8tap_px<P: Pixel>(src: &[P], center: usize, f: &[i8; 8], stride: isize) -> i32 {
    let c = center as isize;
    // For valid streams every tap index lies inside `src` (the MC path either
    // reads inside the padded reference or an emu-edge scratch buffer). Clamp
    // the index defensively so a malformed stream whose reference is smaller
    // than its declared dimensions can never read out of bounds; this is a
    // no-op for valid input where the index is already in range.
    let last = src.len().saturating_sub(1) as isize;
    let s = |i: isize| -> i32 { src[i.clamp(0, last) as usize].into() };
    f[0] as i32 * s(c - 3 * stride)
        + f[1] as i32 * s(c - 2 * stride)
        + f[2] as i32 * s(c - stride)
        + f[3] as i32 * s(c)
        + f[4] as i32 * s(c + stride)
        + f[5] as i32 * s(c + 2 * stride)
        + f[6] as i32 * s(c + 3 * stride)
        + f[7] as i32 * s(c + 4 * stride)
}

#[inline(always)]
fn filter_8tap_i16(mid: &[i16], center: usize, f: &[i8; 8], stride: isize) -> i32 {
    let c = center as isize;
    f[0] as i32 * mid[(c - 3 * stride) as usize] as i32
        + f[1] as i32 * mid[(c - 2 * stride) as usize] as i32
        + f[2] as i32 * mid[(c - stride) as usize] as i32
        + f[3] as i32 * mid[center] as i32
        + f[4] as i32 * mid[(c + stride) as usize] as i32
        + f[5] as i32 * mid[(c + 2 * stride) as usize] as i32
        + f[6] as i32 * mid[(c + 3 * stride) as usize] as i32
        + f[7] as i32 * mid[(c + 4 * stride) as usize] as i32
}

/// 8-tap subpixel MC filter. `src_off` is the origin in `src` — must have
/// 3 rows above and 4 rows below, and 3 cols left and 4 cols right of padding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    put_8tap(
        BitDepth8,
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    let bits = 6 + (filter_type < 0) as i32;
    let intermediate_bits = intermediate_bits(bd);
    let intermediate_rnd = ((1 << bits) >> 1) + ((1 << (bits - intermediate_bits)) >> 1);
    let fh = get_h_filter(mx, filter_type, w);
    let fv = get_v_filter(my, filter_type, h);
    let ss = src_stride as isize;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = vec![0i16; 64 * tmp_h];
            for y in 0..tmp_h {
                for x in 0..w {
                    let si = (src_off as isize
                        + (y as isize - 3) * src_stride as isize
                        + x as isize) as usize;
                    mid[y * 64 + x] = ((filter_8tap_px(src, si, &fh, 1)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits)) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = (y + 3) * 64 + x;
                    dst[y * dst_stride + x] = bd.pixel_clip(
                        (filter_8tap_i16(&mid, mi, &fv, 64)
                            + ((1 << (bits + intermediate_bits)) >> 1))
                            >> (bits + intermediate_bits),
                    );
                }
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    dst[y * dst_stride + x] =
                        bd.pixel_clip((filter_8tap_px(src, si, &fh, 1) + intermediate_rnd) >> bits);
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    dst[y * dst_stride + x] = bd.pixel_clip(
                        (filter_8tap_px(src, si, &fv, ss) + ((1 << bits) >> 1)) >> bits,
                    );
                }
            }
        }
        (None, None) => {
            put(dst, dst_stride, &src[src_off..], src_stride, w, h);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    prep_8tap(
        BitDepth8,
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap<BD: BitDepth>(
    bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[BD::Pixel],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    let bits = 6 + (filter_type < 0) as i32;
    let intermediate_bits = intermediate_bits(bd);
    let bias = prep_bias(bd) as i16;
    let fh = get_h_filter(mx, filter_type, w);
    let fv = get_v_filter(my, filter_type, h);
    let ss = src_stride as isize;

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = vec![0i16; 64 * tmp_h];
            for y in 0..tmp_h {
                for x in 0..w {
                    let si = (src_off as isize
                        + (y as isize - 3) * src_stride as isize
                        + x as isize) as usize;
                    mid[y * 64 + x] = ((filter_8tap_px(src, si, &fh, 1)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits)) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = (y + 3) * 64 + x;
                    tmp[y * tmp_stride + x] =
                        ((filter_8tap_i16(&mid, mi, &fv, 64) + ((1 << bits) >> 1)) >> bits) as i16
                            - bias;
                }
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    tmp[y * tmp_stride + x] = ((filter_8tap_px(src, si, &fh, 1)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits))
                        as i16
                        - bias;
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                for x in 0..w {
                    let si = src_off + y * src_stride + x;
                    tmp[y * tmp_stride + x] = ((filter_8tap_px(src, si, &fv, ss)
                        + ((1 << (bits - intermediate_bits)) >> 1))
                        >> (bits - intermediate_bits))
                        as i16
                        - bias;
                }
            }
        }
        (None, None) => {
            prep(bd, tmp, tmp_stride, &src[src_off..], src_stride, w, h);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    w_mask(
        BitDepth8,
        dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        mask,
        mask_stride,
        sign,
        ss_hor,
        ss_ver,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_mask<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    let ib = intermediate_bits(bd);
    let sh = ib + 6;
    let rnd = (32 << ib) + prep_bias(bd) * 64;
    let mask_sh = bd.bitdepth() as i32 + ib - 4;
    let mask_rnd = 1i32 << (mask_sh - 5);

    let mut t1off = 0usize;
    let mut t2off = 0usize;
    let mut doff = 0usize;
    let mut moff = 0usize;

    for row in 0..h {
        let mut x = 0usize;
        while x < w {
            let m = imin(
                38 + (((tmp1[t1off + x] as i32 - tmp2[t2off + x] as i32).abs() + mask_rnd)
                    >> mask_sh),
                64,
            );
            dst[doff + x] = bd.pixel_clip(
                (tmp1[t1off + x] as i32 * m + tmp2[t2off + x] as i32 * (64 - m) + rnd) >> sh,
            );

            if ss_hor {
                x += 1;
                let n = imin(
                    38 + (((tmp1[t1off + x] as i32 - tmp2[t2off + x] as i32).abs() + mask_rnd)
                        >> mask_sh),
                    64,
                );
                dst[doff + x] = bd.pixel_clip(
                    (tmp1[t1off + x] as i32 * n + tmp2[t2off + x] as i32 * (64 - n) + rnd) >> sh,
                );

                if row & 1 != 0 && ss_ver {
                    mask[moff + (x >> 1)] =
                        ((m + n + mask[moff + (x >> 1)] as i32 + 2 - sign) >> 2) as u8;
                } else if ss_ver {
                    mask[moff + (x >> 1)] = (m + n) as u8;
                } else {
                    mask[moff + (x >> 1)] = ((m + n + 1 - sign) >> 1) as u8;
                }
            } else {
                mask[moff + x] = m as u8;
            }
            x += 1;
        }

        t1off += w;
        t2off += w;
        doff += dst_stride;
        if !ss_ver || (row & 1 != 0) {
            moff += mask_stride;
        }
    }
}

#[inline(always)]
fn bilin(a: i32, b: i32, mxy: i32) -> i32 {
    16 * a + mxy * (b - a)
}

#[inline(always)]
fn bilin_rnd(a: i32, b: i32, mxy: i32, sh: i32) -> i32 {
    (bilin(a, b, mxy) + ((1 << sh) >> 1)) >> sh
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    put_bilin(BitDepth8, dst, dst_stride, src, src_stride, w, h, mx, my);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    let ib = intermediate_bits(bd);
    let intermediate_rnd = (1 << ib) >> 1;
    if mx != 0 {
        if my != 0 {
            let mut mid = vec![0i16; 64 * (h + 1)];
            for y in 0..h + 1 {
                for x in 0..w {
                    let si = y * src_stride + x;
                    let a: i32 = src[si].into();
                    let b: i32 = src[si + 1].into();
                    mid[y * 64 + x] = bilin_rnd(a, b, mx, 4 - ib) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = y * 64 + x;
                    dst[y * dst_stride + x] =
                        bd.pixel_clip(bilin_rnd(mid[mi] as i32, mid[mi + 64] as i32, my, 4 + ib));
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let si = y * src_stride + x;
                    let a: i32 = src[si].into();
                    let b: i32 = src[si + 1].into();
                    let px = bilin_rnd(a, b, mx, 4 - ib);
                    dst[y * dst_stride + x] = bd.pixel_clip((px + intermediate_rnd) >> ib);
                }
            }
        }
    } else if my != 0 {
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x;
                let a: i32 = src[si].into();
                let b: i32 = src[si + src_stride].into();
                dst[y * dst_stride + x] = bd.pixel_clip(bilin_rnd(a, b, my, 4));
            }
        }
    } else {
        put(dst, dst_stride, src, src_stride, w, h);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    prep_bilin(BitDepth8, tmp, tmp_stride, src, src_stride, w, h, mx, my);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin<BD: BitDepth>(
    bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    let ib = intermediate_bits(bd);
    let bias = prep_bias(bd) as i16;
    if mx != 0 {
        if my != 0 {
            let mut mid = vec![0i16; 64 * (h + 1)];
            for y in 0..h + 1 {
                for x in 0..w {
                    let si = y * src_stride + x;
                    let a: i32 = src[si].into();
                    let b: i32 = src[si + 1].into();
                    mid[y * 64 + x] = bilin_rnd(a, b, mx, 4 - ib) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let mi = y * 64 + x;
                    tmp[y * tmp_stride + x] =
                        bilin_rnd(mid[mi] as i32, mid[mi + 64] as i32, my, 4) as i16 - bias;
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let si = y * src_stride + x;
                    let a: i32 = src[si].into();
                    let b: i32 = src[si + 1].into();
                    tmp[y * tmp_stride + x] = bilin_rnd(a, b, mx, 4 - ib) as i16 - bias;
                }
            }
        }
    } else if my != 0 {
        for y in 0..h {
            for x in 0..w {
                let si = y * src_stride + x;
                let a: i32 = src[si].into();
                let b: i32 = src[si + src_stride].into();
                tmp[y * tmp_stride + x] = bilin_rnd(a, b, my, 4 - ib) as i16 - bias;
            }
        }
    } else {
        prep(bd, tmp, tmp_stride, src, src_stride, w, h);
    }
}

pub(crate) fn sad8x8<P: Pixel>(
    p0: &[P],
    p0_stride: usize,
    p1: &[P],
    p1_stride: usize,
    bd_min8: i32,
) -> u32 {
    let mut sad = 0u32;
    for y in 0..8 {
        for x in 0..8 {
            let a: i32 = p0[y * p0_stride + x].into();
            let b: i32 = p1[y * p1_stride + x].into();
            sad += (a - b).unsigned_abs();
        }
    }
    sad >> bd_min8
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sad_refine_mv<P: Pixel>(
    p0: &[P],
    p0_stride: usize,
    p1: &[P],
    p1_stride: usize,
    w: usize,
    h: usize,
    is_implicit: bool,
    bd_min8: i32,
) -> (i32, i32) {
    let sadw = w + 4;
    let sadh = h + 4;
    let sad_thr = (sadw * sadh * 2) as i32;
    let mut best_sad = i32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    if is_implicit {
        best_sad = sad_nxn(
            &p0[2 * p0_stride + 2..],
            p0_stride,
            &p1[2 * p1_stride + 2..],
            p1_stride,
            sadw,
            sadh,
            bd_min8,
        );
        best_sad = (best_sad * 7 + 7) >> 3;
        if best_sad < sad_thr {
            return (best_dx, best_dy);
        }
    }

    for y_off in -2i32..=2 {
        for x_off in -2i32..=2 {
            if x_off == 0 && y_off == 0 {
                continue;
            }
            let sad = sad_nxn(
                &p0[((2 + y_off) as usize) * p0_stride + (2 + x_off) as usize..],
                p0_stride,
                &p1[((2 - y_off) as usize) * p1_stride + (2 - x_off) as usize..],
                p1_stride,
                sadw,
                sadh,
                bd_min8,
            );
            if sad >= best_sad {
                continue;
            }
            best_sad = sad;
            best_dx = x_off;
            best_dy = y_off;
        }
    }
    (best_dx, best_dy)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct OpflRegressionData {
    pub(crate) su2: i32,
    pub(crate) suv: i32,
    pub(crate) sv2: i32,
    pub(crate) suw: i32,
    pub(crate) svw: i32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn opfl_derive_mv<BD: BitDepth>(
    bd: BD,
    out: &mut [OpflRegressionData],
    p0: &[BD::Pixel],
    p0_stride: usize,
    p1: &[BD::Pixel],
    p1_stride: usize,
    w: usize,
    h: usize,
    bs: usize,
    d: [i8; 2],
) {
    let mut tmp0 = [0i16; 64 * 16];
    let mut tmp1 = [0i16; 64 * 16];

    // (bd - 8) with round-to-nearest-away-from-zero; 8bpc leaves them as-is.
    let bd_min8 = bd.bitdepth_min_8();
    let rnd = (1 << bd_min8) >> 1;

    for y in 0..h {
        for x in 0..w {
            let p0p: i32 = p0[y * p0_stride + x].into();
            let p1p: i32 = p1[y * p1_stride + x].into();
            let v = d[0] as i32 * p0p + d[1] as i32 * p1p;
            if BD::BPC == 8 {
                tmp0[y * 64 + x] = v as i16;
                tmp1[y * 64 + x] = (p0p - p1p) as i16;
            } else {
                tmp0[y * 64 + x] = ((v + rnd - (v < 0) as i32) >> bd_min8) as i16;
                tmp1[y * 64 + x] = ((p0p - p1p + rnd - (p1p > p0p) as i32) >> bd_min8) as i16;
            }
        }
    }

    let mut gx0 = [0i16; 64 * 16];
    let mut gy0 = [0i16; 64 * 16];

    let mut bx = 0usize;
    while bx < w {
        let x_end = imin(bx as i32 + 16, w as i32) as usize;
        let min_x = bx & !15;
        let max_x = x_end - 1;
        let max_y = h - 1;
        for y in 0..h {
            for x in bx..x_end {
                let pa = tmp0[y * 64 + imax(min_x as i32, x as i32 - 2) as usize] as i32;
                let pb = tmp0[y * 64 + imax(min_x as i32, x as i32 - 1) as usize] as i32;
                let pc = tmp0[y * 64 + imin(max_x as i32, x as i32 + 1) as usize] as i32;
                let pd = tmp0[y * 64 + imin(max_x as i32, x as i32 + 2) as usize] as i32;
                let e1 = (x + 1 > max_x || x < 1 + min_x) as i32;
                let x0 = ((pc - pb) * 42 + (pd - pa) * -5) * (1 + e1);
                gx0[y * 64 + x] = ((x0 + 63 + (x0 > 0) as i32) >> 7) as i16;

                let qa = tmp0[imax(0, y as i32 - 2) as usize * 64 + x] as i32;
                let qb = tmp0[imax(0, y as i32 - 1) as usize * 64 + x] as i32;
                let qc = tmp0[imin(max_y as i32, y as i32 + 1) as usize * 64 + x] as i32;
                let qd = tmp0[imin(max_y as i32, y as i32 + 2) as usize * 64 + x] as i32;
                let e2 = (y + 1 > max_y || y < 1) as i32;
                let y0 = ((qc - qb) * 42 + (qd - qa) * -5) * (1 + e2);
                gy0[y * 64 + x] = ((y0 + 63 + (y0 > 0) as i32) >> 7) as i16;
            }
        }
        bx += 16;
    }

    let mut oi = 0;
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            let mut su2 = (bs * bs) as i32;
            let mut suv = 0i32;
            let mut sv2 = (bs * bs) as i32;
            let mut suw = 0i32;
            let mut svw = 0i32;
            for py in y..y + bs {
                for px in x..x + bs {
                    let u = gx0[py * 64 + px] as i32;
                    let v = gy0[py * 64 + px] as i32;
                    let ww = tmp1[py * 64 + px] as i32;
                    su2 += u * u;
                    suv += u * v;
                    sv2 += v * v;
                    suw += u * ww;
                    svw += v * ww;
                }
            }
            out[oi] = OpflRegressionData {
                su2,
                suv,
                sv2,
                suw,
                svw,
            };
            oi += 1;
            x += bs;
        }
        y += bs;
    }
}

fn filter_warp_rnd(src: &[i16], x: usize, f: &[i8; 8], stride: usize, sh: i32) -> i16 {
    let v = f[0] as i32 * src[x.wrapping_sub(3 * stride)] as i32
        + f[1] as i32 * src[x.wrapping_sub(2 * stride)] as i32
        + f[2] as i32 * src[x.wrapping_sub(stride)] as i32
        + f[3] as i32 * src[x] as i32
        + f[4] as i32 * src[x + stride] as i32
        + f[5] as i32 * src[x + 2 * stride] as i32
        + f[6] as i32 * src[x + 3 * stride] as i32
        + f[7] as i32 * src[x + 4 * stride] as i32
        + ((1 << sh) >> 1);
    (v >> sh) as i16
}

fn filter_warp_rnd_px<P: Pixel>(src: &[P], x: usize, f: &[i8; 8], stride: usize, sh: i32) -> i16 {
    let s = |i: usize| -> i32 { src[i].into() };
    let v = f[0] as i32 * s(x.wrapping_sub(3 * stride))
        + f[1] as i32 * s(x.wrapping_sub(2 * stride))
        + f[2] as i32 * s(x.wrapping_sub(stride))
        + f[3] as i32 * s(x)
        + f[4] as i32 * s(x + stride)
        + f[5] as i32 * s(x + 2 * stride)
        + f[6] as i32 * s(x + 3 * stride)
        + f[7] as i32 * s(x + 4 * stride)
        + ((1 << sh) >> 1);
    (v >> sh) as i16
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    warp_affine_8x8(
        BitDepth8, dst, dst_stride, src, src_stride, src_off, abcd, mx, my,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let ib = intermediate_bits(bd);
    let mut mid = [0i16; 15 * 8];

    let mut soff = src_off.wrapping_sub(3 * src_stride);
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let fi = (192 + ((tmx + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            mid[y * 8 + x] = filter_warp_rnd_px(src, soff + x, f, 1, 7 - ib);
            tmx += abcd[0] as i32;
        }
        soff += src_stride;
        mx += abcd[1] as i32;
    }

    for y in 0..8 {
        let mid_base = (3 + y) * 8;
        let mut tmy = my;
        for x in 0..8 {
            let fi = (192 + ((tmy + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            let v = filter_warp_rnd(&mid, mid_base + x, f, 8, 7 + ib);
            dst[y * dst_stride + x] = bd.pixel_clip(v as i32);
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8t_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    warp_affine_8x8t(
        BitDepth8, tmp, tmp_stride, src, src_stride, src_off, abcd, mx, my,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8t<BD: BitDepth>(
    bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let ib = intermediate_bits(bd);
    let bias = prep_bias(bd) as i16;
    let mut mid = [0i16; 15 * 8];

    let mut soff = src_off.wrapping_sub(3 * src_stride);
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let fi = (192 + ((tmx + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            mid[y * 8 + x] = filter_warp_rnd_px(src, soff + x, f, 1, 7 - ib);
            tmx += abcd[0] as i32;
        }
        soff += src_stride;
        mx += abcd[1] as i32;
    }

    for y in 0..8 {
        let mid_base = (3 + y) * 8;
        let mut tmy = my;
        for x in 0..8 {
            let fi = (192 + ((tmy + 512) >> 10)) as usize;
            let f = &MC_WARP_FILTER[fi];
            tmp[y * tmp_stride + x] = filter_warp_rnd(&mid, mid_base + x, f, 8, 7) - bias;
            tmy += abcd[2] as i32;
        }
        my += abcd[3] as i32;
    }
}
