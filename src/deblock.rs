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

use crate::headers::FrameHeader;
use crate::intops::{iclip, imin};
use crate::lf_mask::{deblock_quant_thr, deblock_side_thr};
use crate::pixel::{BitDepth, Pixel};

pub(crate) static MAX_WIDTH_Y: [i8; 4] = [1, 3, 6, 8];
pub(crate) static MAX_WIDTH_UV: [i8; 3] = [1, 3, 4];

pub(crate) static Q_FIRST: [i8; 3] = [45, 40, 32];
pub(crate) static Q_THRESH_MULTS: [i8; 8] = [32, 25, 19, 19, 0, 18, 0, 17];
pub(crate) static W_MULT: [i8; 8] = [85, 51, 37, 28, 0, 20, 0, 15];

pub(crate) fn init_deblock_thr_lut_y(
    frame_hdr: &FrameHeader,
    hbd: i32,
    dir: usize,
    qidx: i32,
    lut: &mut [[u32; 16]; 2],
) {
    let qmax = 255 + 48 * hbd;
    let seg = &frame_hdr.segmentation;
    let n = if seg.enabled != 0 { 8 } else { 1 };
    for i in 0..n {
        let yac = if seg.enabled != 0 {
            iclip(qidx + seg.d.delta_q[i] as i32, 0, qmax)
        } else {
            qidx
        };
        let dir_yac = yac + 8 * frame_hdr.deblock.delta_q_y[dir] as i32;
        lut[0][i] = deblock_quant_thr(hbd, dir_yac);
        lut[1][i] = deblock_side_thr(hbd, dir_yac);
    }
}

pub(crate) fn init_deblock_thr_lut_uv(
    frame_hdr: &FrameHeader,
    hbd: i32,
    qidx: i32,
    lut: &mut [[[u32; 16]; 2]; 2],
) {
    let qmax = 255 + 48 * hbd;
    let seg = &frame_hdr.segmentation;
    let n = if seg.enabled != 0 { 8 } else { 1 };
    for i in 0..n {
        let yac = if seg.enabled != 0 {
            iclip(qidx + seg.d.delta_q[i] as i32, 0, qmax)
        } else {
            qidx
        };
        let uac = yac + frame_hdr.quant.uac_delta as i32 + 8 * frame_hdr.deblock.delta_q_u as i32;
        lut[0][0][i] = deblock_quant_thr(hbd, uac);
        lut[0][1][i] = deblock_side_thr(hbd, uac);
        let vac = yac + frame_hdr.quant.vac_delta as i32 + 8 * frame_hdr.deblock.delta_q_v as i32;
        lut[1][0][i] = deblock_quant_thr(hbd, vac);
        lut[1][1][i] = deblock_side_thr(hbd, vac);
    }
}

#[inline(always)]
fn filter_choice_bd<P: Pixel>(
    buf: &[P],
    s: isize,
    t: isize,
    stride: isize,
    max_width_neg: i32,
    max_width_pos: i32,
    q_thr: u32,
    side_thr: u32,
) -> i32 {
    let at = |off: isize| -> i32 { buf[off as usize].into() };
    let mut sd = [0u32; 4];
    for dist in -2i32..2 {
        let d = dist as isize;
        let ds = (at(s + (d - 1) * stride) - at(s + d * stride) * 2 + at(s + (d + 1) * stride))
            .unsigned_abs();
        let dt = (at(t + (d - 1) * stride) - at(t + d * stride) * 2 + at(t + (d + 1) * stride))
            .unsigned_abs();
        sd[(dist + 2) as usize] = (ds + dt + 1) >> 1;
    }

    let high_deriv = sd[0].max(sd[3]);
    if high_deriv > side_thr {
        return 0;
    }
    if max_width_pos == 1 {
        return 1;
    }

    let side_thr2 = side_thr >> 2;
    let mut transition = sd[1] + sd[2];
    if high_deriv > side_thr2 {
        return 1;
    }
    if transition > q_thr * 4 {
        return 1;
    }

    let side_thr3 = side_thr >> 3;
    if high_deriv > side_thr3 {
        return 2;
    }
    if transition > q_thr * 3 {
        return 2;
    }

    let end_thr = (side_thr * 3) >> 4;

    if max_width_neg >= 3 {
        let ds = (at(s - stride) - at(s - 4 * stride) - 3 * (at(s - stride) - at(s - 2 * stride)))
            .unsigned_abs();
        let dt = (at(t - stride) - at(t - 4 * stride) - 3 * (at(t - stride) - at(t - 2 * stride)))
            .unsigned_abs();
        if ((ds + dt + 1) >> 1) > end_thr {
            return 2;
        }
    }

    let ds = (at(s) - at(s + 3 * stride) - 3 * (at(s) - at(s + stride))).unsigned_abs();
    let dt = (at(t) - at(t + 3 * stride) - 3 * (at(t) - at(t + stride))).unsigned_abs();
    if ((ds + dt + 1) >> 1) > end_thr {
        return 2;
    }
    if max_width_pos == 3 {
        return 3;
    }

    transition <<= 4;
    let mut prev_dist = 3i32;
    let mut dist = 4i32;
    while dist <= max_width_pos {
        let q_thr4 = q_thr * Q_FIRST[((dist - 4) >> 1) as usize] as u32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        if transition > q_thr4 {
            return prev_dist;
        }
        let dist2 = imin(7, dist);

        if max_width_neg >= dist2 {
            let ds = (at(s - stride)
                - at(s + (-dist2 as isize - 1) * stride)
                - dist2 * (at(s - stride) - at(s - 2 * stride)))
            .unsigned_abs();
            let dt = (at(t - stride)
                - at(t + (-dist2 as isize - 1) * stride)
                - dist2 * (at(t - stride) - at(t - 2 * stride)))
            .unsigned_abs();
            if ((ds + dt + 1) >> 1) > end_thr4 {
                return prev_dist;
            }
        }

        let ds = (at(s) - at(s + dist2 as isize * stride) - dist2 * (at(s) - at(s + stride)))
            .unsigned_abs();
        let dt = (at(t) - at(t + dist2 as isize * stride) - dist2 * (at(t) - at(t + stride)))
            .unsigned_abs();
        if ((ds + dt + 1) >> 1) > end_thr4 {
            return prev_dist;
        }

        prev_dist = dist;
        dist += 2;
    }

    max_width_pos
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn deblock_bd<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    off: isize,
    q_thr: u32,
    side_thr: u32,
    stridea: isize,
    strideb: isize,
    max_width_pos: i32,
    max_width_neg: i32,
    pos_lossless: bool,
    neg_lossless: bool,
) {
    let bdmax = bd.bitdepth_max();
    let width = filter_choice_bd(
        dst,
        off,
        off + 3 * stridea,
        strideb,
        max_width_neg,
        max_width_pos,
        q_thr,
        side_thr,
    );
    let width_neg = imin(width, max_width_neg);
    let width_pos = width;

    if width_pos < 1 || (neg_lossless && pos_lossless) {
        return;
    }

    let q_thr_clamp = q_thr as i32 * Q_THRESH_MULTS[(width - 1) as usize] as i32;
    if q_thr_clamp <= 0 {
        return;
    }

    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            crate::deblock_dispatch::deblock_apply_8bpc(
                d8,
                off,
                stridea,
                strideb,
                width_neg,
                width_pos,
                q_thr_clamp,
                neg_lossless,
                pos_lossless,
            );
            return;
        }
    } else if let Some(d16) = <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst) {
        crate::deblock_dispatch::deblock_apply_hbd(
            d16,
            off,
            stridea,
            strideb,
            width_neg,
            width_pos,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bdmax,
        );
        return;
    }

    let mut dp = off;
    for _ in 0..4 {
        let d0: i32 = dst[dp as usize].into();
        let dm1: i32 = dst[(dp - strideb) as usize].into();
        let dp1: i32 = dst[(dp + strideb) as usize].into();
        let dm2: i32 = dst[(dp - 2 * strideb) as usize].into();
        let delta_m2 = iclip(
            4 * (3 * (d0 - dm1) - (dp1 - dm2)),
            -q_thr_clamp,
            q_thr_clamp,
        );

        if !neg_lossless {
            let delta_m2_neg = delta_m2 * W_MULT[(width_neg - 1) as usize] as i32;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * strideb) as usize;
                let diff = (delta_m2_neg * (width_neg - j) + (1 << 10)) >> 11;
                let cur: i32 = dst[idx].into();
                dst[idx] = BD::Pixel::from_i32(iclip(cur + diff, 0, bdmax));
            }
        }

        if !pos_lossless {
            let delta_m2_pos = delta_m2 * W_MULT[(width_pos - 1) as usize] as i32;
            for j in 0..width_pos {
                let idx = (dp + j as isize * strideb) as usize;
                let diff = (delta_m2_pos * (width_pos - j) + (1 << 10)) >> 11;
                let cur: i32 = dst[idx].into();
                dst[idx] = BD::Pixel::from_i32(iclip(cur - diff, 0, bdmax));
            }
        }

        dp += stridea;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn deblock_h_sb64y_bd<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            if crate::deblock_dispatch::try_deblock_h_sb64y_8bpc(
                d8, dst_off, stride, vmask, ll_mask, q_thr, side_thr, edge,
            ) {
                return;
            }
        }
    }

    let mut vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32 | vmask[3] as u32;
    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let y = 1u32 << qi;
        let idx = if (vmask[3] as u32 & y) != 0 {
            3usize
        } else if (vmask[2] as u32 & y) != 0 {
            2
        } else {
            ((vmask[1] as u32 & y) != 0) as usize
        };
        let max_width_pos = MAX_WIDTH_Y[idx] as i32;
        let max_width_neg = if edge {
            imin(6, max_width_pos)
        } else {
            max_width_pos
        };
        deblock_bd(
            bd,
            dst,
            (dst_off + qi * 4 * stride) as isize,
            q_thr[qi] as u32,
            side_thr[qi] as u32,
            stride as isize,
            1,
            max_width_pos,
            max_width_neg,
            (ll_mask[1] as u32 & y) != 0,
            (ll_mask[0] as u32 & y) != 0,
        );
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn deblock_v_sb64y_bd<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            if crate::deblock_dispatch::try_deblock_v_sb64y_8bpc(
                d8, dst_off, stride, vmask, ll_mask, q_thr, side_thr, edge,
            ) {
                return;
            }
        }
    }

    let mut vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32 | vmask[3] as u32;
    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let x = 1u32 << qi;
        let idx = if (vmask[3] as u32 & x) != 0 {
            3usize
        } else if (vmask[2] as u32 & x) != 0 {
            2
        } else {
            ((vmask[1] as u32 & x) != 0) as usize
        };
        let max_width_pos = MAX_WIDTH_Y[idx] as i32;
        let max_width_neg = if edge {
            imin(6, max_width_pos)
        } else {
            max_width_pos
        };
        deblock_bd(
            bd,
            dst,
            (dst_off + qi * 4) as isize,
            q_thr[qi] as u32,
            side_thr[qi] as u32,
            1,
            stride as isize,
            max_width_pos,
            max_width_neg,
            (ll_mask[1] as u32 & x) != 0,
            (ll_mask[0] as u32 & x) != 0,
        );
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn deblock_h_sb64uv_bd<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            if crate::deblock_dispatch::try_deblock_h_sb64uv_8bpc(
                d8, dst_off, stride, vmask, ll_mask, q_thr, side_thr, edge,
            ) {
                return;
            }
        }
    }

    let mut vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32;
    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let y = 1u32 << qi;
        let idx = if (vmask[2] as u32 & y) != 0 {
            2usize
        } else {
            ((vmask[1] as u32 & y) != 0) as usize
        };
        let max_width_pos = MAX_WIDTH_UV[idx] as i32;
        let max_width_neg = if edge {
            imin(2, max_width_pos)
        } else {
            max_width_pos
        };
        deblock_bd(
            bd,
            dst,
            (dst_off + qi * 4 * stride) as isize,
            q_thr[qi] as u32,
            side_thr[qi] as u32,
            stride as isize,
            1,
            max_width_pos,
            max_width_neg,
            (ll_mask[1] as u32 & y) != 0,
            (ll_mask[0] as u32 & y) != 0,
        );
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn deblock_v_sb64uv_bd<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    if BD::BPC == 8 {
        if let Some(d8) = <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst) {
            if crate::deblock_dispatch::try_deblock_v_sb64uv_8bpc(
                d8, dst_off, stride, vmask, ll_mask, q_thr, side_thr, edge,
            ) {
                return;
            }
        }
    }

    let mut vm = vmask[0] as u32 | vmask[1] as u32 | vmask[2] as u32;
    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let x = 1u32 << qi;
        let idx = if (vmask[2] as u32 & x) != 0 {
            2usize
        } else {
            ((vmask[1] as u32 & x) != 0) as usize
        };
        let max_width_pos = MAX_WIDTH_UV[idx] as i32;
        let max_width_neg = if edge {
            imin(2, max_width_pos)
        } else {
            max_width_pos
        };
        deblock_bd(
            bd,
            dst,
            (dst_off + qi * 4) as isize,
            q_thr[qi] as u32,
            side_thr[qi] as u32,
            1,
            stride as isize,
            max_width_pos,
            max_width_neg,
            (ll_mask[1] as u32 & x) != 0,
            (ll_mask[0] as u32 & x) != 0,
        );
        vm &= vm - 1;
    }
}

pub(crate) fn backup_db(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    ss_ver: i32,
    sb128: bool,
    mut row: i32,
    row_h: i32,
    w: usize,
    lr_backup: bool,
    n_tc: i32,
) {
    let cdef_backup = (!lr_backup) as i32;
    let sb128_i = sb128 as i32;

    let mut stripe_h = ((64 << (cdef_backup & sb128_i)) - 8 * (row == 0) as i32) >> ss_ver;
    let mut src_off = (stripe_h - 2) as usize * stride;
    let mut dst_off = 0usize;

    if n_tc == 1 {
        if row > 0 {
            let top = 4usize << sb128_i;
            for i in 0..4usize {
                let from = dst_off + (top + i) * stride;
                let to = dst_off + i * stride;
                dst.copy_within(from..from + w, to);
            }
        }
        dst_off += 4 * stride;
    }

    while row + stripe_h <= row_h {
        for _ in 0..4 {
            dst[dst_off..dst_off + w].copy_from_slice(&src[src_off..src_off + w]);
            dst_off += stride;
            src_off += stride;
        }
        row += stripe_h;
        stripe_h = 64 >> ss_ver;
        src_off += (stripe_h - 4) as usize * stride;
    }
}

pub(crate) fn backup_db_hbd(
    dst: &mut [u16],
    src: &[u16],
    stride: usize,
    ss_ver: i32,
    sb128: bool,
    mut row: i32,
    row_h: i32,
    w: usize,
    lr_backup: bool,
    n_tc: i32,
) {
    let cdef_backup = (!lr_backup) as i32;
    let sb128_i = sb128 as i32;

    let mut stripe_h = ((64 << (cdef_backup & sb128_i)) - 8 * (row == 0) as i32) >> ss_ver;
    let mut src_off = (stripe_h - 2) as usize * stride;
    let mut dst_off = 0usize;

    if n_tc == 1 {
        if row > 0 {
            let top = 4usize << sb128_i;
            for i in 0..4usize {
                let from = dst_off + (top + i) * stride;
                let to = dst_off + i * stride;
                dst.copy_within(from..from + w, to);
            }
        }
        dst_off += 4 * stride;
    }

    while row + stripe_h <= row_h {
        for _ in 0..4 {
            dst[dst_off..dst_off + w].copy_from_slice(&src[src_off..src_off + w]);
            dst_off += stride;
            src_off += stride;
        }
        row += stripe_h;
        stripe_h = 64 >> ss_ver;
        src_off += (stripe_h - 4) as usize * stride;
    }
}

pub(crate) struct DeblockApplyParams {
    pub(crate) level_y: [i32; 2],
}

use crate::headers::PixelLayout;
use crate::lf_mask::{Av2Filter, transpose_lossless_mask as lf_transpose_lossless_mask};

/// Bundled per-frame inputs for the deblock pass, mirroring the fields
pub(crate) struct DeblockCtx<'a> {
    pub(crate) frame_hdr: &'a FrameHeader,
    pub(crate) mask: &'a [Av2Filter],
    pub(crate) mask_row: usize,
    pub(crate) sb256w: i32,
    pub(crate) cur_segmap: &'a [u8],
    pub(crate) b4_stride: isize,
    pub(crate) segmap_uv: &'a [u8],
    pub(crate) uv_segmap_stride: isize,
    pub(crate) hbd: i32,
    pub(crate) ss_hor: i32,
    pub(crate) ss_ver: i32,
    pub(crate) bw: i32,
    pub(crate) bh: i32,
    pub(crate) y_stride: isize,
    pub(crate) uv_stride: isize,
    pub(crate) layout: PixelLayout,
}

#[inline]
fn edge_thr(cur: i32, prev: i32) -> i32 {
    if cur != 0 && prev != 0 {
        (cur + prev + 1) >> 1
    } else {
        cur | prev
    }
}

#[inline(always)]
fn mask_has_luma_col(
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    sb64y: usize,
    w4: usize,
    have_left: bool,
) -> bool {
    let start = (!have_left) as usize;
    if start >= w4 {
        return false;
    }
    for hmask in mask[bx4_base + start..bx4_base + w4].iter() {
        if (hmask[0][sb64y] | hmask[1][sb64y] | hmask[2][sb64y] | hmask[3][sb64y]) != 0 {
            return true;
        }
    }
    false
}

#[inline(always)]
fn mask_has_luma_row(
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    sidx: usize,
    h4: usize,
    have_top: bool,
) -> bool {
    let start = (!have_top) as usize;
    if start >= h4 {
        return false;
    }
    for row in mask[starty4 + start..starty4 + h4].iter() {
        if (row[0][sidx] | row[1][sidx] | row[2][sidx] | row[3][sidx]) != 0 {
            return true;
        }
    }
    false
}

#[inline(always)]
fn mask_has_chroma_col(
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    y64: i32,
    ss_ver: i32,
    w4: usize,
    have_left: bool,
) -> bool {
    let start = (!have_left) as usize;
    if start >= w4 {
        return false;
    }
    let mask_idx = ((y64 & 3) >> ss_ver) as usize;
    let mask_shift: u32 = if (y64 & 3) & ss_ver != 0 { 8 } else { 0 };
    let bytes_mask: u32 = if ss_ver != 0 { 0xff } else { 0xffff };
    for hmask in mask[bx4_base + start..bx4_base + w4].iter() {
        let m0 = ((hmask[0][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
        let m1 = ((hmask[1][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
        let m2 = ((hmask[2][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
        if (m0 | m1 | m2) != 0 {
            return true;
        }
    }
    false
}

#[inline(always)]
fn mask_has_chroma_row(
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    sb64x: i32,
    ss_hor: i32,
    h4: usize,
    have_top: bool,
) -> bool {
    let start = (!have_top) as usize;
    if start >= h4 {
        return false;
    }
    let mask_idx = ((sb64x & 3) >> ss_hor) as usize;
    let mask_shift: u32 = if (sb64x & 3) & ss_hor != 0 { 8 } else { 0 };
    let bytes_mask: u32 = if ss_hor != 0 { 0xff } else { 0xffff };
    for row in mask[starty4 + start..starty4 + h4].iter() {
        let m0 = ((row[0][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
        let m1 = ((row[1][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
        let m2 = ((row[2][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
        if (m0 | m1 | m2) != 0 {
            return true;
        }
    }
    false
}

#[inline(always)]
fn fill_left_thr_from_lut(
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    lut: &[[u32; 16]; 2],
    h4: usize,
) {
    let q = lut[0][0] as u8;
    let side = lut[1][0] as u8;
    left_q_thr[..h4].fill(q);
    left_side_thr[..h4].fill(side);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn setup_thr_cols(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    segmap: &[u8],
    seg_off: isize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    // Use real asserts, not debug_asserts, because they give LLVM facts in release.
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));

    let w = w4 as usize;
    let h = h4 as usize;

    let mask_idx = (y64 >> ss_ver) as usize;
    assert!(mask_idx < 4);
    assert!(bx4_base + w <= 64);

    let mask_shift: u32 = if (y64 & ss_ver) != 0 { 8 } else { 0 };

    let q_lut = &thr_lut[0];
    let side_lut = &thr_lut[1];

    // Removes `bx4_base + x4` bounds checks from the inner loop.
    let mask_cols = &mask[bx4_base..bx4_base + w];

    for (y4, (left_q, left_side)) in left_q_thr
        .iter_mut()
        .zip(left_side_thr.iter_mut())
        .take(h)
        .enumerate()
    {
        let mut prev_q_thr = i32::from(*left_q);
        let mut prev_side_thr = i32::from(*left_side);

        // One segmap bounds check per row instead of one per coefficient.
        let row_start = (seg_off + y4 as isize * seg_stride) as usize;
        let seg_row = &segmap[row_start..row_start + w];

        // Transposed stores: q_thr_dst[x4 * 16 + y4].
        // Starting at y4 and stepping by 16 gives exactly that layout.
        let q_out = q_thr_dst[y4..].iter_mut().step_by(16).take(w);
        let side_out = side_thr_dst[y4..].iter_mut().step_by(16).take(w);

        for (((&seg, mask_col), q_dst), side_dst) in seg_row
            .iter()
            .zip(mask_cols.iter())
            .zip(q_out)
            .zip(side_out)
        {
            let seg_id = usize::from(seg);

            // This turns two data-dependent array bounds checks into one explicit
            // range check. If segmap is guaranteed valid, this branch is always cold.
            assert!(seg_id < 16);

            let cur_q_thr = q_lut[seg_id] as i32;
            let cur_side_thr = side_lut[seg_id] as i32;

            let subpu = 3 * (((mask_col[4][mask_idx] >> (mask_shift + y4 as u32)) & 1) as i32);

            let eq = edge_thr(cur_q_thr, prev_q_thr) >> subpu;
            let es = edge_thr(cur_side_thr, prev_side_thr) >> subpu;

            *q_dst = eq as u8;
            *side_dst = es as u8;

            prev_q_thr = cur_q_thr;
            prev_side_thr = cur_side_thr;
        }

        *left_q = prev_q_thr as u8;
        *left_side = prev_side_thr as u8;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn setup_thr_cols_simple(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));

    let w = w4 as usize;
    let h = h4 as usize;
    let mask_idx = (y64 >> ss_ver) as usize;
    assert!(mask_idx < 4);
    assert!(bx4_base + w <= 64);

    let mask_shift: u32 = if (y64 & ss_ver) != 0 { 8 } else { 0 };
    let q = thr_lut[0][0];
    let side = thr_lut[1][0];
    let mask_cols = &mask[bx4_base..bx4_base + w];

    for y4 in 0..h {
        let shift = mask_shift + y4 as u32;
        let q_out = q_thr_dst[y4..].iter_mut().step_by(16).take(w);
        let side_out = side_thr_dst[y4..].iter_mut().step_by(16).take(w);
        for ((mask_col, q_dst), side_dst) in mask_cols.iter().zip(q_out).zip(side_out) {
            let subpu = 3 * (((mask_col[4][mask_idx] >> shift) & 1) as u32);
            *q_dst = (q >> subpu) as u8;
            *side_dst = (side >> subpu) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn setup_thr_cols_dq(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));

    let w = w4 as usize;
    let h = h4 as usize;
    if w == 0 || h == 0 {
        return;
    }

    let mask_idx = (y64 >> ss_ver) as usize;
    assert!(mask_idx < 4);
    assert!(bx4_base + w <= 64);

    let mask_shift: u32 = if (y64 & ss_ver) != 0 { 8 } else { 0 };
    let q = thr_lut[0][0];
    let side = thr_lut[1][0];
    let mask_cols = &mask[bx4_base..bx4_base + w];

    for y4 in 0..h {
        let shift = mask_shift + y4 as u32;
        let q_out = q_thr_dst[y4..].iter_mut().step_by(16).take(w);
        let side_out = side_thr_dst[y4..].iter_mut().step_by(16).take(w);
        let first_subpu = 3 * (((mask_cols[0][4][mask_idx] >> shift) & 1) as u32);
        let first_q = (edge_thr(q as i32, i32::from(left_q_thr[y4])) >> first_subpu) as u8;
        let first_side = (edge_thr(side as i32, i32::from(left_side_thr[y4])) >> first_subpu) as u8;

        for (x4, ((mask_col, q_dst), side_dst)) in
            mask_cols.iter().zip(q_out).zip(side_out).enumerate()
        {
            if x4 == 0 {
                *q_dst = first_q;
                *side_dst = first_side;
            } else {
                let subpu = 3 * (((mask_col[4][mask_idx] >> shift) & 1) as u32);
                *q_dst = (q >> subpu) as u8;
                *side_dst = (side >> subpu) as u8;
            }
        }

        left_q_thr[y4] = q as u8;
        left_side_thr[y4] = side as u8;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn setup_thr_rows_simple(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));

    let w = w4 as usize;
    let h = h4 as usize;
    let mask_idx = (sb64x >> ss_hor) as usize;
    assert!(mask_idx < 4);
    assert!(starty4 + h <= 64);

    let mask_shift: u32 = if (sb64x & ss_hor) != 0 { 8 } else { 0 };
    let q = thr_lut[0][0];
    let side = thr_lut[1][0];
    let mask_rows = &mask[starty4..starty4 + h];

    for ((q_row, side_row), mask_row) in q_thr_dst
        .chunks_exact_mut(16)
        .zip(side_thr_dst.chunks_exact_mut(16))
        .zip(mask_rows.iter())
        .take(h)
    {
        let q_row = &mut q_row[..w];
        let side_row = &mut side_row[..w];
        for (x4, (q_dst, side_dst)) in q_row.iter_mut().zip(side_row.iter_mut()).enumerate() {
            let subpu = 3 * (((mask_row[4][mask_idx] >> (mask_shift + x4 as u32)) & 1) as u32);
            *q_dst = (q >> subpu) as u8;
            *side_dst = (side >> subpu) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn setup_thr_rows_dq(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    above_thr_lut: Option<&[[u32; 16]; 2]>,
    above_seg: Option<(&[u8], isize)>,
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));

    let w = w4 as usize;
    let h = h4 as usize;
    if w == 0 || h == 0 {
        return;
    }

    let mask_idx = (sb64x >> ss_hor) as usize;
    assert!(mask_idx < 4);
    assert!(starty4 + h <= 64);

    let mask_shift: u32 = if (sb64x & ss_hor) != 0 { 8 } else { 0 };
    let q = thr_lut[0][0];
    let side = thr_lut[1][0];
    let mask_rows = &mask[starty4..starty4 + h];

    let mut above_q = [0u8; 16];
    let mut above_side = [0u8; 16];
    if let Some(above_lut) = above_thr_lut {
        if let Some((aseg, aoff)) = above_seg {
            let aoff = usize::try_from(aoff).expect("negative above segment offset");
            assert!(aoff + w <= aseg.len());
            for (x4, &seg) in aseg[aoff..aoff + w].iter().enumerate() {
                let seg_id = usize::from(seg);
                assert!(seg_id < 16);
                above_q[x4] = above_lut[0][seg_id] as u8;
                above_side[x4] = above_lut[1][seg_id] as u8;
            }
        } else {
            above_q[..w].fill(above_lut[0][0] as u8);
            above_side[..w].fill(above_lut[1][0] as u8);
        }
    }

    for (y4, ((q_row, side_row), mask_row)) in q_thr_dst
        .chunks_exact_mut(16)
        .zip(side_thr_dst.chunks_exact_mut(16))
        .zip(mask_rows.iter())
        .take(h)
        .enumerate()
    {
        let q_row = &mut q_row[..w];
        let side_row = &mut side_row[..w];
        for (x4, (q_dst, side_dst)) in q_row.iter_mut().zip(side_row.iter_mut()).enumerate() {
            let subpu = 3 * (((mask_row[4][mask_idx] >> (mask_shift + x4 as u32)) & 1) as u32);
            if y4 == 0 {
                *q_dst = (edge_thr(q as i32, i32::from(above_q[x4])) >> subpu) as u8;
                *side_dst = (edge_thr(side as i32, i32::from(above_side[x4])) >> subpu) as u8;
            } else {
                *q_dst = (q >> subpu) as u8;
                *side_dst = (side >> subpu) as u8;
            }
        }
    }
}

/// Port of `setup_thr_rows_sb64`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn setup_thr_rows(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    segmap: &[u8],
    seg_off: isize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    above_thr_lut: Option<&[[u32; 16]; 2]>,
    above_seg: Option<(&[u8], isize)>,
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));

    let w = w4 as usize;
    let h = h4 as usize;

    let mask_idx = (sb64x >> ss_hor) as usize;
    assert!(mask_idx < 4);
    assert!(starty4 + h <= 64);

    let mask_shift: u32 = if (sb64x & ss_hor) != 0 { 8 } else { 0 };

    let mut prev_q_thr = [0i32; 16];
    let mut prev_side_thr = [0i32; 16];

    if let (Some(above_lut), Some((aseg, aoff))) = (above_thr_lut, above_seg) {
        let aoff = usize::try_from(aoff).expect("negative above segment offset");
        assert!(aoff + w <= aseg.len());

        let above_q_lut = &above_lut[0];
        let above_side_lut = &above_lut[1];

        for ((&seg, q_prev), side_prev) in aseg[aoff..aoff + w]
            .iter()
            .zip(prev_q_thr[..w].iter_mut())
            .zip(prev_side_thr[..w].iter_mut())
        {
            let seg_id = usize::from(seg);
            assert!(seg_id < 16);

            // Keep original semantics:
            //
            // above_q_thr[x4] = above_lut[0][seg_id] as u8;
            // prev_q_thr = above_q_thr[x4] as i32;
            *q_prev = (above_q_lut[seg_id] as u8) as i32;
            *side_prev = (above_side_lut[seg_id] as u8) as i32;
        }
    }

    if w == 0 || h == 0 {
        return;
    }

    let seg_off = usize::try_from(seg_off).expect("negative segment offset");
    let seg_stride = usize::try_from(seg_stride).expect("negative segment stride");

    // `seg_stride == 0` is valid: every output row reads the same segmap row.
    let last_row_start = seg_off + (h - 1) * seg_stride;
    assert!(last_row_start + w <= segmap.len());

    let q_lut = &thr_lut[0];
    let side_lut = &thr_lut[1];

    let mask_rows = &mask[starty4..starty4 + h];

    for (y4, ((q_row, side_row), mask_row)) in q_thr_dst
        .chunks_exact_mut(16)
        .zip(side_thr_dst.chunks_exact_mut(16))
        .zip(mask_rows.iter())
        .take(h)
        .enumerate()
    {
        let row_start = seg_off + y4 * seg_stride;
        let seg_row = &segmap[row_start..row_start + w];

        let q_row = &mut q_row[..w];
        let side_row = &mut side_row[..w];

        let prev_q_row = &mut prev_q_thr[..w];
        let prev_side_row = &mut prev_side_thr[..w];

        for (x4, ((((&seg, q_dst), side_dst), q_prev), side_prev)) in seg_row
            .iter()
            .zip(q_row.iter_mut())
            .zip(side_row.iter_mut())
            .zip(prev_q_row.iter_mut())
            .zip(prev_side_row.iter_mut())
            .enumerate()
        {
            let seg_id = usize::from(seg);
            assert!(seg_id < 16);

            let cur_q_thr = q_lut[seg_id] as i32;
            let cur_side_thr = side_lut[seg_id] as i32;

            let subpu = 3 * (((mask_row[4][mask_idx] >> (mask_shift + x4 as u32)) & 1) as i32);

            let eq = edge_thr(cur_q_thr, *q_prev) >> subpu;
            let es = edge_thr(cur_side_thr, *side_prev) >> subpu;

            *q_dst = eq as u8;
            *side_dst = es as u8;

            *q_prev = cur_q_thr;
            *side_prev = cur_side_thr;
        }
    }
}

/// Must run before the rows pass reads `filter_y[1]`. Operates on the mutable
/// mask, so it is driven from `filter_sbrow` (which holds `&mut lf.mask`).
pub(crate) fn deblock_crop_bottom_edge(
    mask: &mut [Av2Filter],
    mask_row: usize,
    sb256w: i32,
    bw: i32,
    bh: i32,
    sb128: i32,
    sby: i32,
) {
    let y64_start = sby << sb128;
    let y64_end = imin((sby + 1) << sb128, (bh + 15) >> 4);
    for y64 in y64_start..y64_end {
        if (y64 + 1) * 16 + 4 <= bh {
            continue;
        }
        let starty4 = ((y64 * 16) & 0x30) as usize;
        let h4 = imin(bh - y64 * 16, 16);
        let luma_crop_y4 = starty4 as i32 + h4 - 2;
        if luma_crop_y4 < 0 {
            continue;
        }
        for x256 in 0..sb256w as usize {
            if mask_row + x256 >= mask.len() {
                break;
            }
            let w = imin(64, bw - (x256 as i32) * 64);
            let yv = &mut mask[mask_row + x256].filter_y[1][luma_crop_y4 as usize];
            for i in 0..((w + 15) >> 4) as usize {
                let m = yv[3][i];
                yv[3][i] = 0;
                yv[2][i] |= m;
            }
        }
    }
}

fn init_lut_y(ctx: &DeblockCtx, dir: usize, qidx: i32) -> [[u32; 16]; 2] {
    let mut lut = [[0u32; 16]; 2];
    init_deblock_thr_lut_y(ctx.frame_hdr, ctx.hbd, dir, qidx, &mut lut);
    lut
}

fn init_lut_uv(ctx: &DeblockCtx, qidx: i32) -> [[[u32; 16]; 2]; 2] {
    let mut lut = [[[0u32; 16]; 2]; 2];
    init_deblock_thr_lut_uv(ctx.frame_hdr, ctx.hbd, qidx, &mut lut);
    lut
}

/// Port of `deblock_sbrow64_cols` (single-tile). `p_*` are whole planes; the
/// `*_off` are byte offsets to this 64-row band's first pixel.
#[allow(clippy::too_many_arguments)]
fn deblock64_cols<BD: BitDepth>(
    bd: BD,
    ctx: &DeblockCtx,
    p_y: &mut [BD::Pixel],
    y_off: usize,
    p_u: &mut [BD::Pixel],
    p_v: &mut [BD::Pixel],
    uv_off: usize,
    y64: i32,
) {
    let lflvl_row = ctx.mask_row;
    let starty4 = ((y64 * 16) & 0x30) as usize;
    let h4 = imin(ctx.bh - y64 * 16, 16);
    let uv_h4 = h4 >> ctx.ss_ver;
    let y64idx = ((y64 & 3) << 2) as usize;
    let seg_enabled = !ctx.cur_segmap.is_empty();
    let any_lossless = ctx.frame_hdr.any_lossless != 0;

    let seg_stride = if !ctx.cur_segmap.is_empty() {
        ctx.b4_stride
    } else {
        0
    };
    // segmap base for this 64-row band's top row.
    let seg_band = if !ctx.cur_segmap.is_empty() {
        (y64 as isize) * 16 * seg_stride
    } else {
        0
    };

    // luma columns
    if ctx.frame_hdr.deblock.level_y[0] != 0 {
        let mut l_qidx = -1i32;
        let mut lut = [[0u32; 16]; 2];
        let mut left_q_thr = [0u8; 16];
        let mut left_side_thr = [0u8; 16];
        let mut ll_mask = [0u16; 17];
        let mut q_thr = [0u8; 256];
        let mut side_thr = [0u8; 256];
        let n64 = (ctx.bw + 15) >> 4;
        for x64 in 0..n64 {
            let have_left = x64 > 0;
            let col = lflvl_row + (x64 >> 2) as usize;
            if col >= ctx.mask.len() {
                break;
            }
            let col_lflvl = &ctx.mask[col];
            let cur_qidx = col_lflvl.qidx[((x64 & 3) as usize) + y64idx] as i32;
            let q_changed = cur_qidx != l_qidx;
            if q_changed {
                lut = init_lut_y(ctx, 0, cur_qidx);
            }
            let bx4_base = ((x64 & 3) * 16) as usize;
            let w4 = imin(ctx.bw - x64 * 16, 16);
            if !seg_enabled
                && !mask_has_luma_col(
                    &col_lflvl.filter_y[0],
                    bx4_base,
                    (y64 & 3) as usize,
                    w4 as usize,
                    have_left,
                )
            {
                if q_changed {
                    fill_left_thr_from_lut(&mut left_q_thr, &mut left_side_thr, &lut, h4 as usize);
                }
                l_qidx = cur_qidx;
                continue;
            }
            if seg_enabled {
                setup_thr_cols(
                    &mut q_thr,
                    &mut side_thr,
                    ctx.cur_segmap,
                    seg_band + (x64 as isize) * 16,
                    seg_stride,
                    &col_lflvl.filter_y[0],
                    bx4_base,
                    &lut,
                    &mut left_q_thr,
                    &mut left_side_thr,
                    y64 & 3,
                    0,
                    w4,
                    h4,
                );
            } else if q_changed {
                setup_thr_cols_dq(
                    &mut q_thr,
                    &mut side_thr,
                    &col_lflvl.filter_y[0],
                    bx4_base,
                    &lut,
                    &mut left_q_thr,
                    &mut left_side_thr,
                    y64 & 3,
                    0,
                    w4,
                    h4,
                );
            } else {
                setup_thr_cols_simple(
                    &mut q_thr,
                    &mut side_thr,
                    &col_lflvl.filter_y[0],
                    bx4_base,
                    &lut,
                    y64 & 3,
                    0,
                    w4,
                    h4,
                );
            }
            l_qidx = cur_qidx;
            if any_lossless {
                lf_transpose_lossless_mask(
                    &mut ll_mask,
                    &col_lflvl.lossless_mask_y[starty4..],
                    (x64 & 3) as usize,
                    0,
                    0,
                );
            }
            // filter_plane_cols_y
            let cur_off = y_off + (x64 as usize) * 64;
            let ls = ctx.y_stride;
            for x in 0..w4 as usize {
                if !have_left && x == 0 {
                    continue;
                }
                let hmask = &col_lflvl.filter_y[0][bx4_base + x];
                // packed `y64idx` (which is `(y64 & 3) << 2`, whose low 2 bits are
                // always 0). For multi-y64 superblock rows this read must select
                // the correct sb64 sub-row.
                let sb64y = (y64 & 3) as usize;
                let m0 = hmask[0][sb64y];
                let m1 = hmask[1][sb64y];
                let m2 = hmask[2][sb64y];
                let m3 = hmask[3][sb64y];
                if (m0 | m1 | m2 | m3) == 0 {
                    continue;
                }
                let vmask = [m0, m1, m2, m3];
                let llm = if any_lossless {
                    [ll_mask[x], ll_mask[x + 1]]
                } else {
                    [0, 0]
                };
                // first column of an x64 that begins a new tile; for single-tile
                // frames it is always false. Passing `x == 0` here would wrongly
                // clamp max_width_neg at every superblock-column's left edge.
                deblock_h_sb64y_bd(
                    bd,
                    p_y,
                    cur_off + x * 4,
                    ls.unsigned_abs(),
                    &vmask,
                    &llm,
                    &q_thr[x * 16..],
                    &side_thr[x * 16..],
                    false,
                );
            }
        }
    }

    if ctx.frame_hdr.deblock.level_u == 0 && ctx.frame_hdr.deblock.level_v == 0 {
        return;
    }
    if ctx.layout == PixelLayout::I400 {
        return;
    }

    // chroma columns
    let uv_seg_stride = if !ctx.segmap_uv.is_empty() {
        ctx.uv_segmap_stride
    } else {
        0
    };
    let uv_seg_band = if !ctx.segmap_uv.is_empty() {
        (y64 as isize) * (16 >> ctx.ss_ver) as isize * uv_seg_stride
    } else {
        0
    };
    let mut prev_qidx = -1i32;
    let mut lut = [[[0u32; 16]; 2]; 2];
    let mut left_q_thr = [[0u8; 16]; 2];
    let mut left_side_thr = [[0u8; 16]; 2];
    let mut ll_mask = [0u16; 17];
    let mut q_thr = [[0u8; 256]; 2];
    let mut side_thr = [[0u8; 256]; 2];
    let n64 = (ctx.bw + 15) >> 4;
    let apply_u = ctx.frame_hdr.deblock.level_u != 0;
    let apply_v = ctx.frame_hdr.deblock.level_v != 0;
    for x64 in 0..n64 {
        let have_left = x64 > 0;
        let col = lflvl_row + (x64 >> 2) as usize;
        if col >= ctx.mask.len() {
            break;
        }
        let col_lflvl = &ctx.mask[col];
        let cur_qidx = col_lflvl.qidx[((x64 & 3) as usize) + y64idx] as i32;
        let q_changed = cur_qidx != prev_qidx;
        if q_changed {
            lut = init_lut_uv(ctx, cur_qidx);
        }
        let bx4_base = (((x64 & 3) * 16) >> ctx.ss_hor) as usize;
        let uv_w4 = imin(ctx.bw - x64 * 16, 16) >> ctx.ss_hor;
        if ctx.segmap_uv.is_empty()
            && !mask_has_chroma_col(
                &col_lflvl.filter_uv[0],
                bx4_base,
                y64,
                ctx.ss_ver,
                uv_w4 as usize,
                have_left,
            )
        {
            if q_changed {
                for pl in 0..2 {
                    fill_left_thr_from_lut(
                        &mut left_q_thr[pl],
                        &mut left_side_thr[pl],
                        &lut[pl],
                        uv_h4 as usize,
                    );
                }
            }
            prev_qidx = cur_qidx;
            continue;
        }
        for pl in 0..2 {
            if !ctx.segmap_uv.is_empty() {
                setup_thr_cols(
                    &mut q_thr[pl],
                    &mut side_thr[pl],
                    ctx.segmap_uv,
                    uv_seg_band + (x64 as isize) * (16 >> ctx.ss_hor) as isize,
                    uv_seg_stride,
                    &col_lflvl.filter_uv[0],
                    bx4_base,
                    &lut[pl],
                    &mut left_q_thr[pl],
                    &mut left_side_thr[pl],
                    y64 & 3,
                    ctx.ss_ver,
                    uv_w4,
                    uv_h4,
                );
            } else if q_changed {
                setup_thr_cols_dq(
                    &mut q_thr[pl],
                    &mut side_thr[pl],
                    &col_lflvl.filter_uv[0],
                    bx4_base,
                    &lut[pl],
                    &mut left_q_thr[pl],
                    &mut left_side_thr[pl],
                    y64 & 3,
                    ctx.ss_ver,
                    uv_w4,
                    uv_h4,
                );
            } else {
                setup_thr_cols_simple(
                    &mut q_thr[pl],
                    &mut side_thr[pl],
                    &col_lflvl.filter_uv[0],
                    bx4_base,
                    &lut[pl],
                    y64 & 3,
                    ctx.ss_ver,
                    uv_w4,
                    uv_h4,
                );
            }
        }
        prev_qidx = cur_qidx;
        if any_lossless {
            lf_transpose_lossless_mask(
                &mut ll_mask,
                &col_lflvl.lossless_mask_uv[(starty4 >> ctx.ss_ver)..],
                (x64 & 3) as usize,
                ctx.ss_hor,
                ctx.ss_ver,
            );
        }
        let cur_off = uv_off + (x64 as usize) * (64 >> ctx.ss_hor) as usize;
        let ls = ctx.uv_stride;
        let mask_idx = ((y64 & 3) >> ctx.ss_ver) as usize;
        let mask_shift: u32 = if (y64 & 3) & ctx.ss_ver != 0 { 8 } else { 0 };
        let bytes_mask: u32 = if ctx.ss_ver != 0 { 0xff } else { 0xffff };
        for x in 0..uv_w4 as usize {
            if !have_left && x == 0 {
                continue;
            }
            let hmask = &col_lflvl.filter_uv[0][bx4_base + x];
            let m0 = ((hmask[0][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
            let m1 = ((hmask[1][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
            let m2 = ((hmask[2][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
            if (m0 | m1 | m2) == 0 {
                continue;
            }
            let vmask = [m0, m1, m2];
            let llm = if any_lossless {
                [ll_mask[x], ll_mask[x + 1]]
            } else {
                [0, 0]
            };
            // Single-tile: tile_edge is always false (see luma above).
            if apply_u {
                deblock_h_sb64uv_bd(
                    bd,
                    p_u,
                    cur_off + x * 4,
                    ls.unsigned_abs(),
                    &vmask,
                    &llm,
                    &q_thr[0][x * 16..],
                    &side_thr[0][x * 16..],
                    false,
                );
            }
            if apply_v {
                deblock_h_sb64uv_bd(
                    bd,
                    p_v,
                    cur_off + x * 4,
                    ls.unsigned_abs(),
                    &vmask,
                    &llm,
                    &q_thr[1][x * 16..],
                    &side_thr[1][x * 16..],
                    false,
                );
            }
        }
    }
}

/// Port of `deblock_sbrow64_rows` (single-tile).
#[allow(clippy::too_many_arguments)]
fn deblock64_rows<BD: BitDepth>(
    bd: BD,
    ctx: &DeblockCtx,
    p_y: &mut [BD::Pixel],
    y_off: usize,
    p_u: &mut [BD::Pixel],
    p_v: &mut [BD::Pixel],
    uv_off: usize,
    y64: i32,
) {
    let lflvl_row = ctx.mask_row;
    let have_top = y64 > 0;
    let starty4 = ((y64 * 16) & 0x30) as usize;
    let h4 = imin(ctx.bh - y64 * 16, 16);
    let uv_h4 = h4 >> ctx.ss_ver;
    let y64idx = ((y64 & 3) << 2) as usize;
    let a_y64idx = (((y64 + 3) & 3) << 2) as usize;
    let seg_enabled = !ctx.cur_segmap.is_empty();
    let any_lossless = ctx.frame_hdr.any_lossless != 0;

    // above SB256 row for cross-SB-row context (single tile: prev mask row).
    let a_row: Option<usize> = if have_top {
        let above = if starty4 == 0 { ctx.sb256w as usize } else { 0 };
        ctx.mask_row.checked_sub(above)
    } else {
        None
    };

    let seg_stride = if !ctx.cur_segmap.is_empty() {
        ctx.b4_stride
    } else {
        0
    };
    let seg_band = if !ctx.cur_segmap.is_empty() {
        (y64 as isize) * 16 * seg_stride
    } else {
        0
    };

    if ctx.frame_hdr.deblock.level_y[1] != 0 {
        let mut l_qidx = -1i32;
        let mut al_qidx = -1i32;
        let mut lut = [[0u32; 16]; 2];
        let mut a_lut = [[0u32; 16]; 2];
        let mut ll_mask = [0u16; 17];
        let mut q_thr = [0u8; 256];
        let mut side_thr = [0u8; 256];
        let n64 = (ctx.bw + 15) >> 4;
        for x64 in 0..n64 {
            let col = lflvl_row + (x64 >> 2) as usize;
            if col >= ctx.mask.len() {
                break;
            }
            let col_lflvl = &ctx.mask[col];
            let cur_qidx = col_lflvl.qidx[((x64 & 3) as usize) + y64idx] as i32;
            let w4 = imin(ctx.bw - x64 * 16, 16);
            if !mask_has_luma_row(
                &col_lflvl.filter_y[1],
                starty4,
                (x64 & 3) as usize,
                h4 as usize,
                have_top,
            ) {
                // Do not advance the LUT cache on a skipped block without also
                // rebuilding the LUT; otherwise the next active block with the
                // same qidx would reuse an uninitialized/stale threshold table.
                l_qidx = -1;
                continue;
            }
            if any_lossless {
                for y in 0..h4 as usize {
                    ll_mask[y + 1] = col_lflvl.lossless_mask_y[starty4 + y][(x64 & 3) as usize];
                }
            }
            let q_changed = cur_qidx != l_qidx;
            if q_changed {
                lut = init_lut_y(ctx, 1, cur_qidx);
            }
            let mut above_seg: Option<(&[u8], isize)> = None;
            let mut above_lut: Option<&[[u32; 16]; 2]> = None;
            let mut above_qdiff = false;
            if let Some(ar) = a_row {
                let acol = ar + (x64 >> 2) as usize;
                if acol < ctx.mask.len() {
                    let a_lflvl = &ctx.mask[acol];
                    if any_lossless {
                        ll_mask[0] =
                            a_lflvl.lossless_mask_y[(starty4 + 63) & 63][(x64 & 3) as usize];
                    }
                    let a_qidx = a_lflvl.qidx[((x64 & 3) as usize) + a_y64idx] as i32;
                    above_qdiff = a_qidx != cur_qidx;
                    if a_qidx != al_qidx {
                        a_lut = init_lut_y(ctx, 1, a_qidx);
                        al_qidx = a_qidx;
                    }
                    above_lut = Some(&a_lut);
                    // above segmap row is the row directly above seg_band.
                    if seg_enabled {
                        above_seg =
                            Some((ctx.cur_segmap, seg_band + (x64 as isize) * 16 - seg_stride));
                    }
                }
            }
            if seg_enabled {
                setup_thr_rows(
                    &mut q_thr,
                    &mut side_thr,
                    ctx.cur_segmap,
                    seg_band + (x64 as isize) * 16,
                    seg_stride,
                    &col_lflvl.filter_y[1],
                    starty4,
                    &lut,
                    above_lut,
                    above_seg,
                    x64 & 3,
                    0,
                    w4,
                    h4,
                );
            } else if above_qdiff {
                setup_thr_rows_dq(
                    &mut q_thr,
                    &mut side_thr,
                    &col_lflvl.filter_y[1],
                    starty4,
                    &lut,
                    above_lut,
                    None,
                    x64 & 3,
                    0,
                    w4,
                    h4,
                );
            } else {
                setup_thr_rows_simple(
                    &mut q_thr,
                    &mut side_thr,
                    &col_lflvl.filter_y[1],
                    starty4,
                    &lut,
                    x64 & 3,
                    0,
                    w4,
                    h4,
                );
            }
            l_qidx = cur_qidx;
            let cur_off = y_off + (x64 as usize) * 64;
            let ls = ctx.y_stride;
            for y in 0..h4 as usize {
                if !have_top && y == 0 {
                    continue;
                }
                let row = &col_lflvl.filter_y[1][starty4 + y];
                let sidx = (x64 & 3) as usize;
                let m0 = row[0][sidx];
                let m1 = row[1][sidx];
                let m2 = row[2][sidx];
                let m3 = row[3][sidx];
                if (m0 | m1 | m2 | m3) == 0 {
                    continue;
                }
                let vmask = [m0, m1, m2, m3];
                let llm = if any_lossless {
                    [ll_mask[y], ll_mask[y + 1]]
                } else {
                    [0, 0]
                };
                deblock_v_sb64y_bd(
                    bd,
                    p_y,
                    (cur_off as isize + y as isize * 4 * ls) as usize,
                    ls.unsigned_abs(),
                    &vmask,
                    &llm,
                    &q_thr[y * 16..],
                    &side_thr[y * 16..],
                    y == 0,
                );
            }
        }
    }

    if ctx.frame_hdr.deblock.level_u == 0 && ctx.frame_hdr.deblock.level_v == 0 {
        return;
    }
    if ctx.layout == PixelLayout::I400 {
        return;
    }

    let uv_seg_stride = if !ctx.segmap_uv.is_empty() {
        ctx.uv_segmap_stride
    } else {
        0
    };
    let uv_seg_band = if !ctx.segmap_uv.is_empty() {
        (y64 as isize) * (16 >> ctx.ss_ver) as isize * uv_seg_stride
    } else {
        0
    };
    let mut l_qidx = -1i32;
    let mut al_qidx = -1i32;
    let mut lut = [[[0u32; 16]; 2]; 2];
    let mut a_lut = [[[0u32; 16]; 2]; 2];
    let mut ll_mask = [0u16; 17];
    let mut q_thr = [[0u8; 256]; 2];
    let mut side_thr = [[0u8; 256]; 2];
    let n64 = (ctx.bw + 15) >> 4;
    let apply_u = ctx.frame_hdr.deblock.level_u != 0;
    let apply_v = ctx.frame_hdr.deblock.level_v != 0;
    for x64 in 0..n64 {
        let col = lflvl_row + (x64 >> 2) as usize;
        if col >= ctx.mask.len() {
            break;
        }
        let col_lflvl = &ctx.mask[col];
        let cur_qidx = col_lflvl.qidx[((x64 & 3) as usize) + y64idx] as i32;
        let uv_w4 = imin(ctx.bw - x64 * 16, 16) >> ctx.ss_hor;
        if !mask_has_chroma_row(
            &col_lflvl.filter_uv[1],
            starty4 >> ctx.ss_ver,
            x64,
            ctx.ss_hor,
            uv_h4 as usize,
            have_top,
        ) {
            // Same cache rule as luma rows: skipped blocks do not materialize
            // `lut`, so force the next active block to refresh it.
            l_qidx = -1;
            continue;
        }
        if any_lossless {
            for y in 0..uv_h4 as usize {
                ll_mask[y + 1] =
                    col_lflvl.lossless_mask_uv[(starty4 >> ctx.ss_ver) + y][(x64 & 3) as usize];
            }
        }
        let q_changed = cur_qidx != l_qidx;
        if q_changed {
            lut = init_lut_uv(ctx, cur_qidx);
        }
        let mut above_seg: Option<(&[u8], isize)> = None;
        let mut above_present = false;
        let mut above_qdiff = false;
        if let Some(ar) = a_row {
            let acol = ar + (x64 >> 2) as usize;
            if acol < ctx.mask.len() {
                let a_lflvl = &ctx.mask[acol];
                if any_lossless {
                    ll_mask[0] = a_lflvl.lossless_mask_uv[((starty4 + 63) & 63) >> ctx.ss_ver]
                        [(x64 & 3) as usize];
                }
                let a_qidx = a_lflvl.qidx[((x64 & 3) as usize) + a_y64idx] as i32;
                above_qdiff = a_qidx != cur_qidx;
                if a_qidx != al_qidx {
                    a_lut = init_lut_uv(ctx, a_qidx);
                    al_qidx = a_qidx;
                }
                above_present = true;
                if !ctx.segmap_uv.is_empty() {
                    above_seg = Some((
                        ctx.segmap_uv,
                        uv_seg_band + (x64 as isize) * (16 >> ctx.ss_hor) as isize - uv_seg_stride,
                    ));
                }
            }
        }
        for pl in 0..2 {
            let above_lut = if above_present {
                Some(&a_lut[pl])
            } else {
                None
            };
            if !ctx.segmap_uv.is_empty() {
                setup_thr_rows(
                    &mut q_thr[pl],
                    &mut side_thr[pl],
                    ctx.segmap_uv,
                    uv_seg_band + (x64 as isize) * (16 >> ctx.ss_hor) as isize,
                    uv_seg_stride,
                    &col_lflvl.filter_uv[1],
                    starty4 >> ctx.ss_ver,
                    &lut[pl],
                    above_lut,
                    above_seg,
                    x64 & 3,
                    ctx.ss_hor,
                    uv_w4,
                    uv_h4,
                );
            } else if above_qdiff {
                setup_thr_rows_dq(
                    &mut q_thr[pl],
                    &mut side_thr[pl],
                    &col_lflvl.filter_uv[1],
                    starty4 >> ctx.ss_ver,
                    &lut[pl],
                    above_lut,
                    None,
                    x64 & 3,
                    ctx.ss_hor,
                    uv_w4,
                    uv_h4,
                );
            } else {
                setup_thr_rows_simple(
                    &mut q_thr[pl],
                    &mut side_thr[pl],
                    &col_lflvl.filter_uv[1],
                    starty4 >> ctx.ss_ver,
                    &lut[pl],
                    x64 & 3,
                    ctx.ss_hor,
                    uv_w4,
                    uv_h4,
                );
            }
        }
        l_qidx = cur_qidx;
        let cur_off = uv_off + (x64 as usize) * (64 >> ctx.ss_hor) as usize;
        let ls = ctx.uv_stride;
        let mask_idx = ((x64 & 3) >> ctx.ss_hor) as usize;
        let mask_shift: u32 = if (x64 & 3) & ctx.ss_hor != 0 { 8 } else { 0 };
        let bytes_mask: u32 = if ctx.ss_hor != 0 { 0xff } else { 0xffff };
        for y in 0..uv_h4 as usize {
            if !have_top && y == 0 {
                continue;
            }
            let row = &col_lflvl.filter_uv[1][(starty4 >> ctx.ss_ver) + y];
            let m0 = ((row[0][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
            let m1 = ((row[1][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
            let m2 = ((row[2][mask_idx] as u32 >> mask_shift) & bytes_mask) as u16;
            if (m0 | m1 | m2) == 0 {
                continue;
            }
            let vmask = [m0, m1, m2];
            let llm = if any_lossless {
                [ll_mask[y], ll_mask[y + 1]]
            } else {
                [0, 0]
            };
            if apply_u {
                deblock_v_sb64uv_bd(
                    bd,
                    p_u,
                    (cur_off as isize + y as isize * 4 * ls) as usize,
                    ls.unsigned_abs(),
                    &vmask,
                    &llm,
                    &q_thr[0][y * 16..],
                    &side_thr[0][y * 16..],
                    y == 0,
                );
            }
            if apply_v {
                deblock_v_sb64uv_bd(
                    bd,
                    p_v,
                    (cur_off as isize + y as isize * 4 * ls) as usize,
                    ls.unsigned_abs(),
                    &vmask,
                    &llm,
                    &q_thr[1][y * 16..],
                    &side_thr[1][y * 16..],
                    y == 0,
                );
            }
        }
    }
}

/// Deblock exactly one 64px-high band. This mirrors dav2d's
/// `filter_slice_deblock_cols(f, by64)` entry point and is used by the
/// multithreaded scheduler to expose one filter task per sb64 row even for
/// 128x128 superblock frames.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_sb64_cols<BD: BitDepth>(
    bd: BD,
    ctx: &DeblockCtx,
    p_y: &mut [BD::Pixel],
    y_off: usize,
    p_u: &mut [BD::Pixel],
    p_v: &mut [BD::Pixel],
    uv_off: usize,
    by64: i32,
) {
    deblock64_cols(bd, ctx, p_y, y_off, p_u, p_v, uv_off, by64);
}

/// Deblock exactly one 64px-high band. This mirrors dav2d's
/// `filter_slice_deblock_rows(f, by64)` entry point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_sb64_rows<BD: BitDepth>(
    bd: BD,
    ctx: &DeblockCtx,
    p_y: &mut [BD::Pixel],
    y_off: usize,
    p_u: &mut [BD::Pixel],
    p_v: &mut [BD::Pixel],
    uv_off: usize,
    by64: i32,
) {
    deblock64_rows(bd, ctx, p_y, y_off, p_u, p_v, uv_off, by64);
}

pub(crate) fn copy_db_8bpc(
    lr_db: &mut [Vec<u8>; 3],
    src: &[&[u8]; 3],
    strides: &[isize; 2],
    bw: usize,
    bh: usize,
    sby: i32,
    sb128: bool,
    ss_hor: bool,
    ss_ver: bool,
    lr_backup: bool,
) {
    // up by `offset` rows (8 luma rows for sby > 0), and `row`/`row_h` are the
    // offset-adjusted stripe extent. The previous code used the raw sbrow row and
    // an un-offset plane base, so it read the wrong rows for sby > 0.
    let h = (bh * 4) as i32;
    let w = bw * 4;
    let offset = 8 * (sby != 0) as i32;
    let y_stripe = (sby << (6 + sb128 as i32)) - offset;
    let row_h = imin((sby + 1) << (6 + sb128 as i32), h - 1);
    if y_stripe < row_h {
        let ys_off = (y_stripe as isize * strides[0]) as usize;
        backup_db(
            &mut lr_db[0],
            &src[0][ys_off..],
            strides[0].unsigned_abs(),
            0,
            sb128,
            y_stripe,
            row_h,
            w,
            lr_backup,
            1,
        );
    }

    if strides[1] != 0 {
        let cw = w >> (ss_hor as usize);
        let ch = (bh * 4 >> ss_ver as i32) as i32;
        let ss_ver_i = ss_ver as i32;
        let offset_uv = offset >> ss_ver_i;
        let cy_stripe = (sby << ((6 - ss_ver_i) + sb128 as i32)) - offset_uv;
        let crow_h = imin((sby + 1) << ((6 - ss_ver_i) + sb128 as i32), ch - 1);
        if cy_stripe < crow_h {
            let cys_off = (cy_stripe as isize * strides[1]) as usize;
            backup_db(
                &mut lr_db[1],
                &src[1][cys_off..],
                strides[1].unsigned_abs(),
                ss_ver_i,
                sb128,
                cy_stripe,
                crow_h,
                cw,
                lr_backup,
                1,
            );
            backup_db(
                &mut lr_db[2],
                &src[2][cys_off..],
                strides[1].unsigned_abs(),
                ss_ver_i,
                sb128,
                cy_stripe,
                crow_h,
                cw,
                lr_backup,
                1,
            );
        }
    }
}

pub(crate) fn copy_db_hbd(
    lr_db: &mut [Vec<u16>; 3],
    src: &[&[u16]; 3],
    strides: &[isize; 2],
    bw: usize,
    bh: usize,
    sby: i32,
    sb128: bool,
    ss_hor: bool,
    ss_ver: bool,
    lr_backup: bool,
) {
    let h = (bh * 4) as i32;
    let w = bw * 4;
    let offset = 8 * (sby != 0) as i32;
    let y_stripe = (sby << (6 + sb128 as i32)) - offset;
    let row_h = imin((sby + 1) << (6 + sb128 as i32), h - 1);
    if y_stripe < row_h {
        let ys_off = (y_stripe as isize * strides[0]) as usize;
        backup_db_hbd(
            &mut lr_db[0],
            &src[0][ys_off..],
            strides[0].unsigned_abs(),
            0,
            sb128,
            y_stripe,
            row_h,
            w,
            lr_backup,
            1,
        );
    }

    if strides[1] != 0 {
        let cw = w >> (ss_hor as usize);
        let ch = (bh * 4 >> ss_ver as i32) as i32;
        let ss_ver_i = ss_ver as i32;
        let offset_uv = offset >> ss_ver_i;
        let cy_stripe = (sby << ((6 - ss_ver_i) + sb128 as i32)) - offset_uv;
        let crow_h = imin((sby + 1) << ((6 - ss_ver_i) + sb128 as i32), ch - 1);
        if cy_stripe < crow_h {
            let cys_off = (cy_stripe as isize * strides[1]) as usize;
            backup_db_hbd(
                &mut lr_db[1],
                &src[1][cys_off..],
                strides[1].unsigned_abs(),
                ss_ver_i,
                sb128,
                cy_stripe,
                crow_h,
                cw,
                lr_backup,
                1,
            );
            backup_db_hbd(
                &mut lr_db[2],
                &src[2][cys_off..],
                strides[1].unsigned_abs(),
                ss_ver_i,
                sb128,
                cy_stripe,
                crow_h,
                cw,
                lr_backup,
                1,
            );
        }
    }
}
