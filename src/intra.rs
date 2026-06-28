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
use crate::intops::{apply_sign, ulog2};
use crate::pixel::BitDepth;

#[inline]
fn bawp_blk_size_from_samples(n: usize) -> usize {
    // AVM blk_size_log2_bawp/log_to_blk_size mapping, with the input already
    // clamped to BAWP_MAX_REF_NUMB (16). 1..=2 disable that side, 3..=6 use 4,
    // 7..=12 use 8, and 13..=16 use 16 samples after padding.
    match n {
        0..=2 => 0,
        3..=6 => 4,
        7..=12 => 8,
        _ => 16,
    }
}

#[inline]
fn derive_number_ref_samples_bawp(
    above_valid: bool,
    left_valid: bool,
    width: usize,
    height: usize,
) -> (usize, usize) {
    let above_available = above_valid && width != 0;
    let left_available = left_valid && height != 0;

    if above_available && left_available {
        if width == 16 && height == 16 {
            (16, 16)
        } else if width > 4 && height > 4 {
            (8, 8)
        } else if width < 16 && height < 16 {
            (4, 4)
        } else if width == 16 {
            (16, 0)
        } else {
            (0, 16)
        }
    } else if above_available {
        (width, 0)
    } else if left_available {
        (0, height)
    } else {
        (0, 0)
    }
}

#[inline]
fn repeat_pad_i32(buf: &mut [i32; 16], valid: usize, len: usize) {
    if valid == 0 || valid >= len {
        return;
    }

    let (seed, tail) = buf[..len].split_at_mut(valid);
    tail.iter_mut()
        .zip(seed.iter().copied().cycle())
        .for_each(|(dst, src)| *dst = src);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn intrabc_morph_pred_luma<BD: BitDepth>(
    bd: BD,
    plane: &mut [BD::Pixel],
    stride: usize,
    bw4: i32,
    bh4: i32,
    bx: i32,
    by: i32,
    mvx: i32,
    mvy: i32,
    right: i32,
    bottom: i32,
) {
    let w = (bw4 * 4) as usize;
    let h = (bh4 * 4) as usize;
    if w == 0 || h == 0 {
        return;
    }

    let dpx = bx * 4;
    let dpy = by * 4;
    if dpx < 0 || dpy < 0 || dpx >= right || dpy >= bottom {
        return;
    }

    // IntraBC's predictor source uses the same integer component as
    // `intrabc_pred`; the fractional part affects the block copy/filter, but
    // AVM's morph/BAWP template fit samples the integer-position reference
    // template.
    let sx = dpx + (mvx >> 3);
    let sy = dpy + (mvy >> 3);

    let ref_w = if dpx + w as i32 >= right {
        (right - dpx).max(0) as usize
    } else {
        w
    };
    let ref_h = if dpy + h as i32 >= bottom {
        (bottom - dpy).max(0) as usize
    } else {
        h
    };
    if ref_w == 0 || ref_h == 0 {
        return;
    }

    let bw = ref_w.min(16);
    let bh = ref_h.min(16);
    let width = bawp_blk_size_from_samples(bw);
    let height = bawp_blk_size_from_samples(bh);

    let above_valid = dpy > 0 && sy > 0 && sx >= 0 && sx + bw as i32 <= right;
    let left_valid = dpx > 0 && sx > 0 && sy >= 0 && sy + bh as i32 <= bottom;
    let (numb_up, numb_left) =
        derive_number_ref_samples_bawp(above_valid, left_valid, width, height);

    let mut count = 0usize;
    let mut sum_x = 0i32;
    let mut sum_y = 0i32;
    let mut sum_xy = 0i32;
    let mut sum_xx = 0i32;

    let mut ref_pad = [0i32; 16];
    let mut recon_pad = [0i32; 16];

    if numb_up != 0 {
        let step = width / numb_up;
        let start = if step == 1 { 0 } else { step >> 1 };
        let ref_top_off = (sy as usize - 1) * stride + sx as usize;
        let recon_top_off = (dpy as usize - 1) * stride + dpx as usize;

        ref_pad[..bw]
            .iter_mut()
            .zip(recon_pad[..bw].iter_mut())
            .zip(
                plane[ref_top_off..ref_top_off + bw]
                    .iter()
                    .zip(plane[recon_top_off..recon_top_off + bw].iter()),
            )
            .for_each(|((ref_dst, recon_dst), (&ref_px, &recon_px))| {
                *ref_dst = ref_px.into();
                *recon_dst = recon_px.into();
            });

        repeat_pad_i32(&mut ref_pad, bw, width);
        repeat_pad_i32(&mut recon_pad, bw, width);

        ref_pad[start..width]
            .iter()
            .step_by(step)
            .zip(recon_pad[start..width].iter().step_by(step))
            .for_each(|(&x, &y)| {
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            });
        count += numb_up;
    }

    if numb_left != 0 {
        let step = height / numb_left;
        let start = if step == 1 { 0 } else { step >> 1 };
        let ref_left_off = sy as usize * stride + sx as usize - 1;
        let recon_left_off = dpy as usize * stride + dpx as usize - 1;

        ref_pad[..bh]
            .iter_mut()
            .zip(recon_pad[..bh].iter_mut())
            .zip(
                plane[ref_left_off..]
                    .iter()
                    .step_by(stride)
                    .zip(plane[recon_left_off..].iter().step_by(stride))
                    .take(bh),
            )
            .for_each(|((ref_dst, recon_dst), (&ref_px, &recon_px))| {
                *ref_dst = ref_px.into();
                *recon_dst = recon_px.into();
            });

        repeat_pad_i32(&mut ref_pad, bh, height);
        repeat_pad_i32(&mut recon_pad, bh, height);

        ref_pad[start..height]
            .iter()
            .step_by(step)
            .zip(recon_pad[start..height].iter().step_by(step))
            .for_each(|(&x, &y)| {
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            });
        count += numb_left;
    }

    let (alpha, beta) = if count != 0 {
        debug_assert!(count.is_power_of_two());
        let count_l2 = ulog2(count as u32);
        let num = sum_xy - (((sum_x as i64) * (sum_y as i64)) >> count_l2) as i32;
        let den = sum_xx - (((sum_x as i64) * (sum_x as i64)) >> count_l2) as i32;
        let alpha = crate::recon::derive_alpha(num, den, 256);
        let diff = (sum_y << 8) - sum_x * alpha;
        (alpha, apply_sign(diff.abs() >> count_l2, diff))
    } else {
        (256, -128)
    };

    let dst_off = dpy as usize * stride + dpx as usize;
    if dst_off < plane.len() {
        crate::mc::morph(bd, &mut plane[dst_off..], stride, alpha, beta, w, h);
    }
}
