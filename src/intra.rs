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
use crate::intops::{apply_sign, imin, ulog2};
use crate::levels::{BlockSize, TxPartition};
use crate::pixel::BitDepth;
use crate::tables::BLOCK_DIMENSIONS;

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

#[inline]
fn plane_row_range(
    plane_len: usize,
    stride: usize,
    x: i32,
    y: i32,
    n: usize,
) -> Option<core::ops::Range<usize>> {
    if stride == 0 || x < 0 || y < 0 {
        return None;
    }

    let x = x as usize;
    if x.checked_add(n)? > stride {
        return None;
    }

    let start = (y as usize).checked_mul(stride)?.checked_add(x)?;
    let end = start.checked_add(n)?;
    if end <= plane_len {
        Some(start..end)
    } else {
        None
    }
}

#[inline]
fn copy_strided_samples_i32<P: Copy + Into<i32>>(
    plane: &[P],
    stride: usize,
    x: i32,
    y: i32,
    n: usize,
    dst: &mut [i32],
) -> bool {
    if n == 0 {
        return true;
    }
    if stride == 0 || x < 0 || y < 0 || dst.len() < n {
        return false;
    }

    let x = x as usize;
    if x >= stride {
        return false;
    }

    let Some(mut off) = (y as usize)
        .checked_mul(stride)
        .and_then(|row| row.checked_add(x))
    else {
        return false;
    };

    for (i, dst_px) in dst.iter_mut().take(n).enumerate() {
        let Some(&px) = plane.get(off) else {
            return false;
        };
        *dst_px = px.into();
        if i + 1 != n {
            off = match off.checked_add(stride) {
                Some(next) => next,
                None => return false,
            };
        }
    }

    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn intrabc_morph_pred_luma<BD: BitDepth>(
    exec: &crate::exec_context::ExecContext,
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
    if bw4 <= 0 || bh4 <= 0 || stride == 0 {
        return;
    }

    let stride_i32 = stride.min(i32::MAX as usize) as i32;
    let plane_h = plane.len() / stride;
    let plane_h_i32 = plane_h.min(i32::MAX as usize) as i32;
    let right = right.clamp(0, stride_i32);
    let bottom = bottom.clamp(0, plane_h_i32);

    let Some(w_i32) = bw4.checked_mul(4) else {
        return;
    };
    let Some(h_i32) = bh4.checked_mul(4) else {
        return;
    };
    let w = w_i32 as usize;
    let h = h_i32 as usize;

    let Some(dpx) = bx.checked_mul(4) else {
        return;
    };
    let Some(dpy) = by.checked_mul(4) else {
        return;
    };
    if dpx < 0 || dpy < 0 || dpx >= right || dpy >= bottom {
        return;
    }

    // IntraBC's predictor source uses the same integer component as
    // `intrabc_pred`; the fractional part affects the block copy/filter, but
    // AVM's morph/BAWP template fit samples the integer-position reference
    // template.
    let Some(sx) = dpx.checked_add(mvx >> 3) else {
        return;
    };
    let Some(sy) = dpy.checked_add(mvy >> 3) else {
        return;
    };

    let Some(dst_right) = dpx.checked_add(w_i32) else {
        return;
    };
    let Some(dst_bottom) = dpy.checked_add(h_i32) else {
        return;
    };

    let ref_w = if dst_right >= right {
        (right - dpx).max(0) as usize
    } else {
        w
    };
    let ref_h = if dst_bottom >= bottom {
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

    let src_right = sx.checked_add(bw as i32);
    let src_bottom = sy.checked_add(bh as i32);
    let above_valid =
        dpy > 0 && sy > 0 && sy <= bottom && sx >= 0 && matches!(src_right, Some(v) if v <= right);
    let left_valid =
        dpx > 0 && sx > 0 && sx <= right && sy >= 0 && matches!(src_bottom, Some(v) if v <= bottom);
    let (numb_up, numb_left) =
        derive_number_ref_samples_bawp(above_valid, left_valid, width, height);

    let mut count = 0usize;
    let mut sum_x = 0i32;
    let mut sum_y = 0i32;
    let mut sum_xy = 0i32;
    let mut sum_xx = 0i32;

    let mut ref_pad = [0i32; 16];
    let mut recon_pad = [0i32; 16];

    if let Some(step) = width.checked_div(numb_up) {
        let start = if step == 1 { 0 } else { step >> 1 };
        let plane_ro: &[BD::Pixel] = &*plane;
        let ref_top = plane_row_range(plane_ro.len(), stride, sx, sy - 1, bw)
            .and_then(|range| plane_ro.get(range));
        let recon_top = plane_row_range(plane_ro.len(), stride, dpx, dpy - 1, bw)
            .and_then(|range| plane_ro.get(range));

        if let (Some(ref_top), Some(recon_top)) = (ref_top, recon_top) {
            ref_pad[..bw]
                .iter_mut()
                .zip(recon_pad[..bw].iter_mut())
                .zip(ref_top.iter().zip(recon_top.iter()))
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
    }

    if let Some(step) = height.checked_div(numb_left) {
        let start = if step == 1 { 0 } else { step >> 1 };
        let plane_ro: &[BD::Pixel] = &*plane;
        let copied_ref = copy_strided_samples_i32(plane_ro, stride, sx - 1, sy, bh, &mut ref_pad);
        let copied_recon =
            copy_strided_samples_i32(plane_ro, stride, dpx - 1, dpy, bh, &mut recon_pad);

        if copied_ref && copied_recon {
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
        crate::mc::morph(
            exec,
            bd,
            &mut plane[dst_off..],
            stride,
            alpha,
            beta,
            ref_w,
            ref_h,
        );
    }
}

#[inline(always)]
fn shrunken_unit(v: i32, ss: i32) -> i32 {
    (v >> ss).max(1)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_top_right_units(
    is_luma: bool,
    tx_partition: u8,
    is_first_tx: bool,
    is_coded: &[[u64; 64]; 2],
    coded_plane: usize,
    sb_step: i32,
    bsize: BlockSize,
    base_x: i32,
    base_y: i32,
    col_off: i32,
    row_off: i32,
    txw4: i32,
    ss_x: i32,
    ss_y: i32,
    top_available: bool,
    right_available: bool,
    px_to_right_edge4: i32,
) -> i32 {
    if !top_available || !right_available || px_to_right_edge4 <= 0 {
        return 0;
    }
    if !is_luma && txw4 > 8 {
        return 0;
    }
    if is_luma
        && (tx_partition == TxPartition::H5 as u8 || tx_partition == TxPartition::V5 as u8)
        && !is_first_tx
    {
        return 0;
    }

    let bdim = BLOCK_DIMENSIONS[bsize as usize];
    let plane_bw_unit = shrunken_unit(bdim[0] as i32, ss_x);
    let top_right_count_unit = txw4;
    let px_common = imin(top_right_count_unit, px_to_right_edge4);
    if px_common <= 0 {
        return 0;
    }

    if row_off > 0 {
        let plane_bw_unit_64 = shrunken_unit(16, ss_x);
        if bdim[0] as i32 > 16 {
            let tr_col = col_off + top_right_count_unit;
            let plane_bh_unit_64 = shrunken_unit(16, ss_y);
            if tr_col != plane_bw_unit
                && tr_col % plane_bw_unit_64 == 0
                && row_off % plane_bh_unit_64 == 0
            {
                let plane_bw_unit_128 = shrunken_unit(32, ss_x);
                let plane_bh_unit_128 = shrunken_unit(32, ss_y);
                return if (row_off % plane_bh_unit_128) != 0 && (tr_col % plane_bw_unit_128) == 0 {
                    0
                } else {
                    px_common
                };
            }
            let col_off_64 = col_off % plane_bw_unit_64;
            return if col_off_64 + top_right_count_unit < plane_bw_unit_64 {
                px_common
            } else {
                0
            };
        }
        return if col_off + top_right_count_unit < plane_bw_unit {
            px_common
        } else {
            0
        };
    }

    if col_off + top_right_count_unit < plane_bw_unit {
        return px_common;
    }

    let sb_w = shrunken_unit(sb_step, ss_x);
    let sb_h = shrunken_unit(sb_step, ss_y);
    // Availability uses SB-local coordinates (AVM's `& (sb_mi_size - 1)`), but the
    // `is_coded` window is region-local (indexed by `bx & 63` / `by & 63`, reset
    // per SB-row). Compute both: SB-local to decide whether the top-right unit is
    // still inside the current SB, region-local to actually probe the bitmap.
    let tr_mask_row_sb = (base_y & (sb_h - 1)) - 1;
    let tr_mask_col_sb = (base_x & (sb_w - 1)) + plane_bw_unit;
    if tr_mask_row_sb < 0 {
        return px_common;
    }
    if tr_mask_col_sb >= sb_w {
        return 0;
    }
    let row = ((base_y & 63) - 1) as usize;
    let col0 = (base_x & 63) + plane_bw_unit;
    if (base_y & 63) - 1 < 0 || col0 >= 64 {
        return 0;
    }
    if (is_coded[coded_plane][row] & (1u64 << (col0 as u32))) == 0 {
        return 0;
    }

    let mut coded = 0i32;
    while coded < top_right_count_unit {
        let c_sb = tr_mask_col_sb + coded;
        let c_rg = col0 + coded;
        if c_sb >= sb_w || c_rg >= 64 || (is_coded[coded_plane][row] & (1u64 << (c_rg as u32))) == 0
        {
            break;
        }
        coded += 1;
    }
    imin(coded, px_common)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_bottom_left_units(
    is_luma: bool,
    tx_partition: u8,
    is_first_tx: bool,
    is_coded: &[[u64; 64]; 2],
    coded_plane: usize,
    sb_step: i32,
    bsize: BlockSize,
    base_x: i32,
    base_y: i32,
    col_off: i32,
    row_off: i32,
    txh4: i32,
    ss_x: i32,
    ss_y: i32,
    bottom_available: bool,
    left_available: bool,
    px_to_bottom_edge4: i32,
) -> i32 {
    if !bottom_available || !left_available || px_to_bottom_edge4 <= 0 {
        return 0;
    }
    if !is_luma && txh4 > 8 {
        return 0;
    }
    if is_luma
        && (tx_partition == TxPartition::H5 as u8 || tx_partition == TxPartition::V5 as u8)
        && !is_first_tx
    {
        return 0;
    }

    let bdim = BLOCK_DIMENSIONS[bsize as usize];
    let plane_bh_unit = shrunken_unit(bdim[1] as i32, ss_y);
    let bottom_left_count_unit = txh4;
    let px_common = imin(bottom_left_count_unit, px_to_bottom_edge4);
    if px_common <= 0 {
        return 0;
    }

    if bdim[0] as i32 > 16 && col_off > 0 {
        let plane_bw_unit_64 = shrunken_unit(16, ss_x);
        let col_off_64 = col_off % plane_bw_unit_64;
        if col_off_64 == 0 {
            let plane_bh_unit_64 = shrunken_unit(16, ss_y);
            let row_off_64 = row_off % plane_bh_unit_64;
            let plane_bh_unit_limited = imin(plane_bh_unit, plane_bh_unit_64);
            let plane_bw_unit_128 = shrunken_unit(32, ss_x);
            let col_off_128 = col_off % plane_bw_unit_128;
            if col_off_128 == 0 {
                let plane_bh_unit_128 = shrunken_unit(32, ss_y);
                let row_off_128 = row_off % plane_bh_unit_128;
                return if row_off_128 + bottom_left_count_unit < plane_bh_unit_128 {
                    px_common
                } else {
                    0
                };
            }
            return if row_off_64 + bottom_left_count_unit < plane_bh_unit_limited {
                px_common
            } else {
                0
            };
        }
    }

    if col_off > 0 {
        return 0;
    }
    if row_off + bottom_left_count_unit < plane_bh_unit {
        return px_common;
    }

    let sb_w = shrunken_unit(sb_step, ss_x);
    let sb_h = shrunken_unit(sb_step, ss_y);
    // SB-local coordinates gate availability (AVM `& (sb_mi_size - 1)`); the
    // region-local coordinates (`& 63`) actually index the `is_coded` window.
    let bl_mask_row_sb = (base_y & (sb_h - 1)) + plane_bh_unit;
    let bl_mask_col_sb = (base_x & (sb_w - 1)) - 1;
    if bl_mask_col_sb < 0 {
        let plane_bottom_row = (base_y & (sb_h - 1)) + plane_bh_unit;
        return imin(sb_h - plane_bottom_row, px_common).max(0);
    }
    if bl_mask_row_sb >= sb_h {
        return 0;
    }
    let row0 = (base_y & 63) + plane_bh_unit;
    let col = (base_x & 63) - 1;
    if col < 0 || row0 >= 64 {
        return 0;
    }
    if (is_coded[coded_plane][row0 as usize] & (1u64 << (col as u32))) == 0 {
        return 0;
    }

    let mut coded = 0i32;
    while coded < bottom_left_count_unit {
        let r_sb = bl_mask_row_sb + coded;
        let r_rg = row0 + coded;
        if r_sb >= sb_h
            || r_rg >= 64
            || (is_coded[coded_plane][r_rg as usize] & (1u64 << (col as u32))) == 0
        {
            break;
        }
        coded += 1;
    }
    imin(coded, px_common)
}
