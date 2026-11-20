/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Scaled-reference inter prediction (reference frame size differs from the
//! current frame). Mirrors AVM `av2/common/scale.c` (position mapping),
//! `av2_highbd_convolve_2d_scale_c` (translation), the scaled branch of
//! `dec_calc_subpel_params` and `av2_ext_highbd_warp_affine_scaled_c` (warp).
//!
//! These paths only run for scaled references, so they are plain scalar code
//! shared by every ISA.

use crate::internal::ScalableMotionParams;
use crate::mc::{get_h_filter, intermediate_bits, prep_bias};
use crate::pixel::BitDepth;

const REF_SCALE_SHIFT: i32 = 14;
const REF_NO_SCALE: i32 = 1 << REF_SCALE_SHIFT;
const SUBPEL_BITS: i32 = 4;
const SCALE_SUBPEL_BITS: i32 = 10;
const SCALE_SUBPEL_MASK: i32 = (1 << SCALE_SUBPEL_BITS) - 1;
const SCALE_EXTRA_BITS: i32 = SCALE_SUBPEL_BITS - SUBPEL_BITS;
const SCALE_EXTRA_OFF: i32 = (1 << SCALE_EXTRA_BITS) / 2;
const AVM_INTERP_EXTEND: i32 = 4;
/// `ALIGN_POWER_OF_TWO(2 * 256 + 2 * AVM_INTERP_EXTEND, 5)`.
const AVM_BORDER_IN_PIXELS: i32 = 544;
const WARPEDMODEL_PREC_BITS: i32 = 16;
const EXT_WARP_ROUND_BITS: i32 = WARPEDMODEL_PREC_BITS - 6;
const EXT_WARP_TAPS: usize = 6;
const MV_LOW: i32 = -(1 << 16);
const MV_UPP: i32 = 1 << 16;
/// `MV_BORDER` (16 pels in 1/8 pel units).
const MV_BORDER: i32 = 16 << 3;

/// Where an MC call sits inside AVM's prediction block. AVM predicts a whole
/// inter block in one call, while tealdust predicts >64px blocks per 64x64
/// piece. Scaled positions step from the block origin (`pos + x * xs`), so a
/// piece has to start from the block origin's position, not its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct McAnchor {
    /// Offset of the piece from the block origin, luma 4x4 units (luma plane).
    pub(crate) l_dx4: i32,
    pub(crate) l_dy4: i32,
    /// Offset of the chroma piece from the block's chroma origin, luma 4x4
    /// units.
    pub(crate) c_dx4: i32,
    pub(crate) c_dy4: i32,
    /// Full block size in luma 4x4 units; 0 when the call covers the block.
    pub(crate) bw4: i32,
    pub(crate) bh4: i32,
    /// Chroma unit the piece belongs to (luma 4x4 units): the chroma of a
    /// split block is predicted per unit, after all of the unit's luma pieces.
    pub(crate) cu_w4: i32,
    pub(crate) cu_h4: i32,
}

impl McAnchor {
    #[inline(always)]
    pub(crate) fn offset4(&self, pl: usize) -> (i32, i32) {
        if pl == 0 {
            (self.l_dx4, self.l_dy4)
        } else {
            (self.c_dx4, self.c_dy4)
        }
    }
}

/// AVM `struct scale_factors` (fixed-point part), plus the [`McAnchor`] of the
/// current MC call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScaleFactors {
    pub(crate) x_scale_fp: i32,
    pub(crate) y_scale_fp: i32,
    pub(crate) x_step_q4: i32,
    pub(crate) y_step_q4: i32,
    pub(crate) anchor: McAnchor,
}

impl Default for ScaleFactors {
    fn default() -> Self {
        Self::UNSCALED
    }
}

#[inline(always)]
fn round_power_of_two_signed_64(v: i64, n: i32) -> i64 {
    if v < 0 {
        -((-v + ((1i64 << n) >> 1)) >> n)
    } else {
        (v + ((1i64 << n) >> 1)) >> n
    }
}

impl ScaleFactors {
    pub(crate) const UNSCALED: Self = Self {
        x_scale_fp: REF_NO_SCALE,
        y_scale_fp: REF_NO_SCALE,
        x_step_q4: 1 << SCALE_SUBPEL_BITS,
        y_step_q4: 1 << SCALE_SUBPEL_BITS,
        anchor: McAnchor {
            l_dx4: 0,
            l_dy4: 0,
            c_dx4: 0,
            c_dy4: 0,
            bw4: 0,
            bh4: 0,
            cu_w4: 0,
            cu_h4: 0,
        },
    };

    /// Build from the per-reference `svc` pair (`scale == 0` means the
    /// reference has the current frame's size).
    #[inline]
    pub(crate) fn from_svc(svc: &[ScalableMotionParams; 2]) -> Self {
        if svc[0].scale == 0 {
            Self::UNSCALED
        } else {
            Self {
                x_scale_fp: svc[0].scale,
                y_scale_fp: svc[1].scale,
                x_step_q4: svc[0].step,
                y_step_q4: svc[1].step,
                anchor: McAnchor::default(),
            }
        }
    }

    /// Attach the anchor of the MC call these factors are used for.
    #[inline(always)]
    pub(crate) fn with_anchor(mut self, anchor: McAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// AVM `av2_is_scaled`.
    #[inline(always)]
    pub(crate) fn is_scaled(&self) -> bool {
        self.x_scale_fp != REF_NO_SCALE || self.y_scale_fp != REF_NO_SCALE
    }

    /// AVM `scaled_x` (q4 in, q10 out).
    #[inline(always)]
    fn scaled_x(&self, val: i32) -> i32 {
        let off = (self.x_scale_fp - REF_NO_SCALE) * (1 << (SUBPEL_BITS - 1));
        let tval = val as i64 * self.x_scale_fp as i64 + off as i64;
        round_power_of_two_signed_64(tval, REF_SCALE_SHIFT - SCALE_EXTRA_BITS) as i32
    }

    /// AVM `scaled_y` (q4 in, q10 out).
    #[inline(always)]
    fn scaled_y(&self, val: i32) -> i32 {
        let off = (self.y_scale_fp - REF_NO_SCALE) * (1 << (SUBPEL_BITS - 1));
        let tval = val as i64 * self.y_scale_fp as i64 + off as i64;
        round_power_of_two_signed_64(tval, REF_SCALE_SHIFT - SCALE_EXTRA_BITS) as i32
    }

    /// AVM `scaled_warp_x` (q16 in and out).
    #[inline(always)]
    fn scaled_warp_x(&self, val: i64) -> i64 {
        round_power_of_two_signed_64(val * self.x_scale_fp as i64, REF_SCALE_SHIFT)
    }

    /// AVM `scaled_warp_y` (q16 in and out).
    #[inline(always)]
    fn scaled_warp_y(&self, val: i64) -> i64 {
        round_power_of_two_signed_64(val * self.y_scale_fp as i64, REF_SCALE_SHIFT)
    }
}

/// Reference-plane position of a scaled prediction block, in 1/1024 pel
/// (AVM `dec_calc_subpel_params`, `is_scaled` branch).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledPos {
    pub(crate) pos_x: i32,
    pub(crate) pos_y: i32,
}

/// AVM `dec_calc_subpel_params` for a scaled reference. `pix_col`/`pix_row`
/// are the block origin in plane pixels, `mv_col`/`mv_row` the MV in 1/8 luma
/// pel, `plane_w`/`plane_h` the reference plane's (8-aligned) size.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn scaled_block_pos(
    sf: &ScaleFactors,
    pix_col: i32,
    pix_row: i32,
    mv_col: i32,
    mv_row: i32,
    ss_x: i32,
    ss_y: i32,
    plane_w: i32,
    plane_h: i32,
) -> ScaledPos {
    let orig_pos_y = (pix_row << SUBPEL_BITS) + mv_row * (1 << (1 - ss_y));
    let orig_pos_x = (pix_col << SUBPEL_BITS) + mv_col * (1 << (1 - ss_x));
    let mut pos_y = sf.scaled_y(orig_pos_y) + SCALE_EXTRA_OFF;
    let mut pos_x = sf.scaled_x(orig_pos_x) + SCALE_EXTRA_OFF;
    let top = -(((AVM_BORDER_IN_PIXELS >> ss_y) - AVM_INTERP_EXTEND) << SCALE_SUBPEL_BITS);
    let left = -(((AVM_BORDER_IN_PIXELS >> ss_x) - AVM_INTERP_EXTEND) << SCALE_SUBPEL_BITS);
    let bottom = (plane_h + AVM_INTERP_EXTEND) << SCALE_SUBPEL_BITS;
    let right = (plane_w + AVM_INTERP_EXTEND) << SCALE_SUBPEL_BITS;
    pos_y = pos_y.clamp(top, bottom);
    pos_x = pos_x.clamp(left, right);
    ScaledPos { pos_x, pos_y }
}

/// [`scaled_block_pos`] for an MC piece at plane position `pix_col`/`pix_row`
/// of `pl`: positions come from the anchoring block's origin, stepped by the
/// piece's offset (`pos + dx * xs`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn anchored_block_pos(
    sf: &ScaleFactors,
    pl: usize,
    pix_col: i32,
    pix_row: i32,
    mv_col: i32,
    mv_row: i32,
    ss_x: i32,
    ss_y: i32,
    plane_w: i32,
    plane_h: i32,
) -> ScaledPos {
    let (dx4, dy4) = sf.anchor.offset4(pl);
    let dx = (dx4 * 4) >> ss_x;
    let dy = (dy4 * 4) >> ss_y;
    let p = scaled_block_pos(
        sf,
        pix_col - dx,
        pix_row - dy,
        mv_col,
        mv_row,
        ss_x,
        ss_y,
        plane_w,
        plane_h,
    );
    ScaledPos {
        pos_x: p.pos_x + dx * sf.x_step_q4,
        pos_y: p.pos_y + dy * sf.y_step_q4,
    }
}

/// Halved (6-bit) subpel kernel for a scaled MC phase, matching the tables the
/// unscaled kernels use. Phase 0 is the identity tap AVM's scaled convolve
/// still runs through.
#[inline(always)]
fn scaled_kernel(phase: i32, filter_type: i32, len: usize) -> [i8; 8] {
    if phase == 0 {
        [0, 0, 0, 64, 0, 0, 0, 0]
    } else if filter_type == 3 {
        // av2_bilinear_filters: { 128 - 8p, 8p } at taps 3/4, halved.
        let p = phase as i8;
        [0, 0, 0, 64 - 4 * p, 4 * p, 0, 0, 0]
    } else {
        get_h_filter(phase, filter_type, len).unwrap_or([0, 0, 0, 64, 0, 0, 0, 0])
    }
}

/// Output of a scaled prediction: pixels, or the compound intermediate.
pub(crate) enum ScaledDst<'a, P> {
    Put(&'a mut [P], usize),
    Prep(&'a mut [i16], usize),
}

/// AVM `av2_highbd_convolve_2d_scale_c` for one block. Reads `src` (a plane of
/// size `src_w` x `src_h`) with edge replication, which is what AVM's
/// `extend_mc_border` / frame border amount to. `filter_type` is the block's
/// interpolation filter (3 = bilinear).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mc_scaled<BD: BitDepth>(
    bd: BD,
    dst: ScaledDst<'_, BD::Pixel>,
    src: &[BD::Pixel],
    src_stride: usize,
    src_w: i32,
    src_h: i32,
    pos: ScaledPos,
    w: usize,
    h: usize,
    sf: &ScaleFactors,
    filter_type: i32,
) {
    if w == 0 || h == 0 || src_w <= 0 || src_h <= 0 {
        return;
    }
    let ib = intermediate_bits(bd);
    let xs = sf.x_step_q4;
    let ys = sf.y_step_q4;
    let subpel_x = pos.pos_x & SCALE_SUBPEL_MASK;
    let subpel_y = pos.pos_y & SCALE_SUBPEL_MASK;
    let x0 = pos.pos_x >> SCALE_SUBPEL_BITS;
    let y0 = pos.pos_y >> SCALE_SUBPEL_BITS;
    let im_h = ((((h as i32 - 1) * ys + subpel_y) >> SCALE_SUBPEL_BITS) + 8) as usize;
    let last = src.len() as isize - 1;
    let px = |y: i32, x: i32| -> i32 {
        let yy = y.clamp(0, src_h - 1) as isize;
        let xx = x.clamp(0, src_w - 1) as isize;
        src[(yy * src_stride as isize + xx).clamp(0, last) as usize].into()
    };

    // Horizontal pass into the intermediate (rows y0-3 .. y0-3+im_h).
    let h_sh = 6 - ib;
    let h_rnd = (1 << h_sh) >> 1;
    let mut mid = vec![0i16; im_h * w];
    let mut fx = Vec::with_capacity(w);
    for x in 0..w as i32 {
        let x_qn = subpel_x + x * xs;
        let phase = (x_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS;
        fx.push((
            x0 + (x_qn >> SCALE_SUBPEL_BITS),
            scaled_kernel(phase, filter_type, w),
        ));
    }
    for y in 0..im_h {
        let sy = y0 - 3 + y as i32;
        for (x, (ix, f)) in fx.iter().enumerate() {
            let mut sum = 0i32;
            for k in 0..8 {
                if f[k] != 0 {
                    sum += f[k] as i32 * px(sy, ix + k as i32 - 3);
                }
            }
            mid[y * w + x] = ((sum + h_rnd) >> h_sh) as i16;
        }
    }

    // Vertical pass.
    let bias = prep_bias(bd);
    let mut dst = dst;
    for y in 0..h as i32 {
        let y_qn = subpel_y + y * ys;
        let phase = (y_qn & SCALE_SUBPEL_MASK) >> SCALE_EXTRA_BITS;
        let f = scaled_kernel(phase, filter_type, h);
        let row = (y_qn >> SCALE_SUBPEL_BITS) as usize;
        for x in 0..w {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += f[k] as i32 * mid[(row + k) * w + x] as i32;
            }
            match dst {
                ScaledDst::Put(ref mut d, stride) => {
                    let sh = 6 + ib;
                    d[y as usize * stride + x] = bd.pixel_clip((sum + ((1 << sh) >> 1)) >> sh);
                }
                ScaledDst::Prep(ref mut t, stride) => {
                    t[y as usize * stride + x] = (((sum + 32) >> 6) - bias) as i16;
                }
            }
        }
    }
}

/// AVM `av2_ext_warped_filter` (6 taps, 65 phases).
#[rustfmt::skip]
static EXT_WARP_FILTER_SCALED: [[i16; EXT_WARP_TAPS]; 65] = [
    [0, 0, 128, 0, 0, 0], [0, -1, 127, 2, 0, 0], [0, -2, 127, 4, -1, 0], [0, -3, 126, 6, -1, 0],
    [1, -4, 125, 8, -2, 0], [1, -5, 124, 11, -3, 0], [1, -6, 123, 13, -3, 0], [1, -6, 122, 15, -4, 0],
    [1, -7, 120, 17, -4, 1], [1, -8, 119, 20, -5, 1], [1, -9, 118, 22, -5, 1], [1, -9, 117, 24, -6, 1],
    [1, -10, 115, 27, -6, 1], [1, -10, 114, 29, -7, 1], [1, -11, 112, 32, -7, 1], [1, -11, 111, 34, -8, 1],
    [1, -11, 109, 36, -8, 1], [2, -12, 107, 39, -9, 1], [2, -12, 105, 41, -9, 1], [2, -12, 103, 44, -10, 1],
    [2, -13, 102, 46, -10, 1], [2, -13, 99, 49, -10, 1], [2, -13, 98, 51, -11, 1], [2, -13, 95, 54, -11, 1],
    [2, -14, 93, 56, -11, 2], [2, -14, 91, 59, -12, 2], [2, -14, 89, 61, -12, 2], [2, -14, 87, 63, -12, 2],
    [2, -14, 85, 66, -13, 2], [2, -14, 83, 68, -13, 2], [2, -14, 80, 71, -13, 2], [2, -14, 78, 73, -13, 2],
    [2, -13, 75, 75, -13, 2], [2, -13, 73, 78, -14, 2], [2, -13, 71, 80, -14, 2], [2, -13, 68, 83, -14, 2],
    [2, -13, 66, 85, -14, 2], [2, -12, 63, 87, -14, 2], [2, -12, 61, 89, -14, 2], [2, -12, 59, 91, -14, 2],
    [2, -11, 56, 93, -14, 2], [1, -11, 54, 95, -13, 2], [1, -11, 51, 98, -13, 2], [1, -10, 49, 99, -13, 2],
    [1, -10, 46, 102, -13, 2], [1, -10, 44, 103, -12, 2], [1, -9, 41, 105, -12, 2], [1, -9, 39, 107, -12, 2],
    [1, -8, 36, 109, -11, 1], [1, -8, 34, 111, -11, 1], [1, -7, 32, 112, -11, 1], [1, -7, 29, 114, -10, 1],
    [1, -6, 27, 115, -10, 1], [1, -6, 24, 117, -9, 1], [1, -5, 22, 118, -9, 1], [1, -5, 20, 119, -8, 1],
    [1, -4, 17, 120, -7, 1], [0, -4, 15, 122, -6, 1], [0, -3, 13, 123, -6, 1], [0, -3, 11, 124, -5, 1],
    [0, -2, 8, 125, -4, 1], [0, -1, 6, 126, -3, 0], [0, -1, 4, 127, -2, 0], [0, 0, 2, 127, -1, 0],
    [0, 0, 0, 128, 0, 0],
];

/// One `PadBlock` of AVM's warp bounding-box grid (inclusive limits).
#[derive(Clone, Copy, Default, Debug)]
struct WarpBox {
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
}

/// Block geometry the scaled warp needs from the current block (luma 4x4
/// units) for AVM's `clamp_mv_ref` in `get_int_warp_mv_for_fb`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WarpBlockCtx {
    /// Block origin and size in luma 4x4 units.
    pub(crate) bx4: i32,
    pub(crate) by4: i32,
    pub(crate) bw4: i32,
    pub(crate) bh4: i32,
    /// Frame size in luma 4x4 units (`mi_cols` / `mi_rows`).
    pub(crate) mi_cols: i32,
    pub(crate) mi_rows: i32,
}

/// AVM `get_int_warp_mv_for_fb`: integer-precision (1/8 pel) MV of a warp
/// sub-block, clamped like `clamp_mv_ref`.
fn int_warp_mv_for_fb(
    mat: &[i32; 6],
    bw: i32,
    bh: i32,
    mi_col: i32,
    mi_row: i32,
    blk: &WarpBlockCtx,
) -> (i32, i32) {
    let x = mi_col * 4 + (bw >> 1);
    let y = mi_row * 4 + (bh >> 1);
    let xc = ((mat[2] as i64 - (1 << WARPEDMODEL_PREC_BITS)) * x as i64
        + mat[3] as i64 * y as i64
        + mat[0] as i64)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let yc = (mat[4] as i64 * x as i64
        + (mat[5] as i64 - (1 << WARPEDMODEL_PREC_BITS)) * y as i64
        + mat[1] as i64)
        .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let tx = xc >> (WARPEDMODEL_PREC_BITS - 3);
    let ty = yc >> (WARPEDMODEL_PREC_BITS - 3);
    let mut col = tx.clamp(MV_LOW + 1, MV_UPP - 1);
    let mut row = ty.clamp(MV_LOW + 1, MV_UPP - 1);
    // clamp_mv_ref(mv, xd->width * 4, xd->height * 4, xd)
    let bw_px = blk.bw4 * 4;
    let bh_px = blk.bh4 * 4;
    let to_left = -(blk.bx4 * 4 * 8);
    let to_right = (blk.mi_cols - blk.bw4 - blk.bx4) * 4 * 8;
    let to_top = -(blk.by4 * 4 * 8);
    let to_bottom = (blk.mi_rows - blk.bh4 - blk.by4) * 4 * 8;
    let col_min = to_left - bw_px * 8 - MV_BORDER;
    let col_max = to_right + bw_px * 8 + MV_BORDER;
    let row_min = to_top - bh_px * 8 - MV_BORDER;
    let row_max = to_bottom + bh_px * 8 + MV_BORDER;
    col = col.clamp(col_min, col_max);
    row = row.clamp(row_min, row_max);
    (col, row)
}

/// AVM `av2_ext_highbd_warp_affine_scaled_c` for one block, including the
/// per-8x8 `warp_bd_box` set up in `build_inter_predictors_8x8_and_bigger`
/// (`get_int_warp_mv_for_fb` + `av2_get_reference_area_with_padding_single`).
/// `p_col`/`p_row`/`p_w`/`p_h` are the block in plane pixels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_scaled<BD: BitDepth>(
    bd: BD,
    dst: ScaledDst<'_, BD::Pixel>,
    src: &[BD::Pixel],
    src_stride: usize,
    src_w: i32,
    src_h: i32,
    mat: &[i32; 6],
    p_col: i32,
    p_row: i32,
    p_w: i32,
    p_h: i32,
    ss_x: i32,
    ss_y: i32,
    sf: &ScaleFactors,
    blk: &WarpBlockCtx,
) {
    if p_w <= 0 || p_h <= 0 || src_w <= 0 || src_h <= 0 {
        return;
    }
    let bitdepth = bd.bitdepth() as i32;
    // get_conv_params_no_round(): round_0 = 3 (+2 at 12-bit), round_1 = 7
    // for compound, 2 * FILTER_BITS - round_0 otherwise.
    let round_0 = if bitdepth == 12 { 5 } else { 3 };
    let is_compound = matches!(dst, ScaledDst::Prep(..));
    let reduce_bits_vert = if is_compound { 7 } else { 14 - round_0 };
    let offset_bits_horiz = bitdepth + 7 - 1;
    let offset_bits_vert = bitdepth + 14 - round_0;
    let last = src.len() as isize - 1;
    let px = |y: i32, x: i32| -> i32 {
        src[(y as isize * src_stride as isize + x as isize).clamp(0, last) as usize].into()
    };

    // Bounding boxes, one per 8x8 plane sub-block.
    let bbw = p_w.min(8);
    let bbh = p_h.min(8);
    let n_bx = ((p_w + 7) >> 3) as usize;
    let n_by = ((p_h + 7) >> 3) as usize;
    let mut boxes = vec![WarpBox::default(); n_bx * n_by];
    for by_i in 0..n_by {
        for bx_i in 0..n_bx {
            let sub_x = p_col + 8 * bx_i as i32;
            let sub_y = p_row + 8 * by_i as i32;
            let (mv_col, mv_row) = int_warp_mv_for_fb(
                mat,
                bbw << ss_x,
                bbh << ss_y,
                (sub_x << ss_x) >> 2,
                (sub_y << ss_y) >> 2,
                blk,
            );
            let pos = scaled_block_pos(sf, sub_x, sub_y, mv_col, mv_row, ss_x, ss_y, src_w, src_h);
            let x0 = (pos.pos_x >> SCALE_SUBPEL_BITS) - (AVM_INTERP_EXTEND - 1);
            let y0 = (pos.pos_y >> SCALE_SUBPEL_BITS) - (AVM_INTERP_EXTEND - 1);
            let x1 = ((pos.pos_x + (bbw - 1) * sf.x_step_q4) >> SCALE_SUBPEL_BITS)
                + 1
                + AVM_INTERP_EXTEND;
            let y1 = ((pos.pos_y + (bbh - 1) * sf.y_step_q4) >> SCALE_SUBPEL_BITS)
                + 1
                + AVM_INTERP_EXTEND;
            let x0 = x0.clamp(0, src_w - 1);
            let y0 = y0.clamp(0, src_h - 1);
            let x1 = x1.clamp(1, src_w);
            let y1 = y1.clamp(1, src_h);
            boxes[by_i * n_bx + bx_i] = WarpBox {
                left: x0,
                right: x1 - 1,
                top: y0,
                bottom: y1 - 1,
            };
        }
    }

    let fo = (EXT_WARP_TAPS / 2 - 1) as i32;
    let x_step_qn = sf.x_step_q4 << (WARPEDMODEL_PREC_BITS - SCALE_SUBPEL_BITS);
    let y_step_qn = sf.y_step_q4 << (WARPEDMODEL_PREC_BITS - SCALE_SUBPEL_BITS);
    let mut im = [0i32; (2 * 4 + EXT_WARP_TAPS) * 4];
    let bias = prep_bias(bd);
    let comp_off = (1 << (offset_bits_vert - 7)) + (1 << (offset_bits_vert - 7 - 1));
    let mut dst = dst;

    let mut i = p_row;
    while i < p_row + p_h {
        let mut j = p_col;
        while j < p_col + p_w {
            let b = boxes[(((i - p_row) >> 3) as usize) * n_bx + ((j - p_col) >> 3) as usize];
            let src_x = (j + 2) << ss_x;
            let src_y = (i + 2) << ss_y;
            let dst_x = mat[2] as i64 * src_x as i64 + mat[3] as i64 * src_y as i64 + mat[0] as i64;
            let dst_y = mat[4] as i64 * src_x as i64 + mat[5] as i64 * src_y as i64 + mat[1] as i64;
            let x4_org = (dst_x >> ss_x) - 2 * (1 << WARPEDMODEL_PREC_BITS);
            let y4_org = (dst_y >> ss_y) - 2 * (1 << WARPEDMODEL_PREC_BITS);
            let x4 = sf.scaled_warp_x(x4_org);
            let y4 = sf.scaled_warp_y(y4_org);
            let iy4 = (y4 >> WARPEDMODEL_PREC_BITS) as i32;
            let subpel_y_qn = (y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1)) as i32;

            let im_h = (((4 - 1) * y_step_qn + subpel_y_qn) >> WARPEDMODEL_PREC_BITS) as usize
                + EXT_WARP_TAPS;
            let init_y = iy4 - fo;
            for y in 0..im_h {
                let iy = (init_y + y as i32).clamp(b.top, b.bottom);
                let mut x_qn = x4;
                for x in 0..4 {
                    let sx4 = (x_qn & ((1 << WARPEDMODEL_PREC_BITS) - 1)) as i32;
                    let offs_x = (sx4 + (1 << (EXT_WARP_ROUND_BITS - 1))) >> EXT_WARP_ROUND_BITS;
                    let coeffs = &EXT_WARP_FILTER_SCALED[offs_x as usize];
                    let ix = (x_qn >> WARPEDMODEL_PREC_BITS) as i32;
                    let mut sum = 1i32 << offset_bits_horiz;
                    for (m, &c) in coeffs.iter().enumerate() {
                        let sample_x = (ix - fo + m as i32).clamp(b.left, b.right);
                        sum += px(iy, sample_x) * c as i32;
                    }
                    im[y * 4 + x] = (sum + ((1 << round_0) >> 1)) >> round_0;
                    x_qn += x_step_qn as i64;
                }
            }

            let wx = 4.min(p_col + p_w - j);
            let wy = 4.min(p_row + p_h - i);
            for x in 0..wx {
                let mut y_qn = subpel_y_qn as i64;
                for y in 0..wy {
                    let sy4 = (y_qn & ((1 << WARPEDMODEL_PREC_BITS) - 1)) as i32;
                    let offs_y = (sy4 + (1 << (EXT_WARP_ROUND_BITS - 1))) >> EXT_WARP_ROUND_BITS;
                    let coeffs = &EXT_WARP_FILTER_SCALED[offs_y as usize];
                    let base = (y_qn >> WARPEDMODEL_PREC_BITS) as usize + fo as usize;
                    let mut sum = 1i32 << offset_bits_vert;
                    for (m, &c) in coeffs.iter().enumerate() {
                        sum += im[(base + m - fo as usize) * 4 + x as usize] * c as i32;
                    }
                    let sum = (sum + ((1 << reduce_bits_vert) >> 1)) >> reduce_bits_vert;
                    let oy = (i - p_row + y) as usize;
                    let ox = (j - p_col + x) as usize;
                    match dst {
                        ScaledDst::Put(ref mut d, stride) => {
                            d[oy * stride + ox] =
                                bd.pixel_clip(sum - (1 << (bitdepth - 1)) - (1 << bitdepth));
                        }
                        ScaledDst::Prep(ref mut t, stride) => {
                            t[oy * stride + ox] = (sum - comp_off - bias) as i16;
                        }
                    }
                    y_qn += y_step_qn as i64;
                }
            }
            j += 4;
        }
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_factor_mapping_matches_avm() {
        // 150 -> 100 (resize denominator 12 of 8).
        let svc = [
            ScalableMotionParams {
                scale: ((150 << 14) + 50) / 100,
                step: ((((150 << 14) + 50) / 100) + 8) >> 4,
            },
            ScalableMotionParams {
                scale: ((90 << 14) + 30) / 60,
                step: ((((90 << 14) + 30) / 60) + 8) >> 4,
            },
        ];
        let sf = ScaleFactors::from_svc(&svc);
        assert!(sf.is_scaled());
        assert_eq!(sf.x_step_q4, 1536);
        // A zero position maps to the scale centring offset.
        assert_eq!(
            sf.scaled_x(0),
            (((sf.x_scale_fp - REF_NO_SCALE) * 8) + 128) >> 8
        );
        assert!(!ScaleFactors::UNSCALED.is_scaled());
    }

    #[test]
    fn scaled_identity_copy() {
        // A 2x downscale with zero phase picks every other pixel.
        let sf = ScaleFactors {
            x_scale_fp: 2 << 14,
            y_scale_fp: 2 << 14,
            x_step_q4: 2048,
            y_step_q4: 2048,
            anchor: McAnchor::default(),
        };
        let w = 16usize;
        let src: Vec<u8> = (0..w * w).map(|i| (i % 251) as u8).collect();
        let mut out = vec![0u8; 4 * 4];
        mc_scaled(
            crate::pixel::BitDepth8,
            ScaledDst::Put(&mut out, 4),
            &src,
            w,
            w as i32,
            w as i32,
            ScaledPos { pos_x: 0, pos_y: 0 },
            4,
            4,
            &sf,
            0,
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out[y * 4 + x], src[2 * y * w + 2 * x]);
            }
        }
    }
}
