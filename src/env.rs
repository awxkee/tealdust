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

use crate::headers::{FrameHeader, WarpedMotionParams, WarpedMotionType};
use crate::intops::{apply_sign64, iclip, imax};
use crate::levels::{MvXY, RefPair, TIP_FRAME};

#[derive(Clone)]
pub(crate) struct BlockContext {
    pub(crate) fsc: [u8; 64],
    pub(crate) mode: [u8; 64],
    pub(crate) midx: [u8; 64],
    pub(crate) mrl: [u8; 64],
    pub(crate) multi_mrl: [u8; 64],
    pub(crate) dip: [u8; 64],
    pub(crate) lcoef: [u8; 64],
    pub(crate) ccoef: [[u8; 64]; 2],
    pub(crate) seg_pred: [u8; 64],
    pub(crate) skip_txfm: [u8; 64],
    pub(crate) skip_mode: [u8; 64],
    pub(crate) intra: [u8; 64],
    pub(crate) intrabc: [u8; 64],
    pub(crate) morph_pred: [u8; 64],
    pub(crate) comp_type: [u8; 64],
    pub(crate) r#ref: [[i8; 64]; 2],
    pub(crate) motion_mode: [u8; 64],
    pub(crate) amvd: [u8; 64],
    pub(crate) mvprec: [u8; 64],
    pub(crate) filter: [u8; 64],
    pub(crate) tx_lpf_y: [u8; 64],
    pub(crate) tx_lpf_uv: [u8; 64],
    pub(crate) partition: [[u8; 64]; 2],
    pub(crate) uvmode: [u8; 64],
    pub(crate) pal_sz: [u8; 64],
}

impl Default for BlockContext {
    fn default() -> Self {
        Self {
            fsc: [0; 64],
            mode: [0; 64],
            midx: [0; 64],
            mrl: [0; 64],
            multi_mrl: [0; 64],
            dip: [0; 64],
            lcoef: [0; 64],
            ccoef: [[0; 64]; 2],
            seg_pred: [0; 64],
            skip_txfm: [0; 64],
            skip_mode: [0; 64],
            intra: [0; 64],
            intrabc: [0; 64],
            morph_pred: [0; 64],
            comp_type: [0; 64],
            r#ref: [[0; 64]; 2],
            motion_mode: [0; 64],
            amvd: [0; 64],
            mvprec: [0; 64],
            filter: [0; 64],
            tx_lpf_y: [0; 64],
            tx_lpf_uv: [0; 64],
            partition: [[0; 64]; 2],
            uvmode: [0; 64],
            pal_sz: [0; 64],
        }
    }
}

#[derive(Clone)]
pub(crate) struct SBEdgeCtx {
    pub(crate) r#ref: [[i8; 64]; 2],
    pub(crate) motion_mode: [u8; 64],
}

impl Default for SBEdgeCtx {
    fn default() -> Self {
        Self {
            r#ref: [[0; 64]; 2],
            motion_mode: [0; 64],
        }
    }
}

#[inline(always)]
pub(crate) fn get_poc_diff(order_hint_n_bits: i32, poc0: i32, poc1: i32) -> i32 {
    if order_hint_n_bits == 0 {
        return 0;
    }
    let mask = 1 << (order_hint_n_bits - 1);
    let diff = poc0 - poc1;
    (diff & (mask - 1)) - (diff & mask)
}

#[inline(always)]
pub(crate) fn fix_int_mv_precision(mv: &mut MvXY) {
    mv.x = ((mv.x - (mv.x >> 15) + 3) as u32 & !7u32) as i32;
    mv.y = ((mv.y - (mv.y >> 15) + 3) as u32 & !7u32) as i32;
}

#[inline(always)]
pub(crate) fn mv_reduce_prec(mv: &mut MvXY, mv_prec: i32) {
    if mv_prec == 6 {
        return;
    }
    let rnd = 32 >> mv_prec;
    mv.x = mv.x + rnd - (mv.x > 0) as i32;
    mv.y = mv.y + rnd - (mv.y > 0) as i32;
    let mask = !(rnd as u32 * 2 - 1);
    mv.x = (mv.x as u32 & mask) as i32;
    mv.y = (mv.y as u32 & mask) as i32;
}

#[inline(always)]
pub(crate) fn get_warpmv_2d(
    matrix: &[i32; 6],
    bx4: i32,
    by4: i32,
    bw4: i32,
    bh4: i32,
    iw4: i32,
    ih4: i32,
    mv_precision: i32,
) -> MvXY {
    let x = bx4 * 4 + bw4 * 2 - 1;
    let y = by4 * 4 + bh4 * 2 - 1;
    let xc =
        (matrix[2] as i64 - (1 << 16)) * x as i64 + matrix[3] as i64 * y as i64 + matrix[0] as i64;
    let yc =
        (matrix[5] as i64 - (1 << 16)) * y as i64 + matrix[4] as i64 * x as i64 + matrix[1] as i64;
    let not_epel = (mv_precision < 6) as i32;
    let shift = 13 + not_epel;
    let rnd = (1i64 << shift) >> 1;
    let max = 0xffff - not_epel;

    let mut res = MvXY {
        y: iclip(
            apply_sign64(((yc.unsigned_abs() as i64 + rnd) >> shift) << not_epel, yc),
            -max,
            max,
        ),
        x: iclip(
            apply_sign64(((xc.unsigned_abs() as i64 + rnd) >> shift) << not_epel, xc),
            -max,
            max,
        ),
    };
    res.y = iclip(res.y, -(by4 + bh4 + 4) * 32, (ih4 - by4 + 4) * 32);
    res.x = iclip(res.x, -(bx4 + bw4 + 4) * 32, (iw4 - bx4 + 4) * 32);
    res
}

#[inline(always)]
pub(crate) fn get_gmv_2d(
    gmv: &WarpedMotionParams,
    bx4: i32,
    by4: i32,
    bw4: i32,
    bh4: i32,
    iw4: i32,
    ih4: i32,
    hdr: &FrameHeader,
) -> MvXY {
    match gmv.wm_type {
        WarpedMotionType::Affine | WarpedMotionType::RotZoom => {
            let mut res = get_warpmv_2d(
                &gmv.matrix,
                bx4,
                by4,
                bw4,
                bh4,
                iw4,
                ih4,
                hdr.mv_precision as i32 + 3,
            );
            if hdr.force_integer_mv != 0 {
                fix_int_mv_precision(&mut res);
            }
            res
        }
        WarpedMotionType::Translation => {
            let mut res = MvXY {
                y: gmv.matrix[0] >> 13,
                x: gmv.matrix[1] >> 13,
            };
            res.y = iclip(res.y, -(by4 + bh4 + 4) * 32, (ih4 - by4 + 4) * 32);
            res.x = iclip(res.x, -(bx4 + bw4 + 4) * 32, (iw4 - bx4 + 4) * 32);
            if hdr.force_integer_mv != 0 {
                fix_int_mv_precision(&mut res);
            }
            res
        }
        WarpedMotionType::Identity | WarpedMotionType::Invalid => MvXY { x: 0, y: 0 },
    }
}

#[inline(always)]
pub(crate) fn warp_type(mtx: &[i32; 6]) -> WarpedMotionType {
    if mtx[2] != mtx[5] || mtx[3] != -mtx[4] {
        return WarpedMotionType::Affine;
    }
    if mtx[2] != 0x10000 || mtx[3] != 0 {
        return WarpedMotionType::RotZoom;
    }
    if mtx[0] | mtx[1] != 0 {
        WarpedMotionType::Translation
    } else {
        WarpedMotionType::Identity
    }
}

#[inline(always)]
pub(crate) fn get_partition_ctx(
    a: &BlockContext,
    l: &BlockContext,
    b_dim: &[u8],
    plane: usize,
    yb4: usize,
    xb4: usize,
) -> i32 {
    ((a.partition[plane][xb4] >> imax(b_dim[2] as i32 - 1, 0)) & 1) as i32
        + (((l.partition[plane][yb4] >> imax(b_dim[3] as i32 - 1, 0)) & 1) as i32) * 2
}

#[inline(always)]
pub(crate) fn get_partition2_ctx(
    a: &BlockContext,
    l: &BlockContext,
    b_dim: &[u8],
    plane: usize,
    dir: i32,
    yb4: usize,
    xb4: usize,
) -> i32 {
    if dir == 0 {
        let hh4 = (b_dim[1] >> 1) as usize;
        ((l.partition[plane][yb4 + hh4] >> (b_dim[3] - 2)) & 1) as i32
            + (((l.partition[plane][yb4] >> (b_dim[3] - 2)) & 1) as i32) * 2
    } else {
        let hw4 = (b_dim[0] >> 1) as usize;
        ((a.partition[plane][xb4 + hw4] >> (b_dim[2] - 2)) & 1) as i32
            + (((a.partition[plane][xb4] >> (b_dim[2] - 2)) & 1) as i32) * 2
    }
}

#[inline(always)]
pub(crate) fn get_warp_ctx(
    a: &BlockContext,
    a_sb_cache: &SBEdgeCtx,
    l: &BlockContext,
    yb4: usize,
    xb4: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    top_is_at_tile_boundary: bool,
    b_dim: &[u8],
    r#ref: i8,
) -> i32 {
    let mut ctx = 0i32;

    macro_rules! add_bc {
        ($dir:expr, $idx:expr) => {
            ctx += (($dir.r#ref[0][$idx] == r#ref || $dir.r#ref[1][$idx] == r#ref)
                && $dir.motion_mode[$idx] >= 2) as i32;
        };
    }
    macro_rules! add_sb {
        ($dir:expr, $idx:expr) => {
            ctx += (($dir.r#ref[0][$idx] == r#ref || $dir.r#ref[1][$idx] == r#ref)
                && $dir.motion_mode[$idx] >= 2) as i32;
        };
    }

    if have_top {
        if top_is_at_tile_boundary {
            add_sb!(a_sb_cache, xb4 & !1);
            if have_top_right && b_dim[0] >= 4 {
                add_sb!(a_sb_cache, (xb4 + b_dim[0] as usize - 2) & !1);
            }
        } else {
            add_bc!(a, xb4);
            if have_top_right {
                add_bc!(a, xb4 + b_dim[0] as usize - 1);
            }
        }
    }
    if have_left {
        add_bc!(l, yb4);
        if have_bottom_left {
            add_bc!(l, yb4 + b_dim[1] as usize - 1);
        }
    }

    ctx
}

const NEWMV_COMP_MODE_MASK: u32 = (1 << 15)
    | (1 << 19)
    | (1 << 20)
    | (1 << 22)
    | (1 << 23)
    | (1 << 25)
    | (1 << 26)
    | (1 << 27)
    | (1 << 28);

#[inline(always)]
pub(crate) fn get_compref_ctx(
    a: &BlockContext,
    l: &BlockContext,
    yb4: usize,
    xb4: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    b_dim: &[u8],
    r#ref: RefPair,
    tip: RefPair,
) -> i32 {
    let mut row = 0i32;
    let mut col = 0i32;
    let mut newmv = 0i32;
    let (ref0, ref1) = (r#ref.r0(), r#ref.r1());
    let (tip0, tip1) = (tip.r0(), tip.r1());

    macro_rules! add_matching {
        ($dir:expr, $cnt:expr, $idx:expr) => {
            if $dir.r#ref[0][$idx] == TIP_FRAME as i8 && tip0 == ref0 && tip1 == ref1 {
                $cnt += 1;
                newmv += ($dir.mode[$idx] == 15) as i32; // NEWMV
            } else if $dir.r#ref[0][$idx] == ref0 && $dir.r#ref[1][$idx] == ref1 {
                $cnt += 1;
                newmv += (((1u32 << $dir.mode[$idx]) & NEWMV_COMP_MODE_MASK) != 0) as i32;
            }
        };
    }

    if have_top {
        add_matching!(a, col, xb4);
        if have_top_right {
            add_matching!(a, col, xb4 + b_dim[0] as usize - 1);
        }
    }
    if have_left {
        add_matching!(l, row, yb4);
        if have_bottom_left {
            add_matching!(l, row, yb4 + b_dim[1] as usize - 1);
        }
    }

    (row != 0) as i32 + (col != 0) as i32 + 2 * (newmv != 0) as i32
}
