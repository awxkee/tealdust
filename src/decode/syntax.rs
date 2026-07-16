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
use crate::cdf::{CdfModeContext, CdfMvContext};
use crate::env::warp_type;
use crate::headers::{NSWienerPlane, RestorationType, WarpedMotionParams, WarpedMotionType};
use crate::internal::NsWienerBank;
use crate::intops::{iclip, imax, imin, inv_recenter};
use crate::levels::{BlockSize, INVALID_MV, Mv, MvXY, N_BS_SIZES, TxPartition};
use crate::lf_mask::Av2RestorationUnit;
use crate::msac::MsacReader;
use crate::pal::pal_idx_finish;

use crate::refmvs;
use crate::tables::{
    BLOCK_DIMENSIONS, DEFAULT_WM_PARAMS, NS_WIENER_COEF_RANGE_UV, NS_WIENER_COEF_RANGE_Y,
    SUBSET_MASKS_UV, SUBSET_MASKS_Y,
};
use crate::warpmv::{find_affine_int, get_shear_params, set_affine_mv2d};

pub(crate) static SIZE_GROUP: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    // group 0: 4x4, 4x8, 8x4, 4x16, 16x4
    t[BlockSize::Bs4x4 as usize] = 0;
    t[BlockSize::Bs4x8 as usize] = 0;
    t[BlockSize::Bs8x4 as usize] = 0;
    t[BlockSize::Bs4x16 as usize] = 0;
    t[BlockSize::Bs16x4 as usize] = 0;
    // group 1: 8x8, 8x16, 16x8, 8x32, 32x8, 4x32, 32x4
    t[BlockSize::Bs8x8 as usize] = 1;
    t[BlockSize::Bs8x16 as usize] = 1;
    t[BlockSize::Bs16x8 as usize] = 1;
    t[BlockSize::Bs8x32 as usize] = 1;
    t[BlockSize::Bs32x8 as usize] = 1;
    t[BlockSize::Bs4x32 as usize] = 1;
    t[BlockSize::Bs32x4 as usize] = 1;
    // group 2: 16x16, 16x32, 32x16, 16x64, 64x16, 8x64, 64x8, 4x64, 64x4
    t[BlockSize::Bs16x16 as usize] = 2;
    t[BlockSize::Bs16x32 as usize] = 2;
    t[BlockSize::Bs32x16 as usize] = 2;
    t[BlockSize::Bs16x64 as usize] = 2;
    t[BlockSize::Bs64x16 as usize] = 2;
    t[BlockSize::Bs8x64 as usize] = 2;
    t[BlockSize::Bs64x8 as usize] = 2;
    t[BlockSize::Bs4x64 as usize] = 2;
    t[BlockSize::Bs64x4 as usize] = 2;
    // group 3: 32x32+
    t[BlockSize::Bs32x32 as usize] = 3;
    t[BlockSize::Bs32x64 as usize] = 3;
    t[BlockSize::Bs64x32 as usize] = 3;
    t[BlockSize::Bs64x64 as usize] = 3;
    t[BlockSize::Bs64x128 as usize] = 3;
    t[BlockSize::Bs128x64 as usize] = 3;
    t[BlockSize::Bs128x128 as usize] = 3;
    t[BlockSize::Bs128x256 as usize] = 3;
    t[BlockSize::Bs256x128 as usize] = 3;
    t[BlockSize::Bs256x256 as usize] = 3;
    t
};

// TX partition size group per block size
pub(crate) static TX_PART_GROUP: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    t[BlockSize::Bs8x4 as usize] = 0;
    t[BlockSize::Bs4x8 as usize] = 0;
    t[BlockSize::Bs4x4 as usize] = 0;
    t[BlockSize::Bs8x8 as usize] = 1;
    t[BlockSize::Bs16x8 as usize] = 2;
    t[BlockSize::Bs8x16 as usize] = 2;
    t[BlockSize::Bs16x16 as usize] = 3;
    t[BlockSize::Bs32x16 as usize] = 4;
    t[BlockSize::Bs16x32 as usize] = 4;
    t[BlockSize::Bs32x32 as usize] = 5;
    t[BlockSize::Bs64x32 as usize] = 6;
    t[BlockSize::Bs32x64 as usize] = 6;
    t[BlockSize::Bs64x64 as usize] = 7;
    // extended sizes map to 8
    t[BlockSize::Bs64x16 as usize] = 8;
    t[BlockSize::Bs64x8 as usize] = 8;
    t[BlockSize::Bs64x4 as usize] = 8;
    t[BlockSize::Bs32x8 as usize] = 8;
    t[BlockSize::Bs32x4 as usize] = 8;
    t[BlockSize::Bs16x64 as usize] = 8;
    t[BlockSize::Bs16x4 as usize] = 8;
    t[BlockSize::Bs8x64 as usize] = 8;
    t[BlockSize::Bs8x32 as usize] = 8;
    t[BlockSize::Bs4x64 as usize] = 8;
    t[BlockSize::Bs4x32 as usize] = 8;
    t[BlockSize::Bs4x16 as usize] = 8;
    t
};

// TX type group for 2D V/H partition per block size
pub(crate) static TX_TYPE_GROUP_VH: [u8; N_BS_SIZES] = {
    let mut t = [0u8; N_BS_SIZES];
    t[BlockSize::Bs8x8 as usize] = 0;
    t[BlockSize::Bs8x16 as usize] = 1;
    t[BlockSize::Bs16x8 as usize] = 2;
    t[BlockSize::Bs16x16 as usize] = 3;
    t[BlockSize::Bs16x32 as usize] = 4;
    t[BlockSize::Bs32x16 as usize] = 5;
    t[BlockSize::Bs32x32 as usize] = 6;
    t[BlockSize::Bs32x64 as usize] = 7;
    t[BlockSize::Bs64x32 as usize] = 8;
    t[BlockSize::Bs64x64 as usize] = 9;
    t[BlockSize::Bs8x32 as usize] = 10;
    t[BlockSize::Bs8x64 as usize] = 10;
    t[BlockSize::Bs64x8 as usize] = 11;
    t[BlockSize::Bs32x8 as usize] = 11;
    t[BlockSize::Bs16x64 as usize] = 12;
    t[BlockSize::Bs64x16 as usize] = 13;
    t
};

pub(crate) fn jmvd_scale(mv: &mut MvXY, amvd: bool, jmvd_scale_mode: i32) {
    if amvd {
        match jmvd_scale_mode {
            0 => {}
            1 => {
                mv.y *= 2;
                mv.x *= 2;
            }
            2 => {
                mv.y /= 2;
                mv.x /= 2;
            }
            _ => unreachable!(),
        }
    } else {
        match jmvd_scale_mode {
            0 => {}
            1 => mv.y *= 2,
            2 => mv.x *= 2,
            3 => mv.y /= 2,
            4 => mv.x /= 2,
            _ => unreachable!(),
        }
    }
}

pub(crate) fn get_prev_frame_segid(
    by: i32,
    bx: i32,
    w4: i32,
    h4: i32,
    ref_seg_map: &[u8],
    stride: isize,
) -> u32 {
    let mut seg_id = 8u32;
    let mut off = (by as isize * stride + bx as isize) as usize;
    for _ in 0..h4 {
        for x in 0..w4 as usize {
            seg_id = imin(seg_id as i32, ref_seg_map[off + x] as i32) as u32;
        }
        if seg_id == 0 {
            break;
        }
        off = (off as isize + stride) as usize;
    }
    seg_id
}

/// Spatial-prediction of the current-frame segment id (`get_cur_frame_segid`,
/// class into `seg_ctx`.
pub(crate) fn get_cur_frame_segid(
    by: i32,
    bx: i32,
    have_top: bool,
    have_left: bool,
    seg_ctx: &mut i32,
    cur_seg_map: &[u8],
    stride: isize,
) -> u32 {
    let base = (bx as isize + by as isize * stride) as usize;
    if have_left && have_top {
        let l = cur_seg_map[base - 1] as i32;
        let a = cur_seg_map[(base as isize - stride) as usize] as i32;
        let al = cur_seg_map[(base as isize - (stride + 1)) as usize] as i32;
        if l == a && al == l {
            *seg_ctx = 2;
        } else if l == a || al == l || a == al {
            *seg_ctx = 1;
        } else {
            *seg_ctx = 0;
        }
        (if a == al { a } else { l }) as u32
    } else {
        *seg_ctx = 0;
        if have_left {
            cur_seg_map[base - 1] as u32
        } else if have_top {
            cur_seg_map[(base as isize - stride) as usize] as u32
        } else {
            0
        }
    }
}

pub(crate) static MV_PREC_TBL: [[u8; 3]; 2] = [[3, 1, 0], [4, 3, 1]];

use crate::levels::N_PARTITIONS;

// child partition split limits: [w_limit, h_limit]
pub(crate) static PARTITION_LIM: [[u8; 2]; N_PARTITIONS] = [
    [1, 1], // NONE
    [1, 2], // H
    [2, 1], // V
    [2, 4], // H3
    [4, 2], // V3
    [1, 8], // H4A
    [1, 8], // H4B
    [8, 1], // V4A
    [8, 1], // V4B
    [2, 2], // SPLIT
];

pub(crate) static WEDGE_ANGLE_DIST2IDX: [[i8; 4]; 20] = [
    [-1, 0, 1, 2],    // WEDGE_0
    [3, 4, 5, 6],     // WEDGE_14
    [7, 8, 9, 10],    // WEDGE_27
    [11, 12, 13, 14], // WEDGE_45
    [15, 16, 17, 18], // WEDGE_63
    [-1, 19, 20, 21], // WEDGE_90
    [22, 23, 24, 25], // WEDGE_117
    [26, 27, 28, 29], // WEDGE_135
    [30, 31, 32, 33], // WEDGE_153
    [34, 35, 36, 37], // WEDGE_166
    [-1, 38, 39, 40], // WEDGE_180
    [-1, 41, 42, 43], // WEDGE_194
    [-1, 44, 45, 46], // WEDGE_207
    [-1, 47, 48, 49], // WEDGE_225
    [-1, 50, 51, 52], // WEDGE_243
    [-1, 53, 54, 55], // WEDGE_270
    [-1, 56, 57, 58], // WEDGE_297
    [-1, 59, 60, 61], // WEDGE_315
    [-1, 62, 63, 64], // WEDGE_333
    [-1, 65, 66, 67], // WEDGE_346
];

#[derive(Clone, Copy)]
pub(crate) struct PartitionConstants {
    pub(crate) part: [[i8; 4]; 2],
    pub(crate) ctx: [i8; 2],
}

const I: i8 = -1; // BS_INVALID shorthand
use BlockSize::*;

pub(crate) static PARTITION_SUBB: [PartitionConstants; N_BS_SIZES] = {
    let mut t = [PartitionConstants {
        part: [[I; 4]; 2],
        ctx: [I; 2],
    }; N_BS_SIZES];

    t[Bs256x256 as usize] = PartitionConstants {
        part: [
            [Bs256x128 as i8, I, I, Bs128x128 as i8],
            [Bs128x256 as i8, I, I, Bs128x128 as i8],
        ],
        ctx: [9, 12],
    };
    t[Bs256x128 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs128x128 as i8, I, I, I]],
        ctx: [8, I],
    };
    t[Bs128x256 as usize] = PartitionConstants {
        part: [[Bs128x128 as i8, I, I, I], [I, I, I, I]],
        ctx: [7, I],
    };
    t[Bs128x128 as usize] = PartitionConstants {
        part: [
            [Bs128x64 as i8, I, I, Bs64x64 as i8],
            [Bs64x128 as i8, I, I, Bs64x64 as i8],
        ],
        ctx: [6, 9],
    };
    t[Bs128x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs64x64 as i8, I, I, I]],
        ctx: [5, I],
    };
    t[Bs64x128 as usize] = PartitionConstants {
        part: [[Bs64x64 as i8, I, I, I], [I, I, I, I]],
        ctx: [4, I],
    };
    t[Bs64x64 as usize] = PartitionConstants {
        part: [
            [Bs64x32 as i8, Bs64x16 as i8, Bs64x8 as i8, Bs32x32 as i8],
            [Bs32x64 as i8, Bs16x64 as i8, Bs8x64 as i8, Bs32x32 as i8],
        ],
        ctx: [3, 6],
    };
    t[Bs64x32 as usize] = PartitionConstants {
        part: [
            [Bs64x16 as i8, Bs64x8 as i8, Bs64x4 as i8, Bs32x16 as i8],
            [Bs32x32 as i8, Bs16x32 as i8, Bs8x32 as i8, Bs32x16 as i8],
        ],
        ctx: [3, 5],
    };
    t[Bs64x16 as usize] = PartitionConstants {
        part: [
            [Bs64x8 as i8, Bs64x4 as i8, I, Bs32x8 as i8],
            [Bs32x16 as i8, Bs16x16 as i8, Bs8x16 as i8, Bs32x8 as i8],
        ],
        ctx: [15, 14],
    };
    t[Bs64x8 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs64x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs32x64 as usize] = PartitionConstants {
        part: [
            [Bs32x32 as i8, Bs32x16 as i8, Bs32x8 as i8, Bs16x32 as i8],
            [Bs16x64 as i8, Bs8x64 as i8, Bs4x64 as i8, Bs16x32 as i8],
        ],
        ctx: [3, 4],
    };
    t[Bs32x32 as usize] = PartitionConstants {
        part: [
            [Bs32x16 as i8, Bs32x8 as i8, Bs32x4 as i8, Bs16x16 as i8],
            [Bs16x32 as i8, Bs8x32 as i8, Bs4x32 as i8, Bs16x16 as i8],
        ],
        ctx: [2, 3],
    };
    t[Bs32x16 as usize] = PartitionConstants {
        part: [
            [Bs32x8 as i8, Bs32x4 as i8, I, Bs16x8 as i8],
            [Bs16x16 as i8, Bs8x16 as i8, Bs4x16 as i8, Bs16x8 as i8],
        ],
        ctx: [2, 2],
    };
    t[Bs32x8 as usize] = PartitionConstants {
        part: [
            [Bs32x4 as i8, I, I, I],
            [Bs16x8 as i8, Bs8x8 as i8, Bs4x8 as i8, Bs16x4 as i8],
        ],
        ctx: [13, 14],
    };
    t[Bs32x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs16x64 as usize] = PartitionConstants {
        part: [
            [Bs16x32 as i8, Bs16x16 as i8, Bs16x8 as i8, Bs8x32 as i8],
            [Bs8x64 as i8, Bs4x64 as i8, I, Bs8x32 as i8],
        ],
        ctx: [14, 13],
    };
    t[Bs16x32 as usize] = PartitionConstants {
        part: [
            [Bs16x16 as i8, Bs16x8 as i8, Bs16x4 as i8, Bs8x16 as i8],
            [Bs8x32 as i8, Bs4x32 as i8, I, Bs8x16 as i8],
        ],
        ctx: [2, 1],
    };
    t[Bs16x16 as usize] = PartitionConstants {
        part: [
            [Bs16x8 as i8, Bs16x4 as i8, I, Bs8x8 as i8],
            [Bs8x16 as i8, Bs4x16 as i8, I, Bs8x8 as i8],
        ],
        ctx: [1, 0],
    };
    t[Bs16x8 as usize] = PartitionConstants {
        part: [
            [Bs16x4 as i8, I, I, I],
            [Bs8x8 as i8, Bs4x8 as i8, I, Bs8x4 as i8],
        ],
        ctx: [1, 2],
    };
    t[Bs16x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs8x4 as i8, I, I, I]],
        ctx: [11, I],
    };
    t[Bs8x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs8x32 as usize] = PartitionConstants {
        part: [
            [Bs8x16 as i8, Bs8x8 as i8, Bs8x4 as i8, Bs4x16 as i8],
            [Bs4x32 as i8, I, I, I],
        ],
        ctx: [12, 13],
    };
    t[Bs8x16 as usize] = PartitionConstants {
        part: [
            [Bs8x8 as i8, Bs8x4 as i8, I, Bs4x8 as i8],
            [Bs4x16 as i8, I, I, I],
        ],
        ctx: [1, 1],
    };
    t[Bs8x8 as usize] = PartitionConstants {
        part: [[Bs8x4 as i8, I, I, I], [Bs4x8 as i8, I, I, I]],
        ctx: [0, 0],
    };
    t[Bs8x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [Bs4x4 as i8, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x64 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x32 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x16 as usize] = PartitionConstants {
        part: [[Bs4x8 as i8, I, I, I], [I, I, I, I]],
        ctx: [10, I],
    };
    t[Bs4x8 as usize] = PartitionConstants {
        part: [[Bs4x4 as i8, I, I, I], [I, I, I, I]],
        ctx: [0, I],
    };
    t[Bs4x4 as usize] = PartitionConstants {
        part: [[I, I, I, I], [I, I, I, I]],
        ctx: [I, I],
    };
    t
};

// indexed by inter_mode - CompInterPredMode::NearMvNewMv

pub(crate) fn read_wedge_idx<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
) -> i8 {
    let quad = msac.decode_symbol_adapt_n_padded::<3, 4>(cdf_m.wedge_quad()) as usize;
    let angle =
        5 * quad + msac.decode_symbol_adapt_n_padded::<4, 8>(cdf_m.wedge_angle(quad)) as usize;
    let dist = if (angle.wrapping_sub(1)) >= 9 || angle == 5 {
        1 + msac.decode_symbol_adapt_n_padded::<2, 4>(cdf_m.wedge_dist2()) as usize
    } else {
        msac.decode_symbol_adapt_n_padded::<3, 4>(cdf_m.wedge_dist()) as usize
    };
    WEDGE_ANGLE_DIST2IDX[angle][dist]
}

pub(crate) fn decode_4way<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    r: i32,
    cdf: &mut [u16],
    n_bits: i32,
) -> i32 {
    debug_assert!(n_bits >= 4);
    let bin = msac.decode_symbol_adapt(cdf, 3) as i32;
    let rem =
        msac.decode_bools_bypass((n_bits + bin + if bin == 0 { 1 } else { 0 } - 4) as u32) as i32;
    let v = (if bin != 0 { 1 << (n_bits + bin - 4) } else { 0 }) + rem;
    let n = 1 << n_bits;
    if r * 2 <= n {
        inv_recenter(r as u32, v as u32) as i32
    } else {
        n - 1 - inv_recenter((n - 1 - r) as u32, v as u32) as i32
    }
}

pub(crate) fn read_mv_residual<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_mv: &mut CdfMvContext,
    shell_tip: &mut [u16],
    mv_prec: i32,
) -> Mv {
    let n_syms = 9 + mv_prec;
    let h_syms = n_syms >> 1;

    let mut sh_class;
    if msac.decode_bool_adapt(cdf_mv.shell_set()) != 0 {
        let h_syms2 = n_syms - h_syms;
        sh_class = h_syms
            + 1
            + msac.decode_symbol_adapt_padded::<8>(
                cdf_mv.shell_upper(mv_prec as usize),
                imin(h_syms2, 7) as usize,
            ) as i32;
        if mv_prec + sh_class == 21 {
            sh_class += msac.decode_bool_adapt(shell_tip) as i32;
        }
    } else {
        sh_class = msac
            .decode_symbol_adapt_padded::<8>(cdf_mv.shell_lower(mv_prec as usize), h_syms as usize)
            as i32;
    }

    let mut sh_index;
    if sh_class < 2 {
        sh_index = msac.decode_bool_adapt(cdf_mv.shell_offset_low(sh_class as usize)) as i32;
    } else if sh_class == 2 {
        sh_index = msac.decode_bool_adapt(cdf_mv.shell_offset_cl2()) as i32;
        if sh_index != 0 {
            sh_index += msac.decode_bool_bypass() as i32;
            if sh_index == 2 {
                sh_index += msac.decode_bool_bypass() as i32;
            }
        }
    } else {
        sh_index = 0;
        let mut m = 1i32;
        for i in 0..sh_class {
            sh_index |= m * msac.decode_bool_adapt(cdf_mv.shell_offset_hi(i as usize)) as i32;
            m <<= 1;
        }
    }

    if sh_class != 0 {
        sh_index += 1 << sh_class;
    }
    if sh_index == 0 {
        return Mv::from_bits(0);
    }

    let mut pair_index = 0i32;
    if sh_index >= 2 {
        pair_index = msac.decode_bool_adapt(cdf_mv.col_component(0)) as i32;
        if pair_index != 0 && sh_index >= 4 {
            pair_index += msac.decode_bool_adapt(cdf_mv.col_component(1)) as i32;
            if pair_index == 2 && sh_index >= 6 {
                pair_index += msac.decode_uniform((sh_index as u32 >> 1) - 1) as i32;
            }
        }
    }

    let sh = 6 - mv_prec;
    if pair_index * 2 == sh_index {
        let v = (sh_index >> 1) << sh;
        Mv {
            c: MvXY { y: v, x: v },
        }
    } else {
        let b = msac.decode_bool_adapt(cdf_mv.col_index(imin(sh_class, 3) as usize));
        if b != 0 {
            Mv {
                c: MvXY {
                    y: pair_index << sh,
                    x: (sh_index - pair_index) << sh,
                },
            }
        } else {
            Mv {
                c: MvXY {
                    x: pair_index << sh,
                    y: (sh_index - pair_index) << sh,
                },
            }
        }
    }
}

pub(crate) fn read_mv_full<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_mv: &mut CdfMvContext,
    mv_prec: i32,
) -> Mv {
    let mut shell_tip = [cdf_mv.data[114], cdf_mv.data[115]];
    let mv = read_mv_residual(msac, cdf_mv, &mut shell_tip, mv_prec);
    cdf_mv.data[114] = shell_tip[0];
    cdf_mv.data[115] = shell_tip[1];
    mv
}

pub(crate) fn read_amvd<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
) -> Mv {
    let joint = msac.decode_symbol_adapt_n_padded::<3, 4>(cdf_m.amvd_joint()) as i32;
    if joint == 0 {
        return Mv::default();
    }
    let y = if joint & 2 != 0 {
        let s = msac.decode_symbol_adapt_n_padded::<7, 8>(cdf_m.amvd_index(0)) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    } else {
        0
    };
    let x = if joint & 1 != 0 {
        let s = msac.decode_symbol_adapt_n_padded::<7, 8>(cdf_m.amvd_index(1)) as i32;
        if s < 3 { 2 + s * 2 } else { 1 << s }
    } else {
        0
    };
    Mv::from_xy(y, x)
}

#[inline]
fn read_pal_direction(sz: &[i32; 4], read_bit: impl FnOnce() -> u32) -> u32 {
    if imax(sz[2], sz[3]) < 64 {
        read_bit()
    } else {
        0
    }
}

pub(crate) fn read_pal_indices<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    pal_out: &mut [u8],
    scratch: &mut [u8],
    pal_sz: i32,
    sz: &[i32; 4],
) -> i32 {
    let dir = read_pal_direction(sz, || msac.decode_bool_bypass());
    let strides: [isize; 2] = if dir != 0 {
        [1, sz[2] as isize]
    } else {
        [sz[2] as isize, 1]
    };

    let lim1 = sz[dir as usize ^ 1] as usize;
    let lim2 = sz[dir as usize] as usize;
    let pal_cdf_base = (pal_sz - 2) as usize;
    let nsym = (pal_sz - 1) as usize;

    let mut copy = msac.decode_symbol_adapt_n_padded::<2, 4>(cdf_m.pal_idx_identity(3)) as i32;
    if copy == 2 {
        return -1;
    }
    let mut prev_v = msac.decode_uniform(pal_sz as u32) as i32;
    scratch[0] = prev_v as u8;
    if copy == 1 {
        for m in 1..lim2 {
            scratch[(m as isize * strides[1]) as usize] = prev_v as u8;
        }
    } else {
        let mut prev_h = prev_v;
        for m in 1..lim2 {
            let v =
                msac.decode_symbol_adapt_padded::<8>(cdf_m.pal_idx(pal_cdf_base, 0), nsym) as i32;
            prev_h = if v == 0 {
                prev_h
            } else {
                v - (v <= prev_h) as i32
            };
            scratch[(m as isize * strides[1]) as usize] = prev_h as u8;
        }
    }

    let mut off: isize = strides[0];
    for _n in 1..lim1 {
        copy =
            msac.decode_symbol_adapt_n_padded::<2, 4>(cdf_m.pal_idx_identity(copy as usize)) as i32;
        if copy == 2 {
            for m in 0..lim2 {
                let dst = (off + m as isize * strides[1]) as usize;
                let src = (off - strides[0] + m as isize * strides[1]) as usize;
                scratch[dst] = scratch[src];
            }
        } else {
            let v =
                msac.decode_symbol_adapt_padded::<8>(cdf_m.pal_idx(pal_cdf_base, 0), nsym) as i32;
            let next_v = if v == 0 {
                prev_v
            } else {
                v - (v <= prev_v) as i32
            };
            scratch[off as usize] = next_v as u8;

            if copy == 1 {
                for m in 1..lim2 {
                    scratch[(off + m as isize * strides[1]) as usize] = next_v as u8;
                }
            } else {
                let mut prev_tl = prev_v;
                let mut prev_l = next_v;
                for m in 1..lim2 {
                    let prev_t =
                        scratch[(off - strides[0] + m as isize * strides[1]) as usize] as i32;
                    let ctx = if prev_t == prev_l {
                        3 + (prev_tl == prev_l) as usize
                    } else {
                        1 + (prev_t == prev_tl || prev_l == prev_tl) as usize
                    };
                    let v = msac
                        .decode_symbol_adapt_padded::<8>(cdf_m.pal_idx(pal_cdf_base, ctx), nsym)
                        as i32;
                    let p = match ctx {
                        1 => match v {
                            0 | 1 => {
                                if v == dir as i32 {
                                    prev_l
                                } else {
                                    prev_t
                                }
                            }
                            2 => prev_tl,
                            _ => {
                                let s1 = (prev_l < prev_t) as i32;
                                let s2 = (prev_l < prev_tl) as i32;
                                let s3 = (prev_t < prev_tl) as i32;
                                v - (v <= prev_l + s1 + s2) as i32
                                    - (v <= prev_t + s3 + 1 - s1) as i32
                                    - (v <= prev_tl + 1 - s2 + 1 - s3) as i32
                            }
                        },
                        2 => {
                            let prev_l_or_t = prev_l + prev_t - prev_tl;
                            match v {
                                0 => prev_tl,
                                1 => prev_l_or_t,
                                _ => {
                                    let s = (prev_l_or_t < prev_tl) as i32;
                                    v - (v <= prev_l_or_t + s) as i32
                                        - (v <= prev_tl + 1 - s) as i32
                                }
                            }
                        }
                        3 => match v {
                            0 => prev_l,
                            1 => prev_tl,
                            _ => {
                                let s = (prev_l < prev_tl) as i32;
                                v - (v <= prev_l + s) as i32 - (v <= prev_tl + 1 - s) as i32
                            }
                        },
                        4 => {
                            if v == 0 {
                                prev_l
                            } else {
                                v - (v <= prev_l) as i32
                            }
                        }
                        _ => unreachable!(),
                    };
                    scratch[(off + m as isize * strides[1]) as usize] = p as u8;
                    prev_l = p;
                    prev_tl = prev_t;
                }
            }
            prev_v = next_v;
        }
        off += strides[0];
    }

    pal_idx_finish(
        pal_out,
        scratch,
        sz[2] as usize,
        sz[3] as usize,
        sz[0] as usize,
        sz[1] as usize,
    );
    0
}

/// Read a luma palette color list from the bitstream.
///
/// reused-cache mask over the above/left neighbour palettes, then any new entries,
/// and merges them into a sorted palette in `pal` (8 entries). Returns `pal_sz`.
///
/// - `a_pal`/`l_pal`: the above / left palette caches (`t->al_pal[0][bx4]` /
///   `[1][by4]`), 8 entries each.
/// - `a_cache`/`l_cache`: number of valid cache entries above / left
///   (`t->a->pal_sz[bx4]` gated by `by4 & 15`, and `t->l.pal_sz[by4]`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_pal_plane<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    pal: &mut [u16; 8],
    a_pal: &[u16; 8],
    l_pal: &[u16; 8],
    a_cache: i32,
    l_cache: i32,
    bpc: u32,
) -> u8 {
    let pal_sz = msac.decode_symbol_adapt_n_padded::<6, 8>(cdf_m.pal_sz()) as i32 + 2;

    // find cached entries (but don't load them yet)
    let n_cache = l_cache + a_cache;
    let mut n_used_cache = 0i32;
    let mut cache_reuse_mask: u32 = 0;
    let mut off = 0i32;
    {
        let mut n = imin(n_cache, pal_sz);
        while n != 0 {
            let m = msac.decode_bools_bypass(n as u32);
            cache_reuse_mask <<= n;
            cache_reuse_mask |= m;
            n_used_cache += m.count_ones() as i32;
            off += n;
            n = imin(n_cache - off, pal_sz - n_used_cache);
        }
    }

    let mut cache = [0u16; 8];
    if n_used_cache != 0 {
        // `select`: directly copy the selected cache entries from `dir` into cache[].
        let select = |dir: &[u16; 8], cache: &mut [u16; 8]| {
            let mut mask = cache_reuse_mask << (32 - off);
            let mut i = 0usize;
            let mut n = 0u32;
            loop {
                let n_zero = mask.leading_zeros();
                cache[i] = dir[(n + n_zero) as usize];
                i += 1;
                n += n_zero + 1;
                mask <<= n_zero + 1;
                if mask == 0 {
                    break;
                }
            }
        };
        if l_cache == 0 {
            select(a_pal, &mut cache);
        } else if a_cache == 0 {
            select(l_pal, &mut cache);
        } else {
            // sort selected cache entries from a & l into cache[]
            let min_n = imin(a_cache, l_cache);
            let mask0 = cache_reuse_mask << (32 - off);
            let rem_mask = (mask0 << (min_n * 2)) >> min_n;
            let mut shared_mask = mask0.wrapping_sub(rem_mask >> min_n);
            shared_mask = (shared_mask & 0xaaaa_0000) | ((shared_mask & 0x5555_0000) >> 15);
            shared_mask |= shared_mask << 1;
            shared_mask &= 0xcccc_cccc;
            shared_mask |= shared_mask << 2;
            shared_mask &= 0xf0f0_f0f0;
            shared_mask |= shared_mask << 4;
            let a_gt_l = (a_cache > l_cache) as u32;
            let mut a_mask =
                (shared_mask & 0xff00_0000).wrapping_add(a_gt_l.wrapping_mul(rem_mask));
            let mut l_mask =
                ((shared_mask & 0xff00) << 16).wrapping_add((1 - a_gt_l).wrapping_mul(rem_mask));

            let mut i = 0usize;
            let mut a_n = 0u32;
            let mut l_n = 0u32;
            if a_mask != 0 && l_mask != 0 {
                a_n += a_mask.leading_zeros();
                a_mask <<= a_mask.leading_zeros();
                l_n += l_mask.leading_zeros();
                l_mask <<= l_mask.leading_zeros();
                loop {
                    if a_pal[a_n as usize] < l_pal[l_n as usize] {
                        cache[i] = a_pal[a_n as usize];
                        i += 1;
                        a_mask <<= 1;
                        if a_mask == 0 {
                            break;
                        }
                        let nz = a_mask.leading_zeros();
                        a_n += 1 + nz;
                        a_mask <<= nz;
                    } else {
                        cache[i] = l_pal[l_n as usize];
                        i += 1;
                        l_mask <<= 1;
                        if l_mask == 0 {
                            break;
                        }
                        let nz = l_mask.leading_zeros();
                        l_n += 1 + nz;
                        l_mask <<= nz;
                    }
                }
            }
            if a_mask != 0 {
                a_n += a_mask.leading_zeros();
                a_mask <<= a_mask.leading_zeros();
                loop {
                    cache[i] = a_pal[a_n as usize];
                    i += 1;
                    a_mask <<= 1;
                    if a_mask == 0 {
                        break;
                    }
                    let nz = a_mask.leading_zeros();
                    a_n += 1 + nz;
                    a_mask <<= nz;
                }
            } else {
                l_n += l_mask.leading_zeros();
                l_mask <<= l_mask.leading_zeros();
                loop {
                    cache[i] = l_pal[l_n as usize];
                    i += 1;
                    l_mask <<= 1;
                    if l_mask == 0 {
                        break;
                    }
                    let nz = l_mask.leading_zeros();
                    l_n += 1 + nz;
                    l_mask <<= nz;
                }
            }
        }
    }

    // parse new entries
    if n_used_cache < pal_sz {
        let mut i = n_used_cache as usize;
        let mut prev = msac.decode_bools_bypass(bpc) as i32;
        pal[i] = prev as u16;
        i += 1;

        if (i as i32) < pal_sz {
            let mut bits = bpc as i32 - 3 + msac.decode_bools_bypass(2) as i32;
            let max = (1i32 << bpc) - 1;
            loop {
                let delta = msac.decode_bools_bypass(bits as u32) as i32;
                prev = imin(prev + delta + 1, max);
                pal[i] = prev as u16;
                i += 1;
                if prev + 1 >= max {
                    while (i as i32) < pal_sz {
                        pal[i] = max as u16;
                        i += 1;
                    }
                    break;
                }
                bits = imin(bits, 1 + crate::intops::ulog2((max - prev - 1) as u32));
                if (i as i32) >= pal_sz {
                    break;
                }
            }
        }

        // merge selected cache & new entries into pal while sorting cache
        if n_used_cache != 0 {
            let mut n = 0i32;
            let mut m = n_used_cache;
            for k in 0..pal_sz as usize {
                if n < n_used_cache && (m >= pal_sz || cache[n as usize] <= pal[m as usize]) {
                    pal[k] = cache[n as usize];
                    n += 1;
                } else {
                    pal[k] = pal[m as usize];
                    m += 1;
                }
            }
        }
    } else {
        pal[..pal_sz as usize].copy_from_slice(&cache[..pal_sz as usize]);
    }

    pal_sz as u8
}

pub(crate) fn read_tx_part<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    b: &mut crate::levels::Av2Block,
    bs: BlockSize,
    lossless: bool,
    txfm_switchable: bool,
) {
    let bs_idx = bs as usize;
    let b_dim = &BLOCK_DIMENSIONS[bs_idx];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;

    b.tx_part = TxPartition::None as u8;
    if lossless {
        b.tx_size_ll = 0;
        if bs != BlockSize::Bs4x4
            && (if b.is_intra != 0 && b.intrabc == 0 {
                b.fsc != 0
            } else {
                b.skip_txfm == 0
            })
        {
            let szctx = SIZE_GROUP[bs_idx] as usize;
            let inter = (b.is_intra == 0 || b.intrabc != 0) as usize;
            b.tx_size_ll = msac.decode_bool_adapt(cdf_m.txsz_lossless(szctx, inter)) as u8;
        }
    } else if b.skip_txfm == 0 && txfm_switchable && bs != BlockSize::Bs4x4 && imax(bw4, bh4) <= 16
    {
        let inter = (b.is_intra == 0 || b.intrabc != 0) as usize;
        let szctx = TX_PART_GROUP[bs_idx] as usize;
        let is_split = msac.decode_bool_adapt(cdf_m.tx_split(b.fsc as usize, inter, szctx));
        if is_split != 0 {
            if imin(bw4, bh4) >= 2 {
                let ctx = TX_TYPE_GROUP_VH[bs_idx] as usize;
                b.tx_part = 1 + msac.decode_symbol_adapt_n_padded::<6, 8>(cdf_m.tx_part_2d(
                    b.fsc as usize,
                    inter,
                    ctx,
                )) as u8;
            } else if imax(bw4, bh4) >= 4 {
                let ctx = (bw4 >= 4) as usize;
                let tx_part_4way =
                    msac.decode_bool_adapt(cdf_m.tx_part_1d(b.fsc as usize, inter, ctx));
                b.tx_part = TxPartition::H as u8 + ctx as u8 + tx_part_4way as u8 * 2;
            } else {
                debug_assert!(bs == BlockSize::Bs4x8 || bs == BlockSize::Bs8x4);
                b.tx_part = if bs == BlockSize::Bs4x8 {
                    TxPartition::H as u8
                } else {
                    TxPartition::V as u8
                };
            }
        }
    }

    // For valid streams the CDF only ever selects a tx partition that is legal
    // invalid combinations a corrupted stream can produce. Guard here so every
    // downstream `TX_PART_TBL[bs][tx_part]` (an i8 with a -1 sentinel) lookup is
    // safe: fall back to the always-valid TX_PARTITION_NONE. No-op for valid
    // input, where the selected partition is never -1.
    if crate::tables::TX_PART_TBL[bs_idx][b.tx_part as usize] < 0 {
        b.tx_part = TxPartition::None as u8;
    }
}

pub(crate) fn read_restoration_info<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    bank: &mut NsWienerBank,
    lr: &mut Av2RestorationUnit,
    p: usize,
    frame_type: RestorationType,
    ns_plane: &NSWienerPlane,
) {
    let is_uv = (p != 0) as usize;

    if frame_type == RestorationType::Switchable {
        debug_assert!(p == 0);
        if msac.decode_bool_adapt(cdf_m.rst_switchable(0)) != 0 {
            lr.restoration_type = RestorationType::None as u8;
        } else {
            let t = msac.decode_bool_adapt(cdf_m.rst_switchable(1));
            lr.restoration_type = if t != 0 {
                RestorationType::PcWiener as u8
            } else {
                RestorationType::NsWiener as u8
            };
        }
    } else {
        // AVM disables PC-Wiener for chroma, so non-switchable chroma LR can
        // only signal NS-Wiener or None.
        debug_assert!(p == 0 || frame_type == RestorationType::NsWiener);
        let cdf = if frame_type == RestorationType::NsWiener {
            cdf_m.rst_ns_wiener()
        } else {
            cdf_m.rst_pc_wiener()
        };
        let t = msac.decode_bool_adapt(cdf);
        lr.restoration_type = if t != 0 {
            frame_type as u8
        } else {
            RestorationType::None as u8
        };
    }

    if lr.restoration_type == RestorationType::NsWiener as u8 && ns_plane.frame_filters_on == 0 {
        let n_classes = ns_plane.num_classes as usize;
        let mut exact_match_mask = 0u32;
        let mut bank_refs = [0u8; 16];

        for n in 0..n_classes {
            let exact_match = msac.decode_bool_bypass();
            let bank_size = bank.bank_size[n] as i32;
            let mut r = 0i32;
            while r < bank_size - 1 {
                if msac.decode_bool_bypass() != 0 {
                    break;
                }
                r += 1;
            }
            let r_idx = ((bank.bank_idx[n] as i32 - r) & 3) as u8;
            exact_match_mask |= (1 << n) * exact_match;
            bank_refs[n] = r_idx;
        }

        let masks: &[u32] = if is_uv != 0 {
            &SUBSET_MASKS_UV
        } else {
            &SUBSET_MASKS_Y
        };
        let cf_range: &[[i8; 2]] = if is_uv != 0 {
            &NS_WIENER_COEF_RANGE_UV
        } else {
            &NS_WIENER_COEF_RANGE_Y
        };
        let n_coefs = 16 + is_uv * 2;

        for n in 0..n_classes {
            let r = bank_refs[n] as usize;
            let exact = (exact_match_mask >> n) & 1 != 0;

            if exact {
                lr.ns_filter[n][..n_coefs].copy_from_slice(&bank.filter[r][n][..n_coefs]);
                if bank.bank_size[n] == 0 {
                    bank.bank_size[n] = 1;
                }
                continue;
            }

            lr.ns_filter[n][..n_coefs].fill(0);
            let mut s = 0usize;
            while s < 3 - is_uv {
                if msac.decode_bool_adapt(cdf_m.wiener_ns_len(is_uv)) == 0 {
                    break;
                }
                s += 1;
            }
            let mask = masks[s];
            let asym = is_uv != 0 && s != 0 && msac.decode_bool_adapt(cdf_m.wiener_ns_sym()) != 0;

            let ref_filter = &bank.filter[r][n];
            let mut i = 0usize;
            let mut m = mask;
            while i < n_coefs {
                if m & 1 == 0 {
                    i += 1;
                    m >>= 1;
                    continue;
                }
                lr.ns_filter[n][i] = (decode_4way(
                    msac,
                    ref_filter[i] as i32 - cf_range[i][1] as i32,
                    cdf_m.wiener_ns_cf(),
                    cf_range[i][0] as i32,
                ) + cf_range[i][1] as i32) as i8;
                if asym && i >= 6 {
                    lr.ns_filter[n][i + 1] = lr.ns_filter[n][i];
                    i += 1;
                    m >>= 1;
                }
                i += 1;
                m >>= 1;
            }

            let bidx = ((1 + bank.bank_idx[n]) & 3) as usize;
            bank.bank_idx[n] = bidx as u8;
            bank.filter[bidx][n][..n_coefs].copy_from_slice(&lr.ns_filter[n][..n_coefs]);
            if bank.bank_size[n] < 4 {
                bank.bank_size[n] += 1;
            }
        }
    }
}

pub(crate) fn derive_warpmv(
    rt: &refmvs::Tile,
    bx: i32,
    by: i32,
    have_top: bool,
    have_left: bool,
    bw4: i32,
    bh4: i32,
    w4: i32,
    h4: i32,
    ref_idx: i8,
    mv: Mv,
    wmp: &mut WarpedMotionParams,
    sb_step: i32,
    col_end: i32,
) {
    let mut pts = [[[0i32; 2]; 2]; 8];
    let mut np = 0usize;

    macro_rules! add_sample {
        ($dx:expr, $dy:expr, $sx:expr, $sy:expr, $rp:expr) => {{
            let rp: &refmvs::Block = $rp;
            let bd = &BLOCK_DIMENSIONS[rp.bs as usize];
            let rmv = if rp.mf & 2 != 0 { &rp.lmv } else { &rp.mv };
            for n in 0..2usize {
                if rp.reference.ref_at(n) != ref_idx {
                    continue;
                }
                pts[np][0][0] = 16 * (2 * ($dx as i32) + ($sx as i32) * bd[0] as i32) - 8;
                pts[np][0][1] = 16 * (2 * ($dy as i32) + ($sy as i32) * bd[1] as i32) - 8;
                pts[np][1][0] = pts[np][0][0] + rmv[n].x();
                pts[np][1][1] = pts[np][0][1] + rmv[n].y();
                np += 1;
                if np == 8 {
                    break;
                }
            }
        }};
    }

    debug_assert!(bw4 > 1);
    let mut have_topleft = false;
    let mut have_topright = false;
    let is_not_sb_boundary = (by & (sb_step - 1)) != 0;
    let mut init_odd = 0i32;

    if have_top {
        if is_not_sb_boundary {
            let ra_base = ((by - 1) & 63) as usize * 128;
            let r2_x = (bx & 127) as usize;
            let mut off = -(rt.r[ra_base + r2_x].ox4 as i32);
            have_topleft = off == 0;
            while off < w4 && np < 8 {
                let idx = ra_base + ((bx + off) & 127) as usize;
                add_sample!(off, 0, 1, -1, &rt.r[idx]);
                off += BLOCK_DIMENSIONS[rt.r[idx].bs as usize][0] as i32;
            }
            have_topright = off <= bw4;
        } else {
            let ra_off = rt.ra_off;
            let r2_idx = ra_off + (bx >> 1) as usize;
            init_odd = bx & 1;
            have_topleft = true;
            let mut off = if BLOCK_DIMENSIONS[rt.ra[r2_idx].bs as usize][0] as i32
                <= rt.ra[r2_idx].ox4 as i32 + init_odd
            {
                1
            } else {
                0
            };
            let tr_ext = (bx + bw4) & (sb_step - 1) != 0
                && (rt.ra[ra_off + ((bx + bw4) >> 1) as usize].ox4 != 0 || init_odd != 0);
            let tr_ext_i = tr_ext as i32;
            while off < w4 + tr_ext_i && np < 8 {
                let off8 = ra_off + ((bx + off) >> 1) as usize;
                let odd = (bx + off) & 1;
                let ioff = off - rt.ra[off8].ox4 as i32 - odd;
                add_sample!(ioff, 0, 1, -1, &rt.ra[off8]);
                off = ioff + BLOCK_DIMENSIONS[rt.ra[off8].bs as usize][0] as i32 + 1;
            }
            have_topright = true;
        }

        have_topright = have_topright
            && bw4 <= 16
            && bx + bw4 + ((!is_not_sb_boundary) as i32) < col_end
            && (!is_not_sb_boundary
                || ((bx + bw4) & (sb_step - 1) != 0
                    && rt.r[((by - 1) & 63) as usize * 128 + ((bx + bw4) & 127) as usize].mv[0]
                        .y()
                        != INVALID_MV));
    }

    if np < 8 && have_left {
        let left_x = ((bx - 1) & 127) as usize;
        let r_base = (by & 63) as usize * 128;
        let mut off = -(rt.r[r_base + left_x].oy4 as i32);
        have_topleft = have_topleft && off == 0;
        loop {
            let row = ((by & 63) as isize + off as isize) as usize;
            let idx = row * 128 + left_x;
            add_sample!(0, off, -1, 1, &rt.r[idx]);
            off += BLOCK_DIMENSIONS[rt.r[idx].bs as usize][1] as i32;
            if off >= h4 || np >= 8 {
                break;
            }
        }
    } else {
        have_topleft = false;
    }

    if is_not_sb_boundary {
        let ra_base = ((by - 1) & 63) as usize * 128;
        if np < 8 && have_topleft {
            add_sample!(0, 0, -1, -1, &rt.r[ra_base + ((bx - 1) & 127) as usize]);
        }
        if np < 8 && have_topright {
            add_sample!(bw4, 0, 1, -1, &rt.r[ra_base + ((bx + bw4) & 127) as usize]);
        }
    } else {
        if np < 8 && have_topleft {
            let r2 = if bx & (sb_step - 1) != 0 {
                &rt.ra[rt.ra_off + ((bx - 1) >> 1) as usize]
            } else {
                &rt.ra_tl
            };
            if BLOCK_DIMENSIONS[r2.bs as usize][0] as i32 + init_odd == r2.ox4 as i32 + 2 {
                add_sample!(0, 0, -1, -1, r2);
            }
        }
        if np < 8 && have_topright {
            let r2 = &rt.ra[rt.ra_off + ((bx + bw4 + 1) >> 1) as usize];
            if r2.ox4 as i32 == init_odd {
                add_sample!(bw4, 0, 1, -1, r2);
            }
        }
    }

    debug_assert!(np > 0 && np <= 8);

    if find_affine_int(&pts[..np], np, bw4, bh4, mv.xy(), wmp, bx, by) == 0
        && get_shear_params(wmp) == 0
    {
        wmp.wm_type = warp_type(&wmp.matrix);
    } else {
        wmp.wm_type = WarpedMotionType::Invalid;
    }
}

pub(crate) fn extend_warpmv(
    rt: &refmvs::Tile,
    bx: i32,
    by: i32,
    x_off: i32,
    y_off: i32,
    b_dim: &[u8],
    ref0: i8,
    mv0: Mv,
    wmp: &mut WarpedMotionParams,
    sb_step: i32,
    gmv_matrix: &[i32; 6],
) {
    let r = if y_off == -1 && (by & (sb_step - 1)) == 0 {
        if x_off < 0 && (bx & (sb_step - 1)) == 0 {
            &rt.ra_tl
        } else {
            &rt.ra[rt.ra_off + ((bx + x_off) >> 1) as usize]
        }
    } else {
        &rt.r[((by + y_off) & 63) as usize * 128 + ((bx + x_off) & 127) as usize]
    };
    let m = &mut wmp.matrix;

    if r.mf & 2 != 0 {
        if r.warp_type == WarpedMotionType::Invalid as i8 {
            m.copy_from_slice(&DEFAULT_WM_PARAMS.matrix);
        } else {
            m.copy_from_slice(&r.m);
        }
    } else if r.mf & 1 != 0 {
        m.copy_from_slice(gmv_matrix);
    } else {
        m[2..6].copy_from_slice(&DEFAULT_WM_PARAMS.matrix[2..6]);
        let ref_n = (r.reference.ref_at(0) != ref0) as usize;
        m[0] = r.mv[ref_n].x() * (1 << 13);
        m[1] = r.mv[ref_n].y() * (1 << 13);
    }

    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;
    let sx = bx * 4 + 2 * bw4 - 1;
    let sy = by * 4 + 2 * bh4 - 1;
    let mv0c = mv0.xy();
    let px = ((sx as i64) << 16) + mv0c.x as i64 * (1 << 13);
    let py = ((sy as i64) << 16) + mv0c.y as i64 * (1 << 13);

    if x_off >= 0 {
        debug_assert!(y_off == -1);
        let ay = by * 4 - 1;
        let sh = 1 + b_dim[3] as i32;
        let apx = m[2] as i64 * sx as i64 + m[3] as i64 * ay as i64 + m[0] as i64;
        let apy = m[4] as i64 * sx as i64 + m[5] as i64 * ay as i64 + m[1] as i64;
        let m3 = ((px - apx + bh4 as i64 - (px < apx) as i64) >> sh) as i32;
        let m5 = ((py - apy + bh4 as i64 - (py < apy) as i64) >> sh) as i32;
        m[3] = iclip((m3 + 0x20 - (m3 < 0) as i32) & !0x3f, -0x7fc0, 0x7fc0);
        m[5] = iclip((m5 + 0x20 - (m5 < 0x10000) as i32) & !0x3f, 0x8040, 0x17fc0);
    } else {
        debug_assert!(x_off == -1 || (by & (sb_step - 1)) == 0);
        let ax = bx * 4 - 1;
        let sh = 1 + b_dim[2] as i32;
        let lpx = m[2] as i64 * ax as i64 + m[3] as i64 * sy as i64 + m[0] as i64;
        let lpy = m[4] as i64 * ax as i64 + m[5] as i64 * sy as i64 + m[1] as i64;
        let m2 = ((px - lpx + bw4 as i64 - (px < lpx) as i64) >> sh) as i32;
        let m4 = ((py - lpy + bw4 as i64 - (py < lpy) as i64) >> sh) as i32;
        m[2] = iclip((m2 + 0x20 - (m2 < 0x10000) as i32) & !0x3f, 0x8040, 0x17fc0);
        m[4] = iclip((m4 + 0x20 - (m4 < 0) as i32) & !0x3f, -0x7fc0, 0x7fc0);
    }

    set_affine_mv2d(bw4, bh4, mv0c, wmp, bx, by);
    wmp.wm_type = if get_shear_params(wmp) != 0 {
        WarpedMotionType::Invalid
    } else {
        warp_type(&wmp.matrix)
    };
}

#[cfg(test)]
mod tests {
    use super::read_pal_direction;

    #[test]
    fn palette_direction_is_not_read_for_64px_blocks() {
        let sz = [64, 64, 64, 64];
        assert_eq!(read_pal_direction(&sz, || panic!("direction bit read")), 0);

        let sz = [64, 32, 64, 32];
        assert_eq!(read_pal_direction(&sz, || panic!("direction bit read")), 0);
    }

    #[test]
    fn palette_direction_is_read_for_smaller_blocks() {
        let sz = [32, 32, 32, 32];
        assert_eq!(read_pal_direction(&sz, || 1), 1);
    }
}
