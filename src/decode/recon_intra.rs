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
use super::*;
use crate::cdf::{CdfModeContext, CdfMvContext};
use crate::ctx::memset_pow2;
use crate::decode_partition::decode_partition;
use crate::env::BlockContext;

use crate::internal::Pass;
use crate::intops::{imax, imin};
use crate::levels::{Av2Block, BlockPartition, BlockSize, CFL_PRED, TxPartition};

use crate::msac::MsacReader;

use crate::pixel::Pixel;

use crate::tables::{BLOCK_DIMENSIONS, TXFM_DIMENSIONS};

#[allow(clippy::too_many_arguments)]
pub(crate) fn recon_b_intra_luma_phase<
    BD: BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    rb: &mut ReconBCtx<'_, '_, '_, BD, UPDATE_CDF, M>,
    bx: i32,
    by: i32,
    _bx4: usize,
    _by4: usize,
    _intrabc: bool,
    phase: TxPhase,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let recon = &mut *rb.recon;
    let msac = &mut *rb.msac;
    let cdf_m = &mut *rb.cdf_m;
    let a = &mut *rb.a;
    let l = &mut *rb.l;
    let b = rb.b;
    let fi = rb.fi;
    recon_b_intra_luma_geom_phase(
        &mut ReconBCtx {
            recon: &mut *recon,
            msac: &mut *msac,
            cdf_m: &mut *cdf_m,
            a: &mut *a,
            l: &mut *l,
            b,
            fi,
        },
        bx,
        by,
        b.bs as usize,
        phase,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recon_b_intra_luma_geom_phase<
    BD: BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    rb: &mut ReconBCtx<'_, '_, '_, BD, UPDATE_CDF, M>,
    bx: i32,
    by: i32,
    geom_bs: usize,
    phase: TxPhase,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let recon = &mut *rb.recon;
    let msac = &mut *rb.msac;
    let cdf_m = &mut *rb.cdf_m;
    let a = &mut *rb.a;
    let l = &mut *rb.l;
    let b = rb.b;
    let fi = rb.fi;
    let bs = geom_bs;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;

    let tp = &crate::tables::TX_PART_TBL[bs];

    // pb.col_start / pb.row_start are this block's origin (used by is_hv5).
    let pb_col_start = bx;
    let pb_row_start = by;

    if phase != TxPhase::ReadOnly && b.is_intra != 0 && b.intrabc == 0 && b.intra_data().pal_sz != 0
    {
        let bw = BLOCK_DIMENSIONS[bs][0] as usize * 4;
        let bh = BLOCK_DIMENSIONS[bs][1] as usize * 4;
        let stride = recon.frame.y_stride_px;
        let dst_off = 4 * (by as usize * stride + bx as usize);
        let pal: [BD::Pixel; 8] =
            std::array::from_fn(|i| BD::Pixel::from_i32(recon.scratch.pal[i] as i32));
        if BD::BPC == 8 {
            let dst8 = BD::Pixel::slice_as_ne_bytes_mut(&mut recon.dst_y);
            let pal8 = BD::Pixel::slice_as_ne_bytes(&pal);
            crate::ipred_dispatch::pal_pred_8bpc(
                &mut dst8[dst_off..],
                stride,
                pal8,
                &recon.scratch.pal_idx_y[..],
                bw,
                bh,
            );
        } else {
            crate::ipred::pal_pred(
                &mut recon.dst_y[dst_off..],
                stride,
                &pal,
                &recon.scratch.pal_idx_y[..],
                bw,
                bh,
            );
        }
    }

    if lossless {
        let tx = if b.tx_size_ll != 0 {
            crate::tables::MAX_TXFM_SIZE_FOR_BS[bs][3] as usize
        } else {
            0 // TX_4X4
        };
        let t_dim = &TXFM_DIMENSIONS[tx];
        let tw4 = t_dim.w as i32;
        let th4 = t_dim.h as i32;
        let h4 = imin(BLOCK_DIMENSIONS[bs][1] as i32, fi.bh - by);
        let w4 = imin(BLOCK_DIMENSIONS[bs][0] as i32, fi.bw - bx);
        let mut y = 0;
        while y < h4 {
            let mut x = 0;
            while x < w4 {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx + x,
                    by + y,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
                x += tw4;
            }
            y += th4;
        }
        return Ok(());
    }

    let tx_part = b.tx_part as usize;
    let tx = tp[tx_part] as usize;

    match TxPartition::from_raw(b.tx_part) {
        TxPartition::None => {
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
        }
        TxPartition::Split => {
            let t_dim = &TXFM_DIMENSIONS[tx];
            let tw4 = t_dim.w as i32;
            let th4 = t_dim.h as i32;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            let have_v_split = bx + tw4 < fi.bw;
            if have_v_split {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx + tw4,
                    by,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
            }
            if by + th4 >= fi.bh {
                return Ok(());
            }
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by + th4,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if have_v_split {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx + tw4,
                    by + th4,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
            }
        }
        TxPartition::H => {
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if by + th4 >= fi.bh {
                return Ok(());
            }
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by + th4,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
        }
        TxPartition::V => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if bx + tw4 >= fi.bw {
                return Ok(());
            }
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx + tw4,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
        }
        TxPartition::H4 => {
            // started if the previous one did not reach the frame's bottom edge.
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            for i in 0..4 {
                let yy = by + i * th4;
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx,
                    yy,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
                if yy + th4 >= fi.bh {
                    break;
                }
            }
        }
        TxPartition::V4 => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            for i in 0..4 {
                let xx = bx + i * tw4;
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    xx,
                    by,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
                if xx + tw4 >= fi.bw {
                    break;
                }
            }
        }
        TxPartition::H5 => {
            let tx_big = tp[TxPartition::H as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let tw4_small = t_dim_small.w as i32;
            let th4_small = t_dim_small.h as i32;
            let th4_big = TXFM_DIMENSIONS[tx_big].h as i32;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            let have_v_split = bx + tw4_small < fi.bw;
            if have_v_split {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx + tw4_small,
                    by,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
            }
            if by + th4_small >= fi.bh {
                return Ok(());
            }
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx_big,
                bx,
                by + th4_small,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if by + th4_small + th4_big >= fi.bh {
                return Ok(());
            }
            let yb = by + th4_small + th4_big;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                yb,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if have_v_split {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx + tw4_small,
                    yb,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
            }
        }
        TxPartition::V5 => {
            let tx_big = tp[TxPartition::V as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let tw4_small = t_dim_small.w as i32;
            let th4_small = t_dim_small.h as i32;
            let tw4_big = TXFM_DIMENSIONS[tx_big].w as i32;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                bx,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            let have_h_split = by + th4_small < fi.bh;
            if have_h_split {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    bx,
                    by + th4_small,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
            }
            if bx + tw4_small >= fi.bw {
                return Ok(());
            }
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx_big,
                bx + tw4_small,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if bx + tw4_small + tw4_big >= fi.bw {
                return Ok(());
            }
            let xb = bx + tw4_small + tw4_big;
            recon_b_luma_tx_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b,
                    fi,
                },
                tx,
                xb,
                by,
                pb_col_start,
                pb_row_start,
                lossless,
                phase,
            )?;
            if have_h_split {
                recon_b_luma_tx_phase(
                    &mut ReconBCtx {
                        recon: &mut *recon,
                        msac: &mut *msac,
                        cdf_m: &mut *cdf_m,
                        a: &mut *a,
                        l: &mut *l,
                        b,
                        fi,
                    },
                    tx,
                    xb,
                    by + th4_small,
                    pb_col_start,
                    pb_row_start,
                    lossless,
                    phase,
                )?;
            }
        }
    }

    Ok(())
}

/// Reconstruct the chroma planes (U=1, V=2) of an intra (non-IntraBC) block
#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TxPhase {
    Both,
    ReadOnly,
    ReconOnly,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChromaPhase {
    Both,
    ReadOnly,
    ReconOnly,
}

#[inline(always)]
pub(crate) fn chroma_phase_intersect(
    local: ChromaPhase,
    outer: ChromaPhase,
) -> Option<ChromaPhase> {
    match outer {
        ChromaPhase::Both => Some(local),
        ChromaPhase::ReadOnly => match local {
            ChromaPhase::Both | ChromaPhase::ReadOnly => Some(ChromaPhase::ReadOnly),
            ChromaPhase::ReconOnly => None,
        },
        ChromaPhase::ReconOnly => match local {
            ChromaPhase::Both | ChromaPhase::ReconOnly => Some(ChromaPhase::ReconOnly),
            ChromaPhase::ReadOnly => None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recon_b_intra_chroma_phase<
    BD: BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    rb: &mut ReconBCtx<'_, '_, '_, BD, UPDATE_CDF, M>,
    cbx: i32,
    cby: i32,
    cbs: BlockSize,
    sdp_active: bool,
    phase: ChromaPhase,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let recon = &mut *rb.recon;
    let msac = &mut *rb.msac;
    let cdf_m = &mut *rb.cdf_m;
    let a = &mut *rb.a;
    let l = &mut *rb.l;
    let b = rb.b;
    let fi = rb.fi;
    let is_intrabc = b.intrabc != 0;
    let is_intra = b.is_intra != 0 && (sdp_active || !is_intrabc);

    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;

    let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
    let cbw4 = cb_dim[0] as i32;
    let cbh4 = cb_dim[1] as i32;
    let cw4 = imin(fi.bw - cbx, cbw4);
    let ch4 = imin(fi.bh - cby, cbh4);
    let cbw4ss = ((cbw4 + ss_hor) >> ss_hor) as usize;
    let cw4ss = (cw4 + ss_hor) >> ss_hor;
    let ch4ss = (ch4 + ss_ver) >> ss_ver;
    let cbh4ss = ((cbh4 + ss_ver) >> ss_ver) as usize;

    let uvtx = if lossless {
        0usize // TX_4X4
    } else {
        let layout_idx =
            (crate::headers::PixelLayout::I444 as i32 - recon.frame.layout as i32) as usize;
        crate::tables::MAX_TXFM_SIZE_FOR_BS[cbs as u8 as usize][layout_idx] as usize
    };
    let uv_t_dim = &TXFM_DIMENSIONS[uvtx];
    let ctw4 = imin(uv_t_dim.w as i32, (fi.bw - cbx + ss_hor) >> ss_hor);
    let cth4 = imin(uv_t_dim.h as i32, (fi.bh - cby + ss_ver) >> ss_ver);
    let ctw = uv_t_dim.w as usize * 4;
    let cth = uv_t_dim.h as usize * 4;
    let txw = uv_t_dim.w as i32;
    let txh = uv_t_dim.h as i32;

    let bd = recon.bd;
    let bx4 = (cbx & 63) as usize;
    let by4 = (cby & 63) as usize;
    let cbx4 = bx4 >> ss_hor;
    let cby4 = by4 >> ss_ver;
    let ssbx = (cbx >> ss_hor) as usize;
    let ssby = (cby >> ss_ver) as usize;
    let cstride = recon.frame.uv_stride_px;
    let ystride = recon.frame.y_stride_px;
    let sbsz = fi.sb_step;
    use crate::levels::IntraPredMode;

    let orig_uv_mode = b.intra_data().uv_mode;
    let mut angle = b.intra_data().uv_angle as i32;
    let uv_mode_remapped = {
        let m_in = IntraPredMode::from_raw(orig_uv_mode.min(12));
        crate::recon::wide_angle_remap(uv_t_dim, m_in, &mut angle, 0) as u8
    };
    let uv_mode = if orig_uv_mode <= 12 {
        uv_mode_remapped
    } else {
        orig_uv_mode
    };

    let n_tu = cbw4ss * cbh4ss;
    let cf_len = n_tu * 16;
    let cf_need = cf_len * 2;
    let mut cf_uv = recon.scratch.take_chroma_cf::<BD::Coef>();
    if cf_uv.len() < cf_need {
        cf_uv.resize(cf_need, <BD::Coef as crate::pixel::Coeff>::ZERO);
    }
    let (cf_u, cf_v) = cf_uv[..cf_need].split_at_mut(cf_len);
    recon.scratch.chroma_txtp[..n_tu].fill([0u16; 2]);
    recon.scratch.chroma_eob[..n_tu].fill([-1i16; 2]);

    // Snapshot the per-4x4 luma fsc map for the lossless-chroma txtp derivation
    // sidestep the `recon` borrow inside the per-TU coef loop below.
    let luma_fsc_map: [u8; 256] = recon.scratch.luma_fsc_map;

    // IntraBC blocks with skip_txfm code no chroma coefficients: fill the ccoef
    // (For non-IntraBC intra blocks skip_txfm is always 0; the chroma-only SDP
    // tree is sdp_active so skip_txfm does not apply.)
    let chroma_skip_txfm = is_intrabc && b.skip_txfm != 0;

    let mut u_has_cf = 0i32;
    if chroma_skip_txfm {
        if phase != ChromaPhase::ReconOnly {
            for pl in 0..2 {
                let aw = imin(cw4ss, 64 - cbx4 as i32).max(0) as usize;
                let lh = imin(ch4ss, 64 - cby4 as i32).max(0) as usize;
                if aw > 0 {
                    a.ccoef[pl][cbx4..cbx4 + aw].fill(0x40);
                }
                if lh > 0 {
                    l.ccoef[pl][cby4..cby4 + lh].fill(0x40);
                }
            }
        }
    } else if phase != ChromaPhase::ReconOnly {
        for pl in 0..2 {
            let cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
            let mut y = 0;
            while y < ch4ss {
                let mut x = 0;
                while x < cw4ss {
                    let i = (y * cbw4ss as i32 + x) as usize;
                    let mut txtp: u16 = 0;
                    let mut res_ctx: u8 = 0;
                    // TU coefficient region is txw*txh*16 coefs (= ctw*cth), placed at
                    let cf_slot = &mut cf[i * 16..];
                    let tu_n = (uv_t_dim.w as usize * 4) * (uv_t_dim.h as usize * 4);
                    cf_slot[..tu_n].fill(<BD::Coef as crate::pixel::Coeff>::ZERO);

                    let dq_tbl = recon.dq_active[seg_id][1 + pl];
                    let qm_ref: Option<&[u8]> = recon.frame.qm[uvtx][1 + pl].as_deref();

                    let acoef = &a.ccoef[pl][(cbx4 + x as usize)..];
                    let lcoef = &l.ccoef[pl][(cby4 + y as usize)..];

                    let params = crate::recon::DecodeCoefParams {
                        tx: uvtx,
                        // skip/entropy ctx keys off the full chroma coding block
                        // (b.cbs), not the tx-chunk walk size, matching luma's b.bs
                        // and AVM's plane_bsize = chroma_ref_info.bsize_base.
                        bs: b.cbs as usize,
                        plane: (pl + 1) as i32,
                        intra: is_intra,
                        fsc: b.fsc != 0,
                        lossless,
                        sdp_active,
                        y_mode: 0,
                        uv_mode: uv_mode as usize,
                        seq_fsc: recon.frame.seq_fsc,
                        seq_ist: recon.frame.seq_ist,
                        seq_cctx: recon.frame.seq_cctx,
                        chroma_dctonly: false,
                        reduced_txtp_set: recon.frame.reduced_txtp_set,
                        tcq_enabled: recon.frame.tcq,
                        layout: recon.frame.layout,
                        u_has_cf,
                        cbx,
                        cby,
                        luma_fsc_map: &luma_fsc_map,
                        dq_tbl,
                        bitdepth: recon.frame.bitdepth,
                        qm: qm_ref,
                        ss_hor: ss_hor != 0,
                        ss_ver: ss_ver != 0,
                    };

                    let eob = msac.decode_coefs(
                        recon.cdf_coef,
                        cdf_m,
                        acoef,
                        lcoef,
                        &params,
                        cf_slot,
                        &mut txtp,
                        &mut res_ctx,
                        &mut recon.scratch.coef_levels,
                    );
                    if eob == i32::MIN {
                        recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
                        return Err(());
                    }
                    if pl == 0 {
                        u_has_cf = (eob >= 0) as i32;
                    }
                    recon.scratch.chroma_txtp[i][pl] = txtp;
                    recon.scratch.chroma_eob[i][pl] = eob as i16;

                    let aw = imin(ctw4, 64 - (cbx4 + x as usize) as i32).max(0) as usize;
                    let lh = imin(cth4, 64 - (cby4 + y as usize) as i32).max(0) as usize;
                    if aw > 0 {
                        a.ccoef[pl][cbx4 + x as usize..cbx4 + x as usize + aw].fill(res_ctx);
                    }
                    if lh > 0 {
                        l.ccoef[pl][cby4 + y as usize..cby4 + y as usize + lh].fill(res_ctx);
                    }
                    x += txw;
                }
                y += txh;
            }
        }
    } // end coef-read phase

    // Stash decoded coefficients for the deferred recon phase, or restore them.
    if phase == ChromaPhase::ReadOnly {
        let need = cf_need;
        let cf_off = recon.scratch.chroma_tx_cf_mut::<BD::Coef>().len();
        {
            let chroma_tx_cf = recon.scratch.chroma_tx_cf_mut::<BD::Coef>();
            chroma_tx_cf.extend_from_slice(cf_u);
            chroma_tx_cf.extend_from_slice(cf_v);
        }
        let chroma_txtp = recon.scratch.chroma_txtp;
        let chroma_eob = recon.scratch.chroma_eob;
        recon.scratch.chroma_tx.push(ChromaTxRecord {
            cbx: cbx as i16,
            cby: cby as i16,
            cbs: cbs as u8,
            sdp_active,
            n_tu: n_tu as u16,
            cf_off: cf_off as u32,
            cf_len: need as u32,
            u_has_cf,
            txtp: chroma_txtp,
            eob: chroma_eob,
        });
        recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
        return Ok(());
    } else if phase == ChromaPhase::ReconOnly {
        let rec = match recon
            .scratch
            .chroma_tx
            .get(recon.scratch.chroma_tx_rpos)
            .copied()
        {
            Some(rec) => rec,
            None => {
                recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
                return Err(());
            }
        };
        recon.scratch.chroma_tx_rpos += 1;
        debug_assert_eq!(rec.cbx as i32, cbx);
        debug_assert_eq!(rec.cby as i32, cby);
        debug_assert_eq!(rec.cbs, cbs as u8);
        debug_assert_eq!(rec.sdp_active, sdp_active);
        debug_assert_eq!(rec.n_tu as usize, n_tu);
        let cf_off = rec.cf_off as usize;
        let rec_cf_len = rec.cf_len as usize;
        debug_assert_eq!(rec_cf_len, cf_need);
        let chroma_tx_cf = recon.scratch.chroma_tx_cf::<BD::Coef>();
        cf_u.copy_from_slice(&chroma_tx_cf[cf_off..cf_off + cf_len]);
        cf_v.copy_from_slice(&chroma_tx_cf[cf_off + cf_len..cf_off + cf_need]);
        recon.scratch.chroma_txtp = rec.txtp;
        recon.scratch.chroma_eob = rec.eob;
        u_has_cf = rec.u_has_cf;
    }
    let _ = u_has_cf;

    // CfL is intra-only (`if (intra) cfl()`); IntraBC used the mc copy instead.
    if is_intra && uv_mode == CFL_PRED {
        if let Err(e) = cfl_predict_8bpc(recon, b, cbs, uvtx, cbx, cby, fi) {
            recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
            return Err(e);
        }
    }

    let col_end_ss = fi.tile_col_end >> ss_hor;
    let row_end_ss = fi.tile_row_end >> ss_ver;
    let mut y = 0;
    while y < ch4ss {
        let mut x = 0;
        while x < cw4ss {
            let i = (y * cbw4ss as i32 + x) as usize;
            let dst_off = 4 * ((ssby + y as usize) * cstride + ssbx + x as usize);

            // Intra prediction for both planes (skipped for CfL and IntraBC).
            // C gate: `if (intra && b->uv_mode != CFL_PRED)`.
            if is_intra && uv_mode != CFL_PRED {
                for pl in 0..2 {
                    let mut n_tr = 0i32;
                    if cby + (y << ss_ver) > fi.tile_row_start && (ctw as i32) < 64 {
                        let csbsz = sbsz >> ss_hor;
                        let tile_end = col_end_ss;
                        let w = imin(ctw4, tile_end - (ssbx as i32 + x) - ctw4);
                        if (cby + y) & (sbsz - 1) == 0 {
                            n_tr = w;
                        } else {
                            let end = imin((ssbx as i32 + x + csbsz) & !(csbsz - 1), tile_end);
                            let w2 = imin(ctw4, end - (ssbx as i32 + x) - ctw4);
                            if w2 == 0 {
                                n_tr = 0;
                            } else {
                                let shift = (cbx4 as i32 + x + ctw4) as u32;
                                let bits = recon.scratch.is_coded[1]
                                    [(cby4 as i32 + y - 1) as usize]
                                    >> shift;
                                let inv = 0x10000u64 | !bits;
                                n_tr = imin(inv.trailing_zeros() as i32, w2);
                            }
                        }
                    }
                    let mut n_bl = 0i32;
                    if cbx + (x << ss_hor) > fi.tile_col_start && (cth as i32) < 64 {
                        let csbsz = sbsz >> ss_ver;
                        let end = imin((ssby as i32 + y + csbsz) & !(csbsz - 1), row_end_ss);
                        let h = imin(cth4, end - (ssby as i32 + y) - cth4);
                        if (cbx + x) & (sbsz - 1) == 0 || h <= 0 {
                            n_bl = h;
                        } else {
                            let mask = 1u64 << ((cbx4 as i32 + x - 1) as u32);
                            let mut nb = 0;
                            while nb < h {
                                let row = (cby4 as i32 + y + nb + cth4) as usize;
                                if row >= 64 || (recon.scratch.is_coded[1][row] & mask) == 0 {
                                    break;
                                }
                                nb += 1;
                            }
                            n_bl = nb;
                        }
                    }

                    let mut apply_ibp = recon.frame.seq_ibp && uvtx != 0;
                    let sm_top = b.intra_data().is_sm[1].a;
                    let sm_left = b.intra_data().is_sm[1].l;
                    let is_sm_flag = if apply_ibp {
                        (sm_top * crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG)
                            | (sm_left * crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
                    } else {
                        (sm_top | sm_left)
                            * (crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG
                                | crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
                    };
                    apply_ibp &= uv_mode == 0; // DC_PRED
                    let have_left = cbx + (x << ss_hor) > fi.tile_col_start;
                    let have_top = cby + (y << ss_ver) > fi.tile_row_start;
                    let intra_flags = is_sm_flag
                        | if apply_ibp {
                            crate::levels::ANGLE_IBP_FLAG
                        } else {
                            0
                        }
                        | if recon.frame.seq_intra_edge_filter {
                            crate::levels::ANGLE_USE_EDGE_FILTER_FLAG
                        } else {
                            0
                        }
                        | if have_left {
                            crate::levels::ANGLE_HAS_LEFT_FLAG
                        } else {
                            0
                        }
                        | if have_top {
                            crate::levels::ANGLE_HAS_TOP_FLAG
                        } else {
                            0
                        };
                    let pred_mode = if uv_mode == CFL_PRED { 0 } else { uv_mode };

                    let dst_plane: &mut [BD::Pixel] =
                        if pl == 0 { recon.dst_u } else { recon.dst_v };
                    let edge_o: usize = 768;
                    let m = crate::ipred_prepare::prepare_intra_edges(
                        bd,
                        ssbx as i32 + x,
                        ssby as i32 + y,
                        col_end_ss,
                        row_end_ss,
                        n_tr,
                        n_bl,
                        dst_plane,
                        dst_off,
                        cstride,
                        None,
                        pred_mode,
                        txw,
                        txh,
                        angle | intra_flags,
                        recon.edge,
                        edge_o,
                    );
                    let pred_angle = angle | intra_flags;
                    let max_w = 4 * fi.bw - 4 * (cbx + x);
                    let max_h = 4 * fi.bh - 4 * (cby + y);
                    dispatch_ipred(
                        bd,
                        m,
                        dst_plane,
                        dst_off,
                        cstride,
                        recon.edge,
                        edge_o,
                        ctw,
                        cth,
                        pred_angle,
                        max_w,
                        max_h,
                        &recon.frame.ibp_weights,
                    );
                }
            }

            let cctx_enabled = recon.frame.seq_cctx
                && (recon.frame.layout == crate::headers::PixelLayout::I420 || uv_t_dim.min < 8);
            let cctx_type = if cctx_enabled && recon.scratch.chroma_eob[i][0] >= 1 {
                (recon.scratch.chroma_txtp[i][0] >> 8) as i32
            } else {
                0
            };
            if cctx_type != 0 {
                let sz = imin(ctw as i32, 32) as usize * imin(cth as i32, 32) as usize;
                crate::itx::cctx_bd(
                    bd,
                    &mut cf_u[i * 16..],
                    &mut cf_v[i * 16..],
                    &crate::tables::CCTX_ANGLE[(cctx_type - 1) as usize],
                    sz,
                );
                let gt = (recon.scratch.chroma_eob[i][1] > recon.scratch.chroma_eob[i][0]) as usize;
                recon.scratch.chroma_eob[i][1 - gt] = recon.scratch.chroma_eob[i][gt];
                let t0 = recon.scratch.chroma_txtp[i][0] & 0xff;
                recon.scratch.chroma_txtp[i][0] = t0;
                recon.scratch.chroma_txtp[i][1] = t0;
            }

            for pl in 0..2 {
                let eob = recon.scratch.chroma_eob[i][pl];
                if eob != -1 {
                    let cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
                    let mut txtp = recon.scratch.chroma_txtp[i][pl] as u32;
                    // `lossless && b->intra && (sdp_active || !b->intrabc) && dpcm[1]`;
                    // the inter-DDT branch is `seq_hdr->inter_ddt && !b->intra`.
                    // The DDT branch keys off `!b->intra` — an IntraBC block has
                    // `b->intra == 1`, so it takes NEITHER branch. This chroma tx
                    // path only runs for intra/IntraBC blocks, so the DDT branch
                    // never fires; applying it for IntraBC corrupts the transform.
                    if lossless && is_intra && b.intra_data().dpcm[1] != 0 {
                        txtp +=
                            ((1 + (uv_mode == IntraPredMode::VertPred as u8) as u32) as u32) << 8;
                    } else if recon.frame.seq_inter_ddt && b.is_intra == 0 {
                        txtp += txtp & crate::tables::TX_DDT_MASK[uvtx] as u32;
                    }
                    let dst_plane: &mut [BD::Pixel] =
                        if pl == 0 { recon.dst_u } else { recon.dst_v };
                    crate::itx::inv_txfm_add(
                        bd,
                        dst_plane,
                        dst_off,
                        cstride,
                        &mut cf[i * 16..],
                        txtp,
                        eob as i32,
                        uvtx,
                        &mut recon.scratch.itx_tmp,
                    );
                }
            }

            let coded_w = imin(ctw4, 64 - (cbx4 as i32 + x)).max(0) as u32;
            if coded_w > 0 {
                let mask: u64 = (((1u128 << coded_w) - 1) as u64) << ((cbx4 as i32 + x) as u32);
                for yy in 0..cth4 {
                    let row = (cby4 as i32 + y + yy) as usize;
                    if row < 64 {
                        recon.scratch.is_coded[1][row] |= mask;
                    }
                }
            }
            x += txw;
        }
        y += txh;
    }

    let _ = (orig_uv_mode, ystride);
    recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
    Ok(())
}

/// Handles CFL_EXPLICIT / CFL_IMPLICIT (cfl_type < 2) and CFL_MHCCP (cfl_type==2).
#[allow(clippy::too_many_arguments)]
fn cfl_predict_8bpc<BD: BitDepth>(
    recon: &mut ReconCtx<BD>,
    b: &Av2Block,
    bs: BlockSize,
    uvtx: usize,
    cbx: i32,
    cby: i32,
    fi: &SbFrameInfo,
) -> Result<(), ()> {
    use crate::ipred::{
        CFL_HAS_LEFT, CFL_HAS_TOP, CFL_IS_TOP_SB_EDGE, CFL_MHCCP_MAX_EDGE_SAMPLES, cfl_calc_alphas,
        cfl_gen_mat, cfl_gen_y_420, cfl_mhccp_pred, cfl_pred_raw,
    };
    use crate::levels::CflMhDir;

    let bd = recon.bd;
    let ss_hor = recon.frame.ss_hor as usize;
    let ss_ver = recon.frame.ss_ver as usize;
    let ystride = recon.frame.y_stride_px;
    let cstride = recon.frame.uv_stride_px;
    let sbsz = fi.sb_step;
    let ssbx = (cbx >> ss_hor) as usize;
    let ssby = (cby >> ss_ver) as usize;
    let has_top = cby > fi.tile_row_start;
    let has_left = cbx > fi.tile_col_start;
    let is_top_sb_edge = (cby & (sbsz - 1)) == 0;
    let t_dim = &TXFM_DIMENSIONS[uvtx];
    let ctw4 = imin(t_dim.w as i32, (fi.bw - cbx + ss_hor as i32) >> ss_hor) as usize;
    let cth4 = imin(t_dim.h as i32, (fi.bh - cby + ss_ver as i32) >> ss_ver) as usize;
    let ctw = t_dim.w as usize * 4;
    let cth = t_dim.h as usize * 4;
    let filter_type = recon.frame.cfl_ds_filter_index;
    let cfl_type = b.intra_data().cfl_type as i32;
    let cfl_mh_dir_raw = b.intra_data().cfl.mh_dir();
    // Raw symbol (0/1) maps directly to the dispatch index (CENTER=0, TOP=1).
    let dir = CflMhDir::from_raw(cfl_mh_dir_raw.min(3));

    let ysrc_off = (cby as usize * ystride + cbx as usize) * 4;

    if cfl_type < 2 {
        let implicit = cfl_type == 1; // CFL_IMPLICIT
        let coff = (ssby * cstride + ssbx) * 4;
        // ytop / utop / vtop: source rows above the current block used for the CfL
        // above the SB). In single-thread / filters-off decode `prefilter_data`
        // aliases the current plane with `prefilter_data_full_frame` set, so the
        // luma SB-edge row resolves to `ysrc - ystride` (one luma row up) and is
        // downsampled with `bottom = 0` via the CFL_IS_TOP_SB_EDGE flag; the
        // in-plane fallback instead starts `1 + ss_ver` rows up with
        // `bottom = ystride`. The chroma `utop`/`vtop` offsets are `coff - cstride`
        // in both branches (single-thread alias), so only `ytop_off` differs.
        let ytop_off = if is_top_sb_edge && has_top {
            (ysrc_off as isize - ystride as isize) as usize
        } else {
            (ysrc_off as isize - ((1 + ss_ver) as isize) * ystride as isize) as usize
        };
        let utop_off = (coff as isize - cstride as isize) as usize;
        let vtop_off = utop_off;

        let cbw4 = (BLOCK_DIMENSIONS[bs as u8 as usize][0] as usize + ss_hor) >> ss_hor;
        let cbh4 = (BLOCK_DIMENSIONS[bs as u8 as usize][1] as usize + ss_ver) >> ss_ver;
        let wpad = cbw4 - ctw4;
        let hpad = cbh4 - cth4;

        let alpha = b.intra_data().cfl.alpha();
        let cfl_has_left = has_left;
        let cfl_has_top = has_top;
        let flags = (filter_type as u32)
            | if cfl_has_top { CFL_HAS_TOP as u32 } else { 0 }
            | if cfl_has_left { CFL_HAS_LEFT as u32 } else { 0 }
            | if is_top_sb_edge {
                CFL_IS_TOP_SB_EDGE
            } else {
                0
            }
            | (((alpha[0] as u32) << crate::ipred::CFL_ALPHA_U_SHIFT)
                & crate::ipred::CFL_ALPHA_U_MASK)
            | (((alpha[1] as u32) << crate::ipred::CFL_ALPHA_V_SHIFT)
                & crate::ipred::CFL_ALPHA_V_MASK);

        // In C, utop/vtop and the left references alias the same allocation as the
        // goes one step further and runs CfL into temporary chroma blocks with an
        let dst_y: &[BD::Pixel] = &*recon.dst_y;
        let (u_buf, v_buf) = (&mut *recon.dst_u, &mut *recon.dst_v);

        cfl_pred_raw(
            bd,
            dst_y,
            u_buf,
            v_buf,
            ytop_off,
            utop_off,
            vtop_off,
            ysrc_off,
            coff,
            coff,
            ystride as isize,
            cstride as isize,
            wpad,
            hpad,
            ctw,
            cth,
            flags,
            implicit,
            ss_hor,
            ss_ver,
        )?;

        return Ok(());
    }

    let mut refw = (ctw4 * 4) as i32;
    let mut refh = (cth4 * 4) as i32;
    let cbx4 = ((cbx & 63) >> ss_hor) as i32;
    let cby4 = ((cby & 63) >> ss_ver) as i32;
    if has_top {
        let csbsz = sbsz >> ss_hor as i32;
        let tile_end = fi.tile_col_end >> ss_hor as i32;
        let mut w = imax(0, imin(ctw4 as i32, tile_end - ssbx as i32 - ctw4 as i32));
        let n_tr = if is_top_sb_edge {
            w
        } else {
            let end = imin((ssbx as i32 + csbsz) & !(csbsz - 1), tile_end);
            w = imin(ctw4 as i32, end - ssbx as i32 - ctw4 as i32);
            if w == 0 {
                0
            } else {
                let bits =
                    recon.scratch.is_coded[1][(cby4 - 1) as usize] >> ((cbx4 + ctw4 as i32) as u32);
                imin((0x10000u64 | !bits).trailing_zeros() as i32, w)
            }
        };
        refw += n_tr * 4;
    }
    let mut subleft = 0;
    if has_left {
        let csbsz = sbsz >> ss_ver as i32;
        let end = imax(
            0,
            imin(
                (ssby as i32 + csbsz) & !(csbsz - 1),
                fi.tile_row_end >> ss_ver as i32,
            ),
        );
        let h = imin(cth4 as i32, end - ssby as i32 - cth4 as i32);
        let n_bl = if (cbx & (sbsz - 1)) == 0 || h <= 0 {
            h
        } else {
            let mask = 1u64 << ((cbx4 - 1) as u32);
            let mut nb = 0;
            while nb < h {
                if (recon.scratch.is_coded[1][(cby4 + nb + cth4 as i32) as usize] & mask) == 0 {
                    break;
                }
                nb += 1;
            }
            nb
        };
        refh += n_bl * 4;
        refw += 2;
        subleft = (dir != CflMhDir::Left) as i32;
    }
    if refw > (128 >> ss_hor) {
        refw = 128 >> ss_hor;
        subleft = 0;
    }
    refh = imin(refh, (128 >> ss_ver) - 2 * has_top as i32);

    let luma_top_stride = ((refw as usize) + 63) & !63;
    let edge_flags = if has_top { CFL_HAS_TOP } else { 0 }
        | if has_left { CFL_HAS_LEFT } else { 0 }
        | if is_top_sb_edge {
            CFL_IS_TOP_SB_EDGE as i32
        } else {
            0
        };

    let mut luma = [BD::Pixel::default(); crate::ipred::CFL_MHCCP_MAX_LUMA_SIZE];
    // SAFETY: luma plane is a disjoint allocation from chroma planes.
    let ysrc: &[BD::Pixel] = &*recon.dst_y;
    // and `prefilter_data_full_frame` is set, so `ytop_sb_edge` resolves to the
    // explicitly makes `cfl_gen_y` take the `top_sb_edge != NULL` branch (b=0),
    // which differs from the in-plane fallback (b=src_stride) at internal SB
    let ytop_sb_edge: Option<(&[BD::Pixel], usize)> = if is_top_sb_edge && has_top {
        Some((ysrc, ysrc_off - ystride))
    } else {
        None
    };
    cfl_gen_y_420(
        &mut luma,
        luma_top_stride,
        ysrc,
        ysrc_off,
        ytop_sb_edge,
        ystride,
        (refw - subleft) as usize,
        refh as usize,
        ctw,
        cth,
        edge_flags | dir as i32,
        filter_type,
        ss_hor,
        ss_ver,
    );
    refh += has_top as i32;

    let mut mat = [[0i32; 3]; 3];
    let mut imat = [[0u16; CFL_MHCCP_MAX_EDGE_SAMPLES]; 2];
    if has_top || has_left {
        cfl_gen_mat(
            bd,
            &mut mat,
            &mut imat,
            &luma,
            0,
            luma_top_stride,
            refw as usize,
            refh as usize,
            edge_flags,
            dir,
        );
    }

    for pl in 0..2 {
        let mut alpha = [0i32; 3];
        let chroma_off = 4 * (ssby * cstride + ssbx);
        let chroma: &mut [BD::Pixel] = if pl == 0 { recon.dst_u } else { recon.dst_v };
        if has_top || has_left {
            cfl_calc_alphas(
                bd,
                &mut alpha,
                chroma,
                chroma_off,
                None, // ctop_sb_edge (frame-top only)
                cstride,
                refw as usize,
                refh as usize,
                &mut mat,
                &imat,
                edge_flags,
            );
        } else {
            alpha[2] = 0x10000;
        }
        let n_top = if has_top {
            has_top as usize + (dir == CflMhDir::Top) as usize
        } else {
            0
        };
        let src_off = n_top * luma_top_stride;
        // the `chroma` pointer to the block), so slice the destination plane at
        // the block offset rather than the plane origin.
        cfl_mhccp_pred(
            bd,
            &mut chroma[chroma_off..],
            cstride,
            &luma,
            src_off,
            luma_top_stride,
            ctw,
            cth,
            &alpha,
            edge_flags,
            dir,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn recon_b_luma_tx_phase<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    rb: &mut ReconBCtx<'_, '_, '_, BD, UPDATE_CDF, M>,
    tx: usize,
    bx: i32,
    by: i32,
    pb_col_start: i32,
    pb_row_start: i32,
    lossless: bool,
    phase: TxPhase,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let recon = &mut *rb.recon;
    let msac = &mut *rb.msac;
    let cdf_m = &mut *rb.cdf_m;
    let a = &mut *rb.a;
    let l = &mut *rb.l;
    let b = rb.b;
    let fi = rb.fi;
    use crate::levels::IntraPredMode;

    let bd = recon.bd;
    let bx4 = (bx & 63) as usize;
    let by4 = (by & 63) as usize;
    let t_dim = &TXFM_DIMENSIONS[tx];
    let tw = t_dim.w as usize * 4;
    let th = t_dim.h as usize * 4;
    let tw4 = t_dim.w as i32;
    let th4 = t_dim.h as i32;

    let is_intrabc = b.intrabc != 0;
    // The decode-coefs / stx "intra" flag: `b->intra && (sdp_active || !b->intrabc)`
    // — here sdp_active is false in the luma path, so it is false for IntraBC.
    let is_intra = !is_intrabc;

    let intra = b.intra_data();
    let orig_y_mode = intra.y_mode;
    let mut angle = intra.y_angle as i32;

    let y_mode = if is_intra {
        // SAFETY: y_mode is a valid IntraPredMode discriminant (0..=12).
        let y_mode_remapped = {
            let m_in = IntraPredMode::from_raw(orig_y_mode.min(12));
            crate::recon::wide_angle_remap(t_dim, m_in, &mut angle, intra.mrl_index as i32) as u8
        };
        if orig_y_mode <= 12 {
            y_mode_remapped
        } else {
            orig_y_mode
        }
    } else {
        orig_y_mode
    };

    let tu_n = tw * th;
    let mut txtp: u16 = 0;
    let mut res_ctx: u8 = 0;
    // No pre-clear: every inverse-transform path clears cf[..S*S] after use
    // (itx_2d dequant cores + itx.rs WHT/DC/generic), and cf starts zeroed, so
    // cf is already zero on entry here. Verified bit-exact + via prefill poison.

    // IntraBC blocks may set skip_txfm (intra/non-IntraBC blocks force it to 0).
    // When set, no coefficients are coded: eob=-1, txtp=DCT_DCT, stx=0, and the
    // coefficient contexts are updated as skipped.
    let (mut eob, stx, mut txtp) = if phase == TxPhase::ReconOnly {
        let rec = recon
            .scratch
            .luma_tx
            .get(recon.scratch.luma_tx_rpos)
            .copied()
            .ok_or(())?;
        recon.scratch.luma_tx_rpos += 1;
        debug_assert_eq!(rec.tx as usize, tx);
        debug_assert_eq!(rec.bx as i32, bx);
        debug_assert_eq!(rec.by as i32, by);
        debug_assert_eq!(rec.pb_col_start as i32, pb_col_start);
        debug_assert_eq!(rec.pb_row_start as i32, pb_row_start);
        debug_assert_eq!(rec.lossless, lossless);
        let cf_off = rec.cf_off as usize;
        let cf_len = rec.cf_len as usize;
        if cf_len != 0 {
            recon.cf[..cf_len]
                .copy_from_slice(&recon.scratch.luma_tx_cf::<BD::Coef>()[cf_off..cf_off + cf_len]);
        }
        (rec.eob as i32, rec.stx as i32, rec.txtp as u32)
    } else if b.skip_txfm != 0 {
        res_ctx = 0x40;
        (-1i32, 0i32, crate::levels::txtp::DCT_DCT as u32)
    } else {
        let dq_seg = b.seg_id as usize;
        let dq_tbl = recon.dq_active[dq_seg][0]; // plane 0 (luma)
        let qm_ref: Option<&[u8]> = recon.frame.qm[tx][0].as_deref();

        let params = crate::recon::DecodeCoefParams {
            tx,
            bs: b.bs as usize,
            plane: 0,
            intra: is_intra,
            fsc: b.fsc != 0,
            lossless,
            sdp_active: false,
            y_mode: y_mode as usize,
            uv_mode: 0,
            seq_fsc: recon.frame.seq_fsc,
            seq_ist: recon.frame.seq_ist,
            seq_cctx: recon.frame.seq_cctx,
            chroma_dctonly: false,
            reduced_txtp_set: recon.frame.reduced_txtp_set,
            tcq_enabled: recon.frame.tcq,
            layout: recon.frame.layout,
            u_has_cf: 0,
            cbx: 0,
            cby: 0,
            luma_fsc_map: &[],
            dq_tbl,
            bitdepth: recon.frame.bitdepth,
            qm: qm_ref,
            ss_hor: recon.frame.ss_hor != 0,
            ss_ver: recon.frame.ss_ver != 0,
        };

        let eob = msac.decode_coefs(
            recon.cdf_coef,
            cdf_m,
            &a.lcoef[bx4..],
            &l.lcoef[by4..],
            &params,
            recon.cf,
            &mut txtp,
            &mut res_ctx,
            &mut recon.scratch.coef_levels,
        );
        if eob == i32::MIN {
            return Err(());
        }
        let stx = (txtp >> 8) as i32;
        (eob, stx, (txtp & 0xff) as u32)
    };

    let aw = imin(tw4, fi.bw - bx).max(0) as usize;
    let lh = imin(th4, fi.bh - by).max(0) as usize;
    if phase != TxPhase::ReconOnly {
        if aw > 0 {
            a.lcoef[bx4..bx4 + aw].fill(res_ctx);
        }
        if lh > 0 {
            l.lcoef[by4..by4 + lh].fill(res_ctx);
        }
    }

    if phase == TxPhase::ReadOnly {
        let cf_len = if eob != -1 { tu_n } else { 0 };
        let cf_off = recon.scratch.luma_tx_cf_mut::<BD::Coef>().len();
        if cf_len != 0 {
            recon
                .scratch
                .luma_tx_cf_mut::<BD::Coef>()
                .extend_from_slice(&recon.cf[..cf_len]);
            // In the normal path `inv_txfm_add` clears the consumed coefficient
            // region.  ReadOnly returns before the inverse transform, so preserve
            // the same zero-on-entry invariant for the next coefficient block.
            recon.cf[..cf_len].fill(<BD::Coef as crate::pixel::Coeff>::ZERO);
        }
        recon.scratch.luma_tx.push(LumaTxRecord {
            tx: tx as u8,
            bx: bx as i16,
            by: by as i16,
            pb_col_start: pb_col_start as i16,
            pb_row_start: pb_row_start as i16,
            eob: eob as i16,
            stx: stx as i8,
            txtp: txtp as u16,
            cf_off: cf_off as u32,
            cf_len: cf_len as u16,
            lossless,
        });
        return Ok(());
    }

    // dst origin for this tx block.
    let stride = recon.frame.y_stride_px;
    let dst_off = 4 * (by as usize * stride + bx as usize);

    // Skipped for IntraBC: the block-copy prediction was applied before the tx
    if is_intra && intra.pal_sz == 0 {
        let sbsz = fi.sb_step;
        let mrl_idx = intra.mrl_index as i32;
        let mrl_mul = intra.multi_mrl != 0 && tx != 0; // tx != TX_4X4
        let is_hv5 = (by > pb_row_start || bx > pb_col_start)
            && (b.tx_part == TxPartition::H5 as u8 || b.tx_part == TxPartition::V5 as u8);

        let mut n_tr = 0i32;
        if by > fi.tile_row_start {
            let mut w = imin(tw4, fi.tile_col_end - bx - tw4);
            if is_hv5 {
                n_tr = 0;
            } else if (by & (sbsz - 1)) == 0 {
                n_tr = w;
            } else {
                let end = imin((bx + sbsz) & !(sbsz - 1), fi.tile_col_end);
                w = imin(w, end - bx - tw4);
                if w <= 0 {
                    n_tr = 0;
                } else {
                    let xpos = ((bx4 as i32 + tw4) & 63) as u32;
                    let bits = recon.scratch.is_coded[0][by4 - 1] >> xpos;
                    let inv = 0x10000u64 | !bits;
                    n_tr = imin(inv.trailing_zeros() as i32, w);
                }
            }
        }

        let mut n_bl = 0i32;
        if bx > fi.tile_col_start {
            let end = imin((by + sbsz) & !(sbsz - 1), fi.tile_row_end);
            let h = imin(th4, end - by - th4);
            // C distinguishes is_hv5 / bottom-edge as separate n_bl=0 cases
            if is_hv5 || h <= 0 {
                n_bl = 0;
            } else if (bx & (sbsz - 1)) == 0 {
                n_bl = h;
            } else {
                let mask = 1u64 << (((bx4 as i32 - 1) & 63) as u32);
                let mut y = 0;
                while y < h {
                    let row = (by4 as i32 + y + th4) as usize;
                    if row >= 64 || (recon.scratch.is_coded[0][row] & mask) == 0 {
                        break;
                    }
                    y += 1;
                }
                n_bl = y;
            }
        }

        let mut apply_ibp = recon.frame.seq_ibp && tx != 0 && mrl_idx == 0;
        let dip = intra.dip as i32 - 1;
        let sm_top = intra.is_sm[0].a;
        let sm_left = intra.is_sm[0].l;
        let is_sm_flag = if apply_ibp {
            (sm_top * crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG)
                | (sm_left * crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
        } else {
            (sm_top | sm_left)
                * (crate::levels::ANGLE_SMOOTH_TOP_EDGE_FLAG
                    | crate::levels::ANGLE_SMOOTH_LEFT_EDGE_FLAG)
        };
        if intra.y_angle & 1 != 0 {
            apply_ibp = false;
        }
        let have_left = bx > fi.tile_col_start;
        let have_top = by > fi.tile_row_start;
        let intra_flags = crate::levels::ANGLE_IS_LUMA
            | is_sm_flag
            | if recon.frame.seq_intra_edge_filter {
                crate::levels::ANGLE_USE_EDGE_FILTER_FLAG
            } else {
                0
            }
            | if apply_ibp {
                crate::levels::ANGLE_IBP_FLAG
            } else {
                0
            }
            | (mrl_idx << crate::levels::ANGLE_MRL_IDX_SHIFT)
            | if mrl_mul {
                crate::levels::ANGLE_MULTI_MRL_FLAG
            } else {
                0
            }
            | if have_left {
                crate::levels::ANGLE_HAS_LEFT_FLAG
            } else {
                0
            }
            | if have_top {
                crate::levels::ANGLE_HAS_TOP_FLAG
            } else {
                0
            }
            | if dip >= 0 {
                crate::levels::ANGLE_DIP_FLAG
            } else {
                0
            };
        let angle_eff = if dip >= 0 { dip } else { angle };

        // Edge buffer origin: C uses `edge + 128 + !!mrl_idx*9`; we centre in a
        // larger slab so any layout (incl. multi-mrl second edge) fits.
        let edge_o: usize = 768 + if mrl_idx != 0 { 9 } else { 0 };

        // `prefilter_data` copy of the row above the SB; with single-thread /
        // filters-off decode that aliases the current plane's row directly above
        // the block (prefilter_data_full_frame). Passing it makes prepare_intra
        // _edges use the SB-edge row (top_stride 0) instead of stepping mrl_idx+1
        // rows up, which would cross the SB boundary for multi-reference-line
        // directional blocks. The slice is based at column 0 of row `4*by - 1`
        // (prepare adds the `x*4` column offset).
        let prefilter_top: Option<&[BD::Pixel]> = if have_top && (by & (sbsz - 1)) == 0 {
            let base = dst_off - (bx as usize) * 4 - stride;
            let plane: &[BD::Pixel] = &*recon.dst_y;
            Some(&plane[base..])
        } else {
            None
        };

        let m = crate::ipred_prepare::prepare_intra_edges(
            bd,
            bx,
            by,
            fi.tile_col_end,
            fi.tile_row_end,
            n_tr,
            n_bl,
            recon.dst_y,
            dst_off,
            stride,
            prefilter_top,
            y_mode,
            tw4,
            th4,
            angle_eff | intra_flags,
            recon.edge,
            edge_o,
        );

        let pred_angle = angle_eff | intra_flags;
        let max_w = 4 * fi.bw - 4 * bx;
        let max_h = 4 * fi.bh - 4 * by;
        dispatch_ipred(
            bd,
            m,
            recon.dst_y,
            dst_off,
            stride,
            recon.edge,
            edge_o,
            tw,
            th,
            pred_angle,
            max_w,
            max_h,
            &recon.frame.ibp_weights,
        );
    }

    if eob != -1 {
        if stx != 0 {
            const MASK: i32 = (1 << IntraPredMode::HorPred as i32)
                | (1 << IntraPredMode::HorDownPred as i32)
                | (1 << IntraPredMode::VertLeftPred as i32)
                | (1 << IntraPredMode::SmoothHPred as i32);
            // C: transpose = intrabc || !intra || !((mask >> b->y_mode) & 1);
            let transpose = is_intrabc || (MASK >> (y_mode as i32)) & 1 == 0;
            let stype = (stx & 3) - 1;
            let set = (stx >> 2) & 15;
            if tw >= 8 && th >= 8 {
                let koff = (set as usize * 3 + stype as usize) * 1536;
                let idx = (imin(t_dim.lh as i32, 3) - 1) as usize;
                let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_8X8[idx][transpose as usize];
                let mapping =
                    &crate::stx_tables::COEFF8X8_MAPPING[set as usize * 3 + stype as usize];
                crate::stx::stxfm8_dispatch(
                    recon.cf,
                    &crate::stx_tables::STX_8X8_KERNEL[koff..],
                    eob as usize,
                    recon.frame.bitdepth_max,
                    scan_out,
                    mapping,
                );
                eob = [63, 119, 231][idx];
            } else {
                let koff = (set as usize * 3 + stype as usize) * 128;
                let idx = imin(t_dim.lh as i32, 3) as usize;
                let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_4X4[idx][transpose as usize];
                crate::stx::stxfm4_dispatch(
                    recon.cf,
                    &crate::stx_tables::STX_4X4_KERNEL[koff..],
                    eob as usize,
                    recon.frame.bitdepth_max,
                    scan_out,
                );
                eob = [15, 15, 51, 99][idx];
            }
        }

        // DPCM branch on `b->intra && !b->intrabc && b->dpcm[0]` and the inter-DDT
        // ((flip)adst -> (f)ddt) branch on `seq_hdr->inter_ddt && !b->intra`.
        // Crucially the DDT branch keys off `!b->intra`, NOT IntraBC: an IntraBC
        // block has `b->intra == 1`, so it takes NEITHER branch. This luma tx
        // walk only ever runs for intra/IntraBC blocks (`b.is_intra == 1`), so
        // the DDT branch never fires here — applying it for IntraBC corrupts the
        // residual transform type.
        if lossless && is_intra && b.intra_data().dpcm[0] != 0 {
            txtp += ((1 + (y_mode == IntraPredMode::VertPred as u8) as u32) as u32) << 8;
        } else if recon.frame.seq_inter_ddt && b.is_intra == 0 {
            txtp += txtp & crate::tables::TX_DDT_MASK[tx] as u32;
        }

        crate::itx::inv_txfm_add(
            bd,
            recon.dst_y,
            dst_off,
            stride,
            recon.cf,
            txtp,
            eob,
            tx,
            &mut recon.scratch.itx_tmp,
        );
    }

    let coded_w = imin(tw4, 64 - bx4 as i32).max(0) as u32;
    if coded_w > 0 {
        let mask: u64 = (((1u128 << coded_w) - 1) as u64) << (bx4 as u32);
        for y in 0..th4 {
            let row = by4 + y as usize;
            if row < 64 {
                recon.scratch.is_coded[0][row] |= mask;
            }
        }
    }

    // LR no-skip mask (luma): set per coded luma TX block for the PC/NS-Wiener
    if eob != -1 {
        let m = &mut recon.lf_mask[recon.lf_idx];
        let mask_idx = (bx4 >> 4) as usize;
        let lr_mask: u16 = (((1u32 << tw4) - 1) << ((bx4 & 0xf) as u32)) as u16;
        for y in 0..th4 as usize {
            let row = by4 + y;
            if row < 64 && mask_idx < 4 {
                m.lr_noskip_mask[row][mask_idx] |= lr_mask;
            }
        }
    }

    let _ = orig_y_mode; // C restores b->y_mode; we never mutated b.
    Ok(())
}

/// Dispatch the resolved intra predictor `m` into `dst` (mirrors the C
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_ipred<BD: BitDepth>(
    bd: BD,
    m: u8,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    edge: &[BD::Pixel],
    edge_o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_w: i32,
    max_h: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    if BD::BPC == 8 {
        let dst8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst);
        let edge8: &[u8] = BD::Pixel::slice_as_ne_bytes(edge);
        dispatch_ipred_8bpc(
            m,
            dst8,
            dst_off,
            stride,
            edge8,
            edge_o,
            w,
            h,
            angle,
            max_w,
            max_h,
            ibp_weights,
        );
        return;
    }
    if let (Some(dst16), Some(edge16)) = (
        BD::Pixel::try_as_u16_slice_mut(dst),
        BD::Pixel::try_as_u16_slice(edge),
    ) {
        crate::ipred_dispatch::dispatch_ipred_hbd(
            m,
            bd.bitdepth(),
            bd.bitdepth_max() as u16,
            dst16,
            dst_off,
            stride,
            edge16,
            edge_o,
            w,
            h,
            angle,
            max_w,
            max_h,
            ibp_weights,
        );
        return;
    }

    use crate::ipred;
    use crate::levels::*;
    let d = &mut dst[dst_off..];
    match m {
        0 /* DcPred */ => ipred::ipred_dc(bd, d, stride, edge, edge_o, w, h, angle),
        _ if m == DC_128_PRED => ipred::ipred_dc_128(bd, d, stride, w, h),
        _ if m == TOP_DC_PRED => ipred::ipred_dc_top(bd, d, stride, edge, edge_o, w, h, angle),
        _ if m == LEFT_DC_PRED => ipred::ipred_dc_left(bd, d, stride, edge, edge_o, w, h, angle),
        2 /* HorPred */ => ipred::ipred_h(bd, d, stride, edge, edge_o, w, h, angle),
        1 /* VertPred */ => ipred::ipred_v(bd, d, stride, edge, edge_o, w, h, angle),
        12 /* PaethPred */ => ipred::ipred_paeth(bd, d, stride, edge, edge_o, w, h),
        9 /* SmoothPred */ => ipred::ipred_smooth(bd, d, stride, edge, edge_o, w, h),
        10 /* SmoothVPred */ => ipred::ipred_smooth_v(bd, d, stride, edge, edge_o, w, h),
        11 /* SmoothHPred */ => ipred::ipred_smooth_h(bd, d, stride, edge, edge_o, w, h),
        _ if m == Z1_PRED => {
            ipred::ipred_z1(bd, d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights)
        }
        _ if m == Z2_PRED => ipred::ipred_z2(bd, d, stride, edge, edge_o, w, h, angle, max_w, max_h),
        _ if m == Z3_PRED => {
            ipred::ipred_z3(bd, d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights)
        }
        _ if m == DIP_PRED => ipred::ipred_dip(bd, d, stride, edge, edge_o, w, h, angle),
        _ => ipred::ipred_dc_128(bd, d, stride, w, h),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ipred_8bpc(
    m: u8,
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    edge: &[u8],
    edge_o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_w: i32,
    max_h: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let d = &mut dst[dst_off..];
    match m {
        0 /* DcPred */ => crate::ipred_dispatch::ipred_dc(d, stride, edge, edge_o, w, h, angle),
        _ if m == DC_128_PRED => crate::ipred_dispatch::ipred_dc_128(d, stride, w, h),
        _ if m == TOP_DC_PRED => crate::ipred_dispatch::ipred_dc_top(d, stride, edge, edge_o, w, h, angle),
        _ if m == LEFT_DC_PRED => crate::ipred_dispatch::ipred_dc_left(d, stride, edge, edge_o, w, h, angle),
        2 /* HorPred */ => crate::ipred_dispatch::ipred_h(d, stride, edge, edge_o, w, h, angle),
        1 /* VertPred */ => crate::ipred_dispatch::ipred_v(d, stride, edge, edge_o, w, h, angle),
        12 /* PaethPred */ => crate::ipred_dispatch::ipred_paeth(d, stride, edge, edge_o, w, h),
        9 /* SmoothPred */ => crate::ipred_dispatch::ipred_smooth(d, stride, edge, edge_o, w, h),
        10 /* SmoothVPred */ => crate::ipred_dispatch::ipred_smooth_v(d, stride, edge, edge_o, w, h),
        11 /* SmoothHPred */ => crate::ipred_dispatch::ipred_smooth_h(d, stride, edge, edge_o, w, h),
        _ if m == Z1_PRED => {
            crate::ipred_dispatch::ipred_z1(d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights)
        }
        _ if m == Z2_PRED => crate::ipred_dispatch::ipred_z2(d, stride, edge, edge_o, w, h, angle, max_w, max_h),
        _ if m == Z3_PRED => {
            crate::ipred_dispatch::ipred_z3(d, stride, edge, edge_o, w, h, angle, max_w, max_h, ibp_weights)
        }
        _ if m == DIP_PRED => crate::ipred_dispatch::ipred_dip_8bpc(d, stride, edge, edge_o, w, h, angle),
        _ => crate::ipred_dispatch::ipred_dc_128(d, stride, w, h),
    }
}

/// Compound `avg` blend dispatch: NEON 8bpc fast-path (byte-identical) or the
/// generic HBD kernel.
#[inline]
pub(crate) fn mc_avg<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    if BD::BPC == 8 {
        let d8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst);
        crate::mc_dispatch::avg_8bpc(d8, dst_stride, tmp1, tmp2, w, h);
    } else {
        crate::mc::avg(bd, dst, dst_stride, tmp1, tmp2, w, h);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mc_w_avg<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    if BD::BPC == 8 {
        let d8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst);
        crate::mc_dispatch::w_avg_8bpc(d8, dst_stride, tmp1, tmp2, w, h, weight);
    } else {
        crate::mc::w_avg(bd, dst, dst_stride, tmp1, tmp2, w, h, weight);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mc_mask<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &[u8],
) {
    if BD::BPC == 8 {
        let d8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst);
        crate::mc_dispatch::mask_8bpc(d8, dst_stride, tmp1, tmp2, w, h, m);
    } else {
        crate::mc::mask_fn(bd, dst, dst_stride, tmp1, tmp2, w, h, m);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mc_w_mask<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    if BD::BPC == 8 {
        let d8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst);
        crate::mc_dispatch::w_mask_8bpc(
            d8,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            m,
            mask_stride,
            sign,
            ss_hor,
            ss_ver,
        );
    } else {
        crate::mc::w_mask(
            bd,
            dst,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            m,
            mask_stride,
            sign,
            ss_hor,
            ss_ver,
        );
    }
}

/// Bundle of the superblock-decode state that threads unchanged through the
/// `decode_sb` partition recursion. Passing a single `&mut SbCtx` instead of
/// ~16 individual `&mut` arguments removes the per-call pointer crowding (the
/// "pointer congestion" of spilling that many references at every recursive
/// call) and keeps the call sites legible. It is deliberately NOT generic over
/// `BitDepth` -- the bit-depth-dependent `recon` stays a separate argument --
/// so the non-generic `decode_partition` can borrow from it as well.
pub(crate) struct SbCtx<'a, const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>> {
    pub(crate) fi: &'a SbFrameInfo,
    pub(crate) bx: &'a mut i32,
    pub(crate) by: &'a mut i32,
    pub(crate) cbx: &'a mut i32,
    pub(crate) cby: &'a mut i32,
    pub(crate) intra_region: &'a mut i32,
    pub(crate) sdp_cfl_disallowed: &'a mut i32,
    pub(crate) a: &'a mut BlockContext,
    pub(crate) l: &'a mut BlockContext,
    pub(crate) msac: &'a mut M,
    pub(crate) cdf_m: &'a mut CdfModeContext,
    pub(crate) cdf_dmv: &'a mut CdfMvContext,
    pub(crate) part_w: &'a mut Vec<u8>,
    pub(crate) part_w_idx: &'a mut usize,
    pub(crate) part_r: &'a [u8],
    pub(crate) part_r_idx: &'a mut usize,
}

pub fn decode_sb<BD: crate::pixel::BitDepth, const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    ctx: &mut SbCtx<'_, UPDATE_CDF, M>,
    recon: &mut ReconCtx<BD>,
    pass: u8,
    lbs: BlockSize,
    cbs: BlockSize,
    dir_ptr: &mut i32,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    // bs is always valid for a well-formed partition tree; a malformed stream can
    // leave both block sizes invalid, so abort gracefully rather than panic.
    if bs == BlockSize::Invalid {
        return Err(());
    }

    let b_dim = &BLOCK_DIMENSIONS[bs as u8 as usize];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;
    let hw4 = bw4 >> 1;
    let hh4 = bh4 >> 1;
    let qw4 = hw4 >> 1;
    let qh4 = hh4 >> 1;
    let have_h_split = ctx.fi.bw > *ctx.bx + hw4;
    let have_v_split = ctx.fi.bh > *ctx.by + hh4;
    let cbs_orig = cbs;

    if lbs == BlockSize::Bs64x64
        && cbs == BlockSize::Bs64x64
        && ctx.fi.sdp
        && !ctx.fi.is_inter_or_switch
    {
        let mut dir = 0i32;
        decode_sb(ctx, recon, pass, lbs, BlockSize::Invalid, &mut dir)?;
        return decode_sb(ctx, recon, pass, BlockSize::Invalid, cbs, &mut dir);
    }

    let pl = (lbs == BlockSize::Invalid) as usize;
    let pcc = &PARTITION_SUBB[bs as u8 as usize];
    let (bp, cbs) = decode_partition(
        ctx,
        pass,
        lbs,
        cbs,
        bs,
        b_dim,
        bw4,
        bh4,
        qw4,
        qh4,
        have_h_split,
        have_v_split,
        dir_ptr,
    )?;

    if bs == cbs {
        *ctx.cbx = *ctx.bx;
        *ctx.cby = *ctx.by;
    }

    let lim = &PARTITION_LIM[bp as u8 as usize];
    let mut child_dir = ((bw4 <= lim[0] as i32 || bh4 <= lim[1] as i32) as i32) << 24;

    match bp {
        BlockPartition::None => {
            let _b = decode_b(ctx, recon, pass, lbs, cbs)?;
            if pass & (Pass::Entropy as u8) != 0 {
                let bx4 = (*ctx.bx & 63) as usize;
                let by4 = (*ctx.by & 63) as usize;
                if (cbs as i8 | lbs as i8) != BlockSize::Invalid as i8 {
                    // C: case_set(b_dim[2 + i]) writes 1<<b_dim[2+i] bytes (pow2 length),
                    // for both partition[0] and partition[1].
                    memset_pow2(&mut ctx.a.partition[0], bx4, !(b_dim[0] - 1), b_dim[2]);
                    memset_pow2(&mut ctx.a.partition[1], bx4, !(b_dim[0] - 1), b_dim[2]);
                    memset_pow2(&mut ctx.l.partition[0], by4, !(b_dim[1] - 1), b_dim[3]);
                    memset_pow2(&mut ctx.l.partition[1], by4, !(b_dim[1] - 1), b_dim[3]);
                } else {
                    memset_pow2(&mut ctx.a.partition[pl], bx4, !(b_dim[0] - 1), b_dim[2]);
                    memset_pow2(&mut ctx.l.partition[pl], by4, !(b_dim[1] - 1), b_dim[3]);
                }
            }
        }
        BlockPartition::V => {
            if hw4 <= 0 {
                return Err(());
            }
            let sub4 = bs == cbs && (hw4 >> ctx.fi.ss_hor) > 0;
            if !sub4 && pl != 0 {
                return Err(());
            }
            let child_lbs = if pl != 0 {
                BlockSize::Invalid
            } else {
                BlockSize::from_raw(pcc.part[1][0])
            };
            let child_cbs_first = if sub4 {
                BlockSize::from_raw(pcc.part[1][0])
            } else {
                BlockSize::Invalid
            };
            decode_sb(ctx, recon, pass, child_lbs, child_cbs_first, &mut child_dir)?;
            if *ctx.bx + hw4 >= ctx.fi.bw { /* done */
            } else {
                *ctx.bx += hw4;
                let child_cbs_second = if sub4 {
                    BlockSize::from_raw(pcc.part[1][0])
                } else {
                    cbs
                };
                decode_sb(
                    ctx,
                    recon,
                    pass,
                    child_lbs,
                    child_cbs_second,
                    &mut child_dir,
                )?;
                *ctx.bx -= hw4;
            }
        }
        BlockPartition::H => {
            if hh4 <= 0 {
                return Err(());
            }
            let sub4 = bs == cbs && (hh4 >> ctx.fi.ss_ver) > 0;
            if !sub4 && pl != 0 {
                return Err(());
            }
            let child_lbs = if pl != 0 {
                BlockSize::Invalid
            } else {
                BlockSize::from_raw(pcc.part[0][0])
            };
            let child_cbs_first = if sub4 {
                BlockSize::from_raw(pcc.part[0][0])
            } else {
                BlockSize::Invalid
            };
            decode_sb(ctx, recon, pass, child_lbs, child_cbs_first, &mut child_dir)?;
            if *ctx.by + hh4 >= ctx.fi.bh { /* done */
            } else {
                *ctx.by += hh4;
                let child_cbs_second = if sub4 {
                    BlockSize::from_raw(pcc.part[0][0])
                } else {
                    cbs
                };
                decode_sb(
                    ctx,
                    recon,
                    pass,
                    child_lbs,
                    child_cbs_second,
                    &mut child_dir,
                )?;
                *ctx.by -= hh4;
            }
        }
        BlockPartition::Split => {
            // A square SPLIT of a 128×128/256×256 SHARED block. For interior
            // blocks all four children are on-frame; at a right/bottom boundary
            // the off-frame children are skipped (mirroring AVM, which still
            // reads do_square_split there because PARTITION_SPLIT stays eligible
            // even when an implied rect direction is chroma-invalid for the
            // subsampling).
            let sbs = BlockSize::from_raw(pcc.part[0][3]);
            // Monochrome (I400) carries no chroma block (cbs == Invalid): recurse
            // luma-only. Otherwise a SHARED square split must have cbs == lbs and
            // the children carry `sbs` as their (coupled) chroma block size.
            let child_cbs = if cbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else if cbs == lbs {
                sbs
            } else {
                return Err(());
            };
            // top-left (origin is always on-frame)
            decode_sb(ctx, recon, pass, sbs, child_cbs, &mut child_dir)?;
            // top-right
            if *ctx.bx + hw4 < ctx.fi.bw {
                *ctx.bx += hw4;
                decode_sb(ctx, recon, pass, sbs, child_cbs, &mut child_dir)?;
                *ctx.bx -= hw4;
            }
            // bottom row
            if *ctx.by + hh4 < ctx.fi.bh {
                *ctx.by += hh4;
                decode_sb(ctx, recon, pass, sbs, child_cbs, &mut child_dir)?;
                if *ctx.bx + hw4 < ctx.fi.bw {
                    *ctx.bx += hw4;
                    decode_sb(ctx, recon, pass, sbs, child_cbs, &mut child_dir)?;
                    *ctx.bx -= hw4;
                }
                *ctx.by -= hh4;
            }
        }
        BlockPartition::V3 => {
            if qw4 <= 0 || hh4 <= 0 {
                return Err(());
            }
            let sub4 = bs == cbs && (qw4 >> ctx.fi.ss_hor) > 0 && (hh4 >> ctx.fi.ss_ver) > 0;
            if !sub4 && pl != 0 {
                return Err(());
            }
            let i_3only = cbs == BlockSize::Invalid || (!sub4 && bs != BlockSize::Bs32x8);
            let p1_1 = BlockSize::from_raw(pcc.part[1][1]);
            let p1_3 = BlockSize::from_raw(pcc.part[1][3]);
            let lbs_child = if pl != 0 { BlockSize::Invalid } else { p1_1 };
            let cbs_first = if i_3only { BlockSize::Invalid } else { p1_1 };
            decode_sb(ctx, recon, pass, lbs_child, cbs_first, &mut child_dir)?;
            if *ctx.bx + qw4 >= ctx.fi.bw { /* done */
            } else {
                *ctx.bx += qw4;
                if !i_3only {
                    *ctx.cbx = *ctx.bx;
                }
                let lbs_mid = if pl != 0 { BlockSize::Invalid } else { p1_3 };
                let cbs_mid = if sub4 { p1_3 } else { BlockSize::Invalid };
                decode_sb(ctx, recon, pass, lbs_mid, cbs_mid, &mut child_dir)?;
                if *ctx.by + hh4 < ctx.fi.bh {
                    *ctx.by += hh4;
                    let cbs_mid2 = if i_3only {
                        BlockSize::Invalid
                    } else if sub4 {
                        p1_3
                    } else {
                        BlockSize::from_raw(pcc.part[1][0])
                    };
                    decode_sb(ctx, recon, pass, lbs_mid, cbs_mid2, &mut child_dir)?;
                    *ctx.by -= hh4;
                }
                if *ctx.bx + hw4 >= ctx.fi.bw {
                    *ctx.bx -= qw4;
                } else {
                    *ctx.bx += hw4;
                    let cbs_last = if i_3only { cbs } else { p1_1 };
                    decode_sb(ctx, recon, pass, lbs_child, cbs_last, &mut child_dir)?;
                    *ctx.bx -= 3 * qw4;
                }
            }
        }
        BlockPartition::H3 => {
            if qh4 <= 0 || hw4 <= 0 {
                return Err(());
            }
            let sub4 = bs == cbs && (qh4 >> ctx.fi.ss_ver) > 0 && (hw4 >> ctx.fi.ss_hor) > 0;
            if !sub4 && pl != 0 {
                return Err(());
            }
            let i_3only = cbs == BlockSize::Invalid || (!sub4 && bs != BlockSize::Bs8x32);
            let p0_1 = BlockSize::from_raw(pcc.part[0][1]);
            let p0_3 = BlockSize::from_raw(pcc.part[0][3]);
            let lbs_child = if pl != 0 { BlockSize::Invalid } else { p0_1 };
            let cbs_first = if i_3only { BlockSize::Invalid } else { p0_1 };
            decode_sb(ctx, recon, pass, lbs_child, cbs_first, &mut child_dir)?;
            if *ctx.by + qh4 >= ctx.fi.bh { /* done */
            } else {
                *ctx.by += qh4;
                if !i_3only {
                    *ctx.cby = *ctx.by;
                }
                let lbs_mid = if pl != 0 { BlockSize::Invalid } else { p0_3 };
                let cbs_mid = if sub4 { p0_3 } else { BlockSize::Invalid };
                decode_sb(ctx, recon, pass, lbs_mid, cbs_mid, &mut child_dir)?;
                if *ctx.bx + hw4 < ctx.fi.bw {
                    *ctx.bx += hw4;
                    let cbs_mid2 = if i_3only {
                        BlockSize::Invalid
                    } else if sub4 {
                        p0_3
                    } else {
                        BlockSize::from_raw(pcc.part[0][0])
                    };
                    decode_sb(ctx, recon, pass, lbs_mid, cbs_mid2, &mut child_dir)?;
                    *ctx.bx -= hw4;
                }
                if *ctx.by + hh4 >= ctx.fi.bh {
                    *ctx.by -= qh4;
                } else {
                    *ctx.by += hh4;
                    let cbs_last = if i_3only { cbs } else { p0_1 };
                    decode_sb(ctx, recon, pass, lbs_child, cbs_last, &mut child_dir)?;
                    *ctx.by -= 3 * qh4;
                }
            }
        }
        BlockPartition::V4A | BlockPartition::V4B => {
            let ew4 = qw4 >> 1;
            if ew4 <= 0 {
                return Err(());
            }
            let sub4 = bs == cbs && (ew4 >> ctx.fi.ss_hor) > 0;
            if !sub4 && pl != 0 {
                return Err(());
            }
            let p1_2 = BlockSize::from_raw(pcc.part[1][2]);
            let var = bp as i8 - BlockPartition::V4A as i8;
            let p1_nvar = BlockSize::from_raw(pcc.part[1][(!var & 1) as usize]);
            let p1_var = BlockSize::from_raw(pcc.part[1][var as usize]);
            let lbs_edge = if pl != 0 { BlockSize::Invalid } else { p1_2 };
            let lbs_nvar = if pl != 0 { BlockSize::Invalid } else { p1_nvar };
            let lbs_var = if pl != 0 { BlockSize::Invalid } else { p1_var };

            decode_sb(
                ctx,
                recon,
                pass,
                lbs_edge,
                if sub4 { p1_2 } else { BlockSize::Invalid },
                &mut child_dir,
            )?;
            if *ctx.bx + ew4 >= ctx.fi.bw { /* done */
            } else {
                *ctx.bx += ew4;
                decode_sb(
                    ctx,
                    recon,
                    pass,
                    lbs_nvar,
                    if sub4 { p1_nvar } else { BlockSize::Invalid },
                    &mut child_dir,
                )?;
                let w4a = qw4 << var;
                let w4b = hw4 >> var;
                if *ctx.bx + w4a >= ctx.fi.bw {
                    *ctx.bx -= ew4;
                } else {
                    *ctx.bx += w4a;
                    decode_sb(
                        ctx,
                        recon,
                        pass,
                        lbs_var,
                        if sub4 { p1_var } else { BlockSize::Invalid },
                        &mut child_dir,
                    )?;
                    if *ctx.bx + w4b >= ctx.fi.bw {
                        *ctx.bx -= ew4 + w4a;
                    } else {
                        *ctx.bx += w4b;
                        decode_sb(
                            ctx,
                            recon,
                            pass,
                            lbs_edge,
                            if sub4 { p1_2 } else { cbs },
                            &mut child_dir,
                        )?;
                        *ctx.bx -= 7 * ew4;
                    }
                }
            }
        }
        BlockPartition::H4A | BlockPartition::H4B => {
            let eh4 = qh4 >> 1;
            if eh4 <= 0 {
                return Err(());
            }
            let sub4 = bs == cbs && (eh4 >> ctx.fi.ss_ver) > 0;
            if !sub4 && pl != 0 {
                return Err(());
            }
            let p0_2 = BlockSize::from_raw(pcc.part[0][2]);
            let var = bp as i8 - BlockPartition::H4A as i8;
            let p0_nvar = BlockSize::from_raw(pcc.part[0][(!var & 1) as usize]);
            let p0_var = BlockSize::from_raw(pcc.part[0][var as usize]);
            let lbs_edge = if pl != 0 { BlockSize::Invalid } else { p0_2 };
            let lbs_nvar = if pl != 0 { BlockSize::Invalid } else { p0_nvar };
            let lbs_var = if pl != 0 { BlockSize::Invalid } else { p0_var };

            decode_sb(
                ctx,
                recon,
                pass,
                lbs_edge,
                if sub4 { p0_2 } else { BlockSize::Invalid },
                &mut child_dir,
            )?;
            if *ctx.by + eh4 >= ctx.fi.bh { /* done */
            } else {
                *ctx.by += eh4;
                decode_sb(
                    ctx,
                    recon,
                    pass,
                    lbs_nvar,
                    if sub4 { p0_nvar } else { BlockSize::Invalid },
                    &mut child_dir,
                )?;
                let h4a = qh4 << var;
                let h4b = hh4 >> var;
                if *ctx.by + h4a >= ctx.fi.bh {
                    *ctx.by -= eh4;
                } else {
                    *ctx.by += h4a;
                    decode_sb(
                        ctx,
                        recon,
                        pass,
                        lbs_var,
                        if sub4 { p0_var } else { BlockSize::Invalid },
                        &mut child_dir,
                    )?;
                    if *ctx.by + h4b >= ctx.fi.bh {
                        *ctx.by -= eh4 + h4a;
                    } else {
                        *ctx.by += h4b;
                        decode_sb(
                            ctx,
                            recon,
                            pass,
                            lbs_edge,
                            if sub4 { p0_2 } else { cbs },
                            &mut child_dir,
                        )?;
                        *ctx.by -= 7 * eh4;
                    }
                }
            }
        }
        _ => return Err(()),
    }

    *dir_ptr |= (child_dir & 0xff) << 16;

    if *ctx.intra_region != 0 && cbs_orig != BlockSize::Invalid {
        *ctx.cbx = *ctx.bx;
        *ctx.cby = *ctx.by;
        let _b = decode_b(ctx, recon, pass, BlockSize::Invalid, cbs_orig)?;
        *ctx.intra_region = 0;
    }

    Ok(())
}
