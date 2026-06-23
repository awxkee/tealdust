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

use crate::env::BlockContext;
use crate::headers::{AdaptiveBoolean, FrameHeader, MAX_SEGMENTS, RestorationType};
use crate::internal::{LoopFilterState, NsWienerBank, ScalableMotionParams};
use crate::intops::{imax, imin};
use crate::levels::{Av2Block, BlockSize, Mv, RefPair, TIP_FRAME};

use crate::msac::MsacContext;

use crate::pixel::BitDepth;
use crate::quantizer::dq_lookup;
use crate::refmvs;
use crate::tables::{NS_WIENER_COEF_RANGE_UV, NS_WIENER_COEF_RANGE_Y};

mod block_decode;
mod frame_setup;
mod loopfilter;
mod recon_inter;
mod recon_intra;
mod recon_opfl;
mod syntax;
pub(crate) use block_decode::*;
pub(crate) use frame_setup::*;
pub(crate) use loopfilter::*;
pub(crate) use recon_inter::*;
pub(crate) use recon_intra::*;
pub(crate) use recon_opfl::*;
pub(crate) use syntax::*;

pub(crate) fn init_wiener(frame_hdr: &FrameHeader, lf: &mut LoopFilterState) {
    let rtype = frame_hdr.restoration.p[0].restoration_type;
    if rtype == RestorationType::None as u8 {
        return;
    }

    let qidx = frame_hdr.quant.yac as i32;
    lf.base_q = dq_lookup(qidx);

    let idx = if qidx < 130 {
        0
    } else if qidx < 190 {
        1
    } else if qidx < 220 {
        2
    } else {
        3
    };
    lf.wiener_idx = idx;

    if rtype == RestorationType::NsWiener as u8 || rtype == RestorationType::Switchable as u8 {
        let num_classes_idx = frame_hdr.restoration.p[0].ns.num_classes_idx as usize;
        if num_classes_idx > 0 {
            lf.ns_subclass_class_idx = Some(num_classes_idx - 1);
        } else {
            lf.ns_subclass_class_idx = None;
        }
    } else {
        lf.ns_subclass_class_idx = None;
    }
}

pub fn compute_restore_planes(frame_hdr: &FrameHeader) -> i32 {
    let has_y = frame_hdr.restoration.p[0].restoration_type != RestorationType::None as u8
        || frame_hdr.gdf.enabled != AdaptiveBoolean::Off;
    let has_u = frame_hdr.restoration.p[1].restoration_type != RestorationType::None as u8;
    let has_v = frame_hdr.restoration.p[2].restoration_type != RestorationType::None as u8;
    (has_y as i32) | ((has_u as i32) << 1) | ((has_v as i32) << 2)
}

pub fn compute_gdf_ref_dst_idx(frame_hdr: &FrameHeader, absrefdist: &[u8; 7]) -> i32 {
    if frame_hdr.gdf.enabled == AdaptiveBoolean::Off {
        return 0;
    }
    let is_inter_or_switch = (frame_hdr.frame_type as u8) & 1 != 0;
    if !is_inter_or_switch {
        return 0;
    }
    let mut max_dist = 0i32;
    for i in 0..imin(frame_hdr.n_ref_frames as i32, 2) as usize {
        max_dist = imax(max_dist, absrefdist[i] as i32);
    }
    static REF_DST_IDX_TBL: [i32; 12] = [5, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5];
    REF_DST_IDX_TBL[imin(max_dist, 11) as usize]
}

pub(crate) fn init_ns_wiener_bank(bank: &mut NsWienerBank, pl: usize, n_classes: usize) {
    bank.bank_size = [0; 16];
    bank.bank_idx = [0; 16];
    let cf_range: &[[i8; 2]] = if pl > 0 {
        &NS_WIENER_COEF_RANGE_UV
    } else {
        &NS_WIENER_COEF_RANGE_Y
    };
    let n_coeffs = 16 + if pl > 0 { 2 } else { 0 };
    for n in 0..n_classes {
        for m in 0..n_coeffs {
            bank.filter[0][n][m] = cf_range[m][1] + ((1i8 << cf_range[m][0]) >> 1);
        }
    }
}

pub(crate) fn init_start_of_tile_row(
    buf: &mut Vec<u8>,
    sbh: i32,
    tile_rows: u8,
    row_start_sb: &[u16],
) {
    buf.resize(sbh as usize, 0);
    let sbh = sbh as usize;
    let mut sby = 0usize;
    for tile_row in 0..tile_rows as usize {
        if sby >= sbh {
            break;
        }
        buf[sby] = ((tile_row << 1) | 1) as u8;
        sby += 1;
        // For valid streams row_start_sb[tile_row + 1] <= sbh; clamp the bound so
        // a malformed tiling whose row starts exceed the frame height cannot
        // write past the sbh-sized buffer. No-op for valid input.
        let end = (row_start_sb[tile_row + 1] as usize).min(sbh);
        while sby < end {
            buf[sby] = (tile_row << 1) as u8;
            sby += 1;
        }
    }
}

pub fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                r + ((diff + 1) >> 1)
            } else {
                r - (diff >> 1)
            }
        } else {
            diff
        }
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                r + ((diff + 1) >> 1)
            } else {
                r - (diff >> 1)
            }
        } else {
            max - (diff + 1)
        }
    }
}

pub(crate) fn init_quant_tables(
    frame_hdr: &FrameHeader,
    qidx: i32,
    dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS],
    qmax: i32,
) {
    let n = if frame_hdr.segmentation.enabled != 0 {
        8
    } else {
        1
    };
    for i in 0..n {
        let yac = if frame_hdr.segmentation.enabled != 0 {
            qidx + frame_hdr.segmentation.d.delta_q[i] as i32
        } else {
            qidx
        };
        let ydc = yac + frame_hdr.quant.ydc_delta as i32;
        let uac = yac + frame_hdr.quant.uac_delta as i32;
        let udc = yac + frame_hdr.quant.udc_delta as i32;
        let vac = yac + frame_hdr.quant.vac_delta as i32;
        let vdc = yac + frame_hdr.quant.vdc_delta as i32;

        // AVM clamps the effective qindex to the bit-depth MAXQ (255/303/351).
        dq[i][0][0] = dq_lookup(ydc.min(qmax)) as u32;
        dq[i][0][1] = dq_lookup(yac.min(qmax)) as u32;
        dq[i][1][0] = dq_lookup(udc.min(qmax)) as u32;
        dq[i][1][1] = dq_lookup(uac.min(qmax)) as u32;
        dq[i][2][0] = dq_lookup(vdc.min(qmax)) as u32;
        dq[i][2][1] = dq_lookup(vac.min(qmax)) as u32;
    }
}

/// Recompute the per-segment dequant tables for a single qindex (used by the
/// per-superblock delta-q path; mirrors `init_quant_tables` with state pulled
/// from `SbFrameInfo` instead of `FrameHeader`).
pub fn init_quant_tables_fi(
    fi: &SbFrameInfo,
    qidx: i32,
    dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS],
    qmax: i32,
) {
    let n = if fi.seg_enabled { 8 } else { 1 };
    for i in 0..n {
        let yac = if fi.seg_enabled {
            qidx + fi.seg_delta_q[i] as i32
        } else {
            qidx
        };
        let ydc = yac + fi.q_ydc_delta;
        let uac = yac + fi.q_uac_delta;
        let udc = yac + fi.q_udc_delta;
        let vac = yac + fi.q_vac_delta;
        let vdc = yac + fi.q_vdc_delta;

        // AVM clamps the effective qindex to the bit-depth MAXQ (255/303/351).
        dq[i][0][0] = dq_lookup(ydc.min(qmax)) as u32;
        dq[i][0][1] = dq_lookup(yac.min(qmax)) as u32;
        dq[i][1][0] = dq_lookup(udc.min(qmax)) as u32;
        dq[i][1][1] = dq_lookup(uac.min(qmax)) as u32;
        dq[i][2][0] = dq_lookup(vdc.min(qmax)) as u32;
        dq[i][2][1] = dq_lookup(vac.min(qmax)) as u32;
    }
}

pub(crate) const N_SWITCHABLE_FILTERS: usize = 3;

pub(crate) fn reset_context(ctx: &mut BlockContext, keyframe: bool, is_tip_frame: bool) {
    ctx.tx_lpf_y.fill(3);
    ctx.tx_lpf_uv.fill(2);
    if is_tip_frame {
        return;
    }
    ctx.midx.fill(0xff);
    ctx.intra.fill(keyframe as u8);
    ctx.uvmode.fill(0); // DC_PRED
    if keyframe {
        ctx.mode.fill(0); // DC_PRED
    }
    ctx.partition[0].fill(0);
    ctx.partition[1].fill(0);
    ctx.skip_txfm.fill(0);
    ctx.skip_mode.fill(0);
    if !keyframe {
        ctx.r#ref[0].fill(-1);
        ctx.r#ref[1].fill(-1);
        ctx.comp_type.fill(0);
        ctx.mode.fill(13); // NEARMV
    }
    ctx.mrl.fill(0);
    ctx.lcoef.fill(0x40);
    ctx.ccoef[0].fill(0x40);
    ctx.ccoef[1].fill(0x40);
    ctx.filter.fill(N_SWITCHABLE_FILTERS as u8);
    ctx.seg_pred.fill(0);
    ctx.pal_sz.fill(0);
}

#[derive(Default)]
pub struct SbFrameInfo {
    pub bw: i32,
    pub bh: i32,
    pub ss_ver: i32,
    pub ss_hor: i32,
    pub root_bs: BlockSize,
    pub is_inter_or_switch: bool,
    pub sdp: bool,
    pub ext_sdp: bool,
    pub ext_partitions: bool,
    pub uneven_4way: bool,
    pub max_pb_aspect_ratio_log2: u8,
    pub n_passes: i32,
    // Segmentation
    pub seg_enabled: bool,
    pub seg_update_map: bool,
    pub seg_temporal: bool,
    pub seg_preskip: bool,
    pub seg_ext: bool,
    pub seg_last_active_segid: u8,
    pub seg_globalmv_mask: u16,
    pub seg_skip_mask: u16,
    pub seg_lossless: [u8; crate::headers::MAX_SEGMENTS],
    // Delta-q (per-superblock)
    pub delta_q_present: bool,
    pub delta_q_res_log2: u8,
    pub quant_yac: i32,
    pub sb128: i32,
    pub b4_stride: isize,
    // Quantizer deltas, needed to recompute per-SB dequant tables on delta-q.
    pub q_ydc_delta: i32,
    pub q_uac_delta: i32,
    pub q_udc_delta: i32,
    pub q_vac_delta: i32,
    pub q_vdc_delta: i32,
    pub seg_delta_q: [i16; crate::headers::MAX_SEGMENTS],
    // GDF / CDEF-index / CCSO (read at SB / 64x64 boundaries, before delta-q)
    pub gdf_enabled: crate::headers::AdaptiveBoolean,
    pub gdf_is_key: bool,
    pub cur_w: i32,
    pub cur_h: i32,
    pub cdef_enabled: bool,
    pub cdef_on_skiptx: bool,
    pub cdef_n_strengths: u8,
    pub ccso_enabled: [bool; 3],
    pub ccso_sb_reuse: [bool; 3],
    pub sb256w: i32,
    // Frame flags
    pub skip_mode_enabled: bool,
    pub allow_intrabc: bool,
    pub any_lossless: bool,
    pub has_chroma_layout: bool,
    // Sequence features
    pub idtx_intra: bool,
    pub mrls: bool,
    pub mhccp: bool,
    pub cfl: bool,
    pub allow_screen_content_tools: bool,
    pub intra_dip: bool,
    pub force_integer_mv: bool,
    pub max_bvp_drl_bits: u8,
    pub max_drl_bits: u8,
    pub bawp: bool,
    pub txfm_switchable: bool,
    pub skip_mode_refs: RefPair,
    pub n_ref_frames: u8,
    pub warp_motion: bool,
    pub motion_modes: u8,
    pub adaptive_mvd: bool,
    pub flex_mvres: bool,
    pub mv_precision: u8,
    pub mvd_sign_derive: bool,
    pub tip_frame_mode: u8,
    /// `frame_hdr->tip.global_wtd_idx` — selects the TIP block CWP weight from
    pub tip_global_wtd_idx: u8,
    pub six_param_warp_delta: bool,
    pub subpel_filter_mode: u8,
    pub switchable_comp_refs: bool,
    pub num_same_ref_comp: u8,
    pub refdir: [u8; 8],
    pub refdist: [i8; 8],
    pub opfl_refine_type: u8,
    pub masked_compound: bool,
    pub cwp: bool,
    pub refine_mv_enabled: bool,
    pub absrefdist: [u8; 8],
    /// (`comp_type`) neighbour context. -1/-2 sentinel when no future ref.
    pub furthest_future_refidx: i8,
    /// `f->rf.tip.ref` (the TIP reference pair). Used by `get_compref_ctx` to
    /// match TIP-coded neighbours against the current block's compound refs.
    pub tip: RefPair,
    // Tile bounds
    pub tile_col_start: i32,
    pub tile_col_end: i32,
    pub tile_row_start: i32,
    pub tile_row_end: i32,
    pub sb_step: i32,
}

pub struct SbFrameInfoArgs<'a> {
    pub seq_hdr: &'a crate::headers::SequenceHeader,
    pub frame_hdr: &'a FrameHeader,
    pub bw: i32,
    pub bh: i32,
    pub root_bs: BlockSize,
    pub sb_step: i32,
    pub n_passes: i32,
    pub refdir: [u8; 8],
    pub refdist: &'a [i8; 7],
    pub absrefdist: &'a [u8; 7],
    pub skip_mode_refs: RefPair,
    pub tile_col_start: i32,
    pub tile_col_end: i32,
    pub tile_row_start: i32,
    pub tile_row_end: i32,
    pub furthest_future_refidx: i8,
    pub tip: RefPair,
}

impl SbFrameInfo {
    /// Build the per-superblock frame info bundle from the live sequence and
    /// frame headers plus the frame-level geometry and reference state.
    ///
    /// `refdir`/`refdist`/`absrefdist`/`skip_mode_refs` are precomputed on the
    /// `FrameContext` (see `refmvs::init_frame`); the 7-element refmvs arrays are
    /// zero-padded into the 8-element layout this struct uses.
    pub fn from_frame(args: SbFrameInfoArgs<'_>) -> Self {
        let SbFrameInfoArgs {
            seq_hdr,
            frame_hdr,
            bw,
            bh,
            root_bs,
            sb_step,
            n_passes,
            refdir,
            refdist,
            absrefdist,
            skip_mode_refs,
            tile_col_start,
            tile_col_end,
            tile_row_start,
            tile_row_end,
            furthest_future_refidx,
            tip,
        } = args;
        let mut refdist8 = [0i8; 8];
        refdist8[..7].copy_from_slice(refdist);
        let mut absrefdist8 = [0u8; 8];
        absrefdist8[..7].copy_from_slice(absrefdist);

        SbFrameInfo {
            bw,
            bh,
            ss_ver: seq_hdr.ss_ver as i32,
            ss_hor: seq_hdr.ss_hor as i32,
            root_bs,
            is_inter_or_switch: frame_hdr.is_inter_or_switch(),
            sdp: seq_hdr.sdp,
            ext_sdp: seq_hdr.ext_sdp,
            ext_partitions: seq_hdr.ext_partitions,
            uneven_4way: seq_hdr.uneven_4way_partitions,
            max_pb_aspect_ratio_log2: seq_hdr.max_pb_aspect_ratio_log2,
            n_passes,
            seg_enabled: frame_hdr.segmentation.enabled != 0,
            seg_update_map: frame_hdr.segmentation.update_map != 0,
            seg_temporal: frame_hdr.segmentation.temporal != 0,
            seg_preskip: frame_hdr.segmentation.preskip != 0,
            seg_ext: seq_hdr.segmentation.ext,
            seg_last_active_segid: frame_hdr.segmentation.last_active_segid as u8,
            seg_globalmv_mask: frame_hdr.segmentation.d.globalmv_mask,
            seg_skip_mask: frame_hdr.segmentation.d.skip_mask,
            seg_lossless: frame_hdr.segmentation.lossless,
            delta_q_present: frame_hdr.delta.q.present != 0,
            delta_q_res_log2: frame_hdr.delta.q.res_log2,
            quant_yac: frame_hdr.quant.yac as i32,
            sb128: frame_hdr.sb128 as i32,
            b4_stride: (((bw + 63) & !63) as isize),
            q_ydc_delta: frame_hdr.quant.ydc_delta as i32,
            q_uac_delta: frame_hdr.quant.uac_delta as i32,
            q_udc_delta: frame_hdr.quant.udc_delta as i32,
            q_vac_delta: frame_hdr.quant.vac_delta as i32,
            q_vdc_delta: frame_hdr.quant.vdc_delta as i32,
            seg_delta_q: frame_hdr.segmentation.d.delta_q,
            gdf_enabled: frame_hdr.gdf.enabled,
            gdf_is_key: frame_hdr.frame_type == crate::headers::FrameType::Key,
            cur_w: frame_hdr.width,
            cur_h: frame_hdr.height,
            cdef_enabled: frame_hdr.cdef.enabled != 0,
            cdef_on_skiptx: frame_hdr.cdef.on_skiptx != 0,
            cdef_n_strengths: frame_hdr.cdef.n_strengths,
            ccso_enabled: [
                frame_hdr.ccso.p[0].enabled != 0,
                frame_hdr.ccso.p[1].enabled != 0,
                frame_hdr.ccso.p[2].enabled != 0,
            ],
            ccso_sb_reuse: [
                frame_hdr.ccso.p[0].sb_reuse != 0,
                frame_hdr.ccso.p[1].sb_reuse != 0,
                frame_hdr.ccso.p[2].sb_reuse != 0,
            ],
            sb256w: (bw + 63) >> 6,
            skip_mode_enabled: frame_hdr.skip_mode_enabled != 0,
            allow_intrabc: frame_hdr.allow_intrabc != 0,
            any_lossless: frame_hdr.any_lossless != 0,
            has_chroma_layout: seq_hdr.layout != crate::headers::PixelLayout::I400,
            idtx_intra: seq_hdr.idtx_intra,
            mrls: seq_hdr.mrls,
            mhccp: seq_hdr.mhccp,
            cfl: seq_hdr.cfl,
            allow_screen_content_tools: frame_hdr.allow_screen_content_tools != 0,
            intra_dip: seq_hdr.intra_dip,
            force_integer_mv: frame_hdr.force_integer_mv != 0,
            max_bvp_drl_bits: frame_hdr.max_bvp_drl_bits,
            max_drl_bits: frame_hdr.max_drl_bits,
            bawp: frame_hdr.bawp != 0,
            txfm_switchable: frame_hdr.txfm_mode == crate::headers::TxfmMode::Switchable,
            skip_mode_refs,
            n_ref_frames: frame_hdr.n_ref_frames,
            warp_motion: frame_hdr.warp_motion != 0,
            motion_modes: frame_hdr.motion_modes,
            adaptive_mvd: seq_hdr.adaptive_mvd,
            flex_mvres: seq_hdr.flex_mvres,
            mv_precision: frame_hdr.mv_precision,
            mvd_sign_derive: seq_hdr.mvd_sign_derive,
            tip_frame_mode: frame_hdr.tip.frame_mode,
            tip_global_wtd_idx: frame_hdr.tip.global_wtd_idx,
            six_param_warp_delta: seq_hdr.six_param_warp_delta,
            subpel_filter_mode: frame_hdr.subpel_filter_mode as u8,
            switchable_comp_refs: frame_hdr.switchable_comp_refs != 0,
            num_same_ref_comp: seq_hdr.num_same_ref_comp,
            refdir,
            refdist: refdist8,
            opfl_refine_type: frame_hdr.opfl_refine_type,
            masked_compound: seq_hdr.masked_compound,
            cwp: seq_hdr.cwp,
            refine_mv_enabled: seq_hdr.refine_mv,
            absrefdist: absrefdist8,
            furthest_future_refidx,
            tip,
            tile_col_start,
            tile_col_end,
            tile_row_start,
            tile_row_end,
            sb_step,
        }
    }
}

/// Read-only per-frame reconstruction scalars (shared ref).
pub struct ReconFrameCtx<'a> {
    pub dq: &'a [[[u32; 2]; 3]; MAX_SEGMENTS],
    pub qm: &'a [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES],
    pub y_stride_px: usize,
    pub uv_stride_px: usize,
    pub ss_hor: i32,
    pub ss_ver: i32,
    pub bitdepth_max: i32,
    pub seq_fsc: bool,
    pub seq_ist: [bool; 2],
    pub seq_cctx: bool,
    pub layout: crate::headers::PixelLayout,
    // Extra frame/seq state needed by the reconstruction (coef + intra) leaf.
    pub bitdepth: u32,
    pub seg_lossless: [u8; crate::headers::MAX_SEGMENTS],
    pub reduced_txtp_set: i32,
    pub tcq: bool,
    pub seq_intra_edge_filter: bool,
    pub seq_ibp: bool,
    pub seq_inter_ddt: bool,
    /// `seq_hdr->cfl_ds_filter_index` — chroma-from-luma downsampling filter.
    pub cfl_ds_filter_index: i32,
    pub ibp_weights: [[[u8; 16]; 16]; 7],
}

std::thread_local! {
    /// Per-thread reusable coefficient scratch (one max 64x64 transform block).
    ///
    /// The inverse transform zeroes each block's coefficient region after
    /// consuming it (see `itx`), so this buffer is left entirely zero between
    /// blocks and between frames. Reusing it therefore needs no
    /// re-initialization — which removes the per-frame zeroed heap allocation
    /// (`vec![0i32; 64*64]`) that each decode worker used to make. It lives in
    /// TLS as a plain array, so there is no heap allocation at all and no
    /// `MaybeUninit`: it is value-initialized to zero once per thread.
    static CF_SCRATCH_8: core::cell::RefCell<[i16; 64 * 64]> =
        const { core::cell::RefCell::new([0; 64 * 64]) };

    static CF_SCRATCH_16: core::cell::RefCell<[i32; 64 * 64]> =
        const { core::cell::RefCell::new([0; 64 * 64]) };

    /// Per-thread reusable buffer for the pre-luma-LR luma snapshot read by the
    /// chroma single-Wiener cross-component refine. Sized to the full luma plane
    /// (so the kernel keeps indexing it by absolute offset), but only the band
    /// around the current superblock row is refreshed each row — see
    /// `filter_sbrow`. This replaces a full-plane `dst_y.to_vec()` per sbrow,
    /// which was the dominant memory-traffic cost of the filter phase.
    static LUMA_SNAP: core::cell::RefCell<Vec<u8>> =
        const { core::cell::RefCell::new(Vec::new()) };
    static LUMA_SNAP_HBD: core::cell::RefCell<Vec<u16>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, Default)]
struct LumaTxRecord {
    tx: u8,
    bx: i16,
    by: i16,
    pb_col_start: i16,
    pb_row_start: i16,
    eob: i16,
    stx: i8,
    txtp: u16,
    cf_off: u32,
    cf_len: u16,
    lossless: bool,
}

/// Coefficient scratch/replay storage selector.  `ReconScratch` is shared by
/// lowbd and highbd decode workers, so it owns both narrow and wide coefficient
/// vectors and this trait selects the one matching `BD::Coef`.
pub(crate) trait DecodeCoeff: crate::pixel::Coeff {
    fn with_cf_scratch<R>(f: impl FnOnce(&mut [Self]) -> R) -> R;
    fn scratch_chroma_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self>;
    fn scratch_luma_cf_ref(s: &ReconScratch) -> &[Self];
    fn scratch_luma_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self>;
    fn scratch_chroma_tx_cf_ref(s: &ReconScratch) -> &[Self];
    fn scratch_chroma_tx_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self>;
    fn replay_luma_cf(s: &SbReplayStore) -> &[Self];
    fn replay_luma_cf_mut(s: &mut SbReplayStore) -> &mut Vec<Self>;
    fn replay_chroma_cf(s: &SbReplayStore) -> &[Self];
    fn replay_chroma_cf_mut(s: &mut SbReplayStore) -> &mut Vec<Self>;
}

impl DecodeCoeff for i16 {
    #[inline(always)]
    fn with_cf_scratch<R>(f: impl FnOnce(&mut [Self]) -> R) -> R {
        CF_SCRATCH_8.with(|cell| {
            let mut guard = cell.borrow_mut();
            f(&mut guard[..])
        })
    }
    #[inline(always)]
    fn scratch_chroma_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self> {
        &mut s.chroma_cf8
    }
    #[inline(always)]
    fn scratch_luma_cf_ref(s: &ReconScratch) -> &[Self] {
        &s.luma_tx_cf8
    }
    #[inline(always)]
    fn scratch_luma_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self> {
        &mut s.luma_tx_cf8
    }
    #[inline(always)]
    fn scratch_chroma_tx_cf_ref(s: &ReconScratch) -> &[Self] {
        &s.chroma_tx_cf8
    }
    #[inline(always)]
    fn scratch_chroma_tx_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self> {
        &mut s.chroma_tx_cf8
    }
    #[inline(always)]
    fn replay_luma_cf(s: &SbReplayStore) -> &[Self] {
        &s.luma_tx_cf8
    }
    #[inline(always)]
    fn replay_luma_cf_mut(s: &mut SbReplayStore) -> &mut Vec<Self> {
        &mut s.luma_tx_cf8
    }
    #[inline(always)]
    fn replay_chroma_cf(s: &SbReplayStore) -> &[Self] {
        &s.chroma_tx_cf8
    }
    #[inline(always)]
    fn replay_chroma_cf_mut(s: &mut SbReplayStore) -> &mut Vec<Self> {
        &mut s.chroma_tx_cf8
    }
}

impl DecodeCoeff for i32 {
    #[inline(always)]
    fn with_cf_scratch<R>(f: impl FnOnce(&mut [Self]) -> R) -> R {
        CF_SCRATCH_16.with(|cell| {
            let mut guard = cell.borrow_mut();
            f(&mut guard[..])
        })
    }
    #[inline(always)]
    fn scratch_chroma_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self> {
        &mut s.chroma_cf32
    }
    #[inline(always)]
    fn scratch_luma_cf_ref(s: &ReconScratch) -> &[Self] {
        &s.luma_tx_cf32
    }
    #[inline(always)]
    fn scratch_luma_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self> {
        &mut s.luma_tx_cf32
    }
    #[inline(always)]
    fn scratch_chroma_tx_cf_ref(s: &ReconScratch) -> &[Self] {
        &s.chroma_tx_cf32
    }
    #[inline(always)]
    fn scratch_chroma_tx_cf_mut(s: &mut ReconScratch) -> &mut Vec<Self> {
        &mut s.chroma_tx_cf32
    }
    #[inline(always)]
    fn replay_luma_cf(s: &SbReplayStore) -> &[Self] {
        &s.luma_tx_cf32
    }
    #[inline(always)]
    fn replay_luma_cf_mut(s: &mut SbReplayStore) -> &mut Vec<Self> {
        &mut s.luma_tx_cf32
    }
    #[inline(always)]
    fn replay_chroma_cf(s: &SbReplayStore) -> &[Self] {
        &s.chroma_tx_cf32
    }
    #[inline(always)]
    fn replay_chroma_cf_mut(s: &mut SbReplayStore) -> &mut Vec<Self> {
        &mut s.chroma_tx_cf32
    }
}

/// One deferred chroma residual block.
#[derive(Clone, Copy)]
struct ChromaTxRecord {
    cbx: i16,
    cby: i16,
    cbs: u8,
    sdp_active: bool,
    n_tu: u16,
    cf_off: u32,
    cf_len: u32,
    u_has_cf: i32,
    txtp: [[u16; 2]; 256],
    eob: [[i16; 2]; 256],
}

impl Default for ChromaTxRecord {
    fn default() -> Self {
        Self {
            cbx: 0,
            cby: 0,
            cbs: BlockSize::Invalid as u8,
            sdp_active: false,
            n_tu: 0,
            cf_off: 0,
            cf_len: 0,
            u_has_cf: 0,
            txtp: [[0u16; 2]; 256],
            eob: [[-1i16; 2]; 256],
        }
    }
}

/// One parsed leaf block.  This is enough for partition/block replay once the
/// residual readers have queued their coefficient payloads.
#[derive(Clone, Copy, Default)]
struct BlockRecord {
    b: Av2Block,
    bx: i16,
    by: i16,
    cbx: i16,
    cby: i16,
    lbs: i8,
    cbs: i8,
}

/// A complete replay payload for one root superblock.
#[derive(Clone, Default)]
pub(crate) struct SbReplayStore {
    part: Vec<u8>,
    block_rec: Vec<BlockRecord>,
    luma_tx: Vec<LumaTxRecord>,
    luma_tx_cf8: Vec<i16>,
    luma_tx_cf32: Vec<i32>,
    chroma_tx: Vec<ChromaTxRecord>,
    chroma_tx_cf8: Vec<i16>,
    chroma_tx_cf32: Vec<i32>,
}

impl SbReplayStore {
    #[inline]
    fn clear(&mut self) {
        self.part.clear();
        self.block_rec.clear();
        self.luma_tx.clear();
        self.luma_tx_cf8.clear();
        self.luma_tx_cf32.clear();
        self.chroma_tx.clear();
        self.chroma_tx_cf8.clear();
        self.chroma_tx_cf32.clear();
    }

    /// Move the just-parsed replay payload out of the worker-local scratch.
    /// Heap allocations are preserved on the destination between calls.
    #[inline]
    fn capture_from<C: DecodeCoeff>(&mut self, part_w: &[u8], scratch: &mut ReconScratch) {
        self.clear();
        self.part.extend_from_slice(part_w);

        self.block_rec.append(&mut scratch.block_rec);
        self.luma_tx.append(&mut scratch.luma_tx);
        C::replay_luma_cf_mut(self).append(C::scratch_luma_cf_mut(scratch));
        self.chroma_tx.append(&mut scratch.chroma_tx);
        C::replay_chroma_cf_mut(self).append(C::scratch_chroma_tx_cf_mut(scratch));

        scratch.block_rpos = 0;
        scratch.luma_tx_rpos = 0;
        scratch.chroma_tx_rpos = 0;
    }

    /// Prepare a worker-local scratch for a reconstruction replay.  This clones
    /// the compact metadata/coefficient payload for now; the scheduler-facing
    /// version can replace this with a move once each `SbReplayStore` is owned by
    /// exactly one recon task.
    #[inline]
    fn load_into<C: DecodeCoeff>(&self, scratch: &mut ReconScratch) {
        scratch.block_rec.clear();
        scratch.block_rec.extend_from_slice(&self.block_rec);

        scratch.luma_tx.clear();
        scratch.luma_tx.extend_from_slice(&self.luma_tx);
        C::scratch_luma_cf_mut(scratch).clear();
        C::scratch_luma_cf_mut(scratch).extend_from_slice(C::replay_luma_cf(self));

        scratch.chroma_tx.clear();
        scratch.chroma_tx.extend_from_slice(&self.chroma_tx);
        C::scratch_chroma_tx_cf_mut(scratch).clear();
        C::scratch_chroma_tx_cf_mut(scratch).extend_from_slice(C::replay_chroma_cf(self));

        scratch.reset_replay_cursors();
    }
}

/// `Dav2dTaskContext` used by the luma recon leaf). `is_coded[0]` tracks the
/// 64x64 grid of decoded luma tx blocks for top-right / bottom-left availability.
pub struct ReconScratch {
    pub is_coded: [[u64; 64]; 2],
    /// SDP semi-decoupled partitioning: when the luma-only tree decodes a block
    /// it records its intra direction mode (and FSC flag) into this 16x16 map
    /// (indexed `(by & 15) * 16 + (bx & 15)`), which the chroma-only tree reads
    pub luma_intra_dir_mode_map: [u8; 256],
    pub luma_fsc_map: [u8; 256],
    /// Chroma coefficient / metadata storage used to split the chroma decode of
    /// a >64px block into a coef-read phase (with the first luma 64x64) and a
    /// recon phase (with the last), mirroring the `cbs_stage` mechanism in
    pub chroma_cf8: Vec<i16>,
    pub chroma_cf32: Vec<i32>,
    pub chroma_txtp: [[u16; 2]; 256],
    pub chroma_eob: [[i16; 2]; 256],
    pub chroma_u_has_cf: i32,
    /// Deferred chroma residual payloads for entropy/recon replay.
    chroma_tx: Vec<ChromaTxRecord>,
    chroma_tx_cf8: Vec<i16>,
    chroma_tx_cf32: Vec<i32>,
    chroma_tx_rpos: usize,
    /// Parsed block replay stream.
    block_rec: Vec<BlockRecord>,
    block_rpos: usize,
    /// Deferred luma transform payloads for future entropy/recon split.
    /// `luma_tx` records order; `luma_tx_cf` owns the corresponding coefficient
    /// slices. `luma_tx_rpos` is the replay cursor used by ReconOnly.
    luma_tx: Vec<LumaTxRecord>,
    luma_tx_cf8: Vec<i16>,
    luma_tx_cf32: Vec<i32>,
    luma_tx_rpos: usize,
    /// Per-4x4 luma transform type map (full txtp incl. secondary-tx bits),
    /// indexed `(by & 15) * 16 + (bx & 15)`. Written by the luma residual walk
    pub txtp_map: [u16; 256],
    /// TIP per-8x8 refined-MV grid (`t->rmv`): a 16x16 grid (one entry per 8x8
    /// unit within a 128x128 superblock) of `[2 versions][2 refs]` MVs. Written
    /// by the luma `tip_pred` (version 0 = luma MV post-refine, version 1 =
    /// chroma-scaled MV) and read back by the chroma `rmv_uvpred`. Indexed
    /// `((by & 31) >> 1) * 16 + ((bx & 31) >> 1)`.
    pub rmv: [[[crate::levels::Mv; 2]; 2]; 256],
    /// Above/left palette-colour cache (`t->al_pal`): `[a/l][bx4|by4][8 colours]`,
    /// indexed by `bx & 63` (above) / `by & 63` (left). Written by the palette
    /// recon (`copy_pal_block_y`) and read by `read_pal_plane` to build the
    /// per-block palette colour cache. Never explicitly reset — gating is via the
    pub al_pal: [[[u16; 8]; 64]; 2],
    /// Current palette block's colour list (`t->scratch.pal`, 8 entries). Filled
    /// by `read_pal_plane` during parse and consumed by the palette recon fill.
    pub pal: [u16; 8],
    /// Current palette block's packed index map (`t->scratch.pal_idx_y`). `pack`
    /// stores two indices per byte; sized for the largest palette block (64x64).
    pub pal_idx_y: Box<[u8; 64 * 64]>,
    /// Reusable coefficient-level neighbour map for `decode_coefs` (worst case
    /// 33*33 = 1089). Only the prefix actually used by the current transform is
    /// cleared per block, so small TUs avoid a full 1089-byte memset.
    pub coef_levels: [i8; 1089],
    /// Reusable inverse-transform scratch (`Txfm2d` buffer). Threaded into
    /// `inv_txfm_add` so the transform path needs no thread-local / `RefCell`.
    pub itx_tmp: Box<[i32; crate::itx_2d::ITX_TMP_PIXELS]>,
}

impl Default for ReconScratch {
    fn default() -> Self {
        Self {
            is_coded: [[0u64; 64]; 2],
            luma_intra_dir_mode_map: [0u8; 256],
            luma_fsc_map: [0u8; 256],
            chroma_cf8: Vec::new(),
            chroma_cf32: Vec::new(),
            chroma_txtp: [[0u16; 2]; 256],
            chroma_eob: [[-1i16; 2]; 256],
            chroma_u_has_cf: 0,
            chroma_tx: Vec::new(),
            chroma_tx_cf8: Vec::new(),
            chroma_tx_cf32: Vec::new(),
            chroma_tx_rpos: 0,
            block_rec: Vec::new(),
            block_rpos: 0,
            luma_tx: Vec::new(),
            luma_tx_cf8: Vec::new(),
            luma_tx_cf32: Vec::new(),
            luma_tx_rpos: 0,
            txtp_map: [0u16; 256],
            rmv: [[[Mv::default(); 2]; 2]; 256],
            al_pal: [[[0u16; 8]; 64]; 2],
            pal: [0u16; 8],
            pal_idx_y: Box::new([0u8; 64 * 64]),
            coef_levels: [0i8; 1089],
            itx_tmp: Box::new([0i32; crate::itx_2d::ITX_TMP_PIXELS]),
        }
    }
}

impl ReconScratch {
    #[inline]
    pub(crate) fn take_chroma_cf<C: DecodeCoeff>(&mut self) -> Vec<C> {
        core::mem::take(C::scratch_chroma_cf_mut(self))
    }

    #[inline]
    pub(crate) fn put_chroma_cf<C: DecodeCoeff>(&mut self, v: Vec<C>) {
        *C::scratch_chroma_cf_mut(self) = v;
    }

    #[inline]
    pub(crate) fn chroma_tx_cf<C: DecodeCoeff>(&self) -> &[C] {
        // SAFETY-free dispatch through the concrete trait impl.
        C::scratch_chroma_tx_cf_ref(self)
    }

    #[inline]
    pub(crate) fn chroma_tx_cf_mut<C: DecodeCoeff>(&mut self) -> &mut Vec<C> {
        C::scratch_chroma_tx_cf_mut(self)
    }

    #[inline]
    pub(crate) fn luma_tx_cf<C: DecodeCoeff>(&self) -> &[C] {
        C::scratch_luma_cf_ref(self)
    }

    #[inline]
    pub(crate) fn luma_tx_cf_mut<C: DecodeCoeff>(&mut self) -> &mut Vec<C> {
        C::scratch_luma_cf_mut(self)
    }

    #[inline]
    fn reset_for_sbrow(&mut self) {
        self.is_coded = [[0u64; 64]; 2];
        self.luma_intra_dir_mode_map = [0u8; 256];
        self.luma_fsc_map = [0u8; 256];
        self.chroma_txtp = [[0u16; 2]; 256];
        self.chroma_eob = [[-1i16; 2]; 256];
        self.chroma_u_has_cf = 0;
        self.chroma_tx.clear();
        self.chroma_tx_cf8.clear();
        self.chroma_tx_cf32.clear();
        self.chroma_tx_rpos = 0;
        self.block_rec.clear();
        self.block_rpos = 0;
        self.luma_tx.clear();
        self.luma_tx_cf8.clear();
        self.luma_tx_cf32.clear();
        self.luma_tx_rpos = 0;
        self.txtp_map = [0u16; 256];
        self.rmv = [[[Mv::default(); 2]; 2]; 256];
        self.al_pal = [[[0u16; 8]; 64]; 2];
        self.pal = [0u16; 8];
        self.coef_levels = [0i8; 1089];
        self.chroma_cf8.clear();
        self.chroma_cf32.clear();
        self.pal_idx_y.fill(0);
        // `itx_tmp` is fully overwritten on the used prefix.
    }

    #[inline]
    fn reset_replay_cursors(&mut self) {
        self.block_rpos = 0;
        self.luma_tx_rpos = 0;
        self.chroma_tx_rpos = 0;
        self.is_coded = [[0u64; 64]; 2];
    }
}

fn make_refmv_tile(ra_len: usize, bw: i32, bh: i32) -> refmvs::Tile {
    refmvs::Tile {
        rp_proj_off: 0,
        rp_traj_off: 0,
        ra: vec![refmvs::Block::default(); ra_len.max(1)],
        ra_off: 0,
        ra_tl: refmvs::Block::default(),
        r: Box::new([refmvs::Block::default(); 64 * 128]),
        tile_col: refmvs::TileRange { start: 0, end: bw },
        tile_row: refmvs::TileRange { start: 0, end: bh },
        bank: refmvs::MvBank {
            mv: [[[Mv::default(); 2]; 4]; 9],
            cwp_idx: [[0; 4]; 3],
            r#ref: [RefPair::default(); 4],
            size: [0; 9],
            idx: [0; 9],
            hits: [0; 2],
            avail: 0,
        },
        warp: refmvs::WarpBank {
            mat: [[[0; 6]; 4]; 7],
            warp_type: [[0; 4]; 7],
            hits: 0,
            size: [0; 7],
            idx: [0; 7],
        },
    }
}

fn prepare_refmv_tile(rt: &mut refmvs::Tile, ra_len: usize, bw: i32, bh: i32) {
    if rt.ra.len() != ra_len.max(1) {
        rt.ra.resize(ra_len.max(1), refmvs::Block::default());
    }
    rt.rp_proj_off = 0;
    rt.rp_traj_off = 0;
    rt.ra_off = 0;
    rt.ra_tl = refmvs::Block::default();
    rt.tile_col = refmvs::TileRange { start: 0, end: bw };
    rt.tile_row = refmvs::TileRange { start: 0, end: bh };
    rt.bank.size = [0; 9];
    rt.bank.idx = [0; 9];
    rt.bank.hits = [0; 2];
    rt.bank.avail = 0;
    rt.warp.size = [0; 7];
    rt.warp.idx = [0; 7];
    rt.warp.hits = 0;
}

struct DecodeWorkerScratch {
    rt: refmvs::Tile,
    recon_scratch: ReconScratch,
    part_w: Vec<u8>,
    ccso_tmp_buf: Vec<u8>,
    ccso_tmp_buf_hbd: Vec<u16>,
}

impl DecodeWorkerScratch {
    fn new(ra_len: usize, bw: i32, bh: i32) -> Self {
        Self {
            rt: make_refmv_tile(ra_len, bw, bh),
            recon_scratch: ReconScratch::default(),
            part_w: Vec::new(),
            ccso_tmp_buf: Vec::new(),
            ccso_tmp_buf_hbd: Vec::new(),
        }
    }

    fn prepare(&mut self, ra_len: usize, bw: i32, bh: i32) {
        prepare_refmv_tile(&mut self.rt, ra_len, bw, bh);
        self.part_w.clear();
    }
}

std::thread_local! {
    /// Per-pool-worker decode scratch that used to allocate/zero ~512 KiB of
    /// ref-MV state plus recon scratch every frame.
    static WORKER_SCRATCH: core::cell::RefCell<Option<DecodeWorkerScratch>> =
        const { core::cell::RefCell::new(None) };
}
/// Mutable reconstruction borrows bundled so only one new param threads through
/// decode_sb's recursion (Rust auto-reborrows &mut ReconCtx at each call).
pub struct ReconCtx<'a, 'f, BD: BitDepth> {
    pub bd: BD,
    pub dst_y: &'a mut [BD::Pixel],
    pub dst_u: &'a mut [BD::Pixel],
    pub dst_v: &'a mut [BD::Pixel],
    pub cdf_coef: &'a mut crate::cdf::CdfCoefContext,
    pub cf: &'a mut [BD::Coef],
    pub frame: &'a ReconFrameCtx<'f>,
    /// built once per frame; consumed by the compound + interintra recon.
    pub masks: &'f crate::wedge::Masks,
    /// Per-superblock recon scratch (is_coded grid). Reset by the caller at each
    /// superblock boundary, mirroring the C `memset(t->is_coded, ...)`.
    pub scratch: &'a mut ReconScratch,
    /// Temporary edge buffer for `prepare_intra_edges` (`t->scratch.edge`,
    /// 257 entries wide; we use a generous fixed slab indexed from the middle).
    pub edge: &'a mut [BD::Pixel],
    /// Current-frame segment id map (`f->cur_segmap`), `b4_stride * bh` entries.
    /// Written by decode_b over the block footprint when segmentation is enabled.
    pub cur_segmap: &'a mut [u8],
    /// Previous-frame segment map (`f->prev_segmap`), present only when the frame
    /// has a primary reference; `None` for frame-0 / no-primary-ref keyframes.
    pub prev_segmap: Option<&'a [u8]>,
    /// `f->b4_stride` — row stride of the segment maps (in 4x4 units).
    pub b4_stride: isize,
    /// Chroma segment-id map (`f->lf.segmap_uv`), written per chroma block when
    /// segmentation + chroma deblock are on; consumed by the deblock UV pass.
    pub segmap_uv: &'a mut [u8],
    /// `f->lf.uv_segmap_stride`; 0 when `segmap_uv` is unused.
    pub segmap_uv_stride: isize,
    /// Running per-superblock quantizer index (`ts->last_qidx`). Updated by the
    /// per-SB delta-q parse; seeded from `frame_hdr.quant.yac` at tile start.
    pub last_qidx: i32,
    /// Per-superblock recomputed dequant tables (`ts->dqmem`).
    pub dqmem: [[[u32; 2]; 3]; crate::headers::MAX_SEGMENTS],
    /// Currently active dequant tables (`ts->dq`): either the frame-wide
    /// `recon.frame.dq` or `dqmem` when a per-SB delta-q shifts the qindex.
    pub dq_active: [[[u32; 2]; 3]; crate::headers::MAX_SEGMENTS],
    /// Set when a parsed seg_id is out of range (`seg_id >= 16`), mirroring the
    /// C `return -1` that aborts the frame.
    pub seg_id_err: bool,
    /// Loop-filter mask array (`f->lf.mask` / per-SB `Av2Filter`) — the gdf,
    /// cdef-index and ccso reads write into it and read neighbor SB values.
    pub lf_mask: &'a mut [crate::lf_mask::Av2Filter],
    /// Index of the current superblock's `Av2Filter` within `lf_mask`.
    pub lf_idx: usize,
    /// `f->sb256w` — superblock-row stride into `lf_mask` (for top neighbor).
    pub sb256w: i32,
    /// Current-frame CCSO map (`f->cur_ccsomap`), written per-SB; empty if unused.
    pub cur_ccsomap: &'a mut [u8],
    /// Previous-frame CCSO maps (`f->prev_ccsomap`); `None` per plane if absent.
    pub prev_ccsomap: [Option<&'a [u8]>; 3],
    /// Per-tile reference-MV state (`t->rt`): spatial `r` grid + above-row `ra` +
    /// MV/warp banks. Maintained via splat for every block so IntraBC block
    pub rt: &'a mut refmvs::Tile,
    /// Frame-level reference-MV state (`f->rf`): iw4/ih4/sbsz + header refs.
    pub rf: &'a refmvs::Frame,
    /// Current-frame temporal MV grid (`f->mvs` == `rf.rp`), written by the inter
    /// splat so later frames can reference these MVs. Empty when ref_frame_mvs is
    /// disabled. Held separately from `rf` (which is shared immutably) so the
    /// per-block temporal save can mutate it.
    pub cur_mvs: &'a mut [refmvs::TemporalBlock],
    /// Per-reference picture pixel planes for inter motion compensation
    /// (`f->refp[i].p`). `None` for refs the current frame does not use.
    pub refp: &'a [Option<std::sync::Arc<crate::picture::Picture>>; 7],
    /// Per-reference scaling parameters (`f->svc[i]`); scale==0 means unscaled.
    pub svc: &'a [[ScalableMotionParams; 2]; 7],
    /// Inter-chroma `u_has_cf` flag (`t->u_has_cf`): set by the U plane's coef
    /// decode, consumed by the V plane's context (decode_coefs). Per chroma block.
    pub scratch_u_has_cf: i32,
    /// Sequence/frame headers needed by `refmvs_find` for IntraBC candidates.
    pub seq_hdr: &'a crate::headers::SequenceHeader,
    pub frm_hdr: &'a FrameHeader,
    /// Per-block derived warp motion parameters (`t->warpmv[0..2]`), computed in
    /// the inter MV-resolution step and consumed by `recon_b_inter`'s warp MC.
    pub warpmv: [crate::headers::WarpedMotionParams; 2],
    /// Above-superblock-row edge context snapshot (`t->a_sb_cache`). Taken at the
    /// start of each superblock from the above-row block context, before the
    /// current SB's blocks overwrite it. Inter warp/single-ref/has_cs_ext context
    /// derivations consult this (at 8x8 resolution) for blocks at the SB top edge,
    /// because the 8x8 rounding can otherwise read `a` entries already clobbered
    pub a_sb_cache: crate::env::SBEdgeCtx,
    /// Per-plane block-adaptive weighted prediction state (`t->pb.bawp[plane]` =
    /// alpha/beta). The luma plane derives alpha/beta from neighbour templates;
    /// chroma reuses the luma alpha. Sub-blocks of a >64px partition reuse the
    pub bawp_ab: [(i32, i32); 3],
}

/// Arguments for one tile/superblock-row entropy/reconstruction pass.
///
/// This used to be a 30+ argument function. Keeping the frequently-mutated
/// frame/tile state in one named packet makes the call sites audit-able without
/// changing the hot decode path after inlining.
pub(crate) struct DecodeTileSbrowEntropyCtx<'a, 'm, 'f, BD: BitDepth, const UPDATE_CDF: bool> {
    pub(crate) bd: BD,
    pub(crate) fi: &'a SbFrameInfo,
    pub(crate) frame_hdr: &'a FrameHeader,
    pub(crate) ts: &'a mut crate::internal::TileState,
    pub(crate) msac: &'a mut MsacContext<'m, UPDATE_CDF>,
    pub(crate) a_arr: &'a mut [BlockContext],
    pub(crate) lf_mask: &'a mut [crate::lf_mask::Av2Filter],
    pub(crate) lr_mask: &'a mut [crate::lf_mask::Av2Restoration],
    pub(crate) l: &'a mut BlockContext,
    pub(crate) dst_y: &'a mut [BD::Pixel],
    pub(crate) dst_u: &'a mut [BD::Pixel],
    pub(crate) dst_v: &'a mut [BD::Pixel],
    pub(crate) cf: &'a mut [BD::Coef],
    pub(crate) recon_scratch: &'a mut ReconScratch,
    pub(crate) recon_edge: &'a mut [BD::Pixel; 2048],
    pub(crate) recon_frame: &'a ReconFrameCtx<'f>,
    pub(crate) cur_segmap: &'a mut [u8],
    pub(crate) prev_segmap: Option<&'a [u8]>,
    pub(crate) segmap_uv: &'a mut [u8],
    pub(crate) segmap_uv_stride: isize,
    pub(crate) cur_ccsomap: &'a mut [u8],
    pub(crate) prev_ccsomap: [Option<&'a [u8]>; 3],
    pub(crate) part_w: &'a mut Vec<u8>,
    pub(crate) part_r: &'a [u8],
    pub(crate) by: i32,
    pub(crate) sb256w: i32,
    pub(crate) pass: u8,
    pub(crate) root_bs: BlockSize,
    pub(crate) c_root_bs: BlockSize,
    pub(crate) rt: &'a mut refmvs::Tile,
    pub(crate) rf: &'a refmvs::Frame,
    pub(crate) cur_mvs: &'a mut [refmvs::TemporalBlock],
    pub(crate) refp: &'a [Option<std::sync::Arc<crate::picture::Picture>>; 7],
    pub(crate) svc: &'a [[ScalableMotionParams; 2]; 7],
    pub(crate) seq_hdr: &'a crate::headers::SequenceHeader,
    pub(crate) frm_hdr: &'a FrameHeader,
    pub(crate) masks: &'a crate::wedge::Masks,
}

/// Decode one superblock row of a tile (entropy/parse pass).
///
/// for the single-pass, single-thread case. Reads per-superblock restoration
/// info and CDEF index reset, then drives `decode_sb` across the row.
pub(crate) fn decode_tile_sbrow_entropy<BD: BitDepth, const UPDATE_CDF: bool>(
    ctx: DecodeTileSbrowEntropyCtx<'_, '_, '_, BD, UPDATE_CDF>,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let DecodeTileSbrowEntropyCtx {
        bd,
        fi,
        frame_hdr,
        ts,
        msac,
        a_arr,
        lf_mask,
        lr_mask,
        l,
        dst_y,
        dst_u,
        dst_v,
        cf,
        recon_scratch,
        recon_edge,
        recon_frame,
        cur_segmap,
        prev_segmap,
        segmap_uv,
        segmap_uv_stride,
        cur_ccsomap,
        prev_ccsomap,
        part_w,
        part_r,
        by,
        sb256w,
        pass,
        root_bs,
        c_root_bs,
        rt,
        rf,
        cur_mvs,
        refp,
        svc,
        seq_hdr,
        frm_hdr,
        masks,
    } = ctx;
    let sb_step = fi.sb_step;
    let sb256y = by >> 6;
    let tile_row = ts.tiling.row;
    let col_start = ts.tiling.col_start;
    let col_end = ts.tiling.col_end;
    let row_start = ts.tiling.row_start;
    let row_end = ts.tiling.row_end;

    // Per-worker reusable reconstruction scratch.
    recon_scratch.reset_for_sbrow();

    let mut sb_last_qidx = ts.last_qidx;
    let mut sb_dqmem = ts.dqmem;
    let mut sb_seg_err = false;

    let refmvs_active = fi.allow_intrabc || fi.is_inter_or_switch;
    if refmvs_active {
        crate::refmvs::tile_sbrow_init(
            rt,
            rf,
            col_start,
            col_end,
            row_start,
            row_end,
            by >> 6,
            tile_row,
        );
    }
    let is_key_or_intra = frame_hdr.is_key_or_intra();

    let mut bx = col_start;
    while bx < col_end {
        let a_idx = (tile_row * sb256w + (bx >> 6)) as usize;
        let lf_idx = ((bx >> 6) + sb256y * sb256w) as usize;

        // Reset is_coded for this superblock (luma + chroma rows).
        for row in recon_scratch.is_coded.iter_mut() {
            row.fill(0);
        }

        // reset_context path); marks all 4x4 units intra (invalid mv) so that
        // not-yet-decoded neighbours are skipped by refmvs_find.
        if refmvs_active {
            crate::refmvs::reset_sb(
                rt,
                sb_step,
                seq_hdr.refmv_bank,
                is_key_or_intra,
                frm_hdr.tip.frame_mode,
                by,
                bx,
            );
        }

        // frame_mode==2 whole-frame TIP superblocks are not entropy decoded
        // loop-restoration info read, synthesizing a single TIP block per SB
        // instead. Those two reads only apply to the entropy/decode_sb path.
        let is_tip_frame = frm_hdr.tip.frame_mode == 2;

        // Reset CDEF indices for this superblock's coverage in the lf mask.
        if !is_tip_frame {
            match root_bs {
                BlockSize::Bs64x64 => {
                    let idx = (((bx & 0x30) >> 4) + ((by & 0x30) >> 2)) as usize;
                    lf_mask[lf_idx].cdef_idx[idx] = -1;
                }
                BlockSize::Bs128x128 => {
                    let idx = (((bx & 32) >> 4) + ((by & 32) >> 2)) as usize;
                    lf_mask[lf_idx].cdef_idx[idx] = -1;
                    lf_mask[lf_idx].cdef_idx[idx + 1] = -1;
                    lf_mask[lf_idx].cdef_idx[idx + 4] = -1;
                    lf_mask[lf_idx].cdef_idx[idx + 5] = -1;
                }
                BlockSize::Bs256x256 => {
                    for k in 0..16 {
                        lf_mask[lf_idx].cdef_idx[k] = -1;
                    }
                }
                _ => {}
            }
        }

        // Per-plane loop-restoration unit info.
        let sbsz = sb_step * 4;
        if !is_tip_frame {
            for p in 0..3 {
                let (ss_ver, ss_hor) = if p == 0 {
                    (0, 0)
                } else {
                    (fi.ss_ver, fi.ss_hor)
                };
                let rtype_u8 = frame_hdr.restoration.p[p].restoration_type;
                if rtype_u8 == RestorationType::None as u8 {
                    continue;
                }
                // Restoration is active for this plane, so the LR mask is
                // allocated for valid streams. Guard against an empty mask (a
                // degenerate frame from a malformed header) before indexing it.
                if lr_mask.is_empty() || lr_mask[0].lr[p].is_empty() {
                    continue;
                }
                let tx = (4 * (bx - col_start)) >> ss_hor;
                let ty = (4 * (by - row_start)) >> ss_ver;
                let unit_sz_log2 = frame_hdr.restoration.unit_size[(p != 0) as usize] as i32;
                let unit_sz = 1i32 << unit_sz_log2;
                let mask = unit_sz - 1;
                if (tx | ty) & mask != 0 {
                    continue;
                }
                let tw = (col_end * 4) >> ss_hor;
                let th = (row_end * 4) >> ss_ver;
                let half_unit = unit_sz >> 1;
                let fx = (4 * bx) >> ss_hor;
                let fy = (by * 4) >> ss_ver;
                if (ty != 0 && fy + half_unit > th) || (tx != 0 && fx + half_unit > tw) {
                    continue;
                }

                let frame_type = match rtype_u8 {
                    1 => RestorationType::PcWiener,
                    2 => RestorationType::NsWiener,
                    3 => RestorationType::Switchable,
                    _ => RestorationType::None,
                };

                let sbw = sbsz >> ss_hor;
                let sbh = sbsz >> ss_ver;
                let lruw = imax(1, imin(tw - fx + half_unit, sbw) >> unit_sz_log2);
                let lruh = imax(1, imin(th - fy + half_unit, sbh) >> unit_sz_log2);
                let vsh = unit_sz_log2 - 7 + ss_ver;
                let hsh = unit_sz_log2 - 7 + ss_hor;
                // unit_sz_log2 can be 6, giving a negative shift. The shift is a
                // no-op when the corresponding loop count is 1 (x/_y == 0). Guard
                // against the negative-shift overflow a malformed unit size can
                let shl = |v: i32, s: i32| if s >= 0 { v << s } else { v >> (-s) };
                let mut sb_idx = (by >> 6) * sb256w + (bx >> 6);
                let start_unit_idx = (((by & 0x30) >> 2) + ((bx & 0x30) >> 4)) as usize;

                for _y in 0..lruh {
                    for x in 0..lruw {
                        // For valid streams these indices are always in range;
                        // clamp so a malformed unit-size/geometry can't index the
                        // LR mask out of bounds (the entropy read still happens, so
                        // MSAC stays in sync). No-op for valid input.
                        let unit_idx = start_unit_idx.min(lr_mask[0].lr[p].len() - 1);
                        let lr_slot =
                            ((sb_idx + shl(x, hsh)).max(0) as usize).min(lr_mask.len() - 1);
                        let ns_plane = &frame_hdr.restoration.p[p].ns;
                        read_restoration_info(
                            msac,
                            &mut ts.cdf.m,
                            &mut ts.ns_wiener_bank[p],
                            &mut lr_mask[lr_slot].lr[p][unit_idx],
                            p,
                            frame_type,
                            ns_plane,
                        );
                    }
                    sb_idx += shl(sb256w, vsh);
                }
            }
        } // end if !is_tip_frame (loop-restoration info read)
        let mut dir = 0i32;
        let mut sdp_cfl_disallowed = 0i32;
        let mut intra_region = 0i32;
        let mut bx_m = bx;
        let mut by_m = by;
        let mut cbx = bx;
        let mut cby = by;
        let mut part_w_idx = 0usize;
        let mut part_r_idx = 0usize;

        // Active dequant tables for this superblock: frame-wide unless a prior
        // delta-q has shifted the running qindex away from `quant.yac`.
        let dq_active_init = if sb_last_qidx == fi.quant_yac {
            *recon_frame.dq
        } else {
            sb_dqmem
        };

        let mut recon = ReconCtx {
            bd,
            dst_y: &mut *dst_y,
            dst_u: &mut *dst_u,
            dst_v: &mut *dst_v,
            cdf_coef: &mut ts.cdf.coef,
            cf: &mut *cf,
            frame: recon_frame,
            masks,
            scratch: &mut *recon_scratch,
            edge: &mut recon_edge[..],
            cur_segmap: &mut *cur_segmap,
            prev_segmap,
            b4_stride: fi.b4_stride,
            segmap_uv: &mut *segmap_uv,
            segmap_uv_stride,
            last_qidx: sb_last_qidx,
            dqmem: sb_dqmem,
            dq_active: dq_active_init,
            seg_id_err: false,
            lf_mask: &mut *lf_mask,
            lf_idx,
            sb256w,
            cur_ccsomap: &mut cur_ccsomap[..],
            prev_ccsomap: [prev_ccsomap[0], prev_ccsomap[1], prev_ccsomap[2]],
            rt: &mut *rt,
            rf,
            cur_mvs: &mut *cur_mvs,
            refp,
            svc,
            scratch_u_has_cf: 0,
            seq_hdr,
            frm_hdr,
            warpmv: [crate::headers::WarpedMotionParams::default(); 2],
            a_sb_cache: crate::env::SBEdgeCtx::default(),
            bawp_ab: [(256, 0); 3],
        };

        // Snapshot the above-row block context into `a_sb_cache` before the
        // single-ref / has_cs_ext context derivations read this at the SB top
        // edge at 8x8 resolution. `ref` is always copied; `motion_mode` only
        // when there is a superblock above (otherwise the SB-top path is gated
        // off by `have_top`).
        if fi.is_inter_or_switch {
            let a_src = &a_arr[a_idx];
            recon.a_sb_cache.r#ref[0].copy_from_slice(&a_src.r#ref[0]);
            recon.a_sb_cache.r#ref[1].copy_from_slice(&a_src.r#ref[1]);
            if by > row_start {
                recon
                    .a_sb_cache
                    .motion_mode
                    .copy_from_slice(&a_src.motion_mode);
            }
        }

        if is_tip_frame {
            tip_frame_recon_sb(
                &mut recon,
                msac,
                &mut ts.cdf.m,
                &mut a_arr[a_idx],
                l,
                bx,
                by,
                root_bs,
                c_root_bs,
                fi,
            )?;
        } else {
            let intra_split_capable = !fi.is_inter_or_switch && !fi.allow_intrabc;
            let wants_split =
                intra_split_capable && (fi.n_passes > 1 || pass != crate::internal::PASS_ALL);

            if wants_split {
                let do_entropy = (pass & crate::internal::Pass::Entropy as u8) != 0;
                let do_recon = (pass & crate::internal::Pass::Recon as u8) != 0;

                // External split tasks will call this function once with
                // `pass=Entropy` and later with `pass=Recon`.  The old local
                // scaffold (`pass=PASS_ALL`, `n_passes>1`) still uses the same
                // owned store so both paths exercise identical replay payloads.
                let mut local_replay = SbReplayStore::default();

                if do_entropy {
                    {
                        let mut sb_ctx = SbCtx {
                            fi,
                            bx: &mut bx_m,
                            by: &mut by_m,
                            cbx: &mut cbx,
                            cby: &mut cby,
                            intra_region: &mut intra_region,
                            sdp_cfl_disallowed: &mut sdp_cfl_disallowed,
                            a: &mut a_arr[a_idx],
                            l: &mut *l,
                            msac: &mut *msac,
                            cdf_m: &mut ts.cdf.m,
                            cdf_dmv: &mut ts.cdf.dmv,
                            part_w: &mut *part_w,
                            part_w_idx: &mut part_w_idx,
                            part_r,
                            part_r_idx: &mut part_r_idx,
                        };
                        decode_sb(
                            &mut sb_ctx,
                            &mut recon,
                            crate::internal::Pass::Entropy as u8,
                            root_bs,
                            c_root_bs,
                            &mut dir,
                        )?;
                    }

                    if do_recon {
                        local_replay.capture_from::<BD::Coef>(part_w, recon.scratch);
                    }
                }

                if do_recon {
                    if do_entropy {
                        local_replay.load_into::<BD::Coef>(recon.scratch);
                    } else {
                        // A real scheduler replay will have preloaded the scratch
                        // from its tile/sbrow-owned `SbReplayStore` before entering
                        // this pass.  Keep the cursor reset here so replay-only
                        // unit tests can call the pass directly after filling
                        // `recon.scratch`.
                        recon.scratch.reset_replay_cursors();
                    }

                    bx_m = bx;
                    by_m = by;
                    cbx = bx;
                    cby = by;
                    intra_region = 0;
                    sdp_cfl_disallowed = 0;
                    part_r_idx = 0;
                    let mut replay_part_w: Vec<u8> = Vec::new();
                    let mut replay_part_w_idx = 0usize;
                    let mut recon_dir = 0i32;
                    let replay_part_r: &[u8] = if do_entropy {
                        &local_replay.part[..]
                    } else {
                        part_r
                    };
                    let mut sb_ctx = SbCtx {
                        fi,
                        bx: &mut bx_m,
                        by: &mut by_m,
                        cbx: &mut cbx,
                        cby: &mut cby,
                        intra_region: &mut intra_region,
                        sdp_cfl_disallowed: &mut sdp_cfl_disallowed,
                        a: &mut a_arr[a_idx],
                        l: &mut *l,
                        msac: &mut *msac,
                        cdf_m: &mut ts.cdf.m,
                        cdf_dmv: &mut ts.cdf.dmv,
                        part_w: &mut replay_part_w,
                        part_w_idx: &mut replay_part_w_idx,
                        part_r: replay_part_r,
                        part_r_idx: &mut part_r_idx,
                    };
                    decode_sb(
                        &mut sb_ctx,
                        &mut recon,
                        crate::internal::Pass::Recon as u8,
                        root_bs,
                        c_root_bs,
                        &mut recon_dir,
                    )?;
                }
            } else {
                if pass != crate::internal::PASS_ALL {
                    // Partial passes for inter/IntraBC are not safe until the MVRES
                    // replay path is wired.  Fail loudly rather than re-reading
                    // entropy or reconstructing with unresolved motion state.
                    return Err(());
                }
                let mut sb_ctx = SbCtx {
                    fi,
                    bx: &mut bx_m,
                    by: &mut by_m,
                    cbx: &mut cbx,
                    cby: &mut cby,
                    intra_region: &mut intra_region,
                    sdp_cfl_disallowed: &mut sdp_cfl_disallowed,
                    a: &mut a_arr[a_idx],
                    l: &mut *l,
                    msac: &mut *msac,
                    cdf_m: &mut ts.cdf.m,
                    cdf_dmv: &mut ts.cdf.dmv,
                    part_w: &mut *part_w,
                    part_w_idx: &mut part_w_idx,
                    part_r,
                    part_r_idx: &mut part_r_idx,
                };
                decode_sb(
                    &mut sb_ctx,
                    &mut recon,
                    crate::internal::PASS_ALL,
                    root_bs,
                    c_root_bs,
                    &mut dir,
                )?;
            }
        }

        // Persist running delta-q state for the next superblock / sbrow.
        sb_last_qidx = recon.last_qidx;
        sb_dqmem = recon.dqmem;
        sb_seg_err |= recon.seg_id_err;

        // Save THIS superblock's bottom row into the above-row `ra` buffer (and
        // its top-left backup `ra_tl`) for the next SB row's neighbor access.
        // range so that `ra_tl` captures the correct per-SB top-left value; a
        // once-per-row save would leave `ra_tl` reflecting only the row's far
        // right edge and corrupt the SB-boundary top-left MV candidate.
        if refmvs_active {
            let crate::refmvs::Tile {
                r,
                ra,
                ra_tl,
                ra_off,
                ..
            } = &mut *recon.rt;
            crate::refmvs::save_tmvs(
                r,
                &mut ra[*ra_off..],
                ra_tl,
                bx >> 1,
                (bx + sb_step) >> 1,
                row_start >> 1,
                (by + sb_step) >> 1,
                rf.ih8,
                rf.iw8,
            );
        }

        bx += sb_step;
    }

    // Write back the running per-tile delta-q state.
    ts.last_qidx = sb_last_qidx;
    ts.dqmem = sb_dqmem;

    // Abort the frame on an out-of-range segment id (C `return -1`).
    if sb_seg_err {
        return Err(());
    }

    // Error out on symbol-decoder overread. frame_mode==2 whole-frame TIP frames
    // carry no entropy data — their (empty) symbol decoder is never advanced, so
    if frm_hdr.tip.frame_mode != 2 && msac.cnt() <= -15 {
        return Err(());
    }

    Ok(())
}

/// Minimal shared-mutable view used only by the parallel tile-row decode below.
struct DisjointMut<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Clone for DisjointMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DisjointMut<T> {}

// SAFETY: the wrapper is just a `(ptr, len)` pair; sharing it across threads is
// sound provided the disjointness invariant documented above is upheld by the
// caller (the tile-row partition).
unsafe impl<T: Send> Send for DisjointMut<T> {}
unsafe impl<T: Send> Sync for DisjointMut<T> {}

impl<T> DisjointMut<T> {
    fn new(s: &mut [T]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }
    /// SAFETY: the caller must only access indices that are disjoint from every
    /// other concurrent caller (guaranteed here by the tile-row partition).
    #[allow(clippy::mut_from_ref)]
    unsafe fn whole(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

pub fn decode_frame_main(
    fc: &mut crate::internal::FrameContext,
    n_passes: i32,
    n_tc: i32,
    pool: Option<&crate::mtpool::ThreadPool>,
) -> Result<(), ()> {
    if fc.frame_hdr.disable_cdf_update != 0 {
        decode_frame_main_inner::<false>(fc, n_passes, n_tc, pool)
    } else {
        decode_frame_main_inner::<true>(fc, n_passes, n_tc, pool)
    }
}

fn decode_frame_main_inner<const UPDATE_CDF: bool>(
    fc: &mut crate::internal::FrameContext,
    n_passes: i32,
    n_tc: i32,
    pool: Option<&crate::mtpool::ThreadPool>,
) -> Result<(), ()> {
    let crate::internal::FrameContext {
        seq_hdr,
        frame_hdr,
        a,
        ts,
        lf,
        sb256w,
        sb_step,
        root_bs,
        bw,
        bh,
        refdir,
        refdist,
        absrefdist,
        furthest_future_refidx,
        skip_mode_refs,
        cur_pic,
        dq,
        qm,
        bitdepth_max,
        ss_hor,
        ss_ver,
        cur_segmap,
        prev_segmap,
        cur_ccsomap,
        prev_ccsomap,
        b4_stride,
        sb256h,
        sbh,
        rf,
        inloop_filters,
        refp,
        mvs,
        ref_mvs,
        refrefpoc,
        refcnt,
        refpoc,
        svc,
        ..
    } = fc;
    let fc_sb256h = *sb256h;
    let fc_sbh = *sbh;
    let fc_inloop_filters = *inloop_filters;

    let seq_hdr = &**seq_hdr;
    let frame_hdr = &**frame_hdr;
    let root_bs = *root_bs;
    let sb256w = *sb256w;
    let sb_step = *sb_step;
    let bw = *bw;
    let bh = *bh;
    let b4_stride_v = *b4_stride;

    if frame_hdr.segmentation.enabled != 0 {
        let needed = (b4_stride_v as usize) * 64 * (fc_sb256h as usize);
        if cur_segmap.len() != needed {
            cur_segmap.resize(needed, 0);
        }
        cur_segmap.fill(0);
    }
    let prev_segmap_ref: Option<&[u8]> = prev_segmap.as_deref();
    let prev_ccsomap_ref: [Option<&[u8]>; 3] = [
        prev_ccsomap[0].as_deref(),
        prev_ccsomap[1].as_deref(),
        prev_ccsomap[2].as_deref(),
    ];
    let refdir = *refdir;
    let skip_mode_refs = *skip_mode_refs;

    let ss_hor_v = *ss_hor;
    let ss_ver_v = *ss_ver;
    let bitdepth_max_v = *bitdepth_max;
    // The plane allocation is sized for the 128-aligned frame dimensions (see
    // DefaultPicAllocator::alloc_picture: `aligned_h = (h + 127) & !127`), which
    // gives bottom padding past the cropped/visible height. Reconstruction
    // legitimately writes whole transform blocks that overhang the visible edge
    // padded). Span the slices over the *allocated* height so those overhang
    // writes stay in bounds rather than panicking on the cropped height.
    let aligned_h: usize = ((cur_pic.p.h.max(0) as usize) + 127) & !127;
    let y_h: usize = aligned_h;
    let uv_h: usize = if seq_hdr.layout == crate::headers::PixelLayout::I400 {
        0
    } else {
        aligned_h >> ss_ver_v
    };
    // Byte strides as allocated; the recon path indexes planes in *samples*, so
    // `*_stride_px` below is the per-sample stride (byte stride / bytes-per-sample).
    let y_stride_bytes: usize = cur_pic.stride[0].unsigned_abs();
    let uv_stride_bytes: usize = cur_pic.stride[1].unsigned_abs();
    // bitdepth in bits from bitdepth_max (255 -> 8, 1023 -> 10, 4095 -> 12).
    let bitdepth_v: u32 = (crate::intops::ulog2((bitdepth_max_v + 1) as u32)) as u32;
    let bytes_per_sample: usize = if bitdepth_v > 8 { 2 } else { 1 };
    let y_stride_px: usize = y_stride_bytes / bytes_per_sample;
    let uv_stride_px: usize = uv_stride_bytes / bytes_per_sample;
    // `f->seq_hdr->hbd` (0 for 8bpc, 1 for 10/12bpc); used by deblock thresholds.
    let hbd_v: i32 = (bitdepth_v > 8) as i32;
    // once per frame for the compound + interintra recon paths.
    let masks = crate::wedge::masks();
    let recon_frame = ReconFrameCtx {
        dq: &*dq,
        qm: &*qm,
        y_stride_px,
        uv_stride_px,
        ss_hor: ss_hor_v,
        ss_ver: ss_ver_v,
        bitdepth_max: bitdepth_max_v,
        seq_fsc: seq_hdr.fsc,
        seq_ist: seq_hdr.ist,
        seq_cctx: seq_hdr.cctx,
        layout: seq_hdr.layout,
        bitdepth: bitdepth_v,
        seg_lossless: frame_hdr.segmentation.lossless,
        reduced_txtp_set: frame_hdr.reduced_txtp_set as i32,
        tcq: frame_hdr.tcq != 0,
        seq_intra_edge_filter: seq_hdr.intra_edge_filter,
        seq_ibp: seq_hdr.ibp,
        seq_inter_ddt: seq_hdr.inter_ddt,
        cfl_ds_filter_index: seq_hdr.cfl_ds_filter_index as i32,
        ibp_weights: *crate::ibp::ibp_weights(),
    };

    let c_root_bs = if seq_hdr.layout == crate::headers::PixelLayout::I400 {
        BlockSize::Invalid
    } else {
        root_bs
    };
    let cols = frame_hdr.tiling.t.cols as i32;
    let rows = frame_hdr.tiling.t.rows as i32;
    let keyframe = frame_hdr.is_key_or_intra();
    let is_tip = frame_hdr.tip.frame_mode == 2;

    // multi-threaded reset lives in decode_frame_init. Without this, the above
    // neighbour `midx`/mode arrays retain default 0 instead of the 0xff "no
    // neighbour" sentinel, corrupting intra-mode context derivation.
    if n_tc <= 1 {
        let n_a = (sb256w * rows) as usize;
        for ctx in a.iter_mut().take(n_a) {
            reset_context(ctx, keyframe, is_tip);
        }
    }

    let mut l = BlockContext::default();

    // Initialise the reference-MV frame state (`f->rf`) and a per-tile working
    // Tile. For IntraBC on an intra frame only the spatial grid / above-row /
    // banks are needed (no temporal candidates). For inter frames the reference
    // temporal MVs are wired and (when use_ref_frame_mvs) projected per sbrow.
    let allow_intrabc = frame_hdr.allow_intrabc != 0;
    let is_inter_or_switch = frame_hdr.is_inter_or_switch();
    let refmvs_active = allow_intrabc || is_inter_or_switch;
    if refmvs_active {
        refmvs::init_frame(
            rf, seq_hdr, frame_hdr, refpoc, refrefpoc, refcnt, ref_mvs, false, false,
        );
    }

    // ref_frame_mvs is enabled, sized for the whole frame; the per-block splat
    // (splat_oneref_mv's `t_dst`) writes decoded MVs into `rf.rp` so later frames
    let _ = &mvs;
    if refmvs_active && seq_hdr.ref_frame_mvs {
        let needed = (fc_sb256h as usize) * 32 * ((b4_stride_v >> 1) as usize);
        if rf.rp.len() != needed {
            rf.rp = vec![refmvs::TemporalBlock::default(); needed];
        } else {
            rf.rp.fill(refmvs::TemporalBlock::default());
        }
    } else {
        rf.rp.clear();
    }

    // The projection of reference temporal MVs into `rf.rp_proj`
    // path uses a rolling-window projection buffer (`rp_proj_off` is a fixed
    // 2-row top margin, not a per-sbrow base), so each sbrow's load_tmvs
    // overwrites the previous sbrow's projection; batching all rows up front
    // would leave only the last row's data and corrupt the temporal MV grid
    // that the inter tmvp candidates and frame_mode=2 whole-frame TIP recon read.
    let need_load_tmvs = is_inter_or_switch && frame_hdr.use_ref_frame_mvs != 0;
    // Hold the current-frame temporal MV grid separately so the inter splat can
    // mutate it while `rf` itself stays shared immutably (refmvs_find reads it).
    let mut cur_mvs: Vec<refmvs::TemporalBlock> = std::mem::take(&mut rf.rp);
    // Reference pixel planes for inter MC, shared from the FrameContext refp.
    let refp_pics: [Option<std::sync::Arc<crate::picture::Picture>>; 7] =
        std::array::from_fn(|i| refp[i].pic.clone());
    let svc_v = *svc;

    // Compound `comp_type`/`get_compref_ctx` neighbour contexts need the
    // few scalar `rf` fields needed before the loop so `rf` can be re-borrowed
    // mutably (for the per-sbrow load_tmvs) inside it.
    let ffr_idx = *furthest_future_refidx;
    let tip_ref = rf.tip.r#ref;
    let rf_rp_stride = rf.rp_stride;

    // Precompute the per-frame filter parameters once (deblock thresholds, CDEF
    // strength decomposition, etc.) so the per-superblock-row filter pass can run
    // without re-deriving them. Filters are gated by `fc.inloop_filters`.
    let filter_params = FilterFrameParams::new(
        seq_hdr,
        frame_hdr,
        ss_hor_v,
        ss_ver_v,
        cur_pic.stride[0],
        cur_pic.stride[1],
        bitdepth_v as i32,
        fc_inloop_filters,
    );
    let sb128 = frame_hdr.sb128 as i32;

    // for every superblock-row in the tile row, THEN PHASE B runs the deferred
    // filter pass `for sby: filter_sbrow(sby)`.Filters must run only after a
    // whole tile row has decoded because intrabc reads pre-filter pixels to the
    // left within the tile row.
    macro_rules! run_decode_tile_rows {
        ($bd:expr, $Pixel:ty, $Coef:ty) => {{
            let bd_local = $bd;
            let estimated_decode_units = (((bh + sb_step - 1) / sb_step).max(0) as usize)
                .saturating_mul(cols.max(0) as usize);
            let estimated_filter_bands = (bh >> 4).max(0) as usize;
            let do_parallel = (n_tc as usize) >= 2
                && !need_load_tmvs
                && (estimated_decode_units > 1 || estimated_filter_bands > 1);
            let active_pool = pool;
            let (dst_y, dst_u, dst_v): (&mut [$Pixel], &mut [$Pixel], &mut [$Pixel]) = cur_pic
                .plane_slices_rows3_mut::<$Pixel>(
                y_h,
                uv_h,
                seq_hdr.layout != crate::headers::PixelLayout::I400,
            );
            if do_parallel {
                // One persistent dispatch replaces the old decode-barrier-filter
                // two-phase. Every worker runs the same loop and pulls whichever
                // unit is available, preferring filter so that as decode marches
                // top-to-bottom the loop-filter trails a couple of rows behind it
                // on spare workers instead of waiting on a global barrier:
                //   (1) a *ready* filter tile-row — ready once its own and its
                //       vertically-adjacent rows (tr-1,tr,tr+1) have finished
                //       decoding, so every seam pixel/mask entry it reads exists;
                //   (2) else the next decode tile (claimed row-major);
                //   (3) else PARK on a condvar until a decode completion (or the
                //       final filter completion) signals progress. Workers never
                //       busy-spin — the spin is exactly what regressed the earlier
                //       fused attempt, its progress-poll traffic competing with the
                //       decode critical path.
                let n_tiles = (rows as usize) * (cols as usize);
                let rows_us = rows as usize;
                let max_workers = (n_tc as usize).max(1);
                let dst_y_dm = DisjointMut::new(dst_y);
                let dst_u_dm = DisjointMut::new(dst_u);
                let dst_v_dm = DisjointMut::new(dst_v);
                let a_dm = DisjointMut::new(&mut a[..]);
                let filter_params = &filter_params;
                let cseg_dm = DisjointMut::new(&mut cur_segmap[..]);
                let cccso_dm = DisjointMut::new(&mut cur_ccsomap[..]);
                let cmvs_dm = DisjointMut::new(&mut cur_mvs[..]);
                let ts_dm = DisjointMut::new(&mut ts[..]);
                let recon_frame = &recon_frame;
                let refp_pics = &refp_pics;
                let svc_v = &svc_v;
                let rf_imm: &refmvs::Frame = &*rf;
                let prev_ccsomap_ref = prev_ccsomap_ref;
                let refdist: &[i8; 7] = refdist;
                let absrefdist: &[u8; 7] = absrefdist;

                // Split `lf` into disjoint field borrows: masks are written by
                // decode (and the idempotent bottom-edge crop) through DisjointMut;
                // everything else is read-only for the whole dispatch.
                let lf = &mut *lf;
                let uv_segmap_stride = lf.uv_segmap_stride;
                let base_q = lf.base_q;
                let gdf_ref_dst_idx = lf.gdf_ref_dst_idx;
                let wiener_idx = lf.wiener_idx;
                let ns_subclass_class_idx = lf.ns_subclass_class_idx;
                let restore_planes = lf.restore_planes;
                let lf_start_of_tile_row: &[u8] = &lf.start_of_tile_row;
                let lf_lr_cdef_line: &[Vec<u8>; 3] = &lf.lr_cdef_line;
                let lf_lr_cdef_line_hbd: &[Vec<u16>; 3] = &lf.lr_cdef_line_hbd;
                let mask_dm = DisjointMut::new(&mut lf.mask[..]);
                let lrmask_dm = DisjointMut::new(&mut lf.lr_mask[..]);
                let seguv_dm =
                    DisjointMut::new(std::sync::Arc::make_mut(&mut lf.segmap_uv).as_mut_slice());
                let mask_dm = &mask_dm;
                let lrmask_dm = &lrmask_dm;
                let seguv_dm = &seguv_dm;

                let got_err = std::sync::atomic::AtomicBool::new(false);
                let got_err = &got_err;

                let cols_us = cols as usize;
                let tile_nsb: Vec<usize> = {
                    let ts_ref = unsafe { ts_dm.whole() };
                    (0..n_tiles)
                        .map(|t| {
                            let tb = &ts_ref[t].tiling;
                            (((tb.row_end - tb.row_start) + sb_step - 1) / sb_step).max(0) as usize
                        })
                        .collect()
                };
                let tile_nsb = &tile_nsb[..];
                let sb64h_us = ((bh + 15) >> 4).max(0) as usize;
                let by64_tile_row: Vec<usize> = {
                    let ts_ref = unsafe { ts_dm.whole() };
                    let mut map = vec![0usize; sb64h_us];
                    for tr in 0..rows_us {
                        let tb = &ts_ref[tr * cols_us].tiling;
                        let by64_start = ((tb.row_start + 15) >> 4).max(0) as usize;
                        let by64_end = ((tb.row_end + 15) >> 4).max(0) as usize;
                        for slot in map
                            .iter_mut()
                            .take(by64_end.min(sb64h_us))
                            .skip(by64_start.min(sb64h_us))
                        {
                            *slot = tr;
                        }
                    }
                    map
                };
                let by64_tile_row = &by64_tile_row[..];
                let n_decode_units = tile_nsb.iter().copied().sum::<usize>();
                let n_filter_units = sb64h_us;
                // Bands the filter actually processes. The serial path filters
                // `by64 in [row_rs>>4, (row_re>>4).min(sb64h))` per tile row; the union
                // over rows ends at the bottom tile's `row_re = bh`, i.e. `bh>>4`
                // (floor). When `bh` is not a multiple of 16 there is a partial bottom
                // sb64 band (index `sb64h-1`) that the serial filter does NOT process
                // as its own slice — its few rows are covered by the last full band's
                // bottom-edge handling. `n_filter_units` (= `sb64h`) still counts every
                // band for decode coverage, but the filter loop and its completion
                // accounting must use `n_filter_bands` so MT filters exactly the serial
                // set (otherwise MT filters one extra band, which silently matches on
                // some content but diverges by up to the loop-filter delta on others).
                let n_filter_bands = ((bh >> 4).max(0) as usize).min(n_filter_units);
                let n_workers = max_workers
                    .min(n_decode_units.max(n_filter_units).max(1))
                    .max(1);
                let tile_sbrow: Vec<std::sync::atomic::AtomicUsize> = (0..n_tiles)
                    .map(|_| std::sync::atomic::AtomicUsize::new(0))
                    .collect();
                let tile_sbrow = &tile_sbrow[..];
                let tile_busy: Vec<std::sync::atomic::AtomicBool> = (0..n_tiles)
                    .map(|_| std::sync::atomic::AtomicBool::new(false))
                    .collect();
                let tile_busy = &tile_busy[..];
                let dec_remaining: Vec<std::sync::atomic::AtomicUsize> = (0..rows_us)
                    .map(|_| std::sync::atomic::AtomicUsize::new(cols as usize))
                    .collect();
                let dec_remaining = &dec_remaining[..];
                // Dav2d-style fine-grained decode progress for filter pipelining.
                // `dec_remaining` above is still kept for the conservative IntraBC
                // path, where an entire tile-row must remain unfiltered until all
                // possible intra-block-copy users in that tile-row have decoded.
                // When IntraBC is off, a filter sb64 band only needs the current
                // band and its top neighbour decoded across every tile column.
                let sb64_dec_remaining: Vec<std::sync::atomic::AtomicUsize> = (0..n_filter_units)
                    .map(|_| std::sync::atomic::AtomicUsize::new(cols_us))
                    .collect();
                let sb64_dec_remaining = &sb64_dec_remaining[..];
                let granular_filter_ready = !allow_intrabc;
                let flt_done = std::sync::atomic::AtomicUsize::new(0);
                let flt_done = &flt_done;
                // Filter scheduler.
                //   deblock      : single owner, in-order        -> deblock_progress
                //   CDEF SAVE    : single owner, in-order (cheap)-> cdef_save_progress
                //   CDEF FILTER  : data-parallel rows            -> cdef_filter_claim/_done
                //   LR           : data-parallel rows            -> lr_claim/flt_done
                let deblock_rows_busy = std::sync::atomic::AtomicBool::new(false);
                let deblock_rows_busy = &deblock_rows_busy;
                let deblock_cols_claim = std::sync::atomic::AtomicUsize::new(0);
                let deblock_cols_claim = &deblock_cols_claim;
                let cols_done: Vec<std::sync::atomic::AtomicBool> = (0..n_filter_bands)
                    .map(|_| std::sync::atomic::AtomicBool::new(false))
                    .collect();
                let cols_done = &cols_done;
                let deblock_progress = std::sync::atomic::AtomicUsize::new(0);
                let deblock_progress = &deblock_progress;
                let cdef_save_progress = std::sync::atomic::AtomicUsize::new(0);
                let cdef_save_progress = &cdef_save_progress;
                let cdef_save_claim = std::sync::atomic::AtomicUsize::new(0);
                let cdef_save_claim = &cdef_save_claim;
                // Per-band CDEF-SAVE completion. SAVE is data-parallel (each band
                // writes disjoint seam slots: cdef_line[k] and cdef_top one-past, which
                // save(k+1) skips), so it finishes out of order; cdef-filter(k) needs
                // save(k-1) and save(k) (it reads cdef_line[k-1] and cdef_top[k+1]).
                let cdef_save_done: Vec<std::sync::atomic::AtomicBool> = (0..n_filter_bands)
                    .map(|_| std::sync::atomic::AtomicBool::new(false))
                    .collect();
                let cdef_save_done = &cdef_save_done;
                let cdef_filter_claim = std::sync::atomic::AtomicUsize::new(0);
                let cdef_filter_claim = &cdef_filter_claim;
                let cdef_filter_done = std::sync::atomic::AtomicUsize::new(0);
                let cdef_filter_done = &cdef_filter_done;
                let lr_claim = std::sync::atomic::AtomicUsize::new(0);
                let lr_claim = &lr_claim;
                // Per-band CDEF-FILTER completion flags. CDEF-filter rows finish out
                // of order, so LR (which needs its own root's CDEF complete) checks
                // these rather than the done count, letting LR overlap CDEF-filter
                // instead of waiting at a full-CDEF barrier.
                let cdef_band_done: Vec<std::sync::atomic::AtomicBool> = (0..n_filter_bands)
                    .map(|_| std::sync::atomic::AtomicBool::new(false))
                    .collect();
                let cdef_band_done = &cdef_band_done;
                // `filter_progress` retained as the LR completion cursor for
                // park/wake accounting (== flt_done after the LR pass).
                let filter_progress = std::sync::atomic::AtomicUsize::new(0);
                let filter_progress = &filter_progress;
                // Per-frame seam line buffers, pre-sized so the parallel passes never
                // resize. Written by the single-owner SAVE/deblock passes, read by the
                // parallel FILTER/LR passes; the two phases are separated in time by
                // the cursors, so DisjointMut shared access is sound.
                let n_roots_alloc = ((n_filter_bands >> sb128) + 2).max(1);
                let mono_alloc = seq_hdr.layout == crate::headers::PixelLayout::I400;
                ensure_filter_lines(
                    &mut lf.cdef_line_store,
                    &mut lf.cdef_top_store,
                    &mut lf.lr_db_store,
                    bh,
                    filter_params.y_stride,
                    filter_params.uv_stride,
                    n_roots_alloc,
                    mono_alloc,
                );
                if bd_local.bitdepth() > 8 {
                    ensure_filter_lines_hbd(
                        &mut lf.cdef_line_store_hbd,
                        &mut lf.cdef_top_store_hbd,
                        &mut lf.lr_db_store_hbd,
                        bh,
                        filter_params.y_stride,
                        filter_params.uv_stride,
                        n_roots_alloc,
                        mono_alloc,
                    );
                }
                let cdef_line_dm = DisjointMut::new(&mut lf.cdef_line_store[..]);
                let cdef_line_dm = &cdef_line_dm;
                let cdef_top_dm = DisjointMut::new(&mut lf.cdef_top_store[..]);
                let cdef_top_dm = &cdef_top_dm;
                let lr_db_dm = DisjointMut::new(&mut lf.lr_db_store[..]);
                let lr_db_dm = &lr_db_dm;
                let cdef_line_hbd_dm = DisjointMut::new(&mut lf.cdef_line_store_hbd[..]);
                let cdef_line_hbd_dm = &cdef_line_hbd_dm;
                let cdef_top_hbd_dm = DisjointMut::new(&mut lf.cdef_top_store_hbd[..]);
                let cdef_top_hbd_dm = &cdef_top_hbd_dm;
                let lr_db_hbd_dm = DisjointMut::new(&mut lf.lr_db_store_hbd[..]);
                let lr_db_hbd_dm = &lr_db_hbd_dm;
                let park_mx = std::sync::Mutex::new(());
                let park_mx = &park_mx;
                let park_cv = std::sync::Condvar::new();
                let park_cv = &park_cv;
                let park_waiters = std::sync::atomic::AtomicUsize::new(0);
                let park_waiters = &park_waiters;
                // A single global "something progressed" counter. Every producer
                // wake site bumps it (Release) after publishing its work, so an idle
                // worker can detect that progress happened with ONE relaxed-ish
                // atomic load instead of re-scanning all `n_tiles + n_filter_units`
                // per-unit atomics each spin iteration (that scan bounces O(units)
                // cache lines that other workers are actively writing, which is the
                // dominant source of MT timing noise on many-core targets). The
                // epoch is only a spin/scan-avoidance hint: the authoritative park
                // decision is still the full predicate recheck under `park_mx`, so a
                // missed bump can at worst cost a redundant scan, never a lost wakeup.
                let progress_epoch = std::sync::atomic::AtomicU64::new(0);
                let progress_epoch = &progress_epoch;
                // Number of cheap epoch-poll spins before falling back to yield then
                // pthread_cond_wait. Each spin is now a single atomic load (not an
                // O(tiles+units) scan), so this trades a little busy-wait for avoiding
                // the high-variance OS condvar wakeup when work lands a few hundred
                // cycles later. Tunable; measure on the target.
                const PARK_SPIN_ITERS: usize = 96;
                let wake_one = || {
                    // Publish "progress happened" before (possibly) signalling, so a
                    // spinning worker that loads the new epoch also observes the work.
                    progress_epoch.fetch_add(1, std::sync::atomic::Ordering::Release);
                    // Avoid the pthread mutex/condvar path when all workers are
                    // currently running. A waiter increments `park_waiters` before
                    // its final predicate recheck under `park_mx`, so skipping here
                    // cannot lose a wakeup: either the waiter will see the freshly
                    // published atomic state during that final recheck, or it will
                    // have become visible in `park_waiters` before this notify.
                    if park_waiters.load(std::sync::atomic::Ordering::Relaxed) != 0 {
                        let _guard = park_mx.lock().unwrap_or_else(|e| e.into_inner());
                        park_cv.notify_one();
                    }
                };
                let wake_one = &wake_one;
                let wake_all = || {
                    progress_epoch.fetch_add(1, std::sync::atomic::Ordering::Release);
                    if park_waiters.load(std::sync::atomic::Ordering::Relaxed) != 0 {
                        let _guard = park_mx.lock().unwrap_or_else(|e| e.into_inner());
                        park_cv.notify_all();
                    }
                };
                let wake_all = &wake_all;
                let worker_seq = std::sync::atomic::AtomicUsize::new(0);
                let worker_seq = &worker_seq;

                let worker = || {
                    use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
                    let worker_id = worker_seq.fetch_add(1, Relaxed);
                    // A filter sb64 row is ready when the owning tile-row and its
                    // vertical neighbor tile-rows are fully decoded (Acquire pairs
                    // with the decode Release).
                    let tile_row_ready = |tr: usize| -> bool {
                        let lo = tr.saturating_sub(1);
                        let hi = (tr + 1).min(rows_us - 1);
                        (lo..=hi).all(|r| dec_remaining[r].load(Acquire) == 0)
                    };
                    let sb64_decoded = |by64: usize| -> bool {
                        sb64_dec_remaining
                            .get(by64)
                            .map(|r| r.load(Acquire) == 0)
                            .unwrap_or(true)
                    };
                    let filter_ready = |by64: usize| -> bool {
                        if granular_filter_ready {
                            // Deblock rows and the root-SB CDEF seam read the top
                            // neighbor. They do not need the next band: CDEF's
                            // normal by64 slice deliberately leaves the bottom two
                            // 4x4 rows for the following task.
                            sb64_decoded(by64) && (by64 == 0 || sb64_decoded(by64 - 1))
                        } else {
                            by64_tile_row
                                .get(by64)
                                .map(|&tr| tile_row_ready(tr))
                                .unwrap_or(false)
                        }
                    };
                    <$Coef as DecodeCoeff>::with_cf_scratch(|cf| {
                        WORKER_SCRATCH.with(|ws_cell| {
                            let mut ws_guard = ws_cell.borrow_mut();
                            let ws = ws_guard.get_or_insert_with(|| {
                                DecodeWorkerScratch::new(rf_rp_stride.max(1) as usize, bw, bh)
                            });
                            ws.prepare(rf_rp_stride.max(1) as usize, bw, bh);
                            let mut rt = &mut ws.rt;
                            let mut recon_scratch = &mut ws.recon_scratch;
                            let mut part_w = &mut ws.part_w;
                            let ccso_tmp_buf = &mut ws.ccso_tmp_buf;
                            let ccso_tmp_buf_hbd = &mut ws.ccso_tmp_buf_hbd;
                            let mut l = BlockContext::default();
                            let part_r: Vec<u8> = Vec::new();
                            let mut recon_edge = Box::new([<$Pixel as Default>::default(); 2048]);
                            let mut cached_fi: Option<(usize, SbFrameInfo)> = None;

                            loop {
                                if got_err.load(Relaxed) {
                                    return;
                                }
                                // Snapshot the progress epoch *before* the phase-1/2 work
                                // scan below.
                                let seen_epoch = progress_epoch.load(Acquire);

                                let mut claimed = None;
                                let start_t = if n_tiles != 0 { worker_id % n_tiles } else { 0 };
                                for ti in 0..n_tiles {
                                    let t = (start_t + ti) % n_tiles;
                                    if tile_sbrow[t].load(Acquire) >= tile_nsb[t] {
                                        continue;
                                    }
                                    if tile_busy[t].swap(true, AcqRel) {
                                        continue; // another worker is advancing this tile
                                    }
                                    if tile_sbrow[t].load(Acquire) >= tile_nsb[t] {
                                        tile_busy[t].store(false, Release);
                                        continue;
                                    }
                                    claimed = Some(t);
                                    break;
                                }
                                if let Some(t) = claimed {
                                    let tr = t / cols_us;
                                    let ts_idx = t;
                                    let a = unsafe { a_dm.whole() };
                                    let cur_segmap = unsafe { cseg_dm.whole() };
                                    let cur_ccsomap = unsafe { cccso_dm.whole() };
                                    let cur_mvs = unsafe { cmvs_dm.whole() };
                                    let ts = unsafe { ts_dm.whole() };
                                    let dst_y = unsafe { dst_y_dm.whole() };
                                    let dst_u = unsafe { dst_u_dm.whole() };
                                    let dst_v = unsafe { dst_v_dm.whole() };
                                    let mask = unsafe { mask_dm.whole() };
                                    let lr_mask = unsafe { lrmask_dm.whole() };
                                    let segmap_uv = unsafe { seguv_dm.whole() };
                                    let (cs, ce, rs, re) = {
                                        let tb = &ts[ts_idx].tiling;
                                        (tb.col_start, tb.col_end, tb.row_start, tb.row_end)
                                    };
                                    if cached_fi.as_ref().map_or(true, |(ct, _)| *ct != t) {
                                        cached_fi = Some((
                                            t,
                                            SbFrameInfo::from_frame(SbFrameInfoArgs {
                                                seq_hdr,
                                                frame_hdr,
                                                bw,
                                                bh,
                                                root_bs,
                                                sb_step,
                                                n_passes,
                                                refdir,
                                                refdist,
                                                absrefdist,
                                                skip_mode_refs,
                                                tile_col_start: cs,
                                                tile_col_end: ce,
                                                tile_row_start: rs,
                                                tile_row_end: re,
                                                furthest_future_refidx: ffr_idx,
                                                tip: tip_ref,
                                            }),
                                        ));
                                    }
                                    let fi = &cached_fi.as_ref().unwrap().1;

                                    // Keep advancing the same tile while it remains ready.
                                    loop {
                                        let s = tile_sbrow[t].load(Relaxed);
                                        if s >= tile_nsb[t] {
                                            tile_busy[t].store(false, Release);
                                            wake_one();
                                            break;
                                        }

                                        let by = rs + (s as i32) * sb_step;
                                        let buf = std::mem::take(&mut ts[ts_idx].msac_buf);
                                        part_w.clear();
                                        reset_context(&mut l, keyframe, is_tip);
                                        // Resume this tile's parked entropy state for sbrow `s`.
                                        let mut msac = MsacContext::<UPDATE_CDF>::resume(
                                            &buf,
                                            ts[ts_idx].msac_state,
                                        );
                                        let sbrow_res =
                                            decode_tile_sbrow_entropy(DecodeTileSbrowEntropyCtx {
                                                bd: bd_local,
                                                fi,
                                                frame_hdr,
                                                ts: &mut ts[ts_idx],
                                                msac: &mut msac,
                                                a_arr: &mut *a,
                                                lf_mask: &mut *mask,
                                                lr_mask: &mut *lr_mask,
                                                l: &mut l,
                                                dst_y: &mut *dst_y,
                                                dst_u: &mut *dst_u,
                                                dst_v: &mut *dst_v,
                                                cf: &mut *cf,
                                                recon_scratch: &mut recon_scratch,
                                                recon_edge: &mut recon_edge,
                                                recon_frame,
                                                cur_segmap: &mut cur_segmap[..],
                                                prev_segmap: prev_segmap_ref,
                                                segmap_uv: &mut *segmap_uv,
                                                segmap_uv_stride: uv_segmap_stride,
                                                cur_ccsomap: &mut cur_ccsomap[..],
                                                prev_ccsomap: prev_ccsomap_ref,
                                                part_w: &mut part_w,
                                                part_r: &part_r,
                                                by,
                                                sb256w,
                                                pass: crate::internal::PASS_ALL,
                                                root_bs,
                                                c_root_bs,
                                                rt: &mut rt,
                                                rf: rf_imm,
                                                cur_mvs: &mut *cur_mvs,
                                                refp: refp_pics,
                                                svc: svc_v,
                                                seq_hdr,
                                                frm_hdr: frame_hdr,
                                                masks,
                                            });
                                        // Park the advanced entropy state + buffer back.
                                        ts[ts_idx].msac_state = msac.save();
                                        ts[ts_idx].msac_buf = buf;
                                        if sbrow_res.is_err() {
                                            got_err.store(true, Relaxed);
                                            cf.fill(<$Coef as crate::pixel::Coeff>::ZERO);
                                            tile_busy[t].store(false, Release);
                                            wake_all();
                                            return;
                                        }

                                        let new_s = s + 1;
                                        tile_sbrow[t].store(new_s, Release);

                                        // Publish the exact sb64 bands covered by this
                                        // decoded root-sbrow. This is what lets the
                                        // filter side trail decode at sb64 granularity
                                        // instead of waiting for the whole tile-row.
                                        let row_start = by.max(0) as usize;
                                        let row_end =
                                            (by + sb_step).min(re).min(bh).max(by) as usize;
                                        let by64_start = (row_start >> 4).min(n_filter_units);
                                        let by64_end = ((row_end + 15) >> 4).min(n_filter_units);
                                        let mut made_filter_ready = false;
                                        for by64 in by64_start..by64_end {
                                            if sb64_dec_remaining[by64].fetch_sub(1, AcqRel) == 1 {
                                                made_filter_ready = true;
                                            }
                                        }

                                        if new_s >= tile_nsb[t] {
                                            tile_busy[t].store(false, Release);
                                            dec_remaining[tr].fetch_sub(1, AcqRel);
                                            wake_one();
                                            break;
                                        }
                                        if made_filter_ready {
                                            wake_one();
                                        }
                                    }
                                    continue;
                                }

                                // (2) Filter pipeline
                                let mut did_work = false;
                                macro_rules! run_stage {
                                    ($k:expr, $stages:expr) => {{
                                        let by64 = $k as i32;
                                        let cur_segmap = unsafe { cseg_dm.whole() };
                                        let dst_y = unsafe { dst_y_dm.whole() };
                                        let dst_u = unsafe { dst_u_dm.whole() };
                                        let dst_v = unsafe { dst_v_dm.whole() };
                                        if $stages
                                            & (crate::decode::STAGE_DEBLOCK
                                                | crate::decode::STAGE_DEBLOCK_COLS)
                                            != 0
                                        {
                                            let mr = ((by64 >> 2) * sb256w) as usize;
                                            crate::deblock::deblock_crop_bottom_edge(
                                                unsafe { mask_dm.whole() },
                                                mr,
                                                sb256w,
                                                bw,
                                                bh,
                                                0,
                                                by64,
                                            );
                                        }
                                        let sh = FilterShared {
                                            mask: unsafe { &*mask_dm.whole() },
                                            lr_mask: unsafe { &*lrmask_dm.whole() },
                                            segmap_uv: unsafe { &*seguv_dm.whole() },
                                            start_of_tile_row: lf_start_of_tile_row,
                                            lr_cdef_line: lf_lr_cdef_line,
                                            lr_cdef_line_hbd: lf_lr_cdef_line_hbd,
                                            uv_segmap_stride,
                                            base_q,
                                            gdf_ref_dst_idx,
                                            wiener_idx,
                                            ns_subclass_class_idx,
                                            restore_planes,
                                        };
                                        // SAFETY: SAVE/deblock (writers) and FILTER/LR
                                        // (readers) of the seam buffers are separated in
                                        // time by the stage cursors, so the DisjointMut
                                        // accesses never overlap a write with a read.
                                        filter_sb64(
                                            bd_local,
                                            crate::decode::loopfilter::FilterSb64Ctx {
                                                seq_hdr,
                                                frame_hdr,
                                                sh: &sh,
                                                fp: filter_params,
                                                cur_segmap,
                                                b4_stride: b4_stride_v,
                                                hbd: hbd_v,
                                                inloop: fc_inloop_filters,
                                                sbh: fc_sbh,
                                                sb_step,
                                                sb256w,
                                                sb128,
                                                bw,
                                                bh,
                                            },
                                            crate::decode::loopfilter::FilterSb64Scratch {
                                                cdef_line: unsafe { cdef_line_dm.whole() },
                                                cdef_top: unsafe { cdef_top_dm.whole() },
                                                lr_db_line: unsafe { lr_db_dm.whole() },
                                                ccso_tmp_buf: &mut *ccso_tmp_buf,
                                                cdef_line_hbd: unsafe { cdef_line_hbd_dm.whole() },
                                                cdef_top_hbd: unsafe { cdef_top_hbd_dm.whole() },
                                                lr_db_line_hbd: unsafe { lr_db_hbd_dm.whole() },
                                                ccso_tmp_buf_hbd: &mut *ccso_tmp_buf_hbd,
                                            },
                                            crate::decode::loopfilter::FilterSb64Dst {
                                                y: &mut *dst_y,
                                                u: &mut *dst_u,
                                                v: &mut *dst_v,
                                            },
                                            crate::decode::loopfilter::FilterSb64Band {
                                                by64,
                                                stages: $stages,
                                            },
                                        );
                                    }};
                                }

                                // Stage A0: deblock COLS (data-parallel). Vertical edges are
                                // within-band and independent, so any worker filters any
                                // decode-ready band's columns. CAS-claim ready bands.
                                loop {
                                    let k = deblock_cols_claim.load(Acquire);
                                    if k >= n_filter_bands || !filter_ready(k) {
                                        break;
                                    }
                                    if deblock_cols_claim
                                        .compare_exchange_weak(k, k + 1, AcqRel, Relaxed)
                                        .is_err()
                                    {
                                        continue;
                                    }
                                    run_stage!(k, crate::decode::STAGE_DEBLOCK_COLS);
                                    cols_done[k].store(true, Release);
                                    did_work = true;
                                    wake_one();
                                }

                                // Stage A1: deblock ROWS
                                if deblock_rows_busy
                                    .compare_exchange(false, true, AcqRel, Relaxed)
                                    .is_ok()
                                {
                                    let mut adv = false;
                                    loop {
                                        let k = deblock_progress.load(Relaxed);
                                        if k >= n_filter_bands || !cols_done[k].load(Acquire) {
                                            break;
                                        }
                                        run_stage!(k, crate::decode::STAGE_DEBLOCK_ROWS);
                                        deblock_progress.store(k + 1, Release);
                                        did_work = true;
                                        adv = true;
                                    }
                                    deblock_rows_busy.store(false, Release);
                                    if adv {
                                        wake_one();
                                    }
                                }

                                // Stage B: CDEF SAVE
                                loop {
                                    let k = cdef_save_claim.load(Acquire);
                                    if k >= n_filter_bands {
                                        break;
                                    }
                                    let dp = deblock_progress.load(Acquire);
                                    if k + 1 >= dp && dp < n_filter_bands {
                                        break;
                                    }
                                    if cdef_save_claim
                                        .compare_exchange_weak(k, k + 1, AcqRel, Relaxed)
                                        .is_err()
                                    {
                                        continue;
                                    }
                                    run_stage!(k, crate::decode::STAGE_CDEF_SAVE);
                                    cdef_save_done[k].store(true, Release);
                                    cdef_save_progress.fetch_add(1, AcqRel);
                                    did_work = true;
                                    wake_one();
                                }

                                // Stage C: CDEF FILTER
                                loop {
                                    let k = cdef_filter_claim.load(Acquire);
                                    if k >= n_filter_bands {
                                        break;
                                    }
                                    let ready = cdef_save_done[k].load(Acquire)
                                        && (k == 0 || cdef_save_done[k - 1].load(Acquire));
                                    if !ready {
                                        break;
                                    }
                                    if cdef_filter_claim
                                        .compare_exchange_weak(k, k + 1, AcqRel, Relaxed)
                                        .is_err()
                                    {
                                        continue;
                                    }
                                    run_stage!(k, crate::decode::STAGE_CDEF_FILTER);
                                    cdef_band_done[k].store(true, Release);
                                    let d = cdef_filter_done.fetch_add(1, AcqRel) + 1;
                                    did_work = true;
                                    if d >= n_filter_bands {
                                        wake_all();
                                    }
                                }

                                // Stage D: LR
                                loop {
                                    let k = lr_claim.load(Acquire);
                                    if k >= n_filter_bands {
                                        break;
                                    }
                                    let root_first = (k >> sb128) << sb128;
                                    let ready =
                                        (root_first..=k).all(|j| cdef_band_done[j].load(Acquire));
                                    if !ready {
                                        break;
                                    }
                                    if lr_claim
                                        .compare_exchange_weak(k, k + 1, AcqRel, Relaxed)
                                        .is_err()
                                    {
                                        continue;
                                    }
                                    run_stage!(k, crate::decode::STAGE_LR);
                                    filter_progress.fetch_add(1, AcqRel);
                                    let done_now = flt_done.fetch_add(1, AcqRel) + 1;
                                    did_work = true;
                                    if done_now >= n_filter_bands {
                                        wake_all();
                                    }
                                }

                                if did_work {
                                    continue;
                                }

                                // (3) Nothing claimable now.
                                if flt_done.load(Relaxed) >= n_filter_bands {
                                    return;
                                }

                                let mut should_park = true;
                                for _ in 0..PARK_SPIN_ITERS {
                                    if got_err.load(Relaxed)
                                        || flt_done.load(Relaxed) >= n_filter_bands
                                    {
                                        return;
                                    }
                                    if progress_epoch.load(Acquire) != seen_epoch {
                                        should_park = false;
                                        break;
                                    }
                                    std::hint::spin_loop();
                                }
                                if should_park {
                                    for _ in 0..2 {
                                        std::thread::yield_now();
                                        if got_err.load(Relaxed)
                                            || flt_done.load(Relaxed) >= n_filter_bands
                                        {
                                            return;
                                        }
                                        if progress_epoch.load(Acquire) != seen_epoch {
                                            should_park = false;
                                            break;
                                        }
                                    }
                                }
                                if !should_park {
                                    continue;
                                }

                                let guard = park_mx.lock().unwrap_or_else(|e| e.into_inner());
                                park_waiters.fetch_add(1, Relaxed);
                                if got_err.load(Relaxed) || flt_done.load(Relaxed) >= n_filter_bands
                                {
                                    park_waiters.fetch_sub(1, Relaxed);
                                    return;
                                }
                                let more_decode = (0..n_tiles).any(|t| {
                                    tile_sbrow[t].load(Relaxed) < tile_nsb[t]
                                        && !tile_busy[t].load(Relaxed)
                                });
                                let dp = deblock_progress.load(Acquire);
                                let deblock_cols_av = {
                                    let c = deblock_cols_claim.load(Acquire);
                                    c < n_filter_bands && filter_ready(c)
                                };
                                let deblock_rows_av = !deblock_rows_busy.load(Relaxed)
                                    && dp < n_filter_bands
                                    && cols_done.get(dp).map_or(false, |b| b.load(Acquire));
                                let cdef_save_av = {
                                    let c = cdef_save_claim.load(Acquire);
                                    c < n_filter_bands && (c + 1 < dp || dp >= n_filter_bands)
                                };
                                let cdef_filter_av = {
                                    let c = cdef_filter_claim.load(Acquire);
                                    c < n_filter_bands
                                        && cdef_save_done[c].load(Acquire)
                                        && (c == 0 || cdef_save_done[c - 1].load(Acquire))
                                };
                                let lr_av = {
                                    let k = lr_claim.load(Acquire);
                                    k < n_filter_bands && {
                                        let rf = (k >> sb128) << sb128;
                                        (rf..=k).all(|j| cdef_band_done[j].load(Acquire))
                                    }
                                };
                                let ready_filter = deblock_cols_av
                                    || deblock_rows_av
                                    || cdef_save_av
                                    || cdef_filter_av
                                    || lr_av;
                                if more_decode || ready_filter {
                                    park_waiters.fetch_sub(1, Relaxed);
                                    drop(guard);
                                    continue;
                                }
                                let guard = park_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                                park_waiters.fetch_sub(1, Relaxed);
                                drop(guard);
                            }
                        });
                    });
                };
                crate::mtpool::dispatch(active_pool, n_workers, &worker);
                if got_err.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(());
                }
            } else {
                let mut cf = vec![<$Coef as crate::pixel::Coeff>::ZERO; 64 * 64];
                let mut rt = make_refmv_tile(rf_rp_stride.max(1) as usize, bw, bh);
                let mut recon_scratch = ReconScratch::default();
                let mut recon_edge = Box::new([<$Pixel as Default>::default(); 2048]);
                for tr in 0..rows {
                    let ts_base = (tr * cols) as usize;
                    let mut bufs: Vec<Vec<u8>> = Vec::with_capacity(cols as usize);
                    let mut fis: Vec<SbFrameInfo> = Vec::with_capacity(cols as usize);
                    let mut ranges: Vec<(i32, i32)> = Vec::with_capacity(cols as usize); // (rs, re)
                    for tc in 0..cols {
                        let ts_idx = ts_base + tc as usize;
                        let (cs, ce, rs, re) = {
                            let t = &ts[ts_idx].tiling;
                            (t.col_start, t.col_end, t.row_start, t.row_end)
                        };
                        fis.push(SbFrameInfo::from_frame(SbFrameInfoArgs {
                            seq_hdr,
                            frame_hdr,
                            bw,
                            bh,
                            root_bs,
                            sb_step,
                            n_passes,
                            refdir,
                            refdist,
                            absrefdist,
                            skip_mode_refs,
                            tile_col_start: cs,
                            tile_col_end: ce,
                            tile_row_start: rs,
                            tile_row_end: re,
                            furthest_future_refidx: ffr_idx,
                            tip: tip_ref,
                        }));
                        ranges.push((rs, re));
                        bufs.push(std::mem::take(&mut ts[ts_idx].msac_buf));
                    }
                    // The buffers Vec now owns the tile data; build the symbol decoders
                    // borrowing from it (read-only) so they persist across the sby loop.
                    let mut msacs: Vec<MsacContext<'_, UPDATE_CDF>> = bufs
                        .iter()
                        .map(|b| MsacContext::<UPDATE_CDF>::new(b))
                        .collect();
                    let mut part_ws: Vec<Vec<u8>> = (0..cols).map(|_| Vec::new()).collect();
                    let part_r: Vec<u8> = Vec::new();

                    // The tile row spans the same block-row range for every tile-col (only
                    // the column range differs), so derive the sbrow loop from tile-col 0.
                    let (row_rs, row_re) = ranges[0];

                    // PHASE A: decode every superblock-row across all tile-cols.
                    let mut by = row_rs;
                    while by < row_re {
                        // Project reference temporal MVs into the rolling-window `rf.rp_proj`
                        // Run once per sbrow over the full block-row width, before any
                        // tile-col reads it, so the inter tmvp candidates and frame_mode=2
                        // whole-frame TIP recon see this row's projection.
                        if need_load_tmvs {
                            let by_end = (by + sb_step) >> 1;
                            refmvs::load_tmvs(
                                rf,
                                tr,
                                0,
                                bw >> 1,
                                by >> 1,
                                by_end,
                                seq_hdr.mv_traj,
                                frame_hdr.tip.frame_mode,
                                seq_hdr.tip_hole_fill,
                                frame_hdr.tmvp_sample_step as i32,
                                frame_hdr.n_ref_frames as i32,
                            );
                        }
                        let rf_ref: &refmvs::Frame = &*rf;
                        for tc in 0..cols as usize {
                            let ts_idx = ts_base + tc;
                            let (rs, re) = ranges[tc];
                            if by < rs || by >= re {
                                continue;
                            }
                            reset_context(&mut l, keyframe, is_tip);
                            decode_tile_sbrow_entropy(DecodeTileSbrowEntropyCtx {
                                bd: bd_local,
                                fi: &fis[tc],
                                frame_hdr,
                                ts: &mut ts[ts_idx],
                                msac: &mut msacs[tc],
                                a_arr: &mut *a,
                                lf_mask: &mut lf.mask,
                                lr_mask: &mut lf.lr_mask,
                                l: &mut l,
                                dst_y: &mut *dst_y,
                                dst_u: &mut *dst_u,
                                dst_v: &mut *dst_v,
                                cf: &mut cf,
                                recon_scratch: &mut recon_scratch,
                                recon_edge: &mut recon_edge,
                                recon_frame: &recon_frame,
                                cur_segmap: &mut cur_segmap[..],
                                prev_segmap: prev_segmap_ref,
                                segmap_uv: std::sync::Arc::make_mut(&mut lf.segmap_uv)
                                    .as_mut_slice(),
                                segmap_uv_stride: lf.uv_segmap_stride,
                                cur_ccsomap: &mut cur_ccsomap[..],
                                prev_ccsomap: prev_ccsomap_ref,
                                part_w: &mut part_ws[tc],
                                part_r: &part_r,
                                by,
                                sb256w,
                                pass: crate::internal::PASS_ALL,
                                root_bs,
                                c_root_bs,
                                rt: &mut rt,
                                rf: rf_ref,
                                cur_mvs: &mut cur_mvs[..],
                                refp: &refp_pics,
                                svc: &svc_v,
                                seq_hdr,
                                frm_hdr: frame_hdr,
                                masks,
                            })?;
                        }
                        by += sb_step;
                    }

                    // Return the MSAC buffers to the tile states.
                    drop(msacs);
                    for (tc, buf) in bufs.into_iter().enumerate() {
                        ts[ts_base + tc].msac_buf = buf;
                    }

                    // PHASE B: frame-level spec-order filter
                    if tr + 1 == rows {
                        let sb64h = (bh + 15) >> 4;
                        let nfb = ((bh >> 4).max(0) as usize).min(sb64h as usize) as i32;
                        let n_roots = ((sb64h >> sb128) + 2) as usize;
                        let mono = seq_hdr.layout == crate::headers::PixelLayout::I400;
                        ensure_filter_lines(
                            &mut lf.cdef_line_store,
                            &mut lf.cdef_top_store,
                            &mut lf.lr_db_store,
                            bh,
                            filter_params.y_stride,
                            filter_params.uv_stride,
                            n_roots,
                            mono,
                        );
                        if bd_local.bitdepth() > 8 {
                            ensure_filter_lines_hbd(
                                &mut lf.cdef_line_store_hbd,
                                &mut lf.cdef_top_store_hbd,
                                &mut lf.lr_db_store_hbd,
                                bh,
                                filter_params.y_stride,
                                filter_params.uv_stride,
                                n_roots,
                                mono,
                            );
                        }
                        for stage in [
                            crate::decode::STAGE_DEBLOCK,
                            crate::decode::STAGE_CDEF,
                            crate::decode::STAGE_LR,
                        ] {
                            for by64 in 0..nfb {
                                if stage & crate::decode::STAGE_DEBLOCK != 0 {
                                    let mask_row = ((by64 >> 2) * sb256w) as usize;
                                    crate::deblock::deblock_crop_bottom_edge(
                                        &mut lf.mask,
                                        mask_row,
                                        sb256w,
                                        bw,
                                        bh,
                                        0,
                                        by64,
                                    );
                                }
                                let sh = FilterShared {
                                    mask: &lf.mask[..],
                                    lr_mask: &lf.lr_mask[..],
                                    segmap_uv: &lf.segmap_uv[..],
                                    start_of_tile_row: &lf.start_of_tile_row[..],
                                    lr_cdef_line: &lf.lr_cdef_line,
                                    lr_cdef_line_hbd: &lf.lr_cdef_line_hbd,
                                    uv_segmap_stride: lf.uv_segmap_stride,
                                    base_q: lf.base_q,
                                    gdf_ref_dst_idx: lf.gdf_ref_dst_idx,
                                    wiener_idx: lf.wiener_idx,
                                    ns_subclass_class_idx: lf.ns_subclass_class_idx,
                                    restore_planes: lf.restore_planes,
                                };
                                filter_sb64(
                                    bd_local,
                                    crate::decode::loopfilter::FilterSb64Ctx {
                                        seq_hdr,
                                        frame_hdr,
                                        sh: &sh,
                                        fp: &filter_params,
                                        cur_segmap,
                                        b4_stride: b4_stride_v,
                                        hbd: hbd_v,
                                        inloop: fc_inloop_filters,
                                        sbh: fc_sbh,
                                        sb_step,
                                        sb256w,
                                        sb128,
                                        bw,
                                        bh,
                                    },
                                    crate::decode::loopfilter::FilterSb64Scratch {
                                        cdef_line: &mut lf.cdef_line_store,
                                        cdef_top: &mut lf.cdef_top_store,
                                        lr_db_line: &mut lf.lr_db_store,
                                        ccso_tmp_buf: &mut lf.ccso_tmp_buf,
                                        cdef_line_hbd: &mut lf.cdef_line_store_hbd,
                                        cdef_top_hbd: &mut lf.cdef_top_store_hbd,
                                        lr_db_line_hbd: &mut lf.lr_db_store_hbd,
                                        ccso_tmp_buf_hbd: &mut lf.ccso_tmp_buf_hbd,
                                    },
                                    crate::decode::loopfilter::FilterSb64Dst {
                                        y: &mut *dst_y,
                                        u: &mut *dst_u,
                                        v: &mut *dst_v,
                                    },
                                    crate::decode::loopfilter::FilterSb64Band {
                                        by64,
                                        stages: stage,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }};
    }

    if bitdepth_v > 8 {
        run_decode_tile_rows!(crate::pixel::BitDepth16::new(bitdepth_v as u8), u16, i32);
    } else {
        run_decode_tile_rows!(crate::pixel::BitDepth8, u8, i16);
    }

    // Restore the (now-populated) temporal MV grid into `rf.rp` so the
    // reference-list update can publish it to c.refs[i].refmvs.
    rf.rp = cur_mvs;

    Ok(())
}

/// Orchestrate a single-threaded frame decode: init -> CDF init -> main loop.
#[allow(clippy::too_many_arguments)]
pub fn decode_frame(
    fc: &mut crate::internal::FrameContext,
    n_tc: i32,
    n_passes: i32,
    in_cdf: Option<&crate::cdf::CdfContext>,
    qcat: usize,
    pool: Option<&crate::mtpool::ThreadPool>,
) -> Result<(), ()> {
    let frame_hdr = fc.frame_hdr.clone();
    let seq_hdr = fc.seq_hdr.clone();

    decode_frame_init(
        &frame_hdr,
        &seq_hdr,
        &mut fc.lf,
        &mut fc.ts,
        &mut fc.n_ts,
        &mut fc.a,
        &mut fc.a_sz,
        &mut fc.dq,
        &mut fc.qm,
        &fc.absrefdist,
        fc.sbh,
        fc.sb256w,
        fc.sb256h,
        fc.bw,
        fc.bh,
        n_tc,
    );

    if frame_hdr.tip.frame_mode != 2 {
        let r = decode_frame_init_cdf(
            &mut fc.ts,
            &fc.tile,
            &frame_hdr,
            in_cdf,
            qcat,
            fc.sb_shift,
            fc.bw,
            fc.bh,
            n_tc,
            pool,
        );
        r?;
    } else {
        decode_tip_frame_init(&mut fc.ts, &frame_hdr, fc.sb_shift, fc.bw, fc.bh, n_tc);
    }

    let r = decode_frame_main(fc, n_passes, n_tc, pool);
    r?;

    // path the update tile's adapted CDF becomes `out_cdf` with its symbol counts
    // reset. avg_cdf_type (tile CDF shift/accumulate) is not exercised by the
    // single-tile bring-up clips and is deferred. Only produced when CDF update
    // is enabled (otherwise refs keep `in_cdf`, handled by the ref-list update).
    if frame_hdr.tip.frame_mode != 2 && frame_hdr.disable_cdf_update == 0 {
        let upd = frame_hdr.tiling.update as usize;
        if let Some(ts) = fc.ts.get(upd) {
            let mut out = ts.cdf.clone();
            out.reset_count(frame_hdr.is_key_or_intra());
            fc.out_cdf = Some(std::sync::Arc::new(out));
        }
    }

    Ok(())
}

/// Build a `FrameContext` from the decoder context's parsed headers and tile
/// data, then run the decode.
///
/// The context is moved out for the duration of the decode (so it borrows
/// independently of `c`, e.g. `c.pool`) and always moved back afterward — even on
/// error — so its scratch allocations survive to the next frame.
pub fn submit_frame(c: &mut crate::internal::DecoderContext, n_tc: i32) -> Result<(), ()> {
    let mut fc = std::mem::take(&mut c.fc);
    let r = submit_frame_inner(c, &mut fc, n_tc);
    c.fc = fc;
    r
}

fn submit_frame_inner(
    c: &mut crate::internal::DecoderContext,
    fc: &mut crate::internal::FrameContext,
    n_tc: i32,
) -> Result<(), ()> {
    use crate::headers::PixelLayout;

    let seq_hdr = c.seq_hdr.clone().ok_or(())?;
    let frame_hdr = c.frame_hdr.clone().ok_or(())?;

    let sb128 = frame_hdr.sb128 as i32;
    let layout = seq_hdr.layout;
    fc.ss_ver = (layout == PixelLayout::I420) as i32;
    fc.ss_hor = matches!(layout, PixelLayout::I420 | PixelLayout::I422) as i32;
    fc.root_bs = match sb128 {
        0 => BlockSize::Bs64x64,
        1 => BlockSize::Bs128x128,
        _ => BlockSize::Bs256x256,
    };
    fc.bw = ((frame_hdr.width + 7) >> 3) << 1;
    fc.bh = ((frame_hdr.height + 7) >> 3) << 1;
    fc.sb256w = (fc.bw + 63) >> 6;
    fc.sb256h = (fc.bh + 63) >> 6;
    fc.sb_shift = 4 + sb128;
    fc.sb_step = 16 << sb128;
    fc.sbh = (fc.bh + fc.sb_step - 1) >> fc.sb_shift;
    fc.b4_stride = ((fc.bw + 63) & !63) as isize;
    let bpc = 8 + seq_hdr.hbd as i32 * 2;
    fc.bitdepth_max = (1 << bpc) - 1;
    // Intra neighbours have no reference direction; -1 sentinel (mirrors C
    // lib.c init) so compound/ref context lookups treat them correctly.
    fc.refdir_intra = -1;

    fc.tile = std::mem::take(&mut c.tile);
    fc.n_tile_data = c.n_tile_data;
    fc.inloop_filters = c.inloop_filters;

    // Allocate the output picture that reconstruction writes into, drawing from
    // the decoder's persistent pool allocator so a freed frame's planes are
    // recycled rather than re-allocated.
    let allocator = c.pic_allocator.clone();
    fc.cur_pic = crate::picture::Picture::alloc(
        frame_hdr.width,
        frame_hdr.height,
        layout,
        bpc,
        Some(seq_hdr.clone()),
        Some(frame_hdr.clone()),
        allocator,
    )
    .ok_or(())?;

    // Attach display metadata that was parsed from metadata OBUs before this frame.
    fc.cur_pic.content_light_level = c.content_light;

    // Attach the frame's film-grain synthesis params to the picture so the output
    if frame_hdr.film_grain.present != 0 {
        fc.cur_pic.fgm = c.fgm[frame_hdr.film_grain.id as usize];
    }

    let qcat = crate::cdf::cdf_thread_init_static_qcat(frame_hdr.quant.yac as u32) as usize;

    let is_inter_or_switch = frame_hdr.is_inter_or_switch();
    let allow_intrabc = frame_hdr.allow_intrabc != 0;

    // Validate each signalled reference, share its picture into `fc.refp[i]`,
    // and compute the per-ref scaling / global-warp-allowed flags. Single-ref
    // bring-up: scaled references (svc != 0) are validated but the scaled-MC
    // path is a follow-up; an unequal-dimension ref triggers the deferral note.
    let mut ref_coded_width = [0i32; 7];
    if is_inter_or_switch {
        for i in 0..7 {
            let refidx = frame_hdr.refidx[i] as usize;
            let rp = &c.refs[refidx].p;
            let rpic = rp.pic.as_ref();
            let valid = rpic.is_some_and(|p| {
                let pw = p.p.w;
                let ph = p.p.h;
                frame_hdr.width * 2 >= pw
                    && frame_hdr.height * 2 >= ph
                    && frame_hdr.width <= pw * 16
                    && frame_hdr.height <= ph * 16
                    && seq_hdr.layout == p.p.layout
                    && bpc == p.p.bpc
            });
            if !valid {
                return Err(());
            }
            let p = rpic.unwrap();
            fc.refp[i].pic = Some(p.clone());
            fc.refp[i].frame_hdr = rp.frame_hdr.clone();
            ref_coded_width[i] = p.frame_hdr.as_ref().map(|h| h.width).unwrap_or(p.p.w);
            if frame_hdr.width != p.p.w || frame_hdr.height != p.p.h {
                let scale_fac = |ref_sz: i32, this_sz: i32| -> i32 {
                    (((ref_sz << 14) + (this_sz >> 1)) / this_sz) as i32
                };
                fc.svc[i][0].scale = scale_fac(p.p.w, frame_hdr.width);
                fc.svc[i][1].scale = scale_fac(p.p.h, frame_hdr.height);
                fc.svc[i][0].step = (fc.svc[i][0].scale + 8) >> 4;
                fc.svc[i][1].step = (fc.svc[i][1].scale + 8) >> 4;
            } else {
                fc.svc[i][0].scale = 0;
                fc.svc[i][1].scale = 0;
            }
            let mut gm = frame_hdr.gmv.m[i];
            fc.gmv_warp_allowed[i] = (gm.wm_type > crate::headers::WarpedMotionType::Translation
                && frame_hdr.force_integer_mv == 0
                && crate::warpmv::get_shear_params(&mut gm) == 0
                && fc.svc[i][0].scale == 0) as u8;
        }
    }

    // primary_ref_frame == NONE -> static qcat init (keyframe path). Otherwise
    // clone the saved CDF of the primary ref. The avg primary/secondary CDF
    // path (use_pri_sec_cdf) is deferred (not exercised by the bring-up clips).
    let p_ref_idx = frame_hdr.primary_ref_frame;
    let in_cdf: Option<crate::cdf::CdfContext> = if p_ref_idx == crate::headers::PRIMARY_REF_NONE {
        fc.use_pri_sec_cdf = 0;
        None
    } else {
        // entropy context is the weighted average of the primary and secondary
        // reference CDFs: in_cdf = (pri*7 + sec*1 + 4) >> 3 (cdf.c:6650).
        let s_ref_idx = frame_hdr.secondary_ref_frame;
        let use_pri_sec = s_ref_idx != crate::headers::PRIMARY_REF_NONE
            && frame_hdr.frame_type == crate::headers::FrameType::Inter
            && seq_hdr.avg_cdf
            && seq_hdr.avg_cdf_type == 0
            && frame_hdr.tip.frame_mode != 2;
        fc.use_pri_sec_cdf = use_pri_sec as i32;
        let pri_ref = frame_hdr.refidx[p_ref_idx as usize] as usize;
        if use_pri_sec {
            let sec_ref = frame_hdr.refidx[s_ref_idx as usize] as usize;
            let default_cdf = crate::cdf::CdfContext::init_from_defaults(qcat);
            let src1 = c.refs[pri_ref].cdf.as_deref().unwrap_or(&default_cdf);
            let src2 = c.refs[sec_ref].cdf.as_deref().unwrap_or(&default_cdf);
            let mut out = crate::cdf::CdfContext::init_from_defaults(qcat);
            out.pri_sec_average(
                &src1.m, &src1.coef, &src1.mv, &src1.dmv, &src2.m, &src2.coef, &src2.mv, &src2.dmv,
            );
            Some(out)
        } else {
            c.refs[pri_ref].cdf.as_ref().map(|a| (**a).clone())
        }
    };

    fc.seq_hdr = seq_hdr.clone();
    fc.frame_hdr = frame_hdr.clone();
    fc.in_cdf = in_cdf;

    let use_rfm = is_inter_or_switch || allow_intrabc;
    if use_rfm {
        if is_inter_or_switch {
            let poc = frame_hdr.frame_offset as i32;
            let nbits = seq_hdr.order_hint_n_bits as i32;
            let mut furthest_future_refidx: i32 = -2;
            for i in 0..7 {
                let rpoc = fc.refp[i]
                    .frame_hdr
                    .as_ref()
                    .map(|h| h.frame_offset as i32)
                    .unwrap_or(0);
                fc.refpoc[i] = rpoc as u8;
                let delta = crate::env::get_poc_diff(nbits, rpoc, poc);
                fc.refdist[i] = delta as i8;
                fc.absrefdist[i] = delta.unsigned_abs() as u8;
                fc.refdir[i] = (delta > 0) as u8;
                if delta > 0
                    && (furthest_future_refidx < 0
                        || (fc.refdist[furthest_future_refidx as usize] as i32) < delta)
                {
                    furthest_future_refidx = i as i32;
                }
            }
            fc.furthest_future_refidx = furthest_future_refidx as i8;
            // refdir[TIP_FRAME] is a fixed sentinel: TIP references are always
            // treated as a future-direction ref for context derivation
            // ref-context (get_comp_ctx / single-ref) reads refdir[ref] for a
            // TIP neighbour (ref == TIP_FRAME), so this slot must be 1.
            fc.refdir[TIP_FRAME as usize] = 1;
        } else {
            fc.refpoc = [0; 7];
        }
        if frame_hdr.use_ref_frame_mvs != 0 {
            let bw = ((frame_hdr.width + 7) >> 3) << 1;
            let bh = ((frame_hdr.height + 7) >> 3) << 1;
            for i in 0..7 {
                let refidx = frame_hdr.refidx[i] as usize;
                let ref_w = ((ref_coded_width[i] + 7) >> 3) << 1;
                let ref_h = ((fc.refp[i].pic.as_ref().map(|p| p.p.h).unwrap_or(0) + 7) >> 3) << 1;
                if c.refs[refidx].refmvs.is_some() && ref_w == bw && ref_h == bh {
                    fc.ref_mvs[i] = c.refs[refidx].refmvs.clone();
                } else {
                    fc.ref_mvs[i] = None;
                }
                fc.refrefpoc[i] = c.refs[refidx].refpoc;
                fc.refcnt[i] = fc.refp[i]
                    .frame_hdr
                    .as_ref()
                    .map(|h| h.n_ref_frames)
                    .unwrap_or(0);
            }
        }
    }

    if frame_hdr.segmentation.enabled != 0
        && (frame_hdr.segmentation.temporal != 0 || frame_hdr.segmentation.update_map == 0)
    {
        let pri = frame_hdr.primary_ref_frame as usize;
        let ref_w = ((ref_coded_width[pri] + 7) >> 3) << 1;
        let ref_h = ((fc.refp[pri].pic.as_ref().map(|p| p.p.h).unwrap_or(0) + 7) >> 3) << 1;
        let bw = ((frame_hdr.width + 7) >> 3) << 1;
        let bh = ((frame_hdr.height + 7) >> 3) << 1;
        if ref_w == bw && ref_h == bh {
            fc.prev_segmap = c.refs[frame_hdr.refidx[pri] as usize].segmap.clone();
        }
    }

    let skip_mode_r1 = (frame_hdr.skip_mode_enabled != 0
        && frame_hdr.n_ref_frames > 1
        && (fc.absrefdist[0] as i32 - fc.absrefdist[1] as i32).abs() <= 1)
        as i8;
    fc.skip_mode_refs = RefPair {
        r: [0, skip_mode_r1],
    };

    let in_cdf_ref = fc.in_cdf.take();
    // The decoder's worker pool is created in `Decoder::open()`, sized as
    // `n_tc - 1` helper threads because the caller participates in each dispatch.
    // Keep this fallback for contexts constructed directly in tests.
    if n_tc >= 2 && c.pool.is_none() {
        c.pool = Some(crate::mtpool::ThreadPool::new((n_tc - 1) as usize));
    }
    decode_frame(
        &mut *fc,
        n_tc,
        1,
        in_cdf_ref.as_ref(),
        qcat,
        c.pool.as_ref(),
    )?;

    // on failure. Single-thread decode here is synchronous and already succeeded,
    // so we publish after success (equivalent for n_fc == 1). The decoded picture
    // is shared into every refreshed slot so later frames can reference it.
    let cur_pic = std::mem::take(&mut fc.cur_pic);
    let shared = std::sync::Arc::new(cur_pic);
    let out_cdf_for_refs: Option<std::sync::Arc<crate::cdf::CdfContext>> =
        if frame_hdr.disable_cdf_update == 0 {
            fc.out_cdf.clone()
        } else {
            in_cdf_ref.map(std::sync::Arc::new)
        };
    let cur_segmap_arc: Option<Vec<u8>> = if !fc.cur_segmap.is_empty() {
        Some(fc.cur_segmap.clone())
    } else {
        None
    };
    let cur_ccsomap_arc: Option<Vec<u8>> = if !fc.cur_ccsomap.is_empty() {
        Some(fc.cur_ccsomap.clone())
    } else {
        None
    };
    let refresh = frame_hdr.refresh_frame_flags;
    for i in 0..8 {
        if refresh & (1 << i) != 0 {
            c.refs[i].p.pic = Some(shared.clone());
            c.refs[i].p.frame_hdr = Some(frame_hdr.clone());
            c.refs[i].p.showable = frame_hdr.show_immediate == 0;
            c.refs[i].cdf = out_cdf_for_refs.clone();
            c.refs[i].segmap = if frame_hdr.segmentation.update_map != 0 {
                cur_segmap_arc.clone()
            } else {
                fc.prev_segmap.clone()
            };
            if is_inter_or_switch {
                c.refs[i].refmvs = if fc.rf.rp.is_empty() {
                    None
                } else {
                    Some(fc.rf.rp.clone())
                };
            }
            c.refs[i].ccsomap = cur_ccsomap_arc.clone();
            c.refs[i].refpoc = fc.refpoc;
        }
    }

    // Hand the reconstructed picture to the decoder's output path. (Visibility
    // filtering / POC reordering is wired with full output queueing later.)
    // The frame is read-only once decoded, so the output shares the ref slots'
    // plane buffers by reference count rather than deep-copying ~megabytes of
    // pixels. Film grain (applied at output time) writes to a fresh picture, and
    // any other mutation copies on write, so the shared storage is never
    // observed changing.
    let out = shared.shallow_clone();
    c.frame_out.push(out);
    Ok(())
}

/// One colour plane's copy descriptor. `src` is a shared read of the source
/// plane; `dst` launders the destination plane's `&mut` into a raw `(ptr, len)`
/// so the disjoint row bands can be written from several threads at once.
struct PlaneCopy<'a> {
    src: &'a [u8],
    dst: DisjointMut<u8>,
    row_bytes: usize,
    s_stride: usize,
    d_stride: usize,
    ph: usize,
}

/// Deep-copies `src` into a freshly allocated, independently owned picture,
/// parallelising the plane copy across the decoder's worker pool.
///
/// Each participating thread (the caller plus `pool` helpers) claims an id and
/// copies a contiguous, disjoint band of rows from every plane, so the result is
/// byte-identical regardless of thread count. With no pool — or `n_threads <= 1`,
/// or fewer rows than threads — it runs the sequential copy on the caller with no
/// thread hand-off at all. The thread count is clamped to the pool's real
/// capacity so every row is covered exactly once.
pub(crate) fn clone_picture_mt(
    src: &crate::picture::Picture,
    n_threads: u32,
    pool: Option<&crate::mtpool::ThreadPool>,
    allocator: std::sync::Arc<dyn crate::picture::PicAllocator>,
) -> crate::picture::Picture {
    let mut dst = match crate::picture::Picture::alloc(
        src.p.w,
        src.p.h,
        src.p.layout,
        src.p.bpc,
        src.seq_hdr.clone(),
        src.frame_hdr.clone(),
        allocator,
    ) {
        Some(p) => p,
        None => return crate::picture::Picture::new(),
    };
    dst.fgm = src.fgm;
    dst.content_light_level = src.content_light_level;

    let n_planes = if src.p.layout == crate::headers::PixelLayout::I400 {
        1
    } else {
        3
    };
    let bytes = src.bytes_per_sample();

    // Gather per-plane descriptors. `DisjointMut::new` captures only a raw
    // pointer, so taking the three `&mut` plane slices in turn does not hold
    // overlapping borrows of `dst`.
    let mut planes: Vec<PlaneCopy> = Vec::with_capacity(n_planes);
    for pl in 0..n_planes {
        let row_bytes = src.plane_w(pl) * bytes;
        let s_stride = src.stride_bytes(pl);
        let d_stride = dst.stride_bytes(pl);
        let ph = src.plane_h(pl);
        let Some(src_plane) = src.plane_bytes(pl) else {
            continue;
        };
        let Some(dst_plane) = dst.plane_bytes_mut(pl) else {
            continue;
        };
        planes.push(PlaneCopy {
            src: src_plane,
            dst: DisjointMut::new(dst_plane),
            row_bytes,
            s_stride,
            d_stride,
            ph,
        });
    }

    // Only engage the pool when one is present and there is real width to
    // exploit; otherwise n_run == 1 runs everything on the caller, no spawn.
    let want = (n_threads as usize).max(1);
    let active = pool.filter(|_| want >= 2);
    let cap = match active {
        Some(p) => p.workers() + 1,
        None => 1,
    };
    let n_run = want.min(cap).max(1);

    let seq = std::sync::atomic::AtomicUsize::new(0);
    let copy_job = || {
        // One id per participant (exactly `n_run` participants run), so the
        // bands below partition each plane's rows with no gaps or overlap.
        let id = seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .min(n_run - 1);
        for p in &planes {
            let r0 = p.ph * id / n_run;
            let r1 = p.ph * (id + 1) / n_run;
            // SAFETY: every participant writes a disjoint band of rows.
            let dst = unsafe { p.dst.whole() };
            for y in r0..r1 {
                let s0 = y * p.s_stride;
                let d0 = y * p.d_stride;
                dst[d0..d0 + p.row_bytes].copy_from_slice(&p.src[s0..s0 + p.row_bytes]);
            }
        }
    };
    crate::mtpool::dispatch(active, n_run, &copy_job);
    dst
}

pub(crate) fn picture_has_grain(pic: &crate::picture::Picture) -> bool {
    match pic.fgm {
        Some(fgd) => {
            fgd.num_points[0] != 0
                || fgd.num_points[1] != 0
                || fgd.num_points[2] != 0
                || (fgd.clip_to_restricted_range && fgd.chroma_scaling_from_luma)
        }
        None => false,
    }
}

pub(crate) fn apply_grain_to_picture_mt(
    src: &crate::picture::Picture,
    n_threads: u32,
    pool: Option<&crate::mtpool::ThreadPool>,
    allocator: std::sync::Arc<dyn crate::picture::PicAllocator>,
) -> crate::picture::Picture {
    let fgd = match src.fgm {
        Some(f) => f,
        None => return clone_picture_mt(src, n_threads, pool, allocator.clone()),
    };
    if src.p.bpc != 8 {
        // Only 8bpc grain kernels are ported; higher bit depths fall back to a
        // plain copy (no clip corpus exercises >8bpc grain yet).
        return clone_picture_mt(src, n_threads, pool, allocator.clone());
    }

    let mut dst = clone_picture_mt(src, n_threads, pool, allocator.clone());
    let seed = src
        .frame_hdr
        .as_ref()
        .map(|h| h.film_grain.seed)
        .unwrap_or(0);

    let ss_ver = src.p.layout == crate::headers::PixelLayout::I420;
    let ss_hor = matches!(
        src.p.layout,
        crate::headers::PixelLayout::I420 | crate::headers::PixelLayout::I422
    );
    let has_chroma = src.p.layout != crate::headers::PixelLayout::I400;

    let y_stride = src.stride[0];
    let uv_stride = src.stride[1];
    let aligned_h = (src.p.h as usize + 127) & !127;
    let uv_rows = aligned_h >> ss_ver as usize;
    let src_y = src.plane_bytes_rows(0, aligned_h).unwrap_or(&[]);
    let (src_u, src_v): (&[u8], &[u8]) = if has_chroma {
        (
            src.plane_bytes_rows(1, uv_rows).unwrap_or(&[]),
            src.plane_bytes_rows(2, uv_rows).unwrap_or(&[]),
        )
    } else {
        (&[], &[])
    };
    let (dst_y, dst_u, dst_v) = dst.plane_bytes_rows3_mut(aligned_h, uv_rows, has_chroma);

    crate::filmgrain::apply_grain_8bpc_mt(
        dst_y,
        dst_u,
        dst_v,
        src_y,
        src_u,
        src_v,
        y_stride,
        uv_stride,
        &fgd,
        src.p.w as usize,
        src.p.h as usize,
        seed,
        ss_hor,
        ss_ver,
        n_threads,
    );

    dst
}

fn get_snglref_ctx(
    a: &BlockContext,
    l: &BlockContext,
    yb4: usize,
    xb4: usize,
    have_top: bool,
    have_left: bool,
    have_top_right: bool,
    have_bottom_left: bool,
    b_dim: &[u8],
    ref_idx: i8,
) -> usize {
    const NEWMV0_MASK: u32 =
        (1 << 15) | (1 << 20) | (1 << 22) | (1 << 23) | (1 << 26) | (1 << 27) | (1 << 28);
    const NEWMV1_MASK: u32 = (1 << 19) | (1 << 22) | (1 << 25) | (1 << 27);

    let mut row = 0i32;
    let mut col = 0i32;
    let mut newmv = 0i32;

    macro_rules! add_matching {
        ($ctx:expr, $cnt:ident, $idx:expr) => {
            if $ctx.r#ref[0][$idx] as i8 == ref_idx {
                $cnt += 1;
                newmv += (((1u32 << $ctx.mode[$idx]) & NEWMV0_MASK) != 0) as i32;
            } else if $ctx.r#ref[1][$idx] as i8 == ref_idx {
                $cnt += 1;
                newmv += (((1u32 << $ctx.mode[$idx]) & NEWMV1_MASK) != 0) as i32;
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

    ((row != 0) as usize) + ((col != 0) as usize) + 2 * ((newmv != 0) as usize)
}
