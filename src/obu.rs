use std::sync::Arc;

use crate::env::get_poc_diff;
use crate::error::TealdustError;
use crate::getbits::GetBits;
use crate::headers::*;
use crate::internal::{DecoderContext, RefState, TileGroup};
use crate::intops::{iclip, iclip_u8, imax, imin, ulog2};
use crate::warpmv::resolve_divisor_32;

type Result<T> = std::result::Result<T, TealdustError>;

/// Resolve a reference-bank slot by index, rejecting out-of-range indices.
///
/// `hdr.refidx` entries are read from the bitstream, and on the
/// RAS/open-loop-keyframe path they hold long-term frame ids that can exceed
/// the 8-slot reference bank. The C reference indexes `c->refs` with these
/// directly and relies on a later null check; a memory-safe port must bounds
/// check first. A negative `i8` widens to a huge `usize`, so `get` rejects it.
#[inline]
fn ref_slot(refs: &[RefState; 8], idx: i32) -> Result<&RefState> {
    usize::try_from(idx)
        .ok()
        .and_then(|i| refs.get(i))
        .ok_or(TealdustError::InvalidData)
}

static LAYOUTS: [PixelLayout; 4] = [
    PixelLayout::I420,
    PixelLayout::I400,
    PixelLayout::I444,
    PixelLayout::I422,
];

fn check_trailing_bits(gb: &mut GetBits, strict: bool) -> Result<()> {
    let trailing_one = gb.get_bit();

    if gb.has_error() {
        return Err(TealdustError::InvalidData);
    }

    if !strict {
        return Ok(());
    }

    if trailing_one == 0 {
        return Err(TealdustError::InvalidData);
    }

    Ok(())
}

#[inline]
fn tile_log2(sz: i32, tgt: i32) -> i32 {
    let mut k = 0;
    while (sz << k) < tgt {
        k += 1;
    }
    k
}

fn parse_seg_info(seg: &mut SegmentationDataSet, gb: &mut GetBits, n_seg: usize) {
    let mut m: u16 = 1;
    for n in 0..n_seg {
        if gb.get_bit() != 0 {
            seg.delta_q_mask |= m;
            seg.delta_q[n] = gb.get_sbits(10).clamp(-351, 351) as i16;
        }
        seg.skip_mask |= m * gb.get_bit() as u16;
        seg.globalmv_mask |= m * gb.get_bit() as u16;
        m <<= 1;
    }
}

fn parse_tile_info(
    thdr: &mut TileInfo,
    gb: &mut GetBits,
    sbmul: i32,
    sb128: u8,
    seq_sb128: u8,
    w: i32,
    h: i32,
    level: u8,
    tier: u8,
) {
    thdr.uniform = gb.get_bit() != 0;

    let sbsz_log2 = 6 + sb128 as i32;
    let sbsz_min1 = (64 << sb128) - 1;
    let sbw = (w + sbsz_min1) >> sbsz_log2;
    let sbh = (h + sbsz_min1) >> sbsz_log2;
    let w_adj = (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_width_sb = 4096 >> (sbsz_log2 - w_adj);
    let sz_adj = (level >= 14) as i32 + (level >= 18) as i32 + ((level >= 14 && tier != 0) as i32);
    let max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2 - sz_adj);
    thdr.min_log2_cols = tile_log2(max_tile_width_sb, sbw) as u8;
    thdr.max_log2_cols = tile_log2(1, imin(sbw, MAX_TILE_COLS as i32)) as u8;
    thdr.max_log2_rows = tile_log2(1, imin(sbh, MAX_TILE_ROWS as i32)) as u8;
    let min_log2_tiles = imax(
        tile_log2(max_tile_area_sb, sbw * sbh),
        thdr.min_log2_cols as i32,
    );

    if thdr.uniform {
        let seq_sbsz_log2 = 6 + seq_sb128 as i32;
        let fsbw = imax(1, (w + 7) >> seq_sbsz_log2);
        let fsbh = imax(1, (h + 7) >> seq_sbsz_log2);

        thdr.log2_cols = thdr.min_log2_cols;
        while thdr.log2_cols < thdr.max_log2_cols && gb.get_bit() != 0 {
            thdr.log2_cols += 1;
        }
        let tile_w = imax(1, fsbw >> thdr.log2_cols);
        let mut extra = imax(0, fsbw - (tile_w << thdr.log2_cols));
        thdr.cols = 0;
        let mut sbx = 0;
        // log2_cols is capped at max_log2_cols so this is already bounded for
        // valid streams; the explicit MAX_TILE_COLS guard protects against any
        // residual overflow of col_start_sb on malformed input.
        while sbx < fsbw && (thdr.cols as usize) < crate::headers::MAX_TILE_COLS {
            thdr.col_start_sb[thdr.cols as usize] = (sbx * sbmul) as u16;
            let add = tile_w + if extra > 0 { 1 } else { 0 };
            sbx += add;
            thdr.cols += 1;
            extra -= 1;
        }

        thdr.min_log2_rows = imax(min_log2_tiles - thdr.log2_cols as i32, 0) as u8;
        thdr.log2_rows = thdr.min_log2_rows;
        while thdr.log2_rows < thdr.max_log2_rows && gb.get_bit() != 0 {
            thdr.log2_rows += 1;
        }
        let tile_h = imax(1, fsbh >> thdr.log2_rows);
        let mut extra = imax(0, fsbh - (tile_h << thdr.log2_rows));
        thdr.rows = 0;
        let mut sby = 0;
        while sby < fsbh && (thdr.rows as usize) < crate::headers::MAX_TILE_ROWS {
            thdr.row_start_sb[thdr.rows as usize] = (sby * sbmul) as u16;
            let add = tile_h + if extra > 0 { 1 } else { 0 };
            sby += add;
            thdr.rows += 1;
            extra -= 1;
        }
    } else {
        let mut widest_tile = 0;
        thdr.cols = 0;
        let mut sbx = 0;
        // malformed frame size lets the loop run past the col_start_sb array.
        while sbx < sbw && (thdr.cols as usize) < crate::headers::MAX_TILE_COLS {
            thdr.col_start_sb[thdr.cols as usize] = sbx as u16;
            let max_width = imin(sbw - sbx, max_tile_width_sb);
            let w_tile = if max_width > 1 {
                gb.get_uniform(max_width as u32) as i32 + 1
            } else {
                1
            };
            widest_tile = imax(widest_tile, w_tile);
            sbx += w_tile;
            thdr.cols += 1;
        }
        thdr.log2_cols = tile_log2(1, thdr.cols as i32) as u8;

        let max_tile_area_sb_here = if min_log2_tiles > 0 {
            (sbw * sbh) >> (min_log2_tiles + 1)
        } else {
            sbw * sbh
        };
        let max_tile_height_sb = imax(max_tile_area_sb_here / widest_tile, 1);

        thdr.rows = 0;
        let mut sby = 0;
        while sby < sbh && (thdr.rows as usize) < crate::headers::MAX_TILE_ROWS {
            thdr.row_start_sb[thdr.rows as usize] = sby as u16;
            let max_height = imin(sbh - sby, max_tile_height_sb);
            let h_tile = if max_height > 1 {
                gb.get_uniform(max_height as u32) as i32 + 1
            } else {
                1
            };
            sby += h_tile;
            thdr.rows += 1;
        }
        thdr.log2_rows = tile_log2(1, thdr.rows as i32) as u8;
    }
    thdr.col_start_sb[thdr.cols as usize] = sbw as u16;
    thdr.row_start_sb[thdr.rows as usize] = sbh as u16;
}

pub fn parse_tile_info_frmhdr(hdr: &mut FrameHeader, seqhdr: &SequenceHeader, gb: &mut GetBits) {
    hdr.sb128 = if hdr.is_inter_or_switch() {
        seqhdr.sb128
    } else {
        if seqhdr.sb128 != 0 { 1 } else { 0 }
    };

    let mut reuse_allowed = false;
    if seqhdr.tiling.present != AdaptiveBoolean::Off {
        let sbsz_min1 = (64i32 << hdr.sb128) - 1;
        let sbsz_log2 = 6 + hdr.sb128 as i32;
        let sbw = (hdr.width + sbsz_min1) >> sbsz_log2;
        let sbh = (hdr.height + sbsz_min1) >> sbsz_log2;
        if !seqhdr.tiling.t.uniform {
            let seq_sbsz_min1 = (64i32 << seqhdr.sb128) - 1;
            let seq_sbsz_log2 = 6 + seqhdr.sb128 as i32;
            let seq_sbw = (seqhdr.max_width + seq_sbsz_min1) >> seq_sbsz_log2;
            let seq_sbh = (seqhdr.max_height + seq_sbsz_min1) >> seq_sbsz_log2;
            reuse_allowed = seq_sbw == sbw && seq_sbh == sbh;
        } else {
            let tile_w = (sbw + seqhdr.tiling.t.cols as i32 - 1) >> seqhdr.tiling.t.log2_cols;
            let tile_h = (sbh + seqhdr.tiling.t.rows as i32 - 1) >> seqhdr.tiling.t.log2_rows;
            reuse_allowed = tile_w * (seqhdr.tiling.t.cols as i32 - 1) < sbw
                && tile_h * (seqhdr.tiling.t.rows as i32 - 1) < sbh;
        }
    }

    let sbmul;
    if reuse_allowed
        && (seqhdr.tiling.present == AdaptiveBoolean::On
            || (seqhdr.tiling.present == AdaptiveBoolean::Adaptive && gb.get_bit() != 0))
    {
        hdr.tiling.t = seqhdr.tiling.t.clone();
        if hdr.sb128 != seqhdr.sb128 {
            debug_assert!(hdr.sb128 == 1 && seqhdr.sb128 == 2 && hdr.is_key_or_intra());
            sbmul = 2;
            for n in 0..hdr.tiling.t.rows as usize {
                hdr.tiling.t.row_start_sb[n] *= 2;
            }
            for n in 0..hdr.tiling.t.cols as usize {
                hdr.tiling.t.col_start_sb[n] *= 2;
            }
        } else {
            sbmul = 1;
        }
    } else {
        sbmul = if seqhdr.sb128 == 2 && hdr.is_key_or_intra() {
            2
        } else {
            1
        };
        parse_tile_info(
            &mut hdr.tiling.t,
            gb,
            sbmul,
            hdr.sb128,
            seqhdr.sb128,
            hdr.width,
            hdr.height,
            seqhdr.level,
            seqhdr.tier,
        );
    }

    if sbmul == 2 {
        hdr.tiling.t.row_start_sb[hdr.tiling.t.rows as usize] = ((hdr.height + 127) >> 7) as u16;
        hdr.tiling.t.col_start_sb[hdr.tiling.t.cols as usize] = ((hdr.width + 127) >> 7) as u16;
    }
}

pub fn parse_film_grain_data(gb: &mut GetBits, layout: PixelLayout) -> Result<FilmGrainData> {
    let mut fgd = FilmGrainData::default();

    let mut num_pl = 1;
    if layout != PixelLayout::I400 {
        fgd.chroma_scaling_from_luma = gb.get_bit() != 0;
        if !fgd.chroma_scaling_from_luma {
            num_pl = 3;
        }
    }

    for pl in 0..num_pl {
        fgd.num_points[pl] = gb.get_bits(4) as i32;
        if fgd.num_points[pl] > 14 {
            return Err(TealdustError::InvalidData);
        }
        if fgd.num_points[pl] == 0 {
            continue;
        }
        let index_bits = 1 + gb.get_bits(3) as i32;
        let scaling_bits = 5 + gb.get_bits(2) as i32;
        let mut base = 0u32;
        for i in 0..fgd.num_points[pl] as usize {
            base += gb.get_bits(index_bits);
            if base > 255 {
                return Err(TealdustError::InvalidData);
            }
            fgd.points[pl][i][0] = base as u8;
            fgd.points[pl][i][1] = gb.get_bits(scaling_bits) as u8;
        }
    }

    if layout == PixelLayout::I420 && (fgd.num_points[1] != 0) != (fgd.num_points[2] != 0) {
        return Err(TealdustError::InvalidData);
    }

    fgd.scaling_shift = gb.get_bits(2) as i32 + 8;
    fgd.ar_coeff_lag = gb.get_bits(2) as i32;
    let num_pos = 2 * fgd.ar_coeff_lag * (fgd.ar_coeff_lag + 1);
    for pl in 0..3 {
        if fgd.num_points[pl] == 0 && (pl == 0 || !fgd.chroma_scaling_from_luma) {
            continue;
        }
        let num_pl_pos = num_pos + (pl != 0 && fgd.num_points[0] != 0) as i32;
        let coef_bits = 5 + gb.get_bits(2) as i32;
        for i in 0..num_pl_pos as usize {
            fgd.ar_coeffs[pl][i] = (gb.get_bits(coef_bits) as i32 - 128) as i8;
        }
    }
    fgd.ar_coeff_shift = gb.get_bits(2) as u64 + 6;
    fgd.grain_scale_shift = gb.get_bits(2) as i32;
    for pl in 0..2 {
        if fgd.num_points[1 + pl] == 0 {
            continue;
        }
        fgd.uv_mult[pl] = gb.get_bits(8) as i32 - 128;
        fgd.uv_luma_mult[pl] = gb.get_bits(8) as i32 - 128;
        fgd.uv_offset[pl] = gb.get_bits(9) as i32 - 256;
    }
    fgd.overlap_flag = gb.get_bit() != 0;
    fgd.clip_to_restricted_range = gb.get_bit() != 0;
    if fgd.clip_to_restricted_range {
        fgd.mc_identity = gb.get_bit() != 0;
    }
    fgd.block_size = gb.get_bit() as i32;

    Ok(fgd)
}

pub fn rescale_matrix(dm: &mut [i32; 6], sm: &[i32; 6], in_dist: i32, out_dist: i32) {
    let mut shift = 0i32;
    let mut inv = resolve_divisor_32(in_dist.unsigned_abs(), &mut shift);
    if inv >= 512 {
        inv >>= 1;
        shift -= 1;
    }
    if in_dist < 0 {
        inv = -inv;
    }
    let rnd = (1 << shift) >> 1;
    for n in 0..2 {
        let r = iclip(sm[n], -0x400000, 0x400000) * inv;
        let t = ((r + rnd - (r < 0) as i32) >> shift) * out_dist;
        let d = (t + 0x1000 - (t < 0) as i32) & !0x1fff;
        dm[n] = iclip(d, -0x7ffe000, 0x7ffe000);
    }
    for n in 2..6 {
        let b = 0x10000 * (((n as u32).wrapping_sub(3)) > 1) as i32;
        let r = (sm[n] - b) * inv;
        let t = ((r + rnd - (r < 0) as i32) >> shift) * out_dist;
        let d = (t + 32 - (t < 0) as i32) & !63;
        dm[n] = b + iclip(d, -0x7fc0, 0x7fc0);
    }
}

pub fn parse_seq_hdr(gb: &mut GetBits, strict: bool) -> Result<SequenceHeader> {
    let mut hdr = SequenceHeader::default();

    hdr.id = gb.get_vlc() as u8;
    hdr.profile = gb.get_bits(5) as u8;
    // AV2 defines profiles 0–8; the original check (> 2) matches AV1 which only
    // has profiles 0-2. Relax to 8 so real AV2 encoders (e.g. profile 4) pass.
    if hdr.profile > 8 {
        return Err(TealdustError::InvalidData);
    }
    hdr.reduced_still_picture_header = gb.get_bit() != 0;
    hdr.level = gb.get_bits(5) as u8;
    if hdr.level >= 4 && !hdr.reduced_still_picture_header {
        hdr.tier = gb.get_bit() as u8;
    }

    let layout_idx = gb.get_vlc();
    if layout_idx > 3 {
        return Err(TealdustError::InvalidData);
    }
    hdr.layout = LAYOUTS[layout_idx as usize];
    match hdr.layout {
        PixelLayout::I420 | PixelLayout::I400 => {
            hdr.ss_hor = 1;
            hdr.ss_ver = 1;
        }
        PixelLayout::I422 => {
            hdr.ss_hor = 1;
            hdr.ss_ver = 0;
        }
        _ => {}
    }

    hdr.hbd = gb.get_vlc() as u8;
    if hdr.hbd > 2 {
        return Err(TealdustError::InvalidData);
    }
    if hdr.hbd < 2 {
        hdr.hbd ^= 1;
    }

    if hdr.reduced_still_picture_header {
        hdr.still_picture = true;
        hdr.monotonic = true;
    } else {
        hdr.lcr_id = gb.get_bits(3) as u8;
        hdr.still_picture = gb.get_bit() != 0;
        hdr.max_tlayer_id = gb.get_bits(2) as u8;
        hdr.max_mlayer_id = gb.get_bits(3) as u8;
        hdr.monotonic = gb.get_bit() != 0;
    }

    hdr.width_n_bits = gb.get_bits(4) as u8 + 1;
    hdr.height_n_bits = gb.get_bits(4) as u8 + 1;
    hdr.max_width = gb.get_bits(hdr.width_n_bits as i32) as i32 + 1;
    hdr.max_height = gb.get_bits(hdr.height_n_bits as i32) as i32 + 1;

    hdr.crop.enabled = gb.get_bit() != 0;
    if hdr.crop.enabled {
        hdr.crop.left = gb.get_vlc();
        hdr.crop.right = gb.get_vlc();
        hdr.crop.top = gb.get_vlc();
        hdr.crop.bottom = gb.get_vlc();
    }

    if !hdr.reduced_still_picture_header {
        if gb.get_bit() != 0 {
            // max_display_model_info_present
            let _max_initial_display_delay = gb.get_bits(4);
        }
        let decoder_model_info_present = gb.get_bit() != 0;
        if decoder_model_info_present {
            let _num_units = gb.get_bits(32);
            let _max_dec_buf = gb.get_vlc();
            let _max_enc_buf = gb.get_vlc();
        }
    }

    if hdr.max_tlayer_id > 0 {
        hdr.tlayer_dependency_present = gb.get_bit() != 0;
        if hdr.tlayer_dependency_present {
            for n in 1..hdr.max_tlayer_id as usize {
                hdr.tlayer_dependencies[n] = gb.get_bits(n as i32) as u8;
            }
        } else {
            let mut mask = !0u32;
            for n in 1..hdr.max_tlayer_id as usize {
                hdr.tlayer_dependencies[n] = (!mask) as u8;
                mask <<= 1;
            }
        }
    }

    if hdr.max_mlayer_id > 0 {
        hdr.mlayer_dependency_present = gb.get_bit() != 0;
        if hdr.mlayer_dependency_present {
            for n in 1..hdr.max_mlayer_id as usize {
                hdr.mlayer_dependencies[n] = gb.get_bits(n as i32) as u8;
            }
        } else {
            let mut mask = !0u32;
            for n in 1..hdr.max_mlayer_id as usize {
                hdr.mlayer_dependencies[n] = (!mask) as u8;
                mask <<= 1;
            }
        }
    }

    hdr.sb128 = if gb.get_bit() != 0 {
        2
    } else {
        gb.get_bit() as u8
    };

    if hdr.layout != PixelLayout::I400 {
        hdr.sdp = gb.get_bit() != 0;
        if hdr.sdp && !hdr.reduced_still_picture_header {
            hdr.ext_sdp = gb.get_bit() != 0;
        }
    }
    hdr.ext_partitions = gb.get_bit() != 0;
    if hdr.ext_partitions {
        hdr.uneven_4way_partitions = gb.get_bit() != 0;
    }
    hdr.max_pb_aspect_ratio_log2 = if gb.get_bit() != 0 {
        1 + gb.get_bit() as u8
    } else {
        3
    };

    hdr.segmentation.ext = gb.get_bit() != 0;
    hdr.segmentation.info_present = gb.get_bit() != 0;
    if hdr.segmentation.info_present {
        hdr.segmentation.adaptive = gb.get_bit() != 0;
        parse_seg_info(
            &mut hdr.segmentation.d,
            gb,
            8 << (hdr.segmentation.ext as usize),
        );
    }

    hdr.intra_dip = gb.get_bit() != 0;
    hdr.intra_edge_filter = gb.get_bit() != 0;
    hdr.mrls = gb.get_bit() != 0;
    hdr.cfl = gb.get_bit() != 0;
    if hdr.layout != PixelLayout::I400 {
        hdr.cfl_ds_filter_index = gb.get_bits(2) as u8;
    }
    hdr.mhccp = gb.get_bit() != 0;
    hdr.ibp = gb.get_bit() != 0;

    if hdr.reduced_still_picture_header {
        hdr.motion_modes = 1;
    } else {
        hdr.motion_modes = 1; // MM_TRANSLATION = bit 0
        for shift in [1, 2, 3, 4] {
            hdr.motion_modes |= (gb.get_bit() as u8) << shift;
        }
        if hdr.motion_modes & !1 != 0 {
            hdr.frame_motion_modes_present = gb.get_bit() != 0;
        }
        if hdr.motion_modes & (1 << 3) != 0 {
            // MM_WARP_DELTA
            hdr.six_param_warp_delta = gb.get_bit() != 0;
        }
        hdr.masked_compound = gb.get_bit() != 0;
        hdr.ref_frame_mvs = gb.get_bit() != 0;
        if hdr.ref_frame_mvs {
            hdr.reduced_ref_frame_mvs_mode = gb.get_bit() as u8;
        }
        hdr.order_hint_n_bits = gb.get_bits(4) as u8 + 1;
    }

    hdr.refmv_bank = gb.get_bit() != 0;
    // 0 = off, 2 = always (threshold 2), 1 = constraint (threshold 4).
    hdr.drl_reorder = if gb.get_bit() != 0 {
        0
    } else {
        2 - gb.get_bit() as u8
    };

    if hdr.reduced_still_picture_header {
        hdr.ref_frames = 2;
        hdr.def_max_drl_bits = 1;
    } else {
        hdr.explicit_ref_frame_map = gb.get_bit() != 0;
        hdr.ref_frames = if gb.get_bit() != 0 {
            gb.get_bits(4) as u8 + 1
        } else {
            8
        };
        // The reference-frame bank has DAV2D_NUM_REF_FRAMES (8) slots; reference
        // indices are later validated only against `ref_frames`, so a value above
        // 8 would let a malformed stream index the 8-slot `refs` array out of
        // bounds. Valid streams never signal more than 8 reference frames.
        if hdr.ref_frames > 8 {
            return Err(TealdustError::InvalidData);
        }
        hdr.ref_frames_log2 = if hdr.ref_frames <= 2 {
            hdr.ref_frames - 1
        } else {
            1 + ulog2(hdr.ref_frames as u32 - 1) as u8
        };
        hdr.number_of_bits_for_lt_frame_id = gb.get_bits(3) as u8;
        hdr.def_max_drl_bits = gb.get_uniform(5) as u8 + 1;
        hdr.allow_frame_max_drl_bits = gb.get_bit() != 0;
    }
    hdr.def_max_bvp_drl_bits = gb.get_uniform(3) as u8 + 1;
    hdr.allow_max_bvp_drl_bits = gb.get_bit() != 0;
    if !hdr.reduced_still_picture_header {
        hdr.num_same_ref_comp = gb.get_bits(2) as u8;
    }

    if !hdr.reduced_still_picture_header {
        let tip_val = gb.get_bit();
        hdr.tip = tip_val != 0 && (1 + gb.get_bit() as u8) > 0;
        if hdr.tip {
            hdr.tip_hole_fill = gb.get_bit() != 0;
        }
        hdr.mv_traj = gb.get_bit() != 0;
    }
    hdr.bawp = gb.get_bit() != 0;
    if !hdr.reduced_still_picture_header {
        hdr.cwp = gb.get_bit() != 0;
        hdr.imp_msk_bld = gb.get_bit() != 0;
        hdr.db_sub_pu = gb.get_bit() != 0;
        if hdr.tip && hdr.db_sub_pu {
            hdr.tip_explicit_qp = gb.get_bit() != 0;
        }
    }

    if !hdr.reduced_still_picture_header {
        hdr.opfl_refine = gb.get_bits(2) != 0;
        hdr.refine_mv = gb.get_bit() != 0;
        if hdr.tip && (hdr.opfl_refine || hdr.refine_mv) {
            hdr.tip_refine_mv = gb.get_bit() != 0;
        }
        hdr.bru = gb.get_bit() != 0;
        hdr.adaptive_mvd = gb.get_bit() != 0;
        hdr.mvd_sign_derive = gb.get_bit() != 0;
        hdr.flex_mvres = gb.get_bit() != 0;
        hdr.global_motion = gb.get_bit() != 0;
        hdr.short_refresh_frame_flags = gb.get_bit() != 0;
    }

    if hdr.reduced_still_picture_header {
        hdr.screen_content_tools = AdaptiveBoolean::Adaptive;
        hdr.force_integer_mv = AdaptiveBoolean::Adaptive;
    } else {
        hdr.screen_content_tools = if gb.get_bit() != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            if gb.get_bit() != 0 {
                AdaptiveBoolean::On
            } else {
                AdaptiveBoolean::Off
            }
        };
        hdr.force_integer_mv = if hdr.screen_content_tools != AdaptiveBoolean::Off {
            if gb.get_bit() != 0 {
                AdaptiveBoolean::Adaptive
            } else {
                if gb.get_bit() != 0 {
                    AdaptiveBoolean::On
                } else {
                    AdaptiveBoolean::Off
                }
            }
        } else {
            AdaptiveBoolean::Adaptive
        };
    }

    hdr.fsc = gb.get_bit() != 0;
    hdr.idtx_intra = hdr.fsc || gb.get_bit() != 0;
    hdr.ist[0] = gb.get_bit() != 0;
    hdr.ist[1] = gb.get_bit() != 0;
    if hdr.layout != PixelLayout::I400 {
        hdr.chroma_dctonly = gb.get_bit() != 0;
    }
    if !hdr.reduced_still_picture_header {
        hdr.inter_ddt = gb.get_bit() != 0;
    }
    hdr.reduced_tx_part_set = gb.get_bit() != 0;
    if hdr.layout != PixelLayout::I400 {
        hdr.cctx = gb.get_bit() != 0;
    }

    let tcq_bit = gb.get_bit();
    hdr.tcq = if tcq_bit != 0 {
        if !hdr.reduced_still_picture_header && gb.get_bit() != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            AdaptiveBoolean::On
        }
    } else {
        AdaptiveBoolean::Off
    };
    if hdr.tcq != AdaptiveBoolean::On {
        hdr.parity_hiding = gb.get_bit() != 0;
    }

    hdr.avg_cdf = hdr.reduced_still_picture_header || gb.get_bit() != 0;
    if hdr.avg_cdf {
        hdr.avg_cdf_type = if hdr.reduced_still_picture_header || gb.get_bit() != 0 {
            1
        } else {
            0
        };
    }

    if hdr.layout != PixelLayout::I400 {
        hdr.separate_uv_delta_q = gb.get_bit() != 0;
    }
    hdr.equal_ac_dc_q = gb.get_bit() != 0;
    if !hdr.equal_ac_dc_q {
        hdr.base_ydc_dq = gb.get_bits(5) as i8 - 23;
        hdr.ydc_dq_enabled = gb.get_bit() != 0;
    }
    if hdr.layout != PixelLayout::I400 {
        if !hdr.equal_ac_dc_q {
            hdr.base_uvdc_dq = (gb.get_bits(5) as i32 - 23) as u8;
            hdr.uvdc_dq_enabled = gb.get_bit() != 0;
        }
        hdr.base_uvac_dq = (gb.get_bits(5) as i32 - 23) as u8;
        hdr.uvac_dq_enabled = gb.get_bit() != 0;
        if hdr.equal_ac_dc_q {
            hdr.base_uvdc_dq = hdr.base_uvac_dq;
        }
    }

    hdr.disable_loopfilters_across_tiles = gb.get_bit() != 0;
    hdr.cdef = gb.get_bit() != 0;
    hdr.gdf = gb.get_bit() != 0;
    if hdr.gdf && hdr.sb128 == 0 {
        hdr.gdf_unit_matches_sbsz = gb.get_bit() != 0;
    }
    hdr.restoration = gb.get_bit() != 0;
    if hdr.restoration {
        let no_pc_wiener = gb.get_bit() as u8;
        let no_ns_wiener_y = gb.get_bit() as u8;
        hdr.rst_disable_mask[0] = (no_ns_wiener_y << 1) | no_pc_wiener;
        if gb.get_bit() != 0 {
            hdr.rst_disable_mask[1] = (gb.get_bit() as u8) << 1 | 1;
        } else {
            hdr.rst_disable_mask[1] = hdr.rst_disable_mask[0] | 1;
        }
    }
    hdr.ccso = gb.get_bit() != 0;
    if hdr.ccso {
        hdr.ccso_unit_matches_sbsz = gb.get_bit() != 0;
    }
    hdr.cdef_on_skiptx = if hdr.reduced_still_picture_header {
        AdaptiveBoolean::Adaptive
    } else if gb.get_bit() != 0 {
        AdaptiveBoolean::On
    } else if gb.get_bit() != 0 {
        AdaptiveBoolean::Off
    } else {
        AdaptiveBoolean::Adaptive
    };
    hdr.df_par_bits = 2 + gb.get_bits(2) as u8;

    let tiling_present = gb.get_bit();
    if tiling_present != 0 {
        let tiling_type = gb.get_bit();
        hdr.tiling.present = if tiling_type != 0 {
            AdaptiveBoolean::Adaptive
        } else {
            AdaptiveBoolean::On
        };
        parse_tile_info(
            &mut hdr.tiling.t,
            gb,
            1,
            hdr.sb128,
            hdr.sb128,
            hdr.max_width,
            hdr.max_height,
            hdr.level,
            hdr.tier,
        );
    }

    hdr.film_grain_present = gb.get_bit() != 0;

    if gb.has_error() {
        return Err(TealdustError::InvalidData);
    }

    if !strict {
        return Ok(hdr);
    }

    // extension handling — skip for non-strict mode
    let has_extension = gb.get_bit() != 0;
    if has_extension {
        // skip extension bits (we don't parse them)
    }

    check_trailing_bits(gb, strict)?;
    Ok(hdr)
}

pub fn parse_frame_hdr(
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
    obu_type: ObuType,
    gb: &mut GetBits,
) -> Result<FrameHeader> {
    use crate::levels::MotionMode;
    use crate::tables::{
        CCSO_QUANT_SZ, DEFAULT_WM_PARAMS, NS_WIENER_COEF_RANGE_UV, NS_WIENER_COEF_RANGE_Y,
        SUBSET_MASKS_UV, SUBSET_MASKS_Y, WIENER_NS_FILTERS,
    };

    let mut hdr = FrameHeader::default();

    hdr.id = gb.get_vlc() as u8;
    // hdr.id is the frame_parameter_set_id; not used downstream, so only stored.
    let _seqhdr_idx = gb.get_vlc() as u8;
    // _seqhdr_idx identifies which buffered sequence header this frame references.
    // A strict check against seqhdr.id breaks real-world encoders that encode a
    // non-zero sequence-header ID but always emit 0 in frame headers (common in
    // AVIF still-image encoders).  Since the decoder keeps exactly one active
    // sequence header in c.seq_hdr, the reference is unambiguous regardless of
    // the numeric ID values, so we skip the equality assertion.

    hdr.show_existing_frame = (obu_type == ObuType::Sef) as u8;
    if hdr.show_existing_frame != 0 {
        hdr.existing_frame_idx = gb.get_bits(seqhdr.ref_frames_log2 as i32) as i8;
        if hdr.existing_frame_idx as u8 >= seqhdr.ref_frames {
            return Err(TealdustError::InvalidData);
        }
        // consumed by the reference at this point; match it.
        return Ok(hdr);
    }

    if seqhdr.reduced_still_picture_header {
        hdr.frame_type = FrameType::Key;
        hdr.show_immediate = 1;
    } else {
        match obu_type {
            ObuType::ClosedLoopKf | ObuType::OpenLoopKf => {
                hdr.frame_type = FrameType::Key;
            }
            ObuType::Ras | ObuType::Switch => {
                hdr.frame_type = FrameType::Switch;
            }
            ObuType::LeadingTip | ObuType::Tip | ObuType::Bridge => {
                hdr.frame_type = FrameType::Inter;
            }
            _ => {
                if gb.get_bit() == 0 {
                    hdr.frame_type = FrameType::Intra;
                } else {
                    hdr.frame_type = FrameType::Inter;
                }
            }
        }
        hdr.ltr_id = -1;
        if hdr.frame_type == FrameType::Key {
            if seqhdr.number_of_bits_for_lt_frame_id > 0 {
                hdr.ltr_id = gb.get_bits(seqhdr.number_of_bits_for_lt_frame_id as i32) as i8 - 1;
            }
        } else if (obu_type == ObuType::Ras || obu_type == ObuType::OpenLoopKf)
            && seqhdr.number_of_bits_for_lt_frame_id > 0
        {
            hdr.n_ref_frames = gb.get_bits(3) as u8;
            for n in 0..hdr.n_ref_frames as usize {
                hdr.refidx[n] = gb.get_bits(seqhdr.number_of_bits_for_lt_frame_id as i32) as i8;
            }
        }
        if obu_type != ObuType::Bridge {
            if obu_type != ObuType::OpenLoopKf {
                hdr.show_immediate = gb.get_bit() as u8;
            }
            if hdr.show_immediate == 0 && !seqhdr.monotonic {
                hdr.show_implicit = gb.get_bit() as u8;
            }
        }
    }

    hdr.primary_ref_frame = PRIMARY_REF_NONE;
    if !seqhdr.reduced_still_picture_header {
        hdr.frame_size_override = if hdr.frame_type == FrameType::Switch {
            1
        } else {
            gb.get_bit() as u8
        };
        hdr.frame_offset = gb.get_bits(seqhdr.order_hint_n_bits as i32) as u8;
        if hdr.frame_type == FrameType::Inter {
            hdr.primary_ref_signaled = gb.get_bit() as u8;
            if obu_type != ObuType::LeadingTip && obu_type != ObuType::Tip {
                hdr.cross_frame_context = gb.get_bit() as u8;
            }
            if hdr.primary_ref_signaled != 0 {
                hdr.primary_ref_frame = gb.get_bits(3) as u8;
            }
        }
    }

    // refresh_frame_flags
    if obu_type == ObuType::ClosedLoopKf && seqhdr.max_mlayer_id == 0 {
        hdr.refresh_frame_flags = ((1u32 << seqhdr.ref_frames) - 1) as u8;
    } else if obu_type == ObuType::OpenLoopKf || seqhdr.max_mlayer_id > 0 {
        if seqhdr.short_refresh_frame_flags {
            hdr.refresh_frame_flags = 1 << gb.get_bits(seqhdr.ref_frames_log2 as i32);
        } else {
            hdr.refresh_frame_flags = gb.get_bits(seqhdr.ref_frames as i32) as u8;
        }
    } else if hdr.frame_type != FrameType::Switch && seqhdr.short_refresh_frame_flags {
        let refresh = gb.get_bit() != 0;
        if refresh {
            let refresh_idx = gb.get_bits(seqhdr.ref_frames_log2 as i32);
            if refresh_idx >= seqhdr.ref_frames as u32 {
                return Err(TealdustError::InvalidData);
            }
            hdr.refresh_frame_flags = 1 << refresh_idx;
        }
    } else {
        hdr.refresh_frame_flags = gb.get_bits(seqhdr.ref_frames as i32) as u8;
    }

    let mut tip_output_frame = false;

    if hdr.is_inter_or_switch() {
        if hdr.frame_type == FrameType::Switch || seqhdr.explicit_ref_frame_map {
            hdr.n_ref_frames = gb.get_bits(3) as u8;
            if hdr.n_ref_frames as i32 > imin(7, seqhdr.ref_frames as i32) {
                return Err(TealdustError::InvalidData);
            }
            for n in 0..hdr.n_ref_frames as usize {
                hdr.refidx[n] = gb.get_bits(seqhdr.ref_frames_log2 as i32) as i8;
                if hdr.refidx[n] as u8 >= seqhdr.ref_frames {
                    return Err(TealdustError::InvalidData);
                }
            }
        } else {
            hdr.n_ref_frames = get_ref_frames(&mut hdr, seqhdr, refs, false) as u8;
        }
        let poc = hdr.frame_offset as i32;
        for n in 0..hdr.n_ref_frames as usize {
            let refhdr = ref_slot(refs, hdr.refidx[n] as i32)?
                .p
                .frame_hdr
                .as_ref()
                .ok_or(TealdustError::InvalidData)?;
            let pocdiff = get_poc_diff(
                seqhdr.order_hint_n_bits as i32,
                poc,
                refhdr.frame_offset as i32,
            );
            hdr.has_future_refs |= (pocdiff < 0) as u8;
            hdr.has_past_refs |= (pocdiff > 0) as u8;
        }
        hdr.has_bothside_refs = (hdr.has_future_refs != 0 && hdr.has_past_refs != 0) as u8;
    }

    read_frame_size(&mut hdr, seqhdr, refs, gb)?;

    if hdr.is_inter_or_switch() {
        if hdr.frame_type == FrameType::Inter && !seqhdr.explicit_ref_frame_map {
            hdr.n_ref_frames = get_ref_frames(&mut hdr, seqhdr, refs, true) as u8;
        }

        // base_resolution_update (AV2 §5.9.6): not parsed by the reference decoder

        if seqhdr.ref_frame_mvs {
            hdr.use_ref_frame_mvs = gb.get_bit() as u8;
        }
        hdr.tmvp_sample_step = 1
            + (hdr.use_ref_frame_mvs != 0
                && hdr.n_ref_frames > 1
                && seqhdr.sb128 != 0
                && gb.get_bit() != 0) as u8;

        hdr.tip.subpel_filter = FilterMode::Sharp8Tap as u8;
        if seqhdr.tip && hdr.n_ref_frames > 1 && hdr.use_ref_frame_mvs != 0 {
            if obu_type == ObuType::Tip || obu_type == ObuType::LeadingTip {
                hdr.tip.frame_mode = 2; // output
                hdr.opfl_refine_type = 2 * (seqhdr.opfl_refine && seqhdr.tip_refine_mv) as u8;
            } else {
                hdr.tip.frame_mode = gb.get_bit() as u8;
                hdr.opfl_refine_type = if (seqhdr.opfl_refine as u8) < 3 {
                    seqhdr.opfl_refine as u8
                } else if gb.get_bit() != 0 {
                    1
                } else {
                    2 * gb.get_bit() as u8
                };
            }
            if hdr.tip.frame_mode != 0 {
                if seqhdr.tip_hole_fill {
                    hdr.tip.hole_fill = gb.get_bit() as u8;
                }
                if hdr.has_bothside_refs == 0
                    || !seqhdr.tip_refine_mv
                    || (!seqhdr.opfl_refine && !seqhdr.refine_mv)
                {
                    hdr.tip.global_wtd_idx = gb.get_bits(3) as u8;
                }
                if hdr.tip.frame_mode == 2 {
                    if gb.get_bit() == 0 {
                        hdr.tip.gmv_y = gb.get_bits(4) as i8;
                        hdr.tip.gmv_x = gb.get_bits(4) as i8;
                        if hdr.tip.gmv_y != 0 && gb.get_bit() != 0 {
                            hdr.tip.gmv_y = -hdr.tip.gmv_y;
                        }
                        if hdr.tip.gmv_x != 0 && gb.get_bit() != 0 {
                            hdr.tip.gmv_x = -hdr.tip.gmv_x;
                        }
                    }
                    hdr.tip.subpel_filter = if gb.get_bit() != 0 {
                        FilterMode::Sharp8Tap as u8
                    } else if gb.get_bit() != 0 {
                        FilterMode::Regular8Tap as u8
                    } else {
                        FilterMode::Smooth8Tap as u8
                    };
                }
            }
            find_tip_ref_frames(&mut hdr, seqhdr, refs)?;
        } else {
            hdr.opfl_refine_type = if (seqhdr.opfl_refine as u8) < 3 {
                seqhdr.opfl_refine as u8
            } else if gb.get_bit() != 0 {
                1
            } else {
                2 * gb.get_bit() as u8
            };
        }

        if hdr.tip.frame_mode == 2 {
            if seqhdr.db_sub_pu {
                hdr.deblock.sub_pu = gb.get_bit() as u8;
                if hdr.deblock.sub_pu != 0 {
                    hdr.tip.apply_filter = gb.get_bit() as u8;
                    if hdr.tip.apply_filter != 0 {
                        hdr.deblock.level_y[0] = 1;
                        hdr.deblock.level_y[1] = 1;
                        hdr.deblock.level_u = 1;
                        hdr.deblock.level_v = 1;
                    }
                }
            }
            if seqhdr.tip_explicit_qp {
                // TIP explicit QP: yac_delta and (sometimes) u/v ac delta (AV2
                // No bits are consumed by the reference here; match it. When
                // tip_explicit_qp is unset, quant.yac is derived from the two TIP
            } else {
                let tip_ref0 = *hdr
                    .refidx
                    .get(hdr.tip.r#ref[0] as usize)
                    .ok_or(TealdustError::InvalidData)?;
                let tip_ref1 = *hdr
                    .refidx
                    .get(hdr.tip.r#ref[1] as usize)
                    .ok_or(TealdustError::InvalidData)?;
                let ref1hdr = ref_slot(refs, tip_ref0 as i32)?
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(TealdustError::InvalidData)?;
                let ref2hdr = ref_slot(refs, tip_ref1 as i32)?
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(TealdustError::InvalidData)?;
                hdr.quant.yac = (ref1hdr.quant.yac + ref2hdr.quant.yac + 1) >> 1;
            }

            if hdr.tip.apply_filter != 0 {
                parse_tile_info_frmhdr(&mut hdr, seqhdr, gb);
            } else {
                hdr.sb128 = if hdr.is_inter_or_switch() {
                    seqhdr.sb128
                } else {
                    if seqhdr.sb128 != 0 { 1 } else { 0 }
                };
                hdr.tiling.t.rows = 1;
                hdr.tiling.t.cols = 1;
                let shift = 6 + hdr.sb128 as i32;
                hdr.tiling.t.col_start_sb[0] = 0;
                hdr.tiling.t.col_start_sb[1] = ((hdr.width + ((1 << shift) - 1)) >> shift) as u16;
                hdr.tiling.t.row_start_sb[0] = 0;
                hdr.tiling.t.row_start_sb[1] = ((hdr.height + ((1 << shift) - 1)) >> shift) as u16;
            }

            hdr.disable_cdf_update = 1;
            let pri_sec = derive_pri_sec_ref(&hdr, seqhdr, refs);
            hdr.primary_ref_frame = pri_sec[0] as u8;
            hdr.secondary_ref_frame = pri_sec[1] as u8;
            tip_output_frame = true;
        }
    }

    if !tip_output_frame {
        // screen content tools
        hdr.allow_screen_content_tools = if seqhdr.screen_content_tools == AdaptiveBoolean::Adaptive
        {
            gb.get_bit() as u8
        } else {
            seqhdr.screen_content_tools as u8
        };
        if hdr.allow_screen_content_tools != 0 {
            hdr.force_integer_mv = if seqhdr.force_integer_mv == AdaptiveBoolean::Adaptive {
                gb.get_bit() as u8
            } else {
                seqhdr.force_integer_mv as u8
            };
        }

        hdr.allow_intrabc = gb.get_bit() as u8;
        if hdr.allow_intrabc != 0 {
            if hdr.is_key_or_intra() {
                hdr.allow_global_intrabc = gb.get_bit() as u8;
            }
            hdr.allow_local_intrabc = (hdr.allow_global_intrabc == 0 || gb.get_bit() != 0) as u8;
        }
        if hdr.allow_intrabc != 0 {
            hdr.max_bvp_drl_bits = if seqhdr.allow_max_bvp_drl_bits {
                gb.get_ref_uniform(3, seqhdr.def_max_bvp_drl_bits as u32) as u8 + 1
            } else {
                seqhdr.def_max_bvp_drl_bits
            };
        }

        if hdr.is_inter_or_switch() {
            hdr.max_drl_bits = if seqhdr.allow_frame_max_drl_bits {
                gb.get_ref_uniform(3, seqhdr.def_max_drl_bits as u32) as u8 + 1
            } else {
                seqhdr.def_max_drl_bits
            };
            if hdr.force_integer_mv == 0 {
                hdr.mv_precision = if gb.get_bit() != 0 {
                    2
                } else {
                    1 + 2 * gb.get_bit() as u8
                };
            }
            hdr.subpel_filter_mode = if gb.get_bit() != 0 {
                FilterMode::Switchable
            } else {
                match gb.get_bits(2) {
                    0 => FilterMode::Regular8Tap,
                    1 => FilterMode::Smooth8Tap,
                    2 => FilterMode::Sharp8Tap,
                    _ => FilterMode::Bilinear,
                }
            };
            if seqhdr.frame_motion_modes_present {
                hdr.motion_modes = 1;
                let mut n = 2u8;
                while n > 0 {
                    if (seqhdr.motion_modes & n) != 0 && gb.get_bit() != 0 {
                        hdr.motion_modes |= n;
                    }
                    if n == 16 {
                        break;
                    }
                    n <<= 1;
                }
            } else {
                hdr.motion_modes = seqhdr.motion_modes;
            }
        }

        hdr.disable_cdf_update = gb.get_bit() as u8;

        parse_tile_info_frmhdr(&mut hdr, seqhdr, gb);
        if hdr.tiling.t.log2_cols != 0 || hdr.tiling.t.log2_rows != 0 {
            if seqhdr.avg_cdf_type == 0 {
                hdr.tiling.update = gb
                    .get_bits(hdr.tiling.t.log2_cols as i32 + hdr.tiling.t.log2_rows as i32)
                    as u16;
            }
            if hdr.tiling.update >= hdr.tiling.t.cols as u16 * hdr.tiling.t.rows as u16 {
                return Err(TealdustError::InvalidData);
            }
            hdr.tiling.n_bytes = gb.get_bits(2) as u8 + 1;
        }

        // quant
        hdr.quant.yac = gb.get_bits(8 + (seqhdr.hbd != 0) as i32) as u16;
        if seqhdr.ydc_dq_enabled && gb.get_bit() != 0 {
            hdr.quant.ydc_delta = gb.get_sbits(7) as i8;
        }
        if seqhdr.layout != PixelLayout::I400 && (seqhdr.uvdc_dq_enabled || seqhdr.uvac_dq_enabled)
        {
            let diff_uv_delta = if seqhdr.separate_uv_delta_q {
                gb.get_bit() != 0
            } else {
                false
            };
            if seqhdr.uvdc_dq_enabled && gb.get_bit() != 0 {
                hdr.quant.udc_delta = gb.get_sbits(7) as i8;
            }
            if seqhdr.uvac_dq_enabled && gb.get_bit() != 0 {
                hdr.quant.uac_delta = gb.get_sbits(7) as i8;
            }
            if diff_uv_delta {
                if seqhdr.uvdc_dq_enabled && gb.get_bit() != 0 {
                    hdr.quant.vdc_delta = gb.get_sbits(7) as i8;
                }
                if seqhdr.uvac_dq_enabled && gb.get_bit() != 0 {
                    hdr.quant.vac_delta = gb.get_sbits(7) as i8;
                }
            } else {
                hdr.quant.vdc_delta = hdr.quant.udc_delta;
                hdr.quant.vac_delta = hdr.quant.uac_delta;
            }
        }

        hdr.secondary_ref_frame = PRIMARY_REF_NONE;
        if hdr.is_inter_or_switch() {
            let pri_sec = derive_pri_sec_ref(&hdr, seqhdr, refs);
            if hdr.primary_ref_signaled == 0 {
                hdr.primary_ref_frame = pri_sec[0] as u8;
            }
            if hdr.primary_ref_frame != PRIMARY_REF_NONE {
                hdr.secondary_ref_frame =
                    pri_sec[(pri_sec[1] != hdr.primary_ref_frame as i32) as usize] as u8;
            }
        }

        // segmentation
        hdr.segmentation.enabled = gb.get_bit() as u8;
        if hdr.segmentation.enabled != 0 {
            if seqhdr.segmentation.info_present
                && (!seqhdr.segmentation.adaptive || gb.get_bit() != 0)
            {
                hdr.segmentation.d = seqhdr.segmentation.d;
            } else {
                parse_seg_info(
                    &mut hdr.segmentation.d,
                    gb,
                    8 << seqhdr.segmentation.ext as u32,
                );
            }
            if hdr.primary_ref_frame == PRIMARY_REF_NONE {
                hdr.segmentation.update_map = 1;
            } else {
                hdr.segmentation.update_map = gb.get_bit() as u8;
                if hdr.segmentation.update_map != 0 && hdr.frame_type != FrameType::Switch {
                    hdr.segmentation.temporal = gb.get_bit() as u8;
                }
            }
            let mut m = hdr.segmentation.d.skip_mask | hdr.segmentation.d.globalmv_mask;
            hdr.segmentation.preskip = (m != 0) as u8;
            m |= hdr.segmentation.d.delta_q_mask;
            hdr.segmentation.last_active_segid = if m != 0 { ulog2(m as u32) as i8 } else { -1 };
        }

        // qm
        hdr.quant.qm.enabled = gb.get_bit() as u8;
        if hdr.quant.qm.enabled != 0 {
            hdr.quant.qm.num = if hdr.segmentation.enabled != 0 {
                gb.get_bits(2) as u8 + 1
            } else {
                1
            };
            for n in 0..hdr.quant.qm.num as usize {
                hdr.quant.qm.y[n] = gb.get_bits(4) as u8;
                if seqhdr.layout != PixelLayout::I400 {
                    if gb.get_bit() != 0 {
                        hdr.quant.qm.u[n] = hdr.quant.qm.y[n];
                        hdr.quant.qm.v[n] = hdr.quant.qm.y[n];
                    } else {
                        hdr.quant.qm.u[n] = gb.get_bits(4) as u8;
                        hdr.quant.qm.v[n] = if seqhdr.separate_uv_delta_q {
                            gb.get_bits(4) as u8
                        } else {
                            hdr.quant.qm.u[n]
                        };
                    }
                }
            }
        }

        // delta q
        if hdr.quant.yac != 0 {
            hdr.delta.q.present = gb.get_bit() as u8;
            if hdr.delta.q.present != 0 {
                hdr.delta.q.res_log2 = gb.get_bits(2) as u8;
            }
        }

        // lossless
        let delta_lossless = hdr.quant.ydc_delta == 0
            && hdr.quant.udc_delta == 0
            && hdr.quant.uac_delta == 0
            && hdr.quant.vdc_delta == 0
            && hdr.quant.vac_delta == 0;
        hdr.all_lossless = 1;
        hdr.any_lossless = 0;
        for i in 0..MAX_SEGMENTS {
            hdr.segmentation.qidx[i] = if hdr.segmentation.enabled != 0 {
                iclip_u8(hdr.quant.yac as i32 + hdr.segmentation.d.delta_q[i] as i32) as u8
            } else {
                hdr.quant.yac as u8
            };
            hdr.segmentation.lossless[i] = (hdr.segmentation.qidx[i] == 0 && delta_lossless) as u8;
            hdr.all_lossless &= hdr.segmentation.lossless[i];
            hdr.any_lossless |= hdr.segmentation.lossless[i];
        }

        if hdr.all_lossless == 0 {
            hdr.tcq = if seqhdr.tcq == AdaptiveBoolean::Adaptive {
                gb.get_bit() as u8
            } else {
                seqhdr.tcq as u8
            };
        }
        if hdr.all_lossless == 0 && hdr.tcq == 0 && seqhdr.parity_hiding {
            hdr.parity_hiding = gb.get_bit() as u8;
        }

        // deblock
        if hdr.all_lossless == 0 {
            if hdr.frame_type == FrameType::Inter && seqhdr.db_sub_pu {
                hdr.deblock.sub_pu = gb.get_bit() as u8;
            }
            hdr.deblock.level_y[0] = gb.get_bit() as u8;
            hdr.deblock.level_y[1] = gb.get_bit() as u8;
            if seqhdr.layout != PixelLayout::I400
                && (hdr.deblock.level_y[0] != 0 || hdr.deblock.level_y[1] != 0)
            {
                hdr.deblock.level_u = gb.get_bit() as u8;
                hdr.deblock.level_v = gb.get_bit() as u8;
            }
            let bits = seqhdr.df_par_bits as i32;
            let off = 1i32 << (bits - 1);
            if hdr.deblock.level_y[0] != 0 && gb.get_bit() != 0 {
                hdr.deblock.delta_q_y[0] = (gb.get_bits(bits) as i32 - off) as i8;
            }
            if hdr.deblock.level_y[1] != 0 {
                hdr.deblock.delta_q_y[1] = if gb.get_bit() != 0 {
                    (gb.get_bits(bits) as i32 - off) as i8
                } else {
                    hdr.deblock.delta_q_y[0]
                };
            }
            if hdr.deblock.level_u != 0 && gb.get_bit() != 0 {
                hdr.deblock.delta_q_u = (gb.get_bits(bits) as i32 - off) as i8;
            }
            if hdr.deblock.level_v != 0 && gb.get_bit() != 0 {
                hdr.deblock.delta_q_v = (gb.get_bits(bits) as i32 - off) as i8;
            }
        }

        // gdf
        if hdr.all_lossless == 0 && seqhdr.gdf {
            let gdf_bs = 128 << (hdr.sb128 == 2) as i32;
            let mut gdf_val: u8 = (seqhdr.reduced_still_picture_header || gb.get_bit() != 0) as u8;
            if gdf_val != 0 {
                if imax(hdr.width, hdr.height) > gdf_bs {
                    gdf_val += gb.get_bit() as u8;
                }
                let qp_base = if hdr.is_key_or_intra() { 85 } else { 110 };
                let qp_diff = hdr.quant.yac as i32 - qp_base - 48 * seqhdr.hbd as i32;
                let qp_idx_offset = gb.get_bits(2) as i32;
                hdr.gdf.qp_idx = iclip((qp_diff - 37) / 25, 0, 2) as u8 + qp_idx_offset as u8;
                hdr.gdf.scale = gb.get_bits(2) as u8 + 1;
            }
            hdr.gdf.enabled = match gdf_val {
                0 => AdaptiveBoolean::Off,
                1 => AdaptiveBoolean::On,
                _ => AdaptiveBoolean::Adaptive,
            };
        }

        // cdef
        if hdr.all_lossless == 0 && seqhdr.cdef {
            hdr.cdef.enabled = (seqhdr.reduced_still_picture_header || gb.get_bit() != 0) as u8;
            if hdr.cdef.enabled != 0 {
                hdr.cdef.damping = gb.get_bits(2) as u8 + 3;
                hdr.cdef.n_strengths = gb.get_bits(3) as u8 + 1;
                hdr.cdef.on_skiptx = if seqhdr.cdef_on_skiptx == AdaptiveBoolean::Adaptive {
                    gb.get_bit() as u8
                } else {
                    seqhdr.cdef_on_skiptx as u8
                };
                for i in 0..hdr.cdef.n_strengths as usize {
                    let b = gb.get_bit() as i32;
                    hdr.cdef.y_strength[i] = gb.get_bits(6 - 4 * b) as u8;
                    if seqhdr.layout != PixelLayout::I400 {
                        let b = gb.get_bit() as i32;
                        hdr.cdef.uv_strength[i] = gb.get_bits(6 - 4 * b) as u8;
                    }
                }
            }
        }

        let n_bits_ref = if hdr.n_ref_frames <= 2 {
            hdr.n_ref_frames as i32 - 1
        } else {
            1 + ulog2(hdr.n_ref_frames as u32 - 1)
        };

        // restoration
        if hdr.all_lossless == 0 && seqhdr.restoration {
            for p in 0..3usize {
                let disable_mask = seqhdr.rst_disable_mask[if p != 0 { 1 } else { 0 }];
                hdr.restoration.p[p].restoration_type = if disable_mask == 0 {
                    gb.get_bits(2) as u8
                } else if disable_mask == 3 {
                    RestorationType::None as u8
                } else {
                    gb.get_bit() as u8 * (3 - disable_mask)
                };

                if hdr.restoration.p[p].restoration_type >= RestorationType::NsWiener as u8 {
                    let is_inter = hdr.is_inter_or_switch();
                    let pd = &mut hdr.restoration.p[p].ns;
                    pd.frame_filters_on = gb.get_bit() as u8;
                    if pd.frame_filters_on != 0 {
                        if is_inter {
                            pd.temporal = gb.get_bit() as u8;
                        }
                        if pd.temporal != 0 {
                            let mut r#ref = 0u8;
                            if n_bits_ref > 0 {
                                r#ref = gb.get_bits(n_bits_ref) as u8;
                                pd.refidx = r#ref;
                                if r#ref >= hdr.n_ref_frames {
                                    return Err(TealdustError::InvalidData);
                                }
                            }
                            let refhdr = ref_slot(refs, hdr.refidx[r#ref as usize] as i32)?
                                .p
                                .frame_hdr
                                .as_ref()
                                .ok_or(TealdustError::InvalidData)?;
                            let mut rpd = &refhdr.restoration.p[p].ns;
                            if rpd.frame_filters_on == 0 && p != 0 {
                                rpd = &refhdr.restoration.p[3 - p].ns;
                            }
                            if rpd.frame_filters_on == 0 {
                                return Err(TealdustError::InvalidData);
                            }
                            pd.num_classes_idx = rpd.num_classes_idx;
                            pd.num_classes = rpd.num_classes;
                        } else {
                            let val = gb.get_bits(3) as u8;
                            pd.num_classes_idx = val;
                            pd.num_classes = 1
                                + val
                                + imax(val as i32 - 3, 0) as u8
                                + imax(val as i32 - 5, 0) as u8 * 2;
                        }
                    } else {
                        pd.num_classes_idx = 0;
                        pd.num_classes = 1;
                    }
                }
            }

            hdr.restoration.unit_size[0] = 9;
            if hdr.restoration.p[0].restoration_type != 0 {
                if gb.get_bit() != 0 {
                    hdr.restoration.unit_size[0] -= 1;
                } else if hdr.sb128 < 2 && gb.get_bit() == 0 {
                    hdr.restoration.unit_size[0] -= 2 + (hdr.sb128 == 0 && gb.get_bit() == 0) as u8;
                }
            }

            let ss = (seqhdr.layout != PixelLayout::I444) as u8;
            hdr.restoration.unit_size[1] = 9 - ss;
            if hdr.restoration.p[1].restoration_type != 0
                || hdr.restoration.p[2].restoration_type != 0
            {
                if gb.get_bit() != 0 {
                    hdr.restoration.unit_size[1] -= 1;
                } else if hdr.sb128 < 2 && gb.get_bit() == 0 {
                    hdr.restoration.unit_size[1] -= 2 + (hdr.sb128 == 0 && gb.get_bit() == 0) as u8;
                }
                if hdr.restoration.unit_size[1] < 6 - seqhdr.ss_ver {
                    return Err(TealdustError::InvalidData);
                }
            }

            // NS wiener filter parsing
            for p in 0..3usize {
                let mut ref_filters = [[0i8; 18]; 48];
                if hdr.restoration.p[p].ns.frame_filters_on == 0 {
                    continue;
                }
                let n_feat = 16 + 2 * (p != 0) as usize;
                let n_ref_filters = if seqhdr.rst_disable_mask[if p != 0 { 1 } else { 0 }] & 1 != 0
                {
                    16
                } else {
                    48 - hdr.restoration.p[p].ns.num_classes as usize
                };

                if hdr.restoration.p[p].ns.temporal != 0 {
                    let ref_hdr = refs
                        [hdr.refidx[hdr.restoration.p[p].ns.refidx as usize] as usize]
                        .p
                        .frame_hdr
                        .as_ref()
                        .ok_or(TealdustError::InvalidData)?;
                    let mut rpd = &ref_hdr.restoration.p[p].ns;
                    if rpd.frame_filters_on == 0 {
                        rpd = &ref_hdr.restoration.p[3 - p].ns;
                    }
                    let nc = hdr.restoration.p[p].ns.num_classes as usize;
                    for n in 0..nc {
                        hdr.restoration.p[p].ns.filter[n][..n_feat]
                            .copy_from_slice(&rpd.filter[n][..n_feat]);
                    }
                    continue;
                }

                let mut i = 0usize;
                for r in 0..hdr.n_ref_frames as usize {
                    let ref_hdr = ref_slot(refs, hdr.refidx[r] as i32)?
                        .p
                        .frame_hdr
                        .as_ref()
                        .ok_or(TealdustError::InvalidData)?;
                    let dirs: &[i8] = &[0, 1, -1];
                    let mut dir = dirs[if p == 0 {
                        0
                    } else if p == 1 {
                        1
                    } else {
                        2
                    }];
                    let mut p2 = p as i32;
                    loop {
                        let rpd = &ref_hdr.restoration.p[p2 as usize].ns;
                        if rpd.frame_filters_on != 0 {
                            let n_classes =
                                imin(n_ref_filters as i32 - i as i32, rpd.num_classes as i32)
                                    as usize;
                            for n in 0..n_classes {
                                ref_filters[i][..n_feat].copy_from_slice(&rpd.filter[n][..n_feat]);
                                i += 1;
                            }
                        }
                        if dir == 0 {
                            break;
                        }
                        p2 += dir as i32;
                        dir = 0;
                    }
                }

                let n_filters = if seqhdr.rst_disable_mask[if p != 0 { 1 } else { 0 }] & 1 != 0 {
                    16usize
                } else {
                    64
                };
                let n_classes = hdr.restoration.p[p].ns.num_classes as usize;
                let mut grp_cnt = [0u8; 3];
                let mut grp_ref_cnt = [0u8; 3];
                grp_cnt[0] = n_classes as u8;
                grp_cnt[1] = i as u8;
                grp_cnt[2] = (n_filters - n_classes - i) as u8;
                let mut filter_refs = [0u8; 64];
                let mut pred_grp: usize = 2 - (grp_cnt[1] > 2) as usize;
                let nnz_grps = 1 + (grp_cnt[1] != 0) as i32 + (grp_cnt[2] != 0) as i32;
                for n in 0..n_classes {
                    let group = if nnz_grps == 1 || gb.get_bit() == 0 {
                        pred_grp
                    } else if nnz_grps == 2 {
                        2 - (grp_cnt[2] == 0) as usize - pred_grp
                    } else if gb.get_bit() != 0 {
                        2 - (pred_grp == 2) as usize
                    } else {
                        (pred_grp == 0) as usize
                    };
                    grp_ref_cnt[group] += 1;
                    if grp_ref_cnt[group] as usize + (group < pred_grp) as usize
                        > grp_ref_cnt[pred_grp] as usize
                    {
                        pred_grp = group;
                    }
                    let base = grp_cnt[0] as usize * (group != 0) as usize
                        + grp_cnt[1] as usize * (group == 2) as usize;
                    let range = if group != 0 {
                        grp_cnt[group] as u32
                    } else {
                        n as u32 + 1
                    };
                    filter_refs[n] = (base as u32
                        + if range == 1 {
                            0
                        } else {
                            gb.get_bits_subexp_u(range >> 1, range, 4)
                        }) as u8;
                }
                let mut exact_match_mask: u32 = 0;
                for n in 0..n_classes {
                    exact_match_mask |= gb.get_bit() << n;
                }
                let masks: &[u32] = if p != 0 {
                    &SUBSET_MASKS_UV
                } else {
                    &SUBSET_MASKS_Y
                };
                let cf_range: &[[i8; 2]] = if p != 0 {
                    &NS_WIENER_COEF_RANGE_UV
                } else {
                    &NS_WIENER_COEF_RANGE_Y
                };
                static SHUFFLED_INDEX: [u8; 64] = [
                    16, 7, 58, 21, 12, 61, 26, 38, 18, 30, 50, 45, 23, 49, 43, 62, 42, 54, 27, 36,
                    17, 44, 32, 34, 4, 24, 52, 31, 37, 11, 33, 19, 35, 6, 22, 53, 63, 25, 41, 47,
                    1, 59, 0, 28, 40, 55, 48, 8, 5, 51, 9, 46, 56, 60, 15, 2, 13, 14, 57, 29, 3,
                    20, 39, 10,
                ];
                static ZERO: [i8; 18] = [0; 18];
                for n in 0..n_classes {
                    let r = filter_refs[n] as usize;
                    let ref_filter: [i8; 18] = if r == 0 {
                        ZERO
                    } else if r < n_classes {
                        hdr.restoration.p[p].ns.filter[r - 1]
                    } else if r < n_classes + grp_cnt[1] as usize {
                        ref_filters[r - n_classes]
                    } else {
                        let idx = SHUFFLED_INDEX[r - n_classes - grp_cnt[1] as usize] as usize;
                        let mut tmp = [0i8; 18];
                        tmp[..16].copy_from_slice(&WIENER_NS_FILTERS[idx]);
                        tmp
                    };
                    if exact_match_mask & (1 << n) != 0 {
                        hdr.restoration.p[p].ns.filter[n][..n_feat]
                            .copy_from_slice(&ref_filter[..n_feat]);
                        continue;
                    }
                    hdr.restoration.p[p].ns.filter[n][..n_feat].fill(0);
                    let mut s = 0usize;
                    while s < 3 - (p != 0) as usize {
                        if gb.get_bit() == 0 {
                            break;
                        }
                        s += 1;
                    }
                    let mask = masks[s];
                    let mut m = mask;
                    for ii in 0..n_feat {
                        if m & 1 != 0 {
                            let nbits = cf_range[ii][0] as i32;
                            hdr.restoration.p[p].ns.filter[n][ii] = gb.get_bits_subexp_u(
                                (ref_filter[ii] - cf_range[ii][1]) as u32,
                                1 << nbits,
                                nbits - 3,
                            )
                                as i8
                                + cf_range[ii][1];
                        }
                        m >>= 1;
                    }
                }
            }
        }

        // ccso
        if hdr.all_lossless == 0 && seqhdr.ccso {
            hdr.ccso.enabled = (seqhdr.reduced_still_picture_header || gb.get_bit() != 0) as u8;
            if hdr.ccso.enabled != 0 {
                let n_planes = if seqhdr.layout == PixelLayout::I400 {
                    1
                } else {
                    3
                };
                for p in 0..n_planes {
                    hdr.ccso.p[p].enabled = gb.get_bit() as u8;
                    if hdr.ccso.p[p].enabled == 0 {
                        continue;
                    }
                    if hdr.is_inter_or_switch() {
                        hdr.ccso.p[p].reuse = gb.get_bit() as u8;
                        hdr.ccso.p[p].sb_reuse = gb.get_bit() as u8;
                        if hdr.ccso.p[p].reuse != 0 || hdr.ccso.p[p].sb_reuse != 0 {
                            let mut r#ref = 0u8;
                            if n_bits_ref > 0 {
                                r#ref = gb.get_bits(n_bits_ref) as u8;
                                hdr.ccso.p[p].refidx = r#ref;
                                if r#ref >= hdr.n_ref_frames {
                                    return Err(TealdustError::InvalidData);
                                }
                            }
                            let refhdr = ref_slot(refs, hdr.refidx[r#ref as usize] as i32)?
                                .p
                                .frame_hdr
                                .as_ref()
                                .ok_or(TealdustError::InvalidData)?;
                            if hdr.ccso.p[p].reuse != 0 {
                                let w4 = (hdr.width + 3) >> 2;
                                let h4 = (hdr.height + 3) >> 2;
                                let rw4 = (refhdr.width + 3) >> 2;
                                let rh4 = (refhdr.height + 3) >> 2;
                                if w4 != rw4 || h4 != rh4 || refhdr.ccso.p[p].enabled == 0 {
                                    return Err(TealdustError::InvalidData);
                                }
                            }
                        }
                    }
                    if hdr.ccso.p[p].reuse == 0 {
                        hdr.ccso.p[p].bo_only = gb.get_bit() as u8;
                        hdr.ccso.p[p].scale_idx = gb.get_bits(2) as u8;
                        if hdr.ccso.p[p].bo_only != 0 {
                            hdr.ccso.p[p].max_band_log2 = gb.get_bits(3) as u8;
                        } else {
                            hdr.ccso.p[p].quant_idx = gb.get_bits(2) as u8;
                            hdr.ccso.p[p].ext_filter_support = gb.get_bits(3) as u8;
                            if hdr.ccso.p[p].ext_filter_support == 7 {
                                return Err(TealdustError::InvalidData);
                            }
                            let si = hdr.ccso.p[p].scale_idx as usize;
                            let qi = hdr.ccso.p[p].quant_idx as usize;
                            if CCSO_QUANT_SZ[si][qi] != 0 {
                                hdr.ccso.p[p].edge_clf = gb.get_bit() as u8;
                            }
                            hdr.ccso.p[p].max_band_log2 = gb.get_bits(2) as u8;
                        }
                        let n_edge_off_intervals = if hdr.ccso.p[p].bo_only != 0 {
                            1
                        } else {
                            3 - hdr.ccso.p[p].edge_clf as usize
                        };
                        let max_band = 1usize << hdr.ccso.p[p].max_band_log2;
                        hdr.ccso.p[p].filter_off = [0; 64];
                        for n in 0..n_edge_off_intervals {
                            for m_idx in 0..n_edge_off_intervals {
                                let fo_base = n * 16 + m_idx * 4;
                                for o in 0..max_band {
                                    let mut off = 0u8;
                                    while off < 7 {
                                        if gb.get_bit() == 0 {
                                            break;
                                        }
                                        off += 1;
                                    }
                                    hdr.ccso.p[p].filter_off[fo_base + (o >> 1)] |=
                                        off << (4 * (o & 1));
                                }
                            }
                        }
                    } else {
                        let ccso_ref = *hdr
                            .refidx
                            .get(hdr.ccso.p[p].refidx as usize)
                            .ok_or(TealdustError::InvalidData)?;
                        let refhdr = ref_slot(refs, ccso_ref as i32)?
                            .p
                            .frame_hdr
                            .as_ref()
                            .ok_or(TealdustError::InvalidData)?;
                        let rp = &refhdr.ccso.p[p];
                        hdr.ccso.p[p].bo_only = rp.bo_only;
                        hdr.ccso.p[p].scale_idx = rp.scale_idx;
                        hdr.ccso.p[p].quant_idx = rp.quant_idx;
                        hdr.ccso.p[p].ext_filter_support = rp.ext_filter_support;
                        hdr.ccso.p[p].edge_clf = rp.edge_clf;
                        hdr.ccso.p[p].max_band_log2 = rp.max_band_log2;
                        hdr.ccso.p[p].filter_off = rp.filter_off;
                    }
                }
            }
        }

        if hdr.all_lossless == 0 {
            hdr.txfm_mode = if gb.get_bit() != 0 {
                TxfmMode::Switchable
            } else {
                TxfmMode::Largest
            };
        }

        if hdr.is_inter_or_switch() {
            hdr.switchable_comp_refs = gb.get_bit() as u8;
            hdr.skip_mode_enabled = gb.get_bit() as u8;
            if seqhdr.bawp {
                hdr.bawp = gb.get_bit() as u8;
            }
            if seqhdr.motion_modes & (1 << MotionMode::WarpDelta as u8) != 0 {
                hdr.warp_motion = gb.get_bit() as u8;
            }
        }

        hdr.reduced_txtp_set = gb.get_bits(2) as u8;

        for i in 0..7 {
            hdr.gmv.m[i] = DEFAULT_WM_PARAMS;
        }
        if hdr.is_inter_or_switch() && seqhdr.global_motion && gb.get_bit() != 0 {
            if hdr.n_ref_frames == 0 {
                return Err(TealdustError::InvalidData);
            }
            hdr.gmv.r#ref = gb.get_uniform(hdr.n_ref_frames as u32 + 1) as u8;
            let (ref_base_mat, in_dist);
            if hdr.gmv.r#ref == hdr.n_ref_frames {
                ref_base_mat = DEFAULT_WM_PARAMS.matrix;
                in_dist = 1;
            } else {
                let refidx = *hdr
                    .refidx
                    .get(hdr.gmv.r#ref as usize)
                    .ok_or(TealdustError::InvalidData)?;
                let refhdr = ref_slot(refs, refidx as i32)?
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(TealdustError::InvalidData)?;
                if refhdr.n_ref_frames == 0 {
                    ref_base_mat = DEFAULT_WM_PARAMS.matrix;
                    in_dist = 1;
                } else {
                    hdr.gmv.refref = if refhdr.n_ref_frames == 1 {
                        0
                    } else {
                        gb.get_uniform(refhdr.n_ref_frames as u32) as u8
                    };
                    ref_base_mat = refhdr.gmv.m[hdr.gmv.refref as usize].matrix;
                    in_dist = get_poc_diff(
                        seqhdr.order_hint_n_bits as i32,
                        refhdr.frame_offset as i32,
                        ref_slot(refs, refidx as i32)?.refpoc[hdr.gmv.refref as usize] as i32,
                    );
                }
            }
            for i in 0..hdr.n_ref_frames as usize {
                hdr.gmv.m[i].wm_type = if gb.get_bit() == 0 {
                    WarpedMotionType::Identity
                } else if gb.get_bit() != 0 {
                    WarpedMotionType::RotZoom
                } else {
                    WarpedMotionType::Affine
                };
                if hdr.gmv.m[i].wm_type == WarpedMotionType::Identity {
                    continue;
                }
                let mat = &mut hdr.gmv.m[i].matrix;
                let mut ref_mat = [0i32; 6];
                let out_dist = get_poc_diff(
                    seqhdr.order_hint_n_bits as i32,
                    hdr.frame_offset as i32,
                    ref_slot(refs, hdr.refidx[i] as i32)?
                        .p
                        .frame_hdr
                        .as_ref()
                        .ok_or(TealdustError::InvalidData)?
                        .frame_offset as i32,
                );
                rescale_matrix(&mut ref_mat, &ref_base_mat, in_dist, out_dist);

                if hdr.gmv.m[i].wm_type >= WarpedMotionType::RotZoom {
                    mat[2] =
                        (1 << 16) + 64 * gb.get_bits_subexp((ref_mat[2] - (1 << 16)) >> 6, 512);
                    mat[3] = 64 * gb.get_bits_subexp(ref_mat[3] >> 6, 512);
                }
                if hdr.gmv.m[i].wm_type == WarpedMotionType::Affine {
                    mat[4] = 64 * gb.get_bits_subexp(ref_mat[4] >> 6, 512);
                    mat[5] =
                        (1 << 16) + 64 * gb.get_bits_subexp((ref_mat[5] - (1 << 16)) >> 6, 512);
                } else {
                    mat[4] = -mat[3];
                    mat[5] = mat[2];
                }
                mat[0] = gb.get_bits_subexp(ref_mat[0] >> 13, 0x4000) * 8192;
                mat[1] = gb.get_bits_subexp(ref_mat[1] >> 13, 0x4000) * 8192;
            }
        }
    } // end !tip_output_frame

    // grain
    if seqhdr.film_grain_present && (hdr.show_immediate != 0 || hdr.show_implicit != 0) {
        hdr.film_grain.present = (seqhdr.reduced_still_picture_header || gb.get_bit() != 0) as u8;
        if hdr.film_grain.present != 0 {
            hdr.film_grain.id = gb.get_bits(3) as u8;
            hdr.film_grain.seed = gb.get_bits(16);
        }
    }

    Ok(hdr)
}

pub fn parse_tile_hdr(hdr: &FrameHeader, tile: &mut crate::internal::TileGroup, gb: &mut GetBits) {
    let n_tiles = hdr.tiling.t.cols as i32 * hdr.tiling.t.rows as i32;
    let have_tile_pos = if n_tiles > 1 {
        gb.get_bit() != 0
    } else {
        false
    };
    if have_tile_pos {
        let n_bits = hdr.tiling.t.log2_cols as i32 + hdr.tiling.t.log2_rows as i32;
        tile.start = gb.get_bits(n_bits) as i32;
        tile.end = gb.get_bits(n_bits) as i32;
    } else {
        tile.start = 0;
        tile.end = n_tiles - 1;
    }
}

pub fn parse_fgm_hdr(
    gb: &mut GetBits,
    seq_layout: PixelLayout,
) -> Result<[Option<FilmGrainData>; 8]> {
    let mask = gb.get_bits(8) as u8;
    let layout_idx = gb.get_vlc();
    if layout_idx > 3 {
        return Err(TealdustError::InvalidData);
    }
    let layout = LAYOUTS[layout_idx as usize];
    if layout != seq_layout {
        return Err(TealdustError::InvalidData);
    }

    let mut result: [Option<FilmGrainData>; 8] = Default::default();
    for idx in 0..8 {
        if mask & (1 << idx) == 0 {
            continue;
        }
        result[idx] = Some(parse_film_grain_data(gb, layout)?);
    }

    Ok(result)
}

pub fn parse_cll(gb: &mut GetBits) -> ContentLightLevel {
    ContentLightLevel {
        max_content_light_level: gb.get_bits(16) as u16,
        max_frame_average_light_level: gb.get_bits(16) as u16,
    }
}

pub fn parse_mdcv(gb: &mut GetBits) -> MasteringDisplay {
    let mut md = MasteringDisplay::default();
    for i in 0..3 {
        md.primaries[i][0] = gb.get_bits(16) as u16;
        md.primaries[i][1] = gb.get_bits(16) as u16;
    }
    md.white_point[0] = gb.get_bits(16) as u16;
    md.white_point[1] = gb.get_bits(16) as u16;
    md.max_luminance = gb.get_bits(32);
    md.min_luminance = gb.get_bits(32);
    md
}

pub fn parse_ci_hdr(ci: &mut ContentInterpretation, gb: &mut GetBits) -> Result<()> {
    ci.scan_type = match gb.get_bits(2) {
        0 => ScanType::Unknown,
        1 => ScanType::Progressive,
        2 => ScanType::Interlace,
        3 => ScanType::InterlaceComplementary,
        _ => unreachable!(),
    };
    ci.color_description_present = gb.get_bit() != 0;
    ci.chroma_sample_position_present = gb.get_bit() != 0;
    ci.aspect_ratio_info_present = gb.get_bit() != 0;
    ci.timing_info_present = gb.get_bit() != 0;
    ci.extension_present = gb.get_bit() != 0;
    let _ = gb.get_bit(); // reserved

    if ci.color_description_present {
        let desc_type = gb.get_golomb(2);
        ci.color.desc_type = match desc_type {
            0 => ColorDescription::Explicit,
            1 => ColorDescription::Bt709Sdr,
            2 => ColorDescription::Bt2100Pq,
            3 => ColorDescription::Bt2100Hlg,
            4 => ColorDescription::Srgb,
            5 => ColorDescription::SrgbSycc,
            _ => ColorDescription::Explicit, // unknown → treat as explicit with unknown values
        };
        match ci.color.desc_type {
            ColorDescription::Explicit => {
                if desc_type == 0 {
                    ci.color.pri = u8_to_color_pri(gb.get_bits(8) as u8);
                    ci.color.trc = u8_to_trc(gb.get_bits(8) as u8);
                    ci.color.mtrx = u8_to_mc(gb.get_bits(8) as u8);
                } else {
                    ci.color.pri = ColorPrimaries::Unknown;
                    ci.color.trc = TransferCharacteristics::Unknown;
                    ci.color.mtrx = MatrixCoefficients::Unknown;
                }
            }
            ColorDescription::Bt709Sdr => {
                ci.color.pri = ColorPrimaries::Bt709;
                ci.color.trc = TransferCharacteristics::Bt709;
                ci.color.mtrx = MatrixCoefficients::Bt470Bg;
            }
            ColorDescription::Bt2100Pq => {
                ci.color.pri = ColorPrimaries::Bt2020;
                ci.color.trc = TransferCharacteristics::Smpte2084;
                ci.color.mtrx = MatrixCoefficients::Bt2020Ncl;
            }
            ColorDescription::Bt2100Hlg => {
                ci.color.pri = ColorPrimaries::Bt2020;
                ci.color.trc = TransferCharacteristics::Bt2020_10Bit;
                ci.color.mtrx = MatrixCoefficients::Bt2020Ncl;
            }
            ColorDescription::Srgb => {
                ci.color.pri = ColorPrimaries::Bt709;
                ci.color.trc = TransferCharacteristics::Srgb;
                ci.color.mtrx = MatrixCoefficients::Identity;
            }
            ColorDescription::SrgbSycc => {
                ci.color.pri = ColorPrimaries::Bt709;
                ci.color.trc = TransferCharacteristics::Srgb;
                ci.color.mtrx = MatrixCoefficients::Bt470Bg;
            }
        }
        ci.color.range = gb.get_bit() as u8;
    } else {
        ci.color.pri = ColorPrimaries::Unknown;
        ci.color.trc = TransferCharacteristics::Unknown;
        ci.color.mtrx = MatrixCoefficients::Unknown;
    }

    if ci.chroma_sample_position_present {
        ci.chr[0] = u32_to_chr(gb.get_vlc());
        ci.chr[1] = if ci.scan_type == ScanType::Progressive {
            ci.chr[0]
        } else {
            u32_to_chr(gb.get_vlc())
        };
    } else {
        ci.chr[0] = ChromaSamplePosition::Unknown;
        ci.chr[1] = ChromaSamplePosition::Unknown;
    }

    if ci.aspect_ratio_info_present {
        let sar_type = gb.get_bits(8) as u8;
        match sar_type {
            0 => ci.sar.sar_type = AspectRatio::Unknown,
            1 => {
                ci.sar.sar_type = AspectRatio::Sar1_1;
                ci.sar.w = 1;
                ci.sar.h = 1;
            }
            2 => {
                ci.sar.sar_type = AspectRatio::Sar12_11;
                ci.sar.w = 12;
                ci.sar.h = 11;
            }
            3 => {
                ci.sar.sar_type = AspectRatio::Sar10_11;
                ci.sar.w = 10;
                ci.sar.h = 11;
            }
            4 => {
                ci.sar.sar_type = AspectRatio::Sar16_11;
                ci.sar.w = 16;
                ci.sar.h = 11;
            }
            5 => {
                ci.sar.sar_type = AspectRatio::Sar40_33;
                ci.sar.w = 40;
                ci.sar.h = 33;
            }
            6 => {
                ci.sar.sar_type = AspectRatio::Sar24_11;
                ci.sar.w = 24;
                ci.sar.h = 11;
            }
            7 => {
                ci.sar.sar_type = AspectRatio::Sar20_11;
                ci.sar.w = 20;
                ci.sar.h = 11;
            }
            8 => {
                ci.sar.sar_type = AspectRatio::Sar32_11;
                ci.sar.w = 32;
                ci.sar.h = 11;
            }
            9 => {
                ci.sar.sar_type = AspectRatio::Sar80_33;
                ci.sar.w = 80;
                ci.sar.h = 33;
            }
            10 => {
                ci.sar.sar_type = AspectRatio::Sar18_11;
                ci.sar.w = 18;
                ci.sar.h = 11;
            }
            11 => {
                ci.sar.sar_type = AspectRatio::Sar15_11;
                ci.sar.w = 15;
                ci.sar.h = 11;
            }
            12 => {
                ci.sar.sar_type = AspectRatio::Sar64_33;
                ci.sar.w = 64;
                ci.sar.h = 33;
            }
            13 => {
                ci.sar.sar_type = AspectRatio::Sar160_99;
                ci.sar.w = 160;
                ci.sar.h = 99;
            }
            14 => {
                ci.sar.sar_type = AspectRatio::Sar4_3;
                ci.sar.w = 4;
                ci.sar.h = 3;
            }
            15 => {
                ci.sar.sar_type = AspectRatio::Sar3_2;
                ci.sar.w = 3;
                ci.sar.h = 2;
            }
            16 => {
                ci.sar.sar_type = AspectRatio::Sar2_1;
                ci.sar.w = 2;
                ci.sar.h = 1;
            }
            255 => {
                ci.sar.sar_type = AspectRatio::Explicit;
                ci.sar.w = gb.get_vlc();
                ci.sar.h = gb.get_vlc();
            }
            // 17–254 are reserved by the spec; treat as unspecified rather than
            // hard-failing so real-world encoders that emit reserved SAR values
            // still decode successfully.
            _ => {
                ci.sar.sar_type = AspectRatio::Unknown;
            }
        }
    }

    if ci.timing_info_present {
        ci.timing.num_units_in_display_tick = gb.get_bits(32);
        ci.timing.time_scale = gb.get_bits(32);
        if ci.timing.num_units_in_display_tick == 0 || ci.timing.time_scale == 0 {
            return Err(TealdustError::InvalidData);
        }
        ci.timing.equal_elemental_interval = gb.get_bit() as u8;
        if ci.timing.equal_elemental_interval != 0 {
            let t = gb.get_vlc();
            if t == u32::MAX {
                return Err(TealdustError::InvalidData);
            }
            ci.timing.num_ticks_per_elemental_duration = t + 1;
        }
    }

    Ok(())
}

fn u8_to_color_pri(v: u8) -> ColorPrimaries {
    match v {
        1 => ColorPrimaries::Bt709,
        2 => ColorPrimaries::Unknown,
        4 => ColorPrimaries::Bt470M,
        5 => ColorPrimaries::Bt470Bg,
        6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Xyz,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => ColorPrimaries::Unknown,
    }
}

fn u8_to_trc(v: u8) -> TransferCharacteristics {
    match v {
        1 => TransferCharacteristics::Bt709,
        2 => TransferCharacteristics::Unknown,
        4 => TransferCharacteristics::Bt470M,
        5 => TransferCharacteristics::Bt470Bg,
        6 => TransferCharacteristics::Bt601,
        7 => TransferCharacteristics::Smpte240,
        8 => TransferCharacteristics::Linear,
        9 => TransferCharacteristics::Log100,
        10 => TransferCharacteristics::Log100Sqrt10,
        11 => TransferCharacteristics::Iec61966,
        12 => TransferCharacteristics::Bt1361,
        13 => TransferCharacteristics::Srgb,
        14 => TransferCharacteristics::Bt2020_10Bit,
        15 => TransferCharacteristics::Bt2020_12Bit,
        16 => TransferCharacteristics::Smpte2084,
        17 => TransferCharacteristics::Smpte428,
        18 => TransferCharacteristics::Hlg,
        _ => TransferCharacteristics::Unknown,
    }
}

fn u8_to_mc(v: u8) -> MatrixCoefficients {
    match v {
        0 => MatrixCoefficients::Identity,
        1 => MatrixCoefficients::Bt709,
        2 => MatrixCoefficients::Unknown,
        4 => MatrixCoefficients::Fcc,
        5 => MatrixCoefficients::Bt470Bg,
        6 => MatrixCoefficients::Bt601,
        7 => MatrixCoefficients::Smpte240,
        8 => MatrixCoefficients::YCgCo,
        9 => MatrixCoefficients::Bt2020Ncl,
        10 => MatrixCoefficients::Bt2020Cl,
        11 => MatrixCoefficients::Smpte2085,
        12 => MatrixCoefficients::ChromatNcl,
        13 => MatrixCoefficients::ChromatCl,
        14 => MatrixCoefficients::Ictcp,
        15 => MatrixCoefficients::IptC2,
        16 => MatrixCoefficients::YcgcoRe,
        17 => MatrixCoefficients::YcgcoRo,
        _ => MatrixCoefficients::Unknown,
    }
}

fn u32_to_chr(v: u32) -> ChromaSamplePosition {
    match v {
        0 => ChromaSamplePosition::Left,
        1 => ChromaSamplePosition::Center,
        2 => ChromaSamplePosition::TopLeft,
        3 => ChromaSamplePosition::Top,
        4 => ChromaSamplePosition::BottomLeft,
        5 => ChromaSamplePosition::Bottom,
        _ => ChromaSamplePosition::Unknown,
    }
}

pub fn read_frame_size(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
    gb: &mut GetBits,
) -> Result<()> {
    if hdr.frame_size_override != 0 && hdr.is_inter_or_switch() {
        for i in 0..hdr.n_ref_frames as usize {
            if gb.get_bit() != 0 {
                let refhdr = ref_slot(refs, hdr.refidx[i] as i32)?
                    .p
                    .frame_hdr
                    .as_ref()
                    .ok_or(TealdustError::InvalidData)?;
                hdr.width = refhdr.width;
                hdr.height = refhdr.height;
                return Ok(());
            }
        }
    }
    if hdr.frame_size_override != 0 {
        hdr.width = gb.get_bits(seqhdr.width_n_bits as i32) as i32 + 1;
        hdr.height = gb.get_bits(seqhdr.height_n_bits as i32) as i32 + 1;
    } else {
        hdr.width = seqhdr.max_width;
        hdr.height = seqhdr.max_height;
    }
    Ok(())
}

pub fn get_ref_frames(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
    have_resolution: bool,
) -> i32 {
    struct Score {
        score: i32,
        poc: u8,
        pocdiff: i8,
        qidx: u16,
        mlayer: u8,
        _res_ratio_log2: i8,
    }
    let mut ref_info: [Score; 8] = std::array::from_fn(|_| Score {
        score: 0,
        poc: 0,
        pocdiff: 0,
        qidx: 0,
        mlayer: 0,
        _res_ratio_log2: 0,
    });
    let mut sort_idx = [0u8; 8];
    let mut n_refs = 0i32;
    let mut have_fwd_refs = false;
    let poc = hdr.frame_offset as i32;
    let nbits = seqhdr.order_hint_n_bits as i32;

    for n in 0..8 {
        if have_fwd_refs {
            break;
        }
        if let Some(refhdr) = refs[n].p.frame_hdr.as_ref() {
            have_fwd_refs = get_poc_diff(nbits, poc, refhdr.frame_offset as i32) < 0;
        }
    }

    let mlayer = hdr.mlayer_id as i32;
    let tlayer = hdr.tlayer_id as i32;
    let w = hdr.width;
    let h = hdr.height;
    let mut minq = 512i32;
    let mut maxq = -1i32;
    let mut last_refhdr_ptr: Option<*const FrameHeader> = None;

    for n in 0..8usize {
        let refhdr_arc = match refs[n].p.frame_hdr.as_ref() {
            Some(fh) => fh,
            None => continue,
        };
        let refhdr_ptr = Arc::as_ptr(refhdr_arc);
        if last_refhdr_ptr == Some(refhdr_ptr) {
            continue;
        }
        let refhdr = refhdr_arc.as_ref();

        if seqhdr.tlayer_dependency_present {
            if seqhdr.tlayer_dependencies[tlayer as usize] & (1 << refhdr.tlayer_id) == 0 {
                continue;
            }
        } else if tlayer < refhdr.tlayer_id as i32 {
            continue;
        }

        let ref_mlayer = refhdr.mlayer_id;
        if seqhdr.mlayer_dependency_present {
            if seqhdr.mlayer_dependencies[mlayer as usize] & (1 << ref_mlayer) == 0 {
                continue;
            }
        } else if mlayer < ref_mlayer as i32 {
            continue;
        }

        if have_resolution
            && (2 * w < refhdr.width
                || 2 * h < refhdr.height
                || w > 16 * refhdr.width
                || h > 16 * refhdr.height)
        {
            continue;
        }

        let ref_poc = refhdr.frame_offset;
        let pocdiff = get_poc_diff(nbits, poc, ref_poc as i32) as i8;
        let ref_qidx = refhdr.quant.yac;
        let res_ratio = -(ulog2((refhdr.width * refhdr.height) as u32) as i8);
        let tdist = (pocdiff as i32).abs() + mlayer - ref_mlayer as i32;
        let mut score = if have_fwd_refs {
            tdist << 6
        } else {
            128 - (128 >> imin(tdist, 6)) + imax(tdist - 6, 0)
        };
        score += res_ratio as i32 * (1 << 5) + ref_qidx as i32;

        ref_info[n] = Score {
            score,
            poc: ref_poc,
            pocdiff,
            qidx: ref_qidx,
            mlayer: ref_mlayer,
            _res_ratio_log2: res_ratio,
        };

        let mut m = 0usize;
        while m < n_refs as usize {
            let r2 = &ref_info[sort_idx[m] as usize];
            if score == r2.score && ref_poc == r2.poc && ref_mlayer == r2.mlayer {
                break;
            }
            m += 1;
        }
        if (m as i32) < n_refs {
            continue;
        }

        maxq = imax(ref_qidx as i32, maxq);
        minq = imin(ref_qidx as i32, minq);

        while m > 0 {
            let idx = sort_idx[m - 1] as usize;
            if ref_info[idx].score <= score {
                break;
            }
            sort_idx[m] = idx as u8;
            m -= 1;
        }
        sort_idx[m] = n as u8;
        n_refs += 1;
        last_refhdr_ptr = Some(refhdr_ptr);
    }

    if n_refs == 8 {
        let q_thr = (maxq + minq + 1) >> 1;
        let mut maxpocdiff = [0i32; 2];
        let mut num = [0i32; 2];
        let mut furthest_idx = [0usize; 2];
        for n in 0..8usize {
            let r = &ref_info[sort_idx[n] as usize];
            if (r.qidx as i32) < q_thr {
                continue;
            }
            if r.pocdiff > 0 {
                if (r.pocdiff as i32) > maxpocdiff[0] {
                    maxpocdiff[0] = r.pocdiff as i32;
                    furthest_idx[0] = n;
                }
                num[0] += 1;
            } else if r.pocdiff < 0 {
                if (r.pocdiff as i32) < maxpocdiff[1] {
                    maxpocdiff[1] = r.pocdiff as i32;
                    furthest_idx[1] = n;
                }
                num[1] += 1;
            }
        }
        let idx = if num[0] > num[1] {
            furthest_idx[0]
        } else if num[0] < num[1] {
            furthest_idx[1]
        } else {
            furthest_idx[if maxpocdiff[0] < -maxpocdiff[1] { 1 } else { 0 }]
        };
        if idx < 7 {
            sort_idx.copy_within(idx + 1..8, idx);
            sort_idx[7] = idx as u8;
        }
    }

    for n in 0..7usize {
        hdr.refidx[n] = sort_idx[if (n as i32) < n_refs { n } else { 0 }] as i8;
    }

    imin(7, n_refs)
}

pub fn find_tip_ref_frames(
    hdr: &mut FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
) -> Result<()> {
    let n_refs = hdr.n_ref_frames as usize;
    // n_refs >= 2 is required to pick two TIP references; the index arithmetic
    // below underflows for 0 and is degenerate for 1.
    if n_refs < 2 {
        hdr.tip.r#ref[0] = 0;
        hdr.tip.r#ref[1] = 0;
        return Ok(());
    }

    let poc = hdr.frame_offset as i32;
    let nbits = seqhdr.order_hint_n_bits as i32;
    let mut order = [0u8; 7];
    let mut refdist = [0i8; 7];
    let mut n_past = 0usize;

    for n in 0..n_refs {
        let refpoc = ref_slot(refs, hdr.refidx[n] as i32)?
            .p
            .frame_hdr
            .as_ref()
            .ok_or(TealdustError::InvalidData)?
            .frame_offset;
        let dist = get_poc_diff(nbits, refpoc as i32, poc);
        refdist[n] = dist as i8;
        let mut m = n;
        while m > 0 && (refdist[order[m - 1] as usize] as i32) > dist {
            order[m] = order[m - 1];
            m -= 1;
        }
        order[m] = n as u8;
        if dist < 0 {
            n_past += 1;
        }
    }

    if n_past == n_refs {
        hdr.tip.r#ref[0] = order[n_refs - 1] as i8;
        hdr.tip.r#ref[1] = order[n_refs - 2] as i8;
    } else if n_past == 0 {
        hdr.tip.r#ref[0] = order[0] as i8;
        hdr.tip.r#ref[1] = order[1] as i8;
    } else {
        hdr.tip.r#ref[0] = order[n_past - 1] as i8;
        hdr.tip.r#ref[1] = order[n_past] as i8;
    }
    Ok(())
}

pub fn derive_pri_sec_ref(
    hdr: &FrameHeader,
    seqhdr: &SequenceHeader,
    refs: &[RefState; 8],
) -> [i32; 2] {
    let mut result = [PRIMARY_REF_NONE as i32, PRIMARY_REF_NONE as i32];
    let mut best_qdiff = [0i32; 2];
    let mut best_pocdiff = [0i32; 2];
    let mut best_poc = [0i32; 2];
    let mut best = 0usize;
    let qidx = hdr.quant.yac as i32;
    let poc = hdr.frame_offset as i32;
    let nbits = seqhdr.order_hint_n_bits as i32;

    for i in 0..hdr.n_ref_frames as usize {
        let refhdr = match ref_slot(refs, hdr.refidx[i] as i32)
            .ok()
            .and_then(|r| r.p.frame_hdr.as_ref())
        {
            Some(fh) => fh,
            None => continue,
        };
        if refhdr.is_key_or_intra() {
            continue;
        }
        let ref_qidx = refhdr.quant.yac as i32;
        let qdiff = (ref_qidx - qidx).abs();
        let ref_poc = refhdr.frame_offset as i32;
        let pocdiff = get_poc_diff(nbits, poc, ref_poc).abs();
        for n in 0..2usize {
            let m = if n == 0 { best } else { 1 - best };
            if result[m] == PRIMARY_REF_NONE as i32
                || qdiff < best_qdiff[m]
                || (qdiff == best_qdiff[m]
                    && (pocdiff < best_pocdiff[m]
                        || (pocdiff == best_pocdiff[m]
                            && get_poc_diff(nbits, best_poc[m], ref_poc) < 0)))
            {
                let slot = 1 - best;
                result[slot] = i as i32;
                best_pocdiff[slot] = pocdiff;
                best_qdiff[slot] = qdiff;
                best_poc[slot] = ref_poc;
                if n == 0 {
                    best = 1 - best;
                }
                break;
            }
        }
    }

    if best != 0 {
        result.swap(0, 1);
    }
    result
}

fn u32_to_obu_type(v: u32) -> Option<ObuType> {
    match v {
        1 => Some(ObuType::SeqHdr),
        2 => Some(ObuType::Td),
        3 => Some(ObuType::MultiFrameHdr),
        4 => Some(ObuType::ClosedLoopKf),
        5 => Some(ObuType::OpenLoopKf),
        6 => Some(ObuType::LeadingTileGrp),
        7 => Some(ObuType::TileGrp),
        8 => Some(ObuType::Metadata),
        9 => Some(ObuType::MetadataGrp),
        10 => Some(ObuType::Switch),
        11 => Some(ObuType::LeadingSef),
        12 => Some(ObuType::Sef),
        13 => Some(ObuType::LeadingTip),
        14 => Some(ObuType::Tip),
        15 => Some(ObuType::BufRmTiming),
        16 => Some(ObuType::LayerCfgRec),
        17 => Some(ObuType::AtlasSeg),
        18 => Some(ObuType::OpPtSet),
        19 => Some(ObuType::Bridge),
        20 => Some(ObuType::Msdo),
        21 => Some(ObuType::Ras),
        22 => Some(ObuType::Qm),
        23 => Some(ObuType::Fgm),
        24 => Some(ObuType::ContentInterp),
        25 => Some(ObuType::Padding),
        _ => None,
    }
}

pub fn parse_obus(c: &mut DecoderContext, data: &[u8]) -> Result<usize> {
    use crate::levels::ObuMetaType;

    let mut hdr_gb = GetBits::new(data);

    let len = hdr_gb.get_uleb128() as usize;
    hdr_gb.bytealign();
    if hdr_gb.has_error() || len > hdr_gb.remaining_bytes() {
        return Err(TealdustError::InvalidData);
    }
    let body_start = hdr_gb.byte_pos();
    let total_consumed = body_start + len;

    let body = &data[body_start..body_start + len];
    let mut gb = GetBits::new(if body.is_empty() { &[0u8] } else { body });

    let has_extension = gb.get_bit() != 0;
    let obu_type_raw = gb.get_bits(5);
    let tlayer_id = gb.get_bits(2) as i32;

    let mut mlayer_id = 0i32;
    let mut xlayer_id = 0i32;
    if has_extension {
        mlayer_id = gb.get_bits(3) as i32;
        xlayer_id = gb.get_bits(5) as i32;
    }

    if gb.has_error() {
        return Err(TealdustError::InvalidData);
    }

    let obu_type = u32_to_obu_type(obu_type_raw);

    // skip OBUs not belonging to selected operating point
    if obu_type != Some(ObuType::SeqHdr)
        && obu_type != Some(ObuType::Td)
        && has_extension
        && c.operating_point_idc != 0
    {
        return Ok(total_consumed);
    }

    match obu_type {
        Some(ObuType::SeqHdr) => {
            let seq_hdr = parse_seq_hdr(&mut gb, c.strict_std_compliance)?;

            c.operating_point_idc = 0;
            let spatial_mask = c.operating_point_idc >> 8;
            c.max_spatial_id = if spatial_mask != 0 {
                ulog2(spatial_mask)
            } else {
                0
            };

            if c.seq_hdr.is_none() {
                c.frame_hdr = None;
            } else if c.seq_hdr.as_ref().is_none_or(|old| **old != seq_hdr) {
                c.frame_hdr = None;
                c.content_light = None;
                c.mastering_display = None;
                for i in 0..8 {
                    c.refs[i] = RefState::default();
                    c.fgm[i] = None;
                }
                c.ci = None;
            }
            c.seq_hdr = Some(Arc::new(seq_hdr));
        }

        Some(
            ObuType::OpenLoopKf
            | ObuType::ClosedLoopKf
            | ObuType::LeadingTileGrp
            | ObuType::TileGrp
            | ObuType::Switch
            | ObuType::LeadingSef
            | ObuType::Sef
            | ObuType::LeadingTip
            | ObuType::Tip
            | ObuType::Bridge
            | ObuType::Ras,
        ) => {
            let obu_type = obu_type.unwrap();
            let seqhdr = c
                .seq_hdr
                .as_ref()
                .ok_or(TealdustError::InvalidData)?
                .clone();

            let first_tile = matches!(obu_type, ObuType::Sef | ObuType::Tip | ObuType::Bridge)
                || gb.get_bit() != 0;
            let has_hdr = first_tile || gb.get_bit() != 0;

            if has_hdr {
                let mut hdr = parse_frame_hdr(&seqhdr, &c.refs, obu_type, &mut gb)?;
                hdr.tlayer_id = tlayer_id as u8;
                hdr.mlayer_id = mlayer_id as u8;
                hdr.xlayer_id = xlayer_id as u8;
                c.frame_hdr = Some(Arc::new(hdr));
            }

            c.tile.clear();
            c.n_tile_data = 0;
            c.n_tiles = 0;

            if matches!(obu_type, ObuType::Sef | ObuType::Tip | ObuType::Bridge) {
                check_trailing_bits(&mut gb, c.strict_std_compliance)?;
            }

            if let Some(ref fh) = c.frame_hdr {
                if c.frame_size_limit > 0
                    && (fh.width as u64) * (fh.height as u64) > c.frame_size_limit as u64
                {
                    c.frame_hdr = None;
                    return Err(TealdustError::FrameTooLarge);
                }
            }

            if matches!(obu_type, ObuType::Sef | ObuType::Tip | ObuType::Bridge) {
                // frame header only OBU, no tile data
            } else {
                let fh = c
                    .frame_hdr
                    .as_ref()
                    .ok_or(TealdustError::InvalidData)?
                    .clone();
                let mut tg = TileGroup {
                    data: Vec::new(),
                    start: 0,
                    end: 0,
                };
                parse_tile_hdr(&fh, &mut tg, &mut gb);
                gb.bytealign();
                if gb.has_error() {
                    return Err(TealdustError::InvalidData);
                }
                tg.data = gb.remaining_slice().to_vec();

                if tg.start > tg.end || tg.start != c.n_tiles {
                    c.tile.clear();
                    c.n_tile_data = 0;
                    c.n_tiles = 0;
                    return Err(TealdustError::InvalidData);
                }
                c.n_tiles += 1 + tg.end - tg.start;
                c.tile.push(tg);
                c.n_tile_data += 1;
            }
        }

        Some(ObuType::Fgm) => {
            let seqhdr = c
                .seq_hdr
                .as_ref()
                .ok_or(TealdustError::InvalidData)?
                .clone();
            let fgm = parse_fgm_hdr(&mut gb, seqhdr.layout)?;
            for (i, entry) in fgm.into_iter().enumerate() {
                if entry.is_some() {
                    c.fgm[i] = entry;
                }
            }
            check_trailing_bits(&mut gb, c.strict_std_compliance)?;
        }

        Some(ObuType::ContentInterp) => {
            let mut ci = ContentInterpretation::default();
            parse_ci_hdr(&mut ci, &mut gb)?;
            check_trailing_bits(&mut gb, c.strict_std_compliance)?;
            c.ci = Some(ci);
        }

        Some(ObuType::Metadata) => {
            let meta_type = gb.get_uleb128();
            if gb.has_error() {
                return Err(TealdustError::InvalidData);
            }
            match meta_type {
                v if v == ObuMetaType::HdrCll as u32 => {
                    let cll = parse_cll(&mut gb);
                    check_trailing_bits(&mut gb, c.strict_std_compliance)?;
                    c.content_light = Some(cll);
                }
                v if v == ObuMetaType::HdrMdcv as u32 => {
                    let md = parse_mdcv(&mut gb);
                    check_trailing_bits(&mut gb, c.strict_std_compliance)?;
                    c.mastering_display = Some(md);
                }
                _ => {} // ignore unknown metadata
            }
        }

        Some(ObuType::Td) => {
            // temporal delimiter — no action needed
        }

        Some(ObuType::Padding) => {
            // ignore
        }

        _ => {
            // unknown OBU type — ignore
        }
    }

    // post-processing: check if frame is ready to submit
    if c.seq_hdr.is_some() && c.frame_hdr.is_some() {
        let fh = c.frame_hdr.as_ref().unwrap().clone();
        if fh.show_existing_frame != 0 {
            let idx = fh.existing_frame_idx as usize;
            if c.refs[idx].p.frame_hdr.is_none() {
                return Err(TealdustError::InvalidData);
            }
            // (`c->refs[idx].p.p.data[0]`) to be present before it can be queued.
            let ref_pic = match c.refs[idx].p.pic.as_ref() {
                Some(p) if p.has_data() => p.clone(),
                _ => return Err(TealdustError::InvalidData),
            };
            if c.strict_std_compliance && !c.refs[idx].p.showable {
                return Err(TealdustError::InvalidData);
            }
            // referenced stored picture. With output_invisible_frames this is a
            // owned clone of the referenced reconstruction onto the output queue.
            c.frame_out
                .push(crate::decode::clone_picture_mt(&ref_pic, c.n_tc));
            // slot into every other ref slot and clears its showable flag.
            if c.refs[idx]
                .p
                .frame_hdr
                .as_ref()
                .is_some_and(|h| h.frame_type == FrameType::Key)
            {
                let r = idx;
                c.refs[r].p.showable = false;
                for i in 0..8 {
                    if i == r {
                        continue;
                    }
                    c.refs[i].p = c.refs[r].p.clone();
                    c.refs[i].cdf = c.refs[r].cdf.clone();
                    c.refs[i].segmap = c.refs[r].segmap.clone();
                    // motion field for the overwritten slot (no inc back).
                    c.refs[i].refmvs = None;
                }
            }
            c.frame_hdr = None;
        } else {
            let total_tiles = fh.tiling.t.cols as i32 * fh.tiling.t.rows as i32;
            let frame_without_data = fh.tip.frame_mode == 2;
            if c.n_tiles == total_tiles || frame_without_data {
                if !frame_without_data && c.n_tile_data == 0 {
                    return Err(TealdustError::InvalidData);
                }
                // Run the single-threaded frame decode (entropy pass; recon,
                // filters and output are wired in subsequent milestones). Gated
                // during bring-up; errors are non-fatal so header parsing still
                // succeeds.
                if c.run_decode {
                    // Best-effort during bring-up: a frame that cannot be decoded
                    // yet (e.g. inter frames) must not abort header parsing.
                    let _ = crate::decode::submit_frame(c, 1);
                }
                c.frame_hdr = None;
                c.n_tiles = 0;
            }
        }
    }

    Ok(total_consumed)
}
