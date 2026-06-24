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

use crate::internal::Pass;
use crate::intops::{iclip, imax, imin};
use crate::levels::{
    Av2Block, BlockSize, CFL_PRED, CompInterPredMode, InterPredMode, MotionMode, Mv, MvXY,
    N_BS_SIZES, RefPair, TIP_FRAME,
};
use crate::msac::MsacReader;

use crate::tables::BLOCK_DIMENSIONS;

pub(crate) fn decode_b<BD: BitDepth, const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    ctx: &mut SbCtx<'_, UPDATE_CDF, M>,
    recon: &mut ReconCtx<BD>,
    pass: u8,
    lbs: BlockSize,
    cbs: BlockSize,
) -> Result<Av2Block, ()>
where
    BD::Coef: DecodeCoeff,
{
    // Reborrow the bundled state into locals with the original names so the
    // body below is unchanged (decode_b is a leaf: it never re-passes `ctx`).
    let fi = ctx.fi;
    let bx = *ctx.bx;
    let by = *ctx.by;
    let cbx = *ctx.cbx;
    let cby = *ctx.cby;
    let intra_region = *ctx.intra_region;
    let _sdp_cfl_disallowed = *ctx.sdp_cfl_disallowed;
    let a = &mut *ctx.a;
    let l = &mut *ctx.l;
    let msac = &mut *ctx.msac;
    let cdf_m = &mut *ctx.cdf_m;
    let cdf_dmv = &mut *ctx.cdf_dmv;
    let _ = &mut *recon;
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    debug_assert!(bs != BlockSize::Invalid);
    let bs_idx = bs as u8 as usize;

    let b_dim = &BLOCK_DIMENSIONS[bs_idx];
    let bx4 = (bx & 63) as usize;
    let by4 = (by & 63) as usize;
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;

    let w4 = imin(bw4, fi.bw - bx);
    let h4 = imin(bh4, fi.bh - by);
    let have_left = bx > fi.tile_col_start;
    let have_top = by > fi.tile_row_start;
    let have_top_right = bx + bw4 <= fi.tile_col_end;
    let have_bottom_left = by + bh4 <= fi.tile_row_end;
    let has_luma = lbs != BlockSize::Invalid;
    let has_chroma = cbs != BlockSize::Invalid;

    let mut b = Av2Block::default();
    if has_luma {
        b.bs = lbs as i8;
    }
    if has_chroma {
        b.cbs = cbs as i8;
    }

    // Replay-only entry used by the dav2d-style split scaffold.  Partition
    // replay has already positioned bx/by/cbx/cby at this leaf; the entropy
    // pass stored the parsed Av2Block and coefficient payloads in order.
    if (pass & crate::internal::Pass::Entropy as u8) == 0 {
        let rec = recon
            .scratch
            .block_rec
            .get(recon.scratch.block_rpos)
            .copied()
            .ok_or(())?;
        recon.scratch.block_rpos += 1;
        debug_assert_eq!(rec.bx as i32, bx);
        debug_assert_eq!(rec.by as i32, by);
        debug_assert_eq!(rec.cbx as i32, cbx);
        debug_assert_eq!(rec.cby as i32, cby);
        debug_assert_eq!(rec.lbs, lbs as i8);
        debug_assert_eq!(rec.cbs, cbs as i8);
        let b = rec.b;

        // This first replay step is intentionally intra-only.  Inter needs the
        // separate MV-resolution replay path before it can be enabled safely.
        if b.is_intra == 0 || b.intrabc != 0 {
            return Err(());
        }

        if (pass & crate::internal::Pass::Recon as u8) != 0 {
            recon_b_intra_phase(
                &mut ReconBCtx {
                    recon: &mut *recon,
                    msac: &mut *msac,
                    cdf_m: &mut *cdf_m,
                    a: &mut *a,
                    l: &mut *l,
                    b: &b,
                    fi,
                },
                bx,
                by,
                cbx,
                cby,
                lbs,
                cbs,
                has_luma,
                has_chroma,
                TxPhase::ReconOnly,
                ChromaPhase::ReconOnly,
            )?;
        }
        return Ok(b);
    }

    // Pre-compute cross-SB boundary neighbor context values.
    // The C code uses nx[2] pointers into a/l; here we read out
    // the values we need before any mutable operations.
    let (
        nx_skip_mode,
        nx_skip_txfm,
        nx_intra,
        nx_intrabc,
        _nx_xoff,
        n_ctx,
        nx_ref0,
        nx_ref1,
        nx_amvd,
        nx_comp_type,
    ) = {
        let mut sm = [0u8; 2];
        let mut st = [0u8; 2];
        let mut intra_vals = [0u8; 2];
        let mut ibc_vals = [0u8; 2];
        let mut xoff = [0usize; 2];
        let mut r0 = [0i8; 2];
        let mut r1 = [0i8; 2];
        let mut amvd_v = [0u8; 2];
        let mut ct = [0u8; 2];
        let mut idx = 0usize;

        if have_left && by + bh4 <= fi.tile_row_end {
            let off = (by4 + bh4 as usize).saturating_sub(1);
            sm[0] = l.skip_mode[off];
            st[0] = l.skip_txfm[off];
            intra_vals[0] = if l.intra[off] != 0 && l.intrabc[off] == 0 {
                1
            } else {
                0
            };
            ibc_vals[0] = l.intrabc[off];
            r0[0] = l.r#ref[0][off];
            r1[0] = l.r#ref[1][off];
            amvd_v[0] = l.amvd[off];
            ct[0] = l.comp_type[off];
            xoff[0] = off;
            idx += 1;
        }
        if have_top && bx + bw4 <= fi.tile_col_end {
            let off = (bx4 + bw4 as usize).saturating_sub(1);
            sm[idx] = a.skip_mode[off];
            st[idx] = a.skip_txfm[off];
            intra_vals[idx] = if a.intra[off] != 0 && a.intrabc[off] == 0 {
                1
            } else {
                0
            };
            ibc_vals[idx] = a.intrabc[off];
            r0[idx] = a.r#ref[0][off];
            r1[idx] = a.r#ref[1][off];
            amvd_v[idx] = a.amvd[off];
            ct[idx] = a.comp_type[off];
            xoff[idx] = off;
            idx += 1;
        }
        if idx < 2 && have_left {
            sm[idx] = l.skip_mode[by4];
            st[idx] = l.skip_txfm[by4];
            intra_vals[idx] = if l.intra[by4] != 0 && l.intrabc[by4] == 0 {
                1
            } else {
                0
            };
            ibc_vals[idx] = l.intrabc[by4];
            r0[idx] = l.r#ref[0][by4];
            r1[idx] = l.r#ref[1][by4];
            amvd_v[idx] = l.amvd[by4];
            ct[idx] = l.comp_type[by4];
            xoff[idx] = by4;
            idx += 1;
        }
        if idx < 2 {
            sm[idx] = a.skip_mode[bx4];
            st[idx] = a.skip_txfm[bx4];
            intra_vals[idx] = if a.intra[bx4] != 0 && a.intrabc[bx4] == 0 {
                1
            } else {
                0
            };
            ibc_vals[idx] = a.intrabc[bx4];
            r0[idx] = a.r#ref[0][bx4];
            r1[idx] = a.r#ref[1][bx4];
            amvd_v[idx] = a.amvd[bx4];
            ct[idx] = a.comp_type[bx4];
            xoff[idx] = bx4;
            if idx == 0 {
                sm[1] = sm[0];
                st[1] = st[0];
                intra_vals[1] = intra_vals[0];
                ibc_vals[1] = ibc_vals[0];
                r0[1] = r0[0];
                r1[1] = r1[0];
                amvd_v[1] = amvd_v[0];
                ct[1] = ct[0];
                xoff[1] = xoff[0];
            }
            if have_top {
                idx += 1;
            }
        }
        (sm, st, intra_vals, ibc_vals, xoff, idx, r0, r1, amvd_v, ct)
    };

    let mut seg_pred = 0i32;
    if fi.seg_enabled {
        let bx_abs = bx;
        let by_abs = by;
        if !has_luma {
            b.seg_id =
                recon.cur_segmap[(bx_abs as isize + by_abs as isize * recon.b4_stride) as usize];
        } else if !fi.seg_update_map {
            if let Some(prev) = recon.prev_segmap {
                let sid = get_prev_frame_segid(by_abs, bx_abs, w4, h4, prev, recon.b4_stride);
                if sid >= 16 {
                    recon.seg_id_err = true;
                    return Err(());
                }
                b.seg_id = sid as u8;
            } else {
                b.seg_id = 0;
            }
        } else if fi.seg_preskip {
            seg_pred = if fi.seg_temporal {
                let ctx = a.seg_pred[bx4] as usize + l.seg_pred[by4] as usize;
                msac.decode_bool_adapt(cdf_m.seg_pred(ctx)) as i32
            } else {
                0
            };
            if seg_pred != 0 {
                if let Some(prev) = recon.prev_segmap {
                    let sid = get_prev_frame_segid(by_abs, bx_abs, w4, h4, prev, recon.b4_stride);
                    if sid >= 16 {
                        recon.seg_id_err = true;
                        return Err(());
                    }
                    b.seg_id = sid as u8;
                } else {
                    b.seg_id = 0;
                }
            } else {
                let mut seg_ctx = 0i32;
                let pred_seg_id = get_cur_frame_segid(
                    by_abs,
                    bx_abs,
                    have_top,
                    have_left,
                    &mut seg_ctx,
                    recon.cur_segmap,
                    recon.b4_stride,
                );
                let ext_flag = if fi.seg_ext {
                    msac.decode_bool_adapt(cdf_m.seg_id_ext(seg_ctx as usize)) as u32
                } else {
                    0
                };
                let diff = msac
                    .decode_symbol_adapt(cdf_m.seg_id(ext_flag as usize, seg_ctx as usize), 7)
                    + (ext_flag << 3);
                let last_active = fi.seg_last_active_segid as i32;
                let mut sid = neg_deinterleave(diff as i32, pred_seg_id as i32, last_active + 1);
                if sid > last_active {
                    sid = 0;
                }
                if sid >= crate::headers::MAX_SEGMENTS as i32 {
                    sid = 0;
                }
                b.seg_id = sid as u8;
            }
        }
    } else {
        b.seg_id = 0;
    }
    // For valid streams every segment id is in [0, MAX_SEGMENTS); the segment
    // map and predicted-id paths can leave an out-of-range value on a malformed
    // stream, which would overflow the `1 << seg_id` mask and index the
    // MAX_SEGMENTS-sized per-segment tables out of bounds. Clamp once here; this
    // is a no-op for valid input.
    if b.seg_id >= crate::headers::MAX_SEGMENTS as u8 {
        b.seg_id = 0;
    }

    // skip_mode
    if (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) == 0
        && fi.skip_mode_enabled
        && bw4 * bh4 > 2
        && intra_region == 0
    {
        let ctx = nx_skip_mode[0] as usize + nx_skip_mode[1] as usize;
        b.skip_mode = msac.decode_bool_adapt(cdf_m.skip_mode(ctx)) as u8;
    } else {
        b.skip_mode = 0;
    }

    // intra/inter decision
    if b.skip_mode != 0 {
        b.is_intra = 0;
    } else if fi.is_inter_or_switch && intra_region == 0 {
        if fi.has_chroma_layout && lbs != cbs {
            b.is_intra = 0;
        } else {
            // gathered neighbours (nx[0], nx[n_ctx-1]), plus 1 if all are intra.
            let ictx = if n_ctx == 0 {
                0
            } else {
                let i = (n_ctx - 1) as usize;
                let sum = nx_intra[0] as i32 + nx_intra[i] as i32;
                sum + (sum == n_ctx as i32) as i32
            };
            b.is_intra = (msac.decode_bool_adapt(cdf_m.intra(ictx as usize)) == 0) as u8;
        }
    } else {
        b.is_intra = 1;
    }

    // Pre-compute spatial neighbour (nb) context values within SB.
    // These are used by intrabc, FSC, MRL, multi_mrl, DIP, morph_pred.
    // boff[i] = -1 means unavailable.
    let have_top_in_sb = (by & (fi.sb_step - 1)) != 0;
    let (
        nb_fsc,
        nb_mrl,
        nb_multi_mrl,
        nb_intrabc,
        _nb_midx,
        nb_mvprec,
        nb_motion_mode,
        nb_morph,
        nb_dip,
        nb_boff,
        nb_ref0,
        nb_ref1,
        nb_filter,
    ) = if has_luma {
        let mut fsc = [0u8; 2];
        let mut mrl = [0u8; 2];
        let mut mmrl = [0u8; 2];
        let mut ibc = [0u8; 2];
        let mut mid = [0xffu8; 2];
        let mut mvp = [0u8; 2];
        let mut mm = [0u8; 2];
        let mut mp = [0u8; 2];
        let mut dp = [0u8; 2];
        let mut boff = [-1i32; 2];
        // the neighbour's ref pair and filter at boff, captured here so the a/l
        // identity is preserved (boff alone loses it).
        let mut nref0 = [-1i8; 2];
        let mut nref1 = [-1i8; 2];
        let mut nflt = [0u8; 2];
        let mut idx = 0usize;

        if have_left && bh4 == h4 {
            let off = (by4 + bh4 as usize).saturating_sub(1);
            fsc[0] = l.fsc[off];
            mrl[0] = l.mrl[off];
            mmrl[0] = l.multi_mrl[off];
            ibc[0] = l.intrabc[off];
            mid[0] = l.midx[off];
            mvp[0] = l.mvprec[off];
            mm[0] = l.motion_mode[off];
            mp[0] = l.morph_pred[off];
            dp[0] = l.dip[off];
            boff[0] = off as i32;
            nref0[0] = l.r#ref[0][off];
            nref1[0] = l.r#ref[1][off];
            nflt[0] = l.filter[off];
            idx += 1;
        }
        if have_top_in_sb && bw4 == w4 {
            let off = (bx4 + bw4 as usize).saturating_sub(1);
            fsc[idx] = a.fsc[off];
            mrl[idx] = a.mrl[off];
            mmrl[idx] = a.multi_mrl[off];
            ibc[idx] = a.intrabc[off];
            mid[idx] = a.midx[off];
            mvp[idx] = a.mvprec[off];
            mm[idx] = a.motion_mode[off];
            mp[idx] = a.morph_pred[off];
            dp[idx] = a.dip[off];
            boff[idx] = off as i32;
            nref0[idx] = a.r#ref[0][off];
            nref1[idx] = a.r#ref[1][off];
            nflt[idx] = a.filter[off];
            idx += 1;
        }
        if have_left && idx < 2 {
            fsc[idx] = l.fsc[by4];
            mrl[idx] = l.mrl[by4];
            mmrl[idx] = l.multi_mrl[by4];
            ibc[idx] = l.intrabc[by4];
            mid[idx] = l.midx[by4];
            mvp[idx] = l.mvprec[by4];
            mm[idx] = l.motion_mode[by4];
            mp[idx] = l.morph_pred[by4];
            dp[idx] = l.dip[by4];
            boff[idx] = by4 as i32;
            nref0[idx] = l.r#ref[0][by4];
            nref1[idx] = l.r#ref[1][by4];
            nflt[idx] = l.filter[by4];
            idx += 1;
        }
        if have_top_in_sb && idx < 2 {
            fsc[idx] = a.fsc[bx4];
            mrl[idx] = a.mrl[bx4];
            mmrl[idx] = a.multi_mrl[bx4];
            ibc[idx] = a.intrabc[bx4];
            mid[idx] = a.midx[bx4];
            mvp[idx] = a.mvprec[bx4];
            mm[idx] = a.motion_mode[bx4];
            mp[idx] = a.morph_pred[bx4];
            dp[idx] = a.dip[bx4];
            boff[idx] = bx4 as i32;
            nref0[idx] = a.r#ref[0][bx4];
            nref1[idx] = a.r#ref[1][bx4];
            nflt[idx] = a.filter[bx4];
            if idx == 0 {
                fsc[1] = fsc[0];
                mrl[1] = mrl[0];
                mmrl[1] = mmrl[0];
                ibc[1] = ibc[0];
                mid[1] = mid[0];
                mvp[1] = mvp[0];
                mm[1] = mm[0];
                mp[1] = mp[0];
                dp[1] = dp[0];
            }
        }
        (
            fsc, mrl, mmrl, ibc, mid, mvp, mm, mp, dp, boff, nref0, nref1, nflt,
        )
    } else {
        (
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [0xffu8; 2],
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [0u8; 2],
            [-1i32; 2],
            [-1i8; 2],
            [-1i8; 2],
            [0u8; 2],
        )
    };

    // intrabc
    if has_luma {
        b.intrabc = 0;
        if fi.allow_intrabc && imin(bw4, bh4) < 16 && b.is_intra != 0 && intra_region == 0 {
            let ctx = (nb_intrabc[0] + nb_intrabc[1]) as usize;

            b.intrabc = msac.decode_bool_adapt(cdf_m.intrabc(ctx)) as u8;
        }
    }
    let intrabc = has_luma && b.intrabc != 0;

    // skip_txfm
    if fi.seg_skip_mask & (1 << b.seg_id) != 0 {
        b.skip_txfm = 1;
    } else if b.is_intra != 0 && !intrabc {
        if has_luma {
            b.skip_txfm = 0;
        }
    } else {
        let ctx = nx_skip_txfm[0] as usize + nx_skip_txfm[1] as usize + b.skip_mode as usize * 3;
        b.skip_txfm = msac.decode_bool_adapt(cdf_m.skip_txfm(ctx)) as u8;
    }

    if fi.seg_enabled && fi.seg_update_map && !fi.seg_preskip {
        let bx_abs = bx;
        let by_abs = by;
        if !has_luma {
            b.seg_id =
                recon.cur_segmap[(bx_abs as isize + by_abs as isize * recon.b4_stride) as usize];
        } else if b.skip_txfm == 0 && fi.seg_temporal && {
            let ctx = a.seg_pred[bx4] as usize + l.seg_pred[by4] as usize;
            seg_pred = msac.decode_bool_adapt(cdf_m.seg_pred(ctx)) as i32;
            seg_pred != 0
        } {
            if let Some(prev) = recon.prev_segmap {
                let sid = get_prev_frame_segid(by_abs, bx_abs, w4, h4, prev, recon.b4_stride);
                if sid >= 16 {
                    recon.seg_id_err = true;
                    return Err(());
                }
                b.seg_id = sid as u8;
            } else {
                b.seg_id = 0;
            }
        } else {
            let mut seg_ctx = 0i32;
            let pred_seg_id = get_cur_frame_segid(
                by_abs,
                bx_abs,
                have_top,
                have_left,
                &mut seg_ctx,
                recon.cur_segmap,
                recon.b4_stride,
            );
            if b.skip_txfm != 0 && !fi.any_lossless {
                b.seg_id = pred_seg_id as u8;
            } else {
                let ext_flag = if fi.seg_ext {
                    msac.decode_bool_adapt(cdf_m.seg_id_ext(seg_ctx as usize)) as u32
                } else {
                    0
                };
                let diff = msac
                    .decode_symbol_adapt(cdf_m.seg_id(ext_flag as usize, seg_ctx as usize), 7)
                    + (ext_flag << 3);
                let last_active = fi.seg_last_active_segid as i32;
                let mut sid = neg_deinterleave(diff as i32, pred_seg_id as i32, last_active + 1);
                if sid > last_active {
                    sid = 0;
                }
                b.seg_id = sid as u8;
            }
            if b.seg_id >= crate::headers::MAX_SEGMENTS as u8 {
                b.seg_id = 0;
            }
        }
        // Same defensive clamp as the pre-skip path: keep seg_id in range for the
        // segment-map / predicted-id branches above. No-op for valid input.
        if b.seg_id >= crate::headers::MAX_SEGMENTS as u8 {
            b.seg_id = 0;
        }
    }

    let skip_txfm = has_luma && b.skip_txfm != 0;

    if has_luma {
        let gdf_sz_log2 = if fi.gdf_is_key { 1 } else { imax(1, fi.sb128) };
        let gdf_bs = 16 << gdf_sz_log2;
        if (bx | by) & (gdf_bs - 1) == 0 {
            let idx = (((by & 48) >> 2) + ((bx & 48) >> 4)) as usize;
            let flag = if fi.gdf_enabled == crate::headers::AdaptiveBoolean::Adaptive
                && imax(fi.cur_w, fi.cur_h) > 4 * gdf_bs
            {
                msac.decode_bool_adapt(cdf_m.gdf()) as u8
            } else {
                (fi.gdf_enabled != crate::headers::AdaptiveBoolean::Off) as u8
            };
            let n = 1usize << gdf_sz_log2;
            let m = &mut recon.lf_mask[recon.lf_idx];
            m.gdf[idx..idx + n].fill(flag);
            if gdf_bs >= 32 {
                m.gdf[idx + 4..idx + 4 + n].fill(flag);
                if gdf_bs == 64 {
                    m.gdf[idx + 8..idx + 8 + n].fill(flag);
                    m.gdf[idx + 12..idx + 12 + n].fill(flag);
                }
            }
        }
    }

    if fi.cdef_enabled && (!skip_txfm || fi.cdef_on_skiptx) {
        let idx = (((bx & 0x30) >> 4) + ((by & 0x30) >> 2)) as usize;
        if recon.lf_mask[recon.lf_idx].cdef_idx[idx] == -1 {
            let v;
            if fi.cdef_n_strengths == 1 {
                v = 0i8;
            } else {
                let left_cdef_idx = if bx - 16 < fi.tile_col_start {
                    -1i32
                } else if idx & 3 != 0 {
                    recon.lf_mask[recon.lf_idx].cdef_idx[idx - 1] as i32
                } else {
                    recon.lf_mask[recon.lf_idx - 1].cdef_idx[idx + 3] as i32
                };
                let top_cdef_idx = if (by & !15) & (fi.sb_step - 1) == 0 {
                    -1i32
                } else if idx & 0xc != 0 {
                    recon.lf_mask[recon.lf_idx].cdef_idx[idx - 4] as i32
                } else {
                    recon.lf_mask[recon.lf_idx - recon.sb256w as usize].cdef_idx[idx + 12] as i32
                };
                let ctx = if (left_cdef_idx | top_cdef_idx) != -1 {
                    // both edges available
                    let mut c = (left_cdef_idx == 0) as i32 + (top_cdef_idx == 0) as i32;
                    c += (c == 2) as i32;
                    c
                } else {
                    // C: !(left & top) * 2  (logical-not, so 0 -> 1, nonzero -> 0)
                    ((left_cdef_idx & top_cdef_idx) == 0) as i32 * 2
                };
                if msac.decode_bool_adapt(cdf_m.cdef_idx0(ctx as usize)) != 0 {
                    v = 0;
                } else if fi.cdef_n_strengths == 2 {
                    v = 1;
                } else {
                    let rem = fi.cdef_n_strengths as i32 - 3;
                    v = 1 + msac
                        .decode_symbol_adapt(cdf_m.cdef_idx(rem as usize), (rem + 1) as usize)
                        as i8;
                }
            }
            let splat_n = 1usize << imax(0, b_dim[2] as i32 - 4);
            let m = &mut recon.lf_mask[recon.lf_idx];
            m.cdef_idx[idx..idx + splat_n].fill(v);
            if bh4 >= 32 {
                m.cdef_idx[idx + 4..idx + 4 + splat_n].fill(v);
                if bh4 == 64 {
                    m.cdef_idx[idx + 8..idx + 8 + splat_n].fill(v);
                    m.cdef_idx[idx + 12..idx + 12 + splat_n].fill(v);
                }
            }
        }
    }

    if has_luma && (bx | by) & (63 >> (2 - fi.sb128)) == 0 {
        let unit_mi = (63 >> (2 - fi.sb128)) + 1; // ccso unit = SB size in mi (32 for sb128)
        let upr = (64 / unit_mi) as usize; // ccso units per row within a 256px lf region
        let sub_x = ((bx & 63) / unit_mi) as usize;
        let sub_y = ((by & 63) / unit_mi) as usize;
        let sub = sub_y * upr + sub_x;
        let ccso_idx = (3 * ((bx >> 6) + (by >> 6) * fi.sb256w)) as usize;
        for p in 0..3 {
            if !fi.ccso_enabled[p] {
                continue;
            }
            let val = if fi.ccso_sb_reuse[p] {
                match recon.prev_ccsomap[p] {
                    Some(prev) => prev[ccso_idx + p],
                    None => 0,
                }
            } else {
                // ccso is read at the SB's top-left block, which is always at the
                // SB top boundary; fetch_spatial_neighbors excludes the above
                // neighbours there, so the ctx uses the left SB only.
                let left = if bx - unit_mi >= fi.tile_col_start {
                    Some(if sub_x > 0 {
                        recon.lf_mask[recon.lf_idx].ccso_sb[(sub - 1) * 3 + p]
                    } else {
                        recon.lf_mask[recon.lf_idx - 1].ccso_sb[(sub + upr - 1) * 3 + p]
                    })
                } else {
                    None
                };
                let ctx = match left {
                    Some(l) => (l != 0) as usize * 2,
                    None => 0,
                };
                msac.decode_bool_adapt(cdf_m.ccso(p, ctx)) as u8
            };
            recon.lf_mask[recon.lf_idx].ccso[p] = val;
            recon.lf_mask[recon.lf_idx].ccso_sb[sub * 3 + p] = val;
            if !recon.cur_ccsomap.is_empty() {
                recon.cur_ccsomap[ccso_idx + p] = val;
            }
        }
    }

    if has_luma && (bx | by) & (63 >> (2 - fi.sb128)) == 0 {
        let prev_qidx = recon.last_qidx;
        let have_delta_q = fi.delta_q_present && (bs != fi.root_bs || b.skip_txfm == 0);
        if have_delta_q {
            let mut delta_q = msac.decode_symbol_adapt(cdf_m.delta_q(), 7) as i32;
            if delta_q == 7 {
                let n_bits = 1 + msac.decode_bools_bypass(3) as i32;
                delta_q = msac.decode_bools_bypass(n_bits as u32) as i32 + 1 + (1 << n_bits);
            }
            if delta_q != 0 {
                if msac.decode_bool_bypass() != 0 {
                    delta_q = -delta_q;
                }
                delta_q *= 1 << fi.delta_q_res_log2;
            }
            let qmax = 255 + (recon.frame.bitdepth as i32 - 8) * 24;
            recon.last_qidx = iclip(recon.last_qidx + delta_q, 1, qmax);
        }
        let new_qidx = recon.last_qidx;
        if new_qidx == fi.quant_yac {
            recon.dq_active = *recon.frame.dq;
        } else if new_qidx != prev_qidx {
            let qmax = 255 + (recon.frame.bitdepth as i32 - 8) * 24;
            init_quant_tables_fi(fi, new_qidx, &mut recon.dqmem, qmax);
            recon.dq_active = recon.dqmem;
        }

        // 1959) so deblock can recompute its per-64px thresholds per block.
        let bx4 = (bx & 63) as usize;
        let by4 = (by & 63) as usize;
        let qbase = (bx4 >> 4) + ((by4 & 0x30) >> 2);
        let sbsz64 = 1usize << fi.sb128;
        let m = &mut recon.lf_mask[recon.lf_idx];
        let mut qoff = qbase;
        for _ in 0..sbsz64 {
            for x64 in 0..sbsz64 {
                m.qidx[qoff + x64] = new_qidx as u16;
            }
            qoff += 4;
        }
    }

    // Intra mode decoding
    static REORDERED_NONDIR_Y_MODE: [u8; 5] = [0, 9, 10, 11, 12];
    static REORDERED_DIR_Y_MODE: [u8; 8] = [3, 8, 1, 5, 4, 6, 2, 7];

    let mut luma_midx = 0xffu8;
    if b.is_intra != 0 && !intrabc && has_luma {
        static DEFAULT_MODE_LIST_Y: [u8; 56] = [
            17, 45, 3, 10, 24, 31, 38, 52, 15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50,
            54, 16, 18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0, 6,
            7, 13, 21, 27, 28, 34, 35, 41, 49, 55,
        ];

        // DPCM (lossless mode) — gated on THIS segment's lossless flag
        let seg_lossless = fi.seg_lossless[b.seg_id as usize] != 0;
        let dpcm = seg_lossless && msac.decode_bool_adapt(cdf_m.dpcm(0)) != 0;
        let (y_mode, y_angle, midx);

        if dpcm {
            if msac.decode_bool_adapt(cdf_m.dpcm_dir(0)) != 0 {
                y_mode = 2; // HOR_PRED
                midx = 45u8;
            } else {
                y_mode = 1; // VERT_PRED
                midx = 17u8;
            }
            y_angle = 0i8;
            b.intra_data_mut().mrl_index = 0;
            b.intra_data_mut().multi_mrl = 0;
        } else {
            let y_set = msac.decode_symbol_adapt(cdf_m.intra_y_set(), 3) as usize;
            let y_mode_idx;

            if y_set == 0 {
                let y_mode_ctx =
                    (w4 == bw4 && a.midx[(bx4 + bw4 as usize).saturating_sub(1)] != 0xff) as usize
                        + (h4 == bh4 && l.midx[(by4 + bh4 as usize).saturating_sub(1)] != 0xff)
                            as usize;
                let mut idx0 = msac.decode_symbol_adapt(cdf_m.intra_y_idx0(y_mode_ctx), 7) as usize;
                if idx0 == 7 {
                    idx0 += msac.decode_symbol_adapt(cdf_m.intra_y_idx1(y_mode_ctx), 5) as usize;
                }
                y_mode_idx = idx0;
            } else {
                y_mode_idx = y_set * 16 - 3 + msac.decode_bools_bypass(4) as usize;
            }

            if y_mode_idx < 5 {
                y_mode = REORDERED_NONDIR_Y_MODE[y_mode_idx];
                y_angle = 0;
                midx = 0xff;
            } else {
                let dir_idx = y_mode_idx - 5;

                // Build custom mode list from neighbour directional modes
                let mut custom_list = [0u8; 56];
                let mut use_custom = false;
                let mut _list_len = 0usize;

                if bw4 * bh4 > 2 {
                    let mut mask = 0u64;
                    let mut ptr = 0usize;

                    if h4 == bh4 {
                        let lmidx = l.midx[(by4 + bh4 as usize).saturating_sub(1)];
                        if lmidx != 0xff {
                            custom_list[ptr] = lmidx;
                            mask |= 1u64 << lmidx;
                            ptr += 1;
                        }
                    }
                    if w4 == bw4 {
                        let amidx = a.midx[(bx4 + bw4 as usize).saturating_sub(1)];
                        if amidx != 0xff && (ptr == 0 || amidx != custom_list[0]) {
                            custom_list[ptr] = amidx;
                            mask |= 1u64 << amidx;
                            ptr += 1;
                        }
                    }
                    let n_dirs = ptr;
                    if n_dirs > 0 {
                        use_custom = true;
                        if bw4 * bh4 > 4 && dir_idx >= n_dirs {
                            for i in 1..5i32 {
                                for n in 0..n_dirs {
                                    let cmidx = custom_list[n] as i32;
                                    for delta in [-i, i] {
                                        let dmidx = ((cmidx + delta + 56) % 56) as u8;
                                        if mask & (1u64 << dmidx) == 0 {
                                            custom_list[ptr] = dmidx;
                                            mask |= 1u64 << dmidx;
                                            ptr += 1;
                                        }
                                    }
                                }
                            }
                        }
                        if dir_idx >= ptr {
                            for &fmidx in DEFAULT_MODE_LIST_Y.iter() {
                                let bit = 1u64 << fmidx;
                                if mask & bit == 0 {
                                    custom_list[ptr] = fmidx;
                                    ptr += 1;
                                }
                            }
                        }
                        _list_len = ptr;
                    }
                }

                let dir_y_mode_reord = if use_custom {
                    custom_list[dir_idx]
                } else {
                    DEFAULT_MODE_LIST_Y[dir_idx]
                };
                midx = dir_y_mode_reord;
                y_mode = REORDERED_DIR_Y_MODE[(dir_y_mode_reord / 7) as usize];
                y_angle = (dir_y_mode_reord % 7) as i8 - 3;
            }
        }

        {
            let intra = b.intra_data_mut();
            intra.dpcm[0] = dpcm as u8;
            intra.y_mode = y_mode;
            intra.y_angle = y_angle;
        }

        // FSC (Frequency Segmented Coding)
        if imax(bw4, bh4) <= 8 && fi.idtx_intra {
            #[rustfmt::skip]
            static FSC_BSIZE_GROUPS: [u8; N_BS_SIZES] = {
                let mut t = [0u8; N_BS_SIZES];
                t[BlockSize::Bs32x32 as u8 as usize] = 5;
                t[BlockSize::Bs32x16 as u8 as usize] = 5;
                t[BlockSize::Bs32x8 as u8 as usize] = 4;
                t[BlockSize::Bs32x4 as u8 as usize] = 4;
                t[BlockSize::Bs16x32 as u8 as usize] = 5;
                t[BlockSize::Bs16x16 as u8 as usize] = 4;
                t[BlockSize::Bs16x8 as u8 as usize] = 3;
                t[BlockSize::Bs16x4 as u8 as usize] = 3;
                t[BlockSize::Bs8x32 as u8 as usize] = 4;
                t[BlockSize::Bs8x16 as u8 as usize] = 3;
                t[BlockSize::Bs8x8 as u8 as usize] = 2;
                t[BlockSize::Bs8x4 as u8 as usize] = 1;
                t[BlockSize::Bs4x32 as u8 as usize] = 4;
                t[BlockSize::Bs4x16 as u8 as usize] = 3;
                t[BlockSize::Bs4x8 as u8 as usize] = 1;
                t
            };
            let sz_ctx = FSC_BSIZE_GROUPS[bs as u8 as usize] as usize;
            let fsc_ctx = if fi.is_inter_or_switch && intra_region == 0 {
                3usize
            } else {
                (nb_fsc[0] + nb_fsc[1]) as usize
            };
            b.fsc = msac.decode_bool_adapt(cdf_m.fsc(fsc_ctx, sz_ctx)) as u8;
        }

        // MRL (Multi-Reference Line) index
        b.intra_data_mut().mrl_index = 0;
        b.intra_data_mut().multi_mrl = 0;
        if !dpcm && midx != 0xff && fi.mrls {
            let mrl_ctx = (nb_mrl[0] + nb_mrl[1]) as usize;
            let mrl_idx = msac.decode_symbol_adapt(cdf_m.mrl_index(mrl_ctx), 3) as u8;
            b.intra_data_mut().mrl_index = mrl_idx;
            if mrl_idx > 0 {
                let mmrl_ctx = (nb_multi_mrl[0] + nb_multi_mrl[1]) as usize;
                let mmrl = msac.decode_bool_adapt(cdf_m.multi_mrl(mmrl_ctx)) as u8;
                b.intra_data_mut().multi_mrl = mmrl;
            }
        }

        luma_midx = midx;
    }

    // UV chroma mode decoding
    if b.is_intra != 0 && !intrabc && has_chroma {
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbx4 = (cbx & 63) as usize;
        let cby4 = (cby & 63) as usize;
        // Chroma block dims used by the cfl/mhccp allow conditions are
        let cbw4 = (cb_dim[0] as i32) >> fi.ss_hor;
        let cbh4 = (cb_dim[1] as i32) >> fi.ss_ver;

        // For the chroma-only SDP tree (no luma), read the luma block's intra
        let midx = if !has_luma {
            recon.scratch.luma_intra_dir_mode_map[((cby & 15) * 16 + (cbx & 15)) as usize]
        } else {
            luma_midx
        };

        // DPCM for chroma — gated on THIS segment's lossless flag
        let seg_lossless_c = fi.seg_lossless[b.seg_id as usize] != 0;
        b.intra_data_mut().dpcm[1] =
            (seg_lossless_c && msac.decode_bool_adapt(cdf_m.dpcm(1)) != 0) as u8;
        let chroma_dpcm = b.intra_data().dpcm[1] != 0;

        if chroma_dpcm {
            let uv_mode = if msac.decode_bool_adapt(cdf_m.dpcm_dir(1)) != 0 {
                2u8 // HOR_PRED
            } else {
                1u8 // VERT_PRED
            };
            let uv_mode_idx: i32 = if uv_mode == 2 { 45 } else { 17 };
            let uv_angle = if (midx as i32 - uv_mode_idx).unsigned_abs() >= 4 {
                0i8
            } else {
                (midx % 7) as i8 - 3
            };
            {
                let intra = b.intra_data_mut();
                intra.uv_mode = uv_mode;
                intra.uv_angle = uv_angle;
            }
        } else {
            let ll = fi.seg_lossless[b.seg_id as usize] != 0;
            let mhccp_allowed = fi.mhccp
                && imax(cbw4, cbh4) <= if ll { 1 } else { 8 }
                && cbw4 * cbh4 >= if ll { 1 } else { 2 };
            let cfl_allowed = (fi.cfl || mhccp_allowed)
                && (imax(bw4, bh4) > 16 || _sdp_cfl_disallowed == 0)
                && imax(cbw4, cbh4) <= if ll { 1 } else { 16 };

            let cfl_ctx = if cfl_allowed {
                (a.uvmode[cbx4] == CFL_PRED) as usize + (l.uvmode[cby4] == CFL_PRED) as usize
            } else {
                0
            };
            let is_cfl = cfl_allowed && msac.decode_bool_adapt(cdf_m.cfl(cfl_ctx)) != 0;

            if is_cfl {
                {
                    let intra = b.intra_data_mut();
                    intra.uv_mode = CFL_PRED;
                    intra.uv_angle = 0;
                    intra.cfl.set_alpha([0; 2]);
                }
                // CFL parameters
                const CFL_EXPLICIT: i8 = 0;
                const CFL_MHCCP: i8 = 2;
                if mhccp_allowed && (!fi.cfl || msac.decode_bool_adapt(cdf_m.mhccp()) != 0) {
                    let sz_ctx = SIZE_GROUP[bs as u8 as usize] as usize;
                    {
                        let intra = b.intra_data_mut();
                        intra.cfl_type = CFL_MHCCP;
                        intra.cfl.set_mh_dir(
                            msac.decode_symbol_adapt(cdf_m.mhccp_filter_dir(sz_ctx), 2) as u8,
                        );
                    }
                } else {
                    let cfl_type = msac.decode_bool_adapt(cdf_m.cfl_type()) as i8;
                    b.intra_data_mut().cfl_type = cfl_type;
                    if cfl_type == CFL_EXPLICIT {
                        let sign = msac.decode_symbol_adapt(cdf_m.cfl_sign(), 7) as i32 + 1;
                        let sign_u = (sign * 0x56) >> 8;
                        let sign_v = sign - sign_u * 3;
                        if sign_u != 0 {
                            let ctx = (sign_u == 2) as usize * 3 + sign_v as usize;
                            let mut alpha =
                                msac.decode_symbol_adapt(cdf_m.cfl_alpha(ctx), 7) as i8 + 1;
                            if sign_u == 1 {
                                alpha = -alpha;
                            }
                            b.intra_data_mut().cfl.set_alpha_at(0, alpha);
                        }
                        if sign_v != 0 {
                            let ctx = (sign_v == 2) as usize * 3 + sign_u as usize;
                            let mut alpha =
                                msac.decode_symbol_adapt(cdf_m.cfl_alpha(ctx), 7) as i8 + 1;
                            if sign_v == 1 {
                                alpha = -alpha;
                            }
                            b.intra_data_mut().cfl.set_alpha_at(1, alpha);
                        }
                    }
                }
            } else {
                // AV2 UV mode context: 0 = non-directional luma, 1 = directional luma.
                // With ctx=1 the first index slot encodes "same as luma"; with ctx=0
                // all 5 non-dir + 9 dir = 14 modes are directly indexed.
                // `decode_symbol_adapt(7)` returns 0..7; if 7, `bools_bypass(3)` adds
                // 0..7 giving total range 0..14.
                let uv_mode_ctx = (midx != 0xff) as usize;
                let mut uv_mode_idx =
                    msac.decode_symbol_adapt(cdf_m.intra_uv_mode(uv_mode_ctx), 7) as usize;
                if uv_mode_idx == 7 {
                    uv_mode_idx += msac.decode_bools_bypass(3) as usize;
                }
                // AV2 UV directional modes (10 entries): the 8 AV1 directional modes
                // plus PAETH (12) and SMOOTH (9). idx = uv_mode_idx - 5 - uv_mode_ctx.
                static DEFAULT_MODE_LIST_UV_AV2: [u8; 10] = [1, 2, 3, 4, 8, 5, 6, 7, 12, 9];
                static INTRA_DIR_MODE_Y_TO_UV_IDX: [u8; 8] = [2, 4, 0, 5, 3, 6, 1, 7];

                // Maximum valid uv_mode_idx: ctx=0 → 14 modes (0..13 non-dir+dir),
                // ctx=1 → same+5+9=15 modes (0..14).
                // Valid uv_mode_idx layout: [same-as-luma: uv_mode_ctx slots]
                // + [5 non-directional] + [10 directional] = 15 + uv_mode_ctx
                // values, so the index range is 0..=14+uv_mode_ctx. The escape
                // path (symbol 7 + bools_bypass(3)) can legitimately produce 14.
                let max_uv = 5 + DEFAULT_MODE_LIST_UV_AV2.len() + uv_mode_ctx;
                if uv_mode_idx >= max_uv {
                    return Err(());
                }

                if uv_mode_idx < uv_mode_ctx {
                    // Same directional mode as luma
                    {
                        let intra = b.intra_data_mut();
                        intra.uv_mode = REORDERED_DIR_Y_MODE[(midx / 7) as usize];
                        intra.uv_angle = (midx % 7) as i8 - 3;
                    }
                } else if uv_mode_idx - uv_mode_ctx < 5 {
                    // Non-directional mode
                    {
                        let intra = b.intra_data_mut();
                        intra.uv_mode = REORDERED_NONDIR_Y_MODE[uv_mode_idx - uv_mode_ctx];
                        intra.uv_angle = 0;
                    }
                } else {
                    // Directional mode from default UV list.
                    let mut idx = (uv_mode_idx - 5 - uv_mode_ctx) as i32;
                    if uv_mode_ctx != 0 {
                        idx +=
                            (idx >= INTRA_DIR_MODE_Y_TO_UV_IDX[(midx / 7) as usize] as i32) as i32;
                    }
                    {
                        let intra = b.intra_data_mut();
                        intra.uv_mode = DEFAULT_MODE_LIST_UV_AV2[idx as usize];
                        intra.uv_angle = 0;
                    }
                }
            }
        }
    }

    // Palette and DIP (has_luma intra path)
    if b.is_intra != 0 && !intrabc && has_luma {
        let y_mode = b.intra_data().y_mode;
        b.intra_data_mut().pal_sz = 0;

        if fi.allow_screen_content_tools
            && y_mode == 0 // DC_PRED
            && imax(bw4, bh4) <= 16
            && bw4 + bh4 >= 4
        {
            let use_y_pal = msac.decode_bool_adapt(cdf_m.pal_y()) != 0;
            if use_y_pal {
                // Above palette is only reused inside SB64 boundaries (`by4 & 15`).
                let a_cache = if by4 & 15 != 0 {
                    a.pal_sz[bx4] as i32
                } else {
                    0
                };
                let l_cache = l.pal_sz[by4] as i32;
                let a_pal = recon.scratch.al_pal[0][bx4];
                let l_pal = recon.scratch.al_pal[1][by4];
                let mut pal = [0u16; 8];
                let pal_sz = read_pal_plane(
                    msac,
                    cdf_m,
                    &mut pal,
                    &a_pal,
                    &l_pal,
                    a_cache,
                    l_cache,
                    recon.frame.bitdepth,
                );
                recon.scratch.pal = pal;
                b.intra_data_mut().pal_sz = pal_sz;
            }
        }

        // DIP (Directional Intra Prediction enhancement)
        b.intra_data_mut().dip = 0;
        let pal_sz = b.intra_data().pal_sz;
        if y_mode == 0 // DC_PRED
            && fi.intra_dip
            && pal_sz == 0
            && imin(bw4, bh4) >= 2
            && bw4 * bh4 >= 8
        {
            let nb_dip_0 = if nb_boff[0] != -1 { nb_dip[0] } else { 0 };
            let nb_dip_1 = if nb_boff[1] != -1 { nb_dip[1] } else { 0 };
            let ctx = nb_dip_0 as usize + nb_dip_1 as usize;
            let dip_flag = msac.decode_bool_adapt(cdf_m.dip(ctx)) != 0;
            if dip_flag {
                let tp = msac.decode_bools_bypass(1) as u8;
                let m = msac.decode_symbol_adapt(cdf_m.dip_mode(), 5) as u8;
                b.intra_data_mut().dip = (tp << 4) | (m + 1);
            }
        }
    }

    if b.is_intra != 0 && !intrabc && has_luma {
        let pal_sz = b.intra_data().pal_sz as i32;
        if pal_sz != 0 {
            let sz = [w4 * 4, h4 * 4, bw4 * 4, bh4 * 4];
            // pal_idx_finish needs distinct dst (packed) / src (unpacked) buffers;
            let mut idx_scratch = vec![0u8; (bw4 * 4 * bh4 * 4) as usize];
            if read_pal_indices(
                msac,
                cdf_m,
                &mut recon.scratch.pal_idx_y[..],
                &mut idx_scratch[..],
                pal_sz,
                &sz,
            ) < 0
            {
                return Err(());
            }
        }
    }

    // TX partition (intra path)
    if b.is_intra != 0 && !intrabc && has_luma {
        let __seg_ll = fi.seg_lossless[b.seg_id as usize] != 0;
        read_tx_part(msac, cdf_m, &mut b, bs, __seg_ll, fi.txfm_switchable);
    }

    // is_sm flags for reconstruction (smooth mode neighbours)
    if b.is_intra != 0 && !intrabc {
        if has_luma {
            let sm = |mode: u8| -> i32 { (mode == 9 || mode == 10 || mode == 11) as i32 };
            let a_mode = a.mode[bx4];
            let l_mode = l.mode[by4];
            {
                let intra = b.intra_data_mut();
                intra.is_sm[0].a = if a.intra[bx4] != 0 { sm(a_mode) } else { 0 };
                intra.is_sm[0].l = if l.intra[by4] != 0 { sm(l_mode) } else { 0 };
            }
        }
        if has_chroma {
            let sm = |mode: u8| -> i32 { (mode == 9 || mode == 10 || mode == 11) as i32 };
            let cbx4 = (cbx & 63) as usize;
            let cby4 = (cby & 63) as usize;
            {
                let intra = b.intra_data_mut();
                intra.is_sm[1].a = sm(a.uvmode[cbx4]);
                intra.is_sm[1].l = sm(l.uvmode[cby4]);
            }
        }
    }

    // Intra context update
    if b.is_intra != 0 && !intrabc && has_luma {
        let y_mode = b.intra_data().y_mode;
        let mrl_idx = b.intra_data().mrl_index;
        let multi_mrl = b.intra_data().multi_mrl;
        let dip_val = b.intra_data().dip;
        let pal_sz_val = b.intra_data().pal_sz;

        let aw = 1usize << b_dim[2];
        let lh = 1usize << b_dim[3];

        // Above context (a)
        a.fsc[bx4..bx4 + aw].fill(b.fsc);
        a.mode[bx4..bx4 + aw].fill(y_mode);
        a.midx[bx4..bx4 + aw].fill(luma_midx);
        a.mrl[bx4..bx4 + aw].fill((mrl_idx != 0) as u8);
        a.multi_mrl[bx4..bx4 + aw].fill(multi_mrl);
        a.dip[bx4..bx4 + aw].fill((dip_val != 0) as u8);
        a.pal_sz[bx4..bx4 + aw].fill(pal_sz_val);
        a.seg_pred[bx4..bx4 + aw].fill(seg_pred as u8);
        a.skip_mode[bx4..bx4 + aw].fill(0);
        a.intra[bx4..bx4 + aw].fill(1);
        a.intrabc[bx4..bx4 + aw].fill(0);
        a.morph_pred[bx4..bx4 + aw].fill(0);
        a.skip_txfm[bx4..bx4 + aw].fill(b.skip_txfm);
        if fi.is_inter_or_switch {
            a.amvd[bx4..bx4 + aw].fill(0);
            a.mvprec[bx4..bx4 + aw].fill(0);
            a.motion_mode[bx4..bx4 + aw].fill(0);
            a.comp_type[bx4..bx4 + aw].fill(0);
            a.r#ref[0][bx4..bx4 + aw].fill(-1);
            a.r#ref[1][bx4..bx4 + aw].fill(-1);
        }

        // Left context (l)
        l.fsc[by4..by4 + lh].fill(b.fsc);
        l.mode[by4..by4 + lh].fill(y_mode);
        l.midx[by4..by4 + lh].fill(luma_midx);
        l.mrl[by4..by4 + lh].fill((mrl_idx != 0) as u8);
        l.multi_mrl[by4..by4 + lh].fill(multi_mrl);
        l.dip[by4..by4 + lh].fill((dip_val != 0) as u8);
        l.pal_sz[by4..by4 + lh].fill(pal_sz_val);
        l.seg_pred[by4..by4 + lh].fill(seg_pred as u8);
        l.skip_mode[by4..by4 + lh].fill(0);
        l.intra[by4..by4 + lh].fill(1);
        l.intrabc[by4..by4 + lh].fill(0);
        l.morph_pred[by4..by4 + lh].fill(0);
        l.skip_txfm[by4..by4 + lh].fill(b.skip_txfm);
        if fi.is_inter_or_switch {
            l.amvd[by4..by4 + lh].fill(0);
            l.mvprec[by4..by4 + lh].fill(0);
            l.motion_mode[by4..by4 + lh].fill(0);
            l.comp_type[by4..by4 + lh].fill(0);
            l.r#ref[0][by4..by4 + lh].fill(-1);
            l.r#ref[1][by4..by4 + lh].fill(-1);
        }
    }

    // Chroma context update (uvmode)
    if b.is_intra != 0 && !intrabc && has_chroma {
        let uv_mode = b.intra_data().uv_mode;
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbx4 = (cbx & 63) as usize;
        let cby4 = (cby & 63) as usize;
        let cbw4 = 1usize << cb_dim[2];
        let cbh4 = 1usize << cb_dim[3];
        a.uvmode[cbx4..cbx4 + cbw4].fill(uv_mode);
        l.uvmode[cby4..cby4 + cbh4].fill(uv_mode);
    }

    // IntraBC path
    if intrabc {
        b.intra_data_mut().is_refmv = msac.decode_bool_adapt(cdf_m.intrabc_mode()) as u8;

        b.inter_data_mut().drl_idx[0] = 0;
        for _ in 0..fi.max_bvp_drl_bits {
            if msac.decode_bools_bypass(1) == 0 {
                break;
            }
            b.inter_data_mut().drl_idx[0] += 1;
        }

        let is_refmv = b.intra_data().is_refmv;
        b.intra_data_mut().is_qpel = (!fi.force_integer_mv) as u8;
        if is_refmv == 0 && !fi.force_integer_mv {
            b.intra_data_mut().is_qpel = msac.decode_bool_adapt(cdf_m.intrabc_precision()) as u8;
        }

        // IntraBC MV residual
        if is_refmv == 0 {
            let mv_prec = 3 + 2 * (b.intra_data().is_qpel as i32);
            let mv = read_mv_full(msac, cdf_dmv, mv_prec);
            {
                let mut y = mv.y();
                let mut x = mv.x();
                if y != 0 && msac.decode_bools_bypass(1) != 0 {
                    y = -y;
                }
                if x != 0 && msac.decode_bools_bypass(1) != 0 {
                    x = -x;
                }
                b.intra_data_mut().intrabc_mv = crate::levels::Mv::from_xy(y, x);
            }
        }

        b.intra_data_mut().morph_pred = 0;
        if !fi.is_inter_or_switch && fi.bawp && fi.allow_screen_content_tools {
            let nb_mp_0 = if nb_boff[0] != -1 { nb_morph[0] } else { 0 };
            let nb_mp_1 = if nb_boff[1] != -1 { nb_morph[1] } else { 0 };
            let ctx = nb_mp_0 as usize + nb_mp_1 as usize;
            b.intra_data_mut().morph_pred = msac.decode_bool_adapt(cdf_m.morph_pred(ctx)) as u8;
        }
        let morph_pred = b.intra_data().morph_pred;

        // TX partition for IntraBC
        let __seg_ll = fi.seg_lossless[b.seg_id as usize] != 0;
        read_tx_part(msac, cdf_m, &mut b, bs, __seg_ll, fi.txfm_switchable);

        // IntraBC context write-back
        if has_luma {
            let aw = 1usize << b_dim[2];
            let lh = 1usize << b_dim[3];

            a.fsc[bx4..bx4 + aw].fill(0);
            a.mode[bx4..bx4 + aw].fill(0); // DC_PRED
            a.midx[bx4..bx4 + aw].fill(0xff);
            a.mrl[bx4..bx4 + aw].fill(0);
            a.multi_mrl[bx4..bx4 + aw].fill(0);
            a.dip[bx4..bx4 + aw].fill(0);
            a.pal_sz[bx4..bx4 + aw].fill(0);
            a.seg_pred[bx4..bx4 + aw].fill(0);
            a.skip_mode[bx4..bx4 + aw].fill(0);
            a.intrabc[bx4..bx4 + aw].fill(1);
            a.morph_pred[bx4..bx4 + aw].fill(morph_pred);
            a.intra[bx4..bx4 + aw].fill(1);
            a.skip_txfm[bx4..bx4 + aw].fill(b.skip_txfm);
            if fi.is_inter_or_switch {
                a.amvd[bx4..bx4 + aw].fill(0);
                a.mvprec[bx4..bx4 + aw].fill(0);
                a.comp_type[bx4..bx4 + aw].fill(0);
                a.motion_mode[bx4..bx4 + aw].fill(0);
                a.r#ref[0][bx4..bx4 + aw].fill(-1);
                a.r#ref[1][bx4..bx4 + aw].fill(-1);
            }

            l.fsc[by4..by4 + lh].fill(0);
            l.mode[by4..by4 + lh].fill(0);
            l.midx[by4..by4 + lh].fill(0xff);
            l.mrl[by4..by4 + lh].fill(0);
            l.multi_mrl[by4..by4 + lh].fill(0);
            l.dip[by4..by4 + lh].fill(0);
            l.pal_sz[by4..by4 + lh].fill(0);
            l.seg_pred[by4..by4 + lh].fill(0);
            l.skip_mode[by4..by4 + lh].fill(0);
            l.intrabc[by4..by4 + lh].fill(1);
            l.morph_pred[by4..by4 + lh].fill(morph_pred);
            l.intra[by4..by4 + lh].fill(1);
            l.skip_txfm[by4..by4 + lh].fill(b.skip_txfm);
            if fi.is_inter_or_switch {
                l.amvd[by4..by4 + lh].fill(0);
                l.mvprec[by4..by4 + lh].fill(0);
                l.comp_type[by4..by4 + lh].fill(0);
                l.motion_mode[by4..by4 + lh].fill(0);
                l.r#ref[0][by4..by4 + lh].fill(-1);
                l.r#ref[1][by4..by4 + lh].fill(-1);
            }
        }
        if has_chroma {
            let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
            let cbx4 = (cbx & 63) as usize;
            let cby4 = (cby & 63) as usize;
            let cbw4 = 1usize << cb_dim[2];
            let cbh4 = 1usize << cb_dim[3];
            a.uvmode[cbx4..cbx4 + cbw4].fill(0); // DC_PRED
            l.uvmode[cby4..cby4 + cbh4].fill(0);
        }
    }

    // Inter mode path
    let mut mvprec_def = 1u8;
    if b.is_intra == 0 && !intrabc {
        {
            let inter = b.inter_data_mut();
            inter.amvd = 0;
            inter.motion_mode = 0; // Translation
            inter.refine_mv = 0;
        }

        // TIP decision
        let is_tip =
            if b.skip_mode == 0 && fi.tip_frame_mode != 0 && cbs == lbs && imax(bw4, bh4) >= 2 {
                let ctx = (if n_ctx >= 1 {
                    (nx_ref0[0] == TIP_FRAME as i8) as usize
                } else {
                    0
                }) + (if n_ctx >= 2 {
                    (nx_ref0[1] == TIP_FRAME as i8) as usize
                } else {
                    0
                });

                msac.decode_bool_adapt(cdf_m.tip(ctx)) != 0
            } else {
                false
            };

        // Compound decision
        let is_comp = if b.skip_mode != 0 {
            true
        } else if !is_tip
            && (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) == 0
            && fi.switchable_comp_refs
            && bw4 * bh4 >= 4
        {
            // fi.refdir[ref] (refdir_intra is -1 from lib init).
            // refdir_intra is -1 (lib.c init); intra/intrabc neighbours use it.
            let refdir = |r: i8| -> i32 {
                if r < 0 {
                    -1
                } else {
                    fi.refdir[r as usize] as i32
                }
            };
            let ctx = match n_ctx {
                2 => {
                    let refa2 = nx_ref1[0];
                    let refb2 = nx_ref1[1];
                    if refa2 == -1 {
                        let refa1 = nx_ref0[0];
                        if refb2 == -1 {
                            let refb1 = nx_ref0[1];
                            ((refdir(refa1) == 1) ^ (refdir(refb1) == 1)) as usize
                        } else {
                            2 + ((nx_intrabc[0] == 0) && refdir(refa1) != 0) as usize
                        }
                    } else if refb2 == -1 {
                        let refb1 = nx_ref0[1];
                        2 + ((nx_intrabc[1] == 0) && refdir(refb1) != 0) as usize
                    } else {
                        4
                    }
                }
                1 => {
                    let ref2 = nx_ref1[0];
                    if ref2 == -1 {
                        let ref1 = nx_ref0[0];
                        ((nx_intrabc[0] == 0) && refdir(ref1) != 0) as usize
                    } else {
                        3
                    }
                }
                _ => 1,
            };

            msac.decode_bool_adapt(cdf_m.comp(ctx)) != 0
        } else {
            false
        };

        if b.skip_mode != 0 {
            // skip_mode DRL index
            b.inter_data_mut().drl_idx[0] = 0;
            let mut ctx = 0usize;
            for _ in 0..fi.max_drl_bits {
                if msac.decode_bool_adapt(cdf_m.skip_mode_drl_idx(ctx)) == 0 {
                    break;
                }
                b.inter_data_mut().drl_idx[0] += 1;
                if ctx < 2 {
                    ctx += 1;
                }
            }
            // skip_mode ref pair: start from the frame skip_mode_refs, then
            // -2518). The first context neighbour with a valid second ref (or a
            // TIP-coded neighbour) supplies the actual ref pair; otherwise the
            // frame skip_mode_refs stand. This only affects the stored ref
            // context (no entropy is read), but later compound blocks key their
            // ref-selection context on it.
            b.ref_pair = fi.skip_mode_refs;
            for n in 0..n_ctx {
                if nx_ref0[n] == TIP_FRAME as i8 {
                    let tip0 = fi.tip.r0();
                    let tip1 = fi.tip.r1();
                    b.ref_pair = crate::levels::RefPair::from_refs(
                        imin(tip0 as i32, tip1 as i32) as i8,
                        imax(tip0 as i32, tip1 as i32) as i8,
                    );
                    break;
                } else if nx_ref1[n] != -1 {
                    b.ref_pair = crate::levels::RefPair::from_refs(nx_ref0[n], nx_ref1[n]);
                    break;
                } else if nx_ref0[n] != -1 {
                    break;
                }
            }
            {
                let inter = b.inter_data_mut();
                inter.comp_type = 1; // COMP_AVG
                inter.inter_mode = CompInterPredMode::NearMvNearMv as u8;
                inter.cwp_idx = 8;
                // has_subpel_filter=0, then recon_b's filter dispatch sets SHARP).
                inter.filter = 2; // DAV2D_FILTER_8TAP_SHARP
                inter.motion_mode = MotionMode::Translation as u8;
                inter.amvd = 0;
                inter.refine_mv = 0;
            }
        } else if is_comp {
            let n_refs = fi.n_ref_frames as i32;
            let (ref0, ref1): (i8, i8);
            if n_refs > 1 {
                let same_refs = fi.num_same_ref_comp as i32;
                let mut n = 0i32;
                let mut cnt = [0u8; 9];
                if n_ctx > 0 {
                    cnt[(nx_ref0[0] + 1) as usize] += 1;
                    cnt[(nx_ref1[0] + 1) as usize] += 1;
                    if n_ctx > 1 {
                        cnt[(nx_ref0[1] + 1) as usize] += 1;
                        cnt[(nx_ref1[1] + 1) as usize] += 1;
                    }
                }
                let mut cnt_rem = (n_ctx as i32) * 2 - cnt[0] as i32 - cnt[8] as i32;
                let mut refs = [-1i8; 2];
                let mut dir = 0u8;
                let mut maybe_same_ref = if same_refs > 0 { 1i32 } else { 0 };
                let mut i = 0i32;
                while i < n_refs + n - 2 + maybe_same_ref {
                    let cnt_cur = cnt[i as usize + 1] as i32;
                    cnt_rem -= cnt_cur;
                    let bit = if n == 0 && (i == 2 || (i >= n_refs - 2 && i + 1 >= same_refs)) {
                        1
                    } else {
                        let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;
                        let cdf = if n == 0 {
                            cdf_m.comp0_ref(ctx, i as usize)
                        } else {
                            let dir_idx = (dir ^ fi.refdir[i as usize]) as usize;
                            cdf_m.comp1_ref(ctx, dir_idx, i as usize)
                        };
                        msac.decode_bool_adapt(cdf) as i32
                    };
                    if bit != 0 {
                        refs[n as usize] = i as i8;
                        n += 1;
                        if n == 2 {
                            break;
                        }
                        dir = fi.refdir[i as usize];
                    }
                    if maybe_same_ref != 0 {
                        maybe_same_ref = if bit == 0 && i + 1 < same_refs { 1 } else { 0 };
                        if bit != 0 {
                            i -= 1;
                            cnt_rem += cnt_cur;
                        }
                    }
                    i += 1;
                }
                if n < 2 {
                    refs[1] = (n_refs - 1) as i8;
                    if n == 0 {
                        refs[0] = (n_refs - 1 - (same_refs < n_refs) as i32) as i8;
                    }
                }
                ref0 = refs[0];
                ref1 = refs[1];
            } else {
                ref0 = 0;
                ref1 = 0;
            }
            b.ref_pair = RefPair::from_refs(ref0, ref1);

            let comp_ctx = crate::env::get_compref_ctx(
                a,
                l,
                by4,
                bx4,
                have_top,
                have_left,
                have_top_right,
                have_bottom_left,
                b_dim,
                b.ref_pair,
                fi.tip,
            ) as usize;

            let inter_mode: u8;
            if ref0 == ref1 {
                let sym = msac.decode_symbol_adapt(cdf_m.comp_mode_sameref(comp_ctx), 3) as u8;
                let mut m = CompInterPredMode::NearMvNearMv as u8 + sym;
                if m > CompInterPredMode::NearMvNewMv as u8 {
                    m += 1;
                } // skip newmv_nearmv
                inter_mode = m;
            } else {
                let joint_ctx = (fi.refdist[ref0 as usize] != -fi.refdist[ref1 as usize]) as usize;
                if msac.decode_bool_adapt(cdf_m.comp_mode_joint(joint_ctx)) != 0 {
                    inter_mode = CompInterPredMode::JointNewMv as u8;
                } else {
                    inter_mode = CompInterPredMode::NearMvNearMv as u8
                        + msac.decode_symbol_adapt(cdf_m.comp_mode(comp_ctx), 4) as u8;
                }
            };

            let mut final_inter_mode = inter_mode;
            if fi.opfl_refine_type == 1
                && inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
                && imin(bw4, bh4) >= 2
                && fi.refdir[ref0 as usize] != fi.refdir[ref1 as usize]
            {
                let ctx = (inter_mode > CompInterPredMode::NearMvNearMv as u8) as usize;
                if msac.decode_bool_adapt(cdf_m.opfl(ctx)) != 0 {
                    final_inter_mode +=
                        6 - (inter_mode >= CompInterPredMode::GlobalMvGlobalMv as u8) as u8;
                }
            }
            b.inter_data_mut().inter_mode = final_inter_mode;

            use crate::tables::COMP_INTER_PRED_MODES;
            let mode_idx = (final_inter_mode - CompInterPredMode::NearMvNearMv as u8) as usize;
            let m_pair = if mode_idx < COMP_INTER_PRED_MODES.len() {
                COMP_INTER_PRED_MODES[mode_idx]
            } else {
                [InterPredMode::NearMv as u8; 2]
            };
            let is_newmv_mode =
                m_pair[0] == InterPredMode::NewMv as u8 || m_pair[1] == InterPredMode::NewMv as u8;
            if fi.adaptive_mvd && is_newmv_mode {
                let amvd_mode_ctx = match final_inter_mode {
                    x if x == CompInterPredMode::NearMvNewMv as u8 => 0usize,
                    x if x == CompInterPredMode::NewMvNearMv as u8 => 1,
                    x if x == CompInterPredMode::OpflNearMvNewMv as u8 => 2,
                    x if x == CompInterPredMode::OpflNewMvNearMv as u8 => 3,
                    x if x == CompInterPredMode::JointNewMv as u8 => 5,
                    x if x == CompInterPredMode::OpflJointNewMv as u8 => 6,
                    x if x == CompInterPredMode::NewMvNewMv as u8 => 7,
                    x if x == CompInterPredMode::OpflNewMvNewMv as u8 => 8,
                    _ => 0,
                };
                let ctx = (nx_ref0[0] == ref0 && nx_amvd[0] != 0) as usize
                    + (if n_ctx > 1 {
                        nx_ref0[1] == ref0 && nx_amvd[1] != 0
                    } else {
                        false
                    }) as usize;
                b.inter_data_mut().amvd =
                    msac.decode_bool_adapt(cdf_m.amvd(amvd_mode_ctx, ctx)) as i8;
            }
            let amvd_val = b.inter_data().amvd;

            let mut jmvd_scale_mode = 0u8;
            if final_inter_mode == CompInterPredMode::JointNewMv as u8
                || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8
            {
                jmvd_scale_mode = if amvd_val != 0 {
                    msac.decode_symbol_adapt(cdf_m.jmvd_amvd_scale_mode(), 2) as u8
                } else {
                    msac.decode_symbol_adapt(cdf_m.jmvd_scale_mode(), 4) as u8
                };
            }

            // For a NEWMV_NEWMV compound block with two distinct references, when
            // every gathered spatial neighbour on both the left and the top edge
            // references each of the block's refs, a warp_causal flag is read that
            // promotes the block to MM_WARP_CAUSAL. Skipping this read (as the old
            // compound path did) desyncs the bitstream on such blocks.
            if final_inter_mode == CompInterPredMode::NewMvNewMv as u8
                && imin(bw4, bh4) > 1
                && !fi.force_integer_mv
                && ref0 != ref1
                && fi.opfl_refine_type != 2
                && fi.motion_modes & (1 << MotionMode::WarpCausal as u8) != 0
            {
                let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                let match_ref_l =
                    |off: usize, r: i8| -> bool { l.r#ref[0][off] == r || l.r#ref[1][off] == r };
                let match_ref_a =
                    |off: usize, r: i8| -> bool { a.r#ref[0][off] == r || a.r#ref[1][off] == r };
                let match_refs = |r: i8| -> bool {
                    let left = match_ref_l(by4, r)
                        || (by + bh4 <= fi.tile_row_end && match_ref_l(by4 + bh4 as usize - 1, r));
                    let top = if is_sb_boundary {
                        let o0 = bx4 & !1;
                        match_ref_a(o0, r)
                            || (((bx + bw4 - 2) & !1) < fi.tile_col_end
                                && match_ref_a((bx4 + bw4 as usize - 2) & !1, r))
                    } else {
                        match_ref_a(bx4, r)
                            || (bx + bw4 <= fi.tile_col_end
                                && match_ref_a(bx4 + bw4 as usize - 1, r))
                    };
                    left || top
                };
                if match_refs(ref0) && match_refs(ref1) {
                    let x1 = if nb_boff[0] == -1 {
                        MotionMode::Translation as u8
                    } else {
                        nb_motion_mode[0]
                    };
                    let x2 = if nb_boff[1] == -1 {
                        MotionMode::Translation as u8
                    } else {
                        nb_motion_mode[1]
                    };
                    let wc = MotionMode::WarpCausal as u8;
                    let cs_ctx =
                        (x1 >= wc || x2 >= wc) as usize + (x1 == wc) as usize + (x2 == wc) as usize;
                    if msac.decode_bool_adapt(cdf_m.warp_causal(cs_ctx)) != 0 {
                        b.inter_data_mut().motion_mode = MotionMode::WarpCausal as u8;
                    }
                }
            }

            b.inter_data_mut().drl_idx = [0; 2];
            if final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8 {
                let n_drls = 1 + (final_inter_mode <= CompInterPredMode::NearMvNewMv as u8) as i32;
                let max_drl = fi.max_drl_bits as i32;
                let mut n = 0i32;
                let mut ctx = 0usize;
                for r in 0..n_drls {
                    while n < max_drl {
                        if msac.decode_bool_adapt(cdf_m.drl_idx(ctx, comp_ctx)) == 0 {
                            break;
                        }
                        n += 1;
                        if ctx < 2 {
                            ctx += 1;
                        }
                    }
                    b.inter_data_mut().drl_idx[r as usize] = n as u8;
                    if final_inter_mode == CompInterPredMode::NearMvNearMv as u8 && ref0 == ref1 {
                        let drl0 = b.inter_data().drl_idx[0] as i32;
                        n = drl0 + (drl0 < max_drl) as i32;
                    } else {
                        n = 0;
                    }
                    ctx = (n as usize).min(2);
                }
                if n_drls == 1 {
                    b.inter_data_mut().drl_idx[1] = b.inter_data().drl_idx[0];
                }
            }

            let mut mv_prec = 3i32 + fi.mv_precision as i32;
            if mv_prec > 3 && amvd_val == 0 && fi.flex_mvres && is_newmv_mode {
                let mvprec1 = if nb_boff[0] == -1 { 0u8 } else { nb_mvprec[0] };
                let mvprec2 = if nb_boff[1] == -1 { 0u8 } else { nb_mvprec[1] };
                let ctx1 = ((mvprec1 & 1) + (mvprec2 & 1)) as usize;
                if msac.decode_bool_adapt(cdf_m.mvprec_def(ctx1)) == 0 {
                    let ctx2 = ((mvprec1 | mvprec2) >> 1) as usize;
                    let idx = msac
                        .decode_symbol_adapt(cdf_m.mvprec_rem(ctx2, (mv_prec - 4) as usize), 2)
                        as usize;
                    mv_prec = MV_PREC_TBL[(mv_prec == 6) as usize][idx] as i32;
                    mvprec_def = 2;
                }
            }
            b.inter_data_mut().mv_prec = mv_prec as i8;

            if final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8 {
                let is_joint = final_inter_mode == CompInterPredMode::JointNewMv as u8
                    || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8;
                // refdist[0] = |refdist(ref0)|, refdist[1] = ±|refdist(ref1)| with
                // the sign set when the two references point in opposite temporal
                // projection of the non-decoded reference.
                let rd0 = fi.absrefdist[ref0 as usize] as i32;
                let mut rd1 = fi.absrefdist[ref1 as usize] as i32;
                let (start, end) = if is_joint {
                    let s = (rd0 < rd1) as usize;
                    if (fi.refdir[ref0 as usize] ^ fi.refdir[ref1 as usize]) != 0 {
                        rd1 = -rd1;
                    }
                    (s, s + 1)
                } else {
                    (0usize, 2usize)
                };
                {
                    let inter = b.inter_data_mut();
                    inter.mv[0] = crate::levels::Mv::default();
                    inter.mv[1] = crate::levels::Mv::default();
                }
                let mut sum_mvd = 0i32;
                let mut nnzc = 0i32;
                for n in start..end {
                    if m_pair.get(n).copied() != Some(InterPredMode::NewMv as u8) {
                        continue;
                    }
                    let mv = if amvd_val != 0 {
                        read_amvd(msac, cdf_m)
                    } else {
                        read_mv_full(msac, cdf_dmv, mv_prec)
                    };
                    b.inter_data_mut().mv[n] = crate::levels::Mv::from_xy(mv.y(), mv.x());
                    if amvd_val == 0 {
                        let cur = b.inter_data().mv[n];
                        sum_mvd += cur.y() + cur.x();
                        nnzc += (cur.y() != 0) as i32 + (cur.x() != 0) as i32;
                    }
                }

                // sign derivation
                if final_inter_mode != CompInterPredMode::NearMvNearMv as u8
                    && final_inter_mode != CompInterPredMode::OpflNearMvNearMv as u8
                {
                    let bidir_newmv = final_inter_mode == CompInterPredMode::NewMvNewMv as u8
                        || final_inter_mode == CompInterPredMode::OpflNewMvNewMv as u8
                        || final_inter_mode == CompInterPredMode::JointNewMv as u8
                        || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8;
                    let drl0 = b.inter_data().drl_idx[0];
                    let drl1 = b.inter_data().drl_idx[1];
                    if !fi.mvd_sign_derive
                        || drl0 != 0
                        || drl1 != 0
                        || nnzc < 3 * (end as i32 - start as i32) - 2
                        || fi.allow_screen_content_tools
                        || fi.mv_precision == 3
                        || mv_prec >= 5
                        || !bidir_newmv
                        || b.inter_data().motion_mode != MotionMode::Translation as u8
                    {
                        nnzc = 5; // disable sign derivation
                    }
                    sum_mvd >>= 6 - mv_prec;
                    let mut nnzc2 = 0i32;
                    for n in start..end {
                        if m_pair.get(n).copied() != Some(InterPredMode::NewMv as u8) {
                            continue;
                        }
                        let cur_y = b.inter_data().mv[n].y();
                        if cur_y != 0 {
                            nnzc2 += 1;
                            let s = if nnzc2 == nnzc {
                                (sum_mvd & 1) != 0
                            } else {
                                msac.decode_bool_bypass() != 0
                            };
                            if s {
                                b.inter_data_mut().mv[n].set_y(-cur_y);
                            }
                        }
                        let cur_x = b.inter_data().mv[n].x();
                        if cur_x != 0 {
                            nnzc2 += 1;
                            let s = if nnzc2 == nnzc {
                                (sum_mvd & 1) != 0
                            } else {
                                msac.decode_bool_bypass() != 0
                            };
                            if s {
                                b.inter_data_mut().mv[n].set_x(-cur_x);
                            }
                        }
                    }

                    // JOINT/OPFL_JOINT modes only one MV residual is coded; the
                    // other is the temporal projection of it (scaled by the
                    // ref-distance ratio) then jmvd-scaled.
                    if is_joint {
                        let derived = start ^ 1;
                        let source = start;
                        // pack per-ref precision: derived ref uses prec 6.
                        let prec_packed = if derived == 0 {
                            6 | ((mv_prec as u8) << 4)
                        } else {
                            (mv_prec as u8) | (6 << 4)
                        };
                        b.inter_data_mut().mv_prec = prec_packed as i8;
                        let proj = crate::refmvs::mv_projection(
                            b.inter_data().mv[source],
                            rd1,
                            rd0,
                            -0xffff,
                            0xffff,
                        );
                        let mut dmv = proj.xy();
                        jmvd_scale(&mut dmv, amvd_val != 0, jmvd_scale_mode as i32);
                        b.inter_data_mut().mv[derived].c = dmv;
                    } else {
                        // Replicate the single precision into both nibbles
                        // in the resolution loop yields the same value for n=0,1.
                        let p = (mv_prec as u8) & 0xf;
                        b.inter_data_mut().mv_prec = (p | (p << 4)) as i8;
                    }
                }
            }

            b.inter_data_mut().refine_mv = 0;
            // The refine_mv block is skipped entirely (no symbol read) when OPFL
            // refinement is switchable and the inter mode is one of the explicit
            // NEW-MV compound modes — these carry their own refine signaling via
            // reads a spurious refine_mv bit and desyncs the bitstream.
            let opfl_switchable_excl = fi.opfl_refine_type == 1
                && matches!(
                    final_inter_mode,
                    x if x == CompInterPredMode::NearMvNewMv as u8
                        || x == CompInterPredMode::NewMvNearMv as u8
                        || x == CompInterPredMode::NewMvNewMv as u8
                        || x == CompInterPredMode::JointNewMv as u8
                );
            if fi.refine_mv_enabled
                && imin(bw4, bh4) >= 2
                && bw4 * bh4 > 4
                && final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
                && fi.refdist[ref0 as usize] == -fi.refdist[ref1 as usize]
                && recon.svc[ref0 as usize][0].scale == 0
                && recon.svc[ref1 as usize][0].scale == 0
                && !opfl_switchable_excl
            {
                let is_opfl_mode = final_inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8;
                let nearmv_nearmv = final_inter_mode == CompInterPredMode::NearMvNearMv as u8
                    || final_inter_mode == CompInterPredMode::OpflNearMvNearMv as u8
                    || final_inter_mode == CompInterPredMode::OpflJointNewMv as u8;
                if nearmv_nearmv {
                    b.inter_data_mut().refine_mv = 2;
                } else if !is_opfl_mode || fi.opfl_refine_type != 1 {
                    let ctx = (final_inter_mode - CompInterPredMode::NearMvNearMv as u8) as usize;
                    let ctx_clamped = ctx.min(10);
                    b.inter_data_mut().refine_mv =
                        msac.decode_bool_adapt(cdf_m.refine_mv(ctx_clamped)) as u8;
                }
            }
            let refine_mv_val = b.inter_data().refine_mv;

            let has_subpel_filter = final_inter_mode <= CompInterPredMode::JointNewMv as u8
                && refine_mv_val == 0
                && b.inter_data().motion_mode == MotionMode::Translation as u8
                && (final_inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
                    || imin(bw4, bh4) == 1);

            b.inter_data_mut().comp_type = 1; // COMP_AVG
            if final_inter_mode <= CompInterPredMode::JointNewMv as u8
                && refine_mv_val != 1
                && !(final_inter_mode == CompInterPredMode::JointNewMv as u8 && amvd_val != 0)
                && fi.masked_compound
                && imin(bw4, bh4) >= 2
            {
                let ffr = fi.furthest_future_refidx;
                let comptype_ctx = |num: usize| -> i32 {
                    if num >= n_ctx as usize {
                        0
                    } else if nx_ref1[num] != -1 {
                        (nx_comp_type[num] > 1) as i32
                    } else {
                        (nx_ref0[num] == ffr) as i32 * 2
                    }
                };
                let cctx0 = comptype_ctx(0);
                let cctx1 = comptype_ctx(1);
                let ctx = (cctx0
                    + cctx1
                    + (cctx0 != 0 && cctx1 != 0) as i32
                    + (fi.absrefdist[ref0 as usize] == fi.absrefdist[ref1 as usize]) as i32 * 6)
                    as usize;
                let has_mask = msac.decode_bool_adapt(cdf_m.comp_type_masked(ctx)) != 0;
                if has_mask {
                    if imax(bw4, bh4) <= 16
                        && msac.decode_bool_adapt(cdf_m.comp_type_weighted()) == 0
                    {
                        {
                            let inter = b.inter_data_mut();
                            inter.comp_type = 2; // COMP_WEDGE
                            inter.wedge_idx = read_wedge_idx(msac, cdf_m);
                            inter.wedge_sign = msac.decode_bool_bypass() as i8;
                        }
                    } else {
                        {
                            let inter = b.inter_data_mut();
                            inter.comp_type = 3; // COMP_SEG
                            inter.mask_sign = msac.decode_bool_bypass() as u8;
                        }
                    }
                }
            }

            b.inter_data_mut().cwp_idx = 8;
            let comp_type_val = b.inter_data().comp_type;
            if refine_mv_val == 0
                && jmvd_scale_mode == 0
                && fi.cwp
                && comp_type_val == 1
                && (final_inter_mode == CompInterPredMode::NearMvNearMv as u8
                    || final_inter_mode == CompInterPredMode::JointNewMv as u8)
            {
                let mut n = 0u8;
                while n < 4 {
                    if msac.decode_bool_adapt(cdf_m.cwp_idx(n as usize)) == 0 {
                        break;
                    }
                    n += 1;
                }
                // cwp_weighting_factor[!(refdir[ref0] ^ refdir[ref1])][n]
                // refs' directions: same-direction pair -> row 1, opposite -> row 0.
                static CWP_WEIGHTING_FACTOR: [[i8; 5]; 2] = [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
                let xor = (fi.refdir[ref0 as usize] ^ fi.refdir[ref1 as usize]) & 1;
                let row = (xor == 0) as usize;
                b.inter_data_mut().cwp_idx = CWP_WEIGHTING_FACTOR[row][n as usize];
            }

            if refine_mv_val != 0 || final_inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8 {
                b.inter_data_mut().filter = 2; // SHARP
            } else if fi.subpel_filter_mode == 4 && has_subpel_filter {
                const N_SW: u8 = N_SWITCHABLE_FILTERS as u8;
                let flt = |i: usize| -> u8 {
                    if nb_boff[i] != -1 && (nb_ref0[i] == ref0 || nb_ref1[i] == ref0) {
                        nb_filter[i]
                    } else {
                        N_SW
                    }
                };
                let flt0 = flt(0);
                let flt1 = flt(1);
                let fctx = 4 + if flt0 == flt1 || flt1 == N_SW {
                    flt0 as usize
                } else if flt0 == N_SW {
                    flt1 as usize
                } else {
                    N_SW as usize
                };
                b.inter_data_mut().filter = msac.decode_symbol_adapt(cdf_m.filter(fctx), 2) as u8;
            } else if fi.subpel_filter_mode == 4 {
                b.inter_data_mut().filter = 0;
            } else {
                b.inter_data_mut().filter = fi.subpel_filter_mode;
            }
        } else {
            b.inter_data_mut().comp_type = 0; // COMP_INTER_NONE

            let ref0: i8;
            if (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) != 0 {
                ref0 = 0;
            } else if is_tip {
                ref0 = TIP_FRAME as i8;
            } else {
                let n_refs = fi.n_ref_frames as i32;
                let mut i = 0i32;
                if n_refs > 1 {
                    let mut cnt = [0u8; 9];
                    if n_ctx > 0 {
                        cnt[(nx_ref0[0] + 1) as usize] += 1;
                        cnt[(nx_ref1[0] + 1) as usize] += 1;
                        if n_ctx > 1 {
                            cnt[(nx_ref0[1] + 1) as usize] += 1;
                            cnt[(nx_ref1[1] + 1) as usize] += 1;
                        }
                    }
                    let mut cnt_rem = (n_ctx as i32) * 2 - cnt[0] as i32 - cnt[8] as i32;
                    loop {
                        let cnt_cur = cnt[i as usize + 1] as i32;
                        cnt_rem -= cnt_cur;
                        let ctx = (cnt_cur - cnt_rem + 1).clamp(0, 2) as usize;

                        if msac.decode_bool_adapt(cdf_m.single_ref(ctx, i as usize)) != 0 {
                            break;
                        }
                        i += 1;
                        if i >= n_refs - 1 {
                            break;
                        }
                    }
                }
                ref0 = i as i8;
            }
            b.ref_pair = crate::levels::RefPair::from_refs(ref0, -1);
            // CWP index: 8 (equal weight) for non-TIP single-ref; TIP blocks use
            b.inter_data_mut().cwp_idx = if is_tip {
                static TIP_WTS: [i8; 8] = [8, 12, 16, 18, 20, 4, 6, -4];
                TIP_WTS[fi.tip_global_wtd_idx as usize]
            } else {
                8
            };

            let sngl_ctx = get_snglref_ctx(
                a,
                l,
                by4,
                bx4,
                have_top,
                have_left,
                have_top_right,
                have_bottom_left,
                b_dim,
                ref0,
            );

            let inter_mode: u8;
            if (fi.seg_globalmv_mask | fi.seg_skip_mask) & (1 << b.seg_id) != 0 {
                inter_mode = InterPredMode::GlobalMv as u8;
            } else if is_tip {
                inter_mode = InterPredMode::NearMv as u8
                    + 2 * msac.decode_bool_adapt(cdf_m.tip_mode()) as u8;
            } else {
                let mut allow_warp = false;
                if imin(bw4, bh4) >= 2 && fi.warp_motion {
                    // At the top SB boundary the above neighbour is read from the
                    // SB-edge cache snapshot (taken before this SB's blocks ran)
                    // at 8x8 resolution; elsewhere from the live `a` context.
                    let a_sb_cache = &recon.a_sb_cache;
                    let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                    let warp_thr = if is_sb_boundary {
                        ((bx + bw4 - 2) & !1) < fi.tile_col_end
                    } else {
                        have_top_right
                    };
                    let warp_ctx = crate::env::get_warp_ctx(
                        a,
                        a_sb_cache,
                        l,
                        by4,
                        bx4,
                        have_top,
                        have_left,
                        warp_thr,
                        have_bottom_left,
                        is_sb_boundary,
                        b_dim,
                        ref0,
                    );
                    allow_warp = msac.decode_bool_adapt(cdf_m.warp(warp_ctx as usize)) != 0;
                }
                if allow_warp {
                    if !fi.force_integer_mv && msac.decode_bool_adapt(cdf_m.warp_newmv()) == 0 {
                        inter_mode = InterPredMode::WarpNewMv as u8;
                    } else {
                        inter_mode = InterPredMode::WarpMv as u8;
                    }
                } else {
                    inter_mode = InterPredMode::NearMv as u8
                        + msac.decode_symbol_adapt(cdf_m.inter_mode(sngl_ctx), 2) as u8;
                }
            };
            b.inter_data_mut().inter_mode = inter_mode;

            if fi.adaptive_mvd && inter_mode == InterPredMode::NewMv as u8 {
                let ctx = (nx_ref0[0] == ref0 && nx_amvd[0] != 0) as usize
                    + (if n_ctx > 1 {
                        nx_ref0[1] == ref0 && nx_amvd[1] != 0
                    } else {
                        false
                    }) as usize;
                b.inter_data_mut().amvd = msac.decode_bool_adapt(cdf_m.amvd(4, ctx)) as i8;
            }
            let amvd_val = b.inter_data().amvd;

            {
                let inter = b.inter_data_mut();
                inter.warp_ref_idx = 0;
                inter.warpmv_with_mvd = 0;
                inter.bawp[0] = 0;
                inter.bawp[1] = 0;
            }

            if !is_tip && inter_mode <= InterPredMode::NewMv as u8 {
                if fi.bawp && inter_mode != InterPredMode::GlobalMv as u8 && imin(bw4, bh4) >= 2 {
                    let bawp0 = msac.decode_bool_adapt(cdf_m.bawp(0)) as u8;
                    if bawp0 != 0 {
                        let ctx = if inter_mode == InterPredMode::NewMv as u8 {
                            2 - amvd_val as usize
                        } else {
                            0
                        };
                        let explicit = msac.decode_bool_adapt(cdf_m.bawp_explicit(ctx)) as u8;
                        let mut val = bawp0 + explicit;
                        if val == 2 {
                            val += msac.decode_bool_adapt(cdf_m.bawp_explicit_scale()) as u8;
                            val |= (ctx as u8) << 2;
                        }
                        b.inter_data_mut().bawp[0] = val;
                        if has_chroma {
                            b.inter_data_mut().bawp[1] =
                                msac.decode_bool_adapt(cdf_m.bawp(1)) as u8;
                        }
                    }
                }

                let bawp0 = b.inter_data().bawp[0];
                if fi.motion_modes & (1 << MotionMode::InterIntra as u8) != 0
                    && bawp0 == 0
                    && bw4 * bh4 > 2
                    && imax(bw4, bh4) <= 16
                    && inter_mode >= InterPredMode::NearMv as u8
                    && inter_mode <= InterPredMode::NewMv as u8
                {
                    let ctx = SIZE_GROUP[bs_idx] as usize;
                    if msac.decode_bool_adapt(cdf_m.interintra(ctx)) != 0 {
                        {
                            let inter = b.inter_data_mut();
                            inter.motion_mode = MotionMode::InterIntra as u8;
                            inter.interintra_mode =
                                msac.decode_symbol_adapt(cdf_m.interintra_mode(ctx), 3) as u8;
                            inter.wedge_idx = -1;
                        }
                        if imin(bw4, bh4) > 1
                            && msac.decode_bool_adapt(cdf_m.interintra_wedge()) != 0
                        {
                            b.inter_data_mut().wedge_idx = read_wedge_idx(msac, cdf_m);
                        }
                    }
                }
            } else if !is_tip {
                b.inter_data_mut().motion_mode = MotionMode::WarpDelta as u8;

                // signal is only read when a spatial neighbour references the same
                // frame. Without this gate WARPNEWMV blocks read extra symbols and
                // desync the parse. The is_sb_boundary top path uses the a_sb_cache
                // is used directly here (exact for non-boundary; SB-boundary refmv
                // edge handling is a follow-up).
                let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                let match_ref_l =
                    |off: usize| -> bool { l.r#ref[0][off] == ref0 || l.r#ref[1][off] == ref0 };
                let match_ref_a =
                    |off: usize| -> bool { a.r#ref[0][off] == ref0 || a.r#ref[1][off] == ref0 };
                let has_cs_ext = if inter_mode == InterPredMode::WarpNewMv as u8 {
                    let left_match = have_left
                        && (match_ref_l(by4)
                            || (by + bh4 <= fi.tile_row_end
                                && match_ref_l(by4 + bh4 as usize - 1)));
                    let top_match = have_top && {
                        if is_sb_boundary {
                            let o0 = bx4 & !1;
                            match_ref_a(o0)
                                || (((bx + bw4 - 2) & !1) < fi.tile_col_end
                                    && match_ref_a((bx4 + bw4 as usize - 2) & !1))
                        } else {
                            match_ref_a(bx4)
                                || (bx + bw4 <= fi.tile_col_end
                                    && match_ref_a(bx4 + bw4 as usize - 1))
                        }
                    };
                    left_match || top_match
                } else {
                    false
                };

                if inter_mode == InterPredMode::WarpNewMv as u8 && has_cs_ext {
                    // warp extend / causal decision
                    let x1 = if nb_boff[0] == -1 {
                        0
                    } else {
                        nb_motion_mode[0]
                    };
                    let x2 = if nb_boff[1] == -1 {
                        0
                    } else {
                        nb_motion_mode[1]
                    };
                    let ext_ctx = (x1 >= MotionMode::WarpCausal as u8) as usize
                        + (x2 >= MotionMode::WarpCausal as u8) as usize;
                    let mm_flags = fi.motion_modes;
                    if mm_flags & (1 << MotionMode::WarpExtend as u8) != 0
                        && msac.decode_bool_adapt(cdf_m.warp_extend(ext_ctx)) != 0
                    {
                        b.inter_data_mut().motion_mode = MotionMode::WarpExtend as u8;
                    } else if (mm_flags & (3 << MotionMode::WarpCausal as u8))
                        == (3 << MotionMode::WarpCausal as u8)
                    {
                        let cs_ctx = (ext_ctx > 0) as usize
                            + (x1 == MotionMode::WarpCausal as u8) as usize
                            + (x2 == MotionMode::WarpCausal as u8) as usize;
                        if msac.decode_bool_adapt(cdf_m.warp_causal(cs_ctx)) != 0 {
                            b.inter_data_mut().motion_mode = MotionMode::WarpCausal as u8;
                        }
                    } else if mm_flags & (1 << MotionMode::WarpCausal as u8) != 0 {
                        b.inter_data_mut().motion_mode = MotionMode::WarpCausal as u8;
                    }
                }

                // warp_ref_idx
                let motion_mode_val = b.inter_data().motion_mode;
                if motion_mode_val == MotionMode::WarpDelta as u8 {
                    let mut wri = 0u8;
                    while wri < 3 {
                        if msac.decode_bool_adapt(cdf_m.warp_ref_idx(wri as usize)) == 0 {
                            break;
                        }
                        wri += 1;
                    }
                    b.inter_data_mut().warp_ref_idx = wri;
                }

                // warpmv_with_mvd
                let warp_ref_idx = b.inter_data().warp_ref_idx;
                if inter_mode == InterPredMode::WarpMv as u8 && warp_ref_idx < 2 {
                    b.inter_data_mut().warpmv_with_mvd =
                        msac.decode_bool_adapt(cdf_m.warpmv_with_mvd()) as u8;
                }
            }

            b.inter_data_mut().drl_idx[0] = 0;
            if inter_mode != InterPredMode::WarpMv as u8
                && inter_mode != InterPredMode::GlobalMv as u8
            {
                let max_drl = fi.max_drl_bits as i32;
                let mut n = 0i32;
                let mut ctx = 0usize;
                while n < max_drl {
                    let cdf = if is_tip {
                        cdf_m.tip_drl_idx(ctx)
                    } else {
                        cdf_m.drl_idx(ctx, sngl_ctx)
                    };
                    if msac.decode_bool_adapt(cdf) == 0 {
                        break;
                    }
                    n += 1;
                    if ctx < 2 {
                        ctx += 1;
                    }
                }
                b.inter_data_mut().drl_idx[0] = n as u8;
            }

            let mut mv_prec = 3i32 + fi.mv_precision as i32;
            if mv_prec > 3
                && amvd_val == 0
                && fi.flex_mvres
                && (inter_mode == InterPredMode::NewMv as u8
                    || inter_mode == InterPredMode::WarpNewMv as u8)
            {
                let mvprec1 = if nb_boff[0] == -1 { 0u8 } else { nb_mvprec[0] };
                let mvprec2 = if nb_boff[1] == -1 { 0u8 } else { nb_mvprec[1] };
                let ctx1 = ((mvprec1 & 1) + (mvprec2 & 1)) as usize;
                if msac.decode_bool_adapt(cdf_m.mvprec_def(ctx1)) == 0 {
                    let ctx2 = ((mvprec1 | mvprec2) >> 1) as usize;
                    let idx = msac
                        .decode_symbol_adapt(cdf_m.mvprec_rem(ctx2, (mv_prec - 4) as usize), 2)
                        as usize;
                    mv_prec = MV_PREC_TBL[(mv_prec == 6) as usize][idx] as i32;
                    mvprec_def = 2;
                }
            }
            b.inter_data_mut().mv_prec = mv_prec as i8;

            let warpmv_with_mvd = b.inter_data().warpmv_with_mvd;
            if inter_mode == InterPredMode::NewMv as u8
                || inter_mode == InterPredMode::WarpNewMv as u8
                || (inter_mode == InterPredMode::WarpMv as u8 && warpmv_with_mvd != 0)
            {
                let mv = if amvd_val != 0 {
                    read_amvd(msac, cdf_m)
                } else {
                    read_mv_full(msac, cdf_dmv, mv_prec)
                };
                b.inter_data_mut().mv[0] = Mv::from_xy(mv.y(), mv.x());

                // sign derivation
                let nnzc;
                let sum_mvd;
                if amvd_val != 0 {
                    nnzc = 3;
                    sum_mvd = 0;
                } else {
                    let nx = (mv.x() != 0) as i32 + (mv.y() != 0) as i32;
                    sum_mvd = (mv.x() + mv.y()) >> (6 - mv_prec);
                    if inter_mode == InterPredMode::WarpMv as u8
                        || nx == 0
                        || !fi.mvd_sign_derive
                        || b.inter_data().motion_mode != MotionMode::Translation as u8
                        || fi.allow_screen_content_tools
                        || fi.mv_precision == 3
                        || mv_prec >= 5
                    {
                        nnzc = 3;
                    } else {
                        nnzc = nx;
                    }
                }
                let mut nnzc2 = 0i32;
                let cur_mv_y = b.inter_data().mv[0].y();
                if cur_mv_y != 0 {
                    nnzc2 += 1;
                    let s = if nnzc2 == nnzc {
                        (sum_mvd & 1) != 0
                    } else {
                        msac.decode_bool_bypass() != 0
                    };
                    if s {
                        b.inter_data_mut().mv[0].set_y(-cur_mv_y);
                    }
                }
                let cur_mv_x = b.inter_data().mv[0].x();
                if cur_mv_x != 0 {
                    nnzc2 += 1;
                    let s = if nnzc2 == nnzc {
                        (sum_mvd & 1) != 0
                    } else {
                        msac.decode_bool_bypass() != 0
                    };
                    if s {
                        b.inter_data_mut().mv[0].set_x(-cur_mv_x);
                    }
                }
            }

            let motion_mode_val = b.inter_data().motion_mode;
            let warp_ref_idx = b.inter_data().warp_ref_idx;
            if inter_mode == InterPredMode::WarpNewMv as u8
                && motion_mode_val == MotionMode::WarpDelta as u8
                && ((fi.six_param_warp_delta && warp_ref_idx == 1) || warp_ref_idx == 0)
            {
                let prec = msac.decode_bool_adapt(cdf_m.warp_delta_prec(bs_idx));
                let np = if fi.six_param_warp_delta && warp_ref_idx == 1 {
                    4
                } else {
                    2
                };
                let step = 2i8 >> prec;
                for n in 0..np {
                    // -> n=0 -> idx 0, n=1 -> idx 1, n=2 -> idx 1, n=3 -> idx 0.
                    let ctx = ((n as u32).wrapping_sub(1) > 1) as usize;
                    let idx = (ctx == 0) as usize;
                    let mut val = msac.decode_symbol_adapt(cdf_m.warp_delta_param(0, idx), 7) as i8;
                    if val == 7 && prec != 0 {
                        val += msac.decode_symbol_adapt(cdf_m.warp_delta_param(1, idx), 7) as i8;
                    }
                    if val != 0 {
                        if msac.decode_bool_adapt(cdf_m.warp_delta_sign()) != 0 {
                            val = -val;
                        }
                        val *= step;
                    }
                    b.inter_data_mut().matrix[n] = val;
                }
                if np == 2 {
                    b.inter_data_mut().matrix[2] = -0x80;
                }
            } else if motion_mode_val == MotionMode::WarpDelta as u8 {
                b.inter_data_mut().matrix = [0; 4];
            }

            b.inter_data_mut().warp_ii = 0;
            if inter_mode == InterPredMode::WarpMv as u8
                && imin(bw4, bh4) >= 2
                && imax(bw4, bh4) <= 16
            {
                let ctx = SIZE_GROUP[bs_idx] as usize;
                if msac.decode_bool_adapt(cdf_m.warp_interintra(ctx)) != 0 {
                    {
                        let inter = b.inter_data_mut();
                        inter.warp_ii = 1;
                        inter.interintra_mode =
                            msac.decode_symbol_adapt(cdf_m.interintra_mode(ctx), 3) as u8;
                        inter.wedge_idx = if msac.decode_bool_adapt(cdf_m.interintra_wedge()) != 0 {
                            read_wedge_idx(msac, cdf_m)
                        } else {
                            -1
                        };
                    }
                }
            }

            let has_subpel_filter = !is_tip
                && inter_mode <= InterPredMode::NewMv as u8
                && (inter_mode != InterPredMode::GlobalMv as u8 || imin(bw4, bh4) == 1);
            if b.skip_mode != 0 || ref0 == TIP_FRAME as i8 {
                b.inter_data_mut().filter = 2; // SHARP
            } else if fi.subpel_filter_mode == 4 {
                // SWITCHABLE
                if has_subpel_filter {
                    // matched on the block's first reference; comp adds 4.
                    const N_SW: u8 = N_SWITCHABLE_FILTERS as u8;
                    let bref0 = ref0;
                    let comp = b.ref_pair.r1() != -1;
                    let flt = |i: usize| -> u8 {
                        if nb_boff[i] != -1 && (nb_ref0[i] == bref0 || nb_ref1[i] == bref0) {
                            nb_filter[i]
                        } else {
                            N_SW
                        }
                    };
                    let flt0 = flt(0);
                    let flt1 = flt(1);
                    let fctx = (comp as usize) * 4
                        + if flt0 == flt1 || flt1 == N_SW {
                            flt0 as usize
                        } else if flt0 == N_SW {
                            flt1 as usize
                        } else {
                            N_SW as usize
                        };
                    b.inter_data_mut().filter =
                        msac.decode_symbol_adapt(cdf_m.filter(fctx), 2) as u8;
                } else {
                    b.inter_data_mut().filter = 0; // REGULAR
                }
            } else {
                b.inter_data_mut().filter = fi.subpel_filter_mode;
            }
        }

        // TX partition for inter
        if has_luma {
            let __seg_ll = fi.seg_lossless[b.seg_id as usize] != 0;
            read_tx_part(msac, cdf_m, &mut b, bs, __seg_ll, fi.txfm_switchable);
        }

        // Inter context write-back
        if has_luma {
            let aw = 1usize << b_dim[2];
            let lh = 1usize << b_dim[3];
            let inter_mode = b.inter_data().inter_mode;
            let comp_type = b.inter_data().comp_type;
            let motion_mode = b.inter_data().motion_mode;
            let amvd = b.inter_data().amvd;
            let refs = b.ref_pair.refs();
            let filter_val = b.inter_data().filter;

            a.seg_pred[bx4..bx4 + aw].fill(0);
            a.skip_mode[bx4..bx4 + aw].fill(b.skip_mode);
            a.intra[bx4..bx4 + aw].fill(0);
            a.intrabc[bx4..bx4 + aw].fill(0);
            a.morph_pred[bx4..bx4 + aw].fill(0);
            a.midx[bx4..bx4 + aw].fill(0xff);
            a.fsc[bx4..bx4 + aw].fill(0);
            a.skip_txfm[bx4..bx4 + aw].fill(b.skip_txfm);
            a.pal_sz[bx4..bx4 + aw].fill(0);
            a.comp_type[bx4..bx4 + aw].fill(comp_type);
            a.filter[bx4..bx4 + aw].fill(filter_val);
            a.mode[bx4..bx4 + aw].fill(inter_mode);
            a.mrl[bx4..bx4 + aw].fill(0);
            a.multi_mrl[bx4..bx4 + aw].fill(0);
            a.dip[bx4..bx4 + aw].fill(0);
            a.r#ref[0][bx4..bx4 + aw].fill(refs[0]);
            a.r#ref[1][bx4..bx4 + aw].fill(refs[1]);
            a.motion_mode[bx4..bx4 + aw].fill(motion_mode);
            a.amvd[bx4..bx4 + aw].fill(amvd as u8);
            a.mvprec[bx4..bx4 + aw].fill(mvprec_def);

            l.seg_pred[by4..by4 + lh].fill(0);
            l.skip_mode[by4..by4 + lh].fill(b.skip_mode);
            l.intra[by4..by4 + lh].fill(0);
            l.intrabc[by4..by4 + lh].fill(0);
            l.morph_pred[by4..by4 + lh].fill(0);
            l.midx[by4..by4 + lh].fill(0xff);
            l.fsc[by4..by4 + lh].fill(0);
            l.skip_txfm[by4..by4 + lh].fill(b.skip_txfm);
            l.pal_sz[by4..by4 + lh].fill(0);
            l.comp_type[by4..by4 + lh].fill(comp_type);
            l.filter[by4..by4 + lh].fill(filter_val);
            l.mode[by4..by4 + lh].fill(inter_mode);
            l.mrl[by4..by4 + lh].fill(0);
            l.multi_mrl[by4..by4 + lh].fill(0);
            l.dip[by4..by4 + lh].fill(0);
            l.r#ref[0][by4..by4 + lh].fill(refs[0]);
            l.r#ref[1][by4..by4 + lh].fill(refs[1]);
            l.motion_mode[by4..by4 + lh].fill(motion_mode);
            l.amvd[by4..by4 + lh].fill(amvd as u8);
            l.mvprec[by4..by4 + lh].fill(mvprec_def);
        }
        if has_chroma {
            let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
            let cbx4 = (cbx & 63) as usize;
            let cby4 = (cby & 63) as usize;
            let cbw4 = 1usize << cb_dim[2];
            let cbh4 = 1usize << cb_dim[3];
            a.uvmode[cbx4..cbx4 + cbw4].fill(0); // DC_PRED
            l.uvmode[cby4..cby4 + cbh4].fill(0);
        }
    }

    // Write the block's segment id into the current-frame segment map over its
    if fi.seg_enabled && has_luma {
        let seg_id = b.seg_id;
        let stride = recon.b4_stride;
        let bw4u = 1usize << b_dim[2];
        let bh4u = bh4 as usize;
        let mut off = (by as isize * stride + bx as isize) as usize;
        for _ in 0..bh4u {
            recon.cur_segmap[off..off + bw4u].fill(seg_id);
            off = (off as isize + stride) as usize;
        }
    }

    if (pass & crate::internal::Pass::Entropy as u8) != 0 {
        recon.scratch.block_rec.push(BlockRecord {
            b,
            bx: bx as i16,
            by: by as i16,
            cbx: cbx as i16,
            cby: cby as i16,
            lbs: lbs as i8,
            cbs: cbs as i8,
        });
    }

    // Builds the per-4px deblock edge masks (filter_y/filter_uv), the lossless
    // and chroma-seg maps, and the LR no-skip mask. These feed the deferred
    // deblock/LR filter pass. Chroma masks only when chroma deblock is enabled.
    {
        let deblock = recon.frm_hdr.deblock;
        let level_y_on = deblock.level_y[0] != 0 || deblock.level_y[1] != 0;
        let level_uv_on = deblock.level_u != 0 || deblock.level_v != 0;
        let layout = recon.seq_hdr.layout;
        let ss_ver = fi.ss_ver;
        let ss_hor = fi.ss_hor;
        let _cbx4 = (cbx & 63) as usize;
        let _cby4 = (cby & 63) as usize;
        let cb_dim = if has_chroma {
            &BLOCK_DIMENSIONS[cbs as u8 as usize]
        } else {
            b_dim
        };
        let cbw4 = (imin(cb_dim[0] as i32, fi.bw - cbx) >> ss_hor) as usize;
        let cbh4 = (imin(cb_dim[1] as i32, fi.bh - cby) >> ss_ver) as usize;

        // segmap_uv: per-4px chroma seg id used by chroma deblock thresholds.
        if fi.seg_enabled && has_chroma && level_uv_on && recon.segmap_uv_stride != 0 {
            let seg_id = b.seg_id;
            let seg_stride = recon.segmap_uv_stride;
            let mut off =
                (((cby >> ss_ver) as isize) * seg_stride + ((cbx >> ss_hor) as isize)) as usize;
            for _ in 0..cbh4 {
                recon.segmap_uv[off..off + cbw4].fill(seg_id);
                off = (off as isize + seg_stride) as usize;
            }
        }

        // lossless_mask: blocks coded losslessly skip the deblock delta entirely.
        if fi.seg_enabled && fi.seg_lossless[b.seg_id as usize] != 0 {
            let m = &mut recon.lf_mask[recon.lf_idx];
            if has_luma {
                let bw4u = 1usize << b_dim[2];
                let mask: u64 = (!0u64 >> (64 - bw4u)) << bx4;
                let parts = [
                    (mask & 0xffff) as u16,
                    ((mask >> 16) & 0xffff) as u16,
                    ((mask >> 32) & 0xffff) as u16,
                    ((mask >> 48) & 0xffff) as u16,
                ];
                let bh4u = bh4 as usize;
                for y in 0..bh4u {
                    let row = &mut m.lossless_mask_y[by4 + y];
                    for k in 0..4 {
                        if parts[k] != 0 {
                            row[k] |= parts[k];
                        }
                    }
                }
            }
            if has_chroma {
                // lossless_mask_uv is a subsampled-chroma grid, so it must be
                // ss, cbw4/cbh4 = cb_dim >> ss). Using the non-subsampled x/dims
                // corrupted the chroma lossless mask on subsampled (4:2:0) clips.
                let ccbx4 = ((cbx & 63) >> ss_hor) as usize;
                let ccby4 = ((cby & 63) >> ss_ver) as usize;
                let ccbw4 = (cb_dim[0] as i32 >> ss_hor) as usize;
                let ccbh4 = (cb_dim[1] as i32 >> ss_ver) as usize;
                let mask: u64 = (!0u64 >> (64 - ccbw4)) << ccbx4;
                let ss_mask: u64 = if ss_hor != 0 { 0xff } else { 0xffff };
                let sh = 16 >> ss_hor;
                let parts = [
                    (mask & ss_mask) as u16,
                    ((mask >> sh) & ss_mask) as u16,
                    ((mask >> (sh * 2)) & ss_mask) as u16,
                    ((mask >> (sh * 3)) & ss_mask) as u16,
                ];
                for y in 0..ccbh4 {
                    let row = &mut m.lossless_mask_uv[ccby4 + y];
                    for k in 0..4 {
                        if parts[k] != 0 {
                            row[k] |= parts[k];
                        }
                    }
                }
            }
        }

        // create_db_mask: the per-edge filter strength masks (filter_y/uv).
        if level_y_on {
            if has_luma {
                let m = &mut recon.lf_mask[recon.lf_idx];
                crate::lf_mask::create_db_mask(
                    &mut m.filter_y,
                    &b,
                    bs,
                    bx,
                    by,
                    fi.bw,
                    fi.bh,
                    layout,
                    false,
                    &mut a.tx_lpf_y[bx4..],
                    &mut l.tx_lpf_y[by4..],
                    recon.frm_hdr,
                    recon.seq_hdr,
                );
            }
            if has_chroma && level_uv_on {
                let m = &mut recon.lf_mask[recon.lf_idx];
                // tx_lpf_uv is the chroma-subsampled above/left edge-level
                // subsampled chroma 4px offset, not the luma-unit cbx4/cby4.
                let cbx4_ss = ((cbx & 63) >> ss_hor) as usize;
                let cby4_ss = ((cby & 63) >> ss_ver) as usize;
                crate::lf_mask::create_db_mask(
                    &mut m.filter_uv,
                    &b,
                    cbs,
                    cbx,
                    cby,
                    fi.bw,
                    fi.bh,
                    layout,
                    true,
                    &mut a.tx_lpf_uv[cbx4_ss..],
                    &mut l.tx_lpf_uv[cby4_ss..],
                    recon.frm_hdr,
                    recon.seq_hdr,
                );
            }
        }

        // residual; consumed by the multi-class Wiener / GDF LR stages.
        if has_luma && b.skip_txfm == 0 {
            let m = &mut recon.lf_mask[recon.lf_idx];
            let bw4u = b_dim[0] as i32;
            let bh4u = b_dim[1] as i32;
            let mask: u32 = (!0u32 >> imax(0, 32 - bw4u)) << (bx4 & 15);
            let bx_idx = ((bx4 & 0x30) >> 4) as usize;
            let mut nmi = by4 >> 1;
            let mut y = 0;
            while y < bh4u {
                let nm = &mut m.noskip_mask[nmi];
                nm[bx_idx] |= mask as u16;
                if bw4u >= 32 {
                    nm[bx_idx + 1] = mask as u16;
                    if bw4u == 64 {
                        nm[2] = mask as u16;
                        nm[3] = mask as u16;
                    }
                }
                nmi += 1;
                y += 2;
            }
        }
    }

    // Entropy-only replay scaffold for intra/key frames: syntax + filter masks
    // have been produced above; consume and store residual coefficients here,
    // then leave pixel prediction / inverse transforms to PASS_RECON.
    if (pass & crate::internal::Pass::Recon as u8) == 0 {
        if b.is_intra == 0 || b.intrabc != 0 {
            return Err(());
        }
        recon_b_intra_phase(
            &mut ReconBCtx {
                recon: &mut *recon,
                msac: &mut *msac,
                cdf_m: &mut *cdf_m,
                a: &mut *a,
                l: &mut *l,
                b: &b,
                fi,
            },
            bx,
            by,
            cbx,
            cby,
            lbs,
            cbs,
            has_luma,
            has_chroma,
            TxPhase::ReadOnly,
            ChromaPhase::ReadOnly,
        )?;
        return Ok(b);
    }

    // For IntraBC blocks: resolve the block vector by adding the parsed residual
    // to the DRL-selected predictor from the spatial refmvs candidate list, then
    // splat the final BV into the refmvs grid. For intra (non-IntraBC) blocks:
    // splat an "intra" entry (invalid mv) so later IntraBC blocks skip them.
    if (fi.allow_intrabc || fi.is_inter_or_switch) && has_luma && b.is_intra != 0 {
        let by4r = (by & 63) as usize;
        if intrabc {
            let mut mvstack = [crate::refmvs::Candidate::default(); 6];
            let mut n_mvs = 0i32;
            let mut warp_cnt = 0i32;
            crate::refmvs::refmvs_find(
                recon.rt,
                recon.rf,
                &[],
                0,
                &Default::default(),
                &mut mvstack,
                None,
                &mut n_mvs,
                &mut warp_cnt,
                RefPair::from_pair(-1),
                bs as u8,
                false,
                by,
                bx,
                recon.seq_hdr,
                recon.frm_hdr,
            );
            let diff = b.intra_data().intrabc_mv;
            // drl_idx can reach max_drl_bits/max_bvp_drl_bits, which a malformed
            // header can push past the 6-entry mvstack; clamp to the stack bound
            // (no-op for valid streams, where drl < n_mvs <= mvstack.len()).
            let drl = (b.inter_data().drl_idx[0] as usize).min(mvstack.len() - 1);
            let mut mv = mvstack[drl].mv[0];
            if mv.bits() == 0 {
                let sbsz = 64 << fi.sb128;
                if by - fi.sb_step < fi.tile_row_start {
                    mv.set_x(-(8 * (sbsz + 256)));
                } else {
                    mv.set_y(-(8 * sbsz));
                }
            }
            if b.intra_data().is_refmv == 0 {
                if b.intra_data().is_qpel == 0 {
                    {
                        let mut mv_xy = mv.xy();
                        crate::env::fix_int_mv_precision(&mut mv_xy);
                        mv = Mv::from_xy(mv_xy.y, mv_xy.x);
                    }
                }
                mv.set_x(mv.x() + diff.x());
                mv.set_y(mv.y() + diff.y());
            }
            b.intra_data_mut().intrabc_mv = mv;

            let mut s_src = crate::refmvs::Block {
                mv: [
                    mv,
                    Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    },
                ],
                r#ref: RefPair::from_pair(-1),
                bs: bs as u8,
                mf: 0,
                ..Default::default()
            };
            let s_off = by4r * 128 + (bx & 127) as usize;
            let t_src = crate::refmvs::TemporalBlock::default();
            crate::refmvs::splat_mv(
                &mut recon.rt.r[s_off..],
                &mut s_src,
                None,
                0,
                &t_src,
                bw4,
                bh4,
            );
            if recon.seq_hdr.refmv_bank {
                b.ref_pair = RefPair::from_pair(-1);
                // The resolved IntraBC block vector lives in intra_data().intrabc_mv,
                // but bank_add (shared with inter) reads inter_data().mv. Mirror the
                // BV into inter.mv[0] so the ref-MV bank stores the real block vector
                // (single ref => mv[1] unused). Without this the bank stored a stale
                // zero MV, corrupting every later IntraBC block's BV predictor.
                {
                    let bv = b.intra_data().intrabc_mv;
                    let id = b.inter_data_mut();
                    id.mv[0] = bv;
                    id.mv[1] = Mv::default();
                }
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
        } else {
            let mut s_src = crate::refmvs::Block {
                mv: [
                    Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    },
                    Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    },
                ],
                r#ref: RefPair::from_pair(-1),
                bs: bs as u8,
                mf: 0,
                ..Default::default()
            };
            let s_off = by4r * 128 + (bx & 127) as usize;
            // splat_intraref temporal block: ref=-1, mv=INVALID_TRAJ
            // inter/switch frames so later frames don't read stale candidates at
            let t_src = crate::refmvs::TemporalBlock {
                mv: crate::refmvs::TemporalBlockMv::from_packed(
                    crate::refmvs::INVALID_TRAJ as u32 * 0x10001,
                ),
                r#ref: RefPair::from_pair(-1),
            };
            let write_temporal = recon.seq_hdr.ref_frame_mvs && !recon.cur_mvs.is_empty();
            if write_temporal {
                let t_stride = recon.rf.rp_stride;
                let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
                crate::refmvs::splat_mv(
                    &mut recon.rt.r[s_off..],
                    &mut s_src,
                    Some(&mut recon.cur_mvs[t_off as usize..]),
                    t_stride,
                    &t_src,
                    bw4,
                    bh4,
                );
            } else {
                crate::refmvs::splat_mv(
                    &mut recon.rt.r[s_off..],
                    &mut s_src,
                    None,
                    0,
                    &t_src,
                    bw4,
                    bh4,
                );
            }
            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_update(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                );
            }
        }
    }

    // Single-reference only: resolve the block MV (refmvs_find DRL candidate +
    // parsed residual, or the global MV) and splat it into the refmvs grid +
    // temporal grid. Compound (ref[1] != -1), warp-causal/extend/delta motion
    // and TIP are deferred; their per-block MC is handled separately.
    if has_luma && b.is_intra == 0 && !intrabc {
        let by4r = (by & 63) as usize;
        let refs = b.ref_pair.refs();
        let is_comp = refs[1] != -1;
        let inter_mode = b.inter_data().inter_mode;
        let motion_mode = b.inter_data().motion_mode;
        let mv_prec = b.inter_data().mv_prec as i32;
        let amvd = b.inter_data().amvd;

        if !is_comp {
            // Resolve the single-ref block MV (including TIP single-ref, ref0 ==
            // neighbouring blocks see ref[0] == TIP_FRAME in the refmvs grid).
            if inter_mode == InterPredMode::GlobalMv as u8 {
                let gmv = crate::env::get_gmv_2d(
                    &recon.frm_hdr.gmv.m[refs[0] as usize],
                    bx,
                    by,
                    bw4,
                    bh4,
                    recon.rf.iw4,
                    recon.rf.ih4,
                    recon.frm_hdr,
                );
                b.inter_data_mut().mv[0] = Mv::from_xy(gmv.y, gmv.x);
            } else {
                let mut mvstack = [crate::refmvs::Candidate::default(); 6];
                let mut n_mvs = 0i32;
                let mut warp_cnt = 0i32;
                let want_warp = inter_mode > InterPredMode::NewMv as u8;
                let mut warp_arr = [[0i32; 7]; 6];
                let rp_proj_off = recon.rt.rp_proj_off;
                let rp_proj_slice: &[crate::refmvs::SnglMvBlock] = &recon.rf.rp_proj;
                crate::refmvs::refmvs_find(
                    recon.rt,
                    recon.rf,
                    rp_proj_slice,
                    rp_proj_off as isize,
                    &recon.rf.rp_traj,
                    &mut mvstack,
                    if want_warp {
                        Some(&mut warp_arr[..])
                    } else {
                        None
                    },
                    &mut n_mvs,
                    &mut warp_cnt,
                    RefPair::from_refs(refs[0], -1),
                    bs as u8,
                    false,
                    by,
                    bx,
                    recon.seq_hdr,
                    recon.frm_hdr,
                );
                let diff = b.inter_data().mv[0];
                // Clamp drl to the mvstack bound (see the IntraBC path above).
                let drl = (b.inter_data().drl_idx[0] as usize).min(mvstack.len() - 1);
                let mut mv = if inter_mode == InterPredMode::WarpMv as u8 {
                    let wri = b.inter_data().warp_ref_idx as usize;
                    let prec = if b.inter_data().warpmv_with_mvd != 0 {
                        mv_prec
                    } else {
                        6
                    };
                    crate::env::get_warpmv_2d(
                        &[
                            warp_arr[wri][0],
                            warp_arr[wri][1],
                            warp_arr[wri][2],
                            warp_arr[wri][3],
                            warp_arr[wri][4],
                            warp_arr[wri][5],
                        ],
                        bx,
                        by,
                        bw4,
                        bh4,
                        recon.rf.iw4,
                        recon.rf.ih4,
                        prec,
                    )
                } else {
                    mvstack[drl].mv[0].xy()
                };
                if inter_mode == InterPredMode::NewMv as u8
                    || inter_mode == InterPredMode::WarpNewMv as u8
                    || (inter_mode == InterPredMode::WarpMv as u8
                        && b.inter_data().warpmv_with_mvd != 0)
                {
                    if amvd == 0 && mv_prec <= 3 {
                        crate::env::mv_reduce_prec(&mut mv, mv_prec);
                    }
                    mv.x += diff.x();
                    mv.y += diff.y();
                }
                b.inter_data_mut().mv[0] = Mv::from_xy(mv.y, mv.x);

                // Build t->warpmv[0] for the warp motion modes so recon can do
                // warp-affine MC. WARP_DELTA applies the parsed matrix deltas to
                // the base warp candidate; WARP_CAUSAL re-estimates from
                // neighbour samples; WARP_EXTEND extends a neighbour's matrix.
                let motion_mode_v = b.inter_data().motion_mode;
                if motion_mode_v == MotionMode::WarpDelta as u8 {
                    let wri = b.inter_data().warp_ref_idx as usize;
                    let base = &warp_arr[wri];
                    let m = &mut recon.warpmv[0].matrix;
                    let bmat = b.inter_data().matrix;
                    let mut n = 0usize;
                    while n < 4 && bmat[n] != -0x80 {
                        if bmat[n] != 0 {
                            let bb = ((n.wrapping_sub(1)) >= 2) as i32 * 0x10000;
                            m[2 + n] = iclip(
                                base[n + 2] + bmat[n] as i32 * (1 << 10),
                                bb - 0x7fc0,
                                bb + 0x7fc0,
                            );
                        } else {
                            m[2 + n] = base[n + 2];
                        }
                        n += 1;
                    }
                    if bmat[2] == -0x80 {
                        m[5] = m[2];
                        m[4] = -m[3];
                    }
                    crate::warpmv::set_affine_mv2d(
                        bw4,
                        bh4,
                        b.inter_data().mv[0].xy(),
                        &mut recon.warpmv[0],
                        bx,
                        by,
                    );
                    recon.warpmv[0].wm_type =
                        if crate::warpmv::get_shear_params(&mut recon.warpmv[0]) != 0 {
                            crate::headers::WarpedMotionType::Invalid
                        } else {
                            crate::env::warp_type(&recon.warpmv[0].matrix)
                        };
                } else if motion_mode_v == MotionMode::WarpCausal as u8 {
                    let w4 = imin(bw4, fi.bw - bx);
                    let h4 = imin(bh4, fi.bh - by);
                    derive_warpmv(
                        recon.rt,
                        bx,
                        by,
                        have_top,
                        have_left,
                        bw4,
                        bh4,
                        w4,
                        h4,
                        refs[0],
                        b.inter_data().mv[0],
                        &mut recon.warpmv[0],
                        fi.sb_step,
                        fi.tile_col_end,
                    );
                } else if motion_mode_v == MotionMode::WarpExtend as u8 {
                    let is_sb_boundary = (by & (fi.sb_step - 1)) == 0;
                    let mut y_off = 0i32;
                    let mut x_off = 0i32;
                    let cand = &mvstack[drl];
                    if cand.x_off == -1 || cand.y_off == -1 {
                        y_off = cand.y_off as i32;
                        x_off = cand.x_off as i32;
                        let sb_mask = fi.sb_step - 1;
                        let r = if is_sb_boundary && y_off == -1 {
                            if (bx & sb_mask) != 0 || x_off >= 0 {
                                &recon.rt.ra[recon.rt.ra_off + ((bx + x_off) >> 1) as usize]
                            } else {
                                &recon.rt.ra_tl
                            }
                        } else {
                            &recon.rt.r
                                [((by + y_off) & 63) as usize * 128 + ((bx + x_off) & 127) as usize]
                        };
                        if r.r#ref.ref_at(0) == TIP_FRAME as i8 {
                            x_off = 0;
                            y_off = 0;
                        }
                    }
                    let ref0 = refs[0];
                    let match_ref = |r: &crate::refmvs::Block| -> bool {
                        r.r#ref.r0() == ref0 || r.r#ref.r1() == ref0
                    };
                    // left neighbour on the current row, lmt the top neighbour.
                    let tml_ok = have_left && {
                        let r = &recon.rt.r[(by & 63) as usize * 128 + ((bx - 1) & 127) as usize];
                        match_ref(r)
                    };
                    let bml_ok = have_left && by + bh4 <= fi.tile_row_end && {
                        let r = &recon.rt.r
                            [((by + bh4 - 1) & 63) as usize * 128 + ((bx - 1) & 127) as usize];
                        match_ref(r)
                    };
                    let lmt_ok = have_top && {
                        let r = if is_sb_boundary {
                            &recon.rt.ra[recon.rt.ra_off + ((bx & !1) >> 1) as usize]
                        } else {
                            &recon.rt.r[((by - 1) & 63) as usize * 128 + (bx & 127) as usize]
                        };
                        match_ref(r)
                    };
                    let rmt_ok = have_top && bx + bw4 <= fi.tile_col_end && {
                        let r = if is_sb_boundary {
                            &recon.rt.ra[recon.rt.ra_off + (((bx & !1) + bw4 - 2) >> 1) as usize]
                        } else {
                            &recon.rt.r
                                [((by - 1) & 63) as usize * 128 + ((bx + bw4 - 1) & 127) as usize]
                        };
                        match_ref(r)
                    };
                    if x_off != 0 || y_off != 0 {
                        // already set above
                    } else if bml_ok {
                        y_off = bh4 - 1;
                        x_off = -1;
                    } else if rmt_ok {
                        y_off = -1;
                        x_off = -(bx & is_sb_boundary as i32) + bw4 - (1 + is_sb_boundary as i32);
                    } else if tml_ok {
                        y_off = 0;
                        x_off = -1;
                    } else if lmt_ok {
                        y_off = -1;
                        x_off = -(bx & is_sb_boundary as i32);
                    }
                    if x_off != 0 || y_off != 0 {
                        let b_dim_e = &BLOCK_DIMENSIONS[bs as usize];
                        extend_warpmv(
                            recon.rt,
                            bx,
                            by,
                            x_off,
                            y_off,
                            b_dim_e,
                            refs[0],
                            b.inter_data().mv[0],
                            &mut recon.warpmv[0],
                            fi.sb_step,
                            &recon.frm_hdr.gmv.m[refs[0] as usize].matrix,
                        );
                    } else {
                        recon.warpmv[0].wm_type = crate::headers::WarpedMotionType::Invalid;
                    }
                }
            }

            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
            // derived warp matrix to the per-ref warp bank so later WARP_DELTA /
            // WARP_MV blocks can use it as a base candidate.
            if motion_mode > MotionMode::InterIntra as u8
                && recon.warpmv[0].wm_type != crate::headers::WarpedMotionType::Invalid
            {
                crate::refmvs::warp_bank_add(
                    &mut recon.rt.warp,
                    &recon.warpmv[0],
                    refs[0] as usize,
                );
            }
            // global-affine splat (mf==2 / mf==1 with warp) is deferred.
            let blk_mv = b.inter_data().mv[0];
            let gmv_affine = inter_mode == InterPredMode::GlobalMv as u8
                && imin(bw4, bh4) > 1
                && recon.frm_hdr.gmv.m[refs[0] as usize].wm_type
                    > crate::headers::WarpedMotionType::Translation;
            if motion_mode <= MotionMode::InterIntra as u8 && !gmv_affine {
                let mf = (inter_mode == InterPredMode::GlobalMv as u8 && imin(bw4, bh4) > 1) as i8;
                let mut s_src = crate::refmvs::Block {
                    mv: [
                        blk_mv,
                        Mv {
                            c: MvXY {
                                y: crate::levels::INVALID_MV,
                                x: 0,
                            },
                        },
                    ],
                    r#ref: RefPair::from_refs(refs[0], -1),
                    bs: bs as u8,
                    mf,
                    subpel_filter: b.inter_data().filter,
                    ..Default::default()
                };
                let s_off = by4r * 128 + (bx & 127) as usize;
                let mut t_src = crate::refmvs::TemporalBlock::default();
                // Temporal grid write target (rf.rp = f->mvs), unless TIP / no
                // ref_frame_mvs.
                let write_temporal = recon.seq_hdr.ref_frame_mvs
                    && refs[0] != TIP_FRAME as i8
                    && !recon.cur_mvs.is_empty();
                if write_temporal {
                    let q = crate::refmvs::quantize_mv(blk_mv);
                    t_src.mv = crate::refmvs::TemporalBlockMv::from_mvs(q, q);
                    t_src.r#ref = if q.bits() == crate::refmvs::INVALID_TRAJ {
                        RefPair::from_pair(-1)
                    } else {
                        RefPair::from_refs(refs[0], refs[0])
                    };
                    let t_stride = recon.rf.rp_stride;
                    let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
                    crate::refmvs::splat_mv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        Some(&mut recon.cur_mvs[t_off as usize..]),
                        t_stride,
                        &t_src,
                        bw4,
                        bh4,
                    );
                } else {
                    crate::refmvs::splat_mv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        None,
                        0,
                        &t_src,
                        bw4,
                        bh4,
                    );
                }
            } else {
                let s_off = by4r * 128 + (bx & 127) as usize;
                let use_local = motion_mode > MotionMode::InterIntra as u8;
                let wm = if use_local {
                    recon.warpmv[0]
                } else {
                    recon.frm_hdr.gmv.m[refs[0] as usize]
                };
                let mut s_src = crate::refmvs::Block {
                    mv: [
                        blk_mv,
                        Mv {
                            c: MvXY {
                                y: crate::levels::INVALID_MV,
                                x: 0,
                            },
                        },
                    ],
                    r#ref: RefPair::from_refs(refs[0], -1),
                    bs: bs as u8,
                    subpel_filter: b.inter_data().filter,
                    ..Default::default()
                };
                if use_local {
                    s_src.lmv[0] = blk_mv;
                    s_src.lmv[1] = Mv {
                        c: MvXY {
                            y: crate::levels::INVALID_MV,
                            x: 0,
                        },
                    };
                    s_src.mf = 2;
                    s_src.m = wm.matrix;
                    s_src.warp_type = wm.wm_type as i8;
                } else {
                    s_src.mf = 1;
                }
                let mat = &wm.matrix;
                let mvx = (mat[2] as i64 - 0x10000) * (bx as i64 + 1) * 4
                    + mat[3] as i64 * (by as i64 + 1) * 4
                    + mat[0] as i64;
                let mvy = mat[4] as i64 * (bx as i64 + 1) * 4
                    + mat[1] as i64
                    + (mat[5] as i64 - 0x10000) * (by as i64 + 1) * 4;
                let mut t_src = crate::refmvs::TemporalBlock::default();
                t_src.r#ref = RefPair::from_refs(refs[0], refs[0]);
                let write_temporal = recon.seq_hdr.ref_frame_mvs
                    && refs[0] != TIP_FRAME as i8
                    && !recon.cur_mvs.is_empty();
                if write_temporal {
                    let t_stride = recon.rf.rp_stride;
                    let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
                    crate::refmvs::splat_warpmv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        Some(&mut recon.cur_mvs[t_off as usize..]),
                        t_stride,
                        &mut t_src,
                        mvy,
                        mvx,
                        &wm,
                        bw4,
                        bh4,
                    );
                } else {
                    crate::refmvs::splat_warpmv(
                        &mut recon.rt.r[s_off..],
                        &mut s_src,
                        None,
                        0,
                        &mut t_src,
                        mvy,
                        mvx,
                        &wm,
                        bw4,
                        bh4,
                    );
                }
            }
        } else if b.skip_mode != 0 {
            // the two-ref/skip flag set, then copy mvstack[drl_idx[0]].
            use crate::tables::COMP_INTER_PRED_MODES;
            let _ = COMP_INTER_PRED_MODES; // keep import path consistent
            let mut mvstack = [crate::refmvs::Candidate::default(); 6];
            let mut n_mvs = 0i32;
            let mut warp_cnt = 0i32;
            let rp_proj_off = recon.rt.rp_proj_off;
            let rp_proj_slice: &[crate::refmvs::SnglMvBlock] = &recon.rf.rp_proj;
            crate::refmvs::refmvs_find(
                recon.rt,
                recon.rf,
                rp_proj_slice,
                rp_proj_off as isize,
                &recon.rf.rp_traj,
                &mut mvstack,
                None,
                &mut n_mvs,
                &mut warp_cnt,
                RefPair {
                    r: [refs[0], refs[1]],
                },
                bs as u8,
                true,
                by,
                bx,
                recon.seq_hdr,
                recon.frm_hdr,
            );
            let drl = b.inter_data().drl_idx[0] as usize;
            let drl = drl.min(mvstack.len() - 1);
            {
                let inter = b.inter_data_mut();
                inter.mv[0] = mvstack[drl].mv[0];
                inter.mv[1] = mvstack[drl].mv[1];
                inter.cwp_idx = mvstack[drl].cwp_idx;
            }

            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
            splat_tworef_mv(recon, &b, bx, by, by4r, bw4, bh4, bs);
        } else if is_comp {
            // Compound (same-ref-pair) MV resolution + tworef splat
            use crate::tables::COMP_INTER_PRED_MODES;
            if inter_mode == CompInterPredMode::GlobalMvGlobalMv as u8 {
                for n in 0..2 {
                    let gmv = crate::env::get_gmv_2d(
                        &recon.frm_hdr.gmv.m[refs[n] as usize],
                        bx,
                        by,
                        bw4,
                        bh4,
                        recon.rf.iw4,
                        recon.rf.ih4,
                        recon.frm_hdr,
                    );
                    b.inter_data_mut().mv[n] = Mv::from_xy(gmv.y, gmv.x);
                }
            } else {
                let mut mvstack = [crate::refmvs::Candidate::default(); 6];
                let mut n_mvs = 0i32;
                let mut warp_cnt = 0i32;
                let rp_proj_off = recon.rt.rp_proj_off;
                let rp_proj_slice: &[crate::refmvs::SnglMvBlock] = &recon.rf.rp_proj;
                // NEARMV_NEWMV) the full compound ref pair is used. For NEAR
                // modes with equal refs, single-ref find then mirror mv[0]->mv[1].
                // Cross-ref NEAR (two separate single-ref finds) is deferred (not
                // present in the bring-up clip — all blocks are same-ref).
                if inter_mode > CompInterPredMode::NearMvNewMv as u8 {
                    crate::refmvs::refmvs_find(
                        recon.rt,
                        recon.rf,
                        rp_proj_slice,
                        rp_proj_off as isize,
                        &recon.rf.rp_traj,
                        &mut mvstack,
                        None,
                        &mut n_mvs,
                        &mut warp_cnt,
                        RefPair {
                            r: [refs[0], refs[1]],
                        },
                        bs as u8,
                        false,
                        by,
                        bx,
                        recon.seq_hdr,
                        recon.frm_hdr,
                    );
                } else if refs[0] == refs[1] {
                    // Same-ref NEAR: single-ref find then mirror mv[0]->mv[1]
                    crate::refmvs::refmvs_find(
                        recon.rt,
                        recon.rf,
                        rp_proj_slice,
                        rp_proj_off as isize,
                        &recon.rf.rp_traj,
                        &mut mvstack,
                        None,
                        &mut n_mvs,
                        &mut warp_cnt,
                        RefPair::from_refs(refs[0], -1),
                        bs as u8,
                        false,
                        by,
                        bx,
                        recon.seq_hdr,
                        recon.frm_hdr,
                    );
                    for c in mvstack.iter_mut() {
                        c.mv[1] = c.mv[0];
                        c.weight = c.weight.wrapping_mul(0x101);
                    }
                } else {
                    // Cross-ref NEAR (distinct refs): two separate single-ref
                    crate::refmvs::refmvs_find(
                        recon.rt,
                        recon.rf,
                        rp_proj_slice,
                        rp_proj_off as isize,
                        &recon.rf.rp_traj,
                        &mut mvstack,
                        None,
                        &mut n_mvs,
                        &mut warp_cnt,
                        RefPair::from_refs(refs[0], -1),
                        bs as u8,
                        false,
                        by,
                        bx,
                        recon.seq_hdr,
                        recon.frm_hdr,
                    );
                    let mut mvstack2 = [crate::refmvs::Candidate::default(); 6];
                    let mut n_mvs2 = 0i32;
                    let mut warp_cnt2 = 0i32;
                    crate::refmvs::refmvs_find(
                        recon.rt,
                        recon.rf,
                        rp_proj_slice,
                        rp_proj_off as isize,
                        &recon.rf.rp_traj,
                        &mut mvstack2,
                        None,
                        &mut n_mvs2,
                        &mut warp_cnt2,
                        RefPair::from_refs(refs[1], -1),
                        bs as u8,
                        false,
                        by,
                        bx,
                        recon.seq_hdr,
                        recon.frm_hdr,
                    );
                    for n in 0..6 {
                        mvstack[n].mv[1] = mvstack2[n].mv[0];
                        mvstack[n].weight = (mvstack[n].weight & 0xff) | (mvstack2[n].weight << 8);
                    }
                }
                let mode_idx = (inter_mode - CompInterPredMode::NearMvNearMv as u8) as usize;
                let m_pair = COMP_INTER_PRED_MODES[mode_idx.min(COMP_INTER_PRED_MODES.len() - 1)];
                let packed_prec = b.inter_data().mv_prec as i32;
                for n in 0..2 {
                    let diff = b.inter_data().mv[n];
                    // Clamp drl to the mvstack bound (see the single-ref path above).
                    let drl = (b.inter_data().drl_idx[n] as usize).min(mvstack.len() - 1);
                    let mut mv = mvstack[drl].mv[n].xy();
                    if m_pair[n] == InterPredMode::NewMv as u8 {
                        // derived reference carries precision 6 (no reduction).
                        let prec_n = (packed_prec >> (n * 4)) & 0xf;
                        if amvd == 0 && prec_n <= 3 {
                            crate::env::mv_reduce_prec(&mut mv, prec_n);
                        }
                        mv.x += diff.x();
                        mv.y += diff.y();
                    }
                    b.inter_data_mut().mv[n] = Mv::from_xy(mv.y, mv.x);
                }
                // Per-ref warp model fit for compound WARP_CAUSAL
                // from its neighbour samples so recon can do warp-affine MC.
                if b.inter_data().motion_mode == MotionMode::WarpCausal as u8 {
                    let w4 = imin(bw4, fi.bw - bx);
                    let h4 = imin(bh4, fi.bh - by);
                    for i in 0..2 {
                        derive_warpmv(
                            recon.rt,
                            bx,
                            by,
                            have_top,
                            have_left,
                            bw4,
                            bh4,
                            w4,
                            h4,
                            refs[i],
                            b.inter_data().mv[i],
                            &mut recon.warpmv[i],
                            fi.sb_step,
                            fi.tile_col_end,
                        );
                    }
                }
            }

            if recon.seq_hdr.refmv_bank {
                crate::refmvs::bank_add(
                    &mut recon.rt.bank,
                    bs,
                    by,
                    bx,
                    fi.sb_step,
                    fi.sb128 != 0,
                    &b,
                );
            }
            splat_tworef_mv(recon, &b, bx, by, by4r, bw4, bh4, bs);
        }
    }

    if pass & (Pass::Recon as u8) != 0 && b.is_intra != 0 {
        recon_b_intra(
            &mut ReconBCtx {
                recon: &mut *recon,
                msac: &mut *msac,
                cdf_m: &mut *cdf_m,
                a: &mut *a,
                l: &mut *l,
                b: &b,
                fi,
            },
            bx,
            by,
            cbx,
            cby,
            lbs,
            cbs,
            has_luma,
            has_chroma,
        )?;
    } else if pass & (Pass::Recon as u8) != 0 && b.is_intra == 0 && !intrabc {
        recon_b_inter(
            recon,
            msac,
            cdf_m,
            a,
            l,
            &b,
            bx,
            by,
            cbx,
            cby,
            lbs,
            cbs,
            has_luma,
            has_chroma,
            ChromaPhase::Both,
            fi,
        )?;
    }

    // SDP: record the luma-only block's intra direction mode + FSC flag into the
    if fi.sdp && fi.has_chroma_layout && !has_chroma {
        let off = ((by & 15) * 16 + (bx & 15)) as usize;
        let bh4_max16 = imin(bh4, 16) as usize;
        let aw = (1usize << b_dim[2]).min(16);
        for y in 0..bh4_max16 {
            let row = off + y * 16;
            recon.scratch.luma_intra_dir_mode_map[row..row + aw].fill(luma_midx);
            recon.scratch.luma_fsc_map[row..row + aw].fill(b.fsc);
        }
    }

    // block's palette into the above/left caches so the next blocks can reuse it.
    if has_luma && b.is_intra != 0 && !intrabc {
        let pal_sz = b.intra_data().pal_sz;
        if pal_sz != 0 {
            let pal = recon.scratch.pal;
            for x in 0..bw4 as usize {
                recon.scratch.al_pal[0][bx4 + x] = pal;
            }
            for y in 0..bh4 as usize {
                recon.scratch.al_pal[1][by4 + y] = pal;
            }
        }
    }

    Ok(b)
}
