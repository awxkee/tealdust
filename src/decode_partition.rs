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

use crate::decode::{PARTITION_SUBB, SbCtx};
use crate::env::{get_partition_ctx, get_partition2_ctx};
use crate::internal::Pass;
use crate::intops::{iclip, imax, imin};
use crate::levels::{BlockPartition, BlockSize};
use crate::msac::MsacReader;
use crate::tables::{BLOCK_DIMENSIONS, SS_BS};

#[inline(always)]
fn chroma_layout_idx(ss_hor: i32, ss_ver: i32) -> Option<usize> {
    match (ss_hor, ss_ver) {
        // SS_BS is ordered as 4:2:0, 4:2:2, 4:4:4.
        (1, 1) => Some(0),
        (1, 0) => Some(1),
        (0, 0) => Some(2),
        _ => None,
    }
}

#[inline(always)]
fn plane_bsize_for_layout(bs: BlockSize, ss_hor: i32, ss_ver: i32) -> Option<BlockSize> {
    if bs == BlockSize::Invalid {
        return None;
    }
    let idx = chroma_layout_idx(ss_hor, ss_ver)?;
    let raw = SS_BS[bs as usize][idx];
    if raw == 255 {
        None
    } else {
        Some(BlockSize::from_raw(raw as i8))
    }
}

#[inline(always)]
fn plane_dims4_for_layout(bs: BlockSize, ss_hor: i32, ss_ver: i32) -> Option<[i32; 2]> {
    let pbs = plane_bsize_for_layout(bs, ss_hor, ss_ver)?;
    let d = &BLOCK_DIMENSIONS[pbs as usize];
    Some([d[0] as i32, d[1] as i32])
}

#[inline(always)]
fn bsize_from_dims4(w4: i32, h4: i32) -> Option<BlockSize> {
    if w4 <= 0 || h4 <= 0 {
        return None;
    }
    let w4 = w4 as u8;
    let h4 = h4 as u8;
    for (i, d) in BLOCK_DIMENSIONS.iter().enumerate() {
        if d[0] == w4 && d[1] == h4 {
            return Some(BlockSize::from_raw(i as i8));
        }
    }
    None
}

/// Whether a luma-space sub-block of `w4`×`h4` (4px units) has a valid chroma
/// plane block for the current subsampling, mirroring AVM `get_plane_block_size()`
/// / `ss_size_lookup`. Do not validate by shifting dimensions manually: AVM
/// deliberately maps tiny valid bases such as 4x4 in 4:2:0 to a minimum 4x4
/// plane block, while still marking shapes such as 4x8 in 4:2:2 invalid.
#[inline(always)]
fn chroma_sub_valid(w4: i32, h4: i32, ss_hor: i32, ss_ver: i32) -> bool {
    bsize_from_dims4(w4, h4)
        .and_then(|bs| plane_bsize_for_layout(bs, ss_hor, ss_ver))
        .is_some()
}

/// AVM `have_nz_chroma_ref_offset`: whether splitting a `bw4`×`bh4` block into
/// `div` parts along `vert` codes chroma once at the parent's size (`div` is 2
/// for HORZ/VERT, 4 for the 3-way and 8 for the 4-way splits; the 3-way also
/// halves the other axis).
#[inline(always)]
fn chroma_merged_at_parent(
    bw4: i32,
    bh4: i32,
    ss_hor: i32,
    ss_ver: i32,
    vert: bool,
    div: i32,
) -> bool {
    let (cw, ch) = ((bw4 * 4) >> ss_hor, (bh4 * 4) >> ss_ver);
    let (along, across) = if vert { (cw, ch) } else { (ch, cw) };
    let across_min = if div == 4 { 8 } else { 4 };
    along < 4 * div || across < across_min
}

pub(crate) fn decode_partition<const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    ctx: &mut SbCtx<'_, UPDATE_CDF, M>,
    pass: u8,
    lbs: BlockSize,
    cbs: BlockSize,
    bs: BlockSize,
    b_dim: &[u8],
    bw4: i32,
    bh4: i32,
    qw4: i32,
    qh4: i32,
    have_h_split: bool,
    have_v_split: bool,
    dir_ptr: &mut i32,
) -> Result<(BlockPartition, BlockSize), ()> {
    let fi = ctx.fi;
    let bx = &*ctx.bx;
    let by = &*ctx.by;
    let a = &*ctx.a;
    let l = &*ctx.l;
    let msac = &mut *ctx.msac;
    let cdf_m = &mut *ctx.cdf_m;
    let part_w = &mut *ctx.part_w;
    let part_w_idx = &mut *ctx.part_w_idx;
    let part_r = ctx.part_r;
    let part_r_idx = &mut *ctx.part_r_idx;
    let intra_region = &mut *ctx.intra_region;
    let sdp_cfl_disallowed = &mut *ctx.sdp_cfl_disallowed;
    let pl = (lbs == BlockSize::Invalid) as usize;
    let pcc = &PARTITION_SUBB[bs as u8 as usize];
    let mut bp = BlockPartition::Invalid;
    let mut cbs = cbs;

    if pass & (Pass::Entropy as u8) != 0 {
        // AVM `check_is_chroma_size_valid` for a SHARED block that codes its own
        // chroma: a split whose first sub-block has no valid chroma plane size
        // is not signalled, unless the split codes chroma once at this block's
        // size. Only 4:2:2 changes the chroma aspect, e.g. VERT of 16×64 gives
        // 8×64 → 4×64 chroma, so HORZ is implied there.
        let own_chroma = pl == 0 && cbs == lbs;
        let (ss_hor, ss_ver) = (fi.ss_hor, fi.ss_ver);
        let shared_sub_ok = move |vert: bool, div: i32| -> bool {
            if !own_chroma || chroma_merged_at_parent(bw4, bh4, ss_hor, ss_ver, vert, div) {
                return true;
            }
            let (sw4, sh4) = if vert {
                (bw4 / div, bh4)
            } else {
                (bw4, bh4 / div)
            };
            chroma_sub_valid(sw4, sh4, ss_hor, ss_ver)
        };
        let bx4 = (*bx & 63) as usize;
        let by4 = (*by & 63) as usize;
        let bwh4ss = if lbs == BlockSize::Invalid {
            match plane_dims4_for_layout(bs, fi.ss_hor, fi.ss_ver) {
                Some(d) => d,
                None => return Err(()),
            }
        } else {
            [bw4, bh4]
        };
        let mut dir = -1i32;

        if imax(bwh4ss[0], bwh4ss[1]) == 1
            || (pl == 1 && bs == BlockSize::Bs8x8)
            || (pcc.part[0][0] & pcc.part[1][0]) == -1
        {
            bp = BlockPartition::None;
        } else if !have_h_split || !have_v_split {
            if bw4 == bh4 {
                // Boundary-implied direction: right-edge off-frame → VERT,
                // bottom-edge off-frame → HORZ. For SHARED square-split-eligible
                // blocks (128×128/256×256, chroma-coupled) the implied direction
                // is only used when its chroma sub-block is a valid plane size
                // (AVM checks partition_allowed[implied] before forcing). When it
                // is invalid (e.g. I422 VERT → 64×128 → 32×128 chroma INVALID) we
                // leave bp=Invalid so the square-split is read instead.
                let implied_dir = have_v_split as i32;
                let sq_eligible = bs == BlockSize::Bs128x128 || bs == BlockSize::Bs256x256;
                let dir_chroma_ok = if !sq_eligible {
                    true
                } else if implied_dir != 0 {
                    // SHARED block: chroma uses frame subsampling
                    chroma_sub_valid(bw4 >> 1, bh4, fi.ss_hor, fi.ss_ver)
                } else {
                    chroma_sub_valid(bw4, bh4 >> 1, fi.ss_hor, fi.ss_ver)
                };
                if dir_chroma_ok {
                    dir = implied_dir;
                    bp = if !have_v_split {
                        BlockPartition::H
                    } else {
                        BlockPartition::V
                    };
                }
            } else if bw4 > bh4 {
                if !have_h_split || fi.bh <= *by + qh4 {
                    dir = 1;
                    bp = BlockPartition::V;
                }
            } else {
                if !have_v_split || fi.bw <= *bx + qw4 {
                    dir = 0;
                    bp = BlockPartition::H;
                }
            }
        }

        if bp == BlockPartition::Invalid {
            if cbs == BlockSize::Bs64x64
                && lbs == BlockSize::Invalid
                && ((*dir_ptr & 0xff) == 0xff
                    || (*dir_ptr & 0x30003) == 0x10002
                    || (*dir_ptr & 0x30003) == 0x20001)
            {
                if (*dir_ptr & 0xff) == 0xff {
                    bp = BlockPartition::None;
                } else {
                    dir = ((*dir_ptr & 0x30003) == 0x10002) as i32;
                    bp = BlockPartition::from_raw(((*dir_ptr >> 8) & 0xff) as i8);
                }
            } else {
                let mix_inter = fi.is_inter_or_switch && *intra_region == 0;
                let ctx1 = get_partition_ctx(a, l, b_dim, pl, by4, bx4);
                let ctx2 = (ctx1 + pcc.ctx[0] as i32 * 4) as usize;

                let is_split = if mix_inter && b_dim[2] + b_dim[3] == 1 {
                    0u32
                } else if !have_h_split || !have_v_split {
                    1u32
                } else {
                    msac.decode_bool_adapt(cdf_m.part_split(pl, ctx2))
                };

                if is_split == 0 {
                    bp = BlockPartition::None;
                } else {
                    if (bs == BlockSize::Bs128x128 || bs == BlockSize::Bs256x256)
                        && have_v_split
                        && have_h_split
                    {
                        let ctx3 = (ctx1 + (bs == BlockSize::Bs256x256) as i32 * 4) as usize;
                        let is_square = msac.decode_bool_adapt(cdf_m.part_square(ctx3));
                        if is_square != 0 {
                            bp = BlockPartition::Split;
                        }
                    } else if (bs == BlockSize::Bs128x128 || bs == BlockSize::Bs256x256)
                        && bp == BlockPartition::Invalid
                    {
                        // Boundary SHARED square block whose implied rect direction
                        // was chroma-invalid (left bp=Invalid above): AVM still
                        // allows PARTITION_SPLIT (is_square_split_eligible is true
                        // for 128×128/256×256 regardless of boundary), so read the
                        // square-split here. If not square, force the orthogonal
                        // chroma-valid rect.
                        let ctx3 = (ctx1 + (bs == BlockSize::Bs256x256) as i32 * 4) as usize;
                        let is_square = msac.decode_bool_adapt(cdf_m.part_square(ctx3));
                        if is_square != 0 {
                            bp = BlockPartition::Split;
                        } else {
                            // right-edge boundary (implied VERT invalid) → HORZ;
                            // bottom-edge (implied HORZ invalid) → VERT.
                            if have_v_split {
                                dir = 0;
                                bp = BlockPartition::H;
                            } else {
                                dir = 1;
                                bp = BlockPartition::V;
                            }
                        }
                    } else if imax(bw4, bh4) >= 32 {
                        bp = if bw4 > bh4 {
                            BlockPartition::V
                        } else {
                            BlockPartition::H
                        };
                    }

                    if bp == BlockPartition::Invalid {
                        let aspect = 1i32 << fi.max_pb_aspect_ratio_log2;
                        let v_aspect = bw4 * aspect >= bh4 * 2;
                        let h_aspect = bh4 * aspect >= bw4 * 2;
                        if !v_aspect && !h_aspect {
                            return Err(());
                        }

                        if imin(bwh4ss[0], bwh4ss[1]) == 1 {
                            dir = (bwh4ss[0] > bwh4ss[1]) as i32;
                        } else if pl == 1 && (bs == BlockSize::Bs8x16 || bs == BlockSize::Bs8x32) {
                            // chroma: no dimension of 4, so VERT (4-wide) disallowed -> HORZ
                            dir = 0;
                        } else if pl == 1 && (bs == BlockSize::Bs16x8 || bs == BlockSize::Bs32x8) {
                            // chroma: no dimension of 4, so HORZ (4-tall) disallowed -> VERT
                            dir = 1;
                        } else {
                            // A split direction is disallowed when the resulting
                            // chroma sub-block has no valid plane size (AVM
                            // check_is_chroma_size_valid via ss_size_lookup):
                            // chroma dim < 4px or chroma aspect > 8:1.
                            let chroma_ok = |vert: bool| -> bool {
                                if pl == 0 {
                                    return shared_sub_ok(vert, 2);
                                }
                                let (sw4, sh4) = if vert {
                                    (bw4 >> 1, bh4)
                                } else {
                                    (bw4, bh4 >> 1)
                                };
                                chroma_sub_valid(sw4, sh4, fi.ss_hor, fi.ss_ver)
                            };
                            let v_ok = v_aspect && chroma_ok(true);
                            let h_ok = h_aspect && chroma_ok(false);
                            if v_ok && h_ok {
                                let ctx4 = (ctx1 + pcc.ctx[1] as i32 * 4) as usize;

                                dir = msac.decode_bool_adapt(cdf_m.part_dir(pl, ctx4)) as i32;
                            } else {
                                dir = v_ok as i32;
                            }
                        }
                        if pcc.part[dir as usize][0] == -1 {
                            return Err(());
                        }
                        bp = if dir != 0 {
                            BlockPartition::V
                        } else {
                            BlockPartition::H
                        };

                        if imax(bw4, bh4) <= 16 {
                            let bwh4ss2 =
                                plane_dims4_for_layout(bs, fi.ss_hor, fi.ss_ver).unwrap_or([0, 0]);
                            let ndir = (!dir) as usize & 1;
                            let ddir = dir as usize;
                            // AVM bans chroma-tree dimensions of 4 in luma-sized
                            // block terms, independent of subsampling; the
                            // subsampled-size checks below only imply that for
                            // 4:2:0/4:2:2.
                            let (bwpx, bhpx) = (b_dim[0] as i32 * 4, b_dim[1] as i32 * 4);
                            let chroma_tree = pl == 1;
                            let chroma_ext_ok = !chroma_tree
                                || !((bwpx <= 16 && bhpx <= 16)
                                    || (bwpx, bhpx) == (8, 32)
                                    || (bwpx, bhpx) == (32, 8)
                                    || (dir == 0 && (bwpx, bhpx) == (32, 16))
                                    || (dir != 0 && (bwpx, bhpx) == (16, 32)));
                            // AVM check_is_chroma_size_valid: in the chroma tree the
                            // first sub-block of the split must map to a valid plane
                            // size (e.g. I422 VERT_3 of 32×64 → 8×64 → 4×64 invalid,
                            // VERT_4A of 64×64 → 8×64 invalid).
                            let sub_ok = |div: i32| -> bool {
                                let (sw4, sh4) = if dir != 0 {
                                    (bw4 / div, bh4)
                                } else {
                                    (bw4, bh4 / div)
                                };
                                chroma_sub_valid(sw4, sh4, fi.ss_hor, fi.ss_ver)
                            };
                            let chroma_3way_ok = chroma_ext_ok
                                && (!chroma_tree || sub_ok(4))
                                && shared_sub_ok(dir != 0, 4);
                            let chroma_4way_ok = shared_sub_ok(dir != 0, 8)
                                && (!chroma_tree
                                    || (chroma_ext_ok
                                        && sub_ok(8)
                                        && if dir == 0 { bhpx == 64 } else { bwpx == 64 }));
                            let has_hv3 = fi.ext_partitions
                                && chroma_3way_ok
                                && bwh4ss[ndir] >= 4
                                && bwh4ss[ddir] >= 2
                                && b_dim[ndir] as i32 * aspect >= b_dim[ddir] as i32 * 4
                                && (cbs != lbs
                                    || (bwh4ss2[ndir] >= 4 && bwh4ss2[ddir] >= 2)
                                    || (if dir != 0 {
                                        if lbs == BlockSize::Bs32x8 {
                                            have_v_split
                                        } else {
                                            *bx + qw4 * 3 < fi.bw
                                        }
                                    } else {
                                        if lbs == BlockSize::Bs8x32 {
                                            have_h_split
                                        } else {
                                            *by + qh4 * 3 < fi.bh
                                        }
                                    }));
                            let has_hv4ab = bwh4ss[ndir] >= 8
                                && fi.uneven_4way
                                && chroma_4way_ok
                                && b_dim[ndir] as i32 * aspect >= b_dim[ddir] as i32 * 8
                                && (cbs != lbs
                                    || bwh4ss2[ndir] >= 8
                                    || (if dir != 0 {
                                        *bx + (qw4 >> 1) * 7 < fi.bw
                                    } else {
                                        *by + (qh4 >> 1) * 7 < fi.bh
                                    }));

                            if has_hv3 || has_hv4ab {
                                if pcc.part[ddir][1] == -1 {
                                    return Err(());
                                }
                                let ctx5 = get_partition2_ctx(a, l, b_dim, pl, dir, by4, bx4);
                                let ctx6 = (ctx5 + pcc.ctx[0] as i32 * 4) as usize;
                                let is_ext = msac.decode_bool_adapt(cdf_m.part_ext(pl, ctx6));
                                if is_ext != 0 {
                                    bp = if dir != 0 {
                                        BlockPartition::V3
                                    } else {
                                        BlockPartition::H3
                                    };
                                    if has_hv4ab {
                                        if pcc.part[ddir][2] == -1 {
                                            return Err(());
                                        }
                                        let is_4way = if !has_hv3 {
                                            1u32
                                        } else {
                                            msac.decode_bool_adapt(cdf_m.part_4way(pl, ctx6))
                                        };
                                        if is_4way != 0 {
                                            let is_a_or_b = msac.decode_bool_bypass();
                                            bp = BlockPartition::from_raw(
                                                BlockPartition::H4A as i8
                                                    + dir as i8 * 2
                                                    + is_a_or_b as i8,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        dir += (dir != -1) as i32;
        if lbs == BlockSize::Invalid && cbs == BlockSize::Bs64x64 {
            *sdp_cfl_disallowed = (dir != -1 && dir != (*dir_ptr & 0x3)) as i32;
        }
        *dir_ptr |= (dir as u8) as i32 | ((bp as i8 as i32) << 8);

        let mut unmix_bit = 0i32;
        if fi.is_inter_or_switch
            && fi.ext_sdp
            && (cbs as i8 | lbs as i8) != BlockSize::Invalid as i8
            && bp != BlockPartition::None
            && (*dir_ptr & (1 << 24)) == 0
            && (bp as i8) < BlockPartition::H4A as i8
            && imin(bw4, bh4) >= 2
            && bs != fi.root_bs
            && imax(bw4, bh4) <= 16
        {
            let sz = b_dim[2] as i32 + b_dim[3] as i32;
            let ctx = iclip(sz - 4, 0, 3) + (sz == 4) as i32;
            let val = msac.decode_bool_adapt(cdf_m.region_type(ctx as usize));
            *intra_region = (val == 0) as i32;
            unmix_bit = *intra_region;
            if *intra_region != 0 {
                cbs = BlockSize::Invalid;
            }
        }
        if fi.n_passes > 1 {
            let val = bp as u8 | ((unmix_bit as u8) << 7);
            if *part_w_idx == part_w.len() {
                part_w.push(val);
            } else {
                part_w[*part_w_idx] = val;
            }
            *part_w_idx += 1;
        }
    } else {
        let val = match part_r.get(*part_r_idx) {
            Some(&val) => val,
            None => return Err(()),
        };
        *part_r_idx += 1;
        if val & 0x80 != 0 {
            if *intra_region != 0 {
                return Err(());
            }
            *intra_region = 1;
            cbs = BlockSize::Invalid;
        }
        bp = BlockPartition::from_raw((val & 0x7f) as i8);
    }

    Ok((bp, cbs))
}
