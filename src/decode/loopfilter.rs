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
use crate::pixel::Pixel;

pub(crate) struct FilterFrameParams {
    pub(crate) deblock: crate::deblock::DeblockApplyParams,
    pub(crate) y_stride: isize,
    pub(crate) uv_stride: isize,
    pub(crate) ss_hor: bool,
    pub(crate) ss_ver: bool,
    pub(crate) layout: crate::headers::PixelLayout,
    // CDEF
    pub(crate) cdef_damping: i32,
    pub(crate) cdef_on_skiptx: bool,
    pub(crate) cdef_y_strength: [u8; crate::headers::MAX_CDEF_STRENGTHS],
    pub(crate) cdef_uv_strength: [u8; crate::headers::MAX_CDEF_STRENGTHS],
}

impl FilterFrameParams {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        seq_hdr: &crate::headers::SequenceHeader,
        frame_hdr: &crate::headers::FrameHeader,
        ss_hor: i32,
        ss_ver: i32,
        y_stride: isize,
        uv_stride: isize,
        _bitdepth: i32,
        _inloop: u32,
    ) -> Self {
        let db = &frame_hdr.deblock;
        let deblock = crate::deblock::DeblockApplyParams {
            level_y: [db.level_y[0] as i32, db.level_y[1] as i32],
        };
        FilterFrameParams {
            deblock,
            y_stride,
            uv_stride,
            ss_hor: ss_hor != 0,
            ss_ver: ss_ver != 0,
            layout: seq_hdr.layout,
            cdef_damping: frame_hdr.cdef.damping as i32,
            cdef_on_skiptx: frame_hdr.cdef.on_skiptx != 0,
            cdef_y_strength: frame_hdr.cdef.y_strength,
            cdef_uv_strength: frame_hdr.cdef.uv_strength,
        }
    }
}

/// Read-only frame state produced by the decode pass and consumed (without
/// mutation) by every filter worker. Sharing it `&` across workers — instead of
/// deep-cloning it per worker as part of `LoopFilterState` — is what lets the
/// fused pipeline read masks that decode has already published (gated by the
/// per-tile-row progress counters) without copying the whole frame N times.
///
/// The masks are strictly read-only here: the one in-place mutation (the
/// bottom-of-frame tx-edge crop) is hoisted to a single pre-pass
/// (`crop_bottom_edges`) that runs before any filter reads them.
pub(crate) struct FilterShared<'a> {
    pub(crate) mask: &'a [crate::lf_mask::Av2Filter],
    pub(crate) lr_mask: &'a [crate::lf_mask::Av2Restoration],
    pub(crate) segmap_uv: &'a [u8],
    pub(crate) start_of_tile_row: &'a [u8],
    pub(crate) lr_cdef_line: &'a [Vec<u8>; 3],
    pub(crate) lr_cdef_line_hbd: &'a [Vec<u16>; 3],
    pub(crate) uv_segmap_stride: isize,
    pub(crate) base_q: i32,
    pub(crate) gdf_ref_dst_idx: i32,
    pub(crate) wiener_idx: usize,
    pub(crate) ns_subclass_class_idx: Option<usize>,
    pub(crate) restore_planes: i32,
}

/// Empty fallback for `lr_db_line` lookups when a root has no backup (e.g. LR
/// disabled): three empty plane buffers.
pub(crate) const EMPTY_LR_DB_LINE: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];

/// Ensure the per-frame seam line buffers are at full size so the parallel CDEF
/// FILTER / LR passes never resize them (which would race). `cdef_line`/`cdef_top`
/// hold `(bh/2)+2` two-row units; `lr_db_line` holds one 20-line backup per root.
///
/// These buffers live on `LoopFilterState` and are reused across frames; only a
/// resolution/layout change touches the allocator in steady-state benchmarks.
pub(crate) fn ensure_filter_lines(
    cdef_line: &mut Vec<[Vec<u8>; 3]>,
    cdef_top: &mut Vec<[Vec<u8>; 3]>,
    lr_db_line: &mut Vec<[Vec<u8>; 3]>,
    bh: i32,
    y_stride: isize,
    uv_stride: isize,
    n_roots: usize,
    mono: bool,
) {
    let y_ls = y_stride.unsigned_abs();
    let uv_ls = uv_stride.unsigned_abs();
    let need_y = 2 * y_ls;
    let need_uv = 2 * uv_ls;
    let n_units = (bh as usize / 2) + 2;

    fn slot_has(slot: &[Vec<u8>; 3], y_len: usize, uv_len: usize, mono: bool) -> bool {
        slot[0].len() == y_len
            && if mono {
                slot[1].is_empty() && slot[2].is_empty()
            } else {
                slot[1].len() == uv_len && slot[2].len() == uv_len
            }
    }

    let lr_y = y_ls * 20;
    let lr_uv = uv_ls * 20;
    if cdef_line.len() == n_units
        && cdef_top.len() == n_units
        && lr_db_line.len() == n_roots
        && cdef_line
            .first()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && cdef_line
            .last()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && cdef_top
            .first()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && cdef_top
            .last()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && lr_db_line
            .first()
            .map_or(n_roots == 0, |s| slot_has(s, lr_y, lr_uv, mono))
        && lr_db_line
            .last()
            .map_or(n_roots == 0, |s| slot_has(s, lr_y, lr_uv, mono))
    {
        return;
    }

    fn ensure_plane(v: &mut Vec<u8>, len: usize) {
        if v.len() != len {
            v.resize(len, 0);
        }
    }

    fn ensure_slot(slot: &mut [Vec<u8>; 3], y_len: usize, uv_len: usize, mono: bool) {
        ensure_plane(&mut slot[0], y_len);
        if mono {
            slot[1].clear();
            slot[2].clear();
        } else {
            ensure_plane(&mut slot[1], uv_len);
            ensure_plane(&mut slot[2], uv_len);
        }
    }

    cdef_line.resize_with(n_units, || [Vec::new(), Vec::new(), Vec::new()]);
    cdef_top.resize_with(n_units, || [Vec::new(), Vec::new(), Vec::new()]);
    lr_db_line.resize_with(n_roots, || [Vec::new(), Vec::new(), Vec::new()]);

    for slot in cdef_line.iter_mut() {
        ensure_slot(slot, need_y, need_uv, mono);
    }
    for slot in cdef_top.iter_mut() {
        ensure_slot(slot, need_y, need_uv, mono);
    }
    for slot in lr_db_line.iter_mut() {
        ensure_slot(slot, lr_y, lr_uv, mono);
    }
}

#[inline(always)]
fn hbd_sample_stride(byte_stride: isize) -> isize {
    debug_assert_eq!(
        byte_stride % std::mem::size_of::<u16>() as isize,
        0,
        "HBD frame stride must be u16-aligned",
    );
    byte_stride / std::mem::size_of::<u16>() as isize
}

/// High-bit-depth counterpart of [`ensure_filter_lines`]. The frame stores
/// strides in bytes, while these scratch buffers store `u16` samples, so convert
/// once at the boundary and keep all HBD CDEF/LR indexing in sample units.
pub(crate) fn ensure_filter_lines_hbd(
    cdef_line: &mut Vec<[Vec<u16>; 3]>,
    cdef_top: &mut Vec<[Vec<u16>; 3]>,
    lr_db_line: &mut Vec<[Vec<u16>; 3]>,
    bh: i32,
    y_stride: isize,
    uv_stride: isize,
    n_roots: usize,
    mono: bool,
) {
    let y_stride = hbd_sample_stride(y_stride);
    let uv_stride = hbd_sample_stride(uv_stride);
    let y_ls = y_stride.unsigned_abs();
    let uv_ls = uv_stride.unsigned_abs();
    let need_y = 2 * y_ls;
    let need_uv = 2 * uv_ls;
    let n_units = (bh as usize / 2) + 2;

    fn slot_has(slot: &[Vec<u16>; 3], y_len: usize, uv_len: usize, mono: bool) -> bool {
        slot[0].len() == y_len
            && if mono {
                slot[1].is_empty() && slot[2].is_empty()
            } else {
                slot[1].len() == uv_len && slot[2].len() == uv_len
            }
    }

    let lr_y = y_ls * 20;
    let lr_uv = uv_ls * 20;
    if cdef_line.len() == n_units
        && cdef_top.len() == n_units
        && lr_db_line.len() == n_roots
        && cdef_line
            .first()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && cdef_line
            .last()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && cdef_top
            .first()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && cdef_top
            .last()
            .map_or(n_units == 0, |s| slot_has(s, need_y, need_uv, mono))
        && lr_db_line
            .first()
            .map_or(n_roots == 0, |s| slot_has(s, lr_y, lr_uv, mono))
        && lr_db_line
            .last()
            .map_or(n_roots == 0, |s| slot_has(s, lr_y, lr_uv, mono))
    {
        return;
    }

    fn ensure_plane(v: &mut Vec<u16>, len: usize) {
        if v.len() != len {
            v.resize(len, 0);
        }
    }

    fn ensure_slot(slot: &mut [Vec<u16>; 3], y_len: usize, uv_len: usize, mono: bool) {
        ensure_plane(&mut slot[0], y_len);
        if mono {
            slot[1].clear();
            slot[2].clear();
        } else {
            ensure_plane(&mut slot[1], uv_len);
            ensure_plane(&mut slot[2], uv_len);
        }
    }

    cdef_line.resize_with(n_units, || [Vec::new(), Vec::new(), Vec::new()]);
    cdef_top.resize_with(n_units, || [Vec::new(), Vec::new(), Vec::new()]);
    lr_db_line.resize_with(n_roots, || [Vec::new(), Vec::new(), Vec::new()]);

    for slot in cdef_line.iter_mut() {
        ensure_slot(slot, need_y, need_uv, mono);
    }
    for slot in cdef_top.iter_mut() {
        ensure_slot(slot, need_y, need_uv, mono);
    }
    for slot in lr_db_line.iter_mut() {
        ensure_slot(slot, lr_y, lr_uv, mono);
    }
}

pub(crate) const STAGE_DEBLOCK: u8 = 1;
pub(crate) const STAGE_CDEF: u8 = 2;
pub(crate) const STAGE_LR: u8 = 4;
pub(crate) const STAGE_CDEF_SAVE: u8 = 8;
pub(crate) const STAGE_CDEF_FILTER: u8 = 16;
pub(crate) const STAGE_DEBLOCK_COLS: u8 = 32;
pub(crate) const STAGE_DEBLOCK_ROWS: u8 = 64;

pub(crate) struct FilterSb64Scratch<'a> {
    pub(crate) cdef_line: &'a mut [[Vec<u8>; 3]],
    pub(crate) cdef_top: &'a mut [[Vec<u8>; 3]],
    pub(crate) lr_db_line: &'a mut [[Vec<u8>; 3]],
    pub(crate) ccso_tmp_buf: &'a mut Vec<u8>,
    pub(crate) cdef_line_hbd: &'a mut [[Vec<u16>; 3]],
    pub(crate) cdef_top_hbd: &'a mut [[Vec<u16>; 3]],
    pub(crate) lr_db_line_hbd: &'a mut [[Vec<u16>; 3]],
    pub(crate) ccso_tmp_buf_hbd: &'a mut Vec<u16>,
}

pub(crate) struct FilterSb64Dst<'a, BD: BitDepth> {
    pub(crate) y: &'a mut [BD::Pixel],
    pub(crate) u: &'a mut [BD::Pixel],
    pub(crate) v: &'a mut [BD::Pixel],
}

struct FilterSb64DstHbd<'a> {
    y: &'a mut [u16],
    u: &'a mut [u16],
    v: &'a mut [u16],
}

pub(crate) struct FilterSb64Ctx<'a> {
    pub(crate) seq_hdr: &'a crate::headers::SequenceHeader,
    pub(crate) frame_hdr: &'a FrameHeader,
    pub(crate) sh: &'a FilterShared<'a>,
    pub(crate) fp: &'a FilterFrameParams,
    pub(crate) cur_segmap: &'a [u8],
    pub(crate) b4_stride: isize,
    pub(crate) hbd: i32,
    pub(crate) inloop: u32,
    pub(crate) sbh: i32,
    pub(crate) sb_step: i32,
    pub(crate) sb256w: i32,
    pub(crate) sb128: i32,
    pub(crate) bw: i32,
    pub(crate) bh: i32,
}

#[derive(Clone, Copy)]
pub(crate) struct FilterSb64Band {
    pub(crate) by64: i32,
    pub(crate) stages: u8,
}

pub(crate) fn filter_sb64<BD: BitDepth>(
    bd: BD,
    ctx: FilterSb64Ctx<'_>,
    scratch: FilterSb64Scratch<'_>,
    dst: FilterSb64Dst<'_, BD>,
    band: FilterSb64Band,
) {
    if BD::BPC != 8 {
        let FilterSb64Dst { y, u, v } = dst;
        if BD::BPC == 16 {
            if let (Some(y), Some(u), Some(v)) = (
                <BD::Pixel as Pixel>::try_as_u16_slice_mut(y),
                <BD::Pixel as Pixel>::try_as_u16_slice_mut(u),
                <BD::Pixel as Pixel>::try_as_u16_slice_mut(v),
            ) {
                filter_sb64_hbd(
                    crate::pixel::BitDepth16::new(bd.bitdepth()),
                    ctx,
                    scratch,
                    FilterSb64DstHbd { y, u, v },
                    band,
                );
            }
        }
        return;
    }
    let FilterSb64Ctx {
        seq_hdr,
        frame_hdr,
        sh,
        fp,
        cur_segmap,
        b4_stride,
        hbd,
        inloop,
        sbh,
        sb_step,
        sb256w,
        sb128,
        bw,
        bh,
    } = ctx;
    let FilterSb64Scratch {
        cdef_line,
        cdef_top,
        lr_db_line,
        ccso_tmp_buf,
        ..
    } = scratch;
    let FilterSb64Dst {
        y: dst_y,
        u: dst_u,
        v: dst_v,
    } = dst;
    let FilterSb64Band { by64, stages } = band;
    use crate::looprestoration::{
        INLOOPFILTER_CCSO, INLOOPFILTER_CDEF, INLOOPFILTER_DEBLOCK, INLOOPFILTER_GDF,
        INLOOPFILTER_WIENER,
    };

    let _ = bd;

    let dst_y: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst_y);
    let dst_u: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst_u);
    let dst_v: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst_v);

    let sb64h = (bh + 15) >> 4;
    let root_sby = by64 >> sb128;
    let last_sb64_in_root =
        sb128 == 0 || (by64 & ((1 << sb128) - 1)) == ((1 << sb128) - 1) || by64 + 1 >= sb64h;

    let deblock_on = inloop & INLOOPFILTER_DEBLOCK != 0
        && (fp.deblock.level_y[0] != 0 || fp.deblock.level_y[1] != 0);

    let mask_row = ((by64 >> 2) * sb256w) as usize;
    let y_off0 = (by64 * 64) as isize * fp.y_stride;
    let uv_off0 = ((by64 * 64) as isize * fp.uv_stride) >> fp.ss_ver as i32;

    let deblock_any = stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_COLS | STAGE_DEBLOCK_ROWS) != 0;
    if deblock_on && deblock_any {
        let start_of_tile_row = (sh
            .start_of_tile_row
            .get(root_sby as usize)
            .copied()
            .unwrap_or(0)
            & 1)
            != 0;
        let dctx = crate::deblock::DeblockCtx {
            frame_hdr,
            mask: sh.mask,
            mask_row,
            sb256w,
            cur_segmap,
            b4_stride,
            segmap_uv: sh.segmap_uv,
            uv_segmap_stride: sh.uv_segmap_stride,
            hbd,
            ss_hor: fp.ss_hor as i32,
            ss_ver: fp.ss_ver as i32,
            bw,
            bh,
            y_stride: fp.y_stride,
            uv_stride: fp.uv_stride,
            layout: seq_hdr.layout,
        };
        let _ = start_of_tile_row;
        if stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_COLS) != 0 {
            crate::deblock::deblock_sb64_cols(
                crate::pixel::BitDepth8,
                &dctx,
                dst_y,
                y_off0 as usize,
                dst_u,
                dst_v,
                uv_off0 as usize,
                by64,
            );
        }
        if stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_ROWS) != 0 {
            crate::deblock::deblock_sb64_rows(
                crate::pixel::BitDepth8,
                &dctx,
                dst_y,
                y_off0 as usize,
                dst_u,
                dst_v,
                uv_off0 as usize,
                by64,
            );
        }
    }

    // Keep the existing LR backup contract: the current LR implementation is
    // root-sbrow based, so refresh its DB backup immediately before the root row
    // is restored. CDEF/CCSO do not consume `lr_db_line` in this Rust path.
    let copy_db_on = sh.restore_planes != 0
        && inloop & (INLOOPFILTER_WIENER | INLOOPFILTER_GDF) != 0
        && last_sb64_in_root;
    if copy_db_on && stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_ROWS) != 0 {
        let num_lines = 20usize;
        let y_ls = fp.y_stride.unsigned_abs();
        let uv_ls = fp.uv_stride.unsigned_abs();
        let ridx = root_sby as usize;
        let slot = &mut lr_db_line[ridx];
        if slot[0].len() != y_ls * num_lines {
            slot[0].resize(y_ls * num_lines, 0u8);
        }
        if seq_hdr.layout != crate::headers::PixelLayout::I400 {
            for b in slot.iter_mut().skip(1) {
                if b.len() != uv_ls * num_lines {
                    b.resize(uv_ls * num_lines, 0u8);
                }
            }
        }
        let src: [&[u8]; 3] = [&*dst_y, &*dst_u, &*dst_v];
        crate::deblock::copy_db_8bpc(
            slot,
            &src,
            &[fp.y_stride, fp.uv_stride],
            bw as usize,
            bh as usize,
            root_sby,
            frame_hdr.sb128 != 0,
            fp.ss_hor,
            fp.ss_ver,
            sh.restore_planes != 0,
        );
    }

    let cdef_stage_on = stages & (STAGE_CDEF | STAGE_CDEF_SAVE | STAGE_CDEF_FILTER) != 0;
    if cdef_stage_on && seq_hdr.cdef && inloop & (INLOOPFILTER_CDEF | INLOOPFILTER_CCSO) != 0 {
        // Spec order (deblock-all then CDEF-all): the pre-CDEF border SAVE and the
        // FILTER both read the fully deblocked plane. SAVE is a cheap single-owner
        // pass; FILTER is data-parallel across rows. STAGE_CDEF does both (serial).
        let cdef_phase = (if stages & (STAGE_CDEF | STAGE_CDEF_SAVE) != 0 {
            crate::cdef::CDEF_SAVE
        } else {
            0
        }) | (if stages & (STAGE_CDEF | STAGE_CDEF_FILTER) != 0 {
            crate::cdef::CDEF_FILTER
        } else {
            0
        });
        // `cdef_line_toggle` is vestigial (frame-indexed seam lines made the
        // 2-bank toggle unnecessary); cdef_brow ignores it.
        let mut cdef_toggle = 0usize;
        let ccso_on = inloop & INLOOPFILTER_CCSO != 0
            && (frame_hdr.ccso.p[0].enabled != 0
                || frame_hdr.ccso.p[1].enabled != 0
                || frame_hdr.ccso.p[2].enabled != 0);
        let ccso_pcfg = [
            build_ccso_plane_cfg(&frame_hdr, 0),
            build_ccso_plane_cfg(&frame_hdr, 1),
            build_ccso_plane_cfg(&frame_hdr, 2),
        ];
        let any_lossless = frame_hdr.segmentation.enabled != 0
            && (0..MAX_SEGMENTS).any(|i| frame_hdr.segmentation.lossless[i] != 0);

        let _ = (fp.y_stride, fp.uv_stride);
        // Seam line buffers (`cdef_line`/`cdef_top`) are pre-sized once by the
        // caller to the whole frame (`(bh/2)+2` units), so any band's save/read
        // lands at a stable absolute index and the parallel FILTER pass never
        // mutates buffer lengths.
        let mut start = by64 * 16;
        let mut n_blks = 16 - 2 * ((by64 + 1 < sb64h) as i32);
        if by64 > 0 {
            if (start & (sb_step - 1)) == 0 {
                let prev_mask_row = (((by64 - 1) >> 2) * sb256w) as usize;
                let bp = crate::cdef::CdefBrowParams {
                    bw,
                    bh,
                    damping: fp.cdef_damping,
                    layout: fp.layout,
                    on_skip_tx: fp.cdef_on_skiptx,
                    cdef_on: inloop & INLOOPFILTER_CDEF != 0,
                    mask: filter_mask_row(sh.mask, prev_mask_row, sb256w),
                    y_strength: &fp.cdef_y_strength,
                    uv_strength: &fp.cdef_uv_strength,
                    any_lossless,
                    ccso: ccso_on.then_some(crate::cdef::CcsoCfg { p: ccso_pcfg }),
                    sb128: sb128 != 0,
                };
                crate::cdef::cdef_brow_8bpc(
                    crate::cdef::CdefPlaneSetMut {
                        y: &mut *dst_y,
                        u: &mut *dst_u,
                        v: &mut *dst_v,
                    },
                    &bp,
                    [fp.y_stride, fp.uv_stride],
                    crate::cdef::CdefBrowScratch {
                        cdef_line: &mut *cdef_line,
                        cdef_top: &mut *cdef_top,
                        toggle: &mut cdef_toggle,
                        ccso_tmp_buf,
                    },
                    crate::cdef::CdefBrowRange {
                        by_start: start - 2,
                        by_end: start,
                        sby: root_sby,
                        sbrow_start: true,
                        phase: cdef_phase,
                    },
                );
            } else {
                start -= 2;
                n_blks += 2;
            }
        }

        let end = (start + n_blks).min(bh);
        if start < end {
            let bp = crate::cdef::CdefBrowParams {
                bw,
                bh,
                damping: fp.cdef_damping,
                layout: fp.layout,
                on_skip_tx: fp.cdef_on_skiptx,
                cdef_on: inloop & INLOOPFILTER_CDEF != 0,
                mask: filter_mask_row(sh.mask, mask_row, sb256w),
                y_strength: &fp.cdef_y_strength,
                uv_strength: &fp.cdef_uv_strength,
                any_lossless,
                ccso: ccso_on.then_some(crate::cdef::CcsoCfg { p: ccso_pcfg }),
                sb128: sb128 != 0,
            };
            crate::cdef::cdef_brow_8bpc(
                crate::cdef::CdefPlaneSetMut {
                    y: &mut *dst_y,
                    u: &mut *dst_u,
                    v: &mut *dst_v,
                },
                &bp,
                [fp.y_stride, fp.uv_stride],
                crate::cdef::CdefBrowScratch {
                    cdef_line: &mut *cdef_line,
                    cdef_top: &mut *cdef_top,
                    toggle: &mut cdef_toggle,
                    ccso_tmp_buf,
                },
                crate::cdef::CdefBrowRange {
                    by_start: start,
                    by_end: end,
                    sby: root_sby,
                    sbrow_start: false,
                    phase: cdef_phase,
                },
            );
        }
    }

    if stages & STAGE_LR != 0
        && last_sb64_in_root
        && sh.restore_planes != 0
        && inloop & (INLOOPFILTER_WIENER | INLOOPFILTER_GDF) != 0
    {
        LUMA_SNAP.with(|snap_cell| {
            let mut snap = snap_cell.borrow_mut();
            let chroma_lr = {
                let nsw = RestorationType::NsWiener as u8;
                let sw = RestorationType::Switchable as u8;
                let u = frame_hdr.restoration.p[1].restoration_type;
                let v = frame_hdr.restoration.p[2].restoration_type;
                u == nsw || u == sw || v == nsw || v == sw
            };
            let luma_snapshot: &[u8] = if chroma_lr {
                if snap.len() != dst_y.len() {
                    snap.resize(dst_y.len(), 0);
                }
                let sb_luma_h = sb_step * 4;
                let ystride = fp.y_stride.unsigned_abs() as usize;
                let band_lo =
                    (((root_sby * sb_luma_h - 64).max(0)) as usize * ystride).min(dst_y.len());
                let band_hi = ((((root_sby + 1) * sb_luma_h + 64).max(0)) as usize * ystride)
                    .min(dst_y.len());
                snap[band_lo..band_hi].copy_from_slice(&dst_y[band_lo..band_hi]);
                &snap[..]
            } else {
                &[]
            };
            let widx = sh.wiener_idx;
            let pc_subclass_lut: &[u8] = &crate::tables::PC_WIENER_SUB_CLASSIFY[widx];
            let pc_filters: &[[i16; 13]] = &crate::tables::PC_WIENER_FILTERS[widx];
            let ns_subclass_lut: &[u8] = match sh.ns_subclass_class_idx {
                Some(ci) => &crate::tables::PC_WIENER_SUB_CLASSIFY_NS[widx][ci.min(6)],
                None => &crate::tables::PC_WIENER_SUB_CLASSIFY_NS[widx][0],
            };
            let empty_lr_db = EMPTY_LR_DB_LINE;
            let ctx = crate::looprestoration::LrContext {
                restoration_p: &frame_hdr.restoration.p,
                gdf_qp_idx: frame_hdr.gdf.qp_idx as i32,
                gdf_scale: frame_hdr.gdf.scale as i32,
                sb128: frame_hdr.sb128 != 0,
                cfl_ds_filter_index: seq_hdr.cfl_ds_filter_index as i32,
                layout: seq_hdr.layout,
                bw,
                bh,
                sb256w,
                sbh,
                mask: sh.mask,
                lr_mask: sh.lr_mask,
                lr_db_line: lr_db_line.get(root_sby as usize).unwrap_or(&empty_lr_db),
                lr_cdef_line: &sh.lr_cdef_line,
                lf_p_luma: luma_snapshot,
                base_q: sh.base_q,
                gdf_ref_dst_idx: sh.gdf_ref_dst_idx,
                start_of_tile_row: sh.start_of_tile_row,
                ns_subclass_lut,
                pc_subclass_lut,
                pc_filters,
                n_tc: 1,
                inloop_filters: inloop,
                cur_stride: [fp.y_stride, fp.uv_stride],
                unit_size: frame_hdr.restoration.unit_size,
                restore_planes: sh.restore_planes,
            };
            let mut dst: [&mut [u8]; 3] = [dst_y, dst_u, dst_v];
            crate::looprestoration::lr_sbrow_8bpc(&ctx, &mut dst, root_sby);
        });
    }
}

fn filter_sb64_hbd(
    bd: crate::pixel::BitDepth16,
    ctx: FilterSb64Ctx<'_>,
    scratch: FilterSb64Scratch<'_>,
    dst: FilterSb64DstHbd<'_>,
    band: FilterSb64Band,
) {
    let FilterSb64Ctx {
        seq_hdr,
        frame_hdr,
        sh,
        fp,
        cur_segmap,
        b4_stride,
        hbd,
        inloop,
        sbh,
        sb_step,
        sb256w,
        sb128,
        bw,
        bh,
    } = ctx;
    let FilterSb64Scratch {
        cdef_line_hbd: cdef_line,
        cdef_top_hbd: cdef_top,
        lr_db_line_hbd: lr_db_line,
        ccso_tmp_buf_hbd: ccso_tmp_buf,
        ..
    } = scratch;
    let FilterSb64DstHbd {
        y: dst_y,
        u: dst_u,
        v: dst_v,
    } = dst;
    let FilterSb64Band { by64, stages } = band;
    use crate::looprestoration::{
        INLOOPFILTER_CCSO, INLOOPFILTER_CDEF, INLOOPFILTER_DEBLOCK, INLOOPFILTER_WIENER,
    };

    let sb64h = (bh + 15) >> 4;
    let root_sby = by64 >> sb128;
    let last_sb64_in_root =
        sb128 == 0 || (by64 & ((1 << sb128) - 1)) == ((1 << sb128) - 1) || by64 + 1 >= sb64h;

    // `FilterFrameParams` carries picture strides in bytes. From this point on
    // every HBD consumer indexes typed `&[u16]` / `&mut [u16]` planes and `Vec<u16>`
    // seam buffers, so all offsets and strides must be in samples. Mixing the
    // byte stride here advances 10/12-bit planes by 2x and can land the last band
    // exactly at `plane.len()`.
    let y_stride = hbd_sample_stride(fp.y_stride);
    let uv_stride = hbd_sample_stride(fp.uv_stride);
    let y_ls = y_stride.unsigned_abs();
    let uv_ls = uv_stride.unsigned_abs();

    let deblock_on = inloop & INLOOPFILTER_DEBLOCK != 0
        && (fp.deblock.level_y[0] != 0 || fp.deblock.level_y[1] != 0);

    let mask_row = ((by64 >> 2) * sb256w) as usize;
    let y_off0 = (by64 * 64) as isize * y_stride;
    let uv_off0 = ((by64 * 64) as isize * uv_stride) >> fp.ss_ver as i32;

    let deblock_any = stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_COLS | STAGE_DEBLOCK_ROWS) != 0;
    if deblock_on && deblock_any {
        let start_of_tile_row = (sh
            .start_of_tile_row
            .get(root_sby as usize)
            .copied()
            .unwrap_or(0)
            & 1)
            != 0;
        let dctx = crate::deblock::DeblockCtx {
            frame_hdr,
            mask: sh.mask,
            mask_row,
            sb256w,
            cur_segmap,
            b4_stride,
            segmap_uv: sh.segmap_uv,
            uv_segmap_stride: sh.uv_segmap_stride,
            hbd,
            ss_hor: fp.ss_hor as i32,
            ss_ver: fp.ss_ver as i32,
            bw,
            bh,
            y_stride,
            uv_stride,
            layout: seq_hdr.layout,
        };
        let _ = start_of_tile_row;
        if stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_COLS) != 0 {
            crate::deblock::deblock_sb64_cols(
                bd,
                &dctx,
                dst_y,
                y_off0 as usize,
                dst_u,
                dst_v,
                uv_off0 as usize,
                by64,
            );
        }
        if stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_ROWS) != 0 {
            crate::deblock::deblock_sb64_rows(
                bd,
                &dctx,
                dst_y,
                y_off0 as usize,
                dst_u,
                dst_v,
                uv_off0 as usize,
                by64,
            );
        }
    }

    // Keep the existing LR backup contract: the current LR implementation is
    // root-sbrow based, so refresh its DB backup immediately before the root row
    // is restored. CDEF/CCSO do not consume `lr_db_line` in this Rust path.
    let copy_db_on =
        sh.restore_planes != 0 && inloop & INLOOPFILTER_WIENER != 0 && last_sb64_in_root;
    if copy_db_on && stages & (STAGE_DEBLOCK | STAGE_DEBLOCK_ROWS) != 0 {
        let num_lines = 20usize;
        let ridx = root_sby as usize;
        let slot = &mut lr_db_line[ridx];
        if slot[0].len() != y_ls * num_lines {
            slot[0].resize(y_ls * num_lines, 0u16);
        }
        if seq_hdr.layout != crate::headers::PixelLayout::I400 {
            for b in slot.iter_mut().skip(1) {
                if b.len() != uv_ls * num_lines {
                    b.resize(uv_ls * num_lines, 0u16);
                }
            }
        }
        let src: [&[u16]; 3] = [&*dst_y, &*dst_u, &*dst_v];
        crate::deblock::copy_db_hbd(
            slot,
            &src,
            &[y_stride, uv_stride],
            bw as usize,
            bh as usize,
            root_sby,
            frame_hdr.sb128 != 0,
            fp.ss_hor,
            fp.ss_ver,
            sh.restore_planes != 0,
        );
    }

    let cdef_stage_on = stages & (STAGE_CDEF | STAGE_CDEF_SAVE | STAGE_CDEF_FILTER) != 0;
    if cdef_stage_on && seq_hdr.cdef && inloop & (INLOOPFILTER_CDEF | INLOOPFILTER_CCSO) != 0 {
        // Spec order (deblock-all then CDEF-all): the pre-CDEF border SAVE and the
        // FILTER both read the fully deblocked plane. SAVE is a cheap single-owner
        // pass; FILTER is data-parallel across rows. STAGE_CDEF does both (serial).
        let cdef_phase = (if stages & (STAGE_CDEF | STAGE_CDEF_SAVE) != 0 {
            crate::cdef::CDEF_SAVE
        } else {
            0
        }) | (if stages & (STAGE_CDEF | STAGE_CDEF_FILTER) != 0 {
            crate::cdef::CDEF_FILTER
        } else {
            0
        });
        // `cdef_line_toggle` is vestigial (frame-indexed seam lines made the
        // 2-bank toggle unnecessary); cdef_brow ignores it.
        let mut cdef_toggle = 0usize;
        let ccso_on = inloop & INLOOPFILTER_CCSO != 0
            && (frame_hdr.ccso.p[0].enabled != 0
                || frame_hdr.ccso.p[1].enabled != 0
                || frame_hdr.ccso.p[2].enabled != 0);
        let ccso_pcfg = [
            build_ccso_plane_cfg(&frame_hdr, 0),
            build_ccso_plane_cfg(&frame_hdr, 1),
            build_ccso_plane_cfg(&frame_hdr, 2),
        ];
        let any_lossless = frame_hdr.segmentation.enabled != 0
            && (0..MAX_SEGMENTS).any(|i| frame_hdr.segmentation.lossless[i] != 0);

        let mut start = by64 * 16;
        let mut n_blks = 16 - 2 * ((by64 + 1 < sb64h) as i32);
        if by64 > 0 {
            if (start & (sb_step - 1)) == 0 {
                let prev_mask_row = (((by64 - 1) >> 2) * sb256w) as usize;
                let bp = crate::cdef::CdefBrowParams {
                    bw,
                    bh,
                    damping: fp.cdef_damping,
                    layout: fp.layout,
                    on_skip_tx: fp.cdef_on_skiptx,
                    cdef_on: inloop & INLOOPFILTER_CDEF != 0,
                    mask: filter_mask_row(sh.mask, prev_mask_row, sb256w),
                    y_strength: &fp.cdef_y_strength,
                    uv_strength: &fp.cdef_uv_strength,
                    any_lossless,
                    ccso: ccso_on.then_some(crate::cdef::CcsoCfg { p: ccso_pcfg }),
                    sb128: sb128 != 0,
                };
                crate::cdef::cdef_brow(
                    bd,
                    crate::cdef::CdefPlaneSetMut {
                        y: &mut *dst_y,
                        u: &mut *dst_u,
                        v: &mut *dst_v,
                    },
                    &bp,
                    [y_stride, uv_stride],
                    crate::cdef::CdefBrowScratch {
                        cdef_line: &mut *cdef_line,
                        cdef_top: &mut *cdef_top,
                        toggle: &mut cdef_toggle,
                        ccso_tmp_buf,
                    },
                    crate::cdef::CdefBrowRange {
                        by_start: start - 2,
                        by_end: start,
                        sby: root_sby,
                        sbrow_start: true,
                        phase: cdef_phase,
                    },
                );
            } else {
                start -= 2;
                n_blks += 2;
            }
        }

        let end = (start + n_blks).min(bh);
        if start < end {
            let bp = crate::cdef::CdefBrowParams {
                bw,
                bh,
                damping: fp.cdef_damping,
                layout: fp.layout,
                on_skip_tx: fp.cdef_on_skiptx,
                cdef_on: inloop & INLOOPFILTER_CDEF != 0,
                mask: filter_mask_row(sh.mask, mask_row, sb256w),
                y_strength: &fp.cdef_y_strength,
                uv_strength: &fp.cdef_uv_strength,
                any_lossless,
                ccso: ccso_on.then_some(crate::cdef::CcsoCfg { p: ccso_pcfg }),
                sb128: sb128 != 0,
            };
            crate::cdef::cdef_brow(
                bd,
                crate::cdef::CdefPlaneSetMut {
                    y: &mut *dst_y,
                    u: &mut *dst_u,
                    v: &mut *dst_v,
                },
                &bp,
                [y_stride, uv_stride],
                crate::cdef::CdefBrowScratch {
                    cdef_line: &mut *cdef_line,
                    cdef_top: &mut *cdef_top,
                    toggle: &mut cdef_toggle,
                    ccso_tmp_buf,
                },
                crate::cdef::CdefBrowRange {
                    by_start: start,
                    by_end: end,
                    sby: root_sby,
                    sbrow_start: false,
                    phase: cdef_phase,
                },
            );
        }
    }

    if stages & STAGE_LR != 0
        && last_sb64_in_root
        && sh.restore_planes != 0
        && inloop & INLOOPFILTER_WIENER != 0
    {
        LUMA_SNAP_HBD.with(|snap_cell| {
            let mut snap = snap_cell.borrow_mut();
            let chroma_lr = {
                let nsw = RestorationType::NsWiener as u8;
                let sw = RestorationType::Switchable as u8;
                let u = frame_hdr.restoration.p[1].restoration_type;
                let v = frame_hdr.restoration.p[2].restoration_type;
                u == nsw || u == sw || v == nsw || v == sw
            };
            let luma_snapshot: &[u16] = if chroma_lr {
                if snap.len() != dst_y.len() {
                    snap.resize(dst_y.len(), 0);
                }
                let sb_luma_h = sb_step * 4;
                let ystride = y_ls;
                let band_lo =
                    (((root_sby * sb_luma_h - 64).max(0)) as usize * ystride).min(dst_y.len());
                let band_hi = ((((root_sby + 1) * sb_luma_h + 64).max(0)) as usize * ystride)
                    .min(dst_y.len());
                snap[band_lo..band_hi].copy_from_slice(&dst_y[band_lo..band_hi]);
                &snap[..]
            } else {
                &[]
            };
            let widx = sh.wiener_idx;
            let pc_subclass_lut: &[u8] = &crate::tables::PC_WIENER_SUB_CLASSIFY[widx];
            let pc_filters: &[[i16; 13]] = &crate::tables::PC_WIENER_FILTERS[widx];
            let ns_subclass_lut: &[u8] = match sh.ns_subclass_class_idx {
                Some(ci) => &crate::tables::PC_WIENER_SUB_CLASSIFY_NS[widx][ci.min(6)],
                None => &crate::tables::PC_WIENER_SUB_CLASSIFY_NS[widx][0],
            };
            let empty_lr_db = crate::looprestoration::EMPTY_LR_DB_LINE_HBD;
            let ctx = crate::looprestoration::LrContextHbd {
                restoration_p: &frame_hdr.restoration.p,
                sb128: frame_hdr.sb128 != 0,
                cfl_ds_filter_index: seq_hdr.cfl_ds_filter_index as i32,
                layout: seq_hdr.layout,
                bw,
                bh,
                sb256w,
                sbh,
                mask: sh.mask,
                lr_mask: sh.lr_mask,
                lr_db_line: lr_db_line.get(root_sby as usize).unwrap_or(&empty_lr_db),
                lr_cdef_line: sh.lr_cdef_line_hbd,
                lf_p_luma: luma_snapshot,
                base_q: sh.base_q,
                start_of_tile_row: sh.start_of_tile_row,
                ns_subclass_lut,
                pc_subclass_lut,
                pc_filters,
                n_tc: 1,
                inloop_filters: inloop,
                cur_stride: [y_stride, uv_stride],
                unit_size: frame_hdr.restoration.unit_size,
                restore_planes: sh.restore_planes,
                bitdepth_min_8: bd.bitdepth_min_8(),
                bitdepth_max: bd.bitdepth_max(),
            };
            let mut dst: [&mut [u16]; 3] = [dst_y, dst_u, dst_v];
            crate::looprestoration::lr_sbrow_hbd(&ctx, &mut dst, root_sby);
        });
    }
}
/// Borrow the per-SB256 filter-mask row directly.
pub(crate) fn filter_mask_row(
    mask: &[crate::lf_mask::Av2Filter],
    row: usize,
    sb256w: i32,
) -> &[crate::lf_mask::Av2Filter] {
    if row >= mask.len() {
        return &[];
    }
    let end = (row + sb256w.max(0) as usize).min(mask.len());
    &mask[row..end]
}

/// Build the per-plane CCSO config from the frame header.
pub(crate) fn build_ccso_plane_cfg(
    frame_hdr: &FrameHeader,
    pl: usize,
) -> crate::cdef::CcsoPlaneCfg {
    let cp = &frame_hdr.ccso.p[pl];
    let scale_idx = cp.scale_idx as usize;
    let quant_step = crate::tables::CCSO_QUANT_SZ[scale_idx][cp.quant_idx as usize] as i32;
    crate::cdef::CcsoPlaneCfg {
        max_band_log2: cp.max_band_log2 as u32,
        ext_filter: cp.ext_filter_support as usize,
        quant_step,
        edge_clf: cp.edge_clf as u32,
        bo_only: cp.bo_only != 0,
        scale_idx,
        filter_off: cp.filter_off,
    }
}
