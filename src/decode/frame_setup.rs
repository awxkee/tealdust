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
use crate::env::BlockContext;
use crate::headers::{AdaptiveBoolean, FrameHeader, MAX_SEGMENTS, RestorationType};
use crate::internal::LoopFilterState;
use crate::intops::imin;
use crate::lf_mask::{Av2Filter, Av2Restoration};

use crate::msac::MsacContextScalar;

#[inline]
fn try_resize_tile_states(v: &mut Vec<crate::internal::TileState>, len: usize) -> Result<(), ()> {
    if len < v.len() {
        v.truncate(len);
        return Ok(());
    }
    if len > v.len() {
        v.try_reserve_exact(len - v.len()).map_err(|_| ())?;
        v.resize_with(len, Default::default);
    }
    Ok(())
}

#[inline]
fn reset_block_context_value(keyframe: bool, is_tip_frame: bool) -> BlockContext {
    BlockContext {
        fsc: [0; 64],
        mode: if !is_tip_frame && !keyframe {
            [13; 64]
        } else {
            [0; 64]
        },
        midx: if !is_tip_frame { [0xff; 64] } else { [0; 64] },
        mrl: [0; 64],
        multi_mrl: [0; 64],
        dip: [0; 64],
        lcoef: if !is_tip_frame { [0x40; 64] } else { [0; 64] },
        ccoef: if !is_tip_frame {
            [[0x40; 64]; 2]
        } else {
            [[0; 64]; 2]
        },
        seg_pred: [0; 64],
        skip_txfm: [0; 64],
        skip_mode: [0; 64],
        intra: if !is_tip_frame && keyframe {
            [1; 64]
        } else {
            [0; 64]
        },
        intrabc: [0; 64],
        morph_pred: [0; 64],
        comp_type: [0; 64],
        reference: if !is_tip_frame && !keyframe {
            [[-1; 64]; 2]
        } else {
            [[0; 64]; 2]
        },
        motion_mode: [0; 64],
        amvd: [0; 64],
        mvprec: [0; 64],
        filter: if !is_tip_frame {
            [N_SWITCHABLE_FILTERS as u8; 64]
        } else {
            [0; 64]
        },
        tx_lpf_y: [3; 64],
        tx_lpf_uv: [2; 64],
        partition: [[0; 64]; 2],
        uvmode: [0; 64],
        pal_sz: [0; 64],
    }
}

#[inline]
fn try_resize_block_contexts(
    v: &mut Vec<BlockContext>,
    len: usize,
    keyframe: bool,
    is_tip_frame: bool,
    reset_existing: bool,
) -> Result<(), ()> {
    if len < v.len() {
        v.truncate(len);
    }

    let old_len = v.len();
    if len > old_len {
        v.try_reserve_exact(len - old_len).map_err(|_| ())?;
        let ptr = v.as_mut_ptr();
        unsafe {
            if reset_existing {
                for i in old_len..len {
                    ptr.add(i)
                        .write(reset_block_context_value(keyframe, is_tip_frame));
                }
            } else {
                for i in old_len..len {
                    ptr.add(i).write(BlockContext::default());
                }
            }
            v.set_len(len);
        }
    }

    if reset_existing {
        for ctx in v.iter_mut().take(old_len.min(len)) {
            reset_context(ctx, keyframe, is_tip_frame);
        }
    }

    Ok(())
}

#[inline]
fn resize_and_reset_filter_mask(v: &mut Vec<Av2Filter>, len: usize) -> Result<(), ()> {
    if len < v.len() {
        v.truncate(len);
    }
    if len > v.capacity() {
        v.try_reserve_exact(len - v.len()).map_err(|_| ())?;
    }

    let old_len = v.len();
    let ptr = v.as_mut_ptr();
    unsafe {
        for i in 0..old_len {
            Av2Filter::write_reset(ptr.add(i));
        }
        for i in old_len..len {
            Av2Filter::write_reset(ptr.add(i));
        }
        v.set_len(len);
    }
    Ok(())
}

#[inline]
fn resize_and_reset_restoration_mask(v: &mut Vec<Av2Restoration>, len: usize) -> Result<(), ()> {
    if len < v.len() {
        v.truncate(len);
    }
    if len > v.capacity() {
        v.try_reserve_exact(len - v.len()).map_err(|_| ())?;
    }

    let old_len = v.len();
    let ptr = v.as_mut_ptr();
    unsafe {
        for i in 0..old_len {
            Av2Restoration::write_reset(ptr.add(i));
        }
        for i in old_len..len {
            Av2Restoration::write_reset(ptr.add(i));
        }
        v.set_len(len);
    }
    Ok(())
}

#[inline]
fn resize_zero_new_restoration_mask(v: &mut Vec<Av2Restoration>, len: usize) -> Result<(), ()> {
    if len < v.len() {
        v.truncate(len);
    }
    if len > v.capacity() {
        v.try_reserve_exact(len - v.len()).map_err(|_| ())?;
    }

    let old_len = v.len();
    let ptr = v.as_mut_ptr();
    unsafe {
        for i in old_len..len {
            Av2Restoration::write_reset(ptr.add(i));
        }
        v.set_len(len);
    }
    Ok(())
}

pub(crate) fn decode_frame_init(
    frame_hdr: &FrameHeader,
    seq_hdr: &crate::headers::SequenceHeader,
    lf: &mut LoopFilterState,
    ts: &mut Vec<crate::internal::TileState>,
    n_ts: &mut i32,
    a: &mut Vec<BlockContext>,
    a_sz: &mut i32,
    dq: &mut [[[u32; 2]; 3]; MAX_SEGMENTS],
    qm: &mut [[Option<Vec<u8>>; 3]; crate::levels::N_RECT_TX_SIZES],
    absrefdist: &[u8; 7],
    sbh: i32,
    sb256w: i32,
    sb256h: i32,
    _bw: i32,
    _bh: i32,
    n_tc: i32,
) -> Result<(), ()> {
    init_start_of_tile_row(
        &mut lf.start_of_tile_row,
        sbh,
        frame_hdr.tiling.t.rows,
        frame_hdr.tiling.t.row_start_sb.as_ref(),
    )?;

    let new_n_ts = frame_hdr.tiling.t.cols as i32 * frame_hdr.tiling.t.rows as i32;
    if new_n_ts != *n_ts {
        *n_ts = new_n_ts;
    }
    try_resize_tile_states(ts, new_n_ts as usize)?;

    let keyframe = frame_hdr.is_key_or_intra();
    let is_tip = frame_hdr.tip.frame_mode == 2;
    let reset_a_contexts = n_tc > 1;

    let new_a_sz = sb256w * frame_hdr.tiling.t.rows as i32;
    if new_a_sz != *a_sz || reset_a_contexts {
        try_resize_block_contexts(a, new_a_sz as usize, keyframe, is_tip, reset_a_contexts)?;
        *a_sz = new_a_sz;
    }

    let num_sb256 = (sb256w * sb256h) as usize;
    lf.restore_planes = compute_restore_planes(frame_hdr);

    // These masks are always reset before decode reads them.  Avoid
    // `resize_with(Default::default)` here: on grow it first builds large default
    // Av2Filter/Av2Restoration values and then the old code immediately wrote a
    // second Default over every element.  The helpers write directly into spare
    // capacity with the mask reset bit-pattern, then expose the length.
    resize_and_reset_filter_mask(&mut lf.mask, num_sb256)?;
    // The LR mask (~num_sb256 * 24 KiB) is consumed only by loop restoration.  If
    // no plane is restored, existing elements may stay stale because no later
    // code reads them, but newly exposed Vec slots must still be initialized
    // before the Vec length is increased.  LR-enabled frames reset every slot.
    if lf.restore_planes != 0 {
        resize_and_reset_restoration_mask(&mut lf.lr_mask, num_sb256)?;
    } else {
        resize_zero_new_restoration_mask(&mut lf.lr_mask, num_sb256)?;
    }

    init_wiener(frame_hdr, lf);

    // Allocated only when segmentation is on and chroma deblock is enabled.
    if frame_hdr.segmentation.enabled != 0
        && seq_hdr.layout != crate::headers::PixelLayout::I400
        && (frame_hdr.deblock.level_u != 0 || frame_hdr.deblock.level_v != 0)
    {
        let ss_hor = (seq_hdr.layout != crate::headers::PixelLayout::I444) as i32;
        let ss_ver = (seq_hdr.layout == crate::headers::PixelLayout::I420) as i32;
        let stride = (sb256w * (64 >> ss_hor)) as isize;
        let size = stride as usize * sb256h as usize * (64 >> ss_ver) as usize;
        lf.uv_segmap_stride = stride;
        {
            let seg = std::sync::Arc::make_mut(&mut lf.segmap_uv);
            if seg.len() != size {
                seg.try_reserve_exact(size.saturating_sub(seg.len()))
                    .map_err(|_| ())?;
                seg.resize(size, 0);
            }
            seg.fill(0);
        }
    } else {
        lf.uv_segmap_stride = 0;
        std::sync::Arc::make_mut(&mut lf.segmap_uv).clear();
    }

    if frame_hdr.gdf.enabled != AdaptiveBoolean::Off {
        lf.gdf_ref_dst_idx = compute_gdf_ref_dst_idx(frame_hdr, absrefdist);
    }

    let re_sz = sb256h * frame_hdr.tiling.t.cols as i32;
    lf.re_sz = re_sz;

    let qmax = 255 + 48 * seq_hdr.hbd as i32;
    init_quant_tables(frame_hdr, frame_hdr.quant.yac as i32, dq, qmax);

    if frame_hdr.quant.qm.enabled == 0 {
        *qm = Default::default();
    }

    Ok(())
}

pub(crate) fn setup_tile_bounds(
    ts: &mut crate::internal::TileState,
    tile_row: i32,
    tile_col: i32,
    col_start_sb: &[u16],
    row_start_sb: &[u16],
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
) {
    let col_sb_start = col_start_sb[tile_col as usize] as i32;
    let col_sb_end = col_start_sb[tile_col as usize + 1] as i32;
    let row_sb_start = row_start_sb[tile_row as usize] as i32;
    let row_sb_end = row_start_sb[tile_row as usize + 1] as i32;

    ts.tiling.row = tile_row;
    ts.tiling.col = tile_col;
    ts.tiling.col_start = col_sb_start << sb_shift;
    ts.tiling.col_end = imin(col_sb_end << sb_shift, bw);
    ts.tiling.row_start = row_sb_start << sb_shift;
    ts.tiling.row_end = imin(row_sb_end << sb_shift, bh);

    if n_tc > 1 {
        for p in 0..3 {
            ts.progress[p].store(row_sb_start, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn setup_tile_wiener_banks(ts: &mut crate::internal::TileState, frame_hdr: &FrameHeader) {
    for pl in 0..3 {
        let rtype = frame_hdr.restoration.p[pl].restoration_type;
        if rtype == RestorationType::NsWiener as u8 || rtype == RestorationType::Switchable as u8 {
            let n_classes = frame_hdr.restoration.p[pl].ns.num_classes as usize;
            init_ns_wiener_bank(&mut ts.ns_wiener_bank[pl], pl, n_classes);
        }
    }
}

pub(crate) fn setup_tile(
    ts: &mut crate::internal::TileState,
    data: &[u8],
    frame_hdr: &FrameHeader,
    in_cdf: Option<&crate::cdf::CdfContext>,
    qcat: usize,
    tile_row: i32,
    tile_col: i32,
    col_start_sb: &[u16],
    row_start_sb: &[u16],
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
    tile_start_off: u32,
) -> Result<(), ()> {
    if let Some(cdf) = in_cdf {
        ts.cdf = cdf.clone();
    } else {
        ts.cdf = crate::cdf::CdfContext::init_from_defaults(qcat);
    }
    ts.last_qidx = frame_hdr.quant.yac as i32;
    ts.msac_buf.clear();
    ts.msac_buf.try_reserve_exact(data.len()).map_err(|_| ())?;
    ts.msac_buf.extend_from_slice(data);
    // Seed the resumable entropy state so every sbrow — including the first —
    // restores uniformly from `ts.msac_state` (sbrow-granularity scheduling).
    // Without the `adaptive_cdf` feature, the UPDATE_CDF policy is carried in
    // the saved MSAC state so the decode body can stay single-monomorphized.
    ts.msac_state = MsacContextScalar::<true>::new_with_update_cdf(
        &ts.msac_buf,
        frame_hdr.disable_cdf_update == 0,
    )
    .save();
    ts.tile_start_off = tile_start_off;

    setup_tile_bounds(
        ts,
        tile_row,
        tile_col,
        col_start_sb,
        row_start_sb,
        sb_shift,
        bw,
        bh,
        n_tc,
    );
    setup_tile_wiener_banks(ts, frame_hdr);
    Ok(())
}

pub(crate) fn decode_frame_init_cdf(
    ts: &mut [crate::internal::TileState],
    tile_groups: &[crate::internal::TileGroup],
    frame_hdr: &FrameHeader,
    in_cdf: Option<&crate::cdf::CdfContext>,
    qcat: usize,
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
    pool: Option<&crate::mtpool::ThreadPool>,
) -> Result<(), ()> {
    let ti = &frame_hdr.tiling.t;

    // Phase 1 (serial): parse the tile-group structure into per-tile descriptors.
    // This is only offset arithmetic + slicing; the expensive per-tile work — the
    // 17 KB `CdfContext` clone and the tile-bitstream copy inside `setup_tile` —
    // is deferred to the parallel phase so it no longer sits, 64-clones-deep, on
    // the serial critical path ahead of the parallel decode.
    struct TileInit<'a> {
        j: usize,
        data: &'a [u8],
        tile_row: i32,
        tile_col: i32,
        start_off: u32,
    }
    let mut inits: Vec<TileInit> = Vec::new();
    inits.try_reserve_exact(ts.len()).map_err(|_| ())?;
    let mut tile_row = 0i32;
    let mut tile_col = 0i32;
    for tg in tile_groups.iter() {
        let mut data_off = 0usize;
        let mut remaining = tg.data.len();

        for j in tg.start..=tg.end {
            let tile_sz;
            if j == tg.end {
                tile_sz = remaining;
            } else {
                let n_bytes = frame_hdr.tiling.n_bytes as usize;
                if n_bytes > remaining {
                    return Err(());
                }
                let mut sz = 0usize;
                for k in 0..n_bytes {
                    sz |= (tg.data[data_off + k] as usize) << (k * 8);
                }
                sz += 1;
                data_off += n_bytes;
                remaining -= n_bytes;
                if sz > remaining {
                    return Err(());
                }
                tile_sz = sz;
            }

            let tile_data = &tg.data[data_off..data_off + tile_sz];
            let start_off = 0u32;

            inits.push(TileInit {
                j: j as usize,
                data: tile_data,
                tile_row,
                tile_col,
                start_off,
            });

            tile_col += 1;
            if tile_col == ti.cols as i32 {
                tile_col = 0;
                tile_row += 1;
            }

            data_off += tile_sz;
            remaining -= tile_sz;
        }
    }

    // Phase 2 (parallel): run the per-tile setup across the pool. Each unit owns a
    // distinct tile index `j`, so the `ts` writes are disjoint; `setup_tile` reads
    // only shared, read-only frame state otherwise.
    let n_units = inits.len();
    if n_units == 0 {
        return Ok(());
    }
    let active = pool;
    let col_start_sb = ti.col_start_sb.as_ref();
    let row_start_sb = ti.row_start_sb.as_ref();
    let ts_dm = DisjointMut::new(ts);
    let ts_dm = &ts_dm;
    let inits = &inits[..];
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let cursor = &cursor;
    let allocation_failed = std::sync::atomic::AtomicBool::new(false);
    let allocation_failed = &allocation_failed;
    let n_workers = (n_tc as usize).min(n_units).max(1);
    let job = || {
        loop {
            let u = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if u >= n_units {
                break;
            }
            let it = &inits[u];
            // SAFETY: each unit has a distinct tile index `it.j`, so this is the
            // only access to `ts[it.j]` (disjoint from every other worker).
            let ts_slice = unsafe { ts_dm.whole() };
            if setup_tile(
                &mut ts_slice[it.j],
                it.data,
                frame_hdr,
                in_cdf,
                qcat,
                it.tile_row,
                it.tile_col,
                col_start_sb,
                row_start_sb,
                sb_shift,
                bw,
                bh,
                n_tc,
                it.start_off,
            )
            .is_err()
            {
                allocation_failed.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    };
    crate::mtpool::dispatch(active, n_workers, &job);
    if allocation_failed.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(());
    }

    Ok(())
}

pub(crate) fn decode_tip_frame_init(
    ts: &mut [crate::internal::TileState],
    frame_hdr: &FrameHeader,
    sb_shift: i32,
    bw: i32,
    bh: i32,
    n_tc: i32,
) {
    let ti = &frame_hdr.tiling.t;
    let mut tile = 0usize;
    for tile_row in 0..ti.rows as i32 {
        for tile_col in 0..ti.cols as i32 {
            setup_tile_bounds(
                &mut ts[tile],
                tile_row,
                tile_col,
                ti.col_start_sb.as_ref(),
                ti.row_start_sb.as_ref(),
                sb_shift,
                bw,
                bh,
                n_tc,
            );
            tile += 1;
        }
    }
}
