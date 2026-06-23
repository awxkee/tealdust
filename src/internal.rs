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

use std::sync::Arc;
use std::sync::atomic::AtomicI32;

use crate::cdf::CdfContext;
use crate::env::BlockContext;
use crate::headers::{
    ContentInterpretation, ContentLightLevel, FilmGrainData, FrameHeader, MAX_SEGMENTS,
    MasteringDisplay, SequenceHeader,
};
use crate::levels::{BlockSize, N_RECT_TX_SIZES, RefPair};
use crate::lf_mask::{Av2Filter, Av2Restoration};
use crate::refmvs;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Pass {
    Entropy = 1,
    MvRes = 2,
    Recon = 4,
}

pub(crate) const PASS_ALL: u8 = Pass::Entropy as u8 | Pass::MvRes as u8 | Pass::Recon as u8;

#[derive(Clone, Copy, Default)]
pub(crate) struct ScalableMotionParams {
    pub(crate) scale: i32,
    pub(crate) step: i32,
}

#[derive(Default)]
pub(crate) struct NsWienerBank {
    pub(crate) bank_size: [u8; 16],
    pub(crate) bank_idx: [u8; 16],
    pub(crate) filter: [[[i8; 32]; 16]; 4],
}

pub(crate) struct TileState {
    pub(crate) cdf: CdfContext,
    pub(crate) msac_buf: Vec<u8>,
    /// Parked entropy-decoder position between superblock rows (sbrow-granularity
    /// scheduling): set at tile setup, then resumed/re-saved around each sbrow.
    pub(crate) msac_state: crate::msac::MsacState,

    pub(crate) tiling: TileBounds,

    pub(crate) progress: [AtomicI32; 3],

    pub(crate) dqmem: [[[u32; 2]; 3]; MAX_SEGMENTS],
    pub(crate) last_qidx: i32,

    pub(crate) ns_wiener_bank: [NsWienerBank; 3],

    pub(crate) tile_start_off: u32,
}

impl Default for TileState {
    fn default() -> Self {
        Self {
            cdf: Default::default(),
            msac_buf: Vec::new(),
            msac_state: Default::default(),
            tiling: Default::default(),
            progress: [AtomicI32::new(0), AtomicI32::new(0), AtomicI32::new(0)],
            dqmem: [[[0; 2]; 3]; MAX_SEGMENTS],
            last_qidx: 0,
            ns_wiener_bank: Default::default(),
            tile_start_off: 0,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct TileBounds {
    pub(crate) col_start: i32,
    pub(crate) col_end: i32,
    pub(crate) row_start: i32,
    pub(crate) row_end: i32,
    pub(crate) col: i32,
    pub(crate) row: i32,
}

#[derive(Default)]
pub(crate) struct LoopFilterState {
    pub(crate) mask: Vec<Av2Filter>,
    pub(crate) lr_mask: Vec<Av2Restoration>,
    pub(crate) segmap_uv: Arc<Vec<u8>>,
    pub(crate) uv_segmap_stride: isize,
    pub(crate) re_sz: i32,
    pub(crate) base_q: i32,
    pub(crate) gdf_ref_dst_idx: i32,
    pub(crate) start_of_tile_row: Vec<u8>,
    pub(crate) restore_planes: i32,
    pub(crate) wiener_idx: usize,
    pub(crate) ns_subclass_class_idx: Option<usize>,
    pub(crate) lr_cdef_line: [Vec<u8>; 3],
    pub(crate) lr_cdef_line_hbd: [Vec<u16>; 3],

    /// Reusable in-loop filter seam buffers.  These used to be rebuilt with
    /// hundreds/thousands of small heap allocations every frame in the MT path;
    /// keeping them on the frame context lets steady-state decode resize only
    /// when the geometry or chroma layout grows/changes.
    pub(crate) cdef_line_store: Vec<[Vec<u8>; 3]>,
    pub(crate) cdef_top_store: Vec<[Vec<u8>; 3]>,
    pub(crate) lr_db_store: Vec<[Vec<u8>; 3]>,
    pub(crate) cdef_line_store_hbd: Vec<[Vec<u16>; 3]>,
    pub(crate) cdef_top_store_hbd: Vec<[Vec<u16>; 3]>,
    pub(crate) lr_db_store_hbd: Vec<[Vec<u16>; 3]>,

    /// Reusable CCSO luma-padding scratch. `ccso_prep()` grows this only when a
    /// larger CDEF/CCSO band requires it, avoiding a heap allocation per CCSO
    /// block in the serial filter path.
    pub(crate) ccso_tmp_buf: Vec<u8>,
    pub(crate) ccso_tmp_buf_hbd: Vec<u16>,
}

#[derive(Default)]
pub(crate) struct FrameContext {
    pub(crate) seq_hdr: Arc<SequenceHeader>,
    pub(crate) frame_hdr: Arc<FrameHeader>,

    pub(crate) refp: [ThreadPicture; 7],

    pub(crate) mvs: Vec<refmvs::TemporalBlock>,
    pub(crate) ref_mvs: [Option<Vec<refmvs::TemporalBlock>>; 7],

    pub(crate) cur_segmap: Vec<u8>,
    pub(crate) prev_segmap: Option<Vec<u8>>,

    pub(crate) cur_ccsomap: Vec<u8>,
    pub(crate) prev_ccsomap: [Option<Vec<u8>>; 3],

    pub(crate) refpoc: [u8; 7],
    pub(crate) refrefpoc: [[u8; 7]; 7],
    pub(crate) refcnt: [u8; 7],
    pub(crate) refdir: [u8; 8],
    pub(crate) refdir_intra: i8,
    pub(crate) furthest_future_refidx: i8,
    pub(crate) absrefdist: [u8; 7],
    pub(crate) refdist: [i8; 7],
    pub(crate) skip_mode_refs: RefPair,
    pub(crate) gmv_warp_allowed: [u8; 7],
    pub(crate) use_pri_sec_cdf: i32,

    pub(crate) tile: Vec<TileGroup>,
    pub(crate) n_tile_data: i32,

    pub(crate) svc: [[ScalableMotionParams; 2]; 7],

    pub(crate) ts: Vec<TileState>,
    pub(crate) n_ts: i32,

    pub(crate) b4_stride: isize,
    pub(crate) bw: i32,
    pub(crate) bh: i32,
    pub(crate) sb256w: i32,
    pub(crate) sb256h: i32,
    pub(crate) sbh: i32,
    pub(crate) sb_shift: i32,
    pub(crate) sb_step: i32,
    pub(crate) ss_ver: i32,
    pub(crate) ss_hor: i32,

    pub(crate) dq: [[[u32; 2]; 3]; MAX_SEGMENTS],
    pub(crate) qm: [[Option<Vec<u8>>; 3]; N_RECT_TX_SIZES],

    pub(crate) a: Vec<BlockContext>,
    pub(crate) a_sz: i32,
    pub(crate) rf: refmvs::Frame,
    pub(crate) bitdepth_max: i32,
    pub(crate) root_bs: BlockSize,

    pub(crate) lf: LoopFilterState,

    /// In-loop filter flag word (DAV2D_INLOOPFILTER_* bits) threaded from the
    /// decoder's `Settings.inloop_filters`. Defaults to 0 (filters off) so any
    /// path constructing a `FrameContext` directly keeps pre-filter behaviour;
    /// `submit_frame` sets it from the configured filters.
    pub(crate) inloop_filters: u32,

    /// Output picture buffer that reconstruction writes into (the pixel planes
    /// that `cur` only describes as metadata).
    pub(crate) cur_pic: crate::picture::Picture,

    /// the static qcat init (keyframe / primary_ref_frame == NONE).
    pub(crate) in_cdf: Option<CdfContext>,
    /// update tile, count reset. Stored into refreshed `c.refs[i].cdf`.
    pub(crate) out_cdf: Option<Arc<CdfContext>>,
}

#[derive(Clone, Default)]
pub(crate) struct ThreadPicture {
    pub(crate) showable: bool,
    pub(crate) frame_hdr: Option<Arc<FrameHeader>>,
    /// Shared decoded picture pixels. Set when this slot references a fully
    /// reconstructed frame (inter reference setup / ref-list update). `None` for
    pub(crate) pic: Option<Arc<crate::picture::Picture>>,
}

pub(crate) struct TileGroup {
    pub(crate) data: Vec<u8>,
    pub(crate) start: i32,
    pub(crate) end: i32,
}

pub(crate) struct DecoderContext {
    pub(crate) seq_hdr: Option<Arc<SequenceHeader>>,
    pub(crate) frame_hdr: Option<Arc<FrameHeader>>,

    pub(crate) tile: Vec<TileGroup>,
    pub(crate) n_tile_data: i32,
    pub(crate) n_tiles: i32,

    pub(crate) refs: [RefState; 8],

    pub(crate) content_light: Option<ContentLightLevel>,
    pub(crate) mastering_display: Option<MasteringDisplay>,
    pub(crate) ci: Option<ContentInterpretation>,
    pub(crate) fgm: [Option<FilmGrainData>; 8],

    pub(crate) apply_grain: bool,
    pub(crate) operating_point_idc: u32,
    pub(crate) max_spatial_id: i32,
    pub(crate) frame_size_limit: u32,
    pub(crate) strict_std_compliance: bool,

    /// In-loop filter flag word (DAV2D_INLOOPFILTER_* bits) from the configured
    /// `Settings.inloop_filters`. Threaded onto each `FrameContext` in
    /// `submit_frame` so the per-superblock-row filter pass can gate each stage.
    pub(crate) inloop_filters: u32,

    /// Bring-up gate: run the single-threaded frame decode from `parse_obus`.
    /// Off by default while reconstruction and the entropy-path bugs are being
    /// worked through; the orchestration runs end-to-end when enabled.
    pub(crate) run_decode: bool,

    /// Reconstructed pictures awaiting hand-off to the decoder's output queue,
    /// in decode order. `submit_frame` pushes each decoded frame here so that
    /// frames decoded within a single `parse_obus` call are not lost. (Minimal
    /// output path; visibility/POC display reordering lands with full queueing.)
    pub(crate) frame_out: Vec<crate::picture::Picture>,

    /// Worker-thread budget (`Settings.n_threads` resolved to `n_tc`). Used to
    /// parallelise the disjoint-output display passes (output-frame copy / film
    /// grain). `1` keeps every such pass on the byte-identical sequential path.
    pub(crate) n_tc: u32,

    /// Persistent worker pool, owned by this decoder and created lazily on the
    /// first multi-threaded frame (sized to `n_tc`). Tying it to the decoder —
    /// rather than a process-wide static — honours each decoder's own
    /// `n_threads` and frees the workers when the decoder is dropped, instead of
    /// leaking them for the lifetime of the process.
    pub(crate) pool: Option<crate::mtpool::ThreadPool>,

    /// Recycles picture plane buffers across frames so steady-state decoding of a
    /// constant-resolution stream performs no per-frame picture heap traffic.
    /// Shared (cloned `Arc`) into every picture this decoder allocates.
    pub(crate) pic_allocator: std::sync::Arc<dyn crate::picture::PicAllocator>,

    /// Per-frame working state, retained across frames so its scratch buffers
    /// (block context, tile states, loop-filter masks, segment maps) keep their
    /// allocations and `decode_frame_init`'s size-based `resize` becomes a no-op
    /// for a constant-resolution stream — instead of rebuilding and tearing down
    /// the whole context every frame. Taken out by `submit_frame` for the decode
    /// and restored afterward.
    pub(crate) fc: FrameContext,
}

#[derive(Default)]
pub(crate) struct RefState {
    pub(crate) p: ThreadPicture,
    pub(crate) segmap: Option<Vec<u8>>,
    pub(crate) refmvs: Option<Vec<refmvs::TemporalBlock>>,
    pub(crate) ccsomap: Option<Vec<u8>>,
    pub(crate) refpoc: [u8; 7],
    /// `None` until a frame refreshes this slot. Cloned into the next frame's
    /// `in_cdf` when it selects this slot as `primary_ref_frame`.
    pub(crate) cdf: Option<Arc<CdfContext>>,
}
