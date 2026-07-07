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

use core::mem::MaybeUninit;

#[derive(Clone, Default)]
pub struct Av2RestorationUnit {
    pub restoration_type: u8,
    pub ns_filter: [[i8; 32]; 16],
}

#[derive(Clone)]
pub struct Av2Filter {
    pub filter_y: [[[[u16; 4]; 5]; 64]; 2],
    pub filter_uv: [[[[u16; 4]; 5]; 64]; 2],
    pub qidx: [u16; 16],
    pub gdf: [u8; 16],
    pub cdef_idx: [i8; 16],
    pub ccso: [u8; 3],
    pub ccso_sb: [u8; 48],
    pub noskip_mask: [[u16; 4]; 32],
    pub lr_noskip_mask: [[u16; 4]; 64],
    pub lossless_mask_y: [[u16; 4]; 64],
    pub lossless_mask_uv: [[u16; 4]; 64],
}

impl Av2Filter {
    #[inline(always)]
    pub(crate) unsafe fn write_reset(dst: *mut Self) {
        // SAFETY: every field of Av2Filter is an integer array, so all-zero is a
        // valid bit pattern for the whole object.  After the zero fill the object
        // is fully initialized; then patch the only non-zero default field.
        unsafe {
            core::ptr::write_bytes(dst, 0, 1);
            core::ptr::addr_of_mut!((*dst).cdef_idx).write([-1; 16]);
        }
    }
}

impl Default for Av2Filter {
    fn default() -> Self {
        let mut out = MaybeUninit::<Self>::uninit();
        unsafe {
            Self::write_reset(out.as_mut_ptr());
            out.assume_init()
        }
    }
}

#[derive(Clone)]
pub struct Av2Restoration {
    pub lr: [[Av2RestorationUnit; 16]; 3],
}

impl Av2Restoration {
    #[inline(always)]
    pub(crate) unsafe fn write_reset(dst: *mut Self) {
        // SAFETY: Av2Restoration is integer-only storage and its default state is
        // all-zero, so a byte zero-fill creates a valid initialized value.
        unsafe { core::ptr::write_bytes(dst, 0, 1) };
    }
}

impl Default for Av2Restoration {
    fn default() -> Self {
        let mut out = MaybeUninit::<Self>::uninit();
        unsafe {
            Self::write_reset(out.as_mut_ptr());
            out.assume_init()
        }
    }
}

use std::sync::OnceLock;

use crate::intops::{iclip, imax, imin};
use crate::levels::TxPartition;
use crate::quantizer::dq_lookup;
use crate::tables::{DEBLOCK_SIDE_THRESHOLDS, TxfmInfo};

pub type FilterMasks = [[[u16; 4]; 5]; 64];

pub fn mask_outer_edge_l(masks: &mut [[u16; 4]], by4: i32, h4: i32, bwl4c: u8, l: &mut [u8]) {
    debug_assert!((bwl4c as u32) <= 3);
    let mut mask: u64 = 1 << by4;
    for (y, &l) in l[..h4 as usize].iter().enumerate() {
        let sidx = ((by4 as usize) + y) >> 4;
        let smask = (mask >> (sidx << 4)) as u16;
        let lvl = imin(bwl4c as i32, l as i32) as usize;
        masks[lvl][sidx] |= smask;
        mask <<= 1;
    }
    for dst in l[..h4 as usize].iter_mut() {
        *dst = bwl4c;
    }
}

pub fn mask_outer_edge_t(masks: &mut [[u16; 4]], bx4: i32, w4: i32, bhl4c: u8, a: &mut [u8]) {
    debug_assert!((bhl4c as u32) <= 3);
    let mut mask: u64 = 1 << bx4;
    for (x, &a) in a[..w4 as usize].iter().enumerate() {
        let sidx = ((bx4 as usize) + x) >> 4;
        let smask = (mask >> (sidx << 4)) as u16;
        let lvl = imin(bhl4c as i32, a as i32) as usize;
        masks[lvl][sidx] |= smask;
        mask <<= 1;
    }
    for dst in a[..w4 as usize].iter_mut() {
        *dst = bhl4c;
    }
}

pub fn mask_inner_edges_v(
    masks: &mut [FilterMasks; 2],
    inner: u64,
    bx4: i32,
    w4: i32,
    twl4c: i32,
    xoff: i32,
    hstep: i32,
) {
    debug_assert!((twl4c as u32) <= 3);
    let inner1 = (inner & 0xffff) as u16;
    let inner2 = ((inner >> 16) & 0xffff) as u16;
    let inner3 = ((inner >> 32) & 0xffff) as u16;
    let inner4 = (inner >> 48) as u16;
    let t = twl4c as usize;
    let mut x = xoff;
    while x < w4 {
        let idx = (bx4 + x) as usize;
        if inner1 != 0 {
            masks[0][idx][t][0] |= inner1;
        }
        if inner2 != 0 {
            masks[0][idx][t][1] |= inner2;
        }
        if inner3 != 0 {
            masks[0][idx][t][2] |= inner3;
        }
        if inner4 != 0 {
            masks[0][idx][t][3] |= inner4;
        }
        x += hstep;
    }
}

pub fn mask_inner_edges_h(
    masks: &mut [FilterMasks; 2],
    inner: u64,
    by4: i32,
    h4: i32,
    thl4c: i32,
    yoff: i32,
    vstep: i32,
) {
    debug_assert!((thl4c as u32) <= 3);
    let inner1 = (inner & 0xffff) as u16;
    let inner2 = ((inner >> 16) & 0xffff) as u16;
    let inner3 = ((inner >> 32) & 0xffff) as u16;
    let inner4 = (inner >> 48) as u16;
    let t = thl4c as usize;
    let mut y = yoff;
    while y < h4 {
        let idx = (by4 + y) as usize;
        if inner1 != 0 {
            masks[1][idx][t][0] |= inner1;
        }
        if inner2 != 0 {
            masks[1][idx][t][1] |= inner2;
        }
        if inner3 != 0 {
            masks[1][idx][t][2] |= inner3;
        }
        if inner4 != 0 {
            masks[1][idx][t][3] |= inner4;
        }
        y += vstep;
    }
}

pub fn mask_edges_part(
    masks: &mut [FilterMasks; 2],
    by4: i32,
    bx4: i32,
    w4: i32,
    h4: i32,
    tx_part: TxPartition,
    t_dim: &TxfmInfo,
    hlim: i32,
    vlim: i32,
    a: &mut [u8],
    l: &mut [u8],
) {
    let tw4 = t_dim.w as i32;
    let th4 = t_dim.h as i32;
    let twl4c = imin(hlim, t_dim.lw as i32);
    let thl4c = imin(vlim, t_dim.lh as i32);

    if (tx_part as u8) < (TxPartition::H5 as u8) {
        mask_outer_edge_l(&mut masks[0][bx4 as usize], by4, h4, twl4c as u8, l);
        mask_outer_edge_t(&mut masks[1][by4 as usize], bx4, w4, thl4c as u8, a);
        if w4 > tw4 {
            let inner = (!0u64 >> (64 - h4)) << by4;
            mask_inner_edges_v(masks, inner, bx4, w4, twl4c, tw4, tw4);
        }
        if h4 > th4 {
            let inner = (!0u64 >> (64 - w4)) << bx4;
            mask_inner_edges_h(masks, inner, by4, h4, thl4c, th4, th4);
        }
    } else if tx_part as u8 == TxPartition::H5 as u8 {
        debug_assert!(th4 * 4 >= h4 && tw4 * 2 >= w4);
        mask_outer_edge_t(&mut masks[1][by4 as usize], bx4, w4, thl4c as u8, a);
        mask_outer_edge_l(
            &mut masks[0][bx4 as usize],
            by4,
            imin(th4, h4),
            twl4c as u8,
            l,
        );
        if h4 > th4 {
            mask_outer_edge_l(
                &mut masks[0][bx4 as usize],
                by4 + th4,
                imin(2 * th4, h4 - th4),
                imin(twl4c + 1, hlim) as u8,
                &mut l[th4 as usize..],
            );
            if h4 > th4 * 3 {
                mask_outer_edge_l(
                    &mut masks[0][bx4 as usize],
                    by4 + th4 * 3,
                    imin(th4, h4 - 3 * th4),
                    twl4c as u8,
                    &mut l[th4 as usize * 3..],
                );
            }
        }
        let inner = (!0u64 >> (64 - w4)) << bx4;
        mask_inner_edges_h(masks, inner, by4, h4, thl4c, th4, th4 * 2);
        let inner_a = (!0u64 >> (64 - h4)) << by4;
        let inner_b = (!0u64 >> (64 - th4 * 2)) << (by4 + th4);
        let inner_c = inner_a & !inner_b;
        mask_inner_edges_v(masks, inner_c, bx4, w4, twl4c, tw4, tw4);
    } else {
        debug_assert!(tx_part as u8 == TxPartition::V5 as u8 && tw4 * 4 >= w4 && th4 * 2 >= h4);
        mask_outer_edge_l(&mut masks[0][bx4 as usize], by4, h4, twl4c as u8, l);
        mask_outer_edge_t(
            &mut masks[1][by4 as usize],
            bx4,
            imin(tw4, w4),
            thl4c as u8,
            a,
        );
        if w4 > tw4 {
            mask_outer_edge_t(
                &mut masks[1][by4 as usize],
                bx4 + tw4,
                imin(2 * tw4, w4 - tw4),
                imin(thl4c + 1, vlim) as u8,
                &mut a[tw4 as usize..],
            );
            if w4 > tw4 * 3 {
                mask_outer_edge_t(
                    &mut masks[1][by4 as usize],
                    bx4 + tw4 * 3,
                    imin(tw4, w4 - 3 * tw4),
                    thl4c as u8,
                    &mut a[tw4 as usize * 3..],
                );
            }
        }
        let inner = (!0u64 >> (64 - h4)) << by4;
        mask_inner_edges_v(masks, inner, bx4, w4, twl4c, tw4, tw4 * 2);
        let inner_a = (!0u64 >> (64 - w4)) << bx4;
        let inner_b = (!0u64 >> (64 - tw4 * 2)) << (bx4 + tw4);
        let inner_c = inner_a & !inner_b;
        mask_inner_edges_h(masks, inner_c, by4, h4, thl4c, th4, th4);
    }
}

pub fn mask_subpu_edges(
    masks: &mut [FilterMasks; 2],
    by4: i32,
    bx4: i32,
    w4: i32,
    h4: i32,
    twl4c: i32,
    thl4c: i32,
    hsz: i32,
    vsz: i32,
    ds_sub_pu_mask: i32,
) {
    debug_assert!(hsz & (hsz - 1) == 0 && (0..=8).contains(&hsz));
    debug_assert!(vsz & (vsz - 1) == 0 && (0..=8).contains(&vsz));
    debug_assert!((thl4c as u32) <= 2 && (twl4c as u32) <= 2);
    debug_assert!(ds_sub_pu_mask == 15 || ds_sub_pu_mask == 0);

    if hsz != 0 {
        let inner = (!0u64 >> (64 - h4)) << by4;
        let inner0 = (inner & 0xffff) as u16;
        let inner1 = ((inner >> 16) & 0xffff) as u16;
        let inner2 = ((inner >> 32) & 0xffff) as u16;
        let inner3 = (inner >> 48) as u16;
        let mut x = hsz;
        while x < w4 {
            let idx = (bx4 + x) as usize;
            let t = twl4c as usize;
            macro_rules! mask_subpu_v {
                ($e:expr, $iv:expr) => {
                    if $iv != 0 {
                        let m = masks[0][idx][t][$e];
                        masks[0][idx][t][$e] |= $iv;
                        if (x & ds_sub_pu_mask) != 0 {
                            masks[0][idx][4][$e] |= $iv & !m;
                        }
                    }
                };
            }
            mask_subpu_v!(0, inner0);
            mask_subpu_v!(1, inner1);
            mask_subpu_v!(2, inner2);
            mask_subpu_v!(3, inner3);
            x += hsz;
        }
    }

    if vsz != 0 {
        let inner = (!0u64 >> (64 - w4)) << bx4;
        let inner0 = (inner & 0xffff) as u16;
        let inner1 = ((inner >> 16) & 0xffff) as u16;
        let inner2 = ((inner >> 32) & 0xffff) as u16;
        let inner3 = (inner >> 48) as u16;
        let mut y = vsz;
        while y < h4 {
            let idx = (by4 + y) as usize;
            let t = thl4c as usize;
            macro_rules! mask_subpu_h {
                ($e:expr, $iv:expr) => {
                    if $iv != 0 {
                        let m = masks[1][idx][t][$e];
                        masks[1][idx][t][$e] |= $iv;
                        if (y & ds_sub_pu_mask) != 0 {
                            masks[1][idx][4][$e] |= $iv & !m;
                        }
                    }
                };
            }
            mask_subpu_h!(0, inner0);
            mask_subpu_h!(1, inner1);
            mask_subpu_h!(2, inner2);
            mask_subpu_h!(3, inner3);
            y += vsz;
        }
    }
}

const DEBLOCK_THR_CACHE_QMAX: usize = 343;
const DEBLOCK_THR_CACHE_LEN: usize = DEBLOCK_THR_CACHE_QMAX + 1;

pub(crate) type DeblockThrCache = [[[u32; 2]; DEBLOCK_THR_CACHE_LEN]; 2];

static DEBLOCK_THR_CACHE: OnceLock<DeblockThrCache> = OnceLock::new();

#[inline(always)]
fn deblock_quant_thr_uncached(hbd: i32, qidx: i32) -> u32 {
    let qmax = 255 + 48 * hbd;
    ((dq_lookup(iclip(qidx, 0, qmax)) + 4) >> (3 + 6)) as u32
}

#[inline(always)]
fn deblock_side_thr_uncached(hbd: i32, qidx: i32) -> u32 {
    let bitdepth_min_8 = 2 * hbd;
    let q_ind = iclip(qidx - 24 * bitdepth_min_8, 0, 296 - 1);
    let side_thr = DEBLOCK_SIDE_THRESHOLDS[q_ind as usize] as i32;
    imax(side_thr + (1 << 4 >> bitdepth_min_8), 0) as u32 >> (5 - bitdepth_min_8) as u32
}

fn init_deblock_thr_cache() -> DeblockThrCache {
    let mut cache = [[[0u32; 2]; DEBLOCK_THR_CACHE_LEN]; 2];
    for hbd in 0..2 {
        for qidx in 0..DEBLOCK_THR_CACHE_LEN {
            let hbd = hbd as i32;
            let qidx = qidx as i32;
            cache[hbd as usize][qidx as usize][0] = deblock_quant_thr_uncached(hbd, qidx);
            cache[hbd as usize][qidx as usize][1] = deblock_side_thr_uncached(hbd, qidx);
        }
    }
    cache
}

#[inline(always)]
pub(crate) fn deblock_thr_cache() -> &'static DeblockThrCache {
    DEBLOCK_THR_CACHE.get_or_init(init_deblock_thr_cache)
}

#[inline(always)]
pub(crate) fn deblock_thr_from_cache(cache: &DeblockThrCache, hbd: i32, qidx: i32) -> [u32; 2] {
    let hbd = (hbd != 0) as usize;
    let qidx = iclip(qidx, 0, DEBLOCK_THR_CACHE_QMAX as i32) as usize;
    cache[hbd][qidx]
}

pub fn transpose_lossless_mask(
    dst_mask: &mut [u16; 17],
    src_mask: &[[u16; 4]],
    x64: usize,
    ss_hor: i32,
    ss_ver: i32,
) {
    dst_mask[0] = dst_mask[(16 >> ss_hor) as usize];

    for x in 0..(16 >> ss_hor) as usize {
        let mut col_mask: u32 = 0;
        for y in 0..(16 >> ss_ver) as usize {
            col_mask |= ((1 & (src_mask[y][x64] >> x)) as u32) << y;
        }
        dst_mask[x + 1] = col_mask as u16;
    }
}

use crate::headers::{FrameHeader, PixelLayout, SequenceHeader};
use crate::levels::{Av2Block, BlockSize, CompInterPredMode, CompInterType, TIP_FRAME, TxfmSize};
use crate::tables::{BLOCK_DIMENSIONS, MAX_TXFM_SIZE_FOR_BS, TX_PART_TBL, TXFM_DIMENSIONS};

const FILTER_8TAP_SHARP: u8 = 2;

fn subpu_flt_lvl(
    seq_hdr: &SequenceHeader,
    frame_hdr: &FrameHeader,
    bs: BlockSize,
    bw4: i32,
    bh4: i32,
    b: &Av2Block,
    max_lvl: i32,
) -> i32 {
    let r = b.ref_pair.refs();
    if b.is_intra != 0 || frame_hdr.deblock.sub_pu == 0 {
        // do nothing
    } else if r[0] == TIP_FRAME as i8 {
        let opfl = seq_hdr.tip_refine_mv
            && (frame_hdr.tip.frame_mode == 1 || frame_hdr.tip.subpel_filter == FILTER_8TAP_SHARP);
        return 1 + if frame_hdr.tip.frame_mode == 2 {
            !opfl as i32
        } else {
            ((!opfl && imin(bw4, bh4) >= 4) || bs == BlockSize::Bs256x256) as i32
        };
    } else if r[1] != -1 {
        let inter = *b.inter_data();
        if inter.inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8 {
            return 1 - (bs == BlockSize::Bs8x8) as i32;
        } else if inter.refine_mv != 0 && inter.comp_type == CompInterType::Avg as u8 {
            return 2;
        }
    }
    max_lvl
}

pub fn create_db_mask(
    masks: &mut [FilterMasks; 2],
    b: &Av2Block,
    bs: BlockSize,
    bx: i32,
    by: i32,
    iw: i32,
    ih: i32,
    layout: PixelLayout,
    chroma: bool,
    a: &mut [u8],
    l: &mut [u8],
    frame_hdr: &FrameHeader,
    seq_hdr: &SequenceHeader,
) {
    let ss_ver = (chroma && layout == PixelLayout::I420) as i32;
    let ss_hor = (chroma && layout != PixelLayout::I444) as i32;
    let b_dim = &BLOCK_DIMENSIONS[bs as usize];
    let bw4 = imin(iw - bx, b_dim[0] as i32) >> ss_hor;
    let bh4 = imin(ih - by, b_dim[1] as i32) >> ss_ver;
    let bx4 = (bx & 63) >> ss_hor;
    let by4 = (by & 63) >> ss_ver;
    assert!(bw4 > 0 && bh4 > 0);

    let subpu_l2 = subpu_flt_lvl(
        seq_hdr,
        frame_hdr,
        bs,
        b_dim[0] as i32,
        b_dim[1] as i32,
        b,
        3,
    );
    let ds_subpu_mask = (frame_hdr.tip.frame_mode != 2) as i32 * 15;
    let twl4c;
    let thl4c;

    let chroma_i = chroma as i32;
    let lossless = frame_hdr.segmentation.lossless[b.seg_id as usize];
    if b.is_intra != 0 || b.skip_txfm == 0 {
        let tx_part = if chroma {
            TxPartition::None
        } else {
            TxPartition::from_raw(b.tx_part)
        };
        let tx: usize = if lossless != 0 {
            if !chroma && b.tx_size_ll != 0 {
                MAX_TXFM_SIZE_FOR_BS[bs as usize][3] as usize
            } else {
                TxfmSize::Tx4x4 as usize
            }
        } else if chroma {
            MAX_TXFM_SIZE_FOR_BS[bs as usize][PixelLayout::I444 as usize - layout as usize] as usize
        } else {
            TX_PART_TBL[bs as usize][tx_part as usize] as usize
        };
        let t_dim = &TXFM_DIMENSIONS[tx];
        mask_edges_part(
            masks,
            by4,
            bx4,
            bw4,
            bh4,
            tx_part,
            t_dim,
            iclip(subpu_l2 - ss_hor, 0, 3 - chroma_i),
            iclip(subpu_l2 - ss_ver, 0, 3 - chroma_i),
            a,
            l,
        );
        twl4c = imin(subpu_l2, t_dim.lw as i32);
        thl4c = imin(subpu_l2, t_dim.lh as i32);
    } else {
        let (hlim, vlim) = if lossless != 0 {
            (0u8, 0u8)
        } else {
            (
                iclip(imin(subpu_l2, b_dim[2] as i32) - ss_hor, 0, 3 - chroma_i) as u8,
                iclip(imin(subpu_l2, b_dim[3] as i32) - ss_ver, 0, 3 - chroma_i) as u8,
            )
        };
        mask_outer_edge_l(&mut masks[0][bx4 as usize], by4, bh4, hlim, l);
        mask_outer_edge_t(&mut masks[1][by4 as usize], bx4, bw4, vlim, a);
        twl4c = subpu_l2;
        thl4c = subpu_l2;
    }

    if subpu_l2 != 3 {
        let h_subpu_l2 = twl4c - (ss_hor != 0 && twl4c != 0) as i32;
        let v_subpu_l2 = thl4c - (ss_ver != 0 && thl4c != 0) as i32;
        mask_subpu_edges(
            masks,
            by4,
            bx4,
            bw4,
            bh4,
            h_subpu_l2,
            v_subpu_l2,
            (1 << subpu_l2) >> ss_hor,
            (1 << subpu_l2) >> ss_ver,
            ds_subpu_mask,
        );
    }
}
