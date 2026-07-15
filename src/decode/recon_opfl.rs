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
use crate::cdf::CdfModeContext;

use crate::env::BlockContext;

use crate::intops::{iclip, imax, imin};
use crate::levels::{
    Av2Block, BlockSize, InterPredMode, MotionMode, Mv, MvXY, RefPair, TIP_FRAME, TxPartition,
};

use crate::msac::MsacReader;
use crate::pixel::Pixel;

use crate::tables::{BLOCK_DIMENSIONS, TXFM_DIMENSIONS};

/// Rectangular bilinear prefetch into the OPFL `p[i]` scratch (the `mc(t, p[i],
/// Unlike the square TIP variant, this writes `dimw x dimh` luma pixels at 3-bit
/// subpel precision with explicit edge limits.
#[allow(clippy::too_many_arguments)]
fn prep_opfl_prefetch_rect_8bpc<BD: BitDepth>(
    bd: BD,
    exec: &crate::exec_context::ExecContext,
    p: &mut [BD::Pixel],
    p_stride: usize,
    ref_pic: &crate::picture::Picture,
    bx4: i32,
    by4: i32,
    dimw: i32,
    dimh: i32,
    mvx: i32,
    mvy: i32,
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let ref_stride = ref_pic.stride[0].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(0) {
        Some(s) => s,
        None => return,
    };
    let mx = mvx & 7;
    let my = mvy & 7;
    let dx = bx4 * 4 + (mvx >> 3);
    let dy = by4 * 4 + (mvy >> 3);

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + dimw + (mx != 0) as i32 * 4 > right
        || dy + dimh + (my != 0) as i32 * 4 > bottom;
    let dwu = dimw as usize;
    let dhu = dimh as usize;
    let mut emu_buf = if need_emu {
        Some(vec![BD::Pixel::default(); 192 * 192])
    } else {
        None
    };
    let (src, src_off, src_stride) = if let Some(ref mut buf) = emu_buf {
        let emu_w = dwu + (mx != 0) as usize * 7;
        let emu_h = dhu + (my != 0) as usize * 7;
        let emu_stride = 192usize;
        let src_base = ((left + top * ref_stride as i32).max(0) as usize).min(ref_data.len());
        inter_emu_edge_8bpc::<BD>(
            buf,
            emu_stride,
            &ref_data[src_base..],
            ref_stride,
            emu_w,
            emu_h,
            (right - left).max(0) as usize,
            (bottom - top).max(0) as usize,
            dx - (mx != 0) as i32 * 3 - left,
            dy - (my != 0) as i32 * 3 - top,
        );
        let off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
        (&buf[..], off, emu_stride)
    } else {
        let off = dy as usize * ref_stride + dx as usize;
        (ref_data, off, ref_stride)
    };
    if BD::BPC == 8 {
        let p8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(p);
        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
        exec.put_bilin_8bpc_with_scratch(
            p8,
            p_stride,
            &src8[src_off..],
            src_stride,
            dwu,
            dhu,
            mx << 1,
            my << 1,
            inter_scratch,
        );
    } else {
        crate::mc::put_bilin(
            bd,
            p,
            p_stride,
            &src[src_off..],
            src_stride,
            dwu,
            dhu,
            mx << 1,
            my << 1,
        );
    }
}

/// two-reference (same-pair) blocks that either signal an OPFL inter mode
/// (`inter_mode >= OPFL_NEARMV_NEARMV`) or request implicit MV refinement
/// (`refine_mv && comp_type == COMP_INTER_AVG`). Fills the two compound
/// predictors `tmp[0]`/`tmp[1]`, writes per-2x2 / per-bs temporal MVs into the
/// frame MV grid (`update_temporal`), stores the refined per-8x8 MVs in the
/// `recon.scratch.rmv` grid for chroma, and (for BACP) accumulates the
/// out-of-bounds blend mask. Returns the BACP predicate (mask vs average).
#[allow(clippy::too_many_arguments)]
pub(crate) fn opfl_pred_luma<BD: BitDepth>(
    recon: &mut ReconCtx<BD>,
    tmp: &mut [Vec<i16>; 2],
    seg_mask: &mut [u8],
    b: &Av2Block,
    bx: i32,
    by: i32,
    bw4: i32,
    bh4: i32,
    w4: i32,
    h4: i32,
    fi: &SbFrameInfo,
) -> bool {
    let layout = recon.frame.layout;
    let refs = b.ref_pair.refs();
    let r0 = refs[0] as usize;
    let r1 = refs[1] as usize;
    let filter = b.inter_data().filter;
    let cwp_idx = b.inter_data().cwp_idx as i32;
    let inter_mode = b.inter_data().inter_mode;
    let comp_type = b.inter_data().comp_type;
    let refine_mv = b.inter_data().refine_mv;
    let b_mv = [b.inter_data().mv[0].xy(), b.inter_data().mv[1].xy()];
    let r_pair = b.ref_pair;

    // comp_type AVG is value 1 in this codebase (1=AVG,2=WEDGE,3=SEG), matching
    // C `COMP_INTER_AVG` in `refine = comp_type == COMP_INTER_AVG && refine_mv`.
    let refine = comp_type == 1 && refine_mv != 0;
    let opfl = inter_mode >= crate::levels::CompInterPredMode::OpflNearMvNearMv as u8;

    let refp0 = match recon.refp[r0].clone() {
        Some(p) => p,
        None => return false,
    };
    let refp1 = match recon.refp[r1].clone() {
        Some(p) => p,
        None => return false,
    };
    let refp = [&refp0, &refp1];

    let w = fi.bw * 4;
    let h = fi.bh * 4;

    let bacp = recon.seq_hdr.imp_msk_bld && cwp_idx == 8;
    if bacp {
        for m in seg_mask.iter_mut() {
            *m = 0x20;
        }
    }
    let mut have_bacp = false;

    let d0 = fi.absrefdist[r0] as i32;
    let d1 = fi.absrefdist[r1] as i32;
    let dw0 = apply_sign_i8(1 + (d0 > d1) as i32, -(fi.refdist[r0] as i32));
    let dw1 = apply_sign_i8(1 + (d1 > d0) as i32, fi.refdist[r1] as i32);
    let dweights = [dw0, dw1];

    let bs = 2 - (b.bs == BlockSize::Bs8x8 as u8 as i8) as i32;
    let t_swap = (recon.rf.ref_flip & (1u64 << (r0 * 8 + r1))) != 0;

    let yw = (bw4 * 4) as usize;
    let p_stride = (((bw4 + refine as i32 * 2) * 4) as usize + 63) & !63;
    let psz = p_stride * (((bw4 + refine as i32 * 2) * 4) as usize + 8);
    let bd = recon.bd;
    let mut p = [
        vec![BD::Pixel::default(); psz.max(p_stride * 8)],
        vec![BD::Pixel::default(); psz.max(p_stride * 8)],
    ];

    let sh4 = imin(4, bh4);
    let sw4 = imin(4, bw4);

    let mut top = [by * 4 + (b_mv[0].y >> 3) - 3, by * 4 + (b_mv[1].y >> 3) - 3];

    let mut y = 0i32;
    while y < h4 {
        let mut left = [bx * 4 + (b_mv[0].x >> 3) - 3, bx * 4 + (b_mv[1].x >> 3) - 3];
        if refine {
            let mut x = 0i32;
            while x < w4 {
                // bilinear prefetch both refs at b->mv[n]-32 into p[n].
                for n in 0..2 {
                    prep_opfl_prefetch_rect_8bpc(
                        bd,
                        recon.frame.exec,
                        &mut p[n],
                        p_stride,
                        refp[n],
                        bx + x,
                        by + y,
                        (sw4 + 2) * 4,
                        (sh4 + 2) * 4,
                        b_mv[n].x - 32,
                        b_mv[n].y - 32,
                        iclip(left[n], 0, w - 1),
                        iclip(left[n] + 4 * sw4 + 7, 1, w),
                        iclip(top[n], 0, h - 1),
                        iclip(top[n] + 4 * sh4 + 7, 1, h),
                        recon.scratch.inter_mc_tmp_mut(),
                    );
                }
                let (dx, dy) = crate::mc::sad_refine_mv::<BD::Pixel>(
                    &p[0],
                    p_stride,
                    &p[1],
                    p_stride,
                    (sw4 * 4) as usize,
                    (sh4 * 4) as usize,
                    refine_mv == 2,
                    bd.bitdepth_min_8(),
                );
                if opfl {
                    let mut res = [crate::mc::OpflRegressionData::default(); 4];
                    let o0 = ((4 + dy) * p_stride as i32 + (4 + dx)) as usize;
                    let o1 = ((4 - dy) * p_stride as i32 + (4 - dx)) as usize;
                    crate::mc::opfl_derive_mv(
                        bd,
                        &mut res,
                        &p[0][o0..],
                        p_stride,
                        &p[1][o1..],
                        p_stride,
                        (sw4 * 4) as usize,
                        (sh4 * 4) as usize,
                        (bs * 4) as usize,
                        dweights,
                    );
                    let mut ri = 0usize;
                    let mut byi = 0i32;
                    while byi < sh4 {
                        let mut bxi = 0i32;
                        while bxi < sw4 {
                            let mut dd = crate::recon::OpflMvDeltaBlock::default();
                            crate::recon::opfl_mv_adj(&res[ri], &mut dd, dweights);
                            ri += 1;
                            let rmv_idx = (((by + y) & 31) >> 1) as usize * 16
                                + (byi != 0) as usize * 16
                                + (((bx + x + bxi) & 31) >> 1) as usize;
                            let mut mv = [Mv::default(); 2];
                            mv[0] = Mv {
                                c: MvXY {
                                    y: b_mv[0].y * 2 + dd.d[0].y as i32 + dy * 16,
                                    x: b_mv[0].x * 2 + dd.d[0].x as i32 + dx * 16,
                                },
                            };
                            mv[1] = Mv {
                                c: MvXY {
                                    y: b_mv[1].y * 2 + dd.d[1].y as i32 - dy * 16,
                                    x: b_mv[1].x * 2 + dd.d[1].x as i32 - dx * 16,
                                },
                            };
                            for i in 0..2 {
                                let off = ((y + byi) * 4) as usize * yw + ((x + bxi) * 4) as usize;
                                mc_opfl_8bpc(
                                    bd,
                                    recon.frame.exec,
                                    &mut tmp[i],
                                    off,
                                    yw,
                                    bs,
                                    bs,
                                    bx + x + bxi,
                                    by + y + byi,
                                    0,
                                    mv[i].x(),
                                    mv[i].y(),
                                    refp[i],
                                    filter,
                                    iclip(left[i], 0, w - 1),
                                    iclip(left[i] + sw4 * 4 + 7, 1, w),
                                    iclip(top[i], 0, h - 1),
                                    iclip(top[i] + sh4 * 4 + 7, 1, h),
                                    recon.scratch.inter_mc_tmp_mut(),
                                );
                            }
                            let dmv = [
                                Mv {
                                    c: MvXY {
                                        y: (mv[0].y() + (dd.d[0].y > 0) as i32) >> 1,
                                        x: (mv[0].x() + (dd.d[0].x > 0) as i32) >> 1,
                                    },
                                },
                                Mv {
                                    c: MvXY {
                                        y: (mv[1].y() + (dd.d[1].y > 0) as i32) >> 1,
                                        x: (mv[1].x() + (dd.d[1].x > 0) as i32) >> 1,
                                    },
                                },
                            ];
                            update_temporal_grid(
                                recon,
                                by + y + byi,
                                bx + x + bxi,
                                1,
                                1,
                                r_pair,
                                &dmv,
                                t_swap,
                            );
                            if bacp {
                                have_bacp |= crate::recon::get_mask(
                                    seg_mask,
                                    yw,
                                    bx,
                                    x + bxi,
                                    by,
                                    y + byi,
                                    &mv,
                                    4,
                                    4,
                                    2,
                                    2,
                                    w,
                                    h,
                                );
                            }
                            crate::recon::scaledown_16pel_mv_for_chroma(&mut mv, layout);
                            recon.scratch.rmv[rmv_idx][0] = mv;
                            bxi += 2;
                        }
                        byi += 2;
                    }
                } else {
                    let rmv_idx =
                        (((by + y) & 31) >> 1) as usize * 16 + (((bx + x) & 31) >> 1) as usize;
                    let mut mv = [Mv::default(); 2];
                    mv[0] = Mv {
                        c: MvXY {
                            y: b_mv[0].y + dy * 8,
                            x: b_mv[0].x + dx * 8,
                        },
                    };
                    mv[1] = Mv {
                        c: MvXY {
                            y: b_mv[1].y - dy * 8,
                            x: b_mv[1].x - dx * 8,
                        },
                    };
                    for i in 0..2 {
                        let off = (y * 4) as usize * yw + (x * 4) as usize;
                        mc_prep_bounds_8bpc(
                            bd,
                            recon.frame.exec,
                            &mut tmp[i],
                            yw,
                            refp[i],
                            0,
                            bx + x,
                            by + y,
                            sw4,
                            sh4,
                            mv[i].x(),
                            mv[i].y(),
                            filter,
                            recon.frame.ss_hor,
                            recon.frame.ss_ver,
                            iclip(left[i], 0, w - 1),
                            iclip(left[i] + sw4 * 4 + 7, 1, w),
                            iclip(top[i], 0, h - 1),
                            iclip(top[i] + sh4 * 4 + 7, 1, h),
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                        let _ = off;
                    }
                    update_temporal_grid(
                        recon,
                        by + y,
                        bx + x,
                        (sw4 >> 1) as usize,
                        (sh4 >> 1) as usize,
                        r_pair,
                        &mv,
                        t_swap,
                    );
                    crate::recon::scaleup_8pel_mv_for_chroma(&mut mv, layout);
                    if bacp {
                        have_bacp |= crate::recon::get_mask(
                            seg_mask, yw, bx, x, by, y, &mv, 3, 3, sw4, sh4, w, h,
                        );
                    }
                    recon.scratch.rmv[rmv_idx][0] = mv;
                }
                for n in 0..2 {
                    left[n] += 16;
                }
                x += sw4;
            }
        } else {
            debug_assert!(opfl);
            // bilinear prefetch whole rows (full frame bounds).
            for n in 0..2 {
                prep_opfl_prefetch_rect_8bpc(
                    bd,
                    recon.frame.exec,
                    &mut p[n],
                    p_stride,
                    refp[n],
                    bx,
                    by + y,
                    bw4 * 4,
                    sh4 * 4,
                    b_mv[n].x,
                    b_mv[n].y,
                    0,
                    w,
                    0,
                    h,
                    recon.scratch.inter_mc_tmp_mut(),
                );
            }
            // res[bs-block grid]: rows = sh4/bs, cols = bw4/bs.
            let nres = ((sh4 / bs) * (bw4 / bs)) as usize;
            let mut res = vec![crate::mc::OpflRegressionData::default(); nres.max(1)];
            crate::mc::opfl_derive_mv(
                bd,
                &mut res,
                &p[0],
                p_stride,
                &p[1],
                p_stride,
                (bw4 * 4) as usize,
                (sh4 * 4) as usize,
                (bs * 4) as usize,
                dweights,
            );
            // dd[4] for the BS_8x8 (bs==1) accumulation special case.
            let mut dd_acc = [crate::recon::OpflMvDeltaBlock::default(); 4];
            let mut dd_acc_n = 0usize;
            let cols = (bw4 / bs) as usize;
            let mut byi = 0i32;
            let mut row = 0usize;
            while byi < sh4 {
                let mut bxi = 0i32;
                let mut xx = 0usize;
                while bxi < bw4 {
                    let ri = row * cols + xx;
                    let mut dd = crate::recon::OpflMvDeltaBlock::default();
                    crate::recon::opfl_mv_adj(&res[ri], &mut dd, dweights);
                    let store_rmv = !(bs == 1 && (bxi != 0 || byi != 0));
                    let rmv_idx = (((by + y) & 31) >> 1) as usize * 16
                        + (byi != 0) as usize * 16
                        + (((bx + bxi) & 31) >> 1) as usize;
                    let mut mv = [Mv::default(); 2];
                    mv[0] = Mv {
                        c: MvXY {
                            y: b_mv[0].y * 2 + dd.d[0].y as i32,
                            x: b_mv[0].x * 2 + dd.d[0].x as i32,
                        },
                    };
                    mv[1] = Mv {
                        c: MvXY {
                            y: b_mv[1].y * 2 + dd.d[1].y as i32,
                            x: b_mv[1].x * 2 + dd.d[1].x as i32,
                        },
                    };
                    for i in 0..2 {
                        let off = ((y + byi) * 4) as usize * yw + (bxi * 4) as usize;
                        mc_opfl_8bpc(
                            bd,
                            recon.frame.exec,
                            &mut tmp[i],
                            off,
                            yw,
                            bs,
                            bs,
                            bx + bxi,
                            by + y + byi,
                            0,
                            mv[i].x(),
                            mv[i].y(),
                            refp[i],
                            filter,
                            iclip(left[i] + bxi * 4, 0, w - 1),
                            iclip(left[i] + bxi * 4 + 7 + 8, 1, w),
                            iclip(top[i] + byi * 4, 0, h - 1),
                            iclip(top[i] + byi * 4 + 7 + 8, 1, h),
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                    if bs > 1 {
                        let dmv = [
                            Mv {
                                c: MvXY {
                                    y: (mv[0].y() + (dd.d[0].y > 0) as i32) >> 1,
                                    x: (mv[0].x() + (dd.d[0].x > 0) as i32) >> 1,
                                },
                            },
                            Mv {
                                c: MvXY {
                                    y: (mv[1].y() + (dd.d[1].y > 0) as i32) >> 1,
                                    x: (mv[1].x() + (dd.d[1].x > 0) as i32) >> 1,
                                },
                            },
                        ];
                        update_temporal_grid(
                            recon,
                            by + y + byi,
                            bx + bxi,
                            (bs >> 1) as usize,
                            (bs >> 1) as usize,
                            r_pair,
                            &dmv,
                            t_swap,
                        );
                    } else {
                        // BS_8x8: accumulate dd for the post-row averaged write.
                        if dd_acc_n < 4 {
                            dd_acc[dd_acc_n] = dd;
                            dd_acc_n += 1;
                        }
                    }
                    if bacp {
                        have_bacp |= crate::recon::get_mask(
                            seg_mask,
                            yw,
                            bx,
                            bxi,
                            by,
                            y + byi,
                            &mv,
                            4,
                            4,
                            bs,
                            bs,
                            w,
                            h,
                        );
                    }
                    crate::recon::scaledown_16pel_mv_for_chroma(&mut mv, layout);
                    if store_rmv {
                        recon.scratch.rmv[rmv_idx][0] = mv;
                    }
                    bxi += bs;
                    xx += 1;
                }
                byi += bs;
                row += 1;
            }
            if bs == 1 {
                // BS_8x8: average the four 4x4 dd's into a single 8x8 MV
                let s0x = dd_acc[0].d[0].x as i32
                    + dd_acc[1].d[0].x as i32
                    + dd_acc[2].d[0].x as i32
                    + dd_acc[3].d[0].x as i32;
                let s0y = dd_acc[0].d[0].y as i32
                    + dd_acc[1].d[0].y as i32
                    + dd_acc[2].d[0].y as i32
                    + dd_acc[3].d[0].y as i32;
                let s1x = dd_acc[0].d[1].x as i32
                    + dd_acc[1].d[1].x as i32
                    + dd_acc[2].d[1].x as i32
                    + dd_acc[3].d[1].x as i32;
                let s1y = dd_acc[0].d[1].y as i32
                    + dd_acc[1].d[1].y as i32
                    + dd_acc[2].d[1].y as i32
                    + dd_acc[3].d[1].y as i32;
                let dmv = [
                    Mv {
                        c: MvXY {
                            y: (b_mv[0].y * 8 + s0y + 3 + (s0y > 0) as i32) >> 3,
                            x: (b_mv[0].x * 8 + s0x + 3 + (s0x > 0) as i32) >> 3,
                        },
                    },
                    Mv {
                        c: MvXY {
                            y: (b_mv[1].y * 8 + s1y + 3 + (s1y > 0) as i32) >> 3,
                            x: (b_mv[1].x * 8 + s1x + 3 + (s1x > 0) as i32) >> 3,
                        },
                    },
                ];
                update_temporal_grid(recon, by + y, bx, 1, 1, r_pair, &dmv, t_swap);
            }
        }
        for n in 0..2 {
            top[n] += 4 * sh4;
        }
        y += sh4;
    }

    bacp && have_bacp
}

/// Synthesize and reconstruct the single whole-superblock TIP block of a
/// `frame_mode == 2` (whole-frame TIP) frame. Port of `tip_frame_recon_sb`
/// is reconstructed from one synthesized skip_txfm TIP block whose motion comes
/// from the projected temporal-MV grid (`rp_proj`). No bits are read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tip_frame_recon_sb<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    bx: i32,
    by: i32,
    bs: BlockSize,
    cbs: BlockSize,
    fi: &SbFrameInfo,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    static TIP_WTS: [i8; 8] = [8, 12, 16, 18, 20, 4, 6, -4];
    let tip = &recon.frm_hdr.tip;
    let mut b = crate::levels::Av2Block {
        bs: bs as i8,
        cbs: cbs as i8,
        is_intra: 0,
        intrabc: 0,
        seg_id: 0,
        skip_mode: 0,
        skip_txfm: 1,
        tx_part: TxPartition::None as u8,
        fsc: 0,
        tx_size_ll: 0,
        ref_pair: RefPair {
            r: [TIP_FRAME as i8, -1],
        },
        data: crate::levels::Av2BlockData::default(),
    };
    {
        let inter = b.inter_data_mut();
        inter.mv[0] = Mv::from_xy(tip.gmv_y as i32, tip.gmv_x as i32);
        inter.inter_mode = InterPredMode::NearMv as u8;
        inter.motion_mode = MotionMode::Translation as u8;
        inter.filter = tip.subpel_filter;
        inter.cwp_idx = TIP_WTS[tip.global_wtd_idx as usize];
    }

    // The whole superblock is a single TIP block; chroma is co-located.
    let has_luma = true;
    let has_chroma = cbs != BlockSize::Invalid;

    // frame_mode==2 superblocks are not entropy decoded, so the deblock /
    // qidx mask must be built here (mirroring the normal decode_b path) or
    // the filter pass sees an all-zero mask for these frames.
    if recon.frm_hdr.tip.apply_filter != 0 {
        let layout = recon.frame.layout;
        let ss_hor = recon.frame.ss_hor;
        let ss_ver = recon.frame.ss_ver;
        let bx4 = (bx & 63) as usize;
        let by4 = (by & 63) as usize;
        {
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
        if has_chroma {
            let m = &mut recon.lf_mask[recon.lf_idx];
            crate::lf_mask::create_db_mask(
                &mut m.filter_uv,
                &b,
                cbs,
                bx,
                by,
                fi.bw,
                fi.bh,
                layout,
                true,
                &mut a.tx_lpf_uv[bx4 >> ss_hor..],
                &mut l.tx_lpf_uv[by4 >> ss_ver..],
                recon.frm_hdr,
                recon.seq_hdr,
            );
        }
        // Splat the frame quant index into the loop-filter mask qidx grid.
        let qidx = recon.frm_hdr.quant.yac as u16;
        let qbase = (bx4 >> 4) + ((by4 & 0x30) >> 2);
        let sbsz64 = (fi.sb_step >> 4) as usize;
        let m = &mut recon.lf_mask[recon.lf_idx];
        let mut qoff = qbase;
        for _ in 0..sbsz64 {
            for x64 in 0..sbsz64 {
                m.qidx[qoff + x64] = qidx;
            }
            qoff += 4;
        }
    }

    recon_b_inter_tip(
        recon,
        msac,
        cdf_m,
        a,
        l,
        &b,
        bx,
        by,
        bx,
        by,
        bs,
        cbs,
        has_luma,
        has_chroma,
        ChromaPhase::Both,
        fi,
    )
}

/// Reconstruct a TIP (Temporal Interpolated Prediction) inter block (8bpc).
/// `b->ref.ref[0] == TIP_FRAME`: the block has no coded references; instead each
/// 8x8 sub-unit derives its motion from the projected temporal MV grid
/// (`rp_proj`), scaled to the two TIP reference frames, optionally refined with
/// (TIP is `COMP_INTER_NONE` → `COMP_INTER_AVG`: cwp_idx==8 → mask or average).
/// Residual decode is identical to compound (`inter_luma_tx_walk` /
/// `inter_chroma_residual_8bpc`).
#[allow(clippy::too_many_arguments)]
fn recon_b_inter_tip<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    chroma_stage: ChromaPhase,
    fi: &SbFrameInfo,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let bd = recon.bd;
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;
    let layout = recon.frame.layout;
    let filter = b.inter_data().filter;
    let cwp_idx = b.inter_data().cwp_idx as i32;
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    let b_mv0 = b.inter_data().mv[0].xy();

    // The two TIP reference frame indices (f->rf.tip.ref).
    let tip_refs = fi.tip.refs();
    let r0 = tip_refs[0] as usize;
    let r1 = tip_refs[1] as usize;

    let frame_mode = recon.frm_hdr.tip.frame_mode as i32;
    let tip_subpel = recon.frm_hdr.tip.subpel_filter;
    let mut opfl = recon.seq_hdr.tip_refine_mv
        && (frame_mode == 1 || tip_subpel == crate::headers::FilterMode::Sharp8Tap as u8);
    let refine = opfl && frame_mode == 1 && fi.refdist[r0] == -fi.refdist[r1];
    let bw4_full = BLOCK_DIMENSIONS[bs as u8 as usize][0] as i32;
    let bh4_full = BLOCK_DIMENSIONS[bs as u8 as usize][1] as i32;
    let is_256 = bs == BlockSize::Bs256x256;
    let step_shift = if frame_mode == 2 {
        (!opfl) as i32
    } else {
        ((!opfl && imin(bw4_full, bh4_full) >= 4) || is_256) as i32
    };
    let step = 2i32 << step_shift;
    opfl &= recon.seq_hdr.opfl_refine && recon.frm_hdr.has_bothside_refs != 0;

    // BACP (block adaptive compound prediction) masked-blend predicate.
    let bacp = recon.seq_hdr.imp_msk_bld
        && cwp_idx == 8
        && recon.svc[r0][0].scale == 0
        && recon.svc[r1][0].scale == 0;

    let d0 = fi.absrefdist[r0] as i32;
    let d1 = fi.absrefdist[r1] as i32;
    let dw0 = apply_sign_i8(1 + (d0 > d1) as i32, -(fi.refdist[r0] as i32));
    let dw1 = apply_sign_i8(1 + (d1 > d0) as i32, fi.refdist[r1] as i32);
    let dweights = [dw0, dw1];

    let sad8x8_thr: u32 = if frame_mode == 1 { 6 } else { 15 };
    let t_stride = recon.rf.rp_stride;
    let t_swap = (recon.rf.ref_flip & (1u64 << (r0 * 8 + r1))) != 0;
    let r_pair = fi.tip;

    let w = fi.bw * 4;
    let h = fi.bh * 4;

    let refp0 = recon.refp[r0].clone();
    let refp1 = recon.refp[r1].clone();
    let (refp0, refp1) = match (refp0, refp1) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(()),
    };
    let refp = [&refp0, &refp1];

    // SB-local projected-MV grid (`t->rt.rp_proj`).
    let rp_proj_off = recon.rt.rp_proj_off;

    let mut seg_mask = recon.scratch.take_compound_seg_mask();
    let mut luma_bacp = false;
    if has_luma {
        let bw4 = bw4_full;
        let bh4 = bh4_full;
        let w4 = imin(bw4, fi.bw - bx);
        let h4 = imin(bh4, fi.bh - by);
        let yw = (bw4 * 4) as usize;
        let yh = (bh4 * 4) as usize;
        let y_stride = recon.frame.y_stride_px;
        let dst_off = 4 * (by as usize * y_stride + bx as usize);

        let _len = crate::mc_dispatch::compound_tmp_len(yw, yh);
        let mut tmp = recon.scratch.take_compound_tmp(_len);
        if bacp {
            for m in seg_mask.iter_mut() {
                *m = 0x20;
            }
        }
        let p_stride = ((step as usize + 2) * 4 + 15) & !15; // ((step+2)*4*1 + 63)&~63 ⇒ use 16-byte; keep generous
        let p_stride = if p_stride < (step as usize + 2) * 4 {
            (step as usize + 2) * 4
        } else {
            p_stride
        };
        let psz = p_stride * ((step as usize + 2) * 4);
        let mut p = [
            vec![BD::Pixel::default(); psz],
            vec![BD::Pixel::default(); psz],
        ];

        let mut y = 0i32;
        let mut yy = 0i32;
        while y < h4 {
            let off_y8 = (((by + y) & (fi.sb_step - 1)) >> 1) as isize * t_stride;
            let mut x = 0i32;
            while x < w4 {
                let off_8x8 = (off_y8 + ((bx + x) >> 1) as isize) as usize;
                let mut tmv = recon.rf.rp_proj[rp_proj_off + off_8x8].mv;
                if tmv.y() == crate::levels::INVALID_MV {
                    tmv = Mv::from_bits(0);
                }
                // rmv grid slot for this 8x8 (version 0=cmv, 1=chroma-scaled).
                let rmv_idx =
                    (((by + y) & 31) >> 1) as usize * 16 + (((bx + x) & 31) >> 1) as usize;
                let mut cmv = [Mv::default(); 2];
                let mut rmv1 = [Mv::default(); 2];
                let mut left = [0i32; 2];
                let mut top = [0i32; 2];
                for i in 0..2 {
                    let tipmv = crate::refmvs::scale_mv(tmv, recon.rf.tip.sf[i]);
                    let cy = iclip(tipmv.y() + b_mv0.y, -0xffff, 0xffff);
                    let cx = iclip(tipmv.x() + b_mv0.x, -0xffff, 0xffff);
                    cmv[i] = Mv {
                        c: MvXY { y: cy, x: cx },
                    };
                    rmv1[i] = Mv {
                        c: MvXY { y: cy, x: cx },
                    };
                    top[i] = by * 4 + y * 4 + (cy >> 3) - 3;
                    left[i] = bx * 4 + x * 4 + (cx >> 3) - 3;
                }
                crate::recon::scaleup_8pel_mv_for_chroma(&mut rmv1, layout);

                if opfl {
                    // bilinear prefetch both refs into p[i] (3-bit subpel mv).
                    for i in 0..2 {
                        let cy = cmv[i].y();
                        let cx = cmv[i].x();
                        prep_opfl_prefetch_8bpc(
                            bd,
                            recon.frame.exec,
                            &mut p[i],
                            p_stride,
                            refp[i],
                            bx + x,
                            by + y,
                            step,
                            cx - 32,
                            cy - 32,
                            iclip(left[i], 0, w - 1),
                            iclip(left[i] + 7 + step * 4, 1, w),
                            iclip(top[i], 0, h - 1),
                            iclip(top[i] + 7 + step * 4, 1, h),
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                    let (mut dy, mut dx) = (0i32, 0i32);
                    if refine {
                        let (rdx, rdy) = crate::mc::sad_refine_mv::<BD::Pixel>(
                            &p[0],
                            p_stride,
                            &p[1],
                            p_stride,
                            (step * 4) as usize,
                            (step * 4) as usize,
                            true,
                            bd.bitdepth_min_8(),
                        );
                        dy = rdy;
                        dx = rdx;
                        cmv[0].set_y(cmv[0].y() + 8 * dy);
                        cmv[1].set_y(cmv[1].y() - 8 * dy);
                        cmv[0].set_x(cmv[0].x() + 8 * dx);
                        cmv[1].set_x(cmv[1].x() - 8 * dx);
                    }
                    let mut dd = crate::recon::OpflMvDeltaBlock::default();
                    let sad = if is_256 && frame_mode == 1 {
                        0
                    } else {
                        let o0 = ((4 + dy) * p_stride as i32 + (4 + dx)) as usize;
                        let o1 = ((4 - dy) * p_stride as i32 + (4 - dx)) as usize;
                        crate::mc::sad8x8::<BD::Pixel>(
                            &p[0][o0..],
                            p_stride,
                            &p[1][o1..],
                            p_stride,
                            bd.bitdepth_min_8(),
                        )
                    };
                    if sad >= sad8x8_thr {
                        let mut res = [crate::mc::OpflRegressionData::default(); 4];
                        let o0 = ((4 + dy) * p_stride as i32 + (4 + dx)) as usize;
                        let o1 = ((4 - dy) * p_stride as i32 + (4 - dx)) as usize;
                        crate::mc::opfl_derive_mv(
                            bd,
                            &mut res,
                            &p[0][o0..],
                            p_stride,
                            &p[1][o1..],
                            p_stride,
                            (step * 4) as usize,
                            (step * 4) as usize,
                            8,
                            dweights,
                        );
                        crate::recon::opfl_mv_adj(&res[0], &mut dd, dweights);
                    }
                    for i in 0..2 {
                        cmv[i].set_x(cmv[i].x() * 2 + dd.d[i].x as i32);
                        cmv[i].set_y(cmv[i].y() * 2 + dd.d[i].y as i32);
                    }
                    for i in 0..2 {
                        let cy = cmv[i].y();
                        let cx = cmv[i].x();
                        let off = (y * 4) as usize * yw + (x * 4) as usize;
                        mc_opfl_8bpc(
                            bd,
                            recon.frame.exec,
                            &mut tmp[i],
                            off,
                            yw,
                            step,
                            step,
                            bx + x,
                            by + y,
                            0,
                            cx,
                            cy,
                            refp[i],
                            filter,
                            iclip(left[i], 0, w - 1),
                            iclip(left[i] + 7 + step * 4, 1, w),
                            iclip(top[i], 0, h - 1),
                            iclip(top[i] + 7 + step * 4, 1, h),
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                    let dmv = [
                        Mv {
                            c: MvXY {
                                y: (cmv[0].y() + (dd.d[0].y > 0) as i32) >> 1,
                                x: (cmv[0].x() + (dd.d[0].x > 0) as i32) >> 1,
                            },
                        },
                        Mv {
                            c: MvXY {
                                y: (cmv[1].y() + (dd.d[1].y > 0) as i32) >> 1,
                                x: (cmv[1].x() + (dd.d[1].x > 0) as i32) >> 1,
                            },
                        },
                    ];
                    update_temporal_grid(
                        recon,
                        by + y,
                        bx + x,
                        (step >> 1) as usize,
                        (step >> 1) as usize,
                        r_pair,
                        &dmv,
                        t_swap,
                    );
                    if bacp {
                        luma_bacp |= crate::recon::get_mask(
                            &mut seg_mask,
                            (bw4 * 4) as usize,
                            bx,
                            x,
                            by,
                            y,
                            &cmv,
                            4,
                            4,
                            step,
                            step,
                            w,
                            h,
                        );
                    }
                    crate::recon::scaledown_16pel_mv_for_chroma(&mut cmv, layout);
                } else {
                    // non-opfl: plain prep-MC, full bounds.
                    for i in 0..2 {
                        let cy = cmv[i].y();
                        let cx = cmv[i].x();
                        let off = (y * 4) as usize * yw + (x * 4) as usize;
                        inter_mc_plane_prep_at_8bpc(
                            bd,
                            recon.frame.exec,
                            &mut tmp[i],
                            off,
                            yw,
                            refp[i],
                            0,
                            bx + x,
                            by + y,
                            step,
                            step,
                            cx,
                            cy,
                            filter,
                            ss_hor,
                            ss_ver,
                            fi.bw,
                            fi.bh,
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                    update_temporal_grid(
                        recon,
                        by + y,
                        bx + x,
                        (step >> 1) as usize,
                        (step >> 1) as usize,
                        r_pair,
                        &cmv,
                        t_swap,
                    );
                    if step == 4 && frame_mode == 1 {
                        for p in 1..4i32 {
                            let mut tmv2 = recon.rf.rp_proj[rp_proj_off
                                + off_8x8
                                + (p & 1) as usize
                                + (((p & 2) >> 1) as isize * t_stride) as usize]
                                .mv;
                            if tmv2.y() == crate::levels::INVALID_MV {
                                tmv2 = Mv::from_bits(0);
                            }
                            let mut dmv = [Mv::default(); 2];
                            for i in 0..2 {
                                let tipmv = crate::refmvs::scale_mv(tmv2, recon.rf.tip.sf[i]);
                                dmv[i] = Mv {
                                    c: MvXY {
                                        y: iclip(tipmv.y() + b_mv0.y, -0xffff, 0xffff),
                                        x: iclip(tipmv.x() + b_mv0.x, -0xffff, 0xffff),
                                    },
                                };
                            }
                            update_temporal_grid_sub(
                                recon,
                                by + y,
                                bx + x,
                                p,
                                t_stride,
                                r_pair,
                                &dmv,
                                t_swap,
                            );
                        }
                    }
                    if bacp {
                        luma_bacp |= crate::recon::get_mask(
                            &mut seg_mask,
                            (bw4 * 4) as usize,
                            bx,
                            x,
                            by,
                            y,
                            &cmv,
                            3,
                            3,
                            step,
                            step,
                            w,
                            h,
                        );
                    }
                    crate::recon::scaleup_8pel_mv_for_chroma(&mut cmv, layout);
                }
                // store rmv grid: [0]=cmv (post-process), [1]=chroma-scaled base
                recon.scratch.rmv[rmv_idx][0] = cmv;
                recon.scratch.rmv[rmv_idx][1] = rmv1;
                x += step;
            }
            y += step;
            yy += 1;
        }
        let _ = yy;

        // blend (COMP_INTER_NONE → COMP_INTER_AVG, cwp_idx==8).
        let have_bacp = bacp && luma_bacp;
        let (tmp0, tmp1) = tmp.split_at(1);
        if have_bacp {
            mc_mask(
                recon.frame.exec,
                bd,
                &mut recon.dst_y[dst_off..],
                y_stride,
                &tmp0[0],
                &tmp1[0],
                yw,
                yh,
                &seg_mask,
            );
        } else {
            mc_avg(
                recon.frame.exec,
                bd,
                &mut recon.dst_y[dst_off..],
                y_stride,
                &tmp0[0],
                &tmp1[0],
                yw,
                yh,
            );
        }
        recon.scratch.put_compound_tmp(tmp);

        // luma residual.
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        if lossless {
            let tx = if b.tx_size_ll != 0 {
                crate::tables::MAX_TXFM_SIZE_FOR_BS[bs as usize][3] as usize
            } else {
                0
            };
            let t_dim = &TXFM_DIMENSIONS[tx];
            let (tw4, th4) = (t_dim.w as i32, t_dim.h as i32);
            let mut yr = 0;
            while yr < h4 {
                let mut xr = 0;
                while xr < w4 {
                    inter_residual_tx_8bpc(
                        recon,
                        msac,
                        cdf_m,
                        a,
                        l,
                        b,
                        0,
                        tx,
                        bx + xr,
                        by + yr,
                        false,
                        0,
                        fi,
                    )?;
                    xr += tw4;
                }
                yr += th4;
            }
        } else {
            let tp = &crate::tables::TX_PART_TBL[bs as usize];
            let tx = tp[b.tx_part as usize] as usize;
            inter_luma_tx_walk(recon, msac, cdf_m, a, l, b, tx, bx, by, fi)?;
        }
    }

    if has_chroma && cbs != BlockSize::Invalid {
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbw4 = cb_dim[0] as i32;
        let cbh4 = cb_dim[1] as i32;
        let cw4 = imin(fi.bw - cbx, cbw4);
        let ch4 = imin(fi.bh - cby, cbh4);
        let cw4ss = (cw4 + ss_hor) >> ss_hor;
        let ch4ss = (ch4 + ss_ver) >> ss_ver;
        let uv_stride = recon.frame.uv_stride_px;
        let cw = (cbw4 * 4 >> ss_hor) as usize;
        let ch = (cbh4 * 4 >> ss_ver) as usize;

        // r_step = o_step = step.
        let r_step = step;
        let o_step = step;

        // ..)` selects the picture plane. BACP is decided on the U plane and
        // carried to V (`bacpu`). Chroma MC runs only at the recon stage
        // (`Both`/`ReconOnly`); the read stage consumes coefficients only.
        let do_chroma_mc = chroma_stage != ChromaPhase::ReadOnly;
        let mut chroma_bacp = false;
        for plane in (0..2usize).filter(|_| do_chroma_mc) {
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            let _len = crate::mc_dispatch::compound_tmp_len(cw, ch);
            let mut tmp = recon.scratch.take_compound_tmp(_len);
            let pl_bacp = rmv_uvpred(
                recon,
                b,
                &mut tmp,
                plane,
                r_step,
                o_step,
                cbw4,
                cbh4,
                cbx,
                cby,
                r_pair,
                true,
                &mut seg_mask,
                fi,
            );
            let use_bacp = if plane == 0 {
                chroma_bacp = pl_bacp;
                pl_bacp
            } else {
                chroma_bacp
            };
            let dst: &mut [BD::Pixel] = if plane == 0 {
                &mut recon.dst_u[dst_off..]
            } else {
                &mut recon.dst_v[dst_off..]
            };
            let (tmp0, tmp1) = tmp.split_at(1);
            if use_bacp {
                mc_mask(
                    recon.frame.exec,
                    bd,
                    dst,
                    uv_stride,
                    &tmp0[0],
                    &tmp1[0],
                    cw,
                    ch,
                    &seg_mask,
                );
            } else {
                mc_avg(
                    recon.frame.exec,
                    bd,
                    dst,
                    uv_stride,
                    &tmp0[0],
                    &tmp1[0],
                    cw,
                    ch,
                );
            }
            recon.scratch.put_compound_tmp(tmp);
        }

        // chroma residual.
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        let uvtx = if lossless {
            0usize
        } else {
            let layout_idx =
                (crate::headers::PixelLayout::I444 as i32 - recon.frame.layout as i32) as usize;
            crate::tables::MAX_TXFM_SIZE_FOR_BS[cbs as u8 as usize][layout_idx] as usize
        };
        let uv_txtp_seed = recon.scratch.txtp_map[(by & 15) as usize * 16 + (bx & 15) as usize];
        let cbw4ss = (cbw4 + ss_hor) >> ss_hor;
        let cbh4ss = (cbh4 + ss_ver) >> ss_ver;
        inter_chroma_residual_8bpc(
            recon,
            msac,
            cdf_m,
            a,
            l,
            b,
            uvtx,
            cbx,
            cby,
            cbw4ss,
            cbh4ss,
            cw4ss,
            ch4ss,
            uv_txtp_seed,
            chroma_stage,
            fi,
        )?;
    }
    recon.scratch.put_compound_seg_mask(seg_mask);
    Ok(())
}

/// per-8x8 refined-MV grid `t->rmv` (`recon.scratch.rmv`) stored during the luma
/// `tip_pred`, and motion-compensates the chroma plane into `tmp[0]`/`tmp[1]`
/// (i16 prep, stride `bw4*4>>ss_hor`) via `mc_opfl`. Returns BACP for plane 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmv_uvpred<BD: crate::pixel::BitDepth>(
    recon: &mut ReconCtx<BD>,
    b: &Av2Block,
    tmp: &mut [Vec<i16>; 2],
    plane: usize,
    r_step: i32,
    o_step: i32,
    bw4: i32,
    bh4: i32,
    cbx: i32,
    cby: i32,
    r_pair: crate::levels::RefPair,
    tip: bool,
    mask: &mut [u8],
    fi: &SbFrameInfo,
) -> bool {
    let bd = recon.bd;
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;
    let refs = r_pair.refs();
    let r0 = refs[0] as usize;
    let r1 = refs[1] as usize;
    // For non-TIP (compound OPFL) the MC reference-window bounds use the block's
    let b_mv = [b.inter_data().mv[0].xy(), b.inter_data().mv[1].xy()];
    let refp0 = match recon.refp[r0].clone() {
        Some(p) => p,
        None => return false,
    };
    let refp1 = match recon.refp[r1].clone() {
        Some(p) => p,
        None => return false,
    };
    let refp = [&refp0, &refp1];
    let filter = b.inter_data().filter;

    let stride = (bw4 * 4 >> ss_hor) as usize;
    let bacp = plane == 0 && recon.seq_hdr.imp_msk_bld && b.inter_data().cwp_idx == 8;
    if bacp {
        for m in mask.iter_mut().take((bw4 * bh4 * 16) as usize) {
            *m = 0x20;
        }
    }
    let mut have_bacp = false;

    let w = fi.bw * 4 >> ss_hor;
    let h = fi.bh * 4 >> ss_ver;
    let rw4 = imin(bw4, r_step);
    let rh4 = imin(bh4, r_step);
    let ow4 = imin(bw4, o_step);
    let oh4 = imin(bh4, o_step);
    let hhtaps = 2 + 2 * (rw4 > 1 + ss_hor) as i32;
    let hvtaps = 2 + 2 * (rh4 > 1 + ss_ver) as i32;
    let h4 = imin(bh4, fi.bh - cby);
    let w4 = imin(bw4, fi.bw - cbx);

    let mut uvoff = 0usize;
    let mut y = 0i32;
    while y < h4 {
        // rmv_line base for this outer row.
        let mut x = 0i32;
        while x < w4 {
            let rmv_base = (((cby + y) & 31) >> 1) as usize * 16 + (((cbx + x) & 31) >> 1) as usize;
            let rmv = recon.scratch.rmv[rmv_base];
            let mut top = [0i32; 2];
            let mut left = [0i32; 2];
            let mut bottom = [0i32; 2];
            let mut right = [0i32; 2];
            for i in 0..2 {
                let (mvy, mvx) = if tip {
                    (rmv[1][i].y(), rmv[1][i].x())
                } else {
                    (b_mv[i].y, b_mv[i].x)
                };
                top[i] = ((cby + y) * 4 >> ss_ver) + (mvy >> 4);
                left[i] = ((cbx + x) * 4 >> ss_hor) + (mvx >> 4);
                bottom[i] = iclip(top[i] + (4 * rh4 >> ss_ver) + hvtaps, 1, h);
                right[i] = iclip(left[i] + (4 * rw4 >> ss_hor) + hhtaps, 1, w);
                top[i] = iclip(top[i] + 1 - hvtaps, 0, h - 1);
                left[i] = iclip(left[i] + 1 - hhtaps, 0, w - 1);
            }
            let mut uvoffi = uvoff;
            let mut by = 0i32;
            while by < rh4 {
                let mut bx = 0i32;
                while bx < rw4 {
                    // advanced by the outer-row term; reconstruct the absolute
                    // grid index.
                    let rmv2_idx = (((cby + y) & 31) >> 1) as usize * 16
                        + (by != 0) as usize * 16
                        + (((cbx + x + bx) & 31) >> 1) as usize;
                    let rmv2 = recon.scratch.rmv[rmv2_idx];
                    for i in 0..2 {
                        let off = uvoffi + (((x + bx) * 4 >> ss_hor) as usize);
                        let mvy = rmv2[0][i].y();
                        let mvx = rmv2[0][i].x();
                        mc_opfl_8bpc(
                            bd,
                            recon.frame.exec,
                            &mut tmp[i],
                            off,
                            stride,
                            ow4 >> ss_hor,
                            oh4 >> ss_ver,
                            (cbx + x + bx) >> ss_hor,
                            (cby + y + by) >> ss_ver,
                            1 + plane,
                            mvx,
                            mvy,
                            refp[i],
                            filter,
                            left[i],
                            right[i],
                            top[i],
                            bottom[i],
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                    if bacp {
                        have_bacp |= crate::recon::get_mask(
                            mask,
                            (bw4 * 4 >> ss_hor) as usize,
                            cbx >> ss_hor,
                            (x + bx) >> ss_hor,
                            cby >> ss_ver,
                            (y + by) >> ss_ver,
                            &rmv2[0],
                            4,
                            4,
                            ow4 >> ss_hor,
                            oh4 >> ss_ver,
                            fi.bw * 4 >> ss_hor,
                            fi.bh * 4 >> ss_ver,
                        );
                    }
                    bx += ow4;
                }
                uvoffi += (oh4 * 4 * stride as i32 >> ss_ver) as usize;
                by += oh4;
            }
            x += rw4;
        }
        uvoff += (rh4 * 4 * stride as i32 >> ss_ver) as usize;
        y += rh4;
    }
    bacp && have_bacp
}

/// (the `dst8` path) for the prefetch, writing 8bpc pixels with 3-bit subpel,
/// but the OPFL DSPs read p[i] as 8bpc. We replicate the `mc` 8bpc dst path
/// with bilinear filter and bounded edges, producing `(step+2)*4` square px.
#[allow(clippy::too_many_arguments)]
fn prep_opfl_prefetch_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    exec: &crate::exec_context::ExecContext,
    p: &mut [BD::Pixel],
    p_stride: usize,
    ref_pic: &crate::picture::Picture,
    bx4: i32,
    by4: i32,
    step: i32,
    mvx: i32,
    mvy: i32,
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let ref_stride = ref_pic.stride[0].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(0) {
        Some(s) => s,
        None => return,
    };
    let dim = ((step + 2) * 4) as i32;
    let mx = mvx & 7;
    let my = mvy & 7;
    let dx = bx4 * 4 + (mvx >> 3);
    let dy = by4 * 4 + (mvy >> 3);

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + dim + (mx != 0) as i32 * 4 > right
        || dy + dim + (my != 0) as i32 * 4 > bottom;
    let dimu = dim as usize;
    let mut emu_buf = if need_emu {
        Some(vec![BD::Pixel::default(); 192 * 192])
    } else {
        None
    };
    let (src, src_off, src_stride) = if let Some(ref mut buf) = emu_buf {
        let emu_w = dimu + (mx != 0) as usize * 7;
        let emu_h = dimu + (my != 0) as usize * 7;
        let emu_stride = 192usize;
        let src_base = ((left + top * ref_stride as i32).max(0) as usize).min(ref_data.len());
        inter_emu_edge_8bpc::<BD>(
            buf,
            emu_stride,
            &ref_data[src_base..],
            ref_stride,
            emu_w,
            emu_h,
            (right - left).max(0) as usize,
            (bottom - top).max(0) as usize,
            dx - (mx != 0) as i32 * 3 - left,
            dy - (my != 0) as i32 * 3 - top,
        );
        let off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
        (&buf[..], off, emu_stride)
    } else {
        let off = dy as usize * ref_stride + dx as usize;
        (ref_data, off, ref_stride)
    };
    // 3-bit subpel → kernel expects 4-bit (mx << 1).
    if BD::BPC == 8 {
        let p8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(p);
        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
        exec.put_bilin_8bpc_with_scratch(
            p8,
            p_stride,
            &src8[src_off..],
            src_stride,
            dimu,
            dimu,
            mx << 1,
            my << 1,
            inter_scratch,
        );
    } else {
        crate::mc::put_bilin(
            bd,
            p,
            p_stride,
            &src[src_off..],
            src_stride,
            dimu,
            dimu,
            mx << 1,
            my << 1,
        );
    }
}

/// absolute 8x8 position `(by_abs, bx_abs) = (t->by+y, t->bx+x)` into the frame
/// temporal MV grid `f->rf.rp` (`recon.cur_mvs`).
#[allow(clippy::too_many_arguments)]
fn update_temporal_grid<BD: crate::pixel::BitDepth>(
    recon: &mut ReconCtx<BD>,
    by_abs: i32,
    bx_abs: i32,
    w8: usize,
    h8: usize,
    r: crate::levels::RefPair,
    mv: &[crate::levels::Mv; 2],
    swap: bool,
) {
    let t_stride = recon.rf.rp_stride;
    let idx = ((by_abs >> 1) as isize * t_stride + (bx_abs >> 1) as isize) as usize;
    if idx >= recon.cur_mvs.len() {
        return;
    }

    crate::recon::update_temporal(
        &mut recon.cur_mvs[idx..],
        t_stride as usize,
        w8,
        h8,
        r,
        mv,
        swap,
    );
}

/// Per-subblock temporal MV write for the step==4 non-OPFL case
#[allow(clippy::too_many_arguments)]
fn update_temporal_grid_sub<BD: crate::pixel::BitDepth>(
    recon: &mut ReconCtx<BD>,
    by_abs: i32,
    bx_abs: i32,
    p: i32,
    t_stride: isize,
    r: crate::levels::RefPair,
    mv: &[crate::levels::Mv; 2],
    swap: bool,
) {
    let base = (by_abs >> 1) as isize * t_stride + (bx_abs >> 1) as isize;
    let idx = (base + ((p & 2) >> 1) as isize * t_stride + (p & 1) as isize) as usize;
    if idx >= recon.cur_mvs.len() {
        return;
    }
    crate::recon::update_temporal(
        &mut recon.cur_mvs[idx..],
        t_stride as usize,
        1,
        1,
        r,
        mv,
        swap,
    );
}

/// `inter_mc_plane_prep_8bpc` writing at an explicit offset into `tmp`
/// non-OPFL TIP per-8x8 prediction.
#[allow(clippy::too_many_arguments)]
fn inter_mc_plane_prep_at_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    exec: &crate::exec_context::ExecContext,
    tmp: &mut [i16],
    off: usize,
    tmp_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    bw4: i32,
    bh4: i32,
    mvx: i32,
    mvy: i32,
    filter: u8,
    ss_hor: i32,
    ss_ver: i32,
    cur_bw: i32,
    cur_bh: i32,
    inter_scratch: &mut Vec<i16>,
) {
    mc_prep_bounds_8bpc(
        bd,
        exec,
        &mut tmp[off..],
        tmp_stride,
        ref_pic,
        pl,
        bx,
        by,
        bw4,
        bh4,
        mvx,
        mvy,
        filter,
        ss_hor,
        ss_ver,
        0,
        cur_bw * 4 >> if pl != 0 { ss_hor } else { 0 },
        0,
        cur_bh * 4 >> if pl != 0 { ss_ver } else { 0 },
        inter_scratch,
    );
}

/// truncated to i8 (the difference weights live in `union aliasi16 d.i8[2]`).
#[inline]
fn apply_sign_i8(x: i32, s: i32) -> i8 {
    (if s < 0 { -x } else { x }) as i8
}

/// Bounds-aware prep-MC for the OPFL bilinear prefetch (`mc(..., left, right,
/// `inter_mc_plane_prep_8bpc` but with explicit edge limits (the OPFL reference
/// area cannot exceed the original MV's bounding box; the remaining pixels are
/// emulated). 3-bit subpel precision (same as `mc`). Writes into `tmp`
/// (stride `tmp_stride`) at the given block offset. Output buffer must be 8bpc
/// luma/chroma (this clip's refs are unscaled, so the scaled branch is omitted).
#[allow(clippy::too_many_arguments)]
fn mc_prep_bounds_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    exec: &crate::exec_context::ExecContext,
    tmp: &mut [i16],
    tmp_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    bw4: i32,
    bh4: i32,
    mvx: i32,
    mvy: i32,
    filter: u8,
    ss_hor: i32,
    ss_ver: i32,
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => s,
        None => return,
    };

    let mx = mvx & (15 >> (plss_hor == 0) as i32);
    let my = mvy & (15 >> (plss_ver == 0) as i32);
    let dx = bx * h_mul + (mvx >> (3 + plss_hor));
    let dy = by * v_mul + (mvy >> (3 + plss_ver));

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + bw4 * h_mul + (mx != 0) as i32 * 4 > right
        || dy + bh4 * v_mul + (my != 0) as i32 * 4 > bottom;

    let w = (bw4 * h_mul) as usize;
    let h = (bh4 * v_mul) as usize;
    let mxf = mx << (plss_hor == 0) as i32;
    let myf = my << (plss_ver == 0) as i32;
    let is_bilin = filter == 3;

    let mut emu_buf = if need_emu {
        Some(vec![BD::Pixel::default(); 192 * 192])
    } else {
        None
    };
    let (src, src_off, src_stride) = if let Some(ref mut buf) = emu_buf {
        let emu_w = w + (mx != 0) as usize * 7;
        let emu_h = h + (my != 0) as usize * 7;
        let emu_stride = 192usize;
        let src_base = ((left + top * ref_stride as i32).max(0) as usize).min(ref_data.len());
        inter_emu_edge_8bpc::<BD>(
            buf,
            emu_stride,
            &ref_data[src_base..],
            ref_stride,
            emu_w,
            emu_h,
            (right - left).max(0) as usize,
            (bottom - top).max(0) as usize,
            dx - (mx != 0) as i32 * 3 - left,
            dy - (my != 0) as i32 * 3 - top,
        );
        let off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
        (&buf[..], off, emu_stride)
    } else {
        let off = dy as usize * ref_stride + dx as usize;
        (ref_data, off, ref_stride)
    };

    if BD::BPC == 8 {
        // SAFETY: BPC==8 => BD::Pixel == u8.
        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
        if is_bilin {
            exec.prep_bilin_8bpc_with_scratch(
                tmp,
                tmp_stride,
                &src8[src_off..],
                src_stride,
                w,
                h,
                mxf,
                myf,
                inter_scratch,
            );
        } else {
            exec.prep_8tap_8bpc_with_scratch(
                tmp,
                tmp_stride,
                src8,
                src_off,
                src_stride,
                w,
                h,
                mxf,
                myf,
                filter as i32,
                inter_scratch,
            );
        }
    } else if is_bilin {
        crate::mc::prep_bilin(
            bd,
            tmp,
            tmp_stride,
            &src[src_off..],
            src_stride,
            w,
            h,
            mxf,
            myf,
        );
    } else {
        crate::mc::prep_8tap(
            bd,
            tmp,
            tmp_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mxf,
            myf,
            filter as i32,
        );
    }
}

/// 4-bit subpel MV precision and explicit edge limits; output is i16 (prep).
/// Unscaled refs only (this clip). Writes `bw4*4 x bh4*4` (after plane
/// subsampling) into `dst16` at `dst_off`, stride `dst_stride`.
#[allow(clippy::too_many_arguments)]
fn mc_opfl_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    exec: &crate::exec_context::ExecContext,
    dst16: &mut [i16],
    dst_off: usize,
    dst_stride: usize,
    bw4: i32,
    bh4: i32,
    bx4: i32,
    by4: i32,
    pl: usize,
    mvx: i32,
    mvy: i32,
    ref_pic: &crate::picture::Picture,
    filter: u8,
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => s,
        None => return,
    };
    let mx = mvx & 15;
    let my = mvy & 15;
    let dx = bx4 * 4 + (mvx >> 4);
    let dy = by4 * 4 + (mvy >> 4);
    let w = (bw4 * 4) as usize;
    let h = (bh4 * 4) as usize;
    let is_bilin = filter == 3;

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + bw4 * 4 + (mx != 0) as i32 * 4 > right
        || dy + bh4 * 4 + (my != 0) as i32 * 4 > bottom;

    let mut emu_buf = if need_emu {
        Some(vec![BD::Pixel::default(); 192 * 192])
    } else {
        None
    };
    let (src, src_off, src_stride) = if let Some(ref mut buf) = emu_buf {
        let emu_w = w + (mx != 0) as usize * 7;
        let emu_h = h + (my != 0) as usize * 7;
        let emu_stride = 192usize;
        let src_base = ((left + top * ref_stride as i32).max(0) as usize).min(ref_data.len());
        inter_emu_edge_8bpc::<BD>(
            buf,
            emu_stride,
            &ref_data[src_base..],
            ref_stride,
            emu_w,
            emu_h,
            (right - left).max(0) as usize,
            (bottom - top).max(0) as usize,
            dx - (mx != 0) as i32 * 3 - left,
            dy - (my != 0) as i32 * 3 - top,
        );
        let off = emu_stride * (my != 0) as usize * 3 + (mx != 0) as usize * 3;
        (&buf[..], off, emu_stride)
    } else {
        let off = dy as usize * ref_stride + dx as usize;
        (ref_data, off, ref_stride)
    };

    if BD::BPC == 8 {
        // SAFETY: BPC==8 => BD::Pixel == u8.
        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
        if is_bilin {
            exec.prep_bilin_8bpc_with_scratch(
                &mut dst16[dst_off..],
                dst_stride,
                &src8[src_off..],
                src_stride,
                w,
                h,
                mx,
                my,
                inter_scratch,
            );
        } else {
            exec.prep_8tap_8bpc_with_scratch(
                &mut dst16[dst_off..],
                dst_stride,
                src8,
                src_off,
                src_stride,
                w,
                h,
                mx,
                my,
                filter as i32,
                inter_scratch,
            );
        }
    } else if is_bilin {
        crate::mc::prep_bilin(
            bd,
            &mut dst16[dst_off..],
            dst_stride,
            &src[src_off..],
            src_stride,
            w,
            h,
            mx,
            my,
        );
    } else {
        crate::mc::prep_8tap(
            bd,
            &mut dst16[dst_off..],
            dst_stride,
            src,
            src_off,
            src_stride,
            w,
            h,
            mx,
            my,
            filter as i32,
        );
    }
}

/// Reconstruct a single-reference inter block (8bpc): motion-compensate luma +
/// chroma from the reference picture (translational or warp-affine, dispatched
/// on the block's motion_mode / derived warp params), then add the parsed
/// residual transforms. Compound (ref pair), interintra blend, TIP and scaled
/// references are deferred.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recon_b_inter<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    chroma_stage: ChromaPhase,
    fi: &SbFrameInfo,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let bd = recon.bd;
    let refs = b.ref_pair.refs();
    let ref0 = refs[0];
    let ref1 = refs[1];
    if ref0 as usize == crate::levels::TIP_FRAME {
        return recon_b_inter_tip(
            recon,
            msac,
            cdf_m,
            a,
            l,
            b,
            bx,
            by,
            cbx,
            cby,
            lbs,
            cbs,
            has_luma,
            has_chroma,
            chroma_stage,
            fi,
        );
    }
    if ref0 < 0 || ref0 as usize >= 7 {
        return Ok(());
    }

    // Blocks larger than 64px in either dimension are not reconstructed as a
    // single unit (the MC kernels cap at 64px wide): they are split into 64x64
    // chroma coefficients are read with the first luma sub-block and the chroma
    // is reconstructed once (read/recon staging), so the MSAC bitstream ordering
    // matches the C decoder. <=64px blocks fall through to the leaf below.
    {
        let bs0 = if lbs == BlockSize::Invalid { cbs } else { lbs };
        let bdim0 = &BLOCK_DIMENSIONS[bs0 as u8 as usize];
        let bw4_0 = bdim0[0] as i32;
        let bh4_0 = bdim0[1] as i32;
        if imax(bw4_0, bh4_0) > 16 {
            // The >64px split drives the per-sub-block chroma read/recon stage
            // itself, so it is only ever entered at the `Both` top level.
            debug_assert!(chroma_stage == ChromaPhase::Both);
            return recon_b_inter_split(
                recon, msac, cdf_m, a, l, b, bx, by, cbx, cby, lbs, cbs, has_luma, has_chroma, fi,
                bs0, bw4_0, bh4_0,
            );
        }
    }

    let mv = b.inter_data().mv[0].xy();
    let mv1 = b.inter_data().mv[1].xy();
    let filter = b.inter_data().filter;
    let comp_type = b.inter_data().comp_type;
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;

    // Compound (two-ref) prediction: predict both refs into i16 tmp buffers and
    // warp-compound are not present in the bring-up clip and are deferred.
    let is_compound = ref1 >= 0 && (ref1 as usize) < 7;
    if is_compound {
        return recon_b_inter_compound(
            recon,
            msac,
            cdf_m,
            a,
            l,
            b,
            bx,
            by,
            cbx,
            cby,
            lbs,
            cbs,
            has_luma,
            has_chroma,
            chroma_stage,
            fi,
        );
    }

    // Take the reference picture out of recon.refp (immutable Arc) to satisfy the
    // borrow checker while mutating dst planes.
    let refp = match recon.refp[ref0 as usize].clone() {
        Some(p) => p,
        None => return Ok(()),
    };
    let _ = (mv1, comp_type);

    let motion_mode = b.inter_data().motion_mode;
    let inter_mode = b.inter_data().inter_mode;
    let warp_block = {
        let bdim = &BLOCK_DIMENSIONS[lbs as u8 as usize];
        let bw4 = bdim[0] as i32;
        let bh4 = bdim[1] as i32;
        let mut gmv = recon.frm_hdr.gmv.m[ref0 as usize];
        let gmv_warp_allowed = gmv.wm_type > crate::headers::WarpedMotionType::Translation
            && recon.frm_hdr.force_integer_mv == 0
            && crate::warpmv::get_shear_params(&mut gmv) == 0
            && recon.svc[ref0 as usize][0].scale == 0;
        recon.frm_hdr.force_integer_mv == 0
            && ((inter_mode == crate::levels::InterPredMode::GlobalMv as u8
                && imin(bw4, bh4) > 1
                && gmv_warp_allowed)
                || (motion_mode >= MotionMode::WarpCausal as u8
                    && recon.warpmv[0].wm_type > crate::headers::WarpedMotionType::Invalid))
    };
    // Pick which warp params to use: local warpmv for warp motion modes, the
    // frame global motion otherwise (GLOBALMV warp).
    let use_local_warp = motion_mode >= MotionMode::WarpCausal as u8;

    if has_luma {
        let bs = lbs;
        let b_dim = &BLOCK_DIMENSIONS[bs as u8 as usize];
        let bw4 = b_dim[0] as i32;
        let bh4 = b_dim[1] as i32;
        let y_stride = recon.frame.y_stride_px;
        let dst_off = 4 * (by as usize * y_stride + bx as usize);
        let wmp = if use_local_warp {
            recon.warpmv[0]
        } else {
            recon.frm_hdr.gmv.m[ref0 as usize]
        };
        if warp_block {
            // applies for affine warps where the (subsampled) block is >= 8 in
            // both dims; otherwise ext_warp.
            if wmp.affine != 0 && imin(bw4 * 4, bh4 * 4) >= 8 {
                warp_affine_plane_8bpc(
                    bd,
                    recon.frame.exec,
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    &refp,
                    0,
                    bx,
                    by,
                    b_dim,
                    &wmp,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                );
            } else {
                ext_warp_plane_8bpc(
                    bd,
                    recon.frame.exec,
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    &refp,
                    0,
                    bx,
                    by,
                    b_dim,
                    &wmp,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                    recon.scratch.inter_mc_tmp_mut(),
                );
            }
        } else {
            inter_mc_plane_8bpc(
                bd,
                recon.frame.exec,
                &mut recon.dst_y[dst_off..],
                y_stride,
                &refp,
                0,
                bx,
                by,
                bw4,
                bh4,
                mv.x,
                mv.y,
                filter,
                ss_hor,
                ss_ver,
                fi.bw,
                fi.bh,
                recon.scratch.inter_mc_tmp_mut(),
            );
        }

        // over the inter-intra blend for single-ref blocks.
        let bawp0 = b.inter_data().bawp[0] as i32;
        if bawp0 != 0 {
            let w4c = imin(bw4, fi.bw - bx);
            let h4c = imin(bh4, fi.bh - by);
            bawp_plane(
                recon,
                bawp0,
                mv,
                dst_off,
                y_stride,
                &refp,
                ref0 as usize,
                0,
                bw4,
                bh4,
                w4c,
                h4c,
                bx,
                by,
                bs,
                fi,
            );
        } else if motion_mode == MotionMode::InterIntra as u8 || b.inter_data().warp_ii != 0 {
            // prediction over the inter prediction for INTERINTRA / warp-
            // interintra blocks.
            let dst_off_y = dst_off;
            // SAFETY: split the recon borrow — iiblend needs &mut recon for the
            // edge/masks while writing dst_y; pass dst_y through recon directly.
            iiblend_luma_8bpc(recon, &b, dst_off_y, y_stride, bw4, bh4, by, bx, bs, fi);
        }

        // Luma residual: walk b.tx_part geometry (same tp[] as intra,
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        if lossless {
            let tx = if b.tx_size_ll != 0 {
                crate::tables::MAX_TXFM_SIZE_FOR_BS[bs as usize][3] as usize
            } else {
                0
            };
            let t_dim = &TXFM_DIMENSIONS[tx];
            let (tw4, th4) = (t_dim.w as i32, t_dim.h as i32);
            let h4 = imin(bh4, fi.bh - by);
            let w4 = imin(bw4, fi.bw - bx);
            let mut y = 0;
            while y < h4 {
                let mut x = 0;
                while x < w4 {
                    inter_residual_tx_8bpc(
                        recon,
                        msac,
                        cdf_m,
                        a,
                        l,
                        b,
                        0,
                        tx,
                        bx + x,
                        by + y,
                        false,
                        0,
                        fi,
                    )?;
                    x += tw4;
                }
                y += th4;
            }
        } else {
            let tp = &crate::tables::TX_PART_TBL[bs as usize];
            let tx = tp[b.tx_part as usize] as usize;
            inter_luma_tx_walk(recon, msac, cdf_m, a, l, b, tx, bx, by, fi)?;
        }
    }

    if has_chroma && cbs != BlockSize::Invalid {
        let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
        let cbw4 = cb_dim[0] as i32;
        let cbh4 = cb_dim[1] as i32;
        let cw4 = imin(fi.bw - cbx, cbw4);
        let ch4 = imin(fi.bh - cby, cbh4);
        let cw4ss = (cw4 + ss_hor) >> ss_hor;
        let ch4ss = (ch4 + ss_ver) >> ss_ver;

        // Chroma MV: for cbs==lbs or imin(bw4,bh4)>=16 the single block MV is
        // used directly. For sub-8x8 luma coding (cbs != lbs && imin(bw4,bh4)<16)
        // the chroma covers several luma sub-blocks, each with its own MV; chroma
        // MC is done per luma sub-block reading ref/MV/filter from the spatial
        let uv_stride = recon.frame.uv_stride_px;
        let (luma_bw4, luma_bh4) = {
            let ld = &BLOCK_DIMENSIONS[lbs as u8 as usize];
            (ld[0] as i32, ld[1] as i32)
        };
        let sub8x8 =
            lbs != BlockSize::Invalid && cbs != lbs && imin(luma_bw4, luma_bh4) < 16 && !warp_block;
        // Chroma MC + iiblend run only at the recon stage (`Both`/`ReconOnly`);
        // the read stage (`ReadOnly`) only consumes the chroma coefficients.
        let do_chroma_mc = chroma_stage != ChromaPhase::ReadOnly;
        if sub8x8 && do_chroma_mc {
            // Per-sub-block chroma MC from spatial refmvs. cw4/ch4 are chroma
            // 4x4 extents; for each origin sub-block (ox4==oy4==0) MC both planes.
            let base = ((cby & 63) as usize) * 128 + ((cbx & 127) as usize);
            for y in 0..ch4 {
                for x in 0..cw4 {
                    let idx = base + (y as usize) * 128 + (x as usize);
                    let r2 = &recon.rt.r[idx];
                    if r2.ox4 != 0 || r2.oy4 != 0 {
                        continue;
                    }
                    let s_ref0 = r2.reference.ref_at(0);
                    if s_ref0 < 0 || s_ref0 as usize >= 7 {
                        continue;
                    }
                    let s_mv = if r2.mf & 2 != 0 {
                        r2.lmv[0].xy()
                    } else {
                        r2.mv[0].xy()
                    };
                    let s_filter = r2.subpel_filter;
                    let sdim = &BLOCK_DIMENSIONS[r2.bs as usize];
                    let s_bw4 = sdim[0] as i32;
                    let s_bh4 = sdim[1] as i32;
                    let s_refp = match recon.refp[s_ref0 as usize].clone() {
                        Some(p) => p,
                        None => continue,
                    };
                    let s_cbx = cbx + x;
                    let s_cby = cby + y;
                    // Chroma destination pixel: block-origin chroma px plus the
                    // uvoff advances by 4*stride >> ss_ver per luma-4-unit row).
                    let base_off =
                        4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
                    let dst_off = base_off
                        + (((y * 4) >> ss_ver) as usize) * uv_stride
                        + (((x * 4) >> ss_hor) as usize);
                    for pl in 1..3usize {
                        let dst: &mut [BD::Pixel] = if pl == 1 {
                            &mut recon.dst_u[dst_off..]
                        } else {
                            &mut recon.dst_v[dst_off..]
                        };
                        inter_mc_plane_8bpc(
                            bd,
                            recon.frame.exec,
                            dst,
                            uv_stride,
                            &s_refp,
                            pl,
                            s_cbx,
                            s_cby,
                            s_bw4,
                            s_bh4,
                            s_mv.x,
                            s_mv.y,
                            s_filter,
                            ss_hor,
                            ss_ver,
                            fi.bw,
                            fi.bh,
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                }
            }
        }
        // Warp dispatch for chroma uses cb_dim and (>=8px-after-subsample affine).
        let c_wmp = if use_local_warp {
            recon.warpmv[0]
        } else {
            recon.frm_hdr.gmv.m[ref0 as usize]
        };
        // Chroma warp eligibility for the 8x8 affine kernel uses the chroma
        let c_affine = c_wmp.affine != 0 && imin(cbw4 * (4 >> ss_hor), cbh4 * (4 >> ss_ver)) >= 8;
        for pl in (1..3).filter(|_| !sub8x8 && do_chroma_mc) {
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            let dst: &mut [BD::Pixel] = if pl == 1 {
                &mut recon.dst_u[dst_off..]
            } else {
                &mut recon.dst_v[dst_off..]
            };
            if warp_block {
                if c_affine {
                    warp_affine_plane_8bpc(
                        bd,
                        recon.frame.exec,
                        dst,
                        uv_stride,
                        &refp,
                        pl,
                        cbx,
                        cby,
                        cb_dim,
                        &c_wmp,
                        ss_hor,
                        ss_ver,
                        fi.bw,
                        fi.bh,
                    );
                } else {
                    ext_warp_plane_8bpc(
                        bd,
                        recon.frame.exec,
                        dst,
                        uv_stride,
                        &refp,
                        pl,
                        cbx,
                        cby,
                        cb_dim,
                        &c_wmp,
                        ss_hor,
                        ss_ver,
                        fi.bw,
                        fi.bh,
                        recon.scratch.inter_mc_tmp_mut(),
                    );
                }
            } else {
                inter_mc_plane_8bpc(
                    bd,
                    recon.frame.exec,
                    dst,
                    uv_stride,
                    &refp,
                    pl,
                    cbx,
                    cby,
                    cbw4,
                    cbh4,
                    mv.x,
                    mv.y,
                    filter,
                    ss_hor,
                    ss_ver,
                    fi.bw,
                    fi.bh,
                    recon.scratch.inter_mc_tmp_mut(),
                );
            }
        }

        // and before the residual; takes priority over the chroma inter-intra
        // blend. Chroma always passes bawp_idx=1 and reuses the luma alpha.
        let chroma_bawp = !sub8x8 && do_chroma_mc && b.inter_data().bawp[1] != 0;
        if chroma_bawp {
            let cw4c = imin(cbw4, fi.bw - cbx);
            let ch4c = imin(cbh4, fi.bh - cby);
            let blk_bs = BlockSize::from_raw(b.bs);
            for pl in 1..3usize {
                let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
                bawp_plane(
                    recon,
                    1,
                    mv,
                    dst_off,
                    uv_stride,
                    &refp,
                    ref0 as usize,
                    pl,
                    cbw4,
                    cbh4,
                    cw4c,
                    ch4c,
                    cbx,
                    cby,
                    blk_bs,
                    fi,
                );
            }
        }

        // single-ref / compound branches, never on sub8x8 chroma coding (the
        // is the chroma-subsampled block size SS_BS[cbs][layout-1] (wedge keeps
        if !sub8x8
            && !chroma_bawp
            && do_chroma_mc
            && (motion_mode == MotionMode::InterIntra as u8 || b.inter_data().warp_ii != 0)
        {
            let ii_ss_bs = if b.inter_data().wedge_idx == -1 {
                let layout_idx = (recon.frame.layout as usize) - 1;
                BlockSize::from_raw(crate::tables::SS_BS[cbs as usize][layout_idx] as i8)
            } else {
                cbs
            };
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            for pl in 1..3usize {
                iiblend_chroma_8bpc(
                    recon, &b, pl, dst_off, uv_stride, cbw4, cbh4, cby, cbx, ii_ss_bs, fi,
                );
            }
        }

        // Chroma residual per uvtx TU (both planes + CCTX).
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        let uvtx = if lossless {
            0usize
        } else {
            let layout_idx =
                (crate::headers::PixelLayout::I444 as i32 - recon.frame.layout as i32) as usize;
            crate::tables::MAX_TXFM_SIZE_FOR_BS[cbs as u8 as usize][layout_idx] as usize
        };
        // Seed the inter chroma transform type from the co-located luma 4x4's
        let uv_txtp_seed = recon.scratch.txtp_map[(by & 15) as usize * 16 + (bx & 15) as usize];
        let cbw4ss = (cbw4 + ss_hor) >> ss_hor;
        let cbh4ss = (cbh4 + ss_ver) >> ss_ver;
        inter_chroma_residual_8bpc(
            recon,
            msac,
            cdf_m,
            a,
            l,
            b,
            uvtx,
            cbx,
            cby,
            cbw4ss,
            cbh4ss,
            cw4ss,
            ch4ss,
            uv_txtp_seed,
            chroma_stage,
            fi,
        )?;
    }
    Ok(())
}

/// Split an inter block larger than 64px into 64x64 (or 128x128 for 256px)
/// its luma MC + residual; chroma is decoded once via the read/recon staging
/// (`cbs2[0]` reads, `cbs2[1]` reconstructs). Because the chroma sub-block uses
/// the full chroma block size at the block-origin coordinates for both stages,
/// we perform the whole chroma (MC + residual) at the read stage so the MSAC
/// ordering (chroma coefs follow the first luma sub-block) matches the C decoder.
#[allow(clippy::too_many_arguments)]
fn recon_b_inter_split<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    fi: &SbFrameInfo,
    bs: BlockSize,
    bw4: i32,
    bh4: i32,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    static CSPLIT: [[BlockSize; 3]; 3] = [
        [
            BlockSize::Bs64x64,
            BlockSize::Bs128x64,
            BlockSize::Bs128x128,
        ],
        [BlockSize::Bs64x64, BlockSize::Bs128x64, BlockSize::Bs128x64],
        [BlockSize::Bs64x64, BlockSize::Bs64x64, BlockSize::Bs64x128],
    ];
    let ss_hor = fi.ss_hor;
    let ss_ver = fi.ss_ver;
    let y_end = imin(by + bh4, fi.bh);
    let x_end = imin(bx + bw4, fi.bw);
    let (step, lbs2, cbs2i) = if imax(bw4, bh4) == 64 {
        (
            32,
            if lbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                BlockSize::Bs128x128
            },
            if cbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                BlockSize::Bs128x128
            },
        )
    } else {
        let csplit_row = (bs as i32 - BlockSize::Bs128x128 as i32) as usize;
        let csi = (ss_hor + ss_ver) as usize;
        (
            16,
            if lbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                BlockSize::Bs64x64
            },
            if cbs == BlockSize::Invalid {
                BlockSize::Invalid
            } else {
                CSPLIT[csplit_row][csi]
            },
        )
    };
    let _ = (has_luma, has_chroma);

    let mut sub_by = by;
    let mut sub_cby = cby;
    let mut yy = 0;
    while sub_by < y_end {
        let mut sub_bx = bx;
        let mut sub_cbx = cbx;
        let mut xx = 0;
        while sub_bx < x_end {
            // cbs2[0] = chroma coef-read stage, cbs2[1] = chroma recon stage
            let (read_cbs, recon_cbs) = if step == 32 {
                (cbs2i, cbs2i)
            } else {
                let read = if ((xx & ss_hor) | (yy & ss_ver)) == 0 {
                    cbs2i
                } else {
                    BlockSize::Invalid
                };
                let recon = if (ss_hor == 0 || sub_bx + step >= x_end)
                    && (ss_ver == 0 || sub_by + step >= y_end)
                {
                    cbs2i
                } else {
                    BlockSize::Invalid
                };
                (read, recon)
            };

            // Map the read/recon stage validity onto the chroma phase: the
            // coefficients are read with the first luma sub-block (`ReadOnly`)
            // and the residual + MC applied with the last (`ReconOnly`), so the
            // chroma prediction sees the spatial refmvs grid in its final state
            let read_valid = read_cbs != BlockSize::Invalid;
            let recon_valid = recon_cbs != BlockSize::Invalid;
            let sub_has_chroma = read_valid || recon_valid;
            let chroma_stage = match (read_valid, recon_valid) {
                (true, true) => ChromaPhase::Both,
                (true, false) => ChromaPhase::ReadOnly,
                (false, true) => ChromaPhase::ReconOnly,
                (false, false) => ChromaPhase::Both,
            };
            let sub_cbs = if sub_has_chroma {
                cbs2i
            } else {
                BlockSize::Invalid
            };

            recon_b_inter(
                recon,
                msac,
                cdf_m,
                a,
                l,
                b,
                sub_bx,
                sub_by,
                sub_cbx,
                sub_cby,
                lbs2,
                sub_cbs,
                lbs2 != BlockSize::Invalid,
                sub_has_chroma,
                chroma_stage,
                fi,
            )?;

            sub_bx += step;
            if step == 32 {
                sub_cbx += step;
            } else if (xx & ss_hor) == ss_hor {
                sub_cbx += step << ss_hor;
            }
            xx += 1;
        }
        sub_by += step;
        if step == 32 {
            sub_cby += step;
        } else if (yy & ss_ver) == ss_ver {
            sub_cby += step << ss_ver;
        }
        yy += 1;
    }
    Ok(())
}

/// type defines the visitation order and per-tile transform size, which is
/// load-bearing for the per-4x4 coefficient neighbour context (and hence the
/// entropy stream). A naive raster tiling desyncs for non-square partitions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inter_luma_tx_walk<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    tx: usize,
    bx: i32,
    by: i32,
    fi: &SbFrameInfo,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let tp = &crate::tables::TX_PART_TBL[b.bs as usize];
    macro_rules! resid {
        ($tx:expr, $x:expr, $y:expr) => {
            inter_residual_tx_8bpc(recon, msac, cdf_m, a, l, b, 0, $tx, $x, $y, false, 0, fi)?
        };
    }
    match TxPartition::from_raw(b.tx_part) {
        TxPartition::None => {
            resid!(tx, bx, by);
        }
        TxPartition::Split => {
            let t_dim = &TXFM_DIMENSIONS[tx];
            let (tw4, th4) = (t_dim.w as i32, t_dim.h as i32);
            resid!(tx, bx, by);
            let have_v_split = bx + tw4 < fi.bw;
            if have_v_split {
                resid!(tx, bx + tw4, by);
            }
            if by + th4 >= fi.bh {
                return Ok(());
            }
            resid!(tx, bx, by + th4);
            if have_v_split {
                resid!(tx, bx + tw4, by + th4);
            }
        }
        TxPartition::H => {
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            resid!(tx, bx, by);
            if by + th4 >= fi.bh {
                return Ok(());
            }
            resid!(tx, bx, by + th4);
        }
        TxPartition::V => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            resid!(tx, bx, by);
            if bx + tw4 >= fi.bw {
                return Ok(());
            }
            resid!(tx, bx + tw4, by);
        }
        TxPartition::H4 => {
            let th4 = TXFM_DIMENSIONS[tx].h as i32;
            for i in 0..4 {
                let yy = by + i * th4;
                resid!(tx, bx, yy);
                if yy + th4 >= fi.bh {
                    break;
                }
            }
        }
        TxPartition::V4 => {
            let tw4 = TXFM_DIMENSIONS[tx].w as i32;
            for i in 0..4 {
                let xx = bx + i * tw4;
                resid!(tx, xx, by);
                if xx + tw4 >= fi.bw {
                    break;
                }
            }
        }
        TxPartition::H5 => {
            let tx_big = tp[TxPartition::H as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let (tw4_small, th4_small) = (t_dim_small.w as i32, t_dim_small.h as i32);
            let th4_big = TXFM_DIMENSIONS[tx_big].h as i32;
            resid!(tx, bx, by);
            let have_v_split = bx + tw4_small < fi.bw;
            if have_v_split {
                resid!(tx, bx + tw4_small, by);
            }
            if by + th4_small >= fi.bh {
                return Ok(());
            }
            resid!(tx_big, bx, by + th4_small);
            if by + th4_small + th4_big < fi.bh {
                resid!(tx, bx, by + th4_small + th4_big);
                if have_v_split {
                    resid!(tx, bx + tw4_small, by + th4_small + th4_big);
                }
            }
        }
        TxPartition::V5 => {
            let tx_big = tp[TxPartition::V as usize] as usize;
            let t_dim_small = &TXFM_DIMENSIONS[tx];
            let (tw4_small, th4_small) = (t_dim_small.w as i32, t_dim_small.h as i32);
            let tw4_big = TXFM_DIMENSIONS[tx_big].w as i32;
            resid!(tx, bx, by);
            let have_h_split = by + th4_small < fi.bh;
            if have_h_split {
                resid!(tx, bx, by + th4_small);
            }
            if bx + tw4_small >= fi.bw {
                return Ok(());
            }
            resid!(tx_big, bx + tw4_small, by);
            if bx + tw4_small + tw4_big < fi.bw {
                resid!(tx, bx + tw4_small + tw4_big, by);
                if have_h_split {
                    resid!(tx, bx + tw4_small + tw4_big, by + th4_small);
                }
            }
        }
    }
    Ok(())
}
