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
use crate::intra::intrabc_morph_pred_luma;
use crate::levels::{Av2Block, BlockSize, CompInterPredMode, MotionMode, Mv, RefPair};
use crate::msac::MsacReader;
use crate::pixel::Pixel;
use crate::tables::{BLOCK_DIMENSIONS, TXFM_DIMENSIONS};

/// Shared per-block reconstruction state threaded through the intra `recon_b_*`
/// subtree. Replaces the seven leading arguments (`recon, msac, cdf_m, a, l, b,
/// fi`) that every function in the tree passed identically, so each call moves
/// one pointer instead of seven. Each function reborrows the fields into locals
/// of the original names, leaving its body unchanged.
pub(crate) struct ReconBCtx<
    'r,
    'a,
    'f,
    BD: BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
> {
    pub(crate) recon: &'r mut ReconCtx<'a, 'f, BD>,
    pub(crate) msac: &'r mut M,
    pub(crate) cdf_m: &'r mut CdfModeContext,
    pub(crate) a: &'r mut BlockContext,
    pub(crate) l: &'r mut BlockContext,
    pub(crate) b: &'r Av2Block,
    pub(crate) fi: &'r SbFrameInfo,
}

pub(crate) fn recon_b_intra<BD: BitDepth, const UPDATE_CDF: bool, M: MsacReader<UPDATE_CDF>>(
    rb: &mut ReconBCtx<'_, '_, '_, BD, UPDATE_CDF, M>,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    recon_b_intra_phase(
        rb,
        bx,
        by,
        cbx,
        cby,
        lbs,
        cbs,
        has_luma,
        has_chroma,
        TxPhase::Both,
        ChromaPhase::Both,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn recon_b_intra_phase<
    BD: crate::pixel::BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    rb: &mut ReconBCtx<'_, '_, '_, BD, UPDATE_CDF, M>,
    bx: i32,
    by: i32,
    cbx: i32,
    cby: i32,
    lbs: BlockSize,
    cbs: BlockSize,
    has_luma: bool,
    has_chroma: bool,
    luma_phase: TxPhase,
    chroma_outer_phase: ChromaPhase,
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
    let bs = if lbs == BlockSize::Invalid { cbs } else { lbs };
    let b_dim = &BLOCK_DIMENSIONS[bs as u8 as usize];
    let bw4 = b_dim[0] as i32;
    let bh4 = b_dim[1] as i32;

    if imax(bw4, bh4) > 16 {
        // Split into 64x64 (or 128x128) sub-blocks. csplit[bs - 128x128][ss].
        static CSPLIT: [[BlockSize; 3]; 3] = [
            // BS_128x128
            [
                BlockSize::Bs64x64,
                BlockSize::Bs128x64,
                BlockSize::Bs128x128,
            ],
            // BS_128x64
            [BlockSize::Bs64x64, BlockSize::Bs128x64, BlockSize::Bs128x64],
            // BS_64x128
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

        let mut sub_by = by;
        let mut sub_cby = cby;
        let mut yy = 0;
        while sub_by < y_end {
            let mut sub_bx = bx;
            let mut sub_cbx = cbx;
            let mut xx = 0;
            while sub_bx < x_end {
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

                if imax(
                    BLOCK_DIMENSIONS[lbs2 as u8 as usize][0] as i32,
                    BLOCK_DIMENSIONS[lbs2 as u8 as usize][1] as i32,
                ) > 16
                {
                    // 256px case: recurse one more level (lbs2 == 128x128).
                    recon_b_intra_phase(
                        &mut ReconBCtx {
                            recon: &mut *recon,
                            msac: &mut *msac,
                            cdf_m: &mut *cdf_m,
                            a: &mut *a,
                            l: &mut *l,
                            b,
                            fi,
                        },
                        sub_bx,
                        sub_by,
                        sub_cbx,
                        sub_cby,
                        lbs2,
                        if read_cbs != BlockSize::Invalid {
                            read_cbs
                        } else {
                            recon_cbs
                        },
                        lbs2 != BlockSize::Invalid,
                        read_cbs != BlockSize::Invalid || recon_cbs != BlockSize::Invalid,
                        luma_phase,
                        chroma_outer_phase,
                    )?;
                } else {
                    // Luma 64x64 sub-block: the tx walk uses the sub-block size,
                    // but `b.bs` (passed to coef decode) stays the full block.
                    if lbs2 != BlockSize::Invalid {
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
                            sub_bx,
                            sub_by,
                            lbs2 as usize,
                            luma_phase,
                        )?;
                    }
                    // Chroma: read phase with the first sub-block, recon with the last.
                    let phase = match (
                        read_cbs != BlockSize::Invalid,
                        recon_cbs != BlockSize::Invalid,
                    ) {
                        (true, true) => Some(ChromaPhase::Both),
                        (true, false) => Some(ChromaPhase::ReadOnly),
                        (false, true) => Some(ChromaPhase::ReconOnly),
                        (false, false) => None,
                    };
                    if let Some(ph) =
                        phase.and_then(|ph| chroma_phase_intersect(ph, chroma_outer_phase))
                    {
                        let ccbs = if read_cbs != BlockSize::Invalid {
                            read_cbs
                        } else {
                            recon_cbs
                        };
                        let sdp_active = lbs2 == BlockSize::Invalid;
                        recon_b_intra_chroma_phase(
                            &mut ReconBCtx {
                                recon: &mut *recon,
                                msac: &mut *msac,
                                cdf_m: &mut *cdf_m,
                                a: &mut *a,
                                l: &mut *l,
                                b,
                                fi,
                            },
                            sub_cbx,
                            sub_cby,
                            ccbs,
                            sdp_active,
                            ph,
                        )?;
                    }
                }

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
        return Ok(());
    }

    // Leaf: ordinary <=64px block.
    let intrabc = b.intrabc != 0;
    if has_luma {
        let bx4 = (bx & 63) as usize;
        let by4 = (by & 63) as usize;
        // IntraBC: copy the prediction from the current frame at the block
        if intrabc {
            let mv = b.intra_data().intrabc_mv.xy();
            crate::recon::intrabc_pred(
                recon.bd,
                recon.dst_y,
                recon.frame.y_stride_px,
                bw4,
                bh4,
                bx,
                by,
                mv.x as i32,
                mv.y as i32,
                0,
                0,
                fi.bw * 4,
                fi.bh * 4,
            );
            if b.intra_data().morph_pred != 0 {
                intrabc_morph_pred_luma(
                    recon.bd,
                    recon.dst_y,
                    recon.frame.y_stride_px,
                    bw4,
                    bh4,
                    bx,
                    by,
                    mv.x as i32,
                    mv.y as i32,
                    fi.bw * 4,
                    fi.bh * 4,
                );
            }
        }
        recon_b_intra_luma_phase(
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
            bx4,
            by4,
            intrabc,
            luma_phase,
        )?;
    }
    if has_chroma {
        if intrabc {
            let mv = b.intra_data().intrabc_mv.xy();
            let cb_dim = &BLOCK_DIMENSIONS[cbs as u8 as usize];
            let cbw4 = cb_dim[0] as i32;
            let cbh4 = cb_dim[1] as i32;
            let bd = recon.bd;
            for pl in 0..2 {
                let dst_plane: &mut [BD::Pixel] = if pl == 0 { recon.dst_u } else { recon.dst_v };
                crate::recon::intrabc_pred(
                    bd,
                    dst_plane,
                    recon.frame.uv_stride_px,
                    cbw4,
                    cbh4,
                    cbx,
                    cby,
                    mv.x as i32,
                    mv.y as i32,
                    fi.ss_hor,
                    fi.ss_ver,
                    (fi.bw * 4) >> fi.ss_hor,
                    (fi.bh * 4) >> fi.ss_ver,
                );
            }
        }
        let sdp_active = lbs == BlockSize::Invalid;
        recon_b_intra_chroma_phase(
            &mut ReconBCtx {
                recon: &mut *recon,
                msac: &mut *msac,
                cdf_m: &mut *cdf_m,
                a: &mut *a,
                l: &mut *l,
                b,
                fi,
            },
            cbx,
            cby,
            cbs,
            sdp_active,
            chroma_outer_phase,
        )?;
    }
    Ok(())
}

/// translational MC prediction is written to `dst`, BAWP rescales it by a
/// per-block linear model `dst = clip((alpha*dst + beta) >> 8)`. The model
/// `(alpha, beta)` is derived from the reconstructed neighbour template (rows
/// above / columns left of the block) versus the corresponding reference-plane
/// template. The luma plane derives `(alpha, beta)`; chroma reuses the luma
/// `alpha`. `plane` is 0 for luma, 1 for U, 2 for V. `bawp_idx` is the parsed
/// per-block index (luma); chroma always passes 1.
///
/// Edge handling assumes a single tile (`col_start == row_start == 0`) and the
/// single-thread, full-frame-alias decode used by the conformance harness, so
/// the top template row resolves to `dst[-stride]` in both the in-SB and the
#[allow(clippy::too_many_arguments)]
pub(crate) fn bawp_plane<BD: BitDepth>(
    recon: &mut ReconCtx<BD>,
    bawp_idx: i32,
    mv: crate::levels::MvXY,
    dst_off: usize,
    stride: usize,
    ref_pic: &crate::picture::Picture,
    refidx: usize,
    plane: usize,
    bw4: i32,
    bh4: i32,
    w4: i32,
    h4: i32,
    bx: i32,
    by: i32,
    sb_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    let bd = recon.bd;
    let chroma = plane != 0;
    let ss_hor = if chroma { fi.ss_hor } else { 0 };
    let ss_ver = if chroma { fi.ss_ver } else { 0 };
    let h_mul = 4 >> ss_hor;
    let v_mul = 4 >> ss_ver;
    let sb_dim = &BLOCK_DIMENSIONS[sb_bs as u8 as usize];
    let sb_dim0 = sb_dim[0] as i32;
    let sb_dim1 = sb_dim[1] as i32;

    let dst: &mut [BD::Pixel] = match plane {
        0 => recon.dst_y,
        1 => recon.dst_u,
        _ => recon.dst_v,
    };

    // >64px partition sub-blocks reuse the partition's first-block model.
    if (sb_dim0 > (16 << ss_hor) && (bx & (sb_dim0 - 1)) != 0)
        || (sb_dim1 > (16 << ss_ver) && (by & (sb_dim1 - 1)) != 0)
    {
        let (alpha, beta) = recon.bawp_ab[plane];
        if alpha != 256 || beta != 0 {
            crate::mc::morph(
                bd,
                &mut dst[dst_off..],
                stride,
                alpha,
                beta,
                (bw4 * h_mul) as usize,
                (bh4 * v_mul) as usize,
            );
        }
        return;
    }

    // defaults
    recon.bawp_ab[plane] = (256, 0);

    // Inter BAWP (refp != cur): tile edges span the whole frame.
    let tile_top_edge = 0i32;
    let tile_left_edge = 0i32;
    let tile_bottom_edge = fi.bh * v_mul;
    let tile_right_edge = fi.bw * h_mul;

    let mvx = (mv.x + 3 + (mv.x >= 0) as i32) >> (3 + ss_hor);
    let mvy = (mv.y + 3 + (mv.y >= 0) as i32) >> (3 + ss_ver);
    let ref_y = by * v_mul + mvy;
    let ref_x = bx * h_mul + mvx;
    let ref_tmplt_x = ref_x - 1;
    let ref_tmplt_y = ref_y - 1;
    let sb_w4 = imin(sb_dim0, fi.bw - bx);
    let sb_h4 = imin(sb_dim1, fi.bh - by);
    let ref_bottom_edge = ref_y + sb_h4 * v_mul;
    let ref_right_edge = ref_x + sb_w4 * h_mul;

    let can_morph = ref_bottom_edge <= tile_bottom_edge
        && ref_right_edge <= tile_right_edge
        && ref_tmplt_y >= tile_top_edge
        && ref_tmplt_x >= tile_left_edge;
    if !can_morph {
        return;
    }

    // n_edge_samples[have_above && have_left][lh4][lw4][above, left]
    static N_EDGE_SAMPLES: [[[[u8; 2]; 3]; 3]; 2] = [
        [
            [[2, 2], [3, 2], [4, 2]],
            [[2, 3], [3, 3], [4, 3]],
            [[2, 4], [3, 4], [4, 4]],
        ],
        [
            [[2, 2], [2, 2], [4, 0]],
            [[2, 2], [3, 3], [3, 3]],
            [[0, 4], [3, 3], [4, 4]],
        ],
    ];
    // Single tile: col_start == row_start == 0.
    let have_left = bx > 0;
    let have_above = by > 0;
    let lw4 = (imin(crate::intops::ulog2(w4 as u32), 2) - ss_hor) as usize;
    let lh4 = (imin(crate::intops::ulog2(h4 as u32), 2) - ss_ver) as usize;
    let idx = (have_above && have_left) as usize;
    let n_above_l2 = have_above as i32 * N_EDGE_SAMPLES[idx][lh4][lw4][0] as i32;
    let n_left_l2 = have_left as i32 * N_EDGE_SAMPLES[idx][lh4][lw4][1] as i32;

    let ref_stride =
        ref_pic.stride[(plane != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_base = match ref_pic.plane_slice::<BD::Pixel>(plane) {
        Some(s) => s,
        None => return,
    };
    let ref_off = ref_y as usize * ref_stride + ref_x as usize;

    debug_assert!(n_above_l2 == 0 || n_left_l2 == 0 || n_above_l2 == n_left_l2);
    let count_l2 = n_above_l2
        + if n_above_l2 == n_left_l2 {
            (n_above_l2 != 0) as i32
        } else {
            n_left_l2
        };
    let mut sum_x: i32 = 0;
    let mut sum_y: i32 = 0;
    let mut sum_xy: i32 = 0;
    let mut sum_x2: i32 = 0;
    if n_above_l2 != 0 {
        let bw = 4 << lw4;
        let step = bw >> n_above_l2;
        let start = step >> 1;
        // Single-thread full-frame alias: the SB-edge top row resolves to the
        let top_off = (dst_off as isize - stride as isize) as usize;
        let mut i = start;
        while i < bw {
            let x: i32 = ref_base[ref_off - ref_stride + i as usize].into();
            let y: i32 = dst[top_off + i as usize].into();
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            i += step;
        }
    }
    if n_left_l2 != 0 {
        let bh = 4 << lh4;
        let step = bh >> n_left_l2;
        let start = step >> 1;
        let mut i = start;
        while i < bh {
            let x: i32 = ref_base[ref_off + (i as usize) * ref_stride - 1].into();
            let y: i32 =
                dst[(dst_off as isize + (i as isize) * stride as isize - 1) as usize].into();
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            i += step;
        }
    }

    let alpha: i32 = if chroma {
        if have_left || have_above {
            recon.bawp_ab[0].0
        } else {
            256
        }
    } else if bawp_idx != 1 {
        debug_assert!(bawp_idx & 2 != 0);
        let aidx = (1 + (bawp_idx >> 2) + (fi.absrefdist[refidx] as i32 > 4) as i32)
            * (if bawp_idx & 1 != 0 { 1 } else { -1 });
        256 + 16 * aidx
    } else if count_l2 != 0 {
        let num = sum_xy - (((sum_x as i64) * (sum_y as i64)) >> count_l2) as i32;
        let den = sum_x2 - (((sum_x as i64) * (sum_x as i64)) >> count_l2) as i32;
        crate::recon::derive_alpha(num, den, 256)
    } else {
        256
    };
    recon.bawp_ab[plane].0 = alpha;

    let beta: i32 = if count_l2 != 0 {
        let diff = (sum_y << 8) - sum_x * alpha;
        crate::intops::apply_sign(diff.abs() >> count_l2, diff)
    } else {
        -128
    };
    recon.bawp_ab[plane].1 = beta;

    crate::mc::morph(
        bd,
        &mut dst[dst_off..],
        stride,
        alpha,
        beta,
        (bw4 * h_mul) as usize,
        (bh4 * v_mul) as usize,
    );
}

/// Motion-compensate one plane of a single reference into `dst` (8bpc), mirroring
/// block MV (1/8-pel luma units). Uses the proper separable 8-tap / bilinear
/// primitives from `mc.rs`. Scaled references (svc.scale != 0) are not handled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inter_mc_plane_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
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
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: (&[BD::Pixel], i32, i32) = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => {
            let pw = if pl == 0 {
                ref_pic.p.w
            } else {
                (ref_pic.p.w + ss_hor) >> ss_hor
            };
            let ph = if pl == 0 {
                ref_pic.p.h
            } else {
                (ref_pic.p.h + ss_ver) >> ss_ver
            };
            (s, pw, ph)
        }
        None => return,
    };
    let (ref_data, ref_pw, ref_ph) = ref_data;

    let left = 0i32;
    let top = 0i32;
    let right = cur_bw * 4 >> plss_hor;
    let bottom = cur_bh * 4 >> plss_ver;

    let mx = mvx & (15 >> (plss_hor == 0) as i32);
    let my = mvy & (15 >> (plss_ver == 0) as i32);
    let dx = bx * h_mul + (mvx >> (3 + plss_hor));
    let dy = by * v_mul + (mvy >> (3 + plss_ver));

    // then equal the reference plane size. Use the reference's own dimensions as
    // the clamp bounds so a malformed stream that points into a smaller/larger
    // reference can never read past its buffer. For valid streams cur == ref, so
    // these equal `right`/`bottom` and the result is unchanged.
    let iw = imin(right, ref_pw);
    let ih = imin(bottom, ref_ph);

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + bw4 * h_mul + (mx != 0) as i32 * 4 > right
        || dy + bh4 * v_mul + (my != 0) as i32 * 4 > bottom
        // Force emulation if the reference dimensions differ from the current
        // frame (scaled refs are otherwise unhandled) so the direct read below
        // cannot overflow a smaller reference buffer.
        || ref_pw != right
        || ref_ph != bottom;

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
        inter_emu_edge_8bpc::<BD>(
            buf,
            emu_stride,
            ref_data,
            ref_stride,
            emu_w,
            emu_h,
            (iw - left) as usize,
            (ih - top) as usize,
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
        // SAFETY: BPC==8 => BD::Pixel == u8; reinterpret slices to call the
        // byte-identical NEON 8bpc kernels.
        let dst8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(dst);
        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
        if is_bilin {
            crate::mc_dispatch::put_bilin_8bpc_with_scratch(
                dst8,
                dst_stride,
                &src8[src_off..],
                src_stride,
                w,
                h,
                mxf,
                myf,
                inter_scratch,
            );
        } else {
            crate::mc_dispatch::put_8tap_8bpc_with_scratch(
                dst8,
                dst_stride,
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
    } else if let (Some(dst16), Some(src16)) = (
        <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst),
        <BD::Pixel as Pixel>::try_as_u16_slice(src),
    ) {
        if is_bilin {
            crate::mc_dispatch::put_bilin_hbd_with_scratch(
                dst16,
                dst_stride,
                &src16[src_off..],
                src_stride,
                w,
                h,
                mxf,
                myf,
                bd.bitdepth(),
                inter_scratch,
            );
        } else {
            crate::mc_dispatch::put_8tap_hbd_with_scratch(
                dst16,
                dst_stride,
                src16,
                src_off,
                src_stride,
                w,
                h,
                mxf,
                myf,
                filter as i32,
                bd.bitdepth(),
                inter_scratch,
            );
        }
    }
}

/// with `dst == NULL`). Mirrors `inter_mc_plane_8bpc` but uses the `prep`
/// kernels (no final shift to pixels) so the result can be blended by the
/// compound `avg`/`w_avg`/`mask`/`w_mask` kernels. `tmp` is laid out at stride
#[allow(clippy::too_many_arguments)]
fn inter_mc_plane_prep_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    tmp: &mut [i16],
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
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: (&[BD::Pixel], i32, i32) = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => {
            let pw = if pl == 0 {
                ref_pic.p.w
            } else {
                (ref_pic.p.w + ss_hor) >> ss_hor
            };
            let ph = if pl == 0 {
                ref_pic.p.h
            } else {
                (ref_pic.p.h + ss_ver) >> ss_ver
            };
            (s, pw, ph)
        }
        None => return,
    };
    let (ref_data, ref_pw, ref_ph) = ref_data;

    let left = 0i32;
    let top = 0i32;
    let right = cur_bw * 4 >> plss_hor;
    let bottom = cur_bh * 4 >> plss_ver;

    let mx = mvx & (15 >> (plss_hor == 0) as i32);
    let my = mvy & (15 >> (plss_ver == 0) as i32);
    let dx = bx * h_mul + (mvx >> (3 + plss_hor));
    let dy = by * v_mul + (mvy >> (3 + plss_ver));

    // See inter_mc_plane_8bpc: clamp the emu bounds to the reference plane size
    // and force emulation when the reference dimensions differ from the current
    // frame, so a malformed reference cannot be read out of bounds. No-op for
    // valid streams where the reference and current frame match.
    let iw = imin(right, ref_pw);
    let ih = imin(bottom, ref_ph);

    let need_emu = dx - (mx != 0) as i32 * 3 < left
        || dy - (my != 0) as i32 * 3 < top
        || dx + bw4 * h_mul + (mx != 0) as i32 * 4 > right
        || dy + bh4 * v_mul + (my != 0) as i32 * 4 > bottom
        || ref_pw != right
        || ref_ph != bottom;

    let w = (bw4 * h_mul) as usize;
    let h = (bh4 * v_mul) as usize;
    let tmp_stride = w;
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
        inter_emu_edge_8bpc::<BD>(
            buf,
            emu_stride,
            ref_data,
            ref_stride,
            emu_w,
            emu_h,
            (iw - left) as usize,
            (ih - top) as usize,
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
        // SAFETY: BPC==8 => BD::Pixel == u8; the prep kernels write i16 `tmp`.
        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
        if is_bilin {
            crate::mc_dispatch::prep_bilin_8bpc_with_scratch(
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
            crate::mc_dispatch::prep_8tap_8bpc_with_scratch(
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
    } else if let Some(src16) = <BD::Pixel as Pixel>::try_as_u16_slice(src) {
        if is_bilin {
            crate::mc_dispatch::prep_bilin_hbd_with_scratch(
                tmp,
                tmp_stride,
                &src16[src_off..],
                src_stride,
                w,
                h,
                mxf,
                myf,
                bd.bitdepth(),
                inter_scratch,
            );
        } else {
            crate::mc_dispatch::prep_8tap_hbd_with_scratch(
                tmp,
                tmp_stride,
                src16,
                src_off,
                src_stride,
                w,
                h,
                mxf,
                myf,
                filter as i32,
                bd.bitdepth(),
                inter_scratch,
            );
        }
    }
}

/// `warp_affine`, affine path). Predicts the block in 8x8 sub-tiles using the
/// derived warp matrix `wmp`. Only the affine sub-path is implemented (block is
/// >= 8px and `wmp.affine`); callers gate on those conditions, falling back to
/// translational MC otherwise. 8bpc luma + chroma (with subsampling).
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_plane_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    b_dim: &[u8],
    wmp: &crate::headers::WarpedMotionParams,
    ss_hor: i32,
    ss_ver: i32,
    frame_bw: i32,
    frame_bh: i32,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let mat = &wmp.matrix;
    let width = frame_bw * 4 >> plss_hor;
    let height = frame_bh * 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => s,
        None => return,
    };

    let blk_w = b_dim[0] as i32 * h_mul;
    let blk_h = b_dim[1] as i32 * v_mul;
    let abcd: [i16; 4] = wmp.abcd;

    let mut emu = [BD::Pixel::default(); 32 * 32];
    let mut y = 0;
    while y < blk_h {
        let src_y = by * 4 + ((y + 4) << plss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut x = 0;
        while x < blk_w {
            let src_x = bx * 4 + ((x + 4) << plss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> plss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> plss_ver;

            let dx = (mvx >> 16) as i32 - 4;
            let mx =
                (((mvx as i32) & 0xffff) - wmp.abcd[0] as i32 * 4 - wmp.abcd[1] as i32 * 7) & !0x3f;
            let dy = (mvy >> 16) as i32 - 4;
            let my =
                (((mvy as i32) & 0xffff) - wmp.abcd[2] as i32 * 4 - wmp.abcd[3] as i32 * 4) & !0x3f;

            let (src, src_off, src_stride): (&[BD::Pixel], usize, usize) =
                if dx < 3 || dx + 8 + 4 > width || dy < 3 || dy + 8 + 4 > height {
                    crate::mc::emu_edge::<BD::Pixel>(
                        15,
                        15,
                        width as usize,
                        height as usize,
                        (dx - 3) as isize,
                        (dy - 3) as isize,
                        &mut emu,
                        32,
                        ref_data,
                        ref_stride,
                    );
                    (&emu[..], 32 * 3 + 3, 32)
                } else {
                    (ref_data, ref_stride * dy as usize + dx as usize, ref_stride)
                };

            let dst_sub = (y as usize) * dst_stride + x as usize;
            if BD::BPC == 8 {
                // SAFETY: BPC==8 => BD::Pixel == u8.
                let dst8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(&mut dst[dst_sub..]);
                let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
                crate::mc_dispatch::warp_affine_8x8_8bpc(
                    dst8, dst_stride, src8, src_stride, src_off, &abcd, mx, my,
                );
            } else {
                crate::mc::warp_affine_8x8(
                    bd,
                    &mut dst[dst_sub..],
                    dst_stride,
                    src,
                    src_stride,
                    src_off,
                    &abcd,
                    mx,
                    my,
                );
            }
            x += 8;
        }
        y += 8;
    }
}

/// for warp blocks where the 8x8 affine kernel does not apply: non-affine warp
/// types, or after subsampling a block becomes < 8 in either dimension (e.g.
/// chroma of an 8x8 luma block in 4:2:0). Walks `sw`x`sh` windows, then 4x4
/// tiles within, with the per-tile `+0x200` rounding and 6-bit mx/my subpel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ext_warp_plane_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    b_dim: &[u8],
    wmp: &crate::headers::WarpedMotionParams,
    ss_hor: i32,
    ss_ver: i32,
    frame_bw: i32,
    frame_bh: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let mat = &wmp.matrix;
    let w = frame_bw * 4 >> plss_hor;
    let h = frame_bh * 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => s,
        None => return,
    };

    let blk_w = b_dim[0] as i32 * h_mul;
    let blk_h = b_dim[1] as i32 * v_mul;
    let sw = imin(blk_w, 8);
    let hsw = sw >> 1;
    let sh = imin(blk_h, 8);
    let hsh = sh >> 1;

    let mut emu = [BD::Pixel::default(); 32 * 32];
    let mut y = 0;
    while y < blk_h {
        let src_y = by * 4 + ((y + hsh) << plss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut x = 0;
        while x < blk_w {
            let src_x = bx * 4 + ((x + hsw) << plss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> plss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> plss_ver;
            let left_window = (mvx >> 16) as i32 - hsw - 3;
            let top_window = (mvy >> 16) as i32 - hsh - 3;
            let left = iclip(left_window, 0, w - 1);
            let right = iclip(left_window + sw + 7, 1, w);
            let top = iclip(top_window, 0, h - 1);
            let bottom = iclip(top_window + sh + 7, 1, h);

            let mut yy = y;
            while yy < y + sh {
                let src_y2 = by * 4 + ((yy + 2) << plss_ver);
                let mat3_y2 = mat[3] as i64 * src_y2 as i64 + mat[0] as i64;
                let mat5_y2 = mat[5] as i64 * src_y2 as i64 + mat[1] as i64;
                let mut xx = x;
                while xx < x + sw {
                    let src_x2 = bx * 4 + ((xx + 2) << plss_hor);
                    let mvx2 = ((mat[2] as i64 * src_x2 as i64 + mat3_y2) >> plss_hor) + 0x200;
                    let mvy2 = ((mat[4] as i64 * src_x2 as i64 + mat5_y2) >> plss_ver) + 0x200;

                    let dx = (mvx2 >> 16) as i32 - 2;
                    let mx = ((mvx2 >> 10) & 63) as i32;
                    let dy = (mvy2 >> 16) as i32 - 2;
                    let my = ((mvy2 >> 10) & 63) as i32;

                    let (src, src_off, src_stride): (&[BD::Pixel], usize, usize) = if dx - 3 < left
                        || dx + 4 + 4 > right
                        || dy - 3 < top
                        || dy + 4 + 4 > bottom
                    {
                        let region_off = left as usize + top as usize * ref_stride;
                        crate::mc::emu_edge::<BD::Pixel>(
                            11,
                            11,
                            (right - left) as usize,
                            (bottom - top) as usize,
                            (dx - 3 - left) as isize,
                            (dy - 3 - top) as isize,
                            &mut emu,
                            32,
                            &ref_data[region_off..],
                            ref_stride,
                        );
                        (&emu[..], 32 * 3 + 3, 32)
                    } else {
                        (ref_data, ref_stride * dy as usize + dx as usize, ref_stride)
                    };

                    let dst_sub = (yy as usize) * dst_stride + xx as usize;
                    if BD::BPC == 8 {
                        // SAFETY: BPC==8 => BD::Pixel == u8.
                        let dst8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(&mut dst[dst_sub..]);
                        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
                        crate::mc_dispatch::put_8tap_8bpc_with_scratch(
                            dst8,
                            dst_stride,
                            src8,
                            src_off,
                            src_stride,
                            4,
                            4,
                            mx,
                            my,
                            -1,
                            inter_scratch,
                        );
                    } else if let (Some(dst16), Some(src16)) = (
                        <BD::Pixel as Pixel>::try_as_u16_slice_mut(&mut dst[dst_sub..]),
                        <BD::Pixel as Pixel>::try_as_u16_slice(src),
                    ) {
                        crate::mc_dispatch::put_8tap_hbd_with_scratch(
                            dst16,
                            dst_stride,
                            src16,
                            src_off,
                            src_stride,
                            4,
                            4,
                            mx,
                            my,
                            -1,
                            bd.bitdepth(),
                            inter_scratch,
                        );
                    }
                    xx += 4;
                }
                yy += 4;
            }
            x += sw;
        }
        y += sh;
    }
}

/// buffer (stride `bw4*4`) for compound blending instead of writing u8 pixels.
/// Identical addressing/emu-edge logic; only the per-8x8 kernel differs
/// (`warp8x8t` vs `warp8x8`). Falls back to the ext-warp prep for non-affine /
/// sub-8px (after subsampling) blocks.
#[allow(clippy::too_many_arguments)]
fn warp_affine_plane_prep_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    b_dim: &[u8],
    wmp: &crate::headers::WarpedMotionParams,
    ss_hor: i32,
    ss_ver: i32,
    frame_bw: i32,
    frame_bh: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    if wmp.affine == 0 || imin(b_dim[0] as i32 * h_mul, b_dim[1] as i32 * v_mul) < 8 {
        ext_warp_plane_prep_8bpc::<BD>(
            bd,
            tmp,
            tmp_stride,
            ref_pic,
            pl,
            bx,
            by,
            b_dim,
            wmp,
            ss_hor,
            ss_ver,
            frame_bw,
            frame_bh,
            inter_scratch,
        );
        return;
    }
    let mat = &wmp.matrix;
    let width = frame_bw * 4 >> plss_hor;
    let height = frame_bh * 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => s,
        None => return,
    };

    let blk_w = b_dim[0] as i32 * h_mul;
    let blk_h = b_dim[1] as i32 * v_mul;
    let abcd: [i16; 4] = wmp.abcd;

    let mut emu = [BD::Pixel::default(); 32 * 32];
    let mut y = 0;
    while y < blk_h {
        let src_y = by * 4 + ((y + 4) << plss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut x = 0;
        while x < blk_w {
            let src_x = bx * 4 + ((x + 4) << plss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> plss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> plss_ver;

            let dx = (mvx >> 16) as i32 - 4;
            let mx =
                (((mvx as i32) & 0xffff) - wmp.abcd[0] as i32 * 4 - wmp.abcd[1] as i32 * 7) & !0x3f;
            let dy = (mvy >> 16) as i32 - 4;
            let my =
                (((mvy as i32) & 0xffff) - wmp.abcd[2] as i32 * 4 - wmp.abcd[3] as i32 * 4) & !0x3f;

            let (src, src_off, src_stride): (&[BD::Pixel], usize, usize) =
                if dx < 3 || dx + 8 + 4 > width || dy < 3 || dy + 8 + 4 > height {
                    crate::mc::emu_edge::<BD::Pixel>(
                        15,
                        15,
                        width as usize,
                        height as usize,
                        (dx - 3) as isize,
                        (dy - 3) as isize,
                        &mut emu,
                        32,
                        ref_data,
                        ref_stride,
                    );
                    (&emu[..], 32 * 3 + 3, 32)
                } else {
                    (ref_data, ref_stride * dy as usize + dx as usize, ref_stride)
                };

            let dst_sub = (y as usize) * tmp_stride + x as usize;
            if BD::BPC == 8 {
                // SAFETY: BPC==8 => BD::Pixel == u8.
                let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
                crate::mc_dispatch::warp_affine_8x8t_8bpc(
                    &mut tmp[dst_sub..],
                    tmp_stride,
                    src8,
                    src_stride,
                    src_off,
                    &abcd,
                    mx,
                    my,
                );
            } else {
                crate::mc::warp_affine_8x8t(
                    bd,
                    &mut tmp[dst_sub..],
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    &abcd,
                    mx,
                    my,
                );
            }
            x += 8;
        }
        y += 8;
    }
}

/// 8-tap prep (i16) kernel (`ext_warp4x4t` = `prep_8tap(..., -1)`) instead of
/// the u8 `put_8tap`.
#[allow(clippy::too_many_arguments)]
fn ext_warp_plane_prep_8bpc<BD: crate::pixel::BitDepth>(
    bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    ref_pic: &crate::picture::Picture,
    pl: usize,
    bx: i32,
    by: i32,
    b_dim: &[u8],
    wmp: &crate::headers::WarpedMotionParams,
    ss_hor: i32,
    ss_ver: i32,
    frame_bw: i32,
    frame_bh: i32,
    inter_scratch: &mut Vec<i16>,
) {
    let plss_ver = if pl != 0 { ss_ver } else { 0 };
    let plss_hor = if pl != 0 { ss_hor } else { 0 };
    let h_mul = 4 >> plss_hor;
    let v_mul = 4 >> plss_ver;
    let mat = &wmp.matrix;
    let w = frame_bw * 4 >> plss_hor;
    let h = frame_bh * 4 >> plss_ver;
    let ref_stride =
        ref_pic.stride[(pl != 0) as usize].unsigned_abs() / std::mem::size_of::<BD::Pixel>();
    let ref_data: &[BD::Pixel] = match ref_pic.plane_slice::<BD::Pixel>(pl) {
        Some(s) => s,
        None => return,
    };

    let blk_w = b_dim[0] as i32 * h_mul;
    let blk_h = b_dim[1] as i32 * v_mul;
    let sw = imin(blk_w, 8);
    let hsw = sw >> 1;
    let sh = imin(blk_h, 8);
    let hsh = sh >> 1;

    let mut emu = [BD::Pixel::default(); 32 * 32];
    let mut y = 0;
    while y < blk_h {
        let src_y = by * 4 + ((y + hsh) << plss_ver);
        let mat3_y = mat[3] as i64 * src_y as i64 + mat[0] as i64;
        let mat5_y = mat[5] as i64 * src_y as i64 + mat[1] as i64;
        let mut x = 0;
        while x < blk_w {
            let src_x = bx * 4 + ((x + hsw) << plss_hor);
            let mvx = (mat[2] as i64 * src_x as i64 + mat3_y) >> plss_hor;
            let mvy = (mat[4] as i64 * src_x as i64 + mat5_y) >> plss_ver;
            let left_window = (mvx >> 16) as i32 - hsw - 3;
            let top_window = (mvy >> 16) as i32 - hsh - 3;
            let left = iclip(left_window, 0, w - 1);
            let right = iclip(left_window + sw + 7, 1, w);
            let top = iclip(top_window, 0, h - 1);
            let bottom = iclip(top_window + sh + 7, 1, h);

            let mut yy = y;
            while yy < y + sh {
                let src_y2 = by * 4 + ((yy + 2) << plss_ver);
                let mat3_y2 = mat[3] as i64 * src_y2 as i64 + mat[0] as i64;
                let mat5_y2 = mat[5] as i64 * src_y2 as i64 + mat[1] as i64;
                let mut xx = x;
                while xx < x + sw {
                    let src_x2 = bx * 4 + ((xx + 2) << plss_hor);
                    let mvx2 = ((mat[2] as i64 * src_x2 as i64 + mat3_y2) >> plss_hor) + 0x200;
                    let mvy2 = ((mat[4] as i64 * src_x2 as i64 + mat5_y2) >> plss_ver) + 0x200;

                    let dx = (mvx2 >> 16) as i32 - 2;
                    let mx = ((mvx2 >> 10) & 63) as i32;
                    let dy = (mvy2 >> 16) as i32 - 2;
                    let my = ((mvy2 >> 10) & 63) as i32;

                    let (src, src_off, src_stride): (&[BD::Pixel], usize, usize) = if dx - 3 < left
                        || dx + 4 + 4 > right
                        || dy - 3 < top
                        || dy + 4 + 4 > bottom
                    {
                        let region_off = left as usize + top as usize * ref_stride;
                        crate::mc::emu_edge::<BD::Pixel>(
                            11,
                            11,
                            (right - left) as usize,
                            (bottom - top) as usize,
                            (dx - 3 - left) as isize,
                            (dy - 3 - top) as isize,
                            &mut emu,
                            32,
                            &ref_data[region_off..],
                            ref_stride,
                        );
                        (&emu[..], 32 * 3 + 3, 32)
                    } else {
                        (ref_data, ref_stride * dy as usize + dx as usize, ref_stride)
                    };

                    let dst_sub = (yy as usize) * tmp_stride + xx as usize;
                    if BD::BPC == 8 {
                        // SAFETY: BPC==8 => BD::Pixel == u8.
                        let src8: &[u8] = BD::Pixel::slice_as_ne_bytes(src);
                        crate::mc_dispatch::prep_8tap_8bpc_with_scratch(
                            &mut tmp[dst_sub..],
                            tmp_stride,
                            src8,
                            src_off,
                            src_stride,
                            4,
                            4,
                            mx,
                            my,
                            -1,
                            inter_scratch,
                        );
                    } else if let Some(src16) = <BD::Pixel as Pixel>::try_as_u16_slice(src) {
                        crate::mc_dispatch::prep_8tap_hbd_with_scratch(
                            &mut tmp[dst_sub..],
                            tmp_stride,
                            src16,
                            src_off,
                            src_stride,
                            4,
                            4,
                            mx,
                            my,
                            -1,
                            bd.bitdepth(),
                            inter_scratch,
                        );
                    }
                    xx += 4;
                }
                yy += 4;
            }
            x += sw;
        }
        y += sh;
    }
}

/// to the plane bounds. Identical semantics to mc.rs's private `emu_edge`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inter_emu_edge_8bpc<BD: crate::pixel::BitDepth>(
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
) {
    for dy in 0..bh {
        let ay = ((y + dy as i32).max(0) as usize).min(ih.saturating_sub(1));
        let drow = dy * dst_stride;
        let srow = ay * src_stride;
        for dx in 0..bw {
            let ax = ((x + dx as i32).max(0) as usize).min(iw.saturating_sub(1));
            dst[drow + dx] = src.get(srow + ax).copied().unwrap_or_default();
        }
    }
}

/// Add an inter residual transform block onto an already motion-compensated
/// destination (8bpc). Decodes coefficients with the inter coef contexts
/// (`intra = false`), applies the optional secondary transform, then the inverse
/// transform add. Mirrors the residual tail of `recon_b_luma_tx` for inter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inter_residual_tx_8bpc<
    BD: BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    pl: usize,
    tx: usize,
    bx: i32,
    by: i32,
    dst_is_uv: bool,
    txtp_seed: u16,
    fi: &SbFrameInfo,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    use crate::levels::IntraPredMode;
    let t_dim = &TXFM_DIMENSIONS[tx];
    let tw = t_dim.w as usize * 4;
    let th = t_dim.h as usize * 4;
    let tw4 = t_dim.w as i32;
    let th4 = t_dim.h as i32;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;
    let ss_hor = if dst_is_uv { recon.frame.ss_hor } else { 0 };
    let ss_ver = if dst_is_uv { recon.frame.ss_ver } else { 0 };

    let bx4 = ((bx & 63) >> ss_hor) as usize;
    let by4 = ((by & 63) >> ss_ver) as usize;

    // No pre-clear: every inverse-transform path clears cf[..S*S] after use
    // (itx_2d dequant cores + itx.rs WHT/DC/generic), and cf starts zeroed, so
    // cf is already zero on entry here. Verified bit-exact + via prefill poison.

    let mut txtp: u16 = txtp_seed;
    let mut res_ctx: u8 = 0;
    let (mut eob, stx, mut txtp) = if b.skip_txfm != 0 {
        res_ctx = 0x40;
        (-1i32, 0i32, crate::levels::txtp::DCT_DCT as u32)
    } else {
        let dq_tbl = recon.dq_active[seg_id][pl];
        let qm_ref: Option<&[u8]> = recon.frame.qm[tx][pl].as_deref();
        let params = crate::recon::DecodeCoefParams {
            tx,
            bs: b.bs as usize,
            plane: pl as i32,
            intra: false,
            fsc: false,
            lossless,
            sdp_active: false,
            y_mode: 0,
            uv_mode: 0,
            seq_fsc: recon.frame.seq_fsc,
            seq_ist: recon.frame.seq_ist,
            seq_cctx: recon.frame.seq_cctx,
            chroma_dctonly: false,
            reduced_txtp_set: recon.frame.reduced_txtp_set,
            tcq_enabled: recon.frame.tcq,
            layout: recon.frame.layout,
            u_has_cf: recon.scratch_u_has_cf,
            cbx: bx,
            cby: by,
            luma_fsc_map: &[],
            dq_tbl,
            bitdepth: recon.frame.bitdepth,
            qm: qm_ref,
            ss_hor: recon.frame.ss_hor != 0,
            ss_ver: recon.frame.ss_ver != 0,
        };
        let (acoef, lcoef): (&[u8], &[u8]) = if pl == 0 {
            (&a.lcoef[bx4..], &l.lcoef[by4..])
        } else {
            (&a.ccoef[pl - 1][bx4..], &l.ccoef[pl - 1][by4..])
        };
        let eob = msac.decode_coefs(
            recon.cdf_coef,
            cdf_m,
            acoef,
            lcoef,
            &params,
            recon.cf,
            &mut txtp,
            &mut res_ctx,
            &mut recon.scratch.coef_levels,
        );
        if eob == i32::MIN {
            return Err(());
        }
        // transform type for each covered 4x4 so the inter chroma path can seed
        // (txtp &= 0xff) before storing, so only the base type is propagated.
        if pl == 0 {
            let by15 = (by & 15) as usize;
            let bx15 = (bx & 15) as usize;
            let base_txtp = txtp & 0xff;
            for dy in 0..t_dim.h as usize {
                for dx in 0..t_dim.w as usize {
                    let yy = by15 + dy;
                    let xx = bx15 + dx;
                    if yy < 16 && xx < 16 {
                        recon.scratch.txtp_map[yy * 16 + xx] = base_txtp;
                    }
                }
            }
        }
        let stx = (txtp >> 8) as i32;
        (eob, stx, (txtp & 0xff) as u32)
    };

    if pl == 1 {
        recon.scratch_u_has_cf = (eob >= 0) as i32;
    }

    // context fill
    let aw = imin(tw4, (fi.bw >> ss_hor) - (bx >> ss_hor)).max(0) as usize;
    let lh = imin(th4, (fi.bh >> ss_ver) - (by >> ss_ver)).max(0) as usize;
    if pl == 0 {
        if aw > 0 {
            a.lcoef[bx4..bx4 + aw].fill(res_ctx);
        }
        if lh > 0 {
            l.lcoef[by4..by4 + lh].fill(res_ctx);
        }
    } else {
        if aw > 0 {
            a.ccoef[pl - 1][bx4..bx4 + aw].fill(res_ctx);
        }
        if lh > 0 {
            l.ccoef[pl - 1][by4..by4 + lh].fill(res_ctx);
        }
    }

    // later blocks' top-right / bottom-left intra-edge availability (n_tr/n_bl,
    // used by SMOOTH_PRED incl. warp-interintra) sees inter neighbours as coded.
    if pl == 0 {
        let mask: u64 = ((1u64 << tw4) - 1) << (bx4 as u32);
        for y in 0..th4 as usize {
            let row = by4 + y;
            if row < 64 {
                recon.scratch.is_coded[0][row] |= mask;
            }
        }
        // LR no-skip mask (luma): set per coded luma TX block, used by the
        // block has coefficients (eob != -1).
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
    } else if pl == 1 {
        // Chroma `is_coded` is marked once per chroma TX (pl==1 only, mirroring
        let mask: u64 = ((1u64 << tw4) - 1) << (bx4 as u32);
        for y in 0..th4 as usize {
            let row = by4 + y;
            if row < 64 {
                recon.scratch.is_coded[1][row] |= mask;
            }
        }
    }

    if eob == -1 {
        return Ok(());
    }

    let bd = recon.bd;
    let (dst, stride) = if pl == 0 {
        (&mut *recon.dst_y, recon.frame.y_stride_px)
    } else if pl == 1 {
        (&mut *recon.dst_u, recon.frame.uv_stride_px)
    } else {
        (&mut *recon.dst_v, recon.frame.uv_stride_px)
    };
    let dst_off = 4 * ((by >> ss_ver) as usize * stride + (bx >> ss_hor) as usize);

    if stx != 0 && (stx & 3) != 0 {
        // Inter never matches the intra y_mode transpose mask -> transpose = true.
        let transpose = true;
        let stype = (stx & 3) - 1;
        let set = (stx >> 2) & 15;
        if tw >= 8 && th >= 8 {
            let koff = (set as usize * 3 + stype as usize) * 1536;
            let idx = (imin(t_dim.lh as i32, 3) - 1) as usize;
            let scan_out = &crate::stx_tables::STX_SCAN_ORDERS_8X8[idx][transpose as usize];
            let mapping = &crate::stx_tables::COEFF8X8_MAPPING[set as usize * 3 + stype as usize];
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

    if recon.frame.seq_inter_ddt {
        txtp += txtp & crate::tables::TX_DDT_MASK[tx] as u32;
    }
    let _ = IntraPredMode::DcPred;

    let _ = (tw, th);
    crate::itx::inv_txfm_add(
        bd,
        dst,
        dst_off,
        stride,
        recon.cf,
        txtp,
        eob,
        tx,
        &mut recon.scratch.itx_tmp,
    );
    Ok(())
}

/// Inter chroma residual for both planes with the cross-component transform
/// 3925): decode all U then all V TU coefficients (the entropy order), apply
/// CCTX to mix U/V per TU, then inverse-transform-add both planes. CCTX needs
/// U and V coefficients together, so it cannot be done in the per-plane
/// `inter_residual_tx_8bpc`. The chroma prediction is already in dst_u/dst_v.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inter_chroma_residual_8bpc<
    BD: BitDepth,
    const UPDATE_CDF: bool,
    M: MsacReader<UPDATE_CDF>,
>(
    recon: &mut ReconCtx<BD>,
    msac: &mut M,
    cdf_m: &mut CdfModeContext,
    a: &mut BlockContext,
    l: &mut BlockContext,
    b: &Av2Block,
    uvtx: usize,
    cbx: i32,
    cby: i32,
    cbw4ss: i32,
    cbh4ss: i32,
    cw4ss: i32,
    ch4ss: i32,
    txtp_seed: u16,
    phase: ChromaPhase,
    fi: &SbFrameInfo,
) -> Result<(), ()>
where
    BD::Coef: DecodeCoeff,
{
    let uv_t_dim = &TXFM_DIMENSIONS[uvtx];
    let (txw, txh) = (uv_t_dim.w as i32, uv_t_dim.h as i32);
    let bd = recon.bd;
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;
    let seg_id = b.seg_id as usize;
    let lossless = recon.frame.seg_lossless[seg_id] != 0;
    let n_tu = (cbw4ss * cbh4ss) as usize;
    let tu_n = (txw as usize * 4) * (txh as usize * 4);

    // Coefficients for the TU at grid position i = y*cbw4ss + x are placed at
    // cf[i*16] (mirrors C `cf[pl][i*16]`); the per-plane buffer is n_tu*16.
    let cf_len = n_tu * 16;
    let cf_need = cf_len * 2;
    let mut cf_uv = recon.scratch.take_chroma_cf::<BD::Coef>();
    if cf_uv.len() < cf_need {
        cf_uv.resize(cf_need, <BD::Coef as crate::pixel::Coeff>::ZERO);
    }
    let (cf_u, cf_v) = cf_uv[..cf_need].split_at_mut(cf_len);
    recon.scratch.chroma_txtp[..n_tu].fill([0u16; 2]);
    recon.scratch.chroma_eob[..n_tu].fill([-1i16; 2]);

    // Coefficient (entropy) read. For the chroma read/recon staging of >64px
    // read with the first luma sub-block (`ReadOnly`) and stashed, then the
    // residual is applied with the last luma sub-block (`ReconOnly`) once the
    // intervening luma blocks have splatted their MVs into the spatial refmvs
    // grid (so OPFL/refine-mv/BACP chroma prediction sees the final state).
    if phase != ChromaPhase::ReconOnly {
        // Decode all U TUs then all V TUs (entropy order).
        recon.scratch_u_has_cf = 0;
        for pl in 0..2usize {
            let plane_cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
            let mut y = 0;
            while y < ch4ss {
                let mut x = 0;
                while x < cw4ss {
                    let i = (y * cbw4ss + x) as usize;
                    let bx = cbx + (x << ss_hor);
                    let by = cby + (y << ss_ver);
                    let bx4 = ((bx & 63) >> ss_hor) as usize;
                    let by4 = ((by & 63) >> ss_ver) as usize;

                    let cf_slot = &mut plane_cf[i * 16..i * 16 + tu_n];
                    cf_slot.fill(<BD::Coef as crate::pixel::Coeff>::ZERO);

                    // Seed the chroma transform type from the co-located luma 4x4's
                    // recorded txtp. When the chroma block size equals the luma
                    // with `t->bx` advanced per TU); otherwise the block-origin seed
                    // (`txtp_seed`) is used for every TU. For inter blocks the seed
                    // only matters for lossless (WHT vs IDTX), where using the wrong
                    // per-TU luma type produces a transposed residual.
                    let mut txtp: u16 = if b.bs == b.cbs {
                        recon.scratch.txtp_map[(by & 15) as usize * 16 + (bx & 15) as usize]
                    } else {
                        txtp_seed
                    };
                    let mut res_ctx: u8 = 0;
                    let eob = if b.skip_txfm != 0 {
                        res_ctx = 0x40;
                        txtp = crate::levels::txtp::DCT_DCT as u16;
                        -1i32
                    } else {
                        let dq_tbl = recon.dq_active[seg_id][1 + pl];
                        let qm_ref: Option<&[u8]> = recon.frame.qm[uvtx][1 + pl].as_deref();
                        let params = crate::recon::DecodeCoefParams {
                            tx: uvtx,
                            // Chroma coefficient decode keys its skip context on the
                            // CHROMA block size (`b->cbs`), not the luma block size:
                            // For sub-8x8 luma where chroma spans several luma
                            // sub-blocks (bs != cbs) the V-plane `not_one_blk` term in
                            // get_skip_ctx diverges if the luma bs is used.
                            bs: b.cbs as usize,
                            plane: (1 + pl) as i32,
                            intra: false,
                            fsc: false,
                            lossless,
                            sdp_active: false,
                            y_mode: 0,
                            uv_mode: 0,
                            seq_fsc: recon.frame.seq_fsc,
                            seq_ist: recon.frame.seq_ist,
                            seq_cctx: recon.frame.seq_cctx,
                            chroma_dctonly: false,
                            reduced_txtp_set: recon.frame.reduced_txtp_set,
                            tcq_enabled: recon.frame.tcq,
                            layout: recon.frame.layout,
                            u_has_cf: recon.scratch_u_has_cf,
                            cbx: bx,
                            cby: by,
                            luma_fsc_map: &[],
                            dq_tbl,
                            bitdepth: recon.frame.bitdepth,
                            qm: qm_ref,
                            ss_hor: recon.frame.ss_hor != 0,
                            ss_ver: recon.frame.ss_ver != 0,
                        };
                        let acoef = &a.ccoef[pl][bx4..];
                        let lcoef = &l.ccoef[pl][by4..];
                        let e = msac.decode_coefs(
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
                        if e == i32::MIN {
                            recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
                            return Err(());
                        }
                        e
                    };
                    if pl == 0 {
                        recon.scratch_u_has_cf = (eob >= 0) as i32;
                    }
                    recon.scratch.chroma_eob[i][pl] = eob as i16;
                    recon.scratch.chroma_txtp[i][pl] = txtp;

                    // Context fill (a/l ccoef) and is_coded[1] (pl==0 only).
                    let aw = imin(txw, (fi.bw >> ss_hor) - (bx >> ss_hor)).max(0) as usize;
                    let lh = imin(txh, (fi.bh >> ss_ver) - (by >> ss_ver)).max(0) as usize;
                    if aw > 0 {
                        a.ccoef[pl][bx4..bx4 + aw].fill(res_ctx);
                    }
                    if lh > 0 {
                        l.ccoef[pl][by4..by4 + lh].fill(res_ctx);
                    }
                    if pl == 0 {
                        let mask: u64 = ((1u64 << txw) - 1) << (bx4 as u32);
                        for yy in 0..txh as usize {
                            let row = by4 + yy;
                            if row < 64 {
                                recon.scratch.is_coded[1][row] |= mask;
                            }
                        }
                    }
                    x += txw;
                }
                y += txh;
            }
        }
    } // end coef-read phase

    // Stash the decoded coefficients for the deferred recon phase, or restore
    // them.  This is now a FIFO, so future whole-SB entropy/recon splitting can
    // read several chroma blocks before replaying them.
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
            cbs: b.cbs as u8,
            sdp_active: false,
            n_tu: n_tu as u16,
            cf_off: cf_off as u32,
            cf_len: need as u32,
            u_has_cf: recon.scratch_u_has_cf,
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
        debug_assert_eq!(rec.n_tu as usize, n_tu);
        let cf_off = rec.cf_off as usize;
        let rec_cf_len = rec.cf_len as usize;
        debug_assert_eq!(rec_cf_len, cf_need);
        let chroma_tx_cf = recon.scratch.chroma_tx_cf::<BD::Coef>();
        cf_u.copy_from_slice(&chroma_tx_cf[cf_off..cf_off + cf_len]);
        cf_v.copy_from_slice(&chroma_tx_cf[cf_off + cf_len..cf_off + cf_need]);
        recon.scratch.chroma_txtp = rec.txtp;
        recon.scratch.chroma_eob = rec.eob;
        recon.scratch_u_has_cf = rec.u_has_cf;
    }

    let cctx_enabled = recon.frame.seq_cctx
        && (recon.frame.layout == crate::headers::PixelLayout::I420 || uv_t_dim.min < 8);
    let uv_stride = recon.frame.uv_stride_px;
    let mut y = 0;
    while y < ch4ss {
        let mut x = 0;
        while x < cw4ss {
            let i = (y * cbw4ss + x) as usize;
            let bx = cbx + (x << ss_hor);
            let by = cby + (y << ss_ver);

            let cctx_type = if cctx_enabled && recon.scratch.chroma_eob[i][0] >= 0 {
                (recon.scratch.chroma_txtp[i][0] >> 8) as i32
            } else {
                0
            };
            if cctx_type != 0 {
                let sz = imin(txw * 4, 32) as usize * imin(txh * 4, 32) as usize;
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

            for pl in 0..2usize {
                let eob = recon.scratch.chroma_eob[i][pl];
                if eob == -1 {
                    continue;
                }
                let mut txtp = recon.scratch.chroma_txtp[i][pl] as u32;
                if recon.frame.seq_inter_ddt {
                    txtp += txtp & crate::tables::TX_DDT_MASK[uvtx] as u32;
                }
                let dst_off =
                    4 * (((by >> ss_ver) as usize) * uv_stride + ((bx >> ss_hor) as usize));
                let cf = if pl == 0 { &mut *cf_u } else { &mut *cf_v };
                let dst_plane: &mut [BD::Pixel] = if pl == 0 { recon.dst_u } else { recon.dst_v };
                crate::itx::inv_txfm_add(
                    bd,
                    dst_plane,
                    dst_off,
                    uv_stride,
                    &mut cf[i * 16..],
                    txtp,
                    eob as i32,
                    uvtx,
                    &mut recon.scratch.itx_tmp,
                );
            }
            x += txw;
        }
        y += txh;
    }
    recon.scratch.put_chroma_cf::<BD::Coef>(cf_uv);
    Ok(())
}

/// in `dst`; build the intra prediction (DC/V/H/SMOOTH, or wedge) from the
/// reconstructed neighbour edges into a temp buffer, then blend it over `dst`
/// with the II / wedge mask. Plane 0 (luma) or 1/2 (chroma, subsampled).
#[allow(clippy::too_many_arguments)]
pub(crate) fn iiblend_luma_8bpc<BD: crate::pixel::BitDepth>(
    recon: &mut ReconCtx<BD>,
    b: &Av2Block,
    dst_off: usize,
    stride: usize,
    bw4: i32,
    bh4: i32,
    by: i32,
    bx: i32,
    ss_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    iiblend_plane_8bpc(recon, b, 0, dst_off, stride, bw4, bh4, by, bx, ss_bs, fi);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn iiblend_chroma_8bpc<BD: BitDepth>(
    recon: &mut ReconCtx<BD>,
    b: &Av2Block,
    plane: usize,
    dst_off: usize,
    stride: usize,
    bw4: i32,
    bh4: i32,
    by: i32,
    bx: i32,
    ss_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    iiblend_plane_8bpc(
        recon, b, plane, dst_off, stride, bw4, bh4, by, bx, ss_bs, fi,
    );
}

#[allow(clippy::too_many_arguments)]
fn iiblend_plane_8bpc<BD: BitDepth>(
    recon: &mut ReconCtx<BD>,
    b: &Av2Block,
    plane: usize,
    dst_off: usize,
    stride: usize,
    bw4: i32,
    bh4: i32,
    by: i32,
    bx: i32,
    ss_bs: BlockSize,
    fi: &SbFrameInfo,
) {
    use crate::levels::{
        ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, InterIntraPredMode, IntraPredMode,
    };
    let ii_mode = b.inter_data().interintra_mode;
    let wedge_idx = b.inter_data().wedge_idx;
    // II mode -> intra pred mode (II_SMOOTH(3) -> SMOOTH_PRED(9)).
    let m0: u8 = if ii_mode == InterIntraPredMode::SmoothPred as u8 {
        IntraPredMode::SmoothPred as u8
    } else {
        // DC(0)->DcPred(0), V(1)->VertPred(1), H(2)->HorPred(2).
        ii_mode
    };
    let angle: i32 = [0, 90, 180, 0][ii_mode as usize];

    let chroma = plane != 0;
    let ss_hor = if chroma { fi.ss_hor } else { 0 };
    let ss_ver = if chroma { fi.ss_ver } else { 0 };
    let ssbw4 = bw4 >> ss_hor;
    let ssbh4 = bh4 >> ss_ver;
    let w = (ssbw4 * 4) as usize;
    let h = (ssbh4 * 4) as usize;

    let mut n_tr = 0i32;
    let mut n_bl = 0i32;
    if m0 == IntraPredMode::SmoothPred as u8 {
        let bx4 = (bx & 63) as usize;
        let by4 = (by & 63) as usize;
        let sbsz = fi.sb_step;
        if by > fi.tile_row_start {
            let mut wv = imin(bw4, fi.tile_col_end - bx - bw4);
            if (by & (sbsz - 1)) == 0 {
                n_tr = 0; // top sb boundary: simplified (no a_sb_cache)
            } else {
                let end = imin(((bx + sbsz) & !(sbsz - 1)) + 0, fi.tile_col_end);
                wv = imin(wv, end - bx - bw4);
                if wv <= 0 {
                    n_tr = 0;
                } else {
                    let row = ((by4 >> ss_ver) as i32 - 1) as usize;
                    if row < 64 {
                        n_tr = ((recon.scratch.is_coded[chroma as usize][row]
                            >> (((bx4 + bw4 as usize) >> ss_hor) as u32))
                            & 1) as i32;
                    }
                }
            }
        }
        if bx > fi.tile_col_start {
            let end = imin((by + sbsz) & !(sbsz - 1), fi.tile_row_end);
            let hv = imin(bh4, end - by - bh4);
            if hv <= 0 {
                n_bl = 0;
            } else if (bx & (sbsz - 1)) == 0 {
                n_bl = hv;
            } else {
                let row = ((by4 + bh4 as usize) >> ss_ver) as usize;
                if row < 64 {
                    n_bl = ((recon.scratch.is_coded[chroma as usize][row]
                        >> (((bx4 as i32 - 1) >> ss_hor) as u32))
                        & 1) as i32;
                }
            }
        }
    }

    let apply_ibp = recon.frame.seq_ibp && imax(ssbw4, ssbh4) > 1;
    let have_left = bx > fi.tile_col_start;
    let have_top = by > fi.tile_row_start;
    let intra_flags = if apply_ibp { ANGLE_IBP_FLAG } else { 0 }
        | if have_left { ANGLE_HAS_LEFT_FLAG } else { 0 }
        | if have_top { ANGLE_HAS_TOP_FLAG } else { 0 };

    let edge_o: usize = 768;
    let max_w = 4 * (fi.bw >> ss_hor) - 4 * (bx >> ss_hor);
    let max_h = 4 * (fi.bh >> ss_ver) - 4 * (by >> ss_ver);

    // Intra prediction into a temp buffer (stride = w). Borrow the dst plane
    // (read for edge prep) and `edge` (mut) as disjoint fields.
    let bd = recon.bd;
    let mut tmp = vec![BD::Pixel::default(); w * h];
    {
        let ReconCtx {
            dst_y,
            dst_u,
            dst_v,
            edge,
            frame,
            ..
        } = &mut *recon;
        let dst_plane: &[BD::Pixel] = match plane {
            0 => dst_y,
            1 => dst_u,
            _ => dst_v,
        };
        let m = crate::ipred_prepare::prepare_intra_edges(
            bd,
            bx >> ss_hor,
            by >> ss_ver,
            fi.tile_col_end >> ss_hor,
            fi.tile_row_end >> ss_ver,
            n_tr,
            n_bl,
            dst_plane,
            dst_off,
            stride,
            None,
            m0,
            ssbw4,
            ssbh4,
            angle | intra_flags,
            edge,
            edge_o,
        );
        dispatch_ipred(
            bd,
            m,
            &mut tmp,
            0,
            w,
            edge,
            edge_o,
            w,
            h,
            intra_flags,
            max_w,
            max_h,
            &frame.ibp_weights,
        );
    }

    // Mask: II mask (wedge_idx == -1) or wedge mask. Borrow the static/cache
    // mask directly; this avoids a per-block Vec allocation in the compound
    // inter-intra blend path.
    let mask: &[u8] = if wedge_idx == -1 {
        let mode = match ii_mode {
            x if x == InterIntraPredMode::DcPred as u8 => InterIntraPredMode::DcPred,
            x if x == InterIntraPredMode::VertPred as u8 => InterIntraPredMode::VertPred,
            x if x == InterIntraPredMode::HorPred as u8 => InterIntraPredMode::HorPred,
            _ => InterIntraPredMode::SmoothPred,
        };
        &recon
            .masks
            .ii_mask(ss_bs as usize, ssbw4 as usize, ssbh4 as usize, mode)[..w * h]
    } else {
        &recon.masks.wedge_mask(
            ss_bs as usize,
            bw4 as usize,
            bh4 as usize,
            wedge_idx as usize,
            (ss_hor + ss_ver) as usize,
        )[..w * h]
    };
    let _ = (m0, ii_mode);
    let dst_plane: &mut [BD::Pixel] = match plane {
        0 => recon.dst_y,
        1 => recon.dst_u,
        _ => recon.dst_v,
    };
    if BD::BPC == 8 {
        // SAFETY: BPC==8 => BD::Pixel == u8.
        let d8: &mut [u8] = BD::Pixel::slice_as_ne_bytes_mut(&mut dst_plane[dst_off..]);
        let t8: &[u8] = BD::Pixel::slice_as_ne_bytes(&tmp);
        crate::mc_dispatch::blend_8bpc(d8, stride, t8, w, h, &mask);
    } else {
        crate::mc::blend(&mut dst_plane[dst_off..], stride, &tmp, w, h, &mask);
    }
}

/// Splat a resolved COMPOUND block's MVs into the spatial refmvs grid + the
/// only (warp-compound / global-affine deferred). The spatial grid write is the
/// same for AVG/SEG/WEDGE (only the temporal grid differs for WEDGE, handled via
/// the per-2x2 wedge tmvp mask).
#[allow(clippy::too_many_arguments)]
pub(crate) fn splat_tworef_mv<BD: crate::pixel::BitDepth>(
    recon: &mut ReconCtx<BD>,
    b: &Av2Block,
    bx: i32,
    by: i32,
    by4r: usize,
    bw4: i32,
    bh4: i32,
    bs: BlockSize,
) {
    let refs = b.ref_pair.refs();
    let ref0 = refs[0];
    let ref1 = refs[1];
    let cwp_idx = b.inter_data().cwp_idx;
    let comp_type = b.inter_data().comp_type;
    let inter_mode = b.inter_data().inter_mode;
    let blk_mv = [b.inter_data().mv[0], b.inter_data().mv[1]];

    let t_swap = (recon.rf.ref_flip & (1u64 << (ref0 as u32 * 8 + ref1 as u32))) != 0;
    let opfl = inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8;
    let refine_mv = b.inter_data().refine_mv != 0 && comp_type == 1;
    let write_temporal =
        recon.seq_hdr.ref_frame_mvs && (!opfl || !refine_mv) && !recon.cur_mvs.is_empty();

    let s_off = by4r * 128 + (bx & 127) as usize;
    let motion_mode = b.inter_data().motion_mode;

    // When the block uses a local warp (motion_mode > INTERINTRA) or a compound
    // per-4x4 warp-projected MVs (with mf|=2 / mf|=1) into the spatial grid so
    // that later blocks' refmvs candidates read the projected MV, not the
    // nominal block MV.
    let gmv0_affine =
        recon.frm_hdr.gmv.m[ref0 as usize].wm_type > crate::headers::WarpedMotionType::Translation;
    let gmv1_affine =
        recon.frm_hdr.gmv.m[ref1 as usize].wm_type > crate::headers::WarpedMotionType::Translation;
    let is_warp_splat = motion_mode > MotionMode::InterIntra as u8
        || (inter_mode == CompInterPredMode::GlobalMvGlobalMv as u8
            && imin(bw4, bh4) > 1
            && (gmv0_affine || gmv1_affine));
    if is_warp_splat {
        debug_assert!(bw4 > 1 && bh4 > 1);
        let use_local = motion_mode > MotionMode::InterIntra as u8;
        let (wm1, wm2) = if use_local {
            (recon.warpmv[0], recon.warpmv[1])
        } else {
            (
                recon.frm_hdr.gmv.m[ref0 as usize],
                recon.frm_hdr.gmv.m[ref1 as usize],
            )
        };
        let mut mf = (cwp_idx as i32) << 2;
        mf |= if use_local { 2 } else { 1 };
        let mut s_src = crate::refmvs::Block {
            r#ref: crate::levels::RefPair::from_refs(ref0, ref1),
            bs: bs as u8,
            mf: mf as i8,
            subpel_filter: b.inter_data().filter,
            ..Default::default()
        };
        if use_local {
            s_src.lmv = blk_mv;
            s_src.m = wm1.matrix;
            s_src.warp_type = wm1.wm_type as i8;
        } else {
            s_src.mv = blk_mv;
        }
        let mat1 = &wm1.matrix;
        let mat2 = &wm2.matrix;
        let mvx1 = (mat1[2] as i64 - 0x10000) * (bx as i64 + 1) * 4
            + mat1[3] as i64 * (by as i64 + 1) * 4
            + mat1[0] as i64;
        let mvy1 = mat1[4] as i64 * (bx as i64 + 1) * 4
            + mat1[1] as i64
            + (mat1[5] as i64 - 0x10000) * (by as i64 + 1) * 4;
        let mvx2 = (mat2[2] as i64 - 0x10000) * (bx as i64 + 1) * 4
            + mat2[3] as i64 * (by as i64 + 1) * 4
            + mat2[0] as i64;
        let mvy2 = mat2[4] as i64 * (bx as i64 + 1) * 4
            + mat2[1] as i64
            + (mat2[5] as i64 - 0x10000) * (by as i64 + 1) * 4;
        let mut t_src = crate::refmvs::TemporalBlock::default();
        t_src.r#ref = crate::levels::RefPair::from_refs(
            if t_swap { ref1 } else { ref0 },
            if t_swap { ref0 } else { ref1 },
        );
        let wedge_mask: Option<&[u8]> = if comp_type == 2 {
            let wedge_idx = b.inter_data().wedge_idx as usize;
            Some(
                recon
                    .masks
                    .wedge_tmvp(bs as usize, bw4 as usize, bh4 as usize, wedge_idx),
            )
        } else {
            None
        };
        let w_swap = b.inter_data().wedge_sign as i32 ^ t_swap as i32;
        if write_temporal {
            let t_stride = recon.rf.rp_stride;
            let t_off = (by >> 1) as isize * t_stride + (bx >> 1) as isize;
            crate::refmvs::splat_comp_warpmv(
                &mut recon.rt.r[s_off..],
                &mut s_src,
                Some(&mut recon.cur_mvs[t_off as usize..]),
                t_stride,
                &mut t_src,
                mvy1,
                mvx1,
                mvy2,
                mvx2,
                &wm1,
                &wm2,
                bw4,
                bh4,
                t_swap as usize,
                wedge_mask,
                w_swap,
            );
        } else {
            crate::refmvs::splat_comp_warpmv(
                &mut recon.rt.r[s_off..],
                &mut s_src,
                None,
                0,
                &mut t_src,
                mvy1,
                mvx1,
                mvy2,
                mvx2,
                &wm1,
                &wm2,
                bw4,
                bh4,
                t_swap as usize,
                wedge_mask,
                w_swap,
            );
        }
        return;
    }

    let mf = ((cwp_idx as i32) << 2
        | (inter_mode == CompInterPredMode::GlobalMvGlobalMv as u8 && imin(bw4, bh4) > 1) as i32)
        as i8;

    let mut s_src = crate::refmvs::Block {
        mv: blk_mv,
        r#ref: crate::levels::RefPair::from_refs(ref0, ref1),
        bs: bs as u8,
        mf,
        subpel_filter: b.inter_data().filter,
        ..Default::default()
    };

    let mut t_src = crate::refmvs::TemporalBlock::default();
    t_src.r#ref = crate::levels::RefPair::from_refs(
        if t_swap { ref1 } else { ref0 },
        if t_swap { ref0 } else { ref1 },
    );
    let wedge = comp_type == 2;
    // Quantize the (ref-flipped) MVs into the temporal block; for the WEDGE path
    t_src.mv = crate::refmvs::TemporalBlockMv::from_mvs(
        crate::refmvs::quantize_mv(blk_mv[t_swap as usize]),
        crate::refmvs::quantize_mv(blk_mv[!t_swap as usize]),
    );
    if write_temporal && wedge {
        // WEDGE temporal splat (splat_comp_wedgemv, refmvs.c:2473). The spatial
        // grid is splatted plainly; the temporal grid is filled per-2x2 from the
        // wedge TMVP mask (0=ref0, 1=ref1, 2=both).
        crate::refmvs::splat_mv(
            &mut recon.rt.r[s_off..],
            &mut s_src,
            None,
            0,
            &t_src,
            bw4,
            bh4,
        );
        let wedge_idx = b.inter_data().wedge_idx as usize;
        let wedge_sign = b.inter_data().wedge_sign as i32;
        let w_swap = (wedge_sign ^ t_swap as i32) != 0;
        let mask: &[u8] =
            recon
                .masks
                .wedge_tmvp(bs as usize, bw4 as usize, bh4 as usize, wedge_idx);
        let t_stride = recon.rf.rp_stride;
        let m0n = t_src.mv.mv_at(0).bits();
        let m1n = t_src.mv.mv_at(1).bits();
        let r = t_src.r#ref.refs();
        let mask_w = (bw4 >> 1) as usize;
        let mut row = 0i32;
        while row < bh4 {
            let row8 = (by >> 1) + (row >> 1);
            let mrow = (row >> 1) as usize;
            let mut x = 0i32;
            while x < bw4 {
                let col8 = (bx >> 1) + (x >> 1);
                let t_off = row8 as isize * t_stride + col8 as isize;
                if t_off >= 0 && (t_off as usize) < recon.cur_mvs.len() {
                    let d = mask[mrow * mask_w + (x >> 1) as usize] as i32;
                    let dst = &mut recon.cur_mvs[t_off as usize];
                    if d != 2 {
                        let idx = (!((d != 0) ^ w_swap)) as usize;
                        let m = t_src.mv.mv_at(idx).bits();
                        dst.mv = crate::refmvs::TemporalBlockMv::from_packed(m as u32 * 0x10001);
                        dst.r#ref = RefPair::from_pair(if m == crate::refmvs::INVALID_TRAJ {
                            -1
                        } else {
                            (r[idx] as u8 as i16).wrapping_mul(0x101)
                        });
                    } else if m0n == crate::refmvs::INVALID_TRAJ {
                        if m1n == crate::refmvs::INVALID_TRAJ {
                            dst.mv = crate::refmvs::TemporalBlockMv::from_packed(
                                crate::refmvs::INVALID_TRAJ as u32 * 0x10001,
                            );
                            dst.r#ref = RefPair::from_pair(-1);
                        } else {
                            dst.mv =
                                crate::refmvs::TemporalBlockMv::from_packed(m1n as u32 * 0x10001);
                            dst.r#ref = RefPair::from_pair((r[1] as u8 as i16).wrapping_mul(0x101));
                        }
                    } else if m1n == crate::refmvs::INVALID_TRAJ {
                        dst.mv = crate::refmvs::TemporalBlockMv::from_packed(m0n as u32 * 0x10001);
                        dst.r#ref = RefPair::from_pair((r[0] as u8 as i16).wrapping_mul(0x101));
                    } else {
                        *dst = t_src;
                    }
                }
                x += 2;
            }
            row += 2;
        }
    } else if write_temporal {
        {
            let m0 = t_src.mv.mv_at(0);
            let m1 = t_src.mv.mv_at(1);
            let r = t_src.r#ref.refs();
            if m0.bits() == crate::refmvs::INVALID_TRAJ {
                if m1.bits() == crate::refmvs::INVALID_TRAJ {
                    t_src.r#ref = crate::levels::RefPair::from_pair(-1);
                } else {
                    t_src.mv.set_mv(0, m1);
                    t_src.r#ref = crate::levels::RefPair::from_refs(r[1], r[1]);
                }
            } else if m1.bits() == crate::refmvs::INVALID_TRAJ {
                t_src.mv.set_mv(1, m0);
                t_src.r#ref = crate::levels::RefPair::from_refs(r[0], r[0]);
            }
        }
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
        // Spatial-only splat (opfl+refine blocks write temporal during recon).
        let _ = Mv::default();
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
}

/// Reconstruct a same-reference-pair COMPOUND inter block (8bpc). Predicts both
/// references into intermediate i16 buffers and blends per `comp_type`
///  - AVG: plain `avg`, or implicit out-of-bounds `mask` (imp_msk_bld), or
///    `w_avg` when CWP-weighted (cwp_idx != 8).
///  - WEDGE: `mask` blend with the wedge mask (luma + subsampled for chroma).
///  - SEG: luma `w_mask` (derives a subsampled seg mask), chroma `mask` reusing
///    that seg mask.
/// OPFL/refine-MV blocks (inter_mode >= OPFL_NEARMV_NEARMV, or refine_mv on an
/// AVG block) fill `tmp[0]`/`tmp[1]` via `opfl_pred_luma` / `rmv_uvpred` instead
/// of the plain two-reference MC loop; warp-compound / TIP / scaled refs are
/// handled elsewhere. After blend, the parsed residual is added with the same
/// per-TU walk as the single-ref path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn recon_b_inter_compound<
    BD: BitDepth,
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
    let ref0 = refs[0] as usize;
    let ref1 = refs[1] as usize;
    let mv0 = b.inter_data().mv[0].xy();
    let mv1 = b.inter_data().mv[1].xy();
    let mvs = [mv0, mv1];
    let mv_pair = [b.inter_data().mv[0], b.inter_data().mv[1]];
    let filter = b.inter_data().filter;
    let comp_type = b.inter_data().comp_type; // 1=AVG,2=WEDGE,3=SEG
    let cwp_idx = b.inter_data().cwp_idx as i32;
    let wedge_idx = b.inter_data().wedge_idx;
    let wedge_sign = b.inter_data().wedge_sign as usize;
    let mask_sign = b.inter_data().mask_sign as i32;
    let inter_mode = b.inter_data().inter_mode;
    let motion_mode = b.inter_data().motion_mode;
    let ss_hor = recon.frame.ss_hor;
    let ss_ver = recon.frame.ss_ver;

    let refp0 = match recon.refp[ref0].clone() {
        Some(p) => p,
        None => return Ok(()),
    };
    let refp1 = match recon.refp[ref1].clone() {
        Some(p) => p,
        None => return Ok(()),
    };
    let refp = [&refp0, &refp1];

    // cwp==8 and one ref MC partially out of bounds, an out-of-bounds difference
    // mask is used instead of a plain average.
    let imp_base = recon.seq_hdr.imp_msk_bld
        && motion_mode != MotionMode::WarpCausal as u8
        && inter_mode != CompInterPredMode::GlobalMvGlobalMv as u8
        && recon.svc[ref0][0].scale == 0
        && recon.svc[ref1][0].scale == 0;

    // block either signals an OPFL inter mode or requests implicit MV refinement
    // on an AVG-compound block. These use `opfl_pred` to fill `tmp[0]`/`tmp[1]`
    // (per-2x2/per-bs refined MC) instead of the plain two-reference MC loop.
    let refine_mv = b.inter_data().refine_mv;
    let is_opfl = inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8
        || (refine_mv != 0 && comp_type == 1);

    // reference a block uses warp-affine prediction (into the i16 tmp buffer)
    // when its inter mode is GLOBALMV_GLOBALMV with an affine-allowed global
    // motion, or the block is WARP_CAUSAL with a valid per-ref warp matrix.
    // Otherwise plain translational MC. `warp_use[i]` selects the path,
    // `warp_mat[i]` the matrix (local warp for WARP_CAUSAL, frame global motion
    // for GLOBALMV).
    let force_integer_mv = recon.frm_hdr.force_integer_mv;
    let mut warp_use = [false; 2];
    let mut warp_mat = [crate::headers::WarpedMotionParams::default(); 2];
    {
        let bdim = &BLOCK_DIMENSIONS[lbs as u8 as usize];
        let bw4d = bdim[0] as i32;
        let bh4d = bdim[1] as i32;
        for i in 0..2 {
            let mut gmv = recon.frm_hdr.gmv.m[refs[i] as usize];
            let gmv_warp_allowed = gmv.wm_type > crate::headers::WarpedMotionType::Translation
                && force_integer_mv == 0
                && crate::warpmv::get_shear_params(&mut gmv) == 0
                && recon.svc[refs[i] as usize][0].scale == 0;
            let warp = force_integer_mv == 0
                && ((inter_mode == CompInterPredMode::GlobalMvGlobalMv as u8
                    && imin(bw4d, bh4d) > 1
                    && gmv_warp_allowed)
                    || (motion_mode == MotionMode::WarpCausal as u8
                        && recon.warpmv[i].wm_type > crate::headers::WarpedMotionType::Invalid));
            warp_use[i] = warp;
            warp_mat[i] = if motion_mode >= MotionMode::WarpCausal as u8 {
                recon.warpmv[i]
            } else {
                recon.frm_hdr.gmv.m[refs[i] as usize]
            };
        }
    }

    let mut seg_mask = recon.scratch.take_compound_seg_mask();
    let mut seg_mask_stride = 0usize;
    if has_luma {
        let b_dim = &BLOCK_DIMENSIONS[lbs as u8 as usize];
        let bw4 = b_dim[0] as i32;
        let bh4 = b_dim[1] as i32;
        let w = (bw4 * 4) as usize;
        let h = (bh4 * 4) as usize;
        let y_stride = recon.frame.y_stride_px;
        let dst_off = 4 * (by as usize * y_stride + bx as usize);

        let _len = crate::mc_dispatch::compound_tmp_len(w, h);
        let mut tmp = recon.scratch.take_compound_tmp(_len);
        let mut opfl_bacp = false;
        if is_opfl {
            let w4 = imin(bw4, fi.bw - bx);
            let h4 = imin(bh4, fi.bh - by);
            opfl_bacp = opfl_pred_luma(
                recon,
                &mut tmp,
                &mut seg_mask,
                b,
                bx,
                by,
                bw4,
                bh4,
                w4,
                h4,
                fi,
            );
        } else {
            for i in 0..2 {
                if warp_use[i] {
                    warp_affine_plane_prep_8bpc(
                        recon.bd,
                        &mut tmp[i],
                        w,
                        refp[i],
                        0,
                        bx,
                        by,
                        b_dim,
                        &warp_mat[i],
                        ss_hor,
                        ss_ver,
                        fi.bw,
                        fi.bh,
                        recon.scratch.inter_mc_tmp_mut(),
                    );
                } else {
                    inter_mc_plane_prep_8bpc(
                        recon.bd,
                        &mut tmp[i],
                        refp[i],
                        0,
                        bx,
                        by,
                        bw4,
                        bh4,
                        mvs[i].x,
                        mvs[i].y,
                        filter,
                        ss_hor,
                        ss_ver,
                        fi.bw,
                        fi.bh,
                        recon.scratch.inter_mc_tmp_mut(),
                    );
                }
            }
        }

        let (tmp0, tmp1) = tmp.split_at(1);
        match comp_type {
            2 => {
                // WEDGE
                let mask = recon.masks.wedge_mask(
                    lbs as usize,
                    bw4 as usize,
                    bh4 as usize,
                    wedge_idx as usize,
                    0,
                );
                let (a0, a1) = if wedge_sign == 0 {
                    (&tmp0[0], &tmp1[0])
                } else {
                    (&tmp1[0], &tmp0[0])
                };
                mc_mask(
                    recon.bd,
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    a0,
                    a1,
                    w,
                    h,
                    mask,
                );
            }
            3 => {
                // SEG: luma w_mask derives subsampled seg mask for chroma reuse.
                seg_mask_stride = imin(bw4 * 4 >> ss_hor, 64) as usize;
                let (a0, a1) = if mask_sign == 0 {
                    (&tmp0[0], &tmp1[0])
                } else {
                    (&tmp1[0], &tmp0[0])
                };
                mc_w_mask(
                    recon.bd,
                    &mut recon.dst_y[dst_off..],
                    y_stride,
                    a0,
                    a1,
                    w,
                    h,
                    &mut seg_mask,
                    seg_mask_stride,
                    mask_sign,
                    ss_hor != 0,
                    ss_ver != 0,
                );
            }
            _ => {
                // AVG (or implicit mask / CWP weighted)
                if cwp_idx == 8 {
                    // For OPFL/refine blocks the BACP mask is already derived by
                    // the plain-MC `bacp==2` case).
                    let mut bacp = if is_opfl {
                        opfl_bacp as i32
                    } else {
                        2 * imp_base as i32
                    };
                    if bacp == 2 {
                        bacp = crate::recon::get_mask(
                            &mut seg_mask,
                            (bw4 * 4) as usize,
                            bx,
                            0,
                            by,
                            0,
                            &mv_pair,
                            3,
                            3,
                            bw4,
                            bh4,
                            fi.bw * 4,
                            fi.bh * 4,
                        ) as i32;
                    }
                    if bacp != 0 {
                        mc_mask(
                            recon.bd,
                            &mut recon.dst_y[dst_off..],
                            y_stride,
                            &tmp0[0],
                            &tmp1[0],
                            w,
                            h,
                            &seg_mask,
                        );
                    } else {
                        mc_avg(
                            recon.bd,
                            &mut recon.dst_y[dst_off..],
                            y_stride,
                            &tmp0[0],
                            &tmp1[0],
                            w,
                            h,
                        );
                    }
                } else {
                    mc_w_avg(
                        recon.bd,
                        &mut recon.dst_y[dst_off..],
                        y_stride,
                        &tmp0[0],
                        &tmp1[0],
                        w,
                        h,
                        cwp_idx,
                    );
                }
            }
        }
        recon.scratch.put_compound_tmp(tmp);

        // Luma residual (same walk as single-ref).
        let seg_id = b.seg_id as usize;
        let lossless = recon.frame.seg_lossless[seg_id] != 0;
        if lossless {
            let tx = if b.tx_size_ll != 0 {
                crate::tables::MAX_TXFM_SIZE_FOR_BS[lbs as usize][3] as usize
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
            let tp = &crate::tables::TX_PART_TBL[lbs as usize];
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

        // Chroma MC + blend run only at the recon stage (`Both`/`ReconOnly`);
        // the read stage (`ReadOnly`) only consumes the chroma coefficients
        let do_chroma_mc = chroma_stage != ChromaPhase::ReadOnly;

        // o_step = 4>>opfl.
        let opfl_refine = comp_type == 1 && refine_mv != 0;
        let opfl_mode = inter_mode >= CompInterPredMode::OpflNearMvNearMv as u8;
        let r_step = 2 << opfl_refine as i32;
        let o_step = 4 >> opfl_mode as i32;
        let r_pair = b.ref_pair;
        let mut chroma_bacp = false;

        // several luma sub-blocks (cbs != lbs, smallest luma dim < 16), chroma MC
        // is performed per luma 4x4 using each sub-block's own ref/MV/filter read
        // from the spatial refmvs grid (single-ref MC, not compound). This branch
        // takes priority over the compound chroma MC, so the per-plane compound
        // prediction loop is skipped when it applies.
        let (luma_bw4, luma_bh4) = {
            let ld = &BLOCK_DIMENSIONS[lbs as u8 as usize];
            (ld[0] as i32, ld[1] as i32)
        };
        let sub8x8 = lbs != BlockSize::Invalid && cbs != lbs && imin(luma_bw4, luma_bh4) < 16;
        if sub8x8 && do_chroma_mc {
            let base = ((cby & 63) as usize) * 128 + ((cbx & 127) as usize);
            for y in 0..ch4 {
                for x in 0..cw4 {
                    let idx = base + (y as usize) * 128 + (x as usize);
                    let r2 = &recon.rt.r[idx];
                    if r2.ox4 != 0 || r2.oy4 != 0 {
                        continue;
                    }
                    let s_ref0 = r2.r#ref.ref_at(0);
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

        for pl in (1..3usize).filter(|_| !sub8x8 && do_chroma_mc) {
            let dst_off = 4 * ((cby >> ss_ver) as usize * uv_stride + (cbx >> ss_hor) as usize);
            let _len = crate::mc_dispatch::compound_tmp_len(cw, ch);
            let mut tmp = recon.scratch.take_compound_tmp(_len);
            let mut opfl_bacp_chroma = false;
            if is_opfl {
                opfl_bacp_chroma = rmv_uvpred(
                    recon,
                    b,
                    &mut tmp,
                    pl - 1,
                    r_step,
                    o_step,
                    cbw4,
                    cbh4,
                    cbx,
                    cby,
                    r_pair,
                    false,
                    &mut seg_mask,
                    fi,
                );
                if pl == 1 {
                    chroma_bacp = opfl_bacp_chroma;
                }
            } else {
                for i in 0..2 {
                    if warp_use[i] {
                        warp_affine_plane_prep_8bpc(
                            bd,
                            &mut tmp[i],
                            cw,
                            refp[i],
                            pl,
                            cbx,
                            cby,
                            cb_dim,
                            &warp_mat[i],
                            ss_hor,
                            ss_ver,
                            fi.bw,
                            fi.bh,
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    } else {
                        inter_mc_plane_prep_8bpc(
                            bd,
                            &mut tmp[i],
                            refp[i],
                            pl,
                            cbx,
                            cby,
                            cbw4,
                            cbh4,
                            mvs[i].x,
                            mvs[i].y,
                            filter,
                            ss_hor,
                            ss_ver,
                            fi.bw,
                            fi.bh,
                            recon.scratch.inter_mc_tmp_mut(),
                        );
                    }
                }
            }
            let dst: &mut [BD::Pixel] = if pl == 1 {
                &mut recon.dst_u[dst_off..]
            } else {
                &mut recon.dst_v[dst_off..]
            };
            let (tmp0, tmp1) = tmp.split_at(1);
            match comp_type {
                2 => {
                    let mask = recon.masks.wedge_mask(
                        cbs as usize,
                        cbw4 as usize,
                        cbh4 as usize,
                        wedge_idx as usize,
                        (ss_hor + ss_ver) as usize,
                    );
                    let (a0, a1) = if wedge_sign == 0 {
                        (&tmp0[0], &tmp1[0])
                    } else {
                        (&tmp1[0], &tmp0[0])
                    };
                    mc_mask(bd, dst, uv_stride, a0, a1, cw, ch, mask);
                }
                3 => {
                    let (a0, a1) = if mask_sign == 0 {
                        (&tmp0[0], &tmp1[0])
                    } else {
                        (&tmp1[0], &tmp0[0])
                    };
                    mc_mask(bd, dst, uv_stride, a0, a1, cw, ch, &seg_mask);
                }
                _ => {
                    if cwp_idx == 8 {
                        // BACP (implicit masked blend) for chroma. For OPFL the
                        // per-plane mask is derived by `rmv_uvpred`. For plain
                        // compound AVG the chroma mask is recomputed from the
                        // block MVs with chroma-subsampled parameters on the U
                        // NOT inherited from the luma plane.
                        if is_opfl {
                            if chroma_bacp {
                                mc_mask(bd, dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch, &seg_mask);
                            } else {
                                mc_avg(bd, dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch);
                            }
                        } else {
                            if pl == 1 {
                                chroma_bacp = if imp_base {
                                    crate::recon::get_mask(
                                        &mut seg_mask,
                                        (cbw4 * 4 >> ss_hor) as usize,
                                        cbx >> ss_hor,
                                        0,
                                        cby >> ss_ver,
                                        0,
                                        &mv_pair,
                                        3 + ss_hor,
                                        3 + ss_ver,
                                        cbw4 >> ss_hor,
                                        cbh4 >> ss_ver,
                                        fi.bw * 4 >> ss_hor,
                                        fi.bh * 4 >> ss_ver,
                                    )
                                } else {
                                    false
                                };
                            }
                            if chroma_bacp {
                                mc_mask(bd, dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch, &seg_mask);
                            } else {
                                mc_avg(bd, dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch);
                            }
                        }
                    } else {
                        mc_w_avg(bd, dst, uv_stride, &tmp0[0], &tmp1[0], cw, ch, cwp_idx);
                    }
                }
            }
            let _ = opfl_bacp_chroma;
            recon.scratch.put_compound_tmp(tmp);
        }
        let _ = (seg_mask_stride, cb_dim);

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
    recon.scratch.put_compound_seg_mask(seg_mask);
    Ok(())
}
