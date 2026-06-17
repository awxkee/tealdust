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

use crate::intops::imin;
use crate::levels::*;

pub(crate) static MODE_CONV: [[[u8; 2]; 2]; 2] = [
    // DC_PRED
    [
        [DC_128_PRED, TOP_DC_PRED],
        [LEFT_DC_PRED, IntraPredMode::DcPred as u8],
    ],
    // PAETH_PRED
    [
        [DC_128_PRED, IntraPredMode::VertPred as u8],
        [IntraPredMode::HorPred as u8, IntraPredMode::PaethPred as u8],
    ],
];

#[derive(Clone, Copy, Default)]
pub(crate) struct EdgeMask {
    pub(crate) needs_left: bool,
    pub(crate) needs_top: bool,
    pub(crate) needs_topleft: bool,
    pub(crate) needs_topright: bool,
    pub(crate) needs_bottomleft: bool,
}

impl EdgeMask {
    const fn new(left: bool, top: bool, tl: bool, tr: bool, bl: bool) -> Self {
        Self {
            needs_left: left,
            needs_top: top,
            needs_topleft: tl,
            needs_topright: tr,
            needs_bottomleft: bl,
        }
    }
}

pub(crate) fn intra_prediction_edge(mode: u8) -> EdgeMask {
    match mode {
        0  /* DcPred */       => EdgeMask::new(true,  true,  false, false, false),
        1  /* VertPred */     => EdgeMask::new(false, true,  false, false, false),
        2  /* HorPred */      => EdgeMask::new(true,  false, false, false, false),
        _ if mode == LEFT_DC_PRED  => EdgeMask::new(true,  false, false, false, false),
        _ if mode == TOP_DC_PRED   => EdgeMask::new(false, true,  false, false, false),
        _ if mode == DC_128_PRED   => EdgeMask::new(false, false, false, false, false),
        _ if mode == Z1_PRED       => EdgeMask::new(false, true,  true,  true,  false),
        _ if mode == Z2_PRED       => EdgeMask::new(true,  true,  true,  false, false),
        _ if mode == Z3_PRED       => EdgeMask::new(true,  false, true,  false, true),
        9  /* SmoothPred */   => EdgeMask::new(true,  true,  false, true,  true),
        10 /* SmoothVPred */  => EdgeMask::new(false, true,  false, false, true),
        11 /* SmoothHPred */  => EdgeMask::new(true,  false, false, true,  false),
        12 /* PaethPred */    => EdgeMask::new(true,  true,  true,  false, false),
        _ if mode == DIP_PRED      => EdgeMask::new(true,  true,  true,  true,  true),
        _ => EdgeMask::default(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_intra_edges<BD: crate::pixel::BitDepth>(
    bd: BD,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    n_tr: i32,
    n_bl: i32,
    dst: &[BD::Pixel],
    dst_off: usize,
    stride: usize,
    prefilter_toplevel_sb_edge: Option<&[BD::Pixel]>,
    mode: u8,
    tw4: i32,
    th4: i32,
    intra_flags: i32,
    tl: &mut [BD::Pixel],
    tl_o: usize,
) -> u8 {
    use crate::pixel::Pixel;
    debug_assert!(y < h && x < w);
    let mid = (bd.bitdepth_max() + 1) >> 1;
    let fill_left = BD::Pixel::from_i32(mid + 1); // 129 @ 8bpc
    let fill_top = BD::Pixel::from_i32(mid - 1); // 127 @ 8bpc
    let fill_tl = BD::Pixel::from_i32(mid); // 128 @ 8bpc

    let mut is_dir = false;
    let enable_edge_filter = (intra_flags & ANGLE_USE_EDGE_FILTER_FLAG) != 0;
    let angle = intra_flags & 511;
    let apply_dip = (intra_flags & ANGLE_DIP_FLAG) != 0;
    let apply_ibp = (intra_flags & ANGLE_IBP_FLAG) != 0;
    let mrl_idx = ((intra_flags & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let mrl_mul = (intra_flags & ANGLE_MULTI_MRL_FLAG) != 0;
    let have_left = (intra_flags & ANGLE_HAS_LEFT_FLAG) != 0;
    let have_top = (intra_flags & ANGLE_HAS_TOP_FLAG) != 0;

    let mut mode = mode;
    let mut tl_filter = false;

    match mode {
        1..=8 => {
            is_dir = true;
            if angle <= 90 {
                mode = if angle < 90 && (have_top || apply_ibp) {
                    Z1_PRED
                } else {
                    IntraPredMode::VertPred as u8
                };
            } else if angle < 180 {
                mode = Z2_PRED;
            } else {
                mode = if angle > 180 && (have_left || apply_ibp) {
                    Z3_PRED
                } else {
                    IntraPredMode::HorPred as u8
                };
            }
            tl_filter = (Z1_PRED..=Z3_PRED).contains(&mode)
                && have_left
                && have_top
                && mrl_idx == 0
                && enable_edge_filter
                && tw4 + th4 >= 6;
        }
        0 => {
            mode = if apply_dip {
                DIP_PRED
            } else {
                MODE_CONV[0][have_left as usize][have_top as usize]
            };
        }
        12 => {
            debug_assert!(!apply_dip);
            mode = MODE_CONV[1][have_left as usize][have_top as usize];
        }
        _ => {}
    }
    debug_assert!(mrl_idx == 0 || is_dir);

    let mut e = intra_prediction_edge(mode);
    if (mode == Z1_PRED || mode == Z3_PRED) && apply_ibp {
        e = intra_prediction_edge(DIP_PRED);
    }

    let mut top_buf: &[BD::Pixel] = dst;
    let mut dst_top_off: usize = 0;
    let mut dst_top2_off: usize = 0;
    let mut top_stride_val: usize = stride;

    if have_top
        && ((e.needs_top || e.needs_topleft || e.needs_topright)
            || ((e.needs_left || e.needs_bottomleft) && !have_left))
    {
        if let Some(prefilter) = prefilter_toplevel_sb_edge {
            top_buf = prefilter;
            dst_top_off = x as usize * 4;
            dst_top2_off = x as usize * 4;
            top_stride_val = 0;
        } else {
            dst_top_off = dst_off - (mrl_idx + 1) * stride;
            dst_top2_off = dst_off - stride;
        }
    }

    let tw = (tw4 as usize) << 2;
    let th = (th4 as usize) << 2;
    let diag_mrl_idx = if (Z1_PRED..=Z3_PRED).contains(&mode) {
        mrl_idx
    } else {
        0
    };
    let e_stride = (tw + th) * 2 + diag_mrl_idx * 3 + 1;
    let o = tl_o as isize;

    // Left edge
    if e.needs_left || tl_filter {
        let mut sz = if e.needs_left { th } else { 1 };
        let mut sz2 = th;
        if e.needs_bottomleft {
            sz += if apply_dip {
                th >> 2
            } else if is_dir {
                tw + 2 * diag_mrl_idx
            } else {
                1
            };
            sz2 = sz - 2 * diag_mrl_idx;
        }
        let left_base = o - diag_mrl_idx as isize - 1;
        let left2_base = o + e_stride as isize - 1;

        if have_left {
            let left_src = dst_off - 1 - mrl_idx;
            let mut px_have = if e.needs_left {
                imin(th as i32, (h - y) << 2) as usize
            } else {
                1
            };
            let mut i = 0usize;
            while i < px_have {
                tl[(left_base - i as isize) as usize] = dst[left_src + stride * i];
                i += 1;
            }
            if e.needs_bottomleft && n_bl > 0 {
                px_have += imin(n_bl << 2, (sz - th) as i32) as usize;
                while i < px_have {
                    tl[(left_base - i as isize) as usize] = dst[left_src + stride * i];
                    i += 1;
                }
            }
            if px_have < sz {
                let fill_val = tl[(left_base + 1 - i as isize) as usize];
                let start = (left_base + 1 - sz as isize) as usize;
                tl[start..start + sz - px_have].fill(fill_val);
            }
            if mrl_mul {
                let left2_src = dst_off - 1;
                let px2 = imin(i as i32, sz2 as i32) as usize;
                for j in 0..px2 {
                    tl[(left2_base - j as isize) as usize] = dst[left2_src + stride * j];
                }
                if px2 < sz2 {
                    let fill_val = tl[(left2_base + 1 - px2 as isize) as usize];
                    let start = (left2_base + 1 - sz2 as isize) as usize;
                    tl[start..start + sz2 - px2].fill(fill_val);
                }
            }
        } else {
            let fill_val = if have_top {
                top_buf[dst_top_off]
            } else {
                fill_left
            };
            let start = (left_base + 1 - sz as isize) as usize;
            tl[start..start + sz].fill(fill_val);
            if mrl_mul {
                let fill_val2 = if have_top {
                    top_buf[dst_top2_off]
                } else {
                    fill_left
                };
                let start2 = (left2_base + 1 - sz2 as isize) as usize;
                tl[start2..start2 + sz2].fill(fill_val2);
            }
        }
    } else if e.needs_bottomleft {
        debug_assert!(mode == IntraPredMode::SmoothVPred as u8);
        let bl_idx = (o - 1 - th as isize) as usize;
        if !have_left {
            tl[bl_idx] = if have_top {
                top_buf[dst_top_off]
            } else {
                fill_left
            };
        } else if n_bl <= 0 {
            let row = imin(th as i32, (h - y) << 2) as usize - 1;
            tl[bl_idx] = dst[dst_off + stride * row - 1];
        } else {
            tl[bl_idx] = dst[dst_off + stride * th - 1];
        }
    }

    // Top edge
    if e.needs_top || tl_filter {
        let mut sz = if e.needs_top { tw } else { 1 };
        let mut sz2 = tw;
        if e.needs_topright {
            sz += if apply_dip {
                tw >> 2
            } else if is_dir {
                th + 2 * diag_mrl_idx
            } else {
                1
            };
            sz2 = sz - 2 * diag_mrl_idx;
        }
        let top_base = (o + diag_mrl_idx as isize + 1) as usize;
        let top2_base = (o + e_stride as isize + 1) as usize;

        if have_top {
            let mut px_have = if e.needs_top {
                imin(tw as i32, (w - x) << 2) as usize
            } else {
                1
            };
            tl[top_base..top_base + px_have]
                .copy_from_slice(&top_buf[dst_top_off..dst_top_off + px_have]);
            if e.needs_topright && n_tr > 0 {
                px_have += imin(n_tr << 2, (sz - tw) as i32) as usize;
                tl[top_base + tw..top_base + px_have]
                    .copy_from_slice(&top_buf[dst_top_off + tw..dst_top_off + px_have]);
            }
            if px_have < sz {
                let fill_val = tl[top_base + px_have - 1];
                tl[top_base + px_have..top_base + sz].fill(fill_val);
            }
            if mrl_mul {
                let px2 = imin(px_have as i32, sz2 as i32) as usize;
                tl[top2_base..top2_base + px2]
                    .copy_from_slice(&top_buf[dst_top2_off..dst_top2_off + px2]);
                if px2 < sz2 {
                    let fill_val = tl[top2_base + px2 - 1];
                    tl[top2_base + px2..top2_base + sz2].fill(fill_val);
                }
            }
        } else {
            let fill_val = if have_left {
                dst[dst_off - 1 - mrl_idx]
            } else {
                fill_top
            };
            tl[top_base..top_base + sz].fill(fill_val);
            if mrl_mul {
                let fill_val2 = if have_left {
                    dst[dst_off - 1]
                } else {
                    fill_top
                };
                tl[top2_base..top2_base + sz2].fill(fill_val2);
            }
        }
    } else if e.needs_topright {
        debug_assert!(mode == IntraPredMode::SmoothHPred as u8);
        let tr_idx = (o + 1) as usize + tw;
        if !have_top {
            tl[tr_idx] = if have_left {
                dst[dst_off - 1]
            } else {
                fill_top
            };
        } else if n_tr <= 0 {
            let col = imin(tw as i32, (w - x) << 2) as usize - 1;
            tl[tr_idx] = top_buf[dst_top_off + col];
        } else {
            tl[tr_idx] = top_buf[dst_top_off + tw];
        }
    }

    // Topleft pixel
    if e.needs_topleft {
        debug_assert!(diag_mrl_idx == mrl_idx);
        if have_top && have_left {
            for i in (-(mrl_idx as isize))..0 {
                tl[(o + i) as usize] = top_buf[(dst_top_off as isize - mrl_idx as isize - 1
                    + (-i) * top_stride_val as isize)
                    as usize];
            }
            for i in 0..=mrl_idx as isize {
                tl[(o + i) as usize] =
                    top_buf[(dst_top_off as isize - mrl_idx as isize - 1 + i) as usize];
            }
        } else {
            let v = if have_left {
                dst[dst_off - 1 - mrl_idx]
            } else if have_top {
                top_buf[dst_top_off]
            } else {
                fill_tl
            };
            let start = (o - mrl_idx as isize) as usize;
            tl[start..start + 2 * mrl_idx + 1].fill(v);
        }
        tl[(o + e_stride as isize) as usize] = if have_left {
            if have_top {
                top_buf[dst_top2_off - 1]
            } else {
                dst[dst_off - 1]
            }
        } else if have_top {
            top_buf[dst_top2_off]
        } else {
            fill_tl
        };

        if tl_filter {
            let c0: i32 = tl[tl_o].into();
            let cm: i32 = tl[tl_o - 1].into();
            let cp: i32 = tl[tl_o + 1].into();
            let c = c0 + (cm + c0 + cp) * 5;
            tl[tl_o] = BD::Pixel::from_i32((c + 8) >> 4);
        }
    }

    mode
}
