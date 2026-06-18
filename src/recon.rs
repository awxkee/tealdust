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

use crate::headers::PixelLayout;
use crate::intops::{apply_sign, iclip, imax, imin, ulog2, umin};
use crate::levels::{IntraPredMode, Mv, N_BS_SIZES, RefPair, txtp};
use crate::mc::OpflRegressionData;
use crate::msac::MsacContext;
use crate::refmvs::{self, INVALID_TRAJ, TemporalBlock};
use crate::scan::SCANS;
use crate::tables::{
    BLOCK_DIMENSIONS, DIV_RECIP, MODE_TO_ANGLE_MAP, TXFM_DIMENSIONS, TXTP_FROM_UVMODE, TxfmInfo,
};
use crate::warpmv::resolve_divisor_32;

#[inline]
pub(crate) fn decode_exp_golomb(msac: &mut MsacContext, k: u32) -> u32 {
    let length = msac.decode_unary_bypass(21) + k;
    let x = (1u32 << length) + msac.decode_bools_bypass(length);
    x - (1 << k)
}

#[inline]
pub(crate) fn decode_hr(msac: &mut MsacContext, hr_avg: i32) -> i32 {
    let m = ulog2(iclip(hr_avg, 2, 64) as u32) as u32;
    let cmax = imin(m as i32 + 4, 6) as u32;
    let q = msac.decode_unary_bypass(cmax);
    let rem = if q == cmax {
        decode_exp_golomb(msac, m + 1)
    } else {
        msac.decode_bools_bypass(m)
    };
    (rem + (q << m)) as i32
}

#[inline]
pub(crate) fn tcq_next_state(state: i32, abs_level: i32) -> i32 {
    (((state & 0x4) ^ (((abs_level & 1) ^ (state & 0x1)) << 2))
        | ((state & 0x6) >> 1)
        | -0x80000000i32)
        & (state >> 31)
}

pub(crate) fn wide_angle_remap(
    t_dim: &TxfmInfo,
    mode: IntraPredMode,
    angle: &mut i32,
    mrl_idx: i32,
) -> IntraPredMode {
    let mode_u8 = mode as u8;
    if mode_u8.wrapping_sub(1) > IntraPredMode::VertLeftPred as u8 - 1 {
        return mode;
    }

    let mrl_adj = (mrl_idx == 1) as i32 - (mrl_idx == 2) as i32;
    *angle = MODE_TO_ANGLE_MAP[(mode_u8 - 1) as usize] as i32 + *angle * 3 + mrl_adj;

    static THRESH: [u8; 4] = [61, 73, 82, 86];
    let rect = t_dim.lw as i32 - t_dim.lh as i32;

    if rect > 0 {
        debug_assert!(rect <= 4);
        if *angle > 270 - THRESH[(rect - 1) as usize] as i32 {
            *angle -= 180;
            return IntraPredMode::DiagDownLeftPred;
        }
    } else if rect < 0 {
        debug_assert!(rect >= -4);
        if *angle < THRESH[(-1 - rect) as usize] as i32 {
            *angle += 180;
            return IntraPredMode::HorUpPred;
        }
    }

    mode
}

pub(crate) fn gen_mask(
    mask: &mut [u8],
    stride: usize,
    bw: i32,
    bh: i32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    fw: u32,
    fh: u32,
) {
    let mut off = 0;
    for y in 0..bh {
        for x in 0..bw {
            let p0 = ((x0 + x) as u32) < fw && ((y0 + y) as u32) < fh;
            let p1 = ((x1 + x) as u32) < fw && ((y1 + y) as u32) < fh;
            mask[off + x as usize] = (32 * (p0 as i32 - p1 as i32 + 1)) as u8;
        }
        off += stride;
    }
}

pub(crate) fn derive_alpha(num: i32, den: i32, mut alpha: i32) -> i32 {
    let max = (2 << 8) - 1;
    if num != 0 && den != 0 {
        let num_abs = num.abs();
        let shift_n = ulog2(num_abs as u32);
        debug_assert!(den >= 0);
        let shift_d = ulog2(den as u32);
        let e_d = den - (1 << shift_d);
        let f_d = if shift_d > 7 {
            (e_d + (1 << (shift_d - 8))) >> (shift_d - 7)
        } else {
            e_d << (7 - shift_d)
        };
        let f_n = if shift_n > 7 {
            (num_abs + (1 << (shift_n - 8))) >> (shift_n - 7)
        } else {
            num_abs << (7 - shift_n)
        };
        let shift_add = shift_d - shift_n - 8;
        if shift_add <= 1 {
            let shift0 = 9 + 7 + shift_add;
            let tmp_alpha = if shift0 < 0 {
                max
            } else {
                imin((DIV_RECIP[f_d as usize] as i32 * f_n) >> shift0, max)
            };
            if tmp_alpha != 0 {
                alpha = apply_sign(tmp_alpha, num);
            }
        }
    }
    alpha
}

fn read_u16_ne(a: &[u8]) -> u16 {
    u16::from_ne_bytes(a[..2].try_into().unwrap())
}

fn read_u32_ne(a: &[u8]) -> u32 {
    u32::from_ne_bytes(a[..4].try_into().unwrap())
}

fn read_u64_ne(a: &[u8]) -> u64 {
    u64::from_ne_bytes(a[..8].try_into().unwrap())
}

pub(crate) fn get_skip_ctx(
    t_dim: &TxfmInfo,
    bs: usize,
    a: &[u8],
    l: &[u8],
    plane: i32,
    u_has_cf: i32,
    ss_hor: bool,
    ss_ver: bool,
) -> u32 {
    debug_assert!(bs < N_BS_SIZES);
    let b_dim = &BLOCK_DIMENSIONS[bs];

    if plane != 0 {
        let not_one_blk = (b_dim[2] - (b_dim[2] != 0 && ss_hor) as u8 > t_dim.lw)
            || (b_dim[3] - (b_dim[3] != 0 && ss_ver) as u8 > t_dim.lh);

        let ca: bool = match t_dim.lw {
            0 => a[0] != 0x40,
            1 => read_u16_ne(a) != 0x4040,
            2 => read_u32_ne(a) != 0x40404040,
            3 => read_u64_ne(a) != 0x4040404040404040,
            4 => (read_u64_ne(a) | read_u64_ne(&a[8..])) != 0x4040404040404040,
            _ => unreachable!(),
        };
        let cl: bool = match t_dim.lh {
            0 => l[0] != 0x40,
            1 => read_u16_ne(l) != 0x4040,
            2 => read_u32_ne(l) != 0x40404040,
            3 => read_u64_ne(l) != 0x4040404040404040,
            4 => (read_u64_ne(l) | read_u64_ne(&l[8..])) != 0x4040404040404040,
            _ => unreachable!(),
        };

        let offset = if plane == 1 {
            6
        } else {
            6 * u_has_cf + not_one_blk as i32 * 3
        } as u32;
        offset + ca as u32 + cl as u32
    } else if b_dim[2] == t_dim.lw && b_dim[3] == t_dim.lh {
        0
    } else {
        let merge = |dir: &[u8], tx: u8| -> u32 {
            let mut v: u32;
            if tx == 4 {
                let tmp = read_u64_ne(dir) | read_u64_ne(&dir[8..]);
                v = (tmp >> 32) as u32 | tmp as u32;
            } else {
                v = match tx {
                    0 => dir[0] as u32,
                    1 => read_u16_ne(dir) as u32,
                    2 | 3 => read_u32_ne(dir),
                    _ => unreachable!(),
                };
            }
            if tx == 3 {
                v |= read_u32_ne(&dir[4..]);
            }
            if tx >= 2 {
                v |= v >> 16;
            }
            if tx >= 1 {
                v |= v >> 8;
            }
            v
        };
        let la = merge(a, t_dim.lw);
        let ll = merge(l, t_dim.lh);
        (umin(la & 0x3F, 4) + umin(ll & 0x3F, 4) + 3) >> 1
    }
}

pub(crate) fn get_dc_sign_ctx(t_dim: &TxfmInfo, a: &[u8], l: &[u8]) -> u32 {
    let mask: u64 = 0xC0C0C0C0C0C0C0C0;
    let mul: u64 = 0x0101010101010101;
    let mut t: u64 = 0;

    for &(edge, len) in &[(a, t_dim.lw), (l, t_dim.lh)] {
        match len {
            0 => t += (edge[0] >> 6) as u64,
            1 => t += (read_u16_ne(edge) as u64 & mask) >> 6,
            2 => t += (read_u32_ne(edge) as u64 & mask) >> 6,
            3 => t += (read_u64_ne(edge) & mask) >> 6,
            4 => {
                t += (read_u64_ne(&edge[8..]) & mask) >> 6;
                t += (read_u64_ne(edge) & mask) >> 6;
            }
            _ => unreachable!(),
        }
    }

    t = t.wrapping_mul(mul);
    let s = (t >> 56) as i32 - t_dim.w as i32 - t_dim.h as i32;
    (s != 0) as u32 + (s > 0) as u32
}

#[inline]
pub(crate) fn get_lo_ctx(
    levels: &[i8],
    off: usize,
    tx_class: u8,
    hi_mag: &mut u32,
    xy: u32,
    plane: i32,
    stride: usize,
) -> u32 {
    let chroma = plane != 0;
    let lo_freq = xy
        < if chroma {
            1
        } else if tx_class == 0 {
            4
        } else {
            2
        };
    let mut lim: u32 = if lo_freq { 5 } else { 3 };
    let mut lo_mag: u32 = 0;
    let mut hi: u32 = 0;

    macro_rules! add {
        ($v:expr) => {{
            let val = $v as u32;
            lo_mag += val.min(lim);
            hi += val.min(5);
        }};
    }

    add!(levels[off + 1]);
    add!(levels[off + stride]);

    let offset: u32;
    if tx_class == 0 {
        add!(levels[off + stride + 1]);
        if !chroma {
            lo_mag +=
                (levels[off + 2] as u32).min(lim) + (levels[off + 2 * stride] as u32).min(lim);
            if lo_freq {
                offset = if xy == 0 {
                    0
                } else if xy < 2 {
                    9
                } else {
                    16
                };
                lim = if xy == 0 {
                    8
                } else if xy < 2 {
                    6
                } else {
                    4
                };
            } else {
                offset = if xy < 6 {
                    0
                } else if xy < 8 {
                    5
                } else {
                    10
                };
                lim = 4;
            }
        } else {
            lim = 3;
            offset = if plane == 1 { 0 } else { 4 };
        }
    } else {
        if !chroma {
            lim = 3;
            add!(levels[off + 2]);
            lo_mag += (levels[off + 3] as u32).min(3) + (levels[off + 4] as u32).min(3);
            if lo_freq {
                offset = if xy == 0 { 21 } else { 28 };
                lim = if xy == 0 { 6 } else { 4 };
            } else {
                offset = 15;
                lim = 4;
            }
        } else {
            offset = 8;
            lim = 3;
        }
    }

    *hi_mag = (if !chroma && lo_freq && (xy > 0 || tx_class != 0) {
        7
    } else {
        0
    }) + ((hi + 1) >> 1).min(if chroma { 3 } else { 6 });
    offset + ((lo_mag + 1) >> 1).min(lim)
}

#[inline]
pub(crate) fn get_lo_ctx_idtx(levels: &[i8], off: usize, hi_mag: &mut u32, stride: usize) -> u32 {
    let v0 = levels[off - 1] as u32;
    let v1 = levels[off - stride] as u32;
    let lo_mag = v0.min(3) + v1.min(3);
    let hi = v0.min(5) + v1.min(5);
    *hi_mag = hi.min(6);
    lo_mag
}

#[inline]
pub(crate) fn get_sign_ctx_idtx(levels: &[i8], off: usize, stride: usize) -> u32 {
    let sum =
        levels[off - 1] as i32 + levels[off - stride] as i32 + levels[off - stride - 1] as i32;
    let offset = if levels[off] > 3 { 2 } else { 0 };
    match sum {
        -3 => offset + 6,
        -2 | -1 => offset + 2,
        0 => 0,
        1 | 2 => offset + 1,
        3 => offset + 5,
        _ => unreachable!(),
    }
}

pub(crate) fn get_mask(
    mask: &mut [u8],
    stride: usize,
    bx4: i32,
    x4: i32,
    by4: i32,
    y4: i32,
    mv: &[Mv; 2],
    h_subpel_bits: i32,
    v_subpel_bits: i32,
    bw4: i32,
    bh4: i32,
    iw: i32,
    ih: i32,
) -> bool {
    let (mv0, mv1) = (mv[0].xy(), mv[1].xy());
    let x0 = (bx4 + x4) * 4 + (mv0.x >> h_subpel_bits);
    let y0 = (by4 + y4) * 4 + (mv0.y >> v_subpel_bits);
    let x1 = (bx4 + x4) * 4 + (mv1.x >> h_subpel_bits);
    let y1 = (by4 + y4) * 4 + (mv1.y >> v_subpel_bits);
    if x0 < 0
        || x1 < 0
        || y0 < 0
        || y1 < 0
        || x0 + bw4 * 4 >= iw
        || x1 + bw4 * 4 >= iw
        || y0 + bh4 * 4 >= ih
        || y1 + bh4 * 4 >= ih
    {
        let off = (y4 as usize * stride + x4 as usize) * 4;
        gen_mask(
            &mut mask[off..],
            stride,
            bw4 * 4,
            bh4 * 4,
            x0,
            y0,
            x1,
            y1,
            iw as u32,
            ih as u32,
        );
        return true;
    }
    false
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct OpflMvDelta {
    pub(crate) x: i8,
    pub(crate) y: i8,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct OpflMvDeltaBlock {
    pub(crate) d: [OpflMvDelta; 2],
}

pub(crate) fn opfl_mv_adj(r: &OpflRegressionData, dd: &mut OpflMvDeltaBlock, d: [i8; 2]) {
    let mut su2 = r.su2;
    let mut suv = r.suv;
    let mut sv2 = r.sv2;
    let mut suw = r.suw;
    let mut svw = r.svw;
    let nbits_su2 = 1 + ulog2((su2 + (su2 == 0) as i32) as u32);
    let nbits_sv2 = 1 + ulog2((sv2 + (sv2 == 0) as i32) as u32);
    let nbits_suv = 1 + ulog2((suv.abs() + (suv == 0) as i32) as u32);
    let nbits_suw = 1 + ulog2((suw.abs() + (suw == 0) as i32) as u32);
    let nbits_svw = 1 + ulog2((svw.abs() + (svw == 0) as i32) as u32);
    let nbits_max = imax(
        nbits_su2 + nbits_sv2,
        imax(
            imax(nbits_sv2 + nbits_suw, nbits_suv + nbits_svw),
            imax(nbits_su2 + nbits_svw, nbits_suv + nbits_suw),
        ),
    );
    let rbits = imax(0, nbits_max - 23) >> 1;
    if rbits != 0 {
        let rnd = (1 << rbits) >> 1;
        su2 = (su2 + rnd) >> rbits;
        sv2 = (sv2 + rnd) >> rbits;
        suv = (suv + rnd - (suv < 0) as i32) >> rbits;
        suw = (suw + rnd - (suw < 0) as i32) >> rbits;
        svw = (svw + rnd - (svw < 0) as i32) >> rbits;
    }
    let det = su2 * sv2 - suv * suv;
    if det > 0 {
        let mut s = [sv2 * suw - suv * svw, su2 * svw - suv * suw];
        let mut shift = 0i32;
        let idet = resolve_divisor_32(det as u32, &mut shift);
        let idet_bits = ulog2(idet as u32);
        for i in 0..2 {
            if s[i] == 0 {
                continue;
            }
            let mut abss = s[i].abs();
            let rb = imax(0, ulog2(abss as u32) + idet_bits - 22);
            if rb > 0 {
                abss = (abss + ((1 << rb) >> 1)) >> rb;
            }
            let ibits = 3 + rb - shift;
            if ibits >= 0 {
                abss = abss * idet * (1 << ibits);
            } else {
                abss = (abss * idet + ((1 << -ibits) >> 1)) >> -ibits;
            }
            s[i] = apply_sign(abss, s[i]);
        }
        dd.d[0].x = -iclip(d[0] as i32 * s[0], -16, 16) as i8;
        dd.d[0].y = -iclip(d[0] as i32 * s[1], -16, 16) as i8;
        dd.d[1].x = iclip(d[1] as i32 * s[0], -16, 16) as i8;
        dd.d[1].y = iclip(d[1] as i32 * s[1], -16, 16) as i8;
    } else {
        *dd = OpflMvDeltaBlock::default();
    }
}

pub(crate) fn scaledown_16pel_mv_for_chroma(mv: &mut [Mv; 2], layout: PixelLayout) {
    match layout {
        PixelLayout::I420 => {
            for m in mv.iter_mut() {
                let y = m.y();
                m.set_y((y + (y > 0) as i32) >> 1);
            }
            for m in mv.iter_mut() {
                let x = m.x();
                m.set_x((x + (x > 0) as i32) >> 1);
            }
        }
        PixelLayout::I422 => {
            for m in mv.iter_mut() {
                let x = m.x();
                m.set_x((x + (x > 0) as i32) >> 1);
            }
        }
        _ => {}
    }
}

pub(crate) fn scaleup_8pel_mv_for_chroma(mv: &mut [Mv; 2], layout: PixelLayout) {
    match layout {
        PixelLayout::I444 => {
            for m in mv.iter_mut() {
                m.set_x(m.x() << 1);
            }
            for m in mv.iter_mut() {
                m.set_y(m.y() << 1);
            }
        }
        PixelLayout::I422 => {
            for m in mv.iter_mut() {
                m.set_y(m.y() << 1);
            }
        }
        _ => {}
    }
}

pub(crate) fn update_temporal(
    t_dst: &mut [TemporalBlock],
    t_stride: usize,
    w8: usize,
    h8: usize,
    r: RefPair,
    mv: &[Mv; 2],
    swap: bool,
) {
    let s0 = swap as usize;
    let s1 = (!swap) as usize;
    let refs = r.refs();
    let mut r0 = refs[s0];
    let mut r1 = refs[s1];
    let mut mv0 = refmvs::quantize_mv(mv[s0]);
    let mut mv1 = refmvs::quantize_mv(mv[s1]);

    let mut ref_pair = RefPair::from_refs(r0, r1);
    let mv0_n = mv0.packed();
    let mv1_n = mv1.packed();
    if mv0_n == INVALID_TRAJ {
        if mv1_n == INVALID_TRAJ {
            ref_pair = RefPair::from_pair(-1);
        } else {
            mv0 = mv1;
            r0 = r1;
            ref_pair = RefPair::from_refs(r0, r1);
        }
    } else if mv1_n == INVALID_TRAJ {
        mv1 = mv0;
        r1 = r0;
        ref_pair = RefPair::from_refs(r0, r1);
    }

    let t_src = TemporalBlock {
        mv: refmvs::TemporalBlockMv::from_mvs(mv0, mv1),
        r#ref: ref_pair,
    };
    for y in 0..h8 {
        let row = &mut t_dst[y * t_stride..y * t_stride + w8];
        for x in 0..w8 {
            row[x] = t_src;
        }
    }
}

pub(crate) struct DecodeCoefParams<'a> {
    pub(crate) tx: usize,
    pub(crate) bs: usize,
    pub(crate) plane: i32,
    pub(crate) intra: bool,
    pub(crate) fsc: bool,
    pub(crate) lossless: bool,
    pub(crate) sdp_active: bool,
    pub(crate) y_mode: usize,
    pub(crate) uv_mode: usize,
    pub(crate) _seg_id: usize,
    pub(crate) seq_fsc: bool,
    pub(crate) seq_ist: [bool; 2],
    pub(crate) seq_cctx: bool,
    pub(crate) chroma_dctonly: bool,
    pub(crate) reduced_txtp_set: i32,
    pub(crate) tcq_enabled: bool,
    pub(crate) layout: PixelLayout,
    pub(crate) u_has_cf: i32,
    pub(crate) cbx: i32,
    pub(crate) cby: i32,
    pub(crate) luma_fsc_map: &'a [u8],
    pub(crate) dq_tbl: [u32; 2],
    pub(crate) bitdepth: u32,
    pub(crate) qm: Option<&'a [u8]>,
    pub(crate) ss_hor: bool,
    pub(crate) ss_ver: bool,
}

use crate::cdf::{CdfCoefContext, CdfModeContext};

pub(crate) fn decode_coefs(
    msac: &mut MsacContext,
    coef: &mut CdfCoefContext,
    mode: &mut CdfModeContext,
    a: &[u8],
    l: &[u8],
    p: &DecodeCoefParams,
    cf: &mut [i32],
    txtp: &mut u16,
    res_ctx: &mut u8,
) -> i32 {
    let t_dim = &TXFM_DIMENSIONS[p.tx];
    let chroma = p.plane != 0;
    let cf_max = !((!127u32) << p.bitdepth) as i32;

    // skip detection
    let sctx = if p.fsc && !chroma && p.seq_fsc {
        9
    } else {
        get_skip_ctx(t_dim, p.bs, a, l, p.plane, p.u_has_cf, p.ss_hor, p.ss_ver) as usize
    };
    let all_skip = if p.plane == 2 {
        msac.decode_bool_adapt(coef.skip_v(sctx))
    } else {
        let i = if !p.intra || p.fsc { 1 } else { 0 };

        msac.decode_bool_adapt(coef.skip(i, t_dim.ctx as usize, sctx))
    };

    if all_skip != 0 {
        *res_ctx = 0x40;
        *txtp = if !chroma && p.fsc {
            txtp::IDTX as u16
        } else {
            (p.lossless as u16) * txtp::WHT_WHT as u16
        };
        return -1;
    }

    // EOB bin decoding
    let slw = imin(t_dim.lw as i32, 3) as usize;
    let slh = imin(t_dim.lh as i32, 3) as usize;
    let tx2dszctx = slw + slh;
    let eob_ctx = if chroma { 2 } else { (!p.intra) as usize };

    let mut eob: i32 = match tx2dszctx {
        0 => msac.decode_symbol_adapt(coef.eob_bin_16(eob_ctx), 4) as i32,
        1 => msac.decode_symbol_adapt(coef.eob_bin_32(eob_ctx), 5) as i32,
        2 => msac.decode_symbol_adapt(coef.eob_bin_64(eob_ctx), 6) as i32,
        3 => msac.decode_symbol_adapt(coef.eob_bin_128(eob_ctx), 7) as i32,
        4 => {
            let mut e = msac.decode_symbol_adapt(coef.eob_bin_256(eob_ctx), 7) as i32;
            if e == 7 {
                e += msac.decode_bools_bypass(1) as i32;
            }
            e
        }
        5 => {
            let mut e = msac.decode_symbol_adapt(coef.eob_bin_512(eob_ctx), 7) as i32;
            if e == 7 {
                e += msac.decode_bools_bypass(2) as i32;
                if e == 10 {
                    return i32::MIN;
                }
            }
            e
        }
        _ => {
            let mut e = msac.decode_symbol_adapt(coef.eob_bin_1024(eob_ctx), 7) as i32;
            if e == 7 {
                e += msac.decode_bools_bypass(2) as i32;
            }
            e
        }
    };

    if eob > 1 {
        let eob_hi_bit = msac.decode_bool_adapt(coef.eob_hi_bit()) as i32;
        let eob_bin = eob - 2;
        eob = eob_hi_bit | 2;
        if eob_bin != 0 {
            eob = (eob << eob_bin) | msac.decode_bools_bypass(eob_bin as u32) as i32;
        }
    }

    // transform type selection
    static TXTP_LONG_TBL: [[[u8; 4]; 2]; 2] = [
        [
            [txtp::V_DCT, txtp::V_ADST, txtp::V_FLIPADST, txtp::IDTX],
            [txtp::H_DCT, txtp::H_ADST, txtp::H_FLIPADST, txtp::IDTX],
        ],
        [
            [
                txtp::DCT_DCT,
                txtp::ADST_DCT,
                txtp::FLIPADST_DCT,
                txtp::H_DCT,
            ],
            [
                txtp::DCT_DCT,
                txtp::DCT_ADST,
                txtp::DCT_FLIPADST,
                txtp::V_DCT,
            ],
        ],
    ];

    if p.lossless {
        if chroma {
            if p.intra {
                let y_fsc = if !p.sdp_active {
                    p.fsc
                } else {
                    let idx = (p.cby & 15) as usize * 16 + (p.cbx & 15) as usize;
                    p.luma_fsc_map[idx] != 0
                };
                *txtp = if y_fsc {
                    txtp::IDTX as u16
                } else {
                    txtp::WHT_WHT as u16
                };
            } else {
                *txtp &= 0xe7; // IDTX_INV -> IDTX
            }
        } else if p.intra {
            *txtp = if p.fsc {
                txtp::IDTX as u16
            } else {
                txtp::WHT_WHT as u16
            };
        } else if t_dim.max == 0 {
            *txtp = if msac.decode_bool_adapt(mode.txtp_lossless()) != 0 {
                txtp::IDTX as u16
            } else {
                txtp::WHT_WHT as u16
            };
        } else {
            *txtp = txtp::IDTX as u16;
        }
    } else if chroma {
        if p.chroma_dctonly {
            *txtp = txtp::DCT_DCT as u16;
        } else {
            if p.intra {
                *txtp = TXTP_FROM_UVMODE[p.uv_mode] as u16;
            }
            let t = *txtp as u8;
            if (t_dim.w >= 8 && t & 0x02 != 0)
                || (t_dim.h >= 8 && t & 0x40 != 0)
                || (p.tx == 2 /* TX_16X16 */
                && ((t & 0x47 == 0x41) || (t & 0xe2 == 0x22)))
            {
                *txtp = txtp::DCT_DCT as u16;
            } else if t == txtp::IDTX_INV {
                *txtp = txtp::IDTX as u16;
            }
        }
    } else if p.intra {
        if t_dim.sub == 3 {
            *txtp = txtp::DCT_DCT as u16;
        } else if p.fsc {
            *txtp = txtp::IDTX as u16;
        } else if eob == 0 || p.tx == 3 {
            *txtp = txtp::DCT_DCT as u16;
        } else if t_dim.max >= 3 {
            let long_dct = t_dim.max == 4 || msac.decode_bool_adapt(mode.txtp_long32_dct(0)) != 0;
            let short_idx =
                msac.decode_symbol_adapt(mode.txtp_intra_short_1d(t_dim.min as usize), 3) as usize;
            let wh = (t_dim.w < t_dim.h) as usize;
            *txtp = TXTP_LONG_TBL[long_dct as usize][wh][short_idx] as u16;
        } else if p.reduced_txtp_set == 2 {
            *txtp = txtp::DCT_DCT as u16;
        } else {
            let sz_ctx = ((t_dim.lw + t_dim.lh) >> 1) as usize;
            let tx_idx = if p.reduced_txtp_set != 0 {
                msac.decode_bool_adapt(mode.txtp_ext_reduced(t_dim.min as usize)) as usize
            } else {
                msac.decode_symbol_adapt(mode.txtp_ext(t_dim.min as usize), 6) as usize
            };
            static MD_IDX2TYPE: [[[u8; 7]; 13]; 3] = [
                [
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::V_DCT,
                        txtp::V_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_DCT,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::V_ADST,
                        txtp::V_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_DCT,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::V_DCT,
                        txtp::V_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::V_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        txtp::H_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::DCT_ADST,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::V_ADST,
                        txtp::H_ADST,
                    ],
                ],
                [
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::FLIPADST_ADST,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::H_ADST,
                    ],
                ],
                [
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::FLIPADST_ADST,
                        txtp::ADST_FLIPADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::FLIPADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::DCT_DCT,
                        txtp::ADST_ADST,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::V_ADST,
                    ],
                ],
            ];
            *txtp = MD_IDX2TYPE[sz_ctx][p.y_mode][tx_idx] as u16;
        }
    } else {
        // inter
        if t_dim.sub == 3 {
            *txtp = txtp::DCT_DCT as u16;
        } else {
            let y = eob >> (2 + slw as i32);
            let x = eob & ((4 << slw) - 1);
            let xy = x + y;
            let ww = imin(8, t_dim.w as i32);
            let hh = imin(8, t_dim.h as i32);
            let ctx = if xy < 2 {
                1usize
            } else if xy > 4 * (ww + hh) - 4 {
                2
            } else {
                0
            };
            if p.tx == 3 {
                *txtp = if msac.decode_bool_adapt(mode.txtp_inter_dct_idtx(ctx, 3)) != 0 {
                    txtp::DCT_DCT as u16
                } else {
                    txtp::IDTX as u16
                };
            } else if t_dim.max >= 3 {
                let long_dct =
                    t_dim.max == 4 || msac.decode_bool_adapt(mode.txtp_long32_dct(1)) != 0;
                let short_idx = msac
                    .decode_symbol_adapt(mode.txtp_inter_short_1d(ctx, t_dim.min as usize), 3)
                    as usize;
                let wh = (t_dim.w < t_dim.h) as usize;
                *txtp = TXTP_LONG_TBL[long_dct as usize][wh][short_idx] as u16;
            } else if p.reduced_txtp_set == 1 || p.reduced_txtp_set == 2 {
                *txtp = if msac.decode_bool_adapt(mode.txtp_inter_dct_idtx(ctx, t_dim.min as usize))
                    != 0
                {
                    txtp::DCT_DCT as u16
                } else {
                    txtp::IDTX as u16
                };
            } else if p.reduced_txtp_set == 3 {
                let tx_idx = msac
                    .decode_symbol_adapt(mode.txtp_inter_dct_idtx_iddct(ctx, t_dim.min as usize), 3)
                    as usize;
                static TXTP_DCT_IDTX_IDDCT: [u8; 4] =
                    [txtp::DCT_DCT, txtp::V_DCT, txtp::H_DCT, txtp::IDTX];
                *txtp = TXTP_DCT_IDTX_IDDCT[tx_idx] as u16;
            } else {
                let setidx = (p.tx == 2) as usize;
                let set =
                    msac.decode_bool_adapt(mode.txtp_inter_tx_set(setidx, ctx, t_dim.min as usize))
                        as usize;
                let t = if set == 0 {
                    msac.decode_symbol_adapt(mode.txtp_inter_set0(setidx, ctx), 7) as usize
                } else if setidx != 0 {
                    msac.decode_symbol_adapt(mode.txtp_inter_set2(ctx), 3) as usize + 8
                } else {
                    msac.decode_symbol_adapt(mode.txtp_inter_set1(ctx), 7) as usize + 8
                };
                static TXTP_INV_TBL: [[u8; 16]; 2] = [
                    [
                        txtp::IDTX,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::V_ADST,
                        txtp::H_ADST,
                        txtp::V_FLIPADST,
                        txtp::H_FLIPADST,
                        txtp::DCT_DCT,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                    ],
                    [
                        txtp::IDTX,
                        txtp::V_DCT,
                        txtp::H_DCT,
                        txtp::DCT_DCT,
                        txtp::ADST_DCT,
                        txtp::DCT_ADST,
                        txtp::FLIPADST_DCT,
                        txtp::DCT_FLIPADST,
                        txtp::ADST_ADST,
                        txtp::FLIPADST_FLIPADST,
                        txtp::ADST_FLIPADST,
                        txtp::FLIPADST_ADST,
                        0,
                        0,
                        0,
                        0,
                    ],
                ];
                *txtp = TXTP_INV_TBL[setidx][t] as u16;
            }
        }
    }

    let tx_class = txtp::class(*txtp as u8);

    // secondary transform (IST)
    let mut stx_type: u32 = 0;
    if p.seq_ist[(!p.intra) as usize] && !chroma {
        if p.intra {
            if eob >= 1
                && p.y_mode != IntraPredMode::PaethPred as usize
                && (*txtp as u8 == txtp::DCT_DCT || *txtp as u8 == txtp::ADST_ADST)
            {
                let lim = if p.tx == 1 && *txtp as u8 == txtp::DCT_DCT {
                    20
                } else if t_dim.min >= 1 {
                    if *txtp as u8 == txtp::DCT_DCT { 32 } else { 20 }
                } else {
                    8
                };
                stx_type = (eob < lim) as u32;
            }
        } else {
            stx_type =
                (t_dim.min >= 2 && *txtp as u8 == txtp::DCT_DCT && (3..32).contains(&eob)) as u32;
        }
        if stx_type != 0 {
            stx_type =
                msac.decode_symbol_adapt(mode.stx((!p.intra) as usize, t_dim.min as usize), 3);
            if stx_type != 0 && p.intra {
                let mut stx_set: u32;
                if t_dim.min >= 1 && *txtp as u8 == txtp::ADST_ADST {
                    static INV_MAP_ADST: [[u8; 4]; 12] = [
                        [3, 1, 0, 2],
                        [1, 3, 0, 2],
                        [1, 3, 0, 2],
                        [1, 3, 0, 2],
                        [0, 2, 3, 1],
                        [2, 1, 0, 3],
                        [2, 1, 0, 3],
                        [1, 0, 3, 2],
                        [1, 0, 3, 2],
                        [3, 1, 0, 2],
                        [1, 3, 0, 2],
                        [1, 3, 0, 2],
                    ];
                    let s = msac.decode_symbol_adapt(mode.stx_set_adst(), 3) as usize;
                    stx_set = INV_MAP_ADST[p.y_mode][s] as u32;
                } else {
                    static INV_MAP: [[u8; 7]; 12] = [
                        [6, 1, 0, 5, 4, 3, 2],
                        [1, 6, 0, 4, 2, 5, 3],
                        [1, 6, 0, 4, 2, 5, 3],
                        [2, 6, 0, 5, 1, 4, 3],
                        [3, 4, 6, 1, 0, 2, 5],
                        [4, 1, 3, 6, 0, 5, 2],
                        [4, 1, 3, 6, 0, 5, 2],
                        [5, 0, 6, 2, 1, 4, 3],
                        [5, 0, 6, 2, 1, 4, 3],
                        [6, 1, 0, 5, 4, 3, 2],
                        [1, 6, 0, 4, 2, 5, 3],
                        [1, 6, 0, 4, 2, 5, 3],
                    ];
                    let s = msac.decode_symbol_adapt(mode.stx_set(), 6) as usize;
                    stx_set = INV_MAP[p.y_mode][s] as u32;
                }
                stx_set += 7 * (*txtp as u8 == txtp::ADST_ADST) as u32;
                *txtp |= (stx_set << 10) as u16;
            }
            *txtp |= (stx_type << 8) as u16;
        }
    } else if p.seq_cctx
        && p.plane == 1
        && eob >= p.intra as i32
        && !p.lossless
        && (p.layout == PixelLayout::I420 || t_dim.max < 8)
    {
        let cctx = msac.decode_symbol_adapt(mode.cctx(), 6);
        *txtp |= (cctx << 8) as u16;
    }

    // base tokens
    let mut cul_level: u32 = 0;
    let mut dc_tok: i32;
    let tcq_en = p.tcq_enabled && !chroma && tx_class == 0 && !p.lossless;
    let mut hr_avg: i32 = 0;
    let mut tcq_state: i32 = if tcq_en { -0x80000000i32 } else { 0 };
    let has_qm = p.qm.is_some() && (*txtp as u8) < txtp::IDTX;
    let mut dq_shift = tcq_en as i32 + 3 + imax(0, t_dim.ctx as i32 - 2);
    let mut dc_sign_level: u32 = 1 << 6;

    let scan = SCANS[p.tx];

    // IDTX/FSC path
    if p.seq_fsc && (!p.intra || p.fsc) && *txtp as u8 == txtp::IDTX && !chroma {
        *txtp = txtp::IDTX_INV as u16;
        let stride = 1 + (4 << slh);
        // Worst-case stride*((4<<slw)+1) is 33*33 = 1089; use a fixed stack array
        // to avoid a per-block heap allocation. Unused tail stays zeroed, matching
        // the previous fully-zeroed Vec semantics required by the neighbour reads.
        let mut levels = [0i8; 1089];
        let sz_ctx = imin(t_dim.ctx as i32, 2) as usize;
        let sz = (16 << tx2dszctx) - 1;
        let bob = sz - eob;
        let ctx = ((bob > 2 << tx2dszctx) as usize) + ((bob > 4 << tx2dszctx) as usize);
        let mut tok = 1 + msac.decode_symbol_adapt(coef.bob_base_y_tok(sz_ctx, ctx), 2) as i32;
        if tok == 3 {
            tok += msac.decode_symbol_adapt(coef.br_y_tok_idtx(sz_ctx, 0), 3) as i32;
        }
        let shift = slh + 2;
        let mask = (4 << slh) - 1;
        let rc = scan[bob as usize] as usize;
        let x = rc >> shift;
        let y = rc & mask;
        cf[rc] = tok;
        levels[(1 + x) * stride + (y + 1)] = tok as i8;

        for i in (bob + 1)..=sz {
            let rc = scan[i as usize] as usize;
            let x = rc >> shift;
            let y = rc & mask;
            let off = (1 + x) * stride + (1 + y);
            let mut hr_ctx = 0u32;
            let ctx = get_lo_ctx_idtx(&levels, off, &mut hr_ctx, stride);
            let mut tok =
                msac.decode_symbol_adapt(coef.base_y_tok_idtx(sz_ctx, ctx as usize), 3) as i32;
            if tok == 3 {
                tok +=
                    msac.decode_symbol_adapt(coef.br_y_tok_idtx(sz_ctx, hr_ctx as usize), 3) as i32;
            }
            cf[rc] = tok;
            levels[off] = tok as i8;
        }

        let dq = p.dq_tbl[1];
        dq_shift -= tcq_en as i32;
        for i in bob..=sz {
            let rc = scan[i as usize] as usize;
            let tok_val = cf[rc];
            if tok_val == 0 {
                continue;
            }
            let x = rc >> shift;
            let y = rc & mask;
            let off = (1 + x) * stride + (1 + y);
            let ctx = get_sign_ctx_idtx(&levels, off, stride);
            let sign = msac.decode_bool_adapt(coef.sign_idtx(sz_ctx, ctx as usize));
            if i == 0 {
                dc_sign_level = ((sign as i32 - 1) & (2 << 6)) as u32;
            }
            levels[off] = 1 - 2 * sign as i8;

            let mut tok = tok_val;
            let val: i32;
            if tok >= 6 {
                let hr = decode_hr(msac, hr_avg);
                tok += hr;
                hr_avg = (hr_avg + hr) >> 1;
                tok &= 0xfffff;
                val = imin(
                    ((((tok as u32).wrapping_mul(dq)) & 0xffffff).wrapping_add(4) >> dq_shift)
                        as i32,
                    cf_max + sign as i32,
                );
            } else {
                val = ((tok as u32).wrapping_mul(dq).wrapping_add(4) >> dq_shift) as i32;
            }
            cul_level += tok as u32;
            cf[rc] = if sign != 0 { -val } else { val };
        }

        *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
        return eob;
    }

    if eob != 0 {
        // Stack-allocated scratch buffer. 1089 == 33*33 is the worst-case size
        // (stride 33 * 33 rows for the largest transform). Allocating this on the
        // heap per transform block was a significant cost in the decode hot path;
        // a fixed stack array removes the malloc/free and the Vec indirection on
        // every `levels[...]` access.
        let mut levels = [0i8; 1089];
        let is_stx = stx_type != 0 && tx_class == 0;

        macro_rules! decode_coefs_class {
            ($tx_cl:expr, $stride:expr, $shift:expr, $shift2:expr, $mask:expr, $hi_to_low:expr, $xy_expr:ident) => {{
                let hi_to_low_tx: i32 = $hi_to_low;
                let stride: usize = $stride;
                let shift: usize = $shift;
                let shift2: usize = $shift2;
                let mask: usize = $mask;

                // eob token
                let (mut lim, mut tok): (i32, i32);
                let (mut hi_base, mut hi_stride): (usize, usize);
                let (mut lo_base, mut lo_stride, mut lo_nsym): (usize, usize, usize);
                let mut hi_cdf_valid: bool = true;

                let ctx_init = 1 + (eob > 2 << tx2dszctx) as u32 + (eob > 4 << tx2dszctx) as u32;
                if eob >= hi_to_low_tx {
                    lim = 3;
                    if !chroma {
                        tok = 1 + msac.decode_symbol_adapt(
                            coef.eob_base_y_tok_hf(t_dim.ctx as usize, ctx_init as usize),
                            2,
                        ) as i32;
                        hi_base = 1252;
                        hi_stride = 4;
                        lo_base = 452 + (t_dim.ctx as usize) * 160;
                        lo_stride = 4;
                        lo_nsym = 3;
                    } else {
                        tok = 1 + msac
                            .decode_symbol_adapt(coef.eob_base_uv_tok_hf(ctx_init as usize), 2)
                            as i32;
                        hi_base = 4508;
                        hi_stride = 4;
                        lo_base = 4460;
                        lo_stride = 4;
                        lo_nsym = 3;
                    }
                    hi_cdf_valid = true;
                } else {
                    lim = 5;
                    if !chroma {
                        tok = 1 + msac.decode_symbol_adapt(
                            coef.eob_base_y_tok_lf(t_dim.ctx as usize, ctx_init as usize),
                            4,
                        ) as i32;
                        hi_base = 4080;
                        hi_stride = 4;
                        lo_base = 1440 + (t_dim.ctx as usize) * 528;
                        lo_stride = 8;
                        lo_nsym = 5;
                    } else {
                        tok = 1 + msac
                            .decode_symbol_adapt(coef.eob_base_uv_tok_lf(ctx_init as usize), 4)
                            as i32;
                        hi_base = 0;
                        hi_stride = 0;
                        lo_base = 4560;
                        lo_stride = 8;
                        lo_nsym = 5;
                        hi_cdf_valid = false;
                    }
                    if chroma {
                        hi_cdf_valid = false;
                    }
                }

                let (mut rc, mut x, mut y): (usize, usize, usize);
                if $tx_cl == 0 {
                    rc = scan[eob as usize] as usize;
                    x = rc >> shift;
                    y = rc & mask;
                } else if $tx_cl == 1 {
                    x = eob as usize & mask;
                    y = eob as usize >> shift;
                    rc = eob as usize;
                } else {
                    x = eob as usize & mask;
                    y = eob as usize >> shift;
                    rc = (x << shift2) | y;
                }
                if tok == lim && hi_cdf_valid {
                    let hi_idx = if lim == 5 { 7 } else { 0 };
                    let o = hi_base + hi_idx * hi_stride;
                    tok += msac.decode_symbol_adapt(&mut coef.data[o..o + 4], 3) as i32;
                }
                tcq_state = tcq_next_state(tcq_state, tok);
                cf[if is_stx { eob as usize } else { rc }] = tok;
                if $tx_cl == 0 {
                    levels[rc] = tok as i8;
                } else {
                    levels[x * stride + y] = tok as i8;
                }

                // ac tokens (eob-1 down to 1)
                let mut i = eob - 1;
                loop {
                    if i == hi_to_low_tx - 1 {
                        lim = 5;
                        if !chroma {
                            hi_base = 4080;
                            hi_stride = 4;
                            lo_base = 1440 + (t_dim.ctx as usize) * 528;
                            lo_stride = 8;
                            lo_nsym = 5;
                            hi_cdf_valid = true;
                        } else {
                            hi_base = 0;
                            hi_stride = 0;
                            lo_base = 4560;
                            lo_stride = 8;
                            lo_nsym = 5;
                            hi_cdf_valid = false;
                        }
                    }
                    if i == 0 {
                        break;
                    }
                    if $tx_cl == 0 {
                        rc = scan[i as usize] as usize;
                        x = rc >> shift;
                        y = rc & mask;
                    } else if $tx_cl == 1 {
                        x = i as usize & mask;
                        y = i as usize >> shift;
                        rc = i as usize;
                    } else {
                        x = i as usize & mask;
                        y = i as usize >> shift;
                        rc = (x << shift2) | y;
                    }
                    let off = if $tx_cl == 0 { rc } else { x * stride + y };
                    let mut hr_ctx = 0u32;
                    let xy_val: u32 = if $tx_cl == 0 {
                        (x + y) as u32
                    } else {
                        y as u32
                    };
                    let ctx =
                        get_lo_ctx(&levels, off, $tx_cl, &mut hr_ctx, xy_val, p.plane, stride);
                    let tcq_bit = ((tcq_state & 2) >> 1) as u32;
                    let lo_cdf_idx = (ctx * (2 - chroma as u32) + tcq_bit) as usize;
                    let o = lo_base + lo_cdf_idx * lo_stride;
                    let mut tok =
                        msac.decode_symbol_adapt(&mut coef.data[o..o + lo_stride], lo_nsym) as i32;
                    if tok == lim && hi_cdf_valid {
                        let o2 = hi_base + hr_ctx as usize * hi_stride;
                        tok += msac.decode_symbol_adapt(&mut coef.data[o2..o2 + 4], 3) as i32;
                    }
                    tcq_state = tcq_next_state(tcq_state, tok);
                    levels[off] = tok as i8;
                    cf[if is_stx { i as usize } else { rc }] = tok;
                    i -= 1;
                }

                // dc token
                let mut hr_ctx = 0u32;
                let ctx = get_lo_ctx(&levels, 0, $tx_cl, &mut hr_ctx, 0, p.plane, stride);
                let tcq_bit = ((tcq_state & 2) >> 1) as u32;
                let lo_cdf_idx = (ctx * (2 - chroma as u32) + tcq_bit) as usize;
                let o = lo_base + lo_cdf_idx * lo_stride;
                dc_tok = msac.decode_symbol_adapt(&mut coef.data[o..o + lo_stride], lo_nsym) as i32;
                if dc_tok == lim && hi_cdf_valid {
                    let o2 = hi_base + hr_ctx as usize * hi_stride;
                    dc_tok += msac.decode_symbol_adapt(&mut coef.data[o2..o2 + 4], 3) as i32;
                }

                // sign & dequant for AC
                tcq_state = if tcq_en { -0x80000000i32 } else { 0 };
                let ac_dq = p.dq_tbl[1];
                for i in (1..=eob).rev() {
                    if $tx_cl == 0 {
                        rc = if is_stx {
                            i as usize
                        } else {
                            scan[i as usize] as usize
                        };
                    } else if $tx_cl == 1 {
                        y = i as usize >> shift;
                        rc = i as usize;
                    } else {
                        x = i as usize & mask;
                        y = i as usize >> shift;
                        rc = (x << shift2) | y;
                    }
                    let tok_val = cf[rc];
                    if tok_val == 0 {
                        tcq_state = tcq_next_state(tcq_state, 0);
                        continue;
                    }
                    let sign: u32;
                    if $tx_cl == 0 || y > 0 || chroma {
                        sign = msac.decode_bool_bypass();
                    } else {
                        sign = msac.decode_bool_adapt(coef.dc_sign(chroma as usize, 0, 0));
                    }
                    let tcq_bit = ((tcq_state & 2) >> 1) as i32;
                    tcq_state = tcq_next_state(tcq_state, tok_val);
                    let max_br = if i < hi_to_low_tx {
                        if chroma { 5 } else { 8 }
                    } else {
                        6
                    };
                    let mut tok = tok_val;
                    let ac_val: i32;
                    if tok >= max_br - tcq_en as i32 {
                        let hr = decode_hr(msac, hr_avg);
                        tok += hr << tcq_en as i32;
                        hr_avg = (hr_avg + hr) >> 1;
                        tok &= 0xfffff;
                        let v = (tok << tcq_en as i32) - tcq_bit;
                        ac_val = imin(
                            ((((v as u32).wrapping_mul(ac_dq)) & 0xffffff).wrapping_add(4)
                                >> dq_shift) as i32,
                            cf_max + sign as i32,
                        );
                    } else {
                        let v = (tok << tcq_en as i32) - tcq_bit;
                        ac_val =
                            (((v as u32).wrapping_mul(ac_dq)).wrapping_add(4) >> dq_shift) as i32;
                    }
                    cul_level += tok as u32;
                    cf[rc] = if sign != 0 { -ac_val } else { ac_val };
                }
            }};
        }

        // reached here). The class!() macro arg mirrors that scan orientation.
        match tx_class {
            0 => {
                let stride = (4 << slh) as usize;
                let shift = slh + 2;
                let mask = (4 << slh) - 1;
                // `levels` is already fully zeroed at allocation; no re-fill needed.
                let hi_to_low = if chroma { 1i32 } else { 10 };
                decode_coefs_class!(0, stride, shift, 0, mask, hi_to_low, xy_2d);
            }
            2 => {
                let stride = 32usize;
                let shift = slh + 2;
                let mask = (4 << slh) - 1;
                let hi_to_low = (8 << slh) >> chroma as usize;
                decode_coefs_class!(1, stride, shift, 0, mask, hi_to_low, xy_h);
            }
            3 => {
                let stride = 32usize;
                let shift = slw + 2;
                let shift2 = slh + 2;
                let mask = (4 << slw) - 1;
                let hi_to_low = (8 << slw) >> chroma as usize;
                decode_coefs_class!(2, stride, shift, shift2, mask, hi_to_low, xy_v);
            }
            _ => unreachable!(),
        }
    } else if chroma {
        dc_tok = 1 + msac.decode_symbol_adapt(coef.eob_base_uv_tok_lf(0), 4) as i32;
    } else {
        dc_tok =
            1 + msac.decode_symbol_adapt(coef.eob_base_y_tok_lf(t_dim.ctx as usize, 0), 4) as i32;
        if dc_tok == 5 {
            let hi_idx = if tx_class == 0 { 0 } else { 7 };
            dc_tok += msac.decode_symbol_adapt(coef.br_y_tok_lf(hi_idx), 3) as i32;
        }
    }

    if dc_tok == 0 {
        *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
        return eob;
    }

    // dc sign & residual
    let dc_sign: u32;
    if chroma {
        dc_sign = msac.decode_bool_bypass();
    } else {
        let dc_sign_ctx = get_dc_sign_ctx(t_dim, a, l) as usize;
        dc_sign = msac.decode_bool_adapt(coef.dc_sign(chroma as usize, 0, dc_sign_ctx));
    }

    let mut dc_dq = p.dq_tbl[0] as i32;
    dc_sign_level = ((dc_sign as i32 - 1) & (2 << 6)) as u32;

    if has_qm {
        let qm_tbl = p.qm.unwrap();
        dc_dq = (dc_dq * qm_tbl[0] as i32 + 16) >> 5;
        if dc_tok == 15 {
            dc_tok = 0;
            dc_tok &= 0xfffff;
            let dq_val = ((dc_dq * dc_tok) & 0xffffff) >> dq_shift;
            let dq_val = imin(dq_val, cf_max + dc_sign as i32);
            cul_level = dc_tok as u32;
            cf[0] = if dc_sign != 0 { -dq_val } else { dq_val };
        } else {
            let dq_val = dc_dq * dc_tok;
            cul_level = dc_tok as u32;
            let dq_val = dq_val >> dq_shift;
            let dq_val = imin(dq_val, cf_max + dc_sign as i32);
            cf[0] = if dc_sign != 0 { -dq_val } else { dq_val };
        }
    } else {
        let max_br = if chroma { 5 } else { 8 };
        let tcq_bit = (tcq_state & 2) >> 1;
        let dc_val: i32;
        if dc_tok >= max_br - tcq_en as i32 {
            let hr = decode_hr(msac, hr_avg);
            dc_tok += hr << tcq_en as i32;
            dc_tok &= 0xfffff;
            let v = (dc_tok << tcq_en as i32) - tcq_bit;
            dc_val = imin(
                ((((v as u32).wrapping_mul(dc_dq as u32)) & 0xffffff).wrapping_add(4) >> dq_shift)
                    as i32,
                cf_max + dc_sign as i32,
            );
        } else {
            let v = (dc_tok << tcq_en as i32) - tcq_bit;
            dc_val = (((v as u32).wrapping_mul(dc_dq as u32)).wrapping_add(4) >> dq_shift) as i32;
        }
        cul_level += dc_tok as u32;
        cf[0] = if dc_sign != 0 { -dc_val } else { dc_val };
    }

    *res_ctx = (cul_level.min(63) | dc_sign_level) as u8;
    eob
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn intrabc_pred<BD: crate::pixel::BitDepth>(
    bd: BD,
    plane: &mut [BD::Pixel],
    stride: usize,
    bw4: i32,
    bh4: i32,
    bx: i32,
    by: i32,
    mvx: i32,
    mvy: i32,
    ss_hor: i32,
    ss_ver: i32,
    right: i32,
    bottom: i32,
) {
    use crate::pixel::Pixel;
    let left = 0i32;
    let top = 0i32;
    let h_mul = 4 >> ss_hor;
    let v_mul = 4 >> ss_ver;
    // put filter (`mx << !ss_hor`). For an unsubsampled plane that is
    // (mvx & 7) << 1; for a subsampled chroma plane it is (mvx & 15).
    let mx_lo = mvx & (15 >> (ss_hor == 0) as i32);
    let my_lo = mvy & (15 >> (ss_ver == 0) as i32);
    let mx = mx_lo << (ss_hor == 0) as i32;
    let my = my_lo << (ss_ver == 0) as i32;
    // Source (reference) position within the current plane, in samples.
    let sx = bx * h_mul + (mvx >> (3 + ss_hor));
    let sy = by * v_mul + (mvy >> (3 + ss_ver));
    // Destination position (block origin), in samples.
    let dpx = bx * h_mul;
    let dpy = by * v_mul;

    let w = (bw4 * h_mul) as usize;
    let h = (bh4 * v_mul) as usize;

    // Gather the source region (plus one extra row/col of subpel context) into a
    // contiguous scratch buffer (as i32), with edge clamping identical to
    // mc.emu_edge. This also lets src and dst share the same plane buffer safely.
    let src_w = w + (mx_lo != 0) as usize; // extra column for the +stride tap
    let src_h = h + (my_lo != 0) as usize; // extra row for the +stride tap
    let src_stride = src_w;
    let mut srcbuf = vec![0i32; src_stride * src_h];
    for ry in 0..src_h {
        let cy = (sy + ry as i32).clamp(top, bottom - 1) as usize;
        for rx in 0..src_w {
            let cx = (sx + rx as i32).clamp(left, right - 1) as usize;
            srcbuf[ry * src_stride + rx] = plane[cy * stride + cx].into();
        }
    }

    let dst_off = (dpy as usize) * stride + dpx as usize;
    let bdmax = bd.bitdepth_max();

    let ib = crate::mc::intermediate_bits(bd);
    if mx != 0 {
        if my != 0 {
            // 2-pass: horizontal into mid (16-bit), then vertical.
            let mut mid = vec![0i32; src_w * (h + 1)];
            for ry in 0..(h + 1) {
                for x in 0..w {
                    let s = ry * src_stride + x;
                    let v = 16 * srcbuf[s] + mx * (srcbuf[s + 1] - srcbuf[s]);
                    mid[ry * w + x] = (v + ((1 << (4 - ib)) >> 1)) >> (4 - ib);
                }
            }
            for ry in 0..h {
                for x in 0..w {
                    let m0 = mid[ry * w + x];
                    let m1 = mid[(ry + 1) * w + x];
                    let v = 16 * m0 + my * (m1 - m0);
                    let px = (v + ((1 << (4 + ib)) >> 1)) >> (4 + ib);
                    plane[dst_off + ry * stride + x] = BD::Pixel::from_i32(iclip(px, 0, bdmax));
                }
            }
        } else {
            let rnd = (1 << ib) >> 1;
            for ry in 0..h {
                for x in 0..w {
                    let s = ry * src_stride + x;
                    let v = 16 * srcbuf[s] + mx * (srcbuf[s + 1] - srcbuf[s]);
                    let px = (v + ((1 << (4 - ib)) >> 1)) >> (4 - ib);
                    plane[dst_off + ry * stride + x] =
                        BD::Pixel::from_i32(iclip((px + rnd) >> ib, 0, bdmax));
                }
            }
        }
    } else if my != 0 {
        for ry in 0..h {
            for x in 0..w {
                let s0 = ry * src_stride + x;
                let s1 = (ry + 1) * src_stride + x;
                let v = 16 * srcbuf[s0] + my * (srcbuf[s1] - srcbuf[s0]);
                let px = (v + ((1 << 4) >> 1)) >> 4;
                plane[dst_off + ry * stride + x] = BD::Pixel::from_i32(iclip(px, 0, bdmax));
            }
        }
    } else {
        // integer copy
        for ry in 0..h {
            for x in 0..w {
                plane[dst_off + ry * stride + x] = BD::Pixel::from_i32(srcbuf[ry * src_stride + x]);
            }
        }
    }
}
