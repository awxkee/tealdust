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
use crate::itx_1d::{TX1D_FNS, TX1D_FNS_X8, inv_wht_wht_4x4, residual_add};
use crate::pixel::BitDepth;
use crate::scan::LAST_EOB_PER_COL;
use crate::tables::{TX_SHIFT, TXFM_DIMENSIONS};

const WHT_WHT: u32 = 6 | (6 << 5);

/// Inverse transform of `coeff` followed by clipped add into `dst`
/// intermediate range and the final pixel clip both scale with `bd`.
pub(crate) fn inv_txfm_add<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [i32],
    txtp: u32,
    eob: i32,
    tx: usize,
) {
    if txtp & 0xFF == WHT_WHT {
        assert!(tx == 0);
        let mut tmp = [0i32; 16];
        inv_wht_wht_4x4(&coeff[..16].try_into().unwrap(), &mut tmp);
        coeff[..16].fill(0);
        let dpcm_flag = (txtp >> 8) as u8;
        residual_add(bd, &mut dst[dst_off..], stride, &tmp, 4, 4, 0, 0, dpcm_flag);
        return;
    }

    let t_dim = &TXFM_DIMENSIONS[tx];
    let tx_sh = &TX_SHIFT[tx];
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let is_rect2 = ((t_dim.lw + t_dim.lh) & 1) != 0;

    if eob + txtp as i32 == 0 {
        let shift_p1 = tx_sh[0] as i32;
        let shift = shift_p1 + tx_sh[1] as i32 - 12;
        let rnd = (1 << (shift - 1)) + shift_p1 - 6;
        let mut dc = coeff[0];
        coeff[0] = 0;
        if is_rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc + rnd) >> shift;
        for y in 0..h {
            let row = dst_off + y * stride;
            if row >= dst.len() {
                break;
            }
            let d = &mut dst[row..];
            let n = w.min(d.len());
            crate::simd::dc_add_row(bd, d, dc, n);
        }
        return;
    }

    let first_1d_fn = TX1D_FNS[t_dim.lw as usize][(txtp & 7) as usize].unwrap();
    let second_1d_fn = TX1D_FNS[t_dim.lh as usize][((txtp >> 5) & 7) as usize].unwrap();
    let sh = imin(h as i32, 32) as usize;
    let sw = imin(w as i32, 32) as usize;
    // coded depth: row_clip_min = (~bitdepth_max) << 7, row_clip_max = ~min.
    let (row_clip_min, row_clip_max) = if BD::BPC == 8 {
        (i16::MIN as i32, i16::MAX as i32)
    } else {
        let min = ((!bd.bitdepth_max() as u32) << 7) as i32;
        (min, !min)
    };

    let mut tmp = [0i32; 32 * 32];
    let mut col = 0usize;
    let tx_class = (txtp >> 3) & 0x3;

    if tx_class == 0 {
        let off = LAST_EOB_PER_COL.offset[tx] as usize;
        let last_eob = &LAST_EOB_PER_COL.table[off..];
        let mut ei = 0usize;
        loop {
            for x in 0..sw {
                let v = coeff[col + x * sh];
                tmp[col * sw + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            }
            first_1d_fn(&mut tmp[col * sw..], 1);
            col += 1;
            if col & 3 == 0 {
                if eob > last_eob[ei] as i32 {
                    ei += 1;
                } else {
                    break;
                }
            }
        }
    } else {
        let last_nz_col = if tx_class == 2 {
            imin(sh as i32 - 1, eob) as usize
        } else if tx_class == 3 {
            (eob as usize) >> (t_dim.lw as usize + 2)
        } else {
            sh - 1
        };
        loop {
            for x in 0..sw {
                let v = coeff[col + x * sh];
                tmp[col * sw + x] = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            }
            first_1d_fn(&mut tmp[col * sw..], 1);
            col += 1;
            if col > last_nz_col {
                break;
            }
        }
    }

    if col < sh {
        tmp[col * sw..sh * sw].fill(0);
    }
    coeff[..sw * sh].fill(0);

    let shift0 = tx_sh[0] as i32;
    let rnd0 = (1 << shift0) >> 1;
    crate::simd::row_clip(&mut tmp, sw * sh, rnd0, shift0, row_clip_min, row_clip_max);

    let second_1d_fn_x8 = TX1D_FNS_X8[t_dim.lh as usize][((txtp >> 5) & 7) as usize];
    let mut x = 0;
    if let Some(f8) = second_1d_fn_x8 {
        while x + 8 <= sw {
            f8(&mut tmp, x, sw);
            x += 8;
        }
    }
    while x < sw {
        second_1d_fn(&mut tmp[x..], sw);
        x += 1;
    }

    let shift1 = tx_sh[1] as i32;
    let rnd1 = (1 << shift1) >> 1;

    if w > sw {
        if h > sh {
            let mut ci = 0;
            for y in (0..h).step_by(2) {
                for x in (0..w).step_by(2) {
                    let cf = (tmp[ci] + rnd1) >> shift1;
                    ci += 1;
                    let d0 = dst_off + y * stride + x;
                    let d1 = dst_off + (y + 1) * stride + x;
                    dst[d0] = bd.pixel_clip(dst[d0].into() + cf);
                    dst[d0 + 1] = bd.pixel_clip(dst[d0 + 1].into() + cf);
                    dst[d1] = bd.pixel_clip(dst[d1].into() + cf);
                    dst[d1 + 1] = bd.pixel_clip(dst[d1 + 1].into() + cf);
                }
            }
        } else {
            let mut ci = 0;
            for y in 0..h {
                for x in (0..w).step_by(2) {
                    let cf = (tmp[ci] + rnd1) >> shift1;
                    ci += 1;
                    let d = dst_off + y * stride + x;
                    dst[d] = bd.pixel_clip(dst[d].into() + cf);
                    dst[d + 1] = bd.pixel_clip(dst[d + 1].into() + cf);
                }
            }
        }
    } else if h > sh {
        let mut ci = 0;
        for y in (0..h).step_by(2) {
            for x in 0..w {
                let cf = (tmp[ci] + rnd1) >> shift1;
                ci += 1;
                let d0 = dst_off + y * stride + x;
                let d1 = dst_off + (y + 1) * stride + x;
                dst[d0] = bd.pixel_clip(dst[d0].into() + cf);
                dst[d1] = bd.pixel_clip(dst[d1].into() + cf);
            }
        }
    } else {
        let dpcm_flag = (txtp >> 8) as u8;
        residual_add(
            bd,
            &mut dst[dst_off..],
            stride,
            &tmp,
            w,
            h,
            rnd1,
            shift1,
            dpcm_flag,
        );
    }
}

/// Cross-component transform clip at the coded bit depth (`cctx_c` in
pub fn cctx_bd<BD: BitDepth>(bd: BD, u: &mut [i32], v: &mut [i32], angle: &[i16; 3], sz: usize) {
    use crate::itx_1d::cctx;
    cctx(u, v, angle, sz, bd.bitdepth() as i32);
}
