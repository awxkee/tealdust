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
use crate::itx_1d::{TX1D_FNS, TX1D_FNS_X8, inv_wht_wht_4x4, residual_add, residual_add_strided};
use crate::pixel::BitDepth;
use crate::scan::LAST_EOB_PER_COL;
use crate::tables::{TX_SHIFT, TXFM_DIMENSIONS};

const WHT_WHT: u32 = 6 | (6 << 5);

const ITX_TMP_STRIDE: usize = 32;
const ITX_TMP_PIXELS: usize = ITX_TMP_STRIDE * ITX_TMP_STRIDE;

#[derive(Clone)]
struct Txfm2d {
    buf: [i32; ITX_TMP_PIXELS],
}

impl Txfm2d {
    #[inline(always)]
    fn new() -> Self {
        Self {
            buf: [0; ITX_TMP_PIXELS],
        }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [i32] {
        &mut self.buf
    }

    #[inline(always)]
    fn row(&self, y: usize) -> &[i32; ITX_TMP_STRIDE] {
        self.buf[y * ITX_TMP_STRIDE..(y + 1) * ITX_TMP_STRIDE]
            .try_into()
            .unwrap()
    }

    #[inline(always)]
    fn row_mut(&mut self, y: usize) -> &mut [i32; ITX_TMP_STRIDE] {
        (&mut self.buf[y * ITX_TMP_STRIDE..(y + 1) * ITX_TMP_STRIDE])
            .try_into()
            .unwrap()
    }

    #[inline(always)]
    fn clear_tail_rows(&mut self, start: usize, end: usize, width: usize) {
        for y in start..end {
            self.row_mut(y)[..width].fill(0);
        }
    }

    #[inline(always)]
    fn compact<const N: usize>(&self, width: usize, height: usize) -> [i32; N] {
        debug_assert!(width * height <= N);
        let mut out = [0i32; N];
        for y in 0..height {
            let dst = y * width;
            out[dst..dst + width].copy_from_slice(&self.row(y)[..width]);
        }
        out
    }
}

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

    let mut tmp = Txfm2d::new();
    let mut row = 0usize;
    let tx_class = (txtp >> 3) & 0x3;

    if tx_class == 0 {
        let off = LAST_EOB_PER_COL.offset[tx] as usize;
        let last_eob = &LAST_EOB_PER_COL.table[off..];
        let mut ei = 0usize;
        loop {
            let tmp_row = tmp.row_mut(row);
            for (x, dst) in tmp_row[..sw].iter_mut().enumerate() {
                let v = coeff[row + x * sh];
                *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            }
            first_1d_fn(tmp_row, 1);
            row += 1;
            if row & 3 == 0 {
                if eob > last_eob[ei] as i32 {
                    ei += 1;
                } else {
                    break;
                }
            }
        }
    } else {
        let last_nz_row = if tx_class == 2 {
            imin(sh as i32 - 1, eob) as usize
        } else if tx_class == 3 {
            (eob as usize) >> (t_dim.lw as usize + 2)
        } else {
            sh - 1
        };
        loop {
            let tmp_row = tmp.row_mut(row);
            for (x, dst) in tmp_row[..sw].iter_mut().enumerate() {
                let v = coeff[row + x * sh];
                *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            }
            first_1d_fn(tmp_row, 1);
            row += 1;
            if row > last_nz_row {
                break;
            }
        }
    }

    if row < sh {
        tmp.clear_tail_rows(row, sh, sw);
    }
    coeff[..sw * sh].fill(0);

    let shift0 = tx_sh[0] as i32;
    let rnd0 = (1 << shift0) >> 1;
    for y in 0..sh {
        crate::simd::row_clip(tmp.row_mut(y), sw, rnd0, shift0, row_clip_min, row_clip_max);
    }

    let second_1d_fn_x8 = TX1D_FNS_X8[t_dim.lh as usize][((txtp >> 5) & 7) as usize];
    let mut x = 0;
    if let Some(f8) = second_1d_fn_x8 {
        while x + 8 <= sw {
            f8(tmp.as_mut_slice(), x, ITX_TMP_STRIDE);
            x += 8;
        }
    }
    while x < sw {
        second_1d_fn(&mut tmp.as_mut_slice()[x..], ITX_TMP_STRIDE);
        x += 1;
    }

    let shift1 = tx_sh[1] as i32;
    let rnd1 = (1 << shift1) >> 1;

    if w > sw {
        if h > sh {
            for (ty, y) in (0..h).step_by(2).enumerate() {
                let tmp_row = tmp.row(ty);
                for (tx, x) in (0..w).step_by(2).enumerate() {
                    let cf = (tmp_row[tx] + rnd1) >> shift1;
                    let d0 = dst_off + y * stride + x;
                    let d1 = dst_off + (y + 1) * stride + x;
                    dst[d0] = bd.pixel_clip(dst[d0].into() + cf);
                    dst[d0 + 1] = bd.pixel_clip(dst[d0 + 1].into() + cf);
                    dst[d1] = bd.pixel_clip(dst[d1].into() + cf);
                    dst[d1 + 1] = bd.pixel_clip(dst[d1 + 1].into() + cf);
                }
            }
        } else {
            for y in 0..h {
                let tmp_row = tmp.row(y);
                for (tx, x) in (0..w).step_by(2).enumerate() {
                    let cf = (tmp_row[tx] + rnd1) >> shift1;
                    let d = dst_off + y * stride + x;
                    dst[d] = bd.pixel_clip(dst[d].into() + cf);
                    dst[d + 1] = bd.pixel_clip(dst[d + 1].into() + cf);
                }
            }
        }
    } else if h > sh {
        for (ty, y) in (0..h).step_by(2).enumerate() {
            let tmp_row = tmp.row(ty);
            for (x, &v) in tmp_row[..w].iter().enumerate() {
                let cf = (v + rnd1) >> shift1;
                let d0 = dst_off + y * stride + x;
                let d1 = dst_off + (y + 1) * stride + x;
                dst[d0] = bd.pixel_clip(dst[d0].into() + cf);
                dst[d1] = bd.pixel_clip(dst[d1].into() + cf);
            }
        }
    } else {
        let dpcm_flag = (txtp >> 8) as u8;
        if dpcm_flag == 0 {
            residual_add_strided(
                bd,
                &mut dst[dst_off..],
                stride,
                tmp.as_mut_slice(),
                ITX_TMP_STRIDE,
                w,
                h,
                rnd1,
                shift1,
            );
        } else {
            let compact = tmp.compact::<ITX_TMP_PIXELS>(w, h);
            residual_add(
                bd,
                &mut dst[dst_off..],
                stride,
                &compact,
                w,
                h,
                rnd1,
                shift1,
                dpcm_flag,
            );
        }
    }
}

/// Cross-component transform clip at the coded bit depth (`cctx_c` in
pub fn cctx_bd<BD: BitDepth>(bd: BD, u: &mut [i32], v: &mut [i32], angle: &[i16; 3], sz: usize) {
    use crate::itx_1d::cctx;
    cctx(u, v, angle, sz, bd.bitdepth() as i32);
}
