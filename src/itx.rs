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
use crate::itx_2d::{ITX_TMP_PIXELS, ITX_TMP_STRIDE};
use crate::levels::txtp as txtp_kind;
use crate::pixel::BitDepth;
use crate::scan::LAST_EOB_PER_COL;
use crate::tables::{TX_SHIFT, TXFM_DIMENSIONS};

const WHT_WHT: u32 = 6 | (6 << 5);

thread_local! {
    static ITX_SCRATCH: core::cell::RefCell<[i32; ITX_TMP_PIXELS]> =
        const { core::cell::RefCell::new([0; ITX_TMP_PIXELS]) };
}

/// Run `f` with a `Txfm2d` view over this thread's reusable scratch buffer.
#[inline(always)]
fn with_itx_scratch<R>(f: impl FnOnce(&mut Txfm2d) -> R) -> R {
    ITX_SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let mut tmp = Txfm2d { buf: &mut guard };
        f(&mut tmp)
    })
}

struct Txfm2d<'a> {
    buf: &'a mut [i32; ITX_TMP_PIXELS],
}

impl Txfm2d<'_> {
    #[inline(always)]
    fn as_slice(&self) -> &[i32] {
        &self.buf[..]
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [i32] {
        &mut self.buf[..]
    }

    #[inline(always)]
    fn as_mut_array(&mut self) -> &mut [i32; ITX_TMP_PIXELS] {
        &mut *self.buf
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

#[inline(always)]
fn add_tmp_to_dst<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    tmp: &Txfm2d,
    w: usize,
    h: usize,
    sw: usize,
    sh: usize,
    rnd: i32,
    shift: i32,
    dpcm_flag: u8,
) {
    if w > sw {
        if h > sh {
            for (ty, y) in (0..h).step_by(2).enumerate() {
                let tmp_row = tmp.row(ty);
                for (tx, x) in (0..w).step_by(2).enumerate() {
                    let cf = (tmp_row[tx] + rnd) >> shift;
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
                    let cf = (tmp_row[tx] + rnd) >> shift;
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
                let cf = (v + rnd) >> shift;
                let d0 = dst_off + y * stride + x;
                let d1 = dst_off + (y + 1) * stride + x;
                dst[d0] = bd.pixel_clip(dst[d0].into() + cf);
                dst[d1] = bd.pixel_clip(dst[d1].into() + cf);
            }
        }
    } else if dpcm_flag == 0 {
        residual_add_strided(
            bd,
            &mut dst[dst_off..],
            stride,
            tmp.as_slice(),
            ITX_TMP_STRIDE,
            w,
            h,
            rnd,
            shift,
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
            rnd,
            shift,
            dpcm_flag,
        );
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

    let first_kind = (txtp & 7) as usize;
    let second_kind = ((txtp >> 5) & 7) as usize;
    let first_1d_fn = TX1D_FNS[t_dim.lw as usize][first_kind].unwrap();
    let second_1d_fn = TX1D_FNS[t_dim.lh as usize][second_kind].unwrap();
    let sh = imin(h as i32, 32) as usize;
    let sw = imin(w as i32, 32) as usize;
    // coded depth: row_clip_min = (~bitdepth_max) << 7, row_clip_max = ~min.
    let (row_clip_min, row_clip_max) = if BD::BPC == 8 {
        (i16::MIN as i32, i16::MAX as i32)
    } else {
        let min = ((!bd.bitdepth_max() as u32) << 7) as i32;
        (min, !min)
    };

    let shift0 = tx_sh[0] as i32;
    let shift1 = tx_sh[1] as i32;

    if (txtp & 0xFF) == txtp_kind::DCT_DCT as u32 && (txtp >> 8) == 0 && t_dim.lw == t_dim.lh {
        let handled = with_itx_scratch(|tmp| {
            let mut handled = true;

            match tx {
                0 => {
                    let f = crate::itx_2d::idct_dequant_4x4();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                    );
                }
                1 => {
                    let f = crate::itx_2d::idct_dequant_8x8();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                    );
                }
                2 => {
                    let f = crate::itx_2d::idct_dequant_16x16();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                    );
                }
                3 => {
                    let f = crate::itx_2d::idct_dequant_32x32();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                    );
                }
                4 => {
                    let f = crate::itx_2d::idct_dequant_64x64();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                    );
                }
                _ => handled = false,
            }

            if handled {
                let rnd1 = (1 << shift1) >> 1;
                add_tmp_to_dst(bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0);
            }
            handled
        });
        if handled {
            return;
        }
    }

    if (txtp >> 8) == 0
        && ((txtp >> 3) & 0x3) == 0
        && t_dim.lw == t_dim.lh
        && t_dim.lw <= 2
        && crate::itx_2d::is_dct_adst_kind(first_kind)
        && crate::itx_2d::is_dct_adst_kind(second_kind)
        && (first_kind != crate::itx_2d::TX_KIND_DCT || second_kind != crate::itx_2d::TX_KIND_DCT)
    {
        let handled = with_itx_scratch(|tmp| {
            let mut handled = true;

            match tx {
                0 => {
                    let f = crate::itx_2d::iadst_dequant_4x4();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                        first_kind,
                        second_kind,
                    );
                }
                1 => {
                    let f = crate::itx_2d::iadst_dequant_8x8();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                        first_kind,
                        second_kind,
                    );
                }
                2 => {
                    let f = crate::itx_2d::iadst_dequant_16x16();
                    f(
                        coeff,
                        tmp.as_mut_array(),
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                        first_kind,
                        second_kind,
                    );
                }
                _ => handled = false,
            }

            if handled {
                let rnd1 = (1 << shift1) >> 1;
                add_tmp_to_dst(bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0);
            }
            handled
        });
        if handled {
            return;
        }
    }

    with_itx_scratch(|tmp| {
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

        let rnd1 = (1 << shift1) >> 1;

        add_tmp_to_dst(
            bd,
            dst,
            dst_off,
            stride,
            tmp,
            w,
            h,
            sw,
            sh,
            rnd1,
            shift1,
            (txtp >> 8) as u8,
        );
    });
}

/// Cross-component transform clip at the coded bit depth (`cctx_c` in
pub fn cctx_bd<BD: BitDepth>(bd: BD, u: &mut [i32], v: &mut [i32], angle: &[i16; 3], sz: usize) {
    use crate::itx_1d::cctx;
    cctx(u, v, angle, sz, bd.bitdepth() as i32);
}

#[cfg(test)]
mod scratch_reuse_proof {
    // End-to-end proof that the thread-local scratch reuse cannot leak prior
    // contents into the output: inv_txfm_add must produce identical pixels
    // whether the scratch starts zeroed or poisoned with garbage.
    use super::*;
    use crate::levels::txtp as tt;
    use crate::pixel::BitDepth8;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn coef(&mut self) -> i32 {
            (self.next() as i32 % 2049) - 1024
        }
    }

    fn poison() {
        ITX_SCRATCH.with(|c| c.borrow_mut().fill(0xDEAD_BEEFu32 as i32));
    }
    fn zero() {
        ITX_SCRATCH.with(|c| c.borrow_mut().fill(0));
    }

    fn check(tx: usize, txtp: u32, seed: u64) {
        let mut rng = Rng(seed);
        let stride = 64usize;
        for _ in 0..120 {
            let mut coeff = [0i32; 4096];
            for v in coeff.iter_mut() {
                *v = rng.coef();
            }
            let eob = ((rng.next() % 64) + 1) as i32;
            let base = vec![100u8; stride * 64 + stride];

            poison();
            let (mut d1, mut c1) = (base.clone(), coeff);
            inv_txfm_add(BitDepth8, &mut d1, 0, stride, &mut c1, txtp, eob, tx);

            zero();
            let (mut d2, mut c2) = (base.clone(), coeff);
            inv_txfm_add(BitDepth8, &mut d2, 0, stride, &mut c2, txtp, eob, tx);

            assert_eq!(
                d1, d2,
                "dst mismatch tx={} txtp={:#x} eob={}",
                tx, txtp, eob
            );
        }
    }

    #[test]
    fn dct_square_fastpath() {
        for tx in [0usize, 1, 2, 3] {
            check(tx, tt::DCT_DCT as u32, 0x100 + tx as u64);
        }
    }
    #[test]
    fn adst_square_fastpath() {
        for tx in [0usize, 1, 2] {
            check(tx, tt::ADST_ADST as u32, 0x200 + tx as u64);
            check(tx, tt::ADST_DCT as u32, 0x210 + tx as u64);
            check(tx, tt::DCT_ADST as u32, 0x220 + tx as u64);
        }
    }
    #[test]
    fn generic_path() {
        for tx in [0usize, 1, 2] {
            check(tx, tt::IDTX as u32, 0x300 + tx as u64);
            check(tx, tt::FLIPADST_FLIPADST as u32, 0x310 + tx as u64);
        }
    }
}
