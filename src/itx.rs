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

// Test-only switch: when set, the dedicated non-square DCT_DCT cores are
// bypassed so `inv_txfm_add` exercises the original generic path. Used by the
// differential tests to compare the rectangular cores against the proven
// generic implementation. Always false (and free) in production builds.
#[cfg(test)]
pub(crate) static FORCE_GENERIC_ITX: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
#[inline(always)]
fn force_generic_itx() -> bool {
    FORCE_GENERIC_ITX.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(test))]
#[inline(always)]
fn force_generic_itx() -> bool {
    false
}
use crate::levels::txtp as txtp_kind;
use crate::pixel::BitDepth;
use crate::scan::LAST_EOB_PER_COL;
use crate::tables::{TX_SHIFT, TXFM_DIMENSIONS};

const WHT_WHT: u32 = 6 | (6 << 5);

// Per-thread reusable transform scratch. The dequant cores and the generic
// inv_txfm_add path fully write the used S×S region before reading it (rows
// 0..last from the row pass, rows last..S zero-filled) and never read columns
// S..ITX_TMP_STRIDE, so leftover data from a previous transform can never leak
// into the result. That lets us reuse one persistent, already-initialised
// buffer per thread instead of zeroing a fresh 4 KB buffer on every call (a
// memset the compiler can't elide, since the dequant fn is an indirect call).
// Proven bit-exact by the `tmp_init_proof` tests: arbitrary garbage in the
// buffer yields identical output to a zeroed buffer.
std::thread_local! {
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

    // Non-square DCT_DCT. Dims <= 32 use dedicated rectangular cores. The
    // 64-involving sizes have no real 64-point transform (the decoder maps the
    // 64 dimension to inv_dct32), so each computes identically to its clamped
    // (min(W,32), min(H,32)) shape and reuses that core, with the caller's `tx`
    // (eob table) and `is_rect2` (scaling) selecting the correct behavior.
    if (txtp & 0xFF) == txtp_kind::DCT_DCT as u32
        && (txtp >> 8) == 0
        && t_dim.lw != t_dim.lh
        && !force_generic_itx()
    {
        let handled = with_itx_scratch(|tmp| {
            let mut handled = true;

            match tx {
                5 => {
                    let f = crate::itx_2d::idct_dequant_4x8();
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
                6 => {
                    let f = crate::itx_2d::idct_dequant_8x4();
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
                7 => {
                    let f = crate::itx_2d::idct_dequant_8x16();
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
                8 => {
                    let f = crate::itx_2d::idct_dequant_16x8();
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
                9 => {
                    let f = crate::itx_2d::idct_dequant_16x32();
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
                10 => {
                    let f = crate::itx_2d::idct_dequant_32x16();
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
                13 => {
                    let f = crate::itx_2d::idct_dequant_4x16();
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
                14 => {
                    let f = crate::itx_2d::idct_dequant_16x4();
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
                15 => {
                    let f = crate::itx_2d::idct_dequant_8x32();
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
                16 => {
                    let f = crate::itx_2d::idct_dequant_32x8();
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
                19 => {
                    let f = crate::itx_2d::idct_dequant_4x32();
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
                20 => {
                    let f = crate::itx_2d::idct_dequant_32x4();
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
                11 => {
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
                12 => {
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
                17 => {
                    let f = crate::itx_2d::idct_dequant_16x32();
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
                18 => {
                    let f = crate::itx_2d::idct_dequant_32x16();
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
                21 => {
                    let f = crate::itx_2d::idct_dequant_8x32();
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
                22 => {
                    let f = crate::itx_2d::idct_dequant_32x8();
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
                23 => {
                    let f = crate::itx_2d::idct_dequant_4x32();
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
                24 => {
                    let f = crate::itx_2d::idct_dequant_32x4();
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

    if (txtp >> 8) == 0
        && ((txtp >> 3) & 0x3) == 0
        && t_dim.lw != t_dim.lh
        && t_dim.lw <= 2
        && t_dim.lh <= 2
        && crate::itx_2d::is_dct_adst_kind(first_kind)
        && crate::itx_2d::is_dct_adst_kind(second_kind)
        && (first_kind != crate::itx_2d::TX_KIND_DCT || second_kind != crate::itx_2d::TX_KIND_DCT)
        && !force_generic_itx()
    {
        let handled = with_itx_scratch(|tmp| {
            let mut handled = true;

            match tx {
                5 => {
                    let f = crate::itx_2d::iadst_dequant_4x8();
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
                6 => {
                    let f = crate::itx_2d::iadst_dequant_8x4();
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
                7 => {
                    let f = crate::itx_2d::iadst_dequant_8x16();
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
                8 => {
                    let f = crate::itx_2d::iadst_dequant_16x8();
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
                13 => {
                    let f = crate::itx_2d::iadst_dequant_4x16();
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
                14 => {
                    let f = crate::itx_2d::iadst_dequant_16x4();
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

#[cfg(test)]
mod rect_end_to_end {
    //! End-to-end check: full `inv_txfm_add` output with the dedicated
    //! non-square DCT_DCT cores active, compared bit-for-bit against the same
    //! call forced down the original generic path. This is the independent
    //! oracle — the generic path uses its own coefficient orientation, rect2
    //! scaling, eob handling and clipping — so it catches any systematic error
    //! (e.g. a swapped W/H axis) that a core-vs-core test could not.
    use super::*;
    use crate::pixel::BitDepth8;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    // The force-generic switch is process-global, so serialize these tests.
    static LOCK: Mutex<()> = Mutex::new(());

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn run(tx: usize, w: usize, h: usize, seed: u64, trials: usize) {
        run_tt(tx, w, h, 0, seed, trials);
    }

    fn run_tt(tx: usize, w: usize, h: usize, txtp: u32, seed: u64, trials: usize) {
        let _guard = LOCK.lock().unwrap();
        let mut rng = Rng(seed);
        let stride = w;
        let n = w * h;
        let (sw, sh) = (w.min(32), h.min(32));
        for _ in 0..trials {
            let mut coeff0 = vec![0i32; n + 16];
            for v in coeff0[..sw * sh].iter_mut() {
                *v = rng.range(-(1 << 12), 1 << 12);
            }
            let eob = rng.range(1, (sw * sh) as i32);

            let dst_init: Vec<u8> = (0..stride * h).map(|_| rng.range(0, 256) as u8).collect();

            let mut dst_rect = dst_init.clone();
            let mut c_rect = coeff0.clone();
            FORCE_GENERIC_ITX.store(false, Ordering::Relaxed);
            inv_txfm_add::<BitDepth8>(
                BitDepth8,
                &mut dst_rect,
                0,
                stride,
                &mut c_rect,
                txtp,
                eob,
                tx,
            );

            let mut dst_gen = dst_init.clone();
            let mut c_gen = coeff0.clone();
            FORCE_GENERIC_ITX.store(true, Ordering::Relaxed);
            inv_txfm_add::<BitDepth8>(
                BitDepth8,
                &mut dst_gen,
                0,
                stride,
                &mut c_gen,
                txtp,
                eob,
                tx,
            );
            FORCE_GENERIC_ITX.store(false, Ordering::Relaxed);

            assert_eq!(dst_rect, dst_gen, "tx={} {}x{} eob={}", tx, w, h, eob);
        }
    }

    #[test]
    fn e_4x8() {
        run(5, 4, 8, 0xE_48, 800);
    }
    #[test]
    fn e_8x4() {
        run(6, 8, 4, 0xE_84, 800);
    }
    #[test]
    fn e_8x16() {
        run(7, 8, 16, 0xE_816, 800);
    }
    #[test]
    fn e_16x8() {
        run(8, 16, 8, 0xE_168, 800);
    }
    #[test]
    fn e_16x32() {
        run(9, 16, 32, 0xE_1632, 400);
    }
    #[test]
    fn e_32x16() {
        run(10, 32, 16, 0xE_3216, 400);
    }
    #[test]
    fn e_4x16() {
        run(13, 4, 16, 0xE_416, 800);
    }
    #[test]
    fn e_16x4() {
        run(14, 16, 4, 0xE_164, 800);
    }
    #[test]
    fn e_8x32() {
        run(15, 8, 32, 0xE_832, 800);
    }
    #[test]
    fn e_32x8() {
        run(16, 32, 8, 0xE_328, 800);
    }
    #[test]
    fn e_4x32() {
        run(19, 4, 32, 0xE_432, 800);
    }
    #[test]
    fn e_32x4() {
        run(20, 32, 4, 0xE_324, 800);
    }
    // 64-involving sizes (reuse clamped cores; oracle = generic path).
    #[test]
    fn e_32x64() {
        run(11, 32, 64, 0xE_3264, 400);
    }
    #[test]
    fn e_64x32() {
        run(12, 64, 32, 0xE_6432, 400);
    }
    #[test]
    fn e_16x64() {
        run(17, 16, 64, 0xE_1664, 400);
    }
    #[test]
    fn e_64x16() {
        run(18, 64, 16, 0xE_6416, 400);
    }
    #[test]
    fn e_8x64() {
        run(21, 8, 64, 0xE_864, 800);
    }
    #[test]
    fn e_64x8() {
        run(22, 64, 8, 0xE_648, 800);
    }
    #[test]
    fn e_4x64() {
        run(23, 4, 64, 0xE_464, 800);
    }
    #[test]
    fn e_64x4() {
        run(24, 64, 4, 0xE_644, 800);
    }

    // Non-square ADST / mixed-type, all DCT/ADST/FLIPADST combos (minus
    // DCT_DCT). txtp = first_kind | (second_kind << 5); ADST=2, FLIPADST=3.
    fn adst_combos(tx: usize, w: usize, h: usize, seed: u64) {
        let kinds = [0usize, 2, 3]; // DCT, ADST, FLIPADST
        let mut s = seed;
        for &f in &kinds {
            for &sec in &kinds {
                if f == 0 && sec == 0 {
                    continue; // DCT_DCT handled elsewhere
                }
                let txtp = (f as u32) | ((sec as u32) << 5);
                run_tt(tx, w, h, txtp, s, 250);
                s = s.wrapping_add(0x9E3779B97F4A7C15);
            }
        }
    }

    #[test]
    fn a_4x8() {
        adst_combos(5, 4, 8, 0xAD_48);
    }
    #[test]
    fn a_8x4() {
        adst_combos(6, 8, 4, 0xAD_84);
    }
    #[test]
    fn a_8x16() {
        adst_combos(7, 8, 16, 0xAD_816);
    }
    #[test]
    fn a_16x8() {
        adst_combos(8, 16, 8, 0xAD_168);
    }
    #[test]
    fn a_4x16() {
        adst_combos(13, 4, 16, 0xAD_416);
    }
    #[test]
    fn a_16x4() {
        adst_combos(14, 16, 4, 0xAD_164);
    }
}
