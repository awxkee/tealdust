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
use crate::itx_1d::{TX1D_FNS, TX1D_FNS_X8, residual_add_ctx, residual_add_strided_ctx};
use crate::itx_2d::{ITX_TMP_PIXELS, ITX_TMP_STRIDE};

// Test-only switch.

#[allow(clippy::too_many_arguments)]
#[allow(unused)]
pub(crate) fn inv_txfm_add<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [BD::Coef],
    txtp: u32,
    eob: i32,
    tx: usize,
    tmp_buf: &mut [i32; ITX_TMP_PIXELS],
) {
    let exec = crate::exec_context::ExecContext::new();
    inv_txfm_add_ctx(
        &exec, bd, dst, dst_off, stride, coeff, txtp, eob, tx, tmp_buf,
    );
}

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

use crate::levels::{txsz, txtp as txtp_kind};
use crate::pixel::{BitDepth, Coeff};
use crate::scan::LAST_EOB_PER_COL;
use crate::tables::{TX_SHIFT, TXFM_DIMENSIONS};

macro_rules! call_itx {
    ($f:expr, $($arg:expr),+ $(,)?) => {{
        // SAFETY: ITX resolvers only install x86 target-feature kernels after
        // the corresponding runtime CPU feature check succeeds. Scalar and
        // NEON entries are valid under the current target configuration.
        unsafe { ($f)($($arg),+) }
    }};
}

const TX_TYPE_LOW8_MASK: u32 = 0xFF;
const TX_TYPE_KIND_MASK: u32 = 0x7;
const TX_TYPE_CLASS_SHIFT: u32 = 3;
const TX_TYPE_CLASS_MASK: u32 = 0x3;
const TX_TYPE_SECOND_KIND_SHIFT: u32 = 5;
const TX_TYPE_EXT_SHIFT: u32 = 8;

#[inline(always)]
fn tx_type_low8(txtp: u32) -> u32 {
    txtp & TX_TYPE_LOW8_MASK
}

#[inline(always)]
fn tx_type_has_no_extension(txtp: u32) -> bool {
    (txtp >> TX_TYPE_EXT_SHIFT) == 0
}

#[inline(always)]
fn tx_type_class(txtp: u32) -> u32 {
    (txtp >> TX_TYPE_CLASS_SHIFT) & TX_TYPE_CLASS_MASK
}

#[inline(always)]
fn tx_type_first_kind(txtp: u32) -> usize {
    (txtp & TX_TYPE_KIND_MASK) as usize
}

#[inline(always)]
fn tx_type_second_kind(txtp: u32) -> usize {
    ((txtp >> TX_TYPE_SECOND_KIND_SHIFT) & TX_TYPE_KIND_MASK) as usize
}

const WHT_WHT: u32 = 6 | (6 << 5);

// Per-thread reusable transform scratch. The dequant cores and the generic
// inv_txfm_add path fully write the used S×S region before reading it (rows
// 0..last from the row pass, rows last..S zero-filled) and never read columns
// S..ITX_TMP_STRIDE, so leftover data from a previous transform can never leak
// into the result. The buffer is owned by the caller's `ReconScratch` and
// threaded in explicitly, so there is no thread-local lookup or `RefCell`
// borrow on the per-transform path. Proven bit-exact by the `tmp_init_proof`
// tests: arbitrary garbage in the buffer yields identical output to a zeroed one.

/// Run `f` with a `Txfm2d` view over the caller-supplied scratch buffer.
#[inline(always)]
fn with_itx_scratch<R>(buf: &mut [i32; ITX_TMP_PIXELS], f: impl FnOnce(&mut Txfm2d) -> R) -> R {
    let mut tmp = Txfm2d { buf };
    f(&mut tmp)
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
    exec: &crate::exec_context::ExecContext,
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
    #[cfg(target_arch = "aarch64")]
    {
        // If a transform did have to fall back to the tmp path on 8bpc
        // AArch64, keep the final residual add/writeback on NEON instead of
        // dropping into the scalar expansion loops below. The fully fused
        // NEON ITX path above still avoids tmp entirely for DCT/ADST/FLIPADST
        // pairs; this is the safety net for identity/H/V/DPCM-0 fallback
        // shapes.
        if BD::BPC == 8 && dpcm_flag == 0 {
            if let Some(dst8) = <BD::Pixel as crate::pixel::Pixel>::try_as_u8_slice_mut(dst) {
                if unsafe {
                    crate::neon::add_tmp_to_dst_8bpc_neon(
                        dst8,
                        dst_off,
                        stride,
                        tmp.as_slice(),
                        ITX_TMP_STRIDE,
                        w,
                        h,
                        sw,
                        sh,
                        rnd,
                        shift,
                    )
                } {
                    return;
                }
            }
        }
    }

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
        residual_add_strided_ctx(
            exec,
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
        residual_add_ctx(
            exec,
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
fn inv_txfm_add_typed<BD: BitDepth, C: Coeff>(
    exec: &crate::exec_context::ExecContext,
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [C],
    txtp: u32,
    eob: i32,
    tx: usize,
    tmp_buf: &mut [i32; ITX_TMP_PIXELS],
) {
    debug_assert_eq!(
        BD::BPC == 8,
        C::IS_I16,
        "8-bit ITX must use i16 coefficients; high-bit-depth ITX must use i32 coefficients"
    );

    if tx_type_low8(txtp) == WHT_WHT {
        assert!(tx == 0);
        let dpcm_flag = (txtp >> TX_TYPE_EXT_SHIFT) as u8;
        crate::itx_wht_dispatch::inv_wht_wht_4x4_dispatch_ctx(
            exec, bd, dst, dst_off, stride, coeff, dpcm_flag,
        );
        return;
    }

    let t_dim = &TXFM_DIMENSIONS[tx];
    let tx_sh = &TX_SHIFT[tx];
    let hbd = BD::BPC > 8;
    let w = 4 * t_dim.w as usize;
    let h = 4 * t_dim.h as usize;
    let is_rect2 = ((t_dim.lw + t_dim.lh) & 1) != 0;

    if eob + txtp as i32 == 0 {
        let shift_p1 = tx_sh[0] as i32;
        let shift = shift_p1 + tx_sh[1] as i32 - 12;
        let rnd = (1 << (shift - 1)) + shift_p1 - 6;
        let mut dc = coeff[0].to_i32();
        coeff[0] = C::ZERO;
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
            crate::filter::dc_add_row_ctx(exec, bd, d, dc, n);
        }
        return;
    }

    let first_kind = tx_type_first_kind(txtp);
    let second_kind = tx_type_second_kind(txtp);
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

    #[cfg(target_arch = "aarch64")]
    if BD::BPC == 8
        && tx_type_low8(txtp) == txtp_kind::DCT_DCT as u32
        && tx_type_has_no_extension(txtp)
        && t_dim.lw == t_dim.lh
        && (tx == txsz::TX_16X16 || tx == txsz::TX_32X32)
        && !force_generic_itx()
    {
        if let (Some(coeff16), Some(dst8)) = (
            C::try_as_i16_slice_mut(coeff),
            <BD::Pixel as crate::pixel::Pixel>::try_as_u8_slice_mut(dst),
        ) {
            unsafe {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    if tx == txsz::TX_16X16 {
                        crate::neon::idct_dequant_16x16_i16_neon_rdm_fused_8bpc(
                            coeff16,
                            dst8,
                            dst_off,
                            stride,
                            eob,
                            tx,
                            is_rect2,
                            shift0,
                            row_clip_min,
                            row_clip_max,
                            shift1,
                        );
                    } else {
                        crate::neon::idct_dequant_32x32_i16_neon_rdm_fused_8bpc(
                            coeff16,
                            dst8,
                            dst_off,
                            stride,
                            eob,
                            tx,
                            is_rect2,
                            shift0,
                            row_clip_min,
                            row_clip_max,
                            shift1,
                        );
                    }
                } else if tx == txsz::TX_16X16 {
                    crate::neon::idct_dequant_16x16_i16_neon_fused_8bpc(
                        coeff16,
                        dst8,
                        dst_off,
                        stride,
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                        shift1,
                    );
                } else {
                    crate::neon::idct_dequant_32x32_i16_neon_fused_8bpc(
                        coeff16,
                        dst8,
                        dst_off,
                        stride,
                        eob,
                        tx,
                        is_rect2,
                        shift0,
                        row_clip_min,
                        row_clip_max,
                        shift1,
                    );
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // win.
        let can_fuse_neon_8bpc = BD::BPC == 8
            && tx_type_has_no_extension(txtp)
            && tx_type_class(txtp) == 0
            && !force_generic_itx();
        if can_fuse_neon_8bpc {
            if let (Some(coeff16), Some(dst8)) = (
                C::try_as_i16_slice_mut(coeff),
                <BD::Pixel as crate::pixel::Pixel>::try_as_u8_slice_mut(dst),
            ) {
                let handled = unsafe {
                    if std::arch::is_aarch64_feature_detected!("rdm") {
                        crate::neon::itx_dequant_i16_neon_rdm_fused_8bpc(
                            coeff16,
                            dst8,
                            dst_off,
                            stride,
                            w,
                            h,
                            eob,
                            tx,
                            is_rect2,
                            shift0,
                            row_clip_min,
                            row_clip_max,
                            shift1,
                            first_kind,
                            second_kind,
                        )
                    } else {
                        crate::neon::itx_dequant_i16_neon_fused_8bpc(
                            coeff16,
                            dst8,
                            dst_off,
                            stride,
                            w,
                            h,
                            eob,
                            tx,
                            is_rect2,
                            shift0,
                            row_clip_min,
                            row_clip_max,
                            shift1,
                            first_kind,
                            second_kind,
                        )
                    }
                };
                if handled {
                    return;
                }
            }
        }
    }

    if BD::BPC == 8
        && tx_type_low8(txtp) == txtp_kind::DCT_DCT as u32
        && tx_type_has_no_extension(txtp)
        && t_dim.lw == t_dim.lh
        && (tx == txsz::TX_16X16 || tx == txsz::TX_32X32)
        && !force_generic_itx()
    {
        if let (Some(_coeff16), Some(_dst8)) = (
            C::try_as_i16_slice_mut(coeff),
            <BD::Pixel as crate::pixel::Pixel>::try_as_u8_slice_mut(dst),
        ) {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if crate::itx_1d::x86_itx_has_avx512() {
                    if tx == txsz::TX_16X16 {
                        unsafe {
                            crate::avx::idct_dequant_16x16_i16_avx512_fused_8bpc(
                                _coeff16,
                                _dst8,
                                dst_off,
                                stride,
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                                shift1,
                            )
                        };
                    } else {
                        unsafe {
                            crate::avx::idct_dequant_32x32_i16_avx512_fused_8bpc(
                                _coeff16,
                                _dst8,
                                dst_off,
                                stride,
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                                shift1,
                            )
                        };
                    }
                    return;
                }
                if crate::itx_1d::x86_itx_has_avx2() {
                    if tx == txsz::TX_16X16 {
                        unsafe {
                            crate::avx::idct_dequant_16x16_i16_avx2_fused_8bpc(
                                _coeff16,
                                _dst8,
                                dst_off,
                                stride,
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                                shift1,
                            )
                        };
                    } else {
                        unsafe {
                            crate::avx::idct_dequant_32x32_i16_avx2_fused_8bpc(
                                _coeff16,
                                _dst8,
                                dst_off,
                                stride,
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                                shift1,
                            )
                        };
                    }
                    return;
                }
            }
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
            {
                if crate::itx_1d::x86_itx_has_sse41() {
                    if tx == txsz::TX_16X16 {
                        unsafe {
                            crate::sse::idct_dequant_16x16_i16_sse41_fused_8bpc(
                                _coeff16,
                                _dst8,
                                dst_off,
                                stride,
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                                shift1,
                            )
                        };
                    } else {
                        unsafe {
                            crate::sse::idct_dequant_32x32_i16_sse41_fused_8bpc(
                                _coeff16,
                                _dst8,
                                dst_off,
                                stride,
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                                shift1,
                            )
                        };
                    }
                    return;
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        let can_fuse_avx2_8bpc = BD::BPC == 8
            && tx_type_has_no_extension(txtp)
            && tx_type_class(txtp) == 0
            && crate::itx_2d::is_itx_dense_kind(first_kind)
            && crate::itx_2d::is_itx_dense_kind(second_kind)
            && !force_generic_itx();
        if can_fuse_avx2_8bpc {
            if let (Some(coeff16), Some(dst8)) = (
                C::try_as_i16_slice_mut(coeff),
                <BD::Pixel as crate::pixel::Pixel>::try_as_u8_slice_mut(dst),
            ) {
                if crate::itx_1d::x86_itx_has_avx512() {
                    let handled = unsafe {
                        crate::avx::itx_dequant_i16_avx512_fused_8bpc(
                            coeff16,
                            dst8,
                            dst_off,
                            stride,
                            w,
                            h,
                            eob,
                            tx,
                            is_rect2,
                            shift0,
                            row_clip_min,
                            row_clip_max,
                            shift1,
                            first_kind,
                            second_kind,
                        )
                    };
                    if handled {
                        return;
                    }
                }
                if crate::itx_1d::x86_itx_has_avx2() {
                    let handled = unsafe {
                        crate::avx::itx_dequant_i16_avx2_fused_8bpc(
                            coeff16,
                            dst8,
                            dst_off,
                            stride,
                            w,
                            h,
                            eob,
                            tx,
                            is_rect2,
                            shift0,
                            row_clip_min,
                            row_clip_max,
                            shift1,
                            first_kind,
                            second_kind,
                        )
                    };
                    if handled {
                        return;
                    }
                }
            }
        }
    }

    if tx_type_low8(txtp) == txtp_kind::DCT_DCT as u32
        && tx_type_has_no_extension(txtp)
        && t_dim.lw == t_dim.lh
    {
        if BD::BPC == 8 {
            if let Some(coeff16) = C::try_as_i16_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::TX_4X4 => {
                            let f = crate::itx_2d::idct_dequant_4x4_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_8X8 => {
                            let f = crate::itx_2d::idct_dequant_8x8_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_16X16 => {
                            let f = crate::itx_2d::idct_dequant_16x16_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_32X32 => {
                            let f = crate::itx_2d::idct_dequant_32x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_64X64 => {
                            let f = crate::itx_2d::idct_dequant_64x64_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
        if BD::BPC > 8 {
            if let Some(coeff32) = C::try_as_i32_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::TX_4X4 => {
                            let f = crate::itx_2d::idct_dequant_4x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_8X8 => {
                            let f = crate::itx_2d::idct_dequant_8x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_16X16 => {
                            let f = crate::itx_2d::idct_dequant_16x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_32X32 => {
                            let f = crate::itx_2d::idct_dequant_32x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::TX_64X64 => {
                            let f = crate::itx_2d::idct_dequant_64x64(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
    }

    // Non-square DCT_DCT. Dims <= 32 use dedicated rectangular cores. The
    // 64-involving sizes have no real 64-point transform (the decoder maps the
    // 64 dimension to inv_dct32), so each computes identically to its clamped
    // (min(W,32), min(H,32)) shape and reuses that core, with the caller's `tx`
    // (eob table) and `is_rect2` (scaling) selecting the correct behavior.
    if tx_type_low8(txtp) == txtp_kind::DCT_DCT as u32
        && tx_type_has_no_extension(txtp)
        && t_dim.lw != t_dim.lh
        && !force_generic_itx()
    {
        if BD::BPC == 8 {
            if let Some(coeff16) = C::try_as_i16_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::RTX_4X8 => {
                            let f = crate::itx_2d::idct_dequant_4x8_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X4 => {
                            let f = crate::itx_2d::idct_dequant_8x4_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X16 => {
                            let f = crate::itx_2d::idct_dequant_8x16_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X8 => {
                            let f = crate::itx_2d::idct_dequant_16x8_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X32 => {
                            let f = crate::itx_2d::idct_dequant_16x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X16 => {
                            let f = crate::itx_2d::idct_dequant_32x16_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_4X16 => {
                            let f = crate::itx_2d::idct_dequant_4x16_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X4 => {
                            let f = crate::itx_2d::idct_dequant_16x4_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X32 => {
                            let f = crate::itx_2d::idct_dequant_8x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X8 => {
                            let f = crate::itx_2d::idct_dequant_32x8_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_4X32 => {
                            let f = crate::itx_2d::idct_dequant_4x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X4 => {
                            let f = crate::itx_2d::idct_dequant_32x4_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X64 => {
                            let f = crate::itx_2d::idct_dequant_32x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X32 => {
                            let f = crate::itx_2d::idct_dequant_32x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X64 => {
                            let f = crate::itx_2d::idct_dequant_16x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X16 => {
                            let f = crate::itx_2d::idct_dequant_32x16_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X64 => {
                            let f = crate::itx_2d::idct_dequant_8x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X8 => {
                            let f = crate::itx_2d::idct_dequant_32x8_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_4X64 => {
                            let f = crate::itx_2d::idct_dequant_4x32_i16();
                            call_itx!(
                                f,
                                coeff16,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X4 => {
                            let f = crate::itx_2d::idct_dequant_32x4_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
        if BD::BPC > 8 {
            if let Some(coeff32) = C::try_as_i32_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::RTX_4X8 => {
                            let f = crate::itx_2d::idct_dequant_4x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X4 => {
                            let f = crate::itx_2d::idct_dequant_8x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X16 => {
                            let f = crate::itx_2d::idct_dequant_8x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X8 => {
                            let f = crate::itx_2d::idct_dequant_16x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X32 => {
                            let f = crate::itx_2d::idct_dequant_16x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X16 => {
                            let f = crate::itx_2d::idct_dequant_32x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_4X16 => {
                            let f = crate::itx_2d::idct_dequant_4x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X4 => {
                            let f = crate::itx_2d::idct_dequant_16x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X32 => {
                            let f = crate::itx_2d::idct_dequant_8x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X8 => {
                            let f = crate::itx_2d::idct_dequant_32x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_4X32 => {
                            let f = crate::itx_2d::idct_dequant_4x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X4 => {
                            let f = crate::itx_2d::idct_dequant_32x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_32X64 => {
                            let f = crate::itx_2d::idct_dequant_32x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X32 => {
                            let f = crate::itx_2d::idct_dequant_32x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_16X64 => {
                            let f = crate::itx_2d::idct_dequant_16x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X16 => {
                            let f = crate::itx_2d::idct_dequant_32x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_8X64 => {
                            let f = crate::itx_2d::idct_dequant_8x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X8 => {
                            let f = crate::itx_2d::idct_dequant_32x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_4X64 => {
                            let f = crate::itx_2d::idct_dequant_4x32(hbd);
                            call_itx!(
                                f,
                                coeff32,
                                tmp.as_mut_array(),
                                eob,
                                tx,
                                is_rect2,
                                shift0,
                                row_clip_min,
                                row_clip_max,
                            );
                        }
                        txsz::RTX_64X4 => {
                            let f = crate::itx_2d::idct_dequant_32x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
    }

    if tx_type_has_no_extension(txtp)
        && tx_type_class(txtp) == 0
        && t_dim.lw == t_dim.lh
        && t_dim.lw <= 2
        && crate::itx_2d::is_dct_adst_kind(first_kind)
        && crate::itx_2d::is_dct_adst_kind(second_kind)
        && (first_kind != crate::itx_2d::TX_KIND_DCT || second_kind != crate::itx_2d::TX_KIND_DCT)
    {
        if BD::BPC == 8 {
            if let Some(coeff16) = C::try_as_i16_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::TX_4X4 => {
                            let f = crate::itx_2d::iadst_dequant_4x4_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::TX_8X8 => {
                            let f = crate::itx_2d::iadst_dequant_8x8_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::TX_16X16 => {
                            let f = crate::itx_2d::iadst_dequant_16x16_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
        if BD::BPC > 8 {
            if let Some(coeff32) = C::try_as_i32_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::TX_4X4 => {
                            let f = crate::itx_2d::iadst_dequant_4x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::TX_8X8 => {
                            let f = crate::itx_2d::iadst_dequant_8x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::TX_16X16 => {
                            let f = crate::itx_2d::iadst_dequant_16x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
    }

    if tx_type_has_no_extension(txtp)
        && tx_type_class(txtp) == 0
        && t_dim.lw != t_dim.lh
        && t_dim.lw <= 2
        && t_dim.lh <= 2
        && crate::itx_2d::is_dct_adst_kind(first_kind)
        && crate::itx_2d::is_dct_adst_kind(second_kind)
        && (first_kind != crate::itx_2d::TX_KIND_DCT || second_kind != crate::itx_2d::TX_KIND_DCT)
        && !force_generic_itx()
    {
        if BD::BPC == 8 {
            if let Some(coeff16) = C::try_as_i16_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::RTX_4X8 => {
                            let f = crate::itx_2d::iadst_dequant_4x8_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::RTX_8X4 => {
                            let f = crate::itx_2d::iadst_dequant_8x4_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::RTX_8X16 => {
                            let f = crate::itx_2d::iadst_dequant_8x16_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::RTX_16X8 => {
                            let f = crate::itx_2d::iadst_dequant_16x8_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::RTX_4X16 => {
                            let f = crate::itx_2d::iadst_dequant_4x16_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        txsz::RTX_16X4 => {
                            let f = crate::itx_2d::iadst_dequant_16x4_i16();
                            call_itx!(
                                f,
                                coeff16,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
        if BD::BPC > 8 {
            if let Some(coeff32) = C::try_as_i32_slice_mut(coeff) {
                let handled = with_itx_scratch(&mut *tmp_buf, |tmp| {
                    let mut handled = true;

                    match tx {
                        txsz::RTX_4X8 => {
                            let f = crate::itx_2d::iadst_dequant_4x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::RTX_8X4 => {
                            let f = crate::itx_2d::iadst_dequant_8x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::RTX_8X16 => {
                            let f = crate::itx_2d::iadst_dequant_8x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::RTX_16X8 => {
                            let f = crate::itx_2d::iadst_dequant_16x8(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::RTX_4X16 => {
                            let f = crate::itx_2d::iadst_dequant_4x16(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        txsz::RTX_16X4 => {
                            let f = crate::itx_2d::iadst_dequant_16x4(hbd);
                            call_itx!(
                                f,
                                coeff32,
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
                        add_tmp_to_dst(
                            exec, bd, dst, dst_off, stride, tmp, w, h, sw, sh, rnd1, shift1, 0,
                        );
                    }
                    handled
                });
                if handled {
                    return;
                }
            }
        }
    }

    with_itx_scratch(&mut *tmp_buf, |tmp| {
        let mut row = 0usize;
        let tx_class = tx_type_class(txtp);

        if tx_class == 0 {
            let off = LAST_EOB_PER_COL.offset[tx] as usize;
            let last_eob = &LAST_EOB_PER_COL.table[off..];
            let mut ei = 0usize;
            loop {
                let tmp_row = tmp.row_mut(row);
                for (x, dst) in tmp_row[..sw].iter_mut().enumerate() {
                    let v = coeff[row + x * sh].to_i32();
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
                    let v = coeff[row + x * sh].to_i32();
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
        coeff[..sw * sh].fill(C::ZERO);

        let rnd0 = (1 << shift0) >> 1;
        for y in 0..sh {
            crate::filter::row_clip_ctx(
                exec,
                tmp.row_mut(y),
                sw,
                rnd0,
                shift0,
                row_clip_min,
                row_clip_max,
            );
        }

        let second_1d_fn_x8 = TX1D_FNS_X8[t_dim.lh as usize][second_kind];
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
            exec,
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
            (txtp >> TX_TYPE_EXT_SHIFT) as u8,
        );
    });
}

pub(crate) fn inv_txfm_add_ctx<BD: BitDepth>(
    exec: &crate::exec_context::ExecContext,
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [BD::Coef],
    txtp: u32,
    eob: i32,
    tx: usize,
    tmp_buf: &mut [i32; ITX_TMP_PIXELS],
) {
    inv_txfm_add_typed(
        exec, bd, dst, dst_off, stride, coeff, txtp, eob, tx, tmp_buf,
    );
}

fn cctx_bd_i32_ctx<BD: BitDepth>(
    exec: &crate::exec_context::ExecContext,
    bd: BD,
    u: &mut [i32],
    v: &mut [i32],
    angle: &[i16; 3],
    sz: usize,
) {
    crate::itx_1d::cctx_ctx(exec, u, v, angle, sz, bd.bitdepth() as i32);
}

/// Cross-component transform clip at the coded bit depth (`cctx_c`).
pub fn cctx_bd_ctx<BD: BitDepth, C: Coeff>(
    exec: &crate::exec_context::ExecContext,
    bd: BD,
    u: &mut [C],
    v: &mut [C],
    angle: &[i16; 3],
    sz: usize,
) {
    if let (Some(u32), Some(v32)) = (C::try_as_i32_slice_mut(u), C::try_as_i32_slice_mut(v)) {
        cctx_bd_i32_ctx(exec, bd, u32, v32, angle, sz);
        return;
    }

    debug_assert!(sz.is_power_of_two() && (16..=1024).contains(&sz));
    let n = sz.min(u.len()).min(v.len());
    let min = -(1 << (bd.bitdepth() as i32 + 7));
    let max = (1 << (bd.bitdepth() as i32 + 7)) - 1;
    let sina = angle[0] as i32;
    let cosa = angle[1] as i32;
    debug_assert!(angle[2] == -angle[0]);

    if let (Some(u16), Some(v16)) = (C::try_as_i16_slice_mut(u), C::try_as_i16_slice_mut(v)) {
        unsafe { (exec.cctx_i16)(u16, v16, sina, cosa, n, min, max) };
        return;
    }

    for i in 0..n {
        let ui = u[i].to_i32();
        let vi = v[i].to_i32();
        let a = ui * cosa - vi * sina;
        let b = ui * sina + vi * cosa;
        u[i] = C::from_i32(((a + 128 - (a < 0) as i32) >> 8).max(min).min(max));
        v[i] = C::from_i32(((b + 128 - (b < 0) as i32) >> 8).max(min).min(max));
    }
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

    fn poisoned_buf() -> Box<[i32; crate::itx_2d::ITX_TMP_PIXELS]> {
        Box::new([0xDEAD_BEEFu32 as i32; crate::itx_2d::ITX_TMP_PIXELS])
    }
    fn zeroed_buf() -> Box<[i32; crate::itx_2d::ITX_TMP_PIXELS]> {
        Box::new([0i32; crate::itx_2d::ITX_TMP_PIXELS])
    }

    fn check(tx: usize, txtp: u32, seed: u64) {
        let mut rng = Rng(seed);
        let stride = 64usize;
        for _ in 0..120 {
            let mut coeff = [0i16; 4096];
            for v in coeff.iter_mut() {
                *v = rng.coef() as i16;
            }
            let eob = ((rng.next() % 64) + 1) as i32;
            let base = vec![100u8; stride * 64 + stride];

            let mut buf1 = poisoned_buf();
            let (mut d1, mut c1) = (base.clone(), coeff);
            inv_txfm_add(
                BitDepth8, &mut d1, 0, stride, &mut c1, txtp, eob, tx, &mut buf1,
            );

            let mut buf2 = zeroed_buf();
            let (mut d2, mut c2) = (base.clone(), coeff);
            inv_txfm_add(
                BitDepth8, &mut d2, 0, stride, &mut c2, txtp, eob, tx, &mut buf2,
            );

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
            let mut coeff0 = vec![0i16; n + 16];
            for v in coeff0[..sw * sh].iter_mut() {
                *v = rng.range(-(1 << 12), 1 << 12) as i16;
            }
            let eob = rng.range(1, (sw * sh) as i32);

            let dst_init: Vec<u8> = (0..stride * h).map(|_| rng.range(0, 256) as u8).collect();

            let mut dst_rect = dst_init.clone();
            let mut c_rect = coeff0.clone();
            FORCE_GENERIC_ITX.store(false, Ordering::Relaxed);
            let mut buf_rect = Box::new([0i32; crate::itx_2d::ITX_TMP_PIXELS]);
            inv_txfm_add::<BitDepth8>(
                BitDepth8,
                &mut dst_rect,
                0,
                stride,
                &mut c_rect,
                txtp,
                eob,
                tx,
                &mut buf_rect,
            );

            let mut dst_gen = dst_init.clone();
            let mut c_gen = coeff0.clone();
            FORCE_GENERIC_ITX.store(true, Ordering::Relaxed);
            let mut buf_gen = Box::new([0i32; crate::itx_2d::ITX_TMP_PIXELS]);
            inv_txfm_add::<BitDepth8>(
                BitDepth8,
                &mut dst_gen,
                0,
                stride,
                &mut c_gen,
                txtp,
                eob,
                tx,
                &mut buf_gen,
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
