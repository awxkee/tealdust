/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use std::sync::OnceLock;

use crate::itx_1d::{inv_wht_wht_4x4, residual_add};
use crate::pixel::{BitDepth, Coeff, Pixel};

pub(crate) type InvWht4x4Fn8bpc = unsafe fn(&mut [i16], &mut [u8], usize, usize);
pub(crate) type InvWht4x4FnHbd = unsafe fn(&mut [i32], &mut [u16], usize, usize, i32);

static INV_WHT_WHT_4X4_8BPC: OnceLock<Option<InvWht4x4Fn8bpc>> = OnceLock::new();
static INV_WHT_WHT_4X4_HBD: OnceLock<Option<InvWht4x4FnHbd>> = OnceLock::new();

#[inline]
fn resolve_inv_wht_wht_4x4_8bpc() -> Option<InvWht4x4Fn8bpc> {
    *INV_WHT_WHT_4X4_8BPC.get_or_init(|| {
        let mut _f: Option<InvWht4x4Fn8bpc> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::inv_wht_wht_4x4_i16_neon_8bpc as InvWht4x4Fn8bpc);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::itx_1d::x86_itx_has_sse41() {
                _f = Some(crate::sse::inv_wht_wht_4x4_i16_sse41_8bpc as InvWht4x4Fn8bpc);
            }
        }
        _f
    })
}

#[inline]
fn resolve_inv_wht_wht_4x4_hbd() -> Option<InvWht4x4FnHbd> {
    *INV_WHT_WHT_4X4_HBD.get_or_init(|| {
        let mut _f: Option<InvWht4x4FnHbd> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::inv_wht_wht_4x4_i32_neon_hbd as InvWht4x4FnHbd);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::itx_1d::x86_itx_has_sse41() {
                _f = Some(crate::sse::inv_wht_wht_4x4_i32_sse41_hbd as InvWht4x4FnHbd);
            }
        }
        _f
    })
}

#[inline(always)]
fn has_contiguous_4x4(dst_len: usize, dst_off: usize, stride: usize) -> bool {
    if stride < 4 {
        return false;
    }
    let Some(last_row) = 3usize
        .checked_mul(stride)
        .and_then(|v| dst_off.checked_add(v))
    else {
        return false;
    };
    last_row.checked_add(4).is_some_and(|end| end <= dst_len)
}

#[inline(never)]
fn inv_wht_wht_4x4_scalar<BD: BitDepth, C: Coeff>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [C],
    dpcm_flag: u8,
) {
    let mut wht_coeff = [0i32; 16];
    for (dst, &src) in wht_coeff.iter_mut().zip(&coeff[..16]) {
        *dst = src.to_i32();
    }
    let mut tmp = [0i32; 16];
    inv_wht_wht_4x4(&wht_coeff, &mut tmp);
    coeff[..16].fill(C::ZERO);
    residual_add(bd, &mut dst[dst_off..], stride, &tmp, 4, 4, 0, 0, dpcm_flag);
}

#[inline]
pub(crate) fn inv_wht_wht_4x4_dispatch<BD: BitDepth, C: Coeff>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_off: usize,
    stride: usize,
    coeff: &mut [C],
    dpcm_flag: u8,
) {
    debug_assert!(coeff.len() >= 16);

    if dpcm_flag == 0 && has_contiguous_4x4(dst.len(), dst_off, stride) {
        if let (Some(coeff16), Some(dst8), Some(f)) = (
            C::try_as_i16_slice_mut(coeff),
            <BD::Pixel as Pixel>::try_as_u8_slice_mut(dst),
            resolve_inv_wht_wht_4x4_8bpc(),
        ) {
            // SAFETY: the resolver installs target-feature kernels only when the
            // CPU supports them. `has_contiguous_4x4` proves each 4-wide output
            // row is inside `dst`; WHT always reads/writes exactly 16 coeffs.
            unsafe { f(coeff16, dst8, dst_off, stride) };
            return;
        }

        if let (Some(coeff32), Some(dst16), Some(f)) = (
            C::try_as_i32_slice_mut(coeff),
            <BD::Pixel as Pixel>::try_as_u16_slice_mut(dst),
            resolve_inv_wht_wht_4x4_hbd(),
        ) {
            // SAFETY: same dispatch and bounds guarantee as the 8bpc path.  The
            // HBD kernel clips to the coded bit depth carried by `bd`.
            unsafe { f(coeff32, dst16, dst_off, stride, bd.bitdepth_max()) };
            return;
        }
    }

    inv_wht_wht_4x4_scalar(bd, dst, dst_off, stride, coeff, dpcm_flag);
}
