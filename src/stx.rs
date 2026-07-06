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

use crate::intops::{apply_sign, iclip};
use crate::pixel::Coeff;
use std::sync::OnceLock;

pub(crate) type Stx4Fn8bpc = unsafe fn(&mut [i16], &[i8], usize, &[u8; 16]);
pub(crate) type Stx8Fn8bpc = unsafe fn(&mut [i16], &[i8], usize, &[u8; 64], &[u8; 48]);
pub(crate) type Stx4FnHbd = unsafe fn(&mut [i32], &[i8], usize, i32, &[u8; 16]);
pub(crate) type Stx8FnHbd = unsafe fn(&mut [i32], &[i8], usize, i32, &[u8; 64], &[u8; 48]);

pub(crate) fn stxfm<C: Coeff>(
    cf_out: &mut [i32],
    cf: &[C],
    kernel: &[i8],
    sz: usize,
    eob: usize,
    bitdepth_max: i32,
) {
    debug_assert!(sz == 16 || sz == 48);
    debug_assert!(eob < if sz == 16 { 8 } else { 32 });
    let min = -128 * (1 + bitdepth_max);
    let max = 128 * (1 + bitdepth_max) - 1;
    let h = eob + 1;
    for (x, cf_out) in cf_out[..sz].iter_mut().enumerate() {
        let mut sum = 0i32;
        for (y, &cf) in cf[..h].iter().enumerate() {
            sum += cf.to_i32() * kernel[y * sz + x] as i32;
        }
        sum = apply_sign((sum.abs() + 64) >> 7, sum);
        *cf_out = iclip(sum, min, max);
    }
}

#[inline]
pub(crate) fn stxfm4_8bpc_scalar(cf: &mut [i16], kernel: &[i8], eob: usize, scan_out: &[u8; 16]) {
    let mut sums = [0i32; 16];
    stxfm(&mut sums, cf, kernel, 16, eob, 255);
    cf[4..8].fill(0);
    for (&scan, &sum) in scan_out.iter().zip(sums.iter()) {
        cf[scan as usize] = sum as i16;
    }
}

#[inline]
pub(crate) fn stxfm8_8bpc_scalar(
    cf: &mut [i16],
    kernel: &[i8],
    eob: usize,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    let mut sums = [0i32; 48];
    stxfm(&mut sums, cf, kernel, 48, eob, 255);
    cf[..32].fill(0);
    for (&map, &sum) in mapping.iter().zip(sums.iter()) {
        cf[scan_out[map as usize] as usize] = sum as i16;
    }
}

#[inline]
pub(crate) fn stxfm4_hbd_scalar(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 16],
) {
    let mut sums = [0i32; 16];
    stxfm(&mut sums, cf, kernel, 16, eob, bitdepth_max);
    cf[4..8].fill(0);
    for (&scan, &sum) in scan_out.iter().zip(sums.iter()) {
        cf[scan as usize] = sum;
    }
}

#[inline]
pub(crate) fn stxfm8_hbd_scalar(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    let mut sums = [0i32; 48];
    stxfm(&mut sums, cf, kernel, 48, eob, bitdepth_max);
    cf[..32].fill(0);
    for (&map, &sum) in mapping.iter().zip(sums.iter()) {
        cf[scan_out[map as usize] as usize] = sum;
    }
}

static STX4_8BPC: OnceLock<Stx4Fn8bpc> = OnceLock::new();
static STX8_8BPC: OnceLock<Stx8Fn8bpc> = OnceLock::new();
static STX4_HBD: OnceLock<Stx4FnHbd> = OnceLock::new();
static STX8_HBD: OnceLock<Stx8FnHbd> = OnceLock::new();

#[inline]
fn resolve_stxfm4_8bpc() -> Stx4Fn8bpc {
    *STX4_8BPC.get_or_init(|| {
        let mut _f = stxfm4_8bpc_scalar as Stx4Fn8bpc;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::stxfm4_8bpc_neon as Stx4Fn8bpc;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::stxfm4_8bpc_avx2 as Stx4Fn8bpc;
            }
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
                && std::is_x86_feature_detected!("avx512vnni")
            {
                _f = crate::avx::stxfm4_8bpc_avx512 as Stx4Fn8bpc;
            }
        }
        _f
    })
}

#[inline]
fn resolve_stxfm8_8bpc() -> Stx8Fn8bpc {
    *STX8_8BPC.get_or_init(|| {
        let mut _f = stxfm8_8bpc_scalar as Stx8Fn8bpc;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::stxfm8_8bpc_neon as Stx8Fn8bpc;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::stxfm8_8bpc_avx2 as Stx8Fn8bpc;
            }
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
                && std::is_x86_feature_detected!("avx512vnni")
            {
                _f = crate::avx::stxfm8_8bpc_avx512 as Stx8Fn8bpc;
            }
        }
        _f
    })
}

#[inline]
fn resolve_stxfm4_hbd() -> Stx4FnHbd {
    *STX4_HBD.get_or_init(|| {
        let mut _f = stxfm4_hbd_scalar as Stx4FnHbd;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::stxfm4_hbd_neon as Stx4FnHbd;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::stxfm4_hbd_avx2 as Stx4FnHbd;
            }
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                _f = crate::avx::stxfm4_hbd_avx512 as Stx4FnHbd;
            }
        }
        _f
    })
}

#[inline]
fn resolve_stxfm8_hbd() -> Stx8FnHbd {
    *STX8_HBD.get_or_init(|| {
        let mut _f = stxfm8_hbd_scalar as Stx8FnHbd;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::stxfm8_hbd_neon as Stx8FnHbd;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::stxfm8_hbd_avx2 as Stx8FnHbd;
            }
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
            {
                _f = crate::avx::stxfm8_hbd_avx512 as Stx8FnHbd;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn stxfm4_dispatch<C: Coeff>(
    cf: &mut [C],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 16],
) {
    if let Some(cf16) = C::try_as_i16_slice_mut(cf) {
        // SAFETY: resolver installs AVX2/NEON only after runtime feature detection;
        // scalar fallback has no CPU feature requirement.  All callers pass a
        // coefficient block large enough for the scan table they selected.
        unsafe { resolve_stxfm4_8bpc()(cf16, kernel, eob, scan_out) };
        return;
    }
    if let Some(cf32) = C::try_as_i32_slice_mut(cf) {
        // SAFETY: same dispatch guarantee as the 8bpc path.  HBD keeps i32
        // coefficient storage, so this path uses separate 32-bit SIMD kernels
        // and clips to the coded bitdepth STX coefficient range.
        unsafe { resolve_stxfm4_hbd()(cf32, kernel, eob, bitdepth_max, scan_out) };
        return;
    }

    let mut sums = [0i32; 16];
    stxfm(&mut sums, cf, kernel, 16, eob, bitdepth_max);
    cf[4..8].fill(C::ZERO);
    for (&scan, &sum) in scan_out.iter().zip(sums.iter()) {
        cf[scan as usize] = C::from_i32(sum);
    }
}

#[inline]
pub(crate) fn stxfm8_dispatch<C: Coeff>(
    cf: &mut [C],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    if let Some(cf16) = C::try_as_i16_slice_mut(cf) {
        // SAFETY: see stxfm4_dispatch; the mapping/scan tables are static and
        // selected with the same indices as the scalar path.
        unsafe { resolve_stxfm8_8bpc()(cf16, kernel, eob, scan_out, mapping) };
        return;
    }
    if let Some(cf32) = C::try_as_i32_slice_mut(cf) {
        // SAFETY: same as stxfm4_dispatch; HBD uses i32 coefficients and the
        // function pointer is resolved once with runtime CPU feature detection.
        unsafe { resolve_stxfm8_hbd()(cf32, kernel, eob, bitdepth_max, scan_out, mapping) };
        return;
    }

    let mut sums = [0i32; 48];
    stxfm(&mut sums, cf, kernel, 48, eob, bitdepth_max);
    cf[..32].fill(C::ZERO);
    for (&map, &sum) in mapping.iter().zip(sums.iter()) {
        cf[scan_out[map as usize] as usize] = C::from_i32(sum);
    }
}
