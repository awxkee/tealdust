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

use crate::itx_1d::{TX1D_FNS, inv_dct4_1d, inv_dct8_1d, inv_dct16_1d, inv_dct32_1d};
use crate::pixel::Coeff;
use crate::scan::LAST_EOB_PER_COL;
use std::convert::TryInto;
use std::sync::OnceLock;

pub(crate) const ITX_TMP_STRIDE: usize = 32;
pub(crate) const ITX_TMP_PIXELS: usize = ITX_TMP_STRIDE * ITX_TMP_STRIDE;

// ITX entry bodies are expanded into the architecture modules so the large
// row/column control flow stays inside the same feature-local function as the
// AVX2/SSE4.1/NEON intrinsics. The helpers below are deliberately macro bodies,
// not wrapper functions: this avoids the old `arch entry -> shared generic core`
// call boundary that was repeatedly preventing reliable feature-local inlining.
#[macro_export]
macro_rules! itx_idct_dequant_simd4_body {
    ($backend:ty, $n:expr, $s:expr, $coeff_ty:ty, $coeff:expr, $tmp:expr, $eob:expr, $tx:expr, $is_rect2:expr, $shift0:expr, $row_clip_min:expr, $row_clip_max:expr $(,)?) => {{
        if $is_rect2 {
            $crate::itx_2d::idct_dequant_rows_rect_dct_simd4::<$backend, $n, $s, $s, $coeff_ty>(
                $coeff,
                $tmp,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
            );
        } else {
            $crate::itx_2d::idct_dequant_rows_dct_simd4::<$backend, $n, $s, $coeff_ty>(
                $coeff,
                $tmp,
                $eob,
                $tx,
                $shift0,
                $row_clip_min,
                $row_clip_max,
            );
        }

        let mut x = 0usize;
        if <$coeff_ty as $crate::itx_2d::ItxCoeff>::USE_WIDE_16BIT && ($s == 16 || $s == 32) {
            while x + 8 <= $s {
                $crate::itx_2d::dct_1d_wide_x8::<<$backend as $crate::itx_2d::DctSimd4>::Wide, $s>(
                    $tmp, x,
                );
                x += 8;
            }
        }
        while x + 4 <= $s {
            if !(<$coeff_ty as $crate::itx_2d::ItxCoeff>::USE_WIDE_16BIT
                && $crate::itx_2d::itx_1d_wide_x4::<$backend, $s, { $crate::itx_2d::TX_KIND_DCT }>(
                    $tmp, x,
                ))
            {
                $crate::itx_2d::dct_1d_x4::<$backend, $s>($tmp, x);
            }
            x += 4;
        }
        while x < $s {
            $crate::itx_2d::dct_1d::<$s>(&mut $tmp[x..], $crate::itx_2d::ITX_TMP_STRIDE);
            x += 1;
        }
    }};
}

#[macro_export]
macro_rules! itx_kind_dequant_simd4_mono_body {
    ($backend:ty, $n:expr, $s:expr, $coeff_ty:ty, $first_kind:expr, $second_kind:expr, $coeff:expr, $tmp:expr, $eob:expr, $tx:expr, $is_rect2:expr, $shift0:expr, $row_clip_min:expr, $row_clip_max:expr $(,)?) => {{
        if $is_rect2 {
            $crate::itx_2d::itx_dequant_scalar_core_mono::<
                $n,
                $s,
                $coeff_ty,
                $first_kind,
                $second_kind,
            >(
                $coeff,
                $tmp,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
            );
        } else {
            $crate::itx_2d::itx_dequant_rows_simd4::<$backend, $n, $s, $coeff_ty, $first_kind>(
                $coeff,
                $tmp,
                $eob,
                $tx,
                $shift0,
                $row_clip_min,
                $row_clip_max,
            );

            let mut x = 0usize;
            if <$coeff_ty as $crate::itx_2d::ItxCoeff>::USE_WIDE_16BIT && ($s == 8 || $s == 16) {
                while x + 8 <= $s {
                    if !$crate::itx_2d::itx_1d_wide_x8::<$backend, $s, $second_kind>($tmp, x) {
                        break;
                    }
                    x += 8;
                }
            }
            while x + 4 <= $s {
                if !(<$coeff_ty as $crate::itx_2d::ItxCoeff>::USE_WIDE_16BIT
                    && $crate::itx_2d::itx_1d_wide_x4::<$backend, $s, $second_kind>($tmp, x))
                {
                    $crate::itx_2d::itx_1d_x4::<$backend, $s, $second_kind>($tmp, x);
                }
                x += 4;
            }
            while x < $s {
                $crate::itx_2d::tx_1d_scalar_mono::<$s, $second_kind>(
                    &mut $tmp[x..],
                    $crate::itx_2d::ITX_TMP_STRIDE,
                );
                x += 1;
            }
        }
    }};
}

#[macro_export]
macro_rules! itx_kind_dequant_simd4_body {
    ($backend:ty, $n:expr, $s:expr, $coeff_ty:ty, $coeff:expr, $tmp:expr, $eob:expr, $tx:expr, $is_rect2:expr, $shift0:expr, $row_clip_min:expr, $row_clip_max:expr, $first_kind:expr, $second_kind:expr $(,)?) => {{
        match ($first_kind, $second_kind) {
            ($crate::itx_2d::TX_KIND_DCT, $crate::itx_2d::TX_KIND_DCT) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_DCT },
                    { $crate::itx_2d::TX_KIND_DCT },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_DCT, $crate::itx_2d::TX_KIND_ADST) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_DCT },
                    { $crate::itx_2d::TX_KIND_ADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_DCT, $crate::itx_2d::TX_KIND_FLIPADST) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_DCT },
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_ADST, $crate::itx_2d::TX_KIND_DCT) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_ADST },
                    { $crate::itx_2d::TX_KIND_DCT },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_ADST, $crate::itx_2d::TX_KIND_ADST) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_ADST },
                    { $crate::itx_2d::TX_KIND_ADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_ADST, $crate::itx_2d::TX_KIND_FLIPADST) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_ADST },
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_FLIPADST, $crate::itx_2d::TX_KIND_DCT) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    { $crate::itx_2d::TX_KIND_DCT },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_FLIPADST, $crate::itx_2d::TX_KIND_ADST) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    { $crate::itx_2d::TX_KIND_ADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_FLIPADST, $crate::itx_2d::TX_KIND_FLIPADST) => {
                $crate::itx_kind_dequant_simd4_mono_body!(
                    $backend,
                    $n,
                    $s,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            _ => unreachable!(),
        }
    }};
}

#[macro_export]
macro_rules! itx_idct_dequant_rect_simd4_body {
    ($backend:ty, $n:expr, $w:expr, $h:expr, $coeff_ty:ty, $coeff:expr, $tmp:expr, $eob:expr, $tx:expr, $is_rect2:expr, $shift0:expr, $row_clip_min:expr, $row_clip_max:expr $(,)?) => {{
        $crate::itx_2d::idct_dequant_rows_rect_dct_simd4::<$backend, $n, $w, $h, $coeff_ty>(
            $coeff,
            $tmp,
            $eob,
            $tx,
            $is_rect2,
            $shift0,
            $row_clip_min,
            $row_clip_max,
        );
        $crate::itx_2d::rect_col_pass::<$backend, $w, $h, $coeff_ty>($tmp);
    }};
}

#[macro_export]
macro_rules! itx_kind_dequant_rect_simd4_mono_body {
    ($backend:ty, $n:expr, $w:expr, $h:expr, $coeff_ty:ty, $first_kind:expr, $second_kind:expr, $coeff:expr, $tmp:expr, $eob:expr, $tx:expr, $is_rect2:expr, $shift0:expr, $row_clip_min:expr, $row_clip_max:expr $(,)?) => {{
        $crate::itx_2d::itx_dequant_rows_rect_simd4::<$backend, $n, $w, $h, $coeff_ty, $first_kind>(
            $coeff,
            $tmp,
            $eob,
            $tx,
            $is_rect2,
            $shift0,
            $row_clip_min,
            $row_clip_max,
        );

        let mut x = 0usize;
        if <$coeff_ty as $crate::itx_2d::ItxCoeff>::USE_WIDE_16BIT && ($h == 8 || $h == 16) {
            while x + 8 <= $w {
                if !$crate::itx_2d::itx_1d_wide_x8::<$backend, $h, $second_kind>($tmp, x) {
                    break;
                }
                x += 8;
            }
        }
        while x + 4 <= $w {
            if !(<$coeff_ty as $crate::itx_2d::ItxCoeff>::USE_WIDE_16BIT
                && $crate::itx_2d::itx_1d_wide_x4::<$backend, $h, $second_kind>($tmp, x))
            {
                $crate::itx_2d::itx_1d_x4::<$backend, $h, $second_kind>($tmp, x);
            }
            x += 4;
        }
        while x < $w {
            $crate::itx_2d::tx_1d_scalar_mono::<$h, $second_kind>(
                &mut $tmp[x..],
                $crate::itx_2d::ITX_TMP_STRIDE,
            );
            x += 1;
        }
    }};
}

#[macro_export]
macro_rules! itx_kind_dequant_rect_simd4_body {
    ($backend:ty, $n:expr, $w:expr, $h:expr, $coeff_ty:ty, $coeff:expr, $tmp:expr, $eob:expr, $tx:expr, $is_rect2:expr, $shift0:expr, $row_clip_min:expr, $row_clip_max:expr, $first_kind:expr, $second_kind:expr $(,)?) => {{
        match ($first_kind, $second_kind) {
            ($crate::itx_2d::TX_KIND_DCT, $crate::itx_2d::TX_KIND_DCT) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_DCT },
                    { $crate::itx_2d::TX_KIND_DCT },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_DCT, $crate::itx_2d::TX_KIND_ADST) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_DCT },
                    { $crate::itx_2d::TX_KIND_ADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_DCT, $crate::itx_2d::TX_KIND_FLIPADST) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_DCT },
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_ADST, $crate::itx_2d::TX_KIND_DCT) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_ADST },
                    { $crate::itx_2d::TX_KIND_DCT },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_ADST, $crate::itx_2d::TX_KIND_ADST) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_ADST },
                    { $crate::itx_2d::TX_KIND_ADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_ADST, $crate::itx_2d::TX_KIND_FLIPADST) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_ADST },
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_FLIPADST, $crate::itx_2d::TX_KIND_DCT) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    { $crate::itx_2d::TX_KIND_DCT },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_FLIPADST, $crate::itx_2d::TX_KIND_ADST) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    { $crate::itx_2d::TX_KIND_ADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            ($crate::itx_2d::TX_KIND_FLIPADST, $crate::itx_2d::TX_KIND_FLIPADST) => {
                $crate::itx_kind_dequant_rect_simd4_mono_body!(
                    $backend,
                    $n,
                    $w,
                    $h,
                    $coeff_ty,
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    { $crate::itx_2d::TX_KIND_FLIPADST },
                    $coeff,
                    $tmp,
                    $eob,
                    $tx,
                    $is_rect2,
                    $shift0,
                    $row_clip_min,
                    $row_clip_max
                )
            }
            _ => unreachable!(),
        }
    }};
}

pub(crate) type IdctDequantFn<const N: usize> = unsafe fn(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
);

pub(crate) type IadstDequantFn<const N: usize> = unsafe fn(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
);

pub(crate) type IdctDequantI16Fn<const N: usize> = unsafe fn(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
);

pub(crate) type IadstDequantI16Fn<const N: usize> = unsafe fn(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
);

pub(crate) const TX_KIND_DCT: usize = 0;
pub(crate) const TX_KIND_IDENTITY: usize = 1;
pub(crate) const TX_KIND_ADST: usize = 2;
pub(crate) const TX_KIND_FLIPADST: usize = 3;

#[inline(always)]
pub(crate) fn is_dct_adst_kind(kind: usize) -> bool {
    matches!(kind, TX_KIND_DCT | TX_KIND_ADST | TX_KIND_FLIPADST)
}

#[inline(always)]
pub(crate) fn is_itx_dense_kind(kind: usize) -> bool {
    matches!(
        kind,
        TX_KIND_DCT | TX_KIND_IDENTITY | TX_KIND_ADST | TX_KIND_FLIPADST
    )
}

macro_rules! dispatch_dct_adst_pair {
    ($first:expr, $second:expr, |$fk:ident, $sk:ident| $body:expr) => {{
        match ($first, $second) {
            (TX_KIND_DCT, TX_KIND_DCT) => {
                const $fk: usize = TX_KIND_DCT;
                const $sk: usize = TX_KIND_DCT;
                $body
            }
            (TX_KIND_DCT, TX_KIND_ADST) => {
                const $fk: usize = TX_KIND_DCT;
                const $sk: usize = TX_KIND_ADST;
                $body
            }
            (TX_KIND_DCT, TX_KIND_FLIPADST) => {
                const $fk: usize = TX_KIND_DCT;
                const $sk: usize = TX_KIND_FLIPADST;
                $body
            }
            (TX_KIND_ADST, TX_KIND_DCT) => {
                const $fk: usize = TX_KIND_ADST;
                const $sk: usize = TX_KIND_DCT;
                $body
            }
            (TX_KIND_ADST, TX_KIND_ADST) => {
                const $fk: usize = TX_KIND_ADST;
                const $sk: usize = TX_KIND_ADST;
                $body
            }
            (TX_KIND_ADST, TX_KIND_FLIPADST) => {
                const $fk: usize = TX_KIND_ADST;
                const $sk: usize = TX_KIND_FLIPADST;
                $body
            }
            (TX_KIND_FLIPADST, TX_KIND_DCT) => {
                const $fk: usize = TX_KIND_FLIPADST;
                const $sk: usize = TX_KIND_DCT;
                $body
            }
            (TX_KIND_FLIPADST, TX_KIND_ADST) => {
                const $fk: usize = TX_KIND_FLIPADST;
                const $sk: usize = TX_KIND_ADST;
                $body
            }
            (TX_KIND_FLIPADST, TX_KIND_FLIPADST) => {
                const $fk: usize = TX_KIND_FLIPADST;
                const $sk: usize = TX_KIND_FLIPADST;
                $body
            }
            _ => unreachable!(),
        }
    }};
}

pub(crate) trait Dct2dBackend {
    fn idct_dequant_4x4(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    );

    fn idct_dequant_8x8(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    );

    fn idct_dequant_16x16(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    );

    fn idct_dequant_32x32(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    );

    fn idct_dequant_64x64(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    );
}

pub(crate) trait Adst2dBackend {
    fn iadst_dequant_4x4(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
        first_kind: usize,
        second_kind: usize,
    );

    fn iadst_dequant_8x8(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
        first_kind: usize,
        second_kind: usize,
    );

    fn iadst_dequant_16x16(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
        first_kind: usize,
        second_kind: usize,
    );
}

pub(crate) struct ScalarDct2d;

impl Dct2dBackend for ScalarDct2d {
    #[inline(always)]
    fn idct_dequant_4x4(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_scalar_core::<16, 4, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_8x8(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_scalar_core::<64, 8, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_16x16(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_scalar_core::<256, 16, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_32x32(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_scalar_core::<1024, 32, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_64x64(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        // 64x64 DCT uses the same active 32x32 core; the caller expands the
        // residual during add, exactly like the generic path.
        idct_dequant_scalar_core::<1024, 32, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }
}

impl Adst2dBackend for ScalarDct2d {
    #[inline(always)]
    fn iadst_dequant_4x4(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
        first_kind: usize,
        second_kind: usize,
    ) {
        itx_dequant_scalar_core::<16, 4, i32>(
            coeff,
            tmp,
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

    #[inline(always)]
    fn iadst_dequant_8x8(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
        first_kind: usize,
        second_kind: usize,
    ) {
        itx_dequant_scalar_core::<64, 8, i32>(
            coeff,
            tmp,
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

    #[inline(always)]
    fn iadst_dequant_16x16(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
        first_kind: usize,
        second_kind: usize,
    ) {
        itx_dequant_scalar_core::<256, 16, i32>(
            coeff,
            tmp,
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
}

#[inline(always)]
pub(crate) fn row_mut(tmp: &mut [i32; ITX_TMP_PIXELS], y: usize) -> &mut [i32; ITX_TMP_STRIDE] {
    (&mut tmp[y * ITX_TMP_STRIDE..(y + 1) * ITX_TMP_STRIDE])
        .try_into()
        .unwrap()
}

#[inline(always)]
pub(crate) fn dct_1d<const S: usize>(c: &mut [i32], stride: usize) {
    match S {
        4 => inv_dct4_1d(c, stride),
        8 => inv_dct8_1d(c, stride),
        16 => inv_dct16_1d(c, stride),
        32 => inv_dct32_1d(c, stride),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn dct_1d_x8<const S: usize>(tmp: &mut [i32; ITX_TMP_PIXELS], x: usize) -> bool {
    let tx_size = match S {
        8 => 1,
        16 => 2,
        32 => 3,
        _ => return false,
    };

    if let Some(f8) = crate::itx_1d::tx1d_x8_dispatch(tx_size, 0) {
        unsafe { f8(tmp, x, ITX_TMP_STRIDE) };
        true
    } else {
        false
    }
}

pub(crate) fn idct_dequant_scalar_core<const N: usize, const S: usize, C: Coeff>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(S == 4 || S == 8 || S == 16 || S == 32);
    debug_assert!(N <= coeff.len());
    debug_assert!(S * S <= N);

    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        let dst_row = row_mut(tmp, y);
        for (x, dst) in dst_row[..S].iter_mut().enumerate() {
            let v = coeff[y + x * S].to_i32();
            *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }

        dct_1d::<S>(dst_row, 1);
        y += 1;

        if y & 3 == 0 {
            if eob > last_eob[ei] as i32 {
                ei += 1;
            } else {
                break;
            }
        }
    }

    while y < S {
        row_mut(tmp, y)[..S].fill(0);
        y += 1;
    }

    coeff[..S * S].fill(C::ZERO);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..S {
        crate::filter::row_clip(row_mut(tmp, y), S, rnd0, shift0, row_clip_min, row_clip_max);
    }

    let mut x = 0usize;
    while x + 8 <= S {
        if !dct_1d_x8::<S>(tmp, x) {
            for sx in x..x + 8 {
                dct_1d::<S>(&mut tmp[sx..], ITX_TMP_STRIDE);
            }
        }
        x += 8;
    }
    while x < S {
        dct_1d::<S>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

#[inline(always)]
fn tx_size_idx<const S: usize>() -> usize {
    match S {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        _ => unreachable!(),
    }
}

#[inline(always)]
pub(crate) fn tx_1d_scalar_mono<const S: usize, const KIND: usize>(c: &mut [i32], stride: usize) {
    debug_assert!(is_dct_adst_kind(KIND));
    let f = TX1D_FNS[tx_size_idx::<S>()][KIND].expect("unsupported 1D transform");
    f(c, stride);
}

pub(crate) fn itx_dequant_scalar_core_mono<
    const N: usize,
    const S: usize,
    C: Coeff,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(S == 4 || S == 8 || S == 16);
    debug_assert!(N <= coeff.len());
    debug_assert!(S * S <= N);
    debug_assert!(is_dct_adst_kind(FIRST_KIND));
    debug_assert!(is_dct_adst_kind(SECOND_KIND));

    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        let dst_row = row_mut(tmp, y);
        for (x, dst) in dst_row[..S].iter_mut().enumerate() {
            let v = coeff[y + x * S].to_i32();
            *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }

        tx_1d_scalar_mono::<S, FIRST_KIND>(dst_row, 1);
        y += 1;

        if y & 3 == 0 {
            if eob > last_eob[ei] as i32 {
                ei += 1;
            } else {
                break;
            }
        }
    }

    while y < S {
        row_mut(tmp, y)[..S].fill(0);
        y += 1;
    }

    coeff[..S * S].fill(C::ZERO);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..S {
        crate::filter::row_clip(row_mut(tmp, y), S, rnd0, shift0, row_clip_min, row_clip_max);
    }

    let mut x = 0usize;
    if let Some(f8) = crate::itx_1d::tx1d_x8_dispatch(tx_size_idx::<S>(), SECOND_KIND) {
        while x + 8 <= S {
            unsafe { f8(tmp, x, ITX_TMP_STRIDE) };
            x += 8;
        }
    }
    while x < S {
        tx_1d_scalar_mono::<S, SECOND_KIND>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

pub(crate) fn itx_dequant_scalar_core<const N: usize, const S: usize, C: Coeff>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(is_dct_adst_kind(first_kind));
    debug_assert!(is_dct_adst_kind(second_kind));
    dispatch_dct_adst_pair!(first_kind, second_kind, |FK, SK| {
        itx_dequant_scalar_core_mono::<N, S, C, FK, SK>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    });
}

#[inline(always)]
fn clip_row_value(v: i32, rnd: i32, shift: i32, min: i32, max: i32) -> i32 {
    ((v + rnd) >> shift).max(min).min(max)
}

#[inline(always)]
fn even4<T: Copy>(v: &[T; 8]) -> [T; 4] {
    [v[0], v[2], v[4], v[6]]
}

#[inline(always)]
fn odd4<T: Copy>(v: &[T; 8]) -> [T; 4] {
    [v[1], v[3], v[5], v[7]]
}

/// Full size-16 inverse DCT-II kernel `K16[in*16 + out]` for the flat butterfly.
#[rustfmt::skip]
pub(crate) static DCT16_DENSE_KERNEL: [i32; 256] = [
      64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
      90,  87,  80,  70,  57,  43,  26,   9,  -9, -26, -43, -57, -70, -80, -87, -90,
      89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,
      87,  57,   9, -43, -80, -90, -70, -26,  26,  70,  90,  80,  43,  -9, -57, -87,
      83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,
      80,   9, -70, -87, -26,  57,  90,  43, -43, -90, -57,  26,  87,  70,  -9, -80,
      75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,
      70, -43, -87,   9,  90,  26, -80, -57,  57,  80, -26, -90,  -9,  87,  43, -70,
      64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,
      57, -80, -26,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  26,  80, -57,
      50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,
      43, -90,  57,  26, -87,  70,   9, -80,  80,  -9, -70,  87, -26, -57,  90, -43,
      35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,
      26, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -26,
      18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,
       9, -26,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  26,  -9,
];

// Output-major re-layout of DCT16_DENSE_KERNEL for dct16_flat_bylane (cf. DCT32_K*).
pub(crate) static DCT16_KB: [i32; 64] = [
    90, 87, 80, 70, 57, 43, 26, 9, 87, 57, 9, -43, -80, -90, -70, -26, 80, 9, -70, -87, -26, 57,
    90, 43, 70, -43, -87, 9, 90, 26, -80, -57, 57, -80, -26, 90, -9, -87, 43, 70, 43, -90, 57, 26,
    -87, 70, 9, -80, 26, -70, 90, -80, 43, 9, -57, 87, 9, -26, 43, -57, 70, -80, 87, -90,
];
pub(crate) static DCT16_KD: [i32; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];
pub(crate) static DCT16_KF: [i32; 4] = [83, 35, 35, -83];
pub(crate) static DCT16_KG: [i32; 4] = [64, 64, 64, -64];

// i16 wide tables (s16 8-wide widening-MAC). 8-lane groups, small leaves zero-padded.
pub(crate) static DCT32_KBW: [i16; 256] = [
    90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4, 90, 82, 67, 47, 22, -4, -30,
    -54, -73, -85, -90, -88, -78, -61, -39, -13, 88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39,
    73, 90, 85, 61, 22, 85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30,
    82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67, 4, 73, 88, 39, 78, -4, -82, -73, 13, 85,
    67, -22, -88, -61, 30, 90, 54, -39, -90, -47, 73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61,
    -47, -88, -4, 85, 54, 67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47, -73, -61,
    61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67, 54, -85, -4, 88, -47,
    -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73, 47, -90, 39, 54, -90, 30, 61, -88, 22, 67,
    -85, 13, 73, -82, 4, 78, 39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82,
    30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85, 22, -61, 85, -90, 73,
    -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88, 13, -39, 61, -78, 88, -90, 85, -73, 54, -30,
    4, 22, -47, 67, -82, 90, 4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
];
pub(crate) static DCT32_KDW: [i16; 64] = [
    90, 87, 80, 70, 57, 43, 26, 9, 87, 57, 9, -43, -80, -90, -70, -26, 80, 9, -70, -87, -26, 57,
    90, 43, 70, -43, -87, 9, 90, 26, -80, -57, 57, -80, -26, 90, -9, -87, 43, 70, 43, -90, 57, 26,
    -87, 70, 9, -80, 26, -70, 90, -80, 43, 9, -57, 87, 9, -26, 43, -57, 70, -80, 87, -90,
];
pub(crate) static DCT32_KFW: [i16; 32] = [
    89, 75, 50, 18, 0, 0, 0, 0, 75, -18, -89, -50, 0, 0, 0, 0, 50, -89, 18, 75, 0, 0, 0, 0, 18,
    -50, 75, -89, 0, 0, 0, 0,
];
pub(crate) static DCT32_KHW: [i16; 8] = [83, 35, 35, -83, 0, 0, 0, 0];
pub(crate) static DCT32_KGW: [i16; 8] = [64, 64, 64, -64, 0, 0, 0, 0];
pub(crate) static DCT16_KBW: [i16; 64] = [
    90, 87, 80, 70, 57, 43, 26, 9, 87, 57, 9, -43, -80, -90, -70, -26, 80, 9, -70, -87, -26, 57,
    90, 43, 70, -43, -87, 9, 90, 26, -80, -57, 57, -80, -26, 90, -9, -87, 43, 70, 43, -90, 57, 26,
    -87, 70, 9, -80, 26, -70, 90, -80, 43, 9, -57, 87, 9, -26, 43, -57, 70, -80, 87, -90,
];
pub(crate) static DCT16_KDW: [i16; 32] = [
    89, 75, 50, 18, 0, 0, 0, 0, 75, -18, -89, -50, 0, 0, 0, 0, 50, -89, 18, 75, 0, 0, 0, 0, 18,
    -50, 75, -89, 0, 0, 0, 0,
];
pub(crate) static DCT16_KFW: [i16; 8] = [83, 35, 35, -83, 0, 0, 0, 0];
pub(crate) static DCT16_KGW: [i16; 8] = [64, 64, 64, -64, 0, 0, 0, 0];

// i16 ADST/FLIPADST kernels for the s16 8-wide widening-MAC dense matmul.
// Output-major; n=4 rows zero-padded to an 8-lane group.
pub(crate) static DCT4_KW: [i16; 32] = [
    64, 83, 64, 35, 0, 0, 0, 0, 64, 35, -64, -83, 0, 0, 0, 0, 64, -35, -64, 83, 0, 0, 0, 0, 64,
    -83, 64, -35, 0, 0, 0, 0,
];
pub(crate) static ADST4_KW: [i16; 32] = [
    18, 50, 75, 89, 0, 0, 0, 0, 50, 89, 18, -75, 0, 0, 0, 0, 75, 18, -89, 50, 0, 0, 0, 0, 89, -75,
    50, -18, 0, 0, 0, 0,
];
pub(crate) static FLIPADST4_KW: [i16; 32] = [
    89, 75, 50, 18, 0, 0, 0, 0, 75, -18, -89, -50, 0, 0, 0, 0, 50, -89, 18, 75, 0, 0, 0, 0, 18,
    -50, 75, -89, 0, 0, 0, 0,
];
// Dense DCT8 matrix (output-major), bit-exact to the factored `inv_dct8`;
// lets the 8-point DCT dimension of an ADST block use the wide path.
pub(crate) static DCT8_KW: [i16; 64] = [
    64, 89, 83, 75, 64, 50, 35, 18, 64, 75, 35, -18, -64, -89, -83, -50, 64, 50, -35, -89, -64, 18,
    83, 75, 64, 18, -83, -50, 64, 75, -35, -89, 64, -18, -83, 50, 64, -75, -35, 89, 64, -50, -35,
    89, -64, -18, 83, -75, 64, -75, 35, 18, -64, 89, -83, 50, 64, -89, 83, -75, 64, -50, 35, -18,
];
pub(crate) static ADST8_KW: [i16; 64] = [
    11, 34, 54, 71, 84, 88, 79, 50, 28, 74, 89, 68, 17, -44, -83, -69, 44, 89, 48, -41, -89, -44,
    50, 81, 58, 76, -34, -86, 10, 88, 6, -84, 70, 39, -87, 1, 86, -44, -59, 78, 79, -12, -66, 87,
    -35, -44, 86, -62, 86, -58, 12, 38, -75, 88, -74, 40, 89, -86, 79, -70, 58, -44, 29, -14,
];
pub(crate) static ADST16_KW: [i16; 256] = [
    8, 25, 41, 55, 67, 77, 84, 88, 89, 87, 81, 73, 62, 48, 33, 17, 17, 48, 73, 87, 88, 77, 55, 25,
    -8, -41, -67, -84, -89, -81, -62, -33, 25, 67, 88, 81, 48, 0, -48, -81, -88, -67, -25, 25, 67,
    88, 81, 48, 33, 81, 84, 41, -25, -77, -87, -48, 17, 73, 88, 55, -8, -67, -89, -62, 41, 88, 62,
    -17, -81, -77, -8, 67, 87, 33, -48, -89, -55, 25, 84, 73, 48, 88, 25, -67, -81, 0, 81, 67, -25,
    -88, -48, 48, 88, 25, -67, -81, 55, 81, -17, -89, -25, 77, 62, -48, -84, 8, 88, 33, -73, -67,
    41, 87, 62, 67, -55, -73, 48, 77, -41, -81, 33, 84, -25, -87, 17, 88, -8, -89, 67, 48, -81,
    -25, 88, 0, -88, 25, 81, -48, -67, 67, 48, -81, -25, 88, 73, 25, -89, 33, 67, -77, -17, 88,
    -41, -62, 81, 8, -87, 48, 55, -84, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77,
    77, 81, -25, -48, 88, -67, 0, 67, -88, 48, 25, -81, 81, -25, -48, 88, -67, 84, -48, -8, 62,
    -88, 77, -33, -25, 73, -89, 67, -17, -41, 81, -87, 55, 87, -67, 33, 8, -48, 77, -89, 81, -55,
    17, 25, -62, 84, -88, 73, -41, 88, -81, 67, -48, 25, 0, -25, 48, -67, 81, -88, 88, -81, 67,
    -48, 25, 89, -88, 87, -84, 81, -77, 73, -67, 62, -55, 48, -41, 33, -25, 17, -8,
];
pub(crate) static FLIPADST16_KW: [i16; 256] = [
    89, 88, 87, 84, 81, 77, 73, 67, 62, 55, 48, 41, 33, 25, 17, 8, 88, 81, 67, 48, 25, 0, -25, -48,
    -67, -81, -88, -88, -81, -67, -48, -25, 87, 67, 33, -8, -48, -77, -89, -81, -55, -17, 25, 62,
    84, 88, 73, 41, 84, 48, -8, -62, -88, -77, -33, 25, 73, 89, 67, 17, -41, -81, -87, -55, 81, 25,
    -48, -88, -67, 0, 67, 88, 48, -25, -81, -81, -25, 48, 88, 67, 77, 0, -77, -77, 0, 77, 77, 0,
    -77, -77, 0, 77, 77, 0, -77, -77, 73, -25, -89, -33, 67, 77, -17, -88, -41, 62, 81, -8, -87,
    -48, 55, 84, 67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67, -67, 48, 81, -25, -88, 62, -67,
    -55, 73, 48, -77, -41, 81, 33, -84, -25, 87, 17, -88, -8, 89, 55, -81, -17, 89, -25, -77, 62,
    48, -84, -8, 88, -33, -73, 67, 41, -87, 48, -88, 25, 67, -81, 0, 81, -67, -25, 88, -48, -48,
    88, -25, -67, 81, 41, -88, 62, 17, -81, 77, -8, -67, 87, -33, -48, 89, -55, -25, 84, -73, 33,
    -81, 84, -41, -25, 77, -87, 48, 17, -73, 88, -55, -8, 67, -89, 62, 25, -67, 88, -81, 48, 0,
    -48, 81, -88, 67, -25, -25, 67, -88, 81, -48, 17, -48, 73, -87, 88, -77, 55, -25, -8, 41, -67,
    84, -89, 81, -62, 33, 8, -25, 41, -55, 67, -77, 84, -88, 89, -87, 81, -73, 62, -48, 33, -17,
];

pub(crate) static DCT32_KBP_X4: [i32; 512] = [
    5898330, 5898330, 5898330, 5898330, 5570648, 5570648, 5570648, 5570648, 5111890, 5111890,
    5111890, 5111890, 4390985, 4390985, 4390985, 4390985, 3539005, 3539005, 3539005, 3539005,
    2555951, 2555951, 2555951, 2555951, 1441822, 1441822, 1441822, 1441822, 262157, 262157, 262157,
    262157, 5374042, 5374042, 5374042, 5374042, 3080259, 3080259, 3080259, 3080259, -262122,
    -262122, -262122, -262122, -3473438, -3473438, -3473438, -3473438, -5505097, -5505097,
    -5505097, -5505097, -5701722, -5701722, -5701722, -5701722, -3932238, -3932238, -3932238,
    -3932238, -786471, -786471, -786471, -786471, 4391000, 4391000, 4391000, 4391000, -851938,
    -851938, -851938, -851938, -5308470, -5308470, -5308470, -5308470, -5046362, -5046362,
    -5046362, -5046362, -196655, -196655, -196655, -196655, 4784167, 4784167, 4784167, 4784167,
    5570650, 5570650, 5570650, 5570650, 1441853, 1441853, 1441853, 1441853, 3080277, 3080277,
    3080277, 3080277, -4325389, -4325389, -4325389, -4325389, -4718682, -4718682, -4718682,
    -4718682, 2621418, 2621418, 2621418, 2621418, 5767250, 5767250, 5767250, 5767250, -262090,
    -262090, -262090, -262090, -5832765, -5832765, -5832765, -5832765, -1900622, -1900622,
    -1900622, -1900622, 1441874, 1441874, 1441874, 1441874, -5832758, -5832758, -5832758, -5832758,
    917443, 917443, 917443, 917443, 5570638, 5570638, 5570638, 5570638, -3080162, -3080162,
    -3080162, -3080162, -4325466, -4325466, -4325466, -4325466, 4784132, 4784132, 4784132, 4784132,
    2555992, 2555992, 2555992, 2555992, -262066, -262066, -262066, -262066, -4718674, -4718674,
    -4718674, -4718674, 5570573, 5570573, 5570573, 5570573, -1441725, -1441725, -1441725, -1441725,
    -3932248, -3932248, -3932248, -3932248, 5898270, 5898270, 5898270, 5898270, -2555850, -2555850,
    -2555850, -2555850, -3014746, -3014746, -3014746, -3014746, -1966007, -1966007, -1966007,
    -1966007, -1376346, -1376346, -1376346, -1376346, 4390990, 4390990, 4390990, 4390990, -5832743,
    -5832743, -5832743, -5832743, 5439475, 5439475, 5439475, 5439475, -3080131, -3080131, -3080131,
    -3080131, -196696, -196696, -196696, -196696, 3539029, 3539029, 3539029, 3539029, -3538877,
    -3538877, -3538877, -3538877, 2621362, 2621362, 2621362, 2621362, -1441707, -1441707, -1441707,
    -1441707, 327590, 327590, 327590, 327590, 852058, 852058, 852058, 852058, -1900632, -1900632,
    -1900632, -1900632, 3080274, 3080274, 3080274, 3080274, -3932233, -3932233, -3932233, -3932233,
    -4784067, -4784067, -4784067, -4784067, 5439441, 5439441, 5439441, 5439441, -5767138, -5767138,
    -5767138, -5767138, 5963763, 5963763, 5963763, 5963763, -5832708, -5832708, -5832708, -5832708,
    5570582, 5570582, 5570582, 5570582, -5046311, -5046311, -5046311, -5046311, 4390966, 4390966,
    4390966, 4390966, -5570506, -5570506, -5570506, -5570506, 5832700, 5832700, 5832700, 5832700,
    -3932207, -3932207, -3932207, -3932207, 852050, 852050, 852050, 852050, 2621350, 2621350,
    2621350, 2621350, -5111741, -5111741, -5111741, -5111741, 5963754, 5963754, 5963754, 5963754,
    -4718622, -4718622, -4718622, -4718622, -5898193, -5898193, -5898193, -5898193, 3538983,
    3538983, 3538983, 3538983, 2031526, 2031526, 2031526, 2031526, -5767107, -5767107, -5767107,
    -5767107, 4390934, 4390934, 4390934, 4390934, 917419, 917419, 917419, 917419, -5373879,
    -5373879, -5373879, -5373879, 5111812, 5111812, 5111812, 5111812, -5767129, -5767129, -5767129,
    -5767129, -262071, -262071, -262071, -262071, 5963709, 5963709, 5963709, 5963709, -1900591,
    -1900591, -1900591, -1900591, -5111723, -5111723, -5111723, -5111723, 3997709, 3997709,
    3997709, 3997709, 3604390, 3604390, 3604390, 3604390, -5373930, -5373930, -5373930, -5373930,
    -5111778, -5111778, -5111778, -5111778, -3997606, -3997606, -3997606, -3997606, 3538948,
    3538948, 3538948, 3538948, 5439400, 5439400, 5439400, 5439400, -1376295, -1376295, -1376295,
    -1376295, -5898167, -5898167, -5898167, -5898167, -851901, -851901, -851901, -851901, 5636049,
    5636049, 5636049, 5636049, -3997674, -3997674, -3997674, -3997674, -5898155, -5898155,
    -5898155, -5898155, -2555831, -2555831, -2555831, -2555831, 3145724, 3145724, 3145724, 3145724,
    5963698, 5963698, 5963698, 5963698, 3604398, 3604398, 3604398, 3604398, -1900557, -1900557,
    -1900557, -1900557, -5767101, -5767101, -5767101, -5767101, -2555891, -2555891, -2555891,
    -2555891, -5111747, -5111747, -5111747, -5111747, -5898152, -5898152, -5898152, -5898152,
    -4784043, -4784043, -4784043, -4784043, -1966026, -1966026, -1966026, -1966026, 1441796,
    1441796, 1441796, 1441796, 4456401, 4456401, 4456401, 4456401, 5963694, 5963694, 5963694,
    5963694, -851964, -851964, -851964, -851964, -1966058, -1966058, -1966058, -1966058, -3080153,
    -3080153, -3080153, -3080153, -3997642, -3997642, -3997642, -3997642, -4784061, -4784061,
    -4784061, -4784061, -5373874, -5373874, -5373874, -5373874, -5767083, -5767083, -5767083,
    -5767083, -5898150, -5898150, -5898150, -5898150,
];

pub(crate) static DCT32_KDP_X4: [i32; 128] = [
    5701722, 5701722, 5701722, 5701722, 4587600, 4587600, 4587600, 4587600, 2818105, 2818105,
    2818105, 2818105, 589850, 589850, 589850, 589850, 3735639, 3735639, 3735639, 3735639, -2818039,
    -2818039, -2818039, -2818039, -5832784, -5832784, -5832784, -5832784, -1638470, -1638470,
    -1638470, -1638470, 589904, 589904, 589904, 589904, -5636166, -5636166, -5636166, -5636166,
    3801062, 3801062, 3801062, 3801062, 2818138, 2818138, 2818138, 2818138, -2817978, -2817978,
    -2817978, -2817978, 655273, 655273, 655273, 655273, 1704026, 1704026, 1704026, 1704026,
    -3670096, -3670096, -3670096, -3670096, -5242823, -5242823, -5242823, -5242823, 5963750,
    5963750, 5963750, 5963750, -5636105, -5636105, -5636105, -5636105, 4587563, 4587563, 4587563,
    4587563, -5898197, -5898197, -5898197, -5898197, 1703993, 1703993, 1703993, 1703993, 4652969,
    4652969, 4652969, 4652969, -5242871, -5242871, -5242871, -5242871, -4587494, -4587494,
    -4587494, -4587494, -5242790, -5242790, -5242790, -5242790, 589867, 589867, 589867, 589867,
    5767111, 5767111, 5767111, 5767111, -1703927, -1703927, -1703927, -1703927, -3735509, -3735509,
    -3735509, -3735509, -5242810, -5242810, -5242810, -5242810, -5898153, -5898153, -5898153,
    -5898153,
];

pub(crate) static DCT32_KFP_X4: [i32; 64] = [
    4915289, 4915289, 4915289, 4915289, 1179698, 1179698, 1179698, 1179698, 0, 0, 0, 0, 0, 0, 0, 0,
    -1179573, -1179573, -1179573, -1179573, -3211353, -3211353, -3211353, -3211353, 0, 0, 0, 0, 0,
    0, 0, 0, -5832654, -5832654, -5832654, -5832654, 4915218, 4915218, 4915218, 4915218, 0, 0, 0,
    0, 0, 0, 0, 0, -3276782, -3276782, -3276782, -3276782, -5832629, -5832629, -5832629, -5832629,
    0, 0, 0, 0, 0, 0, 0, 0,
];

pub(crate) static DCT32_KHP_X4: [i32; 16] = [
    2293843, 2293843, 2293843, 2293843, -5439453, -5439453, -5439453, -5439453, 0, 0, 0, 0, 0, 0,
    0, 0,
];

pub(crate) static DCT32_KGP_X4: [i32; 16] = [
    4194368, 4194368, 4194368, 4194368, -4194240, -4194240, -4194240, -4194240, 0, 0, 0, 0, 0, 0,
    0, 0,
];

pub(crate) static DCT16_KBP_X4: [i32; 128] = [
    5701722, 5701722, 5701722, 5701722, 4587600, 4587600, 4587600, 4587600, 2818105, 2818105,
    2818105, 2818105, 589850, 589850, 589850, 589850, 3735639, 3735639, 3735639, 3735639, -2818039,
    -2818039, -2818039, -2818039, -5832784, -5832784, -5832784, -5832784, -1638470, -1638470,
    -1638470, -1638470, 589904, 589904, 589904, 589904, -5636166, -5636166, -5636166, -5636166,
    3801062, 3801062, 3801062, 3801062, 2818138, 2818138, 2818138, 2818138, -2817978, -2817978,
    -2817978, -2817978, 655273, 655273, 655273, 655273, 1704026, 1704026, 1704026, 1704026,
    -3670096, -3670096, -3670096, -3670096, -5242823, -5242823, -5242823, -5242823, 5963750,
    5963750, 5963750, 5963750, -5636105, -5636105, -5636105, -5636105, 4587563, 4587563, 4587563,
    4587563, -5898197, -5898197, -5898197, -5898197, 1703993, 1703993, 1703993, 1703993, 4652969,
    4652969, 4652969, 4652969, -5242871, -5242871, -5242871, -5242871, -4587494, -4587494,
    -4587494, -4587494, -5242790, -5242790, -5242790, -5242790, 589867, 589867, 589867, 589867,
    5767111, 5767111, 5767111, 5767111, -1703927, -1703927, -1703927, -1703927, -3735509, -3735509,
    -3735509, -3735509, -5242810, -5242810, -5242810, -5242810, -5898153, -5898153, -5898153,
    -5898153,
];

pub(crate) static DCT16_KDP_X4: [i32; 64] = [
    4915289, 4915289, 4915289, 4915289, 1179698, 1179698, 1179698, 1179698, 0, 0, 0, 0, 0, 0, 0, 0,
    -1179573, -1179573, -1179573, -1179573, -3211353, -3211353, -3211353, -3211353, 0, 0, 0, 0, 0,
    0, 0, 0, -5832654, -5832654, -5832654, -5832654, 4915218, 4915218, 4915218, 4915218, 0, 0, 0,
    0, 0, 0, 0, 0, -3276782, -3276782, -3276782, -3276782, -5832629, -5832629, -5832629, -5832629,
    0, 0, 0, 0, 0, 0, 0, 0,
];

pub(crate) static DCT16_KFP_X4: [i32; 16] = [
    2293843, 2293843, 2293843, 2293843, -5439453, -5439453, -5439453, -5439453, 0, 0, 0, 0, 0, 0,
    0, 0,
];

pub(crate) static DCT16_KGP_X4: [i32; 16] = [
    4194368, 4194368, 4194368, 4194368, -4194240, -4194240, -4194240, -4194240, 0, 0, 0, 0, 0, 0,
    0, 0,
];

pub(crate) static DCT4_KP_X4: [i32; 64] = [
    5439552, 5439552, 5439552, 5439552, 2293824, 2293824, 2293824, 2293824, 0, 0, 0, 0, 0, 0, 0, 0,
    2293824, 2293824, 2293824, 2293824, -5374016, -5374016, -5374016, -5374016, 0, 0, 0, 0, 0, 0,
    0, 0, -2293696, -2293696, -2293696, -2293696, 5504960, 5504960, 5504960, 5504960, 0, 0, 0, 0,
    0, 0, 0, 0, -5439424, -5439424, -5439424, -5439424, -2293696, -2293696, -2293696, -2293696, 0,
    0, 0, 0, 0, 0, 0, 0,
];

pub(crate) static ADST4_KP_X4: [i32; 64] = [
    3276818, 3276818, 3276818, 3276818, 5832779, 5832779, 5832779, 5832779, 0, 0, 0, 0, 0, 0, 0, 0,
    5832754, 5832754, 5832754, 5832754, -4915182, -4915182, -4915182, -4915182, 0, 0, 0, 0, 0, 0,
    0, 0, 1179723, 1179723, 1179723, 1179723, 3342247, 3342247, 3342247, 3342247, 0, 0, 0, 0, 0, 0,
    0, 0, -4915111, -4915111, -4915111, -4915111, -1179598, -1179598, -1179598, -1179598, 0, 0, 0,
    0, 0, 0, 0, 0,
];

pub(crate) static FLIPADST4_KP_X4: [i32; 64] = [
    4915289, 4915289, 4915289, 4915289, 1179698, 1179698, 1179698, 1179698, 0, 0, 0, 0, 0, 0, 0, 0,
    -1179573, -1179573, -1179573, -1179573, -3211353, -3211353, -3211353, -3211353, 0, 0, 0, 0, 0,
    0, 0, 0, -5832654, -5832654, -5832654, -5832654, 4915218, 4915218, 4915218, 4915218, 0, 0, 0,
    0, 0, 0, 0, 0, -3276782, -3276782, -3276782, -3276782, -5832629, -5832629, -5832629, -5832629,
    0, 0, 0, 0, 0, 0, 0, 0,
];

pub(crate) static DCT8_KP_X4: [i32; 128] = [
    5832768, 5832768, 5832768, 5832768, 4915283, 4915283, 4915283, 4915283, 3276864, 3276864,
    3276864, 3276864, 1179683, 1179683, 1179683, 1179683, 4915264, 4915264, 4915264, 4915264,
    -1179613, -1179613, -1179613, -1179613, -5767232, -5767232, -5767232, -5767232, -3211347,
    -3211347, -3211347, -3211347, 3276864, 3276864, 3276864, 3276864, -5767203, -5767203, -5767203,
    -5767203, 1245120, 1245120, 1245120, 1245120, 4915283, 4915283, 4915283, 4915283, 1179712,
    1179712, 1179712, 1179712, -3211347, -3211347, -3211347, -3211347, 4915264, 4915264, 4915264,
    4915264, -5767203, -5767203, -5767203, -5767203, -1179584, -1179584, -1179584, -1179584,
    3342253, 3342253, 3342253, 3342253, -4915136, -4915136, -4915136, -4915136, 5898205, 5898205,
    5898205, 5898205, -3276736, -3276736, -3276736, -3276736, 5898205, 5898205, 5898205, 5898205,
    -1114176, -1114176, -1114176, -1114176, -4915117, -4915117, -4915117, -4915117, -4915136,
    -4915136, -4915136, -4915136, 1179683, 1179683, 1179683, 1179683, 5898176, 5898176, 5898176,
    5898176, 3342253, 3342253, 3342253, 3342253, -5832640, -5832640, -5832640, -5832640, -4915117,
    -4915117, -4915117, -4915117, -3276736, -3276736, -3276736, -3276736, -1179613, -1179613,
    -1179613, -1179613,
];

pub(crate) static ADST8_KP_X4: [i32; 128] = [
    2228235, 2228235, 2228235, 2228235, 4653110, 4653110, 4653110, 4653110, 5767252, 5767252,
    5767252, 5767252, 3276879, 3276879, 3276879, 3276879, 4849692, 4849692, 4849692, 4849692,
    4456537, 4456537, 4456537, 4456537, -2883567, -2883567, -2883567, -2883567, -4456531, -4456531,
    -4456531, -4456531, 5832748, 5832748, 5832748, 5832748, -2686928, -2686928, -2686928, -2686928,
    -2818137, -2818137, -2818137, -2818137, 5308466, 5308466, 5308466, 5308466, 4980794, 4980794,
    4980794, 4980794, -5570594, -5570594, -5570594, -5570594, 5767178, 5767178, 5767178, 5767178,
    -5505018, -5505018, -5505018, -5505018, 2555974, 2555974, 2555974, 2555974, 130985, 130985,
    130985, 130985, -2883498, -2883498, -2883498, -2883498, 5177285, 5177285, 5177285, 5177285,
    -786353, -786353, -786353, -786353, 5767102, 5767102, 5767102, 5767102, -2818083, -2818083,
    -2818083, -2818083, -4063146, -4063146, -4063146, -4063146, -3801002, -3801002, -3801002,
    -3801002, 2490380, 2490380, 2490380, 2490380, 5832629, 5832629, 5832629, 5832629, 2686902,
    2686902, 2686902, 2686902, -5636007, -5636007, -5636007, -5636007, -4587441, -4587441,
    -4587441, -4587441, -2883526, -2883526, -2883526, -2883526, -917475, -917475, -917475, -917475,
];

pub(crate) static ADST16_KP_X4: [i32; 512] = [
    1638408, 1638408, 1638408, 1638408, 3604521, 3604521, 3604521, 3604521, 5046339, 5046339,
    5046339, 5046339, 5767252, 5767252, 5767252, 5767252, 5701721, 5701721, 5701721, 5701721,
    4784209, 4784209, 4784209, 4784209, 3145790, 3145790, 3145790, 3145790, 1114145, 1114145,
    1114145, 1114145, 3145745, 3145745, 3145745, 3145745, 5701705, 5701705, 5701705, 5701705,
    5046360, 5046360, 5046360, 5046360, 1638455, 1638455, 1638455, 1638455, -2621448, -2621448,
    -2621448, -2621448, -5439555, -5439555, -5439555, -5439555, -5242969, -5242969, -5242969,
    -5242969, -2097214, -2097214, -2097214, -2097214, 4390937, 4390937, 4390937, 4390937, 5308504,
    5308504, 5308504, 5308504, 48, 48, 48, 48, -5242928, -5242928, -5242928, -5242928, -4325464,
    -4325464, -4325464, -4325464, 1703911, 1703911, 1703911, 1703911, 5767235, 5767235, 5767235,
    5767235, 3145809, 3145809, 3145809, 3145809, 5308449, 5308449, 5308449, 5308449, 2687060,
    2687060, 2687060, 2687060, -4980761, -4980761, -4980761, -4980761, -3080279, -3080279,
    -3080279, -3080279, 4784145, 4784145, 4784145, 4784145, 3604568, 3604568, 3604568, 3604568,
    -4325384, -4325384, -4325384, -4325384, -3997785, -3997785, -3997785, -3997785, 5767209,
    5767209, 5767209, 5767209, -1114050, -1114050, -1114050, -1114050, -4980817, -4980817,
    -4980817, -4980817, 4456440, 4456440, 4456440, 4456440, 2162775, 2162775, 2162775, 2162775,
    -5767216, -5767216, -5767216, -5767216, 1703881, 1703881, 1703881, 1703881, 4784212, 4784212,
    4784212, 4784212, 5767216, 5767216, 5767216, 5767216, -4390887, -4390887, -4390887, -4390887,
    65455, 65455, 65455, 65455, 4390993, 4390993, 4390993, 4390993, -5701657, -5701657, -5701657,
    -5701657, 3211216, 3211216, 3211216, 3211216, 1638488, 1638488, 1638488, 1638488, -5242947,
    -5242947, -5242947, -5242947, 5308471, 5308471, 5308471, 5308471, -5767185, -5767185, -5767185,
    -5767185, 5111783, 5111783, 5111783, 5111783, -3145666, -3145666, -3145666, -3145666, 589740,
    589740, 589740, 589740, 2162776, 2162776, 2162776, 2162776, -4325449, -4325449, -4325449,
    -4325449, 5701673, 5701673, 5701673, 5701673, 4390974, 4390974, 4390974, 4390974, -4718647,
    -4718647, -4718647, -4718647, 5046320, 5046320, 5046320, 5046320, -5242921, -5242921, -5242921,
    -5242921, 5505057, 5505057, 5505057, 5505057, -5636121, -5636121, -5636121, -5636121, 5767185,
    5767185, 5767185, 5767185, -5767176, -5767176, -5767176, -5767176, 3145795, 3145795, 3145795,
    3145795, -1572945, -1572945, -1572945, -1572945, 88, 88, 88, 88, 1703848, 1703848, 1703848,
    1703848, -3145647, -3145647, -3145647, -3145647, 4456381, 4456381, 4456381, 4456381, -5308368,
    -5308368, -5308368, -5308368, 5832679, 5832679, 5832679, 5832679, 1638473, 1638473, 1638473,
    1638473, 2228135, 2228135, 2228135, 2228135, -5046205, -5046205, -5046205, -5046205, 5832687,
    5832687, 5832687, 5832687, -3997737, -3997737, -3997737, -3997737, 524369, 524369, 524369,
    524369, 3211177, 3211177, 3211177, 3211177, -5504969, -5504969, -5504969, -5504969, 77, 77, 77,
    77, 5111731, 5111731, 5111731, 5111731, -5046272, -5046272, -5046272, -5046272, 77, 77, 77, 77,
    5111731, 5111731, 5111731, 5111731, -5046272, -5046272, -5046272, -5046272, 77, 77, 77, 77,
    5111731, 5111731, 5111731, 5111731, -1638319, -1638319, -1638319, -1638319, 5832656, 5832656,
    5832656, 5832656, 65469, 65469, 65469, 65469, -5767101, -5767101, -5767101, -5767101, 1638448,
    1638448, 1638448, 1638448, 5373871, 5373871, 5373871, 5373871, -3080217, -3080217, -3080217,
    -3080217, -4390824, -4390824, -4390824, -4390824, -3145644, -3145644, -3145644, -3145644,
    4128760, 4128760, 4128760, 4128760, 5111720, 5111720, 5111720, 5111720, -1572897, -1572897,
    -1572897, -1572897, -5832631, -5832631, -5832631, -5832631, -1114045, -1114045, -1114045,
    -1114045, 5373911, 5373911, 5373911, 5373911, 3669929, 3669929, 3669929, 3669929, -4390825,
    -4390825, -4390825, -4390825, 524321, 524321, 524321, 524321, 5111760, 5111760, 5111760,
    5111760, 5373863, 5373863, 5373863, 5373863, 1179593, 1179593, 1179593, 1179593, -4063207,
    -4063207, -4063207, -4063207, -5767084, -5767084, -5767084, -5767084, -2686903, -2686903,
    -2686903, -2686903, -5308328, -5308328, -5308328, -5308328, -3145661, -3145661, -3145661,
    -3145661, 25, 25, 25, 25, 3211239, 3211239, 3211239, 3211239, 5373885, 5373885, 5373885,
    5373885, 5832616, 5832616, 5832616, 5832616, 4456367, 4456367, 4456367, 4456367, 1703888,
    1703888, 1703888, 1703888, -5767079, -5767079, -5767079, -5767079, -5504937, -5504937,
    -5504937, -5504937, -5046191, -5046191, -5046191, -5046191, -4390839, -4390839, -4390839,
    -4390839, -3604418, -3604418, -3604418, -3604418, -2686928, -2686928, -2686928, -2686928,
    -1638367, -1638367, -1638367, -1638367, -524271, -524271, -524271, -524271,
];

pub(crate) static FLIPADST16_KP_X4: [i32; 512] = [
    5767257, 5767257, 5767257, 5767257, 5505111, 5505111, 5505111, 5505111, 5046353, 5046353,
    5046353, 5046353, 4390985, 4390985, 4390985, 4390985, 3604542, 3604542, 3604542, 3604542,
    2687024, 2687024, 2687024, 2687024, 1638433, 1638433, 1638433, 1638433, 524305, 524305, 524305,
    524305, 5308504, 5308504, 5308504, 5308504, 3145795, 3145795, 3145795, 3145795, 25, 25, 25, 25,
    -3080217, -3080217, -3080217, -3080217, -5242947, -5242947, -5242947, -5242947, -5701720,
    -5701720, -5701720, -5701720, -4325457, -4325457, -4325457, -4325457, -1572912, -1572912,
    -1572912, -1572912, 4390999, 4390999, 4390999, 4390999, -524255, -524255, -524255, -524255,
    -4980784, -4980784, -4980784, -4980784, -5242969, -5242969, -5242969, -5242969, -1048631,
    -1048631, -1048631, -1048631, 4063257, 4063257, 4063257, 4063257, 5767252, 5767252, 5767252,
    5767252, 2687049, 2687049, 2687049, 2687049, 3145812, 3145812, 3145812, 3145812, -3997704,
    -3997704, -3997704, -3997704, -4980824, -4980824, -4980824, -4980824, 1703903, 1703903,
    1703903, 1703903, 5832777, 5832777, 5832777, 5832777, 1114179, 1114179, 1114179, 1114179,
    -5242921, -5242921, -5242921, -5242921, -3539031, -3539031, -3539031, -3539031, 1638481,
    1638481, 1638481, 1638481, -5701680, -5701680, -5701680, -5701680, 65469, 65469, 65469, 65469,
    5767235, 5767235, 5767235, 5767235, -1638352, -1638352, -1638352, -1638352, -5242961, -5242961,
    -5242961, -5242961, 3211239, 3211239, 3211239, 3211239, 4391000, 4391000, 4391000, 4391000, 77,
    77, 77, 77, -4980813, -4980813, -4980813, -4980813, 5046272, 5046272, 5046272, 5046272, 77, 77,
    77, 77, -4980813, -4980813, -4980813, -4980813, 5046272, 5046272, 5046272, 5046272, 77, 77, 77,
    77, -4980813, -4980813, -4980813, -4980813, -1638327, -1638327, -1638327, -1638327, -2097241,
    -2097241, -2097241, -2097241, 5046339, 5046339, 5046339, 5046339, -5701649, -5701649, -5701649,
    -5701649, 4128727, 4128727, 4128727, 4128727, -524207, -524207, -524207, -524207, -3080279,
    -3080279, -3080279, -3080279, 5505079, 5505079, 5505079, 5505079, -3145661, -3145661, -3145661,
    -3145661, 1703855, 1703855, 1703855, 1703855, 88, 88, 88, 88, -1572952, -1572952, -1572952,
    -1572952, 3145809, 3145809, 3145809, 3145809, -4325443, -4325443, -4325443, -4325443, 5308464,
    5308464, 5308464, 5308464, -5701657, -5701657, -5701657, -5701657, -4390850, -4390850,
    -4390850, -4390850, 4849609, 4849609, 4849609, 4849609, -5046224, -5046224, -5046224, -5046224,
    5373911, 5373911, 5373911, 5373911, -5504991, -5504991, -5504991, -5504991, 5767143, 5767143,
    5767143, 5767143, -5767151, -5767151, -5767151, -5767151, 5898232, 5898232, 5898232, 5898232,
    -5308361, -5308361, -5308361, -5308361, 5898223, 5898223, 5898223, 5898223, -4980761, -4980761,
    -4980761, -4980761, 3145790, 3145790, 3145790, 3145790, -458836, -458836, -458836, -458836,
    -2162600, -2162600, -2162600, -2162600, 4456375, 4456375, 4456375, 4456375, -5701591, -5701591,
    -5701591, -5701591, -5767120, -5767120, -5767120, -5767120, 4390937, 4390937, 4390937, 4390937,
    65455, 65455, 65455, 65455, -4390831, -4390831, -4390831, -4390831, 5832679, 5832679, 5832679,
    5832679, -3080240, -3080240, -3080240, -3080240, -1638312, -1638312, -1638312, -1638312,
    5373885, 5373885, 5373885, 5373885, -5767127, -5767127, -5767127, -5767127, 1114174, 1114174,
    1114174, 1114174, 5111727, 5111727, 5111727, 5111727, -4325384, -4325384, -4325384, -4325384,
    -2162601, -2162601, -2162601, -2162601, 5898192, 5898192, 5898192, 5898192, -1572919, -1572919,
    -1572919, -1572919, -4784044, -4784044, -4784044, -4784044, -5308383, -5308383, -5308383,
    -5308383, -2686892, -2686892, -2686892, -2686892, 5111783, 5111783, 5111783, 5111783, 3211177,
    3211177, 3211177, 3211177, -4784111, -4784111, -4784111, -4784111, -3604392, -3604392,
    -3604392, -3604392, 4456440, 4456440, 4456440, 4456440, 4128679, 4128679, 4128679, 4128679,
    -4390887, -4390887, -4390887, -4390887, -5308328, -5308328, -5308328, -5308328, 48, 48, 48, 48,
    5373904, 5373904, 5373904, 5373904, 4456360, 4456360, 4456360, 4456360, -1572889, -1572889,
    -1572889, -1572889, -5767101, -5767101, -5767101, -5767101, -3145647, -3145647, -3145647,
    -3145647, -3145711, -3145711, -3145711, -3145711, -5701559, -5701559, -5701559, -5701559,
    -5046184, -5046184, -5046184, -5046184, -1638345, -1638345, -1638345, -1638345, 2752504,
    2752504, 2752504, 2752504, 5570493, 5570493, 5570493, 5570493, 5373863, 5373863, 5373863,
    5373863, 2228162, 2228162, 2228162, 2228162, -1638392, -1638392, -1638392, -1638392, -3604439,
    -3604439, -3604439, -3604439, -5046205, -5046205, -5046205, -5046205, -5767084, -5767084,
    -5767084, -5767084, -5701543, -5701543, -5701543, -5701543, -4784047, -4784047, -4784047,
    -4784047, -3145666, -3145666, -3145666, -3145666, -1114079, -1114079, -1114079, -1114079,
];

pub(crate) static DCT16_DENSE_PAIR_X4: [i32; 512] = [
    5898304, 5898304, 5898304, 5898304, 5701721, 5701721, 5701721, 5701721, 5242963, 5242963,
    5242963, 5242963, 4587595, 4587595, 4587595, 4587595, 3735616, 3735616, 3735616, 3735616,
    2818098, 2818098, 2818098, 2818098, 1703971, 1703971, 1703971, 1703971, 589842, 589842, 589842,
    589842, 5701696, 5701696, 5701696, 5701696, 3735627, 3735627, 3735627, 3735627, 589859, 589859,
    589859, 589859, -2752530, -2752530, -2752530, -2752530, -5177408, -5177408, -5177408, -5177408,
    -5832793, -5832793, -5832793, -5832793, -4522067, -4522067, -4522067, -4522067, -1638450,
    -1638450, -1638450, -1638450, 5242944, 5242944, 5242944, 5242944, 589874, 589874, 589874,
    589874, -4522019, -4522019, -4522019, -4522019, -5636185, -5636185, -5636185, -5636185,
    -1638464, -1638464, -1638464, -1638464, 3735570, 3735570, 3735570, 3735570, 5898323, 5898323,
    5898323, 5898323, 2818123, 2818123, 2818123, 2818123, 4587584, 4587584, 4587584, 4587584,
    -2818030, -2818030, -2818030, -2818030, -5636179, -5636179, -5636179, -5636179, 655310, 655310,
    655310, 655310, 5898304, 5898304, 5898304, 5898304, 1704011, 1704011, 1704011, 1704011,
    -5177379, -5177379, -5177379, -5177379, -3670105, -3670105, -3670105, -3670105, 3735616,
    3735616, 3735616, 3735616, -5177362, -5177362, -5177362, -5177362, -1638483, -1638483,
    -1638483, -1638483, 5898290, 5898290, 5898290, 5898290, -589760, -589760, -589760, -589760,
    -5636171, -5636171, -5636171, -5636171, 2883549, 2883549, 2883549, 2883549, 4587609, 4587609,
    4587609, 4587609, 2818112, 2818112, 2818112, 2818112, -5832754, -5832754, -5832754, -5832754,
    3801053, 3801053, 3801053, 3801053, 1704025, 1704025, 1704025, 1704025, -5636160, -5636160,
    -5636160, -5636160, 4653038, 4653038, 4653038, 4653038, 589907, 589907, 589907, 589907,
    -5177419, -5177419, -5177419, -5177419, 1704000, 1704000, 1704000, 1704000, -4522059, -4522059,
    -4522059, -4522059, 5898275, 5898275, 5898275, 5898275, -5242862, -5242862, -5242862, -5242862,
    2883520, 2883520, 2883520, 2883520, 589913, 589913, 589913, 589913, -3670099, -3670099,
    -3670099, -3670099, 5701682, 5701682, 5701682, 5701682, 589888, 589888, 589888, 589888,
    -1638489, -1638489, -1638489, -1638489, 2818131, 2818131, 2818131, 2818131, -3670091, -3670091,
    -3670091, -3670091, 4587584, 4587584, 4587584, 4587584, -5177394, -5177394, -5177394, -5177394,
    5701667, 5701667, 5701667, 5701667, -5832722, -5832722, -5832722, -5832722, -589760, -589760,
    -589760, -589760, 1769383, 1769383, 1769383, 1769383, -2817965, -2817965, -2817965, -2817965,
    3801013, 3801013, 3801013, 3801013, -4587456, -4587456, -4587456, -4587456, 5308366, 5308366,
    5308366, 5308366, -5701597, -5701597, -5701597, -5701597, 5963758, 5963758, 5963758, 5963758,
    -1703872, -1703872, -1703872, -1703872, 4652981, 4652981, 4652981, 4652981, -5898205, -5898205,
    -5898205, -5898205, 5242898, 5242898, 5242898, 5242898, -2752576, -2752576, -2752576, -2752576,
    -589735, -589735, -589735, -589735, 3801005, 3801005, 3801005, 3801005, -5701582, -5701582,
    -5701582, -5701582, -2817984, -2817984, -2817984, -2817984, 5963726, 5963726, 5963726, 5963726,
    -3670051, -3670051, -3670051, -3670051, -1703847, -1703847, -1703847, -1703847, 5767104,
    5767104, 5767104, 5767104, -4522002, -4522002, -4522002, -4522002, -589741, -589741, -589741,
    -589741, 5308341, 5308341, 5308341, 5308341, -3735488, -3735488, -3735488, -3735488, 5308398,
    5308398, 5308398, 5308398, 1769389, 1769389, 1769389, 1769389, -5898190, -5898190, -5898190,
    -5898190, 589888, 589888, 589888, 589888, 5767093, 5767093, 5767093, 5767093, -2752547,
    -2752547, -2752547, -2752547, -4587431, -4587431, -4587431, -4587431, -4587456, -4587456,
    -4587456, -4587456, 2818066, 2818066, 2818066, 2818066, 5767085, 5767085, 5767085, 5767085,
    -524338, -524338, -524338, -524338, -5898176, -5898176, -5898176, -5898176, -1703861, -1703861,
    -1703861, -1703861, 5308381, 5308381, 5308381, 5308381, 3800999, 3800999, 3800999, 3800999,
    -5242816, -5242816, -5242816, -5242816, -589774, -589774, -589774, -589774, 4653021, 4653021,
    4653021, 4653021, 5767079, 5767079, 5767079, 5767079, 1769408, 1769408, 1769408, 1769408,
    -3735534, -3735534, -3735534, -3735534, -5898157, -5898157, -5898157, -5898157, -2817973,
    -2817973, -2817973, -2817973, -5701568, -5701568, -5701568, -5701568, -3735477, -3735477,
    -3735477, -3735477, -589789, -589789, -589789, -589789, 2883566, 2883566, 2883566, 2883566,
    5308352, 5308352, 5308352, 5308352, 5963687, 5963687, 5963687, 5963687, 4652973, 4652973,
    4652973, 4652973, 1769422, 1769422, 1769422, 1769422, -5898176, -5898176, -5898176, -5898176,
    -5701543, -5701543, -5701543, -5701543, -5242797, -5242797, -5242797, -5242797, -4587445,
    -4587445, -4587445, -4587445, -3735488, -3735488, -3735488, -3735488, -2817998, -2817998,
    -2817998, -2817998, -1703901, -1703901, -1703901, -1703901, -589806, -589806, -589806, -589806,
];

pub(crate) static DCT32_DENSE_PAIR_X4: [i32; 2048] = [
    5898304, 5898304, 5898304, 5898304, 5898330, 5898330, 5898330, 5898330, 5767257, 5767257,
    5767257, 5767257, 5570647, 5570647, 5570647, 5570647, 5374035, 5374035, 5374035, 5374035,
    5111888, 5111888, 5111888, 5111888, 4784203, 4784203, 4784203, 4784203, 4390982, 4390982,
    4390982, 4390982, 3997760, 3997760, 3997760, 3997760, 3539001, 3539001, 3539001, 3539001,
    3080242, 3080242, 3080242, 3080242, 2555947, 2555947, 2555947, 2555947, 1966115, 1966115,
    1966115, 1966115, 1441818, 1441818, 1441818, 1441818, 851986, 851986, 851986, 851986, 262153,
    262153, 262153, 262153, 5898304, 5898304, 5898304, 5898304, 5374039, 5374039, 5374039, 5374039,
    4390987, 4390987, 4390987, 4390987, 3080249, 3080249, 3080249, 3080249, 1441827, 1441827,
    1441827, 1441827, -262135, -262135, -262135, -262135, -1900562, -1900562, -1900562, -1900562,
    -3473451, -3473451, -3473451, -3473451, -4718656, -4718656, -4718656, -4718656, -5505104,
    -5505104, -5505104, -5505104, -5832793, -5832793, -5832793, -5832793, -5701722, -5701722,
    -5701722, -5701722, -5046355, -5046355, -5046355, -5046355, -3932230, -3932230, -3932230,
    -3932230, -2490418, -2490418, -2490418, -2490418, -786458, -786458, -786458, -786458, 5767232,
    5767232, 5767232, 5767232, 4390992, 4390992, 4390992, 4390992, 1966130, 1966130, 1966130,
    1966130, -851959, -851959, -851959, -851959, -3473443, -3473443, -3473443, -3473443, -5308486,
    -5308486, -5308486, -5308486, -5832793, -5832793, -5832793, -5832793, -5046359, -5046359,
    -5046359, -5046359, -3014720, -3014720, -3014720, -3014720, -196634, -196634, -196634, -196634,
    2555922, 2555922, 2555922, 2555922, 4784185, 4784185, 4784185, 4784185, 5898323, 5898323,
    5898323, 5898323, 5570650, 5570650, 5570650, 5570650, 3997771, 3997771, 3997771, 3997771,
    1441835, 1441835, 1441835, 1441835, 5570624, 5570624, 5570624, 5570624, 3080262, 3080262,
    3080262, 3080262, -851950, -851950, -851950, -851950, -4325419, -4325419, -4325419, -4325419,
    -5832787, -5832787, -5832787, -5832787, -4718679, -4718679, -4718679, -4718679, -1376306,
    -1376306, -1376306, -1376306, 2555913, 2555913, 2555913, 2555913, 5374016, 5374016, 5374016,
    5374016, 5767258, 5767258, 5767258, 5767258, 3539019, 3539019, 3539019, 3539019, -262118,
    -262118, -262118, -262118, -3932195, -3932195, -3932195, -3932195, -5832784, -5832784,
    -5832784, -5832784, -5046361, -5046361, -5046361, -5046361, -1900601, -1900601, -1900601,
    -1900601, 5374016, 5374016, 5374016, 5374016, 1441849, 1441849, 1441849, 1441849, -3473426,
    -3473426, -3473426, -3473426, -5832784, -5832784, -5832784, -5832784, -3932243, -3932243,
    -3932243, -3932243, 917478, 917478, 917478, 917478, 5111858, 5111858, 5111858, 5111858,
    5570650, 5570650, 5570650, 5570650, 1966144, 1966144, 1966144, 1966144, -3014665, -3014665,
    -3014665, -3014665, -5832779, -5832779, -5832779, -5832779, -4325463, -4325463, -4325463,
    -4325463, 327645, 327645, 327645, 327645, 4784171, 4784171, 4784171, 4784171, 5767257, 5767257,
    5767257, 5767257, 2555974, 2555974, 2555974, 2555974, 5111872, 5111872, 5111872, 5111872,
    -262101, -262101, -262101, -262101, -5308466, -5308466, -5308466, -5308466, -4718682, -4718682,
    -4718682, -4718682, 917469, 917469, 917469, 917469, 5570617, 5570617, 5570617, 5570617,
    4391001, 4391001, 4391001, 4391001, -1441766, -1441766, -1441766, -1441766, -5701696, -5701696,
    -5701696, -5701696, -3932247, -3932247, -3932247, -3932247, 2031598, 2031598, 2031598, 2031598,
    5898310, 5898310, 5898310, 5898310, 3539027, 3539027, 3539027, 3539027, -2555895, -2555895,
    -2555895, -2555895, -5832779, -5832779, -5832779, -5832779, -3014736, -3014736, -3014736,
    -3014736, 4784192, 4784192, 4784192, 4784192, -1966054, -1966054, -1966054, -1966054, -5832779,
    -5832779, -5832779, -5832779, -1376326, -1376326, -1376326, -1376326, 5111843, 5111843,
    5111843, 5111843, 4391002, 4391002, 4391002, 4391002, -2555886, -2555886, -2555886, -2555886,
    -5832784, -5832784, -5832784, -5832784, -786496, -786496, -786496, -786496, 5373995, 5373995,
    5373995, 5373995, 3997785, 3997785, 3997785, 3997785, -3080183, -3080183, -3080183, -3080183,
    -5701715, -5701715, -5701715, -5701715, -196665, -196665, -196665, -196665, 5570610, 5570610,
    5570610, 5570610, 3539031, 3539031, 3539031, 3539031, 4390976, 4390976, 4390976, 4390976,
    -3538935, -3538935, -3538935, -3538935, -5046361, -5046361, -5046361, -5046361, 2621414,
    2621414, 2621414, 2621414, 5570643, 5570643, 5570643, 5570643, -1441749, -1441749, -1441749,
    -1441749, -5832779, -5832779, -5832779, -5832779, 327623, 327623, 327623, 327623, 5898304,
    5898304, 5898304, 5898304, 852038, 852038, 852038, 852038, -5701682, -5701682, -5701682,
    -5701682, -1900624, -1900624, -1900624, -1900624, 5373987, 5373987, 5373987, 5373987, 3080279,
    3080279, 3080279, 3080279, -4718610, -4718610, -4718610, -4718610, -3932250, -3932250,
    -3932250, -3932250, 3997760, 3997760, 3997760, 3997760, -4718601, -4718601, -4718601, -4718601,
    -3014745, -3014745, -3014745, -3014745, 5373978, 5373978, 5373978, 5373978, 1966163, 1966163,
    1966163, 1966163, -5701675, -5701675, -5701675, -5701675, -786507, -786507, -786507, -786507,
    5898297, 5898297, 5898297, 5898297, -262080, -262080, -262080, -262080, -5832774, -5832774,
    -5832774, -5832774, 1507278, 1507278, 1507278, 1507278, 5570640, 5570640, 5570640, 5570640,
    -2555869, -2555869, -2555869, -2555869, -5046359, -5046359, -5046359, -5046359, 3604462,
    3604462, 3604462, 3604462, 4391002, 4391002, 4391002, 4391002, 3539008, 3539008, 3539008,
    3539008, -5505050, -5505050, -5505050, -5505050, -196683, -196683, -196683, -196683, 5767238,
    5767238, 5767238, 5767238, -3080157, -3080157, -3080157, -3080157, -3932250, -3932250,
    -3932250, -3932250, 5373970, 5373970, 5373970, 5373970, 852048, 852048, 852048, 852048,
    -5832768, -5832768, -5832768, -5832768, 2621397, 2621397, 2621397, 2621397, 4391001, 4391001,
    4391001, 4391001, -5046281, -5046281, -5046281, -5046281, -1376339, -1376339, -1376339,
    -1376339, 5898297, 5898297, 5898297, 5898297, -1966030, -1966030, -1966030, -1966030, -4718679,
    -4718679, -4718679, -4718679, 3080256, 3080256, 3080256, 3080256, -5832747, -5832747, -5832747,
    -5832747, 2621390, 2621390, 2621390, 2621390, 3539034, 3539034, 3539034, 3539034, -5832739,
    -5832739, -5832739, -5832739, 2031559, 2031559, 2031559, 2031559, 3997785, 3997785, 3997785,
    3997785, -5701658, -5701658, -5701658, -5701658, 1507264, 1507264, 1507264, 1507264, 4390999,
    4390999, 4390999, 4390999, -5505042, -5505042, -5505042, -5505042, 917434, 917434, 917434,
    917434, 4784211, 4784211, 4784211, 4784211, -5308425, -5308425, -5308425, -5308425, 327605,
    327605, 327605, 327605, 5111888, 5111888, 5111888, 5111888, 2555968, 2555968, 2555968, 2555968,
    -5701689, -5701689, -5701689, -5701689, 4849646, 4849646, 4849646, 4849646, -262064, -262064,
    -262064, -262064, -4325459, -4325459, -4325459, -4325459, 5898266, 5898266, 5898266, 5898266,
    -3080142, -3080142, -3080142, -3080142, -1900634, -1900634, -1900634, -1900634, 5570624,
    5570624, 5570624, 5570624, -5111799, -5111799, -5111799, -5111799, 917429, 917429, 917429,
    917429, 3997783, 3997783, 3997783, 3997783, -5832739, -5832739, -5832739, -5832739, 3604437,
    3604437, 3604437, 3604437, 1441881, 1441881, 1441881, 1441881, -5308486, -5308486, -5308486,
    -5308486, 1966144, 1966144, 1966144, 1966144, -5046342, -5046342, -5046342, -5046342, 5898258,
    5898258, 5898258, 5898258, -3997653, -3997653, -3997653, -3997653, 327597, 327597, 327597,
    327597, 3539031, 3539031, 3539031, 3539031, -5701682, -5701682, -5701682, -5701682, 5439479,
    5439479, 5439479, 5439479, -2555840, -2555840, -2555840, -2555840, -1376346, -1376346,
    -1376346, -1376346, 4784203, 4784203, 4784203, 4784203, -5832730, -5832730, -5832730, -5832730,
    4456413, 4456413, 4456413, 4456413, -851888, -851888, -851888, -851888, -3014745, -3014745,
    -3014745, -3014745, 5570617, 5570617, 5570617, 5570617, 1441856, 1441856, 1441856, 1441856,
    -3932240, -3932240, -3932240, -3932240, 5570610, 5570610, 5570610, 5570610, -5832713, -5832713,
    -5832713, -5832713, 4849629, 4849629, 4849629, 4849629, -2555834, -2555834, -2555834, -2555834,
    -196697, -196697, -196697, -196697, 3080279, 3080279, 3080279, 3080279, -5046336, -5046336,
    -5046336, -5046336, 5898266, 5898266, 5898266, 5898266, -5373934, -5373934, -5373934, -5373934,
    3604423, 3604423, 3604423, 3604423, -851885, -851885, -851885, -851885, -1900634, -1900634,
    -1900634, -1900634, 4390987, 4390987, 4390987, 4390987, -5701675, -5701675, -5701675, -5701675,
    852032, 852032, 852032, 852032, -2490455, -2490455, -2490455, -2490455, 3997771, 3997771,
    3997771, 3997771, -5046329, -5046329, -5046329, -5046329, 5767203, 5767203, 5767203, 5767203,
    -5832713, -5832713, -5832713, -5832713, 5636078, 5636078, 5636078, 5636078, -4784085, -4784085,
    -4784085, -4784085, 3604416, 3604416, 3604416, 3604416, -1966000, -1966000, -1966000, -1966000,
    327591, 327591, 327591, 327591, 1441882, 1441882, 1441882, 1441882, -3014739, -3014739,
    -3014739, -3014739, 4390982, 4390982, 4390982, 4390982, -5308466, -5308466, -5308466, -5308466,
    5898266, 5898266, 5898266, 5898266, 262208, 262208, 262208, 262208, -786522, -786522, -786522,
    -786522, 1441881, 1441881, 1441881, 1441881, -1900631, -1900631, -1900631, -1900631, 2555987,
    2555987, 2555987, 2555987, -3014736, -3014736, -3014736, -3014736, 3539019, 3539019, 3539019,
    3539019, -3932230, -3932230, -3932230, -3932230, 4390976, 4390976, 4390976, 4390976, -4718649,
    -4718649, -4718649, -4718649, 5111858, 5111858, 5111858, 5111858, -5308459, -5308459, -5308459,
    -5308459, 5570595, 5570595, 5570595, 5570595, -5701658, -5701658, -5701658, -5701658, 5898258,
    5898258, 5898258, 5898258, -5832713, -5832713, -5832713, -5832713, -262080, -262080, -262080,
    -262080, 917414, 917414, 917414, 917414, -1441703, -1441703, -1441703, -1441703, 2031529,
    2031529, 2031529, 2031529, -2555821, -2555821, -2555821, -2555821, 3145648, 3145648, 3145648,
    3145648, -3538869, -3538869, -3538869, -3538869, 4063162, 4063162, 4063162, 4063162, -4390848,
    -4390848, -4390848, -4390848, 4849607, 4849607, 4849607, 4849607, -5111758, -5111758, -5111758,
    -5111758, 5439445, 5439445, 5439445, 5439445, -5570525, -5570525, -5570525, -5570525, 5832678,
    5832678, 5832678, 5832678, -5898222, -5898222, -5898222, -5898222, 5963767, 5963767, 5963767,
    5963767, -851904, -851904, -851904, -851904, 2621353, 2621353, 2621353, 2621353, -3997621,
    -3997621, -3997621, -3997621, 5177287, 5177287, 5177287, 5177287, -5767133, -5767133, -5767133,
    -5767133, 5963767, 5963767, 5963767, 5963767, -5505042, -5505042, -5505042, -5505042, 4784171,
    4784171, 4784171, 4784171, -3473472, -3473472, -3473472, -3473472, 1966160, 1966160, 1966160,
    1966160, -196697, -196697, -196697, -196697, -1441702, -1441702, -1441702, -1441702, 3145645,
    3145645, 3145645, 3145645, -4390842, -4390842, -4390842, -4390842, 5439438, 5439438, 5439438,
    5439438, -5898214, -5898214, -5898214, -5898214, -1441728, -1441728, -1441728, -1441728,
    4063152, 4063152, 4063152, 4063152, -5570510, -5570510, -5570510, -5570510, 5963767, 5963767,
    5963767, 5963767, -4718627, -4718627, -4718627, -4718627, 2555974, 2555974, 2555974, 2555974,
    327591, 327591, 327591, 327591, -3080105, -3080105, -3080105, -3080105, 5177280, 5177280,
    5177280, 5177280, -5898214, -5898214, -5898214, -5898214, 5373970, 5373970, 5373970, 5373970,
    -3473465, -3473465, -3473465, -3473465, 852051, 852051, 852051, 852051, 2031526, 2031526,
    2031526, 2031526, -4390837, -4390837, -4390837, -4390837, 5832661, 5832661, 5832661, 5832661,
    -1966016, -1966016, -1966016, -1966016, 5177274, 5177274, 5177274, 5177274, -5898222, -5898222,
    -5898222, -5898222, 3997739, 3997739, 3997739, 3997739, -196691, -196691, -196691, -196691,
    -3538857, -3538857, -3538857, -3538857, 5832654, 5832654, 5832654, 5832654, -5308425, -5308425,
    -5308425, -5308425, 2555968, 2555968, 2555968, 2555968, 1507238, 1507238, 1507238, 1507238,
    -4784053, -4784053, -4784053, -4784053, 5963750, 5963750, 5963750, 5963750, -4325411, -4325411,
    -4325411, -4325411, 852048, 852048, 852048, 852048, 3145639, 3145639, 3145639, 3145639,
    -5570503, -5570503, -5570503, -5570503, -2555840, -2555840, -2555840, -2555840, 5832647,
    5832647, 5832647, 5832647, -4718610, -4718610, -4718610, -4718610, 262224, 262224, 262224,
    262224, 4456365, 4456365, 4456365, 4456365, -5898214, -5898214, -5898214, -5898214, 3080242,
    3080242, 3080242, 3080242, 2031526, 2031526, 2031526, 2031526, -5570496, -5570496, -5570496,
    -5570496, 5111817, 5111817, 5111817, 5111817, -786507, -786507, -786507, -786507, -3997609,
    -3997609, -3997609, -3997609, 5963741, 5963741, 5963741, 5963741, -3473451, -3473451, -3473451,
    -3473451, -1441703, -1441703, -1441703, -1441703, 5439418, 5439418, 5439418, 5439418, -3080128,
    -3080128, -3080128, -3080128, 5963733, 5963733, 5963733, 5963733, -2490418, -2490418, -2490418,
    -2490418, -3538854, -3538854, -3538854, -3538854, 5963741, 5963741, 5963741, 5963741, -1900601,
    -1900601, -1900601, -1900601, -3997607, -3997607, -3997607, -3997607, 5832678, 5832678,
    5832678, 5832678, -1376320, -1376320, -1376320, -1376320, -4390825, -4390825, -4390825,
    -4390825, 5636078, 5636078, 5636078, 5636078, -786502, -786502, -786502, -786502, -4784045,
    -4784045, -4784045, -4784045, 5439479, 5439479, 5439479, 5439479, -196683, -196683, -196683,
    -196683, -5111728, -5111728, -5111728, -5111728, -3538880, -3538880, -3538880, -3538880,
    5636070, 5636070, 5636070, 5636070, 327605, 327605, 327605, 327605, -5767098, -5767098,
    -5767098, -5767098, 3080227, 3080227, 3080227, 3080227, 4063142, 4063142, 4063142, 4063142,
    -5373934, -5373934, -5373934, -5373934, -851888, -851888, -851888, -851888, 5963712, 5963712,
    5963712, 5963712, -2490411, -2490411, -2490411, -2490411, -4390823, -4390823, -4390823,
    -4390823, 5177335, 5177335, 5177335, 5177335, 1507245, 1507245, 1507245, 1507245, -5898183,
    -5898183, -5898183, -5898183, 1966130, 1966130, 1966130, 1966130, 4849577, 4849577, 4849577,
    4849577, -3997632, -3997632, -3997632, -3997632, 4849655, 4849655, 4849655, 4849655, 3145639,
    3145639, 3145639, 3145639, -5373926, -5373926, -5373926, -5373926, -1965997, -1965997,
    -1965997, -1965997, 5832661, 5832661, 5832661, 5832661, 917429, 917429, 917429, 917429,
    -5898183, -5898183, -5898183, -5898183, 262208, 262208, 262208, 262208, 5963706, 5963706,
    5963706, 5963706, -1376306, -1376306, -1376306, -1376306, -5570480, -5570480, -5570480,
    -5570480, 2555939, 2555939, 2555939, 2555939, 5177257, 5177257, 5177257, 5177257, -3473426,
    -3473426, -3473426, -3473426, -4390822, -4390822, -4390822, -4390822, -4390848, -4390848,
    -4390848, -4390848, 3538953, 3538953, 3538953, 3538953, 5177255, 5177255, 5177255, 5177255,
    -2490394, -2490394, -2490394, -2490394, -5570477, -5570477, -5570477, -5570477, 1441835,
    1441835, 1441835, 1441835, 5963701, 5963701, 5963701, 5963701, -196665, -196665, -196665,
    -196665, -5898176, -5898176, -5898176, -5898176, -851898, -851898, -851898, -851898, 5832654,
    5832654, 5832654, 5832654, 2031536, 2031536, 2031536, 2031536, -5373917, -5373917, -5373917,
    -5373917, -3080105, -3080105, -3080105, -3080105, 4849646, 4849646, 4849646, 4849646, 4063142,
    4063142, 4063142, 4063142, -4784064, -4784064, -4784064, -4784064, 1966106, 1966106, 1966106,
    1966106, 5963701, 5963701, 5963701, 5963701, 1507258, 1507258, 1507258, 1507258, -5111773,
    -5111773, -5111773, -5111773, -4390822, -4390822, -4390822, -4390822, 2555922, 2555922,
    2555922, 2555922, 5963696, 5963696, 5963696, 5963696, 917440, 917440, 917440, 917440, -5373909,
    -5373909, -5373909, -5373909, -3997607, -3997607, -3997607, -3997607, 3080201, 3080201,
    3080201, 3080201, 5832621, 5832621, 5832621, 5832621, 327623, 327623, 327623, 327623, -5570510,
    -5570510, -5570510, -5570510, -3538857, -3538857, -3538857, -3538857, -5111744, -5111744,
    -5111744, -5111744, 262187, 262187, 262187, 262187, 5439438, 5439438, 5439438, 5439438,
    4849574, 4849574, 4849574, 4849574, -786467, -786467, -786467, -786467, -5570503, -5570503,
    -5570503, -5570503, -4390823, -4390823, -4390823, -4390823, 1441818, 1441818, 1441818, 1441818,
    5832640, 5832640, 5832640, 5832640, 4063145, 4063145, 4063145, 4063145, -1900562, -1900562,
    -1900562, -1900562, -5898170, -5898170, -5898170, -5898170, -3538861, -3538861, -3538861,
    -3538861, 2555913, 2555913, 2555913, 2555913, 5963701, 5963701, 5963701, 5963701, 3145648,
    3145648, 3145648, 3145648, -5373888, -5373888, -5373888, -5373888, -1441735, -1441735,
    -1441735, -1441735, 3604462, 3604462, 3604462, 3604462, 5963696, 5963696, 5963696, 5963696,
    4063149, 4063149, 4063149, 4063149, -786458, -786458, -786458, -786458, -5111758, -5111758,
    -5111758, -5111758, -5570470, -5570470, -5570470, -5570470, -1966016, -1966016, -1966016,
    -1966016, 3145719, 3145719, 3145719, 3145719, 5963701, 5963701, 5963701, 5963701, 4456361,
    4456361, 4456361, 4456361, -196643, -196643, -196643, -196643, -4784085, -4784085, -4784085,
    -4784085, -5767079, -5767079, -5767079, -5767079, -2555834, -2555834, -2555834, -2555834,
    -5570496, -5570496, -5570496, -5570496, -3080122, -3080122, -3080122, -3080122, 851986, 851986,
    851986, 851986, 4456405, 4456405, 4456405, 4456405, 5963693, 5963693, 5963693, 5963693,
    4849577, 4849577, 4849577, 4849577, 1507278, 1507278, 1507278, 1507278, -2555895, -2555895,
    -2555895, -2555895, -5373888, -5373888, -5373888, -5373888, -5767078, -5767078, -5767078,
    -5767078, -3538869, -3538869, -3538869, -3538869, 262170, 262170, 262170, 262170, 4063197,
    4063197, 4063197, 4063197, 5963696, 5963696, 5963696, 5963696, 5177255, 5177255, 5177255,
    5177255, 2031559, 2031559, 2031559, 2031559, -5767104, -5767104, -5767104, -5767104, -4390832,
    -4390832, -4390832, -4390832, -1966030, -1966030, -1966030, -1966030, 851977, 851977, 851977,
    851977, 3604445, 3604445, 3604445, 3604445, 5439418, 5439418, 5439418, 5439418, 5963687,
    5963687, 5963687, 5963687, 5177257, 5177257, 5177257, 5177257, 3145664, 3145664, 3145664,
    3145664, 327654, 327654, 327654, 327654, -2555886, -2555886, -2555886, -2555886, -4784071,
    -4784071, -4784071, -4784071, -5898157, -5898157, -5898157, -5898157, -5570470, -5570470,
    -5570470, -5570470, -3997621, -3997621, -3997621, -3997621, -1441749, -1441749, -1441749,
    -1441749, -5898176, -5898176, -5898176, -5898176, -5373865, -5373865, -5373865, -5373865,
    -4390837, -4390837, -4390837, -4390837, -3080135, -3080135, -3080135, -3080135, -1441757,
    -1441757, -1441757, -1441757, 262153, 262153, 262153, 262153, 2031598, 2031598, 2031598,
    2031598, 3604437, 3604437, 3604437, 3604437, 4849600, 4849600, 4849600, 4849600, 5636016,
    5636016, 5636016, 5636016, 5963687, 5963687, 5963687, 5963687, 5832614, 5832614, 5832614,
    5832614, 5177261, 5177261, 5177261, 5177261, 4063162, 4063162, 4063162, 4063162, 2621390,
    2621390, 2621390, 2621390, 917478, 917478, 917478, 917478, -5898176, -5898176, -5898176,
    -5898176, -5898150, -5898150, -5898150, -5898150, -5767079, -5767079, -5767079, -5767079,
    -5570473, -5570473, -5570473, -5570473, -5373869, -5373869, -5373869, -5373869, -5111728,
    -5111728, -5111728, -5111728, -4784053, -4784053, -4784053, -4784053, -4390842, -4390842,
    -4390842, -4390842, -3997632, -3997632, -3997632, -3997632, -3538887, -3538887, -3538887,
    -3538887, -3080142, -3080142, -3080142, -3080142, -2555861, -2555861, -2555861, -2555861,
    -1966045, -1966045, -1966045, -1966045, -1441766, -1441766, -1441766, -1441766, -851950,
    -851950, -851950, -851950, -262135, -262135, -262135, -262135,
];

/// Full size-32 inverse DCT-II kernel `K32[in*32 + out]` for the flat butterfly.
#[rustfmt::skip]
pub(crate) static DCT32_DENSE_KERNEL: [i32; 1024] = [
      64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,
      90,  90,  88,  85,  82,  78,  73,  67,  61,  54,  47,  39,  30,  22,  13,   4,  -4, -13, -22, -30, -39, -47, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90,
      90,  87,  80,  70,  57,  43,  26,   9,  -9, -26, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -26,  -9,   9,  26,  43,  57,  70,  80,  87,  90,
      90,  82,  67,  47,  22,  -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13,  13,  39,  61,  78,  88,  90,  85,  73,  54,  30,   4, -22, -47, -67, -82, -90,
      89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,  89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89,
      88,  67,  30, -13, -54, -82, -90, -78, -47,  -4,  39,  73,  90,  85,  61,  22, -22, -61, -85, -90, -73, -39,   4,  47,  78,  90,  82,  54,  13, -30, -67, -88,
      87,  57,   9, -43, -80, -90, -70, -26,  26,  70,  90,  80,  43,  -9, -57, -87, -87, -57,  -9,  43,  80,  90,  70,  26, -26, -70, -90, -80, -43,   9,  57,  87,
      85,  47, -13, -67, -90, -73, -22,  39,  82,  88,  54,  -4, -61, -90, -78, -30,  30,  78,  90,  61,   4, -54, -88, -82, -39,  22,  73,  90,  67,  13, -47, -85,
      83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,  83,  35, -35, -83, -83, -35,  35,  83,
      82,  22, -54, -90, -61,  13,  78,  85,  30, -47, -90, -67,   4,  73,  88,  39, -39, -88, -73,  -4,  67,  90,  47, -30, -85, -78, -13,  61,  90,  54, -22, -82,
      80,   9, -70, -87, -26,  57,  90,  43, -43, -90, -57,  26,  87,  70,  -9, -80, -80,  -9,  70,  87,  26, -57, -90, -43,  43,  90,  57, -26, -87, -70,   9,  80,
      78,  -4, -82, -73,  13,  85,  67, -22, -88, -61,  30,  90,  54, -39, -90, -47,  47,  90,  39, -54, -90, -30,  61,  88,  22, -67, -85, -13,  73,  82,   4, -78,
      75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,  75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75,
      73, -30, -90, -22,  78,  67, -39, -90, -13,  82,  61, -47, -88,  -4,  85,  54, -54, -85,   4,  88,  47, -61, -82,  13,  90,  39, -67, -78,  22,  90,  30, -73,
      70, -43, -87,   9,  90,  26, -80, -57,  57,  80, -26, -90,  -9,  87,  43, -70, -70,  43,  87,  -9, -90, -26,  80,  57, -57, -80,  26,  90,   9, -87, -43,  70,
      67, -54, -78,  39,  85, -22, -90,   4,  90,  13, -88, -30,  82,  47, -73, -61,  61,  73, -47, -82,  30,  88, -13, -90,  -4,  90,  22, -85, -39,  78,  54, -67,
      64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,
      61, -73, -47,  82,  30, -88, -13,  90,  -4, -90,  22,  85, -39, -78,  54,  67, -67, -54,  78,  39, -85, -22,  90,   4, -90,  13,  88, -30, -82,  47,  73, -61,
      57, -80, -26,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  26,  80, -57, -57,  80,  26, -90,   9,  87, -43, -70,  70,  43, -87,  -9,  90, -26, -80,  57,
      54, -85,  -4,  88, -47, -61,  82,  13, -90,  39,  67, -78, -22,  90, -30, -73,  73,  30, -90,  22,  78, -67, -39,  90, -13, -82,  61,  47, -88,   4,  85, -54,
      50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,  50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50,
      47, -90,  39,  54, -90,  30,  61, -88,  22,  67, -85,  13,  73, -82,   4,  78, -78,  -4,  82, -73, -13,  85, -67, -22,  88, -61, -30,  90, -54, -39,  90, -47,
      43, -90,  57,  26, -87,  70,   9, -80,  80,  -9, -70,  87, -26, -57,  90, -43, -43,  90, -57, -26,  87, -70,  -9,  80, -80,   9,  70, -87,  26,  57, -90,  43,
      39, -88,  73,  -4, -67,  90, -47, -30,  85, -78,  13,  61, -90,  54,  22, -82,  82, -22, -54,  90, -61, -13,  78, -85,  30,  47, -90,  67,   4, -73,  88, -39,
      35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,  35, -83,  83, -35, -35,  83, -83,  35,
      30, -78,  90, -61,   4,  54, -88,  82, -39, -22,  73, -90,  67, -13, -47,  85, -85,  47,  13, -67,  90, -73,  22,  39, -82,  88, -54,  -4,  61, -90,  78, -30,
      26, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -26, -26,  70, -90,  80, -43,  -9,  57, -87,  87, -57,   9,  43, -80,  90, -70,  26,
      22, -61,  85, -90,  73, -39,  -4,  47, -78,  90, -82,  54, -13, -30,  67, -88,  88, -67,  30,  13, -54,  82, -90,  78, -47,   4,  39, -73,  90, -85,  61, -22,
      18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,  18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18,
      13, -39,  61, -78,  88, -90,  85, -73,  54, -30,   4,  22, -47,  67, -82,  90, -90,  82, -67,  47, -22,  -4,  30, -54,  73, -85,  90, -88,  78, -61,  39, -13,
       9, -26,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  26,  -9,  -9,  26, -43,  57, -70,  80, -87,  90, -90,  87, -80,  70, -57,  43, -26,   9,
       4, -13,  22, -30,  39, -47,  54, -61,  67, -73,  78, -82,  85, -88,  90, -90,  90, -90,  88, -85,  82, -78,  73, -67,  61, -54,  47, -39,  30, -22,  13,  -4,
];

pub(crate) fn idct_dequant_4x4_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    ScalarDct2d::idct_dequant_4x4(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_8x8_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    ScalarDct2d::idct_dequant_8x8(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_16x16_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    ScalarDct2d::idct_dequant_16x16(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_32x32_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    ScalarDct2d::idct_dequant_32x32(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_64x64_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    ScalarDct2d::idct_dequant_64x64(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn iadst_dequant_4x4_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    ScalarDct2d::iadst_dequant_4x4(
        coeff,
        tmp,
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

pub(crate) fn iadst_dequant_8x8_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    ScalarDct2d::iadst_dequant_8x8(
        coeff,
        tmp,
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

pub(crate) fn iadst_dequant_16x16_scalar(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    ScalarDct2d::iadst_dequant_16x16(
        coeff,
        tmp,
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

/// Scalar row pass (used for the rect2 sizes, mirroring the generic path's
/// `tx_class == 0` loop with the `* 181 + 128 >> 8` rect2 scaling).
fn idct_dequant_rows_rect_dct_scalar<const N: usize, const W: usize, const H: usize, C: Coeff>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut row = 0usize;

    loop {
        let tmp_row = row_mut(tmp, row);
        for (x, dst) in tmp_row[..W].iter_mut().enumerate() {
            let v = coeff[row + x * H].to_i32();
            *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
        dct_1d::<W>(tmp_row, 1);
        row += 1;
        if row & 3 == 0 {
            if eob > last_eob[ei] as i32 {
                ei += 1;
            } else {
                break;
            }
        }
    }

    while row < H {
        row_mut(tmp, row)[..W].fill(0);
        row += 1;
    }

    coeff[..W * H].fill(C::ZERO);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..H {
        crate::filter::row_clip(row_mut(tmp, y), W, rnd0, shift0, row_clip_min, row_clip_max);
    }
}

/// Pure-scalar rectangular DCT_DCT core (the universal fallback). The column
/// pass uses the scalar `dct_1d`, matching the generic path exactly.
pub(crate) fn idct_dequant_rect_scalar_core<
    const N: usize,
    const W: usize,
    const H: usize,
    C: Coeff,
>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_rows_rect_dct_scalar::<N, W, H, C>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
    for x in 0..W {
        dct_1d::<H>(&mut tmp[x..], ITX_TMP_STRIDE);
    }
}

/// Pure-scalar kind-aware rectangular core (rect2 sizes + universal fallback).
/// Mirrors the generic path: scalar rows with rect2 scaling, scalar columns.
pub(crate) fn itx_dequant_rect_scalar_core_mono<
    const N: usize,
    const W: usize,
    const H: usize,
    C: Coeff,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(is_dct_adst_kind(FIRST_KIND));
    debug_assert!(is_dct_adst_kind(SECOND_KIND));

    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut row = 0usize;

    loop {
        let dst_row = row_mut(tmp, row);
        for (x, dst) in dst_row[..W].iter_mut().enumerate() {
            let v = coeff[row + x * H].to_i32();
            *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
        tx_1d_scalar_mono::<W, FIRST_KIND>(dst_row, 1);
        row += 1;
        if row & 3 == 0 {
            if eob > last_eob[ei] as i32 {
                ei += 1;
            } else {
                break;
            }
        }
    }

    while row < H {
        row_mut(tmp, row)[..W].fill(0);
        row += 1;
    }

    coeff[..W * H].fill(C::ZERO);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..H {
        crate::filter::row_clip(row_mut(tmp, y), W, rnd0, shift0, row_clip_min, row_clip_max);
    }

    for x in 0..W {
        tx_1d_scalar_mono::<H, SECOND_KIND>(&mut tmp[x..], ITX_TMP_STRIDE);
    }
}

pub(crate) fn itx_dequant_rect_scalar_core<
    const N: usize,
    const W: usize,
    const H: usize,
    C: Coeff,
>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(is_dct_adst_kind(first_kind));
    debug_assert!(is_dct_adst_kind(second_kind));
    dispatch_dct_adst_pair!(first_kind, second_kind, |FK, SK| {
        itx_dequant_rect_scalar_core_mono::<N, W, H, C, FK, SK>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    });
}

static DEQUANT_4X4: OnceLock<IdctDequantFn<16>> = OnceLock::new();
static DEQUANT_8X8: OnceLock<IdctDequantFn<64>> = OnceLock::new();
static DEQUANT_16X16: OnceLock<IdctDequantFn<256>> = OnceLock::new();
static DEQUANT_32X32: OnceLock<IdctDequantFn<1024>> = OnceLock::new();
static DEQUANT_64X64: OnceLock<IdctDequantFn<1024>> = OnceLock::new();
static ADST_DEQUANT_4X4: OnceLock<IadstDequantFn<16>> = OnceLock::new();
static ADST_DEQUANT_8X8: OnceLock<IadstDequantFn<64>> = OnceLock::new();
static ADST_DEQUANT_16X16: OnceLock<IadstDequantFn<256>> = OnceLock::new();

#[inline]
pub(crate) fn idct_dequant_4x4(_hbd: bool) -> IdctDequantFn<16> {
    *DEQUANT_4X4.get_or_init(|| {
        let mut f: IdctDequantFn<16> = idct_dequant_4x4_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_4x4_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x4_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x8(_hbd: bool) -> IdctDequantFn<64> {
    *DEQUANT_8X8.get_or_init(|| {
        let mut f: IdctDequantFn<64> = idct_dequant_8x8_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_8x8_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x16(_hbd: bool) -> IdctDequantFn<256> {
    *DEQUANT_16X16.get_or_init(|| {
        let mut f: IdctDequantFn<256> = idct_dequant_16x16_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_16x16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x32(_hbd: bool) -> IdctDequantFn<1024> {
    *DEQUANT_32X32.get_or_init(|| {
        let mut f: IdctDequantFn<1024> = idct_dequant_32x32_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x32_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x32_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x32_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x32_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_64x64(_hbd: bool) -> IdctDequantFn<1024> {
    *DEQUANT_64X64.get_or_init(|| {
        let mut f: IdctDequantFn<1024> = idct_dequant_64x64_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_64x64_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_64x64_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_64x64_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x4(_hbd: bool) -> IadstDequantFn<16> {
    *ADST_DEQUANT_4X4.get_or_init(|| {
        let mut f: IadstDequantFn<16> = iadst_dequant_4x4_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_4x4_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_4x4_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x8(_hbd: bool) -> IadstDequantFn<64> {
    *ADST_DEQUANT_8X8.get_or_init(|| {
        let mut f: IadstDequantFn<64> = iadst_dequant_8x8_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_8x8_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_8x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x16(_hbd: bool) -> IadstDequantFn<256> {
    *ADST_DEQUANT_16X16.get_or_init(|| {
        let mut f: IadstDequantFn<256> = iadst_dequant_16x16_scalar;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_16x16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_16x16_avx2;
            }
        }
        f
    })
}

static DEQUANT_4X8: OnceLock<IdctDequantFn<32>> = OnceLock::new();
static DEQUANT_8X4: OnceLock<IdctDequantFn<32>> = OnceLock::new();
static DEQUANT_8X16: OnceLock<IdctDequantFn<128>> = OnceLock::new();
static DEQUANT_16X8: OnceLock<IdctDequantFn<128>> = OnceLock::new();
static DEQUANT_16X32: OnceLock<IdctDequantFn<512>> = OnceLock::new();
static DEQUANT_32X16: OnceLock<IdctDequantFn<512>> = OnceLock::new();
static DEQUANT_4X16: OnceLock<IdctDequantFn<64>> = OnceLock::new();
static DEQUANT_16X4: OnceLock<IdctDequantFn<64>> = OnceLock::new();
static DEQUANT_8X32: OnceLock<IdctDequantFn<256>> = OnceLock::new();
static DEQUANT_32X8: OnceLock<IdctDequantFn<256>> = OnceLock::new();
static DEQUANT_4X32: OnceLock<IdctDequantFn<128>> = OnceLock::new();
static DEQUANT_32X4: OnceLock<IdctDequantFn<128>> = OnceLock::new();

#[inline]
pub(crate) fn idct_dequant_4x8(_hbd: bool) -> IdctDequantFn<32> {
    *DEQUANT_4X8.get_or_init(|| {
        let mut f: IdctDequantFn<32> = idct_dequant_rect_scalar_core::<32, 4, 8, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_4x8_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_4x8_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x4(_hbd: bool) -> IdctDequantFn<32> {
    *DEQUANT_8X4.get_or_init(|| {
        let mut f: IdctDequantFn<32> = idct_dequant_rect_scalar_core::<32, 8, 4, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_8x4_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_8x4_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x4_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x16(_hbd: bool) -> IdctDequantFn<128> {
    *DEQUANT_8X16.get_or_init(|| {
        let mut f: IdctDequantFn<128> = idct_dequant_rect_scalar_core::<128, 8, 16, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_8x16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_8x16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x8(_hbd: bool) -> IdctDequantFn<128> {
    *DEQUANT_16X8.get_or_init(|| {
        let mut f: IdctDequantFn<128> = idct_dequant_rect_scalar_core::<128, 16, 8, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_16x8_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_16x8_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x32(_hbd: bool) -> IdctDequantFn<512> {
    *DEQUANT_16X32.get_or_init(|| {
        let mut f: IdctDequantFn<512> = idct_dequant_rect_scalar_core::<512, 16, 32, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_16x32_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_16x32_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x32_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x32_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x16(_hbd: bool) -> IdctDequantFn<512> {
    *DEQUANT_32X16.get_or_init(|| {
        let mut f: IdctDequantFn<512> = idct_dequant_rect_scalar_core::<512, 32, 16, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x16(_hbd: bool) -> IdctDequantFn<64> {
    *DEQUANT_4X16.get_or_init(|| {
        let mut f: IdctDequantFn<64> = idct_dequant_rect_scalar_core::<64, 4, 16, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_4x16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_4x16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x4(_hbd: bool) -> IdctDequantFn<64> {
    *DEQUANT_16X4.get_or_init(|| {
        let mut f: IdctDequantFn<64> = idct_dequant_rect_scalar_core::<64, 16, 4, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_16x4_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_16x4_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x4_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x32(_hbd: bool) -> IdctDequantFn<256> {
    *DEQUANT_8X32.get_or_init(|| {
        let mut f: IdctDequantFn<256> = idct_dequant_rect_scalar_core::<256, 8, 32, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_8x32_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_8x32_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x32_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x32_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x8(_hbd: bool) -> IdctDequantFn<256> {
    *DEQUANT_32X8.get_or_init(|| {
        let mut f: IdctDequantFn<256> = idct_dequant_rect_scalar_core::<256, 32, 8, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x8_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x8_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x32(_hbd: bool) -> IdctDequantFn<128> {
    *DEQUANT_4X32.get_or_init(|| {
        let mut f: IdctDequantFn<128> = idct_dequant_rect_scalar_core::<128, 4, 32, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_4x32_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_4x32_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x32_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x32_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x4(_hbd: bool) -> IdctDequantFn<128> {
    *DEQUANT_32X4.get_or_init(|| {
        let mut f: IdctDequantFn<128> = idct_dequant_rect_scalar_core::<128, 32, 4, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x4_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x4_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x4_avx2;
            }
        }
        f
    })
}

static ADST_DEQUANT_4X8: OnceLock<IadstDequantFn<32>> = OnceLock::new();
static ADST_DEQUANT_8X4: OnceLock<IadstDequantFn<32>> = OnceLock::new();
static ADST_DEQUANT_8X16: OnceLock<IadstDequantFn<128>> = OnceLock::new();
static ADST_DEQUANT_16X8: OnceLock<IadstDequantFn<128>> = OnceLock::new();
static ADST_DEQUANT_4X16: OnceLock<IadstDequantFn<64>> = OnceLock::new();
static ADST_DEQUANT_16X4: OnceLock<IadstDequantFn<64>> = OnceLock::new();

#[inline]
pub(crate) fn iadst_dequant_4x8(_hbd: bool) -> IadstDequantFn<32> {
    *ADST_DEQUANT_4X8.get_or_init(|| {
        let mut f: IadstDequantFn<32> = itx_dequant_rect_scalar_core::<32, 4, 8, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_4x8_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_4x8_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_4x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x4(_hbd: bool) -> IadstDequantFn<32> {
    *ADST_DEQUANT_8X4.get_or_init(|| {
        let mut f: IadstDequantFn<32> = itx_dequant_rect_scalar_core::<32, 8, 4, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_8x4_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_8x4_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_8x4_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x16(_hbd: bool) -> IadstDequantFn<128> {
    *ADST_DEQUANT_8X16.get_or_init(|| {
        let mut f: IadstDequantFn<128> = itx_dequant_rect_scalar_core::<128, 8, 16, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_8x16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_8x16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_8x16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x8(_hbd: bool) -> IadstDequantFn<128> {
    *ADST_DEQUANT_16X8.get_or_init(|| {
        let mut f: IadstDequantFn<128> = itx_dequant_rect_scalar_core::<128, 16, 8, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_16x8_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_16x8_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x8_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_16x8_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x16(_hbd: bool) -> IadstDequantFn<64> {
    *ADST_DEQUANT_4X16.get_or_init(|| {
        let mut f: IadstDequantFn<64> = itx_dequant_rect_scalar_core::<64, 4, 16, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_4x16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_4x16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_4x16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x4(_hbd: bool) -> IadstDequantFn<64> {
    *ADST_DEQUANT_16X4.get_or_init(|| {
        let mut f: IadstDequantFn<64> = itx_dequant_rect_scalar_core::<64, 16, 4, i32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_16x4_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_16x4_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x4_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_16x4_avx2;
            }
        }
        f
    })
}

// Low-bit-depth coefficient-specialized dispatch.  These entry points keep the
// decoded coefficient storage as i16 all the way into the SIMD row loaders; only
// transform arithmetic widens to i32.

static DEQUANT_4X4_I16: OnceLock<IdctDequantI16Fn<16>> = OnceLock::new();
static DEQUANT_8X8_I16: OnceLock<IdctDequantI16Fn<64>> = OnceLock::new();
static DEQUANT_16X16_I16: OnceLock<IdctDequantI16Fn<256>> = OnceLock::new();
static DEQUANT_32X32_I16: OnceLock<IdctDequantI16Fn<1024>> = OnceLock::new();
static DEQUANT_64X64_I16: OnceLock<IdctDequantI16Fn<1024>> = OnceLock::new();
static ADST_DEQUANT_4X4_I16: OnceLock<IadstDequantI16Fn<16>> = OnceLock::new();
static ADST_DEQUANT_8X8_I16: OnceLock<IadstDequantI16Fn<64>> = OnceLock::new();
static ADST_DEQUANT_16X16_I16: OnceLock<IadstDequantI16Fn<256>> = OnceLock::new();
static DEQUANT_4X8_I16: OnceLock<IdctDequantI16Fn<32>> = OnceLock::new();
static DEQUANT_8X4_I16: OnceLock<IdctDequantI16Fn<32>> = OnceLock::new();
static DEQUANT_8X16_I16: OnceLock<IdctDequantI16Fn<128>> = OnceLock::new();
static DEQUANT_16X8_I16: OnceLock<IdctDequantI16Fn<128>> = OnceLock::new();
static DEQUANT_16X32_I16: OnceLock<IdctDequantI16Fn<512>> = OnceLock::new();
static DEQUANT_32X16_I16: OnceLock<IdctDequantI16Fn<512>> = OnceLock::new();
static DEQUANT_4X16_I16: OnceLock<IdctDequantI16Fn<64>> = OnceLock::new();
static DEQUANT_16X4_I16: OnceLock<IdctDequantI16Fn<64>> = OnceLock::new();
static DEQUANT_8X32_I16: OnceLock<IdctDequantI16Fn<256>> = OnceLock::new();
static DEQUANT_32X8_I16: OnceLock<IdctDequantI16Fn<256>> = OnceLock::new();
static DEQUANT_4X32_I16: OnceLock<IdctDequantI16Fn<128>> = OnceLock::new();
static DEQUANT_32X4_I16: OnceLock<IdctDequantI16Fn<128>> = OnceLock::new();
static ADST_DEQUANT_4X8_I16: OnceLock<IadstDequantI16Fn<32>> = OnceLock::new();
static ADST_DEQUANT_8X4_I16: OnceLock<IadstDequantI16Fn<32>> = OnceLock::new();
static ADST_DEQUANT_8X16_I16: OnceLock<IadstDequantI16Fn<128>> = OnceLock::new();
static ADST_DEQUANT_16X8_I16: OnceLock<IadstDequantI16Fn<128>> = OnceLock::new();
static ADST_DEQUANT_4X16_I16: OnceLock<IadstDequantI16Fn<64>> = OnceLock::new();
static ADST_DEQUANT_16X4_I16: OnceLock<IadstDequantI16Fn<64>> = OnceLock::new();
#[inline]
pub(crate) fn idct_dequant_4x4_i16() -> IdctDequantI16Fn<16> {
    *DEQUANT_4X4_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<16> = idct_dequant_scalar_core::<16, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_4x4_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x4_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x8_i16() -> IdctDequantI16Fn<64> {
    *DEQUANT_8X8_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<64> = idct_dequant_scalar_core::<64, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_8x8_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x16_i16() -> IdctDequantI16Fn<256> {
    *DEQUANT_16X16_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<256> = idct_dequant_scalar_core::<256, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_16x16_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x32_i16() -> IdctDequantI16Fn<1024> {
    *DEQUANT_32X32_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<1024> = idct_dequant_scalar_core::<1024, 32, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x32_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x32_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x32_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x32_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_64x64_i16() -> IdctDequantI16Fn<1024> {
    *DEQUANT_64X64_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<1024> = idct_dequant_scalar_core::<1024, 32, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_64x64_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_64x64_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_64x64_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x4_i16() -> IadstDequantI16Fn<16> {
    *ADST_DEQUANT_4X4_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<16> = itx_dequant_scalar_core::<16, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_4x4_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_4x4_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x8_i16() -> IadstDequantI16Fn<64> {
    *ADST_DEQUANT_8X8_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<64> = itx_dequant_scalar_core::<64, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_8x8_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_8x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x16_i16() -> IadstDequantI16Fn<256> {
    *ADST_DEQUANT_16X16_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<256> = itx_dequant_scalar_core::<256, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_16x16_i16_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_16x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x8_i16() -> IdctDequantI16Fn<32> {
    *DEQUANT_4X8_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<32> = idct_dequant_rect_scalar_core::<32, 4, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_4x8_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_4x8_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x4_i16() -> IdctDequantI16Fn<32> {
    *DEQUANT_8X4_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<32> = idct_dequant_rect_scalar_core::<32, 8, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_8x4_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_8x4_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x4_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x16_i16() -> IdctDequantI16Fn<128> {
    *DEQUANT_8X16_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<128> = idct_dequant_rect_scalar_core::<128, 8, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_8x16_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_8x16_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x8_i16() -> IdctDequantI16Fn<128> {
    *DEQUANT_16X8_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<128> = idct_dequant_rect_scalar_core::<128, 16, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_16x8_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_16x8_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x32_i16() -> IdctDequantI16Fn<512> {
    *DEQUANT_16X32_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<512> = idct_dequant_rect_scalar_core::<512, 16, 32, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_16x32_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_16x32_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x32_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x32_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x16_i16() -> IdctDequantI16Fn<512> {
    *DEQUANT_32X16_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<512> = idct_dequant_rect_scalar_core::<512, 32, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x16_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x16_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x16_i16() -> IdctDequantI16Fn<64> {
    *DEQUANT_4X16_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<64> = idct_dequant_rect_scalar_core::<64, 4, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_4x16_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_4x16_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x4_i16() -> IdctDequantI16Fn<64> {
    *DEQUANT_16X4_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<64> = idct_dequant_rect_scalar_core::<64, 16, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_16x4_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_16x4_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_16x4_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x32_i16() -> IdctDequantI16Fn<256> {
    *DEQUANT_8X32_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<256> = idct_dequant_rect_scalar_core::<256, 8, 32, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_8x32_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_8x32_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x32_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_8x32_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x8_i16() -> IdctDequantI16Fn<256> {
    *DEQUANT_32X8_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<256> = idct_dequant_rect_scalar_core::<256, 32, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x8_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x8_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x32_i16() -> IdctDequantI16Fn<128> {
    *DEQUANT_4X32_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<128> = idct_dequant_rect_scalar_core::<128, 4, 32, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_4x32_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_4x32_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x32_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_4x32_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x4_i16() -> IdctDequantI16Fn<128> {
    *DEQUANT_32X4_I16.get_or_init(|| {
        let mut f: IdctDequantI16Fn<128> = idct_dequant_rect_scalar_core::<128, 32, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::idct_dequant_32x4_i16_neon_rdm;
                } else {
                    f = crate::neon::idct_dequant_32x4_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::idct_dequant_32x4_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x8_i16() -> IadstDequantI16Fn<32> {
    *ADST_DEQUANT_4X8_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<32> = itx_dequant_rect_scalar_core::<32, 4, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_4x8_i16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_4x8_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_4x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x4_i16() -> IadstDequantI16Fn<32> {
    *ADST_DEQUANT_8X4_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<32> = itx_dequant_rect_scalar_core::<32, 8, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_8x4_i16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_8x4_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_8x4_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x16_i16() -> IadstDequantI16Fn<128> {
    *ADST_DEQUANT_8X16_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<128> = itx_dequant_rect_scalar_core::<128, 8, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_8x16_i16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_8x16_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_8x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x8_i16() -> IadstDequantI16Fn<128> {
    *ADST_DEQUANT_16X8_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<128> = itx_dequant_rect_scalar_core::<128, 16, 8, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_16x8_i16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_16x8_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x8_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_16x8_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x16_i16() -> IadstDequantI16Fn<64> {
    *ADST_DEQUANT_4X16_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<64> = itx_dequant_rect_scalar_core::<64, 4, 16, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_4x16_i16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_4x16_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x16_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_4x16_i16_avx2;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x4_i16() -> IadstDequantI16Fn<64> {
    *ADST_DEQUANT_16X4_I16.get_or_init(|| {
        let mut f: IadstDequantI16Fn<64> = itx_dequant_rect_scalar_core::<64, 16, 4, i16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    f = crate::neon::iadst_dequant_16x4_i16_neon_rdm;
                } else {
                    f = crate::neon::iadst_dequant_16x4_i16_neon;
                }
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x4_i16_sse41;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                f = crate::avx::iadst_dequant_16x4_i16_avx2;
            }
        }
        f
    })
}
