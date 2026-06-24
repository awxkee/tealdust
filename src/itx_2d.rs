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

use crate::itx_1d::{
    ADST4_KERNEL_ROWS, ADST8_KERNEL_ROWS, ADST16_KERNEL_ROWS, DCT8_ODD_KERNEL,
    FLIPADST4_KERNEL_ROWS, FLIPADST16_KERNEL_ROWS, TX1D_FNS, TX1D_FNS_X8, inv_dct4_1d, inv_dct8_1d,
    inv_dct16_1d, inv_dct32_1d,
};
use crate::pixel::Coeff;
use crate::scan::LAST_EOB_PER_COL;
use std::convert::TryInto;
use std::sync::OnceLock;

pub(crate) const ITX_TMP_STRIDE: usize = 32;
pub(crate) const ITX_TMP_PIXELS: usize = ITX_TMP_STRIDE * ITX_TMP_STRIDE;

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
pub(crate) const TX_KIND_ADST: usize = 2;
pub(crate) const TX_KIND_FLIPADST: usize = 3;

#[inline(always)]
pub(crate) fn is_dct_adst_kind(kind: usize) -> bool {
    matches!(kind, TX_KIND_DCT | TX_KIND_ADST | TX_KIND_FLIPADST)
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
fn row_mut(tmp: &mut [i32; ITX_TMP_PIXELS], y: usize) -> &mut [i32; ITX_TMP_STRIDE] {
    (&mut tmp[y * ITX_TMP_STRIDE..(y + 1) * ITX_TMP_STRIDE])
        .try_into()
        .unwrap()
}

#[inline(always)]
fn dct_1d<const S: usize>(c: &mut [i32], stride: usize) {
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
fn tx_1d_scalar_mono<const S: usize, const KIND: usize>(c: &mut [i32], stride: usize) {
    debug_assert!(is_dct_adst_kind(KIND));
    let f = TX1D_FNS[tx_size_idx::<S>()][KIND].expect("unsupported 1D transform");
    f(c, stride);
}

fn itx_dequant_scalar_core_mono<
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

pub(crate) trait DctSimd4 {
    type V: Copy + crate::itx_1d::DctLane;
    /// s16 8-wide widening-MAC backend used by `dct_1d_x8` for the 16/32 sizes.
    type Wide: crate::itx_1d::DctWide;

    unsafe fn zero() -> Self::V;
    unsafe fn splat(v: i32) -> Self::V;
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn rect2_scale(a: Self::V) -> Self::V;
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V;
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V);
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V;
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V;
    unsafe fn to_array(v: Self::V) -> [i32; 4];

    /// Store four row-pass output columns as four contiguous scratch rows.
    ///
    /// The row-pass SIMD lanes carry four source rows for one output column.
    /// The column pass wants row-major scratch, so this is the hot transpose
    /// boundary. Architecture backends override this with a SIMD 4x4 transpose;
    /// the fallback is kept only for scalar-like backends and tests.
    #[inline(always)]
    unsafe fn store4x4_clip(
        tmp: &mut [i32; ITX_TMP_PIXELS],
        off: usize,
        stride: usize,
        v: [Self::V; 4],
        rnd: i32,
        shift: i32,
        min: i32,
        max: i32,
    ) {
        debug_assert!(off + 3 + 3 * stride < ITX_TMP_PIXELS);
        let c0 = unsafe { Self::to_array(v[0]) };
        let c1 = unsafe { Self::to_array(v[1]) };
        let c2 = unsafe { Self::to_array(v[2]) };
        let c3 = unsafe { Self::to_array(v[3]) };
        for r in 0..4 {
            let row = off + r * stride;
            tmp[row] = clip_row_value(c0[r], rnd, shift, min, max);
            tmp[row + 1] = clip_row_value(c1[r], rnd, shift, min, max);
            tmp[row + 2] = clip_row_value(c2[r], rnd, shift, min, max);
            tmp[row + 3] = clip_row_value(c3[r], rnd, shift, min, max);
        }
    }
}

pub(crate) trait ItxCoeff: Coeff {
    const USE_WIDE_16BIT: bool;

    unsafe fn load_simd4<B: DctSimd4>(src: &[Self], off: usize) -> B::V;

    unsafe fn load_wide8<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In;

    unsafe fn load_wide8_rect2<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In;

    unsafe fn load_wide4<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In;

    unsafe fn load_wide4_rect2<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In;
}

impl ItxCoeff for i16 {
    const USE_WIDE_16BIT: bool = true;

    #[inline(always)]
    unsafe fn load_simd4<B: DctSimd4>(src: &[Self], off: usize) -> B::V {
        unsafe { B::load_slice_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide8<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load8_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide8_rect2<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load8_rect2_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide4<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load4_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide4_rect2<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load4_rect2_i16(src, off) }
    }
}

impl ItxCoeff for i32 {
    const USE_WIDE_16BIT: bool = false;

    #[inline(always)]
    unsafe fn load_simd4<B: DctSimd4>(src: &[Self], off: usize) -> B::V {
        unsafe { B::load_slice(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide8<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load8_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide8_rect2<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load8_rect2_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide4<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load4_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load_wide4_rect2<W: crate::itx_1d::DctWide>(src: &[Self], off: usize) -> W::In {
        unsafe { W::load4_rect2_narrow(src, off) }
    }
}

#[inline(always)]
fn vmulc<B: DctSimd4>(v: B::V, k: i32) -> B::V {
    unsafe { B::mul(v, B::splat(k)) }
}

#[inline(always)]
fn clip_row_value(v: i32, rnd: i32, shift: i32, min: i32, max: i32) -> i32 {
    ((v + rnd) >> shift).max(min).min(max)
}

#[inline(always)]
fn sum_row_simd4<B: DctSimd4, const N: usize>(row: &[i8; N], x: &[B::V; N]) -> B::V {
    unsafe {
        let mut acc = B::zero();
        for i in 0..N {
            acc = B::add(acc, vmulc::<B>(x[i], row[i] as i32));
        }
        acc
    }
}

#[inline(always)]
fn load_1d_x4<B: DctSimd4, const N: usize>(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    stride: usize,
) -> [B::V; N] {
    debug_assert!(base + (N - 1) * stride + 3 < ITX_TMP_PIXELS);
    unsafe {
        let zero = B::zero();
        let mut out = [zero; N];
        for (i, dst) in out.iter_mut().enumerate() {
            *dst = B::load(tmp, base + i * stride);
        }
        out
    }
}

#[inline(always)]
fn store_1d_x4<B: DctSimd4, const N: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    base: usize,
    stride: usize,
    v: &[B::V; N],
) {
    unsafe {
        debug_assert!(base + (N - 1) * stride + 3 < ITX_TMP_PIXELS);
        for (i, &src) in v.iter().enumerate() {
            B::store(tmp, base + i * stride, src);
        }
    }
}

#[inline(always)]
fn load_coeff_rows_x4<B: DctSimd4, const S: usize, C: ItxCoeff>(
    coeff: &[C],
    y: usize,
) -> [B::V; S] {
    unsafe {
        let zero = B::zero();
        let mut out = [zero; S];
        for (x, dst) in out.iter_mut().enumerate() {
            *dst = C::load_simd4::<B>(coeff, y + x * S);
        }
        out
    }
}

#[inline(always)]
fn store_row_group_x4_clip<B: DctSimd4, const S: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    v: &[B::V; S],
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    debug_assert_eq!(S & 3, 0);
    let base = y * ITX_TMP_STRIDE;
    let mut x = 0usize;
    while x + 4 <= S {
        unsafe {
            B::store4x4_clip(
                tmp,
                base + x,
                ITX_TMP_STRIDE,
                [v[x], v[x + 1], v[x + 2], v[x + 3]],
                rnd,
                shift,
                min,
                max,
            );
        }
        x += 4;
    }
}

#[inline(always)]
fn store_row_group_wide_x4_clip<DW: crate::itx_1d::DctWide, const W: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    out: &[DW::Acc; W],
    clip: DW::Clip,
) {
    debug_assert_eq!(W & 3, 0);
    let base = y * ITX_TMP_STRIDE;
    let mut x = 0usize;
    while x + 4 <= W {
        unsafe {
            DW::store4x4_strided_clip::<false>(
                tmp,
                base + x,
                ITX_TMP_STRIDE,
                [out[x], out[x + 1], out[x + 2], out[x + 3]],
                clip,
            );
        }
        x += 4;
    }
}

#[inline(always)]
fn store_row_group_wide_x8_clip<DW: crate::itx_1d::DctWide, const W: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    out: &[DW::Acc; W],
    clip: DW::Clip,
) {
    debug_assert_eq!(W & 7, 0);
    let base = y * ITX_TMP_STRIDE;
    let mut x = 0usize;
    while x + 8 <= W {
        unsafe {
            DW::store8x8_strided_clip(
                tmp,
                base + x,
                ITX_TMP_STRIDE,
                [
                    out[x],
                    out[x + 1],
                    out[x + 2],
                    out[x + 3],
                    out[x + 4],
                    out[x + 5],
                    out[x + 6],
                    out[x + 7],
                ],
                clip,
            );
        }
        x += 8;
    }
}

#[inline(always)]
fn process_row_group_itx_wide_x4<
    B: DctSimd4,
    const W: usize,
    const H: usize,
    const RECT2: bool,
    const KIND: usize,
    C: ItxCoeff,
>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    use crate::itx_1d::DctWide;
    debug_assert!(C::USE_WIDE_16BIT);
    let s: [<B::Wide as DctWide>::In; W] = core::array::from_fn(|x| unsafe {
        if RECT2 {
            C::load_wide4_rect2::<B::Wide>(coeff, y + x * H)
        } else {
            C::load_wide4::<B::Wide>(coeff, y + x * H)
        }
    });
    unsafe {
        let clip = B::Wide::make_clip(rnd, shift, min, max);
        let zero = B::Wide::zero();
        let mut out = [zero; W];
        {
            let mut store = |m: usize, acc: <B::Wide as DctWide>::Acc| out[m] = acc;
            match (W, KIND) {
                (32, TX_KIND_DCT) => crate::itx_1d::dct32_wide::<B::Wide>(|j| s[j], &mut store),
                (16, TX_KIND_DCT) => crate::itx_1d::dct16_wide::<B::Wide>(|j| s[j], &mut store),
                (16, TX_KIND_ADST) => {
                    crate::itx_1d::adst16_wide::<B::Wide>(|j| s[j], &mut store, &ADST16_KW, false)
                }
                (16, TX_KIND_FLIPADST) => crate::itx_1d::adst16_wide::<B::Wide>(
                    |j| s[j],
                    &mut store,
                    &FLIPADST16_KW,
                    false,
                ),
                (8, TX_KIND_DCT) => {
                    crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], &mut store, &DCT8_KW, false)
                }
                (8, TX_KIND_ADST) => {
                    crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], &mut store, &ADST8_KW, false)
                }
                (8, TX_KIND_FLIPADST) => {
                    crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], &mut store, &ADST8_KW, true)
                }
                (4, TX_KIND_DCT) => {
                    crate::itx_1d::mat4_wide::<B::Wide>(|j| s[j], &mut store, &DCT4_KW)
                }
                (4, TX_KIND_ADST) => {
                    crate::itx_1d::mat4_wide::<B::Wide>(|j| s[j], &mut store, &ADST4_KW)
                }
                (4, TX_KIND_FLIPADST) => {
                    crate::itx_1d::mat4_wide::<B::Wide>(|j| s[j], &mut store, &FLIPADST4_KW)
                }
                _ => unreachable!(),
            }
        }
        store_row_group_wide_x4_clip::<B::Wide, W>(tmp, y, &out, clip);
    }
}

#[inline(always)]
fn process_row_group_x4<B: DctSimd4, const S: usize, C: ItxCoeff>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    if C::USE_WIDE_16BIT {
        process_row_group_itx_wide_x4::<B, S, S, false, TX_KIND_DCT, C>(
            coeff, tmp, y, rnd, shift, min, max,
        );
        return;
    }

    match S {
        4 => {
            let mut v = load_coeff_rows_x4::<B, 4, C>(coeff, y);
            inv_dct4_simd4::<B>(&mut v);
            store_row_group_x4_clip::<B, 4>(tmp, y, &v, rnd, shift, min, max);
        }
        8 => {
            let mut v = load_coeff_rows_x4::<B, 8, C>(coeff, y);
            inv_dct8_simd4::<B>(&mut v);
            store_row_group_x4_clip::<B, 8>(tmp, y, &v, rnd, shift, min, max);
        }
        16 => {
            let load = |j: usize| unsafe { C::load_simd4::<B>(coeff, y + j * 16) };
            let zero = unsafe { B::zero() };
            let mut out = [zero; 16];
            crate::itx_1d::dct16_flat_bylane::<B::V>(load, |m, v| out[m] = v);
            store_row_group_x4_clip::<B, 16>(tmp, y, &out, rnd, shift, min, max);
        }
        32 => {
            let load = |j: usize| unsafe { C::load_simd4::<B>(coeff, y + j * 32) };
            let zero = unsafe { B::zero() };
            let mut out = [zero; 32];
            crate::itx_1d::dct32_flat::<B::V>(load, |m, v| out[m] = v);
            store_row_group_x4_clip::<B, 32>(tmp, y, &out, rnd, shift, min, max);
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn process_row_group_wide_x8<B: DctSimd4, const S: usize, C: ItxCoeff>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    use crate::itx_1d::DctWide;
    let s: [<B::Wide as DctWide>::In; S] =
        core::array::from_fn(|j| unsafe { C::load_wide8::<B::Wide>(coeff, y + j * S) });
    unsafe {
        let clip = B::Wide::make_clip(rnd, shift, min, max);
        let zero = B::Wide::zero();
        let mut out = [zero; S];
        {
            let mut store = |m: usize, acc: <B::Wide as DctWide>::Acc| out[m] = acc;
            match S {
                16 => crate::itx_1d::dct16_wide::<B::Wide>(|j| s[j], &mut store),
                32 => crate::itx_1d::dct32_wide::<B::Wide>(|j| s[j], &mut store),
                _ => unreachable!(),
            }
        }
        store_row_group_wide_x8_clip::<B::Wide, S>(tmp, y, &out, clip);
    }
}

#[inline(always)]
fn idct_dequant_rows_dct_simd4<B: DctSimd4, const N: usize, const S: usize, C: ItxCoeff>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
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

    // Leading 4-column groups that may be non-zero (eob early-out). The check
    // only reads constants, so it can be hoisted ahead of the transform, which
    // lets the column pass run 8 columns at a time where possible.
    let mut ngrp = 0usize;
    while ngrp < S / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let ncols = ngrp * 4;
    let rnd0 = (1 << shift0) >> 1;

    let mut y = 0usize;
    if C::USE_WIDE_16BIT && (S == 16 || S == 32) {
        while y + 8 <= ncols {
            process_row_group_wide_x8::<B, S, C>(
                coeff,
                tmp,
                y,
                rnd0,
                shift0,
                row_clip_min,
                row_clip_max,
            );
            y += 8;
        }
    }
    while y + 4 <= ncols {
        process_row_group_x4::<B, S, C>(coeff, tmp, y, rnd0, shift0, row_clip_min, row_clip_max);
        y += 4;
    }

    while y < S {
        row_mut(tmp, y)[..S].fill(0);
        y += 1;
    }

    coeff[..S * S].fill(C::ZERO);
}

#[inline(always)]
fn even4<T: Copy>(v: &[T; 8]) -> [T; 4] {
    [v[0], v[2], v[4], v[6]]
}

#[inline(always)]
fn odd4<T: Copy>(v: &[T; 8]) -> [T; 4] {
    [v[1], v[3], v[5], v[7]]
}

#[inline(always)]
fn inv_dct4_simd4<B: DctSimd4>(v: &mut [B::V; 4]) {
    unsafe {
        let a0 = B::add(vmulc::<B>(v[0], 64), vmulc::<B>(v[2], 64));
        let a1 = B::sub(vmulc::<B>(v[0], 64), vmulc::<B>(v[2], 64));
        let b0 = B::add(vmulc::<B>(v[1], 83), vmulc::<B>(v[3], 35));
        let b1 = B::sub(vmulc::<B>(v[1], 35), vmulc::<B>(v[3], 83));

        v[0] = B::add(a0, b0);
        v[1] = B::add(a1, b1);
        v[2] = B::sub(a1, b1);
        v[3] = B::sub(a0, b0);
    }
}

#[inline(always)]
fn inv_dct8_simd4<B: DctSimd4>(v: &mut [B::V; 8]) {
    unsafe {
        let mut e = even4(v);
        inv_dct4_simd4::<B>(&mut e);
        let odd = odd4(v);
        let b0 = sum_row_simd4::<B, 4>(&DCT8_ODD_KERNEL[0], &odd);
        let b1 = sum_row_simd4::<B, 4>(&DCT8_ODD_KERNEL[1], &odd);
        let b2 = sum_row_simd4::<B, 4>(&DCT8_ODD_KERNEL[2], &odd);
        let b3 = sum_row_simd4::<B, 4>(&DCT8_ODD_KERNEL[3], &odd);

        v[0] = B::add(e[0], b0);
        v[7] = B::sub(e[0], b0);
        v[1] = B::add(e[1], b1);
        v[6] = B::sub(e[1], b1);
        v[2] = B::add(e[2], b2);
        v[5] = B::sub(e[2], b2);
        v[3] = B::add(e[3], b3);
        v[4] = B::sub(e[3], b3);
    }
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

#[inline(always)]
fn inv_dct16_simd4<B: DctSimd4>(v: &mut [B::V; 16]) {
    let s = *v;
    crate::itx_1d::dct16_flat_bylane::<B::V>(|j| s[j], |m, x| v[m] = x);
}

#[inline(always)]
fn inv_mat_simd4<B: DctSimd4, const N: usize>(v: &mut [B::V; N], mat: &[[i8; N]; N], flip: bool) {
    let src = *v;
    let zero = unsafe { B::zero() };
    let mut sums = [zero; N];

    for (row, dst) in mat.iter().zip(sums.iter_mut()) {
        *dst = sum_row_simd4::<B, N>(row, &src);
    }

    if flip {
        for (i, &src) in sums.iter().enumerate() {
            v[N - 1 - i] = src;
        }
    } else {
        *v = sums;
    }
}

#[inline(always)]
fn inv_adst4_simd4<B: DctSimd4>(v: &mut [B::V; 4]) {
    inv_mat_simd4::<B, 4>(v, &ADST4_KERNEL_ROWS, false);
}

#[inline(always)]
fn inv_adst8_simd4<B: DctSimd4>(v: &mut [B::V; 8]) {
    inv_mat_simd4::<B, 8>(v, &ADST8_KERNEL_ROWS, false);
}

#[inline(always)]
fn inv_adst16_simd4<B: DctSimd4>(v: &mut [B::V; 16]) {
    inv_mat_simd4::<B, 16>(v, &ADST16_KERNEL_ROWS, false);
}

#[inline(always)]
fn inv_flipadst4_simd4<B: DctSimd4>(v: &mut [B::V; 4]) {
    inv_mat_simd4::<B, 4>(v, &FLIPADST4_KERNEL_ROWS, false);
}

#[inline(always)]
fn inv_flipadst8_simd4<B: DctSimd4>(v: &mut [B::V; 8]) {
    inv_mat_simd4::<B, 8>(v, &ADST8_KERNEL_ROWS, true);
}

#[inline(always)]
fn inv_flipadst16_simd4<B: DctSimd4>(v: &mut [B::V; 16]) {
    inv_mat_simd4::<B, 16>(v, &FLIPADST16_KERNEL_ROWS, false);
}

#[inline(always)]
fn apply_tx4_simd4<B: DctSimd4, const KIND: usize>(v: &mut [B::V; 4]) {
    match KIND {
        TX_KIND_DCT => inv_dct4_simd4::<B>(v),
        TX_KIND_ADST => inv_adst4_simd4::<B>(v),
        TX_KIND_FLIPADST => inv_flipadst4_simd4::<B>(v),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn apply_tx8_simd4<B: DctSimd4, const KIND: usize>(v: &mut [B::V; 8]) {
    match KIND {
        TX_KIND_DCT => inv_dct8_simd4::<B>(v),
        TX_KIND_ADST => inv_adst8_simd4::<B>(v),
        TX_KIND_FLIPADST => inv_flipadst8_simd4::<B>(v),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn apply_tx16_simd4<B: DctSimd4, const KIND: usize>(v: &mut [B::V; 16]) {
    match KIND {
        TX_KIND_DCT => inv_dct16_simd4::<B>(v),
        TX_KIND_ADST => inv_adst16_simd4::<B>(v),
        TX_KIND_FLIPADST => inv_flipadst16_simd4::<B>(v),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn process_row_group_itx_x4<B: DctSimd4, const S: usize, const KIND: usize, C: ItxCoeff>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    if C::USE_WIDE_16BIT {
        process_row_group_itx_wide_x4::<B, S, S, false, KIND, C>(
            coeff, tmp, y, rnd, shift, min, max,
        );
        return;
    }

    match S {
        4 => {
            let mut v = load_coeff_rows_x4::<B, 4, C>(coeff, y);
            apply_tx4_simd4::<B, KIND>(&mut v);
            store_row_group_x4_clip::<B, 4>(tmp, y, &v, rnd, shift, min, max);
        }
        8 => {
            let mut v = load_coeff_rows_x4::<B, 8, C>(coeff, y);
            apply_tx8_simd4::<B, KIND>(&mut v);
            store_row_group_x4_clip::<B, 8>(tmp, y, &v, rnd, shift, min, max);
        }
        16 => {
            let mut v = load_coeff_rows_x4::<B, 16, C>(coeff, y);
            apply_tx16_simd4::<B, KIND>(&mut v);
            store_row_group_x4_clip::<B, 16>(tmp, y, &v, rnd, shift, min, max);
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn process_row_group_itx_wide_x8<
    B: DctSimd4,
    const W: usize,
    const H: usize,
    const RECT2: bool,
    const KIND: usize,
    C: ItxCoeff,
>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    use crate::itx_1d::DctWide;
    let s: [<B::Wide as DctWide>::In; W] = core::array::from_fn(|x| unsafe {
        if RECT2 {
            C::load_wide8_rect2::<B::Wide>(coeff, y + x * H)
        } else {
            C::load_wide8::<B::Wide>(coeff, y + x * H)
        }
    });
    unsafe {
        let clip = B::Wide::make_clip(rnd, shift, min, max);
        let zero = B::Wide::zero();
        let mut out = [zero; W];
        {
            let mut store = |m: usize, acc: <B::Wide as DctWide>::Acc| out[m] = acc;
            match (W, KIND) {
                (32, TX_KIND_DCT) => crate::itx_1d::dct32_wide::<B::Wide>(|j| s[j], &mut store),
                (16, TX_KIND_DCT) => crate::itx_1d::dct16_wide::<B::Wide>(|j| s[j], &mut store),
                (16, TX_KIND_ADST) => {
                    crate::itx_1d::adst16_wide::<B::Wide>(|j| s[j], &mut store, &ADST16_KW, false)
                }
                (16, TX_KIND_FLIPADST) => crate::itx_1d::adst16_wide::<B::Wide>(
                    |j| s[j],
                    &mut store,
                    &FLIPADST16_KW,
                    false,
                ),
                (8, TX_KIND_DCT) => {
                    crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], &mut store, &DCT8_KW, false)
                }
                (8, TX_KIND_ADST) => {
                    crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], &mut store, &ADST8_KW, false)
                }
                (8, TX_KIND_FLIPADST) => {
                    crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], &mut store, &ADST8_KW, true)
                }
                _ => unreachable!(),
            }
        }
        store_row_group_wide_x8_clip::<B::Wide, W>(tmp, y, &out, clip);
    }
}

#[inline(always)]
fn itx_dequant_rows_simd4<
    B: DctSimd4,
    const N: usize,
    const S: usize,
    C: ItxCoeff,
    const FIRST_KIND: usize,
>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(S == 4 || S == 8 || S == 16);
    debug_assert!(N <= coeff.len());
    debug_assert!(S * S <= N);
    debug_assert!(is_dct_adst_kind(FIRST_KIND));

    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];

    // Active 4-column groups (eob early-out), hoisted ahead of the transform so
    // the row pass can run 8 columns at a time where a wide kernel exists.
    let mut ngrp = 0usize;
    while ngrp < S / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let ncols = ngrp * 4;
    let rnd0 = (1 << shift0) >> 1;

    let mut y = 0usize;
    if C::USE_WIDE_16BIT && (S == 16 || S == 8) {
        while y + 8 <= ncols {
            process_row_group_itx_wide_x8::<B, S, S, false, FIRST_KIND, C>(
                coeff,
                tmp,
                y,
                rnd0,
                shift0,
                row_clip_min,
                row_clip_max,
            );
            y += 8;
        }
    }
    while y + 4 <= ncols {
        process_row_group_itx_x4::<B, S, FIRST_KIND, C>(
            coeff,
            tmp,
            y,
            rnd0,
            shift0,
            row_clip_min,
            row_clip_max,
        );
        y += 4;
    }

    while y < S {
        row_mut(tmp, y)[..S].fill(0);
        y += 1;
    }

    coeff[..S * S].fill(C::ZERO);
}

#[inline(always)]
fn itx_1d_wide_x4<B: DctSimd4, const S: usize, const KIND: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    x: usize,
) -> bool {
    use crate::itx_1d::DctWide;
    if !(S == 4 || S == 8 || S == 16 || S == 32) {
        return false;
    }
    let stride = ITX_TMP_STRIDE;
    let s: [<B::Wide as DctWide>::In; S] = {
        let src: &[i32] = &tmp[..];
        core::array::from_fn(|j| unsafe { B::Wide::load4_narrow(src, x + j * stride) })
    };
    let store = |m: usize, acc: <B::Wide as DctWide>::Acc| unsafe {
        B::Wide::store4(tmp, x + m * stride, acc)
    };
    match (S, KIND) {
        (32, TX_KIND_DCT) => crate::itx_1d::dct32_wide::<B::Wide>(|j| s[j], store),
        (16, TX_KIND_DCT) => crate::itx_1d::dct16_wide::<B::Wide>(|j| s[j], store),
        (16, TX_KIND_ADST) => {
            crate::itx_1d::adst16_wide::<B::Wide>(|j| s[j], store, &ADST16_KW, false)
        }
        (16, TX_KIND_FLIPADST) => {
            crate::itx_1d::adst16_wide::<B::Wide>(|j| s[j], store, &FLIPADST16_KW, false)
        }
        (8, TX_KIND_DCT) => crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], store, &DCT8_KW, false),
        (8, TX_KIND_ADST) => {
            crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], store, &ADST8_KW, false)
        }
        (8, TX_KIND_FLIPADST) => {
            crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], store, &ADST8_KW, true)
        }
        (4, TX_KIND_DCT) => crate::itx_1d::mat4_wide::<B::Wide>(|j| s[j], store, &DCT4_KW),
        (4, TX_KIND_ADST) => crate::itx_1d::mat4_wide::<B::Wide>(|j| s[j], store, &ADST4_KW),
        (4, TX_KIND_FLIPADST) => {
            crate::itx_1d::mat4_wide::<B::Wide>(|j| s[j], store, &FLIPADST4_KW)
        }
        _ => return false,
    }
    true
}

#[inline(always)]
fn itx_1d_x4<B: DctSimd4, const S: usize, const KIND: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    x: usize,
) {
    match S {
        4 => {
            let mut v = load_1d_x4::<B, 4>(tmp, x, ITX_TMP_STRIDE);
            apply_tx4_simd4::<B, KIND>(&mut v);
            store_1d_x4::<B, 4>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        8 => {
            let mut v = load_1d_x4::<B, 8>(tmp, x, ITX_TMP_STRIDE);
            apply_tx8_simd4::<B, KIND>(&mut v);
            store_1d_x4::<B, 8>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        16 => {
            let mut v = load_1d_x4::<B, 16>(tmp, x, ITX_TMP_STRIDE);
            apply_tx16_simd4::<B, KIND>(&mut v);
            store_1d_x4::<B, 16>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn itx_1d_wide_x8<B: DctSimd4, const S: usize, const KIND: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    x: usize,
) -> bool {
    use crate::itx_1d::DctWide;
    if !(S == 16 || S == 8) {
        return false;
    }
    let s: [<B::Wide as DctWide>::In; S] = {
        let src: &[i32] = &tmp[..];
        core::array::from_fn(|j| unsafe { B::Wide::load8_narrow(src, x + j * ITX_TMP_STRIDE) })
    };
    let store = |m: usize, acc: <B::Wide as DctWide>::Acc| unsafe {
        B::Wide::store8(tmp, x + m * ITX_TMP_STRIDE, acc)
    };
    match (S, KIND) {
        (16, TX_KIND_DCT) => crate::itx_1d::dct16_wide::<B::Wide>(|j| s[j], store),
        (16, TX_KIND_ADST) => {
            crate::itx_1d::adst16_wide::<B::Wide>(|j| s[j], store, &ADST16_KW, false)
        }
        (16, TX_KIND_FLIPADST) => {
            crate::itx_1d::adst16_wide::<B::Wide>(|j| s[j], store, &FLIPADST16_KW, false)
        }
        (8, TX_KIND_DCT) => crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], store, &DCT8_KW, false),
        (8, TX_KIND_ADST) => {
            crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], store, &ADST8_KW, false)
        }
        (8, TX_KIND_FLIPADST) => {
            crate::itx_1d::adst8_wide::<B::Wide>(|j| s[j], store, &ADST8_KW, true)
        }
        _ => return false,
    }
    true
}

#[inline(always)]
fn itx_dequant_simd4_core_mono<
    B: DctSimd4,
    const N: usize,
    const S: usize,
    C: ItxCoeff,
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
    debug_assert!(is_dct_adst_kind(FIRST_KIND));
    debug_assert!(is_dct_adst_kind(SECOND_KIND));

    if is_rect2 {
        itx_dequant_scalar_core_mono::<N, S, C, FIRST_KIND, SECOND_KIND>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
        return;
    }

    itx_dequant_rows_simd4::<B, N, S, C, FIRST_KIND>(
        coeff,
        tmp,
        eob,
        tx,
        shift0,
        row_clip_min,
        row_clip_max,
    );

    let mut x = 0usize;
    if C::USE_WIDE_16BIT && (S == 8 || S == 16) {
        while x + 8 <= S {
            if !itx_1d_wide_x8::<B, S, SECOND_KIND>(tmp, x) {
                break;
            }
            x += 8;
        }
    }
    while x + 4 <= S {
        if !(C::USE_WIDE_16BIT && itx_1d_wide_x4::<B, S, SECOND_KIND>(tmp, x)) {
            itx_1d_x4::<B, S, SECOND_KIND>(tmp, x);
        }
        x += 4;
    }
    while x < S {
        tx_1d_scalar_mono::<S, SECOND_KIND>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

#[inline(always)]
pub(crate) fn itx_dequant_simd4_core<B: DctSimd4, const N: usize, const S: usize, C: ItxCoeff>(
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
        itx_dequant_simd4_core_mono::<B, N, S, C, FK, SK>(
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub(crate) fn itx_dequant_simd4_core_sse41<
    B: DctSimd4,
    const N: usize,
    const S: usize,
    C: ItxCoeff,
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
        itx_dequant_simd4_core_mono::<B, N, S, C, FK, SK>(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn itx_dequant_simd4_core_avx2<
    B: DctSimd4,
    const N: usize,
    const S: usize,
    C: ItxCoeff,
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
        itx_dequant_simd4_core_mono::<B, N, S, C, FK, SK>(
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
fn dct_1d_x4<B: DctSimd4, const S: usize>(tmp: &mut [i32; ITX_TMP_PIXELS], x: usize) {
    match S {
        4 => {
            let mut v = load_1d_x4::<B, 4>(tmp, x, ITX_TMP_STRIDE);
            inv_dct4_simd4::<B>(&mut v);
            store_1d_x4::<B, 4>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        8 => {
            let mut v = load_1d_x4::<B, 8>(tmp, x, ITX_TMP_STRIDE);
            inv_dct8_simd4::<B>(&mut v);
            store_1d_x4::<B, 8>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        16 => {
            let s = load_1d_x4::<B, 16>(tmp, x, ITX_TMP_STRIDE);
            crate::itx_1d::dct16_flat_bylane::<B::V>(
                |j| s[j],
                |m, v| unsafe { B::store(tmp, x + m * ITX_TMP_STRIDE, v) },
            );
        }
        32 => {
            let s = load_1d_x4::<B, 32>(tmp, x, ITX_TMP_STRIDE);
            crate::itx_1d::dct32_flat::<B::V>(
                |j| s[j],
                |m, v| unsafe { B::store(tmp, x + m * ITX_TMP_STRIDE, v) },
            );
        }
        _ => unreachable!(),
    }
}

/// s16 8-wide second-pass column transform for the 16/32 sizes: loads 8 columns
/// per row from the i32 scratch (narrowing to s16), runs the widening-MAC DCT,
/// stores the s32 results back. Bit-exact to `dct_1d_x4`.
#[inline(always)]
fn dct_1d_wide_x8<W: crate::itx_1d::DctWide, const S: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    x: usize,
) {
    let stride = ITX_TMP_STRIDE;
    let s: [W::In; S] = {
        let src: &[i32] = &tmp[..];
        core::array::from_fn(|j| unsafe { W::load8_narrow(src, x + j * stride) })
    };
    let store = |m: usize, acc: W::Acc| unsafe { W::store8(tmp, x + m * stride, acc) };
    match S {
        16 => crate::itx_1d::dct16_wide::<W>(|j| s[j], store),
        32 => crate::itx_1d::dct32_wide::<W>(|j| s[j], store),
        _ => unreachable!(),
    }
}

#[inline(always)]
pub(crate) fn idct_dequant_simd4_core<B: DctSimd4, const N: usize, const S: usize, C: ItxCoeff>(
    coeff: &mut [C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        // This only occurs for the clamped rectangular 64-involving DCT_DCT
        // cases that reuse the square 32x32 core. Keep them on the same
        // SIMD row pipeline as true rectangular transforms instead of falling
        // back to scalar rows.
        idct_dequant_rows_rect_dct_simd4::<B, N, S, S, C>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    } else {
        idct_dequant_rows_dct_simd4::<B, N, S, C>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    let mut x = 0usize;
    if C::USE_WIDE_16BIT && (S == 16 || S == 32) {
        while x + 8 <= S {
            dct_1d_wide_x8::<B::Wide, S>(tmp, x);
            x += 8;
        }
    }
    while x + 4 <= S {
        if !(C::USE_WIDE_16BIT && itx_1d_wide_x4::<B, S, TX_KIND_DCT>(tmp, x)) {
            dct_1d_x4::<B, S>(tmp, x);
        }
        x += 4;
    }
    while x < S {
        dct_1d::<S>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_simd4_core_sse41<
    B: DctSimd4,
    const N: usize,
    const S: usize,
    C: ItxCoeff,
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
    if is_rect2 {
        idct_dequant_rows_rect_dct_simd4::<B, N, S, S, C>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    } else {
        idct_dequant_rows_dct_simd4::<B, N, S, C>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    let mut x = 0usize;
    if C::USE_WIDE_16BIT && (S == 16 || S == 32) {
        while x + 8 <= S {
            dct_1d_wide_x8::<B::Wide, S>(tmp, x);
            x += 8;
        }
    }
    while x + 4 <= S {
        if !(C::USE_WIDE_16BIT && itx_1d_wide_x4::<B, S, TX_KIND_DCT>(tmp, x)) {
            dct_1d_x4::<B, S>(tmp, x);
        }
        x += 4;
    }
    while x < S {
        dct_1d::<S>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_simd4_core_avx2<
    B: DctSimd4,
    const N: usize,
    const S: usize,
    C: ItxCoeff,
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
    if is_rect2 {
        idct_dequant_rows_rect_dct_simd4::<B, N, S, S, C>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    } else {
        idct_dequant_rows_dct_simd4::<B, N, S, C>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    let mut x = 0usize;
    if C::USE_WIDE_16BIT && (S == 16 || S == 32) {
        while x + 8 <= S {
            dct_1d_wide_x8::<B::Wide, S>(tmp, x);
            x += 8;
        }
    }
    while x + 4 <= S {
        if !(C::USE_WIDE_16BIT && itx_1d_wide_x4::<B, S, TX_KIND_DCT>(tmp, x)) {
            dct_1d_x4::<B, S>(tmp, x);
        }
        x += 4;
    }
    while x < S {
        dct_1d::<S>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

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

/// Rectangular coefficient gather: `W` lanes at column stride `H`, 4 rows deep.
#[inline(always)]
fn load_coeff_rows_rect_x4<B: DctSimd4, const W: usize, const H: usize, C: ItxCoeff>(
    coeff: &[C],
    y: usize,
) -> [B::V; W] {
    unsafe {
        let zero = B::zero();
        let mut out = [zero; W];
        for (x, dst) in out.iter_mut().enumerate() {
            *dst = C::load_simd4::<B>(coeff, y + x * H);
        }
        out
    }
}

/// One group of 4 rows: a `W`-point DCT applied across the `W` gathered lanes.
#[inline(always)]
fn process_row_group_rect_x4<B: DctSimd4, const W: usize, const H: usize, C: ItxCoeff>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    is_rect2: bool,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    if C::USE_WIDE_16BIT {
        if is_rect2 {
            process_row_group_itx_wide_x4::<B, W, H, true, TX_KIND_DCT, C>(
                coeff, tmp, y, rnd, shift, min, max,
            );
        } else {
            process_row_group_itx_wide_x4::<B, W, H, false, TX_KIND_DCT, C>(
                coeff, tmp, y, rnd, shift, min, max,
            );
        }
        return;
    }

    match W {
        4 => {
            let mut v = load_coeff_rows_rect_x4::<B, 4, H, C>(coeff, y);
            if is_rect2 {
                for v in &mut v {
                    *v = unsafe { B::rect2_scale(*v) };
                }
            }
            inv_dct4_simd4::<B>(&mut v);
            store_row_group_x4_clip::<B, 4>(tmp, y, &v, rnd, shift, min, max);
        }
        8 => {
            let mut v = load_coeff_rows_rect_x4::<B, 8, H, C>(coeff, y);
            if is_rect2 {
                for v in &mut v {
                    *v = unsafe { B::rect2_scale(*v) };
                }
            }
            inv_dct8_simd4::<B>(&mut v);
            store_row_group_x4_clip::<B, 8>(tmp, y, &v, rnd, shift, min, max);
        }
        16 => {
            let mut v = load_coeff_rows_rect_x4::<B, 16, H, C>(coeff, y);
            if is_rect2 {
                for v in &mut v {
                    *v = unsafe { B::rect2_scale(*v) };
                }
            }
            inv_dct16_simd4::<B>(&mut v);
            store_row_group_x4_clip::<B, 16>(tmp, y, &v, rnd, shift, min, max);
        }
        32 => {
            let load = |j: usize| unsafe { C::load_simd4::<B>(coeff, y + j * H) };
            let mut row = [unsafe { B::zero() }; 32];
            for j in 0..32 {
                row[j] = load(j);
                if is_rect2 {
                    row[j] = unsafe { B::rect2_scale(row[j]) };
                }
            }
            let zero = unsafe { B::zero() };
            let mut out = [zero; 32];
            crate::itx_1d::dct32_flat::<B::V>(|j| row[j], |m, v| out[m] = v);
            store_row_group_x4_clip::<B, 32>(tmp, y, &out, rnd, shift, min, max);
        }
        _ => unreachable!(),
    }
}

/// SIMD row pass for the non-rect2 case (mirrors `idct_dequant_rows_dct_simd4`
/// with separate `W`/`H`).
#[inline(always)]
fn idct_dequant_rows_rect_dct_simd4<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());

    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];

    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let rnd0 = (1 << shift0) >> 1;

    let mut y = 0usize;
    if C::USE_WIDE_16BIT && (W == 8 || W == 16 || W == 32) {
        if is_rect2 {
            while y + 8 <= nrows {
                process_row_group_itx_wide_x8::<B, W, H, true, TX_KIND_DCT, C>(
                    coeff,
                    tmp,
                    y,
                    rnd0,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                );
                y += 8;
            }
        } else {
            while y + 8 <= nrows {
                process_row_group_itx_wide_x8::<B, W, H, false, TX_KIND_DCT, C>(
                    coeff,
                    tmp,
                    y,
                    rnd0,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                );
                y += 8;
            }
        }
    }
    while y + 4 <= nrows {
        process_row_group_rect_x4::<B, W, H, C>(
            coeff,
            tmp,
            y,
            is_rect2,
            rnd0,
            shift0,
            row_clip_min,
            row_clip_max,
        );
        y += 4;
    }

    while y < H {
        row_mut(tmp, y)[..W].fill(0);
        y += 1;
    }

    coeff[..W * H].fill(C::ZERO);
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

/// Column pass: an `H`-point DCT down each of the `W` columns, 4 columns at a
/// time, with a scalar tail.
#[inline(always)]
fn rect_col_pass<B: DctSimd4, const W: usize, const H: usize, C: ItxCoeff>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
) {
    let mut x = 0usize;
    // Reuses the same widening-MAC column transform validated in the square
    // path (`dct_1d_wide_x8::<_, H>` is identical codegen); only the column
    // count W differs.
    if C::USE_WIDE_16BIT && (H == 16 || H == 32) {
        while x + 8 <= W {
            dct_1d_wide_x8::<B::Wide, H>(tmp, x);
            x += 8;
        }
    }
    while x + 4 <= W {
        if !(C::USE_WIDE_16BIT && itx_1d_wide_x4::<B, H, TX_KIND_DCT>(tmp, x)) {
            dct_1d_x4::<B, H>(tmp, x);
        }
        x += 4;
    }
    while x < W {
        dct_1d::<H>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

/// SIMD-structured rectangular DCT_DCT core (used by the NEON/SSE backends).
#[inline(always)]
pub(crate) fn idct_dequant_rect_simd4_core<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
    idct_dequant_rows_rect_dct_simd4::<B, N, W, H, C>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
    rect_col_pass::<B, W, H, C>(tmp);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub(crate) fn idct_dequant_rect_simd4_core_sse41<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
    idct_dequant_rows_rect_dct_simd4::<B, N, W, H, C>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
    rect_col_pass::<B, W, H, C>(tmp);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_rect_simd4_core_avx2<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
    idct_dequant_rows_rect_dct_simd4::<B, N, W, H, C>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
    rect_col_pass::<B, W, H, C>(tmp);
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

#[inline(always)]
fn process_row_group_itx_rect_x4<
    B: DctSimd4,
    const W: usize,
    const H: usize,
    const KIND: usize,
    C: ItxCoeff,
>(
    coeff: &[C],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    is_rect2: bool,
    rnd: i32,
    shift: i32,
    min: i32,
    max: i32,
) {
    if C::USE_WIDE_16BIT {
        if is_rect2 {
            process_row_group_itx_wide_x4::<B, W, H, true, KIND, C>(
                coeff, tmp, y, rnd, shift, min, max,
            );
        } else {
            process_row_group_itx_wide_x4::<B, W, H, false, KIND, C>(
                coeff, tmp, y, rnd, shift, min, max,
            );
        }
        return;
    }

    match W {
        4 => {
            let mut v = load_coeff_rows_rect_x4::<B, 4, H, C>(coeff, y);
            if is_rect2 {
                for v in &mut v {
                    *v = unsafe { B::rect2_scale(*v) };
                }
            }
            apply_tx4_simd4::<B, KIND>(&mut v);
            store_row_group_x4_clip::<B, 4>(tmp, y, &v, rnd, shift, min, max);
        }
        8 => {
            let mut v = load_coeff_rows_rect_x4::<B, 8, H, C>(coeff, y);
            if is_rect2 {
                for v in &mut v {
                    *v = unsafe { B::rect2_scale(*v) };
                }
            }
            apply_tx8_simd4::<B, KIND>(&mut v);
            store_row_group_x4_clip::<B, 8>(tmp, y, &v, rnd, shift, min, max);
        }
        16 => {
            let mut v = load_coeff_rows_rect_x4::<B, 16, H, C>(coeff, y);
            if is_rect2 {
                for v in &mut v {
                    *v = unsafe { B::rect2_scale(*v) };
                }
            }
            apply_tx16_simd4::<B, KIND>(&mut v);
            store_row_group_x4_clip::<B, 16>(tmp, y, &v, rnd, shift, min, max);
        }
        _ => unreachable!(),
    }
}

/// Kind-aware SIMD row pass (non-rect2), generalized to `W`/`H`.
#[inline(always)]
fn itx_dequant_rows_rect_simd4<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
    const FIRST_KIND: usize,
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
    debug_assert!(W == 4 || W == 8 || W == 16);
    debug_assert!(H == 4 || H == 8 || H == 16);
    debug_assert!(W * H <= N && N <= coeff.len());
    debug_assert!(is_dct_adst_kind(FIRST_KIND));

    let coeff = &mut coeff[..N];
    let off = usize::from(LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &LAST_EOB_PER_COL.table[off..];

    // Active 4-row groups (eob early-out), hoisted so the row pass can run a
    // `W`-point transform across 8 rows at a time where a wide kernel exists.
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let rnd0 = (1 << shift0) >> 1;

    let mut y = 0usize;
    if C::USE_WIDE_16BIT && (W == 8 || W == 16) {
        if is_rect2 {
            while y + 8 <= nrows {
                process_row_group_itx_wide_x8::<B, W, H, true, FIRST_KIND, C>(
                    coeff,
                    tmp,
                    y,
                    rnd0,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                );
                y += 8;
            }
        } else {
            while y + 8 <= nrows {
                process_row_group_itx_wide_x8::<B, W, H, false, FIRST_KIND, C>(
                    coeff,
                    tmp,
                    y,
                    rnd0,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                );
                y += 8;
            }
        }
    }
    while y + 4 <= nrows {
        process_row_group_itx_rect_x4::<B, W, H, FIRST_KIND, C>(
            coeff,
            tmp,
            y,
            is_rect2,
            rnd0,
            shift0,
            row_clip_min,
            row_clip_max,
        );
        y += 4;
    }

    while y < H {
        row_mut(tmp, y)[..W].fill(0);
        y += 1;
    }

    coeff[..W * H].fill(C::ZERO);
}

/// Pure-scalar kind-aware rectangular core (rect2 sizes + universal fallback).
/// Mirrors the generic path: scalar rows with rect2 scaling, scalar columns.
fn itx_dequant_rect_scalar_core_mono<
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

/// SIMD-structured kind-aware rectangular core (used by NEON/SSE). Rect2 goes
/// fully scalar, exactly as the square `itx_dequant_simd4_core` does.
#[inline(always)]
fn itx_dequant_rect_simd4_core_mono<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
    debug_assert!(W == 4 || W == 8 || W == 16);
    debug_assert!(H == 4 || H == 8 || H == 16);
    debug_assert!(is_dct_adst_kind(FIRST_KIND));
    debug_assert!(is_dct_adst_kind(SECOND_KIND));

    itx_dequant_rows_rect_simd4::<B, N, W, H, C, FIRST_KIND>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );

    let mut x = 0usize;
    // Column pass: H-point transform down each of the W columns. Reuses the same
    // widening kernels as the square path (only the column count W differs).
    if C::USE_WIDE_16BIT && (H == 8 || H == 16) {
        while x + 8 <= W {
            if !itx_1d_wide_x8::<B, H, SECOND_KIND>(tmp, x) {
                break;
            }
            x += 8;
        }
    }
    while x + 4 <= W {
        if !(C::USE_WIDE_16BIT && itx_1d_wide_x4::<B, H, SECOND_KIND>(tmp, x)) {
            itx_1d_x4::<B, H, SECOND_KIND>(tmp, x);
        }
        x += 4;
    }
    while x < W {
        tx_1d_scalar_mono::<H, SECOND_KIND>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

/// SIMD-structured kind-aware rectangular core (used by NEON/SSE).
#[inline(always)]
pub(crate) fn itx_dequant_rect_simd4_core<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
        itx_dequant_rect_simd4_core_mono::<B, N, W, H, C, FK, SK>(
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
pub(crate) fn itx_dequant_rect_simd4_core_sse41<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
        itx_dequant_rect_simd4_core_mono::<B, N, W, H, C, FK, SK>(
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn itx_dequant_rect_simd4_core_avx2<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
    C: ItxCoeff,
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
        itx_dequant_rect_simd4_core_mono::<B, N, W, H, C, FK, SK>(
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

// i32 coefficient dispatch is shared by 8-bit legacy-i32 and high-bit-depth.
// High-bit-depth must not bypass this resolver: the SIMD backends use i32
// lanes for `C = i32` (`ItxCoeff::USE_WIDE_16BIT == false`), so no s16
// narrowing / pmaddwd widening kernels are reached for hbd coefficients.
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
