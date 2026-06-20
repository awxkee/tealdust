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
use crate::scan::LAST_EOB_PER_COL;
use std::convert::TryInto;
use std::sync::OnceLock;

pub(crate) const ITX_TMP_STRIDE: usize = 32;
pub(crate) const ITX_TMP_PIXELS: usize = ITX_TMP_STRIDE * ITX_TMP_STRIDE;

pub(crate) type IdctDequantFn<const N: usize> = fn(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
);

pub(crate) type IadstDequantFn<const N: usize> = fn(
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

pub(crate) const TX_KIND_DCT: usize = 0;
pub(crate) const TX_KIND_ADST: usize = 2;
pub(crate) const TX_KIND_FLIPADST: usize = 3;

#[inline(always)]
pub(crate) fn is_dct_adst_kind(kind: usize) -> bool {
    matches!(kind, TX_KIND_DCT | TX_KIND_ADST | TX_KIND_FLIPADST)
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
        idct_dequant_scalar_core::<16, 4>(
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
        idct_dequant_scalar_core::<64, 8>(
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
        idct_dequant_scalar_core::<256, 16>(
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
        idct_dequant_scalar_core::<1024, 32>(
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
        idct_dequant_scalar_core::<1024, 32>(
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
        itx_dequant_scalar_core::<16, 4>(
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
        itx_dequant_scalar_core::<64, 8>(
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
        itx_dequant_scalar_core::<256, 16>(
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

    if let Some(f8) = TX1D_FNS_X8[tx_size][0] {
        f8(tmp, x, ITX_TMP_STRIDE);
        true
    } else {
        false
    }
}

pub(crate) fn idct_dequant_scalar_core<const N: usize, const S: usize>(
    coeff: &mut [i32],
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
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        let dst_row = row_mut(tmp, y);
        for (x, dst) in dst_row[..S].iter_mut().enumerate() {
            let v = coeff[y + x * S];
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

    coeff[..S * S].fill(0);

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
fn tx_1d_scalar<const S: usize>(c: &mut [i32], stride: usize, kind: usize) {
    let f = TX1D_FNS[tx_size_idx::<S>()][kind].expect("unsupported 1D transform");
    f(c, stride);
}

pub(crate) fn itx_dequant_scalar_core<const N: usize, const S: usize>(
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
    debug_assert!(S == 4 || S == 8 || S == 16);
    debug_assert!(N <= coeff.len());
    debug_assert!(S * S <= N);
    debug_assert!(is_dct_adst_kind(first_kind));
    debug_assert!(is_dct_adst_kind(second_kind));

    let coeff = &mut coeff[..N];
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        let dst_row = row_mut(tmp, y);
        for (x, dst) in dst_row[..S].iter_mut().enumerate() {
            let v = coeff[y + x * S];
            *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }

        tx_1d_scalar::<S>(dst_row, 1, first_kind);
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

    coeff[..S * S].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..S {
        crate::filter::row_clip(row_mut(tmp, y), S, rnd0, shift0, row_clip_min, row_clip_max);
    }

    let mut x = 0usize;
    if let Some(f8) = TX1D_FNS_X8[tx_size_idx::<S>()][second_kind] {
        while x + 8 <= S {
            f8(tmp, x, ITX_TMP_STRIDE);
            x += 8;
        }
    }
    while x < S {
        tx_1d_scalar::<S>(&mut tmp[x..], ITX_TMP_STRIDE, second_kind);
        x += 1;
    }
}

pub(crate) fn idct_dequant_rows_dct<const N: usize, const S: usize>(
    coeff: &mut [i32],
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
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        let dst_row = row_mut(tmp, y);
        for (x, dst) in dst_row[..S].iter_mut().enumerate() {
            let v = coeff[y + x * S];
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

    coeff[..S * S].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..S {
        crate::filter::row_clip(row_mut(tmp, y), S, rnd0, shift0, row_clip_min, row_clip_max);
    }
}

pub(crate) trait DctSimd4 {
    type V: Copy + crate::itx_1d::DctLane;

    unsafe fn zero() -> Self::V;
    unsafe fn splat(v: i32) -> Self::V;
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V;
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V;
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V);
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V;
    unsafe fn to_array(v: Self::V) -> [i32; 4];
}

#[inline(always)]
fn vmulc<B: DctSimd4>(v: B::V, k: i32) -> B::V {
    unsafe { B::mul(v, B::splat(k)) }
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
fn load_coeff_rows_x4<B: DctSimd4, const S: usize>(coeff: &[i32], y: usize) -> [B::V; S] {
    unsafe {
        let zero = B::zero();
        let mut out = [zero; S];
        for (x, dst) in out.iter_mut().enumerate() {
            *dst = B::load_slice(coeff, y + x * S);
        }
        out
    }
}

#[inline(always)]
fn store_row_group_x4<B: DctSimd4, const S: usize>(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    v: &[B::V; S],
) {
    unsafe {
        for (x, &vx) in v.iter().enumerate() {
            let lanes = B::to_array(vx);
            tmp[y * ITX_TMP_STRIDE + x] = lanes[0];
            tmp[(y + 1) * ITX_TMP_STRIDE + x] = lanes[1];
            tmp[(y + 2) * ITX_TMP_STRIDE + x] = lanes[2];
            tmp[(y + 3) * ITX_TMP_STRIDE + x] = lanes[3];
        }
    }
}

#[inline(always)]
fn process_row_group_x4<B: DctSimd4, const S: usize>(
    coeff: &[i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
) {
    match S {
        4 => {
            let mut v = load_coeff_rows_x4::<B, 4>(coeff, y);
            inv_dct4_simd4::<B>(&mut v);
            store_row_group_x4::<B, 4>(tmp, y, &v);
        }
        8 => {
            let mut v = load_coeff_rows_x4::<B, 8>(coeff, y);
            inv_dct8_simd4::<B>(&mut v);
            store_row_group_x4::<B, 8>(tmp, y, &v);
        }
        16 => {
            let load = |j: usize| unsafe { B::load_slice(coeff, y + j * 16) };
            let store = |m: usize, v: B::V| {
                let l = unsafe { B::to_array(v) };
                tmp[y * ITX_TMP_STRIDE + m] = l[0];
                tmp[(y + 1) * ITX_TMP_STRIDE + m] = l[1];
                tmp[(y + 2) * ITX_TMP_STRIDE + m] = l[2];
                tmp[(y + 3) * ITX_TMP_STRIDE + m] = l[3];
            };
            crate::itx_1d::dct16_flat::<B::V>(load, store);
        }
        32 => {
            let load = |j: usize| unsafe { B::load_slice(coeff, y + j * 32) };
            let store = |m: usize, v: B::V| {
                let l = unsafe { B::to_array(v) };
                tmp[y * ITX_TMP_STRIDE + m] = l[0];
                tmp[(y + 1) * ITX_TMP_STRIDE + m] = l[1];
                tmp[(y + 2) * ITX_TMP_STRIDE + m] = l[2];
                tmp[(y + 3) * ITX_TMP_STRIDE + m] = l[3];
            };
            crate::itx_1d::dct32_flat::<B::V>(load, store);
        }
        _ => unreachable!(),
    }
}

fn idct_dequant_rows_dct_simd4<B: DctSimd4, const N: usize, const S: usize>(
    coeff: &mut [i32],
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
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        process_row_group_x4::<B, S>(coeff, tmp, y);
        y += 4;

        if eob > last_eob[ei] as i32 {
            ei += 1;
        } else {
            break;
        }
    }

    while y < S {
        row_mut(tmp, y)[..S].fill(0);
        y += 1;
    }

    coeff[..S * S].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..S {
        crate::filter::row_clip(row_mut(tmp, y), S, rnd0, shift0, row_clip_min, row_clip_max);
    }
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
    crate::itx_1d::dct16_flat::<B::V>(|j| s[j], |m, x| v[m] = x);
}

#[inline(always)]
fn inv_dct32_simd4<B: DctSimd4>(v: &mut [B::V; 32]) {
    let s = *v;
    crate::itx_1d::dct32_flat::<B::V>(|j| s[j], |m, x| v[m] = x);
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
fn apply_tx4_simd4<B: DctSimd4>(v: &mut [B::V; 4], kind: usize) {
    match kind {
        TX_KIND_DCT => inv_dct4_simd4::<B>(v),
        TX_KIND_ADST => inv_adst4_simd4::<B>(v),
        TX_KIND_FLIPADST => inv_flipadst4_simd4::<B>(v),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn apply_tx8_simd4<B: DctSimd4>(v: &mut [B::V; 8], kind: usize) {
    match kind {
        TX_KIND_DCT => inv_dct8_simd4::<B>(v),
        TX_KIND_ADST => inv_adst8_simd4::<B>(v),
        TX_KIND_FLIPADST => inv_flipadst8_simd4::<B>(v),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn apply_tx16_simd4<B: DctSimd4>(v: &mut [B::V; 16], kind: usize) {
    match kind {
        TX_KIND_DCT => inv_dct16_simd4::<B>(v),
        TX_KIND_ADST => inv_adst16_simd4::<B>(v),
        TX_KIND_FLIPADST => inv_flipadst16_simd4::<B>(v),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn process_row_group_itx_x4<B: DctSimd4, const S: usize>(
    coeff: &[i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    kind: usize,
) {
    match S {
        4 => {
            let mut v = load_coeff_rows_x4::<B, 4>(coeff, y);
            apply_tx4_simd4::<B>(&mut v, kind);
            store_row_group_x4::<B, 4>(tmp, y, &v);
        }
        8 => {
            let mut v = load_coeff_rows_x4::<B, 8>(coeff, y);
            apply_tx8_simd4::<B>(&mut v, kind);
            store_row_group_x4::<B, 8>(tmp, y, &v);
        }
        16 => {
            let mut v = load_coeff_rows_x4::<B, 16>(coeff, y);
            apply_tx16_simd4::<B>(&mut v, kind);
            store_row_group_x4::<B, 16>(tmp, y, &v);
        }
        _ => unreachable!(),
    }
}

fn itx_dequant_rows_simd4<B: DctSimd4, const N: usize, const S: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
) {
    debug_assert!(S == 4 || S == 8 || S == 16);
    debug_assert!(N <= coeff.len());
    debug_assert!(S * S <= N);
    debug_assert!(is_dct_adst_kind(first_kind));

    let coeff = &mut coeff[..N];
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        process_row_group_itx_x4::<B, S>(coeff, tmp, y, first_kind);
        y += 4;

        if eob > last_eob[ei] as i32 {
            ei += 1;
        } else {
            break;
        }
    }

    while y < S {
        row_mut(tmp, y)[..S].fill(0);
        y += 1;
    }

    coeff[..S * S].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..S {
        crate::filter::row_clip(row_mut(tmp, y), S, rnd0, shift0, row_clip_min, row_clip_max);
    }
}

#[inline(always)]
fn itx_1d_x4<B: DctSimd4, const S: usize>(tmp: &mut [i32; ITX_TMP_PIXELS], x: usize, kind: usize) {
    match S {
        4 => {
            let mut v = load_1d_x4::<B, 4>(tmp, x, ITX_TMP_STRIDE);
            apply_tx4_simd4::<B>(&mut v, kind);
            store_1d_x4::<B, 4>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        8 => {
            let mut v = load_1d_x4::<B, 8>(tmp, x, ITX_TMP_STRIDE);
            apply_tx8_simd4::<B>(&mut v, kind);
            store_1d_x4::<B, 8>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        16 => {
            let mut v = load_1d_x4::<B, 16>(tmp, x, ITX_TMP_STRIDE);
            apply_tx16_simd4::<B>(&mut v, kind);
            store_1d_x4::<B, 16>(tmp, x, ITX_TMP_STRIDE, &v);
        }
        _ => unreachable!(),
    }
}

pub(crate) fn itx_dequant_simd4_core<B: DctSimd4, const N: usize, const S: usize>(
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
    debug_assert!(S == 4 || S == 8 || S == 16);
    debug_assert!(is_dct_adst_kind(first_kind));
    debug_assert!(is_dct_adst_kind(second_kind));

    if is_rect2 {
        itx_dequant_scalar_core::<N, S>(
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
        return;
    }

    itx_dequant_rows_simd4::<B, N, S>(
        coeff,
        tmp,
        eob,
        tx,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
    );

    let mut x = 0usize;
    while x + 4 <= S {
            itx_1d_x4::<B, S>(tmp, x, second_kind);
        x += 4;
    }
    while x < S {
        tx_1d_scalar::<S>(&mut tmp[x..], ITX_TMP_STRIDE, second_kind);
        x += 1;
    }
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
            crate::itx_1d::dct16_flat::<B::V>(
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

#[inline(always)]
pub(crate) fn idct_dequant_simd4_core<B: DctSimd4, const N: usize, const S: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_rows_dct::<N, S>(
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
        idct_dequant_rows_dct_simd4::<B, N, S>(
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
    while x + 4 <= S {
        dct_1d_x4::<B, S>(tmp, x);
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
fn load_coeff_rows_rect_x4<B: DctSimd4, const W: usize, const H: usize>(
    coeff: &[i32],
    y: usize,
) -> [B::V; W] {
    unsafe {
        let zero = B::zero();
        let mut out = [zero; W];
        for (x, dst) in out.iter_mut().enumerate() {
            *dst = B::load_slice(coeff, y + x * H);
        }
        out
    }
}

/// One group of 4 rows: a `W`-point DCT applied across the `W` gathered lanes.
#[inline(always)]
fn process_row_group_rect_x4<B: DctSimd4, const W: usize, const H: usize>(
    coeff: &[i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
) {
    match W {
        4 => {
            let mut v = load_coeff_rows_rect_x4::<B, 4, H>(coeff, y);
            inv_dct4_simd4::<B>(&mut v);
            store_row_group_x4::<B, 4>(tmp, y, &v);
        }
        8 => {
            let mut v = load_coeff_rows_rect_x4::<B, 8, H>(coeff, y);
            inv_dct8_simd4::<B>(&mut v);
            store_row_group_x4::<B, 8>(tmp, y, &v);
        }
        16 => {
            let mut v = load_coeff_rows_rect_x4::<B, 16, H>(coeff, y);
            inv_dct16_simd4::<B>(&mut v);
            store_row_group_x4::<B, 16>(tmp, y, &v);
        }
        32 => {
            let mut v = load_coeff_rows_rect_x4::<B, 32, H>(coeff, y);
            inv_dct32_simd4::<B>(&mut v);
            store_row_group_x4::<B, 32>(tmp, y, &v);
        }
        _ => unreachable!(),
    }
}

/// SIMD row pass for the non-rect2 case (mirrors `idct_dequant_rows_dct_simd4`
/// with separate `W`/`H`).
fn idct_dequant_rows_rect_dct_simd4<B: DctSimd4, const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());

    let coeff = &mut coeff[..N];
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        process_row_group_rect_x4::<B, W, H>(coeff, tmp, y);
        y += 4;
        if eob > last_eob[ei] as i32 {
            ei += 1;
        } else {
            break;
        }
    }

    while y < H {
        row_mut(tmp, y)[..W].fill(0);
        y += 1;
    }

    coeff[..W * H].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..H {
        crate::filter::row_clip(row_mut(tmp, y), W, rnd0, shift0, row_clip_min, row_clip_max);
    }
}

/// Scalar row pass (used for the rect2 sizes, mirroring the generic path's
/// `tx_class == 0` loop with the `* 181 + 128 >> 8` rect2 scaling).
fn idct_dequant_rows_rect_dct_scalar<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    let coeff = &mut coeff[..N];
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut row = 0usize;

    loop {
        let tmp_row = row_mut(tmp, row);
        for (x, dst) in tmp_row[..W].iter_mut().enumerate() {
            let v = coeff[row + x * H];
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

    coeff[..W * H].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..H {
        crate::filter::row_clip(row_mut(tmp, y), W, rnd0, shift0, row_clip_min, row_clip_max);
    }
}

/// Column pass: an `H`-point DCT down each of the `W` columns, 4 columns at a
/// time, with a scalar tail.
#[inline(always)]
fn rect_col_pass<B: DctSimd4, const W: usize, const H: usize>(tmp: &mut [i32; ITX_TMP_PIXELS]) {
    let mut x = 0usize;
    while x + 4 <= W {
        dct_1d_x4::<B, H>(tmp, x);
        x += 4;
    }
    while x < W {
        dct_1d::<H>(&mut tmp[x..], ITX_TMP_STRIDE);
        x += 1;
    }
}

/// SIMD-structured rectangular DCT_DCT core (used by the NEON/SSE backends).
pub(crate) fn idct_dequant_rect_simd4_core<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_rows_rect_dct_scalar::<N, W, H>(
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
        idct_dequant_rows_rect_dct_simd4::<B, N, W, H>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }
    rect_col_pass::<B, W, H>(tmp);
}

/// Pure-scalar rectangular DCT_DCT core (the universal fallback). The column
/// pass uses the scalar `dct_1d`, matching the generic path exactly.
pub(crate) fn idct_dequant_rect_scalar_core<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_rows_rect_dct_scalar::<N, W, H>(
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

// ===========================================================================
// Non-square (rectangular) ADST / mixed-type cores (DCT / ADST / FLIPADST in
// either dimension). These mirror the square `itx_dequant_simd4_core`,
// generalized to independent row width `W` and column height `H`. As in the
// square case, the rect2 sizes go fully scalar; the non-rect2 sizes use
// kind-aware SIMD rows and an `H`-point 4-wide column pass. ADST/FLIPADST are
// only defined for dims <= 16, so `W` and `H` are always in {4, 8, 16} here
// (which is exactly what `apply_tx*_simd4` and `itx_1d_x4` support).
// ===========================================================================

/// One group of 4 rows: a kind-aware `W`-point transform across the gathered
/// lanes (column-major gather at stride `H`).
#[inline(always)]
unsafe fn process_row_group_itx_rect_x4<B: DctSimd4, const W: usize, const H: usize>(
    coeff: &[i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    y: usize,
    kind: usize,
) {
    match W {
        4 => {
            let mut v = load_coeff_rows_rect_x4::<B, 4, H>(coeff, y);
            apply_tx4_simd4::<B>(&mut v, kind);
            store_row_group_x4::<B, 4>(tmp, y, &v);
        }
        8 => {
            let mut v = load_coeff_rows_rect_x4::<B, 8, H>(coeff, y);
            apply_tx8_simd4::<B>(&mut v, kind);
            store_row_group_x4::<B, 8>(tmp, y, &v);
        }
        16 => {
            let mut v = load_coeff_rows_rect_x4::<B, 16, H>(coeff, y);
            apply_tx16_simd4::<B>(&mut v, kind);
            store_row_group_x4::<B, 16>(tmp, y, &v);
        }
        _ => unreachable!(),
    }
}

/// Kind-aware SIMD row pass (non-rect2), generalized to `W`/`H`.
fn itx_dequant_rows_rect_simd4<B: DctSimd4, const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
) {
    debug_assert!(W == 4 || W == 8 || W == 16);
    debug_assert!(H == 4 || H == 8 || H == 16);
    debug_assert!(W * H <= N && N <= coeff.len());
    debug_assert!(is_dct_adst_kind(first_kind));

    let coeff = &mut coeff[..N];
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut y = 0usize;

    loop {
        unsafe {
            process_row_group_itx_rect_x4::<B, W, H>(coeff, tmp, y, first_kind);
        }
        y += 4;
        if eob > last_eob[ei] as i32 {
            ei += 1;
        } else {
            break;
        }
    }

    while y < H {
        row_mut(tmp, y)[..W].fill(0);
        y += 1;
    }

    coeff[..W * H].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..H {
        crate::filter::row_clip(row_mut(tmp, y), W, rnd0, shift0, row_clip_min, row_clip_max);
    }
}

/// Pure-scalar kind-aware rectangular core (rect2 sizes + universal fallback).
/// Mirrors the generic path: scalar rows with rect2 scaling, scalar columns.
pub(crate) fn itx_dequant_rect_scalar_core<const N: usize, const W: usize, const H: usize>(
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
    debug_assert!(is_dct_adst_kind(first_kind));
    debug_assert!(is_dct_adst_kind(second_kind));

    let coeff = &mut coeff[..N];
    let off = LAST_EOB_PER_COL.offset[tx] as usize;
    let last_eob = &LAST_EOB_PER_COL.table[off..];
    let mut ei = 0usize;
    let mut row = 0usize;

    loop {
        let dst_row = row_mut(tmp, row);
        for (x, dst) in dst_row[..W].iter_mut().enumerate() {
            let v = coeff[row + x * H];
            *dst = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
        }
        tx_1d_scalar::<W>(dst_row, 1, first_kind);
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

    coeff[..W * H].fill(0);

    let rnd0 = (1 << shift0) >> 1;
    for y in 0..H {
        crate::filter::row_clip(row_mut(tmp, y), W, rnd0, shift0, row_clip_min, row_clip_max);
    }

    for x in 0..W {
        tx_1d_scalar::<H>(&mut tmp[x..], ITX_TMP_STRIDE, second_kind);
    }
}

/// SIMD-structured kind-aware rectangular core (used by NEON/SSE). Rect2 goes
/// fully scalar, exactly as the square `itx_dequant_simd4_core` does.
pub(crate) fn itx_dequant_rect_simd4_core<
    B: DctSimd4,
    const N: usize,
    const W: usize,
    const H: usize,
>(
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
    debug_assert!(W == 4 || W == 8 || W == 16);
    debug_assert!(H == 4 || H == 8 || H == 16);

    if is_rect2 {
        itx_dequant_rect_scalar_core::<N, W, H>(
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
        return;
    }

    itx_dequant_rows_rect_simd4::<B, N, W, H>(
        coeff,
        tmp,
        eob,
        tx,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
    );

    let mut x = 0usize;
    while x + 4 <= W {
            itx_1d_x4::<B, H>(tmp, x, second_kind);
        x += 4;
    }
    while x < W {
        tx_1d_scalar::<H>(&mut tmp[x..], ITX_TMP_STRIDE, second_kind);
        x += 1;
    }
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
pub(crate) fn idct_dequant_4x4() -> IdctDequantFn<16> {
    *DEQUANT_4X4.get_or_init(|| {
        let mut f = idct_dequant_4x4_scalar as IdctDequantFn<16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_4x4_neon as IdctDequantFn<16>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x4_sse41 as IdctDequantFn<16>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x8() -> IdctDequantFn<64> {
    *DEQUANT_8X8.get_or_init(|| {
        let mut f = idct_dequant_8x8_scalar as IdctDequantFn<64>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_8x8_neon as IdctDequantFn<64>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x8_sse41 as IdctDequantFn<64>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x16() -> IdctDequantFn<256> {
    *DEQUANT_16X16.get_or_init(|| {
        let mut f = idct_dequant_16x16_scalar as IdctDequantFn<256>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_16x16_neon as IdctDequantFn<256>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x16_sse41 as IdctDequantFn<256>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x32() -> IdctDequantFn<1024> {
    *DEQUANT_32X32.get_or_init(|| {
        let mut f = idct_dequant_32x32_scalar as IdctDequantFn<1024>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_32x32_neon as IdctDequantFn<1024>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x32_sse41 as IdctDequantFn<1024>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_64x64() -> IdctDequantFn<1024> {
    *DEQUANT_64X64.get_or_init(|| {
        let mut f = idct_dequant_64x64_scalar as IdctDequantFn<1024>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_64x64_neon as IdctDequantFn<1024>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_64x64_sse41 as IdctDequantFn<1024>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x4() -> IadstDequantFn<16> {
    *ADST_DEQUANT_4X4.get_or_init(|| {
        let mut f = iadst_dequant_4x4_scalar as IadstDequantFn<16>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_4x4_neon as IadstDequantFn<16>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x4_sse41 as IadstDequantFn<16>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x8() -> IadstDequantFn<64> {
    *ADST_DEQUANT_8X8.get_or_init(|| {
        let mut f = iadst_dequant_8x8_scalar as IadstDequantFn<64>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_8x8_neon as IadstDequantFn<64>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x8_sse41 as IadstDequantFn<64>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x16() -> IadstDequantFn<256> {
    *ADST_DEQUANT_16X16.get_or_init(|| {
        let mut f = iadst_dequant_16x16_scalar as IadstDequantFn<256>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_16x16_neon as IadstDequantFn<256>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x16_sse41 as IadstDequantFn<256>;
            }
        }
        f
    })
}

static DEQUANT_RECT_4X8: OnceLock<IdctDequantFn<32>> = OnceLock::new();
static DEQUANT_RECT_8X4: OnceLock<IdctDequantFn<32>> = OnceLock::new();
static DEQUANT_RECT_8X16: OnceLock<IdctDequantFn<128>> = OnceLock::new();
static DEQUANT_RECT_16X8: OnceLock<IdctDequantFn<128>> = OnceLock::new();
static DEQUANT_RECT_16X32: OnceLock<IdctDequantFn<512>> = OnceLock::new();
static DEQUANT_RECT_32X16: OnceLock<IdctDequantFn<512>> = OnceLock::new();
static DEQUANT_RECT_4X16: OnceLock<IdctDequantFn<64>> = OnceLock::new();
static DEQUANT_RECT_16X4: OnceLock<IdctDequantFn<64>> = OnceLock::new();
static DEQUANT_RECT_8X32: OnceLock<IdctDequantFn<256>> = OnceLock::new();
static DEQUANT_RECT_32X8: OnceLock<IdctDequantFn<256>> = OnceLock::new();
static DEQUANT_RECT_4X32: OnceLock<IdctDequantFn<128>> = OnceLock::new();
static DEQUANT_RECT_32X4: OnceLock<IdctDequantFn<128>> = OnceLock::new();

#[inline]
pub(crate) fn idct_dequant_4x8() -> IdctDequantFn<32> {
    *DEQUANT_RECT_4X8.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<32, 4, 8> as IdctDequantFn<32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_4x8_neon as IdctDequantFn<32>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x8_sse41 as IdctDequantFn<32>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x4() -> IdctDequantFn<32> {
    *DEQUANT_RECT_8X4.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<32, 8, 4> as IdctDequantFn<32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_8x4_neon as IdctDequantFn<32>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x4_sse41 as IdctDequantFn<32>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x16() -> IdctDequantFn<128> {
    *DEQUANT_RECT_8X16.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<128, 8, 16> as IdctDequantFn<128>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_8x16_neon as IdctDequantFn<128>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x16_sse41 as IdctDequantFn<128>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x8() -> IdctDequantFn<128> {
    *DEQUANT_RECT_16X8.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<128, 16, 8> as IdctDequantFn<128>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_16x8_neon as IdctDequantFn<128>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x8_sse41 as IdctDequantFn<128>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x32() -> IdctDequantFn<512> {
    *DEQUANT_RECT_16X32.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<512, 16, 32> as IdctDequantFn<512>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_16x32_neon as IdctDequantFn<512>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x32_sse41 as IdctDequantFn<512>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x16() -> IdctDequantFn<512> {
    *DEQUANT_RECT_32X16.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<512, 32, 16> as IdctDequantFn<512>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_32x16_neon as IdctDequantFn<512>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x16_sse41 as IdctDequantFn<512>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x16() -> IdctDequantFn<64> {
    *DEQUANT_RECT_4X16.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<64, 4, 16> as IdctDequantFn<64>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_4x16_neon as IdctDequantFn<64>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x16_sse41 as IdctDequantFn<64>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_16x4() -> IdctDequantFn<64> {
    *DEQUANT_RECT_16X4.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<64, 16, 4> as IdctDequantFn<64>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_16x4_neon as IdctDequantFn<64>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_16x4_sse41 as IdctDequantFn<64>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_8x32() -> IdctDequantFn<256> {
    *DEQUANT_RECT_8X32.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<256, 8, 32> as IdctDequantFn<256>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_8x32_neon as IdctDequantFn<256>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_8x32_sse41 as IdctDequantFn<256>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x8() -> IdctDequantFn<256> {
    *DEQUANT_RECT_32X8.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<256, 32, 8> as IdctDequantFn<256>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_32x8_neon as IdctDequantFn<256>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x8_sse41 as IdctDequantFn<256>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_4x32() -> IdctDequantFn<128> {
    *DEQUANT_RECT_4X32.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<128, 4, 32> as IdctDequantFn<128>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_4x32_neon as IdctDequantFn<128>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_4x32_sse41 as IdctDequantFn<128>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn idct_dequant_32x4() -> IdctDequantFn<128> {
    *DEQUANT_RECT_32X4.get_or_init(|| {
        let mut f = idct_dequant_rect_scalar_core::<128, 32, 4> as IdctDequantFn<128>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::idct_dequant_32x4_neon as IdctDequantFn<128>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::idct_dequant_32x4_sse41 as IdctDequantFn<128>;
            }
        }
        f
    })
}

static ADST_DEQUANT_RECT_4X8: OnceLock<IadstDequantFn<32>> = OnceLock::new();
static ADST_DEQUANT_RECT_8X4: OnceLock<IadstDequantFn<32>> = OnceLock::new();
static ADST_DEQUANT_RECT_8X16: OnceLock<IadstDequantFn<128>> = OnceLock::new();
static ADST_DEQUANT_RECT_16X8: OnceLock<IadstDequantFn<128>> = OnceLock::new();
static ADST_DEQUANT_RECT_4X16: OnceLock<IadstDequantFn<64>> = OnceLock::new();
static ADST_DEQUANT_RECT_16X4: OnceLock<IadstDequantFn<64>> = OnceLock::new();

#[inline]
pub(crate) fn iadst_dequant_4x8() -> IadstDequantFn<32> {
    *ADST_DEQUANT_RECT_4X8.get_or_init(|| {
        let mut f = itx_dequant_rect_scalar_core::<32, 4, 8> as IadstDequantFn<32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_4x8_neon as IadstDequantFn<32>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x8_sse41 as IadstDequantFn<32>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x4() -> IadstDequantFn<32> {
    *ADST_DEQUANT_RECT_8X4.get_or_init(|| {
        let mut f = itx_dequant_rect_scalar_core::<32, 8, 4> as IadstDequantFn<32>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_8x4_neon as IadstDequantFn<32>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x4_sse41 as IadstDequantFn<32>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_8x16() -> IadstDequantFn<128> {
    *ADST_DEQUANT_RECT_8X16.get_or_init(|| {
        let mut f = itx_dequant_rect_scalar_core::<128, 8, 16> as IadstDequantFn<128>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_8x16_neon as IadstDequantFn<128>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_8x16_sse41 as IadstDequantFn<128>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x8() -> IadstDequantFn<128> {
    *ADST_DEQUANT_RECT_16X8.get_or_init(|| {
        let mut f = itx_dequant_rect_scalar_core::<128, 16, 8> as IadstDequantFn<128>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_16x8_neon as IadstDequantFn<128>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x8_sse41 as IadstDequantFn<128>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_4x16() -> IadstDequantFn<64> {
    *ADST_DEQUANT_RECT_4X16.get_or_init(|| {
        let mut f = itx_dequant_rect_scalar_core::<64, 4, 16> as IadstDequantFn<64>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_4x16_neon as IadstDequantFn<64>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_4x16_sse41 as IadstDequantFn<64>;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn iadst_dequant_16x4() -> IadstDequantFn<64> {
    *ADST_DEQUANT_RECT_16X4.get_or_init(|| {
        let mut f = itx_dequant_rect_scalar_core::<64, 16, 4> as IadstDequantFn<64>;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::iadst_dequant_16x4_neon as IadstDequantFn<64>;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::iadst_dequant_16x4_sse41 as IadstDequantFn<64>;
            }
        }
        f
    })
}
