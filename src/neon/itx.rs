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

use crate::itx_2d::ITX_TMP_PIXELS;
use std::arch::aarch64::*;

// Concrete 32x32 DCT kernels. These do not route through DctSimd4/DctWide.

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct16_i32x4_impl(s: &[int32x4_t; 16]) -> [int32x4_t; 16] {
    let z = vdupq_n_s32(0);
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            acc = vmlaq_n_s32(acc, s[j], crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m]);
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_adst16_i32x4_impl(s: &[int32x4_t; 16], flip: bool) -> [int32x4_t; 16] {
    let rows = if flip {
        &crate::itx_1d::FLIPADST16_KERNEL_ROWS
    } else {
        &crate::itx_1d::ADST16_KERNEL_ROWS
    };
    let z = vdupq_n_s32(0);
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let row = &rows[m];
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            acc = vmlaq_n_s32(acc, s[j], row[j] as i32);
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_tx16_i32x4_impl(s: &[int32x4_t; 16], kind: usize) -> [int32x4_t; 16] {
    match kind {
        crate::itx_2d::TX_KIND_DCT => neon_dct16_i32x4_impl(s),
        crate::itx_2d::TX_KIND_ADST => neon_adst16_i32x4_impl(s, false),
        crate::itx_2d::TX_KIND_FLIPADST => neon_adst16_i32x4_impl(s, true),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn iadst_dequant_16x16_neon_i32_impl(
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
    if is_rect2 {
        iadst_dequant_16x16_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        iadst_dequant_16x16_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn iadst_dequant_16x16_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 256);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut s = [vdupq_n_s32(0); 16];
            let mut j = 0usize;
            while j < 16 {
                let mut v = vld1q_s32(coeff.as_ptr().add(y + j * 16));
                if IS_RECT2 {
                    v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                }
                s[j] = v;
                j += 1;
            }
            let out = neon_tx16_i32x4_impl(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 16 {
            tmp[y * 32..y * 32 + 16].fill(0);
            y += 1;
        }
        coeff[..256].fill(0);
        let mut x = 0usize;
        while x < 16 {
            let mut s = [vdupq_n_s32(0); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = vld1q_s32(tmp.as_ptr().add(x + j * 32));
                j += 1;
            }
            let out = neon_tx16_i32x4_impl(&s, second_kind);
            j = 0;
            while j < 16 {
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32), out[j]);
                j += 1;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_store4x4_i32_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    v0: int32x4_t,
    v1: int32x4_t,
    v2: int32x4_t,
    v3: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        macro_rules! clip {
            ($x:expr) => {{ vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32($x, rnd), nsh), minv), maxv) }};
        }
        let c0 = clip!(v0);
        let c1 = clip!(v1);
        let c2 = clip!(v2);
        let c3 = clip!(v3);
        let t01 = vtrnq_s32(c0, c1);
        let t23 = vtrnq_s32(c2, c3);
        let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
        let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
        let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
        let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
        vst1q_s32(tmp.as_mut_ptr().add(off), r0);
        vst1q_s32(tmp.as_mut_ptr().add(off + 32), r1);
        vst1q_s32(tmp.as_mut_ptr().add(off + 64), r2);
        vst1q_s32(tmp.as_mut_ptr().add(off + 96), r3);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 32 {
            let mut v = vld1q_s32(coeff.as_ptr().add(base + j * 32));
            if IS_RECT2 {
                v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
            }
            a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
            a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]);
            a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]);
            a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]);
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i32x4_from_tmp4(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 32 {
            let v = vld1q_s32(tmp.as_ptr().add(base + j * 32));
            a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
            a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]);
            a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]);
            a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]);
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_tx8_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    kind: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 8 {
            let mut v = vld1q_s32(coeff.as_ptr().add(base + j * 8));
            if IS_RECT2 {
                v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
            }
            a0 = vmlaq_n_s32(a0, v, tx8_coeff(kind, m, j));
            a1 = vmlaq_n_s32(a1, v, tx8_coeff(kind, m + 1, j));
            a2 = vmlaq_n_s32(a2, v, tx8_coeff(kind, m + 2, j));
            a3 = vmlaq_n_s32(a3, v, tx8_coeff(kind, m + 3, j));
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_tx8_i32x4_from_tmp4(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    kind: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 8 {
            let v = vld1q_s32(tmp.as_ptr().add(base + j * 32));
            a0 = vmlaq_n_s32(a0, v, tx8_coeff(kind, m, j));
            a1 = vmlaq_n_s32(a1, v, tx8_coeff(kind, m + 1, j));
            a2 = vmlaq_n_s32(a2, v, tx8_coeff(kind, m + 2, j));
            a3 = vmlaq_n_s32(a3, v, tx8_coeff(kind, m + 3, j));
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
fn tx8_coeff(kind: usize, out: usize, input: usize) -> i32 {
    match kind {
        crate::itx_2d::TX_KIND_DCT => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_ADST => crate::itx_2d::ADST8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_FLIPADST => crate::itx_2d::ADST8_KW[(7 - out) * 8 + input] as i32,
        _ => unreachable!(),
    }
}

#[inline]
fn neon_identity_scale(n: usize) -> i16 {
    match n {
        4 => 128,
        8 => 181,
        16 => 256,
        32 => 362,
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_identity_i16x4_coeff_to_i32<const IS_RECT2: bool>(
    coeff: &[i16],
    off: usize,
    scale: i16,
) -> int32x4_t {
    let v = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, off);
    vmull_n_s16(v, scale)
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_identity_i16x4_scratch_to_i32(scratch: &[i16], off: usize, scale: i16) -> int32x4_t {
    let v = neon_load4_i16_scratch(scratch, off);
    vmull_n_s16(v, scale)
}

#[inline]
fn neon_tx_dense_coeff(kind: usize, n: usize, out: usize, input: usize) -> i32 {
    match (kind, n) {
        (crate::itx_2d::TX_KIND_DCT, 4) => crate::itx_2d::DCT4_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 8) => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 16) => crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out],
        (crate::itx_2d::TX_KIND_DCT, 32) => crate::itx_2d::DCT32_DENSE_KERNEL[input * 32 + out],
        (crate::itx_2d::TX_KIND_ADST, 4) => crate::itx_1d::ADST4_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_ADST, 8) => crate::itx_1d::ADST8_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_ADST, 16) => crate::itx_1d::ADST16_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_FLIPADST, 4) => {
            crate::itx_1d::FLIPADST4_KERNEL_ROWS[out][input] as i32
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 8) => {
            crate::itx_1d::ADST8_KERNEL_ROWS[7 - out][input] as i32
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 16) => {
            crate::itx_1d::FLIPADST16_KERNEL_ROWS[out][input] as i32
        }
        _ => unreachable!(),
    }
}
#[inline]
fn neon_tx_dense_coeff_i16_const<const KIND: usize, const N: usize>(
    out: usize,
    input: usize,
) -> i16 {
    match (KIND, N) {
        (crate::itx_2d::TX_KIND_IDENTITY, 4) => {
            if out == input {
                128
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 8) => {
            if out == input {
                181
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 16) => {
            if out == input {
                256
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 32) => {
            if out == input {
                362
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_DCT, 4) => crate::itx_2d::DCT4_KW[out * 8 + input] as i16,
        (crate::itx_2d::TX_KIND_DCT, 8) => crate::itx_2d::DCT8_KW[out * 8 + input] as i16,
        (crate::itx_2d::TX_KIND_DCT, 16) => {
            crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out] as i16
        }
        (crate::itx_2d::TX_KIND_DCT, 32) => {
            crate::itx_2d::DCT32_DENSE_KERNEL[input * 32 + out] as i16
        }
        (crate::itx_2d::TX_KIND_ADST, 4) => crate::itx_1d::ADST4_KERNEL_ROWS[out][input] as i16,
        (crate::itx_2d::TX_KIND_ADST, 8) => crate::itx_1d::ADST8_KERNEL_ROWS[out][input] as i16,
        (crate::itx_2d::TX_KIND_ADST, 16) => crate::itx_1d::ADST16_KERNEL_ROWS[out][input] as i16,
        (crate::itx_2d::TX_KIND_FLIPADST, 4) => {
            crate::itx_1d::FLIPADST4_KERNEL_ROWS[out][input] as i16
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 8) => {
            crate::itx_1d::ADST8_KERNEL_ROWS[7 - out][input] as i16
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 16) => {
            crate::itx_1d::FLIPADST16_KERNEL_ROWS[out][input] as i16
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_load4_i16_coeff_packed_const<const IS_RECT2: bool>(src: &[i16], off: usize) -> int16x4_t {
    debug_assert!(off + 4 <= src.len());
    let v = unsafe { vld1_s16(src.as_ptr().add(off)) };
    if IS_RECT2 {
        let w = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), v, 181));
        vmovn_s32(w)
    } else {
        v
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn neon_load4_i16_coeff_packed_rdm_const<const IS_RECT2: bool>(
    src: &[i16],
    off: usize,
) -> int16x4_t {
    debug_assert!(off + 4 <= src.len());
    let v = unsafe { vld1_s16(src.as_ptr().add(off)) };
    if IS_RECT2 {
        vqrdmulh_s16(v, vdup_n_s16(0x5a80))
    } else {
        v
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_load4_i16_scratch(src: &[i16], off: usize) -> int16x4_t {
    debug_assert!(off + 4 <= src.len());
    unsafe { vld1_s16(src.as_ptr().add(off)) }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_store4x4_i16_clip<const STRIDE: usize>(
    scratch: &mut [i16],
    off: usize,
    v0: int32x4_t,
    v1: int32x4_t,
    v2: int32x4_t,
    v3: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        debug_assert!(STRIDE == 4 || STRIDE == 8 || STRIDE == 16 || STRIDE == 32);
        debug_assert!(off + 3 * STRIDE + 4 <= scratch.len());
        macro_rules! clip {
            ($x:expr) => {{ vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32($x, rnd), nsh), minv), maxv) }};
        }
        let c0 = clip!(v0);
        let c1 = clip!(v1);
        let c2 = clip!(v2);
        let c3 = clip!(v3);
        let t01 = vtrnq_s32(c0, c1);
        let t23 = vtrnq_s32(c2, c3);
        let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
        let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
        let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
        let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
        vst1_s16(scratch.as_mut_ptr().add(off), vmovn_s32(r0));
        vst1_s16(scratch.as_mut_ptr().add(off + STRIDE), vmovn_s32(r1));
        vst1_s16(scratch.as_mut_ptr().add(off + 2 * STRIDE), vmovn_s32(r2));
        vst1_s16(scratch.as_mut_ptr().add(off + 3 * STRIDE), vmovn_s32(r3));
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_store8x8_i16_clip<const STRIDE: usize>(
    scratch: &mut [i16],
    off: usize,
    v0lo: int32x4_t,
    v0hi: int32x4_t,
    v1lo: int32x4_t,
    v1hi: int32x4_t,
    v2lo: int32x4_t,
    v2hi: int32x4_t,
    v3lo: int32x4_t,
    v3hi: int32x4_t,
    v4lo: int32x4_t,
    v4hi: int32x4_t,
    v5lo: int32x4_t,
    v5hi: int32x4_t,
    v6lo: int32x4_t,
    v6hi: int32x4_t,
    v7lo: int32x4_t,
    v7hi: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        debug_assert!(STRIDE == 8 || STRIDE == 16 || STRIDE == 32);
        debug_assert!(off + 7 * STRIDE + 8 <= scratch.len());
        macro_rules! clip {
            ($x:expr) => {{ vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32($x, rnd), nsh), minv), maxv) }};
        }

        let r0 = vcombine_s16(vmovn_s32(clip!(v0lo)), vmovn_s32(clip!(v0hi)));
        let r1 = vcombine_s16(vmovn_s32(clip!(v1lo)), vmovn_s32(clip!(v1hi)));
        let r2 = vcombine_s16(vmovn_s32(clip!(v2lo)), vmovn_s32(clip!(v2hi)));
        let r3 = vcombine_s16(vmovn_s32(clip!(v3lo)), vmovn_s32(clip!(v3hi)));
        let r4 = vcombine_s16(vmovn_s32(clip!(v4lo)), vmovn_s32(clip!(v4hi)));
        let r5 = vcombine_s16(vmovn_s32(clip!(v5lo)), vmovn_s32(clip!(v5hi)));
        let r6 = vcombine_s16(vmovn_s32(clip!(v6lo)), vmovn_s32(clip!(v6hi)));
        let r7 = vcombine_s16(vmovn_s32(clip!(v7lo)), vmovn_s32(clip!(v7hi)));

        let t01 = vtrnq_s16(r0, r1);
        let t23 = vtrnq_s16(r2, r3);
        let t45 = vtrnq_s16(r4, r5);
        let t67 = vtrnq_s16(r6, r7);

        let u02_0 = vtrnq_s32(vreinterpretq_s32_s16(t01.0), vreinterpretq_s32_s16(t23.0));
        let u02_1 = vtrnq_s32(vreinterpretq_s32_s16(t01.1), vreinterpretq_s32_s16(t23.1));
        let u46_0 = vtrnq_s32(vreinterpretq_s32_s16(t45.0), vreinterpretq_s32_s16(t67.0));
        let u46_1 = vtrnq_s32(vreinterpretq_s32_s16(t45.1), vreinterpretq_s32_s16(t67.1));

        macro_rules! join64 {
            ($a:expr, $b:expr, $lane:ident) => {{
                vreinterpretq_s16_s64(vcombine_s64(
                    $lane(vreinterpretq_s64_s32($a)),
                    $lane(vreinterpretq_s64_s32($b)),
                ))
            }};
        }

        let o0 = join64!(u02_0.0, u46_0.0, vget_low_s64);
        let o1 = join64!(u02_0.1, u46_0.1, vget_low_s64);
        let o2 = join64!(u02_1.0, u46_1.0, vget_low_s64);
        let o3 = join64!(u02_1.1, u46_1.1, vget_low_s64);
        let o4 = join64!(u02_0.0, u46_0.0, vget_high_s64);
        let o5 = join64!(u02_0.1, u46_0.1, vget_high_s64);
        let o6 = join64!(u02_1.0, u46_1.0, vget_high_s64);
        let o7 = join64!(u02_1.1, u46_1.1, vget_high_s64);

        vst1q_s16(scratch.as_mut_ptr().add(off), o0);
        vst1q_s16(scratch.as_mut_ptr().add(off + STRIDE), o1);
        vst1q_s16(scratch.as_mut_ptr().add(off + 2 * STRIDE), o2);
        vst1q_s16(scratch.as_mut_ptr().add(off + 3 * STRIDE), o3);
        vst1q_s16(scratch.as_mut_ptr().add(off + 4 * STRIDE), o4);
        vst1q_s16(scratch.as_mut_ptr().add(off + 5 * STRIDE), o5);
        vst1q_s16(scratch.as_mut_ptr().add(off + 6 * STRIDE), o6);
        vst1q_s16(scratch.as_mut_ptr().add(off + 7 * STRIDE), o7);
    }
}

macro_rules! neon_dct16_i16x4_all_body {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(1), crate::itx_2d::DCT16_KBW[base]);
            acc = vmlal_n_s16(acc, load!(3), crate::itx_2d::DCT16_KBW[base + 1]);
            acc = vmlal_n_s16(acc, load!(5), crate::itx_2d::DCT16_KBW[base + 2]);
            acc = vmlal_n_s16(acc, load!(7), crate::itx_2d::DCT16_KBW[base + 3]);
            acc = vmlal_n_s16(acc, load!(9), crate::itx_2d::DCT16_KBW[base + 4]);
            acc = vmlal_n_s16(acc, load!(11), crate::itx_2d::DCT16_KBW[base + 5]);
            acc = vmlal_n_s16(acc, load!(13), crate::itx_2d::DCT16_KBW[base + 6]);
            acc = vmlal_n_s16(acc, load!(15), crate::itx_2d::DCT16_KBW[base + 7]);
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(2), crate::itx_2d::DCT16_KDW[base]);
            acc = vmlal_n_s16(acc, load!(6), crate::itx_2d::DCT16_KDW[base + 1]);
            acc = vmlal_n_s16(acc, load!(10), crate::itx_2d::DCT16_KDW[base + 2]);
            acc = vmlal_n_s16(acc, load!(14), crate::itx_2d::DCT16_KDW[base + 3]);
            d[m] = acc;
            m += 1;
        }
        let f0 = vaddq_s32(
            vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[0]),
            vmull_n_s16(load!(12), crate::itx_2d::DCT16_KFW[1]),
        );
        let f1 = vaddq_s32(
            vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[2]),
            vmull_n_s16(load!(12), crate::itx_2d::DCT16_KFW[3]),
        );
        let g0 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[0]),
            vmull_n_s16(load!(8), crate::itx_2d::DCT16_KGW[1]),
        );
        let g1 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[2]),
            vmull_n_s16(load!(8), crate::itx_2d::DCT16_KGW[3]),
        );
        let cc = [
            vaddq_s32(g0, f0),
            vaddq_s32(g1, f1),
            vsubq_s32(g1, f1),
            vsubq_s32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = vsubq_s32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 8] = vsubq_s32(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! neon_dct32_i16x4_all_body {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut grp = 0usize;
            while grp < 2 {
                let cb = base + grp * 8;
                let k0 = grp * 8;
                acc = vmlal_n_s16(acc, load!(2 * k0 + 1), crate::itx_2d::DCT32_KBW[cb]);
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 1) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 1],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 2) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 2],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 3) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 3],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 4) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 4],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 5) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 5],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 6) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 6],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 7) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 7],
                );
                grp += 1;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(2), crate::itx_2d::DCT32_KDW[base]);
            acc = vmlal_n_s16(acc, load!(6), crate::itx_2d::DCT32_KDW[base + 1]);
            acc = vmlal_n_s16(acc, load!(10), crate::itx_2d::DCT32_KDW[base + 2]);
            acc = vmlal_n_s16(acc, load!(14), crate::itx_2d::DCT32_KDW[base + 3]);
            acc = vmlal_n_s16(acc, load!(18), crate::itx_2d::DCT32_KDW[base + 4]);
            acc = vmlal_n_s16(acc, load!(22), crate::itx_2d::DCT32_KDW[base + 5]);
            acc = vmlal_n_s16(acc, load!(26), crate::itx_2d::DCT32_KDW[base + 6]);
            acc = vmlal_n_s16(acc, load!(30), crate::itx_2d::DCT32_KDW[base + 7]);
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(4), crate::itx_2d::DCT32_KFW[base]);
            acc = vmlal_n_s16(acc, load!(12), crate::itx_2d::DCT32_KFW[base + 1]);
            acc = vmlal_n_s16(acc, load!(20), crate::itx_2d::DCT32_KFW[base + 2]);
            acc = vmlal_n_s16(acc, load!(28), crate::itx_2d::DCT32_KFW[base + 3]);
            f[m] = acc;
            m += 1;
        }
        let h0 = vaddq_s32(
            vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[0]),
            vmull_n_s16(load!(24), crate::itx_2d::DCT32_KHW[1]),
        );
        let h1 = vaddq_s32(
            vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[2]),
            vmull_n_s16(load!(24), crate::itx_2d::DCT32_KHW[3]),
        );
        let g0 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[0]),
            vmull_n_s16(load!(16), crate::itx_2d::DCT32_KGW[1]),
        );
        let g1 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[2]),
            vmull_n_s16(load!(16), crate::itx_2d::DCT32_KGW[3]),
        );
        let e = [
            vaddq_s32(g0, h0),
            vaddq_s32(g1, h1),
            vsubq_s32(g1, h1),
            vsubq_s32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = vaddq_s32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = vsubq_s32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = vsubq_s32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 16] = vsubq_s32(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! neon_dct16_i16x4_all_body_active {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            if ACTIVE > 1 {
                acc = vmlal_n_s16(acc, load!(1), crate::itx_2d::DCT16_KBW[base]);
            }
            if ACTIVE > 3 {
                acc = vmlal_n_s16(acc, load!(3), crate::itx_2d::DCT16_KBW[base + 1]);
            }
            if ACTIVE > 5 {
                acc = vmlal_n_s16(acc, load!(5), crate::itx_2d::DCT16_KBW[base + 2]);
            }
            if ACTIVE > 7 {
                acc = vmlal_n_s16(acc, load!(7), crate::itx_2d::DCT16_KBW[base + 3]);
            }
            if ACTIVE > 9 {
                acc = vmlal_n_s16(acc, load!(9), crate::itx_2d::DCT16_KBW[base + 4]);
            }
            if ACTIVE > 11 {
                acc = vmlal_n_s16(acc, load!(11), crate::itx_2d::DCT16_KBW[base + 5]);
            }
            if ACTIVE > 13 {
                acc = vmlal_n_s16(acc, load!(13), crate::itx_2d::DCT16_KBW[base + 6]);
            }
            if ACTIVE > 15 {
                acc = vmlal_n_s16(acc, load!(15), crate::itx_2d::DCT16_KBW[base + 7]);
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            if ACTIVE > 2 {
                acc = vmlal_n_s16(acc, load!(2), crate::itx_2d::DCT16_KDW[base]);
            }
            if ACTIVE > 6 {
                acc = vmlal_n_s16(acc, load!(6), crate::itx_2d::DCT16_KDW[base + 1]);
            }
            if ACTIVE > 10 {
                acc = vmlal_n_s16(acc, load!(10), crate::itx_2d::DCT16_KDW[base + 2]);
            }
            if ACTIVE > 14 {
                acc = vmlal_n_s16(acc, load!(14), crate::itx_2d::DCT16_KDW[base + 3]);
            }
            d[m] = acc;
            m += 1;
        }
        let f0 = if ACTIVE > 4 {
            let mut t = vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[0]);
            if ACTIVE > 12 {
                t = vmlal_n_s16(t, load!(12), crate::itx_2d::DCT16_KFW[1]);
            }
            t
        } else {
            z
        };
        let f1 = if ACTIVE > 4 {
            let mut t = vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[2]);
            if ACTIVE > 12 {
                t = vmlal_n_s16(t, load!(12), crate::itx_2d::DCT16_KFW[3]);
            }
            t
        } else {
            z
        };
        let mut g0 = vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[0]);
        if ACTIVE > 8 {
            g0 = vmlal_n_s16(g0, load!(8), crate::itx_2d::DCT16_KGW[1]);
        }
        let mut g1 = vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[2]);
        if ACTIVE > 8 {
            g1 = vmlal_n_s16(g1, load!(8), crate::itx_2d::DCT16_KGW[3]);
        }
        let cc = [
            vaddq_s32(g0, f0),
            vaddq_s32(g1, f1),
            vsubq_s32(g1, f1),
            vsubq_s32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = vsubq_s32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 8] = vsubq_s32(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! neon_dct32_i16x4_all_body_active {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut k = 0usize;
            while k < 16 {
                let idx = 2 * k + 1;
                if ACTIVE > idx {
                    acc = vmlal_n_s16(acc, load!(idx), crate::itx_2d::DCT32_KBW[base + k]);
                }
                k += 1;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            let mut k = 0usize;
            while k < 8 {
                let idx = 4 * k + 2;
                if ACTIVE > idx {
                    acc = vmlal_n_s16(acc, load!(idx), crate::itx_2d::DCT32_KDW[base + k]);
                }
                k += 1;
            }
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            if ACTIVE > 4 {
                acc = vmlal_n_s16(acc, load!(4), crate::itx_2d::DCT32_KFW[base]);
            }
            if ACTIVE > 12 {
                acc = vmlal_n_s16(acc, load!(12), crate::itx_2d::DCT32_KFW[base + 1]);
            }
            if ACTIVE > 20 {
                acc = vmlal_n_s16(acc, load!(20), crate::itx_2d::DCT32_KFW[base + 2]);
            }
            if ACTIVE > 28 {
                acc = vmlal_n_s16(acc, load!(28), crate::itx_2d::DCT32_KFW[base + 3]);
            }
            f[m] = acc;
            m += 1;
        }
        let h0 = if ACTIVE > 8 {
            let mut t = vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[0]);
            if ACTIVE > 24 {
                t = vmlal_n_s16(t, load!(24), crate::itx_2d::DCT32_KHW[1]);
            }
            t
        } else {
            z
        };
        let h1 = if ACTIVE > 8 {
            let mut t = vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[2]);
            if ACTIVE > 24 {
                t = vmlal_n_s16(t, load!(24), crate::itx_2d::DCT32_KHW[3]);
            }
            t
        } else {
            z
        };
        let mut g0 = vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[0]);
        if ACTIVE > 16 {
            g0 = vmlal_n_s16(g0, load!(16), crate::itx_2d::DCT32_KGW[1]);
        }
        let mut g1 = vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[2]);
        if ACTIVE > 16 {
            g1 = vmlal_n_s16(g1, load!(16), crate::itx_2d::DCT32_KGW[3]);
        }
        let e = [
            vaddq_s32(g0, h0),
            vaddq_s32(g1, h1),
            vsubq_s32(g1, h1),
            vsubq_s32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = vaddq_s32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = vsubq_s32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = vsubq_s32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 16] = vsubq_s32(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct16_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct16_i16x4_all_body!()
}

#[target_feature(enable = "rdm")]
#[inline]
fn neon_dct16_i16x4_all_from_coeff4_rdm_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct16_i16x4_all_body!()
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct32_i16x4_all_body!()
}

#[target_feature(enable = "rdm")]
#[inline]
fn neon_dct32_i16x4_all_from_coeff4_rdm_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct32_i16x4_all_body!()
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct16_i16x4_all_from_scratch4_stride<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= scratch.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
        };
    }
    unsafe { neon_dct16_i16x4_all_body!() }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i16x4_all_from_scratch4_stride<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= scratch.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
        };
    }
    unsafe { neon_dct32_i16x4_all_body!() }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct16_i16x4_all_from_scratch4_stride_active<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
    let zz = vdup_n_s16(0);
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                neon_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zz
            }
        };
    }
    unsafe { neon_dct16_i16x4_all_body_active!() }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i16x4_all_from_scratch4_stride_active<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
    let zz = vdup_n_s16(0);
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                neon_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zz
            }
        };
    }
    unsafe { neon_dct32_i16x4_all_body_active!() }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct16_i16x4_all_from_scratch4_stride_eob<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
) -> [int32x4_t; 16] {
    if active <= 4 {
        neon_dct16_i16x4_all_from_scratch4_stride_active::<STRIDE, 4>(scratch, base)
    } else if active <= 8 {
        neon_dct16_i16x4_all_from_scratch4_stride_active::<STRIDE, 8>(scratch, base)
    } else {
        neon_dct16_i16x4_all_from_scratch4_stride_active::<STRIDE, 16>(scratch, base)
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i16x4_all_from_scratch4_stride_eob<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
) -> [int32x4_t; 32] {
    if active <= 4 {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 4>(scratch, base)
    } else if active <= 8 {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 8>(scratch, base)
    } else if active <= 16 {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 16>(scratch, base)
    } else {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 32>(scratch, base)
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_dct_i16_neon_impl<const N: usize, const LEN: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_dct_i16_neon_impl_const::<N, LEN, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_dct_i16_neon_impl_const::<N, LEN, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_dct_i16_neon_impl_const<const N: usize, const LEN: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(N == 16 || N == 32);
        debug_assert!(LEN >= N * N);
        debug_assert!(coeff.len() >= N * N);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < N / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut scratch = [0i16; LEN];
        if N == 16 {
            let mut y = 0usize;
            while y + 8 <= ncols {
                let lo = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y);
                let hi =
                    neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y + 4);
                let mut x = 0usize;
                while x < 16 {
                    neon_store8x8_i16_clip::<16>(
                        &mut scratch,
                        y * 16 + x,
                        lo[x],
                        hi[x],
                        lo[x + 1],
                        hi[x + 1],
                        lo[x + 2],
                        hi[x + 2],
                        lo[x + 3],
                        hi[x + 3],
                        lo[x + 4],
                        hi[x + 4],
                        lo[x + 5],
                        hi[x + 5],
                        lo[x + 6],
                        hi[x + 6],
                        lo[x + 7],
                        hi[x + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    x += 8;
                }
                y += 8;
            }
            while y + 4 <= ncols {
                let out = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 16>(coeff, y);
                let mut x = 0usize;
                while x < 16 {
                    neon_store4x4_i16_clip::<16>(
                        &mut scratch,
                        y * 16 + x,
                        out[x],
                        out[x + 1],
                        out[x + 2],
                        out[x + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    x += 4;
                }
                y += 4;
            }
            let mut x = 0usize;
            while x < 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<16>(&scratch, x, ncols);
                let mut m = 0usize;
                while m < 16 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
                x += 4;
            }
        } else {
            let mut y = 0usize;
            while y + 8 <= ncols {
                let lo = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y);
                let hi =
                    neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y + 4);
                let mut x = 0usize;
                while x < 32 {
                    neon_store8x8_i16_clip::<32>(
                        &mut scratch,
                        y * 32 + x,
                        lo[x],
                        hi[x],
                        lo[x + 1],
                        hi[x + 1],
                        lo[x + 2],
                        hi[x + 2],
                        lo[x + 3],
                        hi[x + 3],
                        lo[x + 4],
                        hi[x + 4],
                        lo[x + 5],
                        hi[x + 5],
                        lo[x + 6],
                        hi[x + 6],
                        lo[x + 7],
                        hi[x + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    x += 8;
                }
                y += 8;
            }
            while y + 4 <= ncols {
                let out = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, 32>(coeff, y);
                let mut x = 0usize;
                while x < 32 {
                    neon_store4x4_i16_clip::<32>(
                        &mut scratch,
                        y * 32 + x,
                        out[x],
                        out[x + 1],
                        out[x + 2],
                        out[x + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    x += 4;
                }
                y += 4;
            }
            let mut x = 0usize;
            while x < 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<32>(&scratch, x, ncols);
                let mut m = 0usize;
                while m < 32 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
                x += 4;
            }
        }
        coeff[..N * N].fill(0);
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn idct_dequant_dct_i16_neon_rdm_impl<const N: usize, const LEN: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_dct_i16_neon_rdm_impl_const::<N, LEN, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_dct_i16_neon_rdm_impl_const::<N, LEN, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn idct_dequant_dct_i16_neon_rdm_impl_const<
    const N: usize,
    const LEN: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(N == 16 || N == 32);
        debug_assert!(LEN >= N * N);
        debug_assert!(coeff.len() >= N * N);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < N / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut scratch = [0i16; LEN];
        debug_assert!(N == 32);
        let mut y = 0usize;
        while y + 8 <= ncols {
            let lo = neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, 32>(coeff, y);
            let hi =
                neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, 32>(coeff, y + 4);
            let mut x = 0usize;
            while x < 32 {
                neon_store8x8_i16_clip::<32>(
                    &mut scratch,
                    y * 32 + x,
                    lo[x],
                    hi[x],
                    lo[x + 1],
                    hi[x + 1],
                    lo[x + 2],
                    hi[x + 2],
                    lo[x + 3],
                    hi[x + 3],
                    lo[x + 4],
                    hi[x + 4],
                    lo[x + 5],
                    hi[x + 5],
                    lo[x + 6],
                    hi[x + 6],
                    lo[x + 7],
                    hi[x + 7],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 8;
            }
            y += 8;
        }
        while y + 4 <= ncols {
            let out = neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, 32>(coeff, y);
            let mut x = 0usize;
            while x < 32 {
                neon_store4x4_i16_clip::<32>(
                    &mut scratch,
                    y * 32 + x,
                    out[x],
                    out[x + 1],
                    out[x + 2],
                    out[x + 3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        let mut x = 0usize;
        while x < 32 {
            let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<32>(&scratch, x, ncols);
            let mut m = 0usize;
            while m < 32 {
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                m += 4;
            }
            x += 4;
        }
        coeff[..N * N].fill(0);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i32_impl<const N: usize, const W: usize, const H: usize>(
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
    if is_rect2 {
        tx_dequant_dense_neon_i32_impl_const::<N, W, H, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i32_impl_const::<N, W, H, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i32_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut y = 0usize;
        while y + 4 <= nrows {
            let mut m = 0usize;
            while m < W {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < W {
                    let mut v = vld1q_s32(coeff.as_ptr().add(y + j * H));
                    if IS_RECT2 {
                        v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                    }
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(first_kind, W, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(first_kind, W, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(first_kind, W, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(first_kind, W, m + 3, j));
                    j += 1;
                }
                neon_store4x4_i32_clip(tmp, y * 32 + m, a0, a1, a2, a3, rnd, nsh, minv, maxv);
                m += 4;
            }
            y += 4;
        }
        while y < H {
            tmp[y * 32..y * 32 + W].fill(0);
            y += 1;
        }
        coeff[..W * H].fill(0);

        let mut x = 0usize;
        while x < W {
            let mut m = 0usize;
            while m < H {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < H {
                    let v = vld1q_s32(tmp.as_ptr().add(x + j * 32));
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(second_kind, H, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(second_kind, H, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(second_kind, H, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(second_kind, H, m + 3, j));
                    j += 1;
                }
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a0);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), a1);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), a2);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), a3);
                m += 4;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i16_impl<const N: usize, const W: usize, const H: usize>(
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
) {
    macro_rules! call_kind {
        ($first:expr, $second:expr) => {
            tx_dequant_dense_neon_i16_impl_kind::<N, W, H, { $first }, { $second }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            )
        };
    }
    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i16_impl_kind<
    const N: usize,
    const W: usize,
    const H: usize,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_impl_const::<N, W, H, true, FIRST_KIND, SECOND_KIND>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        tx_dequant_dense_neon_i16_impl_const::<N, W, H, false, FIRST_KIND, SECOND_KIND>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_add4_i32_to_u8(dst: &mut [u8], off: usize, v: int32x4_t, rnd: int32x4_t, nsh: int32x4_t) {
    debug_assert!(off + 4 <= dst.len());
    let r = vshlq_s32(vaddq_s32(v, rnd), nsh);
    let r16 = vmovn_s32(r);
    let d8 = unsafe { vld1_u8(dst.as_ptr().add(off)) };
    let d16 = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(d8)));
    let sum = vadd_s16(d16, r16);
    let sum = vmax_s16(vdup_n_s16(0), vmin_s16(vdup_n_s16(255), sum));
    let packed = vqmovun_s16(vcombine_s16(sum, vdup_n_s16(0)));
    unsafe {
        vst1_lane_u32::<0>(
            dst.as_mut_ptr().add(off) as *mut u32,
            vreinterpret_u32_u8(packed),
        );
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_add4_i32_to_u8_expand_x2(
    dst: &mut [u8],
    off: usize,
    v: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
) {
    debug_assert!(off + 8 <= dst.len());
    let r = vshlq_s32(vaddq_s32(v, rnd), nsh);
    let r16 = vmovn_s32(r);
    let rr = vcombine_s16(r16, r16);
    let rdup = vzip1q_s16(rr, rr);
    let p = unsafe { dst.as_mut_ptr().add(off) };
    let d8 = unsafe { vld1_u8(p) };
    let d16 = vreinterpretq_s16_u16(vmovl_u8(d8));
    let sum = vaddq_s16(d16, rdup);
    let sum = vmaxq_s16(vdupq_n_s16(0), vminq_s16(vdupq_n_s16(255), sum));
    let packed = vqmovun_s16(sum);
    unsafe {
        vst1_u8(p, packed);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_writeback4_i32_u8<const W: usize, const H: usize>(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    x: usize,
    y: usize,
    v: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
) {
    debug_assert!(x + 4 <= W);
    debug_assert!(y < H);
    if out_w > W {
        let ox = x * 2;
        let oy = if out_h > H { y * 2 } else { y };
        let off0 = dst_off + oy * dst_stride + ox;
        neon_add4_i32_to_u8_expand_x2(dst, off0, v, rnd, nsh);
        if out_h > H {
            let off1 = off0 + dst_stride;
            neon_add4_i32_to_u8_expand_x2(dst, off1, v, rnd, nsh);
        }
    } else {
        let ox = x;
        let oy = if out_h > H { y * 2 } else { y };
        let off0 = dst_off + oy * dst_stride + ox;
        neon_add4_i32_to_u8(dst, off0, v, rnd, nsh);
        if out_h > H {
            let off1 = off0 + dst_stride;
            neon_add4_i32_to_u8(dst, off1, v, rnd, nsh);
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn add_tmp_to_dst_8bpc_neon(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    tmp: &[i32],
    tmp_stride: usize,
    w: usize,
    h: usize,
    sw: usize,
    sh: usize,
    rnd: i32,
    shift: i32,
) -> bool {
    debug_assert!(w == sw || w == sw * 2);
    debug_assert!(h == sh || h == sh * 2);
    debug_assert!(sw % 4 == 0);
    let rndv = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-shift);

    if w > sw {
        if h > sh {
            let mut ty = 0usize;
            let mut y = 0usize;
            while y < h {
                let mut x = 0usize;
                while x < sw {
                    let v = unsafe { vld1q_s32(tmp.as_ptr().add(ty * tmp_stride + x)) };
                    let off0 = dst_off + y * dst_stride + x * 2;
                    neon_add4_i32_to_u8_expand_x2(dst, off0, v, rndv, nsh);
                    neon_add4_i32_to_u8_expand_x2(dst, off0 + dst_stride, v, rndv, nsh);
                    x += 4;
                }
                ty += 1;
                y += 2;
            }
        } else {
            let mut y = 0usize;
            while y < h {
                let mut x = 0usize;
                while x < sw {
                    let v = unsafe { vld1q_s32(tmp.as_ptr().add(y * tmp_stride + x)) };
                    let off = dst_off + y * dst_stride + x * 2;
                    neon_add4_i32_to_u8_expand_x2(dst, off, v, rndv, nsh);
                    x += 4;
                }
                y += 1;
            }
        }
    } else if h > sh {
        let mut ty = 0usize;
        let mut y = 0usize;
        while y < h {
            let mut x = 0usize;
            while x < w {
                let v = unsafe { vld1q_s32(tmp.as_ptr().add(ty * tmp_stride + x)) };
                let off0 = dst_off + y * dst_stride + x;
                neon_add4_i32_to_u8(dst, off0, v, rndv, nsh);
                neon_add4_i32_to_u8(dst, off0 + dst_stride, v, rndv, nsh);
                x += 4;
            }
            ty += 1;
            y += 2;
        }
    } else {
        let mut y = 0usize;
        while y < h {
            let mut x = 0usize;
            while x < w {
                let v = unsafe { vld1q_s32(tmp.as_ptr().add(y * tmp_stride + x)) };
                let off = dst_off + y * dst_stride + x;
                neon_add4_i32_to_u8(dst, off, v, rndv, nsh);
                x += 4;
            }
            y += 1;
        }
    }
    true
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i16_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut scratch = [0i16; N];
        let mut y = 0usize;

        if FIRST_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = neon_identity_scale(W);
            while y + 4 <= nrows {
                let mut m = 0usize;
                while m < W {
                    let a0 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 0) * H, scale);
                    let a1 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 1) * H, scale);
                    let a2 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 2) * H, scale);
                    let a3 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 3) * H, scale);
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
                y += 4;
            }
        }

        while y + 8 <= nrows && FIRST_KIND == crate::itx_2d::TX_KIND_DCT && (W == 16 || W == 32) {
            if W == 16 {
                let lo = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let hi = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 16 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            } else {
                let lo = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let hi = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 32 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= nrows {
            if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
                let out = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 16 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
                let out = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 32 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 =
                            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let mut x = 0usize;
        if SECOND_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = neon_identity_scale(H);
            while x < W {
                let mut m = 0usize;
                while m < H {
                    let a = neon_identity_i16x4_scratch_to_i32(&scratch, x + m * W, scale);
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a);
                    m += 1;
                }
                x += 4;
            }
        }
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(&scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(&scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a0);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), a1);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), a2);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), a3);
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn tx_dequant_dense_neon_i16_rdm_impl<const N: usize, const W: usize, const H: usize>(
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
) {
    macro_rules! call_kind {
        ($first:expr, $second:expr) => {
            tx_dequant_dense_neon_i16_rdm_impl_kind::<N, W, H, { $first }, { $second }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            )
        };
    }
    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn tx_dequant_dense_neon_i16_rdm_impl_kind<
    const N: usize,
    const W: usize,
    const H: usize,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_impl_const::<N, W, H, true, FIRST_KIND, SECOND_KIND>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_impl_const::<N, W, H, false, FIRST_KIND, SECOND_KIND>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn tx_dequant_dense_neon_i16_rdm_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut scratch = [0i16; N];
        let mut y = 0usize;
        while y + 8 <= nrows && FIRST_KIND == crate::itx_2d::TX_KIND_DCT && (W == 16 || W == 32) {
            if W == 16 {
                let lo = neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let hi =
                    neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 16 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            } else {
                let lo = neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let hi =
                    neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 32 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= nrows {
            if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
                let out =
                    neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 16 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
                let out =
                    neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 32 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 =
                            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 = neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(
                            coeff,
                            y + (j + 1) * H,
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let mut x = 0usize;
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(&scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(&scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a0);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), a1);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), a2);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), a3);
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_8x8_neon_i32_impl(
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
    if is_rect2 {
        tx_dequant_8x8_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_8x8_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_8x8_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 64);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 2 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 8 {
                let g = neon_tx8_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, first_kind, x);
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 8 {
            tmp[y * 32..y * 32 + 8].fill(0);
            y += 1;
        }
        coeff[..64].fill(0);
        let mut x = 0usize;
        while x < 8 {
            let mut m = 0usize;
            while m < 8 {
                let g = neon_tx8_i32x4_from_tmp4(tmp, x, second_kind, m);
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), g[0]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), g[1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), g[2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), g[3]);
                m += 4;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_16x16_neon_i32_impl(
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
        idct_dequant_16x16_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_16x16_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_16x16_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 256);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        macro_rules! dct16x4_coeff {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let mut v = vld1q_s32(coeff.as_ptr().add($base + j * 16));
                    if IS_RECT2 {
                        v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                    }
                    a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]);
                    a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]);
                    a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]);
                    a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]);
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }
        macro_rules! dct16x4_tmp {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let v = vld1q_s32(tmp.as_ptr().add($base + j * 32));
                    a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]);
                    a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]);
                    a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]);
                    a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]);
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }

        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 16 {
                let g = dct16x4_coeff!(y, x);
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 16 {
            tmp[y * 32..y * 32 + 16].fill(0);
            y += 1;
        }
        coeff[..256].fill(0);

        let mut x = 0usize;
        while x < 16 {
            let mut m = 0usize;
            while m < 16 {
                let g = dct16x4_tmp!(x, m);
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), g[0]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), g[1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), g[2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), g[3]);
                m += 4;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_32x32_neon_i32_impl(
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
        idct_dequant_32x32_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_32x32_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_32x32_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 1024);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 8 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 32 {
                let g = neon_dct32_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, x);
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 32 {
            tmp[y * 32..y * 32 + 32].fill(0);
            y += 1;
        }
        coeff[..1024].fill(0);
        let mut x = 0usize;
        while x < 32 {
            let mut m = 0usize;
            while m < 32 {
                let g = neon_dct32_i32x4_from_tmp4(tmp, x, m);
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), g[0]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), g[1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), g[2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), g[3]);
                m += 4;
            }
            x += 4;
        }
    }
}

macro_rules! idct_neon_fn {
    ($pub:ident, $n:expr, $s:expr) => {
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $s }, { $s }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_neon_fn {
    ($pub:ident, $n:expr, $s:expr) => {
        #[inline]
        pub(crate) fn $pub(
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
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $s }, { $s }>(
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
                )
            };
        }
    };
}

macro_rules! idct_rect_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_rect_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub(
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
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
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
                )
            };
        }
    };
}

idct_neon_fn!(idct_dequant_4x4_neon, 16, 4);
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_8x8_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_8x8_neon_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}
#[inline]
pub(crate) fn idct_dequant_16x16_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        idct_dequant_16x16_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}
#[inline]
pub(crate) fn idct_dequant_32x32_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        idct_dequant_32x32_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}
idct_neon_fn!(idct_dequant_64x64_neon, 1024, 32);
iadst_neon_fn!(iadst_dequant_4x4_neon, 16, 4);
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn iadst_dequant_8x8_neon(
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
    tx_dequant_8x8_neon_i32_impl(
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
    )
}
#[inline]
pub(crate) fn iadst_dequant_16x16_neon(
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
    unsafe {
        iadst_dequant_16x16_neon_i32_impl(
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
        )
    }
}
idct_rect_neon_fn!(idct_dequant_4x8_neon, 32, 4, 8);
idct_rect_neon_fn!(idct_dequant_8x4_neon, 32, 8, 4);
idct_rect_neon_fn!(idct_dequant_8x16_neon, 128, 8, 16);
idct_rect_neon_fn!(idct_dequant_16x8_neon, 128, 16, 8);
idct_rect_neon_fn!(idct_dequant_16x32_neon, 512, 16, 32);
idct_rect_neon_fn!(idct_dequant_32x16_neon, 512, 32, 16);
idct_rect_neon_fn!(idct_dequant_4x16_neon, 64, 4, 16);
idct_rect_neon_fn!(idct_dequant_16x4_neon, 64, 16, 4);
idct_rect_neon_fn!(idct_dequant_8x32_neon, 256, 8, 32);
idct_rect_neon_fn!(idct_dequant_32x8_neon, 256, 32, 8);
idct_rect_neon_fn!(idct_dequant_4x32_neon, 128, 4, 32);
idct_rect_neon_fn!(idct_dequant_32x4_neon, 128, 32, 4);
iadst_rect_neon_fn!(iadst_dequant_4x8_neon, 32, 4, 8);
iadst_rect_neon_fn!(iadst_dequant_8x4_neon, 32, 8, 4);
iadst_rect_neon_fn!(iadst_dequant_8x16_neon, 128, 8, 16);
iadst_rect_neon_fn!(iadst_dequant_16x8_neon, 128, 16, 8);
iadst_rect_neon_fn!(iadst_dequant_4x16_neon, 64, 4, 16);
iadst_rect_neon_fn!(iadst_dequant_16x4_neon, 64, 16, 4);

macro_rules! idct_rect_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub_name(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                $impl_name(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                crate::itx_2d::TX_KIND_DCT,
                crate::itx_2d::TX_KIND_DCT,
            )
        }
    };
}

macro_rules! iadst_rect_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub_name(
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
            unsafe {
                $impl_name(
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
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
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
            tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
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
            )
        }
    };
}

idct_rect_rdm_fn!(
    idct_dequant_4x8_neon_rdm,
    idct_dequant_4x8_neon_rdm_impl,
    32,
    4,
    8
);
idct_rect_rdm_fn!(
    idct_dequant_8x4_neon_rdm,
    idct_dequant_8x4_neon_rdm_impl,
    32,
    8,
    4
);
idct_rect_rdm_fn!(
    idct_dequant_8x16_neon_rdm,
    idct_dequant_8x16_neon_rdm_impl,
    128,
    8,
    16
);
idct_rect_rdm_fn!(
    idct_dequant_16x8_neon_rdm,
    idct_dequant_16x8_neon_rdm_impl,
    128,
    16,
    8
);
idct_rect_rdm_fn!(
    idct_dequant_16x32_neon_rdm,
    idct_dequant_16x32_neon_rdm_impl,
    512,
    16,
    32
);
idct_rect_rdm_fn!(
    idct_dequant_32x16_neon_rdm,
    idct_dequant_32x16_neon_rdm_impl,
    512,
    32,
    16
);
idct_rect_rdm_fn!(
    idct_dequant_4x16_neon_rdm,
    idct_dequant_4x16_neon_rdm_impl,
    64,
    4,
    16
);
idct_rect_rdm_fn!(
    idct_dequant_16x4_neon_rdm,
    idct_dequant_16x4_neon_rdm_impl,
    64,
    16,
    4
);
idct_rect_rdm_fn!(
    idct_dequant_8x32_neon_rdm,
    idct_dequant_8x32_neon_rdm_impl,
    256,
    8,
    32
);
idct_rect_rdm_fn!(
    idct_dequant_32x8_neon_rdm,
    idct_dequant_32x8_neon_rdm_impl,
    256,
    32,
    8
);
idct_rect_rdm_fn!(
    idct_dequant_4x32_neon_rdm,
    idct_dequant_4x32_neon_rdm_impl,
    128,
    4,
    32
);
idct_rect_rdm_fn!(
    idct_dequant_32x4_neon_rdm,
    idct_dequant_32x4_neon_rdm_impl,
    128,
    32,
    4
);

iadst_rect_rdm_fn!(
    iadst_dequant_4x8_neon_rdm,
    iadst_dequant_4x8_neon_rdm_impl,
    32,
    4,
    8
);
iadst_rect_rdm_fn!(
    iadst_dequant_8x4_neon_rdm,
    iadst_dequant_8x4_neon_rdm_impl,
    32,
    8,
    4
);
iadst_rect_rdm_fn!(
    iadst_dequant_8x16_neon_rdm,
    iadst_dequant_8x16_neon_rdm_impl,
    128,
    8,
    16
);
iadst_rect_rdm_fn!(
    iadst_dequant_16x8_neon_rdm,
    iadst_dequant_16x8_neon_rdm_impl,
    128,
    16,
    8
);
iadst_rect_rdm_fn!(
    iadst_dequant_4x16_neon_rdm,
    iadst_dequant_4x16_neon_rdm_impl,
    64,
    4,
    16
);
iadst_rect_rdm_fn!(
    iadst_dequant_16x4_neon_rdm,
    iadst_dequant_16x4_neon_rdm_impl,
    64,
    16,
    4
);

#[inline]
pub(crate) fn idct_dequant_32x32_neon_rdm(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        idct_dequant_32x32_neon_rdm_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}
#[target_feature(enable = "rdm")]
#[inline]
fn idct_dequant_32x32_neon_rdm_impl(
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
        idct_dequant_32x32_neon_rdm_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_32x32_neon_rdm_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn idct_dequant_32x32_neon_rdm_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_neon_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        IS_RECT2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

// Low-bit-depth i16 coefficient entry points.

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i16_fused_8bpc_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut scratch = [0i16; N];
        let mut y = 0usize;

        if FIRST_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = neon_identity_scale(W);
            while y + 4 <= nrows {
                let mut m = 0usize;
                while m < W {
                    let a0 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 0) * H, scale);
                    let a1 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 1) * H, scale);
                    let a2 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 2) * H, scale);
                    let a3 =
                        neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 3) * H, scale);
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
                y += 4;
            }
        }

        while y + 8 <= nrows && FIRST_KIND == crate::itx_2d::TX_KIND_DCT && (W == 16 || W == 32) {
            if W == 16 {
                let lo = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let hi = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 16 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            } else {
                let lo = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let hi = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 32 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= nrows {
            if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
                let out = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 16 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
                let out = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 32 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 =
                            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
        let nsh1 = vdupq_n_s32(-shift1);

        let mut x = 0usize;
        if SECOND_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = neon_identity_scale(H);
            while x < W {
                let mut m = 0usize;
                while m < H {
                    let a = neon_identity_i16x4_scratch_to_i32(&scratch, x + m * W, scale);
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a, rnd1, nsh1,
                    );
                    m += 1;
                }
                x += 4;
            }
        }
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(&scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(&scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut scratch = [0i16; N];
        let mut y = 0usize;
        while y + 8 <= nrows && FIRST_KIND == crate::itx_2d::TX_KIND_DCT && (W == 16 || W == 32) {
            if W == 16 {
                let lo = neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let hi =
                    neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 16 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            } else {
                let lo = neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let hi =
                    neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y + 4);
                let mut m = 0usize;
                while m < 32 {
                    neon_store8x8_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        lo[m],
                        hi[m],
                        lo[m + 1],
                        hi[m + 1],
                        lo[m + 2],
                        hi[m + 2],
                        lo[m + 3],
                        hi[m + 3],
                        lo[m + 4],
                        hi[m + 4],
                        lo[m + 5],
                        hi[m + 5],
                        lo[m + 6],
                        hi[m + 6],
                        lo[m + 7],
                        hi[m + 7],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 8;
                }
            }
            y += 8;
        }
        while y + 4 <= nrows {
            if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
                let out =
                    neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 16 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
                let out =
                    neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, H>(coeff, y);
                let mut m = 0usize;
                while m < 32 {
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        out[m],
                        out[m + 1],
                        out[m + 2],
                        out[m + 3],
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 =
                            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 = neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(
                            coeff,
                            y + (j + 1) * H,
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        &mut scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
        let nsh1 = vdupq_n_s32(-shift1);

        let mut x = 0usize;
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(&scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(&scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(&scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn tx_dequant_dense_neon_i16_fused_8bpc_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    macro_rules! call_kind {
        ($first:expr, $second:expr) => {
            if is_rect2 {
                tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<
                    N,
                    W,
                    H,
                    true,
                    { $first },
                    { $second },
                >(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                )
            } else {
                tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<
                    N,
                    W,
                    H,
                    false,
                    { $first },
                    { $second },
                >(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                )
            }
        };
    }
    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    macro_rules! call_kind {
        ($first:expr, $second:expr) => {
            if is_rect2 {
                tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<
                    N,
                    W,
                    H,
                    true,
                    { $first },
                    { $second },
                >(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                )
            } else {
                tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<
                    N,
                    W,
                    H,
                    false,
                    { $first },
                    { $second },
                >(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                )
            }
        };
    }
    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_IDENTITY, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_IDENTITY,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_IDENTITY)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_IDENTITY) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_IDENTITY
            )
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_kind!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_kind!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => unreachable!(),
    }
}

macro_rules! neon_fused_match_body {
    ($call:ident, $coeff:ident, $dst:ident, $dst_off:ident, $dst_stride:ident, $out_w:ident, $out_h:ident, $eob:ident, $tx:ident, $is_rect2:ident, $shift0:ident, $row_clip_min:ident, $row_clip_max:ident, $shift1:ident, $first_kind:ident, $second_kind:ident) => {{
        match $tx {
            crate::levels::txsz::TX_4X4 => $call::<16, 4, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_8X8 => $call::<64, 8, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_16X16 => $call::<256, 16, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_32X32 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_64X64 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X8 => $call::<32, 4, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X4 => $call::<32, 8, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X16 => $call::<128, 8, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X8 => $call::<128, 16, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X32 => $call::<512, 16, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X16 => $call::<512, 32, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X64 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X32 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X16 => $call::<64, 4, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X4 => $call::<64, 16, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X32 => $call::<256, 8, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X8 => $call::<256, 32, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X64 => $call::<512, 16, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X16 => $call::<512, 32, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X32 => $call::<128, 4, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X4 => $call::<128, 32, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X64 => $call::<256, 8, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X8 => $call::<256, 32, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X64 => $call::<128, 4, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X4 => $call::<128, 32, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            _ => return false,
        }
        true
    }};
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_16x16_i16_neon_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<
            256,
            16,
            16,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<
            256,
            16,
            16,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_32x32_i16_neon_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<
            1024,
            32,
            32,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<
            1024,
            32,
            32,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "rdm")]
#[inline]
pub(crate) fn idct_dequant_16x16_i16_neon_rdm_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<
            256,
            16,
            16,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<
            256,
            16,
            16,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "rdm")]
#[inline]
pub(crate) fn idct_dequant_32x32_i16_neon_rdm_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<
            1024,
            32,
            32,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<
            1024,
            32,
            32,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn itx_dequant_i16_neon_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    if !crate::itx_2d::is_itx_dense_kind(first_kind)
        || !crate::itx_2d::is_itx_dense_kind(second_kind)
    {
        return false;
    }
    neon_fused_match_body!(
        tx_dequant_dense_neon_i16_fused_8bpc_impl,
        coeff,
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
        first_kind,
        second_kind
    )
}

#[target_feature(enable = "rdm")]
#[inline]
pub(crate) fn itx_dequant_i16_neon_rdm_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    if !crate::itx_2d::is_itx_dense_kind(first_kind)
        || !crate::itx_2d::is_itx_dense_kind(second_kind)
    {
        return false;
    }
    neon_fused_match_body!(
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl,
        coeff,
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
        first_kind,
        second_kind
    )
}

macro_rules! idct_i16_neon_fn {
    ($pub:ident, $n:expr, $s:expr) => {
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $s }, { $s }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_i16_neon_fn {
    ($pub:ident, $n:expr, $s:expr) => {
        #[inline]
        pub(crate) fn $pub(
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
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $s }, { $s }>(
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
                )
            };
        }
    };
}

macro_rules! idct_rect_i16_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $w }, { $h }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_rect_i16_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub(
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
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $w }, { $h }>(
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
                )
            };
        }
    };
}

macro_rules! idct_rect_i16_neon_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub_name(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                $impl_name(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            tx_dequant_dense_neon_i16_rdm_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                crate::itx_2d::TX_KIND_DCT,
                crate::itx_2d::TX_KIND_DCT,
            )
        }
    };
}

macro_rules! iadst_rect_i16_neon_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        pub(crate) fn $pub_name(
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
        ) {
            unsafe {
                $impl_name(
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
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
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
        ) {
            tx_dequant_dense_neon_i16_rdm_impl::<{ $n }, { $w }, { $h }>(
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
            )
        }
    };
}

idct_i16_neon_fn!(idct_dequant_4x4_i16_neon, 16, 4);
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_8x8_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_neon_i16_impl::<64, 8, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_16x16_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_impl::<16, 256>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_32x32_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_impl::<32, 1024>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "rdm")]
#[inline]
pub(crate) fn idct_dequant_32x32_i16_neon_rdm(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_rdm_impl::<32, 1024>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn idct_dequant_64x64_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_impl::<32, 1024>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}
iadst_i16_neon_fn!(iadst_dequant_4x4_i16_neon, 16, 4);
#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn iadst_dequant_8x8_i16_neon(
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
) {
    tx_dequant_dense_neon_i16_impl::<64, 8, 8>(
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
    )
}

#[target_feature(enable = "neon")]
#[inline]
pub(crate) fn iadst_dequant_16x16_i16_neon(
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
) {
    tx_dequant_dense_neon_i16_impl::<256, 16, 16>(
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
    )
}

idct_rect_i16_neon_fn!(idct_dequant_4x8_i16_neon, 32, 4, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x8_i16_neon_rdm,
    idct_dequant_4x8_i16_neon_rdm_impl,
    32,
    4,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_8x4_i16_neon, 32, 8, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x4_i16_neon_rdm,
    idct_dequant_8x4_i16_neon_rdm_impl,
    32,
    8,
    4
);
idct_rect_i16_neon_fn!(idct_dequant_8x16_i16_neon, 128, 8, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x16_i16_neon_rdm,
    idct_dequant_8x16_i16_neon_rdm_impl,
    128,
    8,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_16x8_i16_neon, 128, 16, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x8_i16_neon_rdm,
    idct_dequant_16x8_i16_neon_rdm_impl,
    128,
    16,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_16x32_i16_neon, 512, 16, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x32_i16_neon_rdm,
    idct_dequant_16x32_i16_neon_rdm_impl,
    512,
    16,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x16_i16_neon, 512, 32, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x16_i16_neon_rdm,
    idct_dequant_32x16_i16_neon_rdm_impl,
    512,
    32,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_4x16_i16_neon, 64, 4, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x16_i16_neon_rdm,
    idct_dequant_4x16_i16_neon_rdm_impl,
    64,
    4,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_16x4_i16_neon, 64, 16, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x4_i16_neon_rdm,
    idct_dequant_16x4_i16_neon_rdm_impl,
    64,
    16,
    4
);
idct_rect_i16_neon_fn!(idct_dequant_8x32_i16_neon, 256, 8, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x32_i16_neon_rdm,
    idct_dequant_8x32_i16_neon_rdm_impl,
    256,
    8,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x8_i16_neon, 256, 32, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x8_i16_neon_rdm,
    idct_dequant_32x8_i16_neon_rdm_impl,
    256,
    32,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_4x32_i16_neon, 128, 4, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x32_i16_neon_rdm,
    idct_dequant_4x32_i16_neon_rdm_impl,
    128,
    4,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x4_i16_neon, 128, 32, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x4_i16_neon_rdm,
    idct_dequant_32x4_i16_neon_rdm_impl,
    128,
    32,
    4
);
iadst_rect_i16_neon_fn!(iadst_dequant_4x8_i16_neon, 32, 4, 8);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_4x8_i16_neon_rdm,
    iadst_dequant_4x8_i16_neon_rdm_impl,
    32,
    4,
    8
);
iadst_rect_i16_neon_fn!(iadst_dequant_8x4_i16_neon, 32, 8, 4);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_8x4_i16_neon_rdm,
    iadst_dequant_8x4_i16_neon_rdm_impl,
    32,
    8,
    4
);
iadst_rect_i16_neon_fn!(iadst_dequant_8x16_i16_neon, 128, 8, 16);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_8x16_i16_neon_rdm,
    iadst_dequant_8x16_i16_neon_rdm_impl,
    128,
    8,
    16
);
iadst_rect_i16_neon_fn!(iadst_dequant_16x8_i16_neon, 128, 16, 8);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_16x8_i16_neon_rdm,
    iadst_dequant_16x8_i16_neon_rdm_impl,
    128,
    16,
    8
);
iadst_rect_i16_neon_fn!(iadst_dequant_4x16_i16_neon, 64, 4, 16);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_4x16_i16_neon_rdm,
    iadst_dequant_4x16_i16_neon_rdm_impl,
    64,
    4,
    16
);
iadst_rect_i16_neon_fn!(iadst_dequant_16x4_i16_neon, 64, 16, 4);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_16x4_i16_neon_rdm,
    iadst_dequant_16x4_i16_neon_rdm_impl,
    64,
    16,
    4
);
