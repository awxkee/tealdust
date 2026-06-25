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

use crate::itx_1d::DctWide;
use crate::itx_2d::{Adst2dBackend, Dct2dBackend, DctSimd4, ITX_TMP_PIXELS};
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
fn neon_tx16_i16x8_impl(s: &[int16x8_t; 16], kind: usize) -> [(int32x4_t, int32x4_t); 16] {
    let z = (vdupq_n_s32(0), vdupq_n_s32(0));
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            let k0 = match kind {
                crate::itx_2d::TX_KIND_DCT => crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m] as i16,
                crate::itx_2d::TX_KIND_ADST => crate::itx_1d::ADST16_KERNEL_ROWS[m][j] as i16,
                crate::itx_2d::TX_KIND_FLIPADST => {
                    crate::itx_1d::FLIPADST16_KERNEL_ROWS[m][j] as i16
                }
                _ => unreachable!(),
            };
            let k1 = match kind {
                crate::itx_2d::TX_KIND_DCT => {
                    crate::itx_2d::DCT16_DENSE_KERNEL[(j + 1) * 16 + m] as i16
                }
                crate::itx_2d::TX_KIND_ADST => crate::itx_1d::ADST16_KERNEL_ROWS[m][j + 1] as i16,
                crate::itx_2d::TX_KIND_FLIPADST => {
                    crate::itx_1d::FLIPADST16_KERNEL_ROWS[m][j + 1] as i16
                }
                _ => unreachable!(),
            };
            let lo0 = vmlal_n_s16(acc.0, vget_low_s16(s[j]), k0);
            let lo1 = vmlal_n_s16(lo0, vget_low_s16(s[j + 1]), k1);
            let hi0 = vmlal_high_n_s16(acc.1, s[j], k0);
            let hi1 = vmlal_high_n_s16(hi0, s[j + 1], k1);
            acc = (lo1, hi1);
            j += 2;
        }
        out[m] = acc;
        m += 1;
    }
    out
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
                if is_rect2 {
                    v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                }
                s[j] = v;
                j += 1;
            }
            let out = neon_tx16_i32x4_impl(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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
unsafe fn iadst_dequant_16x16_neon_i16_impl(
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
        while y + 8 <= ncols {
            let mut s = [vdupq_n_s16(0); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = neon_load8_i16_impl(coeff, y + j * 16, is_rect2);
                j += 1;
            }
            let out = neon_tx16_i16x8_impl(&s, first_kind);
            let g0 = [
                out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
            ];
            let g1 = [
                out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15],
            ];
            neon_store8x8_wide_clip(tmp, y * 32, &g0, rnd, nsh, minv, maxv);
            neon_store8x8_wide_clip(tmp, y * 32 + 8, &g1, rnd, nsh, minv, maxv);
            y += 8;
        }
        while y + 4 <= ncols {
            let mut s = [vdupq_n_s32(0); 16];
            let mut j = 0usize;
            while j < 16 {
                let v = neon_load4_i16_impl(coeff, y + j * 16, is_rect2);
                s[j] = vmovl_s16(vget_low_s16(v));
                j += 1;
            }
            let out = neon_tx16_i32x4_impl(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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
            let mut s = [vdupq_n_s16(0); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = neon_load8_narrow_i32_impl(tmp, x + j * 32);
                j += 1;
            }
            let out = neon_tx16_i16x8_impl(&s, second_kind);
            j = 0;
            while j < 16 {
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32), out[j].0);
                vst1q_s32(tmp.as_mut_ptr().add(x + 4 + j * 32), out[j].1);
                j += 1;
            }
            x += 8;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_load8_i16_impl(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
    unsafe {
        let x = vld1q_s16(src.as_ptr().add(off));
        if rect2 {
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), vget_low_s16(x), 181));
            let hi = vshrq_n_s32::<8>(vmlal_high_n_s16(vdupq_n_s32(128), x, 181));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        } else {
            x
        }
    }
}
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_load4_i16_impl(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
    unsafe {
        let x = vcombine_s16(vld1_s16(src.as_ptr().add(off)), vdup_n_s16(0));
        if rect2 {
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), vget_low_s16(x), 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        } else {
            x
        }
    }
}
#[target_feature(enable = "rdm")]
#[inline]
unsafe fn neon_rdm_load8_i16_impl(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
    unsafe {
        let x = vld1q_s16(src.as_ptr().add(off));
        if rect2 {
            vqrdmulhq_s16(x, vdupq_n_s16(0x5a80))
        } else {
            x
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_load8_narrow_i32_impl(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        let lo = vld1q_s32(src.as_ptr().add(off));
        let hi = vld1q_s32(src.as_ptr().add(off + 4));
        vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_store4x4_i32_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    v: &[int32x4_t; 4],
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        macro_rules! clip {
            ($x:expr) => {{ vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32($x, rnd), nsh), minv), maxv) }};
        }
        let c0 = clip!(v[0]);
        let c1 = clip!(v[1]);
        let c2 = clip!(v[2]);
        let c3 = clip!(v[3]);
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
fn neon_store4x4_wide_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    acc: &[(int32x4_t, int32x4_t); 4],
    high: bool,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    let v = [
        if high { acc[0].1 } else { acc[0].0 },
        if high { acc[1].1 } else { acc[1].0 },
        if high { acc[2].1 } else { acc[2].0 },
        if high { acc[3].1 } else { acc[3].0 },
    ];
    neon_store4x4_i32_clip(tmp, off, &v, rnd, nsh, minv, maxv);
}
#[target_feature(enable = "neon")]
#[inline]
fn neon_store8x8_wide_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    acc: &[(int32x4_t, int32x4_t); 8],
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    let g0 = [acc[0], acc[1], acc[2], acc[3]];
    let g1 = [acc[4], acc[5], acc[6], acc[7]];
    neon_store4x4_wide_clip(tmp, off, &g0, false, rnd, nsh, minv, maxv);
    neon_store4x4_wide_clip(tmp, off + 4 * 32, &g0, true, rnd, nsh, minv, maxv);
    neon_store4x4_wide_clip(tmp, off + 4, &g1, false, rnd, nsh, minv, maxv);
    neon_store4x4_wide_clip(tmp, off + 4 * 32 + 4, &g1, true, rnd, nsh, minv, maxv);
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_load4_i16_i32(src: &[i16], off: usize, rect2: bool) -> int32x4_t {
    unsafe {
        let x = vld1_s16(src.as_ptr().add(off));
        let mut v = vmovl_s16(x);
        if rect2 {
            v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
        }
        v
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn neon_rdm_load4_i16_i32(src: &[i16], off: usize, rect2: bool) -> int32x4_t {
    unsafe {
        let x = vld1_s16(src.as_ptr().add(off));
        let mut v16 = vcombine_s16(x, vdup_n_s16(0));
        if rect2 {
            v16 = vqrdmulhq_s16(v16, vdupq_n_s16(0x5a80));
        }
        vmovl_s16(vget_low_s16(v16))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn neon_dct32_i32x4_from_coeff4(
    coeff: &[i32],
    base: usize,
    rect2: bool,
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
            if rect2 {
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
fn neon_dct32_i32x4_from_i16_coeff4(
    coeff: &[i16],
    base: usize,
    rect2: bool,
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
            let v = neon_load4_i16_i32(coeff, base + j * 32, rect2);
            a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
            a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]);
            a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]);
            a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]);
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[target_feature(enable = "rdm")]
#[inline]
fn neon_dct32_i32x4_from_i16_rdm_coeff4(
    coeff: &[i16],
    base: usize,
    rect2: bool,
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
            let v = neon_rdm_load4_i16_i32(coeff, base + j * 32, rect2);
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
fn neon_tx8_i32x4_from_coeff4(
    coeff: &[i32],
    base: usize,
    rect2: bool,
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
            if rect2 {
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
fn neon_tx8_i32x4_from_i16_coeff4(
    coeff: &[i16],
    base: usize,
    rect2: bool,
    kind: usize,
    m: usize,
) -> [int32x4_t; 4] {
    let z = vdupq_n_s32(0);
    let mut a0 = z;
    let mut a1 = z;
    let mut a2 = z;
    let mut a3 = z;
    let mut j = 0usize;
    while j < 8 {
        let v = neon_load4_i16_i32(coeff, base + j * 8, rect2);
        a0 = vmlaq_n_s32(a0, v, tx8_coeff(kind, m, j));
        a1 = vmlaq_n_s32(a1, v, tx8_coeff(kind, m + 1, j));
        a2 = vmlaq_n_s32(a2, v, tx8_coeff(kind, m + 2, j));
        a3 = vmlaq_n_s32(a3, v, tx8_coeff(kind, m + 3, j));
        j += 1;
    }
    [a0, a1, a2, a3]
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
                    if is_rect2 {
                        v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                    }
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(first_kind, W, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(first_kind, W, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(first_kind, W, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(first_kind, W, m + 3, j));
                    j += 1;
                }
                let g = [a0, a1, a2, a3];
                neon_store4x4_i32_clip(tmp, y * 32 + m, &g, rnd, nsh, minv, maxv);
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
                    let v = neon_load4_i16_i32(coeff, y + j * H, is_rect2);
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(first_kind, W, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(first_kind, W, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(first_kind, W, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(first_kind, W, m + 3, j));
                    j += 1;
                }
                let g = [a0, a1, a2, a3];
                neon_store4x4_i32_clip(tmp, y * 32 + m, &g, rnd, nsh, minv, maxv);
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
                    let v = neon_rdm_load4_i16_i32(coeff, y + j * H, is_rect2);
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(first_kind, W, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(first_kind, W, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(first_kind, W, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(first_kind, W, m + 3, j));
                    j += 1;
                }
                let g = [a0, a1, a2, a3];
                neon_store4x4_i32_clip(tmp, y * 32 + m, &g, rnd, nsh, minv, maxv);
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
                let g = neon_tx8_i32x4_from_coeff4(coeff, y, is_rect2, first_kind, x);
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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
fn tx_dequant_8x8_neon_i16_impl(
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
                let g = neon_tx8_i32x4_from_i16_coeff4(coeff, y, is_rect2, first_kind, x);
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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
                    if is_rect2 {
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
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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
fn idct_dequant_16x16_neon_i16_impl(
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
                    let v16 = neon_load4_i16_impl(coeff, $base + j * 16, is_rect2);
                    let v = vmovl_s16(vget_low_s16(v16));
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
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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
                let g = neon_dct32_i32x4_from_coeff4(coeff, y, is_rect2, x);
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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

#[target_feature(enable = "neon")]
#[inline]
fn idct_dequant_32x32_neon_i16_impl(
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
                let g = neon_dct32_i32x4_from_i16_coeff4(coeff, y, is_rect2, x);
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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

#[target_feature(enable = "rdm")]
#[inline]
fn idct_dequant_32x32_neon_i16_rdm_impl(
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
                let g = neon_dct32_i32x4_from_i16_rdm_coeff4(coeff, y, is_rect2, x);
                neon_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
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

#[derive(Clone, Copy)]
pub(crate) struct NeonI32x4(int32x4_t);

impl crate::itx_1d::DctLane for NeonI32x4 {
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn zero() -> Self {
        NeonI32x4(vdupq_n_s32(0))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn add(self, o: Self) -> Self {
        NeonI32x4(vaddq_s32(self.0, o.0))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sub(self, o: Self) -> Self {
        NeonI32x4(vsubq_s32(self.0, o.0))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul(self, k: Self) -> Self {
        NeonI32x4(vmulq_s32(self.0, k.0))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dup_load(table: &[i32], idx: usize) -> Self {
        // SAFETY: callers index within the kernel tables.
        NeonI32x4(unsafe { vld1q_dup_s32(table.as_ptr().add(idx)) })
    }
    type Coeffs = int32x4_t;
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_coeffs(table: &[i32], idx: usize) -> int32x4_t {
        // SAFETY: callers index a 4-wide group within the kernel tables.
        unsafe { vld1q_s32(table.as_ptr().add(idx)) }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul_add_lane<const LANE: i32>(self, x: Self, c: int32x4_t) -> Self {
        NeonI32x4(vmlaq_laneq_s32::<LANE>(self.0, x.0, c))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul_add(self, x: Self, k: Self) -> Self {
        NeonI32x4(vmlaq_s32(self.0, x.0, k.0))
    }
}

pub(crate) struct NeonWide;

impl crate::itx_1d::DctWide for NeonWide {
    type In = int16x8_t;
    type Acc = (int32x4_t, int32x4_t);
    type Coeffs = int16x8_t;
    type Clip = (int32x4_t, int32x4_t, int32x4_t, int32x4_t);
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn zero() -> Self::Acc {
        (vdupq_n_s32(0), vdupq_n_s32(0))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        (vaddq_s32(a.0, b.0), vaddq_s32(a.1, b.1))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        (vsubq_s32(a.0, b.0), vsubq_s32(a.1, b.1))
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load_coeffs(table: &[i16], idx: usize) -> int16x8_t {
        unsafe { vld1q_s16(table.as_ptr().add(idx)) }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mul_add_lane<const LANE: i32>(
        acc: Self::Acc,
        x: int16x8_t,
        c: int16x8_t,
    ) -> Self::Acc {
        (
            vmlal_laneq_s16::<LANE>(acc.0, vget_low_s16(x), c),
            vmlal_high_laneq_s16::<LANE>(acc.1, x, c),
        )
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            let hi = vld1q_s32(src.as_ptr().add(off + 4));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            // Exact NEON fallback for CPUs without FEAT_RDM: keep the rect2
            // normalization in i32, then narrow exactly like `load8_narrow`.
            let lo = vld1q_s32(src.as_ptr().add(off));
            let hi = vld1q_s32(src.as_ptr().add(off + 4));
            let r = vdupq_n_s32(128);
            let lo = vshrq_n_s32::<8>(vmlaq_n_s32(r, lo, 181));
            let hi = vshrq_n_s32::<8>(vmlaq_n_s32(r, hi, 181));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            let lo = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), lo, 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load8_i16(src: &[i16], off: usize) -> int16x8_t {
        debug_assert!(off + 8 <= src.len());
        unsafe { vld1q_s16(src.as_ptr().add(off)) }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> int16x8_t {
        unsafe {
            let x = Self::load8_i16(src, off);
            let r = vdupq_n_s32(128);
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(r, vget_low_s16(x), 181));
            let hi = vshrq_n_s32::<8>(vmlal_high_n_s16(r, x, 181));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[inline]
    unsafe fn load4_i16(src: &[i16], off: usize) -> int16x8_t {
        debug_assert!(off + 4 <= src.len());
        unsafe { vcombine_s16(vld1_s16(src.as_ptr().add(off)), vdup_n_s16(0)) }
    }
    #[inline]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> int16x8_t {
        unsafe {
            let x = Self::load4_i16(src, off);
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), vget_low_s16(x), 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        (
            vdupq_n_s32(rnd),
            vdupq_n_s32(-shift),
            vdupq_n_s32(min),
            vdupq_n_s32(max),
        )
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, nsh, minv, maxv) = clip;
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(acc.0, rnd), nsh), minv), maxv);
            let hi = vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(acc.1, rnd), nsh), minv), maxv);
            let p = dst.as_mut_ptr().add(off);
            vst1q_lane_s32::<0>(p.add(0 * stride), lo);
            vst1q_lane_s32::<1>(p.add(1 * stride), lo);
            vst1q_lane_s32::<2>(p.add(2 * stride), lo);
            vst1q_lane_s32::<3>(p.add(3 * stride), lo);
            vst1q_lane_s32::<0>(p.add(4 * stride), hi);
            vst1q_lane_s32::<1>(p.add(5 * stride), hi);
            vst1q_lane_s32::<2>(p.add(6 * stride), hi);
            vst1q_lane_s32::<3>(p.add(7 * stride), hi);
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, nsh, minv, maxv) = clip;
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(acc.0, rnd), nsh), minv), maxv);
            let p = dst.as_mut_ptr().add(off);
            vst1q_lane_s32::<0>(p.add(0 * stride), lo);
            vst1q_lane_s32::<1>(p.add(1 * stride), lo);
            vst1q_lane_s32::<2>(p.add(2 * stride), lo);
            vst1q_lane_s32::<3>(p.add(3 * stride), lo);
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store4x4_strided_clip<const HIGH: bool>(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 4],
        clip: Self::Clip,
    ) {
        unsafe {
            #[target_feature(enable = "neon")]
            #[inline]
            fn clip_vec(
                v: int32x4_t,
                rnd: int32x4_t,
                nsh: int32x4_t,
                minv: int32x4_t,
                maxv: int32x4_t,
            ) -> int32x4_t {
                vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(v, rnd), nsh), minv), maxv)
            }
            let (rnd, nsh, minv, maxv) = clip;
            let c0 = clip_vec(if HIGH { acc[0].1 } else { acc[0].0 }, rnd, nsh, minv, maxv);
            let c1 = clip_vec(if HIGH { acc[1].1 } else { acc[1].0 }, rnd, nsh, minv, maxv);
            let c2 = clip_vec(if HIGH { acc[2].1 } else { acc[2].0 }, rnd, nsh, minv, maxv);
            let c3 = clip_vec(if HIGH { acc[3].1 } else { acc[3].0 }, rnd, nsh, minv, maxv);

            let t01 = vtrnq_s32(c0, c1);
            let t23 = vtrnq_s32(c2, c3);
            let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
            let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
            let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
            let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));

            vst1q_s32(dst.as_mut_ptr().add(off), r0);
            vst1q_s32(dst.as_mut_ptr().add(off + stride), r1);
            vst1q_s32(dst.as_mut_ptr().add(off + 2 * stride), r2);
            vst1q_s32(dst.as_mut_ptr().add(off + 3 * stride), r3);
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            vst1q_s32(dst.as_mut_ptr().add(off), acc.0);
            vst1q_s32(dst.as_mut_ptr().add(off + 4), acc.1);
        }
    }
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            vst1q_s32(dst.as_mut_ptr().add(off), acc.0);
        }
    }
}

#[target_feature(enable = "rdm")]
#[inline]
unsafe fn load8_rect2_narrow_rdm(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        // dav2d-style rect2 normalization. SQRDMULH by 0x5a80 is exactly
        // `(v * 181 + 128) >> 8` for valid s16 lanes, and avoids widening the
        // row-pipeline input twice.
        let lo = vld1q_s32(src.as_ptr().add(off));
        let hi = vld1q_s32(src.as_ptr().add(off + 4));
        let v = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
        vqrdmulhq_s16(v, vdupq_n_s16(0x5a80))
    }
}

#[target_feature(enable = "rdm")]
#[inline]
unsafe fn load4_rect2_narrow_rdm(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        // Same RDM rect2 normalization for 4 active lanes; high lanes stay zero.
        vqrdmulhq_s16(NeonWide::load4_narrow(src, off), vdupq_n_s16(0x5a80))
    }
}

#[target_feature(enable = "rdm")]
#[inline]
unsafe fn load8_rect2_i16_rdm(src: &[i16], off: usize) -> int16x8_t {
    unsafe { vqrdmulhq_s16(NeonWide::load8_i16(src, off), vdupq_n_s16(0x5a80)) }
}

#[target_feature(enable = "rdm")]
#[inline]
unsafe fn load4_rect2_i16_rdm(src: &[i16], off: usize) -> int16x8_t {
    unsafe { vqrdmulhq_s16(NeonWide::load4_i16(src, off), vdupq_n_s16(0x5a80)) }
}

pub(crate) struct NeonWideRdm;

impl crate::itx_1d::DctWide for NeonWideRdm {
    type In = int16x8_t;
    type Acc = (int32x4_t, int32x4_t);
    type Coeffs = int16x8_t;
    type Clip = (int32x4_t, int32x4_t, int32x4_t, int32x4_t);

    #[inline]
    unsafe fn zero() -> Self::Acc {
        unsafe { NeonWide::zero() }
    }

    #[inline]
    unsafe fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { NeonWide::add(a, b) }
    }

    #[inline]
    unsafe fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { NeonWide::sub(a, b) }
    }

    #[inline]
    unsafe fn load_coeffs(table: &[i16], idx: usize) -> Self::Coeffs {
        unsafe { NeonWide::load_coeffs(table, idx) }
    }

    #[inline]
    unsafe fn mul_add_lane<const LANE: i32>(
        acc: Self::Acc,
        x: Self::In,
        c: Self::Coeffs,
    ) -> Self::Acc {
        unsafe { NeonWide::mul_add_lane::<LANE>(acc, x, c) }
    }

    #[inline]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { NeonWide::load8_narrow(src, off) }
    }

    #[inline]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { load8_rect2_narrow_rdm(src, off) }
    }

    #[inline]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { NeonWide::load4_narrow(src, off) }
    }

    #[inline]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { load4_rect2_narrow_rdm(src, off) }
    }
    #[inline]
    unsafe fn load8_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { NeonWide::load8_i16(src, off) }
    }

    #[inline]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { load8_rect2_i16_rdm(src, off) }
    }

    #[inline]
    unsafe fn load4_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { NeonWide::load4_i16(src, off) }
    }

    #[inline]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { load4_rect2_i16_rdm(src, off) }
    }

    #[inline]
    unsafe fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        unsafe { NeonWide::make_clip(rnd, shift, min, max) }
    }

    #[inline]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store8_strided_clip(dst, off, stride, acc, clip) }
    }

    #[inline]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store4_strided_clip(dst, off, stride, acc, clip) }
    }

    #[inline]
    unsafe fn store4x4_strided_clip<const HIGH: bool>(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 4],
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store4x4_strided_clip::<HIGH>(dst, off, stride, acc, clip) }
    }

    #[inline]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { NeonWide::store8(dst, off, acc) }
    }

    #[inline]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { NeonWide::store4(dst, off, acc) }
    }
}

pub(crate) struct NeonDct2d;

impl DctSimd4 for NeonDct2d {
    type V = NeonI32x4;
    type Wide = NeonWide;
    #[inline]
    unsafe fn zero() -> Self::V {
        NeonI32x4(unsafe { vdupq_n_s32(0) })
    }

    #[inline]
    unsafe fn splat(v: i32) -> Self::V {
        NeonI32x4(unsafe { vdupq_n_s32(v) })
    }

    #[inline]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vaddq_s32(a.0, b.0) })
    }

    #[inline]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vsubq_s32(a.0, b.0) })
    }

    #[inline]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vmulq_s32(a.0, b.0) })
    }

    #[inline]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe {
            let scaled = vmlaq_n_s32(vdupq_n_s32(128), a.0, 181);
            NeonI32x4(vshrq_n_s32::<8>(scaled))
        }
    }

    #[inline]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_ptr().add(off) };
        NeonI32x4(unsafe { vld1q_s32(p) })
    }

    #[inline]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_mut_ptr().add(off) };
        unsafe { vst1q_s32(p, v.0) };
    }

    #[inline]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) };
        NeonI32x4(unsafe { vld1q_s32(p) })
    }

    #[inline]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) };
        NeonI32x4(unsafe { vmovl_s16(vld1_s16(p)) })
    }

    #[inline]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), v.0) };
        out
    }

    #[inline]
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
        unsafe {
            #[inline]
            unsafe fn clip_vec(
                v: int32x4_t,
                rnd: int32x4_t,
                sh: int32x4_t,
                minv: int32x4_t,
                maxv: int32x4_t,
            ) -> int32x4_t {
                unsafe { vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(v, rnd), sh), minv), maxv) }
            }

            let rnd = vdupq_n_s32(rnd);
            let sh = vdupq_n_s32(-shift);
            let minv = vdupq_n_s32(min);
            let maxv = vdupq_n_s32(max);

            let c0 = clip_vec(v[0].0, rnd, sh, minv, maxv);
            let c1 = clip_vec(v[1].0, rnd, sh, minv, maxv);
            let c2 = clip_vec(v[2].0, rnd, sh, minv, maxv);
            let c3 = clip_vec(v[3].0, rnd, sh, minv, maxv);

            // Transpose columns-as-lanes into row vectors.
            let t01 = vtrnq_s32(c0, c1);
            let t23 = vtrnq_s32(c2, c3);
            let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
            let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
            let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
            let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));

            vst1q_s32(tmp.as_mut_ptr().add(off), r0);
            vst1q_s32(tmp.as_mut_ptr().add(off + stride), r1);
            vst1q_s32(tmp.as_mut_ptr().add(off + 2 * stride), r2);
            vst1q_s32(tmp.as_mut_ptr().add(off + 3 * stride), r3);
        }
    }
}

pub(crate) struct NeonDct2dRdm;

impl DctSimd4 for NeonDct2dRdm {
    type V = NeonI32x4;
    type Wide = NeonWideRdm;

    #[inline]
    unsafe fn zero() -> Self::V {
        unsafe { NeonDct2d::zero() }
    }

    #[inline]
    unsafe fn splat(v: i32) -> Self::V {
        unsafe { NeonDct2d::splat(v) }
    }

    #[inline]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::add(a, b) }
    }

    #[inline]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::sub(a, b) }
    }

    #[inline]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::mul(a, b) }
    }

    #[inline]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe { NeonDct2d::rect2_scale(a) }
    }

    #[inline]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        unsafe { NeonDct2d::load(tmp, off) }
    }

    #[inline]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        unsafe { NeonDct2d::store(tmp, off, v) }
    }

    #[inline]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        unsafe { NeonDct2d::load_slice(src, off) }
    }

    #[inline]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        unsafe { NeonDct2d::load_slice_i16(src, off) }
    }

    #[inline]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        unsafe { NeonDct2d::to_array(v) }
    }

    #[inline]
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
        unsafe { NeonDct2d::store4x4_clip(tmp, off, stride, v, rnd, shift, min, max) }
    }
}

impl Dct2dBackend for NeonDct2d {
    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<16, 4, 4>(
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

    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<64, 8, 8>(
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

    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<256, 16, 16>(
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

    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<1024, 32, 32>(
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

    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<1024, 32, 32>(
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
}

impl Adst2dBackend for NeonDct2d {
    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<16, 4, 4>(
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

    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<64, 8, 8>(
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

    #[inline]
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
        unsafe {
            tx_dequant_dense_neon_i32_impl::<256, 16, 16>(
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
}

macro_rules! idct_neon_fn {
    ($pub:ident, $backend:ty, $n:expr, $s:expr) => {
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
    ($pub:ident, $backend:ty, $n:expr, $s:expr) => {
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
    ($pub:ident, $backend:ty, $n:expr, $w:expr, $h:expr) => {
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
    ($pub:ident, $backend:ty, $n:expr, $w:expr, $h:expr) => {
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

idct_neon_fn!(idct_dequant_4x4_neon, NeonDct2d, 16, 4);
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
    unsafe {
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
idct_neon_fn!(idct_dequant_64x64_neon, NeonDct2d, 1024, 32);
iadst_neon_fn!(iadst_dequant_4x4_neon, NeonDct2d, 16, 4);
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
    unsafe {
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
idct_rect_neon_fn!(idct_dequant_4x8_neon, NeonDct2d, 32, 4, 8);
idct_rect_neon_fn!(idct_dequant_8x4_neon, NeonDct2d, 32, 8, 4);
idct_rect_neon_fn!(idct_dequant_8x16_neon, NeonDct2d, 128, 8, 16);
idct_rect_neon_fn!(idct_dequant_16x8_neon, NeonDct2d, 128, 16, 8);
idct_rect_neon_fn!(idct_dequant_16x32_neon, NeonDct2d, 512, 16, 32);
idct_rect_neon_fn!(idct_dequant_32x16_neon, NeonDct2d, 512, 32, 16);
idct_rect_neon_fn!(idct_dequant_4x16_neon, NeonDct2d, 64, 4, 16);
idct_rect_neon_fn!(idct_dequant_16x4_neon, NeonDct2d, 64, 16, 4);
idct_rect_neon_fn!(idct_dequant_8x32_neon, NeonDct2d, 256, 8, 32);
idct_rect_neon_fn!(idct_dequant_32x8_neon, NeonDct2d, 256, 32, 8);
idct_rect_neon_fn!(idct_dequant_4x32_neon, NeonDct2d, 128, 4, 32);
idct_rect_neon_fn!(idct_dequant_32x4_neon, NeonDct2d, 128, 32, 4);
iadst_rect_neon_fn!(iadst_dequant_4x8_neon, NeonDct2d, 32, 4, 8);
iadst_rect_neon_fn!(iadst_dequant_8x4_neon, NeonDct2d, 32, 8, 4);
iadst_rect_neon_fn!(iadst_dequant_8x16_neon, NeonDct2d, 128, 8, 16);
iadst_rect_neon_fn!(iadst_dequant_16x8_neon, NeonDct2d, 128, 16, 8);
iadst_rect_neon_fn!(iadst_dequant_4x16_neon, NeonDct2d, 64, 4, 16);
iadst_rect_neon_fn!(iadst_dequant_16x4_neon, NeonDct2d, 64, 16, 4);

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

// Low-bit-depth i16 coefficient entry points.

macro_rules! idct_i16_neon_fn {
    ($pub:ident, $backend:ty, $n:expr, $s:expr) => {
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
    ($pub:ident, $backend:ty, $n:expr, $s:expr) => {
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
    ($pub:ident, $backend:ty, $n:expr, $w:expr, $h:expr) => {
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
    ($pub:ident, $backend:ty, $n:expr, $w:expr, $h:expr) => {
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

macro_rules! idct_i16_neon_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $s:expr) => {
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
            unsafe {
                tx_dequant_dense_neon_i16_rdm_impl::<{ $n }, { $s }, { $s }>(
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
            unsafe {
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
            };
        }
    };
}

idct_i16_neon_fn!(idct_dequant_4x4_i16_neon, NeonDct2d, 16, 4);
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
    unsafe {
        tx_dequant_8x8_neon_i16_impl(
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
}
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
    unsafe {
        idct_dequant_16x16_neon_i16_impl(
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
    unsafe {
        idct_dequant_32x32_neon_i16_impl(
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
    unsafe {
        idct_dequant_32x32_neon_i16_rdm_impl(
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
idct_i16_neon_fn!(idct_dequant_64x64_i16_neon, NeonDct2d, 1024, 32);
iadst_i16_neon_fn!(iadst_dequant_4x4_i16_neon, NeonDct2d, 16, 4);
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
    unsafe {
        tx_dequant_8x8_neon_i16_impl(
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
    unsafe {
        iadst_dequant_16x16_neon_i16_impl(
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
idct_rect_i16_neon_fn!(idct_dequant_4x8_i16_neon, NeonDct2d, 32, 4, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x8_i16_neon_rdm,
    idct_dequant_4x8_i16_neon_rdm_impl,
    32,
    4,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_8x4_i16_neon, NeonDct2d, 32, 8, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x4_i16_neon_rdm,
    idct_dequant_8x4_i16_neon_rdm_impl,
    32,
    8,
    4
);
idct_rect_i16_neon_fn!(idct_dequant_8x16_i16_neon, NeonDct2d, 128, 8, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x16_i16_neon_rdm,
    idct_dequant_8x16_i16_neon_rdm_impl,
    128,
    8,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_16x8_i16_neon, NeonDct2d, 128, 16, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x8_i16_neon_rdm,
    idct_dequant_16x8_i16_neon_rdm_impl,
    128,
    16,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_16x32_i16_neon, NeonDct2d, 512, 16, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x32_i16_neon_rdm,
    idct_dequant_16x32_i16_neon_rdm_impl,
    512,
    16,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x16_i16_neon, NeonDct2d, 512, 32, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x16_i16_neon_rdm,
    idct_dequant_32x16_i16_neon_rdm_impl,
    512,
    32,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_4x16_i16_neon, NeonDct2d, 64, 4, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x16_i16_neon_rdm,
    idct_dequant_4x16_i16_neon_rdm_impl,
    64,
    4,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_16x4_i16_neon, NeonDct2d, 64, 16, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x4_i16_neon_rdm,
    idct_dequant_16x4_i16_neon_rdm_impl,
    64,
    16,
    4
);
idct_rect_i16_neon_fn!(idct_dequant_8x32_i16_neon, NeonDct2d, 256, 8, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x32_i16_neon_rdm,
    idct_dequant_8x32_i16_neon_rdm_impl,
    256,
    8,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x8_i16_neon, NeonDct2d, 256, 32, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x8_i16_neon_rdm,
    idct_dequant_32x8_i16_neon_rdm_impl,
    256,
    32,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_4x32_i16_neon, NeonDct2d, 128, 4, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x32_i16_neon_rdm,
    idct_dequant_4x32_i16_neon_rdm_impl,
    128,
    4,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x4_i16_neon, NeonDct2d, 128, 32, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x4_i16_neon_rdm,
    idct_dequant_32x4_i16_neon_rdm_impl,
    128,
    32,
    4
);
iadst_rect_i16_neon_fn!(iadst_dequant_4x8_i16_neon, NeonDct2d, 32, 4, 8);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_4x8_i16_neon_rdm,
    iadst_dequant_4x8_i16_neon_rdm_impl,
    32,
    4,
    8
);
iadst_rect_i16_neon_fn!(iadst_dequant_8x4_i16_neon, NeonDct2d, 32, 8, 4);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_8x4_i16_neon_rdm,
    iadst_dequant_8x4_i16_neon_rdm_impl,
    32,
    8,
    4
);
iadst_rect_i16_neon_fn!(iadst_dequant_8x16_i16_neon, NeonDct2d, 128, 8, 16);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_8x16_i16_neon_rdm,
    iadst_dequant_8x16_i16_neon_rdm_impl,
    128,
    8,
    16
);
iadst_rect_i16_neon_fn!(iadst_dequant_16x8_i16_neon, NeonDct2d, 128, 16, 8);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_16x8_i16_neon_rdm,
    iadst_dequant_16x8_i16_neon_rdm_impl,
    128,
    16,
    8
);
iadst_rect_i16_neon_fn!(iadst_dequant_4x16_i16_neon, NeonDct2d, 64, 4, 16);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_4x16_i16_neon_rdm,
    iadst_dequant_4x16_i16_neon_rdm_impl,
    64,
    4,
    16
);
iadst_rect_i16_neon_fn!(iadst_dequant_16x4_i16_neon, NeonDct2d, 64, 16, 4);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_16x4_i16_neon_rdm,
    iadst_dequant_16x4_i16_neon_rdm_impl,
    64,
    16,
    4
);
