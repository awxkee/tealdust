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
use crate::itx_2d::{
    Adst2dBackend, Dct2dBackend, DctSimd4, ITX_TMP_PIXELS, idct_dequant_simd4_core,
    itx_dequant_simd4_core,
};
use std::arch::aarch64::*;

// Concrete 32x32 DCT kernels. These do not route through DctSimd4/DctWide.
#[target_feature(enable = "neon")]
unsafe fn neon_dct32_i32x4_hardcoded(s: &[int32x4_t; 32]) -> [int32x4_t; 32] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut b = [z; 16];
        let mut d = [z; 8];
        let mut f = [z; 4];
        let mut out = [z; 32];
        let mut m = 0usize;
        while m < 16 {
            let mut acc = z;
            let mut j = 1usize;
            while j < 32 {
                acc = vmlaq_n_s32(acc, s[j], crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
                j += 2;
            }
            b[m] = acc;
            m += 1;
        }
        m = 0;
        while m < 8 {
            let mut acc = z;
            let mut j = 2usize;
            while j < 32 {
                acc = vmlaq_n_s32(acc, s[j], crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
                j += 4;
            }
            d[m] = acc;
            m += 1;
        }
        m = 0;
        while m < 4 {
            let mut acc = vmulq_n_s32(s[4], crate::itx_2d::DCT32_DENSE_KERNEL[4 * 32 + m]);
            acc = vmlaq_n_s32(acc, s[12], crate::itx_2d::DCT32_DENSE_KERNEL[12 * 32 + m]);
            acc = vmlaq_n_s32(acc, s[20], crate::itx_2d::DCT32_DENSE_KERNEL[20 * 32 + m]);
            acc = vmlaq_n_s32(acc, s[28], crate::itx_2d::DCT32_DENSE_KERNEL[28 * 32 + m]);
            f[m] = acc;
            m += 1;
        }
        let h0 = vmlaq_n_s32(
            vmulq_n_s32(s[8], crate::itx_2d::DCT32_DENSE_KERNEL[8 * 32]),
            s[24],
            crate::itx_2d::DCT32_DENSE_KERNEL[24 * 32],
        );
        let h1 = vmlaq_n_s32(
            vmulq_n_s32(s[8], crate::itx_2d::DCT32_DENSE_KERNEL[8 * 32 + 1]),
            s[24],
            crate::itx_2d::DCT32_DENSE_KERNEL[24 * 32 + 1],
        );
        let g0 = vmlaq_n_s32(
            vmulq_n_s32(s[0], crate::itx_2d::DCT32_DENSE_KERNEL[0]),
            s[16],
            crate::itx_2d::DCT32_DENSE_KERNEL[16 * 32],
        );
        let g1 = vmlaq_n_s32(
            vmulq_n_s32(s[0], crate::itx_2d::DCT32_DENSE_KERNEL[1]),
            s[16],
            crate::itx_2d::DCT32_DENSE_KERNEL[16 * 32 + 1],
        );
        let e = [
            vaddq_s32(g0, h0),
            vaddq_s32(g1, h1),
            vsubq_s32(g1, h1),
            vsubq_s32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 8 {
            cc[i] = if i < 4 {
                vaddq_s32(e[i], f[i])
            } else {
                vsubq_s32(e[7 - i], f[7 - i])
            };
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 16 {
            a[i] = if i < 8 {
                vaddq_s32(cc[i], d[i])
            } else {
                vsubq_s32(cc[15 - i], d[15 - i])
            };
            i += 1;
        }
        let mut kk = 0usize;
        while kk < 16 {
            out[kk] = vaddq_s32(a[kk], b[kk]);
            out[kk + 16] = vsubq_s32(a[15 - kk], b[15 - kk]);
            kk += 1;
        }
        out
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_dct32_i16x8_hardcoded(s: &[int16x8_t; 32]) -> [(int32x4_t, int32x4_t); 32] {
    unsafe {
        macro_rules! coeff8 {
            ($table:ident, $idx:expr) => {{ vld1q_s16(crate::itx_2d::$table.as_ptr().add($idx)) }};
        }
        macro_rules! maddp {
            ($acc:expr, $x0:expr, $x1:expr, $c:expr, $l0:expr, $l1:expr) => {{
                let lo0 = vmlal_laneq_s16::<$l0>(($acc).0, vget_low_s16($x0), $c);
                let lo1 = vmlal_laneq_s16::<$l1>(lo0, vget_low_s16($x1), $c);
                let hi0 = vmlal_high_laneq_s16::<$l0>(($acc).1, $x0, $c);
                let hi1 = vmlal_high_laneq_s16::<$l1>(hi0, $x1, $c);
                (lo1, hi1)
            }};
        }
        let z = (vdupq_n_s32(0), vdupq_n_s32(0));
        let mut b = [z; 16];
        let mut d = [z; 8];
        let mut f = [z; 4];
        let mut out = [z; 32];
        let mut m = 0usize;
        while m < 16 {
            let mut acc = z;
            let mut grp = 0usize;
            while grp < 2 {
                let c = coeff8!(DCT32_KBW, m * 16 + grp * 8);
                let k0 = grp * 8;
                acc = maddp!(acc, s[2 * k0 + 1], s[2 * (k0 + 1) + 1], c, 0, 1);
                acc = maddp!(acc, s[2 * (k0 + 2) + 1], s[2 * (k0 + 3) + 1], c, 2, 3);
                acc = maddp!(acc, s[2 * (k0 + 4) + 1], s[2 * (k0 + 5) + 1], c, 4, 5);
                acc = maddp!(acc, s[2 * (k0 + 6) + 1], s[2 * (k0 + 7) + 1], c, 6, 7);
                grp += 1;
            }
            b[m] = acc;
            m += 1;
        }
        m = 0;
        while m < 8 {
            let c = coeff8!(DCT32_KDW, m * 8);
            let mut acc = z;
            acc = maddp!(acc, s[2], s[6], c, 0, 1);
            acc = maddp!(acc, s[10], s[14], c, 2, 3);
            acc = maddp!(acc, s[18], s[22], c, 4, 5);
            acc = maddp!(acc, s[26], s[30], c, 6, 7);
            d[m] = acc;
            m += 1;
        }
        m = 0;
        while m < 4 {
            let c = coeff8!(DCT32_KFW, m * 8);
            let mut acc = z;
            acc = maddp!(acc, s[4], s[12], c, 0, 1);
            acc = maddp!(acc, s[20], s[28], c, 2, 3);
            f[m] = acc;
            m += 1;
        }
        let ch = coeff8!(DCT32_KHW, 0);
        let h0 = maddp!(z, s[8], s[24], ch, 0, 1);
        let h1 = maddp!(z, s[8], s[24], ch, 2, 3);
        let cg = coeff8!(DCT32_KGW, 0);
        let g0 = maddp!(z, s[0], s[16], cg, 0, 1);
        let g1 = maddp!(z, s[0], s[16], cg, 2, 3);
        let e = [
            (vaddq_s32(g0.0, h0.0), vaddq_s32(g0.1, h0.1)),
            (vaddq_s32(g1.0, h1.0), vaddq_s32(g1.1, h1.1)),
            (vsubq_s32(g1.0, h1.0), vsubq_s32(g1.1, h1.1)),
            (vsubq_s32(g0.0, h0.0), vsubq_s32(g0.1, h0.1)),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 8 {
            cc[i] = if i < 4 {
                (vaddq_s32(e[i].0, f[i].0), vaddq_s32(e[i].1, f[i].1))
            } else {
                (
                    vsubq_s32(e[7 - i].0, f[7 - i].0),
                    vsubq_s32(e[7 - i].1, f[7 - i].1),
                )
            };
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 16 {
            a[i] = if i < 8 {
                (vaddq_s32(cc[i].0, d[i].0), vaddq_s32(cc[i].1, d[i].1))
            } else {
                (
                    vsubq_s32(cc[15 - i].0, d[15 - i].0),
                    vsubq_s32(cc[15 - i].1, d[15 - i].1),
                )
            };
            i += 1;
        }
        let mut kk = 0usize;
        while kk < 16 {
            out[kk] = (vaddq_s32(a[kk].0, b[kk].0), vaddq_s32(a[kk].1, b[kk].1));
            out[kk + 16] = (
                vsubq_s32(a[15 - kk].0, b[15 - kk].0),
                vsubq_s32(a[15 - kk].1, b[15 - kk].1),
            );
            kk += 1;
        }
        out
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_dct16_i32x4_hardcoded(s: &[int32x4_t; 16]) -> [int32x4_t; 16] {
    unsafe {
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
}

#[target_feature(enable = "neon")]
unsafe fn neon_adst16_i32x4_hardcoded(s: &[int32x4_t; 16], flip: bool) -> [int32x4_t; 16] {
    unsafe {
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
}

#[target_feature(enable = "neon")]
unsafe fn neon_tx16_i32x4_hardcoded(s: &[int32x4_t; 16], kind: usize) -> [int32x4_t; 16] {
    unsafe {
        match kind {
            crate::itx_2d::TX_KIND_DCT => neon_dct16_i32x4_hardcoded(s),
            crate::itx_2d::TX_KIND_ADST => neon_adst16_i32x4_hardcoded(s, false),
            crate::itx_2d::TX_KIND_FLIPADST => neon_adst16_i32x4_hardcoded(s, true),
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_tx16_i16x8_hardcoded(
    s: &[int16x8_t; 16],
    kind: usize,
) -> [(int32x4_t, int32x4_t); 16] {
    unsafe {
        let z = (vdupq_n_s32(0), vdupq_n_s32(0));
        let mut out = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let mut acc = z;
            let mut j = 0usize;
            while j < 16 {
                let k0 = match kind {
                    crate::itx_2d::TX_KIND_DCT => {
                        crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m] as i16
                    }
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
                    crate::itx_2d::TX_KIND_ADST => {
                        crate::itx_1d::ADST16_KERNEL_ROWS[m][j + 1] as i16
                    }
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
}

#[target_feature(enable = "neon")]
unsafe fn iadst_dequant_16x16_neon_i32_hardcoded(
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
            let out = neon_tx16_i32x4_hardcoded(&s, first_kind);
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
            let out = neon_tx16_i32x4_hardcoded(&s, second_kind);
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
unsafe fn iadst_dequant_16x16_neon_i16_hardcoded(
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
                s[j] = neon_load8_i16_hardcoded(coeff, y + j * 16, is_rect2);
                j += 1;
            }
            let out = neon_tx16_i16x8_hardcoded(&s, first_kind);
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
                let v = neon_load4_i16_hardcoded(coeff, y + j * 16, is_rect2);
                s[j] = vmovl_s16(vget_low_s16(v));
                j += 1;
            }
            let out = neon_tx16_i32x4_hardcoded(&s, first_kind);
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
                s[j] = neon_load8_narrow_i32_hardcoded(tmp, x + j * 32);
                j += 1;
            }
            let out = neon_tx16_i16x8_hardcoded(&s, second_kind);
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
unsafe fn neon_load8_i16_hardcoded(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
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
unsafe fn neon_load4_i16_hardcoded(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
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
unsafe fn neon_rdm_load8_i16_hardcoded(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
    unsafe {
        let x = vld1q_s16(src.as_ptr().add(off));
        if rect2 {
            vqrdmulhq_s16(x, vdupq_n_s16(0x5a80))
        } else {
            x
        }
    }
}
#[target_feature(enable = "rdm")]
unsafe fn neon_rdm_load4_i16_hardcoded(src: &[i16], off: usize, rect2: bool) -> int16x8_t {
    unsafe {
        let x = vcombine_s16(vld1_s16(src.as_ptr().add(off)), vdup_n_s16(0));
        if rect2 {
            vqrdmulhq_s16(x, vdupq_n_s16(0x5a80))
        } else {
            x
        }
    }
}
#[target_feature(enable = "neon")]
unsafe fn neon_load8_narrow_i32_hardcoded(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        let lo = vld1q_s32(src.as_ptr().add(off));
        let hi = vld1q_s32(src.as_ptr().add(off + 4));
        vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_store4x4_i32_clip(
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
unsafe fn neon_store4x4_wide_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    acc: &[(int32x4_t, int32x4_t); 4],
    high: bool,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        let v = [
            if high { acc[0].1 } else { acc[0].0 },
            if high { acc[1].1 } else { acc[1].0 },
            if high { acc[2].1 } else { acc[2].0 },
            if high { acc[3].1 } else { acc[3].0 },
        ];
        neon_store4x4_i32_clip(tmp, off, &v, rnd, nsh, minv, maxv);
    }
}
#[target_feature(enable = "neon")]
unsafe fn neon_store8x8_wide_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    acc: &[(int32x4_t, int32x4_t); 8],
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        let g0 = [acc[0], acc[1], acc[2], acc[3]];
        let g1 = [acc[4], acc[5], acc[6], acc[7]];
        neon_store4x4_wide_clip(tmp, off, &g0, false, rnd, nsh, minv, maxv);
        neon_store4x4_wide_clip(tmp, off + 4 * 32, &g0, true, rnd, nsh, minv, maxv);
        neon_store4x4_wide_clip(tmp, off + 4, &g1, false, rnd, nsh, minv, maxv);
        neon_store4x4_wide_clip(tmp, off + 4 * 32 + 4, &g1, true, rnd, nsh, minv, maxv);
    }
}

#[target_feature(enable = "neon")]
unsafe fn idct_dequant_16x16_neon_i32_hardcoded(
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
unsafe fn idct_dequant_16x16_neon_i16_hardcoded(
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
                    let v16 = neon_load4_i16_hardcoded(coeff, $base + j * 16, is_rect2);
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
unsafe fn idct_dequant_32x32_neon_i32_hardcoded(
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
            let mut s = [vdupq_n_s32(0); 32];
            let mut j = 0usize;
            while j < 32 {
                let mut v = vld1q_s32(coeff.as_ptr().add(y + j * 32));
                if is_rect2 {
                    v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                }
                s[j] = v;
                j += 1;
            }
            let out = neon_dct32_i32x4_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
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
            let mut s = [vdupq_n_s32(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = vld1q_s32(tmp.as_ptr().add(x + j * 32));
                j += 1;
            }
            let out = neon_dct32_i32x4_hardcoded(&s);
            j = 0;
            while j < 32 {
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32), out[j]);
                j += 1;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn idct_dequant_32x32_neon_i16_hardcoded(
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
        while y + 8 <= ncols {
            let mut s = [vdupq_n_s16(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = neon_load8_i16_hardcoded(coeff, y + j * 32, is_rect2);
                j += 1;
            }
            let out = neon_dct32_i16x8_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [
                    out[x],
                    out[x + 1],
                    out[x + 2],
                    out[x + 3],
                    out[x + 4],
                    out[x + 5],
                    out[x + 6],
                    out[x + 7],
                ];
                neon_store8x8_wide_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
                x += 8;
            }
            y += 8;
        }
        if y + 4 <= ncols {
            let mut s = [vdupq_n_s16(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = neon_load4_i16_hardcoded(coeff, y + j * 32, is_rect2);
                j += 1;
            }
            let out = neon_dct32_i16x8_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                neon_store4x4_wide_clip(tmp, y * 32 + x, &g, false, rnd, nsh, minv, maxv);
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
            let mut s = [vdupq_n_s16(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = neon_load8_narrow_i32_hardcoded(tmp, x + j * 32);
                j += 1;
            }
            let out = neon_dct32_i16x8_hardcoded(&s);
            j = 0;
            while j < 32 {
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32), out[j].0);
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32 + 4), out[j].1);
                j += 1;
            }
            x += 8;
        }
    }
}

#[target_feature(enable = "rdm")]
unsafe fn idct_dequant_32x32_neon_i16_rdm_hardcoded(
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
        while y + 8 <= ncols {
            let mut s = [vdupq_n_s16(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = neon_rdm_load8_i16_hardcoded(coeff, y + j * 32, is_rect2);
                j += 1;
            }
            let out = neon_dct32_i16x8_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [
                    out[x],
                    out[x + 1],
                    out[x + 2],
                    out[x + 3],
                    out[x + 4],
                    out[x + 5],
                    out[x + 6],
                    out[x + 7],
                ];
                neon_store8x8_wide_clip(tmp, y * 32 + x, &g, rnd, nsh, minv, maxv);
                x += 8;
            }
            y += 8;
        }
        if y + 4 <= ncols {
            let mut s = [vdupq_n_s16(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = neon_rdm_load4_i16_hardcoded(coeff, y + j * 32, is_rect2);
                j += 1;
            }
            let out = neon_dct32_i16x8_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                neon_store4x4_wide_clip(tmp, y * 32 + x, &g, false, rnd, nsh, minv, maxv);
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
            let mut s = [vdupq_n_s16(0); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = neon_load8_narrow_i32_hardcoded(tmp, x + j * 32);
                j += 1;
            }
            let out = neon_dct32_i16x8_hardcoded(&s);
            j = 0;
            while j < 32 {
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32), out[j].0);
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32 + 4), out[j].1);
                j += 1;
            }
            x += 8;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct NeonI32x4(int32x4_t);

impl crate::itx_1d::DctLane for NeonI32x4 {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn zero() -> Self {
        NeonI32x4(vdupq_n_s32(0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add(self, o: Self) -> Self {
        NeonI32x4(vaddq_s32(self.0, o.0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn sub(self, o: Self) -> Self {
        NeonI32x4(vsubq_s32(self.0, o.0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mul(self, k: Self) -> Self {
        NeonI32x4(vmulq_s32(self.0, k.0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dup_load(table: &[i32], idx: usize) -> Self {
        // SAFETY: callers index within the kernel tables.
        NeonI32x4(unsafe { vld1q_dup_s32(table.as_ptr().add(idx)) })
    }
    type Coeffs = int32x4_t;
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load_coeffs(table: &[i32], idx: usize) -> int32x4_t {
        // SAFETY: callers index a 4-wide group within the kernel tables.
        unsafe { vld1q_s32(table.as_ptr().add(idx)) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn mul_add_lane<const LANE: i32>(self, x: Self, c: int32x4_t) -> Self {
        NeonI32x4(vmlaq_laneq_s32::<LANE>(self.0, x.0, c))
    }

    #[inline]
    #[target_feature(enable = "neon")]
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
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn zero() -> Self::Acc {
        (vdupq_n_s32(0), vdupq_n_s32(0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        (vaddq_s32(a.0, b.0), vaddq_s32(a.1, b.1))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        (vsubq_s32(a.0, b.0), vsubq_s32(a.1, b.1))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load_coeffs(table: &[i16], idx: usize) -> int16x8_t {
        unsafe { vld1q_s16(table.as_ptr().add(idx)) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
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
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            let hi = vld1q_s32(src.as_ptr().add(off + 4));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
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
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline(always)]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            let lo = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), lo, 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load8_i16(src: &[i16], off: usize) -> int16x8_t {
        debug_assert!(off + 8 <= src.len());
        unsafe { vld1q_s16(src.as_ptr().add(off)) }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> int16x8_t {
        unsafe {
            let x = Self::load8_i16(src, off);
            let r = vdupq_n_s32(128);
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(r, vget_low_s16(x), 181));
            let hi = vshrq_n_s32::<8>(vmlal_high_n_s16(r, x, 181));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[inline(always)]
    unsafe fn load4_i16(src: &[i16], off: usize) -> int16x8_t {
        debug_assert!(off + 4 <= src.len());
        unsafe { vcombine_s16(vld1_s16(src.as_ptr().add(off)), vdup_n_s16(0)) }
    }
    #[inline(always)]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> int16x8_t {
        unsafe {
            let x = Self::load4_i16(src, off);
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), vget_low_s16(x), 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        (
            vdupq_n_s32(rnd),
            vdupq_n_s32(-shift),
            vdupq_n_s32(min),
            vdupq_n_s32(max),
        )
    }
    #[inline]
    #[target_feature(enable = "neon")]
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
    #[inline]
    #[target_feature(enable = "neon")]
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
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store4x4_strided_clip<const HIGH: bool>(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 4],
        clip: Self::Clip,
    ) {
        unsafe {
            #[inline]
            #[target_feature(enable = "neon")]
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

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            vst1q_s32(dst.as_mut_ptr().add(off), acc.0);
            vst1q_s32(dst.as_mut_ptr().add(off + 4), acc.1);
        }
    }
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            vst1q_s32(dst.as_mut_ptr().add(off), acc.0);
        }
    }
}

#[target_feature(enable = "rdm")]
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
unsafe fn load4_rect2_narrow_rdm(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        // Same RDM rect2 normalization for 4 active lanes; high lanes stay zero.
        vqrdmulhq_s16(NeonWide::load4_narrow(src, off), vdupq_n_s16(0x5a80))
    }
}

#[target_feature(enable = "rdm")]
unsafe fn load8_rect2_i16_rdm(src: &[i16], off: usize) -> int16x8_t {
    unsafe { vqrdmulhq_s16(NeonWide::load8_i16(src, off), vdupq_n_s16(0x5a80)) }
}

#[target_feature(enable = "rdm")]
unsafe fn load4_rect2_i16_rdm(src: &[i16], off: usize) -> int16x8_t {
    unsafe { vqrdmulhq_s16(NeonWide::load4_i16(src, off), vdupq_n_s16(0x5a80)) }
}

pub(crate) struct NeonWideRdm;

impl crate::itx_1d::DctWide for NeonWideRdm {
    type In = int16x8_t;
    type Acc = (int32x4_t, int32x4_t);
    type Coeffs = int16x8_t;
    type Clip = (int32x4_t, int32x4_t, int32x4_t, int32x4_t);

    #[inline(always)]
    unsafe fn zero() -> Self::Acc {
        unsafe { NeonWide::zero() }
    }

    #[inline(always)]
    unsafe fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { NeonWide::add(a, b) }
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { NeonWide::sub(a, b) }
    }

    #[inline(always)]
    unsafe fn load_coeffs(table: &[i16], idx: usize) -> Self::Coeffs {
        unsafe { NeonWide::load_coeffs(table, idx) }
    }

    #[inline(always)]
    unsafe fn mul_add_lane<const LANE: i32>(
        acc: Self::Acc,
        x: Self::In,
        c: Self::Coeffs,
    ) -> Self::Acc {
        unsafe { NeonWide::mul_add_lane::<LANE>(acc, x, c) }
    }

    #[inline(always)]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { NeonWide::load8_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { load8_rect2_narrow_rdm(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { NeonWide::load4_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { load4_rect2_narrow_rdm(src, off) }
    }
    #[inline(always)]
    unsafe fn load8_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { NeonWide::load8_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { load8_rect2_i16_rdm(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { NeonWide::load4_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { load4_rect2_i16_rdm(src, off) }
    }

    #[inline(always)]
    unsafe fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        unsafe { NeonWide::make_clip(rnd, shift, min, max) }
    }

    #[inline(always)]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store8_strided_clip(dst, off, stride, acc, clip) }
    }

    #[inline(always)]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store4_strided_clip(dst, off, stride, acc, clip) }
    }

    #[inline(always)]
    unsafe fn store4x4_strided_clip<const HIGH: bool>(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 4],
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store4x4_strided_clip::<HIGH>(dst, off, stride, acc, clip) }
    }

    #[inline(always)]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { NeonWide::store8(dst, off, acc) }
    }

    #[inline(always)]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { NeonWide::store4(dst, off, acc) }
    }
}

pub(crate) struct NeonDct2d;

impl DctSimd4 for NeonDct2d {
    type V = NeonI32x4;
    type Wide = NeonWide;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        NeonI32x4(unsafe { vdupq_n_s32(0) })
    }

    #[inline(always)]
    unsafe fn splat(v: i32) -> Self::V {
        NeonI32x4(unsafe { vdupq_n_s32(v) })
    }

    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vaddq_s32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vsubq_s32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vmulq_s32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe {
            let scaled = vmlaq_n_s32(vdupq_n_s32(128), a.0, 181);
            NeonI32x4(vshrq_n_s32::<8>(scaled))
        }
    }

    #[inline(always)]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_ptr().add(off) };
        NeonI32x4(unsafe { vld1q_s32(p) })
    }

    #[inline(always)]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_mut_ptr().add(off) };
        unsafe { vst1q_s32(p, v.0) };
    }

    #[inline(always)]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) };
        NeonI32x4(unsafe { vld1q_s32(p) })
    }

    #[inline(always)]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) };
        NeonI32x4(unsafe { vmovl_s16(vld1_s16(p)) })
    }

    #[inline(always)]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), v.0) };
        out
    }

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
        unsafe {
            #[inline(always)]
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

    #[inline(always)]
    unsafe fn zero() -> Self::V {
        unsafe { NeonDct2d::zero() }
    }

    #[inline(always)]
    unsafe fn splat(v: i32) -> Self::V {
        unsafe { NeonDct2d::splat(v) }
    }

    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::add(a, b) }
    }

    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::sub(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::mul(a, b) }
    }

    #[inline(always)]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe { NeonDct2d::rect2_scale(a) }
    }

    #[inline(always)]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        unsafe { NeonDct2d::load(tmp, off) }
    }

    #[inline(always)]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        unsafe { NeonDct2d::store(tmp, off, v) }
    }

    #[inline(always)]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        unsafe { NeonDct2d::load_slice(src, off) }
    }

    #[inline(always)]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        unsafe { NeonDct2d::load_slice_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        unsafe { NeonDct2d::to_array(v) }
    }

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
        unsafe { NeonDct2d::store4x4_clip(tmp, off, stride, v, rnd, shift, min, max) }
    }
}

impl Dct2dBackend for NeonDct2d {
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
        idct_dequant_simd4_core::<Self, 16, 4, i32>(
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
        idct_dequant_simd4_core::<Self, 64, 8, i32>(
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
        idct_dequant_simd4_core::<Self, 256, 16, i32>(
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
        idct_dequant_simd4_core::<Self, 1024, 32, i32>(
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
        idct_dequant_simd4_core::<Self, 1024, 32, i32>(
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

impl Adst2dBackend for NeonDct2d {
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
        itx_dequant_simd4_core::<Self, 16, 4, i32>(
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
        itx_dequant_simd4_core::<Self, 64, 8, i32>(
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
        itx_dequant_simd4_core::<Self, 256, 16, i32>(
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
            crate::itx_idct_dequant_simd4_body!(
                $backend,
                { $n },
                { $s },
                i32,
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
            crate::itx_kind_dequant_simd4_body!(
                $backend,
                { $n },
                { $s },
                i32,
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
            crate::itx_idct_dequant_rect_simd4_body!(
                $backend,
                { $n },
                { $w },
                { $h },
                i32,
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
            crate::itx_kind_dequant_rect_simd4_body!(
                $backend,
                { $n },
                { $w },
                { $h },
                i32,
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
    };
}

idct_neon_fn!(idct_dequant_4x4_neon, NeonDct2d, 16, 4);
idct_neon_fn!(idct_dequant_8x8_neon, NeonDct2d, 64, 8);
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
        idct_dequant_16x16_neon_i32_hardcoded(
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
        idct_dequant_32x32_neon_i32_hardcoded(
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
iadst_neon_fn!(iadst_dequant_8x8_neon, NeonDct2d, 64, 8);
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
        iadst_dequant_16x16_neon_i32_hardcoded(
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

        #[inline]
        #[target_feature(enable = "rdm")]
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
            crate::itx_idct_dequant_rect_simd4_body!(
                NeonDct2dRdm,
                { $n },
                { $w },
                { $h },
                i32,
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

        #[inline]
        #[target_feature(enable = "rdm")]
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
            crate::itx_kind_dequant_rect_simd4_body!(
                NeonDct2dRdm,
                { $n },
                { $w },
                { $h },
                i32,
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

#[inline]
#[target_feature(enable = "rdm")]
unsafe fn idct_dequant_32x32_neon_rdm_impl(
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
        idct_dequant_32x32_neon_i32_hardcoded(
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
            crate::itx_idct_dequant_simd4_body!(
                $backend,
                { $n },
                { $s },
                i16,
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
            crate::itx_kind_dequant_simd4_body!(
                $backend,
                { $n },
                { $s },
                i16,
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
            crate::itx_idct_dequant_rect_simd4_body!(
                $backend,
                { $n },
                { $w },
                { $h },
                i16,
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
            crate::itx_kind_dequant_rect_simd4_body!(
                $backend,
                { $n },
                { $w },
                { $h },
                i16,
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

        #[inline]
        #[target_feature(enable = "rdm")]
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
            crate::itx_idct_dequant_simd4_body!(
                NeonDct2dRdm,
                { $n },
                { $s },
                i16,
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

        #[inline]
        #[target_feature(enable = "rdm")]
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
            crate::itx_idct_dequant_rect_simd4_body!(
                NeonDct2dRdm,
                { $n },
                { $w },
                { $h },
                i16,
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

        #[inline]
        #[target_feature(enable = "rdm")]
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
            crate::itx_kind_dequant_rect_simd4_body!(
                NeonDct2dRdm,
                { $n },
                { $w },
                { $h },
                i16,
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
    };
}

idct_i16_neon_fn!(idct_dequant_4x4_i16_neon, NeonDct2d, 16, 4);
idct_i16_neon_fn!(idct_dequant_8x8_i16_neon, NeonDct2d, 64, 8);
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
        idct_dequant_16x16_neon_i16_hardcoded(
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
        idct_dequant_32x32_neon_i16_hardcoded(
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
        idct_dequant_32x32_neon_i16_rdm_hardcoded(
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
iadst_i16_neon_fn!(iadst_dequant_8x8_i16_neon, NeonDct2d, 64, 8);
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
        iadst_dequant_16x16_neon_i16_hardcoded(
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
