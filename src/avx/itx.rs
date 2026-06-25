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
use std::arch::x86_64::*;

use crate::itx_2d::{DctSimd4, ITX_TMP_PIXELS};

// Concrete 32x32 DCT kernels.  These are intentionally backend-local and do not
// pass through DctSimd4/DctWide or any generic 1-D transform wrapper.
#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_dct32_i32x4_hardcoded(s: &[__m128i; 32]) -> [__m128i; 32] {
    unsafe {
        let z = _mm_setzero_si128();
        let mut b = [z; 16];
        let mut d = [z; 8];
        let mut f = [z; 4];
        let mut out = [z; 32];

        let mut m = 0usize;
        while m < 16 {
            let mut acc = z;
            let mut j = 1usize;
            while j < 32 {
                let k = _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s[j], k));
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
                let k = _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s[j], k));
                j += 4;
            }
            d[m] = acc;
            m += 1;
        }

        m = 0;
        while m < 4 {
            let mut acc = _mm_mullo_epi32(
                s[4],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[4 * 32 + m]),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_mullo_epi32(
                    s[12],
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[12 * 32 + m]),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_mullo_epi32(
                    s[20],
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[20 * 32 + m]),
                ),
            );
            acc = _mm_add_epi32(
                acc,
                _mm_mullo_epi32(
                    s[28],
                    _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[28 * 32 + m]),
                ),
            );
            f[m] = acc;
            m += 1;
        }

        let h0 = _mm_add_epi32(
            _mm_mullo_epi32(
                s[8],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[8 * 32]),
            ),
            _mm_mullo_epi32(
                s[24],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[24 * 32]),
            ),
        );
        let h1 = _mm_add_epi32(
            _mm_mullo_epi32(
                s[8],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[8 * 32 + 1]),
            ),
            _mm_mullo_epi32(
                s[24],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[24 * 32 + 1]),
            ),
        );
        let g0 = _mm_add_epi32(
            _mm_mullo_epi32(s[0], _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[0])),
            _mm_mullo_epi32(
                s[16],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[16 * 32]),
            ),
        );
        let g1 = _mm_add_epi32(
            _mm_mullo_epi32(s[0], _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[1])),
            _mm_mullo_epi32(
                s[16],
                _mm_set1_epi32(crate::itx_2d::DCT32_DENSE_KERNEL[16 * 32 + 1]),
            ),
        );
        let e = [
            _mm_add_epi32(g0, h0),
            _mm_add_epi32(g1, h1),
            _mm_sub_epi32(g1, h1),
            _mm_sub_epi32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 8 {
            cc[i] = if i < 4 {
                _mm_add_epi32(e[i], f[i])
            } else {
                _mm_sub_epi32(e[7 - i], f[7 - i])
            };
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 16 {
            a[i] = if i < 8 {
                _mm_add_epi32(cc[i], d[i])
            } else {
                _mm_sub_epi32(cc[15 - i], d[15 - i])
            };
            i += 1;
        }
        let mut kk = 0usize;
        while kk < 16 {
            out[kk] = _mm_add_epi32(a[kk], b[kk]);
            out[kk + 16] = _mm_sub_epi32(a[15 - kk], b[15 - kk]);
            kk += 1;
        }
        out
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_dct32_i16x8_hardcoded(s: &[__m256i; 32]) -> [__m256i; 32] {
    unsafe {
        macro_rules! coeff8 {
            ($table:ident, $idx:expr) => {{
                let c128 =
                    _mm_loadu_si128(crate::itx_2d::$table.as_ptr().add($idx) as *const __m128i);
                _mm256_broadcastsi128_si256(c128)
            }};
        }
        macro_rules! maddp {
            ($acc:expr, $x0:expr, $x1:expr, $c:expr, $imm:expr) => {{
                let k01 = _mm256_shuffle_epi32::<$imm>($c);
                let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16($x0, $x1), k01);
                let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16($x0, $x1), k01);
                let sum8 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                _mm256_add_epi32($acc, sum8)
            }};
        }
        let z = _mm256_setzero_si256();
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
                acc = maddp!(acc, s[2 * k0 + 1], s[2 * (k0 + 1) + 1], c, 0x00);
                acc = maddp!(acc, s[2 * (k0 + 2) + 1], s[2 * (k0 + 3) + 1], c, 0x55);
                acc = maddp!(acc, s[2 * (k0 + 4) + 1], s[2 * (k0 + 5) + 1], c, 0xaa);
                acc = maddp!(acc, s[2 * (k0 + 6) + 1], s[2 * (k0 + 7) + 1], c, 0xff);
                grp += 1;
            }
            b[m] = acc;
            m += 1;
        }
        m = 0;
        while m < 8 {
            let c = coeff8!(DCT32_KDW, m * 8);
            let mut acc = z;
            acc = maddp!(acc, s[2], s[6], c, 0x00);
            acc = maddp!(acc, s[10], s[14], c, 0x55);
            acc = maddp!(acc, s[18], s[22], c, 0xaa);
            acc = maddp!(acc, s[26], s[30], c, 0xff);
            d[m] = acc;
            m += 1;
        }
        m = 0;
        while m < 4 {
            let c = coeff8!(DCT32_KFW, m * 8);
            let mut acc = z;
            acc = maddp!(acc, s[4], s[12], c, 0x00);
            acc = maddp!(acc, s[20], s[28], c, 0x55);
            f[m] = acc;
            m += 1;
        }
        let ch = coeff8!(DCT32_KHW, 0);
        let h0 = maddp!(z, s[8], s[24], ch, 0x00);
        let h1 = maddp!(z, s[8], s[24], ch, 0x55);
        let cg = coeff8!(DCT32_KGW, 0);
        let g0 = maddp!(z, s[0], s[16], cg, 0x00);
        let g1 = maddp!(z, s[0], s[16], cg, 0x55);
        let e = [
            _mm256_add_epi32(g0, h0),
            _mm256_add_epi32(g1, h1),
            _mm256_sub_epi32(g1, h1),
            _mm256_sub_epi32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 8 {
            cc[i] = if i < 4 {
                _mm256_add_epi32(e[i], f[i])
            } else {
                _mm256_sub_epi32(e[7 - i], f[7 - i])
            };
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 16 {
            a[i] = if i < 8 {
                _mm256_add_epi32(cc[i], d[i])
            } else {
                _mm256_sub_epi32(cc[15 - i], d[15 - i])
            };
            i += 1;
        }
        let mut kk = 0usize;
        while kk < 16 {
            out[kk] = _mm256_add_epi32(a[kk], b[kk]);
            out[kk + 16] = _mm256_sub_epi32(a[15 - kk], b[15 - kk]);
            kk += 1;
        }
        out
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_dct16_i32x4_hardcoded(s: &[__m128i; 16]) -> [__m128i; 16] {
    unsafe {
        let z = _mm_setzero_si128();
        let mut out = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let mut acc = z;
            let mut j = 0usize;
            while j < 16 {
                let k = _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m]);
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s[j], k));
                j += 1;
            }
            out[m] = acc;
            m += 1;
        }
        out
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_adst16_i32x4_hardcoded(s: &[__m128i; 16], flip: bool) -> [__m128i; 16] {
    unsafe {
        let rows = if flip {
            &crate::itx_1d::FLIPADST16_KERNEL_ROWS
        } else {
            &crate::itx_1d::ADST16_KERNEL_ROWS
        };
        let z = _mm_setzero_si128();
        let mut out = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let row = &rows[m];
            let mut acc = z;
            let mut j = 0usize;
            while j < 16 {
                let k = _mm_set1_epi32(row[j] as i32);
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(s[j], k));
                j += 1;
            }
            out[m] = acc;
            m += 1;
        }
        out
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_tx16_i32x4_hardcoded(s: &[__m128i; 16], kind: usize) -> [__m128i; 16] {
    unsafe {
        match kind {
            crate::itx_2d::TX_KIND_DCT => avx2_dct16_i32x4_hardcoded(s),
            crate::itx_2d::TX_KIND_ADST => avx2_adst16_i32x4_hardcoded(s, false),
            crate::itx_2d::TX_KIND_FLIPADST => avx2_adst16_i32x4_hardcoded(s, true),
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_tx16_i16x8_hardcoded(s: &[__m256i; 16], kind: usize) -> [__m256i; 16] {
    unsafe {
        let z = _mm256_setzero_si256();
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
                let pair = ((k1 as u16 as i32) << 16) | (k0 as u16 as i32);
                let c = _mm256_set1_epi32(pair);
                let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(s[j], s[j + 1]), c);
                let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(s[j], s[j + 1]), c);
                let sum8 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                acc = _mm256_add_epi32(acc, sum8);
                j += 2;
            }
            out[m] = acc;
            m += 1;
        }
        out
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn iadst_dequant_16x16_avx2_i32_hardcoded(
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
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut s = [_mm_setzero_si128(); 16];
            let mut j = 0usize;
            while j < 16 {
                let mut v = _mm_loadu_si128(coeff.as_ptr().add(y + j * 16) as *const __m128i);
                if is_rect2 {
                    v = _mm_srai_epi32::<8>(_mm_add_epi32(
                        _mm_mullo_epi32(v, _mm_set1_epi32(181)),
                        _mm_set1_epi32(128),
                    ));
                }
                s[j] = v;
                j += 1;
            }
            let out = avx2_tx16_i32x4_hardcoded(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
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
            let mut s = [_mm_setzero_si128(); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = _mm_loadu_si128(tmp.as_ptr().add(x + j * 32) as *const __m128i);
                j += 1;
            }
            let out = avx2_tx16_i32x4_hardcoded(&s, second_kind);
            j = 0;
            while j < 16 {
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + j * 32) as *mut __m128i, out[j]);
                j += 1;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn iadst_dequant_16x16_avx2_i16_hardcoded(
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
        let rnd8 = _mm256_set1_epi32((1 << shift0) >> 1);
        let sh8 = _mm_cvtsi32_si128(shift0);
        let min8 = _mm256_set1_epi32(row_clip_min);
        let max8 = _mm256_set1_epi32(row_clip_max);
        let rnd4 = _mm_set1_epi32((1 << shift0) >> 1);
        let min4 = _mm_set1_epi32(row_clip_min);
        let max4 = _mm_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 8 <= ncols {
            let mut s = [_mm256_setzero_si256(); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = avx2_load8_i16(coeff, y + j * 16, is_rect2);
                j += 1;
            }
            let out = avx2_tx16_i16x8_hardcoded(&s, first_kind);
            let g0 = [
                out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
            ];
            let g1 = [
                out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15],
            ];
            avx2_store8x8_clip_i32(&mut tmp[..], y * 32, 32, &g0, rnd8, sh8, min8, max8);
            avx2_store8x8_clip_i32(&mut tmp[..], y * 32 + 8, 32, &g1, rnd8, sh8, min8, max8);
            y += 8;
        }
        while y + 4 <= ncols {
            let mut s = [_mm_setzero_si128(); 16];
            let mut j = 0usize;
            while j < 16 {
                let v = avx2_load4_i16(coeff, y + j * 16, is_rect2);
                s[j] = _mm_cvtepi16_epi32(_mm256_castsi256_si128(v));
                j += 1;
            }
            let out = avx2_tx16_i32x4_hardcoded(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd4, sh8, min4, max4);
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
            let mut s = [_mm256_setzero_si256(); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = avx2_load8_narrow_i32(tmp, x + j * 32);
                j += 1;
            }
            let out = avx2_tx16_i16x8_hardcoded(&s, second_kind);
            j = 0;
            while j < 16 {
                _mm256_storeu_si256(tmp.as_mut_ptr().add(x + j * 32) as *mut __m256i, out[j]);
                j += 1;
            }
            x += 8;
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_load8_i16(src: &[i16], off: usize, rect2: bool) -> __m256i {
    unsafe {
        let x = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
        let v = _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), x);
        if rect2 {
            _mm256_mulhrs_epi16(v, _mm256_set1_epi16(0x5a80))
        } else {
            v
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_load4_i16(src: &[i16], off: usize, rect2: bool) -> __m256i {
    unsafe {
        let x = _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i);
        let v = _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), x);
        if rect2 {
            _mm256_mulhrs_epi16(v, _mm256_set1_epi16(0x5a80))
        } else {
            v
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_load8_narrow_i32(src: &[i32], off: usize) -> __m256i {
    unsafe {
        let v = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
        let p = _mm256_packs_epi32(v, _mm256_setzero_si256());
        _mm256_permute4x64_epi64::<0xd8>(p)
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_store8x8_clip_i32(
    dst: &mut [i32],
    off: usize,
    stride: usize,
    acc: &[__m256i; 8],
    rnd: __m256i,
    sh: __m128i,
    minv: __m256i,
    maxv: __m256i,
) {
    unsafe {
        macro_rules! clip {
            ($v:expr) => {{
                _mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32($v, rnd), sh), minv),
                    maxv,
                )
            }};
        }
        let c0 = clip!(acc[0]);
        let c1 = clip!(acc[1]);
        let c2 = clip!(acc[2]);
        let c3 = clip!(acc[3]);
        let c4 = clip!(acc[4]);
        let c5 = clip!(acc[5]);
        let c6 = clip!(acc[6]);
        let c7 = clip!(acc[7]);
        let t0 = _mm256_unpacklo_epi32(c0, c1);
        let t1 = _mm256_unpackhi_epi32(c0, c1);
        let t2 = _mm256_unpacklo_epi32(c2, c3);
        let t3 = _mm256_unpackhi_epi32(c2, c3);
        let t4 = _mm256_unpacklo_epi32(c4, c5);
        let t5 = _mm256_unpackhi_epi32(c4, c5);
        let t6 = _mm256_unpacklo_epi32(c6, c7);
        let t7 = _mm256_unpackhi_epi32(c6, c7);
        let a0 = _mm256_unpacklo_epi64(t0, t2);
        let a1 = _mm256_unpackhi_epi64(t0, t2);
        let a2 = _mm256_unpacklo_epi64(t1, t3);
        let a3 = _mm256_unpackhi_epi64(t1, t3);
        let a4 = _mm256_unpacklo_epi64(t4, t6);
        let a5 = _mm256_unpackhi_epi64(t4, t6);
        let a6 = _mm256_unpacklo_epi64(t5, t7);
        let a7 = _mm256_unpackhi_epi64(t5, t7);
        let r0 = _mm256_permute2x128_si256::<0x20>(a0, a4);
        let r1 = _mm256_permute2x128_si256::<0x20>(a1, a5);
        let r2 = _mm256_permute2x128_si256::<0x20>(a2, a6);
        let r3 = _mm256_permute2x128_si256::<0x20>(a3, a7);
        let r4 = _mm256_permute2x128_si256::<0x31>(a0, a4);
        let r5 = _mm256_permute2x128_si256::<0x31>(a1, a5);
        let r6 = _mm256_permute2x128_si256::<0x31>(a2, a6);
        let r7 = _mm256_permute2x128_si256::<0x31>(a3, a7);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, r0);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + stride) as *mut __m256i, r1);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + 2 * stride) as *mut __m256i, r2);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + 3 * stride) as *mut __m256i, r3);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + 4 * stride) as *mut __m256i, r4);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + 5 * stride) as *mut __m256i, r5);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + 6 * stride) as *mut __m256i, r6);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off + 7 * stride) as *mut __m256i, r7);
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_store4x4_clip_i32(
    dst: &mut [i32],
    off: usize,
    stride: usize,
    acc: &[__m256i; 4],
    rnd: __m256i,
    sh: __m128i,
    minv: __m256i,
    maxv: __m256i,
) {
    unsafe {
        macro_rules! clip_lo {
            ($v:expr) => {{
                _mm256_castsi256_si128(_mm256_min_epi32(
                    _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32($v, rnd), sh), minv),
                    maxv,
                ))
            }};
        }
        let c0 = clip_lo!(acc[0]);
        let c1 = clip_lo!(acc[1]);
        let c2 = clip_lo!(acc[2]);
        let c3 = clip_lo!(acc[3]);
        let t0 = _mm_unpacklo_epi32(c0, c1);
        let t1 = _mm_unpackhi_epi32(c0, c1);
        let t2 = _mm_unpacklo_epi32(c2, c3);
        let t3 = _mm_unpackhi_epi32(c2, c3);
        let r0 = _mm_unpacklo_epi64(t0, t2);
        let r1 = _mm_unpackhi_epi64(t0, t2);
        let r2 = _mm_unpacklo_epi64(t1, t3);
        let r3 = _mm_unpackhi_epi64(t1, t3);
        _mm_storeu_si128(dst.as_mut_ptr().add(off) as *mut __m128i, r0);
        _mm_storeu_si128(dst.as_mut_ptr().add(off + stride) as *mut __m128i, r1);
        _mm_storeu_si128(dst.as_mut_ptr().add(off + 2 * stride) as *mut __m128i, r2);
        _mm_storeu_si128(dst.as_mut_ptr().add(off + 3 * stride) as *mut __m128i, r3);
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn avx2_store4x4_i32_clip(
    dst: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    v: &[__m128i; 4],
    rnd: __m128i,
    sh: __m128i,
    minv: __m128i,
    maxv: __m128i,
) {
    unsafe {
        macro_rules! clip {
            ($x:expr) => {{
                _mm_min_epi32(
                    _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32($x, rnd), sh), minv),
                    maxv,
                )
            }};
        }
        let c0 = clip!(v[0]);
        let c1 = clip!(v[1]);
        let c2 = clip!(v[2]);
        let c3 = clip!(v[3]);
        let t0 = _mm_unpacklo_epi32(c0, c1);
        let t1 = _mm_unpackhi_epi32(c0, c1);
        let t2 = _mm_unpacklo_epi32(c2, c3);
        let t3 = _mm_unpackhi_epi32(c2, c3);
        let r0 = _mm_unpacklo_epi64(t0, t2);
        let r1 = _mm_unpackhi_epi64(t0, t2);
        let r2 = _mm_unpacklo_epi64(t1, t3);
        let r3 = _mm_unpackhi_epi64(t1, t3);
        _mm_storeu_si128(tmp_ptr(dst, off), r0);
        _mm_storeu_si128(tmp_ptr(dst, off + 32), r1);
        _mm_storeu_si128(tmp_ptr(dst, off + 64), r2);
        _mm_storeu_si128(tmp_ptr(dst, off + 96), r3);
    }
}

#[inline(always)]
unsafe fn tmp_ptr(dst: &mut [i32; ITX_TMP_PIXELS], off: usize) -> *mut __m128i {
    unsafe { dst.as_mut_ptr().add(off) as *mut __m128i }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn idct_dequant_16x16_avx2_i32_hardcoded(
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
        let z = _mm_setzero_si128();
        let rect_mul = _mm_set1_epi32(181);
        let rect_rnd = _mm_set1_epi32(128);
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);

        macro_rules! load4_i32_coeff {
            ($base:expr, $j:expr) => {{
                let mut v = _mm_loadu_si128(coeff.as_ptr().add($base + $j * 16) as *const __m128i);
                if is_rect2 {
                    v = _mm_srai_epi32::<8>(_mm_add_epi32(_mm_mullo_epi32(v, rect_mul), rect_rnd));
                }
                v
            }};
        }
        macro_rules! dct16x4_coeff {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let v = load4_i32_coeff!($base, j);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]),
                        ),
                    );
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
                    let v = _mm_loadu_si128(tmp.as_ptr().add($base + j * 32) as *const __m128i);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]),
                        ),
                    );
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
                avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
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
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + m * 32) as *mut __m128i, g[0]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 1) * 32) as *mut __m128i, g[1]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 2) * 32) as *mut __m128i, g[2]);
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + (m + 3) * 32) as *mut __m128i, g[3]);
                m += 4;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn idct_dequant_16x16_avx2_i16_hardcoded(
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
        let z = _mm256_setzero_si256();
        let rnd8 = _mm256_set1_epi32((1 << shift0) >> 1);
        let sh8 = _mm_cvtsi32_si128(shift0);
        let min8 = _mm256_set1_epi32(row_clip_min);
        let max8 = _mm256_set1_epi32(row_clip_max);
        let rnd4 = _mm_set1_epi32((1 << shift0) >> 1);
        let min4 = _mm_set1_epi32(row_clip_min);
        let max4 = _mm_set1_epi32(row_clip_max);

        macro_rules! pair {
            ($j:expr, $m:expr) => {{
                let k0 = crate::itx_2d::DCT16_DENSE_KERNEL[$j * 16 + $m] as i16;
                let k1 = crate::itx_2d::DCT16_DENSE_KERNEL[($j + 1) * 16 + $m] as i16;
                _mm256_set1_epi32(((k1 as u16 as i32) << 16) | (k0 as u16 as i32))
            }};
        }
        macro_rules! madd {
            ($acc:expr, $x0:expr, $x1:expr, $c:expr) => {{
                let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16($x0, $x1), $c);
                let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16($x0, $x1), $c);
                _mm256_add_epi32($acc, _mm256_permute2x128_si256::<0x20>(lo, hi))
            }};
        }
        macro_rules! dct16x8_coeff {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut a4 = z;
                let mut a5 = z;
                let mut a6 = z;
                let mut a7 = z;
                let mut j = 0usize;
                while j < 16 {
                    let x0 = avx2_load8_i16(coeff, $base + j * 16, is_rect2);
                    let x1 = avx2_load8_i16(coeff, $base + (j + 1) * 16, is_rect2);
                    a0 = madd!(a0, x0, x1, pair!(j, $m));
                    a1 = madd!(a1, x0, x1, pair!(j, $m + 1));
                    a2 = madd!(a2, x0, x1, pair!(j, $m + 2));
                    a3 = madd!(a3, x0, x1, pair!(j, $m + 3));
                    a4 = madd!(a4, x0, x1, pair!(j, $m + 4));
                    a5 = madd!(a5, x0, x1, pair!(j, $m + 5));
                    a6 = madd!(a6, x0, x1, pair!(j, $m + 6));
                    a7 = madd!(a7, x0, x1, pair!(j, $m + 7));
                    j += 2;
                }
                [a0, a1, a2, a3, a4, a5, a6, a7]
            }};
        }
        macro_rules! load4_i16_i32 {
            ($base:expr, $j:expr) => {{
                _mm_cvtepi16_epi32(_mm256_castsi256_si128(avx2_load4_i16(
                    coeff,
                    $base + $j * 16,
                    is_rect2,
                )))
            }};
        }
        macro_rules! dct16x4_coeff {
            ($base:expr, $m:expr) => {{
                let z4 = _mm_setzero_si128();
                let mut a0 = z4;
                let mut a1 = z4;
                let mut a2 = z4;
                let mut a3 = z4;
                let mut j = 0usize;
                while j < 16 {
                    let v = load4_i16_i32!($base, j);
                    a0 = _mm_add_epi32(
                        a0,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]),
                        ),
                    );
                    a1 = _mm_add_epi32(
                        a1,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]),
                        ),
                    );
                    a2 = _mm_add_epi32(
                        a2,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]),
                        ),
                    );
                    a3 = _mm_add_epi32(
                        a3,
                        _mm_mullo_epi32(
                            v,
                            _mm_set1_epi32(crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]),
                        ),
                    );
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }
        macro_rules! dct16x8_tmp_one {
            ($base:expr, $m:expr) => {{
                let mut acc = z;
                let mut j = 0usize;
                while j < 16 {
                    let x0 = avx2_load8_narrow_i32(tmp, $base + j * 32);
                    let x1 = avx2_load8_narrow_i32(tmp, $base + (j + 1) * 32);
                    acc = madd!(acc, x0, x1, pair!(j, $m));
                    j += 2;
                }
                acc
            }};
        }

        let mut y = 0usize;
        while y + 8 <= ncols {
            let mut x = 0usize;
            while x < 16 {
                let g = dct16x8_coeff!(y, x);
                avx2_store8x8_clip_i32(&mut tmp[..], y * 32 + x, 32, &g, rnd8, sh8, min8, max8);
                x += 8;
            }
            y += 8;
        }
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 16 {
                let g = dct16x4_coeff!(y, x);
                avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd4, sh8, min4, max4);
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
                let acc = dct16x8_tmp_one!(x, m);
                _mm256_storeu_si256(tmp.as_mut_ptr().add(x + m * 32) as *mut __m256i, acc);
                m += 1;
            }
            x += 8;
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn idct_dequant_32x32_avx2_i32_hardcoded(
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
        let rnd = _mm_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm_set1_epi32(row_clip_min);
        let maxv = _mm_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut s = [_mm_setzero_si128(); 32];
            let mut j = 0usize;
            while j < 32 {
                let mut v = _mm_loadu_si128(coeff.as_ptr().add(y + j * 32) as *const __m128i);
                if is_rect2 {
                    v = _mm_srai_epi32::<8>(_mm_add_epi32(
                        _mm_mullo_epi32(v, _mm_set1_epi32(181)),
                        _mm_set1_epi32(128),
                    ));
                }
                s[j] = v;
                j += 1;
            }
            let out = avx2_dct32_i32x4_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                avx2_store4x4_i32_clip(tmp, y * 32 + x, &g, rnd, sh, minv, maxv);
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
            let mut s = [_mm_setzero_si128(); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = _mm_loadu_si128(tmp.as_ptr().add(x + j * 32) as *const __m128i);
                j += 1;
            }
            let out = avx2_dct32_i32x4_hardcoded(&s);
            j = 0;
            while j < 32 {
                _mm_storeu_si128(tmp.as_mut_ptr().add(x + j * 32) as *mut __m128i, out[j]);
                j += 1;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "avx2,sse4.1")]
unsafe fn idct_dequant_32x32_avx2_i16_hardcoded(
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
        let rnd = _mm256_set1_epi32((1 << shift0) >> 1);
        let sh = _mm_cvtsi32_si128(shift0);
        let minv = _mm256_set1_epi32(row_clip_min);
        let maxv = _mm256_set1_epi32(row_clip_max);
        let mut y = 0usize;
        while y + 8 <= ncols {
            let mut s = [_mm256_setzero_si256(); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = avx2_load8_i16(coeff, y + j * 32, is_rect2);
                j += 1;
            }
            let out = avx2_dct32_i16x8_hardcoded(&s);
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
                avx2_store8x8_clip_i32(tmp, y * 32 + x, 32, &g, rnd, sh, minv, maxv);
                x += 8;
            }
            y += 8;
        }
        if y + 4 <= ncols {
            let mut s = [_mm256_setzero_si256(); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = avx2_load4_i16(coeff, y + j * 32, is_rect2);
                j += 1;
            }
            let out = avx2_dct32_i16x8_hardcoded(&s);
            let mut x = 0usize;
            while x < 32 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                avx2_store4x4_clip_i32(tmp, y * 32 + x, 32, &g, rnd, sh, minv, maxv);
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
            let mut s = [_mm256_setzero_si256(); 32];
            let mut j = 0usize;
            while j < 32 {
                s[j] = avx2_load8_narrow_i32(tmp, x + j * 32);
                j += 1;
            }
            let out = avx2_dct32_i16x8_hardcoded(&s);
            j = 0;
            while j < 32 {
                _mm256_storeu_si256(tmp.as_mut_ptr().add(x + j * 32) as *mut __m256i, out[j]);
                j += 1;
            }
            x += 8;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct AvxI32x4(__m128i);

impl crate::itx_1d::DctLane for AvxI32x4 {
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn zero() -> Self {
        AvxI32x4(unsafe { _mm_setzero_si128() })
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn add(self, o: Self) -> Self {
        AvxI32x4(unsafe { _mm_add_epi32(self.0, o.0) })
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn sub(self, o: Self) -> Self {
        AvxI32x4(unsafe { _mm_sub_epi32(self.0, o.0) })
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mul(self, k: Self) -> Self {
        AvxI32x4(unsafe { _mm_mullo_epi32(self.0, k.0) })
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn dup_load(table: &[i32], idx: usize) -> Self {
        // SAFETY: callers index within the kernel tables.
        AvxI32x4(unsafe { _mm_set1_epi32(*table.get_unchecked(idx)) })
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mul_add(self, x: Self, k: Self) -> Self {
        AvxI32x4(unsafe { _mm_add_epi32(self.0, _mm_mullo_epi32(x.0, k.0)) })
    }
    type Coeffs = __m128i;
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load_coeffs(table: &[i32], idx: usize) -> __m128i {
        // SAFETY: callers index a 4-wide group within the kernel tables.
        unsafe { _mm_loadu_si128(table.as_ptr().add(idx) as *const __m128i) }
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mul_add_lane<const LANE: i32>(self, x: Self, c: __m128i) -> Self {
        let bc = unsafe {
            match LANE {
                0 => _mm_shuffle_epi32(c, 0x00),
                1 => _mm_shuffle_epi32(c, 0x55),
                2 => _mm_shuffle_epi32(c, 0xAA),
                _ => _mm_shuffle_epi32(c, 0xFF),
            }
        };
        AvxI32x4(unsafe { _mm_add_epi32(self.0, _mm_mullo_epi32(x.0, bc)) })
    }
}

pub(crate) struct AvxWide;

impl crate::itx_1d::DctWide for AvxWide {
    type In = __m256i;
    type Acc = __m256i;
    type Coeffs = __m256i;
    type Clip = (__m256i, __m128i, __m256i, __m256i);

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn zero() -> Self::Acc {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { _mm256_add_epi32(a, b) }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { _mm256_sub_epi32(a, b) }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load_coeffs(table: &[i16], idx: usize) -> __m256i {
        unsafe {
            let c = _mm_loadu_si128(table.as_ptr().add(idx) as *const __m128i);
            _mm256_broadcastsi128_si256(c)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mul_add_lane<const LANE: i32>(acc: Self::Acc, x: __m256i, c: __m256i) -> Self::Acc {
        let raw = match LANE {
            0 => _mm_extract_epi16(_mm256_castsi256_si128(c), 0),
            1 => _mm_extract_epi16(_mm256_castsi256_si128(c), 1),
            2 => _mm_extract_epi16(_mm256_castsi256_si128(c), 2),
            3 => _mm_extract_epi16(_mm256_castsi256_si128(c), 3),
            4 => _mm_extract_epi16(_mm256_castsi256_si128(c), 4),
            5 => _mm_extract_epi16(_mm256_castsi256_si128(c), 5),
            6 => _mm_extract_epi16(_mm256_castsi256_si128(c), 6),
            _ => _mm_extract_epi16(_mm256_castsi256_si128(c), 7),
        };
        let xlo16 = _mm256_unpacklo_epi16(x, _mm256_setzero_si256());
        let xhi16 = _mm256_unpackhi_epi16(x, _mm256_setzero_si256());
        let lo = _mm256_madd_epi16(xlo16, _mm256_set1_epi16(raw as i16));
        let hi = _mm256_madd_epi16(xhi16, _mm256_set1_epi16(raw as i16));
        // lo contains rows 0..3 in its low 128-bit lane; hi contains rows
        // 4..7 in its low 128-bit lane. Combine those halves explicitly.
        let sum8 = _mm256_permute2x128_si256::<0x20>(lo, hi);
        _mm256_add_epi32(acc, sum8)
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mul_add_pair<const LANE0: i32, const LANE1: i32>(
        acc: Self::Acc,
        x0: __m256i,
        x1: __m256i,
        c: __m256i,
    ) -> Self::Acc {
        let _ = LANE1;
        debug_assert_eq!(LANE1, LANE0 + 1);
        debug_assert_eq!(LANE0 & 1, 0);

        let k01 = match LANE0 {
            0 => _mm256_shuffle_epi32::<0x00>(c),
            2 => _mm256_shuffle_epi32::<0x55>(c),
            4 => _mm256_shuffle_epi32::<0xaa>(c),
            _ => _mm256_shuffle_epi32::<0xff>(c),
        };
        let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(x0, x1), k01);
        let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(x0, x1), k01);
        let sum8 = _mm256_permute2x128_si256::<0x20>(lo, hi);
        _mm256_add_epi32(acc, sum8)
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe {
            let v = _mm256_loadu_si256(src.as_ptr().add(off) as *const __m256i);
            let p = _mm256_packs_epi32(v, _mm256_setzero_si256());
            // packs_epi32 is lane-local: [0..3, z, 4..7, z] -> [0..7, z].
            _mm256_permute4x64_epi64::<0xd8>(p)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe {
            let x = Self::load8_narrow(src, off);
            _mm256_mulhrs_epi16(x, _mm256_set1_epi16(0x5a80))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe {
            let lo = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            let p = _mm_packs_epi32(lo, _mm_setzero_si128());
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), p)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> __m256i {
        unsafe { _mm256_mulhrs_epi16(Self::load4_narrow(src, off), _mm256_set1_epi16(0x5a80)) }
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load8_i16(src: &[i16], off: usize) -> __m256i {
        debug_assert!(off + 8 <= src.len());
        unsafe {
            let x = _mm_loadu_si128(src.as_ptr().add(off) as *const __m128i);
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), x)
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> __m256i {
        unsafe { _mm256_mulhrs_epi16(Self::load8_i16(src, off), _mm256_set1_epi16(0x5a80)) }
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load4_i16(src: &[i16], off: usize) -> __m256i {
        debug_assert!(off + 4 <= src.len());
        unsafe {
            let x = _mm_loadl_epi64(src.as_ptr().add(off) as *const __m128i);
            _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), x)
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> __m256i {
        unsafe { _mm256_mulhrs_epi16(Self::load4_i16(src, off), _mm256_set1_epi16(0x5a80)) }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        (
            _mm256_set1_epi32(rnd),
            _mm_cvtsi32_si128(shift),
            _mm256_set1_epi32(min),
            _mm256_set1_epi32(max),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, sh, minv, maxv) = clip;
            let v = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(acc, rnd), sh), minv),
                maxv,
            );
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            #[inline(always)]
            unsafe fn store_lane0(dst: &mut [i32], off: usize, v: __m128i) {
                unsafe { _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(v)) };
            }

            store_lane0(dst, off, lo);
            store_lane0(dst, off + 1 * stride, _mm_shuffle_epi32::<0x55>(lo));
            store_lane0(dst, off + 2 * stride, _mm_shuffle_epi32::<0xaa>(lo));
            store_lane0(dst, off + 3 * stride, _mm_shuffle_epi32::<0xff>(lo));
            store_lane0(dst, off + 4 * stride, hi);
            store_lane0(dst, off + 5 * stride, _mm_shuffle_epi32::<0x55>(hi));
            store_lane0(dst, off + 6 * stride, _mm_shuffle_epi32::<0xaa>(hi));
            store_lane0(dst, off + 7 * stride, _mm_shuffle_epi32::<0xff>(hi));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, sh, minv, maxv) = clip;
            let v = _mm256_min_epi32(
                _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(acc, rnd), sh), minv),
                maxv,
            );
            let lo = _mm256_castsi256_si128(v);
            #[inline(always)]
            unsafe fn store_lane0(dst: &mut [i32], off: usize, v: __m128i) {
                unsafe { _mm_store_ss(dst.as_mut_ptr().add(off).cast(), _mm_castsi128_ps(v)) };
            }

            store_lane0(dst, off, lo);
            store_lane0(dst, off + 1 * stride, _mm_shuffle_epi32::<0x55>(lo));
            store_lane0(dst, off + 2 * stride, _mm_shuffle_epi32::<0xaa>(lo));
            store_lane0(dst, off + 3 * stride, _mm_shuffle_epi32::<0xff>(lo));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store4x4_strided_clip<const HIGH: bool>(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 4],
        clip: Self::Clip,
    ) {
        unsafe {
            #[inline(always)]
            unsafe fn clip_lane<const HIGH: bool>(
                v: __m256i,
                rnd: __m256i,
                sh: __m128i,
                minv: __m256i,
                maxv: __m256i,
            ) -> __m128i {
                unsafe {
                    let v = _mm256_min_epi32(
                        _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(v, rnd), sh), minv),
                        maxv,
                    );
                    if HIGH {
                        _mm256_extracti128_si256::<1>(v)
                    } else {
                        _mm256_castsi256_si128(v)
                    }
                }
            }
            let (rnd, sh, minv, maxv) = clip;
            let c0 = clip_lane::<HIGH>(acc[0], rnd, sh, minv, maxv);
            let c1 = clip_lane::<HIGH>(acc[1], rnd, sh, minv, maxv);
            let c2 = clip_lane::<HIGH>(acc[2], rnd, sh, minv, maxv);
            let c3 = clip_lane::<HIGH>(acc[3], rnd, sh, minv, maxv);

            let t0 = _mm_unpacklo_epi32(c0, c1);
            let t1 = _mm_unpackhi_epi32(c0, c1);
            let t2 = _mm_unpacklo_epi32(c2, c3);
            let t3 = _mm_unpackhi_epi32(c2, c3);
            let r0 = _mm_unpacklo_epi64(t0, t2);
            let r1 = _mm_unpackhi_epi64(t0, t2);
            let r2 = _mm_unpacklo_epi64(t1, t3);
            let r3 = _mm_unpackhi_epi64(t1, t3);

            _mm_storeu_si128(dst.as_mut_ptr().add(off) as *mut __m128i, r0);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + stride) as *mut __m128i, r1);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + 2 * stride) as *mut __m128i, r2);
            _mm_storeu_si128(dst.as_mut_ptr().add(off + 3 * stride) as *mut __m128i, r3);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store8x8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: [Self::Acc; 8],
        clip: Self::Clip,
    ) {
        debug_assert!(off + 7 + 7 * stride < dst.len());
        unsafe {
            #[inline(always)]
            unsafe fn clip_vec(
                v: __m256i,
                rnd: __m256i,
                sh: __m128i,
                minv: __m256i,
                maxv: __m256i,
            ) -> __m256i {
                unsafe {
                    _mm256_min_epi32(
                        _mm256_max_epi32(_mm256_sra_epi32(_mm256_add_epi32(v, rnd), sh), minv),
                        maxv,
                    )
                }
            }

            #[inline(always)]
            unsafe fn store_row(dst: &mut [i32], off: usize, v: __m256i) {
                unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, v) };
            }

            let (rnd, sh, minv, maxv) = clip;
            let c0 = clip_vec(acc[0], rnd, sh, minv, maxv);
            let c1 = clip_vec(acc[1], rnd, sh, minv, maxv);
            let c2 = clip_vec(acc[2], rnd, sh, minv, maxv);
            let c3 = clip_vec(acc[3], rnd, sh, minv, maxv);
            let c4 = clip_vec(acc[4], rnd, sh, minv, maxv);
            let c5 = clip_vec(acc[5], rnd, sh, minv, maxv);
            let c6 = clip_vec(acc[6], rnd, sh, minv, maxv);
            let c7 = clip_vec(acc[7], rnd, sh, minv, maxv);

            // cN is one output column with lanes [r0cN..r7cN]. First build
            // 4-column row fragments in each 128-bit lane, then join low/high
            // halves from columns 0..3 and 4..7 into eight full rows.
            let t0 = _mm256_unpacklo_epi32(c0, c1);
            let t1 = _mm256_unpackhi_epi32(c0, c1);
            let t2 = _mm256_unpacklo_epi32(c2, c3);
            let t3 = _mm256_unpackhi_epi32(c2, c3);
            let t4 = _mm256_unpacklo_epi32(c4, c5);
            let t5 = _mm256_unpackhi_epi32(c4, c5);
            let t6 = _mm256_unpacklo_epi32(c6, c7);
            let t7 = _mm256_unpackhi_epi32(c6, c7);

            let a0 = _mm256_unpacklo_epi64(t0, t2);
            let a1 = _mm256_unpackhi_epi64(t0, t2);
            let a2 = _mm256_unpacklo_epi64(t1, t3);
            let a3 = _mm256_unpackhi_epi64(t1, t3);
            let a4 = _mm256_unpacklo_epi64(t4, t6);
            let a5 = _mm256_unpackhi_epi64(t4, t6);
            let a6 = _mm256_unpacklo_epi64(t5, t7);
            let a7 = _mm256_unpackhi_epi64(t5, t7);

            let r0 = _mm256_permute2x128_si256::<0x20>(a0, a4);
            let r1 = _mm256_permute2x128_si256::<0x20>(a1, a5);
            let r2 = _mm256_permute2x128_si256::<0x20>(a2, a6);
            let r3 = _mm256_permute2x128_si256::<0x20>(a3, a7);
            let r4 = _mm256_permute2x128_si256::<0x31>(a0, a4);
            let r5 = _mm256_permute2x128_si256::<0x31>(a1, a5);
            let r6 = _mm256_permute2x128_si256::<0x31>(a2, a6);
            let r7 = _mm256_permute2x128_si256::<0x31>(a3, a7);

            store_row(dst, off, r0);
            store_row(dst, off + stride, r1);
            store_row(dst, off + 2 * stride, r2);
            store_row(dst, off + 3 * stride, r3);
            store_row(dst, off + 4 * stride, r4);
            store_row(dst, off + 5 * stride, r5);
            store_row(dst, off + 6 * stride, r6);
            store_row(dst, off + 7 * stride, r7);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, acc) };
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            _mm_storeu_si128(
                dst.as_mut_ptr().add(off) as *mut __m128i,
                _mm256_castsi256_si128(acc),
            )
        };
    }
}

pub(crate) struct AvxDct2d;

impl DctSimd4 for AvxDct2d {
    type V = AvxI32x4;
    type Wide = AvxWide;

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn zero() -> Self::V {
        AvxI32x4(_mm_setzero_si128())
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn splat(v: i32) -> Self::V {
        AvxI32x4(_mm_set1_epi32(v))
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        AvxI32x4(unsafe { _mm_add_epi32(a.0, b.0) })
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        AvxI32x4(unsafe { _mm_sub_epi32(a.0, b.0) })
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        AvxI32x4(_mm_mullo_epi32(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        let scaled = _mm_add_epi32(
            _mm_mullo_epi32(a.0, _mm_set1_epi32(181)),
            _mm_set1_epi32(128),
        );
        AvxI32x4(_mm_srai_epi32::<8>(scaled))
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_ptr().add(off) as *const __m128i };
        AvxI32x4(unsafe { _mm_loadu_si128(p) })
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_mut_ptr().add(off) as *mut __m128i };
        unsafe { _mm_storeu_si128(p, v.0) };
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) as *const __m128i };
        AvxI32x4(unsafe { _mm_loadu_si128(p) })
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) as *const __m128i };
        AvxI32x4(unsafe { _mm_cvtepi16_epi32(_mm_loadl_epi64(p)) })
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        let mut out = [0i32; 4];
        let p = out.as_mut_ptr() as *mut __m128i;
        unsafe { _mm_storeu_si128(p, v.0) };
        out
    }

    #[inline]
    #[target_feature(enable = "avx2,sse4.1")]
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
                v: __m128i,
                rnd: __m128i,
                sh: __m128i,
                minv: __m128i,
                maxv: __m128i,
            ) -> __m128i {
                unsafe {
                    _mm_min_epi32(
                        _mm_max_epi32(_mm_sra_epi32(_mm_add_epi32(v, rnd), sh), minv),
                        maxv,
                    )
                }
            }

            let rnd = _mm_set1_epi32(rnd);
            let sh = _mm_cvtsi32_si128(shift);
            let minv = _mm_set1_epi32(min);
            let maxv = _mm_set1_epi32(max);

            let c0 = clip_vec(v[0].0, rnd, sh, minv, maxv);
            let c1 = clip_vec(v[1].0, rnd, sh, minv, maxv);
            let c2 = clip_vec(v[2].0, rnd, sh, minv, maxv);
            let c3 = clip_vec(v[3].0, rnd, sh, minv, maxv);

            // Transpose columns-as-lanes into four row vectors:
            // cN = [r0cN, r1cN, r2cN, r3cN].
            let t0 = _mm_unpacklo_epi32(c0, c1);
            let t1 = _mm_unpackhi_epi32(c0, c1);
            let t2 = _mm_unpacklo_epi32(c2, c3);
            let t3 = _mm_unpackhi_epi32(c2, c3);
            let r0 = _mm_unpacklo_epi64(t0, t2);
            let r1 = _mm_unpackhi_epi64(t0, t2);
            let r2 = _mm_unpacklo_epi64(t1, t3);
            let r3 = _mm_unpackhi_epi64(t1, t3);

            let p = tmp.as_mut_ptr().add(off) as *mut __m128i;
            _mm_storeu_si128(p, r0);
            _mm_storeu_si128(tmp.as_mut_ptr().add(off + stride) as *mut __m128i, r1);
            _mm_storeu_si128(tmp.as_mut_ptr().add(off + 2 * stride) as *mut __m128i, r2);
            _mm_storeu_si128(tmp.as_mut_ptr().add(off + 3 * stride) as *mut __m128i, r3);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x4_avx2(
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
        AvxDct2d,
        16,
        4,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x8_avx2(
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
        AvxDct2d,
        64,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_avx2(
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
        idct_dequant_16x16_avx2_i32_hardcoded(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_avx2(
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
        idct_dequant_32x32_avx2_i32_hardcoded(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_64x64_avx2(
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
        AvxDct2d,
        1024,
        32,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_4x4_avx2(
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
        AvxDct2d,
        16,
        4,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x8_avx2(
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
        AvxDct2d,
        64,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x16_avx2(
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
        iadst_dequant_16x16_avx2_i32_hardcoded(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x8_avx2(
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
        AvxDct2d,
        32,
        4,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x4_avx2(
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
        AvxDct2d,
        32,
        8,
        4,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x16_avx2(
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
        AvxDct2d,
        128,
        8,
        16,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x8_avx2(
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
        AvxDct2d,
        128,
        16,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x32_avx2(
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
        AvxDct2d,
        512,
        16,
        32,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x16_avx2(
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
        AvxDct2d,
        512,
        32,
        16,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x16_avx2(
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
        AvxDct2d,
        64,
        4,
        16,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x4_avx2(
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
        AvxDct2d,
        64,
        16,
        4,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x32_avx2(
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
        AvxDct2d,
        256,
        8,
        32,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x8_avx2(
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
        AvxDct2d,
        256,
        32,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_4x32_avx2(
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
        AvxDct2d,
        128,
        4,
        32,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x4_avx2(
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
        AvxDct2d,
        128,
        32,
        4,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_4x8_avx2(
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
        AvxDct2d,
        32,
        4,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x4_avx2(
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
        AvxDct2d,
        32,
        8,
        4,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_8x16_avx2(
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
        AvxDct2d,
        128,
        8,
        16,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x8_avx2(
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
        AvxDct2d,
        128,
        16,
        8,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_4x16_avx2(
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
        AvxDct2d,
        64,
        4,
        16,
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

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x4_avx2(
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
        AvxDct2d,
        64,
        16,
        4,
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

// Low-bit-depth i16 coefficient entry points.

macro_rules! idct_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $s:expr) => {
        #[inline]
        #[target_feature(enable = "avx2")]
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
                AvxDct2d,
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
                row_clip_max
            );
        }
    };
}
macro_rules! iadst_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $s:expr) => {
        #[inline]
        #[target_feature(enable = "avx2")]
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
                AvxDct2d,
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
                second_kind
            );
        }
    };
}
macro_rules! idct_rect_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        #[target_feature(enable = "avx2")]
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
                AvxDct2d,
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
                row_clip_max
            );
        }
    };
}
macro_rules! iadst_rect_i16_fn {
    ($pub:ident, $imp:ident, $n:expr, $w:expr, $h:expr) => {
        #[inline]
        #[target_feature(enable = "avx2")]
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
                AvxDct2d,
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
                second_kind
            );
        }
    };
}
idct_i16_fn!(
    idct_dequant_4x4_i16_avx2,
    idct_dequant_4x4_i16_avx2_impl,
    16,
    4
);
idct_i16_fn!(
    idct_dequant_8x8_i16_avx2,
    idct_dequant_8x8_i16_avx2_impl,
    64,
    8
);
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_i16_avx2(
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
        idct_dequant_16x16_avx2_i16_hardcoded(
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
#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_i16_avx2(
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
        idct_dequant_32x32_avx2_i16_hardcoded(
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
idct_i16_fn!(
    idct_dequant_64x64_i16_avx2,
    idct_dequant_64x64_i16_avx2_impl,
    1024,
    32
);
iadst_i16_fn!(
    iadst_dequant_4x4_i16_avx2,
    iadst_dequant_4x4_i16_avx2_impl,
    16,
    4
);
iadst_i16_fn!(
    iadst_dequant_8x8_i16_avx2,
    iadst_dequant_8x8_i16_avx2_impl,
    64,
    8
);
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x16_i16_avx2(
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
        iadst_dequant_16x16_avx2_i16_hardcoded(
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
idct_rect_i16_fn!(
    idct_dequant_4x8_i16_avx2,
    idct_dequant_4x8_i16_avx2_impl,
    32,
    4,
    8
);
idct_rect_i16_fn!(
    idct_dequant_8x4_i16_avx2,
    idct_dequant_8x4_i16_avx2_impl,
    32,
    8,
    4
);
idct_rect_i16_fn!(
    idct_dequant_8x16_i16_avx2,
    idct_dequant_8x16_i16_avx2_impl,
    128,
    8,
    16
);
idct_rect_i16_fn!(
    idct_dequant_16x8_i16_avx2,
    idct_dequant_16x8_i16_avx2_impl,
    128,
    16,
    8
);
idct_rect_i16_fn!(
    idct_dequant_16x32_i16_avx2,
    idct_dequant_16x32_i16_avx2_impl,
    512,
    16,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x16_i16_avx2,
    idct_dequant_32x16_i16_avx2_impl,
    512,
    32,
    16
);
idct_rect_i16_fn!(
    idct_dequant_4x16_i16_avx2,
    idct_dequant_4x16_i16_avx2_impl,
    64,
    4,
    16
);
idct_rect_i16_fn!(
    idct_dequant_16x4_i16_avx2,
    idct_dequant_16x4_i16_avx2_impl,
    64,
    16,
    4
);
idct_rect_i16_fn!(
    idct_dequant_8x32_i16_avx2,
    idct_dequant_8x32_i16_avx2_impl,
    256,
    8,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x8_i16_avx2,
    idct_dequant_32x8_i16_avx2_impl,
    256,
    32,
    8
);
idct_rect_i16_fn!(
    idct_dequant_4x32_i16_avx2,
    idct_dequant_4x32_i16_avx2_impl,
    128,
    4,
    32
);
idct_rect_i16_fn!(
    idct_dequant_32x4_i16_avx2,
    idct_dequant_32x4_i16_avx2_impl,
    128,
    32,
    4
);
iadst_rect_i16_fn!(
    iadst_dequant_4x8_i16_avx2,
    iadst_dequant_4x8_i16_avx2_impl,
    32,
    4,
    8
);
iadst_rect_i16_fn!(
    iadst_dequant_8x4_i16_avx2,
    iadst_dequant_8x4_i16_avx2_impl,
    32,
    8,
    4
);
iadst_rect_i16_fn!(
    iadst_dequant_8x16_i16_avx2,
    iadst_dequant_8x16_i16_avx2_impl,
    128,
    8,
    16
);
iadst_rect_i16_fn!(
    iadst_dequant_16x8_i16_avx2,
    iadst_dequant_16x8_i16_avx2_impl,
    128,
    16,
    8
);
iadst_rect_i16_fn!(
    iadst_dequant_4x16_i16_avx2,
    iadst_dequant_4x16_i16_avx2_impl,
    64,
    4,
    16
);
iadst_rect_i16_fn!(
    iadst_dequant_16x4_i16_avx2,
    iadst_dequant_16x4_i16_avx2_impl,
    64,
    16,
    4
);
