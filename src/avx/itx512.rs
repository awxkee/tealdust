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

use std::arch::x86_64::*;

#[inline(always)]
fn with_avx512_itx_i16_scratch<R>(len: usize, f: impl FnOnce(&mut [i16]) -> R) -> R {
    match len {
        16 => {
            let mut scratch = [0i16; 16];
            f(&mut scratch)
        }
        32 => {
            let mut scratch = [0i16; 32];
            f(&mut scratch)
        }
        64 => {
            let mut scratch = [0i16; 64];
            f(&mut scratch)
        }
        128 => {
            let mut scratch = [0i16; 128];
            f(&mut scratch)
        }
        256 => {
            let mut scratch = [0i16; 256];
            f(&mut scratch)
        }
        512 => {
            let mut scratch = [0i16; 512];
            f(&mut scratch)
        }
        1024 => {
            let mut scratch = [0i16; 1024];
            f(&mut scratch)
        }
        _ => unreachable!("unsupported ITX i16 scratch len {}", len),
    }
}

#[inline(always)]
fn avx512_mask16(live: usize) -> __mmask16 {
    debug_assert!(live <= 16);
    if live == 16 {
        u16::MAX
    } else {
        ((1u32 << live) - 1) as __mmask16
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_load_coeff_i16x16<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
    live: usize,
    row: usize,
) -> __m256i {
    debug_assert!(live <= 16);
    debug_assert!(base + live <= STRIDE);
    debug_assert!(base + row * STRIDE + live <= coeff.len());

    let mask = avx512_mask16(live);
    let ptr = unsafe { coeff.as_ptr().add(base + row * STRIDE) };
    let mut v = unsafe { _mm256_maskz_loadu_epi16(mask, ptr.cast()) };
    if IS_RECT2 {
        v = _mm256_mulhrs_epi16(v, _mm256_set1_epi16(0x5a80));
    }
    v
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_load_scratch_i16x16<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
    row: usize,
) -> __m256i {
    debug_assert!(base + 16 <= STRIDE);
    if row >= active {
        return _mm256_setzero_si256();
    }
    debug_assert!(base + row * STRIDE + 16 <= scratch.len());
    unsafe { _mm256_loadu_si256(scratch.as_ptr().add(base + row * STRIDE).cast()) }
}

#[inline(always)]
fn avx512_pair_coeff(table: &[i32], idx: usize) -> (i16, i16) {
    let p = table[idx * 4];
    (p as i16, (p >> 16) as i16)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_i16_interleave_x16(a: __m256i, b: __m256i) -> __m512i {
    let lo = _mm256_unpacklo_epi16(a, b);
    let hi = _mm256_unpackhi_epi16(a, b);
    let p0 = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(_mm256_castsi256_si128(lo)),
        _mm256_castsi256_si128(hi),
    );
    let p1 = _mm256_inserti128_si256::<1>(
        _mm256_castsi128_si256(_mm256_extracti128_si256::<1>(lo)),
        _mm256_extracti128_si256::<1>(hi),
    );
    _mm512_inserti64x4::<1>(_mm512_zextsi256_si512(p0), p1)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_coeff_pair_512(k0: i16, k1: i16) -> __m512i {
    let pair = (k0 as u16 as u32) | ((k1 as u16 as u32) << 16);
    _mm512_set1_epi32(pair as i32)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_madd_pair_i16_i32(a: __m256i, b: __m256i, table: &[i32], idx: usize) -> __m512i {
    let (k0, k1) = avx512_pair_coeff(table, idx);
    let ab = avx512_i16_interleave_x16(a, b);
    _mm512_dpwssds_epi32(_mm512_setzero_si512(), ab, avx512_coeff_pair_512(k0, k1))
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn avx512_round_clip_i32_to_i16(
    v: __m512i,
    rnd: __m512i,
    sh: __m128i,
    minv: __m512i,
    maxv: __m512i,
) -> __m256i {
    let v = _mm512_sra_epi32(_mm512_add_epi32(v, rnd), sh);
    let v = _mm512_min_epi32(_mm512_max_epi32(v, minv), maxv);
    _mm512_cvtsepi32_epi16(v)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn avx512_transpose16x16_i16_store<const STRIDE: usize>(
    dst: &mut [i16],
    off: usize,
    r: &[__m256i; 16],
) {
    unsafe {
        debug_assert!(off + 15 * STRIDE + 16 <= dst.len());
        debug_assert!(matches!(STRIDE, 16 | 32));

        let a0 = _mm256_unpacklo_epi16(r[0], r[1]);
        let a1 = _mm256_unpackhi_epi16(r[0], r[1]);
        let a2 = _mm256_unpacklo_epi16(r[2], r[3]);
        let a3 = _mm256_unpackhi_epi16(r[2], r[3]);
        let a4 = _mm256_unpacklo_epi16(r[4], r[5]);
        let a5 = _mm256_unpackhi_epi16(r[4], r[5]);
        let a6 = _mm256_unpacklo_epi16(r[6], r[7]);
        let a7 = _mm256_unpackhi_epi16(r[6], r[7]);
        let a8 = _mm256_unpacklo_epi16(r[8], r[9]);
        let a9 = _mm256_unpackhi_epi16(r[8], r[9]);
        let a10 = _mm256_unpacklo_epi16(r[10], r[11]);
        let a11 = _mm256_unpackhi_epi16(r[10], r[11]);
        let a12 = _mm256_unpacklo_epi16(r[12], r[13]);
        let a13 = _mm256_unpackhi_epi16(r[12], r[13]);
        let a14 = _mm256_unpacklo_epi16(r[14], r[15]);
        let a15 = _mm256_unpackhi_epi16(r[14], r[15]);

        let b0 = _mm256_unpacklo_epi32(a0, a2);
        let b1 = _mm256_unpackhi_epi32(a0, a2);
        let b2 = _mm256_unpacklo_epi32(a1, a3);
        let b3 = _mm256_unpackhi_epi32(a1, a3);
        let b4 = _mm256_unpacklo_epi32(a4, a6);
        let b5 = _mm256_unpackhi_epi32(a4, a6);
        let b6 = _mm256_unpacklo_epi32(a5, a7);
        let b7 = _mm256_unpackhi_epi32(a5, a7);
        let b8 = _mm256_unpacklo_epi32(a8, a10);
        let b9 = _mm256_unpackhi_epi32(a8, a10);
        let b10 = _mm256_unpacklo_epi32(a9, a11);
        let b11 = _mm256_unpackhi_epi32(a9, a11);
        let b12 = _mm256_unpacklo_epi32(a12, a14);
        let b13 = _mm256_unpackhi_epi32(a12, a14);
        let b14 = _mm256_unpacklo_epi32(a13, a15);
        let b15 = _mm256_unpackhi_epi32(a13, a15);

        let c0 = _mm256_unpacklo_epi64(b0, b4);
        let c1 = _mm256_unpackhi_epi64(b0, b4);
        let c2 = _mm256_unpacklo_epi64(b1, b5);
        let c3 = _mm256_unpackhi_epi64(b1, b5);
        let c4 = _mm256_unpacklo_epi64(b2, b6);
        let c5 = _mm256_unpackhi_epi64(b2, b6);
        let c6 = _mm256_unpacklo_epi64(b3, b7);
        let c7 = _mm256_unpackhi_epi64(b3, b7);
        let c8 = _mm256_unpacklo_epi64(b8, b12);
        let c9 = _mm256_unpackhi_epi64(b8, b12);
        let c10 = _mm256_unpacklo_epi64(b9, b13);
        let c11 = _mm256_unpackhi_epi64(b9, b13);
        let c12 = _mm256_unpacklo_epi64(b10, b14);
        let c13 = _mm256_unpackhi_epi64(b10, b14);
        let c14 = _mm256_unpacklo_epi64(b11, b15);
        let c15 = _mm256_unpackhi_epi64(b11, b15);

        macro_rules! store_pair {
            ($base:expr, $lo:expr, $hi:expr) => {{
                let r0 = _mm256_permute2x128_si256::<0x20>($lo, $hi);
                let r1 = _mm256_permute2x128_si256::<0x31>($lo, $hi);
                _mm256_storeu_si256(dst.as_mut_ptr().add(off + $base * STRIDE).cast(), r0);
                _mm256_storeu_si256(dst.as_mut_ptr().add(off + ($base + 8) * STRIDE).cast(), r1);
            }};
        }

        store_pair!(0, c0, c8);
        store_pair!(1, c1, c9);
        store_pair!(2, c2, c10);
        store_pair!(3, c3, c11);
        store_pair!(4, c4, c12);
        store_pair!(5, c5, c13);
        store_pair!(6, c6, c14);
        store_pair!(7, c7, c15);
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn avx512_store_rowpass_i16x16<const STRIDE: usize>(
    scratch: &mut [i16],
    row_base: usize,
    out_base: usize,
    rows: &[__m512i; 16],
    rnd: i32,
    shift: i32,
    minv: i32,
    maxv: i32,
) {
    let rnd = _mm512_set1_epi32(rnd);
    let sh = _mm_cvtsi32_si128(shift);
    let minv = _mm512_set1_epi32(minv);
    let maxv = _mm512_set1_epi32(maxv);
    let packed = [
        avx512_round_clip_i32_to_i16(rows[0], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[1], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[2], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[3], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[4], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[5], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[6], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[7], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[8], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[9], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[10], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[11], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[12], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[13], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[14], rnd, sh, minv, maxv),
        avx512_round_clip_i32_to_i16(rows[15], rnd, sh, minv, maxv),
    ];
    avx512_transpose16x16_i16_store::<STRIDE>(scratch, row_base + out_base, &packed);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn avx512_writeback_i32x16_u8(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    x: usize,
    y: usize,
    v: __m512i,
    rnd: i32,
    shift: i32,
) {
    debug_assert!(dst_off + y * dst_stride + x + 16 <= dst.len());
    let r = _mm512_sra_epi32(
        _mm512_add_epi32(v, _mm512_set1_epi32(rnd)),
        _mm_cvtsi32_si128(shift),
    );
    let residual = _mm512_cvtsepi32_epi16(r);
    unsafe {
        let ptr = dst.as_mut_ptr().add(dst_off + y * dst_stride + x);
        let src = _mm_loadu_si128(ptr.cast());
        let pix = _mm256_cvtepu8_epi16(src);
        let sum = _mm256_adds_epi16(pix, residual);
        let out = _mm_packus_epi16(
            _mm256_castsi256_si128(sum),
            _mm256_extracti128_si256::<1>(sum),
        );
        _mm_storeu_si128(ptr.cast(), out);
    }
}

macro_rules! avx512_dct16_body {
    ($z:expr, $load:ident, $madd_pair:ident, $add:ident, $sub:ident, $emit:ident) => {{
        let f0 = $madd_pair!(4, 12, &crate::itx_2d::DCT16_KFP_X4, 0);
        let f1 = $madd_pair!(4, 12, &crate::itx_2d::DCT16_KFP_X4, 1);
        let g0 = $madd_pair!(0, 8, &crate::itx_2d::DCT16_KGP_X4, 0);
        let g1 = $madd_pair!(0, 8, &crate::itx_2d::DCT16_KGP_X4, 1);

        let cc0 = $add!(g0, f0);
        let cc1 = $add!(g1, f1);
        let cc2 = $sub!(g1, f1);
        let cc3 = $sub!(g0, f0);
        let cc = [cc0, cc1, cc2, cc3];

        let mut d = [$z; 4];
        let mut m = 0usize;
        while m < 4 {
            let base = m * 8;
            d[m] = $add!(
                $madd_pair!(2, 6, &crate::itx_2d::DCT16_KDP_X4, base >> 1),
                $madd_pair!(10, 14, &crate::itx_2d::DCT16_KDP_X4, (base >> 1) + 1)
            );
            m += 1;
        }

        let mut b = [$z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = $z;
            acc = $add!(
                acc,
                $madd_pair!(1, 3, &crate::itx_2d::DCT16_KBP_X4, base >> 1)
            );
            acc = $add!(
                acc,
                $madd_pair!(5, 7, &crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 1)
            );
            acc = $add!(
                acc,
                $madd_pair!(9, 11, &crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 2)
            );
            b[m] = $add!(
                acc,
                $madd_pair!(13, 15, &crate::itx_2d::DCT16_KBP_X4, (base >> 1) + 3)
            );
            m += 1;
        }

        let mut k = 0usize;
        while k < 8 {
            let a = if k < 4 {
                $add!(cc[k], d[k])
            } else {
                $sub!(cc[7 - k], d[7 - k])
            };
            $emit!(k, $add!(a, b[k]));
            $emit!(8 + 7 - k, $sub!(a, b[k]));
            k += 1;
        }
    }};
}

macro_rules! avx512_dct32_body {
    ($z:expr, $load:ident, $madd_pair:ident, $add:ident, $sub:ident, $emit:ident) => {{
        let h0 = $madd_pair!(8, 24, &crate::itx_2d::DCT32_KHP_X4, 0);
        let h1 = $madd_pair!(8, 24, &crate::itx_2d::DCT32_KHP_X4, 1);
        let g0 = $madd_pair!(0, 16, &crate::itx_2d::DCT32_KGP_X4, 0);
        let g1 = $madd_pair!(0, 16, &crate::itx_2d::DCT32_KGP_X4, 1);

        let e0 = $add!(g0, h0);
        let e1 = $add!(g1, h1);
        let e2 = $sub!(g1, h1);
        let e3 = $sub!(g0, h0);
        let e = [e0, e1, e2, e3];

        let mut f = [$z; 4];
        let mut m = 0usize;
        while m < 4 {
            let base = m * 8;
            f[m] = $add!(
                $madd_pair!(4, 12, &crate::itx_2d::DCT32_KFP_X4, base >> 1),
                $madd_pair!(20, 28, &crate::itx_2d::DCT32_KFP_X4, (base >> 1) + 1)
            );
            m += 1;
        }

        let mut cc = [$z; 8];
        m = 0;
        while m < 8 {
            cc[m] = if m < 4 {
                $add!(e[m], f[m])
            } else {
                $sub!(e[7 - m], f[7 - m])
            };
            m += 1;
        }

        let mut d = [$z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = $z;
            let mut pair = 0usize;
            while pair < 4 {
                let i0 = 8 * pair + 2;
                acc = $add!(
                    acc,
                    $madd_pair!(i0, i0 + 4, &crate::itx_2d::DCT32_KDP_X4, (base >> 1) + pair)
                );
                pair += 1;
            }
            d[m] = acc;
            m += 1;
        }

        let mut a = [$z; 16];
        m = 0;
        while m < 16 {
            a[m] = if m < 8 {
                $add!(cc[m], d[m])
            } else {
                $sub!(cc[15 - m], d[15 - m])
            };
            m += 1;
        }

        let mut b = [$z; 16];
        m = 0;
        while m < 16 {
            let base = m * 16;
            let mut acc = $z;
            let mut pair = 0usize;
            while pair < 8 {
                let i0 = 4 * pair + 1;
                acc = $add!(
                    acc,
                    $madd_pair!(i0, i0 + 2, &crate::itx_2d::DCT32_KBP_X4, (base >> 1) + pair)
                );
                pair += 1;
            }
            b[m] = acc;
            m += 1;
        }

        let mut k = 0usize;
        while k < 16 {
            $emit!(k, $add!(a[k], b[k]));
            $emit!(16 + 15 - k, $sub!(a[k], b[k]));
            k += 1;
        }
    }};
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_dct16_rows_to_scratch<const IS_RECT2: bool>(
    coeff: &[i16],
    scratch: &mut [i16],
    y: usize,
    live: usize,
    rnd: i32,
    shift: i32,
    minv: i32,
    maxv: i32,
) {
    let z = _mm512_setzero_si512();
    let mut out = [z; 16];
    macro_rules! load {
        ($row:expr) => {
            avx512_load_coeff_i16x16::<IS_RECT2, 16>(coeff, y, live, $row)
        };
    }
    macro_rules! madd_pair {
        ($a:expr, $b:expr, $tbl:expr, $idx:expr) => {
            avx512_madd_pair_i16_i32(load!($a), load!($b), $tbl, $idx)
        };
    }
    macro_rules! add {
        ($a:expr, $b:expr) => {
            _mm512_add_epi32($a, $b)
        };
    }
    macro_rules! sub {
        ($a:expr, $b:expr) => {
            _mm512_sub_epi32($a, $b)
        };
    }
    macro_rules! emit {
        ($idx:expr, $v:expr) => {
            out[$idx] = $v
        };
    }
    avx512_dct16_body!(z, load, madd_pair, add, sub, emit);
    avx512_store_rowpass_i16x16::<16>(scratch, y * 16, 0, &out, rnd, shift, minv, maxv);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_dct32_rows_to_scratch<const IS_RECT2: bool>(
    coeff: &[i16],
    scratch: &mut [i16],
    y: usize,
    live: usize,
    rnd: i32,
    shift: i32,
    minv: i32,
    maxv: i32,
) {
    let z = _mm512_setzero_si512();
    let mut out = [z; 32];
    macro_rules! load {
        ($row:expr) => {
            avx512_load_coeff_i16x16::<IS_RECT2, 32>(coeff, y, live, $row)
        };
    }
    macro_rules! madd_pair {
        ($a:expr, $b:expr, $tbl:expr, $idx:expr) => {
            avx512_madd_pair_i16_i32(load!($a), load!($b), $tbl, $idx)
        };
    }
    macro_rules! add {
        ($a:expr, $b:expr) => {
            _mm512_add_epi32($a, $b)
        };
    }
    macro_rules! sub {
        ($a:expr, $b:expr) => {
            _mm512_sub_epi32($a, $b)
        };
    }
    macro_rules! emit {
        ($idx:expr, $v:expr) => {
            out[$idx] = $v
        };
    }
    avx512_dct32_body!(z, load, madd_pair, add, sub, emit);

    let lo = [
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10],
        out[11], out[12], out[13], out[14], out[15],
    ];
    let hi = [
        out[16], out[17], out[18], out[19], out[20], out[21], out[22], out[23], out[24], out[25],
        out[26], out[27], out[28], out[29], out[30], out[31],
    ];
    avx512_store_rowpass_i16x16::<32>(scratch, y * 32, 0, &lo, rnd, shift, minv, maxv);
    avx512_store_rowpass_i16x16::<32>(scratch, y * 32, 16, &hi, rnd, shift, minv, maxv);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_dct16_scratch_to_dst(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    rnd: i32,
    shift: i32,
) {
    let z = _mm512_setzero_si512();
    macro_rules! load {
        ($row:expr) => {
            avx512_load_scratch_i16x16::<16>(scratch, base, active, $row)
        };
    }
    macro_rules! madd_pair {
        ($a:expr, $b:expr, $tbl:expr, $idx:expr) => {
            avx512_madd_pair_i16_i32(load!($a), load!($b), $tbl, $idx)
        };
    }
    macro_rules! add {
        ($a:expr, $b:expr) => {
            _mm512_add_epi32($a, $b)
        };
    }
    macro_rules! sub {
        ($a:expr, $b:expr) => {
            _mm512_sub_epi32($a, $b)
        };
    }
    macro_rules! emit {
        ($out:expr, $v:expr) => {
            avx512_writeback_i32x16_u8(dst, dst_off, dst_stride, base, $out, $v, rnd, shift)
        };
    }
    avx512_dct16_body!(z, load, madd_pair, add, sub, emit);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_dct32_scratch_to_dst(
    scratch: &[i16],
    base: usize,
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    rnd: i32,
    shift: i32,
) {
    let z = _mm512_setzero_si512();
    macro_rules! load {
        ($row:expr) => {
            avx512_load_scratch_i16x16::<32>(scratch, base, active, $row)
        };
    }
    macro_rules! madd_pair {
        ($a:expr, $b:expr, $tbl:expr, $idx:expr) => {
            avx512_madd_pair_i16_i32(load!($a), load!($b), $tbl, $idx)
        };
    }
    macro_rules! add {
        ($a:expr, $b:expr) => {
            _mm512_add_epi32($a, $b)
        };
    }
    macro_rules! sub {
        ($a:expr, $b:expr) => {
            _mm512_sub_epi32($a, $b)
        };
    }
    macro_rules! emit {
        ($out:expr, $v:expr) => {
            avx512_writeback_i32x16_u8(dst, dst_off, dst_stride, base, $out, $v, rnd, shift)
        };
    }
    avx512_dct32_body!(z, load, madd_pair, add, sub, emit);
}

#[inline(always)]
fn avx512_active_cols(eob: i32, tx: usize, n: usize) -> usize {
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < n / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    ngrp * 4
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn idct_dequant_dct_i16_avx512_fused_8bpc_impl_const<const N: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    debug_assert!(N == 16 || N == 32);
    debug_assert!(coeff.len() >= N * N);
    let ncols = avx512_active_cols(eob, tx, N);
    let rnd0 = (1 << shift0) >> 1;

    with_avx512_itx_i16_scratch(N * N, |scratch| {
        let mut y = 0usize;
        while y < ncols {
            let live = (ncols - y).min(16);
            if N == 16 {
                avx512_dct16_rows_to_scratch::<IS_RECT2>(
                    coeff,
                    scratch,
                    y,
                    live,
                    rnd0,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                );
            } else {
                avx512_dct32_rows_to_scratch::<IS_RECT2>(
                    coeff,
                    scratch,
                    y,
                    live,
                    rnd0,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                );
            }
            y += 16;
        }

        crate::itx_2d::clear_i16_coeff_active_rows(coeff, N, ncols);

        let rnd1 = (1 << shift1) >> 1;
        let mut x = 0usize;
        while x < N {
            if N == 16 {
                avx512_dct16_scratch_to_dst(
                    scratch, x, ncols, dst, dst_off, dst_stride, rnd1, shift1,
                );
            } else {
                avx512_dct32_scratch_to_dst(
                    scratch, x, ncols, dst, dst_off, dst_stride, rnd1, shift1,
                );
            }
            x += 16;
        }
    });
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn idct_dequant_dct_i16_avx512_fused_8bpc_impl<const N: usize>(
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
        idct_dequant_dct_i16_avx512_fused_8bpc_impl_const::<N, true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        idct_dequant_dct_i16_avx512_fused_8bpc_impl_const::<N, false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[inline(always)]
fn avx512_tx16_coeff(kind: usize, out: usize, input: usize) -> i32 {
    match kind {
        crate::itx_2d::TX_KIND_DCT => crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out],
        crate::itx_2d::TX_KIND_ADST => crate::itx_1d::ADST16_KERNEL_ROWS[out][input] as i32,
        crate::itx_2d::TX_KIND_FLIPADST => crate::itx_1d::FLIPADST16_KERNEL_ROWS[out][input] as i32,
        _ => unreachable!(),
    }
}

#[inline(always)]
fn avx512_tx16_supported(kind: usize) -> bool {
    matches!(
        kind,
        crate::itx_2d::TX_KIND_DCT | crate::itx_2d::TX_KIND_ADST | crate::itx_2d::TX_KIND_FLIPADST
    )
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_tx16_dense_rows_to_scratch<const IS_RECT2: bool>(
    coeff: &[i16],
    scratch: &mut [i16],
    y: usize,
    live: usize,
    kind: usize,
    rnd: i32,
    shift: i32,
    minv: i32,
    maxv: i32,
) {
    let z = _mm512_setzero_si512();
    let mut rows = [z; 16];
    let mut out = 0usize;
    while out < 16 {
        let mut acc = z;
        let mut input = 0usize;
        while input < 16 {
            let a = avx512_load_coeff_i16x16::<IS_RECT2, 16>(coeff, y, live, input);
            let b = avx512_load_coeff_i16x16::<IS_RECT2, 16>(coeff, y, live, input + 1);
            let k0 = avx512_tx16_coeff(kind, out, input) as i16;
            let k1 = avx512_tx16_coeff(kind, out, input + 1) as i16;
            acc = _mm512_dpwssds_epi32(
                acc,
                avx512_i16_interleave_x16(a, b),
                avx512_coeff_pair_512(k0, k1),
            );
            input += 2;
        }
        rows[out] = acc;
        out += 1;
    }
    avx512_store_rowpass_i16x16::<16>(scratch, y * 16, 0, &rows, rnd, shift, minv, maxv);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn avx512_tx16_dense_scratch_to_dst(
    scratch: &[i16],
    active: usize,
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    kind: usize,
    rnd: i32,
    shift: i32,
) {
    let z = _mm512_setzero_si512();
    let mut out = 0usize;
    while out < 16 {
        let mut acc = z;
        let mut input = 0usize;
        while input < 16 {
            let a = avx512_load_scratch_i16x16::<16>(scratch, 0, active, input);
            let b = avx512_load_scratch_i16x16::<16>(scratch, 0, active, input + 1);
            let k0 = avx512_tx16_coeff(kind, out, input) as i16;
            let k1 = avx512_tx16_coeff(kind, out, input + 1) as i16;
            acc = _mm512_dpwssds_epi32(
                acc,
                avx512_i16_interleave_x16(a, b),
                avx512_coeff_pair_512(k0, k1),
            );
            input += 2;
        }
        avx512_writeback_i32x16_u8(dst, dst_off, dst_stride, 0, out, acc, rnd, shift);
        out += 1;
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn tx16_dequant_dense_i16_avx512_fused_8bpc_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(coeff.len() >= 256);
    let ncols = avx512_active_cols(eob, tx, 16);
    let rnd0 = (1 << shift0) >> 1;

    with_avx512_itx_i16_scratch(256, |scratch| {
        let mut y = 0usize;
        while y < ncols {
            let live = (ncols - y).min(16);
            avx512_tx16_dense_rows_to_scratch::<IS_RECT2>(
                coeff,
                scratch,
                y,
                live,
                first_kind,
                rnd0,
                shift0,
                row_clip_min,
                row_clip_max,
            );
            y += 16;
        }

        crate::itx_2d::clear_i16_coeff_active_rows(coeff, 16, ncols);
        avx512_tx16_dense_scratch_to_dst(
            scratch,
            ncols,
            dst,
            dst_off,
            dst_stride,
            second_kind,
            (1 << shift1) >> 1,
            shift1,
        );
    });
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn tx16_dequant_dense_i16_avx512_fused_8bpc_impl(
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
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx16_dequant_dense_i16_avx512_fused_8bpc_impl_const::<true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    } else {
        tx16_dequant_dense_i16_avx512_fused_8bpc_impl_const::<false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn idct_dequant_16x16_i16_avx512_fused_8bpc(
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
    idct_dequant_dct_i16_avx512_fused_8bpc_impl::<16>(
        coeff,
        dst,
        dst_off,
        dst_stride,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn idct_dequant_32x32_i16_avx512_fused_8bpc(
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
    idct_dequant_dct_i16_avx512_fused_8bpc_impl::<32>(
        coeff,
        dst,
        dst_off,
        dst_stride,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn itx_dequant_i16_avx512_fused_8bpc(
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
    if tx != crate::levels::txsz::TX_16X16 || out_w != 16 || out_h != 16 {
        return false;
    }
    if !avx512_tx16_supported(first_kind) || !avx512_tx16_supported(second_kind) {
        return false;
    }
    if first_kind == crate::itx_2d::TX_KIND_DCT && second_kind == crate::itx_2d::TX_KIND_DCT {
        idct_dequant_dct_i16_avx512_fused_8bpc_impl::<16>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        );
    } else {
        tx16_dequant_dense_i16_avx512_fused_8bpc_impl(
            coeff,
            dst,
            dst_off,
            dst_stride,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        );
    }
    true
}
