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

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x16_i16(src: &[u8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x8(src: &[i16]) -> __m128i {
    unsafe { _mm_loadu_si128(src.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16(dst: &mut [i16], v: __m256i) {
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8(dst: &mut [i16], v: __m256i) {
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packs_epi32(lo, hi));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x16_from_i16(dst: &mut [u8], v: __m256i) {
    let p8 = _mm256_packus_epi16(v, v);
    let lo = _mm256_castsi256_si128(p8);
    let hi = _mm256_extracti128_si256::<1>(p8);
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_unpacklo_epi64(lo, hi));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x16_round4_from_i16(dst: &mut [u8], v: __m256i) {
    let v = _mm256_srli_epi16::<4>(_mm256_add_epi16(v, _mm256_set1_epi16(8)));
    store_u8x16_from_i16(dst, v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x8_round8_from_i32(dst: &mut [u8], v: __m256i) {
    let v = _mm256_srai_epi32::<8>(_mm256_add_epi32(v, _mm256_set1_epi32(128)));
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p16 = _mm_packus_epi32(lo, hi);
    let p8 = _mm_packus_epi16(p16, p16);
    unsafe {
        _mm_storel_epi64(dst.as_mut_ptr().cast(), p8);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x16_round8_from_i32(dst: &mut [u8], lo: __m256i, hi: __m256i) {
    let lo = _mm256_srai_epi32::<8>(_mm256_add_epi32(lo, _mm256_set1_epi32(128)));
    let hi = _mm256_srai_epi32::<8>(_mm256_add_epi32(hi, _mm256_set1_epi32(128)));
    let lo0 = _mm256_castsi256_si128(lo);
    let lo1 = _mm256_extracti128_si256::<1>(lo);
    let hi0 = _mm256_castsi256_si128(hi);
    let hi1 = _mm256_extracti128_si256::<1>(hi);
    let p16_lo = _mm_packus_epi32(lo0, lo1);
    let p16_hi = _mm_packus_epi32(hi0, hi1);
    let p8 = _mm_packus_epi16(p16_lo, p16_hi);
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), p8) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8_round4_from_i32(dst: &mut [i16], v: __m256i) {
    let v = _mm256_srai_epi32::<4>(_mm256_add_epi32(v, _mm256_set1_epi32(8)));
    store_i16x8(dst, v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16_round4_from_i32(dst: &mut [i16], lo: __m256i, hi: __m256i) {
    let lo = _mm256_srai_epi32::<4>(_mm256_add_epi32(lo, _mm256_set1_epi32(8)));
    let hi = _mm256_srai_epi32::<4>(_mm256_add_epi32(hi, _mm256_set1_epi32(8)));
    let lo0 = _mm256_castsi256_si128(lo);
    let lo1 = _mm256_extracti128_si256::<1>(lo);
    let hi0 = _mm256_castsi256_si128(hi);
    let hi1 = _mm256_extracti128_si256::<1>(hi);
    unsafe {
        _mm256_storeu_si256(
            dst.as_mut_ptr().cast(),
            _mm256_setr_m128i(_mm_packs_epi32(lo0, lo1), _mm_packs_epi32(hi0, hi1)),
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_coeff_u8(mxy: i32) -> __m256i {
    _mm256_set1_epi16(((mxy as i16) << 8) | ((16 - mxy) as i16))
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_coeff_i16(mxy: i32) -> __m256i {
    _mm256_set1_epi32(((mxy as i16 as u16 as i32) << 16) | ((16 - mxy) as i16 as u16 as i32))
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_u8x16_i16(src: &[u8], base: usize, stride: usize, mxy: i32) -> __m256i {
    unsafe {
        let a = _mm_loadu_si128(src.get_unchecked(base..).as_ptr().cast());
        let b = _mm_loadu_si128(src.get_unchecked(base + stride..).as_ptr().cast());
        let c = _mm256_castsi256_si128(bilin_coeff_u8(mxy));
        let lo = _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c);
        let hi = _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), c);
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_i16x8_i32(a16: __m128i, b16: __m128i, mxy: i32) -> __m256i {
    let lo = _mm_unpacklo_epi16(a16, b16);
    let hi = _mm_unpackhi_epi16(a16, b16);
    let pairs = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
    _mm256_madd_epi16(pairs, bilin_coeff_i16(mxy))
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_i16x16_i32(
    a0: __m128i,
    b0: __m128i,
    a1: __m128i,
    b1: __m128i,
    mxy: i32,
) -> (__m256i, __m256i) {
    (bilin_i16x8_i32(a0, b0, mxy), bilin_i16x8_i32(a1, b1, mxy))
}

#[inline(always)]
fn bilin_scalar(a: i32, b: i32, mxy: i32) -> i32 {
    16 * a + mxy * (b - a)
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    mid_scratch: &mut [i16],
) {
    if mx != 0 && my != 0 {
        let mid_stride = w.next_multiple_of(16).max(64);
        let mid = &mut mid_scratch[..mid_stride * (h + 1)];
        for (y, mid_row) in mid.chunks_mut(mid_stride).take(h + 1).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                store_i16x16(mid_chunk, bilin_u8x16_i16(src_row, x, 1, mx));
            }
            let processed = mid_chunks.len() * 16;
            for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                let si = x;
                *mid_px = bilin_scalar(src_row[si] as i32, src_row[si + 1] as i32, mx) as i16;
            }
        }
        for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
            let mid_row = unsafe { mid.get_unchecked(y * mid_stride..) };
            let mid_next_row = unsafe { mid.get_unchecked((y + 1) * mid_stride..) };
            let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let a0 = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b0 = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                let a1 = load_i16x8(unsafe { mid_row.get_unchecked(x + 8..) });
                let b1 = load_i16x8(unsafe { mid_next_row.get_unchecked(x + 8..) });
                let (lo, hi) = bilin_i16x16_i32(a0, b0, a1, b1, my);
                store_u8x16_round8_from_i32(dst_chunk, lo, hi);
            }
            let x16_done = dst_chunks16.len() * 16;
            let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
            for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                store_u8x8_round8_from_i32(dst_chunk, bilin_i16x8_i32(a, b, my));
            }
            let processed = x16_done + dst_chunks8.len() * 8;
            for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                *dst_px = ((bilin_scalar(mid_row[x] as i32, mid_next_row[x] as i32, my) + 128) >> 8)
                    .clamp(0, 255) as u8;
            }
        }
    } else if mx != 0 {
        for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                store_u8x16_round4_from_i16(dst_chunk, bilin_u8x16_i16(src_row, x, 1, mx));
            }
            let processed = dst_chunks.len() * 16;
            for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                let si = x;
                *dst_px =
                    ((bilin_scalar(src_row[si] as i32, src_row[si + 1] as i32, mx) + 8) >> 4) as u8;
            }
        }
    } else if my != 0 {
        for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                store_u8x16_round4_from_i16(dst_chunk, bilin_u8x16_i16(src_row, x, src_stride, my));
            }
            let processed = dst_chunks.len() * 16;
            for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                let si = x;
                *dst_px = ((bilin_scalar(src_row[si] as i32, src_row[si + src_stride] as i32, my)
                    + 8)
                    >> 4) as u8;
            }
        }
    } else {
        for (src_row, dst_row) in src
            .chunks(src_stride)
            .zip(dst.chunks_mut(dst_stride))
            .take(h)
        {
            dst_row[..w].copy_from_slice(&src_row[..w]);
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin_8bpc_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    mid_scratch: &mut [i16],
) {
    if mx != 0 && my != 0 {
        let mid_stride = w.next_multiple_of(16).max(64);
        let mid = &mut mid_scratch[..mid_stride * (h + 1)];
        for (y, mid_row) in mid.chunks_mut(mid_stride).take(h + 1).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                store_i16x16(mid_chunk, bilin_u8x16_i16(src_row, x, 1, mx));
            }
            let processed = mid_chunks.len() * 16;
            for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                let si = x;
                *mid_px = bilin_scalar(src_row[si] as i32, src_row[si + 1] as i32, mx) as i16;
            }
        }
        for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
            let mid_row = unsafe { mid.get_unchecked(y * mid_stride..) };
            let mid_next_row = unsafe { mid.get_unchecked((y + 1) * mid_stride..) };
            let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let a0 = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b0 = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                let a1 = load_i16x8(unsafe { mid_row.get_unchecked(x + 8..) });
                let b1 = load_i16x8(unsafe { mid_next_row.get_unchecked(x + 8..) });
                let (lo, hi) = bilin_i16x16_i32(a0, b0, a1, b1, my);
                store_i16x16_round4_from_i32(tmp_chunk, lo, hi);
            }
            let x16_done = tmp_chunks16.len() * 16;
            let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
            for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                store_i16x8_round4_from_i32(tmp_chunk, bilin_i16x8_i32(a, b, my));
            }
            let processed = x16_done + tmp_chunks8.len() * 8;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                *tmp_px =
                    ((bilin_scalar(mid_row[x] as i32, mid_next_row[x] as i32, my) + 8) >> 4) as i16;
            }
        }
    } else if mx != 0 {
        for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                store_i16x16(tmp_chunk, bilin_u8x16_i16(src_row, x, 1, mx));
            }
            let processed = tmp_chunks.len() * 16;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                let si = x;
                *tmp_px = bilin_scalar(src_row[si] as i32, src_row[si + 1] as i32, mx) as i16;
            }
        }
    } else if my != 0 {
        for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                store_i16x16(tmp_chunk, bilin_u8x16_i16(src_row, x, src_stride, my));
            }
            let processed = tmp_chunks.len() * 16;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                let si = x;
                *tmp_px =
                    bilin_scalar(src_row[si] as i32, src_row[si + src_stride] as i32, my) as i16;
            }
        }
    } else {
        for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let v =
                    _mm256_slli_epi16::<4>(load_u8x16_i16(unsafe { src_row.get_unchecked(x..) }));
                store_i16x16(tmp_chunk, v);
            }
            let processed = tmp_chunks.len() * 16;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                *tmp_px = (src_row[x] as i16) << 4;
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_i16x8_pair_s32(a: __m128i, b: __m128i, c0: i16, c1: i16) -> __m256i {
    let lo = _mm_unpacklo_epi16(a, b);
    let hi = _mm_unpackhi_epi16(a, b);
    let pairs = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
    let coeff = _mm256_set1_epi32(((c1 as u16 as i32) << 16) | c0 as u16 as i32);
    _mm256_madd_epi16(pairs, coeff)
}

#[inline(always)]
fn pair_coeff(a: i8, b: i8) -> i16 {
    (((b as i16 as u16) << 8) | ((a as i16 as u16) & 0xff)) as i16
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_u8x8_h(src: &[u8], base: usize, f: &[i8; 8]) -> __m256i {
    unsafe {
        let s = _mm_loadu_si128(src.get_unchecked(base - 3..).as_ptr().cast());
        let shuf01 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
        let shuf23 = _mm_setr_epi8(2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10);
        let shuf45 = _mm_setr_epi8(4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12);
        let shuf67 = _mm_setr_epi8(6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14);
        let p01 = _mm_maddubs_epi16(
            _mm_shuffle_epi8(s, shuf01),
            _mm_set1_epi16(pair_coeff(f[0], f[1])),
        );
        let p23 = _mm_maddubs_epi16(
            _mm_shuffle_epi8(s, shuf23),
            _mm_set1_epi16(pair_coeff(f[2], f[3])),
        );
        let p45 = _mm_maddubs_epi16(
            _mm_shuffle_epi8(s, shuf45),
            _mm_set1_epi16(pair_coeff(f[4], f[5])),
        );
        let p67 = _mm_maddubs_epi16(
            _mm_shuffle_epi8(s, shuf67),
            _mm_set1_epi16(pair_coeff(f[6], f[7])),
        );
        let sum = _mm_add_epi16(_mm_add_epi16(p01, p23), _mm_add_epi16(p45, p67));
        _mm256_cvtepi16_epi32(sum)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::neg_multiply)]
fn filter_u8x8_v(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    unsafe {
        let mut sum = _mm_setzero_si128();
        macro_rules! add_pair {
            ($a:literal, $b:literal, $oa:expr, $ob:expr) => {{
                if f[$a] != 0 || f[$b] != 0 {
                    let ia = (base as isize + $oa * stride) as usize;
                    let ib = (base as isize + $ob * stride) as usize;
                    let a = _mm_loadl_epi64(src.get_unchecked(ia..).as_ptr().cast());
                    let b = _mm_loadl_epi64(src.get_unchecked(ib..).as_ptr().cast());
                    let pairs = _mm_unpacklo_epi8(a, b);
                    let coeff = _mm_set1_epi16(pair_coeff(f[$a], f[$b]));
                    sum = _mm_add_epi16(sum, _mm_maddubs_epi16(pairs, coeff));
                }
            }};
        }
        add_pair!(0, 1, -3isize, -2isize);
        add_pair!(2, 3, -1isize, 0isize);
        add_pair!(4, 5, 1isize, 2isize);
        add_pair!(6, 7, 3isize, 4isize);
        _mm256_cvtepi16_epi32(sum)
    }
}

/// Whether 8-bit pixels can overflow the 16-bit `maddubs` accumulation: the
/// 64-scale MC filters cannot, the 128-scale `EXT_WARP_FILTER` (opfl/TIP) can.
#[inline(always)]
fn needs_wide(f: &[i8; 8]) -> bool {
    255 * f.iter().map(|&c| (c as i32).max(0)).sum::<i32>() > i16::MAX as i32
}

/// 8 pixels with 32-bit accumulation, for filters where `needs_wide` holds.
#[inline]
#[target_feature(enable = "avx2")]
fn filter_u8x8_wide(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = _mm256_setzero_si256();
    for k in (0..8).step_by(2) {
        let (c0, c1) = (f[k], f[k + 1]);
        if c0 == 0 && c1 == 0 {
            continue;
        }
        let idx0 = (base as isize + OFFSETS[k] * stride) as usize;
        let idx1 = (base as isize + OFFSETS[k + 1] * stride) as usize;
        let (a, b) = unsafe {
            (
                _mm_cvtepu8_epi16(_mm_loadl_epi64(src.get_unchecked(idx0..).as_ptr().cast())),
                _mm_cvtepu8_epi16(_mm_loadl_epi64(src.get_unchecked(idx1..).as_ptr().cast())),
            )
        };
        sum = _mm256_add_epi32(sum, madd_i16x8_pair_s32(a, b, c0 as i16, c1 as i16));
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_u8x8(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    if needs_wide(f) {
        filter_u8x8_wide(src, base, stride, f)
    } else if stride == 1 {
        filter_u8x8_h(src, base, f)
    } else {
        filter_u8x8_v(src, base, stride, f)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::neg_multiply)]
fn filter_u8x16_v(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> (__m256i, __m256i) {
    unsafe {
        let mut lo = _mm_setzero_si128();
        let mut hi = _mm_setzero_si128();
        macro_rules! add_pair {
            ($a:literal, $b:literal, $oa:expr, $ob:expr) => {{
                if f[$a] != 0 || f[$b] != 0 {
                    let ia = (base as isize + $oa * stride) as usize;
                    let ib = (base as isize + $ob * stride) as usize;
                    let a = _mm_loadu_si128(src.get_unchecked(ia..).as_ptr().cast());
                    let b = _mm_loadu_si128(src.get_unchecked(ib..).as_ptr().cast());
                    let coeff = _mm_set1_epi16(pair_coeff(f[$a], f[$b]));
                    lo = _mm_add_epi16(lo, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), coeff));
                    hi = _mm_add_epi16(hi, _mm_maddubs_epi16(_mm_unpackhi_epi8(a, b), coeff));
                }
            }};
        }
        add_pair!(0, 1, -3isize, -2isize);
        add_pair!(2, 3, -1isize, 0isize);
        add_pair!(4, 5, 1isize, 2isize);
        add_pair!(6, 7, 3isize, 4isize);
        (_mm256_cvtepi16_epi32(lo), _mm256_cvtepi16_epi32(hi))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_u8x16(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> (__m256i, __m256i) {
    if needs_wide(f) {
        (
            filter_u8x8_wide(src, base, stride, f),
            filter_u8x8_wide(src, base + 8, stride, f),
        )
    } else if stride == 1 {
        (filter_u8x8_h(src, base, f), filter_u8x8_h(src, base + 8, f))
    } else {
        filter_u8x16_v(src, base, stride, f)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_i16x8_8tap(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = _mm256_setzero_si256();
    for k in (0..8).step_by(2) {
        let c0 = f[k];
        let c1 = f[k + 1];
        if c0 == 0 && c1 == 0 {
            continue;
        }
        let idx0 = (base as isize + OFFSETS[k] * stride) as usize;
        let idx1 = (base as isize + OFFSETS[k + 1] * stride) as usize;
        let a = load_i16x8(unsafe { src.get_unchecked(idx0..) });
        let b = load_i16x8(unsafe { src.get_unchecked(idx1..) });
        sum = _mm256_add_epi32(sum, madd_i16x8_pair_s32(a, b, c0 as i16, c1 as i16));
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_i16x16_8tap(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> (__m256i, __m256i) {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo = _mm256_setzero_si256();
    let mut hi = _mm256_setzero_si256();
    for k in (0..8).step_by(2) {
        let c0 = f[k];
        let c1 = f[k + 1];
        if c0 == 0 && c1 == 0 {
            continue;
        }
        let idx0 = (base as isize + OFFSETS[k] * stride) as usize;
        let idx1 = (base as isize + OFFSETS[k + 1] * stride) as usize;
        let a0 = load_i16x8(unsafe { src.get_unchecked(idx0..) });
        let b0 = load_i16x8(unsafe { src.get_unchecked(idx1..) });
        let a1 = load_i16x8(unsafe { src.get_unchecked(idx0 + 8..) });
        let b1 = load_i16x8(unsafe { src.get_unchecked(idx1 + 8..) });
        lo = _mm256_add_epi32(lo, madd_i16x8_pair_s32(a0, b0, c0 as i16, c1 as i16));
        hi = _mm256_add_epi32(hi, madd_i16x8_pair_s32(a1, b1, c0 as i16, c1 as i16));
    }
    (lo, hi)
}

#[inline(always)]
fn filter_u8_scalar(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> i32 {
    let c = base as isize;
    f[0] as i32 * src[(c - 3 * stride) as usize] as i32
        + f[1] as i32 * src[(c - 2 * stride) as usize] as i32
        + f[2] as i32 * src[(c - stride) as usize] as i32
        + f[3] as i32 * src[base] as i32
        + f[4] as i32 * src[(c + stride) as usize] as i32
        + f[5] as i32 * src[(c + 2 * stride) as usize] as i32
        + f[6] as i32 * src[(c + 3 * stride) as usize] as i32
        + f[7] as i32 * src[(c + 4 * stride) as usize] as i32
}

#[inline(always)]
fn filter_i16_scalar(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> i32 {
    let c = base as isize;
    f[0] as i32 * src[(c - 3 * stride) as usize] as i32
        + f[1] as i32 * src[(c - 2 * stride) as usize] as i32
        + f[2] as i32 * src[(c - stride) as usize] as i32
        + f[3] as i32 * src[base] as i32
        + f[4] as i32 * src[(c + stride) as usize] as i32
        + f[5] as i32 * src[(c + 2 * stride) as usize] as i32
        + f[6] as i32 * src[(c + 3 * stride) as usize] as i32
        + f[7] as i32 * src[(c + 4 * stride) as usize] as i32
}

#[inline(always)]
fn round_scalar(v: i32, rnd: i32, shift: i32) -> i32 {
    if shift == 0 {
        v + rnd
    } else {
        (v + rnd) >> shift
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_s32(v: __m256i, rnd: i32, shift: i32) -> __m256i {
    _mm256_sra_epi32(
        _mm256_add_epi32(v, _mm256_set1_epi32(rnd)),
        _mm_cvtsi32_si128(shift),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x8_clip_shift(dst: &mut [u8], v: __m256i, rnd: i32, shift: i32) {
    unsafe {
        let v = round_s32(v, rnd, shift);
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let p16 = _mm_packus_epi32(lo, hi);
        let p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(dst.as_mut_ptr().cast(), p8);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8_shift(dst: &mut [i16], v: __m256i, rnd: i32, shift: i32) {
    let v = round_s32(v, rnd, shift);
    store_i16x8(dst, v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x16_clip_shift(dst: &mut [u8], lo: __m256i, hi: __m256i, rnd: i32, shift: i32) {
    unsafe {
        let lo = round_s32(lo, rnd, shift);
        let hi = round_s32(hi, rnd, shift);
        let lo0 = _mm256_castsi256_si128(lo);
        let lo1 = _mm256_extracti128_si256::<1>(lo);
        let hi0 = _mm256_castsi256_si128(hi);
        let hi1 = _mm256_extracti128_si256::<1>(hi);
        let lo16 = _mm_packus_epi32(lo0, lo1);
        let hi16 = _mm_packus_epi32(hi0, hi1);
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packus_epi16(lo16, hi16));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16_shift(dst: &mut [i16], lo: __m256i, hi: __m256i, rnd: i32, shift: i32) {
    unsafe {
        let lo = round_s32(lo, rnd, shift);
        let hi = round_s32(hi, rnd, shift);
        let lo0 = _mm256_castsi256_si128(lo);
        let lo1 = _mm256_extracti128_si256::<1>(lo);
        let hi0 = _mm256_castsi256_si128(hi);
        let hi1 = _mm256_extracti128_si256::<1>(hi);
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packs_epi32(lo0, lo1));
        _mm_storeu_si128(dst.as_mut_ptr().add(8).cast(), _mm_packs_epi32(hi0, hi1));
    }
}

/// Output sample of the 16-bit 8-tap paths: rounded words in, saturated to
/// the sample type on store.
trait TapOut: Copy + Default {
    fn from_i32(v: i32) -> Self;
    /// Stores 16 words; `dst` must be valid for 16 samples.
    unsafe fn store16(dst: *mut Self, v: __m256i);
    /// Stores 8 words; `dst` must be valid for 8 samples.
    unsafe fn store8(dst: *mut Self, v: __m128i);
}

impl TapOut for i16 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as i16
    }
    #[inline(always)]
    unsafe fn store16(dst: *mut Self, v: __m256i) {
        unsafe { _mm256_storeu_si256(dst.cast(), v) }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, v: __m128i) {
        unsafe { _mm_storeu_si128(dst.cast(), v) }
    }
}

impl TapOut for u8 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v.clamp(0, 255) as u8
    }
    #[inline(always)]
    unsafe fn store16(dst: *mut Self, v: __m256i) {
        unsafe {
            let p = _mm256_permute4x64_epi64::<0b10_00>(_mm256_packus_epi16(v, v));
            _mm_storeu_si128(dst.cast(), _mm256_castsi256_si128(p))
        }
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, v: __m128i) {
        unsafe { _mm_storel_epi64(dst.cast(), _mm_packus_epi16(v, v)) }
    }
}

/// Stores the first `dst.len()` (< 8) of 8 words.
#[inline(always)]
fn store_tail<O: TapOut>(dst: &mut [O], v: __m128i) {
    let mut t = [O::default(); 8];
    unsafe { O::store8(t.as_mut_ptr(), v) };
    dst.copy_from_slice(&t[..dst.len()]);
}

/// Whether the 16-bit `maddubs` accumulation of 8-bit pixels plus the
/// rounding offset stays inside i16 for every partial sum.
#[inline(always)]
fn fits_i16(f: &[i8; 8], rnd: i32) -> bool {
    let pos: i32 = f.iter().map(|&c| (c as i32).max(0)).sum();
    let neg: i32 = f.iter().map(|&c| (c as i32).min(0)).sum();
    255 * pos + rnd <= i16::MAX as i32 && 255 * neg >= i16::MIN as i32
}

/// The four tap pairs as `maddubs` byte-pair coefficients.
#[inline(always)]
fn pair_coeffs_u8(f: &[i8; 8]) -> [__m256i; 4] {
    std::array::from_fn(|k| unsafe { _mm256_set1_epi16(pair_coeff(f[2 * k], f[2 * k + 1])) })
}

/// The four tap pairs as `madd` word-pair coefficients.
#[inline(always)]
fn pair_coeffs_i16(f: &[i8; 8]) -> [__m256i; 4] {
    std::array::from_fn(|k| unsafe {
        _mm256_set1_epi32(((f[2 * k + 1] as u16 as i32) << 16) | f[2 * k] as u16 as i32)
    })
}

/// Unrounded horizontal sums of 16 pixels; `p` addresses the pixel 3 left of
/// the first one and 24 bytes are read.
#[inline(always)]
unsafe fn h16_s16(p: *const u8, c: &[__m256i; 4]) -> __m256i {
    unsafe {
        let s = _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(p.cast())),
            _mm_loadu_si128(p.add(8).cast()),
        );
        let shuf01 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
        ));
        let two = _mm256_set1_epi8(2);
        let shuf23 = _mm256_add_epi8(shuf01, two);
        let shuf45 = _mm256_add_epi8(shuf23, two);
        let shuf67 = _mm256_add_epi8(shuf45, two);
        let p01 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, shuf01), c[0]);
        let p23 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, shuf23), c[1]);
        let p45 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, shuf45), c[2]);
        let p67 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(s, shuf67), c[3]);
        _mm256_add_epi16(_mm256_add_epi16(p01, p23), _mm256_add_epi16(p45, p67))
    }
}

/// Unrounded horizontal sums of 8 pixels; 16 bytes are read at `p`.
#[inline(always)]
unsafe fn h8_s16(p: *const u8, c: &[__m256i; 4]) -> __m128i {
    unsafe {
        let s = _mm_loadu_si128(p.cast());
        let shuf01 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
        let two = _mm_set1_epi8(2);
        let shuf23 = _mm_add_epi8(shuf01, two);
        let shuf45 = _mm_add_epi8(shuf23, two);
        let shuf67 = _mm_add_epi8(shuf45, two);
        let c = c.map(|c| _mm256_castsi256_si128(c));
        let p01 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, shuf01), c[0]);
        let p23 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, shuf23), c[1]);
        let p45 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, shuf45), c[2]);
        let p67 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, shuf67), c[3]);
        _mm_add_epi16(_mm_add_epi16(p01, p23), _mm_add_epi16(p45, p67))
    }
}

/// Unrounded vertical sums of 16 pixels; `p` addresses the centre row and
/// rows -3..=4 are read, 16 bytes each. Pairs with zero taps are skipped.
#[inline(always)]
unsafe fn v16_s16(p: *const u8, stride: isize, c: &[__m256i; 4], on: [bool; 4]) -> __m256i {
    unsafe {
        let mut sum = _mm256_setzero_si256();
        for k in 0..4 {
            if on[k] {
                let r = p.offset((2 * k as isize - 3) * stride);
                let a = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.cast()));
                let b = _mm256_cvtepu8_epi16(_mm_loadu_si128(r.offset(stride).cast()));
                let pairs = _mm256_or_si256(a, _mm256_slli_epi16::<8>(b));
                sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(pairs, c[k]));
            }
        }
        sum
    }
}

/// As `v16_s16` for 8 pixels, reading 8 bytes per row.
#[inline(always)]
unsafe fn v8_s16(p: *const u8, stride: isize, c: &[__m256i; 4], on: [bool; 4]) -> __m128i {
    unsafe {
        let mut sum = _mm_setzero_si128();
        for k in 0..4 {
            if on[k] {
                let r = p.offset((2 * k as isize - 3) * stride);
                let a = _mm_loadl_epi64(r.cast());
                let b = _mm_loadl_epi64(r.offset(stride).cast());
                let c = _mm256_castsi256_si128(c[k]);
                sum = _mm_add_epi16(sum, _mm_maddubs_epi16(_mm_unpacklo_epi8(a, b), c));
            }
        }
        sum
    }
}

/// Vertical 32-bit sums of 16 intermediate words, saturated back to words
/// after rounding; `p` addresses the centre row, rows -3..=4 are read.
#[inline(always)]
unsafe fn v16_mid(
    p: *const i16,
    stride: isize,
    c: &[__m256i; 4],
    on: [bool; 4],
    rnd: __m256i,
    sh: __m128i,
) -> __m256i {
    unsafe {
        let (mut lo, mut hi) = (rnd, rnd);
        for k in 0..4 {
            if on[k] {
                let r = p.offset((2 * k as isize - 3) * stride);
                let a = _mm256_loadu_si256(r.cast());
                let b = _mm256_loadu_si256(r.offset(stride).cast());
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), c[k]));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), c[k]));
            }
        }
        // Unpack and pack are both lane-local, so the pixel order is kept.
        _mm256_packs_epi32(_mm256_sra_epi32(lo, sh), _mm256_sra_epi32(hi, sh))
    }
}

/// As `v16_mid` for 8 words.
#[inline(always)]
unsafe fn v8_mid(
    p: *const i16,
    stride: isize,
    c: &[__m256i; 4],
    on: [bool; 4],
    rnd: __m256i,
    sh: __m128i,
) -> __m128i {
    unsafe {
        let mut sum = rnd;
        for k in 0..4 {
            if on[k] {
                let r = p.offset((2 * k as isize - 3) * stride);
                let a = _mm_loadu_si128(r.cast());
                let b = _mm_loadu_si128(r.offset(stride).cast());
                let pairs = _mm256_set_m128i(_mm_unpackhi_epi16(a, b), _mm_unpacklo_epi16(a, b));
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pairs, c[k]));
            }
        }
        let v = _mm256_sra_epi32(sum, sh);
        _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
    }
}

/// One row of a 1-D filter over 8-bit pixels with 16-bit accumulation.
/// `base` is the index of the row's first centre pixel and `stride` the tap
/// direction (1 for horizontal). `f` must satisfy `fits_i16(f, rnd)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tap_row_u8<O: TapOut>(
    dst: &mut [O],
    src: &[u8],
    base: usize,
    stride: isize,
    f: &[i8; 8],
    c: &[__m256i; 4],
    on: [bool; 4],
    rnd: i32,
    sh: i32,
) {
    let w = dst.len();
    let horizontal = stride == 1;
    // First and one-past-last index read for a block of `n` pixels at `x`.
    let first = base as isize - 3 * stride;
    let last = |x: usize, n: usize| {
        if horizontal {
            // Whole 16-byte loads starting 3 pixels to the left.
            base + x - 3 + n + 8
        } else {
            (base as isize + 4 * stride) as usize + x + n
        }
    };
    let vrnd = unsafe { _mm256_set1_epi16(rnd as i16) };
    let vsh = unsafe { _mm_cvtsi32_si128(sh) };
    let mut x = 0;
    if first >= 0 {
        // SAFETY: each block checks its furthest read against `src`.
        unsafe {
            let p = src.as_ptr().add(base);
            while x + 16 <= w && last(x, 16) <= src.len() {
                let v = if horizontal {
                    h16_s16(p.add(x).sub(3), c)
                } else {
                    v16_s16(p.add(x), stride, c, on)
                };
                let v = _mm256_sra_epi16(_mm256_add_epi16(v, vrnd), vsh);
                O::store16(dst.as_mut_ptr().add(x), v);
                x += 16;
            }
            while x < w && last(x, 8) <= src.len() {
                let v = if horizontal {
                    h8_s16(p.add(x).sub(3), c)
                } else {
                    v8_s16(p.add(x), stride, c, on)
                };
                let v = _mm_sra_epi16(_mm_add_epi16(v, _mm256_castsi256_si128(vrnd)), vsh);
                if x + 8 <= w {
                    O::store8(dst.as_mut_ptr().add(x), v);
                    x += 8;
                } else {
                    store_tail(&mut dst[x..], v);
                    x = w;
                }
            }
        }
    }
    for (x, d) in dst.iter_mut().enumerate().skip(x) {
        *d = O::from_i32(round_scalar(
            filter_u8_scalar(src, base + x, stride, f),
            rnd,
            sh,
        ));
    }
}

/// Rounding of the 16-bit 8-tap paths: `(rnd, shift)` of the horizontal-only,
/// vertical-only and second (vertical) stage of the 2-D case. The first 2-D
/// stage always rounds by `bits - 4`.
struct TapRounding {
    h: (i32, i32),
    v: (i32, i32),
    hv: (i32, i32),
}

/// 8-tap filtering of 8-bit pixels for filters whose sums fit 16 bits.
/// Returns false, having written nothing, when they do not.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tap8_u8_s16<O: TapOut>(
    dst: &mut [O],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    fh: Option<[i8; 8]>,
    fv: Option<[i8; 8]>,
    bits: i32,
    r: TapRounding,
    mid_scratch: &mut [i16],
) -> bool {
    let active =
        |f: &[i8; 8]| -> [bool; 4] { std::array::from_fn(|k| f[2 * k] != 0 || f[2 * k + 1] != 0) };
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            if !fits_i16(&fh, rnd0) {
                return false;
            }
            let ch = pair_coeffs_u8(&fh);
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            for (y, mid_row) in mid.chunks_mut(mid_stride).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                tap_row_u8(
                    &mut mid_row[..w],
                    src,
                    base,
                    1,
                    &fh,
                    &ch,
                    [true; 4],
                    rnd0,
                    sh0,
                );
            }
            let cv = pair_coeffs_i16(&fv);
            let on = active(&fv);
            let vrnd = unsafe { _mm256_set1_epi32(r.hv.0) };
            let vsh = unsafe { _mm_cvtsi32_si128(r.hv.1) };
            let ms = mid_stride as isize;
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let dst_row = &mut dst_row[..w];
                // SAFETY: rows y..y+8 of `mid` exist (tmp_h = h + 7), and
                // every block of 8 or 16 words starting below `w` lies inside
                // a `mid_stride`-wide row (a multiple of 16).
                unsafe {
                    let p = mid.as_ptr().add((y + 3) * mid_stride);
                    let mut x = 0;
                    while x + 16 <= w {
                        let v = v16_mid(p.add(x), ms, &cv, on, vrnd, vsh);
                        O::store16(dst_row.as_mut_ptr().add(x), v);
                        x += 16;
                    }
                    while x < w {
                        let v = v8_mid(p.add(x), ms, &cv, on, vrnd, vsh);
                        if x + 8 <= w {
                            O::store8(dst_row.as_mut_ptr().add(x), v);
                            x += 8;
                        } else {
                            store_tail(&mut dst_row[x..], v);
                            x = w;
                        }
                    }
                }
            }
        }
        (Some(f), None) | (None, Some(f)) => {
            let horizontal = fh.is_some();
            let (rnd, sh) = if horizontal { r.h } else { r.v };
            if !fits_i16(&f, rnd) {
                return false;
            }
            let c = pair_coeffs_u8(&f);
            let on = active(&f);
            let stride = if horizontal { 1 } else { src_stride as isize };
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                tap_row_u8(&mut dst_row[..w], src, base, stride, &f, &c, on, rnd, sh);
            }
        }
        (None, None) => return false,
    }
    true
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    mid_scratch: &mut [i16],
) {
    let bits = 6 + (filter_type < 0) as i32;
    let intermediate_rnd = ((1 << bits) >> 1) + ((1 << (bits - 4)) >> 1);
    let fh = crate::mc::get_h_filter(mx, filter_type, w);
    let fv = crate::mc::get_v_filter(my, filter_type, h);
    let rounding = TapRounding {
        h: (intermediate_rnd, bits),
        v: ((1 << bits) >> 1, bits),
        hv: ((1 << (bits + 4)) >> 1, bits + 4),
    };
    if tap8_u8_s16(
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        fh,
        fv,
        bits,
        rounding,
        mid_scratch,
    ) {
        return;
    }
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_mut(mid_stride).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u8x16(src, base + x, 1, &fh);
                    store_i16x16_shift(mid_chunk, lo, hi, rnd0, sh0);
                }
                let x16_done = mid_chunks16.len() * 16;
                let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8_shift(mid_chunk, filter_u8x8(src, base + x, 1, &fh), rnd0, sh0);
                }
                let processed = x16_done + mid_chunks8.len() * 8;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px =
                        round_scalar(filter_u8_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let sh1 = bits + 4;
            let rnd1 = (1 << sh1) >> 1;
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_i16x16_8tap(
                        &mid,
                        (y + 3) * mid_stride + x,
                        mid_stride as isize,
                        &fv,
                    );
                    store_u8x16_clip_shift(dst_chunk, lo, hi, rnd1, sh1);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_u8x8_clip_shift(
                        dst_chunk,
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        sh1,
                    );
                }
                let processed = x16_done + dst_chunks8.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = round_scalar(
                        filter_i16_scalar(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        sh1,
                    )
                    .clamp(0, 255) as u8;
                }
            }
        }
        (Some(fh), None) => {
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u8x16(src, base + x, 1, &fh);
                    store_u8x16_clip_shift(dst_chunk, lo, hi, intermediate_rnd, bits);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_u8x8_clip_shift(
                        dst_chunk,
                        filter_u8x8(src, base + x, 1, &fh),
                        intermediate_rnd,
                        bits,
                    );
                }
                let processed = x16_done + dst_chunks8.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = round_scalar(
                        filter_u8_scalar(src, base + x, 1, &fh),
                        intermediate_rnd,
                        bits,
                    )
                    .clamp(0, 255) as u8;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u8x16(src, base + x, ss, &fv);
                    store_u8x16_clip_shift(dst_chunk, lo, hi, (1 << bits) >> 1, bits);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_u8x8_clip_shift(
                        dst_chunk,
                        filter_u8x8(src, base + x, ss, &fv),
                        (1 << bits) >> 1,
                        bits,
                    );
                }
                let processed = x16_done + dst_chunks8.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = round_scalar(
                        filter_u8_scalar(src, base + x, ss, &fv),
                        (1 << bits) >> 1,
                        bits,
                    )
                    .clamp(0, 255) as u8;
                }
            }
        }
        (None, None) => {
            for (src_row, dst_row) in src[src_off..]
                .chunks(src_stride)
                .zip(dst.chunks_mut(dst_stride))
                .take(h)
            {
                dst_row[..w].copy_from_slice(&src_row[..w]);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap_8bpc_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    mid_scratch: &mut [i16],
) {
    let bits = 6 + (filter_type < 0) as i32;
    let fh = crate::mc::get_h_filter(mx, filter_type, w);
    let fv = crate::mc::get_v_filter(my, filter_type, h);
    let rnd0 = (1 << (bits - 4)) >> 1;
    let rounding = TapRounding {
        h: (rnd0, bits - 4),
        v: (rnd0, bits - 4),
        hv: ((1 << bits) >> 1, bits),
    };
    if tap8_u8_s16(
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        fh,
        fv,
        bits,
        rounding,
        mid_scratch,
    ) {
        return;
    }
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_mut(mid_stride).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u8x16(src, base + x, 1, &fh);
                    store_i16x16_shift(mid_chunk, lo, hi, rnd0, sh0);
                }
                let x16_done = mid_chunks16.len() * 16;
                let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8_shift(mid_chunk, filter_u8x8(src, base + x, 1, &fh), rnd0, sh0);
                }
                let processed = x16_done + mid_chunks8.len() * 8;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px =
                        round_scalar(filter_u8_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let rnd1 = (1 << bits) >> 1;
            for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_i16x16_8tap(
                        &mid,
                        (y + 3) * mid_stride + x,
                        mid_stride as isize,
                        &fv,
                    );
                    store_i16x16_shift(tmp_chunk, lo, hi, rnd1, bits);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8_shift(
                        tmp_chunk,
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        bits,
                    );
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = round_scalar(
                        filter_i16_scalar(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        bits,
                    ) as i16;
                }
            }
        }
        (Some(fh), None) => {
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u8x16(src, base + x, 1, &fh);
                    store_i16x16_shift(tmp_chunk, lo, hi, rnd0, sh0);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8_shift(tmp_chunk, filter_u8x8(src, base + x, 1, &fh), rnd0, sh0);
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px =
                        round_scalar(filter_u8_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u8x16(src, base + x, ss, &fv);
                    store_i16x16_shift(tmp_chunk, lo, hi, rnd0, sh0);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8_shift(tmp_chunk, filter_u8x8(src, base + x, ss, &fv), rnd0, sh0);
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px =
                        round_scalar(filter_u8_scalar(src, base + x, ss, &fv), rnd0, sh0) as i16;
                }
            }
        }
        (None, None) => {
            for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let src_row = unsafe { src.get_unchecked(base..) };
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let v = _mm256_slli_epi16::<4>(load_u8x16_i16(unsafe {
                        src_row.get_unchecked(x..)
                    }));
                    store_i16x16(tmp_chunk, v);
                }
                let processed = tmp_chunks.len() * 16;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (src_row[x] as i16) << 4;
                }
            }
        }
    }
}

/// 8-bit `w_mask`: blends two prepared predictions with a weight derived from
/// their difference and writes that weight (subsampled as the chroma layout
/// asks) to `mask`. Matches `mc::w_mask_8bpc`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn w_mask_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    if w < 8 || !w.is_multiple_of(8) {
        return crate::mc::w_mask_8bpc(
            dst,
            dst_stride,
            tmp1,
            tmp2,
            w,
            h,
            mask,
            mask_stride,
            sign,
            ss_hor,
            ss_ver,
        );
    }
    let c8 = _mm256_set1_epi16(8);
    let c38 = _mm256_set1_epi16(38);
    let c64 = _mm256_set1_epi16(64);
    let ones = _mm256_set1_epi16(1);
    let rnd = _mm256_set1_epi32(512);
    let mask_w = if ss_hor { w / 2 } else { w };

    for row in 0..h {
        let t1 = &tmp1[row * w..][..w];
        let t2 = &tmp2[row * w..][..w];
        let d = &mut dst[row * dst_stride..][..w];
        let mask_row = if ss_ver { row / 2 } else { row };
        let mk = &mut mask[mask_row * mask_stride..][..mask_w];
        // 4:2:0 folds the odd row into the sums the even row left behind.
        let fold = ss_ver && row & 1 != 0;

        let mut x = 0;
        while x < w {
            // A full 16-pixel step, or the last 8 in the low lane.
            let n = (w - x).min(16);
            let (a, b) = unsafe {
                if n == 16 {
                    (
                        _mm256_loadu_si256(t1[x..x + 16].as_ptr().cast()),
                        _mm256_loadu_si256(t2[x..x + 16].as_ptr().cast()),
                    )
                } else {
                    (
                        _mm256_zextsi128_si256(_mm_loadu_si128(t1[x..x + 8].as_ptr().cast())),
                        _mm256_zextsi128_si256(_mm_loadu_si128(t2[x..x + 8].as_ptr().cast())),
                    )
                }
            };
            // m = min(38 + ((|a - b| + 8) >> 8), 64)
            let adiff = _mm256_abs_epi16(_mm256_sub_epi16(a, b));
            let m = _mm256_min_epi16(
                _mm256_add_epi16(_mm256_srli_epi16::<8>(_mm256_add_epi16(adiff, c8)), c38),
                c64,
            );
            let inv = _mm256_sub_epi16(c64, m);
            // (a * m + b * (64 - m) + 512) >> 10, lane-local so pixel order holds.
            let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), _mm256_unpacklo_epi16(m, inv));
            let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), _mm256_unpackhi_epi16(m, inv));
            let px = _mm256_packs_epi32(
                _mm256_srai_epi32::<10>(_mm256_add_epi32(lo, rnd)),
                _mm256_srai_epi32::<10>(_mm256_add_epi32(hi, rnd)),
            );
            let bytes = |v: __m256i| -> [u8; 16] {
                let p = _mm256_permute4x64_epi64::<0b10_00>(_mm256_packus_epi16(v, v));
                let mut out = [0u8; 16];
                unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), _mm256_castsi256_si128(p)) };
                out
            };
            d[x..x + n].copy_from_slice(&bytes(px)[..n]);

            if !ss_hor {
                mk[x..x + n].copy_from_slice(&bytes(m)[..n]);
            } else {
                // m + n of each pixel pair, as dwords in order.
                let pair = _mm256_madd_epi16(m, ones);
                let mx = x / 2;
                let v = if fold {
                    let mut prev = [0u8; 8];
                    prev[..n / 2].copy_from_slice(&mk[mx..mx + n / 2]);
                    let prev =
                        _mm256_cvtepu8_epi32(unsafe { _mm_loadl_epi64(prev.as_ptr().cast()) });
                    _mm256_srli_epi32::<2>(_mm256_add_epi32(
                        _mm256_add_epi32(pair, prev),
                        _mm256_set1_epi32(2 - sign),
                    ))
                } else if ss_ver {
                    pair
                } else {
                    _mm256_srli_epi32::<1>(_mm256_add_epi32(pair, _mm256_set1_epi32(1 - sign)))
                };
                let w16 =
                    _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
                let mut out = [0u8; 16];
                unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), _mm_packus_epi16(w16, w16)) };
                mk[mx..mx + n / 2].copy_from_slice(&out[..n / 2]);
            }
            x += n;
        }
    }
}

/// Tap-pair coefficients of the eight per-pixel warp filters at
/// `pos + x * step`: element `i` holds, for each pixel `x` in order, the words
/// `(f_x[2i], f_x[2i + 1])`.
#[inline(always)]
pub(super) fn warp_pair_coeffs(pos: i32, step: i32) -> [__m256i; 4] {
    let f = |x: i32| -> __m128i {
        let taps: &[i8; 8] =
            &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * x + 512) >> 10)) as usize];
        unsafe { _mm_loadl_epi64(taps.as_ptr().cast()) }
    };
    unsafe {
        // Transpose the 8 filters x 4 tap pairs.
        let a01 = _mm_unpacklo_epi16(f(0), f(1));
        let a23 = _mm_unpacklo_epi16(f(2), f(3));
        let a45 = _mm_unpacklo_epi16(f(4), f(5));
        let a67 = _mm_unpacklo_epi16(f(6), f(7));
        let b0 = _mm_unpacklo_epi32(a01, a23);
        let b1 = _mm_unpackhi_epi32(a01, a23);
        let b2 = _mm_unpacklo_epi32(a45, a67);
        let b3 = _mm_unpackhi_epi32(a45, a67);
        [
            _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(b0, b2)),
            _mm256_cvtepi8_epi16(_mm_unpackhi_epi64(b0, b2)),
            _mm256_cvtepi8_epi16(_mm_unpacklo_epi64(b1, b3)),
            _mm256_cvtepi8_epi16(_mm_unpackhi_epi64(b1, b3)),
        ]
    }
}

/// Words of two 8-sample rows interleaved per pixel: `(a[x], b[x])`.
#[inline(always)]
pub(super) fn warp_pair_rows(a: __m128i, b: __m128i) -> __m256i {
    unsafe { _mm256_set_m128i(_mm_unpackhi_epi16(a, b), _mm_unpacklo_epi16(a, b)) }
}

/// The 14 adjacent row pairs of the 15x8 intermediate block.
#[inline(always)]
pub(super) fn warp_mid_pairs(mid: &[i16; 15 * 8]) -> [__m256i; 14] {
    let rows = mid.as_chunks::<8>().0;
    std::array::from_fn(|j| unsafe {
        warp_pair_rows(
            _mm_loadu_si128(rows[j].as_ptr().cast()),
            _mm_loadu_si128(rows[j + 1].as_ptr().cast()),
        )
    })
}

/// Unrounded vertical sums of output row `y`, filters at `my + x * gamma`.
#[inline(always)]
pub(super) fn warp_vert(pairs: &[__m256i; 14], y: usize, my: i32, gamma: i32) -> __m256i {
    let k = warp_pair_coeffs(my, gamma);
    unsafe {
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(pairs[y], k[0]),
                _mm256_madd_epi16(pairs[y + 2], k[1]),
            ),
            _mm256_add_epi32(
                _mm256_madd_epi16(pairs[y + 4], k[2]),
                _mm256_madd_epi16(pairs[y + 6], k[3]),
            ),
        )
    }
}

/// Shuffle picking the byte pairs `(x + 2i, x + 2i + 1)`, x = 0..8, from a
/// vector holding bytes 0..=7 followed by bytes 7..=14.
const fn warp_h_shuffle(i: usize) -> [i8; 16] {
    let mut m = [0i8; 16];
    let mut n = 0;
    while n < 16 {
        let j = n / 2 + n % 2 + 2 * i;
        m[n] = if j < 8 { j as i8 } else { j as i8 + 1 };
        n += 1;
    }
    m
}
static WARP_H_SHUFFLE: [[i8; 16]; 4] = [
    warp_h_shuffle(0),
    warp_h_shuffle(1),
    warp_h_shuffle(2),
    warp_h_shuffle(3),
];

/// One intermediate row: 8 pixels filtered horizontally at `mx + x * alpha`.
/// Reads the 15 bytes at `row_base`.
#[inline]
#[target_feature(enable = "avx2")]
fn warp_horz_u8x8(src: &[u8], row_base: usize, mx: i32, alpha: i32) -> __m256i {
    let k = warp_pair_coeffs(mx, alpha);
    unsafe {
        let p = src.get_unchecked(row_base..).as_ptr();
        let s = _mm_unpacklo_epi64(_mm_loadl_epi64(p.cast()), _mm_loadl_epi64(p.add(7).cast()));
        let mut acc = _mm256_setzero_si256();
        for i in 0..4 {
            let shuf = _mm_loadu_si128(WARP_H_SHUFFLE[i].as_ptr().cast());
            let px = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(s, shuf));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(px, k[i]));
        }
        round_s32(acc, 4, 3)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8_shift_warp(dst: &mut [i16], v: __m256i, rnd: i32, shift: i32) {
    store_i16x8(dst, round_s32(v, rnd, shift));
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u8x8_shift_warp(dst: &mut [u8], v: __m256i, rnd: i32, shift: i32) {
    store_u8x8_clip_shift(dst, v, rnd, shift);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn warp_affine_8x8_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let alpha = abcd[0] as i32;
    let beta = abcd[1] as i32;
    let gamma = abcd[2] as i32;
    let delta = abcd[3] as i32;
    let mut mid = [0i16; 15 * 8];
    let mut row_base = src_off.wrapping_sub(3 * src_stride + 3);

    for mid_row in mid.as_chunks_mut::<8>().0.iter_mut() {
        store_i16x8(mid_row, warp_horz_u8x8(src, row_base, mx, alpha));
        row_base += src_stride;
        mx += beta;
    }
    let pairs = warp_mid_pairs(&mid);

    for (y, dst_row) in dst.chunks_mut(dst_stride).take(8).enumerate() {
        let v = warp_vert(&pairs, y, my, gamma);
        store_u8x8_shift_warp(&mut dst_row[..8], v, 1024, 11);
        my += delta;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn warp_affine_8x8t_8bpc_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
) {
    let alpha = abcd[0] as i32;
    let beta = abcd[1] as i32;
    let gamma = abcd[2] as i32;
    let delta = abcd[3] as i32;
    let mut mid = [0i16; 15 * 8];
    let mut row_base = src_off.wrapping_sub(3 * src_stride + 3);

    for mid_row in mid.as_chunks_mut::<8>().0.iter_mut() {
        store_i16x8(mid_row, warp_horz_u8x8(src, row_base, mx, alpha));
        row_base += src_stride;
        mx += beta;
    }
    let pairs = warp_mid_pairs(&mid);

    for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(8).enumerate() {
        let v = warp_vert(&pairs, y, my, gamma);
        store_i16x8_shift_warp(&mut tmp_row[..8], v, 64, 7);
        my += delta;
    }
}
