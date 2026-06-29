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
        for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(h + 1).enumerate() {
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
        for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
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
        for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
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
        for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
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
            .chunks_exact(src_stride)
            .zip(dst.chunks_exact_mut(dst_stride))
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
        for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(h + 1).enumerate() {
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
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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

#[inline]
#[target_feature(enable = "avx2")]
fn filter_u8x8(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    if stride == 1 {
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
    if stride == 1 {
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
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(tmp_h).enumerate() {
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
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
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
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
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
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
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
                .chunks_exact(src_stride)
                .zip(dst.chunks_exact_mut(dst_stride))
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
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(tmp_h).enumerate() {
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
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
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

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x8_i32_warp(src: &[u8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x8_i32_warp(src: &[i16]) -> __m256i {
    unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn warp_coeff_i32x8(pos: i32, step: i32, tap: usize) -> __m256i {
    let f0 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + 512) >> 10)) as usize];
    let f1 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step + 512) >> 10)) as usize];
    let f2 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * 2 + 512) >> 10)) as usize];
    let f3 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * 3 + 512) >> 10)) as usize];
    let f4 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * 4 + 512) >> 10)) as usize];
    let f5 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * 5 + 512) >> 10)) as usize];
    let f6 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * 6 + 512) >> 10)) as usize];
    let f7 = &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * 7 + 512) >> 10)) as usize];
    _mm256_setr_epi32(
        f0[tap] as i32,
        f1[tap] as i32,
        f2[tap] as i32,
        f3[tap] as i32,
        f4[tap] as i32,
        f5[tap] as i32,
        f6[tap] as i32,
        f7[tap] as i32,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn warp_horz_u8x8(src: &[u8], row_base: usize, mx: i32, alpha: i32) -> __m256i {
    let mut acc = _mm256_setzero_si256();
    for tap in 0..8 {
        let px = load_u8x8_i32_warp(unsafe { src.get_unchecked(row_base + tap..) });
        let coeff = warp_coeff_i32x8(mx, alpha, tap);
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(px, coeff));
    }
    round_s32(acc, 4, 3)
}

#[inline]
#[target_feature(enable = "avx2")]
fn warp_vert_i16x8(mid: &[i16], base: usize, stride: usize, my: i32, gamma: i32) -> __m256i {
    let mut acc = _mm256_setzero_si256();
    for tap in 0..8 {
        let px = load_i16x8_i32_warp(unsafe { mid.get_unchecked(base + tap * stride..) });
        let coeff = warp_coeff_i32x8(my, gamma, tap);
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(px, coeff));
    }
    acc
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

    for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(8).enumerate() {
        let v = warp_vert_i16x8(&mid, y * 8, 8, my, gamma);
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

    for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(8).enumerate() {
        let v = warp_vert_i16x8(&mid, y * 8, 8, my, gamma);
        store_i16x8_shift_warp(&mut tmp_row[..8], v, 64, 7);
        my += delta;
    }
}
