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
    unsafe {
        let p8 = _mm256_packus_epi16(v, v);
        let lo = _mm256_castsi256_si128(p8);
        let hi = _mm256_extracti128_si256::<1>(p8);
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
    unsafe {
        let v = _mm256_srai_epi32::<8>(_mm256_add_epi32(v, _mm256_set1_epi32(128)));
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let p16 = _mm_packus_epi32(lo, hi);
        let p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(dst.as_mut_ptr().cast(), p8);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8_round4_from_i32(dst: &mut [i16], v: __m256i) {
    let v = _mm256_srai_epi32::<4>(_mm256_add_epi32(v, _mm256_set1_epi32(8)));
    store_i16x8(dst, v);
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_u8x16_i16(src: &[u8], base: usize, stride: usize, mxy: i32) -> __m256i {
    let a = load_u8x16_i16(unsafe { src.get_unchecked(base..) });
    let b = load_u8x16_i16(unsafe { src.get_unchecked(base + stride..) });
    _mm256_add_epi16(
        _mm256_slli_epi16::<4>(a),
        _mm256_mullo_epi16(_mm256_sub_epi16(b, a), _mm256_set1_epi16(mxy as i16)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_i16x8_i32(a16: __m128i, b16: __m128i, mxy: i32) -> __m256i {
    let a = _mm256_cvtepi16_epi32(a16);
    let b = _mm256_cvtepi16_epi32(b16);
    _mm256_add_epi32(
        _mm256_slli_epi32::<4>(a),
        _mm256_mullo_epi32(_mm256_sub_epi32(b, a), _mm256_set1_epi32(mxy)),
    )
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
            let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<8>();
            for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                store_u8x8_round8_from_i32(dst_chunk, bilin_i16x8_i32(a, b, my));
            }
            let processed = dst_chunks.len() * 8;
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
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<8>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                store_i16x8_round4_from_i32(tmp_chunk, bilin_i16x8_i32(a, b, my));
            }
            let processed = tmp_chunks.len() * 8;
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
fn load_u8x8_i32(src: &[u8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_i16x8_n_s32(x: __m128i, k: i32) -> __m256i {
    let zero = _mm_setzero_si128();
    let lo = _mm_unpacklo_epi16(x, zero);
    let hi = _mm_unpackhi_epi16(x, zero);
    let xz = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
    let kz = _mm256_set1_epi32((k as i16 as u16) as i32);
    _mm256_madd_epi16(xz, kz)
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_u8x8(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    let offsets = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = _mm256_setzero_si256();
    for k in 0..8 {
        let idx = (base as isize + offsets[k] * stride) as usize;
        let s = load_u8x8_i32(unsafe { src.get_unchecked(idx..) });
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(s, _mm256_set1_epi32(f[k] as i32)));
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_i16x8_8tap(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    let offsets = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = _mm256_setzero_si256();
    for k in 0..8 {
        let idx = (base as isize + offsets[k] * stride) as usize;
        let s = load_i16x8(unsafe { src.get_unchecked(idx..) });
        sum = _mm256_add_epi32(sum, mul_i16x8_n_s32(s, f[k] as i32));
    }
    sum
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
            let mid_stride = w.next_multiple_of(8).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let src_row = unsafe { src.get_unchecked(base..) };
                let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_i16x8_shift(mid_chunk, filter_u8x8(src_row, x, 1, &fh), rnd0, sh0);
                }
                let processed = mid_chunks.len() * 8;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px = round_scalar(filter_u8_scalar(src_row, x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let sh1 = bits + 4;
            let rnd1 = (1 << sh1) >> 1;
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
                let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_u8x8_clip_shift(
                        dst_chunk,
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        sh1,
                    );
                }
                let processed = dst_chunks.len() * 8;
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
                let src_row = unsafe { src.get_unchecked(base..) };
                let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_u8x8_clip_shift(
                        dst_chunk,
                        filter_u8x8(src_row, x, 1, &fh),
                        intermediate_rnd,
                        bits,
                    );
                }
                let processed = dst_chunks.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px =
                        round_scalar(filter_u8_scalar(src_row, x, 1, &fh), intermediate_rnd, bits)
                            .clamp(0, 255) as u8;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_u8x8_clip_shift(
                        dst_chunk,
                        filter_u8x8(src, base + x, ss, &fv),
                        (1 << bits) >> 1,
                        bits,
                    );
                }
                let processed = dst_chunks.len() * 8;
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
            let mid_stride = w.next_multiple_of(8).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let src_row = unsafe { src.get_unchecked(base..) };
                let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_i16x8_shift(mid_chunk, filter_u8x8(src_row, x, 1, &fh), rnd0, sh0);
                }
                let processed = mid_chunks.len() * 8;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px = round_scalar(filter_u8_scalar(src_row, x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let rnd1 = (1 << bits) >> 1;
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_i16x8_shift(
                        tmp_chunk,
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        bits,
                    );
                }
                let processed = tmp_chunks.len() * 8;
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
                let src_row = unsafe { src.get_unchecked(base..) };
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_i16x8_shift(tmp_chunk, filter_u8x8(src_row, x, 1, &fh), rnd0, sh0);
                }
                let processed = tmp_chunks.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = round_scalar(filter_u8_scalar(src_row, x, 1, &fh), rnd0, sh0) as i16;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 8;
                    store_i16x8_shift(tmp_chunk, filter_u8x8(src, base + x, ss, &fv), rnd0, sh0);
                }
                let processed = tmp_chunks.len() * 8;
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
