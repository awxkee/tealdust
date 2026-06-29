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

#[inline(always)]
fn load_u16x8_i32(p: &[u16]) -> __m256i {
    unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(p.as_ptr().cast())) }
}

#[inline(always)]
fn sll_s32(v: __m256i, shift: i32) -> __m256i {
    unsafe { _mm256_sll_epi32(v, _mm_cvtsi32_si128(shift)) }
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
fn store_clip_u16x8(dst: &mut [u16], v: __m256i, rnd: i32, shift: i32, max: __m128i) {
    let v = round_s32(v, rnd, shift);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_min_epu16(_mm_packus_epi32(lo, hi), max);
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), p);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8(dst: &mut [i16], v: __m256i, rnd: i32, shift: i32, bias: i32) {
    let v = _mm256_sub_epi32(round_s32(v, rnd, shift), _mm256_set1_epi32(bias));
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packs_epi32(lo, hi));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_clip_u16x16(
    dst: &mut [u16],
    lo: __m256i,
    hi: __m256i,
    rnd: i32,
    shift: i32,
    max: __m128i,
) {
    let lo = round_s32(lo, rnd, shift);
    let hi = round_s32(hi, rnd, shift);
    let lo0 = _mm256_castsi256_si128(lo);
    let lo1 = _mm256_extracti128_si256::<1>(lo);
    let hi0 = _mm256_castsi256_si128(hi);
    let hi1 = _mm256_extracti128_si256::<1>(hi);
    let p0 = _mm_min_epu16(_mm_packus_epi32(lo0, lo1), max);
    let p1 = _mm_min_epu16(_mm_packus_epi32(hi0, hi1), max);
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), p0);
        _mm_storeu_si128(dst.as_mut_ptr().add(8).cast(), p1);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16(dst: &mut [i16], lo: __m256i, hi: __m256i, rnd: i32, shift: i32, bias: i32) {
    let lo = _mm256_sub_epi32(round_s32(lo, rnd, shift), _mm256_set1_epi32(bias));
    let hi = _mm256_sub_epi32(round_s32(hi, rnd, shift), _mm256_set1_epi32(bias));
    let lo0 = _mm256_castsi256_si128(lo);
    let lo1 = _mm256_extracti128_si256::<1>(lo);
    let hi0 = _mm256_castsi256_si128(hi);
    let hi1 = _mm256_extracti128_si256::<1>(hi);
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packs_epi32(lo0, lo1));
        _mm_storeu_si128(dst.as_mut_ptr().add(8).cast(), _mm_packs_epi32(hi0, hi1));
    }
}

#[inline(always)]
fn load_u16x8(p: &[u16]) -> __m128i {
    unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
}

#[inline(always)]
fn load_i16x8(p: &[i16]) -> __m128i {
    unsafe { _mm_loadu_si128(p.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_coeff_i16(mxy: i32) -> __m256i {
    _mm256_set1_epi32(((mxy as i16 as u16 as i32) << 16) | (16 - mxy) as u16 as i32)
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_bilin_i16x8_s32(a: __m128i, b: __m128i, mxy: i32) -> __m256i {
    let lo = _mm_unpacklo_epi16(a, b);
    let hi = _mm_unpackhi_epi16(a, b);
    let pairs = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
    _mm256_madd_epi16(pairs, bilin_coeff_i16(mxy))
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

#[inline]
#[target_feature(enable = "avx2")]
fn filter_u16x8(src: &[u16], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
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
        let a = load_u16x8(unsafe { src.get_unchecked(idx0..) });
        let b = load_u16x8(unsafe { src.get_unchecked(idx1..) });
        sum = _mm256_add_epi32(sum, madd_i16x8_pair_s32(a, b, c0 as i16, c1 as i16));
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_i16x8(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
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

#[inline(always)]
fn filter_u16x16(src: &[u16], base: usize, stride: isize, f: &[i8; 8]) -> (__m256i, __m256i) {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo = unsafe { _mm256_setzero_si256() };
    let mut hi = unsafe { _mm256_setzero_si256() };
    for k in (0..8).step_by(2) {
        let c0 = f[k];
        let c1 = f[k + 1];
        if c0 == 0 && c1 == 0 {
            continue;
        }
        let idx0 = (base as isize + OFFSETS[k] * stride) as usize;
        let idx1 = (base as isize + OFFSETS[k + 1] * stride) as usize;
        let a0 = load_u16x8(unsafe { src.get_unchecked(idx0..) });
        let b0 = load_u16x8(unsafe { src.get_unchecked(idx1..) });
        let a1 = load_u16x8(unsafe { src.get_unchecked(idx0 + 8..) });
        let b1 = load_u16x8(unsafe { src.get_unchecked(idx1 + 8..) });
        lo = unsafe { _mm256_add_epi32(lo, madd_i16x8_pair_s32(a0, b0, c0 as i16, c1 as i16)) };
        hi = unsafe { _mm256_add_epi32(hi, madd_i16x8_pair_s32(a1, b1, c0 as i16, c1 as i16)) };
    }
    (lo, hi)
}

#[inline(always)]
fn filter_i16x16(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> (__m256i, __m256i) {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo = unsafe { _mm256_setzero_si256() };
    let mut hi = unsafe { _mm256_setzero_si256() };
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
        lo = unsafe { _mm256_add_epi32(lo, madd_i16x8_pair_s32(a0, b0, c0 as i16, c1 as i16)) };
        hi = unsafe { _mm256_add_epi32(hi, madd_i16x8_pair_s32(a1, b1, c0 as i16, c1 as i16)) };
    }
    (lo, hi)
}

#[inline(always)]
fn filter_u16_scalar(src: &[u16], base: usize, stride: isize, f: &[i8; 8]) -> i32 {
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
fn clip(v: i32, bitdepth: u8) -> u16 {
    v.clamp(0, (1 << bitdepth) - 1) as u16
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
fn bilin_u16x8(src: &[u16], base: usize, stride: usize, mxy: i32) -> __m256i {
    let a = load_u16x8(unsafe { src.get_unchecked(base..) });
    let b = load_u16x8(unsafe { src.get_unchecked(base + stride..) });
    madd_bilin_i16x8_s32(a, b, mxy)
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_i16x8(a: __m128i, b: __m128i, mxy: i32) -> __m256i {
    madd_bilin_i16x8_s32(a, b, mxy)
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_u16x16(src: &[u16], base: usize, stride: usize, mxy: i32) -> (__m256i, __m256i) {
    (
        bilin_u16x8(src, base, stride, mxy),
        bilin_u16x8(src, base + 8, stride, mxy),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn bilin_i16x16(
    a0: __m128i,
    b0: __m128i,
    a1: __m128i,
    b1: __m128i,
    mxy: i32,
) -> (__m256i, __m256i) {
    (bilin_i16x8(a0, b0, mxy), bilin_i16x8(a1, b1, mxy))
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_hbd_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    bitdepth: u8,
) {
    let ib = 14 - bitdepth as i32;
    let bias = 8192i32;
    for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
        let src_row = unsafe { src.get_unchecked(y * src_stride..) };
        let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
        for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
            let x = chunk_idx * 16;
            let lo = _mm256_sub_epi32(
                sll_s32(load_u16x8_i32(unsafe { src_row.get_unchecked(x..) }), ib),
                _mm256_set1_epi32(bias),
            );
            let hi = _mm256_sub_epi32(
                sll_s32(
                    load_u16x8_i32(unsafe { src_row.get_unchecked(x + 8..) }),
                    ib,
                ),
                _mm256_set1_epi32(bias),
            );
            let lo0 = _mm256_castsi256_si128(lo);
            let lo1 = _mm256_extracti128_si256::<1>(lo);
            let hi0 = _mm256_castsi256_si128(hi);
            let hi1 = _mm256_extracti128_si256::<1>(hi);
            unsafe {
                _mm_storeu_si128(tmp_chunk.as_mut_ptr().cast(), _mm_packs_epi32(lo0, lo1));
                _mm_storeu_si128(
                    tmp_chunk.as_mut_ptr().add(8).cast(),
                    _mm_packs_epi32(hi0, hi1),
                );
            }
        }
        let x16_done = tmp_chunks16.len() * 16;
        let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
        for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
            let x = x16_done + chunk_idx * 8;
            let s = load_u16x8_i32(unsafe { src_row.get_unchecked(x..) });
            let v = _mm256_sub_epi32(sll_s32(s, ib), _mm256_set1_epi32(bias));
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            unsafe {
                _mm_storeu_si128(tmp_chunk.as_mut_ptr().cast(), _mm_packs_epi32(lo, hi));
            }
        }
        let processed = x16_done + tmp_chunks8.len() * 8;
        for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
            *tmp_px = (((src_row[x] as i32) << ib) - bias) as i16;
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_hbd_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
    mid_scratch: &mut [i16],
) {
    let ib = 14 - bitdepth as i32;
    let maxv = _mm_set1_epi16(((1 << bitdepth) - 1) as i16);
    let intermediate_rnd = (1 << ib) >> 1;
    if mx != 0 && my != 0 {
        let mid_stride = w.next_multiple_of(16).max(64);
        let mid = &mut mid_scratch[..mid_stride * (h + 1)];
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(h + 1).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let (lo, hi) = bilin_u16x16(src_row, x, 1, mx);
                store_i16x16(mid_chunk, lo, hi, rnd0, sh0, 0);
            }
            let x16_done = mid_chunks16.len() * 16;
            let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
            for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                store_i16x8(mid_chunk, bilin_u16x8(src_row, x, 1, mx), rnd0, sh0, 0);
            }
            let processed = x16_done + mid_chunks8.len() * 8;
            for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_row[x + 1] as i32;
                *mid_px = round_scalar(16 * a + mx * (b - a), rnd0, sh0) as i16;
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
                let (lo, hi) = bilin_i16x16(a0, b0, a1, b1, my);
                store_clip_u16x16(dst_chunk, lo, hi, 1 << (3 + ib), 4 + ib, maxv);
            }
            let x16_done = dst_chunks16.len() * 16;
            let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
            for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                let v = bilin_i16x8(a, b, my);
                store_clip_u16x8(dst_chunk, v, 1 << (3 + ib), 4 + ib, maxv);
            }
            let processed = x16_done + dst_chunks8.len() * 8;
            for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                let a = mid_row[x] as i32;
                let b = mid_next_row[x] as i32;
                *dst_px = clip(
                    round_scalar(16 * a + my * (b - a), 1 << (3 + ib), 4 + ib),
                    bitdepth,
                );
            }
        }
    } else if mx != 0 {
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let (lo, hi) = bilin_u16x16(src_row, x, 1, mx);
                let lo = round_s32(lo, rnd0, sh0);
                let hi = round_s32(hi, rnd0, sh0);
                store_clip_u16x16(dst_chunk, lo, hi, intermediate_rnd, ib, maxv);
            }
            let x16_done = dst_chunks16.len() * 16;
            let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
            for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let px = round_s32(bilin_u16x8(src_row, x, 1, mx), rnd0, sh0);
                store_clip_u16x8(dst_chunk, px, intermediate_rnd, ib, maxv);
            }
            let processed = x16_done + dst_chunks8.len() * 8;
            for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_row[x + 1] as i32;
                let px = round_scalar(16 * a + mx * (b - a), rnd0, sh0);
                *dst_px = clip(round_scalar(px, intermediate_rnd, ib), bitdepth);
            }
        }
    } else if my != 0 {
        for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let src_next_row = unsafe { src.get_unchecked((y + 1) * src_stride..) };
            let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let (lo, hi) = bilin_u16x16(src_row, x, src_stride, my);
                store_clip_u16x16(dst_chunk, lo, hi, 8, 4, maxv);
            }
            let x16_done = dst_chunks16.len() * 16;
            let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
            for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                store_clip_u16x8(
                    dst_chunk,
                    bilin_u16x8(src_row, x, src_stride, my),
                    8,
                    4,
                    maxv,
                );
            }
            let processed = x16_done + dst_chunks8.len() * 8;
            for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_next_row[x] as i32;
                *dst_px = clip(round_scalar(16 * a + my * (b - a), 8, 4), bitdepth);
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
pub(crate) fn prep_bilin_hbd_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
    mid_scratch: &mut [i16],
) {
    let ib = 14 - bitdepth as i32;
    let bias = 8192i32;
    if mx != 0 && my != 0 {
        let mid_stride = w.next_multiple_of(16).max(64);
        let mid = &mut mid_scratch[..mid_stride * (h + 1)];
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(h + 1).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let (lo, hi) = bilin_u16x16(src_row, x, 1, mx);
                store_i16x16(mid_chunk, lo, hi, rnd0, sh0, 0);
            }
            let x16_done = mid_chunks16.len() * 16;
            let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
            for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                store_i16x8(mid_chunk, bilin_u16x8(src_row, x, 1, mx), rnd0, sh0, 0);
            }
            let processed = x16_done + mid_chunks8.len() * 8;
            for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_row[x + 1] as i32;
                *mid_px = round_scalar(16 * a + mx * (b - a), rnd0, sh0) as i16;
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
                let (lo, hi) = bilin_i16x16(a0, b0, a1, b1, my);
                store_i16x16(tmp_chunk, lo, hi, 8, 4, bias);
            }
            let x16_done = tmp_chunks16.len() * 16;
            let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
            for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                let v = bilin_i16x8(a, b, my);
                store_i16x8(tmp_chunk, v, 8, 4, bias);
            }
            let processed = x16_done + tmp_chunks8.len() * 8;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                let a = mid_row[x] as i32;
                let b = mid_next_row[x] as i32;
                *tmp_px = (round_scalar(16 * a + my * (b - a), 8, 4) - bias) as i16;
            }
        }
    } else if mx != 0 {
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let (lo, hi) = bilin_u16x16(src_row, x, 1, mx);
                store_i16x16(tmp_chunk, lo, hi, rnd0, sh0, bias);
            }
            let x16_done = tmp_chunks16.len() * 16;
            let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
            for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                store_i16x8(tmp_chunk, bilin_u16x8(src_row, x, 1, mx), rnd0, sh0, bias);
            }
            let processed = x16_done + tmp_chunks8.len() * 8;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_row[x + 1] as i32;
                *tmp_px = (round_scalar(16 * a + mx * (b - a), rnd0, sh0) - bias) as i16;
            }
        }
    } else if my != 0 {
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let src_next_row = unsafe { src.get_unchecked((y + 1) * src_stride..) };
            let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let (lo, hi) = bilin_u16x16(src_row, x, src_stride, my);
                store_i16x16(tmp_chunk, lo, hi, rnd0, sh0, bias);
            }
            let x16_done = tmp_chunks16.len() * 16;
            let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
            for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                store_i16x8(
                    tmp_chunk,
                    bilin_u16x8(src_row, x, src_stride, my),
                    rnd0,
                    sh0,
                    bias,
                );
            }
            let processed = x16_done + tmp_chunks8.len() * 8;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_next_row[x] as i32;
                *tmp_px = (round_scalar(16 * a + my * (b - a), rnd0, sh0) - bias) as i16;
            }
        }
    } else {
        prep_hbd_avx2(tmp, tmp_stride, src, src_stride, w, h, bitdepth);
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_hbd_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
    mid_scratch: &mut [i16],
) {
    let bits = 6 + (filter_type < 0) as i32;
    let ib = 14 - bitdepth as i32;
    let intermediate_rnd = ((1 << bits) >> 1) + ((1 << (bits - ib)) >> 1);
    let fh = crate::mc::get_h_filter(mx, filter_type, w);
    let fv = crate::mc::get_v_filter(my, filter_type, h);
    let maxv = _mm_set1_epi16(((1 << bitdepth) - 1) as i16);
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u16x16(src, base + x, 1, &fh);
                    store_i16x16(mid_chunk, lo, hi, rnd0, sh0, 0);
                }
                let x16_done = mid_chunks16.len() * 16;
                let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8(mid_chunk, filter_u16x8(src, base + x, 1, &fh), rnd0, sh0, 0);
                }
                let processed = x16_done + mid_chunks8.len() * 8;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px =
                        round_scalar(filter_u16_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let sh1 = bits + ib;
            let rnd1 = (1 << sh1) >> 1;
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
                let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) =
                        filter_i16x16(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    store_clip_u16x16(dst_chunk, lo, hi, rnd1, sh1, maxv);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_clip_u16x8(
                        dst_chunk,
                        filter_i16x8(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        sh1,
                        maxv,
                    );
                }
                let processed = x16_done + dst_chunks8.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = clip(
                        round_scalar(
                            filter_i16_scalar(
                                &mid,
                                (y + 3) * mid_stride + x,
                                mid_stride as isize,
                                &fv,
                            ),
                            rnd1,
                            sh1,
                        ),
                        bitdepth,
                    );
                }
            }
        }
        (Some(fh), None) => {
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (dst_chunks16, dst_rem16) = dst_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, dst_chunk) in dst_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u16x16(src, base + x, 1, &fh);
                    store_clip_u16x16(dst_chunk, lo, hi, intermediate_rnd, bits, maxv);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_clip_u16x8(
                        dst_chunk,
                        filter_u16x8(src, base + x, 1, &fh),
                        intermediate_rnd,
                        bits,
                        maxv,
                    );
                }
                let processed = x16_done + dst_chunks8.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = clip(
                        round_scalar(
                            filter_u16_scalar(src, base + x, 1, &fh),
                            intermediate_rnd,
                            bits,
                        ),
                        bitdepth,
                    );
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
                    let (lo, hi) = filter_u16x16(src, base + x, ss, &fv);
                    store_clip_u16x16(dst_chunk, lo, hi, (1 << bits) >> 1, bits, maxv);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_clip_u16x8(
                        dst_chunk,
                        filter_u16x8(src, base + x, ss, &fv),
                        (1 << bits) >> 1,
                        bits,
                        maxv,
                    );
                }
                let processed = x16_done + dst_chunks8.len() * 8;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = clip(
                        round_scalar(
                            filter_u16_scalar(src, base + x, ss, &fv),
                            (1 << bits) >> 1,
                            bits,
                        ),
                        bitdepth,
                    );
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
pub(crate) fn prep_8tap_hbd_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
    bitdepth: u8,
    mid_scratch: &mut [i16],
) {
    let bits = 6 + (filter_type < 0) as i32;
    let ib = 14 - bitdepth as i32;
    let bias = 8192i32;
    let fh = crate::mc::get_h_filter(mx, filter_type, w);
    let fv = crate::mc::get_v_filter(my, filter_type, h);
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(mid_stride).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u16x16(src, base + x, 1, &fh);
                    store_i16x16(mid_chunk, lo, hi, rnd0, sh0, 0);
                }
                let x16_done = mid_chunks16.len() * 16;
                let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8(mid_chunk, filter_u16x8(src, base + x, 1, &fh), rnd0, sh0, 0);
                }
                let processed = x16_done + mid_chunks8.len() * 8;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px =
                        round_scalar(filter_u16_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let rnd1 = (1 << bits) >> 1;
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) =
                        filter_i16x16(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    store_i16x16(tmp_chunk, lo, hi, rnd1, bits, bias);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8(
                        tmp_chunk,
                        filter_i16x8(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        bits,
                        bias,
                    );
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (round_scalar(
                        filter_i16_scalar(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv),
                        rnd1,
                        bits,
                    ) - bias) as i16;
                }
            }
        }
        (Some(fh), None) => {
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u16x16(src, base + x, 1, &fh);
                    store_i16x16(tmp_chunk, lo, hi, rnd0, sh0, bias);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8(
                        tmp_chunk,
                        filter_u16x8(src, base + x, 1, &fh),
                        rnd0,
                        sh0,
                        bias,
                    );
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (round_scalar(filter_u16_scalar(src, base + x, 1, &fh), rnd0, sh0)
                        - bias) as i16;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let (lo, hi) = filter_u16x16(src, base + x, ss, &fv);
                    store_i16x16(tmp_chunk, lo, hi, rnd0, sh0, bias);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    store_i16x8(
                        tmp_chunk,
                        filter_u16x8(src, base + x, ss, &fv),
                        rnd0,
                        sh0,
                        bias,
                    );
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (round_scalar(filter_u16_scalar(src, base + x, ss, &fv), rnd0, sh0)
                        - bias) as i16;
                }
            }
        }
        (None, None) => prep_hbd_avx2(tmp, tmp_stride, &src[src_off..], src_stride, w, h, bitdepth),
    }
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
fn warp_horz_u16x8(
    src: &[u16],
    row_base: usize,
    mx: i32,
    alpha: i32,
    rnd: i32,
    shift: i32,
) -> __m128i {
    let mut acc = _mm256_setzero_si256();
    for tap in 0..8 {
        let px = load_u16x8_i32(unsafe { src.get_unchecked(row_base + tap..) });
        let coeff = warp_coeff_i32x8(mx, alpha, tap);
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(px, coeff));
    }
    let v = round_s32(acc, rnd, shift);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_packs_epi32(lo, hi)
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

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn warp_affine_8x8_hbd_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
    bitdepth: u8,
) {
    let ib = 14 - bitdepth as i32;
    let h_shift = 7 - ib;
    let h_rnd = (1 << h_shift) >> 1;
    let v_shift = 7 + ib;
    let v_rnd = (1 << v_shift) >> 1;
    let max = _mm_set1_epi16(((1 << bitdepth) - 1) as i16);
    let alpha = abcd[0] as i32;
    let beta = abcd[1] as i32;
    let gamma = abcd[2] as i32;
    let delta = abcd[3] as i32;
    let mut mid = [0i16; 15 * 8];
    let mut row_base = src_off.wrapping_sub(3 * src_stride + 3);

    for mid_row in mid.as_chunks_mut::<8>().0.iter_mut() {
        let v = warp_horz_u16x8(src, row_base, mx, alpha, h_rnd, h_shift);
        unsafe { _mm_storeu_si128(mid_row.as_mut_ptr().cast(), v) };
        row_base += src_stride;
        mx += beta;
    }

    for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(8).enumerate() {
        let v = warp_vert_i16x8(&mid, y * 8, 8, my, gamma);
        store_clip_u16x8(&mut dst_row[..8], v, v_rnd, v_shift, max);
        my += delta;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn warp_affine_8x8t_hbd_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mut mx: i32,
    mut my: i32,
    bitdepth: u8,
) {
    let ib = 14 - bitdepth as i32;
    let h_shift = 7 - ib;
    let h_rnd = (1 << h_shift) >> 1;
    let alpha = abcd[0] as i32;
    let beta = abcd[1] as i32;
    let gamma = abcd[2] as i32;
    let delta = abcd[3] as i32;
    let mut mid = [0i16; 15 * 8];
    let mut row_base = src_off.wrapping_sub(3 * src_stride + 3);

    for mid_row in mid.as_chunks_mut::<8>().0.iter_mut() {
        let v = warp_horz_u16x8(src, row_base, mx, alpha, h_rnd, h_shift);
        unsafe { _mm_storeu_si128(mid_row.as_mut_ptr().cast(), v) };
        row_base += src_stride;
        mx += beta;
    }

    for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(8).enumerate() {
        let v = warp_vert_i16x8(&mid, y * 8, 8, my, gamma);
        store_i16x8(&mut tmp_row[..8], v, 64, 7, 8192);
        my += delta;
    }
}

#[cfg(test)]
mod inter_hd_avx_tests {
    use super::*;

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    #[test]
    fn round_s32_avx_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = R(0x51b3_c0ffee_u64 | 1);
        for shift in 0..=14i32 {
            for &rnd in &[0, 1, (1 << shift) >> 1, (1 << shift) - 1, 8, 2048] {
                for _ in 0..1500 {
                    let v: [i32; 8] = std::array::from_fn(|_| r.range(-300_000, 300_000));
                    let vv = unsafe { _mm256_loadu_si256(v.as_ptr() as *const __m256i) };
                    let rv = unsafe { round_s32(vv, rnd, shift) };
                    let mut out = [0i32; 8];
                    unsafe {
                        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, rv);
                    }
                    for i in 0..8 {
                        assert_eq!(
                            out[i],
                            round_scalar(v[i], rnd, shift),
                            "v={} rnd={} shift={}",
                            v[i],
                            rnd,
                            shift
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bilin_u16x8_avx_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = R(0xb171_600d_u64 | 1);
        for mxy in 1..16 {
            for _ in 0..5000 {
                let src: Vec<u16> = (0..64).map(|_| r.range(0, (1 << 12) - 1) as u16).collect();
                let base = 9usize;
                let sv = unsafe { bilin_u16x8(&src, base, 1, mxy) };
                let mut out = [0i32; 8];
                unsafe {
                    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, sv);
                }
                for j in 0..8 {
                    let a = src[base + j] as i32;
                    let b = src[base + j + 1] as i32;
                    assert_eq!(out[j], 16 * a + mxy * (b - a), "mxy={mxy} j={j}");
                }
            }
        }
    }

    #[test]
    fn bilin_i16x8_avx_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = R(0xb171_516e_u64 | 1);
        for mxy in 1..16 {
            for _ in 0..5000 {
                let a: [i16; 8] = std::array::from_fn(|_| r.range(-16000, 16000) as i16);
                let b: [i16; 8] = std::array::from_fn(|_| r.range(-16000, 16000) as i16);
                let av = unsafe { _mm_loadu_si128(a.as_ptr().cast()) };
                let bv = unsafe { _mm_loadu_si128(b.as_ptr().cast()) };
                let sv = unsafe { bilin_i16x8(av, bv, mxy) };
                let mut out = [0i32; 8];
                unsafe {
                    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, sv);
                }
                for j in 0..8 {
                    let aa = a[j] as i32;
                    let bb = b[j] as i32;
                    assert_eq!(out[j], 16 * aa + mxy * (bb - aa), "mxy={mxy} j={j}");
                }
            }
        }
    }

    #[test]
    fn filter_u16x8_avx_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = R(0xfeed_face_u64 | 1);
        for _ in 0..20000 {
            let src: Vec<u16> = (0..40).map(|_| r.range(0, (1 << 12) - 1) as u16).collect();
            let f: [i8; 8] = std::array::from_fn(|_| r.range(-128, 127) as i8);
            let base = 8usize;
            let sv = unsafe { filter_u16x8(&src, base, 1, &f) };
            let mut out = [0i32; 8];
            unsafe {
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, sv);
            }
            for j in 0..8 {
                assert_eq!(out[j], filter_u16_scalar(&src, base + j, 1, &f), "j={j}");
            }
        }
    }

    #[test]
    fn filter_i16x8_avx_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = R(0xc0de_d00d_u64 | 1);
        for _ in 0..20000 {
            let src: Vec<i16> = (0..40).map(|_| r.range(-16000, 16000) as i16).collect();
            let f: [i8; 8] = std::array::from_fn(|_| r.range(-128, 127) as i8);
            let base = 8usize;
            let sv = unsafe { filter_i16x8(&src, base, 1, &f) };
            let mut out = [0i32; 8];
            unsafe {
                _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, sv);
            }
            for j in 0..8 {
                assert_eq!(out[j], filter_i16_scalar(&src, base + j, 1, &f), "j={j}");
            }
        }
    }
}
