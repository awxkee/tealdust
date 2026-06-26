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

#[inline(always)]
fn store_clip_u16x8(dst: &mut [u16], v: __m256i, rnd: i32, shift: i32, max: __m128i) {
    unsafe {
        let v = round_s32(v, rnd, shift);
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let p = _mm_min_epu16(_mm_packus_epi32(lo, hi), max);
        _mm_storeu_si128(dst.as_mut_ptr().cast(), p);
    }
}

#[inline(always)]
fn store_i16x8(dst: &mut [i16], v: __m256i, rnd: i32, shift: i32, bias: i32) {
    unsafe {
        let v = _mm256_sub_epi32(round_s32(v, rnd, shift), _mm256_set1_epi32(bias));
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_packs_epi32(lo, hi));
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

#[inline(always)]
fn mul_i16x8_n_s32(x: __m128i, k: i32) -> __m256i {
    unsafe {
        // PMADDWD gives an element-wise i16*i16 -> i32 multiply when each
        // source lane is interleaved with a zero lane: [x0,0,x1,0,...].
        let zero = _mm_setzero_si128();
        let lo = _mm_unpacklo_epi16(x, zero);
        let hi = _mm_unpackhi_epi16(x, zero);
        let xz = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
        let kz = _mm256_set1_epi32((k as i16 as u16) as i32);
        _mm256_madd_epi16(xz, kz)
    }
}

#[inline(always)]
fn filter_u16x8(src: &[u16], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = unsafe { _mm256_setzero_si256() };
    for k in 0..8 {
        let idx = (base as isize + OFFSETS[k] * stride) as usize;
        let s = load_u16x8(unsafe { src.get_unchecked(idx..) });
        sum = unsafe { _mm256_add_epi32(sum, mul_i16x8_n_s32(s, f[k] as i32)) };
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_i16x8(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> __m256i {
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
    let a16 = load_u16x8(unsafe { src.get_unchecked(base..) });
    let b16 = load_u16x8(unsafe { src.get_unchecked(base + stride..) });
    let a = _mm256_cvtepu16_epi32(a16);
    let diff = _mm_sub_epi16(b16, a16);
    _mm256_add_epi32(_mm256_slli_epi32::<4>(a), mul_i16x8_n_s32(diff, mxy))
}

#[inline(always)]
fn bilin_i16x8(a16: __m128i, b16: __m128i, mxy: i32) -> __m256i {
    unsafe {
        let a = _mm256_cvtepi16_epi32(a16);
        let diff = _mm_sub_epi16(b16, a16);
        _mm256_add_epi32(_mm256_slli_epi32::<4>(a), mul_i16x8_n_s32(diff, mxy))
    }
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
    for y in 0..h {
        let mut x = 0;
        while x + 8 <= w {
            let s = load_u16x8_i32(unsafe { src.get_unchecked(y * src_stride + x..) });
            let v = _mm256_sub_epi32(sll_s32(s, ib), _mm256_set1_epi32(bias));
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            unsafe {
                _mm_storeu_si128(
                    tmp.as_mut_ptr().add(y * tmp_stride + x) as *mut __m128i,
                    _mm_packs_epi32(lo, hi),
                );
            }
            x += 8;
        }
        while x < w {
            tmp[y * tmp_stride + x] = (((src[y * src_stride + x] as i32) << ib) - bias) as i16;
            x += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn put_bilin_hbd_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
) {
    let ib = 14 - bitdepth as i32;
    let maxv = _mm_set1_epi16(((1 << bitdepth) - 1) as i16);
    let intermediate_rnd = (1 << ib) >> 1;
    if mx != 0 && my != 0 {
        let mut mid = vec![0i16; 64 * (h + 1)];
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for y in 0..h + 1 {
            let mut x = 0;
            while x + 8 <= w {
                store_i16x8(
                    unsafe { mid.get_unchecked_mut(y * 64 + x..) },
                    bilin_u16x8(src, y * src_stride + x, 1, mx),
                    rnd0,
                    sh0,
                    0,
                );
                x += 8;
            }
            while x < w {
                let a = src[y * src_stride + x] as i32;
                let b = src[y * src_stride + x + 1] as i32;
                mid[y * 64 + x] = round_scalar(16 * a + mx * (b - a), rnd0, sh0) as i16;
                x += 1;
            }
        }
        for y in 0..h {
            let mut x = 0;
            while x + 8 <= w {
                let a = load_i16x8(unsafe { mid.get_unchecked(y * 64 + x..) });
                let b = load_i16x8(unsafe { mid.get_unchecked((y + 1) * 64 + x..) });
                let v = bilin_i16x8(a, b, my);
                store_clip_u16x8(
                    unsafe { dst.get_unchecked_mut(y * dst_stride + x..) },
                    v,
                    1 << (3 + ib),
                    4 + ib,
                    maxv,
                );
                x += 8;
            }
            while x < w {
                let a = mid[y * 64 + x] as i32;
                let b = mid[(y + 1) * 64 + x] as i32;
                dst[y * dst_stride + x] = clip(
                    round_scalar(16 * a + my * (b - a), 1 << (3 + ib), 4 + ib),
                    bitdepth,
                );
                x += 1;
            }
        }
    } else if mx != 0 {
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for y in 0..h {
            let mut x = 0;
            while x + 8 <= w {
                let px = round_s32(bilin_u16x8(src, y * src_stride + x, 1, mx), rnd0, sh0);
                store_clip_u16x8(
                    unsafe { dst.get_unchecked_mut(y * dst_stride + x..) },
                    px,
                    intermediate_rnd,
                    ib,
                    maxv,
                );
                x += 8;
            }
            while x < w {
                let a = src[y * src_stride + x] as i32;
                let b = src[y * src_stride + x + 1] as i32;
                let px = round_scalar(16 * a + mx * (b - a), rnd0, sh0);
                dst[y * dst_stride + x] = clip(round_scalar(px, intermediate_rnd, ib), bitdepth);
                x += 1;
            }
        }
    } else if my != 0 {
        for y in 0..h {
            let mut x = 0;
            while x + 8 <= w {
                store_clip_u16x8(
                    unsafe { dst.get_unchecked_mut(y * dst_stride + x..) },
                    bilin_u16x8(src, y * src_stride + x, src_stride, my),
                    8,
                    4,
                    maxv,
                );
                x += 8;
            }
            while x < w {
                let a = src[y * src_stride + x] as i32;
                let b = src[(y + 1) * src_stride + x] as i32;
                dst[y * dst_stride + x] = clip(round_scalar(16 * a + my * (b - a), 8, 4), bitdepth);
                x += 1;
            }
        }
    } else {
        for y in 0..h {
            dst[y * dst_stride..y * dst_stride + w]
                .copy_from_slice(&src[y * src_stride..y * src_stride + w]);
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn prep_bilin_hbd_avx2(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    bitdepth: u8,
) {
    let ib = 14 - bitdepth as i32;
    let bias = 8192i32;
    if mx != 0 && my != 0 {
        let mut mid = vec![0i16; 64 * (h + 1)];
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for y in 0..h + 1 {
            let mut x = 0;
            while x + 8 <= w {
                store_i16x8(
                    unsafe { mid.get_unchecked_mut(y * 64 + x..) },
                    bilin_u16x8(src, y * src_stride + x, 1, mx),
                    rnd0,
                    sh0,
                    0,
                );
                x += 8;
            }
            while x < w {
                let a = src[y * src_stride + x] as i32;
                let b = src[y * src_stride + x + 1] as i32;
                mid[y * 64 + x] = round_scalar(16 * a + mx * (b - a), rnd0, sh0) as i16;
                x += 1;
            }
        }
        for y in 0..h {
            let mut x = 0;
            while x + 8 <= w {
                let a = load_i16x8(unsafe { mid.get_unchecked(y * 64 + x..) });
                let b = load_i16x8(unsafe { mid.get_unchecked((y + 1) * 64 + x..) });
                let v = bilin_i16x8(a, b, my);
                store_i16x8(
                    unsafe { tmp.get_unchecked_mut(y * tmp_stride + x..) },
                    v,
                    8,
                    4,
                    bias,
                );
                x += 8;
            }
            while x < w {
                let a = mid[y * 64 + x] as i32;
                let b = mid[(y + 1) * 64 + x] as i32;
                tmp[y * tmp_stride + x] = (round_scalar(16 * a + my * (b - a), 8, 4) - bias) as i16;
                x += 1;
            }
        }
    } else if mx != 0 {
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for y in 0..h {
            let mut x = 0;
            while x + 8 <= w {
                store_i16x8(
                    unsafe { tmp.get_unchecked_mut(y * tmp_stride + x..) },
                    bilin_u16x8(src, y * src_stride + x, 1, mx),
                    rnd0,
                    sh0,
                    bias,
                );
                x += 8;
            }
            while x < w {
                let a = src[y * src_stride + x] as i32;
                let b = src[y * src_stride + x + 1] as i32;
                tmp[y * tmp_stride + x] =
                    (round_scalar(16 * a + mx * (b - a), rnd0, sh0) - bias) as i16;
                x += 1;
            }
        }
    } else if my != 0 {
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for y in 0..h {
            let mut x = 0;
            while x + 8 <= w {
                store_i16x8(
                    unsafe { tmp.get_unchecked_mut(y * tmp_stride + x..) },
                    bilin_u16x8(src, y * src_stride + x, src_stride, my),
                    rnd0,
                    sh0,
                    bias,
                );
                x += 8;
            }
            while x < w {
                let a = src[y * src_stride + x] as i32;
                let b = src[(y + 1) * src_stride + x] as i32;
                tmp[y * tmp_stride + x] =
                    (round_scalar(16 * a + my * (b - a), rnd0, sh0) - bias) as i16;
                x += 1;
            }
        }
    } else {
        prep_hbd_avx2(tmp, tmp_stride, src, src_stride, w, h, bitdepth);
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn put_8tap_hbd_avx2(
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
            let mut mid = vec![0i16; 64 * tmp_h];
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for y in 0..tmp_h {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let mut x = 0;
                while x + 8 <= w {
                    store_i16x8(
                        unsafe { mid.get_unchecked_mut(y * 64 + x..) },
                        filter_u16x8(src, base + x, 1, &fh),
                        rnd0,
                        sh0,
                        0,
                    );
                    x += 8;
                }
                while x < w {
                    mid[y * 64 + x] =
                        round_scalar(filter_u16_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                    x += 1;
                }
            }
            let sh1 = bits + ib;
            let rnd1 = (1 << sh1) >> 1;
            for y in 0..h {
                let mut x = 0;
                while x + 8 <= w {
                    store_clip_u16x8(
                        unsafe { dst.get_unchecked_mut(y * dst_stride + x..) },
                        filter_i16x8(&mid, (y + 3) * 64 + x, 64, &fv),
                        rnd1,
                        sh1,
                        maxv,
                    );
                    x += 8;
                }
                while x < w {
                    dst[y * dst_stride + x] = clip(
                        round_scalar(
                            filter_i16_scalar(&mid, (y + 3) * 64 + x, 64, &fv),
                            rnd1,
                            sh1,
                        ),
                        bitdepth,
                    );
                    x += 1;
                }
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                let base = src_off + y * src_stride;
                let mut x = 0;
                while x + 8 <= w {
                    store_clip_u16x8(
                        unsafe { dst.get_unchecked_mut(y * dst_stride + x..) },
                        filter_u16x8(src, base + x, 1, &fh),
                        intermediate_rnd,
                        bits,
                        maxv,
                    );
                    x += 8;
                }
                while x < w {
                    dst[y * dst_stride + x] = clip(
                        round_scalar(
                            filter_u16_scalar(src, base + x, 1, &fh),
                            intermediate_rnd,
                            bits,
                        ),
                        bitdepth,
                    );
                    x += 1;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            for y in 0..h {
                let base = src_off + y * src_stride;
                let mut x = 0;
                while x + 8 <= w {
                    store_clip_u16x8(
                        unsafe { dst.get_unchecked_mut(y * dst_stride + x..) },
                        filter_u16x8(src, base + x, ss, &fv),
                        (1 << bits) >> 1,
                        bits,
                        maxv,
                    );
                    x += 8;
                }
                while x < w {
                    dst[y * dst_stride + x] = clip(
                        round_scalar(
                            filter_u16_scalar(src, base + x, ss, &fv),
                            (1 << bits) >> 1,
                            bits,
                        ),
                        bitdepth,
                    );
                    x += 1;
                }
            }
        }
        (None, None) => {
            for y in 0..h {
                dst[y * dst_stride..y * dst_stride + w]
                    .copy_from_slice(&src[src_off + y * src_stride..src_off + y * src_stride + w]);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn prep_8tap_hbd_avx2(
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
) {
    let bits = 6 + (filter_type < 0) as i32;
    let ib = 14 - bitdepth as i32;
    let bias = 8192i32;
    let fh = crate::mc::get_h_filter(mx, filter_type, w);
    let fv = crate::mc::get_v_filter(my, filter_type, h);
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mut mid = vec![0i16; 64 * tmp_h];
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for y in 0..tmp_h {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let mut x = 0;
                while x + 8 <= w {
                    store_i16x8(
                        unsafe { mid.get_unchecked_mut(y * 64 + x..) },
                        filter_u16x8(src, base + x, 1, &fh),
                        rnd0,
                        sh0,
                        0,
                    );
                    x += 8;
                }
                while x < w {
                    mid[y * 64 + x] =
                        round_scalar(filter_u16_scalar(src, base + x, 1, &fh), rnd0, sh0) as i16;
                    x += 1;
                }
            }
            let rnd1 = (1 << bits) >> 1;
            for y in 0..h {
                let mut x = 0;
                while x + 8 <= w {
                    store_i16x8(
                        unsafe { tmp.get_unchecked_mut(y * tmp_stride + x..) },
                        filter_i16x8(&mid, (y + 3) * 64 + x, 64, &fv),
                        rnd1,
                        bits,
                        bias,
                    );
                    x += 8;
                }
                while x < w {
                    tmp[y * tmp_stride + x] = (round_scalar(
                        filter_i16_scalar(&mid, (y + 3) * 64 + x, 64, &fv),
                        rnd1,
                        bits,
                    ) - bias) as i16;
                    x += 1;
                }
            }
        }
        (Some(fh), None) => {
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for y in 0..h {
                let base = src_off + y * src_stride;
                let mut x = 0;
                while x + 8 <= w {
                    store_i16x8(
                        unsafe { tmp.get_unchecked_mut(y * tmp_stride + x..) },
                        filter_u16x8(src, base + x, 1, &fh),
                        rnd0,
                        sh0,
                        bias,
                    );
                    x += 8;
                }
                while x < w {
                    tmp[y * tmp_stride + x] =
                        (round_scalar(filter_u16_scalar(src, base + x, 1, &fh), rnd0, sh0) - bias)
                            as i16;
                    x += 1;
                }
            }
        }
        (None, Some(fv)) => {
            let ss = src_stride as isize;
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for y in 0..h {
                let base = src_off + y * src_stride;
                let mut x = 0;
                while x + 8 <= w {
                    store_i16x8(
                        unsafe { tmp.get_unchecked_mut(y * tmp_stride + x..) },
                        filter_u16x8(src, base + x, ss, &fv),
                        rnd0,
                        sh0,
                        bias,
                    );
                    x += 8;
                }
                while x < w {
                    tmp[y * tmp_stride + x] =
                        (round_scalar(filter_u16_scalar(src, base + x, ss, &fv), rnd0, sh0) - bias)
                            as i16;
                    x += 1;
                }
            }
        }
        (None, None) => prep_hbd_avx2(tmp, tmp_stride, &src[src_off..], src_stride, w, h, bitdepth),
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
    fn filter_u16x8_avx_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut r = R(0xfeed_face_u64 | 1);
        for _ in 0..20000 {
            let src: Vec<u16> = (0..40).map(|_| r.range(0, (1 << 12) - 1) as u16).collect();
            let f: [i8; 8] = std::array::from_fn(|_| r.range(-128, 127) as i8);
            let base = 8usize;
            let sv = filter_u16x8(&src, base, 1, &f);
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
            let src: Vec<i16> = (0..40).map(|_| r.range(-20000, 20000) as i16).collect();
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
