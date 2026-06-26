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
fn round2_scalar(x: i32, shift: i32) -> i32 {
    (x + (1 << (shift - 1))) >> shift
}

#[inline]
fn iclip_scalar(v: i32, min_value: i32, max_value: i32) -> i32 {
    v.max(min_value).min(max_value)
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn blend_top_grain_row_avx2(
    dst: &mut [i16],
    old: &[i16],
    grain: &[i16],
    grain_min: i32,
    grain_max: i32,
    old_w: i32,
    new_w: i32,
) {
    let n = dst.len().min(old.len()).min(grain.len());
    let old_w_i16 = _mm256_set1_epi16(old_w as i16);
    let new_w_i16 = _mm256_set1_epi16(new_w as i16);
    let round = _mm256_set1_epi16(16);
    let minv = _mm256_set1_epi16(grain_min as i16);
    let maxv = _mm256_set1_epi16(grain_max as i16);
    let mut x = 0usize;
    while x + 16 <= n {
        let o = unsafe { _mm256_loadu_si256(old.as_ptr().add(x) as *const __m256i) };
        let g = unsafe { _mm256_loadu_si256(grain.as_ptr().add(x) as *const __m256i) };
        let sum = _mm256_add_epi16(
            _mm256_mullo_epi16(o, old_w_i16),
            _mm256_mullo_epi16(g, new_w_i16),
        );
        let out = _mm256_min_epi16(
            _mm256_max_epi16(_mm256_srai_epi16::<5>(_mm256_add_epi16(sum, round)), minv),
            maxv,
        );
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(x) as *mut __m256i, out) };
        x += 16;
    }
    while x < n {
        let v = ((old[x] as i32 * old_w + grain[x] as i32 * new_w + 16) >> 5)
            .max(grain_min)
            .min(grain_max);
        dst[x] = v as i16;
        x += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn srai_epi32_runtime(v: __m128i, shift: i32) -> __m128i {
    match shift {
        8 => _mm_srai_epi32::<8>(v),
        9 => _mm_srai_epi32::<9>(v),
        10 => _mm_srai_epi32::<10>(v),
        11 => _mm_srai_epi32::<11>(v),
        _ => _mm_sra_epi32(v, _mm_cvtsi32_si128(shift)),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn srai_epi32x8_runtime(v: __m256i, shift: i32) -> __m256i {
    match shift {
        8 => _mm256_srai_epi32::<8>(v),
        9 => _mm256_srai_epi32::<9>(v),
        10 => _mm256_srai_epi32::<10>(v),
        11 => _mm256_srai_epi32::<11>(v),
        _ => _mm256_sra_epi32(v, _mm_cvtsi32_si128(shift)),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn noise_8x_i16(grain: __m128i, scale: __m128i, round: __m128i, scaling_shift: i32) -> __m128i {
    let g_lo = _mm_cvtepi16_epi32(grain);
    let s_lo = _mm_cvtepi16_epi32(scale);
    let g_hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(grain));
    let s_hi = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(scale));
    let n_lo = srai_epi32_runtime(
        _mm_add_epi32(_mm_mullo_epi32(s_lo, g_lo), round),
        scaling_shift,
    );
    let n_hi = srai_epi32_runtime(
        _mm_add_epi32(_mm_mullo_epi32(s_hi, g_hi), round),
        scaling_shift,
    );
    _mm_packs_epi32(n_lo, n_hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply_8bpc_vec16(
    dst: *mut u8,
    src: *const u8,
    grain: *const i16,
    scale: *const i16,
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let round = _mm_set1_epi32(1 << (scaling_shift - 1));
    let minv = _mm_set1_epi16(min_value as i16);
    let maxv = _mm_set1_epi16(max_value as i16);

    let src8 = unsafe { _mm_loadu_si128(src as *const __m128i) };
    let src_lo = _mm_cvtepu8_epi16(src8);
    let src_hi = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(src8));
    let g_lo = unsafe { _mm_loadu_si128(grain as *const __m128i) };
    let g_hi = unsafe { _mm_loadu_si128(grain.add(8) as *const __m128i) };
    let s_lo = unsafe { _mm_loadu_si128(scale as *const __m128i) };
    let s_hi = unsafe { _mm_loadu_si128(scale.add(8) as *const __m128i) };

    let n_lo = noise_8x_i16(g_lo, s_lo, round, scaling_shift);
    let n_hi = noise_8x_i16(g_hi, s_hi, round, scaling_shift);
    let out_lo = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(src_lo, n_lo), minv), maxv);
    let out_hi = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(src_hi, n_hi), minv), maxv);
    let out = _mm_packus_epi16(out_lo, out_hi);
    unsafe { _mm_storeu_si128(dst as *mut __m128i, out) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply_hbd_vec4(
    dst: *mut u16,
    src: *const u16,
    grain: *const i16,
    scale: *const i32,
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let round = _mm_set1_epi32(1 << (scaling_shift - 1));
    let minv = _mm_set1_epi32(min_value);
    let maxv = _mm_set1_epi32(max_value);

    let src4 = unsafe { _mm_loadl_epi64(src as *const __m128i) };
    let src32 = _mm_cvtepu16_epi32(src4);
    let grain16 = unsafe { _mm_loadl_epi64(grain as *const __m128i) };
    let grain32 = _mm_cvtepi16_epi32(grain16);
    let scale32 = unsafe { _mm_loadu_si128(scale as *const __m128i) };
    let noise = srai_epi32_runtime(
        _mm_add_epi32(_mm_mullo_epi32(scale32, grain32), round),
        scaling_shift,
    );
    let out32 = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(src32, noise), minv), maxv);
    let out16 = _mm_packus_epi32(out32, out32);
    unsafe { _mm_storel_epi64(dst as *mut __m128i, out16) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply_hbd_vec8(
    dst: *mut u16,
    src: *const u16,
    grain: *const i16,
    scale: *const i32,
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let round = _mm256_set1_epi32(1 << (scaling_shift - 1));
    let minv = _mm256_set1_epi32(min_value);
    let maxv = _mm256_set1_epi32(max_value);

    let src8 = unsafe { _mm_loadu_si128(src as *const __m128i) };
    let src32 = _mm256_cvtepu16_epi32(src8);
    let grain16 = unsafe { _mm_loadu_si128(grain as *const __m128i) };
    let grain32 = _mm256_cvtepi16_epi32(grain16);
    let scale32 = unsafe { _mm256_loadu_si256(scale as *const __m256i) };
    let noise = srai_epi32x8_runtime(
        _mm256_add_epi32(_mm256_mullo_epi32(scale32, grain32), round),
        scaling_shift,
    );
    let out32 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(src32, noise), minv), maxv);
    let out16 = _mm_packus_epi32(
        _mm256_castsi256_si128(out32),
        _mm256_extracti128_si256::<1>(out32),
    );
    unsafe { _mm_storeu_si128(dst as *mut __m128i, out16) };
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn fgy_row_8bpc_avx2(
    dst: &mut [u8],
    src: &[u8],
    grain: &[i16],
    scaling: &[u8; 256],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    let mut x = 0usize;
    while x + 16 <= n {
        let mut scale = [0i16; 16];
        for i in 0..16 {
            scale[i] = scaling[unsafe { *src.get_unchecked(x + i) } as usize] as i16;
        }
        apply_8bpc_vec16(
            unsafe { dst.as_mut_ptr().add(x) },
            unsafe { src.as_ptr().add(x) },
            unsafe { grain.as_ptr().add(x) },
            scale.as_ptr(),
            scaling_shift,
            min_value,
            max_value,
        );
        x += 16;
    }
    while x < n {
        let s = src[x] as i32;
        let noise = round2_scalar(scaling[s as usize] as i32 * grain[x] as i32, scaling_shift);
        dst[x] = iclip_scalar(s + noise, min_value, max_value) as u8;
        x += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn fgy_row_hbd_avx2(
    dst: &mut [u16],
    src: &[u16],
    grain: &[i16],
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    let mut x = 0usize;
    while x + 8 <= n {
        let mut scale = [0i32; 8];
        for i in 0..8 {
            scale[i] = scaling[unsafe { *src.get_unchecked(x + i) } as usize] as i32;
        }
        apply_hbd_vec8(
            unsafe { dst.as_mut_ptr().add(x) },
            unsafe { src.as_ptr().add(x) },
            unsafe { grain.as_ptr().add(x) },
            scale.as_ptr(),
            scaling_shift,
            min_value,
            max_value,
        );
        x += 8;
    }
    while x + 4 <= n {
        let mut scale = [0i32; 4];
        for i in 0..4 {
            scale[i] = scaling[unsafe { *src.get_unchecked(x + i) } as usize] as i32;
        }
        apply_hbd_vec4(
            unsafe { dst.as_mut_ptr().add(x) },
            unsafe { src.as_ptr().add(x) },
            unsafe { grain.as_ptr().add(x) },
            scale.as_ptr(),
            scaling_shift,
            min_value,
            max_value,
        );
        x += 4;
    }
    while x < n {
        let s = src[x] as i32;
        let noise = round2_scalar(scaling[s as usize] as i32 * grain[x] as i32, scaling_shift);
        dst[x] = iclip_scalar(s + noise, min_value, max_value) as u16;
        x += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn fguv_row_8bpc_avx2(
    dst: &mut [u8],
    src: &[u8],
    grain: &[i16],
    luma: &[u8],
    cx_base: usize,
    sx: usize,
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
    uv_luma_mult: i32,
    uv_mult: i32,
    uv_offset: i32,
    chroma_scaling_from_luma: bool,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    let mut x = 0usize;
    while x + 16 <= n {
        let mut scale = [0i16; 16];
        for i in 0..16 {
            let lx = (cx_base + x + i) << sx;
            let l = unsafe { *luma.get_unchecked(lx) } as i32;
            let avg = if sx != 0 {
                (l + unsafe { *luma.get_unchecked(lx + 1) } as i32 + 1) >> 1
            } else {
                l
            };
            let val = if !chroma_scaling_from_luma {
                iclip_scalar(
                    ((avg * uv_luma_mult + unsafe { *src.get_unchecked(x + i) } as i32 * uv_mult)
                        >> 6)
                        + uv_offset,
                    0,
                    255,
                ) as usize
            } else {
                avg as usize
            };
            scale[i] = scaling[val] as i16;
        }
        apply_8bpc_vec16(
            unsafe { dst.as_mut_ptr().add(x) },
            unsafe { src.as_ptr().add(x) },
            unsafe { grain.as_ptr().add(x) },
            scale.as_ptr(),
            scaling_shift,
            min_value,
            max_value,
        );
        x += 16;
    }
    while x < n {
        let lx = (cx_base + x) << sx;
        let l = luma[lx] as i32;
        let avg = if sx != 0 {
            (l + luma[lx + 1] as i32 + 1) >> 1
        } else {
            l
        };
        let val = if !chroma_scaling_from_luma {
            iclip_scalar(
                ((avg * uv_luma_mult + src[x] as i32 * uv_mult) >> 6) + uv_offset,
                0,
                255,
            ) as usize
        } else {
            avg as usize
        };
        let noise = round2_scalar(scaling[val] as i32 * grain[x] as i32, scaling_shift);
        dst[x] = iclip_scalar(src[x] as i32 + noise, min_value, max_value) as u8;
        x += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn fguv_row_hbd_avx2(
    dst: &mut [u16],
    src: &[u16],
    grain: &[i16],
    luma: &[u16],
    cx_base: usize,
    sx: usize,
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
    uv_luma_mult: i32,
    uv_mult: i32,
    uv_offset_scaled: i32,
    chroma_scaling_from_luma: bool,
    bitdepth_max: i32,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    let mut x = 0usize;
    while x + 8 <= n {
        let mut scale = [0i32; 8];
        for i in 0..8 {
            let lx = (cx_base + x + i) << sx;
            let l = unsafe { *luma.get_unchecked(lx) } as i32;
            let avg = if sx != 0 {
                (l + unsafe { *luma.get_unchecked(lx + 1) } as i32 + 1) >> 1
            } else {
                l
            };
            let val = if !chroma_scaling_from_luma {
                iclip_scalar(
                    ((avg * uv_luma_mult + unsafe { *src.get_unchecked(x + i) } as i32 * uv_mult)
                        >> 6)
                        + uv_offset_scaled,
                    0,
                    bitdepth_max,
                ) as usize
            } else {
                avg as usize
            };
            scale[i] = scaling[val] as i32;
        }
        apply_hbd_vec8(
            unsafe { dst.as_mut_ptr().add(x) },
            unsafe { src.as_ptr().add(x) },
            unsafe { grain.as_ptr().add(x) },
            scale.as_ptr(),
            scaling_shift,
            min_value,
            max_value,
        );
        x += 8;
    }
    while x + 4 <= n {
        let mut scale = [0i32; 4];
        for i in 0..4 {
            let lx = (cx_base + x + i) << sx;
            let l = unsafe { *luma.get_unchecked(lx) } as i32;
            let avg = if sx != 0 {
                (l + unsafe { *luma.get_unchecked(lx + 1) } as i32 + 1) >> 1
            } else {
                l
            };
            let val = if !chroma_scaling_from_luma {
                iclip_scalar(
                    ((avg * uv_luma_mult + unsafe { *src.get_unchecked(x + i) } as i32 * uv_mult)
                        >> 6)
                        + uv_offset_scaled,
                    0,
                    bitdepth_max,
                ) as usize
            } else {
                avg as usize
            };
            scale[i] = scaling[val] as i32;
        }
        apply_hbd_vec4(
            unsafe { dst.as_mut_ptr().add(x) },
            unsafe { src.as_ptr().add(x) },
            unsafe { grain.as_ptr().add(x) },
            scale.as_ptr(),
            scaling_shift,
            min_value,
            max_value,
        );
        x += 4;
    }
    while x < n {
        let lx = (cx_base + x) << sx;
        let l = luma[lx] as i32;
        let avg = if sx != 0 {
            (l + luma[lx + 1] as i32 + 1) >> 1
        } else {
            l
        };
        let val = if !chroma_scaling_from_luma {
            iclip_scalar(
                ((avg * uv_luma_mult + src[x] as i32 * uv_mult) >> 6) + uv_offset_scaled,
                0,
                bitdepth_max,
            ) as usize
        } else {
            avg as usize
        };
        let noise = round2_scalar(scaling[val] as i32 * grain[x] as i32, scaling_shift);
        dst[x] = iclip_scalar(src[x] as i32 + noise, min_value, max_value) as u16;
        x += 1;
    }
}
