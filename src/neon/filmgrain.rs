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

use std::arch::aarch64::*;

#[inline]
fn round2_scalar(x: i32, shift: i32) -> i32 {
    (x + (1 << (shift - 1))) >> shift
}

#[inline]
fn iclip_scalar(v: i32, min_value: i32, max_value: i32) -> i32 {
    v.max(min_value).min(max_value)
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn blend_top_grain_row_neon(
    dst: &mut [i16],
    old: &[i16],
    grain: &[i16],
    grain_min: i32,
    grain_max: i32,
    old_w: i32,
    new_w: i32,
) {
    let n = dst.len().min(old.len()).min(grain.len());
    let old_w_v = vdupq_n_s16(old_w as i16);
    let new_w_v = vdupq_n_s16(new_w as i16);
    let round = vdupq_n_s16(16);
    let minv = vdupq_n_s16(grain_min as i16);
    let maxv = vdupq_n_s16(grain_max as i16);
    let mut x = 0usize;
    while x + 8 <= n {
        let o = unsafe { vld1q_s16(old.as_ptr().add(x)) };
        let g = unsafe { vld1q_s16(grain.as_ptr().add(x)) };
        let sum = vaddq_s16(vmulq_s16(o, old_w_v), vmulq_s16(g, new_w_v));
        let out = vminq_s16(vmaxq_s16(vshrq_n_s16(vaddq_s16(sum, round), 5), minv), maxv);
        unsafe { vst1q_s16(dst.as_mut_ptr().add(x), out) };
        x += 8;
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
#[target_feature(enable = "neon")]
fn noise_8x_i16(
    grain: int16x8_t,
    scale: int16x8_t,
    round: int32x4_t,
    neg_shift: int32x4_t,
) -> int16x8_t {
    let n_lo = vshlq_s32(
        vaddq_s32(vmull_s16(vget_low_s16(scale), vget_low_s16(grain)), round),
        neg_shift,
    );
    let n_hi = vshlq_s32(
        vaddq_s32(vmull_s16(vget_high_s16(scale), vget_high_s16(grain)), round),
        neg_shift,
    );
    vcombine_s16(vqmovn_s32(n_lo), vqmovn_s32(n_hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn apply_8bpc_vec16(
    dst: *mut u8,
    src: *const u8,
    grain: *const i16,
    scale: *const i16,
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let round = vdupq_n_s32(1 << (scaling_shift - 1));
    let neg_shift = vdupq_n_s32(-scaling_shift);
    let minv = vdupq_n_s16(min_value as i16);
    let maxv = vdupq_n_s16(max_value as i16);
    let src8 = unsafe { vld1q_u8(src) };
    let src_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src8)));
    let src_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src8)));
    let g_lo = unsafe { vld1q_s16(grain) };
    let g_hi = unsafe { vld1q_s16(grain.add(8)) };
    let s_lo = unsafe { vld1q_s16(scale) };
    let s_hi = unsafe { vld1q_s16(scale.add(8)) };
    let n_lo = noise_8x_i16(g_lo, s_lo, round, neg_shift);
    let n_hi = noise_8x_i16(g_hi, s_hi, round, neg_shift);
    let out_lo = vminq_s16(vmaxq_s16(vaddq_s16(src_lo, n_lo), minv), maxv);
    let out_hi = vminq_s16(vmaxq_s16(vaddq_s16(src_hi, n_hi), minv), maxv);
    unsafe { vst1q_u8(dst, vcombine_u8(vqmovun_s16(out_lo), vqmovun_s16(out_hi))) };
}

#[inline]
#[target_feature(enable = "neon")]
fn apply_hbd_vec4(
    dst: *mut u16,
    src: *const u16,
    grain: *const i16,
    scale: *const i32,
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let round = vdupq_n_s32(1 << (scaling_shift - 1));
    let neg_shift = vdupq_n_s32(-scaling_shift);
    let minv = vdupq_n_s32(min_value);
    let maxv = vdupq_n_s32(max_value);
    let src32 = vreinterpretq_s32_u32(vmovl_u16(unsafe { vld1_u16(src) }));
    let grain32 = vmovl_s16(unsafe { vld1_s16(grain) });
    let scale32 = unsafe { vld1q_s32(scale) };
    let noise = vshlq_s32(vaddq_s32(vmulq_s32(scale32, grain32), round), neg_shift);
    let out = vminq_s32(vmaxq_s32(vaddq_s32(src32, noise), minv), maxv);
    unsafe { vst1_u16(dst, vqmovun_s32(out)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn apply_hbd_vec8(
    dst: *mut u16,
    src: *const u16,
    grain: *const i16,
    scale: *const i32,
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let round = vdupq_n_s32(1 << (scaling_shift - 1));
    let neg_shift = vdupq_n_s32(-scaling_shift);
    let minv = vdupq_n_s32(min_value);
    let maxv = vdupq_n_s32(max_value);
    let src16 = unsafe { vld1q_u16(src) };
    let grain16 = unsafe { vld1q_s16(grain) };
    let src_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(src16)));
    let src_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(src16)));
    let grain_lo = vmovl_s16(vget_low_s16(grain16));
    let grain_hi = vmovl_s16(vget_high_s16(grain16));
    let scale_lo = unsafe { vld1q_s32(scale) };
    let scale_hi = unsafe { vld1q_s32(scale.add(4)) };
    let noise_lo = vshlq_s32(vaddq_s32(vmulq_s32(scale_lo, grain_lo), round), neg_shift);
    let noise_hi = vshlq_s32(vaddq_s32(vmulq_s32(scale_hi, grain_hi), round), neg_shift);
    let out_lo = vminq_s32(vmaxq_s32(vaddq_s32(src_lo, noise_lo), minv), maxv);
    let out_hi = vminq_s32(vmaxq_s32(vaddq_s32(src_hi, noise_hi), minv), maxv);
    unsafe { vst1q_u16(dst, vcombine_u16(vqmovun_s32(out_lo), vqmovun_s32(out_hi))) };
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn fgy_row_8bpc_neon(
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
#[target_feature(enable = "neon")]
pub(crate) fn fgy_row_hbd_neon(
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
#[target_feature(enable = "neon")]
pub(crate) fn fguv_row_8bpc_neon(
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
#[target_feature(enable = "neon")]
pub(crate) fn fguv_row_hbd_neon(
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
