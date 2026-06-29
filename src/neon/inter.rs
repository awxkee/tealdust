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
#[target_feature(enable = "neon")]
fn load_u8x8_i16(src: &[u8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src.as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8(src: &[i16]) -> int16x8_t {
    unsafe { vld1q_s16(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8(dst: &mut [i16], v: int16x8_t) {
    unsafe { vst1q_s16(dst.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x16_round4_from_i16(dst: &mut [u8], lo: int16x8_t, hi: int16x8_t) {
    let rnd = vdupq_n_s16(8);
    let lo = vqmovun_s16(vshrq_n_s16::<4>(vaddq_s16(lo, rnd)));
    let hi = vqmovun_s16(vshrq_n_s16::<4>(vaddq_s16(hi, rnd)));
    unsafe { vst1q_u8(dst.as_mut_ptr(), vcombine_u8(lo, hi)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x8_round8_from_i32(dst: &mut [u8], lo: int32x4_t, hi: int32x4_t) {
    let rnd = vdupq_n_s32(128);
    let lo = vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(lo, rnd)));
    let hi = vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(hi, rnd)));
    unsafe { vst1_u8(dst.as_mut_ptr(), vqmovn_u16(vcombine_u16(lo, hi))) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8_round4_from_i32(dst: &mut [i16], lo: int32x4_t, hi: int32x4_t) {
    let rnd = vdupq_n_s32(8);
    unsafe {
        vst1q_s16(
            dst.as_mut_ptr(),
            vcombine_s16(
                vqmovn_s32(vshrq_n_s32::<4>(vaddq_s32(lo, rnd))),
                vqmovn_s32(vshrq_n_s32::<4>(vaddq_s32(hi, rnd))),
            ),
        )
    };
}

#[inline]
#[target_feature(enable = "neon")]
fn bilin_u8x8_i16(src: &[u8], base: usize, stride: usize, mxy: i32) -> int16x8_t {
    let a = load_u8x8_i16(unsafe { src.get_unchecked(base..) });
    let b = load_u8x8_i16(unsafe { src.get_unchecked(base + stride..) });
    vmlaq_n_s16(vshlq_n_s16::<4>(a), vsubq_s16(b, a), mxy as i16)
}

#[inline]
#[target_feature(enable = "neon")]
fn bilin_i16x8_i32(a: int16x8_t, b: int16x8_t, mxy: i32) -> (int32x4_t, int32x4_t) {
    let a0 = vmovl_s16(vget_low_s16(a));
    let a1 = vmovl_s16(vget_high_s16(a));
    let b0 = vmovl_s16(vget_low_s16(b));
    let b1 = vmovl_s16(vget_high_s16(b));
    (
        vmlaq_n_s32(vshlq_n_s32::<4>(a0), vsubq_s32(b0, a0), mxy),
        vmlaq_n_s32(vshlq_n_s32::<4>(a1), vsubq_s32(b1, a1), mxy),
    )
}

#[inline(always)]
fn bilin_scalar(a: i32, b: i32, mxy: i32) -> i32 {
    16 * a + mxy * (b - a)
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_8bpc_neon(
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
            let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let lo = bilin_u8x8_i16(src_row, x, 1, mx);
                let hi = bilin_u8x8_i16(src_row, x + 8, 1, mx);
                store_i16x8(mid_chunk, lo);
                store_i16x8(unsafe { mid_chunk.get_unchecked_mut(8..) }, hi);
            }
            let x16_done = mid_chunks16.len() * 16;
            let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
            for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let v = bilin_u8x8_i16(src_row, x, 1, mx);
                store_i16x8(mid_chunk, v);
            }
            let processed = x16_done + mid_chunks8.len() * 8;
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
                let (lo, hi) = bilin_i16x8_i32(a, b, my);
                store_u8x8_round8_from_i32(dst_chunk, lo, hi);
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
                let lo = bilin_u8x8_i16(src_row, x, 1, mx);
                let hi = bilin_u8x8_i16(src_row, x + 8, 1, mx);
                store_u8x16_round4_from_i16(dst_chunk, lo, hi);
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
                let lo = bilin_u8x8_i16(src_row, x, src_stride, my);
                let hi = bilin_u8x8_i16(src_row, x + 8, src_stride, my);
                store_u8x16_round4_from_i16(dst_chunk, lo, hi);
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

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin_8bpc_neon(
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
            let (mid_chunks16, mid_rem16) = mid_row[..w].as_chunks_mut::<16>();
            for (chunk_idx, mid_chunk) in mid_chunks16.iter_mut().enumerate() {
                let x = chunk_idx * 16;
                let lo = bilin_u8x8_i16(src_row, x, 1, mx);
                let hi = bilin_u8x8_i16(src_row, x + 8, 1, mx);
                store_i16x8(mid_chunk, lo);
                store_i16x8(unsafe { mid_chunk.get_unchecked_mut(8..) }, hi);
            }
            let x16_done = mid_chunks16.len() * 16;
            let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
            for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let v = bilin_u8x8_i16(src_row, x, 1, mx);
                store_i16x8(mid_chunk, v);
            }
            let processed = x16_done + mid_chunks8.len() * 8;
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
                let (lo, hi) = bilin_i16x8_i32(a, b, my);
                store_i16x8_round4_from_i32(tmp_chunk, lo, hi);
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
                let lo = bilin_u8x8_i16(src_row, x, 1, mx);
                let hi = bilin_u8x8_i16(src_row, x + 8, 1, mx);
                store_i16x8(tmp_chunk, lo);
                store_i16x8(unsafe { tmp_chunk.get_unchecked_mut(8..) }, hi);
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
                let lo = bilin_u8x8_i16(src_row, x, src_stride, my);
                let hi = bilin_u8x8_i16(src_row, x + 8, src_stride, my);
                store_i16x8(tmp_chunk, lo);
                store_i16x8(unsafe { tmp_chunk.get_unchecked_mut(8..) }, hi);
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
                let lo = vshlq_n_s16::<4>(load_u8x8_i16(unsafe { src_row.get_unchecked(x..) }));
                let hi = vshlq_n_s16::<4>(load_u8x8_i16(unsafe { src_row.get_unchecked(x + 8..) }));
                store_i16x8(tmp_chunk, lo);
                store_i16x8(unsafe { tmp_chunk.get_unchecked_mut(8..) }, hi);
            }
            let processed = tmp_chunks.len() * 16;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                *tmp_px = (src_row[x] as i16) << 4;
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_u8x8(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> (int32x4_t, int32x4_t) {
    let offsets = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for k in 0..8 {
        let idx = (base as isize + offsets[k] * stride) as usize;
        let s = unsafe { vld1_u8(src.as_ptr().add(idx)) };
        let s = vreinterpretq_s16_u16(vmovl_u8(s));
        lo = vmlal_n_s16(lo, vget_low_s16(s), f[k] as i16);
        hi = vmlal_n_s16(hi, vget_high_s16(s), f[k] as i16);
    }
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_i16x8_8tap(
    src: &[i16],
    base: usize,
    stride: isize,
    f: &[i8; 8],
) -> (int32x4_t, int32x4_t) {
    let offsets = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for k in 0..8 {
        let idx = (base as isize + offsets[k] * stride) as usize;
        let s = unsafe { vld1q_s16(src.as_ptr().add(idx)) };
        lo = vmlal_n_s16(lo, vget_low_s16(s), f[k] as i16);
        hi = vmlal_n_s16(hi, vget_high_s16(s), f[k] as i16);
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
#[target_feature(enable = "neon")]
fn round_s32(v: int32x4_t, rnd: i32, shift: i32) -> int32x4_t {
    let v = vaddq_s32(v, vdupq_n_s32(rnd));
    if shift == 0 {
        v
    } else {
        vshlq_s32(v, vdupq_n_s32(-shift))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x8_clip_shift(dst: &mut [u8], lo: int32x4_t, hi: int32x4_t, rnd: i32, shift: i32) {
    unsafe {
        let lo = vqmovn_s32(round_s32(lo, rnd, shift));
        let hi = vqmovn_s32(round_s32(hi, rnd, shift));
        vst1_u8(dst.as_mut_ptr(), vqmovun_s16(vcombine_s16(lo, hi)));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8_shift(dst: &mut [i16], lo: int32x4_t, hi: int32x4_t, rnd: i32, shift: i32) {
    unsafe {
        let lo = vqmovn_s32(round_s32(lo, rnd, shift));
        let hi = vqmovn_s32(round_s32(hi, rnd, shift));
        vst1q_s16(dst.as_mut_ptr(), vcombine_s16(lo, hi));
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_8bpc_neon(
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
                    let (lo, hi) = filter_u8x8(src_row, x, 1, &fh);
                    store_i16x8_shift(mid_chunk, lo, hi, rnd0, sh0);
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
                    let (lo, hi) =
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    store_u8x8_clip_shift(dst_chunk, lo, hi, rnd1, sh1);
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
                    let (lo, hi) = filter_u8x8(src_row, x, 1, &fh);
                    store_u8x8_clip_shift(dst_chunk, lo, hi, intermediate_rnd, bits);
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
                    let (lo, hi) = filter_u8x8(src, base + x, ss, &fv);
                    store_u8x8_clip_shift(dst_chunk, lo, hi, (1 << bits) >> 1, bits);
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

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap_8bpc_neon(
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
                    let (lo, hi) = filter_u8x8(src_row, x, 1, &fh);
                    store_i16x8_shift(mid_chunk, lo, hi, rnd0, sh0);
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
                    let (lo, hi) =
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    store_i16x8_shift(tmp_chunk, lo, hi, rnd1, bits);
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
                    let (lo, hi) = filter_u8x8(src_row, x, 1, &fh);
                    store_i16x8_shift(tmp_chunk, lo, hi, rnd0, sh0);
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
                    let (lo, hi) = filter_u8x8(src, base + x, ss, &fv);
                    store_i16x8_shift(tmp_chunk, lo, hi, rnd0, sh0);
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
                let (tmp_chunks16, tmp_rem16) = tmp_row[..w].as_chunks_mut::<16>();
                for (chunk_idx, tmp_chunk) in tmp_chunks16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let lo = load_u8x8_i16(unsafe { src_row.get_unchecked(x..) });
                    let hi = load_u8x8_i16(unsafe { src_row.get_unchecked(x + 8..) });
                    store_i16x8(tmp_chunk, vshlq_n_s16::<4>(lo));
                    store_i16x8(
                        unsafe { tmp_chunk.get_unchecked_mut(8..) },
                        vshlq_n_s16::<4>(hi),
                    );
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let v = load_u8x8_i16(unsafe { src_row.get_unchecked(x..) });
                    store_i16x8(tmp_chunk, vshlq_n_s16::<4>(v));
                }
                let processed = x16_done + tmp_chunks8.len() * 8;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (src_row[x] as i16) << 4;
                }
            }
        }
    }
}
