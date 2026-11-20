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
fn store_u8x16_round8_from_i32(
    dst: &mut [u8],
    lo0: int32x4_t,
    hi0: int32x4_t,
    lo1: int32x4_t,
    hi1: int32x4_t,
) {
    let rnd = vdupq_n_s32(128);
    let p0 = vqmovn_u16(vcombine_u16(
        vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(lo0, rnd))),
        vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(hi0, rnd))),
    ));
    let p1 = vqmovn_u16(vcombine_u16(
        vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(lo1, rnd))),
        vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(hi1, rnd))),
    ));
    unsafe { vst1q_u8(dst.as_mut_ptr(), vcombine_u8(p0, p1)) };
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
fn store_i16x16_round4_from_i32(
    dst: &mut [i16],
    lo0: int32x4_t,
    hi0: int32x4_t,
    lo1: int32x4_t,
    hi1: int32x4_t,
) {
    store_i16x8_round4_from_i32(dst, lo0, hi0);
    store_i16x8_round4_from_i32(unsafe { dst.get_unchecked_mut(8..) }, lo1, hi1);
}

#[inline]
#[target_feature(enable = "neon")]
fn bilin_u8x8_i16(src: &[u8], base: usize, stride: usize, mxy: i32) -> int16x8_t {
    unsafe {
        let a = vld1_u8(src.get_unchecked(base..).as_ptr());
        let b = vld1_u8(src.get_unchecked(base + stride..).as_ptr());
        let sum = vmlal_u8(
            vmull_u8(a, vdup_n_u8((16 - mxy) as u8)),
            b,
            vdup_n_u8(mxy as u8),
        );
        vreinterpretq_s16_u16(sum)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn bilin_i16x8_i32(a: int16x8_t, b: int16x8_t, mxy: i32) -> (int32x4_t, int32x4_t) {
    let c0 = (16 - mxy) as i16;
    let c1 = mxy as i16;
    (
        vmlal_n_s16(vmull_n_s16(vget_low_s16(a), c0), vget_low_s16(b), c1),
        vmlal_n_s16(vmull_n_s16(vget_high_s16(a), c0), vget_high_s16(b), c1),
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
        for (y, mid_row) in mid.chunks_mut(mid_stride).take(h + 1).enumerate() {
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
                let (lo0, hi0) = bilin_i16x8_i32(a0, b0, my);
                let (lo1, hi1) = bilin_i16x8_i32(a1, b1, my);
                store_u8x16_round8_from_i32(dst_chunk, lo0, hi0, lo1, hi1);
            }
            let x16_done = dst_chunks16.len() * 16;
            let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
            for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                let (lo, hi) = bilin_i16x8_i32(a, b, my);
                store_u8x8_round8_from_i32(dst_chunk, lo, hi);
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
        for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
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
            .chunks(src_stride)
            .zip(dst.chunks_mut(dst_stride))
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
        for (y, mid_row) in mid.chunks_mut(mid_stride).take(h + 1).enumerate() {
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
                let (lo0, hi0) = bilin_i16x8_i32(a0, b0, my);
                let (lo1, hi1) = bilin_i16x8_i32(a1, b1, my);
                store_i16x16_round4_from_i32(tmp_chunk, lo0, hi0, lo1, hi1);
            }
            let x16_done = tmp_chunks16.len() * 16;
            let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
            for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                let x = x16_done + chunk_idx * 8;
                let a = load_i16x8(unsafe { mid_row.get_unchecked(x..) });
                let b = load_i16x8(unsafe { mid_next_row.get_unchecked(x..) });
                let (lo, hi) = bilin_i16x8_i32(a, b, my);
                store_i16x8_round4_from_i32(tmp_chunk, lo, hi);
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
        for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
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
        for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(h).enumerate() {
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
fn filter_u8x8_h(src: &[u8], base: usize, f: &[i8; 8]) -> (int32x4_t, int32x4_t) {
    let s = unsafe { vld1q_u8(src.get_unchecked(base - 3..).as_ptr()) };
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);

    macro_rules! add_pair {
        ($a:literal, $b:literal) => {{
            if f[$a] != 0 || f[$b] != 0 {
                let a = vget_low_u8(vextq_u8::<$a>(s, s));
                let b = vget_low_u8(vextq_u8::<$b>(s, s));
                let zipped = vzip_u8(a, b);
                let coeff = coeff_pair_s16(f[$a], f[$b]);
                let prod_lo = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(zipped.0)), coeff);
                let prod_hi = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(zipped.1)), coeff);
                lo = vaddq_s32(lo, vpaddlq_s16(prod_lo));
                hi = vaddq_s32(hi, vpaddlq_s16(prod_hi));
            }
        }};
    }

    add_pair!(0, 1);
    add_pair!(2, 3);
    add_pair!(4, 5);
    add_pair!(6, 7);
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn coeff_pair_s16(a: i8, b: i8) -> int16x8_t {
    let packed = ((a as i16 as u16) as i32) | (((b as i16 as u16) as i32) << 16);
    vreinterpretq_s16_s32(vdupq_n_s32(packed))
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_i16x8_pair_s32(
    lo: int32x4_t,
    hi: int32x4_t,
    a: int16x8_t,
    b: int16x8_t,
    c0: i16,
    c1: i16,
) -> (int32x4_t, int32x4_t) {
    (
        vmlal_n_s16(vmlal_n_s16(lo, vget_low_s16(a), c0), vget_low_s16(b), c1),
        vmlal_n_s16(vmlal_n_s16(hi, vget_high_s16(a), c0), vget_high_s16(b), c1),
    )
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::neg_multiply)]
fn filter_u8x8_v(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> (int32x4_t, int32x4_t) {
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    macro_rules! add_pair {
        ($a:literal, $b:literal, $oa:expr, $ob:expr) => {{
            if f[$a] != 0 || f[$b] != 0 {
                let ia = (base as isize + $oa * stride) as usize;
                let ib = (base as isize + $ob * stride) as usize;
                let a = unsafe { vld1_u8(src.as_ptr().add(ia)) };
                let b = unsafe { vld1_u8(src.as_ptr().add(ib)) };
                let zipped = vzip_u8(a, b);
                let coeff = coeff_pair_s16(f[$a], f[$b]);
                let prod_lo = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(zipped.0)), coeff);
                let prod_hi = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(zipped.1)), coeff);
                lo = vaddq_s32(lo, vpaddlq_s16(prod_lo));
                hi = vaddq_s32(hi, vpaddlq_s16(prod_hi));
            }
        }};
    }
    add_pair!(0, 1, -3isize, -2isize);
    add_pair!(2, 3, -1isize, 0isize);
    add_pair!(4, 5, 1isize, 2isize);
    add_pair!(6, 7, 3isize, 4isize);
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_u8x8(src: &[u8], base: usize, stride: isize, f: &[i8; 8]) -> (int32x4_t, int32x4_t) {
    if stride == 1 {
        filter_u8x8_h(src, base, f)
    } else {
        filter_u8x8_v(src, base, stride, f)
    }
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::neg_multiply)]
fn filter_u8x16_v(
    src: &[u8],
    base: usize,
    stride: isize,
    f: &[i8; 8],
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let mut lo0 = vdupq_n_s32(0);
    let mut hi0 = vdupq_n_s32(0);
    let mut lo1 = vdupq_n_s32(0);
    let mut hi1 = vdupq_n_s32(0);
    macro_rules! add_pair {
        ($a:literal, $b:literal, $oa:expr, $ob:expr) => {{
            if f[$a] != 0 || f[$b] != 0 {
                let ia = (base as isize + $oa * stride) as usize;
                let ib = (base as isize + $ob * stride) as usize;
                let a = unsafe { vld1q_u8(src.as_ptr().add(ia)) };
                let b = unsafe { vld1q_u8(src.as_ptr().add(ib)) };
                let z0 = vzip_u8(vget_low_u8(a), vget_low_u8(b));
                let z1 = vzip_u8(vget_high_u8(a), vget_high_u8(b));
                let coeff = coeff_pair_s16(f[$a], f[$b]);
                let p00 = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(z0.0)), coeff);
                let p01 = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(z0.1)), coeff);
                let p10 = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(z1.0)), coeff);
                let p11 = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(z1.1)), coeff);
                lo0 = vaddq_s32(lo0, vpaddlq_s16(p00));
                hi0 = vaddq_s32(hi0, vpaddlq_s16(p01));
                lo1 = vaddq_s32(lo1, vpaddlq_s16(p10));
                hi1 = vaddq_s32(hi1, vpaddlq_s16(p11));
            }
        }};
    }
    add_pair!(0, 1, -3isize, -2isize);
    add_pair!(2, 3, -1isize, 0isize);
    add_pair!(4, 5, 1isize, 2isize);
    add_pair!(6, 7, 3isize, 4isize);
    (lo0, hi0, lo1, hi1)
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_u8x16(
    src: &[u8],
    base: usize,
    stride: isize,
    f: &[i8; 8],
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    if stride == 1 {
        let (lo0, hi0) = filter_u8x8_h(src, base, f);
        let (lo1, hi1) = filter_u8x8_h(src, base + 8, f);
        (lo0, hi0, lo1, hi1)
    } else {
        filter_u8x16_v(src, base, stride, f)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_i16x8_8tap(
    src: &[i16],
    base: usize,
    stride: isize,
    f: &[i8; 8],
) -> (int32x4_t, int32x4_t) {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for k in (0..8).step_by(2) {
        let c0 = f[k];
        let c1 = f[k + 1];
        if c0 == 0 && c1 == 0 {
            continue;
        }
        let idx0 = (base as isize + OFFSETS[k] * stride) as usize;
        let idx1 = (base as isize + OFFSETS[k + 1] * stride) as usize;
        let a = unsafe { vld1q_s16(src.as_ptr().add(idx0)) };
        let b = unsafe { vld1q_s16(src.as_ptr().add(idx1)) };
        (lo, hi) = madd_i16x8_pair_s32(lo, hi, a, b, c0 as i16, c1 as i16);
    }
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_i16x16_8tap(
    src: &[i16],
    base: usize,
    stride: isize,
    f: &[i8; 8],
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut lo0 = vdupq_n_s32(0);
    let mut hi0 = vdupq_n_s32(0);
    let mut lo1 = vdupq_n_s32(0);
    let mut hi1 = vdupq_n_s32(0);
    for k in (0..8).step_by(2) {
        let c0 = f[k];
        let c1 = f[k + 1];
        if c0 == 0 && c1 == 0 {
            continue;
        }
        let idx0 = (base as isize + OFFSETS[k] * stride) as usize;
        let idx1 = (base as isize + OFFSETS[k + 1] * stride) as usize;
        let a0 = unsafe { vld1q_s16(src.as_ptr().add(idx0)) };
        let b0 = unsafe { vld1q_s16(src.as_ptr().add(idx1)) };
        let a1 = unsafe { vld1q_s16(src.as_ptr().add(idx0 + 8)) };
        let b1 = unsafe { vld1q_s16(src.as_ptr().add(idx1 + 8)) };
        (lo0, hi0) = madd_i16x8_pair_s32(lo0, hi0, a0, b0, c0 as i16, c1 as i16);
        (lo1, hi1) = madd_i16x8_pair_s32(lo1, hi1, a1, b1, c0 as i16, c1 as i16);
    }
    (lo0, hi0, lo1, hi1)
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

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x16_clip_shift(
    dst: &mut [u8],
    lo0: int32x4_t,
    hi0: int32x4_t,
    lo1: int32x4_t,
    hi1: int32x4_t,
    rnd: i32,
    shift: i32,
) {
    unsafe {
        let a0 = vqmovn_s32(round_s32(lo0, rnd, shift));
        let a1 = vqmovn_s32(round_s32(hi0, rnd, shift));
        let b0 = vqmovn_s32(round_s32(lo1, rnd, shift));
        let b1 = vqmovn_s32(round_s32(hi1, rnd, shift));
        let a = vqmovun_s16(vcombine_s16(a0, a1));
        let b = vqmovun_s16(vcombine_s16(b0, b1));
        vst1q_u8(dst.as_mut_ptr(), vcombine_u8(a, b));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x16_shift(
    dst: &mut [i16],
    lo0: int32x4_t,
    hi0: int32x4_t,
    lo1: int32x4_t,
    hi1: int32x4_t,
    rnd: i32,
    shift: i32,
) {
    unsafe {
        let a0 = vqmovn_s32(round_s32(lo0, rnd, shift));
        let a1 = vqmovn_s32(round_s32(hi0, rnd, shift));
        let b0 = vqmovn_s32(round_s32(lo1, rnd, shift));
        let b1 = vqmovn_s32(round_s32(hi1, rnd, shift));
        vst1q_s16(dst.as_mut_ptr(), vcombine_s16(a0, a1));
        vst1q_s16(dst.as_mut_ptr().add(8), vcombine_s16(b0, b1));
    }
}

/// Output sample of the 16-bit 8-tap paths: rounded words in, saturated to
/// the sample type on store.
trait TapOut: Copy + Default {
    fn from_i32(v: i32) -> Self;
    /// Stores 8 words; `dst` must be valid for 8 samples.
    unsafe fn store8(dst: *mut Self, v: int16x8_t);
}

impl TapOut for i16 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as i16
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, v: int16x8_t) {
        unsafe { vst1q_s16(dst, v) }
    }
}

impl TapOut for u8 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v.clamp(0, 255) as u8
    }
    #[inline(always)]
    unsafe fn store8(dst: *mut Self, v: int16x8_t) {
        unsafe { vst1_u8(dst, vqmovun_s16(v)) }
    }
}

/// Stores the first `dst.len()` (< 8) of 8 words.
#[inline(always)]
fn store_tail<O: TapOut>(dst: &mut [O], v: int16x8_t) {
    let mut t = [O::default(); 8];
    unsafe { O::store8(t.as_mut_ptr(), v) };
    dst.copy_from_slice(&t[..dst.len()]);
}

/// Whether the 8-tap sum of 8-bit pixels plus the rounding offset always
/// fits i16, so it can be accumulated in (wrapping) 16-bit lanes.
#[inline(always)]
fn fits_i16(f: &[i8; 8], rnd: i32) -> bool {
    let pos: i32 = f.iter().map(|&c| (c as i32).max(0)).sum();
    let neg: i32 = f.iter().map(|&c| (c as i32).min(0)).sum();
    255 * pos + rnd <= i16::MAX as i32 && 255 * neg >= i16::MIN as i32
}

/// One row of a 1-D filter over 8-bit pixels with 16-bit accumulation.
/// `base` is the index of the row's first centre pixel and `stride` the tap
/// direction (1 for horizontal). `f` must satisfy `fits_i16(f, rnd)`.
#[inline]
#[target_feature(enable = "neon")]
fn tap_row_u8<O: TapOut>(
    dst: &mut [O],
    src: &[u8],
    base: usize,
    stride: isize,
    f: &[i8; 8],
    rnd: i32,
    sh: i32,
) {
    let w = dst.len();
    // First index read, and one past the last for `n` bytes loaded at `x`.
    let first = base as isize - 3 * stride;
    let end = |x: usize, n: usize| (base as isize + 4 * stride) as usize + x + n;
    let mag = f.map(|c| vdupq_n_u8(c.unsigned_abs()));
    let vrnd = vdupq_n_s16(rnd as i16);
    let vsh = vdupq_n_s16(-(sh as i16));
    let round = |v: uint16x8_t| vshlq_s16(vaddq_s16(vreinterpretq_s16_u16(v), vrnd), vsh);
    let mut x = 0;
    if first >= 0 {
        // SAFETY: each block checks its furthest read against `src`.
        unsafe {
            let p = src.as_ptr().add(base);
            let tap = |x: usize, t: usize| p.offset(x as isize + (t as isize - 3) * stride);
            while x + 16 <= w && end(x, 16) <= src.len() {
                // The lanes wrap, which is harmless as the total fits.
                let (mut lo, mut hi) = (vdupq_n_u16(0), vdupq_n_u16(0));
                for t in 0..8 {
                    let s = vld1q_u8(tap(x, t));
                    if f[t] > 0 {
                        lo = vmlal_u8(lo, vget_low_u8(s), vget_low_u8(mag[t]));
                        hi = vmlal_high_u8(hi, s, mag[t]);
                    } else if f[t] < 0 {
                        lo = vmlsl_u8(lo, vget_low_u8(s), vget_low_u8(mag[t]));
                        hi = vmlsl_high_u8(hi, s, mag[t]);
                    }
                }
                O::store8(dst.as_mut_ptr().add(x), round(lo));
                O::store8(dst.as_mut_ptr().add(x + 8), round(hi));
                x += 16;
            }
            while x < w && end(x, 8) <= src.len() {
                let mut acc = vdupq_n_u16(0);
                for t in 0..8 {
                    let s = vld1_u8(tap(x, t));
                    if f[t] > 0 {
                        acc = vmlal_u8(acc, s, vget_low_u8(mag[t]));
                    } else if f[t] < 0 {
                        acc = vmlsl_u8(acc, s, vget_low_u8(mag[t]));
                    }
                }
                if x + 8 <= w {
                    O::store8(dst.as_mut_ptr().add(x), round(acc));
                    x += 8;
                } else {
                    store_tail(&mut dst[x..], round(acc));
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
#[inline]
#[target_feature(enable = "neon")]
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
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let sh0 = bits - 4;
            let rnd0 = (1 << sh0) >> 1;
            if !fits_i16(&fh, rnd0) {
                return false;
            }
            let tmp_h = h + 7;
            let mid_stride = w.next_multiple_of(16).max(64);
            let mid = &mut mid_scratch[..mid_stride * tmp_h];
            for (y, mid_row) in mid.chunks_mut(mid_stride).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                tap_row_u8(&mut mid_row[..w], src, base, 1, &fh, rnd0, sh0);
            }
            let vrnd = vdupq_n_s32(r.hv.0);
            let vsh = vdupq_n_s32(-r.hv.1);
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let dst_row = &mut dst_row[..w];
                let mut x = 0;
                while x < w {
                    // Rows y..y+8 of `mid` exist (tmp_h = h + 7), and every
                    // block of 8 words starting below `w` lies inside a
                    // `mid_stride`-wide row (a multiple of 16).
                    let (lo, hi) =
                        filter_i16x8_8tap(mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    let v = vcombine_s16(
                        vqmovn_s32(vshlq_s32(vaddq_s32(lo, vrnd), vsh)),
                        vqmovn_s32(vshlq_s32(vaddq_s32(hi, vrnd), vsh)),
                    );
                    if x + 8 <= w {
                        unsafe { O::store8(dst_row.as_mut_ptr().add(x), v) };
                        x += 8;
                    } else {
                        store_tail(&mut dst_row[x..], v);
                        x = w;
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
            let stride = if horizontal { 1 } else { src_stride as isize };
            for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
                let base = src_off + y * src_stride;
                tap_row_u8(&mut dst_row[..w], src, base, stride, &f, rnd, sh);
            }
        }
        (None, None) => return false,
    }
    true
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
                    let (lo0, hi0, lo1, hi1) = filter_u8x16(src, base + x, 1, &fh);
                    store_i16x16_shift(mid_chunk, lo0, hi0, lo1, hi1, rnd0, sh0);
                }
                let x16_done = mid_chunks16.len() * 16;
                let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) = filter_u8x8(src, base + x, 1, &fh);
                    store_i16x8_shift(mid_chunk, lo, hi, rnd0, sh0);
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
                    let (lo0, hi0, lo1, hi1) = filter_i16x16_8tap(
                        &mid,
                        (y + 3) * mid_stride + x,
                        mid_stride as isize,
                        &fv,
                    );
                    store_u8x16_clip_shift(dst_chunk, lo0, hi0, lo1, hi1, rnd1, sh1);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) =
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    store_u8x8_clip_shift(dst_chunk, lo, hi, rnd1, sh1);
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
                    let (lo0, hi0, lo1, hi1) = filter_u8x16(src, base + x, 1, &fh);
                    store_u8x16_clip_shift(dst_chunk, lo0, hi0, lo1, hi1, intermediate_rnd, bits);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) = filter_u8x8(src, base + x, 1, &fh);
                    store_u8x8_clip_shift(dst_chunk, lo, hi, intermediate_rnd, bits);
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
                    let (lo0, hi0, lo1, hi1) = filter_u8x16(src, base + x, ss, &fv);
                    store_u8x16_clip_shift(dst_chunk, lo0, hi0, lo1, hi1, (1 << bits) >> 1, bits);
                }
                let x16_done = dst_chunks16.len() * 16;
                let (dst_chunks8, dst_rem) = dst_rem16.as_chunks_mut::<8>();
                for (chunk_idx, dst_chunk) in dst_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) = filter_u8x8(src, base + x, ss, &fv);
                    store_u8x8_clip_shift(dst_chunk, lo, hi, (1 << bits) >> 1, bits);
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
                    let (lo0, hi0, lo1, hi1) = filter_u8x16(src, base + x, 1, &fh);
                    store_i16x16_shift(mid_chunk, lo0, hi0, lo1, hi1, rnd0, sh0);
                }
                let x16_done = mid_chunks16.len() * 16;
                let (mid_chunks8, mid_rem) = mid_rem16.as_chunks_mut::<8>();
                for (chunk_idx, mid_chunk) in mid_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) = filter_u8x8(src, base + x, 1, &fh);
                    store_i16x8_shift(mid_chunk, lo, hi, rnd0, sh0);
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
                    let (lo0, hi0, lo1, hi1) = filter_i16x16_8tap(
                        &mid,
                        (y + 3) * mid_stride + x,
                        mid_stride as isize,
                        &fv,
                    );
                    store_i16x16_shift(tmp_chunk, lo0, hi0, lo1, hi1, rnd1, bits);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) =
                        filter_i16x8_8tap(&mid, (y + 3) * mid_stride + x, mid_stride as isize, &fv);
                    store_i16x8_shift(tmp_chunk, lo, hi, rnd1, bits);
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
                    let (lo0, hi0, lo1, hi1) = filter_u8x16(src, base + x, 1, &fh);
                    store_i16x16_shift(tmp_chunk, lo0, hi0, lo1, hi1, rnd0, sh0);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) = filter_u8x8(src, base + x, 1, &fh);
                    store_i16x8_shift(tmp_chunk, lo, hi, rnd0, sh0);
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
                    let (lo0, hi0, lo1, hi1) = filter_u8x16(src, base + x, ss, &fv);
                    store_i16x16_shift(tmp_chunk, lo0, hi0, lo1, hi1, rnd0, sh0);
                }
                let x16_done = tmp_chunks16.len() * 16;
                let (tmp_chunks8, tmp_rem) = tmp_rem16.as_chunks_mut::<8>();
                for (chunk_idx, tmp_chunk) in tmp_chunks8.iter_mut().enumerate() {
                    let x = x16_done + chunk_idx * 8;
                    let (lo, hi) = filter_u8x8(src, base + x, ss, &fv);
                    store_i16x8_shift(tmp_chunk, lo, hi, rnd0, sh0);
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

#[inline]
#[target_feature(enable = "neon")]
fn round_s32_warp(v: int32x4_t, rnd: i32, shift: i32) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, vdupq_n_s32(rnd)), vdupq_n_s32(-shift))
}

/// Tap `t` of the eight per-pixel warp filters at `pos + x * step`, as one
/// word per pixel: the 8 filters x 8 taps are loaded once and transposed.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn warp_tap_coeffs(pos: i32, step: i32) -> [int16x8_t; 8] {
    let f = |x: i32| -> int8x8_t {
        let taps: &[i8; 8] =
            &crate::tables::MC_WARP_FILTER[(192 + ((pos + step * x + 512) >> 10)) as usize];
        unsafe { vld1_s8(taps.as_ptr()) }
    };
    // Bytes of two filters interleaved, as tap pairs.
    let z8 = |a: int8x8_t, b: int8x8_t| {
        (
            vreinterpret_s16_s8(vzip1_s8(a, b)),
            vreinterpret_s16_s8(vzip2_s8(a, b)),
        )
    };
    let (a0, a1) = z8(f(0), f(1));
    let (b0, b1) = z8(f(2), f(3));
    let (c0, c1) = z8(f(4), f(5));
    let (d0, d1) = z8(f(6), f(7));
    // Four filters per tap, then all eight.
    let z16 = |a: int16x4_t, b: int16x4_t| {
        (
            vreinterpret_s32_s16(vzip1_s16(a, b)),
            vreinterpret_s32_s16(vzip2_s16(a, b)),
        )
    };
    let (p0, p1) = z16(a0, b0);
    let (p2, p3) = z16(a1, b1);
    let (q0, q1) = z16(c0, d0);
    let (q2, q3) = z16(c1, d1);
    let z32 = |a: int32x2_t, b: int32x2_t| {
        [
            vmovl_s8(vreinterpret_s8_s32(vzip1_s32(a, b))),
            vmovl_s8(vreinterpret_s8_s32(vzip2_s32(a, b))),
        ]
    };
    let [t0, t1] = z32(p0, q0);
    let [t2, t3] = z32(p1, q1);
    let [t4, t5] = z32(p2, q2);
    let [t6, t7] = z32(p3, q3);
    [t0, t1, t2, t3, t4, t5, t6, t7]
}

#[inline]
#[target_feature(enable = "neon")]
fn warp_horz_u8x8(src: &[u8], row_base: usize, mx: i32, alpha: i32) -> int16x8_t {
    let c = warp_tap_coeffs(mx, alpha);
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for (tap, c) in c.into_iter().enumerate() {
        let px = unsafe { vld1_u8(src.get_unchecked(row_base + tap..).as_ptr()) };
        let px = vreinterpretq_s16_u16(vmovl_u8(px));
        lo = vmlal_s16(lo, vget_low_s16(px), vget_low_s16(c));
        hi = vmlal_high_s16(hi, px, c);
    }
    let lo = round_s32_warp(lo, 4, 3);
    let hi = round_s32_warp(hi, 4, 3);
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn warp_vert_i16x8(
    mid: &[i16],
    base: usize,
    stride: usize,
    my: i32,
    gamma: i32,
) -> (int32x4_t, int32x4_t) {
    let c = warp_tap_coeffs(my, gamma);
    let mut lo = vdupq_n_s32(0);
    let mut hi = vdupq_n_s32(0);
    for (tap, c) in c.into_iter().enumerate() {
        let px = unsafe { vld1q_s16(mid.get_unchecked(base + tap * stride..).as_ptr()) };
        lo = vmlal_s16(lo, vget_low_s16(px), vget_low_s16(c));
        hi = vmlal_high_s16(hi, px, c);
    }
    (lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x8_shift_warp(dst: &mut [u8], lo: int32x4_t, hi: int32x4_t, rnd: i32, shift: i32) {
    let lo = round_s32_warp(lo, rnd, shift);
    let hi = round_s32_warp(hi, rnd, shift);
    let p16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
    unsafe { vst1_u8(dst.as_mut_ptr(), vqmovn_u16(p16)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8_shift_warp(dst: &mut [i16], lo: int32x4_t, hi: int32x4_t, rnd: i32, shift: i32) {
    let lo = round_s32_warp(lo, rnd, shift);
    let hi = round_s32_warp(hi, rnd, shift);
    unsafe {
        vst1q_s16(
            dst.as_mut_ptr(),
            vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)),
        )
    };
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn warp_affine_8x8_8bpc_neon(
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

    for (y, dst_row) in dst.chunks_mut(dst_stride).take(8).enumerate() {
        let (lo, hi) = warp_vert_i16x8(&mid, y * 8, 8, my, gamma);
        store_u8x8_shift_warp(&mut dst_row[..8], lo, hi, 1024, 11);
        my += delta;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn warp_affine_8x8t_8bpc_neon(
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

    for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(8).enumerate() {
        let (lo, hi) = warp_vert_i16x8(&mid, y * 8, 8, my, gamma);
        store_i16x8_shift_warp(&mut tmp_row[..8], lo, hi, 64, 7);
        my += delta;
    }
}
