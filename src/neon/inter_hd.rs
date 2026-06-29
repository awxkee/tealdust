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

#[inline(always)]
fn load_u16x4_i32(p: &[u16]) -> int32x4_t {
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p.as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn round_s32(v: int32x4_t, rnd: i32, shift: i32) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, vdupq_n_s32(rnd)), vdupq_n_s32(-shift))
}

#[inline]
#[target_feature(enable = "neon")]
fn store_clip_u16x4(dst: &mut [u16], v: int32x4_t, rnd: i32, shift: i32, max: uint16x4_t) {
    let v = round_s32(v, rnd, shift);
    let p = vmin_u16(vqmovun_s32(v), max);
    unsafe {
        vst1_u16(dst.as_mut_ptr(), p);
    }
}

#[inline(always)]
fn store_i16x4(dst: &mut [i16], v: int32x4_t, rnd: i32, shift: i32, bias: i32) {
    unsafe {
        let v = vsubq_s32(round_s32(v, rnd, shift), vdupq_n_s32(bias));
        vst1_s16(dst.as_mut_ptr(), vqmovn_s32(v));
    }
}

#[inline(always)]
fn load_u16x4(p: &[u16]) -> uint16x4_t {
    unsafe { vld1_u16(p.as_ptr()) }
}

#[inline(always)]
fn load_i16x4(p: &[i16]) -> int16x4_t {
    unsafe { vld1_s16(p.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_u16x4(src: &[u16], base: usize, stride: isize, f: &[i8; 8]) -> int32x4_t {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = vdupq_n_s32(0);
    for k in 0..8 {
        let idx = (base as isize + OFFSETS[k] * stride) as usize;
        let s = vreinterpret_s16_u16(load_u16x4(unsafe { src.get_unchecked(idx..) }));
        sum = vmlal_n_s16(sum, s, f[k] as i16);
    }
    sum
}

#[inline]
#[target_feature(enable = "neon")]
fn filter_i16x4(src: &[i16], base: usize, stride: isize, f: &[i8; 8]) -> int32x4_t {
    static OFFSETS: [isize; 8] = [-3isize, -2, -1, 0, 1, 2, 3, 4];
    let mut sum = vdupq_n_s32(0);
    for k in 0..8 {
        let idx = (base as isize + OFFSETS[k] * stride) as usize;
        let s = load_i16x4(unsafe { src.get_unchecked(idx..) });
        sum = vmlal_n_s16(sum, s, f[k] as i16);
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

#[inline(always)]
fn bilin_u16x4(src: &[u16], base: usize, stride: usize, mxy: i32) -> int32x4_t {
    unsafe {
        let a16 = load_u16x4(src.get_unchecked(base..));
        let b16 = load_u16x4(src.get_unchecked(base + stride..));
        let a = vreinterpretq_s32_u32(vmovl_u16(a16));
        let diff = vsub_s16(vreinterpret_s16_u16(b16), vreinterpret_s16_u16(a16));
        vmlal_n_s16(vshlq_n_s32::<4>(a), diff, mxy as i16)
    }
}

#[inline(always)]
fn bilin_i16x4(a16: int16x4_t, b16: int16x4_t, mxy: i32) -> int32x4_t {
    unsafe {
        let a = vmovl_s16(a16);
        let diff = vsub_s16(b16, a16);
        vmlal_n_s16(vshlq_n_s32::<4>(a), diff, mxy as i16)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_hbd_neon(
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
        let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
        for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
            let x = chunk_idx * 4;
            unsafe {
                let s = load_u16x4_i32(src_row.get_unchecked(x..));
                let v = vsubq_s32(vshlq_s32(s, vdupq_n_s32(ib)), vdupq_n_s32(bias));
                vst1_s16(tmp_chunk.as_mut_ptr(), vqmovn_s32(v));
            }
        }
        let processed = tmp_chunks.len() * 4;
        for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
            *tmp_px = (((src_row[x] as i32) << ib) - bias) as i16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn put_bilin_hbd_neon(
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
    let maxv = vdup_n_u16(((1 << bitdepth) - 1) as u16);
    let intermediate_rnd = (1 << ib) >> 1;
    if mx != 0 && my != 0 {
        let mid = &mut mid_scratch[..64 * (h + 1)];
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, mid_row) in mid.chunks_exact_mut(64).take(h + 1).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                store_i16x4(mid_chunk, bilin_u16x4(src_row, x, 1, mx), rnd0, sh0, 0);
            }
            let processed = mid_chunks.len() * 4;
            for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_row[x + 1] as i32;
                *mid_px = round_scalar(16 * a + mx * (b - a), rnd0, sh0) as i16;
            }
        }
        for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
            let mid_row = unsafe { mid.get_unchecked(y * 64..) };
            let mid_next_row = unsafe { mid.get_unchecked((y + 1) * 64..) };
            let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                let a = load_i16x4(unsafe { mid.get_unchecked(y * 64 + x..) });
                let b = load_i16x4(unsafe { mid.get_unchecked((y + 1) * 64 + x..) });
                let v = bilin_i16x4(a, b, my);
                store_clip_u16x4(dst_chunk, v, 1 << (3 + ib), 4 + ib, maxv);
            }
            let processed = dst_chunks.len() * 4;
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
            let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                let px = round_s32(bilin_u16x4(src_row, x, 1, mx), rnd0, sh0);
                store_clip_u16x4(dst_chunk, px, intermediate_rnd, ib, maxv);
            }
            let processed = dst_chunks.len() * 4;
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
            let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                store_clip_u16x4(
                    dst_chunk,
                    bilin_u16x4(src_row, x, src_stride, my),
                    8,
                    4,
                    maxv,
                );
            }
            let processed = dst_chunks.len() * 4;
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

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn prep_bilin_hbd_neon(
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
        let mid = &mut mid_scratch[..64 * (h + 1)];
        let sh0 = 4 - ib;
        let rnd0 = if sh0 == 0 { 0 } else { 1 << (sh0 - 1) };
        for (y, mid_row) in mid.chunks_exact_mut(64).take(h + 1).enumerate() {
            let src_row = unsafe { src.get_unchecked(y * src_stride..) };
            let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                store_i16x4(mid_chunk, bilin_u16x4(src_row, x, 1, mx), rnd0, sh0, 0);
            }
            let processed = mid_chunks.len() * 4;
            for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_row[x + 1] as i32;
                *mid_px = round_scalar(16 * a + mx * (b - a), rnd0, sh0) as i16;
            }
        }
        for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
            let mid_row = unsafe { mid.get_unchecked(y * 64..) };
            let mid_next_row = unsafe { mid.get_unchecked((y + 1) * 64..) };
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                let a = load_i16x4(unsafe { mid.get_unchecked(y * 64 + x..) });
                let b = load_i16x4(unsafe { mid.get_unchecked((y + 1) * 64 + x..) });
                let v = bilin_i16x4(a, b, my);
                store_i16x4(tmp_chunk, v, 8, 4, bias);
            }
            let processed = tmp_chunks.len() * 4;
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
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                store_i16x4(tmp_chunk, bilin_u16x4(src_row, x, 1, mx), rnd0, sh0, bias);
            }
            let processed = tmp_chunks.len() * 4;
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
            let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
            for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                let x = chunk_idx * 4;
                store_i16x4(
                    tmp_chunk,
                    bilin_u16x4(src_row, x, src_stride, my),
                    rnd0,
                    sh0,
                    bias,
                );
            }
            let processed = tmp_chunks.len() * 4;
            for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                let a = src_row[x] as i32;
                let b = src_next_row[x] as i32;
                *tmp_px = (round_scalar(16 * a + my * (b - a), rnd0, sh0) - bias) as i16;
            }
        }
    } else {
        prep_hbd_neon(tmp, tmp_stride, src, src_stride, w, h, bitdepth);
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn put_8tap_hbd_neon(
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
    let maxv = vdup_n_u16(((1 << bitdepth) - 1) as u16);
    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let tmp_h = h + 7;
            let mid = &mut mid_scratch[..64 * tmp_h];
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(64).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let src_row = unsafe { src.get_unchecked(base..) };
                let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_i16x4(mid_chunk, filter_u16x4(src_row, x, 1, &fh), rnd0, sh0, 0);
                }
                let processed = mid_chunks.len() * 4;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px = round_scalar(filter_u16_scalar(src_row, x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let sh1 = bits + ib;
            let rnd1 = (1 << sh1) >> 1;
            for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
                let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_clip_u16x4(
                        dst_chunk,
                        filter_i16x4(&mid, (y + 3) * 64 + x, 64, &fv),
                        rnd1,
                        sh1,
                        maxv,
                    );
                }
                let processed = dst_chunks.len() * 4;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = clip(
                        round_scalar(
                            filter_i16_scalar(&mid, (y + 3) * 64 + x, 64, &fv),
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
                let src_row = unsafe { src.get_unchecked(base..) };
                let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_clip_u16x4(
                        dst_chunk,
                        filter_u16x4(src_row, x, 1, &fh),
                        intermediate_rnd,
                        bits,
                        maxv,
                    );
                }
                let processed = dst_chunks.len() * 4;
                for (x, dst_px) in (processed..w).zip(dst_rem.iter_mut()) {
                    *dst_px = clip(
                        round_scalar(
                            filter_u16_scalar(src_row, x, 1, &fh),
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
                let (dst_chunks, dst_rem) = dst_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, dst_chunk) in dst_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_clip_u16x4(
                        dst_chunk,
                        filter_u16x4(src, base + x, ss, &fv),
                        (1 << bits) >> 1,
                        bits,
                        maxv,
                    );
                }
                let processed = dst_chunks.len() * 4;
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

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn prep_8tap_hbd_neon(
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
            let mid = &mut mid_scratch[..64 * tmp_h];
            let sh0 = bits - ib;
            let rnd0 = (1 << sh0) >> 1;
            for (y, mid_row) in mid.chunks_exact_mut(64).take(tmp_h).enumerate() {
                let base = (src_off as isize + (y as isize - 3) * src_stride as isize) as usize;
                let src_row = unsafe { src.get_unchecked(base..) };
                let (mid_chunks, mid_rem) = mid_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, mid_chunk) in mid_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_i16x4(mid_chunk, filter_u16x4(src_row, x, 1, &fh), rnd0, sh0, 0);
                }
                let processed = mid_chunks.len() * 4;
                for (x, mid_px) in (processed..w).zip(mid_rem.iter_mut()) {
                    *mid_px = round_scalar(filter_u16_scalar(src_row, x, 1, &fh), rnd0, sh0) as i16;
                }
            }
            let rnd1 = (1 << bits) >> 1;
            for (y, tmp_row) in tmp.chunks_exact_mut(tmp_stride).take(h).enumerate() {
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_i16x4(
                        tmp_chunk,
                        filter_i16x4(&mid, (y + 3) * 64 + x, 64, &fv),
                        rnd1,
                        bits,
                        bias,
                    );
                }
                let processed = tmp_chunks.len() * 4;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (round_scalar(
                        filter_i16_scalar(&mid, (y + 3) * 64 + x, 64, &fv),
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
                let src_row = unsafe { src.get_unchecked(base..) };
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_i16x4(tmp_chunk, filter_u16x4(src_row, x, 1, &fh), rnd0, sh0, bias);
                }
                let processed = tmp_chunks.len() * 4;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (round_scalar(filter_u16_scalar(src_row, x, 1, &fh), rnd0, sh0)
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
                let (tmp_chunks, tmp_rem) = tmp_row[..w].as_chunks_mut::<4>();
                for (chunk_idx, tmp_chunk) in tmp_chunks.iter_mut().enumerate() {
                    let x = chunk_idx * 4;
                    store_i16x4(
                        tmp_chunk,
                        filter_u16x4(src, base + x, ss, &fv),
                        rnd0,
                        sh0,
                        bias,
                    );
                }
                let processed = tmp_chunks.len() * 4;
                for (x, tmp_px) in (processed..w).zip(tmp_rem.iter_mut()) {
                    *tmp_px = (round_scalar(filter_u16_scalar(src, base + x, ss, &fv), rnd0, sh0)
                        - bias) as i16;
                }
            }
        }
        (None, None) => prep_hbd_neon(tmp, tmp_stride, &src[src_off..], src_stride, w, h, bitdepth),
    }
}
