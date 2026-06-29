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
fn load_idx2(idx: &[u8]) -> uint8x16_t {
    unsafe { vreinterpretq_u8_u16(vld1q_lane_u16::<2>(idx.as_ptr().cast(), vdupq_n_u16(0))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_idx4(idx: &[u8]) -> uint8x16_t {
    unsafe { vreinterpretq_u8_u32(vld1q_lane_u32::<0>(idx.as_ptr().cast(), vdupq_n_u32(0))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_idx8(idx: &[u8]) -> uint8x16_t {
    unsafe { vcombine_u8(vld1_u8(idx.as_ptr()), vdup_n_u8(0)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_idx16(idx: &[u8]) -> uint8x16_t {
    unsafe { vld1q_u8(idx.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn pal8_expand16(pal: uint8x16_t, packed: uint8x16_t) -> uint8x16_t {
    let mask = vdupq_n_u8(7);
    let lo_idx = vandq_u8(packed, mask);
    let hi_idx = vandq_u8(vshrq_n_u8::<4>(packed), mask);
    let lo = vqtbl1q_u8(pal, lo_idx);
    let hi = vqtbl1q_u8(pal, hi_idx);
    vzip1q_u8(lo, hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn pal8_expand32(pal: uint8x16_t, packed: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
    let mask = vdupq_n_u8(7);
    let lo_idx = vandq_u8(packed, mask);
    let hi_idx = vandq_u8(vshrq_n_u8::<4>(packed), mask);
    let lo = vqtbl1q_u8(pal, lo_idx);
    let hi = vqtbl1q_u8(pal, hi_idx);
    (vzip1q_u8(lo, hi), vzip2q_u8(lo, hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x4(dst: &mut [u8], v: uint8x16_t) {
    unsafe {
        vst1q_lane_u32::<0>(dst.as_mut_ptr().cast(), vreinterpretq_u32_u8(v));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x8(dst: &mut [u8], v: uint8x16_t) {
    unsafe { vst1_u8(dst.as_mut_ptr(), vget_low_u8(v)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u8x16(dst: &mut [u8], v: uint8x16_t) {
    unsafe { vst1q_u8(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn pal16_indices_from_packed8(packed: uint8x16_t) -> uint8x16_t {
    let mask = vdupq_n_u8(7);
    let lo_idx = vandq_u8(packed, mask);
    let hi_idx = vandq_u8(vshrq_n_u8::<4>(packed), mask);
    vzip1q_u8(lo_idx, hi_idx)
}

#[inline]
#[target_feature(enable = "neon")]
fn pal16_shuffle8(pal: uint8x16_t, idx: uint8x16_t) -> uint16x8_t {
    let doubled = vaddq_u8(idx, idx);
    let ctrl = vaddq_u8(
        vzip1q_u8(doubled, doubled),
        vreinterpretq_u8_u16(vdupq_n_u16(0x0100)),
    );
    vreinterpretq_u16_u8(vqtbl1q_u8(pal, ctrl))
}

#[inline]
#[target_feature(enable = "neon")]
fn pal16_shuffle16(pal: uint8x16_t, idx: uint8x16_t) -> (uint16x8_t, uint16x8_t) {
    let doubled = vaddq_u8(idx, idx);
    let bias = vreinterpretq_u8_u16(vdupq_n_u16(0x0100));
    (
        vreinterpretq_u16_u8(vqtbl1q_u8(pal, vaddq_u8(vzip1q_u8(doubled, doubled), bias))),
        vreinterpretq_u16_u8(vqtbl1q_u8(pal, vaddq_u8(vzip2q_u8(doubled, doubled), bias))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x4(dst: &mut [u16], v: uint16x8_t) {
    unsafe { vst1_u16(dst.as_mut_ptr(), vget_low_u16(v)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(dst: &mut [u16], v: uint16x8_t) {
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) }
}

#[target_feature(enable = "neon")]
pub(crate) fn pal_pred_8bpc_neon(
    dst: &mut [u8],
    stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    debug_assert!(pal.len() >= 8);
    debug_assert!((4..=64).contains(&w) && w.is_power_of_two());

    let pal_v = unsafe { vcombine_u8(vld1_u8(pal.as_ptr().cast()), vdup_n_u8(0)) };
    let idx_stride = w >> 1;

    for (dst_row, idx_row) in dst
        .chunks_exact_mut(stride)
        .zip(idx.chunks_exact(idx_stride))
        .take(h)
    {
        match w {
            4 => store_u8x4(dst_row, pal8_expand16(pal_v, load_idx2(idx_row))),
            8 => store_u8x8(dst_row, pal8_expand16(pal_v, load_idx4(idx_row))),
            16 => store_u8x16(dst_row, pal8_expand16(pal_v, load_idx8(idx_row))),
            32 => {
                let (lo, hi) = pal8_expand32(pal_v, load_idx16(idx_row));
                store_u8x16(&mut dst_row[..16], lo);
                store_u8x16(&mut dst_row[16..], hi);
            }
            64 => {
                let (lo, hi) = pal8_expand32(pal_v, load_idx16(idx_row));
                store_u8x16(&mut dst_row[..16], lo);
                store_u8x16(&mut dst_row[16..32], hi);
                let (lo, hi) = pal8_expand32(pal_v, load_idx16(&idx_row[16..]));
                store_u8x16(&mut dst_row[32..48], lo);
                store_u8x16(&mut dst_row[48..], hi);
            }
            _ => crate::ipred::pal_pred(dst_row, stride, pal, idx_row, w, 1),
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn pal_pred_hbd_neon(
    dst: &mut [u16],
    stride: usize,
    pal: &[u16],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    debug_assert!(pal.len() >= 8);
    debug_assert!((4..=64).contains(&w) && w.is_power_of_two());

    let pal_v = unsafe { vld1q_u8(pal.as_ptr().cast()) };
    let idx_stride = w >> 1;

    for (dst_row, idx_row) in dst
        .chunks_exact_mut(stride)
        .zip(idx.chunks_exact(idx_stride))
        .take(h)
    {
        match w {
            4 => store_u16x4(
                dst_row,
                pal16_shuffle8(pal_v, pal16_indices_from_packed8(load_idx2(idx_row))),
            ),
            8 => store_u16x8(
                dst_row,
                pal16_shuffle8(pal_v, pal16_indices_from_packed8(load_idx4(idx_row))),
            ),
            16 => {
                let idx_v = pal16_indices_from_packed8(load_idx8(idx_row));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[..8], lo);
                store_u16x8(&mut dst_row[8..], hi);
            }
            32 => {
                let idx_v = pal16_indices_from_packed8(load_idx8(idx_row));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[..8], lo);
                store_u16x8(&mut dst_row[8..16], hi);
                let idx_v = pal16_indices_from_packed8(load_idx8(&idx_row[8..]));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[16..24], lo);
                store_u16x8(&mut dst_row[24..], hi);
            }
            64 => {
                let (dst_chunks, _) = dst_row[..64].as_chunks_mut::<16>();
                let (idx_chunks, _) = idx_row[..32].as_chunks::<8>();
                for (dst_chunk, idx_chunk) in dst_chunks.iter_mut().zip(idx_chunks.iter()) {
                    let idx_v = pal16_indices_from_packed8(load_idx8(idx_chunk));
                    let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                    store_u16x8(&mut dst_chunk[..8], lo);
                    store_u16x8(&mut dst_chunk[8..], hi);
                }
            }
            _ => crate::ipred::pal_pred(dst_row, stride, pal, idx_row, w, 1),
        }
    }
}
