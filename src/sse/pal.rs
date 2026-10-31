/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_idx2(idx: &[u8]) -> __m128i {
    unsafe { _mm_loadu_si16(idx.as_ptr()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_idx4(idx: &[u8]) -> __m128i {
    unsafe { _mm_castps_si128(_mm_load_ss(idx.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_idx8(idx: &[u8]) -> __m128i {
    unsafe { _mm_loadl_epi64(idx.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_idx16(idx: &[u8]) -> __m128i {
    unsafe { _mm_loadu_si128(idx.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn pal8_expand16(pal: __m128i, packed: __m128i) -> __m128i {
    let mask = _mm_set1_epi8(7);
    let lo_idx = _mm_and_si128(packed, mask);
    let hi_idx = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
    let lo = _mm_shuffle_epi8(pal, lo_idx);
    let hi = _mm_shuffle_epi8(pal, hi_idx);
    _mm_unpacklo_epi8(lo, hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn pal8_expand32(pal: __m128i, packed: __m128i) -> (__m128i, __m128i) {
    let mask = _mm_set1_epi8(7);
    let lo_idx = _mm_and_si128(packed, mask);
    let hi_idx = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
    let lo = _mm_shuffle_epi8(pal, lo_idx);
    let hi = _mm_shuffle_epi8(pal, hi_idx);
    (_mm_unpacklo_epi8(lo, hi), _mm_unpackhi_epi8(lo, hi))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u8x4(dst: &mut [u8], v: __m128i) {
    unsafe { _mm_store_ss(dst.as_mut_ptr().cast(), _mm_castsi128_ps(v)) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u8x8(dst: &mut [u8], v: __m128i) {
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u8x16(dst: &mut [u8], v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn pal16_indices_from_packed8(packed: __m128i) -> __m128i {
    let mask = _mm_set1_epi8(7);
    let lo_idx = _mm_and_si128(packed, mask);
    let hi_idx = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
    _mm_unpacklo_epi8(lo_idx, hi_idx)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn pal16_shuffle8(pal: __m128i, idx: __m128i) -> __m128i {
    let doubled = _mm_add_epi8(idx, idx);
    let ctrl = _mm_add_epi8(
        _mm_unpacklo_epi8(doubled, doubled),
        _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    );
    _mm_shuffle_epi8(pal, ctrl)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn pal16_shuffle16(pal: __m128i, idx: __m128i) -> (__m128i, __m128i) {
    let doubled = _mm_add_epi8(idx, idx);
    let bias = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    (
        _mm_shuffle_epi8(pal, _mm_add_epi8(_mm_unpacklo_epi8(doubled, doubled), bias)),
        _mm_shuffle_epi8(pal, _mm_add_epi8(_mm_unpackhi_epi8(doubled, doubled), bias)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x4(dst: &mut [u16], v: __m128i) {
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x8(dst: &mut [u16], v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), v) }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn pal_pred_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    debug_assert!(pal.len() >= 8);
    debug_assert!((4..=64).contains(&w) && w.is_power_of_two());

    let pal_v = unsafe { _mm_loadu_si64(pal.as_ptr().cast()) };
    let idx_stride = w >> 1;

    for (dst_row, idx_row) in dst
        .chunks_mut(stride)
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

#[target_feature(enable = "sse4.1")]
pub(crate) fn pal_pred_hbd_sse41(
    dst: &mut [u16],
    stride: usize,
    pal: &[u16],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    debug_assert!(pal.len() >= 8);
    debug_assert!((4..=64).contains(&w) && w.is_power_of_two());

    let pal_v = unsafe { _mm_loadu_si128(pal.as_ptr().cast()) };
    let idx_stride = w >> 1;

    for (dst_row, idx_row) in dst
        .chunks_mut(stride)
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
                let idx_v = pal16_indices_from_packed8(load_idx8(idx_row));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[..8], lo);
                store_u16x8(&mut dst_row[8..16], hi);

                let idx_v = pal16_indices_from_packed8(load_idx8(&idx_row[8..]));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[16..24], lo);
                store_u16x8(&mut dst_row[24..32], hi);

                let idx_v = pal16_indices_from_packed8(load_idx8(&idx_row[16..]));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[32..40], lo);
                store_u16x8(&mut dst_row[40..48], hi);

                let idx_v = pal16_indices_from_packed8(load_idx8(&idx_row[24..]));
                let (lo, hi) = pal16_shuffle16(pal_v, idx_v);
                store_u16x8(&mut dst_row[48..56], lo);
                store_u16x8(&mut dst_row[56..], hi);
            }
            _ => crate::ipred::pal_pred(dst_row, stride, pal, idx_row, w, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 33) as u32
    }

    #[test]
    fn pal_sse41_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut seed = 3u64;
        for &w in &[4usize, 8, 16, 32, 64] {
            for &h in &[4usize, 8, 16, 32, 64] {
                for n in 2..=8u32 {
                    // Packed 4-bit indices, two per byte, below the palette size.
                    let idx: Vec<u8> = (0..w * h / 2)
                        .map(|_| (lcg(&mut seed) % n | ((lcg(&mut seed) % n) << 4)) as u8)
                        .collect();
                    let pal16: Vec<u16> = (0..8).map(|_| (lcg(&mut seed) & 1023) as u16).collect();
                    let pal8: Vec<u8> = pal16.iter().map(|&v| v as u8).collect();
                    let stride = w + 5;
                    let (mut a, mut b) = (vec![0u8; stride * h], vec![0u8; stride * h]);
                    crate::ipred::pal_pred(&mut a, stride, &pal8, &idx, w, h);
                    unsafe { pal_pred_8bpc_sse41(&mut b, stride, &pal8, &idx, w, h) };
                    assert_eq!(a, b, "8bpc {w}x{h} n={n}");
                    let (mut a, mut b) = (vec![0u16; stride * h], vec![0u16; stride * h]);
                    crate::ipred::pal_pred(&mut a, stride, &pal16, &idx, w, h);
                    unsafe { pal_pred_hbd_sse41(&mut b, stride, &pal16, &idx, w, h) };
                    assert_eq!(a, b, "hbd {w}x{h} n={n}");
                }
            }
        }
    }
}
