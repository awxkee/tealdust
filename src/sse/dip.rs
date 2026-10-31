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

//! Data-driven intra prediction (DIP) for SSE4.1: the 11-tap weighted sums
//! of the reduced edge are computed four (or two) grid positions at a time,
//! then the sparse grid is upsampled horizontally and vertically.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::dip_tables::DIP_WEIGHTS;
use crate::intops::ulog2;

/// Pixel storage for the DIP kernel (u8 or u16).
trait DipPixel: Copy + Into<i32> {
    fn from_i32(v: i32) -> Self;
}

impl DipPixel for u8 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
}

impl DipPixel for u16 {
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x4_i32(p: &[u16]) -> __m128i {
    unsafe { _mm_cvtepu16_epi32(_mm_loadl_epi64(p.as_ptr().cast())) }
}

/// Accumulates `inp[base..base + 4]` against four weight rows; lane `k` gets
/// the partial dot product of row `k`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn dot4_block4(acc: __m128i, inp: &[i32; 11], w: [&[u16; 11]; 4], base: usize) -> __m128i {
    let r0 = load_u16x4_i32(&w[0][base..]);
    let r1 = load_u16x4_i32(&w[1][base..]);
    let r2 = load_u16x4_i32(&w[2][base..]);
    let r3 = load_u16x4_i32(&w[3][base..]);

    let t01 = _mm_unpacklo_epi32(r0, r1);
    let t23 = _mm_unpacklo_epi32(r2, r3);
    let c0 = _mm_unpacklo_epi64(t01, t23);
    let c1 = _mm_unpackhi_epi64(t01, t23);
    let t01 = _mm_unpackhi_epi32(r0, r1);
    let t23 = _mm_unpackhi_epi32(r2, r3);
    let c2 = _mm_unpacklo_epi64(t01, t23);
    let c3 = _mm_unpackhi_epi64(t01, t23);

    let mut acc = _mm_add_epi32(acc, _mm_mullo_epi32(c0, _mm_set1_epi32(inp[base])));
    acc = _mm_add_epi32(acc, _mm_mullo_epi32(c1, _mm_set1_epi32(inp[base + 1])));
    acc = _mm_add_epi32(acc, _mm_mullo_epi32(c2, _mm_set1_epi32(inp[base + 2])));
    _mm_add_epi32(acc, _mm_mullo_epi32(c3, _mm_set1_epi32(inp[base + 3])))
}

/// Four 11-tap dot products, finished to `((s + 2048) >> 12) - in_sum`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn dot4(inp: &[i32; 11], w: [&[u16; 11]; 4], in_sum: i32) -> __m128i {
    let mut acc = dot4_block4(_mm_setzero_si128(), inp, w, 0);
    acc = dot4_block4(acc, inp, w, 4);
    for i in 8..11 {
        let wi = _mm_setr_epi32(
            w[0][i] as i32,
            w[1][i] as i32,
            w[2][i] as i32,
            w[3][i] as i32,
        );
        acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_set1_epi32(inp[i]), wi));
    }
    _mm_sub_epi32(
        _mm_srai_epi32::<12>(_mm_add_epi32(acc, _mm_set1_epi32(2048))),
        _mm_set1_epi32(in_sum),
    )
}

#[inline(always)]
fn dot1(inp: &[i32; 11], w: &[u16; 11], in_sum: i32) -> i32 {
    let s: i32 = inp.iter().zip(w).map(|(&a, &b)| a * b as i32).sum();
    ((s + 2048) >> 12) - in_sum
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn dip_impl<P: DipPixel>(
    dst: &mut [P],
    stride: usize,
    tl: &[P],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
    max: i32,
) {
    let trans = (mode & 16) != 0;
    let wd = width >> 2;
    let hd = height >> 2;
    let wl2 = ulog2(wd as u32);
    let hl2 = ulog2(hd as u32);
    let wrnd = (width >> 3) as i32;
    let hrnd = (height >> 3) as i32;
    let i_t: usize = if trans { 5 } else { 1 };
    let i_l: usize = if trans { 1 } else { 5 };
    let px = |i: usize| -> i32 { tl[i].into() };

    // Reduced edge: top-left, four top and four left averages, plus the
    // top-right and bottom-left averages.
    let mut inp = [0i32; 11];
    inp[0] = px(o);
    for i in 0..4 {
        let sum: i32 = (0..wd).map(|k| px(o + 1 + i * wd + k)).sum();
        inp[i_t + i] = (sum + wrnd) >> wl2;
        let sum: i32 = (0..hd).map(|k| px(o - 1 - i * hd - k)).sum();
        inp[i_l + i] = (sum + hrnd) >> hl2;
    }
    let sum: i32 = (0..wd).map(|x| px(o + x + width + 1)).sum();
    inp[if trans { 10 } else { 9 }] = (sum + wrnd) >> wl2;
    let sum: i32 = (0..hd).map(|y| px(o - (y + height + 1))).sum();
    inp[if trans { 9 } else { 10 }] = (sum + hrnd) >> hl2;
    let in_sum: i32 = inp.iter().sum();

    let mut uwl2 = wl2 - 1;
    let mut dwl2 = 0i32;
    if uwl2 < 0 {
        dwl2 = -uwl2;
        uwl2 = 0;
    }
    let mut uhl2 = hl2 - 1;
    let mut dhl2 = 0i32;
    if uhl2 < 0 {
        dhl2 = -uhl2;
        uhl2 = 0;
    }
    let step_x = 1usize << uwl2;
    let step_y = 1usize << uhl2;
    let dw = 1usize << dwl2;
    let dh = 1usize << dhl2;
    let grid_w = 8usize >> dwl2;
    let grid_h = 8usize >> dhl2;
    let weights = &DIP_WEIGHTS[(mode & 7) as usize];
    let widx = |ix: usize, iy: usize| if trans { ix * 8 + iy } else { iy * 8 + ix };
    let clampv = _mm_set1_epi32(max);
    let zero = _mm_setzero_si128();

    // Sparse grid: the bottom-right sample of each step_x x step_y cell.
    let mut y = step_y - 1;
    for gy in 0..grid_h {
        let iy = gy * dh;
        let row = &mut dst[y * stride..y * stride + width];
        let mut gx = 0usize;
        while gx + 4 <= grid_w {
            let w = [0, 1, 2, 3].map(|k| &weights[widx((gx + k) * dw, iy)]);
            let v = _mm_min_epi32(_mm_max_epi32(dot4(&inp, w, in_sum), zero), clampv);
            let mut lanes = [0i32; 4];
            unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), v) };
            for (k, &l) in lanes.iter().enumerate() {
                row[(gx + k) * step_x + step_x - 1] = P::from_i32(l);
            }
            gx += 4;
        }
        while gx < grid_w {
            let s = dot1(&inp, &weights[widx(gx * dw, iy)], in_sum);
            row[gx * step_x + step_x - 1] = P::from_i32(s.clamp(0, max));
            gx += 1;
        }
        y += step_y;
    }

    // Horizontal upsampling on the sparse rows, from the left edge.
    if step_x > 1 {
        let mut y = step_y - 1;
        for _ in 0..grid_h {
            let mut p1 = px(o - (y + 1));
            let row = &mut dst[y * stride..y * stride + width];
            for gx in 0..grid_w {
                let x = gx * step_x;
                let p0 = p1;
                p1 = row[x + step_x - 1].into();
                for z in 0..step_x - 1 {
                    let z1 = (z + 1) as i32;
                    row[x + z] = P::from_i32((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2);
                }
            }
            y += step_y;
        }
    }

    // Vertical upsampling between sparse rows, from the top edge.
    if step_y > 1 {
        let mut p0 = [0i32; 64];
        let mut p1 = [0i32; 64];
        let sh = _mm_cvtsi32_si128(uhl2);
        for gy in 0..grid_h {
            let base_y = gy * step_y;
            for x in 0..width {
                p0[x] = if gy == 0 {
                    px(o + 1 + x)
                } else {
                    dst[(base_y - 1) * stride + x].into()
                };
                p1[x] = dst[(base_y + step_y - 1) * stride + x].into();
            }
            for z in 0..step_y - 1 {
                let w1 = (z + 1) as i32;
                let w0 = step_y as i32 - w1;
                let (w0v, w1v) = (_mm_set1_epi32(w0), _mm_set1_epi32(w1));
                let row = &mut dst[(base_y + z) * stride..(base_y + z) * stride + width];
                let mut x = 0usize;
                while x + 4 <= width {
                    let a = unsafe { _mm_loadu_si128(p0.as_ptr().add(x).cast()) };
                    let b = unsafe { _mm_loadu_si128(p1.as_ptr().add(x).cast()) };
                    let v = _mm_srl_epi32(
                        _mm_add_epi32(_mm_mullo_epi32(a, w0v), _mm_mullo_epi32(b, w1v)),
                        sh,
                    );
                    let mut lanes = [0i32; 4];
                    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), v) };
                    for (k, &l) in lanes.iter().enumerate() {
                        row[x + k] = P::from_i32(l);
                    }
                    x += 4;
                }
                while x < width {
                    row[x] = P::from_i32((p0[x] * w0 + p1[x] * w1) >> uhl2);
                    x += 1;
                }
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
pub(crate) fn ipred_dip_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
) {
    dip_impl(dst, stride, tl, o, width, height, mode, 255);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn ipred_dip_hbd_sse41(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
    bitdepth_max: u16,
) {
    dip_impl(dst, stride, tl, o, width, height, mode, bitdepth_max as i32);
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
    fn dip_sse41_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        let sizes = [4usize, 8, 16, 32, 64];
        let mut seed = 7u64;
        for &w in &sizes {
            for &h in &sizes {
                for mode in (0..6).chain(16..22) {
                    for &bits in &[8u32, 10, 12] {
                        let max = (1u32 << bits) - 1;
                        let o = 2 * h + 1;
                        let len = o + 2 * w + 2;
                        let tl: Vec<u16> =
                            (0..len).map(|_| (lcg(&mut seed) & max) as u16).collect();
                        let stride = w + 3;
                        if bits == 8 {
                            let tl8: Vec<u8> = tl.iter().map(|&v| v as u8).collect();
                            let mut a = vec![0u8; stride * h];
                            let mut b = vec![0u8; stride * h];
                            crate::ipred::ipred_dip_8bpc(&mut a, stride, &tl8, o, w, h, mode);
                            unsafe { ipred_dip_8bpc_sse41(&mut b, stride, &tl8, o, w, h, mode) };
                            assert_eq!(a, b, "8bpc {w}x{h} mode {mode}");
                        } else {
                            let mut a = vec![0u16; stride * h];
                            let mut b = vec![0u16; stride * h];
                            crate::ipred_dispatch::ipred_dip_hbd_scalar(
                                &mut a, stride, &tl, o, w, h, mode, max as u16,
                            );
                            unsafe {
                                ipred_dip_hbd_sse41(&mut b, stride, &tl, o, w, h, mode, max as u16)
                            };
                            assert_eq!(a, b, "{bits}-bit {w}x{h} mode {mode}");
                        }
                    }
                }
            }
        }
    }
}
