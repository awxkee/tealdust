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

use crate::avx::_mm256_hsum_epi32;
use crate::dip_tables::DIP_WEIGHTS;
use crate::intops::ulog2;
use crate::levels::ANGLE_MULTI_MRL_FLAG;
use crate::tables::SM_WEIGHTS;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn avg_pred_8bpc_avx2(dst: &mut [u8], stride: usize, tmp: &[u8], w: usize, h: usize) {
    for y in 0..h {
        let dst_row = &mut dst[y * stride..y * stride + w];
        let tmp_row = &tmp[y * 64..y * 64 + w];
        let (d32, drem32) = dst_row.as_chunks_mut::<32>();
        for (d, t) in d32.iter_mut().zip(tmp_row.as_chunks::<32>().0.iter()) {
            store_u8x32_fixed(d, _mm256_avg_epu8(load_u8x32_fixed(d), load_u8x32_fixed(t)));
        }
        let done32 = d32.len() * 32;
        let (d16, drem) = drem32.as_chunks_mut::<16>();
        for (d, t) in d16
            .iter_mut()
            .zip(tmp_row[done32..].as_chunks::<16>().0.iter())
        {
            store_u8x16_fixed(d, _mm_avg_epu8(load_u8x16_fixed(d), load_u8x16_fixed(t)));
        }
        let base = done32 + d16.len() * 16;
        for (i, d) in drem.iter_mut().enumerate() {
            *d = ((*d as u16 + tmp_row[base + i] as u16 + 1) >> 1) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ibp_blend_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    inv: bool,
    weights: &[[u8; 16]; 16],
) {
    let x_shift = w >> 5;
    let y_shift = h >> 5;
    let c128 = _mm256_set1_epi16(128);
    let c64 = _mm256_set1_epi16(64);
    let mut wrow = [0u8; 128];
    for y in 0..h {
        let wy = y >> y_shift;
        for x in 0..w {
            let wx = x >> x_shift;
            wrow[x] = if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            };
        }
        let dst_row = &mut dst[y * stride..y * stride + w];
        let tmp_row = &tmp[y * 64..y * 64 + w];
        let mut x = 0usize;
        while x + 16 <= w {
            let wv = _mm256_cvtepu8_epi16(unsafe {
                _mm_loadu_si128(wrow[x..].as_ptr() as *const __m128i)
            });
            let dv = _mm256_cvtepu8_epi16(unsafe {
                _mm_loadu_si128(dst_row[x..].as_ptr() as *const __m128i)
            });
            let tv = _mm256_cvtepu8_epi16(unsafe {
                _mm_loadu_si128(tmp_row[x..].as_ptr() as *const __m128i)
            });
            let acc = _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mullo_epi16(tv, _mm256_sub_epi16(c128, wv)),
                    _mm256_mullo_epi16(dv, wv),
                ),
                c64,
            );
            let res = _mm256_srli_epi16::<7>(acc);
            store_i16x16_u8_fixed((&mut dst_row[x..x + 16]).try_into().unwrap(), res);
            x += 16;
        }
        while x + 8 <= w {
            let wv =
                _mm_cvtepu8_epi16(unsafe { _mm_loadl_epi64(wrow[x..].as_ptr() as *const __m128i) });
            let dv = _mm_cvtepu8_epi16(unsafe {
                _mm_loadl_epi64(dst_row[x..].as_ptr() as *const __m128i)
            });
            let tv = _mm_cvtepu8_epi16(unsafe {
                _mm_loadl_epi64(tmp_row[x..].as_ptr() as *const __m128i)
            });
            let acc = _mm_add_epi16(
                _mm_add_epi16(
                    _mm_mullo_epi16(tv, _mm_sub_epi16(_mm_set1_epi16(128), wv)),
                    _mm_mullo_epi16(dv, wv),
                ),
                _mm_set1_epi16(64),
            );
            store_i16x8_u8_fixed(
                (&mut dst_row[x..x + 8]).try_into().unwrap(),
                _mm_srli_epi16::<7>(acc),
            );
            x += 8;
        }
        while x < w {
            let wx = x >> x_shift;
            let weight = (if inv {
                weights[wx][wy]
            } else {
                weights[wy][wx]
            }) as u16;
            let t = tmp_row[x] as u16;
            let d = dst_row[x] as u16;
            dst_row[x] = ((t * (128 - weight) + d * weight + 64) >> 7) as u8;
            x += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn sra_i16(v: __m128i, shift: i32) -> __m128i {
    _mm_sra_epi16(v, _mm_cvtsi32_si128(shift))
}

#[inline]
#[target_feature(enable = "avx2")]
fn sra_i16x16(v: __m256i, shift: i32) -> __m256i {
    _mm256_sra_epi16(v, _mm_cvtsi32_si128(shift))
}

#[inline(always)]
fn load_u8x32_fixed(a: &[u8; 32]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn store_u8x32_fixed(a: &mut [u8; 32], v: __m256i) {
    unsafe { _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v) };
}

#[inline(always)]
fn load_u8x16_i16_avx2(a: &[u8; 16]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16_u8_fixed(a: &mut [u8; 16], v: __m256i) {
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let packed = _mm_packus_epi16(lo, hi);
        _mm_storeu_si128(a.as_mut_ptr().cast(), packed);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn splat_row_u8_avx2(row: &mut [u8], v: u8) {
    let vv = _mm256_set1_epi8(v as i8);
    let (c32, r32) = row.as_chunks_mut::<32>();
    for c in c32.iter_mut() {
        store_u8x32_fixed(c, vv);
    }
    let (c16, rem) = r32.as_chunks_mut::<16>();
    let vv16 = _mm_set1_epi8(v as i8);
    for c in c16.iter_mut() {
        store_u8x16_fixed(c, vv16);
    }
    rem.fill(v);
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_v_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if width < 16 {
        crate::ipred_dispatch::ipred_v_scalar(dst, stride, tl, o, width, height, angle);
        return;
    }

    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        let top1 = &tl[o + 1..o + 1 + width];
        let top2 = &tl[o + 1 + e_stride..o + 1 + e_stride + width];
        let (dc32, rem32) = dst[..width].as_chunks_mut::<32>();
        for ((d, a), b) in dc32
            .iter_mut()
            .zip(top1.as_chunks::<32>().0.iter())
            .zip(top2.as_chunks::<32>().0.iter())
        {
            store_u8x32_fixed(d, _mm256_avg_epu8(load_u8x32_fixed(a), load_u8x32_fixed(b)));
        }
        let done32 = dc32.len() * 32;
        let (dc16, drem) = rem32.as_chunks_mut::<16>();
        for ((d, a), b) in dc16
            .iter_mut()
            .zip(top1[done32..].as_chunks::<16>().0.iter())
            .zip(top2[done32..].as_chunks::<16>().0.iter())
        {
            store_u8x16_fixed(d, _mm_avg_epu8(load_u8x16_fixed(a), load_u8x16_fixed(b)));
        }
        let base_x = done32 + dc16.len() * 16;
        for (xi, d) in drem.iter_mut().enumerate() {
            let x = base_x + xi;
            let t1 = tl[o + 1 + x] as u16;
            let t2 = tl[o + 1 + e_stride + x] as u16;
            *d = ((t1 + t2 + 1) >> 1) as u8;
        }
    } else {
        let top = &tl[o + 1..o + 1 + width];
        dst[..width].copy_from_slice(top);
    }

    let mut off = stride;
    for _ in 1..height {
        dst.copy_within(0..width, off);
        off += stride;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_h_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if width < 16 {
        crate::ipred_dispatch::ipred_h_scalar(dst, stride, tl, o, width, height, angle);
        return;
    }

    let e_stride = (width + height) * 2 + 1;
    let mrl = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let mut off = 0usize;
    for y in 0..height {
        let v = if mrl {
            let left = tl[o - 1 - y] as u16;
            let left2 = tl[o + e_stride - 1 - y] as u16;
            ((left + left2 + 1) >> 1) as u8
        } else {
            tl[o - 1 - y]
        };
        splat_row_u8_avx2(&mut dst[off..off + width], v);
        off += stride;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_smooth_v_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_v_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bhl2 = ulog2(h as u32) as i32;
    let rnd = _mm256_set1_epi16((h >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom = tl[o - h - 1] as i16;
    let bottom_v = _mm256_set1_epi16(bottom);
    let bottom_v128 = _mm_set1_epi16(bottom);
    let add32 = _mm256_set1_epi16(32);
    let add32_128 = _mm_set1_epi16(32);

    let mut off = 0usize;
    for y in 0..h {
        let off_y = _mm256_set1_epi16((h - 1 - y) as i16);
        let off_y128 = _mm_set1_epi16((h - 1 - y) as i16);
        let w_ver = _mm256_set1_epi16(weights[y] as i16);
        let w_ver128 = _mm_set1_epi16(weights[y] as i16);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let (c16, r16) = row.as_chunks_mut::<16>();
        for (d, t) in c16.iter_mut().zip(top_src.as_chunks::<16>().0.iter()) {
            let above = load_u8x16_i16_avx2(t);
            let mul = _mm256_mullo_epi16(_mm256_sub_epi16(above, bottom_v), off_y);
            let pred = _mm256_add_epi16(bottom_v, sra_i16x16(_mm256_add_epi16(mul, rnd), bhl2));
            let adj = _mm256_srai_epi16::<6>(_mm256_add_epi16(
                _mm256_mullo_epi16(_mm256_sub_epi16(above, pred), w_ver),
                add32,
            ));
            store_i16x16_u8_fixed(d, _mm256_add_epi16(pred, adj));
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (d, t) in c8.iter_mut().zip(top_src[done..].as_chunks::<8>().0.iter()) {
            let above = load_u8x8_i16_fixed(t);
            let mul = _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v128), off_y128);
            let pred = _mm_add_epi16(
                bottom_v128,
                sra_i16(_mm_add_epi16(mul, _mm_set1_epi16((h >> 1) as i16)), bhl2),
            );
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(above, pred), w_ver128),
                add32_128,
            ));
            store_i16x8_u8_fixed(d, _mm_add_epi16(pred, adj));
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let x = base_x + xi;
            let above = tl[o + 1 + x] as i32;
            let mul = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let pred = bottom as i32 + ((mul + (h >> 1) as i32) >> bhl2);
            *d = (pred + (((above - pred) * weights[y] as i32 + 32) >> 6)) as u8;
        }
        off += stride;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_smooth_h_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_h_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let rnd = _mm256_set1_epi16((w >> 1) as i16);
    let rnd128 = _mm_set1_epi16((w >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let right_v = _mm256_set1_epi16(right);
    let right_v128 = _mm_set1_epi16(right);
    let offsets = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let offsets128 = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    let add32 = _mm256_set1_epi16(32);
    let add32_128 = _mm_set1_epi16(32);

    let mut off = 0usize;
    for y in 0..h {
        let left = tl[o - 1 - y] as i16;
        let left_v = _mm256_set1_epi16(left);
        let left_v128 = _mm_set1_epi16(left);
        let diff = _mm256_set1_epi16(left - right);
        let diff128 = _mm_set1_epi16(left - right);
        let row = &mut dst[off..off + w];
        let (c16, r16) = row.as_chunks_mut::<16>();
        for (ci, (d, wxc)) in c16
            .iter_mut()
            .zip(weights[..w].as_chunks::<16>().0.iter())
            .enumerate()
        {
            let x = ci * 16;
            let x0 = (w - 1 - x) as i16;
            let dist = _mm256_sub_epi16(_mm256_set1_epi16(x0), offsets);
            let wx = load_u8x16_i16_avx2(wxc);
            let pred = _mm256_add_epi16(
                right_v,
                sra_i16x16(_mm256_add_epi16(_mm256_mullo_epi16(diff, dist), rnd), bwl2),
            );
            let adj = _mm256_srai_epi16::<6>(_mm256_add_epi16(
                _mm256_mullo_epi16(_mm256_sub_epi16(left_v, pred), wx),
                add32,
            ));
            store_i16x16_u8_fixed(d, _mm256_add_epi16(pred, adj));
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (ci, (d, wxc)) in c8
            .iter_mut()
            .zip(weights[done..w].as_chunks::<8>().0.iter())
            .enumerate()
        {
            let x0 = (w - 1 - done - ci * 8) as i16;
            let dist = _mm_sub_epi16(_mm_set1_epi16(x0), offsets128);
            let wx = load_u8x8_i16_fixed(wxc);
            let pred = _mm_add_epi16(
                right_v128,
                sra_i16(_mm_add_epi16(_mm_mullo_epi16(diff128, dist), rnd128), bwl2),
            );
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(left_v128, pred), wx),
                add32_128,
            ));
            store_i16x8_u8_fixed(d, _mm_add_epi16(pred, adj));
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let x = base_x + xi;
            let mul = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let pred = right as i32 + ((mul + (w >> 1) as i32) >> bwl2);
            *d = (pred + (((left as i32 - pred) * weights[x] as i32 + 32) >> 6)) as u8;
        }
        off += stride;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_smooth_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let bhl2 = ulog2(h as u32) as i32;
    let rnd_ver = _mm256_set1_epi16((h >> 1) as i16);
    let rnd_hor = _mm256_set1_epi16((w >> 1) as i16);
    let rnd_ver128 = _mm_set1_epi16((h >> 1) as i16);
    let rnd_hor128 = _mm_set1_epi16((w >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let bottom = tl[o - h - 1] as i16;
    let right_v = _mm256_set1_epi16(right);
    let bottom_v = _mm256_set1_epi16(bottom);
    let right_v128 = _mm_set1_epi16(right);
    let bottom_v128 = _mm_set1_epi16(bottom);
    let offsets = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let offsets128 = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    let add32 = _mm256_set1_epi16(32);
    let add32_128 = _mm_set1_epi16(32);
    let one = _mm256_set1_epi16(1);
    let one128 = _mm_set1_epi16(1);

    let mut off = 0usize;
    for y in 0..h {
        let left = tl[o - 1 - y] as i16;
        let left_v = _mm256_set1_epi16(left);
        let left_v128 = _mm_set1_epi16(left);
        let diff_hor = _mm256_set1_epi16(left - right);
        let diff_hor128 = _mm_set1_epi16(left - right);
        let off_ver = _mm256_set1_epi16((h - 1 - y) as i16);
        let off_ver128 = _mm_set1_epi16((h - 1 - y) as i16);
        let w_ver = _mm256_set1_epi16(weights[y] as i16);
        let w_ver128 = _mm_set1_epi16(weights[y] as i16);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let (c16, r16) = row.as_chunks_mut::<16>();
        for (ci, ((d, t), wxc)) in c16
            .iter_mut()
            .zip(top_src.as_chunks::<16>().0.iter())
            .zip(weights[..w].as_chunks::<16>().0.iter())
            .enumerate()
        {
            let x = ci * 16;
            let x0 = (w - 1 - x) as i16;
            let dist = _mm256_sub_epi16(_mm256_set1_epi16(x0), offsets);
            let above = load_u8x16_i16_avx2(t);
            let wx = load_u8x16_i16_avx2(wxc);

            let pred_ver = _mm256_add_epi16(
                bottom_v,
                sra_i16x16(
                    _mm256_add_epi16(
                        _mm256_mullo_epi16(_mm256_sub_epi16(above, bottom_v), off_ver),
                        rnd_ver,
                    ),
                    bhl2,
                ),
            );
            let pred_hor = _mm256_add_epi16(
                right_v,
                sra_i16x16(
                    _mm256_add_epi16(_mm256_mullo_epi16(diff_hor, dist), rnd_hor),
                    bwl2,
                ),
            );
            let pred_ver = _mm256_add_epi16(
                pred_ver,
                _mm256_srai_epi16::<6>(_mm256_add_epi16(
                    _mm256_mullo_epi16(_mm256_sub_epi16(above, pred_ver), w_ver),
                    add32,
                )),
            );
            let pred_hor = _mm256_add_epi16(
                pred_hor,
                _mm256_srai_epi16::<6>(_mm256_add_epi16(
                    _mm256_mullo_epi16(_mm256_sub_epi16(left_v, pred_hor), wx),
                    add32,
                )),
            );
            let out =
                _mm256_srai_epi16::<1>(_mm256_add_epi16(_mm256_add_epi16(pred_ver, pred_hor), one));
            store_i16x16_u8_fixed(d, out);
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (ci, ((d, t), wxc)) in c8
            .iter_mut()
            .zip(top_src[done..].as_chunks::<8>().0.iter())
            .zip(weights[done..w].as_chunks::<8>().0.iter())
            .enumerate()
        {
            let above = load_u8x8_i16_fixed(t);
            let x0 = (w - 1 - done - ci * 8) as i16;
            let dist = _mm_sub_epi16(_mm_set1_epi16(x0), offsets128);
            let wx = load_u8x8_i16_fixed(wxc);

            let pred_ver = _mm_add_epi16(
                bottom_v128,
                sra_i16(
                    _mm_add_epi16(
                        _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v128), off_ver128),
                        rnd_ver128,
                    ),
                    bhl2,
                ),
            );
            let pred_hor = _mm_add_epi16(
                right_v128,
                sra_i16(
                    _mm_add_epi16(_mm_mullo_epi16(diff_hor128, dist), rnd_hor128),
                    bwl2,
                ),
            );
            let pred_ver = _mm_add_epi16(
                pred_ver,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(above, pred_ver), w_ver128),
                    add32_128,
                )),
            );
            let pred_hor = _mm_add_epi16(
                pred_hor,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(left_v128, pred_hor), wx),
                    add32_128,
                )),
            );
            store_i16x8_u8_fixed(
                d,
                _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(pred_ver, pred_hor), one128)),
            );
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let x = base_x + xi;
            let above = tl[o + 1 + x] as i32;
            let mul_ver = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let mul_hor = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let mut pred_ver = bottom as i32 + ((mul_ver + (h >> 1) as i32) >> bhl2);
            let mut pred_hor = right as i32 + ((mul_hor + (w >> 1) as i32) >> bwl2);
            pred_ver += ((above - pred_ver) * weights[y] as i32 + 32) >> 6;
            pred_hor += ((left as i32 - pred_hor) * weights[x] as i32 + 32) >> 6;
            *d = ((pred_ver + pred_hor + 1) >> 1) as u8;
        }
        off += stride;
    }
}

use crate::levels::ANGLE_IBP_FLAG;

#[inline(always)]
fn load_u8x16_fixed(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_u8x16_fixed(a: &mut [u8; 16], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) };
}

/// Horizontal sum of all bytes in `s` (each lane widened to u32).
#[inline]
#[target_feature(enable = "avx2")]
fn sum_u8_avx2(s: &[u8]) -> u32 {
    let zero = _mm256_setzero_si256();
    let mut acc = zero;
    let (chunks, rem) = s.as_chunks::<32>();
    for c in chunks.iter() {
        // SAD against zero gives four u64 partial sums for 32 bytes.
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(load_u8x32_fixed(c), zero));
    }
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256::<1>(acc);
    let acc128 = _mm_add_epi64(lo, hi);
    let mut total =
        (_mm_extract_epi64::<0>(acc128) as u64 + _mm_extract_epi64::<1>(acc128) as u64) as u32;
    for &b in rem {
        total += b as u32;
    }
    total
}

/// Fill a `w x h` block at `off` with the constant byte `dc`.
#[inline]
#[target_feature(enable = "avx2")]
fn splat_fill_avx2(dst: &mut [u8], stride: usize, off: usize, w: usize, h: usize, dc: u8) {
    let mut p = off;
    for _ in 0..h {
        splat_row_u8_avx2(&mut dst[p..p + w], dc);
        p += stride;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_dc_128_8bpc_avx2(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    splat_fill_avx2(dst, stride, 0, w, h, 128);
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_dc_top_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_top_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let sum = sum_u8_avx2(&tl[o + 1..o + 1 + w]);
    let dc = (((w >> 1) as u32 + sum) >> (w as u32).trailing_zeros()) as u8;
    splat_fill_avx2(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_dc_left_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_left_8bpc(dst, stride, tl, o, w, h, angle);
    }
    // Left samples tl[o-1-i], i in 0..h, are the contiguous slice tl[o-h..o]
    // (order does not matter for a sum).
    let sum = sum_u8_avx2(&tl[o - h..o]);
    let dc = (((h >> 1) as u32 + sum) >> (h as u32).trailing_zeros()) as u8;
    splat_fill_avx2(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_dc_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let n_pel = (w + h) as u32;
    let sum = sum_u8_avx2(&tl[o + 1..o + 1 + w]) + sum_u8_avx2(&tl[o - h..o]);
    let dc = if n_pel & (n_pel - 1) == 0 {
        (sum + w as u32) >> n_pel.trailing_zeros()
    } else {
        crate::ipred::fast_div32_dc(sum, n_pel).min(255)
    } as u8;
    splat_fill_avx2(dst, stride, 0, w, h, dc);
}

#[inline(always)]
fn load_u8x8_i16_fixed(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i16x8_u8_fixed(a: &mut [u8; 8], v: __m128i) {
    let packed = unsafe { _mm_packus_epi16(v, _mm_setzero_si128()) };
    unsafe { _mm_storel_epi64(a.as_mut_ptr().cast(), packed) };
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_paeth_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    let topleft = tl[o] as i32;
    if w < 8 {
        return crate::ipred::ipred_paeth_8bpc(dst, stride, tl, o, w, h);
    }
    let tl_v = _mm256_set1_epi16(topleft as i16);
    let tl_v128 = _mm_set1_epi16(topleft as i16);
    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let left_v = _mm256_set1_epi16(left as i16);
        let left_v128 = _mm_set1_epi16(left as i16);
        let top_src = &tl[o + 1..o + 1 + w];
        let (c16, r16) = dst[off..off + w].as_chunks_mut::<16>();
        for (d, t) in c16.iter_mut().zip(top_src.as_chunks::<16>().0.iter()) {
            let top = load_u8x16_i16_avx2(t);
            let base = _mm256_sub_epi16(_mm256_add_epi16(left_v, top), tl_v);
            let ld = _mm256_abs_epi16(_mm256_sub_epi16(left_v, base));
            let td = _mm256_abs_epi16(_mm256_sub_epi16(top, base));
            let tld = _mm256_abs_epi16(_mm256_sub_epi16(tl_v, base));
            let cond_l = _mm256_and_si256(
                _mm256_cmpeq_epi16(ld, _mm256_min_epi16(ld, td)),
                _mm256_cmpeq_epi16(ld, _mm256_min_epi16(ld, tld)),
            );
            let cond_t = _mm256_cmpeq_epi16(td, _mm256_min_epi16(td, tld));
            let inner = _mm256_blendv_epi8(tl_v, top, cond_t);
            let res = _mm256_blendv_epi8(inner, left_v, cond_l);
            store_i16x16_u8_fixed(d, res);
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (d, t) in c8.iter_mut().zip(top_src[done..].as_chunks::<8>().0.iter()) {
            let top_v = load_u8x8_i16_fixed(t);
            let base = _mm_sub_epi16(_mm_add_epi16(left_v128, top_v), tl_v128);
            let ld = _mm_abs_epi16(_mm_sub_epi16(left_v128, base));
            let td = _mm_abs_epi16(_mm_sub_epi16(top_v, base));
            let tld = _mm_abs_epi16(_mm_sub_epi16(tl_v128, base));
            let cond_l = _mm_and_si128(
                _mm_cmpeq_epi16(ld, _mm_min_epi16(ld, td)),
                _mm_cmpeq_epi16(ld, _mm_min_epi16(ld, tld)),
            );
            let cond_t = _mm_cmpeq_epi16(td, _mm_min_epi16(td, tld));
            let inner = _mm_blendv_epi8(tl_v128, top_v, cond_t);
            let res = _mm_blendv_epi8(inner, left_v128, cond_l);
            store_i16x8_u8_fixed(d, res);
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let top = tl[o + 1 + base_x + xi] as i32;
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();
            *d = if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            } as u8;
        }
        off += stride;
    }
}

/// Load 8 bytes and zero-extend to one i32x8 AVX2 lane.
#[inline(always)]
fn load8_u8_i32_avx2(a: &[u8; 8]) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn load8_u8_i32_rev_avx2(a: &[u8; 8]) -> __m256i {
    let v = unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) };
    let mask = unsafe { _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1) };
    unsafe { _mm256_cvtepu8_epi32(_mm_shuffle_epi8(v, mask)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn widen8_at_avx2<const LO: i32>(v: __m128i) -> __m256i {
    _mm256_cvtepu8_epi32(_mm_srli_si128(v, LO))
}

/// Apply the 4-tap directional filter to one 8-lane AVX2 vector.
#[inline]
#[target_feature(enable = "avx2")]
fn dr_filter8_avx2(
    av: __m256i,
    bv: __m256i,
    cv: __m256i,
    dv: __m256i,
    rnd: __m256i,
    zero: __m256i,
    maxv: __m256i,
    w0: __m256i,
    w1: __m256i,
    w2: __m256i,
    w3: __m256i,
) -> __m256i {
    let acc = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(av, w0), _mm256_mullo_epi32(bv, w1)),
        _mm256_add_epi32(_mm256_mullo_epi32(cv, w2), _mm256_mullo_epi32(dv, w3)),
    );
    _mm256_min_epi32(
        _mm256_max_epi32(_mm256_srai_epi32::<7>(_mm256_add_epi32(acc, rnd)), zero),
        maxv,
    )
}

#[inline(always)]
fn store_i32x8_u8_fixed(a: &mut [u8; 8], v: __m256i) {
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let packed16 = _mm_packus_epi32(lo, hi);
        let packed8 = _mm_packus_epi16(packed16, _mm_setzero_si128());
        _mm_storel_epi64(a.as_mut_ptr().cast(), packed8);
    }
}

#[inline(always)]
fn store_i32x8x2_u8_fixed(a: &mut [u8; 16], lo8: __m256i, hi8: __m256i) {
    unsafe {
        let lo16 = _mm_packus_epi32(
            _mm256_castsi256_si128(lo8),
            _mm256_extracti128_si256::<1>(lo8),
        );
        let hi16 = _mm_packus_epi32(
            _mm256_castsi256_si128(hi8),
            _mm256_extracti128_si256::<1>(hi8),
        );
        let packed = _mm_packus_epi16(lo16, hi16);
        _mm_storeu_si128(a.as_mut_ptr().cast(), packed);
    }
}

/// One row of the Z1 luma 4-tap interpolation. Pixels with `base <= max_base_x`
/// are filtered; the rest of the row is set to `fill`.
#[inline]
#[target_feature(enable = "avx2")]
fn z1_luma_row_avx2(
    filt: &[u8],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u8,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u8],
    w: usize,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let av = _mm256_set1_epi32(f.a as i32);
    let bv = _mm256_set1_epi32(f.b as i32);
    let cv = _mm256_set1_epi32(f.c as i32);
    let dv = _mm256_set1_epi32(f.d as i32);
    let rnd = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(255);

    let base_const = (top_off as i32 + base0) as usize;
    let (body, fill_tail) = dst_row.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    for (ci, d) in c16.iter_mut().enumerate() {
        let bi = base_const + ci * 16;
        let va = unsafe { _mm_loadu_si128(filt[bi - 1..].as_ptr() as *const __m128i) };
        let lo8 = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            widen8_at_avx2::<0>(va),
            widen8_at_avx2::<1>(va),
            widen8_at_avx2::<2>(va),
            widen8_at_avx2::<3>(va),
        );
        // group B taps bi+7..bi+10 → byte-offsets 5..8 of a load at bi+2.
        let vb = unsafe { _mm_loadu_si128(filt[bi + 2..].as_ptr() as *const __m128i) };
        let hi8 = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            widen8_at_avx2::<5>(vb),
            widen8_at_avx2::<6>(vb),
            widen8_at_avx2::<7>(vb),
            widen8_at_avx2::<8>(vb),
        );
        store_i32x8x2_u8_fixed(d, lo8, hi8);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for (ci, d) in c8.iter_mut().enumerate() {
        let bi = base_const + done + ci * 8;
        let res = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            load8_u8_i32_avx2((&filt[bi - 1..bi - 1 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[bi..bi + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[bi + 1..bi + 1 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[bi + 2..bi + 2 + 8]).try_into().unwrap()),
        );
        store_i32x8_u8_fixed(d, res);
    }
    let base_x = done + c8.len() * 8;
    for (xi, d) in r8.iter_mut().enumerate() {
        let bi = base_const + base_x + xi;
        let v = f.a as i32 * filt[bi - 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi + 1] as i32
            + f.d as i32 * filt[bi + 2] as i32;
        *d = ((v + 64) >> 7).clamp(0, 255) as u8;
    }
    fill_tail.fill(fill);
}

#[inline]
#[target_feature(enable = "avx2")]
fn z1_chroma_row_avx2(
    filt: &[u8],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u8,
    shift: usize,
    dst_row: &mut [u8],
    w: usize,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let iw = _mm256_set1_epi16((32 - shift) as i16);
    let sw = _mm256_set1_epi16(shift as i16);
    let rnd = _mm256_set1_epi16(16);
    let base_const = (top_off as i32 + base0) as usize;
    let (body, fill_tail) = dst_row.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    for (ci, d) in c16.iter_mut().enumerate() {
        let bi = base_const + ci * 16;
        let a = load_u8x16_i16_avx2((&filt[bi..bi + 16]).try_into().unwrap());
        let b = load_u8x16_i16_avx2((&filt[bi + 1..bi + 17]).try_into().unwrap());
        let v = _mm256_srli_epi16::<5>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(a, iw), _mm256_mullo_epi16(b, sw)),
            rnd,
        ));
        store_i16x16_u8_fixed(d, v);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let iw8 = _mm_set1_epi16((32 - shift) as i16);
    let sw8 = _mm_set1_epi16(shift as i16);
    for (ci, d) in c8.iter_mut().enumerate() {
        let bi = base_const + done + ci * 8;
        let a = load_u8x8_i16_fixed((&filt[bi..bi + 8]).try_into().unwrap());
        let b = load_u8x8_i16_fixed((&filt[bi + 1..bi + 9]).try_into().unwrap());
        let v = _mm_srli_epi16::<5>(_mm_add_epi16(
            _mm_add_epi16(_mm_mullo_epi16(a, iw8), _mm_mullo_epi16(b, sw8)),
            _mm_set1_epi16(16),
        ));
        store_i16x8_u8_fixed(d, v);
    }
    let base_x = done + c8.len() * 8;
    for (xi, d) in r8.iter_mut().enumerate() {
        let bi = base_const + base_x + xi;
        let v = (32 - shift as i32) * filt[bi] as i32 + shift as i32 * filt[bi + 1] as i32;
        *d = ((v + 16) >> 5).clamp(0, 255) as u8;
    }
    fill_tail.fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn ipred_z1_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = [0u8; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z1_8bpc_avx2(
            tmp.as_mut_slice(),
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
            ibp_weights,
        );
        ipred_z1_8bpc_avx2(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
            ibp_weights,
        );
        avg_pred_8bpc_avx2(dst, stride, tmp.as_slice(), w, h);
        return;
    }
    if enable_ibp {
        let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
        let mode_idx = (10 - (a >> 3)).min(6) as usize;
        let mut tmp = [0u8; 64 * 64];
        ipred_z1_8bpc_avx2(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle & !ANGLE_IBP_FLAG,
            max_width,
            max_height,
            ibp_weights,
        );
        ipred_z3_8bpc_avx2(
            tmp.as_mut_slice(),
            64,
            tl,
            o,
            w,
            h,
            (180 + a) | angle_flags,
            max_width,
            max_height,
            ibp_weights,
        );
        ibp_blend_8bpc_avx2(
            dst,
            stride,
            tmp.as_slice(),
            w,
            h,
            false,
            &ibp_weights[mode_idx],
        );
        return;
    }
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;

    let dx = crate::tables::DR_INTRA_DERIVATIVE[a as usize] as i32;
    let max_base_x = (w + h) as i32 - 1 + (mrl_idx as i32 * 2);
    let mut filt = [0u8; 141];
    let top_off = 2usize + mrl_idx;
    let sz = 1 + mrl_idx + w + h + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, 90 - a, is_sm_t)
    } else {
        0
    };
    if str > 0 {
        crate::ipred::filter_edge(
            &mut filt[1..],
            sz,
            1,
            sz as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz + 1];

    let mut ypos = dx * (1 + mrl_idx as i32);
    for y in 0..h {
        let base0 = ypos >> 6;
        let fill = filt[top_off + max_base_x as usize];
        if base0 > max_base_x {
            for row in dst.chunks_mut(stride).take(h).skip(y) {
                row[..w].fill(fill);
            }
            break;
        }
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let dst_row = &mut dst[y * stride..y * stride + w];
        if is_luma {
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z1_luma_row_avx2(&filt, top_off, base0, max_base_x, fill, f, dst_row, w);
        } else {
            z1_chroma_row_avx2(&filt, top_off, base0, max_base_x, fill, shift, dst_row, w);
        }
        ypos += dx;
    }
}

/// Fill `col[0..h]` for one Z3 column: filtered where `base <= max_base_y`,
/// else `fill`.
#[inline]
#[target_feature(enable = "avx2")]
fn z3_luma_col_avx2(
    filt: &[u8],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u8,
    f: &crate::ipred::DrFilter4Tap,
    col: &mut [u8],
    h: usize,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let av = _mm256_set1_epi32(f.a as i32);
    let bv = _mm256_set1_epi32(f.b as i32);
    let cv = _mm256_set1_epi32(f.c as i32);
    let dv = _mm256_set1_epi32(f.d as i32);
    let rnd = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(255);

    let lob = left_off as i32 - base0; // bi_j at y == 0
    let (body, fill_tail) = col.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (ci, d) in c16.iter_mut().enumerate() {
        let bij = lob - (ci * 16) as i32;
        let ra = _mm_shuffle_epi8(
            unsafe { _mm_loadu_si128(filt[(bij - 14) as usize..].as_ptr() as *const __m128i) },
            rev16,
        );
        let lo8 = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            widen8_at_avx2::<0>(ra),
            widen8_at_avx2::<1>(ra),
            widen8_at_avx2::<2>(ra),
            widen8_at_avx2::<3>(ra),
        );
        let rb = _mm_shuffle_epi8(
            unsafe { _mm_loadu_si128(filt[(bij - 17) as usize..].as_ptr() as *const __m128i) },
            rev16,
        );
        let hi8 = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            widen8_at_avx2::<5>(rb),
            widen8_at_avx2::<6>(rb),
            widen8_at_avx2::<7>(rb),
            widen8_at_avx2::<8>(rb),
        );
        store_i32x8x2_u8_fixed(d, lo8, hi8);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for (ci, d) in c8.iter_mut().enumerate() {
        let bij = lob - (done + ci * 8) as i32;
        let (sa, sb, sc, sd) = (
            (bij - 6) as usize,
            (bij - 7) as usize,
            (bij - 8) as usize,
            (bij - 9) as usize,
        );
        let res = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            load8_u8_i32_rev_avx2((&filt[sa..sa + 8]).try_into().unwrap()),
            load8_u8_i32_rev_avx2((&filt[sb..sb + 8]).try_into().unwrap()),
            load8_u8_i32_rev_avx2((&filt[sc..sc + 8]).try_into().unwrap()),
            load8_u8_i32_rev_avx2((&filt[sd..sd + 8]).try_into().unwrap()),
        );
        store_i32x8_u8_fixed(d, res);
    }
    let base_y = done + c8.len() * 8;
    for (yi, d) in r8.iter_mut().enumerate() {
        let bi = (lob - (base_y + yi) as i32) as usize;
        let v = f.a as i32 * filt[bi + 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi - 1] as i32
            + f.d as i32 * filt[bi - 2] as i32;
        *d = ((v + 64) >> 7).clamp(0, 255) as u8;
    }
    fill_tail.fill(fill);
}

#[inline]
#[target_feature(enable = "avx2")]
fn z3_chroma_col_avx2(
    filt: &[u8],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u8,
    shift: usize,
    col: &mut [u8],
    h: usize,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let iw = _mm256_set1_epi16((32 - shift) as i16);
    let sw = _mm256_set1_epi16(shift as i16);
    let rnd = _mm256_set1_epi16(16);
    let lob = left_off as i32 - base0;
    let (body, fill_tail) = col.split_at_mut(n_filter);
    let (c16, r16) = body.as_chunks_mut::<16>();
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (ci, d) in c16.iter_mut().enumerate() {
        let bij = lob - (ci * 16) as i32;
        let a = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(
            unsafe { _mm_loadu_si128(filt[(bij - 15) as usize..].as_ptr() as *const __m128i) },
            rev16,
        ));
        let b = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(
            unsafe { _mm_loadu_si128(filt[(bij - 16) as usize..].as_ptr() as *const __m128i) },
            rev16,
        ));
        let v = _mm256_srli_epi16::<5>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(a, iw), _mm256_mullo_epi16(b, sw)),
            rnd,
        ));
        store_i16x16_u8_fixed(d, v);
    }
    let base_y = c16.len() * 16;
    for (yi, d) in r16.iter_mut().enumerate() {
        let bi = (lob - (base_y + yi) as i32) as usize;
        let v = (32 - shift as i32) * filt[bi] as i32 + shift as i32 * filt[bi - 1] as i32;
        *d = ((v + 16) >> 5).clamp(0, 255) as u8;
    }
    fill_tail.fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn ipred_z3_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = [0u8; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z3_8bpc_avx2(
            tmp.as_mut_slice(),
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
            ibp_weights,
        );
        ipred_z3_8bpc_avx2(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
            ibp_weights,
        );
        avg_pred_8bpc_avx2(dst, stride, tmp.as_slice(), w, h);
        return;
    }
    if enable_ibp {
        if h > 64 {
            return crate::ipred::ipred_z3_8bpc(
                dst,
                stride,
                tl,
                o,
                w,
                h,
                angle,
                max_width,
                max_height,
                ibp_weights,
            );
        }
        let angle_flags = angle & !(511 | ANGLE_IBP_FLAG);
        let mode_idx = ((a - 183) >> 3).min(6) as usize;
        let mut tmp = [0u8; 64 * 64];
        ipred_z3_8bpc_avx2(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle & !ANGLE_IBP_FLAG,
            max_width,
            max_height,
            ibp_weights,
        );
        ipred_z1_8bpc_avx2(
            tmp.as_mut_slice(),
            64,
            tl,
            o,
            w,
            h,
            (a - 180) | angle_flags,
            max_width,
            max_height,
            ibp_weights,
        );
        ibp_blend_8bpc_avx2(
            dst,
            stride,
            tmp.as_slice(),
            w,
            h,
            true,
            &ibp_weights[mode_idx],
        );
        return;
    }
    if h > 64 {
        return crate::ipred::ipred_z3_8bpc(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
        );
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(270 - a) as usize] as i32;
    let max_base_y = (w + h) as i32 - 1 + (mrl_idx as i32 * 2);
    let mut filt = [0u8; 141];
    let left_off = 1 + w + h + mrl_idx * 2;
    let sz = 1 + mrl_idx + w + h + mrl_idx * 2;
    let str = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, a - 180, is_sm_l)
    } else {
        0
    };
    if str > 0 {
        crate::ipred::filter_edge(
            &mut filt[2..],
            sz,
            h as i32 - max_height,
            sz as i32 - 1,
            &tl[o + 1 - sz..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[2..2 + sz].copy_from_slice(&tl[o + 1 - sz..o + 1]);
    }
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];

    let mut col = [0u8; 128];
    let mut ypos = dy * (1 + mrl_idx as i32);
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let base0 = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        if is_luma {
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z3_luma_col_avx2(&filt, left_off, base0, max_base_y, fill, f, &mut col, h);
        } else {
            z3_chroma_col_avx2(&filt, left_off, base0, max_base_y, fill, shift, &mut col, h);
        }
        for (y, &c) in col[..h].iter().enumerate() {
            dst[y * stride + x] = c;
        }
        ypos += dy;
    }
}

/// Fill `dst_row[x_start..w]` from the top reference; `xpos0` is `xpos` at
/// `x_start` and `f`/`shift` are constant across the span.
#[inline]
#[target_feature(enable = "avx2")]
fn z2_top_span_avx2(
    filt: &[u8],
    top_off: usize,
    mut xpos: i32,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u8],
    x_start: usize,
    w: usize,
) {
    let av = _mm256_set1_epi32(f.a as i32);
    let bv = _mm256_set1_epi32(f.b as i32);
    let cv = _mm256_set1_epi32(f.c as i32);
    let dv = _mm256_set1_epi32(f.d as i32);
    let rnd = _mm256_set1_epi32(64);
    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(255);

    let mut x = x_start;
    while x + 16 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 1 < 0 || ti0 + 20 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 1) as usize;
        let lo8 = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            load8_u8_i32_avx2((&filt[sa..sa + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[sa + 1..sa + 1 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[sa + 2..sa + 2 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[sa + 3..sa + 3 + 8]).try_into().unwrap()),
        );
        let hi_sa = sa + 8;
        let hi8 = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            load8_u8_i32_avx2((&filt[hi_sa..hi_sa + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[hi_sa + 1..hi_sa + 1 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[hi_sa + 2..hi_sa + 2 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[hi_sa + 3..hi_sa + 3 + 8]).try_into().unwrap()),
        );
        store_i32x8x2_u8_fixed((&mut dst_row[x..x + 16]).try_into().unwrap(), lo8, hi8);
        x += 16;
        xpos += 64 * 16;
    }
    while x + 8 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 1 < 0 || ti0 + 12 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 1) as usize;
        let res = dr_filter8_avx2(
            av,
            bv,
            cv,
            dv,
            rnd,
            zero,
            maxv,
            load8_u8_i32_avx2((&filt[sa..sa + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[sa + 1..sa + 1 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[sa + 2..sa + 2 + 8]).try_into().unwrap()),
            load8_u8_i32_avx2((&filt[sa + 3..sa + 3 + 8]).try_into().unwrap()),
        );
        store_i32x8_u8_fixed((&mut dst_row[x..x + 8]).try_into().unwrap(), res);
        x += 8;
        xpos += 64 * 8;
    }
    while x < w {
        let base_x = xpos >> 6;
        let ti = top_off as i32 + base_x;
        let v = f.a as i32 * filt[(ti + 1) as usize] as i32
            + f.b as i32 * filt[(ti + 2) as usize] as i32
            + f.c as i32 * filt[(ti + 3) as usize] as i32
            + f.d as i32 * filt[(ti + 4) as usize] as i32;
        dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
        x += 1;
        xpos += 64;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn z2_top_span_chroma_avx2(
    filt: &[u8],
    top_off: usize,
    mut xpos: i32,
    shift: usize,
    dst_row: &mut [u8],
    x_start: usize,
    w: usize,
) {
    let iw = _mm256_set1_epi16((32 - shift) as i16);
    let sw = _mm256_set1_epi16(shift as i16);
    let rnd = _mm256_set1_epi16(16);
    let mut x = x_start;
    while x + 16 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 2 < 0 || ti0 + 19 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 2) as usize;
        let a = load_u8x16_i16_avx2((&filt[sa..sa + 16]).try_into().unwrap());
        let b = load_u8x16_i16_avx2((&filt[sa + 1..sa + 17]).try_into().unwrap());
        let v = _mm256_srli_epi16::<5>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(a, iw), _mm256_mullo_epi16(b, sw)),
            rnd,
        ));
        store_i16x16_u8_fixed((&mut dst_row[x..x + 16]).try_into().unwrap(), v);
        x += 16;
        xpos += 64 * 16;
    }
    while x + 8 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 2 < 0 || ti0 + 11 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 2) as usize;
        let a = load_u8x8_i16_fixed((&filt[sa..sa + 8]).try_into().unwrap());
        let b = load_u8x8_i16_fixed((&filt[sa + 1..sa + 9]).try_into().unwrap());
        let v = _mm_srli_epi16::<5>(_mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(a, _mm_set1_epi16((32 - shift) as i16)),
                _mm_mullo_epi16(b, _mm_set1_epi16(shift as i16)),
            ),
            _mm_set1_epi16(16),
        ));
        store_i16x8_u8_fixed((&mut dst_row[x..x + 8]).try_into().unwrap(), v);
        x += 8;
        xpos += 64 * 8;
    }
    while x < w {
        let base_x = xpos >> 6;
        let ti = top_off as i32 + base_x;
        let v = (32 - shift as i32) * filt[(ti + 2) as usize] as i32
            + shift as i32 * filt[(ti + 3) as usize] as i32;
        dst_row[x] = ((v + 16) >> 5).clamp(0, 255) as u8;
        x += 1;
        xpos += 64;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn ipred_z2_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    let a = angle & 511;
    if mrl_mul {
        let e_stride = (w + h) * 2 + mrl_idx * 3 + 1;
        let mut tmp = [0u8; 64 * 64];
        let base_angle = a | ANGLE_IS_LUMA;
        let first_angle = base_angle | ((mrl_idx as i32) << ANGLE_MRL_IDX_SHIFT);
        ipred_z2_8bpc_avx2(
            tmp.as_mut_slice(),
            64,
            tl,
            o,
            w,
            h,
            first_angle,
            max_width,
            max_height,
        );
        ipred_z2_8bpc_avx2(
            dst,
            stride,
            tl,
            o + e_stride,
            w,
            h,
            base_angle,
            max_width,
            max_height,
        );
        avg_pred_8bpc_avx2(dst, stride, tmp.as_slice(), w, h);
        return;
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(a - 90) as usize] as i32;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[(180 - a) as usize] as i32;

    // Top edge buffer.
    let mut filt = [0u8; 72];
    let top_off = mrl_idx;
    let sz_t = 1 + w + mrl_idx;
    let str_t = if enable_intra_edge_filter && have_top && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, a - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        crate::ipred::filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz_t as i32,
            str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&tl[o..o + sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // Left edge buffer.
    let mut filt2 = [0u8; 72];
    let left_off: usize = h + 2;
    let sz_l = 1 + h + mrl_idx;
    let str_l = if enable_intra_edge_filter && have_left && mrl_idx == 0 {
        crate::ipred::filter_strength((w + h) as i32, 180 - a, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        crate::ipred::filter_edge(
            &mut filt2[1..],
            sz_l,
            h as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - (h + mrl_idx)..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - (h + mrl_idx)..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..h {
        let ypos = (y + 1) as i32;
        let mut xpos = -(ypos + mrl_idx as i32) * dx;
        let mut x = 0usize;
        let dst_row = &mut dst[y * stride..y * stride + w];

        // Left reference span (scalar: shift varies per pixel).
        while x < w && xpos < -(64 * (1 + mrl_idx as i32)) {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - (xpos_l + mrl_idx as i32) * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            if is_luma {
                let f = &crate::ipred::DR_INTERP_FILTER[shift];
                let v = f.a as i32 * filt2[bi - 1] as i32
                    + f.b as i32 * filt2[bi - 2] as i32
                    + f.c as i32 * filt2[bi - 3] as i32
                    + f.d as i32 * filt2[bi - 4] as i32;
                dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
            } else {
                let v = (32 - shift as i32) * filt2[bi - 2] as i32
                    + shift as i32 * filt2[bi - 3] as i32;
                dst_row[x] = ((v + 16) >> 5).clamp(0, 255) as u8;
            }
            x += 1;
            xpos += 64;
        }

        // Top reference span (shift constant for the row).
        if x < w {
            let shift = ((xpos & 0x3F) >> 1) as usize;
            if is_luma {
                let f = &crate::ipred::DR_INTERP_FILTER[shift];
                z2_top_span_avx2(&filt, top_off, xpos, f, dst_row, x, w);
            } else {
                z2_top_span_chroma_avx2(&filt, top_off, xpos, shift, dst_row, x, w);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dip_dot_8bpc_avx2(inp8: __m256i, inp: &[i32; 11], weights: &[u16; 11]) -> i32 {
    let w8 = _mm256_cvtepu16_epi32(unsafe { _mm_loadu_si128(weights.as_ptr() as *const __m128i) });
    let mut s = _mm256_hsum_epi32(_mm256_mullo_epi32(inp8, w8)) as i32;
    s += weights[8] as i32 * inp[8];
    s += weights[9] as i32 * inp[9];
    s += weights[10] as i32 * inp[10];
    s
}

#[inline]
#[target_feature(enable = "avx2")]
fn dip_vertical_interp_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    step_y: usize,
    uhl2: i32,
    grid_h: usize,
) {
    if step_y <= 1 {
        return;
    }
    let mut p0_buf = [0u8; 128];
    let mut p1_buf = [0u8; 128];
    for gy in 0..grid_h {
        let base_y = gy * step_y;
        let sparse_y = base_y + step_y - 1;
        if gy == 0 {
            p0_buf[..w].copy_from_slice(&tl[o + 1..o + 1 + w]);
        } else {
            let prev = (base_y - 1) * stride;
            p0_buf[..w].copy_from_slice(&dst[prev..prev + w]);
        }
        let p1_off = sparse_y * stride;
        p1_buf[..w].copy_from_slice(&dst[p1_off..p1_off + w]);

        for z in 0..step_y - 1 {
            let z1 = (z + 1) as i16;
            let w0 = _mm256_set1_epi16((step_y as i16) - z1);
            let w1 = _mm256_set1_epi16(z1);
            let sh = _mm_cvtsi32_si128(uhl2);
            let row_off = (base_y + z) * stride;
            let row = &mut dst[row_off..row_off + w];
            let mut x = 0usize;
            while x + 16 <= w {
                let a = unsafe { _mm_loadu_si128(p0_buf[x..].as_ptr() as *const __m128i) };
                let b = unsafe { _mm_loadu_si128(p1_buf[x..].as_ptr() as *const __m128i) };
                let al = _mm256_cvtepu8_epi16(a);
                let bl = _mm256_cvtepu8_epi16(b);
                let r = _mm256_srl_epi16(
                    _mm256_add_epi16(_mm256_mullo_epi16(al, w0), _mm256_mullo_epi16(bl, w1)),
                    sh,
                );
                store_i16x16_u8_fixed((&mut row[x..x + 16]).try_into().unwrap(), r);
                x += 16;
            }
            while x + 8 <= w {
                let a = _mm_cvtepu8_epi16(unsafe {
                    _mm_loadl_epi64(p0_buf[x..].as_ptr() as *const __m128i)
                });
                let b = _mm_cvtepu8_epi16(unsafe {
                    _mm_loadl_epi64(p1_buf[x..].as_ptr() as *const __m128i)
                });
                let w0_128 = _mm_set1_epi16((step_y as i16) - z1);
                let w1_128 = _mm_set1_epi16(z1);
                let r = _mm_srl_epi16(
                    _mm_add_epi16(_mm_mullo_epi16(a, w0_128), _mm_mullo_epi16(b, w1_128)),
                    sh,
                );
                store_i16x8_u8_fixed((&mut row[x..x + 8]).try_into().unwrap(), r);
                x += 8;
            }
            while x < w {
                row[x] = ((p0_buf[x] as i32 * (step_y as i32 - z as i32 - 1)
                    + p1_buf[x] as i32 * (z as i32 + 1))
                    >> uhl2) as u8;
                x += 1;
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn ipred_dip_8bpc_avx2(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    mode: i32,
) {
    let trans = (mode & 16) != 0;
    let wd = width >> 2;
    let hd = height >> 2;
    let wl2 = ulog2(wd as u32);
    let hl2 = ulog2(hd as u32);
    let wrnd = width >> 3;
    let hrnd = height >> 3;
    let i_t: usize = if trans { 5 } else { 1 };
    let i_l: usize = if trans { 1 } else { 5 };
    let mut inp = [0i32; 11];
    inp[0] = tl[o] as i32;
    let mut in_sum = inp[0];

    let mut ti = o + 1;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..wd {
            sum += tl[ti] as i32;
            ti += 1;
        }
        inp[i_t + i] = (sum + wrnd as i32) >> wl2;
        in_sum += inp[i_t + i];
    }

    let mut li = o;
    for i in 0..4 {
        let mut sum = 0i32;
        for _ in 0..hd {
            li -= 1;
            sum += tl[li] as i32;
        }
        inp[i_l + i] = (sum + hrnd as i32) >> hl2;
        in_sum += inp[i_l + i];
    }

    let mut sum = 0i32;
    for x in 0..wd {
        sum += tl[o + x + width + 1] as i32;
    }
    let idx_tr = if trans { 10 } else { 9 };
    inp[idx_tr] = (sum + wrnd as i32) >> wl2;
    in_sum += inp[idx_tr];

    sum = 0;
    for y in 0..hd {
        sum += tl[o - (y + height + 1)] as i32;
    }
    let idx_bl = if trans { 9 } else { 10 };
    inp[idx_bl] = (sum + hrnd as i32) >> hl2;
    in_sum += inp[idx_bl];

    let m = (mode & 7) as usize;
    let mut uwl2 = wl2 - 1;
    let mut dwl2 = 0i32;
    if uwl2 < 0 {
        dwl2 = -uwl2;
        uwl2 = 0;
    }
    let step_x = 1usize << uwl2;
    let dw = 1usize << dwl2;
    let mut uhl2 = hl2 - 1;
    let mut dhl2 = 0i32;
    if uhl2 < 0 {
        dhl2 = -uhl2;
        uhl2 = 0;
    }
    let step_y = 1usize << uhl2;
    let dh = 1usize << dhl2;
    let grid_h = 8usize >> dhl2;
    let grid_w = 8usize >> dwl2;
    let inp8 = unsafe { _mm256_loadu_si256(inp.as_ptr() as *const __m256i) };

    let mut y = step_y - 1;
    for gy in 0..grid_h {
        let iy = gy * dh;
        let mut x = step_x - 1;
        let dst_row = &mut dst[y * stride..y * stride + width];
        for gx in 0..grid_w {
            let ix = gx * dw;
            let idx = if trans { ix * 8 + iy } else { iy * 8 + ix };
            let s = dip_dot_8bpc_avx2(inp8, &inp, &DIP_WEIGHTS[m][idx]);
            dst_row[x] = (((s + 2048) >> 12) - in_sum).clamp(0, 255) as u8;
            x += step_x;
        }
        y += step_y;
    }

    if step_x > 1 {
        y = step_y - 1;
        for _gy in 0..grid_h {
            let mut p1 = tl[o - (y + 1)] as i32;
            let mut x = 0usize;
            let dst_row = &mut dst[y * stride..y * stride + width];
            for _gx in 0..grid_w {
                let p0 = p1;
                p1 = dst_row[x + step_x - 1] as i32;
                for z in 0..step_x - 1 {
                    let z1 = (z + 1) as i32;
                    dst_row[x + z] = ((p0 * (step_x as i32 - z1) + p1 * z1) >> uwl2) as u8;
                }
                x += step_x;
            }
            y += step_y;
        }
    }

    dip_vertical_interp_8bpc_avx2(dst, stride, tl, o, width, step_y, uhl2, grid_h);
}

#[cfg(test)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    // Build a top-left edge buffer with `o` at the corner: left = tl[o-1-y],
    // top = tl[o+1+x], topleft = tl[o].
    fn make_tl(w: usize, h: usize, seed: u64) -> (Vec<u8>, usize) {
        let o = h + 1;
        let len = o + 1 + w + 16;
        let mut tl = vec![0u8; len];
        let mut s = seed;
        for v in tl.iter_mut() {
            *v = lcg(&mut s);
        }
        (tl, o)
    }

    const SIZES: &[(usize, usize)] = &[
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (32, 16),
        (8, 32),
        (16, 4),
        (4, 16),
        (4, 8),
        (8, 4),
        (16, 64),
    ];

    #[test]
    fn paeth_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x1234 + w as u64 * 131 + h as u64);
            let stride = w;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_paeth_8bpc(&mut a, stride, &tl, o, w, h);
            unsafe {
                ipred_paeth_8bpc_avx2(&mut b, stride, &tl, o, w, h);
            }
            assert_eq!(a, b, "paeth mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn dc_family_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x9999 + w as u64 * 7 + h as u64);
            let stride = w;
            let angle = 0; // non-IBP path

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_dc_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "dc mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_top_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_dc_top_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "dc_top mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_left_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_dc_left_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "dc_left mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_128_8bpc(&mut a, stride, w, h);
            unsafe {
                ipred_dc_128_8bpc_avx2(&mut b, stride, w, h);
            }
            assert_eq!(a, b, "dc_128 mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn smooth_family_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x5151 + w as u64 * 17 + h as u64);
            let stride = w;
            let cases: &[(
                &str,
                fn(&mut [u8], usize, &[u8], usize, usize, usize),
                unsafe fn(&mut [u8], usize, &[u8], usize, usize, usize),
            )] = &[
                (
                    "smooth",
                    crate::ipred::ipred_smooth_8bpc,
                    ipred_smooth_8bpc_avx2,
                ),
                (
                    "smooth_v",
                    crate::ipred::ipred_smooth_v_8bpc,
                    ipred_smooth_v_8bpc_avx2,
                ),
                (
                    "smooth_h",
                    crate::ipred::ipred_smooth_h_8bpc,
                    ipred_smooth_h_8bpc_avx2,
                ),
            ];
            for (name, scalar, simd) in cases {
                let mut a = vec![0u8; stride * h];
                let mut b = vec![0u8; stride * h];
                scalar(&mut a, stride, &tl, o, w, h);
                unsafe {
                    simd(&mut b, stride, &tl, o, w, h);
                }
                assert_eq!(a, b, "{} mismatch w={} h={}", name, w, h);
            }
        }
    }

    #[test]
    fn v_h_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x2727 + w as u64 * 23 + h as u64);
            let stride = w;
            let angle = 0;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_v_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_v_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "v mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_h_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_h_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "h mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn v_h_mrl_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::levels::ANGLE_MULTI_MRL_FLAG;
        for &(w, h) in SIZES {
            let e_stride = (w + h) * 2 + 1;
            let o = h + 1;
            let len = o + 1 + e_stride + w + 32;
            let mut tl = vec![0u8; len];
            let mut s = 0x3131 + w as u64 * 5 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            let angle = ANGLE_MULTI_MRL_FLAG;

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_v_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_v_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "v_mrl mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_h_8bpc(&mut a, stride, &tl, o, w, h, angle);
            unsafe {
                ipred_h_8bpc_avx2(&mut b, stride, &tl, o, w, h, angle);
            }
            assert_eq!(a, b, "h_mrl mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn z1_luma_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::levels::*;
        let ibp = [[[0u8; 16]; 16]; 7];
        let base_flags = ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG;
        for &(w, h) in SIZES {
            let sz = 1 + w + h;
            let o = 16usize;
            let maxd = (w + h) as i32;
            let len = o + sz + (w + h) + 64;
            let mut tl = vec![0u8; len];
            let mut s = 0x7a7a + w as u64 * 13 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            for a in [3i32, 30, 45, 63, 87] {
                for extra in [
                    0,
                    ANGLE_USE_EDGE_FILTER_FLAG,
                    ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_SMOOTH_TOP_EDGE_FLAG,
                ] {
                    let angle = a | base_flags | extra;
                    let mut sa = vec![0u8; stride * h];
                    let mut sb = vec![0u8; stride * h];
                    crate::ipred::ipred_z1_8bpc(
                        &mut sa, stride, &tl, o, w, h, angle, maxd, maxd, &ibp,
                    );
                    unsafe {
                        ipred_z1_8bpc_avx2(&mut sb, stride, &tl, o, w, h, angle, maxd, maxd, &ibp);
                    }
                    assert_eq!(
                        sa, sb,
                        "z1 mismatch w={} h={} a={} extra={:#x}",
                        w, h, a, extra
                    );
                }
            }
        }
    }

    #[test]
    fn z3_luma_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::levels::*;
        let ibp = [[[0u8; 16]; 16]; 7];
        let base_flags = ANGLE_IS_LUMA | ANGLE_HAS_LEFT_FLAG;
        for &(w, h) in SIZES {
            let o = w + h + 8;
            let maxd = (w + h) as i32;
            let len = o + 16;
            let mut tl = vec![0u8; len];
            let mut s = 0x5d5d + w as u64 * 11 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            for a in [183i32, 200, 225, 250, 267] {
                for extra in [
                    0,
                    ANGLE_USE_EDGE_FILTER_FLAG,
                    ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_SMOOTH_LEFT_EDGE_FLAG,
                ] {
                    let angle = a | base_flags | extra;
                    let mut sa = vec![0u8; stride * h];
                    let mut sb = vec![0u8; stride * h];
                    crate::ipred::ipred_z3_8bpc(
                        &mut sa, stride, &tl, o, w, h, angle, maxd, maxd, &ibp,
                    );
                    unsafe {
                        ipred_z3_8bpc_avx2(&mut sb, stride, &tl, o, w, h, angle, maxd, maxd, &ibp);
                    }
                    assert_eq!(
                        sa, sb,
                        "z3 mismatch w={} h={} a={} extra={:#x}",
                        w, h, a, extra
                    );
                }
            }
        }
    }

    #[test]
    fn z2_luma_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::levels::*;
        let base_flags = ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        for &(w, h) in SIZES {
            let o = h + 8;
            let maxd = (w + h) as i32;
            let len = o + 1 + w + 16;
            let mut tl = vec![0u8; len];
            let mut s = 0x6c6c + w as u64 * 19 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            for a in [100i32, 120, 135, 150, 170] {
                for extra in [
                    0,
                    ANGLE_USE_EDGE_FILTER_FLAG,
                    ANGLE_USE_EDGE_FILTER_FLAG
                        | ANGLE_SMOOTH_TOP_EDGE_FLAG
                        | ANGLE_SMOOTH_LEFT_EDGE_FLAG,
                ] {
                    let angle = a | base_flags | extra;
                    let mut sa = vec![0u8; stride * h];
                    let mut sb = vec![0u8; stride * h];
                    crate::ipred::ipred_z2_8bpc(&mut sa, stride, &tl, o, w, h, angle, maxd, maxd);
                    unsafe {
                        ipred_z2_8bpc_avx2(&mut sb, stride, &tl, o, w, h, angle, maxd, maxd);
                    }
                    assert_eq!(
                        sa, sb,
                        "z2 mismatch w={} h={} a={} extra={:#x}",
                        w, h, a, extra
                    );
                }
            }
        }
    }
}
