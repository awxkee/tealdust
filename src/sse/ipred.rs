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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::intops::ulog2;
use crate::levels::ANGLE_MULTI_MRL_FLAG;
use crate::tables::SM_WEIGHTS;

#[inline(always)]
unsafe fn fill_row_u8(row: &mut [u8], v: u8) {
    let vv = unsafe { _mm_set1_epi8(v as i8) };
    let mut x = 0usize;
    while x + 16 <= row.len() {
        unsafe { _mm_storeu_si128(row.as_mut_ptr().add(x) as *mut __m128i, vv) };
        x += 16;
    }
    if x < row.len() {
        row[x..].fill(v);
    }
}

#[inline(always)]
fn load_u8x8_i16(ptr: *const u8) -> __m128i {
    let bytes = unsafe { _mm_loadl_epi64(ptr as *const __m128i) };
    unsafe { _mm_cvtepu8_epi16(bytes) }
}

#[inline(always)]
fn store_i16x8_u8(ptr: *mut u8, v: __m128i) {
    let packed = unsafe { _mm_packus_epi16(v, _mm_setzero_si128()) };
    unsafe { _mm_storel_epi64(ptr as *mut __m128i, packed) };
}

#[inline(always)]
fn sra_i16(v: __m128i, shift: i32) -> __m128i {
    unsafe { _mm_sra_epi16(v, _mm_cvtsi32_si128(shift)) }
}

#[target_feature(enable = "sse4.1")]
fn ipred_v_8bpc_sse41_impl(
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
        let mut x = 0usize;
        while x + 16 <= width {
            let a = unsafe { _mm_loadu_si128(tl.as_ptr().add(o + 1 + x) as *const __m128i) };
            let b =
                unsafe { _mm_loadu_si128(tl.as_ptr().add(o + 1 + e_stride + x) as *const __m128i) };
            let v = _mm_avg_epu8(a, b);
            unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(x) as *mut __m128i, v) };
            x += 16;
        }
        while x < width {
            let top = tl[o + 1 + x] as u16;
            let top2 = tl[o + 1 + e_stride + x] as u16;
            dst[x] = ((top + top2 + 1) >> 1) as u8;
            x += 1;
        }
    } else {
        let top = &tl[o + 1..o + 1 + width];
        dst.copy_from_slice(top);
    }

    let mut off = stride;
    for _ in 1..height {
        dst.copy_within(0..width, off);
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_h_8bpc_sse41_impl(
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
        let row = &mut dst[off..off + width];
        unsafe { fill_row_u8(row, v) };
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_smooth_v_8bpc_sse41_impl(
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
    let rnd = _mm_set1_epi16((h >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom = tl[o - h - 1] as i16;
    let bottom_v = _mm_set1_epi16(bottom);
    let sh = bhl2;

    let mut off = 0usize;
    for y in 0..h {
        let off_y = _mm_set1_epi16((h - 1 - y) as i16);
        let w_ver = _mm_set1_epi16(weights[y] as i16);
        let row = &mut dst[off..off + w];
        let mut x = 0usize;
        while x + 8 <= w {
            let above = unsafe { load_u8x8_i16(tl.as_ptr().add(o + 1 + x)) };
            let mul = _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v), off_y);
            let pred = _mm_add_epi16(bottom_v, sra_i16(_mm_add_epi16(mul, rnd), sh));
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(above, pred), w_ver),
                _mm_set1_epi16(32),
            ));
            let out = _mm_add_epi16(pred, adj);
            unsafe { store_i16x8_u8(row.as_mut_ptr().add(x), out) };
            x += 8;
        }
        while x < w {
            let above = tl[o + 1 + x] as i32;
            let mul = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let pred = bottom as i32 + ((mul + (h >> 1) as i32) >> bhl2);
            row[x] = (pred + (((above - pred) * weights[y] as i32 + 32) >> 6)) as u8;
            x += 1;
        }
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_smooth_h_8bpc_sse41_impl(
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
    let rnd = _mm_set1_epi16((w >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let right_v = _mm_set1_epi16(right);

    let mut off = 0usize;
    for y in 0..h {
        let left = tl[o - 1 - y] as i16;
        let left_v = _mm_set1_epi16(left);
        let diff = _mm_set1_epi16(left - right);
        let row = &mut dst[off..off + w];
        let mut x = 0usize;
        while x + 8 <= w {
            let x0 = (w - 1 - x) as i16;
            let dist = _mm_setr_epi16(x0, x0 - 1, x0 - 2, x0 - 3, x0 - 4, x0 - 5, x0 - 6, x0 - 7);
            let wx = unsafe { load_u8x8_i16(weights.as_ptr().add(x)) };
            let pred = _mm_add_epi16(
                right_v,
                sra_i16(_mm_add_epi16(_mm_mullo_epi16(diff, dist), rnd), bwl2),
            );
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(left_v, pred), wx),
                _mm_set1_epi16(32),
            ));
            let out = _mm_add_epi16(pred, adj);
            unsafe { store_i16x8_u8(row.as_mut_ptr().add(x), out) };
            x += 8;
        }
        while x < w {
            let mul = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let pred = right as i32 + ((mul + (w >> 1) as i32) >> bwl2);
            row[x] = (pred + (((left as i32 - pred) * weights[x] as i32 + 32) >> 6)) as u8;
            x += 1;
        }
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_smooth_8bpc_sse41_impl(
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
    let rnd_ver = _mm_set1_epi16((h >> 1) as i16);
    let rnd_hor = _mm_set1_epi16((w >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let bottom = tl[o - h - 1] as i16;
    let right_v = _mm_set1_epi16(right);
    let bottom_v = _mm_set1_epi16(bottom);

    let mut off = 0usize;
    for y in 0..h {
        let left = tl[o - 1 - y] as i16;
        let left_v = _mm_set1_epi16(left);
        let diff_hor = _mm_set1_epi16(left - right);
        let off_ver = _mm_set1_epi16((h - 1 - y) as i16);
        let w_ver = _mm_set1_epi16(weights[y] as i16);
        let row = &mut dst[off..off + w];
        let mut x = 0usize;
        while x + 8 <= w {
            let above = unsafe { load_u8x8_i16(tl.as_ptr().add(o + 1 + x)) };
            let x0 = (w - 1 - x) as i16;
            let dist = _mm_setr_epi16(x0, x0 - 1, x0 - 2, x0 - 3, x0 - 4, x0 - 5, x0 - 6, x0 - 7);
            let wx = unsafe { load_u8x8_i16(weights.as_ptr().add(x)) };

            let pred_ver = _mm_add_epi16(
                bottom_v,
                sra_i16(
                    _mm_add_epi16(
                        _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v), off_ver),
                        rnd_ver,
                    ),
                    bhl2,
                ),
            );
            let pred_hor = _mm_add_epi16(
                right_v,
                sra_i16(
                    _mm_add_epi16(_mm_mullo_epi16(diff_hor, dist), rnd_hor),
                    bwl2,
                ),
            );
            let pred_ver = _mm_add_epi16(
                pred_ver,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(above, pred_ver), w_ver),
                    _mm_set1_epi16(32),
                )),
            );
            let pred_hor = _mm_add_epi16(
                pred_hor,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(left_v, pred_hor), wx),
                    _mm_set1_epi16(32),
                )),
            );
            let out = _mm_srai_epi16::<1>(_mm_add_epi16(
                _mm_add_epi16(pred_ver, pred_hor),
                _mm_set1_epi16(1),
            ));
            unsafe { store_i16x8_u8(row.as_mut_ptr().add(x), out) };
            x += 8;
        }
        while x < w {
            let above = tl[o + 1 + x] as i32;
            let mul_ver = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let mul_hor = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let mut pred_ver = bottom as i32 + ((mul_ver + (h >> 1) as i32) >> bhl2);
            let mut pred_hor = right as i32 + ((mul_hor + (w >> 1) as i32) >> bwl2);
            pred_ver += ((above - pred_ver) * weights[y] as i32 + 32) >> 6;
            pred_hor += ((left as i32 - pred_hor) * weights[x] as i32 + 32) >> 6;
            row[x] = ((pred_ver + pred_hor + 1) >> 1) as u8;
            x += 1;
        }
        off += stride;
    }
}

pub(crate) fn ipred_v_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    unsafe { ipred_v_8bpc_sse41_impl(dst, stride, tl, o, width, height, angle) }
}

pub(crate) fn ipred_h_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    unsafe { ipred_h_8bpc_sse41_impl(dst, stride, tl, o, width, height, angle) }
}

pub(crate) fn ipred_smooth_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    unsafe { ipred_smooth_8bpc_sse41_impl(dst, stride, tl, o, width, height) }
}

pub(crate) fn ipred_smooth_v_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    unsafe { ipred_smooth_v_8bpc_sse41_impl(dst, stride, tl, o, width, height) }
}

pub(crate) fn ipred_smooth_h_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    unsafe { ipred_smooth_h_8bpc_sse41_impl(dst, stride, tl, o, width, height) }
}
