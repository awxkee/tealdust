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
        let top1 = &tl[o + 1..o + 1 + width];
        let top2 = &tl[o + 1 + e_stride..o + 1 + e_stride + width];
        let (dc, drem) = dst[..width].as_chunks_mut::<16>();
        for ((d, a), b) in dc
            .iter_mut()
            .zip(top1.as_chunks::<16>().0.iter())
            .zip(top2.as_chunks::<16>().0.iter())
        {
            store_u8x16_fixed(d, _mm_avg_epu8(load_u8x16_fixed(a), load_u8x16_fixed(b)));
        }
        let base_x = (width / 16) * 16;
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
        row.fill(v);
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
        let top_src = &tl[o + 1..o + 1 + w];
        let (rc, rrem) = row.as_chunks_mut::<8>();
        for (d, t) in rc.iter_mut().zip(top_src.as_chunks::<8>().0.iter()) {
            let above = load_u8x8_i16_fixed(t);
            let mul = _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v), off_y);
            let pred = _mm_add_epi16(bottom_v, sra_i16(_mm_add_epi16(mul, rnd), sh));
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(above, pred), w_ver),
                _mm_set1_epi16(32),
            ));
            store_i16x8_u8_fixed(d, _mm_add_epi16(pred, adj));
        }
        let base_x = (w / 8) * 8;
        for (xi, d) in rrem.iter_mut().enumerate() {
            let x = base_x + xi;
            let above = tl[o + 1 + x] as i32;
            let mul = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let pred = bottom as i32 + ((mul + (h >> 1) as i32) >> bhl2);
            *d = (pred + (((above - pred) * weights[y] as i32 + 32) >> 6)) as u8;
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
        let (rc, rrem) = row.as_chunks_mut::<8>();
        for (ci, (d, wxc)) in rc
            .iter_mut()
            .zip(weights[..w].as_chunks::<8>().0.iter())
            .enumerate()
        {
            let x0 = (w - 1 - ci * 8) as i16;
            let dist = _mm_setr_epi16(x0, x0 - 1, x0 - 2, x0 - 3, x0 - 4, x0 - 5, x0 - 6, x0 - 7);
            let wx = load_u8x8_i16_fixed(wxc);
            let pred = _mm_add_epi16(
                right_v,
                sra_i16(_mm_add_epi16(_mm_mullo_epi16(diff, dist), rnd), bwl2),
            );
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(left_v, pred), wx),
                _mm_set1_epi16(32),
            ));
            store_i16x8_u8_fixed(d, _mm_add_epi16(pred, adj));
        }
        let base_x = (w / 8) * 8;
        for (xi, d) in rrem.iter_mut().enumerate() {
            let x = base_x + xi;
            let mul = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let pred = right as i32 + ((mul + (w >> 1) as i32) >> bwl2);
            *d = (pred + (((left as i32 - pred) * weights[x] as i32 + 32) >> 6)) as u8;
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
        let top_src = &tl[o + 1..o + 1 + w];
        let (rc, rrem) = row.as_chunks_mut::<8>();
        for (ci, ((d, t), wxc)) in rc
            .iter_mut()
            .zip(top_src.as_chunks::<8>().0.iter())
            .zip(weights[..w].as_chunks::<8>().0.iter())
            .enumerate()
        {
            let above = load_u8x8_i16_fixed(t);
            let x0 = (w - 1 - ci * 8) as i16;
            let dist = _mm_setr_epi16(x0, x0 - 1, x0 - 2, x0 - 3, x0 - 4, x0 - 5, x0 - 6, x0 - 7);
            let wx = load_u8x8_i16_fixed(wxc);

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
            store_i16x8_u8_fixed(
                d,
                _mm_srai_epi16::<1>(_mm_add_epi16(
                    _mm_add_epi16(pred_ver, pred_hor),
                    _mm_set1_epi16(1),
                )),
            );
        }
        let base_x = (w / 8) * 8;
        for (xi, d) in rrem.iter_mut().enumerate() {
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

// ---------------------------------------------------------------------------
// DC family (dc / dc_top / dc_left / dc_128), 8bpc.
//
// The SIMD work is the edge-pixel reduction (`_mm_sad_epu8`) plus a broadcast
// fill; the rounding/division matches the scalar path bit-for-bit. Blocks that
// request the intra-boundary (IBP) per-pixel blend fall back to scalar.
// ---------------------------------------------------------------------------

use crate::levels::ANGLE_IBP_FLAG;

#[inline(always)]
fn load_u8x16_fixed(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_u8x16_fixed(a: &mut [u8; 16], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

/// Horizontal sum of all bytes in `s` (each lane widened to u32).
#[inline]
#[target_feature(enable = "sse4.1")]
fn sum_u8_sse41(s: &[u8]) -> u32 {
    let zero = unsafe { _mm_setzero_si128() };
    let mut acc = zero;
    let (chunks, rem) = s.as_chunks::<16>();
    for c in chunks.iter() {
        // SAD against zero == sum of the 16 bytes, placed in lanes 0 and 2.
        acc = unsafe { _mm_add_epi64(acc, _mm_sad_epu8(load_u8x16_fixed(c), zero)) };
    }
    let mut total = unsafe { (_mm_extract_epi32::<0>(acc) + _mm_extract_epi32::<2>(acc)) as u32 };
    for &b in rem {
        total += b as u32;
    }
    total
}

/// Fill a `w x h` block at `off` with the constant byte `dc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn splat_fill_sse41(dst: &mut [u8], stride: usize, off: usize, w: usize, h: usize, dc: u8) {
    let v = unsafe { _mm_set1_epi8(dc as i8) };
    let mut p = off;
    for _ in 0..h {
        let (chunks, rem) = dst[p..p + w].as_chunks_mut::<16>();
        for c in chunks.iter_mut() {
            store_u8x16_fixed(c, v);
        }
        rem.fill(dc);
        p += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_128_8bpc_sse41_impl(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    splat_fill_sse41(dst, stride, 0, w, h, 128);
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_top_8bpc_sse41_impl(
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
    let sum = sum_u8_sse41(&tl[o + 1..o + 1 + w]);
    let dc = (((w >> 1) as u32 + sum) >> (w as u32).trailing_zeros()) as u8;
    splat_fill_sse41(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_left_8bpc_sse41_impl(
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
    let sum = sum_u8_sse41(&tl[o - h..o]);
    let dc = (((h >> 1) as u32 + sum) >> (h as u32).trailing_zeros()) as u8;
    splat_fill_sse41(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_8bpc_sse41_impl(
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
    let sum = sum_u8_sse41(&tl[o + 1..o + 1 + w]) + sum_u8_sse41(&tl[o - h..o]);
    let dc = if n_pel & (n_pel - 1) == 0 {
        (sum + w as u32) >> n_pel.trailing_zeros()
    } else {
        crate::ipred::fast_div32_dc(sum, n_pel).min(255)
    } as u8;
    splat_fill_sse41(dst, stride, 0, w, h, dc);
}

pub(crate) fn ipred_dc_128_8bpc_sse41(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    unsafe { ipred_dc_128_8bpc_sse41_impl(dst, stride, w, h) }
}

pub(crate) fn ipred_dc_top_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_top_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle) }
}

pub(crate) fn ipred_dc_left_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_left_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle) }
}

pub(crate) fn ipred_dc_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle) }
}

// ---------------------------------------------------------------------------
// Paeth predictor, 8bpc. Per pixel: base = left + top - topleft; pick whichever
// of {left, top, topleft} is closest to `base` (ties prefer left, then top).
// ---------------------------------------------------------------------------

#[inline(always)]
fn load_u8x8_i16_fixed(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i16x8_u8_fixed(a: &mut [u8; 8], v: __m128i) {
    let packed = unsafe { _mm_packus_epi16(v, _mm_setzero_si128()) };
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, packed) };
}

#[target_feature(enable = "sse4.1")]
fn ipred_paeth_8bpc_sse41_impl(
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
    let tl_v = unsafe { _mm_set1_epi16(topleft as i16) };
    let base_x = (w / 8) * 8;
    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let left_v = unsafe { _mm_set1_epi16(left as i16) };
        let top_src = &tl[o + 1..o + 1 + w];
        let (rc, rrem) = dst[off..off + w].as_chunks_mut::<8>();
        for (d, t) in rc.iter_mut().zip(top_src.as_chunks::<8>().0.iter()) {
            let top_v = load_u8x8_i16_fixed(t);
            unsafe {
                let base = _mm_sub_epi16(_mm_add_epi16(left_v, top_v), tl_v);
                let ld = _mm_abs_epi16(_mm_sub_epi16(left_v, base));
                let td = _mm_abs_epi16(_mm_sub_epi16(top_v, base));
                let tld = _mm_abs_epi16(_mm_sub_epi16(tl_v, base));
                let cond_l = _mm_and_si128(
                    _mm_cmpeq_epi16(ld, _mm_min_epi16(ld, td)),
                    _mm_cmpeq_epi16(ld, _mm_min_epi16(ld, tld)),
                );
                let cond_t = _mm_cmpeq_epi16(td, _mm_min_epi16(td, tld));
                let inner = _mm_blendv_epi8(tl_v, top_v, cond_t);
                let res = _mm_blendv_epi8(inner, left_v, cond_l);
                store_i16x8_u8_fixed(d, res);
            }
        }
        // scalar remainder columns
        for (x, d) in rrem.iter_mut().enumerate() {
            let top = tl[o + 1 + base_x + x] as i32;
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

pub(crate) fn ipred_paeth_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    unsafe { ipred_paeth_8bpc_sse41_impl(dst, stride, tl, o, w, h) }
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
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x1234 + w as u64 * 131 + h as u64);
            let stride = w;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_paeth_8bpc(&mut a, stride, &tl, o, w, h);
            ipred_paeth_8bpc_sse41(&mut b, stride, &tl, o, w, h);
            assert_eq!(a, b, "paeth mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn dc_family_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x9999 + w as u64 * 7 + h as u64);
            let stride = w;
            let angle = 0; // non-IBP path

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_dc_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "dc mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_top_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_dc_top_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "dc_top mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_left_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_dc_left_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "dc_left mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_128_8bpc(&mut a, stride, w, h);
            ipred_dc_128_8bpc_sse41(&mut b, stride, w, h);
            assert_eq!(a, b, "dc_128 mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn smooth_family_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x5151 + w as u64 * 17 + h as u64);
            let stride = w;
            let cases: &[(
                &str,
                fn(&mut [u8], usize, &[u8], usize, usize, usize),
                fn(&mut [u8], usize, &[u8], usize, usize, usize),
            )] = &[
                (
                    "smooth",
                    crate::ipred::ipred_smooth_8bpc,
                    ipred_smooth_8bpc_sse41,
                ),
                (
                    "smooth_v",
                    crate::ipred::ipred_smooth_v_8bpc,
                    ipred_smooth_v_8bpc_sse41,
                ),
                (
                    "smooth_h",
                    crate::ipred::ipred_smooth_h_8bpc,
                    ipred_smooth_h_8bpc_sse41,
                ),
            ];
            for (name, scalar, simd) in cases {
                let mut a = vec![0u8; stride * h];
                let mut b = vec![0u8; stride * h];
                scalar(&mut a, stride, &tl, o, w, h);
                simd(&mut b, stride, &tl, o, w, h);
                assert_eq!(a, b, "{} mismatch w={} h={}", name, w, h);
            }
        }
    }

    #[test]
    fn v_h_match_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x2727 + w as u64 * 23 + h as u64);
            let stride = w;
            let angle = 0;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_v_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_v_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "v mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_h_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_h_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "h mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn v_h_mrl_match_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
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
            ipred_v_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "v_mrl mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_h_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_h_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "h_mrl mismatch w={} h={}", w, h);
        }
    }
}
