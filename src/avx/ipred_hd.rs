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

use std::arch::x86_64::*;

use crate::levels::{ANGLE_IBP_FLAG, ANGLE_MULTI_MRL_FLAG};

#[inline(always)]
fn load_u16x16(a: &[u16; 16]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn store_u16x16(a: &mut [u16; 16], v: __m256i) {
    unsafe { _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v) };
}

#[inline(always)]
const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn _mm_hsumv_epi32(v: __m128i) -> __m128i {
    let mut hi = _mm_shuffle_epi32(v, shuffle(0, 0, 3, 2));
    let mut v = _mm_add_epi32(v, hi);

    hi = _mm_shuffle_epi32(v, shuffle(0, 0, 0, 1));
    v = _mm_add_epi32(v, hi);
    v
}

#[inline]
#[target_feature(enable = "avx2")]
fn _mm256_hsum_epi32(v: __m256i) -> u32 {
    let lo = _mm_hsumv_epi32(_mm256_castsi256_si128(v));
    let hi = _mm_hsumv_epi32(_mm256_extracti128_si256::<1>(v));
    _mm_cvtsi128_si32(_mm_add_epi32(lo, hi)) as u32
}

#[target_feature(enable = "avx2")]
fn sum_u16_avx2(s: &[u16]) -> u32 {
    let mut acc = _mm256_setzero_si256();
    let (chunks, rem) = s.as_chunks::<16>();
    for c in chunks.iter() {
        let v = load_u16x16(c);
        acc = _mm256_add_epi32(acc, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v)));
        acc = _mm256_add_epi32(acc, _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v)));
    }
    let mut total = _mm256_hsum_epi32(acc);
    for &v in rem {
        total += v as u32;
    }
    total
}

#[target_feature(enable = "avx2")]
fn splat_fill_avx2(dst: &mut [u16], stride: usize, off: usize, w: usize, h: usize, dc: u16) {
    let v = _mm256_set1_epi16(dc as i16);
    let mut p = off;
    for _ in 0..h {
        let (chunks, rem) = dst[p..p + w].as_chunks_mut::<16>();
        for c in chunks.iter_mut() {
            store_u16x16(c, v);
        }
        rem.fill(dc);
        p += stride;
    }
}

#[target_feature(enable = "avx2")]
fn ipred_v_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if w < 16 {
        crate::sse::ipred_v_hbd_sse41(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (w + h) * 2 + 1;
        let top1 = &tl[o + 1..o + 1 + w];
        let top2 = &tl[o + 1 + e_stride..o + 1 + e_stride + w];
        let (chunks, rem) = dst[..w].as_chunks_mut::<16>();
        for ((d, a), b) in chunks
            .iter_mut()
            .zip(top1.as_chunks::<16>().0.iter())
            .zip(top2.as_chunks::<16>().0.iter())
        {
            store_u16x16(d, _mm256_avg_epu16(load_u16x16(a), load_u16x16(b)));
        }
        let base = chunks.len() * 16;
        for (i, d) in rem.iter_mut().enumerate() {
            let x = base + i;
            *d = ((top1[x] as u32 + top2[x] as u32 + 1) >> 1) as u16;
        }
    } else {
        dst[..w].copy_from_slice(&tl[o + 1..o + 1 + w]);
    }
    let mut off = stride;
    for _ in 1..h {
        dst.copy_within(0..w, off);
        off += stride;
    }
}

#[target_feature(enable = "avx2")]
fn ipred_h_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if w < 16 {
        crate::sse::ipred_h_hbd_sse41(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    let e_stride = (w + h) * 2 + 1;
    let mrl = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let mut off = 0usize;
    for y in 0..h {
        let v = if mrl {
            ((tl[o - 1 - y] as u32 + tl[o + e_stride - 1 - y] as u32 + 1) >> 1) as u16
        } else {
            tl[o - 1 - y]
        };
        let vv = _mm256_set1_epi16(v as i16);
        let row = &mut dst[off..off + w];
        let (chunks, rem) = row.as_chunks_mut::<16>();
        for c in chunks.iter_mut() {
            store_u16x16(c, vv);
        }
        rem.fill(v);
        off += stride;
    }
}

#[target_feature(enable = "avx2")]
fn ipred_dc_128_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    splat_fill_avx2(dst, stride, 0, w, h, (bitdepth_max + 1) >> 1);
}

#[target_feature(enable = "avx2")]
fn ipred_dc_top_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        crate::sse::ipred_dc_top_hbd_sse41(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    let dc = (((w >> 1) as u32 + sum_u16_avx2(&tl[o + 1..o + 1 + w]))
        >> (w as u32).trailing_zeros()) as u16;
    splat_fill_avx2(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "avx2")]
fn ipred_dc_left_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        crate::sse::ipred_dc_left_hbd_sse41(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    let dc =
        (((h >> 1) as u32 + sum_u16_avx2(&tl[o - h..o])) >> (h as u32).trailing_zeros()) as u16;
    splat_fill_avx2(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "avx2")]
fn ipred_dc_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        crate::sse::ipred_dc_hbd_sse41(dst, stride, tl, o, w, h, angle, bitdepth_max);
        return;
    }
    let n = (w + h) as u32;
    let sum = sum_u16_avx2(&tl[o + 1..o + 1 + w]) + sum_u16_avx2(&tl[o - h..o]);
    let dc = if n & (n - 1) == 0 {
        (sum + w as u32) >> n.trailing_zeros()
    } else {
        crate::ipred::fast_div32_dc(sum, n).min(bitdepth_max as u32)
    } as u16;
    splat_fill_avx2(dst, stride, 0, w, h, dc);
}

pub(crate) fn ipred_v_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_v_hbd_avx2_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_h_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_h_hbd_avx2_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_hbd_avx2_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_top_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_top_hbd_avx2_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_left_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_left_hbd_avx2_impl(dst, stride, tl, o, w, h, angle, bitdepth_max) }
}
pub(crate) fn ipred_dc_128_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    unsafe { ipred_dc_128_hbd_avx2_impl(dst, stride, w, h, bitdepth_max) }
}

pub(crate) fn ipred_paeth_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    crate::sse::ipred_paeth_hbd_sse41(dst, stride, tl, o, w, h, bitdepth_max)
}
pub(crate) fn ipred_smooth_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    crate::sse::ipred_smooth_hbd_sse41(dst, stride, tl, o, w, h, bitdepth_max)
}
pub(crate) fn ipred_smooth_v_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    crate::sse::ipred_smooth_v_hbd_sse41(dst, stride, tl, o, w, h, bitdepth_max)
}
pub(crate) fn ipred_smooth_h_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    bitdepth_max: u16,
) {
    crate::sse::ipred_smooth_h_hbd_sse41(dst, stride, tl, o, w, h, bitdepth_max)
}

#[inline(always)]
fn load_u16x8_i32_slice(s: &[u16]) -> __m256i {
    debug_assert!(s.len() >= 8);
    unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(s.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i32x8_u16_max(a: &mut [u16], v: __m256i, bitdepth_max: u16) {
    debug_assert!(a.len() >= 8);
    let v = unsafe {
        let zero = _mm256_setzero_si256();
        let maxv = _mm256_set1_epi32(bitdepth_max as i32);
        _mm256_min_epi32(_mm256_max_epi32(v, zero), maxv)
    };
    let packed =
        unsafe { _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v)) };
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, packed) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn dr_filter8_hbd_avx2(
    f: &crate::ipred::DrFilter4Tap,
    bitdepth_max: u16,
    a0: __m256i,
    a1: __m256i,
    a2: __m256i,
    a3: __m256i,
) -> __m256i {
    let acc = _mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_set1_epi32(f.a as i32), a0),
            _mm256_mullo_epi32(_mm256_set1_epi32(f.b as i32), a1),
        ),
        _mm256_add_epi32(
            _mm256_mullo_epi32(_mm256_set1_epi32(f.c as i32), a2),
            _mm256_mullo_epi32(_mm256_set1_epi32(f.d as i32), a3),
        ),
    );
    let v = _mm256_srai_epi32::<7>(_mm256_add_epi32(acc, _mm256_set1_epi32(64)));
    _mm256_min_epi32(
        _mm256_max_epi32(v, _mm256_setzero_si256()),
        _mm256_set1_epi32(bitdepth_max as i32),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn z1_luma_row_hbd_avx2(
    filt: &[u16],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u16,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u16],
    w: usize,
    bitdepth_max: u16,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let base_const = (top_off as i32 + base0) as usize;
    let mut x = 0usize;
    while x + 8 <= n_filter {
        let bi = base_const + x;
        let v = dr_filter8_hbd_avx2(
            f,
            bitdepth_max,
            load_u16x8_i32_slice(&filt[bi - 1..]),
            load_u16x8_i32_slice(&filt[bi..]),
            load_u16x8_i32_slice(&filt[bi + 1..]),
            load_u16x8_i32_slice(&filt[bi + 2..]),
        );
        store_i32x8_u16_max(&mut dst_row[x..], v, bitdepth_max);
        x += 8;
    }
    while x < n_filter {
        let bi = base_const + x;
        let v = f.a as i32 * filt[bi - 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi + 1] as i32
            + f.d as i32 * filt[bi + 2] as i32;
        dst_row[x] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
        x += 1;
    }
    dst_row[n_filter..w].fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
fn ipred_z1_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || enable_ibp || !is_luma || mrl_idx != 0 || w < 8 {
        return crate::sse::ipred_z1_hbd_sse41(
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
            bitdepth_max,
        );
    }

    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let a = angle & 511;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[a as usize] as i32;
    let max_base_x = (w + h) as i32 - 1;
    let mut filt = [0u16; 141];
    let top_off = 2usize;
    let sz = 1 + w + h;
    let str = if enable_intra_edge_filter && have_top {
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

    let mut ypos = dx;
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
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let dst_row = &mut dst[y * stride..y * stride + w];
        z1_luma_row_hbd_avx2(
            &filt,
            top_off,
            base0,
            max_base_x,
            fill,
            f,
            dst_row,
            w,
            bitdepth_max,
        );
        ypos += dx;
    }
}

#[inline(always)]
fn setr_i32x8(a0: i32, a1: i32, a2: i32, a3: i32, a4: i32, a5: i32, a6: i32, a7: i32) -> __m256i {
    unsafe { _mm256_setr_epi32(a0, a1, a2, a3, a4, a5, a6, a7) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn z3_luma_col_hbd_avx2(
    filt: &[u16],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u16,
    f: &crate::ipred::DrFilter4Tap,
    col: &mut [u16],
    h: usize,
    bitdepth_max: u16,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let lob = left_off as i32 - base0;
    let mut y = 0usize;
    while y + 8 <= n_filter {
        let bi = lob - y as i32;
        let v = dr_filter8_hbd_avx2(
            f,
            bitdepth_max,
            setr_i32x8(
                filt[(bi + 1) as usize] as i32,
                filt[bi as usize] as i32,
                filt[(bi - 1) as usize] as i32,
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
                filt[(bi - 4) as usize] as i32,
                filt[(bi - 5) as usize] as i32,
                filt[(bi - 6) as usize] as i32,
            ),
            setr_i32x8(
                filt[bi as usize] as i32,
                filt[(bi - 1) as usize] as i32,
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
                filt[(bi - 4) as usize] as i32,
                filt[(bi - 5) as usize] as i32,
                filt[(bi - 6) as usize] as i32,
                filt[(bi - 7) as usize] as i32,
            ),
            setr_i32x8(
                filt[(bi - 1) as usize] as i32,
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
                filt[(bi - 4) as usize] as i32,
                filt[(bi - 5) as usize] as i32,
                filt[(bi - 6) as usize] as i32,
                filt[(bi - 7) as usize] as i32,
                filt[(bi - 8) as usize] as i32,
            ),
            setr_i32x8(
                filt[(bi - 2) as usize] as i32,
                filt[(bi - 3) as usize] as i32,
                filt[(bi - 4) as usize] as i32,
                filt[(bi - 5) as usize] as i32,
                filt[(bi - 6) as usize] as i32,
                filt[(bi - 7) as usize] as i32,
                filt[(bi - 8) as usize] as i32,
                filt[(bi - 9) as usize] as i32,
            ),
        );
        store_i32x8_u16_max(&mut col[y..], v, bitdepth_max);
        y += 8;
    }
    while y < n_filter {
        let bi = (lob - y as i32) as usize;
        let v = f.a as i32 * filt[bi + 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi - 1] as i32
            + f.d as i32 * filt[bi - 2] as i32;
        col[y] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
        y += 1;
    }
    col[n_filter..h].fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
fn ipred_z3_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || enable_ibp || !is_luma || mrl_idx != 0 || !(8..=64).contains(&h) {
        return crate::sse::ipred_z3_hbd_sse41(
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
            bitdepth_max,
        );
    }

    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let a = angle & 511;
    let dy = crate::tables::DR_INTRA_DERIVATIVE[(270 - a) as usize] as i32;
    let max_base_y = (w + h) as i32 - 1;
    let mut filt = [0u16; 141];
    let left_off = 1 + w + h;
    let sz = 1 + w + h;
    let str = if enable_intra_edge_filter && have_left {
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

    let mut col = [0u16; 64];
    let mut ypos = dy;
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let base0 = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        z3_luma_col_hbd_avx2(
            &filt,
            left_off,
            base0,
            max_base_y,
            fill,
            f,
            &mut col,
            h,
            bitdepth_max,
        );
        for (y, &c) in col[..h].iter().enumerate() {
            dst[y * stride + x] = c;
        }
        ypos += dy;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn z2_top_span_hbd_avx2(
    filt: &[u16],
    top_off: usize,
    mut xpos: i32,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u16],
    x_start: usize,
    w: usize,
    bitdepth_max: u16,
) {
    let mut x = x_start;
    while x + 8 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        if ti0 + 1 < 0 || ti0 + 12 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 1) as usize;
        let v = dr_filter8_hbd_avx2(
            f,
            bitdepth_max,
            load_u16x8_i32_slice(&filt[sa..]),
            load_u16x8_i32_slice(&filt[sa + 1..]),
            load_u16x8_i32_slice(&filt[sa + 2..]),
            load_u16x8_i32_slice(&filt[sa + 3..]),
        );
        store_i32x8_u16_max(&mut dst_row[x..], v, bitdepth_max);
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
        dst_row[x] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
        x += 1;
        xpos += 64;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
fn ipred_z2_hbd_avx2_impl(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: u16,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || !is_luma || mrl_idx != 0 || w < 8 {
        return crate::sse::ipred_z2_hbd_sse41(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            bitdepth_max,
        );
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let a = angle & 511;
    let dy = crate::tables::DR_INTRA_DERIVATIVE[(a - 90) as usize] as i32;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[(180 - a) as usize] as i32;

    let mut filt = [0u16; 72];
    let top_off = 0usize;
    let sz_t = 1 + w;
    let str_t = if enable_intra_edge_filter && have_top {
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

    let mut filt2 = [0u16; 72];
    let left_off: usize = h + 2;
    let sz_l = 1 + h;
    let str_l = if enable_intra_edge_filter && have_left {
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
            &tl[o - h..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - h..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..h {
        let ypos = (y + 1) as i32;
        let mut xpos = -ypos * dx;
        let mut x = 0usize;
        let dst_row = &mut dst[y * stride..y * stride + w];
        while x < w && xpos < -64 {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - xpos_l * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            let v = f.a as i32 * filt2[bi - 1] as i32
                + f.b as i32 * filt2[bi - 2] as i32
                + f.c as i32 * filt2[bi - 3] as i32
                + f.d as i32 * filt2[bi - 4] as i32;
            dst_row[x] = (((v + 64) >> 7).clamp(0, bitdepth_max as i32)) as u16;
            x += 1;
            xpos += 64;
        }
        if x < w {
            let shift = ((xpos & 0x3F) >> 1) as usize;
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z2_top_span_hbd_avx2(&filt, top_off, xpos, f, dst_row, x, w, bitdepth_max);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    unsafe {
        ipred_z1_hbd_avx2_impl(
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
            bitdepth_max,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
    bitdepth_max: u16,
) {
    unsafe {
        ipred_z3_hbd_avx2_impl(
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
            bitdepth_max,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2_hbd_avx2(
    dst: &mut [u16],
    stride: usize,
    tl: &[u16],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    bitdepth_max: u16,
) {
    unsafe {
        ipred_z2_hbd_avx2_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            bitdepth_max,
        )
    }
}
