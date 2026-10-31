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

//! 8x8 affine warp for SSE4.1: the AVX2 kernel's 8 x i32 lanes carried as
//! two 4-lane halves. Horizontal pass into a 15x8 intermediate, then the
//! vertical pass per output row.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Eight i32 lanes as (lanes 0..4, lanes 4..8).
#[derive(Clone, Copy)]
struct V8(__m128i, __m128i);

trait WarpPx: Copy {
    fn load8_i32(s: &[Self]) -> V8;
}

impl WarpPx for u8 {
    #[inline(always)]
    fn load8_i32(s: &[u8]) -> V8 {
        let s = &s[..8];
        unsafe {
            let v = _mm_loadl_epi64(s.as_ptr().cast());
            V8(
                _mm_cvtepu8_epi32(v),
                _mm_cvtepu8_epi32(_mm_srli_si128::<4>(v)),
            )
        }
    }
}

impl WarpPx for u16 {
    #[inline(always)]
    fn load8_i32(s: &[u16]) -> V8 {
        let s = &s[..8];
        unsafe {
            let v = _mm_loadu_si128(s.as_ptr().cast());
            V8(
                _mm_cvtepu16_epi32(v),
                _mm_cvtepu16_epi32(_mm_srli_si128::<8>(v)),
            )
        }
    }
}

/// Tap `tap` of the warp filters at `pos, pos + step, .., pos + 7 * step`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn coeff(pos: i32, step: i32, tap: usize) -> V8 {
    let f = |k: i32| {
        crate::tables::MC_WARP_FILTER[(192 + ((pos + step * k + 512) >> 10)) as usize][tap] as i32
    };
    V8(
        _mm_setr_epi32(f(0), f(1), f(2), f(3)),
        _mm_setr_epi32(f(4), f(5), f(6), f(7)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn madd(acc: V8, px: V8, c: V8) -> V8 {
    V8(
        _mm_add_epi32(acc.0, _mm_mullo_epi32(px.0, c.0)),
        _mm_add_epi32(acc.1, _mm_mullo_epi32(px.1, c.1)),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn round(v: V8, rnd: i32, shift: i32) -> V8 {
    let (r, s) = (_mm_set1_epi32(rnd), _mm_cvtsi32_si128(shift));
    V8(
        _mm_sra_epi32(_mm_add_epi32(v.0, r), s),
        _mm_sra_epi32(_mm_add_epi32(v.1, r), s),
    )
}

/// Horizontal pass: 15 rows of 8 i16 samples starting 3 rows and 3 columns
/// before `src_off`, each rounded by `(rnd, shift)`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn horz<P: WarpPx>(
    mid: &mut [i16; 15 * 8],
    src: &[P],
    src_stride: usize,
    src_off: usize,
    mut mx: i32,
    alpha: i32,
    beta: i32,
    rnd: i32,
    shift: i32,
) {
    let mut row_base = src_off.wrapping_sub(3 * src_stride + 3);
    for mid_row in mid.as_chunks_mut::<8>().0.iter_mut() {
        let z = _mm_setzero_si128();
        let mut acc = V8(z, z);
        for tap in 0..8 {
            acc = madd(
                acc,
                P::load8_i32(&src[row_base + tap..]),
                coeff(mx, alpha, tap),
            );
        }
        let v = round(acc, rnd, shift);
        unsafe { _mm_storeu_si128(mid_row.as_mut_ptr().cast(), _mm_packs_epi32(v.0, v.1)) };
        row_base += src_stride;
        mx += beta;
    }
}

/// Vertical pass for output row `y` (unrounded).
#[inline]
#[target_feature(enable = "sse4.1")]
fn vert(mid: &[i16; 15 * 8], y: usize, my: i32, gamma: i32) -> V8 {
    let z = _mm_setzero_si128();
    let mut acc = V8(z, z);
    for tap in 0..8 {
        let row = &mid[(y + tap) * 8..(y + tap) * 8 + 8];
        let v = unsafe { _mm_loadu_si128(row.as_ptr().cast()) };
        let px = V8(
            _mm_cvtepi16_epi32(v),
            _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v)),
        );
        acc = madd(acc, px, coeff(my, gamma, tap));
    }
    acc
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn warp_affine_8x8_8bpc_sse41(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    mut my: i32,
) {
    let [alpha, beta, gamma, delta] = abcd.map(i32::from);
    let mut mid = [0i16; 15 * 8];
    horz(&mut mid, src, src_stride, src_off, mx, alpha, beta, 4, 3);
    for (y, dst_row) in dst.chunks_mut(dst_stride).take(8).enumerate() {
        let v = round(vert(&mid, y, my, gamma), 1024, 11);
        let p16 = _mm_packus_epi32(v.0, v.1);
        unsafe { _mm_storel_epi64(dst_row[..8].as_mut_ptr().cast(), _mm_packus_epi16(p16, p16)) };
        my += delta;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn warp_affine_8x8t_8bpc_sse41(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    mut my: i32,
) {
    let [alpha, beta, gamma, delta] = abcd.map(i32::from);
    let mut mid = [0i16; 15 * 8];
    horz(&mut mid, src, src_stride, src_off, mx, alpha, beta, 4, 3);
    for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(8).enumerate() {
        let v = round(vert(&mid, y, my, gamma), 64, 7);
        unsafe { _mm_storeu_si128(tmp_row[..8].as_mut_ptr().cast(), _mm_packs_epi32(v.0, v.1)) };
        my += delta;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn warp_affine_8x8_hbd_sse41(
    dst: &mut [u16],
    dst_stride: usize,
    src: &[u16],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    mut my: i32,
    bitdepth: u8,
) {
    let ib = 14 - bitdepth as i32;
    let (h_shift, v_shift) = (7 - ib, 7 + ib);
    let max = _mm_set1_epi16(((1 << bitdepth) - 1) as i16);
    let [alpha, beta, gamma, delta] = abcd.map(i32::from);
    let mut mid = [0i16; 15 * 8];
    horz(
        &mut mid,
        src,
        src_stride,
        src_off,
        mx,
        alpha,
        beta,
        (1 << h_shift) >> 1,
        h_shift,
    );
    for (y, dst_row) in dst.chunks_mut(dst_stride).take(8).enumerate() {
        let v = round(vert(&mid, y, my, gamma), (1 << v_shift) >> 1, v_shift);
        let p = _mm_min_epu16(_mm_packus_epi32(v.0, v.1), max);
        unsafe { _mm_storeu_si128(dst_row[..8].as_mut_ptr().cast(), p) };
        my += delta;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn warp_affine_8x8t_hbd_sse41(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    mut my: i32,
    bitdepth: u8,
) {
    let h_shift = 7 - (14 - bitdepth as i32);
    let [alpha, beta, gamma, delta] = abcd.map(i32::from);
    let mut mid = [0i16; 15 * 8];
    horz(
        &mut mid,
        src,
        src_stride,
        src_off,
        mx,
        alpha,
        beta,
        (1 << h_shift) >> 1,
        h_shift,
    );
    let bias = _mm_set1_epi32(8192);
    for (y, tmp_row) in tmp.chunks_mut(tmp_stride).take(8).enumerate() {
        let v = round(vert(&mid, y, my, gamma), 64, 7);
        let p = _mm_packs_epi32(_mm_sub_epi32(v.0, bias), _mm_sub_epi32(v.1, bias));
        unsafe { _mm_storeu_si128(tmp_row[..8].as_mut_ptr().cast(), p) };
        my += delta;
    }
}
