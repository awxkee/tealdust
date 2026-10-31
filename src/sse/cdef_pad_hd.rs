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

//! High-bitdepth CDEF block padding for SSE4.1 (128-bit port of the AVX2 kernel).

use crate::cdef::{CDEF_HAVE_BOTTOM, CDEF_HAVE_LEFT, CDEF_HAVE_RIGHT, CDEF_HAVE_TOP};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn cdef_fill_i16_sse41(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    let sentinel = _mm_set1_epi16(i16::MIN);
    // `chunks_mut`, not `chunks_exact_mut`: a right-edge fill starts mid-row,
    // so its last row is a short tail chunk that must still be filled.
    for row in tmp.chunks_mut(stride).take(h) {
        let row = &mut row[..w];
        if w >= 8 {
            unsafe { _mm_storeu_si128(row.as_mut_ptr().cast(), sentinel) };
            row[8..].fill(i16::MIN);
        } else {
            row.fill(i16::MIN);
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn copy_u16_to_i16_sse41(dst: &mut [i16], src: &[u16]) {
    debug_assert_eq!(dst.len(), src.len());
    let n = src.len();
    if n >= 8 {
        unsafe {
            let v = _mm_loadu_si128(src.as_ptr().cast());
            _mm_storeu_si128(dst.as_mut_ptr().cast(), v);
            if n > 8 {
                let s = src.as_ptr().add(n - 8);
                let d = dst.as_mut_ptr().add(n - 8);
                let v = _mm_loadu_si128(s.cast());
                _mm_storeu_si128(d.cast(), v);
            }
        }
    } else {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = s as i16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn cdef_padding_hbd_sse41_full<const W: usize, const H: usize>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    src_off: usize,
    left: &[[u16; 2]],
    top: &[u16],
    top_off: usize,
    bottom: &[u16],
    bottom_off: usize,
    bottom_stride: usize,
) {
    debug_assert!(W == 4 || W == 8);
    debug_assert!(H == 4 || H == 8);
    debug_assert!(top_off >= 2);
    debug_assert!(bottom_off >= 2);

    let o = 2 * tmp_stride + 2;
    let top_src = top_off - 2;
    let top_dst = o - 2 - 2 * tmp_stride;
    copy_u16_to_i16_sse41(
        &mut tmp[top_dst..top_dst + W + 4],
        &top[top_src..top_src + W + 4],
    );
    copy_u16_to_i16_sse41(
        &mut tmp[top_dst + tmp_stride..top_dst + tmp_stride + W + 4],
        &top[top_src + src_stride..top_src + src_stride + W + 4],
    );

    let mut soff = src_off;
    for y in 0..H {
        let ti = o + y * tmp_stride;
        tmp[ti - 2] = left[y][0] as i16;
        tmp[ti - 1] = left[y][1] as i16;
        copy_u16_to_i16_sse41(&mut tmp[ti..ti + W + 2], &src[soff..soff + W + 2]);
        soff += src_stride;
    }

    let bottom_src = bottom_off - 2;
    let bottom_dst = o - 2 + H * tmp_stride;
    copy_u16_to_i16_sse41(
        &mut tmp[bottom_dst..bottom_dst + W + 4],
        &bottom[bottom_src..bottom_src + W + 4],
    );
    copy_u16_to_i16_sse41(
        &mut tmp[bottom_dst + tmp_stride..bottom_dst + tmp_stride + W + 4],
        &bottom[bottom_src + bottom_stride..bottom_src + bottom_stride + W + 4],
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cdef_padding_hbd_sse41(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u16],
    src_stride: usize,
    src_off: usize,
    left: &[[u16; 2]],
    top: &[u16],
    top_off: usize,
    bottom: &[u16],
    bottom_off: usize,
    bottom_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    const CDEF_HAVE_ALL: u8 = CDEF_HAVE_LEFT | CDEF_HAVE_RIGHT | CDEF_HAVE_TOP | CDEF_HAVE_BOTTOM;
    if edges == CDEF_HAVE_ALL {
        match (w, h) {
            (8, 8) => {
                cdef_padding_hbd_sse41_full::<8, 8>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            (8, 4) => {
                cdef_padding_hbd_sse41_full::<8, 4>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            (4, 8) => {
                cdef_padding_hbd_sse41_full::<4, 8>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            (4, 4) => {
                cdef_padding_hbd_sse41_full::<4, 4>(
                    tmp,
                    tmp_stride,
                    src,
                    src_stride,
                    src_off,
                    left,
                    top,
                    top_off,
                    bottom,
                    bottom_off,
                    bottom_stride,
                );
                return;
            }
            _ => {}
        }
    }

    let o = 2 * tmp_stride + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    if edges & CDEF_HAVE_TOP == 0 {
        let base = o.wrapping_sub(2).wrapping_sub(2 * tmp_stride);
        cdef_fill_i16_sse41(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        cdef_fill_i16_sse41(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        cdef_fill_i16_sse41(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        cdef_fill_i16_sse41(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut toff = top_off;
    for y in y_start..0 {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (toff as i32 + x_start) as usize;
        copy_u16_to_i16_sse41(&mut tmp[ti..ti + copy_w], &top[si..si + copy_w]);
        toff += src_stride;
    }

    for y in 0..h as i32 {
        let ti = (o as i32 + y * tmp_stride as i32 - 2) as usize;
        for x in x_start..0 {
            tmp[ti + (x + 2) as usize] = left[y as usize][(x + 2) as usize] as i16;
        }
    }

    let copy_w = x_end as usize;
    let mut soff = src_off;
    for y in 0..h as i32 {
        let ti = (o as i32 + y * tmp_stride as i32) as usize;
        copy_u16_to_i16_sse41(&mut tmp[ti..ti + copy_w], &src[soff..soff + copy_w]);
        soff += src_stride;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut boff = bottom_off;
    for y in h as i32..y_end {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (boff as i32 + x_start) as usize;
        copy_u16_to_i16_sse41(&mut tmp[ti..ti + copy_w], &bottom[si..si + copy_w]);
        boff += bottom_stride;
    }
}
