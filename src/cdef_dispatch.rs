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

use crate::cdef::{CDEF_HAVE_BOTTOM, CDEF_HAVE_LEFT, CDEF_HAVE_RIGHT, CDEF_HAVE_TOP, constrain};
use crate::tables::CDEF_DIRECTIONS;
use std::sync::OnceLock;

pub(crate) type CdefFilterFn = unsafe fn(
    &mut [u8],
    usize,
    usize,
    &[i16],
    usize,
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    usize,
    usize,
    usize,
);

pub(crate) type CdefFilterHbdFn = unsafe fn(
    &mut [u16],
    usize,
    usize,
    &[i16],
    usize,
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    usize,
    usize,
    usize,
);

pub(crate) type CdefFilterShapeFn =
    unsafe fn(&mut [u8], usize, usize, &[i16], usize, usize, i32, i32, i32, i32, i32, usize);

pub(crate) type CdefFilterHbdShapeFn =
    unsafe fn(&mut [u16], usize, usize, &[i16], usize, usize, i32, i32, i32, i32, i32, usize);

pub(crate) type CdefDir8Fn = unsafe fn(&[u8], usize, &mut u32) -> i32;
pub(crate) type CdefDirHbdFn = unsafe fn(&[u16], usize, i32, &mut u32) -> i32;

pub(crate) type CdefPadding8Fn = unsafe fn(
    &mut [i16],
    usize,
    &[u8],
    usize,
    usize,
    &[[u8; 2]],
    &[u8],
    usize,
    &[u8],
    usize,
    usize,
    usize,
    usize,
    u8,
);

pub(crate) type CdefPaddingHbdFn = unsafe fn(
    &mut [i16],
    usize,
    &[u16],
    usize,
    usize,
    &[[u16; 2]],
    &[u16],
    usize,
    &[u16],
    usize,
    usize,
    usize,
    usize,
    u8,
);

#[inline(always)]
fn cdef_fill_i16(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    for row in tmp.chunks_exact_mut(stride).take(h) {
        row[..w].fill(i16::MIN);
    }
}

#[inline(always)]
fn copy_u8_to_i16_scalar(dst: &mut [i16], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as i16;
    }
}

#[inline(always)]
fn copy_u16_to_i16_scalar(dst: &mut [i16], src: &[u16]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as i16;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_padding_8bpc_scalar(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    left: &[[u8; 2]],
    top: &[u8],
    top_off: usize,
    bottom: &[u8],
    bottom_off: usize,
    bottom_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    let o = 2 * tmp_stride + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    if edges & CDEF_HAVE_TOP == 0 {
        let base = o.wrapping_sub(2).wrapping_sub(2 * tmp_stride);
        cdef_fill_i16(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        cdef_fill_i16(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        cdef_fill_i16(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        cdef_fill_i16(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut toff = top_off;
    for y in y_start..0 {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (toff as i32 + x_start) as usize;
        copy_u8_to_i16_scalar(&mut tmp[ti..ti + copy_w], &top[si..si + copy_w]);
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
        copy_u8_to_i16_scalar(&mut tmp[ti..ti + copy_w], &src[soff..soff + copy_w]);
        soff += src_stride;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut boff = bottom_off;
    for y in h as i32..y_end {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (boff as i32 + x_start) as usize;
        copy_u8_to_i16_scalar(&mut tmp[ti..ti + copy_w], &bottom[si..si + copy_w]);
        boff += bottom_stride;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_padding_hbd_scalar(
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
    let o = 2 * tmp_stride + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    if edges & CDEF_HAVE_TOP == 0 {
        let base = o.wrapping_sub(2).wrapping_sub(2 * tmp_stride);
        cdef_fill_i16(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        cdef_fill_i16(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        cdef_fill_i16(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        cdef_fill_i16(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut toff = top_off;
    for y in y_start..0 {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (toff as i32 + x_start) as usize;
        copy_u16_to_i16_scalar(&mut tmp[ti..ti + copy_w], &top[si..si + copy_w]);
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
        copy_u16_to_i16_scalar(&mut tmp[ti..ti + copy_w], &src[soff..soff + copy_w]);
        soff += src_stride;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut boff = bottom_off;
    for y in h as i32..y_end {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (boff as i32 + x_start) as usize;
        copy_u16_to_i16_scalar(&mut tmp[ti..ti + copy_w], &bottom[si..si + copy_w]);
        boff += bottom_stride;
    }
}

#[inline(always)]
fn cdef_min(a: i32, b: i32) -> i32 {
    // Padding sentinel is i16::MIN. Treating values as unsigned makes the
    // sentinel larger than every valid pixel, matching dav2d's pminuw path.
    if (a as u32) < (b as u32) { a } else { b }
}

#[inline(always)]
pub(crate) fn cdef_find_dir_from_i16_rows(rows: &[[i16; 8]; 8], var: &mut u32) -> i32 {
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        let mut row_sum = 0i32;
        for x in 0..8usize {
            let px = rows[y][x] as i32;
            row_sum += px;
            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
        partial_sum_hv[0][y] = row_sum;
    }

    cdef_find_dir_from_partials(&partial_sum_hv, &partial_sum_diag, &partial_sum_alt, var)
}

#[inline(always)]
pub(crate) fn cdef_find_dir_from_partials(
    partial_sum_hv: &[[i32; 8]; 2],
    partial_sum_diag: &[[i32; 15]; 2],
    partial_sum_alt: &[[i32; 11]; 4],
    var: &mut u32,
) -> i32 {
    let mut cost = [0u32; 8];
    for n in 0..8usize {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as u32;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    const DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7usize {
        let d = DIV_TABLE[n];
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n])
            as u32)
            * d;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n])
            as u32)
            * d;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7]) as u32 * 105;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7]) as u32 * 105;

    for n in 0..4usize {
        let ci = n * 2 + 1;
        for m in 0..5usize {
            cost[ci] += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as u32;
        }
        cost[ci] *= 105;
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            cost[ci] += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m])
                as u32)
                * d;
        }
    }

    let mut best_dir = 0i32;
    let mut best_cost = cost[0];
    for (n, &c) in cost.iter().enumerate().skip(1) {
        if c > best_cost {
            best_cost = c;
            best_dir = n as i32;
        }
    }

    *var = (best_cost - cost[(best_dir ^ 4) as usize]) >> 10;
    best_dir
}

#[inline]
pub(crate) fn cdef_find_dir_8bpc_scalar(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    let mut rows = [[0i16; 8]; 8];
    for y in 0..8usize {
        for x in 0..8usize {
            rows[y][x] = img[y * stride + x] as i16 - 128;
        }
    }
    cdef_find_dir_from_i16_rows(&rows, var)
}

#[inline]
pub(crate) fn cdef_find_dir_hbd_scalar(
    img: &[u16],
    stride: usize,
    bitdepth_min_8: i32,
    var: &mut u32,
) -> i32 {
    let shift = bitdepth_min_8 as u32;
    let mut rows = [[0i16; 8]; 8];
    for y in 0..8usize {
        for x in 0..8usize {
            rows[y][x] = (img[y * stride + x] >> shift) as i16 - 128;
        }
    }
    cdef_find_dir_from_i16_rows(&rows, var)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    let mut dp = dst_off;
    let mut tp = o;
    if pri_strength != 0 && sec_strength != 0 {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                let mut min_v = px;
                let mut max_v = px;
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let p0 = tmp[((tp + x) as isize + off1) as usize] as i32;
                    let p1 = tmp[((tp + x) as isize - off1) as usize] as i32;
                    sum += ptap * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptap * constrain(p1 - px, pri_strength, pri_shift);
                    ptap = (ptap & 3) | 2;
                    min_v = cdef_min(cdef_min(min_v, p0), p1);
                    max_v = p0.max(max_v).max(p1);
                    let off2 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off3 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp[((tp + x) as isize + off2) as usize] as i32;
                    let s1 = tmp[((tp + x) as isize - off2) as usize] as i32;
                    let s2 = tmp[((tp + x) as isize + off3) as usize] as i32;
                    let s3 = tmp[((tp + x) as isize - off3) as usize] as i32;
                    let st = 2 - k as i32;
                    sum += st * constrain(s0 - px, sec_strength, sec_shift);
                    sum += st * constrain(s1 - px, sec_strength, sec_shift);
                    sum += st * constrain(s2 - px, sec_strength, sec_shift);
                    sum += st * constrain(s3 - px, sec_strength, sec_shift);
                    min_v = cdef_min(cdef_min(cdef_min(cdef_min(min_v, s0), s1), s2), s3);
                    max_v = s0.max(max_v).max(s1).max(s2).max(s3);
                }
                let v = px + ((sum - (sum < 0) as i32 + 8) >> 4);
                dst[dp + x] = v.clamp(min_v, max_v) as u16;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    } else if pri_strength != 0 {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off = CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let p0 = tmp[((tp + x) as isize + off) as usize] as i32;
                    let p1 = tmp[((tp + x) as isize - off) as usize] as i32;
                    sum += ptap * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptap * constrain(p1 - px, pri_strength, pri_shift);
                    ptap = (ptap & 3) | 2;
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u16;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    } else {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off2 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp[((tp + x) as isize + off1) as usize] as i32;
                    let s1 = tmp[((tp + x) as isize - off1) as usize] as i32;
                    let s2 = tmp[((tp + x) as isize + off2) as usize] as i32;
                    let s3 = tmp[((tp + x) as isize - off2) as usize] as i32;
                    let st = 2 - k as i32;
                    sum += st * constrain(s0 - px, sec_strength, sec_shift);
                    sum += st * constrain(s1 - px, sec_strength, sec_shift);
                    sum += st * constrain(s2 - px, sec_strength, sec_shift);
                    sum += st * constrain(s3 - px, sec_strength, sec_shift);
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u16;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    let mut dp = dst_off;
    let mut tp = o;
    if pri_strength != 0 && sec_strength != 0 {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                let mut min_v = px;
                let mut max_v = px;
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let p0 = tmp[((tp + x) as isize + off1) as usize] as i32;
                    let p1 = tmp[((tp + x) as isize - off1) as usize] as i32;
                    sum += ptap * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptap * constrain(p1 - px, pri_strength, pri_shift);
                    ptap = (ptap & 3) | 2;
                    min_v = cdef_min(cdef_min(min_v, p0), p1);
                    max_v = p0.max(max_v).max(p1);
                    let off2 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off3 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp[((tp + x) as isize + off2) as usize] as i32;
                    let s1 = tmp[((tp + x) as isize - off2) as usize] as i32;
                    let s2 = tmp[((tp + x) as isize + off3) as usize] as i32;
                    let s3 = tmp[((tp + x) as isize - off3) as usize] as i32;
                    let st = 2 - k as i32;
                    sum += st * constrain(s0 - px, sec_strength, sec_shift);
                    sum += st * constrain(s1 - px, sec_strength, sec_shift);
                    sum += st * constrain(s2 - px, sec_strength, sec_shift);
                    sum += st * constrain(s3 - px, sec_strength, sec_shift);
                    min_v = cdef_min(cdef_min(cdef_min(cdef_min(min_v, s0), s1), s2), s3);
                    max_v = s0.max(max_v).max(s1).max(s2).max(s3);
                }
                let v = px + ((sum - (sum < 0) as i32 + 8) >> 4);
                dst[dp + x] = v.clamp(min_v, max_v) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    } else if pri_strength != 0 {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off = CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let p0 = tmp[((tp + x) as isize + off) as usize] as i32;
                    let p1 = tmp[((tp + x) as isize - off) as usize] as i32;
                    sum += ptap * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptap * constrain(p1 - px, pri_strength, pri_shift);
                    ptap = (ptap & 3) | 2;
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    } else {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off2 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp[((tp + x) as isize + off1) as usize] as i32;
                    let s1 = tmp[((tp + x) as isize - off1) as usize] as i32;
                    let s2 = tmp[((tp + x) as isize + off2) as usize] as i32;
                    let s3 = tmp[((tp + x) as isize - off2) as usize] as i32;
                    let st = 2 - k as i32;
                    sum += st * constrain(s0 - px, sec_strength, sec_shift);
                    sum += st * constrain(s1 - px, sec_strength, sec_shift);
                    sum += st * constrain(s2 - px, sec_strength, sec_shift);
                    sum += st * constrain(s3 - px, sec_strength, sec_shift);
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) unsafe fn cdef_filter_block_8x8_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        8,
        8,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_8x4_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        8,
        4,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_4x8_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        4,
        8,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_4x4_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_8bpc_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        4,
        4,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) unsafe fn cdef_filter_block_8x8_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_hbd_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        8,
        8,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_8x4_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_hbd_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        8,
        4,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_4x8_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_hbd_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        4,
        8,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) unsafe fn cdef_filter_block_4x4_hbd_scalar(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
) {
    cdef_filter_block_hbd_scalar(
        dst,
        dst_stride,
        dst_off,
        tmp,
        tmp_stride,
        o,
        pri_strength,
        sec_strength,
        pri_shift,
        sec_shift,
        pri_tap,
        dir,
        4,
        4,
    );
}

static CDEF_DIR_8BPC: OnceLock<CdefDir8Fn> = OnceLock::new();

#[inline]
fn resolve_cdef_dir_8bpc() -> CdefDir8Fn {
    *CDEF_DIR_8BPC.get_or_init(|| {
        let mut _f = cdef_find_dir_8bpc_scalar as CdefDir8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cdef_find_dir_8bpc_neon as CdefDir8Fn;
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cdef_find_dir_8bpc_sse41 as CdefDir8Fn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cdef_find_dir_8bpc_avx2 as CdefDir8Fn;
            }
        }
        _f
    })
}

static CDEF_DIR_HBD: OnceLock<CdefDirHbdFn> = OnceLock::new();

#[inline]
fn resolve_cdef_dir_hbd() -> CdefDirHbdFn {
    *CDEF_DIR_HBD.get_or_init(|| {
        let mut _f = cdef_find_dir_hbd_scalar as CdefDirHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cdef_find_dir_hbd_neon as CdefDirHbdFn;
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cdef_find_dir_hbd_sse41 as CdefDirHbdFn;
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cdef_find_dir_hbd_avx2 as CdefDirHbdFn;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn cdef_find_dir_8bpc(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    // SAFETY: architecture-specific entries are installed only after runtime
    // feature detection; the scalar default is always sound.
    unsafe { resolve_cdef_dir_8bpc()(img, stride, var) }
}

#[inline]
pub(crate) fn cdef_find_dir_hbd(
    img: &[u16],
    stride: usize,
    bitdepth_min_8: i32,
    var: &mut u32,
) -> i32 {
    // SAFETY: architecture-specific entries are installed only after runtime
    // feature detection; the scalar default is always sound.
    unsafe { resolve_cdef_dir_hbd()(img, stride, bitdepth_min_8, var) }
}

static CDEF_PADDING_8BPC: OnceLock<CdefPadding8Fn> = OnceLock::new();

#[inline]
fn resolve_cdef_padding_8bpc() -> CdefPadding8Fn {
    *CDEF_PADDING_8BPC.get_or_init(|| {
        let mut _f = cdef_padding_8bpc_scalar as CdefPadding8Fn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cdef_padding_8bpc_neon as CdefPadding8Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cdef_padding_8bpc_avx2 as CdefPadding8Fn;
            }
        }
        _f
    })
}

static CDEF_PADDING_HBD: OnceLock<CdefPaddingHbdFn> = OnceLock::new();

#[inline]
fn resolve_cdef_padding_hbd() -> CdefPaddingHbdFn {
    *CDEF_PADDING_HBD.get_or_init(|| {
        let mut _f = cdef_padding_hbd_scalar as CdefPaddingHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cdef_padding_hbd_neon as CdefPaddingHbdFn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cdef_padding_hbd_avx2 as CdefPaddingHbdFn;
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_padding_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    left: &[[u8; 2]],
    top: &[u8],
    top_off: usize,
    bottom: &[u8],
    bottom_off: usize,
    bottom_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    unsafe {
        resolve_cdef_padding_8bpc()(
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
            w,
            h,
            edges,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_padding_hbd(
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
    unsafe {
        resolve_cdef_padding_hbd()(
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
            w,
            h,
            edges,
        )
    }
}

static CDEF_FILTER: OnceLock<CdefFilterFn> = OnceLock::new();

#[inline]
fn resolve_cdef_filter() -> CdefFilterFn {
    *CDEF_FILTER.get_or_init(|| {
        let mut _f = cdef_filter_block_8bpc_scalar as CdefFilterFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cdef_filter_block_8bpc_neon as CdefFilterFn;
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cdef_filter_block_8bpc_sse41 as CdefFilterFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cdef_filter_block_8bpc_avx2 as CdefFilterFn;
            }
        }
        _f
    })
}

static CDEF_FILTER_HBD: OnceLock<CdefFilterHbdFn> = OnceLock::new();

#[inline]
fn resolve_cdef_filter_hbd() -> CdefFilterHbdFn {
    *CDEF_FILTER_HBD.get_or_init(|| {
        let mut _f = cdef_filter_block_hbd_scalar as CdefFilterHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            _f = crate::neon::cdef_filter_block_hbd_neon as CdefFilterHbdFn;
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::cdef_filter_block_hbd_sse41 as CdefFilterHbdFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::cdef_filter_block_hbd_avx2 as CdefFilterHbdFn;
            }
        }
        _f
    })
}

static CDEF_FILTER_SHAPES: OnceLock<[CdefFilterShapeFn; 4]> = OnceLock::new();

#[inline]
fn resolve_cdef_filter_shapes() -> &'static [CdefFilterShapeFn; 4] {
    CDEF_FILTER_SHAPES.get_or_init(|| {
        let mut _f = [
            cdef_filter_block_8x8_8bpc_scalar as CdefFilterShapeFn,
            cdef_filter_block_4x8_8bpc_scalar as CdefFilterShapeFn,
            cdef_filter_block_4x4_8bpc_scalar as CdefFilterShapeFn,
            cdef_filter_block_8x4_8bpc_scalar as CdefFilterShapeFn,
        ];
        #[cfg(target_arch = "aarch64")]
        {
            _f = [
                crate::neon::cdef_filter_block_8x8_8bpc_neon as CdefFilterShapeFn,
                crate::neon::cdef_filter_block_4x8_8bpc_neon as CdefFilterShapeFn,
                crate::neon::cdef_filter_block_4x4_8bpc_neon as CdefFilterShapeFn,
                crate::neon::cdef_filter_block_8x4_8bpc_neon as CdefFilterShapeFn,
            ];
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = [
                    crate::sse::cdef_filter_block_8x8_8bpc_sse41 as CdefFilterShapeFn,
                    crate::sse::cdef_filter_block_4x8_8bpc_sse41 as CdefFilterShapeFn,
                    crate::sse::cdef_filter_block_4x4_8bpc_sse41 as CdefFilterShapeFn,
                    cdef_filter_block_8x4_8bpc_scalar as CdefFilterShapeFn,
                ];
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = [
                    crate::avx::cdef_filter_block_8x8_8bpc_avx2 as CdefFilterShapeFn,
                    crate::avx::cdef_filter_block_4x8_8bpc_avx2 as CdefFilterShapeFn,
                    crate::avx::cdef_filter_block_4x4_8bpc_avx2 as CdefFilterShapeFn,
                    crate::avx::cdef_filter_block_8x4_8bpc_avx2 as CdefFilterShapeFn,
                ];
            }
        }
        _f
    })
}

static CDEF_FILTER_HBD_SHAPES: OnceLock<[CdefFilterHbdShapeFn; 4]> = OnceLock::new();

#[inline]
fn resolve_cdef_filter_hbd_shapes() -> &'static [CdefFilterHbdShapeFn; 4] {
    CDEF_FILTER_HBD_SHAPES.get_or_init(|| {
        let mut _f = [
            cdef_filter_block_8x8_hbd_scalar as CdefFilterHbdShapeFn,
            cdef_filter_block_4x8_hbd_scalar as CdefFilterHbdShapeFn,
            cdef_filter_block_4x4_hbd_scalar as CdefFilterHbdShapeFn,
            cdef_filter_block_8x4_hbd_scalar as CdefFilterHbdShapeFn,
        ];
        #[cfg(target_arch = "aarch64")]
        {
            _f = [
                crate::neon::cdef_filter_block_8x8_hbd_neon as CdefFilterHbdShapeFn,
                crate::neon::cdef_filter_block_4x8_hbd_neon as CdefFilterHbdShapeFn,
                crate::neon::cdef_filter_block_4x4_hbd_neon as CdefFilterHbdShapeFn,
                crate::neon::cdef_filter_block_8x4_hbd_neon as CdefFilterHbdShapeFn,
            ];
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = [
                    crate::sse::cdef_filter_block_8x8_hbd_sse41 as CdefFilterHbdShapeFn,
                    crate::sse::cdef_filter_block_4x8_hbd_sse41 as CdefFilterHbdShapeFn,
                    crate::sse::cdef_filter_block_4x4_hbd_sse41 as CdefFilterHbdShapeFn,
                    cdef_filter_block_8x4_hbd_scalar as CdefFilterHbdShapeFn,
                ];
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = [
                    crate::avx::cdef_filter_block_8x8_hbd_avx2 as CdefFilterHbdShapeFn,
                    crate::avx::cdef_filter_block_4x8_hbd_avx2 as CdefFilterHbdShapeFn,
                    crate::avx::cdef_filter_block_4x4_hbd_avx2 as CdefFilterHbdShapeFn,
                    crate::avx::cdef_filter_block_8x4_hbd_avx2 as CdefFilterHbdShapeFn,
                ];
            }
        }
        _f
    })
}

#[inline(always)]
fn cdef_shape_index(w: usize, h: usize) -> Option<usize> {
    match (w, h) {
        (8, 8) => Some(0),
        (4, 8) => Some(1),
        (4, 4) => Some(2),
        (8, 4) => Some(3),
        _ => None,
    }
}

/// Dispatched 8-bit CDEF filter apply. See `CdefFilterFn` for the argument layout.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    // Match dav1d's fb[shape] model for full and cropped CDEF blocks:
    // 8x8, 8x4, 4x8, and 4x4 are dispatched to fixed-shape kernels.
    // The generic `(w, h)` entry remains only as a safety fallback.
    if let Some(shape) = cdef_shape_index(w, h) {
        // SAFETY: architecture-specific entries are installed only after
        // runtime feature detection; scalar defaults are always sound.
        unsafe {
            resolve_cdef_filter_shapes()[shape](
                dst,
                dst_stride,
                dst_off,
                tmp,
                tmp_stride,
                o,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
            )
        };
        return;
    }

    // SAFETY: resolve only returns the SSE/NEON kernel when the feature was
    // detected; the scalar default is always sound.
    unsafe {
        resolve_cdef_filter()(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        )
    };
}

/// Dispatched high-bit-depth CDEF filter apply. The same function is used for
/// 10-bit and 12-bit because both are stored as native-endian `u16` samples.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_hbd(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    // Match dav1d's fb[shape] model for HBD as well.
    if let Some(shape) = cdef_shape_index(w, h) {
        // SAFETY: architecture-specific entries are installed only after
        // runtime feature detection; scalar defaults are always sound.
        unsafe {
            resolve_cdef_filter_hbd_shapes()[shape](
                dst,
                dst_stride,
                dst_off,
                tmp,
                tmp_stride,
                o,
                pri_strength,
                sec_strength,
                pri_shift,
                sec_shift,
                pri_tap,
                dir,
            )
        };
        return;
    }

    // SAFETY: resolve only returns architecture-specific kernels after runtime
    // feature detection; the scalar default is always sound.
    unsafe {
        resolve_cdef_filter_hbd()(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        )
    };
}
