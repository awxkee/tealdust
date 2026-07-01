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

use crate::cdef::{CDEF_HAVE_BOTTOM, CDEF_HAVE_LEFT, CDEF_HAVE_RIGHT, CDEF_HAVE_TOP};
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn cdef_fill_i16_avx2(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    let sentinel = _mm_set1_epi16(i16::MIN);
    for row in tmp.chunks_exact_mut(stride).take(h) {
        if w >= 8 {
            unsafe { _mm_storeu_si128(row.as_mut_ptr().cast(), sentinel) };
            for v in &mut row[8..w] {
                *v = i16::MIN;
            }
        } else {
            row[..w].fill(i16::MIN);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn copy_u16_to_i16_avx2(dst: &mut [i16], src: &[u16]) {
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
#[target_feature(enable = "avx2")]
fn cdef_padding_hbd_avx2_full<const W: usize, const H: usize>(
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
    copy_u16_to_i16_avx2(
        &mut tmp[top_dst..top_dst + W + 4],
        &top[top_src..top_src + W + 4],
    );
    copy_u16_to_i16_avx2(
        &mut tmp[top_dst + tmp_stride..top_dst + tmp_stride + W + 4],
        &top[top_src + src_stride..top_src + src_stride + W + 4],
    );

    let mut soff = src_off;
    for y in 0..H {
        let ti = o + y * tmp_stride;
        tmp[ti - 2] = left[y][0] as i16;
        tmp[ti - 1] = left[y][1] as i16;
        copy_u16_to_i16_avx2(&mut tmp[ti..ti + W + 2], &src[soff..soff + W + 2]);
        soff += src_stride;
    }

    let bottom_src = bottom_off - 2;
    let bottom_dst = o - 2 + H * tmp_stride;
    copy_u16_to_i16_avx2(
        &mut tmp[bottom_dst..bottom_dst + W + 4],
        &bottom[bottom_src..bottom_src + W + 4],
    );
    copy_u16_to_i16_avx2(
        &mut tmp[bottom_dst + tmp_stride..bottom_dst + tmp_stride + W + 4],
        &bottom[bottom_src + bottom_stride..bottom_src + bottom_stride + W + 4],
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_padding_hbd_avx2(
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
                cdef_padding_hbd_avx2_full::<8, 8>(
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
                cdef_padding_hbd_avx2_full::<8, 4>(
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
                cdef_padding_hbd_avx2_full::<4, 8>(
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
                cdef_padding_hbd_avx2_full::<4, 4>(
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
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        cdef_fill_i16_avx2(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut toff = top_off;
    for y in y_start..0 {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (toff as i32 + x_start) as usize;
        copy_u16_to_i16_avx2(&mut tmp[ti..ti + copy_w], &top[si..si + copy_w]);
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
        copy_u16_to_i16_avx2(&mut tmp[ti..ti + copy_w], &src[soff..soff + copy_w]);
        soff += src_stride;
    }

    let copy_w = (x_end - x_start) as usize;
    let mut boff = bottom_off;
    for y in h as i32..y_end {
        let ti = (o as i32 + x_start + y * tmp_stride as i32) as usize;
        let si = (boff as i32 + x_start) as usize;
        copy_u16_to_i16_avx2(&mut tmp[ti..ti + copy_w], &bottom[si..si + copy_w]);
        boff += bottom_stride;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16x16_2rows(tmp: &[i16], p0: isize, p1: isize, off: isize) -> __m256i {
    unsafe {
        let lo = _mm_loadu_si128(tmp.as_ptr().offset(p0 + off).cast());
        let hi = _mm_loadu_si128(tmp.as_ptr().offset(p1 + off).cast());
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i16xw_2rows<const W: usize>(tmp: &[i16], p0: isize, p1: isize, off: isize) -> __m256i {
    debug_assert!(W == 4 || W == 8);
    unsafe {
        let lo = if W == 8 {
            _mm_loadu_si128(tmp.as_ptr().offset(p0 + off).cast())
        } else {
            _mm_loadl_epi64(tmp.as_ptr().offset(p0 + off).cast())
        };
        let hi = if W == 8 {
            _mm_loadu_si128(tmp.as_ptr().offset(p1 + off).cast())
        } else {
            _mm_loadl_epi64(tmp.as_ptr().offset(p1 + off).cast())
        };
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn cdef_min_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epu16(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
fn constrain_i16(diff: __m256i, threshold: __m256i, shc: __m128i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let adiff = _mm256_abs_epi16(diff);
    let t = _mm256_max_epi16(
        zero,
        _mm256_sub_epi16(threshold, _mm256_srl_epi16(adiff, shc)),
    );
    let m = _mm256_min_epu16(adiff, t);
    _mm256_blendv_epi8(m, _mm256_sub_epi16(zero, m), _mm256_cmpgt_epi16(zero, diff))
}

#[inline]
#[target_feature(enable = "avx2")]
fn add_tap_i16(v: __m256i, tap: i32) -> __m256i {
    match tap {
        1 => v,
        2 => _mm256_add_epi16(v, v),
        3 => _mm256_add_epi16(_mm256_add_epi16(v, v), v),
        4 => _mm256_slli_epi16::<2>(v),
        _ => _mm256_mullo_epi16(v, _mm256_set1_epi16(tap as i16)),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_i16(sum: __m256i, v: __m256i, tap: i32) -> __m256i {
    _mm256_add_epi16(sum, add_tap_i16(v, tap))
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16x8_2rows(dst: &mut [u16], p0: usize, p1: usize, v: __m256i) {
    unsafe {
        _mm_storeu_si128(dst.as_mut_ptr().add(p0).cast(), _mm256_castsi256_si128(v));
        _mm_storeu_si128(
            dst.as_mut_ptr().add(p1).cast(),
            _mm256_extracti128_si256::<1>(v),
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_u16xw_2rows<const W: usize>(dst: &mut [u16], p0: usize, p1: usize, v: __m256i) {
    debug_assert!(W == 4 || W == 8);
    unsafe {
        if W == 8 {
            _mm_storeu_si128(dst.as_mut_ptr().add(p0).cast(), _mm256_castsi256_si128(v));
            _mm_storeu_si128(
                dst.as_mut_ptr().add(p1).cast(),
                _mm256_extracti128_si256::<1>(v),
            );
        } else {
            _mm_storel_epi64(dst.as_mut_ptr().add(p0).cast(), _mm256_castsi256_si128(v));
            _mm_storel_epi64(
                dst.as_mut_ptr().add(p1).cast(),
                _mm256_extracti128_si256::<1>(v),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
fn cdef_filter_block_hbd_avx2_shape<
    const W: usize,
    const H: usize,
    const HAS_PRI: bool,
    const HAS_SEC: bool,
>(
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
    debug_assert!(W == 4 || W == 8);
    debug_assert!(H == 4 || H == 8);
    let clip = HAS_PRI && HAS_SEC;
    let pri_s = _mm256_set1_epi16(pri_strength as i16);
    let sec_s = _mm256_set1_epi16(sec_strength as i16);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm256_setzero_si256();
    let eight = _mm256_set1_epi16(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let mut y = 0usize;

    while y < H {
        let t0 = (o + y * tmp_stride) as isize;
        let t1 = t0 + tmp_stride as isize;
        let load = |off: isize| load_i16xw_2rows::<W>(tmp, t0, t1, off);
        let px = load(0);
        let mut sum = zero;
        let mut min_v = px;
        let mut max_v = px;

        if HAS_PRI {
            let mut ptap = pri_tap;
            for k in 0..2 {
                let off = dirs[dir + 2][k] as isize;
                let p0 = load(off);
                let p1 = load(-off);
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(p0, px), pri_s, pri_shc),
                    ptap,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(p1, px), pri_s, pri_shc),
                    ptap,
                );
                ptap = (ptap & 3) | 2;
                if clip {
                    min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                    max_v = _mm256_max_epi16(max_v, _mm256_max_epi16(p0, p1));
                }
                if HAS_SEC {
                    let off2 = dirs[dir + 4][k] as isize;
                    let off3 = dirs[dir][k] as isize;
                    let s0 = load(off2);
                    let s1 = load(-off2);
                    let s2 = load(off3);
                    let s3 = load(-off3);
                    let st = 2 - k as i32;
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                        st,
                    );
                    min_v = cdef_min_i16(
                        min_v,
                        cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                    );
                    max_v = _mm256_max_epi16(
                        max_v,
                        _mm256_max_epi16(_mm256_max_epi16(s0, s1), _mm256_max_epi16(s2, s3)),
                    );
                }
            }
        } else if HAS_SEC {
            for k in 0..2 {
                let off1 = dirs[dir + 4][k] as isize;
                let off2 = dirs[dir][k] as isize;
                let s0 = load(off1);
                let s1 = load(-off1);
                let s2 = load(off2);
                let s3 = load(-off2);
                let st = 2 - k as i32;
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                    st,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                    st,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                    st,
                );
                sum = madd_i16(
                    sum,
                    constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                    st,
                );
            }
        }

        let mask = _mm256_cmpgt_epi16(zero, sum);
        let delta = _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(sum, mask), eight));
        let mut res = _mm256_add_epi16(px, delta);
        if clip {
            res = _mm256_min_epi16(_mm256_max_epi16(res, min_v), max_v);
        }
        let d0 = dst_off + y * dst_stride;
        let d1 = d0 + dst_stride;
        store_u16xw_2rows::<W>(dst, d0, d1, res);
        y += 2;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
fn cdef_filter_block_hbd_avx2_shape_dispatch<const W: usize, const H: usize>(
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
    match (pri_strength != 0, sec_strength != 0) {
        (true, true) => cdef_filter_block_hbd_avx2_shape::<W, H, true, true>(
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
        ),
        (true, false) => cdef_filter_block_hbd_avx2_shape::<W, H, true, false>(
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
        ),
        (false, true) => cdef_filter_block_hbd_avx2_shape::<W, H, false, true>(
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
        ),
        (false, false) => (),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_8x8_hbd_avx2(
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
    cdef_filter_block_hbd_avx2_shape_dispatch::<8, 8>(
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
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_8x4_hbd_avx2(
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
    cdef_filter_block_hbd_avx2_shape_dispatch::<8, 4>(
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
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_4x8_hbd_avx2(
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
    cdef_filter_block_hbd_avx2_shape_dispatch::<4, 8>(
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
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_4x4_hbd_avx2(
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
    cdef_filter_block_hbd_avx2_shape_dispatch::<4, 4>(
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
    );
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_hbd_avx2(
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

    match (w, h) {
        (8, 8) => {
            cdef_filter_block_8x8_hbd_avx2(
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
            );
            return;
        }
        (8, 4) => {
            cdef_filter_block_hbd_avx2_shape_dispatch::<8, 4>(
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
            );
            return;
        }
        (4, 8) => {
            cdef_filter_block_4x8_hbd_avx2(
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
            );
            return;
        }
        (4, 4) => {
            cdef_filter_block_4x4_hbd_avx2(
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
            );
            return;
        }
        _ => {}
    }

    if w < 8 {
        crate::cdef_dispatch::cdef_filter_block_hbd_scalar(
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
        );
        return;
    }

    let has_pri = pri_strength != 0;
    let has_sec = sec_strength != 0;
    let clip = has_pri && has_sec;
    let pri_s = _mm256_set1_epi16(pri_strength as i16);
    let sec_s = _mm256_set1_epi16(sec_strength as i16);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm256_setzero_si256();
    let eight = _mm256_set1_epi16(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 8;
    let mut y = 0usize;

    while y < h {
        let paired = y + 1 < h;
        for g in 0..groups {
            let bx = g * 8;
            let t0 = (o + y * tmp_stride + bx) as isize;
            let t1 = if paired { t0 + tmp_stride as isize } else { t0 };
            let load = |off: isize| load_i16x16_2rows(tmp, t0, t1, off);
            let px = load(0);
            let mut sum = zero;
            let mut min_v = px;
            let mut max_v = px;

            if has_pri {
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off = dirs[dir + 2][k] as isize;
                    let p0 = load(off);
                    let p1 = load(-off);
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(p0, px), pri_s, pri_shc),
                        ptap,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(p1, px), pri_s, pri_shc),
                        ptap,
                    );
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                        max_v = _mm256_max_epi16(max_v, _mm256_max_epi16(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = 2 - k as i32;
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                            st,
                        );
                        min_v = cdef_min_i16(
                            min_v,
                            cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                        );
                        max_v = _mm256_max_epi16(
                            max_v,
                            _mm256_max_epi16(_mm256_max_epi16(s0, s1), _mm256_max_epi16(s2, s3)),
                        );
                    }
                }
            } else {
                for k in 0..2 {
                    let off1 = dirs[dir + 4][k] as isize;
                    let off2 = dirs[dir][k] as isize;
                    let s0 = load(off1);
                    let s1 = load(-off1);
                    let s2 = load(off2);
                    let s3 = load(-off2);
                    let st = 2 - k as i32;
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s0, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s1, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s2, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm256_sub_epi16(s3, px), sec_s, sec_shc),
                        st,
                    );
                }
            }

            let mask = _mm256_cmpgt_epi16(zero, sum);
            let delta =
                _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_add_epi16(sum, mask), eight));
            let mut res = _mm256_add_epi16(px, delta);
            if clip {
                res = _mm256_min_epi16(_mm256_max_epi16(res, min_v), max_v);
            }
            let d0 = dst_off + y * dst_stride + bx;
            let d1 = if paired { d0 + dst_stride } else { d0 };
            store_u16x8_2rows(dst, d0, d1, res);
        }
        y += if paired { 2 } else { 1 };
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_dir_hbd_pair(img: &[u16], stride: usize, y: usize, sh: __m128i) -> __m256i {
    let lo = unsafe { _mm_loadu_si128(img.as_ptr().add(y * stride).cast()) };
    let hi = unsafe { _mm_loadu_si128(img.as_ptr().add((y + 4) * stride).cast()) };
    let raw = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
    _mm256_sub_epi16(_mm256_srl_epi16(raw, sh), _mm256_set1_epi16(128))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_find_dir_hbd_avx2(
    img: &[u16],
    stride: usize,
    bitdepth_min_8: i32,
    var: &mut u32,
) -> i32 {
    let z = _mm_setzero_si128();
    let mut rows = [z; 8];
    let sh = _mm_cvtsi32_si128(bitdepth_min_8);
    let r04 = load_dir_hbd_pair(img, stride, 0, sh);
    let r15 = load_dir_hbd_pair(img, stride, 1, sh);
    let r26 = load_dir_hbd_pair(img, stride, 2, sh);
    let r37 = load_dir_hbd_pair(img, stride, 3, sh);
    rows[0] = _mm256_castsi256_si128(r04);
    rows[4] = _mm256_extracti128_si256::<1>(r04);
    rows[1] = _mm256_castsi256_si128(r15);
    rows[5] = _mm256_extracti128_si256::<1>(r15);
    rows[2] = _mm256_castsi256_si128(r26);
    rows[6] = _mm256_extracti128_si256::<1>(r26);
    rows[3] = _mm256_castsi256_si128(r37);
    rows[7] = _mm256_extracti128_si256::<1>(r37);
    super::cdef::cdef_find_dir_from_rows_avx2(&rows, var)
}

#[cfg(test)]
mod cdef_hbd_simd_conformance {

    use crate::cdef_dispatch::cdef_filter_block_hbd_scalar;

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn r(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    const TMP_STRIDE: usize = 12; // baked into CDEF_DIRECTIONS offsets
    const O: usize = 2 * TMP_STRIDE + 2; // block at row 2, col 2 of the bordered tmp

    fn run_one(
        rng: &mut R,
        w: usize,
        h: usize,
        bitdepth_max: i32,
        sentinels: bool,
        pri_strength: i32,
        sec_strength: i32,
        pri_shift: i32,
        sec_shift: i32,
        pri_tap: i32,
        dir: usize,
    ) {
        let have_sse = std::is_x86_feature_detected!("sse4.1");
        let have_avx = std::is_x86_feature_detected!("avx2");
        if !have_sse && !have_avx {
            return;
        }
        let tlen = (h + 4) * TMP_STRIDE + 32;
        let mut tmp = vec![0i16; tlen];
        for t in tmp.iter_mut() {
            *t = rng.r(0, bitdepth_max) as i16;
        }
        if sentinels {
            // sprinkle off-frame sentinels (i16::MIN) the way the real pad does
            for _ in 0..(tlen / 8) {
                let i = rng.r(0, tlen as i32 - 1) as usize;
                tmp[i] = i16::MIN;
            }
            // keep the block's own pixels valid so px is never a sentinel
            for y in 0..h {
                for x in 0..w {
                    tmp[O + y * TMP_STRIDE + x] = rng.r(0, bitdepth_max) as i16;
                }
            }
        }
        let dst_stride = w;
        let mut ds = vec![0u16; h * dst_stride];
        cdef_filter_block_hbd_scalar(
            &mut ds,
            dst_stride,
            0,
            &tmp,
            TMP_STRIDE,
            O,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        );
        if have_sse {
            let mut d = vec![0u16; h * dst_stride];
            unsafe {
                crate::sse::cdef_filter_block_hbd_sse41(
                    &mut d,
                    dst_stride,
                    0,
                    &tmp,
                    TMP_STRIDE,
                    O,
                    pri_strength,
                    sec_strength,
                    pri_shift,
                    sec_shift,
                    pri_tap,
                    dir,
                    w,
                    h,
                );
            }
            assert_eq!(
                d, ds,
                "SSE w={w} h={h} bd={bitdepth_max} sent={sentinels} pri={pri_strength} sec={sec_strength} dir={dir} ptap={pri_tap}"
            );
        }
        #[cfg(feature = "avx")]
        if have_avx {
            let mut d = vec![0u16; h * dst_stride];
            unsafe {
                crate::avx::cdef_filter_block_hbd_avx2(
                    &mut d,
                    dst_stride,
                    0,
                    &tmp,
                    TMP_STRIDE,
                    O,
                    pri_strength,
                    sec_strength,
                    pri_shift,
                    sec_shift,
                    pri_tap,
                    dir,
                    w,
                    h,
                );
            }
            assert_eq!(
                d, ds,
                "AVX2 w={w} h={h} bd={bitdepth_max} sent={sentinels} pri={pri_strength} sec={sec_strength} dir={dir} ptap={pri_tap}"
            );
        }
    }

    #[test]
    fn cdef_hbd_matches_scalar() {
        let mut rng = R(0xc0ffee_cdef_2026);
        for &(w, h) in &[(4usize, 4usize), (8, 4), (8, 8), (4, 8)] {
            for &bd in &[1023i32, 4095] {
                // 10-bit and 12-bit
                for &sent in &[false, true] {
                    for dir in 0..8usize {
                        for _ in 0..60 {
                            // exercise all three active branches
                            let (pri, sec) = match rng.r(0, 2) {
                                0 => (rng.r(1, 15), rng.r(1, 4)),
                                1 => (rng.r(1, 15), 0),
                                _ => (0, rng.r(1, 4)),
                            };
                            let pri_tap = if pri != 0 {
                                [3, 4][rng.r(0, 1) as usize]
                            } else {
                                0
                            };
                            let pri_shift = rng.r(0, 6);
                            let sec_shift = rng.r(0, 6);
                            run_one(
                                &mut rng, w, h, bd, sent, pri, sec, pri_shift, sec_shift, pri_tap,
                                dir,
                            );
                        }
                    }
                }
            }
        }
    }
}
