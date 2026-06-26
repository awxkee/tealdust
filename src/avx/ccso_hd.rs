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

#[inline(always)]
fn ccso_tail_hbd(
    dst: &mut [u8],
    dst_base: usize,
    tmp: &[u16],
    row: usize,
    x0: usize,
    x1: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
    bo_only: bool,
) {
    for x in x0..x1 {
        let ti = row + x;
        let c = tmp[ti] as i32;
        let band = (c as u32 >> shift) as u8;
        if bo_only {
            dst[dst_base + x] = band;
        } else {
            let cls0 = crate::ccso::ccso_score(
                tmp[(ti as isize + luma_offset) as usize] as i32 - c,
                quant_step,
                edge_clf,
            );
            let cls1 = crate::ccso::ccso_score(
                tmp[(ti as isize - luma_offset) as usize] as i32 - c,
                quant_step,
                edge_clf,
            );
            dst[dst_base + x] = ((cls0 << 5) | (cls1 << 3)) as u8 | band;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ccso_classify_epi16(diff: __m128i, q: __m128i, nq: __m128i, edge_clf: u32) -> __m128i {
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi16(1);
    let two = _mm_set1_epi16(2);
    let gt = if edge_clf == 0 {
        _mm_cmpgt_epi16(diff, q)
    } else {
        zero
    };
    let lt = _mm_cmpgt_epi16(nq, diff);
    let cls = _mm_blendv_epi8(one, two, gt);
    _mm_blendv_epi8(cls, zero, lt)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ccso_make_idx_8x16(
    c: __m128i,
    p0: __m128i,
    p1: __m128i,
    shiftv: __m128i,
    q: __m128i,
    nq: __m128i,
    edge_clf: u32,
) -> __m128i {
    let band = _mm_srl_epi16(c, shiftv);
    let cls0 = ccso_classify_epi16(_mm_sub_epi16(p0, c), q, nq, edge_clf);
    let cls1 = ccso_classify_epi16(_mm_sub_epi16(p1, c), q, nq, edge_clf);
    _mm_or_si128(
        band,
        _mm_or_si128(_mm_slli_epi16(cls0, 5), _mm_slli_epi16(cls1, 3)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8_step2(ptr: *const u16) -> __m128i {
    let mask = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13, -1, -1,
        -1, -1, -1, -1, -1, -1,
    );
    let v = unsafe { _mm256_loadu_si256(ptr as *const __m256i) };
    let v = _mm256_shuffle_epi8(v, mask);
    _mm_unpacklo_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_step2(ptr: *const u16) -> __m128i {
    let mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let v = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    _mm_shuffle_epi8(v, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8_subsampled(tmp: &[u16], base: usize, ss_hor: usize) -> __m128i {
    if ss_hor == 0 {
        unsafe { _mm_loadu_si128(tmp.as_ptr().add(base) as *const __m128i) }
    } else {
        load_u16x8_step2(unsafe { tmp.as_ptr().add(base) })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_subsampled(tmp: &[u16], base: usize, ss_hor: usize) -> __m128i {
    if ss_hor == 0 {
        unsafe { _mm_loadl_epi64(tmp.as_ptr().add(base) as *const __m128i) }
    } else {
        load_u16x4_step2(unsafe { tmp.as_ptr().add(base) })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_idx4(dst: &mut [u8], base: usize, out16: __m128i) {
    let out = _mm_packus_epi16(out16, out16);
    let packed = _mm_cvtsi128_si32(out) as u32;
    let bytes = packed.to_le_bytes();
    dst[base..base + 4].copy_from_slice(&bytes);
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn ccso_prep_lut_hbd_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u16],
    tmp_stride: usize,
    o: usize,
    w: usize,
    h: usize,
    ss_hor: usize,
    ss_ver: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
    bo_only: bool,
) {
    if ss_hor > 1 {
        crate::ccso::ccso_prep_lut_hbd_scalar(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            ss_hor,
            ss_ver,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
            bo_only,
        );
        return;
    }

    let q = _mm_set1_epi16(quant_step as i16);
    let nq = _mm_set1_epi16((-quant_step) as i16);
    let shiftv = _mm_cvtsi32_si128(shift as i32);

    for y in 0..h {
        let row = o + (y << ss_ver) * tmp_stride;
        let dst_base = y * dst_stride;
        let mut x = 0usize;
        if bo_only {
            while x + 8 <= w {
                let base = row + (x << ss_hor);
                let c = load_u16x8_subsampled(tmp, base, ss_hor);
                let out16 = _mm_srl_epi16(c, shiftv);
                let out8 = _mm_packus_epi16(out16, out16);
                unsafe {
                    _mm_storel_epi64(dst.as_mut_ptr().add(dst_base + x) as *mut __m128i, out8)
                };
                x += 8;
            }
            if x + 4 <= w {
                let base = row + (x << ss_hor);
                let c = load_u16x4_subsampled(tmp, base, ss_hor);
                store_idx4(dst, dst_base + x, _mm_srl_epi16(c, shiftv));
                x += 4;
            }
        } else {
            while x + 8 <= w {
                let base = row + (x << ss_hor);
                let c = load_u16x8_subsampled(tmp, base, ss_hor);
                let p0 = load_u16x8_subsampled(tmp, (base as isize + luma_offset) as usize, ss_hor);
                let p1 = load_u16x8_subsampled(tmp, (base as isize - luma_offset) as usize, ss_hor);
                let out16 = ccso_make_idx_8x16(c, p0, p1, shiftv, q, nq, edge_clf);
                let out8 = _mm_packus_epi16(out16, out16);
                unsafe {
                    _mm_storel_epi64(dst.as_mut_ptr().add(dst_base + x) as *mut __m128i, out8)
                };
                x += 8;
            }
            if x + 4 <= w {
                let base = row + (x << ss_hor);
                let c = load_u16x4_subsampled(tmp, base, ss_hor);
                let p0 = load_u16x4_subsampled(tmp, (base as isize + luma_offset) as usize, ss_hor);
                let p1 = load_u16x4_subsampled(tmp, (base as isize - luma_offset) as usize, ss_hor);
                let out16 = ccso_make_idx_8x16(c, p0, p1, shiftv, q, nq, edge_clf);
                store_idx4(dst, dst_base + x, out16);
                x += 4;
            }
        }
        ccso_tail_hbd(
            dst,
            dst_base,
            tmp,
            row,
            x,
            w,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
            bo_only,
        );
    }
}

#[inline(always)]
fn fill_offsets_8(out: &mut [i16; 8], idx: &[u8], offset_idxs: &[u8], offset_lut: &[i8]) {
    for i in 0..8 {
        out[i] = crate::ccso::ccso_offset(idx[i], offset_idxs, offset_lut) as i16;
    }
}

#[inline(always)]
fn fill_offsets_4_i16(out: &mut [i16; 8], idx: &[u8], offset_idxs: &[u8], offset_lut: &[i8]) {
    for i in 0..4 {
        out[i] = crate::ccso::ccso_offset(idx[i], offset_idxs, offset_lut) as i16;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ccso_add_4x4_hbd(
    dst: &mut [u16],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    xx: usize,
    yy: usize,
    bitdepth_max: i32,
) {
    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi16(bitdepth_max as i16);
    let mut off_tmp = [0i16; 8];
    for y in yy..yy + 4 {
        let ip = y * idx_stride + xx;
        fill_offsets_4_i16(&mut off_tmp, &idx_buf[ip..ip + 4], offset_idxs, offset_lut);
        let off = unsafe { _mm_loadu_si128(off_tmp.as_ptr() as *const __m128i) };
        let dp = y * dst_stride + xx;
        let cur = unsafe { _mm_loadl_epi64(dst.as_ptr().add(dp) as *const __m128i) };
        let out = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(cur, off), zero), maxv);
        unsafe { _mm_storel_epi64(dst.as_mut_ptr().add(dp) as *mut __m128i, out) };
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn ccso_add_hbd_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
    bitdepth_max: i32,
) {
    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi16(bitdepth_max as i16);
    let mut off_tmp = [0i16; 8];
    for yy in (0..h).step_by(4) {
        let mi = yy >> 2;
        let row_mask = ll_mask[mi][0];
        let mut xx = 0usize;
        while xx + 8 <= w {
            let bx = xx >> 2;
            if ((row_mask >> bx) & 0x03) == 0 {
                for y in yy..yy + 4 {
                    let ip = y * idx_stride + xx;
                    fill_offsets_8(&mut off_tmp, &idx_buf[ip..ip + 8], offset_idxs, offset_lut);
                    let off = unsafe { _mm_loadu_si128(off_tmp.as_ptr() as *const __m128i) };
                    let dp = y * dst_stride + xx;
                    let cur = unsafe { _mm_loadu_si128(dst.as_ptr().add(dp) as *const __m128i) };
                    let out = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(cur, off), zero), maxv);
                    unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(dp) as *mut __m128i, out) };
                }
                xx += 8;
            } else {
                for _ in 0..2 {
                    let bx = xx >> 2;
                    if row_mask & (1 << bx) == 0 {
                        ccso_add_4x4_hbd(
                            dst,
                            dst_stride,
                            idx_buf,
                            idx_stride,
                            offset_idxs,
                            offset_lut,
                            xx,
                            yy,
                            bitdepth_max,
                        );
                    }
                    xx += 4;
                }
            }
        }
        while xx < w {
            let bx = xx >> 2;
            if row_mask & (1 << bx) == 0 {
                ccso_add_4x4_hbd(
                    dst,
                    dst_stride,
                    idx_buf,
                    idx_stride,
                    offset_idxs,
                    offset_lut,
                    xx,
                    yy,
                    bitdepth_max,
                );
            }
            xx += 4;
        }
    }
}
