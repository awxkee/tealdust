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
fn ccso_tail_8bpc(
    dst: &mut [u8],
    dst_base: usize,
    tmp: &[u8],
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
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn ccso_prep_lut_8bpc_sse41(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
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
    if ss_hor != 0 {
        crate::ccso::ccso_prep_lut_8bpc_scalar(
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

    let zero = _mm_setzero_si128();
    let q = _mm_set1_epi16(quant_step as i16);
    let nq = _mm_set1_epi16((-quant_step) as i16);
    let shiftv = _mm_cvtsi32_si128(shift as i32);

    for y in 0..h {
        let row = o + (y << ss_ver) * tmp_stride;
        let dst_base = y * dst_stride;
        let mut x = 0usize;
        if bo_only {
            while x + 16 <= w {
                let c = unsafe { _mm_loadu_si128(tmp.as_ptr().add(row + x) as *const __m128i) };
                let lo = _mm_srl_epi16(_mm_unpacklo_epi8(c, zero), shiftv);
                let hi = _mm_srl_epi16(_mm_unpackhi_epi8(c, zero), shiftv);
                let out = _mm_packus_epi16(lo, hi);
                unsafe {
                    _mm_storeu_si128(dst.as_mut_ptr().add(dst_base + x) as *mut __m128i, out)
                };
                x += 16;
            }
        } else {
            while x + 16 <= w {
                let c8 = unsafe { _mm_loadu_si128(tmp.as_ptr().add(row + x) as *const __m128i) };
                let p08 = unsafe {
                    _mm_loadu_si128(
                        tmp.as_ptr()
                            .add(((row + x) as isize + luma_offset) as usize)
                            as *const __m128i,
                    )
                };
                let p18 = unsafe {
                    _mm_loadu_si128(
                        tmp.as_ptr()
                            .add(((row + x) as isize - luma_offset) as usize)
                            as *const __m128i,
                    )
                };
                let clo = _mm_unpacklo_epi8(c8, zero);
                let chi = _mm_unpackhi_epi8(c8, zero);
                let p0lo = _mm_unpacklo_epi8(p08, zero);
                let p0hi = _mm_unpackhi_epi8(p08, zero);
                let p1lo = _mm_unpacklo_epi8(p18, zero);
                let p1hi = _mm_unpackhi_epi8(p18, zero);
                let lo = ccso_make_idx_8x16(clo, p0lo, p1lo, shiftv, q, nq, edge_clf);
                let hi = ccso_make_idx_8x16(chi, p0hi, p1hi, shiftv, q, nq, edge_clf);
                let out = _mm_packus_epi16(lo, hi);
                unsafe {
                    _mm_storeu_si128(dst.as_mut_ptr().add(dst_base + x) as *mut __m128i, out)
                };
                x += 16;
            }
        }
        ccso_tail_8bpc(
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
fn fill_offsets_16(out: &mut [i8; 16], idx: &[u8], offset_idxs: &[u8], offset_lut: &[i8]) {
    for i in 0..16 {
        out[i] = crate::ccso::ccso_offset(idx[i], offset_idxs, offset_lut);
    }
}

#[inline(always)]
fn ccso_add_4x4_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    xx: usize,
    yy: usize,
) {
    for y in yy..yy + 4 {
        for x in xx..xx + 4 {
            let off =
                crate::ccso::ccso_offset(idx_buf[y * idx_stride + x], offset_idxs, offset_lut);
            let cur = dst[y * dst_stride + x] as i32;
            dst[y * dst_stride + x] = (cur + off as i32).clamp(0, 255) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn ccso_add_8bpc_sse41(
    dst: &mut [u8],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
) {
    let zero = _mm_setzero_si128();
    let mut off_tmp = [0i8; 16];
    for yy in (0..h).step_by(4) {
        let mi = yy >> 2;
        let row_mask = ll_mask[mi][0];
        let mut xx = 0usize;
        while xx + 16 <= w {
            let bx = xx >> 2;
            if ((row_mask >> bx) & 0x0f) == 0 {
                for y in yy..yy + 4 {
                    let ip = y * idx_stride + xx;
                    fill_offsets_16(&mut off_tmp, &idx_buf[ip..ip + 16], offset_idxs, offset_lut);
                    let off = unsafe { _mm_loadu_si128(off_tmp.as_ptr() as *const __m128i) };
                    let off_lo = _mm_cvtepi8_epi16(off);
                    let off_hi = _mm_cvtepi8_epi16(_mm_srli_si128(off, 8));
                    let dp = y * dst_stride + xx;
                    let cur = unsafe { _mm_loadu_si128(dst.as_ptr().add(dp) as *const __m128i) };
                    let cur_lo = _mm_unpacklo_epi8(cur, zero);
                    let cur_hi = _mm_unpackhi_epi8(cur, zero);
                    let out = _mm_packus_epi16(
                        _mm_add_epi16(cur_lo, off_lo),
                        _mm_add_epi16(cur_hi, off_hi),
                    );
                    unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(dp) as *mut __m128i, out) };
                }
                xx += 16;
            } else {
                for _ in 0..4 {
                    let bx = xx >> 2;
                    if row_mask & (1 << bx) == 0 {
                        ccso_add_4x4_8bpc(
                            dst,
                            dst_stride,
                            idx_buf,
                            idx_stride,
                            offset_idxs,
                            offset_lut,
                            xx,
                            yy,
                        );
                    }
                    xx += 4;
                }
            }
        }
        while xx < w {
            let bx = xx >> 2;
            if row_mask & (1 << bx) == 0 {
                ccso_add_4x4_8bpc(
                    dst,
                    dst_stride,
                    idx_buf,
                    idx_stride,
                    offset_idxs,
                    offset_lut,
                    xx,
                    yy,
                );
            }
            xx += 4;
        }
    }
}
