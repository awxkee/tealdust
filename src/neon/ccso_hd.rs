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

use std::arch::aarch64::*;

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
#[target_feature(enable = "neon")]
fn ccso_classify_s16(diff: int16x8_t, q: int16x8_t, nq: int16x8_t, edge_clf: u32) -> uint16x8_t {
    let zero = vdupq_n_u16(0);
    let one = vdupq_n_u16(1);
    let two = vdupq_n_u16(2);
    let gt = if edge_clf == 0 {
        vcgtq_s16(diff, q)
    } else {
        zero
    };
    let lt = vcltq_s16(diff, nq);
    let cls = vbslq_u16(gt, two, one);
    vbslq_u16(lt, zero, cls)
}

#[inline]
#[target_feature(enable = "neon")]
fn ccso_make_idx_u16(
    c: uint16x8_t,
    p0: uint16x8_t,
    p1: uint16x8_t,
    sh: int16x8_t,
    q: int16x8_t,
    nq: int16x8_t,
    edge_clf: u32,
) -> uint16x8_t {
    let band = vshlq_u16(c, sh);
    let cs = vreinterpretq_s16_u16(c);
    let cls0 = ccso_classify_s16(vsubq_s16(vreinterpretq_s16_u16(p0), cs), q, nq, edge_clf);
    let cls1 = ccso_classify_s16(vsubq_s16(vreinterpretq_s16_u16(p1), cs), q, nq, edge_clf);
    vorrq_u16(
        band,
        vorrq_u16(vshlq_n_u16::<5>(cls0), vshlq_n_u16::<3>(cls1)),
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn ccso_prep_lut_hbd_neon(
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
    if ss_hor != 0 {
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

    let q = vdupq_n_s16(quant_step as i16);
    let nq = vdupq_n_s16((-quant_step) as i16);
    let sh = vdupq_n_s16(-(shift as i16));

    for y in 0..h {
        let row = o + (y << ss_ver) * tmp_stride;
        let dst_base = y * dst_stride;
        let mut x = 0usize;
        if bo_only {
            while x + 8 <= w {
                let c = unsafe { vld1q_u16(tmp.as_ptr().add(row + x)) };
                let out16 = vshlq_u16(c, sh);
                let out = vqmovn_u16(out16);
                unsafe { vst1_u8(dst.as_mut_ptr().add(dst_base + x), out) };
                x += 8;
            }
        } else {
            while x + 8 <= w {
                let c = unsafe { vld1q_u16(tmp.as_ptr().add(row + x)) };
                let p0 = unsafe {
                    vld1q_u16(
                        tmp.as_ptr()
                            .add(((row + x) as isize + luma_offset) as usize),
                    )
                };
                let p1 = unsafe {
                    vld1q_u16(
                        tmp.as_ptr()
                            .add(((row + x) as isize - luma_offset) as usize),
                    )
                };
                let out16 = ccso_make_idx_u16(c, p0, p1, sh, q, nq, edge_clf);
                let out = vqmovn_u16(out16);
                unsafe { vst1_u8(dst.as_mut_ptr().add(dst_base + x), out) };
                x += 8;
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
    for y in yy..yy + 4 {
        for x in xx..xx + 4 {
            let off =
                crate::ccso::ccso_offset(idx_buf[y * idx_stride + x], offset_idxs, offset_lut);
            let cur = dst[y * dst_stride + x] as i32;
            dst[y * dst_stride + x] = (cur + off as i32).clamp(0, bitdepth_max) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn ccso_add_hbd_neon(
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
    let zero = vdupq_n_s16(0);
    let maxv = vdupq_n_s16(bitdepth_max as i16);
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
                    let off = unsafe { vld1q_s16(off_tmp.as_ptr()) };
                    let dp = y * dst_stride + xx;
                    let cur = unsafe { vreinterpretq_s16_u16(vld1q_u16(dst.as_ptr().add(dp))) };
                    let out = vminq_s16(vmaxq_s16(vaddq_s16(cur, off), zero), maxv);
                    unsafe { vst1q_u16(dst.as_mut_ptr().add(dp), vreinterpretq_u16_s16(out)) };
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
