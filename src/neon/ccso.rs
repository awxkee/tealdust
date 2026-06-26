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

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16_subsampled(tmp: &[u8], base: usize, ss_hor: usize) -> uint8x16_t {
    if ss_hor == 0 {
        unsafe { vld1q_u8(tmp.as_ptr().add(base)) }
    } else {
        let a = unsafe { vld1q_u8(tmp.as_ptr().add(base)) };
        let b = unsafe { vld1q_u8(tmp.as_ptr().add(base + 16)) };
        vuzp1q_u8(a, b)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8_subsampled(tmp: &[u8], base: usize, ss_hor: usize) -> uint8x8_t {
    if ss_hor == 0 {
        unsafe { vld1_u8(tmp.as_ptr().add(base)) }
    } else {
        let a = unsafe { vld1q_u8(tmp.as_ptr().add(base)) };
        vget_low_u8(vuzp1q_u8(a, a))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_idx8(dst: &mut [u8], base: usize, out16: uint16x8_t) {
    unsafe { vst1_u8(dst.as_mut_ptr().add(base), vqmovn_u16(out16)) };
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn ccso_prep_lut_8bpc_neon(
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
    if ss_hor > 1 {
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

    let q = vdupq_n_s16(quant_step as i16);
    let nq = vdupq_n_s16((-quant_step) as i16);
    let sh = vdupq_n_s16(-(shift as i16));

    for y in 0..h {
        let row = o + (y << ss_ver) * tmp_stride;
        let dst_base = y * dst_stride;
        let mut x = 0usize;
        if bo_only {
            while x + 16 <= w {
                let base = row + (x << ss_hor);
                let c = load_u8x16_subsampled(tmp, base, ss_hor);
                let lo = vshlq_u16(vmovl_u8(vget_low_u8(c)), sh);
                let hi = vshlq_u16(vmovl_u8(vget_high_u8(c)), sh);
                let out = vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));
                unsafe { vst1q_u8(dst.as_mut_ptr().add(dst_base + x), out) };
                x += 16;
            }
            if x + 8 <= w {
                let base = row + (x << ss_hor);
                let c = load_u8x8_subsampled(tmp, base, ss_hor);
                store_idx8(dst, dst_base + x, vshlq_u16(vmovl_u8(c), sh));
                x += 8;
            }
        } else {
            while x + 16 <= w {
                let base = row + (x << ss_hor);
                let c = load_u8x16_subsampled(tmp, base, ss_hor);
                let p0 = load_u8x16_subsampled(tmp, (base as isize + luma_offset) as usize, ss_hor);
                let p1 = load_u8x16_subsampled(tmp, (base as isize - luma_offset) as usize, ss_hor);
                let lo = ccso_make_idx_u16(
                    vmovl_u8(vget_low_u8(c)),
                    vmovl_u8(vget_low_u8(p0)),
                    vmovl_u8(vget_low_u8(p1)),
                    sh,
                    q,
                    nq,
                    edge_clf,
                );
                let hi = ccso_make_idx_u16(
                    vmovl_u8(vget_high_u8(c)),
                    vmovl_u8(vget_high_u8(p0)),
                    vmovl_u8(vget_high_u8(p1)),
                    sh,
                    q,
                    nq,
                    edge_clf,
                );
                let out = vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));
                unsafe { vst1q_u8(dst.as_mut_ptr().add(dst_base + x), out) };
                x += 16;
            }
            if x + 8 <= w {
                let base = row + (x << ss_hor);
                let c = load_u8x8_subsampled(tmp, base, ss_hor);
                let p0 = load_u8x8_subsampled(tmp, (base as isize + luma_offset) as usize, ss_hor);
                let p1 = load_u8x8_subsampled(tmp, (base as isize - luma_offset) as usize, ss_hor);
                let out =
                    ccso_make_idx_u16(vmovl_u8(c), vmovl_u8(p0), vmovl_u8(p1), sh, q, nq, edge_clf);
                store_idx8(dst, dst_base + x, out);
                x += 8;
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
fn fill_offsets_4_i16(idx: &[u8], offset_idxs: &[u8], offset_lut: &[i8]) -> int16x4_t {
    let mut out = [0i16; 4];
    out[0] = crate::ccso::ccso_offset(idx[0], offset_idxs, offset_lut) as i16;
    out[1] = crate::ccso::ccso_offset(idx[1], offset_idxs, offset_lut) as i16;
    out[2] = crate::ccso::ccso_offset(idx[2], offset_idxs, offset_lut) as i16;
    out[3] = crate::ccso::ccso_offset(idx[3], offset_idxs, offset_lut) as i16;
    unsafe { vld1_s16(out.as_mut_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "neon")]
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
        let ip = y * idx_stride + xx;
        let off = fill_offsets_4_i16(&idx_buf[ip..ip + 4], offset_idxs, offset_lut);
        let dp = y * dst_stride + xx;
        let src_q = unsafe {
            vreinterpret_u8_u32(vld1_lane_u32::<0>(
                dst.as_ptr().add(dp).cast(),
                vdup_n_u32(0),
            ))
        };
        let cur = vreinterpretq_s16_u16(vmovl_u8(src_q));
        let out = vqmovun_s16(vaddq_s16(cur, vcombine_s16(off, vdup_n_s16(0))));
        unsafe {
            vst1_lane_u32::<0>(dst.as_mut_ptr().add(dp).cast(), vreinterpret_u32_u8(out));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn ccso_add_8bpc_neon(
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
                    let off = unsafe { vld1q_s8(off_tmp.as_ptr()) };
                    let off_lo = vmovl_s8(vget_low_s8(off));
                    let off_hi = vmovl_s8(vget_high_s8(off));
                    let dp = y * dst_stride + xx;
                    let cur = unsafe { vld1q_u8(dst.as_ptr().add(dp)) };
                    let cur_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cur)));
                    let cur_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cur)));
                    let out_lo = vqmovun_s16(vaddq_s16(cur_lo, off_lo));
                    let out_hi = vqmovun_s16(vaddq_s16(cur_hi, off_hi));
                    let out = vcombine_u8(out_lo, out_hi);
                    unsafe { vst1q_u8(dst.as_mut_ptr().add(dp), out) };
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
