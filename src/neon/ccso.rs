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
fn ccso_tail_8bpc<const SS_HOR: usize, const BO_ONLY: bool>(
    dst_row: &mut [u8],
    tmp: &[u8],
    row: usize,
    x0: usize,
    x1: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
) {
    for (x, out) in dst_row[x0..x1].iter_mut().enumerate() {
        let x = x + x0;
        let ti = row + (x << SS_HOR);
        let c = tmp[ti] as i32;
        let band = (c as u32 >> shift) as u8;
        if BO_ONLY {
            *out = band;
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
            *out = ((cls0 << 5) | (cls1 << 3)) as u8 | band;
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
fn load_u8x16_subsampled<const SS_HOR: usize>(tmp: &[u8], base: usize) -> uint8x16_t {
    if SS_HOR == 0 {
        unsafe { vld1q_u8(tmp.as_ptr().add(base)) }
    } else {
        let a = unsafe { vld1q_u8(tmp.as_ptr().add(base)) };
        let b = unsafe { vld1q_u8(tmp.as_ptr().add(base + 16)) };
        vuzp1q_u8(a, b)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8_subsampled<const SS_HOR: usize>(tmp: &[u8], base: usize) -> uint8x8_t {
    if SS_HOR == 0 {
        unsafe { vld1_u8(tmp.as_ptr().add(base)) }
    } else {
        let a = unsafe { vld1q_u8(tmp.as_ptr().add(base)) };
        vget_low_u8(vuzp1q_u8(a, a))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn ccso_prep_lut_8bpc_neon_impl<const SS_HOR: usize, const SS_VER: usize, const BO_ONLY: bool>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    tmp_stride: usize,
    o: usize,
    w: usize,
    h: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
) {
    let q = vdupq_n_s16(quant_step as i16);
    let nq = vdupq_n_s16((-quant_step) as i16);
    let sh = vdupq_n_s16(-(shift as i16));

    for (y, dst_row) in dst.chunks_exact_mut(dst_stride).take(h).enumerate() {
        let row = o + (y << SS_VER) * tmp_stride;
        let dst_row = &mut dst_row[..w];
        let mut x;
        if BO_ONLY {
            let n16 = {
                let (dst16, _) = dst_row.as_chunks_mut::<16>();
                for (chunk_idx, out16) in dst16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let base = row + (x << SS_HOR);
                    let c = load_u8x16_subsampled::<SS_HOR>(tmp, base);
                    let lo = vshlq_u16(vmovl_u8(vget_low_u8(c)), sh);
                    let hi = vshlq_u16(vmovl_u8(vget_high_u8(c)), sh);
                    let out = vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));
                    unsafe { vst1q_u8(out16.as_mut_ptr(), out) };
                }
                dst16.len()
            };
            x = n16 * 16;
            if x + 8 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u8x8_subsampled::<SS_HOR>(tmp, base);
                let out8 = &mut dst_row[x..x + 8].as_chunks_mut::<8>().0[0];
                unsafe { vst1_u8(out8.as_mut_ptr(), vqmovn_u16(vshlq_u16(vmovl_u8(c), sh))) };
                x += 8;
            }
        } else {
            let n16 = {
                let (dst16, _) = dst_row.as_chunks_mut::<16>();
                for (chunk_idx, out16) in dst16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let base = row + (x << SS_HOR);
                    let c = load_u8x16_subsampled::<SS_HOR>(tmp, base);
                    let p0 = load_u8x16_subsampled::<SS_HOR>(
                        tmp,
                        (base as isize + luma_offset) as usize,
                    );
                    let p1 = load_u8x16_subsampled::<SS_HOR>(
                        tmp,
                        (base as isize - luma_offset) as usize,
                    );
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
                    unsafe { vst1q_u8(out16.as_mut_ptr(), out) };
                }
                dst16.len()
            };
            x = n16 * 16;
            if x + 8 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u8x8_subsampled::<SS_HOR>(tmp, base);
                let p0 =
                    load_u8x8_subsampled::<SS_HOR>(tmp, (base as isize + luma_offset) as usize);
                let p1 =
                    load_u8x8_subsampled::<SS_HOR>(tmp, (base as isize - luma_offset) as usize);
                let out =
                    ccso_make_idx_u16(vmovl_u8(c), vmovl_u8(p0), vmovl_u8(p1), sh, q, nq, edge_clf);
                let out8 = &mut dst_row[x..x + 8];
                unsafe { vst1_u8(out8.as_mut_ptr(), vqmovn_u16(out)) };
                x += 8;
            }
        }
        ccso_tail_8bpc::<SS_HOR, BO_ONLY>(
            dst_row,
            tmp,
            row,
            x,
            w,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        );
    }
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
    match (ss_hor, ss_ver, bo_only) {
        (0, 0, true) => ccso_prep_lut_8bpc_neon_impl::<0, 0, true>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (0, 0, false) => ccso_prep_lut_8bpc_neon_impl::<0, 0, false>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 0, true) => ccso_prep_lut_8bpc_neon_impl::<1, 0, true>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 0, false) => ccso_prep_lut_8bpc_neon_impl::<1, 0, false>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 1, true) => ccso_prep_lut_8bpc_neon_impl::<1, 1, true>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 1, false) => ccso_prep_lut_8bpc_neon_impl::<1, 1, false>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        _ => crate::ccso::ccso_prep_lut_8bpc_scalar(
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
        ),
    }
}

#[inline(always)]
fn fill_offsets_16(out: &mut [i8; 16], idx: &[u8; 16], offset_map: &[i8; 256]) {
    for (out, &idx) in out.iter_mut().zip(idx.iter()) {
        *out = offset_map[idx as usize];
    }
}

#[inline(always)]
fn fill_offsets_4_i16(idx: &[u8; 4], offset_map: &[i8; 256]) -> int16x4_t {
    let mut out = [0i16; 4];
    for (out, &idx) in out.iter_mut().zip(idx.iter()) {
        *out = offset_map[idx as usize] as i16;
    }
    unsafe { vld1_s16(out.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn ccso_add_4x4_8bpc(
    dst_rows: &mut [u8],
    dst_stride: usize,
    idx_rows: &[u8],
    idx_stride: usize,
    block_h: usize,
    offset_map: &[i8; 256],
    xx: usize,
) {
    for (yy, idx_row) in idx_rows.chunks_exact(idx_stride).take(block_h).enumerate() {
        let dst_row = &mut dst_rows[yy * dst_stride..];
        let idx4 = &idx_row[xx..xx + 4].as_chunks::<4>().0[0];
        let off = fill_offsets_4_i16(idx4, offset_map);
        let dst4 = &mut dst_row[xx..xx + 4];
        let src_q =
            unsafe { vreinterpret_u8_u32(vld1_lane_u32::<0>(dst4.as_ptr().cast(), vdup_n_u32(0))) };
        let cur = vreinterpretq_s16_u16(vmovl_u8(src_q));
        let out = vqmovun_s16(vaddq_s16(cur, vcombine_s16(off, vdup_n_s16(0))));
        unsafe {
            vst1_lane_u32::<0>(dst4.as_mut_ptr().cast(), vreinterpret_u32_u8(out));
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
    let offset_map = crate::ccso::ccso_build_offset_map(offset_idxs, offset_lut);
    let mut off_tmp = [0i8; 16];
    let n_blocks = (h + 3) >> 2;
    for (by, mask) in (0..h).step_by(4).zip(ll_mask[..n_blocks].iter()) {
        let block_h = (h - by).min(4);
        // `dst` may already start at an x-offset inside the picture row, so
        // chunking it by `dst_stride` would mix row tails with the next row.
        let dst_rows = &mut dst[by * dst_stride..];
        let idx_row_start = by * idx_stride;
        let idx_rows = &idx_buf[idx_row_start..idx_row_start + block_h * idx_stride];
        let row_mask = mask[0];
        let mut xx = 0usize;
        while xx + 16 <= w {
            let bx = xx >> 2;
            if ((row_mask >> bx) & 0x0f) == 0 {
                for (yy, idx_row) in idx_rows.chunks_exact(idx_stride).take(block_h).enumerate() {
                    let dst_row = &mut dst_rows[yy * dst_stride..];
                    let idx16 = &idx_row[xx..xx + 16].as_chunks::<16>().0[0];
                    fill_offsets_16(&mut off_tmp, idx16, &offset_map);
                    let off = unsafe { vld1q_s8(off_tmp.as_ptr()) };
                    let off_lo = vmovl_s8(vget_low_s8(off));
                    let off_hi = vmovl_s8(vget_high_s8(off));
                    let dst16 = &mut dst_row[xx..xx + 16];
                    let cur = unsafe { vld1q_u8(dst16.as_ptr()) };
                    let cur_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(cur)));
                    let cur_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(cur)));
                    let out_lo = vqmovun_s16(vaddq_s16(cur_lo, off_lo));
                    let out_hi = vqmovun_s16(vaddq_s16(cur_hi, off_hi));
                    let out = vcombine_u8(out_lo, out_hi);
                    unsafe { vst1q_u8(dst16.as_mut_ptr(), out) };
                }
                xx += 16;
            } else {
                for _ in 0..4 {
                    let bx = xx >> 2;
                    if row_mask & (1 << bx) == 0 {
                        ccso_add_4x4_8bpc(
                            dst_rows,
                            dst_stride,
                            idx_rows,
                            idx_stride,
                            block_h,
                            &offset_map,
                            xx,
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
                    dst_rows,
                    dst_stride,
                    idx_rows,
                    idx_stride,
                    block_h,
                    &offset_map,
                    xx,
                );
            }
            xx += 4;
        }
    }
}
