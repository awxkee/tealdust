/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice
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
fn ccso_tail_hbd<const SS_HOR: usize, const BO_ONLY: bool>(
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
) {
    for x in x0..x1 {
        let ti = row + (x << SS_HOR);
        let c = tmp[ti] as i32;
        let band = (c as u32 >> shift) as u8;
        if BO_ONLY {
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
fn load_u16x8_subsampled<const SS_HOR: usize>(tmp: &[u16], base: usize) -> uint16x8_t {
    if SS_HOR == 0 {
        unsafe { vld1q_u16(tmp.as_ptr().add(base)) }
    } else {
        let a = unsafe { vld1q_u16(tmp.as_ptr().add(base)) };
        let b = unsafe { vld1q_u16(tmp.as_ptr().add(base + 8)) };
        vuzp1q_u16(a, b)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x4_subsampled<const SS_HOR: usize>(tmp: &[u16], base: usize) -> uint16x4_t {
    if SS_HOR == 0 {
        unsafe { vld1_u16(tmp.as_ptr().add(base)) }
    } else {
        let a = unsafe { vld1q_u16(tmp.as_ptr().add(base)) };
        vget_low_u16(vuzp1q_u16(a, a))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_idx4(dst: &mut [u8], base: usize, out16: uint16x8_t) {
    let out = vqmovn_u16(out16);
    unsafe {
        vst1_lane_u32::<0>(dst.as_mut_ptr().add(base).cast(), vreinterpret_u32_u8(out));
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn ccso_prep_lut_hbd_neon_impl<const SS_HOR: usize, const SS_VER: usize, const BO_ONLY: bool>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u16],
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

    for y in 0..h {
        let row = o + (y << SS_VER) * tmp_stride;
        let dst_base = y * dst_stride;
        let mut x = 0usize;
        if BO_ONLY {
            while x + 8 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u16x8_subsampled::<SS_HOR>(tmp, base);
                let out16 = vshlq_u16(c, sh);
                let out = vqmovn_u16(out16);
                unsafe { vst1_u8(dst.as_mut_ptr().add(dst_base + x), out) };
                x += 8;
            }
            if x + 4 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u16x4_subsampled::<SS_HOR>(tmp, base);
                let out16 = vcombine_u16(c, vdup_n_u16(0));
                store_idx4(dst, dst_base + x, vshlq_u16(out16, sh));
                x += 4;
            }
        } else {
            while x + 8 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u16x8_subsampled::<SS_HOR>(tmp, base);
                let p0 =
                    load_u16x8_subsampled::<SS_HOR>(tmp, (base as isize + luma_offset) as usize);
                let p1 =
                    load_u16x8_subsampled::<SS_HOR>(tmp, (base as isize - luma_offset) as usize);
                let out16 = ccso_make_idx_u16(c, p0, p1, sh, q, nq, edge_clf);
                let out = vqmovn_u16(out16);
                unsafe { vst1_u8(dst.as_mut_ptr().add(dst_base + x), out) };
                x += 8;
            }
            if x + 4 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u16x4_subsampled::<SS_HOR>(tmp, base);
                let p0 =
                    load_u16x4_subsampled::<SS_HOR>(tmp, (base as isize + luma_offset) as usize);
                let p1 =
                    load_u16x4_subsampled::<SS_HOR>(tmp, (base as isize - luma_offset) as usize);
                let out16 = ccso_make_idx_u16(
                    vcombine_u16(c, vdup_n_u16(0)),
                    vcombine_u16(p0, vdup_n_u16(0)),
                    vcombine_u16(p1, vdup_n_u16(0)),
                    sh,
                    q,
                    nq,
                    edge_clf,
                );
                store_idx4(dst, dst_base + x, out16);
                x += 4;
            }
        }
        ccso_tail_hbd::<SS_HOR, BO_ONLY>(
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
        );
    }
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
    match (ss_hor, ss_ver, bo_only) {
        (0, 0, true) => ccso_prep_lut_hbd_neon_impl::<0, 0, true>(
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
        (0, 0, false) => ccso_prep_lut_hbd_neon_impl::<0, 0, false>(
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
        (1, 0, true) => ccso_prep_lut_hbd_neon_impl::<1, 0, true>(
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
        (1, 0, false) => ccso_prep_lut_hbd_neon_impl::<1, 0, false>(
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
        (1, 1, true) => ccso_prep_lut_hbd_neon_impl::<1, 1, true>(
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
        (1, 1, false) => ccso_prep_lut_hbd_neon_impl::<1, 1, false>(
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
        _ => crate::ccso::ccso_prep_lut_hbd_scalar(
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
fn fill_offsets_8(out: &mut [i16; 8], idx: &[u8; 8], offset_map: &[i8; 256]) {
    for (out, &idx) in out.iter_mut().zip(idx.iter()) {
        *out = offset_map[idx as usize] as i16;
    }
}

#[inline(always)]
fn fill_offsets_4_i16(out: &mut [i16; 4], idx: &[u8; 4], offset_map: &[i8; 256]) {
    for (out, &idx) in out.iter_mut().zip(idx.iter()) {
        *out = offset_map[idx as usize] as i16;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ccso_add_4x4_hbd(
    dst_rows: &mut [u16],
    dst_stride: usize,
    idx_rows: &[u8],
    idx_stride: usize,
    offset_map: &[i8; 256],
    xx: usize,
    bitdepth_max: i32,
) {
    let zero = vdup_n_s16(0);
    let maxv = vdup_n_s16(bitdepth_max as i16);
    let mut off_tmp = [0i16; 4];
    for (dst_row, idx_row) in dst_rows
        .chunks_exact_mut(dst_stride)
        .zip(idx_rows.chunks_exact(idx_stride))
    {
        let idx4 = &idx_row[xx..xx + 4].as_chunks::<4>().0[0];
        fill_offsets_4_i16(&mut off_tmp, idx4, offset_map);
        let off = unsafe { vld1_s16(off_tmp.as_ptr()) };
        let cur = unsafe { vreinterpret_s16_u16(vld1_u16(dst_row.as_ptr().add(xx))) };
        let out = vmin_s16(vmax_s16(vadd_s16(cur, off), zero), maxv);
        unsafe { vst1_u16(dst_row.as_mut_ptr().add(xx), vreinterpret_u16_s16(out)) };
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
    let offset_map = crate::ccso::ccso_build_offset_map(offset_idxs, offset_lut);
    let mut off_tmp = [0i16; 8];
    let n_blocks = (h + 3) >> 2;
    let dst_block_len = dst_stride * 4 * n_blocks;
    let idx_block_len = idx_stride * 4 * n_blocks;
    for ((dst_rows, idx_rows), mask) in dst[..dst_block_len]
        .chunks_exact_mut(dst_stride * 4)
        .zip(idx_buf[..idx_block_len].chunks_exact(idx_stride * 4))
        .zip(ll_mask[..n_blocks].iter())
    {
        let row_mask = mask[0];
        let mut xx = 0usize;
        while xx + 8 <= w {
            let bx = xx >> 2;
            if ((row_mask >> bx) & 0x03) == 0 {
                for (dst_row, idx_row) in dst_rows
                    .chunks_exact_mut(dst_stride)
                    .zip(idx_rows.chunks_exact(idx_stride))
                {
                    let idx8 = &idx_row[xx..xx + 8].as_chunks::<8>().0[0];
                    fill_offsets_8(&mut off_tmp, idx8, &offset_map);
                    let off = unsafe { vld1q_s16(off_tmp.as_ptr()) };
                    let cur = unsafe { vreinterpretq_s16_u16(vld1q_u16(dst_row.as_ptr().add(xx))) };
                    let out = vminq_s16(vmaxq_s16(vaddq_s16(cur, off), zero), maxv);
                    unsafe { vst1q_u16(dst_row.as_mut_ptr().add(xx), vreinterpretq_u16_s16(out)) };
                }
                xx += 8;
            } else {
                for _ in 0..2 {
                    let bx = xx >> 2;
                    if row_mask & (1 << bx) == 0 {
                        ccso_add_4x4_hbd(
                            dst_rows,
                            dst_stride,
                            idx_rows,
                            idx_stride,
                            &offset_map,
                            xx,
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
                    dst_rows,
                    dst_stride,
                    idx_rows,
                    idx_stride,
                    &offset_map,
                    xx,
                    bitdepth_max,
                );
            }
            xx += 4;
        }
    }
}
