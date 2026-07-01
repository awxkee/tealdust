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

use crate::itx_2d::ITX_TMP_PIXELS;
use std::arch::aarch64::*;

#[inline(always)]
fn with_neon_itx_i16_scratch<R>(len: usize, f: impl FnOnce(&mut [i16]) -> R) -> R {
    assert!(len <= ITX_TMP_PIXELS);
    let mut scratch = [0i16; ITX_TMP_PIXELS];
    f(&mut scratch[..len])
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_wht4_i32x4(
    in0: int32x4_t,
    in1: int32x4_t,
    in2: int32x4_t,
    in3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let t0 = vaddq_s32(in0, in1);
    let t2 = vsubq_s32(in2, in3);
    let t4 = vshrq_n_s32::<1>(vsubq_s32(t0, t2));
    let t3 = vsubq_s32(t4, in3);
    let t1 = vsubq_s32(t4, in1);

    (vsubq_s32(t0, t3), t3, t1, vaddq_s32(t2, t1))
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_transpose4x4_i32(
    r0: int32x4_t,
    r1: int32x4_t,
    r2: int32x4_t,
    r3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let t01 = vtrnq_s32(r0, r1);
    let t23 = vtrnq_s32(r2, r3);
    (
        vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0)),
        vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1)),
        vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0)),
        vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_load_u8x4_pair(dst: &[u8], off0: usize, off1: usize) -> uint8x8_t {
    unsafe {
        let lanes = vld1_lane_u32::<0>(dst.as_ptr().add(off0).cast::<u32>(), vdup_n_u32(0));
        vreinterpret_u8_u32(vld1_lane_u32::<1>(
            dst.as_ptr().add(off1).cast::<u32>(),
            lanes,
        ))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_store_u8x4_pair(dst: &mut [u8], off0: usize, off1: usize, v: uint8x8_t) {
    let lanes = vreinterpret_u32_u8(v);
    unsafe {
        vst1_lane_u32::<0>(dst.as_mut_ptr().add(off0).cast::<u32>(), lanes);
        vst1_lane_u32::<1>(dst.as_mut_ptr().add(off1).cast::<u32>(), lanes);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_store_wht_4x4_8bpc(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    r0: int32x4_t,
    r1: int32x4_t,
    r2: int32x4_t,
    r3: int32x4_t,
) {
    let row0 = dst_off;
    let row1 = dst_off + stride;
    let row2 = row1 + stride;
    let row3 = row2 + stride;

    let d01 = neon_load_u8x4_pair(dst, row0, row1);
    let d23 = neon_load_u8x4_pair(dst, row2, row3);

    let r01 = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
    let r23 = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));
    let d01 = vreinterpretq_s16_u16(vmovl_u8(d01));
    let d23 = vreinterpretq_s16_u16(vmovl_u8(d23));

    let out01 = vqmovun_s16(vqaddq_s16(d01, r01));
    let out23 = vqmovun_s16(vqaddq_s16(d23, r23));

    neon_store_u8x4_pair(dst, row0, row1, out01);
    neon_store_u8x4_pair(dst, row2, row3, out23);
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_store_wht_row_hbd(dst: &mut [u16], off: usize, residual: int32x4_t, bitdepth_max: i32) {
    let d = unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(dst.as_ptr().add(off)))) };
    let p = vminq_s32(
        vmaxq_s32(vaddq_s32(d, residual), vdupq_n_s32(0)),
        vdupq_n_s32(bitdepth_max),
    );
    unsafe {
        vst1_u16(dst.as_mut_ptr().add(off), vqmovun_s32(p));
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn inv_wht_wht_4x4_i16_neon_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 16);
        let c0 = vshrq_n_s32::<3>(vmovl_s16(vld1_s16(coeff.as_ptr())));
        let c1 = vshrq_n_s32::<3>(vmovl_s16(vld1_s16(coeff.as_ptr().add(4))));
        let c2 = vshrq_n_s32::<3>(vmovl_s16(vld1_s16(coeff.as_ptr().add(8))));
        let c3 = vshrq_n_s32::<3>(vmovl_s16(vld1_s16(coeff.as_ptr().add(12))));

        let (c0, c1, c2, c3) = neon_wht4_i32x4(c0, c1, c2, c3);
        let (r0, r1, r2, r3) = neon_transpose4x4_i32(c0, c1, c2, c3);
        let (r0, r1, r2, r3) = neon_wht4_i32x4(r0, r1, r2, r3);

        neon_store_wht_4x4_8bpc(dst, dst_off, stride, r0, r1, r2, r3);

        let z = vdupq_n_s16(0);
        vst1q_s16(coeff.as_mut_ptr(), z);
        vst1q_s16(coeff.as_mut_ptr().add(8), z);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn inv_wht_wht_4x4_i32_neon_hbd(
    coeff: &mut [i32],
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    bitdepth_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 16);
        let c0 = vshrq_n_s32::<3>(vld1q_s32(coeff.as_ptr()));
        let c1 = vshrq_n_s32::<3>(vld1q_s32(coeff.as_ptr().add(4)));
        let c2 = vshrq_n_s32::<3>(vld1q_s32(coeff.as_ptr().add(8)));
        let c3 = vshrq_n_s32::<3>(vld1q_s32(coeff.as_ptr().add(12)));

        let (c0, c1, c2, c3) = neon_wht4_i32x4(c0, c1, c2, c3);
        let (r0, r1, r2, r3) = neon_transpose4x4_i32(c0, c1, c2, c3);
        let (r0, r1, r2, r3) = neon_wht4_i32x4(r0, r1, r2, r3);
        neon_store_wht_row_hbd(dst, dst_off, r0, bitdepth_max);
        neon_store_wht_row_hbd(dst, dst_off + stride, r1, bitdepth_max);
        neon_store_wht_row_hbd(dst, dst_off + stride * 2, r2, bitdepth_max);
        neon_store_wht_row_hbd(dst, dst_off + stride * 3, r3, bitdepth_max);

        let z = vdupq_n_s32(0);
        vst1q_s32(coeff.as_mut_ptr(), z);
        vst1q_s32(coeff.as_mut_ptr().add(4), z);
        vst1q_s32(coeff.as_mut_ptr().add(8), z);
        vst1q_s32(coeff.as_mut_ptr().add(12), z);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct16_i32x4_impl(s: &[int32x4_t; 16]) -> [int32x4_t; 16] {
    let z = vdupq_n_s32(0);
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            acc = vmlaq_n_s32(acc, s[j], crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + m]);
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_adst16_i32x4_impl(s: &[int32x4_t; 16], flip: bool) -> [int32x4_t; 16] {
    let rows = if flip {
        &crate::itx_1d::FLIPADST16_KERNEL_ROWS
    } else {
        &crate::itx_1d::ADST16_KERNEL_ROWS
    };
    let z = vdupq_n_s32(0);
    let mut out = [z; 16];
    let mut m = 0usize;
    while m < 16 {
        let row = &rows[m];
        let mut acc = z;
        let mut j = 0usize;
        while j < 16 {
            acc = vmlaq_n_s32(acc, s[j], row[j] as i32);
            j += 1;
        }
        out[m] = acc;
        m += 1;
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_tx16_i32x4_impl(s: &[int32x4_t; 16], kind: usize) -> [int32x4_t; 16] {
    match kind {
        crate::itx_2d::TX_KIND_DCT => neon_dct16_i32x4_impl(s),
        crate::itx_2d::TX_KIND_ADST => neon_adst16_i32x4_impl(s, false),
        crate::itx_2d::TX_KIND_FLIPADST => neon_adst16_i32x4_impl(s, true),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn iadst_dequant_16x16_neon_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        iadst_dequant_16x16_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        iadst_dequant_16x16_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn iadst_dequant_16x16_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 256);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut s = [vdupq_n_s32(0); 16];
            let mut j = 0usize;
            while j < 16 {
                let mut v = vld1q_s32(coeff.as_ptr().add(y + j * 16));
                if IS_RECT2 {
                    v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                }
                s[j] = v;
                j += 1;
            }
            let out = neon_tx16_i32x4_impl(&s, first_kind);
            let mut x = 0usize;
            while x < 16 {
                let g = [out[x], out[x + 1], out[x + 2], out[x + 3]];
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 16 {
            tmp[y * 32..y * 32 + 16].fill(0);
            y += 1;
        }
        coeff[..256].fill(0);
        let mut x = 0usize;
        while x < 16 {
            let mut s = [vdupq_n_s32(0); 16];
            let mut j = 0usize;
            while j < 16 {
                s[j] = vld1q_s32(tmp.as_ptr().add(x + j * 32));
                j += 1;
            }
            let out = neon_tx16_i32x4_impl(&s, second_kind);
            j = 0;
            while j < 16 {
                vst1q_s32(tmp.as_mut_ptr().add(x + j * 32), out[j]);
                j += 1;
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_store4x4_i32_clip(
    tmp: &mut [i32; ITX_TMP_PIXELS],
    off: usize,
    v0: int32x4_t,
    v1: int32x4_t,
    v2: int32x4_t,
    v3: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        macro_rules! clip {
            ($x:expr) => {{ vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32($x, rnd), nsh), minv), maxv) }};
        }
        let c0 = clip!(v0);
        let c1 = clip!(v1);
        let c2 = clip!(v2);
        let c3 = clip!(v3);
        let t01 = vtrnq_s32(c0, c1);
        let t23 = vtrnq_s32(c2, c3);
        let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
        let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
        let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
        let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
        vst1q_s32(tmp.as_mut_ptr().add(off), r0);
        vst1q_s32(tmp.as_mut_ptr().add(off + 32), r1);
        vst1q_s32(tmp.as_mut_ptr().add(off + 64), r2);
        vst1q_s32(tmp.as_mut_ptr().add(off + 96), r3);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_residual_round_i32x4(v: int32x4_t, rnd: int32x4_t, nsh: int32x4_t) -> int32x4_t {
    vshlq_s32(vaddq_s32(v, rnd), nsh)
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_residual_add_u8x4(
    dst: &mut [u8],
    off: usize,
    v: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
) {
    unsafe {
        debug_assert!(off + 4 <= dst.len());
        let r = neon_residual_round_i32x4(v, rnd, nsh);
        let p = dst.as_mut_ptr().add(off);
        let d4 = vreinterpret_u8_u32(vdup_n_u32(core::ptr::read_unaligned(p.cast::<u32>())));
        let d32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(d4))));
        let sum = vaddq_s32(d32, r);
        let out16 = vqmovun_s32(sum);
        let out8 = vqmovn_u16(vcombine_u16(out16, out16));
        vst1_lane_u32::<0>(p.cast(), vreinterpret_u32_u8(out8));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_residual_add_u8x4_expand_x2(
    dst: &mut [u8],
    off: usize,
    v: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
) {
    unsafe {
        debug_assert!(off + 8 <= dst.len());
        let r = neon_residual_round_i32x4(v, rnd, nsh);
        let r16 = vqmovn_s32(r);
        let zipped = vzip_s16(r16, r16);
        let r16x8 = vcombine_s16(zipped.0, zipped.1);
        let p = dst.as_mut_ptr().add(off);
        let d8 = vld1_u8(p);
        let d16 = vreinterpretq_s16_u16(vmovl_u8(d8));
        let sum = vqaddq_s16(d16, r16x8);
        vst1_u8(p, vqmovun_s16(sum));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_writeback4_i32_u8<const W: usize, const H: usize>(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    x: usize,
    y: usize,
    v: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
) {
    debug_assert!(x + 4 <= W);
    debug_assert!(y < H);
    if out_w > W {
        let ox = x * 2;
        let oy = if out_h > H { y * 2 } else { y };
        let off0 = dst_off + oy * dst_stride + ox;
        neon_residual_add_u8x4_expand_x2(dst, off0, v, rnd, nsh);
        if out_h > H {
            neon_residual_add_u8x4_expand_x2(dst, off0 + dst_stride, v, rnd, nsh);
        }
    } else {
        let ox = x;
        let oy = if out_h > H { y * 2 } else { y };
        let off0 = dst_off + oy * dst_stride + ox;
        neon_residual_add_u8x4(dst, off0, v, rnd, nsh);
        if out_h > H {
            neon_residual_add_u8x4(dst, off0 + dst_stride, v, rnd, nsh);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn add_tmp_to_dst_8bpc_neon(
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    tmp: &[i32],
    tmp_stride: usize,
    w: usize,
    h: usize,
    sw: usize,
    sh: usize,
    rnd: i32,
    shift: i32,
) -> bool {
    debug_assert!(sw <= tmp_stride);
    debug_assert!(w == sw || w == sw * 2);
    debug_assert!(h == sh || h == sh * 2);

    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-shift);

    if w > sw {
        if h > sh {
            let mut ty = 0usize;
            while ty < sh {
                let y = ty * 2;
                let mut tx = 0usize;
                while tx < sw {
                    unsafe {
                        let v = vld1q_s32(tmp.as_ptr().add(ty * tmp_stride + tx));
                        let off0 = dst_off + y * dst_stride + tx * 2;
                        neon_residual_add_u8x4_expand_x2(dst, off0, v, rnd_v, nsh);
                        neon_residual_add_u8x4_expand_x2(dst, off0 + dst_stride, v, rnd_v, nsh);
                    }
                    tx += 4;
                }
                ty += 1;
            }
        } else {
            let mut y = 0usize;
            while y < h {
                let mut tx = 0usize;
                while tx < sw {
                    unsafe {
                        let v = vld1q_s32(tmp.as_ptr().add(y * tmp_stride + tx));
                        let off = dst_off + y * dst_stride + tx * 2;
                        neon_residual_add_u8x4_expand_x2(dst, off, v, rnd_v, nsh);
                    }
                    tx += 4;
                }
                y += 1;
            }
        }
    } else if h > sh {
        let mut ty = 0usize;
        while ty < sh {
            let y = ty * 2;
            let mut x = 0usize;
            while x < w {
                unsafe {
                    let v = vld1q_s32(tmp.as_ptr().add(ty * tmp_stride + x));
                    let off0 = dst_off + y * dst_stride + x;
                    neon_residual_add_u8x4(dst, off0, v, rnd_v, nsh);
                    neon_residual_add_u8x4(dst, off0 + dst_stride, v, rnd_v, nsh);
                }
                x += 4;
            }
            ty += 1;
        }
    } else {
        let mut y = 0usize;
        while y < h {
            let mut x = 0usize;
            while x < w {
                unsafe {
                    let v = vld1q_s32(tmp.as_ptr().add(y * tmp_stride + x));
                    let off = dst_off + y * dst_stride + x;
                    neon_residual_add_u8x4(dst, off, v, rnd_v, nsh);
                }
                x += 4;
            }
            y += 1;
        }
    }

    true
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct32_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 32 {
            let mut v = vld1q_s32(coeff.as_ptr().add(base + j * 32));
            if IS_RECT2 {
                v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
            }
            a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
            a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]);
            a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]);
            a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]);
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct32_i32x4_from_tmp4(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 32 {
            let v = vld1q_s32(tmp.as_ptr().add(base + j * 32));
            a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m]);
            a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 1]);
            a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 2]);
            a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT32_DENSE_KERNEL[j * 32 + m + 3]);
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_tx8_i32x4_from_coeff4_const<const IS_RECT2: bool>(
    coeff: &[i32],
    base: usize,
    kind: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 8 {
            let mut v = vld1q_s32(coeff.as_ptr().add(base + j * 8));
            if IS_RECT2 {
                v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
            }
            a0 = vmlaq_n_s32(a0, v, tx8_coeff(kind, m, j));
            a1 = vmlaq_n_s32(a1, v, tx8_coeff(kind, m + 1, j));
            a2 = vmlaq_n_s32(a2, v, tx8_coeff(kind, m + 2, j));
            a3 = vmlaq_n_s32(a3, v, tx8_coeff(kind, m + 3, j));
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_tx8_i32x4_from_tmp4(
    tmp: &[i32; ITX_TMP_PIXELS],
    base: usize,
    kind: usize,
    m: usize,
) -> [int32x4_t; 4] {
    unsafe {
        let z = vdupq_n_s32(0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        let mut j = 0usize;
        while j < 8 {
            let v = vld1q_s32(tmp.as_ptr().add(base + j * 32));
            a0 = vmlaq_n_s32(a0, v, tx8_coeff(kind, m, j));
            a1 = vmlaq_n_s32(a1, v, tx8_coeff(kind, m + 1, j));
            a2 = vmlaq_n_s32(a2, v, tx8_coeff(kind, m + 2, j));
            a3 = vmlaq_n_s32(a3, v, tx8_coeff(kind, m + 3, j));
            j += 1;
        }
        [a0, a1, a2, a3]
    }
}

#[inline]
fn tx8_coeff(kind: usize, out: usize, input: usize) -> i32 {
    match kind {
        crate::itx_2d::TX_KIND_DCT => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_ADST => crate::itx_2d::ADST8_KW[out * 8 + input] as i32,
        crate::itx_2d::TX_KIND_FLIPADST => crate::itx_2d::ADST8_KW[(7 - out) * 8 + input] as i32,
        _ => unreachable!(),
    }
}

#[inline]
fn neon_identity_scale(n: usize) -> i16 {
    match n {
        4 => 128,
        8 => 181,
        16 => 256,
        32 => 362,
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_identity_i16x4_coeff_to_i32<const IS_RECT2: bool>(
    coeff: &[i16],
    off: usize,
    scale: i16,
) -> int32x4_t {
    let v = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, off);
    vmull_n_s16(v, scale)
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_identity_i16x4_scratch_to_i32(scratch: &[i16], off: usize, scale: i16) -> int32x4_t {
    let v = neon_load4_i16_scratch(scratch, off);
    vmull_n_s16(v, scale)
}

#[inline]
fn neon_tx_dense_coeff(kind: usize, n: usize, out: usize, input: usize) -> i32 {
    match (kind, n) {
        (crate::itx_2d::TX_KIND_DCT, 4) => crate::itx_2d::DCT4_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 8) => crate::itx_2d::DCT8_KW[out * 8 + input] as i32,
        (crate::itx_2d::TX_KIND_DCT, 16) => crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out],
        (crate::itx_2d::TX_KIND_DCT, 32) => crate::itx_2d::DCT32_DENSE_KERNEL[input * 32 + out],
        (crate::itx_2d::TX_KIND_ADST, 4) => crate::itx_1d::ADST4_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_ADST, 8) => crate::itx_1d::ADST8_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_ADST, 16) => crate::itx_1d::ADST16_KERNEL_ROWS[out][input] as i32,
        (crate::itx_2d::TX_KIND_FLIPADST, 4) => {
            crate::itx_1d::FLIPADST4_KERNEL_ROWS[out][input] as i32
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 8) => {
            crate::itx_1d::ADST8_KERNEL_ROWS[7 - out][input] as i32
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 16) => {
            crate::itx_1d::FLIPADST16_KERNEL_ROWS[out][input] as i32
        }
        _ => unreachable!(),
    }
}

#[inline]
fn neon_tx_dense_coeff_i16(kind: usize, n: usize, out: usize, input: usize) -> i16 {
    match (kind, n) {
        (crate::itx_2d::TX_KIND_IDENTITY, 4) => {
            if out == input {
                128
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 8) => {
            if out == input {
                181
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 16) => {
            if out == input {
                256
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_IDENTITY, 32) => {
            if out == input {
                362
            } else {
                0
            }
        }
        (crate::itx_2d::TX_KIND_DCT, 4) => crate::itx_2d::DCT4_KW[out * 8 + input] as i16,
        (crate::itx_2d::TX_KIND_DCT, 8) => crate::itx_2d::DCT8_KW[out * 8 + input] as i16,
        (crate::itx_2d::TX_KIND_DCT, 16) => {
            crate::itx_2d::DCT16_DENSE_KERNEL[input * 16 + out] as i16
        }
        (crate::itx_2d::TX_KIND_DCT, 32) => {
            crate::itx_2d::DCT32_DENSE_KERNEL[input * 32 + out] as i16
        }
        (crate::itx_2d::TX_KIND_ADST, 4) => crate::itx_1d::ADST4_KERNEL_ROWS[out][input] as i16,
        (crate::itx_2d::TX_KIND_ADST, 8) => crate::itx_1d::ADST8_KERNEL_ROWS[out][input] as i16,
        (crate::itx_2d::TX_KIND_ADST, 16) => crate::itx_1d::ADST16_KERNEL_ROWS[out][input] as i16,
        (crate::itx_2d::TX_KIND_FLIPADST, 4) => {
            crate::itx_1d::FLIPADST4_KERNEL_ROWS[out][input] as i16
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 8) => {
            crate::itx_1d::ADST8_KERNEL_ROWS[7 - out][input] as i16
        }
        (crate::itx_2d::TX_KIND_FLIPADST, 16) => {
            crate::itx_1d::FLIPADST16_KERNEL_ROWS[out][input] as i16
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn neon_tx_dense_coeff_i16_const<const KIND: usize, const N: usize>(
    out: usize,
    input: usize,
) -> i16 {
    neon_tx_dense_coeff_i16(KIND, N, out, input)
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_load4_i16_coeff_packed_const<const IS_RECT2: bool>(src: &[i16], off: usize) -> int16x4_t {
    debug_assert!(off + 4 <= src.len());
    let v = unsafe { vld1_s16(src.as_ptr().add(off)) };
    if IS_RECT2 {
        let w = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), v, 181));
        vqmovn_s32(w)
    } else {
        v
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn neon_load4_i16_coeff_packed_rdm_const<const IS_RECT2: bool>(
    src: &[i16],
    off: usize,
) -> int16x4_t {
    debug_assert!(off + 4 <= src.len());
    let v = unsafe { vld1_s16(src.as_ptr().add(off)) };
    if IS_RECT2 {
        vqrdmulh_s16(v, vdup_n_s16(0x5a80))
    } else {
        v
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_load4_i16_scratch(src: &[i16], off: usize) -> int16x4_t {
    debug_assert!(off + 4 <= src.len());
    unsafe { vld1_s16(src.as_ptr().add(off)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_store4x4_i16_clip<const STRIDE: usize>(
    scratch: &mut [i16],
    off: usize,
    v0: int32x4_t,
    v1: int32x4_t,
    v2: int32x4_t,
    v3: int32x4_t,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) {
    unsafe {
        debug_assert!(STRIDE == 4 || STRIDE == 8 || STRIDE == 16 || STRIDE == 32);
        debug_assert!(off + 3 * STRIDE + 4 <= scratch.len());
        macro_rules! clip {
            ($x:expr) => {{ vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32($x, rnd), nsh), minv), maxv) }};
        }
        let c0 = clip!(v0);
        let c1 = clip!(v1);
        let c2 = clip!(v2);
        let c3 = clip!(v3);
        let t01 = vtrnq_s32(c0, c1);
        let t23 = vtrnq_s32(c2, c3);
        let r0 = vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0));
        let r1 = vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1));
        let r2 = vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0));
        let r3 = vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1));
        vst1_s16(scratch.as_mut_ptr().add(off), vqmovn_s32(r0));
        vst1_s16(scratch.as_mut_ptr().add(off + STRIDE), vqmovn_s32(r1));
        vst1_s16(scratch.as_mut_ptr().add(off + 2 * STRIDE), vqmovn_s32(r2));
        vst1_s16(scratch.as_mut_ptr().add(off + 3 * STRIDE), vqmovn_s32(r3));
    }
}

macro_rules! neon_dct16_i16x4_all_body {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(1), crate::itx_2d::DCT16_KBW[base]);
            acc = vmlal_n_s16(acc, load!(3), crate::itx_2d::DCT16_KBW[base + 1]);
            acc = vmlal_n_s16(acc, load!(5), crate::itx_2d::DCT16_KBW[base + 2]);
            acc = vmlal_n_s16(acc, load!(7), crate::itx_2d::DCT16_KBW[base + 3]);
            acc = vmlal_n_s16(acc, load!(9), crate::itx_2d::DCT16_KBW[base + 4]);
            acc = vmlal_n_s16(acc, load!(11), crate::itx_2d::DCT16_KBW[base + 5]);
            acc = vmlal_n_s16(acc, load!(13), crate::itx_2d::DCT16_KBW[base + 6]);
            acc = vmlal_n_s16(acc, load!(15), crate::itx_2d::DCT16_KBW[base + 7]);
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(2), crate::itx_2d::DCT16_KDW[base]);
            acc = vmlal_n_s16(acc, load!(6), crate::itx_2d::DCT16_KDW[base + 1]);
            acc = vmlal_n_s16(acc, load!(10), crate::itx_2d::DCT16_KDW[base + 2]);
            acc = vmlal_n_s16(acc, load!(14), crate::itx_2d::DCT16_KDW[base + 3]);
            d[m] = acc;
            m += 1;
        }
        let f0 = vaddq_s32(
            vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[0]),
            vmull_n_s16(load!(12), crate::itx_2d::DCT16_KFW[1]),
        );
        let f1 = vaddq_s32(
            vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[2]),
            vmull_n_s16(load!(12), crate::itx_2d::DCT16_KFW[3]),
        );
        let g0 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[0]),
            vmull_n_s16(load!(8), crate::itx_2d::DCT16_KGW[1]),
        );
        let g1 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[2]),
            vmull_n_s16(load!(8), crate::itx_2d::DCT16_KGW[3]),
        );
        let cc = [
            vaddq_s32(g0, f0),
            vaddq_s32(g1, f1),
            vsubq_s32(g1, f1),
            vsubq_s32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = vsubq_s32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 8] = vsubq_s32(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! neon_dct32_i16x4_all_body {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut grp = 0usize;
            while grp < 2 {
                let cb = base + grp * 8;
                let k0 = grp * 8;
                acc = vmlal_n_s16(acc, load!(2 * k0 + 1), crate::itx_2d::DCT32_KBW[cb]);
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 1) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 1],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 2) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 2],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 3) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 3],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 4) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 4],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 5) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 5],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 6) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 6],
                );
                acc = vmlal_n_s16(
                    acc,
                    load!(2 * (k0 + 7) + 1),
                    crate::itx_2d::DCT32_KBW[cb + 7],
                );
                grp += 1;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(2), crate::itx_2d::DCT32_KDW[base]);
            acc = vmlal_n_s16(acc, load!(6), crate::itx_2d::DCT32_KDW[base + 1]);
            acc = vmlal_n_s16(acc, load!(10), crate::itx_2d::DCT32_KDW[base + 2]);
            acc = vmlal_n_s16(acc, load!(14), crate::itx_2d::DCT32_KDW[base + 3]);
            acc = vmlal_n_s16(acc, load!(18), crate::itx_2d::DCT32_KDW[base + 4]);
            acc = vmlal_n_s16(acc, load!(22), crate::itx_2d::DCT32_KDW[base + 5]);
            acc = vmlal_n_s16(acc, load!(26), crate::itx_2d::DCT32_KDW[base + 6]);
            acc = vmlal_n_s16(acc, load!(30), crate::itx_2d::DCT32_KDW[base + 7]);
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            acc = vmlal_n_s16(acc, load!(4), crate::itx_2d::DCT32_KFW[base]);
            acc = vmlal_n_s16(acc, load!(12), crate::itx_2d::DCT32_KFW[base + 1]);
            acc = vmlal_n_s16(acc, load!(20), crate::itx_2d::DCT32_KFW[base + 2]);
            acc = vmlal_n_s16(acc, load!(28), crate::itx_2d::DCT32_KFW[base + 3]);
            f[m] = acc;
            m += 1;
        }
        let h0 = vaddq_s32(
            vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[0]),
            vmull_n_s16(load!(24), crate::itx_2d::DCT32_KHW[1]),
        );
        let h1 = vaddq_s32(
            vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[2]),
            vmull_n_s16(load!(24), crate::itx_2d::DCT32_KHW[3]),
        );
        let g0 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[0]),
            vmull_n_s16(load!(16), crate::itx_2d::DCT32_KGW[1]),
        );
        let g1 = vaddq_s32(
            vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[2]),
            vmull_n_s16(load!(16), crate::itx_2d::DCT32_KGW[3]),
        );
        let e = [
            vaddq_s32(g0, h0),
            vaddq_s32(g1, h1),
            vsubq_s32(g1, h1),
            vsubq_s32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = vaddq_s32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = vsubq_s32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = vsubq_s32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 16] = vsubq_s32(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! neon_dct16_i16x4_all_body_active {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 8];
        let mut m = 0usize;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            if ACTIVE > 1 {
                acc = vmlal_n_s16(acc, load!(1), crate::itx_2d::DCT16_KBW[base]);
            }
            if ACTIVE > 3 {
                acc = vmlal_n_s16(acc, load!(3), crate::itx_2d::DCT16_KBW[base + 1]);
            }
            if ACTIVE > 5 {
                acc = vmlal_n_s16(acc, load!(5), crate::itx_2d::DCT16_KBW[base + 2]);
            }
            if ACTIVE > 7 {
                acc = vmlal_n_s16(acc, load!(7), crate::itx_2d::DCT16_KBW[base + 3]);
            }
            if ACTIVE > 9 {
                acc = vmlal_n_s16(acc, load!(9), crate::itx_2d::DCT16_KBW[base + 4]);
            }
            if ACTIVE > 11 {
                acc = vmlal_n_s16(acc, load!(11), crate::itx_2d::DCT16_KBW[base + 5]);
            }
            if ACTIVE > 13 {
                acc = vmlal_n_s16(acc, load!(13), crate::itx_2d::DCT16_KBW[base + 6]);
            }
            if ACTIVE > 15 {
                acc = vmlal_n_s16(acc, load!(15), crate::itx_2d::DCT16_KBW[base + 7]);
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            if ACTIVE > 2 {
                acc = vmlal_n_s16(acc, load!(2), crate::itx_2d::DCT16_KDW[base]);
            }
            if ACTIVE > 6 {
                acc = vmlal_n_s16(acc, load!(6), crate::itx_2d::DCT16_KDW[base + 1]);
            }
            if ACTIVE > 10 {
                acc = vmlal_n_s16(acc, load!(10), crate::itx_2d::DCT16_KDW[base + 2]);
            }
            if ACTIVE > 14 {
                acc = vmlal_n_s16(acc, load!(14), crate::itx_2d::DCT16_KDW[base + 3]);
            }
            d[m] = acc;
            m += 1;
        }
        let f0 = if ACTIVE > 4 {
            let mut t = vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[0]);
            if ACTIVE > 12 {
                t = vmlal_n_s16(t, load!(12), crate::itx_2d::DCT16_KFW[1]);
            }
            t
        } else {
            z
        };
        let f1 = if ACTIVE > 4 {
            let mut t = vmull_n_s16(load!(4), crate::itx_2d::DCT16_KFW[2]);
            if ACTIVE > 12 {
                t = vmlal_n_s16(t, load!(12), crate::itx_2d::DCT16_KFW[3]);
            }
            t
        } else {
            z
        };
        let mut g0 = vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[0]);
        if ACTIVE > 8 {
            g0 = vmlal_n_s16(g0, load!(8), crate::itx_2d::DCT16_KGW[1]);
        }
        let mut g1 = vmull_n_s16(load!(0), crate::itx_2d::DCT16_KGW[2]);
        if ACTIVE > 8 {
            g1 = vmlal_n_s16(g1, load!(8), crate::itx_2d::DCT16_KGW[3]);
        }
        let cc = [
            vaddq_s32(g0, f0),
            vaddq_s32(g1, f1),
            vsubq_s32(g1, f1),
            vsubq_s32(g0, f0),
        ];
        let mut a = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 8 {
            a[i] = vsubq_s32(cc[7 - i], d[7 - i]);
            i += 1;
        }
        let mut out = [z; 16];
        let mut k = 0usize;
        while k < 8 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 8] = vsubq_s32(a[7 - k], b[7 - k]);
            k += 1;
        }
        out
    }};
}

macro_rules! neon_dct32_i16x4_all_body_active {
    () => {{
        let z = vdupq_n_s32(0);
        let mut b = [z; 16];
        let mut m = 0usize;
        while m < 16 {
            let base = m * 16;
            let mut acc = z;
            let mut k = 0usize;
            while k < 16 {
                let idx = 2 * k + 1;
                if ACTIVE > idx {
                    acc = vmlal_n_s16(acc, load!(idx), crate::itx_2d::DCT32_KBW[base + k]);
                }
                k += 1;
            }
            b[m] = acc;
            m += 1;
        }
        let mut d = [z; 8];
        m = 0;
        while m < 8 {
            let base = m * 8;
            let mut acc = z;
            let mut k = 0usize;
            while k < 8 {
                let idx = 4 * k + 2;
                if ACTIVE > idx {
                    acc = vmlal_n_s16(acc, load!(idx), crate::itx_2d::DCT32_KDW[base + k]);
                }
                k += 1;
            }
            d[m] = acc;
            m += 1;
        }
        let mut f = [z; 4];
        m = 0;
        while m < 4 {
            let base = m * 8;
            let mut acc = z;
            if ACTIVE > 4 {
                acc = vmlal_n_s16(acc, load!(4), crate::itx_2d::DCT32_KFW[base]);
            }
            if ACTIVE > 12 {
                acc = vmlal_n_s16(acc, load!(12), crate::itx_2d::DCT32_KFW[base + 1]);
            }
            if ACTIVE > 20 {
                acc = vmlal_n_s16(acc, load!(20), crate::itx_2d::DCT32_KFW[base + 2]);
            }
            if ACTIVE > 28 {
                acc = vmlal_n_s16(acc, load!(28), crate::itx_2d::DCT32_KFW[base + 3]);
            }
            f[m] = acc;
            m += 1;
        }
        let h0 = if ACTIVE > 8 {
            let mut t = vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[0]);
            if ACTIVE > 24 {
                t = vmlal_n_s16(t, load!(24), crate::itx_2d::DCT32_KHW[1]);
            }
            t
        } else {
            z
        };
        let h1 = if ACTIVE > 8 {
            let mut t = vmull_n_s16(load!(8), crate::itx_2d::DCT32_KHW[2]);
            if ACTIVE > 24 {
                t = vmlal_n_s16(t, load!(24), crate::itx_2d::DCT32_KHW[3]);
            }
            t
        } else {
            z
        };
        let mut g0 = vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[0]);
        if ACTIVE > 16 {
            g0 = vmlal_n_s16(g0, load!(16), crate::itx_2d::DCT32_KGW[1]);
        }
        let mut g1 = vmull_n_s16(load!(0), crate::itx_2d::DCT32_KGW[2]);
        if ACTIVE > 16 {
            g1 = vmlal_n_s16(g1, load!(16), crate::itx_2d::DCT32_KGW[3]);
        }
        let e = [
            vaddq_s32(g0, h0),
            vaddq_s32(g1, h1),
            vsubq_s32(g1, h1),
            vsubq_s32(g0, h0),
        ];
        let mut cc = [z; 8];
        let mut i = 0usize;
        while i < 4 {
            cc[i] = vaddq_s32(e[i], f[i]);
            i += 1;
        }
        while i < 8 {
            cc[i] = vsubq_s32(e[7 - i], f[7 - i]);
            i += 1;
        }
        let mut a = [z; 16];
        i = 0;
        while i < 8 {
            a[i] = vaddq_s32(cc[i], d[i]);
            i += 1;
        }
        while i < 16 {
            a[i] = vsubq_s32(cc[15 - i], d[15 - i]);
            i += 1;
        }
        let mut out = [z; 32];
        let mut k = 0usize;
        while k < 16 {
            out[k] = vaddq_s32(a[k], b[k]);
            out[k + 16] = vsubq_s32(a[15 - k], b[15 - k]);
            k += 1;
        }
        out
    }};
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct16_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct16_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "rdm")]
fn neon_dct16_i16x4_all_from_coeff4_rdm_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(base + 15 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct16_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct32_i16x4_all_from_coeff4_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct32_i16x4_all_body!()
}

#[inline]
#[target_feature(enable = "rdm")]
fn neon_dct32_i16x4_all_from_coeff4_rdm_stride_const<const IS_RECT2: bool, const STRIDE: usize>(
    coeff: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(base + 31 * STRIDE + 4 <= coeff.len());
    macro_rules! load {
        ($idx:expr) => {
            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, base + ($idx) * STRIDE)
        };
    }
    neon_dct32_i16x4_all_body!()
}

#[target_feature(enable = "neon")]
fn neon_dct16_i16x4_coeff_rows_to_scratch<const IS_RECT2: bool, const COEFF_STRIDE: usize>(
    coeff: &[i16],
    scratch: &mut [i16],
    mut y: usize,
    nrows: usize,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) -> usize {
    while y + 4 <= nrows {
        let out = neon_dct16_i16x4_all_from_coeff4_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 16;
        let mut m = 0usize;
        while m < 16 {
            neon_store4x4_i16_clip::<16>(
                scratch,
                row_base + m,
                out[m],
                out[m + 1],
                out[m + 2],
                out[m + 3],
                rnd,
                nsh,
                minv,
                maxv,
            );
            m += 4;
        }
        y += 4;
    }
    y
}

#[target_feature(enable = "neon")]
fn neon_dct32_i16x4_coeff_rows_to_scratch<const IS_RECT2: bool, const COEFF_STRIDE: usize>(
    coeff: &[i16],
    scratch: &mut [i16],
    mut y: usize,
    nrows: usize,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) -> usize {
    while y + 4 <= nrows {
        let out = neon_dct32_i16x4_all_from_coeff4_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 32;
        let mut m = 0usize;
        while m < 32 {
            neon_store4x4_i16_clip::<32>(
                scratch,
                row_base + m,
                out[m],
                out[m + 1],
                out[m + 2],
                out[m + 3],
                rnd,
                nsh,
                minv,
                maxv,
            );
            m += 4;
        }
        y += 4;
    }
    y
}

#[target_feature(enable = "rdm")]
fn neon_dct16_i16x4_coeff_rows_to_scratch_rdm<const IS_RECT2: bool, const COEFF_STRIDE: usize>(
    coeff: &[i16],
    scratch: &mut [i16],
    mut y: usize,
    nrows: usize,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) -> usize {
    while y + 4 <= nrows {
        let out =
            neon_dct16_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 16;
        let mut m = 0usize;
        while m < 16 {
            neon_store4x4_i16_clip::<16>(
                scratch,
                row_base + m,
                out[m],
                out[m + 1],
                out[m + 2],
                out[m + 3],
                rnd,
                nsh,
                minv,
                maxv,
            );
            m += 4;
        }
        y += 4;
    }
    y
}

#[target_feature(enable = "rdm")]
fn neon_dct32_i16x4_coeff_rows_to_scratch_rdm<const IS_RECT2: bool, const COEFF_STRIDE: usize>(
    coeff: &[i16],
    scratch: &mut [i16],
    mut y: usize,
    nrows: usize,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
) -> usize {
    while y + 4 <= nrows {
        let out =
            neon_dct32_i16x4_all_from_coeff4_rdm_stride_const::<IS_RECT2, COEFF_STRIDE>(coeff, y);
        let row_base = y * 32;
        let mut m = 0usize;
        while m < 32 {
            neon_store4x4_i16_clip::<32>(
                scratch,
                row_base + m,
                out[m],
                out[m + 1],
                out[m + 2],
                out[m + 3],
                rnd,
                nsh,
                minv,
                maxv,
            );
            m += 4;
        }
        y += 4;
    }
    y
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct16_i16x4_all_from_scratch4_stride_active<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
) -> [int32x4_t; 16] {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
    let zz = vdup_n_s16(0);
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                neon_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zz
            }
        };
    }
    neon_dct16_i16x4_all_body_active!()
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct32_i16x4_all_from_scratch4_stride_active<const STRIDE: usize, const ACTIVE: usize>(
    scratch: &[i16],
    base: usize,
) -> [int32x4_t; 32] {
    debug_assert!(ACTIVE == 4 || ACTIVE == 8 || ACTIVE == 16 || ACTIVE == 32);
    debug_assert!(base + (ACTIVE - 1) * STRIDE + 4 <= scratch.len());
    let zz = vdup_n_s16(0);
    macro_rules! load {
        ($idx:expr) => {
            if ($idx) < ACTIVE {
                neon_load4_i16_scratch(scratch, base + ($idx) * STRIDE)
            } else {
                zz
            }
        };
    }
    neon_dct32_i16x4_all_body_active!()
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct16_i16x4_all_from_scratch4_stride_eob<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
) -> [int32x4_t; 16] {
    if active <= 4 {
        neon_dct16_i16x4_all_from_scratch4_stride_active::<STRIDE, 4>(scratch, base)
    } else if active <= 8 {
        neon_dct16_i16x4_all_from_scratch4_stride_active::<STRIDE, 8>(scratch, base)
    } else {
        neon_dct16_i16x4_all_from_scratch4_stride_active::<STRIDE, 16>(scratch, base)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn neon_dct32_i16x4_all_from_scratch4_stride_eob<const STRIDE: usize>(
    scratch: &[i16],
    base: usize,
    active: usize,
) -> [int32x4_t; 32] {
    if active <= 4 {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 4>(scratch, base)
    } else if active <= 8 {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 8>(scratch, base)
    } else if active <= 16 {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 16>(scratch, base)
    } else {
        neon_dct32_i16x4_all_from_scratch4_stride_active::<STRIDE, 32>(scratch, base)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_dequant_dct_i16_neon_impl<const N: usize, const LEN: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_dct_i16_neon_impl_const::<N, LEN, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_dct_i16_neon_impl_const::<N, LEN, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_dequant_dct_i16_neon_impl_const<const N: usize, const LEN: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(N == 16 || N == 32);
    debug_assert!(LEN >= N * N);
    debug_assert!(coeff.len() >= N * N);
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < N / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let ncols = ngrp * 4;
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    with_neon_itx_i16_scratch(LEN, |scratch| unsafe {
        if N == 16 {
            let _ = neon_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, 16>(
                coeff, scratch, 0, ncols, rnd, nsh, minv, maxv,
            );
            let mut x = 0usize;
            while x < 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<16>(scratch, x, ncols);
                let mut m = 0usize;
                while m < 16 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
                x += 4;
            }
        } else {
            let _ = neon_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, 32>(
                coeff, scratch, 0, ncols, rnd, nsh, minv, maxv,
            );
            let mut x = 0usize;
            while x < 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<32>(scratch, x, ncols);
                let mut m = 0usize;
                while m < 32 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
                x += 4;
            }
        }
        crate::itx_2d::clear_i16_coeff_active_rows::<N>(coeff, ncols);
    });
}

#[inline]
#[target_feature(enable = "rdm")]
fn idct_dequant_dct_i16_neon_rdm_impl<const N: usize, const LEN: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_dct_i16_neon_rdm_impl_const::<N, LEN, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_dct_i16_neon_rdm_impl_const::<N, LEN, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn idct_dequant_dct_i16_neon_rdm_impl_const<
    const N: usize,
    const LEN: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    debug_assert!(N == 16 || N == 32);
    debug_assert!(LEN >= N * N);
    debug_assert!(coeff.len() >= N * N);
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < N / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let ncols = ngrp * 4;
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    with_neon_itx_i16_scratch(LEN, |scratch| unsafe {
        debug_assert!(N == 32);
        let _ = neon_dct32_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, 32>(
            coeff, scratch, 0, ncols, rnd, nsh, minv, maxv,
        );
        let mut x = 0usize;
        while x < 32 {
            let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<32>(scratch, x, ncols);
            let mut m = 0usize;
            while m < 32 {
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                m += 4;
            }
            x += 4;
        }
        crate::itx_2d::clear_i16_coeff_active_rows::<N>(coeff, ncols);
    });
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i32_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i32_impl_const::<N, W, H, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i32_impl_const::<N, W, H, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_4x4_neon_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    _eob: i32,
    _tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_4x4_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_4x4_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn tx_dequant_4x4_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 16);
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        macro_rules! load_col {
            ($j:expr) => {{
                let mut v = vld1q_s32(coeff.as_ptr().add(($j) * 4));
                if IS_RECT2 {
                    v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                }
                v
            }};
        }

        let c0 = load_col!(0);
        let c1 = load_col!(1);
        let c2 = load_col!(2);
        let c3 = load_col!(3);

        macro_rules! row_pass {
            ($m:expr) => {{
                let mut a = z;
                a = vmlaq_n_s32(a, c0, neon_tx_dense_coeff(first_kind, 4, $m, 0));
                a = vmlaq_n_s32(a, c1, neon_tx_dense_coeff(first_kind, 4, $m, 1));
                a = vmlaq_n_s32(a, c2, neon_tx_dense_coeff(first_kind, 4, $m, 2));
                a = vmlaq_n_s32(a, c3, neon_tx_dense_coeff(first_kind, 4, $m, 3));
                a
            }};
        }

        neon_store4x4_i32_clip(
            tmp,
            0,
            row_pass!(0),
            row_pass!(1),
            row_pass!(2),
            row_pass!(3),
            rnd,
            nsh,
            minv,
            maxv,
        );
        coeff[..16].fill(0);

        let r0 = vld1q_s32(tmp.as_ptr());
        let r1 = vld1q_s32(tmp.as_ptr().add(32));
        let r2 = vld1q_s32(tmp.as_ptr().add(64));
        let r3 = vld1q_s32(tmp.as_ptr().add(96));

        macro_rules! col_pass {
            ($m:expr) => {{
                let mut a = z;
                a = vmlaq_n_s32(a, r0, neon_tx_dense_coeff(second_kind, 4, $m, 0));
                a = vmlaq_n_s32(a, r1, neon_tx_dense_coeff(second_kind, 4, $m, 1));
                a = vmlaq_n_s32(a, r2, neon_tx_dense_coeff(second_kind, 4, $m, 2));
                a = vmlaq_n_s32(a, r3, neon_tx_dense_coeff(second_kind, 4, $m, 3));
                a
            }};
        }

        vst1q_s32(tmp.as_mut_ptr(), col_pass!(0));
        vst1q_s32(tmp.as_mut_ptr().add(32), col_pass!(1));
        vst1q_s32(tmp.as_mut_ptr().add(64), col_pass!(2));
        vst1q_s32(tmp.as_mut_ptr().add(96), col_pass!(3));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i32_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
        debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
        debug_assert!(W * H <= N && N <= coeff.len());
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < H / 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let nrows = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        let mut y = 0usize;
        while y + 4 <= nrows {
            let mut m = 0usize;
            while m < W {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < W {
                    let mut v = vld1q_s32(coeff.as_ptr().add(y + j * H));
                    if IS_RECT2 {
                        v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                    }
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(first_kind, W, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(first_kind, W, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(first_kind, W, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(first_kind, W, m + 3, j));
                    j += 1;
                }
                neon_store4x4_i32_clip(tmp, y * 32 + m, a0, a1, a2, a3, rnd, nsh, minv, maxv);
                m += 4;
            }
            y += 4;
        }
        while y < H {
            tmp[y * 32..y * 32 + W].fill(0);
            y += 1;
        }
        coeff[..W * H].fill(0);

        let mut x = 0usize;
        while x < W {
            let mut vin = [z; H];
            {
                let mut j = 0usize;
                while j < H {
                    vin[j] = vld1q_s32(tmp.as_ptr().add(x + j * 32));
                    j += 1;
                }
            }
            let mut m = 0usize;
            while m < H {
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < H {
                    let v = vin[j];
                    a0 = vmlaq_n_s32(a0, v, neon_tx_dense_coeff(second_kind, H, m, j));
                    a1 = vmlaq_n_s32(a1, v, neon_tx_dense_coeff(second_kind, H, m + 1, j));
                    a2 = vmlaq_n_s32(a2, v, neon_tx_dense_coeff(second_kind, H, m + 2, j));
                    a3 = vmlaq_n_s32(a3, v, neon_tx_dense_coeff(second_kind, H, m + 3, j));
                    j += 1;
                }
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a0);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), a1);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), a2);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), a3);
                m += 4;
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_neon_i16_impl_kind::<N, W, H>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_impl_kind<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_impl_const::<N, W, H, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i16_impl_const::<N, W, H, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    with_neon_itx_i16_scratch(N, |scratch| unsafe {
        let mut y = 0usize;

        if first_kind == crate::itx_2d::TX_KIND_IDENTITY {
            y = identity_pass::<W, H, IS_RECT2>(coeff, nrows, rnd, nsh, minv, maxv, scratch, y);
        }

        if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = neon_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = neon_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 =
                            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(first_kind, W, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(first_kind, W, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let mut x = 0usize;
        if second_kind == crate::itx_2d::TX_KIND_IDENTITY {
            let scale = neon_identity_scale(H);
            while x < W {
                let mut m = 0usize;
                while m < H {
                    let a = neon_identity_i16x4_scratch_to_i32(scratch, x + m * W, scale);
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a);
                    m += 1;
                }
                x += 4;
            }
        }
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(second_kind, H, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(second_kind, H, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a0);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), a1);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), a2);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), a3);
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    });
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn identity_pass<const W: usize, const H: usize, const IS_RECT2: bool>(
    coeff: &mut [i16],
    nrows: usize,
    rnd: int32x4_t,
    nsh: int32x4_t,
    minv: int32x4_t,
    maxv: int32x4_t,
    scratch: &mut [i16],
    y: usize,
) -> usize {
    let scale = neon_identity_scale(W);
    let mut y = y;
    while y + 4 <= nrows {
        let mut m = 0usize;
        while m < W {
            let a0 = neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 0) * H, scale);
            let a1 = neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 1) * H, scale);
            let a2 = neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 2) * H, scale);
            let a3 = neon_identity_i16x4_coeff_to_i32::<IS_RECT2>(coeff, y + (m + 3) * H, scale);
            neon_store4x4_i16_clip::<W>(scratch, y * W + m, a0, a1, a2, a3, rnd, nsh, minv, maxv);
            m += 4;
        }
        y += 4;
    }
    y
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn neon_identity_second_pass<const W: usize, const H: usize>(
    scratch: &[i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    rnd1: int32x4_t,
    nsh1: int32x4_t,
    mut x: usize,
) -> usize {
    let scale = neon_identity_scale(H);
    while x < W {
        let mut m = 0usize;
        while m < H {
            let a = neon_identity_i16x4_scratch_to_i32(scratch, x + m * W, scale);
            neon_writeback4_i32_u8::<W, H>(
                dst, dst_off, dst_stride, out_w, out_h, x, m, a, rnd1, nsh1,
            );
            m += 1;
        }
        x += 4;
    }
    x
}

#[inline]
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_neon_i16_rdm_impl_kind::<N, W, H>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

#[inline]
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_impl_kind<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_impl_const::<N, W, H, true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_impl_const::<N, W, H, false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    with_neon_itx_i16_scratch(N, |scratch| unsafe {
        let mut y = 0usize;
        if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = neon_dct16_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = neon_dct32_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 =
                            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 = neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(
                            coeff,
                            y + (j + 1) * H,
                        );
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(first_kind, W, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(first_kind, W, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let mut x = 0usize;
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), out[m]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), out[m + 1]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), out[m + 2]);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), out[m + 3]);
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(second_kind, H, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(second_kind, H, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), a0);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), a1);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), a2);
                    vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), a3);
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    });
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_8x8_neon_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_8x8_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_8x8_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_8x8_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        debug_assert!(coeff.len() >= 64);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 2 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 8 {
                let g = neon_tx8_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, first_kind, x);
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 8 {
            tmp[y * 32..y * 32 + 8].fill(0);
            y += 1;
        }
        coeff[..64].fill(0);
        let mut x = 0usize;
        while x < 8 {
            // Compute both output-row groups from the pristine row-pass result
            // before storing either (in-place aliasing: storing m=0 would
            // overwrite rows 0-3 that the m=4 group still needs to read).
            let g_lo = neon_tx8_i32x4_from_tmp4(tmp, x, second_kind, 0);
            let g_hi = neon_tx8_i32x4_from_tmp4(tmp, x, second_kind, 4);
            for (m, g) in [(0usize, &g_lo), (4usize, &g_hi)] {
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), g[0]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), g[1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), g[2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), g[3]);
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_dequant_16x16_neon_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_16x16_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_16x16_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn idct_dequant_16x16_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 256);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 4 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let z = vdupq_n_s32(0);
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);

        macro_rules! dct16x4_coeff {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let mut v = vld1q_s32(coeff.as_ptr().add($base + j * 16));
                    if IS_RECT2 {
                        v = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), v, 181));
                    }
                    a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]);
                    a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]);
                    a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]);
                    a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]);
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }
        macro_rules! dct16x4_tmp {
            ($base:expr, $m:expr) => {{
                let mut a0 = z;
                let mut a1 = z;
                let mut a2 = z;
                let mut a3 = z;
                let mut j = 0usize;
                while j < 16 {
                    let v = vld1q_s32(tmp.as_ptr().add($base + j * 32));
                    a0 = vmlaq_n_s32(a0, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m]);
                    a1 = vmlaq_n_s32(a1, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 1]);
                    a2 = vmlaq_n_s32(a2, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 2]);
                    a3 = vmlaq_n_s32(a3, v, crate::itx_2d::DCT16_DENSE_KERNEL[j * 16 + $m + 3]);
                    j += 1;
                }
                [a0, a1, a2, a3]
            }};
        }

        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 16 {
                let g = dct16x4_coeff!(y, x);
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 16 {
            tmp[y * 32..y * 32 + 16].fill(0);
            y += 1;
        }
        coeff[..256].fill(0);

        let mut x = 0usize;
        while x < 16 {
            let g0 = dct16x4_tmp!(x, 0);
            let g4 = dct16x4_tmp!(x, 4);
            let g8 = dct16x4_tmp!(x, 8);
            let g12 = dct16x4_tmp!(x, 12);
            for (m, g) in [(0usize, &g0), (4, &g4), (8, &g8), (12, &g12)] {
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), g[0]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 1) * 32), g[1]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 2) * 32), g[2]);
                vst1q_s32(tmp.as_mut_ptr().add(x + (m + 3) * 32), g[3]);
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
fn idct_dequant_32x32_neon_i32_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_32x32_neon_i32_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_32x32_neon_i32_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "neon")]
fn idct_dequant_32x32_neon_i32_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        debug_assert!(coeff.len() >= 1024);
        let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
        let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
        let mut ngrp = 0usize;
        while ngrp < 8 {
            ngrp += 1;
            if eob <= last_eob[ngrp - 1] as i32 {
                break;
            }
        }
        let ncols = ngrp * 4;
        let rnd = vdupq_n_s32((1 << shift0) >> 1);
        let nsh = vdupq_n_s32(-shift0);
        let minv = vdupq_n_s32(row_clip_min);
        let maxv = vdupq_n_s32(row_clip_max);
        let mut y = 0usize;
        while y + 4 <= ncols {
            let mut x = 0usize;
            while x < 32 {
                let g = neon_dct32_i32x4_from_coeff4_const::<IS_RECT2>(coeff, y, x);
                neon_store4x4_i32_clip(
                    tmp,
                    y * 32 + x,
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                    rnd,
                    nsh,
                    minv,
                    maxv,
                );
                x += 4;
            }
            y += 4;
        }
        while y < 32 {
            tmp[y * 32..y * 32 + 32].fill(0);
            y += 1;
        }
        coeff[..1024].fill(0);
        let mut x = 0usize;
        while x < 32 {
            let mut stage = core::mem::MaybeUninit::<[i32; 32 * 4]>::uninit();
            let stage_ptr = stage.as_mut_ptr().cast::<i32>();
            let mut m = 0usize;
            while m < 32 {
                let g = neon_dct32_i32x4_from_tmp4(tmp, x, m);
                vst1q_s32(stage_ptr.add(m * 4), g[0]);
                vst1q_s32(stage_ptr.add((m + 1) * 4), g[1]);
                vst1q_s32(stage_ptr.add((m + 2) * 4), g[2]);
                vst1q_s32(stage_ptr.add((m + 3) * 4), g[3]);
                m += 4;
            }
            let mut m = 0usize;
            while m < 32 {
                let row = vld1q_s32(stage_ptr.add(m * 4));
                vst1q_s32(tmp.as_mut_ptr().add(x + m * 32), row);
                m += 1;
            }
            x += 4;
        }
    }
}

macro_rules! idct_neon_fn {
    ($pub_fn:ident, $n:expr, $s:expr) => {
        pub(crate) fn $pub_fn(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $s }, { $s }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! idct_rect_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_rect_neon_fn {
    ($pub_fn:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub_fn(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            unsafe {
                tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    first_kind,
                    second_kind,
                )
            };
        }
    };
}

pub(crate) fn idct_dequant_4x4_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        tx_dequant_4x4_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            crate::itx_2d::TX_KIND_DCT,
            crate::itx_2d::TX_KIND_DCT,
        )
    }
}
#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_8x8_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_8x8_neon_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}

pub(crate) fn idct_dequant_16x16_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        idct_dequant_16x16_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}
pub(crate) fn idct_dequant_32x32_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        idct_dequant_32x32_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}
idct_neon_fn!(idct_dequant_64x64_neon, 1024, 32);

pub(crate) fn iadst_dequant_4x4_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        tx_dequant_4x4_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}
#[target_feature(enable = "neon")]
pub(crate) fn iadst_dequant_8x8_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_8x8_neon_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}
pub(crate) fn iadst_dequant_16x16_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    unsafe {
        iadst_dequant_16x16_neon_i32_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            first_kind,
            second_kind,
        )
    }
}
idct_rect_neon_fn!(idct_dequant_4x8_neon, 32, 4, 8);
idct_rect_neon_fn!(idct_dequant_8x4_neon, 32, 8, 4);
idct_rect_neon_fn!(idct_dequant_8x16_neon, 128, 8, 16);
idct_rect_neon_fn!(idct_dequant_16x8_neon, 128, 16, 8);
idct_rect_neon_fn!(idct_dequant_16x32_neon, 512, 16, 32);
idct_rect_neon_fn!(idct_dequant_32x16_neon, 512, 32, 16);
idct_rect_neon_fn!(idct_dequant_4x16_neon, 64, 4, 16);
idct_rect_neon_fn!(idct_dequant_16x4_neon, 64, 16, 4);
idct_rect_neon_fn!(idct_dequant_8x32_neon, 256, 8, 32);
idct_rect_neon_fn!(idct_dequant_32x8_neon, 256, 32, 8);
idct_rect_neon_fn!(idct_dequant_4x32_neon, 128, 4, 32);
idct_rect_neon_fn!(idct_dequant_32x4_neon, 128, 32, 4);
iadst_rect_neon_fn!(iadst_dequant_4x8_neon, 32, 4, 8);
iadst_rect_neon_fn!(iadst_dequant_8x4_neon, 32, 8, 4);
iadst_rect_neon_fn!(iadst_dequant_8x16_neon, 128, 8, 16);
iadst_rect_neon_fn!(iadst_dequant_16x8_neon, 128, 16, 8);
iadst_rect_neon_fn!(iadst_dequant_4x16_neon, 64, 4, 16);
iadst_rect_neon_fn!(iadst_dequant_16x4_neon, 64, 16, 4);

macro_rules! idct_rect_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub_name(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                $impl_name(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                crate::itx_2d::TX_KIND_DCT,
                crate::itx_2d::TX_KIND_DCT,
            )
        }
    };
}

macro_rules! iadst_rect_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub_name(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            unsafe {
                $impl_name(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    first_kind,
                    second_kind,
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
            coeff: &mut [i32],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            tx_dequant_dense_neon_i32_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                first_kind,
                second_kind,
            )
        }
    };
}

idct_rect_rdm_fn!(
    idct_dequant_4x8_neon_rdm,
    idct_dequant_4x8_neon_rdm_impl,
    32,
    4,
    8
);
idct_rect_rdm_fn!(
    idct_dequant_8x4_neon_rdm,
    idct_dequant_8x4_neon_rdm_impl,
    32,
    8,
    4
);
idct_rect_rdm_fn!(
    idct_dequant_8x16_neon_rdm,
    idct_dequant_8x16_neon_rdm_impl,
    128,
    8,
    16
);
idct_rect_rdm_fn!(
    idct_dequant_16x8_neon_rdm,
    idct_dequant_16x8_neon_rdm_impl,
    128,
    16,
    8
);
idct_rect_rdm_fn!(
    idct_dequant_16x32_neon_rdm,
    idct_dequant_16x32_neon_rdm_impl,
    512,
    16,
    32
);
idct_rect_rdm_fn!(
    idct_dequant_32x16_neon_rdm,
    idct_dequant_32x16_neon_rdm_impl,
    512,
    32,
    16
);
idct_rect_rdm_fn!(
    idct_dequant_4x16_neon_rdm,
    idct_dequant_4x16_neon_rdm_impl,
    64,
    4,
    16
);
idct_rect_rdm_fn!(
    idct_dequant_16x4_neon_rdm,
    idct_dequant_16x4_neon_rdm_impl,
    64,
    16,
    4
);
idct_rect_rdm_fn!(
    idct_dequant_8x32_neon_rdm,
    idct_dequant_8x32_neon_rdm_impl,
    256,
    8,
    32
);
idct_rect_rdm_fn!(
    idct_dequant_32x8_neon_rdm,
    idct_dequant_32x8_neon_rdm_impl,
    256,
    32,
    8
);
idct_rect_rdm_fn!(
    idct_dequant_4x32_neon_rdm,
    idct_dequant_4x32_neon_rdm_impl,
    128,
    4,
    32
);
idct_rect_rdm_fn!(
    idct_dequant_32x4_neon_rdm,
    idct_dequant_32x4_neon_rdm_impl,
    128,
    32,
    4
);

iadst_rect_rdm_fn!(
    iadst_dequant_4x8_neon_rdm,
    iadst_dequant_4x8_neon_rdm_impl,
    32,
    4,
    8
);
iadst_rect_rdm_fn!(
    iadst_dequant_8x4_neon_rdm,
    iadst_dequant_8x4_neon_rdm_impl,
    32,
    8,
    4
);
iadst_rect_rdm_fn!(
    iadst_dequant_8x16_neon_rdm,
    iadst_dequant_8x16_neon_rdm_impl,
    128,
    8,
    16
);
iadst_rect_rdm_fn!(
    iadst_dequant_16x8_neon_rdm,
    iadst_dequant_16x8_neon_rdm_impl,
    128,
    16,
    8
);
iadst_rect_rdm_fn!(
    iadst_dequant_4x16_neon_rdm,
    iadst_dequant_4x16_neon_rdm_impl,
    64,
    4,
    16
);
iadst_rect_rdm_fn!(
    iadst_dequant_16x4_neon_rdm,
    iadst_dequant_16x4_neon_rdm_impl,
    64,
    16,
    4
);

pub(crate) fn idct_dequant_32x32_neon_rdm(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    unsafe {
        idct_dequant_32x32_neon_rdm_impl(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}
#[target_feature(enable = "rdm")]
fn idct_dequant_32x32_neon_rdm_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    if is_rect2 {
        idct_dequant_32x32_neon_rdm_impl_const::<true>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    } else {
        idct_dequant_32x32_neon_rdm_impl_const::<false>(
            coeff,
            tmp,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
        )
    }
}

#[target_feature(enable = "rdm")]
fn idct_dequant_32x32_neon_rdm_impl_const<const IS_RECT2: bool>(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_neon_i32_impl(
        coeff,
        tmp,
        eob,
        tx,
        IS_RECT2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_fused_8bpc_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    if W == 4 && H == 4 {
        tx_dequant_dense_neon_i16_fused_4x4::<N, W, H, IS_RECT2>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        );
        return;
    }

    with_neon_itx_i16_scratch(N, |scratch| {
        let mut y = 0usize;

        if first_kind == crate::itx_2d::TX_KIND_IDENTITY {
            y = identity_pass::<W, H, IS_RECT2>(coeff, nrows, rnd, nsh, minv, maxv, scratch, y);
        }

        if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = neon_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = neon_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 =
                            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(first_kind, W, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(first_kind, W, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
        let nsh1 = vdupq_n_s32(-shift1);

        let mut x = 0usize;
        if second_kind == crate::itx_2d::TX_KIND_IDENTITY {
            x = neon_identity_second_pass::<W, H>(
                scratch, dst, dst_off, dst_stride, out_w, out_h, rnd1, nsh1, x,
            );
        }
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(second_kind, H, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(second_kind, H, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    });
}

// Hot fused path: only the curated square hot pairs use const kind
// generics. The broad fallback below remains runtime-kind SIMD.
#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_fused_8bpc_hot_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    with_neon_itx_i16_scratch(N, |scratch| {
        let mut y = 0usize;

        if FIRST_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            y = identity_pass::<W, H, IS_RECT2>(coeff, nrows, rnd, nsh, minv, maxv, scratch, y);
        }

        if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = neon_dct16_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = neon_dct32_i16x4_coeff_rows_to_scratch::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 =
                            neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, y + (j + 1) * H);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
        let nsh1 = vdupq_n_s32(-shift1);

        let mut x = 0usize;
        if SECOND_KIND == crate::itx_2d::TX_KIND_IDENTITY {
            x = neon_identity_second_pass::<W, H>(
                scratch, dst, dst_off, dst_stride, out_w, out_h, rnd1, nsh1, x,
            );
        }
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    });
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_fused_4x4<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
    let nsh1 = vdupq_n_s32(-shift1);
    let c0 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 0);
    let c1 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 4);
    let c2 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 8);
    let c3 = neon_load4_i16_coeff_packed_const::<IS_RECT2>(coeff, 12);
    macro_rules! rrow {
        ($m:expr) => {{
            let mut a = vmlal_n_s16(z, c0, neon_tx_dense_coeff_i16(first_kind, 4, $m, 0));
            a = vmlal_n_s16(a, c1, neon_tx_dense_coeff_i16(first_kind, 4, $m, 1));
            a = vmlal_n_s16(a, c2, neon_tx_dense_coeff_i16(first_kind, 4, $m, 2));
            a = vmlal_n_s16(a, c3, neon_tx_dense_coeff_i16(first_kind, 4, $m, 3));
            vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(a, rnd), nsh), minv), maxv)
        }};
    }
    let cc0 = rrow!(0);
    let cc1 = rrow!(1);
    let cc2 = rrow!(2);
    let cc3 = rrow!(3);
    // Transpose (output-cols -> rows), identical to neon_store4x4_i16_clip.
    let t01 = vtrnq_s32(cc0, cc1);
    let t23 = vtrnq_s32(cc2, cc3);
    let r0 = vqmovn_s32(vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0)));
    let r1 = vqmovn_s32(vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1)));
    let r2 = vqmovn_s32(vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0)));
    let r3 = vqmovn_s32(vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1)));
    macro_rules! rcol {
        ($m:expr) => {{
            let mut b = vmlal_n_s16(z, r0, neon_tx_dense_coeff_i16(second_kind, 4, $m, 0));
            b = vmlal_n_s16(b, r1, neon_tx_dense_coeff_i16(second_kind, 4, $m, 1));
            b = vmlal_n_s16(b, r2, neon_tx_dense_coeff_i16(second_kind, 4, $m, 2));
            b = vmlal_n_s16(b, r3, neon_tx_dense_coeff_i16(second_kind, 4, $m, 3));
            b
        }};
    }
    neon_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        0,
        rcol!(0),
        rnd1,
        nsh1,
    );
    neon_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        1,
        rcol!(1),
        rnd1,
        nsh1,
    );
    neon_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        2,
        rcol!(2),
        rnd1,
        nsh1,
    );
    neon_writeback4_i32_u8::<4, 4>(
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        0,
        3,
        rcol!(3),
        rnd1,
        nsh1,
    );
    coeff[..W * H].fill(0);
}

#[inline(never)]
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    if W == 4 && H == 4 {
        tx_dequant_dense_neon_i16_fused_4x4::<N, W, H, IS_RECT2>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        );
        return;
    }

    with_neon_itx_i16_scratch(N, |scratch| {
        let mut y = 0usize;
        if first_kind == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = neon_dct16_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        } else if first_kind == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = neon_dct32_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 =
                            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 = neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(
                            coeff,
                            y + (j + 1) * H,
                        );
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(first_kind, W, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(first_kind, W, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(first_kind, W, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(first_kind, W, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
        let nsh1 = vdupq_n_s32(-shift1);

        let mut x = 0usize;
        while x < W {
            if second_kind == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else if second_kind == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(a0, x0, neon_tx_dense_coeff_i16(second_kind, H, m, j));
                        a0 = vmlal_n_s16(a0, x1, neon_tx_dense_coeff_i16(second_kind, H, m, j + 1));
                        a1 = vmlal_n_s16(a1, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 1, j));
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(a2, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 2, j));
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(a3, x0, neon_tx_dense_coeff_i16(second_kind, H, m + 3, j));
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16(second_kind, H, m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    });
}

// Hot fused path: only the curated square hot pairs use const kind
// generics. The broad fallback below remains runtime-kind SIMD.
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_fused_8bpc_hot_impl_const<
    const N: usize,
    const W: usize,
    const H: usize,
    const IS_RECT2: bool,
    const FIRST_KIND: usize,
    const SECOND_KIND: usize,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    debug_assert!(W == 4 || W == 8 || W == 16 || W == 32);
    debug_assert!(H == 4 || H == 8 || H == 16 || H == 32);
    debug_assert!(W * H <= N && N <= coeff.len());
    let off = usize::from(crate::scan::LAST_EOB_PER_COL.offset[tx]);
    let last_eob = &crate::scan::LAST_EOB_PER_COL.table[off..];
    let mut ngrp = 0usize;
    while ngrp < H / 4 {
        ngrp += 1;
        if eob <= last_eob[ngrp - 1] as i32 {
            break;
        }
    }
    let nrows = ngrp * 4;
    let z = vdupq_n_s32(0);
    let rnd = vdupq_n_s32((1 << shift0) >> 1);
    let nsh = vdupq_n_s32(-shift0);
    let minv = vdupq_n_s32(row_clip_min);
    let maxv = vdupq_n_s32(row_clip_max);

    with_neon_itx_i16_scratch(N, |scratch| {
        let mut y = 0usize;
        if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 16 {
            y = neon_dct16_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        } else if FIRST_KIND == crate::itx_2d::TX_KIND_DCT && W == 32 {
            y = neon_dct32_i16x4_coeff_rows_to_scratch_rdm::<IS_RECT2, H>(
                coeff, scratch, y, nrows, rnd, nsh, minv, maxv,
            );
        }
        while y + 4 <= nrows {
            {
                let mut m = 0usize;
                while m < W {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < W {
                        let x0 =
                            neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(coeff, y + j * H);
                        let x1 = neon_load4_i16_coeff_packed_rdm_const::<IS_RECT2>(
                            coeff,
                            y + (j + 1) * H,
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<FIRST_KIND, W>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_store4x4_i16_clip::<W>(
                        scratch,
                        y * W + m,
                        a0,
                        a1,
                        a2,
                        a3,
                        rnd,
                        nsh,
                        minv,
                        maxv,
                    );
                    m += 4;
                }
            }
            y += 4;
        }
        let rnd1 = vdupq_n_s32((1 << shift1) >> 1);
        let nsh1 = vdupq_n_s32(-shift1);

        let mut x = 0usize;
        while x < W {
            if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 16 {
                let out = neon_dct16_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 16 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else if SECOND_KIND == crate::itx_2d::TX_KIND_DCT && H == 32 {
                let out = neon_dct32_i16x4_all_from_scratch4_stride_eob::<W>(scratch, x, nrows);
                let mut m = 0usize;
                while m < 32 {
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, out[m], rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        out[m + 1],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        out[m + 2],
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        out[m + 3],
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            } else {
                let mut m = 0usize;
                while m < H {
                    let mut a0 = z;
                    let mut a1 = z;
                    let mut a2 = z;
                    let mut a3 = z;
                    let mut j = 0usize;
                    while j < H {
                        let x0 = neon_load4_i16_scratch(scratch, x + j * W);
                        let x1 = neon_load4_i16_scratch(scratch, x + (j + 1) * W);
                        a0 = vmlal_n_s16(
                            a0,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j),
                        );
                        a0 = vmlal_n_s16(
                            a0,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m, j + 1),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j),
                        );
                        a1 = vmlal_n_s16(
                            a1,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 1, j + 1),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j),
                        );
                        a2 = vmlal_n_s16(
                            a2,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 2, j + 1),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x0,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j),
                        );
                        a3 = vmlal_n_s16(
                            a3,
                            x1,
                            neon_tx_dense_coeff_i16_const::<SECOND_KIND, H>(m + 3, j + 1),
                        );
                        j += 2;
                    }
                    neon_writeback4_i32_u8::<W, H>(
                        dst, dst_off, dst_stride, out_w, out_h, x, m, a0, rnd1, nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 1,
                        a1,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 2,
                        a2,
                        rnd1,
                        nsh1,
                    );
                    neon_writeback4_i32_u8::<W, H>(
                        dst,
                        dst_off,
                        dst_stride,
                        out_w,
                        out_h,
                        x,
                        m + 3,
                        a3,
                        rnd1,
                        nsh1,
                    );
                    m += 4;
                }
            }
            x += 4;
        }
        coeff[..W * H].fill(0);
    });
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_fused_4x4_impl(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_fused_4x4::<16, 4, 4, true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i16_fused_4x4::<16, 4, 4, false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    }
}

#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_fused_hot_square<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    debug_assert_eq!(W, H);
    macro_rules! call_pair {
        ($first:expr, $second:expr) => {{
            tx_dequant_dense_neon_i16_fused_8bpc_hot_impl_const::<
                N,
                W,
                H,
                false,
                { $first },
                { $second },
            >(
                coeff,
                dst,
                dst_off,
                dst_stride,
                out_w,
                out_h,
                eob,
                tx,
                shift0,
                row_clip_min,
                row_clip_max,
                shift1,
            );
            true
        }};
    }

    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => false,
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_fused_hot_square<
    const N: usize,
    const W: usize,
    const H: usize,
>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    debug_assert_eq!(W, H);
    macro_rules! call_pair {
        ($first:expr, $second:expr) => {{
            tx_dequant_dense_neon_i16_rdm_fused_8bpc_hot_impl_const::<
                N,
                W,
                H,
                false,
                { $first },
                { $second },
            >(
                coeff,
                dst,
                dst_off,
                dst_stride,
                out_w,
                out_h,
                eob,
                tx,
                shift0,
                row_clip_min,
                row_clip_max,
                shift1,
            );
            true
        }};
    }

    match (first_kind, second_kind) {
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(crate::itx_2d::TX_KIND_DCT, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT) => {
            call_pair!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_DCT)
        }
        (crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(crate::itx_2d::TX_KIND_ADST, crate::itx_2d::TX_KIND_FLIPADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST) => {
            call_pair!(crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_ADST)
        }
        (crate::itx_2d::TX_KIND_FLIPADST, crate::itx_2d::TX_KIND_FLIPADST) => {
            call_pair!(
                crate::itx_2d::TX_KIND_FLIPADST,
                crate::itx_2d::TX_KIND_FLIPADST
            )
        }
        _ => false,
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn tx_dequant_dense_neon_i16_fused_8bpc_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<N, W, H, true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i16_fused_8bpc_impl_const::<N, W, H, false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl<const N: usize, const W: usize, const H: usize>(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<N, W, H, true>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl_const::<N, W, H, false>(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        )
    }
}

macro_rules! neon_fused_match_body {
    ($call:ident, $coeff:ident, $dst:ident, $dst_off:ident, $dst_stride:ident, $out_w:ident, $out_h:ident, $eob:ident, $tx:ident, $is_rect2:ident, $shift0:ident, $row_clip_min:ident, $row_clip_max:ident, $shift1:ident, $first_kind:ident, $second_kind:ident) => {{
        match $tx {
            crate::levels::txsz::TX_8X8 => $call::<64, 8, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_16X16 => $call::<256, 16, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_32X32 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::TX_64X64 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X8 => $call::<32, 4, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X4 => $call::<32, 8, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X16 => $call::<128, 8, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X8 => $call::<128, 16, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X32 => $call::<512, 16, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X16 => $call::<512, 32, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X64 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X32 => $call::<1024, 32, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X16 => $call::<64, 4, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X4 => $call::<64, 16, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X32 => $call::<256, 8, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X8 => $call::<256, 32, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_16X64 => $call::<512, 16, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X16 => $call::<512, 32, 16>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X32 => $call::<128, 4, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_32X4 => $call::<128, 32, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_8X64 => $call::<256, 8, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X8 => $call::<256, 32, 8>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_4X64 => $call::<128, 4, 32>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            crate::levels::txsz::RTX_64X4 => $call::<128, 32, 4>(
                $coeff,
                $dst,
                $dst_off,
                $dst_stride,
                $out_w,
                $out_h,
                $eob,
                $tx,
                $is_rect2,
                $shift0,
                $row_clip_min,
                $row_clip_max,
                $shift1,
                $first_kind,
                $second_kind,
            ),
            _ => return false,
        }
        true
    }};
}

// Keep the very large 32-point i16 DCT bodies in fixed call targets. The
// dispatch wrappers stay tiny, while the heavy transform graph is isolated
// behind one call. Rect2 remains the only boolean specialization inside the
// fixed 32-point island.
#[target_feature(enable = "neon")]
fn idct_dequant_32x32_i16_neon_fixed_impl(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_impl::<32, 1024>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "rdm")]
fn idct_dequant_32x32_i16_neon_rdm_fixed_impl(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_rdm_impl::<32, 1024>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "neon")]
fn idct_dequant_32x32_i16_neon_fixed_fused_8bpc_impl(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_fused_8bpc_hot_impl_const::<
            1024,
            32,
            32,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_fused_8bpc_hot_impl_const::<
            1024,
            32,
            32,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "rdm")]
fn idct_dequant_32x32_i16_neon_rdm_fixed_fused_8bpc_impl(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_hot_impl_const::<
            1024,
            32,
            32,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_hot_impl_const::<
            1024,
            32,
            32,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            32,
            32,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_16x16_i16_neon_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_fused_8bpc_hot_impl_const::<
            256,
            16,
            16,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_fused_8bpc_hot_impl_const::<
            256,
            16,
            16,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_32x32_i16_neon_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    idct_dequant_32x32_i16_neon_fixed_fused_8bpc_impl(
        coeff,
        dst,
        dst_off,
        dst_stride,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
    )
}

#[target_feature(enable = "rdm")]
pub(crate) fn idct_dequant_16x16_i16_neon_rdm_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    if is_rect2 {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_hot_impl_const::<
            256,
            16,
            16,
            true,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    } else {
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_hot_impl_const::<
            256,
            16,
            16,
            false,
            { crate::itx_2d::TX_KIND_DCT },
            { crate::itx_2d::TX_KIND_DCT },
        >(
            coeff,
            dst,
            dst_off,
            dst_stride,
            16,
            16,
            eob,
            tx,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
        )
    }
}

#[target_feature(enable = "rdm")]
pub(crate) fn idct_dequant_32x32_i16_neon_rdm_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
) {
    idct_dequant_32x32_i16_neon_rdm_fixed_fused_8bpc_impl(
        coeff,
        dst,
        dst_off,
        dst_stride,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn itx_dequant_i16_neon_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    if !crate::itx_2d::is_itx_dense_kind(first_kind)
        || !crate::itx_2d::is_itx_dense_kind(second_kind)
    {
        return false;
    }
    if tx == crate::levels::txsz::TX_4X4 {
        tx_dequant_dense_neon_i16_fused_4x4_impl(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        );
        return true;
    }

    // Keep the very common square 8/16 DCT/ADST/FLIPADST pairs const-kind.
    // Other shapes/kinds still use the runtime-kind SIMD body below.
    if !is_rect2 {
        let handled_hot = match tx {
            crate::levels::txsz::TX_8X8 => tx_dequant_dense_neon_i16_fused_hot_square::<64, 8, 8>(
                coeff,
                dst,
                dst_off,
                dst_stride,
                out_w,
                out_h,
                eob,
                tx,
                shift0,
                row_clip_min,
                row_clip_max,
                shift1,
                first_kind,
                second_kind,
            ),
            crate::levels::txsz::TX_16X16 => {
                tx_dequant_dense_neon_i16_fused_hot_square::<256, 16, 16>(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                    first_kind,
                    second_kind,
                )
            }
            _ => false,
        };
        if handled_hot {
            return true;
        }
    }

    neon_fused_match_body!(
        tx_dequant_dense_neon_i16_fused_8bpc_impl,
        coeff,
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
        first_kind,
        second_kind
    )
}

#[target_feature(enable = "rdm")]
pub(crate) fn itx_dequant_i16_neon_rdm_fused_8bpc(
    coeff: &mut [i16],
    dst: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    out_w: usize,
    out_h: usize,
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    shift1: i32,
    first_kind: usize,
    second_kind: usize,
) -> bool {
    if !crate::itx_2d::is_itx_dense_kind(first_kind)
        || !crate::itx_2d::is_itx_dense_kind(second_kind)
    {
        return false;
    }
    if tx == crate::levels::txsz::TX_4X4 {
        tx_dequant_dense_neon_i16_fused_4x4_impl(
            coeff,
            dst,
            dst_off,
            dst_stride,
            out_w,
            out_h,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
            shift1,
            first_kind,
            second_kind,
        );
        return true;
    }

    // Keep the very common square 8/16 DCT/ADST/FLIPADST pairs const-kind.
    // Other shapes/kinds still use the runtime-kind SIMD body below.
    if !is_rect2 {
        let handled_hot = match tx {
            crate::levels::txsz::TX_8X8 => {
                tx_dequant_dense_neon_i16_rdm_fused_hot_square::<64, 8, 8>(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                    first_kind,
                    second_kind,
                )
            }
            crate::levels::txsz::TX_16X16 => {
                tx_dequant_dense_neon_i16_rdm_fused_hot_square::<256, 16, 16>(
                    coeff,
                    dst,
                    dst_off,
                    dst_stride,
                    out_w,
                    out_h,
                    eob,
                    tx,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    shift1,
                    first_kind,
                    second_kind,
                )
            }
            _ => false,
        };
        if handled_hot {
            return true;
        }
    }

    neon_fused_match_body!(
        tx_dequant_dense_neon_i16_rdm_fused_8bpc_impl,
        coeff,
        dst,
        dst_off,
        dst_stride,
        out_w,
        out_h,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        shift1,
        first_kind,
        second_kind
    )
}

macro_rules! idct_i16_neon_fn {
    ($pub:ident, $n:expr, $s:expr) => {
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $s }, { $s }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_i16_neon_fn {
    ($pub:ident, $n:expr, $s:expr) => {
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $s }, { $s }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    first_kind,
                    second_kind,
                )
            };
        }
    };
}

macro_rules! idct_rect_i16_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $w }, { $h }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    crate::itx_2d::TX_KIND_DCT,
                    crate::itx_2d::TX_KIND_DCT,
                )
            };
        }
    };
}

macro_rules! iadst_rect_i16_neon_fn {
    ($pub:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            unsafe {
                tx_dequant_dense_neon_i16_impl::<{ $n }, { $w }, { $h }>(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    first_kind,
                    second_kind,
                )
            };
        }
    };
}

macro_rules! idct_rect_i16_neon_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub_name(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            unsafe {
                $impl_name(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
        ) {
            tx_dequant_dense_neon_i16_rdm_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                crate::itx_2d::TX_KIND_DCT,
                crate::itx_2d::TX_KIND_DCT,
            )
        }
    };
}

macro_rules! iadst_rect_i16_neon_rdm_fn {
    ($pub_name:ident, $impl_name:ident, $n:expr, $w:expr, $h:expr) => {
        pub(crate) fn $pub_name(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            unsafe {
                $impl_name(
                    coeff,
                    tmp,
                    eob,
                    tx,
                    is_rect2,
                    shift0,
                    row_clip_min,
                    row_clip_max,
                    first_kind,
                    second_kind,
                )
            }
        }
        #[target_feature(enable = "rdm")]
        #[inline]
        fn $impl_name(
            coeff: &mut [i16],
            tmp: &mut [i32; ITX_TMP_PIXELS],
            eob: i32,
            tx: usize,
            is_rect2: bool,
            shift0: i32,
            row_clip_min: i32,
            row_clip_max: i32,
            first_kind: usize,
            second_kind: usize,
        ) {
            tx_dequant_dense_neon_i16_rdm_impl::<{ $n }, { $w }, { $h }>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
                first_kind,
                second_kind,
            )
        }
    };
}

idct_i16_neon_fn!(idct_dequant_4x4_i16_neon, 16, 4);

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_8x8_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    tx_dequant_dense_neon_i16_impl::<64, 8, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        crate::itx_2d::TX_KIND_DCT,
        crate::itx_2d::TX_KIND_DCT,
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_16x16_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_impl::<16, 256>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_32x32_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_i16_neon_fixed_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "rdm")]
pub(crate) fn idct_dequant_32x32_i16_neon_rdm(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_32x32_i16_neon_rdm_fixed_impl(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_64x64_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    idct_dequant_dct_i16_neon_impl::<32, 1024>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    )
}
iadst_i16_neon_fn!(iadst_dequant_4x4_i16_neon, 16, 4);

#[target_feature(enable = "neon")]
pub(crate) fn iadst_dequant_8x8_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_neon_i16_impl::<64, 8, 8>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

#[target_feature(enable = "neon")]
pub(crate) fn iadst_dequant_16x16_i16_neon(
    coeff: &mut [i16],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
    first_kind: usize,
    second_kind: usize,
) {
    tx_dequant_dense_neon_i16_impl::<256, 16, 16>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
        first_kind,
        second_kind,
    )
}

idct_rect_i16_neon_fn!(idct_dequant_4x8_i16_neon, 32, 4, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x8_i16_neon_rdm,
    idct_dequant_4x8_i16_neon_rdm_impl,
    32,
    4,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_8x4_i16_neon, 32, 8, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x4_i16_neon_rdm,
    idct_dequant_8x4_i16_neon_rdm_impl,
    32,
    8,
    4
);
idct_rect_i16_neon_fn!(idct_dequant_8x16_i16_neon, 128, 8, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x16_i16_neon_rdm,
    idct_dequant_8x16_i16_neon_rdm_impl,
    128,
    8,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_16x8_i16_neon, 128, 16, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x8_i16_neon_rdm,
    idct_dequant_16x8_i16_neon_rdm_impl,
    128,
    16,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_16x32_i16_neon, 512, 16, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x32_i16_neon_rdm,
    idct_dequant_16x32_i16_neon_rdm_impl,
    512,
    16,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x16_i16_neon, 512, 32, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x16_i16_neon_rdm,
    idct_dequant_32x16_i16_neon_rdm_impl,
    512,
    32,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_4x16_i16_neon, 64, 4, 16);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x16_i16_neon_rdm,
    idct_dequant_4x16_i16_neon_rdm_impl,
    64,
    4,
    16
);
idct_rect_i16_neon_fn!(idct_dequant_16x4_i16_neon, 64, 16, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_16x4_i16_neon_rdm,
    idct_dequant_16x4_i16_neon_rdm_impl,
    64,
    16,
    4
);
idct_rect_i16_neon_fn!(idct_dequant_8x32_i16_neon, 256, 8, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_8x32_i16_neon_rdm,
    idct_dequant_8x32_i16_neon_rdm_impl,
    256,
    8,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x8_i16_neon, 256, 32, 8);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x8_i16_neon_rdm,
    idct_dequant_32x8_i16_neon_rdm_impl,
    256,
    32,
    8
);
idct_rect_i16_neon_fn!(idct_dequant_4x32_i16_neon, 128, 4, 32);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_4x32_i16_neon_rdm,
    idct_dequant_4x32_i16_neon_rdm_impl,
    128,
    4,
    32
);
idct_rect_i16_neon_fn!(idct_dequant_32x4_i16_neon, 128, 32, 4);
idct_rect_i16_neon_rdm_fn!(
    idct_dequant_32x4_i16_neon_rdm,
    idct_dequant_32x4_i16_neon_rdm_impl,
    128,
    32,
    4
);
iadst_rect_i16_neon_fn!(iadst_dequant_4x8_i16_neon, 32, 4, 8);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_4x8_i16_neon_rdm,
    iadst_dequant_4x8_i16_neon_rdm_impl,
    32,
    4,
    8
);
iadst_rect_i16_neon_fn!(iadst_dequant_8x4_i16_neon, 32, 8, 4);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_8x4_i16_neon_rdm,
    iadst_dequant_8x4_i16_neon_rdm_impl,
    32,
    8,
    4
);
iadst_rect_i16_neon_fn!(iadst_dequant_8x16_i16_neon, 128, 8, 16);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_8x16_i16_neon_rdm,
    iadst_dequant_8x16_i16_neon_rdm_impl,
    128,
    8,
    16
);
iadst_rect_i16_neon_fn!(iadst_dequant_16x8_i16_neon, 128, 16, 8);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_16x8_i16_neon_rdm,
    iadst_dequant_16x8_i16_neon_rdm_impl,
    128,
    16,
    8
);
iadst_rect_i16_neon_fn!(iadst_dequant_4x16_i16_neon, 64, 4, 16);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_4x16_i16_neon_rdm,
    iadst_dequant_4x16_i16_neon_rdm_impl,
    64,
    4,
    16
);
iadst_rect_i16_neon_fn!(iadst_dequant_16x4_i16_neon, 64, 16, 4);
iadst_rect_i16_neon_rdm_fn!(
    iadst_dequant_16x4_i16_neon_rdm,
    iadst_dequant_16x4_i16_neon_rdm_impl,
    64,
    16,
    4
);
