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

use crate::itx_1d::DctWide;
use crate::itx_2d::{
    Adst2dBackend, Dct2dBackend, DctSimd4, ITX_TMP_PIXELS, idct_dequant_simd4_core,
    itx_dequant_simd4_core,
};
use std::arch::aarch64::*;

#[derive(Clone, Copy)]
pub(crate) struct NeonI32x4(int32x4_t);

impl crate::itx_1d::DctLane for NeonI32x4 {
    #[inline(always)]
    fn zero() -> Self {
        NeonI32x4(unsafe { vdupq_n_s32(0) })
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        NeonI32x4(unsafe { vaddq_s32(self.0, o.0) })
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        NeonI32x4(unsafe { vsubq_s32(self.0, o.0) })
    }
    #[inline(always)]
    fn mul(self, k: Self) -> Self {
        NeonI32x4(unsafe { vmulq_s32(self.0, k.0) })
    }
    #[inline(always)]
    fn dup_load(table: &[i32], idx: usize) -> Self {
        // SAFETY: callers index within the kernel tables.
        NeonI32x4(unsafe { vld1q_dup_s32(table.as_ptr().add(idx)) })
    }
    type Coeffs = int32x4_t;
    #[inline(always)]
    fn load_coeffs(table: &[i32], idx: usize) -> int32x4_t {
        // SAFETY: callers index a 4-wide group within the kernel tables.
        unsafe { vld1q_s32(table.as_ptr().add(idx)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(self, x: Self, c: int32x4_t) -> Self {
        // self + x * c[LANE] in one fused by-lane MLA; no per-coefficient load.
        NeonI32x4(unsafe { vmlaq_laneq_s32::<LANE>(self.0, x.0, c) })
    }
    #[inline(always)]
    fn mul_add(self, x: Self, k: Self) -> Self {
        NeonI32x4(unsafe { vmlaq_s32(self.0, x.0, k.0) })
    }
}

pub(crate) struct NeonWide;

impl crate::itx_1d::DctWide for NeonWide {
    type In = int16x8_t;
    type Acc = (int32x4_t, int32x4_t);
    type Coeffs = int16x8_t;
    type Clip = (int32x4_t, int32x4_t, int32x4_t, int32x4_t);
    #[inline(always)]
    fn zero() -> Self::Acc {
        unsafe { (vdupq_n_s32(0), vdupq_n_s32(0)) }
    }
    #[inline(always)]
    fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { (vaddq_s32(a.0, b.0), vaddq_s32(a.1, b.1)) }
    }
    #[inline(always)]
    fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        unsafe { (vsubq_s32(a.0, b.0), vsubq_s32(a.1, b.1)) }
    }
    #[inline(always)]
    fn load_coeffs(table: &[i16], idx: usize) -> int16x8_t {
        unsafe { vld1q_s16(table.as_ptr().add(idx)) }
    }
    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(acc: Self::Acc, x: int16x8_t, c: int16x8_t) -> Self::Acc {
        unsafe {
            (
                vmlal_laneq_s16::<LANE>(acc.0, vget_low_s16(x), c),
                vmlal_high_laneq_s16::<LANE>(acc.1, x, c),
            )
        }
    }
    #[inline(always)]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            let hi = vld1q_s32(src.as_ptr().add(off + 4));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[inline(always)]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            // Exact NEON fallback for CPUs without FEAT_RDM: keep the rect2
            // normalization in i32, then narrow exactly like `load8_narrow`.
            let lo = vld1q_s32(src.as_ptr().add(off));
            let hi = vld1q_s32(src.as_ptr().add(off + 4));
            let r = vdupq_n_s32(128);
            let lo = vshrq_n_s32::<8>(vmlaq_n_s32(r, lo, 181));
            let hi = vshrq_n_s32::<8>(vmlaq_n_s32(r, hi, 181));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[inline(always)]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline(always)]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> int16x8_t {
        unsafe {
            let lo = vld1q_s32(src.as_ptr().add(off));
            let lo = vshrq_n_s32::<8>(vmlaq_n_s32(vdupq_n_s32(128), lo, 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline(always)]
    unsafe fn load8_i16(src: &[i16], off: usize) -> int16x8_t {
        debug_assert!(off + 8 <= src.len());
        unsafe { vld1q_s16(src.as_ptr().add(off)) }
    }
    #[inline(always)]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> int16x8_t {
        unsafe {
            let x = Self::load8_i16(src, off);
            let r = vdupq_n_s32(128);
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(r, vget_low_s16(x), 181));
            let hi = vshrq_n_s32::<8>(vmlal_high_n_s16(r, x, 181));
            vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
        }
    }
    #[inline(always)]
    unsafe fn load4_i16(src: &[i16], off: usize) -> int16x8_t {
        debug_assert!(off + 4 <= src.len());
        unsafe { vcombine_s16(vld1_s16(src.as_ptr().add(off)), vdup_n_s16(0)) }
    }
    #[inline(always)]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> int16x8_t {
        unsafe {
            let x = Self::load4_i16(src, off);
            let lo = vshrq_n_s32::<8>(vmlal_n_s16(vdupq_n_s32(128), vget_low_s16(x), 181));
            vcombine_s16(vmovn_s32(lo), vdup_n_s16(0))
        }
    }
    #[inline(always)]
    fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        unsafe {
            (
                vdupq_n_s32(rnd),
                vdupq_n_s32(-shift),
                vdupq_n_s32(min),
                vdupq_n_s32(max),
            )
        }
    }
    #[inline(always)]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, nsh, minv, maxv) = clip;
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(acc.0, rnd), nsh), minv), maxv);
            let hi = vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(acc.1, rnd), nsh), minv), maxv);
            let p = dst.as_mut_ptr().add(off);
            vst1q_lane_s32::<0>(p.add(0 * stride), lo);
            vst1q_lane_s32::<1>(p.add(1 * stride), lo);
            vst1q_lane_s32::<2>(p.add(2 * stride), lo);
            vst1q_lane_s32::<3>(p.add(3 * stride), lo);
            vst1q_lane_s32::<0>(p.add(4 * stride), hi);
            vst1q_lane_s32::<1>(p.add(5 * stride), hi);
            vst1q_lane_s32::<2>(p.add(6 * stride), hi);
            vst1q_lane_s32::<3>(p.add(7 * stride), hi);
        }
    }
    #[inline(always)]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe {
            let (rnd, nsh, minv, maxv) = clip;
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(acc.0, rnd), nsh), minv), maxv);
            let p = dst.as_mut_ptr().add(off);
            vst1q_lane_s32::<0>(p.add(0 * stride), lo);
            vst1q_lane_s32::<1>(p.add(1 * stride), lo);
            vst1q_lane_s32::<2>(p.add(2 * stride), lo);
            vst1q_lane_s32::<3>(p.add(3 * stride), lo);
        }
    }
    #[inline(always)]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            vst1q_s32(dst.as_mut_ptr().add(off), acc.0);
            vst1q_s32(dst.as_mut_ptr().add(off + 4), acc.1);
        }
    }
    #[inline(always)]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe {
            vst1q_s32(dst.as_mut_ptr().add(off), acc.0);
        }
    }
}

#[target_feature(enable = "rdm")]
unsafe fn load8_rect2_narrow_rdm(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        // dav2d-style rect2 normalization. SQRDMULH by 0x5a80 is exactly
        // `(v * 181 + 128) >> 8` for valid s16 lanes, and avoids widening the
        // row-pipeline input twice.
        let lo = vld1q_s32(src.as_ptr().add(off));
        let hi = vld1q_s32(src.as_ptr().add(off + 4));
        let v = vcombine_s16(vmovn_s32(lo), vmovn_s32(hi));
        vqrdmulhq_s16(v, vdupq_n_s16(0x5a80))
    }
}

#[target_feature(enable = "rdm")]
unsafe fn load4_rect2_narrow_rdm(src: &[i32], off: usize) -> int16x8_t {
    unsafe {
        // Same RDM rect2 normalization for 4 active lanes; high lanes stay zero.
        vqrdmulhq_s16(NeonWide::load4_narrow(src, off), vdupq_n_s16(0x5a80))
    }
}

#[target_feature(enable = "rdm")]
unsafe fn load8_rect2_i16_rdm(src: &[i16], off: usize) -> int16x8_t {
    unsafe { vqrdmulhq_s16(NeonWide::load8_i16(src, off), vdupq_n_s16(0x5a80)) }
}

#[target_feature(enable = "rdm")]
unsafe fn load4_rect2_i16_rdm(src: &[i16], off: usize) -> int16x8_t {
    unsafe { vqrdmulhq_s16(NeonWide::load4_i16(src, off), vdupq_n_s16(0x5a80)) }
}

pub(crate) struct NeonWideRdm;

impl crate::itx_1d::DctWide for NeonWideRdm {
    type In = int16x8_t;
    type Acc = (int32x4_t, int32x4_t);
    type Coeffs = int16x8_t;
    type Clip = (int32x4_t, int32x4_t, int32x4_t, int32x4_t);

    #[inline(always)]
    fn zero() -> Self::Acc {
        NeonWide::zero()
    }

    #[inline(always)]
    fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        NeonWide::add(a, b)
    }

    #[inline(always)]
    fn sub(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        NeonWide::sub(a, b)
    }

    #[inline(always)]
    fn load_coeffs(table: &[i16], idx: usize) -> Self::Coeffs {
        NeonWide::load_coeffs(table, idx)
    }

    #[inline(always)]
    fn mul_add_lane<const LANE: i32>(acc: Self::Acc, x: Self::In, c: Self::Coeffs) -> Self::Acc {
        NeonWide::mul_add_lane::<LANE>(acc, x, c)
    }

    #[inline(always)]
    unsafe fn load8_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { NeonWide::load8_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load8_rect2_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { load8_rect2_narrow_rdm(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { NeonWide::load4_narrow(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_rect2_narrow(src: &[i32], off: usize) -> Self::In {
        unsafe { load4_rect2_narrow_rdm(src, off) }
    }
    #[inline(always)]
    unsafe fn load8_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { NeonWide::load8_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load8_rect2_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { load8_rect2_i16_rdm(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { NeonWide::load4_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn load4_rect2_i16(src: &[i16], off: usize) -> Self::In {
        unsafe { load4_rect2_i16_rdm(src, off) }
    }

    #[inline(always)]
    fn make_clip(rnd: i32, shift: i32, min: i32, max: i32) -> Self::Clip {
        NeonWide::make_clip(rnd, shift, min, max)
    }

    #[inline(always)]
    unsafe fn store8_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store8_strided_clip(dst, off, stride, acc, clip) }
    }

    #[inline(always)]
    unsafe fn store4_strided_clip(
        dst: &mut [i32],
        off: usize,
        stride: usize,
        acc: Self::Acc,
        clip: Self::Clip,
    ) {
        unsafe { NeonWide::store4_strided_clip(dst, off, stride, acc, clip) }
    }

    #[inline(always)]
    unsafe fn store8(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { NeonWide::store8(dst, off, acc) }
    }

    #[inline(always)]
    unsafe fn store4(dst: &mut [i32], off: usize, acc: Self::Acc) {
        unsafe { NeonWide::store4(dst, off, acc) }
    }
}

pub(crate) struct NeonDct2d;

impl DctSimd4 for NeonDct2d {
    type V = NeonI32x4;
    type Wide = NeonWide;
    #[inline(always)]
    unsafe fn zero() -> Self::V {
        NeonI32x4(unsafe { vdupq_n_s32(0) })
    }

    #[inline(always)]
    unsafe fn splat(v: i32) -> Self::V {
        NeonI32x4(unsafe { vdupq_n_s32(v) })
    }

    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vaddq_s32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vsubq_s32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        NeonI32x4(unsafe { vmulq_s32(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe {
            let scaled = vmlaq_n_s32(vdupq_n_s32(128), a.0, 181);
            NeonI32x4(vshrq_n_s32::<8>(scaled))
        }
    }

    #[inline(always)]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_ptr().add(off) };
        NeonI32x4(unsafe { vld1q_s32(p) })
    }

    #[inline(always)]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        debug_assert!(off + 4 <= ITX_TMP_PIXELS);
        let p = unsafe { tmp.as_mut_ptr().add(off) };
        unsafe { vst1q_s32(p, v.0) };
    }

    #[inline(always)]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) };
        NeonI32x4(unsafe { vld1q_s32(p) })
    }

    #[inline(always)]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        debug_assert!(off + 4 <= src.len());
        let p = unsafe { src.as_ptr().add(off) };
        NeonI32x4(unsafe { vmovl_s16(vld1_s16(p)) })
    }

    #[inline(always)]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        let mut out = [0i32; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), v.0) };
        out
    }
}

pub(crate) struct NeonDct2dRdm;

impl DctSimd4 for NeonDct2dRdm {
    type V = NeonI32x4;
    type Wide = NeonWideRdm;

    #[inline(always)]
    unsafe fn zero() -> Self::V {
        unsafe { NeonDct2d::zero() }
    }

    #[inline(always)]
    unsafe fn splat(v: i32) -> Self::V {
        unsafe { NeonDct2d::splat(v) }
    }

    #[inline(always)]
    unsafe fn add(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::add(a, b) }
    }

    #[inline(always)]
    unsafe fn sub(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::sub(a, b) }
    }

    #[inline(always)]
    unsafe fn mul(a: Self::V, b: Self::V) -> Self::V {
        unsafe { NeonDct2d::mul(a, b) }
    }

    #[inline(always)]
    unsafe fn rect2_scale(a: Self::V) -> Self::V {
        unsafe { NeonDct2d::rect2_scale(a) }
    }

    #[inline(always)]
    unsafe fn load(tmp: &[i32; ITX_TMP_PIXELS], off: usize) -> Self::V {
        unsafe { NeonDct2d::load(tmp, off) }
    }

    #[inline(always)]
    unsafe fn store(tmp: &mut [i32; ITX_TMP_PIXELS], off: usize, v: Self::V) {
        unsafe { NeonDct2d::store(tmp, off, v) }
    }

    #[inline(always)]
    unsafe fn load_slice(src: &[i32], off: usize) -> Self::V {
        unsafe { NeonDct2d::load_slice(src, off) }
    }

    #[inline(always)]
    unsafe fn load_slice_i16(src: &[i16], off: usize) -> Self::V {
        unsafe { NeonDct2d::load_slice_i16(src, off) }
    }

    #[inline(always)]
    unsafe fn to_array(v: Self::V) -> [i32; 4] {
        unsafe { NeonDct2d::to_array(v) }
    }
}

impl Dct2dBackend for NeonDct2d {
    #[inline(always)]
    fn idct_dequant_4x4(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 16, 4, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_8x8(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 64, 8, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_16x16(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 256, 16, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_32x32(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 1024, 32, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }

    #[inline(always)]
    fn idct_dequant_64x64(
        coeff: &mut [i32],
        tmp: &mut [i32; ITX_TMP_PIXELS],
        eob: i32,
        tx: usize,
        is_rect2: bool,
        shift0: i32,
        row_clip_min: i32,
        row_clip_max: i32,
    ) {
        idct_dequant_simd4_core::<Self, 1024, 32, i32>(
            coeff,
            tmp,
            eob,
            tx,
            is_rect2,
            shift0,
            row_clip_min,
            row_clip_max,
        );
    }
}

impl Adst2dBackend for NeonDct2d {
    #[inline(always)]
    fn iadst_dequant_4x4(
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
        itx_dequant_simd4_core::<Self, 16, 4, i32>(
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
        );
    }

    #[inline(always)]
    fn iadst_dequant_8x8(
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
        itx_dequant_simd4_core::<Self, 64, 8, i32>(
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
        );
    }

    #[inline(always)]
    fn iadst_dequant_16x16(
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
        itx_dequant_simd4_core::<Self, 256, 16, i32>(
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
        );
    }
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
    NeonDct2d::idct_dequant_4x4(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    NeonDct2d::idct_dequant_8x8(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
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
    NeonDct2d::idct_dequant_16x16(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
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
    NeonDct2d::idct_dequant_32x32(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_64x64_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    NeonDct2d::idct_dequant_64x64(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

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
    NeonDct2d::iadst_dequant_4x4(
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
    );
}

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
    NeonDct2d::iadst_dequant_8x8(
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
    );
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
    NeonDct2d::iadst_dequant_16x16(
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
    );
}

pub(crate) fn idct_dequant_4x8_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 32, 4, 8, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_8x4_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 32, 8, 4, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_8x16_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 128, 8, 16, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_16x8_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 128, 16, 8, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_16x32_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 512, 16, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_32x16_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 512, 32, 16, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_4x16_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 64, 4, 16, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_16x4_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 64, 16, 4, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_8x32_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 256, 8, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_32x8_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 256, 32, 8, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_4x32_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 128, 4, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn idct_dequant_32x4_neon(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2d, 128, 32, 4, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

pub(crate) fn iadst_dequant_4x8_neon(
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
    crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2d, 32, 4, 8, i32>(
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
    );
}

pub(crate) fn iadst_dequant_8x4_neon(
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
    crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2d, 32, 8, 4, i32>(
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
    );
}

pub(crate) fn iadst_dequant_8x16_neon(
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
    crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2d, 128, 8, 16, i32>(
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
    );
}

pub(crate) fn iadst_dequant_16x8_neon(
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
    crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2d, 128, 16, 8, i32>(
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
    );
}

pub(crate) fn iadst_dequant_4x16_neon(
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
    crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2d, 64, 4, 16, i32>(
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
    );
}

pub(crate) fn iadst_dequant_16x4_neon(
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
    crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2d, 64, 16, 4, i32>(
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
    );
}

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
            crate::itx_2d::idct_dequant_rect_simd4_core::<NeonDct2dRdm, $n, $w, $h, i32>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            );
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
            crate::itx_2d::itx_dequant_rect_simd4_core::<NeonDct2dRdm, $n, $w, $h, i32>(
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
            );
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
unsafe fn idct_dequant_32x32_neon_rdm_impl(
    coeff: &mut [i32],
    tmp: &mut [i32; ITX_TMP_PIXELS],
    eob: i32,
    tx: usize,
    is_rect2: bool,
    shift0: i32,
    row_clip_min: i32,
    row_clip_max: i32,
) {
    crate::itx_2d::idct_dequant_simd4_core::<NeonDct2dRdm, 1024, 32, i32>(
        coeff,
        tmp,
        eob,
        tx,
        is_rect2,
        shift0,
        row_clip_min,
        row_clip_max,
    );
}

// Low-bit-depth i16 coefficient entry points.

macro_rules! idct_i16_neon_fn {
    ($pub:ident, $backend:ty, $n:expr, $s:expr) => {
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
            idct_dequant_simd4_core::<$backend, { $n }, { $s }, i16>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            );
        }
    };
}
macro_rules! iadst_i16_neon_fn {
    ($pub:ident, $backend:ty, $n:expr, $s:expr) => {
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
            itx_dequant_simd4_core::<$backend, { $n }, { $s }, i16>(
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
            );
        }
    };
}
macro_rules! idct_rect_i16_neon_fn {
    ($pub:ident, $backend:ty, $n:expr, $w:expr, $h:expr) => {
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
            crate::itx_2d::idct_dequant_rect_simd4_core::<$backend, { $n }, { $w }, { $h }, i16>(
                coeff,
                tmp,
                eob,
                tx,
                is_rect2,
                shift0,
                row_clip_min,
                row_clip_max,
            );
        }
    };
}
macro_rules! iadst_rect_i16_neon_fn {
    ($pub:ident, $backend:ty, $n:expr, $w:expr, $h:expr) => {
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
            crate::itx_2d::itx_dequant_rect_simd4_core::<$backend, { $n }, { $w }, { $h }, i16>(
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
            );
        }
    };
}
idct_i16_neon_fn!(idct_dequant_4x4_i16_neon, NeonDct2d, 16, 4);
idct_i16_neon_fn!(idct_dequant_8x8_i16_neon, NeonDct2d, 64, 8);
idct_i16_neon_fn!(idct_dequant_16x16_i16_neon, NeonDct2d, 256, 16);
idct_i16_neon_fn!(idct_dequant_32x32_i16_neon, NeonDct2d, 1024, 32);
idct_i16_neon_fn!(idct_dequant_32x32_i16_neon_rdm, NeonDct2dRdm, 1024, 32);
idct_i16_neon_fn!(idct_dequant_64x64_i16_neon, NeonDct2d, 1024, 32);
iadst_i16_neon_fn!(iadst_dequant_4x4_i16_neon, NeonDct2d, 16, 4);
iadst_i16_neon_fn!(iadst_dequant_8x8_i16_neon, NeonDct2d, 64, 8);
iadst_i16_neon_fn!(iadst_dequant_16x16_i16_neon, NeonDct2d, 256, 16);
idct_rect_i16_neon_fn!(idct_dequant_4x8_i16_neon, NeonDct2d, 32, 4, 8);
idct_rect_i16_neon_fn!(idct_dequant_4x8_i16_neon_rdm, NeonDct2dRdm, 32, 4, 8);
idct_rect_i16_neon_fn!(idct_dequant_8x4_i16_neon, NeonDct2d, 32, 8, 4);
idct_rect_i16_neon_fn!(idct_dequant_8x4_i16_neon_rdm, NeonDct2dRdm, 32, 8, 4);
idct_rect_i16_neon_fn!(idct_dequant_8x16_i16_neon, NeonDct2d, 128, 8, 16);
idct_rect_i16_neon_fn!(idct_dequant_8x16_i16_neon_rdm, NeonDct2dRdm, 128, 8, 16);
idct_rect_i16_neon_fn!(idct_dequant_16x8_i16_neon, NeonDct2d, 128, 16, 8);
idct_rect_i16_neon_fn!(idct_dequant_16x8_i16_neon_rdm, NeonDct2dRdm, 128, 16, 8);
idct_rect_i16_neon_fn!(idct_dequant_16x32_i16_neon, NeonDct2d, 512, 16, 32);
idct_rect_i16_neon_fn!(idct_dequant_16x32_i16_neon_rdm, NeonDct2dRdm, 512, 16, 32);
idct_rect_i16_neon_fn!(idct_dequant_32x16_i16_neon, NeonDct2d, 512, 32, 16);
idct_rect_i16_neon_fn!(idct_dequant_32x16_i16_neon_rdm, NeonDct2dRdm, 512, 32, 16);
idct_rect_i16_neon_fn!(idct_dequant_4x16_i16_neon, NeonDct2d, 64, 4, 16);
idct_rect_i16_neon_fn!(idct_dequant_4x16_i16_neon_rdm, NeonDct2dRdm, 64, 4, 16);
idct_rect_i16_neon_fn!(idct_dequant_16x4_i16_neon, NeonDct2d, 64, 16, 4);
idct_rect_i16_neon_fn!(idct_dequant_16x4_i16_neon_rdm, NeonDct2dRdm, 64, 16, 4);
idct_rect_i16_neon_fn!(idct_dequant_8x32_i16_neon, NeonDct2d, 256, 8, 32);
idct_rect_i16_neon_fn!(idct_dequant_8x32_i16_neon_rdm, NeonDct2dRdm, 256, 8, 32);
idct_rect_i16_neon_fn!(idct_dequant_32x8_i16_neon, NeonDct2d, 256, 32, 8);
idct_rect_i16_neon_fn!(idct_dequant_32x8_i16_neon_rdm, NeonDct2dRdm, 256, 32, 8);
idct_rect_i16_neon_fn!(idct_dequant_4x32_i16_neon, NeonDct2d, 128, 4, 32);
idct_rect_i16_neon_fn!(idct_dequant_4x32_i16_neon_rdm, NeonDct2dRdm, 128, 4, 32);
idct_rect_i16_neon_fn!(idct_dequant_32x4_i16_neon, NeonDct2d, 128, 32, 4);
idct_rect_i16_neon_fn!(idct_dequant_32x4_i16_neon_rdm, NeonDct2dRdm, 128, 32, 4);
iadst_rect_i16_neon_fn!(iadst_dequant_4x8_i16_neon, NeonDct2d, 32, 4, 8);
iadst_rect_i16_neon_fn!(iadst_dequant_4x8_i16_neon_rdm, NeonDct2dRdm, 32, 4, 8);
iadst_rect_i16_neon_fn!(iadst_dequant_8x4_i16_neon, NeonDct2d, 32, 8, 4);
iadst_rect_i16_neon_fn!(iadst_dequant_8x4_i16_neon_rdm, NeonDct2dRdm, 32, 8, 4);
iadst_rect_i16_neon_fn!(iadst_dequant_8x16_i16_neon, NeonDct2d, 128, 8, 16);
iadst_rect_i16_neon_fn!(iadst_dequant_8x16_i16_neon_rdm, NeonDct2dRdm, 128, 8, 16);
iadst_rect_i16_neon_fn!(iadst_dequant_16x8_i16_neon, NeonDct2d, 128, 16, 8);
iadst_rect_i16_neon_fn!(iadst_dequant_16x8_i16_neon_rdm, NeonDct2dRdm, 128, 16, 8);
iadst_rect_i16_neon_fn!(iadst_dequant_4x16_i16_neon, NeonDct2d, 64, 4, 16);
iadst_rect_i16_neon_fn!(iadst_dequant_4x16_i16_neon_rdm, NeonDct2dRdm, 64, 4, 16);
iadst_rect_i16_neon_fn!(iadst_dequant_16x4_i16_neon, NeonDct2d, 64, 16, 4);
iadst_rect_i16_neon_fn!(iadst_dequant_16x4_i16_neon_rdm, NeonDct2dRdm, 64, 16, 4);
