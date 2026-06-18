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

use core::arch::aarch64::*;

#[inline(always)]
fn predict_one(dc: i32, alpha: i32, ac: i32) -> u8 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, 255) as u8
}

/// Apply alpha to 8 lanes of mean-removed AC (`ac_lo`,`ac_hi` = 4+4 i32) and
/// produce 8 clipped bytes.
#[inline]
#[target_feature(enable = "neon")]
fn apply8(ac_lo: int32x4_t, ac_hi: int32x4_t, alpha: i32, dc: i32) -> uint8x8_t {
    let av = vdupq_n_s32(alpha);
    let dcv = vdupq_n_s32(dc);
    let r1024 = vdupq_n_s32(1024);

    let diff_lo = vmulq_s32(av, ac_lo);
    let mag_lo = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff_lo), r1024));
    // apply_sign: negate magnitude where diff < 0
    let neg_lo = vnegq_s32(mag_lo);
    let mask_lo = vcltq_s32(diff_lo, vdupq_n_s32(0));
    let signed_lo = vbslq_s32(mask_lo, neg_lo, mag_lo);
    let val_lo = vaddq_s32(dcv, signed_lo);

    let diff_hi = vmulq_s32(av, ac_hi);
    let mag_hi = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff_hi), r1024));
    let neg_hi = vnegq_s32(mag_hi);
    let mask_hi = vcltq_s32(diff_hi, vdupq_n_s32(0));
    let signed_hi = vbslq_s32(mask_hi, neg_hi, mag_hi);
    let val_hi = vaddq_s32(dcv, signed_hi);

    // i32 -> u16 (sat, clamps negatives to 0) -> u8 (sat to 255) == clamp(0,255)
    let u16x8 = vcombine_u16(vqmovun_s32(val_lo), vqmovun_s32(val_hi));
    vqmovn_u16(u16x8)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn cfl_apply_420_8bpc_neon_impl(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
) {
    let dc0v = vdupq_n_s32(dc0);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let mut x = 0usize;
        while x + 8 <= xlim {
            let xl = x << 1;
            let top = unsafe { vld1q_u8(y.as_ptr().add(yrow + xl)) };
            let bot = unsafe { vld1q_u8(y.as_ptr().add(yrow + xl + ystride)) };
            // pairwise add adjacent bytes -> 8x u16, then add the two rows
            let tsum = vpaddlq_u8(top);
            let bsum = vpaddlq_u8(bot);
            let sum16 = vaddq_u16(tsum, bsum);
            let sum_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sum16)));
            let sum_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sum16)));
            // ac = (sum << 1) - dc0
            let ac_lo = vsubq_s32(vshlq_n_s32::<1>(sum_lo), dc0v);
            let ac_hi = vsubq_s32(vshlq_n_s32::<1>(sum_hi), dc0v);

            if alpha0 != 0 {
                let r = apply8(ac_lo, ac_hi, alpha0, dc1);
                unsafe { vst1_u8(u.as_mut_ptr().add(urow + x), r) };
            }
            if alpha1 != 0 {
                let r = apply8(ac_lo, ac_hi, alpha1, dc2);
                unsafe { vst1_u8(v.as_mut_ptr().add(vrow + x), r) };
            }
            x += 8;
        }
        // scalar remainder
        while x < xlim {
            let xl = x << 1;
            let ac = ((y[yrow + xl] as i32
                + y[yrow + xl + 1] as i32
                + y[yrow + xl + ystride] as i32
                + y[yrow + xl + ystride + 1] as i32)
                << 1)
                - dc0;
            if alpha0 != 0 {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if alpha1 != 0 {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
            x += 1;
        }
        // right padding
        if alpha0 != 0 {
            let last = u[urow + xlim - 1];
            for xpad in xlim..w {
                u[urow + xpad] = last;
            }
        }
        if alpha1 != 0 {
            let last = v[vrow + xlim - 1];
            for xpad in xlim..w {
                v[vrow + xpad] = last;
            }
        }
        yrow += ystride << 1;
        urow += cstride;
        vrow += cstride;
    }
    // bottom padding
    if alpha0 != 0 {
        let src = urow0 + (ylim - 1) * cstride;
        for yy in ylim..h {
            let dst = urow0 + yy * cstride;
            u.copy_within(src..src + w, dst);
        }
    }
    if alpha1 != 0 {
        let src = vrow0 + (ylim - 1) * cstride;
        for yy in ylim..h {
            let dst = vrow0 + yy * cstride;
            v.copy_within(src..src + w, dst);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_apply_420_8bpc_neon(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
) {
    unsafe {
        cfl_apply_420_8bpc_neon_impl(
            y, u, v, yrow0, urow0, vrow0, ystride, cstride, w, h, xlim, ylim, dc0, dc1, dc2,
            alpha0, alpha1,
        )
    }
}
