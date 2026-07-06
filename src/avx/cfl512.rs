/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use std::arch::x86_64::*;

use crate::cfl_dispatch::{CflAlphaAccum8, CflApply8, CflGenMat8, CflGenYRow8, CflMhccpPred8};

const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

#[inline(always)]
fn predict_one(dc: i32, alpha: i32, ac: i32) -> u8 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, 255) as u8
}

#[inline(always)]
fn pad_bottom(plane: &mut [u8], row0: usize, stride: usize, w: usize, h: usize, ylim: usize) {
    debug_assert_ne!(ylim, 0);
    let src = row0 + (ylim - 1) * stride;
    for yy in ylim..h {
        let dst = row0 + yy * stride;
        plane.copy_within(src..src + w, dst);
    }
}

#[inline(always)]
fn cfl_ac_420_scalar(y: &[u8], yrow: usize, ystride: usize, x: usize, dc0: i32) -> i32 {
    let xl = x << 1;
    ((y[yrow + xl] as i32
        + y[yrow + xl + 1] as i32
        + y[yrow + xl + ystride] as i32
        + y[yrow + xl + ystride + 1] as i32)
        << 1)
        - dc0
}

#[inline(always)]
fn cfl_ac_422_scalar(y: &[u8], yrow: usize, x: usize, dc0: i32) -> i32 {
    let xl = x << 1;
    ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn load_u8x16_i32(src: &[u8]) -> __m512i {
    debug_assert!(src.len() >= 16);
    unsafe { _mm512_cvtepu8_epi32(_mm_loadu_si128(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn load_u8x32_i16(src: &[u8]) -> __m512i {
    debug_assert!(src.len() >= 32);
    unsafe { _mm512_cvtepu8_epi16(_mm256_loadu_si256(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn load_u8x64(src: &[u8]) -> __m512i {
    debug_assert!(src.len() >= 64);
    unsafe { _mm512_loadu_si512(src.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn load_u16x32(src: &[u16]) -> __m512i {
    debug_assert!(src.len() >= 32);
    unsafe { _mm512_loadu_si512(src.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn store_u8x32(dst: &mut [u8], v: __m256i) {
    debug_assert!(dst.len() >= 32);
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), v) };
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn store_i32x16(dst: &mut [i32; 16], v: __m512i) {
    unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast(), v) };
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn reduce_i32x16(v: __m512i) -> i32 {
    let mut tmp = [0i32; 16];
    store_i32x16(&mut tmp, v);
    tmp.iter().sum()
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn alpha_abs_i16(alpha: i32) -> __m512i {
    _mm512_set1_epi16((if alpha < 0 { -alpha } else { alpha }) as i16)
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn apply32_i16_ac(ac: __m512i, alpha_abs: __m512i, alpha: i32, dc_v: __m512i) -> __m256i {
    let zero = _mm512_setzero_si512();
    let ac_neg = _mm512_cmpgt_epi16_mask(zero, ac);
    let sign = if alpha < 0 { !ac_neg } else { ac_neg };
    let mag = _mm512_mulhrs_epi16(_mm512_slli_epi16::<4>(_mm512_abs_epi16(ac)), alpha_abs);
    let neg_mag = _mm512_sub_epi16(zero, mag);
    let signed = _mm512_mask_blend_epi16(sign, mag, neg_mag);
    _mm512_cvtusepi16_epi8(_mm512_add_epi16(dc_v, signed))
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn ac32_420_uniform(cur: __m512i, bot: __m512i, dc0v: __m512i) -> __m512i {
    let ones = _mm512_set1_epi8(1);
    let csum = _mm512_maddubs_epi16(cur, ones);
    let bsum = _mm512_maddubs_epi16(bot, ones);
    _mm512_sub_epi16(_mm512_slli_epi16::<1>(_mm512_add_epi16(csum, bsum)), dc0v)
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn ac32_422_uniform(row: __m512i, dc0v: __m512i) -> __m512i {
    let ones = _mm512_set1_epi8(1);
    _mm512_sub_epi16(
        _mm512_slli_epi16::<2>(_mm512_maddubs_epi16(row, ones)),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn ac32_444(row: __m512i, dc0v: __m512i) -> __m512i {
    _mm512_sub_epi16(_mm512_slli_epi16::<3>(row), dc0v)
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_apply_420_8bpc_avx512(args: CflApply8<'_>) {
    if args.params.filter_type == CFL_FLT_TYPE_VSTRIP
        || args.params.filter_type == CFL_FLT_TYPE_GAUSS
    {
        crate::avx::cfl_apply_420_8bpc_avx2(args);
        return;
    }

    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let xfull = (xlim / 32) * 32;
    let dc0v = _mm512_set1_epi16(dc0 as i16);
    let dc1v = _mm512_set1_epi16(dc1 as i16);
    let dc2v = _mm512_set1_epi16(dc2 as i16);
    let alpha0_abs = alpha_abs_i16(alpha0);
    let alpha1_abs = alpha_abs_i16(alpha1);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _cy in 0..ylim {
        for x in (0..xfull).step_by(32) {
            let xl = x << 1;
            let ac = ac32_420_uniform(
                load_u8x64(&y[yrow + xl..]),
                load_u8x64(&y[yrow + ystride + xl..]),
                dc0v,
            );
            if do_u {
                let out = apply32_i16_ac(ac, alpha0_abs, alpha0, dc1v);
                store_u8x32(&mut u[urow + x..], out);
            }
            if do_v {
                let out = apply32_i16_ac(ac, alpha1_abs, alpha1, dc2v);
                store_u8x32(&mut v[vrow + x..], out);
            }
        }

        for x in xfull..xlim {
            let ac = cfl_ac_420_scalar(y, yrow, ystride, x, dc0);
            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }
        yrow += ystride << 1;
        urow += cstride;
        vrow += cstride;
    }
    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_apply_422_8bpc_avx512(args: CflApply8<'_>) {
    if args.params.filter_type == CFL_FLT_TYPE_VSTRIP
        || args.params.filter_type == CFL_FLT_TYPE_GAUSS
    {
        crate::avx::cfl_apply_422_8bpc_avx2(args);
        return;
    }

    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let xfull = (xlim / 32) * 32;
    let dc0v = _mm512_set1_epi16(dc0 as i16);
    let dc1v = _mm512_set1_epi16(dc1 as i16);
    let dc2v = _mm512_set1_epi16(dc2 as i16);
    let alpha0_abs = alpha_abs_i16(alpha0);
    let alpha1_abs = alpha_abs_i16(alpha1);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        for x in (0..xfull).step_by(32) {
            let ac = ac32_422_uniform(load_u8x64(&y[yrow + (x << 1)..]), dc0v);
            if do_u {
                let out = apply32_i16_ac(ac, alpha0_abs, alpha0, dc1v);
                store_u8x32(&mut u[urow + x..], out);
            }
            if do_v {
                let out = apply32_i16_ac(ac, alpha1_abs, alpha1, dc2v);
                store_u8x32(&mut v[vrow + x..], out);
            }
        }

        for x in xfull..xlim {
            let ac = cfl_ac_422_scalar(y, yrow, x, dc0);
            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }
        yrow += ystride;
        urow += cstride;
        vrow += cstride;
    }
    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_apply_444_8bpc_avx512(args: CflApply8<'_>) {
    let CflApply8 {
        y,
        u,
        v,
        layout,
        area,
        params,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let xfull = (xlim / 32) * 32;
    let dc0v = _mm512_set1_epi16(dc0 as i16);
    let dc1v = _mm512_set1_epi16(dc1 as i16);
    let dc2v = _mm512_set1_epi16(dc2 as i16);
    let alpha0_abs = alpha_abs_i16(alpha0);
    let alpha1_abs = alpha_abs_i16(alpha1);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        for x in (0..xfull).step_by(32) {
            let ac = ac32_444(load_u8x32_i16(&y[yrow + x..]), dc0v);
            if do_u {
                let out = apply32_i16_ac(ac, alpha0_abs, alpha0, dc1v);
                store_u8x32(&mut u[urow + x..], out);
            }
            if do_v {
                let out = apply32_i16_ac(ac, alpha1_abs, alpha1, dc2v);
                store_u8x32(&mut v[vrow + x..], out);
            }
        }

        for x in xfull..xlim {
            let ac = ((y[yrow + x] as i32) << 3) - dc0;
            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }
        yrow += ystride;
        urow += cstride;
        vrow += cstride;
    }
    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[inline(always)]
fn load_strided_u8x16(samples: &[u8], mut off: usize, stride: usize) -> [u8; 16] {
    let mut tmp = [0u8; 16];
    for dst in &mut tmp {
        *dst = samples[off];
        off += stride;
    }
    tmp
}

#[inline(always)]
fn load_strided_u8x32(samples: &[u8], mut off: usize, stride: usize) -> [u8; 32] {
    let mut tmp = [0u8; 32];
    for dst in &mut tmp {
        *dst = samples[off];
        off += stride;
    }
    tmp
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_gen_mat_8bpc_avx512(args: CflGenMat8<'_>) {
    if args.len < 16 {
        crate::cfl_dispatch::cfl_gen_mat_8bpc_scalar(args);
        return;
    }

    let CflGenMat8 {
        sums,
        imat0,
        imat1,
        imat_off,
        y,
        v0_off,
        v0_stride,
        v1_off,
        v1_stride,
        len,
    } = args;

    let mut acc00 = _mm512_setzero_si512();
    let mut acc01 = _mm512_setzero_si512();
    let mut acc0 = _mm512_setzero_si512();
    let mut acc11 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let chunks = len / 16;
    let processed = chunks * 16;

    for chunk_idx in 0..chunks {
        let rel = chunk_idx * 16;
        let v0 = if v0_stride == 1 {
            load_u8x16_i32(&y[v0_off + rel..])
        } else {
            let tmp = load_strided_u8x16(y, v0_off + rel * v0_stride, v0_stride);
            load_u8x16_i32(&tmp)
        };
        let raw1 = if v1_stride == 1 {
            load_u8x16_i32(&y[v1_off + rel..])
        } else {
            let tmp = load_strided_u8x16(y, v1_off + rel * v1_stride, v1_stride);
            load_u8x16_i32(&tmp)
        };
        let v1 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
            _mm512_mullo_epi32(raw1, raw1),
            _mm512_set1_epi32(128),
        ));

        acc00 = _mm512_add_epi32(acc00, _mm512_mullo_epi32(v0, v0));
        acc01 = _mm512_add_epi32(acc01, _mm512_mullo_epi32(v0, v1));
        acc0 = _mm512_add_epi32(acc0, v0);
        acc11 = _mm512_add_epi32(acc11, _mm512_mullo_epi32(v1, v1));
        acc1 = _mm512_add_epi32(acc1, v1);

        let mut v0_tmp = [0i32; 16];
        let mut v1_tmp = [0i32; 16];
        store_i32x16(&mut v0_tmp, v0);
        store_i32x16(&mut v1_tmp, v1);
        let out = imat_off + rel;
        for (i, (&a, &b)) in v0_tmp.iter().zip(v1_tmp.iter()).enumerate() {
            imat0[out + i] = a as u16;
            imat1[out + i] = b as u16;
        }
    }

    sums.m00 += reduce_i32x16(acc00);
    sums.m01 += reduce_i32x16(acc01);
    sums.sum0 += reduce_i32x16(acc0);
    sums.m11 += reduce_i32x16(acc11);
    sums.sum1 += reduce_i32x16(acc1);

    if processed < len {
        crate::cfl_dispatch::cfl_gen_mat_8bpc_scalar(crate::cfl_dispatch::CflGenMat8 {
            sums,
            imat0,
            imat1,
            imat_off: imat_off + processed,
            y,
            v0_off: v0_off + processed * v0_stride,
            v0_stride,
            v1_off: v1_off + processed * v1_stride,
            v1_stride,
            len: len - processed,
        });
    }
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_alpha_accum_8bpc_avx512(args: CflAlphaAccum8<'_>) {
    if args.len < 32 {
        crate::cfl_dispatch::cfl_alpha_accum_8bpc_scalar(args);
        return;
    }

    let CflAlphaAccum8 {
        alpha,
        samples,
        sample_off,
        sample_stride,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    } = args;

    let ones = _mm512_set1_epi16(1);
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let chunks = len / 32;
    let processed = chunks * 32;

    if sample_stride == 1 {
        for chunk_idx in 0..chunks {
            let rel = chunk_idx * 32;
            let v = load_u8x32_i16(&samples[sample_off + rel..]);
            let i = imat_off + rel;
            let m0 = load_u16x32(&imat0[i..]);
            let m1 = load_u16x32(&imat1[i..]);
            acc0 = _mm512_dpwssds_epi32(acc0, v, m0);
            acc1 = _mm512_dpwssds_epi32(acc1, v, m1);
            acc2 = _mm512_dpwssds_epi32(acc2, v, ones);
        }
    } else {
        let mut off = sample_off;
        for chunk_idx in 0..chunks {
            let i = imat_off + chunk_idx * 32;
            let s = load_strided_u8x32(samples, off, sample_stride);
            off += 32 * sample_stride;
            let v = load_u8x32_i16(&s);
            let m0 = load_u16x32(&imat0[i..]);
            let m1 = load_u16x32(&imat1[i..]);
            acc0 = _mm512_dpwssds_epi32(acc0, v, m0);
            acc1 = _mm512_dpwssds_epi32(acc1, v, m1);
            acc2 = _mm512_dpwssds_epi32(acc2, v, ones);
        }
    }

    alpha[0] += reduce_i32x16(acc0);
    alpha[1] += reduce_i32x16(acc1);
    alpha[2] += reduce_i32x16(acc2) << a2sh;

    if processed < len {
        crate::cfl_dispatch::cfl_alpha_accum_8bpc_scalar(CflAlphaAccum8 {
            alpha,
            samples,
            sample_off: sample_off + processed * sample_stride,
            sample_stride,
            imat0,
            imat1,
            imat_off: imat_off + processed,
            len: len - processed,
            a2sh,
        });
    }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn mhccp_round_signed_shift16(v: __m512i) -> __m512i {
    let zero = _mm512_setzero_si512();
    let neg = _mm512_cmpgt_epi32_mask(zero, v);
    let mag = _mm512_srai_epi32::<16>(_mm512_add_epi32(
        _mm512_abs_epi32(v),
        _mm512_set1_epi32(1 << 15),
    ));
    _mm512_mask_sub_epi32(mag, neg, zero, mag)
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn mhccp_mul32(v: __m512i, alpha: i32) -> __m512i {
    mhccp_round_signed_shift16(_mm512_mullo_epi32(v, _mm512_set1_epi32(alpha)))
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn mhccp_sqrnd8(v: __m512i) -> __m512i {
    _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(v, v),
        _mm512_set1_epi32(128),
    ))
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn mhccp_pred16(v0: __m512i, v1: __m512i, alpha: [i32; 3], a2v2: __m512i) -> __m512i {
    _mm512_add_epi32(
        _mm512_add_epi32(
            mhccp_mul32(v0, alpha[0]),
            mhccp_mul32(mhccp_sqrnd8(v1), alpha[1]),
        ),
        a2v2,
    )
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn mhccp_store_u8x16(dst: &mut [u8; 16], v: __m512i) {
    let mut tmp = [0i32; 16];
    store_i32x16(&mut tmp, v);
    for (d, &s) in dst.iter_mut().zip(tmp.iter()) {
        *d = s.clamp(0, 255) as u8;
    }
}

#[inline(always)]
fn mhccp_pred_one_8(alpha: &[i32; 3], a2v2: i32, v0: i32, v1: i32) -> u8 {
    let sq = (v1 * v1 + 128) >> 8;
    (crate::ipred::mul32(alpha[0], v0, 16) + crate::ipred::mul32(alpha[1], sq, 16) + a2v2)
        .clamp(0, 255) as u8
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_mhccp_pred_8bpc_avx512(args: CflMhccpPred8<'_>) {
    if !crate::cfl_dispatch::cfl_mhccp_coeffs_fit_fast_mul(&args.alpha) || args.w < 16 {
        crate::cfl_dispatch::cfl_mhccp_pred_8bpc_scalar(args);
        return;
    }

    let CflMhccpPred8 {
        dst,
        dst_stride,
        src,
        src_off,
        src_top_stride,
        w,
        h,
        alpha,
        edge_flags,
        dir,
    } = args;
    let has_t = edge_flags & (1 << 2) != 0;
    let has_l = edge_flags & (1 << 3) != 0;
    let dir_t = dir == crate::levels::CflMhDir::Top;
    let dir_l = dir == crate::levels::CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let a2v2_scalar = crate::ipred::mul32(alpha[2], 128, 16);
    let a2v2 = _mm512_set1_epi32(a2v2_scalar);

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && y < h {
        let dst_row = &mut dst[..w];
        let (dst16, dst_tail) = dst_row.as_chunks_mut::<16>();
        let prev = sp - src_top_stride;
        for (i, chunk) in dst16.iter_mut().enumerate() {
            let x = i * 16;
            let out = mhccp_pred16(
                load_u8x16_i32(&src[prev + x..]),
                load_u8x16_i32(&src[sp + x..]),
                alpha,
                a2v2,
            );
            mhccp_store_u8x16(chunk, out);
        }
        let done = dst16.len() * 16;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            *d = mhccp_pred_one_8(
                &alpha,
                a2v2_scalar,
                src[prev + x] as i32,
                src[sp + x] as i32,
            );
        }
        sp += w;
        y = 1;
    }

    for (row_y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate().skip(y) {
        let dst_row = &mut dst_row[..w];
        let mut x0 = 0usize;
        if dir_l {
            let v0 = if has_l {
                src[left_off + row_y * n_left + 1] as i32
            } else {
                src[sp] as i32
            };
            dst_row[0] = mhccp_pred_one_8(&alpha, a2v2_scalar, v0, src[sp] as i32);
            x0 = 1;
        }

        let (dst16, dst_tail) = dst_row[x0..].as_chunks_mut::<16>();
        for (i, chunk) in dst16.iter_mut().enumerate() {
            let x = x0 + i * 16;
            let v0_off = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x - 1
            } else {
                sp + x
            };
            let out = mhccp_pred16(
                load_u8x16_i32(&src[v0_off..]),
                load_u8x16_i32(&src[sp + x..]),
                alpha,
                a2v2,
            );
            mhccp_store_u8x16(chunk, out);
        }
        let done = x0 + dst16.len() * 16;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            let v0_idx = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x.saturating_sub(1)
            } else {
                sp + x
            };
            *d = mhccp_pred_one_8(&alpha, a2v2_scalar, src[v0_idx] as i32, src[sp + x] as i32);
        }
        sp += w;
    }
}

#[inline]
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
fn gen_y32_uniform(src: &[u8], src_off: usize, bottom_offset: usize, x: usize) -> __m256i {
    let xl = x << 1;
    let ones = _mm512_set1_epi8(1);
    let cur = load_u8x64(&src[src_off + xl..]);
    let bot = load_u8x64(&src[src_off + bottom_offset + xl..]);
    let sum = _mm512_add_epi16(
        _mm512_maddubs_epi16(cur, ones),
        _mm512_maddubs_epi16(bot, ones),
    );
    _mm512_cvtusepi16_epi8(_mm512_srli_epi16::<2>(sum))
}

#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn cfl_gen_y_row_8bpc_avx512(args: CflGenYRow8<'_>) {
    if args.filter_type != 0 {
        crate::avx::cfl_gen_y_row_8bpc_avx2(args);
        return;
    }

    let CflGenYRow8 {
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type: _,
    } = args;

    let (chunks, rem) = dst.as_chunks_mut::<32>();
    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let x = n_left + chunk_idx * 32;
        store_u8x32(chunk, gen_y32_uniform(src, src_off, bottom_offset, x));
    }

    if !rem.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_8bpc_scalar(CflGenYRow8 {
            dst: rem,
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left: n_left + chunks.len() * 32,
            filter_type: 0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfl_dispatch::{CflArea, CflGenMatSums, CflLayout, CflParams};
    use crate::levels::CflMhDir;

    #[inline]
    fn has_avx512_cfl() -> bool {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512vnni")
    }

    fn gen_u8(len: usize, mut state: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            out.push((state >> 24) as u8);
        }
        out
    }

    fn fill_u16<const N: usize>(state: u32) -> [u16; N] {
        let src = gen_u8(N, state);
        core::array::from_fn(|i| src[i] as u16)
    }

    #[inline]
    fn sums_tuple(s: CflGenMatSums) -> (i32, i32, i32, i32, i32) {
        (s.m00, s.m01, s.sum0, s.m11, s.sum1)
    }

    fn run_apply_compare(subsampling: u8) {
        let yrow0 = 19;
        let urow0 = 7;
        let vrow0 = 11;
        let ystride = 192;
        let cstride = 80;
        let w = 45;
        let h = 9;
        let xlim = 37;
        let ylim = 6;
        let y_len = match subsampling {
            0 => yrow0 + (ylim * 2 + 1) * ystride,
            _ => yrow0 + (ylim + 1) * ystride,
        };
        let uv_len = vrow0 + h * cstride + w + 32;
        let y = gen_u8(y_len, 0x1234_5678 + subsampling as u32);
        let mut u_ref = gen_u8(uv_len, 0x2234_5678 + subsampling as u32);
        let mut v_ref = gen_u8(uv_len, 0x3234_5678 + subsampling as u32);
        let mut u_avx = u_ref.clone();
        let mut v_avx = v_ref.clone();
        let layout = CflLayout {
            yrow0,
            urow0,
            vrow0,
            ystride,
            cstride,
        };
        let area = CflArea { w, h, xlim, ylim };
        let params = CflParams {
            dc0: 991,
            dc1: 129,
            dc2: 117,
            alpha0: 37,
            alpha1: -41,
            filter_type: 0,
        };

        match subsampling {
            0 => crate::cfl_dispatch::cfl_apply_420_8bpc_scalar(CflApply8 {
                y: &y,
                u: &mut u_ref,
                v: &mut v_ref,
                layout,
                area,
                params,
            }),
            1 => crate::cfl_dispatch::cfl_apply_422_8bpc_scalar(CflApply8 {
                y: &y,
                u: &mut u_ref,
                v: &mut v_ref,
                layout,
                area,
                params,
            }),
            _ => crate::cfl_dispatch::cfl_apply_444_8bpc_scalar(CflApply8 {
                y: &y,
                u: &mut u_ref,
                v: &mut v_ref,
                layout,
                area,
                params,
            }),
        }

        unsafe {
            match subsampling {
                0 => cfl_apply_420_8bpc_avx512(CflApply8 {
                    y: &y,
                    u: &mut u_avx,
                    v: &mut v_avx,
                    layout,
                    area,
                    params,
                }),
                1 => cfl_apply_422_8bpc_avx512(CflApply8 {
                    y: &y,
                    u: &mut u_avx,
                    v: &mut v_avx,
                    layout,
                    area,
                    params,
                }),
                _ => cfl_apply_444_8bpc_avx512(CflApply8 {
                    y: &y,
                    u: &mut u_avx,
                    v: &mut v_avx,
                    layout,
                    area,
                    params,
                }),
            }
        }

        assert_eq!(u_avx, u_ref, "U mismatch for subsampling {subsampling}");
        assert_eq!(v_avx, v_ref, "V mismatch for subsampling {subsampling}");
    }

    #[test]
    fn cfl_apply_avx512_matches_scalar() {
        if !has_avx512_cfl() {
            return;
        }
        run_apply_compare(0);
        run_apply_compare(1);
        run_apply_compare(2);
    }

    #[test]
    fn cfl_gen_y_row_avx512_matches_scalar() {
        if !has_avx512_cfl() {
            return;
        }

        let dst_len = 49;
        let n_left = 5;
        let src_off = 23;
        let bottom_offset = 192;
        let src_len = src_off + bottom_offset + ((n_left + dst_len) << 1) + 64;
        let src = gen_u8(src_len, 0x4567_89ab);
        let top = gen_u8(256, 0x5567_89ab);
        let mut dst_ref = gen_u8(dst_len, 0x6567_89ab);
        let mut dst_avx = dst_ref.clone();

        crate::cfl_dispatch::cfl_gen_y_row_8bpc_scalar(CflGenYRow8 {
            dst: &mut dst_ref,
            src: &src,
            src_off,
            top: &top,
            top_off: 3,
            bottom_offset,
            n_left,
            filter_type: 0,
        });
        unsafe {
            cfl_gen_y_row_8bpc_avx512(CflGenYRow8 {
                dst: &mut dst_avx,
                src: &src,
                src_off,
                top: &top,
                top_off: 3,
                bottom_offset,
                n_left,
                filter_type: 0,
            });
        }

        assert_eq!(dst_avx, dst_ref);
    }

    #[test]
    fn cfl_gen_mat_avx512_matches_scalar() {
        if !has_avx512_cfl() {
            return;
        }

        let len = 79;
        let imat_off = 7;
        let v0_off = 5;
        let v0_stride = 3;
        let v1_off = 11;
        let v1_stride = 2;
        let y_len = 1 + (v0_off + (len - 1) * v0_stride).max(v1_off + (len - 1) * v1_stride);
        let y = gen_u8(y_len, 0x7654_3210);
        let mut sums_ref = CflGenMatSums::default();
        let mut sums_avx = CflGenMatSums::default();
        let mut imat0_ref = [0x55u16; crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES];
        let mut imat1_ref = [0xaau16; crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES];
        let mut imat0_avx = imat0_ref;
        let mut imat1_avx = imat1_ref;

        crate::cfl_dispatch::cfl_gen_mat_8bpc_scalar(CflGenMat8 {
            sums: &mut sums_ref,
            imat0: &mut imat0_ref,
            imat1: &mut imat1_ref,
            imat_off,
            y: &y,
            v0_off,
            v0_stride,
            v1_off,
            v1_stride,
            len,
        });
        unsafe {
            cfl_gen_mat_8bpc_avx512(CflGenMat8 {
                sums: &mut sums_avx,
                imat0: &mut imat0_avx,
                imat1: &mut imat1_avx,
                imat_off,
                y: &y,
                v0_off,
                v0_stride,
                v1_off,
                v1_stride,
                len,
            });
        }

        assert_eq!(sums_tuple(sums_avx), sums_tuple(sums_ref));
        assert_eq!(imat0_avx, imat0_ref);
        assert_eq!(imat1_avx, imat1_ref);
    }

    fn run_alpha_accum_compare(sample_stride: usize) {
        let len = 83;
        let sample_off = 13;
        let imat_off = 17;
        let samples_len = sample_off + (len - 1) * sample_stride + 1;
        let samples = gen_u8(samples_len, 0x8765_4321 + sample_stride as u32);
        let imat0 = fill_u16::<{ crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES }>(0x9988_7766);
        let imat1 = fill_u16::<{ crate::ipred::CFL_MHCCP_MAX_EDGE_SAMPLES }>(0x8877_6655);
        let mut alpha_ref = [3, -9, 7];
        let mut alpha_avx = alpha_ref;

        crate::cfl_dispatch::cfl_alpha_accum_8bpc_scalar(CflAlphaAccum8 {
            alpha: &mut alpha_ref,
            samples: &samples,
            sample_off,
            sample_stride,
            imat0: &imat0,
            imat1: &imat1,
            imat_off,
            len,
            a2sh: 2,
        });
        unsafe {
            cfl_alpha_accum_8bpc_avx512(CflAlphaAccum8 {
                alpha: &mut alpha_avx,
                samples: &samples,
                sample_off,
                sample_stride,
                imat0: &imat0,
                imat1: &imat1,
                imat_off,
                len,
                a2sh: 2,
            });
        }

        assert_eq!(alpha_avx, alpha_ref);
    }

    #[test]
    fn cfl_alpha_accum_avx512_matches_scalar() {
        if !has_avx512_cfl() {
            return;
        }
        run_alpha_accum_compare(1);
        run_alpha_accum_compare(3);
    }

    fn run_mhccp_compare(dir: CflMhDir) {
        let w = 37;
        let h = 7;
        let dst_stride = 48;
        let src_off = 128;
        let src_top_stride = 64;
        let edge_flags = (1 << 2) | (1 << 3);
        let has_t = true;
        let has_l = true;
        let dir_t = dir == CflMhDir::Top;
        let dir_l = dir == CflMhDir::Left;
        let n_top = if has_t { 1 + dir_t as usize } else { 0 };
        let n_left = if has_l { 1 + dir_l as usize } else { 0 };
        let left_off = src_off + 64 * 64 + n_left * n_top;
        let src_len = left_off + h * n_left + 32;
        let src = gen_u8(src_len, 0x1357_2468 + dir as u32);
        let mut dst_ref = gen_u8(dst_stride * h, 0x2468_1357 + dir as u32);
        let mut dst_avx = dst_ref.clone();
        let alpha = [512, -384, 1024];

        crate::cfl_dispatch::cfl_mhccp_pred_8bpc_scalar(CflMhccpPred8 {
            dst: &mut dst_ref,
            dst_stride,
            src: &src,
            src_off,
            src_top_stride,
            w,
            h,
            alpha,
            edge_flags,
            dir,
        });
        unsafe {
            cfl_mhccp_pred_8bpc_avx512(CflMhccpPred8 {
                dst: &mut dst_avx,
                dst_stride,
                src: &src,
                src_off,
                src_top_stride,
                w,
                h,
                alpha,
                edge_flags,
                dir,
            });
        }

        assert_eq!(dst_avx, dst_ref, "MHCCP mismatch for dir {dir:?}");
    }

    #[test]
    fn cfl_mhccp_pred_avx512_matches_scalar() {
        if !has_avx512_cfl() {
            return;
        }
        run_mhccp_compare(CflMhDir::Center);
        run_mhccp_compare(CflMhDir::Top);
        run_mhccp_compare(CflMhDir::Left);
    }
}
