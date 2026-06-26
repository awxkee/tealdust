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

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::cfl_dispatch::CflApply8;
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
fn load_u8x32(a: &[u8; 32]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn store_u8x16(a: &mut [u8; 16], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline(always)]
fn store_u8x32(a: &mut [u8; 32], v: __m256i) {
    unsafe { _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn combine_m128(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i32x8_pair_to_i16x16(lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i16x16_to_u8x16(v: __m256i, zero: __m256i) -> __m128i {
    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(
        v, zero,
    )))
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply16_i16_ac(
    ac: __m256i,
    alpha_v: __m256i,
    dc_v: __m256i,
    r1024: __m256i,
    zero: __m256i,
) -> __m128i {
    // Order-preserving madd: cvtepu16 zero-extends i16 ac -> [ac,0] per 32-bit lane
    // sequentially (unlike unpacklo, which crosses 128-bit lanes); madd then
    // reduces to ac*alpha. alpha re-narrowed to i16 (safe: |alpha| <= 16).
    let alpha16 = _mm256_packs_epi32(alpha_v, alpha_v);
    let ac_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(ac));
    let ac_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(ac));

    let diff_lo = _mm256_madd_epi16(ac_lo, alpha16);
    let mag_lo = _mm256_srli_epi32::<11>(_mm256_add_epi32(_mm256_abs_epi32(diff_lo), r1024));
    let val_lo = _mm256_add_epi32(
        dc_v,
        _mm256_blendv_epi8(
            mag_lo,
            _mm256_sub_epi32(zero, mag_lo),
            _mm256_cmpgt_epi32(zero, diff_lo),
        ),
    );

    let diff_hi = _mm256_madd_epi16(ac_hi, alpha16);
    let mag_hi = _mm256_srli_epi32::<11>(_mm256_add_epi32(_mm256_abs_epi32(diff_hi), r1024));
    let val_hi = _mm256_add_epi32(
        dc_v,
        _mm256_blendv_epi8(
            mag_hi,
            _mm256_sub_epi32(zero, mag_hi),
            _mm256_cmpgt_epi32(zero, diff_hi),
        ),
    );

    pack_i16x16_to_u8x16(pack_i32x8_pair_to_i16x16(val_lo, val_hi), zero)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_420_i16(top: __m256i, bot: __m256i, ones: __m256i, dc0v: __m256i) -> __m256i {
    let tsum = _mm256_maddubs_epi16(top, ones);
    let bsum = _mm256_maddubs_epi16(bot, ones);
    let sum16 = _mm256_add_epi16(tsum, bsum);
    _mm256_sub_epi16(_mm256_slli_epi16::<1>(sum16), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_422_uniform_i16(row: __m256i, ones: __m256i, dc0v: __m256i) -> __m256i {
    let sum16 = _mm256_maddubs_epi16(row, ones);
    _mm256_sub_epi16(_mm256_slli_epi16::<2>(sum16), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_422_gauss_i16(row: __m256i, even_mask: __m256i, dc0v: __m256i) -> __m256i {
    let shuffled = _mm256_shuffle_epi8(row, even_mask);
    let lo = _mm256_castsi256_si128(shuffled);
    let hi = _mm256_extracti128_si256::<1>(shuffled);
    let evens = _mm_unpacklo_epi64(lo, hi);
    let y16 = _mm256_cvtepu8_epi16(evens);
    _mm256_sub_epi16(_mm256_slli_epi16::<3>(y16), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_422_i16<const GAUSS: bool>(
    row: __m256i,
    ones: __m256i,
    even_mask: __m256i,
    dc0v: __m256i,
) -> __m256i {
    if GAUSS {
        ac16_422_gauss_i16(row, even_mask, dc0v)
    } else {
        ac16_422_uniform_i16(row, ones, dc0v)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_444_i16(y: __m128i, dc0v: __m256i) -> __m256i {
    let y16 = _mm256_cvtepu8_epi16(y);
    _mm256_sub_epi16(_mm256_slli_epi16::<3>(y16), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply32_444_i16_ac(
    src: __m256i,
    dc0v: __m256i,
    alpha_v: __m256i,
    dc_v: __m256i,
    r1024: __m256i,
    zero: __m256i,
) -> __m256i {
    let lo = apply16_i16_ac(
        ac16_444_i16(_mm256_castsi256_si128(src), dc0v),
        alpha_v,
        dc_v,
        r1024,
        zero,
    );
    let hi = apply16_i16_ac(
        ac16_444_i16(_mm256_extracti128_si256::<1>(src), dc0v),
        alpha_v,
        dc_v,
        r1024,
        zero,
    );
    combine_m128(lo, hi)
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_apply_420_8bpc_avx2(args: CflApply8<'_>) {
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

    let nfull = xlim / 16; // whole 16-chroma (=32-luma) groups
    let xfull = nfull * 16;
    let lfull = nfull * 32;

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));

    let ones = _mm256_set1_epi8(1);
    let dc0v = _mm256_set1_epi16(dc0 as i16);
    let alpha0v = _mm256_set1_epi32(alpha0);
    let alpha1v = _mm256_set1_epi32(alpha1);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let r1024 = _mm256_set1_epi32(1024);
    let zero = _mm256_setzero_si256();

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let top = y[yrow..yrow + lfull].as_chunks::<32>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<32>()
            .0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for (((du, dv), t), b) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac16_420_i16(load_u8x32(t), load_u8x32(b), ones, dc0v);
                    store_u8x16(du, apply16_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                    store_u8x16(dv, apply16_i16_ac(ac, alpha1v, dc2v, r1024, zero));
                }
            }
            (true, false) => {
                for ((d, t), b) in u[urow..urow + xfull]
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac16_420_i16(load_u8x32(t), load_u8x32(b), ones, dc0v);
                    store_u8x16(d, apply16_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                }
            }
            (false, true) => {
                for ((d, t), b) in v[vrow..vrow + xfull]
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac16_420_i16(load_u8x32(t), load_u8x32(b), ones, dc0v);
                    store_u8x16(d, apply16_i16_ac(ac, alpha1v, dc2v, r1024, zero));
                }
            }
            (false, false) => unreachable!(),
        }

        for x in xfull..xlim {
            let xl = x << 1;
            let ac = ((y[yrow + xl] as i32
                + y[yrow + xl + 1] as i32
                + y[yrow + xl + ystride] as i32
                + y[yrow + xl + ystride + 1] as i32)
                << 1)
                - dc0;
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

#[inline(always)]
fn cfl_ac_422_scalar_filter<const GAUSS: bool>(y: &[u8], yrow: usize, x: usize, dc0: i32) -> i32 {
    let xl = x << 1;
    if GAUSS {
        ((y[yrow + xl] as i32) << 3) - dc0
    } else {
        ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
    }
}

#[target_feature(enable = "avx2")]
fn cfl_apply_422_8bpc_avx2_impl<const GAUSS: bool>(args: CflApply8<'_>) {
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

    let nfull = xlim / 16;
    let xfull = nfull * 16;
    let lfull = nfull * 32;

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));

    let ones = _mm256_set1_epi8(1);
    let dc0v = _mm256_set1_epi16(dc0 as i16);
    let alpha0v = _mm256_set1_epi32(alpha0);
    let alpha1v = _mm256_set1_epi32(alpha1);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let r1024 = _mm256_set1_epi32(1024);
    let zero = _mm256_setzero_si256();
    let even_mask = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128, 0, 2, 4, 6, 8,
        10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    );

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let row = y[yrow..yrow + lfull].as_chunks::<32>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for ((du, dv), yy) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(row.iter()) {
                    let ac = ac16_422_i16::<GAUSS>(load_u8x32(yy), ones, even_mask, dc0v);
                    store_u8x16(du, apply16_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                    store_u8x16(dv, apply16_i16_ac(ac, alpha1v, dc2v, r1024, zero));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    let ac = ac16_422_i16::<GAUSS>(load_u8x32(yy), ones, even_mask, dc0v);
                    store_u8x16(du, apply16_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;
                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    let ac = ac16_422_i16::<GAUSS>(load_u8x32(yy), ones, even_mask, dc0v);
                    store_u8x16(dv, apply16_i16_ac(ac, alpha1v, dc2v, r1024, zero));
                }
            }
            (false, false) => unreachable!(),
        }

        for x in xfull..xlim {
            let ac = cfl_ac_422_scalar_filter::<GAUSS>(y, yrow, x, dc0);
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

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_apply_444_8bpc_avx2(args: CflApply8<'_>) {
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

    let nfull = xlim / 32;
    let xfull = nfull * 32;

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));

    let dc0v = _mm256_set1_epi16(dc0 as i16);
    let alpha0v = _mm256_set1_epi32(alpha0);
    let alpha1v = _mm256_set1_epi32(alpha1);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let r1024 = _mm256_set1_epi32(1024);
    let zero = _mm256_setzero_si256();

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let row = y[yrow..yrow + xfull].as_chunks::<32>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<32>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<32>().0;

                for ((du, dv), yy) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(row.iter()) {
                    let yy = load_u8x32(yy);
                    store_u8x32(du, apply32_444_i16_ac(yy, dc0v, alpha0v, dc1v, r1024, zero));
                    store_u8x32(dv, apply32_444_i16_ac(yy, dc0v, alpha1v, dc2v, r1024, zero));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<32>().0;
                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    store_u8x32(
                        du,
                        apply32_444_i16_ac(load_u8x32(yy), dc0v, alpha0v, dc1v, r1024, zero),
                    );
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<32>().0;
                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    store_u8x32(
                        dv,
                        apply32_444_i16_ac(load_u8x32(yy), dc0v, alpha1v, dc2v, r1024, zero),
                    );
                }
            }
            (false, false) => unreachable!(),
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

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_apply_422_8bpc_avx2(args: CflApply8<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => crate::cfl_dispatch::cfl_apply_422_8bpc_scalar(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_422_8bpc_avx2_impl::<true>(args),
        _ => cfl_apply_422_8bpc_avx2_impl::<false>(args),
    }
}
