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
fn load_u8x8(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn load_u8x4_tail(src: &[u8]) -> __m128i {
    debug_assert!(src.len() >= 4);
    unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) }
}

#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn load_u8x32(a: &[u8; 32]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
fn store_u8x4(a: &mut [u8; 4], v: __m128i) {
    unsafe { _mm_store_ss(a.as_mut_ptr().cast(), _mm_castsi128_ps(v)) }
}

#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: __m128i) {
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, v) };
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
fn pack_i16x16_to_u8x16(v: __m256i, zero: __m256i) -> __m128i {
    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(
        v, zero,
    )))
}

#[inline]
#[target_feature(enable = "avx2")]
fn alpha_abs_i16(alpha: i32) -> __m256i {
    _mm256_set1_epi16((if alpha < 0 { -alpha } else { alpha }) as i16)
}

#[inline]
#[target_feature(enable = "avx2")]
fn alpha_sign_i16(alpha: i32) -> __m256i {
    _mm256_set1_epi16((alpha >> 31) as i16)
}

#[inline]
#[target_feature(enable = "avx2")]
fn alpha_abs_i16_128(alpha: i32) -> __m128i {
    _mm_set1_epi16((if alpha < 0 { -alpha } else { alpha }) as i16)
}

#[inline]
#[target_feature(enable = "avx2")]
fn alpha_sign_i16_128(alpha: i32) -> __m128i {
    _mm_set1_epi16((alpha >> 31) as i16)
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply8_i16_ac(
    ac: __m128i,
    alpha_abs: __m128i,
    alpha_sign: __m128i,
    dc_v: __m128i,
    zero: __m128i,
) -> __m128i {
    let ac_sign = _mm_cmpgt_epi16(zero, ac);
    let mag = _mm_mulhrs_epi16(_mm_slli_epi16::<4>(_mm_abs_epi16(ac)), alpha_abs);
    let neg_mag = _mm_sub_epi16(zero, mag);
    let sign = _mm_xor_si128(ac_sign, alpha_sign);
    let signed = _mm_blendv_epi8(mag, neg_mag, sign);
    _mm_packus_epi16(_mm_add_epi16(dc_v, signed), zero)
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply16_i16_ac(
    ac: __m256i,
    alpha_abs: __m256i,
    alpha_sign: __m256i,
    dc_v: __m256i,
    zero: __m256i,
) -> __m128i {
    let ac_sign = _mm256_cmpgt_epi16(zero, ac);
    let mag = _mm256_mulhrs_epi16(_mm256_slli_epi16::<4>(_mm256_abs_epi16(ac)), alpha_abs);
    let neg_mag = _mm256_sub_epi16(zero, mag);
    let sign = _mm256_xor_si256(ac_sign, alpha_sign);
    let signed = _mm256_blendv_epi8(mag, neg_mag, sign);
    pack_i16x16_to_u8x16(_mm256_add_epi16(dc_v, signed), zero)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_i16(cur: __m128i, bot: __m128i, ones: __m128i, dc0v: __m128i) -> __m128i {
    let csum = _mm_maddubs_epi16(cur, ones);
    let bsum = _mm_maddubs_epi16(bot, ones);
    let sum16 = _mm_add_epi16(csum, bsum);
    _mm_sub_epi16(_mm_slli_epi16::<1>(sum16), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn even_u8x8_to_i16(row: __m128i, even_mask: __m128i) -> __m128i {
    _mm_shuffle_epi8(row, even_mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn left_u8x8_to_i16(row: __m128i, prev_byte: u8, left_mask: __m128i) -> __m128i {
    let prev = _mm_set1_epi8(prev_byte as i8);
    let shifted = _mm_alignr_epi8::<15>(row, prev);
    _mm_shuffle_epi8(shifted, left_mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_vstrip_i16(
    cur: __m128i,
    bot: __m128i,
    prev_cur: u8,
    prev_bot: u8,
    center_right_w: __m128i,
    left_mask: __m128i,
    dc0v: __m128i,
) -> __m128i {
    let cur_left = left_u8x8_to_i16(cur, prev_cur, left_mask);
    let bot_left = left_u8x8_to_i16(bot, prev_bot, left_mask);
    let cur_center_right = _mm_maddubs_epi16(cur, center_right_w);
    let bot_center_right = _mm_maddubs_epi16(bot, center_right_w);
    _mm_sub_epi16(
        _mm_add_epi16(
            _mm_add_epi16(cur_left, bot_left),
            _mm_add_epi16(cur_center_right, bot_center_right),
        ),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_gauss_i16(
    cur: __m128i,
    top: __m128i,
    bot: __m128i,
    prev_cur: u8,
    center_right_w: __m128i,
    even_mask: __m128i,
    left_mask: __m128i,
    dc0v: __m128i,
) -> __m128i {
    let left = left_u8x8_to_i16(cur, prev_cur, left_mask);
    let center_right = _mm_maddubs_epi16(cur, center_right_w);
    let top = even_u8x8_to_i16(top, even_mask);
    let bot = even_u8x8_to_i16(bot, even_mask);
    _mm_sub_epi16(
        _mm_add_epi16(_mm_add_epi16(left, center_right), _mm_add_epi16(top, bot)),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_filter_i16<const FILTER: u32>(
    cur: __m128i,
    top: __m128i,
    bot: __m128i,
    prev_cur: u8,
    prev_bot: u8,
    ones: __m128i,
    even_mask: __m128i,
    vstrip_center_right_w: __m128i,
    gauss_center_right_w: __m128i,
    left_mask: __m128i,
    dc0v: __m128i,
) -> __m128i {
    if FILTER == CFL_FLT_TYPE_VSTRIP {
        ac8_420_vstrip_i16(
            cur,
            bot,
            prev_cur,
            prev_bot,
            vstrip_center_right_w,
            left_mask,
            dc0v,
        )
    } else if FILTER == CFL_FLT_TYPE_GAUSS {
        ac8_420_gauss_i16(
            cur,
            top,
            bot,
            prev_cur,
            gauss_center_right_w,
            even_mask,
            left_mask,
            dc0v,
        )
    } else {
        ac8_420_i16(cur, bot, ones, dc0v)
    }
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
fn even_u8x16_to_i16(row: __m256i, even_mask: __m256i) -> __m256i {
    let shuffled = _mm256_shuffle_epi8(row, even_mask);
    let lo = _mm256_castsi256_si128(shuffled);
    let hi = _mm256_extracti128_si256::<1>(shuffled);
    _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn left_u8x16_to_i16(row: __m256i, prev_byte: u8, left_mask: __m128i) -> __m256i {
    let lo = _mm256_castsi256_si128(row);
    let hi = _mm256_extracti128_si256::<1>(row);
    let prev = _mm_set1_epi8(prev_byte as i8);

    let shifted_lo = _mm_alignr_epi8::<15>(lo, prev);
    let shifted_hi = _mm_alignr_epi8::<15>(hi, lo);
    let left_lo = _mm_shuffle_epi8(shifted_lo, left_mask);
    let left_hi = _mm_shuffle_epi8(shifted_hi, left_mask);
    combine_m128(left_lo, left_hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_420_vstrip_i16(
    top: __m256i,
    bot: __m256i,
    prev_top: u8,
    prev_bot: u8,
    center_right_w: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    let top_left = left_u8x16_to_i16(top, prev_top, left_mask);
    let bot_left = left_u8x16_to_i16(bot, prev_bot, left_mask);
    let top_center_right = _mm256_maddubs_epi16(top, center_right_w);
    let bot_center_right = _mm256_maddubs_epi16(bot, center_right_w);
    _mm256_sub_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(top_left, bot_left),
            _mm256_add_epi16(top_center_right, bot_center_right),
        ),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_420_gauss_i16(
    row: __m256i,
    top: __m256i,
    bot: __m256i,
    prev_byte: u8,
    center_right_w: __m256i,
    even_mask: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    // ss_hor=ss_ver=1 GAUSS uses:
    //   left + 4 * center + right + top + bottom - dc
    // where top is clamped to center on 64px vertical boundaries by the caller.
    let left = left_u8x16_to_i16(row, prev_byte, left_mask);
    let center_right = _mm256_maddubs_epi16(row, center_right_w);
    let top = even_u8x16_to_i16(top, even_mask);
    let bot = even_u8x16_to_i16(bot, even_mask);
    _mm256_sub_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(left, center_right),
            _mm256_add_epi16(top, bot),
        ),
        dc0v,
    )
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
    _mm256_sub_epi16(
        _mm256_slli_epi16::<3>(even_u8x16_to_i16(row, even_mask)),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_422_vstrip_i16(
    row: __m256i,
    prev_byte: u8,
    center_right_w: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    let left = left_u8x16_to_i16(row, prev_byte, left_mask);
    let center_right = _mm256_maddubs_epi16(row, center_right_w);
    _mm256_sub_epi16(
        _mm256_slli_epi16::<1>(_mm256_add_epi16(center_right, left)),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_422_uniform_i16(row: __m128i, ones: __m128i, dc0v: __m128i) -> __m128i {
    let sum16 = _mm_maddubs_epi16(row, ones);
    _mm_sub_epi16(_mm_slli_epi16::<2>(sum16), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_422_gauss_i16(row: __m128i, even_mask: __m128i, dc0v: __m128i) -> __m128i {
    _mm_sub_epi16(_mm_slli_epi16::<3>(even_u8x8_to_i16(row, even_mask)), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_422_vstrip_i16(
    row: __m128i,
    prev_byte: u8,
    center_right_w: __m128i,
    left_mask: __m128i,
    dc0v: __m128i,
) -> __m128i {
    let left = left_u8x8_to_i16(row, prev_byte, left_mask);
    let center_right = _mm_maddubs_epi16(row, center_right_w);
    _mm_sub_epi16(_mm_slli_epi16::<1>(_mm_add_epi16(center_right, left)), dc0v)
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
    alpha_abs: __m256i,
    alpha_sign: __m256i,
    dc_v: __m256i,
    zero: __m256i,
) -> __m256i {
    let lo = apply16_i16_ac(
        ac16_444_i16(_mm256_castsi256_si128(src), dc0v),
        alpha_abs,
        alpha_sign,
        dc_v,
        zero,
    );
    let hi = apply16_i16_ac(
        ac16_444_i16(_mm256_extracti128_si256::<1>(src), dc0v),
        alpha_abs,
        alpha_sign,
        dc_v,
        zero,
    );
    combine_m128(lo, hi)
}

#[inline(always)]
fn cfl_ac_420_scalar_filter<const FILTER: u32>(
    y: &[u8],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: i32,
) -> i32 {
    let xl = x << 1;
    let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
    if FILTER == CFL_FLT_TYPE_GAUSS {
        let top = if (cy & 31) == 0 {
            yrow + xl
        } else {
            yrow + xl - ystride
        };
        y[yrow + left] as i32
            + 4 * y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[top] as i32
            + y[yrow + xl + ystride] as i32
            - dc0
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        y[yrow + left] as i32
            + 2 * y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[yrow + left + ystride] as i32
            + 2 * y[yrow + xl + ystride] as i32
            + y[yrow + xl + ystride + 1] as i32
            - dc0
    } else {
        ((y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[yrow + xl + ystride] as i32
            + y[yrow + xl + ystride + 1] as i32)
            << 1)
            - dc0
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac16_420_filter_i16<const FILTER: u32>(
    cur: __m256i,
    top: __m256i,
    bot: __m256i,
    prev_cur: u8,
    prev_bot: u8,
    ones: __m256i,
    even_mask: __m256i,
    vstrip_center_right_w: __m256i,
    gauss_center_right_w: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    if FILTER == CFL_FLT_TYPE_VSTRIP {
        ac16_420_vstrip_i16(
            cur,
            bot,
            prev_cur,
            prev_bot,
            vstrip_center_right_w,
            left_mask,
            dc0v,
        )
    } else if FILTER == CFL_FLT_TYPE_GAUSS {
        ac16_420_gauss_i16(
            cur,
            top,
            bot,
            prev_cur,
            gauss_center_right_w,
            even_mask,
            left_mask,
            dc0v,
        )
    } else {
        ac16_420_i16(cur, bot, ones, dc0v)
    }
}

#[target_feature(enable = "avx2")]
fn cfl_apply_420_8bpc_avx2_impl<const FILTER: u32>(args: CflApply8<'_>) {
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
    let alpha0_abs = alpha_abs_i16(alpha0);
    let alpha1_abs = alpha_abs_i16(alpha1);
    let alpha0_sign = alpha_sign_i16(alpha0);
    let alpha1_sign = alpha_sign_i16(alpha1);
    let dc1v = _mm256_set1_epi16(dc1 as i16);
    let dc2v = _mm256_set1_epi16(dc2 as i16);
    let zero = _mm256_setzero_si256();
    let even_mask = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128, 0, 2, 4, 6, 8,
        10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let vstrip_center_right_w = _mm256_setr_epi8(
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        2, 1,
    );
    let gauss_center_right_w = _mm256_setr_epi8(
        4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
        4, 1,
    );
    let vstrip_left_mask = _mm_setr_epi8(
        0, -128, 2, -128, 4, -128, 6, -128, 8, -128, 10, -128, 12, -128, 14, -128,
    );
    let ones128 = _mm_set1_epi8(1);
    let dc0v128 = _mm_set1_epi16(dc0 as i16);
    let alpha0_abs128 = alpha_abs_i16_128(alpha0);
    let alpha1_abs128 = alpha_abs_i16_128(alpha1);
    let alpha0_sign128 = alpha_sign_i16_128(alpha0);
    let alpha1_sign128 = alpha_sign_i16_128(alpha1);
    let dc1v128 = _mm_set1_epi16(dc1 as i16);
    let dc2v128 = _mm_set1_epi16(dc2 as i16);
    let zero128 = _mm_setzero_si128();
    let even_mask128 = _mm_setr_epi8(
        0, -128, 2, -128, 4, -128, 6, -128, 8, -128, 10, -128, 12, -128, 14, -128,
    );
    let vstrip_center_right_w128 = _mm_setr_epi8(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1);
    let gauss_center_right_w128 = _mm_setr_epi8(4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for cy in 0..ylim {
        let cur = y[yrow..yrow + lfull].as_chunks::<32>().0;
        let top = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
            y[yrow - ystride..yrow - ystride + lfull]
                .as_chunks::<32>()
                .0
        } else {
            cur
        };
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<32>()
            .0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for (i, (((du, dv), yy), (tt, bb))) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let xl = (i * 16) << 1;
                    let prev_cur = if FILTER == CFL_FLT_TYPE_VSTRIP || FILTER == CFL_FLT_TYPE_GAUSS
                    {
                        if (xl & 63) == 0 {
                            y[yrow + xl]
                        } else {
                            y[yrow + xl - 1]
                        }
                    } else {
                        0
                    };
                    let prev_bot = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        if (xl & 63) == 0 {
                            y[yrow + ystride + xl]
                        } else {
                            y[yrow + ystride + xl - 1]
                        }
                    } else {
                        0
                    };
                    let ac = ac16_420_filter_i16::<FILTER>(
                        load_u8x32(yy),
                        load_u8x32(tt),
                        load_u8x32(bb),
                        prev_cur,
                        prev_bot,
                        ones,
                        even_mask,
                        vstrip_center_right_w,
                        gauss_center_right_w,
                        vstrip_left_mask,
                        dc0v,
                    );
                    store_u8x16(du, apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero));
                    store_u8x16(dv, apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero));
                }
            }
            (true, false) => {
                for (i, ((d, yy), (tt, bb))) in u[urow..urow + xfull]
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let xl = (i * 16) << 1;
                    let prev_cur = if FILTER == CFL_FLT_TYPE_VSTRIP || FILTER == CFL_FLT_TYPE_GAUSS
                    {
                        if (xl & 63) == 0 {
                            y[yrow + xl]
                        } else {
                            y[yrow + xl - 1]
                        }
                    } else {
                        0
                    };
                    let prev_bot = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        if (xl & 63) == 0 {
                            y[yrow + ystride + xl]
                        } else {
                            y[yrow + ystride + xl - 1]
                        }
                    } else {
                        0
                    };
                    let ac = ac16_420_filter_i16::<FILTER>(
                        load_u8x32(yy),
                        load_u8x32(tt),
                        load_u8x32(bb),
                        prev_cur,
                        prev_bot,
                        ones,
                        even_mask,
                        vstrip_center_right_w,
                        gauss_center_right_w,
                        vstrip_left_mask,
                        dc0v,
                    );
                    store_u8x16(d, apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero));
                }
            }
            (false, true) => {
                for (i, ((d, yy), (tt, bb))) in v[vrow..vrow + xfull]
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let xl = (i * 16) << 1;
                    let prev_cur = if FILTER == CFL_FLT_TYPE_VSTRIP || FILTER == CFL_FLT_TYPE_GAUSS
                    {
                        if (xl & 63) == 0 {
                            y[yrow + xl]
                        } else {
                            y[yrow + xl - 1]
                        }
                    } else {
                        0
                    };
                    let prev_bot = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        if (xl & 63) == 0 {
                            y[yrow + ystride + xl]
                        } else {
                            y[yrow + ystride + xl - 1]
                        }
                    } else {
                        0
                    };
                    let ac = ac16_420_filter_i16::<FILTER>(
                        load_u8x32(yy),
                        load_u8x32(tt),
                        load_u8x32(bb),
                        prev_cur,
                        prev_bot,
                        ones,
                        even_mask,
                        vstrip_center_right_w,
                        gauss_center_right_w,
                        vstrip_left_mask,
                        dc0v,
                    );
                    store_u8x16(d, apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero));
                }
            }
            (false, false) => unreachable!(),
        }

        let mut xtail = xfull;
        if xlim - xtail >= 8 {
            let xl = xtail << 1;
            let yy = &y[yrow + xl..yrow + xl + 16].as_chunks::<16>().0[0];
            let tt = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
                &y[yrow - ystride + xl..yrow - ystride + xl + 16]
                    .as_chunks::<16>()
                    .0[0]
            } else {
                yy
            };
            let bb = &y[yrow + ystride + xl..yrow + ystride + xl + 16]
                .as_chunks::<16>()
                .0[0];
            let prev_cur = if FILTER == CFL_FLT_TYPE_VSTRIP || FILTER == CFL_FLT_TYPE_GAUSS {
                if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                }
            } else {
                0
            };
            let prev_bot = if FILTER == CFL_FLT_TYPE_VSTRIP {
                if (xl & 63) == 0 {
                    y[yrow + ystride + xl]
                } else {
                    y[yrow + ystride + xl - 1]
                }
            } else {
                0
            };
            let ac = ac8_420_filter_i16::<FILTER>(
                load_u8x16(yy),
                load_u8x16(tt),
                load_u8x16(bb),
                prev_cur,
                prev_bot,
                ones128,
                even_mask128,
                vstrip_center_right_w128,
                gauss_center_right_w128,
                vstrip_left_mask,
                dc0v128,
            );
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 8].as_chunks_mut::<8>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                    store_u8x8(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 8;
        }

        if xlim - xtail >= 4 {
            let xl = xtail << 1;
            let yy = &y[yrow + xl..yrow + xl + 8].as_chunks::<8>().0[0];
            let tt = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
                &y[yrow - ystride + xl..yrow - ystride + xl + 8]
                    .as_chunks::<8>()
                    .0[0]
            } else {
                yy
            };
            let bb = &y[yrow + ystride + xl..yrow + ystride + xl + 8]
                .as_chunks::<8>()
                .0[0];
            let prev_cur = if FILTER == CFL_FLT_TYPE_VSTRIP || FILTER == CFL_FLT_TYPE_GAUSS {
                if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                }
            } else {
                0
            };
            let prev_bot = if FILTER == CFL_FLT_TYPE_VSTRIP {
                if (xl & 63) == 0 {
                    y[yrow + ystride + xl]
                } else {
                    y[yrow + ystride + xl - 1]
                }
            } else {
                0
            };
            let ac = ac8_420_filter_i16::<FILTER>(
                load_u8x8(yy),
                load_u8x8(tt),
                load_u8x8(bb),
                prev_cur,
                prev_bot,
                ones128,
                even_mask128,
                vstrip_center_right_w128,
                gauss_center_right_w128,
                vstrip_left_mask,
                dc0v128,
            );
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                    store_u8x4(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 4;
        }

        for x in xtail..xlim {
            let ac = cfl_ac_420_scalar_filter::<FILTER>(y, yrow, ystride, cy, x, dc0);
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

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_apply_420_8bpc_avx2(args: CflApply8<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => cfl_apply_420_8bpc_avx2_impl::<CFL_FLT_TYPE_VSTRIP>(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_420_8bpc_avx2_impl::<CFL_FLT_TYPE_GAUSS>(args),
        _ => cfl_apply_420_8bpc_avx2_impl::<0>(args),
    }
}

#[inline(always)]
fn cfl_ac_422_scalar_filter<const FILTER: u32>(y: &[u8], yrow: usize, x: usize, dc0: i32) -> i32 {
    let xl = x << 1;
    if FILTER == CFL_FLT_TYPE_GAUSS {
        ((y[yrow + xl] as i32) << 3) - dc0
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
        (y[yrow + left] as i32 + 2 * y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) * 2 - dc0
    } else {
        ((y[yrow + xl] as i32 + y[yrow + xl + 1] as i32) << 2) - dc0
    }
}

#[target_feature(enable = "avx2")]
fn cfl_apply_422_8bpc_avx2_impl<const FILTER: u32>(args: CflApply8<'_>) {
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
    let alpha0_abs = alpha_abs_i16(alpha0);
    let alpha1_abs = alpha_abs_i16(alpha1);
    let alpha0_sign = alpha_sign_i16(alpha0);
    let alpha1_sign = alpha_sign_i16(alpha1);
    let dc1v = _mm256_set1_epi16(dc1 as i16);
    let dc2v = _mm256_set1_epi16(dc2 as i16);
    let zero = _mm256_setzero_si256();
    let even_mask = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128, 0, 2, 4, 6, 8,
        10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    );
    let vstrip_center_right_w = _mm256_setr_epi8(
        2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        2, 1,
    );
    let vstrip_left_mask = _mm_setr_epi8(
        0, -128, 2, -128, 4, -128, 6, -128, 8, -128, 10, -128, 12, -128, 14, -128,
    );
    let ones128 = _mm_set1_epi8(1);
    let dc0v128 = _mm_set1_epi16(dc0 as i16);
    let alpha0_abs128 = alpha_abs_i16_128(alpha0);
    let alpha1_abs128 = alpha_abs_i16_128(alpha1);
    let alpha0_sign128 = alpha_sign_i16_128(alpha0);
    let alpha1_sign128 = alpha_sign_i16_128(alpha1);
    let dc1v128 = _mm_set1_epi16(dc1 as i16);
    let dc2v128 = _mm_set1_epi16(dc2 as i16);
    let zero128 = _mm_setzero_si128();
    let even_mask128 = _mm_setr_epi8(
        0, -128, 2, -128, 4, -128, 6, -128, 8, -128, 10, -128, 12, -128, 14, -128,
    );
    let vstrip_center_right_w128 = _mm_setr_epi8(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1);
    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let row = y[yrow..yrow + lfull].as_chunks::<32>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for (i, ((du, dv), yy)) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(row.iter())
                    .enumerate()
                {
                    let yy = load_u8x32(yy);
                    let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        let x = (i * 16) << 1;
                        let prev = if (x & 63) == 0 {
                            y[yrow + x]
                        } else {
                            y[yrow + x - 1]
                        };
                        ac16_422_vstrip_i16(yy, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
                    } else if FILTER == CFL_FLT_TYPE_GAUSS {
                        ac16_422_gauss_i16(yy, even_mask, dc0v)
                    } else {
                        ac16_422_uniform_i16(yy, ones, dc0v)
                    };
                    store_u8x16(du, apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero));
                    store_u8x16(dv, apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                for (i, (du, yy)) in u_chunks.iter_mut().zip(row.iter()).enumerate() {
                    let yy = load_u8x32(yy);
                    let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        let x = (i * 16) << 1;
                        let prev = if (x & 63) == 0 {
                            y[yrow + x]
                        } else {
                            y[yrow + x - 1]
                        };
                        ac16_422_vstrip_i16(yy, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
                    } else if FILTER == CFL_FLT_TYPE_GAUSS {
                        ac16_422_gauss_i16(yy, even_mask, dc0v)
                    } else {
                        ac16_422_uniform_i16(yy, ones, dc0v)
                    };
                    store_u8x16(du, apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;
                for (i, (dv, yy)) in v_chunks.iter_mut().zip(row.iter()).enumerate() {
                    let yy = load_u8x32(yy);
                    let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        let x = (i * 16) << 1;
                        let prev = if (x & 63) == 0 {
                            y[yrow + x]
                        } else {
                            y[yrow + x - 1]
                        };
                        ac16_422_vstrip_i16(yy, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
                    } else if FILTER == CFL_FLT_TYPE_GAUSS {
                        ac16_422_gauss_i16(yy, even_mask, dc0v)
                    } else {
                        ac16_422_uniform_i16(yy, ones, dc0v)
                    };
                    store_u8x16(dv, apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero));
                }
            }
            (false, false) => unreachable!(),
        }

        let mut xtail = xfull;
        if xlim - xtail >= 8 {
            let xl = xtail << 1;
            let yy = &y[yrow + xl..yrow + xl + 16].as_chunks::<16>().0[0];
            let yy = load_u8x16(yy);
            let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                let prev = if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                };
                ac8_422_vstrip_i16(
                    yy,
                    prev,
                    vstrip_center_right_w128,
                    vstrip_left_mask,
                    dc0v128,
                )
            } else if FILTER == CFL_FLT_TYPE_GAUSS {
                ac8_422_gauss_i16(yy, even_mask128, dc0v128)
            } else {
                ac8_422_uniform_i16(yy, ones128, dc0v128)
            };
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 8].as_chunks_mut::<8>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                    store_u8x8(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 8;
        }

        if xlim - xtail >= 4 {
            let xl = xtail << 1;
            let yy = &y[yrow + xl..yrow + xl + 8].as_chunks::<8>().0[0];
            let yy = load_u8x8(yy);
            let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                let prev = if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                };
                ac8_422_vstrip_i16(
                    yy,
                    prev,
                    vstrip_center_right_w128,
                    vstrip_left_mask,
                    dc0v128,
                )
            } else if FILTER == CFL_FLT_TYPE_GAUSS {
                ac8_422_gauss_i16(yy, even_mask128, dc0v128)
            } else {
                ac8_422_uniform_i16(yy, ones128, dc0v128)
            };
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                    store_u8x4(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 4;
        }

        for x in xtail..xlim {
            let ac = cfl_ac_422_scalar_filter::<FILTER>(y, yrow, x, dc0);
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
    let alpha0_abs = alpha_abs_i16(alpha0);
    let alpha1_abs = alpha_abs_i16(alpha1);
    let alpha0_sign = alpha_sign_i16(alpha0);
    let alpha1_sign = alpha_sign_i16(alpha1);
    let dc1v = _mm256_set1_epi16(dc1 as i16);
    let dc2v = _mm256_set1_epi16(dc2 as i16);
    let zero = _mm256_setzero_si256();

    let alpha0_abs128 = alpha_abs_i16_128(alpha0);
    let alpha1_abs128 = alpha_abs_i16_128(alpha1);
    let alpha0_sign128 = alpha_sign_i16_128(alpha0);
    let alpha1_sign128 = alpha_sign_i16_128(alpha1);
    let dc1v128 = _mm_set1_epi16(dc1 as i16);
    let dc2v128 = _mm_set1_epi16(dc2 as i16);
    let zero128 = _mm_setzero_si128();

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
                    store_u8x32(
                        du,
                        apply32_444_i16_ac(yy, dc0v, alpha0_abs, alpha0_sign, dc1v, zero),
                    );
                    store_u8x32(
                        dv,
                        apply32_444_i16_ac(yy, dc0v, alpha1_abs, alpha1_sign, dc2v, zero),
                    );
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<32>().0;
                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    store_u8x32(
                        du,
                        apply32_444_i16_ac(
                            load_u8x32(yy),
                            dc0v,
                            alpha0_abs,
                            alpha0_sign,
                            dc1v,
                            zero,
                        ),
                    );
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<32>().0;
                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    store_u8x32(
                        dv,
                        apply32_444_i16_ac(
                            load_u8x32(yy),
                            dc0v,
                            alpha1_abs,
                            alpha1_sign,
                            dc2v,
                            zero,
                        ),
                    );
                }
            }
            (false, false) => unreachable!(),
        }

        let mut xtail = xfull;

        if xlim - xtail >= 16 {
            let yy = &y[yrow + xtail..yrow + xtail + 16].as_chunks::<16>().0[0];
            let ac = ac16_444_i16(load_u8x16(yy), dc0v);
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 16].as_chunks_mut::<16>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 16].as_chunks_mut::<16>();
                    store_u8x16(
                        &mut du[0],
                        apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero),
                    );
                    store_u8x16(
                        &mut dv[0],
                        apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 16].as_chunks_mut::<16>();
                    store_u8x16(
                        &mut du[0],
                        apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 16].as_chunks_mut::<16>();
                    store_u8x16(
                        &mut dv[0],
                        apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 16;
        }

        if xlim - xtail >= 8 {
            let yy = &y[yrow + xtail..yrow + xtail + 8].as_chunks::<8>().0[0];
            let ac = ac16_444_i16(load_u8x8(yy), dc0v);
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 8].as_chunks_mut::<8>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut du[0],
                        apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero),
                    );
                    store_u8x8(
                        &mut dv[0],
                        apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut du[0],
                        apply16_i16_ac(ac, alpha0_abs, alpha0_sign, dc1v, zero),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 8].as_chunks_mut::<8>();
                    store_u8x8(
                        &mut dv[0],
                        apply16_i16_ac(ac, alpha1_abs, alpha1_sign, dc2v, zero),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 8;
        }

        if xlim - xtail >= 4 {
            let ac = _mm256_castsi256_si128(ac16_444_i16(load_u8x4_tail(&y[yrow + xtail..]), dc0v));
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                    store_u8x4(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut du[0],
                        apply8_i16_ac(ac, alpha0_abs128, alpha0_sign128, dc1v128, zero128),
                    );
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_u8x4(
                        &mut dv[0],
                        apply8_i16_ac(ac, alpha1_abs128, alpha1_sign128, dc2v128, zero128),
                    );
                }
                (false, false) => unreachable!(),
            }
            xtail += 4;
        }

        for x in xtail..xlim {
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
        CFL_FLT_TYPE_VSTRIP => cfl_apply_422_8bpc_avx2_impl::<CFL_FLT_TYPE_VSTRIP>(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_422_8bpc_avx2_impl::<CFL_FLT_TYPE_GAUSS>(args),
        _ => cfl_apply_422_8bpc_avx2_impl::<0>(args),
    }
}
