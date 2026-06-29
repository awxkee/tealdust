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

use core::arch::x86_64::*;

use crate::cfl_dispatch::{CflAlphaAccumHbd, CflApplyHbd, CflGenMatHbd, CflMhccpPredHbd};
const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

#[inline(always)]
fn pad_bottom(plane: &mut [u16], row0: usize, stride: usize, w: usize, h: usize, ylim: usize) {
    debug_assert_ne!(ylim, 0);
    let src = row0 + (ylim - 1) * stride;
    for yy in ylim..h {
        let dst = row0 + yy * stride;
        plane.copy_within(src..src + w, dst);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8(a: &[u16; 8]) -> __m256i {
    _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), unsafe {
        _mm_loadu_si128(a.as_ptr() as *const __m128i)
    })
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x16(a: &[u16; 16]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8_i32(a: &[u16; 8]) -> __m256i {
    _mm256_cvtepu16_epi32(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_i32(a: &[u16; 4]) -> __m256i {
    _mm256_cvtepu16_epi32(unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) })
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_tail(src: &[u16]) -> __m256i {
    debug_assert!(src.len() >= 4);

    let q = unsafe { _mm_loadu_si64(src.as_ptr().cast()) };
    _mm256_cvtepu16_epi32(q)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x2_i32_tail(src: &[u16]) -> __m256i {
    debug_assert!(src.len() >= 2);

    let q = unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) };
    _mm256_cvtepu16_epi32(q)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x4_u16_clip(a: &mut [u16; 4], v: __m256i, max_v: __m256i) {
    let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v);
    let p = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm_setzero_si128());
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, p) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8_u16_clip(a: &mut [u16; 8], v: __m256i, max_v: __m256i) {
    let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_packus_epi32(lo, hi);
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, p) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x2_u16_clip(a: &mut [u16], v: __m256i, max_v: __m256i) {
    debug_assert!(a.len() >= 2);
    let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v);
    let p = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm_setzero_si128());
    unsafe {
        _mm_store_ss(a.as_mut_ptr().cast(), _mm_castsi128_ps(p));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x1_u16_clip(a: &mut u16, v: __m256i, max_v: __m256i) {
    let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v);
    let p = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm_setzero_si128());
    unsafe {
        _mm_storeu_si16(a as *mut u16 as *mut u8, p);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x2_422_tail(src: &[u16]) -> __m256i {
    debug_assert!(src.len() >= 2);
    let lo = unsafe { _mm_castps_si128(_mm_load_ss(src.as_ptr().cast())) };
    _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), lo)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_i32(top: __m256i, bot: __m256i, ones: __m256i, dc0v: __m256i) -> __m256i {
    let top = _mm256_madd_epi16(top, ones);
    let bot = _mm256_madd_epi16(bot, ones);
    _mm256_sub_epi32(_mm256_slli_epi32::<1>(_mm256_add_epi32(top, bot)), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn even_u16x8_to_i32(src: __m256i, even_mask: __m256i) -> __m256i {
    let even = _mm256_shuffle_epi8(src, even_mask);
    let even = _mm256_permute4x64_epi64::<0xd8>(even);
    _mm256_cvtepu16_epi32(_mm256_castsi256_si128(even))
}

#[inline]
#[target_feature(enable = "avx2")]
fn left_u16x8_to_i32(src: __m256i, prev_sample: u16, left_mask: __m128i) -> __m256i {
    let lo = _mm256_castsi256_si128(src);
    let hi = _mm256_extracti128_si256::<1>(src);
    let prev = _mm_set1_epi16(prev_sample as i16);

    let shifted_lo = _mm_alignr_epi8::<14>(lo, prev);
    let shifted_hi = _mm_alignr_epi8::<14>(hi, lo);
    let left_lo = _mm_shuffle_epi8(shifted_lo, left_mask);
    let left_hi = _mm_shuffle_epi8(shifted_hi, left_mask);
    _mm256_cvtepu16_epi32(_mm_unpacklo_epi64(left_lo, left_hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_vstrip_i32(
    cur: __m256i,
    bot: __m256i,
    prev_cur: u16,
    prev_bot: u16,
    center_right_w: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    let cur_left = left_u16x8_to_i32(cur, prev_cur, left_mask);
    let bot_left = left_u16x8_to_i32(bot, prev_bot, left_mask);
    let cur_center_right = _mm256_madd_epi16(cur, center_right_w);
    let bot_center_right = _mm256_madd_epi16(bot, center_right_w);
    _mm256_sub_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(cur_left, bot_left),
            _mm256_add_epi32(cur_center_right, bot_center_right),
        ),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_420_gauss_i32(
    cur: __m256i,
    top: __m256i,
    bot: __m256i,
    prev_cur: u16,
    center_right_w: __m256i,
    even_mask: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    // ss_hor=ss_ver=1 GAUSS uses:
    //   left + 4 * center + right + top + bottom - dc
    // with top clamped to the current row at 64-luma vertical boundaries.
    let left = left_u16x8_to_i32(cur, prev_cur, left_mask);
    let center_right = _mm256_madd_epi16(cur, center_right_w);
    let top = even_u16x8_to_i32(top, even_mask);
    let bot = even_u16x8_to_i32(bot, even_mask);
    _mm256_sub_epi32(
        _mm256_add_epi32(
            _mm256_add_epi32(left, center_right),
            _mm256_add_epi32(top, bot),
        ),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_422_uniform_i32(src: __m256i, ones: __m256i, dc0v: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_slli_epi32::<2>(_mm256_madd_epi16(src, ones)), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_422_gauss_i32(src: __m256i, dc0v: __m256i) -> __m256i {
    let mask = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13, -1, -1,
        -1, -1, -1, -1, -1, -1,
    );
    let even = _mm256_shuffle_epi8(src, mask);
    let even = _mm256_permute4x64_epi64::<0xd8>(even);
    let y = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(even));
    _mm256_sub_epi32(_mm256_slli_epi32::<3>(y), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_422_vstrip_i32(
    src: __m256i,
    prev_sample: u16,
    center_right_w: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    let lo = _mm256_castsi256_si128(src);
    let hi = _mm256_extracti128_si256::<1>(src);
    let prev = _mm_set1_epi16(prev_sample as i16);

    // Shift by one u16 while preserving the 128-bit lane boundary. The upper
    // lane receives the low lane's last sample, which is the left neighbor of
    // its first pair.
    let shifted_lo = _mm_alignr_epi8::<14>(lo, prev);
    let shifted_hi = _mm_alignr_epi8::<14>(hi, lo);
    let left_lo = _mm_shuffle_epi8(shifted_lo, left_mask);
    let left_hi = _mm_shuffle_epi8(shifted_hi, left_mask);
    let left = _mm_unpacklo_epi64(left_lo, left_hi);
    let left = _mm256_cvtepu16_epi32(left);

    let center_right = _mm256_madd_epi16(src, center_right_w);
    _mm256_sub_epi32(
        _mm256_slli_epi32::<1>(_mm256_add_epi32(center_right, left)),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn ac8_444_i32(src: __m256i, dc0v: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_slli_epi32::<3>(src), dc0v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_i32x8_i16_n(ac: __m256i, alpha: i32) -> __m256i {
    let lo = _mm256_castsi256_si128(ac);
    let hi = _mm256_extracti128_si256::<1>(ac);
    let ac16 = _mm_packs_epi32(lo, hi);
    let zero = _mm_setzero_si128();
    let loz = _mm_unpacklo_epi16(ac16, zero);
    let hiz = _mm_unpackhi_epi16(ac16, zero);
    let acz = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(loz), hiz);
    let av = _mm256_set1_epi32((alpha as i16 as u16) as i32);
    _mm256_madd_epi16(acz, av)
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply8_i32_ac(ac: __m256i, alpha: i32, dc_v: __m256i) -> __m256i {
    let diff = mul_i32x8_i16_n(ac, alpha);
    let mag = _mm256_srai_epi32::<11>(_mm256_add_epi32(
        _mm256_abs_epi32(diff),
        _mm256_set1_epi32(1024),
    ));
    let sign = _mm256_srai_epi32::<31>(diff);
    let signed = _mm256_sub_epi32(_mm256_xor_si256(mag, sign), sign);
    _mm256_add_epi32(dc_v, signed)
}

#[inline(always)]
fn cfl_ac_420_hbd_scalar_filter<const FILTER: u32>(
    y: &[u16],
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
fn ac8_420_filter_i32<const FILTER: u32>(
    cur: __m256i,
    top: __m256i,
    bot: __m256i,
    prev_cur: u16,
    prev_bot: u16,
    ones: __m256i,
    even_mask: __m256i,
    vstrip_center_right_w: __m256i,
    gauss_center_right_w: __m256i,
    left_mask: __m128i,
    dc0v: __m256i,
) -> __m256i {
    if FILTER == CFL_FLT_TYPE_VSTRIP {
        ac8_420_vstrip_i32(
            cur,
            bot,
            prev_cur,
            prev_bot,
            vstrip_center_right_w,
            left_mask,
            dc0v,
        )
    } else if FILTER == CFL_FLT_TYPE_GAUSS {
        ac8_420_gauss_i32(
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
        ac8_420_i32(cur, bot, ones, dc0v)
    }
}

#[target_feature(enable = "avx2")]
fn cfl_apply_420_hbd_avx2_impl<const FILTER: u32>(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
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

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    let ones = _mm256_set1_epi16(1);
    let vstrip_center_right_w = _mm256_setr_epi16(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1);
    let gauss_center_right_w = _mm256_setr_epi16(4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1);
    let even_mask = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9, 12, 13, -1, -1,
        -1, -1, -1, -1, -1, -1,
    );
    let vstrip_left_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let dc0v = _mm256_set1_epi32(dc0);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let max_v = _mm256_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for cy in 0..ylim {
        let cur = y[yrow..yrow + lfull].as_chunks::<16>().0;
        let top = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
            y[yrow - ystride..yrow - ystride + lfull]
                .as_chunks::<16>()
                .0
        } else {
            cur
        };
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<16>()
            .0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (i, (((du, dv), yy), (tt, bb))) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let xl = (i * 8) << 1;
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
                    let ac = ac8_420_filter_i32::<FILTER>(
                        load_u16x16(yy),
                        load_u16x16(tt),
                        load_u16x16(bb),
                        prev_cur,
                        prev_bot,
                        ones,
                        even_mask,
                        vstrip_center_right_w,
                        gauss_center_right_w,
                        vstrip_left_mask,
                        dc0v,
                    );
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                for (i, ((du, yy), (tt, bb))) in u_chunks
                    .iter_mut()
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let xl = (i * 8) << 1;
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
                    let ac = ac8_420_filter_i32::<FILTER>(
                        load_u16x16(yy),
                        load_u16x16(tt),
                        load_u16x16(bb),
                        prev_cur,
                        prev_bot,
                        ones,
                        even_mask,
                        vstrip_center_right_w,
                        gauss_center_right_w,
                        vstrip_left_mask,
                        dc0v,
                    );
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (i, ((dv, yy), (tt, bb))) in v_chunks
                    .iter_mut()
                    .zip(cur.iter())
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let xl = (i * 8) << 1;
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
                    let ac = ac8_420_filter_i32::<FILTER>(
                        load_u16x16(yy),
                        load_u16x16(tt),
                        load_u16x16(bb),
                        prev_cur,
                        prev_bot,
                        ones,
                        even_mask,
                        vstrip_center_right_w,
                        gauss_center_right_w,
                        vstrip_left_mask,
                        dc0v,
                    );
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (false, false) => unreachable!(),
        }

        let mut xtail = xfull;
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
            let ac = ac8_420_filter_i32::<FILTER>(
                load_u16x8(yy),
                load_u16x8(tt),
                load_u16x8(bb),
                prev_cur,
                prev_bot,
                ones,
                even_mask,
                vstrip_center_right_w,
                gauss_center_right_w,
                vstrip_left_mask,
                dc0v,
            );
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_i32x4_u16_clip(&mut du[0], apply8_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x4_u16_clip(&mut dv[0], apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    store_i32x4_u16_clip(&mut du[0], apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_i32x4_u16_clip(&mut dv[0], apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
                (false, false) => unreachable!(),
            }
            xtail += 4;
        }

        if xlim - xtail >= 2 {
            let xl = xtail << 1;
            let yy = load_u16x4_tail(&y[yrow + xl..]);
            let tt = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
                load_u16x4_tail(&y[yrow - ystride + xl..])
            } else {
                yy
            };
            let bb = load_u16x4_tail(&y[yrow + ystride + xl..]);
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
            let ac = ac8_420_filter_i32::<FILTER>(
                yy,
                tt,
                bb,
                prev_cur,
                prev_bot,
                ones,
                even_mask,
                vstrip_center_right_w,
                gauss_center_right_w,
                vstrip_left_mask,
                dc0v,
            );
            if do_u {
                store_i32x2_u16_clip(
                    &mut u[urow + xtail..urow + xtail + 2],
                    apply8_i32_ac(ac, alpha0, dc1v),
                    max_v,
                );
            }
            if do_v {
                store_i32x2_u16_clip(
                    &mut v[vrow + xtail..vrow + xtail + 2],
                    apply8_i32_ac(ac, alpha1, dc2v),
                    max_v,
                );
            }
            xtail += 2;
        }

        for x in xtail..xlim {
            let ac = cfl_ac_420_hbd_scalar_filter::<FILTER>(y, yrow, ystride, cy, x, dc0);
            if do_u {
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
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
pub(crate) fn cfl_apply_420_hbd_avx2(args: CflApplyHbd<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => cfl_apply_420_hbd_avx2_impl::<CFL_FLT_TYPE_VSTRIP>(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_420_hbd_avx2_impl::<CFL_FLT_TYPE_GAUSS>(args),
        _ => cfl_apply_420_hbd_avx2_impl::<0>(args),
    }
}

#[target_feature(enable = "avx2")]
fn cfl_apply_422_hbd_avx2_impl<const FILTER: u32>(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
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

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    let ones = _mm256_set1_epi16(1);
    let vstrip_center_right_w = _mm256_setr_epi16(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1);
    let vstrip_left_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let dc0v = _mm256_set1_epi32(dc0);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let max_v = _mm256_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + lfull].as_chunks::<16>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (i, ((du, dv), s)) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(src)
                    .enumerate()
                {
                    let src = load_u16x16(s);
                    let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        let x = (i * 8) << 1;
                        let prev = if (x & 63) == 0 {
                            y[yrow + x]
                        } else {
                            y[yrow + x - 1]
                        };
                        ac8_422_vstrip_i32(src, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
                    } else if FILTER == CFL_FLT_TYPE_GAUSS {
                        ac8_422_gauss_i32(src, dc0v)
                    } else {
                        ac8_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                for (i, (du, s)) in u_chunks.iter_mut().zip(src).enumerate() {
                    let src = load_u16x16(s);
                    let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        let x = (i * 8) << 1;
                        let prev = if (x & 63) == 0 {
                            y[yrow + x]
                        } else {
                            y[yrow + x - 1]
                        };
                        ac8_422_vstrip_i32(src, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
                    } else if FILTER == CFL_FLT_TYPE_GAUSS {
                        ac8_422_gauss_i32(src, dc0v)
                    } else {
                        ac8_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (i, (dv, s)) in v_chunks.iter_mut().zip(src).enumerate() {
                    let src = load_u16x16(s);
                    let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                        let x = (i * 8) << 1;
                        let prev = if (x & 63) == 0 {
                            y[yrow + x]
                        } else {
                            y[yrow + x - 1]
                        };
                        ac8_422_vstrip_i32(src, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
                    } else if FILTER == CFL_FLT_TYPE_GAUSS {
                        ac8_422_gauss_i32(src, dc0v)
                    } else {
                        ac8_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (false, false) => unreachable!(),
        }
        let mut xtail = xfull;
        if xlim - xtail >= 4 {
            let xl = xtail << 1;
            let s = &y[yrow + xl..yrow + xl + 8].as_chunks::<8>().0[0];
            let src = load_u16x8(s);
            let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                let prev = if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                };
                ac8_422_vstrip_i32(src, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
            } else if FILTER == CFL_FLT_TYPE_GAUSS {
                ac8_422_gauss_i32(src, dc0v)
            } else {
                ac8_422_uniform_i32(src, ones, dc0v)
            };
            match (do_u, do_v) {
                (true, true) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_i32x4_u16_clip(&mut du[0], apply8_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x4_u16_clip(&mut dv[0], apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
                (true, false) => {
                    let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                    store_i32x4_u16_clip(&mut du[0], apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
                (false, true) => {
                    let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                    store_i32x4_u16_clip(&mut dv[0], apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
                (false, false) => unreachable!(),
            }
            xtail += 4;
        }

        if xlim - xtail >= 2 {
            let xl = xtail << 1;
            let src = load_u16x4_tail(&y[yrow + xl..]);
            let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                let prev = if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                };
                ac8_422_vstrip_i32(src, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
            } else if FILTER == CFL_FLT_TYPE_GAUSS {
                ac8_422_gauss_i32(src, dc0v)
            } else {
                ac8_422_uniform_i32(src, ones, dc0v)
            };
            if do_u {
                store_i32x2_u16_clip(
                    &mut u[urow + xtail..urow + xtail + 2],
                    apply8_i32_ac(ac, alpha0, dc1v),
                    max_v,
                );
            }
            if do_v {
                store_i32x2_u16_clip(
                    &mut v[vrow + xtail..vrow + xtail + 2],
                    apply8_i32_ac(ac, alpha1, dc2v),
                    max_v,
                );
            }
            xtail += 2;
        }

        if xtail < xlim {
            let xl = xtail << 1;
            let src = load_u16x2_422_tail(&y[yrow + xl..]);
            let ac = if FILTER == CFL_FLT_TYPE_VSTRIP {
                let prev = if (xl & 63) == 0 {
                    y[yrow + xl]
                } else {
                    y[yrow + xl - 1]
                };
                ac8_422_vstrip_i32(src, prev, vstrip_center_right_w, vstrip_left_mask, dc0v)
            } else if FILTER == CFL_FLT_TYPE_GAUSS {
                ac8_422_gauss_i32(src, dc0v)
            } else {
                ac8_422_uniform_i32(src, ones, dc0v)
            };
            if do_u {
                store_i32x1_u16_clip(&mut u[urow + xtail], apply8_i32_ac(ac, alpha0, dc1v), max_v);
            }
            if do_v {
                store_i32x1_u16_clip(&mut v[vrow + xtail], apply8_i32_ac(ac, alpha1, dc2v), max_v);
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
pub(crate) fn cfl_apply_444_hbd_avx2(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
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

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let dc0v = _mm256_set1_epi32(dc0);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let max_v = _mm256_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + xfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let ac = ac8_444_i32(load_u16x8_i32(s), dc0v);
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let ac = ac8_444_i32(load_u16x8_i32(s), dc0v);
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let ac = ac8_444_i32(load_u16x8_i32(s), dc0v);
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (false, false) => unreachable!(),
        }
        let mut xtail = xfull;

        if xlim - xtail >= 4 {
            let yy = &y[yrow + xtail..yrow + xtail + 4].as_chunks::<4>().0[0];
            let ac = ac8_444_i32(load_u16x4_i32(yy), dc0v);
            if do_u {
                let (du, _) = u[urow + xtail..urow + xtail + 4].as_chunks_mut::<4>();
                store_i32x4_u16_clip(
                    du.last_mut().unwrap(),
                    apply8_i32_ac(ac, alpha0, dc1v),
                    max_v,
                );
            }
            if do_v {
                let (dv, _) = v[vrow + xtail..vrow + xtail + 4].as_chunks_mut::<4>();
                store_i32x4_u16_clip(
                    dv.last_mut().unwrap(),
                    apply8_i32_ac(ac, alpha1, dc2v),
                    max_v,
                );
            }
            xtail += 4;
        }

        if xlim - xtail >= 2 {
            let ac = ac8_444_i32(load_u16x2_i32_tail(&y[yrow + xtail..]), dc0v);
            if do_u {
                store_i32x2_u16_clip(
                    &mut u[urow + xtail..urow + xtail + 2],
                    apply8_i32_ac(ac, alpha0, dc1v),
                    max_v,
                );
            }
            if do_v {
                store_i32x2_u16_clip(
                    &mut v[vrow + xtail..vrow + xtail + 2],
                    apply8_i32_ac(ac, alpha1, dc2v),
                    max_v,
                );
            }
            xtail += 2;
        }

        for x in xtail..xlim {
            let ac = ((y[yrow + x] as i32) << 3) - dc0;
            if do_u {
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
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
pub(crate) fn cfl_apply_422_hbd_avx2(args: CflApplyHbd<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => cfl_apply_422_hbd_avx2_impl::<CFL_FLT_TYPE_VSTRIP>(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_422_hbd_avx2_impl::<CFL_FLT_TYPE_GAUSS>(args),
        _ => cfl_apply_422_hbd_avx2_impl::<0>(args),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn reduce_i32x8(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256::<1>(v);
    let lo = _mm256_castsi256_si128(v);
    let sum = _mm_add_epi32(lo, hi);
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b1110_1110>(sum));
    let sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b0101_0101>(sum));
    _mm_cvtsi128_si32(sum)
}

#[inline(always)]
fn load_strided_u16x16(samples: &[u16], mut off: usize, stride: usize) -> [u16; 16] {
    let mut tmp = [0u16; 16];
    for dst in &mut tmp {
        *dst = samples[off];
        off += stride;
    }
    tmp
}

#[inline(always)]
fn load_strided_u16x8(samples: &[u16], mut off: usize, stride: usize) -> [u16; 8] {
    let mut tmp = [0u16; 8];
    for dst in &mut tmp {
        *dst = samples[off];
        off += stride;
    }
    tmp
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16_slice_x8_i32(a: &[u16]) -> __m256i {
    debug_assert!(a.len() >= 8);
    _mm256_cvtepu16_epi32(unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) })
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8_u16(a: &mut [u16; 8], v: __m256i) {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_packus_epi32(lo, hi);
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, p) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn mhccp_sqrnd_u16x8_avx2(v: __m256i, bitdepth: i32) -> __m256i {
    _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(v, v),
            _mm256_set1_epi32(1 << (bitdepth - 1)),
        ),
        _mm_cvtsi32_si128(bitdepth),
    )
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_gen_mat_hbd_avx2(args: CflGenMatHbd<'_>) {
    if args.len < 8 {
        crate::cfl_dispatch::cfl_gen_mat_hbd_scalar(args);
        return;
    }

    let CflGenMatHbd {
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
        bitdepth,
    } = args;

    let mut acc00 = _mm256_setzero_si256();
    let mut acc01 = _mm256_setzero_si256();
    let mut acc0 = _mm256_setzero_si256();
    let mut acc11 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let chunks = len / 8;
    let processed = chunks * 8;

    for chunk_idx in 0..chunks {
        let rel = chunk_idx * 8;
        let v0 = if v0_stride == 1 {
            load_u16_slice_x8_i32(&y[v0_off + rel..])
        } else {
            let tmp = load_strided_u16x8(y, v0_off + rel * v0_stride, v0_stride);
            load_u16x8_i32(&tmp)
        };
        let raw1 = if v1_stride == 1 {
            load_u16_slice_x8_i32(&y[v1_off + rel..])
        } else {
            let tmp = load_strided_u16x8(y, v1_off + rel * v1_stride, v1_stride);
            load_u16x8_i32(&tmp)
        };
        let v1 = mhccp_sqrnd_u16x8_avx2(raw1, bitdepth);

        acc00 = _mm256_add_epi32(acc00, _mm256_mullo_epi32(v0, v0));
        acc01 = _mm256_add_epi32(acc01, _mm256_mullo_epi32(v0, v1));
        acc0 = _mm256_add_epi32(acc0, v0);
        acc11 = _mm256_add_epi32(acc11, _mm256_mullo_epi32(v1, v1));
        acc1 = _mm256_add_epi32(acc1, v1);

        let out = imat_off + rel;
        let dst0: &mut [u16; 8] = (&mut imat0[out..out + 8]).try_into().unwrap();
        let dst1: &mut [u16; 8] = (&mut imat1[out..out + 8]).try_into().unwrap();
        store_i32x8_u16(dst0, v0);
        store_i32x8_u16(dst1, v1);
    }

    sums.m00 += reduce_i32x8(acc00);
    sums.m01 += reduce_i32x8(acc01);
    sums.sum0 += reduce_i32x8(acc0);
    sums.m11 += reduce_i32x8(acc11);
    sums.sum1 += reduce_i32x8(acc1);

    if processed < len {
        crate::cfl_dispatch::cfl_gen_mat_hbd_scalar(crate::cfl_dispatch::CflGenMatHbd {
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
            bitdepth,
        });
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_alpha_accum_hbd_avx2(args: CflAlphaAccumHbd<'_>) {
    if args.len < 16 {
        crate::cfl_dispatch::cfl_alpha_accum_hbd_scalar(args);
        return;
    }

    let CflAlphaAccumHbd {
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

    let ones = _mm256_set1_epi16(1);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let chunks = len / 16;
    let processed = chunks * 16;

    if sample_stride == 1 {
        let sample_chunks = samples[sample_off..sample_off + processed]
            .as_chunks::<16>()
            .0;
        for (chunk_idx, s) in sample_chunks.iter().enumerate() {
            let i = imat_off + chunk_idx * 16;
            let v = load_u16x16(s);
            let m0 = load_u16x16((&imat0[i..i + 16]).try_into().unwrap());
            let m1 = load_u16x16((&imat1[i..i + 16]).try_into().unwrap());
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(v, m0));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(v, m1));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(v, ones));
        }
    } else {
        let mut off = sample_off;
        for chunk_idx in 0..chunks {
            let i = imat_off + chunk_idx * 16;
            let s = load_strided_u16x16(samples, off, sample_stride);
            off += 16 * sample_stride;
            let v = load_u16x16(&s);
            let m0 = load_u16x16((&imat0[i..i + 16]).try_into().unwrap());
            let m1 = load_u16x16((&imat1[i..i + 16]).try_into().unwrap());
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(v, m0));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(v, m1));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(v, ones));
        }
    }

    alpha[0] += reduce_i32x8(acc0);
    alpha[1] += reduce_i32x8(acc1);
    alpha[2] += reduce_i32x8(acc2) << a2sh;

    if processed < len {
        crate::cfl_dispatch::cfl_alpha_accum_hbd_scalar(crate::cfl_dispatch::CflAlphaAccumHbd {
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
#[target_feature(enable = "avx2")]
fn mhccp_round_signed_shift16_hbd_avx2(v: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let sign = _mm256_cmpgt_epi32(zero, v);
    let mag = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_abs_epi32(v),
        _mm256_set1_epi32(1 << 15),
    ));
    _mm256_sub_epi32(_mm256_xor_si256(mag, sign), sign)
}

#[inline]
#[target_feature(enable = "avx2")]
fn mhccp_mul32_hbd_avx2(v: __m256i, alpha: i32) -> __m256i {
    mhccp_round_signed_shift16_hbd_avx2(_mm256_mullo_epi32(v, _mm256_set1_epi32(alpha)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mhccp_sqrnd_hbd_avx2(v: __m256i, bitdepth: i32) -> __m256i {
    _mm256_sra_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(v, v),
            _mm256_set1_epi32(1 << (bitdepth - 1)),
        ),
        _mm_cvtsi32_si128(bitdepth),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn mhccp_pred_hbd_avx2(
    v0: __m256i,
    v1: __m256i,
    alpha: [i32; 3],
    a2v2: __m256i,
    bitdepth: i32,
) -> __m256i {
    _mm256_add_epi32(
        _mm256_add_epi32(
            mhccp_mul32_hbd_avx2(v0, alpha[0]),
            mhccp_mul32_hbd_avx2(mhccp_sqrnd_hbd_avx2(v1, bitdepth), alpha[1]),
        ),
        a2v2,
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn mhccp_load_u16x16_i32_halves(src: &[u16; 16]) -> (__m256i, __m256i) {
    let v = load_u16x16(src);
    (
        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v)),
        _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x16_u16_clip(a: &mut [u16; 16], lo: __m256i, hi: __m256i, max_v: __m256i) {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_min_epi32(_mm256_max_epi32(lo, zero), max_v);
    let hi = _mm256_min_epi32(_mm256_max_epi32(hi, zero), max_v);
    let p = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(lo, hi));
    unsafe { _mm256_storeu_si256(a.as_mut_ptr() as *mut __m256i, p) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn mhccp_pred_hbd_x16_avx2(
    v0: &[u16; 16],
    v1: &[u16; 16],
    alpha: [i32; 3],
    a2v2: __m256i,
    bitdepth: i32,
) -> (__m256i, __m256i) {
    let (v0_lo, v0_hi) = mhccp_load_u16x16_i32_halves(v0);
    let (v1_lo, v1_hi) = mhccp_load_u16x16_i32_halves(v1);
    (
        mhccp_pred_hbd_avx2(v0_lo, v1_lo, alpha, a2v2, bitdepth),
        mhccp_pred_hbd_avx2(v0_hi, v1_hi, alpha, a2v2, bitdepth),
    )
}

#[inline(always)]
fn mhccp_pred_one_hbd(
    alpha: &[i32; 3],
    a2v2: i32,
    v0: i32,
    v1: i32,
    bitdepth: i32,
    bitdepth_max: i32,
) -> u16 {
    let sq = (v1 * v1 + (1 << (bitdepth - 1))) >> bitdepth;
    (crate::ipred::mul32(alpha[0], v0, 16) + crate::ipred::mul32(alpha[1], sq, 16) + a2v2)
        .clamp(0, bitdepth_max) as u16
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_mhccp_pred_hbd_avx2(args: CflMhccpPredHbd<'_>) {
    if !crate::cfl_dispatch::cfl_mhccp_coeffs_fit_fast_mul(&args.alpha)
        || args.w < 8
        || args.bitdepth > 12
    {
        crate::cfl_dispatch::cfl_mhccp_pred_hbd_scalar(args);
        return;
    }

    let CflMhccpPredHbd {
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
        bitdepth,
        bitdepth_max,
    } = args;
    let has_t = edge_flags & (1 << 2) != 0;
    let has_l = edge_flags & (1 << 3) != 0;
    let dir_t = dir == crate::levels::CflMhDir::Top;
    let dir_l = dir == crate::levels::CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let mid = 1 << (bitdepth - 1);
    let a2v2_scalar = crate::ipred::mul32(alpha[2], mid, 16);
    let a2v2 = _mm256_set1_epi32(a2v2_scalar);
    let max_v = _mm256_set1_epi32(bitdepth_max);

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && y < h {
        let dst_row = &mut dst[..w];
        let (dst16, r16) = dst_row.as_chunks_mut::<16>();
        let prev = sp - src_top_stride;
        for (i, chunk) in dst16.iter_mut().enumerate() {
            let x = i * 16;
            let (lo, hi) = mhccp_pred_hbd_x16_avx2(
                (&src[prev + x..prev + x + 16]).try_into().unwrap(),
                (&src[sp + x..sp + x + 16]).try_into().unwrap(),
                alpha,
                a2v2,
                bitdepth,
            );
            store_i32x16_u16_clip(chunk, lo, hi, max_v);
        }
        let done16 = dst16.len() * 16;
        let (dst8, dst_tail) = r16.as_chunks_mut::<8>();
        for (i, chunk) in dst8.iter_mut().enumerate() {
            let x = done16 + i * 8;
            let out = mhccp_pred_hbd_avx2(
                load_u16x8_i32((&src[prev + x..prev + x + 8]).try_into().unwrap()),
                load_u16x8_i32((&src[sp + x..sp + x + 8]).try_into().unwrap()),
                alpha,
                a2v2,
                bitdepth,
            );
            store_i32x8_u16_clip(chunk, out, max_v);
        }
        let done = done16 + dst8.len() * 8;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            *d = mhccp_pred_one_hbd(
                &alpha,
                a2v2_scalar,
                src[prev + x] as i32,
                src[sp + x] as i32,
                bitdepth,
                bitdepth_max,
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
            dst_row[0] = mhccp_pred_one_hbd(
                &alpha,
                a2v2_scalar,
                v0,
                src[sp] as i32,
                bitdepth,
                bitdepth_max,
            );
            x0 = 1;
        }

        let (dst16, r16) = dst_row[x0..].as_chunks_mut::<16>();
        for (i, chunk) in dst16.iter_mut().enumerate() {
            let x = x0 + i * 16;
            let v0_off = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x - 1
            } else {
                sp + x
            };
            let (lo, hi) = mhccp_pred_hbd_x16_avx2(
                (&src[v0_off..v0_off + 16]).try_into().unwrap(),
                (&src[sp + x..sp + x + 16]).try_into().unwrap(),
                alpha,
                a2v2,
                bitdepth,
            );
            store_i32x16_u16_clip(chunk, lo, hi, max_v);
        }
        let done16 = x0 + dst16.len() * 16;
        let (dst8, dst_tail) = r16.as_chunks_mut::<8>();
        for (i, chunk) in dst8.iter_mut().enumerate() {
            let x = done16 + i * 8;
            let v0_off = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x - 1
            } else {
                sp + x
            };
            let out = mhccp_pred_hbd_avx2(
                load_u16x8_i32((&src[v0_off..v0_off + 8]).try_into().unwrap()),
                load_u16x8_i32((&src[sp + x..sp + x + 8]).try_into().unwrap()),
                alpha,
                a2v2,
                bitdepth,
            );
            store_i32x8_u16_clip(chunk, out, max_v);
        }
        let done = done16 + dst8.len() * 8;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            let v0_idx = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x.saturating_sub(1)
            } else {
                sp + x
            };
            *d = mhccp_pred_one_hbd(
                &alpha,
                a2v2_scalar,
                src[v0_idx] as i32,
                src[sp + x] as i32,
                bitdepth,
                bitdepth_max,
            );
        }
        sp += w;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn gen_y8_hbd_i32<const FILTER: i32>(
    src: &[u16],
    src_off: usize,
    top: &[u16],
    top_off: usize,
    bottom_offset: usize,
    x: usize,
) -> __m256i {
    let xl = x << 1;
    if FILTER == 1 {
        let left_w = _mm256_setr_epi16(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
        let center_right_w = _mm256_setr_epi16(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1);
        let cur_left = load_u16x16(
            src[src_off + xl - 1..src_off + xl - 1 + 16]
                .try_into()
                .unwrap(),
        );
        let cur_center = load_u16x16(src[src_off + xl..src_off + xl + 16].try_into().unwrap());
        let bot_left = load_u16x16(
            src[src_off + bottom_offset + xl - 1..src_off + bottom_offset + xl - 1 + 16]
                .try_into()
                .unwrap(),
        );
        let bot_center = load_u16x16(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 16]
                .try_into()
                .unwrap(),
        );
        let cur = _mm256_add_epi32(
            _mm256_madd_epi16(cur_left, left_w),
            _mm256_madd_epi16(cur_center, center_right_w),
        );
        let bot = _mm256_add_epi32(
            _mm256_madd_epi16(bot_left, left_w),
            _mm256_madd_epi16(bot_center, center_right_w),
        );
        _mm256_srai_epi32::<3>(_mm256_add_epi32(cur, bot))
    } else if FILTER == 2 {
        let left_w = _mm256_setr_epi16(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
        let center_right_w = _mm256_setr_epi16(4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1);
        let center_w = left_w;
        let cur_left = load_u16x16(
            src[src_off + xl - 1..src_off + xl - 1 + 16]
                .try_into()
                .unwrap(),
        );
        let cur_center = load_u16x16(src[src_off + xl..src_off + xl + 16].try_into().unwrap());
        let top_c = load_u16x16(top[top_off + xl..top_off + xl + 16].try_into().unwrap());
        let bot_c = load_u16x16(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 16]
                .try_into()
                .unwrap(),
        );
        let cur = _mm256_add_epi32(
            _mm256_madd_epi16(cur_left, left_w),
            _mm256_madd_epi16(cur_center, center_right_w),
        );
        let tb = _mm256_add_epi32(
            _mm256_madd_epi16(top_c, center_w),
            _mm256_madd_epi16(bot_c, center_w),
        );
        _mm256_srai_epi32::<3>(_mm256_add_epi32(cur, tb))
    } else {
        let ones = _mm256_set1_epi16(1);
        let cur = load_u16x16(src[src_off + xl..src_off + xl + 16].try_into().unwrap());
        let bot = load_u16x16(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 16]
                .try_into()
                .unwrap(),
        );
        _mm256_srai_epi32::<2>(_mm256_add_epi32(
            _mm256_madd_epi16(cur, ones),
            _mm256_madd_epi16(bot, ones),
        ))
    }
}

#[target_feature(enable = "avx2")]
fn cfl_gen_y_row_hbd_avx2_impl<const FILTER: i32>(args: crate::cfl_dispatch::CflGenYRowHbd<'_>) {
    let crate::cfl_dispatch::CflGenYRowHbd {
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type: _,
    } = args;

    let mut processed = 0usize;
    if FILTER != 0 && n_left == 0 && !dst.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_hbd_scalar(crate::cfl_dispatch::CflGenYRowHbd {
            dst: &mut dst[..1],
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left,
            filter_type: FILTER,
        });
        processed = 1;
    }

    let max_v = _mm256_set1_epi32(u16::MAX as i32);
    let (chunks, rem) = dst[processed..].as_chunks_mut::<8>();
    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let x = n_left + processed + chunk_idx * 8;
        store_i32x8_u16_clip(
            chunk,
            gen_y8_hbd_i32::<FILTER>(src, src_off, top, top_off, bottom_offset, x),
            max_v,
        );
    }
    processed += chunks.len() * 8;

    if !rem.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_hbd_scalar(crate::cfl_dispatch::CflGenYRowHbd {
            dst: rem,
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left: n_left + processed,
            filter_type: FILTER,
        });
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_gen_y_row_hbd_avx2(args: crate::cfl_dispatch::CflGenYRowHbd<'_>) {
    match args.filter_type {
        1 => cfl_gen_y_row_hbd_avx2_impl::<1>(args),
        2 => cfl_gen_y_row_hbd_avx2_impl::<2>(args),
        _ => cfl_gen_y_row_hbd_avx2_impl::<0>(args),
    }
}
