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

use crate::cfl_dispatch::CflApplyHbd;
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

#[inline(always)]
fn load_u16x2_422_tail(src: &[u16]) -> __m256i {
    debug_assert!(src.len() >= 2);
    let mut tmp = [0u16; 16];
    tmp[0] = src[0];
    tmp[1] = src[1];
    unsafe { load_u16x16(&tmp) }
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
