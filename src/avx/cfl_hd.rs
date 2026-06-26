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
fn store_i32x8_u16_clip(a: &mut [u16; 8], v: __m256i, max_v: __m256i) {
    let v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max_v);
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let p = _mm_packus_epi32(lo, hi);
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, p) };
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

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_apply_420_hbd_avx2(args: CflApplyHbd<'_>) {
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
    let dc0v = _mm256_set1_epi32(dc0);
    let dc1v = _mm256_set1_epi32(dc1);
    let dc2v = _mm256_set1_epi32(dc2);
    let max_v = _mm256_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let top = y[yrow..yrow + lfull].as_chunks::<16>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<16>()
            .0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (((du, dv), t), b) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(top)
                    .zip(bot)
                {
                    let ac = ac8_420_i32(load_u16x16(t), load_u16x16(b), ones, dc0v);
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                for ((du, t), b) in u_chunks.iter_mut().zip(top).zip(bot) {
                    let ac = ac8_420_i32(load_u16x16(t), load_u16x16(b), ones, dc0v);
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for ((dv, t), b) in v_chunks.iter_mut().zip(top).zip(bot) {
                    let ac = ac8_420_i32(load_u16x16(t), load_u16x16(b), ones, dc0v);
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
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
fn cfl_apply_422_hbd_avx2_impl<const GAUSS: bool>(args: CflApplyHbd<'_>) {
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
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let src = load_u16x16(s);
                    let ac = if GAUSS {
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
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let src = load_u16x16(s);
                    let ac = if GAUSS {
                        ac8_422_gauss_i32(src, dc0v)
                    } else {
                        ac8_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x8_u16_clip(du, apply8_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let src = load_u16x16(s);
                    let ac = if GAUSS {
                        ac8_422_gauss_i32(src, dc0v)
                    } else {
                        ac8_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x8_u16_clip(dv, apply8_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (false, false) => unreachable!(),
        }
        for x in xfull..xlim {
            let ac = crate::cfl_dispatch::cfl_ac_422_hbd_scalar(
                y,
                yrow,
                x,
                dc0,
                if GAUSS { CFL_FLT_TYPE_GAUSS } else { 0 },
            );
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
        for x in xfull..xlim {
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
        CFL_FLT_TYPE_VSTRIP => crate::cfl_dispatch::cfl_apply_422_hbd_scalar(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_422_hbd_avx2_impl::<true>(args),
        _ => cfl_apply_422_hbd_avx2_impl::<false>(args),
    }
}
