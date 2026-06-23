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

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
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
#[target_feature(enable = "sse4.1")]
fn load_u16x8(a: &[u16; 8]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn load_u16x4_i32(a: &[u16; 4]) -> __m128i {
    _mm_cvtepu16_epi32(unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) })
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_u16x4(a: &mut [u16; 4], v: __m128i) {
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x4_u16_clip(a: &mut [u16; 4], v: __m128i, max_v: __m128i) {
    let v = _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), max_v);
    let p = _mm_packus_epi32(v, v);
    store_u16x4(a, p);
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac4_420_i32(top: __m128i, bot: __m128i, ones: __m128i, dc0v: __m128i) -> __m128i {
    let top = _mm_madd_epi16(top, ones);
    let bot = _mm_madd_epi16(bot, ones);
    _mm_sub_epi32(_mm_slli_epi32::<1>(_mm_add_epi32(top, bot)), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac4_422_uniform_i32(src: __m128i, ones: __m128i, dc0v: __m128i) -> __m128i {
    _mm_sub_epi32(_mm_slli_epi32::<2>(_mm_madd_epi16(src, ones)), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac4_422_gauss_i32(src: __m128i, dc0v: __m128i) -> __m128i {
    let mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let even = _mm_shuffle_epi8(src, mask);
    _mm_sub_epi32(_mm_slli_epi32::<3>(_mm_cvtepu16_epi32(even)), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac4_444_i32(src: __m128i, dc0v: __m128i) -> __m128i {
    _mm_sub_epi32(_mm_slli_epi32::<3>(src), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn mul_i32x4_i16_n(ac: __m128i, alpha: i32) -> __m128i {
    // HBD CFL AC is bounded to i16 even for 12-bit: [-(4095*8), +(4095*8)].
    // Use PMADDWD as four independent i16*alpha -> i32 multiplies.
    let ac16 = _mm_packs_epi32(ac, _mm_setzero_si128());
    let acz = _mm_unpacklo_epi16(ac16, _mm_setzero_si128());
    let av = _mm_set1_epi32((alpha as i16 as u16) as i32);
    _mm_madd_epi16(acz, av)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn apply4_i32_ac(ac: __m128i, alpha: i32, dc_v: __m128i) -> __m128i {
    let diff = mul_i32x4_i16_n(ac, alpha);
    let mag = _mm_srai_epi32::<11>(_mm_add_epi32(_mm_abs_epi32(diff), _mm_set1_epi32(1024)));
    let sign = _mm_srai_epi32::<31>(diff);
    let signed = _mm_sub_epi32(_mm_xor_si128(mag, sign), sign);
    _mm_add_epi32(dc_v, signed)
}

#[target_feature(enable = "sse4.1")]
fn cfl_apply_420_hbd_sse41_impl(args: CflApplyHbd<'_>) {
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

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let lfull = nfull * 8;

    let ones = _mm_set1_epi16(1);
    let dc0v = _mm_set1_epi32(dc0);
    let dc1v = _mm_set1_epi32(dc1);
    let dc2v = _mm_set1_epi32(dc2);
    let max_v = _mm_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let top = y[yrow..yrow + lfull].as_chunks::<8>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (((du, dv), t), b) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(top)
                    .zip(bot)
                {
                    let ac = ac4_420_i32(load_u16x8(t), load_u16x8(b), ones, dc0v);
                    store_i32x4_u16_clip(du, apply4_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x4_u16_clip(dv, apply4_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for ((du, t), b) in u_chunks.iter_mut().zip(top).zip(bot) {
                    let ac = ac4_420_i32(load_u16x8(t), load_u16x8(b), ones, dc0v);
                    store_i32x4_u16_clip(du, apply4_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((dv, t), b) in v_chunks.iter_mut().zip(top).zip(bot) {
                    let ac = ac4_420_i32(load_u16x8(t), load_u16x8(b), ones, dc0v);
                    store_i32x4_u16_clip(dv, apply4_i32_ac(ac, alpha1, dc2v), max_v);
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

#[target_feature(enable = "sse4.1")]
fn cfl_apply_422_hbd_sse41_impl<const GAUSS: bool>(args: CflApplyHbd<'_>) {
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

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let lfull = nfull * 8;

    let ones = _mm_set1_epi16(1);
    let dc0v = _mm_set1_epi32(dc0);
    let dc1v = _mm_set1_epi32(dc1);
    let dc2v = _mm_set1_epi32(dc2);
    let max_v = _mm_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + lfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let src = load_u16x8(s);
                    let ac = if GAUSS {
                        ac4_422_gauss_i32(src, dc0v)
                    } else {
                        ac4_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x4_u16_clip(du, apply4_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x4_u16_clip(dv, apply4_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let src = load_u16x8(s);
                    let ac = if GAUSS {
                        ac4_422_gauss_i32(src, dc0v)
                    } else {
                        ac4_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x4_u16_clip(du, apply4_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let src = load_u16x8(s);
                    let ac = if GAUSS {
                        ac4_422_gauss_i32(src, dc0v)
                    } else {
                        ac4_422_uniform_i32(src, ones, dc0v)
                    };
                    store_i32x4_u16_clip(dv, apply4_i32_ac(ac, alpha1, dc2v), max_v);
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

#[target_feature(enable = "sse4.1")]
fn cfl_apply_444_hbd_sse41_impl(args: CflApplyHbd<'_>) {
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

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let dc0v = _mm_set1_epi32(dc0);
    let dc1v = _mm_set1_epi32(dc1);
    let dc2v = _mm_set1_epi32(dc2);
    let max_v = _mm_set1_epi32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + xfull].as_chunks::<4>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let ac = ac4_444_i32(load_u16x4_i32(s), dc0v);
                    store_i32x4_u16_clip(du, apply4_i32_ac(ac, alpha0, dc1v), max_v);
                    store_i32x4_u16_clip(dv, apply4_i32_ac(ac, alpha1, dc2v), max_v);
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let ac = ac4_444_i32(load_u16x4_i32(s), dc0v);
                    store_i32x4_u16_clip(du, apply4_i32_ac(ac, alpha0, dc1v), max_v);
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let ac = ac4_444_i32(load_u16x4_i32(s), dc0v);
                    store_i32x4_u16_clip(dv, apply4_i32_ac(ac, alpha1, dc2v), max_v);
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

pub(crate) fn cfl_apply_420_hbd_sse41(args: CflApplyHbd<'_>) {
    unsafe { cfl_apply_420_hbd_sse41_impl(args) }
}

pub(crate) fn cfl_apply_422_hbd_sse41(args: CflApplyHbd<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => crate::cfl_dispatch::cfl_apply_422_hbd_scalar(args),
        CFL_FLT_TYPE_GAUSS => unsafe { cfl_apply_422_hbd_sse41_impl::<true>(args) },
        _ => unsafe { cfl_apply_422_hbd_sse41_impl::<false>(args) },
    }
}

pub(crate) fn cfl_apply_444_hbd_sse41(args: CflApplyHbd<'_>) {
    unsafe { cfl_apply_444_hbd_sse41_impl(args) }
}
