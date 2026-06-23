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

/// Load 16 luma bytes from a fixed-size array reference (bounds-safe).
#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

/// Store the low 8 bytes of `v` into a fixed-size array reference (bounds-safe).
#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: __m128i) {
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, v) };
}

#[inline(always)]
fn store_u8x16(a: &mut [u8; 16], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
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

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac8_420_i16(top: __m128i, bot: __m128i, ones: __m128i, dc0v: __m128i) -> __m128i {
    let tsum = _mm_maddubs_epi16(top, ones);
    let bsum = _mm_maddubs_epi16(bot, ones);
    let sum16 = _mm_add_epi16(tsum, bsum);
    _mm_sub_epi16(_mm_slli_epi16(sum16, 1), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac8_444_i16(y8: __m128i, dc0v: __m128i) -> __m128i {
    let y16 = _mm_unpacklo_epi8(y8, _mm_setzero_si128());
    _mm_sub_epi16(_mm_slli_epi16(y16, 3), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac8_422_uniform_i16(row: __m128i, ones: __m128i, dc0v: __m128i) -> __m128i {
    let sum16 = _mm_maddubs_epi16(row, ones);
    _mm_sub_epi16(_mm_slli_epi16(sum16, 2), dc0v)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac8_422_gauss_i16(row: __m128i, even_mask: __m128i, dc0v: __m128i) -> __m128i {
    ac8_444_i16(_mm_shuffle_epi8(row, even_mask), dc0v)
}

/// Apply alpha to 8 i16 AC lanes and produce 8 clipped bytes in the low 8 bytes.
///
/// The downsampling/AC stage stays i16; only the alpha multiply widens to i32.
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply8_i16_ac(
    ac: __m128i,
    alpha_v: __m128i,
    dc_v: __m128i,
    r1024: __m128i,
    zero: __m128i,
) -> __m128i {
    let alpha16 = _mm_packs_epi32(alpha_v, alpha_v);

    let diff_lo = _mm_madd_epi16(_mm_unpacklo_epi16(ac, zero), alpha16);
    let mag_lo = _mm_srli_epi32(_mm_add_epi32(_mm_abs_epi32(diff_lo), r1024), 11);
    let val_lo = _mm_add_epi32(
        dc_v,
        _mm_blendv_epi8(
            mag_lo,
            _mm_sub_epi32(zero, mag_lo),
            _mm_cmpgt_epi32(zero, diff_lo),
        ),
    );

    let diff_hi = _mm_madd_epi16(_mm_unpackhi_epi16(ac, zero), alpha16);
    let mag_hi = _mm_srli_epi32(_mm_add_epi32(_mm_abs_epi32(diff_hi), r1024), 11);
    let val_hi = _mm_add_epi32(
        dc_v,
        _mm_blendv_epi8(
            mag_hi,
            _mm_sub_epi32(zero, mag_hi),
            _mm_cmpgt_epi32(zero, diff_hi),
        ),
    );

    // i32 -> i16 (signed sat) -> u8 (unsigned sat) == clamp(0, 255)
    _mm_packus_epi16(_mm_packs_epi32(val_lo, val_hi), zero)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn apply16_444_i16_ac(
    src: __m128i,
    dc0v: __m128i,
    alpha_v: __m128i,
    dc_v: __m128i,
    r1024: __m128i,
    zero: __m128i,
) -> __m128i {
    let out0 = apply8_i16_ac(ac8_444_i16(src, dc0v), alpha_v, dc_v, r1024, zero);
    let out1 = apply8_i16_ac(
        ac8_444_i16(_mm_srli_si128(src, 8), dc0v),
        alpha_v,
        dc_v,
        r1024,
        zero,
    );
    _mm_unpacklo_epi64(out0, out1)
}

#[target_feature(enable = "sse4.1")]
fn cfl_apply_420_8bpc_sse41_impl(args: CflApply8<'_>) {
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

    let nfull = xlim / 8; // whole 8-chroma (=16-luma) groups
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));

    let ones = _mm_set1_epi8(1);
    let dc0v = _mm_set1_epi16(dc0 as i16);
    let alpha0v = _mm_set1_epi32(alpha0);
    let alpha1v = _mm_set1_epi32(alpha1);
    let dc1v = _mm_set1_epi32(dc1);
    let dc2v = _mm_set1_epi32(dc2);
    let r1024 = _mm_set1_epi32(1024);
    let zero = _mm_setzero_si128();

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
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), ones, dc0v);
                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, r1024, zero));
                }
            }
            (true, false) => {
                for ((d, t), b) in u[urow..urow + xfull]
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), ones, dc0v);
                    store_u8x8(d, apply8_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                }
            }
            (false, true) => {
                for ((d, t), b) in v[vrow..vrow + xfull]
                    .as_chunks_mut::<8>()
                    .0
                    .iter_mut()
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), ones, dc0v);
                    store_u8x8(d, apply8_i16_ac(ac, alpha1v, dc2v, r1024, zero));
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

#[inline]
#[target_feature(enable = "sse4.1")]
fn ac8_422_i16<const GAUSS: bool>(
    row: __m128i,
    ones: __m128i,
    even_mask: __m128i,
    dc0v: __m128i,
) -> __m128i {
    if GAUSS {
        ac8_422_gauss_i16(row, even_mask, dc0v)
    } else {
        ac8_422_uniform_i16(row, ones, dc0v)
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

#[target_feature(enable = "sse4.1")]
fn cfl_apply_422_8bpc_sse41_impl<const GAUSS: bool>(args: CflApply8<'_>) {
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

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));

    let ones = _mm_set1_epi8(1);
    let dc0v = _mm_set1_epi16(dc0 as i16);
    let alpha0v = _mm_set1_epi32(alpha0);
    let alpha1v = _mm_set1_epi32(alpha1);
    let dc1v = _mm_set1_epi32(dc1);
    let dc2v = _mm_set1_epi32(dc2);
    let r1024 = _mm_set1_epi32(1024);
    let zero = _mm_setzero_si128();
    let even_mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
    );

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let row = y[yrow..yrow + lfull].as_chunks::<16>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for ((du, dv), yy) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(row.iter()) {
                    let ac = ac8_422_i16::<GAUSS>(load_u8x16(yy), ones, even_mask, dc0v);
                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, r1024, zero));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    let ac = ac8_422_i16::<GAUSS>(load_u8x16(yy), ones, even_mask, dc0v);
                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, r1024, zero));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;
                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    let ac = ac8_422_i16::<GAUSS>(load_u8x16(yy), ones, even_mask, dc0v);
                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, r1024, zero));
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

#[target_feature(enable = "sse4.1")]
fn cfl_apply_444_8bpc_sse41_impl(args: CflApply8<'_>) {
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

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));

    let dc0v = _mm_set1_epi16(dc0 as i16);
    let alpha0v = _mm_set1_epi32(alpha0);
    let alpha1v = _mm_set1_epi32(alpha1);
    let dc1v = _mm_set1_epi32(dc1);
    let dc2v = _mm_set1_epi32(dc2);
    let r1024 = _mm_set1_epi32(1024);
    let zero = _mm_setzero_si128();

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let row = y[yrow..yrow + xfull].as_chunks::<16>().0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;

                for ((du, dv), yy) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(row.iter()) {
                    let yy = load_u8x16(yy);
                    store_u8x16(du, apply16_444_i16_ac(yy, dc0v, alpha0v, dc1v, r1024, zero));
                    store_u8x16(dv, apply16_444_i16_ac(yy, dc0v, alpha1v, dc2v, r1024, zero));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<16>().0;
                for (du, yy) in u_chunks.iter_mut().zip(row.iter()) {
                    store_u8x16(
                        du,
                        apply16_444_i16_ac(load_u8x16(yy), dc0v, alpha0v, dc1v, r1024, zero),
                    );
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<16>().0;
                for (dv, yy) in v_chunks.iter_mut().zip(row.iter()) {
                    store_u8x16(
                        dv,
                        apply16_444_i16_ac(load_u8x16(yy), dc0v, alpha1v, dc2v, r1024, zero),
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

pub(crate) fn cfl_apply_420_8bpc_sse41(args: CflApply8<'_>) {
    unsafe { cfl_apply_420_8bpc_sse41_impl(args) }
}

pub(crate) fn cfl_apply_422_8bpc_sse41(args: CflApply8<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => crate::cfl_dispatch::cfl_apply_422_8bpc_scalar(args),
        CFL_FLT_TYPE_GAUSS => unsafe { cfl_apply_422_8bpc_sse41_impl::<true>(args) },
        _ => unsafe { cfl_apply_422_8bpc_sse41_impl::<false>(args) },
    }
}

pub(crate) fn cfl_apply_444_8bpc_sse41(args: CflApply8<'_>) {
    unsafe { cfl_apply_444_8bpc_sse41_impl(args) }
}
