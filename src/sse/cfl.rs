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
#[inline]
#[target_feature(enable = "sse4.1")]
fn ac_pair(top: __m128i, bot: __m128i, ones: __m128i, dc0v: __m128i) -> (__m128i, __m128i) {
    let tsum = _mm_maddubs_epi16(top, ones);
    let bsum = _mm_maddubs_epi16(bot, ones);
    let sum16 = _mm_add_epi16(tsum, bsum);
    let sum_lo = _mm_cvtepu16_epi32(sum16);
    let sum_hi = _mm_cvtepu16_epi32(_mm_srli_si128(sum16, 8));
    let ac_lo = _mm_sub_epi32(_mm_slli_epi32(sum_lo, 1), dc0v);
    let ac_hi = _mm_sub_epi32(_mm_slli_epi32(sum_hi, 1), dc0v);
    (ac_lo, ac_hi)
}

/// Apply alpha to 8 AC lanes and produce 8 clipped bytes in the low 8 bytes.
#[inline]
#[target_feature(enable = "sse4.1")]
fn apply8(ac_lo: __m128i, ac_hi: __m128i, alpha: i32, dc: i32) -> __m128i {
    let av = _mm_set1_epi32(alpha);
    let dcv = _mm_set1_epi32(dc);
    let r1024 = _mm_set1_epi32(1024);

    let diff_lo = _mm_mullo_epi32(av, ac_lo);
    let mag_lo = _mm_srli_epi32(_mm_add_epi32(_mm_abs_epi32(diff_lo), r1024), 11);
    let val_lo = _mm_add_epi32(dcv, _mm_sign_epi32(mag_lo, diff_lo));

    let diff_hi = _mm_mullo_epi32(av, ac_hi);
    let mag_hi = _mm_srli_epi32(_mm_add_epi32(_mm_abs_epi32(diff_hi), r1024), 11);
    let val_hi = _mm_add_epi32(dcv, _mm_sign_epi32(mag_hi, diff_hi));

    // i32 -> i16 (signed sat) -> u8 (unsigned sat) == clamp(0, 255)
    _mm_packus_epi16(_mm_packs_epi32(val_lo, val_hi), _mm_setzero_si128())
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
fn cfl_apply_420_8bpc_sse41_impl(
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
    let nfull = xlim / 8; // whole 8-chroma (=16-luma) groups
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    let ones = _mm_set1_epi8(1);
    let dc0v = _mm_set1_epi32(dc0);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let top = y[yrow..yrow + lfull].as_chunks::<16>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<16>()
            .0;

        if alpha0 != 0 {
            for ((d, t), b) in u[urow..urow + xfull]
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(top.iter())
                .zip(bot.iter())
            {
                let (lo, hi) = ac_pair(load_u8x16(t), load_u8x16(b), ones, dc0v);
                store_u8x8(d, apply8(lo, hi, alpha0, dc1));
            }
        }
        if alpha1 != 0 {
            for ((d, t), b) in v[vrow..vrow + xfull]
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(top.iter())
                .zip(bot.iter())
            {
                let (lo, hi) = ac_pair(load_u8x16(t), load_u8x16(b), ones, dc0v);
                store_u8x8(d, apply8(lo, hi, alpha1, dc2));
            }
        }
        // scalar remainder columns
        for x in xfull..xlim {
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
        }
        // right padding
        if alpha0 != 0 {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if alpha1 != 0 {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
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
pub(crate) fn cfl_apply_420_8bpc_sse41(
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
        cfl_apply_420_8bpc_sse41_impl(
            y, u, v, yrow0, urow0, vrow0, ystride, cstride, w, h, xlim, ylim, dc0, dc1, dc2,
            alpha0, alpha1,
        )
    }
}
