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
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn mul3_i32(v: __m128i) -> __m128i {
    _mm_add_epi32(v, _mm_slli_epi32::<1>(v))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul4_i32(v: __m128i) -> __m128i {
    _mm_slli_epi32::<2>(v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn deblock_delta_i32(
    d0: __m128i,
    dm1: __m128i,
    dp1: __m128i,
    dm2: __m128i,
    nqc: __m128i,
    qc: __m128i,
) -> __m128i {
    let d0_m1 = _mm_sub_epi32(d0, dm1);
    let dp1_m2 = _mm_sub_epi32(dp1, dm2);
    let inner = _mm_sub_epi32(mul3_i32(d0_m1), dp1_m2);
    _mm_min_epi32(_mm_max_epi32(mul4_i32(inner), nqc), qc)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load4_u8_i32(dst: &[u8], base: isize, stride_line: isize) -> __m128i {
    if stride_line == 1 {
        unsafe {
            _mm_cvtepu8_epi32(_mm_castps_si128(_mm_load_ss(
                dst.as_ptr().add(base as usize).cast(),
            )))
        }
    } else {
        // Four rows of the same horizontal edge.  This is still a gather, but it
        // stays register-only instead of using a temporary stack array.
        unsafe {
            let p = dst.as_ptr();
            _mm_setr_epi32(
                *p.add(base as usize) as i32,
                *p.add((base + stride_line) as usize) as i32,
                *p.add((base + 2 * stride_line) as usize) as i32,
                *p.add((base + 3 * stride_line) as usize) as i32,
            )
        }
    }
}

/// Scatter a pre-clipped (`0..=255`) i32x4 back to the 4 line positions.
#[inline]
#[target_feature(enable = "avx2")]
fn store4_clip_u8(dst: &mut [u8], base: isize, stride_line: isize, v: __m128i) {
    if stride_line == 1 {
        let p8 = _mm_packus_epi16(_mm_packs_epi32(v, v), _mm_packs_epi32(v, v));
        unsafe {
            _mm_store_ss(
                dst.as_mut_ptr().add(base as usize).cast(),
                _mm_castsi128_ps(p8),
            )
        }
    } else {
        unsafe {
            let p = dst.as_mut_ptr();
            *p.add(base as usize) = _mm_cvtsi128_si32(v) as u8;
            *p.add((base + stride_line) as usize) = _mm_extract_epi32::<1>(v) as u8;
            *p.add((base + 2 * stride_line) as usize) = _mm_extract_epi32::<2>(v) as u8;
            *p.add((base + 3 * stride_line) as usize) = _mm_extract_epi32::<3>(v) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load4_u8_i16_oriented<const CONTIG: bool>(
    dst: &[u8],
    base: isize,
    stride_line: isize,
) -> __m128i {
    unsafe {
        let p = dst.as_ptr();
        if CONTIG {
            let word = std::ptr::read_unaligned(p.add(base as usize).cast::<i32>());
            _mm_cvtepu8_epi16(_mm_cvtsi32_si128(word))
        } else {
            _mm_setr_epi16(
                *p.add(base as usize) as i16,
                *p.add((base + stride_line) as usize) as i16,
                *p.add((base + 2 * stride_line) as usize) as i16,
                *p.add((base + 3 * stride_line) as usize) as i16,
                0,
                0,
                0,
                0,
            )
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store4_clip_u8_i16_oriented<const CONTIG: bool>(
    dst: &mut [u8],
    base: isize,
    stride_line: isize,
    v: __m128i,
) {
    unsafe {
        let p = dst.as_mut_ptr();
        let p8 = _mm_packus_epi16(v, v);
        if CONTIG {
            _mm_store_ss(p.add(base as usize).cast(), _mm_castsi128_ps(p8));
        } else {
            let packed = _mm_cvtsi128_si32(p8) as u32;
            *p.add(base as usize) = (packed & 0xff) as u8;
            *p.add((base + stride_line) as usize) = ((packed >> 8) & 0xff) as u8;
            *p.add((base + 2 * stride_line) as usize) = ((packed >> 16) & 0xff) as u8;
            *p.add((base + 3 * stride_line) as usize) = (packed >> 24) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn deblock_delta_i16(
    d0: __m128i,
    dm1: __m128i,
    dp1: __m128i,
    dm2: __m128i,
    nqc: __m128i,
    qc: __m128i,
) -> __m128i {
    let d0_m1 = _mm_sub_epi16(d0, dm1);
    let dp1_m2 = _mm_sub_epi16(dp1, dm2);
    let inner = _mm_sub_epi16(_mm_add_epi16(d0_m1, _mm_slli_epi16::<1>(d0_m1)), dp1_m2);
    let delta = _mm_slli_epi16::<2>(inner);
    _mm_min_epi16(_mm_max_epi16(delta, nqc), qc)
}

#[inline]
#[target_feature(enable = "avx2")]
fn deblock_diff_i16(delta: __m128i, width: i32, tap: i32) -> __m128i {
    let coeff = (crate::deblock::W_MULT[(width - 1) as usize] as i32 * tap * 16) as i16;
    _mm_mulhrs_epi16(delta, _mm_set1_epi16(coeff))
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_8bpc_avx2_const_oriented<const WN: i32, const WP: i32, const CONTIG: bool>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    apply_neg: bool,
    apply_pos: bool,
) {
    debug_assert!((1..=8).contains(&WN));
    debug_assert!((1..=8).contains(&WP));
    debug_assert!(apply_neg || apply_pos);
    debug_assert!(q_thr_clamp <= i16::MAX as i32);

    let qc = _mm_set1_epi16(q_thr_clamp as i16);
    let nqc = _mm_set1_epi16(-(q_thr_clamp as i16));
    let d0 = load4_u8_i16_oriented::<CONTIG>(dst, off, stride_line);
    let dm1 = load4_u8_i16_oriented::<CONTIG>(dst, off - stride_tap, stride_line);
    let dp1 = load4_u8_i16_oriented::<CONTIG>(dst, off + stride_tap, stride_line);
    let dm2 = load4_u8_i16_oriented::<CONTIG>(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i16(d0, dm1, dp1, dm2, nqc, qc);

    if apply_neg {
        let mut j = 0;
        while j < WN {
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u8_i16_oriented::<CONTIG>(dst, base, stride_line);
            let diff = deblock_diff_i16(delta, WN, WN - j);
            store4_clip_u8_i16_oriented::<CONTIG>(dst, base, stride_line, _mm_add_epi16(cur, diff));
            j += 1;
        }
    }

    if apply_pos {
        let mut j = 0;
        while j < WP {
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u8_i16_oriented::<CONTIG>(dst, base, stride_line);
            let diff = deblock_diff_i16(delta, WP, WP - j);
            store4_clip_u8_i16_oriented::<CONTIG>(dst, base, stride_line, _mm_sub_epi16(cur, diff));
            j += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_8bpc_avx2_const<const WN: i32, const WP: i32>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    // Split orientation and lossless-side handling once.  Dav2d's kernels do
    // not carry per-tap lossless branches through the hot arithmetic body; the
    // const side flags let LLVM erase the unused half-filter for tile/lossless
    // edge cases while keeping the common both-sides path straight-line.
    if stride_line == 1 {
        deblock_apply_8bpc_avx2_const_oriented::<WN, WP, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
    } else {
        debug_assert_eq!(stride_tap, 1);
        deblock_apply_8bpc_avx2_const_oriented::<WN, WP, false>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_8bpc_avx2_specialized(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) -> bool {
    // The dav2d-style i16/pmulhrsw apply path needs the clipping threshold in
    // signed 16-bit lanes.  8bpc thresholds are expected to fit, but keep the
    // old i32 fallback available for malformed/extreme inputs.
    if q_thr_clamp > i16::MAX as i32 {
        return false;
    }

    match (width_neg, width_pos) {
        (1, 1) => deblock_apply_8bpc_avx2_const::<1, 1>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (1, 2) => deblock_apply_8bpc_avx2_const::<1, 2>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 2) => deblock_apply_8bpc_avx2_const::<2, 2>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 3) => deblock_apply_8bpc_avx2_const::<2, 3>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (1, 3) => deblock_apply_8bpc_avx2_const::<1, 3>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (3, 3) => deblock_apply_8bpc_avx2_const::<3, 3>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (1, 4) => deblock_apply_8bpc_avx2_const::<1, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 4) => deblock_apply_8bpc_avx2_const::<2, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (3, 4) => deblock_apply_8bpc_avx2_const::<3, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (4, 4) => deblock_apply_8bpc_avx2_const::<4, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (1, 6) => deblock_apply_8bpc_avx2_const::<1, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 6) => deblock_apply_8bpc_avx2_const::<2, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (3, 6) => deblock_apply_8bpc_avx2_const::<3, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (4, 6) => deblock_apply_8bpc_avx2_const::<4, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (6, 6) => deblock_apply_8bpc_avx2_const::<6, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (1, 8) => deblock_apply_8bpc_avx2_const::<1, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 8) => deblock_apply_8bpc_avx2_const::<2, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (3, 8) => deblock_apply_8bpc_avx2_const::<3, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (4, 8) => deblock_apply_8bpc_avx2_const::<4, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (6, 8) => deblock_apply_8bpc_avx2_const::<6, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (8, 8) => deblock_apply_8bpc_avx2_const::<8, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        _ => return false,
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn deblock_apply_8bpc_avx2(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    if q_thr_clamp <= 0 || (neg_lossless && pos_lossless) {
        return;
    }

    // Dav2d emits separate FILTER 1/3/4/6/8 bodies instead of one
    // runtime-width loop.  Keep the Rust API generic, but dispatch common
    // luma/chroma edge-width pairs to const-generic AVX2 kernels so the tap
    // loops and W_MULT loads fold away.
    if deblock_apply_8bpc_avx2_specialized(
        dst,
        off,
        stride_line,
        stride_tap,
        width_neg,
        width_pos,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
    ) {
        return;
    }

    let qc = _mm_set1_epi32(q_thr_clamp);
    let nqc = _mm_set1_epi32(-q_thr_clamp);
    let rnd = _mm_set1_epi32(1 << 10);
    let zero = _mm_setzero_si128();
    let v255 = _mm_set1_epi32(255);
    let d0 = load4_u8_i32(dst, off, stride_line);
    let dm1 = load4_u8_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u8_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u8_i32(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i32(d0, dm1, dp1, dm2, nqc, qc);

    if !neg_lossless {
        let dn = _mm_mullo_epi32(
            delta,
            _mm_set1_epi32(crate::deblock::W_MULT[(width_neg - 1) as usize] as i32),
        );
        for j in 0..width_neg {
            let diff = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_mullo_epi32(dn, _mm_set1_epi32(width_neg - j)),
                rnd,
            ));
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u8_i32(dst, base, stride_line);
            let res = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(cur, diff), zero), v255);
            store4_clip_u8(dst, base, stride_line, res);
        }
    }

    if !pos_lossless {
        let dpv = _mm_mullo_epi32(
            delta,
            _mm_set1_epi32(crate::deblock::W_MULT[(width_pos - 1) as usize] as i32),
        );
        for j in 0..width_pos {
            let diff = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_mullo_epi32(dpv, _mm_set1_epi32(width_pos - j)),
                rnd,
            ));
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u8_i32(dst, base, stride_line);
            let res = _mm_min_epi32(_mm_max_epi32(_mm_sub_epi32(cur, diff), zero), v255);
            store4_clip_u8(dst, base, stride_line, res);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load4_u16_i32(dst: &[u16], base: isize, stride_line: isize) -> __m128i {
    if stride_line == 1 {
        unsafe {
            _mm_cvtepu16_epi32(_mm_loadl_epi64(
                dst.as_ptr().add(base as usize) as *const __m128i
            ))
        }
    } else {
        unsafe {
            let p = dst.as_ptr();
            _mm_setr_epi32(
                *p.add(base as usize) as i32,
                *p.add((base + stride_line) as usize) as i32,
                *p.add((base + 2 * stride_line) as usize) as i32,
                *p.add((base + 3 * stride_line) as usize) as i32,
            )
        }
    }
}

/// Scatter a pre-clipped (`0..=bitdepth_max`) i32x4 back to 4 HBD samples.
#[inline]
#[target_feature(enable = "avx2")]
fn store4_clip_u16(dst: &mut [u16], base: isize, stride_line: isize, v: __m128i) {
    if stride_line == 1 {
        let p16 = _mm_packus_epi32(v, v);
        unsafe {
            _mm_storel_epi64(dst.as_mut_ptr().add(base as usize) as *mut __m128i, p16);
        }
    } else {
        unsafe {
            let p = dst.as_mut_ptr();
            *p.add(base as usize) = _mm_cvtsi128_si32(v) as u16;
            *p.add((base + stride_line) as usize) = _mm_extract_epi32::<1>(v) as u16;
            *p.add((base + 2 * stride_line) as usize) = _mm_extract_epi32::<2>(v) as u16;
            *p.add((base + 3 * stride_line) as usize) = _mm_extract_epi32::<3>(v) as u16;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load4_u16_i32_oriented<const CONTIG: bool>(
    dst: &[u16],
    base: isize,
    stride_line: isize,
) -> __m128i {
    unsafe {
        let p = dst.as_ptr();
        if CONTIG {
            _mm_cvtepu16_epi32(_mm_loadl_epi64(p.add(base as usize) as *const __m128i))
        } else {
            _mm_setr_epi32(
                *p.add(base as usize) as i32,
                *p.add((base + stride_line) as usize) as i32,
                *p.add((base + 2 * stride_line) as usize) as i32,
                *p.add((base + 3 * stride_line) as usize) as i32,
            )
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store4_clip_u16_oriented<const CONTIG: bool>(
    dst: &mut [u16],
    base: isize,
    stride_line: isize,
    v: __m128i,
) {
    unsafe {
        let p = dst.as_mut_ptr();
        if CONTIG {
            let p16 = _mm_packus_epi32(v, v);
            _mm_storel_epi64(p.add(base as usize) as *mut __m128i, p16);
        } else {
            *p.add(base as usize) = _mm_cvtsi128_si32(v) as u16;
            *p.add((base + stride_line) as usize) = _mm_extract_epi32::<1>(v) as u16;
            *p.add((base + 2 * stride_line) as usize) = _mm_extract_epi32::<2>(v) as u16;
            *p.add((base + 3 * stride_line) as usize) = _mm_extract_epi32::<3>(v) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_hbd_avx2_const_oriented<
    const WN: i32,
    const WP: i32,
    const CONTIG: bool,
    const APPLY_NEG: bool,
    const APPLY_POS: bool,
>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    bitdepth_max: i32,
) {
    debug_assert!((1..=8).contains(&WN));
    debug_assert!((1..=8).contains(&WP));
    debug_assert!(APPLY_NEG || APPLY_POS);

    let qc = _mm_set1_epi32(q_thr_clamp);
    let nqc = _mm_set1_epi32(-q_thr_clamp);
    let rnd = _mm_set1_epi32(1 << 10);
    let zero = _mm_setzero_si128();
    let vmax = _mm_set1_epi32(bitdepth_max);
    let d0 = load4_u16_i32_oriented::<CONTIG>(dst, off, stride_line);
    let dm1 = load4_u16_i32_oriented::<CONTIG>(dst, off - stride_tap, stride_line);
    let dp1 = load4_u16_i32_oriented::<CONTIG>(dst, off + stride_tap, stride_line);
    let dm2 = load4_u16_i32_oriented::<CONTIG>(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i32(d0, dm1, dp1, dm2, nqc, qc);

    if APPLY_NEG {
        let dn = _mm_mullo_epi32(
            delta,
            _mm_set1_epi32(crate::deblock::W_MULT[(WN - 1) as usize] as i32),
        );
        let mut j = 0;
        while j < WN {
            let diff = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_mullo_epi32(dn, _mm_set1_epi32(WN - j)),
                rnd,
            ));
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u16_i32_oriented::<CONTIG>(dst, base, stride_line);
            let res = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(cur, diff), zero), vmax);
            store4_clip_u16_oriented::<CONTIG>(dst, base, stride_line, res);
            j += 1;
        }
    }

    if APPLY_POS {
        let dpv = _mm_mullo_epi32(
            delta,
            _mm_set1_epi32(crate::deblock::W_MULT[(WP - 1) as usize] as i32),
        );
        let mut j = 0;
        while j < WP {
            let diff = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_mullo_epi32(dpv, _mm_set1_epi32(WP - j)),
                rnd,
            ));
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u16_i32_oriented::<CONTIG>(dst, base, stride_line);
            let res = _mm_min_epi32(_mm_max_epi32(_mm_sub_epi32(cur, diff), zero), vmax);
            store4_clip_u16_oriented::<CONTIG>(dst, base, stride_line, res);
            j += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_hbd_avx2_const_sides<const WN: i32, const WP: i32, const CONTIG: bool>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    match (neg_lossless, pos_lossless) {
        (false, false) => deblock_apply_hbd_avx2_const_oriented::<WN, WP, CONTIG, true, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            bitdepth_max,
        ),
        (false, true) => deblock_apply_hbd_avx2_const_oriented::<WN, WP, CONTIG, true, false>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            bitdepth_max,
        ),
        (true, false) => deblock_apply_hbd_avx2_const_oriented::<WN, WP, CONTIG, false, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            bitdepth_max,
        ),
        (true, true) => {}
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_hbd_avx2_const<const WN: i32, const WP: i32>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    if stride_line == 1 {
        deblock_apply_hbd_avx2_const_sides::<WN, WP, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        );
    } else {
        debug_assert_eq!(stride_tap, 1);
        deblock_apply_hbd_avx2_const_sides::<WN, WP, false>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_hbd_avx2_specialized(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) -> bool {
    match (width_neg, width_pos) {
        (1, 1) => deblock_apply_hbd_avx2_const::<1, 1>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (1, 2) => deblock_apply_hbd_avx2_const::<1, 2>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 2) => deblock_apply_hbd_avx2_const::<2, 2>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 3) => deblock_apply_hbd_avx2_const::<2, 3>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (1, 3) => deblock_apply_hbd_avx2_const::<1, 3>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (3, 3) => deblock_apply_hbd_avx2_const::<3, 3>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (1, 4) => deblock_apply_hbd_avx2_const::<1, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 4) => deblock_apply_hbd_avx2_const::<2, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (3, 4) => deblock_apply_hbd_avx2_const::<3, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (4, 4) => deblock_apply_hbd_avx2_const::<4, 4>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (1, 6) => deblock_apply_hbd_avx2_const::<1, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 6) => deblock_apply_hbd_avx2_const::<2, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (3, 6) => deblock_apply_hbd_avx2_const::<3, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (4, 6) => deblock_apply_hbd_avx2_const::<4, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (6, 6) => deblock_apply_hbd_avx2_const::<6, 6>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (1, 8) => deblock_apply_hbd_avx2_const::<1, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 8) => deblock_apply_hbd_avx2_const::<2, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (3, 8) => deblock_apply_hbd_avx2_const::<3, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (4, 8) => deblock_apply_hbd_avx2_const::<4, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (6, 8) => deblock_apply_hbd_avx2_const::<6, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (8, 8) => deblock_apply_hbd_avx2_const::<8, 8>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        _ => return false,
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn deblock_apply_hbd_avx2(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    if q_thr_clamp <= 0 || (neg_lossless && pos_lossless) {
        return;
    }

    if deblock_apply_hbd_avx2_specialized(
        dst,
        off,
        stride_line,
        stride_tap,
        width_neg,
        width_pos,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
        bitdepth_max,
    ) {
        return;
    }

    let qc = _mm_set1_epi32(q_thr_clamp);
    let nqc = _mm_set1_epi32(-q_thr_clamp);
    let rnd = _mm_set1_epi32(1 << 10);
    let zero = _mm_setzero_si128();
    let vmax = _mm_set1_epi32(bitdepth_max);
    let d0 = load4_u16_i32(dst, off, stride_line);
    let dm1 = load4_u16_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u16_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u16_i32(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i32(d0, dm1, dp1, dm2, nqc, qc);

    if !neg_lossless {
        let dn = _mm_mullo_epi32(
            delta,
            _mm_set1_epi32(crate::deblock::W_MULT[(width_neg - 1) as usize] as i32),
        );
        for j in 0..width_neg {
            let diff = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_mullo_epi32(dn, _mm_set1_epi32(width_neg - j)),
                rnd,
            ));
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u16_i32(dst, base, stride_line);
            let res = _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(cur, diff), zero), vmax);
            store4_clip_u16(dst, base, stride_line, res);
        }
    }

    if !pos_lossless {
        let dpv = _mm_mullo_epi32(
            delta,
            _mm_set1_epi32(crate::deblock::W_MULT[(width_pos - 1) as usize] as i32),
        );
        for j in 0..width_pos {
            let diff = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_mullo_epi32(dpv, _mm_set1_epi32(width_pos - j)),
                rnd,
            ));
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u16_i32(dst, base, stride_line);
            let res = _mm_min_epi32(_mm_max_epi32(_mm_sub_epi32(cur, diff), zero), vmax);
            store4_clip_u16(dst, base, stride_line, res);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn select_i32(mask: bool, yes: i32, no: i32) -> i32 {
    let m = -(mask as i32);
    (yes & m) | (no & !m)
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_avg_abs2_from_lanes(v: __m128i) -> u32 {
    let sum = _mm_add_epi16(v, _mm_srli_si128::<2>(v));
    ((_mm_extract_epi16::<0>(sum) as u32) + 1) >> 1
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_second_deriv_8bpc_avx2(
    buf: &[u8],
    s: isize,
    t: isize,
    stride: isize,
    dist: isize,
) -> u32 {
    unsafe {
        let p = buf.as_ptr();
        let s0 = *p.add((s + (dist - 1) * stride) as usize) as i16;
        let s1 = *p.add((s + dist * stride) as usize) as i16;
        let s2 = *p.add((s + (dist + 1) * stride) as usize) as i16;
        let t0 = *p.add((t + (dist - 1) * stride) as usize) as i16;
        let t1 = *p.add((t + dist * stride) as usize) as i16;
        let t2 = *p.add((t + (dist + 1) * stride) as usize) as i16;
        let a = _mm_setr_epi16(s0, t0, 0, 0, 0, 0, 0, 0);
        let b = _mm_setr_epi16(s1, t1, 0, 0, 0, 0, 0, 0);
        let c = _mm_setr_epi16(s2, t2, 0, 0, 0, 0, 0, 0);
        let deriv = _mm_add_epi16(_mm_sub_epi16(a, _mm_slli_epi16::<1>(b)), c);
        filter_avg_abs2_from_lanes(_mm_abs_epi16(deriv))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn filter_end_deriv_8bpc_avx2(
    buf: &[u8],
    s0: isize,
    s1: isize,
    s2: isize,
    t0: isize,
    t1: isize,
    t2: isize,
    c0: i16,
    c1: i16,
    c2: i16,
) -> u32 {
    unsafe {
        let p = buf.as_ptr();
        let a = _mm_setr_epi16(
            *p.add(s0 as usize) as i16,
            *p.add(t0 as usize) as i16,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let b = _mm_setr_epi16(
            *p.add(s1 as usize) as i16,
            *p.add(t1 as usize) as i16,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let c = _mm_setr_epi16(
            *p.add(s2 as usize) as i16,
            *p.add(t2 as usize) as i16,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let v = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(a, _mm_set1_epi16(c0)),
                _mm_mullo_epi16(b, _mm_set1_epi16(c1)),
            ),
            _mm_mullo_epi16(c, _mm_set1_epi16(c2)),
        );
        filter_avg_abs2_from_lanes(_mm_abs_epi16(v))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn filter_choice_8bpc_avx2_const<const MAX_WIDTH_NEG: i32, const MAX_WIDTH_POS: i32>(
    buf: &[u8],
    s: isize,
    t: isize,
    stride: isize,
    q_thr: u32,
    side_thr: u32,
) -> i32 {
    debug_assert!((1..=8).contains(&MAX_WIDTH_POS));
    debug_assert!((1..=8).contains(&MAX_WIDTH_NEG));
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    // Const max-width variant for the SB-level AVX2 walkers.  Dav2d expands
    // separate FILTER bodies for 1/3/4/6/8-wide cases; carrying max_width_pos
    // and max_width_neg as runtime integers keeps branches in the hot filter
    // choice even after vectorizing the derivative math.  With const limits,
    // LLVM can erase the 4/6/8 probes that are impossible for the current mask
    // lane and fold the negative-side checks for tile/frame edges.
    let sd_m2 = filter_second_deriv_8bpc_avx2(buf, s, t, stride, -2);
    let sd_m1 = filter_second_deriv_8bpc_avx2(buf, s, t, stride, -1);
    let sd_0 = filter_second_deriv_8bpc_avx2(buf, s, t, stride, 0);
    let sd_1 = filter_second_deriv_8bpc_avx2(buf, s, t, stride, 1);

    let high_deriv = sd_m2.max(sd_1);
    let transition = sd_m1 + sd_0;

    let fail0 = high_deriv > side_thr;
    if MAX_WIDTH_POS == 1 {
        return select_i32(fail0, 0, 1);
    }

    let fail1 = high_deriv > (side_thr >> 2) || transition > q_thr * 4;

    let end_thr = (side_thr * 3) >> 4;
    let neg3_fail = if MAX_WIDTH_NEG >= 3 {
        filter_end_deriv_8bpc_avx2(
            buf,
            s - stride,
            s - 2 * stride,
            s - 4 * stride,
            t - stride,
            t - 2 * stride,
            t - 4 * stride,
            -2,
            3,
            -1,
        ) > end_thr
    } else {
        false
    };
    let pos3_fail = filter_end_deriv_8bpc_avx2(
        buf,
        s,
        s + stride,
        s + 3 * stride,
        t,
        t + stride,
        t + 3 * stride,
        -2,
        3,
        -1,
    ) > end_thr;
    let fail2 = high_deriv > (side_thr >> 3) || transition > q_thr * 3 || neg3_fail || pos3_fail;

    if MAX_WIDTH_POS == 3 {
        let mut width = 3;
        width = select_i32(fail2, 2, width);
        width = select_i32(fail1, 1, width);
        return select_i32(fail0, 0, width);
    }

    let transition4 = transition << 4;
    let mut fail4 = false;
    let mut fail6 = false;
    let mut fail8 = false;

    if MAX_WIDTH_POS >= 4 {
        let dist = 4i32;
        let dist2 = 4i32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        let neg_fail = if MAX_WIDTH_NEG >= dist2 {
            filter_end_deriv_8bpc_avx2(
                buf,
                s - stride,
                s - (dist2 as isize + 1) * stride,
                s - 2 * stride,
                t - stride,
                t - (dist2 as isize + 1) * stride,
                t - 2 * stride,
                (1 - dist2) as i16,
                -1,
                dist2 as i16,
            ) > end_thr4
        } else {
            false
        };
        let pos_fail = filter_end_deriv_8bpc_avx2(
            buf,
            s,
            s + dist2 as isize * stride,
            s + stride,
            t,
            t + dist2 as isize * stride,
            t + stride,
            (1 - dist2) as i16,
            -1,
            dist2 as i16,
        ) > end_thr4;
        fail4 = transition4 > q_thr * crate::deblock::Q_FIRST[0] as u32 || neg_fail || pos_fail;
    }

    if MAX_WIDTH_POS >= 6 {
        let dist = 6i32;
        let dist2 = 6i32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        let neg_fail = if MAX_WIDTH_NEG >= dist2 {
            filter_end_deriv_8bpc_avx2(
                buf,
                s - stride,
                s - (dist2 as isize + 1) * stride,
                s - 2 * stride,
                t - stride,
                t - (dist2 as isize + 1) * stride,
                t - 2 * stride,
                (1 - dist2) as i16,
                -1,
                dist2 as i16,
            ) > end_thr4
        } else {
            false
        };
        let pos_fail = filter_end_deriv_8bpc_avx2(
            buf,
            s,
            s + dist2 as isize * stride,
            s + stride,
            t,
            t + dist2 as isize * stride,
            t + stride,
            (1 - dist2) as i16,
            -1,
            dist2 as i16,
        ) > end_thr4;
        fail6 = transition4 > q_thr * crate::deblock::Q_FIRST[1] as u32 || neg_fail || pos_fail;
    }

    if MAX_WIDTH_POS >= 8 {
        let dist = 8i32;
        let dist2 = 7i32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        let neg_fail = if MAX_WIDTH_NEG >= dist2 {
            filter_end_deriv_8bpc_avx2(
                buf,
                s - stride,
                s - (dist2 as isize + 1) * stride,
                s - 2 * stride,
                t - stride,
                t - (dist2 as isize + 1) * stride,
                t - 2 * stride,
                (1 - dist2) as i16,
                -1,
                dist2 as i16,
            ) > end_thr4
        } else {
            false
        };
        let pos_fail = filter_end_deriv_8bpc_avx2(
            buf,
            s,
            s + dist2 as isize * stride,
            s + stride,
            t,
            t + dist2 as isize * stride,
            t + stride,
            (1 - dist2) as i16,
            -1,
            dist2 as i16,
        ) > end_thr4;
        fail8 = transition4 > q_thr * crate::deblock::Q_FIRST[2] as u32 || neg_fail || pos_fail;
    }

    let mut width = MAX_WIDTH_POS;
    width = select_i32(MAX_WIDTH_POS >= 8 && fail8, 6, width);
    width = select_i32(MAX_WIDTH_POS >= 6 && fail6, 4, width);
    width = select_i32(MAX_WIDTH_POS >= 4 && fail4, 3, width);
    width = select_i32(fail2, 2, width);
    width = select_i32(fail1, 1, width);
    select_i32(fail0, 0, width)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_apply_8bpc_avx2_width_constmax<const MAX_WIDTH_NEG: i32, const CONTIG: bool>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    debug_assert!((1..=8).contains(&MAX_WIDTH_NEG));
    debug_assert!(q_thr_clamp <= i16::MAX as i32);

    match width {
        1 => deblock_apply_8bpc_avx2_const_oriented::<1, 1, CONTIG>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        2 => {
            if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_avx2_const_oriented::<2, 2, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_avx2_const_oriented::<1, 2, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        3 => {
            if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_avx2_const_oriented::<3, 3, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_avx2_const_oriented::<2, 3, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_avx2_const_oriented::<1, 3, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        4 => {
            if MAX_WIDTH_NEG >= 4 {
                deblock_apply_8bpc_avx2_const_oriented::<4, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_avx2_const_oriented::<3, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_avx2_const_oriented::<2, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_avx2_const_oriented::<1, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        6 => {
            if MAX_WIDTH_NEG >= 6 {
                deblock_apply_8bpc_avx2_const_oriented::<6, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 4 {
                deblock_apply_8bpc_avx2_const_oriented::<4, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_avx2_const_oriented::<3, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_avx2_const_oriented::<2, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_avx2_const_oriented::<1, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        8 => {
            if MAX_WIDTH_NEG >= 8 {
                deblock_apply_8bpc_avx2_const_oriented::<8, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 6 {
                deblock_apply_8bpc_avx2_const_oriented::<6, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 4 {
                deblock_apply_8bpc_avx2_const_oriented::<4, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_avx2_const_oriented::<3, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_avx2_const_oriented::<2, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_avx2_const_oriented::<1, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_8bpc_avx2_const_max<
    const MAX_WIDTH_NEG: i32,
    const MAX_WIDTH_POS: i32,
    const CONTIG: bool,
>(
    dst: &mut [u8],
    off: isize,
    q_thr: u32,
    side_thr: u32,
    stridea: isize,
    strideb: isize,
    pos_lossless: bool,
    neg_lossless: bool,
) {
    debug_assert!((1..=8).contains(&MAX_WIDTH_POS));
    debug_assert!((1..=8).contains(&MAX_WIDTH_NEG));
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    let width = filter_choice_8bpc_avx2_const::<MAX_WIDTH_NEG, MAX_WIDTH_POS>(
        dst,
        off,
        off + 3 * stridea,
        strideb,
        q_thr,
        side_thr,
    );
    if width < 1 || (neg_lossless && pos_lossless) {
        return;
    }

    let q_thr_clamp = q_thr as i32 * crate::deblock::Q_THRESH_MULTS[(width - 1) as usize] as i32;
    if q_thr_clamp <= 0 {
        return;
    }

    if q_thr_clamp > i16::MAX as i32 {
        deblock_apply_8bpc_avx2(
            dst,
            off,
            stridea,
            strideb,
            width.min(MAX_WIDTH_NEG),
            width,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
        return;
    }

    deblock_apply_8bpc_avx2_width_constmax::<MAX_WIDTH_NEG, CONTIG>(
        dst,
        off,
        stridea,
        strideb,
        width,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
    );
}

#[inline]
#[target_feature(enable = "avx2")]
fn deblock_mask_class_bits(mask: u16, higher: u16, both_lossless: u16) -> u32 {
    (mask & !higher & !both_lossless) as u32
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn deblock_sb64_8bpc_avx2_mask<
    const MAX_WIDTH_NEG: i32,
    const MAX_WIDTH_POS: i32,
    const HORIZONTAL: bool,
    const CONTIG: bool,
>(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    mut vm: u32,
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
) {
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    // This is closer to dav2d's SB filter macros: the caller has already split
    // the edge mask into one max-width class, so the loop body no longer has to
    // rediscover idx from vmask[3]/[2]/[1] for every filtered edge.  Lossless
    // both-side edges and q==0 edges are also skipped before filter-choice,
    // avoiding the expensive derivative probes when apply would be a no-op.
    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let bit = 1u32 << qi;
        let q = q_thr[qi] as u32;
        if q != 0 {
            let pos_ll = (ll_mask[1] as u32 & bit) != 0;
            let neg_ll = (ll_mask[0] as u32 & bit) != 0;
            if !(pos_ll && neg_ll) {
                let side = side_thr[qi] as u32;
                let off = if HORIZONTAL {
                    (dst_off + qi * 4 * stride) as isize
                } else {
                    (dst_off + qi * 4) as isize
                };
                let stridea = if HORIZONTAL { stride as isize } else { 1 };
                let strideb = if HORIZONTAL { 1 } else { stride as isize };
                deblock_8bpc_avx2_const_max::<MAX_WIDTH_NEG, MAX_WIDTH_POS, CONTIG>(
                    dst, off, q, side, stridea, strideb, pos_ll, neg_ll,
                );
            }
        }
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn deblock_h_sb64y_8bpc_avx2(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m3 = deblock_mask_class_bits(vmask[3], 0, both_lossless);
    let m2 = deblock_mask_class_bits(vmask[2], vmask[3], both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2] | vmask[3], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2] | vmask[3], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_avx2_mask::<1, 1, true, false>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        deblock_sb64_8bpc_avx2_mask::<3, 3, true, false>(
            dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
        );
    }
    if m2 != 0 {
        deblock_sb64_8bpc_avx2_mask::<6, 6, true, false>(
            dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
        );
    }
    if m3 != 0 {
        if edge {
            deblock_sb64_8bpc_avx2_mask::<6, 8, true, false>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_avx2_mask::<8, 8, true, false>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn deblock_v_sb64y_8bpc_avx2(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m3 = deblock_mask_class_bits(vmask[3], 0, both_lossless);
    let m2 = deblock_mask_class_bits(vmask[2], vmask[3], both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2] | vmask[3], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2] | vmask[3], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_avx2_mask::<1, 1, false, true>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        deblock_sb64_8bpc_avx2_mask::<3, 3, false, true>(
            dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
        );
    }
    if m2 != 0 {
        deblock_sb64_8bpc_avx2_mask::<6, 6, false, true>(
            dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
        );
    }
    if m3 != 0 {
        if edge {
            deblock_sb64_8bpc_avx2_mask::<6, 8, false, true>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_avx2_mask::<8, 8, false, true>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn deblock_h_sb64uv_8bpc_avx2(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m2 = deblock_mask_class_bits(vmask[2], 0, both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_avx2_mask::<1, 1, true, false>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        if edge {
            deblock_sb64_8bpc_avx2_mask::<2, 3, true, false>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_avx2_mask::<3, 3, true, false>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        }
    }
    if m2 != 0 {
        if edge {
            deblock_sb64_8bpc_avx2_mask::<2, 4, true, false>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_avx2_mask::<4, 4, true, false>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn deblock_v_sb64uv_8bpc_avx2(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m2 = deblock_mask_class_bits(vmask[2], 0, both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_avx2_mask::<1, 1, false, true>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        if edge {
            deblock_sb64_8bpc_avx2_mask::<2, 3, false, true>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_avx2_mask::<3, 3, false, true>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        }
    }
    if m2 != 0 {
        if edge {
            deblock_sb64_8bpc_avx2_mask::<2, 4, false, true>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_avx2_mask::<4, 4, false, true>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::deblock_dispatch::{deblock_apply_8bpc_scalar, deblock_apply_hbd_scalar};

    // Tiny xorshift RNG for deterministic random configs.
    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i32
        }
    }

    #[test]
    fn deblock_apply_sse_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        const W: usize = 64;
        const H: usize = 64;
        let mut rng = R(0x9e3779b97f4a7c15);
        for _ in 0..40_000 {
            // random plane
            let mut base_buf = vec![0u8; W * H];
            for b in base_buf.iter_mut() {
                *b = (rng.next() & 0xff) as u8;
            }
            // pick orientation: 0 => vertical (stride_line=1), 1 => horizontal
            let vertical = (rng.next() & 1) == 0;
            let (stride_line, stride_tap): (isize, isize) = if vertical {
                (1, W as isize)
            } else {
                (W as isize, 1)
            };
            // off centred so all taps (<=8) and 4 lines stay in-bounds
            let row = rng.range(16, 40) as isize;
            let col = rng.range(16, 40) as isize;
            let off = row * W as isize + col;
            let width_neg = rng.range(1, 8);
            let width_pos = rng.range(1, 8);
            let q_thr_clamp = rng.range(0, 4000);
            let neg_lossless = (rng.next() & 7) == 0;
            let pos_lossless = (rng.next() & 7) == 0;

            let mut a = base_buf.clone();
            let mut b = base_buf.clone();
            deblock_apply_8bpc_scalar(
                &mut a,
                off,
                stride_line,
                stride_tap,
                width_neg,
                width_pos,
                q_thr_clamp,
                neg_lossless,
                pos_lossless,
            );
            unsafe {
                super::deblock_apply_8bpc_avx2(
                    &mut b,
                    off,
                    stride_line,
                    stride_tap,
                    width_neg,
                    width_pos,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                );
            }
            assert_eq!(
                a, b,
                "mismatch vertical={vertical} wn={width_neg} wp={width_pos} qc={q_thr_clamp}"
            );
        }
    }

    #[test]
    fn deblock_apply_hbd_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        const W: usize = 64;
        const H: usize = 64;
        let mut rng = R(0x243f6a8885a308d3);
        for &bitdepth_max in &[1023, 4095] {
            for _ in 0..20_000 {
                let mut base_buf = vec![0u16; W * H];
                for b in base_buf.iter_mut() {
                    *b = (rng.next() % (bitdepth_max as u64 + 1)) as u16;
                }
                let vertical = (rng.next() & 1) == 0;
                let (stride_line, stride_tap): (isize, isize) = if vertical {
                    (1, W as isize)
                } else {
                    (W as isize, 1)
                };
                let row = rng.range(16, 40) as isize;
                let col = rng.range(16, 40) as isize;
                let off = row * W as isize + col;
                let width_neg = rng.range(1, 8);
                let width_pos = rng.range(1, 8);
                let q_thr_clamp = rng.range(0, 8000);
                let neg_lossless = (rng.next() & 7) == 0;
                let pos_lossless = (rng.next() & 7) == 0;

                let mut a = base_buf.clone();
                let mut b = base_buf.clone();
                deblock_apply_hbd_scalar(
                    &mut a,
                    off,
                    stride_line,
                    stride_tap,
                    width_neg,
                    width_pos,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                    bitdepth_max,
                );
                unsafe {
                    super::deblock_apply_hbd_avx2(
                        &mut b,
                        off,
                        stride_line,
                        stride_tap,
                        width_neg,
                        width_pos,
                        q_thr_clamp,
                        neg_lossless,
                        pos_lossless,
                        bitdepth_max,
                    );
                }
                assert_eq!(
                    a, b,
                    "hbd mismatch vertical={vertical} wn={width_neg} wp={width_pos} qc={q_thr_clamp} bdmax={bitdepth_max}"
                );
            }
        }
    }
}
