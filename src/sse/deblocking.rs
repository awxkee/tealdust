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
#[target_feature(enable = "sse4.1")]
fn load4_u8_i32(dst: &[u8], base: isize, stride_line: isize) -> __m128i {
    if stride_line == 1 {
        // Load four adjacent 8bpc samples and zero-extend them to four i32 lanes.
        // Keeping the four bytes packed in one i32 lane breaks the deblock math
        // and makes the vertical SIMD path unlike dav2d's lane-wise arithmetic.
        let word =
            unsafe { std::ptr::read_unaligned(dst.as_ptr().add(base as usize).cast::<i32>()) };
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(word))
    } else {
        // Four rows of the same horizontal edge.  This is still a gather, but it
        // stays register-only instead of using a temporary stack array.
        _mm_setr_epi32(
            dst[base as usize] as i32,
            dst[(base + stride_line) as usize] as i32,
            dst[(base + 2 * stride_line) as usize] as i32,
            dst[(base + 3 * stride_line) as usize] as i32,
        )
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
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
        // Register scatter of four clipped i32 lanes.  Dav2d's horizontal AVX2
        // path is a full transpose kernel; this keeps the current Rust apply
        // structure SIMD without the old store-to-stack penalty.
        dst[base as usize] = _mm_cvtsi128_si32(v) as u8;
        dst[(base + stride_line) as usize] = _mm_extract_epi32::<1>(v) as u8;
        dst[(base + 2 * stride_line) as usize] = _mm_extract_epi32::<2>(v) as u8;
        dst[(base + 3 * stride_line) as usize] = _mm_extract_epi32::<3>(v) as u8;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn deblock_apply_8bpc_sse41(
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
    let qc = _mm_set1_epi32(q_thr_clamp);
    let nqc = _mm_set1_epi32(-q_thr_clamp);
    let rnd = _mm_set1_epi32(1 << 10);
    let zero = _mm_setzero_si128();
    let v255 = _mm_set1_epi32(255);
    let three = _mm_set1_epi32(3);
    let four = _mm_set1_epi32(4);

    let d0 = load4_u8_i32(dst, off, stride_line);
    let dm1 = load4_u8_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u8_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u8_i32(dst, off - 2 * stride_tap, stride_line);
    // delta_m2 = clip(4*(3*(d0-dm1) - (dp1-dm2)), -qc, qc)
    let inner = _mm_sub_epi32(
        _mm_mullo_epi32(three, _mm_sub_epi32(d0, dm1)),
        _mm_sub_epi32(dp1, dm2),
    );
    let delta = _mm_min_epi32(_mm_max_epi32(_mm_mullo_epi32(four, inner), nqc), qc);

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
#[target_feature(enable = "sse4.1")]
fn load4_u16_i32(dst: &[u16], base: isize, stride_line: isize) -> __m128i {
    if stride_line == 1 {
        unsafe {
            _mm_cvtepu16_epi32(_mm_loadl_epi64(
                dst.as_ptr().add(base as usize) as *const __m128i
            ))
        }
    } else {
        _mm_setr_epi32(
            dst[base as usize] as i32,
            dst[(base + stride_line) as usize] as i32,
            dst[(base + 2 * stride_line) as usize] as i32,
            dst[(base + 3 * stride_line) as usize] as i32,
        )
    }
}

/// Scatter a pre-clipped (`0..=bitdepth_max`) i32x4 back to 4 HBD samples.
#[inline]
#[target_feature(enable = "sse4.1")]
fn store4_clip_u16(dst: &mut [u16], base: isize, stride_line: isize, v: __m128i) {
    if stride_line == 1 {
        let p16 = _mm_packus_epi32(v, v);
        unsafe {
            _mm_storel_epi64(dst.as_mut_ptr().add(base as usize) as *mut __m128i, p16);
        }
    } else {
        dst[base as usize] = _mm_cvtsi128_si32(v) as u16;
        dst[(base + stride_line) as usize] = _mm_extract_epi32::<1>(v) as u16;
        dst[(base + 2 * stride_line) as usize] = _mm_extract_epi32::<2>(v) as u16;
        dst[(base + 3 * stride_line) as usize] = _mm_extract_epi32::<3>(v) as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn deblock_apply_hbd_sse41(
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
    let qc = _mm_set1_epi32(q_thr_clamp);
    let nqc = _mm_set1_epi32(-q_thr_clamp);
    let rnd = _mm_set1_epi32(1 << 10);
    let zero = _mm_setzero_si128();
    let vmax = _mm_set1_epi32(bitdepth_max);
    let three = _mm_set1_epi32(3);
    let four = _mm_set1_epi32(4);

    let d0 = load4_u16_i32(dst, off, stride_line);
    let dm1 = load4_u16_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u16_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u16_i32(dst, off - 2 * stride_tap, stride_line);
    let inner = _mm_sub_epi32(
        _mm_mullo_epi32(three, _mm_sub_epi32(d0, dm1)),
        _mm_sub_epi32(dp1, dm2),
    );
    let delta = _mm_min_epi32(_mm_max_epi32(_mm_mullo_epi32(four, inner), nqc), qc);

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

#[cfg(test)]
mod deblock_sse_tests {
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
        if !std::is_x86_feature_detected!("sse4.1") {
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
                super::deblock_apply_8bpc_sse41(
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
    fn deblock_apply_hbd_sse41_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
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
                    super::deblock_apply_hbd_sse41(
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
