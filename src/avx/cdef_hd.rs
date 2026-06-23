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

use std::arch::x86_64::*;

#[inline(always)]
fn load_i16x8_i32(a: &[i16; 8]) -> __m256i {
    unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i32x8_u16(a: &mut [u16; 8], v: __m256i) {
    unsafe {
        let p16 = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(v, v));
        _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(p16));
    }
}

#[inline(always)]
fn constrain_v(diff: __m256i, threshold: __m256i, shc: __m128i) -> __m256i {
    unsafe {
        let adiff = _mm256_abs_epi32(diff);
        let t = _mm256_max_epi32(
            _mm256_setzero_si256(),
            _mm256_sub_epi32(threshold, _mm256_srl_epi32(adiff, shc)),
        );
        let m = _mm256_min_epi32(adiff, t);
        _mm256_blendv_epi8(
            m,
            _mm256_sub_epi32(_mm256_setzero_si256(), m),
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), diff),
        )
    }
}

#[inline(always)]
fn mul_i32x8_i16_n(v: __m256i, k: i32) -> __m256i {
    unsafe {
        // CDEF constrain() is strength-bounded, so it fits i16. Use PMADDWD
        // as eight independent i16*tap -> i32 multiplies.
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let v16 = _mm_packs_epi32(lo, hi);
        let zero = _mm_setzero_si128();
        let loz = _mm_unpacklo_epi16(v16, zero);
        let hiz = _mm_unpackhi_epi16(v16, zero);
        let vz = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(loz), hiz);
        let kz = _mm256_set1_epi32((k as i16 as u16) as i32);
        _mm256_madd_epi16(vz, kz)
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_filter_block_hbd_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::cdef_dispatch::cdef_filter_block_hbd_scalar(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        );
        return;
    }

    let has_pri = pri_strength != 0;
    let has_sec = sec_strength != 0;
    let clip = has_pri && has_sec;
    let pri_s = _mm256_set1_epi32(pri_strength);
    let sec_s = _mm256_set1_epi32(sec_strength);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm256_setzero_si256();
    let eight = _mm256_set1_epi32(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 8;
    let mut dp = dst_off;
    let mut tp = o;

    for _y in 0..h {
        for g in 0..groups {
            let bx = g * 8;
            let tpx = (tp + bx) as isize;
            let load = |off: isize| {
                load_i16x8_i32((&tmp[(tpx + off) as usize..][..8]).try_into().unwrap())
            };
            let px = load(0);
            let mut sum = zero;
            let mut min_v = px;
            let mut max_v = px;

            if has_pri {
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off1 = dirs[dir + 2][k] as isize;
                    let p0 = load(off1);
                    let p1 = load(-off1);
                    let pt = ptap;
                    sum = _mm256_add_epi32(
                        sum,
                        mul_i32x8_i16_n(constrain_v(_mm256_sub_epi32(p0, px), pri_s, pri_shc), pt),
                    );
                    sum = _mm256_add_epi32(
                        sum,
                        mul_i32x8_i16_n(constrain_v(_mm256_sub_epi32(p1, px), pri_s, pri_shc), pt),
                    );
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = _mm256_min_epi32(min_v, _mm256_min_epi32(p0, p1));
                        max_v = _mm256_max_epi32(max_v, _mm256_max_epi32(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = 2 - k as i32;
                        sum = _mm256_add_epi32(
                            sum,
                            mul_i32x8_i16_n(
                                constrain_v(_mm256_sub_epi32(s0, px), sec_s, sec_shc),
                                st,
                            ),
                        );
                        sum = _mm256_add_epi32(
                            sum,
                            mul_i32x8_i16_n(
                                constrain_v(_mm256_sub_epi32(s1, px), sec_s, sec_shc),
                                st,
                            ),
                        );
                        sum = _mm256_add_epi32(
                            sum,
                            mul_i32x8_i16_n(
                                constrain_v(_mm256_sub_epi32(s2, px), sec_s, sec_shc),
                                st,
                            ),
                        );
                        sum = _mm256_add_epi32(
                            sum,
                            mul_i32x8_i16_n(
                                constrain_v(_mm256_sub_epi32(s3, px), sec_s, sec_shc),
                                st,
                            ),
                        );
                        min_v = _mm256_min_epi32(
                            min_v,
                            _mm256_min_epi32(_mm256_min_epi32(s0, s1), _mm256_min_epi32(s2, s3)),
                        );
                        max_v = _mm256_max_epi32(
                            max_v,
                            _mm256_max_epi32(_mm256_max_epi32(s0, s1), _mm256_max_epi32(s2, s3)),
                        );
                    }
                }
            } else {
                for k in 0..2 {
                    let off1 = dirs[dir + 4][k] as isize;
                    let off2 = dirs[dir][k] as isize;
                    let s0 = load(off1);
                    let s1 = load(-off1);
                    let s2 = load(off2);
                    let s3 = load(-off2);
                    let st = 2 - k as i32;
                    sum = _mm256_add_epi32(
                        sum,
                        mul_i32x8_i16_n(constrain_v(_mm256_sub_epi32(s0, px), sec_s, sec_shc), st),
                    );
                    sum = _mm256_add_epi32(
                        sum,
                        mul_i32x8_i16_n(constrain_v(_mm256_sub_epi32(s1, px), sec_s, sec_shc), st),
                    );
                    sum = _mm256_add_epi32(
                        sum,
                        mul_i32x8_i16_n(constrain_v(_mm256_sub_epi32(s2, px), sec_s, sec_shc), st),
                    );
                    sum = _mm256_add_epi32(
                        sum,
                        mul_i32x8_i16_n(constrain_v(_mm256_sub_epi32(s3, px), sec_s, sec_shc), st),
                    );
                }
            }

            let mask = _mm256_cmpgt_epi32(zero, sum);
            let delta =
                _mm256_srai_epi32::<4>(_mm256_add_epi32(_mm256_add_epi32(sum, mask), eight));
            let mut res = _mm256_add_epi32(px, delta);
            if clip {
                res = _mm256_min_epi32(_mm256_max_epi32(res, min_v), max_v);
            }
            store_i32x8_u16((&mut dst[dp + bx..dp + bx + 8]).try_into().unwrap(), res);
        }
        dp += dst_stride;
        tp += tmp_stride;
    }
}

#[cfg(test)]
mod cdef_hbd_simd_conformance {

    use crate::cdef_dispatch::cdef_filter_block_hbd_scalar;

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn r(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    const TMP_STRIDE: usize = 12; // baked into CDEF_DIRECTIONS offsets
    const O: usize = 2 * TMP_STRIDE + 2; // block at row 2, col 2 of the bordered tmp

    fn run_one(
        rng: &mut R,
        w: usize,
        h: usize,
        bitdepth_max: i32,
        sentinels: bool,
        pri_strength: i32,
        sec_strength: i32,
        pri_shift: i32,
        sec_shift: i32,
        pri_tap: i32,
        dir: usize,
    ) {
        let have_sse = std::is_x86_feature_detected!("sse4.1");
        let have_avx = std::is_x86_feature_detected!("avx2");
        if !have_sse && !have_avx {
            return;
        }
        let tlen = (h + 4) * TMP_STRIDE + 32;
        let mut tmp = vec![0i16; tlen];
        for t in tmp.iter_mut() {
            *t = rng.r(0, bitdepth_max) as i16;
        }
        if sentinels {
            // sprinkle off-frame sentinels (i16::MIN) the way the real pad does
            for _ in 0..(tlen / 8) {
                let i = rng.r(0, tlen as i32 - 1) as usize;
                tmp[i] = i16::MIN;
            }
            // keep the block's own pixels valid so px is never a sentinel
            for y in 0..h {
                for x in 0..w {
                    tmp[O + y * TMP_STRIDE + x] = rng.r(0, bitdepth_max) as i16;
                }
            }
        }
        let dst_stride = w;
        let mut ds = vec![0u16; h * dst_stride];
        cdef_filter_block_hbd_scalar(
            &mut ds,
            dst_stride,
            0,
            &tmp,
            TMP_STRIDE,
            O,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        );
        if have_sse {
            let mut d = vec![0u16; h * dst_stride];
            unsafe {
                crate::sse::cdef_filter_block_hbd_sse41(
                    &mut d,
                    dst_stride,
                    0,
                    &tmp,
                    TMP_STRIDE,
                    O,
                    pri_strength,
                    sec_strength,
                    pri_shift,
                    sec_shift,
                    pri_tap,
                    dir,
                    w,
                    h,
                );
            }
            assert_eq!(
                d, ds,
                "SSE w={w} h={h} bd={bitdepth_max} sent={sentinels} pri={pri_strength} sec={sec_strength} dir={dir} ptap={pri_tap}"
            );
        }
        #[cfg(feature = "avx")]
        if have_avx {
            let mut d = vec![0u16; h * dst_stride];
            unsafe {
                crate::avx::cdef_filter_block_hbd_avx2(
                    &mut d,
                    dst_stride,
                    0,
                    &tmp,
                    TMP_STRIDE,
                    O,
                    pri_strength,
                    sec_strength,
                    pri_shift,
                    sec_shift,
                    pri_tap,
                    dir,
                    w,
                    h,
                );
            }
            assert_eq!(
                d, ds,
                "AVX2 w={w} h={h} bd={bitdepth_max} sent={sentinels} pri={pri_strength} sec={sec_strength} dir={dir} ptap={pri_tap}"
            );
        }
    }

    #[test]
    fn cdef_hbd_matches_scalar() {
        let mut rng = R(0xc0ffee_cdef_2026);
        for &(w, h) in &[(4usize, 4usize), (8, 4), (8, 8), (4, 8)] {
            for &bd in &[1023i32, 4095] {
                // 10-bit and 12-bit
                for &sent in &[false, true] {
                    for dir in 0..8usize {
                        for _ in 0..60 {
                            // exercise all three active branches
                            let (pri, sec) = match rng.r(0, 2) {
                                0 => (rng.r(1, 15), rng.r(1, 4)),
                                1 => (rng.r(1, 15), 0),
                                _ => (0, rng.r(1, 4)),
                            };
                            let pri_tap = if pri != 0 {
                                [3, 4][rng.r(0, 1) as usize]
                            } else {
                                0
                            };
                            let pri_shift = rng.r(0, 6);
                            let sec_shift = rng.r(0, 6);
                            run_one(
                                &mut rng, w, h, bd, sent, pri, sec, pri_shift, sec_shift, pri_tap,
                                dir,
                            );
                        }
                    }
                }
            }
        }
    }
}
