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

#[inline(always)]
fn load_i16x4_i32(a: &[i16; 4]) -> __m128i {
    unsafe { _mm_cvtepi16_epi32(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i32x4_u16(a: &mut [u16; 4], v: __m128i) {
    unsafe {
        let p16 = _mm_packus_epi32(v, v);
        _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, p16);
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn constrain_v(diff: __m128i, threshold: __m128i, shc: __m128i) -> __m128i {
    let adiff = _mm_abs_epi32(diff);
    let t = _mm_max_epi32(
        _mm_setzero_si128(),
        _mm_sub_epi32(threshold, _mm_srl_epi32(adiff, shc)),
    );
    let m = _mm_min_epi32(adiff, t);
    _mm_blendv_epi8(
        m,
        _mm_sub_epi32(_mm_setzero_si128(), m),
        _mm_cmpgt_epi32(_mm_setzero_si128(), diff),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn mul_i32x4_i16_n(v: __m128i, k: i32) -> __m128i {
    let v16 = _mm_packs_epi32(v, _mm_setzero_si128());
    let vz = _mm_unpacklo_epi16(v16, _mm_setzero_si128());
    let kz = _mm_set1_epi32((k as i16 as u16) as i32);
    _mm_madd_epi16(vz, kz)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cdef_filter_block_hbd_sse41(
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
    let has_pri = pri_strength != 0;
    let has_sec = sec_strength != 0;
    let clip = has_pri && has_sec;
    let pri_s = _mm_set1_epi32(pri_strength);
    let sec_s = _mm_set1_epi32(sec_strength);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm_setzero_si128();
    let eight = _mm_set1_epi32(8);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 4;
    let mut dp = dst_off;
    let mut tp = o;

    for _y in 0..h {
        for g in 0..groups {
            let bx = g * 4;
            let tpx = (tp + bx) as isize;
            let load = |off: isize| {
                load_i16x4_i32((&tmp[(tpx + off) as usize..][..4]).try_into().unwrap())
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
                    sum = _mm_add_epi32(
                        sum,
                        mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(p0, px), pri_s, pri_shc), pt),
                    );
                    sum = _mm_add_epi32(
                        sum,
                        mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(p1, px), pri_s, pri_shc), pt),
                    );
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = _mm_min_epi32(min_v, _mm_min_epi32(p0, p1));
                        max_v = _mm_max_epi32(max_v, _mm_max_epi32(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = 2 - k as i32;
                        sum = _mm_add_epi32(
                            sum,
                            mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s0, px), sec_s, sec_shc), st),
                        );
                        sum = _mm_add_epi32(
                            sum,
                            mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s1, px), sec_s, sec_shc), st),
                        );
                        sum = _mm_add_epi32(
                            sum,
                            mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s2, px), sec_s, sec_shc), st),
                        );
                        sum = _mm_add_epi32(
                            sum,
                            mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s3, px), sec_s, sec_shc), st),
                        );
                        min_v = _mm_min_epi32(
                            min_v,
                            _mm_min_epi32(_mm_min_epi32(s0, s1), _mm_min_epi32(s2, s3)),
                        );
                        max_v = _mm_max_epi32(
                            max_v,
                            _mm_max_epi32(_mm_max_epi32(s0, s1), _mm_max_epi32(s2, s3)),
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
                    sum = _mm_add_epi32(
                        sum,
                        mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s0, px), sec_s, sec_shc), st),
                    );
                    sum = _mm_add_epi32(
                        sum,
                        mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s1, px), sec_s, sec_shc), st),
                    );
                    sum = _mm_add_epi32(
                        sum,
                        mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s2, px), sec_s, sec_shc), st),
                    );
                    sum = _mm_add_epi32(
                        sum,
                        mul_i32x4_i16_n(constrain_v(_mm_sub_epi32(s3, px), sec_s, sec_shc), st),
                    );
                }
            }

            let mask = _mm_cmpgt_epi32(zero, sum);
            let delta = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_add_epi32(sum, mask), eight));
            let mut res = _mm_add_epi32(px, delta);
            if clip {
                res = _mm_min_epi32(_mm_max_epi32(res, min_v), max_v);
            }
            store_i32x4_u16((&mut dst[dp + bx..dp + bx + 4]).try_into().unwrap(), res);
        }
        dp += dst_stride;
        tp += tmp_stride;
    }
}
