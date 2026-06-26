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
fn load_i16x4(tmp: &[i16], p: isize, off: isize) -> __m128i {
    unsafe { _mm_loadl_epi64(tmp.as_ptr().offset(p + off).cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn cdef_min_i16(a: __m128i, b: __m128i) -> __m128i {
    _mm_min_epu16(a, b)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn constrain_i16(diff: __m128i, threshold: __m128i, shc: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let adiff = _mm_abs_epi16(diff);
    let t = _mm_max_epi16(zero, _mm_sub_epi16(threshold, _mm_srl_epi16(adiff, shc)));
    let m = _mm_min_epu16(adiff, t);
    _mm_blendv_epi8(m, _mm_sub_epi16(zero, m), _mm_cmpgt_epi16(zero, diff))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn madd_i16(sum: __m128i, v: __m128i, tap: i32) -> __m128i {
    _mm_add_epi16(sum, _mm_mullo_epi16(v, _mm_set1_epi16(tap as i16)))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn store_i16x4_u8(dst: &mut [u8], p: usize, v: __m128i) {
    let packed = _mm_packus_epi16(v, v);
    unsafe {
        _mm_store_ss(dst.as_mut_ptr().add(p).cast(), _mm_castsi128_ps(packed));
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cdef_filter_block_8bpc_sse41(
    dst: &mut [u8],
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
    let pri_s = _mm_set1_epi16(pri_strength as i16);
    let sec_s = _mm_set1_epi16(sec_strength as i16);
    let pri_shc = _mm_cvtsi32_si128(pri_shift);
    let sec_shc = _mm_cvtsi32_si128(sec_shift);
    let zero = _mm_setzero_si128();
    let eight = _mm_set1_epi16(8);
    let lowmask = _mm_set1_epi16(0xff);
    let dirs = &crate::tables::CDEF_DIRECTIONS;
    let groups = w / 4;
    let mut dp = dst_off;
    let mut tp = o;

    for _ in 0..h {
        for g in 0..groups {
            let bx = g * 4;
            let tpx = (tp + bx) as isize;
            let load = |off: isize| load_i16x4(tmp, tpx, off);
            let px = load(0);
            let mut sum = zero;
            let mut min_v = px;
            let mut max_v = px;

            if has_pri {
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off = dirs[dir + 2][k] as isize;
                    let p0 = load(off);
                    let p1 = load(-off);
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm_sub_epi16(p0, px), pri_s, pri_shc),
                        ptap,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm_sub_epi16(p1, px), pri_s, pri_shc),
                        ptap,
                    );
                    ptap = (ptap & 3) | 2;
                    if clip {
                        min_v = cdef_min_i16(min_v, cdef_min_i16(p0, p1));
                        max_v = _mm_max_epi16(max_v, _mm_max_epi16(p0, p1));
                    }
                    if has_sec {
                        let off2 = dirs[dir + 4][k] as isize;
                        let off3 = dirs[dir][k] as isize;
                        let s0 = load(off2);
                        let s1 = load(-off2);
                        let s2 = load(off3);
                        let s3 = load(-off3);
                        let st = 2 - k as i32;
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm_sub_epi16(s0, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm_sub_epi16(s1, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm_sub_epi16(s2, px), sec_s, sec_shc),
                            st,
                        );
                        sum = madd_i16(
                            sum,
                            constrain_i16(_mm_sub_epi16(s3, px), sec_s, sec_shc),
                            st,
                        );
                        min_v = cdef_min_i16(
                            min_v,
                            cdef_min_i16(cdef_min_i16(s0, s1), cdef_min_i16(s2, s3)),
                        );
                        max_v = _mm_max_epi16(
                            max_v,
                            _mm_max_epi16(_mm_max_epi16(s0, s1), _mm_max_epi16(s2, s3)),
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
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm_sub_epi16(s0, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm_sub_epi16(s1, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm_sub_epi16(s2, px), sec_s, sec_shc),
                        st,
                    );
                    sum = madd_i16(
                        sum,
                        constrain_i16(_mm_sub_epi16(s3, px), sec_s, sec_shc),
                        st,
                    );
                }
            }

            let mask = _mm_cmpgt_epi16(zero, sum);
            let delta = _mm_srai_epi16::<4>(_mm_add_epi16(_mm_add_epi16(sum, mask), eight));
            let mut res = _mm_add_epi16(px, delta);
            if clip {
                res = _mm_min_epi16(_mm_max_epi16(res, min_v), max_v);
            }
            res = _mm_and_si128(res, lowmask);
            store_i16x4_u8(dst, dp + bx, res);
        }
        dp += dst_stride;
        tp += tmp_stride;
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cdef_find_dir_8bpc_sse41(img: &[u8], stride: usize, var: &mut u32) -> i32 {
    let mut rows = [[0i16; 8]; 8];
    let bias = _mm_set1_epi16(128);
    for (y, row) in rows.iter_mut().enumerate() {
        let src = unsafe { img.as_ptr().add(y * stride) };
        let raw = unsafe { _mm_loadl_epi64(src.cast()) };
        let pix = _mm_cvtepu8_epi16(raw);
        let centered = _mm_sub_epi16(pix, bias);
        unsafe { _mm_storeu_si128(row.as_mut_ptr().cast(), centered) };
    }
    crate::cdef_dispatch::cdef_find_dir_from_i16_rows(&rows, var)
}

#[cfg(test)]
mod tests {
    use crate::cdef_dispatch::cdef_filter_block_8bpc_scalar;

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
    fn cdef_filter_sse_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        const TMP_STRIDE: usize = 12;
        const O: usize = 2 * TMP_STRIDE + 2;
        const DST_STRIDE: usize = 16;
        let mut rng = R(0xd1b54a32d192ed03);
        for _ in 0..40_000 {
            let mut tmp = [0i16; 144];
            for t in tmp.iter_mut() {
                *t = rng.range(0, 255) as i16;
            }
            // choose variant: ensure at least one strength nonzero
            let variant = rng.range(0, 2); // 0 pri+sec, 1 pri-only, 2 sec-only
            let pri_strength = if variant == 2 { 0 } else { rng.range(1, 63) };
            let sec_strength = if variant == 1 { 0 } else { rng.range(1, 63) };
            let pri_shift = rng.range(0, 7);
            let sec_shift = rng.range(0, 7);
            let pri_tap = if pri_strength != 0 {
                4 - (pri_strength & 1)
            } else {
                0
            };
            let dir = rng.range(0, 7) as usize;
            let w = if rng.range(0, 1) == 0 { 4 } else { 8 };
            let h = if rng.range(0, 1) == 0 { 4 } else { 8 };

            let mut a = [0u8; DST_STRIDE * 8];
            let mut b = [0u8; DST_STRIDE * 8];
            cdef_filter_block_8bpc_scalar(
                &mut a,
                DST_STRIDE,
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
            unsafe {
                super::cdef_filter_block_8bpc_sse41(
                    &mut b,
                    DST_STRIDE,
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
                a, b,
                "mismatch variant={variant} dir={dir} w={w} h={h} pri={pri_strength} sec={sec_strength}"
            );
        }
    }
}
