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
pub(crate) fn residual_add_row_8bpc_sse41(
    dst: &mut [u8],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = _mm_set1_epi32(rnd);
    let sh = _mm_cvtsi32_si128(shift);
    let mut x = 0;
    while x + 8 <= n {
        let c0 = _mm_loadu_si128(c[x..].as_ptr() as *const __m128i);
        let c1 = _mm_loadu_si128(c[x + 4..].as_ptr() as *const __m128i);
        let cf0 = _mm_sra_epi32(_mm_add_epi32(c0, rnd_v), sh);
        let cf1 = _mm_sra_epi32(_mm_add_epi32(c1, rnd_v), sh);
        let dpix = _mm_loadl_epi64(dst[x..].as_ptr() as *const __m128i);
        let d0 = _mm_cvtepu8_epi32(dpix);
        let d1 = _mm_cvtepu8_epi32(_mm_srli_si128(dpix, 4));
        let r16 = _mm_packs_epi32(_mm_add_epi32(d0, cf0), _mm_add_epi32(d1, cf1));
        let r8 = _mm_packus_epi16(r16, r16);
        _mm_storel_epi64(dst[x..].as_mut_ptr() as *mut __m128i, r8);
        x += 8;
    }
    while x < n {
        let v = (c[x] + rnd) >> shift;
        dst[x] = (dst[x] as i32 + v).clamp(0, 255) as u8;
        x += 1;
    }
}

/// 8-bit DC add: `dst[i] = clip(dst[i] + dc, 0, 255)`, 8 px per iteration.
/// Same double-saturating pack as the residual add, so bit-exact with scalar.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn dc_add_row_8bpc_sse41(dst: &mut [u8], dc: i32, n: usize) {
    let dc_v = _mm_set1_epi32(dc);
    let mut x = 0;
    while x + 8 <= n {
        let dpix = _mm_loadl_epi64(dst[x..].as_ptr() as *const __m128i);
        let d0 = _mm_cvtepu8_epi32(dpix);
        let d1 = _mm_cvtepu8_epi32(_mm_srli_si128(dpix, 4));
        let r16 = _mm_packs_epi32(_mm_add_epi32(d0, dc_v), _mm_add_epi32(d1, dc_v));
        let r8 = _mm_packus_epi16(r16, r16);
        _mm_storel_epi64(dst[x..].as_mut_ptr() as *mut __m128i, r8);
        x += 8;
    }
    while x < n {
        dst[x] = (dst[x] as i32 + dc).clamp(0, 255) as u8;
        x += 1;
    }
}

#[cfg(test)]
mod residual_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn scalar(dst: &mut [u8], c: &[i32], n: usize, rnd: i32, shift: i32) {
        for i in 0..n {
            dst[i] = ((dst[i] as i32) + ((c[i] + rnd) >> shift)).clamp(0, 255) as u8;
        }
    }

    #[test]
    fn residual_8bpc_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..30000 {
            let n = rng.range(1, 40) as usize; // exercises 8-wide body + scalar tail
            let shift = rng.range(0, 13); // includes shift==0 (WHT path)
            let rnd = if shift == 0 {
                0
            } else {
                rng.range(0, 1 << shift)
            };
            let mut c = vec![0i32; n + 8];
            for v in c.iter_mut() {
                // mix ranges: small, mid, and large enough to overflow i16 at the
                // pack (so clip-to-[0,255] via double-saturate is exercised)
                *v = match rng.next() % 4 {
                    0 => rng.range(-400, 400),
                    1 => rng.range(-100_000, 100_000),
                    2 => rng.range(-(1 << 24), 1 << 24),
                    _ => 0,
                };
            }
            let mut base = vec![0u8; n + 8];
            for v in base.iter_mut() {
                *v = rng.range(0, 255) as u8;
            }
            let mut a = base.clone();
            let mut b = base.clone();
            scalar(&mut a, &c, n, rnd, shift);
            unsafe { residual_add_row_8bpc_sse41(&mut b, &c, n, rnd, shift) };
            assert_eq!(a, b, "n={n} shift={shift} rnd={rnd}");
        }
    }

    #[test]
    fn dc_8bpc_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0xfeed_face_0bad_c0de);
        for _ in 0..30000 {
            let n = rng.range(1, 40) as usize;
            let dc = match rng.next() % 4 {
                0 => rng.range(-400, 400),
                1 => rng.range(-100_000, 100_000),
                2 => rng.range(-(1 << 24), 1 << 24),
                _ => 0,
            };
            let mut base = vec![0u8; n + 8];
            for v in base.iter_mut() {
                *v = rng.range(0, 255) as u8;
            }
            let mut a = base.clone();
            let mut b = base.clone();
            for i in 0..n {
                a[i] = ((a[i] as i32) + dc).clamp(0, 255) as u8;
            }
            unsafe { dc_add_row_8bpc_sse41(&mut b, dc, n) };
            assert_eq!(a, b, "n={n} dc={dc}");
        }
    }
}

/// itx row-clip: `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)`, 4 i32/iter.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn row_clip_sse41(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = _mm_set1_epi32(rnd);
    let sh = _mm_cvtsi32_si128(shift);
    let min_v = _mm_set1_epi32(min);
    let max_v = _mm_set1_epi32(max);
    let mut x = 0;
    while x + 4 <= n {
        let v = _mm_loadu_si128(tmp[x..].as_ptr() as *const __m128i);
        let v = _mm_sra_epi32(_mm_add_epi32(v, rnd_v), sh);
        let v = _mm_min_epi32(_mm_max_epi32(v, min_v), max_v);
        _mm_storeu_si128(tmp[x..].as_mut_ptr() as *mut __m128i, v);
        x += 4;
    }
    while x < n {
        tmp[x] = ((tmp[x] + rnd) >> shift).max(min).min(max);
        x += 1;
    }
}

/// cctx rotate+clip over two i32 planes, 4 lanes/iter. `cmpgt(0, a)` yields the
/// `-1` mask where `a < 0`, so `a + 128 + mask == a + 128 - (a < 0)`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn cctx_row_sse41(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    let sina_v = _mm_set1_epi32(sina);
    let cosa_v = _mm_set1_epi32(cosa);
    let c128 = _mm_set1_epi32(128);
    let zero = _mm_setzero_si128();
    let min_v = _mm_set1_epi32(min);
    let max_v = _mm_set1_epi32(max);
    let sh8 = _mm_cvtsi32_si128(8);
    let mut i = 0;
    while i + 4 <= sz {
        let uu = _mm_loadu_si128(u[i..].as_ptr() as *const __m128i);
        let vv = _mm_loadu_si128(v[i..].as_ptr() as *const __m128i);
        let a = _mm_sub_epi32(_mm_mullo_epi32(uu, cosa_v), _mm_mullo_epi32(vv, sina_v));
        let b = _mm_add_epi32(_mm_mullo_epi32(uu, sina_v), _mm_mullo_epi32(vv, cosa_v));
        let ra = _mm_sra_epi32(
            _mm_add_epi32(_mm_add_epi32(a, c128), _mm_cmpgt_epi32(zero, a)),
            sh8,
        );
        let rb = _mm_sra_epi32(
            _mm_add_epi32(_mm_add_epi32(b, c128), _mm_cmpgt_epi32(zero, b)),
            sh8,
        );
        let ra = _mm_min_epi32(_mm_max_epi32(ra, min_v), max_v);
        let rb = _mm_min_epi32(_mm_max_epi32(rb, min_v), max_v);
        _mm_storeu_si128(u[i..].as_mut_ptr() as *mut __m128i, ra);
        _mm_storeu_si128(v[i..].as_mut_ptr() as *mut __m128i, rb);
        i += 4;
    }
    while i < sz {
        let a = u[i] * cosa - v[i] * sina;
        let b = u[i] * sina + v[i] * cosa;
        u[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        v[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
        i += 1;
    }
}

/// `dst[x] = clip((t1[x] + t2[x] + rnd) >> sh, 0, 255)`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn avg_row_8bpc_sse41(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let mut x = 0;
    while x + 8 <= n {
        let a = _mm_loadu_si128(t1[x..].as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(t2[x..].as_ptr() as *const __m128i);
        let a_lo = _mm_cvtepi16_epi32(a);
        let a_hi = _mm_cvtepi16_epi32(_mm_srli_si128(a, 8));
        let b_lo = _mm_cvtepi16_epi32(b);
        let b_hi = _mm_cvtepi16_epi32(_mm_srli_si128(b, 8));
        let lo = _mm_sra_epi32(_mm_add_epi32(_mm_add_epi32(a_lo, b_lo), rnd_v), shc);
        let hi = _mm_sra_epi32(_mm_add_epi32(_mm_add_epi32(a_hi, b_hi), rnd_v), shc);
        let p16 = _mm_packs_epi32(lo, hi);
        _mm_storel_epi64(
            dst[x..].as_mut_ptr() as *mut __m128i,
            _mm_packus_epi16(p16, p16),
        );
        x += 8;
    }
    while x < n {
        dst[x] = ((t1[x] as i32 + t2[x] as i32 + rnd) >> sh).clamp(0, 255) as u8;
        x += 1;
    }
}

/// `dst[x] = clip((t1[x]*weight + t2[x]*(16-weight) + rnd) >> sh, 0, 255)`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn w_avg_row_8bpc_sse41(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    let w1 = _mm_set1_epi32(weight);
    let w2 = _mm_set1_epi32(16 - weight);
    let rnd_v = _mm_set1_epi32(rnd);
    let shc = _mm_cvtsi32_si128(sh);
    let mut x = 0;
    while x + 8 <= n {
        let a = _mm_loadu_si128(t1[x..].as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(t2[x..].as_ptr() as *const __m128i);
        let a_lo = _mm_cvtepi16_epi32(a);
        let a_hi = _mm_cvtepi16_epi32(_mm_srli_si128(a, 8));
        let b_lo = _mm_cvtepi16_epi32(b);
        let b_hi = _mm_cvtepi16_epi32(_mm_srli_si128(b, 8));
        let lo = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a_lo, w1), _mm_mullo_epi32(b_lo, w2)),
                rnd_v,
            ),
            shc,
        );
        let hi = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(a_hi, w1), _mm_mullo_epi32(b_hi, w2)),
                rnd_v,
            ),
            shc,
        );
        let p16 = _mm_packs_epi32(lo, hi);
        _mm_storel_epi64(
            dst[x..].as_mut_ptr() as *mut __m128i,
            _mm_packus_epi16(p16, p16),
        );
        x += 8;
    }
    while x < n {
        dst[x] = ((t1[x] as i32 * weight + t2[x] as i32 * (16 - weight) + rnd) >> sh).clamp(0, 255)
            as u8;
        x += 1;
    }
}

/// `dst[x] = clip((t1[x]*m + t2[x]*(64-m) + rnd) >> sh, 0, 255)`, `m = mask[x]`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn mask_row_8bpc_sse41(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = _mm_set1_epi32(rnd);
    let c64 = _mm_set1_epi32(64);
    let shc = _mm_cvtsi32_si128(sh);
    let mut x = 0;
    while x + 8 <= n {
        let a = _mm_loadu_si128(t1[x..].as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(t2[x..].as_ptr() as *const __m128i);
        let mv = _mm_loadl_epi64(mask[x..].as_ptr() as *const __m128i);
        let a_lo = _mm_cvtepi16_epi32(a);
        let a_hi = _mm_cvtepi16_epi32(_mm_srli_si128(a, 8));
        let b_lo = _mm_cvtepi16_epi32(b);
        let b_hi = _mm_cvtepi16_epi32(_mm_srli_si128(b, 8));
        let m_lo = _mm_cvtepu8_epi32(mv);
        let m_hi = _mm_cvtepu8_epi32(_mm_srli_si128(mv, 4));
        let lo = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(a_lo, m_lo),
                    _mm_mullo_epi32(b_lo, _mm_sub_epi32(c64, m_lo)),
                ),
                rnd_v,
            ),
            shc,
        );
        let hi = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(a_hi, m_hi),
                    _mm_mullo_epi32(b_hi, _mm_sub_epi32(c64, m_hi)),
                ),
                rnd_v,
            ),
            shc,
        );
        let p16 = _mm_packs_epi32(lo, hi);
        _mm_storel_epi64(
            dst[x..].as_mut_ptr() as *mut __m128i,
            _mm_packus_epi16(p16, p16),
        );
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        dst[x] = ((t1[x] as i32 * m + t2[x] as i32 * (64 - m) + rnd) >> sh).clamp(0, 255) as u8;
        x += 1;
    }
}

/// `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6`, `m = mask[x]` (in-range, no clip).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn blend_row_8bpc_sse41(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    let c64 = _mm_set1_epi32(64);
    let rnd_v = _mm_set1_epi32(32);
    let sh6 = _mm_cvtsi32_si128(6);
    let mut x = 0;
    while x + 8 <= n {
        let dv = _mm_loadl_epi64(dst[x..].as_ptr() as *const __m128i);
        let tv = _mm_loadl_epi64(tmp[x..].as_ptr() as *const __m128i);
        let mv = _mm_loadl_epi64(mask[x..].as_ptr() as *const __m128i);
        let d_lo = _mm_cvtepu8_epi32(dv);
        let d_hi = _mm_cvtepu8_epi32(_mm_srli_si128(dv, 4));
        let t_lo = _mm_cvtepu8_epi32(tv);
        let t_hi = _mm_cvtepu8_epi32(_mm_srli_si128(tv, 4));
        let m_lo = _mm_cvtepu8_epi32(mv);
        let m_hi = _mm_cvtepu8_epi32(_mm_srli_si128(mv, 4));
        let lo = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(d_lo, _mm_sub_epi32(c64, m_lo)),
                    _mm_mullo_epi32(t_lo, m_lo),
                ),
                rnd_v,
            ),
            sh6,
        );
        let hi = _mm_sra_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_mullo_epi32(d_hi, _mm_sub_epi32(c64, m_hi)),
                    _mm_mullo_epi32(t_hi, m_hi),
                ),
                rnd_v,
            ),
            sh6,
        );
        let p16 = _mm_packs_epi32(lo, hi);
        _mm_storel_epi64(
            dst[x..].as_mut_ptr() as *mut __m128i,
            _mm_packus_epi16(p16, p16),
        );
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        let d = dst[x] as i32;
        let t = tmp[x] as i32;
        dst[x] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
        x += 1;
    }
}

/// `dst[x] = clip((alpha*dst[x] + beta) >> 8, 0, 255)`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn morph_row_8bpc_sse41(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    let a_v = _mm_set1_epi32(alpha);
    let b_v = _mm_set1_epi32(beta);
    let sh8 = _mm_cvtsi32_si128(8);
    let mut x = 0;
    while x + 8 <= n {
        let dv = _mm_loadl_epi64(dst[x..].as_ptr() as *const __m128i);
        let d_lo = _mm_cvtepu8_epi32(dv);
        let d_hi = _mm_cvtepu8_epi32(_mm_srli_si128(dv, 4));
        let lo = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(d_lo, a_v), b_v), sh8);
        let hi = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(d_hi, a_v), b_v), sh8);
        let p16 = _mm_packs_epi32(lo, hi);
        _mm_storel_epi64(
            dst[x..].as_mut_ptr() as *mut __m128i,
            _mm_packus_epi16(p16, p16),
        );
        x += 8;
    }
    while x < n {
        dst[x] = ((alpha * dst[x] as i32 + beta) >> 8).clamp(0, 255) as u8;
        x += 1;
    }
}

/// GDF residual add: `dst[x] = clip(dst[x] + sign(d)*((|d|+8)>>4), 0, 255)`,
/// `d = err[x]*scale`. `cmpgt(0, d)` is the `-1` mask where `d < 0`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_add_run_8bpc_sse41(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let sc = _mm_set1_epi32(scale);
    let rnd = _mm_set1_epi32(8);
    let sh4 = _mm_cvtsi32_si128(4);
    let zero = _mm_setzero_si128();
    let mut x = 0;
    while x + 8 <= n {
        let ev = _mm_loadl_epi64(err[x..].as_ptr() as *const __m128i);
        let e_lo = _mm_cvtepi8_epi32(ev);
        let e_hi = _mm_cvtepi8_epi32(_mm_srli_si128(ev, 4));
        let diff_lo = _mm_mullo_epi32(e_lo, sc);
        let diff_hi = _mm_mullo_epi32(e_hi, sc);
        let mag_lo = _mm_sra_epi32(_mm_add_epi32(_mm_abs_epi32(diff_lo), rnd), sh4);
        let mag_hi = _mm_sra_epi32(_mm_add_epi32(_mm_abs_epi32(diff_hi), rnd), sh4);
        let adj_lo = _mm_blendv_epi8(
            mag_lo,
            _mm_sub_epi32(zero, mag_lo),
            _mm_cmpgt_epi32(zero, diff_lo),
        );
        let adj_hi = _mm_blendv_epi8(
            mag_hi,
            _mm_sub_epi32(zero, mag_hi),
            _mm_cmpgt_epi32(zero, diff_hi),
        );
        let dv = _mm_loadl_epi64(dst[x..].as_ptr() as *const __m128i);
        let r_lo = _mm_add_epi32(_mm_cvtepu8_epi32(dv), adj_lo);
        let r_hi = _mm_add_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(dv, 4)), adj_hi);
        let p16 = _mm_packs_epi32(r_lo, r_hi);
        _mm_storel_epi64(
            dst[x..].as_mut_ptr() as *mut __m128i,
            _mm_packus_epi16(p16, p16),
        );
        x += 8;
    }
    while x < n {
        let diff = err[x] as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let adj = if diff < 0 { -mag } else { mag };
        dst[x] = (dst[x] as i32 + adj).clamp(0, 255) as u8;
        x += 1;
    }
}

#[cfg(test)]
mod rowops_sse_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
        fn i16(&mut self) -> i16 {
            self.range(-32768, 32767) as i16
        }
        fn u8(&mut self) -> u8 {
            self.range(0, 255) as u8
        }
    }

    #[test]
    fn avg_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a01);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let sh = rng.range(1, 14);
            let rnd = rng.range(0, 1 << sh);
            let t1: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let t2: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                a[x] = ((t1[x] as i32 + t2[x] as i32 + rnd) >> sh).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { avg_row_8bpc_sse41(&mut b, &t1, &t2, n, rnd, sh) };
            assert_eq!(a, b, "n={n} sh={sh} rnd={rnd}");
        }
    }

    #[test]
    fn w_avg_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a02);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let sh = rng.range(1, 14);
            let rnd = rng.range(0, 1 << sh);
            let weight = rng.range(0, 16);
            let t1: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let t2: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                a[x] = ((t1[x] as i32 * weight + t2[x] as i32 * (16 - weight) + rnd) >> sh)
                    .clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { w_avg_row_8bpc_sse41(&mut b, &t1, &t2, n, weight, rnd, sh) };
            assert_eq!(a, b, "n={n} sh={sh} w={weight}");
        }
    }

    #[test]
    fn mask_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a03);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let sh = rng.range(1, 14);
            let rnd = rng.range(0, 1 << sh);
            let t1: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let t2: Vec<i16> = (0..n + 8).map(|_| rng.i16()).collect();
            let mask: Vec<u8> = (0..n + 8).map(|_| rng.range(0, 64) as u8).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                let m = mask[x] as i32;
                a[x] =
                    ((t1[x] as i32 * m + t2[x] as i32 * (64 - m) + rnd) >> sh).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { mask_row_8bpc_sse41(&mut b, &t1, &t2, &mask, n, rnd, sh) };
            assert_eq!(a, b, "n={n} sh={sh}");
        }
    }

    #[test]
    fn blend_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a04);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let tmp: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mask: Vec<u8> = (0..n + 8).map(|_| rng.range(0, 64) as u8).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                let m = mask[x] as i32;
                let d = a[x] as i32;
                let t = tmp[x] as i32;
                a[x] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
            }
            let mut b = base.clone();
            unsafe { blend_row_8bpc_sse41(&mut b, &tmp, &mask, n) };
            assert_eq!(a, b, "n={n}");
        }
    }

    #[test]
    fn morph_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a05);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let alpha = rng.range(-100_000, 100_000);
            let beta = rng.range(-(1 << 20), 1 << 20);
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                a[x] = ((alpha * a[x] as i32 + beta) >> 8).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { morph_row_8bpc_sse41(&mut b, alpha, beta, n) };
            assert_eq!(a, b, "n={n} alpha={alpha} beta={beta}");
        }
    }

    #[test]
    fn gdf_add_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a06);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let scale = rng.range(0, 100_000);
            let err: Vec<i8> = (0..n + 8).map(|_| rng.range(-128, 127) as i8).collect();
            let base: Vec<u8> = (0..n + 8).map(|_| rng.u8()).collect();
            let mut a = base.clone();
            for x in 0..n {
                let diff = err[x] as i32 * scale;
                let mag = (diff.abs() + 8) >> 4;
                let adj = if diff < 0 { -mag } else { mag };
                a[x] = (a[x] as i32 + adj).clamp(0, 255) as u8;
            }
            let mut b = base.clone();
            unsafe { gdf_add_run_8bpc_sse41(&mut b, &err, scale, n) };
            assert_eq!(a, b, "n={n} scale={scale}");
        }
    }

    #[test]
    fn row_clip_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a07);
        for _ in 0..20000 {
            let n = rng.range(1, 40) as usize;
            let shift = rng.range(0, 14);
            let rnd = if shift == 0 {
                0
            } else {
                rng.range(0, 1 << shift)
            };
            let min = rng.range(-(1 << 20), 0);
            let max = rng.range(0, 1 << 20);
            let src: Vec<i32> = (0..n + 8)
                .map(|_| match rng.next() % 3 {
                    0 => rng.range(-1000, 1000),
                    1 => rng.range(-(1 << 28), 1 << 28),
                    _ => 0,
                })
                .collect();
            let mut a = src.clone();
            for x in 0..n {
                a[x] = ((a[x] + rnd) >> shift).max(min).min(max);
            }
            let mut b = src.clone();
            unsafe { row_clip_sse41(&mut b, n, rnd, shift, min, max) };
            assert_eq!(a, b, "n={n} shift={shift}");
        }
    }

    #[test]
    fn cctx_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let mut rng = Rng(0x0a08);
        for _ in 0..20000 {
            let sz = rng.range(1, 40) as usize;
            // bound so u*cosa - v*sina cannot overflow i32 (matches decoder ranges).
            let sina = rng.range(-256, 256);
            let cosa = rng.range(-256, 256);
            let min = rng.range(-(1 << 20), 0);
            let max = rng.range(0, 1 << 20);
            let u0: Vec<i32> = (0..sz + 8)
                .map(|_| rng.range(-(1 << 22), 1 << 22))
                .collect();
            let v0: Vec<i32> = (0..sz + 8)
                .map(|_| rng.range(-(1 << 22), 1 << 22))
                .collect();
            let (mut ua, mut va) = (u0.clone(), v0.clone());
            for i in 0..sz {
                let a = ua[i] * cosa - va[i] * sina;
                let b = ua[i] * sina + va[i] * cosa;
                ua[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
                va[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
            }
            let (mut ub, mut vb) = (u0.clone(), v0.clone());
            unsafe { cctx_row_sse41(&mut ub, &mut vb, sina, cosa, sz, min, max) };
            assert_eq!(ua, ub, "u sz={sz}");
            assert_eq!(va, vb, "v sz={sz}");
        }
    }
}

/// GDF gradient: per-column `|2*b - a - c|` (each `>> shift`) summed over the 2
/// rows into 8 lanes, then pair-reduced to `ncells` cells via `hadd`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn gdf_gradient_group_sse41(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    center_rows: [&[u8]; 2],
    a_rows: [&[u8]; 2],
    c_rows: [&[u8]; 2],
    col0: usize,
    dx: i32,
    shift: u32,
) {
    let shc = _mm_cvtsi32_si128(shift as i32);
    let mut acc_lo = _mm_setzero_si128();
    let mut acc_hi = _mm_setzero_si128();
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let bv = _mm_loadl_epi64(center_rows[y][bcol..].as_ptr() as *const __m128i);
        let av = _mm_loadl_epi64(a_rows[y][acol..].as_ptr() as *const __m128i);
        let cv = _mm_loadl_epi64(c_rows[y][ccol..].as_ptr() as *const __m128i);
        let b_lo = _mm_srl_epi32(_mm_cvtepu8_epi32(bv), shc);
        let b_hi = _mm_srl_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(bv, 4)), shc);
        let a_lo = _mm_srl_epi32(_mm_cvtepu8_epi32(av), shc);
        let a_hi = _mm_srl_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(av, 4)), shc);
        let c_lo = _mm_srl_epi32(_mm_cvtepu8_epi32(cv), shc);
        let c_hi = _mm_srl_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(cv, 4)), shc);
        let t_lo = _mm_sub_epi32(_mm_sub_epi32(_mm_add_epi32(b_lo, b_lo), a_lo), c_lo);
        let t_hi = _mm_sub_epi32(_mm_sub_epi32(_mm_add_epi32(b_hi, b_hi), a_hi), c_hi);
        acc_lo = _mm_add_epi32(acc_lo, _mm_abs_epi32(t_lo));
        acc_hi = _mm_add_epi32(acc_hi, _mm_abs_epi32(t_hi));
    }
    // hadd pairs adjacent lanes: [a0+a1, a2+a3, b0+b1, b2+b3].
    let pair = _mm_hadd_epi32(acc_lo, acc_hi);
    let mut out = [0i32; 4];
    _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, pair);
    for k in 0..ncells {
        dst[base_cell + k][d] = out[k] as u16;
    }
}

#[cfg(test)]
mod gdf_gradient_sse_test {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn scalar(
        dst: &mut [[u16; 4]],
        d: usize,
        base_cell: usize,
        ncells: usize,
        center_rows: [&[u8]; 2],
        a_rows: [&[u8]; 2],
        c_rows: [&[u8]; 2],
        col0: usize,
        dx: i32,
        shift: u32,
    ) {
        let mut acc = [0i32; 8];
        for y in 0..2 {
            let bcol = col0 - 1;
            let acol = (bcol as i32 - dx) as usize;
            let ccol = (bcol as i32 + dx) as usize;
            for k in 0..8 {
                let b = (center_rows[y][bcol + k] as i32) >> shift;
                let a = (a_rows[y][acol + k] as i32) >> shift;
                let c = (c_rows[y][ccol + k] as i32) >> shift;
                acc[k] += (b + b - a - c).abs();
            }
        }
        for k in 0..ncells {
            dst[base_cell + k][d] = (acc[2 * k] + acc[2 * k + 1]) as u16;
        }
    }

    #[test]
    fn gdf_gradient_sse_matches_scalar() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        let l = 64usize;
        let mut rng = Rng(0x0a09);
        for _ in 0..20000 {
            let dx = rng.range(0, 16);
            let bcol = rng.range(dx, (l as i32) - 8 - dx) as usize;
            let col0 = bcol + 1;
            let ncells = rng.range(1, 4) as usize;
            let d = rng.range(0, 3) as usize;
            let shift = rng.range(0, 7) as u32;
            let mk =
                |rng: &mut Rng| -> Vec<u8> { (0..l).map(|_| rng.range(0, 255) as u8).collect() };
            let cr = [mk(&mut rng), mk(&mut rng)];
            let ar = [mk(&mut rng), mk(&mut rng)];
            let crr = [mk(&mut rng), mk(&mut rng)];
            let mut da = [[0u16; 4]; 4];
            let mut db = [[0u16; 4]; 4];
            scalar(
                &mut da,
                d,
                0,
                ncells,
                [&cr[0], &cr[1]],
                [&ar[0], &ar[1]],
                [&crr[0], &crr[1]],
                col0,
                dx,
                shift,
            );
            unsafe {
                gdf_gradient_group_sse41(
                    &mut db,
                    d,
                    0,
                    ncells,
                    [&cr[0], &cr[1]],
                    [&ar[0], &ar[1]],
                    [&crr[0], &crr[1]],
                    col0,
                    dx,
                    shift,
                );
            }
            assert_eq!(
                da, db,
                "dx={dx} col0={col0} ncells={ncells} d={d} shift={shift}"
            );
        }
    }
}
