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

#[repr(align(32))]
struct AlignedU32x8([u32; 8]);
#[repr(align(32))]
struct AlignedU64x4([u64; 4]);

// Prefix sums of the bypass interval: sum_{j<k} 2^(47-j) = rng * MUL32[k-1] << 32.
static UNARY_MUL32: [AlignedU32x8; 2] = [
    AlignedU32x8([
        0x8000, 0xC000, 0xE000, 0xF000, 0xF800, 0xFC00, 0xFE00, 0xFF00,
    ]),
    AlignedU32x8([
        0xFF80, 0xFFC0, 0xFFE0, 0xFFF0, 0xFFF8, 0xFFFC, 0xFFFE, 0xFFFF,
    ]),
];
// k = 17..=20 as prefix >> 16; k = 21 is checked with a scalar 64-bit compare.
static UNARY_MUL64: AlignedU64x4 =
    AlignedU64x4([0xFFFF_8000, 0xFFFF_C000, 0xFFFF_E000, 0xFFFF_F000]);

/// Branch-free truncated-unary bypass: compares `dif` against all interval
/// prefix sums at once instead of a serial data-dependent per-bit loop.
#[target_feature(enable = "avx2")]
pub(crate) fn unary_bypass_kernel_avx2(dif: u64, rng: u32, max_bits: u32) -> (u32, u32, u64) {
    debug_assert!(max_bits == 5 || max_bits == 6 || max_bits == 21);
    debug_assert!(rng & 1 == 0);
    debug_assert!((dif >> 48) < rng as u64);

    let stop_mask = unsafe {
        let half_rng = _mm256_set1_epi32((rng >> 1) as i32);
        let hi = _mm256_set1_epi32((dif >> 33) as i32);
        // Lane sign of hi - (rng>>1)*mul flags dif < rng * prefix_k (exact: rng is even).
        let p0 = _mm256_mullo_epi32(
            half_rng,
            _mm256_load_si256(UNARY_MUL32[0].0.as_ptr().cast()),
        );
        let m0 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sub_epi32(hi, p0))) as u32;
        if max_bits <= 8 {
            m0
        } else {
            let p1 = _mm256_mullo_epi32(
                half_rng,
                _mm256_load_si256(UNARY_MUL32[1].0.as_ptr().cast()),
            );
            let m1 = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_sub_epi32(hi, p1))) as u32;
            let hi64 = _mm256_set1_epi64x((dif >> 17) as i64);
            let p2 = _mm256_mul_epu32(
                _mm256_set1_epi64x((rng >> 1) as i64),
                _mm256_load_si256(UNARY_MUL64.0.as_ptr().cast()),
            );
            let m2 = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_sub_epi64(hi64, p2))) as u32;
            let m3 = ((dif < (rng as u64) * 0xFFFF_F800_0000) as u32) << 20;
            m0 | (m1 << 8) | (m2 << 16) | m3
        }
    };
    let q = (stop_mask | (1u32 << max_bits)).trailing_zeros();
    // Undo all q subtractions with one multiply: vw_sum = rng * (2^48 - 2^(48-q)).
    let vw_sum = (rng as u64) * ((1u64 << 48) - (1u64 << (48 - q)));
    let bits = q + (q < max_bits) as u32;
    (q, bits, ((dif - vw_sum + 1) << bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msac::unary_bypass_kernel_scalar;

    #[test]
    fn unary_kernel_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..200_000 {
            let rng = ((0x8000 | (next() & 0x7FFF)) & !1) as u32;
            let dif = next() % ((rng as u64) << 48);
            for &mb in &[5u32, 6, 21] {
                let a = unary_bypass_kernel_scalar(dif, rng, mb);
                let b = unsafe { unary_bypass_kernel_avx2(dif, rng, mb) };
                assert_eq!(a, b, "dif={dif:#x} rng={rng:#x} max_bits={mb}");
            }
        }
        for rng in [0x8000u32, 0x8002, 0x9246, 0xFFFE] {
            for q in 1..=21u32 {
                let s_q = (rng as u64) * ((1u64 << 48) - (1u64 << (48 - q)));
                for dif in [s_q.wrapping_sub(1), s_q, s_q + 1] {
                    if dif >> 48 >= rng as u64 {
                        continue;
                    }
                    for &mb in &[5u32, 6, 21] {
                        let a = unary_bypass_kernel_scalar(dif, rng, mb);
                        let b = unsafe { unary_bypass_kernel_avx2(dif, rng, mb) };
                        assert_eq!(a, b, "dif={dif:#x} rng={rng:#x} max_bits={mb}");
                    }
                }
            }
        }
    }
}
