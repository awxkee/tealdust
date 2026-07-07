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

#[inline]
#[target_feature(enable = "avx2")]
fn load_i8x16_i16(ptr: *const i8) -> __m256i {
    unsafe { _mm256_cvtepi8_epi16(_mm_loadu_si128(ptr as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn coeff_pair(c0: i16, c1: i16) -> __m256i {
    let packed = (c0 as u16 as u32) | ((c1 as u16 as u32) << 16);
    _mm256_set1_epi32(packed as i32)
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_pair_16(
    acc_lo: __m256i,
    acc_hi: __m256i,
    coeffs: __m256i,
    k0: __m256i,
    k1: __m256i,
) -> (__m256i, __m256i) {
    let k_lo = _mm256_unpacklo_epi16(k0, k1);
    let k_hi = _mm256_unpackhi_epi16(k0, k1);
    (
        _mm256_add_epi32(acc_lo, _mm256_madd_epi16(k_lo, coeffs)),
        _mm256_add_epi32(acc_hi, _mm256_madd_epi16(k_hi, coeffs)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_pack_16(acc_lo: __m256i, acc_hi: __m256i) -> __m256i {
    let neg1 = _mm256_set1_epi32(-1);
    let adj_lo = _mm256_cmpgt_epi32(acc_lo, neg1);
    let adj_hi = _mm256_cmpgt_epi32(acc_hi, neg1);
    let acc_lo = _mm256_srai_epi32::<7>(_mm256_sub_epi32(acc_lo, adj_lo));
    let acc_hi = _mm256_srai_epi32::<7>(_mm256_sub_epi32(acc_hi, adj_hi));
    _mm256_packs_epi32(acc_lo, acc_hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn stx4_sums(kernel: &[i8], cf: &[i16], eob: usize) -> __m256i {
    let mut acc_lo = _mm256_set1_epi32(63);
    let mut acc_hi = acc_lo;
    let mut y = 0usize;
    while y <= eob {
        let c0 = unsafe { *cf.get_unchecked(y) };
        let c1 = if y < eob {
            unsafe { *cf.get_unchecked(y + 1) }
        } else {
            0
        };
        let coeffs = coeff_pair(c0, c1);
        let k0 = load_i8x16_i16(unsafe { kernel.as_ptr().add(y * 16) });
        let k1 = load_i8x16_i16(unsafe { kernel.as_ptr().add((y + 1) * 16) });
        (acc_lo, acc_hi) = madd_pair_16(acc_lo, acc_hi, coeffs, k0, k1);
        y += 2;
    }
    round_pack_16(acc_lo, acc_hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn stx8_sums(kernel: &[i8], cf: &[i16], eob: usize) -> (__m256i, __m256i, __m256i) {
    let mut acc0_lo = _mm256_set1_epi32(63);
    let mut acc0_hi = acc0_lo;
    let mut acc1_lo = acc0_lo;
    let mut acc1_hi = acc0_lo;
    let mut acc2_lo = acc0_lo;
    let mut acc2_hi = acc0_lo;

    let mut y = 0usize;
    while y <= eob {
        let c0 = unsafe { *cf.get_unchecked(y) };
        let c1 = if y < eob {
            unsafe { *cf.get_unchecked(y + 1) }
        } else {
            0
        };
        let coeffs = coeff_pair(c0, c1);
        let row0 = unsafe { kernel.as_ptr().add(y * 48) };
        let row1 = unsafe { kernel.as_ptr().add((y + 1) * 48) };

        let k0 = load_i8x16_i16(row0);
        let k1 = load_i8x16_i16(row1);
        (acc0_lo, acc0_hi) = madd_pair_16(acc0_lo, acc0_hi, coeffs, k0, k1);

        let k0 = load_i8x16_i16(unsafe { row0.add(16) });
        let k1 = load_i8x16_i16(unsafe { row1.add(16) });
        (acc1_lo, acc1_hi) = madd_pair_16(acc1_lo, acc1_hi, coeffs, k0, k1);

        let k0 = load_i8x16_i16(unsafe { row0.add(32) });
        let k1 = load_i8x16_i16(unsafe { row1.add(32) });
        (acc2_lo, acc2_hi) = madd_pair_16(acc2_lo, acc2_hi, coeffs, k0, k1);

        y += 2;
    }

    (
        round_pack_16(acc0_lo, acc0_hi),
        round_pack_16(acc1_lo, acc1_hi),
        round_pack_16(acc2_lo, acc2_hi),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x16(dst: &mut [i16], v: __m256i) {
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), v) };
}

#[inline(always)]
fn scatter_stx4_i16(cf: &mut [i16], sums: &[i16; 16], scan_out: &[u8; 16]) {
    let dst = cf.as_mut_ptr();
    let src = sums.as_ptr();
    let map = scan_out.as_ptr();
    macro_rules! st {
        ($n:expr) => {
            unsafe { *dst.add(*map.add($n) as usize) = *src.add($n) };
        };
    }
    st!(0);
    st!(1);
    st!(2);
    st!(3);
    st!(4);
    st!(5);
    st!(6);
    st!(7);
    st!(8);
    st!(9);
    st!(10);
    st!(11);
    st!(12);
    st!(13);
    st!(14);
    st!(15);
}

#[inline(always)]
fn scatter_stx8_i16(cf: &mut [i16], sums: &[i16; 48], scan_out: &[u8; 64], mapping: &[u8; 48]) {
    let dst = cf.as_mut_ptr();
    let src = sums.as_ptr();
    let scan = scan_out.as_ptr();
    let map = mapping.as_ptr();
    macro_rules! st {
        ($n:expr) => {
            unsafe { *dst.add(*scan.add(*map.add($n) as usize) as usize) = *src.add($n) };
        };
    }
    st!(0);
    st!(1);
    st!(2);
    st!(3);
    st!(4);
    st!(5);
    st!(6);
    st!(7);
    st!(8);
    st!(9);
    st!(10);
    st!(11);
    st!(12);
    st!(13);
    st!(14);
    st!(15);
    st!(16);
    st!(17);
    st!(18);
    st!(19);
    st!(20);
    st!(21);
    st!(22);
    st!(23);
    st!(24);
    st!(25);
    st!(26);
    st!(27);
    st!(28);
    st!(29);
    st!(30);
    st!(31);
    st!(32);
    st!(33);
    st!(34);
    st!(35);
    st!(36);
    st!(37);
    st!(38);
    st!(39);
    st!(40);
    st!(41);
    st!(42);
    st!(43);
    st!(44);
    st!(45);
    st!(46);
    st!(47);
}

#[inline]
#[target_feature(enable = "avx2")]
fn zero_stx4_scan_tail_i16_avx2(sums: __m256i, scan_out: &[u8; 16]) -> __m256i {
    let idx = unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(scan_out.as_ptr().cast())) };
    let ge4 = _mm256_cmpgt_epi16(idx, _mm256_set1_epi16(3));
    let lt8 = _mm256_cmpgt_epi16(_mm256_set1_epi16(8), idx);
    let mask = _mm256_and_si256(ge4, lt8);
    _mm256_blendv_epi8(sums, _mm256_setzero_si256(), mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn zero_stx8_i16_avx2(cf: &mut [i16]) {
    let zero = _mm256_setzero_si256();
    let dst = cf.as_mut_ptr() as *mut __m256i;
    unsafe {
        _mm256_storeu_si256(dst, zero);
        _mm256_storeu_si256(dst.add(1), zero);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn stxfm4_8bpc_avx2(cf: &mut [i16], kernel: &[i8], eob: usize, scan_out: &[u8; 16]) {
    debug_assert!(eob < 8);
    debug_assert!(kernel.len() >= 8 * 16);

    let sums_v = zero_stx4_scan_tail_i16_avx2(stx4_sums(kernel, cf, eob), scan_out);
    let mut sums = [0i16; 16];
    store_i16x16(&mut sums, sums_v);

    scatter_stx4_i16(cf, &sums, scan_out);
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn stxfm8_8bpc_avx2(
    cf: &mut [i16],
    kernel: &[i8],
    eob: usize,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    debug_assert!(eob < 32);
    debug_assert!(kernel.len() >= 32 * 48);

    let (s0, s1, s2) = stx8_sums(kernel, cf, eob);
    let mut sums = [0i16; 48];
    store_i16x16(&mut sums[..16], s0);
    store_i16x16(&mut sums[16..32], s1);
    store_i16x16(&mut sums[32..48], s2);
    zero_stx8_i16_avx2(cf);
    scatter_stx8_i16(cf, &sums, scan_out, mapping);
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i8x8_i32(ptr: *const i8) -> __m256i {
    unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(ptr as *const __m128i)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn mac_hbd_8(acc: __m256i, coeff: i32, kernel: *const i8) -> __m256i {
    let k = load_i8x8_i32(kernel);
    let c = _mm256_set1_epi32(coeff);
    _mm256_add_epi32(acc, _mm256_mullo_epi32(k, c))
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_clip_hbd_8(acc: __m256i, min_v: __m256i, max_v: __m256i) -> __m256i {
    let adj = _mm256_cmpgt_epi32(acc, _mm256_set1_epi32(-1));
    let v = _mm256_srai_epi32::<7>(_mm256_sub_epi32(acc, adj));
    _mm256_min_epi32(_mm256_max_epi32(v, min_v), max_v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn stx4_sums_hbd(kernel: &[i8], cf: &[i32], eob: usize, bitdepth_max: i32) -> (__m256i, __m256i) {
    let min_v = _mm256_set1_epi32(-128 * (1 + bitdepth_max));
    let max_v = _mm256_set1_epi32(128 * (1 + bitdepth_max) - 1);
    let mut acc0 = _mm256_set1_epi32(63);
    let mut acc1 = acc0;

    let mut y = 0usize;
    while y <= eob {
        let c = unsafe { *cf.get_unchecked(y) };
        let row = unsafe { kernel.as_ptr().add(y * 16) };
        acc0 = mac_hbd_8(acc0, c, row);
        acc1 = mac_hbd_8(acc1, c, unsafe { row.add(8) });
        y += 1;
    }

    (
        round_clip_hbd_8(acc0, min_v, max_v),
        round_clip_hbd_8(acc1, min_v, max_v),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn stx8_sums_hbd(
    kernel: &[i8],
    cf: &[i32],
    eob: usize,
    bitdepth_max: i32,
) -> (__m256i, __m256i, __m256i, __m256i, __m256i, __m256i) {
    let min_v = _mm256_set1_epi32(-128 * (1 + bitdepth_max));
    let max_v = _mm256_set1_epi32(128 * (1 + bitdepth_max) - 1);
    let mut acc0 = _mm256_set1_epi32(63);
    let mut acc1 = acc0;
    let mut acc2 = acc0;
    let mut acc3 = acc0;
    let mut acc4 = acc0;
    let mut acc5 = acc0;

    let mut y = 0usize;
    while y <= eob {
        let c = unsafe { *cf.get_unchecked(y) };
        let row = unsafe { kernel.as_ptr().add(y * 48) };
        acc0 = mac_hbd_8(acc0, c, row);
        acc1 = mac_hbd_8(acc1, c, unsafe { row.add(8) });
        acc2 = mac_hbd_8(acc2, c, unsafe { row.add(16) });
        acc3 = mac_hbd_8(acc3, c, unsafe { row.add(24) });
        acc4 = mac_hbd_8(acc4, c, unsafe { row.add(32) });
        acc5 = mac_hbd_8(acc5, c, unsafe { row.add(40) });
        y += 1;
    }

    (
        round_clip_hbd_8(acc0, min_v, max_v),
        round_clip_hbd_8(acc1, min_v, max_v),
        round_clip_hbd_8(acc2, min_v, max_v),
        round_clip_hbd_8(acc3, min_v, max_v),
        round_clip_hbd_8(acc4, min_v, max_v),
        round_clip_hbd_8(acc5, min_v, max_v),
    )
}

#[inline(always)]
fn scatter_stx4_i32(cf: &mut [i32], sums: &[i32; 16], scan_out: &[u8; 16]) {
    let dst = cf.as_mut_ptr();
    let src = sums.as_ptr();
    let map = scan_out.as_ptr();
    macro_rules! st {
        ($n:expr) => {
            unsafe { *dst.add(*map.add($n) as usize) = *src.add($n) };
        };
    }
    st!(0);
    st!(1);
    st!(2);
    st!(3);
    st!(4);
    st!(5);
    st!(6);
    st!(7);
    st!(8);
    st!(9);
    st!(10);
    st!(11);
    st!(12);
    st!(13);
    st!(14);
    st!(15);
}

#[inline(always)]
fn scatter_stx8_i32(cf: &mut [i32], sums: &[i32; 48], scan_out: &[u8; 64], mapping: &[u8; 48]) {
    let dst = cf.as_mut_ptr();
    let src = sums.as_ptr();
    let scan = scan_out.as_ptr();
    let map = mapping.as_ptr();
    macro_rules! st {
        ($n:expr) => {
            unsafe { *dst.add(*scan.add(*map.add($n) as usize) as usize) = *src.add($n) };
        };
    }
    st!(0);
    st!(1);
    st!(2);
    st!(3);
    st!(4);
    st!(5);
    st!(6);
    st!(7);
    st!(8);
    st!(9);
    st!(10);
    st!(11);
    st!(12);
    st!(13);
    st!(14);
    st!(15);
    st!(16);
    st!(17);
    st!(18);
    st!(19);
    st!(20);
    st!(21);
    st!(22);
    st!(23);
    st!(24);
    st!(25);
    st!(26);
    st!(27);
    st!(28);
    st!(29);
    st!(30);
    st!(31);
    st!(32);
    st!(33);
    st!(34);
    st!(35);
    st!(36);
    st!(37);
    st!(38);
    st!(39);
    st!(40);
    st!(41);
    st!(42);
    st!(43);
    st!(44);
    st!(45);
    st!(46);
    st!(47);
}

#[inline]
#[target_feature(enable = "avx2")]
fn zero_stx4_scan_tail_i32_avx2(sums: __m256i, scan_out: *const u8) -> __m256i {
    let idx = unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(scan_out.cast())) };
    let ge4 = _mm256_cmpgt_epi32(idx, _mm256_set1_epi32(3));
    let lt8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), idx);
    let mask = _mm256_and_si256(ge4, lt8);
    _mm256_blendv_epi8(sums, _mm256_setzero_si256(), mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn zero_stx8_i32_avx2(cf: &mut [i32]) {
    let zero = _mm256_setzero_si256();
    let dst = cf.as_mut_ptr() as *mut __m256i;
    unsafe {
        _mm256_storeu_si256(dst, zero);
        _mm256_storeu_si256(dst.add(1), zero);
        _mm256_storeu_si256(dst.add(2), zero);
        _mm256_storeu_si256(dst.add(3), zero);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn stxfm4_hbd_avx2(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 16],
) {
    debug_assert!(eob < 8);
    debug_assert!(kernel.len() >= 8 * 16);

    let (s0, s1) = stx4_sums_hbd(kernel, cf, eob, bitdepth_max);
    let s0 = zero_stx4_scan_tail_i32_avx2(s0, scan_out.as_ptr());
    let s1 = zero_stx4_scan_tail_i32_avx2(s1, unsafe { scan_out.as_ptr().add(8) });
    let mut sums = [0i32; 16];
    unsafe {
        _mm256_storeu_si256((&mut sums[..8]).as_mut_ptr().cast(), s0);
        _mm256_storeu_si256((&mut sums[8..16]).as_mut_ptr().cast(), s1);
    }

    scatter_stx4_i32(cf, &sums, scan_out);
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn stxfm8_hbd_avx2(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    debug_assert!(eob < 32);
    debug_assert!(kernel.len() >= 32 * 48);

    let (s0, s1, s2, s3, s4, s5) = stx8_sums_hbd(kernel, cf, eob, bitdepth_max);
    let mut sums = [0i32; 48];
    unsafe {
        _mm256_storeu_si256((&mut sums[..8]).as_mut_ptr().cast(), s0);
        _mm256_storeu_si256((&mut sums[8..16]).as_mut_ptr().cast(), s1);
        _mm256_storeu_si256((&mut sums[16..24]).as_mut_ptr().cast(), s2);
        _mm256_storeu_si256((&mut sums[24..32]).as_mut_ptr().cast(), s3);
        _mm256_storeu_si256((&mut sums[32..40]).as_mut_ptr().cast(), s4);
        _mm256_storeu_si256((&mut sums[40..48]).as_mut_ptr().cast(), s5);
    }

    zero_stx8_i32_avx2(cf);
    scatter_stx8_i32(cf, &sums, scan_out, mapping);
}
