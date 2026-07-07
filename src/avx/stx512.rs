/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn load_i8x16_i16_512(ptr: *const i8) -> __m512i {
    let bytes = unsafe { _mm_loadu_si128(ptr.cast()) };
    let words = _mm256_cvtepi8_epi16(bytes);
    _mm512_zextsi256_si512(words)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn coeff_pair_512(c0: i16, c1: i16) -> __m512i {
    let pair = (c0 as u16 as u32) | ((c1 as u16 as u32) << 16);
    _mm512_set1_epi32(pair as i32)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn madd_pair_16_512(
    acc_lo: __m512i,
    acc_hi: __m512i,
    coeffs: __m512i,
    k0: __m512i,
    k1: __m512i,
) -> (__m512i, __m512i) {
    let k_lo = _mm512_unpacklo_epi16(k0, k1);
    let k_hi = _mm512_unpackhi_epi16(k0, k1);
    (
        _mm512_dpwssd_epi32(acc_lo, k_lo, coeffs),
        _mm512_dpwssd_epi32(acc_hi, k_hi, coeffs),
    )
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn round_pack_16(acc_lo: __m512i, acc_hi: __m512i) -> __m512i {
    let minus_one = _mm512_set1_epi32(-1);
    let one = _mm512_set1_epi32(1);
    let lo_mask = _mm512_cmpgt_epi32_mask(acc_lo, minus_one);
    let hi_mask = _mm512_cmpgt_epi32_mask(acc_hi, minus_one);
    let lo = _mm512_srai_epi32::<7>(_mm512_mask_add_epi32(acc_lo, lo_mask, acc_lo, one));
    let hi = _mm512_srai_epi32::<7>(_mm512_mask_add_epi32(acc_hi, hi_mask, acc_hi, one));
    _mm512_packs_epi32(lo, hi)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn stx16_sums_vnni(kernel: &[i8], cf: &[i16], eob: usize, row_stride: usize, x: usize) -> __m512i {
    let mut acc_lo = _mm512_set1_epi32(63);
    let mut acc_hi = acc_lo;

    let mut y = 0usize;
    while y <= eob {
        let c0 = unsafe { *cf.get_unchecked(y) };
        let c1 = if y < eob {
            unsafe { *cf.get_unchecked(y + 1) }
        } else {
            0
        };
        let coeffs = coeff_pair_512(c0, c1);
        let row0 = y * row_stride + x;
        let row1 = if y < eob {
            (y + 1) * row_stride + x
        } else {
            row0
        };
        let k0 = load_i8x16_i16_512(unsafe { kernel.as_ptr().add(row0) });
        let k1 = load_i8x16_i16_512(unsafe { kernel.as_ptr().add(row1) });
        (acc_lo, acc_hi) = madd_pair_16_512(acc_lo, acc_hi, coeffs, k0, k1);
        y += 2;
    }

    round_pack_16(acc_lo, acc_hi)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
fn store_low_i16x16(dst: &mut [i16], v: __m512i) {
    let v = _mm512_castsi512_si256(v);
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), v) };
}

#[inline]
fn scatter_stx4_i16(cf: &mut [i16], sums: &[i16; 16], scan_out: &[u8; 16]) {
    for (&dst, &sum) in scan_out.iter().zip(sums.iter()) {
        cf[dst as usize] = sum;
    }
}

#[inline]
fn scatter_stx8_i16(cf: &mut [i16], sums: &[i16; 48], scan_out: &[u8; 64], mapping: &[u8; 48]) {
    for (&map, &sum) in mapping.iter().zip(sums.iter()) {
        cf[scan_out[map as usize] as usize] = sum;
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn zero_stx4_scan_tail_i16_avx512(sums: __m512i, scan_out: &[u8; 16]) -> __m512i {
    let idx = _mm512_zextsi256_si512(unsafe {
        _mm256_cvtepu8_epi16(_mm_loadu_si128(scan_out.as_ptr().cast()))
    });
    let ge4 = _mm512_cmpgt_epi16_mask(idx, _mm512_set1_epi16(3));
    let lt8 = _mm512_cmpgt_epi16_mask(_mm512_set1_epi16(8), idx);
    let mask = ge4 & lt8;
    _mm512_mask_mov_epi16(sums, mask, _mm512_setzero_si512())
}

#[inline]
#[target_feature(enable = "avx512f")]
fn zero_stx8_i16_avx512(cf: &mut [i16]) {
    let zero = _mm512_setzero_si512();
    let dst = cf.as_mut_ptr().cast::<__m512i>();
    unsafe { _mm512_storeu_si512(dst, zero) };
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn stxfm4_8bpc_avx512(cf: &mut [i16], kernel: &[i8], eob: usize, scan_out: &[u8; 16]) {
    debug_assert!(eob < 8);
    debug_assert!(kernel.len() >= 8 * 16);

    let sums_v = zero_stx4_scan_tail_i16_avx512(stx16_sums_vnni(kernel, cf, eob, 16, 0), scan_out);
    let mut sums = [0i16; 16];
    store_low_i16x16(&mut sums, sums_v);

    scatter_stx4_i16(cf, &sums, scan_out);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) fn stxfm8_8bpc_avx512(
    cf: &mut [i16],
    kernel: &[i8],
    eob: usize,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    debug_assert!(eob < 32);
    debug_assert!(kernel.len() >= 32 * 48);

    let s0 = stx16_sums_vnni(kernel, cf, eob, 48, 0);
    let s1 = stx16_sums_vnni(kernel, cf, eob, 48, 16);
    let s2 = stx16_sums_vnni(kernel, cf, eob, 48, 32);

    let mut sums = [0i16; 48];
    store_low_i16x16(&mut sums[..16], s0);
    store_low_i16x16(&mut sums[16..32], s1);
    store_low_i16x16(&mut sums[32..48], s2);

    zero_stx8_i16_avx512(cf);
    scatter_stx8_i16(cf, &sums, scan_out, mapping);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load_i8x16_i32(ptr: *const i8) -> __m512i {
    unsafe { _mm512_cvtepi8_epi32(_mm_loadu_si128(ptr.cast())) }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac_hbd_16(acc: __m512i, coeff: i32, kernel: *const i8) -> __m512i {
    let k = load_i8x16_i32(kernel);
    let c = _mm512_set1_epi32(coeff);
    _mm512_add_epi32(acc, _mm512_mullo_epi32(k, c))
}

#[inline]
#[target_feature(enable = "avx512f")]
fn round_clip_hbd_16(acc: __m512i, min_v: __m512i, max_v: __m512i) -> __m512i {
    let one = _mm512_set1_epi32(1);
    let mask = _mm512_cmpgt_epi32_mask(acc, _mm512_set1_epi32(-1));
    let v = _mm512_srai_epi32::<7>(_mm512_mask_add_epi32(acc, mask, acc, one));
    _mm512_min_epi32(_mm512_max_epi32(v, min_v), max_v)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn stx16_sums_hbd(
    kernel: &[i8],
    cf: &[i32],
    eob: usize,
    bitdepth_max: i32,
    row_stride: usize,
    x: usize,
) -> __m512i {
    let min_v = _mm512_set1_epi32(-128 * (1 + bitdepth_max));
    let max_v = _mm512_set1_epi32(128 * (1 + bitdepth_max) - 1);
    let mut acc = _mm512_set1_epi32(63);

    let mut y = 0usize;
    while y <= eob {
        let c = unsafe { *cf.get_unchecked(y) };
        acc = mac_hbd_16(acc, c, unsafe { kernel.as_ptr().add(y * row_stride + x) });
        y += 1;
    }

    round_clip_hbd_16(acc, min_v, max_v)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load_u8x16_i32(src: &[u8; 16]) -> __m512i {
    unsafe { _mm512_cvtepu8_epi32(_mm_loadu_si128(src.as_ptr().cast())) }
}

#[inline]
#[target_feature(enable = "avx512f")]
fn scatter_i32x16(dst: &mut [i32], idx: __m512i, v: __m512i) {
    unsafe { _mm512_i32scatter_epi32::<4>(dst.as_mut_ptr(), idx, v) };
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn scatter_stx4_i32(cf: &mut [i32], sums: __m512i, scan_out: &[u8; 16]) {
    let idx = load_u8x16_i32(scan_out);
    scatter_i32x16(cf, idx, sums);
}

#[inline]
fn stx8_indices(scan_out: &[u8; 64], mapping: &[u8; 48], base: usize) -> [i32; 16] {
    core::array::from_fn(|i| scan_out[mapping[base + i] as usize] as i32)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn scatter_stx8_i32(
    cf: &mut [i32],
    s0: __m512i,
    s1: __m512i,
    s2: __m512i,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    let i0 = stx8_indices(scan_out, mapping, 0);
    let i1 = stx8_indices(scan_out, mapping, 16);
    let i2 = stx8_indices(scan_out, mapping, 32);
    let i0 = unsafe { _mm512_loadu_si512(i0.as_ptr().cast()) };
    let i1 = unsafe { _mm512_loadu_si512(i1.as_ptr().cast()) };
    let i2 = unsafe { _mm512_loadu_si512(i2.as_ptr().cast()) };
    scatter_i32x16(cf, i0, s0);
    scatter_i32x16(cf, i1, s1);
    scatter_i32x16(cf, i2, s2);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn zero_stx4_scan_tail_i32_avx512(sums: __m512i, scan_out: &[u8; 16]) -> __m512i {
    let idx = load_u8x16_i32(scan_out);
    let ge4 = _mm512_cmpgt_epi32_mask(idx, _mm512_set1_epi32(3));
    let lt8 = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(8), idx);
    let mask = ge4 & lt8;
    _mm512_mask_mov_epi32(sums, mask, _mm512_setzero_si512())
}

#[inline]
#[target_feature(enable = "avx512f")]
fn zero_stx8_i32_avx512(cf: &mut [i32]) {
    let zero = _mm512_setzero_si512();
    let dst = cf.as_mut_ptr().cast::<__m512i>();
    unsafe {
        _mm512_storeu_si512(dst, zero);
        _mm512_storeu_si512(dst.add(1), zero);
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) fn stxfm4_hbd_avx512(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 16],
) {
    debug_assert!(eob < 8);
    debug_assert!(kernel.len() >= 8 * 16);

    let sums = zero_stx4_scan_tail_i32_avx512(
        stx16_sums_hbd(kernel, cf, eob, bitdepth_max, 16, 0),
        scan_out,
    );
    scatter_stx4_i32(cf, sums, scan_out);
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) fn stxfm8_hbd_avx512(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    debug_assert!(eob < 32);
    debug_assert!(kernel.len() >= 32 * 48);

    let s0 = stx16_sums_hbd(kernel, cf, eob, bitdepth_max, 48, 0);
    let s1 = stx16_sums_hbd(kernel, cf, eob, bitdepth_max, 48, 16);
    let s2 = stx16_sums_hbd(kernel, cf, eob, bitdepth_max, 48, 32);

    zero_stx8_i32_avx512(cf);
    scatter_stx8_i32(cf, s0, s1, s2, scan_out, mapping);
}
