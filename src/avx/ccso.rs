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
fn ccso_tail_8bpc<const SS_HOR: usize, const BO_ONLY: bool>(
    dst_row: &mut [u8],
    tmp: &[u8],
    row: usize,
    x0: usize,
    x1: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
) {
    for (x, out) in dst_row[x0..x1].iter_mut().enumerate() {
        let x = x + x0;
        let ti = row + (x << SS_HOR);
        let c = tmp[ti] as i32;
        let band = (c as u32 >> shift) as u8;
        if BO_ONLY {
            *out = band;
        } else {
            let cls0 = crate::ccso::ccso_score(
                tmp[(ti as isize + luma_offset) as usize] as i32 - c,
                quant_step,
                edge_clf,
            );
            let cls1 = crate::ccso::ccso_score(
                tmp[(ti as isize - luma_offset) as usize] as i32 - c,
                quant_step,
                edge_clf,
            );
            *out = ((cls0 << 5) | (cls1 << 3)) as u8 | band;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ccso_classify_epi16(diff: __m128i, q: __m128i, nq: __m128i, edge_clf: u32) -> __m128i {
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi16(1);
    let two = _mm_set1_epi16(2);
    let gt = if edge_clf == 0 {
        _mm_cmpgt_epi16(diff, q)
    } else {
        zero
    };
    let lt = _mm_cmpgt_epi16(nq, diff);
    let cls = _mm_blendv_epi8(one, two, gt);
    _mm_blendv_epi8(cls, zero, lt)
}

#[inline]
#[target_feature(enable = "avx2")]
fn ccso_make_idx_8x16(
    c: __m128i,
    p0: __m128i,
    p1: __m128i,
    shiftv: __m128i,
    q: __m128i,
    nq: __m128i,
    edge_clf: u32,
) -> __m128i {
    let band = _mm_srl_epi16(c, shiftv);
    let cls0 = ccso_classify_epi16(_mm_sub_epi16(p0, c), q, nq, edge_clf);
    let cls1 = ccso_classify_epi16(_mm_sub_epi16(p1, c), q, nq, edge_clf);
    _mm_or_si128(
        band,
        _mm_or_si128(_mm_slli_epi16(cls0, 5), _mm_slli_epi16(cls1, 3)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x16_step2(ptr: *const u8) -> __m128i {
    let mask = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1,
        -1, -1, -1, -1, -1, -1, -1,
    );
    let v = unsafe { _mm256_loadu_si256(ptr as *const __m256i) };
    let v = _mm256_shuffle_epi8(v, mask);
    _mm_unpacklo_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x8_step2(ptr: *const u8) -> __m128i {
    let mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let v = unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    _mm_shuffle_epi8(v, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x16_subsampled<const SS_HOR: usize>(tmp: &[u8], base: usize) -> __m128i {
    if SS_HOR == 0 {
        unsafe { _mm_loadu_si128(tmp.as_ptr().add(base) as *const __m128i) }
    } else {
        load_u8x16_step2(unsafe { tmp.as_ptr().add(base) })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u8x8_subsampled<const SS_HOR: usize>(tmp: &[u8], base: usize) -> __m128i {
    if SS_HOR == 0 {
        unsafe { _mm_loadl_epi64(tmp.as_ptr().add(base) as *const __m128i) }
    } else {
        load_u8x8_step2(unsafe { tmp.as_ptr().add(base) })
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_idx8(dst: &mut [u8; 8], out16: __m128i) {
    let out = _mm_packus_epi16(out16, out16);
    unsafe { _mm_storel_epi64(dst.as_mut_ptr() as *mut __m128i, out) };
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
fn ccso_prep_lut_8bpc_avx2_impl<const SS_HOR: usize, const SS_VER: usize, const BO_ONLY: bool>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    tmp_stride: usize,
    o: usize,
    w: usize,
    h: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
) {
    let zero = _mm_setzero_si128();
    let q = _mm_set1_epi16(quant_step as i16);
    let nq = _mm_set1_epi16((-quant_step) as i16);
    let shiftv = _mm_cvtsi32_si128(shift as i32);

    for (y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate() {
        let row = o + (y << SS_VER) * tmp_stride;
        let dst_row = &mut dst_row[..w];
        let mut x;
        if BO_ONLY {
            let n16 = {
                let (dst16, _) = dst_row.as_chunks_mut::<16>();
                for (chunk_idx, out16) in dst16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let base = row + (x << SS_HOR);
                    let c = load_u8x16_subsampled::<SS_HOR>(tmp, base);
                    let lo = _mm_srl_epi16(_mm_unpacklo_epi8(c, zero), shiftv);
                    let hi = _mm_srl_epi16(_mm_unpackhi_epi8(c, zero), shiftv);
                    let out = _mm_packus_epi16(lo, hi);
                    unsafe { _mm_storeu_si128(out16.as_mut_ptr() as *mut __m128i, out) };
                }
                dst16.len()
            };
            x = n16 * 16;
            if x + 8 <= w {
                let base = row + (x << SS_HOR);
                let c = load_u8x8_subsampled::<SS_HOR>(tmp, base);
                let out16 = _mm_srl_epi16(_mm_unpacklo_epi8(c, zero), shiftv);
                let out8 = &mut dst_row[x..x + 8].as_chunks_mut::<8>().0[0];
                store_idx8(out8, out16);
                x += 8;
            }
        } else {
            let n16 = {
                let (dst16, _) = dst_row.as_chunks_mut::<16>();
                for (chunk_idx, out16) in dst16.iter_mut().enumerate() {
                    let x = chunk_idx * 16;
                    let base = row + (x << SS_HOR);
                    let c8 = load_u8x16_subsampled::<SS_HOR>(tmp, base);
                    let p08 = load_u8x16_subsampled::<SS_HOR>(
                        tmp,
                        (base as isize + luma_offset) as usize,
                    );
                    let p18 = load_u8x16_subsampled::<SS_HOR>(
                        tmp,
                        (base as isize - luma_offset) as usize,
                    );
                    let clo = _mm_unpacklo_epi8(c8, zero);
                    let chi = _mm_unpackhi_epi8(c8, zero);
                    let p0lo = _mm_unpacklo_epi8(p08, zero);
                    let p0hi = _mm_unpackhi_epi8(p08, zero);
                    let p1lo = _mm_unpacklo_epi8(p18, zero);
                    let p1hi = _mm_unpackhi_epi8(p18, zero);
                    let lo = ccso_make_idx_8x16(clo, p0lo, p1lo, shiftv, q, nq, edge_clf);
                    let hi = ccso_make_idx_8x16(chi, p0hi, p1hi, shiftv, q, nq, edge_clf);
                    let out = _mm_packus_epi16(lo, hi);
                    unsafe { _mm_storeu_si128(out16.as_mut_ptr() as *mut __m128i, out) };
                }
                dst16.len()
            };
            x = n16 * 16;
            if x + 8 <= w {
                let base = row + (x << SS_HOR);
                let c8 = load_u8x8_subsampled::<SS_HOR>(tmp, base);
                let p08 =
                    load_u8x8_subsampled::<SS_HOR>(tmp, (base as isize + luma_offset) as usize);
                let p18 =
                    load_u8x8_subsampled::<SS_HOR>(tmp, (base as isize - luma_offset) as usize);
                let out16 = ccso_make_idx_8x16(
                    _mm_unpacklo_epi8(c8, zero),
                    _mm_unpacklo_epi8(p08, zero),
                    _mm_unpacklo_epi8(p18, zero),
                    shiftv,
                    q,
                    nq,
                    edge_clf,
                );
                let out8 = &mut dst_row[x..x + 8].as_chunks_mut::<8>().0[0];
                store_idx8(out8, out16);
                x += 8;
            }
        }
        ccso_tail_8bpc::<SS_HOR, BO_ONLY>(
            dst_row,
            tmp,
            row,
            x,
            w,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn ccso_prep_lut_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    tmp_stride: usize,
    o: usize,
    w: usize,
    h: usize,
    ss_hor: usize,
    ss_ver: usize,
    shift: u32,
    luma_offset: isize,
    quant_step: i32,
    edge_clf: u32,
    bo_only: bool,
) {
    match (ss_hor, ss_ver, bo_only) {
        (0, 0, true) => ccso_prep_lut_8bpc_avx2_impl::<0, 0, true>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (0, 0, false) => ccso_prep_lut_8bpc_avx2_impl::<0, 0, false>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 0, true) => ccso_prep_lut_8bpc_avx2_impl::<1, 0, true>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 0, false) => ccso_prep_lut_8bpc_avx2_impl::<1, 0, false>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 1, true) => ccso_prep_lut_8bpc_avx2_impl::<1, 1, true>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        (1, 1, false) => ccso_prep_lut_8bpc_avx2_impl::<1, 1, false>(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
        ),
        _ => crate::ccso::ccso_prep_lut_8bpc_scalar(
            dst,
            dst_stride,
            tmp,
            tmp_stride,
            o,
            w,
            h,
            ss_hor,
            ss_ver,
            shift,
            luma_offset,
            quant_step,
            edge_clf,
            bo_only,
        ),
    }
}

#[inline(always)]
fn fill_offsets_4_i16(out: &mut [i16; 8], idx: &[u8; 4], offset_map: &[i8; 256]) {
    for (out, &idx) in out[..4].iter_mut().zip(idx.iter()) {
        *out = offset_map[idx as usize] as i16;
    }
}

#[repr(C, align(16))]
pub(crate) struct AlignedSseS16(pub(crate) [i16; 8]);

#[inline]
#[target_feature(enable = "avx2")]
fn ccso_add_4x4_8bpc(
    dst_rows: &mut [u8],
    dst_stride: usize,
    idx_rows: &[u8],
    idx_stride: usize,
    block_h: usize,
    offset_map: &[i8; 256],
    xx: usize,
) {
    let zero = _mm_setzero_si128();
    let mut off_tmp = AlignedSseS16([0; 8]);
    for (yy, idx_row) in idx_rows.chunks(idx_stride).take(block_h).enumerate() {
        let dst_row = &mut dst_rows[yy * dst_stride..];
        let idx4 = &idx_row[xx..xx + 4].as_chunks::<4>().0[0];
        fill_offsets_4_i16(&mut off_tmp.0, idx4, offset_map);
        let off = unsafe { _mm_load_si128(off_tmp.0.as_ptr().cast()) };
        let dst4 = &mut dst_row[xx..xx + 4].as_chunks_mut::<4>().0[0];
        let cur = unsafe { _mm_castps_si128(_mm_load_ss(dst4.as_ptr().cast())) };
        let cur = _mm_unpacklo_epi8(cur, zero);
        let out = _mm_packus_epi16(_mm_add_epi16(cur, off), zero);
        unsafe {
            _mm_store_ss(dst4.as_mut_ptr().cast(), _mm_castsi128_ps(out));
        }
    }
}

/// Vector form of `ccso_build_offset_map`: the 64 bytes of packed 3-bit
/// offset indices as four 16-entry shuffle tables, and the 8 offsets.
struct OffsetTables {
    idxs: [__m256i; 4],
    lut: __m256i,
}

impl OffsetTables {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn new(offset_idxs: &[u8; 64], offset_lut: &[i8; 8]) -> Self {
        let rows = offset_idxs.as_chunks::<16>().0;
        unsafe {
            Self {
                idxs: std::array::from_fn(|t| {
                    _mm256_broadcastsi128_si256(_mm_loadu_si128(rows[t].as_ptr().cast()))
                }),
                lut: _mm256_broadcastsi128_si256(_mm_loadl_epi64(offset_lut.as_ptr().cast())),
            }
        }
    }

    /// The offset of each LUT index byte: entry `i & 127` of the offset map,
    /// i.e. `lut[(idxs[i >> 1] >> (4 * (i & 1))) & 7]`.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn offsets(&self, idx: __m256i) -> __m256i {
        // Byte holding the index: (i & 127) >> 1, in 0..64.
        let byte = _mm256_and_si256(_mm256_srli_epi16::<1>(idx), _mm256_set1_epi8(0x3f));
        // A shuffle index with bit 7 set yields 0, so each table only
        // answers for its own 16 bytes: (byte ^ 16t) + 0x70 has bit 7 clear
        // exactly when byte is in table t.
        let bias = _mm256_set1_epi8(0x70);
        let mut packed = _mm256_setzero_si256();
        for (t, table) in self.idxs.iter().enumerate() {
            let sel = _mm256_xor_si256(byte, _mm256_set1_epi8((t as i8) << 4));
            let hit = _mm256_shuffle_epi8(*table, _mm256_adds_epu8(sel, bias));
            packed = _mm256_or_si256(packed, hit);
        }
        // Odd indices take the high nibble; blendv looks at bit 7 only.
        let high = _mm256_srli_epi16::<4>(packed);
        let odd = _mm256_slli_epi16::<7>(idx);
        let nibble = _mm256_blendv_epi8(packed, high, odd);
        _mm256_shuffle_epi8(self.lut, _mm256_and_si256(nibble, _mm256_set1_epi8(7)))
    }
}

/// `clamp(cur + off, 0, 255)` per byte for unsigned `cur` and signed `off`.
#[inline]
#[target_feature(enable = "avx2")]
fn add_offsets_u8(cur: __m256i, off: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let up = _mm256_max_epi8(off, zero);
    let down = _mm256_max_epi8(_mm256_sub_epi8(zero, off), zero);
    _mm256_subs_epu8(_mm256_adds_epu8(cur, up), down)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn ccso_add_8bpc_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
) {
    // Header tables always have these sizes; anything else takes the
    // portable kernel.
    let (Ok(idxs), Ok(lut)) = (
        <&[u8; 64]>::try_from(offset_idxs),
        <&[i8; 8]>::try_from(offset_lut),
    ) else {
        return crate::ccso::ccso_add_8bpc_scalar(
            dst,
            dst_stride,
            idx_buf,
            idx_stride,
            offset_idxs,
            offset_lut,
            w,
            h,
            ll_mask,
        );
    };
    let tables = OffsetTables::new(idxs, lut);
    // Only the 4x4 fallback for partly masked spans needs the byte map.
    let mut offset_map: Option<[i8; 256]> = None;
    let n_blocks = (h + 3) >> 2;
    for (by, mask) in (0..h).step_by(4).zip(ll_mask[..n_blocks].iter()) {
        let block_h = (h - by).min(4);
        // `dst` may already start at an x-offset inside the picture row, so
        // chunking it by `dst_stride` would mix row tails with the next row.
        let dst_rows = &mut dst[by * dst_stride..];
        let idx_row_start = by * idx_stride;
        let idx_rows = &idx_buf[idx_row_start..idx_row_start + block_h * idx_stride];
        let row_mask = mask[0];
        let mut xx = 0usize;
        while xx + 32 <= w {
            let bx = xx >> 2;
            if ((row_mask >> bx) & 0xff) != 0 {
                break;
            }

            for (yy, idx_row) in idx_rows.chunks(idx_stride).take(block_h).enumerate() {
                let dst_row = &mut dst_rows[yy * dst_stride..];
                let idx32 = &idx_row[xx..xx + 32].as_chunks::<32>().0[0];
                let off = tables.offsets(unsafe { _mm256_loadu_si256(idx32.as_ptr().cast()) });
                let dst32 = &mut dst_row[xx..xx + 32].as_chunks_mut::<32>().0[0];
                let cur = unsafe { _mm256_loadu_si256(dst32.as_ptr().cast()) };
                let out = add_offsets_u8(cur, off);
                unsafe { _mm256_storeu_si256(dst32.as_mut_ptr().cast(), out) };
            }
            xx += 32;
        }
        while xx + 16 <= w {
            let bx = xx >> 2;
            if ((row_mask >> bx) & 0x0f) == 0 {
                for (yy, idx_row) in idx_rows.chunks(idx_stride).take(block_h).enumerate() {
                    let dst_row = &mut dst_rows[yy * dst_stride..];
                    let idx16 = &idx_row[xx..xx + 16].as_chunks::<16>().0[0];
                    let idx = unsafe { _mm_loadu_si128(idx16.as_ptr().cast()) };
                    let off = tables.offsets(_mm256_zextsi128_si256(idx));
                    let dst16 = &mut dst_row[xx..xx + 16].as_chunks_mut::<16>().0[0];
                    let cur = unsafe { _mm_loadu_si128(dst16.as_ptr() as *const __m128i) };
                    let out =
                        _mm256_castsi256_si128(add_offsets_u8(_mm256_zextsi128_si256(cur), off));
                    unsafe { _mm_storeu_si128(dst16.as_mut_ptr() as *mut __m128i, out) };
                }
                xx += 16;
            } else {
                for _ in 0..4 {
                    let bx = xx >> 2;
                    if row_mask & (1 << bx) == 0 {
                        ccso_add_4x4_8bpc(
                            dst_rows,
                            dst_stride,
                            idx_rows,
                            idx_stride,
                            block_h,
                            offset_map.get_or_insert_with(|| {
                                crate::ccso::ccso_build_offset_map(offset_idxs, offset_lut)
                            }),
                            xx,
                        );
                    }
                    xx += 4;
                }
            }
        }
        while xx < w {
            let bx = xx >> 2;
            if row_mask & (1 << bx) == 0 {
                ccso_add_4x4_8bpc(
                    dst_rows,
                    dst_stride,
                    idx_rows,
                    idx_stride,
                    block_h,
                    offset_map.get_or_insert_with(|| {
                        crate::ccso::ccso_build_offset_map(offset_idxs, offset_lut)
                    }),
                    xx,
                );
            }
            xx += 4;
        }
    }
}

#[cfg(test)]
mod ccso_add_tests {
    struct Rng(u64);

    impl Rng {
        fn r(&mut self, n: u32) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32 % n
        }
    }

    /// The AVX2 add against the scalar one: every LUT index byte, all offset
    /// scales, pixels near both clamps, masked and unmasked spans.
    #[test]
    fn add_8bpc_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng(0xcc50_add0_1357);
        for iter in 0..3000 {
            let w = [64usize, 60, 48, 36, 32, 20, 16, 8, 4][rng.r(9) as usize];
            let h = [8usize, 4][rng.r(2) as usize];
            let (ds, is) = (w + rng.r(9) as usize, 64);
            let filter_off: [u8; 64] = core::array::from_fn(|_| rng.r(256) as u8);
            let lut = &crate::tables::CCSO_OFFSET[rng.r(4) as usize];
            let idx: Vec<u8> = (0..is * h).map(|_| rng.r(256) as u8).collect();
            let edge = iter % 3 == 0;
            let src: Vec<u8> = (0..ds * h)
                .map(|_| {
                    if edge {
                        [0, 3, 250, 255][rng.r(4) as usize]
                    } else {
                        rng.r(256) as u8
                    }
                })
                .collect();
            let masked = rng.r(3) == 0;
            let ll: Vec<[u16; 4]> = (0..2)
                .map(|_| [if masked { rng.r(1 << 16) as u16 } else { 0 }, 0, 0, 0])
                .collect();
            let (mut d0, mut d1) = (src.clone(), src.clone());
            crate::ccso::ccso_add_8bpc_scalar(&mut d0, ds, &idx, is, &filter_off, lut, w, h, &ll);
            unsafe {
                super::ccso_add_8bpc_avx2(&mut d1, ds, &idx, is, &filter_off, lut, w, h, &ll)
            };
            assert_eq!(d0, d1, "iter {iter} {w}x{h} masked {masked}");
        }
    }
}
