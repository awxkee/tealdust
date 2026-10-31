/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Superblock-row deblock (high bitdepth) and loop-filter threshold setup for SSE4.1 (128-bit ports of the AVX2 kernels).

use super::deblocking::deblock_apply_hbd_sse41;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn deblock_mask_class_bits(mask: u16, higher: u16, both_lossless: u16) -> u32 {
    (mask & !higher & !both_lossless) as u32
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
fn deblock_sb64_hbd_sse41_mask<
    const MAX_WIDTH_NEG: i32,
    const MAX_WIDTH_POS: i32,
    const HORIZONTAL: bool,
>(
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    mut vm: u32,
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    bitdepth_max: i32,
) {
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let bit = 1u32 << qi;
        let q = q_thr[qi] as u32;
        if q != 0 {
            let pos_ll = (ll_mask[1] as u32 & bit) != 0;
            let neg_ll = (ll_mask[0] as u32 & bit) != 0;
            if !(pos_ll && neg_ll) {
                let off = if HORIZONTAL {
                    (dst_off + qi * 4 * stride) as isize
                } else {
                    (dst_off + qi * 4) as isize
                };
                let stridea = if HORIZONTAL { stride as isize } else { 1 };
                let strideb = if HORIZONTAL { 1 } else { stride as isize };
                crate::deblock_dispatch::deblock_hbd_edge_with(
                    dst,
                    off,
                    q,
                    side_thr[qi] as u32,
                    stridea,
                    strideb,
                    MAX_WIDTH_POS,
                    MAX_WIDTH_NEG,
                    pos_ll,
                    neg_ll,
                    bitdepth_max,
                    deblock_apply_hbd_sse41,
                );
            }
        }
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn deblock_h_sb64y_hbd_sse41(
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
    bitdepth_max: i32,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m3 = deblock_mask_class_bits(vmask[3], 0, both_lossless);
    let m2 = deblock_mask_class_bits(vmask[2], vmask[3], both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2] | vmask[3], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2] | vmask[3], both_lossless);

    if m0 != 0 {
        deblock_sb64_hbd_sse41_mask::<1, 1, true>(
            dst,
            dst_off,
            stride,
            m0,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m1 != 0 {
        deblock_sb64_hbd_sse41_mask::<3, 3, true>(
            dst,
            dst_off,
            stride,
            m1,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m2 != 0 {
        deblock_sb64_hbd_sse41_mask::<6, 6, true>(
            dst,
            dst_off,
            stride,
            m2,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m3 != 0 {
        if edge {
            deblock_sb64_hbd_sse41_mask::<6, 8, true>(
                dst,
                dst_off,
                stride,
                m3,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        } else {
            deblock_sb64_hbd_sse41_mask::<8, 8, true>(
                dst,
                dst_off,
                stride,
                m3,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn deblock_v_sb64y_hbd_sse41(
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
    bitdepth_max: i32,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m3 = deblock_mask_class_bits(vmask[3], 0, both_lossless);
    let m2 = deblock_mask_class_bits(vmask[2], vmask[3], both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2] | vmask[3], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2] | vmask[3], both_lossless);

    if m0 != 0 {
        deblock_sb64_hbd_sse41_mask::<1, 1, false>(
            dst,
            dst_off,
            stride,
            m0,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m1 != 0 {
        deblock_sb64_hbd_sse41_mask::<3, 3, false>(
            dst,
            dst_off,
            stride,
            m1,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m2 != 0 {
        deblock_sb64_hbd_sse41_mask::<6, 6, false>(
            dst,
            dst_off,
            stride,
            m2,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m3 != 0 {
        if edge {
            deblock_sb64_hbd_sse41_mask::<6, 8, false>(
                dst,
                dst_off,
                stride,
                m3,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        } else {
            deblock_sb64_hbd_sse41_mask::<8, 8, false>(
                dst,
                dst_off,
                stride,
                m3,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn deblock_h_sb64uv_hbd_sse41(
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
    bitdepth_max: i32,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m2 = deblock_mask_class_bits(vmask[2], 0, both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2], both_lossless);

    if m0 != 0 {
        deblock_sb64_hbd_sse41_mask::<1, 1, true>(
            dst,
            dst_off,
            stride,
            m0,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m1 != 0 {
        if edge {
            deblock_sb64_hbd_sse41_mask::<2, 3, true>(
                dst,
                dst_off,
                stride,
                m1,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        } else {
            deblock_sb64_hbd_sse41_mask::<3, 3, true>(
                dst,
                dst_off,
                stride,
                m1,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        }
    }
    if m2 != 0 {
        if edge {
            deblock_sb64_hbd_sse41_mask::<2, 4, true>(
                dst,
                dst_off,
                stride,
                m2,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        } else {
            deblock_sb64_hbd_sse41_mask::<4, 4, true>(
                dst,
                dst_off,
                stride,
                m2,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn deblock_v_sb64uv_hbd_sse41(
    dst: &mut [u16],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
    bitdepth_max: i32,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m2 = deblock_mask_class_bits(vmask[2], 0, both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2], both_lossless);

    if m0 != 0 {
        deblock_sb64_hbd_sse41_mask::<1, 1, false>(
            dst,
            dst_off,
            stride,
            m0,
            ll_mask,
            q_thr,
            side_thr,
            bitdepth_max,
        );
    }
    if m1 != 0 {
        if edge {
            deblock_sb64_hbd_sse41_mask::<2, 3, false>(
                dst,
                dst_off,
                stride,
                m1,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        } else {
            deblock_sb64_hbd_sse41_mask::<3, 3, false>(
                dst,
                dst_off,
                stride,
                m1,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        }
    }
    if m2 != 0 {
        if edge {
            deblock_sb64_hbd_sse41_mask::<2, 4, false>(
                dst,
                dst_off,
                stride,
                m2,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        } else {
            deblock_sb64_hbd_sse41_mask::<4, 4, false>(
                dst,
                dst_off,
                stride,
                m2,
                ll_mask,
                q_thr,
                side_thr,
                bitdepth_max,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_lut_u8x16_sse41(lut: &[u32; 16]) -> __m128i {
    let mut tbl = [0u8; 16];
    for i in 0..16 {
        tbl[i] = lut[i] as u8;
    }
    unsafe { _mm_loadu_si128(tbl.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_load_seg_u8x16_sse41(seg: &[u8], off: usize, w: usize) -> __m128i {
    let mut tmp = [0u8; 16];
    tmp[..w].copy_from_slice(&seg[off..off + w]);
    unsafe { _mm_loadu_si128(tmp.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_mask_bits_u8x16_sse41(bits: u16) -> __m128i {
    let bit_masks = _mm_setr_epi8(
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        0x80u8 as i8,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        0x80u8 as i8,
    );
    let lo = _mm_set1_epi8(bits as u8 as i8);
    let hi = _mm_set1_epi8((bits >> 8) as u8 as i8);
    let bytemask = _mm_unpacklo_epi64(lo, hi);
    let hit = _mm_and_si128(bytemask, bit_masks);
    _mm_andnot_si128(
        _mm_cmpeq_epi8(hit, _mm_setzero_si128()),
        _mm_cmpeq_epi8(hit, hit),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_shr3_u8x16_sse41(v: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    let lo = _mm_unpacklo_epi8(v, z);
    let hi = _mm_unpackhi_epi8(v, z);
    _mm_packus_epi16(_mm_srli_epi16::<3>(lo), _mm_srli_epi16::<3>(hi))
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_apply_subpu_u8x16_sse41(v: __m128i, bits: u16) -> __m128i {
    let m = setup_mask_bits_u8x16_sse41(bits);
    _mm_blendv_epi8(v, setup_shr3_u8x16_sse41(v), m)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_subpu_pair_u8x16_sse41(q: __m128i, s: __m128i, bits: u16) -> (__m128i, __m128i) {
    let s = setup_apply_subpu_u8x16_sse41(s, bits);
    let q = setup_apply_subpu_u8x16_sse41(q, bits);
    (
        _mm_andnot_si128(_mm_cmpeq_epi8(s, _mm_setzero_si128()), q),
        s,
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_edge_u8x16_sse41(cur: __m128i, prev: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    let all = _mm_cmpeq_epi8(z, z);
    let cur_z = _mm_cmpeq_epi8(cur, z);
    let prev_z = _mm_cmpeq_epi8(prev, z);
    let both = _mm_andnot_si128(_mm_or_si128(cur_z, prev_z), all);
    let avg = _mm_avg_epu8(cur, prev);
    let ored = _mm_or_si128(cur, prev);
    _mm_blendv_epi8(ored, avg, both)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_prev_from_left_u8x16_sse41(cur: __m128i, left: u8) -> __m128i {
    _mm_insert_epi8::<0>(_mm_slli_si128::<1>(cur), left as i32)
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_store_u8x16_sse41(dst: &mut [u8; 256], off: usize, v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().add(off).cast(), v) };
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn setup_store_tmp_u8x16_sse41(v: __m128i) -> [u8; 16] {
    let mut tmp = [0u8; 16];
    unsafe { _mm_storeu_si128(tmp.as_mut_ptr().cast(), v) };
    tmp
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn setup_thr_rows_simple_8bpc_sse41(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));
    let h = h4 as usize;
    let mask_idx = (sb64x >> ss_hor) as usize;
    assert!(mask_idx < 4);
    assert!(starty4 + h <= 64);
    let mask_shift: u32 = if (sb64x & ss_hor) != 0 { 8 } else { 0 };
    let qv = _mm_set1_epi8(thr_lut[0][0] as u8 as i8);
    let sv = _mm_set1_epi8(thr_lut[1][0] as u8 as i8);
    for y in 0..h {
        let bits = (mask[starty4 + y][4][mask_idx] >> mask_shift) as u16;
        let (qo, so) = setup_subpu_pair_u8x16_sse41(qv, sv, bits);
        setup_store_u8x16_sse41(q_thr_dst, y * 16, qo);
        setup_store_u8x16_sse41(side_thr_dst, y * 16, so);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn setup_thr_cols_simple_8bpc_sse41(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));
    let w = w4 as usize;
    let mask_idx = (y64 >> ss_ver) as usize;
    assert!(mask_idx < 4);
    assert!(bx4_base + w <= 64);
    let mask_shift: u32 = if (y64 & ss_ver) != 0 { 8 } else { 0 };
    let qv = _mm_set1_epi8(thr_lut[0][0] as u8 as i8);
    let sv = _mm_set1_epi8(thr_lut[1][0] as u8 as i8);
    for x in 0..w {
        let bits = (mask[bx4_base + x][4][mask_idx] >> mask_shift) as u16;
        let (qo, so) = setup_subpu_pair_u8x16_sse41(qv, sv, bits);
        setup_store_u8x16_sse41(q_thr_dst, x * 16, qo);
        setup_store_u8x16_sse41(side_thr_dst, x * 16, so);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn setup_thr_rows_dq_8bpc_sse41(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    above_thr_lut: Option<&[[u32; 16]; 2]>,
    above_seg: Option<(&[u8], isize)>,
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));
    let w = w4 as usize;
    let h = h4 as usize;
    if w == 0 || h == 0 {
        return;
    }
    let mask_idx = (sb64x >> ss_hor) as usize;
    assert!(mask_idx < 4);
    assert!(starty4 + h <= 64);
    let mask_shift: u32 = if (sb64x & ss_hor) != 0 { 8 } else { 0 };
    let qv = _mm_set1_epi8(thr_lut[0][0] as u8 as i8);
    let sv = _mm_set1_epi8(thr_lut[1][0] as u8 as i8);
    let (above_q, above_s) = if let Some(alut) = above_thr_lut {
        if let Some((aseg, aoff)) = above_seg {
            let off = usize::try_from(aoff).expect("negative above segment offset");
            assert!(off + w <= aseg.len());
            let segv = setup_load_seg_u8x16_sse41(aseg, off, w);
            (
                _mm_shuffle_epi8(setup_lut_u8x16_sse41(&alut[0]), segv),
                _mm_shuffle_epi8(setup_lut_u8x16_sse41(&alut[1]), segv),
            )
        } else {
            (
                _mm_set1_epi8(alut[0][0] as u8 as i8),
                _mm_set1_epi8(alut[1][0] as u8 as i8),
            )
        }
    } else {
        (_mm_setzero_si128(), _mm_setzero_si128())
    };
    let bits0 = (mask[starty4][4][mask_idx] >> mask_shift) as u16;
    let (qo, so) = setup_subpu_pair_u8x16_sse41(
        setup_edge_u8x16_sse41(qv, above_q),
        setup_edge_u8x16_sse41(sv, above_s),
        bits0,
    );
    setup_store_u8x16_sse41(q_thr_dst, 0, qo);
    setup_store_u8x16_sse41(side_thr_dst, 0, so);
    for y in 1..h {
        let bits = (mask[starty4 + y][4][mask_idx] >> mask_shift) as u16;
        let (qo, so) = setup_subpu_pair_u8x16_sse41(qv, sv, bits);
        setup_store_u8x16_sse41(q_thr_dst, y * 16, qo);
        setup_store_u8x16_sse41(side_thr_dst, y * 16, so);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn setup_thr_cols_dq_8bpc_sse41(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));
    let w = w4 as usize;
    let h = h4 as usize;
    if w == 0 || h == 0 {
        return;
    }
    let mask_idx = (y64 >> ss_ver) as usize;
    assert!(mask_idx < 4);
    assert!(bx4_base + w <= 64);
    let mask_shift: u32 = if (y64 & ss_ver) != 0 { 8 } else { 0 };
    let qv = _mm_set1_epi8(thr_lut[0][0] as u8 as i8);
    let sv = _mm_set1_epi8(thr_lut[1][0] as u8 as i8);
    let left_q = unsafe { _mm_loadu_si128(left_q_thr.as_ptr().cast()) };
    let left_s = unsafe { _mm_loadu_si128(left_side_thr.as_ptr().cast()) };
    for x in 0..w {
        let bits = (mask[bx4_base + x][4][mask_idx] >> mask_shift) as u16;
        let qbase = if x == 0 {
            setup_edge_u8x16_sse41(qv, left_q)
        } else {
            qv
        };
        let sbase = if x == 0 {
            setup_edge_u8x16_sse41(sv, left_s)
        } else {
            sv
        };
        let (qo, so) = setup_subpu_pair_u8x16_sse41(qbase, sbase, bits);
        setup_store_u8x16_sse41(q_thr_dst, x * 16, qo);
        setup_store_u8x16_sse41(side_thr_dst, x * 16, so);
    }
    left_q_thr[..h].fill(thr_lut[0][0] as u8);
    left_side_thr[..h].fill(thr_lut[1][0] as u8);
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn setup_thr_rows_seg_8bpc_sse41(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    segmap: &[u8],
    seg_off: isize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    above_thr_lut: Option<&[[u32; 16]; 2]>,
    above_seg: Option<(&[u8], isize)>,
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));
    let w = w4 as usize;
    let h = h4 as usize;
    let mask_idx = (sb64x >> ss_hor) as usize;
    assert!(mask_idx < 4);
    assert!(starty4 + h <= 64);
    if w == 0 || h == 0 {
        return;
    }
    let seg_off = usize::try_from(seg_off).expect("negative segment offset");
    let seg_stride = usize::try_from(seg_stride).expect("negative segment stride");
    assert!(seg_off + (h - 1) * seg_stride + w <= segmap.len());
    let mask_shift: u32 = if (sb64x & ss_hor) != 0 { 8 } else { 0 };
    let qlut = setup_lut_u8x16_sse41(&thr_lut[0]);
    let slut = setup_lut_u8x16_sse41(&thr_lut[1]);
    let (mut prev_q, mut prev_s) =
        if let (Some(alut), Some((aseg, aoff))) = (above_thr_lut, above_seg) {
            let off = usize::try_from(aoff).expect("negative above segment offset");
            assert!(off + w <= aseg.len());
            let segv = setup_load_seg_u8x16_sse41(aseg, off, w);
            (
                _mm_shuffle_epi8(setup_lut_u8x16_sse41(&alut[0]), segv),
                _mm_shuffle_epi8(setup_lut_u8x16_sse41(&alut[1]), segv),
            )
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
    for y in 0..h {
        let row = seg_off + y * seg_stride;
        let segv = setup_load_seg_u8x16_sse41(segmap, row, w);
        let cur_q = _mm_shuffle_epi8(qlut, segv);
        let cur_s = _mm_shuffle_epi8(slut, segv);
        let bits = (mask[starty4 + y][4][mask_idx] >> mask_shift) as u16;
        let (qo, so) = setup_subpu_pair_u8x16_sse41(
            setup_edge_u8x16_sse41(cur_q, prev_q),
            setup_edge_u8x16_sse41(cur_s, prev_s),
            bits,
        );
        setup_store_u8x16_sse41(q_thr_dst, y * 16, qo);
        setup_store_u8x16_sse41(side_thr_dst, y * 16, so);
        prev_q = cur_q;
        prev_s = cur_s;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn setup_thr_cols_seg_8bpc_sse41(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    segmap: &[u8],
    seg_off: isize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) {
    assert!((0..=16).contains(&w4));
    assert!((0..=16).contains(&h4));
    let w = w4 as usize;
    let h = h4 as usize;
    let mask_idx = (y64 >> ss_ver) as usize;
    assert!(mask_idx < 4);
    assert!(bx4_base + w <= 64);
    if w == 0 || h == 0 {
        return;
    }
    let seg_off = usize::try_from(seg_off).expect("negative segment offset");
    let seg_stride = usize::try_from(seg_stride).expect("negative segment stride");
    assert!(seg_off + (h - 1) * seg_stride + w <= segmap.len());
    let mask_shift: u32 = if (y64 & ss_ver) != 0 { 8 } else { 0 };
    let qlut = setup_lut_u8x16_sse41(&thr_lut[0]);
    let slut = setup_lut_u8x16_sse41(&thr_lut[1]);
    for y in 0..h {
        let row = seg_off + y * seg_stride;
        let segv = setup_load_seg_u8x16_sse41(segmap, row, w);
        let cur_q = _mm_shuffle_epi8(qlut, segv);
        let cur_s = _mm_shuffle_epi8(slut, segv);
        let cur_q_arr = setup_store_tmp_u8x16_sse41(cur_q);
        let cur_s_arr = setup_store_tmp_u8x16_sse41(cur_s);
        let prev_q = setup_prev_from_left_u8x16_sse41(cur_q, left_q_thr[y]);
        let prev_s = setup_prev_from_left_u8x16_sse41(cur_s, left_side_thr[y]);
        let mut bits = 0u16;
        let shift = mask_shift + y as u32;
        for x in 0..w {
            bits |= ((mask[bx4_base + x][4][mask_idx] >> shift) & 1) << x;
        }
        let (qo, so) = setup_subpu_pair_u8x16_sse41(
            setup_edge_u8x16_sse41(cur_q, prev_q),
            setup_edge_u8x16_sse41(cur_s, prev_s),
            bits,
        );
        let q_arr = setup_store_tmp_u8x16_sse41(qo);
        let s_arr = setup_store_tmp_u8x16_sse41(so);
        for x in 0..w {
            q_thr_dst[x * 16 + y] = q_arr[x];
            side_thr_dst[x * 16 + y] = s_arr[x];
        }
        left_q_thr[y] = cur_q_arr[w - 1];
        left_side_thr[y] = cur_s_arr[w - 1];
    }
}
