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

use std::arch::aarch64::*;

#[inline]
fn load4_u8_i32(dst: &[u8], base: isize, stride_line: isize) -> int32x4_t {
    unsafe {
        let p = dst.as_ptr();
        if stride_line == 1 {
            let b = vreinterpret_u8_u32(vld1_lane_u32::<0>(
                p.add(base as usize).cast::<u32>(),
                vdup_n_u32(0),
            ));
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b))))
        } else {
            let arr = [
                *p.add(base as usize) as i32,
                *p.add((base + stride_line) as usize) as i32,
                *p.add((base + 2 * stride_line) as usize) as i32,
                *p.add((base + 3 * stride_line) as usize) as i32,
            ];
            vld1q_s32(arr.as_ptr())
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn store4_clip_u8(dst: &mut [u8], base: isize, stride_line: isize, v: int32x4_t) {
    unsafe {
        let p = dst.as_mut_ptr();
        let u16x4 = vqmovun_s32(v);
        let u8x8 = vqmovn_u16(vcombine_u16(u16x4, u16x4));
        if stride_line == 1 {
            vst1_lane_u32::<0>(
                p.add(base as usize).cast::<u32>(),
                vreinterpret_u32_u8(u8x8),
            );
        } else {
            let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(u8x8));
            *p.add(base as usize) = (packed & 0xff) as u8;
            *p.add((base + stride_line) as usize) = ((packed >> 8) & 0xff) as u8;
            *p.add((base + 2 * stride_line) as usize) = ((packed >> 16) & 0xff) as u8;
            *p.add((base + 3 * stride_line) as usize) = (packed >> 24) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn load4_u8_i16_oriented<const CONTIG: bool>(
    dst: &[u8],
    base: isize,
    stride_line: isize,
) -> int16x8_t {
    unsafe {
        let p = dst.as_ptr();
        if CONTIG {
            let b = vreinterpret_u8_u32(vld1_lane_u32::<0>(
                p.add(base as usize).cast::<u32>(),
                vdup_n_u32(0),
            ));
            vreinterpretq_s16_u16(vmovl_u8(b))
        } else {
            let mut v = vdupq_n_s16(0);
            v = vsetq_lane_s16::<0>(*p.add(base as usize) as i16, v);
            v = vsetq_lane_s16::<1>(*p.add((base + stride_line) as usize) as i16, v);
            v = vsetq_lane_s16::<2>(*p.add((base + 2 * stride_line) as usize) as i16, v);
            vsetq_lane_s16::<3>(*p.add((base + 3 * stride_line) as usize) as i16, v)
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn store4_clip_u8_i16_oriented<const CONTIG: bool>(
    dst: &mut [u8],
    base: isize,
    stride_line: isize,
    v: int16x8_t,
) {
    unsafe {
        let p = dst.as_mut_ptr();
        let u8x8 = vreinterpret_u32_u8(vqmovun_s16(v));
        if CONTIG {
            vst1_lane_u32::<0>(p.add(base as usize).cast::<u32>(), u8x8);
        } else {
            let packed = vget_lane_u32::<0>(u8x8);
            *p.add(base as usize) = (packed & 0xff) as u8;
            *p.add((base + stride_line) as usize) = ((packed >> 8) & 0xff) as u8;
            *p.add((base + 2 * stride_line) as usize) = ((packed >> 16) & 0xff) as u8;
            *p.add((base + 3 * stride_line) as usize) = (packed >> 24) as u8;
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_delta_i16(
    d0: int16x8_t,
    dm1: int16x8_t,
    dp1: int16x8_t,
    dm2: int16x8_t,
    nqc: int16x8_t,
    qc: int16x8_t,
) -> int16x8_t {
    let d0_m1 = vsubq_s16(d0, dm1);
    let dp1_m2 = vsubq_s16(dp1, dm2);
    let inner = vsubq_s16(vaddq_s16(d0_m1, vaddq_s16(d0_m1, d0_m1)), dp1_m2);
    vminq_s16(vmaxq_s16(vshlq_n_s16::<2>(inner), nqc), qc)
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_diff_i16(delta: int16x8_t, width: i32, tap: i32) -> int16x8_t {
    let coeff = (crate::deblock::W_MULT[(width - 1) as usize] as i32 * tap * 16) as i16;
    vqrdmulhq_s16(delta, vdupq_n_s16(coeff))
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_extract_i16(v: int16x8_t, lane: i32) -> i16 {
    match lane {
        0 => vgetq_lane_s16::<0>(v),
        1 => vgetq_lane_s16::<1>(v),
        2 => vgetq_lane_s16::<2>(v),
        _ => vgetq_lane_s16::<3>(v),
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn load_i16x8(a: [i16; 8]) -> int16x8_t {
    unsafe { vld1q_s16(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_h_sym4_rows(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    delta: int16x8_t,
    apply_neg: bool,
    apply_pos: bool,
) {
    let wm = (crate::deblock::W_MULT[3] as i16) * 16;
    let neg = if apply_neg { 1 } else { 0 };
    let pos = if apply_pos { -1 } else { 0 };
    let coeff = load_i16x8([wm, wm * 2, wm * 3, wm * 4, wm * 4, wm * 3, wm * 2, wm]);
    // Keep the tap coefficient positive and apply +/- after sqrdmulh.
    // The rounded high multiply is not sign-symmetric, while scalar computes
    // the positive rounded diff first and then adds/subtracts it.
    let sign = load_i16x8([neg, neg, neg, neg, pos, pos, pos, pos]);

    unsafe {
        let p = dst.as_mut_ptr();
        let mut r = 0;
        while r < 4 {
            let row = off + r * stride_line - 4;
            let bytes = vld1_u8(p.add(row as usize));
            let pix = vreinterpretq_s16_u16(vmovl_u8(bytes));
            let d = deblock_extract_i16(delta, r as i32);
            let diff = vmulq_s16(vqrdmulhq_s16(vdupq_n_s16(d), coeff), sign);
            let res = vaddq_s16(pix, diff);
            vst1_u8(p.add(row as usize), vqmovun_s16(res));
            r += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_h_sym8_rows(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    delta: int16x8_t,
    apply_neg: bool,
    apply_pos: bool,
) {
    let wm = (crate::deblock::W_MULT[7] as i16) * 16;
    let neg = if apply_neg { 1 } else { 0 };
    let pos = if apply_pos { -1 } else { 0 };
    let coeff_lo = load_i16x8([wm, wm * 2, wm * 3, wm * 4, wm * 5, wm * 6, wm * 7, wm * 8]);
    let coeff_hi = load_i16x8([wm * 8, wm * 7, wm * 6, wm * 5, wm * 4, wm * 3, wm * 2, wm]);
    // Keep the tap coefficient positive and apply +/- after sqrdmulh.
    // The rounded high multiply is not sign-symmetric, while scalar computes
    // the positive rounded diff first and then adds/subtracts it.
    let sign_lo = vdupq_n_s16(neg);
    let sign_hi = vdupq_n_s16(pos);

    unsafe {
        let p = dst.as_mut_ptr();
        let mut r = 0;
        while r < 4 {
            let row = off + r * stride_line - 8;
            let bytes = vld1q_u8(p.add(row as usize));
            let pix_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes)));
            let pix_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bytes)));
            let d = deblock_extract_i16(delta, r as i32);
            let d_v = vdupq_n_s16(d);
            let diff_lo = vmulq_s16(vqrdmulhq_s16(d_v, coeff_lo), sign_lo);
            let diff_hi = vmulq_s16(vqrdmulhq_s16(d_v, coeff_hi), sign_hi);
            let res_lo = vaddq_s16(pix_lo, diff_lo);
            let res_hi = vaddq_s16(pix_hi, diff_hi);
            vst1q_u8(
                p.add(row as usize),
                vcombine_u8(vqmovun_s16(res_lo), vqmovun_s16(res_hi)),
            );
            r += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_const_oriented<const WN: i32, const WP: i32, const CONTIG: bool>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    debug_assert!((1..=8).contains(&WN));
    debug_assert!((1..=8).contains(&WP));
    let apply_neg = !neg_lossless;
    let apply_pos = !pos_lossless;
    debug_assert!(apply_neg || apply_pos);
    debug_assert!(q_thr_clamp <= i16::MAX as i32);

    let qc = vdupq_n_s16(q_thr_clamp as i16);
    let nqc = vdupq_n_s16(-(q_thr_clamp as i16));
    let d0 = load4_u8_i16_oriented::<CONTIG>(dst, off, stride_line);
    let dm1 = load4_u8_i16_oriented::<CONTIG>(dst, off - stride_tap, stride_line);
    let dp1 = load4_u8_i16_oriented::<CONTIG>(dst, off + stride_tap, stride_line);
    let dm2 = load4_u8_i16_oriented::<CONTIG>(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i16(d0, dm1, dp1, dm2, nqc, qc);

    if !CONTIG && stride_tap == 1 && WN == WP {
        if WN == 8 {
            deblock_apply_8bpc_neon_h_sym8_rows(dst, off, stride_line, delta, apply_neg, apply_pos);
            return;
        }
        if WN == 4 {
            deblock_apply_8bpc_neon_h_sym4_rows(dst, off, stride_line, delta, apply_neg, apply_pos);
            return;
        }
    }

    if apply_neg {
        let mut j = 0;
        while j < WN {
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u8_i16_oriented::<CONTIG>(dst, base, stride_line);
            let diff = deblock_diff_i16(delta, WN, WN - j);
            store4_clip_u8_i16_oriented::<CONTIG>(dst, base, stride_line, vaddq_s16(cur, diff));
            j += 1;
        }
    }

    if apply_pos {
        let mut j = 0;
        while j < WP {
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u8_i16_oriented::<CONTIG>(dst, base, stride_line);
            let diff = deblock_diff_i16(delta, WP, WP - j);
            store4_clip_u8_i16_oriented::<CONTIG>(dst, base, stride_line, vsubq_s16(cur, diff));
            j += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn transpose16x16_u8_neon(r: &mut [uint8x16_t; 16]) {
    // Native NEON transpose spelling.  vtrn de-interleaves even/odd elements at
    // each stage, so the raw outputs are produced in bit-reversed column order:
    // 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15.
    // The final assignment below restores the natural column order.
    let z = vdupq_n_u8(0);
    let mut t = [z; 16];
    let mut u = [z; 16];
    let mut v = [z; 16];
    let mut o = [z; 16];

    let mut i = 0;
    while i < 8 {
        let a = r[i * 2];
        let b = r[i * 2 + 1];
        t[i * 2] = vtrn1q_u8(a, b);
        t[i * 2 + 1] = vtrn2q_u8(a, b);
        i += 1;
    }

    i = 0;
    while i < 4 {
        let b = i * 4;
        let t0 = vreinterpretq_u16_u8(t[b]);
        let t1 = vreinterpretq_u16_u8(t[b + 1]);
        let t2 = vreinterpretq_u16_u8(t[b + 2]);
        let t3 = vreinterpretq_u16_u8(t[b + 3]);
        u[b] = vreinterpretq_u8_u16(vtrn1q_u16(t0, t2));
        u[b + 1] = vreinterpretq_u8_u16(vtrn2q_u16(t0, t2));
        u[b + 2] = vreinterpretq_u8_u16(vtrn1q_u16(t1, t3));
        u[b + 3] = vreinterpretq_u8_u16(vtrn2q_u16(t1, t3));
        i += 1;
    }

    i = 0;
    while i < 2 {
        let b = i * 8;
        let u0 = vreinterpretq_u32_u8(u[b]);
        let u1 = vreinterpretq_u32_u8(u[b + 1]);
        let u2 = vreinterpretq_u32_u8(u[b + 2]);
        let u3 = vreinterpretq_u32_u8(u[b + 3]);
        let u4 = vreinterpretq_u32_u8(u[b + 4]);
        let u5 = vreinterpretq_u32_u8(u[b + 5]);
        let u6 = vreinterpretq_u32_u8(u[b + 6]);
        let u7 = vreinterpretq_u32_u8(u[b + 7]);
        v[b] = vreinterpretq_u8_u32(vtrn1q_u32(u0, u4));
        v[b + 1] = vreinterpretq_u8_u32(vtrn2q_u32(u0, u4));
        v[b + 2] = vreinterpretq_u8_u32(vtrn1q_u32(u1, u5));
        v[b + 3] = vreinterpretq_u8_u32(vtrn2q_u32(u1, u5));
        v[b + 4] = vreinterpretq_u8_u32(vtrn1q_u32(u2, u6));
        v[b + 5] = vreinterpretq_u8_u32(vtrn2q_u32(u2, u6));
        v[b + 6] = vreinterpretq_u8_u32(vtrn1q_u32(u3, u7));
        v[b + 7] = vreinterpretq_u8_u32(vtrn2q_u32(u3, u7));
        i += 1;
    }

    let v0 = vreinterpretq_u64_u8(v[0]);
    let v1 = vreinterpretq_u64_u8(v[1]);
    let v2 = vreinterpretq_u64_u8(v[2]);
    let v3 = vreinterpretq_u64_u8(v[3]);
    let v4 = vreinterpretq_u64_u8(v[4]);
    let v5 = vreinterpretq_u64_u8(v[5]);
    let v6 = vreinterpretq_u64_u8(v[6]);
    let v7 = vreinterpretq_u64_u8(v[7]);
    let v8 = vreinterpretq_u64_u8(v[8]);
    let v9 = vreinterpretq_u64_u8(v[9]);
    let v10 = vreinterpretq_u64_u8(v[10]);
    let v11 = vreinterpretq_u64_u8(v[11]);
    let v12 = vreinterpretq_u64_u8(v[12]);
    let v13 = vreinterpretq_u64_u8(v[13]);
    let v14 = vreinterpretq_u64_u8(v[14]);
    let v15 = vreinterpretq_u64_u8(v[15]);

    o[0] = vreinterpretq_u8_u64(vtrn1q_u64(v0, v8));
    o[1] = vreinterpretq_u8_u64(vtrn2q_u64(v0, v8));
    o[2] = vreinterpretq_u8_u64(vtrn1q_u64(v1, v9));
    o[3] = vreinterpretq_u8_u64(vtrn2q_u64(v1, v9));
    o[4] = vreinterpretq_u8_u64(vtrn1q_u64(v2, v10));
    o[5] = vreinterpretq_u8_u64(vtrn2q_u64(v2, v10));
    o[6] = vreinterpretq_u8_u64(vtrn1q_u64(v3, v11));
    o[7] = vreinterpretq_u8_u64(vtrn2q_u64(v3, v11));
    o[8] = vreinterpretq_u8_u64(vtrn1q_u64(v4, v12));
    o[9] = vreinterpretq_u8_u64(vtrn2q_u64(v4, v12));
    o[10] = vreinterpretq_u8_u64(vtrn1q_u64(v5, v13));
    o[11] = vreinterpretq_u8_u64(vtrn2q_u64(v5, v13));
    o[12] = vreinterpretq_u8_u64(vtrn1q_u64(v6, v14));
    o[13] = vreinterpretq_u8_u64(vtrn2q_u64(v6, v14));
    o[14] = vreinterpretq_u8_u64(vtrn1q_u64(v7, v15));
    o[15] = vreinterpretq_u8_u64(vtrn2q_u64(v7, v15));

    r[0] = o[0];
    r[1] = o[8];
    r[2] = o[4];
    r[3] = o[12];
    r[4] = o[2];
    r[5] = o[10];
    r[6] = o[6];
    r[7] = o[14];
    r[8] = o[1];
    r[9] = o[9];
    r[10] = o[5];
    r[11] = o[13];
    r[12] = o[3];
    r[13] = o[11];
    r[14] = o[7];
    r[15] = o[15];
}

#[inline]
#[target_feature(enable = "rdm")]
fn cvtepu8_lo_i16(v: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)))
}

#[inline]
#[target_feature(enable = "rdm")]
fn cvtepu8_hi_i16(v: uint8x16_t) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v)))
}

#[inline]
#[target_feature(enable = "rdm")]
fn pack_u8_from_i16x2(lo: int16x8_t, hi: int16x8_t) -> uint8x16_t {
    vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi))
}

#[inline]
#[target_feature(enable = "rdm")]
fn repeated_qclamp4_w8(q_thr: &[u8], qi: usize) -> (int16x8_t, int16x8_t) {
    let m = crate::deblock::Q_THRESH_MULTS[7] as i16;
    let q0 = (q_thr[qi] as i16) * m;
    let q1 = (q_thr[qi + 1] as i16) * m;
    let q2 = (q_thr[qi + 2] as i16) * m;
    let q3 = (q_thr[qi + 3] as i16) * m;
    (
        load_i16x8([q0, q0, q0, q0, q1, q1, q1, q1]),
        load_i16x8([q2, q2, q2, q2, q3, q3, q3, q3]),
    )
}

#[inline]
#[target_feature(enable = "rdm")]
fn repeated_apply_mask4(ll: u16, qi: usize) -> (int16x8_t, int16x8_t) {
    let m0 = if (ll & (1u16 << qi)) == 0 { -1i16 } else { 0 };
    let m1 = if (ll & (1u16 << (qi + 1))) == 0 {
        -1i16
    } else {
        0
    };
    let m2 = if (ll & (1u16 << (qi + 2))) == 0 {
        -1i16
    } else {
        0
    };
    let m3 = if (ll & (1u16 << (qi + 3))) == 0 {
        -1i16
    } else {
        0
    };
    (
        load_i16x8([m0, m0, m0, m0, m1, m1, m1, m1]),
        load_i16x8([m2, m2, m2, m2, m3, m3, m3, m3]),
    )
}

#[inline]
#[target_feature(enable = "rdm")]
fn and_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vreinterpretq_s16_u16(vandq_u16(
        vreinterpretq_u16_s16(a),
        vreinterpretq_u16_s16(b),
    ))
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_h_w8x4_transpose(
    dst: &mut [u8],
    off: isize,
    stride: isize,
    qlo: int16x8_t,
    qhi: int16x8_t,
    neg_mask_lo: int16x8_t,
    neg_mask_hi: int16x8_t,
    pos_mask_lo: int16x8_t,
    pos_mask_hi: int16x8_t,
) {
    unsafe {
        let p = dst.as_mut_ptr();
        let z8 = vdupq_n_u8(0);
        let z16 = vdupq_n_s16(0);
        let mut cols = [z8; 16];

        let mut r = 0;
        while r < 16 {
            let row = off + r as isize * stride - 8;
            cols[r] = vld1q_u8(p.add(row as usize));
            r += 1;
        }

        transpose16x16_u8_neon(&mut cols);

        let d0_lo = cvtepu8_lo_i16(cols[8]);
        let d0_hi = cvtepu8_hi_i16(cols[8]);
        let dm1_lo = cvtepu8_lo_i16(cols[7]);
        let dm1_hi = cvtepu8_hi_i16(cols[7]);
        let dp1_lo = cvtepu8_lo_i16(cols[9]);
        let dp1_hi = cvtepu8_hi_i16(cols[9]);
        let dm2_lo = cvtepu8_lo_i16(cols[6]);
        let dm2_hi = cvtepu8_hi_i16(cols[6]);

        let delta_lo = deblock_delta_i16(d0_lo, dm1_lo, dp1_lo, dm2_lo, vsubq_s16(z16, qlo), qlo);
        let delta_hi = deblock_delta_i16(d0_hi, dm1_hi, dp1_hi, dm2_hi, vsubq_s16(z16, qhi), qhi);
        let wm = (crate::deblock::W_MULT[7] as i16) * 16;

        let mut c = 0;
        while c < 8 {
            let tap = (c + 1) as i16;
            let coeff = vdupq_n_s16(wm * tap);
            let diff_lo = and_s16(vqrdmulhq_s16(delta_lo, coeff), neg_mask_lo);
            let diff_hi = and_s16(vqrdmulhq_s16(delta_hi, coeff), neg_mask_hi);
            let pix_lo = cvtepu8_lo_i16(cols[c]);
            let pix_hi = cvtepu8_hi_i16(cols[c]);
            cols[c] = pack_u8_from_i16x2(vaddq_s16(pix_lo, diff_lo), vaddq_s16(pix_hi, diff_hi));
            c += 1;
        }

        c = 8;
        while c < 16 {
            let tap = (16 - c) as i16;
            let coeff = vdupq_n_s16(wm * tap);
            let diff_lo = and_s16(vqrdmulhq_s16(delta_lo, coeff), pos_mask_lo);
            let diff_hi = and_s16(vqrdmulhq_s16(delta_hi, coeff), pos_mask_hi);
            let pix_lo = cvtepu8_lo_i16(cols[c]);
            let pix_hi = cvtepu8_hi_i16(cols[c]);
            cols[c] = pack_u8_from_i16x2(vsubq_s16(pix_lo, diff_lo), vsubq_s16(pix_hi, diff_hi));
            c += 1;
        }

        transpose16x16_u8_neon(&mut cols);

        r = 0;
        while r < 16 {
            let row = off + r as isize * stride - 8;
            vst1q_u8(p.add(row as usize), cols[r]);
            r += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_const<const WN: i32, const WP: i32>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    if stride_line == 1 {
        deblock_apply_8bpc_neon_const_oriented::<WN, WP, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
    } else {
        debug_assert_eq!(stride_tap, 1);
        deblock_apply_8bpc_neon_const_oriented::<WN, WP, false>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
    }
}

macro_rules! dispatch_8bpc_pair_neon {
    ($dst:expr, $off:expr, $stride_line:expr, $stride_tap:expr, $q:expr, $neg_ll:expr, $pos_ll:expr, $wn:literal, $wp:literal) => {{
        deblock_apply_8bpc_neon_const::<$wn, $wp>(
            $dst,
            $off,
            $stride_line,
            $stride_tap,
            $q,
            $neg_ll,
            $pos_ll,
        )
    }};
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_specialized(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) -> bool {
    if q_thr_clamp > i16::MAX as i32 {
        return false;
    }

    match (width_neg, width_pos) {
        (1, 1) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            1,
            1
        ),
        (1, 2) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            1,
            2
        ),
        (2, 2) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            2,
            2
        ),
        (2, 3) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            2,
            3
        ),
        (1, 3) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            1,
            3
        ),
        (3, 3) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            3,
            3
        ),
        (1, 4) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            1,
            4
        ),
        (2, 4) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            2,
            4
        ),
        (3, 4) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            3,
            4
        ),
        (4, 4) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            4,
            4
        ),
        (1, 6) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            1,
            6
        ),
        (2, 6) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            2,
            6
        ),
        (3, 6) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            3,
            6
        ),
        (4, 6) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            4,
            6
        ),
        (6, 6) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            6,
            6
        ),
        (1, 8) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            1,
            8
        ),
        (2, 8) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            2,
            8
        ),
        (3, 8) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            3,
            8
        ),
        (4, 8) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            4,
            8
        ),
        (6, 8) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            6,
            8
        ),
        (8, 8) => dispatch_8bpc_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            8,
            8
        ),
        _ => return false,
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_apply_8bpc_neon(
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
    if q_thr_clamp <= 0 || (neg_lossless && pos_lossless) {
        return;
    }

    if deblock_apply_8bpc_neon_specialized(
        dst,
        off,
        stride_line,
        stride_tap,
        width_neg,
        width_pos,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
    ) {
        return;
    }

    let qc = vdupq_n_s32(q_thr_clamp);
    let nqc = vdupq_n_s32(-q_thr_clamp);
    let rnd = vdupq_n_s32(1 << 10);
    let zero = vdupq_n_s32(0);
    let v255 = vdupq_n_s32(255);
    let d0 = load4_u8_i32(dst, off, stride_line);
    let dm1 = load4_u8_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u8_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u8_i32(dst, off - 2 * stride_tap, stride_line);
    let d0_m1 = vsubq_s32(d0, dm1);
    let dp1_m2 = vsubq_s32(dp1, dm2);
    let inner = vsubq_s32(vaddq_s32(d0_m1, vaddq_s32(d0_m1, d0_m1)), dp1_m2);
    let delta = vminq_s32(vmaxq_s32(vshlq_n_s32::<2>(inner), nqc), qc);

    if !neg_lossless {
        let dn = vmulq_s32(
            delta,
            vdupq_n_s32(crate::deblock::W_MULT[(width_neg - 1) as usize] as i32),
        );
        for j in 0..width_neg {
            let diff = vshrq_n_s32::<11>(vaddq_s32(vmulq_s32(dn, vdupq_n_s32(width_neg - j)), rnd));
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u8_i32(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vaddq_s32(cur, diff), zero), v255);
            store4_clip_u8(dst, base, stride_line, res);
        }
    }

    if !pos_lossless {
        let dpv = vmulq_s32(
            delta,
            vdupq_n_s32(crate::deblock::W_MULT[(width_pos - 1) as usize] as i32),
        );
        for j in 0..width_pos {
            let diff =
                vshrq_n_s32::<11>(vaddq_s32(vmulq_s32(dpv, vdupq_n_s32(width_pos - j)), rnd));
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u8_i32(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vsubq_s32(cur, diff), zero), v255);
            store4_clip_u8(dst, base, stride_line, res);
        }
    }
}

#[inline]
fn load4_u16_i32(dst: &[u16], base: isize, stride_line: isize) -> int32x4_t {
    unsafe {
        let p = dst.as_ptr();
        if stride_line == 1 {
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p.add(base as usize))))
        } else {
            let arr = [
                *p.add(base as usize) as i32,
                *p.add((base + stride_line) as usize) as i32,
                *p.add((base + 2 * stride_line) as usize) as i32,
                *p.add((base + 3 * stride_line) as usize) as i32,
            ];
            vld1q_s32(arr.as_ptr())
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn store4_clip_u16(dst: &mut [u16], base: isize, stride_line: isize, v: int32x4_t) {
    unsafe {
        let p = dst.as_mut_ptr();
        let u16x4 = vqmovun_s32(v);
        if stride_line == 1 {
            vst1_u16(p.add(base as usize), u16x4);
        } else {
            *p.add(base as usize) = vget_lane_u16::<0>(u16x4);
            *p.add((base + stride_line) as usize) = vget_lane_u16::<1>(u16x4);
            *p.add((base + 2 * stride_line) as usize) = vget_lane_u16::<2>(u16x4);
            *p.add((base + 3 * stride_line) as usize) = vget_lane_u16::<3>(u16x4);
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn load4_u16_i32_oriented<const CONTIG: bool>(
    dst: &[u16],
    base: isize,
    stride_line: isize,
) -> int32x4_t {
    unsafe {
        let p = dst.as_ptr();
        if CONTIG {
            vreinterpretq_s32_u32(vmovl_u16(vld1_u16(p.add(base as usize))))
        } else {
            let mut v = vdupq_n_s32(0);
            v = vsetq_lane_s32::<0>(*p.add(base as usize) as i32, v);
            v = vsetq_lane_s32::<1>(*p.add((base + stride_line) as usize) as i32, v);
            v = vsetq_lane_s32::<2>(*p.add((base + 2 * stride_line) as usize) as i32, v);
            vsetq_lane_s32::<3>(*p.add((base + 3 * stride_line) as usize) as i32, v)
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn store4_clip_u16_oriented<const CONTIG: bool>(
    dst: &mut [u16],
    base: isize,
    stride_line: isize,
    v: int32x4_t,
) {
    unsafe {
        let p = dst.as_mut_ptr();
        let u16x4 = vqmovun_s32(v);
        if CONTIG {
            vst1_u16(p.add(base as usize), u16x4);
        } else {
            *p.add(base as usize) = vget_lane_u16::<0>(u16x4);
            *p.add((base + stride_line) as usize) = vget_lane_u16::<1>(u16x4);
            *p.add((base + 2 * stride_line) as usize) = vget_lane_u16::<2>(u16x4);
            *p.add((base + 3 * stride_line) as usize) = vget_lane_u16::<3>(u16x4);
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_delta_i32(
    d0: int32x4_t,
    dm1: int32x4_t,
    dp1: int32x4_t,
    dm2: int32x4_t,
    nqc: int32x4_t,
    qc: int32x4_t,
) -> int32x4_t {
    let d0_m1 = vsubq_s32(d0, dm1);
    let dp1_m2 = vsubq_s32(dp1, dm2);
    let inner = vsubq_s32(vaddq_s32(d0_m1, vaddq_s32(d0_m1, d0_m1)), dp1_m2);
    vminq_s32(vmaxq_s32(vshlq_n_s32::<2>(inner), nqc), qc)
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_diff_i32(delta: int32x4_t, width: i32, tap: i32) -> int32x4_t {
    let rnd = vdupq_n_s32(1 << 10);
    let w = vdupq_n_s32(crate::deblock::W_MULT[(width - 1) as usize] as i32 * tap);
    vshrq_n_s32::<11>(vaddq_s32(vmulq_s32(delta, w), rnd))
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_hbd_neon_const_oriented<
    const WN: i32,
    const WP: i32,
    const CONTIG: bool,
    const APPLY_NEG: bool,
    const APPLY_POS: bool,
>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    bitdepth_max: i32,
) {
    debug_assert!((1..=8).contains(&WN));
    debug_assert!((1..=8).contains(&WP));
    debug_assert!(APPLY_NEG || APPLY_POS);

    let qc = vdupq_n_s32(q_thr_clamp);
    let nqc = vdupq_n_s32(-q_thr_clamp);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(bitdepth_max);

    let d0 = load4_u16_i32_oriented::<CONTIG>(dst, off, stride_line);
    let dm1 = load4_u16_i32_oriented::<CONTIG>(dst, off - stride_tap, stride_line);
    let dp1 = load4_u16_i32_oriented::<CONTIG>(dst, off + stride_tap, stride_line);
    let dm2 = load4_u16_i32_oriented::<CONTIG>(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i32(d0, dm1, dp1, dm2, nqc, qc);

    if APPLY_NEG {
        let mut j = 0;
        while j < WN {
            let diff = deblock_diff_i32(delta, WN, WN - j);
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u16_i32_oriented::<CONTIG>(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vaddq_s32(cur, diff), zero), vmax);
            store4_clip_u16_oriented::<CONTIG>(dst, base, stride_line, res);
            j += 1;
        }
    }

    if APPLY_POS {
        let mut j = 0;
        while j < WP {
            let diff = deblock_diff_i32(delta, WP, WP - j);
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u16_i32_oriented::<CONTIG>(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vsubq_s32(cur, diff), zero), vmax);
            store4_clip_u16_oriented::<CONTIG>(dst, base, stride_line, res);
            j += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_hbd_neon_const_sides<const WN: i32, const WP: i32, const CONTIG: bool>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    match (neg_lossless, pos_lossless) {
        (false, false) => deblock_apply_hbd_neon_const_oriented::<WN, WP, CONTIG, true, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            bitdepth_max,
        ),
        (false, true) => deblock_apply_hbd_neon_const_oriented::<WN, WP, CONTIG, true, false>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            bitdepth_max,
        ),
        (true, false) => deblock_apply_hbd_neon_const_oriented::<WN, WP, CONTIG, false, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            bitdepth_max,
        ),
        (true, true) => {}
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_hbd_neon_const<const WN: i32, const WP: i32>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    if stride_line == 1 {
        deblock_apply_hbd_neon_const_sides::<WN, WP, true>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        );
    } else {
        debug_assert_eq!(stride_tap, 1);
        deblock_apply_hbd_neon_const_sides::<WN, WP, false>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        );
    }
}

macro_rules! dispatch_hbd_pair_neon {
    ($dst:expr, $off:expr, $stride_line:expr, $stride_tap:expr, $q:expr, $neg_ll:expr, $pos_ll:expr, $bdmax:expr, $wn:literal, $wp:literal) => {{
        deblock_apply_hbd_neon_const::<$wn, $wp>(
            $dst,
            $off,
            $stride_line,
            $stride_tap,
            $q,
            $neg_ll,
            $pos_ll,
            $bdmax,
        )
    }};
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_hbd_neon_specialized(
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
) -> bool {
    match (width_neg, width_pos) {
        (1, 1) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            1,
            1
        ),
        (1, 2) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            1,
            2
        ),
        (2, 2) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            2,
            2
        ),
        (2, 3) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            2,
            3
        ),
        (1, 3) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            1,
            3
        ),
        (3, 3) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            3,
            3
        ),
        (1, 4) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            1,
            4
        ),
        (2, 4) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            2,
            4
        ),
        (3, 4) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            3,
            4
        ),
        (4, 4) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            4,
            4
        ),
        (1, 6) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            1,
            6
        ),
        (2, 6) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            2,
            6
        ),
        (3, 6) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            3,
            6
        ),
        (4, 6) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            4,
            6
        ),
        (6, 6) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            6,
            6
        ),
        (1, 8) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            1,
            8
        ),
        (2, 8) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            2,
            8
        ),
        (3, 8) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            3,
            8
        ),
        (4, 8) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            4,
            8
        ),
        (6, 8) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            6,
            8
        ),
        (8, 8) => dispatch_hbd_pair_neon!(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
            8,
            8
        ),
        _ => return false,
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_apply_hbd_neon(
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
    if q_thr_clamp <= 0 || (neg_lossless && pos_lossless) {
        return;
    }

    if deblock_apply_hbd_neon_specialized(
        dst,
        off,
        stride_line,
        stride_tap,
        width_neg,
        width_pos,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
        bitdepth_max,
    ) {
        return;
    }

    let qc = vdupq_n_s32(q_thr_clamp);
    let nqc = vdupq_n_s32(-q_thr_clamp);
    let rnd = vdupq_n_s32(1 << 10);
    let zero = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(bitdepth_max);

    let d0 = load4_u16_i32(dst, off, stride_line);
    let dm1 = load4_u16_i32(dst, off - stride_tap, stride_line);
    let dp1 = load4_u16_i32(dst, off + stride_tap, stride_line);
    let dm2 = load4_u16_i32(dst, off - 2 * stride_tap, stride_line);
    let delta = deblock_delta_i32(d0, dm1, dp1, dm2, nqc, qc);

    if !neg_lossless {
        let dn = vmulq_s32(
            delta,
            vdupq_n_s32(crate::deblock::W_MULT[(width_neg - 1) as usize] as i32),
        );
        for j in 0..width_neg {
            let diff = vshrq_n_s32::<11>(vaddq_s32(vmulq_s32(dn, vdupq_n_s32(width_neg - j)), rnd));
            let base = off + (-(j as isize) - 1) * stride_tap;
            let cur = load4_u16_i32(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vaddq_s32(cur, diff), zero), vmax);
            store4_clip_u16(dst, base, stride_line, res);
        }
    }

    if !pos_lossless {
        let dpv = vmulq_s32(
            delta,
            vdupq_n_s32(crate::deblock::W_MULT[(width_pos - 1) as usize] as i32),
        );
        for j in 0..width_pos {
            let diff =
                vshrq_n_s32::<11>(vaddq_s32(vmulq_s32(dpv, vdupq_n_s32(width_pos - j)), rnd));
            let base = off + (j as isize) * stride_tap;
            let cur = load4_u16_i32(dst, base, stride_line);
            let res = vminq_s32(vmaxq_s32(vsubq_s32(cur, diff), zero), vmax);
            store4_clip_u16(dst, base, stride_line, res);
        }
    }
}

#[inline]
#[target_feature(enable = "rdm")]
fn select_i32(mask: bool, yes: i32, no: i32) -> i32 {
    let m = -(mask as i32);
    (yes & m) | (no & !m)
}

#[inline]
#[target_feature(enable = "rdm")]
fn filter_avg_abs2_from_lanes(v: int16x4_t) -> u32 {
    ((vget_lane_s16::<0>(v) as u32 + vget_lane_s16::<1>(v) as u32) + 1) >> 1
}

#[inline]
#[target_feature(enable = "rdm")]
fn filter_second_deriv_8bpc_neon(
    buf: &[u8],
    s: isize,
    t: isize,
    stride: isize,
    dist: isize,
) -> u32 {
    unsafe {
        let p = buf.as_ptr();
        let s0 = *p.add((s + (dist - 1) * stride) as usize) as i16;
        let s1 = *p.add((s + dist * stride) as usize) as i16;
        let s2 = *p.add((s + (dist + 1) * stride) as usize) as i16;
        let t0 = *p.add((t + (dist - 1) * stride) as usize) as i16;
        let t1 = *p.add((t + dist * stride) as usize) as i16;
        let t2 = *p.add((t + (dist + 1) * stride) as usize) as i16;
        let a = vset_lane_s16::<1>(t0, vset_lane_s16::<0>(s0, vdup_n_s16(0)));
        let b = vset_lane_s16::<1>(t1, vset_lane_s16::<0>(s1, vdup_n_s16(0)));
        let c = vset_lane_s16::<1>(t2, vset_lane_s16::<0>(s2, vdup_n_s16(0)));
        let deriv = vadd_s16(vsub_s16(a, vadd_s16(b, b)), c);
        filter_avg_abs2_from_lanes(vabs_s16(deriv))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn filter_end_deriv_8bpc_neon(
    buf: &[u8],
    s0: isize,
    s1: isize,
    s2: isize,
    t0: isize,
    t1: isize,
    t2: isize,
    c0: i16,
    c1: i16,
    c2: i16,
) -> u32 {
    unsafe {
        let p = buf.as_ptr();
        let a = vset_lane_s16::<1>(
            *p.add(t0 as usize) as i16,
            vset_lane_s16::<0>(*p.add(s0 as usize) as i16, vdup_n_s16(0)),
        );
        let b = vset_lane_s16::<1>(
            *p.add(t1 as usize) as i16,
            vset_lane_s16::<0>(*p.add(s1 as usize) as i16, vdup_n_s16(0)),
        );
        let c = vset_lane_s16::<1>(
            *p.add(t2 as usize) as i16,
            vset_lane_s16::<0>(*p.add(s2 as usize) as i16, vdup_n_s16(0)),
        );
        let v = vadd_s16(
            vadd_s16(vmul_n_s16(a, c0), vmul_n_s16(b, c1)),
            vmul_n_s16(c, c2),
        );
        filter_avg_abs2_from_lanes(vabs_s16(v))
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn filter_choice_8bpc_neon_const<const MAX_WIDTH_NEG: i32, const MAX_WIDTH_POS: i32>(
    buf: &[u8],
    s: isize,
    t: isize,
    stride: isize,
    q_thr: u32,
    side_thr: u32,
) -> i32 {
    debug_assert!((1..=8).contains(&MAX_WIDTH_POS));
    debug_assert!((1..=8).contains(&MAX_WIDTH_NEG));
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    let sd_m2 = filter_second_deriv_8bpc_neon(buf, s, t, stride, -2);
    let sd_m1 = filter_second_deriv_8bpc_neon(buf, s, t, stride, -1);
    let sd_0 = filter_second_deriv_8bpc_neon(buf, s, t, stride, 0);
    let sd_1 = filter_second_deriv_8bpc_neon(buf, s, t, stride, 1);

    let high_deriv = sd_m2.max(sd_1);
    let transition = sd_m1 + sd_0;

    let fail0 = high_deriv > side_thr;
    if MAX_WIDTH_POS == 1 {
        return select_i32(fail0, 0, 1);
    }

    let fail1 = high_deriv > (side_thr >> 2) || transition > q_thr * 4;

    let end_thr = (side_thr * 3) >> 4;
    let neg3_fail = if MAX_WIDTH_NEG >= 3 {
        filter_end_deriv_8bpc_neon(
            buf,
            s - stride,
            s - 2 * stride,
            s - 4 * stride,
            t - stride,
            t - 2 * stride,
            t - 4 * stride,
            -2,
            3,
            -1,
        ) > end_thr
    } else {
        false
    };
    let pos3_fail = filter_end_deriv_8bpc_neon(
        buf,
        s,
        s + stride,
        s + 3 * stride,
        t,
        t + stride,
        t + 3 * stride,
        -2,
        3,
        -1,
    ) > end_thr;
    let fail2 = high_deriv > (side_thr >> 3) || transition > q_thr * 3 || neg3_fail || pos3_fail;

    if MAX_WIDTH_POS == 3 {
        let mut width = 3;
        width = select_i32(fail2, 2, width);
        width = select_i32(fail1, 1, width);
        return select_i32(fail0, 0, width);
    }

    let transition4 = transition << 4;
    let mut fail4 = false;
    let mut fail6 = false;
    let mut fail8 = false;

    if MAX_WIDTH_POS >= 4 {
        let dist = 4i32;
        let dist2 = 4i32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        let neg_fail = if MAX_WIDTH_NEG >= dist2 {
            filter_end_deriv_8bpc_neon(
                buf,
                s - stride,
                s - (dist2 as isize + 1) * stride,
                s - 2 * stride,
                t - stride,
                t - (dist2 as isize + 1) * stride,
                t - 2 * stride,
                (1 - dist2) as i16,
                -1,
                dist2 as i16,
            ) > end_thr4
        } else {
            false
        };
        let pos_fail = filter_end_deriv_8bpc_neon(
            buf,
            s,
            s + dist2 as isize * stride,
            s + stride,
            t,
            t + dist2 as isize * stride,
            t + stride,
            (1 - dist2) as i16,
            -1,
            dist2 as i16,
        ) > end_thr4;
        fail4 = transition4 > q_thr * crate::deblock::Q_FIRST[0] as u32 || neg_fail || pos_fail;
    }

    if MAX_WIDTH_POS >= 6 {
        let dist = 6i32;
        let dist2 = 6i32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        let neg_fail = if MAX_WIDTH_NEG >= dist2 {
            filter_end_deriv_8bpc_neon(
                buf,
                s - stride,
                s - (dist2 as isize + 1) * stride,
                s - 2 * stride,
                t - stride,
                t - (dist2 as isize + 1) * stride,
                t - 2 * stride,
                (1 - dist2) as i16,
                -1,
                dist2 as i16,
            ) > end_thr4
        } else {
            false
        };
        let pos_fail = filter_end_deriv_8bpc_neon(
            buf,
            s,
            s + dist2 as isize * stride,
            s + stride,
            t,
            t + dist2 as isize * stride,
            t + stride,
            (1 - dist2) as i16,
            -1,
            dist2 as i16,
        ) > end_thr4;
        fail6 = transition4 > q_thr * crate::deblock::Q_FIRST[1] as u32 || neg_fail || pos_fail;
    }

    if MAX_WIDTH_POS >= 8 {
        let dist = 8i32;
        let dist2 = 7i32;
        let end_thr4 = (side_thr * dist as u32) >> 4;
        let neg_fail = if MAX_WIDTH_NEG >= dist2 {
            filter_end_deriv_8bpc_neon(
                buf,
                s - stride,
                s - (dist2 as isize + 1) * stride,
                s - 2 * stride,
                t - stride,
                t - (dist2 as isize + 1) * stride,
                t - 2 * stride,
                (1 - dist2) as i16,
                -1,
                dist2 as i16,
            ) > end_thr4
        } else {
            false
        };
        let pos_fail = filter_end_deriv_8bpc_neon(
            buf,
            s,
            s + dist2 as isize * stride,
            s + stride,
            t,
            t + dist2 as isize * stride,
            t + stride,
            (1 - dist2) as i16,
            -1,
            dist2 as i16,
        ) > end_thr4;
        fail8 = transition4 > q_thr * crate::deblock::Q_FIRST[2] as u32 || neg_fail || pos_fail;
    }

    let mut width = MAX_WIDTH_POS;
    width = select_i32(MAX_WIDTH_POS >= 8 && fail8, 6, width);
    width = select_i32(MAX_WIDTH_POS >= 6 && fail6, 4, width);
    width = select_i32(MAX_WIDTH_POS >= 4 && fail4, 3, width);
    width = select_i32(fail2, 2, width);
    width = select_i32(fail1, 1, width);
    select_i32(fail0, 0, width)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_apply_8bpc_neon_width_constmax<const MAX_WIDTH_NEG: i32, const CONTIG: bool>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    debug_assert!((1..=8).contains(&MAX_WIDTH_NEG));
    debug_assert!(q_thr_clamp <= i16::MAX as i32);

    match width {
        1 => deblock_apply_8bpc_neon_const_oriented::<1, 1, CONTIG>(
            dst,
            off,
            stride_line,
            stride_tap,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        2 => {
            if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_neon_const_oriented::<2, 2, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_neon_const_oriented::<1, 2, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        3 => {
            if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_neon_const_oriented::<3, 3, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_neon_const_oriented::<2, 3, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_neon_const_oriented::<1, 3, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        4 => {
            if MAX_WIDTH_NEG >= 4 {
                deblock_apply_8bpc_neon_const_oriented::<4, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_neon_const_oriented::<3, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_neon_const_oriented::<2, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_neon_const_oriented::<1, 4, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        6 => {
            if MAX_WIDTH_NEG >= 6 {
                deblock_apply_8bpc_neon_const_oriented::<6, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 4 {
                deblock_apply_8bpc_neon_const_oriented::<4, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_neon_const_oriented::<3, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_neon_const_oriented::<2, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_neon_const_oriented::<1, 6, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        8 => {
            if MAX_WIDTH_NEG >= 8 {
                deblock_apply_8bpc_neon_const_oriented::<8, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 6 {
                deblock_apply_8bpc_neon_const_oriented::<6, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 4 {
                deblock_apply_8bpc_neon_const_oriented::<4, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 3 {
                deblock_apply_8bpc_neon_const_oriented::<3, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else if MAX_WIDTH_NEG >= 2 {
                deblock_apply_8bpc_neon_const_oriented::<2, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            } else {
                deblock_apply_8bpc_neon_const_oriented::<1, 8, CONTIG>(
                    dst,
                    off,
                    stride_line,
                    stride_tap,
                    q_thr_clamp,
                    neg_lossless,
                    pos_lossless,
                )
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_8bpc_neon_const_max<
    const MAX_WIDTH_NEG: i32,
    const MAX_WIDTH_POS: i32,
    const CONTIG: bool,
>(
    dst: &mut [u8],
    off: isize,
    q_thr: u32,
    side_thr: u32,
    stridea: isize,
    strideb: isize,
    pos_lossless: bool,
    neg_lossless: bool,
) {
    debug_assert!((1..=8).contains(&MAX_WIDTH_POS));
    debug_assert!((1..=8).contains(&MAX_WIDTH_NEG));
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    let width = filter_choice_8bpc_neon_const::<MAX_WIDTH_NEG, MAX_WIDTH_POS>(
        dst,
        off,
        off + 3 * stridea,
        strideb,
        q_thr,
        side_thr,
    );
    if width < 1 || (neg_lossless && pos_lossless) {
        return;
    }

    let q_thr_clamp = q_thr as i32 * crate::deblock::Q_THRESH_MULTS[(width - 1) as usize] as i32;
    if q_thr_clamp <= 0 {
        return;
    }

    if q_thr_clamp > i16::MAX as i32 {
        deblock_apply_8bpc_neon(
            dst,
            off,
            stridea,
            strideb,
            width.min(MAX_WIDTH_NEG),
            width,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
        return;
    }

    deblock_apply_8bpc_neon_width_constmax::<MAX_WIDTH_NEG, CONTIG>(
        dst,
        off,
        stridea,
        strideb,
        width,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn try_deblock_h_sb64_w8_run4_transpose(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    qi: usize,
    vm: u32,
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
) -> bool {
    debug_assert!(qi + 3 < 16);
    let run = 0x0fu32 << qi;
    if (vm & run) != run {
        return false;
    }

    let mut i = 0;
    while i < 4 {
        let bit = 1u16 << (qi + i);
        let q = q_thr[qi + i] as u32;
        if q == 0 || ((ll_mask[0] & bit) != 0 && (ll_mask[1] & bit) != 0) {
            return false;
        }
        let off = (dst_off + (qi + i) * 4 * stride) as isize;
        let width = filter_choice_8bpc_neon_const::<8, 8>(
            dst,
            off,
            off + 3 * stride as isize,
            1,
            q,
            side_thr[qi + i] as u32,
        );
        if width != 8 {
            return false;
        }
        i += 1;
    }

    let (qlo, qhi) = repeated_qclamp4_w8(q_thr, qi);
    let (neg_mask_lo, neg_mask_hi) = repeated_apply_mask4(ll_mask[0], qi);
    let (pos_mask_lo, pos_mask_hi) = repeated_apply_mask4(ll_mask[1], qi);
    let off = (dst_off + qi * 4 * stride) as isize;
    deblock_apply_8bpc_neon_h_w8x4_transpose(
        dst,
        off,
        stride as isize,
        qlo,
        qhi,
        neg_mask_lo,
        neg_mask_hi,
        pos_mask_lo,
        pos_mask_hi,
    );
    true
}

#[inline]
#[target_feature(enable = "rdm")]
fn deblock_mask_class_bits(mask: u16, higher: u16, both_lossless: u16) -> u32 {
    (mask & !higher & !both_lossless) as u32
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_sb64_8bpc_neon_mask<
    const MAX_WIDTH_NEG: i32,
    const MAX_WIDTH_POS: i32,
    const HORIZONTAL: bool,
    const CONTIG: bool,
>(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    mut vm: u32,
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
) {
    debug_assert!(MAX_WIDTH_NEG <= MAX_WIDTH_POS);

    if HORIZONTAL && !CONTIG && MAX_WIDTH_NEG == 8 && MAX_WIDTH_POS == 8 {
        let mut qi = 0usize;
        while qi <= 12 {
            let run = 0x0fu32 << qi;
            if (vm & run) == run
                && try_deblock_h_sb64_w8_run4_transpose(
                    dst, dst_off, stride, qi, vm, ll_mask, q_thr, side_thr,
                )
            {
                vm &= !run;
            }
            qi += 4;
        }
    }

    while vm != 0 {
        let qi = vm.trailing_zeros() as usize;
        let bit = 1u32 << qi;
        let q = q_thr[qi] as u32;
        if q != 0 {
            let pos_ll = (ll_mask[1] as u32 & bit) != 0;
            let neg_ll = (ll_mask[0] as u32 & bit) != 0;
            if !(pos_ll && neg_ll) {
                let side = side_thr[qi] as u32;
                let off = if HORIZONTAL {
                    (dst_off + qi * 4 * stride) as isize
                } else {
                    (dst_off + qi * 4) as isize
                };
                let stridea = if HORIZONTAL { stride as isize } else { 1 };
                let strideb = if HORIZONTAL { 1 } else { stride as isize };
                deblock_8bpc_neon_const_max::<MAX_WIDTH_NEG, MAX_WIDTH_POS, CONTIG>(
                    dst, off, q, side, stridea, strideb, pos_ll, neg_ll,
                );
            }
        }
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_h_sb64y_8bpc_neon(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m3 = deblock_mask_class_bits(vmask[3], 0, both_lossless);
    let m2 = deblock_mask_class_bits(vmask[2], vmask[3], both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2] | vmask[3], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2] | vmask[3], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_neon_mask::<1, 1, true, false>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        deblock_sb64_8bpc_neon_mask::<3, 3, true, false>(
            dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
        );
    }
    if m2 != 0 {
        deblock_sb64_8bpc_neon_mask::<6, 6, true, false>(
            dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
        );
    }
    if m3 != 0 {
        if edge {
            deblock_sb64_8bpc_neon_mask::<6, 8, true, false>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_neon_mask::<8, 8, true, false>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_v_sb64y_8bpc_neon(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m3 = deblock_mask_class_bits(vmask[3], 0, both_lossless);
    let m2 = deblock_mask_class_bits(vmask[2], vmask[3], both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2] | vmask[3], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2] | vmask[3], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_neon_mask::<1, 1, false, true>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        deblock_sb64_8bpc_neon_mask::<3, 3, false, true>(
            dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
        );
    }
    if m2 != 0 {
        deblock_sb64_8bpc_neon_mask::<6, 6, false, true>(
            dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
        );
    }
    if m3 != 0 {
        if edge {
            deblock_sb64_8bpc_neon_mask::<6, 8, false, true>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_neon_mask::<8, 8, false, true>(
                dst, dst_off, stride, m3, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_h_sb64uv_8bpc_neon(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m2 = deblock_mask_class_bits(vmask[2], 0, both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_neon_mask::<1, 1, true, false>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        if edge {
            deblock_sb64_8bpc_neon_mask::<2, 3, true, false>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_neon_mask::<3, 3, true, false>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        }
    }
    if m2 != 0 {
        if edge {
            deblock_sb64_8bpc_neon_mask::<2, 4, true, false>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_neon_mask::<4, 4, true, false>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_v_sb64uv_8bpc_neon(
    dst: &mut [u8],
    dst_off: usize,
    stride: usize,
    vmask: &[u16],
    ll_mask: &[u16],
    q_thr: &[u8],
    side_thr: &[u8],
    edge: bool,
) {
    let both_lossless = ll_mask[0] & ll_mask[1];
    let m2 = deblock_mask_class_bits(vmask[2], 0, both_lossless);
    let m1 = deblock_mask_class_bits(vmask[1], vmask[2], both_lossless);
    let m0 = deblock_mask_class_bits(vmask[0], vmask[1] | vmask[2], both_lossless);

    if m0 != 0 {
        deblock_sb64_8bpc_neon_mask::<1, 1, false, true>(
            dst, dst_off, stride, m0, ll_mask, q_thr, side_thr,
        );
    }
    if m1 != 0 {
        if edge {
            deblock_sb64_8bpc_neon_mask::<2, 3, false, true>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_neon_mask::<3, 3, false, true>(
                dst, dst_off, stride, m1, ll_mask, q_thr, side_thr,
            );
        }
    }
    if m2 != 0 {
        if edge {
            deblock_sb64_8bpc_neon_mask::<2, 4, false, true>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        } else {
            deblock_sb64_8bpc_neon_mask::<4, 4, false, true>(
                dst, dst_off, stride, m2, ll_mask, q_thr, side_thr,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
fn deblock_sb64_hbd_neon_mask<
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
                    deblock_apply_hbd_neon,
                );
            }
        }
        vm &= vm - 1;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_h_sb64y_hbd_neon(
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
        deblock_sb64_hbd_neon_mask::<1, 1, true>(
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
        deblock_sb64_hbd_neon_mask::<3, 3, true>(
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
        deblock_sb64_hbd_neon_mask::<6, 6, true>(
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
            deblock_sb64_hbd_neon_mask::<6, 8, true>(
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
            deblock_sb64_hbd_neon_mask::<8, 8, true>(
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
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_v_sb64y_hbd_neon(
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
        deblock_sb64_hbd_neon_mask::<1, 1, false>(
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
        deblock_sb64_hbd_neon_mask::<3, 3, false>(
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
        deblock_sb64_hbd_neon_mask::<6, 6, false>(
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
            deblock_sb64_hbd_neon_mask::<6, 8, false>(
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
            deblock_sb64_hbd_neon_mask::<8, 8, false>(
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
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_h_sb64uv_hbd_neon(
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
        deblock_sb64_hbd_neon_mask::<1, 1, true>(
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
            deblock_sb64_hbd_neon_mask::<2, 3, true>(
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
            deblock_sb64_hbd_neon_mask::<3, 3, true>(
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
            deblock_sb64_hbd_neon_mask::<2, 4, true>(
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
            deblock_sb64_hbd_neon_mask::<4, 4, true>(
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
#[target_feature(enable = "rdm")]
pub(crate) fn deblock_v_sb64uv_hbd_neon(
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
        deblock_sb64_hbd_neon_mask::<1, 1, false>(
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
            deblock_sb64_hbd_neon_mask::<2, 3, false>(
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
            deblock_sb64_hbd_neon_mask::<3, 3, false>(
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
            deblock_sb64_hbd_neon_mask::<2, 4, false>(
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
            deblock_sb64_hbd_neon_mask::<4, 4, false>(
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
fn setup_lut_u8x16_neon(lut: &[u32; 16]) -> uint8x16_t {
    let mut tbl = [0u8; 16];
    for i in 0..16 {
        tbl[i] = lut[i] as u8;
    }
    unsafe { vld1q_u8(tbl.as_ptr()) }
}

#[inline]
fn setup_load_seg_u8x16_neon(seg: &[u8], off: usize, w: usize) -> uint8x16_t {
    let mut tmp = [0u8; 16];
    tmp[..w].copy_from_slice(&seg[off..off + w]);
    unsafe { vld1q_u8(tmp.as_ptr()) }
}

#[inline]
fn setup_mask_bits_u8x16_neon(bits: u16) -> uint8x16_t {
    let tmp = [
        if bits & (1 << 0) != 0 { 0xff } else { 0 },
        if bits & (1 << 1) != 0 { 0xff } else { 0 },
        if bits & (1 << 2) != 0 { 0xff } else { 0 },
        if bits & (1 << 3) != 0 { 0xff } else { 0 },
        if bits & (1 << 4) != 0 { 0xff } else { 0 },
        if bits & (1 << 5) != 0 { 0xff } else { 0 },
        if bits & (1 << 6) != 0 { 0xff } else { 0 },
        if bits & (1 << 7) != 0 { 0xff } else { 0 },
        if bits & (1 << 8) != 0 { 0xff } else { 0 },
        if bits & (1 << 9) != 0 { 0xff } else { 0 },
        if bits & (1 << 10) != 0 { 0xff } else { 0 },
        if bits & (1 << 11) != 0 { 0xff } else { 0 },
        if bits & (1 << 12) != 0 { 0xff } else { 0 },
        if bits & (1 << 13) != 0 { 0xff } else { 0 },
        if bits & (1 << 14) != 0 { 0xff } else { 0 },
        if bits & (1 << 15) != 0 { 0xff } else { 0 },
    ];
    unsafe { vld1q_u8(tmp.as_ptr()) }
}

#[inline]
fn setup_apply_subpu_u8x16_neon(v: uint8x16_t, bits: u16) -> uint8x16_t {
    unsafe { vbslq_u8(setup_mask_bits_u8x16_neon(bits), vshrq_n_u8::<3>(v), v) }
}

#[inline]
fn setup_edge_u8x16_neon(cur: uint8x16_t, prev: uint8x16_t) -> uint8x16_t {
    unsafe {
        let z = vdupq_n_u8(0);
        let both = vmvnq_u8(vorrq_u8(vceqq_u8(cur, z), vceqq_u8(prev, z)));
        vbslq_u8(both, vrhaddq_u8(cur, prev), vorrq_u8(cur, prev))
    }
}

#[inline]
fn setup_store_u8x16_neon(dst: &mut [u8; 256], off: usize, v: uint8x16_t) {
    unsafe { vst1q_u8(dst.as_mut_ptr().add(off), v) };
}

#[inline]
fn setup_store_tmp_u8x16_neon(v: uint8x16_t) -> [u8; 16] {
    let mut tmp = [0u8; 16];
    unsafe { vst1q_u8(tmp.as_mut_ptr(), v) };
    tmp
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn setup_thr_rows_simple_8bpc_neon(
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
    unsafe {
        let qv = vdupq_n_u8(thr_lut[0][0] as u8);
        let sv = vdupq_n_u8(thr_lut[1][0] as u8);
        for y in 0..h {
            let bits = (mask[starty4 + y][4][mask_idx] >> mask_shift) as u16;
            setup_store_u8x16_neon(q_thr_dst, y * 16, setup_apply_subpu_u8x16_neon(qv, bits));
            setup_store_u8x16_neon(side_thr_dst, y * 16, setup_apply_subpu_u8x16_neon(sv, bits));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn setup_thr_cols_simple_8bpc_neon(
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
    unsafe {
        let qv = vdupq_n_u8(thr_lut[0][0] as u8);
        let sv = vdupq_n_u8(thr_lut[1][0] as u8);
        for x in 0..w {
            let bits = (mask[bx4_base + x][4][mask_idx] >> mask_shift) as u16;
            setup_store_u8x16_neon(q_thr_dst, x * 16, setup_apply_subpu_u8x16_neon(qv, bits));
            setup_store_u8x16_neon(side_thr_dst, x * 16, setup_apply_subpu_u8x16_neon(sv, bits));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn setup_thr_rows_dq_8bpc_neon(
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
    unsafe {
        let qv = vdupq_n_u8(thr_lut[0][0] as u8);
        let sv = vdupq_n_u8(thr_lut[1][0] as u8);
        let (above_q, above_s) = if let Some(alut) = above_thr_lut {
            if let Some((aseg, aoff)) = above_seg {
                let off = usize::try_from(aoff).expect("negative above segment offset");
                assert!(off + w <= aseg.len());
                let segv = setup_load_seg_u8x16_neon(aseg, off, w);
                (
                    vqtbl1q_u8(setup_lut_u8x16_neon(&alut[0]), segv),
                    vqtbl1q_u8(setup_lut_u8x16_neon(&alut[1]), segv),
                )
            } else {
                (vdupq_n_u8(alut[0][0] as u8), vdupq_n_u8(alut[1][0] as u8))
            }
        } else {
            (vdupq_n_u8(0), vdupq_n_u8(0))
        };
        let bits0 = (mask[starty4][4][mask_idx] >> mask_shift) as u16;
        setup_store_u8x16_neon(
            q_thr_dst,
            0,
            setup_apply_subpu_u8x16_neon(setup_edge_u8x16_neon(qv, above_q), bits0),
        );
        setup_store_u8x16_neon(
            side_thr_dst,
            0,
            setup_apply_subpu_u8x16_neon(setup_edge_u8x16_neon(sv, above_s), bits0),
        );
        for y in 1..h {
            let bits = (mask[starty4 + y][4][mask_idx] >> mask_shift) as u16;
            setup_store_u8x16_neon(q_thr_dst, y * 16, setup_apply_subpu_u8x16_neon(qv, bits));
            setup_store_u8x16_neon(side_thr_dst, y * 16, setup_apply_subpu_u8x16_neon(sv, bits));
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn setup_thr_cols_dq_8bpc_neon(
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
    unsafe {
        let qv = vdupq_n_u8(thr_lut[0][0] as u8);
        let sv = vdupq_n_u8(thr_lut[1][0] as u8);
        let left_q = vld1q_u8(left_q_thr.as_ptr());
        let left_s = vld1q_u8(left_side_thr.as_ptr());
        for x in 0..w {
            let bits = (mask[bx4_base + x][4][mask_idx] >> mask_shift) as u16;
            let qbase = if x == 0 {
                setup_edge_u8x16_neon(qv, left_q)
            } else {
                qv
            };
            let sbase = if x == 0 {
                setup_edge_u8x16_neon(sv, left_s)
            } else {
                sv
            };
            setup_store_u8x16_neon(q_thr_dst, x * 16, setup_apply_subpu_u8x16_neon(qbase, bits));
            setup_store_u8x16_neon(
                side_thr_dst,
                x * 16,
                setup_apply_subpu_u8x16_neon(sbase, bits),
            );
        }
    }
    left_q_thr[..h].fill(thr_lut[0][0] as u8);
    left_side_thr[..h].fill(thr_lut[1][0] as u8);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn setup_thr_rows_seg_8bpc_neon(
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
    unsafe {
        let qlut = setup_lut_u8x16_neon(&thr_lut[0]);
        let slut = setup_lut_u8x16_neon(&thr_lut[1]);
        let (mut prev_q, mut prev_s) =
            if let (Some(alut), Some((aseg, aoff))) = (above_thr_lut, above_seg) {
                let off = usize::try_from(aoff).expect("negative above segment offset");
                assert!(off + w <= aseg.len());
                let segv = setup_load_seg_u8x16_neon(aseg, off, w);
                (
                    vqtbl1q_u8(setup_lut_u8x16_neon(&alut[0]), segv),
                    vqtbl1q_u8(setup_lut_u8x16_neon(&alut[1]), segv),
                )
            } else {
                (vdupq_n_u8(0), vdupq_n_u8(0))
            };
        for y in 0..h {
            let row = seg_off + y * seg_stride;
            let segv = setup_load_seg_u8x16_neon(segmap, row, w);
            let cur_q = vqtbl1q_u8(qlut, segv);
            let cur_s = vqtbl1q_u8(slut, segv);
            let bits = (mask[starty4 + y][4][mask_idx] >> mask_shift) as u16;
            setup_store_u8x16_neon(
                q_thr_dst,
                y * 16,
                setup_apply_subpu_u8x16_neon(setup_edge_u8x16_neon(cur_q, prev_q), bits),
            );
            setup_store_u8x16_neon(
                side_thr_dst,
                y * 16,
                setup_apply_subpu_u8x16_neon(setup_edge_u8x16_neon(cur_s, prev_s), bits),
            );
            prev_q = cur_q;
            prev_s = cur_s;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn setup_thr_cols_seg_8bpc_neon(
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
    unsafe {
        let qlut = setup_lut_u8x16_neon(&thr_lut[0]);
        let slut = setup_lut_u8x16_neon(&thr_lut[1]);
        for y in 0..h {
            let row = seg_off + y * seg_stride;
            let segv = setup_load_seg_u8x16_neon(segmap, row, w);
            let cur_q = vqtbl1q_u8(qlut, segv);
            let cur_s = vqtbl1q_u8(slut, segv);
            let cur_q_arr = setup_store_tmp_u8x16_neon(cur_q);
            let cur_s_arr = setup_store_tmp_u8x16_neon(cur_s);
            let mut prev_q_arr = [0u8; 16];
            let mut prev_s_arr = [0u8; 16];
            prev_q_arr[0] = left_q_thr[y];
            prev_s_arr[0] = left_side_thr[y];
            prev_q_arr[1..].copy_from_slice(&cur_q_arr[..15]);
            prev_s_arr[1..].copy_from_slice(&cur_s_arr[..15]);
            let prev_q = vld1q_u8(prev_q_arr.as_ptr());
            let prev_s = vld1q_u8(prev_s_arr.as_ptr());
            let mut bits = 0u16;
            let shift = mask_shift + y as u32;
            for x in 0..w {
                bits |= ((mask[bx4_base + x][4][mask_idx] >> shift) & 1) << x;
            }
            let q_arr = setup_store_tmp_u8x16_neon(setup_apply_subpu_u8x16_neon(
                setup_edge_u8x16_neon(cur_q, prev_q),
                bits,
            ));
            let s_arr = setup_store_tmp_u8x16_neon(setup_apply_subpu_u8x16_neon(
                setup_edge_u8x16_neon(cur_s, prev_s),
                bits,
            ));
            for x in 0..w {
                q_thr_dst[x * 16 + y] = q_arr[x];
                side_thr_dst[x * 16 + y] = s_arr[x];
            }
            left_q_thr[y] = cur_q_arr[w - 1];
            left_side_thr[y] = cur_s_arr[w - 1];
        }
    }
}
