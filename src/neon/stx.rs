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
#[target_feature(enable = "neon")]
fn mac8(
    mut acc_lo: int32x4_t,
    mut acc_hi: int32x4_t,
    row0: *const i8,
    row1: *const i8,
    c0: i16,
    c1: i16,
) -> (int32x4_t, int32x4_t) {
    let k0 = unsafe { vmovl_s8(vld1_s8(row0)) };
    let k1 = unsafe { vmovl_s8(vld1_s8(row1)) };
    let c0v = vdup_n_s16(c0);
    let c1v = vdup_n_s16(c1);

    acc_lo = vmlal_s16(acc_lo, vget_low_s16(k0), c0v);
    acc_hi = vmlal_s16(acc_hi, vget_high_s16(k0), c0v);
    acc_lo = vmlal_s16(acc_lo, vget_low_s16(k1), c1v);
    acc_hi = vmlal_s16(acc_hi, vget_high_s16(k1), c1v);
    (acc_lo, acc_hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn round_pack_8(acc_lo: int32x4_t, acc_hi: int32x4_t) -> int16x8_t {
    let neg1 = vdupq_n_s32(-1);
    let adj_lo = vreinterpretq_s32_u32(vcgtq_s32(acc_lo, neg1));
    let adj_hi = vreinterpretq_s32_u32(vcgtq_s32(acc_hi, neg1));
    let lo = vshrq_n_s32::<7>(vsubq_s32(acc_lo, adj_lo));
    let hi = vshrq_n_s32::<7>(vsubq_s32(acc_hi, adj_hi));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn stx4_sums(kernel: &[i8], cf: &[i16], eob: usize) -> (int16x8_t, int16x8_t) {
    let mut acc0_lo = vdupq_n_s32(63);
    let mut acc0_hi = acc0_lo;
    let mut acc1_lo = acc0_lo;
    let mut acc1_hi = acc0_lo;

    let mut y = 0usize;
    while y <= eob {
        let c0 = unsafe { *cf.get_unchecked(y) };
        let c1 = if y < eob {
            unsafe { *cf.get_unchecked(y + 1) }
        } else {
            0
        };
        let row0 = unsafe { kernel.as_ptr().add(y * 16) };
        let row1 = unsafe { kernel.as_ptr().add((y + 1) * 16) };
        (acc0_lo, acc0_hi) = mac8(acc0_lo, acc0_hi, row0, row1, c0, c1);
        (acc1_lo, acc1_hi) = mac8(
            acc1_lo,
            acc1_hi,
            unsafe { row0.add(8) },
            unsafe { row1.add(8) },
            c0,
            c1,
        );
        y += 2;
    }

    (
        round_pack_8(acc0_lo, acc0_hi),
        round_pack_8(acc1_lo, acc1_hi),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn stx8_sums(
    kernel: &[i8],
    cf: &[i16],
    eob: usize,
) -> (
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
    int16x8_t,
) {
    let mut acc0_lo = vdupq_n_s32(63);
    let mut acc0_hi = acc0_lo;
    let mut acc1_lo = acc0_lo;
    let mut acc1_hi = acc0_lo;
    let mut acc2_lo = acc0_lo;
    let mut acc2_hi = acc0_lo;
    let mut acc3_lo = acc0_lo;
    let mut acc3_hi = acc0_lo;
    let mut acc4_lo = acc0_lo;
    let mut acc4_hi = acc0_lo;
    let mut acc5_lo = acc0_lo;
    let mut acc5_hi = acc0_lo;

    let mut y = 0usize;
    while y <= eob {
        let c0 = unsafe { *cf.get_unchecked(y) };
        let c1 = if y < eob {
            unsafe { *cf.get_unchecked(y + 1) }
        } else {
            0
        };
        let row0 = unsafe { kernel.as_ptr().add(y * 48) };
        let row1 = unsafe { kernel.as_ptr().add((y + 1) * 48) };

        (acc0_lo, acc0_hi) = mac8(acc0_lo, acc0_hi, row0, row1, c0, c1);
        (acc1_lo, acc1_hi) = mac8(
            acc1_lo,
            acc1_hi,
            unsafe { row0.add(8) },
            unsafe { row1.add(8) },
            c0,
            c1,
        );
        (acc2_lo, acc2_hi) = mac8(
            acc2_lo,
            acc2_hi,
            unsafe { row0.add(16) },
            unsafe { row1.add(16) },
            c0,
            c1,
        );
        (acc3_lo, acc3_hi) = mac8(
            acc3_lo,
            acc3_hi,
            unsafe { row0.add(24) },
            unsafe { row1.add(24) },
            c0,
            c1,
        );
        (acc4_lo, acc4_hi) = mac8(
            acc4_lo,
            acc4_hi,
            unsafe { row0.add(32) },
            unsafe { row1.add(32) },
            c0,
            c1,
        );
        (acc5_lo, acc5_hi) = mac8(
            acc5_lo,
            acc5_hi,
            unsafe { row0.add(40) },
            unsafe { row1.add(40) },
            c0,
            c1,
        );

        y += 2;
    }

    (
        round_pack_8(acc0_lo, acc0_hi),
        round_pack_8(acc1_lo, acc1_hi),
        round_pack_8(acc2_lo, acc2_hi),
        round_pack_8(acc3_lo, acc3_hi),
        round_pack_8(acc4_lo, acc4_hi),
        round_pack_8(acc5_lo, acc5_hi),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i16x8(dst: &mut [i16], v: int16x8_t) {
    unsafe { vst1q_s16(dst.as_mut_ptr(), v) };
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
#[target_feature(enable = "neon")]
fn zero_stx8_i16_neon(cf: &mut [i16]) {
    let zero = vdupq_n_s16(0);
    let dst = cf.as_mut_ptr();
    unsafe {
        vst1q_s16(dst, zero);
        vst1q_s16(dst.add(8), zero);
        vst1q_s16(dst.add(16), zero);
        vst1q_s16(dst.add(24), zero);
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn stxfm4_8bpc_neon(cf: &mut [i16], kernel: &[i8], eob: usize, scan_out: &[u8; 16]) {
    debug_assert!(eob < 8);
    debug_assert!(kernel.len() >= 8 * 16);

    let (s0, s1) = stx4_sums(kernel, cf, eob);
    let mut sums = [0i16; 16];
    store_i16x8(&mut sums[..8], s0);
    store_i16x8(&mut sums[8..16], s1);

    cf[4..8].fill(0);
    scatter_stx4_i16(cf, &sums, scan_out);
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn stxfm8_8bpc_neon(
    cf: &mut [i16],
    kernel: &[i8],
    eob: usize,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    debug_assert!(eob < 32);
    debug_assert!(kernel.len() >= 32 * 48);

    let (s0, s1, s2, s3, s4, s5) = stx8_sums(kernel, cf, eob);
    let mut sums = [0i16; 48];
    store_i16x8(&mut sums[..8], s0);
    store_i16x8(&mut sums[8..16], s1);
    store_i16x8(&mut sums[16..24], s2);
    store_i16x8(&mut sums[24..32], s3);
    store_i16x8(&mut sums[32..40], s4);
    store_i16x8(&mut sums[40..48], s5);
    zero_stx8_i16_neon(cf);
    scatter_stx8_i16(cf, &sums, scan_out, mapping);
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i8x4_i32(ptr: *const i8) -> int32x4_t {
    // Load exactly four kernel bytes.  Do not use vld1_s8 here: the final
    // 4-wide chunk of a 16/48-wide row would otherwise read past the table.
    let raw = unsafe { vreinterpret_s8_u32(vld1_lane_u32::<0>(ptr.cast(), vdup_n_u32(0))) };
    vmovl_s16(vget_low_s16(vmovl_s8(raw)))
}

#[inline]
#[target_feature(enable = "neon")]
fn mac_hbd_4(acc: int32x4_t, coeff: i32, kernel: *const i8) -> int32x4_t {
    let k = load_i8x4_i32(kernel);
    let c = vdupq_n_s32(coeff);
    vmlaq_s32(acc, k, c)
}

#[inline]
#[target_feature(enable = "neon")]
fn round_clip_hbd_4(acc: int32x4_t, min_v: int32x4_t, max_v: int32x4_t) -> int32x4_t {
    // Same signed-bias rounding as the scalar STX path, then explicit clip to
    // [-128 * (1 + bitdepth_max), 128 * (1 + bitdepth_max) - 1].
    let adj = vreinterpretq_s32_u32(vcgtq_s32(acc, vdupq_n_s32(-1)));
    let v = vshrq_n_s32::<7>(vsubq_s32(acc, adj));
    vminq_s32(vmaxq_s32(v, min_v), max_v)
}

#[inline]
#[target_feature(enable = "neon")]
fn stx4_sums_hbd(
    kernel: &[i8],
    cf: &[i32],
    eob: usize,
    bitdepth_max: i32,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let min_v = vdupq_n_s32(-128 * (1 + bitdepth_max));
    let max_v = vdupq_n_s32(128 * (1 + bitdepth_max) - 1);
    let mut acc0 = vdupq_n_s32(63);
    let mut acc1 = acc0;
    let mut acc2 = acc0;
    let mut acc3 = acc0;

    let mut y = 0usize;
    while y <= eob {
        let c = unsafe { *cf.get_unchecked(y) };
        let row = unsafe { kernel.as_ptr().add(y * 16) };
        acc0 = mac_hbd_4(acc0, c, row);
        acc1 = mac_hbd_4(acc1, c, unsafe { row.add(4) });
        acc2 = mac_hbd_4(acc2, c, unsafe { row.add(8) });
        acc3 = mac_hbd_4(acc3, c, unsafe { row.add(12) });
        y += 1;
    }

    (
        round_clip_hbd_4(acc0, min_v, max_v),
        round_clip_hbd_4(acc1, min_v, max_v),
        round_clip_hbd_4(acc2, min_v, max_v),
        round_clip_hbd_4(acc3, min_v, max_v),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn stx8_sums_hbd(kernel: &[i8], cf: &[i32], eob: usize, bitdepth_max: i32) -> [int32x4_t; 12] {
    let min_v = vdupq_n_s32(-128 * (1 + bitdepth_max));
    let max_v = vdupq_n_s32(128 * (1 + bitdepth_max) - 1);
    let mut acc = [vdupq_n_s32(63); 12];

    let mut y = 0usize;
    while y <= eob {
        let c = unsafe { *cf.get_unchecked(y) };
        let row = unsafe { kernel.as_ptr().add(y * 48) };
        let mut x = 0usize;
        while x < 12 {
            acc[x] = mac_hbd_4(acc[x], c, unsafe { row.add(x * 4) });
            x += 1;
        }
        y += 1;
    }

    let mut x = 0usize;
    while x < 12 {
        acc[x] = round_clip_hbd_4(acc[x], min_v, max_v);
        x += 1;
    }
    acc
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4(dst: &mut [i32], v: int32x4_t) {
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) };
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
#[target_feature(enable = "neon")]
fn zero_stx8_i32_neon(cf: &mut [i32]) {
    let zero = vdupq_n_s32(0);
    let dst = cf.as_mut_ptr();
    unsafe {
        vst1q_s32(dst, zero);
        vst1q_s32(dst.add(4), zero);
        vst1q_s32(dst.add(8), zero);
        vst1q_s32(dst.add(12), zero);
        vst1q_s32(dst.add(16), zero);
        vst1q_s32(dst.add(20), zero);
        vst1q_s32(dst.add(24), zero);
        vst1q_s32(dst.add(28), zero);
    }
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn stxfm4_hbd_neon(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 16],
) {
    debug_assert!(eob < 8);
    debug_assert!(kernel.len() >= 8 * 16);

    let (s0, s1, s2, s3) = stx4_sums_hbd(kernel, cf, eob, bitdepth_max);
    let mut sums = [0i32; 16];
    store_i32x4(&mut sums[..4], s0);
    store_i32x4(&mut sums[4..8], s1);
    store_i32x4(&mut sums[8..12], s2);
    store_i32x4(&mut sums[12..16], s3);

    cf[4..8].fill(0);
    scatter_stx4_i32(cf, &sums, scan_out);
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn stxfm8_hbd_neon(
    cf: &mut [i32],
    kernel: &[i8],
    eob: usize,
    bitdepth_max: i32,
    scan_out: &[u8; 64],
    mapping: &[u8; 48],
) {
    debug_assert!(eob < 32);
    debug_assert!(kernel.len() >= 32 * 48);

    let s = stx8_sums_hbd(kernel, cf, eob, bitdepth_max);
    let mut sums = [0i32; 48];
    let mut x = 0usize;
    while x < 12 {
        store_i32x4(&mut sums[x * 4..x * 4 + 4], s[x]);
        x += 1;
    }
    zero_stx8_i32_neon(cf);
    scatter_stx8_i32(cf, &sums, scan_out, mapping);
}
