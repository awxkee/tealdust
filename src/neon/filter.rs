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

use crate::gdf_tables::{GDF_ALPHA, GDF_INTER_ERROR, GDF_INTRA_ERROR, GDF_WEIGHT};

static GDF_PREP_COORDS_8BPC: [[i8; 2]; 18] = [
    [6, 0],
    [5, 0],
    [4, 0],
    [3, 0],
    [2, 1],
    [2, 0],
    [2, -1],
    [1, 2],
    [1, 1],
    [1, 0],
    [1, -1],
    [1, -2],
    [0, 6],
    [0, 5],
    [0, 4],
    [0, 3],
    [0, 2],
    [0, 1],
];

#[inline(always)]
fn load_i16x4_i32(a: &[i16; 4]) -> int32x4_t {
    unsafe { vmovl_s16(vld1_s16(a.as_ptr())) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x4_i32(a: &[u8; 4]) -> int32x4_t {
    let dup = unsafe { vreinterpret_u8_u32(vld1_lane_u32::<0>(a.as_ptr().cast(), vdup_n_u32(0))) };
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(dup))))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8_i32x4(a: &[u8; 8]) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vmovl_u8(vld1_u8(a.as_ptr())) };
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16_i32x4(a: &[u8; 16]) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u8(a.as_ptr()) };
    let lo = vmovl_u8(vget_low_u8(v));
    let hi = vmovl_u8(vget_high_u8(v));
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo))),
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8_i32x2(a: &[i16; 8]) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vld1q_s16(a.as_ptr()) };
    (vmovl_s16(vget_low_s16(w)), vmovl_s16(vget_high_s16(w)))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8(a: &[i16; 8]) -> int16x8_t {
    unsafe { vld1q_s16(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_i16x8_const(
    a: int16x8_t,
    b: int16x8_t,
    w1: int16x4_t,
    w2: int16x4_t,
) -> (int32x4_t, int32x4_t) {
    (
        vmlal_s16(vmull_s16(vget_low_s16(a), w1), vget_low_s16(b), w2),
        vmlal_s16(vmull_s16(vget_high_s16(a), w1), vget_high_s16(b), w2),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_i16x8(a: int16x8_t, b: int16x8_t, w1: int16x8_t, w2: int16x8_t) -> (int32x4_t, int32x4_t) {
    (
        vmlal_s16(
            vmull_s16(vget_low_s16(a), vget_low_s16(w1)),
            vget_low_s16(b),
            vget_low_s16(w2),
        ),
        vmlal_s16(
            vmull_s16(vget_high_s16(a), vget_high_s16(w1)),
            vget_high_s16(b),
            vget_high_s16(w2),
        ),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x16_i32x4(a: &[i16; 16]) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    unsafe {
        let lo = vld1q_s16(a.as_ptr());
        let hi = vld1q_s16(a.as_ptr().add(8));
        (
            vmovl_s16(vget_low_s16(lo)),
            vmovl_s16(vget_high_s16(lo)),
            vmovl_s16(vget_low_s16(hi)),
            vmovl_s16(vget_high_s16(hi)),
        )
    }
}

#[inline(always)]
fn load_i32x4(a: &[i32; 4]) -> int32x4_t {
    unsafe { vld1q_s32(a.as_ptr()) }
}

#[inline(always)]
fn store_i32x4(a: &mut [i32; 4], v: int32x4_t) {
    unsafe { vst1q_s32(a.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4_u8(a: &mut [u8; 4], v: int32x4_t) {
    let u16x4 = vqmovun_s32(v);
    let u8x8 = vqmovn_u16(vcombine_u16(u16x4, u16x4));
    unsafe {
        vst1_lane_u32::<0>(a.as_mut_ptr().cast(), vreinterpret_u32_u8(u8x8));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x8_u8(a: &mut [u8; 8], lo: int32x4_t, hi: int32x4_t) {
    let u16x8 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
    unsafe { vst1_u8(a.as_mut_ptr(), vqmovn_u16(u16x8)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x16_u8(a: &mut [u8; 16], v0: int32x4_t, v1: int32x4_t, v2: int32x4_t, v3: int32x4_t) {
    let lo = vqmovn_u16(vcombine_u16(vqmovun_s32(v0), vqmovun_s32(v1)));
    let hi = vqmovn_u16(vcombine_u16(vqmovun_s32(v2), vqmovun_s32(v3)));
    unsafe { vst1q_u8(a.as_mut_ptr(), vcombine_u8(lo, hi)) };
}

#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> uint8x16_t {
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline(always)]
fn store_u8x16(a: &mut [u8; 16], v: uint8x16_t) {
    unsafe { vst1q_u8(a.as_mut_ptr(), v) };
}

#[inline(always)]
fn load_u8x8(a: &[u8; 8]) -> uint8x8_t {
    unsafe { vld1_u8(a.as_ptr()) }
}

#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: uint8x8_t) {
    unsafe { vst1_u8(a.as_mut_ptr(), v) };
}

#[inline(always)]
fn load_u8x8_i16(a: &[u8; 8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(a.as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x16_i16x2(a: &[u8; 16]) -> (int16x8_t, int16x8_t) {
    let v = unsafe { vld1q_u8(a.as_ptr()) };
    (
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))),
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn residual_add_row_8bpc_neon(
    dst: &mut [u8],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-shift);
    let f = |cv: int32x4_t| vshlq_s32(vaddq_s32(cv, rnd_v), nsh);
    let (d16, r16) = dst[..n].as_chunks_mut::<16>();
    let (cc16, _) = c[..n].as_chunks::<16>();
    for (d, cv) in d16.iter_mut().zip(cc16) {
        let c0 = f(load_i32x4((&cv[..4]).try_into().unwrap()));
        let c1 = f(load_i32x4((&cv[4..8]).try_into().unwrap()));
        let c2 = f(load_i32x4((&cv[8..12]).try_into().unwrap()));
        let c3 = f(load_i32x4((&cv[12..16]).try_into().unwrap()));
        let (d0, d1, d2, d3) = load_u8x16_i32x4(&*d);
        store_i32x16_u8(
            d,
            vaddq_s32(d0, c0),
            vaddq_s32(d1, c1),
            vaddq_s32(d2, c2),
            vaddq_s32(d3, c3),
        );
    }
    let done = d16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (cc8, _) = c[done..n].as_chunks::<8>();
    for (d, cv) in c8.iter_mut().zip(cc8) {
        let cf_lo = f(load_i32x4((&cv[..4]).try_into().unwrap()));
        let cf_hi = f(load_i32x4((&cv[4..]).try_into().unwrap()));
        let (d_lo, d_hi) = load_u8x8_i32x4(&*d);
        store_i32x8_u8(d, vaddq_s32(d_lo, cf_lo), vaddq_s32(d_hi, cf_hi));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (cc4, cr) = c[done..n].as_chunks::<4>();
    for (d, cv) in c4.iter_mut().zip(cc4) {
        let cf = f(load_i32x4(cv));
        let dv = load_u8x4_i32(d);
        store_i32x4_u8(d, vaddq_s32(dv, cf));
    }
    for (d, &cv) in r4.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, 255) as u8;
    }
}

/// `dst[i] = clip(dst[i] + dc, 0, 255)`.
#[target_feature(enable = "neon")]
pub(crate) fn dc_add_row_8bpc_neon(dst: &mut [u8], dc: i32, n: usize) {
    if dc == 0 {
        return;
    }

    let amt = if dc > 0 {
        dc.min(255) as u8
    } else {
        dc.saturating_neg().min(255) as u8
    };

    let (c64, r64) = dst[..n].as_chunks_mut::<64>();
    let (c16, r16) = r64.as_chunks_mut::<16>();
    let (c8, r8) = r16.as_chunks_mut::<8>();

    if dc > 0 {
        let amt16 = vdupq_n_u8(amt);
        for d in c64.iter_mut() {
            let (d01, d23) = d.split_at_mut(32);
            let (d0, d1) = d01.split_at_mut(16);
            let (d2, d3) = d23.split_at_mut(16);
            let d0: &mut [u8; 16] = d0.try_into().unwrap();
            let d1: &mut [u8; 16] = d1.try_into().unwrap();
            let d2: &mut [u8; 16] = d2.try_into().unwrap();
            let d3: &mut [u8; 16] = d3.try_into().unwrap();
            store_u8x16(d0, vqaddq_u8(load_u8x16(&*d0), amt16));
            store_u8x16(d1, vqaddq_u8(load_u8x16(&*d1), amt16));
            store_u8x16(d2, vqaddq_u8(load_u8x16(&*d2), amt16));
            store_u8x16(d3, vqaddq_u8(load_u8x16(&*d3), amt16));
        }

        for d in c16.iter_mut() {
            store_u8x16(d, vqaddq_u8(load_u8x16(&*d), amt16));
        }

        let amt8 = vdup_n_u8(amt);
        for d in c8.iter_mut() {
            store_u8x8(d, vqadd_u8(load_u8x8(&*d), amt8));
        }

        for d in r8.iter_mut() {
            *d = d.saturating_add(amt);
        }
    } else {
        let amt16 = vdupq_n_u8(amt);
        for d in c64.iter_mut() {
            let (d01, d23) = d.split_at_mut(32);
            let (d0, d1) = d01.split_at_mut(16);
            let (d2, d3) = d23.split_at_mut(16);
            let d0: &mut [u8; 16] = d0.try_into().unwrap();
            let d1: &mut [u8; 16] = d1.try_into().unwrap();
            let d2: &mut [u8; 16] = d2.try_into().unwrap();
            let d3: &mut [u8; 16] = d3.try_into().unwrap();
            store_u8x16(d0, vqsubq_u8(load_u8x16(&*d0), amt16));
            store_u8x16(d1, vqsubq_u8(load_u8x16(&*d1), amt16));
            store_u8x16(d2, vqsubq_u8(load_u8x16(&*d2), amt16));
            store_u8x16(d3, vqsubq_u8(load_u8x16(&*d3), amt16));
        }

        for d in c16.iter_mut() {
            store_u8x16(d, vqsubq_u8(load_u8x16(&*d), amt16));
        }

        let amt8 = vdup_n_u8(amt);
        for d in c8.iter_mut() {
            store_u8x8(d, vqsub_u8(load_u8x8(&*d), amt8));
        }

        for d in r8.iter_mut() {
            *d = d.saturating_sub(amt);
        }
    }
}

/// itx row-clip: `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)` (i32 in/out).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn row_clip_neon(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-shift);
    let min_v = vdupq_n_s32(min);
    let max_v = vdupq_n_s32(max);
    let clip =
        |v: int32x4_t| vminq_s32(vmaxq_s32(vshlq_s32(vaddq_s32(v, rnd_v), nsh), min_v), max_v);
    let (c32, r32) = tmp[..n].as_chunks_mut::<32>();
    macro_rules! clip8 {
        ($c:expr) => {{
            let (lo, hi) = $c.split_at_mut(4);
            let lo: &mut [i32; 4] = lo.try_into().unwrap();
            let hi: &mut [i32; 4] = hi.try_into().unwrap();
            let r_lo = clip(load_i32x4(&*lo));
            let r_hi = clip(load_i32x4(&*hi));
            store_i32x4(lo, r_lo);
            store_i32x4(hi, r_hi);
        }};
    }
    for ch in c32.iter_mut() {
        let (c01, c23) = ch.split_at_mut(16);
        let (c0, c1) = c01.split_at_mut(8);
        let (c2, c3) = c23.split_at_mut(8);
        clip8!(c0);
        clip8!(c1);
        clip8!(c2);
        clip8!(c3);
    }
    let (c8, r8) = r32.as_chunks_mut::<8>();
    for ch in c8.iter_mut() {
        let r_lo = clip(load_i32x4((&ch[..4]).try_into().unwrap()));
        let r_hi = clip(load_i32x4((&ch[4..]).try_into().unwrap()));
        store_i32x4((&mut ch[..4]).try_into().unwrap(), r_lo);
        store_i32x4((&mut ch[4..]).try_into().unwrap(), r_hi);
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for ch in c4.iter_mut() {
        let r = clip(load_i32x4(ch));
        store_i32x4(ch, r);
    }
    for t in r4.iter_mut() {
        *t = ((*t + rnd) >> shift).max(min).min(max);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn cctx_row_neon(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    let sina_v = vdupq_n_s32(sina);
    let cosa_v = vdupq_n_s32(cosa);
    let c128 = vdupq_n_s32(128);
    let zero = vdupq_n_s32(0);
    let min_v = vdupq_n_s32(min);
    let max_v = vdupq_n_s32(max);
    let rot = |uu: int32x4_t, vv: int32x4_t| -> (int32x4_t, int32x4_t) {
        let a = vsubq_s32(vmulq_s32(uu, cosa_v), vmulq_s32(vv, sina_v));
        let b = vaddq_s32(vmulq_s32(uu, sina_v), vmulq_s32(vv, cosa_v));
        let amask = vreinterpretq_s32_u32(vcltq_s32(a, zero));
        let bmask = vreinterpretq_s32_u32(vcltq_s32(b, zero));
        let ra = vshrq_n_s32::<8>(vaddq_s32(vaddq_s32(a, c128), amask));
        let rb = vshrq_n_s32::<8>(vaddq_s32(vaddq_s32(b, c128), bmask));
        (
            vminq_s32(vmaxq_s32(ra, min_v), max_v),
            vminq_s32(vmaxq_s32(rb, min_v), max_v),
        )
    };
    let (uc8, ur8) = u[..sz].as_chunks_mut::<8>();
    let (vc8, vr8) = v[..sz].as_chunks_mut::<8>();
    for (uch, vch) in uc8.iter_mut().zip(vc8.iter_mut()) {
        let u_lo = load_i32x4((&uch[..4]).try_into().unwrap());
        let u_hi = load_i32x4((&uch[4..]).try_into().unwrap());
        let v_lo = load_i32x4((&vch[..4]).try_into().unwrap());
        let v_hi = load_i32x4((&vch[4..]).try_into().unwrap());
        let (ra_lo, rb_lo) = rot(u_lo, v_lo);
        let (ra_hi, rb_hi) = rot(u_hi, v_hi);
        store_i32x4((&mut uch[..4]).try_into().unwrap(), ra_lo);
        store_i32x4((&mut uch[4..]).try_into().unwrap(), ra_hi);
        store_i32x4((&mut vch[..4]).try_into().unwrap(), rb_lo);
        store_i32x4((&mut vch[4..]).try_into().unwrap(), rb_hi);
    }
    let (uc4, ur4) = ur8.as_chunks_mut::<4>();
    let (vc4, vr4) = vr8.as_chunks_mut::<4>();
    for (uch, vch) in uc4.iter_mut().zip(vc4.iter_mut()) {
        let (ra, rb) = rot(load_i32x4(uch), load_i32x4(vch));
        store_i32x4(uch, ra);
        store_i32x4(vch, rb);
    }
    for (uu, vv) in ur4.iter_mut().zip(vr4.iter_mut()) {
        let a = *uu * cosa - *vv * sina;
        let b = *uu * sina + *vv * cosa;
        *uu = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        *vv = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
    }
}

/// `dst[x] = clip((t1[x] + t2[x] + rnd) >> sh, 0, 255)`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn avg_row_8bpc_neon(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-sh);
    let f = |a: int32x4_t, b: int32x4_t| vshlq_s32(vaddq_s32(vaddq_s32(a, b), rnd_v), nsh);
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in c16.iter_mut().zip(a16).zip(b16) {
        let (a0, a1, a2, a3) = load_i16x16_i32x4(a);
        let (b0, b1, b2, b3) = load_i16x16_i32x4(b);
        store_i32x16_u8(d, f(a0, b0), f(a1, b1), f(a2, b2), f(a3, b3));
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, _) = t1[done..n].as_chunks::<8>();
    let (b8, _) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        store_i32x8_u8(d, f(a0, b0), f(a1, b1));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    for ((d, a), b) in c4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u8(d, f(load_i16x4_i32(a), load_i16x4_i32(b)));
    }
    for ((d, &a), &b) in r4.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 + b as i32 + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = clip((t1[x]*weight + t2[x]*(16-weight) + rnd) >> sh, 0, 255)`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn w_avg_row_8bpc_neon(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    let w1 = vdup_n_s16(weight as i16);
    let w2 = vdup_n_s16((16 - weight) as i16);
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-sh);
    let f = |s: int32x4_t| vshlq_s32(vaddq_s32(s, rnd_v), nsh);

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    for ((d, a), b) in c16.iter_mut().zip(a16).zip(b16) {
        let (s0, s1) = madd_i16x8_const(
            load_i16x8((&a[..8]).try_into().unwrap()),
            load_i16x8((&b[..8]).try_into().unwrap()),
            w1,
            w2,
        );
        let (s2, s3) = madd_i16x8_const(
            load_i16x8((&a[8..]).try_into().unwrap()),
            load_i16x8((&b[8..]).try_into().unwrap()),
            w1,
            w2,
        );
        store_i32x16_u8(d, f(s0), f(s1), f(s2), f(s3));
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, _) = t1[done..n].as_chunks::<8>();
    let (b8, _) = t2[done..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let (s0, s1) = madd_i16x8_const(load_i16x8(a), load_i16x8(b), w1, w2);
        store_i32x8_u8(d, f(s0), f(s1));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let w1_32 = vdupq_n_s32(weight);
    let w2_32 = vdupq_n_s32(16 - weight);
    let f4 = |a: int32x4_t, b: int32x4_t| {
        vshlq_s32(
            vaddq_s32(vaddq_s32(vmulq_s32(a, w1_32), vmulq_s32(b, w2_32)), rnd_v),
            nsh,
        )
    };
    for ((d, a), b) in c4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u8(d, f4(load_i16x4_i32(a), load_i16x4_i32(b)));
    }
    for ((d, &a), &b) in r4.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = clip((t1[x]*m + t2[x]*(64-m) + rnd) >> sh, 0, 255)`, `m = mask[x]`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn mask_row_8bpc_neon(
    dst: &mut [u8],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let c64_16 = vdupq_n_s16(64);
    let nsh = vdupq_n_s32(-sh);
    let f = |s: int32x4_t| vshlq_s32(vaddq_s32(s, rnd_v), nsh);

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (a16, _) = t1[..n].as_chunks::<16>();
    let (b16, _) = t2[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for (((d, a), b), m) in c16.iter_mut().zip(a16).zip(b16).zip(m16) {
        let (m0, m1) = load_u8x16_i16x2(m);
        let (s0, s1) = madd_i16x8(
            load_i16x8((&a[..8]).try_into().unwrap()),
            load_i16x8((&b[..8]).try_into().unwrap()),
            m0,
            vsubq_s16(c64_16, m0),
        );
        let (s2, s3) = madd_i16x8(
            load_i16x8((&a[8..]).try_into().unwrap()),
            load_i16x8((&b[8..]).try_into().unwrap()),
            m1,
            vsubq_s16(c64_16, m1),
        );
        store_i32x16_u8(d, f(s0), f(s1), f(s2), f(s3));
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (a8, _) = t1[done..n].as_chunks::<8>();
    let (b8, _) = t2[done..n].as_chunks::<8>();
    let (m8, _) = mask[done..n].as_chunks::<8>();
    for (((d, a), b), m) in c8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let mv = load_u8x8_i16(m);
        let (s0, s1) = madd_i16x8(load_i16x8(a), load_i16x8(b), mv, vsubq_s16(c64_16, mv));
        store_i32x8_u8(d, f(s0), f(s1));
    }
    let done = done + c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    let c64 = vdupq_n_s32(64);
    let f4 = |a: int32x4_t, b: int32x4_t, m: int32x4_t| {
        vshlq_s32(
            vaddq_s32(
                vaddq_s32(vmulq_s32(a, m), vmulq_s32(b, vsubq_s32(c64, m))),
                rnd_v,
            ),
            nsh,
        )
    };
    for (((d, a), b), m) in c4.iter_mut().zip(a4).zip(b4).zip(m4) {
        store_i32x4_u8(
            d,
            f4(load_i16x4_i32(a), load_i16x4_i32(b), load_u8x4_i32(m)),
        );
    }
    for (((d, &a), &b), &m) in r4.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6`, `m = mask[x]`.
/// Uses dav2d's NEON precision shape: `umull/umlal` into u16, then rounded narrow.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn blend_row_8bpc_neon(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    let c64 = vdup_n_u8(64);
    let f = |d: uint8x8_t, t: uint8x8_t, m: uint8x8_t| {
        let inv_m = vsub_u8(c64, m);
        vrshrn_n_u16::<6>(vmlal_u8(vmull_u8(t, m), d, inv_m))
    };

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (t16, _) = tmp[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for ((d, t), m) in c16.iter_mut().zip(t16).zip(m16) {
        let dv = load_u8x16(&*d);
        let tv = load_u8x16(t);
        let mv = load_u8x16(m);
        store_u8x16(
            d,
            vcombine_u8(
                f(vget_low_u8(dv), vget_low_u8(tv), vget_low_u8(mv)),
                f(vget_high_u8(dv), vget_high_u8(tv), vget_high_u8(mv)),
            ),
        );
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (t8, tr) = tmp[done..n].as_chunks::<8>();
    let (m8, mr) = mask[done..n].as_chunks::<8>();
    for ((d, t), m) in c8.iter_mut().zip(t8).zip(m8) {
        store_u8x8(d, f(load_u8x8(&*d), load_u8x8(t), load_u8x8(m)));
    }
    for ((d, &t), &m) in r8.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u8;
    }
}

/// `dst[x] = clip((alpha*dst[x] + beta) >> 8, 0, 255)`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn morph_row_8bpc_neon(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    if !(i16::MIN as i32..=i16::MAX as i32).contains(&alpha) {
        for d in dst[..n].iter_mut() {
            *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, 255) as u8;
        }
        return;
    }

    let a_v = vdup_n_s16(alpha as i16);
    let b_v = vdupq_n_s32(beta);
    let f = |d: uint8x8_t| {
        let d = vreinterpretq_s16_u16(vmovl_u8(d));
        (
            vshrq_n_s32::<8>(vaddq_s32(vmull_s16(vget_low_s16(d), a_v), b_v)),
            vshrq_n_s32::<8>(vaddq_s32(vmull_s16(vget_high_s16(d), a_v), b_v)),
        )
    };

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    for d in c16.iter_mut() {
        let dv = load_u8x16(&*d);
        let (o0, o1) = f(vget_low_u8(dv));
        let (o2, o3) = f(vget_high_u8(dv));
        store_i32x16_u8(d, o0, o1, o2, o3);
    }
    let (c8, r8) = r16.as_chunks_mut::<8>();
    for d in c8.iter_mut() {
        let (o0, o1) = f(load_u8x8(&*d));
        store_i32x8_u8(d, o0, o1);
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for d in c4.iter_mut() {
        let r = vshrq_n_s32::<8>(vaddq_s32(
            vmulq_s32(load_u8x4_i32(d), vdupq_n_s32(alpha)),
            b_v,
        ));
        store_i32x4_u8(d, r);
    }
    for d in r4.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, 255) as u8;
    }
}

/// GDF residual add: `dst[x] = clip(dst[x] + sign(e)*((|e|+8)>>4), 0, 255)`,
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_add_run_8bpc_neon(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let sc = vdupq_n_s16(scale as i16);
    let rnd = vdupq_n_s16(8);
    let zero = vdupq_n_s16(0);
    let adj = |e: int16x8_t| {
        let diff = vmulq_s16(e, sc);
        let mag = vshrq_n_s16::<4>(vaddq_s16(vabsq_s16(diff), rnd));
        vbslq_s16(vcltq_s16(diff, zero), vnegq_s16(mag), mag)
    };

    let (dst16, dst_rem16) = dst[..n].as_chunks_mut::<16>();
    let (err16, err_rem16) = err[..n].as_chunks::<16>();
    for (d, e) in dst16.iter_mut().zip(err16) {
        let e = unsafe { vld1q_s8(e.as_ptr()) };
        let d0 = unsafe { vld1q_u8(d.as_ptr()) };
        let e0 = vmovl_s8(vget_low_s8(e));
        let e1 = vmovl_s8(vget_high_s8(e));
        let d0_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(d0)));
        let d0_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(d0)));
        let o0 = vaddq_s16(d0_lo, adj(e0));
        let o1 = vaddq_s16(d0_hi, adj(e1));
        unsafe {
            vst1q_u8(
                d.as_mut_ptr(),
                vcombine_u8(vqmovun_s16(o0), vqmovun_s16(o1)),
            )
        };
    }

    let (dst8, dst_tail) = dst_rem16.as_chunks_mut::<8>();
    let (err8, err_tail) = err_rem16.as_chunks::<8>();
    for (d, e) in dst8.iter_mut().zip(err8) {
        let e = unsafe { vld1_s8(e.as_ptr()) };
        let d0 = unsafe { vld1_u8(d.as_ptr()) };
        let o = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(d0)), adj(vmovl_s8(e)));
        unsafe { vst1_u8(d.as_mut_ptr(), vqmovun_s16(o)) };
    }

    for (d, &e) in dst_tail.iter_mut().zip(err_tail) {
        let diff = e as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let a = if diff < 0 { -mag } else { mag };
        *d = (*d as i32 + a).clamp(0, 255) as u8;
    }
}

/// GDF gradient: per-column `|2*b - a - c|` summed over the 2 rows into
/// 8 i16 lanes, then pair-reduced to up to four 2x2 output cells.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_gradient_group_neon(
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
    let mut acc = vdupq_n_s16(0);
    let nsh = vdupq_n_s16(-(shift as i16));
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let b = unsafe { vld1_u8(center_rows[y].as_ptr().add(bcol)) };
        let a = unsafe { vld1_u8(a_rows[y].as_ptr().add(acol)) };
        let c = unsafe { vld1_u8(c_rows[y].as_ptr().add(ccol)) };
        let b = vreinterpretq_s16_u16(vshlq_u16(vmovl_u8(b), nsh));
        let a = vreinterpretq_s16_u16(vshlq_u16(vmovl_u8(a), nsh));
        let c = vreinterpretq_s16_u16(vshlq_u16(vmovl_u8(c), nsh));
        let t = vsubq_s16(vsubq_s16(vaddq_s16(b, b), a), c);
        acc = vaddq_s16(acc, vabsq_s16(t));
    }
    let pair = vpaddq_s16(acc, acc);
    store_gdf_gradient_cells_i16x8(dst, d, base_cell, ncells, pair);
}

#[inline]
#[target_feature(enable = "neon")]
fn store_gdf_gradient_cells_i16x8(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    v: int16x8_t,
) {
    if ncells > 0 {
        dst[base_cell][d] = vgetq_lane_s16::<0>(v) as u16;
    }
    if ncells > 1 {
        dst[base_cell + 1][d] = vgetq_lane_s16::<1>(v) as u16;
    }
    if ncells > 2 {
        dst[base_cell + 2][d] = vgetq_lane_s16::<2>(v) as u16;
    }
    if ncells > 3 {
        dst[base_cell + 3][d] = vgetq_lane_s16::<3>(v) as u16;
    }
}

#[inline]
fn gdf_prep_apply_sign(v: i32) -> i32 {
    if v < 0 {
        -((v.wrapping_neg() + (1 << 14)) >> 15)
    } else {
        (v + (1 << 14)) >> 15
    }
}

#[inline]
fn gdf_prep_lookup_error(ref_dst_idx: usize, error_lut_base: usize, full_idx: usize) -> i8 {
    if ref_dst_idx == 0 {
        GDF_INTRA_ERROR[error_lut_base + full_idx]
    } else {
        GDF_INTER_ERROR[error_lut_base + full_idx]
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn gdf_load_pair_u8_i32(row: &[u8], col: usize) -> int32x4_t {
    let pair = unsafe {
        vreinterpret_u8_u16(vld1_lane_u16::<0>(
            row[col..].as_ptr().cast(),
            vdup_n_u16(0),
        ))
    };
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(pair))))
}

#[inline]
#[target_feature(enable = "neon")]
fn gdf_clip_i32x4(v: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, lo), hi)
}

#[inline(always)]
fn gdf_prep_full_idx(v0: i32, v1: i32, v2: i32, scale: i32) -> usize {
    let scale2 = scale as usize * 2;
    let v0 = gdf_prep_apply_sign(v0 * scale);
    let v1 = gdf_prep_apply_sign(v1 * scale);
    let v2 = gdf_prep_apply_sign(v2 * scale);
    let s0 = (v0.clamp(-scale, scale - 1) + scale) as usize;
    let s1 = (v1.clamp(-scale, scale - 1) + scale) as usize;
    let s2 = (v2.clamp(-scale, scale - 1) + scale) as usize;
    ((s0 * scale2) + s1) * scale2 + s2
}
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_prep_pair_8bpc_neon(
    rows: [&[u8]; 13],
    col: usize,
    cls: usize,
    shared_vals: [i32; 3],
    alpha_base: usize,
    weight_base: usize,
    error_lut_base: usize,
    scale: i32,
    ref_dst_idx: usize,
) -> [i8; 2] {
    let m = gdf_load_pair_u8_i32(rows[6], col);
    let v_lo = vdupq_n_s32(-512);
    let v_hi = vdupq_n_s32(511);
    let mut acc0 = vdupq_n_s32(shared_vals[0]);
    let mut acc1 = vdupq_n_s32(shared_vals[1]);
    let mut acc2 = vdupq_n_s32(shared_vals[2]);

    for (k, &[dy, dx]) in GDF_PREP_COORDS_8BPC.iter().enumerate() {
        let dy = dy as i32;
        let dx = dx as i32;
        let alpha = GDF_ALPHA[alpha_base + k * 4 + cls] as i32;
        let alpha_v = vdupq_n_s32(alpha);
        let neg_alpha_v = vdupq_n_s32(-alpha);
        let a_col = (col as i32 - dx) as usize;
        let b_col = (col as i32 + dx) as usize;
        let a = gdf_load_pair_u8_i32(rows[(6 - dy) as usize], a_col);
        let b = gdf_load_pair_u8_i32(rows[(6 + dy) as usize], b_col);
        let above = gdf_clip_i32x4(vshlq_n_s32::<2>(vsubq_s32(a, m)), neg_alpha_v, alpha_v);
        let below = gdf_clip_i32x4(vshlq_n_s32::<2>(vsubq_s32(b, m)), neg_alpha_v, alpha_v);
        let v = gdf_clip_i32x4(vaddq_s32(above, below), v_lo, v_hi);
        acc0 = vaddq_s32(
            acc0,
            vmulq_s32(v, vdupq_n_s32(GDF_WEIGHT[weight_base + k * 4 + cls] as i32)),
        );
        acc1 = vaddq_s32(
            acc1,
            vmulq_s32(
                v,
                vdupq_n_s32(GDF_WEIGHT[weight_base + 88 + k * 4 + cls] as i32),
            ),
        );
        acc2 = vaddq_s32(
            acc2,
            vmulq_s32(
                v,
                vdupq_n_s32(GDF_WEIGHT[weight_base + 176 + k * 4 + cls] as i32),
            ),
        );
    }

    [
        gdf_prep_lookup_error(
            ref_dst_idx,
            error_lut_base,
            gdf_prep_full_idx(
                vgetq_lane_s32::<0>(acc0),
                vgetq_lane_s32::<0>(acc1),
                vgetq_lane_s32::<0>(acc2),
                scale,
            ),
        ),
        gdf_prep_lookup_error(
            ref_dst_idx,
            error_lut_base,
            gdf_prep_full_idx(
                vgetq_lane_s32::<1>(acc0),
                vgetq_lane_s32::<1>(acc1),
                vgetq_lane_s32::<1>(acc2),
                scale,
            ),
        ),
    ]
}

/// cctx rotate+clip over two i16 coefficient planes, widening only inside the SIMD arithmetic.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cctx_row_i16_neon(
    u: &mut [i16],
    v: &mut [i16],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let sina16 = sina as i16;
        let cosa16 = cosa as i16;
        let c128 = vdupq_n_s32(128);
        let min_v = vdupq_n_s32(min);
        let max_v = vdupq_n_s32(max);
        let zero = vdupq_n_s32(0);
        let rot = |uu: int16x4_t, vv: int16x4_t| -> (int32x4_t, int32x4_t) {
            let a = vmlsl_n_s16(vmull_n_s16(uu, cosa16), vv, sina16);
            let b = vmlal_n_s16(vmull_n_s16(uu, sina16), vv, cosa16);
            let ra = vminq_s32(
                vmaxq_s32(
                    vshrq_n_s32::<8>(vaddq_s32(
                        vaddq_s32(a, c128),
                        vreinterpretq_s32_u32(vcltq_s32(a, zero)),
                    )),
                    min_v,
                ),
                max_v,
            );
            let rb = vminq_s32(
                vmaxq_s32(
                    vshrq_n_s32::<8>(vaddq_s32(
                        vaddq_s32(b, c128),
                        vreinterpretq_s32_u32(vcltq_s32(b, zero)),
                    )),
                    min_v,
                ),
                max_v,
            );
            (ra, rb)
        };
        let (u_chunks, ur) = u[..sz].as_chunks_mut::<8>();
        let (v_chunks, vr) = v[..sz].as_chunks_mut::<8>();
        for (uch, vch) in u_chunks.iter_mut().zip(v_chunks.iter_mut()) {
            let uu16 = vld1q_s16(uch.as_ptr());
            let vv16 = vld1q_s16(vch.as_ptr());
            let (ulo, vlo) = rot(vget_low_s16(uu16), vget_low_s16(vv16));
            let (uhi, vhi) = rot(vget_high_s16(uu16), vget_high_s16(vv16));
            vst1q_s16(
                uch.as_mut_ptr(),
                vcombine_s16(vmovn_s32(ulo), vmovn_s32(uhi)),
            );
            vst1q_s16(
                vch.as_mut_ptr(),
                vcombine_s16(vmovn_s32(vlo), vmovn_s32(vhi)),
            );
        }
        for (uu, vv) in ur.iter_mut().zip(vr.iter_mut()) {
            let ui = *uu as i32;
            let vi = *vv as i32;
            let a = ui * cosa - vi * sina;
            let b = ui * sina + vi * cosa;
            *uu = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max) as i16;
            *vv = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max) as i16;
        }
    }
}
