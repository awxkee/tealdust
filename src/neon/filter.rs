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

#[inline(always)]
fn load_i16x4_i32(a: &[i16; 4]) -> int32x4_t {
    unsafe { vmovl_s16(vld1_s16(a.as_ptr())) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x4_i32(a: &[u8; 4]) -> int32x4_t {
    // Pull the 4 bytes through a scalar u32 (no NEON over-read of a [u8; 4]).
    let dup = vreinterpret_u8_u32(vdup_n_u32(u32::from_le_bytes(*a)));
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(dup))))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i8x4_i32(a: &[i8; 4]) -> int32x4_t {
    let bytes = [a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8];
    let dup = vreinterpret_s8_u32(vdup_n_u32(u32::from_le_bytes(bytes)));
    vmovl_s16(vget_low_s16(vmovl_s8(dup)))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8_i32x2(a: &[u8; 8]) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vmovl_u8(vld1_u8(a.as_ptr())) };
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(w))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i8x8_i32x2(a: &[i8; 8]) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vmovl_s8(vld1_s8(a.as_ptr())) };
    (vmovl_s16(vget_low_s16(w)), vmovl_s16(vget_high_s16(w)))
}
#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8_i32x2(a: &[i16; 8]) -> (int32x4_t, int32x4_t) {
    let w = unsafe { vld1q_s16(a.as_ptr()) };
    (vmovl_s16(vget_low_s16(w)), vmovl_s16(vget_high_s16(w)))
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

#[inline(always)]
fn store_i16x8_u8(a: &mut [u8; 8], v: int16x8_t) {
    unsafe { vst1_u8(a.as_mut_ptr(), vqmovun_s16(v)) };
}

#[inline(always)]
fn store_i16x8x2_u8(a: &mut [u8; 16], lo: int16x8_t, hi: int16x8_t) {
    unsafe {
        vst1q_u8(
            a.as_mut_ptr(),
            vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi)),
        )
    };
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
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (cc8, _) = c[..n].as_chunks::<8>();
    for (d, cv) in c8.iter_mut().zip(cc8) {
        let cf_lo = vshlq_s32(
            vaddq_s32(load_i32x4((&cv[..4]).try_into().unwrap()), rnd_v),
            nsh,
        );
        let cf_hi = vshlq_s32(
            vaddq_s32(load_i32x4((&cv[4..]).try_into().unwrap()), rnd_v),
            nsh,
        );
        let (d_lo, d_hi) = load_u8x8_i32x2(&*d);
        store_i32x8_u8(d, vaddq_s32(d_lo, cf_lo), vaddq_s32(d_hi, cf_hi));
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (cc4, cr) = c[done..n].as_chunks::<4>();
    for (d, cv) in c4.iter_mut().zip(cc4) {
        let cf = vshlq_s32(vaddq_s32(load_i32x4(cv), rnd_v), nsh);
        let dv = load_u8x4_i32(d);
        store_i32x4_u8(d, vaddq_s32(dv, cf));
    }
    for (d, &cv) in r4.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, 255) as u8;
    }
}

/// `dst[i] = clip(dst[i] + dc, 0, 255)`.
#[inline]
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

    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (c8, r8) = r16.as_chunks_mut::<8>();

    if dc > 0 {
        let amt16 = vdupq_n_u8(amt);
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
    let (c8, r8) = tmp[..n].as_chunks_mut::<8>();
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
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        let lo = f(a0, b0);
        let hi = f(a1, b1);
        store_i32x8_u8(d, lo, hi);
    }
    let done = c8.len() * 8;
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
    let w1 = vdupq_n_s32(weight);
    let w2 = vdupq_n_s32(16 - weight);
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-sh);
    let f = |a: int32x4_t, b: int32x4_t| {
        vshlq_s32(
            vaddq_s32(vaddq_s32(vmulq_s32(a, w1), vmulq_s32(b, w2)), rnd_v),
            nsh,
        )
    };
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in c8.iter_mut().zip(a8).zip(b8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        let lo = f(a0, b0);
        let hi = f(a1, b1);
        store_i32x8_u8(d, lo, hi);
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    for ((d, a), b) in c4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u8(d, f(load_i16x4_i32(a), load_i16x4_i32(b)));
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
    let c64 = vdupq_n_s32(64);
    let nsh = vdupq_n_s32(-sh);
    let f = |a: int32x4_t, b: int32x4_t, m: int32x4_t| {
        vshlq_s32(
            vaddq_s32(
                vaddq_s32(vmulq_s32(a, m), vmulq_s32(b, vsubq_s32(c64, m))),
                rnd_v,
            ),
            nsh,
        )
    };
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    let (m8, _) = mask[..n].as_chunks::<8>();
    for (((d, a), b), m) in c8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        let (m0, m1) = load_u8x8_i32x2(m);
        let lo = f(a0, b0, m0);
        let hi = f(a1, b1, m1);
        store_i32x8_u8(d, lo, hi);
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    for (((d, a), b), m) in c4.iter_mut().zip(a4).zip(b4).zip(m4) {
        store_i32x4_u8(d, f(load_i16x4_i32(a), load_i16x4_i32(b), load_u8x4_i32(m)));
    }
    for (((d, &a), &b), &m) in r4.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, 255) as u8;
    }
}

/// `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6`, `m = mask[x]`. The weighted
/// average stays in [0,255] so it fits i16 lanes: 2x-unrolled to 16 px/iter.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn blend_row_8bpc_neon(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    let c64 = vdupq_n_s16(64);
    let rnd_v = vdupq_n_s16(32);
    let f = |d: int16x8_t, t: int16x8_t, m: int16x8_t| {
        vshrq_n_s16::<6>(vaddq_s16(
            vaddq_s16(vmulq_s16(d, vsubq_s16(c64, m)), vmulq_s16(t, m)),
            rnd_v,
        ))
    };
    let (c16, r16) = dst[..n].as_chunks_mut::<16>();
    let (t16, _) = tmp[..n].as_chunks::<16>();
    let (m16, _) = mask[..n].as_chunks::<16>();
    for ((d, t), m) in c16.iter_mut().zip(t16).zip(m16) {
        let (d0, d1) = load_u8x16_i16x2(&*d);
        let (t0, t1) = load_u8x16_i16x2(t);
        let (m0, m1) = load_u8x16_i16x2(m);
        let o0 = f(d0, t0, m0);
        let o1 = f(d1, t1, m1);
        store_i16x8x2_u8(d, o0, o1);
    }
    let done = c16.len() * 16;
    let (c8, r8) = r16.as_chunks_mut::<8>();
    let (t8, tr) = tmp[done..n].as_chunks::<8>();
    let (m8, mr) = mask[done..n].as_chunks::<8>();
    for ((d, t), m) in c8.iter_mut().zip(t8).zip(m8) {
        let o = f(load_u8x8_i16(d), load_u8x8_i16(t), load_u8x8_i16(m));
        store_i16x8_u8(d, o);
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
    let a_v = vdupq_n_s32(alpha);
    let b_v = vdupq_n_s32(beta);
    let f = |d: int32x4_t| vshrq_n_s32::<8>(vaddq_s32(vmulq_s32(d, a_v), b_v));
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in c8.iter_mut() {
        let (d0, d1) = load_u8x8_i32x2(&*d);
        let lo = f(d0);
        let hi = f(d1);
        store_i32x8_u8(d, lo, hi);
    }
    let (c4, r4) = r8.as_chunks_mut::<4>();
    for d in c4.iter_mut() {
        let r = f(load_u8x4_i32(d));
        store_i32x4_u8(d, r);
    }
    for d in r4.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, 255) as u8;
    }
}

/// GDF residual add: `dst[x] = clip(dst[x] + sign(e)*((|e|+8)>>4), 0, 255)`,
/// `e = err[x]*scale`. `vcltq_s32(e, 0)` selects the negated magnitude.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_add_run_8bpc_neon(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let sc = vdupq_n_s32(scale);
    let rnd = vdupq_n_s32(8);
    let zero = vdupq_n_s32(0);
    let adj = |e: int32x4_t| {
        let diff = vmulq_s32(e, sc);
        let mag = vshrq_n_s32::<4>(vaddq_s32(vabsq_s32(diff), rnd));
        vbslq_s32(vcltq_s32(diff, zero), vnegq_s32(mag), mag)
    };
    let (c8, r8) = dst[..n].as_chunks_mut::<8>();
    let (e8, _) = err[..n].as_chunks::<8>();
    for (d, e) in c8.iter_mut().zip(e8) {
        let (e0, e1) = load_i8x8_i32x2(e);
        let a_lo = adj(e0);
        let a_hi = adj(e1);
        let (d_lo, d_hi) = load_u8x8_i32x2(&*d);
        store_i32x8_u8(d, vaddq_s32(d_lo, a_lo), vaddq_s32(d_hi, a_hi));
    }
    let done = c8.len() * 8;
    let (c4, r4) = r8.as_chunks_mut::<4>();
    let (e4, er) = err[done..n].as_chunks::<4>();
    for (d, e) in c4.iter_mut().zip(e4) {
        let a = adj(load_i8x4_i32(e));
        let dv = load_u8x4_i32(d);
        store_i32x4_u8(d, vaddq_s32(dv, a));
    }
    for (d, &e) in r4.iter_mut().zip(er) {
        let diff = e as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let a = if diff < 0 { -mag } else { mag };
        *d = ((*d as i32) + a).clamp(0, 255) as u8;
    }
}

/// GDF gradient: per-column `|2*b - a - c|` (each `>> shift`) summed over the 2
/// rows into 8 lanes, then pair-reduced to `ncells` cells via `vpaddq`.
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
    let nsh = vdupq_n_s32(-(shift as i32));
    let mut acc_lo = vdupq_n_s32(0);
    let mut acc_hi = vdupq_n_s32(0);
    for y in 0..2 {
        let bcol = col0 - 1;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let brow: &[u8; 8] = center_rows[y][bcol..bcol + 8].try_into().unwrap();
        let arow: &[u8; 8] = a_rows[y][acol..acol + 8].try_into().unwrap();
        let crow: &[u8; 8] = c_rows[y][ccol..ccol + 8].try_into().unwrap();
        let sh = |a: &[u8; 4]| vshlq_s32(load_u8x4_i32(a), nsh);
        let b_lo = sh((&brow[..4]).try_into().unwrap());
        let b_hi = sh((&brow[4..]).try_into().unwrap());
        let a_lo = sh((&arow[..4]).try_into().unwrap());
        let a_hi = sh((&arow[4..]).try_into().unwrap());
        let c_lo = sh((&crow[..4]).try_into().unwrap());
        let c_hi = sh((&crow[4..]).try_into().unwrap());
        let t_lo = vsubq_s32(vsubq_s32(vaddq_s32(b_lo, b_lo), a_lo), c_lo);
        let t_hi = vsubq_s32(vsubq_s32(vaddq_s32(b_hi, b_hi), a_hi), c_hi);
        acc_lo = vaddq_s32(acc_lo, vabsq_s32(t_lo));
        acc_hi = vaddq_s32(acc_hi, vabsq_s32(t_hi));
    }
    // vpaddq pairs adjacent lanes: [a0+a1, a2+a3, b0+b1, b2+b3].
    let pair = vpaddq_s32(acc_lo, acc_hi);
    let mut out = [0i32; 4];
    store_i32x4(&mut out, pair);
    for k in 0..ncells {
        dst[base_cell + k][d] = out[k] as u16;
    }
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
        let mut i = 0usize;
        while i + 8 <= sz {
            let uu16 = vld1q_s16(u.as_ptr().add(i));
            let vv16 = vld1q_s16(v.as_ptr().add(i));
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
            let (ulo, vlo) = rot(vget_low_s16(uu16), vget_low_s16(vv16));
            let (uhi, vhi) = rot(vget_high_s16(uu16), vget_high_s16(vv16));
            vst1q_s16(
                u.as_mut_ptr().add(i),
                vcombine_s16(vmovn_s32(ulo), vmovn_s32(uhi)),
            );
            vst1q_s16(
                v.as_mut_ptr().add(i),
                vcombine_s16(vmovn_s32(vlo), vmovn_s32(vhi)),
            );
            i += 8;
        }
        while i < sz {
            let ui = u[i] as i32;
            let vi = v[i] as i32;
            let a = ui * cosa - vi * sina;
            let b = ui * sina + vi * cosa;
            u[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max) as i16;
            v[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max) as i16;
            i += 1;
        }
    }
}
