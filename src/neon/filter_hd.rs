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

const GDF_PREP_COORDS: [[i8; 2]; 18] = [
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
    [0, 3],
    [0, 2],
    [0, 1],
    [0, -1],
    [0, -2],
    [0, -3],
];

#[inline(always)]
fn load_i32x4(a: &[i32; 4]) -> int32x4_t {
    unsafe { vld1q_s32(a.as_ptr()) }
}

#[inline(always)]
fn load_i16x4_i32(a: &[i16; 4]) -> int32x4_t {
    unsafe { vmovl_s16(vld1_s16(a.as_ptr())) }
}

#[inline(always)]
fn load_u16x4_i32(a: &[u16; 4]) -> int32x4_t {
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(a.as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x4_i32(a: &[u8; 4]) -> int32x4_t {
    let dup = vreinterpret_u8_u32(vdup_n_u32(u32::from_le_bytes(*a)));
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(dup))))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8_i32x2(a: &[i16; 8]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_s16(a.as_ptr()) };
    (vmovl_s16(vget_low_s16(v)), vmovl_s16(vget_high_s16(v)))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i16x8(a: &[i16; 8]) -> int16x8_t {
    unsafe { vld1q_s16(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8_i16(a: &[u8; 8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vmovl_u8(vld1_u8(a.as_ptr()))) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8x8_u16(a: &[u8; 8]) -> uint16x8_t {
    unsafe { vmovl_u8(vld1_u8(a.as_ptr())) }
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
fn load_u16x8_i32x2(a: &[u16; 8]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u16(a.as_ptr()) };
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8(a: &[u16; 8]) -> uint16x8_t {
    unsafe { vld1q_u16(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u16x8_s16(a: &[u16; 8]) -> int16x8_t {
    unsafe { vreinterpretq_s16_u16(vld1q_u16(a.as_ptr())) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_u16x8(a: &mut [u16; 8], v: uint16x8_t) {
    unsafe { vst1q_u16(a.as_mut_ptr(), v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4_u16_clip(a: &mut [u16; 4], v: int32x4_t, max_v: int32x4_t) {
    let v = vminq_s32(vmaxq_s32(v, vdupq_n_s32(0)), max_v);
    unsafe { vst1_u16(a.as_mut_ptr(), vqmovun_s32(v)) };
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x8_u16_clip(a: &mut [u16; 8], lo: int32x4_t, hi: int32x4_t, max_v: int32x4_t) {
    let zero = vdupq_n_s32(0);
    let lo = vminq_s32(vmaxq_s32(lo, zero), max_v);
    let hi = vminq_s32(vmaxq_s32(hi, zero), max_v);
    unsafe {
        vst1q_u16(
            a.as_mut_ptr(),
            vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)),
        )
    };
}

#[inline]
#[target_feature(enable = "neon")]
fn shr(v: int32x4_t, sh: i32) -> int32x4_t {
    vshlq_s32(v, vdupq_n_s32(-sh))
}

#[target_feature(enable = "neon")]
pub(crate) fn residual_add_row_hbd_neon(
    dst: &mut [u16],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
    bitdepth_max: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |d: int32x4_t, cv: int32x4_t| vaddq_s32(d, shr(vaddq_s32(cv, rnd_v), shift));

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (c8, _) = c[..n].as_chunks::<8>();
    for (d, cv) in d8.iter_mut().zip(c8) {
        let (d0, d1) = load_u16x8_i32x2(&*d);
        let c0 = load_i32x4((&cv[..4]).try_into().unwrap());
        let c1 = load_i32x4((&cv[4..]).try_into().unwrap());
        store_i32x8_u16_clip(d, f(d0, c0), f(d1, c1), max_v);
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (c4, cr) = c[done..n].as_chunks::<4>();
    for (d, cv) in d4.iter_mut().zip(c4) {
        store_i32x4_u16_clip(d, f(load_u16x4_i32(&*d), load_i32x4(cv)), max_v);
    }
    for (d, &cv) in r4.iter_mut().zip(cr) {
        *d = ((*d as i32) + ((cv + rnd) >> shift)).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_add_row_hbd_neon(dst: &mut [u16], dc: i32, n: usize, bitdepth_max: i32) {
    if dc == 0 {
        return;
    }
    let dc_v = vdupq_n_s32(dc);
    let max_v = vdupq_n_s32(bitdepth_max);
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        let (d0, d1) = load_u16x8_i32x2(&*d);
        store_i32x8_u16_clip(d, vaddq_s32(d0, dc_v), vaddq_s32(d1, dc_v), max_v);
    }
    let (d4, r4) = r8.as_chunks_mut::<4>();
    for d in d4.iter_mut() {
        store_i32x4_u16_clip(d, vaddq_s32(load_u16x4_i32(&*d), dc_v), max_v);
    }
    for d in r4.iter_mut() {
        *d = ((*d as i32) + dc).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn avg_row_hbd_neon(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |a: int32x4_t, b: int32x4_t| shr(vaddq_s32(vaddq_s32(a, b), rnd_v), sh);

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in d8.iter_mut().zip(a8).zip(b8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        store_i32x8_u16_clip(d, f(a0, b0), f(a1, b1), max_v);
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    for ((d, a), b) in d4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u16_clip(d, f(load_i16x4_i32(a), load_i16x4_i32(b)), max_v);
    }
    for ((d, &a), &b) in r4.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 + b as i32 + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn w_avg_row_hbd_neon(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let w1 = vdup_n_s16(weight as i16);
    let w2 = vdup_n_s16((16 - weight) as i16);
    let rnd_v = vdupq_n_s32(rnd);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |s: int32x4_t| shr(vaddq_s32(s, rnd_v), sh);

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    for ((d, a), b) in d8.iter_mut().zip(a8).zip(b8) {
        let (s0, s1) = madd_i16x8_const(load_i16x8(a), load_i16x8(b), w1, w2);
        store_i32x8_u16_clip(d, f(s0), f(s1), max_v);
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let w1_32 = vdupq_n_s32(weight);
    let w2_32 = vdupq_n_s32(16 - weight);
    let f4 = |a: int32x4_t, b: int32x4_t| {
        shr(
            vaddq_s32(vaddq_s32(vmulq_s32(a, w1_32), vmulq_s32(b, w2_32)), rnd_v),
            sh,
        )
    };
    for ((d, a), b) in d4.iter_mut().zip(a4).zip(b4) {
        store_i32x4_u16_clip(d, f4(load_i16x4_i32(a), load_i16x4_i32(b)), max_v);
    }
    for ((d, &a), &b) in r4.iter_mut().zip(ar).zip(br) {
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, bitdepth_max)
            as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn mask_row_hbd_neon(
    dst: &mut [u16],
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
    bitdepth_max: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let c64_16 = vdupq_n_s16(64);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |s: int32x4_t| shr(vaddq_s32(s, rnd_v), sh);

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    let (m8, _) = mask[..n].as_chunks::<8>();
    for (((d, a), b), m) in d8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let mv = load_u8x8_i16(m);
        let (s0, s1) = madd_i16x8(load_i16x8(a), load_i16x8(b), mv, vsubq_s16(c64_16, mv));
        store_i32x8_u16_clip(d, f(s0), f(s1), max_v);
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    let c64 = vdupq_n_s32(64);
    let f4 = |a: int32x4_t, b: int32x4_t, m: int32x4_t| {
        shr(
            vaddq_s32(
                vaddq_s32(vmulq_s32(a, m), vmulq_s32(b, vsubq_s32(c64, m))),
                rnd_v,
            ),
            sh,
        )
    };
    for (((d, a), b), m) in d4.iter_mut().zip(a4).zip(b4).zip(m4) {
        store_i32x4_u16_clip(
            d,
            f4(load_i16x4_i32(a), load_i16x4_i32(b), load_u8x4_i32(m)),
            max_v,
        );
    }
    for (((d, &a), &b), &m) in r4.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn blend_row_hbd_neon(dst: &mut [u16], tmp: &[u16], mask: &[u8], n: usize) {
    let c64 = vdupq_n_u16(64);
    let f = |d: uint16x8_t, t: uint16x8_t, m: uint16x8_t| {
        let inv_m = vsubq_u16(c64, m);
        let lo = vmlal_u16(
            vmull_u16(vget_low_u16(t), vget_low_u16(m)),
            vget_low_u16(d),
            vget_low_u16(inv_m),
        );
        let hi = vmlal_u16(
            vmull_u16(vget_high_u16(t), vget_high_u16(m)),
            vget_high_u16(d),
            vget_high_u16(inv_m),
        );
        vcombine_u16(vrshrn_n_u32::<6>(lo), vrshrn_n_u32::<6>(hi))
    };

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (t8, _) = tmp[..n].as_chunks::<8>();
    let (m8, _) = mask[..n].as_chunks::<8>();
    for ((d, t), m) in d8.iter_mut().zip(t8).zip(m8) {
        store_u16x8(d, f(load_u16x8(&*d), load_u16x8(t), load_u8x8_u16(m)));
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (t4, tr) = tmp[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    let c64_32 = vdupq_n_s32(64);
    let rnd_v = vdupq_n_s32(32);
    let max_v = vdupq_n_s32(0xffff);
    let f4 = |d: int32x4_t, t: int32x4_t, m: int32x4_t| {
        vshrq_n_s32::<6>(vaddq_s32(
            vaddq_s32(vmulq_s32(d, vsubq_s32(c64_32, m)), vmulq_s32(t, m)),
            rnd_v,
        ))
    };
    for ((d, t), m) in d4.iter_mut().zip(t4).zip(m4) {
        store_i32x4_u16_clip(
            d,
            f4(load_u16x4_i32(&*d), load_u16x4_i32(t), load_u8x4_i32(m)),
            max_v,
        );
    }
    for ((d, &t), &m) in r4.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u16;
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn morph_row_hbd_neon(
    dst: &mut [u16],
    alpha: i32,
    beta: i32,
    n: usize,
    bitdepth_max: i32,
) {
    if !(i16::MIN as i32..=i16::MAX as i32).contains(&alpha) || bitdepth_max > i16::MAX as i32 {
        for d in dst[..n].iter_mut() {
            *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, bitdepth_max) as u16;
        }
        return;
    }

    let a_v = vdup_n_s16(alpha as i16);
    let b_v = vdupq_n_s32(beta);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |d: int16x8_t| {
        (
            vshrq_n_s32::<8>(vaddq_s32(vmull_s16(vget_low_s16(d), a_v), b_v)),
            vshrq_n_s32::<8>(vaddq_s32(vmull_s16(vget_high_s16(d), a_v), b_v)),
        )
    };

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        let (o0, o1) = f(load_u16x8_s16(&*d));
        store_i32x8_u16_clip(d, o0, o1, max_v);
    }
    let (d4, r4) = r8.as_chunks_mut::<4>();
    for d in d4.iter_mut() {
        let r = vshrq_n_s32::<8>(vaddq_s32(
            vmulq_s32(load_u16x4_i32(&*d), vdupq_n_s32(alpha)),
            b_v,
        ));
        store_i32x4_u16_clip(d, r, max_v);
    }
    for d in r4.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, bitdepth_max) as u16;
    }
}

#[inline]
fn gdf_bitdepth_from_max(bitdepth_max: i32) -> i32 {
    if bitdepth_max <= 0xff {
        8
    } else if bitdepth_max <= 0x3ff {
        10
    } else {
        12
    }
}

/// High-bit-depth GDF residual add.
///
/// Matches dav2d/AVM scalar scaling: 10-bit residuals are shifted by 2 on add,
/// while 12-bit residuals are used at full precision.
#[target_feature(enable = "neon")]
pub(crate) fn gdf_add_run_hbd_neon(
    dst: &mut [u16],
    err: &[i8],
    scale: i32,
    n: usize,
    bitdepth_max: i32,
) {
    let bitdepth = gdf_bitdepth_from_max(bitdepth_max);
    let shift = 12 - bitdepth;
    let sc = vdupq_n_s16(scale as i16);
    let rnd = vdupq_n_s16(if shift == 0 { 0 } else { 1 << (shift - 1) });
    let nsh = vdupq_n_s16(-(shift as i16));
    let zero = vdupq_n_s16(0);
    let max_v = vdupq_n_s16(bitdepth_max as i16);
    let adj = |e: int16x8_t| {
        let diff = vmulq_s16(e, sc);
        let mag = vshlq_s16(vaddq_s16(vabsq_s16(diff), rnd), nsh);
        vbslq_s16(vcltq_s16(diff, zero), vnegq_s16(mag), mag)
    };
    let clip = |v: int16x8_t| vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, zero), max_v));

    let (d8, tail_dst) = dst[..n].as_chunks_mut::<8>();
    let (e8, tail_err) = err[..n].as_chunks::<8>();
    for (d, e) in d8.iter_mut().zip(e8) {
        let d0 = vreinterpretq_s16_u16(unsafe { vld1q_u16(d.as_ptr()) });
        let e0 = unsafe { vmovl_s8(vld1_s8(e.as_ptr())) };
        unsafe { vst1q_u16(d.as_mut_ptr(), clip(vaddq_s16(d0, adj(e0)))) };
    }

    let rnd_scalar = if shift == 0 { 0 } else { 1 << (shift - 1) };
    for (d, &e) in tail_dst.iter_mut().zip(tail_err) {
        let diff = e as i32 * scale;
        let mag = (diff.abs() + rnd_scalar) >> shift;
        let adj = if diff < 0 { -mag } else { mag };
        *d = (*d as i32 + adj).clamp(0, bitdepth_max) as u16;
    }
}

/// HBD GDF gradient: per-column `|2*b - a - c|` summed over two rows,
/// then pair-reduced to up to four 2x2 output cells.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_gradient_group_hbd_neon(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    center_rows: [&[u16]; 2],
    a_rows: [&[u16]; 2],
    c_rows: [&[u16]; 2],
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
        let b = unsafe { vld1q_u16(center_rows[y].as_ptr().add(bcol)) };
        let a = unsafe { vld1q_u16(a_rows[y].as_ptr().add(acol)) };
        let c = unsafe { vld1q_u16(c_rows[y].as_ptr().add(ccol)) };
        let b = vreinterpretq_s16_u16(vshlq_u16(b, nsh));
        let a = vreinterpretq_s16_u16(vshlq_u16(a, nsh));
        let c = vreinterpretq_s16_u16(vshlq_u16(c, nsh));
        let t = vsubq_s16(vsubq_s16(vaddq_s16(b, b), a), c);
        acc = vaddq_s16(acc, vabsq_s16(t));
    }
    let pair = vpaddq_s16(acc, acc);
    let mut out = [0i16; 8];
    unsafe { vst1q_s16(out.as_mut_ptr(), pair) };
    for k in 0..ncells {
        dst[base_cell + k][d] = out[k] as u16;
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
fn gdf_load_pair_i32(row: &[u16], col: usize, nsh: int32x4_t) -> int32x4_t {
    let raw = unsafe { core::ptr::read_unaligned(row.as_ptr().add(col).cast::<u32>()) };
    let pair = vreinterpret_u16_u32(vdup_n_u32(raw));
    vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(pair)), nsh)
}

#[inline]
#[target_feature(enable = "neon")]
fn gdf_clip_i32x4(v: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    vminq_s32(vmaxq_s32(v, lo), hi)
}

#[inline]
#[target_feature(enable = "neon")]
fn gdf_store_i32x4(v: int32x4_t) -> [i32; 4] {
    let mut out = [0i32; 4];
    unsafe { vst1q_s32(out.as_mut_ptr(), v) };
    out
}

/// HBD GDF prep inner 2-pixel pair.
///
/// The pair shares `cls` and gradient-derived `shared_vals`, so NEON is used
/// across the two x samples while the final LUT indexing remains scalar.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_prep_pair_hbd_neon(
    rows: [&[u16]; 13],
    col: usize,
    cls: usize,
    shared_vals: [i32; 3],
    alpha_base: usize,
    weight_base: usize,
    error_lut_base: usize,
    scale: i32,
    down_shift: u32,
    up_scale: i32,
    ref_dst_idx: usize,
) -> [i8; 2] {
    let nsh = vdupq_n_s32(-(down_shift as i32));
    let m = gdf_load_pair_i32(rows[6], col, nsh);
    let up_scale_v = vdupq_n_s32(up_scale);
    let v_lo = vdupq_n_s32(-512);
    let v_hi = vdupq_n_s32(511);
    let mut acc0 = vdupq_n_s32(shared_vals[0]);
    let mut acc1 = vdupq_n_s32(shared_vals[1]);
    let mut acc2 = vdupq_n_s32(shared_vals[2]);

    for (k, &[dy, dx]) in GDF_PREP_COORDS.iter().enumerate() {
        let dy = dy as i32;
        let dx = dx as i32;
        let alpha = GDF_ALPHA[alpha_base + k * 4 + cls] as i32;
        let alpha_v = vdupq_n_s32(alpha);
        let neg_alpha_v = vdupq_n_s32(-alpha);
        let a_col = (col as i32 - dx) as usize;
        let b_col = (col as i32 + dx) as usize;
        let a = gdf_load_pair_i32(rows[(6 - dy) as usize], a_col, nsh);
        let b = gdf_load_pair_i32(rows[(6 + dy) as usize], b_col, nsh);
        let above = gdf_clip_i32x4(vmulq_s32(vsubq_s32(a, m), up_scale_v), neg_alpha_v, alpha_v);
        let below = gdf_clip_i32x4(vmulq_s32(vsubq_s32(b, m), up_scale_v), neg_alpha_v, alpha_v);
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

    let vals = [
        gdf_store_i32x4(acc0),
        gdf_store_i32x4(acc1),
        gdf_store_i32x4(acc2),
    ];
    let mut out = [0i8; 2];
    for lane in 0..2 {
        let mut full_idx = 0usize;
        for idx_vals in &vals {
            let v = gdf_prep_apply_sign(idx_vals[lane] * scale);
            let sub_idx = (v.clamp(-scale, scale - 1) + scale) as usize;
            full_idx = full_idx * (scale as usize * 2) + sub_idx;
        }
        out[lane] = gdf_prep_lookup_error(ref_dst_idx, error_lut_base, full_idx);
    }
    out
}
