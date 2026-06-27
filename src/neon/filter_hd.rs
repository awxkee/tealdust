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
fn load_u16x8_i32x2(a: &[u16; 8]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u16(a.as_ptr()) };
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v))),
    )
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
pub(crate) unsafe fn avg_row_hbd_neon(
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
    let w1 = vdupq_n_s32(weight);
    let w2 = vdupq_n_s32(16 - weight);
    let rnd_v = vdupq_n_s32(rnd);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |a: int32x4_t, b: int32x4_t| {
        shr(
            vaddq_s32(vaddq_s32(vmulq_s32(a, w1), vmulq_s32(b, w2)), rnd_v),
            sh,
        )
    };

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
        *d = ((a as i32 * weight + b as i32 * (16 - weight) + rnd) >> sh).clamp(0, bitdepth_max)
            as u16;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mask_row_hbd_neon(
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
    let c64 = vdupq_n_s32(64);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |a: int32x4_t, b: int32x4_t, m: int32x4_t| {
        shr(
            vaddq_s32(
                vaddq_s32(vmulq_s32(a, m), vmulq_s32(b, vsubq_s32(c64, m))),
                rnd_v,
            ),
            sh,
        )
    };

    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (a8, _) = t1[..n].as_chunks::<8>();
    let (b8, _) = t2[..n].as_chunks::<8>();
    let (m8, _) = mask[..n].as_chunks::<8>();
    for (((d, a), b), m) in d8.iter_mut().zip(a8).zip(b8).zip(m8) {
        let (a0, a1) = load_i16x8_i32x2(a);
        let (b0, b1) = load_i16x8_i32x2(b);
        let m0 = load_u8x4_i32((&m[..4]).try_into().unwrap());
        let m1 = load_u8x4_i32((&m[4..]).try_into().unwrap());
        store_i32x8_u16_clip(d, f(a0, b0, m0), f(a1, b1, m1), max_v);
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (a4, ar) = t1[done..n].as_chunks::<4>();
    let (b4, br) = t2[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    for (((d, a), b), m) in d4.iter_mut().zip(a4).zip(b4).zip(m4) {
        store_i32x4_u16_clip(
            d,
            f(load_i16x4_i32(a), load_i16x4_i32(b), load_u8x4_i32(m)),
            max_v,
        );
    }
    for (((d, &a), &b), &m) in r4.iter_mut().zip(ar).zip(br).zip(mr) {
        let mk = m as i32;
        *d = ((a as i32 * mk + b as i32 * (64 - mk) + rnd) >> sh).clamp(0, bitdepth_max) as u16;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn blend_row_hbd_neon(dst: &mut [u16], tmp: &[u16], mask: &[u8], n: usize) {
    let c64 = vdupq_n_s32(64);
    let rnd_v = vdupq_n_s32(32);
    let max_v = vdupq_n_s32(0xffff);
    let f = |d: int32x4_t, t: int32x4_t, m: int32x4_t| {
        vshrq_n_s32::<6>(vaddq_s32(
            vaddq_s32(vmulq_s32(d, vsubq_s32(c64, m)), vmulq_s32(t, m)),
            rnd_v,
        ))
    };
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    let (t8, _) = tmp[..n].as_chunks::<8>();
    let (m8, _) = mask[..n].as_chunks::<8>();
    for ((d, t), m) in d8.iter_mut().zip(t8).zip(m8) {
        let (d0, d1) = load_u16x8_i32x2(&*d);
        let (t0, t1) = load_u16x8_i32x2(t);
        let m0 = load_u8x4_i32((&m[..4]).try_into().unwrap());
        let m1 = load_u8x4_i32((&m[4..]).try_into().unwrap());
        store_i32x8_u16_clip(d, f(d0, t0, m0), f(d1, t1, m1), max_v);
    }
    let done = d8.len() * 8;
    let (d4, r4) = r8.as_chunks_mut::<4>();
    let (t4, tr) = tmp[done..n].as_chunks::<4>();
    let (m4, mr) = mask[done..n].as_chunks::<4>();
    for ((d, t), m) in d4.iter_mut().zip(t4).zip(m4) {
        store_i32x4_u16_clip(
            d,
            f(load_u16x4_i32(&*d), load_u16x4_i32(t), load_u8x4_i32(m)),
            max_v,
        );
    }
    for ((d, &t), &m) in r4.iter_mut().zip(tr).zip(mr) {
        let mk = m as i32;
        *d = (((*d as i32) * (64 - mk) + (t as i32) * mk + 32) >> 6) as u16;
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn morph_row_hbd_neon(
    dst: &mut [u16],
    alpha: i32,
    beta: i32,
    n: usize,
    bitdepth_max: i32,
) {
    let a_v = vdupq_n_s32(alpha);
    let b_v = vdupq_n_s32(beta);
    let max_v = vdupq_n_s32(bitdepth_max);
    let f = |d: int32x4_t| vshrq_n_s32::<8>(vaddq_s32(vmulq_s32(d, a_v), b_v));
    let (d8, r8) = dst[..n].as_chunks_mut::<8>();
    for d in d8.iter_mut() {
        let (d0, d1) = load_u16x8_i32x2(&*d);
        store_i32x8_u16_clip(d, f(d0), f(d1), max_v);
    }
    let (d4, r4) = r8.as_chunks_mut::<4>();
    for d in d4.iter_mut() {
        store_i32x4_u16_clip(d, f(load_u16x4_i32(&*d)), max_v);
    }
    for d in r4.iter_mut() {
        *d = ((alpha * (*d as i32) + beta) >> 8).clamp(0, bitdepth_max) as u16;
    }
}
