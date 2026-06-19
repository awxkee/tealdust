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
pub(crate) fn residual_add_row_8bpc_neon(
    dst: &mut [u8],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-shift);
    let mut x = 0;
    while x + 8 <= n {
        let c0 = unsafe { vld1q_s32(c[x..].as_ptr()) };
        let c1 = unsafe { vld1q_s32(c[x + 4..].as_ptr()) };
        let cf0 = vshlq_s32(vaddq_s32(c0, rnd_v), nsh);
        let cf1 = vshlq_s32(vaddq_s32(c1, rnd_v), nsh);
        let dpix = unsafe { vld1_u8(dst[x..].as_ptr()) };
        let d16 = vmovl_u8(dpix);
        let d0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(d16)));
        let d1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(d16)));
        let r0 = vaddq_s32(d0, cf0);
        let r1 = vaddq_s32(d1, cf1);
        let r16 = vcombine_u16(vqmovun_s32(r0), vqmovun_s32(r1));
        let r8 = vqmovn_u16(r16);
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), r8);
        }
        x += 8;
    }
    while x < n {
        let v = (c[x] + rnd) >> shift;
        dst[x] = (dst[x] as i32 + v).clamp(0, 255) as u8;
        x += 1;
    }
}

/// 8-bit DC add (NEON mirror): `dst[i] = clip(dst[i] + dc, 0, 255)`, 8 px/iter.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn dc_add_row_8bpc_neon(dst: &mut [u8], dc: i32, n: usize) {
    let dc_v = vdupq_n_s32(dc);
    let mut x = 0;
    while x + 8 <= n {
        let dpix = unsafe { vld1_u8(dst[x..].as_ptr()) };
        let d16 = vmovl_u8(dpix);
        let d0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(d16)));
        let d1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(d16)));
        let r16 = vcombine_u16(
            vqmovun_s32(vaddq_s32(d0, dc_v)),
            vqmovun_s32(vaddq_s32(d1, dc_v)),
        );
        let r8 = vqmovn_u16(r16);
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), r8);
        }
        x += 8;
    }
    while x < n {
        dst[x] = (dst[x] as i32 + dc).clamp(0, 255) as u8;
        x += 1;
    }
}

/// itx row-clip (NEON): `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn row_clip_neon(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = vdupq_n_s32(rnd);
    let nsh = vdupq_n_s32(-shift);
    let min_v = vdupq_n_s32(min);
    let max_v = vdupq_n_s32(max);
    let mut x = 0;
    while x + 4 <= n {
        let v = unsafe { vld1q_s32(tmp[x..].as_ptr()) };
        let v = vshlq_s32(vaddq_s32(v, rnd_v), nsh);
        let v = vminq_s32(vmaxq_s32(v, min_v), max_v);
        unsafe {
            vst1q_s32(tmp[x..].as_mut_ptr(), v);
        }
        x += 4;
    }
    while x < n {
        tmp[x] = ((tmp[x] + rnd) >> shift).max(min).min(max);
        x += 1;
    }
}

/// cctx rotate+clip over two i32 planes (NEON), 4 lanes/iter. `vcltzq` gives the
/// `-1` mask where the lane is negative, matching `- (a < 0)`.
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
    let nsh8 = vdupq_n_s32(-8);
    let mut i = 0;
    while i + 4 <= sz {
        let uu = unsafe { vld1q_s32(u[i..].as_ptr()) };
        let vv = unsafe { vld1q_s32(v[i..].as_ptr()) };
        let a = vsubq_s32(vmulq_s32(uu, cosa_v), vmulq_s32(vv, sina_v));
        let b = vaddq_s32(vmulq_s32(uu, sina_v), vmulq_s32(vv, cosa_v));
        let amask = vreinterpretq_s32_u32(vcltq_s32(a, zero));
        let bmask = vreinterpretq_s32_u32(vcltq_s32(b, zero));
        let ra = vshlq_s32(vaddq_s32(vaddq_s32(a, c128), amask), nsh8);
        let rb = vshlq_s32(vaddq_s32(vaddq_s32(b, c128), bmask), nsh8);
        let ra = vminq_s32(vmaxq_s32(ra, min_v), max_v);
        let rb = vminq_s32(vmaxq_s32(rb, min_v), max_v);
        unsafe {
            vst1q_s32(u[i..].as_mut_ptr(), ra);
            vst1q_s32(v[i..].as_mut_ptr(), rb);
        }
        i += 4;
    }
    while i < sz {
        let a = u[i] * cosa - v[i] * sina;
        let b = u[i] * sina + v[i] * cosa;
        u[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        v[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
        i += 1;
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
    let mut x = 0;
    while x + 8 <= n {
        let a = unsafe { vld1q_s16(t1[x..].as_ptr()) };
        let b = unsafe { vld1q_s16(t2[x..].as_ptr()) };
        let lo = vshlq_s32(
            vaddq_s32(
                vaddq_s32(vmovl_s16(vget_low_s16(a)), vmovl_s16(vget_low_s16(b))),
                rnd_v,
            ),
            nsh,
        );
        let hi = vshlq_s32(
            vaddq_s32(
                vaddq_s32(vmovl_s16(vget_high_s16(a)), vmovl_s16(vget_high_s16(b))),
                rnd_v,
            ),
            nsh,
        );
        let r16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), vqmovn_u16(r16));
        }
        x += 8;
    }
    while x < n {
        dst[x] = ((t1[x] as i32 + t2[x] as i32 + rnd) >> sh).clamp(0, 255) as u8;
        x += 1;
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
    let mut x = 0;
    while x + 8 <= n {
        let a = unsafe { vld1q_s16(t1[x..].as_ptr()) };
        let b = unsafe { vld1q_s16(t2[x..].as_ptr()) };
        let lo = vshlq_s32(
            vaddq_s32(
                vaddq_s32(
                    vmulq_s32(vmovl_s16(vget_low_s16(a)), w1),
                    vmulq_s32(vmovl_s16(vget_low_s16(b)), w2),
                ),
                rnd_v,
            ),
            nsh,
        );
        let hi = vshlq_s32(
            vaddq_s32(
                vaddq_s32(
                    vmulq_s32(vmovl_s16(vget_high_s16(a)), w1),
                    vmulq_s32(vmovl_s16(vget_high_s16(b)), w2),
                ),
                rnd_v,
            ),
            nsh,
        );
        let r16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), vqmovn_u16(r16));
        }
        x += 8;
    }
    while x < n {
        dst[x] = ((t1[x] as i32 * weight + t2[x] as i32 * (16 - weight) + rnd) >> sh).clamp(0, 255)
            as u8;
        x += 1;
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
    let mut x = 0;
    while x + 8 <= n {
        let a = unsafe { vld1q_s16(t1[x..].as_ptr()) };
        let b = unsafe { vld1q_s16(t2[x..].as_ptr()) };
        let mw = unsafe { vmovl_u8(vld1_u8(mask[x..].as_ptr())) };
        let m_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mw)));
        let m_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mw)));
        let lo = vshlq_s32(
            vaddq_s32(
                vaddq_s32(
                    vmulq_s32(vmovl_s16(vget_low_s16(a)), m_lo),
                    vmulq_s32(vmovl_s16(vget_low_s16(b)), vsubq_s32(c64, m_lo)),
                ),
                rnd_v,
            ),
            nsh,
        );
        let hi = vshlq_s32(
            vaddq_s32(
                vaddq_s32(
                    vmulq_s32(vmovl_s16(vget_high_s16(a)), m_hi),
                    vmulq_s32(vmovl_s16(vget_high_s16(b)), vsubq_s32(c64, m_hi)),
                ),
                rnd_v,
            ),
            nsh,
        );
        let r16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), vqmovn_u16(r16));
        }
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        dst[x] = ((t1[x] as i32 * m + t2[x] as i32 * (64 - m) + rnd) >> sh).clamp(0, 255) as u8;
        x += 1;
    }
}

/// `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6`, `m = mask[x]` (in-range).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn blend_row_8bpc_neon(dst: &mut [u8], tmp: &[u8], mask: &[u8], n: usize) {
    let c64 = vdupq_n_s32(64);
    let rnd_v = vdupq_n_s32(32);
    let nsh6 = vdupq_n_s32(-6);
    let mut x = 0;
    while x + 8 <= n {
        let dw = unsafe { vmovl_u8(vld1_u8(dst[x..].as_ptr())) };
        let tw = unsafe { vmovl_u8(vld1_u8(tmp[x..].as_ptr())) };
        let mw = unsafe { vmovl_u8(vld1_u8(mask[x..].as_ptr())) };
        let d_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(dw)));
        let d_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(dw)));
        let t_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tw)));
        let t_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tw)));
        let m_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(mw)));
        let m_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(mw)));
        let lo = vshlq_s32(
            vaddq_s32(
                vaddq_s32(vmulq_s32(d_lo, vsubq_s32(c64, m_lo)), vmulq_s32(t_lo, m_lo)),
                rnd_v,
            ),
            nsh6,
        );
        let hi = vshlq_s32(
            vaddq_s32(
                vaddq_s32(vmulq_s32(d_hi, vsubq_s32(c64, m_hi)), vmulq_s32(t_hi, m_hi)),
                rnd_v,
            ),
            nsh6,
        );
        let r16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), vqmovn_u16(r16));
        }
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        let d = dst[x] as i32;
        let t = tmp[x] as i32;
        dst[x] = ((d * (64 - m) + t * m + 32) >> 6) as u8;
        x += 1;
    }
}

/// `dst[x] = clip((alpha*dst[x] + beta) >> 8, 0, 255)`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn morph_row_8bpc_neon(dst: &mut [u8], alpha: i32, beta: i32, n: usize) {
    let a_v = vdupq_n_s32(alpha);
    let b_v = vdupq_n_s32(beta);
    let nsh8 = vdupq_n_s32(-8);
    let mut x = 0;
    while x + 8 <= n {
        let dw = unsafe { vmovl_u8(vld1_u8(dst[x..].as_ptr())) };
        let d_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(dw)));
        let d_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(dw)));
        let lo = vshlq_s32(vaddq_s32(vmulq_s32(d_lo, a_v), b_v), nsh8);
        let hi = vshlq_s32(vaddq_s32(vmulq_s32(d_hi, a_v), b_v), nsh8);
        let r16 = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), vqmovn_u16(r16));
        }
        x += 8;
    }
    while x < n {
        dst[x] = ((alpha * dst[x] as i32 + beta) >> 8).clamp(0, 255) as u8;
        x += 1;
    }
}

/// GDF residual add: `dst[x] = clip(dst[x] + sign(d)*((|d|+8)>>4), 0, 255)`,
/// `d = err[x]*scale`. `vcltq_s32(d, 0)` selects the negated magnitude.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) fn gdf_add_run_8bpc_neon(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let sc = vdupq_n_s32(scale);
    let rnd = vdupq_n_s32(8);
    let nsh4 = vdupq_n_s32(-4);
    let zero = vdupq_n_s32(0);
    let mut x = 0;
    while x + 8 <= n {
        let ew = unsafe { vmovl_s8(vld1_s8(err[x..].as_ptr())) };
        let e_lo = vmovl_s16(vget_low_s16(ew));
        let e_hi = vmovl_s16(vget_high_s16(ew));
        let diff_lo = vmulq_s32(e_lo, sc);
        let diff_hi = vmulq_s32(e_hi, sc);
        let mag_lo = vshlq_s32(vaddq_s32(vabsq_s32(diff_lo), rnd), nsh4);
        let mag_hi = vshlq_s32(vaddq_s32(vabsq_s32(diff_hi), rnd), nsh4);
        let adj_lo = vbslq_s32(vcltq_s32(diff_lo, zero), vnegq_s32(mag_lo), mag_lo);
        let adj_hi = vbslq_s32(vcltq_s32(diff_hi, zero), vnegq_s32(mag_hi), mag_hi);
        let dw = unsafe { vmovl_u8(vld1_u8(dst[x..].as_ptr())) };
        let r_lo = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(dw))), adj_lo);
        let r_hi = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(dw))), adj_hi);
        let r16 = vcombine_u16(vqmovun_s32(r_lo), vqmovun_s32(r_hi));
        unsafe {
            vst1_u8(dst[x..].as_mut_ptr(), vqmovn_u16(r16));
        }
        x += 8;
    }
    while x < n {
        let diff = err[x] as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let adj = if diff < 0 { -mag } else { mag };
        dst[x] = (dst[x] as i32 + adj).clamp(0, 255) as u8;
        x += 1;
    }
}

/// GDF gradient (NEON): per-column `|2*b - a - c|` (each `>> shift`) summed over
/// the 2 rows into 8 lanes, then pair-reduced to `ncells` cells via `vpaddq`.
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
        let bw = unsafe { vmovl_u8(vld1_u8(center_rows[y][bcol..].as_ptr())) };
        let aw = unsafe { vmovl_u8(vld1_u8(a_rows[y][acol..].as_ptr())) };
        let cw = unsafe { vmovl_u8(vld1_u8(c_rows[y][ccol..].as_ptr())) };
        let b_lo = vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(bw))), nsh);
        let b_hi = vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(bw))), nsh);
        let a_lo = vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(aw))), nsh);
        let a_hi = vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(aw))), nsh);
        let c_lo = vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(cw))), nsh);
        let c_hi = vshlq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(cw))), nsh);
        let t_lo = vsubq_s32(vsubq_s32(vaddq_s32(b_lo, b_lo), a_lo), c_lo);
        let t_hi = vsubq_s32(vsubq_s32(vaddq_s32(b_hi, b_hi), a_hi), c_hi);
        acc_lo = vaddq_s32(acc_lo, vabsq_s32(t_lo));
        acc_hi = vaddq_s32(acc_hi, vabsq_s32(t_hi));
    }
    // vpaddq pairs adjacent lanes: [a0+a1, a2+a3, b0+b1, b2+b3].
    let pair = vpaddq_s32(acc_lo, acc_hi);
    let mut out = [0i32; 4];
    unsafe {
        vst1q_s32(out.as_mut_ptr(), pair);
    }
    for k in 0..ncells {
        dst[base_cell + k][d] = out[k] as u16;
    }
}
