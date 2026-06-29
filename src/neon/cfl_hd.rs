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

use core::arch::aarch64::*;

use crate::cfl_dispatch::{CflAlphaAccumHbd, CflApplyHbd, CflMhccpPredHbd};
const CFL_FLT_TYPE_VSTRIP: u32 = 1;
const CFL_FLT_TYPE_GAUSS: u32 = 2;

#[inline(always)]
fn pad_bottom(plane: &mut [u16], row0: usize, stride: usize, w: usize, h: usize, ylim: usize) {
    debug_assert_ne!(ylim, 0);
    let src = row0 + (ylim - 1) * stride;
    for yy in ylim..h {
        let dst = row0 + yy * stride;
        plane.copy_within(src..src + w, dst);
    }
}

#[inline(always)]
fn load_u16x8(a: &[u16; 8]) -> uint16x8_t {
    unsafe { vld1q_u16(a.as_ptr()) }
}

#[inline(always)]
fn load_u16x4(a: &[u16; 4]) -> uint16x4_t {
    unsafe { vld1_u16(a.as_ptr()) }
}

#[inline(always)]
fn load_u16x8_tail4(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 4);
    let mut tmp = [0u16; 8];
    tmp[..4].copy_from_slice(&src[..4]);
    load_u16x8(&tmp)
}

#[inline(always)]
fn load_u16x8_tail2(src: &[u16]) -> uint16x8_t {
    debug_assert!(src.len() >= 2);
    let mut tmp = [0u16; 8];
    tmp[..2].copy_from_slice(&src[..2]);
    load_u16x8(&tmp)
}

#[inline(always)]
fn load_u16x4_tail2(src: &[u16]) -> uint16x4_t {
    debug_assert!(src.len() >= 2);
    let mut tmp = [0u16; 4];
    tmp[..2].copy_from_slice(&src[..2]);
    load_u16x4(&tmp)
}

#[inline(always)]
fn store_u16x4(a: &mut [u16; 4], v: uint16x4_t) {
    unsafe { vst1_u16(a.as_mut_ptr(), v) };
}

#[inline(always)]
fn store_u16x2(a: &mut [u16], v: uint16x4_t) {
    debug_assert!(a.len() >= 2);
    let mut tmp = [0u16; 4];
    unsafe { vst1_u16(tmp.as_mut_ptr(), v) };
    a[..2].copy_from_slice(&tmp[..2]);
}

#[inline(always)]
fn store_u16x1(a: &mut u16, v: uint16x4_t) {
    *a = unsafe { vget_lane_u16::<0>(v) };
}

#[inline]
#[target_feature(enable = "neon")]
fn even_u16x4(src: uint16x8_t) -> uint16x4_t {
    vget_low_u16(vuzp1q_u16(src, src))
}

#[inline]
#[target_feature(enable = "neon")]
fn odd_u16x4(src: uint16x8_t) -> uint16x4_t {
    vget_low_u16(vuzp2q_u16(src, src))
}

#[inline]
#[target_feature(enable = "neon")]
fn left_u16x4(src: uint16x8_t, prev: u16) -> uint16x4_t {
    let shifted = vextq_u16::<7>(vdupq_n_u16(prev), src);
    even_u16x4(shifted)
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_420_i32(top: uint16x8_t, bot: uint16x8_t, dc0v: int32x4_t) -> int32x4_t {
    let top = vpaddlq_u16(top);
    let bot = vpaddlq_u16(bot);
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<1>(vaddq_u32(top, bot))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_420_filter_i32<const FILTER: u32>(
    cur: uint16x8_t,
    top: uint16x8_t,
    bot: uint16x8_t,
    prev_cur: u16,
    prev_bot: u16,
    dc0v: int32x4_t,
) -> int32x4_t {
    if FILTER == CFL_FLT_TYPE_VSTRIP {
        let left_cur = vmovl_u16(left_u16x4(cur, prev_cur));
        let center_cur = vmovl_u16(even_u16x4(cur));
        let right_cur = vmovl_u16(odd_u16x4(cur));
        let left_bot = vmovl_u16(left_u16x4(bot, prev_bot));
        let center_bot = vmovl_u16(even_u16x4(bot));
        let right_bot = vmovl_u16(odd_u16x4(bot));

        let cur_sum = vaddq_u32(vaddq_u32(left_cur, vshlq_n_u32::<1>(center_cur)), right_cur);
        let bot_sum = vaddq_u32(vaddq_u32(left_bot, vshlq_n_u32::<1>(center_bot)), right_bot);
        vsubq_s32(vreinterpretq_s32_u32(vaddq_u32(cur_sum, bot_sum)), dc0v)
    } else if FILTER == CFL_FLT_TYPE_GAUSS {
        let left = vmovl_u16(left_u16x4(cur, prev_cur));
        let center = vmovl_u16(even_u16x4(cur));
        let right = vmovl_u16(odd_u16x4(cur));
        let top = vmovl_u16(even_u16x4(top));
        let bot = vmovl_u16(even_u16x4(bot));

        let sum = vaddq_u32(
            vaddq_u32(vaddq_u32(left, vshlq_n_u32::<2>(center)), right),
            vaddq_u32(top, bot),
        );
        vsubq_s32(vreinterpretq_s32_u32(sum), dc0v)
    } else {
        ac4_420_i32(cur, bot, dc0v)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_422_uniform_i32(src: uint16x8_t, dc0v: int32x4_t) -> int32x4_t {
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<2>(vpaddlq_u16(src))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_422_gauss_i32(src: uint16x8_t, dc0v: int32x4_t) -> int32x4_t {
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<3>(vmovl_u16(even_u16x4(src)))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_422_filter_i32<const FILTER: u32>(src: uint16x8_t, prev: u16, dc0v: int32x4_t) -> int32x4_t {
    if FILTER == CFL_FLT_TYPE_GAUSS {
        ac4_422_gauss_i32(src, dc0v)
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        let left = vmovl_u16(left_u16x4(src, prev));
        let center = vmovl_u16(even_u16x4(src));
        let right = vmovl_u16(odd_u16x4(src));
        let sum = vshlq_n_u32::<1>(vaddq_u32(vaddq_u32(left, vshlq_n_u32::<1>(center)), right));
        vsubq_s32(vreinterpretq_s32_u32(sum), dc0v)
    } else {
        ac4_422_uniform_i32(src, dc0v)
    }
}

#[inline(always)]
fn cfl_ac_420_hbd_scalar_filter<const FILTER: u32>(
    y: &[u16],
    yrow: usize,
    ystride: usize,
    cy: usize,
    x: usize,
    dc0: i32,
) -> i32 {
    let xl = x << 1;
    let left = ((xl as i32) & -64).max(xl as i32 - 1) as usize;
    if FILTER == CFL_FLT_TYPE_GAUSS {
        let top = if (cy & 31) == 0 {
            yrow + xl
        } else {
            yrow + xl - ystride
        };
        y[yrow + left] as i32
            + 4 * y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[top] as i32
            + y[yrow + xl + ystride] as i32
            - dc0
    } else if FILTER == CFL_FLT_TYPE_VSTRIP {
        y[yrow + left] as i32
            + 2 * y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[yrow + left + ystride] as i32
            + 2 * y[yrow + xl + ystride] as i32
            + y[yrow + xl + ystride + 1] as i32
            - dc0
    } else {
        ((y[yrow + xl] as i32
            + y[yrow + xl + 1] as i32
            + y[yrow + xl + ystride] as i32
            + y[yrow + xl + ystride + 1] as i32)
            << 1)
            - dc0
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn ac4_444_i32(src: uint16x4_t, dc0v: int32x4_t) -> int32x4_t {
    vsubq_s32(
        vreinterpretq_s32_u32(vshlq_n_u32::<3>(vmovl_u16(src))),
        dc0v,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn apply4_i32_ac(ac: int32x4_t, alpha: i32, dc_v: int32x4_t, max_v: int32x4_t) -> uint16x4_t {
    let diff = vmull_n_s16(vqmovn_s32(ac), alpha as i16);
    let mag = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff), vdupq_n_s32(1024)));
    let signed = vbslq_s32(vcltq_s32(diff, vdupq_n_s32(0)), vnegq_s32(mag), mag);
    let val = vminq_s32(vmaxq_s32(vaddq_s32(dc_v, signed), vdupq_n_s32(0)), max_v);
    vqmovun_s32(val)
}

#[target_feature(enable = "neon")]
fn cfl_apply_420_hbd_neon_impl<const FILTER: u32>(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let lfull = nfull * 8;

    let dc0v = vdupq_n_s32(dc0);
    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for cy in 0..ylim {
        let cur = y[yrow..yrow + lfull].as_chunks::<8>().0;
        let top_row = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
            yrow - ystride
        } else {
            yrow
        };
        let top = y[top_row..top_row + lfull].as_chunks::<8>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (i, (((du, dv), c), (t, b))) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(cur)
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let luma_x = i * 8;
                    let prev_cur = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let prev_bot = if (luma_x & 63) == 0 {
                        y[yrow + ystride + luma_x]
                    } else {
                        y[yrow + ystride + luma_x - 1]
                    };
                    let ac = ac4_420_filter_i32::<FILTER>(
                        load_u16x8(c),
                        load_u16x8(t),
                        load_u16x8(b),
                        prev_cur,
                        prev_bot,
                        dc0v,
                    );
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (i, ((du, c), (t, b))) in u_chunks
                    .iter_mut()
                    .zip(cur)
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let luma_x = i * 8;
                    let prev_cur = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let prev_bot = if (luma_x & 63) == 0 {
                        y[yrow + ystride + luma_x]
                    } else {
                        y[yrow + ystride + luma_x - 1]
                    };
                    let ac = ac4_420_filter_i32::<FILTER>(
                        load_u16x8(c),
                        load_u16x8(t),
                        load_u16x8(b),
                        prev_cur,
                        prev_bot,
                        dc0v,
                    );
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (i, ((dv, c), (t, b))) in v_chunks
                    .iter_mut()
                    .zip(cur)
                    .zip(top.iter().zip(bot.iter()))
                    .enumerate()
                {
                    let luma_x = i * 8;
                    let prev_cur = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let prev_bot = if (luma_x & 63) == 0 {
                        y[yrow + ystride + luma_x]
                    } else {
                        y[yrow + ystride + luma_x - 1]
                    };
                    let ac = ac4_420_filter_i32::<FILTER>(
                        load_u16x8(c),
                        load_u16x8(t),
                        load_u16x8(b),
                        prev_cur,
                        prev_bot,
                        dc0v,
                    );
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (false, false) => unreachable!(),
        }

        let x2full = xfull + ((xlim - xfull) / 2) * 2;
        let mut x = xfull;
        while x < x2full {
            let luma_x = x << 1;
            let top_row = if FILTER == CFL_FLT_TYPE_GAUSS && (cy & 31) != 0 {
                yrow - ystride
            } else {
                yrow
            };
            let prev_cur = if (luma_x & 63) == 0 {
                y[yrow + luma_x]
            } else {
                y[yrow + luma_x - 1]
            };
            let prev_bot = if (luma_x & 63) == 0 {
                y[yrow + ystride + luma_x]
            } else {
                y[yrow + ystride + luma_x - 1]
            };
            let ac = ac4_420_filter_i32::<FILTER>(
                load_u16x8_tail4(&y[yrow + luma_x..]),
                load_u16x8_tail4(&y[top_row + luma_x..]),
                load_u16x8_tail4(&y[yrow + ystride + luma_x..]),
                prev_cur,
                prev_bot,
                dc0v,
            );
            if do_u {
                store_u16x2(
                    &mut u[urow + x..urow + x + 2],
                    apply4_i32_ac(ac, alpha0, dc1v, max_v),
                );
            }
            if do_v {
                store_u16x2(
                    &mut v[vrow + x..vrow + x + 2],
                    apply4_i32_ac(ac, alpha1, dc2v, max_v),
                );
            }
            x += 2;
        }

        for x in x2full..xlim {
            let ac = cfl_ac_420_hbd_scalar_filter::<FILTER>(y, yrow, ystride, cy, x, dc0);
            if do_u {
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }
        yrow += ystride << 1;
        urow += cstride;
        vrow += cstride;
    }
    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[target_feature(enable = "neon")]
fn cfl_apply_422_hbd_neon_impl<const FILTER: u32>(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let lfull = nfull * 8;

    let dc0v = vdupq_n_s32(dc0);
    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + lfull].as_chunks::<8>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (i, ((du, dv), s)) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(src)
                    .enumerate()
                {
                    let s = load_u16x8(s);
                    let luma_x = i * 8;
                    let prev = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let ac = ac4_422_filter_i32::<FILTER>(s, prev, dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (i, (du, s)) in u_chunks.iter_mut().zip(src).enumerate() {
                    let s = load_u16x8(s);
                    let luma_x = i * 8;
                    let prev = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let ac = ac4_422_filter_i32::<FILTER>(s, prev, dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (i, (dv, s)) in v_chunks.iter_mut().zip(src).enumerate() {
                    let s = load_u16x8(s);
                    let luma_x = i * 8;
                    let prev = if (luma_x & 63) == 0 {
                        y[yrow + luma_x]
                    } else {
                        y[yrow + luma_x - 1]
                    };
                    let ac = ac4_422_filter_i32::<FILTER>(s, prev, dc0v);
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (false, false) => unreachable!(),
        }
        let x2full = xfull + ((xlim - xfull) / 2) * 2;
        let mut x = xfull;
        while x < x2full {
            let luma_x = x << 1;
            let prev = if (luma_x & 63) == 0 {
                y[yrow + luma_x]
            } else {
                y[yrow + luma_x - 1]
            };
            let ac =
                ac4_422_filter_i32::<FILTER>(load_u16x8_tail4(&y[yrow + luma_x..]), prev, dc0v);
            if do_u {
                store_u16x2(
                    &mut u[urow + x..urow + x + 2],
                    apply4_i32_ac(ac, alpha0, dc1v, max_v),
                );
            }
            if do_v {
                store_u16x2(
                    &mut v[vrow + x..vrow + x + 2],
                    apply4_i32_ac(ac, alpha1, dc2v, max_v),
                );
            }
            x += 2;
        }

        if x < xlim {
            let luma_x = x << 1;
            let prev = if (luma_x & 63) == 0 {
                y[yrow + luma_x]
            } else {
                y[yrow + luma_x - 1]
            };
            let ac =
                ac4_422_filter_i32::<FILTER>(load_u16x8_tail2(&y[yrow + luma_x..]), prev, dc0v);
            if do_u {
                store_u16x1(&mut u[urow + x], apply4_i32_ac(ac, alpha0, dc1v, max_v));
            }
            if do_v {
                store_u16x1(&mut v[vrow + x], apply4_i32_ac(ac, alpha1, dc2v, max_v));
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }
        yrow += ystride;
        urow += cstride;
        vrow += cstride;
    }
    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[target_feature(enable = "neon")]
fn cfl_apply_444_hbd_neon_impl(args: CflApplyHbd<'_>) {
    let CflApplyHbd {
        y,
        u,
        v,
        layout,
        area,
        params,
        bitdepth_max,
    } = args;
    let crate::cfl_dispatch::CflLayout {
        yrow0,
        urow0,
        vrow0,
        ystride,
        cstride,
    } = layout;
    let crate::cfl_dispatch::CflArea { w, h, xlim, ylim } = area;
    let crate::cfl_dispatch::CflParams {
        dc0,
        dc1,
        dc2,
        alpha0,
        alpha1,
        filter_type: _,
    } = params;

    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;
    if !do_u && !do_v {
        return;
    }

    let nfull = xlim / 4;
    let xfull = nfull * 4;
    let dc0v = vdupq_n_s32(dc0);
    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;
    for _y in 0..ylim {
        let src = y[yrow..yrow + xfull].as_chunks::<4>().0;
        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for ((du, dv), s) in u_chunks.iter_mut().zip(v_chunks.iter_mut()).zip(src) {
                    let ac = ac4_444_i32(load_u16x4(s), dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<4>().0;
                for (du, s) in u_chunks.iter_mut().zip(src) {
                    let ac = ac4_444_i32(load_u16x4(s), dc0v);
                    store_u16x4(du, apply4_i32_ac(ac, alpha0, dc1v, max_v));
                }
            }
            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<4>().0;
                for (dv, s) in v_chunks.iter_mut().zip(src) {
                    let ac = ac4_444_i32(load_u16x4(s), dc0v);
                    store_u16x4(dv, apply4_i32_ac(ac, alpha1, dc2v, max_v));
                }
            }
            (false, false) => unreachable!(),
        }
        let x2full = xfull + ((xlim - xfull) / 2) * 2;
        let mut x = xfull;
        while x < x2full {
            let ac = ac4_444_i32(load_u16x4_tail2(&y[yrow + x..]), dc0v);
            if do_u {
                store_u16x2(
                    &mut u[urow + x..urow + x + 2],
                    apply4_i32_ac(ac, alpha0, dc1v, max_v),
                );
            }
            if do_v {
                store_u16x2(
                    &mut v[vrow + x..vrow + x + 2],
                    apply4_i32_ac(ac, alpha1, dc2v, max_v),
                );
            }
            x += 2;
        }

        for x in x2full..xlim {
            let ac = ((y[yrow + x] as i32) << 3) - dc0;
            if do_u {
                u[urow + x] = crate::cfl_dispatch::predict_one_hbd(dc1, alpha0, ac, bitdepth_max);
            }
            if do_v {
                v[vrow + x] = crate::cfl_dispatch::predict_one_hbd(dc2, alpha1, ac, bitdepth_max);
            }
        }
        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }
        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }
        yrow += ystride;
        urow += cstride;
        vrow += cstride;
    }
    if do_u {
        pad_bottom(u, urow0, cstride, w, h, ylim);
    }
    if do_v {
        pad_bottom(v, vrow0, cstride, w, h, ylim);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_apply_420_hbd_neon(args: CflApplyHbd<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => cfl_apply_420_hbd_neon_impl::<CFL_FLT_TYPE_VSTRIP>(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_420_hbd_neon_impl::<CFL_FLT_TYPE_GAUSS>(args),
        _ => cfl_apply_420_hbd_neon_impl::<0>(args),
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_apply_422_hbd_neon(args: CflApplyHbd<'_>) {
    match args.params.filter_type {
        CFL_FLT_TYPE_VSTRIP => cfl_apply_422_hbd_neon_impl::<CFL_FLT_TYPE_VSTRIP>(args),
        CFL_FLT_TYPE_GAUSS => cfl_apply_422_hbd_neon_impl::<CFL_FLT_TYPE_GAUSS>(args),
        _ => cfl_apply_422_hbd_neon_impl::<0>(args),
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_apply_444_hbd_neon(args: CflApplyHbd<'_>) {
    cfl_apply_444_hbd_neon_impl(args)
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_mul32_hbd_neon(v: int32x4_t, alpha: i32) -> int32x4_t {
    let prod = vmulq_n_s32(v, alpha);
    let mag = vshrq_n_s32::<16>(vaddq_s32(vabsq_s32(prod), vdupq_n_s32(1 << 15)));
    vbslq_s32(vcltq_s32(prod, vdupq_n_s32(0)), vnegq_s32(mag), mag)
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_sqrnd_hbd_neon(v: int32x4_t, bitdepth: i32) -> int32x4_t {
    vshlq_s32(
        vaddq_s32(vmulq_s32(v, v), vdupq_n_s32(1 << (bitdepth - 1))),
        vdupq_n_s32(-bitdepth),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_pred4_hbd_neon(
    v0: int32x4_t,
    v1: int32x4_t,
    alpha: [i32; 3],
    a2v2: int32x4_t,
    bitdepth: i32,
) -> int32x4_t {
    vaddq_s32(
        vaddq_s32(
            mhccp_mul32_hbd_neon(v0, alpha[0]),
            mhccp_mul32_hbd_neon(mhccp_sqrnd_hbd_neon(v1, bitdepth), alpha[1]),
        ),
        a2v2,
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_load_u16x8_i32_halves(src: &[u16; 8]) -> (int32x4_t, int32x4_t) {
    let v = unsafe { vld1q_u16(src.as_ptr()) };
    (
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))),
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn mhccp_store_u16x8_neon(dst: &mut [u16; 8], lo: int32x4_t, hi: int32x4_t, max_v: int32x4_t) {
    let zero = vdupq_n_s32(0);
    let lo = vminq_s32(vmaxq_s32(lo, zero), max_v);
    let hi = vminq_s32(vmaxq_s32(hi, zero), max_v);
    unsafe {
        vst1q_u16(
            dst.as_mut_ptr(),
            vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)),
        )
    };
}

#[inline(always)]
fn accum_alpha_u16x8(acc: uint32x4_t, v: uint16x8_t, m: uint16x8_t) -> uint32x4_t {
    unsafe {
        let acc = vmlal_u16(acc, vget_low_u16(v), vget_low_u16(m));
        vmlal_u16(acc, vget_high_u16(v), vget_high_u16(m))
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_alpha_accum_hbd_neon(args: CflAlphaAccumHbd<'_>) {
    if args.sample_stride != 1 || args.len < 8 {
        crate::cfl_dispatch::cfl_alpha_accum_hbd_scalar(args);
        return;
    }

    let CflAlphaAccumHbd {
        alpha,
        samples,
        sample_off,
        sample_stride: _,
        imat0,
        imat1,
        imat_off,
        len,
        a2sh,
    } = args;

    let mut acc0 = vdupq_n_u32(0);
    let mut acc1 = vdupq_n_u32(0);
    let mut acc2 = vdupq_n_u32(0);
    let mut processed = 0usize;

    let (sample_chunks, sample_rem) = samples[sample_off..sample_off + len].as_chunks::<8>();
    for (chunk_idx, s) in sample_chunks.iter().enumerate() {
        let i = imat_off + chunk_idx * 8;
        let v = load_u16x8(s);
        let m0 = load_u16x8((&imat0[i..i + 8]).try_into().unwrap());
        let m1 = load_u16x8((&imat1[i..i + 8]).try_into().unwrap());
        acc0 = accum_alpha_u16x8(acc0, v, m0);
        acc1 = accum_alpha_u16x8(acc1, v, m1);
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_low_u16(v)));
        acc2 = vaddq_u32(acc2, vmovl_u16(vget_high_u16(v)));
    }
    processed += sample_chunks.len() * 8;

    alpha[0] += vaddvq_u32(acc0) as i32;
    alpha[1] += vaddvq_u32(acc1) as i32;
    alpha[2] += (vaddvq_u32(acc2) as i32) << a2sh;

    if !sample_rem.is_empty() {
        crate::cfl_dispatch::cfl_alpha_accum_hbd_scalar(crate::cfl_dispatch::CflAlphaAccumHbd {
            alpha,
            samples,
            sample_off: sample_off + processed,
            sample_stride: 1,
            imat0,
            imat1,
            imat_off: imat_off + processed,
            len: sample_rem.len(),
            a2sh,
        });
    }
}

#[inline(always)]
fn mhccp_pred_one_hbd_neon(
    alpha: &[i32; 3],
    a2v2: i32,
    v0: i32,
    v1: i32,
    bitdepth: i32,
    bitdepth_max: i32,
) -> u16 {
    let sq = (v1 * v1 + (1 << (bitdepth - 1))) >> bitdepth;
    (crate::ipred::mul32(alpha[0], v0, 16) + crate::ipred::mul32(alpha[1], sq, 16) + a2v2)
        .clamp(0, bitdepth_max) as u16
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_mhccp_pred_hbd_neon(args: CflMhccpPredHbd<'_>) {
    if !crate::cfl_dispatch::cfl_mhccp_coeffs_fit_fast_mul(&args.alpha)
        || args.w < 8
        || args.bitdepth > 12
    {
        crate::cfl_dispatch::cfl_mhccp_pred_hbd_scalar(args);
        return;
    }

    let CflMhccpPredHbd {
        dst,
        dst_stride,
        src,
        src_off,
        src_top_stride,
        w,
        h,
        alpha,
        edge_flags,
        dir,
        bitdepth,
        bitdepth_max,
    } = args;
    let has_t = edge_flags & (1 << 2) != 0;
    let has_l = edge_flags & (1 << 3) != 0;
    let dir_t = dir == crate::levels::CflMhDir::Top;
    let dir_l = dir == crate::levels::CflMhDir::Left;
    let n_top = if has_t { 1 + dir_t as usize } else { 0 };
    let n_left = if has_l { 1 + dir_l as usize } else { 0 };
    let left_off = src_off + 64 * 64 + n_left * n_top;
    let mid = 1 << (bitdepth - 1);
    let a2v2_scalar = crate::ipred::mul32(alpha[2], mid, 16);
    let a2v2 = vdupq_n_s32(a2v2_scalar);
    let max_v = vdupq_n_s32(bitdepth_max);

    let mut sp = src_off;
    let mut y = 0usize;
    if dir_t && has_t && y < h {
        let dst_row = &mut dst[..w];
        let (dst_chunks, dst_tail) = dst_row.as_chunks_mut::<8>();
        let prev = sp - src_top_stride;
        for (i, chunk) in dst_chunks.iter_mut().enumerate() {
            let x = i * 8;
            let (v0_lo, v0_hi) =
                mhccp_load_u16x8_i32_halves((&src[prev + x..prev + x + 8]).try_into().unwrap());
            let (v1_lo, v1_hi) =
                mhccp_load_u16x8_i32_halves((&src[sp + x..sp + x + 8]).try_into().unwrap());
            mhccp_store_u16x8_neon(
                chunk,
                mhccp_pred4_hbd_neon(v0_lo, v1_lo, alpha, a2v2, bitdepth),
                mhccp_pred4_hbd_neon(v0_hi, v1_hi, alpha, a2v2, bitdepth),
                max_v,
            );
        }
        let done = dst_chunks.len() * 8;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            *d = mhccp_pred_one_hbd_neon(
                &alpha,
                a2v2_scalar,
                src[prev + x] as i32,
                src[sp + x] as i32,
                bitdepth,
                bitdepth_max,
            );
        }
        sp += w;
        y = 1;
    }

    for (row_y, dst_row) in dst.chunks_mut(dst_stride).take(h).enumerate().skip(y) {
        let dst_row = &mut dst_row[..w];
        let mut x0 = 0usize;
        if dir_l {
            let v0 = if has_l {
                src[left_off + row_y * n_left + 1] as i32
            } else {
                src[sp] as i32
            };
            dst_row[0] = mhccp_pred_one_hbd_neon(
                &alpha,
                a2v2_scalar,
                v0,
                src[sp] as i32,
                bitdepth,
                bitdepth_max,
            );
            x0 = 1;
        }
        let (dst_chunks, dst_tail) = dst_row[x0..].as_chunks_mut::<8>();
        for (i, chunk) in dst_chunks.iter_mut().enumerate() {
            let x = x0 + i * 8;
            let v0_off = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x - 1
            } else {
                sp + x
            };
            let (v0_lo, v0_hi) =
                mhccp_load_u16x8_i32_halves((&src[v0_off..v0_off + 8]).try_into().unwrap());
            let (v1_lo, v1_hi) =
                mhccp_load_u16x8_i32_halves((&src[sp + x..sp + x + 8]).try_into().unwrap());
            mhccp_store_u16x8_neon(
                chunk,
                mhccp_pred4_hbd_neon(v0_lo, v1_lo, alpha, a2v2, bitdepth),
                mhccp_pred4_hbd_neon(v0_hi, v1_hi, alpha, a2v2, bitdepth),
                max_v,
            );
        }
        let done = x0 + dst_chunks.len() * 8;
        for (x, d) in (done..w).zip(dst_tail.iter_mut()) {
            let v0_idx = if dir_t {
                sp + x - ((((row_y > 0) as usize) | has_t as usize) * w)
            } else if dir_l {
                sp + x.saturating_sub(1)
            } else {
                sp + x
            };
            *d = mhccp_pred_one_hbd_neon(
                &alpha,
                a2v2_scalar,
                src[v0_idx] as i32,
                src[sp + x] as i32,
                bitdepth,
                bitdepth_max,
            );
        }
        sp += w;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn pair_madd_u16x8(v: uint16x8_t, weights: uint16x4_t) -> uint32x4_t {
    let lo = vmull_u16(vget_low_u16(v), weights);
    let hi = vmull_u16(vget_high_u16(v), weights);
    vcombine_u32(
        vpadd_u32(vget_low_u32(lo), vget_high_u32(lo)),
        vpadd_u32(vget_low_u32(hi), vget_high_u32(hi)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn gen_y4_hbd_u16x4<const FILTER: i32>(
    src: &[u16],
    src_off: usize,
    top: &[u16],
    top_off: usize,
    bottom_offset: usize,
    x: usize,
) -> uint16x4_t {
    let xl = x << 1;
    let out = if FILTER == 1 {
        let left_w = vcreate_u16(0x0000_0001_0000_0001);
        let center_right_w = vcreate_u16(0x0001_0002_0001_0002);
        let cur_left = load_u16x8(
            src[src_off + xl - 1..src_off + xl - 1 + 8]
                .try_into()
                .unwrap(),
        );
        let cur_center = load_u16x8(src[src_off + xl..src_off + xl + 8].try_into().unwrap());
        let bot_left = load_u16x8(
            src[src_off + bottom_offset + xl - 1..src_off + bottom_offset + xl - 1 + 8]
                .try_into()
                .unwrap(),
        );
        let bot_center = load_u16x8(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 8]
                .try_into()
                .unwrap(),
        );
        let cur = vaddq_u32(
            pair_madd_u16x8(cur_left, left_w),
            pair_madd_u16x8(cur_center, center_right_w),
        );
        let bot = vaddq_u32(
            pair_madd_u16x8(bot_left, left_w),
            pair_madd_u16x8(bot_center, center_right_w),
        );
        vshrq_n_u32::<3>(vaddq_u32(cur, bot))
    } else if FILTER == 2 {
        let left_w = vcreate_u16(0x0000_0001_0000_0001);
        let center_right_w = vcreate_u16(0x0001_0004_0001_0004);
        let center_w = left_w;
        let cur_left = load_u16x8(
            src[src_off + xl - 1..src_off + xl - 1 + 8]
                .try_into()
                .unwrap(),
        );
        let cur_center = load_u16x8(src[src_off + xl..src_off + xl + 8].try_into().unwrap());
        let top_c = load_u16x8(top[top_off + xl..top_off + xl + 8].try_into().unwrap());
        let bot_c = load_u16x8(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 8]
                .try_into()
                .unwrap(),
        );
        let cur = vaddq_u32(
            pair_madd_u16x8(cur_left, left_w),
            pair_madd_u16x8(cur_center, center_right_w),
        );
        let tb = vaddq_u32(
            pair_madd_u16x8(top_c, center_w),
            pair_madd_u16x8(bot_c, center_w),
        );
        vshrq_n_u32::<3>(vaddq_u32(cur, tb))
    } else {
        let ones = vdup_n_u16(1);
        let cur = load_u16x8(src[src_off + xl..src_off + xl + 8].try_into().unwrap());
        let bot = load_u16x8(
            src[src_off + bottom_offset + xl..src_off + bottom_offset + xl + 8]
                .try_into()
                .unwrap(),
        );
        vshrq_n_u32::<2>(vaddq_u32(
            pair_madd_u16x8(cur, ones),
            pair_madd_u16x8(bot, ones),
        ))
    };
    vmovn_u32(out)
}

#[target_feature(enable = "neon")]
fn cfl_gen_y_row_hbd_neon_impl<const FILTER: i32>(args: crate::cfl_dispatch::CflGenYRowHbd<'_>) {
    let crate::cfl_dispatch::CflGenYRowHbd {
        dst,
        src,
        src_off,
        top,
        top_off,
        bottom_offset,
        n_left,
        filter_type: _,
    } = args;

    let mut processed = 0usize;
    if FILTER != 0 && n_left == 0 && !dst.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_hbd_scalar(crate::cfl_dispatch::CflGenYRowHbd {
            dst: &mut dst[..1],
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left,
            filter_type: FILTER,
        });
        processed = 1;
    }

    let (chunks, rem) = dst[processed..].as_chunks_mut::<4>();
    for (chunk_idx, chunk) in chunks.iter_mut().enumerate() {
        let x = n_left + processed + chunk_idx * 4;
        store_u16x4(
            chunk,
            gen_y4_hbd_u16x4::<FILTER>(src, src_off, top, top_off, bottom_offset, x),
        );
    }
    processed += chunks.len() * 4;

    if !rem.is_empty() {
        crate::cfl_dispatch::cfl_gen_y_row_hbd_scalar(crate::cfl_dispatch::CflGenYRowHbd {
            dst: rem,
            src,
            src_off,
            top,
            top_off,
            bottom_offset,
            n_left: n_left + processed,
            filter_type: FILTER,
        });
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_gen_y_row_hbd_neon(args: crate::cfl_dispatch::CflGenYRowHbd<'_>) {
    match args.filter_type {
        1 => cfl_gen_y_row_hbd_neon_impl::<1>(args),
        2 => cfl_gen_y_row_hbd_neon_impl::<2>(args),
        _ => cfl_gen_y_row_hbd_neon_impl::<0>(args),
    }
}
