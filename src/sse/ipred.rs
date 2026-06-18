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

use crate::intops::ulog2;
use crate::levels::ANGLE_MULTI_MRL_FLAG;
use crate::tables::SM_WEIGHTS;

#[inline(always)]
fn sra_i16(v: __m128i, shift: i32) -> __m128i {
    unsafe { _mm_sra_epi16(v, _mm_cvtsi32_si128(shift)) }
}

#[target_feature(enable = "sse4.1")]
fn ipred_v_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if width < 16 {
        crate::ipred_dispatch::ipred_v_scalar(dst, stride, tl, o, width, height, angle);
        return;
    }

    if angle & ANGLE_MULTI_MRL_FLAG != 0 {
        let e_stride = (width + height) * 2 + 1;
        let top1 = &tl[o + 1..o + 1 + width];
        let top2 = &tl[o + 1 + e_stride..o + 1 + e_stride + width];
        let (dc, drem) = dst[..width].as_chunks_mut::<16>();
        for ((d, a), b) in dc
            .iter_mut()
            .zip(top1.as_chunks::<16>().0.iter())
            .zip(top2.as_chunks::<16>().0.iter())
        {
            store_u8x16_fixed(d, _mm_avg_epu8(load_u8x16_fixed(a), load_u8x16_fixed(b)));
        }
        let base_x = (width / 16) * 16;
        for (xi, d) in drem.iter_mut().enumerate() {
            let x = base_x + xi;
            let t1 = tl[o + 1 + x] as u16;
            let t2 = tl[o + 1 + e_stride + x] as u16;
            *d = ((t1 + t2 + 1) >> 1) as u8;
        }
    } else {
        let top = &tl[o + 1..o + 1 + width];
        dst[..width].copy_from_slice(top);
    }

    let mut off = stride;
    for _ in 1..height {
        dst.copy_within(0..width, off);
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_h_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    if width < 16 {
        crate::ipred_dispatch::ipred_h_scalar(dst, stride, tl, o, width, height, angle);
        return;
    }

    let e_stride = (width + height) * 2 + 1;
    let mrl = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let mut off = 0usize;
    for y in 0..height {
        let v = if mrl {
            let left = tl[o - 1 - y] as u16;
            let left2 = tl[o + e_stride - 1 - y] as u16;
            ((left + left2 + 1) >> 1) as u8
        } else {
            tl[o - 1 - y]
        };
        let row = &mut dst[off..off + width];
        row.fill(v);
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_smooth_v_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_v_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bhl2 = ulog2(h as u32) as i32;
    let rnd = _mm_set1_epi16((h >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let bottom = tl[o - h - 1] as i16;
    let bottom_v = _mm_set1_epi16(bottom);
    let sh = bhl2;

    let mut off = 0usize;
    for y in 0..h {
        let off_y = _mm_set1_epi16((h - 1 - y) as i16);
        let w_ver = _mm_set1_epi16(weights[y] as i16);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let add32 = _mm_set1_epi16(32);
        let (c16, r16) = row.as_chunks_mut::<16>();
        for (d, t) in c16.iter_mut().zip(top_src.as_chunks::<16>().0.iter()) {
            let a0 = load_u8x8_i16_fixed((&t[..8]).try_into().unwrap());
            let a1 = load_u8x8_i16_fixed((&t[8..]).try_into().unwrap());
            let mul0 = _mm_mullo_epi16(_mm_sub_epi16(a0, bottom_v), off_y);
            let pred0 = _mm_add_epi16(bottom_v, sra_i16(_mm_add_epi16(mul0, rnd), sh));
            let adj0 = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(a0, pred0), w_ver),
                add32,
            ));
            let mul1 = _mm_mullo_epi16(_mm_sub_epi16(a1, bottom_v), off_y);
            let pred1 = _mm_add_epi16(bottom_v, sra_i16(_mm_add_epi16(mul1, rnd), sh));
            let adj1 = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(a1, pred1), w_ver),
                add32,
            ));
            store_i16x8x2_u8_fixed(d, _mm_add_epi16(pred0, adj0), _mm_add_epi16(pred1, adj1));
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (d, t) in c8.iter_mut().zip(top_src[done..].as_chunks::<8>().0.iter()) {
            let above = load_u8x8_i16_fixed(t);
            let mul = _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v), off_y);
            let pred = _mm_add_epi16(bottom_v, sra_i16(_mm_add_epi16(mul, rnd), sh));
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(above, pred), w_ver),
                add32,
            ));
            store_i16x8_u8_fixed(d, _mm_add_epi16(pred, adj));
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let x = base_x + xi;
            let above = tl[o + 1 + x] as i32;
            let mul = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let pred = bottom as i32 + ((mul + (h >> 1) as i32) >> bhl2);
            *d = (pred + (((above - pred) * weights[y] as i32 + 32) >> 6)) as u8;
        }
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_smooth_h_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_h_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let rnd = _mm_set1_epi16((w >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let right_v = _mm_set1_epi16(right);

    let mut off = 0usize;
    for y in 0..h {
        let left = tl[o - 1 - y] as i16;
        let left_v = _mm_set1_epi16(left);
        let diff = _mm_set1_epi16(left - right);
        let row = &mut dst[off..off + w];
        let add32 = _mm_set1_epi16(32);
        let (c16, r16) = row.as_chunks_mut::<16>();
        for (ci, (d, wxc)) in c16
            .iter_mut()
            .zip(weights[..w].as_chunks::<16>().0.iter())
            .enumerate()
        {
            let x = ci * 16;
            let xlo = (w - 1 - x) as i16;
            let xhi = (w - 1 - x - 8) as i16;
            let dist0 = _mm_setr_epi16(
                xlo,
                xlo - 1,
                xlo - 2,
                xlo - 3,
                xlo - 4,
                xlo - 5,
                xlo - 6,
                xlo - 7,
            );
            let dist1 = _mm_setr_epi16(
                xhi,
                xhi - 1,
                xhi - 2,
                xhi - 3,
                xhi - 4,
                xhi - 5,
                xhi - 6,
                xhi - 7,
            );
            let wx0 = load_u8x8_i16_fixed((&wxc[..8]).try_into().unwrap());
            let wx1 = load_u8x8_i16_fixed((&wxc[8..]).try_into().unwrap());
            let pred0 = _mm_add_epi16(
                right_v,
                sra_i16(_mm_add_epi16(_mm_mullo_epi16(diff, dist0), rnd), bwl2),
            );
            let adj0 = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(left_v, pred0), wx0),
                add32,
            ));
            let pred1 = _mm_add_epi16(
                right_v,
                sra_i16(_mm_add_epi16(_mm_mullo_epi16(diff, dist1), rnd), bwl2),
            );
            let adj1 = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(left_v, pred1), wx1),
                add32,
            ));
            store_i16x8x2_u8_fixed(d, _mm_add_epi16(pred0, adj0), _mm_add_epi16(pred1, adj1));
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (ci, (d, wxc)) in c8
            .iter_mut()
            .zip(weights[done..w].as_chunks::<8>().0.iter())
            .enumerate()
        {
            let x0 = (w - 1 - done - ci * 8) as i16;
            let dist = _mm_setr_epi16(x0, x0 - 1, x0 - 2, x0 - 3, x0 - 4, x0 - 5, x0 - 6, x0 - 7);
            let wx = load_u8x8_i16_fixed(wxc);
            let pred = _mm_add_epi16(
                right_v,
                sra_i16(_mm_add_epi16(_mm_mullo_epi16(diff, dist), rnd), bwl2),
            );
            let adj = _mm_srai_epi16::<6>(_mm_add_epi16(
                _mm_mullo_epi16(_mm_sub_epi16(left_v, pred), wx),
                add32,
            ));
            store_i16x8_u8_fixed(d, _mm_add_epi16(pred, adj));
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let x = base_x + xi;
            let mul = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let pred = right as i32 + ((mul + (w >> 1) as i32) >> bwl2);
            *d = (pred + (((left as i32 - pred) * weights[x] as i32 + 32) >> 6)) as u8;
        }
        off += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_smooth_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    if w < 8 {
        crate::ipred_dispatch::ipred_smooth_scalar(dst, stride, tl, o, w, h);
        return;
    }

    let bwl2 = ulog2(w as u32) as i32;
    let bhl2 = ulog2(h as u32) as i32;
    let rnd_ver = _mm_set1_epi16((h >> 1) as i16);
    let rnd_hor = _mm_set1_epi16((w >> 1) as i16);
    let n_pel = w * h;
    let scale = (n_pel >= 64) as usize + (n_pel > 512) as usize;
    let weights = &SM_WEIGHTS[scale];
    let right = tl[o + w + 1] as i16;
    let bottom = tl[o - h - 1] as i16;
    let right_v = _mm_set1_epi16(right);
    let bottom_v = _mm_set1_epi16(bottom);

    let mut off = 0usize;
    for y in 0..h {
        let left = tl[o - 1 - y] as i16;
        let left_v = _mm_set1_epi16(left);
        let diff_hor = _mm_set1_epi16(left - right);
        let off_ver = _mm_set1_epi16((h - 1 - y) as i16);
        let w_ver = _mm_set1_epi16(weights[y] as i16);
        let row = &mut dst[off..off + w];
        let top_src = &tl[o + 1..o + 1 + w];
        let add32 = _mm_set1_epi16(32);
        let one = _mm_set1_epi16(1);
        let (c16, r16) = row.as_chunks_mut::<16>();
        for (ci, ((d, t), wxc)) in c16
            .iter_mut()
            .zip(top_src.as_chunks::<16>().0.iter())
            .zip(weights[..w].as_chunks::<16>().0.iter())
            .enumerate()
        {
            let x = ci * 16;
            let xlo = (w - 1 - x) as i16;
            let xhi = (w - 1 - x - 8) as i16;
            let dist0 = _mm_setr_epi16(
                xlo,
                xlo - 1,
                xlo - 2,
                xlo - 3,
                xlo - 4,
                xlo - 5,
                xlo - 6,
                xlo - 7,
            );
            let dist1 = _mm_setr_epi16(
                xhi,
                xhi - 1,
                xhi - 2,
                xhi - 3,
                xhi - 4,
                xhi - 5,
                xhi - 6,
                xhi - 7,
            );
            let a0 = load_u8x8_i16_fixed((&t[..8]).try_into().unwrap());
            let a1 = load_u8x8_i16_fixed((&t[8..]).try_into().unwrap());
            let wx0 = load_u8x8_i16_fixed((&wxc[..8]).try_into().unwrap());
            let wx1 = load_u8x8_i16_fixed((&wxc[8..]).try_into().unwrap());

            let pv0 = _mm_add_epi16(
                bottom_v,
                sra_i16(
                    _mm_add_epi16(
                        _mm_mullo_epi16(_mm_sub_epi16(a0, bottom_v), off_ver),
                        rnd_ver,
                    ),
                    bhl2,
                ),
            );
            let ph0 = _mm_add_epi16(
                right_v,
                sra_i16(
                    _mm_add_epi16(_mm_mullo_epi16(diff_hor, dist0), rnd_hor),
                    bwl2,
                ),
            );
            let pv0 = _mm_add_epi16(
                pv0,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(a0, pv0), w_ver),
                    add32,
                )),
            );
            let ph0 = _mm_add_epi16(
                ph0,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(left_v, ph0), wx0),
                    add32,
                )),
            );
            let out0 = _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(pv0, ph0), one));

            let pv1 = _mm_add_epi16(
                bottom_v,
                sra_i16(
                    _mm_add_epi16(
                        _mm_mullo_epi16(_mm_sub_epi16(a1, bottom_v), off_ver),
                        rnd_ver,
                    ),
                    bhl2,
                ),
            );
            let ph1 = _mm_add_epi16(
                right_v,
                sra_i16(
                    _mm_add_epi16(_mm_mullo_epi16(diff_hor, dist1), rnd_hor),
                    bwl2,
                ),
            );
            let pv1 = _mm_add_epi16(
                pv1,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(a1, pv1), w_ver),
                    add32,
                )),
            );
            let ph1 = _mm_add_epi16(
                ph1,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(left_v, ph1), wx1),
                    add32,
                )),
            );
            let out1 = _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(pv1, ph1), one));

            store_i16x8x2_u8_fixed(d, out0, out1);
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (ci, ((d, t), wxc)) in c8
            .iter_mut()
            .zip(top_src[done..].as_chunks::<8>().0.iter())
            .zip(weights[done..w].as_chunks::<8>().0.iter())
            .enumerate()
        {
            let above = load_u8x8_i16_fixed(t);
            let x0 = (w - 1 - done - ci * 8) as i16;
            let dist = _mm_setr_epi16(x0, x0 - 1, x0 - 2, x0 - 3, x0 - 4, x0 - 5, x0 - 6, x0 - 7);
            let wx = load_u8x8_i16_fixed(wxc);

            let pred_ver = _mm_add_epi16(
                bottom_v,
                sra_i16(
                    _mm_add_epi16(
                        _mm_mullo_epi16(_mm_sub_epi16(above, bottom_v), off_ver),
                        rnd_ver,
                    ),
                    bhl2,
                ),
            );
            let pred_hor = _mm_add_epi16(
                right_v,
                sra_i16(
                    _mm_add_epi16(_mm_mullo_epi16(diff_hor, dist), rnd_hor),
                    bwl2,
                ),
            );
            let pred_ver = _mm_add_epi16(
                pred_ver,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(above, pred_ver), w_ver),
                    add32,
                )),
            );
            let pred_hor = _mm_add_epi16(
                pred_hor,
                _mm_srai_epi16::<6>(_mm_add_epi16(
                    _mm_mullo_epi16(_mm_sub_epi16(left_v, pred_hor), wx),
                    add32,
                )),
            );
            store_i16x8_u8_fixed(
                d,
                _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(pred_ver, pred_hor), one)),
            );
        }
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let x = base_x + xi;
            let above = tl[o + 1 + x] as i32;
            let mul_ver = (above - bottom as i32) * (h as i32 - 1 - y as i32);
            let mul_hor = (left as i32 - right as i32) * (w as i32 - 1 - x as i32);
            let mut pred_ver = bottom as i32 + ((mul_ver + (h >> 1) as i32) >> bhl2);
            let mut pred_hor = right as i32 + ((mul_hor + (w >> 1) as i32) >> bwl2);
            pred_ver += ((above - pred_ver) * weights[y] as i32 + 32) >> 6;
            pred_hor += ((left as i32 - pred_hor) * weights[x] as i32 + 32) >> 6;
            *d = ((pred_ver + pred_hor + 1) >> 1) as u8;
        }
        off += stride;
    }
}

pub(crate) fn ipred_v_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    unsafe { ipred_v_8bpc_sse41_impl(dst, stride, tl, o, width, height, angle) }
}

pub(crate) fn ipred_h_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    unsafe { ipred_h_8bpc_sse41_impl(dst, stride, tl, o, width, height, angle) }
}

pub(crate) fn ipred_smooth_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    unsafe { ipred_smooth_8bpc_sse41_impl(dst, stride, tl, o, width, height) }
}

pub(crate) fn ipred_smooth_v_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    unsafe { ipred_smooth_v_8bpc_sse41_impl(dst, stride, tl, o, width, height) }
}

pub(crate) fn ipred_smooth_h_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
) {
    unsafe { ipred_smooth_h_8bpc_sse41_impl(dst, stride, tl, o, width, height) }
}

use crate::levels::ANGLE_IBP_FLAG;

#[inline(always)]
fn load_u8x16_fixed(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
fn store_u8x16_fixed(a: &mut [u8; 16], v: __m128i) {
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, v) };
}

/// Horizontal sum of all bytes in `s` (each lane widened to u32).
#[inline]
#[target_feature(enable = "sse4.1")]
fn sum_u8_sse41(s: &[u8]) -> u32 {
    let zero = _mm_setzero_si128();
    let mut acc = zero;
    let (chunks, rem) = s.as_chunks::<16>();
    for c in chunks.iter() {
        // SAD against zero == sum of the 16 bytes, placed in lanes 0 and 2.
        acc = _mm_add_epi64(acc, _mm_sad_epu8(load_u8x16_fixed(c), zero));
    }
    let mut total = (_mm_extract_epi32::<0>(acc) + _mm_extract_epi32::<2>(acc)) as u32;
    for &b in rem {
        total += b as u32;
    }
    total
}

/// Fill a `w x h` block at `off` with the constant byte `dc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn splat_fill_sse41(dst: &mut [u8], stride: usize, off: usize, w: usize, h: usize, dc: u8) {
    let v = _mm_set1_epi8(dc as i8);
    let mut p = off;
    for _ in 0..h {
        let (chunks, rem) = dst[p..p + w].as_chunks_mut::<16>();
        for c in chunks.iter_mut() {
            store_u8x16_fixed(c, v);
        }
        rem.fill(dc);
        p += stride;
    }
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_128_8bpc_sse41_impl(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    splat_fill_sse41(dst, stride, 0, w, h, 128);
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_top_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_top_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let sum = sum_u8_sse41(&tl[o + 1..o + 1 + w]);
    let dc = (((w >> 1) as u32 + sum) >> (w as u32).trailing_zeros()) as u8;
    splat_fill_sse41(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_left_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_left_8bpc(dst, stride, tl, o, w, h, angle);
    }
    // Left samples tl[o-1-i], i in 0..h, are the contiguous slice tl[o-h..o]
    // (order does not matter for a sum).
    let sum = sum_u8_sse41(&tl[o - h..o]);
    let dc = (((h >> 1) as u32 + sum) >> (h as u32).trailing_zeros()) as u8;
    splat_fill_sse41(dst, stride, 0, w, h, dc);
}

#[target_feature(enable = "sse4.1")]
fn ipred_dc_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    if angle & ANGLE_IBP_FLAG != 0 {
        return crate::ipred::ipred_dc_8bpc(dst, stride, tl, o, w, h, angle);
    }
    let n_pel = (w + h) as u32;
    let sum = sum_u8_sse41(&tl[o + 1..o + 1 + w]) + sum_u8_sse41(&tl[o - h..o]);
    let dc = if n_pel & (n_pel - 1) == 0 {
        (sum + w as u32) >> n_pel.trailing_zeros()
    } else {
        crate::ipred::fast_div32_dc(sum, n_pel).min(255)
    } as u8;
    splat_fill_sse41(dst, stride, 0, w, h, dc);
}

pub(crate) fn ipred_dc_128_8bpc_sse41(dst: &mut [u8], stride: usize, w: usize, h: usize) {
    unsafe { ipred_dc_128_8bpc_sse41_impl(dst, stride, w, h) }
}

pub(crate) fn ipred_dc_top_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_top_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle) }
}

pub(crate) fn ipred_dc_left_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_left_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle) }
}

pub(crate) fn ipred_dc_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
) {
    unsafe { ipred_dc_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle) }
}

#[inline(always)]
fn load_u8x8_i16_fixed(a: &[u8; 8]) -> __m128i {
    unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(a.as_ptr() as *const __m128i)) }
}

#[inline(always)]
fn store_i16x8_u8_fixed(a: &mut [u8; 8], v: __m128i) {
    let packed = unsafe { _mm_packus_epi16(v, _mm_setzero_si128()) };
    unsafe { _mm_storel_epi64(a.as_mut_ptr() as *mut __m128i, packed) };
}

/// Pack two i16x8 lanes (saturating to u8) and store 16 bytes at once.
#[inline(always)]
fn store_i16x8x2_u8_fixed(a: &mut [u8; 16], lo: __m128i, hi: __m128i) {
    let packed = unsafe { _mm_packus_epi16(lo, hi) };
    unsafe { _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, packed) };
}

#[target_feature(enable = "sse4.1")]
fn ipred_paeth_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    let topleft = tl[o] as i32;
    if w < 8 {
        return crate::ipred::ipred_paeth_8bpc(dst, stride, tl, o, w, h);
    }
    let tl_v = _mm_set1_epi16(topleft as i16);
    let mut off = 0;
    for y in 0..h {
        let left = tl[o - 1 - y] as i32;
        let left_v = _mm_set1_epi16(left as i16);
        let top_src = &tl[o + 1..o + 1 + w];
        let (c16, r16) = dst[off..off + w].as_chunks_mut::<16>();
        for (d, t) in c16.iter_mut().zip(top_src.as_chunks::<16>().0.iter()) {
            let t0 = load_u8x8_i16_fixed((&t[..8]).try_into().unwrap());
            let t1 = load_u8x8_i16_fixed((&t[8..]).try_into().unwrap());
            let base0 = _mm_sub_epi16(_mm_add_epi16(left_v, t0), tl_v);
            let ld0 = _mm_abs_epi16(_mm_sub_epi16(left_v, base0));
            let td0 = _mm_abs_epi16(_mm_sub_epi16(t0, base0));
            let tld0 = _mm_abs_epi16(_mm_sub_epi16(tl_v, base0));
            let cond_l0 = _mm_and_si128(
                _mm_cmpeq_epi16(ld0, _mm_min_epi16(ld0, td0)),
                _mm_cmpeq_epi16(ld0, _mm_min_epi16(ld0, tld0)),
            );
            let cond_t0 = _mm_cmpeq_epi16(td0, _mm_min_epi16(td0, tld0));
            let res0 = _mm_blendv_epi8(_mm_blendv_epi8(tl_v, t0, cond_t0), left_v, cond_l0);
            let base1 = _mm_sub_epi16(_mm_add_epi16(left_v, t1), tl_v);
            let ld1 = _mm_abs_epi16(_mm_sub_epi16(left_v, base1));
            let td1 = _mm_abs_epi16(_mm_sub_epi16(t1, base1));
            let tld1 = _mm_abs_epi16(_mm_sub_epi16(tl_v, base1));
            let cond_l1 = _mm_and_si128(
                _mm_cmpeq_epi16(ld1, _mm_min_epi16(ld1, td1)),
                _mm_cmpeq_epi16(ld1, _mm_min_epi16(ld1, tld1)),
            );
            let cond_t1 = _mm_cmpeq_epi16(td1, _mm_min_epi16(td1, tld1));
            let res1 = _mm_blendv_epi8(_mm_blendv_epi8(tl_v, t1, cond_t1), left_v, cond_l1);
            store_i16x8x2_u8_fixed(d, res0, res1);
        }
        let done = c16.len() * 16;
        let (c8, r8) = r16.as_chunks_mut::<8>();
        for (d, t) in c8.iter_mut().zip(top_src[done..].as_chunks::<8>().0.iter()) {
            let top_v = load_u8x8_i16_fixed(t);
            let base = _mm_sub_epi16(_mm_add_epi16(left_v, top_v), tl_v);
            let ld = _mm_abs_epi16(_mm_sub_epi16(left_v, base));
            let td = _mm_abs_epi16(_mm_sub_epi16(top_v, base));
            let tld = _mm_abs_epi16(_mm_sub_epi16(tl_v, base));
            let cond_l = _mm_and_si128(
                _mm_cmpeq_epi16(ld, _mm_min_epi16(ld, td)),
                _mm_cmpeq_epi16(ld, _mm_min_epi16(ld, tld)),
            );
            let cond_t = _mm_cmpeq_epi16(td, _mm_min_epi16(td, tld));
            let inner = _mm_blendv_epi8(tl_v, top_v, cond_t);
            let res = _mm_blendv_epi8(inner, left_v, cond_l);
            store_i16x8_u8_fixed(d, res);
        }
        // scalar remainder columns
        let base_x = done + c8.len() * 8;
        for (xi, d) in r8.iter_mut().enumerate() {
            let top = tl[o + 1 + base_x + xi] as i32;
            let base = left + top - topleft;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (topleft - base).abs();
            *d = if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                topleft
            } as u8;
        }
        off += stride;
    }
}

pub(crate) fn ipred_paeth_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
) {
    unsafe { ipred_paeth_8bpc_sse41_impl(dst, stride, tl, o, w, h) }
}

/// Load 8 bytes and zero-extend to two i32x4 lanes (low 4, high 4).
#[inline(always)]
fn load8_u8_i32(a: &[u8]) -> (__m128i, __m128i) {
    let v = unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) };
    unsafe {
        (
            _mm_cvtepu8_epi32(v),
            _mm_cvtepu8_epi32(_mm_srli_si128(v, 4)),
        )
    }
}

/// One row of the Z1 luma 4-tap interpolation. Pixels with `base <= max_base_x`
/// are filtered; the rest of the row is set to `fill`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn z1_luma_row_sse41(
    filt: &[u8],
    top_off: usize,
    base0: i32,
    max_base_x: i32,
    fill: u8,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u8],
    w: usize,
) {
    let n_filter = ((max_base_x - base0 + 1).max(0) as usize).min(w);
    let av = _mm_set1_epi32(f.a as i32);
    let bv = _mm_set1_epi32(f.b as i32);
    let cv = _mm_set1_epi32(f.c as i32);
    let dv = _mm_set1_epi32(f.d as i32);
    let rnd = _mm_set1_epi32(64);
    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi32(255);

    let mut x = 0usize;
    while x + 8 <= n_filter {
        let bi = (top_off as i32 + base0) as usize + x;
        let w0 = load8_u8_i32(&filt[bi - 1..bi - 1 + 8]);
        let w1 = load8_u8_i32(&filt[bi..bi + 8]);
        let w2 = load8_u8_i32(&filt[bi + 1..bi + 1 + 8]);
        let w3 = load8_u8_i32(&filt[bi + 2..bi + 2 + 8]);
        unsafe {
            let acc_lo = _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(av, w0.0), _mm_mullo_epi32(bv, w1.0)),
                _mm_add_epi32(_mm_mullo_epi32(cv, w2.0), _mm_mullo_epi32(dv, w3.0)),
            );
            let acc_hi = _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(av, w0.1), _mm_mullo_epi32(bv, w1.1)),
                _mm_add_epi32(_mm_mullo_epi32(cv, w2.1), _mm_mullo_epi32(dv, w3.1)),
            );
            let res_lo = _mm_min_epi32(
                _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(acc_lo, rnd), 7), zero),
                maxv,
            );
            let res_hi = _mm_min_epi32(
                _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(acc_hi, rnd), 7), zero),
                maxv,
            );
            let packed = _mm_packus_epi16(_mm_packus_epi32(res_lo, res_hi), zero);
            _mm_storel_epi64(dst_row[x..x + 8].as_mut_ptr() as *mut __m128i, packed);
        }
        x += 8;
    }
    while x < n_filter {
        let bi = (top_off as i32 + base0) as usize + x;
        let v = f.a as i32 * filt[bi - 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi + 1] as i32
            + f.d as i32 * filt[bi + 2] as i32;
        dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
        x += 1;
    }
    dst_row[n_filter..w].fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
fn ipred_z1_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    // Common luma path only; defer everything else to the scalar reference.
    if mrl_mul || enable_ibp || !is_luma || mrl_idx != 0 || w < 8 {
        return crate::ipred::ipred_z1_8bpc(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
        );
    }
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let a = angle & 511;

    let dx = crate::tables::DR_INTRA_DERIVATIVE[a as usize] as i32;
    let max_base_x = (w + h) as i32 - 1;
    let mut filt = [0u8; 141];
    let top_off = 2usize;
    let sz = 1 + w + h;
    let str = if enable_intra_edge_filter && have_top {
        crate::ipred::get_filter_strength((w + h) as i32, 90 - a, is_sm_t)
    } else {
        0
    };
    if str > 0 {
        crate::ipred::filter_edge(
            &mut filt[1..],
            sz,
            1,
            sz as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz + 1];

    let mut ypos = dx;
    for y in 0..h {
        let base0 = ypos >> 6;
        let fill = filt[top_off + max_base_x as usize];
        if base0 > max_base_x {
            for row in dst.chunks_mut(stride).take(h).skip(y) {
                row[..w].fill(fill);
            }
            break;
        }
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let dst_row = &mut dst[y * stride..y * stride + w];
        z1_luma_row_sse41(&filt, top_off, base0, max_base_x, fill, f, dst_row, w);
        ypos += dx;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z1_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    unsafe {
        ipred_z1_8bpc_sse41_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
        )
    }
}

/// Load 8 bytes and zero-extend to two i32x4 lanes, REVERSED: lane `k` holds
/// `a[7 - k]`.
#[inline(always)]
fn load8_u8_i32_rev(a: &[u8]) -> (__m128i, __m128i) {
    let v = unsafe { _mm_loadl_epi64(a.as_ptr() as *const __m128i) };
    let mask = unsafe { _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1) };
    let rev = unsafe { _mm_shuffle_epi8(v, mask) };
    unsafe {
        (
            _mm_cvtepu8_epi32(rev),
            _mm_cvtepu8_epi32(_mm_srli_si128(rev, 4)),
        )
    }
}

/// Fill `col[0..h]` for one Z3 column: filtered where `base <= max_base_y`,
/// else `fill`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn z3_luma_col_sse41(
    filt: &[u8],
    left_off: usize,
    base0: i32,
    max_base_y: i32,
    fill: u8,
    f: &crate::ipred::DrFilter4Tap,
    col: &mut [u8],
    h: usize,
) {
    let n_filter = ((max_base_y - base0 + 1).max(0) as usize).min(h);
    let av = _mm_set1_epi32(f.a as i32);
    let bv = _mm_set1_epi32(f.b as i32);
    let cv = _mm_set1_epi32(f.c as i32);
    let dv = _mm_set1_epi32(f.d as i32);
    let rnd = _mm_set1_epi32(64);
    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi32(255);

    let mut y = 0usize;
    while y + 8 <= n_filter {
        let bi_j = left_off as i32 - base0 - y as i32;
        // tap a/b/c/d read filt[bi+1], filt[bi], filt[bi-1], filt[bi-2]; the
        // reversed windows start at bi_j-6, bi_j-7, bi_j-8, bi_j-9.
        let sa = (bi_j - 6) as usize;
        let sb = (bi_j - 7) as usize;
        let sc = (bi_j - 8) as usize;
        let sd = (bi_j - 9) as usize;
        let wa = load8_u8_i32_rev(&filt[sa..sa + 8]);
        let wb = load8_u8_i32_rev(&filt[sb..sb + 8]);
        let wc = load8_u8_i32_rev(&filt[sc..sc + 8]);
        let wd = load8_u8_i32_rev(&filt[sd..sd + 8]);
        let acc_lo = _mm_add_epi32(
            _mm_add_epi32(_mm_mullo_epi32(av, wa.0), _mm_mullo_epi32(bv, wb.0)),
            _mm_add_epi32(_mm_mullo_epi32(cv, wc.0), _mm_mullo_epi32(dv, wd.0)),
        );
        let acc_hi = _mm_add_epi32(
            _mm_add_epi32(_mm_mullo_epi32(av, wa.1), _mm_mullo_epi32(bv, wb.1)),
            _mm_add_epi32(_mm_mullo_epi32(cv, wc.1), _mm_mullo_epi32(dv, wd.1)),
        );
        let res_lo = _mm_min_epi32(
            _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(acc_lo, rnd), 7), zero),
            maxv,
        );
        let res_hi = _mm_min_epi32(
            _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(acc_hi, rnd), 7), zero),
            maxv,
        );
        let packed = _mm_packus_epi16(_mm_packus_epi32(res_lo, res_hi), zero);
        unsafe {
            _mm_storel_epi64(col[y..y + 8].as_mut_ptr() as *mut __m128i, packed);
        }
        y += 8;
    }
    while y < n_filter {
        let bi = (left_off as i32 - base0 - y as i32) as usize;
        let v = f.a as i32 * filt[bi + 1] as i32
            + f.b as i32 * filt[bi] as i32
            + f.c as i32 * filt[bi - 1] as i32
            + f.d as i32 * filt[bi - 2] as i32;
        col[y] = ((v + 64) >> 7).clamp(0, 255) as u8;
        y += 1;
    }
    col[n_filter..h].fill(fill);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
fn ipred_z3_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let enable_ibp = angle & ANGLE_IBP_FLAG != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || enable_ibp || !is_luma || mrl_idx != 0 || h > 64 {
        return crate::ipred::ipred_z3_8bpc(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
        );
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let a = angle & 511;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(270 - a) as usize] as i32;
    let max_base_y = (w + h) as i32 - 1;
    let mut filt = [0u8; 141];
    let left_off = 1 + w + h;
    let sz = 1 + w + h;
    let str = if enable_intra_edge_filter && have_left {
        crate::ipred::get_filter_strength((w + h) as i32, a - 180, is_sm_l)
    } else {
        0
    };
    if str > 0 {
        crate::ipred::filter_edge(
            &mut filt[2..],
            sz,
            h as i32 - max_height,
            sz as i32 - 1,
            &tl[o + 1 - sz..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[2..2 + sz].copy_from_slice(&tl[o + 1 - sz..o + 1]);
    }
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];

    let mut col = [0u8; 64];
    let mut ypos = dy;
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &crate::ipred::DR_INTERP_FILTER[shift];
        let base0 = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        z3_luma_col_sse41(&filt, left_off, base0, max_base_y, fill, f, &mut col, h);
        for (y, &c) in col[..h].iter().enumerate() {
            dst[y * stride + x] = c;
        }
        ypos += dy;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z3_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
    ibp_weights: &[[[u8; 16]; 16]; 7],
) {
    unsafe {
        ipred_z3_8bpc_sse41_impl(
            dst,
            stride,
            tl,
            o,
            w,
            h,
            angle,
            max_width,
            max_height,
            ibp_weights,
        )
    }
}

/// Fill `dst_row[x_start..w]` from the top reference; `xpos0` is `xpos` at
/// `x_start` and `f`/`shift` are constant across the span.
#[inline]
#[target_feature(enable = "sse4.1")]
fn z2_top_span_sse41(
    filt: &[u8],
    top_off: usize,
    mut xpos: i32,
    f: &crate::ipred::DrFilter4Tap,
    dst_row: &mut [u8],
    x_start: usize,
    w: usize,
) {
    let av = _mm_set1_epi32(f.a as i32);
    let bv = _mm_set1_epi32(f.b as i32);
    let cv = _mm_set1_epi32(f.c as i32);
    let dv = _mm_set1_epi32(f.d as i32);
    let rnd = _mm_set1_epi32(64);
    let zero = _mm_setzero_si128();
    let maxv = _mm_set1_epi32(255);

    let mut x = x_start;
    while x + 8 <= w {
        let base_x = xpos >> 6;
        let ti0 = top_off as i32 + base_x;
        // Keep every lane's window (filt[ti+1..=ti+4] for ti0..ti0+7) in bounds.
        if ti0 + 1 < 0 || ti0 + 12 > filt.len() as i32 {
            break;
        }
        let sa = (ti0 + 1) as usize;
        let w0 = load8_u8_i32(&filt[sa..sa + 8]);
        let w1 = load8_u8_i32(&filt[sa + 1..sa + 1 + 8]);
        let w2 = load8_u8_i32(&filt[sa + 2..sa + 2 + 8]);
        let w3 = load8_u8_i32(&filt[sa + 3..sa + 3 + 8]);
        unsafe {
            let acc_lo = _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(av, w0.0), _mm_mullo_epi32(bv, w1.0)),
                _mm_add_epi32(_mm_mullo_epi32(cv, w2.0), _mm_mullo_epi32(dv, w3.0)),
            );
            let acc_hi = _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(av, w0.1), _mm_mullo_epi32(bv, w1.1)),
                _mm_add_epi32(_mm_mullo_epi32(cv, w2.1), _mm_mullo_epi32(dv, w3.1)),
            );
            let res_lo = _mm_min_epi32(
                _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(acc_lo, rnd), 7), zero),
                maxv,
            );
            let res_hi = _mm_min_epi32(
                _mm_max_epi32(_mm_srai_epi32(_mm_add_epi32(acc_hi, rnd), 7), zero),
                maxv,
            );
            let packed = _mm_packus_epi16(_mm_packus_epi32(res_lo, res_hi), zero);
            _mm_storel_epi64(dst_row[x..x + 8].as_mut_ptr() as *mut __m128i, packed);
        }
        x += 8;
        xpos += 64 * 8;
    }
    while x < w {
        let base_x = xpos >> 6;
        let ti = (top_off as i32 + base_x) as usize;
        let v = f.a as i32 * filt[ti + 1] as i32
            + f.b as i32 * filt[ti + 2] as i32
            + f.c as i32 * filt[ti + 3] as i32
            + f.d as i32 * filt[ti + 4] as i32;
        dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
        x += 1;
        xpos += 64;
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "sse4.1")]
fn ipred_z2_8bpc_sse41_impl(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    use crate::levels::*;
    let mrl_mul = angle & ANGLE_MULTI_MRL_FLAG != 0;
    let is_luma = angle & ANGLE_IS_LUMA != 0;
    let mrl_idx = ((angle & ANGLE_MRL_IDX_MASK) >> ANGLE_MRL_IDX_SHIFT) as usize;
    if mrl_mul || !is_luma || mrl_idx != 0 {
        return crate::ipred::ipred_z2_8bpc(dst, stride, tl, o, w, h, angle, max_width, max_height);
    }
    let is_sm_l = angle & ANGLE_SMOOTH_LEFT_EDGE_FLAG != 0;
    let is_sm_t = angle & ANGLE_SMOOTH_TOP_EDGE_FLAG != 0;
    let enable_intra_edge_filter = angle & ANGLE_USE_EDGE_FILTER_FLAG != 0;
    let have_top = angle & ANGLE_HAS_TOP_FLAG != 0;
    let have_left = angle & ANGLE_HAS_LEFT_FLAG != 0;
    let a = angle & 511;

    let dy = crate::tables::DR_INTRA_DERIVATIVE[(a - 90) as usize] as i32;
    let dx = crate::tables::DR_INTRA_DERIVATIVE[(180 - a) as usize] as i32;

    // Top edge buffer.
    let mut filt = [0u8; 72];
    let top_off = 0usize;
    let sz_t = 1 + w;
    let str_t = if enable_intra_edge_filter && have_top {
        crate::ipred::get_filter_strength((w + h) as i32, a - 90, is_sm_t)
    } else {
        0
    };
    if str_t > 0 {
        crate::ipred::filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz_t as i32,
            str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&tl[o..o + sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // Left edge buffer.
    let mut filt2 = [0u8; 72];
    let left_off: usize = h + 2;
    let sz_l = 1 + h;
    let str_l = if enable_intra_edge_filter && have_left {
        crate::ipred::get_filter_strength((w + h) as i32, 180 - a, is_sm_l)
    } else {
        0
    };
    if str_l > 0 {
        crate::ipred::filter_edge(
            &mut filt2[1..],
            sz_l,
            h as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - h..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - h..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..h {
        let ypos = (y + 1) as i32;
        let mut xpos = -ypos * dx;
        let mut x = 0usize;
        let dst_row = &mut dst[y * stride..y * stride + w];

        // Left reference span (scalar: shift varies per pixel).
        while x < w && xpos < -64 {
            let xpos_l = (x + 1) as i32;
            let ypos_l = ((y as i32) << 6) - xpos_l * dy;
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = (left_off as i32 - base_y) as usize;
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            let v = f.a as i32 * filt2[bi - 1] as i32
                + f.b as i32 * filt2[bi - 2] as i32
                + f.c as i32 * filt2[bi - 3] as i32
                + f.d as i32 * filt2[bi - 4] as i32;
            dst_row[x] = ((v + 64) >> 7).clamp(0, 255) as u8;
            x += 1;
            xpos += 64;
        }

        // Top reference span (shift constant for the row).
        if x < w {
            let shift = ((xpos & 0x3F) >> 1) as usize;
            let f = &crate::ipred::DR_INTERP_FILTER[shift];
            z2_top_span_sse41(&filt, top_off, xpos, f, dst_row, x, w);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_z2_8bpc_sse41(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    w: usize,
    h: usize,
    angle: i32,
    max_width: i32,
    max_height: i32,
) {
    unsafe { ipred_z2_8bpc_sse41_impl(dst, stride, tl, o, w, h, angle, max_width, max_height) }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as u8
    }

    // Build a top-left edge buffer with `o` at the corner: left = tl[o-1-y],
    // top = tl[o+1+x], topleft = tl[o].
    fn make_tl(w: usize, h: usize, seed: u64) -> (Vec<u8>, usize) {
        let o = h + 1;
        let len = o + 1 + w + 16;
        let mut tl = vec![0u8; len];
        let mut s = seed;
        for v in tl.iter_mut() {
            *v = lcg(&mut s);
        }
        (tl, o)
    }

    const SIZES: &[(usize, usize)] = &[
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (32, 16),
        (8, 32),
        (16, 4),
        (4, 16),
        (4, 8),
        (8, 4),
        (16, 64),
    ];

    #[test]
    fn paeth_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x1234 + w as u64 * 131 + h as u64);
            let stride = w;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_paeth_8bpc(&mut a, stride, &tl, o, w, h);
            ipred_paeth_8bpc_sse41(&mut b, stride, &tl, o, w, h);
            assert_eq!(a, b, "paeth mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn dc_family_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x9999 + w as u64 * 7 + h as u64);
            let stride = w;
            let angle = 0; // non-IBP path

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_dc_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "dc mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_top_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_dc_top_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "dc_top mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_left_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_dc_left_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "dc_left mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_dc_128_8bpc(&mut a, stride, w, h);
            ipred_dc_128_8bpc_sse41(&mut b, stride, w, h);
            assert_eq!(a, b, "dc_128 mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn smooth_family_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x5151 + w as u64 * 17 + h as u64);
            let stride = w;
            let cases: &[(
                &str,
                fn(&mut [u8], usize, &[u8], usize, usize, usize),
                fn(&mut [u8], usize, &[u8], usize, usize, usize),
            )] = &[
                (
                    "smooth",
                    crate::ipred::ipred_smooth_8bpc,
                    ipred_smooth_8bpc_sse41,
                ),
                (
                    "smooth_v",
                    crate::ipred::ipred_smooth_v_8bpc,
                    ipred_smooth_v_8bpc_sse41,
                ),
                (
                    "smooth_h",
                    crate::ipred::ipred_smooth_h_8bpc,
                    ipred_smooth_h_8bpc_sse41,
                ),
            ];
            for (name, scalar, simd) in cases {
                let mut a = vec![0u8; stride * h];
                let mut b = vec![0u8; stride * h];
                scalar(&mut a, stride, &tl, o, w, h);
                simd(&mut b, stride, &tl, o, w, h);
                assert_eq!(a, b, "{} mismatch w={} h={}", name, w, h);
            }
        }
    }

    #[test]
    fn v_h_match_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        for &(w, h) in SIZES {
            let (tl, o) = make_tl(w, h, 0x2727 + w as u64 * 23 + h as u64);
            let stride = w;
            let angle = 0;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_v_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_v_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "v mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_h_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_h_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "h mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn v_h_mrl_match_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        use crate::levels::ANGLE_MULTI_MRL_FLAG;
        for &(w, h) in SIZES {
            let e_stride = (w + h) * 2 + 1;
            let o = h + 1;
            let len = o + 1 + e_stride + w + 32;
            let mut tl = vec![0u8; len];
            let mut s = 0x3131 + w as u64 * 5 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            let angle = ANGLE_MULTI_MRL_FLAG;

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_v_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_v_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "v_mrl mismatch w={} h={}", w, h);

            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            crate::ipred::ipred_h_8bpc(&mut a, stride, &tl, o, w, h, angle);
            ipred_h_8bpc_sse41(&mut b, stride, &tl, o, w, h, angle);
            assert_eq!(a, b, "h_mrl mismatch w={} h={}", w, h);
        }
    }

    #[test]
    fn z1_luma_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        use crate::levels::*;
        let ibp = [[[0u8; 16]; 16]; 7];
        let base_flags = ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG;
        for &(w, h) in SIZES {
            let sz = 1 + w + h;
            let o = 16usize;
            let maxd = (w + h) as i32;
            let len = o + sz + (w + h) + 64;
            let mut tl = vec![0u8; len];
            let mut s = 0x7a7a + w as u64 * 13 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            for a in [3i32, 30, 45, 63, 87] {
                for extra in [
                    0,
                    ANGLE_USE_EDGE_FILTER_FLAG,
                    ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_SMOOTH_TOP_EDGE_FLAG,
                ] {
                    let angle = a | base_flags | extra;
                    let mut sa = vec![0u8; stride * h];
                    let mut sb = vec![0u8; stride * h];
                    crate::ipred::ipred_z1_8bpc(
                        &mut sa, stride, &tl, o, w, h, angle, maxd, maxd, &ibp,
                    );
                    ipred_z1_8bpc_sse41(&mut sb, stride, &tl, o, w, h, angle, maxd, maxd, &ibp);
                    assert_eq!(
                        sa, sb,
                        "z1 mismatch w={} h={} a={} extra={:#x}",
                        w, h, a, extra
                    );
                }
            }
        }
    }

    #[test]
    fn z3_luma_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        use crate::levels::*;
        let ibp = [[[0u8; 16]; 16]; 7];
        let base_flags = ANGLE_IS_LUMA | ANGLE_HAS_LEFT_FLAG;
        for &(w, h) in SIZES {
            let o = w + h + 8;
            let maxd = (w + h) as i32;
            let len = o + 16;
            let mut tl = vec![0u8; len];
            let mut s = 0x5d5d + w as u64 * 11 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            for a in [183i32, 200, 225, 250, 267] {
                for extra in [
                    0,
                    ANGLE_USE_EDGE_FILTER_FLAG,
                    ANGLE_USE_EDGE_FILTER_FLAG | ANGLE_SMOOTH_LEFT_EDGE_FLAG,
                ] {
                    let angle = a | base_flags | extra;
                    let mut sa = vec![0u8; stride * h];
                    let mut sb = vec![0u8; stride * h];
                    crate::ipred::ipred_z3_8bpc(
                        &mut sa, stride, &tl, o, w, h, angle, maxd, maxd, &ibp,
                    );
                    ipred_z3_8bpc_sse41(&mut sb, stride, &tl, o, w, h, angle, maxd, maxd, &ibp);
                    assert_eq!(
                        sa, sb,
                        "z3 mismatch w={} h={} a={} extra={:#x}",
                        w, h, a, extra
                    );
                }
            }
        }
    }

    #[test]
    fn z2_luma_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }
        use crate::levels::*;
        let base_flags = ANGLE_IS_LUMA | ANGLE_HAS_TOP_FLAG | ANGLE_HAS_LEFT_FLAG;
        for &(w, h) in SIZES {
            let o = h + 8;
            let maxd = (w + h) as i32;
            let len = o + 1 + w + 16;
            let mut tl = vec![0u8; len];
            let mut s = 0x6c6c + w as u64 * 19 + h as u64;
            for v in tl.iter_mut() {
                *v = lcg(&mut s);
            }
            let stride = w;
            for a in [100i32, 120, 135, 150, 170] {
                for extra in [
                    0,
                    ANGLE_USE_EDGE_FILTER_FLAG,
                    ANGLE_USE_EDGE_FILTER_FLAG
                        | ANGLE_SMOOTH_TOP_EDGE_FLAG
                        | ANGLE_SMOOTH_LEFT_EDGE_FLAG,
                ] {
                    let angle = a | base_flags | extra;
                    let mut sa = vec![0u8; stride * h];
                    let mut sb = vec![0u8; stride * h];
                    crate::ipred::ipred_z2_8bpc(&mut sa, stride, &tl, o, w, h, angle, maxd, maxd);
                    ipred_z2_8bpc_sse41(&mut sb, stride, &tl, o, w, h, angle, maxd, maxd);
                    assert_eq!(
                        sa, sb,
                        "z2 mismatch w={} h={} a={} extra={:#x}",
                        w, h, a, extra
                    );
                }
            }
        }
    }
}
