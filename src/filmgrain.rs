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

use crate::headers::FilmGrainData;
use crate::intops::iclip;
use crate::tables::GAUSSIAN_SEQUENCE;
use std::sync::OnceLock;

pub(crate) const GRAIN_WIDTH: usize = 82;
pub(crate) const GRAIN_HEIGHT: usize = 73;
pub(crate) const SUB_GRAIN_WIDTH: usize = 44;
pub(crate) const SUB_GRAIN_HEIGHT: usize = 38;

pub(crate) fn get_random_number(bits: u32, state: &mut u32) -> u32 {
    let r = *state;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    (*state >> (16 - bits)) & ((1 << bits) - 1)
}

pub(crate) fn round2(x: i32, shift: u32) -> i32 {
    (x + ((1 << shift) >> 1)) >> shift
}

#[inline]
fn avg_chroma_luma<T: Copy + Into<i32>>(
    luma: &[T],
    luma_width: usize,
    lx: usize,
    sx: usize,
) -> i32 {
    let l0 = luma[lx].into();
    if sx != 0 {
        let l1 = if lx + 1 < luma_width {
            luma[lx + 1].into()
        } else {
            l0
        };
        (l0 + l1 + 1) >> 1
    } else {
        l0
    }
}

#[inline]
fn scaled_gaussian_table(shift: i32) -> [i16; 2048] {
    debug_assert!(shift >= 0);
    let shift = shift as u32;
    let mut table = [0i16; 2048];
    for (dst, &src) in table.iter_mut().zip(GAUSSIAN_SEQUENCE.iter()) {
        *dst = round2(src as i32, shift) as i16;
    }
    table
}

pub(crate) fn generate_scaling_8bpc(points: &[[u8; 2]], scaling: &mut [u8; 256]) {
    let num = points.len();
    if num == 0 {
        scaling.fill(0);
        return;
    }

    let first_x = points[0][0] as usize;
    scaling[..first_x].fill(points[0][1]);

    for pair in points.windows(2) {
        let bx = pair[0][0] as i32;
        let by = pair[0][1] as i32;
        let ex = pair[1][0] as i32;
        let ey = pair[1][1] as i32;
        let dx = ex - bx;
        let dy = ey - by;
        debug_assert!(dx > 0);
        let delta = dy * ((0x10000 + (dx >> 1)) / dx);
        let mut d = 0x8000i32;
        for out in &mut scaling[bx as usize..ex as usize] {
            *out = (by + (d >> 16)) as u8;
            d += delta;
        }
    }

    let n = points[num - 1][0] as usize;
    scaling[n..].fill(points[num - 1][1]);
}

pub(crate) fn generate_scaling_hbd(points: &[[u8; 2]], bitdepth: usize, scaling: &mut [u8]) {
    debug_assert!(bitdepth > 8);
    let size = 1usize << bitdepth;
    debug_assert!(scaling.len() >= size);
    let scaling = &mut scaling[..size];
    let shift_x = bitdepth - 8;
    let pad = 1usize << shift_x;

    let num = points.len();
    if num == 0 {
        scaling.fill(0);
        return;
    }

    let first_x = (points[0][0] as usize) << shift_x;
    scaling[..first_x].fill(points[0][1]);

    for pair in points.windows(2) {
        let bx = pair[0][0] as i32;
        let by = pair[0][1] as i32;
        let ex = pair[1][0] as i32;
        let ey = pair[1][1] as i32;
        let dx = ex - bx;
        let dy = ey - by;
        debug_assert!(dx > 0);
        let delta = dy * ((0x10000 + (dx >> 1)) / dx);
        let mut d = 0x8000i32;
        let start = (bx as usize) << shift_x;
        let end = (ex as usize) << shift_x;
        for out in scaling[start..end].iter_mut().step_by(pad) {
            *out = (by + (d >> 16)) as u8;
            d += delta;
        }
    }

    let n = (points[num - 1][0] as usize) << shift_x;
    scaling[n..].fill(points[num - 1][1]);

    let rnd = pad >> 1;
    for pair in points.windows(2) {
        let bx = (pair[0][0] as usize) << shift_x;
        let ex = (pair[1][0] as usize) << shift_x;
        let mut x = bx;
        while x < ex {
            let range = scaling[x + pad] as i32 - scaling[x] as i32;
            let base = scaling[x] as i32;
            let mut r = rnd as i32;
            for out in &mut scaling[x + 1..x + pad] {
                r += range;
                *out = (base + (r >> shift_x)) as u8;
            }
            x += pad;
        }
    }
}

pub(crate) fn generate_grain_y(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
) {
    let shift = 4 + data.grain_scale_shift;
    let scaled_gaussian = scaled_gaussian_table(shift);
    let grain_ctr = 128;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    for y in 0..GRAIN_HEIGHT {
        for x in 0..GRAIN_WIDTH {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = scaled_gaussian[value];
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    if ar_lag == 0 {
        return;
    }

    for y in ar_pad..GRAIN_HEIGHT {
        for x in ar_pad..GRAIN_WIDTH - ar_pad {
            let coeff = &data.ar_coeffs[0];
            let mut sum = 0i32;
            let mut ci = 0usize;
            for dy in y.wrapping_sub(ar_lag)..=y {
                let dx_start = x.wrapping_sub(ar_lag);
                let dx_end = if dy == y { x } else { x + ar_lag + 1 };
                for dx in dx_start..dx_end {
                    if dy == y && dx == x {
                        break;
                    }
                    sum += coeff[ci] as i32 * buf[dy][dx] as i32;
                    ci += 1;
                }
            }

            let grain = buf[y][x] as i32 + round2(sum, data.ar_coeff_shift as u32);
            buf[y][x] = iclip(grain, grain_min, grain_max) as i16;
        }
    }
}

pub(crate) fn generate_grain_uv(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    buf_y: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
    uv: usize,
    subx: bool,
    suby: bool,
) {
    seed ^= if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = 4 + data.grain_scale_shift;
    let scaled_gaussian = scaled_gaussian_table(shift);
    let grain_ctr = 128;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let chroma_w = if subx { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH };
    let chroma_h = if suby { SUB_GRAIN_HEIGHT } else { GRAIN_HEIGHT };

    for y in 0..chroma_h {
        for x in 0..chroma_w {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = scaled_gaussian[value];
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    let subx_i = subx as usize;
    let suby_i = suby as usize;
    if ar_lag == 0 && data.num_points[0] == 0 {
        return;
    }

    for y in ar_pad..chroma_h {
        for x in ar_pad..chroma_w - ar_pad {
            let coeff = &data.ar_coeffs[1 + uv];
            let mut sum = 0i32;
            let mut ci = 0usize;
            'outer: for dy in y.wrapping_sub(ar_lag)..=y {
                let dx_start = x.wrapping_sub(ar_lag);
                // Current row stops at (and includes) the center pixel dx==x,
                // case uses the final AR coeff for the luma contribution, then
                // breaks). Off rows span the full [-ar_lag, +ar_lag] window.
                let dx_end = if dy == y { x + 1 } else { x + ar_lag + 1 };
                for dx in dx_start..dx_end {
                    if dy == y && dx == x {
                        if data.num_points[0] > 0 {
                            let luma_x = ((x - ar_pad) << subx_i) + ar_pad;
                            let luma_y = ((y - ar_pad) << suby_i) + ar_pad;
                            let mut luma = 0i32;
                            for i in 0..=suby_i {
                                for j in 0..=subx_i {
                                    luma += buf_y[luma_y + i][luma_x + j] as i32;
                                }
                            }
                            luma = round2(luma, (subx_i + suby_i) as u32);
                            sum += luma * coeff[ci] as i32;
                        }
                        break 'outer;
                    }
                    sum += coeff[ci] as i32 * buf[dy][dx] as i32;
                    ci += 1;
                }
            }

            let grain = buf[y][x] as i32 + round2(sum, data.ar_coeff_shift as u32);
            buf[y][x] = iclip(grain, grain_min, grain_max) as i16;
        }
    }
}

pub(crate) fn sample_lut(
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bs: usize,
    offsets: &[[[i32; 2]; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
) -> i16 {
    let off = &offsets[bx][by];
    let offx = 3 + (2 >> subx) * (3 + off[1] as usize);
    let offy = 3 + (2 >> suby) * (3 + off[0] as usize);
    grain_lut[offy + y + (bs >> suby) * by][offx + x + (bs >> subx) * bx]
}

#[inline]
fn sample_lut_row<'a>(
    grain_lut: &'a [[i16; GRAIN_WIDTH]],
    bs: usize,
    offsets: &[[[i32; 2]; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
) -> &'a [i16] {
    let off = &offsets[bx][by];
    let offx = 3 + (2 >> subx) * (3 + off[1] as usize);
    let offy = 3 + (2 >> suby) * (3 + off[0] as usize);
    let row = offy + y + (bs >> suby) * by;
    let col = offx + x + (bs >> subx) * bx;
    &grain_lut[row][col..]
}

#[allow(unused)]
fn blend_top_grain_row_scalar(
    dst: &mut [i16],
    old: &[i16],
    grain: &[i16],
    grain_min: i32,
    grain_max: i32,
    old_w: i32,
    new_w: i32,
) {
    let n = dst.len().min(old.len()).min(grain.len());
    for ((d, &old), &grain) in dst[..n].iter_mut().zip(&old[..n]).zip(&grain[..n]) {
        *d = iclip(
            round2(old as i32 * old_w + grain as i32 * new_w, 5),
            grain_min,
            grain_max,
        ) as i16;
    }
}

#[inline]
fn blend_top_grain_row_dispatch(
    dst: &mut [i16],
    old: &[i16],
    grain: &[i16],
    grain_min: i32,
    grain_max: i32,
    old_w: i32,
    new_w: i32,
) {
    static F: OnceLock<BlendTopGrainRowFn> = OnceLock::new();
    let f = F.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return crate::avx::blend_top_grain_row_avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::blend_top_grain_row_neon
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            blend_top_grain_row_scalar
        }
    });
    unsafe { f(dst, old, grain, grain_min, grain_max, old_w, new_w) }
}

#[inline]
fn blend_top_grain_row<'a>(
    tmp: &'a mut [i16],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bs: usize,
    offsets: &[[[i32; 2]; 2]; 2],
    subx: usize,
    suby: usize,
    xstart: usize,
    y: usize,
    len: usize,
    grain_min: i32,
    grain_max: i32,
    old_w: i32,
    new_w: i32,
) -> &'a [i16] {
    debug_assert!(tmp.len() >= len);
    let grain = sample_lut_row(grain_lut, bs, offsets, subx, suby, 0, 0, xstart, y);
    let old = sample_lut_row(grain_lut, bs, offsets, subx, suby, 0, 1, xstart, y);
    blend_top_grain_row_dispatch(
        &mut tmp[..len],
        &old[..len],
        &grain[..len],
        grain_min,
        grain_max,
        old_w,
        new_w,
    );
    &tmp[..len]
}

#[inline]
fn blend_left_grain_row<'a>(
    tmp: &'a mut [i16; 2],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bs: usize,
    offsets: &[[[i32; 2]; 2]; 2],
    subx: usize,
    suby: usize,
    y: usize,
    len: usize,
    grain_min: i32,
    grain_max: i32,
    weights: &[[i32; 2]; 2],
) -> &'a [i16] {
    debug_assert!(len <= 2);
    for (x, (tmp, weights)) in tmp[..len].iter_mut().zip(&weights[..len]).enumerate() {
        let grain = sample_lut(grain_lut, bs, offsets, subx, suby, 0, 0, x, y) as i32;
        let old = sample_lut(grain_lut, bs, offsets, subx, suby, 1, 0, x, y) as i32;
        *tmp = iclip(
            round2(old * weights[0] + grain * weights[1], 5),
            grain_min,
            grain_max,
        ) as i16;
    }
    &tmp[..len]
}

#[inline]
fn blend_top_left_grain_row<'a>(
    tmp: &'a mut [i16; 2],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bs: usize,
    offsets: &[[[i32; 2]; 2]; 2],
    subx: usize,
    suby: usize,
    y: usize,
    len: usize,
    grain_min: i32,
    grain_max: i32,
    h_weights: &[[i32; 2]; 2],
    v_weights: &[[i32; 2]; 2],
) -> &'a [i16] {
    debug_assert!(len <= 2);
    let v_weights = v_weights[y];
    for (x, (tmp, h_weights)) in tmp[..len].iter_mut().zip(&h_weights[..len]).enumerate() {
        let mut top = sample_lut(grain_lut, bs, offsets, subx, suby, 0, 1, x, y) as i32;
        let old_top = sample_lut(grain_lut, bs, offsets, subx, suby, 1, 1, x, y) as i32;
        top = iclip(
            round2(old_top * h_weights[0] + top * h_weights[1], 5),
            grain_min,
            grain_max,
        );

        let mut grain = sample_lut(grain_lut, bs, offsets, subx, suby, 0, 0, x, y) as i32;
        let old = sample_lut(grain_lut, bs, offsets, subx, suby, 1, 0, x, y) as i32;
        grain = iclip(
            round2(old * h_weights[0] + grain * h_weights[1], 5),
            grain_min,
            grain_max,
        );

        *tmp = iclip(
            round2(top * v_weights[0] + grain * v_weights[1], 5),
            grain_min,
            grain_max,
        ) as i16;
    }
    &tmp[..len]
}

#[derive(Clone, Copy)]
struct GrainPlaneMut<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: this wrapper is only used by the film-grain row-band dispatcher.  Each
// worker claims a unique grain block row and writes only that row band's output
// rows, so concurrent accesses are disjoint even though every worker receives a
// view of the whole destination plane.
unsafe impl<T: Send> Send for GrainPlaneMut<T> {}
unsafe impl<T: Send> Sync for GrainPlaneMut<T> {}

impl<T> GrainPlaneMut<T> {
    #[inline]
    fn new(s: &mut [T]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn whole(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[inline]
fn filmgrain_thread_count(
    rows: usize,
    n_threads: u32,
    pool: Option<&crate::mtpool::ThreadPool>,
) -> (Option<&crate::mtpool::ThreadPool>, usize) {
    let want = (n_threads as usize).max(1);
    let active = pool.filter(|_| want >= 2 && rows >= 2);
    let cap = match active {
        Some(p) => p.workers() + 1,
        None => 1,
    };
    (active, want.min(cap).min(rows).max(1))
}

type BlendTopGrainRowFn = unsafe fn(&mut [i16], &[i16], &[i16], i32, i32, i32, i32);
type FgyRow8Fn = unsafe fn(&mut [u8], &[u8], &[i16], &[u8; 256], i32, i32, i32);
type FgyRowHbdFn = unsafe fn(&mut [u16], &[u16], &[i16], &[u8], i32, i32, i32);
type FguvRow8Fn = unsafe fn(
    &mut [u8],
    &[u8],
    &[i16],
    &[u8],
    usize,
    usize,
    usize,
    &[u8],
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    bool,
);
type FguvRowHbdFn = unsafe fn(
    &mut [u16],
    &[u16],
    &[i16],
    &[u16],
    usize,
    usize,
    usize,
    &[u8],
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    bool,
    i32,
);

#[allow(unused)]
fn fgy_row_8bpc_scalar(
    dst: &mut [u8],
    src: &[u8],
    grain: &[i16],
    scaling: &[u8; 256],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    for ((d, &s), &grain) in dst[..n].iter_mut().zip(&src[..n]).zip(&grain[..n]) {
        let s = s as i32;
        let noise = round2(
            scaling[s as usize] as i32 * grain as i32,
            scaling_shift as u32,
        );
        *d = iclip(s + noise, min_value, max_value) as u8;
    }
}

#[allow(unused)]
fn fgy_row_hbd_scalar(
    dst: &mut [u16],
    src: &[u16],
    grain: &[i16],
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    for ((d, &s), &grain) in dst[..n].iter_mut().zip(&src[..n]).zip(&grain[..n]) {
        let s = s as i32;
        let noise = round2(
            scaling[s as usize] as i32 * grain as i32,
            scaling_shift as u32,
        );
        *d = iclip(s + noise, min_value, max_value) as u16;
    }
}

#[allow(unused)]
fn fguv_row_8bpc_scalar(
    dst: &mut [u8],
    src: &[u8],
    grain: &[i16],
    luma: &[u8],
    cx_base: usize,
    luma_width: usize,
    sx: usize,
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
    uv_luma_mult: i32,
    uv_mult: i32,
    uv_offset: i32,
    chroma_scaling_from_luma: bool,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    for (x, ((d, &s), &grain)) in dst[..n]
        .iter_mut()
        .zip(&src[..n])
        .zip(&grain[..n])
        .enumerate()
    {
        let lx = (cx_base + x) << sx;
        let avg = avg_chroma_luma(luma, luma_width, lx, sx);
        let s = s as i32;
        let val = if !chroma_scaling_from_luma {
            iclip(
                ((avg * uv_luma_mult + s * uv_mult) >> 6) + uv_offset,
                0,
                255,
            ) as usize
        } else {
            avg as usize
        };
        let noise = round2(scaling[val] as i32 * grain as i32, scaling_shift as u32);
        *d = iclip(s + noise, min_value, max_value) as u8;
    }
}

#[allow(unused)]
fn fguv_row_hbd_scalar(
    dst: &mut [u16],
    src: &[u16],
    grain: &[i16],
    luma: &[u16],
    cx_base: usize,
    luma_width: usize,
    sx: usize,
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
    uv_luma_mult: i32,
    uv_mult: i32,
    uv_offset_scaled: i32,
    chroma_scaling_from_luma: bool,
    bitdepth_max: i32,
) {
    let n = dst.len().min(src.len()).min(grain.len());
    for (x, ((d, &s), &grain)) in dst[..n]
        .iter_mut()
        .zip(&src[..n])
        .zip(&grain[..n])
        .enumerate()
    {
        let lx = (cx_base + x) << sx;
        let avg = avg_chroma_luma(luma, luma_width, lx, sx);
        let s = s as i32;
        let val = if !chroma_scaling_from_luma {
            iclip(
                ((avg * uv_luma_mult + s * uv_mult) >> 6) + uv_offset_scaled,
                0,
                bitdepth_max,
            ) as usize
        } else {
            avg as usize
        };
        let noise = round2(scaling[val] as i32 * grain as i32, scaling_shift as u32);
        *d = iclip(s + noise, min_value, max_value) as u16;
    }
}

#[inline]
fn fgy_row_8bpc_dispatch(
    dst: &mut [u8],
    src: &[u8],
    grain: &[i16],
    scaling: &[u8; 256],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    static F: OnceLock<FgyRow8Fn> = OnceLock::new();
    let f = F.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return crate::avx::fgy_row_8bpc_avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::fgy_row_8bpc_neon
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fgy_row_8bpc_scalar
        }
    });
    unsafe {
        f(
            dst,
            src,
            grain,
            scaling,
            scaling_shift,
            min_value,
            max_value,
        )
    }
}

#[inline]
fn fgy_row_hbd_dispatch(
    dst: &mut [u16],
    src: &[u16],
    grain: &[i16],
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
) {
    static F: OnceLock<FgyRowHbdFn> = OnceLock::new();
    let f = F.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return crate::avx::fgy_row_hbd_avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return crate::neon::fgy_row_hbd_neon;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fgy_row_hbd_scalar
        }
    });
    unsafe {
        f(
            dst,
            src,
            grain,
            scaling,
            scaling_shift,
            min_value,
            max_value,
        )
    }
}

#[inline]
fn fguv_row_8bpc_dispatch(
    dst: &mut [u8],
    src: &[u8],
    grain: &[i16],
    luma: &[u8],
    cx_base: usize,
    luma_width: usize,
    sx: usize,
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
    uv_luma_mult: i32,
    uv_mult: i32,
    uv_offset: i32,
    chroma_scaling_from_luma: bool,
) {
    static F: OnceLock<FguvRow8Fn> = OnceLock::new();
    let f = F.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return crate::avx::fguv_row_8bpc_avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::fguv_row_8bpc_neon
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fguv_row_8bpc_scalar
        }
    });
    unsafe {
        f(
            dst,
            src,
            grain,
            luma,
            cx_base,
            luma_width,
            sx,
            scaling,
            scaling_shift,
            min_value,
            max_value,
            uv_luma_mult,
            uv_mult,
            uv_offset,
            chroma_scaling_from_luma,
        )
    }
}

#[inline]
fn fguv_row_hbd_dispatch(
    dst: &mut [u16],
    src: &[u16],
    grain: &[i16],
    luma: &[u16],
    cx_base: usize,
    luma_width: usize,
    sx: usize,
    scaling: &[u8],
    scaling_shift: i32,
    min_value: i32,
    max_value: i32,
    uv_luma_mult: i32,
    uv_mult: i32,
    uv_offset_scaled: i32,
    chroma_scaling_from_luma: bool,
    bitdepth_max: i32,
) {
    static F: OnceLock<FguvRowHbdFn> = OnceLock::new();
    let f = F.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return crate::avx::fguv_row_hbd_avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return crate::neon::fguv_row_hbd_neon;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fguv_row_hbd_scalar
        }
    });
    unsafe {
        f(
            dst,
            src,
            grain,
            luma,
            cx_base,
            luma_width,
            sx,
            scaling,
            scaling_shift,
            min_value,
            max_value,
            uv_luma_mult,
            uv_mult,
            uv_offset_scaled,
            chroma_scaling_from_luma,
            bitdepth_max,
        )
    }
}

pub(crate) fn fgy_32x32xn_8bpc(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    in_seed: u32,
    pw: usize,
    scaling: &[u8; 256],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bh: i32,
    row_num: i32,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let grain_ctr = 128;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let bs = (16 << data.block_size) as usize;

    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16i32, 235i32)
    } else {
        (0, 255)
    };

    let mut seed = [0u32; 2];
    for i in 0..rows {
        seed[i] = in_seed;
        seed[i] ^= ((((row_num - i as i32) * 37 + 178) & 0xFF) as u32) << 8;
        seed[i] ^= (((row_num - i as i32) * 173 + 105) & 0xFF) as u32;
    }

    let mut offsets = [[[0i32; 2]; 2]; 2];
    let w: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let mut bx = 0usize;
    while bx < pw {
        let bw = bs.min(pw - bx) as i32;

        if data.overlap_flag && bx > 0 {
            for i in 0..rows {
                for n in 0..2 {
                    offsets[1][i][n] = offsets[0][i][n];
                }
            }
        }

        for i in 0..rows {
            for n in 0..2 {
                offsets[0][i][n] = (((3 - data.block_size) as u32
                    * get_random_number(9, &mut seed[i]))
                    >> 6) as i32;
                for _ in 0..3 {
                    get_random_number(16, &mut seed[i]);
                }
            }
        }

        let ystart = if data.overlap_flag && row_num > 0 {
            2.min(bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx > 0 {
            2.min(bw)
        } else {
            0
        };

        for y in ystart..bh {
            if xstart < bw {
                let si = y as usize * stride + xstart as usize + bx;
                let len = (bw - xstart) as usize;
                let grain = sample_lut_row(
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    0,
                    0,
                    xstart as usize,
                    y as usize,
                );
                fgy_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let grain = blend_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w,
                );
                let si = y as usize * stride + bx;
                fgy_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
        }

        let mut overlap_row = [0i16; GRAIN_WIDTH];
        for y in 0..ystart {
            if xstart < bw {
                let si = y as usize * stride + xstart as usize + bx;
                let len = (bw - xstart) as usize;
                let grain = blend_top_grain_row(
                    &mut overlap_row,
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    xstart as usize,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    w[y as usize][0],
                    w[y as usize][1],
                );
                fgy_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let grain = blend_top_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w,
                    &w,
                );
                let si = y as usize * stride + bx;
                fgy_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
        }

        bx += bs;
    }
}

pub(crate) fn fguv_32x32xn_8bpc(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    in_seed: u32,
    pw: usize,
    scaling: &[u8; 256],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bh: i32,
    row_num: i32,
    luma_row: &[u8],
    luma_stride: usize,
    luma_width: usize,
    uv: usize,
    is_id: bool,
    sx: usize,
    sy: usize,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let grain_ctr = 128;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let bs = (16 << data.block_size) as usize;

    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16i32, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };

    let mut seed = [0u32; 2];
    for i in 0..rows {
        seed[i] = in_seed;
        seed[i] ^= ((((row_num - i as i32) * 37 + 178) & 0xFF) as u32) << 8;
        seed[i] ^= (((row_num - i as i32) * 173 + 105) & 0xFF) as u32;
    }

    let mut offsets = [[[0i32; 2]; 2]; 2];
    let w: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];

    let mut bx = 0usize;
    while bx < pw {
        let bw = ((bs >> sx).min(pw - bx)) as i32;

        if data.overlap_flag && bx > 0 {
            for i in 0..rows {
                for n in 0..2 {
                    offsets[1][i][n] = offsets[0][i][n];
                }
            }
        }

        for i in 0..rows {
            for n in 0..2 {
                offsets[0][i][n] = (((3 - data.block_size) as u32
                    * get_random_number(9, &mut seed[i]))
                    >> 6) as i32;
                for _ in 0..3 {
                    get_random_number(16, &mut seed[i]);
                }
            }
        }

        let ystart = if data.overlap_flag && row_num > 0 {
            (2 >> sy as i32).min(bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx > 0 {
            (2 >> sx as i32).min(bw)
        } else {
            0
        };

        for y in ystart..bh {
            if xstart < bw {
                let si = y as usize * stride + bx + xstart as usize;
                let len = (bw - xstart) as usize;
                let ly = (y as usize) << sy;
                let grain = sample_lut_row(
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    0,
                    0,
                    xstart as usize,
                    y as usize,
                );
                fguv_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx + xstart as usize,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv],
                    data.chroma_scaling_from_luma,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let ly = (y as usize) << sy;
                let grain = blend_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w[sx],
                );
                let si = y as usize * stride + bx;
                fguv_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv],
                    data.chroma_scaling_from_luma,
                );
            }
        }

        let mut overlap_row = [0i16; GRAIN_WIDTH];
        for y in 0..ystart {
            if xstart < bw {
                let si = y as usize * stride + bx + xstart as usize;
                let len = (bw - xstart) as usize;
                let ly = (y as usize) << sy;
                let grain = blend_top_grain_row(
                    &mut overlap_row,
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    xstart as usize,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    w[sy][y as usize][0],
                    w[sy][y as usize][1],
                );
                fguv_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx + xstart as usize,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv],
                    data.chroma_scaling_from_luma,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let ly = (y as usize) << sy;
                let grain = blend_top_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w[sx],
                    &w[sy],
                );
                let si = y as usize * stride + bx;
                fguv_row_8bpc_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv],
                    data.chroma_scaling_from_luma,
                );
            }
        }

        bx += bs >> sx;
    }
}

pub(crate) fn generate_grain_y_hbd(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
    bitdepth: usize,
) {
    debug_assert!(bitdepth > 8);
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let scaled_gaussian = scaled_gaussian_table(shift);
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    for y in 0..GRAIN_HEIGHT {
        for x in 0..GRAIN_WIDTH {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = scaled_gaussian[value];
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    if ar_lag == 0 {
        return;
    }

    for y in ar_pad..GRAIN_HEIGHT {
        for x in ar_pad..GRAIN_WIDTH - ar_pad {
            let coeff = &data.ar_coeffs[0];
            let mut sum = 0i32;
            let mut ci = 0usize;
            for dy in y.wrapping_sub(ar_lag)..=y {
                let dx_start = x.wrapping_sub(ar_lag);
                let dx_end = if dy == y { x } else { x + ar_lag + 1 };
                for dx in dx_start..dx_end {
                    if dy == y && dx == x {
                        break;
                    }
                    sum += coeff[ci] as i32 * buf[dy][dx] as i32;
                    ci += 1;
                }
            }

            let grain = buf[y][x] as i32 + round2(sum, data.ar_coeff_shift as u32);
            buf[y][x] = iclip(grain, grain_min, grain_max) as i16;
        }
    }
}

pub(crate) fn generate_grain_uv_hbd(
    buf: &mut [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    buf_y: &[[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    data: &FilmGrainData,
    mut seed: u32,
    uv: usize,
    subx: bool,
    suby: bool,
    bitdepth: usize,
) {
    debug_assert!(bitdepth > 8);
    let bitdepth_min_8 = bitdepth as i32 - 8;
    seed ^= if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = 4 - bitdepth_min_8 + data.grain_scale_shift;
    let scaled_gaussian = scaled_gaussian_table(shift);
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;

    let chroma_w = if subx { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH };
    let chroma_h = if suby { SUB_GRAIN_HEIGHT } else { GRAIN_HEIGHT };

    for y in 0..chroma_h {
        for x in 0..chroma_w {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = scaled_gaussian[value];
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    let subx_i = subx as usize;
    let suby_i = suby as usize;
    if ar_lag == 0 && data.num_points[0] == 0 {
        return;
    }

    for y in ar_pad..chroma_h {
        for x in ar_pad..chroma_w - ar_pad {
            let coeff = &data.ar_coeffs[1 + uv];
            let mut sum = 0i32;
            let mut ci = 0usize;
            'outer: for dy in y.wrapping_sub(ar_lag)..=y {
                let dx_start = x.wrapping_sub(ar_lag);
                let dx_end = if dy == y { x + 1 } else { x + ar_lag + 1 };
                for dx in dx_start..dx_end {
                    if dy == y && dx == x {
                        if data.num_points[0] > 0 {
                            let luma_x = ((x - ar_pad) << subx_i) + ar_pad;
                            let luma_y = ((y - ar_pad) << suby_i) + ar_pad;
                            let mut luma = 0i32;
                            for i in 0..=suby_i {
                                for j in 0..=subx_i {
                                    luma += buf_y[luma_y + i][luma_x + j] as i32;
                                }
                            }
                            luma = round2(luma, (subx_i + suby_i) as u32);
                            sum += luma * coeff[ci] as i32;
                        }
                        break 'outer;
                    }
                    sum += coeff[ci] as i32 * buf[dy][dx] as i32;
                    ci += 1;
                }
            }

            let grain = buf[y][x] as i32 + round2(sum, data.ar_coeff_shift as u32);
            buf[y][x] = iclip(grain, grain_min, grain_max) as i16;
        }
    }
}

pub(crate) fn fgy_32x32xn_hbd(
    dst: &mut [u16],
    src: &[u16],
    stride: usize,
    data: &FilmGrainData,
    in_seed: u32,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bh: i32,
    row_num: i32,
    bitdepth: usize,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let bitdepth_max = (1i32 << bitdepth) - 1;
    let bs = (16 << data.block_size) as usize;

    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16i32 << bitdepth_min_8, 235i32 << bitdepth_min_8)
    } else {
        (0, bitdepth_max)
    };

    let mut seed = [0u32; 2];
    for i in 0..rows {
        seed[i] = in_seed;
        seed[i] ^= ((((row_num - i as i32) * 37 + 178) & 0xFF) as u32) << 8;
        seed[i] ^= (((row_num - i as i32) * 173 + 105) & 0xFF) as u32;
    }

    let mut offsets = [[[0i32; 2]; 2]; 2];
    let w: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let mut bx = 0usize;
    while bx < pw {
        let bw = bs.min(pw - bx) as i32;

        if data.overlap_flag && bx > 0 {
            for i in 0..rows {
                for n in 0..2 {
                    offsets[1][i][n] = offsets[0][i][n];
                }
            }
        }

        for i in 0..rows {
            for n in 0..2 {
                offsets[0][i][n] = (((3 - data.block_size) as u32
                    * get_random_number(9, &mut seed[i]))
                    >> 6) as i32;
                for _ in 0..3 {
                    get_random_number(16, &mut seed[i]);
                }
            }
        }

        let ystart = if data.overlap_flag && row_num > 0 {
            2.min(bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx > 0 {
            2.min(bw)
        } else {
            0
        };

        for y in ystart..bh {
            if xstart < bw {
                let si = y as usize * stride + xstart as usize + bx;
                let len = (bw - xstart) as usize;
                let grain = sample_lut_row(
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    0,
                    0,
                    xstart as usize,
                    y as usize,
                );
                fgy_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let grain = blend_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w,
                );
                let si = y as usize * stride + bx;
                fgy_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
        }

        let mut overlap_row = [0i16; GRAIN_WIDTH];
        for y in 0..ystart {
            if xstart < bw {
                let si = y as usize * stride + xstart as usize + bx;
                let len = (bw - xstart) as usize;
                let grain = blend_top_grain_row(
                    &mut overlap_row,
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    xstart as usize,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    w[y as usize][0],
                    w[y as usize][1],
                );
                fgy_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let grain = blend_top_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    0,
                    0,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w,
                    &w,
                );
                let si = y as usize * stride + bx;
                fgy_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                );
            }
        }

        bx += bs;
    }
}

pub(crate) fn fguv_32x32xn_hbd(
    dst: &mut [u16],
    src: &[u16],
    stride: usize,
    data: &FilmGrainData,
    in_seed: u32,
    pw: usize,
    scaling: &[u8],
    grain_lut: &[[i16; GRAIN_WIDTH]],
    bh: i32,
    row_num: i32,
    luma_row: &[u16],
    luma_stride: usize,
    luma_width: usize,
    uv: usize,
    is_id: bool,
    sx: usize,
    sy: usize,
    bitdepth: usize,
) {
    let rows = 1 + (data.overlap_flag && row_num > 0) as usize;
    let bitdepth_min_8 = bitdepth as i32 - 8;
    let grain_ctr = 128 << bitdepth_min_8;
    let grain_min = -grain_ctr;
    let grain_max = grain_ctr - 1;
    let bitdepth_max = (1i32 << bitdepth) - 1;
    let bs = (16 << data.block_size) as usize;

    let (min_value, max_value) = if data.clip_to_restricted_range {
        (
            16i32 << bitdepth_min_8,
            (if is_id { 235i32 } else { 240i32 }) << bitdepth_min_8,
        )
    } else {
        (0, bitdepth_max)
    };

    let mut seed = [0u32; 2];
    for i in 0..rows {
        seed[i] = in_seed;
        seed[i] ^= ((((row_num - i as i32) * 37 + 178) & 0xFF) as u32) << 8;
        seed[i] ^= (((row_num - i as i32) * 173 + 105) & 0xFF) as u32;
    }

    let mut offsets = [[[0i32; 2]; 2]; 2];
    let w: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];

    let mut bx = 0usize;
    while bx < pw {
        let bw = ((bs >> sx).min(pw - bx)) as i32;

        if data.overlap_flag && bx > 0 {
            for i in 0..rows {
                for n in 0..2 {
                    offsets[1][i][n] = offsets[0][i][n];
                }
            }
        }

        for i in 0..rows {
            for n in 0..2 {
                offsets[0][i][n] = (((3 - data.block_size) as u32
                    * get_random_number(9, &mut seed[i]))
                    >> 6) as i32;
                for _ in 0..3 {
                    get_random_number(16, &mut seed[i]);
                }
            }
        }

        let ystart = if data.overlap_flag && row_num > 0 {
            (2 >> sy as i32).min(bh)
        } else {
            0
        };
        let xstart = if data.overlap_flag && bx > 0 {
            (2 >> sx as i32).min(bw)
        } else {
            0
        };

        for y in ystart..bh {
            if xstart < bw {
                let si = y as usize * stride + bx + xstart as usize;
                let len = (bw - xstart) as usize;
                let ly = (y as usize) << sy;
                let grain = sample_lut_row(
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    0,
                    0,
                    xstart as usize,
                    y as usize,
                );
                fguv_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx + xstart as usize,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv] << bitdepth_min_8,
                    data.chroma_scaling_from_luma,
                    bitdepth_max,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let ly = (y as usize) << sy;
                let grain = blend_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w[sx],
                );
                let si = y as usize * stride + bx;
                fguv_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv] << bitdepth_min_8,
                    data.chroma_scaling_from_luma,
                    bitdepth_max,
                );
            }
        }

        let mut overlap_row = [0i16; GRAIN_WIDTH];
        for y in 0..ystart {
            if xstart < bw {
                let si = y as usize * stride + bx + xstart as usize;
                let len = (bw - xstart) as usize;
                let ly = (y as usize) << sy;
                let grain = blend_top_grain_row(
                    &mut overlap_row,
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    xstart as usize,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    w[sy][y as usize][0],
                    w[sy][y as usize][1],
                );
                fguv_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx + xstart as usize,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv] << bitdepth_min_8,
                    data.chroma_scaling_from_luma,
                    bitdepth_max,
                );
            }
            if xstart > 0 {
                let mut left_row = [0i16; 2];
                let len = xstart as usize;
                let ly = (y as usize) << sy;
                let grain = blend_top_left_grain_row(
                    &mut left_row,
                    grain_lut,
                    bs,
                    &offsets,
                    sx,
                    sy,
                    y as usize,
                    len,
                    grain_min,
                    grain_max,
                    &w[sx],
                    &w[sy],
                );
                let si = y as usize * stride + bx;
                fguv_row_hbd_dispatch(
                    &mut dst[si..si + len],
                    &src[si..si + len],
                    grain,
                    &luma_row[ly * luma_stride..],
                    bx,
                    luma_width,
                    sx,
                    scaling,
                    data.scaling_shift,
                    min_value,
                    max_value,
                    data.uv_luma_mult[uv],
                    data.uv_mult[uv],
                    data.uv_offset[uv] << bitdepth_min_8,
                    data.chroma_scaling_from_luma,
                    bitdepth_max,
                );
            }
        }

        bx += bs >> sx;
    }
}

pub(crate) struct GrainLut {
    pub(crate) y: [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    pub(crate) u: [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
    pub(crate) v: [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
}

impl GrainLut {
    pub(crate) fn new() -> Self {
        Self {
            y: [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
            u: [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
            v: [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT],
        }
    }
}

impl Default for GrainLut {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_grain_8bpc(
    fgd: &FilmGrainData,
    grain_lut: &mut GrainLut,
    scaling: &mut [Vec<u8>; 3],
    seed: u32,
    ss_x: bool,
    ss_y: bool,
) {
    // LUT is ALWAYS generated: the chroma LUTs derive from it. The chroma LUTs
    // are generated with the plane's subsampling so the sub-grid dimensions
    generate_grain_y(&mut grain_lut.y, fgd, seed);

    if fgd.num_points[1] > 0 || fgd.chroma_scaling_from_luma {
        generate_grain_uv(&mut grain_lut.u, &grain_lut.y, fgd, seed, 0, ss_x, ss_y);
    }
    if fgd.num_points[2] > 0 || fgd.chroma_scaling_from_luma {
        generate_grain_uv(&mut grain_lut.v, &grain_lut.y, fgd, seed, 1, ss_x, ss_y);
    }

    if fgd.num_points[0] > 0 || fgd.chroma_scaling_from_luma {
        scaling[0].resize(256, 0);
        generate_scaling_8bpc(
            &fgd.points[0][..fgd.num_points[0] as usize],
            scaling[0].as_mut_slice().try_into().unwrap(),
        );
    }

    if !fgd.chroma_scaling_from_luma {
        for uv in 0..2 {
            if fgd.num_points[uv + 1] > 0 {
                scaling[uv + 1].resize(256, 0);
                generate_scaling_8bpc(
                    &fgd.points[uv + 1][..fgd.num_points[uv + 1] as usize],
                    scaling[uv + 1].as_mut_slice().try_into().unwrap(),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_grain_hbd(
    fgd: &FilmGrainData,
    grain_lut: &mut GrainLut,
    scaling: &mut [Vec<u8>; 3],
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: usize,
) {
    let scaling_size = 1usize << bitdepth;

    // The luma grain LUT is always generated because chroma AR can reference it.
    generate_grain_y_hbd(&mut grain_lut.y, fgd, seed, bitdepth);

    if fgd.num_points[1] > 0 || fgd.chroma_scaling_from_luma {
        generate_grain_uv_hbd(
            &mut grain_lut.u,
            &grain_lut.y,
            fgd,
            seed,
            0,
            ss_x,
            ss_y,
            bitdepth,
        );
    }
    if fgd.num_points[2] > 0 || fgd.chroma_scaling_from_luma {
        generate_grain_uv_hbd(
            &mut grain_lut.v,
            &grain_lut.y,
            fgd,
            seed,
            1,
            ss_x,
            ss_y,
            bitdepth,
        );
    }

    if fgd.num_points[0] > 0 || fgd.chroma_scaling_from_luma {
        scaling[0].resize(scaling_size, 0);
        generate_scaling_hbd(
            &fgd.points[0][..fgd.num_points[0] as usize],
            bitdepth,
            scaling[0].as_mut_slice(),
        );
    }

    if !fgd.chroma_scaling_from_luma {
        for uv in 0..2 {
            if fgd.num_points[uv + 1] > 0 {
                scaling[uv + 1].resize(scaling_size, 0);
                generate_scaling_hbd(
                    &fgd.points[uv + 1][..fgd.num_points[uv + 1] as usize],
                    bitdepth,
                    scaling[uv + 1].as_mut_slice(),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_row_hbd(
    dst_y: &mut [u16],
    dst_u: &mut [u16],
    dst_v: &mut [u16],
    src_y: &[u16],
    src_u: &[u16],
    src_v: &[u16],
    y_stride: usize,
    uv_stride: usize,
    fgd: &FilmGrainData,
    grain_lut: &GrainLut,
    scaling: &[Vec<u8>; 3],
    w: usize,
    h: usize,
    row: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: usize,
) {
    let bs = (16usize) << fgd.block_size;
    let row_start = row * bs;

    if fgd.num_points[0] > 0 && !scaling[0].is_empty() {
        let bh = (h - row_start).min(bs);
        let y_off = row_start * y_stride;
        let src_slice = if y_off < src_y.len() {
            &src_y[y_off..]
        } else {
            return;
        };
        let dst_slice = if y_off < dst_y.len() {
            &mut dst_y[y_off..]
        } else {
            return;
        };

        fgy_32x32xn_hbd(
            dst_slice,
            src_slice,
            y_stride,
            fgd,
            seed,
            w,
            scaling[0].as_slice(),
            &grain_lut.y,
            bh as i32,
            row as i32,
            bitdepth,
        );
    }

    let has_uv = |uv: usize| -> bool {
        (fgd.num_points[uv + 1] > 0 || fgd.chroma_scaling_from_luma)
            && !scaling_for_uv(scaling, fgd, uv).is_empty()
    };

    let ch = (((h - row_start).min(bs) + ss_y as usize) >> ss_y as usize) as i32;
    let cw = (w + ss_x as usize) >> ss_x as usize;
    let uv_off = (row_start >> (ss_y as usize)) * uv_stride;
    let luma_off = row_start * y_stride;

    if has_uv(0) && uv_off < src_u.len() && uv_off < dst_u.len() {
        fguv_32x32xn_hbd(
            &mut dst_u[uv_off..],
            &src_u[uv_off..],
            uv_stride,
            fgd,
            seed,
            cw,
            scaling_for_uv(scaling, fgd, 0),
            &grain_lut.u,
            ch,
            row as i32,
            &src_y[luma_off..],
            y_stride,
            w,
            0,
            fgd.mc_identity,
            ss_x as usize,
            ss_y as usize,
            bitdepth,
        );
    }

    if has_uv(1) && uv_off < src_v.len() && uv_off < dst_v.len() {
        fguv_32x32xn_hbd(
            &mut dst_v[uv_off..],
            &src_v[uv_off..],
            uv_stride,
            fgd,
            seed,
            cw,
            scaling_for_uv(scaling, fgd, 1),
            &grain_lut.v,
            ch,
            row as i32,
            &src_y[luma_off..],
            y_stride,
            w,
            1,
            fgd.mc_identity,
            ss_x as usize,
            ss_y as usize,
            bitdepth,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_hbd_mt(
    dst_y: &mut [u16],
    dst_u: &mut [u16],
    dst_v: &mut [u16],
    src_y: &[u16],
    src_u: &[u16],
    src_v: &[u16],
    y_stride: usize,
    uv_stride: usize,
    fgd: &FilmGrainData,
    w: usize,
    h: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    bitdepth: usize,
    n_threads: u32,
    pool: Option<&crate::mtpool::ThreadPool>,
) {
    let mut grain_lut = GrainLut::new();
    let mut scaling = [Vec::new(), Vec::new(), Vec::new()];

    prep_grain_hbd(
        fgd,
        &mut grain_lut,
        &mut scaling,
        seed,
        ss_x,
        ss_y,
        bitdepth,
    );

    let bs = (16usize) << fgd.block_size;
    let rows = h.div_ceil(bs);

    let (active, n_run) = filmgrain_thread_count(rows, n_threads, pool);
    if n_run <= 1 {
        for row in 0..rows {
            apply_grain_row_hbd(
                dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, &grain_lut,
                &scaling, w, h, row, seed, ss_x, ss_y, bitdepth,
            );
        }
        return;
    }

    let dst_y_mut = GrainPlaneMut::new(dst_y);
    let dst_u_mut = GrainPlaneMut::new(dst_u);
    let dst_v_mut = GrainPlaneMut::new(dst_v);
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let job = || loop {
        let row = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if row >= rows {
            break;
        }
        // SAFETY: row is unique per worker and apply_grain_row_hbd writes only
        // that grain block row's destination rows in each plane.
        let dst_y = unsafe { dst_y_mut.whole() };
        let dst_u = unsafe { dst_u_mut.whole() };
        let dst_v = unsafe { dst_v_mut.whole() };
        apply_grain_row_hbd(
            dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, &grain_lut,
            &scaling, w, h, row, seed, ss_x, ss_y, bitdepth,
        );
    };
    crate::mtpool::dispatch(active, n_run, &job);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_row_8bpc(
    dst_y: &mut [u8],
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    y_stride: isize,
    uv_stride: isize,
    fgd: &FilmGrainData,
    grain_lut: &GrainLut,
    scaling: &[Vec<u8>; 3],
    w: usize,
    h: usize,
    row: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
) {
    let bs = (16usize) << fgd.block_size;
    let row_start = row * bs;

    if fgd.num_points[0] > 0 && !scaling[0].is_empty() {
        // `bh = imin(out->p.h - row*bs, bs)`).
        let bh = (h - row_start).min(bs);
        let y_off = row_start * y_stride.unsigned_abs();
        let src_slice = if y_off < src_y.len() {
            &src_y[y_off..]
        } else {
            return;
        };
        let dst_slice = if y_off < dst_y.len() {
            &mut dst_y[y_off..]
        } else {
            return;
        };

        fgy_32x32xn_8bpc(
            dst_slice,
            src_slice,
            y_stride.unsigned_abs(),
            fgd,
            seed,
            w,
            scaling[0].as_slice().try_into().unwrap(),
            &grain_lut.y,
            bh as i32,
            row as i32,
        );
    }

    let has_uv = |uv: usize| -> bool {
        (fgd.num_points[uv + 1] > 0 || fgd.chroma_scaling_from_luma)
            && !scaling_for_uv(scaling, fgd, uv).is_empty()
    };

    let ch = (((h - row_start).min(bs) + ss_y as usize) >> ss_y as usize) as i32;
    let cw = (w + ss_x as usize) >> ss_x as usize;
    let uv_off = (row_start >> (ss_y as usize)) * uv_stride.unsigned_abs();
    let luma_off = row_start * y_stride.unsigned_abs();

    if has_uv(0) && uv_off < src_u.len() && uv_off < dst_u.len() {
        let uv_scaling: &[u8; 256] = scaling_for_uv(scaling, fgd, 0).try_into().unwrap();
        fguv_32x32xn_8bpc(
            &mut dst_u[uv_off..],
            &src_u[uv_off..],
            uv_stride.unsigned_abs(),
            fgd,
            seed,
            cw,
            uv_scaling,
            &grain_lut.u,
            ch,
            row as i32,
            &src_y[luma_off..],
            y_stride.unsigned_abs(),
            w,
            0,
            fgd.mc_identity,
            ss_x as usize,
            ss_y as usize,
        );
    }

    if has_uv(1) && uv_off < src_v.len() && uv_off < dst_v.len() {
        let uv_scaling: &[u8; 256] = scaling_for_uv(scaling, fgd, 1).try_into().unwrap();
        fguv_32x32xn_8bpc(
            &mut dst_v[uv_off..],
            &src_v[uv_off..],
            uv_stride.unsigned_abs(),
            fgd,
            seed,
            cw,
            uv_scaling,
            &grain_lut.v,
            ch,
            row as i32,
            &src_y[luma_off..],
            y_stride.unsigned_abs(),
            w,
            1,
            fgd.mc_identity,
            ss_x as usize,
            ss_y as usize,
        );
    }
}

fn scaling_for_uv<'a>(scaling: &'a [Vec<u8>; 3], fgd: &FilmGrainData, uv: usize) -> &'a [u8] {
    if fgd.chroma_scaling_from_luma {
        &scaling[0]
    } else {
        &scaling[uv + 1]
    }
}

/// Film grain synthesis with optional row-band parallelism.
///
/// The output is partitioned into independent `bs`-tall row bands
/// the destination planes and reads only the (read-only) ungrained `src` planes
/// plus the precomputed `grain_lut`/`scaling`; the per-pixel grain RNG is
/// re-derived from absolute position inside the kernels. The bands therefore
/// touch disjoint output memory and share no mutable state, so distributing them
/// across threads yields output byte-identical to the sequential loop.
///
/// `n_threads <= 1` runs the exact sequential loop (single-thread path
/// unchanged).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_8bpc_mt(
    dst_y: &mut [u8],
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    y_stride: isize,
    uv_stride: isize,
    fgd: &FilmGrainData,
    w: usize,
    h: usize,
    seed: u32,
    ss_x: bool,
    ss_y: bool,
    n_threads: u32,
    pool: Option<&crate::mtpool::ThreadPool>,
) {
    let mut grain_lut = GrainLut::new();
    let mut scaling = [Vec::new(), Vec::new(), Vec::new()];

    prep_grain_8bpc(fgd, &mut grain_lut, &mut scaling, seed, ss_x, ss_y);

    let bs = (16usize) << fgd.block_size;
    let rows = h.div_ceil(bs);

    let (active, n_run) = filmgrain_thread_count(rows, n_threads, pool);
    if n_run <= 1 {
        for row in 0..rows {
            apply_grain_row_8bpc(
                dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, &grain_lut,
                &scaling, w, h, row, seed, ss_x, ss_y,
            );
        }
        return;
    }

    let dst_y_mut = GrainPlaneMut::new(dst_y);
    let dst_u_mut = GrainPlaneMut::new(dst_u);
    let dst_v_mut = GrainPlaneMut::new(dst_v);
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    let job = || loop {
        let row = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if row >= rows {
            break;
        }
        // SAFETY: row is unique per worker and apply_grain_row_8bpc writes only
        // that grain block row's destination rows in each plane.
        let dst_y = unsafe { dst_y_mut.whole() };
        let dst_u = unsafe { dst_u_mut.whole() };
        let dst_v = unsafe { dst_v_mut.whole() };
        apply_grain_row_8bpc(
            dst_y, dst_u, dst_v, src_y, src_u, src_v, y_stride, uv_stride, fgd, &grain_lut,
            &scaling, w, h, row, seed, ss_x, ss_y,
        );
    };
    crate::mtpool::dispatch(active, n_run, &job);
}
