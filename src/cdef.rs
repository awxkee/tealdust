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

use crate::ccso::{ccso_add, ccso_prep};
use crate::intops::{apply_sign, iclip, imax, imin, ulog2};
use crate::pixel::{BitDepth, BitDepth8, Pixel};
use crate::tables::CDEF_DIRECTIONS;

pub(crate) const CDEF_HAVE_LEFT: u8 = 1 << 0;
pub(crate) const CDEF_HAVE_RIGHT: u8 = 1 << 1;
pub(crate) const CDEF_HAVE_TOP: u8 = 1 << 2;
pub(crate) const CDEF_HAVE_BOTTOM: u8 = 1 << 3;

#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_padding<BD: BitDepth>(
    _bd: BD,
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    src_off: usize,
    left: &[[BD::Pixel; 2]],
    top: &[BD::Pixel],
    top_off: usize,
    bottom: &[BD::Pixel],
    bottom_off: usize,
    bottom_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    let o = 2 * tmp_stride + 2;

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    let mut y_start: i32 = -2;
    let mut y_end: i32 = h as i32 + 2;

    if edges & CDEF_HAVE_TOP == 0 {
        let base = o.wrapping_sub(2).wrapping_sub(2 * tmp_stride);
        fill(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_start = 0;
    }
    if edges & CDEF_HAVE_BOTTOM == 0 {
        let base = o + h * tmp_stride - 2;
        fill(&mut tmp[base..], tmp_stride, w + 4, 2);
        y_end -= 2;
    }
    if edges & CDEF_HAVE_LEFT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 - 2) as usize;
        fill(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        let base = (o as i32 + y_start * tmp_stride as i32 + w as i32) as usize;
        fill(&mut tmp[base..], tmp_stride, 2, (y_end - y_start) as usize);
        x_end -= 2;
    }

    let mut toff = top_off;
    for y in y_start..0 {
        for x in x_start..x_end {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = Into::<i32>::into(top[(toff as i32 + x) as usize]) as i16;
        }
        toff += src_stride;
    }

    for y in 0..h as i32 {
        for x in x_start..0 {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = Into::<i32>::into(left[y as usize][(2 + x) as usize]) as i16;
        }
    }

    let mut soff = src_off;
    for y in 0..h as i32 {
        for x in 0..x_end {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = Into::<i32>::into(src[(soff as i32 + x) as usize]) as i16;
        }
        soff += src_stride;
    }

    let mut boff = bottom_off;
    for y in h as i32..y_end {
        for x in x_start..x_end {
            let ti = (o as i32 + x + y * tmp_stride as i32) as usize;
            tmp[ti] = Into::<i32>::into(bottom[(boff as i32 + x) as usize]) as i16;
        }
        boff += bottom_stride;
    }
}

#[inline(always)]
pub(crate) fn constrain(diff: i32, threshold: i32, shift: i32) -> i32 {
    let adiff = diff.abs();
    apply_sign(imin(adiff, imax(0, threshold - (adiff >> shift))), diff)
}

pub(crate) fn fill(tmp: &mut [i16], stride: usize, w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            tmp[y * stride + x] = i16::MIN;
        }
    }
}

pub(crate) fn cdef_find_dir_bd<BD: BitDepth>(
    bd: BD,
    img: &[BD::Pixel],
    stride: usize,
    var: &mut u32,
) -> i32 {
    let bitdepth_min_8 = bd.bitdepth_min_8();
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        for x in 0..8usize {
            let px = (Into::<i32>::into(img[y * stride + x]) >> bitdepth_min_8) - 128;

            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_hv[0][y] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_hv[1][x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
    }

    let mut cost = [0u32; 8];
    for n in 0..8 {
        cost[2] += (partial_sum_hv[0][n] * partial_sum_hv[0][n]) as u32;
        cost[6] += (partial_sum_hv[1][n] * partial_sum_hv[1][n]) as u32;
    }
    cost[2] *= 105;
    cost[6] *= 105;

    static DIV_TABLE: [u32; 7] = [840, 420, 280, 210, 168, 140, 120];
    for n in 0..7usize {
        let d = DIV_TABLE[n];
        cost[0] += ((partial_sum_diag[0][n] * partial_sum_diag[0][n]
            + partial_sum_diag[0][14 - n] * partial_sum_diag[0][14 - n])
            as u32)
            * d;
        cost[4] += ((partial_sum_diag[1][n] * partial_sum_diag[1][n]
            + partial_sum_diag[1][14 - n] * partial_sum_diag[1][14 - n])
            as u32)
            * d;
    }
    cost[0] += (partial_sum_diag[0][7] * partial_sum_diag[0][7]) as u32 * 105;
    cost[4] += (partial_sum_diag[1][7] * partial_sum_diag[1][7]) as u32 * 105;

    for n in 0..4usize {
        let ci = n * 2 + 1;
        for m in 0..5usize {
            cost[ci] += (partial_sum_alt[n][3 + m] * partial_sum_alt[n][3 + m]) as u32;
        }
        cost[ci] *= 105;
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            cost[ci] += ((partial_sum_alt[n][m] * partial_sum_alt[n][m]
                + partial_sum_alt[n][10 - m] * partial_sum_alt[n][10 - m])
                as u32)
                * d;
        }
    }

    let mut best_dir = 0i32;
    let mut best_cost = cost[0];
    for n in 1..8 {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n as i32;
        }
    }

    *var = (best_cost - cost[(best_dir ^ 4) as usize]) >> 10;
    best_dir
}

pub(crate) fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if var >> 6 != 0 {
        imin(ulog2(var >> 6), 12)
    } else {
        0
    };
    (strength * (4 + i) + 8) >> 4
}

pub(crate) const BACKUP_2X8_Y: u8 = 1 << 0;
pub(crate) const BACKUP_2X8_UV: u8 = 1 << 1;

#[inline]
fn copy_cdef_bottom_ref<P: Pixel>(
    dst: &mut [P],
    compact_stride: usize,
    src: &[P],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    dst.fill(P::default());
    if edges & CDEF_HAVE_BOTTOM == 0 {
        return;
    }

    let mut x_start: i32 = -2;
    let mut x_end: i32 = w as i32 + 2;
    if edges & CDEF_HAVE_LEFT == 0 {
        x_start = 0;
    }
    if edges & CDEF_HAVE_RIGHT == 0 {
        x_end -= 2;
    }

    for row in 0..2usize {
        let base = src_off + (h + row) * src_stride;
        let dst_base = row * compact_stride + 2;
        for x in x_start..x_end {
            dst[(dst_base as i32 + x) as usize] = src[(base as i32 + x) as usize];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    dst_off: usize,
    left: &[[BD::Pixel; 2]],
    top: &[BD::Pixel],
    top_off: usize,
    bottom: &[BD::Pixel],
    bottom_off: usize,
    bottom_stride: usize,
    pri_strength: i32,
    sec_strength: i32,
    dir: usize,
    damping: i32,
    w: usize,
    h: usize,
    edges: u8,
) {
    let tmp_stride: usize = 12;
    let mut tmp_buf = [0i16; 144];
    let o = 2 * tmp_stride + 2;

    cdef_padding(
        bd,
        &mut tmp_buf,
        tmp_stride,
        &*dst,
        dst_stride,
        dst_off,
        left,
        top,
        top_off,
        bottom,
        bottom_off,
        bottom_stride,
        w,
        h,
        edges,
    );

    let bitdepth_min_8 = bd.bitdepth_min_8();
    let mut dp = dst_off;
    let mut tp = o;

    if pri_strength != 0 {
        let pri_tap = 4 - ((pri_strength >> bitdepth_min_8) & 1);
        let pri_shift = imax(0, damping - ulog2(pri_strength as u32));
        if sec_strength != 0 {
            let sec_shift = damping - ulog2(sec_strength as u32);
            for _y in 0..h {
                for x in 0..w {
                    let px = Into::<i32>::into(dst[dp + x]);
                    let mut sum = 0i32;
                    let mut max_v = px;
                    let mut min_v = px;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off1 = CDEF_DIRECTIONS[dir + 2][k] as isize;
                        let p0 = tmp_buf[((tp + x) as isize + off1) as usize] as i32;
                        let p1 = tmp_buf[((tp + x) as isize - off1) as usize] as i32;
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = (pri_tap_k & 3) | 2;
                        min_v = imin(p0, min_v);
                        max_v = imax(p0, max_v);
                        min_v = imin(p1, min_v);
                        max_v = imax(p1, max_v);
                        let off2 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                        let off3 = CDEF_DIRECTIONS[dir][k] as isize;
                        let s0 = tmp_buf[((tp + x) as isize + off2) as usize] as i32;
                        let s1 = tmp_buf[((tp + x) as isize - off2) as usize] as i32;
                        let s2 = tmp_buf[((tp + x) as isize + off3) as usize] as i32;
                        let s3 = tmp_buf[((tp + x) as isize - off3) as usize] as i32;
                        let sec_tap = 2 - k as i32;
                        sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                        sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                        min_v = imin(s0, min_v);
                        max_v = imax(s0, max_v);
                        min_v = imin(s1, min_v);
                        max_v = imax(s1, max_v);
                        min_v = imin(s2, min_v);
                        max_v = imax(s2, max_v);
                        min_v = imin(s3, min_v);
                        max_v = imax(s3, max_v);
                    }
                    dst[dp + x] = BD::Pixel::from_i32(iclip(
                        px + ((sum - (sum < 0) as i32 + 8) >> 4),
                        min_v,
                        max_v,
                    ));
                }
                dp += dst_stride;
                tp += tmp_stride;
            }
        } else {
            for _y in 0..h {
                for x in 0..w {
                    let px = Into::<i32>::into(dst[dp + x]);
                    let mut sum = 0i32;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2 {
                        let off = CDEF_DIRECTIONS[dir + 2][k] as isize;
                        let p0 = tmp_buf[((tp + x) as isize + off) as usize] as i32;
                        let p1 = tmp_buf[((tp + x) as isize - off) as usize] as i32;
                        sum += pri_tap_k * constrain(p0 - px, pri_strength, pri_shift);
                        sum += pri_tap_k * constrain(p1 - px, pri_strength, pri_shift);
                        pri_tap_k = (pri_tap_k & 3) | 2;
                    }
                    dst[dp + x] = BD::Pixel::from_i32(px + ((sum - (sum < 0) as i32 + 8) >> 4));
                }
                dp += dst_stride;
                tp += tmp_stride;
            }
        }
    } else {
        let sec_shift = damping - ulog2(sec_strength as u32);
        for _y in 0..h {
            for x in 0..w {
                let px = Into::<i32>::into(dst[dp + x]);
                let mut sum = 0i32;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off2 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp_buf[((tp + x) as isize + off1) as usize] as i32;
                    let s1 = tmp_buf[((tp + x) as isize - off1) as usize] as i32;
                    let s2 = tmp_buf[((tp + x) as isize + off2) as usize] as i32;
                    let s3 = tmp_buf[((tp + x) as isize - off2) as usize] as i32;
                    let sec_tap = 2 - k as i32;
                    sum += sec_tap * constrain(s0 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s1 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s2 - px, sec_strength, sec_shift);
                    sum += sec_tap * constrain(s3 - px, sec_strength, sec_shift);
                }
                dst[dp + x] = BD::Pixel::from_i32(px + ((sum - (sum < 0) as i32 + 8) >> 4));
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    }
}
/// Per-superblock-row CDEF parameters threaded from the filter driver.
/// Per-plane CCSO configuration (cross-component sample offset), from the frame
/// header's `ccso.p[pl]`. `quant_step` and `offset_lut` are resolved from the
/// scale/quant indices.
#[derive(Clone, Copy)]
pub(crate) struct CcsoPlaneCfg {
    pub(crate) max_band_log2: u32,
    pub(crate) ext_filter: usize,
    pub(crate) quant_step: i32,
    pub(crate) edge_clf: u32,
    pub(crate) bo_only: bool,
    pub(crate) scale_idx: usize,
    pub(crate) filter_off: [u8; 64],
}

/// CCSO configuration for a CDEF brow. Per-SB enable flags live in
/// `CdefBrowParams::mask`; this struct only carries the resolved header config.
pub(crate) struct CcsoCfg {
    pub(crate) p: [CcsoPlaneCfg; 3],
}

pub(crate) struct CdefBrowParams<'a> {
    pub(crate) bw: i32,
    pub(crate) bh: i32,
    pub(crate) damping: i32,
    pub(crate) layout: crate::headers::PixelLayout,
    pub(crate) on_skip_tx: bool,
    pub(crate) cdef_on: bool,
    /// Per-SB256 filter masks for this CDEF row. The row is borrowed directly
    /// from `lf.mask`; CDEF no longer allocates per-field row Vecs.
    pub(crate) mask: &'a [crate::lf_mask::Av2Filter],
    /// Per-cdef-index raw strengths (0..n_strengths) for Y and UV.
    pub(crate) y_strength: &'a [u8],
    pub(crate) uv_strength: &'a [u8],
    /// Per-SB256 lossless masks; CDEF (and CCSO) skip losslessly-coded blocks.
    /// `any_lossless` is false unless the frame is segmented with a lossless seg.
    pub(crate) any_lossless: bool,
    /// CCSO config; `None` when CCSO is disabled for this frame.
    pub(crate) ccso: Option<CcsoCfg>,
}

impl<'a> CdefBrowParams<'a> {
    #[inline(always)]
    fn mask_at(&self, sb256x: usize) -> Option<&crate::lf_mask::Av2Filter> {
        self.mask.get(sb256x)
    }

    #[inline(always)]
    fn cdef_idx(&self, sb256x: usize, sb64_idx: usize) -> i8 {
        self.mask_at(sb256x)
            .map(|m| m.cdef_idx[sb64_idx])
            .unwrap_or(-1)
    }

    #[inline(always)]
    fn ccso_mask(&self, sb256x: usize) -> [u8; 3] {
        self.mask_at(sb256x).map(|m| m.ccso).unwrap_or([0; 3])
    }

    #[inline(always)]
    fn noskip_mask(&self, sb256x: usize, by_idx: usize, sb64x_idx: usize) -> u16 {
        if self.on_skip_tx {
            !0u16
        } else {
            self.mask_at(sb256x)
                .filter(|_| by_idx < 32 && sb64x_idx < 4)
                .map(|m| m.noskip_mask[by_idx][sb64x_idx])
                .unwrap_or(0)
        }
    }

    #[inline(always)]
    fn lossless_y(&self, sb256x: usize) -> Option<&[[u16; 4]; 64]> {
        self.mask_at(sb256x).map(|m| &m.lossless_mask_y)
    }

    #[inline(always)]
    fn lossless_uv(&self, sb256x: usize) -> Option<&[[u16; 4]; 64]> {
        self.mask_at(sb256x).map(|m| &m.lossless_mask_uv)
    }
}

static UV_DIRS: [[u8; 8]; 2] = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 0, 2, 4, 5, 6, 6, 6]];

/// Backup the bottom 2 rows of the current 8-row band into a toggled CDEF line
/// bank (each plane bank is laid out with the plane's positive stride spacing so
fn cdef_backup2lines_bank<P: Pixel>(
    bank: &mut [Vec<P>; 3],
    src_y: &[P],
    src_u: &[P],
    src_v: &[P],
    y_off: usize,
    uv_off: usize,
    y_stride: usize,
    uv_stride: usize,
    layout: crate::headers::PixelLayout,
) {
    // Luma: copy rows 6,7 of the band (`src + 6*stride`, 2*stride bytes).
    let s = y_off + 6 * y_stride;
    let n = (2 * y_stride)
        .min(src_y.len().saturating_sub(s))
        .min(bank[0].len());
    bank[0][..n].copy_from_slice(&src_y[s..s + n]);

    if layout != crate::headers::PixelLayout::I400 {
        let uv_off_rows = if layout == crate::headers::PixelLayout::I420 {
            2
        } else {
            6
        };
        let s = uv_off + uv_off_rows * uv_stride;
        let n = (2 * uv_stride)
            .min(src_u.len().saturating_sub(s))
            .min(bank[1].len());
        bank[1][..n].copy_from_slice(&src_u[s..s + n]);
        let n = (2 * uv_stride)
            .min(src_v.len().saturating_sub(s))
            .min(bank[2].len());
        bank[2][..n].copy_from_slice(&src_v[s..s + n]);
    }
}

/// Backup a pre-CDEF 2x8 left-column block from a plane into `dst[8]`
fn cdef_backup2x8<P: Pixel>(
    dst: &mut [[P; 2]; 8],
    src: &[P],
    base: usize,
    stride: usize,
    x_off: usize,
    rows: usize,
) {
    let mut off = base;
    for d in dst.iter_mut().take(rows) {
        let s = off + x_off - 2;
        d[0] = src[s];
        d[1] = src[s + 1];
        off += stride;
    }
}

/// `have_tt == 0` path). `cdef_line` is the toggled top-row backup whose `tf`
/// bank holds the previous band's bottom 2 rows; `*toggle` flips per 8-row band.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_brow_8bpc(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    p: &CdefBrowParams,
    y_stride: isize,
    uv_stride: isize,
    cdef_line: &mut [[Vec<u8>; 3]; 2],
    toggle: &mut usize,
    by_start: i32,
    by_end: i32,
    sby: i32,
    sbrow_start: bool,
) {
    cdef_brow(
        BitDepth8,
        y,
        u,
        v,
        p,
        y_stride,
        uv_stride,
        cdef_line,
        toggle,
        by_start,
        by_end,
        sby,
        sbrow_start,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_brow<BD: BitDepth>(
    bd: BD,
    y: &mut [BD::Pixel],
    u: &mut [BD::Pixel],
    v: &mut [BD::Pixel],
    p: &CdefBrowParams,
    y_stride: isize,
    uv_stride: isize,
    cdef_line: &mut [[Vec<BD::Pixel>; 3]; 2],
    toggle: &mut usize,
    by_start: i32,
    by_end: i32,
    sby: i32,
    sbrow_start: bool,
) {
    let _ = sby;
    let bitdepth_min_8 = bd.bitdepth_min_8();
    let damping = p.damping + bitdepth_min_8;
    let y_ls = y_stride.unsigned_abs();
    let uv_ls = uv_stride.unsigned_abs();
    let layout = p.layout;
    let ss_hor = (layout != crate::headers::PixelLayout::I444) as usize;
    let ss_ver = (layout == crate::headers::PixelLayout::I420) as usize;
    let uv_dir = &UV_DIRS[(layout == crate::headers::PixelLayout::I422) as usize];
    let sbsz = 16i32;
    let sb64w = (p.bw + sbsz - 1) >> 4;
    let have_chroma = layout != crate::headers::PixelLayout::I400;

    // Plane base offset of the band's first row.
    let mut row_y = by_start as usize * 4 * y_ls;
    let mut row_uv = ((by_start as usize * 4) >> ss_ver) * uv_ls;

    let mut edge_top = if by_start > 0 { CDEF_HAVE_TOP } else { 0 };

    let mut by = by_start;
    while by < by_end {
        let tf = *toggle;
        let by_idx = ((by & 0x3e) >> 1) as usize;
        let mut edges = edge_top | CDEF_HAVE_BOTTOM;
        if by + 2 >= p.bh {
            edges &= !CDEF_HAVE_BOTTOM;
        }

        // Back up pre-filter bottom 2 rows of this band for the next band's top.
        // single-thread (have_tt == 0) path the `!have_tt` term always holds, so
        // every band with HAVE_BOTTOM backs up (needed so the next superblock-
        // row's CDEF seam re-filter reads the correct top line).
        let _ = (sbrow_start, by_end);
        if (edges & CDEF_HAVE_BOTTOM) != 0 {
            let other = 1 - tf;
            cdef_backup2lines_bank(
                &mut cdef_line[other],
                y,
                u,
                v,
                row_y,
                row_uv,
                y_ls,
                uv_ls,
                layout,
            );
        }

        // Left 2x8 backups (toggled `bit`), one per plane, pre-CDEF.
        let mut lr_bak: [[[[BD::Pixel; 2]; 8]; 3]; 2] = [[[[BD::Pixel::default(); 2]; 8]; 3]; 2];
        let mut bit = 0usize;
        edges &= !CDEF_HAVE_LEFT;
        edges |= CDEF_HAVE_RIGHT;
        // the next SB's first block can reuse / skip the left-column backup.
        let mut prev_flag = 0u8;

        // Per-sb base offsets that advance with iptrs.
        let mut sb_y = row_y;
        let mut sb_uv = row_uv;

        // CCSO LUT-index scratch (per-SB, recomputed each sbx). Sized for the
        let mut ccso_lut_idx: [[u8; 64 * 8]; 3] = [[0u8; 64 * 8]; 3];

        for sbx in 0..sb64w {
            let sb256x = (sbx >> 2) as usize;
            let sb64x_idx = (sbx & 3) as usize;
            let sb64_idx = (((by & 0x30) >> 2) + (sbx & 3)) as usize;
            let cdef_idx = p.cdef_idx(sb256x, sb64_idx);

            if let Some(cc) = &p.ccso {
                let ccm = p.ccso_mask(sb256x);
                let flag = ccm[0] | ccm[1] | ccm[2];
                let do_left = flag & !prev_flag;
                prev_flag |= flag;
                if do_left != 0 && (edges & CDEF_HAVE_LEFT) != 0 {
                    if do_left & BACKUP_2X8_Y != 0 {
                        cdef_backup2x8(&mut lr_bak[bit][0], y, sb_y, y_ls, 0, 8);
                    }
                    if have_chroma && do_left & BACKUP_2X8_UV != 0 {
                        cdef_backup2x8(&mut lr_bak[bit][1], u, sb_uv, uv_ls, 0, 8 >> ss_ver);
                        cdef_backup2x8(&mut lr_bak[bit][2], v, sb_uv, uv_ls, 0, 8 >> ss_ver);
                    }
                }
                for pl in 0..3 {
                    if ccm[pl] == 0 {
                        continue;
                    }
                    let cfg = &cc.p[pl];
                    let pl_ss_hor = if pl != 0 { ss_hor } else { 0 };
                    let pl_ss_ver = if pl != 0 { ss_ver } else { 0 };
                    let mut sb_edges = edges;
                    if (sbx + 1) * sbsz >= p.bw {
                        sb_edges &= !CDEF_HAVE_RIGHT;
                    }
                    let w_full = imin(sbsz, p.bw - sbx * sbsz) * 4;
                    if w_full <= 0 {
                        continue;
                    }
                    // top/bottom luma lines (single-thread sb_st_y case).
                    let top_off = sbx as usize * (sbsz as usize) * 4;
                    let bot_off = sb_y + 8 * y_ls;
                    ccso_prep(
                        bd,
                        &mut ccso_lut_idx[pl],
                        64 >> pl_ss_hor,
                        y,
                        y_ls,
                        sb_y,
                        &lr_bak[bit][0],
                        &cdef_line[tf][0],
                        top_off,
                        y,
                        bot_off,
                        cfg.max_band_log2,
                        cfg.ext_filter,
                        cfg.quant_step,
                        cfg.edge_clf,
                        cfg.bo_only,
                        sb_edges,
                        (w_full >> pl_ss_hor) as usize,
                        (8 >> pl_ss_ver) as usize,
                        pl_ss_hor,
                        pl_ss_ver,
                    );
                }
            }

            let cdef_active = !(cdef_idx == -1
                || !p.cdef_on
                || ((p
                    .y_strength
                    .get(cdef_idx.max(0) as usize)
                    .copied()
                    .unwrap_or(0)
                    == 0)
                    && (p
                        .uv_strength
                        .get(cdef_idx.max(0) as usize)
                        .copied()
                        .unwrap_or(0)
                        == 0)));

            if cdef_active {
                let noskip_full = p.noskip_mask(sb256x, by_idx, sb64x_idx);

                let y_lvl = p.y_strength[cdef_idx as usize] as i32;
                let uv_lvl = p.uv_strength[cdef_idx as usize] as i32;
                let flag = (y_lvl != 0) as u8 + (((uv_lvl != 0) as u8) << 1);

                let y_pri_lvl = (y_lvl >> 2) << bitdepth_min_8;
                let mut y_sec_lvl = y_lvl & 3;
                y_sec_lvl += (y_sec_lvl == 3) as i32;
                y_sec_lvl <<= bitdepth_min_8;

                let uv_pri_lvl = (uv_lvl >> 2) << bitdepth_min_8;
                let mut uv_sec_lvl = uv_lvl & 3;
                uv_sec_lvl += (uv_sec_lvl == 3) as i32;
                uv_sec_lvl <<= bitdepth_min_8;

                // adjacent 4px rows at column sb64x_idx for luma, subsampled for uv.
                // Zero unless the frame is segmented-lossless.
                let (y_ll0, y_ll1, uv_ll0, uv_ll1) = if p.any_lossless {
                    let yr = (2 * by_idx).min(63);
                    let yl = p.lossless_y(sb256x);
                    let ul = p.lossless_uv(sb256x);
                    let y0 = yl.map(|m| m[yr][sb64x_idx]).unwrap_or(0);
                    let y1 = yl.map(|m| m[(yr + 1).min(63)][sb64x_idx]).unwrap_or(0);
                    let uvr = ((2 * by_idx) >> ss_ver).min(63);
                    let u0 = ul.map(|m| m[uvr][sb64x_idx]).unwrap_or(0);
                    let u1 = ul
                        .map(|m| m[(uvr + (1 - ss_ver)).min(63)][sb64x_idx])
                        .unwrap_or(0);
                    (y0, y1, u0, u1)
                } else {
                    (0, 0, 0, 0)
                };

                let mut b_y = sb_y;
                let mut b_uv = sb_uv;
                let mut bx = sbx * sbsz;
                let sb_bx_end = imin((sbx + 1) * sbsz, p.bw);
                while bx < sb_bx_end {
                    if bx + 2 >= p.bw {
                        edges &= !CDEF_HAVE_RIGHT;
                    }

                    let bx_mask = 3u16 << (bx & 14);
                    let y_lossless = ((y_ll0 | y_ll1) & bx_mask) != 0;
                    let uvbx_mask = (3u16 >> ss_hor) << ((bx & 14) >> ss_hor);
                    let uv_lossless = ((uv_ll0 | uv_ll1) & uvbx_mask) != 0;
                    if (noskip_full & bx_mask) == 0 || (y_lossless && uv_lossless) {
                        prev_flag = 0;
                        edges |= CDEF_HAVE_LEFT;
                        b_y += 8;
                        b_uv += 8 >> ss_hor;
                        bx += 2;
                        continue;
                    }

                    let do_left = flag & !prev_flag;
                    prev_flag = flag;
                    if do_left != 0 && (edges & CDEF_HAVE_LEFT) != 0 {
                        if do_left & BACKUP_2X8_Y != 0 {
                            cdef_backup2x8(&mut lr_bak[bit][0], y, b_y, y_ls, 0, 8);
                        }
                        if have_chroma && do_left & BACKUP_2X8_UV != 0 {
                            cdef_backup2x8(&mut lr_bak[bit][1], u, b_uv, uv_ls, 0, 8 >> ss_ver);
                            cdef_backup2x8(&mut lr_bak[bit][2], v, b_uv, uv_ls, 0, 8 >> ss_ver);
                        }
                    }
                    if (edges & CDEF_HAVE_RIGHT) != 0 {
                        let other = 1 - bit;
                        if flag & BACKUP_2X8_Y != 0 {
                            cdef_backup2x8(&mut lr_bak[other][0], y, b_y, y_ls, 8, 8);
                        }
                        if have_chroma && flag & BACKUP_2X8_UV != 0 {
                            cdef_backup2x8(
                                &mut lr_bak[other][1],
                                u,
                                b_uv,
                                uv_ls,
                                8 >> ss_hor,
                                8 >> ss_ver,
                            );
                            cdef_backup2x8(
                                &mut lr_bak[other][2],
                                v,
                                b_uv,
                                uv_ls,
                                8 >> ss_hor,
                                8 >> ss_ver,
                            );
                        }
                    }

                    let mut variance = 0u32;
                    let dir = if y_pri_lvl != 0 || uv_pri_lvl != 0 {
                        cdef_find_dir_bd(bd, &y[b_y..], y_ls, &mut variance) as usize
                    } else {
                        0
                    };

                    // Luma top/bottom: top from the toggled pre-CDEF line bank.
                    // The bottom reference only needs the two rows around this
                    // 8x8 block, so copy a compact stack buffer instead of
                    // cloning the whole plane.
                    const CDEF_REF_STRIDE: usize = 12;
                    let top_col = bx as usize * 4;
                    let mut y_bottom = [BD::Pixel::default(); 2 * CDEF_REF_STRIDE];
                    copy_cdef_bottom_ref(&mut y_bottom, CDEF_REF_STRIDE, y, b_y, y_ls, 8, 8, edges);
                    if y_pri_lvl != 0 {
                        let adj = adjust_strength(y_pri_lvl, variance);
                        if (adj != 0 || y_sec_lvl != 0) && !y_lossless {
                            cdef_filter_block(
                                bd,
                                y,
                                y_ls,
                                b_y,
                                &lr_bak[bit][0],
                                &cdef_line[tf][0],
                                top_col,
                                &y_bottom,
                                2,
                                CDEF_REF_STRIDE,
                                adj,
                                y_sec_lvl,
                                dir,
                                damping,
                                8,
                                8,
                                edges,
                            );
                        }
                    } else if y_sec_lvl != 0 && !y_lossless {
                        cdef_filter_block(
                            bd,
                            y,
                            y_ls,
                            b_y,
                            &lr_bak[bit][0],
                            &cdef_line[tf][0],
                            top_col,
                            &y_bottom,
                            2,
                            CDEF_REF_STRIDE,
                            0,
                            y_sec_lvl,
                            0,
                            damping,
                            8,
                            8,
                            edges,
                        );
                    }

                    if uv_lvl != 0 && have_chroma && !uv_lossless {
                        let uvdir = if uv_pri_lvl != 0 {
                            uv_dir[dir] as usize
                        } else {
                            0
                        };
                        let cw = 8 >> ss_hor;
                        let ch = 8 >> ss_ver;
                        let top_col_uv = (bx as usize * 4) >> ss_hor;
                        let mut u_bottom = [BD::Pixel::default(); 2 * CDEF_REF_STRIDE];
                        let mut v_bottom = [BD::Pixel::default(); 2 * CDEF_REF_STRIDE];
                        copy_cdef_bottom_ref(
                            &mut u_bottom,
                            CDEF_REF_STRIDE,
                            u,
                            b_uv,
                            uv_ls,
                            cw,
                            ch,
                            edges,
                        );
                        copy_cdef_bottom_ref(
                            &mut v_bottom,
                            CDEF_REF_STRIDE,
                            v,
                            b_uv,
                            uv_ls,
                            cw,
                            ch,
                            edges,
                        );
                        cdef_filter_block(
                            bd,
                            u,
                            uv_ls,
                            b_uv,
                            &lr_bak[bit][1],
                            &cdef_line[tf][1],
                            top_col_uv,
                            &u_bottom,
                            2,
                            CDEF_REF_STRIDE,
                            uv_pri_lvl,
                            uv_sec_lvl,
                            uvdir,
                            damping - 1,
                            cw,
                            ch,
                            edges,
                        );
                        cdef_filter_block(
                            bd,
                            v,
                            uv_ls,
                            b_uv,
                            &lr_bak[bit][2],
                            &cdef_line[tf][2],
                            top_col_uv,
                            &v_bottom,
                            2,
                            CDEF_REF_STRIDE,
                            uv_pri_lvl,
                            uv_sec_lvl,
                            uvdir,
                            damping - 1,
                            cw,
                            ch,
                            edges,
                        );
                    }

                    bit ^= 1;
                    edges |= CDEF_HAVE_LEFT;
                    b_y += 8;
                    b_uv += 8 >> ss_hor;
                    bx += 2;
                }
            } else {
                // prev_flag); CCSO add still runs below.
                prev_flag = 0;
                edges |= CDEF_HAVE_LEFT;
            }

            if let Some(cc) = &p.ccso {
                let ccm = p.ccso_mask(sb256x);
                let flag = ccm[0] | ((ccm[1] | ccm[2]) << 1);
                let do_right = flag & !prev_flag;
                if do_right != 0 && (sbx + 1) * sbsz < p.bw {
                    if do_right & BACKUP_2X8_Y != 0 {
                        cdef_backup2x8(&mut lr_bak[bit][0], y, sb_y, y_ls, sbsz as usize * 4, 8);
                    }
                    if have_chroma && do_right & BACKUP_2X8_UV != 0 {
                        cdef_backup2x8(
                            &mut lr_bak[bit][1],
                            u,
                            sb_uv,
                            uv_ls,
                            (sbsz as usize * 4) >> ss_hor,
                            8 >> ss_ver,
                        );
                        cdef_backup2x8(
                            &mut lr_bak[bit][2],
                            v,
                            sb_uv,
                            uv_ls,
                            (sbsz as usize * 4) >> ss_hor,
                            8 >> ss_ver,
                        );
                    }
                    prev_flag |= do_right;
                }
                // lossless masks for the add (zero for non-segmented-lossless).
                let by_idx_ll = (2 * by_idx) as usize;
                for pl in 0..3 {
                    if ccm[pl] == 0 {
                        continue;
                    }
                    let cfg = &cc.p[pl];
                    let pl_ss_hor = if pl != 0 { ss_hor } else { 0 };
                    let pl_ss_ver = if pl != 0 { ss_ver } else { 0 };
                    let w_full = imin(sbsz, p.bw - sbx * sbsz) * 4;
                    let (dst, dst_off, dst_ls): (&mut [BD::Pixel], usize, usize) = if pl == 0 {
                        (y, sb_y, y_ls)
                    } else if pl == 1 {
                        (u, sb_uv, uv_ls)
                    } else {
                        (v, sb_uv, uv_ls)
                    };
                    // sb64x_idx] and the kernel reads [yy>>2][0]. Build a per-row
                    // view whose column 0 is the sb64x_idx column (all-zero unless
                    // segmented-lossless). 2 rows (8px) cover one CCSO band.
                    let mut ll_buf = [[0u16; 4]; 2];
                    if p.any_lossless {
                        let src = if pl == 0 {
                            p.lossless_y(sb256x)
                        } else {
                            p.lossless_uv(sb256x)
                        };
                        if let Some(m) = src {
                            let base = by_idx_ll >> pl_ss_ver;
                            for (r, slot) in ll_buf.iter_mut().enumerate() {
                                let row = base + r;
                                if row < 64 {
                                    slot[0] = m[row][sb64x_idx];
                                }
                            }
                        }
                    }
                    let ll: &[[u16; 4]] = &ll_buf;
                    ccso_add(
                        bd,
                        &mut dst[dst_off..],
                        dst_ls,
                        &ccso_lut_idx[pl],
                        (64 >> pl_ss_hor) as usize,
                        &cfg.filter_off,
                        &crate::tables::CCSO_OFFSET[cfg.scale_idx],
                        (w_full >> pl_ss_hor) as usize,
                        (8 >> pl_ss_ver) as usize,
                        ll,
                    );
                }
            }

            sb_y += (sbsz * 4) as usize;
            sb_uv += ((sbsz * 4) as usize) >> ss_hor;
        }

        row_y += 8 * y_ls;
        row_uv += (8 * uv_ls) >> ss_ver;
        *toggle ^= 1;
        edge_top = CDEF_HAVE_TOP;
        let _ = by_idx;
        by += 2;
    }
}
