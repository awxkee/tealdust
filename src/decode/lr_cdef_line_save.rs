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
pub(crate) const CDEF_LINE_ROWS_Y: usize = 6;
/// Number of chroma rows stored per tile row (`tile_row_m1 * 2 * stride + x`,
/// consumed from offset 0).
pub(crate) const CDEF_LINE_ROWS_UV: usize = 2;

/// Ensure `lr_cdef_line[plane]` is sized to hold `n_tile_rows` blocks at the
/// plane stride: `n_tile_rows * 6 * y_stride` for luma and
/// `n_tile_rows * 2 * uv_stride` for chroma. No-op once at full size.
pub(crate) fn ensure_lr_cdef_line(
    lr_cdef_line: &mut [Vec<u8>; 3],
    n_tile_rows: usize,
    y_stride: isize,
    uv_stride: isize,
    mono: bool,
) -> Result<(), ()> {
    let y_ls = y_stride.unsigned_abs();
    let uv_ls = uv_stride.unsigned_abs();
    let need_y = n_tile_rows * CDEF_LINE_ROWS_Y * y_ls;
    let need_uv = n_tile_rows * CDEF_LINE_ROWS_UV * uv_ls;

    fn ensure_plane(v: &mut Vec<u8>, len: usize) -> Result<(), ()> {
        if v.len() != len {
            if len > v.len() {
                v.try_reserve_exact(len - v.len()).map_err(|_| ())?;
            }
            v.resize(len, 0);
        }
        Ok(())
    }

    ensure_plane(&mut lr_cdef_line[0], need_y)?;
    if mono {
        lr_cdef_line[1].clear();
        lr_cdef_line[2].clear();
    } else {
        ensure_plane(&mut lr_cdef_line[1], need_uv)?;
        ensure_plane(&mut lr_cdef_line[2], need_uv)?;
    }
    Ok(())
}

/// Save the CDEF-filtered bottom rows of tile row `tile_row` into
/// `lr_cdef_line`, to be read as the integrated top context by the first
/// sbrow of tile row `tile_row + 1`.
///
/// `src` are the *post-CDEF* planes (the in-place `dst_y/u/v` after the CDEF
/// pass for this tile row has completed). `tile_row_bottom_y` is the luma row
/// index (0-based, in pixels) of the last row of this tile row, i.e. the tile
/// row covers luma rows `..=tile_row_bottom_y`. We copy the 6 luma rows ending
/// at `tile_row_bottom_y` so that the 4 rows the kernel consumes (block rows
/// 2..6) are the bottom-most 4 CDEF rows of the tile row — the rows physically
/// adjacent to the seam.
///
/// Geometry is the exact inverse of the `lr_stripe` read:
///   luma  dst slot base = tile_row * 6 * y_ls,  rows 0..6
///   chroma dst slot base = tile_row * 2 * uv_ls, rows 0..2
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_lr_cdef_line_8bpc(
    lr_cdef_line: &mut [Vec<u8>; 3],
    src: &[&[u8]; 3],
    strides: &[isize; 2], // [y_stride, uv_stride]
    tile_row: usize,
    tile_row_bottom_y: i32, // luma pixel row index of this tile row's last row
    frame_w: i32,           // luma plane width in pixels
    frame_h: i32,           // luma plane height in pixels
    ss_hor: i32,
    ss_ver: i32,
    mono: bool,
) {
    let y_ls = strides[0].unsigned_abs();
    let uv_ls = strides[1].unsigned_abs();

    // Luma. Copy 6 rows ending at tile_row_bottom_y (clamped to plane).
    {
        let dst_base = tile_row * CDEF_LINE_ROWS_Y * y_ls;
        let w = (frame_w as usize).min(y_ls);
        // Source rows: the 6 rows whose bottom row == tile_row_bottom_y.
        // top source row index = tile_row_bottom_y - 5.
        let top_src = tile_row_bottom_y - (CDEF_LINE_ROWS_Y as i32 - 1);
        for r in 0..CDEF_LINE_ROWS_Y {
            let sy = top_src + r as i32;
            let dst_off = dst_base + r * y_ls;
            if dst_off + w > lr_cdef_line[0].len() {
                break;
            }
            if sy < 0 || sy >= frame_h {
                // Out-of-plane rows (top of frame): leave as zero / replicate
                // nearest in-plane row to avoid reading OOB. Replicate row 0.
                let rep = sy.clamp(0, frame_h - 1) as usize;
                let src_off = rep * y_ls;
                if src_off + w <= src[0].len() {
                    lr_cdef_line[0][dst_off..dst_off + w]
                        .copy_from_slice(&src[0][src_off..src_off + w]);
                }
                continue;
            }
            let src_off = sy as usize * y_ls;
            if src_off + w <= src[0].len() {
                lr_cdef_line[0][dst_off..dst_off + w]
                    .copy_from_slice(&src[0][src_off..src_off + w]);
            }
        }
    }

    if mono {
        return;
    }

    // Chroma. 2 rows ending at the subsampled bottom row.
    let cw = (frame_w >> ss_hor) as usize;
    let ch = (frame_h >> ss_ver) as i32;
    let bottom_uv = tile_row_bottom_y >> ss_ver;
    for plane in 1..3usize {
        let dst_base = tile_row * CDEF_LINE_ROWS_UV * uv_ls;
        let w = cw.min(uv_ls);
        let top_src = bottom_uv - (CDEF_LINE_ROWS_UV as i32 - 1);
        for r in 0..CDEF_LINE_ROWS_UV {
            let sy = top_src + r as i32;
            let dst_off = dst_base + r * uv_ls;
            if dst_off + w > lr_cdef_line[plane].len() {
                break;
            }
            let rep = sy.clamp(0, ch - 1) as usize;
            let src_off = rep * uv_ls;
            if src_off + w <= src[plane].len() {
                lr_cdef_line[plane][dst_off..dst_off + w]
                    .copy_from_slice(&src[plane][src_off..src_off + w]);
            }
        }
    }
}

/// Sample-stride version of `ensure_lr_cdef_line` for `Vec<u16>` planes.
pub(crate) fn ensure_lr_cdef_line_hbd(
    lr_cdef_line: &mut [Vec<u16>; 3],
    n_tile_rows: usize,
    y_stride_samples: isize,
    uv_stride_samples: isize,
    mono: bool,
) -> Result<(), ()> {
    let y_ls = y_stride_samples.unsigned_abs();
    let uv_ls = uv_stride_samples.unsigned_abs();
    let need_y = n_tile_rows * CDEF_LINE_ROWS_Y * y_ls;
    let need_uv = n_tile_rows * CDEF_LINE_ROWS_UV * uv_ls;

    fn ensure_plane(v: &mut Vec<u16>, len: usize) -> Result<(), ()> {
        if v.len() != len {
            if len > v.len() {
                v.try_reserve_exact(len - v.len()).map_err(|_| ())?;
            }
            v.resize(len, 0);
        }
        Ok(())
    }

    ensure_plane(&mut lr_cdef_line[0], need_y)?;
    if mono {
        lr_cdef_line[1].clear();
        lr_cdef_line[2].clear();
    } else {
        ensure_plane(&mut lr_cdef_line[1], need_uv)?;
        ensure_plane(&mut lr_cdef_line[2], need_uv)?;
    }
    Ok(())
}

/// Sample-stride version of `save_lr_cdef_line_8bpc` for `&[u16]` planes.
/// `strides` are in u16 samples, not bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn save_lr_cdef_line_hbd(
    lr_cdef_line: &mut [Vec<u16>; 3],
    src: &[&[u16]; 3],
    strides_samples: &[isize; 2],
    tile_row: usize,
    tile_row_bottom_y: i32,
    frame_w: i32,
    frame_h: i32,
    ss_hor: i32,
    ss_ver: i32,
    mono: bool,
) {
    let y_ls = strides_samples[0].unsigned_abs();
    let uv_ls = strides_samples[1].unsigned_abs();

    {
        let dst_base = tile_row * CDEF_LINE_ROWS_Y * y_ls;
        let w = (frame_w as usize).min(y_ls);
        let top_src = tile_row_bottom_y - (CDEF_LINE_ROWS_Y as i32 - 1);
        for r in 0..CDEF_LINE_ROWS_Y {
            let sy = top_src + r as i32;
            let dst_off = dst_base + r * y_ls;
            if dst_off + w > lr_cdef_line[0].len() {
                break;
            }
            let rep = sy.clamp(0, frame_h - 1) as usize;
            let src_off = rep * y_ls;
            if src_off + w <= src[0].len() {
                lr_cdef_line[0][dst_off..dst_off + w]
                    .copy_from_slice(&src[0][src_off..src_off + w]);
            }
        }
    }

    if mono {
        return;
    }

    let cw = (frame_w >> ss_hor) as usize;
    let ch = (frame_h >> ss_ver) as i32;
    let bottom_uv = tile_row_bottom_y >> ss_ver;
    for plane in 1..3usize {
        let dst_base = tile_row * CDEF_LINE_ROWS_UV * uv_ls;
        let w = cw.min(uv_ls);
        let top_src = bottom_uv - (CDEF_LINE_ROWS_UV as i32 - 1);
        for r in 0..CDEF_LINE_ROWS_UV {
            let sy = top_src + r as i32;
            let dst_off = dst_base + r * uv_ls;
            if dst_off + w > lr_cdef_line[plane].len() {
                break;
            }
            let rep = sy.clamp(0, ch - 1) as usize;
            let src_off = rep * uv_ls;
            if src_off + w <= src[plane].len() {
                lr_cdef_line[plane][dst_off..dst_off + w]
                    .copy_from_slice(&src[plane][src_off..src_off + w]);
            }
        }
    }
}
