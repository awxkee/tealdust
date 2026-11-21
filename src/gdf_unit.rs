/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! Pieces of the unit-level GDF error preparation shared by the SIMD kernels.
//!
//! The 13 source rows around the current one are kept as 10-bit normalized
//! u16 rows (8-bit samples `<< 2`, 12-bit samples `>> 2`), so one kernel
//! serves every bit depth and the taps are plain 16-bit loads.

use crate::looprestoration::{LR_HAVE_LEFT, LR_HAVE_RIGHT};

/// Row buffers hold 6 edge pixels on each side of up to 64 pixels, plus slack
/// so whole SIMD blocks can be read past a narrower unit.
pub(crate) const ROW_STRIDE: usize = 96;
pub(crate) const ROW_ORIGIN: usize = 8;
/// Gradient cells written in whole SIMD blocks (33 are meaningful at most).
pub(crate) const GRAD_CELLS: usize = 40;

/// `[dy, dx]` slots of `GDF_PREP_COORDS`, named by their class-0 role.
pub(crate) static TAP_V: [usize; 6] = [9, 5, 3, 2, 1, 0]; // [d, 0], d = 1..=6
pub(crate) static TAP_H: [usize; 6] = [17, 16, 15, 14, 13, 12]; // [0, d], d = 1..=6
pub(crate) static TAP_D: [usize; 2] = [8, 10]; // [1, 1], [1, -1]
pub(crate) static TAP_K: [usize; 4] = [4, 6, 7, 11]; // [2, 1], [2, -1], [1, 2], [1, -2]

pub(crate) trait GdfPixel: Copy {
    /// The sample at 10-bit precision.
    fn norm(self, down_shift: u32) -> u16;
}

impl GdfPixel for u8 {
    #[inline(always)]
    fn norm(self, _down_shift: u32) -> u16 {
        (self as u16) << 2
    }
}

impl GdfPixel for u16 {
    #[inline(always)]
    fn norm(self, down_shift: u32) -> u16 {
        self >> down_shift
    }
}

/// Copies `w` pixels at `src[src_o..]` with 6 edge pixels on each side. The
/// left edge comes from `left` when given, else from the row itself.
#[inline(always)]
pub(crate) fn backup_row<P: GdfPixel>(
    dst: &mut [u16; ROW_STRIDE],
    src: &[P],
    src_o: usize,
    left: Option<&[P; 6]>,
    w: usize,
    edges: u8,
    down_shift: u32,
) {
    let o = ROW_ORIGIN;
    let n = |v: &P| v.norm(down_shift);
    if edges & LR_HAVE_LEFT != 0 {
        let left = match left {
            Some(left) => &left[..],
            None => &src[src_o - 6..src_o],
        };
        for (d, s) in dst[o - 6..o].iter_mut().zip(left) {
            *d = n(s);
        }
    } else {
        dst[o - 6..o].fill(n(&src[src_o]));
    }
    for (d, s) in dst[o..o + w].iter_mut().zip(&src[src_o..src_o + w]) {
        *d = n(s);
    }
    if edges & LR_HAVE_RIGHT != 0 {
        for (d, s) in dst[o + w..o + w + 6]
            .iter_mut()
            .zip(&src[src_o + w..src_o + w + 6])
        {
            *d = n(s);
        }
    } else {
        dst[o + w..o + w + 6].fill(n(&src[src_o + w - 1]));
    }
}
