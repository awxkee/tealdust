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

pub(crate) fn pal_idx_finish(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    debug_assert!((4..=64).contains(&bw) && (bw & (bw - 1)) == 0);
    debug_assert!((4..=64).contains(&bh) && (bh & (bh - 1)) == 0);
    debug_assert!(w >= 4 && w <= bw && (w & 3) == 0);
    debug_assert!(h >= 4 && h <= bh && (h & 3) == 0);

    let dst_w = w / 2;
    let dst_bw = bw / 2;

    for y in 0..h {
        let src_row = &src[y * bw..];
        let dst_row = &mut dst[y * dst_bw..];
        for (x, dst) in dst_row[..dst_w].iter_mut().enumerate() {
            *dst = src_row[x * 2] | (src_row[x * 2 + 1] << 4);
        }
        if dst_w < dst_bw {
            let fill = src_row[w - 1] * 0x11;
            dst_row[dst_w..dst_bw].fill(fill);
        }
    }

    if h < bh {
        let last_row_start = (h - 1) * dst_bw;
        for y in h..bh {
            let row_start = y * dst_bw;
            for x in 0..dst_bw {
                dst[row_start + x] = dst[last_row_start + x];
            }
        }
    }
}
