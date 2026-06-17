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

use crate::intops::ulog2;
use crate::tables::DIV_RECIP;

#[inline]
fn fast_div32(num: u32, den: u32) -> u8 {
    let shift = ulog2(den) as u32;
    let rem = den - (1 << shift);
    let idx = ((rem << 7) + (1 << (shift - 1))) >> shift;
    debug_assert!(idx <= 128);
    let shift = shift + 2;
    let res = ((num as u64 * DIV_RECIP[idx as usize] as u64) + ((1u64 << shift) >> 1)) >> shift;
    debug_assert!(res < 256);
    res as u8
}

pub(crate) fn init_ibp_weights() -> [[[u8; 16]; 16]; 7] {
    static DR_DY_Q6: [u32; 7] = [682, 256, 170, 128, 81, 64, 50];
    let mut weights = [[[0u8; 16]; 16]; 7];
    for m in 0..7 {
        let dy = DR_DY_Q6[m];
        for y in 0..16 {
            let yy = ((y + 1) as u32) << 6;
            let mut y_pos = dy;
            for x in 0..16 {
                weights[m][y][x] = fast_div32(y_pos, yy + y_pos);
                y_pos += dy;
            }
        }
    }
    weights
}
