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

use crate::intops::{apply_sign, iclip};
use crate::pixel::Coeff;

pub(crate) fn stxfm<C: Coeff>(
    cf_out: &mut [i32],
    cf: &[C],
    kernel: &[i8],
    sz: usize,
    eob: usize,
    bitdepth_max: i32,
) {
    debug_assert!(sz == 16 || sz == 48);
    debug_assert!(eob < if sz == 16 { 8 } else { 32 });
    let min = -128 * (1 + bitdepth_max);
    let max = 128 * (1 + bitdepth_max) - 1;
    let h = eob + 1;
    for (x, cf_out) in cf_out[..sz].iter_mut().enumerate() {
        let mut sum = 0i32;
        for (y, &cf) in cf[..h].iter().enumerate() {
            sum += cf.to_i32() * kernel[y * sz + x] as i32;
        }
        sum = apply_sign((sum.abs() + 64) >> 7, sum);
        *cf_out = iclip(sum, min, max);
    }
}
