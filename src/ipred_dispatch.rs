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

#[allow(clippy::too_many_arguments)]
pub(crate) fn ipred_v(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    crate::ipred::ipred_v_8bpc(dst, stride, tl, o, width, height, angle);
}

/// HOR_PRED: replicate the left column across each row.
/// Falls back to scalar when the AV2 MRL-averaging flag is set.
#[allow(clippy::too_many_arguments)]
pub fn ipred_h(
    dst: &mut [u8],
    stride: usize,
    tl: &[u8],
    o: usize,
    width: usize,
    height: usize,
    angle: i32,
) {
    crate::ipred::ipred_h_8bpc(dst, stride, tl, o, width, height, angle);
}

/// SMOOTH_V_PRED: vertical smooth interpolation.
pub fn ipred_smooth_v(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    crate::ipred::ipred_smooth_v_8bpc(dst, stride, tl, o, w, h);
}

/// SMOOTH_H_PRED: horizontal smooth interpolation.
pub fn ipred_smooth_h(dst: &mut [u8], stride: usize, tl: &[u8], o: usize, w: usize, h: usize) {
    crate::ipred::ipred_smooth_h_8bpc(dst, stride, tl, o, w, h);
}
