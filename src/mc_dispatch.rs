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

#[inline]
pub(crate) fn compound_tmp_len(w: usize, h: usize) -> usize {
    (w * h).next_multiple_of(16)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_8tap_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    crate::mc::put_8tap_8bpc(
        dst,
        dst_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_8tap_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_off: usize,
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    filter_type: i32,
) {
    crate::mc::prep_8tap_8bpc(
        tmp,
        tmp_stride,
        src,
        src_off,
        src_stride,
        w,
        h,
        mx,
        my,
        filter_type,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn put_bilin_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    crate::mc::put_bilin_8bpc(dst, dst_stride, src, src_stride, w, h, mx, my);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prep_bilin_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
) {
    crate::mc::prep_bilin_8bpc(tmp, tmp_stride, src, src_stride, w, h, mx, my);
}

pub(crate) fn avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    crate::mc::avg_8bpc(dst, dst_stride, tmp1, tmp2, w, h);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_avg_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    crate::mc::w_avg_8bpc(dst, dst_stride, tmp1, tmp2, w, h, weight);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &[u8],
) {
    crate::mc::mask_8bpc(dst, dst_stride, tmp1, tmp2, w, h, m);
}

pub(crate) fn blend_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    m: &[u8],
) {
    crate::mc::blend_8bpc(dst, dst_stride, tmp, w, h, m);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_mask_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    m: &mut [u8],
    mask_stride: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    // 444 (no ss), 422 (h-only), 420 (h+v). (false, true) does not occur in
    // AV2 chroma layouts; callers must not route that case here.
    crate::mc::w_mask_8bpc(
        dst,
        dst_stride,
        tmp1,
        tmp2,
        w,
        h,
        m,
        mask_stride,
        sign,
        ss_hor,
        ss_ver,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    crate::mc::warp_affine_8x8_8bpc(dst, dst_stride, src, src_stride, src_off, abcd, mx, my);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_affine_8x8t_8bpc(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    src_off: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    crate::mc::warp_affine_8x8t_8bpc(tmp, tmp_stride, src, src_stride, src_off, abcd, mx, my);
}
