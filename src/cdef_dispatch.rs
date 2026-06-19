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

use crate::cdef::constrain;
use crate::tables::CDEF_DIRECTIONS;
use std::sync::OnceLock;

pub(crate) type CdefFilterFn = unsafe fn(
    &mut [u8],
    usize,
    usize,
    &[i16],
    usize,
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    usize,
    usize,
    usize,
);

#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block_8bpc_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    let mut dp = dst_off;
    let mut tp = o;
    if pri_strength != 0 && sec_strength != 0 {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                let mut min_v = px;
                let mut max_v = px;
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let p0 = tmp[((tp + x) as isize + off1) as usize] as i32;
                    let p1 = tmp[((tp + x) as isize - off1) as usize] as i32;
                    sum += ptap * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptap * constrain(p1 - px, pri_strength, pri_shift);
                    ptap = (ptap & 3) | 2;
                    min_v = p0.min(min_v).min(p1);
                    max_v = p0.max(max_v).max(p1);
                    let off2 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off3 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp[((tp + x) as isize + off2) as usize] as i32;
                    let s1 = tmp[((tp + x) as isize - off2) as usize] as i32;
                    let s2 = tmp[((tp + x) as isize + off3) as usize] as i32;
                    let s3 = tmp[((tp + x) as isize - off3) as usize] as i32;
                    let st = 2 - k as i32;
                    sum += st * constrain(s0 - px, sec_strength, sec_shift);
                    sum += st * constrain(s1 - px, sec_strength, sec_shift);
                    sum += st * constrain(s2 - px, sec_strength, sec_shift);
                    sum += st * constrain(s3 - px, sec_strength, sec_shift);
                    min_v = s0.min(min_v).min(s1).min(s2).min(s3);
                    max_v = s0.max(max_v).max(s1).max(s2).max(s3);
                }
                let v = px + ((sum - (sum < 0) as i32 + 8) >> 4);
                dst[dp + x] = v.clamp(min_v, max_v) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    } else if pri_strength != 0 {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                let mut ptap = pri_tap;
                for k in 0..2 {
                    let off = CDEF_DIRECTIONS[dir + 2][k] as isize;
                    let p0 = tmp[((tp + x) as isize + off) as usize] as i32;
                    let p1 = tmp[((tp + x) as isize - off) as usize] as i32;
                    sum += ptap * constrain(p0 - px, pri_strength, pri_shift);
                    sum += ptap * constrain(p1 - px, pri_strength, pri_shift);
                    ptap = (ptap & 3) | 2;
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    } else {
        for _y in 0..h {
            for x in 0..w {
                let px = tmp[tp + x] as i32;
                let mut sum = 0i32;
                for k in 0..2 {
                    let off1 = CDEF_DIRECTIONS[dir + 4][k] as isize;
                    let off2 = CDEF_DIRECTIONS[dir][k] as isize;
                    let s0 = tmp[((tp + x) as isize + off1) as usize] as i32;
                    let s1 = tmp[((tp + x) as isize - off1) as usize] as i32;
                    let s2 = tmp[((tp + x) as isize + off2) as usize] as i32;
                    let s3 = tmp[((tp + x) as isize - off2) as usize] as i32;
                    let st = 2 - k as i32;
                    sum += st * constrain(s0 - px, sec_strength, sec_shift);
                    sum += st * constrain(s1 - px, sec_strength, sec_shift);
                    sum += st * constrain(s2 - px, sec_strength, sec_shift);
                    sum += st * constrain(s3 - px, sec_strength, sec_shift);
                }
                dst[dp + x] = (px + ((sum - (sum < 0) as i32 + 8) >> 4)) as u8;
            }
            dp += dst_stride;
            tp += tmp_stride;
        }
    }
}

static CDEF_FILTER: OnceLock<CdefFilterFn> = OnceLock::new();

#[inline]
fn resolve_cdef_filter() -> CdefFilterFn {
    *CDEF_FILTER.get_or_init(|| {
        let mut f = cdef_filter_block_8bpc_scalar as CdefFilterFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::cdef_filter_block_8bpc_neon as CdefFilterFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::cdef_filter_block_8bpc_sse41 as CdefFilterFn;
            }
        }
        f
    })
}

/// Dispatched 8-bit CDEF filter apply. See `CdefFilterFn` for the argument layout.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cdef_filter_block_8bpc(
    dst: &mut [u8],
    dst_stride: usize,
    dst_off: usize,
    tmp: &[i16],
    tmp_stride: usize,
    o: usize,
    pri_strength: i32,
    sec_strength: i32,
    pri_shift: i32,
    sec_shift: i32,
    pri_tap: i32,
    dir: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: resolve only returns the SSE/NEON kernel when the feature was
    // detected; the scalar default is always sound.
    unsafe {
        resolve_cdef_filter()(
            dst,
            dst_stride,
            dst_off,
            tmp,
            tmp_stride,
            o,
            pri_strength,
            sec_strength,
            pri_shift,
            sec_shift,
            pri_tap,
            dir,
            w,
            h,
        )
    };
}
