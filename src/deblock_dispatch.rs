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

use std::sync::OnceLock;

pub(crate) type DeblockApplyFn =
    unsafe fn(&mut [u8], isize, isize, isize, i32, i32, i32, bool, bool);

#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_apply_8bpc_scalar(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    let mut dp = off;
    for _ in 0..4 {
        let d0 = dst[dp as usize] as i32;
        let dm1 = dst[(dp - stride_tap) as usize] as i32;
        let dp1 = dst[(dp + stride_tap) as usize] as i32;
        let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
        let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let dn = delta_m2 * crate::deblock::W_MULT[(width_neg - 1) as usize] as i32;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * stride_tap) as usize;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, 255) as u8;
            }
        }

        if !pos_lossless {
            let dpv = delta_m2 * crate::deblock::W_MULT[(width_pos - 1) as usize] as i32;
            for j in 0..width_pos {
                let idx = (dp + (j as isize) * stride_tap) as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, 255) as u8;
            }
        }

        dp += stride_line;
    }
}

static DEBLOCK_APPLY: OnceLock<DeblockApplyFn> = OnceLock::new();

#[inline]
fn resolve_deblock_apply() -> DeblockApplyFn {
    *DEBLOCK_APPLY.get_or_init(|| {
        let mut f = deblock_apply_8bpc_scalar as DeblockApplyFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::deblock_apply_8bpc_neon as DeblockApplyFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::deblock_apply_8bpc_sse41 as DeblockApplyFn;
            }
        }
        f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn deblock_apply_8bpc(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    // SAFETY: `resolve_deblock_apply` only returns the SSE/NEON kernel when the
    // corresponding feature was detected; the scalar default is always sound.
    unsafe {
        resolve_deblock_apply()(
            dst,
            off,
            stride_line,
            stride_tap,
            width_neg,
            width_pos,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        )
    };
}
