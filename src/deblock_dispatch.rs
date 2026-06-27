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

pub(crate) type DeblockApply8bpcFn =
    unsafe fn(&mut [u8], isize, isize, isize, i32, i32, i32, bool, bool);
pub(crate) type DeblockApplyHbdFn =
    unsafe fn(&mut [u16], isize, isize, isize, i32, i32, i32, bool, bool, i32);
pub(crate) type DeblockSb64Fn =
    unsafe fn(&mut [u8], usize, usize, &[u16], &[u16], &[u8], &[u8], bool);

pub(crate) type DeblockSetupColsSeg8bpcFn = unsafe fn(
    &mut [u8; 256],
    &mut [u8; 256],
    &[u8],
    isize,
    isize,
    &[[[u16; 4]; 5]; 64],
    usize,
    &[[u32; 16]; 2],
    &mut [u8; 16],
    &mut [u8; 16],
    i32,
    i32,
    i32,
    i32,
);
pub(crate) type DeblockSetupRowsSeg8bpcFn = unsafe fn(
    &mut [u8; 256],
    &mut [u8; 256],
    &[u8],
    isize,
    isize,
    &[[[u16; 4]; 5]; 64],
    usize,
    &[[u32; 16]; 2],
    Option<&[[u32; 16]; 2]>,
    Option<(&[u8], isize)>,
    i32,
    i32,
    i32,
    i32,
);
pub(crate) type DeblockSetupColsDq8bpcFn = unsafe fn(
    &mut [u8; 256],
    &mut [u8; 256],
    &[[[u16; 4]; 5]; 64],
    usize,
    &[[u32; 16]; 2],
    &mut [u8; 16],
    &mut [u8; 16],
    i32,
    i32,
    i32,
    i32,
);
pub(crate) type DeblockSetupRowsDq8bpcFn = unsafe fn(
    &mut [u8; 256],
    &mut [u8; 256],
    &[[[u16; 4]; 5]; 64],
    usize,
    &[[u32; 16]; 2],
    Option<&[[u32; 16]; 2]>,
    Option<(&[u8], isize)>,
    i32,
    i32,
    i32,
    i32,
);
pub(crate) type DeblockSetupSimple8bpcFn = unsafe fn(
    &mut [u8; 256],
    &mut [u8; 256],
    &[[[u16; 4]; 5]; 64],
    usize,
    &[[u32; 16]; 2],
    i32,
    i32,
    i32,
    i32,
);

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_apply_8bpc_scalar_h_const<const WN: i32, const WP: i32>(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    debug_assert!((1..=8).contains(&WN));
    debug_assert!((1..=8).contains(&WP));

    if neg_lossless && pos_lossless {
        return;
    }

    let wmul_neg = crate::deblock::W_MULT[(WN - 1) as usize] as i32;
    let wmul_pos = crate::deblock::W_MULT[(WP - 1) as usize] as i32;
    let mut dp = off;

    for _ in 0..4 {
        let p = dp as usize;
        let d0 = dst[p] as i32;
        let dm1 = dst[p - 1] as i32;
        let dp1 = dst[p + 1] as i32;
        let dm2 = dst[p - 2] as i32;
        let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let dn = delta_m2 * wmul_neg;
            for j in 0..WN {
                let idx = p - j as usize - 1;
                let diff = (dn * (WN - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, 255) as u8;
            }
        }

        if !pos_lossless {
            let dpv = delta_m2 * wmul_pos;
            for j in 0..WP {
                let idx = p + j as usize;
                let diff = (dpv * (WP - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, 255) as u8;
            }
        }

        dp += stride_line;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_apply_8bpc_scalar_h_specialized(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) -> bool {
    match (width_neg, width_pos) {
        (1, 1) => deblock_apply_8bpc_scalar_h_const::<1, 1>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 2) => deblock_apply_8bpc_scalar_h_const::<2, 2>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 3) => deblock_apply_8bpc_scalar_h_const::<2, 3>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (2, 4) => deblock_apply_8bpc_scalar_h_const::<2, 4>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (3, 3) => deblock_apply_8bpc_scalar_h_const::<3, 3>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (4, 4) => deblock_apply_8bpc_scalar_h_const::<4, 4>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (6, 6) => deblock_apply_8bpc_scalar_h_const::<6, 6>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (6, 8) => deblock_apply_8bpc_scalar_h_const::<6, 8>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        (8, 8) => deblock_apply_8bpc_scalar_h_const::<8, 8>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        ),
        _ => return false,
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_apply_hbd_scalar_h_const<const WN: i32, const WP: i32>(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    debug_assert!((1..=8).contains(&WN));
    debug_assert!((1..=8).contains(&WP));

    if neg_lossless && pos_lossless {
        return;
    }

    let wmul_neg = crate::deblock::W_MULT[(WN - 1) as usize] as i32;
    let wmul_pos = crate::deblock::W_MULT[(WP - 1) as usize] as i32;
    let mut dp = off;

    for _ in 0..4 {
        let p = dp as usize;
        let d0 = dst[p] as i32;
        let dm1 = dst[p - 1] as i32;
        let dp1 = dst[p + 1] as i32;
        let dm2 = dst[p - 2] as i32;
        let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let dn = delta_m2 * wmul_neg;
            for j in 0..WN {
                let idx = p - j as usize - 1;
                let diff = (dn * (WN - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, bitdepth_max) as u16;
            }
        }

        if !pos_lossless {
            let dpv = delta_m2 * wmul_pos;
            for j in 0..WP {
                let idx = p + j as usize;
                let diff = (dpv * (WP - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, bitdepth_max) as u16;
            }
        }

        dp += stride_line;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_apply_hbd_scalar_h_specialized(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) -> bool {
    match (width_neg, width_pos) {
        (1, 1) => deblock_apply_hbd_scalar_h_const::<1, 1>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 2) => deblock_apply_hbd_scalar_h_const::<2, 2>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 3) => deblock_apply_hbd_scalar_h_const::<2, 3>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (2, 4) => deblock_apply_hbd_scalar_h_const::<2, 4>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (3, 3) => deblock_apply_hbd_scalar_h_const::<3, 3>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (4, 4) => deblock_apply_hbd_scalar_h_const::<4, 4>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (6, 6) => deblock_apply_hbd_scalar_h_const::<6, 6>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (6, 8) => deblock_apply_hbd_scalar_h_const::<6, 8>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        (8, 8) => deblock_apply_hbd_scalar_h_const::<8, 8>(
            dst,
            off,
            stride_line,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        ),
        _ => return false,
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_apply_8bpc_scalar_h(
    dst: &mut [u8],
    off: isize,
    stride_line: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
) {
    if deblock_apply_8bpc_scalar_h_specialized(
        dst,
        off,
        stride_line,
        width_neg,
        width_pos,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
    ) {
        return;
    }

    if neg_lossless && pos_lossless {
        return;
    }

    let wmul_neg = crate::deblock::W_MULT[(width_neg - 1) as usize] as i32;
    let wmul_pos = crate::deblock::W_MULT[(width_pos - 1) as usize] as i32;
    let mut dp = off;

    for _ in 0..4 {
        let p = dp as usize;
        let d0 = dst[p] as i32;
        let dm1 = dst[p - 1] as i32;
        let dp1 = dst[p + 1] as i32;
        let dm2 = dst[p - 2] as i32;
        let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let dn = delta_m2 * wmul_neg;
            for j in 0..width_neg {
                let idx = p - j as usize - 1;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, 255) as u8;
            }
        }

        if !pos_lossless {
            let dpv = delta_m2 * wmul_pos;
            for j in 0..width_pos {
                let idx = p + j as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, 255) as u8;
            }
        }

        dp += stride_line;
    }
}

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
    if stride_tap == 1 {
        deblock_apply_8bpc_scalar_h(
            dst,
            off,
            stride_line,
            width_neg,
            width_pos,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
        );
        return;
    }

    if neg_lossless && pos_lossless {
        return;
    }

    let wmul_neg = crate::deblock::W_MULT[(width_neg - 1) as usize] as i32;
    let wmul_pos = crate::deblock::W_MULT[(width_pos - 1) as usize] as i32;
    let mut dp = off;

    if !neg_lossless && !pos_lossless {
        for _ in 0..4 {
            let d0 = dst[dp as usize] as i32;
            let dm1 = dst[(dp - stride_tap) as usize] as i32;
            let dp1 = dst[(dp + stride_tap) as usize] as i32;
            let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
            let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

            let dn = delta_m2 * wmul_neg;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * stride_tap) as usize;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, 255) as u8;
            }

            let dpv = delta_m2 * wmul_pos;
            for j in 0..width_pos {
                let idx = (dp + (j as isize) * stride_tap) as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, 255) as u8;
            }

            dp += stride_line;
        }
    } else if !neg_lossless {
        for _ in 0..4 {
            let d0 = dst[dp as usize] as i32;
            let dm1 = dst[(dp - stride_tap) as usize] as i32;
            let dp1 = dst[(dp + stride_tap) as usize] as i32;
            let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
            let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

            let dn = delta_m2 * wmul_neg;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * stride_tap) as usize;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, 255) as u8;
            }

            dp += stride_line;
        }
    } else {
        for _ in 0..4 {
            let d0 = dst[dp as usize] as i32;
            let dm1 = dst[(dp - stride_tap) as usize] as i32;
            let dp1 = dst[(dp + stride_tap) as usize] as i32;
            let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
            let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

            let dpv = delta_m2 * wmul_pos;
            for j in 0..width_pos {
                let idx = (dp + (j as isize) * stride_tap) as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, 255) as u8;
            }

            dp += stride_line;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_apply_hbd_scalar_h(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    if deblock_apply_hbd_scalar_h_specialized(
        dst,
        off,
        stride_line,
        width_neg,
        width_pos,
        q_thr_clamp,
        neg_lossless,
        pos_lossless,
        bitdepth_max,
    ) {
        return;
    }

    if neg_lossless && pos_lossless {
        return;
    }

    let wmul_neg = crate::deblock::W_MULT[(width_neg - 1) as usize] as i32;
    let wmul_pos = crate::deblock::W_MULT[(width_pos - 1) as usize] as i32;
    let mut dp = off;

    for _ in 0..4 {
        let p = dp as usize;
        let d0 = dst[p] as i32;
        let dm1 = dst[p - 1] as i32;
        let dp1 = dst[p + 1] as i32;
        let dm2 = dst[p - 2] as i32;
        let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

        if !neg_lossless {
            let dn = delta_m2 * wmul_neg;
            for j in 0..width_neg {
                let idx = p - j as usize - 1;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, bitdepth_max) as u16;
            }
        }

        if !pos_lossless {
            let dpv = delta_m2 * wmul_pos;
            for j in 0..width_pos {
                let idx = p + j as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, bitdepth_max) as u16;
            }
        }

        dp += stride_line;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_apply_hbd_scalar(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    if stride_tap == 1 {
        deblock_apply_hbd_scalar_h(
            dst,
            off,
            stride_line,
            width_neg,
            width_pos,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        );
        return;
    }

    if neg_lossless && pos_lossless {
        return;
    }

    let wmul_neg = crate::deblock::W_MULT[(width_neg - 1) as usize] as i32;
    let wmul_pos = crate::deblock::W_MULT[(width_pos - 1) as usize] as i32;
    let mut dp = off;

    if !neg_lossless && !pos_lossless {
        for _ in 0..4 {
            let d0 = dst[dp as usize] as i32;
            let dm1 = dst[(dp - stride_tap) as usize] as i32;
            let dp1 = dst[(dp + stride_tap) as usize] as i32;
            let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
            let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

            let dn = delta_m2 * wmul_neg;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * stride_tap) as usize;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, bitdepth_max) as u16;
            }

            let dpv = delta_m2 * wmul_pos;
            for j in 0..width_pos {
                let idx = (dp + (j as isize) * stride_tap) as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, bitdepth_max) as u16;
            }

            dp += stride_line;
        }
    } else if !neg_lossless {
        for _ in 0..4 {
            let d0 = dst[dp as usize] as i32;
            let dm1 = dst[(dp - stride_tap) as usize] as i32;
            let dp1 = dst[(dp + stride_tap) as usize] as i32;
            let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
            let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

            let dn = delta_m2 * wmul_neg;
            for j in 0..width_neg {
                let idx = (dp + (-(j as isize) - 1) * stride_tap) as usize;
                let diff = (dn * (width_neg - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 + diff).clamp(0, bitdepth_max) as u16;
            }

            dp += stride_line;
        }
    } else {
        for _ in 0..4 {
            let d0 = dst[dp as usize] as i32;
            let dm1 = dst[(dp - stride_tap) as usize] as i32;
            let dp1 = dst[(dp + stride_tap) as usize] as i32;
            let dm2 = dst[(dp - 2 * stride_tap) as usize] as i32;
            let delta_m2 = (4 * (3 * (d0 - dm1) - (dp1 - dm2))).clamp(-q_thr_clamp, q_thr_clamp);

            let dpv = delta_m2 * wmul_pos;
            for j in 0..width_pos {
                let idx = (dp + (j as isize) * stride_tap) as usize;
                let diff = (dpv * (width_pos - j) + (1 << 10)) >> 11;
                dst[idx] = (dst[idx] as i32 - diff).clamp(0, bitdepth_max) as u16;
            }

            dp += stride_line;
        }
    }
}

static DEBLOCK_APPLY_8BPC: OnceLock<DeblockApply8bpcFn> = OnceLock::new();
static DEBLOCK_APPLY_HBD: OnceLock<DeblockApplyHbdFn> = OnceLock::new();
static DEBLOCK_H_SB64Y_8BPC: OnceLock<Option<DeblockSb64Fn>> = OnceLock::new();
static DEBLOCK_V_SB64Y_8BPC: OnceLock<Option<DeblockSb64Fn>> = OnceLock::new();
static DEBLOCK_H_SB64UV_8BPC: OnceLock<Option<DeblockSb64Fn>> = OnceLock::new();
static DEBLOCK_V_SB64UV_8BPC: OnceLock<Option<DeblockSb64Fn>> = OnceLock::new();

static SETUP_THR_COLS_SEG_8BPC: OnceLock<Option<DeblockSetupColsSeg8bpcFn>> = OnceLock::new();
static SETUP_THR_ROWS_SEG_8BPC: OnceLock<Option<DeblockSetupRowsSeg8bpcFn>> = OnceLock::new();
static SETUP_THR_COLS_DQ_8BPC: OnceLock<Option<DeblockSetupColsDq8bpcFn>> = OnceLock::new();
static SETUP_THR_ROWS_DQ_8BPC: OnceLock<Option<DeblockSetupRowsDq8bpcFn>> = OnceLock::new();
static SETUP_THR_COLS_SIMPLE_8BPC: OnceLock<Option<DeblockSetupSimple8bpcFn>> = OnceLock::new();
static SETUP_THR_ROWS_SIMPLE_8BPC: OnceLock<Option<DeblockSetupSimple8bpcFn>> = OnceLock::new();

#[inline]
fn resolve_deblock_apply_8bpc() -> DeblockApply8bpcFn {
    *DEBLOCK_APPLY_8BPC.get_or_init(|| {
        let mut _f = deblock_apply_8bpc_scalar as DeblockApply8bpcFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                _f = crate::neon::deblock_apply_8bpc_neon as DeblockApply8bpcFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::deblock_apply_8bpc_sse41 as DeblockApply8bpcFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::deblock_apply_8bpc_avx2 as DeblockApply8bpcFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_deblock_apply_hbd() -> DeblockApplyHbdFn {
    *DEBLOCK_APPLY_HBD.get_or_init(|| {
        let mut _f = deblock_apply_hbd_scalar as DeblockApplyHbdFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                _f = crate::neon::deblock_apply_hbd_neon as DeblockApplyHbdFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                _f = crate::sse::deblock_apply_hbd_sse41 as DeblockApplyHbdFn;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::avx::deblock_apply_hbd_avx2 as DeblockApplyHbdFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_deblock_h_sb64y_8bpc() -> Option<DeblockSb64Fn> {
    *DEBLOCK_H_SB64Y_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSb64Fn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                _f = Some(crate::neon::deblock_h_sb64y_8bpc_neon as DeblockSb64Fn);
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::deblock_h_sb64y_8bpc_avx2 as DeblockSb64Fn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_deblock_v_sb64y_8bpc() -> Option<DeblockSb64Fn> {
    *DEBLOCK_V_SB64Y_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSb64Fn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                _f = Some(crate::neon::deblock_v_sb64y_8bpc_neon as DeblockSb64Fn);
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::deblock_v_sb64y_8bpc_avx2 as DeblockSb64Fn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_deblock_h_sb64uv_8bpc() -> Option<DeblockSb64Fn> {
    *DEBLOCK_H_SB64UV_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSb64Fn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                _f = Some(crate::neon::deblock_h_sb64uv_8bpc_neon as DeblockSb64Fn);
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::deblock_h_sb64uv_8bpc_avx2 as DeblockSb64Fn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_deblock_v_sb64uv_8bpc() -> Option<DeblockSb64Fn> {
    *DEBLOCK_V_SB64UV_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSb64Fn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                _f = Some(crate::neon::deblock_v_sb64uv_8bpc_neon as DeblockSb64Fn);
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::deblock_v_sb64uv_8bpc_avx2 as DeblockSb64Fn);
            }
        }
        _f
    })
}

macro_rules! define_try_deblock_sb64_8bpc {
    ($name:ident, $resolver:ident) => {
        #[allow(clippy::too_many_arguments)]
        #[inline]
        pub(crate) fn $name(
            dst: &mut [u8],
            dst_off: usize,
            stride: usize,
            vmask: &[u16],
            ll_mask: &[u16],
            q_thr: &[u8],
            side_thr: &[u8],
            edge: bool,
        ) -> bool {
            let Some(f) = $resolver() else {
                return false;
            };
            // SAFETY: the resolver returns target-feature entry points only after
            // the corresponding runtime feature probe succeeds. Otherwise the
            // caller continues into the existing scalar/generic path.
            unsafe { f(dst, dst_off, stride, vmask, ll_mask, q_thr, side_thr, edge) };
            true
        }
    };
}

define_try_deblock_sb64_8bpc!(try_deblock_h_sb64y_8bpc, resolve_deblock_h_sb64y_8bpc);
define_try_deblock_sb64_8bpc!(try_deblock_v_sb64y_8bpc, resolve_deblock_v_sb64y_8bpc);
define_try_deblock_sb64_8bpc!(try_deblock_h_sb64uv_8bpc, resolve_deblock_h_sb64uv_8bpc);
define_try_deblock_sb64_8bpc!(try_deblock_v_sb64uv_8bpc, resolve_deblock_v_sb64uv_8bpc);

#[inline]
fn resolve_setup_thr_cols_seg_8bpc() -> Option<DeblockSetupColsSeg8bpcFn> {
    *SETUP_THR_COLS_SEG_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSetupColsSeg8bpcFn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::setup_thr_cols_seg_8bpc_neon as DeblockSetupColsSeg8bpcFn);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::setup_thr_cols_seg_8bpc_avx2 as DeblockSetupColsSeg8bpcFn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_setup_thr_rows_seg_8bpc() -> Option<DeblockSetupRowsSeg8bpcFn> {
    *SETUP_THR_ROWS_SEG_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSetupRowsSeg8bpcFn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::setup_thr_rows_seg_8bpc_neon as DeblockSetupRowsSeg8bpcFn);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::setup_thr_rows_seg_8bpc_avx2 as DeblockSetupRowsSeg8bpcFn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_setup_thr_cols_dq_8bpc() -> Option<DeblockSetupColsDq8bpcFn> {
    *SETUP_THR_COLS_DQ_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSetupColsDq8bpcFn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::setup_thr_cols_dq_8bpc_neon as DeblockSetupColsDq8bpcFn);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::setup_thr_cols_dq_8bpc_avx2 as DeblockSetupColsDq8bpcFn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_setup_thr_rows_dq_8bpc() -> Option<DeblockSetupRowsDq8bpcFn> {
    *SETUP_THR_ROWS_DQ_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSetupRowsDq8bpcFn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::setup_thr_rows_dq_8bpc_neon as DeblockSetupRowsDq8bpcFn);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::setup_thr_rows_dq_8bpc_avx2 as DeblockSetupRowsDq8bpcFn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_setup_thr_cols_simple_8bpc() -> Option<DeblockSetupSimple8bpcFn> {
    *SETUP_THR_COLS_SIMPLE_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSetupSimple8bpcFn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::setup_thr_cols_simple_8bpc_neon as DeblockSetupSimple8bpcFn);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::setup_thr_cols_simple_8bpc_avx2 as DeblockSetupSimple8bpcFn);
            }
        }
        _f
    })
}

#[inline]
fn resolve_setup_thr_rows_simple_8bpc() -> Option<DeblockSetupSimple8bpcFn> {
    *SETUP_THR_ROWS_SIMPLE_8BPC.get_or_init(|| {
        let mut _f: Option<DeblockSetupSimple8bpcFn> = None;
        #[cfg(target_arch = "aarch64")]
        {
            _f = Some(crate::neon::setup_thr_rows_simple_8bpc_neon as DeblockSetupSimple8bpcFn);
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = Some(crate::avx::setup_thr_rows_simple_8bpc_avx2 as DeblockSetupSimple8bpcFn);
            }
        }
        _f
    })
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn try_setup_thr_cols_seg_8bpc(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    segmap: &[u8],
    seg_off: isize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) -> bool {
    let Some(f) = resolve_setup_thr_cols_seg_8bpc() else {
        return false;
    };
    unsafe {
        f(
            q_thr_dst,
            side_thr_dst,
            segmap,
            seg_off,
            seg_stride,
            mask,
            bx4_base,
            thr_lut,
            left_q_thr,
            left_side_thr,
            y64,
            ss_ver,
            w4,
            h4,
        )
    };
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn try_setup_thr_rows_seg_8bpc(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    segmap: &[u8],
    seg_off: isize,
    seg_stride: isize,
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    above_thr_lut: Option<&[[u32; 16]; 2]>,
    above_seg: Option<(&[u8], isize)>,
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) -> bool {
    let Some(f) = resolve_setup_thr_rows_seg_8bpc() else {
        return false;
    };
    unsafe {
        f(
            q_thr_dst,
            side_thr_dst,
            segmap,
            seg_off,
            seg_stride,
            mask,
            starty4,
            thr_lut,
            above_thr_lut,
            above_seg,
            sb64x,
            ss_hor,
            w4,
            h4,
        )
    };
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn try_setup_thr_cols_dq_8bpc(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    left_q_thr: &mut [u8; 16],
    left_side_thr: &mut [u8; 16],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) -> bool {
    let Some(f) = resolve_setup_thr_cols_dq_8bpc() else {
        return false;
    };
    unsafe {
        f(
            q_thr_dst,
            side_thr_dst,
            mask,
            bx4_base,
            thr_lut,
            left_q_thr,
            left_side_thr,
            y64,
            ss_ver,
            w4,
            h4,
        )
    };
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn try_setup_thr_rows_dq_8bpc(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    above_thr_lut: Option<&[[u32; 16]; 2]>,
    above_seg: Option<(&[u8], isize)>,
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) -> bool {
    let Some(f) = resolve_setup_thr_rows_dq_8bpc() else {
        return false;
    };
    unsafe {
        f(
            q_thr_dst,
            side_thr_dst,
            mask,
            starty4,
            thr_lut,
            above_thr_lut,
            above_seg,
            sb64x,
            ss_hor,
            w4,
            h4,
        )
    };
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn try_setup_thr_cols_simple_8bpc(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    bx4_base: usize,
    thr_lut: &[[u32; 16]; 2],
    y64: i32,
    ss_ver: i32,
    w4: i32,
    h4: i32,
) -> bool {
    let Some(f) = resolve_setup_thr_cols_simple_8bpc() else {
        return false;
    };
    unsafe {
        f(
            q_thr_dst,
            side_thr_dst,
            mask,
            bx4_base,
            thr_lut,
            y64,
            ss_ver,
            w4,
            h4,
        )
    };
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn try_setup_thr_rows_simple_8bpc(
    q_thr_dst: &mut [u8; 256],
    side_thr_dst: &mut [u8; 256],
    mask: &[[[u16; 4]; 5]; 64],
    starty4: usize,
    thr_lut: &[[u32; 16]; 2],
    sb64x: i32,
    ss_hor: i32,
    w4: i32,
    h4: i32,
) -> bool {
    let Some(f) = resolve_setup_thr_rows_simple_8bpc() else {
        return false;
    };
    unsafe {
        f(
            q_thr_dst,
            side_thr_dst,
            mask,
            starty4,
            thr_lut,
            sb64x,
            ss_hor,
            w4,
            h4,
        )
    };
    true
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
    // x86 horizontal filtering is no longer forced to scalar here: the SSE/AVX
    // apply kernels now use register gathers/scatters for the four rows instead
    // of the old temporary-stack path.  A full dav2d-style SB64 transpose kernel
    // would still be better, but this keeps arithmetic SIMD-enabled for both
    // orientations.

    // SAFETY: `resolve_deblock_apply` only returns the SSE/NEON kernel when the
    // corresponding feature was detected; the scalar default is always sound.
    unsafe {
        resolve_deblock_apply_8bpc()(
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

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn deblock_apply_hbd(
    dst: &mut [u16],
    off: isize,
    stride_line: isize,
    stride_tap: isize,
    width_neg: i32,
    width_pos: i32,
    q_thr_clamp: i32,
    neg_lossless: bool,
    pos_lossless: bool,
    bitdepth_max: i32,
) {
    // Keep the SIMD HBD apply path enabled for both orientations.  The x86
    // register gather/scatter path is still not as strong as dav2d's full
    // transpose kernel, but it avoids falling back to four scalar row filters.

    // SAFETY: `resolve_deblock_apply_hbd` only returns the SIMD kernel after
    // runtime feature detection. The caller provides an exclusive pixel slice;
    // offsets and widths are identical to the already-validated scalar path.
    unsafe {
        resolve_deblock_apply_hbd()(
            dst,
            off,
            stride_line,
            stride_tap,
            width_neg,
            width_pos,
            q_thr_clamp,
            neg_lossless,
            pos_lossless,
            bitdepth_max,
        )
    };
}
