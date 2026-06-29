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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::headers::WarpedMotionParams;
use crate::intops::{apply_sign64, iclip};
use crate::levels::{Mv, MvXY, RefPair};
use crate::refmvs::{Block, INVALID_TRAJ, TemporalBlock, TemporalBlockMv, quantize_mv};

#[inline]
#[target_feature(enable = "avx2")]
fn store_block(dst: &mut Block, src: &Block) {
    if std::mem::size_of::<Block>() == 64 {
        let s = src as *const Block as *const __m256i;
        let d = dst as *mut Block as *mut __m256i;
        unsafe {
            let lo = _mm256_loadu_si256(s);
            let hi = _mm256_loadu_si256(s.add(1));
            _mm256_storeu_si256(d, lo);
            _mm256_storeu_si256(d.add(1), hi);
        }
    } else {
        *dst = *src;
    }
}

#[inline(always)]
fn temporal_width(bw4: i32) -> usize {
    ((bw4 + 1) >> 1) as usize
}

#[target_feature(enable = "avx2")]
pub(crate) fn splat_mv_avx2(
    s_dst: &mut [Block],
    s_src: &mut Block,
    mut t_dst: Option<&mut [TemporalBlock]>,
    t_stride: isize,
    t_src: &TemporalBlock,
    bw4: i32,
    bh4: i32,
) {
    let w = bw4 as usize;
    let h = bh4 as usize;
    let t_w = temporal_width(bw4);
    let t_stride = t_stride as usize;

    s_src.oy4 = 0;
    for (yp, row_pair) in s_dst
        .as_chunks_mut::<256>()
        .0
        .iter_mut()
        .take(h.div_ceil(2))
        .enumerate()
    {
        let y = yp * 2;
        let oy = y as u8;
        let (top_row, bottom_part) = row_pair.split_at_mut(128);
        let bottom_row = &mut bottom_part[..128];

        for (xp, top_pair) in top_row[..w].chunks_mut(2).enumerate() {
            let x = xp * 2;
            let ox = x as u8;
            let mut b = *s_src;
            b.oy4 = oy;
            b.ox4 = ox;
            store_block(&mut top_pair[0], &b);
            if top_pair.len() > 1 {
                b.ox4 = ox + 1;
                store_block(&mut top_pair[1], &b);
            }
            if y + 1 < h {
                let bottom_pair = &mut bottom_row[x..usize::min(x + 2, w)];
                b.oy4 = oy + 1;
                b.ox4 = ox;
                store_block(&mut bottom_pair[0], &b);
                if bottom_pair.len() > 1 {
                    b.ox4 = ox + 1;
                    store_block(&mut bottom_pair[1], &b);
                }
            }
        }

        if let Some(td) = t_dst.as_deref_mut() {
            let t_row = &mut td[yp * t_stride..][..t_w];
            for t in t_row.iter_mut() {
                *t = *t_src;
            }
        }
    }

    s_src.ox4 = ((bw4 + 1) & !1) as u8;
    s_src.oy4 = (h.div_ceil(2) * 2) as u8;
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn splat_warpmv_avx2(
    s_dst: &mut [Block],
    s_src: &mut Block,
    mut t_dst: Option<&mut [TemporalBlock]>,
    t_stride: isize,
    t_src: &mut TemporalBlock,
    mut mvy: i64,
    mut mvx: i64,
    mat: &WarpedMotionParams,
    bw4: i32,
    bh4: i32,
) {
    debug_assert!(bw4 > 1 && bh4 > 1);
    let w = bw4 as usize;
    let h_pairs = (bh4 >> 1) as usize;
    let t_stride = t_stride as usize;

    s_src.oy4 = 0;
    for (yp, row_pair) in s_dst
        .as_chunks_mut::<256>()
        .0
        .iter_mut()
        .take(h_pairs)
        .enumerate()
    {
        let oy = (yp * 2) as u8;
        let (top_row, bottom_part) = row_pair.split_at_mut(128);
        let bottom_row = &mut bottom_part[..128];
        let mut mvxi = mvx;
        let mut mvyi = mvy;

        for (xp, top_pair) in top_row[..w].as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let ox = (xp * 2) as u8;
            let warpmv = Mv {
                c: MvXY {
                    y: iclip(
                        apply_sign64((mvyi.abs() + 4096) >> 13, mvyi),
                        -0xffff,
                        0xffff,
                    ),
                    x: iclip(
                        apply_sign64((mvxi.abs() + 4096) >> 13, mvxi),
                        -0xffff,
                        0xffff,
                    ),
                },
            };
            if s_src.mf & 2 != 0 {
                s_src.mv[0] = warpmv;
            }
            let qmv = quantize_mv(warpmv);
            t_src.mv = TemporalBlockMv::from_mvs(qmv, qmv);

            let mut b = *s_src;
            b.oy4 = oy;
            b.ox4 = ox;
            store_block(&mut top_pair[0], &b);
            b.ox4 = ox + 1;
            store_block(&mut top_pair[1], &b);
            b.oy4 = oy + 1;
            store_block(&mut bottom_row[(xp * 2) + 1], &b);
            b.ox4 = ox;
            store_block(&mut bottom_row[xp * 2], &b);

            if let Some(td) = t_dst.as_deref_mut() {
                let ti = yp * t_stride + xp;
                let n = t_src.mv.packed();
                td[ti].mv = TemporalBlockMv::from_packed(n);
                td[ti].r#ref = RefPair::from_pair(if n == INVALID_TRAJ as u32 * 0x10001 {
                    -1
                } else {
                    t_src.r#ref.pair()
                });
            }

            mvxi += (mat.matrix[2] as i64 - 0x10000) * 8;
            mvyi += mat.matrix[4] as i64 * 8;
        }
        mvx += mat.matrix[3] as i64 * 8;
        mvy += (mat.matrix[5] as i64 - 0x10000) * 8;
    }

    s_src.ox4 = bw4 as u8;
    s_src.oy4 = (h_pairs * 2) as u8;
}
