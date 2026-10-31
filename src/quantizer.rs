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

use crate::levels::N_RECT_TX_SIZES;

/// Dequantization lookup function.
///
/// Maps a quantizer index to a dequantization value.
pub(crate) fn dq_lookup(qidx: i32) -> i32 {
    let qidx = qidx.clamp(0, 351);
    if qidx == 0 {
        return 64;
    }
    let qidx = qidx - 1;
    let shift = qidx / 24;
    let qidx = (qidx % 24) as usize;
    static DQ_LOOKUP_TBL: [u8; 24] = [
        40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55, 57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76,
        78,
    ];
    (DQ_LOOKUP_TBL[qidx] as i32) << shift
}

/// Predefined quantization matrices for every transform size, indexed
/// `[level][is_chroma][rect tx size]`. Level 15 is the flat matrix, i.e. QM
/// off, and has no table.
///
/// Like dav2d, each matrix is stored transposed (coefficients are stored
/// column-major), so the `WxH` table serves the `HxW` transform.
pub(crate) struct QmTables {
    tables: [[[&'static [u8]; N_RECT_TX_SIZES]; 2]; 15],
}

impl QmTables {
    #[inline]
    pub(crate) fn get(&self, level: u8, chroma: bool, tx: usize) -> Option<&'static [u8]> {
        self.tables
            .get(level as usize)
            .map(|t| t[chroma as usize][tx])
    }
}

/// `dst` takes every `hstep`-th column and `vstep`-th row of the 32-wide `src`.
fn subsample(dst: &mut Vec<u8>, src: &[u8], h: usize, hstep: usize, vstep: usize) {
    for y in (0..h).step_by(vstep) {
        dst.extend(src[y * 32..y * 32 + 32].iter().step_by(hstep));
    }
}

/// Transposes the `w x h` matrix `src`.
fn transpose(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            dst[x * h + y] = src[y * w + x];
        }
    }
    dst
}

/// Expands the row-packed lower triangle of a symmetric `sz x sz` matrix.
fn untriangle(src: &[u8], sz: usize) -> Vec<u8> {
    let mut dst = vec![0u8; sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let (r, c) = if x <= y { (y, x) } else { (x, y) };
            dst[y * sz + x] = src[r * (r + 1) / 2 + c];
        }
    }
    dst
}

pub(crate) fn qm_tables() -> &'static QmTables {
    static TABLES: std::sync::OnceLock<QmTables> = std::sync::OnceLock::new();
    TABLES.get_or_init(build_qm_tables)
}

#[cold]
fn build_qm_tables() -> QmTables {
    use crate::levels::txsz::*;
    use crate::qm_tables::{QM_TBL_32X16, QM_TBL_32X32_T};

    let leak = |v: Vec<u8>| -> &'static [u8] { Box::leak(v.into_boxed_slice()) };
    let mut tables = [[[&[][..]; N_RECT_TX_SIZES]; 2]; 15];
    for (lvl, level_tables) in tables.iter_mut().enumerate() {
        for (j, t) in level_tables.iter_mut().enumerate() {
            let t32x16 = &QM_TBL_32X16[lvl][j][..];
            let t32x32 = untriangle(&QM_TBL_32X32_T[lvl][j], 32);
            let sub = |src: &[u8], off: usize, h: usize, hstep: usize, vstep: usize| {
                let mut v = Vec::new();
                subsample(&mut v, &src[off..], h, hstep, vstep);
                v
            };
            let q4x4 = sub(&t32x32, 32 * 3 + 3, 32, 8, 8);
            let q8x4 = sub(t32x16, 32 + 1, 16, 4, 4);
            let q8x8 = sub(&t32x32, 32 + 1, 32, 4, 4);
            let q16x4 = sub(t32x16, 32, 16, 2, 4);
            let q16x8 = sub(t32x16, 0, 16, 2, 2);
            let q16x16 = sub(&t32x32, 0, 32, 2, 2);
            let q32x8 = sub(t32x16, 0, 16, 1, 2);
            let q32x4 = sub(t32x16, 32, 16, 1, 4);
            let q4x8 = transpose(&q8x4, 8, 4);
            let q4x16 = transpose(&q16x4, 16, 4);
            let q8x16 = transpose(&q16x8, 16, 8);
            let q4x32 = transpose(&q32x4, 32, 4);
            let q8x32 = transpose(&q32x8, 32, 8);
            let q16x32 = transpose(t32x16, 32, 16);

            // The w/h in each assignment is swapped on purpose: coefficients
            // are stored transposed.
            t[TX_4X4] = leak(q4x4);
            t[TX_8X8] = leak(q8x8);
            t[TX_16X16] = leak(q16x16);
            t[TX_32X32] = leak(t32x32);
            t[RTX_4X8] = leak(q8x4);
            t[RTX_8X4] = leak(q4x8);
            t[RTX_4X16] = leak(q16x4);
            t[RTX_16X4] = leak(q4x16);
            t[RTX_4X32] = leak(q32x4);
            t[RTX_32X4] = leak(q4x32);
            t[RTX_8X16] = leak(q16x8);
            t[RTX_16X8] = leak(q8x16);
            t[RTX_8X32] = leak(q32x8);
            t[RTX_32X8] = leak(q8x32);
            t[RTX_16X32] = t32x16;
            t[RTX_32X16] = leak(q16x32);

            t[TX_64X64] = t[TX_32X32];
            t[RTX_64X32] = t[TX_32X32];
            t[RTX_64X16] = t[RTX_32X16];
            t[RTX_64X8] = t[RTX_32X8];
            t[RTX_64X4] = t[RTX_32X4];
            t[RTX_32X64] = t[TX_32X32];
            t[RTX_16X64] = t[RTX_16X32];
            t[RTX_8X64] = t[RTX_8X32];
            t[RTX_4X64] = t[RTX_4X32];
        }
    }
    QmTables { tables }
}

#[cfg(test)]
mod qm_table_tests {
    use super::*;
    use crate::tables::TXFM_DIMENSIONS;

    #[test]
    fn predefined_qm_tables_have_tx_shapes() {
        let t = qm_tables();
        assert!(t.get(15, false, 0).is_none());
        for level in 0..15u8 {
            for chroma in [false, true] {
                for tx in 0..N_RECT_TX_SIZES {
                    let d = &TXFM_DIMENSIONS[tx];
                    let (w, h) = ((d.w as usize * 4).min(32), (d.h as usize * 4).min(32));
                    let m = t.get(level, chroma, tx).unwrap();
                    assert_eq!(m.len(), w * h, "level {level} tx {tx}");
                    // Square matrices are symmetric, and the stored WxH table
                    // is the transpose of the HxW one.
                    if w == h {
                        assert!((0..w).all(|y| (0..w).all(|x| m[y * w + x] == m[x * w + y])));
                    }
                }
                for (a, b, w, h) in [
                    (5, 6, 4, 8),
                    (13, 14, 4, 16),
                    (19, 20, 4, 32),
                    (7, 8, 8, 16),
                    (15, 16, 8, 32),
                    (9, 10, 16, 32),
                ] {
                    let (ma, mb) = (
                        t.get(level, chroma, a).unwrap(),
                        t.get(level, chroma, b).unwrap(),
                    );
                    assert!(
                        (0..h).all(|y| (0..w).all(|x| ma[x * h + y] == mb[y * w + x])),
                        "tx {a}/{b}"
                    );
                }
            }
        }
    }
}
