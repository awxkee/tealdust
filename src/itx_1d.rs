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

use crate::levels::{N_TX_1D_TYPES, N_TX_SIZES};
use crate::simd::I32x8 as i32x8;
use std::convert::TryInto;

pub type Itx1dFn = fn(c: &mut [i32], stride: usize);

static ADST4_KERNEL: [i8; 16] = [
    18, 50, 75, 89, 50, 89, 18, -75, 75, 18, -89, 50, 89, -75, 50, -18,
];

static ADST8_KERNEL: [i8; 64] = [
    11, 34, 54, 71, 84, 88, 79, 50, 28, 74, 89, 68, 17, -44, -83, -69, 44, 89, 48, -41, -89, -44,
    50, 81, 58, 76, -34, -86, 10, 88, 6, -84, 70, 39, -87, 1, 86, -44, -59, 78, 79, -12, -66, 87,
    -35, -44, 86, -62, 86, -58, 12, 38, -75, 88, -74, 40, 89, -86, 79, -70, 58, -44, 29, -14,
];

static ADST16_KERNEL: [i8; 256] = [
    8, 25, 41, 55, 67, 77, 84, 88, 89, 87, 81, 73, 62, 48, 33, 17, 17, 48, 73, 87, 88, 77, 55, 25,
    -8, -41, -67, -84, -89, -81, -62, -33, 25, 67, 88, 81, 48, 0, -48, -81, -88, -67, -25, 25, 67,
    88, 81, 48, 33, 81, 84, 41, -25, -77, -87, -48, 17, 73, 88, 55, -8, -67, -89, -62, 41, 88, 62,
    -17, -81, -77, -8, 67, 87, 33, -48, -89, -55, 25, 84, 73, 48, 88, 25, -67, -81, 0, 81, 67, -25,
    -88, -48, 48, 88, 25, -67, -81, 55, 81, -17, -89, -25, 77, 62, -48, -84, 8, 88, 33, -73, -67,
    41, 87, 62, 67, -55, -73, 48, 77, -41, -81, 33, 84, -25, -87, 17, 88, -8, -89, 67, 48, -81,
    -25, 88, 0, -88, 25, 81, -48, -67, 67, 48, -81, -25, 88, 73, 25, -89, 33, 67, -77, -17, 88,
    -41, -62, 81, 8, -87, 48, 55, -84, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77, 77, 0, -77,
    77, 81, -25, -48, 88, -67, 0, 67, -88, 48, 25, -81, 81, -25, -48, 88, -67, 84, -48, -8, 62,
    -88, 77, -33, -25, 73, -89, 67, -17, -41, 81, -87, 55, 87, -67, 33, 8, -48, 77, -89, 81, -55,
    17, 25, -62, 84, -88, 73, -41, 88, -81, 67, -48, 25, 0, -25, 48, -67, 81, -88, 88, -81, 67,
    -48, 25, 89, -88, 87, -84, 81, -77, 73, -67, 62, -55, 48, -41, 33, -25, 17, -8,
];

static FLIPADST4_KERNEL: [i8; 16] = [
    89, 75, 50, 18, 75, -18, -89, -50, 50, -89, 18, 75, 18, -50, 75, -89,
];

static FLIPADST16_KERNEL: [i8; 256] = [
    89, 88, 87, 84, 81, 77, 73, 67, 62, 55, 48, 41, 33, 25, 17, 8, 88, 81, 67, 48, 25, 0, -25, -48,
    -67, -81, -88, -88, -81, -67, -48, -25, 87, 67, 33, -8, -48, -77, -89, -81, -55, -17, 25, 62,
    84, 88, 73, 41, 84, 48, -8, -62, -88, -77, -33, 25, 73, 89, 67, 17, -41, -81, -87, -55, 81, 25,
    -48, -88, -67, 0, 67, 88, 48, -25, -81, -81, -25, 48, 88, 67, 77, 0, -77, -77, 0, 77, 77, 0,
    -77, -77, 0, 77, 77, 0, -77, -77, 73, -25, -89, -33, 67, 77, -17, -88, -41, 62, 81, -8, -87,
    -48, 55, 84, 67, -48, -81, 25, 88, 0, -88, -25, 81, 48, -67, -67, 48, 81, -25, -88, 62, -67,
    -55, 73, 48, -77, -41, 81, 33, -84, -25, 87, 17, -88, -8, 89, 55, -81, -17, 89, -25, -77, 62,
    48, -84, -8, 88, -33, -73, 67, 41, -87, 48, -88, 25, 67, -81, 0, 81, -67, -25, 88, -48, -48,
    88, -25, -67, 81, 41, -88, 62, 17, -81, 77, -8, -67, 87, -33, -48, 89, -55, -25, 84, -73, 33,
    -81, 84, -41, -25, 77, -87, 48, 17, -73, 88, -55, -8, 67, -89, 62, 25, -67, 88, -81, 48, 0,
    -48, 81, -88, 67, -25, -25, 67, -88, 81, -48, 17, -48, 73, -87, 88, -77, 55, -25, -8, 41, -67,
    84, -89, 81, -62, 33, 8, -25, 41, -55, 67, -77, 84, -88, 89, -87, 81, -73, 62, -48, 33, -17,
];

static DDT8_KERNEL: [i8; 64] = [
    4, 6, 22, 57, 96, 103, 78, 56, 7, 14, 48, 94, 73, -17, -79, -96, 15, 36, 85, 76, -43, -80, 7,
    98, 33, 77, 88, -26, -69, 56, 56, -77, 65, 100, 0, -73, 55, 15, -82, 54, 98, 45, -86, 34, 20,
    -66, 79, -33, 106, -57, -23, 54, -71, 75, -56, 19, 80, -98, 82, -66, 53, -41, 26, -6,
];

static DDT16_KERNEL: [i8; 256] = [
    12, 17, 37, 45, 47, 60, 64, 82, 89, 100, 92, 84, 69, 50, 51, 44, 15, 23, 49, 60, 60, 74, 70,
    73, 48, 9, -35, -71, -83, -79, -89, -95, 19, 30, 60, 69, 61, 64, 40, 3, -53, -99, -91, -46, 2,
    47, 73, 124, 23, 38, 69, 73, 49, 28, -19, -80, -96, -45, 42, 88, 75, 14, -17, -126, 30, 48, 75,
    66, 19, -31, -79, -91, -5, 84, 71, -16, -78, -60, -45, 108, 39, 61, 75, 40, -29, -87, -78, 10,
    89, 36, -69, -67, 18, 67, 89, -81, 51, 76, 61, -8, -77, -82, 11, 94, 16, -81, -22, 79, 50, -37,
    -103, 54, 66, 87, 29, -65, -83, 4, 92, 18, -83, 4, 85, -22, -85, -6, 97, -30, 78, 83, -18, -91,
    -16, 88, 28, -84, 12, 73, -60, -46, 81, 49, -83, 16, 88, 59, -67, -57, 75, 54, -85, -5, 75,
    -60, -17, 84, -43, -80, 71, -6, 94, 19, -96, 21, 93, -55, -41, 80, -51, -17, 77, -68, -6, 98,
    -56, 1, 97, -30, -83, 86, 3, -77, 82, -17, -43, 76, -70, 15, 53, -99, 44, 3, 93, -73, -28, 81,
    -92, 29, 39, -70, 81, -55, 11, 46, -81, 90, -31, -4, 83, -99, 40, 8, -74, 88, -83, 47, -14,
    -21, 56, -83, 88, -71, 22, 5, 68, -99, 84, -69, 32, 3, -37, 55, -75, 81, -83, 82, -69, 48, -11,
    -3, 50, -76, 83, -90, 97, -86, 83, -68, 67, -56, 49, -40, 32, -19, 5, 2,
];

static DCT8_ODD_KERNEL: [[i8; 4]; 4] = [
    [89, 75, 50, 18],
    [75, -18, -89, -50],
    [50, -89, 18, 75],
    [18, -50, 75, -89],
];

static DCT16_ODD_KERNEL: [[i8; 8]; 8] = [
    [90, 87, 80, 70, 57, 43, 26, 9],
    [87, 57, 9, -43, -80, -90, -70, -26],
    [80, 9, -70, -87, -26, 57, 90, 43],
    [70, -43, -87, 9, 90, 26, -80, -57],
    [57, -80, -26, 90, -9, -87, 43, 70],
    [43, -90, 57, 26, -87, 70, 9, -80],
    [26, -70, 90, -80, 43, 9, -57, 87],
    [9, -26, 43, -57, 70, -80, 87, -90],
];

static DCT32_ODD_KERNEL: [[i8; 16]; 16] = [
    [
        90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 47, 39, 30, 22, 13, 4,
    ],
    [
        90, 82, 67, 47, 22, -4, -30, -54, -73, -85, -90, -88, -78, -61, -39, -13,
    ],
    [
        88, 67, 30, -13, -54, -82, -90, -78, -47, -4, 39, 73, 90, 85, 61, 22,
    ],
    [
        85, 47, -13, -67, -90, -73, -22, 39, 82, 88, 54, -4, -61, -90, -78, -30,
    ],
    [
        82, 22, -54, -90, -61, 13, 78, 85, 30, -47, -90, -67, 4, 73, 88, 39,
    ],
    [
        78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 30, 90, 54, -39, -90, -47,
    ],
    [
        73, -30, -90, -22, 78, 67, -39, -90, -13, 82, 61, -47, -88, -4, 85, 54,
    ],
    [
        67, -54, -78, 39, 85, -22, -90, 4, 90, 13, -88, -30, 82, 47, -73, -61,
    ],
    [
        61, -73, -47, 82, 30, -88, -13, 90, -4, -90, 22, 85, -39, -78, 54, 67,
    ],
    [
        54, -85, -4, 88, -47, -61, 82, 13, -90, 39, 67, -78, -22, 90, -30, -73,
    ],
    [
        47, -90, 39, 54, -90, 30, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78,
    ],
    [
        39, -88, 73, -4, -67, 90, -47, -30, 85, -78, 13, 61, -90, 54, 22, -82,
    ],
    [
        30, -78, 90, -61, 4, 54, -88, 82, -39, -22, 73, -90, 67, -13, -47, 85,
    ],
    [
        22, -61, 85, -90, 73, -39, -4, 47, -78, 90, -82, 54, -13, -30, 67, -88,
    ],
    [
        13, -39, 61, -78, 88, -90, 85, -73, 54, -30, 4, 22, -47, 67, -82, 90,
    ],
    [
        4, -13, 22, -30, 39, -47, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
    ],
];

#[inline(always)]
fn load_1d<const N: usize>(c: &[i32], stride: usize) -> [i32; N] {
    let span = (N - 1) * stride;
    let c = &c[..=span];
    let mut out = [0i32; N];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = c[i * stride];
    }
    out
}

#[inline(always)]
fn store_1d<const N: usize>(c: &mut [i32], stride: usize, v: &[i32; N]) {
    let span = (N - 1) * stride;
    let c = &mut c[..=span];
    for (i, &src) in v.iter().enumerate() {
        c[i * stride] = src;
    }
}

#[inline(always)]
fn sum_row4(row: &[i8; 4], x: &[i32; 4]) -> i32 {
    let mut acc = 0i32;
    acc += (row[0] as i32) * x[0];
    acc += (row[1] as i32) * x[1];
    acc += (row[2] as i32) * x[2];
    acc += (row[3] as i32) * x[3];
    acc
}

#[inline(always)]
fn sum_row8(row: &[i8; 8], x: &[i32; 8]) -> i32 {
    let mut acc = 0i32;
    acc += (row[0] as i32) * x[0];
    acc += (row[1] as i32) * x[1];
    acc += (row[2] as i32) * x[2];
    acc += (row[3] as i32) * x[3];
    acc += (row[4] as i32) * x[4];
    acc += (row[5] as i32) * x[5];
    acc += (row[6] as i32) * x[6];
    acc += (row[7] as i32) * x[7];
    acc
}

#[inline(always)]
fn sum_row16(row: &[i8; 16], x: &[i32; 16]) -> i32 {
    let mut acc = 0i32;
    acc += (row[0] as i32) * x[0];
    acc += (row[1] as i32) * x[1];
    acc += (row[2] as i32) * x[2];
    acc += (row[3] as i32) * x[3];
    acc += (row[4] as i32) * x[4];
    acc += (row[5] as i32) * x[5];
    acc += (row[6] as i32) * x[6];
    acc += (row[7] as i32) * x[7];
    acc += (row[8] as i32) * x[8];
    acc += (row[9] as i32) * x[9];
    acc += (row[10] as i32) * x[10];
    acc += (row[11] as i32) * x[11];
    acc += (row[12] as i32) * x[12];
    acc += (row[13] as i32) * x[13];
    acc += (row[14] as i32) * x[14];
    acc += (row[15] as i32) * x[15];
    acc
}

#[inline(always)]
fn odd_4(v: &[i32; 8]) -> [i32; 4] {
    [v[1], v[3], v[5], v[7]]
}

#[inline(always)]
fn even_4_from_8(v: &[i32; 8]) -> [i32; 4] {
    [v[0], v[2], v[4], v[6]]
}

#[inline(always)]
fn odd_8(v: &[i32; 16]) -> [i32; 8] {
    [v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]]
}

#[inline(always)]
fn even_8_from_16(v: &[i32; 16]) -> [i32; 8] {
    [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]
}

#[inline(always)]
fn odd_16(v: &[i32; 32]) -> [i32; 16] {
    [
        v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15], v[17], v[19], v[21], v[23], v[25],
        v[27], v[29], v[31],
    ]
}

#[inline(always)]
fn even_16_from_32(v: &[i32; 32]) -> [i32; 16] {
    [
        v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14], v[16], v[18], v[20], v[22], v[24],
        v[26], v[28], v[30],
    ]
}

#[inline(always)]
fn inv_dct4_array(v: &mut [i32; 4]) {
    let a0 = v[0] * 64 + v[2] * 64;
    let a1 = v[0] * 64 - v[2] * 64;
    let b0 = v[1] * 83 + v[3] * 35;
    let b1 = v[1] * 35 - v[3] * 83;

    v[0] = a0 + b0;
    v[1] = a1 + b1;
    v[2] = a1 - b1;
    v[3] = a0 - b0;
}

#[inline(always)]
fn inv_dct8_array(v: &mut [i32; 8]) {
    let mut e = even_4_from_8(v);
    inv_dct4_array(&mut e);
    let odd = odd_4(v);
    let b0 = sum_row4(&DCT8_ODD_KERNEL[0], &odd);
    let b1 = sum_row4(&DCT8_ODD_KERNEL[1], &odd);
    let b2 = sum_row4(&DCT8_ODD_KERNEL[2], &odd);
    let b3 = sum_row4(&DCT8_ODD_KERNEL[3], &odd);

    v[0] = e[0] + b0;
    v[7] = e[0] - b0;
    v[1] = e[1] + b1;
    v[6] = e[1] - b1;
    v[2] = e[2] + b2;
    v[5] = e[2] - b2;
    v[3] = e[3] + b3;
    v[4] = e[3] - b3;
}

#[inline(always)]
fn inv_dct16_array(v: &mut [i32; 16]) {
    let mut e = even_8_from_16(v);
    inv_dct8_array(&mut e);
    let odd = odd_8(v);
    let b0 = sum_row8(&DCT16_ODD_KERNEL[0], &odd);
    let b1 = sum_row8(&DCT16_ODD_KERNEL[1], &odd);
    let b2 = sum_row8(&DCT16_ODD_KERNEL[2], &odd);
    let b3 = sum_row8(&DCT16_ODD_KERNEL[3], &odd);
    let b4 = sum_row8(&DCT16_ODD_KERNEL[4], &odd);
    let b5 = sum_row8(&DCT16_ODD_KERNEL[5], &odd);
    let b6 = sum_row8(&DCT16_ODD_KERNEL[6], &odd);
    let b7 = sum_row8(&DCT16_ODD_KERNEL[7], &odd);

    v[0] = e[0] + b0;
    v[15] = e[0] - b0;
    v[1] = e[1] + b1;
    v[14] = e[1] - b1;
    v[2] = e[2] + b2;
    v[13] = e[2] - b2;
    v[3] = e[3] + b3;
    v[12] = e[3] - b3;
    v[4] = e[4] + b4;
    v[11] = e[4] - b4;
    v[5] = e[5] + b5;
    v[10] = e[5] - b5;
    v[6] = e[6] + b6;
    v[9] = e[6] - b6;
    v[7] = e[7] + b7;
    v[8] = e[7] - b7;
}

#[inline(always)]
fn inv_dct32_array(v: &mut [i32; 32]) {
    let mut e = even_16_from_32(v);
    inv_dct16_array(&mut e);
    let odd = odd_16(v);
    let b0 = sum_row16(&DCT32_ODD_KERNEL[0], &odd);
    let b1 = sum_row16(&DCT32_ODD_KERNEL[1], &odd);
    let b2 = sum_row16(&DCT32_ODD_KERNEL[2], &odd);
    let b3 = sum_row16(&DCT32_ODD_KERNEL[3], &odd);
    let b4 = sum_row16(&DCT32_ODD_KERNEL[4], &odd);
    let b5 = sum_row16(&DCT32_ODD_KERNEL[5], &odd);
    let b6 = sum_row16(&DCT32_ODD_KERNEL[6], &odd);
    let b7 = sum_row16(&DCT32_ODD_KERNEL[7], &odd);
    let b8 = sum_row16(&DCT32_ODD_KERNEL[8], &odd);
    let b9 = sum_row16(&DCT32_ODD_KERNEL[9], &odd);
    let b10 = sum_row16(&DCT32_ODD_KERNEL[10], &odd);
    let b11 = sum_row16(&DCT32_ODD_KERNEL[11], &odd);
    let b12 = sum_row16(&DCT32_ODD_KERNEL[12], &odd);
    let b13 = sum_row16(&DCT32_ODD_KERNEL[13], &odd);
    let b14 = sum_row16(&DCT32_ODD_KERNEL[14], &odd);
    let b15 = sum_row16(&DCT32_ODD_KERNEL[15], &odd);

    v[0] = e[0] + b0;
    v[31] = e[0] - b0;
    v[1] = e[1] + b1;
    v[30] = e[1] - b1;
    v[2] = e[2] + b2;
    v[29] = e[2] - b2;
    v[3] = e[3] + b3;
    v[28] = e[3] - b3;
    v[4] = e[4] + b4;
    v[27] = e[4] - b4;
    v[5] = e[5] + b5;
    v[26] = e[5] - b5;
    v[6] = e[6] + b6;
    v[25] = e[6] - b6;
    v[7] = e[7] + b7;
    v[24] = e[7] - b7;
    v[8] = e[8] + b8;
    v[23] = e[8] - b8;
    v[9] = e[9] + b9;
    v[22] = e[9] - b9;
    v[10] = e[10] + b10;
    v[21] = e[10] - b10;
    v[11] = e[11] + b11;
    v[20] = e[11] - b11;
    v[12] = e[12] + b12;
    v[19] = e[12] - b12;
    v[13] = e[13] + b13;
    v[18] = e[13] - b13;
    v[14] = e[14] + b14;
    v[17] = e[14] - b14;
    v[15] = e[15] + b15;
    v[16] = e[15] - b15;
}

#[inline(always)]
fn inv_dct4_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<4>(c, stride);
    inv_dct4_array(&mut v);
    store_1d::<4>(c, stride, &v);
}

#[inline(always)]
fn inv_dct8_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<8>(c, stride);
    inv_dct8_array(&mut v);
    store_1d::<8>(c, stride, &v);
}

#[inline(always)]
fn inv_dct16_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<16>(c, stride);
    inv_dct16_array(&mut v);
    store_1d::<16>(c, stride, &v);
}

#[inline(always)]
fn inv_dct32_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<32>(c, stride);
    inv_dct32_array(&mut v);
    store_1d::<32>(c, stride, &v);
}

#[inline(always)]
fn scale_array<const N: usize>(v: &mut [i32; N], scale: i32) {
    for x in v.iter_mut() {
        *x *= scale;
    }
}

#[inline(never)]
fn inv_dst_1d(c: &mut [i32], start: usize, stride: usize, mat: &[i8], n: usize, flip: bool) {
    let mut sums = [0i32; 16];
    let mut mi = 0;

    for i in 0..n {
        let mut sum = 0i32;
        for j in 0..n {
            sum += mat[mi] as i32 * c[start + j * stride];
            mi += 1;
        }
        sums[i] = sum;
    }

    if flip {
        for i in 0..n {
            c[start + (n - 1 - i) * stride] = sums[i];
        }
    } else {
        for i in 0..n {
            c[start + i * stride] = sums[i];
        }
    }
}

fn inv_adst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST4_KERNEL, 4, false);
}

fn inv_adst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST8_KERNEL, 8, false);
}

fn inv_adst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST16_KERNEL, 16, false);
}

fn inv_flipadst4_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &FLIPADST4_KERNEL, 4, false);
}

fn inv_flipadst8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &ADST8_KERNEL, 8, true);
}

fn inv_flipadst16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &FLIPADST16_KERNEL, 16, false);
}

fn inv_ddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT8_KERNEL, 8, false);
}

fn inv_ddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT16_KERNEL, 16, false);
}

fn inv_flipddt8_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT8_KERNEL, 8, true);
}

fn inv_flipddt16_1d(c: &mut [i32], stride: usize) {
    inv_dst_1d(c, 0, stride, &DDT16_KERNEL, 16, true);
}

fn inv_identity4_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<4>(c, stride);
    scale_array(&mut v, 128);
    store_1d::<4>(c, stride, &v);
}

fn inv_identity8_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<8>(c, stride);
    scale_array(&mut v, 181);
    store_1d::<8>(c, stride, &v);
}

fn inv_identity16_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<16>(c, stride);
    scale_array(&mut v, 256);
    store_1d::<16>(c, stride, &v);
}

fn inv_identity32_1d(c: &mut [i32], stride: usize) {
    let mut v = load_1d::<32>(c, stride);
    scale_array(&mut v, 362);
    store_1d::<32>(c, stride, &v);
}

/// `(&mut [i32], base, stride)` — vectorized 1-D transform over 8 columns.
pub(crate) type Itx1dFnX8 = fn(&mut [i32], usize, usize);

#[inline(always)]
fn ldx8(c: &[i32], off: usize) -> i32x8 {
    let src: &[i32; 8] = c[off..off + 8].try_into().unwrap();
    i32x8::from(*src)
}

#[inline(always)]
fn stx8(c: &mut [i32], off: usize, v: i32x8) {
    let dst: &mut [i32; 8] = (&mut c[off..off + 8]).try_into().unwrap();
    *dst = v.to_array();
}

#[inline(always)]
fn mulc(v: i32x8, k: i32) -> i32x8 {
    v * i32x8::splat(k)
}

#[inline(always)]
fn load_1d_x8<const N: usize>(c: &[i32], base: usize, stride: usize) -> [i32x8; N] {
    let span = base + (N - 1) * stride + 7;
    let c = &c[..=span];
    let zero = i32x8::splat(0);
    let mut out = [zero; N];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = ldx8(c, base + i * stride);
    }
    out
}

#[inline(always)]
fn store_1d_x8<const N: usize>(c: &mut [i32], base: usize, stride: usize, v: &[i32x8; N]) {
    let span = base + (N - 1) * stride + 7;
    let c = &mut c[..=span];
    for (i, &src) in v.iter().enumerate() {
        stx8(c, base + i * stride, src);
    }
}

#[inline(always)]
fn sum_row4_x8(row: &[i8; 4], x: &[i32x8; 4]) -> i32x8 {
    let mut acc = i32x8::splat(0);
    acc += mulc(x[0], row[0] as i32);
    acc += mulc(x[1], row[1] as i32);
    acc += mulc(x[2], row[2] as i32);
    acc += mulc(x[3], row[3] as i32);
    acc
}

#[inline(always)]
fn sum_row8_x8(row: &[i8; 8], x: &[i32x8; 8]) -> i32x8 {
    let mut acc = i32x8::splat(0);
    acc += mulc(x[0], row[0] as i32);
    acc += mulc(x[1], row[1] as i32);
    acc += mulc(x[2], row[2] as i32);
    acc += mulc(x[3], row[3] as i32);
    acc += mulc(x[4], row[4] as i32);
    acc += mulc(x[5], row[5] as i32);
    acc += mulc(x[6], row[6] as i32);
    acc += mulc(x[7], row[7] as i32);
    acc
}

#[inline(always)]
fn sum_row16_x8(row: &[i8; 16], x: &[i32x8; 16]) -> i32x8 {
    let mut acc = i32x8::splat(0);
    acc += mulc(x[0], row[0] as i32);
    acc += mulc(x[1], row[1] as i32);
    acc += mulc(x[2], row[2] as i32);
    acc += mulc(x[3], row[3] as i32);
    acc += mulc(x[4], row[4] as i32);
    acc += mulc(x[5], row[5] as i32);
    acc += mulc(x[6], row[6] as i32);
    acc += mulc(x[7], row[7] as i32);
    acc += mulc(x[8], row[8] as i32);
    acc += mulc(x[9], row[9] as i32);
    acc += mulc(x[10], row[10] as i32);
    acc += mulc(x[11], row[11] as i32);
    acc += mulc(x[12], row[12] as i32);
    acc += mulc(x[13], row[13] as i32);
    acc += mulc(x[14], row[14] as i32);
    acc += mulc(x[15], row[15] as i32);
    acc
}

#[inline(always)]
fn odd_4_x8(v: &[i32x8; 8]) -> [i32x8; 4] {
    [v[1], v[3], v[5], v[7]]
}

#[inline(always)]
fn even_4_from_8_x8(v: &[i32x8; 8]) -> [i32x8; 4] {
    [v[0], v[2], v[4], v[6]]
}

#[inline(always)]
fn odd_8_x8(v: &[i32x8; 16]) -> [i32x8; 8] {
    [v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]]
}

#[inline(always)]
fn even_8_from_16_x8(v: &[i32x8; 16]) -> [i32x8; 8] {
    [v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]]
}

#[inline(always)]
fn odd_16_x8(v: &[i32x8; 32]) -> [i32x8; 16] {
    [
        v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15], v[17], v[19], v[21], v[23], v[25],
        v[27], v[29], v[31],
    ]
}

#[inline(always)]
fn even_16_from_32_x8(v: &[i32x8; 32]) -> [i32x8; 16] {
    [
        v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14], v[16], v[18], v[20], v[22], v[24],
        v[26], v[28], v[30],
    ]
}

#[inline(always)]
fn inv_dct4_array_x8(v: &mut [i32x8; 4]) {
    let a0 = mulc(v[0], 64) + mulc(v[2], 64);
    let a1 = mulc(v[0], 64) - mulc(v[2], 64);
    let b0 = mulc(v[1], 83) + mulc(v[3], 35);
    let b1 = mulc(v[1], 35) - mulc(v[3], 83);

    v[0] = a0 + b0;
    v[1] = a1 + b1;
    v[2] = a1 - b1;
    v[3] = a0 - b0;
}

#[inline(always)]
fn inv_dct8_array_x8(v: &mut [i32x8; 8]) {
    let mut e = even_4_from_8_x8(v);
    inv_dct4_array_x8(&mut e);
    let odd = odd_4_x8(v);
    let b0 = sum_row4_x8(&DCT8_ODD_KERNEL[0], &odd);
    let b1 = sum_row4_x8(&DCT8_ODD_KERNEL[1], &odd);
    let b2 = sum_row4_x8(&DCT8_ODD_KERNEL[2], &odd);
    let b3 = sum_row4_x8(&DCT8_ODD_KERNEL[3], &odd);

    v[0] = e[0] + b0;
    v[7] = e[0] - b0;
    v[1] = e[1] + b1;
    v[6] = e[1] - b1;
    v[2] = e[2] + b2;
    v[5] = e[2] - b2;
    v[3] = e[3] + b3;
    v[4] = e[3] - b3;
}

#[inline(always)]
fn inv_dct16_array_x8(v: &mut [i32x8; 16]) {
    let mut e = even_8_from_16_x8(v);
    inv_dct8_array_x8(&mut e);
    let odd = odd_8_x8(v);
    let b0 = sum_row8_x8(&DCT16_ODD_KERNEL[0], &odd);
    let b1 = sum_row8_x8(&DCT16_ODD_KERNEL[1], &odd);
    let b2 = sum_row8_x8(&DCT16_ODD_KERNEL[2], &odd);
    let b3 = sum_row8_x8(&DCT16_ODD_KERNEL[3], &odd);
    let b4 = sum_row8_x8(&DCT16_ODD_KERNEL[4], &odd);
    let b5 = sum_row8_x8(&DCT16_ODD_KERNEL[5], &odd);
    let b6 = sum_row8_x8(&DCT16_ODD_KERNEL[6], &odd);
    let b7 = sum_row8_x8(&DCT16_ODD_KERNEL[7], &odd);

    v[0] = e[0] + b0;
    v[15] = e[0] - b0;
    v[1] = e[1] + b1;
    v[14] = e[1] - b1;
    v[2] = e[2] + b2;
    v[13] = e[2] - b2;
    v[3] = e[3] + b3;
    v[12] = e[3] - b3;
    v[4] = e[4] + b4;
    v[11] = e[4] - b4;
    v[5] = e[5] + b5;
    v[10] = e[5] - b5;
    v[6] = e[6] + b6;
    v[9] = e[6] - b6;
    v[7] = e[7] + b7;
    v[8] = e[7] - b7;
}

#[inline(always)]
fn inv_dct32_array_x8(v: &mut [i32x8; 32]) {
    let mut e = even_16_from_32_x8(v);
    inv_dct16_array_x8(&mut e);
    let odd = odd_16_x8(v);
    let b0 = sum_row16_x8(&DCT32_ODD_KERNEL[0], &odd);
    let b1 = sum_row16_x8(&DCT32_ODD_KERNEL[1], &odd);
    let b2 = sum_row16_x8(&DCT32_ODD_KERNEL[2], &odd);
    let b3 = sum_row16_x8(&DCT32_ODD_KERNEL[3], &odd);
    let b4 = sum_row16_x8(&DCT32_ODD_KERNEL[4], &odd);
    let b5 = sum_row16_x8(&DCT32_ODD_KERNEL[5], &odd);
    let b6 = sum_row16_x8(&DCT32_ODD_KERNEL[6], &odd);
    let b7 = sum_row16_x8(&DCT32_ODD_KERNEL[7], &odd);
    let b8 = sum_row16_x8(&DCT32_ODD_KERNEL[8], &odd);
    let b9 = sum_row16_x8(&DCT32_ODD_KERNEL[9], &odd);
    let b10 = sum_row16_x8(&DCT32_ODD_KERNEL[10], &odd);
    let b11 = sum_row16_x8(&DCT32_ODD_KERNEL[11], &odd);
    let b12 = sum_row16_x8(&DCT32_ODD_KERNEL[12], &odd);
    let b13 = sum_row16_x8(&DCT32_ODD_KERNEL[13], &odd);
    let b14 = sum_row16_x8(&DCT32_ODD_KERNEL[14], &odd);
    let b15 = sum_row16_x8(&DCT32_ODD_KERNEL[15], &odd);

    v[0] = e[0] + b0;
    v[31] = e[0] - b0;
    v[1] = e[1] + b1;
    v[30] = e[1] - b1;
    v[2] = e[2] + b2;
    v[29] = e[2] - b2;
    v[3] = e[3] + b3;
    v[28] = e[3] - b3;
    v[4] = e[4] + b4;
    v[27] = e[4] - b4;
    v[5] = e[5] + b5;
    v[26] = e[5] - b5;
    v[6] = e[6] + b6;
    v[25] = e[6] - b6;
    v[7] = e[7] + b7;
    v[24] = e[7] - b7;
    v[8] = e[8] + b8;
    v[23] = e[8] - b8;
    v[9] = e[9] + b9;
    v[22] = e[9] - b9;
    v[10] = e[10] + b10;
    v[21] = e[10] - b10;
    v[11] = e[11] + b11;
    v[20] = e[11] - b11;
    v[12] = e[12] + b12;
    v[19] = e[12] - b12;
    v[13] = e[13] + b13;
    v[18] = e[13] - b13;
    v[14] = e[14] + b14;
    v[17] = e[14] - b14;
    v[15] = e[15] + b15;
    v[16] = e[15] - b15;
}

#[inline(always)]
fn inv_dct4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<4>(c, base, stride);
    inv_dct4_array_x8(&mut v);
    store_1d_x8::<4>(c, base, stride, &v);
}

#[inline(always)]
fn inv_dct8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<8>(c, base, stride);
    inv_dct8_array_x8(&mut v);
    store_1d_x8::<8>(c, base, stride, &v);
}

#[inline(always)]
fn inv_dct16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<16>(c, base, stride);
    inv_dct16_array_x8(&mut v);
    store_1d_x8::<16>(c, base, stride, &v);
}

#[inline(always)]
fn inv_dct32_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<32>(c, base, stride);
    inv_dct32_array_x8(&mut v);
    store_1d_x8::<32>(c, base, stride, &v);
}

#[inline(always)]
fn scale_array_x8<const N: usize>(v: &mut [i32x8; N], scale: i32) {
    for x in v.iter_mut() {
        *x = mulc(*x, scale);
    }
}

fn inv_dst_1d_x8(c: &mut [i32], base: usize, stride: usize, mat: &[i8], n: usize, flip: bool) {
    let zero = i32x8::splat(0);
    let mut sums = [zero; 16];
    let mut mi = 0;
    for sum in sums.iter_mut().take(n) {
        let mut acc = zero;
        for j in 0..n {
            acc += mulc(ldx8(c, base + j * stride), mat[mi] as i32);
            mi += 1;
        }
        *sum = acc;
    }
    if flip {
        for i in 0..n {
            stx8(c, base + (n - 1 - i) * stride, sums[i]);
        }
    } else {
        for i in 0..n {
            stx8(c, base + i * stride, sums[i]);
        }
    }
}

fn inv_adst4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST4_KERNEL, 4, false);
}
fn inv_adst8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST8_KERNEL, 8, false);
}
fn inv_adst16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST16_KERNEL, 16, false);
}
fn inv_flipadst4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &FLIPADST4_KERNEL, 4, false);
}
fn inv_flipadst8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &ADST8_KERNEL, 8, true);
}
fn inv_flipadst16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &FLIPADST16_KERNEL, 16, false);
}
fn inv_ddt8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT8_KERNEL, 8, false);
}
fn inv_ddt16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT16_KERNEL, 16, false);
}
fn inv_flipddt8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT8_KERNEL, 8, true);
}
fn inv_flipddt16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    inv_dst_1d_x8(c, base, stride, &DDT16_KERNEL, 16, true);
}

fn inv_identity4_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<4>(c, base, stride);
    scale_array_x8(&mut v, 128);
    store_1d_x8::<4>(c, base, stride, &v);
}
fn inv_identity8_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<8>(c, base, stride);
    scale_array_x8(&mut v, 181);
    store_1d_x8::<8>(c, base, stride, &v);
}
fn inv_identity16_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<16>(c, base, stride);
    scale_array_x8(&mut v, 256);
    store_1d_x8::<16>(c, base, stride, &v);
}
fn inv_identity32_1d_x8(c: &mut [i32], base: usize, stride: usize) {
    let mut v = load_1d_x8::<32>(c, base, stride);
    scale_array_x8(&mut v, 362);
    store_1d_x8::<32>(c, base, stride, &v);
}

/// SoA-batched counterpart of [`TX1D_FNS`] (same `[tx_size][tx_1d_type]` layout).
pub(crate) static TX1D_FNS_X8: [[Option<Itx1dFnX8>; N_TX_1D_TYPES - 1]; N_TX_SIZES] = {
    const DCT: usize = 0;
    const IDENTITY: usize = 1;
    const ADST: usize = 2;
    const FLIPADST: usize = 3;
    const DDT: usize = 4;
    const FLIPDDT: usize = 5;
    const NONE: Option<Itx1dFnX8> = None;

    let mut t = [[NONE; N_TX_1D_TYPES - 1]; N_TX_SIZES];

    t[0][DCT] = Some(inv_dct4_1d_x8 as Itx1dFnX8);
    t[0][IDENTITY] = Some(inv_identity4_1d_x8);
    t[0][ADST] = Some(inv_adst4_1d_x8);
    t[0][FLIPADST] = Some(inv_flipadst4_1d_x8);

    t[1][DCT] = Some(inv_dct8_1d_x8);
    t[1][IDENTITY] = Some(inv_identity8_1d_x8);
    t[1][ADST] = Some(inv_adst8_1d_x8);
    t[1][FLIPADST] = Some(inv_flipadst8_1d_x8);
    t[1][DDT] = Some(inv_ddt8_1d_x8);
    t[1][FLIPDDT] = Some(inv_flipddt8_1d_x8);

    t[2][DCT] = Some(inv_dct16_1d_x8);
    t[2][IDENTITY] = Some(inv_identity16_1d_x8);
    t[2][ADST] = Some(inv_adst16_1d_x8);
    t[2][FLIPADST] = Some(inv_flipadst16_1d_x8);
    t[2][DDT] = Some(inv_ddt16_1d_x8);
    t[2][FLIPDDT] = Some(inv_flipddt16_1d_x8);

    t[3][DCT] = Some(inv_dct32_1d_x8);
    t[3][IDENTITY] = Some(inv_identity32_1d_x8);

    t[4][DCT] = Some(inv_dct32_1d_x8);

    t
};

pub(crate) fn inv_wht4_1d(c: &mut [i32], stride: usize) {
    let in0 = c[0 * stride];
    let in1 = c[stride];
    let in2 = c[2 * stride];
    let in3 = c[3 * stride];

    let t0 = in0 + in1;
    let t2 = in2 - in3;
    let t4 = (t0 - t2) >> 1;
    let t3 = t4 - in3;
    let t1 = t4 - in1;

    c[0 * stride] = t0 - t3;
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = t2 + t1;
}

pub(crate) fn cctx(u: &mut [i32], v: &mut [i32], angle: &[i16; 3], sz: usize, bitdepth: i32) {
    debug_assert!(sz.is_power_of_two() && (16..=1024).contains(&sz));
    let min = -(1 << (bitdepth + 7));
    let max = (1 << (bitdepth + 7)) - 1;
    let sina = angle[0] as i32;
    let cosa = angle[1] as i32;
    debug_assert!(angle[2] == -angle[0]);
    crate::simd::cctx_row(u, v, sina, cosa, sz, min, max);
}

pub(crate) fn inv_wht_wht_4x4(coeff: &[i32; 16], tmp: &mut [i32; 16]) {
    for y in 0..4 {
        for x in 0..4 {
            tmp[y * 4 + x] = coeff[y + x * 4] >> 3;
        }
        inv_wht4_1d(&mut tmp[y * 4..], 1);
    }
    for x in 0..4 {
        inv_wht4_1d(&mut tmp[x..], 4);
    }
}

// Table excludes Wht (index 6), hence N_TX_1D_TYPES - 1 = 6 columns
pub static TX1D_FNS: [[Option<Itx1dFn>; N_TX_1D_TYPES - 1]; N_TX_SIZES] = {
    const DCT: usize = 0;
    const IDENTITY: usize = 1;
    const ADST: usize = 2;
    const FLIPADST: usize = 3;
    const DDT: usize = 4;
    const FLIPDDT: usize = 5;
    const NONE: Option<Itx1dFn> = None;

    let mut t = [[NONE; N_TX_1D_TYPES - 1]; N_TX_SIZES];

    // TX_4X4
    t[0][DCT] = Some(inv_dct4_1d);
    t[0][IDENTITY] = Some(inv_identity4_1d);
    t[0][ADST] = Some(inv_adst4_1d);
    t[0][FLIPADST] = Some(inv_flipadst4_1d);

    // TX_8X8
    t[1][DCT] = Some(inv_dct8_1d);
    t[1][IDENTITY] = Some(inv_identity8_1d);
    t[1][ADST] = Some(inv_adst8_1d);
    t[1][FLIPADST] = Some(inv_flipadst8_1d);
    t[1][DDT] = Some(inv_ddt8_1d);
    t[1][FLIPDDT] = Some(inv_flipddt8_1d);

    // TX_16X16
    t[2][DCT] = Some(inv_dct16_1d);
    t[2][IDENTITY] = Some(inv_identity16_1d);
    t[2][ADST] = Some(inv_adst16_1d);
    t[2][FLIPADST] = Some(inv_flipadst16_1d);
    t[2][DDT] = Some(inv_ddt16_1d);
    t[2][FLIPDDT] = Some(inv_flipddt16_1d);

    // TX_32X32
    t[3][DCT] = Some(inv_dct32_1d);
    t[3][IDENTITY] = Some(inv_identity32_1d);

    // TX_64X64
    t[4][DCT] = Some(inv_dct32_1d);

    t
};

/// type `BD::Pixel`; the reconstructed value is clipped into `[0, bitdepth_max]`.
pub(crate) fn residual_add<BD: crate::pixel::BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    stride: usize,
    c: &[i32],
    w: usize,
    h: usize,
    rnd: i32,
    shift: i32,
    dpcm_flag: u8,
) {
    match dpcm_flag {
        1 => {
            let mut ci = 0;
            for (c, dst) in c.chunks_exact(w).zip(dst.chunks_exact_mut(stride)).take(h) {
                let mut acc = 0i32;
                for dst in dst[..w].iter_mut() {
                    acc += (c[ci] + rnd) >> shift;
                    let p: i32 = (*dst).into();
                    *dst = bd.pixel_clip(p + acc);
                    ci += 1;
                }
            }
        }
        2 => {
            for x in 0..w {
                let mut acc = 0i32;
                for y in 0..h {
                    acc += (c[y * w + x] + rnd) >> shift;
                    let p = dst[y * stride + x].into();
                    dst[y * stride + x] = bd.pixel_clip(p + acc);
                }
            }
        }
        // dpcm_flag 0 — and any non-1/2 value, which is an invalid combination
        // `switch (dpcm_flag) { default: assert(0); case 0: ... }` falls through
        // from `default` into `case 0`, i.e. the plain non-DPCM residual add.
        _ => {
            for y in 0..h {
                let row = y * stride;
                if row >= dst.len() {
                    break;
                }
                let cw = y * w;
                let d = &mut dst[row..];
                let cr = &c[cw.min(c.len())..];
                let n = w.min(d.len()).min(cr.len());
                crate::simd::residual_add_row(bd, d, cr, n, rnd, shift);
            }
        }
    }
}
