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
mod cfl;
mod ipred;
mod itx;
mod looprestoration;

pub(crate) use itx::{
    iadst_dequant_4x4_sse41, iadst_dequant_4x8_sse41, iadst_dequant_4x16_sse41,
    iadst_dequant_8x4_sse41, iadst_dequant_8x8_sse41, iadst_dequant_8x16_sse41,
    iadst_dequant_16x4_sse41, iadst_dequant_16x8_sse41, iadst_dequant_16x16_sse41,
    idct_dequant_4x4_sse41, idct_dequant_4x8_sse41, idct_dequant_4x16_sse41,
    idct_dequant_4x32_sse41, idct_dequant_8x4_sse41, idct_dequant_8x8_sse41,
    idct_dequant_8x16_sse41, idct_dequant_8x32_sse41, idct_dequant_16x4_sse41,
    idct_dequant_16x8_sse41, idct_dequant_16x16_sse41, idct_dequant_16x32_sse41,
    idct_dequant_32x4_sse41, idct_dequant_32x8_sse41, idct_dequant_32x16_sse41,
    idct_dequant_32x32_sse41, idct_dequant_64x64_sse41,
};

pub(crate) use cfl::cfl_apply_420_8bpc_sse41;
pub(crate) use ipred::*;
pub(crate) use looprestoration::{ns_wiener_fir_run_sse41, pc_wiener_fir_run_sse41};
