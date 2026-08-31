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
mod ccso;
mod ccso_hd;
mod cdef;
mod cdef_hd;
mod cfl;
mod cfl_hd;
mod deblocking;
mod filmgrain;
mod filter;
mod filter_hd;
mod ipred;
mod ipred_hd;
mod itx;
mod looprestoration;
mod looprestoration_hd;
mod mc;
mod mc_hbd;
mod pal;
mod refmvs;
mod stx;

pub(crate) use ccso::{ccso_add_8bpc_neon, ccso_prep_lut_8bpc_neon};
pub(crate) use ccso_hd::{ccso_add_hbd_neon, ccso_prep_lut_hbd_neon};
pub(crate) use cdef::{
    cdef_filter_block_4x4_8bpc_neon, cdef_filter_block_4x8_8bpc_neon, cdef_filter_block_8bpc_neon,
    cdef_filter_block_8x4_8bpc_neon, cdef_filter_block_8x8_8bpc_neon, cdef_find_dir_8bpc_neon,
    cdef_padding_8bpc_neon,
};
pub(crate) use cdef_hd::{
    cdef_filter_block_4x4_hbd_neon, cdef_filter_block_4x8_hbd_neon, cdef_filter_block_8x4_hbd_neon,
    cdef_filter_block_8x8_hbd_neon, cdef_filter_block_hbd_neon, cdef_find_dir_hbd_neon,
    cdef_padding_hbd_neon,
};
pub(crate) use cfl::{
    cfl_alpha_accum_8bpc_neon, cfl_apply_420_8bpc_neon, cfl_apply_422_8bpc_neon,
    cfl_apply_444_8bpc_neon, cfl_gen_mat_8bpc_neon, cfl_gen_y_row_8bpc_neon,
    cfl_mhccp_pred_8bpc_neon,
};
pub(crate) use cfl_hd::{
    cfl_alpha_accum_hbd_neon, cfl_apply_420_hbd_neon, cfl_apply_422_hbd_neon,
    cfl_apply_444_hbd_neon, cfl_gen_mat_hbd_neon, cfl_gen_y_row_hbd_neon, cfl_mhccp_pred_hbd_neon,
};
pub(crate) use deblocking::{
    deblock_apply_8bpc_neon, deblock_apply_hbd_neon, deblock_h_sb64uv_8bpc_neon,
    deblock_h_sb64uv_hbd_neon, deblock_h_sb64y_8bpc_neon, deblock_h_sb64y_hbd_neon,
    deblock_v_sb64uv_8bpc_neon, deblock_v_sb64uv_hbd_neon, deblock_v_sb64y_8bpc_neon,
    deblock_v_sb64y_hbd_neon, setup_thr_cols_dq_8bpc_neon, setup_thr_cols_seg_8bpc_neon,
    setup_thr_cols_simple_8bpc_neon, setup_thr_rows_dq_8bpc_neon, setup_thr_rows_seg_8bpc_neon,
    setup_thr_rows_simple_8bpc_neon,
};
pub(crate) use filmgrain::{
    blend_top_grain_row_neon, fguv_row_8bpc_neon, fguv_row_hbd_neon, fgy_row_8bpc_neon,
    fgy_row_hbd_neon,
};
pub(crate) use filter::{
    avg_row_8bpc_neon, blend_row_8bpc_neon, cctx_row_i16_neon, cctx_row_neon, dc_add_row_8bpc_neon,
    gdf_add_run_8bpc_neon, gdf_gradient_group_neon, gdf_prep_pair_8bpc_neon, mask_row_8bpc_neon,
    morph_row_8bpc_neon, residual_add_row_8bpc_neon, row_clip_neon, w_avg_row_8bpc_neon,
};
pub(crate) use filter_hd::{
    avg_row_hbd_neon, blend_row_hbd_neon, dc_add_row_hbd_neon, gdf_add_run_hbd_neon,
    gdf_gradient_group_hbd_neon, gdf_prep_pair_hbd_neon, mask_row_hbd_neon, morph_row_hbd_neon,
    residual_add_row_hbd_neon, w_avg_row_hbd_neon,
};
pub(crate) use ipred::*;
pub(crate) use ipred_hd::*;
pub(crate) use itx::*;
pub(crate) use looprestoration::{
    ns_wiener_fir_run_neon, ns_wiener_uv_fir_run_neon, pc_wiener_fir_run_neon,
};
pub(crate) use looprestoration_hd::{
    ns_wiener_fir_run_hbd_neon, ns_wiener_uv_fir_run_hbd_neon, pc_wiener_fir_run_hbd_neon,
};
pub(crate) use mc::*;
pub(crate) use mc_hbd::*;
pub(crate) use pal::{pal_pred_8bpc_neon, pal_pred_hbd_neon};
pub(crate) use refmvs::{splat_mv_neon, splat_warpmv_neon};
pub(crate) use stx::{stxfm4_8bpc_neon, stxfm4_hbd_neon, stxfm8_8bpc_neon, stxfm8_hbd_neon};
