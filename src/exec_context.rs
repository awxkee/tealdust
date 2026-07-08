/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::{
    cdef_dispatch, cfl_dispatch, deblock_dispatch, filter, ipred_dispatch, itx_wht_dispatch,
    mc_dispatch, rowops_dispatch,
};

/// Decoder-local table of already-resolved DSP entry points.
///
/// Resolver functions still use process-wide `OnceLock`s, so CPU feature probing
/// remains centralized and thread-safe. A `Decoder` stores those resolved function
/// pointers in this context once and decode hot paths call the fields directly.
#[derive(Clone)]
pub(crate) struct ExecContext {
    pub(crate) ipred_v: ipred_dispatch::IntraPred8Fn,
    pub(crate) ipred_h: ipred_dispatch::IntraPred8Fn,
    pub(crate) ipred_dc: ipred_dispatch::IntraPred8Fn,
    pub(crate) ipred_dc_top: ipred_dispatch::IntraPred8Fn,
    pub(crate) ipred_dc_left: ipred_dispatch::IntraPred8Fn,
    pub(crate) ipred_dc_128: ipred_dispatch::DcPred128Fn,
    pub(crate) ipred_paeth: ipred_dispatch::SmoothPred8Fn,
    pub(crate) ipred_smooth: ipred_dispatch::SmoothPred8Fn,
    pub(crate) ipred_smooth_v: ipred_dispatch::SmoothPred8Fn,
    pub(crate) ipred_smooth_h: ipred_dispatch::SmoothPred8Fn,
    pub(crate) ipred_z1: ipred_dispatch::Z1Pred8Fn,
    pub(crate) ipred_z2: ipred_dispatch::Z2Pred8Fn,
    pub(crate) ipred_z3: ipred_dispatch::Z1Pred8Fn,
    pub(crate) ipred_dip_8bpc: ipred_dispatch::DipPred8Fn,
    pub(crate) pal_pred_8bpc: ipred_dispatch::PalPred8Fn,
    pub(crate) pal_pred_hbd: ipred_dispatch::PalPredHbdFn,

    pub(crate) ipred_v_hbd: ipred_dispatch::IntraPredHbdFn,
    pub(crate) ipred_h_hbd: ipred_dispatch::IntraPredHbdFn,
    pub(crate) ipred_dc_hbd: ipred_dispatch::IntraPredHbdFn,
    pub(crate) ipred_dc_top_hbd: ipred_dispatch::IntraPredHbdFn,
    pub(crate) ipred_dc_left_hbd: ipred_dispatch::IntraPredHbdFn,
    pub(crate) ipred_dc_128_hbd: ipred_dispatch::DcPred128HbdFn,
    pub(crate) ipred_paeth_hbd: ipred_dispatch::SmoothPredHbdFn,
    pub(crate) ipred_smooth_hbd: ipred_dispatch::SmoothPredHbdFn,
    pub(crate) ipred_smooth_v_hbd: ipred_dispatch::SmoothPredHbdFn,
    pub(crate) ipred_smooth_h_hbd: ipred_dispatch::SmoothPredHbdFn,
    pub(crate) ipred_z1_hbd: ipred_dispatch::Z1PredHbdFn,
    pub(crate) ipred_z2_hbd: ipred_dispatch::Z2PredHbdFn,
    pub(crate) ipred_z3_hbd: ipred_dispatch::Z1PredHbdFn,
    pub(crate) ipred_dip_hbd: ipred_dispatch::DipPredHbdFn,

    pub(crate) cfl_gen_mat_8bpc: cfl_dispatch::CflGenMat8Fn,
    pub(crate) cfl_gen_mat_hbd: cfl_dispatch::CflGenMatHbdFn,
    pub(crate) cfl_alpha_accum_8bpc: cfl_dispatch::CflAlphaAccum8Fn,
    pub(crate) cfl_alpha_accum_hbd: cfl_dispatch::CflAlphaAccumHbdFn,
    pub(crate) cfl_gen_y_row_8bpc: cfl_dispatch::CflGenYRow8Fn,
    pub(crate) cfl_gen_y_row_hbd: cfl_dispatch::CflGenYRowHbdFn,
    pub(crate) cfl_mhccp_pred_8bpc: cfl_dispatch::CflMhccpPred8Fn,
    pub(crate) cfl_mhccp_pred_hbd: cfl_dispatch::CflMhccpPredHbdFn,
    pub(crate) cfl_apply_420_8bpc: cfl_dispatch::CflApplyFn,
    pub(crate) cfl_apply_420_8bpc_filtered: cfl_dispatch::CflApplyFn,
    pub(crate) cfl_apply_422_8bpc: cfl_dispatch::CflApplyFn,
    pub(crate) cfl_apply_444_8bpc: cfl_dispatch::CflApplyFn,
    pub(crate) cfl_apply_420_hbd: cfl_dispatch::CflApplyHbdFn,
    pub(crate) cfl_apply_420_hbd_filtered: cfl_dispatch::CflApplyHbdFn,
    pub(crate) cfl_apply_422_hbd: cfl_dispatch::CflApplyHbdFn,
    pub(crate) cfl_apply_444_hbd: cfl_dispatch::CflApplyHbdFn,

    pub(crate) cdef_dir_8bpc: cdef_dispatch::CdefDir8Fn,
    pub(crate) cdef_dir_hbd: cdef_dispatch::CdefDirHbdFn,
    pub(crate) cdef_padding_8bpc: cdef_dispatch::CdefPadding8Fn,
    pub(crate) cdef_padding_hbd: cdef_dispatch::CdefPaddingHbdFn,
    pub(crate) cdef_filter: cdef_dispatch::CdefFilterFn,
    pub(crate) cdef_filter_hbd: cdef_dispatch::CdefFilterHbdFn,
    pub(crate) cdef_filter_shapes: [cdef_dispatch::CdefFilterShapeFn; 4],
    pub(crate) cdef_filter_hbd_shapes: [cdef_dispatch::CdefFilterHbdShapeFn; 4],

    pub(crate) deblock_apply_8bpc: deblock_dispatch::DeblockApply8bpcFn,
    pub(crate) deblock_apply_hbd: deblock_dispatch::DeblockApplyHbdFn,
    pub(crate) deblock_h_sb64y_8bpc: Option<deblock_dispatch::DeblockSb64Fn>,
    pub(crate) deblock_v_sb64y_8bpc: Option<deblock_dispatch::DeblockSb64Fn>,
    pub(crate) deblock_h_sb64uv_8bpc: Option<deblock_dispatch::DeblockSb64Fn>,
    pub(crate) deblock_v_sb64uv_8bpc: Option<deblock_dispatch::DeblockSb64Fn>,
    pub(crate) deblock_h_sb64y_hbd: Option<deblock_dispatch::DeblockSb64HbdFn>,
    pub(crate) deblock_v_sb64y_hbd: Option<deblock_dispatch::DeblockSb64HbdFn>,
    pub(crate) deblock_h_sb64uv_hbd: Option<deblock_dispatch::DeblockSb64HbdFn>,
    pub(crate) deblock_v_sb64uv_hbd: Option<deblock_dispatch::DeblockSb64HbdFn>,
    pub(crate) setup_thr_cols_seg_8bpc: Option<deblock_dispatch::DeblockSetupColsSeg8bpcFn>,
    pub(crate) setup_thr_rows_seg_8bpc: Option<deblock_dispatch::DeblockSetupRowsSeg8bpcFn>,
    pub(crate) setup_thr_cols_dq_8bpc: Option<deblock_dispatch::DeblockSetupColsDq8bpcFn>,
    pub(crate) setup_thr_rows_dq_8bpc: Option<deblock_dispatch::DeblockSetupRowsDq8bpcFn>,
    pub(crate) setup_thr_cols_simple_8bpc: Option<deblock_dispatch::DeblockSetupSimple8bpcFn>,
    pub(crate) setup_thr_rows_simple_8bpc: Option<deblock_dispatch::DeblockSetupSimple8bpcFn>,

    pub(crate) residual_add: rowops_dispatch::ResidualAddFn,
    pub(crate) dc_add: rowops_dispatch::DcAddFn,
    pub(crate) row_clip: rowops_dispatch::RowClipFn,
    pub(crate) cctx: rowops_dispatch::CctxFn,
    pub(crate) avg: rowops_dispatch::AvgFn,
    pub(crate) w_avg: rowops_dispatch::WAvgFn,
    pub(crate) mask: rowops_dispatch::MaskFn,
    pub(crate) blend: rowops_dispatch::BlendFn,
    pub(crate) morph: rowops_dispatch::MorphFn,
    pub(crate) residual_add_hbd: rowops_dispatch::ResidualAddHbdFn,
    pub(crate) dc_add_hbd: rowops_dispatch::DcAddHbdFn,
    pub(crate) avg_hbd: rowops_dispatch::AvgHbdFn,
    pub(crate) w_avg_hbd: rowops_dispatch::WAvgHbdFn,
    pub(crate) mask_hbd: rowops_dispatch::MaskHbdFn,
    pub(crate) blend_hbd: rowops_dispatch::BlendHbdFn,
    pub(crate) morph_hbd: rowops_dispatch::MorphHbdFn,
    pub(crate) gdf_add: rowops_dispatch::GdfAddFn,
    pub(crate) gdf_add_hbd: rowops_dispatch::GdfAddHbdFn,
    pub(crate) gdf_grad: rowops_dispatch::GdfGradFn,
    pub(crate) gdf_grad_hbd: rowops_dispatch::GdfGradHbdFn,
    pub(crate) gdf_prep_pair_8bpc: rowops_dispatch::GdfPrepPair8bpcFn,
    pub(crate) gdf_prep_pair_hbd: rowops_dispatch::GdfPrepPairHbdFn,
    pub(crate) cctx_i16: rowops_dispatch::CctxI16Fn,

    pub(crate) inv_wht_wht_4x4_8bpc: Option<itx_wht_dispatch::InvWht4x4Fn8bpc>,
    pub(crate) inv_wht_wht_4x4_hbd: Option<itx_wht_dispatch::InvWht4x4FnHbd>,

    pub(crate) put_bilin_hbd: mc_dispatch::PutBilinHbdFn,
    pub(crate) prep_bilin_hbd: mc_dispatch::PrepBilinHbdFn,
    pub(crate) put_8tap_hbd: mc_dispatch::Put8tapHbdFn,
    pub(crate) prep_8tap_hbd: mc_dispatch::Prep8tapHbdFn,
    pub(crate) put_bilin_8bpc: mc_dispatch::PutBilin8bpcFn,
    pub(crate) prep_bilin_8bpc: mc_dispatch::PrepBilin8bpcFn,
    pub(crate) put_8tap_8bpc: mc_dispatch::Put8tap8bpcFn,
    pub(crate) prep_8tap_8bpc: mc_dispatch::Prep8tap8bpcFn,
    pub(crate) warp_8bpc: mc_dispatch::Warp8Fn,
    pub(crate) warp_t_8bpc: mc_dispatch::Warp8tFn,
    pub(crate) warp_hbd: mc_dispatch::WarpHbdFn,
    pub(crate) warp_t_hbd: mc_dispatch::WarpHbdTfn,

    pub(crate) ns_wiener_fir: filter::NsWienerFirFn,
    pub(crate) pc_wiener_fir: filter::PcWienerFirFn,
    pub(crate) ns_wiener_uv_fir: filter::NsWienerUvFirFn,
    pub(crate) ns_wiener_fir_hbd: filter::NsWienerFirHbdFn,
    pub(crate) pc_wiener_fir_hbd: filter::PcWienerFirHbdFn,
    pub(crate) ns_wiener_uv_fir_hbd: filter::NsWienerUvFirHbdFn,
}

impl Default for ExecContext {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ExecContext {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            ipred_v: ipred_dispatch::resolve_ipred_v(),
            ipred_h: ipred_dispatch::resolve_ipred_h(),
            ipred_dc: ipred_dispatch::resolve_ipred_dc(),
            ipred_dc_top: ipred_dispatch::resolve_ipred_dc_top(),
            ipred_dc_left: ipred_dispatch::resolve_ipred_dc_left(),
            ipred_dc_128: ipred_dispatch::resolve_ipred_dc_128(),
            ipred_paeth: ipred_dispatch::resolve_ipred_paeth(),
            ipred_smooth: ipred_dispatch::resolve_ipred_smooth(),
            ipred_smooth_v: ipred_dispatch::resolve_ipred_smooth_v(),
            ipred_smooth_h: ipred_dispatch::resolve_ipred_smooth_h(),
            ipred_z1: ipred_dispatch::resolve_ipred_z1(),
            ipred_z2: ipred_dispatch::resolve_ipred_z2(),
            ipred_z3: ipred_dispatch::resolve_ipred_z3(),
            ipred_dip_8bpc: ipred_dispatch::resolve_ipred_dip_8bpc(),
            pal_pred_8bpc: ipred_dispatch::resolve_pal_pred_8bpc(),
            pal_pred_hbd: ipred_dispatch::resolve_pal_pred_hbd(),
            ipred_v_hbd: ipred_dispatch::resolve_ipred_v_hbd(),
            ipred_h_hbd: ipred_dispatch::resolve_ipred_h_hbd(),
            ipred_dc_hbd: ipred_dispatch::resolve_ipred_dc_hbd(),
            ipred_dc_top_hbd: ipred_dispatch::resolve_ipred_dc_top_hbd(),
            ipred_dc_left_hbd: ipred_dispatch::resolve_ipred_dc_left_hbd(),
            ipred_dc_128_hbd: ipred_dispatch::resolve_ipred_dc_128_hbd(),
            ipred_paeth_hbd: ipred_dispatch::resolve_ipred_paeth_hbd(),
            ipred_smooth_hbd: ipred_dispatch::resolve_ipred_smooth_hbd(),
            ipred_smooth_v_hbd: ipred_dispatch::resolve_ipred_smooth_v_hbd(),
            ipred_smooth_h_hbd: ipred_dispatch::resolve_ipred_smooth_h_hbd(),
            ipred_z1_hbd: ipred_dispatch::resolve_ipred_z1_hbd(),
            ipred_z2_hbd: ipred_dispatch::resolve_ipred_z2_hbd(),
            ipred_z3_hbd: ipred_dispatch::resolve_ipred_z3_hbd(),
            ipred_dip_hbd: ipred_dispatch::resolve_ipred_dip_hbd(),

            cfl_gen_mat_8bpc: cfl_dispatch::resolve_cfl_gen_mat_8bpc(),
            cfl_gen_mat_hbd: cfl_dispatch::resolve_cfl_gen_mat_hbd(),
            cfl_alpha_accum_8bpc: cfl_dispatch::resolve_cfl_alpha_accum_8bpc(),
            cfl_alpha_accum_hbd: cfl_dispatch::resolve_cfl_alpha_accum_hbd(),
            cfl_gen_y_row_8bpc: cfl_dispatch::resolve_cfl_gen_y_row_8bpc(),
            cfl_gen_y_row_hbd: cfl_dispatch::resolve_cfl_gen_y_row_hbd(),
            cfl_mhccp_pred_8bpc: cfl_dispatch::resolve_cfl_mhccp_pred_8bpc(),
            cfl_mhccp_pred_hbd: cfl_dispatch::resolve_cfl_mhccp_pred_hbd(),
            cfl_apply_420_8bpc: cfl_dispatch::resolve_cfl_apply_420(),
            cfl_apply_420_8bpc_filtered: cfl_dispatch::resolve_cfl_apply_420_filtered(),
            cfl_apply_422_8bpc: cfl_dispatch::resolve_cfl_apply_422(),
            cfl_apply_444_8bpc: cfl_dispatch::resolve_cfl_apply_444(),
            cfl_apply_420_hbd: cfl_dispatch::resolve_cfl_apply_420_hbd(),
            cfl_apply_420_hbd_filtered: cfl_dispatch::resolve_cfl_apply_420_hbd_filtered(),
            cfl_apply_422_hbd: cfl_dispatch::resolve_cfl_apply_422_hbd(),
            cfl_apply_444_hbd: cfl_dispatch::resolve_cfl_apply_444_hbd(),

            cdef_dir_8bpc: cdef_dispatch::resolve_cdef_dir_8bpc(),
            cdef_dir_hbd: cdef_dispatch::resolve_cdef_dir_hbd(),
            cdef_padding_8bpc: cdef_dispatch::resolve_cdef_padding_8bpc(),
            cdef_padding_hbd: cdef_dispatch::resolve_cdef_padding_hbd(),
            cdef_filter: cdef_dispatch::resolve_cdef_filter(),
            cdef_filter_hbd: cdef_dispatch::resolve_cdef_filter_hbd(),
            cdef_filter_shapes: *cdef_dispatch::resolve_cdef_filter_shapes(),
            cdef_filter_hbd_shapes: *cdef_dispatch::resolve_cdef_filter_hbd_shapes(),

            deblock_apply_8bpc: deblock_dispatch::resolve_deblock_apply_8bpc(),
            deblock_apply_hbd: deblock_dispatch::resolve_deblock_apply_hbd(),
            deblock_h_sb64y_8bpc: deblock_dispatch::resolve_deblock_h_sb64y_8bpc(),
            deblock_v_sb64y_8bpc: deblock_dispatch::resolve_deblock_v_sb64y_8bpc(),
            deblock_h_sb64uv_8bpc: deblock_dispatch::resolve_deblock_h_sb64uv_8bpc(),
            deblock_v_sb64uv_8bpc: deblock_dispatch::resolve_deblock_v_sb64uv_8bpc(),
            deblock_h_sb64y_hbd: deblock_dispatch::resolve_deblock_h_sb64y_hbd(),
            deblock_v_sb64y_hbd: deblock_dispatch::resolve_deblock_v_sb64y_hbd(),
            deblock_h_sb64uv_hbd: deblock_dispatch::resolve_deblock_h_sb64uv_hbd(),
            deblock_v_sb64uv_hbd: deblock_dispatch::resolve_deblock_v_sb64uv_hbd(),
            setup_thr_cols_seg_8bpc: deblock_dispatch::resolve_setup_thr_cols_seg_8bpc(),
            setup_thr_rows_seg_8bpc: deblock_dispatch::resolve_setup_thr_rows_seg_8bpc(),
            setup_thr_cols_dq_8bpc: deblock_dispatch::resolve_setup_thr_cols_dq_8bpc(),
            setup_thr_rows_dq_8bpc: deblock_dispatch::resolve_setup_thr_rows_dq_8bpc(),
            setup_thr_cols_simple_8bpc: deblock_dispatch::resolve_setup_thr_cols_simple_8bpc(),
            setup_thr_rows_simple_8bpc: deblock_dispatch::resolve_setup_thr_rows_simple_8bpc(),

            residual_add: rowops_dispatch::resolve_residual_add(),
            dc_add: rowops_dispatch::resolve_dc_add(),
            row_clip: rowops_dispatch::resolve_row_clip(),
            cctx: rowops_dispatch::resolve_cctx(),
            avg: rowops_dispatch::resolve_avg(),
            w_avg: rowops_dispatch::resolve_w_avg(),
            mask: rowops_dispatch::resolve_mask(),
            blend: rowops_dispatch::resolve_blend(),
            morph: rowops_dispatch::resolve_morph(),
            residual_add_hbd: rowops_dispatch::resolve_residual_add_hbd(),
            dc_add_hbd: rowops_dispatch::resolve_dc_add_hbd(),
            avg_hbd: rowops_dispatch::resolve_avg_hbd(),
            w_avg_hbd: rowops_dispatch::resolve_w_avg_hbd(),
            mask_hbd: rowops_dispatch::resolve_mask_hbd(),
            blend_hbd: rowops_dispatch::resolve_blend_hbd(),
            morph_hbd: rowops_dispatch::resolve_morph_hbd(),
            gdf_add: rowops_dispatch::resolve_gdf_add(),
            gdf_add_hbd: rowops_dispatch::resolve_gdf_add_hbd(),
            gdf_grad: rowops_dispatch::resolve_gdf_grad(),
            gdf_grad_hbd: rowops_dispatch::resolve_gdf_grad_hbd(),
            gdf_prep_pair_8bpc: rowops_dispatch::resolve_gdf_prep_pair_8bpc(),
            gdf_prep_pair_hbd: rowops_dispatch::resolve_gdf_prep_pair_hbd(),
            cctx_i16: rowops_dispatch::resolve_cctx_i16(),

            inv_wht_wht_4x4_8bpc: itx_wht_dispatch::resolve_inv_wht_wht_4x4_8bpc(),
            inv_wht_wht_4x4_hbd: itx_wht_dispatch::resolve_inv_wht_wht_4x4_hbd(),

            put_bilin_hbd: mc_dispatch::resolve_put_bilin_hbd(),
            prep_bilin_hbd: mc_dispatch::resolve_prep_bilin_hbd(),
            put_8tap_hbd: mc_dispatch::resolve_put_8tap_hbd(),
            prep_8tap_hbd: mc_dispatch::resolve_prep_8tap_hbd(),
            put_bilin_8bpc: mc_dispatch::resolve_put_bilin_8bpc(),
            prep_bilin_8bpc: mc_dispatch::resolve_prep_bilin_8bpc(),
            put_8tap_8bpc: mc_dispatch::resolve_put_8tap_8bpc(),
            prep_8tap_8bpc: mc_dispatch::resolve_prep_8tap_8bpc(),
            warp_8bpc: mc_dispatch::resolve_warp_8bpc(),
            warp_t_8bpc: mc_dispatch::resolve_warp_t_8bpc(),
            warp_hbd: mc_dispatch::resolve_warp_hbd(),
            warp_t_hbd: mc_dispatch::resolve_warp_t_hbd(),

            ns_wiener_fir: filter::ns_wiener_fir_run(),
            pc_wiener_fir: filter::pc_wiener_fir_run(),
            ns_wiener_uv_fir: filter::ns_wiener_uv_fir_run(),
            ns_wiener_fir_hbd: filter::ns_wiener_fir_run_hbd(),
            pc_wiener_fir_hbd: filter::pc_wiener_fir_run_hbd(),
            ns_wiener_uv_fir_hbd: filter::ns_wiener_uv_fir_run_hbd(),
        }
    }

    #[inline]
    fn inter_tmp(scratch: &mut Vec<i16>, len: usize) -> &mut [i16] {
        if scratch.len() < len {
            scratch.resize(len, 0);
        }
        &mut scratch[..len]
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn put_bilin_8bpc_with_scratch(
        &self,
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(scratch, mc_dispatch::inter_bilin_8bpc_tmp_len(w, h, mx, my));
        unsafe { (self.put_bilin_8bpc)(dst, dst_stride, src, src_stride, w, h, mx, my, mid) };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn prep_bilin_8bpc_with_scratch(
        &self,
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u8],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(scratch, mc_dispatch::inter_bilin_8bpc_tmp_len(w, h, mx, my));
        unsafe { (self.prep_bilin_8bpc)(tmp, tmp_stride, src, src_stride, w, h, mx, my, mid) };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn put_8tap_8bpc_with_scratch(
        &self,
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_off: usize,
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        filter: i32,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(
            scratch,
            mc_dispatch::inter_8tap_8bpc_tmp_len(w, h, mx, my, filter),
        );
        unsafe {
            (self.put_8tap_8bpc)(
                dst, dst_stride, src, src_off, src_stride, w, h, mx, my, filter, mid,
            )
        };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn prep_8tap_8bpc_with_scratch(
        &self,
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u8],
        src_off: usize,
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        filter: i32,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(
            scratch,
            mc_dispatch::inter_8tap_8bpc_tmp_len(w, h, mx, my, filter),
        );
        unsafe {
            (self.prep_8tap_8bpc)(
                tmp, tmp_stride, src, src_off, src_stride, w, h, mx, my, filter, mid,
            )
        };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn put_bilin_hbd_with_scratch(
        &self,
        dst: &mut [u16],
        dst_stride: usize,
        src: &[u16],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        bitdepth: u8,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(scratch, mc_dispatch::inter_bilin_hbd_tmp_len(w, h, mx, my));
        unsafe {
            (self.put_bilin_hbd)(
                dst, dst_stride, src, src_stride, w, h, mx, my, bitdepth, mid,
            )
        };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn prep_bilin_hbd_with_scratch(
        &self,
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u16],
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        bitdepth: u8,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(scratch, mc_dispatch::inter_bilin_hbd_tmp_len(w, h, mx, my));
        unsafe {
            (self.prep_bilin_hbd)(
                tmp, tmp_stride, src, src_stride, w, h, mx, my, bitdepth, mid,
            )
        };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn put_8tap_hbd_with_scratch(
        &self,
        dst: &mut [u16],
        dst_stride: usize,
        src: &[u16],
        src_off: usize,
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        filter: i32,
        bitdepth: u8,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(
            scratch,
            mc_dispatch::inter_8tap_hbd_tmp_len(w, h, mx, my, filter),
        );
        unsafe {
            (self.put_8tap_hbd)(
                dst, dst_stride, src, src_off, src_stride, w, h, mx, my, filter, bitdepth, mid,
            )
        };
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn prep_8tap_hbd_with_scratch(
        &self,
        tmp: &mut [i16],
        tmp_stride: usize,
        src: &[u16],
        src_off: usize,
        src_stride: usize,
        w: usize,
        h: usize,
        mx: i32,
        my: i32,
        filter: i32,
        bitdepth: u8,
        scratch: &mut Vec<i16>,
    ) {
        let mid = Self::inter_tmp(
            scratch,
            mc_dispatch::inter_8tap_hbd_tmp_len(w, h, mx, my, filter),
        );
        unsafe {
            (self.prep_8tap_hbd)(
                tmp, tmp_stride, src, src_off, src_stride, w, h, mx, my, filter, bitdepth, mid,
            )
        };
    }

    #[inline(always)]
    pub(crate) unsafe fn call_ipred_8bpc(
        &self,
        m: u8,
        dst: &mut [u8],
        stride: usize,
        edge: &[u8],
        edge_o: usize,
        w: usize,
        h: usize,
        angle: i32,
        max_w: i32,
        max_h: i32,
        ibp_weights: &[[[u8; 16]; 16]; 7],
    ) {
        use crate::levels::*;
        unsafe {
            match m {
                0 => (self.ipred_dc)(dst, stride, edge, edge_o, w, h, angle),
                _ if m == DC_128_PRED => (self.ipred_dc_128)(dst, stride, w, h),
                _ if m == TOP_DC_PRED => {
                    (self.ipred_dc_top)(dst, stride, edge, edge_o, w, h, angle)
                }
                _ if m == LEFT_DC_PRED => {
                    (self.ipred_dc_left)(dst, stride, edge, edge_o, w, h, angle)
                }
                2 => (self.ipred_h)(dst, stride, edge, edge_o, w, h, angle),
                1 => (self.ipred_v)(dst, stride, edge, edge_o, w, h, angle),
                12 => (self.ipred_paeth)(dst, stride, edge, edge_o, w, h),
                9 => (self.ipred_smooth)(dst, stride, edge, edge_o, w, h),
                10 => (self.ipred_smooth_v)(dst, stride, edge, edge_o, w, h),
                11 => (self.ipred_smooth_h)(dst, stride, edge, edge_o, w, h),
                _ if m == Z1_PRED => (self.ipred_z1)(
                    dst,
                    stride,
                    edge,
                    edge_o,
                    w,
                    h,
                    angle,
                    max_w,
                    max_h,
                    ibp_weights,
                ),
                _ if m == Z2_PRED => {
                    (self.ipred_z2)(dst, stride, edge, edge_o, w, h, angle, max_w, max_h)
                }
                _ if m == Z3_PRED => (self.ipred_z3)(
                    dst,
                    stride,
                    edge,
                    edge_o,
                    w,
                    h,
                    angle,
                    max_w,
                    max_h,
                    ibp_weights,
                ),
                _ if m == DIP_PRED => (self.ipred_dip_8bpc)(dst, stride, edge, edge_o, w, h, angle),
                _ => (self.ipred_dc_128)(dst, stride, w, h),
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn call_ipred_hbd(
        &self,
        m: u8,
        bitdepth_max: u16,
        dst: &mut [u16],
        stride: usize,
        edge: &[u16],
        edge_o: usize,
        w: usize,
        h: usize,
        angle: i32,
        max_w: i32,
        max_h: i32,
        ibp_weights: &[[[u8; 16]; 16]; 7],
    ) {
        use crate::levels::*;
        unsafe {
            match m {
                0 => (self.ipred_dc_hbd)(dst, stride, edge, edge_o, w, h, angle, bitdepth_max),
                _ if m == DC_128_PRED => (self.ipred_dc_128_hbd)(dst, stride, w, h, bitdepth_max),
                _ if m == TOP_DC_PRED => {
                    (self.ipred_dc_top_hbd)(dst, stride, edge, edge_o, w, h, angle, bitdepth_max)
                }
                _ if m == LEFT_DC_PRED => {
                    (self.ipred_dc_left_hbd)(dst, stride, edge, edge_o, w, h, angle, bitdepth_max)
                }
                2 => (self.ipred_h_hbd)(dst, stride, edge, edge_o, w, h, angle, bitdepth_max),
                1 => (self.ipred_v_hbd)(dst, stride, edge, edge_o, w, h, angle, bitdepth_max),
                12 => (self.ipred_paeth_hbd)(dst, stride, edge, edge_o, w, h, bitdepth_max),
                9 => (self.ipred_smooth_hbd)(dst, stride, edge, edge_o, w, h, bitdepth_max),
                10 => (self.ipred_smooth_v_hbd)(dst, stride, edge, edge_o, w, h, bitdepth_max),
                11 => (self.ipred_smooth_h_hbd)(dst, stride, edge, edge_o, w, h, bitdepth_max),
                _ if m == Z1_PRED => (self.ipred_z1_hbd)(
                    dst,
                    stride,
                    edge,
                    edge_o,
                    w,
                    h,
                    angle,
                    max_w,
                    max_h,
                    ibp_weights,
                    bitdepth_max,
                ),
                _ if m == Z2_PRED => (self.ipred_z2_hbd)(
                    dst,
                    stride,
                    edge,
                    edge_o,
                    w,
                    h,
                    angle,
                    max_w,
                    max_h,
                    bitdepth_max,
                ),
                _ if m == Z3_PRED => (self.ipred_z3_hbd)(
                    dst,
                    stride,
                    edge,
                    edge_o,
                    w,
                    h,
                    angle,
                    max_w,
                    max_h,
                    ibp_weights,
                    bitdepth_max,
                ),
                _ if m == DIP_PRED => {
                    (self.ipred_dip_hbd)(dst, stride, edge, edge_o, w, h, angle, bitdepth_max)
                }
                _ => (self.ipred_dc_128_hbd)(dst, stride, w, h, bitdepth_max),
            }
        }
    }
}
