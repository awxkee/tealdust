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

pub(crate) const TIP_FRAME: usize = 7;
pub(crate) const INVALID_MV: i32 = 0x200000;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ObuMetaType {
    HdrCll = 1,
    HdrMdcv = 2,
    Scalability = 3,
    ItutT35 = 4,
    Timecode = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum TxfmSize {
    #[default]
    Tx4x4 = 0,
    #[allow(unused)]
    Tx8x8 = 1,
    #[allow(unused)]
    Tx16x16 = 2,
    #[allow(unused)]
    Tx32x32 = 3,
    #[allow(unused)]
    Tx64x64 = 4,
}
pub(crate) const N_TX_SIZES: usize = 5;
pub(crate) const N_RECT_TX_SIZES: usize = 25;
pub(crate) const N_TX_1D_TYPES: usize = 7;
// TxfmType encoding: hor_1d[0:2] | tx_class[3:4] | ver_1d[5:7]
macro_rules! txtp {
    ($hor:expr, $ver:expr, $class:expr) => {
        ($hor) | (($class) << 3) | (($ver) << 5)
    };
}
pub(crate) mod txtp {
    pub(crate) const DCT_DCT: u8 = txtp!(0, 0, 0);
    pub(crate) const ADST_DCT: u8 = txtp!(0, 2, 0);
    pub(crate) const DCT_ADST: u8 = txtp!(2, 0, 0);
    pub(crate) const ADST_ADST: u8 = txtp!(2, 2, 0);
    pub(crate) const DCT_FLIPADST: u8 = txtp!(3, 0, 0);
    pub(crate) const FLIPADST_DCT: u8 = txtp!(0, 3, 0);
    pub(crate) const FLIPADST_FLIPADST: u8 = txtp!(3, 3, 0);
    pub(crate) const FLIPADST_ADST: u8 = txtp!(2, 3, 0);
    pub(crate) const ADST_FLIPADST: u8 = txtp!(3, 2, 0);
    pub(crate) const IDTX: u8 = txtp!(1, 1, 0);
    pub(crate) const IDTX_INV: u8 = txtp!(1, 1, 1);
    pub(crate) const V_DCT: u8 = txtp!(1, 0, 3);
    pub(crate) const H_DCT: u8 = txtp!(0, 1, 2);
    pub(crate) const V_ADST: u8 = txtp!(1, 2, 3);
    pub(crate) const H_ADST: u8 = txtp!(2, 1, 2);
    pub(crate) const V_FLIPADST: u8 = txtp!(1, 3, 3);
    pub(crate) const H_FLIPADST: u8 = txtp!(3, 1, 2);
    pub(crate) const WHT_WHT: u8 = txtp!(6, 6, 0);

    #[inline(always)]
    pub(crate) const fn class(t: u8) -> u8 {
        (t >> 3) & 3
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum IntraPredMode {
    #[default]
    DcPred = 0,
    VertPred = 1,
    HorPred = 2,
    DiagDownLeftPred = 3,
    DiagDownRightPred = 4,
    VertRightPred = 5,
    HorDownPred = 6,
    HorUpPred = 7,
    VertLeftPred = 8,
    SmoothPred = 9,
    SmoothVPred = 10,
    SmoothHPred = 11,
    PaethPred = 12,
}
pub(crate) const CFL_PRED: u8 = 13;
pub(crate) const N_UV_INTRA_PRED_MODES: usize = 14;
pub(crate) const LEFT_DC_PRED: u8 = 3; // = DIAG_DOWN_LEFT_PRED
pub(crate) const TOP_DC_PRED: u8 = 4;
pub(crate) const DC_128_PRED: u8 = 5;
pub(crate) const Z1_PRED: u8 = 6;
pub(crate) const Z2_PRED: u8 = 7;
pub(crate) const Z3_PRED: u8 = 8;
pub(crate) const DIP_PRED: u8 = 13; // = N_INTRA_PRED_MODES

impl IntraPredMode {
    #[inline(always)]
    pub(crate) fn from_raw(val: u8) -> Self {
        match val {
            0 => Self::DcPred,
            1 => Self::VertPred,
            2 => Self::HorPred,
            3 => Self::DiagDownLeftPred,
            4 => Self::DiagDownRightPred,
            5 => Self::VertRightPred,
            6 => Self::HorDownPred,
            7 => Self::HorUpPred,
            8 => Self::VertLeftPred,
            9 => Self::SmoothPred,
            10 => Self::SmoothVPred,
            11 => Self::SmoothHPred,
            12 => Self::PaethPred,
            _ => {
                debug_assert!(false, "invalid IntraPredMode: {val}");
                Self::DcPred
            }
        }
    }
}

pub(crate) const ANGLE_SMOOTH_LEFT_EDGE_FLAG: i32 = 1 << 9;
pub(crate) const ANGLE_SMOOTH_TOP_EDGE_FLAG: i32 = 1 << 10;
pub(crate) const ANGLE_USE_EDGE_FILTER_FLAG: i32 = 1 << 11;
pub(crate) const ANGLE_IBP_FLAG: i32 = 1 << 12;
pub(crate) const ANGLE_MRL_IDX_SHIFT: i32 = 13;
pub(crate) const ANGLE_MRL_IDX_MASK: i32 = 3 << 13;
pub(crate) const ANGLE_MULTI_MRL_FLAG: i32 = 1 << 15;
pub(crate) const ANGLE_HAS_LEFT_FLAG: i32 = 1 << 16;
pub(crate) const ANGLE_HAS_TOP_FLAG: i32 = 1 << 17;
pub(crate) const ANGLE_DIP_FLAG: i32 = 1 << 18;
pub(crate) const ANGLE_IS_LUMA: i32 = 1 << 19;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum InterIntraPredMode {
    DcPred = 0,
    VertPred = 1,
    HorPred = 2,
    SmoothPred = 3,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub(crate) enum BlockPartition {
    Invalid = -1,
    None = 0,
    H = 1,
    V = 2,
    H3 = 3,
    V3 = 4,
    H4A = 5,
    H4B = 6,
    V4A = 7,
    V4B = 8,
    Split = 9,
}
pub(crate) const N_PARTITIONS: usize = 10;

impl BlockPartition {
    #[inline(always)]
    pub(crate) fn from_raw(val: i8) -> Self {
        match val {
            -1 => Self::Invalid,
            0 => Self::None,
            1 => Self::H,
            2 => Self::V,
            3 => Self::H3,
            4 => Self::V3,
            5 => Self::H4A,
            6 => Self::H4B,
            7 => Self::V4A,
            8 => Self::V4B,
            9 => Self::Split,
            _ => {
                debug_assert!(false, "invalid BlockPartition: {val}");
                Self::Invalid
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum TxPartition {
    None = 0,
    Split = 1,
    H = 2,
    V = 3,
    H4 = 4,
    V4 = 5,
    H5 = 6,
    V5 = 7,
}

impl TxPartition {
    #[inline(always)]
    pub(crate) fn from_raw(val: u8) -> Self {
        match val {
            0 => Self::None,
            1 => Self::Split,
            2 => Self::H,
            3 => Self::V,
            4 => Self::H4,
            5 => Self::V4,
            6 => Self::H5,
            7 => Self::V5,
            _ => {
                debug_assert!(false, "invalid TxPartition: {val}");
                Self::None
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub(crate) enum BlockSize {
    #[default]
    Invalid = -1,
    Bs256x256 = 0,
    Bs256x128 = 1,
    Bs128x256 = 2,
    Bs128x128 = 3,
    Bs128x64 = 4,
    Bs64x128 = 5,
    Bs64x64 = 6,
    Bs64x32 = 7,
    Bs64x16 = 8,
    Bs64x8 = 9,
    Bs64x4 = 10,
    Bs32x64 = 11,
    Bs32x32 = 12,
    Bs32x16 = 13,
    Bs32x8 = 14,
    Bs32x4 = 15,
    Bs16x64 = 16,
    Bs16x32 = 17,
    Bs16x16 = 18,
    Bs16x8 = 19,
    Bs16x4 = 20,
    Bs8x64 = 21,
    Bs8x32 = 22,
    Bs8x16 = 23,
    Bs8x8 = 24,
    Bs8x4 = 25,
    Bs4x64 = 26,
    Bs4x32 = 27,
    Bs4x16 = 28,
    Bs4x8 = 29,
    Bs4x4 = 30,
}
pub(crate) const N_BS_SIZES: usize = 31;

impl BlockSize {
    #[inline(always)]
    pub(crate) fn from_raw(val: i8) -> Self {
        match val {
            -1 => Self::Invalid,
            0 => Self::Bs256x256,
            1 => Self::Bs256x128,
            2 => Self::Bs128x256,
            3 => Self::Bs128x128,
            4 => Self::Bs128x64,
            5 => Self::Bs64x128,
            6 => Self::Bs64x64,
            7 => Self::Bs64x32,
            8 => Self::Bs64x16,
            9 => Self::Bs64x8,
            10 => Self::Bs64x4,
            11 => Self::Bs32x64,
            12 => Self::Bs32x32,
            13 => Self::Bs32x16,
            14 => Self::Bs32x8,
            15 => Self::Bs32x4,
            16 => Self::Bs16x64,
            17 => Self::Bs16x32,
            18 => Self::Bs16x16,
            19 => Self::Bs16x8,
            20 => Self::Bs16x4,
            21 => Self::Bs8x64,
            22 => Self::Bs8x32,
            23 => Self::Bs8x16,
            24 => Self::Bs8x8,
            25 => Self::Bs8x4,
            26 => Self::Bs4x64,
            27 => Self::Bs4x32,
            28 => Self::Bs4x16,
            29 => Self::Bs4x8,
            30 => Self::Bs4x4,
            _ => {
                debug_assert!(false, "invalid BlockSize: {val}");
                Self::Invalid
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum InterPredMode {
    NearMv = 13,
    GlobalMv = 14,
    NewMv = 15,
    WarpMv = 16,
    WarpNewMv = 17,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CompInterPredMode {
    NearMvNearMv = 18,
    NearMvNewMv = 19,
    NewMvNearMv = 20,
    GlobalMvGlobalMv = 21,
    NewMvNewMv = 22,
    JointNewMv = 23,
    OpflNearMvNearMv = 24,
    OpflNearMvNewMv = 25,
    OpflNewMvNearMv = 26,
    OpflNewMvNewMv = 27,
    OpflJointNewMv = 28,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum CompInterType {
    #[default]
    None = 0,
    Avg = 1,
    Wedge = 2,
    Seg = 3,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum MotionMode {
    #[default]
    Translation = 0,
    InterIntra = 1,
    WarpCausal = 2,
    WarpDelta = 3,
    WarpExtend = 4,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum CflType {
    #[default]
    Explicit = 0,
    Implicit = 1,
    Mhccp = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum CflMhDir {
    #[default]
    Center = 0,
    Top = 1,
    Left = 2,
    All = 3,
}

impl CflMhDir {
    #[inline(always)]
    pub(crate) fn from_raw(val: u8) -> Self {
        match val {
            0 => Self::Center,
            1 => Self::Top,
            2 => Self::Left,
            3 => Self::All,
            _ => {
                debug_assert!(false, "invalid CflMhDir: {val}");
                Self::Center
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(8))]
pub(crate) struct Mv {
    pub(crate) c: MvXY,
}

impl Mv {
    #[inline(always)]
    pub(crate) fn xy(self) -> MvXY {
        self.c
    }

    #[inline(always)]
    pub(crate) fn y(self) -> i32 {
        self.xy().y
    }

    #[inline(always)]
    pub(crate) fn x(self) -> i32 {
        self.xy().x
    }

    #[inline(always)]
    pub(crate) fn bits(self) -> u64 {
        u64::from_ne_bytes([
            self.c.y.to_ne_bytes()[0],
            self.c.y.to_ne_bytes()[1],
            self.c.y.to_ne_bytes()[2],
            self.c.y.to_ne_bytes()[3],
            self.c.x.to_ne_bytes()[0],
            self.c.x.to_ne_bytes()[1],
            self.c.x.to_ne_bytes()[2],
            self.c.x.to_ne_bytes()[3],
        ])
    }

    #[inline(always)]
    pub(crate) fn from_xy(y: i32, x: i32) -> Self {
        Self { c: MvXY { y, x } }
    }

    #[inline(always)]
    pub(crate) fn from_bits(n: u64) -> Self {
        let b = n.to_ne_bytes();
        Self {
            c: MvXY {
                y: i32::from_ne_bytes([b[0], b[1], b[2], b[3]]),
                x: i32::from_ne_bytes([b[4], b[5], b[6], b[7]]),
            },
        }
    }

    #[inline(always)]
    pub(crate) fn set_y(&mut self, y: i32) {
        let x = self.x();
        *self = Self::from_xy(y, x);
    }

    #[inline(always)]
    pub(crate) fn set_x(&mut self, x: i32) {
        let y = self.y();
        *self = Self::from_xy(y, x);
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub(crate) struct MvXY {
    pub(crate) y: i32,
    pub(crate) x: i32,
}

impl std::fmt::Debug for Mv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.xy();
        write!(f, "Mv({}, {})", c.y, c.x)
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(2))]
pub(crate) struct RefPair {
    pub(crate) r: [i8; 2],
}

impl RefPair {
    #[inline(always)]
    pub(crate) fn refs(self) -> [i8; 2] {
        self.r
    }

    #[inline(always)]
    pub(crate) fn r0(self) -> i8 {
        self.refs()[0]
    }

    #[inline(always)]
    pub(crate) fn r1(self) -> i8 {
        self.refs()[1]
    }

    #[inline(always)]
    pub(crate) fn ref_at(self, idx: usize) -> i8 {
        self.refs()[idx]
    }

    #[inline(always)]
    pub(crate) fn pair(self) -> i16 {
        i16::from_ne_bytes([self.r[0] as u8, self.r[1] as u8])
    }

    #[inline(always)]
    pub(crate) fn from_refs(r0: i8, r1: i8) -> Self {
        Self { r: [r0, r1] }
    }

    #[inline(always)]
    pub(crate) fn from_pair(pair: i16) -> Self {
        {
            let b = pair.to_ne_bytes();
            Self {
                r: [b[0] as i8, b[1] as i8],
            }
        }
    }
}

impl std::fmt::Debug for RefPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.refs();
        write!(f, "RefPair({}, {})", r[0], r[1])
    }
}

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub(crate) struct IsSm {
    pub(crate) a: i32,
    pub(crate) l: i32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct CflAlphaOrMhDir {
    pub(crate) cfl_alpha: [i8; 2],
}

impl CflAlphaOrMhDir {
    #[inline(always)]
    pub(crate) fn alpha(self) -> [i8; 2] {
        self.cfl_alpha
    }

    #[inline(always)]
    pub(crate) fn mh_dir(self) -> u8 {
        self.cfl_alpha[0] as u8
    }

    #[inline(always)]
    pub(crate) fn set_alpha(&mut self, alpha: [i8; 2]) {
        self.cfl_alpha = alpha;
    }

    #[inline(always)]
    pub(crate) fn set_alpha_at(&mut self, idx: usize, alpha: i8) {
        self.cfl_alpha[idx] = alpha;
    }

    #[inline(always)]
    pub(crate) fn set_mh_dir(&mut self, dir: u8) {
        self.cfl_alpha[0] = dir as i8;
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Av2BlockIntra {
    pub(crate) intrabc_mv: Mv,
    pub(crate) dpcm: [u8; 2],
    pub(crate) y_mode: u8,
    pub(crate) mrl_index: u8,
    pub(crate) multi_mrl: u8,
    pub(crate) dip: u8,
    pub(crate) morph_pred: u8,
    pub(crate) is_refmv: u8,
    pub(crate) is_qpel: u8,
    pub(crate) uv_mode: u8,
    pub(crate) pal_sz: u8,
    pub(crate) y_angle: i8,
    pub(crate) uv_angle: i8,
    pub(crate) cfl_type: i8,
    pub(crate) cfl: CflAlphaOrMhDir,
    pub(crate) is_sm: [IsSm; 2],
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Av2BlockInter {
    pub(crate) mv: [Mv; 2],
    pub(crate) wedge_idx: i8,
    pub(crate) wedge_sign: i8,
    pub(crate) mask_sign: u8,
    pub(crate) interintra_mode: u8,
    pub(crate) matrix: [i8; 4],
    pub(crate) drl_idx: [u8; 2],
    pub(crate) warp_ref_idx: u8,
    pub(crate) warpmv_with_mvd: u8,
    pub(crate) comp_type: u8,
    pub(crate) inter_mode: u8,
    pub(crate) motion_mode: u8,
    pub(crate) warp_ii: u8,
    pub(crate) cwp_idx: i8,
    pub(crate) mv_prec: i8,
    pub(crate) amvd: i8,
    pub(crate) bawp: [u8; 2],
    pub(crate) filter: u8,
    pub(crate) refine_mv: u8,
    pub(crate) mtxbak: [i32; 6],
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Av2BlockData {
    pub(crate) intra: Av2BlockIntra,
    pub(crate) inter: Av2BlockInter,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Av2Block {
    pub(crate) bs: i8,
    pub(crate) cbs: i8,
    pub(crate) is_intra: u8,
    pub(crate) intrabc: u8,
    pub(crate) seg_id: u8,
    pub(crate) skip_mode: u8,
    pub(crate) skip_txfm: u8,
    pub(crate) tx_part: u8,
    pub(crate) fsc: u8,
    pub(crate) tx_size_ll: u8,
    pub(crate) ref_pair: RefPair,
    pub(crate) data: Av2BlockData,
}

impl Av2Block {
    #[inline(always)]
    pub(crate) fn intra_data(&self) -> &Av2BlockIntra {
        &self.data.intra
    }

    #[inline(always)]
    pub(crate) fn inter_data(&self) -> &Av2BlockInter {
        &self.data.inter
    }

    #[inline(always)]
    pub(crate) fn intra_data_mut(&mut self) -> &mut Av2BlockIntra {
        &mut self.data.intra
    }

    #[inline(always)]
    pub(crate) fn inter_data_mut(&mut self) -> &mut Av2BlockInter {
        &mut self.data.inter
    }
}
