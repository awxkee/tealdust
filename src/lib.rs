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

//! ```no_run
//! use tealdust::{Decoder, Settings, Data, TealdustError};
//!
//! let mut decoder = Decoder::open(&Settings::default()).unwrap();
//!
//! // Feed compressed data
//! let obu_data: Vec<u8> = std::fs::read("input.obu").unwrap();
//! decoder.send_data(Some(Data::wrap(obu_data))).unwrap();
//!
//! // Retrieve decoded pictures
//! loop {
//!     match decoder.get_picture() {
//!         Ok(picture) => { /* process decoded frame */ }
//!         Err(TealdustError::Again) => break, // need more input
//!         Err(TealdustError::Eof) => break,   // end of stream
//!         Err(e) => panic!("decode error: {e}"),
//!     }
//! }
//! ```

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::result_unit_err)]
#![allow(clippy::while_immutable_condition)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::precedence)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::doc_lazy_continuation)]

pub(crate) mod ccso;
pub(crate) mod cdef;
pub(crate) mod cdf;
pub(crate) mod ctx;
pub(crate) mod deblock;
pub(crate) mod decode;
pub(crate) mod dip_tables;
pub(crate) mod env;
pub(crate) mod filmgrain;
pub(crate) mod gdf_tables;
pub(crate) mod getbits;
pub(crate) mod ibp;
pub(crate) mod internal;
pub(crate) mod intops;
pub(crate) mod ipred;
pub(crate) mod ipred_prepare;
pub(crate) mod itx;
pub(crate) mod itx_1d;
pub(crate) mod itx_2d;
pub(crate) mod itx_wht_dispatch;
pub(crate) mod lf_mask;
pub(crate) mod looprestoration;
pub(crate) mod mc;
pub(crate) mod mc_dispatch;
pub(crate) mod msac;
pub(crate) mod mtpool;
pub(crate) mod obu;
pub(crate) mod pal;
pub(crate) mod pixel;
pub(crate) mod quantizer;
pub(crate) mod recon;
pub(crate) mod refmvs;
pub(crate) mod scan;
pub(crate) mod stx;
pub(crate) mod stx_tables;
pub(crate) mod tables;
pub(crate) mod warpmv;
pub(crate) mod wedge;

mod avif;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod cdef_dispatch;
mod cfl_dispatch;
mod data;
mod deblock_dispatch;
mod decode_partition;
mod decoder;
mod error;
mod filter;
mod headers;
mod intra;
mod ipred_dispatch;
mod levels;
#[cfg(target_arch = "aarch64")]
mod neon;
mod picture;
mod rowops_dispatch;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod sse;

pub use avif::*;
pub use data::Data;
pub use decoder::{
    DEFAULT_FRAME_SIZE_LIMIT, DecodeFrameType, Decoder, InloopFilterType, MAX_FRAME_DELAY,
    MAX_THREADS, Settings,
};
pub use decoder::{get_frame_delay, version, version_api};
pub use error::TealdustError;
pub use headers::{ColorPrimaries, ContentLightLevel, MatrixCoefficients, TransferCharacteristics};
pub use headers::{FrameHeader, PixelLayout, SequenceHeader};
pub use picture::{EventFlags, PicAllocator, Picture, PlaneStorage};
