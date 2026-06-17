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
#![warn(unsafe_op_in_unsafe_fn)]

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
pub(crate) mod lf_mask;
pub(crate) mod looprestoration;
pub(crate) mod mc;
pub(crate) mod mc_neon;
pub(crate) mod msac;
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
mod data;
mod decoder;
mod error;
mod headers;
mod ipred_neon;
mod levels;
mod picture;
mod simd;

pub use avif::*;
pub use data::Data;
pub use decoder::{
    DecodeFrameType, Decoder, InloopFilterType, MAX_FRAME_DELAY, MAX_THREADS, Settings,
};
pub use decoder::{get_frame_delay, version, version_api};
pub use error::TealdustError;
pub use headers::{ColorPrimaries, ContentLightLevel, MatrixCoefficients, TransferCharacteristics};
pub use headers::{FrameHeader, PixelLayout, SequenceHeader};
pub use picture::{EventFlags, PicAllocator, Picture, PlaneStorage};
