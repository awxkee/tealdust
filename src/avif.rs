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

//! # AVIF / AV2 File Format Parser and Decoder
//!
//! This module parses AVIF image files (ISO/IEC 23000-22) — an ISOBMFF container
//! (ISO 14496-12) that wraps AV2-compressed image samples — and feeds the extracted
//! bitstream into [`Decoder`] to produce decoded [`Picture`]s.
//!
//! ## Container structure
//!
//! ```text
//! AVIF file
//! ├── ftyp  — file-type box (brand = 'avif' or 'avis')
//! ├── meta  — metadata container (FullBox)
//! │   ├── hdlr  — handler = 'pict'
//! │   ├── iloc  — item location (maps item_id → byte range in mdat)
//! │   ├── iinf  — item info (maps item_id → type 'av01'/'av02'/'grid')
//! │   ├── iprp  — item properties
//! │   │   ├── ipco  — property container
//! │   │   │   ├── ispe  — image spatial extents (width / height)
//! │   │   │   ├── av1C / av2C — codec configuration
//! │   │   │   ├── colr  — color information
//! │   │   │   ├── pixi  — pixel information (bits per channel)
//! │   │   │   ├── pasp  — pixel aspect ratio
//! │   │   │   ├── irot  — image rotation (0 / 90 / 180 / 270° anti-clockwise)
//! │   │   │   ├── imir  — image mirror (horizontal-axis / vertical-axis)
//! │   │   │   └── clap  — clean aperture (visible crop rectangle)
//! │   │   └── ipma  — item property association
//! │   └── iref  — item references (dimg / thmb / cdsc)
//! └── mdat  — media data (raw AV2 OBU samples, one per item)
//! ```
//!

use crate::data::Data;
use crate::decoder::{Decoder, Settings};
use crate::error::TealdustError;
use crate::getbits::GetBits;
use crate::headers::{ContentLightLevel, ObuType, PixelLayout};
use crate::levels::ObuMetaType;
use crate::picture::Picture;
use crate::{ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Default number of items accepted from an `iinf` box.
pub const DEFAULT_AVIF_MAX_ITEMS: u32 = 1024;
/// Default number of property entries accepted from an `ipco` box.
pub const DEFAULT_AVIF_MAX_IPCO_PROPS: usize = 256;
/// Default number of `ipma` entry associations per item.
pub const DEFAULT_AVIF_MAX_IPMA_ENTRIES: u32 = 4096;
/// Default extents per `iloc` item.
pub const DEFAULT_AVIF_MAX_EXTENTS_PER_ITEM: u16 = 32;
/// Default total OBU bytes assembled from `iloc` extents before feeding the
/// decoder.
pub const DEFAULT_AVIF_MAX_OBU_BYTES: usize = 64 * 1024 * 1024; // 64 MiB
/// Default image dimension accepted from `ispe`.
pub const DEFAULT_AVIF_MAX_IMAGE_DIMENSION: u32 = 65536;
/// Default allowed `iloc` item count (v0/v1 = u16, v2 = u32; cap both).
pub const DEFAULT_AVIF_MAX_ILOC_ITEMS: u32 = 1024;
/// Default number of `ftyp` compatible brands scanned (prevents O(n) on junk).
pub const DEFAULT_AVIF_MAX_COMPAT_BRANDS: usize = 64;
/// Default byte length of an `auxC` URN string accepted from the bitstream.
pub const DEFAULT_AVIF_MAX_AUXC_URN_LEN: usize = 512;
/// Default number of item references consumed from one `iref` type-ref box.
pub const DEFAULT_AVIF_MAX_IREF_REFS: u16 = 256;

/// High-level AVIF parser/decoder configuration.
///
/// This wraps the AV2 [`Settings`] used internally by [`Decoder`] and the
/// container-level caps that protect AVIF box parsing and OBU assembly. The
/// default is suitable for untrusted still images: it keeps the existing parser
/// limits, enables reconstruction, and uses the host's available parallelism.
/// Set an individual limit to `0` to disable that specific container cap.
#[derive(Debug, Clone)]
pub struct AvifSettings {
    /// Settings passed to the underlying AV2 decoder.
    ///
    /// [`Settings::run_decode`] is forced to `true` while decoding AVIF because
    /// the high-level API always returns pixels. Other fields, including
    /// `n_threads`, `frame_size_limit`, filters, strictness, and grain policy are
    /// passed through as provided.
    pub decoder_settings: Settings,
    /// Maximum total OBU bytes assembled from codec-config OBUs plus item extents.
    pub max_obu_bytes: usize,
    /// Maximum image width or height accepted from `ispe`.
    pub max_image_dimension: u32,
    /// Maximum number of items accepted from an `iinf` box.
    pub max_items: u32,
    /// Maximum number of property entries accepted from an `ipco` box.
    pub max_ipco_props: usize,
    /// Maximum number of `ipma` entries accepted.
    pub max_ipma_entries: u32,
    /// Maximum number of extents accepted for a single `iloc` item.
    pub max_extents_per_item: u16,
    /// Maximum number of items accepted from an `iloc` box.
    pub max_iloc_items: u32,
    /// Maximum number of compatible brands scanned in `ftyp`.
    pub max_compat_brands: usize,
    /// Maximum byte length of an `auxC` URN.
    pub max_auxc_urn_len: usize,
    /// Maximum number of references consumed from one `iref` type-ref box.
    pub max_iref_refs: u16,
}

impl Default for AvifSettings {
    fn default() -> Self {
        let mut decoder_settings = Settings::default();
        decoder_settings.n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        decoder_settings.run_decode = true;

        Self {
            decoder_settings,
            max_obu_bytes: DEFAULT_AVIF_MAX_OBU_BYTES,
            max_image_dimension: DEFAULT_AVIF_MAX_IMAGE_DIMENSION,
            max_items: DEFAULT_AVIF_MAX_ITEMS,
            max_ipco_props: DEFAULT_AVIF_MAX_IPCO_PROPS,
            max_ipma_entries: DEFAULT_AVIF_MAX_IPMA_ENTRIES,
            max_extents_per_item: DEFAULT_AVIF_MAX_EXTENTS_PER_ITEM,
            max_iloc_items: DEFAULT_AVIF_MAX_ILOC_ITEMS,
            max_compat_brands: DEFAULT_AVIF_MAX_COMPAT_BRANDS,
            max_auxc_urn_len: DEFAULT_AVIF_MAX_AUXC_URN_LEN,
            max_iref_refs: DEFAULT_AVIF_MAX_IREF_REFS,
        }
    }
}

/// Errors that can occur when parsing or decoding an AVIF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvifError {
    /// The file is too short to contain a valid AVIF container.
    TooShort,
    /// The `ftyp` box is missing or does not carry an AVIF-compatible brand.
    NotAvif,
    /// A required metadata box (`meta`, `iloc`, `iinf`, …) is missing.
    MissingBox(&'static str),
    /// A box header or field value is out of range or otherwise invalid.
    InvalidBox,
    /// The primary item is not a supported codec type.
    UnsupportedCodec,
    /// Grid (AVIF sequence / tiled image) decoding is not yet supported.
    GridNotSupported,
    /// An `iloc` extent references bytes outside the file.
    ExtentOutOfBounds,
    /// The underlying AV2 bitstream decoder returned an error.
    DecodeError(TealdustError),
    /// The `av2C` / `av1C` codec-config box is present but malformed.
    InvalidCodecConfig,
    /// An attacker-controlled count or size exceeded a safety limit.
    LimitExceeded,
    /// A fallible heap allocation failed while parsing/container-copying.
    OutOfMemory,
}

impl fmt::Display for AvifError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => write!(f, "file too short"),
            Self::NotAvif => write!(f, "not an AVIF file (missing avif/avis brand)"),
            Self::MissingBox(b) => write!(f, "required box '{b}' not found"),
            Self::InvalidBox => write!(f, "malformed ISOBMFF box"),
            Self::UnsupportedCodec => write!(f, "unsupported item codec type"),
            Self::GridNotSupported => write!(f, "grid/sequence AVIF items not yet supported"),
            Self::ExtentOutOfBounds => write!(f, "iloc extent references bytes outside the file"),
            Self::DecodeError(e) => write!(f, "AV2 decode error: {e}"),
            Self::InvalidCodecConfig => write!(f, "invalid av2C/av1C codec config box"),
            Self::LimitExceeded => write!(f, "parser limit exceeded"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for AvifError {}

impl From<TealdustError> for AvifError {
    fn from(e: TealdustError) -> Self {
        Self::DecodeError(e)
    }
}

type Result<T> = std::result::Result<T, AvifError>;

/// Color information extracted from the `colr` item property.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColorInfo {
    /// Color primaries (ISO 23091-2 / H.273).
    pub color_primaries: ColorPrimaries,
    /// Transfer characteristics.
    pub transfer_characteristics: TransferCharacteristics,
    /// Matrix coefficients.
    pub matrix_coefficients: MatrixCoefficients,
    /// Full-range flag (`0` = limited, `1` = full).
    pub full_range: bool,
}

/// The standard URN identifying an alpha auxiliary image item.
///
/// Defined in HEIF (ISO 23008-12) §6.10.3 and referenced by the AVIF spec.
pub const ALPHA_AUXILIARY_URN: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";

/// Identifies the purpose of an auxiliary image item, parsed from the `auxC`
/// item property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryType {
    /// The item is an alpha (transparency) plane.
    Alpha,
    /// The item is a depth map.
    Depth,
    /// Some other auxiliary type (URN stored for inspection).
    Other(String),
}

impl AuxiliaryType {
    /// Parse from a null-terminated URN byte string (the `auxC` payload after
    /// the FullBox 4-byte header).
    fn from_urn_bytes(bytes: &[u8]) -> Self {
        // Trim a trailing NUL if present.
        let s = bytes
            .iter()
            .position(|&b| b == 0)
            .map_or(bytes, |i| &bytes[..i]);
        match s {
            b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha" => Self::Alpha,
            b"urn:mpeg:mpegB:cicp:systems:auxiliary:depth" => Self::Depth,
            _ => Self::Other(String::from_utf8_lossy(s).into_owned()),
        }
    }

    /// Returns `true` if this auxiliary item carries alpha transparency data.
    #[inline]
    pub fn is_alpha(&self) -> bool {
        *self == Self::Alpha
    }
}

/// Parsed `av2C` (or `av1C`) codec-configuration record.
///
/// The `av2C` box stores a sequence of OBUs (Sequence Header + Metadata) that
/// precede every sample.  We hold the raw bytes and feed them to the decoder
/// before the sample OBUs.
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Raw `configOBUs` bytes (zero or more OBUs).
    pub config_obus: Vec<u8>,
    /// High bit-depth flag from the config record.
    pub high_bitdepth: bool,
    /// Twelve-bit flag.
    pub twelve_bit: bool,
    /// Monochrome flag.
    pub monochrome: bool,
    /// Chroma subsampling: x.
    pub chroma_subsampling_x: u8,
    /// Chroma subsampling: y.
    pub chroma_subsampling_y: u8,
    /// Chroma sample position.
    pub chroma_sample_position: u8,
}

impl CodecConfig {
    /// Derive the [`PixelLayout`] that the config record implies.
    pub fn pixel_layout(&self) -> PixelLayout {
        if self.monochrome {
            PixelLayout::I400
        } else if self.chroma_subsampling_x != 0 && self.chroma_subsampling_y != 0 {
            PixelLayout::I420
        } else if self.chroma_subsampling_x != 0 {
            PixelLayout::I422
        } else {
            PixelLayout::I444
        }
    }

    /// Bits per component implied by the config record.
    pub fn bits_per_component(&self) -> u8 {
        if self.twelve_bit {
            12
        } else if self.high_bitdepth {
            10
        } else {
            8
        }
    }
}

/// Image dimensions from the `ispe` item property.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpatialExtents {
    /// Image width in pixels (bounded by [`AvifSettings::max_image_dimension`]).
    pub width: u32,
    /// Image height in pixels (bounded by [`AvifSettings::max_image_dimension`]).
    pub height: u32,
}

/// Pixel-aspect-ratio from the `pasp` item property.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelAspectRatio {
    pub h_spacing: u32,
    pub v_spacing: u32,
}

/// Display orientation, using the eight EXIF orientation values. HEIF expresses
/// orientation with a rotation property (`irot`, anticlockwise multiples of 90°)
/// and a mirror property (`imir`); the diagonal EXIF values map to a mirror
/// followed by a rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    /// 1 — upright, no transform.
    #[default]
    Normal,
    /// 2 — mirrored horizontally.
    FlipH,
    /// 3 — rotated 180°.
    Rotate180,
    /// 4 — mirrored vertically.
    FlipV,
    /// 5 — transpose (mirror H then rotate 90° clockwise).
    Transpose,
    /// 6 — rotated 90° clockwise.
    Rotate90,
    /// 7 — transverse (mirror H then rotate 90° anticlockwise).
    Transverse,
    /// 8 — rotated 90° anticlockwise.
    Rotate270,
}

impl Orientation {
    /// Map a raw EXIF Orientation value (1..=8) to an [`Orientation`]; anything
    /// out of range is treated as `Normal`.
    pub fn from_exif(v: u16) -> Self {
        match v {
            2 => Orientation::FlipH,
            3 => Orientation::Rotate180,
            4 => Orientation::FlipV,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => Orientation::Normal,
        }
    }

    /// True when no orientation transform is needed (so neither `irot` nor
    /// `imir` is written).
    pub fn is_identity(self) -> bool {
        self.irot_steps() == 0 && self.imir_axis().is_none()
    }

    /// The `imir` axis when a mirror is part of the transform: `Some(false)` for
    /// a vertical mirroring axis (left-right flip), `Some(true)` for a
    /// horizontal mirroring axis (top-bottom flip), or `None` for no mirror.
    /// (HEIF `imir`: `axis == 0` mirrors about a vertical axis.)
    pub(crate) fn imir_axis(self) -> Option<bool> {
        match self {
            Orientation::FlipH | Orientation::Transpose | Orientation::Transverse => Some(false),
            Orientation::FlipV => Some(true),
            _ => None,
        }
    }

    /// `irot` rotation in anticlockwise 90° steps (0..=3).
    pub(crate) fn irot_steps(self) -> u8 {
        match self {
            Orientation::Normal | Orientation::FlipH | Orientation::FlipV => 0,
            Orientation::Rotate180 => 2,
            Orientation::Rotate90 => 3,
            Orientation::Rotate270 => 1,
            Orientation::Transpose => 3,
            Orientation::Transverse => 1,
        }
    }

    /// Reconstruct an [`Orientation`] from the `irot` step count (0..=3) and the
    /// `imir` axis (`None`, `Some(false)` = vertical axis / left-right flip,
    /// `Some(true)` = horizontal axis / top-bottom flip). Inverse of the
    /// `irot_steps` / `imir_axis` pair this crate writes.
    pub(crate) fn from_irot_imir(steps: u8, axis: Option<bool>) -> Self {
        match (steps & 3, axis) {
            (0, None) => Orientation::Normal,
            (2, None) => Orientation::Rotate180,
            (3, None) => Orientation::Rotate90,
            (1, None) => Orientation::Rotate270,
            (0, Some(false)) => Orientation::FlipH,
            (0, Some(true)) => Orientation::FlipV,
            (3, Some(false)) => Orientation::Transpose,
            (1, Some(false)) => Orientation::Transverse,
            _ => Orientation::Normal,
        }
    }
}

/// A rational number as a (numerator, denominator) pair.
///
/// Denominators come directly from the bitstream and **must be checked for
/// zero before dividing**.  The parser validates them at parse time and
/// returns [`AvifError::InvalidBox`] for zero denominators.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rational {
    pub numerator: i32,
    pub denominator: i32,
}

impl Rational {
    /// Evaluate to an `f64`.  Panics if `denominator == 0`.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

/// Clean aperture from the `clap` item property (ISO 14496-12 §12.1.4).
///
/// Defines the visible sub-rectangle of the coded image in *sample* coordinates
/// relative to the image centre.  All four fields are rational numbers so that
/// an integer pixel grid can be addressed exactly even after sub-sample offsets.
///
/// The visible rectangle can be computed as:
///
/// ```text
/// x_centre = (coded_width  - 1) / 2  +  horiz_off
/// y_centre = (coded_height - 1) / 2  +  vert_off
///
/// left   = x_centre - (clean_width  - 1) / 2
/// top    = y_centre - (clean_height - 1) / 2
/// right  = left + clean_width
/// bottom = top  + clean_height
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CleanAperture {
    /// Clean aperture width as a rational (numerator / denominator > 0).
    pub width: Rational,
    /// Clean aperture height as a rational.
    pub height: Rational,
    /// Horizontal offset of the aperture centre from the image centre.
    pub horiz_off: Rational,
    /// Vertical offset of the aperture centre from the image centre.
    pub vert_off: Rational,
}

impl CleanAperture {
    /// Compute the integer pixel crop rectangle `(x, y, w, h)` in the coded
    /// image for the given `coded_width × coded_height`, rounding to the
    /// nearest pixel.  Returns `None` if any denominator is zero or if the
    /// resulting rectangle falls outside `[0, coded_width) × [0, coded_height)`.
    pub fn to_crop_rect(self, coded_width: u32, coded_height: u32) -> Option<(u32, u32, u32, u32)> {
        let cw = coded_width as f64;
        let ch = coded_height as f64;

        let clean_w = self.width.to_f64();
        let clean_h = self.height.to_f64();
        let h_off = self.horiz_off.to_f64();
        let v_off = self.vert_off.to_f64();

        if clean_w <= 0.0 || clean_h <= 0.0 {
            return None;
        }

        let x_centre = (cw - 1.0) / 2.0 + h_off;
        let y_centre = (ch - 1.0) / 2.0 + v_off;
        let left = x_centre - (clean_w - 1.0) / 2.0;
        let top = y_centre - (clean_h - 1.0) / 2.0;

        let x = left.round() as i64;
        let y = top.round() as i64;
        let w = clean_w.round() as i64;
        let h = clean_h.round() as i64;

        if x < 0 || y < 0 || w <= 0 || h <= 0 {
            return None;
        }
        if x + w > coded_width as i64 {
            return None;
        }
        if y + h > coded_height as i64 {
            return None;
        }

        Some((x as u32, y as u32, w as u32, h as u32))
    }
}

/// The four-character-code type of an `iinf` item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemType {
    /// AV1 image item (AVIF using AV1 codec).
    Av01,
    /// AV2 image item (AVIF using AV2 codec).
    Av02,
    /// Grid image (tiled/sequence composite).
    Grid,
    /// Thumbnail.
    Thumb,
    /// ICC color profile embedded as an item.
    Prof,
    /// Any other item type (ignored by the decoder).
    Other([u8; 4]),
}

impl ItemType {
    fn from_fourcc(b: [u8; 4]) -> Self {
        match &b {
            b"av01" => Self::Av01,
            b"av02" => Self::Av02,
            b"grid" => Self::Grid,
            b"thmb" => Self::Thumb,
            b"prof" => Self::Prof,
            _ => Self::Other(b),
        }
    }

    /// Returns `true` if this item holds an AV1 or AV2 compressed image.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Av01 | Self::Av02)
    }
}

/// Describes a single item inside the AVIF `meta` box.
#[derive(Debug, Clone)]
pub struct AvifItem {
    pub item_id: u16,
    pub item_type: ItemType,
    /// Decoded spatial extents from `ispe`.
    pub extents: Option<SpatialExtents>,
    /// Codec configuration from `av2C` / `av1C`.
    pub codec_config: Option<CodecConfig>,
    /// Color information from `colr`.
    pub color_info: Option<ColorInfo>,
    /// Pixel-aspect-ratio from `pasp`.
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    /// Byte extents inside the file (from `iloc`).
    pub iloc_extents: Vec<(u64, u64)>,
    /// Auxiliary image type from `auxC` (present on alpha / depth items).
    pub auxiliary_type: Option<AuxiliaryType>,
    /// Whether the colour channels in the primary item are premultiplied by
    /// this alpha item's values.  Set when the `prem` item reference is
    /// present from the primary item to this alpha item.
    pub premultiplied_alpha: bool,
    /// Display orientation from `irot` and/or `imir` item properties.
    /// `None` means the decoder found neither property (treat as identity).
    pub orientation: Option<Orientation>,
    /// Visible crop rectangle from the `clap` item property.
    /// `None` if the property is absent (the full coded image is displayed).
    pub clean_aperture: Option<CleanAperture>,
    /// ICC profile from a `colr` box of type `rICC` or `prof`.
    pub icc_profile: Option<Vec<u8>>,
}

impl AvifItem {
    fn new(item_id: u16, item_type: ItemType) -> Self {
        Self {
            item_id,
            item_type,
            extents: None,
            codec_config: None,
            color_info: None,
            pixel_aspect_ratio: None,
            iloc_extents: Vec::new(),
            auxiliary_type: None,
            premultiplied_alpha: false,
            orientation: None,
            clean_aperture: None,
            icc_profile: None,
        }
    }
}

/// Metadata about a decoded AVIF image, available before or alongside pixel data.
#[derive(Debug, Clone)]
pub struct AvifImageInfo {
    /// Width in pixels (from `ispe`).
    pub width: u32,
    /// Height in pixels (from `ispe`).
    pub height: u32,
    /// Pixel layout derived from the codec config.
    pub pixel_layout: PixelLayout,
    /// Bits per component.
    pub bits_per_component: u8,
    /// Colour information, if present.
    pub color_info: Option<ColorInfo>,
    /// Content light level metadata from HDR CLL metadata OBUs, if present.
    pub content_light_level: Option<ContentLightLevel>,
    /// Pixel-aspect-ratio, if present.
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    /// Item type of the primary image.
    pub item_type: ItemType,
    /// Whether the file contains a separate alpha (transparency) item linked
    /// to the primary item via an `auxl` `iref` reference.
    pub has_alpha: bool,
    /// Whether the colour channels are premultiplied by alpha (`prem` iref).
    pub premultiplied_alpha: bool,
    /// Display orientation from `irot` / `imir` properties; `None` = identity.
    pub orientation: Option<Orientation>,
    /// Visible crop rectangle from the `clap` property
    pub clean_aperture: Option<CleanAperture>,
    /// ICC profile, if the `colr` box carried `rICC` or `prof` data.
    pub icc_profile: Option<Vec<u8>>,
}

/// A fully parsed AVIF container, ready for decoding.
///
/// Produced by [`AvifParser::parse`].
#[derive(Debug)]
pub struct AvifContainer {
    /// All items found in `iinf`.
    pub items: HashMap<u16, AvifItem>,
    /// `pitm` primary item ID.
    pub primary_item_id: u16,
    /// Brand string from `ftyp` (e.g. `avif`).
    pub brand: [u8; 4],
    /// Item ID of the alpha auxiliary image linked to the primary item via
    /// an `auxl` `iref` reference, if any.
    pub alpha_item_id: Option<u16>,
}

impl AvifContainer {
    /// Return a reference to the primary image item.
    pub fn primary_item(&self) -> Option<&AvifItem> {
        self.items.get(&self.primary_item_id)
    }

    /// Return a reference to the alpha auxiliary item, if present.
    pub fn alpha_item(&self) -> Option<&AvifItem> {
        self.alpha_item_id.and_then(|id| self.items.get(&id))
    }
}
#[derive(Debug)]
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    fn remaining(&self) -> usize {
        // FUZZ: saturating_sub prevents underflow if pos ever exceeds data.len()
        // (which should be impossible with safe reads but provides a safe fallback).
        self.data.len().saturating_sub(self.pos)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn read_u8(&mut self) -> Result<u8> {
        let v = self
            .data
            .get(self.pos)
            .copied()
            .ok_or(AvifError::TooShort)?;
        self.pos += 1;
        Ok(v)
    }

    fn read_u16_be(&mut self) -> Result<u16> {
        let b = self.read_bytes::<2>()?;
        Ok(u16::from_be_bytes(b))
    }

    fn read_u32_be(&mut self) -> Result<u32> {
        let b = self.read_bytes::<4>()?;
        Ok(u32::from_be_bytes(b))
    }

    fn read_u64_be(&mut self) -> Result<u64> {
        let b = self.read_bytes::<8>()?;
        Ok(u64::from_be_bytes(b))
    }

    fn read_bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        // FUZZ: checked_add prevents pos+N from wrapping on pathological inputs.
        let end = self.pos.checked_add(N).ok_or(AvifError::TooShort)?;
        if end > self.data.len() {
            return Err(AvifError::TooShort);
        }
        let mut buf = [0u8; N];
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(buf)
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8]> {
        // FUZZ: checked_add prevents wrapping.
        let end = self.pos.checked_add(n).ok_or(AvifError::TooShort)?;
        if end > self.data.len() {
            return Err(AvifError::TooShort);
        }
        let sl = &self.data[self.pos..end];
        self.pos = end;
        Ok(sl)
    }

    /// Create a sub-reader for exactly `len` bytes starting at the current position.
    fn sub_reader(&mut self, len: usize) -> Result<Reader<'a>> {
        // FUZZ: checked_add prevents wrapping.
        let end = self.pos.checked_add(len).ok_or(AvifError::TooShort)?;
        if end > self.data.len() {
            return Err(AvifError::TooShort);
        }
        let sl = &self.data[self.pos..end];
        self.pos = end;
        Ok(Reader { data: sl, pos: 0 })
    }
}

#[derive(Debug, Clone, Copy)]
struct BoxHeader {
    /// Four-character-code.
    fourcc: [u8; 4],
    /// Header byte size (8 for normal, 16 for extended).
    #[allow(dead_code)]
    header_size: u64,
}

/// Read the next ISOBMFF box header from `r`, returning header + a sub-reader
/// strictly limited to the payload bytes.
///
/// # Fuzzer hardening
///
/// - Rejects size < 8 (invalid per spec) to prevent underflow in payload_size.
/// - Rejects extended-size < 16 (would underflow after subtracting header).
/// - `payload_size` is derived from the box's own declared size, so a sub-reader
///   can never reach outside the bytes already consumed from the parent reader:
///   the sub_reader call will return TooShort if the box claims more bytes than
///   are available.
/// - size=0 ("box extends to EOF") is rejected in nested contexts: we only accept
///   it at the top level by treating the entire remaining parent slice as the box.
///   Here we reject it unconditionally and let callers handle EOF by checking
///   `is_empty()` before calling this function.
fn read_box_header<'a>(r: &mut Reader<'a>) -> Result<(BoxHeader, Reader<'a>)> {
    // FUZZ: require at least 8 bytes before attempting any reads.
    if r.remaining() < 8 {
        return Err(AvifError::TooShort);
    }

    let size_field = r.read_u32_be()?;
    let fourcc = r.read_bytes::<4>()?;

    let (total_size, header_size): (u64, u64) = match size_field {
        // FUZZ: size=0 ("extends to EOF") is rejected outright.  In the original
        // code it used `r.data.len()` which is the *sub-reader* length, not the
        // file length, giving a wrong total in nested contexts and allowing the
        // payload_size calculation to produce an incorrect (potentially huge)
        // value.  Real-world AVIF encoders never emit size=0 except as the
        // outermost box; we just refuse it to keep the logic simple.
        0 => return Err(AvifError::InvalidBox),

        1 => {
            // Extended 64-bit size: next 8 bytes hold the *total* box size.
            let ext = r.read_u64_be()?;
            // FUZZ: must be ≥ 16 (4+4+8 header bytes), otherwise payload_size
            // would underflow.
            if ext < 16 {
                return Err(AvifError::InvalidBox);
            }
            (ext, 16)
        }

        // FUZZ: sizes 2–7 are reserved/invalid and would underflow when
        // computing payload_size = total - 8.
        n if n < 8 => return Err(AvifError::InvalidBox),

        n => (n as u64, 8),
    };

    // FUZZ: checked_sub prevents underflow if total_size < header_size (which
    // the checks above make impossible, but we be defensive anyway).
    let payload_size = total_size
        .checked_sub(header_size)
        .ok_or(AvifError::InvalidBox)?;

    // FUZZ: usize cast is required for sub_reader; reject sizes that would
    // truncate on 32-bit platforms.
    let payload_size_usize = usize::try_from(payload_size).map_err(|_| AvifError::InvalidBox)?;

    // sub_reader will return TooShort if payload_size_usize > r.remaining(),
    // so no further bounds check is needed here.
    let payload = r.sub_reader(payload_size_usize)?;

    Ok((
        BoxHeader {
            fourcc,
            header_size,
        },
        payload,
    ))
}

fn read_fullbox_header(r: &mut Reader<'_>) -> Result<(u8, u32)> {
    let version = r.read_u8()?;
    let b = r.read_bytes::<3>()?;
    let flags = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
    Ok((version, flags))
}

/// Stateless AVIF container parser.
///
/// Call [`AvifParser::parse`] with the raw file bytes to obtain an
/// [`AvifContainer`] that can then be passed to [`AvifDecoder`].
pub struct AvifParser;

impl AvifParser {
    /// Parse an AVIF file from `data` and return the container metadata.
    ///
    /// No pixel decoding occurs here; only the box structure is walked.
    pub fn parse(data: &[u8]) -> Result<AvifContainer> {
        Self::parse_with_settings(data, &AvifSettings::default())
    }

    /// Parse an AVIF file with explicit parser limits.
    ///
    /// No pixel decoding occurs here; only the box structure is walked.
    pub fn parse_with_settings(data: &[u8], settings: &AvifSettings) -> Result<AvifContainer> {
        let mut r = Reader::new(data);

        let mut brand = [0u8; 4];
        let mut found_ftyp = false;
        let mut primary_item_id: u16 = 0;
        let mut items: HashMap<u16, AvifItem> = HashMap::new();

        // Defer iloc/ipma/iref parsing until iinf and ipco are complete.
        let mut iloc_raw: Option<Vec<u8>> = None;
        let mut ipco_props: Vec<([u8; 4], Vec<u8>)> = Vec::new();
        let mut ipma_raw: Option<Vec<u8>> = None;
        // iref raw bytes: deferred so item IDs are populated first.
        let mut iref_raw: Option<Vec<u8>> = None;

        // FUZZ: only parse the first `meta` box encountered; duplicates are
        // silently ignored to avoid re-entrant state corruption.
        let mut seen_meta = false;

        while !r.is_empty() {
            let (hdr, mut payload) = match read_box_header(&mut r) {
                Ok(v) => v,
                // FUZZ: a truncated trailing box is silently skipped; it
                // cannot carry valid content so we just stop the outer loop.
                Err(AvifError::TooShort) => break,
                Err(e) => return Err(e),
            };

            match &hdr.fourcc {
                b"ftyp" => {
                    // FUZZ: only parse the first ftyp box (must be first in
                    // spec, but we are tolerant of position; second occurrence
                    // is ignored).
                    if !found_ftyp {
                        brand = payload.read_bytes::<4>()?;
                        let _minor_version = payload.read_u32_be()?;

                        let mut avif_brand = is_avif_brand(&brand);

                        // FUZZ: cap the compatible-brand scan so a huge ftyp
                        // box cannot cause O(n) scanning. A zero setting disables
                        // this parser cap for trusted inputs.
                        let mut brand_count = 0usize;
                        while payload.remaining() >= 4
                            && (settings.max_compat_brands == 0
                                || brand_count < settings.max_compat_brands)
                        {
                            let compat = payload.read_bytes::<4>()?;
                            if is_avif_brand(&compat) {
                                avif_brand = true;
                            }
                            brand_count += 1;
                        }

                        if !avif_brand {
                            return Err(AvifError::NotAvif);
                        }
                        found_ftyp = true;
                    }
                }

                b"meta" if !seen_meta => {
                    seen_meta = true;
                    let (_ver, _flags) = read_fullbox_header(&mut payload)?;
                    Self::parse_meta(
                        &mut payload,
                        &mut primary_item_id,
                        &mut items,
                        &mut iloc_raw,
                        &mut ipco_props,
                        &mut ipma_raw,
                        &mut iref_raw,
                        settings,
                    )?;
                }

                // mdat and all other top-level boxes are consumed implicitly
                // by the sub-reader (their bytes are skipped).
                _ => {}
            }
        }

        if !found_ftyp {
            return Err(AvifError::NotAvif);
        }

        // Resolve iloc offsets into absolute file positions.
        if let Some(iloc_bytes) = iloc_raw {
            // FUZZ: pass the real file length so extent validation is exact.
            Self::apply_iloc(&iloc_bytes, &mut items, data.len() as u64, settings)?;
        }

        // Apply ipco properties to items via ipma associations.
        if !ipco_props.is_empty() {
            if let Some(ipma_bytes) = ipma_raw {
                Self::apply_ipma(&ipma_bytes, &ipco_props, &mut items, settings)?;
            }
        }

        // Resolve iref references to find the alpha item for the primary item.
        let alpha_item_id = if let Some(iref_bytes) = iref_raw {
            Self::resolve_alpha_from_iref(&iref_bytes, primary_item_id, &mut items, settings)?
        } else {
            None
        };

        Ok(AvifContainer {
            items,
            primary_item_id,
            brand,
            alpha_item_id,
        })
    }

    fn parse_meta(
        r: &mut Reader<'_>,
        primary_item_id: &mut u16,
        items: &mut HashMap<u16, AvifItem>,
        iloc_raw: &mut Option<Vec<u8>>,
        ipco_props: &mut Vec<([u8; 4], Vec<u8>)>,
        ipma_raw: &mut Option<Vec<u8>>,
        iref_raw: &mut Option<Vec<u8>>,
        settings: &AvifSettings,
    ) -> Result<()> {
        // FUZZ: track which critical singleton boxes have been seen; a second
        // occurrence is rejected to prevent state-confusion attacks.
        let mut seen_pitm = false;
        let mut seen_iinf = false;
        let mut seen_iloc = false;
        let mut seen_iprp = false;
        let mut seen_iref = false;

        while !r.is_empty() {
            let (hdr, mut payload) = match read_box_header(r) {
                Ok(v) => v,
                Err(AvifError::TooShort) => break,
                Err(e) => return Err(e),
            };

            match &hdr.fourcc {
                b"pitm" if !seen_pitm => {
                    seen_pitm = true;
                    let (ver, _flags) = read_fullbox_header(&mut payload)?;
                    *primary_item_id = if ver == 0 {
                        payload.read_u16_be()?
                    } else {
                        let v = payload.read_u32_be()?;
                        // pitm v1 uses a u32 item_ID; our internal representation
                        // is u16 so IDs > 65535 are clamped to 0 (invalid sentinel).
                        u16::try_from(v).unwrap_or(0)
                    };
                }

                b"iinf" if !seen_iinf => {
                    seen_iinf = true;
                    let (ver, _flags) = read_fullbox_header(&mut payload)?;
                    let raw_count: u32 = if ver == 0 {
                        payload.read_u16_be()? as u32
                    } else {
                        payload.read_u32_be()?
                    };

                    // FUZZ: cap item count before allocating or looping.
                    if settings.max_items != 0 && raw_count > settings.max_items {
                        return Err(AvifError::LimitExceeded);
                    }
                    let count = raw_count;

                    for _ in 0..count {
                        let (_, mut ie_payload) = read_box_header(&mut payload)?;
                        let (iv, _) = read_fullbox_header(&mut ie_payload)?;
                        // infe item_id field width (ISO 14496-12 §8.11.6.2):
                        //   v0 / v1 / v2 → item_ID is u16
                        //   v3           → item_ID is u32
                        let item_id: u16 = if iv < 3 {
                            ie_payload.read_u16_be()?
                        } else {
                            let v = ie_payload.read_u32_be()?;
                            // u32 item IDs > 65535 can't be represented; skip
                            // silently — they belong to a feature we don't support.
                            u16::try_from(v).unwrap_or(0)
                        };
                        // item_protection_index (u16) — ignored.
                        let _ = ie_payload.read_u16_be()?;
                        // item_type is present in v1/v2/v3 but not v0.
                        // For v0 we'd need to read a name string; since AVIF
                        // always uses v2+, treat v0 as having no type.
                        let item_type_bytes = if iv >= 1 {
                            ie_payload.read_bytes::<4>()?
                        } else {
                            *b"\0\0\0\0"
                        };
                        let item_type = ItemType::from_fourcc(item_type_bytes);
                        // FUZZ: or_insert_with avoids clobbering an existing
                        // entry if iinf contains duplicate item_ids.
                        items
                            .entry(item_id)
                            .or_insert_with(|| AvifItem::new(item_id, item_type));
                    }
                }

                b"iloc" if !seen_iloc => {
                    seen_iloc = true;
                    // Store raw bytes; resolved after iinf is fully parsed.
                    let raw = payload.read_slice(payload.remaining())?;
                    *iloc_raw = Some(raw.to_vec());
                }

                b"iprp" if !seen_iprp => {
                    seen_iprp = true;
                    Self::parse_iprp(&mut payload, ipco_props, ipma_raw, settings)?;
                }

                b"iref" if !seen_iref => {
                    seen_iref = true;
                    // Store raw bytes (includes the FullBox header); resolved
                    // after ipma has applied auxC properties to items.
                    let raw = payload.read_slice(payload.remaining())?;
                    *iref_raw = Some(raw.to_vec());
                }

                // All other singleton-once boxes and unknown boxes are skipped.
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_iprp(
        r: &mut Reader<'_>,
        ipco_props: &mut Vec<([u8; 4], Vec<u8>)>,
        ipma_raw: &mut Option<Vec<u8>>,
        settings: &AvifSettings,
    ) -> Result<()> {
        let mut seen_ipco = false;
        let mut seen_ipma = false;

        while !r.is_empty() {
            let (hdr, mut payload) = match read_box_header(r) {
                Ok(v) => v,
                Err(AvifError::TooShort) => break,
                Err(e) => return Err(e),
            };

            match &hdr.fourcc {
                b"ipco" if !seen_ipco => {
                    seen_ipco = true;
                    while !payload.is_empty() {
                        let (phdr, mut ppayload) = match read_box_header(&mut payload) {
                            Ok(v) => v,
                            Err(AvifError::TooShort) => break,
                            Err(e) => return Err(e),
                        };

                        // FUZZ: cap the total number of ipco properties to
                        // prevent unbounded Vec growth.  The Vec<u8> payloads
                        // are already bounded by the sub-reader, so this is a
                        // count-not-bytes limit.
                        if settings.max_ipco_props != 0
                            && ipco_props.len() >= settings.max_ipco_props
                        {
                            return Err(AvifError::LimitExceeded);
                        }

                        let raw = ppayload.read_slice(ppayload.remaining())?;
                        ipco_props.push((phdr.fourcc, raw.to_vec()));
                    }
                }

                b"ipma" if !seen_ipma => {
                    seen_ipma = true;
                    let raw = payload.read_slice(payload.remaining())?;
                    *ipma_raw = Some(raw.to_vec());
                }

                _ => {}
            }
        }
        Ok(())
    }

    fn apply_iloc(
        iloc_bytes: &[u8],
        items: &mut HashMap<u16, AvifItem>,
        file_len: u64,
        settings: &AvifSettings,
    ) -> Result<()> {
        let mut r = Reader::new(iloc_bytes);
        let (ver, _flags) = read_fullbox_header(&mut r)?;

        let byte = r.read_u8()?;
        let byte2 = r.read_u8()?;

        let offset_size = (byte >> 4) & 0x0F;
        let length_size = byte & 0x0F;
        let base_offset_size = (byte2 >> 4) & 0x0F;
        let index_size = if ver >= 1 { byte2 & 0x0F } else { 0 };

        // FUZZ: the spec only allows 0, 4, or 8 for these field sizes.
        // Accept 1 and 2 as well (some encoders use them in practice), but
        // reject anything else (3, 5, 6, 7, 9–15) to prevent the unusual
        // byte-by-byte read path from being exercised on garbage inputs.
        for &sz in &[offset_size, length_size, base_offset_size, index_size] {
            if !matches!(sz, 0 | 1 | 2 | 4 | 8) {
                return Err(AvifError::InvalidBox);
            }
        }

        let raw_item_count: u32 = if ver < 2 {
            r.read_u16_be()? as u32
        } else {
            r.read_u32_be()?
        };

        // FUZZ: cap item count before looping.
        if settings.max_iloc_items != 0 && raw_item_count > settings.max_iloc_items {
            return Err(AvifError::LimitExceeded);
        }

        for _ in 0..raw_item_count {
            let item_id: u16 = if ver < 2 {
                r.read_u16_be()?
            } else {
                let v = r.read_u32_be()?;
                u16::try_from(v).map_err(|_| AvifError::InvalidBox)?
            };

            // construction_method (version 1/2 only).
            if ver >= 1 {
                let cm = r.read_u16_be()?;
                // Method 0 = file offset; 1 = idat; 2 = item.
                // Only method 0 is supported; skip and consume the rest of
                // this entry for non-zero methods.
                if cm & 0x0F != 0 {
                    let _ = r.read_u16_be()?; // data_reference_index
                    let _ = read_sized_int(&mut r, base_offset_size)?; // base_offset
                    let ec = r.read_u16_be()?;
                    // FUZZ: reject over-large skipped entries too. Capping here
                    // would leave unread extent records in this iloc entry and
                    // desynchronize the parser for the next item.
                    if settings.max_extents_per_item != 0 && ec > settings.max_extents_per_item {
                        return Err(AvifError::LimitExceeded);
                    }
                    for _ in 0..ec {
                        if index_size > 0 {
                            let _ = read_sized_int(&mut r, index_size)?;
                        }
                        let _ = read_sized_int(&mut r, offset_size)?;
                        let _ = read_sized_int(&mut r, length_size)?;
                    }
                    continue;
                }
            }

            let _ = r.read_u16_be()?; // data_reference_index
            let base_offset = read_sized_int(&mut r, base_offset_size)?;
            let extent_count = r.read_u16_be()?;

            // FUZZ: cap extents per item; prevents large allocations via
            // Vec::with_capacity and limits the inner loop iterations.
            if settings.max_extents_per_item != 0 && extent_count > settings.max_extents_per_item {
                return Err(AvifError::LimitExceeded);
            }

            let mut extents: Vec<(u64, u64)> = Vec::new();
            extents
                .try_reserve_exact(extent_count as usize)
                .map_err(|_| AvifError::OutOfMemory)?;

            for _ in 0..extent_count {
                if index_size > 0 {
                    let _ = read_sized_int(&mut r, index_size)?;
                }
                let ext_offset = read_sized_int(&mut r, offset_size)?;
                let ext_length = read_sized_int(&mut r, length_size)?;

                // FUZZ: checked arithmetic prevents wrapping on 64-bit; on a
                // 32-bit host the usize cast later in decode() is also guarded.
                let abs_offset = base_offset
                    .checked_add(ext_offset)
                    .ok_or(AvifError::InvalidBox)?;

                let end = abs_offset
                    .checked_add(ext_length)
                    .ok_or(AvifError::InvalidBox)?;

                // FUZZ: validate against the real file length, not any
                // sub-reader length, so crafted offsets cannot reach outside
                // the file.
                if end > file_len {
                    return Err(AvifError::ExtentOutOfBounds);
                }

                extents.push((abs_offset, ext_length));
            }

            if let Some(item) = items.get_mut(&item_id) {
                item.iloc_extents = extents;
            }
        }
        Ok(())
    }

    fn apply_ipma(
        ipma_bytes: &[u8],
        ipco_props: &[([u8; 4], Vec<u8>)],
        items: &mut HashMap<u16, AvifItem>,
        settings: &AvifSettings,
    ) -> Result<()> {
        let mut r = Reader::new(ipma_bytes);
        let (ver, flags) = read_fullbox_header(&mut r)?;

        let entry_count = r.read_u32_be()?;

        // FUZZ: cap entry count before looping.
        if settings.max_ipma_entries != 0 && entry_count > settings.max_ipma_entries {
            return Err(AvifError::LimitExceeded);
        }

        for _ in 0..entry_count {
            let item_id: u16 = if ver < 1 {
                r.read_u16_be()?
            } else {
                let v = r.read_u32_be()?;
                // v1 ipma uses u32 item IDs; cap to u16 since that's our
                // internal representation.  IDs > 65535 are skipped below.
                u16::try_from(v).unwrap_or(0)
            };

            let assoc_count = r.read_u8()?;
            // assoc_count is u8, so at most 255 iterations — no cap needed.
            for _ in 0..assoc_count {
                // Bit 15 (or 7): essential flag; remaining bits: 1-based property index.
                let prop_idx: u16 = if flags & 1 != 0 {
                    let word = r.read_u16_be()?;
                    word & 0x7FFF
                } else {
                    let byte = r.read_u8()?;
                    (byte & 0x7F) as u16
                };

                if prop_idx == 0 {
                    continue; // 0 = no property, per spec
                }

                // FUZZ: wrapping_sub is safe here because prop_idx ≥ 1.
                let idx = (prop_idx as usize).wrapping_sub(1); // convert to 0-based

                // FUZZ: bounds-check the property index against the actual
                // ipco_props slice; out-of-range indices are silently skipped
                // (the spec says this is a parse error, but skipping is safe).
                if idx >= ipco_props.len() {
                    continue;
                }

                let (fourcc, raw_prop) = &ipco_props[idx];
                if let Some(item) = items.get_mut(&item_id) {
                    Self::apply_property(item, fourcc, raw_prop, settings)?;
                }
            }
        }
        Ok(())
    }

    fn apply_property(
        item: &mut AvifItem,
        fourcc: &[u8; 4],
        raw: &[u8],
        settings: &AvifSettings,
    ) -> Result<()> {
        match fourcc {
            b"ispe" => {
                // ispe: FullBox (version=0, flags=0), image_width u32, image_height u32.
                // Total: 4 (fullbox) + 4 (width) + 4 (height) = 12 bytes minimum.
                if raw.len() < 12 {
                    return Err(AvifError::InvalidBox);
                }
                // raw[0..4] = version(1) + flags(3); raw[4..8] = width; raw[8..12] = height.
                let width = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
                let height = u32::from_be_bytes([raw[8], raw[9], raw[10], raw[11]]);

                // FUZZ: reject zero or absurdly large dimensions to prevent
                // downstream overflow in plane-size arithmetic.
                if width == 0 || height == 0 {
                    return Err(AvifError::InvalidBox);
                }
                if settings.max_image_dimension != 0
                    && (width > settings.max_image_dimension
                        || height > settings.max_image_dimension)
                {
                    return Err(AvifError::LimitExceeded);
                }

                item.extents = Some(SpatialExtents { width, height });
            }

            b"av2C" | b"av1C" => {
                // av2C/av1C layout (AOM HEIF-AV1/AV2 spec §2.2.1):
                //   byte 0: marker(1=MSB) | version(7)
                //   byte 1: seq_profile(3) | seq_level_idx_0(5)
                //   byte 2: seq_tier_0(1) | high_bitdepth(1) | twelve_bit(1) | monochrome(1)
                //           | chroma_subsampling_x(1) | chroma_subsampling_y(1)
                //           | chroma_sample_position(2)
                //   byte 3: reserved(3) | initial_presentation_delay_present(1) | ...
                //   bytes 4+ (or 5+ if ipd_present): configOBUs
                if raw.len() < 4 {
                    return Err(AvifError::InvalidCodecConfig);
                }

                // FUZZ: validate marker bit to catch garbage payloads early.
                if raw[0] & 0x80 == 0 {
                    return Err(AvifError::InvalidCodecConfig);
                }

                let b2 = raw[2];
                let b3 = raw[3];

                let high_bitdepth = (b2 >> 6) & 1 != 0;
                let twelve_bit = (b2 >> 5) & 1 != 0;
                let monochrome = (b2 >> 4) & 1 != 0;
                let chroma_subsampling_x = (b2 >> 3) & 1;
                let chroma_subsampling_y = (b2 >> 2) & 1;
                let chroma_sample_position = b2 & 0x3;

                let ipd_present = (b3 >> 4) & 1 != 0;
                let config_obu_start = if ipd_present { 5 } else { 4 };

                // FUZZ: slice index is always ≤ raw.len() because raw.len() ≥ 4
                // and config_obu_start is at most 5; the get() handles the
                // config_obu_start=5, raw.len()=4 case safely.
                let config_obus = raw.get(config_obu_start..).unwrap_or(&[]).to_vec();

                item.codec_config = Some(CodecConfig {
                    config_obus,
                    high_bitdepth,
                    twelve_bit,
                    monochrome,
                    chroma_subsampling_x,
                    chroma_subsampling_y,
                    chroma_sample_position,
                });
            }

            b"colr" => {
                // colr: color_type (4 bytes) + type-specific payload.
                // Three subtypes are defined:
                //   "nclx" — CICP color info (primaries, TRC, matrix, range)
                //   "rICC" — restricted ICC profile (ICC.1 v2/v4, embedded)
                //   "prof" — unrestricted ICC profile (ICC.1 v4, embedded)
                if raw.len() < 4 {
                    return Err(AvifError::InvalidBox);
                }
                match &raw[..4] {
                    b"nclx" => {
                        // nclx: color_type(4) + primaries(2) + trc(2) + matrix(2)
                        //       + full_range(1) = 11 bytes minimum.
                        if raw.len() < 11 {
                            return Err(AvifError::InvalidBox);
                        }
                        let cp = u16::from_be_bytes([raw[4], raw[5]]);
                        let tc = u16::from_be_bytes([raw[6], raw[7]]);
                        let mc = u16::from_be_bytes([raw[8], raw[9]]);
                        let full_range = (raw[10] >> 7) & 1;
                        item.color_info = Some(ColorInfo {
                            color_primaries: ColorPrimaries::from(cp),
                            transfer_characteristics: TransferCharacteristics::from(tc),
                            matrix_coefficients: MatrixCoefficients::from(mc),
                            full_range: full_range != 0,
                        });
                    }
                    b"rICC" | b"prof" => {
                        // rICC / prof: color_type(4) + raw ICC profile bytes.
                        // We need at least the 4-byte type code; an empty profile
                        // is technically invalid but we tolerate it gracefully.
                        let profile_bytes = raw[4..].to_vec();
                        item.icc_profile = Some(profile_bytes);
                    }
                    _ => {
                        // Unknown color type — ignore silently per spec §12.
                    }
                }
            }

            b"pasp" => {
                // pasp: hSpacing(4) + vSpacing(4) = 8 bytes.
                if raw.len() < 8 {
                    // FUZZ: changed from silent skip to explicit error so the
                    // parser surfaces truncated pasp boxes rather than silently
                    // leaving pixel_aspect_ratio unset.
                    return Err(AvifError::InvalidBox);
                }
                let h = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
                let v = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);

                // FUZZ: zero spacing would cause division-by-zero in callers
                // that compute display aspect ratio.
                if h == 0 || v == 0 {
                    return Err(AvifError::InvalidBox);
                }

                item.pixel_aspect_ratio = Some(PixelAspectRatio {
                    h_spacing: h,
                    v_spacing: v,
                });
            }

            b"auxC" => {
                // auxC: FullBox (4 bytes) + null-terminated UTF-8 URN string.
                // Used to identify alpha, depth, and other auxiliary item types.
                // Minimum: 4 (fullbox) + 1 (at least one URN byte or NUL) = 5 bytes.
                if raw.len() < 5 {
                    return Err(AvifError::InvalidBox);
                }
                // raw[0..4] = version(1) + flags(3); rest is the URN.
                let urn_bytes = &raw[4..];

                // FUZZ: cap URN length before cloning into AuxiliaryType::Other.
                if settings.max_auxc_urn_len != 0 && urn_bytes.len() > settings.max_auxc_urn_len {
                    return Err(AvifError::LimitExceeded);
                }

                item.auxiliary_type = Some(AuxiliaryType::from_urn_bytes(urn_bytes));
            }

            b"irot" => {
                // irot: a single byte.  Bits [1:0] encode the anticlockwise
                // rotation in 90° steps: 0=0°, 1=90°, 2=180°, 3=270°.
                // No FullBox header — the payload is exactly 1 byte.
                if raw.is_empty() {
                    return Err(AvifError::InvalidBox);
                }
                let steps = raw[0] & 0x03;
                // Preserve any previously-parsed imir axis on this item.
                let axis = item.orientation.and_then(|o| o.imir_axis());
                item.orientation = Some(Orientation::from_irot_imir(steps, axis));
            }

            b"imir" => {
                // imir: a single byte.  Bit 0 is the axis:
                //   0 = vertical axis   → left-right (horizontal) flip
                //   1 = horizontal axis → top-bottom (vertical) flip
                // No FullBox header.
                if raw.is_empty() {
                    return Err(AvifError::InvalidBox);
                }
                // false = vertical axis (left-right flip), true = horizontal axis.
                let axis = Some(raw[0] & 0x01 != 0);
                // Preserve any previously-parsed irot steps on this item.
                let steps = item.orientation.map_or(0, |o| o.irot_steps());
                item.orientation = Some(Orientation::from_irot_imir(steps, axis));
            }

            b"clap" => {
                // clap: 8 × u32 big-endian = 4 rational pairs.
                // Layout: cleanApertureWidthN, cleanApertureWidthD,
                //         cleanApertureHeightN, cleanApertureHeightD,
                //         horizOffN, horizOffD, vertOffN, vertOffD.
                // Total: 32 bytes.  No FullBox header.
                if raw.len() < 32 {
                    return Err(AvifError::InvalidBox);
                }
                let read_u32 = |off: usize| -> i32 {
                    i32::from_be_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
                };
                let width_n = read_u32(0);
                let width_d = read_u32(4);
                let height_n = read_u32(8);
                let height_d = read_u32(12);
                let hoff_n = read_u32(16);
                let hoff_d = read_u32(20);
                let voff_n = read_u32(24);
                let voff_d = read_u32(28);

                // FUZZ: zero denominators would cause divide-by-zero in callers.
                if width_d == 0 || height_d == 0 || hoff_d == 0 || voff_d == 0 {
                    return Err(AvifError::InvalidBox);
                }
                // FUZZ: non-positive aperture size is nonsensical.
                if width_n <= 0 || height_n <= 0 {
                    return Err(AvifError::InvalidBox);
                }

                item.clean_aperture = Some(CleanAperture {
                    width: Rational {
                        numerator: width_n,
                        denominator: width_d,
                    },
                    height: Rational {
                        numerator: height_n,
                        denominator: height_d,
                    },
                    horiz_off: Rational {
                        numerator: hoff_n,
                        denominator: hoff_d,
                    },
                    vert_off: Rational {
                        numerator: voff_n,
                        denominator: voff_d,
                    },
                });
            }

            // All other property types are ignored.
            _ => {}
        }
        Ok(())
    }

    /// Walk an `iref` box and return the item ID of the alpha auxiliary image
    /// associated with `primary_item_id`, if any.
    ///
    /// ## How AVIF alpha references work
    ///
    /// The `iref` box contains one or more typed reference sub-boxes.  Each
    /// sub-box has a four-character reference type (`auxl`, `thmb`, `dimg`, …)
    /// and a list of `(from_item_id, to_item_id…)` pairs.  For alpha:
    ///
    /// ```text
    /// iref
    ///   auxl  from_item=<alpha_item_id>  to_item=<primary_item_id>
    ///   prem  from_item=<primary_item_id> to_item=<alpha_item_id>   (optional)
    /// ```
    ///
    /// So the alpha item *points at* the primary item, not the other way round.
    /// We scan every `auxl` entry; when `to_item_id == primary_item_id` and the
    /// `from_item` has an `AuxiliaryType::Alpha` property (set by `auxC` via
    /// `ipma`), we record it.  We also scan `prem` to set
    /// `premultiplied_alpha` on the alpha item.
    ///
    /// ## Fuzzer hardening
    ///
    /// - `iref` is a FullBox; version drives the item-ID field width (0 → u16,
    ///   1 → u32, down-cast to u16).
    /// - The reference count inside each typed-reference sub-box is u16 (up to
    ///   65535 per spec); we cap it with [`AvifSettings::max_iref_refs`].
    /// - Unknown reference types are skipped by consuming their payload via the
    ///   sub-reader (already bounded by box size).
    fn resolve_alpha_from_iref(
        iref_bytes: &[u8],
        primary_item_id: u16,
        items: &mut HashMap<u16, AvifItem>,
        settings: &AvifSettings,
    ) -> Result<Option<u16>> {
        let mut r = Reader::new(iref_bytes);
        let (ver, _flags) = read_fullbox_header(&mut r)?;

        let mut alpha_item_id: Option<u16> = None;

        while !r.is_empty() {
            let (hdr, mut payload) = match read_box_header(&mut r) {
                Ok(v) => v,
                Err(AvifError::TooShort) => break,
                Err(e) => return Err(e),
            };

            let ref_type = &hdr.fourcc;
            let is_auxl = ref_type == b"auxl";
            let is_prem = ref_type == b"prem";

            if !is_auxl && !is_prem {
                // Skip unknown reference types; their bytes are already
                // consumed by the sub-reader.
                continue;
            }

            // from_item_id: the item that *has* the reference.
            let from_item_id: u16 = if ver == 0 {
                payload.read_u16_be()?
            } else {
                let v = payload.read_u32_be()?;
                u16::try_from(v).unwrap_or(0)
            };

            // reference_count: number of target item IDs that follow.
            let ref_count = payload.read_u16_be()?;

            // FUZZ: cap reference count per entry.
            if settings.max_iref_refs != 0 && ref_count > settings.max_iref_refs {
                return Err(AvifError::LimitExceeded);
            }

            for _ in 0..ref_count {
                let to_item_id: u16 = if ver == 0 {
                    payload.read_u16_be()?
                } else {
                    let v = payload.read_u32_be()?;
                    u16::try_from(v).unwrap_or(0)
                };

                if is_auxl && to_item_id == primary_item_id {
                    // `from_item_id` claims to be an auxiliary image for the
                    // primary.  Accept it as alpha only if its auxC property
                    // (set during ipma application) identifies it as alpha.
                    if let Some(item) = items.get(&from_item_id) {
                        if item.auxiliary_type.as_ref().is_some_and(|a| a.is_alpha()) {
                            // Last writer wins if multiple alpha items declare
                            // themselves; real encoders only emit one.
                            alpha_item_id = Some(from_item_id);
                        }
                    }
                }

                if is_prem && from_item_id == primary_item_id {
                    // `to_item_id` is the alpha item whose channel data
                    // the primary is premultiplied against.
                    if let Some(item) = items.get_mut(&to_item_id) {
                        item.premultiplied_alpha = true;
                    }
                }
            }
        }

        Ok(alpha_item_id)
    }
}

/// Read an unsigned integer of `size` bytes from `r`.
///
/// Only sizes 0, 1, 2, 4, and 8 are accepted; other values are rejected.
/// (The caller in `apply_iloc` already validates the sizes from the bitstream,
/// so this function is a second defensive layer.)
fn read_sized_int(r: &mut Reader<'_>, size: u8) -> Result<u64> {
    // FUZZ: explicit match on all allowed sizes; any other value returns an
    // error rather than falling through to the unchecked byte-by-byte path.
    match size {
        0 => Ok(0),
        1 => Ok(r.read_u8()? as u64),
        2 => Ok(r.read_u16_be()? as u64),
        4 => Ok(r.read_u32_be()? as u64),
        8 => Ok(r.read_u64_be()?),
        _ => Err(AvifError::InvalidBox),
    }
}

#[inline]
fn is_avif_brand(b: &[u8; 4]) -> bool {
    matches!(b, b"avif" | b"avis" | b"av02")
}

/// Decoded alpha (transparency) plane from the auxiliary alpha item.
///
/// Presence as `Some` on [`AvifImage::alpha`] is the authoritative signal
/// that the image has an alpha channel; callers never need to check length.
/// Dimensions always equal the primary image dimensions.
#[derive(Debug, Clone)]
pub struct AlphaPlane {
    /// Raw plane bytes, row-major.  For 8-bit alpha each sample is one byte;
    /// for 10/12-bit alpha each sample is two bytes in little-endian order
    /// (matching the layout used by the AV2 decoder for HBD planes).
    pub data: Vec<u8>,
    /// Row stride in bytes.
    pub stride: usize,
    /// Bits per alpha sample (8, 10, or 12).
    pub bits_per_component: u8,
}

/// A fully decoded AVIF image: metadata + raw plane data.
pub struct AvifImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout (I400, I420, I422, I444).
    pub pixel_layout: PixelLayout,
    /// Bits per component (8, 10, or 12).
    pub bits_per_component: u8,
    /// Colour information, if present in the container.
    pub color_info: Option<ColorInfo>,
    /// Content light level metadata from HDR CLL metadata OBUs, if present.
    pub content_light_level: Option<ContentLightLevel>,
    /// Pixel-aspect-ratio, if present.
    pub pixel_aspect_ratio: Option<PixelAspectRatio>,
    /// Luma (Y) plane bytes (row-major, stride = width * bytes_per_sample).
    pub planes: [Vec<u8>; 3],
    /// Row stride for each plane, in bytes.
    pub strides: [usize; 2],
    /// Alpha (transparency) plane, or `None` if the image has no alpha channel.
    ///
    /// When `Some`, the plane dimensions always equal `width × height` and the
    /// sample layout matches the primary image's bit depth convention (8-bit →
    /// one byte per sample; 10/12-bit → two LE bytes per sample).
    pub alpha: Option<AlphaPlane>,
    /// Whether the colour planes in `planes` are premultiplied by `alpha`.
    /// Consumers must un-premultiply before compositing if this is `true`.
    pub premultiplied_alpha: bool,
    /// Display orientation from `irot` / `imir`; [`Orientation::Normal`]
    /// when neither property is present.
    pub orientation: Orientation,
    /// Visible crop rectangle from the `clap` property.
    /// `None` means the property was absent and the full coded frame is visible.
    pub clean_aperture: Option<CleanAperture>,
    /// ICC profile, if the `colr` box carried `rICC` or `prof` data.
    pub icc_profile: Option<Vec<u8>>,
    /// The raw underlying [`Picture`] from the AV2 decoder.
    pub picture: Arc<Picture>,
}

impl fmt::Debug for AvifImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AvifImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixel_layout", &self.pixel_layout)
            .field("bits_per_component", &self.bits_per_component)
            .field("color_info", &self.color_info)
            .field("content_light_level", &self.content_light_level)
            .field("has_alpha", &self.alpha.is_some())
            .field("premultiplied_alpha", &self.premultiplied_alpha)
            .field("orientation", &self.orientation)
            .field("clean_aperture", &self.clean_aperture)
            .field("has_icc_profile", &self.icc_profile.is_some())
            .finish_non_exhaustive()
    }
}

/// High-level AVIF / AV2 image decoder.
///
/// Wraps [`AvifParser`] (container parsing) and [`Decoder`] (AV2 bitstream
/// decoding) into a single convenient API.
pub struct AvifDecoder<'a> {
    file: &'a [u8],
    container: AvifContainer,
    settings: AvifSettings,
}

impl<'a> AvifDecoder<'a> {
    /// Parse the AVIF container from `data` (the raw file bytes).
    ///
    /// This is a fast metadata-only pass; no pixel decoding happens here.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Self::with_settings(data, AvifSettings::default())
    }

    /// Parse the AVIF container from `data` with explicit parser and decoder
    /// settings.
    ///
    /// This is a fast metadata-only pass; no pixel decoding happens here.
    pub fn with_settings(data: &'a [u8], settings: AvifSettings) -> Result<Self> {
        let container = AvifParser::parse_with_settings(data, &settings)?;
        Ok(Self {
            file: data,
            container,
            settings,
        })
    }

    /// Return image metadata for the primary item without decoding pixels.
    pub fn image_info(&self) -> Result<AvifImageInfo> {
        let item = self
            .container
            .primary_item()
            .ok_or(AvifError::MissingBox("pitm"))?;

        let extents = item.extents.ok_or(AvifError::MissingBox("ispe"))?;
        let cfg = item
            .codec_config
            .as_ref()
            .ok_or(AvifError::MissingBox("av2C"))?;

        let has_alpha = self.container.alpha_item_id.is_some();
        let premultiplied_alpha = self
            .container
            .alpha_item()
            .is_some_and(|a| a.premultiplied_alpha);

        Ok(AvifImageInfo {
            width: extents.width,
            height: extents.height,
            pixel_layout: cfg.pixel_layout(),
            bits_per_component: cfg.bits_per_component(),
            color_info: item.color_info,
            content_light_level: self.item_content_light_level(item)?,
            pixel_aspect_ratio: item.pixel_aspect_ratio,
            item_type: item.item_type,
            has_alpha,
            premultiplied_alpha,
            orientation: item.orientation,
            clean_aperture: item.clean_aperture,
            icc_profile: item.icc_profile.clone(),
        })
    }

    /// Decode the primary image item and return its pixel data.
    ///
    /// The caller receives an [`AvifImage`] whose `planes` vectors contain
    /// copies of the plane data (safe to use without lifetime concerns).
    ///
    /// When the container carries a separate alpha auxiliary item (linked via
    /// an `auxl` `iref` reference with an `auxC` property set to the alpha
    /// URN), it is decoded with a second AV2 decoder pass and its luma plane
    /// is returned in [`AvifImage::alpha`].  The dimensions of the alpha plane
    /// are validated to match the primary image; a mismatch is a hard error.
    pub fn decode(&mut self) -> Result<AvifImage> {
        let item = self
            .container
            .primary_item()
            .ok_or(AvifError::MissingBox("pitm"))?;

        if item.item_type == ItemType::Grid {
            return Err(AvifError::GridNotSupported);
        }
        if !item.item_type.is_image() {
            return Err(AvifError::UnsupportedCodec);
        }

        let extents = item.extents.ok_or(AvifError::MissingBox("ispe"))?;
        let cfg = item
            .codec_config
            .as_ref()
            .ok_or(AvifError::MissingBox("av2C"))?;
        let alpha_item_id = self.container.alpha_item_id;
        let premultiplied_alpha = alpha_item_id
            .and_then(|id| self.container.items.get(&id))
            .is_some_and(|a| a.premultiplied_alpha);

        let primary_obu = self.assemble_obu(item)?;
        if primary_obu.is_empty() {
            return Err(AvifError::MissingBox("mdat"));
        }

        let picture = self.run_decoder(primary_obu)?;

        let layout = picture.p.layout;
        let bpc = u8::try_from(picture.p.bpc).map_err(|_| AvifError::InvalidCodecConfig)?;
        let bps: usize = picture.bytes_per_sample();

        if cfg.pixel_layout() != layout || cfg.bits_per_component() != bpc {
            return Err(AvifError::InvalidCodecConfig);
        }

        // FUZZ: i32 → usize casts are safe because picture dimensions come
        // from the decoder which validates them against Settings::frame_size_limit.
        let pic_w = picture.p.w as usize;
        let pic_h = picture.p.h as usize;

        let y_stride = picture.stride[0].unsigned_abs();
        let uv_stride = picture.stride[1].unsigned_abs();

        let uv_h = match layout {
            PixelLayout::I420 => pic_h.div_ceil(2),
            PixelLayout::I400 => 0,
            _ => pic_h,
        };
        let uv_w = match layout {
            PixelLayout::I420 | PixelLayout::I422 => pic_w.div_ceil(2),
            PixelLayout::I400 => 0,
            _ => pic_w,
        };

        // FUZZ: all multiplications in copy_plane are checked to prevent
        // integer overflow when computing row pointers or allocation sizes.
        let plane_y = copy_plane(picture.plane_bytes(0), y_stride, pic_h, pic_w, bps)?;
        let plane_u = copy_plane(picture.plane_bytes(1), uv_stride, uv_h, uv_w, bps)?;
        let plane_v = copy_plane(picture.plane_bytes(2), uv_stride, uv_h, uv_w, bps)?;

        // AvifImage::strides are byte strides for the compact output buffers.
        // For high-bit-depth output this must be width * 2, not width.
        let out_y_stride = pic_w.checked_mul(bps).ok_or(AvifError::InvalidBox)?;
        let out_uv_stride = uv_w.checked_mul(bps).ok_or(AvifError::InvalidBox)?;

        let content_light_level = picture.content_light_level;
        let picture_arc = Arc::new(picture);

        let alpha: Option<AlphaPlane> = if let Some(aid) = alpha_item_id {
            let alpha_item = self
                .container
                .items
                .get(&aid)
                .ok_or(AvifError::MissingBox("alpha item"))?;

            if !alpha_item.item_type.is_image() {
                return Err(AvifError::UnsupportedCodec);
            }

            let alpha_extents = alpha_item
                .extents
                .ok_or(AvifError::MissingBox("alpha ispe"))?;

            // Validate alpha dimensions match the primary image.
            // The AVIF spec (§4.8.2) requires them to be identical.
            if alpha_extents.width != extents.width || alpha_extents.height != extents.height {
                return Err(AvifError::InvalidBox);
            }

            let alpha_cfg = alpha_item
                .codec_config
                .as_ref()
                .ok_or(AvifError::MissingBox("alpha av2C"))?;

            let alpha_obu = self.assemble_obu(alpha_item)?;
            if alpha_obu.is_empty() {
                return Err(AvifError::MissingBox("alpha mdat"));
            }

            let alpha_picture = self.run_decoder(alpha_obu)?;

            // Validate decoded dimensions against the declared extents.
            if alpha_picture.p.w as u32 != extents.width
                || alpha_picture.p.h as u32 != extents.height
            {
                return Err(AvifError::InvalidBox);
            }

            let alpha_bpc =
                u8::try_from(alpha_picture.p.bpc).map_err(|_| AvifError::InvalidCodecConfig)?;
            let alpha_bps = alpha_picture.bytes_per_sample();
            if alpha_cfg.pixel_layout() != PixelLayout::I400
                || alpha_picture.p.layout != PixelLayout::I400
                || alpha_cfg.bits_per_component() != alpha_bpc
            {
                return Err(AvifError::InvalidCodecConfig);
            }

            let a_w = alpha_picture.p.w as usize;
            let a_h = alpha_picture.p.h as usize;
            let a_stride = alpha_picture.stride[0].unsigned_abs();

            // Alpha is encoded as a monochrome (I400) AV2 stream; the
            // transparency values live in the luma plane.
            let data = copy_plane(alpha_picture.plane_bytes(0), a_stride, a_h, a_w, alpha_bps)?;
            let alpha_out_stride = a_w.checked_mul(alpha_bps).ok_or(AvifError::InvalidBox)?;

            Some(AlphaPlane {
                data,
                stride: alpha_out_stride,
                bits_per_component: alpha_bpc,
            })
        } else {
            None
        };

        Ok(AvifImage {
            width: extents.width,
            height: extents.height,
            pixel_layout: layout,
            bits_per_component: bpc,
            color_info: self.container.primary_item().and_then(|i| i.color_info),
            content_light_level,
            pixel_aspect_ratio: self
                .container
                .primary_item()
                .and_then(|i| i.pixel_aspect_ratio),
            planes: [plane_y, plane_u, plane_v],
            strides: [out_y_stride, out_uv_stride],
            alpha,
            premultiplied_alpha,
            orientation: self
                .container
                .primary_item()
                .and_then(|i| i.orientation)
                .unwrap_or(Orientation::Normal),
            clean_aperture: self.container.primary_item().and_then(|i| i.clean_aperture),
            icc_profile: self
                .container
                .primary_item()
                .and_then(|i| i.icc_profile.clone()),
            picture: picture_arc,
        })
    }

    /// Return content light level metadata for the primary item, if present.
    ///
    /// This scans the AV2 OBU stream for HDR CLL metadata without decoding
    /// pixels. The same value is also copied to [`AvifImage::content_light_level`]
    /// after [`decode`](Self::decode).
    pub fn content_light_level(&self) -> Result<Option<ContentLightLevel>> {
        let item = self
            .container
            .primary_item()
            .ok_or(AvifError::MissingBox("pitm"))?;
        self.item_content_light_level(item)
    }

    /// Return a reference to the parsed container metadata.
    pub fn container(&self) -> &AvifContainer {
        &self.container
    }

    /// Return the parser/decoder settings used by this decoder.
    pub fn settings(&self) -> &AvifSettings {
        &self.settings
    }

    fn item_content_light_level(&self, item: &AvifItem) -> Result<Option<ContentLightLevel>> {
        let obu_data = self.assemble_obu(item)?;
        scan_content_light_level_from_obus(&obu_data)
    }

    /// Assemble a contiguous OBU byte stream from an item's `configOBUs` and
    /// `iloc` extents, subject to [`AvifSettings::max_obu_bytes`].
    fn assemble_obu(&self, item: &AvifItem) -> Result<Vec<u8>> {
        let cfg = item
            .codec_config
            .as_ref()
            .ok_or(AvifError::MissingBox("av2C"))?;

        let config_len = cfg.config_obus.len();
        let sample_len: u64 = item
            .iloc_extents
            .iter()
            .try_fold(0u64, |acc, &(_, len)| acc.checked_add(len))
            .ok_or(AvifError::InvalidBox)?;

        // FUZZ: cap total OBU length before allocating.
        let total_obu_len: u64 = (config_len as u64)
            .checked_add(sample_len)
            .ok_or(AvifError::InvalidBox)?;

        if self.settings.max_obu_bytes != 0 && total_obu_len > self.settings.max_obu_bytes as u64 {
            return Err(AvifError::LimitExceeded);
        }

        let mut obu_data: Vec<u8> = Vec::new();
        obu_data
            .try_reserve_exact(total_obu_len as usize)
            .map_err(|_| AvifError::OutOfMemory)?;
        if !cfg.config_obus.is_empty() {
            obu_data.extend_from_slice(&cfg.config_obus);
        }
        for &(abs_offset, ext_len) in &item.iloc_extents {
            // FUZZ: checked arithmetic for start+len on usize.
            let start = usize::try_from(abs_offset).map_err(|_| AvifError::ExtentOutOfBounds)?;
            let len = usize::try_from(ext_len).map_err(|_| AvifError::ExtentOutOfBounds)?;
            let end = start.checked_add(len).ok_or(AvifError::ExtentOutOfBounds)?;

            let slice = self
                .file
                .get(start..end)
                .ok_or(AvifError::ExtentOutOfBounds)?;
            obu_data.extend_from_slice(slice);
        }
        Ok(obu_data)
    }

    /// Open an AV2 decoder, feed it `obu_data`, drain one [`Picture`].
    fn run_decoder(&self, obu_data: Vec<u8>) -> Result<Picture> {
        let mut settings = self.settings.decoder_settings.clone();
        settings.run_decode = true;

        let mut decoder = Decoder::open(&settings).map_err(AvifError::DecodeError)?;
        decoder
            .send_data(Some(Data::wrap(obu_data)))
            .map_err(AvifError::DecodeError)?;

        // Signal end-of-stream, then drain one picture.
        let _ = decoder.send_data(None);

        match decoder.get_picture() {
            Ok(p) => Ok(p),
            Err(TealdustError::Again) => {
                let _ = decoder.send_data(None);
                match decoder.get_picture() {
                    Ok(p) => Ok(p),
                    Err(e) => Err(AvifError::DecodeError(e)),
                }
            }
            Err(e) => Err(AvifError::DecodeError(e)),
        }
    }
}

fn scan_content_light_level_from_obus(obu_data: &[u8]) -> Result<Option<ContentLightLevel>> {
    let mut pos = 0usize;
    let mut content_light_level = None;

    while pos < obu_data.len() {
        let (obu_len, leb_len) = read_leb128_usize(&obu_data[pos..])?;
        pos = pos.checked_add(leb_len).ok_or(AvifError::InvalidBox)?;

        let end = pos.checked_add(obu_len).ok_or(AvifError::InvalidBox)?;
        let body = obu_data.get(pos..end).ok_or(AvifError::InvalidBox)?;
        pos = end;

        if body.is_empty() {
            continue;
        }

        let mut gb = GetBits::new(body);
        let has_extension = gb.get_bit() != 0;
        let obu_type_raw = gb.get_bits(5);
        let _temporal_layer_id = gb.get_bits(2);

        if has_extension {
            let _multilayer_id = gb.get_bits(3);
            let _extension_layer_id = gb.get_bits(5);
        }

        if gb.has_error() {
            return Err(AvifError::InvalidCodecConfig);
        }

        if obu_type_raw == ObuType::Metadata as u32 {
            let meta_type = gb.get_uleb128();
            if gb.has_error() {
                return Err(AvifError::InvalidCodecConfig);
            }

            if meta_type == ObuMetaType::HdrCll as u32 {
                let cll = crate::obu::parse_cll(&mut gb);
                if gb.has_error() {
                    return Err(AvifError::InvalidCodecConfig);
                }
                content_light_level = Some(cll);
            }
        }
    }

    Ok(content_light_level)
}

fn read_leb128_usize(data: &[u8]) -> Result<(usize, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;

    for (i, &byte) in data.iter().take(8).enumerate() {
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            let len = i + 1;
            let value = usize::try_from(value).map_err(|_| AvifError::LimitExceeded)?;
            return Ok((value, len));
        }
        shift += 7;
    }

    Err(AvifError::InvalidBox)
}

fn copy_plane(
    plane: Option<&[u8]>,
    stride: usize,
    h: usize,
    w: usize,
    bps: usize,
) -> Result<Vec<u8>> {
    let Some(plane) = plane else {
        return Ok(Vec::new());
    };
    if h == 0 || w == 0 {
        return Ok(Vec::new());
    }

    let w_bytes = w.checked_mul(bps).ok_or(AvifError::InvalidBox)?;
    let total = h.checked_mul(w_bytes).ok_or(AvifError::InvalidBox)?;

    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| AvifError::OutOfMemory)?;
    for row in 0..h {
        let row_offset = row.checked_mul(stride).ok_or(AvifError::InvalidBox)?;
        let row_end = row_offset
            .checked_add(w_bytes)
            .ok_or(AvifError::InvalidBox)?;
        let row_slice = plane
            .get(row_offset..row_end)
            .ok_or(AvifError::InvalidBox)?;
        out.extend_from_slice(row_slice);
    }
    Ok(out)
}

/// Quick probe: returns `true` if `data` begins with an ISOBMFF `ftyp` box
/// carrying a recognised AVIF brand (`avif`, `avis`, or `av02`).
///
/// This is intentionally lenient — it only inspects the first 12 bytes.
pub fn probe_avif(data: &[u8]) -> bool {
    if data.len() < 12 {
        return false;
    }
    // Expect ftyp as first box.
    if &data[4..8] != b"ftyp" {
        return false;
    }
    let brand = [data[8], data[9], data[10], data[11]];
    is_avif_brand(&brand)
}
