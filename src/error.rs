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

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TealdustError {
    Eof,
    Again,
    /// Generic malformed-bitstream fallback kept for API compatibility and for
    /// call sites that have not yet been split into a more specific reason.
    InvalidData,
    /// Malformed OBU envelope or payload dispatch state.
    InvalidObu,
    /// Malformed sequence header syntax or unsupported header combination.
    InvalidSequenceHeader,
    /// Malformed frame header syntax or invalid frame-header-derived state.
    InvalidFrameHeader,
    /// Invalid tiling layout or tile-group header.
    InvalidTileInfo,
    /// Tile payload decoded to invalid block/coefficient/reconstruction state.
    InvalidTileData,
    /// Invalid, missing, or incompatible reference frame state.
    InvalidReferenceFrame,
    /// Invalid film-grain syntax/state.
    InvalidFilmGrainData,
    /// Invalid content-interpretation metadata syntax/state.
    InvalidContentInterpretation,
    /// Strict trailing-bit validation failed.
    InvalidTrailingBits,
    /// Decoder reached frame parsing before a sequence header was available.
    MissingSequenceHeader,
    /// Decoder reached tile/frame submission before a frame header was available.
    MissingFrameHeader,
    /// Frame-context/scratch/picture setup failed before tile decode started.
    FrameSetupFailed,
    /// Per-tile entropy/CDF setup failed before the main decode loop.
    CdfInitFailed,
    FrameTooLarge,
    InvalidParam,
    OutOfMemory,
}

impl fmt::Display for TealdustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "end of stream"),
            Self::Again => write!(f, "need more data"),
            Self::InvalidData => write!(f, "invalid or corrupt bitstream data"),
            Self::InvalidObu => write!(f, "invalid or corrupt OBU"),
            Self::InvalidSequenceHeader => write!(f, "invalid sequence header"),
            Self::InvalidFrameHeader => write!(f, "invalid frame header"),
            Self::InvalidTileInfo => write!(f, "invalid tile layout or tile header"),
            Self::InvalidTileData => write!(f, "invalid or corrupt tile data"),
            Self::InvalidReferenceFrame => write!(f, "invalid or missing reference frame"),
            Self::InvalidFilmGrainData => write!(f, "invalid film-grain data"),
            Self::InvalidContentInterpretation => {
                write!(f, "invalid content-interpretation metadata")
            }
            Self::InvalidTrailingBits => write!(f, "invalid trailing bits"),
            Self::MissingSequenceHeader => write!(f, "missing sequence header"),
            Self::MissingFrameHeader => write!(f, "missing frame header"),
            Self::FrameSetupFailed => write!(f, "frame setup failed"),
            Self::CdfInitFailed => write!(f, "CDF/tile entropy initialization failed"),
            Self::FrameTooLarge => write!(f, "frame dimensions exceed limit"),
            Self::InvalidParam => write!(f, "invalid parameter"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for TealdustError {}
