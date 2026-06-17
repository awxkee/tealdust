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

/// A storage type for one pixel sample (`u8` for 8bpc, `u16` for 10/12bpc).
pub trait Pixel: Copy + Default + Send + Sync + Into<i32> + 'static {
    /// Compile-time storage width in bits (8 for `u8`, 16 for `u16`). This is
    /// NOT the coded bit depth — for HBD the coded depth (10/12) is carried at
    /// runtime by [`BitDepth::bitdepth_max`].
    const BITDEPTH: u8;
    /// Largest value representable in the storage type.
    const MAX: Self;

    fn from_i32(v: i32) -> Self;
    fn as_u16(self) -> u16;

    /// View a pixel slice as native-endian bytes.
    ///
    /// This centralizes the only representation cast needed by the decoder's
    /// legacy byte-oriented DSP entry points. It is safe for the built-in pixel
    /// types because `u8` and `u16` have no invalid bit patterns and the result
    /// borrows the original slice.
    fn slice_as_ne_bytes(samples: &[Self]) -> &[u8];

    /// Mutable byte view of a pixel slice.
    fn slice_as_ne_bytes_mut(samples: &mut [Self]) -> &mut [u8];

    /// Typed view of owned picture-plane storage. Implemented only by the
    /// built-in pixel types, so this is a safe enum match rather than a cast.
    fn slice_from_plane_storage(storage: &crate::picture::PlaneStorage) -> Option<&[Self]>;

    /// Mutable typed view of owned picture-plane storage.
    fn slice_from_plane_storage_mut(
        storage: &mut crate::picture::PlaneStorage,
    ) -> Option<&mut [Self]>;
}

impl Pixel for u8 {
    const BITDEPTH: u8 = 8;
    const MAX: u8 = 0xFF;

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self as u16
    }

    #[inline(always)]
    fn slice_as_ne_bytes(samples: &[Self]) -> &[u8] {
        samples
    }

    #[inline(always)]
    fn slice_as_ne_bytes_mut(samples: &mut [Self]) -> &mut [u8] {
        samples
    }

    #[inline(always)]
    fn slice_from_plane_storage(storage: &crate::picture::PlaneStorage) -> Option<&[Self]> {
        match storage {
            crate::picture::PlaneStorage::U8(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[inline(always)]
    fn slice_from_plane_storage_mut(
        storage: &mut crate::picture::PlaneStorage,
    ) -> Option<&mut [Self]> {
        match storage {
            crate::picture::PlaneStorage::U8(v) => Some(v.as_mut_slice()),
            _ => None,
        }
    }
}

impl Pixel for u16 {
    const BITDEPTH: u8 = 16;
    const MAX: u16 = 0xFFFF;

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }

    #[inline(always)]
    fn as_u16(self) -> u16 {
        self
    }

    #[inline(always)]
    fn slice_as_ne_bytes(samples: &[Self]) -> &[u8] {
        let len = std::mem::size_of_val(samples);
        // SAFETY: `u16` has no invalid bit patterns. The returned byte slice is
        // tied to the input lifetime and covers exactly the same allocation.
        unsafe { core::slice::from_raw_parts(samples.as_ptr() as *const u8, len) }
    }

    #[inline(always)]
    fn slice_as_ne_bytes_mut(samples: &mut [Self]) -> &mut [u8] {
        let len = std::mem::size_of_val(samples);
        // SAFETY: same allocation/lifetime as `samples`; `u16` permits all byte
        // patterns in native-endian storage.
        unsafe { core::slice::from_raw_parts_mut(samples.as_mut_ptr() as *mut u8, len) }
    }

    #[inline(always)]
    fn slice_from_plane_storage(storage: &crate::picture::PlaneStorage) -> Option<&[Self]> {
        match storage {
            crate::picture::PlaneStorage::U16(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[inline(always)]
    fn slice_from_plane_storage_mut(
        storage: &mut crate::picture::PlaneStorage,
    ) -> Option<&mut [Self]> {
        match storage {
            crate::picture::PlaneStorage::U16(v) => Some(v.as_mut_slice()),
            _ => None,
        }
    }
}

///
/// `BitDepth8` is a zero-sized type (the coded depth is always 8). `BitDepth16`
/// carries the coded bit depth at runtime (`bitdepth_max = (1 << bd) - 1`),
/// because a single `u16` storage type backs both 10- and 12-bit streams.
///
/// DSP kernels and the recon path are written once against this trait and
/// instantiated for both `BitDepth8` and `BitDepth16`; the `u8` instantiation
/// is byte-identical to the prior hard-coded `_8bpc` code.
pub trait BitDepth: Clone + Copy + Send + Sync + 'static {
    /// Sample storage type (`u8` or `u16`).
    type Pixel: Pixel;
    /// Intermediate/coefficient signed type wide enough for this depth.
    type Coef: Copy;

    /// Compile-time storage width in bits (8 or 16).
    const BPC: u8;

    /// Construct from the coded bit depth (8, 10 or 12). For `BitDepth8` the
    /// argument is ignored.
    fn new(bitdepth: u8) -> Self;

    /// Coded bit depth (8/10/12).
    fn bitdepth(&self) -> u8;

    /// `(1 << bitdepth) - 1` — the clip ceiling for reconstructed pixels.
    fn bitdepth_max(&self) -> i32;

    /// kernels (deblock thresholds, cdef clips, intermediate rounding).
    #[inline(always)]
    fn bitdepth_min_8(&self) -> i32 {
        self.bitdepth() as i32 - 8
    }

    /// Clip a reconstructed sample into `[0, bitdepth_max]`.
    #[inline(always)]
    fn pixel_clip(&self, v: i32) -> Self::Pixel {
        let max = self.bitdepth_max();
        Self::Pixel::from_i32(v.clamp(0, max))
    }
}

/// 8-bit reconstruction (`u8` samples, fixed `bitdepth_max = 255`).
#[derive(Clone, Copy, Default)]
pub struct BitDepth8;

impl BitDepth for BitDepth8 {
    type Pixel = u8;
    type Coef = i16;
    const BPC: u8 = 8;

    #[inline(always)]
    fn new(_bitdepth: u8) -> Self {
        BitDepth8
    }
    #[inline(always)]
    fn bitdepth(&self) -> u8 {
        8
    }
    #[inline(always)]
    fn bitdepth_max(&self) -> i32 {
        255
    }
}

/// High-bit-depth reconstruction (`u16` samples, runtime 10/12-bit `bitdepth`).
#[derive(Clone, Copy)]
pub struct BitDepth16 {
    bitdepth: u8,
}

impl BitDepth for BitDepth16 {
    type Pixel = u16;
    type Coef = i32;
    const BPC: u8 = 16;

    #[inline(always)]
    fn new(bitdepth: u8) -> Self {
        debug_assert!(bitdepth == 10 || bitdepth == 12);
        BitDepth16 { bitdepth }
    }
    #[inline(always)]
    fn bitdepth(&self) -> u8 {
        self.bitdepth
    }
    #[inline(always)]
    fn bitdepth_max(&self) -> i32 {
        (1 << self.bitdepth) - 1
    }
}
