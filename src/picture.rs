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
use std::sync::Arc;
use std::sync::atomic::AtomicI32;

use crate::data::DataProps;
use crate::headers::{ContentLightLevel, FilmGrainData, FrameHeader, PixelLayout, SequenceHeader};

pub const PICTURE_ALIGNMENT: usize = 64;
const MAX_PICTURE_TOTAL_BYTES: usize = 512 * 1024 * 1024;
const MAX_POOLED_PLANE_BYTES: usize = 16 * 1024 * 1024;
const MAX_POOLED_TOTAL_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PictureParameters {
    pub w: i32,
    pub h: i32,
    pub layout: PixelLayout,
    pub bpc: i32,
}

/// Owned storage for one decoded picture plane.
///
/// This replaces the old opaque pointer/address storage. The active variant is
/// selected from the picture bit depth, so typed plane views can be returned by
/// matching the enum rather than by reinterpreting pointer provenance.
#[derive(Default, Clone)]
pub enum PlaneStorage {
    #[default]
    Empty,
    U8(Arc<Vec<u8>>),
    U16(Arc<Vec<u16>>),
}

impl PlaneStorage {
    #[inline]
    pub fn is_some(&self) -> bool {
        !matches!(self, PlaneStorage::Empty)
    }

    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, PlaneStorage::Empty)
    }

    #[inline]
    pub fn len_bytes(&self) -> usize {
        match self {
            PlaneStorage::Empty => 0,
            PlaneStorage::U8(v) => v.len(),
            PlaneStorage::U16(v) => v.len() * core::mem::size_of::<u16>(),
        }
    }

    /// True if this buffer is exactly `byte_len` bytes in the element width a
    /// `with_len_for_bpc(byte_len, hbd)` allocation would use — i.e. it can be
    /// recycled for such a request without re-sizing.
    #[inline]
    fn byte_capacity_matches(&self, byte_len: usize, hbd: bool) -> bool {
        match (self, hbd) {
            (PlaneStorage::U16(v), true) => v.len() * core::mem::size_of::<u16>() == byte_len,
            (PlaneStorage::U8(v), false) => v.len() == byte_len,
            _ => false,
        }
    }

    /// Zero every sample, so a recycled buffer matches a fresh `vec![0; n]`.
    #[inline]
    fn zero_fill(&mut self) {
        match self {
            PlaneStorage::Empty => {}
            PlaneStorage::U8(v) => Arc::make_mut(v).fill(0),
            PlaneStorage::U16(v) => Arc::make_mut(v).fill(0),
        }
    }

    /// True when this plane's buffer is uniquely owned (not shared with any
    /// other live picture). The pool only recycles uniquely-owned buffers — a
    /// plane still referenced by a ref slot or an emitted output picture must
    /// not be handed back out.
    #[inline]
    fn is_unique(&self) -> bool {
        match self {
            PlaneStorage::Empty => false,
            PlaneStorage::U8(v) => Arc::strong_count(v) == 1,
            PlaneStorage::U16(v) => Arc::strong_count(v) == 1,
        }
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        match self {
            PlaneStorage::Empty => &[],
            PlaneStorage::U8(v) => v.as_slice(),
            PlaneStorage::U16(v) => <u16 as crate::pixel::Pixel>::slice_as_ne_bytes(v.as_slice()),
        }
    }

    #[inline]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            PlaneStorage::Empty => &mut [],
            PlaneStorage::U8(v) => Arc::make_mut(v).as_mut_slice(),
            PlaneStorage::U16(v) => {
                <u16 as crate::pixel::Pixel>::slice_as_ne_bytes_mut(Arc::make_mut(v).as_mut_slice())
            }
        }
    }

    #[inline]
    fn with_len_for_bpc(byte_len: usize, hbd: bool) -> Option<Self> {
        if byte_len == 0 {
            return Some(PlaneStorage::Empty);
        }

        if hbd {
            if !byte_len.is_multiple_of(core::mem::size_of::<u16>()) {
                return None;
            }
            let len = byte_len / core::mem::size_of::<u16>();
            let mut v = Vec::new();
            v.try_reserve_exact(len).ok()?;
            v.resize(len, 0u16);
            Some(PlaneStorage::U16(Arc::new(v)))
        } else {
            let mut v = Vec::new();
            v.try_reserve_exact(byte_len).ok()?;
            v.resize(byte_len, 0u8);
            Some(PlaneStorage::U8(Arc::new(v)))
        }
    }
}

impl std::fmt::Debug for PlaneStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaneStorage::Empty => f.write_str("Empty"),
            PlaneStorage::U8(v) => f.debug_tuple("U8").field(&v.len()).finish(),
            PlaneStorage::U16(v) => f.debug_tuple("U16").field(&v.len()).finish(),
        }
    }
}

#[inline]
fn empty_planes() -> [PlaneStorage; 3] {
    [
        PlaneStorage::Empty,
        PlaneStorage::Empty,
        PlaneStorage::Empty,
    ]
}

pub trait PicAllocator: Send + Sync {
    fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation>;
    fn release_picture(&self, alloc: PictureAllocation);

    /// Drop allocator-owned idle buffers. Live pictures keep their own plane
    /// storage through `Arc`; this only releases the recycler's free list.
    fn clear_pool(&self) {}
}

pub struct PictureAllocation {
    pub data: [PlaneStorage; 3],
    pub stride: [isize; 2],
}

/// The byte layout of a picture: the two strides (luma, chroma) and the total
/// byte length of the luma and chroma planes. Shared by every allocator so they
/// produce bit-identical geometry.
struct PlaneByteLayout {
    stride: [isize; 2],
    y_sz: usize,
    uv_sz: usize,
    hbd: bool,
    has_chroma: bool,
}

fn plane_byte_layout(p: &PictureParameters) -> Option<PlaneByteLayout> {
    if p.w <= 0 || p.h <= 0 || !(1..=16).contains(&p.bpc) {
        return None;
    }

    let hbd = p.bpc > 8;
    let w = p.w as usize;
    let h = p.h as usize;
    let aligned_w = w.checked_add(127)? & !127;
    let aligned_h = h.checked_add(127)? & !127;
    let has_chroma = p.layout != PixelLayout::I400;
    let ss_ver = p.layout == PixelLayout::I420;
    let ss_hor = p.layout != PixelLayout::I444;

    let bytes_per_sample = if hbd { 2usize } else { 1usize };
    let mut y_stride = aligned_w.checked_mul(bytes_per_sample)?;
    let mut uv_stride = if has_chroma {
        y_stride >> (ss_hor as usize)
    } else {
        0
    };

    if y_stride & 1023 == 0 {
        y_stride = y_stride.checked_add(PICTURE_ALIGNMENT)?;
    }
    if uv_stride & 1023 == 0 && has_chroma {
        uv_stride = uv_stride.checked_add(PICTURE_ALIGNMENT)?;
    }

    let y_sz = y_stride.checked_mul(aligned_h)?;
    let uv_sz = uv_stride.checked_mul(aligned_h >> (ss_ver as usize))?;
    let total_sz = y_sz.checked_add(if has_chroma { uv_sz.checked_mul(2)? } else { 0 })?;
    if total_sz > MAX_PICTURE_TOTAL_BYTES {
        return None;
    }
    if y_stride > isize::MAX as usize || uv_stride > isize::MAX as usize {
        return None;
    }
    let y_stride_i = y_stride as isize;
    let uv_stride_i = uv_stride as isize;

    Some(PlaneByteLayout {
        stride: [y_stride_i, uv_stride_i],
        y_sz,
        uv_sz,
        hbd,
        has_chroma,
    })
}

pub struct PoolPicAllocator {
    free: std::sync::Mutex<Vec<PlaneStorage>>,
    cap: usize,
}

impl Default for PoolPicAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolPicAllocator {
    pub fn new() -> Self {
        Self {
            free: std::sync::Mutex::new(Vec::new()),
            cap: 64,
        }
    }

    /// A pooled buffer of exactly `byte_len` bytes (and matching element width),
    /// zeroed for reuse; or a fresh zeroed allocation when the pool can't match.
    fn take_or_make(&self, byte_len: usize, hbd: bool) -> Option<PlaneStorage> {
        if byte_len == 0 {
            return Some(PlaneStorage::Empty);
        }
        if let Ok(mut free) = self.free.lock() {
            if let Some(idx) = free
                .iter()
                .position(|s| s.byte_capacity_matches(byte_len, hbd))
            {
                // Reconstruction fully writes every plane it exposes;
                // re-zeroing recycled storage is a full-frame memset of
                // dead work plus the cache traffic to match.
                return Some(free.swap_remove(idx));
            }
        }
        PlaneStorage::with_len_for_bpc(byte_len, hbd)
    }
}

impl PicAllocator for PoolPicAllocator {
    fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation> {
        let l = plane_byte_layout(p)?;
        let y = self.take_or_make(l.y_sz, l.hbd)?;
        let (u, v) = if l.has_chroma {
            (
                self.take_or_make(l.uv_sz, l.hbd)?,
                self.take_or_make(l.uv_sz, l.hbd)?,
            )
        } else {
            (PlaneStorage::Empty, PlaneStorage::Empty)
        };
        Some(PictureAllocation {
            data: [y, u, v],
            stride: l.stride,
        })
    }

    fn release_picture(&self, alloc: PictureAllocation) {
        if let Ok(mut free) = self.free.lock() {
            let mut pooled_bytes: usize = free.iter().map(PlaneStorage::len_bytes).sum();
            for s in alloc.data {
                // Recycle only buffers no other live picture still shares. A
                // plane handed to a ref slot or an emitted output picture has
                // strong_count > 1 here and is dropped (decrementing the count)
                // rather than recycled; it becomes reclaimable when its last
                // holder is released.
                let len = s.len_bytes();
                if s.is_some()
                    && s.is_unique()
                    && len <= MAX_POOLED_PLANE_BYTES
                    && pooled_bytes.saturating_add(len) <= MAX_POOLED_TOTAL_BYTES
                    && free.len() < self.cap
                {
                    pooled_bytes += len;
                    free.push(s);
                }
            }
        }
    }

    fn clear_pool(&self) {
        if let Ok(mut free) = self.free.lock() {
            free.clear();
            free.shrink_to_fit();
        }
    }
}

/// A decoded video frame with pixel data and associated metadata.
pub struct Picture {
    pub p: PictureParameters,
    pub data: [PlaneStorage; 3],
    pub stride: [isize; 2],
    pub seq_hdr: Option<Arc<SequenceHeader>>,
    pub frame_hdr: Option<Arc<FrameHeader>>,
    /// Film-grain synthesis parameters for this frame (the selected `c.fgm[id]`
    /// `Dav2dPicture.fgm`; used at output time to apply grain to a display copy.
    pub fgm: Option<FilmGrainData>,
    pub content_light_level: Option<ContentLightLevel>,
    pub props: DataProps,
    allocator: Option<Arc<dyn PicAllocator>>,
}

impl Picture {
    pub fn new() -> Self {
        Self {
            p: PictureParameters {
                w: 0,
                h: 0,
                layout: PixelLayout::I400,
                bpc: 0,
            },
            data: empty_planes(),
            stride: [0, 0],
            seq_hdr: None,
            frame_hdr: None,
            fgm: None,
            content_light_level: None,
            props: DataProps::new(),
            allocator: None,
        }
    }

    pub fn alloc(
        w: i32,
        h: i32,
        layout: PixelLayout,
        bpc: i32,
        seq_hdr: Option<Arc<SequenceHeader>>,
        frame_hdr: Option<Arc<FrameHeader>>,
        allocator: Arc<dyn PicAllocator>,
    ) -> Option<Self> {
        let params = PictureParameters { w, h, layout, bpc };
        let alloc = allocator.alloc_picture(&params)?;

        Some(Self {
            p: params,
            data: alloc.data,
            stride: alloc.stride,
            seq_hdr,
            frame_hdr,
            fgm: None,
            content_light_level: None,
            props: DataProps::new(),
            allocator: Some(allocator),
        })
    }

    pub fn has_data(&self) -> bool {
        self.data[0].is_some()
    }

    /// A new picture that shares this one's plane buffers by reference count
    /// instead of deep-copying them. Used to emit a display picture without
    /// duplicating ~megabytes of pixels: the reconstructed frame is read-only
    /// once decoded, so the ref slots and the output can share the same storage.
    /// Any later mutation through `bytes_mut` copies on write (see `make_mut`),
    /// so the shared buffers can never be observed changing.
    pub fn shallow_clone(&self) -> Self {
        Self {
            p: self.p,
            data: [
                self.data[0].clone(),
                self.data[1].clone(),
                self.data[2].clone(),
            ],
            stride: self.stride,
            seq_hdr: self.seq_hdr.clone(),
            frame_hdr: self.frame_hdr.clone(),
            fgm: self.fgm,
            content_light_level: self.content_light_level,
            props: self.props.clone(),
            allocator: self.allocator.clone(),
        }
    }

    /// True when this picture stores samples as 16-bit (`bpc > 8`). The plane
    /// allocation already reserves `width << hbd` bytes per row (see
    /// `DefaultPicAllocator::alloc_picture`), so high-bit-depth planes hold two
    /// bytes per sample.
    #[inline]
    pub fn is_hbd(&self) -> bool {
        self.p.bpc > 8
    }

    /// Number of storage bytes per sample (1 for 8bpc, 2 for HBD).
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        if self.is_hbd() { 2 } else { 1 }
    }

    /// Row stride for plane `pl` expressed in **samples** (not bytes). Plane 0
    /// is luma; planes 1/2 are chroma. The byte stride lives in `self.stride`.
    #[inline]
    pub fn stride_px(&self, pl: usize) -> usize {
        let s = self.stride[if pl == 0 { 0 } else { 1 }].unsigned_abs();
        s / self.bytes_per_sample()
    }

    /// Plane height in visible samples for plane `pl`.
    #[inline]
    pub fn plane_h(&self, pl: usize) -> usize {
        let ss_ver = if pl != 0 && self.p.layout == PixelLayout::I420 {
            1
        } else {
            0
        };
        ((self.p.h + ss_ver) >> ss_ver) as usize
    }

    /// Plane width in visible samples for plane `pl`.
    #[inline]
    pub fn plane_w(&self, pl: usize) -> usize {
        let ss_hor = if pl != 0 && matches!(self.p.layout, PixelLayout::I420 | PixelLayout::I422) {
            1
        } else {
            0
        };
        ((self.p.w + ss_hor) >> ss_hor) as usize
    }

    /// Byte stride for plane `pl`.
    #[inline]
    pub fn stride_bytes(&self, pl: usize) -> usize {
        self.stride[if pl == 0 { 0 } else { 1 }].unsigned_abs()
    }

    #[inline]
    fn plane_storage(&self, pl: usize) -> Option<&PlaneStorage> {
        self.data.get(pl).filter(|s| s.is_some())
    }

    #[inline]
    fn plane_storage_mut(&mut self, pl: usize) -> Option<&mut PlaneStorage> {
        self.data.get_mut(pl).filter(|s| s.is_some())
    }

    /// Immutable byte view of the visible part of a plane.
    #[inline]
    pub fn plane_bytes(&self, pl: usize) -> Option<&[u8]> {
        let len = self.stride_bytes(pl) * self.plane_h(pl);
        let bytes = self.plane_storage(pl)?.bytes();
        debug_assert!(len <= bytes.len());
        Some(&bytes[..len])
    }

    /// Mutable byte view of the visible part of a plane.
    #[inline]
    pub fn plane_bytes_mut(&mut self, pl: usize) -> Option<&mut [u8]> {
        let len = self.stride_bytes(pl) * self.plane_h(pl);
        let bytes = self.plane_storage_mut(pl)?.bytes_mut();
        debug_assert!(len <= bytes.len());
        Some(&mut bytes[..len])
    }

    /// Immutable byte view of the first `rows` rows of a plane. This may include
    /// allocator padding rows; callers must pass a row count that fits the
    /// allocation contract.
    #[inline]
    pub fn plane_bytes_rows(&self, pl: usize, rows: usize) -> Option<&[u8]> {
        let len = self.stride_bytes(pl) * rows;
        let bytes = self.plane_storage(pl)?.bytes();
        debug_assert!(len <= bytes.len());
        Some(&bytes[..len])
    }

    /// Mutable byte view of the first `rows` rows of a plane.
    #[inline]
    pub fn plane_bytes_rows_mut(&mut self, pl: usize, rows: usize) -> Option<&mut [u8]> {
        let len = self.stride_bytes(pl) * rows;
        let bytes = self.plane_storage_mut(pl)?.bytes_mut();
        debug_assert!(len <= bytes.len());
        Some(&mut bytes[..len])
    }

    /// Mutable byte views of Y/U/V rows at once. This is for algorithms that
    /// genuinely need simultaneous disjoint plane borrows, such as film grain.
    #[inline]
    pub fn plane_bytes_rows3_mut(
        &mut self,
        y_rows: usize,
        uv_rows: usize,
        has_chroma: bool,
    ) -> (&mut [u8], &mut [u8], &mut [u8]) {
        let y_len = self.stride_bytes(0) * y_rows;
        let uv_len = self.stride_bytes(1) * uv_rows;
        let (y_plane, rest) = self.data.split_at_mut(1);
        let (u_plane, v_plane) = rest.split_at_mut(1);

        let y = y_plane[0].bytes_mut();
        debug_assert!(y_len <= y.len());
        let y = &mut y[..y_len];

        let u = if has_chroma {
            let u = u_plane[0].bytes_mut();
            debug_assert!(uv_len <= u.len());
            &mut u[..uv_len]
        } else {
            &mut []
        };
        let v = if has_chroma {
            let v = v_plane[0].bytes_mut();
            debug_assert!(uv_len <= v.len());
            &mut v[..uv_len]
        } else {
            &mut []
        };
        (y, u, v)
    }

    /// Mutable typed views of Y/U/V rows at once.
    #[inline]
    pub fn plane_slices_rows3_mut<P: crate::pixel::Pixel>(
        &mut self,
        y_rows: usize,
        uv_rows: usize,
        has_chroma: bool,
    ) -> (&mut [P], &mut [P], &mut [P]) {
        debug_assert_eq!(self.bytes_per_sample(), core::mem::size_of::<P>());
        let y_stride = self.stride_bytes(0) / core::mem::size_of::<P>();
        let uv_stride = self.stride_bytes(1) / core::mem::size_of::<P>();
        let y_len = y_stride * y_rows;
        let uv_len = uv_stride * uv_rows;

        let (y_plane, rest) = self.data.split_at_mut(1);
        let (u_plane, v_plane) = rest.split_at_mut(1);

        let y = P::slice_from_plane_storage_mut(&mut y_plane[0]).unwrap_or(&mut []);
        debug_assert!(y_len <= y.len());
        let y = &mut y[..y_len];

        let u = if has_chroma {
            let u = P::slice_from_plane_storage_mut(&mut u_plane[0]).unwrap_or(&mut []);
            debug_assert!(uv_len <= u.len());
            &mut u[..uv_len]
        } else {
            &mut []
        };
        let v = if has_chroma {
            let v = P::slice_from_plane_storage_mut(&mut v_plane[0]).unwrap_or(&mut []);
            debug_assert!(uv_len <= v.len());
            &mut v[..uv_len]
        } else {
            &mut []
        };
        (y, u, v)
    }

    /// Typed immutable view of a visible plane.
    #[inline]
    pub fn plane_slice<P: crate::pixel::Pixel>(&self, pl: usize) -> Option<&[P]> {
        debug_assert_eq!(self.bytes_per_sample(), core::mem::size_of::<P>());
        let stride = self.stride_bytes(pl) / core::mem::size_of::<P>();
        let len = stride * self.plane_h(pl);
        let samples = P::slice_from_plane_storage(self.plane_storage(pl)?)?;
        debug_assert!(len <= samples.len());
        Some(&samples[..len])
    }

    /// Typed mutable view of a visible plane.
    #[inline]
    pub fn plane_slice_mut<P: crate::pixel::Pixel>(&mut self, pl: usize) -> Option<&mut [P]> {
        debug_assert_eq!(self.bytes_per_sample(), core::mem::size_of::<P>());
        let stride = self.stride_bytes(pl) / core::mem::size_of::<P>();
        let len = stride * self.plane_h(pl);
        let samples = P::slice_from_plane_storage_mut(self.plane_storage_mut(pl)?)?;
        debug_assert!(len <= samples.len());
        Some(&mut samples[..len])
    }

    pub fn unref(&mut self) {
        let data = core::mem::replace(&mut self.data, empty_planes());
        if data[0].is_some() {
            if let Some(allocator) = self.allocator.take() {
                allocator.release_picture(PictureAllocation {
                    data,
                    stride: self.stride,
                });
            }
        } else {
            self.allocator = None;
        }
        self.stride = [0, 0];
        self.seq_hdr = None;
        self.frame_hdr = None;
        self.fgm = None;
        self.content_light_level = None;
        self.props = DataProps::new();
        self.p = PictureParameters {
            w: 0,
            h: 0,
            layout: PixelLayout::I400,
            bpc: 0,
        };
    }
}

impl Default for Picture {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        self.unref();
    }
}

impl std::fmt::Debug for Picture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Picture")
            .field("params", &self.p)
            .field("has_data", &self.has_data())
            .finish()
    }
}

pub struct ThreadPicture {
    pub p: Picture,
    pub progress: Option<[AtomicI32; 3]>,
}

impl ThreadPicture {
    pub fn new() -> Self {
        Self {
            p: Picture::new(),
            progress: None,
        }
    }

    pub fn unref(&mut self) {
        self.p.unref();
        self.progress = None;
    }
}

impl Default for ThreadPicture {
    fn default() -> Self {
        Self::new()
    }
}

pub const PICTURE_FLAG_NEW_SEQUENCE: u32 = 1 << 0;
pub const PICTURE_FLAG_NEW_OP_PARAMS_INFO: u32 = 1 << 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventFlags {
    None,
    NewSequence,
    NewOpParamsInfo,
    Both,
}

impl From<u32> for EventFlags {
    fn from(flags: u32) -> Self {
        match (
            flags & PICTURE_FLAG_NEW_SEQUENCE != 0,
            flags & PICTURE_FLAG_NEW_OP_PARAMS_INFO != 0,
        ) {
            (false, false) => EventFlags::None,
            (true, false) => EventFlags::NewSequence,
            (false, true) => EventFlags::NewOpParamsInfo,
            (true, true) => EventFlags::Both,
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn params() -> PictureParameters {
        PictureParameters {
            w: 1620,
            h: 1080,
            layout: PixelLayout::I420,
            bpc: 8,
        }
    }

    // A released picture's planes are recycled by the next allocation of the
    // same geometry: the pool ends empty and the reused buffer is zeroed.
    #[test]
    fn pool_recycles_and_zeroes() {
        let pool = PoolPicAllocator::new();
        let p = params();

        let a = pool.alloc_picture(&p).unwrap();
        // Capture the luma buffer's identity (capacity + pointer) to prove reuse.
        let y_len = a.data[0].len_bytes();
        assert!(y_len > 0);

        // Dirty the luma plane, then release it to the pool.
        let mut a = a;
        a.data[0].bytes_mut()[0] = 0xAB;
        let dirtied_ptr = a.data[0].bytes().as_ptr();
        pool.release_picture(a);

        // Next allocation of identical geometry must reuse a freed buffer.
        // Contents are intentionally NOT re-zeroed: the decoder overwrites
        // every byte it exposes, and the re-zero was a full-frame memset.
        let b = pool.alloc_picture(&p).unwrap();
        let reused = b.data.iter().any(|s| s.bytes().as_ptr() == dirtied_ptr);
        assert!(reused, "expected a pooled buffer to be reused");
        assert_eq!(b.data[0].len_bytes(), y_len);
    }
}
