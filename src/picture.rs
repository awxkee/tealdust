use std::sync::Arc;
use std::sync::atomic::AtomicI32;

use crate::data::DataProps;
use crate::headers::{ContentLightLevel, FilmGrainData, FrameHeader, PixelLayout, SequenceHeader};

pub const PICTURE_ALIGNMENT: usize = 64;

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
#[derive(Default)]
pub enum PlaneStorage {
    #[default]
    Empty,
    U8(Vec<u8>),
    U16(Vec<u16>),
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
            PlaneStorage::U8(v) => v.as_mut_slice(),
            PlaneStorage::U16(v) => {
                <u16 as crate::pixel::Pixel>::slice_as_ne_bytes_mut(v.as_mut_slice())
            }
        }
    }

    #[inline]
    fn with_len_for_bpc(byte_len: usize, hbd: bool) -> Self {
        if byte_len == 0 {
            PlaneStorage::Empty
        } else if hbd {
            debug_assert_eq!(byte_len % core::mem::size_of::<u16>(), 0);
            PlaneStorage::U16(vec![0u16; byte_len / core::mem::size_of::<u16>()])
        } else {
            PlaneStorage::U8(vec![0u8; byte_len])
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
}

pub struct PictureAllocation {
    pub data: [PlaneStorage; 3],
    pub stride: [isize; 2],
}

pub struct DefaultPicAllocator;

impl Default for DefaultPicAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultPicAllocator {
    pub fn new() -> Self {
        Self
    }
}

impl PicAllocator for DefaultPicAllocator {
    fn alloc_picture(&self, p: &PictureParameters) -> Option<PictureAllocation> {
        let hbd = p.bpc > 8;
        let aligned_w = (p.w as usize + 127) & !127;
        let aligned_h = (p.h as usize + 127) & !127;
        let has_chroma = p.layout != PixelLayout::I400;
        let ss_ver = p.layout == PixelLayout::I420;
        let ss_hor = p.layout != PixelLayout::I444;

        let mut y_stride = (aligned_w << (hbd as usize)) as isize;
        let mut uv_stride = if has_chroma {
            y_stride >> (ss_hor as usize)
        } else {
            0
        };

        if y_stride & 1023 == 0 {
            y_stride += PICTURE_ALIGNMENT as isize;
        }
        if uv_stride & 1023 == 0 && has_chroma {
            uv_stride += PICTURE_ALIGNMENT as isize;
        }

        let y_sz = y_stride as usize * aligned_h;
        let uv_sz = uv_stride as usize * (aligned_h >> (ss_ver as usize));

        let y = PlaneStorage::with_len_for_bpc(y_sz, hbd);
        let u = if has_chroma {
            PlaneStorage::with_len_for_bpc(uv_sz, hbd)
        } else {
            PlaneStorage::Empty
        };
        let v = if has_chroma {
            PlaneStorage::with_len_for_bpc(uv_sz, hbd)
        } else {
            PlaneStorage::Empty
        };

        Some(PictureAllocation {
            data: [y, u, v],
            stride: [y_stride, uv_stride],
        })
    }

    fn release_picture(&self, _alloc: PictureAllocation) {
        // Dropping the owned PlaneStorage Vecs releases the picture. A reusable
        // allocator can still implement this trait later by keeping typed Vecs
        // in a pool, without exposing raw addresses in Picture.
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
    pub _visible: bool,
    pub _showable: bool,
    pub progress: Option<[AtomicI32; 3]>,
    pub _flags: u32,
}

impl ThreadPicture {
    pub fn new() -> Self {
        Self {
            p: Picture::new(),
            _visible: false,
            _showable: false,
            progress: None,
            _flags: 0,
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
