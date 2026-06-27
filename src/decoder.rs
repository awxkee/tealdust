use std::sync::atomic::{AtomicBool, Ordering};

use crate::data::Data;
use crate::error::TealdustError;
use crate::internal::DecoderContext;
use crate::obu;
use crate::picture::{Picture, ThreadPicture};

pub const MAX_THREADS: u32 = 256;
pub const MAX_FRAME_DELAY: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Which in-loop filters to apply during decoding.
#[non_exhaustive]
#[derive(Default)]
pub enum InloopFilterType {
    None = 0,
    Deblock = 1,
    Cdef = 2,
    Restoration = 4,
    Wiener = 8,
    Gdf = 16,
    #[default]
    All = 31,
}

impl InloopFilterType {
    /// (DEBLOCK=1<<0, CDEF=1<<1, CCSO=1<<2, WIENER=1<<3, GDF=1<<4). The enum's
    /// numeric repr already matches these C bits; bit 2 is published as
    pub(crate) fn to_flags(self) -> u32 {
        self as u8 as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
/// Which frame types to decode.
#[non_exhaustive]
#[derive(Default)]
pub enum DecodeFrameType {
    #[default]
    All = 0,
    Reference = 1,
    Intra = 2,
    Key = 3,
}

/// Decoder configuration. Use `Settings::default()` for sensible defaults.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Number of worker threads (tile-row parallel decode). 0 = single-threaded.
    pub n_threads: u32,
    /// Maximum frame delay for pipelining. 0 = auto based on thread count.
    pub max_frame_delay: u32,
    /// Apply film grain synthesis to decoded output.
    pub apply_grain: bool,
    /// Scalability operating point index (0–31).
    pub operating_point: u32,
    /// Output all temporal/spatial layers.
    pub all_layers: bool,
    /// Maximum frame size in pixels (width × height). 0 = unlimited.
    pub frame_size_limit: u32,
    /// Abort on spec-violating bitstreams instead of best-effort.
    pub strict_std_compliance: bool,
    /// Output frames not marked for display.
    pub output_invisible_frames: bool,
    /// Which in-loop filters to apply.
    pub inloop_filters: InloopFilterType,
    /// Which frame types to decode.
    pub decode_frame_type: DecodeFrameType,
    /// Bring-up gate: actually run reconstruction (intra only so far) and emit
    /// pictures. Default off while recon/filters are incomplete; enabled by the
    /// conformance harness. Will become unconditional once decode is complete.
    pub run_decode: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            n_threads: 0,
            max_frame_delay: 0,
            apply_grain: true,
            operating_point: 0,
            all_layers: true,
            frame_size_limit: 0,
            strict_std_compliance: false,
            output_invisible_frames: false,
            inloop_filters: InloopFilterType::All,
            decode_frame_type: DecodeFrameType::All,
            run_decode: false,
        }
    }
}

fn get_num_threads(s: &Settings) -> (u32, u32) {
    #[rustfmt::skip]
    static FC_LUT: [u8; 49] = [
        1,
        2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    ];

    let n_tc = s.n_threads.clamp(1, MAX_THREADS);

    let n_fc = if s.max_frame_delay > 0 {
        s.max_frame_delay.min(n_tc)
    } else if n_tc < 50 {
        FC_LUT[(n_tc - 1) as usize] as u32
    } else {
        8
    };

    (n_tc, n_fc)
}

pub fn get_frame_delay(s: &Settings) -> Result<u32, TealdustError> {
    if s.n_threads > MAX_THREADS || s.max_frame_delay > MAX_FRAME_DELAY {
        return Err(TealdustError::InvalidParam);
    }
    let (_, n_fc) = get_num_threads(s);
    Ok(n_fc)
}

struct OutputQueue {
    pic: ThreadPicture,
}

/// AV2 bitstream decoder.
///
/// Feed compressed OBU data with [`send_data`](Self::send_data), then
/// pull decoded frames with [`get_picture`](Self::get_picture).
pub struct Decoder {
    n_tc: u32,
    n_fc: u32,

    ctx: DecoderContext,

    input: Data,
    drain: bool,
    flush: AtomicBool,

    dpb: Vec<OutputQueue>,
    dpb_in: usize,
    dpb_out: usize,
    dpb_sz: usize,
    /// POC (frame_offset) of the most recently appended output frame, mirroring
    /// deferred `show_implicit` reference frames in display order.
    dpb_poc: u8,
}

impl Decoder {
    /// Create a new decoder with the given settings.
    pub fn open(s: &Settings) -> Result<Self, TealdustError> {
        if s.n_threads > MAX_THREADS || s.max_frame_delay > MAX_FRAME_DELAY {
            return Err(TealdustError::InvalidParam);
        }
        if s.operating_point > 31 {
            return Err(TealdustError::InvalidParam);
        }

        let (n_tc, n_fc) = get_num_threads(s);

        let dpb_sz = n_fc as usize + 16;
        let mut dpb = Vec::with_capacity(dpb_sz);
        for _ in 0..dpb_sz {
            dpb.push(OutputQueue {
                pic: ThreadPicture::new(),
            });
        }

        let ctx = DecoderContext {
            seq_hdr: None,
            frame_hdr: None,
            tile: Vec::new(),
            n_tile_data: 0,
            n_tiles: 0,
            refs: Default::default(),
            content_light: None,
            mastering_display: None,
            ci: None,
            fgm: Default::default(),
            apply_grain: s.apply_grain,
            operating_point_idc: 0,
            max_spatial_id: 0,
            frame_size_limit: s.frame_size_limit,
            strict_std_compliance: s.strict_std_compliance,
            inloop_filters: s.inloop_filters.to_flags(),
            run_decode: s.run_decode,
            frame_out: Vec::new(),
            n_tc,
            pool: if n_tc >= 2 {
                Some(crate::mtpool::ThreadPool::new((n_tc - 1) as usize))
            } else {
                None
            },
            pic_allocator: std::sync::Arc::new(crate::picture::PoolPicAllocator::new()),
            fc: crate::internal::FrameContext::default(),
        };

        Ok(Self {
            n_tc,
            n_fc,
            ctx,
            input: Data::new(),
            drain: false,
            flush: AtomicBool::new(false),
            dpb,
            dpb_in: 0,
            dpb_out: 0,
            dpb_sz,
            dpb_poc: 0,
        })
    }

    /// Feed compressed data to the decoder. Pass `None` to signal end-of-stream.
    ///
    /// Returns `Err(Again)` if the decoder hasn't consumed previous data yet;
    /// call `get_picture` to drain output before sending more.
    pub fn send_data(&mut self, data: Option<Data>) -> Result<(), TealdustError> {
        match data {
            None => {
                self.drain = true;
                Ok(())
            }
            Some(d) => {
                if self.drain {
                    return Err(TealdustError::Eof);
                }
                if d.is_empty() || d.len() > usize::MAX / 2 {
                    return Err(TealdustError::InvalidParam);
                }
                if self.input.has_data() {
                    return Err(TealdustError::Again);
                }
                self.input = d;
                Ok(())
            }
        }
    }

    /// Retrieve a decoded picture from the output queue.
    ///
    /// Returns `Err(Again)` when no picture is available yet (send more data).
    /// Returns `Err(Eof)` when the stream has been fully drained.
    pub fn get_picture(&mut self) -> Result<Picture, TealdustError> {
        self.gen_picture()?;

        if self.drain {
            self.queue_flush();
        }

        self.output_image()
    }

    fn output_picture_ready(&self) -> bool {
        if self.dpb_out == self.dpb_in {
            return false;
        }
        true
    }

    fn gen_picture(&mut self) -> Result<(), TealdustError> {
        if self.output_picture_ready() {
            return Ok(());
        }

        while !self.input.is_empty() {
            let data = match self.input.data() {
                Some(d) => d,
                None => break,
            };
            match obu::parse_obus(&mut self.ctx, data) {
                Ok(consumed) => {
                    if consumed > self.input.len() {
                        self.input.unref();
                        return Err(TealdustError::InvalidData);
                    }
                    self.input.consume(consumed);
                    if self.input.is_empty() {
                        self.input.unref();
                    }
                    // Frames reconstructed during parsing: enqueue all of them in
                    // decode order (a single parse_obus call may decode several).
                    // Drain directly into the output ring; collecting into a temporary
                    // Vec added one avoidable allocation on the single-thread path.
                    let dpb = &mut self.dpb;
                    let dpb_sz = self.dpb_sz;
                    let dpb_in = &mut self.dpb_in;
                    let dpb_poc = &mut self.dpb_poc;
                    for pic in self.ctx.frame_out.drain(..) {
                        // recently queued frame so end-of-stream queue_flush can
                        // re-display deferred show_implicit frames in order.
                        if let Some(fh) = pic.frame_hdr.as_ref() {
                            *dpb_poc = fh.frame_offset;
                        }
                        dpb[*dpb_in].pic.p = pic;
                        *dpb_in += 1;
                        if *dpb_in == dpb_sz {
                            *dpb_in = 0;
                        }
                    }
                }
                Err(_e) => {
                    self.input.unref();
                    return Err(TealdustError::InvalidData);
                }
            }

            if self.output_picture_ready() {
                break;
            }
        }

        Ok(())
    }

    fn output_image(&mut self) -> Result<Picture, TealdustError> {
        if self.dpb_in == self.dpb_out {
            if !self.drain {
                return Err(TealdustError::Again);
            }
            self.drain = false;
            return Err(TealdustError::Eof);
        }

        let q = &mut self.dpb[self.dpb_out];
        let mut pic = Picture::new();
        std::mem::swap(&mut pic, &mut q.pic.p);
        q.pic.unref();

        self.dpb_out += 1;
        if self.dpb_out == self.dpb_sz {
            self.dpb_out = 0;
        }

        // Film grain is display-only: it must not feed inter prediction, so it is
        // applied to a fresh output copy here (the DPB/reference copy stays
        // The grain synthesis + base copy are parallelised across `n_tc` threads
        // (`n_tc == 1` keeps the byte-identical sequential path).
        if self.ctx.apply_grain && crate::decode::picture_has_grain(&pic) {
            let grained = crate::decode::apply_grain_to_picture_mt(
                &pic,
                self.n_tc,
                self.ctx.pool.as_ref(),
                self.ctx.pic_allocator.clone(),
            );
            pic.unref();
            return Ok(grained);
        }

        Ok(pic)
    }

    ///
    /// Frames coded with `show_implicit` are not displayed at decode time; they
    /// are held in the reference store and emitted in display order once the
    /// stream drains. This re-queues each such reference whose POC is later than
    /// the last-displayed POC (`dpb_poc`), smallest-first, exactly once per slot.
    fn queue_flush(&mut self) {
        let nb = match self.ctx.seq_hdr.as_ref() {
            Some(s) => s.order_hint_n_bits as i32,
            None => return,
        };
        let mut mask: u32 = 0;
        loop {
            let mut cand: Option<(usize, u8)> = None; // (slot, poc)
            for n in 0..8 {
                if mask & (1 << n) != 0 {
                    continue;
                }
                let r = &self.ctx.refs[n];
                let pic = match r.p.pic.as_ref() {
                    Some(p) if p.has_data() => p,
                    _ => continue,
                };
                let hdr = match r.p.frame_hdr.as_ref() {
                    Some(h) => h,
                    None => continue,
                };
                if hdr.show_implicit == 0 {
                    continue;
                }
                let ipoc = pic
                    .frame_hdr
                    .as_ref()
                    .map(|h| h.frame_offset)
                    .unwrap_or(hdr.frame_offset);
                if crate::env::get_poc_diff(nb, ipoc as i32, self.dpb_poc as i32) > 0
                    && (cand.is_none()
                        || crate::env::get_poc_diff(nb, ipoc as i32, cand.unwrap().1 as i32) < 0)
                {
                    cand = Some((n, ipoc));
                }
            }
            let (slot, ipoc) = match cand {
                Some(c) => c,
                None => break,
            };
            // Append a fresh, independently-owned copy of the stored picture.
            let pic = self.ctx.refs[slot].p.pic.as_ref().unwrap().clone();
            self.dpb[self.dpb_in].pic.p = crate::decode::clone_picture_mt(
                &pic,
                self.n_tc,
                self.ctx.pool.as_ref(),
                self.ctx.pic_allocator.clone(),
            );
            self.dpb_in += 1;
            if self.dpb_in == self.dpb_sz {
                self.dpb_in = 0;
            }
            self.dpb_poc = ipoc;
            mask |= 1 << slot;
        }
    }

    /// Reset the decoder state, discarding all buffered data and references.
    pub fn flush(&mut self) {
        self.input.unref();

        for q in &mut self.dpb {
            if q.pic.p.has_data() {
                q.pic.unref();
            }
        }
        self.dpb_in = 0;
        self.dpb_out = 0;
        self.drain = false;

        for r in &mut self.ctx.refs {
            r.segmap = None;
            r.refmvs = None;
            r.ccsomap = None;
            r.p.frame_hdr = None;
            r.refpoc = [0; 7];
        }

        self.ctx.frame_hdr = None;
        self.ctx.seq_hdr = None;
        self.ctx.tile.clear();
        self.ctx.n_tile_data = 0;
        self.ctx.n_tiles = 0;

        self.flush.store(false, Ordering::Release);
    }

    pub fn n_threads(&self) -> u32 {
        self.n_tc
    }

    pub fn n_frame_contexts(&self) -> u32 {
        self.n_fc
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.flush();
    }
}

pub fn version() -> &'static str {
    "0.1.0"
}

pub fn version_api() -> u32 {
    1 << 8
}
