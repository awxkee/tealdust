//! Multi-tile streams with `disable_loopfilters_across_tiles = 1`: AVM neither
//! deblocks tile seam edges nor lets CDEF read across them (a seam is padded
//! like a frame edge). Both fixtures are maroontree stills with deblock and CDEF
//! on: a 256x256 4:0:0 frame in 4x4 tiles and a 512x512 4:2:0 frame in 2x2
//! tiles. Expected hashes are FNV-1a 64 of `avmdec --rawvideo` output.

use tealdust::{AvifDecoder, AvifSettings, PixelLayout};

fn fnv1a64(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Decode and pack the planes tightly in avmdec's raw output order.
fn decode_packed(data: &[u8], n_threads: u32) -> Vec<u8> {
    let mut settings = AvifSettings::default();
    settings.decoder_settings.n_threads = n_threads;
    let img = AvifDecoder::with_settings(data, settings)
        .unwrap()
        .decode()
        .unwrap();
    let (w, h) = (img.width as usize, img.height as usize);
    let bps = if img.bits_per_component > 8 { 2 } else { 1 };
    let (cw, ch) = match img.pixel_layout {
        PixelLayout::I400 => (0, 0),
        PixelLayout::I420 => (w.div_ceil(2), h.div_ceil(2)),
        PixelLayout::I422 => (w.div_ceil(2), h),
        PixelLayout::I444 => (w, h),
    };
    let mut out = Vec::new();
    for (p, (pw, ph)) in [(w, h), (cw, ch), (cw, ch)].into_iter().enumerate() {
        let stride = img.strides[p.min(1)];
        for y in 0..ph {
            out.extend_from_slice(&img.planes[p][y * stride..y * stride + pw * bps]);
        }
    }
    out
}

fn check(data: &[u8], len: usize, hash: u64) {
    for n_threads in [0, 4] {
        let out = decode_packed(data, n_threads);
        assert_eq!(out.len(), len, "n_threads={n_threads}");
        assert_eq!(fnv1a64(&out), hash, "n_threads={n_threads}");
    }
}

#[test]
fn tiled_400_matches_avmdec() {
    check(
        include_bytes!("data/tiled_cdef_400.avif"),
        256 * 256,
        0x56cc_d9d5_eb0b_e30e,
    );
}

#[test]
fn tiled_420_matches_avmdec() {
    check(
        include_bytes!("data/tiled_cdef_420.avif"),
        512 * 512 * 3 / 2,
        0xc437_5981_b878_c186,
    );
}
