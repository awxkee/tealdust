use crate::cdef::{CDEF_HAVE_BOTTOM, CDEF_HAVE_LEFT, CDEF_HAVE_RIGHT, CDEF_HAVE_TOP};
use crate::intops::iclip;
use crate::pixel::{BitDepth, Pixel};

pub static CCSO_POS: [[i8; 2]; 7] = [
    [-1, 0],
    [0, -1],
    [-1, -1],
    [-1, 1],
    [-1, -2],
    [1, -2],
    [0, 2],
];

#[inline(always)]
pub fn ccso_score(diff: i32, quant_step: i32, edge_classifier: u32) -> u32 {
    if diff > quant_step && edge_classifier == 0 {
        return 2;
    }
    if diff < -quant_step {
        return 0;
    }
    1
}

fn ccso_padding<P: Pixel>(
    tmp: &mut [P],
    tmp_stride: usize,
    o: usize,
    src: &[P],
    src_stride: usize,
    src_off: usize,
    left: &[[P; 2]],
    top: &[P],
    top_off: usize,
    bottom: &[P],
    bottom_off: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    let x_min: i32 = if edges & CDEF_HAVE_LEFT != 0 { -2 } else { 0 };
    let x_max: i32 = w as i32 - 1 + if edges & CDEF_HAVE_RIGHT != 0 { 2 } else { 0 };
    let y_min: i32 = if edges & CDEF_HAVE_TOP != 0 { -2 } else { 0 };
    let y_max: i32 = h as i32 - 1 + if edges & CDEF_HAVE_BOTTOM != 0 { 2 } else { 0 };

    for y in -2i32..h as i32 + 2 {
        let src_y = iclip(y, y_min, y_max);
        for x in -2i32..w as i32 + 2 {
            let src_x = iclip(x, x_min, x_max);
            let v = if src_y < 0 {
                top[(top_off as i32 + src_x + (2 + src_y) * src_stride as i32) as usize]
            } else if src_y >= h as i32 {
                bottom
                    [(bottom_off as i32 + src_x + (src_y - h as i32) * src_stride as i32) as usize]
            } else if src_x < 0 {
                left[src_y as usize][(2 + src_x) as usize]
            } else {
                src[(src_off as i32 + src_x + src_y * src_stride as i32) as usize]
            };
            tmp[(o as i32 + x + y * tmp_stride as i32) as usize] = v;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ccso_prep<BD: BitDepth>(
    bd: BD,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[BD::Pixel],
    src_stride: usize,
    src_off: usize,
    left: &[[BD::Pixel; 2]],
    top: &[BD::Pixel],
    top_off: usize,
    bottom: &[BD::Pixel],
    bottom_off: usize,
    max_band_log2: u32,
    ext_filter: usize,
    quant_step: i32,
    edge_clf: u32,
    bo_only: bool,
    edges: u8,
    w: usize,
    h: usize,
    ss_hor: usize,
    ss_ver: usize,
) {
    let shift = bd.bitdepth() as u32 - max_band_log2;
    let dy = CCSO_POS[ext_filter][0] as isize;
    let dx = CCSO_POS[ext_filter][1] as isize;
    let tmp_stride: usize = 68;
    let luma_offset = dx + dy * tmp_stride as isize;
    let mut tmp_buf = vec![BD::Pixel::default(); tmp_stride * (h.max(8) * (1 << ss_ver) + 4 + 4)];
    let o = 2 * tmp_stride + 2;

    ccso_padding(
        &mut tmp_buf,
        tmp_stride,
        o,
        src,
        src_stride,
        src_off,
        left,
        top,
        top_off,
        bottom,
        bottom_off,
        w << ss_hor,
        h << ss_ver,
        edges,
    );

    for y in 0..h {
        for x in 0..w {
            let x_luma = x << ss_hor;
            let ti = o + (y << ss_ver) * tmp_stride + x_luma;
            let c: i32 = tmp_buf[ti].into();
            let band = (c as u32 >> shift) as u8;
            if bo_only {
                dst[y * dst_stride + x] = band;
            } else {
                let cls0 = ccso_score(
                    Into::<i32>::into(tmp_buf[(ti as isize + luma_offset) as usize]) - c,
                    quant_step,
                    edge_clf,
                );
                let cls1 = ccso_score(
                    Into::<i32>::into(tmp_buf[(ti as isize - luma_offset) as usize]) - c,
                    quant_step,
                    edge_clf,
                );
                dst[y * dst_stride + x] = ((cls0 << 5) | (cls1 << 3)) as u8 | band;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn ccso_add<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    dst_stride: usize,
    idx_buf: &[u8],
    idx_stride: usize,
    offset_idxs: &[u8],
    offset_lut: &[i8],
    w: usize,
    h: usize,
    ll_mask: &[[u16; 4]],
) {
    for yy in (0..h).step_by(4) {
        let mi = yy >> 2;
        let mut bx = 0usize;
        let mut xx = 0usize;
        while xx < w {
            if ll_mask[mi][0] & (1 << bx) == 0 {
                for y in yy..yy + 4 {
                    for x in xx..xx + 4 {
                        let i = idx_buf[y * idx_stride + x];
                        let byte_idx = (i >> 1) as usize;
                        let half_idx = (i & 1) as usize;
                        let offset_idx = (7 & (offset_idxs[byte_idx] >> (4 * half_idx))) as usize;
                        let cur: i32 = dst[y * dst_stride + x].into();
                        dst[y * dst_stride + x] =
                            bd.pixel_clip(cur + offset_lut[offset_idx] as i32);
                    }
                }
            }
            xx += 4;
            bx += 1;
        }
    }
}
