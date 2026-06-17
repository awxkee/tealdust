pub fn pal_idx_finish(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    debug_assert!((4..=64).contains(&bw) && (bw & (bw - 1)) == 0);
    debug_assert!((4..=64).contains(&bh) && (bh & (bh - 1)) == 0);
    debug_assert!(w >= 4 && w <= bw && (w & 3) == 0);
    debug_assert!(h >= 4 && h <= bh && (h & 3) == 0);

    let dst_w = w / 2;
    let dst_bw = bw / 2;

    for y in 0..h {
        let src_row = &src[y * bw..];
        let dst_row = &mut dst[y * dst_bw..];
        for x in 0..dst_w {
            dst_row[x] = src_row[x * 2] | (src_row[x * 2 + 1] << 4);
        }
        if dst_w < dst_bw {
            let fill = src_row[w - 1] * 0x11;
            dst_row[dst_w..dst_bw].fill(fill);
        }
    }

    if h < bh {
        let last_row_start = (h - 1) * dst_bw;
        for y in h..bh {
            let row_start = y * dst_bw;
            for x in 0..dst_bw {
                dst[row_start + x] = dst[last_row_start + x];
            }
        }
    }
}
