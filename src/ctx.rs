#[inline(always)]
pub fn memset_pow2(buf: &mut [u8], off: usize, val: u8, log2_n: u8) {
    let n = 1usize << log2_n;
    buf[off..off + n].fill(val);
}
