use crate::intops::{apply_sign, iclip};

pub fn stxfm(
    cf_out: &mut [i32],
    cf: &[i32],
    kernel: &[i8],
    sz: usize,
    eob: usize,
    bitdepth_max: i32,
) {
    debug_assert!(sz == 16 || sz == 48);
    debug_assert!(eob < if sz == 16 { 8 } else { 32 });
    let min = -128 * (1 + bitdepth_max);
    let max = 128 * (1 + bitdepth_max) - 1;
    let h = eob + 1;
    for x in 0..sz {
        let mut sum = 0i32;
        for y in 0..h {
            sum += cf[y] * kernel[y * sz + x] as i32;
        }
        sum = apply_sign((sum.abs() + 64) >> 7, sum);
        cf_out[x] = iclip(sum, min, max);
    }
}
