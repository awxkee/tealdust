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

pub(crate) static MSAC_RATE: [[u8; 3]; 125] = [
    [4, 5, 6],
    [4, 5, 5],
    [4, 5, 4],
    [4, 5, 7],
    [4, 5, 7],
    [4, 4, 6],
    [4, 4, 5],
    [4, 4, 4],
    [4, 4, 7],
    [4, 4, 7],
    [4, 3, 6],
    [4, 3, 5],
    [4, 3, 4],
    [4, 3, 7],
    [4, 3, 7],
    [4, 6, 6],
    [4, 6, 5],
    [4, 6, 4],
    [4, 6, 7],
    [4, 6, 7],
    [4, 6, 6],
    [4, 6, 5],
    [4, 6, 4],
    [4, 6, 7],
    [4, 6, 7],
    [3, 5, 6],
    [3, 5, 5],
    [3, 5, 4],
    [3, 5, 7],
    [3, 5, 7],
    [3, 4, 6],
    [3, 4, 5],
    [3, 4, 4],
    [3, 4, 7],
    [3, 4, 7],
    [3, 3, 6],
    [3, 3, 5],
    [3, 3, 4],
    [3, 3, 7],
    [3, 3, 7],
    [3, 6, 6],
    [3, 6, 5],
    [3, 6, 4],
    [3, 6, 7],
    [3, 6, 7],
    [3, 6, 6],
    [3, 6, 5],
    [3, 6, 4],
    [3, 6, 7],
    [3, 6, 7],
    [2, 5, 6],
    [2, 5, 5],
    [2, 5, 4],
    [2, 5, 7],
    [2, 5, 7],
    [2, 4, 6],
    [2, 4, 5],
    [2, 4, 4],
    [2, 4, 7],
    [2, 4, 7],
    [2, 3, 6],
    [2, 3, 5],
    [2, 3, 4],
    [2, 3, 7],
    [2, 3, 7],
    [2, 6, 6],
    [2, 6, 5],
    [2, 6, 4],
    [2, 6, 7],
    [2, 6, 7],
    [2, 6, 6],
    [2, 6, 5],
    [2, 6, 4],
    [2, 6, 7],
    [2, 6, 7],
    [5, 5, 6],
    [5, 5, 5],
    [5, 5, 4],
    [5, 5, 7],
    [5, 5, 7],
    [5, 4, 6],
    [5, 4, 5],
    [5, 4, 4],
    [5, 4, 7],
    [5, 4, 7],
    [5, 3, 6],
    [5, 3, 5],
    [5, 3, 4],
    [5, 3, 7],
    [5, 3, 7],
    [5, 6, 6],
    [5, 6, 5],
    [5, 6, 4],
    [5, 6, 7],
    [5, 6, 7],
    [5, 6, 6],
    [5, 6, 5],
    [5, 6, 4],
    [5, 6, 7],
    [5, 6, 7],
    [5, 5, 6],
    [5, 5, 5],
    [5, 5, 4],
    [5, 5, 7],
    [5, 5, 7],
    [5, 4, 6],
    [5, 4, 5],
    [5, 4, 4],
    [5, 4, 7],
    [5, 4, 7],
    [5, 3, 6],
    [5, 3, 5],
    [5, 3, 4],
    [5, 3, 7],
    [5, 3, 7],
    [5, 6, 6],
    [5, 6, 5],
    [5, 6, 4],
    [5, 6, 7],
    [5, 6, 7],
    [5, 6, 6],
    [5, 6, 5],
    [5, 6, 4],
    [5, 6, 7],
    [5, 6, 7],
];

#[repr(align(16))]
struct Aligned<T>(T);

static MSAC_MIN_PROB_INNER: Aligned<[[u16; 8]; 7]> = Aligned([
    [63, 65535, 65535, 65535, 65535, 65535, 65535, 65535],
    [47, 87, 65535, 65535, 65535, 65535, 65535, 65535],
    [31, 63, 95, 65535, 65535, 65535, 65535, 65535],
    [31, 55, 79, 103, 65535, 65535, 65535, 65535],
    [23, 47, 63, 87, 111, 65535, 65535, 65535],
    [23, 39, 55, 79, 95, 111, 65535, 65535],
    [15, 31, 47, 63, 79, 95, 111, 65535],
]);

pub(crate) static MSAC_MIN_PROB: &[[u16; 8]; 7] = &MSAC_MIN_PROB_INNER.0;

pub(crate) struct MsacContext<'a> {
    buf_pos: usize,
    buf: &'a [u8],
    dif: u64,
    rng: u32,
    cnt: i32,
    allow_update_cdf: bool,
}

impl<'a> MsacContext<'a> {
    pub(crate) fn new(data: &'a [u8], disable_cdf_update_flag: bool) -> Self {
        let mut s = Self {
            buf_pos: 0,
            buf: data,
            dif: !0u64 >> 1,
            rng: 0x8000,
            cnt: -15,
            allow_update_cdf: !disable_cdf_update_flag,
        };
        s.ctx_refill();
        s
    }

    #[inline]
    fn ctx_refill(&mut self) {
        let start = self.buf_pos;
        let len = self.buf.len();

        if start >= len {
            return;
        }

        let c = 40 - self.cnt;
        debug_assert!(c >= 0);
        debug_assert!(c <= 55);

        let c = c as u32;
        let available = len - start;
        let n = ((c as usize >> 3) + 1).min(available);

        if available >= 8 {
            let chunk = &self.buf[start..start + 8];
            let val = u64::from_be_bytes(chunk.try_into().unwrap());

            let refill = (val >> (56 - c)) & (u64::MAX << (c & 7));

            self.dif ^= refill;
            self.buf_pos = start + n;
            self.cnt += (n as i32) * 8;
        } else {
            let mut c_shift = c;
            let mut dif = self.dif;

            for &byte in &self.buf[start..start + n] {
                dif ^= (byte as u64) << c_shift;
                c_shift -= 8;
            }

            self.dif = dif;
            self.buf_pos = start + n;
            self.cnt += (n as i32) * 8;
        }
    }

    #[inline]
    fn ctx_norm(&mut self, dif: u64, rng: u32) {
        debug_assert!(rng <= 65535 && rng > 0);

        let d = rng.leading_zeros() ^ 16;
        let cnt = self.cnt;

        self.dif = ((dif + 1) << d) - 1;
        self.rng = rng << d;
        self.cnt = cnt - d as i32;

        if (cnt as u32) < d {
            self.ctx_refill();
        }
    }

    pub(crate) fn decode_bools_bypass(&mut self, n_bits: u32) -> u32 {
        debug_assert!(n_bits > 0 && n_bits <= 32);
        if (self.cnt as u32) < n_bits {
            self.ctx_refill();
        }

        let r = self.rng as u64;
        let mut dif = self.dif;
        debug_assert!(r & 1 == 0);
        debug_assert!((dif >> 48) < r);
        let mut vw = r << 47;
        let mut ret: u32 = 0;
        for _ in 0..n_bits {
            ret <<= 1;
            if dif >= vw {
                dif -= vw;
            } else {
                ret |= 1;
            }
            vw >>= 1;
        }
        self.dif = ((dif + 1) << n_bits) - 1;
        self.cnt -= n_bits as i32;
        ret
    }

    #[inline]
    pub(crate) fn decode_bool_bypass(&mut self) -> u32 {
        if self.cnt < 1 {
            self.ctx_refill();
        }

        let vw = (self.rng as u64) << 47;
        let dif = self.dif;

        debug_assert!(self.rng & 1 == 0);
        debug_assert!((dif >> 48) < self.rng as u64);

        let ge = (dif >= vw) as u64;
        let mask = 0u64.wrapping_sub(ge);

        let dif = dif - (vw & mask);

        self.dif = ((dif + 1) << 1) - 1;
        self.cnt -= 1;

        (ge as u32) ^ 1
    }

    pub(crate) fn decode_unary_bypass(&mut self, max_bits: u32) -> u32 {
        debug_assert!(max_bits == 5 || max_bits == 6 || max_bits == 21);
        if (self.cnt as u32) < max_bits {
            self.ctx_refill();
        }

        let r = self.rng as u64;
        let mut dif = self.dif;
        debug_assert!(r & 1 == 0);
        debug_assert!((dif >> 48) < r);
        let mut vw = r << 47;
        let mut ret: u32 = 0;
        let mut bit: u32 = 0;
        while bit < max_bits {
            if dif >= vw {
                dif -= vw;
                vw >>= 1;
                ret += 1;
                bit += 1;
            } else {
                bit += 1;
                break;
            }
        }
        self.dif = ((dif + 1) << bit) - 1;
        self.cnt -= bit as i32;
        ret
    }

    #[inline]
    fn decode_bool_raw(&mut self, f: u32) -> u32 {
        let r = self.rng;
        let dif = self.dif;
        debug_assert!((dif >> 48) < r as u64);
        let p = ((f >> 7) << 4) + 8;
        let mut v = (((r >> 8) * p) >> 7) << 3;
        let vw = (v as u64) << 48;
        let ret = if dif >= vw { 1 } else { 0 };
        let new_dif = dif - ret as u64 * vw;
        if ret != 0 {
            v = r - v;
        }
        self.ctx_norm(new_dif, v);
        (ret == 0) as u32
    }

    #[inline]
    pub(crate) fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        macro_rules! decode_n {
            ($n:literal) => {{
                if cdf.len() <= $n {
                    return 0;
                }

                // Safe compile-time sized array conversion (Zero-cost)
                let cdf_all: &mut [u16; $n + 1] = (&mut cdf[..=$n])
                    .try_into()
                    .unwrap();

                let min_prob: &[u16; $n + 1] = (&MSAC_MIN_PROB[$n - 1][..=$n])
                    .try_into()
                    .unwrap();

                let c = (self.dif >> 48) as u32;
                let r = self.rng >> 8;

                let mut v_arr = [0u32; $n + 2];
                v_arr[0] = self.rng;

                for i in 0..=$n {
                    let p_raw = (cdf_all[i] | 127) as i32 - min_prob[i] as i32;
                    let p = p_raw.max(0) as u32;
                    v_arr[i + 1] = ((r * p) >> 10) << 3;
                }

                let mut mask = 0u32;
                for i in 0..=$n {
                    mask |= ((c < v_arr[i + 1]) as u32) << i;
                }

                let val_usize = (mask.trailing_ones() as usize).min($n);
                let val = val_usize as u32;

                let u = v_arr[val_usize];
                let v = v_arr[val_usize + 1];

                debug_assert!(val <= $n);
                debug_assert!(u <= self.rng);

                self.ctx_norm(self.dif - ((v as u64) << 48), u - v);

                if self.allow_update_cdf {
                    let (cdf_syms, cdf_count) = cdf_all.split_at_mut($n);

                    let cdf_syms: &mut [u16; $n] = cdf_syms.try_into().unwrap();
                    let cdf_count: &mut [u16; 1] = cdf_count.try_into().unwrap();

                    let pc = cdf_count[0];
                    let count = (pc & 0xFF) as u8;

                    debug_assert!(count <= 32);

                    let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize]
                        + if $n > 2 { 1 } else { 0 };

                    for (i, cdf_i) in cdf_syms.iter_mut().enumerate() {
                        let mask = ((i < val_usize) as u16).wrapping_neg(); // 0xFFFF if true, 0x0000 if false
                        let v_true = (32768 - *cdf_i) >> rate;
                        let v_false = (*cdf_i >> rate).wrapping_neg();

                        let shift_val = (v_true & mask) | (v_false & !mask);
                        *cdf_i = cdf_i.wrapping_add(shift_val);
                    }

                    cdf_count[0] = pc + u16::from(count < 32);
                }

                val
            }};
        }

        // Fully safe exhaustive match
        match n_symbols {
            1 => decode_n!(1),
            2 => decode_n!(2),
            3 => decode_n!(3),
            4 => decode_n!(4),
            5 => decode_n!(5),
            6 => decode_n!(6),
            7 => decode_n!(7),
            _ => unreachable!("invalid MSAC symbol count"),
        }
    }

    #[inline]
    pub(crate) fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        let bit = self.decode_bool_raw(cdf[0] as u32);

        if self.allow_update_cdf {
            let pc = cdf[1];
            let count = (pc & 0xFF) as u8;
            let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize];
            if bit != 0 {
                cdf[0] += (32768 - cdf[0]) >> rate;
            } else {
                cdf[0] -= cdf[0] >> rate;
            }
            cdf[1] = pc + if count < 32 { 1 } else { 0 };
        }

        bit
    }

    pub(crate) fn decode_uniform(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let l = crate::intops::ulog2(n) + 1;
        debug_assert!(l > 1);
        let m = (1u32 << l) - n;
        let v = self.decode_bools_bypass((l - 1) as u32);
        if v < m {
            v
        } else {
            (v << 1) - m + self.decode_bool_bypass()
        }
    }

    /// Current internal bit count. Used to detect symbol-decoder overread
    /// (`cnt <= -15` after decoding a tile superblock row).
    pub(crate) fn cnt(&self) -> i32 {
        self.cnt
    }
}
