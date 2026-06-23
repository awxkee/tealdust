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

pub(crate) struct MsacContext<'a, const UPDATE_CDF: bool> {
    buf_pos: usize,
    buf: &'a [u8],
    dif: u64,
    rng: u32,
    cnt: i32,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MsacState {
    buf_pos: usize,
    dif: u64,
    rng: u32,
    cnt: i32,
}

impl<'a, const UPDATE_CDF: bool> MsacContext<'a, UPDATE_CDF> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut s = Self {
            buf_pos: 0,
            buf: data,
            dif: !0u64 >> 1,
            rng: 0x8000,
            cnt: -15,
        };
        s.ctx_refill();
        s
    }

    /// Snapshot the resumable state (everything but the buffer borrow).
    pub(crate) fn save(&self) -> MsacState {
        MsacState {
            buf_pos: self.buf_pos,
            dif: self.dif,
            rng: self.rng,
            cnt: self.cnt,
        }
    }

    /// Rebuild a live context from an owned buffer plus a prior snapshot. No
    /// `ctx_refill` here: the snapshot already reflects a refilled state, so this
    /// is a pure restore (re-running refill would consume extra bytes).
    pub(crate) fn resume(data: &'a [u8], st: MsacState) -> Self {
        Self {
            buf_pos: st.buf_pos,
            buf: data,
            dif: st.dif,
            rng: st.rng,
            cnt: st.cnt,
        }
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

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        // Keep the dynamic interface for non-hot call sites, but dispatch to
        // the fixed-size implementation so the actual decoder body is shared
        // and monomorphized where possible.
        match n_symbols {
            1 => self.decode_symbol_adapt_n::<1>(cdf),
            2 => self.decode_symbol_adapt_n::<2>(cdf),
            3 => self.decode_symbol_adapt_n::<3>(cdf),
            4 => self.decode_symbol_adapt_n::<4>(cdf),
            5 => self.decode_symbol_adapt_n::<5>(cdf),
            6 => self.decode_symbol_adapt_n::<6>(cdf),
            7 => self.decode_symbol_adapt_n::<7>(cdf),
            _ => unreachable!("invalid MSAC symbol count"),
        }
    }

    /// Fixed-symbol-count variant of [`decode_symbol_adapt`].
    ///
    /// This keeps the public safe slice interface, but removes the hot runtime
    /// `match n_symbols`/`try_into` scaffolding at call sites where the symbol
    /// count is known statically. The generated code is intentionally written
    /// with a fixed maximum stack array instead of `[T; N + k]`, so it stays on
    /// stable Rust without generic-const-expr requirements.
    #[inline(always)]
    pub(crate) fn decode_symbol_adapt_n<const N: usize>(&mut self, cdf: &mut [u16]) -> u32 {
        debug_assert!((1..=7).contains(&N));

        if cdf.len() <= N {
            return 0;
        }

        let min_prob = &MSAC_MIN_PROB[N - 1];
        let c = (self.dif >> 48) as u32;
        let r = self.rng >> 8;

        // Branchy interval search. The previous version computed every range
        // boundary into a stack array, then scanned all boundaries with masks.
        // In the coefficient hot path symbol 0/1 dominate, so most calls only
        // need one or two boundaries. This mirrors the usual entropy decoder
        // shape more closely and avoids a lot of multiply/shift work.
        let mut u = self.rng;
        let mut v = 0u32;
        let mut val_usize = N;

        for i in 0..N {
            let p_raw = (cdf[i] | 127) as i32 - min_prob[i] as i32;
            let p = p_raw.max(0) as u32;
            let boundary = ((r * p) >> 10) << 3;

            if c >= boundary {
                v = boundary;
                val_usize = i;
                break;
            }

            u = boundary;
        }

        if val_usize == N {
            let p_raw = (cdf[N] | 127) as i32 - min_prob[N] as i32;
            let p = p_raw.max(0) as u32;
            v = ((r * p) >> 10) << 3;
        }

        debug_assert!(u <= self.rng);
        debug_assert!(u >= v);
        self.ctx_norm(self.dif - ((v as u64) << 48), u - v);

        if UPDATE_CDF {
            let pc = cdf[N];
            let count = (pc & 0xFF) as u8;

            debug_assert!(count <= 32);

            let rate =
                MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize] + if N > 2 { 1 } else { 0 };

            for cdf_i in cdf[..val_usize].iter_mut() {
                *cdf_i = cdf_i.wrapping_add((32768u16 - *cdf_i) >> rate);
            }
            for cdf_i in cdf[val_usize..N].iter_mut() {
                *cdf_i = cdf_i.wrapping_sub(*cdf_i >> rate);
            }

            cdf[N] = pc + u16::from(count < 32);
        }

        val_usize as u32
    }

    #[inline]
    pub(crate) fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        let bit = self.decode_bool_raw(cdf[0] as u32);

        if UPDATE_CDF {
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
