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

/// Result of the fused high-range bypass decode.
pub(crate) enum HrDecode {
    Commit {
        val: i32,
        bits: u32,
        dif: u64,
    },
    /// Code too long for the available bit window; decode sequentially.
    Fallback,
}

/// Fused `decode_hr`: truncated unary + suffix (or exp-Golomb when
/// saturated) computed from `dif` in one pass with a single divide and a
/// single renormalization, instead of chained per-op renorms.
/// Pure: commits nothing. Caller must guarantee `cnt` covers `bits`.
#[inline(always)]
pub(crate) fn hr_bypass_kernel(dif: u64, rng: u32, cnt: u32, cmax: u32, m: u32) -> HrDecode {
    debug_assert!(rng & 1 == 0 && (dif >> 48) < rng as u64);
    debug_assert!((1..=6).contains(&cmax) && (1..=6).contains(&m));
    let r = rng as u64;
    let msb_r = (31 - rng.leading_zeros()) as i32;
    let d_rem = (r << 48) - dif;
    let k = (48 + msb_r - (63 - d_rem.leading_zeros()) as i32).clamp(0, cmax as i32) as u32;
    let over = (d_rem > (r << (48 - k))) as u32;
    let q = k - over;
    if q < cmax {
        let c = q + 1;
        let x = (r << (48 - q)) - d_rem;
        let s = 48 - c - m;
        let q2 = ((x >> s) as u32) / rng;
        let rem = x - (((q2 as u64) * r) << s);
        let total = c + m;
        let suffix = !q2 & ((1u32 << m) - 1);
        return HrDecode::Commit {
            val: (suffix + (q << m)) as i32,
            bits: total,
            dif: ((rem + 1) << total) - 1,
        };
    }
    // Saturated: cmax ones with no stop bit, then exp-Golomb with k = m + 1.
    let x1 = (r << (48 - cmax)) - d_rem;
    let dif2 = ((x1 + 1) << cmax) - 1; // logical renorm, in-register only
    let d_rem2 = (r << 48) - dif2;
    let k2 = (48 + msb_r - (63 - d_rem2.leading_zeros()) as i32).clamp(0, 21) as u32;
    let over2 = (d_rem2 > (r << (48 - k2))) as u32;
    let q21 = k2 - over2;
    let c2 = q21 + (q21 < 21) as u32;
    let len = q21 + m + 1;
    let total = cmax + c2 + len;
    if total > cnt {
        return HrDecode::Fallback;
    }
    let x2 = (r << (48 - q21)) - d_rem2;
    let s2 = 48 - c2 - len;
    let q2 = if len <= 16 {
        (((x2 >> s2) as u32) / rng) as u64
    } else {
        x2 / (r << s2)
    };
    let rem = x2 - ((q2 * r) << s2);
    let golomb = (1u32 << len) + (!(q2 as u32) & (u32::MAX >> (32 - len))) - (1u32 << (m + 1));
    HrDecode::Commit {
        val: (golomb + (cmax << m)) as i32,
        bits: total,
        dif: ((rem + 1) << (c2 + len)) - 1,
    }
}

/// Branch-free truncated-unary bypass in closed form.
///
/// The unary value is `q` = #{k >= 1 : dif >= rng * (2^48 - 2^(48-k))}, which
/// with `d_rem = (rng << 48) - dif` is the largest `k` with
/// `d_rem <= rng << (48 - k)`; an MSB estimate corrected by one exact compare
/// yields it without the serial data-dependent per-bit loop.
#[inline(always)]
pub(crate) fn unary_bypass_kernel(dif: u64, rng: u32, max_bits: u32) -> (u32, u32, u64) {
    debug_assert!(rng & 1 == 0);
    debug_assert!((dif >> 48) < rng as u64);
    let d_rem = ((rng as u64) << 48) - dif; // >= 1 by the dif invariant
    let k = (48 + (31 - rng.leading_zeros()) as i32 - (63 - d_rem.leading_zeros()) as i32)
        .clamp(0, max_bits as i32) as u32;
    // q is k or k - 1; when k == 0 the compare is always false.
    let over = (d_rem > ((rng as u64) << (48 - k))) as u32;
    let q = k - over;
    let bits = q + (q < max_bits) as u32;
    let dif = ((rng as u64) << (48 - q)) - d_rem; // == dif - sum of the q intervals
    (q, bits, ((dif + 1) << bits) - 1)
}

#[inline(always)]
pub(crate) unsafe fn msac_load_be64_unchecked(buf: &[u8], start: usize) -> u64 {
    debug_assert!(start + 8 <= buf.len());
    unsafe {
        u64::from_be(core::ptr::read_unaligned(
            buf.as_ptr().add(start).cast::<u64>(),
        ))
    }
}

#[cold]
#[inline(never)]
pub(crate) fn msac_refill_eob(buf: &[u8], start: usize, c: u32, mut dif: u64) -> (u64, usize, i32) {
    let len = buf.len();
    if start >= len {
        return (dif, start, 0);
    }

    let n = ((c as usize >> 3) + 1).min(len - start);

    for (i, &buf) in buf[start..start + n].iter().enumerate() {
        let shift = c - ((i as u32) << 3);
        dif ^= (buf as u64) << shift;
    }

    (dif, start + n, (n as i32) * 8)
}

pub(crate) struct MsacContextScalar<'a, const UPDATE_CDF: bool> {
    pub(crate) buf_pos: usize,
    pub(crate) buf: &'a [u8],
    pub(crate) dif: u64,
    pub(crate) rng: u32,
    pub(crate) cnt: i32,
    #[cfg(not(feature = "adaptive_cdf"))]
    pub(crate) update_cdf: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct MsacState {
    pub(crate) buf_pos: usize,
    pub(crate) dif: u64,
    pub(crate) rng: u32,
    pub(crate) cnt: i32,
    #[cfg(not(feature = "adaptive_cdf"))]
    pub(crate) update_cdf: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for MsacState {
    fn default() -> Self {
        Self {
            buf_pos: 0,
            dif: 0,
            rng: 0,
            cnt: 0,
            #[cfg(not(feature = "adaptive_cdf"))]
            update_cdf: true,
        }
    }
}

#[allow(dead_code)]
pub(crate) type MsacContext<'a, const UPDATE_CDF: bool> = MsacContextScalar<'a, UPDATE_CDF>;

pub(crate) struct ScalarMsacBackend;

#[cfg(all(target_arch = "x86_64", feature = "sse"))]
pub(crate) struct SseMsacBackend;

pub(crate) trait MsacBackend<const UPDATE_CDF: bool> {
    type Ctx<'a>: MsacReader<UPDATE_CDF> + 'a
    where
        Self: 'a;

    #[allow(unused)]
    fn new<'a>(data: &'a [u8]) -> Self::Ctx<'a>;
    fn resume<'a>(data: &'a [u8], st: MsacState) -> Self::Ctx<'a>;
}

impl<const UPDATE_CDF: bool> MsacBackend<UPDATE_CDF> for ScalarMsacBackend {
    type Ctx<'a> = MsacContextScalar<'a, UPDATE_CDF>;

    #[inline(always)]
    fn new<'a>(data: &'a [u8]) -> Self::Ctx<'a> {
        MsacContextScalar::new(data)
    }

    #[inline(always)]
    fn resume<'a>(data: &'a [u8], st: MsacState) -> Self::Ctx<'a> {
        MsacContextScalar::resume(data, st)
    }
}

#[cfg(all(target_arch = "x86_64", feature = "sse"))]
impl<const UPDATE_CDF: bool> MsacBackend<UPDATE_CDF> for SseMsacBackend {
    type Ctx<'a> = crate::sse::MsacContextSse<'a, UPDATE_CDF>;

    #[inline(always)]
    fn new<'a>(data: &'a [u8]) -> Self::Ctx<'a> {
        crate::sse::MsacContextSse::new(data)
    }

    #[inline(always)]
    fn resume<'a>(data: &'a [u8], st: MsacState) -> Self::Ctx<'a> {
        crate::sse::MsacContextSse::resume(data, st)
    }
}

pub(crate) trait MsacReader<const UPDATE_CDF: bool> {
    fn save(&self) -> MsacState;
    fn decode_bools_bypass(&mut self, n_bits: u32) -> u32;
    fn decode_hr_bypass(&mut self, cmax: u32, m: u32) -> i32;
    fn decode_bool_bypass(&mut self) -> u32;
    fn decode_unary_bypass(&mut self, max_bits: u32) -> u32;
    fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32;
    fn decode_symbol_adapt_padded<const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
        n_symbols: usize,
    ) -> u32;
    fn decode_symbol_adapt_n_padded<const N: usize, const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
    ) -> u32;
    fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32;
    fn decode_uniform(&mut self, n: u32) -> u32;
    fn decode_coefs<C: crate::pixel::Coeff>(
        &mut self,
        coef: &mut crate::cdf::CdfCoefContext,
        mode: &mut crate::cdf::CdfModeContext,
        a: &[u8],
        l: &[u8],
        p: &crate::recon::DecodeCoefParams,
        cf: &mut [C],
        txtp: &mut u16,
        res_ctx: &mut u8,
        levels_scratch: &mut [i8; 1089],
        nz_scratch: &mut [u32; 1024],
    ) -> i32;
    fn cnt(&self) -> i32;
}

impl<'a, const UPDATE_CDF: bool> MsacContextScalar<'a, UPDATE_CDF> {
    #[inline(always)]
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self::new_with_update_cdf(data, true)
    }

    #[inline(always)]
    pub(crate) fn new_with_update_cdf(data: &'a [u8], update_cdf: bool) -> Self {
        #[cfg(feature = "adaptive_cdf")]
        let _ = update_cdf;

        let mut s = Self {
            buf_pos: 0,
            buf: data,
            dif: !0u64 >> 1,
            rng: 0x8000,
            cnt: -15,
            #[cfg(not(feature = "adaptive_cdf"))]
            update_cdf,
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
            #[cfg(not(feature = "adaptive_cdf"))]
            update_cdf: self.update_cdf,
        }
    }

    /// Rebuild a live context from an owned buffer plus a prior snapshot. No
    /// `ctx_refill` here: the snapshot already reflects a refilled state, so this
    /// is a pure restore (re-running refill would consume extra bytes).
    #[inline(always)]
    pub(crate) fn resume(data: &'a [u8], st: MsacState) -> Self {
        Self {
            buf_pos: st.buf_pos,
            buf: data,
            dif: st.dif,
            rng: st.rng,
            cnt: st.cnt,
            #[cfg(not(feature = "adaptive_cdf"))]
            update_cdf: st.update_cdf,
        }
    }

    #[inline(always)]
    fn should_update_cdf(&self) -> bool {
        #[cfg(feature = "adaptive_cdf")]
        {
            UPDATE_CDF
        }
        #[cfg(not(feature = "adaptive_cdf"))]
        {
            self.update_cdf
        }
    }

    #[inline(always)]
    pub(crate) fn ctx_refill(&mut self) {
        let start = self.buf_pos;
        let c = 40 - self.cnt;
        debug_assert!(c >= 0);

        let c = c as u32;
        let n = (c as usize >> 3) + 1;

        if start + 8 <= self.buf.len() {
            let val = unsafe { msac_load_be64_unchecked(self.buf, start) };
            let refill = (val >> (56 - c)) & (u64::MAX << (c & 7));

            self.dif ^= refill;
            self.buf_pos = start + n;
            self.cnt += (n as i32) * 8;
        } else {
            let (dif, buf_pos, cnt_inc) = msac_refill_eob(self.buf, start, c, self.dif);
            self.dif = dif;
            self.buf_pos = buf_pos;
            self.cnt += cnt_inc;
        }
    }

    #[inline(always)]
    pub(crate) fn ctx_norm(&mut self, dif: u64, rng: u32) {
        debug_assert!(rng <= 65535 && rng > 0);

        // Skewed symbols keep the range normalized (d == 0) most of the time;
        // a predicted branch is cheaper on the dif/rng chain than the
        // unconditional lzcnt + shifts.
        if rng >= 0x8000 {
            self.dif = dif;
            self.rng = rng;
            return;
        }
        let d = rng.leading_zeros() ^ 16;
        let cnt = self.cnt;

        self.dif = ((dif + 1) << d) - 1;
        self.rng = rng << d;
        self.cnt = cnt - d as i32;

        if (cnt as u32) < d {
            self.ctx_refill();
        }
    }

    #[inline(always)]
    pub(crate) fn decode_hr_bypass(&mut self, cmax: u32, m: u32) -> i32 {
        // One conservative refill covers every fused path (bits <= 40; refill
        // tops cnt above 40). Early refills are order-transparent.
        if (self.cnt as u32) < 40 {
            self.ctx_refill();
        }
        match hr_bypass_kernel(self.dif, self.rng, self.cnt as u32, cmax, m) {
            HrDecode::Commit { val, bits, dif } => {
                self.dif = dif;
                self.cnt -= bits as i32;
                val
            }
            HrDecode::Fallback => {
                let q = self.decode_unary_bypass(cmax);
                debug_assert_eq!(q, cmax);
                let length = self.decode_unary_bypass(21) + m + 1;
                let golomb = (1u32 << length) + self.decode_bools_bypass(length) - (1 << (m + 1));
                (golomb + (cmax << m)) as i32
            }
        }
    }

    #[inline(always)]
    pub(crate) fn decode_bools_bypass(&mut self, n_bits: u32) -> u32 {
        debug_assert!(n_bits > 0 && n_bits <= 32);
        if (self.cnt as u32) < n_bits {
            self.ctx_refill();
        }

        let r = self.rng as u64;
        let dif = self.dif;
        debug_assert!(r & 1 == 0);
        debug_assert!((dif >> 48) < r);
        if n_bits < 4 {
            let mut dif = dif;
            let mut vw = r << 47;
            let mut ret: u32 = 0;
            for _ in 0..n_bits {
                let ge = u32::from(dif >= vw);
                let mask = 0u64.wrapping_sub(ge as u64);
                dif = dif.wrapping_sub(vw & mask);
                ret = (ret << 1) | (ge ^ 1);
                vw >>= 1;
            }
            self.dif = ((dif + 1) << n_bits) - 1;
            self.cnt -= n_bits as i32;
            return ret;
        }
        // The per-bit loop is restoring division; one divide replaces the
        // n-step serial chain and its n renormalizations. For n <= 16 the
        // nested-floor identity reduces it to a 32-bit divide:
        // dif / (rng << (48-n)) == (dif >> (48-n)) / rng, numerator < 2^32.
        let s = 48 - n_bits;
        let q = if n_bits <= 16 {
            (((dif >> s) as u32) / self.rng) as u64
        } else {
            dif / (r << s)
        };
        self.dif = ((dif - ((q * r) << s) + 1) << n_bits) - 1;
        self.cnt -= n_bits as i32;
        !(q as u32) & (u32::MAX >> (32 - n_bits))
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
        let (ret, bits, dif) = unary_bypass_kernel(self.dif, self.rng, max_bits);
        self.dif = dif;
        self.cnt -= bits as i32;
        ret
    }

    #[inline]
    fn decode_bool_raw(&mut self, f: u32) -> u32 {
        let r = self.rng;
        let dif = self.dif;
        debug_assert!((dif >> 48) < r as u64);

        let p = ((f >> 7) << 4) + 8;
        let v = (((r >> 8) * p) >> 7) << 3;
        let vw = (v as u64) << 48;

        let ge = u32::from(dif >= vw);
        let ge_u64 = ge as u64;
        let ge_mask64 = 0u64.wrapping_sub(ge_u64);
        let ge_mask32 = 0u32.wrapping_sub(ge);

        let new_dif = dif.wrapping_sub(vw & ge_mask64);
        let new_rng = (v & !ge_mask32) | ((r - v) & ge_mask32);

        self.ctx_norm(new_dif, new_rng);
        ge ^ 1
    }

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        if !self.should_update_cdf() && n_symbols == 3 {
            return self.decode_symbol_adapt3_no_update_scalar(cdf);
        }
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

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt_padded<const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
        n_symbols: usize,
    ) -> u32 {
        match n_symbols {
            1 => self.decode_symbol_adapt_n_padded::<1, LANES>(cdf),
            2 => self.decode_symbol_adapt_n_padded::<2, LANES>(cdf),
            3 => self.decode_symbol_adapt_n_padded::<3, LANES>(cdf),
            4 => self.decode_symbol_adapt_n_padded::<4, LANES>(cdf),
            5 => self.decode_symbol_adapt_n_padded::<5, LANES>(cdf),
            6 => self.decode_symbol_adapt_n_padded::<6, LANES>(cdf),
            7 => self.decode_symbol_adapt_n_padded::<7, LANES>(cdf),
            _ => unreachable!("invalid MSAC symbol count"),
        }
    }

    /// Fixed-symbol-count variant of [`Self::decode_symbol_adapt`].
    ///
    /// This keeps the public safe slice interface, but removes the hot runtime
    /// `match n_symbols`/`try_into` scaffolding at call sites where the symbol
    /// count is known statically. The generated code is intentionally written
    /// with a fixed maximum stack array instead of `[T; N + k]`, so it stays on
    /// stable Rust without generic-const-expr requirements.
    #[inline(always)]
    pub(crate) fn decode_symbol_adapt_n<const N: usize>(&mut self, cdf: &mut [u16]) -> u32 {
        debug_assert!((1..=7).contains(&N));

        #[cfg(feature = "adaptive_cdf")]
        if !UPDATE_CDF && N == 3 {
            return self.decode_symbol_adapt3_no_update_scalar(cdf);
        }

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
            let p = ((cdf[i] | 127) as u32) - min_prob[i] as u32;
            let boundary = ((r * p) >> 10) << 3;

            if c >= boundary {
                v = boundary;
                val_usize = i;
                break;
            }

            u = boundary;
        }

        if val_usize == N {
            debug_assert_eq!(min_prob[N], 65535);
            v = 0;
        }

        debug_assert!(u <= self.rng);
        debug_assert!(u >= v);
        self.ctx_norm(self.dif - ((v as u64) << 48), u - v);

        if self.should_update_cdf() {
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

    fn decode_symbol_adapt_n_padded<const N: usize, const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
    ) -> u32 {
        debug_assert!((1..=7).contains(&N));

        #[cfg(feature = "adaptive_cdf")]
        if !UPDATE_CDF && N == 3 {
            return self.decode_symbol_adapt3_no_update_scalar_l(cdf);
        }

        if LANES <= N {
            return 0;
        }

        let min_prob = &MSAC_MIN_PROB[N - 1];
        let c = (self.dif >> 48) as u32;
        let r = self.rng >> 8;

        let mut u = self.rng;
        let mut v = 0u32;
        let mut val_usize = N;

        for i in 0..N {
            let p = ((cdf[i] | 127) as u32) - min_prob[i] as u32;
            let boundary = ((r * p) >> 10) << 3;

            if c >= boundary {
                v = boundary;
                val_usize = i;
                break;
            }

            u = boundary;
        }

        if val_usize == N {
            debug_assert_eq!(min_prob[N], 65535);
            v = 0;
        }

        debug_assert!(u <= self.rng);
        debug_assert!(u >= v);
        self.ctx_norm(self.dif - ((v as u64) << 48), u - v);

        if self.should_update_cdf() {
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

        if self.should_update_cdf() {
            let pc = cdf[1];
            let count = (pc & 0xFF) as u8;
            let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize];
            let b = bit as i32; // 0 or 1
            let c = cdf[0] as i32;
            cdf[0] = (c - b - ((c - 32769 * b) >> rate)) as u16;
            cdf[1] = pc + if count < 32 { 1 } else { 0 };
        }

        bit
    }

    #[cfg(feature = "adaptive_cdf")]
    #[inline(always)]
    pub(crate) fn decode_symbol_adapt3_no_update_scalar_l<const LANES: usize>(
        &mut self,
        cdf: &[u16; LANES],
    ) -> u32 {
        debug_assert!(!self.should_update_cdf());

        let c = (self.dif >> 48) as u32;
        let r = self.rng >> 8;

        let p0 = ((cdf[0] | 127) as u32) - 31;
        let b0 = ((r * p0) >> 10) << 3;
        if c >= b0 {
            self.ctx_norm(self.dif - ((b0 as u64) << 48), self.rng - b0);
            return 0;
        }

        let p1 = ((cdf[1] | 127) as u32) - 63;
        let b1 = ((r * p1) >> 10) << 3;
        if c >= b1 {
            self.ctx_norm(self.dif - ((b1 as u64) << 48), b0 - b1);
            return 1;
        }

        let p2 = ((cdf[2] | 127) as u32) - 95;
        let b2 = ((r * p2) >> 10) << 3;
        if c >= b2 {
            self.ctx_norm(self.dif - ((b2 as u64) << 48), b1 - b2);
            return 2;
        }

        // Sentinel lane: v = 0, u = b2.
        self.ctx_norm(self.dif, b2);
        3
    }

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt3_no_update_scalar(&mut self, cdf: &[u16]) -> u32 {
        debug_assert!(!self.should_update_cdf());

        if cdf.len() <= 3 {
            return 0;
        }

        let c = (self.dif >> 48) as u32;
        let r = self.rng >> 8;

        let p0 = ((cdf[0] | 127) as u32) - 31;
        let b0 = ((r * p0) >> 10) << 3;
        if c >= b0 {
            self.ctx_norm(self.dif - ((b0 as u64) << 48), self.rng - b0);
            return 0;
        }

        let p1 = ((cdf[1] | 127) as u32) - 63;
        let b1 = ((r * p1) >> 10) << 3;
        if c >= b1 {
            self.ctx_norm(self.dif - ((b1 as u64) << 48), b0 - b1);
            return 1;
        }

        let p2 = ((cdf[2] | 127) as u32) - 95;
        let b2 = ((r * p2) >> 10) << 3;
        if c >= b2 {
            self.ctx_norm(self.dif - ((b2 as u64) << 48), b1 - b2);
            return 2;
        }

        // Sentinel lane: v = 0, u = b2.
        self.ctx_norm(self.dif, b2);
        3
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

impl<'a, const UPDATE_CDF: bool> MsacReader<UPDATE_CDF> for MsacContextScalar<'a, UPDATE_CDF> {
    #[inline(always)]
    fn save(&self) -> MsacState {
        MsacContextScalar::save(self)
    }

    #[inline(always)]
    fn decode_bools_bypass(&mut self, n_bits: u32) -> u32 {
        MsacContextScalar::decode_bools_bypass(self, n_bits)
    }

    #[inline(always)]
    fn decode_hr_bypass(&mut self, cmax: u32, m: u32) -> i32 {
        MsacContextScalar::decode_hr_bypass(self, cmax, m)
    }

    #[inline(always)]
    fn decode_bool_bypass(&mut self) -> u32 {
        MsacContextScalar::decode_bool_bypass(self)
    }

    #[inline(always)]
    fn decode_unary_bypass(&mut self, max_bits: u32) -> u32 {
        MsacContextScalar::decode_unary_bypass(self, max_bits)
    }

    #[inline(always)]
    fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        MsacContextScalar::decode_symbol_adapt(self, cdf, n_symbols)
    }

    #[inline(always)]
    fn decode_symbol_adapt_padded<const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
        n_symbols: usize,
    ) -> u32 {
        MsacContextScalar::decode_symbol_adapt_padded(self, cdf, n_symbols)
    }

    #[inline(always)]
    fn decode_symbol_adapt_n_padded<const N: usize, const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
    ) -> u32 {
        MsacContextScalar::decode_symbol_adapt_n_padded::<N, LANES>(self, cdf)
    }

    #[inline(always)]
    fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        MsacContextScalar::decode_bool_adapt(self, cdf)
    }

    #[inline(always)]
    fn decode_uniform(&mut self, n: u32) -> u32 {
        MsacContextScalar::decode_uniform(self, n)
    }

    #[inline(always)]
    fn decode_coefs<C: crate::pixel::Coeff>(
        &mut self,
        coef: &mut crate::cdf::CdfCoefContext,
        mode: &mut crate::cdf::CdfModeContext,
        a: &[u8],
        l: &[u8],
        p: &crate::recon::DecodeCoefParams,
        cf: &mut [C],
        txtp: &mut u16,
        res_ctx: &mut u8,
        levels_scratch: &mut [i8; 1089],
        nz_scratch: &mut [u32; 1024],
    ) -> i32 {
        crate::recon::decode_coefs_scalar(
            self,
            coef,
            mode,
            a,
            l,
            p,
            cf,
            txtp,
            res_ctx,
            levels_scratch,
            nz_scratch,
        )
    }

    #[inline(always)]
    fn cnt(&self) -> i32 {
        MsacContextScalar::cnt(self)
    }
}

#[cfg(test)]
mod unary_kernel_tests {
    use super::unary_bypass_kernel;

    fn reference(mut dif: u64, rng: u32, max_bits: u32) -> (u32, u32, u64) {
        let mut vw = (rng as u64) << 47;
        let (mut ret, mut bit) = (0u32, 0u32);
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
        (ret, bit, ((dif + 1) << bit) - 1)
    }

    #[test]
    fn closed_form_matches_reference_loop() {
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..400_000 {
            let rng = ((0x8000 | (next() & 0x7FFF)) & !1) as u32;
            let dif = next() % ((rng as u64) << 48);
            for &mb in &[5u32, 6, 21] {
                assert_eq!(
                    reference(dif, rng, mb),
                    unary_bypass_kernel(dif, rng, mb),
                    "dif={dif:#x} rng={rng:#x} max_bits={mb}"
                );
            }
        }
        for rng in [0x8000u32, 0x8002, 0x9246, 0xFFFE] {
            for q in 1..=21u32 {
                let s_q = (rng as u64) * ((1u64 << 48) - (1u64 << (48 - q)));
                for dif in [s_q.wrapping_sub(1), s_q, s_q + 1] {
                    if dif >> 48 >= rng as u64 {
                        continue;
                    }
                    for &mb in &[5u32, 6, 21] {
                        assert_eq!(
                            reference(dif, rng, mb),
                            unary_bypass_kernel(dif, rng, mb),
                            "dif={dif:#x} rng={rng:#x} q={q} max_bits={mb}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod bools_bypass_div_tests {
    #[test]
    fn divide_matches_bit_loop() {
        let mut s: u64 = 0xfeed_face_cafe_f00d;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..300_000 {
            let rng = ((0x8000 | (next() & 0x7FFF)) & !1) as u64;
            let dif = next() % (rng << 48);
            let n = 1 + (next() % 32) as u32;
            // reference bit loop
            let (mut d2, mut vw, mut r2) = (dif, rng << 47, 0u32);
            for _ in 0..n {
                let ge = u32::from(d2 >= vw);
                d2 -= vw & 0u64.wrapping_sub(ge as u64);
                r2 = (r2 << 1) | (ge ^ 1);
                vw >>= 1;
            }
            let ref_dif = ((d2 + 1) << n) - 1;
            // closed form
            let d = rng << (48 - n);
            let q = dif / d;
            let cf_dif = ((dif - q * d + 1) << n) - 1;
            let cf_ret = !(q as u32) & (u32::MAX >> (32 - n));
            assert_eq!(
                (r2, ref_dif),
                (cf_ret, cf_dif),
                "dif={dif:#x} rng={rng:#x} n={n}"
            );
        }
    }
}

#[cfg(test)]
mod hr_kernel_tests {
    use super::{HrDecode, hr_bypass_kernel};

    // Sequential reference: unary loop, renorm, then suffix loop or golomb.
    fn unary_ref(dif: &mut u64, rng: u64, max: u32) -> u32 {
        let mut vw = rng << 47;
        let (mut ret, mut bit) = (0u32, 0u32);
        while bit < max {
            if *dif >= vw {
                *dif -= vw;
                vw >>= 1;
                ret += 1;
                bit += 1;
            } else {
                bit += 1;
                break;
            }
        }
        *dif = ((*dif + 1) << bit) - 1;
        ret
    }
    fn bools_ref(dif: &mut u64, rng: u64, n: u32) -> u32 {
        let mut vw = rng << 47;
        let mut ret = 0u32;
        for _ in 0..n {
            let ge = u32::from(*dif >= vw);
            *dif -= vw & 0u64.wrapping_sub(ge as u64);
            ret = (ret << 1) | (ge ^ 1);
            vw >>= 1;
        }
        *dif = ((*dif + 1) << n) - 1;
        ret
    }
    fn hr_ref(mut dif: u64, rng: u64, cmax: u32, m: u32) -> (i32, u64) {
        let q = unary_ref(&mut dif, rng, cmax);
        let rem = if q == cmax {
            let len = unary_ref(&mut dif, rng, 21) + m + 1;
            (1u32 << len) + bools_ref(&mut dif, rng, len) - (1 << (m + 1))
        } else {
            bools_ref(&mut dif, rng, m)
        };
        ((rem + (q << m)) as i32, dif)
    }

    #[test]
    fn fused_hr_matches_sequential() {
        let mut s: u64 = 0xb0a7_10ad_5eed_c0de;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut fallbacks = 0u32;
        for _ in 0..400_000 {
            let rng = ((0x8000 | (next() & 0x7FFF)) & !1) as u32;
            let dif = next() % ((rng as u64) << 48);
            let m = 1 + (next() % 6) as u32;
            let cmax = (m + 4).min(6);
            match hr_bypass_kernel(dif, rng, 48, cmax, m) {
                HrDecode::Commit { val, bits, dif: nd } => {
                    let (rv, rd) = hr_ref(dif, rng as u64, cmax, m);
                    assert_eq!((val, nd), (rv, rd), "dif={dif:#x} rng={rng:#x} m={m}");
                    assert!(bits <= 48);
                }
                HrDecode::Fallback => fallbacks += 1,
            }
            // a low bit budget must never commit more bits than available
            if let HrDecode::Commit { bits, .. } = hr_bypass_kernel(dif, rng, 20, cmax, m) {
                assert!(bits <= 20);
            }
        }
        assert!(fallbacks < 4_000, "fallback should be rare: {fallbacks}");
    }
}
