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

#![cfg(target_arch = "x86_64")]

use core::convert::TryInto;

use crate::msac::{
    MSAC_MIN_PROB, MSAC_RATE, MsacReader, MsacState, msac_load_be64_unchecked, msac_refill_eob,
};
use core::arch::x86_64::*;

pub(crate) struct MsacContextSse<'a, const UPDATE_CDF: bool> {
    pub(crate) buf_pos: usize,
    pub(crate) buf: &'a [u8],
    pub(crate) dif: u64,
    pub(crate) rng: u32,
    pub(crate) cnt: i32,
}

impl<'a, const UPDATE_CDF: bool> MsacContextSse<'a, UPDATE_CDF> {
    #[inline(always)]
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

    #[inline(always)]
    pub(crate) fn save(&self) -> MsacState {
        MsacState {
            buf_pos: self.buf_pos,
            dif: self.dif,
            rng: self.rng,
            cnt: self.cnt,
        }
    }

    #[inline(always)]
    pub(crate) fn resume(data: &'a [u8], st: MsacState) -> Self {
        Self {
            buf_pos: st.buf_pos,
            buf: data,
            dif: st.dif,
            rng: st.rng,
            cnt: st.cnt,
        }
    }

    #[inline(always)]
    pub(crate) fn ctx_refill(&mut self) {
        let start = self.buf_pos;
        let c = 40 - self.cnt;
        debug_assert!(c >= 0);
        debug_assert!(c <= 55);

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
            let ge = u32::from(dif >= vw);
            let mask = 0u64.wrapping_sub(ge as u64);
            dif = dif.wrapping_sub(vw & mask);
            ret = (ret << 1) | (ge ^ 1);
            vw >>= 1;
        }
        self.dif = ((dif + 1) << n_bits) - 1;
        self.cnt -= n_bits as i32;
        ret
    }

    #[inline(always)]
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

    #[inline(always)]
    pub(crate) fn decode_unary_bypass_scalar(&mut self, max_bits: u32) -> u32 {
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

    #[inline(always)]
    pub(crate) fn decode_unary_bypass(&mut self, max_bits: u32) -> u32 {
        self.decode_unary_bypass_scalar(max_bits)
    }

    #[inline(always)]
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
    pub(crate) fn decode_symbol_adapt_n_scalar<const N: usize>(&mut self, cdf: &mut [u16]) -> u32 {
        debug_assert!((1..=7).contains(&N));

        if cdf.len() <= N {
            return 0;
        }

        let min_prob = &MSAC_MIN_PROB[N - 1];
        let c = (self.dif >> 48) as u32;
        let r = self.rng >> 8;

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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
    pub(crate) fn cnt(&self) -> i32 {
        self.cnt
    }

    #[inline(always)]
    fn decode_symbol_adapt_sse2(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        match n_symbols {
            1 => self.decode_symbol_adapt_n_sse2::<1>(cdf),
            2 => self.decode_symbol_adapt_n_sse2::<2>(cdf),
            3 => self.decode_symbol_adapt_n_sse2::<3>(cdf),
            4 => self.decode_symbol_adapt_n_sse2::<4>(cdf),
            5 => self.decode_symbol_adapt_n_sse2::<5>(cdf),
            6 => self.decode_symbol_adapt_n_sse2::<6>(cdf),
            7 => self.decode_symbol_adapt_n_sse2::<7>(cdf),
            _ => unreachable!("invalid MSAC symbol count"),
        }
    }

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt_padded_sse2<const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
        n_symbols: usize,
    ) -> u32 {
        match n_symbols {
            1 => self.decode_symbol_adapt_n_padded_sse2::<1, LANES>(cdf),
            2 => self.decode_symbol_adapt_n_padded_sse2::<2, LANES>(cdf),
            3 => self.decode_symbol_adapt_n_padded_sse2::<3, LANES>(cdf),
            4 => self.decode_symbol_adapt_n_padded_sse2::<4, LANES>(cdf),
            5 => self.decode_symbol_adapt_n_padded_sse2::<5, LANES>(cdf),
            6 => self.decode_symbol_adapt_n_padded_sse2::<6, LANES>(cdf),
            7 => self.decode_symbol_adapt_n_padded_sse2::<7, LANES>(cdf),
            _ => unreachable!("invalid MSAC symbol count"),
        }
    }

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt_n_sse2<const N: usize>(&mut self, cdf: &mut [u16]) -> u32 {
        debug_assert!((1..=7).contains(&N));

        if cdf.len() <= N {
            return 0;
        }

        if N <= 3 && cdf.len() >= 4 {
            let cdf = (&mut cdf[..4])
                .try_into()
                .expect("4-lane CDF accessor must expose 4 values");
            return msac_decode_symbol_adapt_sse2::<UPDATE_CDF, N, 4>(self, cdf);
        }

        if N >= 4 && cdf.len() >= 8 {
            let cdf = (&mut cdf[..8])
                .try_into()
                .expect("8-lane CDF accessor must expose 8 values");
            return msac_decode_symbol_adapt_sse2::<UPDATE_CDF, N, 8>(self, cdf);
        }

        self.decode_symbol_adapt_n_scalar::<N>(cdf)
    }

    #[inline(always)]
    pub(crate) fn decode_symbol_adapt_n_padded_sse2<const N: usize, const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
    ) -> u32 {
        debug_assert!((1..=7).contains(&N));
        debug_assert!(LANES == 2 || LANES == 4 || LANES == 8);
        debug_assert!(N < LANES);

        if N <= 3 && LANES >= 4 {
            let cdf = (&mut cdf[..4])
                .try_into()
                .expect("4-lane CDF accessor must expose 4 values");
            return msac_decode_symbol_adapt_sse2::<UPDATE_CDF, N, 4>(self, cdf);
        }

        if N >= 4 && LANES >= 8 {
            let cdf = (&mut cdf[..8])
                .try_into()
                .expect("8-lane CDF accessor must expose 8 values");
            return msac_decode_symbol_adapt_sse2::<UPDATE_CDF, N, 8>(self, cdf);
        }

        self.decode_symbol_adapt_n_scalar::<N>(&mut cdf[..])
    }
}

impl<'a, const UPDATE_CDF: bool> MsacReader<UPDATE_CDF> for MsacContextSse<'a, UPDATE_CDF> {
    #[inline(always)]
    fn save(&self) -> MsacState {
        MsacContextSse::save(self)
    }

    #[inline(always)]
    fn decode_bools_bypass(&mut self, n_bits: u32) -> u32 {
        MsacContextSse::decode_bools_bypass(self, n_bits)
    }

    #[inline(always)]
    fn decode_bool_bypass(&mut self) -> u32 {
        MsacContextSse::decode_bool_bypass(self)
    }

    #[inline(always)]
    fn decode_unary_bypass(&mut self, max_bits: u32) -> u32 {
        MsacContextSse::decode_unary_bypass(self, max_bits)
    }

    #[inline(always)]
    fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { MsacContextSse::decode_symbol_adapt_sse2(self, cdf, n_symbols) }
    }

    #[inline(always)]
    fn decode_symbol_adapt_padded<const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
        n_symbols: usize,
    ) -> u32 {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { MsacContextSse::decode_symbol_adapt_padded_sse2::<LANES>(self, cdf, n_symbols) }
    }

    #[inline(always)]
    fn decode_symbol_adapt_n_padded<const N: usize, const LANES: usize>(
        &mut self,
        cdf: &mut [u16; LANES],
    ) -> u32 {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { MsacContextSse::decode_symbol_adapt_n_padded_sse2::<N, LANES>(self, cdf) }
    }

    #[inline(always)]
    fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        MsacContextSse::decode_bool_adapt(self, cdf)
    }

    #[inline(always)]
    fn decode_uniform(&mut self, n: u32) -> u32 {
        MsacContextSse::decode_uniform(self, n)
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
    ) -> i32 {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe {
            crate::recon::decode_coefs_sse2(
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
            )
        }
    }

    #[inline(always)]
    fn cnt(&self) -> i32 {
        MsacContextSse::cnt(self)
    }
}

#[inline(always)]
fn load_cdf<const LANES: usize>(cdf: &[u16; LANES]) -> __m128i {
    debug_assert!(LANES == 4 || LANES == 8);
    unsafe {
        if LANES == 4 {
            _mm_loadl_epi64(cdf.as_ptr().cast::<__m128i>())
        } else {
            _mm_loadu_si128(cdf.as_ptr().cast::<__m128i>())
        }
    }
}

#[inline(always)]
fn load_min_prob<const LANES: usize, const N: usize>() -> __m128i {
    debug_assert!(LANES == 4 || LANES == 8);
    let ptr = MSAC_MIN_PROB[N - 1].as_ptr().cast::<__m128i>();
    unsafe {
        if LANES == 4 {
            _mm_loadl_epi64(ptr)
        } else {
            _mm_loadu_si128(ptr)
        }
    }
}

#[inline(always)]
fn update_cdf_sse2<const N: usize, const LANES: usize>(
    cdf: &mut [u16; LANES],
    cdf_v: __m128i,
    ge_mask: __m128i,
    rate: u8,
) {
    debug_assert!((1..=7).contains(&N));
    debug_assert!(LANES == 4 || LANES == 8);
    debug_assert!(N < LANES);

    unsafe {
        let shift = _mm_cvtsi32_si128(rate as i32);
        let half = _mm_set1_epi16(0x8000u16 as i16);

        let add_delta = _mm_srl_epi16(_mm_sub_epi16(half, cdf_v), shift);
        let sub_delta = _mm_srl_epi16(cdf_v, shift);
        let add_path = _mm_add_epi16(cdf_v, add_delta);
        let sub_path = _mm_sub_epi16(cdf_v, sub_delta);

        // ge_mask is all-ones for lanes i >= decoded symbol and zero for lanes
        // i < decoded symbol.  That exactly matches the two AV1 CDF update halves:
        // before val move toward 32768, at/after val move toward zero.
        let updated = _mm_or_si128(
            _mm_and_si128(ge_mask, sub_path),
            _mm_andnot_si128(ge_mask, add_path),
        );

        if LANES == 4 {
            _mm_storel_epi64(cdf.as_mut_ptr().cast::<__m128i>(), updated);
        } else {
            _mm_storeu_si128(cdf.as_mut_ptr().cast::<__m128i>(), updated);
        }
    }
}
#[repr(C, align(16))]
struct AlignedU16x8([u16; 8]);

static PW_127: AlignedU16x8 = AlignedU16x8([127; 8]);

#[inline(always)]
fn pw_127() -> __m128i {
    unsafe { _mm_load_si128(PW_127.0.as_ptr().cast()) }
}

#[repr(C, align(16))]
pub(crate) struct AlignedSse9(pub(crate) [u16; 9]);

#[inline(always)]
fn msac_decode_symbol_adapt_sse2<const UPDATE_CDF: bool, const N: usize, const LANES: usize>(
    s: &mut MsacContextSse<'_, UPDATE_CDF>,
    cdf: &mut [u16; LANES],
) -> u32 {
    debug_assert!(LANES == 4 || LANES == 8);
    debug_assert!(N < LANES);

    let cdf_v = load_cdf::<LANES>(cdf);
    let min_prob = load_min_prob::<LANES, N>();
    let r = s.rng >> 8;

    // (4) Broadcast c = (dif >> 48) straight from an XMM holding dif.
    // dif>>48 is the top u16 of the low 64 bits, i.e. word lane 3 of dif.
    // movq dif into xmm, broadcast word 3 across all 8 lanes.
    let dif_xmm = unsafe { _mm_cvtsi64_si128(s.dif as i64) };
    // pshuflw picks lane 3 of the low quadword into all four low words,
    // then pshufd broadcasts the low quadword to the whole register.
    let c_v = unsafe {
        let lo = _mm_shufflelo_epi16::<0b11_11_11_11>(dif_xmm);
        _mm_shuffle_epi32::<0b00_00_00_00>(lo)
    };

    // p = max((cdf | 127) - min_prob, 0)   (subs saturates to 0)
    let p = unsafe { _mm_subs_epu16(_mm_or_si128(cdf_v, pw_127()), min_prob) };

    // boundary = ((r * p) >> 10) << 3, done as ((p * (r<<6))>>16) << 3 via mulhi.
    let scale = unsafe { _mm_set1_epi16(((r << 6) & 0xffff) as i16) };
    let boundaries_v = unsafe { _mm_slli_epi16::<3>(_mm_mulhi_epu16(p, scale)) };

    // cmp lane j = 0xFFFF iff c >= boundary[j]  (subs_epu16 == 0 means b<=c).
    let cmp = unsafe { _mm_cmpeq_epi16(_mm_subs_epu16(boundaries_v, c_v), _mm_setzero_si128()) };

    // (3) No `& 0x5555`. pcmpeqw gives paired bytes per word lane, so the raw
    // byte mask shifted right by 1 indexes word lanes directly via tzcnt.
    // We OR in a sentinel "always-set" pair just past lane N so that a
    // fall-through (c < every real boundary) resolves to i == N deterministically
    // and we never read a stale upper lane. The sentinel bit is placed at byte
    // position 2*N (word lane N).
    let raw = unsafe { _mm_movemask_epi8(cmp) as u32 };
    let sentinel = 1u32 << (2 * N); // low byte of word lane N
    let mask = raw | sentinel;

    let i = (mask.trailing_zeros() >> 1) as usize;

    let mut bounds = core::mem::MaybeUninit::<AlignedSse9>::uninit();
    let bounds_ptr = bounds.as_mut_ptr().cast::<u16>();
    unsafe {
        bounds_ptr.write(s.rng as u16);
        _mm_storeu_si128(bounds_ptr.add(1).cast(), boundaries_v);
    }
    let initialized = (unsafe { bounds.assume_init() }).0;

    let u = initialized[i] as u32;
    let v = initialized[i + 1] as u32;

    debug_assert!(u <= s.rng);
    debug_assert!(u >= v);
    s.ctx_norm(s.dif - ((v as u64) << 48), u - v);

    if UPDATE_CDF {
        let pc = cdf[N];
        let count = (pc & 0xff) as u8;
        debug_assert!(count <= 32);
        let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize] + if N > 2 { 1 } else { 0 };

        update_cdf_sse2::<N, LANES>(cdf, cdf_v, cmp, rate);
        cdf[N] = pc + u16::from(count < 32);
    }

    i as u32
}
