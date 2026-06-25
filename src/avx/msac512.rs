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

use crate::msac::{MSAC_MIN_PROB, MSAC_RATE, MsacReader, MsacState};
use core::arch::x86_64::*;

pub(crate) struct MsacContextAvx512<'a, const UPDATE_CDF: bool> {
    pub(crate) buf_pos: usize,
    pub(crate) buf: &'a [u8],
    pub(crate) dif: u64,
    pub(crate) rng: u32,
    pub(crate) cnt: i32,
}

impl<'a, const UPDATE_CDF: bool> MsacContextAvx512<'a, UPDATE_CDF> {
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

    #[inline(always)]
    fn ctx_norm_symbol(&mut self, inv_dif: u64, rng: u32) {
        debug_assert!(rng <= 65535 && rng > 0);

        let d = rng.leading_zeros() ^ 16;
        let cnt = self.cnt;

        //   inv_dif = !(dif - (v << 48))
        //   dif     = !(inv_dif << d)
        self.dif = !(inv_dif << d);
        self.rng = rng << d;
        self.cnt = cnt - d as i32;

        if (cnt as u32) < d {
            self.ctx_refill();
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

        // dav2d keeps this branchless with SUB/CMOV.  Express the same state
        // transition in Rust so LLVM can lower it without a hard branch in the
        // coefficient hot path.  The returned AV1 bit is one when dif < vw.
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

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) fn decode_unary_bypass_avx512(&mut self, max_bits: u32) -> u32 {
        debug_assert!(max_bits == 5 || max_bits == 6 || max_bits == 21);

        if (self.cnt as u32) < max_bits {
            self.ctx_refill();
        }

        let ret = msac_unary_bypass_ret_avx512(self.dif, self.rng, max_bits);
        let bits = ret + u32::from(ret < max_bits);
        let dif = self.dif - unary_success_sum(self.rng, ret);

        self.dif = ((dif + 1) << bits) - 1;
        self.cnt -= bits as i32;
        ret
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) fn decode_symbol_adapt_avx512(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        match n_symbols {
            1 => self.decode_symbol_adapt_n_avx512::<1>(cdf),
            2 => self.decode_symbol_adapt_n_avx512::<2>(cdf),
            3 => self.decode_symbol_adapt_n_avx512::<3>(cdf),
            4 => self.decode_symbol_adapt_n_avx512::<4>(cdf),
            5 => self.decode_symbol_adapt_n_avx512::<5>(cdf),
            6 => self.decode_symbol_adapt_n_avx512::<6>(cdf),
            7 => self.decode_symbol_adapt_n_avx512::<7>(cdf),
            _ => unreachable!("invalid MSAC symbol count"),
        }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) fn decode_symbol_adapt_n_avx512<const N: usize>(&mut self, cdf: &mut [u16]) -> u32 {
        debug_assert!((1..=7).contains(&N));

        if cdf.len() <= N {
            return 0;
        }

        if N <= 2 || (N <= 4 && cdf.len() < 4) || (N > 4 && cdf.len() < 8) {
            return self.decode_symbol_adapt_n_scalar::<N>(cdf);
        }

        msac_decode_symbol_adapt_avx512::<UPDATE_CDF, N>(self, cdf)
    }
}

impl<'a, const UPDATE_CDF: bool> MsacReader<UPDATE_CDF> for MsacContextAvx512<'a, UPDATE_CDF> {
    #[inline(always)]
    fn save(&self) -> MsacState {
        MsacContextAvx512::save(self)
    }

    #[inline(always)]
    fn decode_bools_bypass(&mut self, n_bits: u32) -> u32 {
        MsacContextAvx512::decode_bools_bypass(self, n_bits)
    }

    #[inline(always)]
    fn decode_bool_bypass(&mut self) -> u32 {
        MsacContextAvx512::decode_bool_bypass(self)
    }

    #[inline(always)]
    fn decode_unary_bypass(&mut self, max_bits: u32) -> u32 {
        // SAFETY: this backend is only constructed after the AVX-512F/DQ runtime guard.
        unsafe { MsacContextAvx512::decode_unary_bypass_avx512(self, max_bits) }
    }

    #[inline(always)]
    fn decode_symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        // SAFETY: this backend is only constructed after the AVX-512F/DQ runtime guard.
        unsafe { MsacContextAvx512::decode_symbol_adapt_avx512(self, cdf, n_symbols) }
    }

    #[inline(always)]
    fn decode_symbol_adapt_n<const N: usize>(&mut self, cdf: &mut [u16]) -> u32 {
        // SAFETY: this backend is only constructed after the AVX-512F/DQ runtime guard.
        unsafe { MsacContextAvx512::decode_symbol_adapt_n_avx512::<N>(self, cdf) }
    }

    #[inline(always)]
    fn decode_bool_adapt(&mut self, cdf: &mut [u16]) -> u32 {
        MsacContextAvx512::decode_bool_adapt(self, cdf)
    }

    #[inline(always)]
    fn decode_uniform(&mut self, n: u32) -> u32 {
        MsacContextAvx512::decode_uniform(self, n)
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
        // SAFETY: this backend is only constructed after the AVX-512F/DQ runtime guard.
        unsafe {
            crate::recon::decode_coefs_avx512(
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
        MsacContextAvx512::cnt(self)
    }
}

#[repr(C, align(64))]
struct AlignedAvx512I32<T>(T);

#[repr(C, align(64))]
struct AlignedAvx512U64<T>(T);

static UNARY_MUL32: AlignedAvx512I32<[i32; 16]> = AlignedAvx512I32([
    0x8000, 0xc000, 0xe000, 0xf000, 0xf800, 0xfc00, 0xfe00, 0xff00, 0xff80, 0xffc0, 0xffe0, 0xfff0,
    0xfff8, 0xfffc, 0xfffe, 0xffff,
]);

// dav2d's AVX-512 unary-bypass21 compares dif >> 16 against these Q32
// cumulative thresholds for bits 17..21.  The last three lanes are kept zero
// so their unsigned comparisons are guaranteed false.
static UNARY_MUL64: AlignedAvx512U64<[u64; 8]> = AlignedAvx512U64([
    0xffff_8000,
    0xffff_c000,
    0xffff_e000,
    0xffff_f000,
    0xffff_f800,
    0,
    0,
    0,
]);

#[inline(always)]
fn unary_success_sum(rng: u32, ret: u32) -> u64 {
    if ret == 0 {
        0
    } else {
        ((rng as u64) * ((1u64 << ret) - 1)) << (48 - ret)
    }
}

#[target_feature(enable = "avx512f,avx512dq")]
fn msac_unary_bypass_ret_avx512(dif: u64, rng: u32, max_bits: u32) -> u32 {
    debug_assert!(max_bits == 5 || max_bits == 6 || max_bits == 21);

    let r32 = _mm512_set1_epi32(rng as i32);
    let d32 = _mm512_set1_epi32((dif >> 32) as i32);
    let mul32 = unsafe { _mm512_load_si512(UNARY_MUL32.0.as_ptr().cast()) };
    let thr32 = _mm512_mullo_epi32(r32, mul32);
    let limit = max_bits.min(16);
    let fail32 = _mm512_cmp_epu32_mask(thr32, d32, 6) as u32;
    let mask32 = (fail32 & ((1u32 << limit) - 1)) | (1u32 << limit);
    let ret32 = mask32.trailing_zeros();
    if ret32 < limit || max_bits <= 16 {
        return ret32;
    }

    let r64 = _mm512_set1_epi64(rng as i64);
    let d64 = _mm512_set1_epi64((dif >> 16) as i64);
    let mul64 = unsafe { _mm512_load_si512(UNARY_MUL64.0.as_ptr().cast()) };
    let thr64 = _mm512_mul_epu32(r64, mul64);
    let fail64 = (_mm512_cmp_epu64_mask(thr64, d64, 6) as u32) & 0x1f;
    let mask64 = (fail64 << 16) | (1u32 << 21);
    mask64.trailing_zeros()
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn load_cdf<const N: usize>(cdf: &[u16]) -> __m128i {
    unsafe {
        if N <= 4 {
            _mm_loadl_epi64(cdf.as_ptr().cast::<__m128i>())
        } else {
            _mm_loadu_si128(cdf.as_ptr().cast::<__m128i>())
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn load_min_prob<const N: usize>() -> __m128i {
    let ptr = MSAC_MIN_PROB[N - 1].as_ptr().cast::<__m128i>();
    unsafe {
        if N <= 4 {
            _mm_loadl_epi64(ptr)
        } else {
            _mm_loadu_si128(ptr)
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn update_cdf_avx512<const N: usize>(cdf: &mut [u16], cdf_v: __m128i, ge_mask: __m128i, rate: u8) {
    debug_assert!((1..=7).contains(&N));
    debug_assert!(cdf.len() > N);

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

    let mut tmp = core::mem::MaybeUninit::<AlignedSse8>::uninit();
    unsafe { _mm_store_si128(tmp.as_mut_ptr().cast(), updated) };
    let initialized = (unsafe { tmp.assume_init() }).0;
    cdf[..N].copy_from_slice(&initialized[..N]);
}

#[repr(C, align(16))]
pub(crate) struct AlignedSse9(pub(crate) [u16; 9]);

#[repr(C, align(16))]
pub(crate) struct AlignedSse8(pub(crate) [u16; 8]);

#[target_feature(enable = "avx512f,avx512dq")]
fn msac_decode_symbol_adapt_avx512<const UPDATE_CDF: bool, const N: usize>(
    s: &mut MsacContextAvx512<'_, UPDATE_CDF>,
    cdf: &mut [u16],
) -> u32 {
    let cdf_v = load_cdf::<N>(cdf);
    let min_prob = load_min_prob::<N>();
    let c = (s.dif >> 48) as u16;
    let r = s.rng >> 8;

    let p = _mm_subs_epu16(_mm_or_si128(cdf_v, _mm_set1_epi16(127)), min_prob);
    let scale = _mm_set1_epi16(((r << 6) & 0xffff) as i16);
    let boundaries_v = _mm_slli_epi16(_mm_mulhi_epu16(p, scale), 3);
    let cmp = _mm_cmpeq_epi16(
        _mm_subs_epu16(boundaries_v, _mm_set1_epi16(c as i16)),
        _mm_setzero_si128(),
    );

    let mask_lanes = if N == 4 { N } else { N + 1 };
    let mask = ((_mm_movemask_epi8(cmp) as u32) & 0x5555) & ((1u32 << (mask_lanes * 2)) - 1);

    let mut bounds = core::mem::MaybeUninit::<AlignedSse9>::uninit();
    let bounds_ptr = bounds.as_mut_ptr().cast::<u16>();
    unsafe {
        bounds_ptr.write(s.rng as u16);
        _mm_store_si128(bounds_ptr.add(1).cast(), boundaries_v);
    }

    let initialized = (unsafe { bounds.assume_init() }).0;

    let (val, v, u) = if N == 4 && mask == 0 {
        // cdf[4] was not loaded; its min_prob sentinel would have produced v=0.
        let u = initialized[N] as u32;
        (N, 0, u)
    } else {
        let i = (mask.trailing_zeros() >> 1) as usize;
        let u = initialized[i] as u32;
        let v = initialized[i + 1] as u32;
        (i, v, u)
    };

    debug_assert!(u <= s.rng);
    debug_assert!(u >= v);
    let inv_dif = (!s.dif).wrapping_add((v as u64) << 48);
    s.ctx_norm_symbol(inv_dif, u - v);

    if UPDATE_CDF {
        let pc = cdf[N];
        let count = (pc & 0xff) as u8;
        debug_assert!(count <= 32);
        let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize] + if N > 2 { 1 } else { 0 };

        update_cdf_avx512::<N>(cdf, cdf_v, cmp, rate);
        cdf[N] = pc + u16::from(count < 32);
    }

    val as u32
}
