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

use crate::intops::{inv_recenter, ulog2};

pub(crate) struct GetBits<'a> {
    state: u64,
    bits_left: i32,
    error: bool,
    ptr: usize,
    data: &'a [u8],
}

impl<'a> GetBits<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            state: 0,
            bits_left: 0,
            error: false,
            ptr: 0,
            data,
        }
    }

    #[inline]
    pub(crate) fn has_error(&self) -> bool {
        self.error
    }

    pub(crate) fn get_bit(&mut self) -> u32 {
        if self.bits_left == 0 {
            if self.ptr >= self.data.len() {
                self.error = true;
                return 0;
            }
            let byte = self.data[self.ptr] as u64;
            self.ptr += 1;
            self.bits_left = 7;
            self.state = byte << 57;
            return (byte >> 7) as u32;
        }

        let state = self.state;
        self.bits_left -= 1;
        self.state = state << 1;
        (state >> 63) as u32
    }

    #[inline]
    fn refill(&mut self, n: i32) {
        debug_assert!(self.bits_left >= 0 && self.bits_left < 32);
        let mut st: u32 = 0;
        loop {
            if self.ptr >= self.data.len() {
                self.error = true;
                if st != 0 {
                    break;
                }
                return;
            }
            st = (st << 8) | self.data[self.ptr] as u32;
            self.ptr += 1;
            self.bits_left += 8;
            if n <= self.bits_left {
                break;
            }
        }
        self.state |= (st as u64) << (64 - self.bits_left);
    }

    pub(crate) fn get_bits(&mut self, n: i32) -> u32 {
        debug_assert!((0..=32).contains(&n));
        if !(0..=32).contains(&n) {
            self.error = true;
            return 0;
        }
        // A zero-width field carries no bits and reads as 0. The C reference
        // asserts n > 0 and relies on the field never being zero in practice;
        // for AV2, ref_frames_log2 is legitimately 0 when ref_frames == 1, so a
        // 0-bit reference index reaches here on otherwise-valid streams.
        if n == 0 {
            return 0;
        }
        if n > self.bits_left {
            self.refill(n);
            if n > self.bits_left {
                self.error = true;
                self.bits_left = 0;
                self.state = 0;
                return 0;
            }
        }
        let state = self.state;
        self.bits_left -= n;
        self.state = state << n;
        (state >> (64 - n)) as u32
    }

    pub(crate) fn get_sbits(&mut self, n: i32) -> i32 {
        debug_assert!((0..=32).contains(&n));
        if !(0..=32).contains(&n) {
            self.error = true;
            return 0;
        }
        if n == 0 {
            return 0;
        }
        if n > self.bits_left {
            self.refill(n);
            if n > self.bits_left {
                self.error = true;
                self.bits_left = 0;
                self.state = 0;
                return 0;
            }
        }
        let state = self.state;
        self.bits_left -= n;
        self.state = state << n;
        ((state as i64) >> (64 - n)) as i32
    }

    pub(crate) fn get_uleb128(&mut self) -> u32 {
        let mut val: u64 = 0;
        let mut i: u32 = 0;

        loop {
            let v = self.get_bits(8);
            let more = v & 0x80;
            val |= ((v & 0x7F) as u64) << i;
            i += 7;
            if more == 0 || i >= 56 {
                break;
            }
        }

        if val > u32::MAX as u64 {
            self.error = true;
            return 0;
        }

        val as u32
    }

    pub(crate) fn get_golomb(&mut self, k: u32) -> u32 {
        debug_assert!(k < 32);
        if k >= 32 {
            self.error = true;
            return 0;
        }
        let mut bits: u32 = 0;
        while bits < 32 - k {
            if self.get_bit() == 0 {
                break;
            }
            bits += 1;
        }
        if bits + k == 32 {
            return u32::MAX;
        }
        (bits << k) | self.get_bits(k as i32)
    }

    pub(crate) fn get_uniform(&mut self, max: u32) -> u32 {
        // Valid streams always call this with max > 1. A malformed stream can
        // derive max <= 1 (e.g. a corrupted subexp range); the only legal value
        // in a range of size <= 1 is 0, so return it without consuming bits or
        // hitting ulog2(0) UB. No-op for valid input.
        if max <= 1 {
            return 0;
        }
        let l = ulog2(max) + 1;
        debug_assert!(l > 1);
        let m = (1u32 << l) - max;
        let v = self.get_bits(l - 1);
        if v < m {
            v
        } else {
            (v << 1) - m + self.get_bit()
        }
    }

    pub(crate) fn get_vlc(&mut self) -> u32 {
        if self.get_bit() != 0 {
            return 0;
        }

        let mut n_bits: i32 = 0;
        loop {
            n_bits += 1;
            if n_bits == 32 {
                return u32::MAX;
            }
            if self.get_bit() != 0 {
                break;
            }
        }

        ((1u32 << n_bits) - 1) + self.get_bits(n_bits)
    }

    pub(crate) fn get_bits_subexp_u(&mut self, ref_val: u32, n: u32, k: i32) -> u32 {
        if n == 0 || ref_val >= n || k < 0 || k >= 32 {
            self.error = true;
            return 0;
        }

        let mut v: u32 = 0;

        let mut i = 0;
        loop {
            let b = if i != 0 { k + i - 1 } else { k };
            if b < 0 || b >= 32 {
                self.error = true;
                return 0;
            }
            let a = 1u32 << b;

            if n <= v.saturating_add(3u32.saturating_mul(a)) {
                v = v.saturating_add(self.get_uniform(n.saturating_sub(v)));
                break;
            }

            if self.get_bit() == 0 {
                v = v.saturating_add(self.get_bits(b));
                break;
            }

            v = v.saturating_add(a);
            i += 1;
        }

        if ref_val.saturating_mul(2) <= n {
            let rec = inv_recenter(ref_val, v);
            if rec >= n {
                self.error = true;
                return 0;
            }
            rec
        } else {
            let rec = inv_recenter(n - 1 - ref_val, v);
            if rec >= n {
                self.error = true;
                return 0;
            }
            n - 1 - rec
        }
    }

    pub(crate) fn get_bits_subexp(&mut self, ref_val: i32, n: u32) -> i32 {
        if n == 0 || n > (i32::MAX as u32) {
            self.error = true;
            return 0;
        }
        let off = n as i32 - 1;
        let ref_u = match ref_val.checked_add(off) {
            Some(v) if v >= 0 => v as u32,
            _ => {
                self.error = true;
                return 0;
            }
        };
        let n2 = match n.checked_add(off as u32) {
            Some(v) => v,
            None => {
                self.error = true;
                return 0;
            }
        };
        self.get_bits_subexp_u(ref_u, n2, 3) as i32 - off
    }

    pub(crate) fn get_ref_uniform(&mut self, max: u32, def: u32) -> u32 {
        if max <= 1 {
            return 0;
        }
        let def = def.min(max - 1);
        if self.get_bit() == 0 {
            return def;
        }
        let res = self.get_uniform(max - 1);
        res + if res >= def { 1 } else { 0 }
    }

    #[inline]
    pub(crate) fn bytealign(&mut self) {
        debug_assert!(self.bits_left <= 7);
        self.bits_left = 0;
        self.state = 0;
    }

    #[inline]
    pub(crate) fn byte_pos(&self) -> usize {
        self.ptr
    }

    #[inline]
    pub(crate) fn remaining_bytes(&self) -> usize {
        self.data.len() - self.ptr
    }

    #[inline]
    pub(crate) fn remaining_slice(&self) -> &'a [u8] {
        &self.data[self.ptr..]
    }
}
