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

#[inline(always)]
pub(crate) fn imax(a: i32, b: i32) -> i32 {
    a.max(b)
}

#[inline(always)]
pub(crate) fn imin(a: i32, b: i32) -> i32 {
    a.min(b)
}

#[inline(always)]
pub(crate) fn umin(a: u32, b: u32) -> u32 {
    a.min(b)
}

#[inline(always)]
pub(crate) fn iclip(v: i32, min: i32, max: i32) -> i32 {
    v.clamp(min, max)
}

#[inline(always)]
pub(crate) fn iclip64to32(v: i64, min: i32, max: i32) -> i32 {
    if v < min as i64 {
        min
    } else if v > max as i64 {
        max
    } else {
        v as i32
    }
}

#[inline(always)]
pub(crate) fn apply_sign(v: i32, s: i32) -> i32 {
    if s < 0 { -v } else { v }
}

#[inline(always)]
pub(crate) fn apply_sign64(v: i64, s: i64) -> i32 {
    if s < 0 { -(v as i32) } else { v as i32 }
}

#[inline(always)]
pub(crate) fn ulog2(v: u32) -> i32 {
    31 ^ v.leading_zeros() as i32
}

#[inline(always)]
pub(crate) fn u64log2(v: u64) -> i32 {
    63 ^ v.leading_zeros() as i32
}

#[inline(always)]
pub(crate) fn inv_recenter(r: u32, v: u32) -> u32 {
    if v > r << 1 {
        v
    } else if v & 1 == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}
