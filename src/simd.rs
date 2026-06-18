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

use crate::pixel::{BitDepth, Pixel};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct I32x8([i32; 8]);

impl I32x8 {
    #[inline(always)]
    pub(crate) fn splat(v: i32) -> Self {
        I32x8([v; 8])
    }

    #[inline(always)]
    pub(crate) fn to_array(self) -> [i32; 8] {
        self.0
    }

    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].max(rhs.0[i]);
        }
        I32x8(r)
    }

    #[inline(always)]
    pub(crate) fn min(self, rhs: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].min(rhs.0[i]);
        }
        I32x8(r)
    }

    #[inline(always)]
    pub(crate) fn abs(self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].abs();
        }
        I32x8(r)
    }

    /// Returns a mask: lane = -1 (all ones) where self < rhs, else 0.
    /// Returns -1 in each lane where self < rhs, else 0 (all-zeros mask).
    #[inline(always)]
    pub(crate) fn cmp_lt(self, rhs: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if self.0[i] < rhs.0[i] { -1 } else { 0 };
        }
        I32x8(r)
    }

    /// Blend: for each lane, pick `t` if mask lane is non-zero (i.e. == -1), else `f`.
    /// Selects t-lane where mask is non-zero (-1), f-lane otherwise.
    #[inline(always)]
    pub(crate) fn blend(self, t: Self, f: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = if self.0[i] != 0 { t.0[i] } else { f.0[i] };
        }
        I32x8(r)
    }
}

impl From<[i32; 8]> for I32x8 {
    #[inline(always)]
    fn from(a: [i32; 8]) -> Self {
        I32x8(a)
    }
}

impl core::ops::Add for I32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].wrapping_add(rhs.0[i]);
        }
        I32x8(r)
    }
}

impl core::ops::Sub for I32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].wrapping_sub(rhs.0[i]);
        }
        I32x8(r)
    }
}

impl core::ops::Mul for I32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].wrapping_mul(rhs.0[i]);
        }
        I32x8(r)
    }
}

impl core::ops::Neg for I32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i].wrapping_neg();
        }
        I32x8(r)
    }
}

/// Arithmetic right shift — sign-propagating, matching scalar `i32 >>`.
impl core::ops::Shr<I32x8> for I32x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: I32x8) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i] >> rhs.0[i];
        }
        I32x8(r)
    }
}

impl core::ops::Shl<I32x8> for I32x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: I32x8) -> Self {
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = self.0[i] << rhs.0[i];
        }
        I32x8(r)
    }
}

impl core::ops::AddAssign for I32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.0[i] = self.0[i].wrapping_add(rhs.0[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers (mirrors original load*/store* helpers).
// ---------------------------------------------------------------------------

/// Load 8 consecutive `i16` (sign-extended) into an `I32x8`.
#[inline(always)]
fn load8_i16(s: &[i16]) -> I32x8 {
    I32x8([
        s[0] as i32,
        s[1] as i32,
        s[2] as i32,
        s[3] as i32,
        s[4] as i32,
        s[5] as i32,
        s[6] as i32,
        s[7] as i32,
    ])
}

/// Load 8 consecutive `u8` (zero-extended) into an `I32x8`.
#[inline(always)]
fn load8_u8(s: &[u8]) -> I32x8 {
    I32x8([
        s[0] as i32,
        s[1] as i32,
        s[2] as i32,
        s[3] as i32,
        s[4] as i32,
        s[5] as i32,
        s[6] as i32,
        s[7] as i32,
    ])
}

/// Load 8 consecutive `i32` into an `I32x8`.
#[inline(always)]
fn load8_i32(s: &[i32]) -> I32x8 {
    I32x8([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// Store an `I32x8` to 8 consecutive `i32`.
#[inline(always)]
fn store8_i32(dst: &mut [i32], v: I32x8) {
    dst[..8].copy_from_slice(&v.0);
}

/// Load 8 consecutive pixels (`u8`/`u16`, zero-extended) into an `I32x8`.
#[inline(always)]
fn load8_pix<P: Pixel>(s: &[P]) -> I32x8 {
    I32x8([
        s[0].into(),
        s[1].into(),
        s[2].into(),
        s[3].into(),
        s[4].into(),
        s[5].into(),
        s[6].into(),
        s[7].into(),
    ])
}

/// Store an `I32x8`, clamped to `[0, bitdepth_max]` then narrowed.
#[inline(always)]
fn store8_clip<BD: BitDepth>(bd: BD, dst: &mut [BD::Pixel], v: I32x8) {
    let c = v.max(I32x8::splat(0)).min(I32x8::splat(bd.bitdepth_max()));
    for k in 0..8 {
        dst[k] = BD::Pixel::from_i32(c.0[k]);
    }
}

/// Store an `I32x8`, truncated with no clamp.
#[inline(always)]
fn store8_trunc<P: Pixel>(dst: &mut [P], v: I32x8) {
    for k in 0..8 {
        dst[k] = P::from_i32(v.0[k]);
    }
}

/// Arithmetic right shift of an `I32x8` by a uniform amount.
#[inline(always)]
fn sra(v: I32x8, sh: i32) -> I32x8 {
    v >> I32x8::splat(sh)
}

// ---------------------------------------------------------------------------
// Public kernel API (unchanged signatures).
// ---------------------------------------------------------------------------

/// `avg` row: `dst[x] = clip((tmp1[x] + tmp2[x] + rnd) >> sh)` for `x in 0..n`.
#[inline]
pub(crate) fn avg_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = I32x8::splat(rnd);
    let mut x = 0;
    while x + 8 <= n {
        let r = sra(load8_i16(&tmp1[x..]) + load8_i16(&tmp2[x..]) + rnd_v, sh);
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        dst[x] = bd.pixel_clip((tmp1[x] as i32 + tmp2[x] as i32 + rnd) >> sh);
        x += 1;
    }
}

/// `w_avg` row: `dst[x] = clip((tmp1[x]*weight + tmp2[x]*(16-weight) + rnd) >> sh)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn w_avg_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    n: usize,
    weight: i32,
    rnd: i32,
    sh: i32,
) {
    let w1 = I32x8::splat(weight);
    let w2 = I32x8::splat(16 - weight);
    let rnd_v = I32x8::splat(rnd);
    let mut x = 0;
    while x + 8 <= n {
        let r = sra(
            load8_i16(&tmp1[x..]) * w1 + load8_i16(&tmp2[x..]) * w2 + rnd_v,
            sh,
        );
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        dst[x] =
            bd.pixel_clip((tmp1[x] as i32 * weight + tmp2[x] as i32 * (16 - weight) + rnd) >> sh);
        x += 1;
    }
}

/// `mask` row: `dst[x] = clip((tmp1[x]*m + tmp2[x]*(64-m) + rnd) >> sh)`, `m = mask[x]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn mask_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    tmp1: &[i16],
    tmp2: &[i16],
    mask: &[u8],
    n: usize,
    rnd: i32,
    sh: i32,
) {
    let rnd_v = I32x8::splat(rnd);
    let c64 = I32x8::splat(64);
    let mut x = 0;
    while x + 8 <= n {
        let m = load8_u8(&mask[x..]);
        let r = sra(
            load8_i16(&tmp1[x..]) * m + load8_i16(&tmp2[x..]) * (c64 - m) + rnd_v,
            sh,
        );
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        dst[x] = bd.pixel_clip((tmp1[x] as i32 * m + tmp2[x] as i32 * (64 - m) + rnd) >> sh);
        x += 1;
    }
}

/// `blend` row: `dst[x] = (dst[x]*(64-m) + tmp[x]*m + 32) >> 6` (truncate, no clamp).
#[inline]
pub(crate) fn blend_row<P: Pixel>(dst: &mut [P], tmp: &[P], mask: &[u8], n: usize) {
    let c64 = I32x8::splat(64);
    let rnd_v = I32x8::splat(32);
    let mut x = 0;
    while x + 8 <= n {
        let m = load8_u8(&mask[x..]);
        let d = load8_pix(&dst[x..]);
        let t = load8_pix(&tmp[x..]);
        let r = sra(d * (c64 - m) + t * m + rnd_v, 6);
        store8_trunc(&mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let m = mask[x] as i32;
        let d: i32 = dst[x].into();
        let t: i32 = tmp[x].into();
        dst[x] = P::from_i32((d * (64 - m) + t * m + 32) >> 6);
        x += 1;
    }
}

/// `morph` row: `dst[x] = clip((alpha*dst[x] + beta) >> 8)`.
#[inline]
pub(crate) fn morph_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    alpha: i32,
    beta: i32,
    n: usize,
) {
    let a_v = I32x8::splat(alpha);
    let b_v = I32x8::splat(beta);
    let mut x = 0;
    while x + 8 <= n {
        let r = sra(load8_pix(&dst[x..]) * a_v + b_v, 8);
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let d: i32 = dst[x].into();
        dst[x] = bd.pixel_clip((alpha * d + beta) >> 8);
        x += 1;
    }
}

/// itx DC-only row: `dst[x] = clip(dst[x] + dc)` for `x in 0..n`.
#[inline]
pub(crate) fn dc_add_row<BD: BitDepth>(bd: BD, dst: &mut [BD::Pixel], dc: i32, n: usize) {
    let dc_v = I32x8::splat(dc);
    let mut x = 0;
    while x + 8 <= n {
        let r = load8_pix(&dst[x..]) + dc_v;
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let p: i32 = dst[x].into();
        dst[x] = bd.pixel_clip(p + dc);
        x += 1;
    }
}

/// itx row-clip pass: `tmp[i] = clip((tmp[i] + rnd) >> shift, min, max)` in place.
#[inline]
pub(crate) fn row_clip(tmp: &mut [i32], n: usize, rnd: i32, shift: i32, min: i32, max: i32) {
    let rnd_v = I32x8::splat(rnd);
    let min_v = I32x8::splat(min);
    let max_v = I32x8::splat(max);
    let mut i = 0;
    while i + 8 <= n {
        let v = sra(load8_i32(&tmp[i..]) + rnd_v, shift)
            .max(min_v)
            .min(max_v);
        store8_i32(&mut tmp[i..], v);
        i += 8;
    }
    while i < n {
        tmp[i] = ((tmp[i] + rnd) >> shift).max(min).min(max);
        i += 1;
    }
}

/// itx plain residual-add row: `dst[x] = clip(dst[x] + ((c[x]+rnd)>>shift))`.
#[inline]
pub(crate) fn residual_add_row<BD: BitDepth>(
    bd: BD,
    dst: &mut [BD::Pixel],
    c: &[i32],
    n: usize,
    rnd: i32,
    shift: i32,
) {
    let rnd_v = I32x8::splat(rnd);
    let mut x = 0;
    while x + 8 <= n {
        let cf = sra(load8_i32(&c[x..]) + rnd_v, shift);
        let r = load8_pix(&dst[x..]) + cf;
        store8_clip(bd, &mut dst[x..], r);
        x += 8;
    }
    while x < n {
        let p: i32 = dst[x].into();
        dst[x] = bd.pixel_clip(p + ((c[x] + rnd) >> shift));
        x += 1;
    }
}

/// `cctx` row: cross-component-transform rotate + clip over two i32 planes.
/// `u'[i] = iclip((u*cosa - v*sina + 128 - (a<0)) >> 8, min, max)`,
/// `v'[i] = iclip((u*sina + v*cosa + 128 - (b<0)) >> 8, min, max)`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cctx_row(
    u: &mut [i32],
    v: &mut [i32],
    sina: i32,
    cosa: i32,
    sz: usize,
    min: i32,
    max: i32,
) {
    let sina_v = I32x8::splat(sina);
    let cosa_v = I32x8::splat(cosa);
    let c128 = I32x8::splat(128);
    let zero = I32x8::splat(0);
    let min_v = I32x8::splat(min);
    let max_v = I32x8::splat(max);
    let mut i = 0;
    while i + 8 <= sz {
        let uu = load8_i32(&u[i..]);
        let vv = load8_i32(&v[i..]);
        let a = uu * cosa_v - vv * sina_v;
        let b = uu * sina_v + vv * cosa_v;
        // `a.cmp_lt(zero)` yields -1 lanes where a<0, i.e. `+ (-1)` == `- (a<0)`.
        let ra = sra(a + c128 + a.cmp_lt(zero), 8).max(min_v).min(max_v);
        let rb = sra(b + c128 + b.cmp_lt(zero), 8).max(min_v).min(max_v);
        store8_i32(&mut u[i..], ra);
        store8_i32(&mut v[i..], rb);
        i += 8;
    }
    while i < sz {
        let a = u[i] * cosa - v[i] * sina;
        let b = u[i] * sina + v[i] * cosa;
        u[i] = ((a + 128 - (a < 0) as i32) >> 8).max(min).min(max);
        v[i] = ((b + 128 - (b < 0) as i32) >> 8).max(min).min(max);
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Loop-restoration FIR kernels.
// ---------------------------------------------------------------------------

/// One symmetric FIR tap: `a` is read from `row_p` at `+dx`, `b` from `row_m`
/// at `-dx` (relative to the per-pixel column `o + x`).
pub(crate) struct WienerTap<'a> {
    pub row_p: &'a [u8],
    pub row_m: &'a [u8],
    pub dx: i32,
    pub coef: i32,
}

// ---------------------------------------------------------------------------
// Loop-restoration FIR dispatch (mirrors the itx neon/sse/scalar pattern).
//
// `ns_wiener_fir_run()` / `pc_wiener_fir_run()` return a cached function
// pointer chosen once at runtime: hand-written NEON on aarch64, the portable
// I32x8 path on x86, and a pure-scalar fallback everywhere else. Callers fetch
// the pointer once per row run, exactly like `idct_dequant_4x4()` in itx_2d.
// ---------------------------------------------------------------------------

type NsWienerFirFn = fn(&mut [u8], &[u8], usize, &[WienerTap<'_>], usize);
type PcWienerFirFn = fn(&mut [u8], &[u8], i32, usize, &[WienerTap<'_>], usize);

static NS_WIENER_FIR: std::sync::OnceLock<NsWienerFirFn> = std::sync::OnceLock::new();
static PC_WIENER_FIR: std::sync::OnceLock<PcWienerFirFn> = std::sync::OnceLock::new();

#[inline]
pub(crate) fn ns_wiener_fir_run() -> NsWienerFirFn {
    *NS_WIENER_FIR.get_or_init(|| {
        let mut f = ns_wiener_fir_run_scalar as NsWienerFirFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::ns_wiener_fir_run_neon as NsWienerFirFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::ns_wiener_fir_run_sse41 as NsWienerFirFn;
            }
        }
        f
    })
}

#[inline]
pub(crate) fn pc_wiener_fir_run() -> PcWienerFirFn {
    *PC_WIENER_FIR.get_or_init(|| {
        let mut f = pc_wiener_fir_run_scalar as PcWienerFirFn;
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                f = crate::neon::pc_wiener_fir_run_neon as PcWienerFirFn;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse4.1") {
                f = crate::sse::pc_wiener_fir_run_sse41 as PcWienerFirFn;
            }
        }
        f
    })
}

/// Pure-scalar "NS" Wiener FIR — the reference implementation and the fallback
/// for targets without a hand-written SIMD kernel.
pub(crate) fn ns_wiener_fir_run_scalar(
    dst: &mut [u8],
    center: &[u8],
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    for x in 0..n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m << 7;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b - 2 * m) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
    }
}

/// Pure-scalar "PC" Wiener FIR.
pub(crate) fn pc_wiener_fir_run_scalar(
    dst: &mut [u8],
    center: &[u8],
    center_coef: i32,
    col0: usize,
    taps: &[WienerTap],
    n: usize,
) {
    for x in 0..n {
        let c = col0 + x;
        let m = center[c] as i32;
        let mut s = m * center_coef;
        for t in taps {
            let a = t.row_p[(c as i32 + t.dx) as usize] as i32;
            let b = t.row_m[(c as i32 - t.dx) as usize] as i32;
            s += (a + b) * t.coef;
        }
        dst[x] = ((s + 64) >> 7).clamp(0, 255) as u8;
    }
}

/// GDF residual add over a run of `n` consecutive pixels.
#[inline]
pub(crate) fn gdf_add_run(dst: &mut [u8], err: &[i8], scale: i32, n: usize) {
    let rnd = I32x8::splat(8);
    let sc = I32x8::splat(scale);
    let zero = I32x8::splat(0);
    let mut x = 0;
    while x + 8 <= n {
        let diff = I32x8::from([
            err[x] as i32,
            err[x + 1] as i32,
            err[x + 2] as i32,
            err[x + 3] as i32,
            err[x + 4] as i32,
            err[x + 5] as i32,
            err[x + 6] as i32,
            err[x + 7] as i32,
        ]) * sc;
        let mag = sra(diff.abs() + rnd, 4);
        // apply_sign: negate where diff < 0.
        let neg = diff.cmp_lt(zero);
        let adj = neg.blend(zero - mag, mag);
        let d = load8_u8(&dst[x..]) + adj;
        let v = d.max(zero).min(I32x8::splat(255));
        let arr = v.to_array();
        for k in 0..8 {
            dst[x + k] = arr[k] as u8;
        }
        x += 8;
    }
    while x < n {
        let diff = err[x] as i32 * scale;
        let mag = (diff.abs() + 8) >> 4;
        let adj = if diff < 0 { -mag } else { mag };
        dst[x] = (dst[x] as i32 + adj).clamp(0, 255) as u8;
        x += 1;
    }
}

/// GDF gradient: accumulate per-column gradient into 8 lanes, then pair-reduce.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn gdf_gradient_group(
    dst: &mut [[u16; 4]],
    d: usize,
    base_cell: usize,
    ncells: usize,
    center_rows: [&[u8]; 2],
    a_rows: [&[u8]; 2],
    c_rows: [&[u8]; 2],
    col0: usize,
    dx: i32,
    shift: u32,
) {
    let sh = I32x8::splat(shift as i32);
    let mut acc = I32x8::splat(0);
    for y in 0..2 {
        let bcol = col0 - 1;
        let b = load8_u8(&center_rows[y][bcol..]) >> sh;
        let acol = (bcol as i32 - dx) as usize;
        let ccol = (bcol as i32 + dx) as usize;
        let a = load8_u8(&a_rows[y][acol..]) >> sh;
        let c = load8_u8(&c_rows[y][ccol..]) >> sh;
        acc += (b + b - a - c).abs();
    }
    let arr = acc.to_array();
    for k in 0..ncells {
        dst[base_cell + k][d] = (arr[2 * k] + arr[2 * k + 1]) as u16;
    }
}

#[cfg(test)]
mod wiener_scalar_proof {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn u8(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % ((hi - lo) as u64 + 1)) as i32
        }
    }

    fn buf(rng: &mut Rng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.u8()).collect()
    }

    #[test]
    fn ns_wiener_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15A);
        let f = ns_wiener_fir_run();
        for _ in 0..400 {
            let len = 256usize;
            let center = buf(&mut rng, len);
            let n_taps = rng.range(1, 8) as usize;
            let rows: Vec<(Vec<u8>, Vec<u8>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        buf(&mut rng, len),
                        buf(&mut rng, len),
                        rng.range(1, 16),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let taps: Vec<WienerTap> = rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTap {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let col0 = 64usize;
            let n = rng.range(1, 100) as usize;
            let mut d_ref = vec![0u8; n];
            let mut d_dsp = vec![0u8; n];
            ns_wiener_fir_run_scalar(&mut d_ref, &center, col0, &taps, n);
            f(&mut d_dsp, &center, col0, &taps, n);
            assert_eq!(d_ref, d_dsp, "ns dispatch mismatch n={} taps={}", n, n_taps);
        }
    }

    #[test]
    fn pc_wiener_dispatch_matches_scalar() {
        let mut rng = Rng(0xD15B);
        let f = pc_wiener_fir_run();
        for _ in 0..400 {
            let len = 256usize;
            let center = buf(&mut rng, len);
            let center_coef = rng.range(-128, 128);
            let n_taps = rng.range(1, 6) as usize;
            let rows: Vec<(Vec<u8>, Vec<u8>, i32, i32)> = (0..n_taps)
                .map(|_| {
                    (
                        buf(&mut rng, len),
                        buf(&mut rng, len),
                        rng.range(1, 16),
                        rng.range(-64, 64),
                    )
                })
                .collect();
            let taps: Vec<WienerTap> = rows
                .iter()
                .map(|(p, m, dx, coef)| WienerTap {
                    row_p: p,
                    row_m: m,
                    dx: *dx,
                    coef: *coef,
                })
                .collect();
            let col0 = 64usize;
            let n = rng.range(1, 100) as usize;
            let mut d_ref = vec![0u8; n];
            let mut d_dsp = vec![0u8; n];
            pc_wiener_fir_run_scalar(&mut d_ref, &center, center_coef, col0, &taps, n);
            f(&mut d_dsp, &center, center_coef, col0, &taps, n);
            assert_eq!(d_ref, d_dsp, "pc dispatch mismatch n={} taps={}", n, n_taps);
        }
    }
}
