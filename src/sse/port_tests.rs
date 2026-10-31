/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

//! SSE4.1 ports of the 128-bit AVX2 kernels against their scalar references.

use crate::rowops_dispatch::{GdfPrepRow, gdf_prep_class};

struct Rng(u64);

impl Rng {
    fn r(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32 % n
    }
}

fn have() -> bool {
    std::is_x86_feature_detected!("sse4.1")
}

#[test]
fn cdef_padding_matches_scalar() {
    if !have() {
        return;
    }
    let mut rng = Rng(0x00c0_ffee);
    for &(w, h) in &[(4usize, 4usize), (8, 8), (4, 8), (8, 4)] {
        for edges in 0..16u8 {
            for &bits in &[8u32, 10, 12] {
                let max = 1 << bits;
                let stride = w + 9;
                let bottom_stride = stride + 3;
                let src: Vec<u16> = (0..stride * (h + 1)).map(|_| rng.r(max) as u16).collect();
                let top: Vec<u16> = (0..stride * 2 + 8).map(|_| rng.r(max) as u16).collect();
                let bottom: Vec<u16> = (0..bottom_stride * 2 + 8)
                    .map(|_| rng.r(max) as u16)
                    .collect();
                let left: Vec<[u16; 2]> = (0..h)
                    .map(|_| [rng.r(max) as u16, rng.r(max) as u16])
                    .collect();
                let tmp_stride = 16;
                let len = tmp_stride * (h + 4) + 4;
                let (mut a, mut b) = (vec![7i16; len], vec![7i16; len]);
                if bits == 8 {
                    let n8 = |v: &[u16]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
                    let (s8, t8, b8) = (n8(&src), n8(&top), n8(&bottom));
                    let l8: Vec<[u8; 2]> = left.iter().map(|p| [p[0] as u8, p[1] as u8]).collect();
                    crate::cdef_dispatch::cdef_padding_8bpc_scalar(
                        &mut a,
                        tmp_stride,
                        &s8,
                        stride,
                        2,
                        &l8,
                        &t8,
                        2,
                        &b8,
                        2,
                        bottom_stride,
                        w,
                        h,
                        edges,
                    );
                    unsafe {
                        super::cdef_padding_8bpc_sse41(
                            &mut b,
                            tmp_stride,
                            &s8,
                            stride,
                            2,
                            &l8,
                            &t8,
                            2,
                            &b8,
                            2,
                            bottom_stride,
                            w,
                            h,
                            edges,
                        )
                    };
                } else {
                    crate::cdef_dispatch::cdef_padding_hbd_scalar(
                        &mut a,
                        tmp_stride,
                        &src,
                        stride,
                        2,
                        &left,
                        &top,
                        2,
                        &bottom,
                        2,
                        bottom_stride,
                        w,
                        h,
                        edges,
                    );
                    unsafe {
                        super::cdef_padding_hbd_sse41(
                            &mut b,
                            tmp_stride,
                            &src,
                            stride,
                            2,
                            &left,
                            &top,
                            2,
                            &bottom,
                            2,
                            bottom_stride,
                            w,
                            h,
                            edges,
                        )
                    };
                }
                assert_eq!(a, b, "{w}x{h} edges={edges:04b} {bits}b");
            }
        }
    }
}

type Sb64Hbd = fn(&mut [u16], usize, usize, &[u16], &[u16], &[u8], &[u8], bool, i32);
type Sb64HbdSse = unsafe fn(&mut [u16], usize, usize, &[u16], &[u16], &[u8], &[u8], bool, i32);

#[test]
fn deblock_sb64_hbd_matches_scalar() {
    if !have() {
        return;
    }
    use crate::deblock_dispatch as dd;
    let cases: [(&str, Sb64Hbd, Sb64HbdSse); 4] = [
        (
            "h_y",
            dd::deblock_h_sb64y_hbd_fast,
            super::deblock_h_sb64y_hbd_sse41,
        ),
        (
            "v_y",
            dd::deblock_v_sb64y_hbd_fast,
            super::deblock_v_sb64y_hbd_sse41,
        ),
        (
            "h_uv",
            dd::deblock_h_sb64uv_hbd_fast,
            super::deblock_h_sb64uv_hbd_sse41,
        ),
        (
            "v_uv",
            dd::deblock_v_sb64uv_hbd_fast,
            super::deblock_v_sb64uv_hbd_sse41,
        ),
    ];
    let mut rng = Rng(0x0d0b_10c4);
    let stride = 96;
    for (name, scalar, sse) in cases {
        for iter in 0..400 {
            let bits = if iter & 1 == 0 { 10 } else { 12 };
            let max = (1u32 << bits) - 1;
            // Smooth content with steps so the filters both engage and skip.
            let mut v = rng.r(max) as i32;
            let plane: Vec<u16> = (0..stride * 96)
                .map(|i| {
                    if i % 7 == 0 {
                        v = rng.r(max) as i32;
                    }
                    v = (v + rng.r(9) as i32 - 4).clamp(0, max as i32);
                    v as u16
                })
                .collect();
            let vmask: [u16; 4] = core::array::from_fn(|_| rng.r(65536) as u16);
            let ll_mask: [u16; 2] = core::array::from_fn(|_| {
                if rng.r(4) == 0 {
                    rng.r(65536) as u16
                } else {
                    0
                }
            });
            let q_thr: [u8; 16] = core::array::from_fn(|_| rng.r(64) as u8);
            let side_thr: [u8; 16] = core::array::from_fn(|_| rng.r(64) as u8);
            let edge = rng.r(2) == 1;
            let off = 16 * stride + 16;
            let (mut a, mut b) = (plane.clone(), plane);
            scalar(
                &mut a, off, stride, &vmask, &ll_mask, &q_thr, &side_thr, edge, max as i32,
            );
            unsafe {
                sse(
                    &mut b, off, stride, &vmask, &ll_mask, &q_thr, &side_thr, edge, max as i32,
                )
            };
            assert!(a == b, "{name} iter {iter}");
        }
    }
}

#[test]
fn gdf_gradient_hbd_matches_scalar() {
    if !have() {
        return;
    }
    let mut rng = Rng(0x09ad_1e17);
    for _ in 0..500 {
        let bits = [10u32, 12][rng.r(2) as usize];
        let rows: Vec<Vec<u16>> = (0..6)
            .map(|_| (0..40).map(|_| rng.r(1 << bits) as u16).collect())
            .collect();
        let dx = rng.r(3) as i32 - 1;
        let col0 = 3 + rng.r(20) as usize;
        let d = rng.r(4) as usize;
        let ncells = 1 + rng.r(4) as usize;
        let shift = bits - 8;
        let pick = |i: usize| -> [&[u16]; 2] { [&rows[i][..], &rows[i + 1][..]] };
        let (mut a, mut b) = (vec![[1u16; 4]; 8], vec![[1u16; 4]; 8]);
        crate::rowops_dispatch::gdf_gradient_group_hbd_scalar(
            &mut a,
            d,
            1,
            ncells,
            pick(2),
            pick(0),
            pick(4),
            col0,
            dx,
            shift,
        );
        unsafe {
            super::gdf_gradient_group_hbd_sse41(
                &mut b,
                d,
                1,
                ncells,
                pick(2),
                pick(0),
                pick(4),
                col0,
                dx,
                shift,
            )
        };
        assert_eq!(a, b, "dx={dx} col0={col0} d={d} n={ncells}");
    }
}

fn prep_params(rng: &mut Rng, grad_shift: i32) -> GdfPrepRow {
    let ref_dst_idx = rng.r(6) as usize;
    let qp_idx = rng.r(6) as usize;
    let (error_lut_base, scale) = if ref_dst_idx == 0 {
        (qp_idx * 4096, 8)
    } else {
        ((ref_dst_idx - 1) * 6000 + qp_idx * 1000, 5)
    };
    GdfPrepRow {
        alpha_base: ref_dst_idx * 528 + qp_idx * 88,
        weight_base: ref_dst_idx * 1584 + qp_idx * 264,
        bias_idx: ref_dst_idx * 6 + qp_idx,
        error_lut_base,
        scale,
        ref_dst_idx,
        grad_shift,
    }
}

#[test]
fn gdf_prep_pair_matches_scalar() {
    if !have() {
        return;
    }
    let mut rng = Rng(0x0001_9a1f);
    for iter in 0..600 {
        let hbd = iter & 1 == 1;
        let bits = if hbd {
            [10u32, 12][rng.r(2) as usize]
        } else {
            8
        };
        let p = prep_params(&mut rng, if hbd { 4 } else { 2 });
        let grads: [Vec<[u16; 4]>; 2] = core::array::from_fn(|_| {
            (0..8)
                .map(|_| core::array::from_fn(|_| rng.r(2048) as u16))
                .collect()
        });
        let (cls, shared) = gdf_prep_class([&grads[0][..], &grads[1][..]], rng.r(4) as usize, &p);
        let rows: Vec<Vec<u16>> = (0..13)
            .map(|_| (0..40).map(|_| rng.r(1 << bits) as u16).collect())
            .collect();
        let col = 8 + rng.r(20) as usize;
        if hbd {
            let r: [&[u16]; 13] = core::array::from_fn(|i| &rows[i][..]);
            let down_shift = if bits == 12 { 2 } else { 0 };
            let a = crate::rowops_dispatch::gdf_prep_pair_hbd_scalar(
                r,
                col,
                cls,
                shared,
                p.alpha_base,
                p.weight_base,
                p.error_lut_base,
                p.scale,
                down_shift,
                1,
                p.ref_dst_idx,
            );
            let b = unsafe {
                super::gdf_prep_pair_hbd_sse41(
                    r,
                    col,
                    cls,
                    shared,
                    p.alpha_base,
                    p.weight_base,
                    p.error_lut_base,
                    p.scale,
                    down_shift,
                    1,
                    p.ref_dst_idx,
                )
            };
            assert_eq!(a, b, "hbd iter {iter}");
        } else {
            let r8: Vec<Vec<u8>> = rows
                .iter()
                .map(|v| v.iter().map(|&x| x as u8).collect())
                .collect();
            let r: [&[u8]; 13] = core::array::from_fn(|i| &r8[i][..]);
            let a = crate::rowops_dispatch::gdf_prep_pair_8bpc_scalar(
                r,
                col,
                cls,
                shared,
                p.alpha_base,
                p.weight_base,
                p.error_lut_base,
                p.scale,
                p.ref_dst_idx,
            );
            let b = unsafe {
                super::gdf_prep_pair_8bpc_sse41(
                    r,
                    col,
                    cls,
                    shared,
                    p.alpha_base,
                    p.weight_base,
                    p.error_lut_base,
                    p.scale,
                    p.ref_dst_idx,
                )
            };
            assert_eq!(a, b, "8bpc iter {iter}");
        }
    }
}

#[test]
fn deblock_sb64_8bpc_matches_per_edge() {
    if !have() {
        return;
    }
    use crate::pixel::BitDepth8;
    let fast = crate::exec_context::ExecContext::new();
    let mut slow = crate::exec_context::ExecContext::new();
    slow.deblock_h_sb64y_8bpc = None;
    slow.deblock_v_sb64y_8bpc = None;
    slow.deblock_h_sb64uv_8bpc = None;
    slow.deblock_v_sb64uv_8bpc = None;
    let mut rng = Rng(0x8b9c_d0e1);
    let stride = 96;
    for iter in 0..1600 {
        let mut v = rng.r(256) as i32;
        let plane: Vec<u8> = (0..stride * 96)
            .map(|i| {
                if i % 7 == 0 {
                    v = rng.r(256) as i32;
                }
                v = (v + rng.r(9) as i32 - 4).clamp(0, 255);
                v as u8
            })
            .collect();
        let vmask: [u16; 4] = core::array::from_fn(|_| rng.r(65536) as u16);
        let ll_mask: [u16; 2] = core::array::from_fn(|_| {
            if rng.r(4) == 0 {
                rng.r(65536) as u16
            } else {
                0
            }
        });
        let q_thr: [u8; 16] = core::array::from_fn(|_| rng.r(64) as u8);
        let side_thr: [u8; 16] = core::array::from_fn(|_| rng.r(64) as u8);
        let edge = rng.r(2) == 1;
        let off = 16 * stride + 16;
        let run = |exec: &crate::exec_context::ExecContext| {
            let mut p = plane.clone();
            let args = (&vmask[..], &ll_mask[..], &q_thr[..], &side_thr[..]);
            match iter % 4 {
                0 => crate::deblock::deblock_h_sb64y_bd(
                    BitDepth8, exec, &mut p, off, stride, args.0, args.1, args.2, args.3, edge,
                ),
                1 => crate::deblock::deblock_v_sb64y_bd(
                    BitDepth8, exec, &mut p, off, stride, args.0, args.1, args.2, args.3, edge,
                ),
                2 => crate::deblock::deblock_h_sb64uv_bd(
                    BitDepth8, exec, &mut p, off, stride, args.0, args.1, args.2, args.3, edge,
                ),
                _ => crate::deblock::deblock_v_sb64uv_bd(
                    BitDepth8, exec, &mut p, off, stride, args.0, args.1, args.2, args.3, edge,
                ),
            }
            p
        };
        assert!(run(&fast) == run(&slow), "variant {} iter {iter}", iter % 4);
    }
}

#[test]
fn warp_affine_matches_scalar() {
    use crate::mc_dispatch::{Warp8Fn, Warp8tFn, WarpHbdFn, WarpHbdTfn};
    if !have() {
        return;
    }
    let mut rng = Rng(0x5eed_3a7f_11c4_9d02);
    let stride = 40usize;
    let src_off = 4 * stride + 4;
    let s = |rng: &mut Rng, n: i32| rng.r(2 * n as u32 + 1) as i32 - n;
    for iter in 0..4000 {
        let abcd = [0; 4].map(|_| s(&mut rng, 4096) as i16);
        let (mx, my) = (s(&mut rng, 65536), s(&mut rng, 65536));
        // Mostly random pixels, sometimes saturated, to exercise the clips.
        let edge = iter % 4 == 0;
        let src8: Vec<u8> = (0..stride * 20)
            .map(|_| {
                if edge {
                    [0, 255][rng.r(2) as usize]
                } else {
                    rng.r(256) as u8
                }
            })
            .collect();
        let (mut a, mut b) = (vec![0u8; 8 * 16], vec![0u8; 8 * 16]);
        let f: Warp8Fn = crate::mc_dispatch::warp_affine_8x8_8bpc_scalar;
        unsafe {
            f(&mut a, 16, &src8, stride, src_off, &abcd, mx, my);
            super::warp_affine_8x8_8bpc_sse41(&mut b, 16, &src8, stride, src_off, &abcd, mx, my);
        }
        assert_eq!(a, b, "8bpc abcd={abcd:?} mx={mx} my={my}");
        let (mut a, mut b) = (vec![0i16; 8 * 16], vec![0i16; 8 * 16]);
        let f: Warp8tFn = crate::mc_dispatch::warp_affine_8x8t_8bpc_scalar;
        unsafe {
            f(&mut a, 16, &src8, stride, src_off, &abcd, mx, my);
            super::warp_affine_8x8t_8bpc_sse41(&mut b, 16, &src8, stride, src_off, &abcd, mx, my);
        }
        assert_eq!(a, b, "8bpc t abcd={abcd:?} mx={mx} my={my}");
        for bd in [10u8, 12] {
            let max = (1u32 << bd) - 1;
            let src16: Vec<u16> = (0..stride * 20)
                .map(|_| if edge { [0, max][rng.r(2) as usize] } else { rng.r(max + 1) } as u16)
                .collect();
            let (mut a, mut b) = (vec![0u16; 8 * 16], vec![0u16; 8 * 16]);
            let f: WarpHbdFn = crate::mc_dispatch::warp_affine_8x8_hbd_scalar;
            unsafe {
                f(&mut a, 16, &src16, stride, src_off, &abcd, mx, my, bd);
                super::warp_affine_8x8_hbd_sse41(
                    &mut b, 16, &src16, stride, src_off, &abcd, mx, my, bd,
                );
            }
            assert_eq!(a, b, "hbd{bd} abcd={abcd:?} mx={mx} my={my}");
            let (mut a, mut b) = (vec![0i16; 8 * 16], vec![0i16; 8 * 16]);
            let f: WarpHbdTfn = crate::mc_dispatch::warp_affine_8x8t_hbd_scalar;
            unsafe {
                f(&mut a, 16, &src16, stride, src_off, &abcd, mx, my, bd);
                super::warp_affine_8x8t_hbd_sse41(
                    &mut b, 16, &src16, stride, src_off, &abcd, mx, my, bd,
                );
            }
            assert_eq!(a, b, "hbd{bd} t abcd={abcd:?} mx={mx} my={my}");
        }
    }
}

#[test]
fn gdf_add_run_hbd_matches_scalar() {
    if !have() {
        return;
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..2000 {
        let bd_max = [0xff, 0x3ff, 0xfff][rng.r(3) as usize];
        let n = rng.r(70) as usize;
        let scale = rng.r(1 << 12) as i32 - (1 << 11);
        let err: Vec<i8> = (0..n).map(|_| rng.r(256) as u8 as i8).collect();
        let d0: Vec<u16> = (0..n).map(|_| rng.r(bd_max as u32 + 1) as u16).collect();
        let (mut a, mut b) = (d0.clone(), d0);
        crate::rowops_dispatch::gdf_add_run_hbd_scalar(&mut a, &err, scale, n, bd_max);
        unsafe { super::gdf_add_run_hbd_sse41(&mut b, &err, scale, n, bd_max) };
        assert_eq!(a, b, "bd_max={bd_max} n={n} scale={scale}");
    }
}
