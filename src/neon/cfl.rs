use core::arch::aarch64::*;

#[inline(always)]
fn predict_one(dc: i32, alpha: i32, ac: i32) -> u8 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 1024) >> 11;
    let signed = if diff < 0 { -mag } else { mag };
    (dc + signed).clamp(0, 255) as u8
}

#[inline(always)]
fn load_u8x16(a: &[u8; 16]) -> uint8x16_t {
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline(always)]
fn store_u8x8(a: &mut [u8; 8], v: uint8x8_t) {
    unsafe { vst1_u8(a.as_mut_ptr(), v) };
}

/// Form 8 mean-removed AC lanes:
///
///     ac = (sum2x2 << 1) - dc0
///
/// For 8-bit this fits comfortably in i16.
#[inline]
#[target_feature(enable = "neon")]
fn ac8_420_i16(top: uint8x16_t, bot: uint8x16_t, dc0v: int16x8_t) -> int16x8_t {
    let top_pairs = vpaddlq_u8(top); // u16x8, adjacent horizontal pairs
    let bot_pairs = vpaddlq_u8(bot); // u16x8

    let sum2x2 = vaddq_u16(top_pairs, bot_pairs); // <= 1020
    let sum2x2_x2 = vshlq_n_u16::<1>(sum2x2); // <= 2040

    vsubq_s16(vreinterpretq_s16_u16(sum2x2_x2), dc0v)
}

/// Apply alpha to 8 i16 AC lanes.
///
/// Only this function widens to i32, because `alpha * ac` may need i32.
/// Everything before this stays i16.
#[inline]
#[target_feature(enable = "neon")]
fn apply8_i16_ac(
    ac: int16x8_t,
    alpha_v: int16x4_t,
    dc_v: int32x4_t,
    round_v: int32x4_t,
    zero_v: int32x4_t,
) -> uint8x8_t {
    let ac_lo = vget_low_s16(ac);
    let ac_hi = vget_high_s16(ac);

    // i16 * i16 -> i32. This is the only widening part.
    let diff_lo = vmull_s16(ac_lo, alpha_v);
    let diff_hi = vmull_s16(ac_hi, alpha_v);

    let mag_lo = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff_lo), round_v));
    let mag_hi = vshrq_n_s32::<11>(vaddq_s32(vabsq_s32(diff_hi), round_v));

    let signed_lo = vbslq_s32(vcltq_s32(diff_lo, zero_v), vnegq_s32(mag_lo), mag_lo);
    let signed_hi = vbslq_s32(vcltq_s32(diff_hi, zero_v), vnegq_s32(mag_hi), mag_hi);

    let val_lo = vaddq_s32(dc_v, signed_lo);
    let val_hi = vaddq_s32(dc_v, signed_hi);

    vqmovn_u16(vcombine_u16(vqmovun_s32(val_lo), vqmovun_s32(val_hi)))
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
fn cfl_apply_420_8bpc_neon_impl(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
) {
    let do_u = alpha0 != 0;
    let do_v = alpha1 != 0;

    if !do_u && !do_v {
        return;
    }

    assert_ne!(xlim, 0);
    assert_ne!(ylim, 0);

    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&dc0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha0));
    assert!((i16::MIN as i32..=i16::MAX as i32).contains(&alpha1));

    let nfull = xlim / 8;
    let xfull = nfull * 8;
    let lfull = nfull * 16;

    let dc0v = vdupq_n_s16(dc0 as i16);

    let alpha0v = vdup_n_s16(alpha0 as i16);
    let alpha1v = vdup_n_s16(alpha1 as i16);

    let dc1v = vdupq_n_s32(dc1);
    let dc2v = vdupq_n_s32(dc2);

    let round_v = vdupq_n_s32(1024);
    let zero_v = vdupq_n_s32(0);

    let mut yrow = yrow0;
    let mut urow = urow0;
    let mut vrow = vrow0;

    for _y in 0..ylim {
        let top = y[yrow..yrow + lfull].as_chunks::<16>().0;
        let bot = y[yrow + ystride..yrow + ystride + lfull]
            .as_chunks::<16>()
            .0;

        match (do_u, do_v) {
            (true, true) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for (((du, dv), t), b) in u_chunks
                    .iter_mut()
                    .zip(v_chunks.iter_mut())
                    .zip(top.iter())
                    .zip(bot.iter())
                {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), dc0v);

                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, round_v, zero_v));
                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, round_v, zero_v));
                }
            }

            (true, false) => {
                let u_chunks = u[urow..urow + xfull].as_chunks_mut::<8>().0;

                for ((du, t), b) in u_chunks.iter_mut().zip(top.iter()).zip(bot.iter()) {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), dc0v);

                    store_u8x8(du, apply8_i16_ac(ac, alpha0v, dc1v, round_v, zero_v));
                }
            }

            (false, true) => {
                let v_chunks = v[vrow..vrow + xfull].as_chunks_mut::<8>().0;

                for ((dv, t), b) in v_chunks.iter_mut().zip(top.iter()).zip(bot.iter()) {
                    let ac = ac8_420_i16(load_u8x16(t), load_u8x16(b), dc0v);

                    store_u8x8(dv, apply8_i16_ac(ac, alpha1v, dc2v, round_v, zero_v));
                }
            }

            (false, false) => unreachable!(),
        }

        for x in xfull..xlim {
            let xl = x << 1;

            let ac = ((y[yrow + xl] as i32
                + y[yrow + xl + 1] as i32
                + y[yrow + xl + ystride] as i32
                + y[yrow + xl + ystride + 1] as i32)
                << 1)
                - dc0;

            if do_u {
                u[urow + x] = predict_one(dc1, alpha0, ac);
            }
            if do_v {
                v[vrow + x] = predict_one(dc2, alpha1, ac);
            }
        }

        if do_u {
            let last = u[urow + xlim - 1];
            u[urow + xlim..urow + w].fill(last);
        }

        if do_v {
            let last = v[vrow + xlim - 1];
            v[vrow + xlim..vrow + w].fill(last);
        }

        yrow += ystride << 1;
        urow += cstride;
        vrow += cstride;
    }

    if do_u {
        let src = urow0 + (ylim - 1) * cstride;
        for yy in ylim..h {
            let dst = urow0 + yy * cstride;
            u.copy_within(src..src + w, dst);
        }
    }

    if do_v {
        let src = vrow0 + (ylim - 1) * cstride;
        for yy in ylim..h {
            let dst = vrow0 + yy * cstride;
            v.copy_within(src..src + w, dst);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_apply_420_8bpc_neon(
    y: &[u8],
    u: &mut [u8],
    v: &mut [u8],
    yrow0: usize,
    urow0: usize,
    vrow0: usize,
    ystride: usize,
    cstride: usize,
    w: usize,
    h: usize,
    xlim: usize,
    ylim: usize,
    dc0: i32,
    dc1: i32,
    dc2: i32,
    alpha0: i32,
    alpha1: i32,
) {
    unsafe {
        cfl_apply_420_8bpc_neon_impl(
            y, u, v, yrow0, urow0, vrow0, ystride, cstride, w, h, xlim, ylim, dc0, dc1, dc2,
            alpha0, alpha1,
        )
    }
}
