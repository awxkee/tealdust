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
        let mut c = 40 - self.cnt;
        debug_assert!(c >= 0);

        let start = self.buf_pos;
        let len = self.buf.len();

        if start >= len {
            return;
        }

        let max_read = (c as usize >> 3) + 1;
        let n = max_read.min(len - start);

        let src = &self.buf[start..start + n];

        let mut dif = self.dif;

        for &byte in src {
            dif ^= (byte as u64) << c;
            c -= 8;
        }

        self.buf_pos = start + n;
        self.dif = dif;
        self.cnt = 40 - c;
    }

    #[inline]
    fn ctx_norm(&mut self, dif: u64, rng: u32) {
        let d = 15 ^ (31 ^ rng.leading_zeros());
        let cnt = self.cnt;
        debug_assert!(rng <= 65535);
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
        self.decode_bools_bypass(1)
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
                debug_assert!(cdf.len() > $n);

                let cdf_all: &mut [u16; $n + 1] = (&mut cdf[..($n + 1)])
                    .try_into()
                    .expect("invalid MSAC CDF length");

                let min_prob: &[u16; $n + 1] = (&MSAC_MIN_PROB[$n - 1][..($n + 1)])
                    .try_into()
                    .expect("invalid MSAC min-prob table length");

                let c = (self.dif >> 48) as u32;
                let r = self.rng >> 8;

                let mut u = self.rng;
                let mut v = self.rng;
                let mut val = 0u32;

                for (&cdf_i, &min_i) in cdf_all.iter().zip(min_prob.iter()) {
                    u = v;

                    let p_raw = (cdf_i | 127) as i32 - min_i as i32;
                    let p = p_raw.max(0) as u32;

                    v = ((r * p) >> 10) << 3;

                    if c >= v {
                        break;
                    }

                    val += 1;
                }

                debug_assert!(val <= $n);
                debug_assert!(u <= self.rng);

                self.ctx_norm(self.dif - ((v as u64) << 48), u - v);

                if self.allow_update_cdf {
                    let (cdf_syms, cdf_count) = cdf_all.split_at_mut($n);

                    let cdf_syms: &mut [u16; $n] =
                        cdf_syms.try_into().expect("invalid MSAC symbol CDF length");

                    let cdf_count: &mut [u16; 1] =
                        cdf_count.try_into().expect("invalid MSAC count CDF length");

                    let pc = cdf_count[0];
                    let count = (pc & 0xFF) as u8;

                    debug_assert!(count <= 32);

                    let rate = MSAC_RATE[(pc >> 8) as usize][(count >> 4) as usize]
                        + if $n > 2 { 1 } else { 0 };

                    let val_usize = val as usize;

                    for (i, cdf_i) in cdf_syms.iter_mut().enumerate() {
                        if i < val_usize {
                            *cdf_i += (32768 - *cdf_i) >> rate;
                        } else {
                            *cdf_i -= *cdf_i >> rate;
                        }
                    }

                    cdf_count[0] = pc + u16::from(count < 32);
                }

                val
            }};
        }

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
