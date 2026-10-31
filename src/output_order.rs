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

//! Frame output order, following AVM's `av2_output_frame_buffers`.
//!
//! A decoded frame is output when it is shown immediately or via a
//! show-existing frame. Frames coded with `show_implicit` wait in the
//! reference slots and are output in display order: before the next output
//! that follows them, right after the output they directly follow, when their
//! slot is about to be overwritten, or at end of stream.

use crate::internal::{DecoderContext, ThreadPicture};
use std::sync::atomic::Ordering::Relaxed;

/// AVM `get_disp_order_hint`: unwraps `order_hint` against the largest
/// display order among the stored frames that are output.
pub(crate) fn disp_order_hint(
    c: &DecoderContext,
    order_hint: u32,
    order_hint_bits: u32,
    closed_loop_key: bool,
) -> i64 {
    if closed_loop_key {
        return order_hint as i64;
    }
    let max = c
        .refs
        .iter()
        .filter(|r| r.p.pic.is_some())
        .filter(|r| {
            r.p.frame_hdr
                .as_ref()
                .is_some_and(|h| h.show_immediate != 0 || h.show_implicit != 0)
        })
        .map(|r| r.p.disp_order)
        .max()
        .unwrap_or(0)
        .max(0);
    let factor = 1i64 << order_hint_bits;
    let mut cur = order_hint as i64;
    while (max - cur).abs() >= factor >> 1 {
        if cur > max {
            return cur;
        }
        cur += factor;
    }
    cur
}

/// AVM `is_frame_eligible_for_output`: an implicitly shown frame that has not
/// been output yet.
fn eligible(p: &ThreadPicture) -> bool {
    p.pic.is_some()
        && p.frame_hdr.as_ref().is_some_and(|h| h.show_implicit != 0)
        && p.out_done.as_ref().is_some_and(|d| !d.load(Relaxed))
}

fn emit(c: &mut DecoderContext, p: &ThreadPicture) {
    let Some(pic) = p.pic.as_ref() else {
        return;
    };
    if let Some(d) = p.out_done.as_ref() {
        d.store(true, Relaxed);
    }
    c.frame_out.push(pic.shallow_clone());
}

/// The eligible slot with the smallest display order matching `pred`.
fn earliest(c: &DecoderContext, pred: impl Fn(i64) -> bool) -> Option<usize> {
    (0..c.refs.len())
        .filter(|&i| eligible(&c.refs[i].p) && pred(c.refs[i].p.disp_order))
        .min_by_key(|&i| c.refs[i].p.disp_order)
}

/// Outputs `trigger`, preceded by every waiting frame that comes before it in
/// display order and followed by the waiting frames that directly follow it.
pub(crate) fn output_frame_buffers(c: &mut DecoderContext, trigger: &ThreadPicture) {
    while let Some(i) = earliest(c, |d| d < trigger.disp_order) {
        let p = c.refs[i].p.clone();
        emit(c, &p);
    }
    emit(c, trigger);
    let mut next = trigger.disp_order + 1;
    while let Some(i) = earliest(c, |d| d == next) {
        let p = c.refs[i].p.clone();
        emit(c, &p);
        next += 1;
    }
}

/// Before reference slot `slot` is overwritten: output the frame it holds if
/// that frame is still waiting.
pub(crate) fn output_before_overwrite(c: &mut DecoderContext, slot: usize) {
    if eligible(&c.refs[slot].p) {
        let p = c.refs[slot].p.clone();
        output_frame_buffers(c, &p);
    }
}

/// End of stream: outputs every waiting frame in display order.
pub(crate) fn flush_remaining(c: &mut DecoderContext) {
    while let Some(i) = earliest(c, |_| true) {
        let p = c.refs[i].p.clone();
        emit(c, &p);
    }
}
