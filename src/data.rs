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

use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct DataProps {
    pub timestamp: i64,
    pub duration: i64,
    pub offset: i64,
    pub size: usize,
}

impl DataProps {
    pub fn new() -> Self {
        Self {
            timestamp: i64::MIN,
            offset: -1,
            ..Default::default()
        }
    }

    pub fn copy_from(&mut self, src: &DataProps) {
        self.timestamp = src.timestamp;
        self.duration = src.duration;
        self.offset = src.offset;
        self.size = src.size;
    }
}

/// A reference-counted byte buffer for compressed bitstream data.
#[derive(Clone)]
pub struct Data {
    buf: Option<Arc<Vec<u8>>>,
    offset: usize,
    len: usize,
    pub props: DataProps,
}

impl Data {
    pub fn new() -> Self {
        Self {
            buf: None,
            offset: 0,
            len: 0,
            props: DataProps::new(),
        }
    }

    pub fn create(size: usize) -> Option<Self> {
        if size > usize::MAX / 2 {
            return None;
        }
        let buf = vec![0u8; size];
        Some(Self {
            buf: Some(Arc::new(buf)),
            offset: 0,
            len: size,
            props: DataProps {
                size,
                ..DataProps::new()
            },
        })
    }

    pub fn wrap(data: Vec<u8>) -> Self {
        let len = data.len();
        Self {
            buf: Some(Arc::new(data)),
            offset: 0,
            len,
            props: DataProps {
                size: len,
                ..DataProps::new()
            },
        }
    }

    pub fn data(&self) -> Option<&[u8]> {
        self.buf
            .as_ref()
            .map(|b| &b[self.offset..self.offset + self.len])
    }

    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        let offset = self.offset;
        let len = self.len;
        self.buf
            .as_mut()
            .and_then(|b| Arc::get_mut(b).map(|v| &mut v[offset..offset + len]))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.buf.is_none()
    }

    pub fn has_data(&self) -> bool {
        self.buf.is_some()
    }

    pub fn consume(&mut self, n: usize) {
        assert!(n <= self.len);
        self.offset += n;
        self.len -= n;
        if self.len == 0 {
            self.unref();
        }
    }

    pub fn unref(&mut self) {
        self.buf = None;
        self.offset = 0;
        self.len = 0;
        self.props = DataProps::new();
    }
}

impl Default for Data {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Data")
            .field("len", &self.len)
            .field("has_data", &self.buf.is_some())
            .finish()
    }
}
