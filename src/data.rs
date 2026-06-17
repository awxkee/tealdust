use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct DataProps {
    pub timestamp: i64,
    pub duration: i64,
    pub offset: i64,
    pub size: usize,
    pub user_data: Option<Arc<UserData>>,
}

pub struct UserData {
    // Opaque user pointer stored as an address so the wrapper itself does not
    // need manual Send/Sync impls. The address is only converted back to a raw
    // pointer when returned to the caller or passed to the user callback.
    data_addr: usize,
    free_callback: Box<dyn Fn(*const u8) + Send + Sync>,
}

impl UserData {
    pub fn new(data: *const u8, free_callback: Box<dyn Fn(*const u8) + Send + Sync>) -> Self {
        Self {
            data_addr: data as usize,
            free_callback,
        }
    }

    pub fn data(&self) -> *const u8 {
        self.data_addr as *const u8
    }
}

impl Drop for UserData {
    fn drop(&mut self) {
        (self.free_callback)(self.data());
    }
}

impl std::fmt::Debug for UserData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserData")
            .field("data", &self.data())
            .finish()
    }
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
        self.user_data = src.user_data.clone();
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
