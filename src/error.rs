use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TealdustError {
    Eof,
    Again,
    InvalidData,
    FrameTooLarge,
    InvalidParam,
    OutOfMemory,
}

impl fmt::Display for TealdustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eof => write!(f, "end of stream"),
            Self::Again => write!(f, "need more data"),
            Self::InvalidData => write!(f, "invalid or corrupt bitstream data"),
            Self::FrameTooLarge => write!(f, "frame dimensions exceed limit"),
            Self::InvalidParam => write!(f, "invalid parameter"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for TealdustError {}
