// Conditional logging support: uses log crate when enabled, eprintln fallback when disabled.

#[cfg(feature = "log")]
pub use log::warn;

#[cfg(not(feature = "log"))]
macro_rules! warn_fallback {
    ($($arg:tt)*) => {
        eprintln!($($arg)*)
    };
}

#[cfg(not(feature = "log"))]
pub(crate) use warn_fallback as warn;
