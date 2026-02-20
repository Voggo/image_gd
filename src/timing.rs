use log::Level;
use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Instant;

fn parse_bool_env(value: &str) -> bool {
    !matches!(
        value,
        "0" | "false" | "FALSE" | "False" | "no" | "NO" | "off" | "OFF"
    )
}

fn timing_enabled() -> bool {
    static TIMING_ENABLED: OnceLock<bool> = OnceLock::new();
    *TIMING_ENABLED.get_or_init(|| {
        std::env::var("ENTRO_GD_TIMING")
            .map(|v| parse_bool_env(v.as_str()))
            .unwrap_or(true)
    })
}

/// Scope-based timer that logs elapsed time when dropped.
///
/// Behavior:
/// - Respects log level filtering (`RUST_LOG`).
/// - Can be globally disabled with `ENTRO_GD_TIMING=0`.
/// - Has near-zero overhead when disabled.
pub struct ScopedTimer {
    start: Option<Instant>,
    label: Cow<'static, str>,
    level: Level,
}

impl ScopedTimer {
    pub fn new(level: Level, label: impl Into<Cow<'static, str>>) -> Self {
        let enabled = timing_enabled() && log::log_enabled!(level);
        ScopedTimer {
            start: enabled.then(Instant::now),
            label: label.into(),
            level,
        }
    }

    pub fn info(label: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Info, label)
    }

    pub fn debug(label: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Debug, label)
    }

    pub fn trace(label: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Trace, label)
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            log::log!(self.level, "{} completed in {:.3?}", self.label, elapsed);
        }
    }
}
