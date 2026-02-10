pub mod error;
pub mod data_loader;
pub mod preprocessor;
pub mod compression;

pub use error::EntroGdError;

/// Initialize logging using env_logger. Safe to call multiple times.
pub fn init_logging() {
	let _ = env_logger::builder()
		.is_test(cfg!(test))
		.try_init();
}
