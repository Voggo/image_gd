# AGENTS.md

Guidance for coding agents working in `entro_gd`.

## Project Snapshot

- **Language:** Rust (`edition = "2024"`)
- **Crate type:** Library crate with examples, tests, and criterion benches.

## Commands and Verification

Prefer specific validation. Standard `cargo build` and `cargo check` apply, but mind the features.

- **Check everything:** `cargo check --all-targets --all-features`
- **Lint strictly:** `cargo clippy --all-targets --all-features`
- **Run all tests:** `cargo test --all-targets --all-features`

### High-Signal Run Commands

Many examples require specific file paths.

- **CSV Demo:** `cargo run --example load_csv`
- **CSV Compress:** `cargo run --example compress_decompress_example -- data/data-10000-8-int.csv`
- **Image Compress:** `cargo run --example compress_decompress_image_example -- data/images/rustacean.png`

**Benchmarks:**
- Main: `cargo bench --bench benchmark`
- Filter by name: `cargo bench --bench benchmark -- <Substring>`

## Repository Conventions

- **Error Handling:** Central error type is `EntroGdError` (`src/error.rs`). Do not invent new top-level error types.
- **Diagnostics & Profiling:** Use `tracing` macros for logs, and `ScopedTimer` for scoped performance instrumentation.
- **Environment Controls:** 
  - `ENTRO_GD_TIMING` toggles timers.
  - `ENTRO_GD_LOG_DIR` sets the file log location.
  - `RUST_LOG` handles log filtering.
- **Bitwise Logic:** `use bitvec::prelude::*;` is idiomatic in bit-heavy modules.

## Artifact and Repo Hygiene

- Do not commit generated artifacts under ignored directories (`target/`, `logs/`, or generated compressed outputs inside `data/**`).
- Keep temporary benchmark/experiment output in `target/` or ignored paths.
