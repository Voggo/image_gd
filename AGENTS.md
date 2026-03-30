# AGENTS.md

Guidance for coding agents working in `entro_gd`.

## Project Snapshot

- Language: Rust (`edition = "2024"`)
- Crate type: library crate with examples, tests, and criterion benches
- Primary build tool: `cargo`
- Optional feature flag: `experiment-runner`
- Key folders:
  - `src/` core library and modules
  - `tests/` integration tests
  - `examples/` runnable workflows
  - `benches/` criterion benchmarks
  - `configs/` experiment profile configuration

## Cursor/Copilot Rules

Checked locations for repository-specific agent rules:

- `.cursorrules`
- `.cursor/rules/`
- `.github/copilot-instructions.md`

Current status: none of these files/directories exist in this repository.

If they are added later, treat them as higher-priority local instructions and update this file.

## Setup and Build Commands

Use these commands from repository root.

### Build / Check

- Debug build: `cargo build`
- Release build: `cargo build --release`
- Fast validation: `cargo check --all-targets --all-features`

### Formatting

- Format code: `cargo fmt --all`
- Check formatting only: `cargo fmt --all -- --check`

### Linting

- Lint all targets/features: `cargo clippy --all-targets --all-features`
- Strict lint (when requested): `cargo clippy --all-targets --all-features -- -D warnings`

### Tests

- All tests (recommended before merge): `cargo test --all-targets --all-features`
- Unit tests only: `cargo test --lib`
- Integration tests only: `cargo test --tests`

## Single-Test Commands (Important)

Prefer these patterns when validating a targeted change.

- Run one exact test by function name:
  - `cargo test <test_name> -- --exact --nocapture`
- Run one integration test file:
  - `cargo test --test data_loading_tests`
- Run one exact test inside that integration file:
  - `cargo test --test data_loading_tests <test_name> -- --exact --nocapture`
- If exact name is unknown, run substring search first:
  - `cargo test <substring>`
  - then rerun with `-- --exact --nocapture`

## Examples and Benchmark Commands

### Examples

- CSV quick demo: `cargo run --example load_csv`
- CSV compress/decompress:
  - `cargo run --example compress_decompress_example -- data/data-10000-8-int.csv`
- Image compress/decompress:
  - `cargo run --example compress_decompress_image_example -- data/images/rustacean.png`

### Experiment Runner (feature-gated)

- Default config run:
  - `cargo run --release --example run_folder_experiments --features experiment-runner -- --config configs/experiment_profiles.json --output target/experiment-dashboard.csv`
- Run on folder recursively:
  - `cargo run --release --example run_folder_experiments --features experiment-runner -- --path data/images --recursive --output target/experiment-dashboard.csv`

### Benchmarks

- Main bench target: `cargo bench --bench benchmark`
- Individual implementation bench target: `cargo bench --bench benchmark_individual`
- Filter criterion bench by name substring:
  - `cargo bench --bench benchmark -- CompressionCore`

## Coding Conventions

Follow existing code style in this repository first.

### Imports

- Prefer explicit imports over wildcard imports.
- Typical order:
  1. `std` imports
  2. external crate imports
  3. `crate::...` imports
- Keep grouped imports readable (`use std::path::{Path, PathBuf};`).
- Use `bitvec::prelude::*` only in bit-heavy modules where already idiomatic.

### Formatting and Structure

- Let `rustfmt` decide line wrapping and spacing.
- Keep modules focused; split very large logic into helper functions.
- Use early-return guards for invalid preconditions.
- Keep pipelines explicit (`load -> preprocess -> entropy -> select -> encode`).

### Types and API Design

- Prefer strong enums/structs for domain states over primitive flags.
- Use `Result<T, EntroGdError>` for fallible library operations.
- Use `Option<T>` for truly optional data, not magic values.
- Use `AsRef<Path>` for file-path APIs that accept multiple path-like inputs.
- Derive `Copy` only for small value-like types where it improves ergonomics.

### Naming

- Types/traits/enums: `UpperCamelCase`
- Functions/modules/files/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test names: start with `test_` and describe behavior clearly.
- Builder-style APIs should use `with_*` naming.

### Error Handling

- Central error type is `EntroGdError` (`src/error.rs`).
- Add specific variants for new error classes instead of overloading existing variants.
- Error messages should include actionable context (path, row/column, expected vs actual).
- Prefer `?` propagation and `From` impls over manual conversion boilerplate.
- Avoid `panic!` in library/runtime paths for recoverable failures.
- `unwrap()`/`expect()` are acceptable in tests/benches/examples when intentional.

### Logging and Timing

- Prefer `tracing` macros instead of `println!` for runtime diagnostics.
- Use `ScopedTimer` for scoped performance instrumentation where useful.
- Respect env-based controls:
  - `LOG` / `RUST_LOG` for log filtering
  - `ENTRO_GD_TIMING` to toggle timers
  - `ENTRO_GD_LOG_DIR` for file log location

### Feature Flags

- Keep `experiment-runner` code behind `#[cfg(feature = "experiment-runner")]`.
- If editing feature-gated paths, validate with `--all-features`.

### Tests

- Put module-specific tests in `#[cfg(test)]` blocks near implementation.
- Put cross-module behavior tests in `tests/` integration files.
- Add regression tests when fixing bugs.
- For format/serialization logic, include roundtrip and malformed-input coverage.

### Performance and Memory

- Be deliberate with cloning large bit-vectors or datasets in hot paths.
- Prefer borrowing and iterators where practical.
- Keep benchmark code reproducible and isolate setup from timed sections.

## Agent Change Checklist

Before handing off changes, do the smallest relevant verification set:

1. `cargo fmt --all`
2. `cargo clippy --all-targets --all-features`
3. Targeted tests for touched code (at least one single-test command when applicable)
4. `cargo test --all-targets --all-features` for broad-impact changes

If any step is skipped, explain why.

## Artifact and Repo Hygiene

- Do not commit generated artifacts under ignored directories (`target/`, `logs/`, generated compressed outputs in `data/**`).
- Keep temporary benchmark/experiment output in `target/` or ignored paths.
- Avoid unrelated refactors in the same change unless requested.
