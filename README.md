# entro_gd

## Experiment runner quick start (copy/paste)

### 1) Run all experiments from default config (recommended)

```bash
cargo run --release --example run_folder_experiments --features experiment-runner -- \
  --output target/experiment-dashboard.csv
```

Default config path: `configs/experiment_profiles.json`

### 2) Run on a specific folder directly

```bash
cargo run --release --example run_folder_experiments --features experiment-runner -- \
  --path data/images \
  --recursive \
  --output target/experiment-dashboard.csv
```

### 3) Run on a single file

```bash
cargo run --release --example run_folder_experiments --features experiment-runner -- \
  --path data/images/rustacean.png \
  --output target/experiment-dashboard-single.csv
```

### 4) View results in terminal

```bash
python3 scripts/view_experiment_dashboard.py --input target/experiment-dashboard.csv --top 5
```

### 5) View results with plot (if matplotlib installed)

```bash
python3 scripts/view_experiment_dashboard.py --input target/experiment-dashboard.csv --plot
```

---

## Config files

- JSON example: `configs/experiment_profiles.json`

Config supports:
- explicit profile lists (`csv_profiles`, `image_profiles`)
- grouped sweeps (`csv_profile_groups`, `image_profile_groups`)
- integer values or ranges with stride (`values` or `range: { start, end, step }`)

### JSON config reference

Top-level keys:
- `input_path`: string (file/folder)
- `recursive`: boolean
- `csv_profiles`: array of explicit CSV profiles
- `image_profiles`: array of explicit image profiles
- `csv_profile_groups`: array of CSV sweep groups (Cartesian expansion)
- `image_profile_groups`: array of image sweep groups (Cartesian expansion)

CSV explicit profile fields:
- `name`: string
- `has_headers`: boolean
- `float_storage`: `f32` | `f64`
- `missing_value_policy`: `error` | `zero`
- `preprocess`:
  - `float_scaling`: `disabled` | `scaled_signed_int` | `scaled_offset_signed_int`
  - `max_decimal_scale`: integer (typically `0..9`)
  - `integer_zero_normalization`: boolean
- `m_max`: integer
- `patience`: integer
- `select_impl`: `naive` | `optimized_v1` | `optimized_v2` | `optimized_v3`
- `encode_impl`: `naive` | `optimized` | `rle` | `huffman` (Huffman is rejected for CSV)

Image explicit profile fields:
- `name`: string
- `build`:
  - `colorspace`: `srgb_with_linear_alpha` | `linear`
  - `color_model`: `rgb` | `y_co_cg` | `y_co_cg_r`
  - `pixel_grouping`: integer `> 0`
  - `grouping_transform`: `raw` | `for_first_pixel` | `for_min`
- `m_max`: integer
- `patience`: integer
- `select_impl`: `naive` | `optimized_v1` | `optimized_v2` | `optimized_v3`
- `encode_impl`: `naive` | `optimized` | `rle` | `huffman`

Sweep syntax:
- Numeric sweep: `{"values": [1, 2, 4]}` or `{"range": {"start": 0, "end": 60, "step": 30}}`
- Enum/bool sweep: use arrays, e.g. `"encode_impl": ["optimized", "rle"]`

---

## One-time shortcut (optional)

If the command is too long, add this to your shell config (`~/.zshrc`):

```bash
alias cargo-exp='cargo run --release --example run_folder_experiments --features experiment-runner --'
```

Then run:

```bash
cargo-exp --config configs/experiment_profiles.json --output target/experiment-dashboard.csv
```

---

## Minimal workflow for thesis experiments

```bash
# 1) run
cargo run --release --example run_folder_experiments --features experiment-runner -- --config configs/experiment_profiles.json --output target/experiment-dashboard.csv
# or with shortcut alias
cargo-exp --config configs/experiment_profiles.json --output target/experiment-dashboard.csv

# 2) inspect by file
python3 scripts/view_experiment_dashboard.py --input target/experiment-dashboard.csv --top 5
```
