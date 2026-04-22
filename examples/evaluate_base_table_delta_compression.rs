use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use entro_gd::data_loader::{CsvDataLoader, DataLoader, FloatStorage};
use entro_gd::prelude::*;
use entro_gd::{
    BaseBitImpl, BitDataSet, BuildBaseTable, BuildBitDataSet, BuildImageBitDataSet,
    BuildSortedBaseTable, EntroGdError, InferFeatureSpecs, PreprocessOptions, SelectBasesOptimized,
};

fn main() -> Result<(), EntroGdError> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!(
            "Usage: {} <input_csv_or_image> [--unsorted]\n  default: uses sorted base table",
            args[0]
        );
        return Ok(());
    }

    let input_path = PathBuf::from(&args[1]);
    let use_sorted = !args.iter().any(|arg| arg == "--unsorted");

    let bit_data = load_bit_data(&input_path)?;

    let context = if use_sorted {
        EntropyBatched {}
            .then(SelectBasesOptimized {
                patience: 20,
                base_bit_impl: BaseBitImpl::BatchGroups,
            })
            .then(BuildSortedBaseTable {})
            .process(bit_data)?
    } else {
        EntropyBatched {}
            .then(SelectBasesOptimized {
                patience: 20,
                base_bit_impl: BaseBitImpl::BatchGroups,
            })
            .then(BuildBaseTable {})
            .process(bit_data)?
    };

    let rows: Vec<&entro_gd::BitView> = context
        .variable_base_table
        .iter()
        .map(|(bits, _)| bits.as_bitslice())
        .collect();

    if rows.is_empty() {
        println!("No variable base rows found.");
        return Ok(());
    }

    let row_width = rows[0].len();
    let entry_count = rows.len();

    let fixed_raw_bits = entry_count * row_width;

    let sort_key_order = build_sort_key_local_order(&context);
    let sort_key_bits: Vec<entro_gd::BitStream> = rows
        .iter()
        .map(|row| build_sort_key_bits(row, &sort_key_order))
        .collect();
    let sort_key_views: Vec<&entro_gd::BitView> = sort_key_bits
        .iter()
        .map(|bits| bits.as_bitslice())
        .collect();

    let absolute_per_row_bits: usize = sort_key_views
        .iter()
        .map(|key| bits_needed_unsigned(key))
        .sum();

    let (delta_bits_signed_prev, delta_bit_length_distribution) =
        estimate_delta_bits_signed_prev_with_distribution(&sort_key_views);

    println!("=== Base-table delta lower-bound quick evaluation ===");
    println!("Input: {}", input_path.display());
    println!("Rows in variable base table: {}", entry_count);
    println!("Bits per row (fixed width): {}", row_width);
    println!(
        "Table order: {}",
        if use_sorted { "sorted" } else { "unsorted" }
    );
    println!(
        "Sort-key columns used: {} (entropy order propagated: {})",
        sort_key_order.len(),
        context.entropy_sorted_column_order.is_some()
    );
    println!();
    println!("Baseline raw fixed-width storage:");
    println!(
        "  {} bits ({:.2} bytes)",
        fixed_raw_bits,
        fixed_raw_bits as f64 / 8.0
    );
    println!();
    println!("Theoretical lower bound (no coding overhead):");
    println!(
        "  Absolute values, per-row minimal width: {} bits ({:.2} bytes)",
        absolute_per_row_bits,
        absolute_per_row_bits as f64 / 8.0
    );
    println!(
        "  Delta-to-previous row, signed minimal width: {} bits ({:.2} bytes)",
        delta_bits_signed_prev,
        delta_bits_signed_prev as f64 / 8.0
    );
    println!();
    println!("Ratios vs fixed-width baseline:");
    println!(
        "  Absolute minimal / fixed: {:.4}",
        absolute_per_row_bits as f64 / fixed_raw_bits as f64
    );
    println!(
        "  Delta minimal / fixed:    {:.4}",
        delta_bits_signed_prev as f64 / fixed_raw_bits as f64
    );
    println!();
    println!("Delta signed-bit-length distribution (per delta to previous row):");
    if delta_bit_length_distribution.is_empty() {
        println!("  (no deltas)");
    } else {
        let total_delta_bits_only: usize = delta_bit_length_distribution
            .iter()
            .map(|(bit_len, count)| bit_len * count)
            .sum();
        let total_deltas: usize = delta_bit_length_distribution.values().sum();
        let mut cumulative_bits = 0usize;
        let mut cumulative_count = 0usize;

        println!(
            "  {:>8} | {:>7} | {:>10} | {:>10} | {:>10}",
            "bit_len", "count", "cum_count", "cum<=len%", "cum_bits%"
        );
        println!("  ----------------------------------------------");
        for (bit_len, count) in &delta_bit_length_distribution {
            cumulative_count += count;
            cumulative_bits += bit_len * count;

            let cumulative_count_pct = if total_deltas == 0 {
                0.0
            } else {
                100.0 * cumulative_count as f64 / total_deltas as f64
            };
            let cumulative_bits_pct = if total_delta_bits_only == 0 {
                0.0
            } else {
                100.0 * cumulative_bits as f64 / total_delta_bits_only as f64
            };

            println!(
                "  {:>6}b | {:>7} | {:>10} | {:>9.2}% | {:>9.2}%",
                bit_len, count, cumulative_count, cumulative_count_pct, cumulative_bits_pct
            );
        }
    }

    Ok(())
}

fn load_bit_data(input_path: &Path) -> Result<BitDataSet, EntroGdError> {
    if is_image_path(input_path) {
        BuildImageBitDataSet {
            colorspace: ImageColorSpace::SrgbWithLinearAlpha,
            color_model: ImageColorModel::YCoCgR,
            pixel_grouping: 4,
            grouping_transform: ImageGroupingTransform::ForFirstPixel,
            pad_rows_to_word: false,
        }
        .process(input_path.to_path_buf())
    } else {
        let loader = CsvDataLoader::new(true).with_float_storage(FloatStorage::F32);
        let loaded = loader.load(input_path)?;

        InferFeatureSpecs {
            options: PreprocessOptions::default(),
        }
        .then(BuildBitDataSet::default())
        .process(loaded.dataset)
    }
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "gif"
            )
        })
        .unwrap_or(false)
}

fn estimate_delta_bits_signed_prev_with_distribution(
    rows: &[&entro_gd::BitView],
) -> (usize, BTreeMap<usize, usize>) {
    if rows.is_empty() {
        return (0, BTreeMap::new());
    }

    let mut distribution: BTreeMap<usize, usize> = BTreeMap::new();

    // First value stored as an absolute value with minimal unsigned width.
    let mut total_bits = bits_needed_unsigned(rows[0]);

    for pair in rows.windows(2) {
        let prev = pair[0];
        let current = pair[1];

        let magnitude = abs_diff_unsigned(current, prev);
        let magnitude_bits = bits_needed_unsigned(magnitude.as_bitslice());

        // Lower-bound signed representation: 0 bits for zero delta, else sign + magnitude bits.
        let signed_bits = if magnitude_bits == 0 {
            0
        } else {
            1 + magnitude_bits
        };

        *distribution.entry(signed_bits).or_insert(0) += 1;
        total_bits += signed_bits;
    }

    (total_bits, distribution)
}

fn build_sort_key_local_order(context: &entro_gd::PreEncodeContext) -> Vec<usize> {
    let row_width = context
        .variable_base_table
        .first()
        .map(|(bits, _)| bits.len())
        .unwrap_or(0);

    if let Some(order) = &context.entropy_sorted_column_order {
        let filtered: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&idx| idx < row_width)
            .collect();
        if !filtered.is_empty() {
            return filtered;
        }
    }

    (0..row_width).collect()
}

fn build_sort_key_bits(row: &entro_gd::BitView, order: &[usize]) -> entro_gd::BitStream {
    // `compare_unsigned()` treats higher index as more significant.
    // We therefore append order in reverse so order[0] becomes the most-significant key bit.
    let mut out = entro_gd::BitStream::with_capacity(order.len());
    for &col_idx in order.iter().rev() {
        out.push(row.get(col_idx).map(|b| *b).unwrap_or(false));
    }
    out
}

fn bits_needed_unsigned(bits: &entro_gd::BitView) -> usize {
    for idx in (0..bits.len()).rev() {
        if bits.get(idx).map(|b| *b).unwrap_or(false) {
            return idx + 1;
        }
    }
    0
}

fn abs_diff_unsigned(lhs: &entro_gd::BitView, rhs: &entro_gd::BitView) -> entro_gd::BitStream {
    match compare_unsigned(lhs, rhs) {
        Ordering::Greater | Ordering::Equal => subtract_unsigned(lhs, rhs),
        Ordering::Less => subtract_unsigned(rhs, lhs),
    }
}

fn compare_unsigned(lhs: &entro_gd::BitView, rhs: &entro_gd::BitView) -> Ordering {
    let max_len = lhs.len().max(rhs.len());
    for idx in (0..max_len).rev() {
        let l = lhs.get(idx).map(|b| *b).unwrap_or(false);
        let r = rhs.get(idx).map(|b| *b).unwrap_or(false);
        match l.cmp(&r) {
            Ordering::Equal => continue,
            non_equal => return non_equal,
        }
    }
    Ordering::Equal
}

fn subtract_unsigned(
    minuend: &entro_gd::BitView,
    subtrahend: &entro_gd::BitView,
) -> entro_gd::BitStream {
    let max_len = minuend.len().max(subtrahend.len());
    let mut out = entro_gd::BitStream::with_capacity(max_len);

    let mut borrow: i8 = 0;
    for idx in 0..max_len {
        let a = if minuend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };
        let b = if subtrahend.get(idx).map(|b| *b).unwrap_or(false) {
            1
        } else {
            0
        };

        let mut diff = a - b - borrow;
        if diff < 0 {
            diff += 2;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(diff == 1);
    }

    while out.last().map(|bit| *bit) == Some(false) {
        out.pop();
    }

    out
}
