use bitvec::field::BitField;
use bitvec::prelude::*;
use polars::prelude::*;
use std::fmt::{self, Display};
use std::path::{Path, PathBuf};
use tracing::{debug, info, trace};

use crate::error::EntroGdError;
use crate::filter_pipeline::Filter;
use crate::timing::ScopedTimer;

const MAX_DECIMAL_SCALE: u8 = 9;
pub const DEFAULT_ALIGN_ROWS_TO_WORD: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelGrouping(pub u32, pub u32);

impl PixelGrouping {
    pub const fn new(width: u32, height: u32) -> Self {
        Self(width, height)
    }

    pub const fn width(self) -> u32 {
        self.0
    }

    pub const fn height(self) -> u32 {
        self.1
    }

    pub const fn total_pixels(self) -> u32 {
        self.0 * self.1
    }
}

impl Default for PixelGrouping {
    fn default() -> Self {
        Self(1, 1)
    }
}

impl Display for PixelGrouping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.0, self.1)
    }
}

// ---------------------------------------------------------------------------
// Tabular ingestion
// ---------------------------------------------------------------------------

/// Where a tabular [`DataFrame`] comes from. Every variant just needs to be
/// able to produce a `DataFrame`; everything downstream (transforms, bit
/// packing) only ever operates on that `DataFrame`, so adding a new source
/// format (Parquet, JSON, Arrow IPC, ...) later is purely additive.
#[derive(Debug, Clone)]
pub enum TabularSource {
    Csv { path: PathBuf },
}

impl TabularSource {
    pub fn csv(path: impl Into<PathBuf>) -> Self {
        TabularSource::Csv { path: path.into() }
    }

    pub fn load(&self) -> Result<DataFrame, EntroGdError> {
        match self {
            TabularSource::Csv { path } => load_csv(path),
        }
    }
}

/// Load a CSV file into a Polars [`DataFrame`] using Polars' own reader
/// rather than a hand-rolled parser, so column typing, quoting, and encoding
/// edge cases are handled for us.
pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<DataFrame, EntroGdError> {
    let path = path.as_ref();
    trace!("Loading CSV dataset from {}", path.display());

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))
        .map_err(|e| EntroGdError::InvalidDataType {
            message: format!("failed to open CSV '{}': {}", path.display(), e),
        })?
        .finish()
        .map_err(|e| EntroGdError::InvalidDataType {
            message: format!("failed to parse CSV '{}': {}", path.display(), e),
        })?;

    info!(
        "Loaded CSV '{}': {} row(s) x {} column(s)",
        path.display(),
        df.height(),
        df.width()
    );

    Ok(df)
}

// ---------------------------------------------------------------------------
// Preprocessing options
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatScalingMode {
    /// Store the raw IEEE-754 bits of the original float, unmodified.
    Disabled,
    /// Scale to an integer, then bias by a fixed, data-independent offset
    /// (derived from the integer type's range) instead of scanning the
    /// column for its true minimum. Faster, but does not shrink the bit
    /// width beyond the scaled type's natural size.
    ScaledSignedInt,
    /// Scale to an integer, then zero-normalize using the column's actual
    /// minimum so the value range fits the smallest possible unsigned
    /// integer. This is the default: best compression, small extra scan.
    ScaledOffsetSignedInt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessOptions {
    /// How (or whether) floating point columns are converted to integers.
    pub float_scaling: FloatScalingMode,
    /// Number of decimal places to preserve when scaling floats to
    /// integers, e.g. `decimal_scale = 2` multiplies by 100 before
    /// rounding. This is a fixed, user-supplied precision rather than
    /// something searched for per-column, which keeps preprocessing fast.
    pub decimal_scale: u8,
    /// Whether integer columns (signed or unsigned) are zero-normalized
    /// against their actual observed minimum. When disabled, signed
    /// columns still have to become unsigned to be bit-packed, but do so
    /// via a fixed type-range bias instead of a data scan.
    pub integer_zero_normalization: bool,
}

impl Default for PreprocessOptions {
    fn default() -> Self {
        PreprocessOptions {
            float_scaling: FloatScalingMode::ScaledOffsetSignedInt,
            decimal_scale: MAX_DECIMAL_SCALE,
            integer_zero_normalization: true,
        }
    }
}

/// Filter for building a [`BitDataSet`] from a [`DataFrame`].
#[derive(Debug, Clone, Copy)]
pub struct BuildBitDataSet {
    pub options: PreprocessOptions,
    pub pad_rows_to_word: bool,
}

impl Default for BuildBitDataSet {
    fn default() -> Self {
        Self {
            options: PreprocessOptions::default(),
            pad_rows_to_word: DEFAULT_ALIGN_ROWS_TO_WORD,
        }
    }
}

impl Filter for BuildBitDataSet {
    type Input = DataFrame;
    type Output = BitDataSet;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let _timer = ScopedTimer::info("Building BitDataSet");
        BitDataSet::from_dataframe(input, self.options, self.pad_rows_to_word)
    }
}

pub(crate) fn aligned_stride(chunk_size: usize, pad_rows_to_word: bool) -> usize {
    if pad_rows_to_word {
        chunk_size.next_multiple_of(usize::BITS as usize)
    } else {
        chunk_size
    }
}

// ---------------------------------------------------------------------------
// Feature schema / metadata
// ---------------------------------------------------------------------------

/// Transform metadata used to recover the original value of a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureTransform {
    /// Stored verbatim (or, for floats, as raw IEEE-754 bits).
    None,
    /// Float scaled by `10^decimal_scale`, then biased by a fixed,
    /// data-independent offset (see [`FloatScalingMode::ScaledSignedInt`]).
    ScaledSignedInt { decimal_scale: u8 },
    /// Signed integer, zero-normalized by subtracting `min_value`
    /// (`min_value` may itself be negative).
    OffsetSignedInt { min_value: i64 },
    /// Unsigned integer, zero-normalized by subtracting `min_value`.
    OffsetUnsignedInt { min_value: u64 },
    /// Float scaled by `10^decimal_scale`, then zero-normalized against the
    /// column's actual (scaled) minimum.
    ScaledOffsetSignedInt { decimal_scale: u8, min_value: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDataType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt(u16),
}

impl TryFrom<DataType> for FeatureDataType {
    type Error = EntroGdError;

    fn try_from(data_type: DataType) -> Result<Self, Self::Error> {
        match data_type {
            DataType::Float16 => Ok(FeatureDataType::Float16),
            DataType::Float32 => Ok(FeatureDataType::Float32),
            DataType::Float64 => Ok(FeatureDataType::Float64),
            DataType::Int8 => Ok(FeatureDataType::Int8),
            DataType::Int16 => Ok(FeatureDataType::Int16),
            DataType::Int32 => Ok(FeatureDataType::Int32),
            DataType::Int64 => Ok(FeatureDataType::Int64),
            DataType::UInt8 => Ok(FeatureDataType::UInt8),
            DataType::UInt16 => Ok(FeatureDataType::UInt16),
            DataType::UInt32 => Ok(FeatureDataType::UInt32),
            DataType::UInt64 => Ok(FeatureDataType::UInt64),
            other => Err(EntroGdError::InvalidDataType {
                message: format!("data type {:?} is not yet supported", other),
            }),
        }
    }
}

impl FeatureDataType {
    pub fn bits(&self) -> usize {
        match self {
            FeatureDataType::Float16 => 16,
            FeatureDataType::Float32 => 32,
            FeatureDataType::Float64 => 64,
            FeatureDataType::Int8 => 8,
            FeatureDataType::Int16 => 16,
            FeatureDataType::Int32 => 32,
            FeatureDataType::Int64 => 64,
            FeatureDataType::UInt8 => 8,
            FeatureDataType::UInt16 => 16,
            FeatureDataType::UInt32 => 32,
            FeatureDataType::UInt64 => 64,
            FeatureDataType::UInt(bits) => *bits as usize,
        }
    }

    /// Smallest unsigned integer type that can hold every value in
    /// `0..=max_value`.
    fn smallest_unsigned_for(max_value: u64) -> FeatureDataType {
        if max_value <= u8::MAX as u64 {
            FeatureDataType::UInt8
        } else if max_value <= u16::MAX as u64 {
            FeatureDataType::UInt16
        } else if max_value <= u32::MAX as u64 {
            FeatureDataType::UInt32
        } else {
            let bits = usize::BITS as usize - max_value.leading_zeros() as usize;
            FeatureDataType::UInt(bits.max(1) as u16)
        }
    }
}

/// Per-feature schema entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    pub data_type: FeatureDataType,
    pub transform: FeatureTransform,
}

impl FeatureSpec {
    pub fn new(data_type: FeatureDataType) -> Self {
        FeatureSpec {
            data_type,
            transform: FeatureTransform::None,
        }
    }
}

/// High-level metadata describing a bit-packed dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitDataCompressionInfo {
    pub features: Vec<FeatureSpec>,
    pub original_size_bits: usize,
    pub n_data_samples: usize,
    pub m_condensed_samples: Option<usize>,
    pub m_condensed_sample_weights: Option<Vec<usize>>,
    feature_offsets: Vec<usize>,
    chunk_size: usize,
    row_stride_bits: usize,
}

/// High-level metadata describing a bit-packed dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitDataInfo {
    pub compression: BitDataCompressionInfo,
    pub reconstruction: BitDataReconstructionInfo,
}

impl BitDataInfo {
    pub fn new(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
    ) -> Result<Self, EntroGdError> {
        Self::new_with_reconstruction_info(
            features,
            original_size_bits,
            BitDataReconstructionInfo::Tabular,
        )
    }

    pub fn new_with_reconstruction_info(
        features: Vec<FeatureSpec>,
        original_size_bits: usize,
        reconstruction: BitDataReconstructionInfo,
    ) -> Result<Self, EntroGdError> {
        debug!(
            "Creating BitDataInfo with {} features and original_size_bits={}.",
            features.len(),
            original_size_bits
        );
        if features.is_empty() {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "features is empty".to_string(),
            });
        }

        let mut offsets = Vec::with_capacity(features.len());
        let mut running = 0usize;
        for spec in features.iter() {
            offsets.push(running);
            running += spec.data_type.bits();
        }

        let compression = BitDataCompressionInfo {
            features,
            original_size_bits,
            n_data_samples: original_size_bits / running,
            m_condensed_samples: None,
            m_condensed_sample_weights: None,
            feature_offsets: offsets,
            chunk_size: running,
            row_stride_bits: running,
        };

        let info = BitDataInfo {
            compression,
            reconstruction,
        };

        debug!(
            "BitDataInfo ready: chunk_size={} bits, n_data_samples={}",
            info.chunk_size(),
            info.n_data_samples()
        );

        Ok(info)
    }

    pub fn features(&self) -> &[FeatureSpec] {
        &self.compression.features
    }

    pub fn feature_spec(&self, feature_idx: usize) -> &FeatureSpec {
        &self.compression.features[feature_idx]
    }

    pub fn original_size_bits(&self) -> usize {
        self.compression.original_size_bits
    }

    pub fn n_data_samples(&self) -> usize {
        self.compression.n_data_samples
    }

    pub fn m_condensed_samples(&self) -> Option<usize> {
        self.compression.m_condensed_samples
    }

    pub fn m_condensed_sample_weights(&self) -> Option<&[usize]> {
        self.compression.m_condensed_sample_weights.as_deref()
    }

    pub fn set_condensed_sample_weights(&mut self, weights: Option<Vec<usize>>) {
        self.compression.m_condensed_samples = weights.as_ref().map(Vec::len);
        self.compression.m_condensed_sample_weights = weights;
    }

    pub fn num_features(&self) -> usize {
        self.compression.features.len()
    }

    pub fn chunk_size(&self) -> usize {
        self.compression.chunk_size
    }

    pub fn row_stride(&self) -> usize {
        self.compression.row_stride_bits
    }

    pub fn is_row_padded_to_word(&self) -> bool {
        self.compression.row_stride_bits > self.compression.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.compression.features[feature_idx].data_type.bits()
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.compression.feature_offsets[feature_idx]
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        if bit_pos >= self.compression.chunk_size {
            return None;
        }
        match self.compression.feature_offsets.binary_search(&bit_pos) {
            Ok(idx) => Some(idx),
            Err(0) => None,
            Err(idx) => Some(idx - 1),
        }
    }

    pub fn with_original_size_bits(&self, original_size_bits: usize) -> Self {
        self.with_original_size_bits_and_row_stride(original_size_bits, self.row_stride())
    }

    pub fn with_original_size_bits_and_row_stride(
        &self,
        original_size_bits: usize,
        row_stride_bits: usize,
    ) -> Self {
        let chunk_size = self.compression.chunk_size;
        let row_stride_bits = row_stride_bits.max(chunk_size);
        BitDataInfo {
            compression: BitDataCompressionInfo {
                features: self.compression.features.clone(),
                original_size_bits,
                n_data_samples: original_size_bits / chunk_size,
                m_condensed_samples: self.compression.m_condensed_samples,
                m_condensed_sample_weights: self.compression.m_condensed_sample_weights.clone(),
                feature_offsets: self.compression.feature_offsets.clone(),
                chunk_size,
                row_stride_bits,
            },
            reconstruction: self.reconstruction.clone(),
        }
    }
}

/// Represents the bit-level preprocessed data where each row is a contiguous chunk of bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitData {
    /// The underlying bit storage - all chunks stored contiguously
    pub data: BitVec<usize, Lsb0>,
    /// Logical bits per chunk (what algorithms see)
    pub chunk_size: usize,
    /// Physical bits per chunk including word-alignment padding
    pub stride: usize,
    /// Number of rows/chunks (number of records)
    pub num_rows: usize,
}

impl BitData {
    /// Extend the BitData with additional bits from a BitSlice (used for adding condensed samples)
    pub fn extend_from_bitslice(
        &mut self,
        bits: &BitSlice<usize, Lsb0>,
    ) -> Result<(), EntroGdError> {
        if bits.len() != self.chunk_size {
            return Err(EntroGdError::BitSliceLengthMismatch {
                expected: self.chunk_size,
                actual: bits.len(),
            });
        }
        self.data.extend(bits);
        append_row_padding(&mut self.data, self.stride.saturating_sub(self.chunk_size));
        self.num_rows += 1; // Treat the new bits as an additional row/chunk
        Ok(())
    }

    /// Get a slice of bits for a specific row/chunk
    #[inline(always)]
    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Lsb0> {
        let start = row * self.stride;
        let end = start + self.chunk_size;
        &self.data[start..end]
    }

    /// Get a specific bit by linear bit index without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `bit_idx < self.data.len()`.
    #[inline(always)]
    pub(crate) unsafe fn get_bit_linear_unchecked(&self, bit_idx: usize) -> bool {
        debug_assert!(bit_idx < self.data.len());
        unsafe { *self.data.get_unchecked(bit_idx) }
    }

    /// Get a specific bit by row and bit position within the chunk
    #[inline(always)]
    pub fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        let idx = row * self.stride + bit_in_chunk;
        self.data[idx]
    }

    /// Get a specific bit by row and bit position within the chunk without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.num_rows` and `bit_in_chunk < self.chunk_size`.
    #[inline(always)]
    pub(crate) unsafe fn get_bit_unchecked(&self, row: usize, bit_in_chunk: usize) -> bool {
        debug_assert!(row < self.num_rows);
        debug_assert!(bit_in_chunk < self.chunk_size);
        let idx = row * self.stride + bit_in_chunk;
        unsafe { self.get_bit_linear_unchecked(idx) }
    }

    /// Get raw access to the underlying bit vector
    pub fn raw(&self) -> &BitVec<usize, Lsb0> {
        &self.data
    }

    /// Get the total number of bits stored
    pub fn total_bits(&self) -> usize {
        self.chunk_size * self.num_rows
    }
}

pub(crate) fn append_row_padding(data: &mut BitVec<usize, Lsb0>, padding_bits: usize) {
    if padding_bits > 0 {
        data.resize(data.len() + padding_bits, false);
    }
}

/// Combines BitData with its metadata for algorithms that need both.
#[derive(Debug, Clone)]
pub struct BitDataSet {
    pub data: BitData,
    pub info: BitDataInfo,
}

impl BitDataSet {
    /// Build a [`BitDataSet`] directly from a [`TabularSource`] (loads then transforms).
    pub fn from_source(
        source: &TabularSource,
        options: PreprocessOptions,
        pad_rows_to_word: bool,
    ) -> Result<Self, EntroGdError> {
        let dataframe = source.load()?;
        Self::from_dataframe(dataframe, options, pad_rows_to_word)
    }

    /// Create a `BitDataSet` from a `DataFrame`:
    ///   1. Transform every column (zero-normalize integers, scale + zero-normalize
    ///      floats) and pick the smallest unsigned integer type that fits it.
    ///   2. Preallocate the destination bit buffer to its final size
    ///      (`row_stride_bits * num_rows`) up front.
    ///   3. Fill the buffer one column (feature) at a time, which keeps memory
    ///      access to the source data sequential even though writes into the
    ///      row-major bit buffer are strided.
    pub fn from_dataframe(
        dataframe: DataFrame,
        options: PreprocessOptions,
        pad_rows_to_word: bool,
    ) -> Result<Self, EntroGdError> {
        let num_rows = dataframe.height();
        if num_rows == 0 {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "cannot build a BitDataSet from an empty dataframe".to_string(),
            });
        }

        let columns = dataframe.columns();
        if columns.is_empty() {
            return Err(EntroGdError::InvalidFeatureSpec {
                message: "dataframe has no columns".to_string(),
            });
        }

        info!(
            "Preprocessing {} column(s) x {} row(s) into a BitDataSet",
            columns.len(),
            num_rows
        );

        let transformed: Vec<TransformedColumn> = columns
            .iter()
            .map(|col| {
                let series = col.as_materialized_series();
                transform_column(&series, &options)
            })
            .collect::<Result<_, _>>()?;

        let features: Vec<FeatureSpec> = transformed.iter().map(|t| t.spec.clone()).collect();
        let chunk_size: usize = features.iter().map(|f| f.data_type.bits()).sum();
        let stride = aligned_stride(chunk_size, pad_rows_to_word);
        let original_size_bits = num_rows * chunk_size;

        let mut info = BitDataInfo::new(features, original_size_bits)?;
        if stride != chunk_size {
            info = info.with_original_size_bits_and_row_stride(original_size_bits, stride);
        }

        debug!(
            "BitDataSet layout: chunk_size={} bits, stride={} bits, num_rows={}",
            chunk_size, stride, num_rows
        );

        // Preallocate the whole buffer up front rather than growing it row by row.
        let mut bits: BitVec<usize, Lsb0> = BitVec::repeat(false, stride * num_rows);

        // Fill column-by-column: for each feature, walk its already-transformed
        // UInt64 Series and store each value into its row slot.
        for (feature_idx, column) in transformed.iter().enumerate() {
            let bit_width = info.feature_bits(feature_idx);
            let base_offset = info.feature_offset(feature_idx);
            let ca = column.series.u64().map_err(|e| EntroGdError::InvalidDataType {
                message: format!(
                    "internal error: transformed column '{}' is not UInt64: {}",
                    column.series.name(),
                    e
                ),
            })?;
            for (row, value) in ca.into_no_null_iter().enumerate() {
                let start = row * stride + base_offset;
                let end = start + bit_width;
                bits[start..end].store_le::<u64>(value);
            }
        }

        let data = BitData {
            data: bits,
            chunk_size,
            stride,
            num_rows,
        };

        Ok(BitDataSet { data, info })
    }

    /// Get a slice of bits for a specific feature within a row
    #[inline(always)]
    pub fn get_feature(&self, row: usize, feature: usize) -> &BitSlice<usize, Lsb0> {
        let chunk_start = row * self.data.stride;
        let feat_start = chunk_start + self.info.feature_offset(feature);
        let feat_end = feat_start + self.info.feature_bits(feature);
        &self.data.data[feat_start..feat_end]
    }

    /// Get a specific bit by row, feature, and bit index within the feature
    #[inline(always)]
    pub fn get_bit_by_feature(&self, row: usize, feature: usize, bit: usize) -> bool {
        let idx = row * self.data.stride + self.info.feature_offset(feature) + bit;
        self.data.data[idx]
    }

    /// Get a feature slice without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.data.num_rows` and `feature < self.info.num_features()`.
    #[inline(always)]
    pub(crate) unsafe fn get_feature_unchecked(
        &self,
        row: usize,
        feature: usize,
    ) -> &BitSlice<usize, Lsb0> {
        debug_assert!(row < self.data.num_rows);
        debug_assert!(feature < self.info.num_features());
        let chunk_start = row * self.data.stride;
        let feature_offset = self.info.feature_offset(feature);
        let feature_bits = self.info.feature_bits(feature);
        let feat_start = chunk_start + feature_offset;
        let feat_end = feat_start + feature_bits;
        debug_assert!(feat_end <= self.data.data.len());
        unsafe { self.data.data.get_unchecked(feat_start..feat_end) }
    }

    /// Get a chunk without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.data.num_rows`.
    #[inline(always)]
    pub(crate) unsafe fn get_chunk_unchecked(&self, row: usize) -> &BitSlice<usize, Lsb0> {
        debug_assert!(row < self.data.num_rows);
        let start = row * self.data.stride;
        let end = start + self.data.chunk_size;
        unsafe { self.data.data.get_unchecked(start..end) }
    }

    /// Get a bit without bounds checks.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.data.num_rows` and `bit_in_chunk < self.data.chunk_size`.
    #[inline(always)]
    pub(crate) unsafe fn get_bit_unchecked(&self, row: usize, bit_in_chunk: usize) -> bool {
        debug_assert!(row < self.data.num_rows);
        debug_assert!(bit_in_chunk < self.data.chunk_size);
        unsafe { self.data.get_bit_unchecked(row, bit_in_chunk) }
    }

    pub fn num_rows(&self) -> usize {
        self.data.num_rows
    }

    pub fn num_features(&self) -> usize {
        self.info.num_features()
    }

    pub fn chunk_size(&self) -> usize {
        self.data.chunk_size
    }

    pub fn feature_bits(&self, feature_idx: usize) -> usize {
        self.info.feature_bits(feature_idx)
    }

    pub fn feature_offset(&self, feature_idx: usize) -> usize {
        self.info.feature_offset(feature_idx)
    }

    pub fn feature_index_for_bit(&self, bit_pos: usize) -> Option<usize> {
        self.info.feature_index_for_bit(bit_pos)
    }

    pub fn get_bit(&self, row: usize, bit_in_chunk: usize) -> bool {
        self.data.get_bit(row, bit_in_chunk)
    }

    pub fn get_chunk(&self, row: usize) -> &BitSlice<usize, Lsb0> {
        self.data.get_chunk(row)
    }
}

impl Display for BitDataSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BitData:")?;
        writeln!(f, "  Number of rows: {}", self.data.num_rows)?;
        writeln!(f, "  Number of features: {}", self.info.num_features())?;
        writeln!(f, "  Chunk size (bits per row): {}", self.data.chunk_size)?;
        writeln!(f, "  Total bits: {}", self.data.total_bits())?;
        write!(f, "Bit  : ")?;
        for bit_pos in 0..self.data.chunk_size {
            write!(f, "{} ", bit_pos)?;
        }
        for row in 0..self.data.num_rows {
            writeln!(f)?;
            write!(f, "Row {}: ", row)?;
            for bit_pos in 0..self.data.chunk_size {
                let bit = self.data.get_bit(row, bit_pos);
                write!(f, "{} ", if bit { '1' } else { '0' })?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Column transforms
// ---------------------------------------------------------------------------

/// A column after transformation: its final schema entry, plus the
/// transformed `Series` (UInt64) holding the packable value for every row.
struct TransformedColumn {
    spec: FeatureSpec,
    series: Series,
}

fn apply_transform(s: &Series, transform: &FeatureTransform) -> PolarsResult<Series> {
    match transform {
        FeatureTransform::None => match s.dtype() {
            DataType::Float64 => {
                let ca = s.f64()?;
                let out: UInt64Chunked =
                    ca.apply_nonnull_values_generic(DataType::UInt64, |v| v.to_bits());
                Ok(out.into_series())
            }
            DataType::Float32 => {
                let ca = s.f32()?;
                let out: UInt64Chunked =
                    ca.apply_nonnull_values_generic(DataType::UInt64, |v| v.to_bits() as u64);
                Ok(out.into_series())
            }
            _ => s.cast(&DataType::UInt64),
        },

        FeatureTransform::ScaledSignedInt { decimal_scale } => {
            let factor = 10f64.powi(*decimal_scale as i32);
            let ca = s.f64()?;
            let out: UInt64Chunked =
                ca.apply_nonnull_values_generic(DataType::UInt64, |v| {
                    let scaled = (v * factor).round() as i64;
                    (scaled as u64).wrapping_sub(i64::MIN as u64)
                });
            Ok(out.into_series())
        }

        FeatureTransform::OffsetSignedInt { min_value } => {
            let ca = s.i64()?;
            let out: UInt64Chunked =
                ca.apply_nonnull_values_generic(DataType::UInt64, |v| {
                    (v as u64).wrapping_sub(*min_value as u64)
                });
            Ok(out.into_series())
        }

        FeatureTransform::OffsetUnsignedInt { min_value } => {
            let ca = s.u64()?;
            let out: UInt64Chunked =
                ca.apply_nonnull_values_generic(DataType::UInt64, |v| v - min_value);
            Ok(out.into_series())
        }

        FeatureTransform::ScaledOffsetSignedInt {
            decimal_scale,
            min_value,
        } => {
            let factor = 10f64.powi(*decimal_scale as i32);
            let ca = s.f64()?;
            let out: UInt64Chunked =
                ca.apply_nonnull_values_generic(DataType::UInt64, |v| {
                    let scaled = (v * factor).round() as i64;
                    (scaled as u64).wrapping_sub(*min_value as u64)
                });
            Ok(out.into_series())
        }
    }
}

fn transform_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    if series.null_count() > 0 {
        return Err(EntroGdError::InvalidDataType {
            message: format!(
                "column '{}' contains null values, which are not yet supported",
                series.name()
            ),
        });
    }

    match series.dtype() {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            transform_float_column(series, options)
        }
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            transform_signed_column(series, options)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            transform_unsigned_column(series, options)
        }
        other => Err(EntroGdError::InvalidDataType {
            message: format!(
                "column '{}' has unsupported dtype {:?}",
                series.name(),
                other
            ),
        }),
    }
}

fn cast_err(column: &str, e: PolarsError) -> EntroGdError {
    EntroGdError::InvalidDataType {
        message: format!("failed to cast column '{}': {}", column, e),
    }
}

fn empty_column_err(column: &str) -> EntroGdError {
    EntroGdError::InvalidFeatureSpec {
        message: format!("column '{}' has no rows to derive a range from", column),
    }
}

/// Fixed, data-independent bias for mapping a signed integer type to
/// unsigned without scanning the data (used when zero-normalization is
/// disabled). This is just the type's natural minimum, i.e. the standard
/// "flip the sign bit" trick.
fn fixed_signed_bias(dtype: &DataType) -> i64 {
    match dtype {
        DataType::Int8 => i8::MIN as i64,
        DataType::Int16 => i16::MIN as i64,
        DataType::Int32 => i32::MIN as i64,
        DataType::Int64 => i64::MIN,
        _ => 0,
    }
}

fn transform_signed_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    let name = series.name().to_string();
    let casted = series
        .cast(&DataType::Int64)
        .map_err(|e| cast_err(&name, e))?;
    let ca = casted.i64().map_err(|e| cast_err(&name, e))?;

    let min_value = if options.integer_zero_normalization {
        ca.min().ok_or_else(|| empty_column_err(&name))?
    } else {
        fixed_signed_bias(series.dtype())
    };

    let transform = FeatureTransform::OffsetSignedInt { min_value };
    let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
    let range = out
        .u64()
        .map_err(|e| cast_err(&name, e))?
        .max()
        .unwrap_or(0);

    Ok(TransformedColumn {
        spec: FeatureSpec {
            data_type: FeatureDataType::smallest_unsigned_for(range),
            transform,
        },
        series: out,
    })
}

fn transform_unsigned_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    let name = series.name().to_string();
    let casted = series
        .cast(&DataType::UInt64)
        .map_err(|e| cast_err(&name, e))?;
    let ca = casted.u64().map_err(|e| cast_err(&name, e))?;

    if !options.integer_zero_normalization {
        let max_value = ca.max().ok_or_else(|| empty_column_err(&name))?;
        return Ok(TransformedColumn {
            spec: FeatureSpec {
                data_type: FeatureDataType::smallest_unsigned_for(max_value),
                transform: FeatureTransform::None,
            },
            series: casted,
        });
    }

    let min_value = ca.min().ok_or_else(|| empty_column_err(&name))?;
    let transform = FeatureTransform::OffsetUnsignedInt { min_value };
    let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
    let range = out
        .u64()
        .map_err(|e| cast_err(&name, e))?
        .max()
        .unwrap_or(0);

    Ok(TransformedColumn {
        spec: FeatureSpec {
            data_type: FeatureDataType::smallest_unsigned_for(range),
            transform,
        },
        series: out,
    })
}

fn transform_float_column(
    series: &Series,
    options: &PreprocessOptions,
) -> Result<TransformedColumn, EntroGdError> {
    let name = series.name().to_string();

    if options.float_scaling == FloatScalingMode::Disabled {
        let data_type: FeatureDataType = series.dtype().clone().try_into()?;
        let transform = FeatureTransform::None;
        let out = apply_transform(series, &transform).map_err(|e| cast_err(&name, e))?;
        return Ok(TransformedColumn {
            spec: FeatureSpec {
                data_type,
                transform,
            },
            series: out,
        });
    }

    let casted = series
        .cast(&DataType::Float64)
        .map_err(|e| cast_err(&name, e))?;
    let ca = casted.f64().map_err(|e| cast_err(&name, e))?;

    let decimal_scale = options.decimal_scale.min(MAX_DECIMAL_SCALE);
    let multiplier = 10f64.powi(decimal_scale as i32);

    // Pre-pass: validate non-finite and overflow, and find scaled minimum
    let scaled_min = ca
        .into_no_null_iter()
        .try_fold(i64::MAX, |min_acc, v| {
            if !v.is_finite() {
                return Err(EntroGdError::InvalidDataType {
                    message: format!(
                        "column '{}' contains a non-finite value ({}) that cannot be scaled",
                        name, v
                    ),
                });
            }
            let s = v * multiplier;
            if s < i64::MIN as f64 || s > i64::MAX as f64 {
                return Err(EntroGdError::InvalidDataType {
                    message: format!(
                        "column '{}' overflows i64 once scaled by 10^{}",
                        name, decimal_scale
                    ),
                });
            }
            Ok(min_acc.min(s.round() as i64))
        })?;

    match options.float_scaling {
        FloatScalingMode::Disabled => unreachable!(),
        FloatScalingMode::ScaledSignedInt => {
            let transform = FeatureTransform::ScaledSignedInt { decimal_scale };
            let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
            let range = out
                .u64()
                .map_err(|e| cast_err(&name, e))?
                .max()
                .unwrap_or(0);
            Ok(TransformedColumn {
                spec: FeatureSpec {
                    data_type: FeatureDataType::smallest_unsigned_for(range),
                    transform,
                },
                series: out,
            })
        }
        FloatScalingMode::ScaledOffsetSignedInt => {
            let transform = FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value: scaled_min,
            };
            let out = apply_transform(&casted, &transform).map_err(|e| cast_err(&name, e))?;
            let range = out
                .u64()
                .map_err(|e| cast_err(&name, e))?
                .max()
                .unwrap_or(0);
            Ok(TransformedColumn {
                spec: FeatureSpec {
                    data_type: FeatureDataType::smallest_unsigned_for(range),
                    transform,
                },
                series: out,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Image reconstruction metadata (unrelated to tabular preprocessing, kept as-is)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGroupingTransform {
    /// Store grouped channel bytes directly.
    Raw = 0,
    /// Store the first byte, then zig-zag encoded residuals relative to it.
    ForFirstPixel = 1,
    /// Store the minimum byte, then zig-zag encoded residuals relative to it.
    ForMin = 2,
}

impl ImageGroupingTransform {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Result<Self, EntroGdError> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::ForFirstPixel),
            2 => Ok(Self::ForMin),
            _ => Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported image grouping transform {} (expected 0, 1, or 2)",
                    value
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorModel {
    /// Store channels as RGB(A).
    Rgb = 0,
    /// Store channels as YCoCg(A), where Co/Cg are biased into byte range.
    YCoCg = 1,
    /// Store channels as reversible YCoCg-R(A), where Co/Cg are wrapped to 8-bit and biased.
    YCoCgR = 2,
}

impl ImageColorModel {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Result<Self, EntroGdError> {
        match value {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::YCoCg),
            2 => Ok(Self::YCoCgR),
            _ => Err(EntroGdError::InvalidMetadata {
                message: format!(
                    "unsupported image color model {} (expected 0, 1, or 2)",
                    value
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageReconstructionInfo {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub color_model: ImageColorModel,
    pub pixel_grouping: PixelGrouping,
    pub grouping_transform: ImageGroupingTransform,
    /// 0 = sRGB + linear alpha, 1 = all linear
    pub colorspace: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitDataReconstructionInfo {
    Tabular,
    Image(ImageReconstructionInfo),
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct the original value of a feature from its bit-packed
/// representation by reversing the stored [`FeatureTransform`].
pub fn reconstruct_feature_value(bits: &BitSlice<usize, Lsb0>, spec: &FeatureSpec) -> f64 {
    let raw: u64 = bits.load_le::<u64>();

    match spec.transform {
        FeatureTransform::None => match spec.data_type {
            FeatureDataType::Float16 => f64::from(f32::from_bits((raw as u32) << 16)),
            FeatureDataType::Float32 => f64::from(f32::from_bits(raw as u32)),
            FeatureDataType::Float64 => f64::from_bits(raw),
            _ => raw as f64,
        },
        FeatureTransform::ScaledSignedInt { decimal_scale } => {
            let signed = (raw as i128).wrapping_add(i64::MIN as i128);
            signed as f64 / 10f64.powi(decimal_scale as i32)
        }
        FeatureTransform::OffsetSignedInt { min_value } => {
            (raw as i64).wrapping_add(min_value) as f64
        }
        FeatureTransform::OffsetUnsignedInt { min_value } => (raw + min_value) as f64,
        FeatureTransform::ScaledOffsetSignedInt {
            decimal_scale,
            min_value,
        } => {
            let signed = (raw as i64).wrapping_add(min_value) as f64;
            signed / 10f64.powi(decimal_scale as i32)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn load_le(slice: &BitSlice<usize, Lsb0>) -> u64 {
        slice.load_le::<u64>()
    }

    #[test]
    fn smallest_unsigned_for_picks_tightest_type() {
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(0),
            FeatureDataType::UInt8
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(255),
            FeatureDataType::UInt8
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(256),
            FeatureDataType::UInt16
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u16::MAX as u64),
            FeatureDataType::UInt16
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u16::MAX as u64 + 1),
            FeatureDataType::UInt32
        );
        assert_eq!(
            FeatureDataType::smallest_unsigned_for(u32::MAX as u64 + 1),
            FeatureDataType::UInt(33)
        );
    }

    #[test]
    fn signed_column_zero_normalizes_to_smallest_uint() {
        let s = Series::new("temperature".into(), &[-10i32, 0, 5, 20]);
        let options = PreprocessOptions::default();
        let transformed = transform_column(&s, &options).unwrap();

        // range = 20 - (-10) = 30, fits in UInt8
        assert_eq!(transformed.spec.data_type, FeatureDataType::UInt8);
        assert_eq!(
            transformed.spec.transform,
            FeatureTransform::OffsetSignedInt { min_value: -10 }
        );
        let values: Vec<u64> = transformed
            .series
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![0, 10, 15, 30]);
    }

    #[test]
    fn unsigned_column_zero_normalizes() {
        let s = Series::new("count".into(), &[100u32, 105, 110]);
        let options = PreprocessOptions::default();
        let transformed = transform_column(&s, &options).unwrap();

        assert_eq!(transformed.spec.data_type, FeatureDataType::UInt8);
        assert_eq!(
            transformed.spec.transform,
            FeatureTransform::OffsetUnsignedInt { min_value: 100 }
        );
        let values: Vec<u64> = transformed
            .series
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![0, 5, 10]);
    }

    #[test]
    fn float_column_scales_and_offsets() {
        let s = Series::new("price".into(), &[1.50f64, 2.25, 0.75]);
        let mut options = PreprocessOptions::default();
        options.decimal_scale = 2;
        let transformed = transform_column(&s, &options).unwrap();

        // scaled: 150, 225, 75 -> min 75 -> offsets: 75, 150, 0
        match transformed.spec.transform {
            FeatureTransform::ScaledOffsetSignedInt {
                decimal_scale,
                min_value,
            } => {
                assert_eq!(decimal_scale, 2);
                assert_eq!(min_value, 75);
            }
            other => panic!("unexpected transform: {:?}", other),
        }
        let values: Vec<u64> = transformed
            .series
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![75, 150, 0]);
    }

    #[test]
    fn from_dataframe_round_trips_values_through_bit_storage() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), &[-5i32, 0, 5]).into(),
                Series::new("b".into(), &[10i32, 20, 30]).into(),
            ],
        )
        .unwrap();

        let dataset = BitDataSet::from_dataframe(df, PreprocessOptions::default(), false).unwrap();
        assert_eq!(dataset.num_rows(), 3);
        assert_eq!(dataset.num_features(), 2);

        // Column "a": min -5 -> offsets 0, 5, 10
        assert_eq!(load_le(dataset.get_feature(0, 0)), 0);
        assert_eq!(load_le(dataset.get_feature(1, 0)), 5);
        assert_eq!(load_le(dataset.get_feature(2, 0)), 10);

        // Column "b": min 10 -> offsets 0, 10, 20
        assert_eq!(load_le(dataset.get_feature(0, 1)), 0);
        assert_eq!(load_le(dataset.get_feature(1, 1)), 10);
        assert_eq!(load_le(dataset.get_feature(2, 1)), 20);
    }
}
