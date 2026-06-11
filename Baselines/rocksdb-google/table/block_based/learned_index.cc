//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/block_based/learned_index.h"
#include "util/coding.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ROCKSDB_NAMESPACE {

static const uint32_t kLearnedIndexFormatMagic = 0x4C493200;  // "LI2\0"
static const uint32_t kLearnedIndexFormatVersion = 1;

// ============================================================================
// KeyEncoder Implementation
// ============================================================================

void KeyEncoder::AnalyzeKeys(const std::vector<std::string>& keys) {
  if (keys.empty()) return;

  // Find maximum key length
  max_key_length_ = 0;
  for (const auto& key : keys) {
    max_key_length_ = std::max(max_key_length_, static_cast<uint32_t>(key.size()));
  }

  // Initialize min/max arrays
  char_min_.resize(max_key_length_, 255);
  char_max_.resize(max_key_length_, 0);

  // Find min and max character at each position
  for (const auto& key : keys) {
    for (size_t i = 0; i < key.size(); ++i) {
      uint8_t ch = static_cast<uint8_t>(key[i]);
      char_min_[i] = std::min(char_min_[i], ch);
      char_max_[i] = std::max(char_max_[i], ch);
    }
    // For positions beyond this key, treat as 0
    for (size_t i = key.size(); i < max_key_length_; ++i) {
      char_min_[i] = std::min(char_min_[i], static_cast<uint8_t>(0));
    }
  }

  SetCharStats(max_key_length_, char_min_, char_max_);
}

void KeyEncoder::SetCharStats(uint32_t max_key_length,
                              const std::vector<uint8_t>& char_min,
                              const std::vector<uint8_t>& char_max) {
  max_key_length_ = max_key_length;
  char_min_ = char_min;
  char_max_ = char_max;

  char_bases_.clear();
  char_bases_.resize(max_key_length_);

  if (max_key_length_ == 0) {
    return;
  }

  // Compute total range product using long double to preserve precision.
  long double total_range = 1.0L;
  for (uint32_t i = 0; i < max_key_length_; ++i) {
    uint32_t range = static_cast<uint32_t>(char_max_[i]) -
                     static_cast<uint32_t>(char_min_[i]) + 1;
    total_range *= static_cast<long double>(range);
  }

  // Compute normalized bases so encoded keys are in [0, 1).
  // base_i = product(range_{j>i}) / total_range
  long double suffix = 1.0L;
  for (int i = static_cast<int>(max_key_length_) - 1; i >= 0; --i) {
    char_bases_[i] = static_cast<double>(suffix / total_range);
    uint32_t range = static_cast<uint32_t>(char_max_[i]) -
                     static_cast<uint32_t>(char_min_[i]) + 1;
    suffix *= static_cast<long double>(range);
  }
}

double KeyEncoder::EncodeKey(const Slice& key) const {
  if (char_bases_.empty()) return 0.0;

  long double encoded = 0.0L;
  size_t key_size = key.size();
  
  for (size_t i = 0; i < max_key_length_; ++i) {
    uint8_t ch = 0;
    if (i < key_size) {
      ch = static_cast<uint8_t>(key.data()[i]);
    }
    // Normalize character to [0, range)
    uint8_t normalized = ch >= char_min_[i] ? ch - char_min_[i] : 0;
    encoded += static_cast<long double>(normalized) *
               static_cast<long double>(char_bases_[i]);
  }
  
  return static_cast<double>(encoded);
}

void KeyEncoder::SerializeMetadata(std::string* dst) const {
  PutVarint32(dst, max_key_length_);
  for (uint32_t i = 0; i < max_key_length_; ++i) {
    dst->push_back(static_cast<char>(char_min_[i]));
    dst->push_back(static_cast<char>(char_max_[i]));
  }
}

bool KeyEncoder::DeserializeMetadata(const Slice& src) {
  Slice input = src;
  
  if (!GetVarint32(&input, &max_key_length_)) {
    return false;
  }

  if (input.size() < max_key_length_ * 2) {
    return false;
  }

  char_min_.resize(max_key_length_);
  char_max_.resize(max_key_length_);
  char_bases_.resize(max_key_length_);

  for (uint32_t i = 0; i < max_key_length_; ++i) {
    char_min_[i] = static_cast<uint8_t>(input.data()[i * 2]);
    char_max_[i] = static_cast<uint8_t>(input.data()[i * 2 + 1]);
  }

  // Recompute bases
  SetCharStats(max_key_length_, char_min_, char_max_);

  return true;
}

// ============================================================================
// LearnedIndex Implementation
// ============================================================================

bool LearnedIndex::Train(const std::vector<TrainingEntry>& training_data,
                         uint64_t target_block_size) {
  if (training_data.size() < 2) {
    return false;  // Need at least 2 points for linear regression
  }

  target_block_size_ = target_block_size;
  total_entries_ = training_data.size();

  // Sort training data by key (should already be sorted, but ensure it)
  std::vector<TrainingEntry> sorted_data = training_data;
  std::sort(sorted_data.begin(), sorted_data.end(),
            [](const TrainingEntry& a, const TrainingEntry& b) {
              return a.key < b.key;
            });

  // Extract keys for encoder analysis
  std::vector<std::string> keys;
  keys.reserve(sorted_data.size());
  for (const auto& entry : sorted_data) {
    keys.push_back(entry.key);
  }

  // Train the key encoder
  encoder_.AnalyzeKeys(keys);

  // Prepare data for linear regression
  // X = encoded keys, Y = cumulative_bytes (byte offset of this key in SST)
  // This is the core of Google's approach: f(key) → byte_offset
  std::vector<double> x_values;
  std::vector<double> y_values;
  x_values.reserve(sorted_data.size());
  y_values.reserve(sorted_data.size());

  for (size_t i = 0; i < sorted_data.size(); ++i) {
    double encoded_key = encoder_.EncodeKey(sorted_data[i].key);
    x_values.push_back(encoded_key);
    // Y is cumulative_bytes - the byte offset of this entry
    y_values.push_back(static_cast<double>(sorted_data[i].cumulative_bytes));
  }
  
  // Record total bytes for block_id calculation during read
  if (!sorted_data.empty()) {
    total_bytes_ = sorted_data.back().cumulative_bytes;
  }

  // Ordinary Least Squares (OLS) linear regression
  // Compute slope and intercept
  size_t n = x_values.size();
  double sum_x = std::accumulate(x_values.begin(), x_values.end(), 0.0);
  double sum_y = std::accumulate(y_values.begin(), y_values.end(), 0.0);
  double sum_xx = 0.0, sum_xy = 0.0;

  for (size_t i = 0; i < n; ++i) {
    sum_xx += x_values[i] * x_values[i];
    sum_xy += x_values[i] * y_values[i];
  }

  double mean_x = sum_x / n;
  double mean_y = sum_y / n;

  // slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
  double numerator = n * sum_xy - sum_x * sum_y;
  double denominator = n * sum_xx - sum_x * sum_x;

  if (std::abs(denominator) < 1e-10) {
    // Degenerate case: all keys are the same
    model_.SetSlope(0.0);
    model_.SetIntercept(mean_y);
  } else {
    double slope = numerator / denominator;
    double intercept = mean_y - slope * mean_x;
    model_.SetSlope(slope);
    model_.SetIntercept(intercept);
  }

  trained_ = true;
  return true;
}

bool LearnedIndex::TrainFromStats(const KeyEncoder& encoder,
                                  const RegressionStats& stats,
                                  uint64_t target_block_size,
                                  uint64_t total_entries,
                                  uint64_t total_bytes) {
  if (stats.n < 1) {
    return false;
  }

  encoder_ = encoder;
  target_block_size_ = target_block_size;
  total_entries_ = total_entries;
  total_bytes_ = total_bytes;
  segmented_ = false;
  segments_.clear();

  long double n = static_cast<long double>(stats.n);
  long double sum_x = stats.sum_x;
  long double sum_y = stats.sum_y;
  long double sum_xx = stats.sum_xx;
  long double sum_xy = stats.sum_xy;

  long double mean_x = sum_x / n;
  long double mean_y = sum_y / n;

  long double numerator = n * sum_xy - sum_x * sum_y;
  long double denominator = n * sum_xx - sum_x * sum_x;

  if (fabsl(denominator) < 1e-18L) {
    model_.SetSlope(0.0);
    model_.SetIntercept(static_cast<double>(mean_y));
  } else {
    long double slope = numerator / denominator;
    long double intercept = mean_y - slope * mean_x;
    model_.SetSlope(static_cast<double>(slope));
    model_.SetIntercept(static_cast<double>(intercept));
  }

  trained_ = true;
  return true;
}

bool LearnedIndex::TrainFromSegmentStats(
    const KeyEncoder& encoder, const std::vector<RegressionStats>& stats,
    const std::vector<double>& max_encoded_keys, uint64_t target_block_size,
    uint64_t total_entries, uint64_t total_bytes) {
  if (stats.empty() || stats.size() != max_encoded_keys.size()) {
    return false;
  }

  encoder_ = encoder;
  target_block_size_ = target_block_size;
  total_entries_ = total_entries;
  total_bytes_ = total_bytes;
  segmented_ = true;
  segments_.clear();
  segments_.reserve(stats.size());

  for (size_t i = 0; i < stats.size(); ++i) {
    const auto& s = stats[i];
    if (s.n < 1) {
      return false;
    }

    long double n = static_cast<long double>(s.n);
    long double sum_x = s.sum_x;
    long double sum_y = s.sum_y;
    long double sum_xx = s.sum_xx;
    long double sum_xy = s.sum_xy;

    long double mean_x = sum_x / n;
    long double mean_y = sum_y / n;

    long double numerator = n * sum_xy - sum_x * sum_y;
    long double denominator = n * sum_xx - sum_x * sum_x;

    double slope = 0.0;
    double intercept = static_cast<double>(mean_y);
    if (fabsl(denominator) >= 1e-18L) {
      long double lslope = numerator / denominator;
      long double lintercept = mean_y - lslope * mean_x;
      slope = static_cast<double>(lslope);
      intercept = static_cast<double>(lintercept);
    }

    SegmentModel seg;
    seg.slope = slope;
    seg.intercept = intercept;
    seg.max_encoded_key = max_encoded_keys[i];
    segments_.push_back(seg);
  }

  // Set a default model as the first segment (for compatibility)
  model_.SetSlope(segments_[0].slope);
  model_.SetIntercept(segments_[0].intercept);

  trained_ = true;
  return true;
}

// Core prediction method: block_id = floor(f(key) / tau)
// where f(key) predicts byte_offset and tau is target_block_size
uint32_t LearnedIndex::PredictBlock(const Slice& key, uint32_t /*num_blocks*/) const {
  if (!trained_ || target_block_size_ == 0) {
    return 0;
  }

  double encoded_key = encoder_.EncodeKey(key);

  double predicted_byte_offset = 0.0;
  if (segmented_ && !segments_.empty()) {
    // Find segment by upper_bound on max_encoded_key
    size_t lo = 0;
    size_t hi = segments_.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (encoded_key <= segments_[mid].max_encoded_key) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    size_t idx = (lo < segments_.size()) ? lo : (segments_.size() - 1);
    predicted_byte_offset =
        segments_[idx].slope * encoded_key + segments_[idx].intercept;
  } else {
    // f(key) predicts cumulative_bytes (byte offset)
    predicted_byte_offset = model_.Predict(encoded_key);
  }
  
  // Clamp to valid range [0, total_bytes_]
  if (predicted_byte_offset < 0) {
    predicted_byte_offset = 0;
  }
  if (predicted_byte_offset > total_bytes_) {
    predicted_byte_offset = static_cast<double>(total_bytes_);
  }
  
  // block_id = floor(byte_offset / tau)
  // This is the core formula from the paper
  uint32_t block_id = static_cast<uint32_t>(predicted_byte_offset / target_block_size_);
  
  return block_id;
}

// Write path version: same formula, used during SST construction
uint32_t LearnedIndex::PredictBlock(const Slice& key) const {
  return PredictBlock(key, 0);  // num_blocks not used in new formula
}

void LearnedIndex::Serialize(std::string* dst) const {
  // Format:
  // [trained_flag(1)] [magic(4)] [version(4)] [segmented_flag(1)]
  // [target_block_size(8)] [total_entries(8)] [total_bytes(8)]
  // [slope(8)] [intercept(8)] [encoder_metadata]
  // [segment_count(4)] [segment entries...]
  
  dst->push_back(trained_ ? 1 : 0);
  PutFixed32(dst, kLearnedIndexFormatMagic);
  PutFixed32(dst, kLearnedIndexFormatVersion);
  dst->push_back(segmented_ ? 1 : 0);
  PutFixed64(dst, target_block_size_);
  PutFixed64(dst, total_entries_);
  PutFixed64(dst, total_bytes_);  // Add total_bytes for read path
  
  // Serialize model as raw doubles
  double slope = model_.GetSlope();
  double intercept = model_.GetIntercept();
  dst->append(reinterpret_cast<const char*>(&slope), sizeof(double));
  dst->append(reinterpret_cast<const char*>(&intercept), sizeof(double));
  
  // Serialize encoder
  encoder_.SerializeMetadata(dst);

  // Serialize segments
  PutFixed32(dst, static_cast<uint32_t>(segments_.size()));
  for (const auto& seg : segments_) {
    dst->append(reinterpret_cast<const char*>(&seg.slope), sizeof(double));
    dst->append(reinterpret_cast<const char*>(&seg.intercept), sizeof(double));
    dst->append(reinterpret_cast<const char*>(&seg.max_encoded_key),
                sizeof(double));
  }
}

bool LearnedIndex::Deserialize(const Slice& src) {
  if (src.size() < 1 + 4 + 4 + 1 + 8 + 8 + 8 + 8 + 8) {
    return false;
  }

  Slice input = src;
  trained_ = (input.data()[0] != 0);
  input.remove_prefix(1);

  uint32_t magic = DecodeFixed32(input.data());
  if (magic == kLearnedIndexFormatMagic) {
    input.remove_prefix(4);
    uint32_t version = DecodeFixed32(input.data());
    input.remove_prefix(4);
    if (version != kLearnedIndexFormatVersion) {
      return false;
    }
    segmented_ = (input.data()[0] != 0);
    input.remove_prefix(1);

    target_block_size_ = DecodeFixed64(input.data());
    input.remove_prefix(8);
    total_entries_ = DecodeFixed64(input.data());
    input.remove_prefix(8);
    total_bytes_ = DecodeFixed64(input.data());
    input.remove_prefix(8);

    double slope, intercept;
    memcpy(&slope, input.data(), sizeof(double));
    input.remove_prefix(sizeof(double));
    memcpy(&intercept, input.data(), sizeof(double));
    input.remove_prefix(sizeof(double));
    model_.SetSlope(slope);
    model_.SetIntercept(intercept);

    if (!encoder_.DeserializeMetadata(input)) {
      return false;
    }

    uint32_t seg_count = 0;
    if (!GetFixed32(&input, &seg_count)) {
      return false;
    }
    segments_.clear();
    segments_.reserve(seg_count);
    for (uint32_t i = 0; i < seg_count; ++i) {
      if (input.size() < sizeof(double) * 3) {
        return false;
      }
      SegmentModel seg;
      memcpy(&seg.slope, input.data(), sizeof(double));
      input.remove_prefix(sizeof(double));
      memcpy(&seg.intercept, input.data(), sizeof(double));
      input.remove_prefix(sizeof(double));
      memcpy(&seg.max_encoded_key, input.data(), sizeof(double));
      input.remove_prefix(sizeof(double));
      segments_.push_back(seg);
    }
    return true;
  }

  // Backward-compatible: old format without magic/version/segments.
  if (src.size() < 1 + 8 + 8 + 8 + 8 + 8) {
    return false;
  }

  input = src;
  trained_ = (input.data()[0] != 0);
  input.remove_prefix(1);

  target_block_size_ = DecodeFixed64(input.data());
  input.remove_prefix(8);
  total_entries_ = DecodeFixed64(input.data());
  input.remove_prefix(8);
  total_bytes_ = DecodeFixed64(input.data());
  input.remove_prefix(8);

  double slope, intercept;
  memcpy(&slope, input.data(), sizeof(double));
  input.remove_prefix(sizeof(double));
  memcpy(&intercept, input.data(), sizeof(double));
  input.remove_prefix(sizeof(double));
  model_.SetSlope(slope);
  model_.SetIntercept(intercept);

  segmented_ = false;
  segments_.clear();

  return encoder_.DeserializeMetadata(input);
}

LearnedIndex::Stats LearnedIndex::GetStats() const {
  Stats stats;
  stats.model_slope = model_.GetSlope();
  stats.model_intercept = model_.GetIntercept();
  stats.num_blocks = 0;  // Will be set by caller
  stats.target_block_size = target_block_size_;
  return stats;
}

// ============================================================================
// BlockLocationMapper Implementation
// ============================================================================

void BlockLocationMapper::AddBlock(uint64_t offset, uint32_t size) {
  block_offsets_.push_back(offset);
  block_sizes_.push_back(size);
}

BlockLocationMapper::BlockLocation 
BlockLocationMapper::GetBlockLocation(uint32_t block_id) const {
  if (block_id >= block_offsets_.size()) {
    return {0, 0};
  }
  return {block_offsets_[block_id], block_sizes_[block_id]};
}

uint32_t BlockLocationMapper::FindBlockByBytes(uint64_t cumulative_bytes) const {
  if (block_offsets_.empty()) {
    return 0;
  }
  
  // Binary search to find the block containing cumulative_bytes
  // block_offsets_ stores the start offset of each block
  // We want to find the largest offset <= cumulative_bytes
  auto it = std::upper_bound(block_offsets_.begin(), block_offsets_.end(), cumulative_bytes);
  if (it == block_offsets_.begin()) {
    return 0;
  }
  --it;
  return static_cast<uint32_t>(it - block_offsets_.begin());
}

void BlockLocationMapper::Serialize(std::string* dst) const {
  // Format: [num_blocks(4)] [offset_0(8)] [size_0(4)] [offset_1(8)] [size_1(4)] ...
  PutFixed32(dst, static_cast<uint32_t>(block_offsets_.size()));
  
  for (size_t i = 0; i < block_offsets_.size(); ++i) {
    PutFixed64(dst, block_offsets_[i]);
    PutFixed32(dst, block_sizes_[i]);
  }
}

bool BlockLocationMapper::Deserialize(const Slice& src) {
  Slice input = src;
  
  uint32_t num_blocks;
  if (!GetFixed32(&input, &num_blocks)) {
    return false;
  }

  block_offsets_.clear();
  block_sizes_.clear();
  block_offsets_.reserve(num_blocks);
  block_sizes_.reserve(num_blocks);

  for (uint32_t i = 0; i < num_blocks; ++i) {
    uint64_t offset;
    uint32_t size;
    if (!GetFixed64(&input, &offset) || !GetFixed32(&input, &size)) {
      return false;
    }
    block_offsets_.push_back(offset);
    block_sizes_.push_back(size);
  }

  return true;
}

}  // namespace ROCKSDB_NAMESPACE
