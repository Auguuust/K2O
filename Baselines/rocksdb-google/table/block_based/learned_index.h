//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Learned Index for Block-based SSTable
// Based on "Learned Indexes for a Google-scale Disk-based Database"
// https://arxiv.org/pdf/2012.12501

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// LinearModel implements a simple linear regression: f(x) = slope * x + intercept
class LinearModel {
 public:
  LinearModel() : slope_(0.0), intercept_(0.0) {}
  LinearModel(double slope, double intercept) 
      : slope_(slope), intercept_(intercept) {}

  // Predict the byte offset for a given encoded key
  double Predict(double encoded_key) const {
    return slope_ * encoded_key + intercept_;
  }

  double GetSlope() const { return slope_; }
  double GetIntercept() const { return intercept_; }

  void SetSlope(double slope) { slope_ = slope; }
  void SetIntercept(double intercept) { intercept_ = intercept; }

 private:
  double slope_;
  double intercept_;
};

// KeyEncoder converts string keys to monotonic double values
// This ensures the learned model maintains monotonicity
class KeyEncoder {
 public:
  KeyEncoder() = default;

  // Analyze keys to determine encoding parameters
  // This must be called before encoding any keys
  void AnalyzeKeys(const std::vector<std::string>& keys);

  // Set encoding stats directly (useful for streaming training)
  void SetCharStats(uint32_t max_key_length,
                    const std::vector<uint8_t>& char_min,
                    const std::vector<uint8_t>& char_max);

  // Encode a single key to a double value
  double EncodeKey(const Slice& key) const;

  // Get encoding metadata size for serialization
  size_t GetMetadataSize() const {
    return sizeof(uint32_t) + char_bases_.size() * (sizeof(uint8_t) * 2);
  }

  // Serialize encoding metadata
  void SerializeMetadata(std::string* dst) const;
  
  // Deserialize encoding metadata
  bool DeserializeMetadata(const Slice& src);

 private:
  // For each character position, store [min_char, max_char]
  // to compute the base for that position
  std::vector<uint8_t> char_min_;
  std::vector<uint8_t> char_max_;
  std::vector<double> char_bases_;  // Computed bases for each position
  uint32_t max_key_length_ = 0;
};

// TrainingData holds key-value pairs for training
struct TrainingEntry {
  std::string key;
  uint64_t cumulative_bytes;  // Cumulative bytes up to this key
  uint64_t entry_size;        // Size of this individual entry (for sorting)
  
  TrainingEntry(const std::string& k, uint64_t bytes, uint64_t size = 0) 
      : key(k), cumulative_bytes(bytes), entry_size(size) {}
};

// LearnedIndex trains and uses a learned index model
// Following Google's approach: model predicts entry index, then maps to block
class LearnedIndex {
 public:
  LearnedIndex() = default;

  // Train the model on collected data
  // training_data: vector of (key, cumulative_bytes) pairs
  // target_block_size: desired average block size in bytes
  // Returns true if training succeeded
  bool Train(const std::vector<TrainingEntry>& training_data,
             uint64_t target_block_size);

  struct RegressionStats {
    uint64_t n = 0;
    long double sum_x = 0.0L;
    long double sum_y = 0.0L;
    long double sum_xx = 0.0L;
    long double sum_xy = 0.0L;
  };

  // Train the model from precomputed regression stats and an encoder.
  bool TrainFromStats(const KeyEncoder& encoder, const RegressionStats& stats,
                      uint64_t target_block_size, uint64_t total_entries,
                      uint64_t total_bytes);

  struct SegmentModel {
    double slope = 0.0;
    double intercept = 0.0;
    double max_encoded_key = 0.0;  // Upper bound of this segment
  };

  bool TrainFromSegmentStats(const KeyEncoder& encoder,
                             const std::vector<RegressionStats>& stats,
                             const std::vector<double>& max_encoded_keys,
                             uint64_t target_block_size, uint64_t total_entries,
                             uint64_t total_bytes);

  // Predict which block a key should go to (for read path)
  // num_blocks: actual number of blocks in the SST file
  // Returns the predicted block number
  uint32_t PredictBlock(const Slice& key, uint32_t num_blocks) const;
  
  // Legacy predict for write path (uses estimated entries_per_block)
  uint32_t PredictBlock(const Slice& key) const;

  // Check if the model is trained and ready
  bool IsTrained() const { return trained_; }

  // Get the target block size
  uint64_t GetTargetBlockSize() const { return target_block_size_; }
  
  // Get total entries used in training
  uint64_t GetTotalEntries() const { return total_entries_; }

  // Serialize the learned index (model + encoder + block mapping)
  void Serialize(std::string* dst) const;

  // Deserialize the learned index
  bool Deserialize(const Slice& src);

  // Get model statistics for debugging
  struct Stats {
    double model_slope;
    double model_intercept;
    uint32_t num_blocks;
    uint64_t target_block_size;
  };
  Stats GetStats() const;

  // Get the encoded key value (for debugging)
  double GetEncodedKey(const Slice& key) const {
    return encoder_.EncodeKey(key);
  }

 private:
  LinearModel model_;
  KeyEncoder encoder_;
  uint64_t target_block_size_ = 0;
  uint64_t total_entries_ = 0;
  uint64_t total_bytes_ = 0;  // Total bytes from training data (for clamping)
  bool trained_ = false;
  bool segmented_ = false;
  std::vector<SegmentModel> segments_;
};

// BlockLocationMapper stores the physical location of each block
// Implements compressed storage as blocks are approximately uniformly spaced
class BlockLocationMapper {
 public:
  BlockLocationMapper() = default;

  // Add a block location
  void AddBlock(uint64_t offset, uint32_t size);

  // Get block location and size
  struct BlockLocation {
    uint64_t offset;
    uint32_t size;
  };
  BlockLocation GetBlockLocation(uint32_t block_id) const;
  
  // Find block by cumulative bytes (binary search)
  // Returns the block that contains the given byte offset
  uint32_t FindBlockByBytes(uint64_t cumulative_bytes) const;

  // Get total number of blocks
  uint32_t GetNumBlocks() const { return static_cast<uint32_t>(block_offsets_.size()); }
  
  // Get total bytes (last block end offset)
  uint64_t GetTotalBytes() const {
    if (block_offsets_.empty()) return 0;
    return block_offsets_.back() + block_sizes_.back();
  }

  // Serialize the mapper
  void Serialize(std::string* dst) const;

  // Deserialize the mapper
  bool Deserialize(const Slice& src);

  // Get compressed size estimate
  size_t EstimateSize() const {
    // Store first offset + deltas (varint encoded)
    return sizeof(uint64_t) + block_offsets_.size() * sizeof(uint32_t);
  }

 private:
  std::vector<uint64_t> block_offsets_;
  std::vector<uint32_t> block_sizes_;
};

}  // namespace ROCKSDB_NAMESPACE
