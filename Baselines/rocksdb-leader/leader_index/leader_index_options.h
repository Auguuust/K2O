//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV options and configuration for RocksDB

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Options for LeaderKV learned index
struct LeaderIndexOptions {
  // Whether to enable learned index
  bool enabled = false;

  // Error bound for PLR (Piecewise Linear Regression)
  // Larger values reduce model size but may increase lookup time
  uint8_t error_bound = 32;

  // Number of segments grouped together in the second level
  uint16_t group_size = 16;

  // Path prefix for storing learned index model files
  // If empty, models are stored alongside SST files
  std::string model_path_prefix;

  // Whether to train index during compaction
  bool train_during_compaction = true;

  // Whether to load existing models on startup
  bool load_existing_models = true;

  // Minimum number of entries required to build a learned index for a file
  uint64_t min_entries_for_index = 1000;

  // Maximum model size (in bytes) per SST file
  // If 0, no limit is applied
  uint64_t max_model_size = 0;

  // Default constructor
  LeaderIndexOptions() = default;

  // Copy constructor
  LeaderIndexOptions(const LeaderIndexOptions& other) = default;

  // Move constructor
  LeaderIndexOptions(LeaderIndexOptions&& other) noexcept = default;

  // Copy assignment
  LeaderIndexOptions& operator=(const LeaderIndexOptions& other) = default;

  // Move assignment
  LeaderIndexOptions& operator=(LeaderIndexOptions&& other) noexcept = default;
};

// Global learned index manager (singleton)
class LeaderIndexManager {
 public:
  static LeaderIndexManager& GetInstance() {
    static LeaderIndexManager instance;
    return instance;
  }

  void SetOptions(const LeaderIndexOptions& options) {
    options_ = options;
  }

  const LeaderIndexOptions& GetOptions() const { return options_; }

  bool IsEnabled() const { return options_.enabled; }

 private:
  LeaderIndexManager() = default;
  LeaderIndexOptions options_;
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
