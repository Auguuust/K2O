//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Learned Index Public API for RocksDB
//
//  This header provides the public interface for configuring and using
//  LeaderKV learned indexes in RocksDB.

#pragma once

#include <cstdint>
#include <string>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// Options for LeaderKV learned index
// These options can be passed to RocksDB to enable learned index functionality
struct LeaderKVOptions {
  // Whether to enable learned index for SST files
  // Default: false
  bool enabled = false;

  // Error bound for Piecewise Linear Regression (PLR)
  // Larger values result in smaller model size but potentially slower lookups
  // Default: 32
  uint8_t error_bound = 32;

  // Number of segments grouped together in the second level of the index
  // Default: 16
  uint16_t group_size = 16;

  // Path prefix for storing learned index model files
  // If empty, models are stored alongside their corresponding SST files
  // Default: empty (use SST file location)
  std::string model_path_prefix;

  // Whether to train the learned index during compaction
  // Default: true
  bool train_during_compaction = true;

  // Whether to load existing learned index models on database open
  // Default: true
  bool load_existing_models = true;

  // Minimum number of key-value pairs required in an SST file
  // to build a learned index for it
  // Default: 1000
  uint64_t min_entries_for_index = 1000;

  // Maximum size (in bytes) of the learned index model per SST file
  // If 0, no limit is applied
  // Default: 0 (no limit)
  uint64_t max_model_size = 0;

  // Create default options
  LeaderKVOptions() = default;
};

}  // namespace ROCKSDB_NAMESPACE
