//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV - Learned Index Integration for RocksDB
//  Main header file that includes all LeaderKV components
//
//  LeaderKV implements a two-level learned index structure based on
//  Piecewise Linear Regression (PLR) to accelerate point lookups in SST files.
//
//  Key components:
//  - PLR: Piecewise Linear Regression for building segment models
//  - LearnedIndex: Two-level learned index structure
//  - LearnedIndexData: Manages learned index data for SST files
//  - LearnedIndexBuilder: Builds learned index during SST creation
//  - LearnedIndexReader: Reads and queries learned index

#pragma once

#include "leader_index/bucket.h"
#include "leader_index/conflict.h"
#include "leader_index/leader_index_options.h"
#include "leader_index/learned_index.h"
#include "leader_index/learned_index_builder.h"
#include "leader_index/learned_index_data.h"
#include "leader_index/learned_index_reader.h"
#include "leader_index/linear_model.h"
#include "leader_index/node.h"
#include "leader_index/plr.h"
#include "leader_index/util.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Version information
constexpr int LEADER_INDEX_VERSION_MAJOR = 1;
constexpr int LEADER_INDEX_VERSION_MINOR = 0;
constexpr int LEADER_INDEX_VERSION_PATCH = 0;

inline std::string GetLeaderIndexVersion() {
  return std::to_string(LEADER_INDEX_VERSION_MAJOR) + "." +
         std::to_string(LEADER_INDEX_VERSION_MINOR) + "." +
         std::to_string(LEADER_INDEX_VERSION_PATCH);
}

// Enable/disable learned index globally
inline void EnableLearnedIndex(bool enable = true) {
  learned_index_enabled = enable;
  LeaderIndexManager::GetInstance().SetOptions(LeaderIndexOptions{.enabled = enable});
}

// Configure learned index options
inline void ConfigureLearnedIndex(const LeaderIndexOptions& options) {
  learned_index_enabled = options.enabled;
  ERROR_BOUND = options.error_bound;
  GROUP_SIZE = options.group_size;
  LeaderIndexManager::GetInstance().SetOptions(options);
}

// Check if learned index is enabled
inline bool IsLearnedIndexEnabled() {
  return learned_index_enabled;
}

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
