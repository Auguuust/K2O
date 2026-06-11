//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Learned Index Builder - builds index during SST creation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "leader_index/learned_index.h"
#include "leader_index/learned_index_data.h"
#include "leader_index/leader_index_options.h"
#include "leader_index/util.h"
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// LearnedIndexBuilder collects keys during SST building and trains the model
class LearnedIndexBuilder {
 public:
  LearnedIndexBuilder(const LeaderIndexOptions& options,
                      const std::string& sst_file_path)
      : options_(options),
        sst_file_path_(sst_file_path),
        num_entries_(0),
        is_finalized_(false) {}

  ~LearnedIndexBuilder() = default;

  // Add a key to the builder
  void Add(const Slice& key) {
    if (!options_.enabled) return;

    uint64_t key_int = SliceToInteger(key);
    keys_.push_back(key_int);
    num_entries_++;
  }

  // Add entry boundary information (for block-based tables)
  void AddBlockEntry(uint64_t block_offset, const std::string& first_key) {
    block_boundaries_.emplace_back(block_offset, first_key);
  }

  // Finalize the builder and train the model
  bool Finalize() {
    if (!options_.enabled || is_finalized_) return false;
    if (num_entries_ < options_.min_entries_for_index) {
      is_finalized_ = true;
      return false;
    }

    // Sort keys if not already sorted
    // (SST keys should already be sorted)

    // Create learned index data
    learned_data_ = std::make_unique<LearnedIndexData>(10);
    learned_data_->sst_keys_ = std::move(keys_);

    // Train the model
    bool success = learned_data_->Learn();
    is_finalized_ = true;

    return success;
  }

  // Get the model path for this SST file
  std::string GetModelPath() const {
    if (!options_.model_path_prefix.empty()) {
      return options_.model_path_prefix + "/" +
             std::to_string(GetFileNumber()) + ".leader";
    }
    // Store alongside SST file by default
    return sst_file_path_ + ".leader";
  }

  // Write the trained model to disk
  uint64_t WriteModel() {
    if (!is_finalized_ || !learned_data_ || !learned_data_->Learned()) {
      return 0;
    }
    return learned_data_->WriteModel(GetModelPath());
  }

  // Get number of entries
  uint64_t GetNumEntries() const { return num_entries_; }

  // Check if model was successfully built
  bool HasModel() const {
    return is_finalized_ && learned_data_ && learned_data_->Learned();
  }

  // Get learned index data (ownership transfer)
  std::unique_ptr<LearnedIndexData> ReleaseLearnedData() {
    return std::move(learned_data_);
  }

 private:
  uint64_t GetFileNumber() const {
    // Extract file number from SST path
    size_t last_slash = sst_file_path_.rfind('/');
    std::string filename = (last_slash != std::string::npos)
                               ? sst_file_path_.substr(last_slash + 1)
                               : sst_file_path_;
    size_t dot_pos = filename.find('.');
    if (dot_pos != std::string::npos) {
      try {
        return std::stoull(filename.substr(0, dot_pos));
      } catch (...) {
        return 0;
      }
    }
    return 0;
  }

  LeaderIndexOptions options_;
  std::string sst_file_path_;
  std::vector<uint64_t> keys_;
  std::vector<std::pair<uint64_t, std::string>> block_boundaries_;
  uint64_t num_entries_;
  bool is_finalized_;
  std::unique_ptr<LearnedIndexData> learned_data_;
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
