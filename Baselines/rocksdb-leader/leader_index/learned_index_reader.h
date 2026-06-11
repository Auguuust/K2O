//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Learned Index Reader - reads and uses trained index for lookups

#pragma once

#include <memory>
#include <string>

#include "leader_index/learned_index.h"
#include "leader_index/learned_index_data.h"
#include "leader_index/leader_index_options.h"
#include "leader_index/util.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// LearnedIndexReader loads and uses a trained learned index for point queries
class LearnedIndexReader {
 public:
  LearnedIndexReader() : is_loaded_(false) {}

  ~LearnedIndexReader() = default;

  // Load the learned index from disk
  Status Load(const std::string& model_path) {
    if (is_loaded_) {
      return Status::OK();
    }

    learned_data_ = std::make_unique<LearnedIndexData>(10);

    // Check if file exists
    FILE* f = fopen(model_path.c_str(), "r");
    if (f == nullptr) {
      return Status::NotFound("Learned index file not found: " + model_path);
    }
    fclose(f);

    learned_data_->ReadModel(model_path);
    model_path_ = model_path;
    is_loaded_ = true;

    return Status::OK();
  }

  // Load from an existing LearnedIndexData object
  void LoadFromData(std::unique_ptr<LearnedIndexData> data) {
    learned_data_ = std::move(data);
    is_loaded_ = (learned_data_ != nullptr && learned_data_->Learned());
  }

  // Check if index is loaded
  bool IsLoaded() const { return is_loaded_; }

  // Get the predicted position range for a key
  // Returns (lower_bound, upper_bound) of the predicted position
  std::pair<uint64_t, uint64_t> GetPosition(const Slice& key) const {
    if (!is_loaded_ || !learned_data_) {
      return std::make_pair(0, 0);
    }
    return learned_data_->GetPosition(key);
  }

  // Get the maximum valid position
  uint64_t GetMaxPosition() const {
    if (!is_loaded_ || !learned_data_) {
      return 0;
    }
    return learned_data_->MaxPosition();
  }

  // Get the number of entries
  uint64_t GetNumEntries() const {
    if (!is_loaded_ || !learned_data_) {
      return 0;
    }
    return learned_data_->size_;
  }

  // Get the minimum key in the index
  uint64_t GetMinKey() const {
    if (!is_loaded_ || !learned_data_) {
      return 0;
    }
    return learned_data_->min_key_;
  }

  // Get the maximum key in the index
  uint64_t GetMaxKey() const {
    if (!is_loaded_ || !learned_data_) {
      return 0;
    }
    return learned_data_->max_key_;
  }

  // Get the model path
  const std::string& GetModelPath() const { return model_path_; }

  // Get the error bound used by this index
  double GetError() const {
    if (!is_loaded_ || !learned_data_) {
      return 0;
    }
    return learned_data_->GetError();
  }

  // Check if a key is within the index range
  bool ContainsKey(const Slice& key) const {
    if (!is_loaded_ || !learned_data_) {
      return false;
    }
    uint64_t key_int = SliceToInteger(key);
    return key_int >= learned_data_->min_key_ &&
           key_int <= learned_data_->max_key_;
  }

  // Binary search fallback within predicted range
  // Used when the learned index prediction needs refinement
  template <typename Comparator>
  size_t BinarySearchInRange(const std::vector<Slice>& keys, const Slice& target,
                             size_t lower, size_t upper,
                             const Comparator& cmp) const {
    while (lower < upper) {
      size_t mid = lower + (upper - lower) / 2;
      int compare_result = cmp(keys[mid], target);
      if (compare_result < 0) {
        lower = mid + 1;
      } else if (compare_result > 0) {
        upper = mid;
      } else {
        return mid;  // Exact match
      }
    }
    return lower;  // Return insertion point
  }

 private:
  std::unique_ptr<LearnedIndexData> learned_data_;
  std::string model_path_;
  bool is_loaded_;
};

// Factory function to create a LearnedIndexReader for an SST file
inline std::unique_ptr<LearnedIndexReader> CreateLearnedIndexReader(
    const std::string& sst_file_path, const LeaderIndexOptions& options) {
  auto reader = std::make_unique<LearnedIndexReader>();

  if (!options.enabled) {
    return reader;
  }

  std::string model_path;
  if (!options.model_path_prefix.empty()) {
    // Extract file number from SST path
    size_t last_slash = sst_file_path.rfind('/');
    std::string filename = (last_slash != std::string::npos)
                               ? sst_file_path.substr(last_slash + 1)
                               : sst_file_path;
    size_t dot_pos = filename.find('.');
    if (dot_pos != std::string::npos) {
      model_path =
          options.model_path_prefix + "/" + filename.substr(0, dot_pos) + ".leader";
    } else {
      model_path = options.model_path_prefix + "/" + filename + ".leader";
    }
  } else {
    model_path = sst_file_path + ".leader";
  }

  Status s = reader->Load(model_path);
  // It's okay if loading fails - we just won't use the learned index

  return reader;
}

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
