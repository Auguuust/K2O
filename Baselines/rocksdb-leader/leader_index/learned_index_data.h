//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Learned Index Data management for SST files
//  Ported from LeaderKV project

#pragma once

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "leader_index/learned_index.h"
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Accumulated num entries array for tracking entries per block
class AccumulatedNumEntriesArray {
 public:
  std::vector<std::pair<uint64_t, std::string>> array;

 public:
  AccumulatedNumEntriesArray() = default;

  void Add(uint64_t num_entries, std::string&& key) {
    array.emplace_back(num_entries, std::move(key));
  }

  bool Search(const Slice& key, uint64_t lower, uint64_t upper, size_t* index,
              uint64_t* relative_lower, uint64_t* relative_upper) {
    if (array.empty()) return false;

    size_t left = 0, right = array.size() - 1;
    while (left < right) {
      size_t mid = (left + right) / 2;
      if (lower < array[mid].first)
        right = mid;
      else
        left = mid + 1;
    }

    if (upper >= array[left].first) {
      while (true) {
        if (left >= array.size()) return false;
        if (key.compare(array[left].second) <= 0) break;
        lower = array[left].first;
        ++left;
      }
      upper = std::min(upper, array[left].first - 1);
    }
    *index = left;
    *relative_lower = left > 0 ? lower - array[left - 1].first : lower;
    *relative_upper = left > 0 ? upper - array[left - 1].first : upper;
    return true;
  }

  bool SearchNoError(uint64_t position, size_t* index,
                     uint64_t* relative_position) {
    if (array.empty()) return false;
    *index = position / array[0].first;
    *relative_position = position % array[0].first;
    return *index < array.size();
  }

  uint64_t NumEntries() const {
    return array.empty() ? 0 : array.back().first;
  }
};

// LearnedIndexData manages the learned index for a single SST file
class LearnedIndexData {
 private:
  double error_;
  std::atomic<bool> learned_;
  std::atomic<bool> aborted_;
  bool learned_not_atomic_;
  std::atomic<bool> learning_;
  int allowed_seek_;
  int current_seek_;

 public:
  bool filled_;
  bool is_level_;

  uint64_t min_key_;
  uint64_t max_key_;
  uint64_t size_;

 public:
  LearnedIndex<double> leader_index_;
  std::vector<uint64_t> sst_keys_;
  AccumulatedNumEntriesArray num_entries_accumulated_;

  int level_;
  mutable int served_;
  uint64_t cost_;

  explicit LearnedIndexData(int allowed_seek)
      : error_(32),
        learned_(false),
        aborted_(false),
        learned_not_atomic_(false),
        learning_(false),
        allowed_seek_(allowed_seek),
        current_seek_(0),
        filled_(false),
        is_level_(false),
        min_key_(0),
        max_key_(0),
        size_(0),
        level_(0),
        served_(0),
        cost_(0) {}

  LearnedIndexData(const LearnedIndexData& other) = delete;

  // Get the predicted position range for a key
  std::pair<uint64_t, uint64_t> GetPosition(const Slice& target_x) {
    // Convert slice to double key
    double target_int = 0;
    if (target_x.size() >= sizeof(uint64_t)) {
      // Assuming numeric keys stored as big-endian uint64
      uint64_t key_val = 0;
      const char* data = target_x.data();
      for (size_t i = 0; i < std::min(target_x.size(), sizeof(uint64_t)); ++i) {
        key_val = (key_val << 8) | static_cast<unsigned char>(data[i]);
      }
      target_int = static_cast<double>(key_val);
    } else {
      // For small keys, try to parse as number
      std::string key_str(target_x.data(), target_x.size());
      try {
        target_int = std::stod(key_str);
      } catch (...) {
        return std::make_pair(size_, size_);
      }
    }

    if (target_int < static_cast<double>(min_key_) ||
        target_int > static_cast<double>(max_key_)) {
      return std::make_pair(size_, size_);
    }

    auto range = leader_index_.Find(target_int);
    if (range.first == -1 && range.second == -1) {
      return std::make_pair(size_, size_);
    }
    uint64_t lower = static_cast<uint64_t>(std::max(range.first, int64_t(0)));
    uint64_t upper =
        static_cast<uint64_t>(std::min(range.second, static_cast<int64_t>(size_ - 1)));
    return std::make_pair(lower, upper);
  }

  uint64_t MaxPosition() const { return size_ - 1; }

  double GetError() const { return error_; }

  // Train the learned index on the collected keys
  bool Learn() {
    if (sst_keys_.empty()) return false;

    // IMPORTANT: Keys must be sorted for PLR training
    std::sort(sst_keys_.begin(), sst_keys_.end());

    std::vector<double> double_sst_vec;
    size_ = sst_keys_.size();
    double_sst_vec.reserve(size_);
    for (auto& key : sst_keys_) {
      double_sst_vec.push_back(static_cast<double>(key));
    }

    leader_index_.BulkLoad(double_sst_vec);

    // Record min and max keys
    min_key_ = sst_keys_.front();
    max_key_ = sst_keys_.back();

    learned_.store(true);
    return true;
  }

  bool Learned() {
    if (learned_not_atomic_) return true;
    else if (learned_.load()) {
      learned_not_atomic_ = true;
      return true;
    }
    return false;
  }

  bool IsLearning() const { return learning_.load(); }

  void SetLearning(bool v) { learning_.store(v); }

  // Write the learned model to a file
  uint64_t WriteModel(const std::string& filename) {
    while (!learned_.load()) {
      // Wait for learning to complete
    }
    return leader_index_.Serialize(filename, sst_keys_, size_);
  }

  // Write the learned model with DKTable metadata to a file
  // Format: [DKTable Header (48 bytes)][Learned Index Model]
  // DKTable Header:
  //   magic (8 bytes): "DKTABLE\0"
  //   value_area_offset (8 bytes)
  //   learn_area_offset (8 bytes)
  //   num_entries (8 bytes)
  //   entry_size (4 bytes)
  //   internal_key_size (4 bytes)
  //   block_num_entries (4 bytes)
  //   reserved (4 bytes)
  uint64_t WriteModelWithDKTableMeta(const std::string& filename,
                                     uint64_t value_area_offset,
                                     uint64_t learn_area_offset,
                                     uint64_t num_entries,
                                     uint32_t entry_size,
                                     uint32_t internal_key_size,
                                     uint32_t block_num_entries) {
    while (!learned_.load()) {
      // Wait for learning to complete
    }
    
    // Create a temp file for the model
    std::string model_tmp = filename + ".model_tmp";
    uint64_t model_size = leader_index_.Serialize(model_tmp, sst_keys_, size_);
    if (model_size == 0) {
      return 0;
    }
    
    // Read model data
    FILE* model_file = fopen(model_tmp.c_str(), "rb");
    if (!model_file) {
      return 0;
    }
    fseek(model_file, 0, SEEK_END);
    size_t model_file_size = ftell(model_file);
    fseek(model_file, 0, SEEK_SET);
    std::vector<char> model_data(model_file_size);
    if (fread(model_data.data(), 1, model_file_size, model_file) != model_file_size) {
      fclose(model_file);
      return 0;
    }
    fclose(model_file);
    remove(model_tmp.c_str());
    
    // Write final .li file with header
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) {
      return 0;
    }
    
    // Write magic
    const char magic[8] = {'D', 'K', 'T', 'A', 'B', 'L', 'E', '\0'};
    fwrite(magic, 1, 8, f);
    
    // Write DKTable metadata
    fwrite(&value_area_offset, sizeof(value_area_offset), 1, f);
    fwrite(&learn_area_offset, sizeof(learn_area_offset), 1, f);
    fwrite(&num_entries, sizeof(num_entries), 1, f);
    fwrite(&entry_size, sizeof(entry_size), 1, f);
    fwrite(&internal_key_size, sizeof(internal_key_size), 1, f);
    fwrite(&block_num_entries, sizeof(block_num_entries), 1, f);
    uint32_t reserved = 0;
    fwrite(&reserved, sizeof(reserved), 1, f);
    
    // Write model data
    fwrite(model_data.data(), 1, model_data.size(), f);
    
    fclose(f);
    return 48 + model_data.size();
  }

  // Read the learned model from a file
  void ReadModel(const std::string& filename) {
    leader_index_.DeserializeBuild(filename, sst_keys_, min_key_, max_key_,
                                   size_);
    learned_.store(true);
  }
  
  // Read the learned model with DKTable metadata from a file
  // Returns true if successful and file has DKTable header
  bool ReadModelWithDKTableMeta(const std::string& filename,
                                uint64_t& value_area_offset,
                                uint64_t& learn_area_offset,
                                uint64_t& num_entries,
                                uint32_t& entry_size,
                                uint32_t& internal_key_size,
                                uint32_t& block_num_entries) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) {
      return false;
    }
    
    // Read and check magic
    char magic[8];
    if (fread(magic, 1, 8, f) != 8) {
      fclose(f);
      return false;
    }
    
    if (memcmp(magic, "DKTABLE\0", 8) != 0) {
      // Old format without DKTable header, fall back to standard ReadModel
      fclose(f);
      ReadModel(filename);
      return false;
    }
    
    // Read DKTable metadata
    if (fread(&value_area_offset, sizeof(value_area_offset), 1, f) != 1 ||
        fread(&learn_area_offset, sizeof(learn_area_offset), 1, f) != 1 ||
        fread(&num_entries, sizeof(num_entries), 1, f) != 1 ||
        fread(&entry_size, sizeof(entry_size), 1, f) != 1 ||
        fread(&internal_key_size, sizeof(internal_key_size), 1, f) != 1 ||
        fread(&block_num_entries, sizeof(block_num_entries), 1, f) != 1) {
      fclose(f);
      return false;
    }
    
    // Skip reserved
    uint32_t reserved;
    if (fread(&reserved, sizeof(reserved), 1, f) != 1) {
      fclose(f);
      return false;
    }
    
    // Get remaining file size for model data
    long header_size = 48;
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long model_size = file_size - header_size;
    fclose(f);
    
    if (model_size <= 0) {
      return false;
    }
    
    // Create temp file with model data only
    std::string model_tmp = filename + ".model_tmp";
    FILE* src = fopen(filename.c_str(), "rb");
    FILE* dst = fopen(model_tmp.c_str(), "wb");
    if (!src || !dst) {
      if (src) fclose(src);
      if (dst) fclose(dst);
      return false;
    }
    
    fseek(src, header_size, SEEK_SET);
    std::vector<char> buf(model_size);
    if (fread(buf.data(), 1, model_size, src) != (size_t)model_size) {
      fclose(src);
      fclose(dst);
      return false;
    }
    fwrite(buf.data(), 1, model_size, dst);
    fclose(src);
    fclose(dst);
    
    // Read model from temp file
    leader_index_.DeserializeBuild(model_tmp, sst_keys_, min_key_, max_key_, size_);
    remove(model_tmp.c_str());
    
    learned_.store(true);
    return true;
  }
};

// FileLearnedIndexData manages learned indices for all SST files
class FileLearnedIndexData {
 private:
  std::mutex mutex_;
  std::vector<LearnedIndexData*> file_learned_index_data_;

 public:
  uint64_t watermark_;

  FileLearnedIndexData() : watermark_(0) {}

  ~FileLearnedIndexData() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto pointer : file_learned_index_data_) {
      delete pointer;
    }
  }

  // Get or create learned index data for a file
  LearnedIndexData* GetModel(int number) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Create new learned index data if not exist
    if (static_cast<size_t>(number) >= file_learned_index_data_.size()) {
      file_learned_index_data_.resize(number + 1, nullptr);
    }
    if (file_learned_index_data_[number] == nullptr) {
      file_learned_index_data_[number] = new LearnedIndexData(10);
    }
    return file_learned_index_data_[number];
  }

  std::vector<uint64_t>& GetData(int file_num) {
    auto* model = GetModel(file_num);
    return model->sst_keys_;
  }

  bool Learned(int file_num) {
    LearnedIndexData* model = GetModel(file_num);
    return model->Learned();
  }

  AccumulatedNumEntriesArray* GetAccumulatedArray(int file_num) {
    auto* model = GetModel(file_num);
    return &model->num_entries_accumulated_;
  }

  std::pair<uint64_t, uint64_t> GetPosition(const Slice& key, int file_num) {
    if (static_cast<size_t>(file_num) >= file_learned_index_data_.size() ||
        file_learned_index_data_[file_num] == nullptr) {
      return std::make_pair(0, 0);
    }
    return file_learned_index_data_[file_num]->GetPosition(key);
  }
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
