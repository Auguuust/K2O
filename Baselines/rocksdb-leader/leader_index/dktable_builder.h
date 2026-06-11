//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// DKTable Builder - Helper class for building DKTable format SST files
// DKTable format: [Value Area] [Learn Area] [Meta Blocks] [Footer]

#pragma once

#include <memory>
#include <vector>
#include <array>
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "leader_index/learned_index_data.h"
#include "leader_index/util.h"

namespace ROCKSDB_NAMESPACE {

// Helper class to manage DKTable-specific data during SST construction
class DKTableBuilder {
 public:
  static constexpr size_t kMaxDKTableEntries = 2000000;
  static constexpr size_t kMaxKeySize = 40;

  DKTableBuilder() : current_idx_(0), value_area_size_(0) {
    keys_.reserve(10000);
    addr_info_.reserve(10000);
  }

  // Record a key-value pair for later Learn Area construction
  // offset: offset of the value in the Value Area
  // value_size: size of the value
  // key: the full internal key
  void RecordEntry(uint64_t offset, uint32_t value_size, const Slice& key) {
    if (current_idx_ >= kMaxDKTableEntries) {
      return;  // Silently ignore overflow for now
    }

    // Store key
    std::array<char, kMaxKeySize> key_buf{};
    size_t key_len = std::min(key.size(), kMaxKeySize);
    std::memcpy(key_buf.data(), key.data(), key_len);
    keys_.push_back(key_buf);

    // Store address info: [offset(4 bytes), length(4 bytes)]
    std::array<char, 8> addr_buf{};
    EncodeFixed32(addr_buf.data(), static_cast<uint32_t>(offset));
    EncodeFixed32(addr_buf.data() + 4, value_size);
    addr_info_.push_back(addr_buf);

    current_idx_++;
  }

  // Get the number of entries recorded
  uint32_t GetEntryCount() const { return current_idx_; }

  // Get key at index i
  Slice GetKey(uint32_t i) const {
    if (i >= current_idx_) return Slice();
    // Find actual key length (trim null bytes)
    size_t len = 0;
    for (size_t j = 0; j < kMaxKeySize; ++j) {
      if (keys_[i][j] != 0) len = j + 1;
    }
    return Slice(keys_[i].data(), len);
  }

  // Get address info at index i
  Slice GetAddrInfo(uint32_t i) const {
    if (i >= current_idx_) return Slice();
    return Slice(addr_info_[i].data(), 8);
  }

  // Extract numeric keys for learned index training
  std::vector<uint64_t> ExtractNumericKeys() const {
    std::vector<uint64_t> numeric_keys;
    numeric_keys.reserve(current_idx_);
    
    for (uint32_t i = 0; i < current_idx_; ++i) {
      Slice key = GetKey(i);
      if (key.size() > 0) {
        // Extract user key (remove 8-byte suffix for internal key format)
        size_t user_key_size = key.size() >= 8 ? key.size() - 8 : key.size();
        Slice user_key(key.data(), user_key_size);
        numeric_keys.push_back(SliceToInteger(user_key));
      }
    }
    return numeric_keys;
  }

  void SetValueAreaSize(uint64_t size) { value_area_size_ = size; }
  uint64_t GetValueAreaSize() const { return value_area_size_; }

 private:
  std::vector<std::array<char, kMaxKeySize>> keys_;
  std::vector<std::array<char, 8>> addr_info_;
  uint32_t current_idx_;
  uint64_t value_area_size_;  // Total size of Value Area
};

}  // namespace ROCKSDB_NAMESPACE
