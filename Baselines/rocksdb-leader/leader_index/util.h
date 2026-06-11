//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV utility functions for RocksDB

#pragma once

#include <cstdint>
#include <string>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Configuration flags for LeaderKV
extern bool learned_index_enabled;
extern uint32_t model_error;
extern uint64_t block_num_entries;
extern uint64_t block_size;
extern uint64_t entry_size;
extern int internal_key_size;
extern int internal_addr_info_size;

// Convert a Slice key to uint64 for learned index
inline uint64_t SliceToInteger(const Slice& slice) {
  uint64_t result = 0;
  const char* data = slice.data();
  size_t size = slice.size();

  // Handle numeric string keys
  bool is_numeric = true;
  for (size_t i = 0; i < size; ++i) {
    if (!isdigit(static_cast<unsigned char>(data[i]))) {
      is_numeric = false;
      break;
    }
  }

  if (is_numeric && size > 0 && size <= 20) {
    // Parse as decimal number
    for (size_t i = 0; i < size; ++i) {
      result = result * 10 + (data[i] - '0');
    }
  } else {
    // Treat as binary data - use first 8 bytes as big-endian uint64
    size_t bytes_to_use = std::min(size, sizeof(uint64_t));
    for (size_t i = 0; i < bytes_to_use; ++i) {
      result = (result << 8) | static_cast<unsigned char>(data[i]);
    }
  }

  return result;
}

// Extract integer from a char pointer with given size
inline uint64_t ExtractInteger(const char* pos, size_t size) {
  uint64_t result = 0;
  for (size_t i = 0; i < size; ++i) {
    result = (result << 8) | static_cast<unsigned char>(pos[i]);
  }
  return result;
}

// Generate a key string from a numeric value
inline std::string GenerateKey(uint64_t value, size_t key_size = 16) {
  std::string result = std::to_string(value);
  if (result.size() < key_size) {
    result = std::string(key_size - result.size(), '0') + result;
  }
  return result;
}

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
