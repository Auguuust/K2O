//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Bucket implementation for handling conflicts in learned index
//  Ported from LeaderKV project

#pragma once

#include <iostream>
#include <optional>
#include <utility>
#include <cstdint>

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Bucket class for handling conflicts (multiple entries mapping to same position)
template <typename KT, typename VT>
class Bucket {
  typedef std::pair<KT, VT> KVT;

 public:
  KVT* data_;
  uint32_t size_;
  KT x_;   // minimum key
  KT x2_;  // maximum key

 public:
  Bucket() : data_(nullptr), size_(0), x_(0), x2_(0) {}

  Bucket(const KVT* kvs, uint32_t size, uint32_t capacity) : size_(size) {
    data_ = new KVT[capacity];
    for (uint32_t i = 0; i < size; ++i) {
      data_[i] = kvs[i];
    }
  }

  ~Bucket() {
    delete[] data_;
    data_ = nullptr;
  }

  // Find the entry whose key range contains the target key
  std::optional<VT> Find(KT key) const {
    // Using binary search for better performance
    uint32_t l = 0, r = size_;
    while (l < r) {
      uint32_t m = (l + r) >> 1;
      if (key > data_[m].first) {
        l = m + 1;
      } else {
        r = m;
      }
    }
    // idx is the position of binary insert
    int idx = l;
    if (idx < static_cast<int>(size_) && data_[idx].first <= key) {
      return data_[idx].second;
    } else {
      idx -= 1;
    }
    if (idx >= 0 && data_[idx].first <= key) {
      return data_[idx].second;
    }
    return std::nullopt;
  }
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
