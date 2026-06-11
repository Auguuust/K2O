//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV TNode (Tree Node) implementation for Leader learned index
//  Ported from LeaderKV project

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <cassert>

#include "leader_index/bucket.h"
#include "leader_index/conflict.h"
#include "leader_index/linear_model.h"
#include "leader_index/plr.h"

// Bit manipulation macros
#define BIT_TYPE uint8_t
#define BIT_SIZE (sizeof(BIT_TYPE) * 8)
#define BIT_LEN(x) (static_cast<uint32_t>(std::ceil((x) * 1.0 / BIT_SIZE)))
#define BIT_IDX(x) ((x) / BIT_SIZE)
#define BIT_POS(x) ((x) % BIT_SIZE)
#define SET_BIT_ONE(x, n) ((x) |= (1 << (n)))
#define SET_BIT_ZERO(x, n) ((x) &= (~(1 << (n))))
#define REV_BIT(x, n) ((x) ^= (1 << (n)))
#define GET_BIT(x, n) (((x) >> (n)) & 1)

namespace ROCKSDB_NAMESPACE {
namespace leader {

template <typename KT>
class TNode;

// Union for storing different entry types in a node
template <typename KT>
union InternalEntry {
  TNode<KT>* child_;               // Child node pointer
  Segment* seg_;                   // Segment pointer (leaf level)
  Bucket<KT, Segment*>* bucket_seg_;      // Bucket for segments
  Bucket<KT, TNode<KT>*>* bucket_node_;   // Bucket for nodes

  InternalEntry() : child_(nullptr) {}
};

template <typename KT>
struct Entry {
  InternalEntry<KT> internal;
  InternalEntry<KT> last_entry;  // Last valid entry for lookup
};

// Entry types in the learned index
enum EntryType {
  kNone = 0,     // Empty but stores last valid entry address
  kNode = 1,     // Internal node pointer
  kSegment = 2,  // Leaf segment pointer
  kBucket = 3    // Bucket for handling conflicts
};

// Hyperparameters for building learned index
struct HyperParameter {
  const uint32_t kMaxBucketSize = 6;
  const uint32_t kMinBucketSize = 1;
  const double kSizeAmplification = 2;
  const double kTailPercent = 0.99;
};

// Helper functions
template <typename T>
inline T LeaderMin(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
inline T LeaderMax(T a, T b) {
  return a > b ? a : b;
}

// TNode: A tree node in the two-level Leader learned index
template <typename KT>
class TNode {
 public:
  LinearModel<KT>* model_;
  uint32_t capacity_;
  uint8_t* bitmap0_;  // Bit i indicates entry type (with bitmap1_)
  uint8_t* bitmap1_;
  Entry<KT>* entries_;
  KT x_;              // Minimum key managed by this node
  KT x2_;             // Maximum key managed by this node
  bool is_bucket_seg_;  // Whether bucket stores segments

  explicit TNode()
      : model_(nullptr),
        capacity_(0),
        bitmap0_(nullptr),
        bitmap1_(nullptr),
        entries_(nullptr),
        x_(1e30),
        x2_(0),
        is_bucket_seg_(true) {}

  ~TNode() { DestroyNode(); }

  void DestroyNode() {
    if (model_ != nullptr) {
      delete model_;
      model_ = nullptr;

      for (uint32_t i = 0; i < capacity_; ++i) {
        uint8_t type_i = GetEntryType(i);
        if (type_i == kBucket) {
          if (is_bucket_seg_) {
            if (entries_[i].internal.bucket_seg_) {
              delete entries_[i].internal.bucket_seg_;
              entries_[i].internal.bucket_seg_ = nullptr;
            }
          } else {
            if (entries_[i].internal.bucket_node_) {
              delete entries_[i].internal.bucket_node_;
              entries_[i].internal.bucket_node_ = nullptr;
            }
          }
        } else if (type_i == kNode) {
          uint32_t j = i;
          for (; j < capacity_; ++j) {
            uint8_t type_j = GetEntryType(j);
            if (type_j != kNode ||
                entries_[j].internal.child_ != entries_[i].internal.child_) {
              break;
            }
          }
          if (entries_[i].internal.child_ != nullptr) {
            delete entries_[i].internal.child_;
            entries_[i].internal.child_ = nullptr;
          }
          i = j - 1;  // Prepare for next iteration
        } else if (type_i == kSegment) {
          // Segments are owned by vector, no need to delete
        }
      }

      delete[] bitmap0_;
      bitmap0_ = nullptr;
      delete[] bitmap1_;
      bitmap1_ = nullptr;
    }

    if (entries_ != nullptr) {
      delete[] entries_;
      entries_ = nullptr;
    }
    capacity_ = 0;
  }

  // Get entry type using 2-bit encoding in bitmaps
  uint8_t GetEntryType(uint32_t idx) const {
    uint8_t b0 = GET_BIT(bitmap0_[BIT_IDX(idx)], BIT_POS(idx));
    uint8_t b1 = GET_BIT(bitmap1_[BIT_IDX(idx)], BIT_POS(idx));
    return (b1 << 1) | b0;
  }

  void SetEntryType(uint32_t idx, uint8_t type) {
    if (type & 1) {
      SET_BIT_ONE(bitmap0_[BIT_IDX(idx)], BIT_POS(idx));
    } else {
      SET_BIT_ZERO(bitmap0_[BIT_IDX(idx)], BIT_POS(idx));
    }
    if (type & 2) {
      SET_BIT_ONE(bitmap1_[BIT_IDX(idx)], BIT_POS(idx));
    } else {
      SET_BIT_ZERO(bitmap1_[BIT_IDX(idx)], BIT_POS(idx));
    }
  }

  // Build the node with given keys and entries
  void Build(const KT* kvs_key, Entry<double>** kvs_entry, uint32_t size,
             HyperParameter& hyper_para, bool to_index_seg) {
    ConflictsInfo* ci =
        BuildLinearModel(kvs_key, size, model_, hyper_para.kSizeAmplification);
    assert(ci != nullptr);

    // Allocate memory for the node
    const uint32_t bit_len = BIT_LEN(ci->max_size_);
    capacity_ = ci->max_size_;
    bitmap0_ = new BIT_TYPE[bit_len];
    bitmap1_ = new BIT_TYPE[bit_len];
    entries_ = new Entry<KT>[capacity_];
    is_bucket_seg_ = to_index_seg;
    memset(bitmap0_, 0, sizeof(BIT_TYPE) * bit_len);
    memset(bitmap1_, 0, sizeof(BIT_TYPE) * bit_len);

    // Build the node entries
    for (uint32_t i = 0, j = 0; i < ci->num_conflicts_; ++i) {
      const uint32_t p = ci->positions_[i];
      const uint32_t c = ci->conflicts_[i];

      if (c == 1) {
        if (to_index_seg) {
          // Build leaf layer with segments
          SetEntryType(p, kSegment);
          entries_[p].internal.seg_ = reinterpret_cast<Segment*>(kvs_entry[j]);
          x_ = LeaderMin(x_, entries_[p].internal.seg_->x_);
          x2_ = LeaderMax(x2_, entries_[p].internal.seg_->x2_);
        } else {
          // Build internal layer with nodes
          SetEntryType(p, kNode);
          entries_[p].internal.child_ =
              reinterpret_cast<TNode<KT>*>(kvs_entry[j]);
          x_ = LeaderMin(x_, entries_[p].internal.child_->x_);
          x2_ = LeaderMax(x2_, entries_[p].internal.child_->x2_);
        }
      } else if (c > 1) {
        // Handle conflicts with bucket
        SetEntryType(p, kBucket);
        std::vector<std::pair<KT, Entry<KT>*>> kvs;
        kvs.reserve(c);
        for (uint32_t k = 0; k < c; ++k) {
          kvs.emplace_back(
              std::make_pair(kvs_key[j + k], kvs_entry[j + k]));
        }

        if (to_index_seg) {
          entries_[p].internal.bucket_seg_ = new Bucket<KT, Segment*>(
              reinterpret_cast<std::pair<KT, Segment*>*>(kvs.data()), c, c);
          auto* bucket_seg = entries_[p].internal.bucket_seg_;
          auto* seg =
              reinterpret_cast<Segment*>(bucket_seg->data_[0].second);
          x_ = LeaderMin(x_, seg->x_);
          bucket_seg->x_ = seg->x_;
          seg = reinterpret_cast<Segment*>(
              bucket_seg->data_[bucket_seg->size_ - 1].second);
          x2_ = LeaderMax(x2_, seg->x2_);
          bucket_seg->x2_ = seg->x2_;
        } else {
          entries_[p].internal.bucket_node_ = new Bucket<KT, TNode<KT>*>(
              reinterpret_cast<std::pair<KT, TNode<KT>*>*>(kvs.data()), c, c);
          auto* bucket_node = entries_[p].internal.bucket_node_;
          auto* node =
              reinterpret_cast<TNode<KT>*>(bucket_node->data_[0].second);
          x_ = LeaderMin(x_, node->x_);
          bucket_node->x_ = node->x_;
          node = reinterpret_cast<TNode<KT>*>(
              bucket_node->data_[bucket_node->size_ - 1].second);
          x2_ = LeaderMax(x2_, node->x2_);
          bucket_node->x2_ = node->x2_;
        }
      }
      j = j + c;
    }

    // Record last valid entry for each slot
    if (to_index_seg) {
      Segment* last_valid_seg = nullptr;
      for (int32_t i = 0; i < static_cast<int32_t>(capacity_); ++i) {
        uint8_t entry_type = GetEntryType(i);
        if (entry_type == kNone) {
          entries_[i].last_entry.seg_ = last_valid_seg;
        } else if (entry_type == kSegment) {
          entries_[i].last_entry.seg_ = last_valid_seg;
          last_valid_seg = entries_[i].internal.seg_;
        } else if (entry_type == kBucket) {
          entries_[i].last_entry.seg_ = last_valid_seg;
          auto* bucket_seg = entries_[i].internal.bucket_seg_;
          last_valid_seg = reinterpret_cast<Segment*>(
              bucket_seg->data_[bucket_seg->size_ - 1].second);
        }
      }
    } else {
      TNode<KT>* last_valid_child = nullptr;
      for (int32_t i = 0; i < static_cast<int32_t>(capacity_); ++i) {
        uint8_t entry_type = GetEntryType(i);
        if (entry_type == kNone) {
          entries_[i].last_entry.child_ = last_valid_child;
        } else if (entry_type == kNode) {
          entries_[i].last_entry.child_ = last_valid_child;
          last_valid_child = entries_[i].internal.child_;
        } else if (entry_type == kBucket) {
          entries_[i].last_entry.child_ = last_valid_child;
          auto* bucket_node = entries_[i].internal.bucket_node_;
          last_valid_child = reinterpret_cast<TNode<KT>*>(
              bucket_node->data_[bucket_node->size_ - 1].second);
        }
      }
    }

    delete ci;
  }
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
