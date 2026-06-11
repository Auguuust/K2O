//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Learned Index implementation for RocksDB
//  Ported from LeaderKV project

#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "leader_index/node.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Adjustable hyperparameters of learned index
extern uint8_t ERROR_BOUND;
extern uint16_t GROUP_SIZE;
using ERROR = std::pair<int64_t, int64_t>;
inline const ERROR NOT_FOUND = std::make_pair(-1, -1);

// Learned index implementation
template <typename KT>
class LearnedIndex {
 private:
  TNode<KT>* root_;
  HyperParameter hyper_para_;
  std::vector<Segment> segs_;

 public:
  LearnedIndex() : root_(nullptr) {}

  ~LearnedIndex() {
    delete root_;
    root_ = nullptr;
  }

  void BulkLoad(const std::vector<KT>& keys) {
    auto plr_model = PLR(ERROR_BOUND);
    segs_ = std::move(plr_model.Train(keys));
    Segment last_seg(keys.back(), 0, keys.size() - 1, keys.back());
    segs_.emplace_back(last_seg);

    uint32_t num_group =
        static_cast<uint32_t>(ceil(static_cast<double>(segs_.size()) /
                                   GROUP_SIZE));
    std::vector<TNode<KT>*> nodes;
    nodes.reserve(num_group);

    for (uint32_t i = 0; i < num_group; i += 1) {
      std::vector<double> vks_key;
      std::vector<Entry<double>*> vks_entry;
      uint32_t id_j_upper =
          LeaderMin<uint32_t>(GROUP_SIZE * (i + 1), segs_.size());
      vks_key.reserve(id_j_upper - GROUP_SIZE * i);
      vks_entry.reserve(id_j_upper - GROUP_SIZE * i);

      for (uint32_t j = GROUP_SIZE * i; j < id_j_upper; j += 1) {
        vks_key.push_back(segs_[j].x_);
        vks_entry.push_back(reinterpret_cast<Entry<double>*>(&segs_[j]));
      }

      auto node = new TNode<KT>();
      node->Build(vks_key.data(), vks_entry.data(), vks_key.size(),
                  hyper_para_, true);
      nodes.push_back(node);
    }

    std::vector<double> vkt_key;
    std::vector<Entry<double>*> vkt_entry;
    vkt_key.reserve(nodes.size());
    vkt_entry.reserve(nodes.size());

    for (uint32_t i = 0; i < nodes.size(); i += 1) {
      vkt_key.push_back(nodes[i]->x_);
      vkt_entry.push_back(reinterpret_cast<Entry<double>*>(nodes[i]));
    }

    root_ = new TNode<KT>();
    root_->Build(vkt_key.data(), vkt_entry.data(), vkt_key.size(),
                 hyper_para_, false);
  }

  std::pair<int64_t, int64_t> Find(KT key) {
    int32_t idx = root_->model_->Predict(key);
    idx = LeaderMin(LeaderMax(idx, 0),
                    static_cast<int32_t>(root_->capacity_ - 1));
    TNode<KT>* L2_node = nullptr;

    uint8_t type = root_->GetEntryType(idx);
    if (type == kNode) {
      L2_node = root_->entries_[idx].internal.child_;
      if (key < L2_node->x_) {
        L2_node = root_->entries_[idx].last_entry.child_;
      }
    } else if (type == kBucket) {
      auto* bucket_node = root_->entries_[idx].internal.bucket_node_;
      if (bucket_node->x_ <= key && key <= bucket_node->x2_) {
        auto found = bucket_node->Find(key);
        if (found.has_value()) {
          L2_node = found.value();
        }
      } else {
        L2_node = root_->entries_[idx].last_entry.child_;
      }
    }

    if (L2_node == nullptr || key < L2_node->x_ || key > L2_node->x2_) {
      return NOT_FOUND;
    }

    idx = L2_node->model_->Predict(key);
    idx = LeaderMin(LeaderMax(idx, 0),
                    static_cast<int32_t>(L2_node->capacity_ - 1));
    Segment* seg = nullptr;
    type = L2_node->GetEntryType(idx);
    if (type == kSegment) {
      seg = L2_node->entries_[idx].internal.seg_;
      if (key < seg->x_) {
        seg = L2_node->entries_[idx].last_entry.seg_;
      }
    } else if (type == kBucket) {
      auto* bucket_seg = L2_node->entries_[idx].internal.bucket_seg_;
      if (bucket_seg->x_ <= key && key <= bucket_seg->x2_) {
        auto found = bucket_seg->Find(key);
        if (found.has_value()) {
          seg = found.value();
        }
      } else {
        seg = L2_node->entries_[idx].last_entry.seg_;
      }
    }

    if (seg == nullptr || key < seg->x_ || key > seg->x2_) {
      return NOT_FOUND;
    }
    if (std::isnan(seg->k_)) {
      return NOT_FOUND;
    }
    int64_t predicted_pos = static_cast<int>(seg->k_ * key + seg->b_);
    return {predicted_pos - ERROR_BOUND, predicted_pos + ERROR_BOUND};
  }

  uint64_t Serialize(const std::string& model_path,
                     std::vector<uint64_t>& sst_key, uint64_t size) {
    assert(root_ != nullptr);
    if (root_ == nullptr) return 0;

    // Calculate buffer size dynamically:
    // - Model metadata: ~1KB (generous estimate for tree structure)
    // - sst_key array: 8 bytes * sst_key.size()
    // - Add 1MB padding for safety
    const size_t model_overhead = 1024 * 1024;  // 1MB for model structure
    const size_t key_data_size = sst_key.size() * sizeof(uint64_t);
    const size_t buffer_size = model_overhead + key_data_size;
    
    std::vector<char> leader_data_vec(buffer_size);
    char* leader_data = leader_data_vec.data();
    size_t cur_ptr = 0;

    memcpy(leader_data, reinterpret_cast<const char*>(&size), sizeof(size));
    cur_ptr += sizeof(size);
    memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&root_->capacity_),
           sizeof(root_->capacity_));
    cur_ptr += sizeof(root_->capacity_);
    memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&root_->x_),
           sizeof(root_->x_));
    cur_ptr += sizeof(root_->x_);
    memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&root_->x2_),
           sizeof(root_->x2_));
    cur_ptr += sizeof(root_->x2_);
    memcpy(leader_data + cur_ptr,
           reinterpret_cast<const char*>(&root_->model_->slope_),
           sizeof(root_->model_->slope_));
    cur_ptr += sizeof(root_->model_->slope_);
    memcpy(leader_data + cur_ptr,
           reinterpret_cast<const char*>(&root_->model_->intercept_),
           sizeof(root_->model_->intercept_));
    cur_ptr += sizeof(root_->model_->intercept_);

    std::vector<TNode<KT>*> nxt_lvl_nodes;
    for (uint32_t i = 0; i < root_->capacity_; i += 1) {
      uint8_t type = root_->GetEntryType(i);
      // Note: kNone entries can exist due to sparse predictions
      // We serialize them as type=2 so they can be restored on deserialization
      uint32_t tmp_type;
      if (type == kNone) {
        tmp_type = 2;  // Mark as empty slot
        memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tmp_type),
               sizeof(tmp_type));
        cur_ptr += sizeof(tmp_type);
      } else if (type == kNode) {
        nxt_lvl_nodes.push_back(root_->entries_[i].internal.child_);
        tmp_type = 0;
        memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tmp_type),
               sizeof(tmp_type));
        cur_ptr += sizeof(tmp_type);
        memcpy(leader_data + cur_ptr,
               reinterpret_cast<const char*>(&root_->entries_[i].internal.child_->x_),
               sizeof(root_->entries_[i].internal.child_->x_));
        cur_ptr += sizeof(root_->entries_[i].internal.child_->x_);
        memcpy(leader_data + cur_ptr,
               reinterpret_cast<const char*>(&root_->entries_[i].internal.child_->x2_),
               sizeof(root_->entries_[i].internal.child_->x2_));
        cur_ptr += sizeof(root_->entries_[i].internal.child_->x2_);
      } else if (type == kBucket) {
        tmp_type = 1;
        memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tmp_type),
               sizeof(tmp_type));
        cur_ptr += sizeof(tmp_type);
        Bucket<KT, TNode<KT>*>* bucket_node =
            root_->entries_[i].internal.bucket_node_;
        memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&bucket_node->x_),
               sizeof(bucket_node->x_));
        cur_ptr += sizeof(bucket_node->x_);
        memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&bucket_node->x2_),
               sizeof(bucket_node->x2_));
        cur_ptr += sizeof(bucket_node->x2_);
        memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&bucket_node->size_),
               sizeof(bucket_node->size_));
        cur_ptr += sizeof(bucket_node->size_);

        for (uint32_t j = 0; j < bucket_node->size_; j += 1) {
          auto* tnode = reinterpret_cast<TNode<KT>*>(bucket_node->data_[j].second);
          nxt_lvl_nodes.push_back(tnode);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tnode->x_),
                 sizeof(tnode->x_));
          cur_ptr += sizeof(tnode->x_);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tnode->x2_),
                 sizeof(tnode->x2_));
          cur_ptr += sizeof(tnode->x2_);
        }
      }
    }

    for (uint32_t j = 0; j < nxt_lvl_nodes.size(); j += 1) {
      auto tnode = nxt_lvl_nodes[j];
      memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tnode->capacity_),
             sizeof(tnode->capacity_));
      cur_ptr += sizeof(tnode->capacity_);
      memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tnode->x_),
             sizeof(tnode->x_));
      cur_ptr += sizeof(tnode->x_);
      memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tnode->x2_),
             sizeof(tnode->x2_));
      cur_ptr += sizeof(tnode->x2_);
      memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tnode->model_->slope_),
             sizeof(tnode->model_->slope_));
      cur_ptr += sizeof(tnode->model_->slope_);
      memcpy(leader_data + cur_ptr,
             reinterpret_cast<const char*>(&tnode->model_->intercept_),
             sizeof(tnode->model_->intercept_));
      cur_ptr += sizeof(tnode->model_->intercept_);

      for (uint32_t i = 0; i < tnode->capacity_; i += 1) {
        uint8_t type = tnode->GetEntryType(i);
        // Note: kNone entries can exist due to sparse predictions
        uint32_t tmp_type;
        if (type == kNone) {
          tmp_type = 2;  // Mark as empty slot
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tmp_type),
                 sizeof(tmp_type));
          cur_ptr += sizeof(tmp_type);
        } else if (type == kSegment) {
          tmp_type = 0;
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tmp_type),
                 sizeof(tmp_type));
          cur_ptr += sizeof(tmp_type);
          auto seg = tnode->entries_[i].internal.seg_;
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->x_),
                 sizeof(seg->x_));
          cur_ptr += sizeof(seg->x_);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->x2_),
                 sizeof(seg->x2_));
          cur_ptr += sizeof(seg->x2_);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->k_),
                 sizeof(seg->k_));
          cur_ptr += sizeof(seg->k_);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->b_),
                 sizeof(seg->b_));
          cur_ptr += sizeof(seg->b_);
        } else if (type == kBucket) {
          tmp_type = 1;
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&tmp_type),
                 sizeof(tmp_type));
          cur_ptr += sizeof(tmp_type);
          Bucket<KT, Segment*>* bucket_seg =
              tnode->entries_[i].internal.bucket_seg_;
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&bucket_seg->x_),
                 sizeof(bucket_seg->x_));
          cur_ptr += sizeof(bucket_seg->x_);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&bucket_seg->x2_),
                 sizeof(bucket_seg->x2_));
          cur_ptr += sizeof(bucket_seg->x2_);
          memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&bucket_seg->size_),
                 sizeof(bucket_seg->size_));
          cur_ptr += sizeof(bucket_seg->size_);
          for (uint32_t k = 0; k < bucket_seg->size_; k += 1) {
            auto* seg = reinterpret_cast<Segment*>(bucket_seg->data_[k].second);
            memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->x_),
                   sizeof(seg->x_));
            cur_ptr += sizeof(seg->x_);
            memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->x2_),
                   sizeof(seg->x2_));
            cur_ptr += sizeof(seg->x2_);
            memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->k_),
                   sizeof(seg->k_));
            cur_ptr += sizeof(seg->k_);
            memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&seg->b_),
                   sizeof(seg->b_));
            cur_ptr += sizeof(seg->b_);
          }
        }
      }
    }

    uint64_t index_file_size = cur_ptr;

    uint32_t size_sst_key = static_cast<uint32_t>(sst_key.size());
    memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&size_sst_key),
           sizeof(size_sst_key));
    cur_ptr += sizeof(size_sst_key);
    for (uint32_t i = 0; i < size_sst_key; i += 1) {
      memcpy(leader_data + cur_ptr, reinterpret_cast<const char*>(&sst_key[i]),
             sizeof(sst_key[i]));
      cur_ptr += sizeof(sst_key[i]);
    }

    size_t file_size = cur_ptr;
    int fd = open(model_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
      return 0;
    }
    if (ftruncate(fd, file_size) != 0) {
      close(fd);
      return 0;
    }
    char* mmapped = static_cast<char*>(mmap(NULL, file_size,
                                            PROT_READ | PROT_WRITE,
                                            MAP_SHARED, fd, 0));
    close(fd);
    if (mmapped == reinterpret_cast<char*>(MAP_FAILED)) {
      return 0;
    }
    memcpy(mmapped, leader_data, file_size);
    msync(mmapped, file_size, MS_SYNC);
    munmap(mmapped, file_size);
    return index_file_size;
  }

  void DeserializeBuild(const std::string& model_path,
                        std::vector<uint64_t>& sst_key, uint64_t& min_key,
                        uint64_t& max_key, uint64_t& size) {
    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd == -1) {
      return;
    }
    off_t fileSize = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    void* fileMemory = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fileMemory == MAP_FAILED) {
      close(fd);
      return;
    }
    char* data = static_cast<char*>(fileMemory);
    uint64_t offset = 0;
    size = *(reinterpret_cast<uint64_t*>(data));
    offset += sizeof(size);

    root_ = new TNode<KT>();
    root_->model_ = new LinearModel<KT>();
    root_->capacity_ = *(reinterpret_cast<uint32_t*>(data + offset));
    offset += sizeof(root_->capacity_);
    root_->x_ = *(reinterpret_cast<double*>(data + offset));
    offset += sizeof(root_->x_);
    root_->x2_ = *(reinterpret_cast<double*>(data + offset));
    offset += sizeof(root_->x2_);
    root_->model_->slope_ = *(reinterpret_cast<long double*>(data + offset));
    offset += sizeof(root_->model_->slope_);
    root_->model_->intercept_ =
        *(reinterpret_cast<long double*>(data + offset));
    offset += sizeof(root_->model_->intercept_);
    root_->is_bucket_seg_ = false;
    root_->entries_ = new Entry<KT>[root_->capacity_];
    uint32_t num_bitmap = BIT_LEN(root_->capacity_);
    root_->bitmap0_ = new BIT_TYPE[num_bitmap];
    root_->bitmap1_ = new BIT_TYPE[num_bitmap];
    std::memset(root_->bitmap0_, 0, sizeof(BIT_TYPE) * num_bitmap);
    std::memset(root_->bitmap1_, 0, sizeof(BIT_TYPE) * num_bitmap);
    min_key = root_->x_;
    max_key = root_->x2_;

    std::vector<TNode<KT>*> nxt_lvl_nodes;
    TNode<KT>* last_valid_child_node = nullptr;
    for (uint32_t i = 0; i < root_->capacity_; i += 1) {
      uint32_t type = *(reinterpret_cast<uint32_t*>(data + offset));
      offset += sizeof(type);
      if (type == 2) {
        // kNone - empty slot, just record last_entry for fallback lookup
        root_->entries_[i].last_entry.child_ = last_valid_child_node;
        // SetEntryType defaults to kNone (bitmap initialized to 0)
      } else if (type == 0) {
        auto* node = new TNode<KT>();
        node->x_ = *(reinterpret_cast<double*>(data + offset));
        offset += sizeof(node->x_);
        node->x2_ = *(reinterpret_cast<double*>(data + offset));
        offset += sizeof(node->x2_);
        root_->entries_[i].internal.child_ = node;
        root_->entries_[i].last_entry.child_ = last_valid_child_node;
        root_->SetEntryType(i, kNode);
        nxt_lvl_nodes.push_back(node);
        last_valid_child_node = node;
      } else if (type == 1) {
        auto* bucket_node = new Bucket<KT, TNode<KT>*>();
        bucket_node->x_ = *(reinterpret_cast<double*>(data + offset));
        offset += sizeof(bucket_node->x_);
        bucket_node->x2_ = *(reinterpret_cast<double*>(data + offset));
        offset += sizeof(bucket_node->x2_);
        bucket_node->size_ = *(reinterpret_cast<uint32_t*>(data + offset));
        offset += sizeof(bucket_node->size_);
        bucket_node->data_ = new std::pair<KT, TNode<KT>*>[bucket_node->size_];
        root_->entries_[i].last_entry.child_ = last_valid_child_node;
        for (uint32_t j = 0; j < bucket_node->size_; j += 1) {
          auto* node = new TNode<KT>();
          node->x_ = *(reinterpret_cast<double*>(data + offset));
          offset += sizeof(node->x_);
          node->x2_ = *(reinterpret_cast<double*>(data + offset));
          offset += sizeof(node->x2_);
          bucket_node->data_[j].first = node->x_;
          bucket_node->data_[j].second = node;
          nxt_lvl_nodes.push_back(node);
          if (j == bucket_node->size_ - 1) {
            last_valid_child_node = node;
          }
        }
        root_->entries_[i].internal.bucket_node_ = bucket_node;
        root_->SetEntryType(i, kBucket);
      }
    }

    Segment* last_valid_seg = nullptr;
    for (uint32_t i = 0; i < nxt_lvl_nodes.size(); i += 1) {
      auto* node = nxt_lvl_nodes[i];
      node->model_ = new LinearModel<KT>();
      node->capacity_ = *(reinterpret_cast<uint32_t*>(data + offset));
      offset += sizeof(node->capacity_);
      node->x_ = *(reinterpret_cast<double*>(data + offset));
      offset += sizeof(node->x_);
      node->x2_ = *(reinterpret_cast<double*>(data + offset));
      offset += sizeof(node->x2_);
      node->model_->slope_ = *(reinterpret_cast<long double*>(data + offset));
      offset += sizeof(node->model_->slope_);
      node->model_->intercept_ =
          *(reinterpret_cast<long double*>(data + offset));
      offset += sizeof(node->model_->intercept_);
      node->entries_ = new Entry<KT>[node->capacity_];
      num_bitmap = BIT_LEN(node->capacity_);
      node->bitmap0_ = new BIT_TYPE[num_bitmap];
      node->bitmap1_ = new BIT_TYPE[num_bitmap];
      std::memset(node->bitmap0_, 0, sizeof(BIT_TYPE) * num_bitmap);
      std::memset(node->bitmap1_, 0, sizeof(BIT_TYPE) * num_bitmap);

      for (uint32_t j = 0; j < node->capacity_; j += 1) {
        uint32_t type = *(reinterpret_cast<uint32_t*>(data + offset));
        offset += sizeof(type);
        if (type == 2) {
          // kNone - empty slot, just record last_entry for fallback lookup
          node->entries_[j].last_entry.seg_ = last_valid_seg;
          // SetEntryType defaults to kNone (bitmap initialized to 0)
        } else if (type == 0) {
          auto* seg = new Segment();
          seg->x_ = *(reinterpret_cast<double*>(data + offset));
          offset += sizeof(seg->x_);
          seg->x2_ = *(reinterpret_cast<double*>(data + offset));
          offset += sizeof(seg->x2_);
          seg->k_ = *(reinterpret_cast<long double*>(data + offset));
          offset += sizeof(seg->k_);
          seg->b_ = *(reinterpret_cast<long double*>(data + offset));
          offset += sizeof(seg->b_);
          node->entries_[j].internal.seg_ = seg;
          node->entries_[j].last_entry.seg_ = last_valid_seg;
          node->SetEntryType(j, kSegment);
          last_valid_seg = seg;
        } else if (type == 1) {
          auto* bucket_seg = new Bucket<KT, Segment*>();
          bucket_seg->x_ = *(reinterpret_cast<double*>(data + offset));
          offset += sizeof(bucket_seg->x_);
          bucket_seg->x2_ = *(reinterpret_cast<double*>(data + offset));
          offset += sizeof(bucket_seg->x2_);
          bucket_seg->size_ = *(reinterpret_cast<uint32_t*>(data + offset));
          offset += sizeof(bucket_seg->size_);
          bucket_seg->data_ =
              new std::pair<KT, Segment*>[bucket_seg->size_];
          node->entries_[j].last_entry.seg_ = last_valid_seg;
          for (uint32_t k = 0; k < bucket_seg->size_; k += 1) {
            auto* seg = new Segment();
            seg->x_ = *(reinterpret_cast<double*>(data + offset));
            offset += sizeof(seg->x_);
            seg->x2_ = *(reinterpret_cast<double*>(data + offset));
            offset += sizeof(seg->x2_);
            seg->k_ = *(reinterpret_cast<long double*>(data + offset));
            offset += sizeof(seg->k_);
            seg->b_ = *(reinterpret_cast<long double*>(data + offset));
            offset += sizeof(seg->b_);
            bucket_seg->data_[k].first = seg->x_;
            bucket_seg->data_[k].second = seg;
            if (k == bucket_seg->size_ - 1) {
              last_valid_seg = seg;
            }
          }
          node->entries_[j].internal.bucket_seg_ = bucket_seg;
          node->SetEntryType(j, kBucket);
        }
      }
    }

    uint32_t sst_key_size = *(reinterpret_cast<uint32_t*>(data + offset));
    offset += sizeof(sst_key_size);
    sst_key.resize(sst_key_size);
    for (uint32_t i = 0; i < sst_key_size; i += 1) {
      sst_key[i] = *(reinterpret_cast<uint64_t*>(data + offset));
      offset += sizeof(sst_key[i]);
    }

    munmap(fileMemory, fileSize);
    close(fd);
  }

  uint64_t IndexSize() const {
    if (root_ == nullptr) return 0;
    uint64_t index_size = 0;
    std::vector<TNode<KT>*> nxt_lvl_nodes;

    index_size += sizeof(TNode<KT>) + sizeof(LinearModel<KT>) +
                  sizeof(BIT_TYPE) * 2 * BIT_LEN(root_->capacity_) +
                  sizeof(Entry<KT>) * root_->capacity_;

    for (uint32_t i = 0; i < root_->capacity_; i += 1) {
      uint8_t type = root_->GetEntryType(i);
      if (type == kNode) {
        nxt_lvl_nodes.push_back(root_->entries_[i].internal.child_);
      } else if (type == kBucket) {
        Bucket<KT, TNode<KT>*>* bucket_node =
            root_->entries_[i].internal.bucket_node_;
        index_size += sizeof(Bucket<KT, TNode<KT>*>);
        for (uint32_t j = 0; j < bucket_node->size_; j += 1) {
          auto* tnode = reinterpret_cast<TNode<KT>*>(bucket_node->data_[j].second);
          nxt_lvl_nodes.push_back(tnode);
        }
      }
    }

    for (uint32_t j = 0; j < nxt_lvl_nodes.size(); j += 1) {
      auto tnode = nxt_lvl_nodes[j];
      index_size += sizeof(TNode<KT>) + sizeof(LinearModel<KT>) +
                    sizeof(BIT_TYPE) * 2 * BIT_LEN(tnode->capacity_) +
                    sizeof(Entry<KT>) * tnode->capacity_;
      for (uint32_t i = 0; i < tnode->capacity_; i += 1) {
        uint8_t type = tnode->GetEntryType(i);
        if (type == kSegment) {
          index_size += sizeof(Segment);
        } else if (type == kBucket) {
          Bucket<KT, Segment*>* bucket_seg =
              tnode->entries_[i].internal.bucket_seg_;
          index_size += sizeof(Bucket<KT, Segment*>);
          for (uint32_t k = 0; k < bucket_seg->size_; k += 1) {
            index_size += sizeof(Segment);
          }
        }
      }
    }
    return index_size;
  }
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
