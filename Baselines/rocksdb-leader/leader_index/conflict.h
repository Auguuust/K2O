//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Conflicts Info for building linear model
//  Ported from LeaderKV project

#pragma once

#include <cstdint>
#include <algorithm>
#include "leader_index/linear_model.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Structure to track conflicts when building learned index
struct ConflictsInfo {
  uint32_t* conflicts_;
  uint32_t* positions_;
  uint32_t num_conflicts_;
  uint32_t max_size_;
  double min_key_;
  double max_key_;

  ConflictsInfo(uint32_t size, uint32_t max_size)
      : num_conflicts_(0),
        max_size_(max_size),
        min_key_(1e30),
        max_key_(0) {
    conflicts_ = new uint32_t[size];
    positions_ = new uint32_t[size];
  }

  ~ConflictsInfo() {
    if (conflicts_ != nullptr) {
      delete[] conflicts_;
      conflicts_ = nullptr;
    }
    if (positions_ != nullptr) {
      delete[] positions_;
      positions_ = nullptr;
    }
  }

  void AddConflict(uint32_t position, uint32_t conflict) {
    positions_[num_conflicts_] = position;
    conflicts_[num_conflicts_] = conflict;
    num_conflicts_++;
  }
};

// Build a linear model for the given keys and return conflict information
// Returns nullptr if slope equals zero
template <typename KT>
ConflictsInfo* BuildLinearModel(const KT* kvs, uint32_t size,
                                LinearModel<KT>*& model, double size_amp) {
  if (model != nullptr) {
    model->slope_ = model->intercept_ = 0;
  } else {
    delete model;
    model = new LinearModel<KT>();
  }

  // Find a linear regression model that minimizes conflicts
  const KT min_key = kvs[0];
  const KT max_key = kvs[size - 1];
  auto max_size = static_cast<uint32_t>(size * size_amp);

  LinearModelBuilder<KT> builder;
  for (uint32_t i = 0; i < size; ++i) {
    builder.Add(kvs[i], static_cast<double>(i));
  }
  builder.Build(model);
  model->intercept_ = -model->slope_ * (min_key) + 0.5;

  const int64_t predicted_size = model->Predict(max_key) + 1;
  if (predicted_size > 1) {
    max_size = std::min(predicted_size, static_cast<int64_t>(max_size));
  }

  const uint32_t first_pos = std::min(
      std::max(model->Predict(min_key), static_cast<int64_t>(0)),
      static_cast<int64_t>(max_size - 1));
  const uint32_t last_pos = std::min(
      std::max(model->Predict(max_key), static_cast<int64_t>(0)),
      static_cast<int64_t>(max_size - 1));

  if (last_pos == first_pos) {
    if (size == 1) {
      model->slope_ = 0;
      model->intercept_ = 0;
    } else {
      // Model fails to predict since all positions are the same
      model->slope_ = size / (max_key - min_key);
      model->intercept_ = -model->slope_ * (min_key) + 0.5;
    }
  }

  auto* ci = new ConflictsInfo(size, max_size);
  ci->min_key_ = min_key;
  ci->max_key_ = max_key;

  uint32_t p_last = first_pos;
  uint32_t conflict = 1;

  for (uint32_t i = 1; i < size; ++i) {
    const uint32_t p = std::min(
        std::max(model->Predict(kvs[i]), static_cast<int64_t>(0)),
        static_cast<int64_t>(max_size - 1));
    if (p == p_last) {
      conflict++;
    } else {
      ci->AddConflict(p_last, conflict);
      p_last = p;
      conflict = 1;
    }
  }
  ci->AddConflict(p_last, conflict);

  return ci;
}

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
