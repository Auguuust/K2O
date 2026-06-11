//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Linear Model implementation for RocksDB
//  Ported from LeaderKV project

#pragma once

#include <cstdint>
#include <climits>
#include <algorithm>
#include <cmath>

namespace ROCKSDB_NAMESPACE {
namespace leader {

typedef double DATA_TYPE;

// Linear model for prediction: y = slope * x + intercept
template <class KT>
class LinearModel {
 public:
  DATA_TYPE slope_;
  DATA_TYPE intercept_;

  LinearModel() : slope_(0), intercept_(0) {}

  inline int64_t Predict(KT key) const {
    return static_cast<int64_t>(floor(slope_ * key + intercept_));
  }
};

// Builder class for constructing LinearModel through least squares regression
template <class KT>
class LinearModelBuilder {
 public:
  int count_;
  double x_sum_;
  double y_sum_;
  double xx_sum_;
  double xy_sum_;
  KT x_min_;
  KT x_max_;
  double y_min_;
  double y_max_;

  LinearModelBuilder()
      : count_(0),
        x_sum_(0),
        y_sum_(0),
        xx_sum_(0),
        xy_sum_(0),
        x_min_(std::numeric_limits<KT>::max()),
        x_max_(std::numeric_limits<KT>::lowest()),
        y_min_(std::numeric_limits<double>::max()),
        y_max_(std::numeric_limits<double>::lowest()) {}

  inline void Add(KT x, double y) {
    count_ += 1;
    x_sum_ += static_cast<double>(x);
    y_sum_ += static_cast<double>(y);
    xx_sum_ += static_cast<double>(x) * x;
    xy_sum_ += static_cast<double>(x) * y;
    x_min_ = std::min(x, x_min_);
    x_max_ = std::max(x, x_max_);
    y_min_ = std::min(y, y_min_);
    y_max_ = std::max(y, y_max_);
  }

  void Build(LinearModel<KT>* lrm) {
    if (count_ <= 1) {
      lrm->slope_ = 0;
      lrm->intercept_ = static_cast<long double>(y_sum_);
      return;
    }
    if (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_ == 0) {
      lrm->slope_ = 0;
      lrm->intercept_ = static_cast<long double>(y_sum_) / count_;
      return;
    }
    auto slope = static_cast<long double>(
        (static_cast<long double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
        (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_));
    auto intercept = static_cast<long double>(
        (y_sum_ - static_cast<long double>(slope) * x_sum_) / count_);
    lrm->slope_ = slope;
    lrm->intercept_ = intercept;
  }
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
