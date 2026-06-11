//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Piecewise Linear Regression (PLR) implementation for RocksDB
//  Ported from LeaderKV project

#pragma once

#include <string>
#include <vector>
#include <cmath>

namespace ROCKSDB_NAMESPACE {
namespace leader {

typedef double DATA_TYPE;

// Point structure for PLR computation
struct Point {
  long double x_;
  long double y_;

  Point() = default;
  Point(long double x, long double y) : x_(x), y_(y) {}
};

// Line structure for PLR
struct Line {
  long double a_;  // slope
  long double b_;  // intercept
};

// Segment represents a linear piece with key range [x_, x2_]
class Segment {
 public:
  explicit Segment(double x = 0, DATA_TYPE k = 0, DATA_TYPE b = 0, double x2 = 0)
      : x_(x), k_(k), b_(b), x2_(x2) {}

  double x_;       // start key
  DATA_TYPE k_;    // slope
  DATA_TYPE b_;    // intercept
  double x2_;      // end key
};

// Helper functions for PLR computation
inline long double GetSlope(const Point& p1, const Point& p2) {
  return (p2.y_ - p1.y_) / (p2.x_ - p1.x_);
}

inline Line GetLine(const Point& p1, const Point& p2) {
  long double a = GetSlope(p1, p2);
  long double b = -a * p1.x_ + p1.y_;
  return Line{a, b};
}

inline Point GetIntersection(const Line& l1, const Line& l2) {
  long double a = l1.a_;
  long double b = l2.a_;
  long double c = l1.b_;
  long double d = l2.b_;
  return Point((d - c) / (a - b), (a * d - b * c) / (a - b));
}

inline bool IsAbove(const Point& pt, const Line& l) {
  return pt.y_ > l.a_ * pt.x_ + l.b_;
}

inline bool IsBelow(const Point& pt, const Line& l) {
  return pt.y_ < l.a_ * pt.x_ + l.b_;
}

inline Point GetUpperBound(const Point& pt, double gamma) {
  return Point(pt.x_, pt.y_ + gamma);
}

inline Point GetLowerBound(const Point& pt, double gamma) {
  return Point(pt.x_, pt.y_ - gamma);
}

// Greedy PLR algorithm for constructing segments
class GreedyPLR {
 private:
  uint8_t state_;
  double gamma_;
  Point last_pt_;
  Point s0_;
  Point s1_;
  Line rho_lower_;
  Line rho_upper_;
  Point sint_;

  void Setup() {
    rho_lower_ = GetLine(GetUpperBound(s0_, gamma_), GetLowerBound(s1_, gamma_));
    rho_upper_ = GetLine(GetLowerBound(s0_, gamma_), GetUpperBound(s1_, gamma_));
    sint_ = GetIntersection(rho_upper_, rho_lower_);
  }

  Segment CurrentSegment() const {
    long double segment_start = s0_.x_;
    long double avg_slope = (rho_lower_.a_ + rho_upper_.a_) / 2.0;
    long double intercept = -avg_slope * sint_.x_ + sint_.y_;
    return Segment(segment_start, avg_slope, intercept, last_pt_.x_);
  }

  Segment ProcessInternal(const Point& pt) {
    if (!(IsAbove(pt, rho_lower_) && IsBelow(pt, rho_upper_))) {
      Segment prev_segment = CurrentSegment();
      s0_ = pt;
      state_ = 1;  // need1
      return prev_segment;
    }
    Point s_upper = GetUpperBound(pt, gamma_);
    Point s_lower = GetLowerBound(pt, gamma_);
    if (IsBelow(s_upper, rho_upper_)) {
      rho_upper_ = GetLine(sint_, s_upper);
    }
    if (IsAbove(s_lower, rho_lower_)) {
      rho_lower_ = GetLine(sint_, s_lower);
    }
    return Segment(0, 0, 0, 0);
  }

 public:
  explicit GreedyPLR(double gamma) 
      : state_(2), gamma_(gamma), last_pt_(), s0_(), s1_(), 
        rho_lower_(), rho_upper_(), sint_() {}  // 2 = need2

  Segment Process(const Point& pt) {
    last_pt_ = pt;
    if (state_ == 2) {  // need2
      s0_ = pt;
      state_ = 1;  // need1
    } else if (state_ == 1) {  // need1
      s1_ = pt;
      Setup();
      state_ = 3;  // ready
    } else if (state_ == 3) {  // ready
      return ProcessInternal(pt);
    }
    return Segment(0, 0, 0, 0);
  }

  Segment Finish() {
    if (state_ == 2) {  // need2
      state_ = 4;  // finished
      return Segment(0, 0, 0, 0);
    } else if (state_ == 1) {  // need1
      state_ = 4;  // finished
      return Segment(s0_.x_, 0, s0_.y_, last_pt_.x_);
    } else if (state_ == 3) {  // ready
      state_ = 4;  // finished
      return CurrentSegment();
    }
    return Segment(0, 0, 0, 0);
  }
};

// PLR class for training piecewise linear models on a set of keys
class PLR {
 private:
  double gamma_;
  std::vector<Segment> segments_;

 public:
  explicit PLR(double gamma) : gamma_(gamma) {}

  std::vector<Segment>& Train(const std::vector<double>& keys) {
    GreedyPLR plr_(gamma_);
    size_t size = keys.size();
    segments_.reserve(size + 1);
    for (size_t i = 0; i < size; ++i) {
      Segment seg = plr_.Process(Point(keys[i], i));
      if (seg.x_ != 0 || seg.k_ != 0 || seg.b_ != 0) {
        segments_.emplace_back(seg);
      }
    }
    Segment last = plr_.Finish();
    if (last.x_ != 0 || last.k_ != 0 || last.b_ != 0) {
      segments_.emplace_back(last);
    }
    segments_.shrink_to_fit();
    return segments_;
  }
};

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
