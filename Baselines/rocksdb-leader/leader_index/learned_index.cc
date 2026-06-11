//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV Learned Index constants

#include "leader_index/learned_index.h"

namespace ROCKSDB_NAMESPACE {
namespace leader {

uint8_t ERROR_BOUND = 32;
uint16_t GROUP_SIZE = 16;
// NOT_FOUND is defined as inline const in learned_index.h

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
