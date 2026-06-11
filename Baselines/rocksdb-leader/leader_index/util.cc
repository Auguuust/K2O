//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  LeaderKV utility implementations

#include "leader_index/util.h"
#include <utility>

namespace ROCKSDB_NAMESPACE {
namespace leader {

// Learned index configuration
bool learned_index_enabled = false;
uint32_t model_error = 32;
uint64_t block_num_entries = 0;
uint64_t block_size = 4096;
uint64_t entry_size = 0;
int internal_key_size = 0;
int internal_addr_info_size = 0;

// LearnedIndex global parameters
uint8_t ERROR_BOUND = 32;
uint16_t GROUP_SIZE = 16;

}  // namespace leader
}  // namespace ROCKSDB_NAMESPACE
