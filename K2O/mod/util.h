//
// Created by daiyi on 2020/02/02.
//

#ifndef LEVELDB_UTIL_H
#define LEVELDB_UTIL_H

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "db/db_impl/db_impl.h"
#include "rocksdb/slice.h"
#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include <x86intrin.h>

using std::string;
using std::vector;
using std::map;
using rocksdb::Slice;

namespace pirMod {

class FileLearnedIndexData;
class LearnedIndexData;
class FileStats;

extern rocksdb::Env* env;
extern FileLearnedIndexData* file_data;
extern std::atomic<bool> learned_index_enabled;

uint64_t ExtractInteger(const char* pos, size_t size);
string generate_key(const string& key);
string generate_value(uint64_t value);
uint64_t SliceToInteger(const Slice& slice);
int compare(const Slice& slice, const string& string);
bool operator<(const Slice& slice, const string& string);
bool operator>(const Slice& slice, const string& string);
bool operator<=(const Slice& slice, const string& string);
bool operator>=(const Slice& slice, const string& string);
uint64_t get_time_difference(timespec start, timespec stop);

}


#endif //LEVELDB_UTIL_H
