//
// Created by daiyi on 2020/02/02.
//

#include <util/mutexlock.h>
#include "util.h"
#include <x86intrin.h>
#include <fstream>

using std::to_string;

namespace pirMod {

uint32_t key_size = 20;
uint32_t value_size = 64;
rocksdb::Env* env;
FileLearnedIndexData* file_data = nullptr;
std::atomic<bool> learned_index_enabled = false;

uint64_t ExtractInteger(const char* pos, size_t size) {
  char* temp = new char[size + 1];
  memcpy(temp, pos, size);
  temp[size] = '\0';
  uint64_t result = (uint64_t) atol(temp);
  delete[] temp;
  return result;
}

string generate_key(const string& key) {
  int num_placehold = key_size - key.length();
  if (num_placehold < 0) {
    string result = key.substr(-num_placehold);
    return result;
  }
  string result = string(num_placehold, '0') + key;
  return result;
}

string generate_value(uint64_t value) {
  string value_string = to_string(value);
  int num_placehold = value_size - value_string.length();
  if (num_placehold < 0) {
    string result = value_string.substr(-num_placehold);
    return result;
  }
  string result = string(num_placehold, '0') + value_string;
  return result;
}

uint64_t SliceToInteger(const Slice& key) {
  uint64_t x = 0;
  size_t n = std::min<size_t>(8, key.size());
  for (size_t i = 0; i < n; i++) {
    x = (x << 8) | (uint8_t)key[i];
  }
  // 如果 key < 8 字节，左对齐
  x <<= (8 - n) * 8;
  return x;
}

int compare(const Slice& slice, const string& string) {
  return memcmp((void*) slice.data(), string.c_str(), slice.size());
}

bool operator<(const Slice& slice, const string& string) {
  return memcmp((void*) slice.data(), string.c_str(), slice.size()) < 0;
}

bool operator>(const Slice& slice, const string& string) {
  return memcmp((void*) slice.data(), string.c_str(), slice.size()) > 0;
}

bool operator<=(const Slice& slice, const string& string) {
  return memcmp((void*) slice.data(), string.c_str(), slice.size()) <= 0;
}

bool operator>=(const Slice& slice, const string& string) {
  return memcmp((void*) slice.data(), string.c_str(), slice.size()) >= 0;
}

uint64_t get_time_difference(timespec start, timespec stop) {
  return (stop.tv_sec - start.tv_sec) * 1000000000 + stop.tv_nsec - start.tv_nsec;
}

}
