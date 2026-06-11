#include "table/block_based/learned_index_builder.h"

namespace ROCKSDB_NAMESPACE {

void LearnedIndexBuilder::FindShortestInternalKeySeparator(
    const Comparator& comparator, std::string* start, const Slice& limit) {
  // Attempt to shorten the user portion of the key
  Slice user_start = ExtractUserKey(*start);
  Slice user_limit = ExtractUserKey(limit);
  std::string tmp(user_start.data(), user_start.size());
  comparator.FindShortestSeparator(&tmp, user_limit);
  if (tmp.size() <= user_start.size() &&
      comparator.Compare(user_start, tmp) < 0) {
    // User key has become shorter physically, but larger logically.
    // Tack on the earliest possible number to the shortened user key.
    PutFixed64(&tmp,
               PackSequenceAndType(kMaxSequenceNumber, kValueTypeForSeek));
    assert(InternalKeyComparator(&comparator).Compare(*start, tmp) < 0);
    assert(InternalKeyComparator(&comparator).Compare(tmp, limit) < 0);
    start->swap(tmp);
  }
}

void LearnedIndexBuilder::FindShortInternalKeySuccessor(
    const Comparator& comparator, std::string* key) {
  Slice user_key = ExtractUserKey(*key);
  std::string tmp(user_key.data(), user_key.size());
  comparator.FindShortSuccessor(&tmp);
  if (tmp.size() <= user_key.size() && comparator.Compare(user_key, tmp) < 0) {
    // User key has become shorter physically, but larger logically.
    // Tack on the earliest possible number to the shortened user key.
    PutFixed64(&tmp,
               PackSequenceAndType(kMaxSequenceNumber, kValueTypeForSeek));
    assert(InternalKeyComparator(&comparator).Compare(*key, tmp) < 0);
    key->swap(tmp);
  }
}

}
