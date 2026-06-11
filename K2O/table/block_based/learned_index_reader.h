#pragma once
#include "table/block_based/index_reader_common.h"

namespace ROCKSDB_NAMESPACE {
class LearnedIndexReader : public BlockBasedTable::IndexReaderCommon {
 public:
  // Read index from the file and create an intance for
  // `BinarySearchIndexReader`.
  // On success, index_reader will be populated; otherwise it will remain
  // unmodified.
  static Status Create(const BlockBasedTable* table, const ReadOptions& ro,
                       FilePrefetchBuffer* prefetch_buffer, bool use_cache,
                       bool prefetch, bool pin,
                       BlockCacheLookupContext* lookup_context,
                       std::unique_ptr<IndexReader>* index_reader);

  InternalIteratorBase<IndexValue>* NewIterator(
      const ReadOptions& read_options, bool /* disable_prefix_seek */,
      IndexBlockIter* iter, GetContext* get_context,
      BlockCacheLookupContext* lookup_context) override;

  size_t ApproximateMemoryUsage() const override {
    size_t usage = ApproximateIndexBlockMemoryUsage();
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
    usage += malloc_usable_size(const_cast<LearnedIndexReader*>(this));
#else
    usage += sizeof(*this);
#endif  // ROCKSDB_MALLOC_USABLE_SIZE
    return usage;
  }

 private:
  LearnedIndexReader(const BlockBasedTable* t,
                          CachableEntry<Block>&& index_block)
      : IndexReaderCommon(t, std::move(index_block)) {}
};
}
