#include "table/block_based/learned_index_reader.h"

namespace ROCKSDB_NAMESPACE {
Status LearnedIndexReader::Create(
    const BlockBasedTable* table, const ReadOptions& ro,
    FilePrefetchBuffer* prefetch_buffer, bool use_cache, bool prefetch,
    bool pin, BlockCacheLookupContext* lookup_context,
    std::unique_ptr<IndexReader>* index_reader) {
  assert(table != nullptr);
  assert(table->get_rep());
  assert(!pin || prefetch);
  assert(index_reader != nullptr);

  CachableEntry<Block> index_block;
  if (prefetch || !use_cache) {
    const Status s =
        ReadIndexBlock(table, prefetch_buffer, ro, use_cache,
                       /*get_context=*/nullptr, lookup_context, &index_block);
    if (!s.ok()) {
      return s;
    }

    if (use_cache && !pin) {
      index_block.Reset();
    }
  }

  index_reader->reset(
      new LearnedIndexReader(table, std::move(index_block)));

  return Status::OK();
}

InternalIteratorBase<IndexValue>* LearnedIndexReader::NewIterator(
    const ReadOptions& read_options, bool /* disable_prefix_seek */,
    IndexBlockIter* iter, GetContext* get_context,
    BlockCacheLookupContext* lookup_context) {
  const BlockBasedTable::Rep* rep = table()->get_rep();
  const bool no_io = (read_options.read_tier == kBlockCacheTier);
  CachableEntry<Block> index_block;
  const Status s = GetOrReadIndexBlock(no_io, get_context, lookup_context,
                                       &index_block, read_options);
  if (!s.ok()) {
    if (iter != nullptr) {
      iter->Invalidate(s);
      return iter;
    }

    return NewErrorInternalIterator<IndexValue>(s);
  }

  Statistics* kNullStats = nullptr;
  // We don't return pinned data from index blocks, so no need
  // to set `block_contents_pinned`.
  auto it = index_block.GetValue()->NewIndexIterator(
      internal_comparator()->user_comparator(),
      rep->get_global_seqno(BlockType::kIndex), iter, kNullStats, true,
      index_has_first_key(), index_key_includes_seq(), index_value_is_full(),
      false /* block_contents_pinned */, user_defined_timestamps_persisted());

  assert(it != nullptr);
  index_block.TransferTo(it);

  return it;
}
}
