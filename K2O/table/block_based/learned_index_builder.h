#pragma once

#include "table/block_based/index_builder.h"

namespace ROCKSDB_NAMESPACE {
    class LearnedIndexBuilder : public IndexBuilder {
 public:
  LearnedIndexBuilder(
      const InternalKeyComparator* comparator,
      const int index_block_restart_interval, const uint32_t format_version,
      const bool use_value_delta_encoding,
      BlockBasedTableOptions::IndexShorteningMode shortening_mode,
      bool include_first_key, size_t ts_sz,
      const bool persist_user_defined_timestamps)
      : IndexBuilder(comparator, ts_sz, persist_user_defined_timestamps),
        index_block_builder_(
            index_block_restart_interval, true /*use_delta_encoding*/,
            use_value_delta_encoding,
            BlockBasedTableOptions::kDataBlockBinarySearch /* index_type */,
            0.75 /* data_block_hash_table_util_ratio */, ts_sz,
            persist_user_defined_timestamps, false /* is_user_key */),
        index_block_builder_without_seq_(
            index_block_restart_interval, true /*use_delta_encoding*/,
            use_value_delta_encoding,
            BlockBasedTableOptions::kDataBlockBinarySearch /* index_type */,
            0.75 /* data_block_hash_table_util_ratio */, ts_sz,
            persist_user_defined_timestamps, true /* is_user_key */),
        use_value_delta_encoding_(use_value_delta_encoding),
        include_first_key_(include_first_key),
        shortening_mode_(shortening_mode) {
    // Making the default true will disable the feature for old versions
    seperator_is_key_plus_seq_ = (format_version <= 2);
  }

  void OnKeyAdded(const Slice& key) override {
    if (include_first_key_ && current_block_first_internal_key_.empty()) {
      current_block_first_internal_key_.assign(key.data(), key.size());
    }
  }

  void AddIndexEntry(std::string* last_key_in_current_block,
                     const Slice* first_key_in_next_block,
                     const BlockHandle& block_handle) override {
    if (first_key_in_next_block != nullptr) {
      if (shortening_mode_ !=
          BlockBasedTableOptions::IndexShorteningMode::kNoShortening) {
        FindShortestInternalKeySeparator(*comparator_->user_comparator(),
                                         last_key_in_current_block,
                                         *first_key_in_next_block);
      }
      if (!seperator_is_key_plus_seq_ &&
          ShouldUseKeyPlusSeqAsSeparator(*last_key_in_current_block,
                                         *first_key_in_next_block)) {
        seperator_is_key_plus_seq_ = true;
      }
    } else {
      if (shortening_mode_ == BlockBasedTableOptions::IndexShorteningMode::
                                  kShortenSeparatorsAndSuccessor) {
        FindShortInternalKeySuccessor(*comparator_->user_comparator(),
                                      last_key_in_current_block);
      }
    }
    auto sep = Slice(*last_key_in_current_block);

    assert(!include_first_key_ || !current_block_first_internal_key_.empty());
    // When UDT should not be persisted, the index block builders take care of
    // stripping UDT from the key, for the first internal key contained in the
    // IndexValue, we need to explicitly do the stripping here before passing
    // it to the block builders.
    std::string first_internal_key_buf;
    Slice first_internal_key = current_block_first_internal_key_;
    if (!current_block_first_internal_key_.empty() && ts_sz_ > 0 &&
        !persist_user_defined_timestamps_) {
      StripTimestampFromInternalKey(&first_internal_key_buf,
                                    current_block_first_internal_key_, ts_sz_);
      first_internal_key = first_internal_key_buf;
    }
    IndexValue entry(block_handle, first_internal_key);
    std::string encoded_entry;
    std::string delta_encoded_entry;
    entry.EncodeTo(&encoded_entry, include_first_key_, nullptr);
    if (use_value_delta_encoding_ && !last_encoded_handle_.IsNull()) {
      entry.EncodeTo(&delta_encoded_entry, include_first_key_,
                     &last_encoded_handle_);
    } else {
      // If it's the first block, or delta encoding is disabled,
      // BlockBuilder::Add() below won't use delta-encoded slice.
    }
    last_encoded_handle_ = block_handle;
    const Slice delta_encoded_entry_slice(delta_encoded_entry);

    // TODO(yuzhangyu): fix this when "FindShortInternalKeySuccessor"
    //  optimization is available.
    // Timestamp aware comparator currently doesn't provide override for
    // "FindShortInternalKeySuccessor" optimization. So the actual
    // last key in current block is used as the key for indexing the current
    // block. As a result, when UDTs should not be persisted, it's safe to strip
    // away the UDT from key in index block as data block does the same thing.
    // What are the implications if a "FindShortInternalKeySuccessor"
    // optimization is provided.
    index_block_builder_.Add(sep, encoded_entry, &delta_encoded_entry_slice);
    if (!seperator_is_key_plus_seq_) {
      index_block_builder_without_seq_.Add(ExtractUserKey(sep), encoded_entry,
                                           &delta_encoded_entry_slice);
    }

    current_block_first_internal_key_.clear();
  }

  using IndexBuilder::Finish;
  Status Finish(IndexBlocks* index_blocks,
                const BlockHandle& /*last_partition_block_handle*/) override {
    if (seperator_is_key_plus_seq_) {
      index_blocks->index_block_contents = index_block_builder_.Finish();
    } else {
      index_blocks->index_block_contents =
          index_block_builder_without_seq_.Finish();
    }
    index_size_ = index_blocks->index_block_contents.size();
    return Status::OK();
  }

  size_t IndexSize() const override { return index_size_; }

  bool seperator_is_key_plus_seq() override {
    return seperator_is_key_plus_seq_;
  }

  // Changes *key to a short string >= *key.
  //
  static void FindShortestInternalKeySeparator(const Comparator& comparator,
                                               std::string* start,
                                               const Slice& limit);

  static void FindShortInternalKeySuccessor(const Comparator& comparator,
                                            std::string* key);

  friend class PartitionedIndexBuilder;

 private:
  BlockBuilder index_block_builder_;
  BlockBuilder index_block_builder_without_seq_;
  const bool use_value_delta_encoding_;
  bool seperator_is_key_plus_seq_;
  const bool include_first_key_;
  BlockBasedTableOptions::IndexShorteningMode shortening_mode_;
  BlockHandle last_encoded_handle_ = BlockHandle::NullBlockHandle();
  std::string current_block_first_internal_key_;
};

}
