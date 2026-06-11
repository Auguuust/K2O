//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// LearnedBlockBasedTableBuilder implements Google's approach:
// Use learned index to guide block boundary decisions

#pragma once

#include "table/block_based/block_based_table_builder.h"
#include "table/block_based/learned_index.h"
#include "table/block_based/learned_index_manager.h"
#include "rocksdb/env.h"
#include "file/writable_file_writer.h"
#include <memory>

namespace ROCKSDB_NAMESPACE {

// LearnedBlockBasedTableBuilder extends BlockBasedTableBuilder
// to support learned-index-guided block construction
class LearnedBlockBasedTableBuilder : public BlockBasedTableBuilder {
 public:
  LearnedBlockBasedTableBuilder(
      const BlockBasedTableOptions& table_options,
      const TableBuilderOptions& table_builder_options,
      WritableFileWriter* file,
      bool enable_learned_index = true);

  ~LearnedBlockBasedTableBuilder() override;

  // Override Add to collect training data
  void Add(const Slice& key, const Slice& value) override;

  // Override Finish to train and write learned index
  Status Finish() override;
  
  // Override Abandon to handle early termination
  // If table is being abandoned, we still need to complete the learned index workflow
  void Abandon() override;
  
  // Override IsEmpty to account for buffered data during collection phase
  bool IsEmpty() const override;
  
  // Override NumEntries to account for buffered data during collection phase
  uint64_t NumEntries() const override;

  // Get learned index statistics
  struct LearnedIndexStats {
    bool enabled;
    bool trained;
    uint32_t num_training_samples;
    uint32_t predicted_num_blocks;
    double model_slope;
    double model_intercept;
  };
  LearnedIndexStats GetLearnedIndexStats() const;

 private:
  // Two-pass approach for learned index:
  // Pass 1: Buffer all data and collect training info
  // Pass 2: Train model, then replay data with model-guided block boundaries
  
  struct BufferedEntry {
    std::string internal_key;  // Full internal key (user_key + sequence + type)
    std::string value;
    size_t entry_index;  // Track order for debugging
    
    BufferedEntry(const Slice& k, const Slice& v, size_t idx = 0) 
        : internal_key(k.data(), k.size()), value(v.data(), v.size()), entry_index(idx) {}
  };
  
  struct TrainingBuffer {
    uint64_t cumulative_bytes;
    uint64_t num_entries;
    uint32_t max_key_length;
    std::vector<uint8_t> char_min;
    std::vector<uint8_t> char_max;
    bool collection_phase;
    
    TrainingBuffer()
        : cumulative_bytes(0),
          num_entries(0),
          max_key_length(0),
          collection_phase(true) {}
  };

  bool enable_learned_index_;
  bool finish_called_ = false;  // Debug: track if Finish() was called
  bool sentinel_written_ = false;  // Track if sentinel entry was written
  std::unique_ptr<TrainingBuffer> training_buffer_;
  std::unique_ptr<LearnedIndex> learned_index_;
  std::unique_ptr<BlockLocationMapper> block_mapper_;
  
  uint32_t current_predicted_block_;  // The logical block_id from model prediction
  uint32_t last_flushed_block_id_ = UINT32_MAX;
  uint32_t physical_block_count_ = 0; // Actual number of physical blocks written
  uint64_t current_block_bytes_;
  uint64_t entries_per_block_ = 32;   // Entries per block for write-time decisions
  uint64_t current_entry_index_ = 0;  // Current entry index during replay
  std::string last_user_key_;
  
  // Store file info for saving learned index
  std::string sst_file_path_;
  uint64_t file_number_;
  uint64_t original_block_size_;      // Original block size before modification
  uint64_t target_block_size_;
  uint32_t min_training_keys_;
  uint64_t segment_keys_;
  uint64_t li_build_begin_cycles_ = 0;

  Env* env_ = nullptr;
  std::string temp_kv_path_;
  std::unique_ptr<WritableFileWriter> temp_kv_writer_;
  
  // Helper methods
  void CollectTrainingData(const Slice& key, const Slice& value);
  Status TrainLearnedIndex();
  Status CloseTempKVWriter();
  Status InitTempKVWriterIfNeeded();
  Status WriteTempKV(const Slice& key, const Slice& value);
  Status ReplayTempKV(bool use_learned_index);
  Status ScanTempKVForStats(const KeyEncoder& encoder,
                            LearnedIndex::RegressionStats* stats,
                            uint64_t* total_bytes,
                            std::vector<LearnedIndex::RegressionStats>*
                                segment_stats,
                            std::vector<double>* segment_max_encoded_keys);
  bool ShouldFlushBlock(const Slice& key, const Slice& value);
  void WriteEmptyBlocksUntil(uint32_t target_block_id);  // Write empty placeholder blocks
  Status WriteLearnedIndexMetadata();
  
 protected:
  // Override to track block locations for learned index
  void OnDataBlockWritten(uint64_t offset, uint32_t size) override;
  
  // Override flush decision hook - completely control block boundaries
  // during Pass 2 replay based on learned index predictions
  bool ShouldFlushBeforeAdd(const Slice& key, const Slice& value,
                            bool policy_says_flush) override;
};

}  // namespace ROCKSDB_NAMESPACE
