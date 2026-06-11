//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/block_based/learned_block_based_table_builder.h"
#include "db/dbformat.h"
#include "file/writable_file_writer.h"
#include "util/coding.h"
#include "rocksdb/file_system.h"
#include "file/filename.h"
#include "my_trace.h"
#include <algorithm>

namespace ROCKSDB_NAMESPACE {

LearnedBlockBasedTableBuilder::LearnedBlockBasedTableBuilder(
    const BlockBasedTableOptions& table_options,
    const TableBuilderOptions& table_builder_options,
    WritableFileWriter* file,
    bool enable_learned_index)
    : BlockBasedTableBuilder(
          table_options, table_builder_options, file),
      enable_learned_index_(enable_learned_index),
      sentinel_written_(false),
      current_predicted_block_(0),
      current_block_bytes_(0),
      file_number_(table_builder_options.cur_file_num),
      original_block_size_(table_options.block_size),
      target_block_size_(table_options.learned_index_target_block_size == 0
                             ? table_options.block_size
                             : table_options.learned_index_target_block_size),
      min_training_keys_(table_options.learned_index_min_training_keys),
      segment_keys_(table_options.learned_index_segment_keys),
      env_(table_builder_options.ioptions.env) {
  
  if (enable_learned_index_) {
    li_build_begin_cycles_ = rdtscp();
    training_buffer_ = std::make_unique<TrainingBuffer>();
    learned_index_ = std::make_unique<LearnedIndex>();
    block_mapper_ = std::make_unique<BlockLocationMapper>();
    
    // Get the SST file path from WritableFileWriter
    if (file) {
      sst_file_path_ = file->file_name();
    }

    // If file name is empty, derive from db path + file number
    if (sst_file_path_.empty()) {
      if (!table_builder_options.ioptions.db_paths.empty()) {
        const std::string& db_path =
            table_builder_options.ioptions.db_paths[0].path;
        sst_file_path_ = MakeTableFileName(db_path, file_number_);
      }
    }

    // Initialize temp KV file for streaming storage (avoid buffering all data)
    InitTempKVWriterIfNeeded().PermitUncheckedError();
  }
}

LearnedBlockBasedTableBuilder::~LearnedBlockBasedTableBuilder() = default;

void LearnedBlockBasedTableBuilder::Abandon() {
  // fprintf(stderr, "[LearnedIndex] Abandon() called!\n");
  fflush(stderr);
  
  if (!enable_learned_index_ || !training_buffer_) {
    BlockBasedTableBuilder::Abandon();
    return;
  }
  
  // Cleanup temp file if any
  CloseTempKVWriter().PermitUncheckedError();
  if (env_ && !temp_kv_path_.empty()) {
    env_->DeleteFile(temp_kv_path_).PermitUncheckedError();
  }

  // Default abandon behavior
  BlockBasedTableBuilder::Abandon();
}

void LearnedBlockBasedTableBuilder::Add(const Slice& key, const Slice& value) {
  if (!enable_learned_index_) {
    // Fall back to standard behavior
    BlockBasedTableBuilder::Add(key, value);
    return;
  }

  if (training_buffer_->collection_phase) {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Pass 1: ONLY buffer data, do NOT write yet
    // We'll write everything in Pass 2 with model guidance
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    size_t entry_count = training_buffer_->num_entries;
    if (entry_count < 5) {
      // fprintf(stderr, "[LearnedIndex] Pass 1: Buffering entry %zu\n", entry_count + 1);
      fflush(stderr);
    }
    
    // Ensure temp writer initialized
    Status s = InitTempKVWriterIfNeeded();
    if (!s.ok()) {
      LearnedIndexManager::GetInstance().RecordTrainingFailure();
      enable_learned_index_ = false;
      BlockBasedTableBuilder::Add(key, value);
      return;
    }

    // Stream key-value to temp file for replay in Pass 2
    s = WriteTempKV(key, value);
    if (!s.ok()) {
      // Fallback to standard mode if temp file write fails
      CloseTempKVWriter().PermitUncheckedError();
      if (env_ && !temp_kv_path_.empty()) {
        env_->DeleteFile(temp_kv_path_).PermitUncheckedError();
      }
      LearnedIndexManager::GetInstance().RecordTrainingFailure();
      enable_learned_index_ = false;
      BlockBasedTableBuilder::Add(key, value);
      return;
    }

    // Collect training stats for model
    CollectTrainingData(key, value);

    // NOTE: We don't write in Pass 1.
  } else {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Pass 2: Write data guided by learned index
    // The ShouldFlushBeforeAdd() hook is called by parent's Add(),
    // which delegates to our ShouldFlushBlock() for learned index decisions.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    static std::atomic<uint64_t> pass2_count{0};
    // if (++pass2_count <= 5) {
    //   // fprintf(stderr, "[LI-WRITE] Pass 2 entry #%lu\n", pass2_count.load());
    // }
    
    // Simply call parent Add - it will use our ShouldFlushBeforeAdd() hook
    // to determine block boundaries based on learned index predictions
    BlockBasedTableBuilder::Add(key, value);
  }
}

void LearnedBlockBasedTableBuilder::CollectTrainingData(
    const Slice& key, const Slice& value) {
  // Train on user key so read path can predict using user key.
  Slice user_key = ExtractUserKey(key);

  // Calculate entry size matching BlockBuilder format (approximate):
  // varint(shared_bytes) + varint(unshared_bytes) + varint(value_len) + key + value
  // Approximate with 5 bytes overhead per entry.
  size_t entry_size = key.size() + value.size() + 5;
  training_buffer_->cumulative_bytes += entry_size;

  training_buffer_->num_entries++;

  // Update key stats for encoder (min/max per byte position)
  uint32_t key_len = static_cast<uint32_t>(user_key.size());
  if (key_len > training_buffer_->max_key_length) {
    uint32_t old_len = training_buffer_->max_key_length;
    training_buffer_->char_min.resize(key_len, 255);
    training_buffer_->char_max.resize(key_len, 0);
    // For new positions, earlier keys are shorter, so min should be 0.
    for (uint32_t i = old_len; i < key_len; ++i) {
      training_buffer_->char_min[i] = 0;
      training_buffer_->char_max[i] = 0;
    }
    training_buffer_->max_key_length = key_len;
  }

  for (uint32_t i = 0; i < key_len; ++i) {
    uint8_t ch = static_cast<uint8_t>(user_key.data()[i]);
    training_buffer_->char_min[i] = std::min(training_buffer_->char_min[i], ch);
    training_buffer_->char_max[i] = std::max(training_buffer_->char_max[i], ch);
  }
  for (uint32_t i = key_len; i < training_buffer_->max_key_length; ++i) {
    training_buffer_->char_min[i] = std::min(training_buffer_->char_min[i],
                                             static_cast<uint8_t>(0));
  }
}

Status LearnedBlockBasedTableBuilder::TrainLearnedIndex() {
  if (!enable_learned_index_ || !training_buffer_) {
    return Status::OK();
  }

  if (training_buffer_->num_entries < min_training_keys_) {
    // Allow training on small files to maximize learned index coverage.
    // Fall back only if there are zero entries.
    if (training_buffer_->num_entries == 0) {
      LearnedIndexManager::GetInstance().RecordTrainingFailure();
      return Status::InvalidArgument("No training data collected");
    }
  }

  // Finalize encoder from collected stats.
  KeyEncoder encoder;
  encoder.SetCharStats(training_buffer_->max_key_length,
                       training_buffer_->char_min,
                       training_buffer_->char_max);

  // Scan temp file to compute regression stats.
  LearnedIndex::RegressionStats stats;
  uint64_t total_bytes = 0;
  std::vector<LearnedIndex::RegressionStats> segment_stats;
  std::vector<double> segment_max_encoded_keys;
  Status s = ScanTempKVForStats(encoder, &stats, &total_bytes, &segment_stats,
                                &segment_max_encoded_keys);
  if (!s.ok()) {
    LearnedIndexManager::GetInstance().RecordTrainingFailure();
    return s;
  }

  // Train the learned index from stats.
  bool success = false;
  if (!segment_stats.empty()) {
    success = learned_index_->TrainFromSegmentStats(
        encoder, segment_stats, segment_max_encoded_keys, target_block_size_,
        training_buffer_->num_entries, total_bytes);
  } else {
    success = learned_index_->TrainFromStats(
        encoder, stats, target_block_size_, training_buffer_->num_entries,
        total_bytes);
  }

  if (!success) {
    LearnedIndexManager::GetInstance().RecordTrainingFailure();
    return Status::Corruption("Failed to train learned index");
  }

  // Switch to writing phase
  training_buffer_->collection_phase = false;

  // Calculate entries per block for write phase (best-effort estimate)
  entries_per_block_ = 32;
  if (training_buffer_->num_entries > 0 && target_block_size_ > 0) {
    uint64_t avg_entry = total_bytes / training_buffer_->num_entries;
    if (avg_entry > 0) {
      entries_per_block_ = std::max<uint64_t>(1, target_block_size_ / avg_entry);
    }
  }

  return Status::OK();
}

Status LearnedBlockBasedTableBuilder::CloseTempKVWriter() {
  if (temp_kv_writer_) {
    IOOptions io_opts;
    IOStatus s = temp_kv_writer_->Sync(io_opts, false /* use_fsync */);
    if (!s.ok()) {
      return s;
    }
    s = temp_kv_writer_->Close(io_opts);
    temp_kv_writer_.reset();
    return s;
  }
  return Status::OK();
}

Status LearnedBlockBasedTableBuilder::WriteTempKV(const Slice& key,
                                                  const Slice& value) {
  if (!temp_kv_writer_) {
    return Status::InvalidArgument("Temp KV writer not initialized");
  }
  std::string buf;
  buf.reserve(8 + key.size() + value.size());
  PutFixed32(&buf, static_cast<uint32_t>(key.size()));
  PutFixed32(&buf, static_cast<uint32_t>(value.size()));
  buf.append(key.data(), key.size());
  buf.append(value.data(), value.size());
  IOOptions io_opts;
  return temp_kv_writer_->Append(io_opts, Slice(buf));
}

Status LearnedBlockBasedTableBuilder::InitTempKVWriterIfNeeded() {
  if (temp_kv_writer_) {
    return Status::OK();
  }
  if (env_ == nullptr) {
    return Status::InvalidArgument("Env not set for temp KV writer");
  }
  if (sst_file_path_.empty()) {
    return Status::InvalidArgument("SST file path not set for temp KV writer");
  }

  temp_kv_path_ = sst_file_path_ + ".li.kvtmp";
  auto fs = env_->GetFileSystem();
  FileOptions fopts;
  std::unique_ptr<FSWritableFile> wf;
  IOStatus io_s = fs->NewWritableFile(temp_kv_path_, fopts, &wf, nullptr);
  if (!io_s.ok()) {
    return io_s;
  }
  temp_kv_writer_.reset(
      new WritableFileWriter(std::move(wf), temp_kv_path_, fopts));
  return Status::OK();
}

Status LearnedBlockBasedTableBuilder::ReplayTempKV(bool use_learned_index) {
  if (!env_ || temp_kv_path_.empty()) {
    return Status::InvalidArgument("Temp KV path not set");
  }

  std::unique_ptr<SequentialFile> sf;
  Status s = env_->NewSequentialFile(temp_kv_path_, &sf, EnvOptions());
  if (!s.ok()) {
    return s;
  }

  while (true) {
    char len_buf[8];
    Slice len_slice;
    s = sf->Read(sizeof(len_buf), &len_slice, len_buf);
    if (!s.ok()) {
      return s;
    }
    if (len_slice.size() == 0) {
      break;  // EOF
    }
    if (len_slice.size() != sizeof(len_buf)) {
      return Status::Corruption("Temp KV file truncated (length header)");
    }
    uint32_t key_len = DecodeFixed32(len_buf);
    uint32_t value_len = DecodeFixed32(len_buf + 4);

    std::string key_buf;
    std::string value_buf;
    key_buf.resize(key_len);
    value_buf.resize(value_len);
    Slice key_slice;
    Slice value_slice;
    if (key_len > 0) {
      s = sf->Read(key_len, &key_slice, &key_buf[0]);
    } else {
      key_slice = Slice();
    }
    if (!s.ok()) {
      return s;
    }
    if (key_slice.size() != key_len) {
      return Status::Corruption("Temp KV file truncated (key)");
    }
    if (value_len > 0) {
      s = sf->Read(value_len, &value_slice, &value_buf[0]);
    } else {
      value_slice = Slice();
    }
    if (!s.ok()) {
      return s;
    }
    if (value_slice.size() != value_len) {
      return Status::Corruption("Temp KV file truncated (value)");
    }

    Slice key(key_buf);
    Slice value(value_buf);
    if (use_learned_index) {
      Add(key, value);
    } else {
      BlockBasedTableBuilder::Add(key, value);
    }
  }
  return Status::OK();
}

Status LearnedBlockBasedTableBuilder::ScanTempKVForStats(
    const KeyEncoder& encoder, LearnedIndex::RegressionStats* stats,
    uint64_t* total_bytes,
    std::vector<LearnedIndex::RegressionStats>* segment_stats,
    std::vector<double>* segment_max_encoded_keys) {
  if (!env_ || temp_kv_path_.empty()) {
    return Status::InvalidArgument("Temp KV path not set");
  }
  if (!stats || !total_bytes) {
    return Status::InvalidArgument("Invalid stats pointers");
  }

  std::unique_ptr<SequentialFile> sf;
  Status s = env_->NewSequentialFile(temp_kv_path_, &sf, EnvOptions());
  if (!s.ok()) {
    return s;
  }

  uint64_t cumulative_bytes = 0;
  uint64_t entry_index = 0;
  uint64_t total_entries = training_buffer_->num_entries;
  uint32_t segments = 0;
  uint64_t seg_keys = segment_keys_ == 0 ? 1000000 : segment_keys_;
  if (total_entries > 0) {
    segments = static_cast<uint32_t>(std::max<uint64_t>(
        1, std::min<uint64_t>(1024, (total_entries + seg_keys - 1) / seg_keys)));
  }
  uint64_t entries_per_segment =
      segments > 0 ? (total_entries + segments - 1) / segments : 0;
  if (segment_stats && segment_max_encoded_keys && segments > 1) {
    segment_stats->assign(segments, LearnedIndex::RegressionStats());
    segment_max_encoded_keys->assign(segments, 0.0);
  }

  while (true) {
    char len_buf[8];
    Slice len_slice;
    s = sf->Read(sizeof(len_buf), &len_slice, len_buf);
    if (!s.ok()) {
      return s;
    }
    if (len_slice.size() == 0) {
      break;  // EOF
    }
    if (len_slice.size() != sizeof(len_buf)) {
      return Status::Corruption("Temp KV file truncated (length header)");
    }
    uint32_t key_len = DecodeFixed32(len_buf);
    uint32_t value_len = DecodeFixed32(len_buf + 4);

    std::string key_buf;
    std::string value_buf;
    key_buf.resize(key_len);
    value_buf.resize(value_len);
    Slice key_slice;
    Slice value_slice;
    if (key_len > 0) {
      s = sf->Read(key_len, &key_slice, &key_buf[0]);
    } else {
      key_slice = Slice();
    }
    if (!s.ok()) {
      return s;
    }
    if (key_slice.size() != key_len) {
      return Status::Corruption("Temp KV file truncated (key)");
    }
    if (value_len > 0) {
      s = sf->Read(value_len, &value_slice, &value_buf[0]);
    } else {
      value_slice = Slice();
    }
    if (!s.ok()) {
      return s;
    }
    if (value_slice.size() != value_len) {
      return Status::Corruption("Temp KV file truncated (value)");
    }

    size_t entry_size = key_len + value_len + 5;
    cumulative_bytes += entry_size;
    Slice user_key = ExtractUserKey(Slice(key_buf));
    double encoded_key = encoder.EncodeKey(user_key);

    stats->n++;
    stats->sum_x += static_cast<long double>(encoded_key);
    stats->sum_y += static_cast<long double>(cumulative_bytes);
    stats->sum_xx += static_cast<long double>(encoded_key) *
                     static_cast<long double>(encoded_key);
    stats->sum_xy += static_cast<long double>(encoded_key) *
                     static_cast<long double>(cumulative_bytes);

    if (segment_stats && segment_max_encoded_keys && segments > 1) {
      uint64_t seg = std::min<uint64_t>(
          segments - 1, entry_index / entries_per_segment);
      auto& seg_stats = (*segment_stats)[seg];
      seg_stats.n++;
      seg_stats.sum_x += static_cast<long double>(encoded_key);
      seg_stats.sum_y += static_cast<long double>(cumulative_bytes);
      seg_stats.sum_xx += static_cast<long double>(encoded_key) *
                          static_cast<long double>(encoded_key);
      seg_stats.sum_xy += static_cast<long double>(encoded_key) *
                          static_cast<long double>(cumulative_bytes);
      (*segment_max_encoded_keys)[seg] = encoded_key;
    }
    entry_index++;
  }

  *total_bytes = cumulative_bytes;
  if (segment_stats && segment_max_encoded_keys && segments <= 1) {
    segment_stats->clear();
    segment_max_encoded_keys->clear();
  }
  return Status::OK();
}

bool LearnedBlockBasedTableBuilder::ShouldFlushBlock(
    const Slice& key, const Slice& value) {
  (void)value;  // Suppress unused parameter warning
  
  if (!learned_index_ || !learned_index_->IsTrained()) {
    return false;
  }

  // Predict based on user key to match training.
  Slice user_key = ExtractUserKey(key);
  
  // Use the learned model to predict block_id
  // block_id = floor(f(key) / tau) - this is the core formula from the paper
  uint32_t new_block_idx = learned_index_->PredictBlock(user_key);

  current_entry_index_++;

  if (current_predicted_block_ == UINT32_MAX) {
    current_predicted_block_ = new_block_idx;
    last_user_key_.assign(user_key.data(), user_key.size());
    return false;  // No flush - first entry
  }

  // Ensure all versions of the same user key stay in the same block.
  if (user_key.size() == last_user_key_.size() &&
      memcmp(user_key.data(), last_user_key_.data(), user_key.size()) == 0) {
    return false;
  }

  if (new_block_idx != current_predicted_block_) {
    last_flushed_block_id_ = current_predicted_block_;
    current_predicted_block_ = new_block_idx;
    last_user_key_.assign(user_key.data(), user_key.size());
    return true;  // Flush current block
  }

  last_user_key_.assign(user_key.data(), user_key.size());
  return false;  // Continue adding to current block
}

Status LearnedBlockBasedTableBuilder::Finish() {
  finish_called_ = true;
  
  // fprintf(stderr, "[LearnedIndex] Finish() called! enable=%d, collection_phase=%d, buffered=%zu\n",
          // enable_learned_index_,
          // training_buffer_ ? training_buffer_->collection_phase : -1,
          // training_buffer_ ? training_buffer_->buffered_data.size() : 0);
  fflush(stderr);
  
  if (!enable_learned_index_) {
    return BlockBasedTableBuilder::Finish();
  }

  Status s;
  LocalOp learned_index_train_timer;
  LocalOp learned_index_persist_timer;
  bool trace_learned_index_build = false;
  
  if (training_buffer_ && training_buffer_->collection_phase) {
    // Close temp writer before reading
    s = CloseTempKVWriter();
    if (!s.ok()) {
      return s;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 1: Train the learned index
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    learned_index_train_timer.begin();
    
    // fprintf(stderr, "[LearnedIndex] Training with %zu entries...\n", 
            // training_buffer_->entries.size());
    fflush(stderr);
    
    s = TrainLearnedIndex();
    if (!s.ok()) {
      // Training failed, fall back to standard approach
      // fprintf(stderr, "[LearnedIndex] Training failed: %s. Falling back to standard mode.\n",
              // s.ToString().c_str());
      fflush(stderr);
      
      enable_learned_index_ = false;
      training_buffer_->collection_phase = false;

      // Replay all data using standard approach
      s = ReplayTempKV(false /* use_learned_index */);
      if (!s.ok()) {
        return s;
      }
      s = BlockBasedTableBuilder::Finish();
      if (env_ && !temp_kv_path_.empty()) {
        env_->DeleteFile(temp_kv_path_).PermitUncheckedError();
      }
      return s;
    }
    learned_index_train_timer.record_phase();
    learned_index_train_timer.commit(kLearnedIndexTrainTime);
    trace_learned_index_build = true;
    
    // fprintf(stderr, "[LearnedIndex] Training successful! Replaying %zu entries with model guidance...\n",
    //         training_buffer_->buffered_data.size());
    fflush(stderr);
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 2: Switch to writing phase
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    training_buffer_->collection_phase = false;
    current_predicted_block_ = UINT32_MAX;  // Special marker: not yet initialized
    last_flushed_block_id_ = UINT32_MAX;
    last_user_key_.clear();
    current_block_bytes_ = 0;
    current_entry_index_ = 0;  // Reset entry index for replay
    physical_block_count_ = 0;  // Reset physical block counter
    
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Step 3: Replay all data with model guidance (original order)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    s = ReplayTempKV(true /* use_learned_index */);
    if (!s.ok()) {
      return s;
    }
  }

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // Step 4: Finalize the table
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
  s = BlockBasedTableBuilder::Finish();
  if (!s.ok()) {
    return s;
  }

  // Write learned index metadata to file
  if (trace_learned_index_build) {
    learned_index_persist_timer.begin();
  }
  s = WriteLearnedIndexMetadata();
  if (s.ok() && trace_learned_index_build) {
    learned_index_persist_timer.record_phase();
    learned_index_persist_timer.commit(kLearnedIndexPersistTime);
    LocalOp learned_index_timer;
    learned_index_timer.phase_cycles[0] = rdtscp() - li_build_begin_cycles_;
    learned_index_timer.max_phase_idx = 1;
    learned_index_timer.commit(kLearnedIndexBuildTime);
  }

  // Cleanup temp file
  if (env_ && !temp_kv_path_.empty()) {
    env_->DeleteFile(temp_kv_path_).PermitUncheckedError();
  }
  
  return s;
}

Status LearnedBlockBasedTableBuilder::WriteLearnedIndexMetadata() {
  if (!learned_index_ || !learned_index_->IsTrained()) {
    return Status::OK();
  }

  // Check if LearnedIndexManager is initialized
  LearnedIndexManager& manager = LearnedIndexManager::GetInstance();
  if (!manager.IsInitialized()) {
    // Manager not initialized, skip saving
    // This can happen during unit tests or when learned index is not fully configured
    return Status::OK();
  }
  
  // Save to .li file and register in memory
  Status s = manager.SaveLearnedIndex(sst_file_path_, file_number_,
                                      *learned_index_, *block_mapper_);
  if (!s.ok()) {
    // Log warning but don't fail the table build
    // fprintf(stderr, "[LearnedIndex] Warning: Failed to save learned index for %s: %s\n",
    //         sst_file_path_.c_str(), s.ToString().c_str());
  }
  
  return Status::OK();
}

LearnedBlockBasedTableBuilder::LearnedIndexStats
LearnedBlockBasedTableBuilder::GetLearnedIndexStats() const {
  LearnedIndexStats stats;
  stats.enabled = enable_learned_index_;
  stats.trained = learned_index_ && learned_index_->IsTrained();
  stats.num_training_samples = 
      training_buffer_ ? static_cast<uint32_t>(training_buffer_->num_entries) : 0;
  
  if (stats.trained) {
    auto model_stats = learned_index_->GetStats();
    stats.predicted_num_blocks = model_stats.num_blocks;
    stats.model_slope = model_stats.model_slope;
    stats.model_intercept = model_stats.model_intercept;
  } else {
    stats.predicted_num_blocks = 0;
    stats.model_slope = 0.0;
    stats.model_intercept = 0.0;
  }
  
  return stats;
}

bool LearnedBlockBasedTableBuilder::IsEmpty() const {
  if (!enable_learned_index_ || !training_buffer_) {
    return BlockBasedTableBuilder::IsEmpty();
  }
  
  // During collection phase, check if we have buffered data
  if (training_buffer_->collection_phase) {
    return training_buffer_->num_entries == 0 &&
           BlockBasedTableBuilder::IsEmpty();
  }
  
  // After collection phase, defer to parent
  return BlockBasedTableBuilder::IsEmpty();
}

uint64_t LearnedBlockBasedTableBuilder::NumEntries() const {
  if (!enable_learned_index_ || !training_buffer_) {
    return BlockBasedTableBuilder::NumEntries();
  }
  
  // During collection phase, count buffered entries
  if (training_buffer_->collection_phase) {
    return training_buffer_->num_entries + BlockBasedTableBuilder::NumEntries();
  }
  
  // After collection phase, defer to parent
  return BlockBasedTableBuilder::NumEntries();
}

void LearnedBlockBasedTableBuilder::OnDataBlockWritten(uint64_t offset, uint32_t size) {
  // Only record block locations when learned index is enabled and we're past collection phase
  if (enable_learned_index_ && block_mapper_ && 
      training_buffer_ && !training_buffer_->collection_phase) {
    
    // The current_predicted_block_ tells us what logical block_id this data belongs to
    // We need to ensure block_mapper_ indices match logical block_ids
    // If there's a gap (e.g., from block 0 to block 3), insert empty placeholders for 1 and 2
    
    uint32_t logical_block_id = current_predicted_block_;
    if (last_flushed_block_id_ != UINT32_MAX) {
      logical_block_id = last_flushed_block_id_;
      last_flushed_block_id_ = UINT32_MAX;
    }
    uint32_t current_mapper_size = block_mapper_->GetNumBlocks();
    
    // Insert empty placeholder blocks for any gaps
    while (current_mapper_size < logical_block_id) {
      // Add empty block (offset=0, size=0) as placeholder
      block_mapper_->AddBlock(0, 0);
      // fprintf(stderr, "[LI-BLOCK] Empty placeholder block #%u inserted\n", current_mapper_size);
      current_mapper_size++;
    }
    
    // Now add the actual block at the correct logical position
    block_mapper_->AddBlock(offset, size);
    physical_block_count_++;
    
    // Debug: show block recording
    static std::atomic<uint64_t> block_write_count{0};
    // if (++block_write_count <= 10 || block_write_count % 1000 == 0) {
    //   fprintf(stderr, "[LI-BLOCK] Block #%u (logical) written at offset=%lu, size=%u, physical=%u\n",
    //           logical_block_id, offset, size, physical_block_count_);
    // }
  }
}

bool LearnedBlockBasedTableBuilder::ShouldFlushBeforeAdd(
    const Slice& key, const Slice& value, bool policy_says_flush) {
  // During collection phase (Pass 1), use default policy
  // We're not actually writing data blocks in Pass 1
  if (!enable_learned_index_ || !training_buffer_ || 
      training_buffer_->collection_phase) {
    return policy_says_flush;
  }
  
  // During Pass 2 (replay), COMPLETELY ignore the default policy
  // Use only learned index predictions to determine block boundaries
  return ShouldFlushBlock(key, value);
}

}  // namespace ROCKSDB_NAMESPACE
