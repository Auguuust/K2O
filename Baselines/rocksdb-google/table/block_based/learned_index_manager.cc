//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/block_based/learned_index_manager.h"

#include <algorithm>
#include <cstring>

#include "file/filename.h"
#include "rocksdb/env.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

// File format for .li files:
// [magic_number:4] [version:4]
// [learned_index_size:4] [learned_index_data:...]
// [block_mapper_size:4] [block_mapper_data:...]
// [checksum:4]

static const uint32_t kLearnedIndexMagic = 0x4C49444D;  // "LIDM" - Learned Index Data Model
static const uint32_t kLearnedIndexVersion = 1;

LearnedIndexManager::LearnedIndexManager()
    : env_(nullptr), initialized_(false) {}

LearnedIndexManager::~LearnedIndexManager() {
  Clear();
}

LearnedIndexManager& LearnedIndexManager::GetInstance() {
  static LearnedIndexManager instance;
  return instance;
}

void LearnedIndexManager::Initialize(Env* env, const std::string& db_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  env_ = env;
  db_path_ = db_path;
  initialized_ = true;
}

std::string LearnedIndexManager::GetLearnedIndexFilePath(
    const std::string& sst_file_path) {
  // Replace .sst extension with .li
  std::string li_path = sst_file_path;
  size_t dot_pos = li_path.rfind(".sst");
  if (dot_pos != std::string::npos) {
    li_path.replace(dot_pos, 4, ".li");
  } else {
    // If no .sst extension, just append .li
    li_path += ".li";
  }
  return li_path;
}

Status LearnedIndexManager::WriteToFile(const std::string& file_path,
                                        const LearnedIndex& learned_index,
                                        const BlockLocationMapper& block_mapper) {
  if (!env_) {
    return Status::InvalidArgument("LearnedIndexManager not initialized");
  }
  
  // Serialize learned index
  std::string li_data;
  learned_index.Serialize(&li_data);
  
  // Serialize block mapper
  std::string mapper_data;
  block_mapper.Serialize(&mapper_data);
  
  // Build file content
  std::string content;
  content.reserve(16 + li_data.size() + mapper_data.size());
  
  // Header
  PutFixed32(&content, kLearnedIndexMagic);
  PutFixed32(&content, kLearnedIndexVersion);
  
  // Learned index data
  PutFixed32(&content, static_cast<uint32_t>(li_data.size()));
  content.append(li_data);
  
  // Block mapper data
  PutFixed32(&content, static_cast<uint32_t>(mapper_data.size()));
  content.append(mapper_data);
  
  // Simple checksum (sum of all bytes)
  uint32_t checksum = 0;
  for (char c : content) {
    checksum += static_cast<uint8_t>(c);
  }
  PutFixed32(&content, checksum);
  
  // Write to file
  std::unique_ptr<WritableFile> file;
  Status s = env_->NewWritableFile(file_path, &file, EnvOptions());
  if (!s.ok()) {
    return s;
  }
  
  s = file->Append(content);
  if (!s.ok()) {
    return s;
  }
  
  s = file->Sync();
  if (!s.ok()) {
    return s;
  }
  
  return file->Close();
}

Status LearnedIndexManager::ReadFromFile(const std::string& file_path,
                                         LearnedIndex* learned_index,
                                         BlockLocationMapper* block_mapper) {
  if (!env_) {
    return Status::InvalidArgument("LearnedIndexManager not initialized");
  }
  
  // Read entire file
  std::string content;
  Status s = ReadFileToString(env_, file_path, &content);
  if (!s.ok()) {
    return s;
  }
  
  if (content.size() < 20) {  // Minimum valid size
    return Status::Corruption("Learned index file too small");
  }
  
  const char* data = content.data();
  size_t offset = 0;
  
  // Verify magic number
  uint32_t magic = DecodeFixed32(data + offset);
  offset += 4;
  if (magic != kLearnedIndexMagic) {
    return Status::Corruption("Invalid learned index file magic number");
  }
  
  // Verify version
  uint32_t version = DecodeFixed32(data + offset);
  offset += 4;
  if (version != kLearnedIndexVersion) {
    return Status::NotSupported("Unsupported learned index file version");
  }
  
  // Read learned index data
  uint32_t li_size = DecodeFixed32(data + offset);
  offset += 4;
  if (offset + li_size > content.size()) {
    return Status::Corruption("Invalid learned index data size");
  }
  Slice li_slice(data + offset, li_size);
  offset += li_size;
  
  // Read block mapper data
  uint32_t mapper_size = DecodeFixed32(data + offset);
  offset += 4;
  if (offset + mapper_size > content.size()) {
    return Status::Corruption("Invalid block mapper data size");
  }
  Slice mapper_slice(data + offset, mapper_size);
  offset += mapper_size;
  
  // Verify checksum
  if (offset + 4 > content.size()) {
    return Status::Corruption("Missing checksum");
  }
  uint32_t stored_checksum = DecodeFixed32(data + offset);
  uint32_t computed_checksum = 0;
  for (size_t i = 0; i < offset; i++) {
    computed_checksum += static_cast<uint8_t>(content[i]);
  }
  if (stored_checksum != computed_checksum) {
    return Status::Corruption("Learned index file checksum mismatch");
  }
  
  // Deserialize learned index
  if (!learned_index->Deserialize(li_slice)) {
    return Status::Corruption("Failed to deserialize learned index");
  }
  
  // Deserialize block mapper
  if (!block_mapper->Deserialize(mapper_slice)) {
    return Status::Corruption("Failed to deserialize block mapper");
  }
  
  return Status::OK();
}

Status LearnedIndexManager::SaveLearnedIndex(const std::string& sst_file_path,
                                             uint64_t file_number,
                                             const LearnedIndex& learned_index,
                                             const BlockLocationMapper& block_mapper) {
  std::string li_path = GetLearnedIndexFilePath(sst_file_path);
  
  // fprintf(stderr, "[LI-SAVE] Saving learned index for file %lu to %s, num_blocks=%u\n",
  //         file_number, li_path.c_str(), block_mapper.GetNumBlocks());
  
  // Write to file first
  Status s = WriteToFile(li_path, learned_index, block_mapper);
  if (!s.ok()) {
    // fprintf(stderr, "[LI-SAVE] Failed to write file: %s\n", s.ToString().c_str());
    return s;
  }
  
  // Add to memory
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto entry = std::make_unique<LearnedIndexEntry>();
  entry->file_number = file_number;
  entry->file_path = li_path;
  entry->learned_index = std::make_unique<LearnedIndex>();
  entry->block_mapper = std::make_unique<BlockLocationMapper>();
  
  // Copy the learned index and block mapper
  std::string li_data, mapper_data;
  learned_index.Serialize(&li_data);
  block_mapper.Serialize(&mapper_data);
  entry->learned_index->Deserialize(li_data);
  entry->block_mapper->Deserialize(mapper_data);
  
  indexes_[file_number] = std::move(entry);
  
  return Status::OK();
}

const LearnedIndexEntry* LearnedIndexManager::GetLearnedIndex(
    uint64_t file_number) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = indexes_.find(file_number);
  if (it != indexes_.end()) {
    return it->second.get();
  }
  return nullptr;
}

bool LearnedIndexManager::HasLearnedIndex(uint64_t file_number) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return indexes_.find(file_number) != indexes_.end();
}

Status LearnedIndexManager::LoadLearnedIndex(const std::string& sst_file_path,
                                             uint64_t file_number) {
  std::string li_path = GetLearnedIndexFilePath(sst_file_path);
  
  // Check if file exists
  if (!env_->FileExists(li_path).ok()) {
    return Status::NotFound("Learned index file not found", li_path);
  }
  
  // Check if already loaded
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (indexes_.find(file_number) != indexes_.end()) {
      return Status::OK();  // Already loaded
    }
  }
  
  // Read from file
  auto entry = std::make_unique<LearnedIndexEntry>();
  entry->file_number = file_number;
  entry->file_path = li_path;
  entry->learned_index = std::make_unique<LearnedIndex>();
  entry->block_mapper = std::make_unique<BlockLocationMapper>();
  
  Status s = ReadFromFile(li_path, entry->learned_index.get(),
                          entry->block_mapper.get());
  if (!s.ok()) {
    return s;
  }
  
  // Add to memory
  std::lock_guard<std::mutex> lock(mutex_);
  indexes_[file_number] = std::move(entry);
  
  return Status::OK();
}

Status LearnedIndexManager::LoadAllLearnedIndexes(const std::string& db_path) {
  if (!env_) {
    return Status::InvalidArgument("LearnedIndexManager not initialized");
  }
  
  std::vector<std::string> files;
  Status s = env_->GetChildren(db_path, &files);
  if (!s.ok()) {
    return s;
  }
  
  for (const auto& file : files) {
    if (file.size() > 3 && file.substr(file.size() - 3) == ".li") {
      // Extract file number from filename
      // Format: XXXXXX.li where XXXXXX is the file number
      std::string num_str = file.substr(0, file.size() - 3);
      uint64_t file_number = 0;
      try {
        file_number = std::stoull(num_str);
      } catch (...) {
        continue;  // Skip invalid filenames
      }
      
      std::string full_path = db_path + "/" + file;
      std::string sst_path = db_path + "/" + num_str + ".sst";
      
      // Only load if corresponding SST file exists
      if (env_->FileExists(sst_path).ok()) {
        s = LoadLearnedIndex(sst_path, file_number);
        if (!s.ok() && !s.IsNotFound()) {
          // Log warning but continue
          fprintf(stderr, "[LearnedIndexManager] Warning: Failed to load %s: %s\n",
                  full_path.c_str(), s.ToString().c_str());
        }
      } else {
        // SST file doesn't exist, delete orphaned .li file
        env_->DeleteFile(full_path).PermitUncheckedError();
      }
    }
  }
  
  return Status::OK();
}

Status LearnedIndexManager::RemoveLearnedIndex(uint64_t file_number) {
  std::string li_path;
  
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = indexes_.find(file_number);
    if (it == indexes_.end()) {
      return Status::OK();  // Not found, nothing to remove
    }
    li_path = it->second->file_path;
    indexes_.erase(it);
  }
  
  // Delete file (outside of lock)
  if (env_ && !li_path.empty()) {
    Status s = env_->DeleteFile(li_path);
    if (!s.ok() && !s.IsNotFound()) {
      // Log warning but don't fail
      fprintf(stderr, "[LearnedIndexManager] Warning: Failed to delete %s: %s\n",
              li_path.c_str(), s.ToString().c_str());
    }
  }
  
  return Status::OK();
}

Status LearnedIndexManager::RemoveLearnedIndexes(
    const std::vector<uint64_t>& file_numbers) {
  for (uint64_t num : file_numbers) {
    RemoveLearnedIndex(num);  // Ignore errors for individual files
  }
  return Status::OK();
}

LearnedIndexManager::Stats LearnedIndexManager::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  Stats stats;
  stats.total_indexes = indexes_.size();
  stats.total_memory_bytes = 0;
  stats.training_failures = training_failures_.load(std::memory_order_relaxed);
  stats.learned_reads = learned_reads_.load(std::memory_order_relaxed);
  stats.learned_hits = learned_hits_.load(std::memory_order_relaxed);
  stats.learned_misses = learned_misses_.load(std::memory_order_relaxed);
  stats.learned_fallbacks = learned_fallbacks_.load(std::memory_order_relaxed);
  
  for (const auto& pair : indexes_) {
    // Approximate memory usage per entry
    stats.total_memory_bytes += sizeof(LearnedIndexEntry);
    stats.total_memory_bytes += sizeof(LearnedIndex);
    stats.total_memory_bytes += sizeof(BlockLocationMapper);
    // Add approximate size of serialized data
    if (pair.second->learned_index) {
      stats.total_memory_bytes += 100;  // Approximate
    }
    if (pair.second->block_mapper) {
      stats.total_memory_bytes += pair.second->block_mapper->GetNumBlocks() * 12;
    }
  }
  
  return stats;
}

void LearnedIndexManager::RecordTrainingFailure() {
  training_failures_.fetch_add(1, std::memory_order_relaxed);
}

void LearnedIndexManager::RecordLearnedRead(bool hit, bool fallback) {
  learned_reads_.fetch_add(1, std::memory_order_relaxed);
  if (hit) {
    learned_hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    learned_misses_.fetch_add(1, std::memory_order_relaxed);
  }
  if (fallback) {
    learned_fallbacks_.fetch_add(1, std::memory_order_relaxed);
  }
}

void LearnedIndexManager::RecordLearnedFallback() {
  learned_fallbacks_.fetch_add(1, std::memory_order_relaxed);
}

void LearnedIndexManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  indexes_.clear();
}

}  // namespace ROCKSDB_NAMESPACE
