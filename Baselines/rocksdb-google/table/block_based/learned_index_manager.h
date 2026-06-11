//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// LearnedIndexManager: Global manager for learned index models
// - Manages .li files (learned index files) corresponding to SST files
// - Keeps learned index models in memory for fast lookup
// - Handles cleanup when SST files are deleted (compaction, etc.)

#pragma once

#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>

#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "table/block_based/learned_index.h"

namespace ROCKSDB_NAMESPACE {

// LearnedIndexEntry holds a learned index model in memory
struct LearnedIndexEntry {
  std::unique_ptr<LearnedIndex> learned_index;
  std::unique_ptr<BlockLocationMapper> block_mapper;
  uint64_t file_number;
  std::string file_path;  // Full path to .li file
  
  LearnedIndexEntry() : file_number(0) {}
};

// LearnedIndexManager is a singleton that manages all learned index models
// Thread-safe for concurrent access
class LearnedIndexManager {
 public:
  // Get the singleton instance
  static LearnedIndexManager& GetInstance();
  
  // Initialize the manager with an Env (for file operations)
  void Initialize(Env* env, const std::string& db_path);
  
  // Check if the manager is initialized
  bool IsInitialized() const { return initialized_; }
  
  // =====================================================
  // Write Path APIs
  // =====================================================
  
  // Save a learned index to a .li file and register it in memory
  // Called after building an SST file with learned index
  // sst_file_path: full path to the SST file (e.g., /path/to/db/000123.sst)
  // file_number: the SST file number (e.g., 123)
  Status SaveLearnedIndex(const std::string& sst_file_path,
                          uint64_t file_number,
                          const LearnedIndex& learned_index,
                          const BlockLocationMapper& block_mapper);
  
  // =====================================================
  // Read Path APIs
  // =====================================================
  
  // Get a learned index for a given SST file number
  // Returns nullptr if no learned index exists for this file
  // The returned pointer is valid until RemoveLearnedIndex is called
  const LearnedIndexEntry* GetLearnedIndex(uint64_t file_number) const;
  
  // Check if a learned index exists for a given SST file
  bool HasLearnedIndex(uint64_t file_number) const;
  
  // Load a learned index from disk if not already in memory
  // Called when opening an existing database
  Status LoadLearnedIndex(const std::string& sst_file_path, uint64_t file_number);
  
  // Load all learned indexes from a directory
  // Called during database open to restore learned indexes
  Status LoadAllLearnedIndexes(const std::string& db_path);
  
  // =====================================================
  // Cleanup APIs (for compaction, etc.)
  // =====================================================
  
  // Remove a learned index from memory and delete its .li file
  // Called when an SST file is deleted (e.g., after compaction)
  Status RemoveLearnedIndex(uint64_t file_number);
  
  // Remove multiple learned indexes
  Status RemoveLearnedIndexes(const std::vector<uint64_t>& file_numbers);
  
  // =====================================================
  // Utility APIs
  // =====================================================
  
  // Get the .li file path for a given SST file path
  static std::string GetLearnedIndexFilePath(const std::string& sst_file_path);
  
  // Get statistics
  struct Stats {
    size_t total_indexes;
    size_t total_memory_bytes;  // Approximate memory usage
    uint64_t training_failures;
    uint64_t learned_reads;
    uint64_t learned_hits;
    uint64_t learned_misses;
    uint64_t learned_fallbacks;
  };
  Stats GetStats() const;

  // Recorders for learned index stats
  void RecordTrainingFailure();
  void RecordLearnedRead(bool hit, bool fallback);
  void RecordLearnedFallback();
  
  // Clear all learned indexes (for testing or shutdown)
  void Clear();
  
  // Destructor
  ~LearnedIndexManager();

 private:
  // Private constructor for singleton
  LearnedIndexManager();
  
  // Disable copy and assignment
  LearnedIndexManager(const LearnedIndexManager&) = delete;
  LearnedIndexManager& operator=(const LearnedIndexManager&) = delete;
  
  // Internal helper to write learned index to file
  Status WriteToFile(const std::string& file_path,
                     const LearnedIndex& learned_index,
                     const BlockLocationMapper& block_mapper);
  
  // Internal helper to read learned index from file
  Status ReadFromFile(const std::string& file_path,
                      LearnedIndex* learned_index,
                      BlockLocationMapper* block_mapper);
  
  // Environment for file operations
  Env* env_;
  std::string db_path_;
  bool initialized_;
  
  // Map from SST file number to learned index entry
  // Protected by mutex for thread safety
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<LearnedIndexEntry>> indexes_;

  // Counters (process-wide)
  std::atomic<uint64_t> training_failures_{0};
  std::atomic<uint64_t> learned_reads_{0};
  std::atomic<uint64_t> learned_hits_{0};
  std::atomic<uint64_t> learned_misses_{0};
  std::atomic<uint64_t> learned_fallbacks_{0};
};

}  // namespace ROCKSDB_NAMESPACE
