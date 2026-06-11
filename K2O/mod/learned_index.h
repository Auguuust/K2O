#pragma once

#include <vector>
#include <cstring>
#include <atomic>
#include <mutex>

#include "my_trace.h"
#include "util.h"
#include "table/format.h"

using std::string;
using rocksdb::Slice;
using rocksdb::Version;
using rocksdb::FileMetaData;
using rocksdb::Status;
using rocksdb::PutFixed32;
using rocksdb::PutFixed64;
using rocksdb::BlockHandle;
using rocksdb::Cache;
using rocksdb::NewLRUCache;
using rocksdb::Env;
using rocksdb::EnvOptions;
using rocksdb::WritableFile;
using rocksdb::SequentialFile;

namespace pirMod {

static std::string ModelFileName(uint64_t file_number) {
  char buf[100];
  snprintf(buf, sizeof(buf), "%06llu.%s", (unsigned long long)file_number, "pm");
  return buf;
}

static std::string ModelFileName(std::string dbname, uint64_t file_number) {
  return dbname + "/" + ModelFileName(file_number);
}

template <typename T>
class Span {
public:
  using element_type = T;
  using value_type   = std::remove_cv_t<T>;
  using size_type    = std::size_t;
  using pointer      = T*;
  using iterator     = T*;

  // default
  constexpr Span() noexcept : data_(nullptr), size_(0) {}

  // pointer + size
  constexpr Span(T* ptr, size_type count) noexcept
      : data_(ptr), size_(count) {}

  // from vector
  template <typename Alloc>
  Span(std::vector<value_type, Alloc>& v) noexcept
      : data_(v.data()), size_(v.size()) {}

  template <typename Alloc>
  Span(const std::vector<value_type, Alloc>& v) noexcept
      : data_(v.data()), size_(v.size()) {}

  // iterators
  iterator begin() const noexcept { return data_; }
  iterator end()   const noexcept { return data_ + size_; }

  // capacity
  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }

  // element access
  T& operator[](size_type idx) const {
    assert(idx < size_);
    return data_[idx];
  }

  T* data() const noexcept { return data_; }

  // subspan
  Span<T> subspan(size_type offset, size_type count) const {
    assert(offset <= size_);
    assert(offset + count <= size_);
    return Span<T>(data_ + offset, count);
  }

  T& front() const {
    assert(size_ > 0);
    return data_[0];
  }


  T& back() const {
    assert(size_ > 0);
    return data_[size_ - 1];
  }

private:
  T* data_;
  size_type size_;
};



class PIRHandle {
 public:
  PIRHandle() = default;

  void EncodeTo(std::string* dst) const;
  Status DecodeFrom(Slice* input);
  int64_t PredIndex(const Slice& key);
  uint32_t PredOffset(const Slice& key);
  int64_t PredIndex(uint64_t key);
  uint32_t PredOffset(uint64_t key);
  size_t EncodeSize() const;
  size_t TrainedArrayMemoryUsage() const;
  size_t MemoryUsage() const;
  void Print() const;
  

  double slope = 0.0;
  uint64_t last_key;
  uint64_t first_key;
  uint64_t block_offset;
  std::vector<uint32_t> trained_array;
};

extern bool use_secondary_learned_index;
extern size_t model_total_size;
extern size_t model_count;

class LearnedIndexData {
private:
  uint32_t curr_pir_handle_idx_ = 0;
  uint32_t curr_pred_offset_ = 0;
  uint32_t curr_secondary_handle_idx_ = 0;
  bool root_called_next_ = false;
  bool secondary_called_next_ = false;
  PIRHandle* curr_secondary_handle_ = nullptr;
  std::string dbname_;
  uint64_t file_number_;
  uint64_t target_key_;
  uint64_t build_begin_cycles_ = 0;
  
public:
  std::string file_path_name;
  bool written = false;
  bool deleted = false;
  std::vector<PIRHandle*> pir_handles;
  std::vector<PIRHandle*> secondary_handles;
  PIRHandle root_handle;

  // LearnedIndexData() : root_handle(new PIRHandle()) {
  //   printf("LearnedIndexData created\n");
  // }
  LearnedIndexData(std::string dbname, uint64_t number) : dbname_(dbname), file_number_(number), file_path_name(ModelFileName(dbname, number)), written(false) {
    build_begin_cycles_ = rdtscp();
    // printf("LearnedIndexData created\n");
    // pir_handles.reserve(20000);
  }
  LearnedIndexData(const LearnedIndexData& other) = delete;
  ~LearnedIndexData() {
    for (auto pir_handle : pir_handles) {
      delete pir_handle;
    }
    // printf("LearnedIndexData destroyed\n");
  }

  void SeekReset(const Slice& key) {
    target_key_ = SliceToInteger(key);
    curr_pir_handle_idx_ = 0;
    curr_pred_offset_ = 0;
    curr_secondary_handle_idx_ = 0;
    root_called_next_ = false;
    secondary_called_next_ = false;
    curr_secondary_handle_ = nullptr;
  }

  void SeekToFirst(const Slice& key) {
    SeekReset(key);
    if (!use_secondary_learned_index) {
      SeekToFirstL2();
    } else {
      SeekToFirstL3();
    }
  }
  void Next() {
    if (!Valid()) {
      return;
    }
    if (!use_secondary_learned_index) {
      NextL2();
    } else {
      NextL3();
    }
  }
  inline PIRHandle* pir_handle() {
    return pir_handles[curr_pir_handle_idx_];
  }
  inline uint32_t pred_offset() {
    return curr_pred_offset_;
  }
  inline uint64_t block_offset() {
    return pir_handles[curr_pir_handle_idx_]->block_offset;
  }
  inline bool Valid() {
    if (use_secondary_learned_index) {
      return curr_pir_handle_idx_ < pir_handles.size() && curr_secondary_handle_idx_ < secondary_handles.size();
    }
    return curr_pir_handle_idx_ < pir_handles.size();
  }
  inline void Invalidate() {
    curr_pir_handle_idx_ = pir_handles.size();
  }
  inline uint64_t file_number() const {
    return file_number_;
  }

  inline uint64_t build_begin_cycles() const {
    return build_begin_cycles_;
  }

  PIRHandle* NewPIRHandle();

  void TrainRootPIRHandle() {
    if (use_secondary_learned_index) {
      TrainSecondary();
    } else {
      TrainRoot(root_handle, Span<PIRHandle*>(pir_handles));
    }
  }

  Status WriteModel() {
    Status s;
    if (!use_secondary_learned_index) {
      s = WriteModelL2();
    } else {
      s = WriteModelL3();
    }
    // printf("[LearnedIndexData] Average model size: %.2f MB over %lu models\n",
    //         model_total_size / 1024.0 / 1024.0 / model_count, model_count);
    return s;
  }

  Status ReadModel() {
    Status s;
    if (!use_secondary_learned_index) {
      s = ReadModelL2();
    } else {
      s = ReadModelL3();
    }

    if (!s.ok()) {
      fprintf(stderr, "Error reading LearnedIndexData model from file: %s\n", s.ToString().c_str());
    }

    return s;
  }
  size_t MemoryUsage() const {
    if (!use_secondary_learned_index) {
      return MemoryUsageL2();
    } else {
      return MemoryUsageL3();
    }
  }

private:
  void TrainRoot(PIRHandle& root, Span<PIRHandle*> handles, uint32_t start_idx = 0);
  void TrainSecondary();

  void RootPredict(PIRHandle& root, uint32_t& handle_idx);

  void SeekToFirstL2();
  void NextL2();
  Status WriteModelL2();
  Status ReadModelL2();
  size_t MemoryUsageL2() const;

  void SeekToFirstL3();
  void NextL3();
  Status WriteModelL3();
  Status ReadModelL3();
  size_t MemoryUsageL3() const;

  bool ParseNextPIRHandle();
};

class ModelGuard {
 public:
  ModelGuard() : cache_(nullptr), handle_(nullptr), model_(nullptr) {}
  ModelGuard(Cache* cache, Cache::Handle* handle)
      : cache_(cache), handle_(handle),
        model_(static_cast<LearnedIndexData*>(cache->Value(handle))) {}
  
  ModelGuard(LearnedIndexData* model)
      : cache_(nullptr), handle_(nullptr), model_(model) {}

  ~ModelGuard() {
    if (handle_) {
      cache_->Release(handle_);
    }
  }

  bool valid() const { return model_ != nullptr; }

  LearnedIndexData* operator->() const { return model_; }
  LearnedIndexData& operator*() const { return *model_; }

  LearnedIndexData* get() const { return model_; }

 private:
  Cache* cache_;
  Cache::Handle* handle_;
  LearnedIndexData* model_;
};

extern size_t li_cache_size;

class FileLearnedIndexData {
public:
  enum class Mode {
    kAllInMemory,
    kCache,
    kNoCache,
  };

  explicit FileLearnedIndexData(std::string db_name, Mode mode = Mode::kCache)
      : mode_(mode), dbname(db_name) {
    printf("[FileLearnedIndexData] Initialized for DB: %s\n",
           db_name.c_str());
    switch (mode_) {
      case Mode::kAllInMemory:
        file_learned_index_data.reserve(10000);
        break;
      case Mode::kCache:
        model_cache = NewLRUCache(li_cache_size);
        printf("Learned Index Cache Size: %lu\n", li_cache_size);
        break;
      case Mode::kNoCache:
        model_cache = NewLRUCache(0);
        break;
      default:
        assert(false);
    }
  }

  ~FileLearnedIndexData() {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto pointer : file_learned_index_data) {
      delete pointer;
    }
  }

  LearnedIndexData* NewModel(uint64_t number) {
    return new LearnedIndexData(dbname, number);
  }

  // ===== 对外统一接口 =====
  void FinishModel(LearnedIndexData* model) {
    static size_t total_model_size_in_memory = 0;
    static size_t max_model_size_in_memory = 0;
    static size_t model_count_in_memory = 0;
    model_count_in_memory++;
    total_model_size_in_memory += model->MemoryUsage();
    max_model_size_in_memory = std::max(max_model_size_in_memory, model->MemoryUsage());

    // if (model_count_in_memory % 100 == 0) {
    //   printf("[FinishModelAllInMemory] Total Models: %lu, Average Model Size: %.2f MB, Max Model Size: %.2f MB\n",
    //         model_count_in_memory,
    //         (total_model_size_in_memory / 1024.0 / 1024.0) / model_count_in_memory,
    //         max_model_size_in_memory / 1024.0 / 1024.0);
    // }
    LocalOp learned_index_timer;
    learned_index_timer.phase_cycles[0] = rdtscp() - model->build_begin_cycles();
    learned_index_timer.max_phase_idx = 1;
    learned_index_timer.commit(kLearnedIndexBuildTime);
    switch (mode_) {
      case Mode::kAllInMemory:
        FinishModelAllInMemory(model);
        break;
      case Mode::kCache:
        FinishModelCache(model);
        break;
      case Mode::kNoCache:
        FinishModelNoCache(model);
        break;
      default:
        assert(false);
    }
  }

  ModelGuard GetModel(uint64_t number) {
    switch (mode_) {
      case Mode::kAllInMemory:
        return GetModelAllInMemory(number);
      case Mode::kCache:
        return GetModelCache(number);
      case Mode::kNoCache:
        return GetModelNoCache(number);
      default:
        assert(false);
    }
    return ModelGuard();
  }

  bool DeleteModel(uint64_t number) {
    switch (mode_) {
      case Mode::kAllInMemory:
        return DeleteModelAllInMemory(number);
      case Mode::kCache:
        return DeleteModelCache(number);
      case Mode::kNoCache:
        return DeleteModelNoCache(number);
      default:
        assert(false);
        return false;
    }
  }

private:
  // ===== 原有实现 =====
  void FinishModelAllInMemory(LearnedIndexData* model);
  LearnedIndexData* GetModelAllInMemory(uint64_t number);
  bool DeleteModelAllInMemory(uint64_t number);

  void FinishModelCache(LearnedIndexData* model);
  ModelGuard GetModelCache(uint64_t number);
  bool DeleteModelCache(uint64_t number);

  void FinishModelNoCache(LearnedIndexData* model);
  ModelGuard GetModelNoCache(uint64_t number);
  bool DeleteModelNoCache(uint64_t number);
  
private:
  Mode mode_;
  std::mutex mutex;
  std::shared_ptr<Cache> model_cache;
  std::vector<LearnedIndexData*> file_learned_index_data;
  std::string dbname;
};
extern FileLearnedIndexData::Mode li_type;

}
