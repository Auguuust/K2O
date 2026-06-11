//
// Created by daiyi on 2020/02/02.
//

#include <utility>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <span>

#include "util/coding.h"
#include "db/version_set.h"
#include "mod/learned_index.h"

namespace pirMod {

FileLearnedIndexData::Mode li_type = FileLearnedIndexData::Mode::kCache;
size_t li_cache_size = 1 * 1024 * 1024 * 1024;
bool use_secondary_learned_index = false;

void PIRHandle::EncodeTo(std::string* dst) const {
  uint32_t n = trained_array.size();
  PutFixed32(dst, n);

  dst->append(reinterpret_cast<const char*>(trained_array.data()),
              n * sizeof(uint32_t));

  PutFixed64(dst, block_offset);
  PutFixed64(dst, last_key);
  PutFixed64(dst, first_key);

  dst->append(reinterpret_cast<const char*>(&slope), sizeof(slope));
}

size_t PIRHandle::EncodeSize() const {
  return sizeof(uint32_t)*(trained_array.size()+1) + sizeof(uint64_t)*3 + sizeof(double); 
}


Status PIRHandle::DecodeFrom(Slice* input) {
  uint32_t n;
  if (!GetFixed32(input, &n)) {
    return Status::Corruption("bad trained_array size");
  }

  size_t need =
      n * sizeof(uint32_t) +
      sizeof(block_offset) +
      sizeof(last_key) +
      sizeof(first_key) +
      sizeof(double);

  if (input->size() < need) {
    return Status::Corruption("truncated pir handle");
  }

  trained_array.clear();
  trained_array.resize(n);
  memcpy(trained_array.data(), input->data(), n * sizeof(uint32_t));
  input->remove_prefix(n * sizeof(uint32_t));

  GetFixed64(input, &block_offset);
  GetFixed64(input, &last_key);
  GetFixed64(input, &first_key);

  memcpy(&slope, input->data(), sizeof(slope));
  input->remove_prefix(sizeof(slope));

  return Status::OK();
}


int64_t PIRHandle::PredIndex(const Slice& key) {
  return static_cast<int64_t>(std::ceil(slope * static_cast<double>(SliceToInteger(key)-first_key)));
}

uint32_t PIRHandle::PredOffset(const Slice& key) {
  int64_t pred_idx = PredIndex(key);
  if (pred_idx < 0 || static_cast<size_t>(pred_idx) >= trained_array.size()) {
    return UINT32_MAX;
  }
  return trained_array[pred_idx];
}

int64_t PIRHandle::PredIndex(uint64_t key) {
  return static_cast<int64_t>(std::ceil(slope * static_cast<double>(key-first_key)));
}

uint32_t PIRHandle::PredOffset(uint64_t key) {
  int64_t pred_idx = PredIndex(key);
  if (pred_idx < 0 || static_cast<size_t>(pred_idx) >= trained_array.size()) {
    return UINT32_MAX;
  }
  return trained_array[pred_idx];
}

size_t PIRHandle::TrainedArrayMemoryUsage() const {
  return trained_array.capacity() * sizeof(uint32_t);
}

size_t PIRHandle::MemoryUsage() const {
  size_t size = 0;

  // 对象本身（不含 vector buffer）
  size += sizeof(PIRHandle);

  // trained_array 实际数据
  size += TrainedArrayMemoryUsage();

  return size;
}

void PIRHandle::Print() const {
  printf("PIRHandle: trained_array size=%lu slope=%.8f first_key=%lu last_key=%lu\n",
         trained_array.size(), slope, first_key, last_key);
}

void LearnedIndexData::RootPredict(PIRHandle& root, uint32_t& handle_idx) {
  int64_t pred_idx = root.PredIndex(target_key_);
  // printf("target_key: %lu, root first_key: %lu, pred_idx: %ld\n", target_key_, root.first_key, pred_idx);
  pred_idx = pred_idx < 0 ? 0 : pred_idx;

  if (pred_idx >= static_cast<int64_t>(root.trained_array.size())) {
    Invalidate();
    return;
  }

  handle_idx = 0;

  if (pred_idx > 0) {
    handle_idx = root.trained_array[pred_idx];
  }
}

void LearnedIndexData::SeekToFirstL2() {
  RootPredict(root_handle, curr_pir_handle_idx_);
  if (!Valid()) {
    return;
  }

  // fprintf(stderr, "[SeekPIRHandle] key: %lu, pred_pir_idx: %ld, pir_handle_idx=%u, pir_handles size=%lu\n", SliceToInteger(key), pred_pir_idx, curr_pir_handle_idx_, pir_handles.size());

  PIRHandle* pir_handle = pir_handles[curr_pir_handle_idx_];
  curr_pred_offset_ = pir_handle->PredOffset(target_key_);
  if (curr_pred_offset_ == UINT32_MAX) {
    curr_pir_handle_idx_++;
    if (!Valid()) {
      return;
    }
    pir_handle = pir_handles[curr_pir_handle_idx_];
    curr_pred_offset_ = pir_handle->PredOffset(target_key_);
    if (curr_pred_offset_ == UINT32_MAX) {
      Invalidate();
      return;
    }
  }
}

void LearnedIndexData::NextL2() {
  curr_pir_handle_idx_++;
  if (!Valid()) {
    return;
  }
  PIRHandle* pir_handle = pir_handles[curr_pir_handle_idx_];
  curr_pred_offset_ = pir_handle->PredOffset(target_key_);
  if (curr_pred_offset_ == UINT32_MAX) {
      Invalidate();
      return;
  }
}

bool LearnedIndexData::ParseNextPIRHandle() {
  PIRHandle* pir_handle = pir_handles[curr_pir_handle_idx_];
  // printf("  PIRHandle %u: last_key=%lu\n", curr_pir_handle_idx_, pir_handle->last_key);
  curr_pred_offset_ = pir_handle->PredOffset(target_key_);
  if (curr_pred_offset_ == UINT32_MAX) {
    return false;
  }
  return true;
}

// static int rtn[10] = {0};
// static int total = 0;
// static int key_may_in_sec1 = 0;
// static int key_may_in_sec2 = 0;

void LearnedIndexData::SeekToFirstL3() {
  // total++;
  // printf("=== SeekToFirstL3 Call %d ===\n", total);
  // for (int i = 0; i < 8; i++) {
  //   printf("rtn[%d]=%d ", i, rtn[i]);
  // }
  // printf("\n");
  // printf("Key May in Secondary 1: %d / %d\n", key_may_in_sec1, total);
  // printf("Key May in Secondary 2: %d / %d\n", key_may_in_sec2, total);
  RootPredict(root_handle, curr_secondary_handle_idx_);
  if (!Valid()) {
    return;
  }

  // printf("Seek Key: %lu\n", target_key_);

  curr_secondary_handle_ = secondary_handles[curr_secondary_handle_idx_];
  // printf("Secondary 1 Last Key: %lu\n", curr_secondary_handle_->last_key);
  // if (target_key_ <= curr_secondary_handle_->last_key) {
  //   key_may_in_sec1++;
  // }
  
  RootPredict(*curr_secondary_handle_, curr_pir_handle_idx_);
  if (Valid()) {
    if (ParseNextPIRHandle()) {
      // 第一个 PIRHandle 有效则返回
      return;
    }
    NextL3();
    return;
  }

  if (target_key_ < curr_secondary_handle_->last_key) {
    return;
  }

  secondary_called_next_ = true;
  NextL3();
}

void LearnedIndexData::NextL3() {
  if (root_called_next_ && secondary_called_next_) {
    // 四个位置都尝试过了
    Invalidate();
    return;
  }
  
  if (secondary_called_next_) {
    if (target_key_ < curr_secondary_handle_->last_key) {
      Invalidate();
      return;
    }
    // 尝试下一个 Secondary
    curr_secondary_handle_idx_++;
    curr_pir_handle_idx_ = 0;
    root_called_next_ = true;
    secondary_called_next_ = false;

    if (!Valid()) {
      return;
    }
    curr_secondary_handle_ = secondary_handles[curr_secondary_handle_idx_];
    // printf("Secondary 2 Last Key: %lu\n", curr_secondary_handle_->last_key);
    // if (target_key_ <= curr_secondary_handle_->last_key) {
    //   key_may_in_sec2++;
    // }
    RootPredict(*curr_secondary_handle_, curr_pir_handle_idx_);

    if (!Valid()) {
      return;
    }

    if (ParseNextPIRHandle()) {
      // 第1个 PIRHandle 有效则返回
      return;
    }

    NextL3();
    return;
  }

  curr_pir_handle_idx_++;
  secondary_called_next_ = true;
  if (!Valid()) {
    return;
  }
  // 尝试下一个 PIRHandle
  if (ParseNextPIRHandle()) {
    // 第2个 PIRHandle 有效则返回
    return;
  }
  
  NextL3();
}

PIRHandle* LearnedIndexData::NewPIRHandle() {
  pir_handles.push_back(new PIRHandle());
  return pir_handles.back();
}

void LearnedIndexData::TrainRoot(PIRHandle& root, Span<PIRHandle*> handles, uint32_t start_idx) {
  auto &slope = root.slope;
  auto &trained_array = root.trained_array;
  auto &last_key = root.last_key;
  auto &first_key = root.first_key;

  first_key = handles[0]->first_key;

  PIRHandle* prev_pir_handle = nullptr;

  for (auto pir_handle : handles) {
    if (prev_pir_handle) {
      double dx = static_cast<double>(pir_handle->last_key) -
                  static_cast<double>(prev_pir_handle->last_key);
      double dy = 1.01;  // each PIRHandle represents one block

      // dx must be > 0, dy >= 0
      double curr_slope = dy / dx;
      // printf("[TrainRoot] prev_last_key=%lu curr_last_key=%lu dx=%.2f dy=%.2f curr_slope=%.8f slope=%.8f\n",
      //        prev_pir_handle->last_key, pir_handle->last_key,
      //        dx, dy, curr_slope, slope);

      slope = std::max(slope, curr_slope);
    }
    prev_pir_handle = pir_handle;
  }

  last_key = handles.back()->last_key;
  uint32_t trained_array_num = static_cast<uint32_t>(ceil(slope * static_cast<double>(last_key-first_key))) + 1;
  trained_array.reserve(trained_array_num);
  uint32_t array_idx = 0;
  for (size_t pir_idx = 0; pir_idx < handles.size(); ++pir_idx) {
    auto pir_handle = handles[pir_idx];
    uint32_t pred_idx = static_cast<uint32_t>(ceil(slope * static_cast<double>(pir_handle->last_key-first_key))) + 1;
    
    trained_array.insert(trained_array.end(), pred_idx - array_idx, pir_idx+start_idx);
    array_idx = pred_idx;
    // printf("Idx %lu pred to Idx %u\n", pir_idx, pred_idx);
  }
}

void LearnedIndexData::TrainSecondary() {
  Span<PIRHandle*> pir_handles_span(pir_handles);
  size_t pir_per_secondary = static_cast<size_t>(std::ceil(
      static_cast<double>(pir_handles.size()) / static_cast<double>(100)));
  size_t num_secondary = (static_cast<size_t>(std::ceil(
      static_cast<double>(pir_handles.size()) / static_cast<double>(pir_per_secondary))));
  secondary_handles.clear();
  secondary_handles.reserve(num_secondary);

  for (size_t i = 0; i < num_secondary; ++i) {
    secondary_handles.push_back(new PIRHandle());
  }

  size_t offset = 0;
  auto it = secondary_handles.begin();
  // printf("Training %lu secondary PIRHandles, each with ~%lu PIRHandles\n", num_secondary, pir_per_secondary);
  // all but last secondary
  for (size_t i = 0; i < num_secondary - 1; ++i, ++it) {
    auto secondary_root = *(it);
    TrainRoot(*secondary_root, pir_handles_span.subspan(offset, pir_per_secondary), offset);
    offset += pir_per_secondary;
    // secondary_root->Print();
  }
  // last secondary
  auto secondary_root = *(it);
  TrainRoot(*secondary_root, pir_handles_span.subspan(offset, pir_handles.size() - offset), offset);
  // secondary_root->Print();

  TrainRoot(root_handle, Span<PIRHandle*>(secondary_handles));
  // printf("Trained Root PIRHandle:\n");
  // root_handle.Print();
}

size_t model_total_size = 0;
size_t model_count = 0;

Status LearnedIndexData::WriteModelL2() {
  if (written) {
    return Status::OK();
  }
  std::string buf;
  size_t n = sizeof(uint32_t);  // pir_handle 数量
  for (auto h : pir_handles) {
    n += h->EncodeSize();
  }
  n += root_handle.EncodeSize();
  buf.reserve(n);  // 避免反复 realloc

  // 1. pir_handle 数量
  PutFixed32(&buf, static_cast<uint32_t>(pir_handles.size()));

  // 2. pir_handles
  for (auto* h : pir_handles) {
    h->EncodeTo(&buf);
  }

  // 3. root_handle
  root_handle.EncodeTo(&buf);

  std::unique_ptr<WritableFile> file;
  EnvOptions env_opts;
  Status s = env->NewWritableFile(file_path_name, &file, env_opts);
  if (!s.ok()) {
    return s;
  }

  s = file->Append(buf);
  if (!s.ok()) {
    return s;
  }

  // fprintf(stderr,
  //         "[WriteModel] Writing learned index model to %s, "
  //         "Model Size = %.2f MB, File Size = %.2f MB\n",
  //         file_path_name.c_str(),
  //         MemoryUsage() / 1024.0 / 1024.0,
  //         buf.capacity() / 1024.0 / 1024.0);

  s = file->Flush();
  if (!s.ok()) {
    return s;
  }

  // 如果 model 是 crash-safe 必须的，保留 Sync
  s = file->Sync();
  if (!s.ok()) {
    return s;
  }

  written = true;

  model_count++;
  model_total_size += buf.capacity();

  return Status::OK();
}

Status LearnedIndexData::ReadModelL2() {
  // fprintf(stderr,
  //         "[ReadModel] Reading learned index model from %s\n",
  //         file_path_name.c_str());

  EnvOptions env_opt;
  std::unique_ptr<SequentialFile> file;
  Status s = env->NewSequentialFile(file_path_name, &file, env_opt);
  if (!s.ok()) {
    return s;
  }

  std::string contents;
  constexpr size_t kReadBufSize = 4 << 20;  // 4MB
  char scratch[kReadBufSize];
  Slice fragment;

  while (true) {
    s = file->Read(kReadBufSize, &fragment, scratch);
    if (!s.ok()) {
      return s;
    }
    if (fragment.empty()) {
      break;
    }
    contents.append(fragment.data(), fragment.size());
  }

  Slice input(contents);

  // 1. pir_handle count
  uint32_t num;
  if (!GetFixed32(&input, &num)) {
    return Status::Corruption("Failed to read pir_handle count");
  }

  pir_handles.clear();
  pir_handles.reserve(num);

  // 2. pir_handles
  for (uint32_t i = 0; i < num; ++i) {
    auto* h = new PIRHandle();
    s = h->DecodeFrom(&input);
    if (!s.ok()) {
      delete h;
      return s;
    }
    pir_handles.push_back(h);
  }

  // 3. root handle
  s = root_handle.DecodeFrom(&input);
  if (!s.ok()) {
    return s;
  }

  // Optional: detect trailing garbage
  if (!input.empty()) {
    return Status::Corruption("Extra bytes at end of model file");
  }

  written = true;

  return Status::OK();
}

size_t LearnedIndexData::MemoryUsageL2() const {
  size_t size = 0;

  // 对象本身
  size += sizeof(LearnedIndexData);

  // pir_handles vector 壳
  size += pir_handles.capacity() * sizeof(PIRHandle*);

  // root_handle
  size += root_handle.TrainedArrayMemoryUsage();

  // 每个 PIRHandle
  for (auto& p : pir_handles) {
    size += p->MemoryUsage();
  }

  return size;
}



Status LearnedIndexData::WriteModelL3() {
  if (written) {
    return Status::OK();
  }
  std::string buf;
  size_t n = sizeof(uint32_t);  // pir_handle 数量
  for (auto h : pir_handles) {
    n += h->EncodeSize();
  }
  n += sizeof(uint32_t);  // secondary 数量
  for (auto h : secondary_handles) {
    n += h->EncodeSize();
  }
  n += root_handle.EncodeSize();
  buf.reserve(n);  // 避免反复 realloc

  // 1. pir_handle 数量
  PutFixed32(&buf, static_cast<uint32_t>(pir_handles.size()));

  // 2. pir_handles
  for (auto* h : pir_handles) {
    h->EncodeTo(&buf);
  }

  // 3. secondary_handle 数量
  PutFixed32(&buf, static_cast<uint32_t>(secondary_handles.size()));

  // 4. secondary_handles
  for (auto* h : secondary_handles) {
    h->EncodeTo(&buf);
  }

  // 5. root_handle
  root_handle.EncodeTo(&buf);

  std::unique_ptr<WritableFile> file;
  EnvOptions env_opts;
  Status s = env->NewWritableFile(file_path_name, &file, env_opts);
  if (!s.ok()) {
    return s;
  }

  s = file->Append(buf);
  if (!s.ok()) {
    return s;
  }

  // fprintf(stderr,
  //         "[WriteModel] Writing learned index model to %s, "
  //         "Model Size = %.2f MB, File Size = %.2f MB\n",
  //         file_path_name.c_str(),
  //         MemoryUsage() / 1024.0 / 1024.0,
  //         buf.capacity() / 1024.0 / 1024.0);

  s = file->Flush();
  if (!s.ok()) {
    return s;
  }

  // 如果 model 是 crash-safe 必须的，保留 Sync
  s = file->Sync();
  if (!s.ok()) {
    return s;
  }

  written = true;

  model_count++;
  model_total_size += buf.capacity();

  return Status::OK();
}

Status LearnedIndexData::ReadModelL3() {
  // fprintf(stderr,
  //         "[ReadModel] Reading learned index model from %s\n",
  //         file_path_name.c_str());

  EnvOptions env_opt;
  std::unique_ptr<SequentialFile> file;
  Status s = env->NewSequentialFile(file_path_name, &file, env_opt);
  if (!s.ok()) {
    return s;
  }

  std::string contents;
  constexpr size_t kReadBufSize = 4 << 20;  // 4MB
  char scratch[kReadBufSize];
  Slice fragment;

  while (true) {
    s = file->Read(kReadBufSize, &fragment, scratch);
    if (!s.ok()) {
      return s;
    }
    if (fragment.empty()) {
      break;
    }
    contents.append(fragment.data(), fragment.size());
  }

  Slice input(contents);

  // 1. pir_handle count
  uint32_t num;
  if (!GetFixed32(&input, &num)) {
    return Status::Corruption("Failed to read pir_handle count");
  }

  pir_handles.clear();
  pir_handles.reserve(num);

  // 2. pir_handles
  for (uint32_t i = 0; i < num; ++i) {
    auto* h = new PIRHandle();
    s = h->DecodeFrom(&input);
    if (!s.ok()) {
      delete h;
      return s;
    }
    pir_handles.push_back(h);
  }

  // 3. secondary count
  if (!GetFixed32(&input, &num)) {
    return Status::Corruption("Failed to read secondary_handle count");
  }

  secondary_handles.clear();
  secondary_handles.reserve(num);

  // 4. secondary_handles
  for (uint32_t i = 0; i < num; ++i) {
    auto* h = new PIRHandle();
    s = h->DecodeFrom(&input);
    if (!s.ok()) {
      delete h;
      return s;
    }
    secondary_handles.push_back(h);
  }

  // 5. root handle
  s = root_handle.DecodeFrom(&input);
  if (!s.ok()) {
    return s;
  }

  // Optional: detect trailing garbage
  if (!input.empty()) {
    return Status::Corruption("Extra bytes at end of model file");
  }

  written = true;

  return Status::OK();
}

size_t LearnedIndexData::MemoryUsageL3() const {
  size_t size = 0;

  // 对象本身
  size += sizeof(LearnedIndexData);

  // pir_handles vector 壳
  size += pir_handles.capacity() * sizeof(PIRHandle*);
  // 每个 PIRHandle
  for (auto& p : pir_handles) {
    size += p->MemoryUsage();
  }

  // secondary_handles vector 壳
  size += secondary_handles.capacity() * sizeof(PIRHandle*);
  for (auto& s : secondary_handles) {
    size += s->MemoryUsage();
  }

  // root_handle
  size += root_handle.TrainedArrayMemoryUsage();

  return size;
}


void DeleteLearnedIndexData(
    rocksdb::Cache::ObjectPtr value,
    rocksdb::MemoryAllocator* /*allocator*/) {
  auto li = static_cast<LearnedIndexData*>(value);
  if (!li->deleted && !li->written) {
    Status s = li->WriteModel();
    if (!s.ok()) {
      fprintf(stderr, "Error writing LearnedIndexData model to file during deletion: %s\n", s.ToString().c_str());
    }
  }
  // printf("[DeleteLearnedIndexData] Deleting learned index model for file number %lu from cache\n",
  //        li->file_number());
  delete static_cast<LearnedIndexData*>(value);
}

const rocksdb::Cache::CacheItemHelper kLearnedIndexCacheHelper(
    rocksdb::CacheEntryRole::kMisc,
    DeleteLearnedIndexData);

void FileLearnedIndexData::FinishModelCache(LearnedIndexData* model) {
  auto number = model->file_number();
  Slice key(reinterpret_cast<const char*>(&number), sizeof(number));

  std::lock_guard<std::mutex> lock(mutex);

  size_t charge = model->MemoryUsage();
  // printf("[FinishModel] Caching learned index model for file number %lu, "
  //         "Model Size = %.2f MB\n",
  //         number,
  //         charge / 1024.0 / 1024.0);

  auto s = model_cache->Insert(
      key,
      model,
      &kLearnedIndexCacheHelper,
      charge);
  if (!s.ok()) {
    printf("Cache Insert failed: %s, charge=%zu\n",
          s.ToString().c_str(), charge);
  }
  printf("[FinishModel] Model Cache Capacity: %.2f MB, Usage: %.2f MB\n",
         model_cache->GetCapacity() / 1024.0 / 1024.0,
         model_cache->GetUsage() / 1024.0 / 1024.0);
}

ModelGuard FileLearnedIndexData::GetModelCache(uint64_t number) {
  Slice key(reinterpret_cast<const char*>(&number), sizeof(number));

  // 1. fast path
  Cache::Handle* handle = model_cache->Lookup(key);
  if (handle != nullptr) {
    // fprintf(stderr, "[GetModelCache] Cache hit for file number %lu\n", number);
    return ModelGuard(model_cache.get(), handle);
  }

  // 2. slow path
  std::lock_guard<std::mutex> lock(mutex);

  // double check
  handle = model_cache->Lookup(key);
  if (handle != nullptr) {
    return ModelGuard(model_cache.get(), handle);
  }

  // create & load
  auto* model = new LearnedIndexData(dbname, number);
  Status s = model->ReadModel();
  if (!s.ok()) {
    delete model;
    return ModelGuard();
  }

  size_t charge = model->MemoryUsage();

  s = model_cache->Insert(
      key,
      model,
      &kLearnedIndexCacheHelper,
      charge,
      &handle);

  if (!s.ok()) {
    delete model;
    return ModelGuard();
  }

  return ModelGuard(model_cache.get(), handle);
}


bool FileLearnedIndexData::DeleteModelCache(uint64_t number) {
  Slice key(reinterpret_cast<const char*>(&number), sizeof(number));

  auto handle = GetModelCache(number);
  handle->deleted = true;

  {
    std::lock_guard<std::mutex> lock(mutex);
    model_cache->Erase(key);
    // 真正 delete 要等 refcount == 0
  }

  std::string pm_fname = ModelFileName(dbname, number);
  Status s = env->DeleteFile(pm_fname);
  if (!s.ok() && !env->FileExists(pm_fname).IsNotFound()) {
    fprintf(stderr,
            "[DeleteModel] Delete pm file %s FAILED -- %s\n",
            pm_fname.c_str(),
            s.ToString().c_str());
    return false;
  }

  return true;
}


void FileLearnedIndexData::FinishModelNoCache(LearnedIndexData* model) {
  auto number = model->file_number();
  std::lock_guard<std::mutex> lock(mutex);

  size_t charge = model->MemoryUsage();
  printf("[FinishModel] Writing learned index model for file number %lu, "
          "Model Size = %.2f MB\n",
          number,
          charge / 1024.0 / 1024.0);

  Status s = model->WriteModel();
  if (!s.ok()) {
    fprintf(stderr, "Error writing LearnedIndexData model to file during deletion: %s\n", s.ToString().c_str());
  }
  delete model;
}

ModelGuard FileLearnedIndexData::GetModelNoCache(uint64_t number) {
  Slice key(reinterpret_cast<const char*>(&number), sizeof(number));

  std::lock_guard<std::mutex> lock(mutex);

  // create & load
  auto* model = new LearnedIndexData(dbname, number);
  Status s = model->ReadModel();
  if (!s.ok()) {
    delete model;
    return ModelGuard();
  }

  size_t charge = model->MemoryUsage();

  Cache::Handle* handle = nullptr;
  s = model_cache->Insert(
      key,
      model,
      &kLearnedIndexCacheHelper,
      charge,
      &handle);

  if (!s.ok()) {
    delete model;
    return ModelGuard();
  }

  return ModelGuard(model_cache.get(), handle);
}


bool FileLearnedIndexData::DeleteModelNoCache(uint64_t number) {
  std::string pm_fname = ModelFileName(dbname, number);
  Status s = env->DeleteFile(pm_fname);
  if (!s.ok() && !env->FileExists(pm_fname).IsNotFound()) {
    fprintf(stderr,
            "[DeleteModel] Delete pm file %s FAILED -- %s\n",
            pm_fname.c_str(),
            s.ToString().c_str());
    return false;
  }

  return true;
}


void FileLearnedIndexData::FinishModelAllInMemory(LearnedIndexData* model) {
  std::lock_guard<std::mutex> lock(mutex);
  
  auto number = model->file_number();
  // create new learned index data if not exist
  if (file_learned_index_data.size() <= static_cast<size_t>(number))
    file_learned_index_data.resize(number + 1, nullptr);
  if (file_learned_index_data[number] != nullptr) {
    fprintf(stderr, "Warning: Overwriting existing LearnedIndexData for file number %lu\n", number);
    delete file_learned_index_data[number];
  }
  file_learned_index_data[number] = model;
}

LearnedIndexData* FileLearnedIndexData::GetModelAllInMemory(uint64_t number) {
  std::lock_guard<std::mutex> lock(mutex);
  // create new learned index data if not exist
  if (file_learned_index_data.size() <= static_cast<size_t>(number))
    return nullptr;
  // directly return if already exist
  return file_learned_index_data[number];
}

bool FileLearnedIndexData::DeleteModelAllInMemory(uint64_t number) {
  std::lock_guard<std::mutex> lock(mutex);
  if (file_learned_index_data.size() <= static_cast<size_t>(number) ||
      file_learned_index_data[number] == nullptr) {
    return false;
  }
  delete file_learned_index_data[number];
  file_learned_index_data[number] = nullptr;

  std::string pm_fname = ModelFileName(dbname, number);
  Status s = env->DeleteFile(pm_fname);
  if (!s.ok() && !env->FileExists(pm_fname).IsNotFound()) {
    fprintf(stderr,
                  "[DeleteModel] Delete pm file %s FAILED -- %s",
                  pm_fname.c_str(),
                  s.ToString().c_str());
    return false;
  }
  return true;
}

}
