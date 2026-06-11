#include "my_trace.h"

#ifdef DEBUG_TRACE
std::array<TypeStats, MAX_TYPE> g_phase_stats;

// CPU频率，用于将周期转为时间
double tsc_freq_ghz = 3.0;
// LocalOp创建开销
uint64_t local_op_create_overhead_cycles = 28;
// begin()调用开销
uint64_t begin_overhead_cycles = 87;
// phase_record()调用开销
uint64_t record_phase_overhead_cycles = 133;
// commit()调用开销
uint64_t commit_overhead_cycles = 1373;

const char* type_strs[MAX_TYPE] = {
    "GetTime",
    "TableGetFound",
    "TableGetNotFound",
    "IndexBlockTime",
    "LIIndexBlockTime",
    "DataBlockTime",
    "LIDataBlockTime",
    "ReadBlockCacheHit",
    "ReadBlockCacheMiss",
    "ReadBlock",
    "ReadDataBlock",
    "ReadFile",
    "ReadFileLoop",
    "ReadRandomAccessFile",
    "AddToBlockBuilder",
    "BlockBuilderFinish",
    "FlushTotalTime",
    "FlushTime",
    "FlushBuildTableTime",
    "SSTBuildTime",
    "CompactionSSTBuildTime",
    "LearnedIndexBuildTime",
    "LearnedIndexTrainTime",
    "LearnedIndexPersistTime",
    "FlushAddKey",
    "CompactionTotalTime",
    "CompactionTime",
    "CompactionAddKey",
    "TableBuilderFlush",
    "TableBuilderWriteBlock",
    "TableBuilderWriteBlock2",
    "TableBuilderWriteBlock3",
    "TableBuilderAdd"
};

const char *get_time_strs[MAX_PHASES] = {
    "Prepare",
    "Read Memtable",
    "Read LSMTree",
    "After Read"
};

const char *table_get_strs[MAX_PHASES] = {
    "Table Get"
};

const char *index_block_time_strs[MAX_PHASES] = {
    "IndexIter Create",
    "IndexIter Seek",
    "Before DataBlock"
};

const char *li_index_block_time_strs[MAX_PHASES] = {
    "Get LI Model",
    "Seek First PIRHandle",
    "Seek Loop"
};

const char *data_block_time_strs[MAX_PHASES] = {
    "DataBlockIter Create",
    "DataBlockIter Seek",
    "After Seek"
};

const char *read_block_phases_strs[MAX_PHASES] = {
    "cache read prepare", "get from cache", "cache hit handle", "block read prepare", "block read", "stats & compression", "put block in cache", "finish trace"
};

const char* read_block_from_disk_phases_strs[MAX_PHASES] = {
    "prepare io options",
    "read block from file",
    "after read"
};

const char* read_file_phases_strs[MAX_PHASES] = {
    "read prepare",
    "file read",
    "after read"
};

const char *read_file_loop_phases_strs[MAX_PHASES] = {
    "per read loop"
};

const char *read_random_access_file_phases_strs[MAX_PHASES] = {
    "Pread Time",
};

const char *add_to_block_builder_phases_strs[MAX_PHASES] = {
    "PIR Time",
    "Origin Add Time"
};

const char *block_builder_finish_phases_strs[MAX_PHASES] = {
    "Finish Block",
    "Training PIR"
};

const char *flush_total_time_strs[MAX_PHASES] = {
    "Flush Total Time"
};

const char *flush_time_strs[MAX_PHASES] = {
    "Build Table",
    "Log Flush",
    "Write File"
};

const char *flush_build_table_time_strs[MAX_PHASES] = {
    "Prepare",
    "New Table",
    "Setup Compaction Iterator",
    "Add Keys",
    "Finish Table",
    "Close File"
};

const char *flush_add_key_strs[MAX_PHASES] = {
    "Add Key Time"
};

const char *sst_build_time_strs[MAX_PHASES] = {
    "Prepare",
    "Add Keys",
    "Finish Table",
    "Sync Close"
};

const char *compaction_sst_build_time_strs[MAX_PHASES] = {
    "Compaction SST Build Time"
};

const char *learned_index_build_time_strs[MAX_PHASES] = {
    "Learned Index Build Time"
};

const char *learned_index_train_time_strs[MAX_PHASES] = {
    "Learned Index Train Time"
};

const char *learned_index_persist_time_strs[MAX_PHASES] = {
    "Learned Index Persist Time"
};

const char *compaction_total_time_strs[MAX_PHASES] = {
    "Compaction Total Time"
};

const char *compaction_time_strs[MAX_PHASES] = {
    "Setup",
    "CIter Seek to First",
    "Set File Func",
    "Write KV",
    "Write File"
};

const char *compaction_add_key_strs[MAX_PHASES] = {
    "Before Add Key",
    "Add Key Time",
    "After Add Key"
};

const char *table_builder_flush_strs[MAX_PHASES] = {
    "Flush Time",
    "After Flush"
};

const char *table_builder_write_block_strs[MAX_PHASES] = {
    "Block Finish",
    "Swap & Reset",
    "Write Block"
};

const char *table_builder_write_block_2_strs[MAX_PHASES] = {
    "Compress Block",
    "Write Maybe Compressed Block",
    "After Write Block"
};

const char *table_builder_write_block_3_strs[MAX_PHASES] = {
    "Prepare IO Options",
    "Set Block Handle",
    "File Append",
    "Write & Append Trailer",
    "After Write Block"
};

const char *table_builder_add_strs[MAX_PHASES] = {
    "Before Add",
    "Add Time",
    "After Add"
};


const char** type_phases_strs[MAX_TYPE] = {
    get_time_strs,
    table_get_strs,
    table_get_strs,
    index_block_time_strs,
    li_index_block_time_strs,
    data_block_time_strs,
    data_block_time_strs,
    read_block_phases_strs,
    read_block_phases_strs,
    read_block_from_disk_phases_strs,
    read_block_from_disk_phases_strs,
    read_file_phases_strs,
    read_file_loop_phases_strs,
    read_random_access_file_phases_strs,
    add_to_block_builder_phases_strs,
    block_builder_finish_phases_strs,
    flush_total_time_strs,
    flush_time_strs,
    flush_build_table_time_strs,
    sst_build_time_strs,
    compaction_sst_build_time_strs,
    learned_index_build_time_strs,
    learned_index_train_time_strs,
    learned_index_persist_time_strs,
    flush_add_key_strs,
    compaction_total_time_strs,
    compaction_time_strs,
    compaction_add_key_strs,
    table_builder_flush_strs,
    table_builder_write_block_strs,
    table_builder_write_block_2_strs,
    table_builder_write_block_3_strs,
    table_builder_add_strs
};

// inline uint64_t rdtsc_ordered() {
//     unsigned eax, edx;
//     asm volatile("cpuid" ::: "rax","rbx","rcx","rdx");
//     asm volatile("rdtsc" : "=a"(eax), "=d"(edx));
//     return ((uint64_t)edx << 32) | eax;
// }

// double measure_tsc_freq_precise() {
//     using namespace std::chrono;
//     auto t0 = rdtsc_ordered();
//     auto c0 = steady_clock::now();

//     std::this_thread::sleep_for(std::chrono::seconds(1));

//     auto t1 = rdtsc_ordered();
//     auto c1 = steady_clock::now();

//     double dt = duration_cast<duration<double>>(c1 - c0).count();
//     return (t1 - t0) / dt / 1e9;  // GHz
// }

double measure_tsc_freq_precise() {
    using namespace std::chrono;

    timespec ts1, ts2;
    uint64_t t1, t2;
    unsigned aux;

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
    t1 = __rdtscp(&aux);

    std::this_thread::sleep_for(milliseconds(200));

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts2);
    t2 = __rdtscp(&aux);

    double ns =
        (ts2.tv_sec - ts1.tv_sec) * 1e9 +
        (ts2.tv_nsec - ts1.tv_nsec);

    return (t2 - t1) / ns; // cycles per ns
}


void InitTraceStats() {
    tsc_freq_ghz = measure_tsc_freq_precise();
    printf("Measured TSC frequency: %.3f GHz\n", tsc_freq_ghz);
    uint64_t t1;
    uint64_t t2;
    uint64_t diff = 16;
    int check_loops = 10000;
    for (int i = 0; i < check_loops; ++i) {
        t1 = rdtscp();
        t2 = rdtscp();
        diff = std::min(diff, t2 - t1);
    }
    local_op_create_overhead_cycles = 10;
    begin_overhead_cycles = 22;
    record_phase_overhead_cycles = 22;
    commit_overhead_cycles = 100;
    printf("Measured rdtscp overhead: %lu cycles\n", diff);
    for (int i = 0; i < check_loops; ++i) {
        t1 = rdtscp();
        asm volatile("" ::: "memory");
        LocalOp op;
        asm volatile("" ::: "memory");
        t2 = rdtscp();
        local_op_create_overhead_cycles = std::min(local_op_create_overhead_cycles, t2 - t1 - diff);

        t1 = rdtscp();
        op.begin();
        t2 = rdtscp();
        begin_overhead_cycles = std::min(begin_overhead_cycles, t2 - t1 - diff);

        t1 = rdtscp();
        op.record_phase();
        t2 = rdtscp();
        record_phase_overhead_cycles = std::min(record_phase_overhead_cycles, t2 - t1 - diff);

        op.record_phase();
        op.record_phase();
        op.record_phase();

        t1 = rdtscp();
        op.commit(TEST_TYPE);
        t2 = rdtscp();
        commit_overhead_cycles = std::min(commit_overhead_cycles, t2 - t1 - diff);
    }

    printf("Measured LocalOp create overhead: %lu cycles\n", local_op_create_overhead_cycles);
    printf("Measured LocalOp begin() overhead: %lu cycles\n", begin_overhead_cycles);
    printf("Measured LocalOp record_phase() overhead: %lu cycles\n", record_phase_overhead_cycles);
    printf("Measured LocalOp commit() overhead: %lu cycles\n", commit_overhead_cycles);

    for (size_t i = 0; i < MAX_TYPE; ++i) {
        g_phase_stats[i].reset();
    }
}

#else

// void InitTraceStats() {}

#endif
