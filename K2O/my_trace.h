#pragma once
#include <x86intrin.h>
#include <cstdint>
#include <array>
#include <atomic>
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <fstream>

#define DEBUG_TRACE 1  // 打开统计，默认关闭

enum OpTypes {
    kGetTime,
    kTableGetFound,
    kTableGetNotFound,
    kIndexBlockTime,
    kLIIndexBlockTime,
    kDataBlockTime,
    kLIDataBlockTime,
    kReadBlockCacheHit,
    kReadBlockCacheMiss,
    kReadBlock,
    kReadDataBlock,
    kReadFile,
    kReadFileLoop,
    kReadRandomAccessFile,
    kAddToBlockBuilder,
    kBlockBuilderFinish,
    kFlushTotalTime,
    kFlushTime,
    kFlushBuildTableTime,
    kSSTBuildTime,
    kCompactionSSTBuildTime,
    kLearnedIndexBuildTime,
    kLearnedIndexTrainTime,
    kLearnedIndexPersistTime,
    kFlushAddKey,
    kCompactionTotalTime,
    kCompactionTime,
    kCompactionAddKey,
    kTableBuilderFlush,
    kTableBuilderWriteBlock,
    kTableBuilderWriteBlock2,
    kTableBuilderWriteBlock3,
    kTableBuilderAdd,
    TEST_TYPE,
    MAX_TYPE
};

#ifdef DEBUG_TRACE

inline uint64_t rdtscp() {
    unsigned aux;
    return __rdtscp(&aux);
}

extern double tsc_freq_ghz;
extern uint64_t local_op_create_overhead_cycles;
extern uint64_t begin_overhead_cycles;
extern uint64_t record_phase_overhead_cycles;
extern uint64_t commit_overhead_cycles;

// 最大阶段数
constexpr size_t MAX_PHASES = 16;

constexpr size_t MAX_HIT_TYPE = 4;

extern const int type_include_other_types[MAX_TYPE];

extern const char *type_strs[MAX_TYPE];
// 存储各类型使用的字符串的数组指针数组
extern const char **type_phases_strs[MAX_TYPE];

// phase统计结构
struct PhaseStats {
    std::atomic<uint64_t> total_cycles{0};
    std::atomic<uint64_t> count{0};

    double avg_us() const {
        uint64_t cnt = count.load(std::memory_order_relaxed);
        if (cnt == 0) return 0.0;
        double rtn = (total_cycles.load(std::memory_order_relaxed) * 1.0 / tsc_freq_ghz) / cnt / 1000.0;
        return rtn > 0 ? rtn : 0.0;
    }

    void reset() {
        total_cycles.store(0, std::memory_order_relaxed);
        count.store(0, std::memory_order_relaxed);
    }
};

// 类型统计结构
struct TypeStats {
    std::array<PhaseStats, MAX_PHASES> phase_stats;
    std::atomic<uint64_t> count{0};  // 该类型总次数
    std::atomic<uint64_t> loop_count{0}; // 该类型总循环次数，仅对循环类型有效
    double avg_total_us() const {
        uint64_t cnt = count.load(std::memory_order_relaxed);
        if (cnt == 0) return 0.0;
        double total = 0;
        for (size_t i = 0; i < MAX_PHASES; ++i) {
            if (phase_stats[i].count.load(std::memory_order_relaxed) == 0) continue;
            total += phase_stats[i].total_cycles.load(std::memory_order_relaxed);
        }
        double rtn = (total * 1.0 / tsc_freq_ghz) / cnt / 1000.0;
        return rtn > 0 ? rtn : 0.0;
    }

    void reset() {
        for (size_t i = 0; i < MAX_PHASES; ++i) {
            phase_stats[i].reset();
        }
        count.store(0, std::memory_order_relaxed);
        loop_count.store(0, std::memory_order_relaxed);
    }
};

// 全局统计对象
extern std::array<TypeStats, MAX_TYPE> g_phase_stats;

// ---------------- LocalOp ----------------
// 用于一次操作中收集phase，type未知时也可先统计
struct LocalOp {
    uint64_t last_ts = 0;             
    uint64_t phase_cycles[MAX_PHASES] = {0};
    size_t max_phase_idx = 0;
    size_t loop_count = 0;

    void begin() { last_ts = rdtscp(); }

    void loop_begin() { loop_count++; }

    // 自动记录当前phase耗时
    void record_phase() {
        uint64_t now = rdtscp();
        if (max_phase_idx < MAX_PHASES) {
            phase_cycles[max_phase_idx++] = now - last_ts;
        }
        last_ts = now;
    }

    // 手动记录某个phase耗时
    void record_phase(size_t phase) {
        uint64_t now = rdtscp();
        if (phase < MAX_PHASES) {
            phase_cycles[phase] += now - last_ts;
            if (phase >= max_phase_idx) max_phase_idx = phase + 1;
        }
        last_ts = now;
    }

    // type确定后提交到全局统计
    void commit(int type) {
        TypeStats &type_stat = g_phase_stats[type];
        // printf("LocalOp commit type=%d with max_phase_idx=%zu\n", type, max_phase_idx);
        type_stat.count.fetch_add(1, std::memory_order_relaxed);
        type_stat.loop_count.fetch_add(loop_count, std::memory_order_relaxed);
        for (size_t i = 0; i < max_phase_idx; ++i) {
            // if (phase_cycles[i] == 0) continue;  // 有大量数据只通过短路径时，会使得长路径的平均值失真，必须所有的计数都统一
            type_stat.phase_stats[i].total_cycles.fetch_add(phase_cycles[i], std::memory_order_relaxed);
            type_stat.phase_stats[i].count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    double total_us() const {
        double total = 0.0f;
        for (size_t phase = 0; phase < max_phase_idx; ++phase) {
            total += (phase_cycles[phase] * 1.0 / tsc_freq_ghz) / 1000.0;
        }
        return total;
    }

    void dump(std::string prefix = "") {
        double total_us = 0.0f;
        for (size_t phase = 0; phase < MAX_PHASES; ++phase) {
            uint64_t cyc = phase_cycles[phase];
            if (cyc == 0) continue;

            double phase_us = (cyc * 1.0 / tsc_freq_ghz) / 1000.0;
            total_us += phase_us;
            printf("%s Phase %ld: %.2fus\n", prefix.c_str(), phase, phase_us);
        }
        printf("%s total_us: %.2f\n", prefix.c_str(), total_us);
    }
};

void InitTraceStats();

// ---------------- dump_stats ----------------
inline void dump_stats() {
    for (size_t type = 0; type < TEST_TYPE; ++type) {
        const TypeStats &type_stat = g_phase_stats[type];
        uint64_t type_cnt = type_stat.count.load();
        if (type_cnt == 0) {
            printf("==== Type %s: no data ====\n", type_strs[type]);
            continue;
        }
        const char** phases_strs = type_phases_strs[type];

        if (!phases_strs) {
            std::cout << "Warning: type_phases_strs[" << type << "] is nullptr\n";
            continue;
        }

        std::cout << "==== Type " << type_strs[type] << " (total count=" << type_cnt << ") ====\n";
        std::cout << "Loop count=" << type_stat.loop_count.load() << "\n";
        for (size_t phase = 0; phase < MAX_PHASES; ++phase) {
            const PhaseStats &ps = type_stat.phase_stats[phase];
            uint64_t cnt = ps.count.load();
            uint64_t cyc = ps.total_cycles.load();
            if (cnt == 0) continue;

            double avg_us = (cyc * 1.0 / tsc_freq_ghz) / cnt / 1000.0;
            double total_ms = cyc * 1.0 / tsc_freq_ghz / 1e6;
            const char* phase_name = (phases_strs[phase] ? phases_strs[phase] : "UNKNOWN");
            std::cout << "Phase " << phase << " (" << phase_name << "): "
                      << " count=" << cnt
                      << " avg=" << avg_us << " us"
                      << " total=" << total_ms << " ms\n";
        }
    }
}

inline void dump_json(const char* filename) {
    std::ofstream avg_out(filename);
    avg_out << "{\n";
    
    auto print_avg = [&](const char* name, double avg, bool end = false) {
        avg_out << "    \"" << name << "\": ";
        avg_out << std::fixed << std::setprecision(3) << avg;
        if (!end) avg_out << ",\n";
        else      avg_out << "\n";
    };

    bool first = true;
    for (size_t type = 0; type < TEST_TYPE; type++) {
        const auto& p = g_phase_stats[type];
        if(p.count.load() == 0) continue;
        const char** phases_strs = type_phases_strs[type];
        if (!first) avg_out << ",\n";
        first = false;

        avg_out << "  \"" << type_strs[type] << "\": {\n";
        avg_out << "    \"count\": " << p.count.load() << ",\n";
        if (p.loop_count.load() > 0) {
        print_avg("avg_loop", (double)p.loop_count.load() / p.count.load());
        }

        for (size_t j = 0; j < MAX_PHASES; j++) {
        const auto& phase = p.phase_stats[j];
        if (phase.count.load() == 0) continue;
        print_avg(phases_strs[j], phase.avg_us());
        }
        print_avg("total", p.avg_total_us(), /*end=*/true);
        avg_out << "  }";
    }

    avg_out << "\n}\n";
}

#else
// ---------------- 空操作版本 ----------------
inline uint64_t rdtscp() { return 0; }

struct LocalOp {
    void begin() {}
    void loop_begin() {}
    void record_phase() {}
    void record_phase(size_t /*phase*/) {}
    void commit(int /*type*/) {}
    void dump(std::string /*prefix*/) {}
};

inline void InitTraceStats() {}

inline void dump_stats() {}
inline void dump_json(const char* /*filename*/) {}
#endif
