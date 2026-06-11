# Integration Guide: Learned Index in RocksDB

## Quick Start

### 1. Build with Learned Index Support

Add the new source files to your build:

```bash
# Add to your Makefile or CMakeLists.txt
SOURCES += \
    table/block_based/learned_index.cc \
    table/block_based/learned_block_based_table_builder.cc

# Build
make static_lib
make db_bench
```

### 2. Enable in Your Application

```cpp
#include "rocksdb/db.h"
#include "rocksdb/table.h"

int main() {
    rocksdb::DB* db;
    rocksdb::Options options;
    
    // Configure learned index
    rocksdb::BlockBasedTableOptions table_options;
    table_options.enable_learned_index = true;
    table_options.learned_index_target_block_size = 4 * 1024;  // 4KB
    table_options.learned_index_min_training_keys = 100;
    
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));
    
    // Open database
    rocksdb::Status s = rocksdb::DB::Open(options, "/tmp/testdb", &db);
    assert(s.ok());
    
    // Use database normally
    s = db->Put(rocksdb::WriteOptions(), "key1", "value1");
    
    std::string value;
    s = db->Get(rocksdb::ReadOptions(), "key1", &value);
    
    delete db;
    return 0;
}
```

### 3. Benchmark Comparison

```bash
# Baseline (traditional blocks)
./db_bench \
    --benchmarks=fillseq,readrandom,readseq \
    --num=1000000 \
    --key_size=16 \
    --value_size=100 \
    --block_size=4096 \
    --enable_learned_index=false

# With learned index
./db_bench \
    --benchmarks=fillseq,readrandom,readseq \
    --num=1000000 \
    --key_size=16 \
    --value_size=100 \
    --learned_index_target_block_size=4096 \
    --enable_learned_index=true
```

## Advanced Configuration

### Tuning Parameters

```cpp
// For small SSTables (<1MB)
table_options.learned_index_min_training_keys = 50;
table_options.learned_index_target_block_size = 2 * 1024;  // 2KB blocks

// For large SSTables (>100MB)
table_options.learned_index_min_training_keys = 1000;
table_options.learned_index_target_block_size = 8 * 1024;  // 8KB blocks

// For string keys with high variability
table_options.learned_index_target_block_size = 4 * 1024;  // Standard

// For numeric/timestamp keys
table_options.learned_index_target_block_size = 4 * 1024;  // Works well
```

### Workload-Specific Settings

```cpp
// Read-heavy workload (optimize for lookup speed)
table_options.enable_learned_index = true;
table_options.cache_index_and_filter_blocks = true;
table_options.pin_top_level_index_and_filter = true;

// Write-heavy workload (balance write amplification)
table_options.enable_learned_index = true;
table_options.learned_index_target_block_size = 8 * 1024;  // Larger blocks

// Scan-heavy workload
table_options.enable_learned_index = true;
table_options.learned_index_target_block_size = 16 * 1024;  // Fewer blocks
```

## Monitoring and Debugging

### Enable Detailed Logging

```cpp
options.info_log_level = rocksdb::InfoLogLevel::DEBUG_LEVEL;

// In your code
auto stats = builder->GetLearnedIndexStats();
if (stats.trained) {
    LOG(INFO) << "Learned index trained:";
    LOG(INFO) << "  Training samples: " << stats.num_training_samples;
    LOG(INFO) << "  Predicted blocks: " << stats.predicted_num_blocks;
    LOG(INFO) << "  Model slope: " << stats.model_slope;
    LOG(INFO) << "  Model intercept: " << stats.model_intercept;
}
```

### Statistics to Monitor

```cpp
// Check index effectiveness
rocksdb::ColumnFamilyMetaData cf_meta;
db->GetColumnFamilyMetaData(&cf_meta);

for (const auto& level : cf_meta.levels) {
    for (const auto& file : level.files) {
        // Compare file size to number of blocks
        // Smaller index = better learned index effectiveness
    }
}
```

## Migration from Traditional Blocks

### Step 1: Test with New SSTables Only

```cpp
// Keep existing SSTables with traditional indexes
// New SSTables use learned index
options.table_factory = learned_table_factory;
```

### Step 2: Gradual Rollout

```cpp
// Use percentage-based rollout
bool use_learned = (file_number % 10) < 3;  // 30% of new files
if (use_learned) {
    table_options.enable_learned_index = true;
}
```

### Step 3: Full Migration

```cpp
// After validation, enable for all new SSTables
table_options.enable_learned_index = true;

// Trigger full compaction to migrate old files
db->CompactRange(CompactRangeOptions(), nullptr, nullptr);
```

## Troubleshooting

### Common Issues

**Issue**: Model training fails
```
Solution: Check learned_index_min_training_keys setting
- Reduce minimum if you have small SSTables
- Ensure keys are diverse enough
```

**Issue**: Performance worse than traditional
```
Solution: Check your workload characteristics
- Very small SSTables (<100KB) may not benefit
- Random string keys work best
- Sequential integer keys might not benefit
```

**Issue**: Index size larger than expected
```
Solution: Verify block locations are being compressed
- Check BlockLocationMapper implementation
- Ensure target_block_size is appropriate
```

### Debug Helpers

```cpp
// Add assertions to catch issues early
#ifndef NDEBUG
void ValidateLearnedIndex(const LearnedIndex& index,
                         const std::vector<std::string>& keys) {
    uint32_t prev_block = 0;
    for (const auto& key : keys) {
        uint32_t block = index.PredictBlock(key);
        assert(block >= prev_block);  // Monotonicity check
        prev_block = block;
    }
}
#endif
```

## Performance Expectations

Based on Google's paper, you should see:

| Metric | Expected Improvement |
|--------|---------------------|
| Index Size | 50-90% reduction |
| Read Latency (p99) | 30-40% faster |
| Read Throughput | 50-60% higher |
| Scan Performance | 20-30% faster |
| Write Performance | ~No change |

### When You Won't See Benefits

- Very small SSTables (<100KB)
- Highly random, non-monotonic keys
- Already cached indexes (no disk I/O)
- SSD with very fast random access

### When You Will See Best Results

- Large SSTables (>10MB)
- String keys with natural ordering
- Read-heavy workloads
- HDD or slower storage
- Large databases where index doesn't fit in cache

## Example: Complete Integration

```cpp
#include <iostream>
#include "rocksdb/db.h"
#include "rocksdb/table.h"
#include "rocksdb/statistics.h"

int main() {
    // Create options
    rocksdb::Options options;
    options.create_if_missing = true;
    options.statistics = rocksdb::CreateDBStatistics();
    
    // Configure learned index
    rocksdb::BlockBasedTableOptions table_options;
    table_options.enable_learned_index = true;
    table_options.learned_index_target_block_size = 4 * 1024;
    table_options.learned_index_min_training_keys = 100;
    
    // Optional: enable caching for better performance
    table_options.cache_index_and_filter_blocks = true;
    table_options.pin_top_level_index_and_filter = true;
    
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));
    
    // Open database
    rocksdb::DB* db;
    rocksdb::Status s = rocksdb::DB::Open(options, "/tmp/learned_index_db", &db);
    if (!s.ok()) {
        std::cerr << "Failed to open database: " << s.ToString() << std::endl;
        return 1;
    }
    
    // Write data
    rocksdb::WriteOptions write_opts;
    for (int i = 0; i < 100000; ++i) {
        char key[32], value[100];
        snprintf(key, sizeof(key), "user%08d", i);
        snprintf(value, sizeof(value), "value_%d", i);
        s = db->Put(write_opts, key, value);
        if (!s.ok()) {
            std::cerr << "Write failed: " << s.ToString() << std::endl;
        }
    }
    
    // Force flush to create SSTable
    s = db->Flush(rocksdb::FlushOptions());
    
    // Read data
    rocksdb::ReadOptions read_opts;
    std::string value;
    for (int i = 0; i < 10000; ++i) {
        char key[32];
        snprintf(key, sizeof(key), "user%08d", i);
        s = db->Get(read_opts, key, &value);
        if (!s.ok()) {
            std::cerr << "Read failed: " << s.ToString() << std::endl;
        }
    }
    
    // Print statistics
    std::cout << options.statistics->ToString() << std::endl;
    
    delete db;
    return 0;
}
```

## Next Steps

1. **Test in Staging**: Deploy to a non-production environment first
2. **Monitor Metrics**: Compare index sizes and read latencies
3. **Tune Parameters**: Adjust target_block_size based on your data
4. **Gradual Rollout**: Enable for a subset of traffic first
5. **Full Migration**: After validation, enable globally

## Support

For issues or questions:
- Check the README: `LEARNED_INDEX_README.md`
- Run tests: `./learned_index_test`
- Review the paper: https://arxiv.org/pdf/2012.12501
