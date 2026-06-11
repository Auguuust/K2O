# Learned Index Implementation for RocksDB

## Overview

This implementation integrates learned indexes into RocksDB following the approach from Google's paper:
["Learned Indexes for a Google-scale Disk-based Database"](https://arxiv.org/pdf/2012.12501)

## Key Innovation

Unlike traditional approaches where learned indexes are trained **after** data is written, this implementation follows Google's key insight:

> **Use the learned model to *define* which records go in which block, rather than learning where they are.**

This ensures:
- ✅ Model predictions are always 100% correct (zero error)
- ✅ Significantly smaller index size (model + block pointers vs full B-tree)
- ✅ Fewer disk reads for index lookups
- ✅ Better cache efficiency

## Architecture

### Core Components

1. **LearnedIndex** (`learned_index.h/cc`)
   - `KeyEncoder`: Converts string keys to monotonic double values
   - `LinearModel`: Simple linear regression (f(x) = slope * x + intercept)
   - Training on cumulative byte offsets

2. **LearnedBlockBasedTableBuilder** (`learned_block_based_table_builder.h/cc`)
   - Extends `BlockBasedTableBuilder`
   - Two-pass approach:
     - **Pass 1**: Collect all keys and cumulative bytes
     - **Pass 2**: Train model, write blocks guided by predictions

3. **BlockLocationMapper** (`learned_index.h/cc`)
   - Compressed storage of block locations
   - Maps predicted block number → disk offset + size

### How It Works

```
Traditional Approach:            Google's Learned Index Approach:
─────────────────────           ────────────────────────────────
1. Write data to blocks         1. Collect all keys + byte offsets
   (fill to fixed size)         2. Train model: key → byte offset
2. Build index after            3. Predict block = offset / target_size
3. Index learns where           4. Write data to match predictions
   data was placed              5. Model is ALWAYS correct!
```

### Training Process

```python
# Pseudocode
for each (key, value) in data:
    cumulative_bytes += sizeof(key) + sizeof(value)
    training_data.append((key, cumulative_bytes))

# Train linear model: encoded_key → predicted_bytes
model = OLS_regression(training_data)

# Use model to guide block boundaries
for each (key, value):
    predicted_block = model.predict(encode(key)) / target_block_size
    if predicted_block != current_block:
        flush_current_block()
        start_new_block()
    add_to_current_block(key, value)
```

## Usage

### Enable in Options

```cpp
#include "rocksdb/table.h"

rocksdb::Options options;
rocksdb::BlockBasedTableOptions table_options;

// Enable learned index
table_options.enable_learned_index = true;

// Configure target block size (default: 4KB)
table_options.learned_index_target_block_size = 4 * 1024;

// Minimum keys required for training (default: 100)
table_options.learned_index_min_training_keys = 100;

options.table_factory.reset(
    rocksdb::NewBlockBasedTableFactory(table_options));

// Open DB with learned index enabled
rocksdb::DB* db;
rocksdb::Status s = rocksdb::DB::Open(options, "/path/to/db", &db);
```

### Using the Learned Table Builder Directly

```cpp
#include "table/block_based/learned_block_based_table_builder.h"

// Create builder with learned index enabled
auto builder = new LearnedBlockBasedTableBuilder(
    table_options,
    table_builder_options,
    file,
    true  // enable_learned_index
);

// Add keys (training data is collected automatically)
for (const auto& [key, value] : data) {
    builder->Add(key, value);
}

// Finish building (trains model and writes blocks)
Status s = builder->Finish();

// Get statistics
auto stats = builder->GetLearnedIndexStats();
std::cout << "Model slope: " << stats.model_slope << std::endl;
std::cout << "Predicted blocks: " << stats.predicted_num_blocks << std::endl;
```

## Performance Benefits

Based on Google's paper, learned indexes provide:

| Metric | Improvement |
|--------|-------------|
| **Point Lookup Latency (p99)** | ↓ 38% |
| **Point Lookup Latency (mean)** | ↓ 36% |
| **Scan Latency (mean)** | ↓ 22% |
| **Point Lookup Throughput (p99)** | ↑ 54% |
| **Scan Throughput (mean)** | ↑ 28% |

### Why Such Improvements?

1. **Smaller Index Size**: Model + block pointers << full B-tree
2. **Fewer Index Blocks**: Often fits in 1 block vs 2-level tree
3. **Better Cache Hit Rate**: Less index data to cache
4. **Reduced Decompression**: Fewer blocks to decompress

## Comparison with Other Approaches

| Approach | Data Placement | Model Accuracy | Implementation |
|----------|---------------|----------------|----------------|
| **Bourbon/LearnedLevelDB** | Traditional (fill blocks) | Has error bounds | Passive learning |
| **Google (This)** | Model-guided | Always correct | Active design |
| **Traditional B-Tree** | N/A | Always correct | Large index |

## Implementation Details

### Key Encoding

String keys are converted to monotonic doubles:

```cpp
// Example: keys "aab", "bdd", "bcb"
// Position 0: range [a,b] = 2 values → base = 4*3 = 12
// Position 1: range [a,d] = 4 values → base = 3
// Position 2: range [b,d] = 3 values → base = 1

double encode("bcb") = 1*12 + 2*3 + 0*1 = 18
```

### Model Serialization

```
[trained_flag:1] [target_block_size:8] 
[slope:8] [intercept:8] 
[max_key_length:varint] [char_min/max pairs...]
```

### Block Location Mapping

```
[num_blocks:4]
[offset_0:8] [size_0:4]
[offset_1:8] [size_1:4]
...
```

## Testing

```bash
# Build with learned index support
make learned_index_test

# Run tests
./learned_index_test

# Benchmark comparison
./db_bench --enable_learned_index=true \
           --benchmarks=fillseq,readrandom \
           --learned_index_target_block_size=4096
```

## Limitations and Future Work

### Current Limitations

1. **Two-pass approach**: Requires collecting all data before writing
   - Could be optimized with online learning
2. **Fixed target block size**: Could adapt based on workload
3. **String keys only**: Optimized for string keys
   - Could extend to fixed-size keys

### Future Enhancements

1. **Adaptive block sizing**: Learn optimal block sizes per region
2. **Hierarchical models**: RMI-style multi-level models
3. **Online learning**: Update model during compaction
4. **Compression-aware**: Account for compression ratios

## References

1. [Learned Indexes for a Google-scale Disk-based Database](https://arxiv.org/pdf/2012.12501)
   - Abu-Libdeh et al., 2020

2. [From WiscKey to Bourbon: A Learned Index for LSM Trees](https://www.usenix.org/system/files/osdi20-dai_0.pdf)
   - Dai et al., OSDI 2020

3. [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)
   - Kraska et al., SIGMOD 2018

## Authors

Implementation based on Google's approach for RocksDB integration.

## License

Same as RocksDB (GPLv2 and Apache 2.0)
