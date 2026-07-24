# K2O: Precise Key-to-Offset Learned Indexing for Read-Efficient LSM-Tree KV Stores

## Project Overview

This repository is a prototype implementation of the paper **K2O: Precise Key-to-Offset Learned Indexing for Read-Efficient LSM-Tree KV Stores**.

The repository contains the following methods:

| Method in Paper | Directory |
|-----------------|-----------|
| K2O | `K2O/` |
| GoogleLI | `Baselines/rocksdb-google/` |
| LeaderKV | `Baselines/rocksdb-leader/` |

## Artifact Documentation

The artifact documentation is provided as separate root-level files:

- [REQUIREMENTS](./REQUIREMENTS): operating system, dependencies, resources,
  privileges, and workload inputs;
- [INSTALL](./INSTALL): dependency installation, compilation, smoke testing,
  and full-experiment setup;
- [STATUS](./STATUS): implemented scope, current limitations, and expected
  outputs;
- [LICENSE](./LICENSE): repository-level license notice and links to the
  complete license texts.

All methods use the same directory structure and share the same build and
benchmark workflow. In the instructions below, `/path/to/<method>` refers to
the directory of the method you want to evaluate. To switch methods, only
replace `<method>` with the corresponding directory path shown above.

## System Requirements

### Swap Memory
- **Recommended Swap Size**: >= 8 GB

### Dependencies
- **Operating System**: Linux (Ubuntu 20.04+ or CentOS 7+)
- **Build Tools**:
  - CMake >= 3.10
  - GCC or Clang (C++17 support)
  - Make >= 4.0
- **Libraries**:
  - zlib-devel / libz-dev
  - snappy-devel / libsnappy-dev
  - lz4-devel / liblz4-dev
  - zstd-devel / libzstd-dev
  - gflags-devel / libgflags-dev
  - bzip2-devel / libbz2-dev
  - pthread (usually built-in)

### System Privileges
- Requires `sudo` privileges for clearing system page cache:
  ```bash
  sudo tee /proc/sys/vm/drop_caches
  ```

## Build Instructions

### 1. Navigate to build directory
```bash
cd /path/to/<method>/build
```

### 2. Clean old build files (if needed)
```bash
rm -rf CMakeCache.txt CMakeFiles
```

### 3. Configure and build
```bash
cmake ..
make db_bench -j$(nproc)
```

After compilation, the executable `db_bench` will be generated in the
`/path/to/<method>/build/` directory.

## Running Guide

### Basic Usage

Run tests from the selected method directory via `build/test.py`:

```bash
cd /path/to/<method>
python ./build/test.py
```

`test.py` automatically:
1. Compiles `db_bench` in the selected method's `build/` directory (if needed)
2. Cleans up old database
3. Runs specified benchmark tests
4. Collects and saves results to the log directory

### Configuration Parameters

Edit the following parameters in `build/test.py` to adjust test behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DB_PATH` | `/mnt/n1/dbpath/` | Database storage path |
| `WORKLOAD_PATH` | `../ycsb_workload` | YCSB workload files directory |
| `EXP_ROOT` | `../logs` | Experiment logs output root |
| `LOAD_NUM` | 64000000 | Number of operations in YCSB load phase |
| `RUNNING_NUM` | 10000000 | Number of operations in YCSB run phase |
| `FILL_NUM` | 64000000 | Number of fill operations in random tests |
| `READ_NUM` | 10000000 | Number of read operations in random tests |
| `DEFAULT_KEY_SIZE` | 24 | Default key size in bytes |
| `DEFAULT_VALUE_SIZE` | 128 | Default value size in bytes |

## Supported Test Types

### 1. Random Tests (fillrandom + readrandom)

Simulates typical LSM-Tree workloads with sequential writes and random reads.

**Supported Value Size Variants**:
- 128 bytes
- 256 bytes
- 512 bytes
- 1024 bytes
- 2048 bytes
- 4096 bytes

**Enable by** - Uncommenting in the `main()` function of `test.py`:
```python
for value_size in [128, 256, 512, 1024, 2048, 4096]:
    for i in range(3):  # 3 iterations
        for use_li in [False, True]:  # Compare with/without learned index
            exp_log_path = f"{EXP_ROOT}/k{DEFAULT_KEY_SIZE}_v{value_size}/random_{i}/{'li' if use_li else 'no_li'}/"
            run_test(use_li=use_li, test_type="random", ycsb_type='x', 
                    key_size=DEFAULT_KEY_SIZE, value_size=value_size, exp_log_path=exp_log_path)
```

### 2. YCSB Workload Tests

Supports the standard 6 workloads from Yahoo Cloud Serving Benchmark (YCSB).

**Supported Workload Types**:
- **Workload A**: 50% reads / 50% writes
- **Workload B**: 95% reads / 5% writes
- **Workload C**: 100% reads
- **Workload D**: 95% reads with recency / 5% writes
- **Workload E**: 95% short-range scans / 5% inserts
- **Workload F**: 50% reads / 50% read-modify-write

**Enable by** - Uncommenting in the `main()` function of `test.py`:
```python
for workload in ['a', 'b', 'c', 'd', 'e', 'f']:
    for i in range(3):  # 3 iterations
        for use_li in [False, True]:
            exp_log_path = f"{EXP_ROOT}/ycsb/{workload}/{i}/{'li' if use_li else 'no_li'}/"
            run_test(use_li=use_li, test_type="ycsb", ycsb_type=workload, exp_log_path=exp_log_path)
```

### 3. Real-World Workload Tests

We provide two representative real datasets (Wikipedia, Facebook) for read-write mixed workload tests.

**Supported Workload Patterns**:
- **readonly**: Read-only operations
- **readheavy**: Read-heavy (95% reads / 5% writes)
- **writeheavy**: Write-heavy (5% reads / 95% writes)
- **balanced**: Balanced (50% reads / 50% writes)

**Supported Datasets**:
- **wiki**: Wikipedia dataset
- **fb**: Facebook dataset

**Enable by** - Uncommenting in the `main()` function of `test.py`:
```python
for workload in ['readonly', 'readheavy', 'writeheavy', 'balanced']:
    for dataset in ['wiki', 'fb']:
        for use_li in [True]:
            exp_log_path = f"{EXP_ROOT}/ycsb/{workload}/{dataset}/{'li' if use_li else 'no_li'}/"
            run_test(use_li=use_li, test_type="ycsb", ycsb_type=workload, 
                    exp_log_path=exp_log_path, dataset=dataset)
```

## Usage Examples

### Single Random Test
```python
# Add or uncomment in test.py main():
run_test(use_li=True, test_type="random", ycsb_type='x',
        key_size=DEFAULT_KEY_SIZE, value_size=DEFAULT_VALUE_SIZE,
        exp_log_path=f"{EXP_ROOT}/single_random_test/li/")
```

### Single YCSB Workload B Test
```python
run_test(use_li=True, test_type="ycsb", ycsb_type="b",
        exp_log_path=f"{EXP_ROOT}/ycsb/b/0/li/")
```

## Results Output

After completion, experiment results are saved in the `EXP_ROOT` directory, including:

- **`log.txt`**: Complete db_bench run log
- **`{bench}_{iops}.txt`**: Extracted IOPS results
- **`{bench}_{li_tag}_stats_{suffix}.json`**: Performance statistics and trace breakdown (JSON format)
- **`{bench}_{li_tag}_io_counts_{suffix}.json`**: IO count statistics (block reads per operation, etc.)

## Troubleshooting

### Permission Error
If you encounter a "permission denied" error (when clearing page cache), ensure sudo password-less access is configured or enter the password correctly.

## License

This artifact follows the RocksDB dual-license structure. See the root-level [LICENSE](./LICENSE) file for the repository-level notice and the locations of the complete license texts.
