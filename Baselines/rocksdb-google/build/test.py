#!/usr/bin/env python3
"""
RocksDB Benchmark Test Script
Equivalent Python version of test.sh
"""

import os
import subprocess
import sys
import re
import shutil
import time
from pathlib import Path


# ==================== Configuration ====================
DB_PATH = "/mnt/n1/dbpath/google"
WORKLOAD_PATH = "../ycsb_workload"
EXP_ROOT = "../new_exp"

LOAD_NUM = 64000000
RUNNING_NUM = 10000000
FILL_NUM = 64000000
READ_NUM = 10000000
DEFAULT_KEY_SIZE = 24
DEFAULT_VALUE_SIZE = 128
BREAKDOWN_SUFFIX = DEFAULT_VALUE_SIZE
BLOOM_BITS = -1

COMMON_ARGS = [
    "--compression_type=None",
    "--format_version=5",
    "--max_background_compactions=4",
    "--max_background_jobs=5",
    "--max_background_flushes=-1",
    "--stats_dump_period_sec=120",
]


def run_command(cmd, check=True, capture_output=False):
    """Execute shell command"""
    if isinstance(cmd, list):
        cmd_str = ' '.join(cmd)
    else:
        cmd_str = cmd
    
    print(f"Running: {cmd_str}")
    
    if capture_output:
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            sys.exit(1)
        return result
    else:
        result = subprocess.run(cmd_str, shell=True)
        if check and result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            sys.exit(1)
        return result


def wait_for_db_bench():
    """Wait for all db_bench processes to finish"""
    while True:
        result = subprocess.run("pgrep -x db_bench", shell=True, capture_output=True)
        if result.returncode != 0:
            # No db_bench process found
            break
        print("Waiting for db_bench to finish...")
        time.sleep(5)
    print("db_bench process finished.")


def save_iops_file(log_file, exp_dir, tag):
    """Extract and save IOPS information from log file"""
    if not os.path.exists(log_file):
        return
    
    with open(log_file, 'r') as f:
        content = f.read()

    for line in content.split('\n'):
        if line.startswith(tag):
            match = re.search(r'(\d+)\s+ops/sec', line)
            if match:
                iops = match.group(1)
                output_file = os.path.join(exp_dir, f"{tag}_{iops}.txt")
                with open(output_file, 'w') as f:
                    f.write(line + '\n')
                print(f"Saved IOPS file: {output_file}")
            break


def collect_trace_breakdown(exp_dir, bench_names, use_li, suffix):
    li_tag = "li" if use_li else "no_li"
    for bench in bench_names:
        dst_name = f"{bench}_{li_tag}_stats_{suffix}.json"
        dst_path = os.path.join(exp_dir, dst_name)
        src_candidates = [
            os.path.join(exp_dir, f"{bench}_{li_tag}_stats.json"),
            os.path.join(exp_dir, f"{bench}_trace_stats.json"),  # legacy naming
            f"{bench}_{li_tag}_stats.json",
            f"{bench}_trace_stats.json",
        ]
        for src_path in src_candidates:
            if os.path.exists(src_path):
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    os.replace(src_path, dst_path)
                print(f"Saved breakdown: {dst_path}")
                break


def parse_total_operations(log_content, bench_tag):
    ops = None
    pattern = re.compile(rf"^\s*{re.escape(bench_tag)}\s*:.*?(\d+)\s+operations;",
                         re.IGNORECASE)
    for line in log_content.splitlines():
        m = pattern.search(line)
        if m:
            ops = int(m.group(1))
    return ops


def save_io_count_breakdown(exp_dir, bench_name, use_li, suffix, log_file=None):
    li_tag = "li" if use_li else "no_li"
    stats_path = os.path.join(exp_dir, f"{bench_name}_{li_tag}_stats_{suffix}.json")
    if not os.path.exists(stats_path):
        return

    try:
        import json
        with open(stats_path, "r") as f:
            stats = json.load(f)
    except Exception as e:
        print(f"Failed to read breakdown json {stats_path}: {e}")
        return

    total_ops = None
    if log_file and os.path.exists(log_file):
        with open(log_file, "r") as f:
            total_ops = parse_total_operations(f.read(), bench_name)

    def get_count(metric_name):
        metric = stats.get(metric_name, {})
        return int(metric.get("count", 0) or 0) if isinstance(metric, dict) else 0

    read_file_loop = stats.get("ReadFileLoop", {})
    avg_loop = float(read_file_loop.get("avg_loop", 0) or 0) if isinstance(read_file_loop, dict) else 0.0
    loop_count = get_count("ReadFileLoop")
    est_total_loops = int(round(loop_count * avg_loop)) if loop_count else 0

    out = {
        "bench": bench_name,
        "use_li": use_li,
        "suffix": suffix,
        "total_operations": total_ops,
        "ReadBlockCacheHit_count": get_count("ReadBlockCacheHit"),
        "ReadBlockCacheMiss_count": get_count("ReadBlockCacheMiss"),
        "ReadBlock_count": get_count("ReadBlock"),
        "ReadDataBlock_count": get_count("ReadDataBlock"),
        "DataBlockReadCount": get_count("ReadDataBlock"),
        "ReadFile_count": get_count("ReadFile"),
        "ReadFileLoop_count": loop_count,
        "ReadFileLoop_avg_loop": avg_loop,
        "ReadFileLoop_estimated_total_loops": est_total_loops,
        "ReadRandomAccessFile_count": get_count("ReadRandomAccessFile"),
    }
    if total_ops and total_ops > 0:
        for k, v in list(out.items()):
            if k.endswith("_count") and isinstance(v, int):
                out[k.replace("_count", "_per_op")] = v / total_ops
        out["DataBlock_read_per_op"] = out["ReadDataBlock_count"] / total_ops
        if est_total_loops:
            out["ReadFileLoop_estimated_total_loops_per_op"] = est_total_loops / total_ops

    out_path = os.path.join(exp_dir, f"{bench_name}_{li_tag}_io_counts_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Saved IO count breakdown: {out_path}")

def run_test(use_li, test_type, ycsb_type='x', key_size=None, value_size=None, exp_log_path=None, dataset=None):
    """
    Run benchmark test
    
    Args:
        use_li: bool - whether to use learned index (True/False)
        test_type: str - 'random' or 'ycsb'
        ycsb_type: str - workload type 'a'-'f' (for ycsb only)
        key_size: int - key size in bytes
        value_size: int - value size in bytes
        exp_log_path: str - path to experiment log directory
    """
    # Set defaults
    if key_size is None:
        key_size = DEFAULT_KEY_SIZE
    if value_size is None:
        value_size = DEFAULT_VALUE_SIZE

    exp_dir = exp_log_path

    if test_type == "random":
        benchmarks = "fillrandom,stats,readrandom,stats"
        workload = ""
    elif test_type == "ycsb":
        benchmarks = "ycsb"
        workload = f"{WORKLOAD_PATH}/workload_{dataset}_{ycsb_type}"
    elif test_type == "ycsb" and dataset is not None:
        benchmarks = "ycsb"
        workload = f"{WORKLOAD_PATH}/workload{ycsb_type}"
    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(1)
    
    log_path = os.path.join(exp_dir, "log.txt")
    
    # ========== Prepare ==========
    print("=" * 50)
    print("================== Run Test ==================")
    print(f"use_li={use_li}, type={test_type}, ycsb={ycsb_type}")
    print(f"key_size={key_size}, value_size={value_size}")
    print(f"log -> {log_path}")
    print("=" * 50)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    os.makedirs(exp_dir, exist_ok=True)
    
    # ========== Build command ==========\
    if test_type == "random":
        cmd = [
            "./db_bench",
            f"--db={DB_PATH}",
            f"--benchmarks={benchmarks}",
            f"--enable_learned_index={'true' if use_li else 'false'}",
            f"--num={FILL_NUM}",
            f"--reads={READ_NUM}",
            f"--key_size={key_size}",
            f"--value_size={value_size}",
            f"--bloom_bits={BLOOM_BITS}",
            "--histogram=1",
            f"--latency_stats_dir={exp_dir}",
            f"--seed=42",
            f"--learned_index_strict_mode=true",
            f"--learned_index_segment_keys={4 * 1000000}"
        ]
    elif test_type == "ycsb":
        cmd = [
            "./db_bench",
            f"--db={DB_PATH}",
            f"--benchmarks={benchmarks}",
            f"--enable_learned_index={'true' if use_li else 'false'}",
            f"--load_num={LOAD_NUM}",
            f"--running_num={RUNNING_NUM}",
            f"--key_size={key_size}",
            f"--value_size={value_size}",
            f"--bloom_bits={BLOOM_BITS}",
            "--histogram=1",
            f"--latency_stats_dir={exp_dir}",
            f"--seed=42",
            f"--learned_index_strict_mode=true",
            f"--learned_index_segment_keys={4 * 1000000}"
        ]
    
    if workload:
        cmd.append(f"--ycsb_workload={workload}")
    
    cmd.extend(COMMON_ARGS)

    cmd_str = ' '.join(cmd) + f" > {log_path}"

    print("Sleep 20 seconds to wait for system stable...")
    time.sleep(20)

    print("Free memory cache...")
    run_command("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
    # ========== Run benchmark ==========
    run_command(cmd_str, check=False)

    wait_for_db_bench()
    
    # ========== Save IOPS files ==========
    if test_type == "random":
        save_iops_file(log_path, exp_dir, "fillrandom")
        save_iops_file(log_path, exp_dir, "readrandom")
        collect_trace_breakdown(exp_dir, ["fillrandom", "readrandom"], use_li, DEFAULT_VALUE_SIZE)
        save_io_count_breakdown(exp_dir, "fillrandom", use_li, DEFAULT_VALUE_SIZE, log_path)
        save_io_count_breakdown(exp_dir, "readrandom", use_li, DEFAULT_VALUE_SIZE, log_path)
    elif test_type == "ycsb":
        save_iops_file(log_path, exp_dir, "ycsb")
        collect_trace_breakdown(exp_dir, ["ycsb"], use_li, BREAKDOWN_SUFFIX)
        save_io_count_breakdown(exp_dir, "ycsb", use_li, BREAKDOWN_SUFFIX, log_path)
    
    print(f"Test completed. Results saved to: {exp_dir}")


def main():
    """Main function"""
    print("=" * 50)
    print("================== Build db_bench ===================")
    print("=" * 50)
    nproc = os.cpu_count() or 4
    run_command(f"make db_bench -j{nproc}")
    
    print("=" * 50)
    print("================== Delete Old DB Path ===================")
    print("=" * 50)
    
    # ========== Run Tests ==========
    
    # fillrandom - readrandom with different value sizes
    # for value_size in [128, 256, 512, 1024, 2048]:
    #     for use_li in [True]:
    #         for i in range(3):
    #             exp_log_path = f"{EXP_ROOT}/k{DEFAULT_KEY_SIZE}_v{value_size}/random_{i}/{'li' if use_li else 'no_li'}/"
    #             print(f"\nIteration {i + 1} for value_size={value_size}, use_li={use_li}")
    #             run_test(use_li=use_li, test_type="random", ycsb_type='x', 
    #                     key_size=DEFAULT_KEY_SIZE, value_size=value_size, exp_log_path=exp_log_path)

    # Uncomment for single test
    # run_test(use_li=True, test_type="random", ycsb_type='x', 
    #          key_size=DEFAULT_KEY_SIZE, value_size=128, exp_log_path=f"{EXP_ROOT}/k{DEFAULT_KEY_SIZE}_v128/random_0/li/")
    
    # ========== YCSB Tests (commented out) ==========
    # for workload in ['a', 'b', 'c', 'd', 'e', 'f']:
    #     for use_li in [True]:
    #         for i in range(3):
    #             exp_log_path = f"{EXP_ROOT}/ycsb/{workload}/{i}/{'li' if use_li else 'no_li'}/"
    #             print(f"\nIteration {i + 1} for YCSB workload={workload}, use_li={use_li}")
    #             run_test(use_li=use_li, test_type="ycsb", ycsb_type=workload, exp_log_path=exp_log_path)
    # run_test(use_li=True, test_type="ycsb", ycsb_type="b", exp_log_path=f"{EXP_ROOT}/ycsb/b/0/li/")
    
        # ========== Real World Tests (commented out) ==========
    # for workload in ['readonly', 'readheavy', 'balanced', 'writeheavy']:
    #     for dataset in ['wiki', 'fb']:
    #         for use_li in [True]:
    #             exp_log_path = f"{EXP_ROOT}/ycsb/{workload}/{dataset}/{'li' if use_li else 'no_li'}/"
    #             print(f"\nworkload={workload}, dataset={dataset}, use_li={use_li}")
    #             run_test(use_li=use_li, test_type="ycsb", ycsb_type=workload, exp_log_path=exp_log_path, dataset=dataset)
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
