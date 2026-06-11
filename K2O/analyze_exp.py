import os
import re

EXP_ROOT = "exp_20260123"
TAGS = ["fillrandom", "readrandom"]    # fillrandom / readrandom / ycsb
OPS_SUBDIR = "OPS"

for TAG in TAGS:
    data = {}  # value_size -> { "li": iops, "no_li": iops }

    file_pat = re.compile(rf"{TAG}_(\d+)\.txt")
    kv_pat = re.compile(r"k(\d+)_v(\d+)")

    for kv in os.listdir(EXP_ROOT):
        kv_path = os.path.join(EXP_ROOT, kv)
        if not os.path.isdir(kv_path):
            continue

        m = kv_pat.match(kv)
        if not m:
            continue

        key_size = int(m.group(1))    # 现在不用，但以后能扩展
        value_size = int(m.group(2))

        for li_type in ["li", "no_li"]:
            ops_dir = os.path.join(kv_path, li_type, OPS_SUBDIR)
            if not os.path.exists(ops_dir):
                continue

            for fname in os.listdir(ops_dir):
                fm = file_pat.match(fname)
                if not fm:
                    continue

                iops = int(fm.group(1))
                data.setdefault(value_size, {})[li_type] = iops

    # ---------- 输出 ----------
    print("value_size\tno_li\tli\timprovement(%)")

    for v in sorted(data.keys()):
        no_li = data[v].get("no_li")
        li = data[v].get("li")

        if no_li is None or li is None:
            continue

        improvement = (li - no_li) / no_li * 100
        print(f"{v}\t{no_li}\t{li}\t{improvement:.2f}")
