import os
import re
import mmap
import pandas as pd
from tqdm import tqdm
import logging
from multiprocessing import Pool

# ── Logging 설정 ───────────────────────────
logging.basicConfig(
    filename="henry_extraction.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── 패턴 정의 (CO2 없이) ───────────────────────────
PATTERNS = [
    (re.compile(r"Average Henry coefficient:\s+([0-9Ee\+\-\.]+)"), "henry_coeff"),
]

# ── 메타데이터 파서 ───────────────────────────
def parse_meta(system0_path):
    entry = os.path.dirname(system0_path)
    base = os.path.dirname(entry)
    parts = base.split("_")
    if len(parts) < 5:
        return None
    return {
        "name": "_".join(parts[:-4]),
        "gas": parts[-4],
        "temp": float(parts[-3].rstrip("K")),
        "pressure": float(parts[-2].rstrip("bar")),
        "cutoff": parts[-1],
        "path": system0_path,
    }

# ── tail mmap ───────────────────────────
def tail_mmap(filepath, size=32*1024):
    """
    메모리맵으로 파일 끝에서 최대 size 바이트만 읽어 문자열로 반환.
    """
    with open(filepath, "rb") as f:
        fileno = f.fileno()
        file_size = os.fstat(fileno).st_size
        start = max(0, file_size - size)
        mm = mmap.mmap(fileno, length=0, access=mmap.ACCESS_READ)
        data = mm[start:]
        mm.close()
    return data.decode(errors="ignore")

# ── 폴더 처리 ───────────────────────────
def process_folder(system0):
    meta = parse_meta(system0)
    if meta is None:
        return None

    data_files = [f for f in os.listdir(system0) if f.endswith(".data")]
    if not data_files:
        return None

    full = os.path.join(system0, data_files[0])
    text = tail_mmap(full, size=32*1024)

    for regex, col in PATTERNS:
        m = regex.search(text)
        meta[col] = float(m.group(1)) if m else float("nan")
    return meta

# ── 메인 실행 ───────────────────────────
if __name__ == "__main__":
    import sys

    cpu = os.cpu_count() or 4
    n_procs = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, cpu // 4)

    # 현재 디렉토리 하위 1-depth 폴더에서 Output/System_0 찾기
    folders = []
    for d in os.listdir("."):
        if os.path.isdir(d):
            sys0 = os.path.join(d, "Output", "System_0")
            if os.path.isdir(sys0):
                folders.append(sys0)

    out_csv = "07_henry_result.csv"
    cols = ["name", "gas", "temp", "pressure", "cutoff", "path"] + [c for _, c in PATTERNS]
    pd.DataFrame([], columns=cols).to_csv(out_csv, index=False)

    chunk = []
    count = 0

    with Pool(n_procs) as pool:
        for res in tqdm(pool.imap_unordered(process_folder, folders),
                        total=len(folders),
                        desc="Extracting Henry coefficients"):
            if res:
                chunk.append(res)
                count += 1

            if count and count % 100 == 0:
                pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=False, index=False)
                logging.info(f"Saved {count} records so far...")
                chunk.clear()

    if chunk:
        pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=False, index=False)

    print(f"Done! Saved {count} records to {out_csv}")
    logging.info(f"Done! Saved {count} records to {out_csv}")
