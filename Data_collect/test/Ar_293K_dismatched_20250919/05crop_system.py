#!/usr/bin/env python3
import os
import re
import mmap
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# ——————————————————————
# 정규식 패턴 & 컬럼명 매핑
# ——————————————————————
PATTERNS = [
    (re.compile(r"Average loading absolute \[mol/kg framework\]\s*([0-9.Ee+-]+)"),
     "abs_mol_per_kg_framework"),
    (re.compile(r"Average loading absolute \[molecules/unit cell\]\s*([0-9.Ee+-]+)"),
     "abs_molecules_per_uc"),
    (re.compile(r"Average loading absolute \[milligram/gram framework\]\s*([0-9.Ee+-]+)"),
     "abs_mg_per_g_framework"),
    (re.compile(r"Average loading absolute \[cm\^3 \(STP\)/gr framework\]\s*([0-9.Ee+-]+)"),
     "abs_cm3STP_per_g"),
    (re.compile(r"Average loading absolute \[cm\^3 \(STP\)/cm\^3 framework\]\s*([0-9.Ee+-]+)"),
     "abs_cm3STP_per_cm3"),
]

def parse_meta(system0_path):
    entry = os.path.dirname(system0_path)
    base  = os.path.dirname(entry)
    parts = base.split("_")
    if len(parts) < 5:
        return None
    return {
        "name":     "_".join(parts[:-4]),
        "gas":      parts[-4],
        "temp":     float(parts[-3].rstrip("K")),
        "pressure": float(parts[-2].rstrip("bar")),
        "cutoff":   parts[-1],
        "path":     system0_path,
    }

def tail_mmap(filepath, size=32*1024):
    """
    메모리맵으로 파일 끝에서 최대 size 바이트만 읽어 문자열로 반환.
    """
    with open(filepath, "rb") as f:
        fileno = f.fileno()
        file_size = os.fstat(fileno).st_size
        # 읽기 시작 위치
        start = max(0, file_size - size)
        # mmap 전체했지만 slice로 마지막만 사용
        mm = mmap.mmap(fileno, length=0, access=mmap.ACCESS_READ)
        data = mm[start:]
        mm.close()
    return data.decode(errors="ignore")

def process_folder(system0):
    meta = parse_meta(system0)
    if meta is None:
        return None

    # 첫 파일
    try:
        fname = os.listdir(system0)[0]
    except IndexError:
        return None
    full = os.path.join(system0, fname)

    # mmap tail 읽기
    text = tail_mmap(full, size=32*1024)
    
    # 패턴 매칭
    for regex, col in PATTERNS:
        m = regex.search(text)
        meta[col] = float(m.group(1)) if m else float("nan")
    return meta

if __name__ == "__main__":
    import sys
    # 동시 프로세스 수를 크게 낮춰 I/O 경합 완화 (예: 코어 수의 1/4)
    cpu = os.cpu_count() or 4
    n_procs = int(sys.argv[1]) if len(sys.argv)>1 else max(1, cpu // 4)

    # System_0 폴더 목록
    tops    = [d for d in os.listdir(".") if os.path.isdir(os.path.join(d, "Output", "System_0"))]
    folders = [os.path.join(d, "Output", "System_0") for d in tops]

    out_csv = "07result.csv"
    cols = ["name","gas","temp","pressure","cutoff","path"] + [c for _,c in PATTERNS]
    # 헤더 작성
    pd.DataFrame([], columns=cols).to_csv(out_csv, index=False)

    chunk = []
    count = 0

    with Pool(n_procs) as pool:
        for res in tqdm(pool.imap_unordered(process_folder, folders),
                        total=len(folders),
                        desc="Extracting loadings"):
            if res:
                chunk.append(res)
                count += 1
            # 100개마다 중간저장
            if count and count % 100 == 0:
                pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=False, index=False)
                chunk.clear()

    # 남은 청크 저장
    if chunk:
        pd.DataFrame(chunk).to_csv(out_csv, mode="a", header=False, index=False)

    print(f"Done! Saved {count} records to {out_csv}")
