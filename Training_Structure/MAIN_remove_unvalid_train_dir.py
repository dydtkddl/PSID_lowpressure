import argparse
import os
import shutil
from pathlib import Path

def isvalid(x: int) -> int:
    """디렉토리 내 파일 개수 기준으로 유효성 체크"""
    return 0 if x <= 1 else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="실험 결과 상위 디렉토리 경로")
    args = parser.parse_args()

    S_PATH = Path(args.path)

    # 1단계 하위 디렉토리 목록
    S_DIRS = [x for x in os.listdir(S_PATH) if os.path.isdir(S_PATH / x)]

    for S in S_DIRS:
        trial_dir = S_PATH / S / "trial_001"

        if not trial_dir.exists():
            print(f"[SKIP] {trial_dir} 없음")
            shutil.rmtree(S_PATH / S, ignore_errors=True)
            print(f"[REMOVE] {S_PATH/S}")
            continue

        FLAG = isvalid(len(os.listdir(trial_dir)))

        if FLAG == 1:
            print(f"[KEEP] {S}")
        else:
            shutil.rmtree(S_PATH / S)
            print(f"[REMOVE] {S_PATH/S}")

if __name__ == "__main__":
    main()

