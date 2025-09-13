# run_commands_parallel.py
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm


def worker(cpu_id: int, job_queue: Queue, result_queue: Queue):
    """하나의 워커: queue에서 커맨드 꺼내서 실행"""
    while True:
        cmd = job_queue.get()
        if cmd is None:
            break
        try:
            # CPU 고정 실행
            res = subprocess.run(f"taskset -c {cpu_id} {cmd}", shell=True)
            rc = res.returncode
        except Exception:
            rc = -999
        result_queue.put((cmd, rc))


def main():
    parser = argparse.ArgumentParser(description="Run command file in parallel (with tqdm)")
    parser.add_argument("--cmd-file", type=str, required=True,
                        help="Path to command file (txt with one command per line)")
    parser.add_argument("--num-cpus", type=int, default=1,
                        help="Number of parallel workers (max CPUs)")
    args = parser.parse_args()

    cmd_file = Path(args.cmd_file)
    if not cmd_file.exists():
        raise FileNotFoundError(f"Command file not found: {cmd_file}")

    # 커맨드 읽기
    commands = cmd_file.read_text(encoding="utf-8").strip().splitlines()
    print(commands)
    commands = [ x for x in commands if "random_with_input" not in x  ]
    total = len(commands)
    print(f"[INFO] Loaded {total} commands from {cmd_file}")

    job_queue = Queue()
    result_queue = Queue()

    # 작업 큐에 커맨드 넣기
    for cmd in commands:
        job_queue.put(cmd)
    # 워커 종료 신호
    for _ in range(args.num_cpus):
        job_queue.put(None)

    # 워커 시작
    workers = []
    for i in range(args.num_cpus):
        p = Process(target=worker, args=(i, job_queue, result_queue))
        p.start()
        workers.append(p)

    # 결과 수집 + tqdm 프로그레스바
    results = []
    with tqdm(total=total, desc="Running commands", ncols=100) as pbar:
        for _ in range(total):
            cmd, rc = result_queue.get()
            results.append((cmd, rc))
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{status}] {cmd}")
            pbar.update(1)

    # 워커 종료 대기
    for p in workers:
        p.join()

    # 실패 요약
    fails = [c for c, rc in results if rc != 0]
    print("\n========== SUMMARY ==========")
    print(f"Total commands: {len(commands)}")
    print(f"Success: {len(commands) - len(fails)}")
    print(f"Failed : {len(fails)}")
    if fails:
        print("First 5 failures:")
        for f in fails[:5]:
            print(" -", f)


if __name__ == "__main__":
    main()

