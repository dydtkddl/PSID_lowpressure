import os
import subprocess
from joblib import Parallel, delayed
import argparse

# 시뮬레이션 실행 함수
def run_simulation(sim_dir, raspa_dir):
    try:
        print(f"Running simulation in {sim_dir}...")
        simulation_command = f"{raspa_dir}/bin/simulate simulation.input"
        subprocess.run(simulation_command, shell=True, check=True, cwd=sim_dir)  # 해당 디렉토리에서 실행
        print(f"Simulation completed in {sim_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in {sim_dir}: {e}")

def main():
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description="Run simulations in parallel across directories.")
    parser.add_argument("--num_cpus", type=int, default=10, help="Number of CPUs to use for parallel processing (-1 for all available CPUs).")
    
    RASPA_DIR = os.getenv("RASPA_DIR")
    if not RASPA_DIR:
        raise EnvironmentError("RASPA_DIR 환경 변수가 설정되지 않았습니다.")

    args = parser.parse_args()
    # 현재 디렉토리의 서브디렉토리 가져오기
    sim_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    # 병렬 실행
    num_cpus = args.num_cpus
    Parallel(n_jobs=num_cpus)(delayed(run_simulation)(sim_dir, RASPA_DIR) for sim_dir in sim_dirs)

if __name__ == "__main__":
    main()

