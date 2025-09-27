import subprocess
import glob
import os
import logging
from multiprocessing import Process, Manager
from tqdm import tqdm
import time

# --- 설정 ---
# 사용 가능한 원격 호스트 리스트
HOSTS = [f'ga{i:02d}' for i in range(5)]  # ga00, ga01, ga02, ga03, ga04

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def node_worker(hostname, task_queue, pbar):
    """
    하나의 노드(호스트)를 전담하여 작업을 순차적으로 처리하는 워커 함수.
    이전 작업이 끝나기 전까지는 절대로 다음 작업을 시작하지 않습니다.
    """
    # 워커별 로거 이름 설정
    log = logging.getLogger(hostname)
    
    # 원격지에서 실행할 현재 작업 디렉터리 (로컬과 동일하게)
    cwd = os.getcwd().replace('\\', '/')
    
    while True:
        # 큐에서 작업(cmd 파일)을 하나 가져옴. 큐가 비어있으면 작업이 들어올 때까지 대기.
        cmd_file = task_queue.get()
        
        # 큐에서 None을 받으면 워커를 종료하라는 신호로 인식
        if cmd_file is None:
            log.info("받을 작업이 없어 종료합니다.")
            break
            
        log.info(f"'{cmd_file}' 작업 처리 시작...")
        
        # 원격 서버에서 실행할 전체 명령어 구성
        remote_command = (
            f"cd {cwd} && "
            f"python ../run_commands_parallel.py --cmd-file {cmd_file} --num-cpus 25"
        )
        ssh_command = ['ssh', hostname, remote_command]
        
        try:
            # subprocess.run은 해당 프로세스가 끝날 때까지 대기하므로,
            # 이 코드를 통해 노드 내 작업의 순차 실행이 보장됨.
            result = subprocess.run(
                ssh_command,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                log.info(f"✅ '{cmd_file}' 작업 성공.")
            else:
                log.error(f"❌ '{cmd_file}' 작업 실패 (코드: {result.returncode}).")
                log.error(f"오류 내용:\n{result.stderr.strip()}")
                
        except Exception as e:
            log.error(f"❌ '{cmd_file}' 실행 중 예외 발생: {e}")
        finally:
            # 작업 성공/실패 여부와 관계없이 진행 막대 업데이트
            pbar.update(1)

if __name__ == "__main__":
    # 1. 현재 디렉터리에서 .cmd 파일 목록을 찾음
    cmd_files = sorted(glob.glob('*.txt'))
    
    if not cmd_files:
        logging.warning("현재 디렉터리에서 실행할 .cmd 파일을 찾을 수 없습니다.")
        exit()
        
    total_tasks = len(cmd_files)
    logging.info(f"총 {total_tasks}개의 작업을 {len(HOSTS)}개의 노드에 분배합니다.")
    logging.info(f"대상 파일: {cmd_files}")
    
    # 2. 여러 프로세스가 공유할 수 있는 Manager 큐 생성
    manager = Manager()
    task_queue = manager.Queue()

    # 3. 진행 상황을 표시할 tqdm 객체 생성
    pbar = tqdm(total=total_tasks, desc="전체 진행률")
    
    # 4. 모든 cmd 파일을 작업 큐에 추가
    for cmd in cmd_files:
        task_queue.put(cmd)

    # 5. 각 워커 프로세스를 종료시키기 위한 신호(None)를 호스트 수만큼 추가
    for _ in HOSTS:
        task_queue.put(None)
        
    # 6. 각 호스트를 전담할 워커 프로세스를 생성하고 시작
    processes = []
    for host in HOSTS:
        p = Process(target=node_worker, args=(host, task_queue, pbar))
        processes.append(p)
        p.start()
        logging.info(f"[{host}] 워커가 작업을 시작했습니다.")
        
    # 7. 모든 워커 프로세스가 종료될 때까지 대기
    for p in processes:
        p.join()
        
    # 8. 모든 작업 완료 후 정리
    pbar.close()
    print("\n🎉 모든 작업이 성공적으로 완료되었습니다.")
