import os

l = [
    "He_0.01bar_293K_mofsimplify_list_seed42_12023_part1",
    "He_0.01bar_293K_mofsimplify_list_seed42_12023_part2",
    "He_0.05bar_293K_mofsimplify_list_seed42_12023_part1",
    "He_0.05bar_293K_mofsimplify_list_seed42_12023_part2",
    "He_0.2bar_293K_mofsimplify_list_seed42_12023_part1",
    "He_0.2bar_293K_mofsimplify_list_seed42_12023_part2",
    "He_0.35bar_293K_mofsimplify_list_seed42_12023_part1",
    "He_0.35bar_293K_mofsimplify_list_seed42_12023_part2"
]

base_dir = os.getcwd()  # 현재 작업 디렉토리 저장

for folder in l:
    print(folder)
    # 디렉토리 변경
    os.chdir(os.path.join(base_dir, folder))
    print(f"Changed directory to {os.getcwd()}")
    os.system("python 02make_multiple_simulations.py --n_jobs 33")
    # 원래 디렉토리로 돌아가기
    os.chdir(base_dir)

