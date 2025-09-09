import os

base_dir = "."  # 기준 경로 (원하는 상위 폴더 지정 가능)

for name in os.listdir(base_dir):
    old_path = os.path.join(base_dir, name)
    if os.path.isdir(old_path):  # 폴더만 처리
        new_name = name.replace("\\", "_")
        new_path = os.path.join(base_dir, new_name)
        if new_name != name:  # 바뀐 게 있을 때만 실행
            os.rename(old_path, new_path)
            print(f"Renamed: {name}  →  {new_name}")

