import pandas as pd
import os
import argparse
from tqdm.contrib.concurrent import process_map

def process_for_path(path):
    sim_name = path.split("/Output/System_0")[0]
    sim_splited = sim_name.split("_")
    cutoff = sim_splited[-1]
    pressure = float(sim_splited[-2].split("bar")[0])
    temp = float(sim_splited[-3].split("K")[0])
    gas = sim_splited[-4]
    mof = "_".join(sim_splited[:-4])
    return {
        "name": mof,
        "gas": gas,
        "pressure": pressure,
        "temp": temp,
        "cutoff": cutoff,
    }

def process_directory(dir_name):
    try:
        path = os.path.join(dir_name, "Output/System_0")
        first_file = os.listdir(path)[0]
        file_path = os.path.join(path, first_file)

        with open(file_path, "r") as f:
            data = f.read()

        dic = process_for_path(path)
        dic["path"] = path

        dic["Average loading absolute [mol/kg framework]"] = float(
            data.split("Average loading absolute [mol/kg framework]")[1].split(" +/-")[0].split()[0]
        )
        dic["Average loading excess [mol/kg framework]"] = float(
            data.split("Average loading excess [mol/kg framework]")[1].split(" +/-")[0].split()[0]
        )
        dic["Average loading absolute [molecules/unit cell]"] = float(
            data.split("Average loading absolute [molecules/unit cell]")[1].split(" +/-")[0].split()[0]
        )
        dic["Average loading excess [molecules/unit cell]"] = float(
            data.split("Average loading excess [molecules/unit cell]")[1].split(" +/-")[0].split()[0]
        )

        return dic

    except Exception as e:
        print(f"❌ error: {dir_name}/Output/System_0 -> {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs to use")
    args = parser.parse_args()

    dirs = [x for x in os.listdir() if os.path.isdir(x)]

    # tqdm 병렬 처리 with progress bar
    results = process_map(process_directory, dirs, max_workers=args.num_cpus, chunksize=1, desc="Processing directories")

    # None 제거
    clean_results = [r for r in results if r is not None]

    pd.DataFrame(clean_results).to_csv("07result.csv", index=False)
    print("✅ Saved to 07result.csv")

if __name__ == "__main__":
    main()

