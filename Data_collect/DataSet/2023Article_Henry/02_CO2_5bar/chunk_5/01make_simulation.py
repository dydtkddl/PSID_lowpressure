import argparse
import os
import pyrascont

def replace_placeholders(template, replacements):
    for key, value in replacements.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template

def process_mof_simulations(mof, pressure, temp, gas, base_input,  cutoffvdw, cutoffchargecharge, cutoffchargebonddipole, cutoffbonddipolebonddipole, NumberOfCycles, NumberOfInitializationCycles, PrintEvery):
    # 선택한 모프의 UNITCELL계산을 위한 코드
    raspa_dir = os.environ.get("RASPA_DIR")
    if not raspa_dir:
        raise EnvironmentError("RASPA_DIR 환경 변수가 설정되어 있지 않습니다.")
    # CIF 파일 경로 설정
    cifs_path = os.path.join(raspa_dir, "share", "raspa", "structures", "cif")
    res_ucell = pyrascont.cif2Ucell(os.path.join(cifs_path, mof), cutoffvdw, Display=False)
    unitcell_str = ' '.join(map(str, res_ucell))

    ## BASETEMPLATE의 값을 치환하는 코드
    replacement_values = {
        "MOF": mof,
        "UNITCELL": unitcell_str,
        "PRESSURE": str(pressure * 100000),
        "TEMP": temp,
        "GAS": gas,
        "CUTOFFVDW" : cutoffvdw,
        "CUTOFFCHARGECHARGE" : cutoffchargecharge,
        "CUTOFFCHARGEBONDDIPOLE" : cutoffchargebonddipole,
        "CUTOFFBONDDIPOLEBONDDIPOLE" : cutoffbonddipolebonddipole,
        "NumberOfCycles": NumberOfCycles,
        "NumberOfInitializationCycles": NumberOfInitializationCycles,
        "PrintEvery": PrintEvery,
    }
    final_content = replace_placeholders(base_input, replacement_values)

    ## 현재 디렉토리에 해당 모프 시뮬레이션 폴더를 생성 및 시뮬레이션 인풋도 넣기.
    result_dir_ = f"{mof}_{gas}_{temp}K_{pressure}bar_{cutoffvdw}"
    os.makedirs(result_dir_, exist_ok=True)
    simulation_file = os.path.join(result_dir_, "simulation.input").replace("\\", "/")
    with open(simulation_file, "w") as f:
        f.write(final_content)
    # print(f"{mof} 시뮬레이션 파일 생성 완료: {simulation_file}")

def main():
    parser = argparse.ArgumentParser(description="Process MOF simulations.")

    parser.add_argument("--mof", type=str, required=True, help="MOF file name (without extension)")
    parser.add_argument("--pressure", type=float, required=True, help="Pressure in bar")
    parser.add_argument("--temp", type=float, required=True, help="Temperature in Kelvin")
    parser.add_argument("--gas", type=str, required=True, help="Gas type")
    parser.add_argument("--base_input", type=str, required=True, help="Base input template for simulation")
    parser.add_argument("--cutoffvdw", type=float, required=True, help="cutoffvdw distance in Angstroms")
    parser.add_argument("--cutoffchargecharge", type=float, required=True, help="cutoffchargecharge distance in Angstroms")
    parser.add_argument("--cutoffchargebonddipole", type=float, required=True, help="cutoffchargebonddipole distance in Angstroms")
    parser.add_argument("--cutoffbonddipolebonddipole", type=float, required=True, help="cutoffbonddipolebonddipole distance in Angstroms")
    parser.add_argument("--NumberOfCycles", type=int, required=True, help="Number of cycles")
    parser.add_argument("--NumberOfInitializationCycles", type=int, required=True, help="Number of initialization cycles")
    parser.add_argument("--PrintEvery", type=int, required=True, help="Print frequency")

    args = parser.parse_args()

    with open(args.base_input, "r") as f:
        base_input_content = f.read()

    process_mof_simulations(
        mof=args.mof,
        pressure=args.pressure,
        temp=args.temp,
        gas=args.gas,
        base_input=base_input_content,
        cutoffvdw=args.cutoffvdw,
        cutoffchargecharge=args.cutoffchargecharge,
        cutoffchargebonddipole=args.cutoffchargebonddipole,
        cutoffbonddipolebonddipole=args.cutoffbonddipolebonddipole,
        NumberOfCycles=args.NumberOfCycles,
        NumberOfInitializationCycles=args.NumberOfInitializationCycles,
        PrintEvery=args.PrintEvery,
    )

if __name__ == "__main__":
    main()

