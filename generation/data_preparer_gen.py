import csv
import os
import re
import subprocess
import time
from typing import Any, Dict, List

import numpy as np

OUTPUT_DIR = "jsp_cnn_data_gen"
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de salida {OUTPUT_DIR}. Detalles: {e}"
    )

MODEL_MZN_PATH = "generation/model.mzn"
TIME_LIMIT_MS = 60000
TIME_LIMIT_S = TIME_LIMIT_MS / 1000

LOG_DIR = os.path.join(OUTPUT_DIR, "solver_logs_temp")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de logs temporal {LOG_DIR}. Detalles: {e}"
    )

SOLVER_STRATEGIES = [
    ("gecode", "input_order", "GECODE_DEFAULT"),
    ("chuffed", "input_order", "CHUFFED_DEFAULT"),
]

GENERATION_CASES = [
    (4, 4, 5),
    (8, 8, 5),
    (12, 12, 5),
    (15, 15, 5),
]

try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.generation import GeneralInstanceGenerator
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: No se puede importar job_shop_lib. "
        "Asegúrate de que 'job-shop-lib' esté instalado y funcional. Deteniendo ejecución."
    ) from e


def save_instance_dzn(instance_obj: JobShopInstance, instance_name: str) -> str:
    dzn_path = os.path.join(OUTPUT_DIR, f"{instance_name}.dzn")

    proc_time_rows = []
    machine_of_op_rows = []

    try:
        for job_ops in instance_obj.jobs:
            pt_row = []
            mach_row = []
            for op in job_ops:
                pt_row.append(op.duration)

                machine_info = op.machines[0]
                if isinstance(machine_info, int):
                    machine_id_0_based = machine_info
                elif hasattr(machine_info, "id"):
                    machine_id_0_based = machine_info.id
                else:
                    raise AttributeError(
                        f"Estructura inesperada para la máquina en operación {op}."
                    )

                mach_row.append(machine_id_0_based + 1)

            proc_time_rows.append(pt_row)
            machine_of_op_rows.append(mach_row)
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Fallo al extraer datos de la instancia {instance_name}. Detalles: {e}"
        )

    def format_mzn_flat_array(matrix_2d):
        flat_list = [item for row in matrix_2d for item in row]
        return f"[{', '.join(map(str, flat_list))}]"

    try:
        with open(dzn_path, "w") as f:
            f.write(f"JOBS = {instance_obj.num_jobs};\n")
            f.write(f"MACHINES = {instance_obj.num_machines};\n\n")
            f.write(
                f"PROC_TIME = array2d(SET_JOBS, SET_POS, {format_mzn_flat_array(proc_time_rows)});\n"
            )
            f.write(
                f"MACHINE_OF_OP = array2d(SET_JOBS, SET_POS, {format_mzn_flat_array(machine_of_op_rows)});\n"
            )
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Fallo al escribir el archivo .dzn {dzn_path}. Detalles: {e}"
        )

    return dzn_path


def parse_minizinc_stats(mzn_text: str) -> Dict[str, Any]:
    """Extrae el Makespan (END) y el runtime."""
    stats = {"makespan": float("inf"), "runtime": TIME_LIMIT_S}

    try:
        all_makespans = re.findall(r"END=\s*(\d+)", mzn_text)

        if all_makespans:
            best_makespan = min(float(m) for m in all_makespans)
            stats["makespan"] = best_makespan

        t = re.search(r"%%%mzn-stat: solveTime=([0-9\.]+)", mzn_text)
        if not t:
            t = re.search(r"%%%mzn-stat: time=([0-9\.]+)", mzn_text)

        stats["runtime"] = float(t.group(1)) if t else TIME_LIMIT_S
        stats["solved_binary"] = 1 if stats["makespan"] < float("inf") else 0
    except Exception as e:
        print(f"❌ ERROR: Fallo al parsear las estadísticas de MiniZinc. Detalles: {e}")
        stats["runtime"] = TIME_LIMIT_S
        stats["solved_binary"] = 0

    return stats


def execute_minizinc_solver(
    solver: str, strategy: str, dzn_path: str
) -> Dict[str, Any]:
    """Ejecuta un solver de MiniZinc, INYECTANDO una heurística simple para forzar la búsqueda."""

    mzn_content = ""
    try:
        with open(MODEL_MZN_PATH, "r") as f:
            mzn_content = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"FATAL ERROR: No se encontró el modelo MiniZinc en la ruta: {MODEL_MZN_PATH}"
        )

    solve_line_original = re.search(
        r"^\s*solve\s+minimize\s+END_MAKESPAN\s*;", mzn_content, re.MULTILINE
    )

    if solve_line_original:
        solve_template = f"solve :: int_search(S_FLAT, {strategy}, indomain_min, complete) minimize END_MAKESPAN;"
        mzn_patched = mzn_content.replace(solve_line_original.group(0), solve_template)
    else:
        mzn_patched = mzn_content

    tmp_mzn_path = os.path.join(LOG_DIR, f"_temp_{solver}_{strategy}.mzn")
    try:
        with open(tmp_mzn_path, "w") as f:
            f.write(mzn_patched)
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Fallo al escribir el modelo temporal {tmp_mzn_path}. Detalles: {e}"
        )

    cmd = [
        "minizinc",
        "--solver",
        solver,
        "--statistics",
        "--time-limit",
        str(TIME_LIMIT_MS),
        tmp_mzn_path,
        dzn_path,
    ]

    stdout = ""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIME_LIMIT_S + 5
        )
        stdout = proc.stdout
        os.unlink(tmp_mzn_path)
    except subprocess.TimeoutExpired:
        stdout = f"%%%mzn-stat: solveTime={TIME_LIMIT_S}\n"
    except FileNotFoundError:
        raise RuntimeError(
            f"FATAL ERROR: MiniZinc no encontrado. Asegúrate de que esté en tu PATH."
        )

    return parse_minizinc_stats(stdout)


def prepare_data_and_ground_truth_minizinc_gen():
    """Genera instancias aleatorias de JSP y ejecuta el pipeline de MiniZinc."""

    all_instances: List[JobShopInstance] = []

    print("--- 1. Generando Instancias JSP Balanceadas ---")

    instance_count = 0
    generator = GeneralInstanceGenerator(duration_range=(1, 20), seed=42)

    for jobs, machines, count in GENERATION_CASES:
        for i in range(count):
            instance_obj = generator.generate(num_jobs=jobs, num_machines=machines)
            instance_obj.name = f"GEN_{jobs}x{machines}_{i+1}"
            all_instances.append(instance_obj)
            instance_count += 1

    print(f"✅ Generadas un total de {instance_count} instancias.")
    print("-" * 50)

    solver_keys = [name for _, _, name in SOLVER_STRATEGIES]

    csv_rows = [
        [
            "Instance_Name",
            "Raw_Text_Path",
            "N_Jobs",
            "N_Machines",
            "Best_Makespan_Found",
        ]
        + [f"{s}_Runtime_s" for s in solver_keys]
    ]

    for index, instance_obj in enumerate(all_instances):
        start_time = time.time()
        name = instance_obj.name

        print(
            f"🚀 Procesando Instancia {index + 1}/{instance_count}: {name} ({instance_obj.num_jobs}x{instance_obj.num_machines})"
        )

        try:
            dzn_path = save_instance_dzn(instance_obj, name)
            raw_text_path = dzn_path

            all_results = {}
            best_makespan = float("inf")
            total_solver_time = 0.0

            for solver, strategy, key in SOLVER_STRATEGIES:
                solver_start_time = time.time()
                print(f"  -> Ejecutando Solver {key}...")
                stats = execute_minizinc_solver(solver, strategy, dzn_path)
                solver_end_time = time.time()

                runtime_s = stats["runtime"]

                print(f"     - Tiempo MiniZinc (Reportado): {runtime_s:.3f}s")
                print(
                    f"     - Tiempo de Ejecución (Wall-Time): {(solver_end_time - solver_start_time):.3f}s"
                )

                all_results[key] = stats

                if stats["makespan"] < best_makespan:
                    best_makespan = stats["makespan"]

                total_solver_time += runtime_s

            row = [
                name,
                raw_text_path,
                instance_obj.num_jobs,
                instance_obj.num_machines,
                best_makespan if best_makespan != float("inf") else "inf",
            ]

            row.extend([f"{all_results[key]['runtime']:.3f}" for key in solver_keys])

            csv_rows.append(row)

            end_time = time.time()
            elapsed_time = end_time - start_time

            print(
                f"✅ Instancia {name} Finalizada. Makespan Óptimo/Mejor Encontrado: {row[4]}. "
                f"Tiempo Total (Wall-Time): {elapsed_time:.3f}s. "
                f"Tiempo de Solvers (Sumado): {total_solver_time:.3f}s."
            )
            print("-" * 50)

        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"❌ ERROR fatal al procesar instancia {name}. Detalles: {e}")
            print(f"Tiempo Transcurrido hasta el fallo: {elapsed_time:.3f}s.")
            print("-" * 50)

    csv_path = os.path.join(OUTPUT_DIR, "ground_truth_jsp_generated_dataset.csv")
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\n✨ Proceso Completado. Resultados guardados en: {csv_path}")
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Fallo al escribir el archivo CSV {csv_path}. Detalles: {e}"
        )

    return csv_path
