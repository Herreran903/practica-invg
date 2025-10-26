import csv
import os
import re
import subprocess
import time
from typing import Any, Dict

OUTPUT_DIR = "jsp_cnn_data_mzn"
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de salida {OUTPUT_DIR}. Detalles: {e}"
    )

INSTANCE_NAMES = ["ft06", "ft10", "la01", "abz5"]
MODEL_MZN_PATH = "jobshop_model.mzn"
TIME_LIMIT_MS = 60000
TIME_LIMIT_S = TIME_LIMIT_MS / 1000
PENALTY_FACTOR_K = 10.0

SOLVER_STRATEGIES = [
    ("gecode", "default", "GECODE_DEFAULT"),
    ("chuffed", "default", "CHUFFED_DEFAULT"),
]

LOG_DIR = os.path.join(OUTPUT_DIR, "solver_logs_temp")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de logs temporal {LOG_DIR}. Detalles: {e}"
    )


try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.benchmarking import load_benchmark_instance
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: No se puede importar job_shop_lib. "
        "Asegúrate de que 'job-shop-lib' esté instalado y funcional. Deteniendo ejecución."
    ) from e


def save_instance_dzn(instance_obj: JobShopInstance, instance_name: str) -> str:
    """Convierte JobShopInstance a un archivo de datos .dzn."""
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
                        f"No se pudo determinar el ID de la máquina para la operación {op}."
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
    """Extrae makespan (END), tiempo total y estado binario."""
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
    except Exception:
        stats["runtime"] = TIME_LIMIT_S
        stats["solved_binary"] = 0

    return stats


def execute_minizinc_solver(
    solver: str, strategy: str, dzn_path: str
) -> Dict[str, Any]:
    """Ejecuta un solver de MiniZinc con su heurística interna."""

    cmd = [
        "minizinc",
        "--solver",
        solver,
        "--statistics",
        "--time-limit",
        str(TIME_LIMIT_MS),
        MODEL_MZN_PATH,
        dzn_path,
    ]

    stdout = ""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIME_LIMIT_S + 5
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired:
        stdout = f"%%%mzn-stat: solveTime={TIME_LIMIT_S}\n"
    except FileNotFoundError:
        raise RuntimeError(
            f"FATAL ERROR: MiniZinc no encontrado. Asegúrate de que esté en tu PATH."
        )

    return parse_minizinc_stats(stdout)


def calculate_relative_performance_score(
    stats: Dict[str, Any], optimum: float
) -> float:
    """
    Calcula un score que combina runtime y la desviación del makespan óptimo (PAR-like).
    Cuanto MENOR sea el score, MEJOR es el solver.
    El score usa PENALTY_FACTOR_K si no se encontró una solución válida.
    """
    runtime = stats["runtime"]
    makespan_found = stats["makespan"]

    if makespan_found == float("inf") or optimum <= 0 or optimum == float("inf"):
        return TIME_LIMIT_S * PENALTY_FACTOR_K

    else:
        gap = (makespan_found - optimum) / optimum

        score = runtime + (TIME_LIMIT_S * gap)

        return min(score, TIME_LIMIT_S * PENALTY_FACTOR_K)


def prepare_data_and_ground_truth_minizinc():
    """Genera instancias aleatorias de JSP y ejecuta el pipeline de MiniZinc."""

    print("--- 1. Ejecutando Pipeline MiniZinc para Instancias Académicas ---")
    print(f"Instancias: {', '.join(INSTANCE_NAMES)}")
    print(f"Límite de Tiempo por Solver/Instancia: {TIME_LIMIT_S}s")
    print(f"Factor de Penalización (K): {PENALTY_FACTOR_K}")
    print("-" * 75)

    solver_keys = [name for _, _, name in SOLVER_STRATEGIES]

    csv_rows = [
        [
            "Instance_Name",
            "Raw_Text_Path",
            "N_Jobs",
            "N_Machines",
            "Best_Makespan_Found",
            "Optimum",
        ]
        + [f"{s}_Runtime_s" for s in solver_keys]
        + [f"{s}_Score_S_rel" for s in solver_keys]
    ]

    total_instances = len(INSTANCE_NAMES)

    for index, name in enumerate(INSTANCE_NAMES):
        start_time_instance = time.time()

        print(f"🚀 Procesando Instancia {index + 1}/{total_instances}: {name}")

        try:
            instance_obj = load_benchmark_instance(name)
            dzn_path = save_instance_dzn(instance_obj, name)
            raw_text_path = dzn_path

            optimum = instance_obj.metadata.get("optimum", float("inf"))
            if not isinstance(optimum, (int, float)) or optimum <= 0:
                print(
                    f"  ⚠️ Advertencia: Óptimo no válido ({optimum}), usando 1.0 temporalmente para score."
                )
                optimum = 1.0

            print(
                f"  - Dimensiones: {instance_obj.num_jobs}x{instance_obj.num_machines}. Óptimo Conocido: {optimum:.0f}"
            )

            all_results = {}
            best_makespan = float("inf")

            for solver, strategy, key in SOLVER_STRATEGIES:
                solver_start_time = time.time()
                print(f"  -> Ejecutando Solver {key}...")

                stats = execute_minizinc_solver(solver, strategy, dzn_path)
                stats["score"] = calculate_relative_performance_score(stats, optimum)
                all_results[key] = stats

                makespan_str = (
                    f"{stats['makespan']:.0f}"
                    if stats["makespan"] < float("inf")
                    else "inf"
                )
                runtime_s = stats["runtime"]

                print(
                    f"     - Resultado: Makespan={makespan_str}, Score={stats['score']:.2f}"
                )
                print(f"     - Tiempo MiniZinc (Reportado): {runtime_s:.3f}s")

                if stats["makespan"] < best_makespan:
                    best_makespan = stats["makespan"]

            best_makespan_str = (
                f"{best_makespan:.0f}" if best_makespan < float("inf") else "inf"
            )

            row = [
                name,
                raw_text_path,
                instance_obj.num_jobs,
                instance_obj.num_machines,
                best_makespan_str,
                f"{optimum:.0f}" if optimum < float("inf") else "inf",
            ]

            row.extend([f"{all_results[key]['runtime']:.3f}" for key in solver_keys])
            row.extend([f"{all_results[key]['score']:.2f}" for key in solver_keys])

            csv_rows.append(row)

            end_time_instance = time.time()
            elapsed_time_instance = end_time_instance - start_time_instance

            print(
                f"✅ Instancia {name} Finalizada. Mejor Makespan: {best_makespan_str}. "
                f"Tiempo Total (Wall-Time): {elapsed_time_instance:.3f}s."
            )
            print("-" * 75)

        except Exception as e:
            end_time_instance = time.time()
            elapsed_time_instance = end_time_instance - start_time_instance
            print(f"❌ ERROR fatal al procesar instancia {name}. Detalles: {e}")
            print(f"Tiempo Transcurrido hasta el fallo: {elapsed_time_instance:.3f}s.")
            print("-" * 75)

    csv_path = os.path.join(OUTPUT_DIR, "ground_truth_jsp_minizinc_dataset.csv")
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


if __name__ == "__main__":
    prepare_data_and_ground_truth_minizinc()
