import csv
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Tuple

try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.benchmarking import load_benchmark_instance
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: No se puede importar job_shop_lib. "
        "Asegúrate de que 'job-shop-lib' esté instalado y funcional."
    ) from e


def _save_instance_dzn(
    instance_obj: JobShopInstance,
    instance_name: str,
    output_dir: str,
) -> str:
    """
    Convierte un JobShopInstance a archivo .dzn legible por MiniZinc y lo guarda en output_dir.
    Escribe:
      - JOBS, MACHINES
      - PROC_TIME (array2d aplanado por filas)
      - MACHINE_OF_OP (array2d aplanado por filas; máquinas 1..M)
    """
    os.makedirs(output_dir, exist_ok=True)
    dzn_path = os.path.join(output_dir, f"{instance_name}.dzn")

    proc_time_rows: List[List[int]] = []
    machine_of_op_rows: List[List[int]] = []

    try:
        for job_ops in instance_obj.jobs:
            pt_row: List[int] = []
            mach_row: List[int] = []
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
                mach_row.append(machine_id_0_based + 1)  # máquinas 1..M

            proc_time_rows.append(pt_row)
            machine_of_op_rows.append(mach_row)
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Fallo al extraer datos de la instancia {instance_name}. Detalles: {e}"
        )

    def _flat_array(matrix_2d: List[List[int]]) -> str:
        flat_list = [item for row in matrix_2d for item in row]
        return f"[{', '.join(map(str, flat_list))}]"

    try:
        with open(dzn_path, "w") as f:
            f.write(f"JOBS = {instance_obj.num_jobs};\n")
            f.write(f"MACHINES = {instance_obj.num_machines};\n\n")
            f.write(
                f"PROC_TIME = array2d(SET_JOBS, SET_POS, {_flat_array(proc_time_rows)});\n"
            )
            f.write(
                f"MACHINE_OF_OP = array2d(SET_JOBS, SET_POS, {_flat_array(machine_of_op_rows)});\n"
            )
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Fallo al escribir el archivo .dzn {dzn_path}. Detalles: {e}"
        )

    return dzn_path


def _parse_minizinc_stats(mzn_text: str, time_limit_s: float) -> Dict[str, Any]:
    """
    Extrae estadísticas básicas de la salida textual de MiniZinc:
      - END=<int> → mejor makespan reportado
      - %%%mzn-stat: solveTime= / time= → tiempo (s) reportado
    """
    stats = {"makespan": float("inf"), "runtime": time_limit_s}

    try:
        all_makespans = re.findall(r"END=\s*(\d+)", mzn_text)
        if all_makespans:
            stats["makespan"] = min(float(m) for m in all_makespans)

        t = re.search(r"%%%mzn-stat:\s*solveTime=([0-9\.]+)", mzn_text)
        if not t:
            t = re.search(r"%%%mzn-stat:\s*time=([0-9\.]+)", mzn_text)

        stats["runtime"] = float(t.group(1)) if t else time_limit_s
        stats["solved_binary"] = 1 if stats["makespan"] < float("inf") else 0
    except Exception:
        stats["runtime"] = time_limit_s
        stats["solved_binary"] = 0

    return stats


def _execute_minizinc_solver(
    solver: str,
    strategy_label: str,  # solo informativo
    dzn_path: str,
    model_mzn_path: str,
    time_limit_ms: int,
    time_limit_s: float,
) -> Dict[str, Any]:
    """Ejecuta MiniZinc con el solver indicado sobre un .dzn y devuelve estadísticas parseadas."""
    cmd = [
        "minizinc",
        "--solver",
        solver,
        "--statistics",
        "--time-limit",
        str(time_limit_ms),
        model_mzn_path,
        dzn_path,
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=time_limit_s + 5
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired:
        stdout = f"%%%mzn-stat: solveTime={time_limit_s}\n"
    except FileNotFoundError:
        raise RuntimeError(
            "FATAL ERROR: MiniZinc no encontrado. Asegúrate de que esté en el PATH."
        )

    return _parse_minizinc_stats(stdout, time_limit_s)


def _relative_performance_score(
    stats: Dict[str, Any],
    optimum: float,
    time_limit_s: float,
    penalty_factor_k: float,
) -> float:
    """
    Score de desempeño relativo (tipo PAR):
      - runtime del solver
      - gap respecto al óptimo
    Menor score ⇒ mejor. Si no hay solución válida, penaliza con K·TIME_LIMIT.
    """
    runtime = stats["runtime"]
    makespan_found = stats["makespan"]

    if makespan_found == float("inf") or optimum <= 0 or optimum == float("inf"):
        return time_limit_s * penalty_factor_k
    gap = (makespan_found - optimum) / optimum
    score = runtime + (time_limit_s * gap)
    return min(score, time_limit_s * penalty_factor_k)


def prepare_data_and_ground_truth_minizinc(
    *,
    output_dir: str,
    out_name: str,
    instance_names: List[str],
    model_mzn_path: str,
    time_limit_ms: int,
    penalty_factor_k: float,
    solver_strategies: List[
        Tuple[str, str, str]
    ],  # (solver_id, strategy_label, key_ident)
) -> str:
    """
    Pipeline académico (JSPLIB) parametrizado (sin constantes internas).

    Parámetros:
      - output_dir, out_name: ruta y nombre del CSV de salida
      - instance_names: lista de instancias (e.g., ["ft06","ft10",...])
      - model_mzn_path: ruta al modelo .mzn
      - time_limit_ms: límite de tiempo por solver/instancia
      - penalty_factor_k: factor K para penalización PAR
      - solver_strategies: lista (solver, strategy_label, key)

    Retorna:
      - Ruta al CSV generado (output_dir/out_name).
    """
    os.makedirs(output_dir, exist_ok=True)
    time_limit_s = time_limit_ms / 1000.0

    print("=== INICIO (ACADÉMICO – MiniZinc) ===")
    print(f"Instancias: {', '.join(instance_names)}")
    print(f"Límite de tiempo por solver/instancia: {time_limit_s:.0f}s")
    print(f"Factor de penalización (K): {penalty_factor_k}")
    print("-" * 75)

    solver_keys = [name for _, _, name in solver_strategies]
    header = (
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
    )
    csv_rows: List[List[str]] = [header]

    total_instances = len(instance_names)
    print("[1/2] Ejecutando solvers y recolectando métricas ...")

    for index, name in enumerate(instance_names):
        t0 = time.time()
        print(f"- Instancia {index + 1}/{total_instances}: {name}")

        try:
            instance_obj = load_benchmark_instance(name)
            dzn_path = _save_instance_dzn(instance_obj, name, output_dir)
            raw_text_path = dzn_path

            optimum = instance_obj.metadata.get("optimum", float("inf"))
            if not isinstance(optimum, (int, float)) or optimum <= 0:
                print(
                    "  Aviso: óptimo no válido; se usará 1.0 temporalmente para score."
                )
                optimum = 1.0

            print(
                f"  Dimensiones: {instance_obj.num_jobs}x{instance_obj.num_machines}. "
                f"Óptimo conocido: {optimum:.0f}"
            )

            all_results: Dict[str, Dict[str, Any]] = {}
            best_makespan = float("inf")

            for solver, strategy, key in solver_strategies:
                print(f"  -> Solver {key} ...")
                stats = _execute_minizinc_solver(
                    solver=solver,
                    strategy_label=strategy,
                    dzn_path=dzn_path,
                    model_mzn_path=model_mzn_path,
                    time_limit_ms=time_limit_ms,
                    time_limit_s=time_limit_s,
                )
                stats["score"] = _relative_performance_score(
                    stats, optimum, time_limit_s, penalty_factor_k
                )
                all_results[key] = stats

                makespan_str = (
                    f"{stats['makespan']:.0f}"
                    if stats["makespan"] < float("inf")
                    else "inf"
                )
                print(
                    f"     Resultado: makespan={makespan_str}, score={stats['score']:.2f}"
                )
                print(f"     Tiempo (reportado MiniZinc): {stats['runtime']:.3f}s")

                if stats["makespan"] < best_makespan:
                    best_makespan = stats["makespan"]

            best_makespan_str = (
                f"{best_makespan:.0f}" if best_makespan < float("inf") else "inf"
            )

            row: List[str] = [
                name,
                raw_text_path,
                str(instance_obj.num_jobs),
                str(instance_obj.num_machines),
                best_makespan_str,
                f"{optimum:.0f}" if optimum < float("inf") else "inf",
            ]
            row.extend([f"{all_results[key]['runtime']:.3f}" for key in solver_keys])
            row.extend([f"{all_results[key]['score']:.2f}" for key in solver_keys])
            csv_rows.append(row)

            elapsed = time.time() - t0
            print(
                f"  Instancia finalizada. Mejor makespan: {best_makespan_str}. "
                f"Tiempo total (wall): {elapsed:.3f}s."
            )
            print("-" * 75)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: Fallo al procesar instancia {name}. Detalles: {e}")
            print(f"Tiempo transcurrido hasta el fallo: {elapsed:.3f}s.")
            print("-" * 75)

    csv_path = os.path.join(output_dir, out_name)
    print("\n[2/2] Escribiendo CSV ...")
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"[2/2] Listo: resultados guardados en {csv_path}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Fallo al escribir el archivo CSV {csv_path}. Detalles: {e}"
        )

    print("\n=== FIN ===")
    print(f"Directorio de salida: {os.path.abspath(output_dir)}")
    print(f"CSV: {csv_path}")
    return csv_path
