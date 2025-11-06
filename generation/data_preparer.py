# ==============================================
# data_preparer.py — Generación académica (JSPLIB) + ejecución MiniZinc
# ------------------------------------------------
# Propósito:
#   - Cargar instancias clásicas del Job Shop (FT06, FT10, LA01, ABZ5).
#   - Convertir cada instancia a .dzn para MiniZinc.
#   - Ejecutar distintos solvers con límite de tiempo fijo.
#   - Parsear estadísticas (makespan, runtime) y derivar un score relativo.
#   - Guardar un CSV con el mejor makespan, métricas por solver y rutas a .dzn.
# Uso directo:
#   python -m generation.data_preparer
# Notas:
#   - Requiere MiniZinc en PATH.
#   - Requiere 'job-shop-lib' para cargar instancias JSPLIB.
#   - No modifica la semántica original; solo añade comentarios y docstrings.
# ==============================================

import csv
import os
import re
import subprocess
import time
from typing import Any, Dict

# Directorio donde se almacenan salidas intermedias (.dzn) y el CSV final.
OUTPUT_DIR = "jsp_cnn_data_mzn"
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de salida {OUTPUT_DIR}. Detalles: {e}"
    )

# Conjunto de instancias académicas a evaluar.
INSTANCE_NAMES = ["ft06", "ft10", "la01", "abz5"]

# Ruta del modelo MiniZinc a usar (disyuntivo JSP).
MODEL_MZN_PATH = "jobshop_model.mzn"

# Configuración de tiempos y penalización.
TIME_LIMIT_MS = 60000
TIME_LIMIT_S = TIME_LIMIT_MS / 1000
PENALTY_FACTOR_K = 10.0

# Lista de solvers/estrategias a ejecutar. 'key' es la etiqueta que se usará en CSV.
SOLVER_STRATEGIES = [
    ("gecode", "default", "GECODE_DEFAULT"),
    ("chuffed", "default", "CHUFFED_DEFAULT"),
]

# Directorio temporal para volcar logs/modelos parches si se necesitara.
LOG_DIR = os.path.join(OUTPUT_DIR, "solver_logs_temp")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de logs temporal {LOG_DIR}. Detalles: {e}"
    )

# Importación de la librería de instancias y utilidades del JSP.
try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.benchmarking import load_benchmark_instance
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: No se puede importar job_shop_lib. "
        "Asegúrate de que 'job-shop-lib' esté instalado y funcional. Deteniendo ejecución."
    ) from e


def save_instance_dzn(instance_obj: JobShopInstance, instance_name: str) -> str:
    """
    Convierte un objeto JobShopInstance a un archivo de datos .dzn legible por MiniZinc.

    Qué escribe:
      - JOBS y MACHINES (enteros).
      - PROC_TIME: matriz 2D aplanada por filas con las duraciones.
      - MACHINE_OF_OP: matriz 2D aplanada con id de máquina (1..M) por operación.

    Retorna:
      - Ruta absoluta (o relativa) del .dzn generado.
    """
    dzn_path = os.path.join(OUTPUT_DIR, f"{instance_name}.dzn")

    # Se construyen matrices "planas" leyendo cada operación de cada job.
    proc_time_rows = []
    machine_of_op_rows = []

    try:
        for job_ops in instance_obj.jobs:
            pt_row = []
            mach_row = []
            for op in job_ops:
                # Duración de la operación → PROC_TIME
                pt_row.append(op.duration)

                # Id de la primera máquina candidata (0-based o por objeto con .id)
                machine_info = op.machines[0]
                if isinstance(machine_info, int):
                    machine_id_0_based = machine_info
                elif hasattr(machine_info, "id"):
                    machine_id_0_based = machine_info.id
                else:
                    raise AttributeError(
                        f"No se pudo determinar el ID de la máquina para la operación {op}."
                    )

                # MiniZinc espera máquinas 1..M
                mach_row.append(machine_id_0_based + 1)

            proc_time_rows.append(pt_row)
            machine_of_op_rows.append(mach_row)
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Fallo al extraer datos de la instancia {instance_name}. Detalles: {e}"
        )

    def format_mzn_flat_array(matrix_2d):
        # Aplana la matriz 2D por filas y la formatea como [a, b, c, ...]
        flat_list = [item for row in matrix_2d for item in row]
        return f"[{', '.join(map(str, flat_list))}]"

    try:
        with open(dzn_path, "w") as f:
            # Tamaños
            f.write(f"JOBS = {instance_obj.num_jobs};\n")
            f.write(f"MACHINES = {instance_obj.num_machines};\n\n")
            # Matrices 2D aplanadas
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
    """
    Extrae estadísticas básicas de la salida textual de MiniZinc.

    Busca:
      - END=xxxx → mejor makespan reportado.
      - %%%mzn-stat: solveTime=... o time=... → tiempo (s) reportado.

    Si no hay solución, makespan = inf y solved_binary = 0.
    """
    stats = {"makespan": float("inf"), "runtime": TIME_LIMIT_S}

    try:
        # MiniZinc puede imprimir varias soluciones. Tomamos el mínimo END observado.
        all_makespans = re.findall(r"END=\s*(\d+)", mzn_text)

        if all_makespans:
            best_makespan = min(float(m) for m in all_makespans)
            stats["makespan"] = best_makespan

        # Distintas versiones taggean solveTime o time.
        t = re.search(r"%%%mzn-stat: solveTime=([0-9\.]+)", mzn_text)
        if not t:
            t = re.search(r"%%%mzn-stat: time=([0-9\.]+)", mzn_text)

        stats["runtime"] = float(t.group(1)) if t else TIME_LIMIT_S
        stats["solved_binary"] = 1 if stats["makespan"] < float("inf") else 0
    except Exception:
        # En caso de parseo fallido, caemos a valores conservadores.
        stats["runtime"] = TIME_LIMIT_S
        stats["solved_binary"] = 0

    return stats


def execute_minizinc_solver(
    solver: str, strategy: str, dzn_path: str
) -> Dict[str, Any]:
    """
    Ejecuta MiniZinc con el solver indicado sobre un .dzn y devuelve estadísticas parseadas.

    Parámetros:
      - solver: id de solver para 'minizinc --solver <id>'.
      - strategy: etiqueta informativa (la heurística no se inyecta aquí).
      - dzn_path: ruta al archivo de datos .dzn.
    """
    # Comando base 'minizinc ... modelo.mzn datos.dzn'
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
        # Si el proceso supera el timeout externo, simulamos solveTime=cap.
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
    Calcula un score relativo de desempeño (tipo PAR) combinando:
      - runtime del solver,
      - desviación (gap) respecto al óptimo conocido.

    Menor score ⇒ mejor. Si no hay solución válida, penaliza con K·TIME_LIMIT.
    """
    runtime = stats["runtime"]
    makespan_found = stats["makespan"]

    if makespan_found == float("inf") or optimum <= 0 or optimum == float("inf"):
        return TIME_LIMIT_S * PENALTY_FACTOR_K
    else:
        gap = (makespan_found - optimum) / optimum
        # Suma runtime + parte proporcional del tiempo límite por gap.
        score = runtime + (TIME_LIMIT_S * gap)
        # Cota superior de seguridad.
        return min(score, TIME_LIMIT_S * PENALTY_FACTOR_K)


def prepare_data_and_ground_truth_minizinc():
    """
    Pipeline del modo académico (JSPLIB):
      1) Carga cada instancia por nombre (load_benchmark_instance).
      2) Genera su .dzn correspondiente.
      3) Ejecuta todos los solvers configurados y calcula métricas/score.
      4) Escribe un CSV resumen en OUTPUT_DIR.

    Devuelve: ruta al CSV generado.
    """

    print("--- 1. Ejecutando Pipeline MiniZinc para Instancias Académicas ---")
    print(f"Instancias: {', '.join(INSTANCE_NAMES)}")
    print(f"Límite de Tiempo por Solver/Instancia: {TIME_LIMIT_S}s")
    print(f"Factor de Penalización (K): {PENALTY_FACTOR_K}")
    print("-" * 75)

    # Claves/etiquetas que se usarán por columna en el CSV.
    solver_keys = [name for _, _, name in SOLVER_STRATEGIES]

    # Cabecera del CSV: datos generales + columnas por solver (runtime y score).
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
            # 1) Cargar instancia por nombre desde la librería
            instance_obj = load_benchmark_instance(name)
            # 2) Guardar .dzn para MiniZinc
            dzn_path = save_instance_dzn(instance_obj, name)
            raw_text_path = dzn_path

            # Óptimo conocido (si no hay, usar 1.0 para evitar división por cero)
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

            # 3) Ejecutar cada solver y acumular resultados
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

            # 4) Escribir fila con métricas agregadas
            row = [
                name,
                raw_text_path,
                instance_obj.num_jobs,
                instance_obj.num_machines,
                best_makespan_str,
                f"{optimum:.0f}" if optimum < float("inf") else "inf",
            ]

            # Añadir runtimes y scores por solver, en el mismo orden que solver_keys
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

    # Escribir el CSV final con todas las filas
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
    # Permite ejecutar este módulo directamente: genera el CSV académico.
    prepare_data_and_ground_truth_minizinc()
