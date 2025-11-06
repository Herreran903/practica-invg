# ==============================================
# data_preparer_gen.py — Generación aleatoria + benchmarking MiniZinc
# ------------------------------------------------
# Propósito:
#   - Generar instancias JSP balanceadas (N×M) con job-shop-lib.
#   - Convertir cada instancia a .dzn para MiniZinc.
#   - Ejecutar múltiples solvers/configs (CP y MIP), tiempos y seeds.
#   - Parsear estadísticas (makespan, runtime, wall) y elegir ganador por bloque.
#   - Guardar un CSV con métricas por solver y bloque (T, seed).
# Uso:
#   python -m generation.data_preparer_gen
# Requisitos:
#   - MiniZinc en PATH; modelos en generation/model*.mzn.
#   - job-shop-lib instalado.
#   - NumPy (para utilidades y guardado), Pandas (solo en image_converter), etc.
# ==============================================

import csv
import os
import re
import subprocess
import time
from typing import Any, Dict, List

import numpy as np

# =========================
#   CONFIGURACIÓN GENERAL
# =========================

OUTPUT_DIR = "jsp_cnn_data_gen"
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de salida {OUTPUT_DIR}. Detalles: {e}"
    )

# --- Modelos por paradigma ---
CP_MODEL_MZN_PATH = "generation/model.mzn"  # CP / LCG / CP-SAT (disyuntivo)
MIP_MODEL_MZN_PATH = (
    "generation/model_linear.mzn"  # MIP (lineal). Si no existe, pon None.
)

# --- Límites y seeds ---
TIME_LIMITS_MS = [5000, 30000, 60000]  # 5s, 30s, 60s
RANDOM_SEEDS = [1, 2, 3]

# (Compatibilidad con funciones que esperan estas constantes)
TIME_LIMIT_MS = 60000
TIME_LIMIT_S = TIME_LIMIT_MS / 1000

LOG_DIR = os.path.join(OUTPUT_DIR, "solver_logs_temp")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except OSError as e:
    raise RuntimeError(
        f"❌ ERROR: No se pudo crear el directorio de logs temporal {LOG_DIR}. Detalles: {e}"
    )

# Candidatos (se filtrarán por disponibilidad y modelo)
# id (--solver), key (etiqueta), type ("cp" o "mip"), opts
SOLVER_CANDIDATES = [
    (
        "gecode",
        "GECODE_FF",
        "cp",
        {"strategy": "first_fail", "supports_seed": True, "inject_search": True},
    ),
    (
        "chuffed",
        "CHUFFED_IO",
        "cp",
        {"strategy": "input_order", "supports_seed": True, "inject_search": True},
    ),
    (
        "cp-sat",
        "CPSAT_FF",
        "cp",
        {"strategy": "first_fail", "supports_seed": True, "inject_search": True},
    ),
    (
        "coin-bc",
        "CBC_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "scip",
        "SCIP_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "highs",
        "HIGHS_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "cplex",
        "CPLEX_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "gurobi",
        "GUROBI_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
]

# Generación de instancias (jobs, machines, count por tamaño)
GENERATION_CASES = [
    (4, 4, 5),
    (6, 6, 5),
    (8, 8, 5),
    (10, 10, 5),
    # (12, 12, 5),
    # (15, 15, 5),
]

# =========================
#   IMPORTS DE LIBRERÍA
# =========================

try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.generation import GeneralInstanceGenerator
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: No se puede importar job_shop_lib. "
        "Asegúrate de que 'job-shop-lib' esté instalado y funcional. Deteniendo ejecución."
    ) from e


# =========================
#   UTILIDADES MINI-ZINC
# =========================


def _list_available_solver_ids() -> List[str]:
    """Devuelve ids detectados por `minizinc --solvers` en minúscula.
    Si el comando falla, retorna lista vacía (se filtrará después).
    """
    try:
        out = subprocess.run(
            ["minizinc", "--solvers"], capture_output=True, text=True, timeout=10
        )
        text = out.stdout.lower()
    except Exception:
        text = ""
    found = set()
    for sid, _, _, _ in SOLVER_CANDIDATES:
        if sid in text:
            found.add(sid)
    return list(found)


def build_solver_configs() -> List[tuple]:
    """
    Filtra SOLVER_CANDIDATES por disponibilidad en esta máquina y por modelo requerido.
    Retorna lista de tuplas (solver_id, key, type, opts) listas para ejecución.
    """
    available = set(_list_available_solver_ids())
    configs = []
    for sid, key, stype, opts in SOLVER_CANDIDATES:
        if sid not in available:
            continue
        if stype == "mip" and not MIP_MODEL_MZN_PATH:
            # No hay modelo lineal → saltamos MIP
            continue
        configs.append((sid, key, stype, opts))
    if not configs:
        raise RuntimeError(
            "No hay solvers utilizables (revisa instalación o rutas de modelo)."
        )
    return configs


def save_instance_dzn(instance_obj: JobShopInstance, instance_name: str) -> str:
    """Convierte JobShopInstance a DZN plano para MiniZinc.
    Escribe JOBS, MACHINES, PROC_TIME (duraciones) y MACHINE_OF_OP (máquinas).
    """
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
                mach_row.append(machine_id_0_based + 1)  # máquinas 1..M

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
    """
    Extrae Makespan (END) y runtime desde la salida de MiniZinc.
    Retorna dict con: makespan, runtime, had_time_tag, solved_binary.
    """
    stats = {"makespan": float("inf"), "runtime": TIME_LIMIT_S, "had_time_tag": False}
    try:
        all_makespans = re.findall(r"END=\s*(\d+)", mzn_text)
        if all_makespans:
            stats["makespan"] = min(float(m) for m in all_makespans)

        t = re.search(r"%%%mzn-stat:\s*solveTime=([0-9\.]+)", mzn_text)
        if not t:
            t = re.search(r"%%%mzn-stat:\s*time=([0-9\.]+)", mzn_text)

        if t:
            stats["runtime"] = float(t.group(1))
            stats["had_time_tag"] = True

        stats["solved_binary"] = 1 if stats["makespan"] < float("inf") else 0
    except Exception as e:
        print(f"❌ ERROR: Fallo al parsear estadísticas de MiniZinc. Detalles: {e}")
        stats["runtime"] = TIME_LIMIT_S
        stats["solved_binary"] = 0
        stats["had_time_tag"] = False
    return stats


def _patch_model_if_needed(
    model_path: str, inject: bool, strategy: str, tmp_out_path: str
) -> str:
    """
    Para modelos CP/LCG/CP-SAT: si 'inject' es True, reemplaza la línea de solve
    por una variante con búsqueda explícita (int_search) usando la 'strategy' dada.
    Para MIP o si no se desea inyección, copia el modelo tal cual a 'tmp_out_path'.
    """
    try:
        with open(model_path, "r") as f:
            mzn = f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"FATAL ERROR: No se encontró el modelo en la ruta: {model_path}"
        )

    if inject and strategy:
        m = re.search(r"^\s*solve\s+minimize\s+END_MAKESPAN\s*;", mzn, re.MULTILINE)
        if m:
            repl = (
                f"solve :: int_search(S_FLAT, {strategy}, indomain_min, complete) minimize END_MAKESPAN;"
            )
            mzn = mzn.replace(m.group(0), repl)

    try:
        with open(tmp_out_path, "w") as f:
            f.write(mzn)
    except Exception as e:
        raise RuntimeError(
            f"❌ ERROR: Fallo al escribir modelo temporal {tmp_out_path}. Detalles: {e}"
        )
    return tmp_out_path


def execute_minizinc_solver_generic(
    solver_id: str,
    key: str,
    stype: str,
    opts: Dict[str, Any],
    dzn_path: str,
    time_limit_ms: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Ejecuta un solver (CP o MIP) con el modelo correcto, inyectando búsqueda y seed
    solo cuando aplica. Retorna estadísticas parseadas + metadatos (solver, key, etc.).
    """
    time_limit_s = time_limit_ms / 1000.0
    is_cp = stype == "cp"
    inject = bool(opts.get("inject_search", False))
    strat = opts.get("strategy")
    use_seed = bool(opts.get("supports_seed", False))

    # Modelo según paradigma
    model_path = CP_MODEL_MZN_PATH if is_cp else MIP_MODEL_MZN_PATH
    if not model_path or not os.path.exists(model_path):
        # Cuando MIP no está disponible, devolvemos un resultado 'no resuelto'.
        return {
            "makespan": float("inf"),
            "runtime": time_limit_s,
            "wall_time_s": 0.0,
            "solved_binary": 0,
        }

    # Parche temporal (o copia tal cual)
    tmp_mzn = os.path.join(LOG_DIR, f"__tmp_{key}_seed{seed}_T{time_limit_ms}.mzn")
    patched_path = _patch_model_if_needed(model_path, inject and is_cp, strat, tmp_mzn)

    # Soporte de librerías dinámicas para ciertos solvers (macOS ARM típico)
    extra = []
    if solver_id == "cplex":
        dll = os.environ.get("CPLEX_DLL") or os.path.join(
            os.environ.get("CPLEX_STUDIO_DIR", "/Applications/CPLEX_Studio2211"),
            "cplex", "bin", "arm64_osx", "libcplex.dylib"
        )
        if os.path.exists(dll):
            extra += ["--cplex-dll", dll]
    elif solver_id == "highs":
        dll = os.environ.get("HIGHS_DLL") or "/opt/homebrew/opt/highs/lib/libhighs.dylib"
        if os.path.exists(dll):
            extra += ["--highs-dll", dll]

    # Construcción del comando MiniZinc
    cmd = [
        "minizinc",
        "--solver",
        solver_id,
        "--statistics",
        "--time-limit",
        str(time_limit_ms),
        patched_path,
        dzn_path,
    ]

    if extra:
        cmd[1:1] = extra  # Inserta justo después de 'minizinc'
    # Semilla solo para CP/LCG/CP-SAT que la soporten
    if is_cp and use_seed:
        cmd[1:1] = ["--random-seed", str(seed)]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=time_limit_s + 10
        )
        stdout = proc.stdout
        stderr = proc.stderr
        ret = proc.returncode
    except subprocess.TimeoutExpired:
        stdout = f"%%%mzn-stat: solveTime={time_limit_s}\n"
        stderr = ""
        ret = 124  # típico timeout
    except FileNotFoundError:
        raise RuntimeError(
            "FATAL ERROR: MiniZinc no encontrado. Asegúrate de que esté en tu PATH."
        )
    finally:
        wall = time.time() - t0
        try:
            os.unlink(patched_path)
        except Exception:
            pass

    stats = parse_minizinc_stats(stdout)

    # Si NO hubo tag de tiempo y el wall es muy inferior al cap, usa wall.
    if not stats.get("had_time_tag", False):
        stats["runtime"] = min(stats.get("runtime", TIME_LIMIT_S), wall)

    # Adjunta metadatos útiles para depurar
    stats.update(
        {
            "solver": solver_id,
            "key": key,
            "type": stype,
            "time_limit_s": time_limit_s,
            "seed": seed,
            "wall_time_s": wall,
            "returncode": ret,
        }
    )

    # Log suave si el solver terminó rápido sin tag de tiempo o con error
    if (not stats.get("had_time_tag", False)) or (ret != 0):
        err_snip = (stderr or "").strip().splitlines()
        err_snip = " | ".join(err_snip[:3])[:200]  # primeras líneas, truncado
        print(
            f"     ⚠️ nota: ret={ret}, had_time_tag={stats['had_time_tag']}, stderr='{err_snip}'"
        )

    return stats


# =========================
#   PIPELINE PRINCIPAL
# =========================


def prepare_data_and_ground_truth_minizinc_gen():
    """
    Genera instancias aleatorias de JSP y ejecuta MiniZinc con múltiples solvers,
    tiempos límite (5/30/60s) y semillas (1,2,3). Para cada bloque (T, seed):
      - Ejecuta todas las configs disponibles.
      - Marca 'Winner_Key' como el solver resuelto con menor runtime.

    CSV de salida:
      Instance_Name, Raw_Text_Path, N_Jobs, N_Machines, Time_Limit_s, Seed,
      Winner_Key, y por cada key: _Runtime_s, _Makespan, _Wall_s.

    Devuelve: ruta al CSV generado en OUTPUT_DIR.
    """
    all_instances: List[JobShopInstance] = []

    print("--- 1. Generando Instancias JSP Balanceadas ---")
    generator = GeneralInstanceGenerator(duration_range=(1, 20), seed=42)

    instance_count = 0
    for jobs, machines, count in GENERATION_CASES:
        for i in range(count):
            instance_obj = generator.generate(num_jobs=jobs, num_machines=machines)
            instance_obj.name = f"GEN_{jobs}x{machines}_{i+1}"
            all_instances.append(instance_obj)
            instance_count += 1

    print(f"✅ Generadas un total de {instance_count} instancias.")
    print("-" * 50)

    # Construye la lista de configuraciones utilizables en esta máquina
    SOLVER_CONFIGS = build_solver_configs()
    config_keys = [key for (_, key, _, _) in SOLVER_CONFIGS]

    # Cabecera del CSV
    header = [
        "Instance_Name",
        "Raw_Text_Path",
        "N_Jobs",
        "N_Machines",
        "Time_Limit_s",
        "Seed",
        "Winner_Key",
    ]
    for key in config_keys:
        header += [f"{key}_Runtime_s", f"{key}_Makespan", f"{key}_Wall_s"]
    csv_rows = [header]

    # Recorre instancias
    for index, instance_obj in enumerate(all_instances):
        name = instance_obj.name
        print(f"--------------------------------------------------")
        print(
            f"🚀 Procesando Instancia {index + 1}/{instance_count}: {name} ({instance_obj.num_jobs}x{instance_obj.num_machines})"
        )

        try:
            dzn_path = save_instance_dzn(instance_obj, name)
            raw_text_path = dzn_path

            for T_ms in TIME_LIMITS_MS:
                T_s = T_ms / 1000.0
                for seed in RANDOM_SEEDS:
                    print(f"── Bloque @ {int(T_s)}s, seed={seed}")
                    results: Dict[str, Dict[str, Any]] = {}

                    for solver_id, key, stype, opts in SOLVER_CONFIGS:
                        print(f"  -> {key} @ {int(T_s)}s seed={seed}")
                        stats = execute_minizinc_solver_generic(
                            solver_id, key, stype, opts, dzn_path, T_ms, seed
                        )
                        results[key] = stats
                        mk = (
                            "inf"
                            if stats["makespan"] == float("inf")
                            else f"{int(stats['makespan'])}"
                        )
                        print(
                            f"     • solved={stats['solved_binary']}  makespan={mk}  "
                            f"runtime={stats['runtime']:.3f}s  wall={stats['wall_time_s']:.3f}s"
                        )

                    # Ganador entre los que resolvieron: menor runtime
                    solved_keys = [
                        k for k, s in results.items() if s.get("solved_binary", 0) == 1
                    ]
                    winner_key = (
                        min(solved_keys, key=lambda k: results[k]["runtime"]) if solved_keys else "NONE"
                    )

                    # Resumen del bloque
                    if solved_keys:
                        best_key = min(solved_keys, key=lambda k: results[k]["runtime"])
                        best_rt = results[best_key]["runtime"]
                        print(
                            f"  ⇒ Resumen: Winner={winner_key} | solved={len(solved_keys)}/{len(SOLVER_CONFIGS)} | best_runtime={best_rt:.3f}s"
                        )
                    else:
                        print(
                            f"  ⇒ Resumen: Winner=NONE | solved=0/{len(SOLVER_CONFIGS)}"
                        )

                    row = [
                        name,
                        raw_text_path,
                        instance_obj.num_jobs,
                        instance_obj.num_machines,
                        f"{T_s:.0f}",
                        seed,
                        winner_key,
                    ]
                    for key in config_keys:
                        st = results[key]
                        ms = st["makespan"] if st["makespan"] != float("inf") else "inf"
                        row += [f"{st['runtime']:.3f}", ms, f"{st['wall_time_s']:.3f}"]
                    csv_rows.append(row)

            print("  ✅ OK")
        except Exception as e:
            print(f"❌ ERROR fatal al procesar instancia {name}. Detalles: {e}")

    # Escribir CSV
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


# =========================
#   MAIN
# =========================

if __name__ == "__main__":
    prepare_data_and_ground_truth_minizinc_gen()
