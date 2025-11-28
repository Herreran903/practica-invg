import csv
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.generation import GeneralInstanceGenerator
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: No se puede importar job_shop_lib. "
        "Asegúrate de que 'job-shop-lib' esté instalado y funcional."
    ) from e


def _list_available_solver_ids() -> List[str]:
    """Devuelve ids detectados por `minizinc --solvers` en minúscula."""
    try:
        out = subprocess.run(
            ["minizinc", "--solvers"], capture_output=True, text=True, timeout=10
        )
        text = out.stdout.lower()
    except Exception:
        text = ""
    tokens = set()
    for tok in re.findall(r"[a-z0-9\-]+", text):
        tokens.add(tok.strip())
    return list(tokens)


def _build_solver_configs(
    solver_candidates: List[Tuple[str, str, str, Dict[str, Any]]],
    mip_model_path: Optional[str],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Filtra solver_candidates por disponibilidad y por modelo requerido.
    Retorna tuplas (solver_id, key, type, opts).
    """
    available = set(_list_available_solver_ids())
    configs: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for sid, key, stype, opts in solver_candidates:
        if sid not in available:
            continue
        if stype == "mip" and not mip_model_path:
            continue
        configs.append((sid, key, stype, opts))
    if not configs:
        raise RuntimeError(
            "No hay solvers utilizables (revisa instalación o rutas de modelo)."
        )
    return configs


def _patch_model_if_needed(
    model_path: str, inject: bool, strategy: Optional[str], tmp_out_path: str
) -> str:
    """
    Para modelos CP/LCG/CP-SAT: si 'inject' es True, reemplaza la línea de solve
    por una variante con int_search usando 'strategy'. Para MIP o sin inyección,
    copia el modelo tal cual a 'tmp_out_path'.
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
            repl = f"solve :: int_search(S_FLAT, {strategy}, indomain_min, complete) minimize END_MAKESPAN;"
            mzn = mzn.replace(m.group(0), repl)

    with open(tmp_out_path, "w") as f:
        f.write(mzn)
    return tmp_out_path


def _save_instance_dzn(
    instance_obj: JobShopInstance, instance_name: str, output_dir: str
) -> str:
    """Convierte JobShopInstance a DZN plano para MiniZinc."""
    dzn_path = os.path.join(output_dir, f"{instance_name}.dzn")

    proc_time_rows: List[List[int]] = []
    machine_of_op_rows: List[List[int]] = []

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
                    f"Estructura inesperada para la máquina en operación {op}."
                )
            mach_row.append(machine_id_0_based + 1)

        proc_time_rows.append(pt_row)
        machine_of_op_rows.append(mach_row)

    def _flat_array(matrix_2d: List[List[int]]) -> str:
        flat_list = [item for row in matrix_2d for item in row]
        return f"[{', '.join(map(str, flat_list))}]"

    with open(dzn_path, "w") as f:
        f.write(f"JOBS = {instance_obj.num_jobs};\n")
        f.write(f"MACHINES = {instance_obj.num_machines};\n\n")
        f.write(
            f"PROC_TIME = array2d(SET_JOBS, SET_POS, {_flat_array(proc_time_rows)});\n"
        )
        f.write(
            f"MACHINE_OF_OP = array2d(SET_JOBS, SET_POS, {_flat_array(machine_of_op_rows)});\n"
        )

    return dzn_path


def _parse_minizinc_stats(mzn_text: str, time_limit_s: float) -> Dict[str, Any]:
    """
    Extrae Makespan (END) y runtime desde la salida de MiniZinc.
    Retorna: {makespan, runtime, had_time_tag, solved_binary}
    """
    stats = {"makespan": float("inf"), "runtime": time_limit_s, "had_time_tag": False}
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
    except Exception:
        stats["runtime"] = time_limit_s
        stats["solved_binary"] = 0
        stats["had_time_tag"] = False
    return stats


def _execute_minizinc_solver_generic(
    solver_id: str,
    key: str,
    stype: str,
    opts: Dict[str, Any],
    dzn_path: str,
    time_limit_ms: int,
    cp_model_path: str,
    mip_model_path: Optional[str],
    log_dir: str,
) -> Dict[str, Any]:
    """
    Ejecuta un solver (CP o MIP) con el modelo correcto, inyectando búsqueda y seed
    solo cuando aplica. Devuelve estadísticas parseadas + metadatos.
    """
    time_limit_s = time_limit_ms / 1000.0
    is_cp = stype == "cp"
    inject = bool(opts.get("inject_search", False))
    strat = opts.get("strategy")
    use_seed = bool(opts.get("supports_seed", False))

    model_path = cp_model_path if is_cp else mip_model_path
    if not model_path or not os.path.exists(model_path):
        return {
            "makespan": float("inf"),
            "runtime": time_limit_s,
            "wall_time_s": 0.0,
            "solved_binary": 0,
            "had_time_tag": False,
        }

    tmp_mzn = os.path.join(log_dir, f"__tmp_{key}_T{time_limit_ms}.mzn")
    patched_path = _patch_model_if_needed(model_path, inject and is_cp, strat, tmp_mzn)

    extra: List[str] = []
    if solver_id == "cplex":
        dll = os.environ.get("CPLEX_DLL") or os.path.join(
            os.environ.get("CPLEX_STUDIO_DIR", "/Applications/CPLEX_Studio2211"),
            "cplex",
            "bin",
            "arm64_osx",
            "libcplex.dylib",
        )
        if os.path.exists(dll):
            extra += ["--cplex-dll", dll]
    elif solver_id == "highs":
        dll = (
            os.environ.get("HIGHS_DLL") or "/opt/homebrew/opt/highs/lib/libhighs.dylib"
        )
        if os.path.exists(dll):
            extra += ["--highs-dll", dll]

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
        cmd[1:1] = extra

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
        ret = 124
    except FileNotFoundError:
        raise RuntimeError(
            "FATAL ERROR: MiniZinc no encontrado. Asegúrate de que esté en el PATH."
        )
    finally:
        wall = time.time() - t0
        try:
            os.unlink(patched_path)
        except Exception:
            pass

    stats = _parse_minizinc_stats(stdout, time_limit_s)
    if not stats.get("had_time_tag", False):
        stats["runtime"] = min(stats.get("runtime", time_limit_s), wall)

    stats.update(
        {
            "solver": solver_id,
            "key": key,
            "type": stype,
            "time_limit_s": time_limit_s,
            "wall_time_s": wall,
            "returncode": ret,
        }
    )

    if ret != 0:
        err_snip = (stderr or "").strip().splitlines()
        err_snip = " | ".join(err_snip[:3])[:200]
        print(
            f"     nota: ret={ret}, had_time_tag={stats['had_time_tag']}, stderr='{err_snip}'"
        )

    return stats


def prepare_data_and_ground_truth_minizinc_gen(
    *,
    output_dir: str,
    out_name: str,
    cp_model_path: str,
    mip_model_path: Optional[str],
    time_limits_ms: List[int],
    solver_candidates: List[Tuple[str, str, str, Dict[str, Any]]],
    random_seeds: List[int],
    generation_cases: List[Tuple[int, int, int]],
) -> str:
    """
    Genera instancias aleatorias de JSP y ejecuta MiniZinc con múltiples solvers,
    tiempos y seeds. Para cada bloque (T, seed):
      - Ejecuta todas las configs disponibles.
      - Winner_Key = solver resuelto con menor runtime.
    CSV de salida:
      Instance_Name, Raw_Text_Path, N_Jobs, N_Machines, Time_Limit_s, Seed, Winner_Key,
      y por cada key: _Runtime_s, _Makespan, _Wall_s.
    Devuelve: ruta al CSV generado en output_dir/out_name.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "solver_logs_temp")
    os.makedirs(log_dir, exist_ok=True)

    all_instances: List[JobShopInstance] = []

    print("=== INICIO (GENERADO – MiniZinc) ===")
    print("[1/3] Generando instancias JSP balanceadas ...")
    generator = GeneralInstanceGenerator(duration_range=(1, 20), seed=42)

    instance_count = 0
    for jobs, machines, count in generation_cases:
        for i in range(count):
            inst = generator.generate(num_jobs=jobs, num_machines=machines)
            inst.name = f"GEN_{jobs}x{machines}_{i+1}"
            all_instances.append(inst)
            instance_count += 1

    print(f"Instancias generadas: {instance_count}")
    print("-" * 50)

    print("[2/3] Construyendo configuraciones de solvers disponibles ...")
    solver_configs = _build_solver_configs(solver_candidates, mip_model_path)
    config_keys = [key for (_, key, _, _) in solver_configs]
    print(f"Solvers utilizables: {', '.join([k for (_, k, _, _) in solver_configs])}")

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
    csv_rows: List[List[str]] = [header]

    print("[3/3] Ejecutando benchmarking por bloques (T, seed) ...")

    for index, instance_obj in enumerate(all_instances):
        name = instance_obj.name
        print("--------------------------------------------------")
        print(
            f"- Instancia {index + 1}/{instance_count}: {name} ({instance_obj.num_jobs}x{instance_obj.num_machines})"
        )

        try:
            dzn_path = _save_instance_dzn(instance_obj, name, output_dir)
            raw_text_path = dzn_path

            for T_ms in time_limits_ms:
                T_s = T_ms / 1000.0
                for seed in random_seeds:
                    print(f"  Bloque @ {int(T_s)}s, seed={seed}")
                    results: Dict[str, Dict[str, Any]] = {}

                    for solver_id, key, stype, opts in solver_configs:
                        print(f"    -> {key} @ {int(T_s)}s")
                        stats = _execute_minizinc_solver_generic(
                            solver_id=solver_id,
                            key=key,
                            stype=stype,
                            opts=opts,
                            dzn_path=dzn_path,
                            time_limit_ms=T_ms,
                            cp_model_path=cp_model_path,
                            mip_model_path=mip_model_path,
                            log_dir=log_dir,
                        )
                        stats["seed"] = seed
                        results[key] = stats

                        mk = (
                            "inf"
                            if stats["makespan"] == float("inf")
                            else f"{int(stats['makespan'])}"
                        )
                        print(
                            f"       solved={stats['solved_binary']}  makespan={mk}  "
                            f"runtime={stats['runtime']:.3f}s  wall={stats['wall_time_s']:.3f}s"
                        )

                    solved_keys = [
                        k for k, s in results.items() if s.get("solved_binary", 0) == 1
                    ]
                    winner_key = (
                        min(solved_keys, key=lambda k: results[k]["runtime"])
                        if solved_keys
                        else "NONE"
                    )

                    if solved_keys:
                        best_key = min(solved_keys, key=lambda k: results[k]["runtime"])
                        best_rt = results[best_key]["runtime"]
                        print(
                            f"  Resumen: Winner={winner_key} | solved={len(solved_keys)}/{len(solver_configs)} | best_runtime={best_rt:.3f}s"
                        )
                    else:
                        print(
                            f"  Resumen: Winner=NONE | solved=0/{len(solver_configs)}"
                        )

                    row: List[str] = [
                        name,
                        raw_text_path,
                        str(instance_obj.num_jobs),
                        str(instance_obj.num_machines),
                        f"{T_s:.0f}",
                        str(seed),
                        winner_key,
                    ]
                    for key in config_keys:
                        st = results[key]
                        ms = st["makespan"] if st["makespan"] != float("inf") else "inf"
                        row += [f"{st['runtime']:.3f}", ms, f"{st['wall_time_s']:.3f}"]
                    csv_rows.append(row)

            print("  OK")
        except Exception as e:
            print(f"ERROR: Fallo al procesar instancia {name}. Detalles: {e}")

    csv_path = os.path.join(output_dir, out_name)
    print("\nEscribiendo CSV ...")
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Listo: resultados guardados en {csv_path}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Fallo al escribir el archivo CSV {csv_path}. Detalles: {e}"
        )

    print("\n=== FIN ===")
    print(f"Directorio de salida: {os.path.abspath(output_dir)}")
    print(f"CSV: {csv_path}")
    return csv_path
