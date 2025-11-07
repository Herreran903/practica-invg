# -*- coding: utf-8 -*-
"""
Convierte un escenario ASlib en un CSV de "ground truth" para experimentos SAT.

Propósito:
- Lee `algorithm_runs.arff`, pivotea runtimes/estados por instancia/solver,
  determina el solver ganador por instancia y agrega rutas crudas a archivos.
- Produce un CSV listo para pipelines posteriores (features, imágenes, etc.).

Contexto de uso:
- Preparación de datos para selección de solver / aprendizaje automático
  a partir de resultados en formato ASlib.

Dependencias externas clave:
- NumPy, Pandas.

Uso (CLI sugerido desde otro script):
- Llamar `prepare_data_and_ground_truth_aslib(scenario_dir, out_csv, ...)`.
"""

import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_TIMEOUT_S = 5000.0


def _read_text(path: str) -> str:
    """Lee un archivo de texto como UTF-8 (ignorando errores) y retorna su contenido.

    Args:
        path (str): Ruta del archivo.

    Returns:
        str: Contenido del archivo.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_arff_minimal(arff_path: str) -> Tuple[List[str], List[List[str]]]:
    """Parseo mínimo de un ARFF "denso" (sin soporte para filas sparse).

    Extrae nombres de atributos y filas de datos, ignorando comentarios y líneas vacías.
    No soporta registros con formato `{...}` (sparse).

    Args:
        arff_path (str): Ruta al archivo ARFF.

    Returns:
        Tuple[List[str], List[List[str]]]: Nombres de atributos y filas como listas de str.

    Raises:
        ValueError: Si no se encuentra la sección @data.
        NotImplementedError: Si se detecta un registro ARFF en formato sparse.

    Decisión clave:
        Se implementa un parser mínimo (rápido y sin dependencias) suficiente para
        `algorithm_runs.arff`; se excluye sparse para mantener simplicidad.
    """
    # --- Preprocesamiento de líneas: limpiar, quitar vacías y comentarios ---
    lines = []
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            lines.append(s)

    # --- Recorrido de encabezado: recolectar @attribute y ubicar @data ---
    attr_names: List[str] = []
    data_idx = None
    for i, s in enumerate(lines):
        if s.lower().startswith("@attribute"):
            m = re.match(r"@attribute\s+([^\s]+)\s+(.+)", s, re.IGNORECASE)
            if not m:
                continue
            # Nota: se remueven comillas en el nombre del atributo
            attr = m.group(1).strip("'").strip('"')
            attr_names.append(attr)
        elif s.lower().startswith("@data"):
            data_idx = i + 1
            break

    if data_idx is None:
        raise ValueError(f"No se encontró @data en {arff_path}")

    # --- Lectura de filas de datos (solo formato denso, separado por coma) ---
    rows: List[List[str]] = []
    for s in lines[data_idx:]:
        if s.startswith("{"):
            # Por qué: el formato sparse requiere parsing especial -> fuera de alcance aquí
            raise NotImplementedError("ARFF sparse no soportado.")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != len(attr_names):
            # Nota: se descarta fila malformada para mantener consistencia
            continue
        rows.append(parts)

    return attr_names, rows


def _load_algorithm_runs_df(arff_path: str) -> pd.DataFrame:
    """Carga `algorithm_runs.arff` en un DataFrame normalizado.

    Renombra columnas comunes a un esquema estándar: instance_id, algorithm,
    runtime, runstatus.

    Args:
        arff_path (str): Ruta a algorithm_runs.arff.

    Returns:
        pd.DataFrame: DataFrame con columnas normalizadas y tipos adecuados.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    # --- Parseo mínimo del ARFF y construcción del DataFrame base ---
    names, rows = _parse_arff_minimal(arff_path)
    df = pd.DataFrame(rows, columns=names)

    # --- Normalización de nombres de columnas esperadas ---
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("instance_id", "instance"):
            rename[c] = "instance_id"
        elif cl in ("algorithm", "solver"):
            rename[c] = "algorithm"
        elif cl in ("runtime", "run_time", "runtime_secs", "rtime"):
            rename[c] = "runtime"
        elif cl in ("runstatus", "status", "run_status"):
            rename[c] = "runstatus"
    df = df.rename(columns=rename)

    need = {"instance_id", "algorithm", "runtime", "runstatus"}
    if not need.issubset(df.columns):
        raise ValueError(f"Faltan columnas {need} en {arff_path}")

    # --- Tipificación/limpieza básica ---
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["runstatus"] = df["runstatus"].astype(str).str.strip()
    return df


def _try_read_timeout_from_description(desc_path: str) -> Optional[float]:
    """Intenta extraer el límite de tiempo del archivo `description.txt`.

    Busca patrones comunes como 'cutoff time', 'cpu limit', 'time limit'.

    Args:
        desc_path (str): Ruta a description.txt.

    Returns:
        Optional[float]: Límite de tiempo detectado en segundos, o None si no se encuentra.
    """
    if not os.path.exists(desc_path):
        return None
    txt = _read_text(desc_path)

    # --- Patrones posibles para capturar números decimales/enteros ---
    for pat in (
        r"cutoff[_\s]*time\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
        r"cpu[_\s]*limit\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
        r"time[_\s]*limit\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
    ):
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                # Nota: si el parseo falla, continúa con siguiente patrón
                pass
    return None


def _build_instance_path_map(instances_dir: Optional[str]) -> Dict[str, str]:
    """Crea un mapeo 'nombre_de_archivo' → 'ruta_absoluta' recorriendo un directorio.

    Args:
        instances_dir (Optional[str]): Directorio raíz donde buscar instancias crudas.

    Returns:
        Dict[str, str]: Mapa por nombre de archivo (clave) a ruta absoluta (valor).
    """
    mapping: Dict[str, str] = {}

    # --- Si no hay carpeta válida, retornar vacío ---
    if not instances_dir or not os.path.isdir(instances_dir):
        return mapping

    # --- Recorrido recursivo del árbol de directorios ---
    for root, _, files in os.walk(instances_dir):
        for fn in files:
            mapping[fn] = os.path.join(root, fn)

    return mapping


def _load_instance_map_csv(instance_map_csv: Optional[str]) -> Dict[str, str]:
    """Carga un CSV con mapeo explícito 'instance_id' → 'file_path'.

    Args:
        instance_map_csv (Optional[str]): Ruta al CSV de mapeo.

    Returns:
        Dict[str, str]: Diccionario instance_id → file_path.

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    if not instance_map_csv or not os.path.exists(instance_map_csv):
        return {}
    df = pd.read_csv(instance_map_csv)

    need = {"instance_id", "file_path"}
    if not need.issubset(df.columns):
        raise ValueError("instance_map_csv debe tener columnas: instance_id,file_path")

    return dict(zip(df["instance_id"].astype(str), df["file_path"].astype(str)))


def _resolve_raw_text_path(
    instance_id: str,
    map_by_filename: Dict[str, str],
    map_by_id: Dict[str, str],
) -> Optional[str]:
    """Resuelve la ruta cruda de una instancia por id o por nombre de archivo.

    Prioridad:
      1) Mapa por id (instance_id → file_path).
      2) Mapa por nombre de archivo (filename → ruta).

    Args:
        instance_id (str): Identificador de la instancia (o nombre de archivo).
        map_by_filename (Dict[str, str]): Mapa filename → ruta.
        map_by_id (Dict[str, str]): Mapa instance_id → ruta.

    Returns:
        Optional[str]: Ruta encontrada o None si no se pudo resolver.
    """
    # --- Buscar por id explícito ---
    if instance_id in map_by_id and os.path.exists(map_by_id[instance_id]):
        return map_by_id[instance_id]

    # --- Buscar por coincidencia exacta de nombre de archivo ---
    if instance_id in map_by_filename and os.path.exists(map_by_filename[instance_id]):
        return map_by_filename[instance_id]

    return None


def _build_pivot_runtime_table(
    runs_df: pd.DataFrame, timeout_s: float
) -> tuple[pd.DataFrame, list[str]]:
    """Construye una tabla pivote con runtimes y estados por instancia/solver.

    - Rellena runtimes faltantes con `timeout_s`.
    - Normaliza estados (runstatus) y genera columnas *_Status.

    Args:
        runs_df (pandas.DataFrame): DataFrame normalizado de ejecuciones.
        timeout_s (float): Límite de tiempo a usar para faltantes.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            DataFrame pivoteado + lista de columnas *_Runtime_s (ordenadas).

    Decisión clave:
        Se toma el menor runtime por (instancia, solver) en caso de duplicados;
        esto es conservador y favorece la mejor corrida observada.
    """

    def norm_status(s: str) -> str:
        # Nota: algunos estados llegan como 'TIMEOUT: ...' → nos quedamos con la etiqueta
        return (s or "").split(":")[0].strip()

    # --- Copia defensiva y normalizaciones básicas ---
    runs = runs_df.copy()
    runs["algorithm"] = runs["algorithm"].astype(str)
    runs["runstatus_norm"] = runs["runstatus"].map(norm_status)

    # --- Ordenar por instancia/solver/runtime y quedarnos con la primera (mejor) ---
    runs = runs.sort_values(["instance_id", "algorithm", "runtime"]).drop_duplicates(
        ["instance_id", "algorithm"], keep="first"
    )

    # --- Pivotes separados para runtime y runstatus ---
    pt_rt = runs.pivot(
        index="instance_id", columns="algorithm", values="runtime"
    ).fillna(timeout_s)
    pt_st = runs.pivot(
        index="instance_id", columns="algorithm", values="runstatus_norm"
    ).fillna("TIMEOUT")

    # --- Renombrar columnas a *_Runtime_s y *_Status para consistencia posterior ---
    runtime_cols, new_rt, new_st = [], {}, {}
    for alg in pt_rt.columns:
        rc = f"{alg}_Runtime_s"
        sc = f"{alg}_Status"
        runtime_cols.append(rc)
        new_rt[alg] = rc
        new_st[alg] = sc

    pt_rt = pt_rt.rename(columns=new_rt)
    pt_st = pt_st.rename(columns=new_st)

    # --- Unir ambos pivotes en un solo DataFrame (por índice) ---
    merged = pt_rt.merge(
        pt_st, left_index=True, right_index=True, how="left"
    ).reset_index()
    return merged, runtime_cols


def _compute_winner_key(
    row: pd.Series, runtime_cols: List[str], timeout_s: float
) -> str:
    """Determina el solver ganador (tiempo mínimo < timeout) para una fila pivote.

    Args:
        row (pd.Series): Fila del pivote con columnas *_Runtime_s.
        runtime_cols (List[str]): Lista de columnas de runtime por solver.
        timeout_s (float): Límite de tiempo en segundos.

    Returns:
        str: Nombre de solver ganador (sin sufijo), o "NONE" si ninguno resuelve.
    """
    # --- Filtrar solvers que resolvieron (runtime < timeout) ---
    solved = [(c, float(row[c])) for c in runtime_cols if float(row[c]) < timeout_s]
    if not solved:
        return "NONE"

    # --- Elegir el de menor tiempo y remover sufijo estándar ---
    winner_col, _ = min(solved, key=lambda kv: kv[1])
    return winner_col.replace("_Runtime_s", "")


def prepare_data_and_ground_truth_aslib(
    scenario_dir: str,
    out_csv: str,
    instances_dir: Optional[str] = None,
    instance_map_csv: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    """Genera un CSV de ground truth a partir de un escenario ASlib.

    Pasos:
      1) Carga `algorithm_runs.arff` y normaliza columnas.
      2) Lee `description.txt` para inferir timeout (si no se pasa por parámetro).
      3) Pivotea runtimes/estados por instancia×solver y computa `Winner_Key`.
      4) Resuelve rutas crudas de instancias (por id o por nombre de archivo).
      5) Escribe el CSV final con columnas base, *_Runtime_s y *_Status.

    Args:
        scenario_dir (str): Carpeta del escenario ASlib (contiene el ARFF y description.txt).
        out_csv (str): Ruta de salida del CSV a generar.
        instances_dir (Optional[str]): Carpeta con instancias crudas (para mapear por filename).
        instance_map_csv (Optional[str]): CSV con columnas instance_id,file_path para mapear por id.
        timeout_s (Optional[float]): Límite de tiempo a usar; si None, se intenta leer de description.txt
            y si no está, se usa DEFAULT_TIMEOUT_S.

    Returns:
        str: Ruta absoluta o relativa del CSV generado.

    Raises:
        FileNotFoundError: Si no existe `algorithm_runs.arff` en `scenario_dir`.

    Decisión clave:
        Se prioriza `timeout_s` explícito > description.txt > DEFAULT_TIMEOUT_S para
        garantizar un valor consistente incluso si falta metadata.
    """
    # --- Asegurar carpeta de salida del CSV ---
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # --- Validar existencia de algorithm_runs.arff ---
    arff_path = os.path.join(scenario_dir, "algorithm_runs.arff")
    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"No se encontró algorithm_runs.arff en {scenario_dir}")

    print("=== INICIO (ASlib → CSV) ===")
    print(f"Escenario: {os.path.abspath(scenario_dir)}")
    print(f"Archivo ARFF: {arff_path}")
    print("[1/3] Cargando ejecuciones de solvers ...")

    # --- Cargar y normalizar runs ---
    runs_df = _load_algorithm_runs_df(arff_path)

    # --- Inferir límite de tiempo (si aplica) ---
    desc_path = os.path.join(scenario_dir, "description.txt")
    tl_desc = _try_read_timeout_from_description(desc_path)
    timeout_used = float(
        timeout_s
        if timeout_s is not None
        else (tl_desc if tl_desc else DEFAULT_TIMEOUT_S)
    )
    print(f"[2/3] Límite de tiempo usado: {timeout_used:.0f}s")

    # --- Pivote de runtimes/estados y lista de columnas de runtime ---
    pivot_df, runtime_cols = _build_pivot_runtime_table(runs_df, timeout_used)
    print(
        f"[2/3] Pivot listo: {pivot_df.shape[0]} instancias × {len(runtime_cols)} solvers"
    )

    # --- Resolver rutas crudas por instancia ---
    print("[3/3] Resolviendo rutas crudas de instancias ...")
    map_by_filename = _build_instance_path_map(instances_dir)
    map_by_id = _load_instance_map_csv(instance_map_csv)

    raw_paths = []
    for _, r in pivot_df.iterrows():
        iid = str(r["instance_id"])
        p = _resolve_raw_text_path(iid, map_by_filename, map_by_id)
        # Nota: si no se encuentra o no existe físicamente, se deja cadena vacía
        rp = p if p and os.path.exists(str(p)) else ""
        rp = os.path.abspath(rp) if rp else ""
        raw_paths.append(rp)

    # --- Enriquecimiento de columnas de salida ---
    pivot_df["Raw_Text_Path"] = raw_paths
    pivot_df["Time_Limit_s"] = timeout_used
    pivot_df["Winner_Key"] = [
        _compute_winner_key(r, runtime_cols, timeout_used)
        for _, r in pivot_df.iterrows()
    ]
    pivot_df["Instance_Id"] = pivot_df["instance_id"]
    pivot_df["Instance_Name"] = pivot_df["instance_id"].apply(
        lambda s: os.path.splitext(os.path.basename(str(s)))[0]
    )

    # --- Orden de columnas finales: base + métricas ---
    base = [
        "Instance_Id",
        "Instance_Name",
        "Raw_Text_Path",
        "Time_Limit_s",
        "Winner_Key",
    ]
    status_cols = [c for c in pivot_df.columns if c.endswith("_Status")]
    cols = base + runtime_cols + status_cols

    # --- Escritura del CSV de salida ---
    print("Escribiendo CSV de salida ...")
    pivot_df[cols].to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Listo: Ground Truth (ASlib) guardado en: {out_csv}")
    print("=== FIN (ASlib → CSV) ===")

    return out_csv
