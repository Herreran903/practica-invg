import csv
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

DEFAULT_TIMEOUT_S = 5000.0

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _parse_arff_minimal(arff_path: str) -> Tuple[List[str], List[List[str]]]:
    lines = []
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            lines.append(s)
    attr_names: List[str] = []
    data_idx = None
    for i, s in enumerate(lines):
        if s.lower().startswith("@attribute"):
            m = re.match(r"@attribute\s+([^\s]+)\s+(.+)", s, re.IGNORECASE)
            if not m: 
                continue
            attr = m.group(1).strip("'").strip('"')
            attr_names.append(attr)
        elif s.lower().startswith("@data"):
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError(f"No se encontró @data en {arff_path}")
    rows: List[List[str]] = []
    for s in lines[data_idx:]:
        if s.startswith("{"):
            raise NotImplementedError("ARFF sparse no soportado.")
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != len(attr_names):
            continue
        rows.append(parts)
    return attr_names, rows

def _load_algorithm_runs_df(arff_path: str) -> pd.DataFrame:
    names, rows = _parse_arff_minimal(arff_path)
    df = pd.DataFrame(rows, columns=names)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("instance_id","instance"): rename[c]="instance_id"
        elif cl in ("algorithm","solver"):   rename[c]="algorithm"
        elif cl in ("runtime","run_time","runtime_secs","rtime"): rename[c]="runtime"
        elif cl in ("runstatus","status","run_status"): rename[c]="runstatus"
    df = df.rename(columns=rename)
    need = {"instance_id","algorithm","runtime","runstatus"}
    if not need.issubset(df.columns):
        raise ValueError(f"Faltan columnas {need} en {arff_path}")
    df["runtime"]   = pd.to_numeric(df["runtime"], errors="coerce")
    df["runstatus"] = df["runstatus"].astype(str).str.strip()
    return df

def _try_read_timeout_from_description(desc_path: str) -> Optional[float]:
    if not os.path.exists(desc_path):
        return None
    txt = _read_text(desc_path)
    for pat in (r"cutoff[_\s]*time\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
                r"cpu[_\s]*limit\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
                r"time[_\s]*limit\s*[:=]\s*([0-9]+(\.[0-9]+)?)"):
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            try: return float(m.group(1))
            except: pass
    return None

def _build_instance_path_map(instances_dir: Optional[str]) -> Dict[str,str]:
    mapping: Dict[str,str] = {}
    if not instances_dir or not os.path.isdir(instances_dir):
        return mapping
    for root,_,files in os.walk(instances_dir):
        for fn in files:
            mapping[fn] = os.path.join(root, fn)
    return mapping

def _load_instance_map_csv(instance_map_csv: Optional[str]) -> Dict[str,str]:
    if not instance_map_csv or not os.path.exists(instance_map_csv):
        return {}
    df = pd.read_csv(instance_map_csv)
    need = {"instance_id","file_path"}
    if not need.issubset(df.columns):
        raise ValueError("instance_map_csv debe tener columnas: instance_id,file_path")
    return dict(zip(df["instance_id"].astype(str), df["file_path"].astype(str)))

def _resolve_raw_text_path(instance_id: str,
                           map_by_filename: Dict[str,str],
                           map_by_id: Dict[str,str]) -> Optional[str]:
    if instance_id in map_by_id and os.path.exists(map_by_id[instance_id]):
        return map_by_id[instance_id]
    if instance_id in map_by_filename and os.path.exists(map_by_filename[instance_id]):
        return map_by_filename[instance_id]
    return None

def _build_pivot_runtime_table(runs_df: pd.DataFrame, timeout_s: float) -> tuple[pd.DataFrame, list[str]]:
    def norm_status(s: str) -> str:
        return (s or "").split(":")[0].strip()
    runs = runs_df.copy()
    runs["algorithm"] = runs["algorithm"].astype(str)
    runs["runstatus_norm"] = runs["runstatus"].map(norm_status)
    runs = runs.sort_values(["instance_id","algorithm","runtime"]).drop_duplicates(["instance_id","algorithm"], keep="first")
    pt_rt = runs.pivot(index="instance_id", columns="algorithm", values="runtime").fillna(timeout_s)
    pt_st = runs.pivot(index="instance_id", columns="algorithm", values="runstatus_norm").fillna("TIMEOUT")
    runtime_cols, new_rt, new_st = [], {}, {}
    for alg in pt_rt.columns:
        rc = f"{alg}_Runtime_s"; sc = f"{alg}_Status"
        runtime_cols.append(rc); new_rt[alg]=rc; new_st[alg]=sc
    pt_rt = pt_rt.rename(columns=new_rt)
    pt_st = pt_st.rename(columns=new_st)
    merged = pt_rt.merge(pt_st, left_index=True, right_index=True, how="left").reset_index()
    return merged, runtime_cols

def _compute_winner_key(row: pd.Series, runtime_cols: List[str], timeout_s: float) -> str:
    solved = [(c, float(row[c])) for c in runtime_cols if float(row[c]) < timeout_s]
    if not solved: return "NONE"
    winner_col,_ = min(solved, key=lambda kv: kv[1])
    return winner_col.replace("_Runtime_s","")

def prepare_data_and_ground_truth_aslib(
    scenario_dir: str,
    out_csv: str,
    instances_dir: Optional[str] = None,
    instance_map_csv: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    arff_path = os.path.join(scenario_dir, "algorithm_runs.arff")
    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"No se encontró algorithm_runs.arff en {scenario_dir}")
    print(f"🔹 Cargando runs: {arff_path}")
    runs_df = _load_algorithm_runs_df(arff_path)

    desc_path = os.path.join(scenario_dir, "description.txt")
    tl_desc = _try_read_timeout_from_description(desc_path)
    timeout_used = float(timeout_s if timeout_s is not None else (tl_desc if tl_desc else DEFAULT_TIMEOUT_S))
    print(f"🔹 Time limit usado: {timeout_used:.0f}s")

    pivot_df, runtime_cols = _build_pivot_runtime_table(runs_df, timeout_used)
    print(f"🔹 Pivot listo: {pivot_df.shape[0]} instancias × {len(runtime_cols)} solvers")

    map_by_filename = _build_instance_path_map(instances_dir)
    map_by_id = _load_instance_map_csv(instance_map_csv)

    raw_paths = []
    for _, r in pivot_df.iterrows():
        iid = str(r["instance_id"])
        p = _resolve_raw_text_path(iid, map_by_filename, map_by_id)
        raw_paths.append(p if p and os.path.exists(str(p)) else "")
    pivot_df["Raw_Text_Path"] = raw_paths
    pivot_df["Time_Limit_s"] = timeout_used
    pivot_df["Winner_Key"] = [ _compute_winner_key(r, runtime_cols, timeout_used) for _, r in pivot_df.iterrows() ]
    pivot_df["Instance_Id"] = pivot_df["instance_id"]
    pivot_df["Instance_Name"] = pivot_df["instance_id"].apply(lambda s: os.path.splitext(os.path.basename(str(s)))[0])

    base = ["Instance_Id","Instance_Name","Raw_Text_Path","Time_Limit_s","Winner_Key"]
    status_cols = [c for c in pivot_df.columns if c.endswith("_Status")]
    cols = base + runtime_cols + status_cols
    pivot_df[cols].to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✅ Ground Truth (ASlib) guardado en: {out_csv}")
    return out_csv
