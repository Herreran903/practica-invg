# -*- coding: utf-8 -*-
"""
Convierte instancias JSP descritas en archivos .dzn en matrices 2D
"estilo CONVJSSP" y genera archivos .npy con dichas matrices.

Representación (estilo CONVJSSP):
- Cada instancia se representa como una matriz 2D de tamaño:
    (num_jobs, num_machines)

  donde cada celda contiene la DURACIÓN de la operación
  (tiempo de procesamiento), ya estandarizada:

    x_norm = (x - media) / desviación_estándar

Supone que los .dzn fueron generados por _save_instance_dzn, con formato:

    JOBS = <int>;
    MACHINES = <int>;

    PROC_TIME = array2d(SET_JOBS, SET_POS, [v1, v2, ...]);
    MACHINE_OF_OP = array2d(SET_JOBS, SET_POS, [m1, m2, ...]);

Uso:
    python image_converter_convjssp.py ruta/al/ground_truth.csv

Y el CSV debe tener al menos:
    - 'Instance_Name'
    - 'Raw_Text_Path'
"""

import os
import re
from typing import List

import numpy as np
import pandas as pd


def _parse_dzn_int(name: str, text: str) -> int:
    """
    Extrae una asignación entera tipo `NAME = 10;` desde el texto .dzn.
    """
    pattern = rf"{name}\s*=\s*([0-9]+)\s*;"
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"No se encontró la definición de {name} en el .dzn")
    return int(m.group(1))


def _parse_dzn_array2d(name: str, text: str) -> List[int]:
    """
    Extrae una lista de enteros desde una definición tipo:

        NAME = array2d(SET_JOBS, SET_POS, [v1, v2, ..., vk]);

    Devuelve la lista [v1, v2, ..., vk] como ints.
    """
    pattern = rf"{name}\s*=\s*array2d\([^,]+,[^,]+,\s*\[(.*?)\]\s*\)\s*;"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No se encontró la definición de {name} como array2d en el .dzn")

    inside = m.group(1)
    tokens = [t.strip() for t in inside.replace("\n", " ").split(",") if t.strip()]
    try:
        values = [int(tok) for tok in tokens]
    except ValueError as e:
        raise ValueError(f"No se pudieron convertir los valores de {name} a enteros") from e
    return values


def _dzn_to_matrix(dzn_path: str, standardize: bool = True) -> np.ndarray:
    """
    Convierte un archivo .dzn con JOBS, MACHINES y PROC_TIME
    en una matriz 2D de duraciones:

        (num_jobs, num_machines)

    Si `standardize=True`, aplica estandarización:
        x_norm = (x - media) / std
    """
    if not os.path.exists(dzn_path):
        raise FileNotFoundError(f"No se encontró el archivo .dzn: {dzn_path}")

    with open(dzn_path, "r") as f:
        text = f.read()

    num_jobs = _parse_dzn_int("JOBS", text)
    num_machines = _parse_dzn_int("MACHINES", text)

    proc_flat = _parse_dzn_array2d("PROC_TIME", text)
    expected_len = num_jobs * num_machines
    if len(proc_flat) != expected_len:
        raise ValueError(
            f"Longitud de PROC_TIME inconsistente con JOBS*MACHINES. "
            f"PROC_TIME tiene {len(proc_flat)}, pero JOBS*MACHINES = {expected_len}."
        )

    # Matriz de duraciones (jobs x machines)
    proc_mat = np.array(proc_flat, dtype=np.float32).reshape((num_jobs, num_machines))

    if standardize:
        mean = proc_mat.mean()
        std = proc_mat.std()
        if std == 0.0:
            raise ValueError(
                f"Desviación estándar nula en PROC_TIME de {dzn_path}, "
                f"no se puede estandarizar."
            )
        proc_mat = (proc_mat - mean) / std

    return proc_mat


def generate_all_images(csv_path: str):
    """
    Genera matrices 2D (estilo CONVJSSP) a partir de instancias listadas en un CSV.

    Args:
        csv_path (str): Ruta al archivo CSV con las columnas:
                        - 'Instance_Name'
                        - 'Raw_Text_Path' (ruta al .dzn)

    Efectos:
        - Crea un subdirectorio 'images' (o lo reutiliza).
        - Guarda un archivo .npy por cada fila del CSV, con una matriz de forma:
              (num_jobs, num_machines)
        - Actualiza el CSV agregando la columna 'Image_Npy_Path' con la ruta al .npy.
    """
    print("=== INICIO (Conversión DZN → Matriz 2D estilo CONVJSSP) ===")
    print(f"CSV de entrada: {os.path.abspath(csv_path)}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en la ruta: {csv_path}")

    print("[1/2] Leyendo CSV ...")
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Filas detectadas: {total}")

    output_dir = os.path.dirname(csv_path)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    npy_paths: List[str] = []

    print("[2/2] Generando matrices (.npy) ...")
    for index, row in df.iterrows():
        name = row["Instance_Name"]
        dzn_path = row["Raw_Text_Path"]

        print(f"- {index + 1}/{total}: {name}")
        try:
            mat = _dzn_to_matrix(dzn_path, standardize=True)
        except Exception as e:
            print(f"    ERROR al procesar {name} desde {dzn_path}: {e}")
            raise

        npy_path = os.path.join(images_dir, f"{name}_matrix.npy")
        np.save(npy_path, mat)
        npy_paths.append(npy_path)

        print(f"    -> Matriz guardada en: {npy_path}  (shape={mat.shape})")

    print("Escribiendo CSV actualizado ...")
    df["Image_Npy_Path"] = npy_paths
    df.to_csv(csv_path, index=False)

    print("Listo: conversión a matriz estilo CONVJSSP completada.")
    print("=== FIN (DZN → Matriz 2D) ===")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso: python image_converter_convjssp.py ruta/al/ground_truth.csv")
        sys.exit(1)

    generate_all_matrices(sys.argv[1])
