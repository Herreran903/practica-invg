# visualize_tensor.py
"""
Script para visualizar matrices JSP generadas por image_converter_convjssp.py.

Permite:
- Leer el CSV (ground truth)
- Seleccionar una instancia por nombre o índice
- Cargar el .npy asociado (matriz (JOBS, MACHINES))
- Imprimir la forma y algunos valores
- Dibujar un heatmap de duraciones estandarizadas

Uso:
    python visualize_tensor.py ruta/al/ground_truth.csv --index 0
    python visualize_tensor.py ruta/al/ground_truth.csv --name FT06
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_matrix_from_csv(csv_path: str, index: int = None, name: str = None) -> np.ndarray:
    """
    Carga una matriz 2D a partir del CSV usando un índice de fila o un nombre de instancia.

    Args:
        csv_path (str): Ruta al CSV con la columna 'Image_Npy_Path'.
        index (int, opcional): Índice de fila a cargar.
        name (str, opcional): Nombre de instancia (columna 'Instance_Name').

    Returns:
        np.ndarray: Matriz 2D (JOBS, MACHINES).
    """
    df = pd.read_csv(csv_path)

    if index is not None and name is not None:
        raise ValueError("Usa solo --index o solo --name, no ambos.")

    if index is not None:
        if index < 0 or index >= len(df):
            raise IndexError(f"Índice fuera de rango. CSV tiene {len(df)} filas.")
        row = df.iloc[index]
    elif name is not None:
        matches = df[df["Instance_Name"] == name]
        if matches.empty:
            raise ValueError(f"No se encontró ninguna instancia con nombre '{name}'.")
        row = matches.iloc[0]
    else:
        # Por defecto: primera fila
        row = df.iloc[0]

    npy_path = row["Image_Npy_Path"]
    if not isinstance(npy_path, str) or not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"No se encontró el archivo .npy para la instancia seleccionada: {npy_path}"
        )

    print(f"Instancia seleccionada: {row['Instance_Name']}")
    print(f"Ruta .npy: {npy_path}")
    matrix = np.load(npy_path)
    return matrix


def visualize_matrix(matrix: np.ndarray, show_values: bool = False) -> None:
    """
    Visualiza una matriz 2D (JOBS, MACHINES) mostrando un heatmap de duraciones.

    Args:
        matrix (np.ndarray): Matriz 2D (JOBS, MACHINES) de duraciones estandarizadas.
        show_values (bool): Si es True, imprime la matriz en consola.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"Se esperaba una matriz 2D de forma (JOBS, MACHINES), "
            f"pero se obtuvo {matrix.shape}."
        )

    jobs, machines = matrix.shape
    print(f"Shape de la matriz: {matrix.shape}")
    print(f"JOBS = {jobs}, MACHINES = {machines}")

    if show_values:
        print("\nMatriz de duraciones (estandarizadas):")
        print(matrix)

    plt.figure(figsize=(6, 5))
    plt.title("Duraciones estandarizadas (estilo CONVJSSP)")
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.xlabel("Operación (posición en el job)")
    plt.ylabel("Job")
    plt.colorbar(im, label="Tiempo de procesamiento (z-score)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualizador de matrices JSP generadas estilo CONVJSSP"
    )
    parser.add_argument(
        "csv",
        type=str,
        help="Ruta al CSV (ground_truth) con la columna 'Image_Npy_Path'.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Índice de fila a visualizar (0-based).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nombre de instancia (columna 'Instance_Name') a visualizar.",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Imprimir la matriz de duraciones en consola.",
    )

    args = parser.parse_args()

    matrix = load_matrix_from_csv(args.csv, index=args.index, name=args.name)
    visualize_matrix(matrix, show_values=args.show_values)


if __name__ == "__main__":
    main()
