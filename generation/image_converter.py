# ==============================================
# image_converter.py — De texto (.dzn) a imagen (.npy)
# ----------------------------------------------------
# Propósito:
#   - Tomar el archivo .dzn (texto) de cada instancia y convertirlo
#     en una matriz 2D (grayscale) de tamaño fijo, normalizada.
#   - Guardar la matriz como .npy y anotar la ruta en el CSV.
# Uso:
#   - Llamado por el runner principal tras generar el CSV.
#   - También se puede importar y usar generate_all_images(csv_path).
# ==============================================

import os

import numpy as np
import pandas as pd
from PIL import Image

# Tamaño destino del lado de la imagen cuadrada
TARGET_IMAGE_SIZE = 128


def convert_raw_text_to_image_matrix(raw_text_path, target_size=TARGET_IMAGE_SIZE):
    """
    Convierte el contenido textual de un archivo (p.ej., .dzn) a una imagen 2D.

    Pasos:
      1) Lee el archivo como texto plano.
      2) Convierte cada carácter en su código ASCII (0..255).
      3) Ajusta al mayor cuadrado perfecto posible (lado×lado).
      4) Reescala a target_size×target_size (LANCZOS) y convierte a float32.
      5) Normaliza (z-score) si la desviación estándar es > 0.

    Retorna:
      - np.ndarray de shape (target_size, target_size), dtype float32.
    """
    # 1) Lectura segura del archivo de texto
    with open(raw_text_path, "r") as f:
        contenido = f.read()

    # 2) Texto → lista de códigos ASCII
    vector_ascii = [ord(c) for c in contenido]
    N = len(vector_ascii)

    # 3) Tomar el mayor cuadrado perfecto utilizable
    lado = int(np.sqrt(N))
    N_usable = lado * lado

    if N_usable == 0:
        # Archivo vacío: retorna imagen en ceros para mantener contrato de salida.
        return np.zeros((target_size, target_size), dtype=np.float32)

    vector_cuadrado = vector_ascii[:N_usable]
    imagen_inicial = np.array(vector_cuadrado, dtype=np.uint8).reshape((lado, lado))

    # 4) Reescalado al tamaño destino con filtro de alta calidad
    imagen_pil = Image.fromarray(imagen_inicial)
    imagen_escalada = imagen_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # 5) A float32 y normalización por z-score
    imagen_final = np.array(imagen_escalada, dtype=np.float32)

    std_dev = np.std(imagen_final)
    if std_dev > 0:
        imagen_normalizada = (imagen_final - np.mean(imagen_final)) / std_dev
    else:
        imagen_normalizada = imagen_final

    return imagen_normalizada


def generate_all_images(csv_path):
    """
    Lee el CSV producido en el paso de ground truth y genera la imagen para cada fila.

    Entradas esperadas del CSV:
      - Columnas al menos: 'Instance_Name', 'Raw_Text_Path'.

    Efectos:
      - Guarda un archivo .npy por instancia en el mismo directorio del CSV.
      - Agrega/actualiza la columna 'Image_Npy_Path' en el CSV.
    """
    # Cargar el CSV
    df = pd.read_csv(csv_path)

    # Salidas se guardan junto al CSV
    output_dir = os.path.dirname(csv_path)

    image_paths = []

    # Recorre filas y convierte cada .dzn a imagen
    for index, row in df.iterrows():
        name = row["Instance_Name"]
        raw_path = row["Raw_Text_Path"]

        image_matrix = convert_raw_text_to_image_matrix(raw_path)

        npy_path = os.path.join(output_dir, f"{name}_image.npy")
        np.save(npy_path, image_matrix)
        image_paths.append(npy_path)

    # Persistir rutas y reescribir CSV
    df["Image_Npy_Path"] = image_paths
    df.to_csv(csv_path, index=False)

    print(f"✅ Image conversion complete. NPY paths added to CSV.")
