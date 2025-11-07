"""
Este script convierte archivos de texto plano en imágenes 2D y genera archivos .npy
con las matrices correspondientes. Está diseñado para procesar múltiples archivos
de texto listados en un archivo CSV.

Contexto de uso:
- Útil para tareas de preprocesamiento en aprendizaje automático o análisis de datos,
  donde se requiere convertir datos textuales en representaciones visuales.

Dependencias externas:
- numpy, pandas, Pillow (PIL)

Uso:
1. Ejecutar desde un entorno Python:
   python image_converter.py
2. Asegurarse de que el archivo CSV tenga las columnas 'Instance_Name' y 'Raw_Text_Path'.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image

TARGET_IMAGE_SIZE = 128


def convert_raw_text_to_image_matrix(raw_text_path, target_size=TARGET_IMAGE_SIZE):
    """
    Convierte el contenido textual de un archivo en una imagen 2D.

    Args:
        raw_text_path (str): Ruta al archivo de texto plano.
        target_size (int): Tamaño deseado de la imagen cuadrada (por defecto 128x128).

    Returns:
        np.ndarray: Matriz 2D de tipo float32 con la imagen generada.

    Decisión clave:
        - Se utiliza el método LANCZOS para reescalar la imagen debido a su alta calidad
          en la reducción de tamaño.
    """
    # Leer el contenido del archivo como texto plano
    with open(raw_text_path, "r") as f:
        contenido = f.read()

    # Convertir cada carácter en su código ASCII
    vector_ascii = [ord(c) for c in contenido]
    N = len(vector_ascii)

    # Calcular el mayor cuadrado perfecto posible
    lado = int(np.sqrt(N))
    N_usable = lado * lado

    # Si no hay datos suficientes, devolver una matriz de ceros
    if N_usable == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Ajustar el vector al cuadrado perfecto y convertir a matriz 2D
    vector_cuadrado = vector_ascii[:N_usable]
    imagen_inicial = np.array(vector_cuadrado, dtype=np.uint8).reshape((lado, lado))

    # Reescalar la imagen al tamaño objetivo
    imagen_pil = Image.fromarray(imagen_inicial)
    imagen_escalada = imagen_pil.resize(
        (target_size, target_size), Image.Resampling.LANCZOS
    )

    # Convertir la imagen reescalada a un array numpy de tipo float32
    imagen_final = np.array(imagen_escalada, dtype=np.float32)

    # Normalizar la imagen (z-score) si la desviación estándar es mayor a 0
    std_dev = np.std(imagen_final)
    if std_dev > 0:
        imagen_normalizada = (imagen_final - np.mean(imagen_final)) / std_dev
    else:
        imagen_normalizada = imagen_final

    return imagen_normalizada


def generate_all_images(csv_path):
    """
    Genera imágenes 2D a partir de archivos de texto listados en un archivo CSV.

    Args:
        csv_path (str): Ruta al archivo CSV con las columnas 'Instance_Name' y 'Raw_Text_Path'.

    Efectos:
        - Guarda un archivo .npy por cada fila en un subdirectorio 'images'.
        - Actualiza el CSV con una nueva columna 'Image_Npy_Path'.

    Raises:
        FileNotFoundError: Si el archivo CSV o algún archivo de texto no existe.
    """
    print("=== INICIO (Conversión de texto a imagen) ===")
    print(f"CSV de entrada: {os.path.abspath(csv_path)}")

    # Leer el archivo CSV
    print("[1/2] Leyendo CSV ...")
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Filas detectadas: {total}")

    # Crear el directorio de salida para las imágenes
    output_dir = os.path.dirname(csv_path)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_paths = []

    # Procesar cada fila del CSV
    print("[2/2] Generando matrices e imágenes (.npy) ...")
    for index, row in df.iterrows():
        name = row["Instance_Name"]
        raw_path = row["Raw_Text_Path"]

        # Convertir el archivo de texto en una matriz de imagen
        image_matrix = convert_raw_text_to_image_matrix(raw_path)

        # Guardar la matriz como archivo .npy
        npy_path = os.path.join(images_dir, f"{name}_image.npy")
        np.save(npy_path, image_matrix)
        image_paths.append(npy_path)

        print(f"- {index + 1}/{total}: {name} -> {npy_path}")

    # Actualizar el CSV con las rutas de las imágenes generadas
    print("Escribiendo CSV actualizado ...")
    df["Image_Npy_Path"] = image_paths
    df.to_csv(csv_path, index=False)

    print("Listo: conversión a imagen completada.")
    print("=== FIN (Conversión de texto a imagen) ===")
