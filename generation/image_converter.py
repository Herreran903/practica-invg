import os

import numpy as np
import pandas as pd
from PIL import Image

TARGET_IMAGE_SIZE = 128


def convert_raw_text_to_image_matrix(raw_text_path, target_size=TARGET_IMAGE_SIZE):
    with open(raw_text_path, "r") as f:
        contenido = f.read()

    vector_ascii = [ord(c) for c in contenido]
    N = len(vector_ascii)

    lado = int(np.sqrt(N))
    N_usable = lado * lado

    if N_usable == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    vector_cuadrado = vector_ascii[:N_usable]
    imagen_inicial = np.array(vector_cuadrado, dtype=np.uint8).reshape((lado, lado))

    imagen_pil = Image.fromarray(imagen_inicial)
    imagen_escalada = imagen_pil.resize(
        (target_size, target_size), Image.Resampling.LANCZOS
    )

    imagen_final = np.array(imagen_escalada, dtype=np.float32)

    std_dev = np.std(imagen_final)
    if std_dev > 0:
        imagen_normalizada = (imagen_final - np.mean(imagen_final)) / std_dev
    else:
        imagen_normalizada = imagen_final

    return imagen_normalizada


def generate_all_images(csv_path):
    """Lee el CSV y genera la matriz de imagen para cada instancia."""
    df = pd.read_csv(csv_path)

    output_dir = os.path.dirname(csv_path)

    image_paths = []

    for index, row in df.iterrows():
        name = row["Instance_Name"]
        raw_path = row["Raw_Text_Path"]

        image_matrix = convert_raw_text_to_image_matrix(raw_path)

        npy_path = os.path.join(output_dir, f"{name}_image.npy")
        np.save(npy_path, image_matrix)
        image_paths.append(npy_path)

    df["Image_Npy_Path"] = image_paths
    df.to_csv(csv_path, index=False)

    print(f"✅ Image conversion complete. NPY paths added to CSV.")
