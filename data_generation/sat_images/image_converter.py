"""
Este script convierte archivos de texto en matrices de imágenes y guarda las matrices en formato `.npy`.

Contexto de uso:
Se utiliza para preprocesar datos en formato de texto y convertirlos en representaciones visuales (imágenes)
que pueden ser usadas en tareas de aprendizaje automático o análisis.

Dependencias externas:
- numpy
- pandas
- PIL (Pillow)

Uso:
1. Ejecutar desde un entorno Python, asegurándose de que el archivo CSV de entrada y las rutas de instancias sean válidas.
2. Ejemplo:
    python image_converter.py input.csv /path/to/instances_root
"""

import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image

# Tamaño objetivo de las imágenes generadas
TARGET_IMAGE_SIZE = 128

# Mapeo de prefijos para resolver rutas de instancias
PREFIX_MAP = {
    "SAT-Race-2010-CNF": "Application SAT+UNSAT/SAT Race 2010",
    "SATCompetition2007": "SATCompetition2007/industrial",
    "SATCompetition2009": "SATCompetition2009/application",
    "SATCompetition2011": "SATCompetition2011/application",
    "SC2012_Application_Debugged": "SC2012_Application_Debugged/satrace-unselected",
}


def _resolve_path(instances_root: str, instance_id: str):
    """
    Resuelve la ruta completa de un archivo dado su ID y la raíz de instancias.

    Args:
        instances_root (str): Directorio raíz donde buscar las instancias.
        instance_id (str): Identificador del archivo o ruta parcial.

    Returns:
        str | None: Ruta completa del archivo si se encuentra, de lo contrario None.
    """
    # Validación básica del ID de la instancia
    if not isinstance(instance_id, str) or not instance_id.strip():
        return None

    # Intento directo con la ruta proporcionada
    p1 = os.path.join(instances_root, instance_id)
    if os.path.exists(p1):
        return p1

    # Intento con prefijo mapeado
    parts = instance_id.split("/", 1)
    if len(parts) == 2 and parts[0] in PREFIX_MAP:
        p2 = os.path.join(instances_root, PREFIX_MAP[parts[0]], parts[1])
        if os.path.exists(p2):
            return p2

    # Búsqueda exhaustiva en el árbol de directorios
    base = os.path.basename(instance_id)
    for root, _, files in os.walk(instances_root):
        if base in files:
            return os.path.join(root, base)

    # Si no se encuentra, retornar None
    return None


def convert_raw_text_to_image_matrix(
    raw_text_path: str, target_size: int = TARGET_IMAGE_SIZE
):
    """
    Convierte un archivo de texto en una matriz de imagen normalizada.

    Args:
        raw_text_path (str): Ruta al archivo de texto.
        target_size (int): Tamaño objetivo de la imagen (ancho y alto).

    Returns:
        np.ndarray: Matriz de la imagen generada, normalizada.

    Raises:
        FileNotFoundError: Si el archivo de texto no existe.
    """
    # Leer el archivo como un buffer de bytes
    with open(raw_text_path, "rb") as f:
        data = f.read()

    # Convertir los bytes en un vector de enteros sin signo
    vector = np.frombuffer(data, dtype=np.uint8)
    N = vector.size
    if N == 0:
        # Retornar una matriz vacía si el archivo está vacío
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Calcular el lado máximo de un cuadrado usable
    lado = int(np.floor(np.sqrt(N)))
    usable = lado * lado
    if usable <= 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Crear una matriz cuadrada a partir del vector
    img0 = vector[:usable].reshape(lado, lado).astype(np.uint8)

    # Redimensionar la imagen al tamaño objetivo
    img = Image.fromarray(img0, mode="L").resize(
        (target_size, target_size), Image.Resampling.LANCZOS
    )

    # Convertir la imagen en un array numpy y normalizar
    arr = np.asarray(img, dtype=np.float32)
    std = float(arr.std())
    if std > 0:
        arr = (arr - float(arr.mean())) / std
    return arr


def generate_all_images(csv_path: str, instances_root: str):
    """
    Genera matrices de imágenes a partir de archivos de texto y actualiza un CSV con las rutas generadas.

    Args:
        csv_path (str): Ruta al archivo CSV de entrada.
        instances_root (str): Directorio raíz donde buscar los archivos de texto.

    Returns:
        None
    """
    print("=== INICIO (Conversión de texto a imagen) ===")
    print(f"CSV de entrada: {os.path.abspath(csv_path)}")
    print(f"Raíz de instancias: {os.path.abspath(instances_root)}")
    print("[1/3] Leyendo CSV ...")

    # Leer el archivo CSV
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Filas detectadas: {total}")

    # Crear directorio de salida para las imágenes
    out_dir = os.path.abspath(os.path.dirname(csv_path) or ".")
    img_dir = os.path.abspath(os.path.join(out_dir, "images"))
    os.makedirs(img_dir, exist_ok=True)

    # Inicializar listas y contadores
    raw_paths = []
    image_paths = []
    ok, miss = 0, 0

    print("[2/3] Generando matrices e imágenes (.npy) ...")
    for idx, row in df.iterrows():
        # Resolver la ruta del archivo de texto
        raw = row.get("Raw_Text_Path")
        if not (isinstance(raw, str) and os.path.exists(raw)):
            raw = _resolve_path(instances_root, str(row.get("Instance_Id", "")))

        if raw and os.path.exists(raw):
            raw_paths.append(raw)
            try:
                # Convertir el archivo de texto en una matriz de imagen
                arr = convert_raw_text_to_image_matrix(raw, TARGET_IMAGE_SIZE)
                iid = str(row.get("Instance_Id", row.get("Instance_Name", "")))
                h = hashlib.md5(iid.encode("utf-8", errors="ignore")).hexdigest()[:12]
                base = os.path.splitext(os.path.basename(iid))[0]
                npy_path = os.path.abspath(os.path.join(img_dir, f"{base}__{h}.npy"))
                np.save(npy_path, arr.astype(np.float32))
                image_paths.append(npy_path)
                ok += 1
            except Exception as e:
                # Manejo de errores durante la conversión
                print(f"Aviso: fallo al convertir {raw}. Detalles: {e}")
                image_paths.append("")
                miss += 1
        else:
            # Si no se encuentra el archivo, registrar como faltante
            raw_paths.append("" if raw is None else raw)
            image_paths.append("")
            miss += 1

    print("[3/3] Escribiendo CSV actualizado ...")
    # Actualizar el CSV con las rutas generadas
    df["Raw_Text_Path"] = raw_paths
    df["Image_Npy_Path"] = image_paths
    df.to_csv(csv_path, index=False)

    print(f"Listo: conversión a imagen completada. {ok} correctas, {miss} sin imagen.")
    print("=== FIN (Conversión de texto a imagen) ===")
