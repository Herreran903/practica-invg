"""
Este script genera un archivo CSV de "ground truth" y convierte datos de instancias
en características de imagen (.npy) a partir de un escenario ASlib.

Contexto:
Se utiliza en el procesamiento de datos para experimentos de aprendizaje automático
relacionados con problemas de satisfacibilidad (SAT) y otras áreas.

Dependencias externas:
- `data_preparer`: Módulo para preparar datos y generar el archivo CSV.
- `image_converter`: Módulo para convertir datos en imágenes.

Uso:
$ python main.py --scenario_dir <ruta_escenario> --instances_dir <ruta_instancias>
Ejemplo:
$ python main.py --scenario_dir ./scenario --instances_dir ./instances
"""

import argparse
import os
import sys

# Directorio y nombres de salida predeterminados
OUTPUT_DIR = "sat_cnn_data_gen"
OUT_NAME = "ground_truth_aslib.csv"
INSTANCE_MAP_CSV = None  # Mapa opcional de instancias
TIMEOUT_S = None  # Tiempo límite opcional para procesamiento

# Importación de módulos externos con manejo de errores
try:
    from data_preparer import prepare_data_and_ground_truth_aslib
    from image_converter import generate_all_images
except ImportError as e:
    print("ERROR: Fallo en la importación modular. Revise nombres/rutas de módulos.")
    print(f"Detalles: {e}")
    sys.exit(1)


def main():
    """
    Punto de entrada principal del script. Procesa un escenario ASlib y genera un
    archivo CSV y características de imagen.

    Args:
        --scenario_dir (str): Ruta al directorio del escenario ASlib.
        --instances_dir (str): Ruta al directorio con archivos crudos CNF/XCSP/DZN.

    Raises:
        SystemExit: Si ocurre un error en la importación de módulos o en los pasos
        de procesamiento.
    """
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Runner ASlib → CSV + Imágenes")
    parser.add_argument(
        "--scenario_dir",
        required=True,
        help="Carpeta del escenario ASlib (debe contener algorithm_runs.arff).",
    )
    parser.add_argument(
        "--instances_dir",
        required=True,
        help="Carpeta con archivos crudos CNF/XCSP/DZN para Raw_Text_Path y conversión a imágenes.",
    )
    args = parser.parse_args()

    # Crear el directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.abspath(os.path.join(OUTPUT_DIR, OUT_NAME))

    # Información inicial
    print("=== INICIO (ASLIB) ===")
    print(f"Escenario:     {os.path.abspath(args.scenario_dir)}")
    print(f"Instances dir: {os.path.abspath(args.instances_dir)}")
    print(f"Salida CSV:    {out_csv}")

    # Paso 1: Generar el archivo CSV de "ground truth"
    print("\n[1/2] Generando Ground Truth ...")
    try:
        csv_file_path = prepare_data_and_ground_truth_aslib(
            scenario_dir=args.scenario_dir,
            out_csv=out_csv,
            instances_dir=args.instances_dir,
            instance_map_csv=INSTANCE_MAP_CSV,
            timeout_s=TIMEOUT_S,
        )
        print(f"[1/2] Listo: {csv_file_path}")
    except Exception as e:
        print(f"ERROR en el Paso 1: {e}")
        sys.exit(1)

    # Paso 2: Convertir datos en características de imagen
    print("\n[2/2] Convirtiendo a características de imagen (.npy) ...")
    try:
        generate_all_images(csv_file_path, args.instances_dir)
        print("[2/2] Listo: imágenes generadas")
    except Exception as e:
        print(f"ERROR en el Paso 2: {e}")
        sys.exit(1)

    # Información final
    print("\n=== FIN ===")
    print(f"Directorio de salida: {os.path.abspath(OUTPUT_DIR)}")
    print(f"CSV: {out_csv}")
    print("Archivos: .npy (X) y .csv (Y).")


if __name__ == "__main__":
    main()
