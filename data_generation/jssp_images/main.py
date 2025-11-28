"""
Este script genera datasets para problemas de programación de tareas (JSP) en dos modos:
académico (usando instancias fijas de JSPLIB) y generado (instancias aleatorias balanceadas).

Contexto:
Se utiliza para crear datos de entrenamiento y prueba para modelos de aprendizaje automático
que trabajan con problemas de JSP.

Dependencias externas:
- `data_preparer` y `data_preparer_gen`: módulos para preparar datos y generar instancias.
- `image_converter`: módulo para convertir datos a características de imagen.

Uso:
1. Modo académico:
    python main.py --mode academic
2. Modo generado:
    python main.py --mode generated
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

# Constantes globales para configuración de modelos, solvers y generación de datos
CP_MODEL_MZN_PATH = "generation/model.mzn"
MIP_MODEL_MZN_PATH: Optional[str] = "generation/model_linear.mzn"

# Lista de solvers disponibles con sus configuraciones
SOLVER_CANDIDATES = [
    # Solver basado en programación por restricciones (CP)
    (
        "gecode",
        "GECODE_FF",
        "cp",
        {"strategy": "first_fail", "supports_seed": True, "inject_search": True},
    ),
    (
        "chuffed",
        "CHUFFED_IO",
        "cp",
        {"strategy": "input_order", "supports_seed": True, "inject_search": True},
    ),
    (
        "cp-sat",
        "CPSAT_FF",
        "cp",
        {"strategy": "first_fail", "supports_seed": True, "inject_search": True},
    ),
    # Solvers basados en programación lineal entera (MIP)
    (
        "coin-bc",
        "CBC_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "scip",
        "SCIP_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "highs",
        "HIGHS_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "cplex",
        "CPLEX_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
    (
        "gurobi",
        "GUROBI_DEF",
        "mip",
        {"strategy": None, "supports_seed": False, "inject_search": False},
    ),
]

# Instancias académicas predefinidas
INSTANCE_NAMES = ["ft06", "ft10", "la01", "abz5"]

# Límites de tiempo en milisegundos para los solvers
TIME_LIMITS_MS: List[int] = [5000, 30000, 60000]
TIME_LIMIT_MS = 60000

# Semillas aleatorias para generación reproducible
RANDOM_SEEDS: List[int] = [1, 2, 3]

# Configuración de casos de generación (máquinas, trabajos, instancias)
GENERATION_CASES: List[Tuple[int, int, int]] = [
    (4, 4, 5),
    (6, 6, 5),
    (8, 8, 5),
    (10, 10, 5),
]

# Directorios y nombres de salida
OUTPUT_DIR_ACAD = "jsp_cnn_data_acad"
OUT_NAME_ACAD = "ground_truth_jsp_academic.csv"
OUTPUT_DIR_GEN = "jsp_cnn_data_gen"
OUT_NAME_GEN = "ground_truth_jsp_generated_dataset.csv"

# Importación de módulos externos con manejo de errores
try:
    from data_preparer import (
        prepare_data_and_ground_truth_minizinc as run_academic_mode,
    )
    from data_preparer_gen import (
        prepare_data_and_ground_truth_minizinc_gen as run_generated_mode,
    )
    from image_converter import generate_all_images
except ImportError as e:
    print("ERROR: Fallo en la importación modular. Revise los nombres de los archivos.")
    print(f"Detalles: {e}")
    sys.exit(1)


def main():
    """
    Punto de entrada principal del script. Ejecuta la generación de datasets JSP en modo académico o generado.

    Args:
         None (los argumentos se leen desde la línea de comandos).

    Raises:
         SystemExit: Si ocurre un error en la importación, generación de datos o conversión de imágenes.
    """
    # Configuración del parser de argumentos
    parser = argparse.ArgumentParser(
        description="Runner para la generación del dataset JSP (Texto a Imagen).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["academic", "generated"],
        help=(
            "Modo de generación de instancias:\n"
            "  academic  : usa instancias JSPLIB fijas.\n"
            "  generated : genera instancias aleatorias balanceadas."
        ),
    )
    args = parser.parse_args()

    # Selección del modo de generación
    if args.mode == "academic":
        mode_name = "ACADÉMICO (JSPLIB)"
        output_dir = OUTPUT_DIR_ACAD
        out_name = OUT_NAME_ACAD

        print(f"=== INICIO ({mode_name}) ===")
        print("[1/2] Ejecutando solvers y generando Ground Truth (Y) ...")
        try:
            # Crear directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)
            # Generar datos académicos
            csv_file_path = run_academic_mode(
                output_dir=OUTPUT_DIR_ACAD,
                out_name=OUT_NAME_ACAD,
                instance_names=INSTANCE_NAMES,
                model_mzn_path=CP_MODEL_MZN_PATH,
                time_limit_ms=TIME_LIMIT_MS,
                penalty_factor_k=10.0,
                solver_strategies=[
                    ("gecode", "default", "GECODE_DEFAULT"),
                    ("chuffed", "default", "CHUFFED_DEFAULT"),
                ],
            )
            print(f"[1/2] Listo: {csv_file_path}")
        except Exception as e:
            print(f"ERROR en el Paso 1: {e}")
            sys.exit(1)

    elif args.mode == "generated":
        mode_name = "GENERADO (Aleatorio)"
        output_dir = OUTPUT_DIR_GEN
        out_name = OUT_NAME_GEN

        print(f"=== INICIO ({mode_name}) ===")
        print("[1/2] Ejecutando solvers y generando Ground Truth (Y) ...")
        try:
            # Crear directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)
            # Generar datos aleatorios
            csv_file_path = run_generated_mode(
                output_dir=OUTPUT_DIR_GEN,
                out_name=OUT_NAME_GEN,
                cp_model_path=CP_MODEL_MZN_PATH,
                mip_model_path=MIP_MODEL_MZN_PATH,
                time_limits_ms=TIME_LIMITS_MS,
                solver_candidates=SOLVER_CANDIDATES,
                random_seeds=RANDOM_SEEDS,
                generation_cases=GENERATION_CASES,
            )
            print(f"[1/2] Listo: {csv_file_path}")
        except Exception as e:
            print(f"ERROR en el Paso 1: {e}")
            sys.exit(1)
    else:
        sys.exit(1)

    # Conversión de datos a características de imagen
    print("\n[2/2] Convirtiendo .dzn a características de imagen (.npy) ...")
    try:
        generate_all_images(csv_file_path)
        print("[2/2] Listo: conversión a imagen completada")
    except Exception as e:
        print(f"ERROR en el Paso 2: {e}")
        sys.exit(1)

    # Finalización del script
    print("\n=== FIN ===")
    print(f"Directorio de salida: {os.path.dirname(csv_file_path)}")
    print("Archivos: .npy (X) y .csv (Y).")


if __name__ == "__main__":
    main()
