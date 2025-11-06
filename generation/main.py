# ==============================================
# main.py — Runner del pipeline de generación JSP
# ----------------------------------------------
# Qué hace:
#   - Orquesta dos pasos para construir el dataset texto→imagen:
#     1) Ejecuta solvers sobre instancias JSP y guarda "ground truth" (CSV).
#     2) Convierte cada .dzn (texto) a una imagen 2D normalizada y guarda .npy.
# Cómo usar:
#   python -m generation.main --mode {academic|generated}
# Requisitos:
#   - MiniZinc instalado y accesible en PATH.
#   - job-shop-lib instalado para construir/cargar instancias.
#   - NumPy, Pandas y Pillow para la conversión a imagen.
# Salidas:
#   - Un directorio con:
#       • ground_truth_*.csv (Y)
#       • *_image.npy por instancia (X)
#       • archivos .dzn intermedios
# ==============================================

import argparse  # Manejo de argumentos CLI
import os        # Manejo de rutas y directorios
import sys       # Salida controlada y terminación del programa

# Importa las funciones de los módulos internos.
# - data_preparer: modo "academic" con instancias JSPLIB clásicas.
# - data_preparer_gen: modo "generated" con instancias aleatorias balanceadas.
# - image_converter: convierte el archivo .dzn (texto) a matriz de imagen y guarda .npy.
try:
    from data_preparer import (
        prepare_data_and_ground_truth_minizinc as run_academic_mode,
    )
    from data_preparer_gen import (
        prepare_data_and_ground_truth_minizinc_gen as run_generated_mode,
    )
    from image_converter import generate_all_images
except ImportError as e:
    # Error típico si se cambió el nombre de los archivos o no están en el mismo paquete.
    print("❌ ERROR: Fallo en la importación modular. Revise los nombres de los archivos.")
    print(f"Detalles del error: {e}")
    sys.exit(1)


def main():
    """
    Punto de entrada del runner.

    Flujo completo:
      1) Según --mode:
         - 'academic': carga instancias clásicas JSPLIB (FT06, FT10, etc.)
         - 'generated': genera instancias aleatorias balanceadas de N×M
         Ejecuta MiniZinc con distintos solvers/estrategias y crea un CSV con:
           Instance_Name, Raw_Text_Path, dimensiones, métricas por solver, etc.
      2) Lee el CSV del paso 1 y, por cada fila, convierte el archivo 'Raw_Text_Path'
         (un .dzn) en una matriz de imagen 128×128 normalizada. Guarda .npy
         y agrega la ruta 'Image_Npy_Path' al mismo CSV.
    """
    # 1) Definición de CLI con descripción legible (permite saltos de línea en help).
    parser = argparse.ArgumentParser(
        description="Runner para la generación del dataset JSP (Texto a Imagen).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Argumento obligatorio para elegir el modo de construcción del dataset.
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["academic", "generated"],
        help=(
            "Modo de generación de instancias:\n"
            " 'academic': Usa instancias JSPLIB fijas (FT06, FT10, etc.).\n"
            " 'generated': Genera instancias aleatorias balanceadas (N x M variable)."
        ),
    )

    # 2) Parseo de argumentos desde la línea de comandos.
    args = parser.parse_args()

    # 3) Selección del modo → mapeo a la función ejecutora adecuada.
    if args.mode == "academic":
        mode_name = "ACADÉMICO (JSPLIB)"
        execution_function = run_academic_mode
    elif args.mode == "generated":
        mode_name = "GENERADO (Aleatorio)"
        execution_function = run_generated_mode
    else:
        # Por seguridad: argparse ya controla 'choices', pero mantenemos salida defensiva.
        sys.exit(1)

    print(f"--- 🛠️ INICIANDO PIPELINE ({mode_name}) ---")

    # Paso 1: Ejecutar solvers y construir Ground Truth (CSV)
    print("1/2: Ejecutando Solvers y generando Ground Truth (Y)...")

    try:
        # execution_function devuelve la ruta al CSV generado.
        csv_file_path = execution_function()
        output_dir = os.path.dirname(csv_file_path)
        print(f"✅ 1/2: Ground Truth guardado en: {csv_file_path}")
    except Exception as e:
        # Errores típicos: MiniZinc no instalado, modelo no encontrado, etc.
        print(f"❌ ERROR FATAL en el Paso 1 (Ejecución del Solver): {e}")
        sys.exit(1)

    # Paso 2: Convertir los .dzn en matrices de imagen y guardarlas
    print("\n2/2: Convirtiendo datos brutos (.dzn) a Features de Imagen (.npy) (X)...")

    try:
        # Agrega columna 'Image_Npy_Path' al mismo CSV.
        generate_all_images(csv_file_path)
        print("✅ 2/2: Conversión a Imagen completada.")
    except Exception as e:
        print(f"❌ ERROR en el Paso 2 (Conversión a Imagen): {e}")
        sys.exit(1)

    # Resumen final y ubicación de archivos.
    print("\n--- ✨ PIPELINE COMPLETADO EXITOSAMENTE ---")
    print(f"Output del Dataset: {output_dir}")
    print("Archivos listos: .npy (Features X) y .csv (Ground Truth Y).")


if __name__ == "__main__":
    main()
