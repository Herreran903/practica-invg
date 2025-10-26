import argparse
import os
import sys

try:
    from data_preparer import (
        prepare_data_and_ground_truth_minizinc as run_academic_mode,
    )
    from data_preparer_gen import (
        prepare_data_and_ground_truth_minizinc_gen as run_generated_mode,
    )
    from image_converter import generate_all_images
except ImportError as e:
    print(
        f"❌ ERROR: Fallo en la importación modular. Revise los nombres de los archivos."
    )
    print(f"Detalles del error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Runner para la generación del dataset JSP (Texto a Imagen).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["academic", "generated"],
        help="""Modo de generación de instancias:
'academic': Usa instancias JSPLIB fijas (FT06, FT10, etc.).
'generated': Genera instancias aleatorias balanceadas (N x M variable).""",
    )

    args = parser.parse_args()

    if args.mode == "academic":
        mode_name = "ACADÉMICO (JSPLIB)"
        execution_function = run_academic_mode
    elif args.mode == "generated":
        mode_name = "GENERADO (Aleatorio)"
        execution_function = run_generated_mode
    else:
        sys.exit(1)

    print(f"--- 🛠️ INICIANDO PIPELINE ({mode_name}) ---")

    print("1/2: Ejecutando Solvers y generando Ground Truth (Y)...")

    try:
        csv_file_path = execution_function()
        output_dir = os.path.dirname(csv_file_path)
        print(f"✅ 1/2: Ground Truth guardado en: {csv_file_path}")
    except Exception as e:
        print(f"❌ ERROR FATAL en el Paso 1 (Ejecución del Solver): {e}")
        sys.exit(1)

    print("\n2/2: Convirtiendo datos brutos (.dzn) a Features de Imagen (.npy) (X)...")

    try:
        generate_all_images(csv_file_path)
        print("✅ 2/2: Conversión a Imagen completada.")
    except Exception as e:
        print(f"❌ ERROR en el Paso 2 (Conversión a Imagen): {e}")
        sys.exit(1)

    print("\n--- ✨ PIPELINE COMPLETADO EXITOSAMENTE ---")
    print(f"Output del Dataset: {output_dir}")
    print("Archivos listos: .npy (Features X) y .csv (Ground Truth Y).")


if __name__ == "__main__":
    main()
