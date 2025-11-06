import argparse
import os
import sys

try:
    from data_preparer import prepare_data_and_ground_truth_aslib
    from image_converter import generate_all_images
except ImportError as e:
    print("❌ ERROR: Fallo en la importación modular. Revise los nombres de los archivos.")
    print(f"Detalles del error: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Runner para la generación del dataset SAT/CSP desde ASlib (Texto→Imagen).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["aslib"],
        help="Modo de preparación de datos: 'aslib' (lee algorithm_runs.arff y genera CSV).",
    )
    parser.add_argument("--scenario_dir", required=True, type=str,
                        help="Carpeta del escenario ASlib (debe contener algorithm_runs.arff).")
    parser.add_argument("--out_csv", required=True, type=str,
                        help="Ruta del CSV de salida (se creará/overwrite).")
    parser.add_argument("--instances_dir", required=False, type=str, default=None,
                        help="Carpeta con archivos crudos CNF/XCSP (opcional; se puede usar --instance_map_csv).")
    parser.add_argument("--instance_map_csv", required=False, type=str, default=None,
                        help="CSV con columnas instance_id,file_path para mapear archivos crudos.")
    parser.add_argument("--timeout_s", required=False, type=float, default=None,
                        help="Time limit (s). Si no se pasa, intenta leerse de description.txt; si falla, usa 5000s.")

    args = parser.parse_args()

    if args.mode != "aslib":
        print("❌ Modo no soportado en esta versión. Use --mode aslib")
        sys.exit(1)

    print(f"--- 🛠️ INICIANDO PIPELINE (ASLIB) ---")

    print("1/2: Generando Ground Truth (runtimes por solver e info de instancia)...")
    try:
        csv_file_path = prepare_data_and_ground_truth_aslib(
            scenario_dir=args.scenario_dir,
            out_csv=args.out_csv,
            instances_dir=args.instances_dir,
            instance_map_csv=args.instance_map_csv,
            timeout_s=args.timeout_s,
        )
        output_dir = os.path.dirname(csv_file_path) or "."
        print(f"✅ 1/2: Ground Truth guardado en: {csv_file_path}")
    except Exception as e:
        print(f"❌ ERROR FATAL en el Paso 1 (ASlib → CSV): {e}")
        sys.exit(1)

    print("\n2/2: Convirtiendo archivos crudos (CNF/XCSP) a Features de Imagen (.npy) (X)...")
    try:
        generate_all_images(csv_file_path, args.instances_dir)
        print("✅ 2/2: Conversión a Imagen completada.")
    except Exception as e:
        print(f"❌ ERROR en el Paso 2 (Conversión a Imagen): {e}")
        sys.exit(1)

    print("\n--- ✨ PIPELINE COMPLETADO EXITOSAMENTE ---")
    print(f"Output del Dataset: {output_dir}")
    print("Archivos listos: .npy (Features X) y .csv (Ground Truth Y).")

if __name__ == "__main__":
    main()
