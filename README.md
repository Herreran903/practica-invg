# Guía de ejecución rápida — comandos para SAT, JSP y visualizador

Objetivo: que puedas invocar cada script (generación y entrenamiento para SAT/JSP, y el visualizador) con los comandos correctos desde la raíz del repo.

Requisitos previos
- Python 3.9+ instalado.
- Instala dependencias:
  - pip install -r [requirements.txt](requirements.txt)
- Para JSP (solo generación): MiniZinc CLI en PATH y al menos un solver CP (gecode/chuffed).
  - Verifica: minizinc --version

Notas importantes
- Ejecuta SIEMPRE los comandos desde la raíz del repositorio para que las rutas relativas funcionen.
- Si moviste carpetas, vuelve a generar el CSV porque contiene rutas a .npy.

1) Generar datos para SAT (CSV + .npy)
Script: [generation_sat/main.py](generation_sat/main.py)

1.1. Si usarás el escenario ASlib provisto, primero extrae las instancias (ejemplo: sc2012-application)
tar -xf generation_sat/instances/sc2012-application.tar -C generation_sat/instances/

Esto creará la carpeta:
- [generation_sat/instances/sc2012-application](generation_sat/instances/sc2012-application)

1.2. Ejecuta la generación (ground truth + conversión a imágenes .npy)
python generation_sat/main.py --scenario_dir generation_sat/aslib/sc2012-application --instances_dir generation_sat/instances/sc2012-application

Salida esperada
- Carpeta: [sat_cnn_data_gen/](sat_cnn_data_gen)
- CSV: [sat_cnn_data_gen/ground_truth_aslib.csv](sat_cnn_data_gen/ground_truth_aslib.csv)
- Imágenes .npy referenciadas desde la columna Image_Npy_Path en el CSV.

2) Generar datos para JSP (CSV + .npy)
Script: [generation_jsp/main.py](generation_jsp/main.py)
Modelos MiniZinc: [generation_jsp/model.mzn](generation_jsp/model.mzn), [generation_jsp/model_linear.mzn](generation_jsp/model_linear.mzn)

2.1. Modo académico (instancias fijas JSPLIB)
python generation_jsp/main.py --mode academic

Salida: carpeta [jsp_cnn_data_acad/](jsp_cnn_data_acad) con CSV (ground_truth_jsp_academic.csv) y .npy.

2.2. Modo generado (instancias aleatorias balanceadas + varios solvers/tiempos)
python generation_jsp/main.py --mode generated

Salida: carpeta [jsp_cnn_data_gen/](jsp_cnn_data_gen) con CSV (ground_truth_jsp_generated_dataset.csv) y .npy.

Comprobación MiniZinc (si falla la generación JSP)
- Verifica MiniZinc y solvers:
  - minizinc --version
  - minizinc --solvers
- Asegúrate de ejecutar desde la raíz del repo para que encuentre los .mzn.

3) Entrenamiento SAT (CNN)
Script: [training_sat/train.py](training_sat/train.py)
Entrada: el CSV generado en el paso 1 (con columna Image_Npy_Path válida).

3.1. Clasificación (mejor solver), 5 folds, 25 épocas
python training_sat/train.py --csv sat_cnn_data_gen/ground_truth_aslib.csv --task classification --epochs 25 --folds 5

3.2. Multietiqueta (solvers viables bajo límite), con métricas de resolved_rate/AST
python training_sat/train.py --csv sat_cnn_data_gen/ground_truth_aslib.csv --task multilabel --epochs 20 --folds 5 --time_limit 1800 --feat_time_col Feature_Extract_s

3.3. Filtrar a un subconjunto de solvers (ejemplo)
python training_sat/train.py --csv sat_cnn_data_gen/ground_truth_aslib.csv --task classification --solvers clasp1,glucose2,lingeling --epochs 25 --folds 5

3.4. Repeticiones de K-Fold (p. ej., 5x5)
python training_sat/train.py --csv sat_cnn_data_gen/ground_truth_aslib.csv --task classification --epochs 25 --folds 5 --repeats 5

Salida esperada
- Carpeta raíz de reportes: [training_reports_sat/](training_reports_sat)
- Subcarpeta con timestamp (ej. cnn_solver_selector_sat_YYYYMMDD_HHMMSS)
- Métricas por fold, resúmenes globales y gráficas.

4) Entrenamiento JSP (CNN)
Script: [training_jsp/train.py](training_jsp/train.py)
Entrada: uno de los CSV del paso 2 (académico o generado), con Image_Npy_Path válido.

4.1. Clasificación (mejor solver) con datos académicos
python training_jsp/train.py --csv jsp_cnn_data_acad/ground_truth_jsp_academic.csv --task classification --epochs 25 --folds 5

4.2. Multietiqueta con datos generados
python training_jsp/train.py --csv jsp_cnn_data_gen/ground_truth_jsp_generated_dataset.csv --task multilabel --epochs 20 --folds 5

4.3. Regresión (tiempos por solver), ejemplo con datos académicos
python training_jsp/train.py --csv jsp_cnn_data_acad/ground_truth_jsp_academic.csv --task regression --epochs 25 --folds 5

Salida esperada
- Carpeta raíz de reportes: [training_reports_jsp/](training_reports_jsp)
- Subcarpeta con timestamp (ej. cnn_solver_selector_jsp_YYYYMMDD_HHMMSS)
- Métricas por fold y resúmenes.

5) Visualizar un archivo .npy
Script: [visualizador.py](visualizador.py)

Comando
python visualizador.py ruta/al/archivo.npy

Ejemplo
python visualizador.py sat_cnn_data_gen/images/ejemplo.npy

6) Flujos completos (end-to-end)

6.1. SAT completo
tar -xf generation_sat/instances/sc2012-application.tar -C generation_sat/instances/
python generation_sat/main.py --scenario_dir generation_sat/aslib/sc2012-application --instances_dir generation_sat/instances/sc2012-application
python training_sat/train.py --csv sat_cnn_data_gen/ground_truth_aslib.csv --task classification --epochs 25 --folds 5 --repeats 2

6.2. JSP completo (modo académico)
python generation_jsp/main.py --mode academic
python training_jsp/train.py --csv jsp_cnn_data_acad/ground_truth_jsp_academic.csv --task classification --epochs 25 --folds 5

6.3. JSP completo (modo generado)
python generation_jsp/main.py --mode generated
python training_jsp/train.py --csv jsp_cnn_data_gen/ground_truth_jsp_generated_dataset.csv --task multilabel --epochs 20 --folds 5

Errores comunes y cómo resolver rápido
- “El CSV no tiene 'Image_Npy_Path'”: primero ejecuta la generación (SAT/JSP) para crear el CSV con rutas a .npy.
- “No encuentra .npy”: ejecuta desde la raíz del repo o vuelve a generar el CSV para actualizar rutas.
- Fallo MiniZinc en JSP: verifica minizinc --version y minizinc --solvers; instala/activa solvers (gecode/chuffed).
- Si cambiaste rutas de .mzn, ajusta las constantes de modelo en [generation_jsp/main.py](generation_jsp/main.py).

Archivos principales
- Generación SAT: [generation_sat/main.py](generation_sat/main.py)
- Generación JSP: [generation_jsp/main.py](generation_jsp/main.py), [generation_jsp/model.mzn](generation_jsp/model.mzn), [generation_jsp/model_linear.mzn](generation_jsp/model_linear.mzn)
- Entrenamiento SAT: [training_sat/train.py](training_sat/train.py)
- Entrenamiento JSP: [training_jsp/train.py](training_jsp/train.py)
- Visualización: [visualizador.py](visualizador.py)