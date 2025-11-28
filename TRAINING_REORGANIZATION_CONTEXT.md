# Contexto de Reorganización de Training Scripts

## Objetivo General
Reorganizar los scripts de entrenamiento en `training/` siguiendo el mismo patrón profesional y modular usado para reorganizar `data_generation/` (que ya está completado en `src/data_generation/`).

## Trabajo Completado Hasta Ahora

### 1. Data Generation (YA COMPLETADO - Referencia)
Se reorganizaron exitosamente tres módulos bajo `src/data_generation/`:
- **jssp_images/**: Generación de imágenes en escala de grises desde JSSP
- **jssp_tensors/**: Generación de tensores/matrices desde JSSP  
- **sat_images/**: Generación de imágenes desde instancias SAT

Cada uno sigue esta estructura modular:
```
src/data_generation/[modulo]/
├── config.yaml              # Configuración centralizada
├── config_loader.py         # Carga y validación de config
├── [modulos_especificos].py # Lógica dividida en archivos enfocados
├── cli.py                   # Interfaz de línea de comandos
├── __init__.py             # Exports del módulo
└── README.md               # Documentación completa
```

### 2. Training Scripts (EN PROGRESO)
Actualmente reorganizando `training/jssp/train_images.py` → `src/training/jssp_images/`

**Archivos ya creados:**
1. **`src/training/jssp_images/config.yaml`** (132 líneas)
   - Configuración completa con secciones: experiment, data, model, training, output
   - Parámetros de arquitectura CNN, hiperparámetros, k-fold, etc.
   - Configuraciones específicas por tarea (classification/multilabel/regression)

2. **`src/training/jssp_images/config_loader.py`** (127 líneas)
   - `load_config()`: Carga y valida YAML
   - `merge_cli_args()`: Mezcla args CLI con config
   - `resolve_paths()`: Resuelve rutas relativas a absolutas

3. **`src/training/jssp_images/data_utils.py`** (310 líneas)
   - `detect_solver_cols()`: Detecta columnas *_Runtime_s y *_Score_S_rel
   - `build_labels()`: Construye etiquetas según tarea (classification/multilabel/regression)
   - `make_dataset()`: Crea tf.data.Dataset para entrenamiento
   - `normalize_image_paths()`, `filter_valid_images()`: Preprocesamiento de datos
   - `bss_index()`: Calcula baseline single solver
   - `multilabel_targets()`: Construye targets binarios para multilabel

## Archivos Fuente de Referencia

### Scripts Originales a Reorganizar:
1. **`training/jssp/train_images.py`** (885 líneas)
   - CNN para selección de solver JSSP desde imágenes 128x128x1
   - Soporta 3 tareas: classification, multilabel, regression
   - K-Fold cross-validation con baseline BSS
   - Genera métricas, gráficos (confusion matrix, PR curves, scatter plots)

2. **`training/jssp/train_tensor.py`** (752 líneas)
   - Similar a train_images pero para tensores 3D (JOBS, MACHINES, 2)
   - Padding a tamaño fijo MAX_JOBS=10, MAX_MACHINES=10
   - Misma estructura de tareas y evaluación

3. **`training/sat/train_images.py`** (892 líneas)
   - CNN para selección de solver SAT desde imágenes
   - Incluye métricas adicionales: resolved_rate, AST (Average Solving Time)
   - Soporta repeticiones de K-Fold (ej: 5x5)
   - Maneja columnas de status (*_Status) y tiempo de features

## Estructura Modular Objetivo

Para cada módulo de training (`jssp_images`, `jssp_tensors`, `sat_images`):

```
src/training/[modulo]/
├── config.yaml           # ✅ YA CREADO (jssp_images)
├── config_loader.py      # ✅ YA CREADO (jssp_images)
├── data_utils.py         # ✅ YA CREADO (jssp_images)
├── model_builder.py      # ⏳ PENDIENTE - Construcción de arquitectura CNN
├── training_loop.py      # ⏳ PENDIENTE - Lógica de entrenamiento por fold
├── evaluation.py         # ⏳ PENDIENTE - Métricas y evaluación
├── visualization.py      # ⏳ PENDIENTE - Gráficos (confusion, PR, scatter)
├── cli.py               # ⏳ PENDIENTE - CLI profesional con argparse
├── __init__.py          # ⏳ PENDIENTE - Exports
└── README.md            # ⏳ PENDIENTE - Documentación
```

## Próximos Pasos para Continuar

### Para `src/training/jssp_images/`:

1. **Crear `model_builder.py`**:
   - Extraer función `build_cnn()` del script original
   - Hacer configurable desde config.yaml (capas, filtros, dropout, etc.)
   - Soportar las 3 tareas con activaciones/losses apropiadas

2. **Crear `training_loop.py`**:
   - Extraer `train_fold()` y `run_kfold()` del script original
   - Usar early stopping configurable
   - Guardar checkpoints y predicciones por fold
   - Calcular baseline BSS

3. **Crear `evaluation.py`**:
   - Funciones de métricas por tarea (accuracy, f1, MAE)
   - Cálculo de métricas agregadas (mean, std)
   - Guardar resultados en JSON/CSV

4. **Crear `visualization.py`**:
   - `plot_confusion()`: Matriz de confusión
   - `plot_class_bars()`: Accuracy por clase
   - `plot_pr_multilabel()`: Curvas PR para multilabel
   - `plot_regression_scatter()`: Scatter plots para regresión
   - `plot_metrics_per_fold()`: Barras de métricas por fold

5. **Crear `cli.py`**:
   - Argparse con argumentos: --config, --csv, --task, --epochs, --folds, etc.
   - Cargar config, mezclar con CLI args
   - Ejecutar pipeline completo
   - Guardar resultados en `training/jssp/results/[timestamp]/`

6. **Crear `__init__.py`**:
   - Exportar funciones principales para uso como módulo

7. **Crear `README.md`**:
   - Explicar qué hace el módulo
   - Listar archivos y su propósito
   - Mostrar ejemplos de uso desde raíz del proyecto
   - Documentar configuración y parámetros

### Luego Repetir para:
- **`src/training/jssp_tensors/`**: Similar a jssp_images pero con:
  - Entrada: tensores 3D (MAX_JOBS, MAX_MACHINES, 2) con padding
  - Reutilizar data_utils adaptado para tensores
  - Arquitectura CNN ajustada para entrada 10x10x2

- **`src/training/sat_images/`**: Similar a jssp_images pero con:
  - Métricas adicionales: resolved_rate, AST_sec
  - Soporte para repeticiones de K-Fold (--repeats)
  - Manejo de columnas *_Status y feat_time_col
  - Filtrado opcional de solvers (--solvers)

## Principios de Diseño (Mantener Consistencia)

1. **Configuración Centralizada**: Todo en config.yaml, sin hardcoded values
2. **Modularidad**: Cada archivo con un propósito claro y enfocado
3. **Documentación**: Docstrings completos, type hints, comentarios donde necesario
4. **Nomenclatura Consistente**: Prefijos jssp_* y sat_* según corresponda
5. **Rutas Relativas**: Asumir ejecución desde raíz del proyecto
6. **Resultados Organizados**: Guardar en `training/[jssp|sat]/results/[run_name_timestamp]/`
7. **CLI Profesional**: Help text claro, ejemplos, validación de argumentos

## Comandos de Ejemplo Esperados

```bash
# JSSP Images
python -m src.training.jssp_images.cli \
  --config src/training/jssp_images/config.yaml \
  --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \
  --task classification \
  --epochs 30 \
  --folds 5

# JSSP Tensors  
python -m src.training.jssp_tensors.cli \
  --config src/training/jssp_tensors/config.yaml \
  --csv data/jssp/datasets/jsp_cnn_data_gen_2/ground_truth.csv \
  --task regression

# SAT Images
python -m src.training.sat_images.cli \
  --config src/training/sat_images/config.yaml \
  --csv data/sat/datasets/sat_cnn_data_gen_all/ground_truth.csv \
  --task classification \
  --folds 5 \
  --repeats 5 \
  --time_limit 1800
```

## Notas Importantes

- Los scripts originales en `training/` NO se deben borrar hasta verificar que todo funciona
- Mantener compatibilidad con estructura de resultados existente
- Asegurar reproducibilidad con seeds fijos
- Validar que todas las rutas se resuelvan correctamente desde raíz del proyecto
- Los resultados deben guardarse en `training/[jssp|sat]/results/` como antes