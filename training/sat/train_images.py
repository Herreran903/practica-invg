# -*- coding: utf-8 -*-
"""
Entrena y evalúa una CNN para seleccionar/estimar el mejor solver en SAT a partir
de imágenes .npy generadas desde instancias (features en formato imagen).

NOVEDADES:
- --repeats: permite 5x5 (u otro) repitiendo la K-Fold con distintas semillas.
- --time_limit: tiempo límite (s) para AST y resolved_rate (default 1800).
- --feat_time_col: columna con tiempo de extracción de características (s) para sumar a AST.
- Reporta resolved_rate y AST_sec por fold (y global) incluyendo feature_time y penalización con timeout.

Dependencias:
NumPy, Pandas, scikit-learn, TensorFlow/Keras y Matplotlib.

Uso:
$ python train.py --csv sat_cnn_data_gen/ground_truth_sat.csv --task classification --epochs 30 --folds 5 --repeats 5 --time_limit 1800 --feat_time_col Feature_Extract_s
$ python train.py --csv data.csv --task multilabel --use_score --epochs 20 --folds 5 --repeats 5
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold

# ---------------------------------------------------------------------
# Reproducibilidad básica
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ---------------------------------------------------------------------
# Constantes generales del pipeline
# ---------------------------------------------------------------------
TARGET_H = TARGET_W = 128
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------------------------
# Configuración de salidas (pueden sobrescribirse por CLI)
# ---------------------------------------------------------------------
OUT_PARENT_DIR = "training_reports_sat"
RUN_NAME = "cnn_solver_selector_sat"
TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
APPEND_TASK_TO_DIRNAME = False


# =====================================================================
# Utilidades: detección de columnas y construcción de etiquetas
# =====================================================================
def detect_solver_cols(df: pd.DataFrame):
    """Detecta columnas de runtimes y (opcional) scores por solver."""
    runtime_cols = sorted([c for c in df.columns if c.endswith("_Runtime_s")])
    score_cols = sorted([c for c in df.columns if c.endswith("_Score_S_rel")])
    if not runtime_cols:
        raise ValueError("No hay columnas *_Runtime_s en el CSV.")
    return {"runtime": runtime_cols, "score": score_cols}


def argmin_runtime_or_score(row, solver_cols, use_score: bool = False):
    """Retorna el índice del mejor solver (mínimo valor) para una fila."""
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    vals = row[cols].astype(float).values
    idx = int(np.nanargmin(vals))
    return idx, cols


def bss_index(train_df: pd.DataFrame, solver_cols, use_score: bool = False):
    """Calcula el índice del Baseline Single Solver (BSS) por promedio en train."""
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    means = train_df[cols].astype(float).mean(axis=0).values
    return int(np.nanargmin(means))


def multilabel_targets(row, runtime_cols, time_limit_s: float):
    """Vector binario de solvers viables (runtime < time_limit_s)."""
    vals = row[runtime_cols].astype(float).values
    mask = np.isfinite(vals)
    out = np.zeros_like(vals, dtype=np.float32)
    out[mask] = (vals[mask] < time_limit_s).astype(np.float32)
    return out


# =====================================================================
# Lectura de imágenes .npy y construcción de datasets
# =====================================================================
def load_npy(path):
    """Carga una imagen .npy y asegura formato [H, W, 1] en float32."""
    arr = np.load(path.decode("utf-8")).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[-1] != 1:
        arr = arr[..., :1]
    return arr


def make_ds(paths, labels, task, batch_size, shuffle):
    """Crea un tf.data.Dataset a partir de rutas .npy y etiquetas."""
    x = tf.constant(paths)
    y = tf.constant(labels)

    def _map(p, t):
        img = tf.numpy_function(load_npy, [p], tf.float32)
        img.set_shape([TARGET_H, TARGET_W, 1])
        if task == "classification":
            t = tf.cast(t, tf.int32)
        else:
            t = tf.cast(t, tf.float32)
        return img, t

    ds = tf.data.Dataset.from_tensor_slices((x, y)).map(
        _map, num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# =====================================================================
# Modelo CNN
# =====================================================================
def build_cnn(out_dim, task, lr: float = 1e-3):
    """Construye y compila una CNN compacta para las tres tareas."""
    inputs = tf.keras.Input(shape=(TARGET_H, TARGET_W, 1))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    if task == "classification":
        outputs = tf.keras.layers.Dense(out_dim, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    elif task == "multilabel":
        outputs = tf.keras.layers.Dense(out_dim, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.AUC(curve="PR", name="auc_pr")]
    else:
        outputs = tf.keras.layers.Dense(out_dim, activation="linear")(x)
        loss = "mae"
        metrics = ["mae"]

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=metrics)
    return model


# =====================================================================
# Helpers
# =====================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion(y_true, y_pred, class_names, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_class_bars(y_true, y_pred, class_names, out_png):
    recalls = []
    for c in range(len(class_names)):
        idx = np.where(y_true == c)[0]
        recalls.append(np.nan if len(idx) == 0 else np.mean(y_pred[idx] == c))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(class_names, recalls)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy por clase")
    ax.set_title("Rendimiento por clase")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_pr_multilabel(y_true_bin, y_scores, class_names, out_png, table_csv):
    ap_per_label = {}
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for j, name in enumerate(class_names):
        yt = y_true_bin[:, j]
        ys = y_scores[:, j]
        if np.unique(yt).size < 2:
            continue
        p, r, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        ap_per_label[name] = ap
        ax.plot(r, p, label=f"{name} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall por etiqueta")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    pd.DataFrame(
        {"label": list(ap_per_label.keys()), "AP": list(ap_per_label.values())}
    ).to_csv(table_csv, index=False)


def plot_regression_scatter(y_true, y_pred, class_names, out_png, table_csv):
    fig, axes = plt.subplots(
        1, len(class_names), figsize=(4 * len(class_names), 4), dpi=150, squeeze=False
    )
    maes = []
    for j, name in enumerate(class_names):
        ax = axes[0, j]
        ax.scatter(y_true[:, j], y_pred[:, j], s=8, alpha=0.6)
        ax.plot(
            [y_true[:, j].min(), y_true[:, j].max()],
            [y_true[:, j].min(), y_true[:, j].max()],
            "k--",
            lw=1,
        )
        ax.set_xlabel("Real (s)")
        ax.set_ylabel("Predicho (s)")
        ax.set_title(name)
        maes.append(mean_absolute_error(y_true[:, j], y_pred[:, j]))
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    pd.DataFrame({"solver": class_names, "MAE": maes}).to_csv(table_csv, index=False)


# =====================================================================
# Resolved-rate y AST (clasificación y multilabel)
# =====================================================================
def _status_col_for(runtime_col: str) -> str:
    return runtime_col.replace("_Runtime_s", "_Status")


def _is_ok_row(row: pd.Series, runtime_col: str, time_limit_s: float) -> bool:
    """True si solver resuelve: *_Status OK/SAT/UNSAT, o runtime < time_limit_s."""
    status_col = _status_col_for(runtime_col)
    if status_col in row.index and pd.notna(row[status_col]):
        s = str(row[status_col]).strip().lower()
        if s in ("ok", "sat", "unsat"):
            return True
        if s in ("timeout", "time_out", "timedout", "memout", "crash", "error", "fail"):
            return False
    val = row.get(runtime_col, np.nan)
    try:
        rt = float(val)
    except Exception:
        rt = np.nan
    return np.isfinite(rt) and (rt < time_limit_s)


def compute_resolved_rate(
    val_df: pd.DataFrame, solver_runtime_cols: list[str], y_pred: np.ndarray, time_limit_s: float
) -> float:
    ok_flags = []
    for i, cls_idx in enumerate(y_pred):
        try:
            runtime_col = solver_runtime_cols[int(cls_idx)]
        except Exception:
            ok_flags.append(False)
            continue
        row = val_df.iloc[i]
        ok_flags.append(_is_ok_row(row, runtime_col, time_limit_s))
    return float(np.mean(ok_flags)) if ok_flags else 0.0


def compute_resolved_rate_multilabel(
    val_df: pd.DataFrame, solver_runtime_cols: list[str], y_pred_bin: np.ndarray, time_limit_s: float
) -> float:
    ok_flags = []
    n = y_pred_bin.shape[0]
    for i in range(n):
        row = val_df.iloc[i]
        pred_idxs = np.where(y_pred_bin[i] >= 0.3)[0]
        if pred_idxs.size == 0:
            ok_flags.append(False)
            continue
        any_ok = False
        for j in pred_idxs:
            runtime_col = solver_runtime_cols[j]
            if _is_ok_row(row, runtime_col, time_limit_s):
                any_ok = True
                break
        ok_flags.append(any_ok)
    return float(np.mean(ok_flags)) if ok_flags else 0.0


def compute_ast_classification(
    val_df: pd.DataFrame, solver_runtime_cols: list[str], y_pred: np.ndarray,
    feat_time_col: str, time_limit_s: float
) -> float:
    times = []
    for i, cls_idx in enumerate(y_pred):
        row = val_df.iloc[i]
        runtime_col = solver_runtime_cols[int(cls_idx)]
        rt = float(row.get(runtime_col, np.inf))
        feat_t = float(row.get(feat_time_col, 0.0))
        if not np.isfinite(rt) or rt >= time_limit_s:
            times.append(feat_t + time_limit_s)
        else:
            times.append(feat_t + rt)
    return float(np.mean(times)) if times else 0.0


def compute_ast_multilabel(
    val_df: pd.DataFrame, solver_runtime_cols: list[str], y_pred_bin: np.ndarray,
    feat_time_col: str, time_limit_s: float
) -> float:
    times = []
    for i in range(y_pred_bin.shape[0]):
        row = val_df.iloc[i]
        pred_idxs = np.where(y_pred_bin[i] >= 0.5)[0]
        feat_t = float(row.get(feat_time_col, 0.0))
        if pred_idxs.size == 0:
            times.append(feat_t + time_limit_s)
            continue
        rts = []
        for j in pred_idxs:
            rt = float(row.get(solver_runtime_cols[j], np.inf))
            rts.append(rt)
        best = np.nanmin(rts) if len(rts) else np.inf
        if not np.isfinite(best) or best >= time_limit_s:
            times.append(feat_t + time_limit_s)
        else:
            times.append(feat_t + best)
    return float(np.mean(times)) if times else 0.0


def ast_bss(val_df, rt_cols, bss_idx, feat_time_col, time_limit_s):
    times = []
    for _, row in val_df.iterrows():
        rt = float(row.get(rt_cols[bss_idx], np.inf))
        feat_t = float(row.get(feat_time_col, 0.0))
        if not np.isfinite(rt) or rt >= time_limit_s:
            times.append(feat_t + time_limit_s)
        else:
            times.append(feat_t + rt)
    return float(np.mean(times)) if times else 0.0


# =====================================================================
# Construcción de etiquetas por tarea
# =====================================================================
def build_labels(df, solver_cols, task, use_score, time_limit_s: float):
    if task == "classification":
        y = []
        for _, r in df.iterrows():
            idx, _ = argmin_runtime_or_score(r, solver_cols, use_score)
            y.append(idx)
        y = np.array(y, dtype=np.int32)
    elif task == "multilabel":
        rt_cols = solver_cols["runtime"]
        y = np.stack([multilabel_targets(r, rt_cols, time_limit_s) for _, r in df.iterrows()], axis=0)
    else:
        rt_cols = solver_cols["runtime"]
        y = df[rt_cols].astype(float).values
        # penalización fuerte a NaN (p. ej. solver que no corrió)
        y[~np.isfinite(y)] = time_limit_s * 10.0
    return y


# =====================================================================
# Entrenamiento/Evaluación por fold
# =====================================================================
def train_fold(
    train_df, val_df, solver_cols, task, use_score, epochs, batch_size, outdir, fold_idx,
    feat_time_col: str, time_limit_s: float
):
    """Entrena y evalúa un fold; guarda predicciones, métricas y gráficas."""
    # Etiquetas y rutas
    y_train = build_labels(train_df, solver_cols, task, use_score, time_limit_s)
    y_val = build_labels(val_df, solver_cols, task, use_score, time_limit_s)
    paths_train = train_df["Image_Npy_Path"].tolist()
    paths_val = val_df["Image_Npy_Path"].tolist()

    # Dimensión de salida
    out_dim = (
        len(solver_cols["runtime"])
        if task != "classification"
        else len(
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
    )

    # Datasets
    ds_train = make_ds(paths_train, y_train, task, batch_size, shuffle=True)
    ds_val = make_ds(paths_val, y_val, task, batch_size, shuffle=False)

    # Modelo + early stopping
    model = build_cnn(out_dim, task)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
    ]
    _ = model.fit(
        ds_train, validation_data=ds_val, epochs=epochs, callbacks=cb, verbose=1
    )

    # Inferencia
    y_true_list, y_pred_list, y_score_list = [], [], []
    for xb, yb in ds_val:
        out = model(xb, training=False).numpy()
        if task == "classification":
            y_true_list.append(yb.numpy())
            y_pred_list.append(np.argmax(out, axis=1))
        elif task == "multilabel":
            y_true_list.append(yb.numpy())
            y_score_list.append(out)
            y_pred_list.append((out >= 0.5).astype(np.float32))
        else:
            y_true_list.append(yb.numpy())
            y_pred_list.append(out)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    metrics = {}

    # Persistir predicciones
    np.save(os.path.join(outdir, f"fold{fold_idx}_y_true.npy"), y_true)
    np.save(os.path.join(outdir, f"fold{fold_idx}_y_pred.npy"), y_pred)

    class_names = None
    if task == "classification":
        metrics["acc"] = accuracy_score(y_true, y_pred)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        cols = (
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
        class_names = [
            c.replace("_Runtime_s", "").replace("_Score_S_rel", "") for c in cols
        ]
        labels_full = list(range(len(class_names)))
        cols_runtime_for_eval = cols

        metrics["resolved_rate"] = compute_resolved_rate(val_df, cols_runtime_for_eval, y_pred, time_limit_s)
        metrics["AST_sec"] = compute_ast_classification(val_df, cols_runtime_for_eval, y_pred, feat_time_col, time_limit_s)

        # CSV detalle resolución
        detail_rows = []
        for i, cls_idx in enumerate(y_pred):
            runtime_col = cols_runtime_for_eval[int(cls_idx)]
            status_col = _status_col_for(runtime_col)
            row = val_df.iloc[i]
            detail_rows.append(
                {
                    "Image_Npy_Path": row["Image_Npy_Path"],
                    "pred_solver": runtime_col.replace("_Runtime_s", ""),
                    "pred_runtime": row.get(runtime_col, np.nan),
                    "pred_status": row.get(status_col, np.nan) if status_col in val_df.columns else np.nan,
                    "resolved_ok": _is_ok_row(row, runtime_col, time_limit_s),
                    "feat_time_s": row.get(feat_time_col, np.nan),
                }
            )
        pd.DataFrame(detail_rows).to_csv(
            os.path.join(outdir, f"fold{fold_idx}_resolved_detail.csv"), index=False
        )

        plot_confusion(
            y_true,
            y_pred,
            class_names,
            os.path.join(outdir, f"fold{fold_idx}_confusion.png"),
        )
        plot_class_bars(
            y_true,
            y_pred,
            class_names,
            os.path.join(outdir, f"fold{fold_idx}_class_bars.png"),
        )

        rpt = classification_report(
            y_true,
            y_pred,
            labels=labels_full,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(rpt).transpose().to_csv(
            os.path.join(outdir, f"fold{fold_idx}_cls_report.csv")
        )

    elif task == "multilabel":
        y_scores = np.concatenate(y_score_list, axis=0) if y_score_list else y_pred
        metrics["f1_micro"] = f1_score(
            y_true.flatten(), y_pred.flatten(), average="micro", zero_division=0
        )
        cols = solver_cols["runtime"]
        class_names = [c.replace("_Runtime_s", "") for c in cols]

        metrics["resolved_rate"] = compute_resolved_rate_multilabel(val_df, cols, y_pred, time_limit_s)
        metrics["AST_sec"] = compute_ast_multilabel(val_df, cols, y_pred, feat_time_col, time_limit_s)

        # CSV detalle multilabel
        detail_rows = []
        for i in range(y_pred.shape[0]):
            row = val_df.iloc[i]
            pred_idxs = np.where(y_pred[i] >= 0.5)[0]
            pred_solvers = [class_names[j] for j in pred_idxs]
            solver_ok = []
            for j in pred_idxs:
                runtime_col = cols[j]
                solver_ok.append(
                    {
                        "solver": class_names[j],
                        "runtime": row.get(runtime_col, np.nan),
                        "status": row.get(_status_col_for(runtime_col), np.nan) if _status_col_for(runtime_col) in val_df.columns else np.nan,
                        "resolved_ok": _is_ok_row(row, runtime_col, time_limit_s),
                    }
                )
            any_ok = any(s["resolved_ok"] for s in solver_ok) if solver_ok else False
            detail_rows.append(
                {
                    "Image_Npy_Path": row["Image_Npy_Path"],
                    "pred_solvers": ";".join(pred_solvers) if pred_solvers else "",
                    "any_resolved_ok": any_ok,
                    "feat_time_s": row.get(feat_time_col, np.nan),
                    "detail": json.dumps(solver_ok, ensure_ascii=False),
                }
            )
        pd.DataFrame(detail_rows).to_csv(
            os.path.join(outdir, f"fold{fold_idx}_resolved_detail_multilabel.csv"),
            index=False,
        )

        plot_pr_multilabel(
            y_true_bin=y_true,
            y_scores=y_scores,
            class_names=class_names,
            out_png=os.path.join(outdir, f"fold{fold_idx}_pr_curves.png"),
            table_csv=os.path.join(outdir, f"fold{fold_idx}_ap_per_label.csv"),
        )

        f1_per_label = [
            f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
            for j in range(y_true.shape[1])
        ]
        pd.DataFrame({"label": class_names, "F1": f1_per_label}).to_csv(
            os.path.join(outdir, f"fold{fold_idx}_f1_per_label.csv"), index=False
        )

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.bar(class_names, f1_per_label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fold{fold_idx}_f1_bars.png"))
        plt.close(fig)

    else:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        cols = solver_cols["runtime"]
        class_names = [c.replace("_Runtime_s", "") for c in cols]
        plot_regression_scatter(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_png=os.path.join(outdir, f"fold{fold_idx}_reg_scatter.png"),
            table_csv=os.path.join(outdir, f"fold{fold_idx}_mae_per_solver.csv"),
        )

    with open(os.path.join(outdir, f"fold{fold_idx}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return model, metrics


# =====================================================================
# K-Fold + BSS + repetición
# =====================================================================
def run_kfold(
    df, task, solver_cols, use_score, epochs, batch, folds, root_outdir,
    seed: int, time_limit_s: float, feat_time_col: str
):
    """Ejecución de K-Fold con baseline BSS, reportes por fold y agregados."""
    # Splitter
    if task == "classification":
        labels = build_labels(df, solver_cols, task, use_score, time_limit_s)
        binc = np.bincount(labels)
        min_class = binc.min()
        if min_class < folds:
            print(
                f"⚠️  Clase minoritaria con {min_class} muestras < folds={folds}. Ajustando folds a {max(2, min_class)}."
            )
            folds = max(2, int(min_class))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(df, labels)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(df)

    cols_for_names = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    class_names = [
        c.replace("_Runtime_s", "").replace("_Score_S_rel", "") for c in cols_for_names
    ]

    results = []
    per_fold_rows = []
    for i, (tr, va) in enumerate(splits, start=1):
        fold_dir = ensure_dir(os.path.join(root_outdir, f"fold_{i}"))
        train_df = df.iloc[tr].reset_index(drop=True)
        val_df = df.iloc[va].reset_index(drop=True)

        # Baseline BSS
        bss_idx_i = bss_index(train_df, solver_cols, use_score)
        cols = cols_for_names
        bss_col = cols[bss_idx_i]

        if task == "classification":
            y_val_true = build_labels(val_df, solver_cols, task, use_score, time_limit_s)
            y_bss = np.full_like(y_val_true, bss_idx_i)
            bss_acc = accuracy_score(y_val_true, y_bss)
            bss_ast = ast_bss(val_df, cols, bss_idx_i, feat_time_col, time_limit_s)
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_acc={bss_acc:.4f} | BSS_AST={bss_ast:.1f}s")
        elif task == "multilabel":
            rt_cols = solver_cols["runtime"]
            y_val_true = np.stack(
                [multilabel_targets(r, rt_cols, time_limit_s) for _, r in val_df.iterrows()], axis=0
            )
            y_bss = np.zeros_like(y_val_true)
            y_bss[:, bss_idx_i] = 1.0
            bss_acc = f1_score(
                y_val_true.flatten(), y_bss.flatten(), average="micro", zero_division=0
            )
            bss_ast = ast_bss(val_df, rt_cols, bss_idx_i, feat_time_col, time_limit_s)
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_f1_micro={bss_acc:.4f} | BSS_AST={bss_ast:.1f}s")
        else:
            rt_cols = solver_cols["runtime"]
            const_pred = (
                train_df[rt_cols[bss_idx_i]].astype(float).fillna(time_limit_s * 10).mean()
            )
            y_val_true = val_df[rt_cols].astype(float).fillna(time_limit_s * 10).values
            y_bss = np.full_like(y_val_true, const_pred)
            bss_acc = mean_absolute_error(y_val_true, y_bss)
            bss_ast = None
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_mae={bss_acc:.4f}")

        # Entrenamiento
        print(f"[FOLD {i}] Entrenando modelo...")
        _, metrics = train_fold(
            train_df, val_df, solver_cols, task, use_score, epochs, batch, fold_dir, i,
            feat_time_col, time_limit_s
        )
        print(f"[FOLD {i}] VAL metrics: {metrics}")

        key = (
            "acc"
            if task == "classification"
            else ("f1_micro" if task == "multilabel" else "mae")
        )
        results.append(metrics[key])
        per_fold_rows.append(
            {"fold": i, "metric": key, "value": metrics[key], "baseline_bss": bss_acc}
        )

        if "resolved_rate" in metrics:
            per_fold_rows.append(
                {"fold": i, "metric": "resolved_rate", "value": metrics["resolved_rate"], "baseline_bss": None}
            )
        if "AST_sec" in metrics:
            per_fold_rows.append(
                {"fold": i, "metric": "AST_sec", "value": metrics["AST_sec"], "baseline_bss": bss_ast}
            )

    pf_df = pd.DataFrame(per_fold_rows)
    pf_df.to_csv(os.path.join(root_outdir, "metrics_per_fold.csv"), index=False)

    mean_, std_ = float(np.mean(results)), float(np.std(results))
    agg = {"metric": key, "mean": mean_, "std": std_}
    with open(os.path.join(root_outdir, "metrics_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars_df = pf_df[pf_df["metric"] == key]
    ax.bar(bars_df["fold"].astype(str), bars_df["value"])
    ax.set_ylabel(key)
    ax.set_xlabel("Fold")
    ax.set_title(f"{key} por fold")
    for xi, val in zip(bars_df["fold"].astype(str), bars_df["value"]):
        ax.text(int(xi) - 1, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root_outdir, f"{key}_per_fold.png"))
    plt.close(fig)

    return results, (mean_, std_)


# =====================================================================
# Main (CLI)
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="Ruta al CSV final (con Image_Npy_Path).")
    ap.add_argument("--task", required=True, choices=["classification", "multilabel", "regression"])
    ap.add_argument("--use_score", action="store_true", help="Usar *_Score_S_rel (si existen) para el 'mejor' solver.")
    ap.add_argument("--solvers", type=str, default=None,
                    help="Lista separada por coma con los solvers a usar (ej: clasp1,glucose2,lingeling).")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1, help="Repeticiones de la K-Fold (p.ej., 5 para 5x5).")
    ap.add_argument("--time_limit", type=float, default=1800.0, help="Tiempo límite (s) para AST/Resolved (default 1800).")
    ap.add_argument("--feat_time_col", type=str, default=None,
                    help="Columna con tiempo de extracción de features (s). Si no existe, se asume 0.")
    # Salidas
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--out_parent", type=str, default=None)
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()

    # Carga CSV
    df = pd.read_csv(args.csv)
    if "Image_Npy_Path" not in df.columns:
        raise ValueError("El CSV no tiene 'Image_Npy_Path'. Corre la conversión primero.")

    # Normalización de rutas a absolutas
    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    def _norm_path(p):
        p = "" if (p is None or (isinstance(p, float) and np.isnan(p))) else str(p).strip()
        if not p:
            return ""
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(csv_dir, p))

    df["Image_Npy_Path"] = df["Image_Npy_Path"].apply(_norm_path)
    exists_mask = df["Image_Npy_Path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    missing = int((~exists_mask).sum())
    total = int(len(df))
    if missing > 0:
        print(f"⚠️  {missing}/{total} filas no tienen imagen válida. Serán descartadas.")
    df = df.loc[exists_mask].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No hay imágenes válidas tras normalizar rutas.")

    # Detección de columnas
    solver_cols = detect_solver_cols(df)

    # Filtrado opcional de solvers
    if args.solvers:
        selected = [s.strip() for s in args.solvers.split(",") if s.strip()]
        missing_solvers = [s for s in selected if f"{s}_Runtime_s" not in df.columns]
        if missing_solvers:
            raise ValueError(f"Los siguientes solvers no están en el CSV: {missing_solvers}")

        keep_runtime = [f"{s}_Runtime_s" for s in selected]
        keep_score = [f"{s}_Score_S_rel" for s in selected if f"{s}_Score_S_rel" in df.columns]
        keep_status = [f"{s}_Status" for s in selected if f"{s}_Status" in df.columns]

        base_cols = ["Image_Npy_Path"]
        df = df[base_cols + keep_runtime + keep_score + keep_status]
        solver_cols = {"runtime": keep_runtime, "score": keep_score}

        # Diagnósticos
        rt_cols = solver_cols["runtime"]
        any_true = (df[rt_cols].astype(float) < args.time_limit).any(axis=1)
        print("any_true %:", any_true.mean())

        bad_status = {"timeout","time_out","timedout","memout","crash","error","fail"}
        cnt_incons = 0
        tot_incons = 0
        for _, row in df.iterrows():
            for c in rt_cols:
                status_c = c.replace("_Runtime_s", "_Status")
                if status_c in df.columns and pd.notna(row[status_c]):
                    s = str(row[status_c]).strip().lower()
                    r = float(row[c]) if pd.notna(row[c]) else np.inf
                    if r < args.time_limit:
                        tot_incons += 1
                        if s in bad_status:
                            cnt_incons += 1
        print("inconsistencias runtime<time_limit & status malo:", cnt_incons, "/", tot_incons)

    # Columna de tiempo de features
    if args.feat_time_col and args.feat_time_col not in df.columns:
        raise ValueError(f"La columna de tiempo de features '{args.feat_time_col}' no existe en el CSV.")
    if not args.feat_time_col:
        df["_feat_time_zero_"] = 0.0
        args.feat_time_col = "_feat_time_zero_"

    # Salidas
    ts = datetime.now().strftime(TIMESTAMP_FMT)
    parent_dir = args.out_parent or args.outdir or OUT_PARENT_DIR
    run_name = args.run_name or RUN_NAME
    run_dirname = f"{run_name}_{args.task}_{ts}" if APPEND_TASK_TO_DIRNAME else f"{run_name}_{ts}"
    root_outdir = ensure_dir(os.path.join(parent_dir, run_dirname))

    print(f"Reportes en: {root_outdir}")
    print(f"Solvers (runtime): {len(solver_cols['runtime'])} | Scores: {bool(solver_cols['score'])}")
    print(f"Tarea: {args.task} | use_score={args.use_score} | folds={args.folds} | repeats={args.repeats} | epochs={args.epochs}")
    print(f"time_limit={args.time_limit} | feat_time_col='{args.feat_time_col}'")

    # Repeticiones (p. ej., 5 para 5x5)
    metric_name = {"classification": "ACC", "multilabel": "F1_micro", "regression": "MAE"}[args.task]
    all_runs = []
    global_rows = []

    t0 = time.time()
    for rep in range(args.repeats):
        seed_rep = SEED + rep
        rep_dir = ensure_dir(os.path.join(root_outdir, f"rep_{rep+1}"))
        print(f"\n=== Repetición {rep+1}/{args.repeats} (seed={seed_rep}) ===")
        res, (mean_, std_) = run_kfold(
            df=df,
            task=args.task,
            solver_cols=solver_cols,
            use_score=args.use_score,
            epochs=args.epochs,
            batch=args.batch_size,
            folds=args.folds,
            root_outdir=rep_dir,
            seed=seed_rep,
            time_limit_s=args.time_limit,
            feat_time_col=args.feat_time_col,
        )
        all_runs.extend(res)
        global_rows.append({"rep": rep+1, "mean": mean_, "std": std_})

    t1 = time.time()

    # Agregado global sobre todas las repeticiones
    global_mean, global_std = float(np.mean(all_runs)), float(np.std(all_runs))
    pd.DataFrame(global_rows).to_csv(os.path.join(root_outdir, "metrics_summary_per_rep.csv"), index=False)
    with open(os.path.join(root_outdir, "metrics_summary_GLOBAL.json"), "w") as f:
        json.dump({"metric": metric_name, "mean": global_mean, "std": global_std}, f, indent=2)

    print(f"\n=== GLOBAL ({args.task}, {args.folds}x{args.repeats}) === {metric_name}: {global_mean:.4f} ± {global_std:.4f}")
    print(f"Tiempo total: {t1 - t0:.2f}s")

    with open(os.path.join(root_outdir, "README.txt"), "w") as f:
        f.write(
            f"Tarea: {args.task}\nuse_score: {args.use_score}\nfolds: {args.folds}\nrepeats: {args.repeats}\n"
        )
        f.write(f"time_limit: {args.time_limit}\nfeat_time_col: {args.feat_time_col}\n")
        f.write(f"GLOBAL {metric_name}: {global_mean:.4f} ± {global_std:.4f}\n")
        f.write("Archivos:\n - metrics_summary_GLOBAL.json\n - metrics_summary_per_rep.csv\n - rep_*/metrics_*.json\n")


if __name__ == "__main__":
    main()
