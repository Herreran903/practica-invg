# -*- coding: utf-8 -*-
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

# =========================
# Config reproducibilidad
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# =========================
# Constantes
# =========================
TIME_LIMIT_S = 60.0
TARGET_H = TARGET_W = 128
AUTOTUNE = tf.data.AUTOTUNE


# =========================
# Utils columnas / targets
# =========================
def detect_solver_cols(df: pd.DataFrame):
    runtime_cols = sorted([c for c in df.columns if c.endswith("_Runtime_s")])
    score_cols = sorted([c for c in df.columns if c.endswith("_Score_S_rel")])
    if not runtime_cols:
        raise ValueError("No hay columnas *_Runtime_s en el CSV.")
    return {"runtime": runtime_cols, "score": score_cols}


def argmin_runtime_or_score(row, solver_cols, use_score=False):
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    vals = row[cols].astype(float).values
    idx = int(np.nanargmin(vals))
    return idx, cols


def bss_index(train_df: pd.DataFrame, solver_cols, use_score=False):
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    means = train_df[cols].astype(float).mean(axis=0).values
    return int(np.nanargmin(means))


def multilabel_targets(row, runtime_cols):
    vals = row[runtime_cols].astype(float).values
    mask = np.isfinite(vals)
    out = np.zeros_like(vals, dtype=np.float32)
    out[mask] = (vals[mask] < TIME_LIMIT_S).astype(np.float32)
    return out


# =========================
# Carga de .npy
# =========================
def load_npy(path):
    arr = np.load(path.decode("utf-8")).astype(np.float32)  # HxW
    if arr.ndim == 2:
        arr = arr[..., None]  # HxWx1
    elif arr.ndim == 3 and arr.shape[-1] != 1:
        arr = arr[..., :1]
    return arr


def make_ds(paths, labels, task, batch_size, shuffle):
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


# =========================
# Modelo CNN
# =========================
def build_cnn(out_dim, task, lr=1e-3):
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


# =========================
# Labels por tarea
# =========================
def build_labels(df, solver_cols, task, use_score):
    if task == "classification":
        y = []
        for _, r in df.iterrows():
            idx, _ = argmin_runtime_or_score(r, solver_cols, use_score)
            y.append(idx)
        y = np.array(y, dtype=np.int32)
    elif task == "multilabel":
        rt_cols = solver_cols["runtime"]
        y = np.stack([multilabel_targets(r, rt_cols) for _, r in df.iterrows()], axis=0)
    else:
        rt_cols = solver_cols["runtime"]
        y = df[rt_cols].astype(float).values
        y[~np.isfinite(y)] = TIME_LIMIT_S * 10.0
    return y


# =========================
# Report/Plots helpers
# =========================
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
    # accuracy por clase (recall por clase)
    recalls = []
    for c in range(len(class_names)):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            recalls.append(np.nan)
        else:
            recalls.append(np.mean(y_pred[idx] == c))
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
    # y_true_bin, y_scores: [N, C]
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
    # guardar AP por etiqueta
    pd.DataFrame(
        {"label": list(ap_per_label.keys()), "AP": list(ap_per_label.values())}
    ).to_csv(table_csv, index=False)


def plot_regression_scatter(y_true, y_pred, class_names, out_png, table_csv):
    # y_true/pred: [N, C]
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


# =========================
# Entrenamiento por fold
# =========================
def train_fold(
    train_df, val_df, solver_cols, task, use_score, epochs, batch_size, outdir, fold_idx
):
    y_train = build_labels(train_df, solver_cols, task, use_score)
    y_val = build_labels(val_df, solver_cols, task, use_score)

    paths_train = train_df["Image_Npy_Path"].tolist()
    paths_val = val_df["Image_Npy_Path"].tolist()

    out_dim = (
        len(solver_cols["runtime"])
        if task != "classification"
        else len(
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
    )

    ds_train = make_ds(paths_train, y_train, task, batch_size, shuffle=True)
    ds_val = make_ds(paths_val, y_val, task, batch_size, shuffle=False)

    model = build_cnn(out_dim, task)
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
    ]
    _ = model.fit(
        ds_train, validation_data=ds_val, epochs=epochs, callbacks=cb, verbose=1
    )

    # Evaluación (evita retracing: usa model(xb, training=False))
    y_true_list, y_pred_list, y_score_list = [], [], []
    for xb, yb in ds_val:
        out = model(xb, training=False).numpy()
        if task == "classification":
            y_true_list.append(yb.numpy())
            y_pred_list.append(np.argmax(out, axis=1))
        elif task == "multilabel":
            y_true_list.append(yb.numpy())
            y_score_list.append(out)  # scores sigmoide
            y_pred_list.append((out >= 0.5).astype(np.float32))
        else:
            y_true_list.append(yb.numpy())
            y_pred_list.append(out)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    metrics = {}

    # Salvar predicciones del fold
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
        # métricas globales
        metrics["f1_micro"] = f1_score(
            y_true.flatten(), y_pred.flatten(), average="micro", zero_division=0
        )
        cols = solver_cols["runtime"]
        class_names = [c.replace("_Runtime_s", "") for c in cols]
        # PR curves y AP por etiqueta
        plot_pr_multilabel(
            y_true_bin=y_true,
            y_scores=y_scores,
            class_names=class_names,
            out_png=os.path.join(outdir, f"fold{fold_idx}_pr_curves.png"),
            table_csv=os.path.join(outdir, f"fold{fold_idx}_ap_per_label.csv"),
        )
        # Heatmap simple de F1 por etiqueta
        f1_per_label = []
        for j in range(y_true.shape[1]):
            f1_per_label.append(f1_score(y_true[:, j], y_pred[:, j], zero_division=0))
        pd.DataFrame({"label": class_names, "F1": f1_per_label}).to_csv(
            os.path.join(outdir, f"fold{fold_idx}_f1_per_label.csv"), index=False
        )
        # Grafiquito de barras F1 por etiqueta
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.bar(class_names, f1_per_label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fold{fold_idx}_f1_bars.png"))
        plt.close(fig)
    else:  # regression
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        cols = solver_cols["runtime"]
        class_names = [c.replace("_Runtime_s", "") for c in cols]
        # Plots y tabla por solver
        plot_regression_scatter(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_png=os.path.join(outdir, f"fold{fold_idx}_reg_scatter.png"),
            table_csv=os.path.join(outdir, f"fold{fold_idx}_mae_per_solver.csv"),
        )

    # guardar métricas del fold
    with open(os.path.join(outdir, f"fold{fold_idx}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return model, metrics


# =========================
# K-Fold + BSS
# =========================
def run_kfold(df, task, solver_cols, use_score, epochs, batch, folds, root_outdir):
    # Ajuste dinámico de folds en clasificación si hay clases con muy pocas muestras
    if task == "classification":
        labels = build_labels(df, solver_cols, task, use_score)
        binc = np.bincount(labels)
        min_class = binc.min()
        if min_class < folds:
            print(
                f"⚠️  Clase minoritaria con {min_class} muestras < folds={folds}. Ajustando folds a {max(2, min_class)}."
            )
            folds = max(2, int(min_class))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = splitter.split(df, labels)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = splitter.split(df)

    # nombres de clases para reportes
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
        bss_idx = bss_index(train_df, solver_cols, use_score)
        cols = (
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
        bss_col = cols[bss_idx]

        if task == "classification":
            y_val_true = build_labels(val_df, solver_cols, task, use_score)
            y_bss = np.full_like(y_val_true, bss_idx)
            bss_acc = accuracy_score(y_val_true, y_bss)
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_acc={bss_acc:.4f}")
        elif task == "multilabel":
            rt_cols = solver_cols["runtime"]
            y_val_true = np.stack(
                [multilabel_targets(r, rt_cols) for _, r in val_df.iterrows()], axis=0
            )
            y_bss = np.zeros_like(y_val_true)
            y_bss[:, bss_idx] = 1.0
            bss_acc = f1_score(
                y_val_true.flatten(), y_bss.flatten(), average="micro", zero_division=0
            )
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_f1_micro={bss_acc:.4f}")
        else:
            rt_cols = solver_cols["runtime"]
            const_pred = (
                train_df[rt_cols[bss_idx]]
                .astype(float)
                .fillna(TIME_LIMIT_S * 10)
                .mean()
            )
            y_val_true = val_df[rt_cols].astype(float).fillna(TIME_LIMIT_S * 10).values
            y_bss = np.full_like(y_val_true, const_pred)
            bss_acc = mean_absolute_error(y_val_true, y_bss)
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_mae={bss_acc:.4f}")

        print(f"[FOLD {i}] Entrenando modelo...")
        _, metrics = train_fold(
            train_df, val_df, solver_cols, task, use_score, epochs, batch, fold_dir, i
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

    # guardar tabla per-fold
    pf_df = pd.DataFrame(per_fold_rows)
    pf_df.to_csv(os.path.join(root_outdir, "metrics_per_fold.csv"), index=False)

    # agregados
    mean, std = float(np.mean(results)), float(np.std(results))
    agg = {"metric": key, "mean": mean, "std": std}
    with open(os.path.join(root_outdir, "metrics_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # bar final por fold
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(pf_df["fold"].astype(str), pf_df["value"])
    ax.set_ylabel(key)
    ax.set_xlabel("Fold")
    ax.set_title(f"{key} por fold")
    for xi, val in zip(pf_df["fold"].astype(str), pf_df["value"]):
        ax.text(int(xi) - 1, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root_outdir, f"{key}_per_fold.png"))
    plt.close(fig)

    return results, (mean, std)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", required=True, type=str, help="Ruta al CSV final (con Image_Npy_Path)."
    )
    ap.add_argument(
        "--task", required=True, choices=["classification", "multilabel", "regression"]
    )
    ap.add_argument(
        "--use_score",
        action="store_true",
        help="Usar *_Score_S_rel (si existen) para el 'mejor' solver.",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--outdir", type=str, default=None, help="Directorio base para reportes."
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "Image_Npy_Path" not in df.columns:
        raise ValueError(
            "El CSV no tiene 'Image_Npy_Path'. Corre la conversión primero."
        )
    solver_cols = detect_solver_cols(df)

    # carpeta de salida con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_outdir = args.outdir or os.path.join("training_reports", f"{args.task}_{ts}")
    ensure_dir(root_outdir)

    print(f"Reportes en: {root_outdir}")
    print(
        f"Solvers (runtime): {len(solver_cols['runtime'])} | Scores: {bool(solver_cols['score'])}"
    )
    print(
        f"Tarea: {args.task} | use_score={args.use_score} | folds={args.folds} | epochs={args.epochs}"
    )

    t0 = time.time()
    res, (mean, std) = run_kfold(
        df,
        args.task,
        solver_cols,
        args.use_score,
        args.epochs,
        args.batch_size,
        args.folds,
        root_outdir,
    )
    t1 = time.time()

    metric = {"classification": "ACC", "multilabel": "F1_micro", "regression": "MAE"}[
        args.task
    ]
    print(f"\n=== RESULTADO FINAL ({args.task}) ===")
    print(f"{metric} (KFold): {mean:.4f} ± {std:.4f}")
    print(f"Tiempo total: {t1 - t0:.2f}s")
    # escribir resumen legible
    with open(os.path.join(root_outdir, "README.txt"), "w") as f:
        f.write(
            f"Tarea: {args.task}\nuse_score: {args.use_score}\nfolds: {args.folds}\n"
        )
        f.write(f"Metric ({metric}): {mean:.4f} ± {std:.4f}\n")
        f.write(
            "Archivos:\n - metrics_per_fold.csv\n - metrics_summary.json\n - gráficos por fold (*.png)\n"
        )


if __name__ == "__main__":
    main()
