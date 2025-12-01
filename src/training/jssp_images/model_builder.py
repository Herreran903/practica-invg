# -*- coding: utf-8 -*-
"""
Model builder for JSSP image-based solver selection.

This module provides functions to build and compile CNN models for three tasks:
- Classification: Select the best solver (softmax output)
- Multilabel: Identify viable solvers under time limit (sigmoid output)
- Regression: Predict runtime for each solver (linear output)

The architecture is configurable via config.yaml parameters.
"""

from typing import Literal, Tuple

import tensorflow as tf


def build_cnn(
    input_shape: Tuple[int, int, int],
    output_dim: int,
    task: Literal["classification", "multilabel", "regression"],
    conv_filters: list[int] = [16, 32, 64],
    conv_kernel_size: int = 3,
    pool_size: int = 2,
    dropout_conv: float = 0.25,
    dense_units: int = 256,
    dropout_dense: float = 0.5,
    learning_rate: float = 0.03,
    momentum: float = 0.9,
) -> tf.keras.Model:
    """
    Build and compile a CNN model for JSSP solver selection.

    Architecture:
    - Multiple Conv2D + MaxPooling2D blocks (configurable)
    - Dropout after convolutions
    - Flatten
    - Dense layer with dropout
    - Task-specific output layer

    Args:
        input_shape: Shape of input images (height, width, channels).
        output_dim: Number of output units (classes/solvers).
        task: Task type - 'classification', 'multilabel', or 'regression'.
        conv_filters: List of filter counts for each Conv2D layer.
        conv_kernel_size: Kernel size for Conv2D layers.
        pool_size: Pool size for MaxPooling2D layers.
        dropout_conv: Dropout rate after convolutions.
        dense_units: Number of units in the dense layer.
        dropout_dense: Dropout rate after dense layer.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        Compiled Keras model ready for training.

    Raises:
        ValueError: If task is not one of the supported types.

    Design Decision:
        Lightweight architecture to prevent overfitting on 128x128x1 images.
        Task-specific activation and loss functions ensure proper training.
    """
    if task not in ["classification", "multilabel", "regression"]:
        raise ValueError(
            f"Invalid task '{task}'. Must be 'classification', 'multilabel', or 'regression'."
        )

    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    x = inputs

    # Convolutional blocks: (Conv2D + MaxPooling2D + Dropout) repeated
    # By default uses three blocks with filters [16, 32, 64].
    for i, filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv2D(
            filters,
            conv_kernel_size,
            padding="same",
            activation="relu",
            name=f"conv_{i+1}",
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size, name=f"pool_{i+1}")(x)
        x = tf.keras.layers.Dropout(
            dropout_conv, name=f"dropout_conv_{i+1}"
        )(x)

    # Flatten and dense layers: two dense layers with dropout in between
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(dropout_dense, name="dropout_dense_1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_2")(x)

    # Task-specific output layer
    if task == "classification":
        outputs = tf.keras.layers.Dense(
            output_dim, activation="softmax", name="output_softmax"
        )(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    elif task == "multilabel":
        outputs = tf.keras.layers.Dense(
            output_dim, activation="sigmoid", name="output_sigmoid"
        )(x)
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.AUC(curve="PR", name="auc_pr")]
    else:  # regression
        outputs = tf.keras.layers.Dense(
            output_dim, activation="linear", name="output_linear"
        )(x)
        loss = "mae"
        metrics = ["mae"]

    # Build and compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"cnn_{task}")
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=True,
        ),
        loss=loss,
        metrics=metrics,
    )

    return model


def build_model_from_config(
    config: dict,
    output_dim: int,
    task: Literal["classification", "multilabel", "regression"],
) -> tf.keras.Model:
    """
    Build a CNN model using parameters from configuration dictionary.

    This is a convenience wrapper around build_cnn() that extracts
    architecture parameters from a config dict (typically loaded from YAML).

    Args:
        config: Configuration dictionary with 'model' section containing
                architecture parameters.
        output_dim: Number of output units (classes/solvers).
        task: Task type - 'classification', 'multilabel', or 'regression'.

    Returns:
        Compiled Keras model.

    Example:
        >>> config = load_config("config.yaml")
        >>> model = build_model_from_config(config, output_dim=5, task="classification")
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    # Extract input shape from data config
    target_h = data_cfg.get("target_height", 128)
    target_w = data_cfg.get("target_width", 128)
    input_shape = (target_h, target_w, 1)

    # Extract architecture parameters (kept for backward compatibility; defaults
    # match the paper: three Conv2D blocks with filters [16, 32, 64])
    conv_filters = model_cfg.get("conv_filters", [16, 32, 64])
    conv_kernel_size = model_cfg.get("conv_kernel_size", 3)
    pool_size = model_cfg.get("pool_size", 2)
    dropout_conv = model_cfg.get("dropout_conv", 0.25)
    dense_units = model_cfg.get("dense_units", 256)
    dropout_dense = model_cfg.get("dropout_dense", 0.5)

    # Extract training parameters (SGD with Nesterov momentum, as in the paper)
    training_cfg = config.get("training", {})
    learning_rate = training_cfg.get("learning_rate", 0.03)
    momentum = training_cfg.get("momentum", 0.9)

    return build_cnn(
        input_shape=input_shape,
        output_dim=output_dim,
        task=task,
        conv_filters=conv_filters,
        conv_kernel_size=conv_kernel_size,
        pool_size=pool_size,
        dropout_conv=dropout_conv,
        dense_units=dense_units,
        dropout_dense=dropout_dense,
        learning_rate=learning_rate,
        momentum=momentum,
    )


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a string representation of the model architecture.

    Args:
        model: Keras model to summarize.

    Returns:
        String containing model summary.
    """
    import io

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()
