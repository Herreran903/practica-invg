# -*- coding: utf-8 -*-
"""
Model builder for JSSP tensor-based solver selection.

Builds CNN models for 3D tensor inputs (JOBS x MACHINES x 2).
Architecture adapted for smaller spatial dimensions compared to images.
"""

import tensorflow as tf
from typing import Literal, Tuple


def build_cnn(
    input_shape: Tuple[int, int, int],
    output_dim: int,
    task: Literal["classification", "multilabel", "regression"],
    conv_filters: list[int] = [32, 64, 128],
    conv_kernel_size: int = 3,
    pool_size: int = 2,
    dropout_dense: float = 0.4,
    dense_units: int = 256,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build and compile CNN for JSSP tensor inputs.
    
    Architecture optimized for 10x10x2 inputs:
    - Multiple Conv2D + MaxPooling2D blocks
    - Flatten
    - Dense layer with dropout
    - Task-specific output layer
    
    Args:
        input_shape: Shape of input tensors (max_jobs, max_machines, n_channels).
        output_dim: Number of output units.
        task: Task type.
        conv_filters: List of filter counts for Conv2D layers.
        conv_kernel_size: Kernel size for convolutions.
        pool_size: Pool size for MaxPooling2D.
        dropout_dense: Dropout rate after dense layer.
        dense_units: Number of units in dense layer.
        learning_rate: Learning rate for Adam.
    
    Returns:
        Compiled Keras model.
    """
    if task not in ["classification", "multilabel", "regression"]:
        raise ValueError(f"Invalid task '{task}'")
    
    inputs = tf.keras.Input(shape=input_shape, name="tensor_input")
    x = inputs
    
    # Convolutional blocks
    for i, filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv2D(
            filters,
            conv_kernel_size,
            padding="same",
            activation="relu",
            name=f"conv_{i+1}",
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size, name=f"pool_{i+1}")(x)
    
    # Flatten and dense
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(dropout_dense, name="dropout_dense")(x)
    
    # Task-specific output
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
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"cnn_tensor_{task}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    Build CNN model using configuration dictionary.
    
    Args:
        config: Configuration dictionary.
        output_dim: Number of output units.
        task: Task type.
    
    Returns:
        Compiled Keras model.
    """
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    
    # Input shape from data config
    max_jobs = data_cfg.get("max_jobs", 10)
    max_machines = data_cfg.get("max_machines", 10)
    n_channels = data_cfg.get("n_channels", 2)
    input_shape = (max_jobs, max_machines, n_channels)
    
    # Architecture parameters
    conv_filters = model_cfg.get("conv_filters", [32, 64, 128])
    conv_kernel_size = model_cfg.get("conv_kernel_size", 3)
    pool_size = model_cfg.get("pool_size", 2)
    dropout_dense = model_cfg.get("dropout_dense", 0.4)
    dense_units = model_cfg.get("dense_units", 256)
    
    # Training parameters
    learning_rate = training_cfg.get("learning_rate", 1e-3)
    
    return build_cnn(
        input_shape=input_shape,
        output_dim=output_dim,
        task=task,
        conv_filters=conv_filters,
        conv_kernel_size=conv_kernel_size,
        pool_size=pool_size,
        dropout_dense=dropout_dense,
        dense_units=dense_units,
        learning_rate=learning_rate,
    )


def get_model_summary(model: tf.keras.Model) -> str:
    """Get string representation of model architecture."""
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()