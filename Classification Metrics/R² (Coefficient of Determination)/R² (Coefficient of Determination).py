# !pip install torchmetrics
import numpy as np
import torch
import tensorflow as tf
from torchmetrics import R2Score

np.random.seed(42)  # Set the seed for reproducibility

# Custom R² score
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) * (y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true)) * (y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)


# Data
y_true_np = np.random.rand(10)
y_pred_np = np.random.rand(10)

y_true_torch = torch.tensor(y_true_np, dtype=torch.float32)
y_pred_torch = torch.tensor(y_pred_np, dtype=torch.float32)

y_true_tf = tf.constant(y_true_np, dtype=tf.float32)
y_pred_tf = tf.constant(y_pred_np, dtype=tf.float32)

# Reshape tensors to be 2D for TensorFlow
y_true_tf = tf.reshape(y_true_tf, (-1, 1))
y_pred_tf = tf.reshape(y_pred_tf, (-1, 1))



# PyTorch
r2_metric_torch = R2Score()
r2_torch = r2_metric_torch(y_pred_torch, y_true_torch)

# TensorFlow
r2_metric_tf = tf.keras.metrics.R2Score()
r2_metric_tf.update_state(y_true_tf, y_pred_tf)
r2_tf = r2_metric_tf.result()

# Custom R² Score
r2_np = r2_score(y_true_np, y_pred_np)


# Print Statements
print(f"Custom R² Score (Numpy):\t {r2_np:.4f}")
print(f"Built-in R² Score (PyTorch):\t {r2_torch:.4f}")
print(f"Built-in R² Score (TensorFlow):\t {r2_tf.numpy():.4f}")