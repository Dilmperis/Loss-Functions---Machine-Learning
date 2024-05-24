import numpy as np
import torch
from torchmetrics import Accuracy, Precision, Recall, Specificity, F1Score
import tensorflow as tf
from tensorflow.keras import metrics

def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_positives / (true_positives + false_positives)

def recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    return true_positives / (true_positives + false_negatives)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)

def specificity(y_true, y_pred):
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_negatives / (true_negatives + false_positives)

# Example:
y_true = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0])

print("Accuracy:", f"{accuracy(y_true, y_pred):.3f}")
print("Precision:", f"{precision(y_true, y_pred):.3f}")
print("Recall:", f"{recall(y_true, y_pred):.3f}")
print("F1 Score:", f"{f1_score(y_true, y_pred):.3f}")
print("Specificity:", f"{specificity(y_true, y_pred):.3f}", '\n')

#######################################################################################################################
# Same Metrics with Pytorch

y_true_torch = torch.tensor(y_true)
y_pred_torch = torch.tensor(y_pred)

accuracy = Accuracy(task="binary")
precision = Precision(task="binary")
recall = Recall(task="binary")
f1 = F1Score(task="binary")
specificity = Specificity(task="binary")

accuracy_val = accuracy(y_pred_torch, y_true_torch)
precision_val = precision(y_pred_torch, y_true_torch)
recall_val = recall(y_pred_torch, y_true_torch)
f1_val = f1(y_pred_torch, y_true_torch)
specificity_val = specificity(y_pred_torch, y_true_torch)

print(f"Pytorch Accuracy: {accuracy_val.item():.3f}")
print("Pytorch Precision:", f"{precision_val.item():.3f}")
print("Pytorch Recall:", f"{recall_val.item():.3f}")
print("Pytorch F1 Score:", f"{f1_val.item():.3f}")
print("Pytorch Specificity:", f"{specificity_val.item():.3f}")

#######################################################################################################################
# Same Metrics with TensorFlow (NEEDS DEBUGGING!)

# y_true_tf = tf.convert_to_tensor(y_true)
# y_pred_tf = tf.convert_to_tensor(y_pred)

# # Reshape inputs for TensorFlow metrics
# y_true_tf_reshaped = tf.expand_dims(y_true_tf, axis=-1)
# y_pred_tf_reshaped = tf.expand_dims(y_pred_tf, axis=-1)

# accuracy = metrics.Accuracy()
# precision = metrics.Precision()
# recall = metrics.Recall()
# f1 = metrics.F1Score()

# accuracy_val = accuracy(y_true_tf, y_pred_tf)
# precision_val = precision(y_true_tf, y_pred_tf)
# recall_val = recall(y_true_tf, y_pred_tf)
# f1_val = f1(y_true_tf_reshaped, y_pred_tf_reshaped)  # Use reshaped inputs

# # Print the metrics
# print("TensorFlow Accuracy:", f"{accuracy_val.numpy():.3f}")
# print("TensorFlow Precision:", f"{precision_val.numpy():.3f}")
# print("TensorFlow Recall:", f"{recall_val.numpy():.3f}")
# print("TensorFlow F1 Score:", f"{f1_val.numpy():.3f}")

