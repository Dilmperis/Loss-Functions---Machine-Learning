import torch
from tensorflow.keras import losses
import numpy as np
np.random.seed(0)

def custom_mape(y_pred, y_true):  # Custom
    return np.mean((np.abs(y_pred - y_true) / np.abs(y_pred)) * 100)

# Given a toy target and prediction of 5 samples:
y_pred = np.array([.2, .4, .7, 1., 1.3])
y_true = np.array([.5, .4, .6, 1.5, 2])

tensor_pred = torch.tensor(y_pred)
tensor_true = torch.tensor(y_true)

# Losses
mape_custom = custom_mape(y_pred, y_true) # Custom
# Pytorch DOES NOT HAVE Built-in MAPE
mape_tf = losses.MAPE(y_pred, y_true)  # Tensorflow

# Print Statements
print(f'Custom MAPE loss:\t {mape_custom:.2f}%')
print(f'TensorFlow MAPE loss:\t {mape_tf:.2f}%')
