import torch
from torch.nn import MSELoss
from tensorflow.keras import losses
import numpy as np
np.random.seed(0)

def custom_mse(y_pred, y_true):  # Custom
    return np.mean((y_pred - y_true) ** 2)

# Given a toy target and prediction of 5 samples:
y_pred = np.array([.2, .4, .7, 1., 1.3])
y_true = np.array([.5, .4, .6, 1.5, 2])

tensor_pred = torch.tensor(y_pred)
tensor_true = torch.tensor(y_true)

#Losses
mse_custom = custom_mse(y_pred, y_true) # Custom
MSE_torch = MSELoss()  # Pytorch
mse_torch = MSE_torch(tensor_pred, tensor_true)
mse_tf = losses.MSE(y_pred, y_true)  # Tensorflow

# Print Statements
print(f'Custom MSE loss:\t {mse_custom:.2f}')
print(f'Torch MSE loss:\t\t {mse_torch:.2f}')
print(f'TensorFlow MSE loss:\t {mse_tf:.2f}')
