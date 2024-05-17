import torch
from torch.nn import L1Loss
from tensorflow.keras import losses
import numpy as np
np.random.seed(0)

def custom_mae(y_pred, y_true):                 # Custom
    return np.mean(np.abs(y_pred - y_true))

# Given a toy target and prediction of 5 samples:
y_pred = np.array([.2, .4, .7, 1., 1.3])
y_true = np.array([.5, .4, .6, 1.5, 2])

tensor_pred = torch.tensor(y_pred)
tensor_true = torch.tensor(y_true)

# Losses
loss_custom = custom_mae(y_pred, y_true)         # Custom
l1_loss = L1Loss()                               # Pytorch
loss_torch = l1_loss(tensor_pred, tensor_true)
tf_loss = losses.MAE(y_pred, y_true)             # Tensorflow


#Print Statements
print(f'Custom MAE loss:\t {loss_custom}')
print(f'Torch MAE loss:\t\t {loss_torch}')
print(f'TensorFlow MAE loss:\t {tf_loss}')
