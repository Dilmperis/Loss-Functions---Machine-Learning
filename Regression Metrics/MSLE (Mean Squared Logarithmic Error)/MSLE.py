from tensorflow.keras import losses
import numpy as np
np.random.seed(0)

def custom_msle(y_pred, y_true):  # Custom
    return np.mean((np.log(1 + y_true) - np.log(1 + y_pred)) ** 2)

# Given a toy target and prediction of 5 samples:
y_pred = np.array([.2, .4, .7, 1., 1.3])
y_true = np.array([.5, .4, .6, 1.5, 2])

tensor_pred = torch.tensor(y_pred)
tensor_true = torch.tensor(y_true)

# Losses
msle_custom = custom_msle(y_pred, y_true)
# Pytorch DOES NOT HAVE Built-in MSLE
msle_tf = losses.MSLE(y_pred, y_true)  # Tensorflow

# Print Statements
print(f'Custom MSLE loss:\t {msle_custom:.5f}')
print(f'TensorFlow MSLE loss:\t {msle_tf:.5f}')
