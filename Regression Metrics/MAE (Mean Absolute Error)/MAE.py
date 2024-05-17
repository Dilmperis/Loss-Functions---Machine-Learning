from torch.nn import L1Loss
from tensorflow.keras import losses
np.random.seed(0)

def custom_mae(y_pred, y_true):                 # Custom
    return np.mean(np.abs(y_pred - y_true))

loss_custom = custom_mae(y_pred, y_true)

l1_loss = L1Loss()                               # Pytorch
loss_torch = l1_loss(tensor_pred, tensor_true)

tf_loss = losses.MAE(y_pred, y_true)             # Tensorflow


#Print Statements
print(f'Custom MAE loss:\t {loss_custom}')
print(f'Torch MAE loss:\t\t {loss_torch}')
print(f'TensorFlow MAE loss:\t {tf_loss}')
