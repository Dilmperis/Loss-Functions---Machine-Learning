def custom_mse(y_pred, y_true):  # Custom
    return np.mean((y_pred - y_true) ** 2)


mse_custom = custom_mse(y_pred, y_true)

MSE_torch = MSELoss()  # Pytorch
mse_torch = MSE_torch(tensor_pred, tensor_true)

mse_tf = losses.MSE(y_pred, y_true)  # Tensorflow

# Print Statements
print(f'Custom MSE loss:\t {mse_custom}')
print(f'Torch MSE loss:\t\t {mse_torch}')
print(f'TensorFlow MSE loss:\t {mse_tf}')
