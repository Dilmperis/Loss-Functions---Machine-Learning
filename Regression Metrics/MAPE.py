def custom_mape(y_pred, y_true):  # Custom
    return np.mean((np.abs(y_pred - y_true) / np.abs(y_pred)) * 100)


mape_custom = custom_mape(y_pred, y_true)

# Pytorch DOES NOT HAVE Built-in MAPE

mape_tf = losses.MAPE(y_pred, y_true)  # Tensorflow

# Print Statements
print(f'Custom MAPE loss:\t {mape_custom:.2f}%')
print(f'TensorFlow MAPE loss:\t {mape_tf:.2f}%')