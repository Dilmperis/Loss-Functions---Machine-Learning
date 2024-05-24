import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf


def softmax(input):
    total_sum = sum(np.exp(input))
    results = []
    for i in input:
        results.append(round(np.exp(i)/total_sum, 4))
    return results


def cross_entropy(logits, true_labels):
    number_samples = len(true_labels)
    sum_entropy = 0
    for logit, label in zip(logits,true_labels):
        sum_entropy += -np.log(softmax(logit)[label])
    return (1/number_samples) * sum_entropy

# Example
values = [
    [2.3, 0.5, -1.2, 0.3, 0.1, 1.0, -0.5, 0.8, 0.7, 0.2],
    [0.1, 0.2, 1.5, -0.8, 0.4, -1.0, 0.3, 0.8, 0.7, -0.2],
    [0.3, -0.5, 0.8, -1.2, 2.0, 0.7, 0.1, 0.4, 0.2, -0.9],
    [-1.5, 0.2, 0.3, 0.5, -0.8, 1.1, 0.3, -0.2, 0.9, 0.2]
]
labels = [0, 2, 4, 5]

loss = cross_entropy(values, labels)
print(f"Custom Cross-Entropy Loss:\t {loss:.4f}")

# Pytorch
'''
Cross Entropy with Pytorch is accomplished with the module torch.nn.CrossEntropyLoss()
This module takes as parameters the model's predictions and the target.
The model's predictions are expected to contain the unnormalized logits for each class (which do `not` need
to be positive or sum to 1, in general). 

The `target` is expected to contain either: 
1) Class indices in the range :math:`[0, C)` where `C` is the number of classes
2) Probabilities for each class

In this implementation we will create an example for each one of the target forms.
'''

# Class indices
logits = torch.tensor(values, dtype=torch.float32)
true_labels = torch.tensor(labels, dtype=torch.long)

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, true_labels)

print(f"Pytorch Cross-Entropy Loss:\t {loss.item():.4f}")


# Tensorflow
'''
In TensorFlow, there is no single function exactly equivalent to nn.CrossEntropyLoss in PyTorch that combines 
the softmax and cross-entropy calculation into one step for sparse labels. However, TensorFlow does provide 
combined operations that achieve the same functionality, specifically tailored for sparse and dense labels. 
These are tf.nn.sparse_softmax_cross_entropy_with_logits for sparse labels and 
tf.nn.softmax_cross_entropy_with_logits for dense labels.
'''

logits = tf.constant(values, dtype=tf.float32)
true_labels = tf.constant(labels, dtype=tf.int32)

cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=true_labels) # Compute the cross-entropy loss
mean_loss = tf.reduce_mean(cross_entropy_loss) # Compute the mean loss

print(f"Tensorflow Cross-Entropy Loss:\t {mean_loss.numpy():.4f}")

########################################################################################################################
# Cross Entropy with target as class probabilities instead of class indices
torch.manual_seed(42) # Set the seed for reproducibility
input = torch.randn(3, 5)
target = torch.randn(3, 5).softmax(dim=1)

# Custom function of cross entropy with probabilities targets:
def custom_cross_entropy_loss(input_tensor, target_tensor):
    softmax_input = F.softmax(input_tensor, dim=1)   # Compute the softmax of the input tensor along the second dimension
    loss = -torch.sum(target_tensor * torch.log(softmax_input))
    loss /= input_tensor.size(0) # take the mean of the loss
    return loss

custom_loss = custom_cross_entropy_loss(input, target)
print("\n")
print(f"Custom Cross-entropy Loss with probabilities:\t\t  {custom_loss:.4f}")


# Example of Pytorch with target as class probabilities instead of class indices
loss = nn.CrossEntropyLoss()
output = loss(input, target)

print(f'Pytorch Cross entropy Loss with probabilities:\t\t  {output.item():.4f}')

#  TensorFlow's built-in function
tf_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
tf_loss = tf_loss_fn(target, input) # Caution, tf_loss_fn requires the (target, input) and not (input, target)
print(f"TensorFlow Cross-entropy Loss with probabilities:\t  {tf_loss.numpy():.4f}")
