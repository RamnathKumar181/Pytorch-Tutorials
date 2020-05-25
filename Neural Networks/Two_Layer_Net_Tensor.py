# -*- coding: utf-8 -*-
"""
Two Layer Network: Tensors
----------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.

This implementation uses PyTorch tensors to manually compute the forward pass,
loss, and backward pass.

A PyTorch Tensor is basically the same as a numpy array: it does not know
anything about deep learning or computational graphs or gradients, and is just
a generic n-dimensional array to be used for arbitrary numeric computation.

The biggest difference between a numpy array and a PyTorch Tensor is that
a PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU,
just cast the Tensor to a cuda datatype.
"""

import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

def createDataset(N,D_in,D_out):
  x = torch.randn(N, D_in, device=device, dtype=dtype)
  y = torch.randn(N, D_out, device=device, dtype=dtype)
  return x,y

def initializeWeights():
  # randn(x,y) creates a 2D matrix of size x*y
  # Here, w1 is the weights to the first layer and w2 is to the output layer
  w1 = torch.randn(D_in, H, device=device, dtype=dtype)
  w2 = torch.randn(H, D_out, device=device, dtype=dtype)
  return w1,w2

if __name__=="__main__":

  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  N, D_in, H, D_out = 64, 1000, 100, 10

  # Create random input and output data
  x,y = createDataset(N,D_in,D_out)

  # Randomly initialize weights
  w1,w2 = initializeWeights()

  # Initialize other hyperparameters
  lr = 1e-6
  epochs = 500

  # Iterate through epochs
  for t in range(epochs):
    # Forward pass
    a1 = x.mm(w1)
    z1 = a1.clamp(min=0)
    y_pred = z1.mm(w2)

    # Compute and print loss
    loss = 0.5*(y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = z1.t().mm(grad_y_pred)
    grad_z1 = grad_y_pred.mm(w2.t())
    grad_a1 = grad_z1.clone()
    grad_a1[a1 < 0] = 0
    grad_w1 = x.t().mm(grad_a1)

    # Update weights using gradient descent
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
