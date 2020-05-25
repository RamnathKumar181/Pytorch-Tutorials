# -*- coding: utf-8 -*-
"""
Two Layer Network: Tensors and autograd
-------------------------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.

This implementation computes the forward pass using operations on PyTorch
Tensors, and uses PyTorch autograd to compute gradients.


A PyTorch Tensor represents a node in a computational graph. If ``x`` is a
Tensor that has ``x.requires_grad=True`` then ``x.grad`` is another Tensor
holding the gradient of ``x`` with respect to some scalar value.
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
  w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
  w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
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
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss
    loss = 0.5*(y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
