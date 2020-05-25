# -*- coding: utf-8 -*-

"""
Two Layer Network: Numpy
--------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x using Euclidean error.

This implementation uses numpy to manually compute the forward pass, loss, and
backward pass.

A numpy array is a generic n-dimensional array; it does not know anything about
deep learning or gradients or computational graphs, and is just a way to perform
generic numeric computations.
"""

import numpy as np

def createDataset(N,D_in,D_out):
  x = np.random.randn(N,D_in)
  y = np.random.randn(N,D_out)
  return x,y

def initializeWeights():
  # randn(x,y) creates a 2D matrix of size x*y
  # Here, w1 is the weights to the first layer and w2 is to the output layer
  w1 = np.random.randn(D_in,H)
  w2 = np.random.randn(H,D_out)
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
    a1 = x.dot(w1)
    z1 = np.maximum(a1,0)
    y_pred = z1.dot(w2)

    # Compute and print loss
    loss = 0.5*np.square(y_pred-y).sum()
    print(t,loss)

    # Backpropagate to compute gradients and update weights
    grad_y_pred = y_pred-y
    grad_w2 = z1.T.dot(grad_y_pred)
    grad_z1 = grad_y_pred.dot(w2.T)
    grad_a1 = grad_z1.copy()
    grad_a1[a1 < 0] = 0
    grad_w1 = x.T.dot(grad_a1)
    # Update weights
    w1 = w1- lr * grad_w1
    w2 = w2- lr * grad_w2
