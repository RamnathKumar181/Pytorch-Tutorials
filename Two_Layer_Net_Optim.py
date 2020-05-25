# -*- coding: utf-8 -*-
"""
PyTorch: optim
--------------

A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.

Rather than manually updating the weights of the model as we have been doing,
we use the optim package to define an Optimizer that will update the weights
for us. The optim package defines many optimization algorithms that are commonly
used for deep learning, including SGD+momentum, RMSProp, Adam, etc.
"""

import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

def createDataset(N,D_in,D_out):
  x = torch.randn(N, D_in, device=device, dtype=dtype)
  y = torch.randn(N, D_out, device=device, dtype=dtype)
  return x,y

def Network():        
  # Use the nn package to define our model as a sequence of layers. nn.Sequential
  # is a Module which contains other Modules, and applies them in sequence to
  # produce its output. Each Linear Module computes output from input using a
  # linear function, and holds internal Tensors for its weight and bias.
  model = torch.nn.Sequential(
      torch.nn.Linear(D_in, H),
      torch.nn.ReLU(),
      torch.nn.Linear(H, D_out)
      )
        
  return model

if __name__=="__main__":

  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  N, D_in, H, D_out = 64, 1000, 100, 10

  # Create random input and output data
  x,y = createDataset(N,D_in,D_out)

  # Initialize other hyperparameters
  lr = 1e-4
  epochs = 500
  model = Network()
  loss_fn = torch.nn.MSELoss(reduction='sum')
  optimizer = torch.optim.Adam(model.parameters(), lr)

  for t in range(epochs):

    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
