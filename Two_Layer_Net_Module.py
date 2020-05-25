# -*- coding: utf-8 -*-
"""
PyTorch: Custom nn Modules
--------------------------

A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""

import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

def createDataset(N,D_in,D_out):
  x = torch.randn(N, D_in, device=device, dtype=dtype)
  y = torch.randn(N, D_out, device=device, dtype=dtype)
  return x,y

class Network(torch.nn.Module):        
  def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Network, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

  def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

if __name__=="__main__":

  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  N, D_in, H, D_out = 64, 1000, 100, 10

  # Create random input and output data
  x,y = createDataset(N,D_in,D_out)

  # Initialize other hyperparameters
  lr = 1e-4
  epochs = 500
  
  # Construct our model by instantiating the class defined above
  model = Network(D_in, H, D_out)

  loss_fn = torch.nn.MSELoss(reduction='sum')
  optimizer = torch.optim.SGD(model.parameters(), lr)

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
