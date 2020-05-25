# -*- coding: utf-8 -*-
"""
Different ways of building a network in Pytorch
--------------------------------------

1. Using torch.nn.functional
2. Refactor using nn.Module
3. Refactor using nn.Linear
4. Refactor using optim
5. Refactor using Dataset
6. Refactor using DataLoader
7. Adding validation
8. Create fit() and get_data()
"""

"""
Using torch.nn.functional
--------------------------------------
"""

# Using torch.nn.functional

import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
with torch.no_grad():
    weights -= weights.grad * lr
    bias -= bias.grad * lr
    weights.grad.zero_()
    bias.grad.zero_()

"""
Refactor using nn.Module
--------------------------------------
"""

# Refactor using nn.Module

from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic()
print(loss_func(model(xb), yb))
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    model.zero_grad()

"""
Refactor using nn.Linear
--------------------------------------
"""

# Refactor using nn.Linear

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Mnist_Logistic()
print(loss_func(model(xb), yb))We'll now do a little refactoring of our own. Since we go through a similar
process twice of calculating the loss for both the training set and the
validation set, let's make that into its own function, ``loss_batch``, which
computes the loss for one batch.

We pass an optimizer in for the training set, and use it to perform
backprop.  For the validation set, we don't pass an optimizer, so the
method doesn't perform backprop.

"""
Refactor using optim
--------------------------------------
"""

# Refactor using optim

from torch import optim

opt = optim.SGD(model.parameters(), lr=lr)

# bs is batch size

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

"""
Refactor using Dataset
--------------------------------------
"""

# Refactor using Dataset

from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

# Previously, we had to do this
# xb = x_train[start_i:end_i]
# yb = y_train[start_i:end_i]

# Now, we can do this
# xb, yb = train_ds[i * bs: i * bs + bs]

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

"""
Refactor using DataLoader
--------------------------------------
"""

# Refactor using DataLoader

from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

# Previously, we had to do this
# xb = x_train[start_i:end_i]
# yb = y_train[start_i:end_i]

# Now, we can do this
# xb, yb = train_ds[i * bs: i * bs + bs]

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

"""
Adding validation
--------------------------------------
"""

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

"""
Create fit() and get_data()
----------------------------------
"""

# Define how loss is calculated and updated at each epoch

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# fit() runs the necessary operations to train our model and compute the
# training and validation losses for each epoch.

import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

# get_data returns dataloaders for the training and validation sets.

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# Now, our whole process of obtaining the data loaders and fitting the
# model can be run in 3 lines of code:s

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
