# -*- coding: utf-8 -*-
"""
MNIST classification
============================
Using the best approaches at each step!
"""

from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F

# Use GPU if available

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Download MNIST dataset 

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

# This dataset is in numpy array format, and has been stored using pickle,
# a python-specific format for serializing data.

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# Each image is 28 x 28, and is being stored as a flattened row of length
# 784 (=28x28). Let's take a look at one; we need to reshape it to 2d
# first.

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

# PyTorch uses torch.tensor, rather than numpy arrays, so we need to
# convert our data.

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())


# We pass an optimizer in for the training set, and use it to perform
# backprop.  For the validation set, we don't pass an optimizer, so the
# method doesn't perform backprop.

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# fit runs the necessary operations to train our model and compute the
# training and validation losses for each epoch.


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

class Mnist_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


lr = 0.1
epochs = 10
bs = 64
loss_func = F.cross_entropy

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

# Wrapping the dataloader to the input

train_ds = TensorDataset(x_train.to(device), y_train.to(device))
valid_ds = TensorDataset(x_valid.to(device), y_valid.to(device))
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

# Now, our whole process of obtaining the data loaders and fitting the
# model can be run in 3 lines of code:

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model = Mnist_CNN()
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model.to(device), loss_func, opt, train_dl, valid_dl)
