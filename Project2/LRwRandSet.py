#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:26:26 2019

@author: Yssubhi
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import P2mastercode as p2mc

# create random dataset

X = torch.randn(100, 1)*10
y = p2mc.y
plt.plot(X.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')

# create new class
# class is used as a blueprint for our construction

class LR(nn.Module):
    # self represents the instance of the class
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    # define forward method
    def forward(self, x):
        pred = self.linear(x)
        return pred


torch.manual_seed(1)

model = LR(1, 1)
print(list(model.parameters()))

x = torch.zeros([216, 1])
model.forward(x)

print(model.forward(x))

[w, b] = model.parameters()
print(w, b)
# w1 = w[0][0].item
# b1 = b[0].item
# print(w1, b1)

# putting it into a function

def get_params():
    return (w[0][0].item(), b[0].item())

def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array([-30, 30])
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X,y)
    plt.show()

plot_fit('Initial Model')


# torch mean ** loss nn.MSELoss()
criterion = nn.MSELoss()
# torch stocastic gradient decent, lr learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = 0.000000001)

# train model

epochs = 10000
losses = []

for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print("epoch:", i, "loss:", loss.item())
    
    losses.append(loss)
    optimizer.zero_grad
    loss.backward()
    optimizer.step()
    
plt.plot(range(epochs), losses)
plt.xlabel('epochs')
plt.ylabel('losses')
plot_fit('loss data')

# back to plot_fit
plot_fit('Trained Model')





























