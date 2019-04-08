#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:34:59 2019

@author: Yssubhi
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import mpl_toolkits.mplot3d


PCA1 = [-4.71962709e-02, 4.59202582e-01, -4.31519119e-01, 3.84723737e-02, -4.65269404e-01, 6.46903792e-03, 6.18464093e-01, -1.41888971e-02, 5.61133024e-17]
PCA2 = [-1.16129342e-01, -4.44649368e-01, -4.68385310e-01, 2.31340881e-01, 8.89921156e-02, 7.11438659e-01, 3.92013064e-02, -1.72107337e-02, -1.02028148e-16]
PCA3 = [4.24915095e-02, -4.37506308e-01, 7.90993432e-02, -2.34179173e-01, -8.32522395e-01, -2.16251500e-02, -2.28294439e-01, -2.56419642e-03, -4.53046882e-17]


plt.figure(2)
plt.scatter(PCA1,PCA2)


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(PCA1, PCA2, PCA3)

plt.show()

X = PCA1
y = PCA2

# linear regression on PCA1/PCA2
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


model = LR(1, 1)
print(list(model.parameters()))

x = torch.tensor([[1.0], [2.0]])
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
    x1 = np.array([-1, 1])
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

