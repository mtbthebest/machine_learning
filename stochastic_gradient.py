#!usr/bin/env python
import numpy as np
n_epoch = 50
t0,t1 = 5,50

X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)
X_b = np.c_[np.ones((100, 1)), X]

eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)

def learning_schedule(t):
    return t0/(t + t1)

for epoch in range(n_epoch):
    for i in range(m):
        random_index  = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

