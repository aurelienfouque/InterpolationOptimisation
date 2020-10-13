#!/usr/local/bin/python3.7

import numpy as np
from numpy.linalg import solve as sol
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from decimal import Decimal

plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size':10})

w0 = w1 = w2 = w3 = w4 = w5 = 0.
W = np.array([w0, w1, w2, w3, w4, w5])

#X = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
X = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
z = np.array([1., 0., 0., 0.])
Xt = np.array([0., 0.])
U = np.ones(z.size)

def forward(u):
  # W[3] = 0.
  # W[4] = 0.
  # W[5] = 0.
    return W[0] + W[1]*u[0] + W[2]*u[1] + W[3]*u[0]*u[1] +  W[4]*u[0]**2 + W[5]*u[1]**2 

def loss(u, v):
    return ((u - v)**2).mean()

def grad(c0, c1, c2, c3, c4, c5, w, t):
    a = np.dot(2 * c0, w - t).mean()
    b = np.dot(2 * c1, w - t).mean()
    c = np.dot(2 * c2, w - t).mean()
    d = np.dot(2 * c3, w - t).mean()
    e = np.dot(2 * c4, w - t).mean()
    f = np.dot(2 * c5, w - t).mean()
    return np.array([a, b, c, d, e, f])


print(f'Prediction before training: \
      z({Xt}) = {forward(Xt):.3f}')

learnRate = 0.01
nIters = 1000

for epoch in range(nIters):
    zp = forward(X.T)
    L = loss(z, zp)
    dw = grad(U, X[:,0], X[:,1], X[:,0]*X[:,1], X[:,0]**2, X[:,1]**2,  zp, z)
    W -= learnRate * dw 
    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}: \
          w0 = {W[0]:.3f}, \
          w1 = {W[1]:.3f}, \
          w2 = {W[2]:.3f}, \
          w3 = {W[3]:.3f}, \
          w4 = {W[4]:.3f}, \
          w5 = {W[5]:.3f}, \
          loss = {L:.1E}')

print(f'Prediction after training: \
      z({Xt}) = {forward(Xt):.3f}')

ax = plt.axes(projection='3d')
xg, xd, yg, yd, res = -3, 3, -3, 3, 100
x1, x2 = np.meshgrid(np.linspace(xg, xd, res), np.linspace(yg, yd, res))
x = np.array([x1, x2])
ax.plot_wireframe(x1, x2,  forward(x), alpha=.3, color='grey')
ax.plot3D(X[:,0], X[:,1], z, 'o') 
ax.plot3D(Xt.reshape(1,2)[:,0], Xt.reshape(1,2)[:,1], forward(Xt.T), 'o')
plt.show()
