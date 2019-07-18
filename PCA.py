#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:42:39 2019

@author: vaeahc
"""
import matplotlib.pyplot as plt
import numpy as np

x1 = [-10.0, -9.0, -8.0, -6.9, -6.2, -4.9, -4.0, -3.2, -2.1, -1.0, 0.0, 1.1, 2.1, 2.9, 4.3, 5.0, 6.2]
x2 = [-9.8, -9.1, -8.2, -7.0, -6.0, -5.0, -4.2, -3.0, -2.0, -1.1, 0.2, 1.0, 2.2, 3.1, 4.1, 4.9, 6.0]
plt.scatter(x1, x2)
plt.show()
x1 = np.resize(np.array(x1), (-1, 1))
x2 = np.resize(np.array(x2), (-1, 1))
X = np.concatenate((np.array(x1), np.array(x2)), axis = 1)
X_mean = np.mean(X, axis = 0)
X_temp = X - X_mean
C = np.dot(X_temp.T, X_temp) / (np.shape(X_temp)[0] - 1)
u, s, v = np.linalg.svd(C)
X_res = np.dot(X, u[:, 0:1])
x = np.linspace(-11, 8, 16)
y = X_res
plt.plot(x, y)
plt.show()