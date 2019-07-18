#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:05:59 2019

@author: vaeahc
"""
import numpy as np
import matplotlib.pyplot as plt

sigmax = 1
sigmay = 2
rou = -0.4 #0.4, 0.9, 0, -0.4, -0.9
sigma = np.array([[sigmax ** 2, rou * sigmax * sigmay], [rou * sigmax * sigmay, sigmay ** 2]])
sigma_det = np.abs(sigmax ** 2 * sigmay ** 2 - rou ** 2 * sigmax ** 2 * sigmay ** 2)
sigma_inv = 1 / sigma_det * np.array([[sigmay ** 2, -rou * sigmax * sigmay], [-rou * sigmax * sigmay, sigmax ** 2]])
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
g = 1 / (2 * np.pi * np.sqrt(1 - rou ** 2) * sigmax * sigmay) * np.exp(-(x ** 2 * sigmax **2 + y ** 2 * sigmay ** 2 - 2 * rou * x * y * sigmax * sigmay) / (2 * (1 - rou ** 2) * sigmax ** 2 * sigmay ** 2))
plt.contourf(x, y, g)
plt.contour(x, y, g)
plt.show()