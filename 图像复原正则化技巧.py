#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:38:56 2019

@author: vaeahc
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('lena512color.tiff', 0).astype(np.float64)

Gaussian = np.zeros((3,3))
sigma_x = 1
sigma_y = 2
for i in range(3):
    for j in range(3):
        Gaussian[i][j] = 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-(i - 1) ** 2 / (2 * sigma_x ** 2) - j ** 2 / (2 * sigma_y ** 2))
Gaussian /= np.sum(Gaussian)

laplace = np.array([[0, 0.25, 0], [0.25, -1, 0.25], [0, 0.25, 0]])

row = np.zeros((1, im.shape[1]))
im = np.row_stack((row, im, row))
col = np.zeros((im.shape[0], 1))
im = np.column_stack((col, im, col))

for i in range(1, im.shape[0] - 1):
    for j in range(1, im.shape[1] - 1):
        im[i][j] = np.sum(Gaussian * im[i - 1 : i + 2, j - 1 : j + 2])       
im[1:-1, 1:-1] += np.random.randn(im[1:-1, 1:-1].shape[0], im[1:-1, 1:-1].shape[1]) * 5

plt.imshow(im[1:-1, 1:-1], 'gray')
plt.show()

Gaussian = np.mat(Gaussian)
laplace = np.mat(laplace)
alpha = 0.1
temp = np.linalg.inv(Gaussian.H * Gaussian + alpha * laplace.H * laplace)
Gaussian_1 = temp * Gaussian.H

for i in range(1, im.shape[0] - 1):
    for j in range(1, im.shape[1] - 1):
        im[i][j] = np.sum(Gaussian_1 * im[i-1 : i+2, j-1 : j+2])
plt.imshow(im[1:-1, 1:-1], 'gray')
plt.show()













