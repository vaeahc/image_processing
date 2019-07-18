#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:53:00 2019

@author: vaeahc
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('lena512color.tiff', 0)
print('原图')
plt.imshow(im, 'gray')
plt.show()

Gaussian = np.zeros((3, 3))
sigma = 0.5
for i in range(3):
    for j in range(3):
        Gaussian[i][j] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((i - 1) ** 2 + (j - 1) ** 2) / (2 * sigma ** 2))
Gaussian /= np.sum(Gaussian)
        
#padding处理        
row = np.zeros((1, im.shape[1]))
im = np.row_stack((row, im, row))
col = np.zeros((im.shape[0], 1))
im = np.column_stack((col, im, col))

#高斯低通滤波
for i in range(1, im.shape[0] - 1):
    for j in range(1, im.shape[1] - 1):
        im[i][j] = np.sum(Gaussian * im[i - 1: i + 2, j - 1: j + 2])
print('高斯低通滤波')
plt.imshow(im[1:-1, 1:-1], 'gray')
plt.show()

#提取水平和竖直方向梯度，幅度和夹角
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
grad_x = np.zeros(np.shape(im))
grad_y = np.zeros(np.shape(im))
Amp = np.zeros(np.shape(im))
dire = np.zeros(np.shape(im))
for i in range(1, im.shape[0] - 1):
    for j in range(1, im.shape[1] - 1):
        grad_x[i][j] = np.sum(sobel_x * im[i - 1: i + 2, j - 1: j + 2])
        grad_y[i][j] = np.sum(sobel_y * im[i - 1: i + 2, j - 1: j + 2])
Amp = np.sqrt(grad_x ** 2 + grad_y ** 2)
dire = np.arctan(grad_y / grad_x) + np.pi / 2
print('梯度图像')
plt.imshow(Amp[1:-1, 1:-1], 'gray')
plt.show()

#NMS(非极大值抑制), 非插值方法
for i in range(1, Amp.shape[0] - 1):
    for j in range(1, Amp.shape[1] - 1):
        angel = np.abs(dire[i][j])
        if angel < np.pi / 8 or angel >= np.pi * 7 / 8:
            if Amp[i][j] < Amp[i][j - 1] or Amp[i][j] < Amp[i][j + 1]:
                Amp[i][j] = 0           
                
        elif angel >= np.pi / 8 and angel < np.pi * 3 / 8:
            if Amp[i][j] < Amp[i + 1][j - 1] or Amp[i][j] < Amp[i - 1][j + 1]:
                Amp[i][j] = 0
            
        elif angel >= np.pi * 3 /8 and angel < np.pi * 5 /8:
            if Amp[i][j] < Amp[i - 1][j] or Amp[i][j] < Amp[i + 1][j]:
                Amp[i][j] = 0
            
        elif angel >= np.pi * 5 / 8 and angel < np.pi * 7 / 8:
            if Amp[i][j] < Amp[i - 1][j - 1] or Amp[i][j] < Amp[i + 1][j + 1]:
                Amp[i][j] = 0
print('非极大值抑制后梯度图像')
plt.imshow(Amp[1:-1, 1:-1], 'gray')
plt.show()

#双阈值处理，取 B=100, A = 60
A = 100
B = 150
for i in range(1, Amp.shape[0] - 1):
    for j in range(1, Amp.shape[1] - 1):
        if Amp[i][j] >= B:
            Amp[i][j] = 255
        elif Amp[i][j] <= A:
            Amp[i][j] = 0
for i in range(1, Amp.shape[0] - 1):
    for j in range(1, Amp.shape[1] - 1):
        if Amp[i][j] > A and Amp[i][j] < B:
            if Amp[i - 1][j - 1] == 255 or Amp[i - 1][j] == 255 or Amp[i - 1][j + 1] == 255 or Amp[i][j - 1] == 255 or Amp[i][j + 1] == 255 or Amp[i + 1][j - 1] == 255 or Amp[i + 1][j] == 255 or Amp[i + 1][j + 1] == 255:
                Amp[i][j] == 255
            else:
                Amp[i][j] == 0
print('双阈值后图像')
plt.imshow(Amp[1:-1, 1:-1], 'gray')
plt.show()

            
            

        
        

