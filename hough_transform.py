#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:59:32 2019

@author: vaeahc
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
src = cv2.imread('IMG_0381.tif', 0)
'''
img1 = cv2.GaussianBlur(src,(3,3),0)
edges1 = cv2.Canny(img1, 50, 150, apertureSize = 3)
lines1 = cv2.HoughLines(edges1,1,np.pi/180,118)
'''
height, width = np.shape(src)
hough_space = 500
hough_interval = np.pi / hough_space
max_length = int(width + height) + 1
hough_area = np.zeros((hough_space, 2 * max_length))
threshold = 118
res = []

edge = cv2.Canny(src, 50, 150)

for i in range(height):
    for j in range(width):
        if edge[i][j] == 0:
            continue
        for k in range(hough_space):
            r = int(j * np.cos(k * hough_interval) + i * np.sin(k * hough_interval))
            r += max_length
            if r >= 0 and r < 2 * max_length:
                hough_area[k][r] += 1

for row in range(hough_space):
    for col in range(2 * max_length):
        if hough_area[row][col] < threshold:
            continue
        hough_value = hough_area[row][col]
        isline = True
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i != 0 or j != 0:
                    y = row + i
                    x = col + j
                    if x < 0:
                        continue
                    if x < 2 * max_length:
                        if y < 0:
                            y += hough_space
                        elif y >= hough_space:
                            y -= hough_space
                    if hough_area[y][x] < hough_value:
                        continue
                    isline = False
                    break
        if isline:
            res.append([row, col])
            
result = src.copy()

for line in res:
    rho = line[1] - max_length
    theta = line[0] * hough_interval
    if theta < np.pi / 4 or theta > np.pi * 3 / 4:
        pt1 = (int(rho / np.cos(theta)), 0)
        pt2 = (int((rho - height * np.sin(theta)) / np.cos(theta)), height)
        cv2.line(edge, pt1, pt2, (255))
    else:
        pt1 = (0, int(rho / np.sin(theta)))
        pt2 = (width, int((rho - width * np.cos(theta)) / np.sin(theta)))
        cv2.line(edge, pt1, pt2, (255), 1)
    
plt.imshow(edge, 'gray')
plt.show()
        


        





            
            
    



