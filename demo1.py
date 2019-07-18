#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:54:22 2019

@author: vaea
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def cmp(trg):
    res = []
    for i in range(len(trg)):
        label = 1
        if not res:
            res.append(trg[i])
        for j in range(len(res)):
            if abs(res[j][0] - trg[i][0]) <= 0.1 and abs(res[j][1] - trg[i][1] <= 15):
                label = 0
                break
        if label:
            res.append(trg[i])
    return res

def cal_point(r1, r2, theta1, theta2):
    return np.dot(np.linalg.inv(np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])), np.array([[r1], [r2]]))

def get_lines(edge, threshold):
    height, width = np.shape(edge)
    hough_space = 500
    hough_interval = np.pi / hough_space
    max_length = int(width + height) + 1
    hough_area = np.zeros((hough_space, 2 * max_length))
    res = []

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
    hor = []
    vert = []
    for line in res:
        rho = line[1] - max_length
        theta = line[0] * hough_interval
        if theta < np.pi / 4 or theta > np.pi * 3 / 4:
            hor.append([theta, rho])
        else:
            vert.append([theta, rho])
    
    
    return hor, vert

def get_perspection(row, col, mat_trans):
    trg = np.zeros([row, col])
    for i in range(row):
        for j in range(col):
            temp_mat = np.dot(np.linalg.inv(mat_trans), np.array([j, i, 1]))
            temp_x = temp_mat[0] / temp_mat[2]
            temp_y = temp_mat[1] / temp_mat[2]
            if temp_x < 0 or temp_x > col -1 or temp_y < 0 or temp_y > row - 1:
                continue
            f_up = (math.ceil(temp_x) - temp_x) * img[int(temp_y), int(temp_x)] + (temp_x - int(temp_x)) * img[math.ceil(temp_y), int(temp_x)]
            f_down = (math.ceil(temp_x) - temp_x) * img[int(temp_y), math.ceil(temp_x)] + (temp_x - int(temp_x)) * img[math.ceil(temp_y), math.ceil(temp_x)]
            f_center = (math.ceil(temp_y) - temp_y) * f_up + (temp_y - int(temp_y)) * f_down
            trg[i, j] = f_center
    return trg
            
    
img = cv2.imread('IMG_2878.tif', 0)
plt.imshow(img, 'gray')
plt.show()
img = cv2.GaussianBlur(img,(3,3),0)
edges = cv2.Canny(img, 50, 150, apertureSize = 3)
#threshold = 90
hor, vert = get_lines(edges, 128)
#lines = cv2.HoughLines(edges,1,np.pi/180,90) 

result = img.copy()
hor = cmp(hor)
vert = cmp(vert)

points = []
for array1 in vert:
    for array2 in hor:
        points.append(cal_point(array1[1], array2[1], array1[0], array2[0]))
pst1 = []
pst2 = np.float32([[0, 0], [np.shape(img)[1], 0], [0, np.shape(img)[0]], [np.shape(img)[1], np.shape(img)[0]]])
points = sorted(points, key = lambda x: x[1])
points = sorted(points[:2], key = lambda x: x[0]) + sorted(points[2:], key = lambda x: x[0])
for point in points:
    cv2.circle(result, (point[0], point[1]), 5, (255, 0, 0), 10)
    pst1.append([point[0], point[1]])

perspective_trans = cv2.getPerspectiveTransform(np.float32(pst1), pst2)
dst = get_perspection(np.shape(img)[0], np.shape(img)[1], perspective_trans)

#dst = cv2.warpPerspective(img, perspective_trans, (np.shape(img)[1], np.shape(img)[0]))


plt.imshow(result, 'gray')
plt.show()
plt.imshow(dst, 'gray')
plt.show()
