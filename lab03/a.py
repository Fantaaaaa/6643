#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:22:21 2019

@author: fantaaaa
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv


source_img = np.int8(mpimg.imread('a.jpg'))
source_img = np.uint8(np.array(source_img[:,:,0]*0.3+source_img[:,:,1]*0.3+source_img[:,:,2]*0.4))
target_nn_img = np.uint8(np.zeros(source_img.shape))
target_bi_img = np.uint8(np.zeros(source_img.shape))

fig = plt.figure(figsize=[9,3])
gs = fig.add_gridspec(1, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('source img')
ax1.imshow(source_img,cmap='gray')

translation = np.array([[1, 0, 10], [0, 1, -10], [0, 0, 1]])
rotation = np.array([[math.cos(0.1*(math.pi)),-math.sin(0.1*(math.pi)), 0], [math.sin(0.1*(math.pi)),math.cos(0.1*(math.pi)), 0], [0, 0, 1]])#18
scaling = np.array([[1.1, 0, 0], [0, 0.9, 0], [0, 0, 1]])
A = np.dot(np.dot(scaling,rotation),translation)
A_inv = inv(A)

#apply affine
height,width = source_img.shape
target = np.array([0,0,1])
source = np.array([0,0,1])
A = np.array([0,0,1])
B = np.array([0,0,1])
C = np.array([0,0,1])
D = np.array([0,0,1])

#nearest neighbor
for i in range(height):
    for j in range(width):
        target = ([i,j,1])
        source = np.round(np.dot(A_inv,target))
        m,n,_ = np.int16(source)
        if (0<m<width and 0<n<height):
            target_nn_img[i,j] = source_img[m,n]

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('NN')
ax2.imshow(target_nn_img,cmap='gray')
          
#bilinear interpolation
for i in range(height-1):
    for j in range(width-1):
        x = (scaling[0][0]*i)-int(scaling[0][0]*i)
        y = (scaling[1][1]*j)-int(scaling[1][1]*j)
        
        A = np.ceil(np.dot(A_inv,([i,j,1])))
        B = np.ceil(np.dot(A_inv,([i+1,j,1])))
        C = np.ceil(np.dot(A_inv,([i,j+1,1])))
        D = np.ceil(np.dot(A_inv,([i+1,j+1,1])))
        m1,n1,_ = np.int16(np.floor(A))
        m2,n2,_ = np.int16(np.floor(B))
        m3,n3,_ = np.int16(np.floor(C))
        m4,n4,_ = np.int16(np.floor(D))
        
        # bilinear interpolation
        if (0<m1<width and 0<n1<height and 0<m4<width and 0<n4<height and 0<m3<width and 0<n3<height and 0<m2<width and 0<n2<height):
            p1 = source_img[m1,n1]
            p2 = source_img[m2,n2]
            p3 = source_img[m3,n3]
            p4 = source_img[m4,n4]
            target_bi_img[i,j] = np.int16((p1*(1-x)*(1-y))+(p2*(x)*(1-y))+(p3*(1-x)*(y))+(p4*(x)*(y)))

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('BI')
ax3.imshow(target_bi_img,cmap='gray')