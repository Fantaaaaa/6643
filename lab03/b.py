#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:29:21 2019

@author: fantaaaa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv

def onclick(event):
    global a,A,result_img,n,coo_t,coo_s
    point = [int(event.xdata), int(event.ydata)]
    if len(coo_s)<n: 
        coo_s.append(point)
        print ('the source image choosen point: (' ,point[0],',',point[1],')')
    elif len(coo_t)<n:
        coo_t.append(point)
        print ('the target image choosen point: (' ,point[0],',',point[1],')')
      
    #calculate a vector
    if (len(coo_s) == n and len(coo_t)==n):
        X = []
        x = []
        lm_s = np.array(np.asarray(coo_s)) 
        lm_t = np.array(np.asarray(coo_t))
        for i in range(n):
            X.append([lm_s[i][0],lm_s[i][1],1,0,0,0])
            X.append([0,0,0,lm_s[i][0],lm_s[i][1],1])
            x.append([lm_t[i][0]])
            x.append([lm_t[i][1]])
            
        #SVD 
        X_inv = np.linalg.pinv(X)
        a = np.dot(X_inv,x)
        A = np.array([[float(a[0]), float(a[1]), float(a[2])], 
                      [float(a[3]), float(a[4]), float(a[5])], 
                      [0, 0, 1]])
        A_inv = inv(A)
        
        #apply affine using nearest neighbor
        for i in range(height):
            for j in range(width):
                result = ([i,j,1])
                source = np.round(np.dot(A_inv,result))
                m,n,_ = np.uint16(source)
                if (0<n<width and 0<m<height):
                    result_img[j,i] = source_img[n,m]            

        fig1 = plt.figure(figsize=[9,3])
        gs1 = fig1.add_gridspec(1, 3)
        ax3 = fig1.add_subplot(gs1[0, 0])
        ax3.set_title('result img')
        ax3.imshow(result_img,cmap='gray')
        
        #combine resource image and result image
        for i in range(height):
            for j in range(width):
                combine_img[i,j]  = 0.5*result_img[i,j] + 0.5*source_img[i,j]    
                combine_img1[i,j] = 0.5*result_img[i,j] + 0.5*target_img[i,j]

        ax4 = fig1.add_subplot(gs1[0, 1])
        ax4.set_title('resource_img+result_img')
        ax4.imshow(combine_img,cmap='gray')
        
        ax5 = fig1.add_subplot(gs1[0,2])
        ax5.set_title('result_img+target_img')
        ax5.imshow(combine_img1,cmap='gray')
  
print('enter the number of pairs of landmarks')
n = int(input())
print('First select landmarks on the left image, then right image')
      
coo_s = []
coo_t = []
point = []
a = []
A = []
  
source_img = np.int8(mpimg.imread('source.jpg'))
source_img = np.uint8(np.array(source_img[:,:,0]))
target_img = np.int8(mpimg.imread('target.jpg'))
target_img = np.uint8(np.array(target_img[:,:,0]))
result_img = np.uint8(np.zeros(source_img.shape))
combine_img = np.uint8(np.zeros(source_img.shape))
combine_img1 = np.uint8(np.zeros(source_img.shape))
height,width = result_img.shape

source = np.array([0,0,1])
result = np.array([0,0,1])

fig = plt.figure(figsize=[6,3])
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('source img')
ax1.imshow(source_img,cmap='gray')
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('target img')
ax2.imshow(target_img,cmap='gray')
cid = fig.canvas.mpl_connect('button_press_event', onclick)