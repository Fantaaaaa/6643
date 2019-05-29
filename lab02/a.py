#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = np.uint8(mpimg.imread('2.png')*255.0)


def correlation(kernel, image):
    height, width = image.shape
    output = np.float32(np.zeros_like(image))
    kSize, _ = kernel.shape
    edge = int(kSize / 2)
    # edge remains the same
    output[0:edge, 0:height] = image[0:edge, 0:height]
    output[0:width, 0:edge] = image[0:width, 0:edge]
    output[width - edge:width, 0:height] = image[width - edge:width, 0:height]
    output[0:width, height - edge:height] = image[0:width, height - edge:height]
    # correlation 
    for x in range(edge, height - edge):
        for y in range(edge, width - edge):
            output[y, x] = (kernel * image[y - edge:y + edge + 1, x - edge:x + edge + 1]).sum()
    return output

def gaussian(kernel, image):
    height, width = image.shape
    output = np.float32(np.zeros_like(image))
    kSize = kernel.size
    edge = int(kSize / 2)
    # edge remains the same
    output[0:edge, 0:height] = image[0:edge, 0:height]
    output[0:width, 0:edge] = image[0:width, 0:edge]
    output[width - edge:width, 0:height] = image[width - edge:width, 0:height]
    output[0:width, height - edge:height] = image[0:width, height - edge:height]
    # first horizontal, then vertical
    for x in range(edge, height - edge):
        for y in range(edge, width - edge):
            output[y, x] = (kernel * (np.transpose(kernel) * image[y - edge:y + edge + 1, x - edge:x + edge + 1])).sum()
    return output

def thinedge(image):
    thin_edge = image.copy()
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            if image[i][j]==1:
                edge = (image[i-1:i+2,j-1:j+2]).sum()
                if edge<9:
                    thin_edge[i][j]=255
    return thin_edge
    

# box filter
kernel_3  = np.ones((3,3),np.float32)/9
kernel_5  = np.ones((5,5),np.float32)/25
kernel_9 = np.ones((9,9),np.float32)/81
plt.figure(8)
plt.title('original image')
plt.imshow(img,cmap='gray')
plt.figure(0)
plt.title('3*3 box filter')
a1 = correlation (kernel_3,img)
plt.imshow(np.uint8(correlation (kernel_3,img)),cmap='gray')
plt.figure(1)
plt.title('5*5 box filter')
plt.imshow(np.uint8(correlation (kernel_5,img)),cmap='gray')
plt.figure(2)
plt.title('9*9 box filter')
plt.imshow(np.uint8(correlation (kernel_9,img)),cmap='gray')

# gaussian filter
kernel_gaussian = np.array([[0.03, 0.07, 0.12, 0.18, 0.20, 0.18, 0.12, 0.07, 0.03]])
plt.figure(3)
plt.title('gaussian filter')
plt.imshow(np.uint8(gaussian (kernel_gaussian, img)),cmap='gray')

# Laplacian of Gaussian
smooth_image = gaussian (kernel_gaussian, img)
kernel_laplacian = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
LoG = correlation(kernel_laplacian,smooth_image)
plt.figure(4)
plt.title('laplacian of gaussian')
plt.imshow(LoG, cmap='gray')

# scale to [0-128]
plt.figure(5)
plt.title('LoG scale to 0 - 128')
scale_0128 = LoG*(128/(LoG.max()-LoG.min()))
scale_0128 += (0-scale_0128.min())
plt.imshow(scale_0128, cmap='gray')


#scale to [0-1]
plt.figure(6)
plt.title('LoG scale to 0 - 1')
scaled_01 = np.where(LoG < 0, 0, 1)
plt.imshow(scaled_01, cmap='gray')

#thin edge
plt.figure(7)
plt.title('Thin edge')
plt.imshow(thinedge(scaled_01), cmap='gray')
