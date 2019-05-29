#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pylab as plt
import matplotlib.image as mpimg


def gray(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    g_img = np.array(r*0.3) + np.array(g*0.59) + np.array(b*0.11)
    return np.uint8(g_img)

def correlation (kernel,image):
    height,width = image.shape
    output = np.float16(np.zeros_like(image))
    hSize,wSize = kernel.shape
    hedge = int(hSize/2)
    wedge = int(wSize/2)

    #normalization
    nor = (np.array(kernel).flatten()).sum()
    kernel = np.float16(kernel/nor)
    
    for x in range (height):
        for y in range (width):
            shape = image[x:x+hSize,y:y+wSize].shape
            if (shape==kernel.shape):
                output[x+hedge,y+wedge] = np.array((kernel*image[x:x+hSize,y:y+wSize]).sum(),np.float32)    
    return output
    
template = np.array(mpimg.imread('template.png')*255.0)
img = np.array(mpimg.imread('input.png')*255.0)

#grayscale
img = gray(img)
template = gray(template)

#autocorrelation without preprocess
plt.figure(0)
plt.title('autocorrelation without preprocess')
plt.imshow(np.uint8(correlation(template,img)),cmap='gray')


#binary image
img = np.where(img>220,0,1)
template = np.where(template>185,0,1)
plt.figure(1)
plt.title('binary image')
plt.imshow(img,cmap='gray')
plt.figure(2)
plt.title('binary template')
plt.imshow(template,cmap='gray')

#modify template
template = np.where(template==0,-1,1)

#correlation
matching = correlation(template,img)

#scale to 0-255
scale = matching*(256/(matching.max()-matching.min()))
scale += (0-scale.min())
plt.figure(3)
plt.title('template matching')
plt.imshow(np.uint8(scale),cmap='gray')

#find peak value
peak_value = np.unravel_index(np.argmax(matching, axis=None), matching.shape)
print (peak_value)
