#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc import imshow
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image

from PIL import Image, ImageOps

###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################
in_arg = sys.argv[1]

# open the image and convert to grayscale
name = ""+in_arg+".jpeg" 
a = Image.open(name).convert('L')
a2 = np.copy(a)
a3 = scipy.misc.toimage(a2)
######################################################################################
#perform gaussian low pass filter, which blur high frequency variation and only keep low frequency variation in images
l = scipy.misc.fromimage(a)
l2 = fftim.fft2(l)
l3 = fftim.fftshift(l2)
M = l3.shape[0]
N = l3.shape[1]
H_low = np.ones((M,N))
H_high = np.ones((M,N))
center1 = M/2
center2 = N/2
l3_0 = 30.0 #cutoff radius for low pass filter
l4_0 = 10.0 #cutoff radius for high pass filter
cutoff_low = int(l3_0)
cutoff_high = int(l4_0)
t1 = 2*l3_0
t2 = 2*l4_0

#Gaussian Low Pass Filter
for i in range (1, M):
	for j in range (1,N):
		r1 = (i-center1)**2+(j-center2)**2
		r = math.sqrt(r1)
		if r > l3_0:
			H_low[i,j] = math.exp(-r**2/t1**2)

H_low = scipy.misc.toimage(H_low)
con_low = l3*H_low
l4 = abs(fftim.ifft2(con_low))
l5 = scipy.misc.toimage(l4)


# Gaussian High Pass Filter 
for i in range (1, M):
        for j in range (1,N):
                r3 = (i-center1)**2+(j-center2)**2
                r = math.sqrt(r3)
                if 0 < r < l4_0:
                        H_high[i,j] = 1 - math.exp(-r**2/t2**2) 

H_high = scipy.misc.toimage(H_high)
con_high = l3*H_high
l6 = abs(fftim.ifft2(con_high))
l7 = scipy.misc.toimage(l6)



#permform mean filter (blur image, reduce resolution which could mean reduce noise)
#initialize mean filter size 2 by 2 (higher filter size causes more blur image)
#k = np.ones((1,1))/1

# perform convolution
#b = scipy.ndimage.filters.convolve(a,k)

#perform median filter (get rid of specks or salt pepper noise)
#c = scipy.ndimage.filters.median_filter(b,size=2,footprint=None,output=None,mode='reflect',cval=0.0,origin=0)

#perform log transformation to enhance contrast (only use this if contrast is so low that naked eyes basically cannot see anything)
#d = c.astype(float) #convert c to type float
#d_max = np.max(cd) #get max value of d 
#d3 = (255.0*np.log(1+d))/np.log(1+d_max) #perform log transofrmation
#d4 = d3.astype(int) # d3 is converted to type int

#perform contrast stretching to enhace contrast (replaces log transformation)
#d_max = d.max() #find max pixel value
#d_min = d.min() #find min pixel value
#d2 = d.astype(float) #convert c to type float
#d4 = 255*(d2-d_min)/(d_max-d_min) #contrast stretching transformation

#########################################################################################
#convert from an ndarray to an image
#print(c4.shape)
#print(c4.size)

print(a2.shape, l4.shape, l6.shape)
a3.save('fix_'+str(in_arg)+'.jpeg')
l5.save('lowpass'+str(cutoff_low)+'_'+str(in_arg)+'.jpeg')
l7.save('highpass'+str(cutoff_high)+'_'+str(in_arg)+'.jpeg')
