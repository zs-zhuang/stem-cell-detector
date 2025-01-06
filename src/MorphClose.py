#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc import imshow
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image

from PIL import Image, ImageOps
import cv2
###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################

in_arg = sys.argv[1]
im = cv2.imread(in_arg+'.jpeg', 0)

r = 5
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
close = cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel1)

#perform contrast stretching to enhace contrast (replaces log transformation)
d_max = close.max() #find max pixel value
d_min = close.min() #find min pixel value
d2 = close.astype(float) #convert c to type float
d3 = 255*(d2-d_min)/(d_max-d_min) #contrast stretching transformation


b = scipy.misc.toimage(d3)
b.save('MorphClose'+str(r)+'_'+in_arg+'.jpeg')
