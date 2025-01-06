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

#########################################################################################

#imgA = cv2.imread('highpass10_stemcell11.jpeg', 0)
imgA = cv2.imread('MorphClose5_stemcell11.jpeg', 0)
imgB = cv2.imread('MorphClose8_stemcell11.jpeg', 0)
imgC = cv2.imread('lowpass30_AdaptiveThreshold_stemcell11.jpeg', 0)


# Positive sample example
A_good1 = imgA[100:212, 128:242]
B_good1 = imgB[100:212, 128:242]
C_good1 = imgC[100:212, 128:242]

# Negative sample example
A_bad1 = imgA[6:87, 310:395]
B_bad1 = imgB[6:87, 310:395]
C_bad1 = imgC[6:87, 310:395]

# Differentiated Colony Sample
A_diff1 = imgA[93:155, 31:52]
B_diff1 = imgB[93:155, 31:52]
C_diff1 = imgC[93:155, 31:52]

A_diff2 = imgA[269:292, 142:234]
B_diff2 = imgB[269:292, 142:234]
C_diff2 = imgC[269:292, 142:234]

A_diff3 = imgA[113:190, 300:317]
B_diff3 = imgB[113:190, 300:317]
C_diff3 = imgC[113:190, 300:317]

A_diff4 = imgA[236:263, 100:132]
B_diff4 = imgB[236:263, 100:132]
C_diff4 = imgC[236:263, 100:132]


#Save positive images
Ag1 = scipy.misc.toimage(A_good1)
Ag1.save('A_good1.jpeg')

Bg1 = scipy.misc.toimage(B_good1)
Bg1.save('B_good1.jpeg')

Cg1 = scipy.misc.toimage(C_good1)
Cg1.save('C_good1.jpeg')

#Save negative images
Ab1 = scipy.misc.toimage(A_bad1)
Ab1.save('A_bad1.jpeg')

Bb1 = scipy.misc.toimage(B_bad1)
Bb1.save('B_bad1.jpeg')

Cb1 = scipy.misc.toimage(C_bad1)
Cb1.save('C_bad1.jpeg')


#Save differentiated image
Ad1 = scipy.misc.toimage(A_diff1)
Ad1.save('A_diff1.jpeg')
Ad2 = scipy.misc.toimage(A_diff2)
Ad2.save('A_diff2.jpeg')
Ad3 = scipy.misc.toimage(A_diff3)
Ad3.save('A_diff3.jpeg')
Ad4 = scipy.misc.toimage(A_diff4)
Ad4.save('A_diff4.jpeg')

Bd1 = scipy.misc.toimage(B_diff1)
Bd1.save('B_diff1.jpeg')
Bd2 = scipy.misc.toimage(B_diff2)
Bd2.save('B_diff2.jpeg')
Bd3 = scipy.misc.toimage(B_diff3)
Bd3.save('B_diff3.jpeg')
Bd4 = scipy.misc.toimage(B_diff4)
Bd4.save('B_diff4.jpeg')


Cd1 = scipy.misc.toimage(C_diff1)
Cd1.save('C_diff1.jpeg')
Cd2 = scipy.misc.toimage(C_diff2)
Cd2.save('C_diff2.jpeg')
Cd3 = scipy.misc.toimage(C_diff3)
Cd3.save('C_diff3.jpeg')
Cd4 = scipy.misc.toimage(C_diff4)
Cd4.save('C_diff4.jpeg')






