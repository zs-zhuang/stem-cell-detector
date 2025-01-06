#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc.pilutil import Image
from skimage import io

from PIL import Image, ImageOps

###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################
#Open image and get basic info

in_arg = sys.argv[1] #example bad1, bad2, bad3, good1, good2
r = int(sys.argv[2]) #neighborhood size is (2r+1)^2
t1 = sys.argv[3] #target value 1 for good colony pixel, -1 for bad colony pixel
t2 = sys.argv[4]

name1 = "A_"+in_arg+".jpeg"
name2 = "B_"+in_arg+".jpeg"
name3 = "C_"+in_arg+".jpeg"
print(in_arg, r, t1, t2)


out_feature = "feature_"+in_arg
out_target = "target_"+in_arg
#out_position = "position_"+name
out_file1 = open (out_feature, 'w')
out_file2 = open(out_target, 'w')
#out_file3 = open(out_position, 'w')

#filename1 = str(in_arg)
#print(filename)


a1 = io.imread(name1)
a2 = io.imread(name2)
a3 = io.imread(name3)
print(a1.shape, a2.shape, a3.shape)
#print(a1.dtype, a2.dtype, a3.dtype)
#print(a.mean())
#print(a.max())
#print(a.min())

#a_mean = a.mean()
nrows, ncols = a1.shape
#print(nrows)
#print(ncols)

"""
if nrows <= ncols:
	r = int(nrows/200)
else:
	r = int(ncols/200)
"""
area = (2*r+1)**2

#print (a[0:3,0:2])
#########################################################################################

for x in range (r, nrows+1-r):
#for x in range (200, 201):
	for y in range(r, ncols+1-r):
	#for y in range(300, 301):
		Ixy1 = a1[x, y]
		Ixy2 = a2[x, y]
		Ixy3 = a3[x, y]
		#print(str(x)+' '+str(y)+' '+str(Ixy))
		lowy = int(y-r)
		highy = int(y+r+1)
		lowx = int(x-r)
		highx = int(x+r+1)
		#print (a[lowx:highx,lowy:highy])
		b1 = a1[lowx:highx,lowy:highy].copy()
		b2 = a2[lowx:highx,lowy:highy].copy()
		b3 = a3[lowx:highx,lowy:highy].copy()

		Imean1 = np.mean(b1)
		Imean2 = np.mean(b2)
		Imean3 = np.mean(b3)

		Istd1 = np.std(b1)
		Istd2 = np.std(b2)
		Istd3 = np.std(b3)

		nblack1 = (b1 < 50).sum()
		nwhite1 = (b1 > 200).sum()
		frac_black1 = nblack1/area
		frac_white1 = nwhite1/area

		nblack2 = (b2 < 50).sum()
		nwhite2 = (b2 > 200).sum()
		frac_black2 = nblack2/area
		frac_white2 = nwhite2/area

		nblack3 = (b3 < 50).sum()
		nwhite3 = (b3 > 200).sum()
		frac_black3 = nblack3/area
		frac_white3 = nwhite3/area
		

		#print(x,y,Ixy,Imean,Istd,nblack,nwhite,frac_black,frac_white)
		#print(np.mean(a2))
		#print(np.std(a2))
		#print((a2 < 50).sum())
		#print((a2 > 200).sum())
		#print(np.mean(a2, axis=(0,1))) #same as the command above without specifiying axis

		#write the following column to a file
		# position x, position y, Ixy, Imean, Istd, frac_black, frac_white
		#out_file1.write(str(Ixy1)+' '+str(Imean1)+' '+str(Istd1)+' '+str(Ixy2)+' '+str(Imean2)+' '+str(Istd2)+' '+str(Ixy3)+' '+str(Imean3)+' '+str(Istd3)+' '+'\n')
		out_file1.write(str(Ixy1)+' '+str(Imean1)+' '+str(Istd1)+' '+str(frac_black1)+' '+str(frac_white1)+' '+str(Ixy2)+' '+str(Imean2)+' '+str(Istd2)+' '+str(frac_black2)+' '+str(frac_white2)+' '+str(Ixy3)+' '+str(Imean3)+' '+str(Istd3)+' '+str(frac_black3)+' '+str(frac_white3)+'\n')
		out_file2.write(str(t1)+' '+str(t2)+'\n')
		#out_file3.write(str(x)+' '+str(y)+'\n')
