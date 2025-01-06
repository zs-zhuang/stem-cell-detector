#! /usr/bin/python3

import os, sys, math, string

import numpy as np,  scipy.ndimage

from scipy.misc.pilutil import Image
from skimage import io
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
import cv2
from sklearn import model_selection as ms
from sklearn.metrics import accuracy_score, jaccard_similarity_score

###from skimage.feature import corner_harris, corner_subpix, corner_peaks


#########################################################################################

in_arg = sys.argv[1]
common_name = in_arg

# Import data 
featurefile = 'train_feature_data' #multi column file that contains feature values
targetfile = 'train_target_data' #single column file that specify whether a pixel is part of a good colony 1 or bad colony -1

X = np.loadtxt(featurefile)
Y = np.loadtxt(targetfile)

X = X.astype(np.float32)
Y = Y.astype(np.float32)

#print(X.shape, X.dtype)
#print(Y.shape, Y.dtype)

#Standardize or normalize features

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#print(X_train.shape)

#set up MLP, n hidden layer
mlp = cv2.ml.ANN_MLP_create()
n_input = 15
n_hidden1 = 11
#n_hidden2 = 3
n_output = 2
mlp.setLayerSizes(np.array([n_input, n_hidden1, n_output]))
mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2.5, 1.0)
mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
term_mode = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
term_max_iter = 50000#50000
term_eps = 0.00001 #0.00000001
mlp.setTermCriteria((term_mode, term_max_iter, term_eps))

#run mlp
mlp.train(X_scale, cv2.ml.ROW_SAMPLE, Y)
_, Y_pred = mlp.predict(X_scale)
Y_pred2 = np.sign(Y_pred)
#score = accuracy_score(Y_pred2, Y)
#np.savetxt('Y_pred_training', Y_pred2, fmt='%1.0i')
#print(score)

length, _ = Y_pred2.shape
total = length
for m in range(0, length):
	if Y_pred2[m][0] != Y[m][0] or Y_pred2[m][1] != Y[m][1]:
		total = total -1
score = total/length	
print(score)

# Make Prediction
featurefile = ('feature_'+common_name) #multi column file that contains feature values
predictionfile = ('prediction_'+common_name)

test = np.loadtxt(featurefile)
test = test.astype(np.float32)
test_scale = min_max_scaler.fit_transform(test)

_, testP =mlp.predict(test_scale)
testP2 = np.sign(testP)
#testP3 = testP.astype(int)
np.savetxt(predictionfile, testP2, fmt='%1.0i')
