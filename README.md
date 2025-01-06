stem-cell-detector
===========

# Summary

This repository contains scripts that segment images of stem cell colonies on petri dishes (see example original and result images).

# Detailed Instruction

To Create data to train SVM

1. take the original image and run a highpass and lowpass filter. First use clean_image.py, then use prep_frequency_filter_image.py, this will write out three new images, fix_stemcell5.jpeg,  highpass_stemcell5.jpeg  lowpass_stemcell5.jpeg


2. take the above three images, run crop_image.py to crop out a few areas that I know for sure are good or bad, save each cropped image as a good sample or bad sample image

3. run extract_features_training.py filename r value (where total neighborhood size is square 2r+1) 1(if good sample, -1 if bad sample) (also see bash_prep file)

For example:
 ./extract_features_training.py Stemcell1_bad_sample2.jpg 2 -1
 ./extract_features_training.py Stemcell1_good_sample2.jpg 2 1

4. the above program will write out two files for each sample/training image
a feature file that contains columns of data that tells the feature of a given colony
a target file that is the same length as feature file that contains target tag for SVM, 1 or -1

5. Cat all the feature and target file together to make a final training data set

6. run train_SVM_OpenCV.py, try LINEAR, INTER, SIGMOID, RBF and decide which kernel to use, keep default parameters
(can use a scikit version of the program that does the same thing, but it's much slower)

6b. run gridsearch_param_OrderMagnitude_OpenCV.py and gridsearch_param_FineTuning_OpenCV.py to get the optimal parameters to use for RBF kernel if that's the kernel of choice (write a different program for other kernel when necessary), scikit-learn based program, extremely slow, might look for alternative ways to do this in OpenCV -- this may or may not be a good idea as parameter search can force SVM to become very specialized in predicting training data and lead to poor accuracy for new data

7. repeat feature extraction for real image data and write out a feature file and position file. Position file contains pixel coordinates which are used later to visualize which pixel is predicted to be inside a good colony 

8. run_SVM_OpenCV.py (since there is a problem with load and save trained SVM, it will be necessary to retrain the SVM every time I want to predict new data) this program will train the SVM and use it to predict new data, the prediction will be a one column file saved to hard disk

9. After getting prediction, run visualize_prediction.py to color positive/good pixel in bright red color on the original image

10. For visual purpose, I want to get rid of the outer frame on the image due to r. Use final_crop.py.
