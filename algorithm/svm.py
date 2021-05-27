#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:49:33 2019

@author: shuai wang
"""
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
# import matplotlib.pyplot as plt
# import pdb
# Import datasets, classifiers and performance metrics
from data_load_svm import *
from hyper_parameters import *
from sklearn import svm, metrics
import random, time
from termcolor import cprint

class Timer:
    def __init__(self, print_=True):
        self.print_ = print_
        pass
    
    def __enter__(self):
        self.start_time = time.time()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        if self.print_:
            print('Time Elapsed: %.2f seconds.'%self.elapsed_time)
    pass

random.seed(SPLIT_SEED)
index = random.sample(range(1,201),200)

all_data, all_labels = train_data_load( *([ index[:100] ]*5) )
vali_data, vali_labels = val_data_load( *([ index[100:] ]*5) )
used_data = all_data[0:FLAGS.sample_size,:,:]
used_labels = all_labels[0:FLAGS.sample_size]

# whiten the image
def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np

# Create a classifier: a support vector classifier
classifier = svm.SVC(C=1,gamma=0.001)

vali_data = whitening_image(vali_data)
used_data = whitening_image(used_data)
#
train_data = []
test_data = []
for i in range(500):
    temp1 = [[0 for x in range(1)] for y in range(5808)]
    count = 0
    for j in range(44):
        for k in range(44):
            for l in range(3):
                temp1[count] = used_data[i,j,k,l]
                count = count + 1
    train_data.append(temp1)
for i in range(500):
    temp2 = [[0 for x in range(1)] for y in range(5808)]
    count = 0
    for j in range(44):
        for k in range(44):
            for l in range(3):
                temp2[count] = vali_data[i,j,k,l]
                count = count + 1
    test_data.append(temp2)

with Timer():
    cprint('SVM Training start.', 'green')
    classifier.fit(train_data, used_labels)
    cprint('SVM Training finished.', 'red')
#
expected = vali_labels
predicted = classifier.predict(test_data)
acc = metrics.accuracy_score(expected, predicted)
print('test accuracy: ', acc)
