#!/usr/bin/env python3
from __future__ import print_function
from tqdm import tqdm
from sys import argv
from pathlib import Path
#
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
from data_load_cnn import *
from hyper_parameters import *

##===================================== Load the Dataset =====================================##
random.seed(SPLIT_SEED)
index = random.sample(range(1,201),100)

vali_data, vali_labels = val_data_load( *[index]*5 )

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

vali_data = whitening_image(vali_data)

test_data = []
for i in range(500):
    temp2 = [[0 for x in range(1)] for y in range(6912)]
    count = 0
    for j in range(48):
        for k in range(48):
            for l in range(3):
                temp2[count] = vali_data[i,j,k,l]
                count = count + 1
    test_data.append(temp2)
test_data = np.array(test_data)
test_labels = np.array(vali_labels)
##===================================== Load the Dataset =====================================##

def compute_uncertainty(sess, v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    res_list=[]

    for j in range(len(y_pre)):
        one_list=[]
        for i in range(len(y_pre[0])):
            one_list.append(y_pre[j][i])
        res_list.append(max(one_list))
    result = sum(res_list)/len(res_list)
    result = min(res_list)
    return result

def compute_accuracy(sess, v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal( tf.argmax(y_pre,1), tf.argmax(v_ys,1) )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

##============================== Build the Network ==============================##
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 6912])/1.   # 28x28
ys = tf.placeholder(tf.float32, [None, 5])
# lr = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 48, 48, 3])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([12*12*64, 128])
b_fc1 = bias_variable([128])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([128, 5])
b_fc2 = bias_variable([5])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# number of parameters: (5*5+1)*32+(5*5*32+1)*64+(7*7*64+1)*1024+(1024+1)*10
# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum( ys * tf.log(prediction), reduction_indices=[1] ))       # loss
# train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
##============================== Build the Network ==============================##

##=============================== Inference Procedure ===============================##
sess  = tf.Session()
saver = tf.train.Saver()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
#
try:
    ckpt_index = int(argv[1])
    last_ckpt = 'ckpt/cnn-model-%d'%ckpt_index
    assert( Path(last_ckpt+'.index').exists() )
except:
    last_ckpt = tf.train.latest_checkpoint('ckpt')
if last_ckpt:
    saver.restore(sess, last_ckpt)
    print( '===== success loading from %s ====='%last_ckpt )
#
acc = compute_accuracy(sess, test_data, test_labels)
print(acc)
##=============================== Inference Procedure ===============================##
