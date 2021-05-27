#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from hyper_parameters import *
import cv2
from sys import argv

def whitening_image(image):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    mean = np.mean(image)
    # Use adjusted standard deviation here, in case the std == 0.
    std = np.max([
        np.std(image),
        1.0 / np.sqrt(IMG_HEIGHT*IMG_WIDTH*IMG_DEPTH)
    ])
    image = (image - mean) / std
    return image

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

def build_network():
    xs = tf.placeholder(tf.float32, [None, 6912])/1.   # 28x28
    ys = tf.placeholder(tf.float32, [None, 5])
    # lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 48, 48, 3])
    # print(x_image.shape)  # [n_samples, 28,28,1]

    ## conv1 layer ##
    W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 3, out size 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 48x48x32
    h_pool1 = max_pool_2x2(h_conv1)                          # output size 24x24x32

    ## conv2 layer ##
    W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 24x24x64
    h_pool2 = max_pool_2x2(h_conv2)                          # output size 12x12x64

    ## fc1 layer ##
    W_fc1 = weight_variable([12*12*64, 128])
    b_fc1 = bias_variable([128])
    # [n_samples, 12, 12, 64] ->> [n_samples, 12*12*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12*12*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    W_fc2 = weight_variable([128, 5])
    b_fc2 = bias_variable([5])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return (xs, ys, keep_prob, prediction)

def main():
    if len(argv) != 2:
        print('This program requires one argument: *.jpg file path')
        return
    _file = argv[1]
    data = cv2.imread(_file).astype('float32')
    data = whitening_image(data)
    #
    _count = 0
    data_ser = np.zeros((IMG_HEIGHT*IMG_WIDTH*IMG_DEPTH,), dtype=np.float32)
    for j in range(IMG_HEIGHT):
        for k in range(IMG_WIDTH):
            for l in range(IMG_DEPTH):
                data_ser[_count] = data[j,k,l]
                _count += 1
    v_xs = np.expand_dims(data_ser, axis=0)
    #
    (xs, ys, keep_prob, pred) = build_network()
    sess = tf.Session()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    #
    saver = tf.train.Saver()
    last_ckpt = tf.train.latest_checkpoint('ckpt')
    saver.restore(sess, last_ckpt)
    #
    y_pred = sess.run(pred, feed_dict={xs: v_xs, keep_prob:1})[0]
    _label = y_pred.argmax() + 1
    # print(last_ckpt, _label, y_pred)
    print('The used model:', last_ckpt)
    print('Prediction label: ', _label)
    print('Raw prediction result:', y_pred)
    pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e
    finally:
        pass
