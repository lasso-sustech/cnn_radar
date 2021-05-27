#!/usr/bin/env python3
import numpy as np
import h5py
import random

DATASET_PATH = '../dataset'

def train_data_load(train_index1,train_index2,train_index3,train_index4,train_index5):
    all_data = []
    all_label = []
    #
    label = [1,0,0,0,0]
    for i in train_index1: #Train_data_1_50
        data_load_address = '%s/t_0.2_1'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    #
    label = [0,1,0,0,0]
    for i in train_index2: #Train_data_1_50
        data_load_address = '%s/t_0.2_2'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    label = [0,0,1,0,0]
    for i in train_index3: #Train_data_1_50
        data_load_address = '%s/t_0.2_3'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    label = [0,0,0,1,0]
    for i in train_index4: #Train_data_1_50
        data_load_address = '%s/t_0.2_4'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    label = [0,0,0,0,1]
    for i in train_index5: #Train_data_1_50
        data_load_address = '%s/t_0.2_5'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    print('training data load')
    # print(np.shape(all_data))
    # print(np.shape(all_label))
    return all_data, all_label
# train_data_load()

def val_data_load(test_index1,test_index2,test_index3,test_index4,test_index5):
    all_data = []
    all_label = []
    #
    label = [1,0,0,0,0]
    for i in test_index1: #Train_data_1_50
        data_load_address = '%s/t_0.2_1'%DATASET_PATH #16
        mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    #
    label = [0,1,0,0,0]
    for i in test_index2: #Train_data_1_50
        data_load_address = '%s/t_0.2_2'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    label = [0,0,1,0,0]
    for i in test_index3: #Train_data_1_50
        data_load_address = '%s/t_0.2_3'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    label = [0,0,0,1,0]
    for i in test_index4: #Train_data_1_50
        data_load_address = '%s/t_0.2_4'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    label = [0,0,0,0,1]
    for i in test_index5: #Train_data_1_50
        data_load_address = '%s/t_0.2_5'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    #
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    # all_data = whitening_image(all_data)
    print('valdiating data load')
    # print(np.shape(all_data))
    # print(np.shape(all_label))
    return all_data, all_label
# val_data_load()
