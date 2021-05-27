# from cifar10_input import *
import numpy as np
import h5py
import os
import random

DATASET_PATH = '../dataset'

def train_data_load(train_index1,train_index2,train_index3,train_index4,train_index5):
    all_data = []
    all_label = []
    label = 1
    for i in train_index1:
        data_load_address = '%s/t_0.2_1'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 2
    for i in train_index2:
        data_load_address = '%s/t_0.2_2'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 3
    for i in train_index3:
        data_load_address = '%s/t_0.2_3'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 4
    for i in train_index4:
        data_load_address = '%s/t_0.2_4'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 5
    for i in train_index5:
        data_load_address = '%s/t_0.2_5'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    print('training data load')
    return all_data, all_label

def val_data_load(test_index1,test_index2,test_index3,test_index4,test_index5):
    all_data = []
    all_label = []
    label = 1
    for i in test_index1:
        data_load_address = '%s/t_0.2_1'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 2
    for i in test_index2:
        data_load_address = '%s/t_0.2_2'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 3
    for i in test_index3:
        data_load_address = '%s/t_0.2_3'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 4
    for i in test_index4:
        data_load_address = '%s/t_0.2_4'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = 5
    for i in test_index5:
        data_load_address = '%s/t_0.2_5'%DATASET_PATH
        mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        data = data.astype('float32')
        all_data.append(data)
        all_label.append(label)
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    print('valdiating data load')
    return all_data, all_label
