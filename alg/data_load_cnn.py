from cifar10_input import *
import numpy as np
import h5py
import os
import random


# batch_size = 256
# epochs = 100
# learning_rate = 2e-3 # bigger to train faster
# num_workers = 4
# print_freq = 500  
# train_test_ratio = 0.8
# # parameters for data
# feedback_bits = 128
# img_height = 16
# img_width = 32
# img_channels = 2


# Data loading
# data_load_address = './data'
# mat = h5py.File(data_load_address + '/2.mat', 'r')
#data = np.transpose(mat['H_train'])  # shape=(320000, 1024)
#data = data.astype('float32')
#data = np.reshape(data, [len(data), img_channels, img_height, img_width])
# split data for training(80%) and validation(20%)
#np.random.shuffle(data)
#start = int(data.shape[0] * train_test_ratio)
#x_train, x_test = data[:start], data[start:]
# def train_data_load():
#     all_data = []
#     all_label = []
#     label = 1
#     for i in range(12):
#         data_load_address = './Train_data_1'
#         mat = h5py.File(data_load_address + '/' + str(i+1) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 2
#     for i in range(12):
#         data_load_address = './Train_data_2'
#         mat = h5py.File(data_load_address + '/' + str(i+1) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     all_data = np.array(all_data)
#     all_label = np.array(all_label)
#     # print(np.shape(all_data))
#     # print(np.shape(all_label))
#     return all_data, all_label


# def val_data_load():
#     all_data = []
#     all_label = []
#     label = 1
#     for i in range(13):
#         data_load_address = './Val_data_1'
#         mat = h5py.File(data_load_address + '/' + str(i+1) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 2
#     for i in range(13):
#         data_load_address = './Val_data_2'
#         mat = h5py.File(data_load_address + '/' + str(i+1) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     all_data = np.array(all_data)
#     all_label = np.array(all_label)
#     all_data = whitening_image(all_data)
#     # print(np.shape(all_data))
#     # print(np.shape(all_label))
#     return all_data, all_label

# def train_data_load():
#     all_data = []
#     all_label = []
#     label = 1
#     for i in range(50):
#         data_load_address = './Train_data_1_50'
#         mat = h5py.File(data_load_address + '/' + str(i+1) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 2
#     for i in range(50):
#         data_load_address = './Val_data_1_50'
#         mat = h5py.File(data_load_address + '/' + str(i+1) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     all_data = np.array(all_data)
#     all_label = np.array(all_label)
#     # print(np.shape(all_data))
#     # print(np.shape(all_label))
#     return all_data, all_label


# def val_data_load():
#     all_data = []
#     all_label = []
#     label = 1
#     for i in range(50):
#         data_load_address = './Train_data_2_50'
#         mat = h5py.File(data_load_address + '/' + str(i+51) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 2
#     for i in range(50):
#         data_load_address = './Val_data_2_50'
#         mat = h5py.File(data_load_address + '/' + str(i+51) + '.mat', 'r')
#         data = mat['cdata']
#         data = np.transpose(data)
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     all_data = np.array(all_data)
#     all_label = np.array(all_label)
#     all_data = whitening_image(all_data)
#     # print(np.shape(all_data))
#     # print(np.shape(all_label))
#     return all_data, all_label

def train_data_load(train_index1,train_index2,train_index3,train_index4,train_index5):
    all_data = []
    all_label = []
    label = [1,0,0,0,0]
    # order = np.random.choice(230, 100)
    # for i in range(1,201,2): #Train_data_1_50
    # for i in train_index: #Train_data_1_50
    for i in train_index1: #Train_data_1_50
        data_load_address = './t_0.2_1'
        mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    # for i in range(101,151): #Train_data_1_50
    #     data_load_address = './DATA_1'
    #     mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    label = [0,1,0,0,0]
    # order = np.random.choice(230, 100)
    # for i in range(1,201,2): #Val_data_1_50
    # for i in train_index: #Train_data_1_50
    for i in train_index2: #Train_data_1_50
        data_load_address = './t_0.2_2'
        mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    # for i in range(101,151): #Train_data_1_50
    #     data_load_address = './DATA_2'
    #     mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    # label = 3
    # order = np.random.choice(230, 100)
    label = [0,0,1,0,0]
    # for i in range(1,201,2): #1_data_1_50
    # for i in train_index: #Train_data_1_50
    for i in train_index3: #Train_data_1_50
        data_load_address = './t_0.2_3'
        mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    # for i in range(101,151): #Train_data_1_50
    #     data_load_address = './DATA_3'
    #     mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    label = [0,0,0,1,0]
    for i in train_index4: #Train_data_1_50
        data_load_address = './t_0.2_4'
        mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = [0,0,0,0,1]
    for i in train_index4: #Train_data_1_50
        data_load_address = './t_0.2_5'
        mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    # # order = np.random.choice(230, 100)
    # # for i in range(1,201,2): #2_data_1_50
    # # for i in train_index: #Train_data_1_50
    # for i in range(1,101): #Train_data_1_50
    #     data_load_address = './DATA_4'
    #     mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    # # for i in range(101,151): #Train_data_1_50
    # #     data_load_address = './DATA_4'
    # #     mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
    # #     data = mat['temp']
    # #     data = np.transpose(data)
    # #     # data = data[:,0:220,:]
    # #     data = data.astype('float32')
    # #     all_data.append(data) 
    # #     all_label.append(label)
    # #     # print(np.shape(data))
    # label = [0,0,0,0,1]
    # # order = np.random.choice(230, 100)
    # # for i in range(1,201,2): #3_data_1_50
    # # for i in train_index: #Train_data_1_50
    # for i in range(1,101): #Train_data_1_50
    #     data_load_address = './DATA_5'
    #     mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
        # print(np.shape(data))
    # for i in range(101,151): #Train_data_1_50
    #     data_load_address = './DATA_5'
    #     mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
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
    label = [1,0,0,0,0]
    # order = np.random.choice(230, 100)
    # for i in range(2,192,2):
    # for i in test_index: #Train_data_1_50
    for i in test_index1: #Train_data_1_50
        data_load_address = './t_0.2_1' #16
        mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    # for i in range(151,231): #Train_data_1_50
    #     data_load_address = './DATA_1' #16
    #     mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    label = [0,1,0,0,0]
    # order = np.random.choice(230, 100)
    # for i in range(2,192,2):
    # for i in test_index: #Train_data_1_50
    for i in test_index2: #Train_data_1_50
        data_load_address = './t_0.2_2'
        mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    # for i in range(151,231): #Train_data_1_50
    #     data_load_address = './DATA_2'
    #     mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    label = [0,0,1,0,0]
    # order = np.random.choice(230, 100)
    # for i in range(2,192,2): #1_data_2_50_1
    # for i in test_index: #Train_data_1_50
    for i in test_index3: #Train_data_1_50
        data_load_address = './t_0.2_3'
        mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
        # print(np.shape(data))
    # for i in range(151,231): #Train_data_1_50
    #     data_load_address = './DATA_3'
    #     mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    label = [0,0,0,1,0]
    for i in test_index3: #Train_data_1_50
        data_load_address = './t_0.2_4'
        mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    label = [0,0,0,0,1]
    for i in test_index3: #Train_data_1_50
        data_load_address = './t_0.2_5'
        mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
        data = mat['temp']
        data = np.transpose(data)
        # data = data[:,0:220,:]
        data = data.astype('float32')
        all_data.append(data) 
        all_label.append(label)
    # # order = np.random.choice(230, 100)
    # # for i in range(2,192,2):
    # # for i in test_index: #Train_data_1_50
    # for i in range(101,201): #Train_data_1_50
    #     data_load_address = './DATA_4'
    #     mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    # # for i in range(151,231): #Train_data_1_50
    # #     data_load_address = './DATA_4'
    # #     mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
    # #     data = mat['temp']
    # #     data = np.transpose(data)
    # #     # data = data[:,0:220,:]
    # #     data = data.astype('float32')
    # #     all_data.append(data) 
    # #     all_label.append(label)
    # #     # print(np.shape(data))
    # label = [0,0,0,0,1]
    # # order = np.random.choice(230, 100)
    # # for i in range(2,42,2):
    # # for i in test_index: #Train_data_1_50
    # for i in range(101,201): #Train_data_1_50
    #     data_load_address = './DATA_5'
    #     mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
        # print(np.shape(data))
    # for i in range(151,171): #Train_data_1_50
    #     data_load_address = './DATA_5'
    #     mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
    #     data = mat['temp']
    #     data = np.transpose(data)
    #     # data = data[:,0:220,:]
    #     data = data.astype('float32')
    #     all_data.append(data) 
    #     all_label.append(label)
    #     # print(np.shape(data))
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    # all_data = whitening_image(all_data)
    print('valdiating data load')
    # print(np.shape(all_data))
    # print(np.shape(all_label))
    return all_data, all_label

# def test_data_load():
#     all_data = []
#     all_label = []
#     label = 1
#     # order = np.random.choice(230, 100)
#     for i in range(196,231,1):
#         data_load_address = './DATA_1' #16
#         mat = h5py.File(data_load_address + '/' + str(i) + '.mat', 'r')
#         data = mat['temp']
#         data = np.transpose(data)
#         # data = data[:,0:220,:]
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 2
#     # order = np.random.choice(230, 100)
#     for i in range(196,231,1):
#         data_load_address = './DATA_2'
#         mat = h5py.File(data_load_address + '/' + str(i+230) + '.mat', 'r')
#         data = mat['temp']
#         data = np.transpose(data)
#         # data = data[:,0:220,:]
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 3
#     # order = np.random.choice(230, 100)
#     for i in range(196,231,1):
#         data_load_address = './DATA_3'
#         mat = h5py.File(data_load_address + '/' + str(i+460) + '.mat', 'r')
#         data = mat['temp']
#         data = np.transpose(data)
#         # data = data[:,0:220,:]
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 4
#     # order = np.random.choice(230, 100)
#     for i in range(196,231,1):
#         data_load_address = './DATA_4'
#         mat = h5py.File(data_load_address + '/' + str(i+690) + '.mat', 'r')
#         data = mat['temp']
#         data = np.transpose(data)
#         # data = data[:,0:220,:]
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     label = 5
#     # order = np.random.choice(230, 100)
#     for i in range(221,231,1):
#         data_load_address = './DATA_5'
#         mat = h5py.File(data_load_address + '/' + str(i+920) + '.mat', 'r')
#         data = mat['temp']
#         data = np.transpose(data)
#         # data = data[:,0:220,:]
#         data = data.astype('float32')
#         all_data.append(data) 
#         all_label.append(label)
#         # print(np.shape(data))
#     all_data = np.array(all_data)
#     all_label = np.array(all_label)
#     all_data = whitening_image(all_data)
#     print('test data load')
#     # print(np.shape(all_data))
#     # print(np.shape(all_label))
#     return all_data, all_label
# test_data_load()