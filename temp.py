from ctypes import sizeof
from sys import maxsize
import torch
from torch import nn
# import tensorflow as tf
import numpy as np
import queue
import pandas as pd
# from dataset import carla_vec
from torch.utils.data import DataLoader
import os
import json
import matplotlib.image as mpimg
# import tensorflow as tf

a = np.array([1,2,3,4,5,6,7,8,9,10])
a = a[1,3]
print(a)

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()
# print()

# is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
# print('tensorflow gpu is %s' % is_cuda_gpu_available)
# print('pytorch gpu is %s' % torch.cuda.is_available())

# path = '/home/lshi23/carla_test/data/vector.json'
# with open(path) as f:
#     data = json.load(f)
#     print(data)
# a = np.array([[1,2],[3,4]])
# b = np.array([5,6])
# print(a@b)
# try:
#     path = '/home/lshi23/carla_test/data'
#     for f in os.listdir(path):
#         if f.endswith('.json'):
#             os.remove(os.path.join(path, f))

# finally:
#     pass

# json_path = '/home/lshi23/carla_test/data/json'
# json_dir = os.listdir(json_path)
# json_dir.sort()

# img_path = '/home/lshi23/carla_test/data/rgb_out'
# img_dir = os.listdir(img_path)
# img_dir.sort()

# num_frame = len(json_dir)
# num_horizon = 8
# num_datapoints = int(num_frame/(num_horizon+1))

# print('Datapoint creation process')

# for i in range(num_datapoints):
    
#     datapoint = {'Input':{}, 'Output':{}}
    
#     # Input 
#     img_path_single = os.path.join(img_path,img_dir[9*i])
#     img_np = mpimg.imread(img_path_single)
#     datapoint['Input']['rgb_input'] = img_np.tolist()                    # numpy array
    
#     json_path_single = os.path.join(json_path, json_dir[9*i])
#     with open(json_path_single) as f:
#         json_single_data = json.load(f)
#         datapoint['Input']['vec_input'] = json_single_data      # dict
    
#     datapoint['Input']['action_input'] = [0, 0]    # numpy array
    
#     # Output
#     for j in range(num_horizon):
#         json_path_horizon = os.path.join(json_path, json_dir[9*i+j+1])
#         with open(json_path_horizon) as f:
#             json_horizon_data = json.load(f)
#             datapoint['Output'][j] = json_horizon_data
            
#     print ("----%.0f%%----" % (100 * i/num_datapoints))
#     json_object = json.dumps(datapoint, indent=2)
#     with open('/home/lshi23/carla_test/data/%04d.json' % i, 'w') as outfile:outfile.write(json_object)


# a = tf.keras.Input(shape=(3, 128))
# b = tf.keras.Input(shape=(3, 32))

# c = tf.concat([a, b], axis=2)
# print(c)

# file_path = '/home/lshi23/carla_test/data/json'
# dataset = []
# dir = os.listdir(file_path)
# dir.sort()
# for f in dir:
#     path2 = os.path.join(file_path, f)
#     with open(path2) as f2:
#         data = json.load(f2)
#         dataset.append(data)

# m = nn.Linear(20, 30)
# input = torch.randn(128, 20)
# output = m(input)
# print(output.shape)

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# tf.concat([t1, t2], 0)
# print(t1.shape)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(30, activation='relu'))
# tf_input = np.random.rand(128, 20)
# tf_input = tf.constant(tf_input)
# tf_output = model(tf_input)
# print(tf_output.shape)

# rnn = nn.LSTM(10, 30, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 30)
# c0 = torch.randn(2, 3, 30)
# output, (hn, cn) = rnn(input, (h0, c0))
# print(output.shape)