# from calendar import EPOCH
# import torch
# from torch import nn, tensor
# import os
# from torchviz import make_dot

# rnn = nn.LSTM(2, 1, 1)  # 2 input, 4 hidden, 1 layer
# rnn.bias = False
# input = tensor([[1, 2]], dtype=torch.float32)
# # h0 = tensor([[0, 0, 0, 0]], dtype=torch.float32)
# # c0 = tensor([[0, 0, 0, 0]], dtype=torch.float32)
# output, (hn, cn) = rnn(input)
# print(output)

import os 
import shutil
img_path = '/home/lshi23/carla_test/data/image/rgb_out'
test_img_path = '/home/lshi23/carla_test/data/test_dataset/image/rgb_out'
src_path = os.path.join(img_path, '000100.jpg')
dst_path = os.path.join(test_img_path )  
shutil.move(src_path, dst_path)    