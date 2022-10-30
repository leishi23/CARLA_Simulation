from calendar import EPOCH
import numpy as np
import matplotlib.image as mpimg
import torch
from torch import nn
import pylab as pl
from torch.utils.data import DataLoader
from dataset import carla_rgb
import matplotlib.pyplot as plt
from torchvision import transforms

# hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 4

# dataset
image_file_path = '/home/lshi23/carla_test/data/image'
img_dataset = carla_rgb(image_file_path)
train_img_dataloader = DataLoader(img_dataset, batch_size=BATCH_SIZE)

data_file_path = '/home/lshi23/carla_test/data/datapoints'
# dataset = carla_data(data_file_path)
# train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):

    def __init__(self):

        super().__init__()     

        # self.conv_1 = nn.Conv2d(3, 32, 5, stride=2)
        # self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        # self.conv_3 = nn.Conv2d(64, 64, 3, stride=2)
        # self.relu = nn.ReLU()

        self._setup_layers()


    def _setup_layers(self):
    # def forward(self, x):

        self._obs_im_model = nn.Sequential([
            ('CNN/Conv1', nn.Conv2d(3, 32, 5, stride=2)),
            ('CNN/ReLU1', nn.ReLU()),
            ('CNN/Conv2', nn.Conv2d(32, 64, 3, stride=2)),
            ('CNN/ReLU2', nn.ReLU()),
            ('CNN/Conv3', nn.Conv2d(64, 64, 3, stride=2)),
            ('CNN/Flatten', torch.flatten()),
            ('CNN/Dense1', nn.Linear(289536, 256)),
            ('CNN/ReLU3', nn.ReLU()),
            ('CNN/Dense2', nn.Linear(256, 128)),         # (256,128)-open source code   or   (256,256)-essay
            
        ])
        
        self._obs_vec_model = nn.Sequential([
            ('CNN/Dense3', nn.Linear(11, 32)),          # shape 2, gps 1, imu 3, jackal 5 ??????
            ('CNN/ReLU4', nn.ReLU()),
            ('CNN/Dense4', nn.Linear(32, 32))
        ])
        
        self._obs_lowd_model = nn.Sequential([
            ('CNN/Dense5', nn.Linear(160, 128)),        # 160 = 128 + 32 ???
            ('CNN/ReLU4', nn.ReLU()),
            ('CNN/Dense6', nn.Linear(128, 128))
        ])

        self._action_model = nn.Sequential([
        ('action/Dense1', nn.Linear(2, 16)),            # verified input (8,2), output(8,16) when nn.Linear(2,16)
        ('action/ReLU3',nn.ReLU()),
        ('action/Dense2', nn.Linear(16, 16))
        ])

        self._rnn_cell = nn.LSTM(16, 64, 8)             # (input_size, hidden_size/num_units, num_layers)  
                                                        # but debug badgr shows it's 16 input features

        self._output_model = nn.Sequential([
            ('output/Dense1', nn.Linear(64, 32)),       # 8*64, horizon is 8 and hidden layer features are 64
            ('output/ReLU3',nn.ReLU()),
            ('output/Dense2', nn.Linear(32, 4))         # 4 is the output dimension, actually 8*4
        ])

# model = Model()

# optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

for step in range(EPOCHS):

    img_train_features, img_train_labels = next(iter(train_img_dataloader))
    # train_features = next(iter(train_dataloader))
    # img_train_features = img_train_features.to(device).type(torch.float32)
    
    img = img_train_features[step]
    img = transforms.ToPILImage()(img).convert('RGB')
    plt.imshow(img)
    plt.show()
    plt.pause(0.1)
    # print(train_features['Input']['action_input'])
    # print(step)
    
    # model_output = model(img)
    # # print(model_output)

    # # Compute loss

    # loss_func = torch.nn.MSELoss(reduction ='sum')
    # loss = loss_func(img, model_output)

    # # Zero gradients, perform a backward pass, and update the weights.

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # if step % 10 == 0:
    #     img_show = pl.imshow(model_output)
    #     pl.draw