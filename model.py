from calendar import EPOCH
import torch
from torch import nn

class Model(nn.Module):

    def __init__(self):

        super().__init__()     
        # self._setup_layers()


    # def _setup_layers(self):

    def obs_im_model(input):
        
        model = nn.Sequential([
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
        
        output = model(input)
        return output
        
        
    def obs_vec_model(input):
        
        model = nn.Sequential([
            ('CNN/Dense3', nn.Linear(11, 32)),          # shape 2, gps 1, imu 3, jackal 5 ??????
            ('CNN/ReLU4', nn.ReLU()),
            ('CNN/Dense4', nn.Linear(32, 32))
        ])
        
        output = model(input)
        return output
    
    
    def obs_lowd_model(input):
        
        model = nn.Sequential([
            ('CNN/Dense5', nn.Linear(160, 128)),        # 160 = 128 + 32 ???
            ('CNN/ReLU4', nn.ReLU()),
            ('CNN/Dense6', nn.Linear(128, 128))
        ])
        
        output = model(input)
        return output
    
    
    def action_model(input):

        model = nn.Sequential([
        ('action/Dense1', nn.Linear(2, 16)),            # verified input (8,2), output(8,16) when nn.Linear(2,16)
        ('action/ReLU3',nn.ReLU()),
        ('action/Dense2', nn.Linear(16, 16))
        ])
        
        output = model(input)
        return output
    
    
    def rnn_cell(input):

        rnn_cell = nn.LSTM(16, 64, 8)             # (input_size, hidden_size/num_units, num_layers)  
                                                        # but debug badgr shows it's 16 input features
        output = rnn_cell(input)
        return output
    
    
    def output_model(input):
        
        model = nn.Sequential([
            ('output/Dense1', nn.Linear(64, 32)),       # 8*64, horizon is 8 and hidden layer features are 64
            ('output/ReLU3',nn.ReLU()),
            ('output/Dense2', nn.Linear(32, 4))         # 4 is the output dimension, actually 8*4
        ])
        output = model(input)
        return output