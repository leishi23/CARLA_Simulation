from calendar import EPOCH
import numpy as np
import matplotlib.image as mpimg
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader
from dataset import carla_vec, carla_rgb, carla_action
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import queue
import pandas as pd

# according to the nn.LSTM documentation, I need to set an environment variable here
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

# hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = int(1e7)
BATCH_SIZE = 1
horizon = 8
WEIGHT_DECAY = 1e-4

# ground truth dataset
ground_truth_folder_path = '/home/lshi23/carla_test/data/ground_truth'          # path to the ground truth of every datapoints folder
ground_truth_file_list = os.listdir(ground_truth_folder_path)
ground_truth_file_list.sort()

# action input dataset
action_input_folder_path = '/home/lshi23/carla_test/data/action_input'
action_input_file_list = os.listdir(action_input_folder_path)
action_input_file_list.sort()

# image input dataset
img_file_path = '/home/lshi23/carla_test/data/image'
img_dataset = carla_rgb(img_file_path)
img_train_dataloader = DataLoader(img_dataset, batch_size=BATCH_SIZE)

datapoints = len(os.listdir(ground_truth_folder_path))
BATCH_NUM = int(datapoints/BATCH_SIZE)                                                              # number of batches

assert len(os.listdir(os.path.join(img_file_path, "rgb_out"))) == datapoints, "check if the rgb dataset length corresponds to ground truth dataset length"
assert len(os.listdir(action_input_folder_path)) == datapoints, "check if the action dataset length corresponds to the ground truth dataset length"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_plot(data, loss_queue):
    loss_queue.put(data)
    
def rotmat(yaw):
    zeros = torch.tensor(0, dtype=torch.float32).to(device)
    ones = torch.tensor(1, dtype=torch.float32).to(device)
    temp = torch.stack([torch.cos(yaw), -torch.sin(yaw), zeros, torch.sin(yaw), torch.cos(yaw), 
                        zeros, zeros, zeros, ones])
    return torch.reshape(temp, (3, 3))

loss_queue = queue.Queue(maxsize=1)

plt.ion()
fig, ax = plt.subplots()
ax.set_title('Loss Plot')
ax.set_ylabel('Training Loss')
ax.set_xlabel('Steps (x10)')
loss_plot_list = list()

class obs_im_model(nn.Module):
        
    def __init__(self):
        super().__init__() 

        self.cnn_conv1 = nn.Conv2d(3, 32, 5, stride=2)
        self.cnn_relu1 = nn.ReLU()
        self.cnn_conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.cnn_relu2 = nn.ReLU()
        self.cnn_conv3 = nn.Conv2d(64, 64, 3, stride=2)
        self.cnn_dense1 = nn.Linear(8960, 256)
        self.cnn_relu3 = nn.ReLU()
        self.cnn_dense2 = nn.Linear(256, 128)        # (256,128)-open source code   or   (256,256)-essay 

    
    def forward(self, x):
        
        x = self.cnn_conv1(x)
        x = self.cnn_relu1(x)
        x = self.cnn_conv2(x)
        x = self.cnn_relu2(x)
        x = self.cnn_conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.cnn_dense1(x)
        x = self.cnn_relu3(x)
        x = self.cnn_dense2(x)
        
        return x
        
     
class obs_vec_model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.CNN_Dense3 = nn.Linear(11, 32)       # shape 2, gps 1, imu 3, jackal 5 ??????
        self.CNN_ReLU4 = nn.ReLU()
        self.CNN_Dense4 = nn.Linear(32, 32)

      
    def forward(self, x):
        x = self.CNN_Dense3(x)
        x = self.CNN_ReLU4(x)
        x = self.CNN_Dense4(x)
        return x
    
    
class obs_lowd_model(nn.Module):
        
    def __init__(self):
        super().__init__()
        
        self.cnn_dense5 = nn.Linear(128, 128)       # 160 = 128 + 32 ???
        self.cnn_relu4 = nn.ReLU()
        self.cnn_dense6 = nn.Linear(128, 128)
     
    def forward(self, x):
        x = self.cnn_dense5(x)
        x = self.cnn_relu4(x)
        x = self.cnn_dense6(x)
        return x
    
    
class action_input_model(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.action_dense1 =  nn.Linear(2, 16)           # verified input (8,2), output(8,16) when nn.Linear(2,16)
        self.action_relu3 = nn.ReLU()
        self.action_dense2 = nn.Linear(16, 16)
        
    def forward(self, x):
        
        x = self.action_dense1(x)
        x = self.action_relu3(x)
        x = self.action_dense2(x)
        return x
    
    
class rnn_cell(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.rnn_cell = nn.LSTM(16, 64, 8, batch_first = True)             # (input_size, hidden_size/num_units, num_layers)  
                                                        
    def forward(self, x, h, c):
        output = self.rnn_cell(x, (h, c))
        return output
    
    
class output_model_1(nn.Module):
        
    def __init__(self):
        super().__init__()
        
        self.output_dense1 = nn.Linear(64, 32)       # hidden layer features are 64
        self.output_relu3 = nn.ReLU()
        self.output_dense2 = nn.Linear(32, 4)       # 4 is the output dimension, actually 8*4
        
    def forward(self, x):
        
        x = self.output_dense1(x)
        x = self.output_relu3(x)
        x = self.output_dense2(x)
        return x


class combined_model(obs_im_model, obs_vec_model, obs_lowd_model, action_input_model, rnn_cell, output_model_1):
    
    def __init__(self):
        super().__init__()
        
        
    def forward(self, img_train_dataloader, ground_truth_files_list, action_input_files_list):
        
        model_output = torch.empty((BATCH_NUM*BATCH_SIZE, horizon, 4), device=device)
        
        for j in range(BATCH_NUM):
            
            img_raw_data, _ = next(iter(img_train_dataloader))  # [batch_size, 3, Height, Width]
            img_raw_data = img_raw_data.to(device).type(torch.float32)/255
            
            # str list for ground truth files and action input files
            ground_truth_files = ground_truth_files_list[j*BATCH_SIZE:(j+1)*BATCH_SIZE]           
            action_input_files = action_input_files_list[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            
            # img_raw_data = img_raw_data[0]
            # img = transforms.ToPILImage()(img_raw_data).convert('RGB')
            # plt.imshow(img)
            # plt.show()
            # plt.pause(0.001)
            
            action_input_data = torch.empty(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
            for i in range(BATCH_SIZE):   
                action_input_temp_path = os.path.join(action_input_folder_path, action_input_files[i])       # path for action input file                
                action_input_temp = pd.read_csv(action_input_temp_path, header=None).values                  # a 2*8 ndarray, 1st row is linear velocity, 2nd row is angular velocity, column is timestep
                action_input_temp = np.transpose(action_input_temp)                                          # 8*2 ndarray
                action_input_temp = torch.from_numpy(action_input_temp).to(device).type(torch.float32)
                action_input_data[i, :, :] = action_input_temp
                
            ground_truth_data = torch.empty(BATCH_SIZE, horizon+1, 9, device=device)                         # 9 is [collision, location 3, velocity, yaw, angular velocity, latlong 2]
            for i in range(BATCH_SIZE):
                ground_truth_temp_path = os.path.join(ground_truth_folder_path, ground_truth_files[i])
                ground_truth_temp = pd.read_csv(ground_truth_temp_path, header=None).values
                ground_truth_temp = np.transpose(ground_truth_temp)                                          # transpose is necessary, because the pd read data in row instead of column, row is feature, column is timestep
                ground_truth_temp = torch.from_numpy(ground_truth_temp).to(device).type(torch.float32)
                ground_truth_data[i, :, :] = ground_truth_temp
                
            # run the model
            cnn_model1 = obs_im_model().to(device)
            cnn_out1 = cnn_model1(img_raw_data)
            
            cnn_model2 = obs_lowd_model().to(device)
            cnn_out2 = cnn_model2(cnn_out1)
            
            action_model = action_input_model().to(device)
            action_input_processed = action_model(action_input_data)
            
            initial_state_c_temp, initial_state_h_temp = torch.split(cnn_out2, 64, dim=-1)
            initial_state_c = torch.empty(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_c_temp)
            initial_state_h = torch.empty(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_h_temp)
            
            # initial_state_c = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
            # initial_state_c[0] = initial_state_c_temp
            # initial_state_h = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
            # initial_state_h[0] = initial_state_h_temp
            
            
            lstm = rnn_cell().to(device)
            lstm_out, _ = lstm(action_input_processed, initial_state_h, initial_state_c)
            
            output_model = output_model_1().to(device)
            model_output_temp = output_model(lstm_out)                     # [position x, position y, position z, collision], diff from ground_truth
                         
                    
            for i in range(BATCH_SIZE):
                yaw_data = ground_truth_data[i, 0, 5]      # yaw is from vehicle transform frame where clockwise is positive. In rotation matrix, yaw is from world frame where counter-clockwise is positive. When reverse the yaw, it's still counter-clockwise.
                rotmatrix = rotmat(yaw_data)
                model_output_temp_local_position = model_output_temp[i, :, :3]
                model_output_temp_global_position = torch.matmul(model_output_temp_local_position, rotmatrix)
                model_output[i+j*BATCH_SIZE, :,:3] = model_output_temp_global_position + ground_truth_data[i, 0, 1:4]     # add the initial position for timestep 0 to the output position 
                model_output[i+j*BATCH_SIZE, :, 3] = model_output_temp[i, :, 3]
            
        return model_output     # [batch_size, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]


model = combined_model().to(device)

# Adam optimizer (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for step in range(EPOCHS):
            
    model_output = model(img_train_dataloader, ground_truth_file_list, action_input_file_list)
    
    # Loss function: MSE for position, cross entropy for collision
    # To get the ground truth data first for the model output
    
    ground_truth_position = torch.empty(BATCH_SIZE*BATCH_NUM, horizon, 3, device=device)
    ground_truth_collision = torch.empty(BATCH_SIZE*BATCH_NUM, horizon, device=device)
    for i in range(BATCH_NUM):
        for j in range(BATCH_SIZE):
            ground_truth_file_path = os.path.join(ground_truth_folder_path, ground_truth_file_list[i*BATCH_SIZE+j])
            ground_truth_temp = pd.read_csv(ground_truth_file_path, header=None).values
            ground_truth_temp = np.transpose(ground_truth_temp)
            ground_truth_temp = torch.from_numpy(ground_truth_temp).to(device).type(torch.float32)
            ground_truth_position_temp = ground_truth_temp[1:horizon+1, 1:4]
            ground_truth_collision_temp = ground_truth_temp[1:horizon+1, 0]
            ground_truth_position[j+BATCH_SIZE*i] = ground_truth_position_temp
            ground_truth_collision[j+BATCH_SIZE*i] = ground_truth_collision_temp
    
    loss_mse = nn.MSELoss(reduction='mean')
    loss_position = loss_mse(model_output[:, :, :2], ground_truth_position[:,:,:2])
    loss_position.retain_grad()
     
    loss_cross_entropy = nn.CrossEntropyLoss(reduction='sum')        
    loss_collision = loss_cross_entropy(model_output[:, :, 3], ground_truth_collision)
    if loss_collision != 0:
        print('loss_collision', loss_collision)
    
    loss = loss_position + loss_collision
    loss.retain_grad()
    optimizer.zero_grad()
    loss.backward()
    
    print('loss grad is', loss.grad)
    for name, p in model.named_parameters():
        print(name, 'gradient is', p.grad)
    
    optimizer.step()
    
    if step % 10 == 0:
        print('step: ', step, 'loss: ', loss.item())
        loss_plot(loss.item(), loss_queue)
        loss_data = loss_queue.get()
        loss_plot_list.append(loss_data)
        ax.plot(loss_plot_list, color='blue')
        plt.show()
        plt.pause(0.001)
        
    if step % 3000 == 0 and step != 0:
        print('model saved')
    