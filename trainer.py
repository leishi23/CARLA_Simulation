from calendar import EPOCH
import numpy as np
import matplotlib.image as mpimg
import torch
from torch import nn, tensor
from torch.utils.data import DataLoader
from dataset import carla_rgb
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import queue
import pandas as pd

# according to the nn.LSTM documentation, I need to set an environment variable here
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

# hyperparameters
LEARNING_RATE = 5e-4
EPOCHS = int(1e7)
BATCH_SIZE = 5
horizon = 8
WEIGHT_DECAY = 1e-6

# ground truth dataset
ground_truth_folder_path = 'data/ground_truth'          # path to the ground truth of every datapoints folder
ground_truth_file_list = os.listdir(ground_truth_folder_path)
ground_truth_file_list.sort()

# action input dataset
action_input_folder_path = 'data/action_input'
action_input_file_list = os.listdir(action_input_folder_path)
action_input_file_list.sort()

# image input dataset
img_file_path = 'data/image'
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
ax.set_xlabel('Epochs (x10)')
loss_plot_list = list()
class combined_model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.obs_im_model = nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2),
        nn.Flatten(start_dim=1),
        nn.Linear(8960, 256),
        nn.ReLU(),
        nn.Linear(256, 128) 
        )
        
        self.obs_lowd_model = nn.Sequential(
        nn.Linear(128, 128),       
        nn.ReLU(),
        nn.Linear(128, 128)
        )
        
        self.action_input_model = nn.Sequential(
        nn.Linear(2, 16),           # verified input (8,2), output(8,16) when nn.Linear(2,16)
        nn.ReLU(),
        nn.Linear(16, 16)
        )
        
        self.rnn_cell = nn.LSTM(16, 64, 8, batch_first = True)             # (input_size, hidden_size/num_units, num_layers)
        
        self.output_model = nn.Sequential(
        nn.Linear(64, 32),       # hidden layer features are 64
        nn.ReLU(),
        nn.Linear(32, 4)       # 4 is the output dimension, actually 8*4   
        )
        
    def forward(self, img_train_dataloader, ground_truth_files_list, action_input_files_list):
        
        model_output = torch.randn((BATCH_NUM*BATCH_SIZE, horizon, 4), device=device)
        
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
            
            action_input_data = torch.randn(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity
            for i in range(BATCH_SIZE):   
                action_input_temp_path = os.path.join(action_input_folder_path, action_input_files[i])       # path for action input file                
                action_input_temp = pd.read_csv(action_input_temp_path, header=None).values                  # a 2*8 ndarray, 1st row is linear velocity, 2nd row is angular velocity, column is timestep
                action_input_temp = np.transpose(action_input_temp)                                          # 8*2 ndarray
                action_input_temp = torch.from_numpy(action_input_temp).to(device).type(torch.float32)
                action_input_data[i, :, :] = action_input_temp
                
            ground_truth_data = torch.randn(BATCH_SIZE, horizon+1, 9, device=device)                         # 9 is [collision, location 3, velocity, yaw, angular velocity, latlong 2]
            for i in range(BATCH_SIZE):
                ground_truth_temp_path = os.path.join(ground_truth_folder_path, ground_truth_files[i])
                ground_truth_temp = pd.read_csv(ground_truth_temp_path, header=None).values
                ground_truth_temp = np.transpose(ground_truth_temp)                                          # transpose is necessary, because the pd read data in row instead of column, row is feature, column is timestep
                ground_truth_temp = torch.from_numpy(ground_truth_temp).to(device).type(torch.float32)
                ground_truth_data[i, :, :] = ground_truth_temp
                
            cnn_out1 = self.obs_im_model(img_raw_data)
            cnn_out2 = self.obs_lowd_model(cnn_out1)
            action_input_processed = self.action_input_model(action_input_data)
            
            initial_state_c_temp, initial_state_h_temp = torch.split(cnn_out2, 64, dim=-1)
            initial_state_c = torch.randn(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_c_temp)
            initial_state_h = torch.randn(horizon, BATCH_SIZE, 64, device=device).copy_(initial_state_h_temp)
            
            # initial_state_c = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
            # initial_state_c[0] = initial_state_c_temp
            # initial_state_h = torch.zeros(horizon, BATCH_SIZE, 64, device=device)
            # initial_state_h[0] = initial_state_h_temp       
            
            lstm_out, _ = self.rnn_cell(action_input_processed, (initial_state_h, initial_state_c))
            
            model_output_temp = self.output_model(lstm_out)           # [position x, position y, position z, collision], diff from ground_truth              
                    
            for i in range(BATCH_SIZE):
                yaw_data = ground_truth_data[i, 0, 5]      # yaw is from CARLA vehicle transform frame where clockwise is positive. In rotation matrix, yaw is from world frame where counter-clockwise is positive. When reverse the yaw, it's still counter-clockwise.
                rotmatrix = rotmat(yaw_data)
                model_output_temp_local_position = model_output_temp[i, :, :3]
                model_output_temp_global_position = torch.matmul(model_output_temp_local_position, rotmatrix)
                model_output[i+j*BATCH_SIZE, :,:3] = model_output_temp_global_position + ground_truth_data[i, 0, 1:4]     # add the initial position for timestep 0 to the output position 
                model_output[i+j*BATCH_SIZE, :, 3] = model_output_temp[i, :, 3]
            
        return model_output     # [batch_size, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]

model = combined_model().to(device)

# Adam optimizer (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_max = 0.0

for step in range(EPOCHS):
            
    model_output = model(img_train_dataloader, ground_truth_file_list, action_input_file_list)
    model_output.retain_grad()
    
    # Loss function: MSE for position, cross entropy for collision
    # To get the ground truth data first for the model output
    
    ground_truth_position = torch.randn(BATCH_SIZE*BATCH_NUM, horizon, 3, device=device)
    ground_truth_collision = torch.randn(BATCH_SIZE*BATCH_NUM, horizon, device=device)
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
    
    loss_mse = nn.MSELoss(reduction='sum')
    loss_position = loss_mse(model_output[:, :, :3], ground_truth_position)
    loss_position.retain_grad()
     
    loss_cross_entropy = nn.CrossEntropyLoss(reduction='sum')        
    loss_collision = loss_cross_entropy(model_output[:, :, 3], ground_truth_collision)
    # if loss_collision != 0:
    #     print('loss_collision', loss_collision)
    
    loss = (loss_position + loss_collision)/BATCH_NUM
    loss_max = max(loss_max, loss)
    loss.retain_grad()
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    
    # make_dot(loss.mean(), params = dict(model.named_parameters())).render("combined_model", format="png")
    # transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
    # graph = hl.build_graph(model, torch.zeros([1, 3, 96, 128]), transforms=transforms)
    # graph.theme = hl.graph.THEMES['blue'].copy()
    # graph.save('rnn_hiddenlayer', format='png')
    
    optimizer.step()
    
    if step % 10 == 0:
        print('Epoch: ', step, 'loss: ', loss.item())
        # print('loss grad is', loss_position.grad)
        # for name, p in model.named_parameters():
        #     print(name, 'gradient is', p.grad)
        loss_plot(loss.item(), loss_queue)
        loss_data = loss_queue.get()
        loss_plot_list.append(loss_data)
        ax.plot(loss_plot_list, color='blue')
        plt.show()
        plt.pause(0.001)
        
    if step % 300 == 0 and step != 0:
        print('come on, Lei, you can do it!')
        
    if loss < loss_max*0.0001 or loss < 0.055:
        print('loss is less than 1% of the max loss or loss is smaller than 1.5, save model, break')
        torch.save(model.state_dict(), '/home/lshi23/carla_test/combined_model.pt')
        break
    