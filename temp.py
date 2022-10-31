from calendar import EPOCH
import torch
from torch import nn, tensor
import os

# hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = int(1e7)
BATCH_SIZE = 1
horizon = 8
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                                                        
    def forward(self, x):
        output = self.rnn_cell(x)
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


class combined_model( action_input_model, rnn_cell, output_model_1):
    
    def __init__(self):
        super().__init__()
        
    def forward(self):
       
        action_input_data = torch.empty(BATCH_SIZE, horizon, 2, device=device)                           # 2 is linear and angular velocity

        # run the model
        
        action_model = action_input_model().to(device)
        action_input_processed = action_model(action_input_data)
        
        lstm = rnn_cell().to(device)
        lstm_out, _ = lstm(action_input_processed)
        
        output_model = output_model_1().to(device)
        model_output_temp = output_model(lstm_out)                     # [position x, position y, position z, collision], diff from ground_truth
        
        return model_output_temp     # [batch_size, horizon+1, 4], horizon+1 is timestep, 4 is [position x, position y, position z, collision]


model = combined_model().to(device)

# Adam optimizer (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for step in range(EPOCHS):
            
    model_output = model()
    
    # Loss function: MSE for position, cross entropy for collision
    # To get the ground truth data first for the model output
    
    ground_truth_position = torch.empty(1, horizon, 3, device=device)
    ground_truth_collision = torch.empty(1, horizon, device=device)
    
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
    
    print('loss', loss.item())
    print('loss grad is', loss_position.grad)
    # for name, p in model.named_parameters():
    #     print(name, 'gradient is', p.grad)
    
    optimizer.step()
    