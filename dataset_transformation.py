import os
import json
import pandas as pd
import numpy as np

try:
    path = '/home/lshi23/carla_test/data/vector.json'
    if os.path.exists(path):
        os.remove(path)
finally:
    pass

json_path = '/home/lshi23/carla_test/data/raw_data'
json_dir = os.listdir(json_path)
json_dir.sort()

img_path = '/home/lshi23/carla_test/data/image/rgb_out'
img_dir = os.listdir(img_path)
img_dir.sort()

num_frame = len(json_dir)                   # how many time steps collected
num_horizon = 8                             # how many time steps in the horizon
num_steps = num_horizon + 1                 # how many time steps in one datapoint (including current time step)
num_datapoints = int(num_frame/num_steps)   # how many datapoints in total
num_remains = num_frame % num_steps         # how many time steps left and need to be discarded

# when ctrl C in data_collection.py, maybe #image is ONE bigger than #json
if num_frame < len(img_dir):
    os.remove(os.path.join(img_path, img_dir[-1]))
    img_dir = img_dir[:-1]
   
assert num_frame == len(img_dir), 'number of json and image files are not equal'
 
# to delete the remains image and json files, also update the directory
for i in range(num_remains):
    os.remove(os.path.join(json_path, json_dir[-1]))
    os.remove(os.path.join(img_path, img_dir[-1]))
    img_dir = img_dir[:-1]
    json_dir = json_dir[:-1]

assert len(json_dir) % num_steps == 0, 'number of json/image files are not multiple of num_steps, remains are not deleted'

print('Datapoint creation process')

for i in range(num_datapoints):
    
    # Input: delete image num_steps*i ~ num_steps*i+horizon, and put action input
    # Image input 
    for j in range(num_steps*i+1,num_steps*i+num_steps):
        os.remove(os.path.join(img_path, img_dir[j]))
    
    # Action input for datapoint i
    action_input = np.zeros((2, num_horizon))
    for j in range(num_horizon):
        json_path_single = os.path.join(json_path, json_dir[num_steps*i+j])                 # action input for time step 0 ~ horizon-1
        with open(json_path_single) as f:
            json_single_data = json.load(f)
            linear_velocity = json_single_data[4]
            angular_velocity = json_single_data[6]
            action_input[0][j] = linear_velocity
            action_input[1][j] = angular_velocity
    df1 = pd.DataFrame(action_input)
    df1.to_csv('/home/lshi23/carla_test/data/action_input/action_input_%06d.csv' % i, index=False, header=False)
    
    # Output: the ground truth vector for datapoint i
    ground_truth = np.zeros((9,num_steps))
    for j in range(num_horizon+1):
        json_path_horizon = os.path.join(json_path, json_dir[num_steps*i+j])              # vector output for time step 0 ~ horizon
        with open(json_path_horizon) as f:
            json_horizon_data = json.load(f)
            ground_truth[0][j] = json_horizon_data[0]
            ground_truth[1][j] = json_horizon_data[1]
            ground_truth[2][j] = json_horizon_data[2]
            ground_truth[3][j] = json_horizon_data[3]
            ground_truth[4][j] = json_horizon_data[4]
            ground_truth[5][j] = json_horizon_data[5]
            ground_truth[6][j] = json_horizon_data[6]
            ground_truth[7][j] = json_horizon_data[7]
            ground_truth[8][j] = json_horizon_data[8]                                     
            # 9: [collision, location 3, velocity, yaw, angular velocity, latlong 2]
    df2 = pd.DataFrame(ground_truth)
    df2.to_csv('/home/lshi23/carla_test/data/ground_truth/ground_truth_%06d.csv' % i, index=False, header=False)
            
    print ("----%.0f%%----" % (100 * i/num_datapoints))
       