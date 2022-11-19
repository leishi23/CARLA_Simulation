import glob
import os
import sys
import queue
import carla
import random
import numpy as np
import math
import json

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    # to delete the rgb images folder
    location = "/home/lshi23/carla_test/data"
    dir01 = "image/rgb_out"
    path01 = os.path.join(location, dir01)
    for file in os.listdir(path01):
        os.remove(os.path.join(path01, file))

    # to delete the json files folder
    dir02 = "raw_data"
    path02 = os.path.join(location, dir02)
    for f in os.listdir(path02):os.remove(os.path.join(path02, f))
    
    # to delete the action input folder
    dir03 = "action_input"
    path03 = os.path.join(location, dir03)
    for f in os.listdir(path03):os.remove(os.path.join(path03, f))
    
    # to delete the ground truth folder
    dir04 = "ground_truth"
    path04 = os.path.join(location, dir04)
    for f in os.listdir(path04):os.remove(os.path.join(path04, f))
    
    dir05 = "test_dataset/image/rgb_out"
    path05 = os.path.join(location, dir05)
    for f in os.listdir(path05):os.remove(os.path.join(path05, f))
    
    dir06 = "test_dataset/action_input"
    path06 = os.path.join(location, dir06)
    for f in os.listdir(path06):os.remove(os.path.join(path06, f))
    
    dir07 = "test_dataset/ground_truth"
    path07 = os.path.join(location, dir07)
    for f in os.listdir(path07):os.remove(os.path.join(path07, f))

finally:
    pass

# ensure the data folder is empty
assert os.listdir(path01) == [] , "The rgb folder is empty"
assert os.listdir(path02) == [] , "The json folder is empty"
assert os.listdir(path03) == [] , "The action_input folder is empty"
assert os.listdir(path04) == [] , "The ground_truth folder is empty"


def process_img(data, rgb_queue):
    rgb_queue.put(data)
    # print('rgb time is %s' % data.timestamp)
    # print('rgb frame is %s' % data.frame)

def process_imu(data, imu_angular_vel_queue):
    imu_angular_vel_queue.put(data.gyroscope.z)
    # print('imu time is %s' % data.timestamp)
    # print('imu frame is %s' % data.frame) 
    
def process_gnss(data, gnss_queue):
    gnss_queue.put(data)
    # print('gnss time is %s' % data.timestamp)
    # print('gnss frame is %s' % data.frame)

def process_lidar(data, lidar_queue):
    lidar_queue.put(data)
    # print('lidar time is %s' % data.timestamp)
    # print('lidar frame is %s' % data.frame)
    
def process_dp(data, dp_queue):
    dp_queue.put(data)
    # print('dp time is %s' % data.timestamp)
    # print('dp frame is %s' % data.frame)

try:
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    client.load_world('formal')
    # client.start_recorder("/home/carla/recording01.log")
    IM_WIDTH = 128
    IM_HEIGHT = 96             

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.25  # Sets the fixed time step 0.05 by default
    # settings.substepping = True
    # settings.max_substep_delta_time = 0.01
    # settings.max_substeps = 10
    world.apply_settings(settings)
    
    # Set up the traffic manager
    traffic_manager = client.get_trafficmanager()    
    traffic_manager.set_synchronous_mode(True)
    
    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(0)
    
    actor_list = []

    # Get the blueprint library and filter for the vehicle blueprints
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')

    # spawn points for vehicles
    spawn_points = world.get_map().get_spawn_points()
    # print('spawn points are %s' % spawn_points)
    # ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
    ego_bp = world.get_blueprint_library().find('vehicle.audi.tt')
    # ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
    ego_vehicle = world.spawn_actor(ego_bp, carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0)))
    actor_list.append(ego_vehicle)
    print('created ego_%s' % ego_vehicle.type_id)

    # for _ in range(0, 10):
    #     # This time we are using try_spawn_actor. If the spot is already
    #     # occupied by another object, the function will return None.
    #     npc = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
    #     if npc is not None:
    #         actor_list.append(npc)
    #         npc.set_autopilot(True)
    
    # ego_vehicle.set_autopilot(True)

    # Create a transform to place the camera on top of the vehicle
    camera_transform = carla.Transform(carla.Location(x=0.5, z=2.5))

    # We create sensors through a blueprint that defines its properties
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")

    imu_bp = world.get_blueprint_library().find('sensor.other.imu')
    gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')

    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', str(32))
    lidar_bp.set_attribute('points_per_second', str(90000))
    lidar_bp.set_attribute('rotation_frequency', str(40))
    lidar_bp.set_attribute('range', str(20))
    lidar_bp.set_attribute('lower_fov', str(-25))
    lidar_location = carla.Location(0, 0, 2)
    lidar_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)

    # We spawn the camera and attach it to our ego vehicle
    camera01 = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    gnss = world.spawn_actor(gnss_bp, camera_transform, attach_to=ego_vehicle)
    IMU = world.spawn_actor(imu_bp, camera_transform, attach_to=ego_vehicle)
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    

    actor_list.append(camera01)
    actor_list.append(IMU)
    actor_list.append(gnss)
    actor_list.append(lidar)

    # print('created %s' % camera01.type_id)
    # print('created %s' % IMU.type_id)
    # print('created %s' % gnss.type_id)
    # print('created %s' % lidar.type_id)


    # The sensor data will be saved in thread-safe Queues
    rgb_image_queue = queue.Queue(maxsize=1)   
    imu_angular_vel_queue = queue.Queue(maxsize=1)   
    gnss_queue = queue.Queue(maxsize=1)    
    lidar_queue = queue.Queue(maxsize=1) 
    
    
    # Below is from actor get() function
    location_queue = queue.Queue(maxsize=1)
    vel_queue = queue.Queue(maxsize=1)
    yaw_queue = queue.Queue(maxsize=1)

    camera01.listen(lambda data: process_img(data, rgb_image_queue))
    IMU.listen(lambda data: process_imu(data, imu_angular_vel_queue))
    gnss.listen(lambda data: process_gnss(data, gnss_queue))
    lidar.listen(lambda data: process_lidar(data, lidar_queue))
    steer = 0.0
    
    # set the spectator
    spectator = world.get_spectator()
    spectator.set_transform(
    carla.Transform(ego_vehicle.get_location() + carla.Location(z=10), carla.Rotation(pitch=-90)))

    # set the time counter
    time_counter = 0

    while world is not None:
        
        # to add some noise to foward velocity and steering angle
        forward_velocity = np.random.normal(3, 1)
        ego_vehicle.enable_constant_velocity(carla.Vector3D(x=forward_velocity,y=0,z=0))
        
        steer += np.random.normal(0, 0.075)
        if steer > 1.0:
            steer = 1.0
        if steer < -1.0:
            steer = -1.0
        # print('steer is %s' % steer)
        ego_vehicle.apply_control(carla.VehicleControl(steer=steer))
        
        # to add noise on autopilot mode angular velocity
        # control = ego_vehicle.get_control()
        # control.steer += float(np.random.normal(0, 0.01))
        # ego_vehicle.apply_control(control)
        
        # Use the actor get() 
        location_queue.put(ego_vehicle.get_location())
        vel_queue.put(ego_vehicle.get_velocity())
        ego_transform = ego_vehicle.get_transform()
        yaw = ego_transform.rotation.yaw
        yaw_queue.put(yaw)
        
        world.tick()
        
        # get the frame 
        world_snapshot = world.get_snapshot()
        frame = world_snapshot.frame
        time_counter += 0.25
        
        snapshot = world.get_snapshot()
        delta_seconds = snapshot.timestamp.delta_seconds
        elapsed_seconds = snapshot.timestamp.elapsed_seconds
        # print('elapsed_seconds is %s seconds' % elapsed_seconds)
        # print('delta_seconds  is %s seconds' % delta_seconds)

        try:
            # Get the data once it's received.
            image_data = rgb_image_queue.get()
            z_axis_angular_vel_data = imu_angular_vel_queue.get()       # rad/s
            gnss_data = gnss_queue.get()
            location_data = location_queue.get()    # float 
            vel_data = vel_queue.get()      # float
            lidar_data = lidar_queue.get()
            yaw_data = yaw_queue.get()      # float
            yaw_data_radians = yaw_data * np.pi / 180
            
        except queue.Empty:
            print("[Warning] Some sensor data has been missed")
            continue
        
        # to convert the fixed frame velocity to body frame and get the forward speed
        yaw_global = np.radians(yaw_data)
        rotation_global = np.array([
            [np.cos(yaw_global), -np.sin(yaw_global)],
            [np.sin(yaw_global), np.cos(yaw_global)]
        ])
        vel_global = np.array([vel_data.y, vel_data.x])
        vel_local = np.matmul(rotation_global, vel_global)
        vel_local = vel_local[1]
        
        # print('vel_local is %s' % vel_local)
        # print('angular_vel is %s' % z_axis_angular_vel_data)

        if time_counter > 5:  # frame 6 is the first frame with position data lost, so we start from frame 10
            image_data.save_to_disk('/home/lshi23/carla_test/data/image/rgb_out/%06d.jpg' % image_data.frame)
        
        # to get the nearest obstacle distance and angle from lidar raw data 
        distance = []
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
        # Point cloud in lidar sensor space array of shape (p_cloud_size, 3).
        local_lidar_points = np.array(p_cloud[:, :3])
        for i in range(p_cloud_size):
            temp = pow(pow(local_lidar_points[i][0], 2) + pow(local_lidar_points[i][1], 2)
                       + pow(local_lidar_points[i][2], 2), 0.5)
            distance.append(temp)
            
        if len(distance) == 0:
            nearest_dist = 10
        else:
            nearest_dist = min(distance)    # nearest distance is numpy.float64
            
        if nearest_dist < 3:
            collision = 1
        else:
            collision = 0
            
        # min_idx = distance.index(nearest_dist)
        # min_x = local_lidar_points[min_idx][0]
        # min_y = local_lidar_points[min_idx][1]
        # angle = math.atan2(min_y, min_x)
        # angle = math.degrees(angle)     # angle is float 
        
        latitude = gnss_data.latitude   # float 
        longitude = gnss_data.longitude # float 
       
        # 1*9 dimension 
        data_timestamp = [collision, location_data.x, location_data.y, location_data.z, vel_local, yaw_data_radians, z_axis_angular_vel_data, latitude, longitude]
        
        # to save the data_timestamp in disk by jason file format 
        json_object = json.dumps(data_timestamp, indent=4)  
        if time_counter > 5:          # to avoid the first few frames  
            with open("/home/lshi23/carla_test/data/raw_data/%06d.json" % frame, "w") as outfile:outfile.write(json_object)
        
        if collision == 1:
            random_reset_location = np.random.rand(2)*160-80      # to reset the ego vehicle to a random location, [-80, 80]
            ego_vehicle.set_transform(carla.Transform(location = carla.Location(x=random_reset_location[0], y=random_reset_location[1], z=0.0), rotation = carla.Rotation(pitch=0.0, yaw= float(np.random.rand(1)*360), roll=0.0)))
            time_counter = 0
            
        if time_counter == 90:
            random_reset_location = np.random.rand(2)*160-80      # to reset the ego vehicle to a random location, [-80, 80]
            print('random_reset_location is %s' % random_reset_location)
            ego_vehicle.set_transform(carla.Transform(location = carla.Location(x=random_reset_location[0], y=random_reset_location[1], z=0.0), rotation = carla.Rotation(pitch=0.0, yaw= float(np.random.rand(1)*360), roll=0.0)))
            time_counter = 0
                        
        # print('yaw is %s' % yaw_data)
        # print('location x is %s' % location_data.x)
        # print('location y is %s' % location_data.y)
        # if frame%10 == 0:
        #     print('Collected data number is % s' % frame)


        
finally:
    print("Over")