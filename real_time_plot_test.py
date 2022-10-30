import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pylab as pl
import glob
import os
import shutil
import sys
import numpy as np
import queue
import datetime
import carla
import random
import randimage


####################################### Lidar ######################################
# dir_path = "/home/lshi23/research_course_code/rgb_out/*.*"
# res = glob.glob(dir_path) # get the image path
# res.sort()
# img = None
# for i in res:
#     image_path = i
#     im = mpimg.imread(image_path)
#     if img is None:
#         img = pl.imshow(im)
#     else:
#         img.set_data(im)
#     pl.pause(.05)
#     pl.draw


####################################### Depth ######################################
# dir_path = "/home/lshi23/research_course_code/depth_out/*.*"
# res = glob.glob(dir_path) # get the image path
# res.sort()
# img = None
# for i in res:
#     image_path = i
#     im = mpimg.imread(image_path)
#     if img is None:
#         img = pl.imshow(im,cmap='gray')
#     else:        
#         img.set_data(im)
#     pl.pause(.05)
#     pl.draw


################################### ColorConverter ##################################
DP_IM_WIDTH = 800
DP_IM_HEIGHT = 600
fig_dp, ax_dp = plt.subplots()
dp_array = np.random.randint(0, 100, size=(DP_IM_HEIGHT, DP_IM_WIDTH), dtype=np.uint8)
l_dp = ax_dp.imshow(dp_array, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
ax_dp.set_title('Depth')
img_size = (DP_IM_HEIGHT, DP_IM_WIDTH)


for i in range(100):

    dp_array = randimage.get_random_image(img_size)
    dp_array = np.reshape(dp_array, (DP_IM_HEIGHT, DP_IM_WIDTH, 3))
    dp_array = dp_array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(dp_array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)

    # Convert to logarithmic depth.
    logdepth = np.ones(normalized_depth.shape) + \
        (np.log(normalized_depth) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    # logdepth *= 255.0
    # Expand to three colors.
    dp_array = np.repeat(logdepth[:, :, np.newaxis], 3, axis=2)
    # dp_array.astype(np.uint8)
    l_dp.set_data(dp_array)
    
    plt.pause(0.001)
    print(i)


