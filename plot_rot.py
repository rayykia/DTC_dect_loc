import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
import shutil
from loguru import logger
from tqdm import tqdm


from camera import Camera
from tracking_utils import box_center, set_device, quaternions_to_SO3
from rosbag_utils import bundeled_data_from_bag, image_stream, camera_config, read_pose
from viz_utils import save_video
from converter import LLtoUTM, UTMtoLL
from apriltag_utils import AprilTagDetector

import argparse


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_camera_poses_topdown_colored_from_separate(translations, rotations, colormap_name='viridis'):
    n = len(translations)

    # Version-agnostic colormap
    try:
        cmap = cm.colormaps[colormap_name]
    except AttributeError:
        cmap = cm.get_cmap(colormap_name)

    norm = mcolors.Normalize(vmin=0, vmax=n-1)
    colors = [cmap(norm(i)) for i in range(n)]

    # Prepare quiver data
    xs, ys, dxs, dys, cs = [], [], [], [], []
    scale = 0.5

    for i, (t, R) in enumerate(zip(translations, rotations)):
        local_dir = np.array([0, -1, 0])
        world_dir = R @ local_dir
        world_dir = np.asarray(world_dir).flatten()

        x, y = t[0], t[1]
        dx, dy = world_dir[0], world_dir[1]

        norm_dir = np.linalg.norm([dx, dy])
        if norm_dir > 0:
            dx, dy = dx / norm_dir, dy / norm_dir

        xs.append(x)
        ys.append(y)
        dxs.append(dx)
        dys.append(dy)
        cs.append(i)

    xs = np.array(xs)
    ys = np.array(ys)
    dxs = np.array(dxs) * scale
    dys = np.array(dys) * scale
    cs = np.array(cs)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Quiver plot with color mapping
    q = ax.quiver(xs, ys, dxs, dys, cs, angles='xy', scale_units='xy', scale=1,
                  cmap=cmap, norm=norm)

    # Add colorbar
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label('Time (frame index)')
    cbar.set_ticks([0, n-1])
    cbar.set_ticklabels(['start', 'end'])

    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_title('Top-down view of camera -Y directions with time coloring')
    ax.grid()
    plt.savefig('rot2.png', dpi=300)  
    
if __name__ == '__main__':

    j = 3
    bag_pth = '/mnt/UNENCRYPTED/ruichend/seq/seq3/seq_3.bag'

    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    
    import rosbag
    bag = rosbag.Bag(bag_pth)
    pose_timestamps, translation, quat = read_pose(bag, pose_topic)
    rotation = quaternions_to_SO3(quat)
    
    Ric = np.array([
        [0,  -1,  0],
        [-1, 0,  0],
        [0,  0, -1]
    ])
    # zs = []
    # for rot in rotation:
    #     rot = Rx_180 @ rot
    #     z = rot @ np.array([[0], [0], [1]])
    #     zs.append(z[2])
    # plt.plot(pose_timestamps, zs)
    # plt.savefig('z.png', dpi=300)
    rotation = np.array([rot@Ric for rot in rotation])  # Convert to world frame
    plot_camera_poses_topdown_colored_from_separate(translation[::20], rotation[::20], colormap_name='plasma')