"""Plot the rotation matrix."""

import numpy as np
import cv2
import os
import shutil
import rosbag


from camera import Camera
from tracking_utils import box_center, set_device, quaternions_to_SO3
from rosbag_utils import bundeled_data_from_bag, image_stream, camera_config, read_pose
from viz_utils import save_video
from converter import LLtoUTM, UTMtoLL
from apriltag_utils import AprilTagDetector

import numpy as np
import matplotlib.pyplot as plt




def plot_rotation_matrix(R, filename='test.png'):
    """
    Plots the world frame and a rotated frame defined by rotation matrix R
    in 4 views (3D, XY, XZ, YZ) and saves as a PNG image.
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
        filename (str): filename to save the figure
    """
    origin = np.zeros(3)
    world_axes = np.eye(3)
    rotated_axes = R

    # Helper to plot one frame
    def plot_frame(ax, origin, axes, style='solid', alpha=1.0):
        colors = ['r', 'g', 'b']
        for i in range(3):
            ax.quiver(*origin, *axes[:,i], color=colors[i], 
                      length=1.0, normalize=True, linestyle=style, alpha=alpha)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Define views: (subplot position, elev, azim, title)
    views = [
        (1, 30, 45, '3D View'),
        (2, 90, -90, 'XY View (Top Down)'),
        (3, 0, -90, 'XZ View (Side)'),
        (4, 0, 0, 'YZ View (Front)')
    ]

    for pos, elev, azim, title in views:
        ax = fig.add_subplot(2, 2, pos, projection='3d')
        # Plot world frame
        plot_frame(ax, origin, world_axes, style='solid', alpha=0.8)
        # Plot rotated frame
        plot_frame(ax, origin, rotated_axes, style='dashed', alpha=0.8)

        # Axes settings
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved rotation visualization to {filename}")
    
    
if __name__ == "__main__":
    # Example rotation matrix: 45-degree rotation about Z axis
    bag_pth = '/mnt/UNENCRYPTED/ruichend/seq/seq3/seq_3.bag'

    pose_topic = '/mavros/local_position/pose'
    imu_topic = '/imu/imu'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    
    bag = rosbag.Bag(bag_pth)
    # pose_timestamps, translation, quat = read_pose(bag, pose_topic)
    
    q1 = np.array([0.09980092942714691, -0.060778893530368805, -0.3739680051803589, 0.9200509190559387])
    q2 = np.array([0.001736475224499148, 0.0031054926975067378, 0.860854027525059, -0.5088395658848877])
    rotation = quaternions_to_SO3(q1)
    
    T_ci = np.array([
            [ 0.99961803, -0.01195821, -0.0249158,   0.01737192],
            [ 0.01171022, 0.99988067, -0.01007557, -0.01208277],
            [ 0.02503332,  0.00977995,  0.99963878, -0.05170631],
            [ 0.,          0.,          0.,          1.        ]
        ])

    R_ci = T_ci[:3, :3]
    
    R = rotation @ R_ci.T
    
    
    theta = np.radians(-90)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    Rx_180 = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    Ry_180 = np.array([
        [-1,  0,  0],
        [0, 1,  0],
        [0,  0, -1]
    ])
    
    R_bc = np.array([
        [0.995,  0,  -0.099],
        [0.099, 0,  0.995],
        [0,  1, 0]
    ])
    
    print(R_bc@R_bc.T)

    plot_rotation_matrix(R_bc, 'Rbc.png')
