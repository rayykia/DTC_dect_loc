"""Test code for project world -> camera."""

import cv2
import numpy as np
from rosbag_utils import image_stream
from viz_utils import compress_vid
from camera import Camera
import os

from loguru import logger

if __name__ == "__main__":
        
    north_ned = np.array([1, 0, 1])
    east_ned = np.array([0, 1, 1])
    down_ned = np.array([0, 0, 2])

    arrow_length = 0.7

    north_color = (0, 255, 0)  # green for north
    east_color = (255, 0, 255) # megenta for east
    down_color = (0, 0, 255)   # red for down
    thickness = 2

    
    
    
    j = 3
    #############################################################################
    # bag_pth = '/mnt/ENCRYPTED/workshop2/20250311/course-1/dione/course_1.bag'
    bag_pth = f'/mnt/UNENCRYPTED/ruichend/seq/seq{j}/seq_{j}.bag'
    save_frames_to = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_april'
    save_vid = False
    # vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_april.mp4'
    vid_pth = 'ridT.mp4'
    #############################################################################


    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')
    
    cam = Camera.load_config('camchain.yaml')
    
    t_cam2body = np.array([1.297, -1.282, -0.150])
    t_body2cam = np.array([1.30, 0.15, -1.30])
    R_bc = np.array([
        [0.995,  0,  -0.099],
        [0.099, 0,  0.995],
        [0,  1, 0]
    ])
    
    T_ci = np.array([
            [ 0.99961803, -0.01195821, -0.0249158,   0.01737192],
            [ 0.01171022, 0.99988067, -0.01007557, -0.01208277],
            [ 0.02503332,  0.00977995,  0.99963878, -0.05170631],
            [ 0.,          0.,          0.,          1.        ]
            ])
    R_ci = T_ci[:3, :3]
    t_imu2cam = T_ci[:3, 3]
    
    t_body2imu_camera = t_body2cam - t_imu2cam
    t_body2imu = R_ci.T @ t_body2imu_camera

    logger.info(f"t_body2imu: {t_body2imu}")
    
    t_cam2imu = np.array([-0.01592941, 0.01279475, 0.05199873])
    
    writer = None
    
    Rz_90 = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    R_dis = []
    for ts, frame, translation, imu_R, mav_R, zone in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=True, 
        use_pose_rot = True, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    ):

        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter("temp.mp4", fourcc, 20, (width, height))
        
        R_cw = mav_R @ imu_R

        # Transform NED vectors from world to camera frame
        cam_origin = np.array([0, 0, 1]) # origin in cam frame
        north_cam = R_cw @ (arrow_length * north_ned)
        east_cam = R_cw @ (arrow_length * east_ned)
        down_cam = R_cw @ (arrow_length * down_ned)

        # Project to image space
        north_img = cam.project(north_cam)[0].astype(int)
        east_img = cam.project(east_cam)[0].astype(int)
        down_img = cam.project(down_cam)[0].astype(int)
        origin_img = cam.project(cam_origin)[0].astype(int)

        # Draw arrows
        cv2.arrowedLine(frame, tuple(origin_img), tuple(north_img), north_color, thickness, tipLength=0.3)
        cv2.arrowedLine(frame, tuple(origin_img), tuple(east_img), east_color, thickness, tipLength=0.3)
        cv2.arrowedLine(frame, tuple(origin_img), tuple(down_img), down_color, thickness, tipLength=0.3)
        writer.write(frame)
        
    if writer:
        writer.release()
        compress_vid("temp.mp4", vid_pth)
        
        