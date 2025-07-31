"""Test code for project world -> camera."""

import cv2
import numpy as np
from rosbag_utils import image_stream
from viz_utils import compress_vid
from camera import Camera
import os

from loguru import logger
import shutil

if __name__ == "__main__":
        
    north_ned = np.array([1, 0, 1])
    east_ned = np.array([0, 1, 1])
    down_ned = np.array([0, 0, 2])
    

    arrow_length = 0.7

    north_color = (0, 255, 0)  # green for north
    east_color = (255, 0, 255) # megenta for east
    down_color = (0, 0, 255)   # red for down
    mag_color = (0, 0, 0)  # black for magnetic field
    thickness = 2

    
    
    
    j = 3
    #############################################################################
    # bag_pth = '/mnt/ENCRYPTED/workshop2/20250311/course-1/dione/course_1.bag'
    bag_pth = f'/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/dry_run_1.bag'
    save_frames_to = f'/mnt/UNENCRYPTED/ruichend/results/dry_run_1'
    save_vid = True
    # vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_april.mp4'
    vid_pth = 'ri_m.mp4'
    #############################################################################

    if os.path.isdir(save_frames_to):
        shutil.rmtree(save_frames_to)
        logger.info(f"Removed previous frames: {save_frames_to}")
    os.mkdir(save_frames_to)

    if save_vid and os.path.isfile(vid_pth):
            os.remove(vid_pth)

    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')
    
    cam = Camera.load_config('camchain.yaml')
    

    
    Rz_90 = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    from localization import UAVCalibration
    calib = UAVCalibration()

    writer = None
    for ts, frame, translation, imu_R, mav_R, zonem, magnetic_field in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=True, 
        use_pose_rot = True, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic,
        magnetic_field_topic='/imu/magnetic_field'
    ):

        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter("temp.mp4", fourcc, 20, (width, height))
        
        
        R_iw = imu_R.T
        
        magnetic_field /= np.linalg.norm(magnetic_field)

        
        R_cw = calib.R_ci @ R_iw

        # Transform NED vectors from world to camera frame
        cam_origin = np.array([0, 0, 1]) # origin in cam frame
        north_cam = R_cw @ (arrow_length * north_ned).reshape(-1, 1)
        east_cam = R_cw @ (arrow_length * east_ned).reshape(-1, 1)
        down_cam = R_cw @ (arrow_length * down_ned).reshape(-1, 1)
        mag_cam = 0.5 * calib.R_ci @ magnetic_field.reshape(-1, 1)
        
        north_cam = north_cam.flatten()
        east_cam = east_cam.flatten()
        down_cam = down_cam.flatten()
        mag_cam = mag_cam.flatten()

        # Project to image space
        north_img = cam.project(north_cam)[0].astype(int)
        east_img = cam.project(east_cam)[0].astype(int)
        down_img = cam.project(down_cam)[0].astype(int)
        mag_img = cam.project(mag_cam)[0].astype(int)
        origin_img = cam.project(cam_origin)[0].astype(int)

        # Draw arrows
        cv2.arrowedLine(frame, tuple(origin_img), tuple(north_img), north_color, thickness, tipLength=0.1)
        cv2.arrowedLine(frame, tuple(origin_img), tuple(east_img), east_color, thickness, tipLength=0.1)
        cv2.arrowedLine(frame, tuple(origin_img), tuple(down_img), down_color, thickness, tipLength=0.1)
        cv2.arrowedLine(frame, tuple(origin_img), tuple(mag_img), mag_color, 4, tipLength=0.1)
        writer.write(frame)
        
    if writer:
        writer.release()
        compress_vid("temp.mp4", vid_pth)
        
        