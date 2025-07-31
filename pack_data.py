"""Not updated yet, debug AprilTag first."""

import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
from loguru import logger
from tqdm import tqdm
import shutil


from camera import Camera
from tracking_utils import box_center, set_device
from rosbag_utils import bundeled_data_from_bag, image_stream, camera_config
from viz_utils import save_video
from apriltag_utils import AprilTagDetector
import argparse



class UAVCalibration:
    """Calibration parameters for the UAV.
    
    Body frame is in the same orientation as  the IMU frame but with an offset.
    """
    def __init__(self):
        self.T_ci = np.array([
            [ 0.011106298412152327,  0.9999324199187616,  0.0034359468839849595, 0.036802732375442404],
            [-0.999832733821092,  0.01115499451474039,  -0.014493808237225339,  -0.008332238900780303],
            [-0.014531156713131006 , -0.0032743996068676073, 0.9998890557415823,  -0.08775357009176091],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
        ])
        self.R_ci = self.T_ci[:3, :3]
        self.R_ic = self.R_ci.T  # Rotation: camera to IMU
        self.t_imu2cam = self.T_ci[:3, 3]  # Translation from IMU to camera in camera frame
        self.t_body2imu = np.array([-0.0389588892, 0, -0.2796108098])  # body to IMU in IMU frame
        self.t_imu2body = -self.t_body2imu  # IMU to body in body frame
        # self.t_imu2body = np.array([11.0, 0.0, 26.0])  # IMU to body in body frame
        self.t_body2cam = (self.T_ci @ np.hstack((self.t_body2imu, [1])).reshape(-1, 1)).flatten()[:3]  # body to camera in camera frame
        self.t_body2cam_imu = (self.R_ic @ self.t_body2cam.reshape(-1, 1)).flatten()  # body to camera in IMU frame
        # self.t_cam2body = -(self.R_ic @ self.t_imu2cam.reshape(-1, 1)).flatten()  # camera to body in IMU/body frame
        self.t_cam2body = -self.t_body2cam_imu
        
    def get_alt_cam(self, alt_body, R_wi):
        """
        Get the altitude of the camera in the world frame.
        :param alt_body: Altitude in body frame.
        :param R_wi: Rotation from world to IMU frame.
        :return: Altitude of the camera in the world frame.
        """
        alt_offset = (R_wi @ self.t_body2cam_imu.reshape(-1, 1)).flatten()[-1]
        assert alt_offset <= 0, "Drone upside down?"
        alt_cam = alt_body - (R_wi @ self.t_body2cam_imu.reshape(-1, 1)).flatten()[-1]
        return alt_cam


if __name__ == '__main__':
    #############################################################################
    bag_pth = "/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/dry_run_1.bag"
    #############################################################################


    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')

    device = set_device()

    cam = Camera.load_config('camchain.yaml')
    logger.info(cam)

    logger.info("Loading YOLO model...")
    ckpt = 'checkpoints/11x_ft.pt'
    model = YOLO(ckpt)
    model.to(device)
    
    calib = UAVCalibration()
    i = 0
    confidence_threshold = 0.67
    logger.info(f"YOLO confidence threshold: {confidence_threshold}")
    
    
    coords = []
    detections = []
    pose_records = []
    for ts, frame, translation, R_wi, zone in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=True, 
        use_pose_rot = False, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    ):
        
        
        result = model.predict(frame, verbose=False, conf=confidence_threshold)[0]

        frame_detections = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            img_coord = box_center(box, center=False)

            frame_detections.append(xyxy.tolist())
        
        if frame_detections == []:
            continue
        pose_records.append({
            "timestamp":ts,
            "translation": translation,
            "R_imu": R_wi,
            "detections": frame_detections
        })

    np.save('logs/dr1_dect_pose2.npy', pose_records)
