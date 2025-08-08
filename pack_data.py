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

from calibration import UAVCalibration


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

    np.save('logs/dr1_dect_pose3.npy', pose_records)
