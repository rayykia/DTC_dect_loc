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



def pixel_localization(
        camera: Camera,
        pixel_coord: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        calibration: UAVCalibration
):
    """Run pixel localization from the UAV data.
    
    Assume body frame aligns with IMU frame in orientation.
    Pipeline: 
        1. Get ray in camera frame from pixel coordinates.
        2. Transform ray to IMU/body frame and NED frame.
        3. Get the altitude of the camera and get the scale.
        4. Align the ray using the scale in IMU/body frame.
        5. Target in body frame: t_cam2body + ray_imubodyframe
        6. Target in NED frame: GPS_in_NED + rotation@target_imubodyframe
    
    Args:
        camera (Camera): Camera object with calibration parameters.
        pixel_coord (np.ndarray): Pixel coordinates in the image.
        translation (np.ndarray): Translation vector from the UAV pose, [northing, easting, altitude] in NED.
        rotation (np.ndarray): Rotation matrix from the UAV pose, IMU2NED.
        calibration (UAVCalibration): Calibration parameters for the UAV.
    """
    
    # 1. Get ray in camera frame from pixel coordinates.
    pixel_coord = pixel_coord.reshape(-1, 2)  # Ensure pixel_coord is 2D
    ray_cameraFrame = camera.reproject(pixel_coord, fix_distortion=True)  # Reproject to get ray in camera frame
    
    # 2. Transform ray to IMU/body frame and NED frame.
    ray_imubodyFrame = (calibration.R_ic @ ray_cameraFrame.reshape(-1, 1)).flatten()  # ray in IMU/body frame
    ray_nedFrame = (rotation @ ray_imubodyFrame.reshape(-1, 1)).flatten()  # ray in NED frame
    
    # 3. Get the altitude of the camera and get the scale.
    alt_cam = calibration.get_alt_cam(translation[-1], rotation)  # altitude of the camera in NED frame
    scale =  - alt_cam / ray_nedFrame[-1]  # scale factor to align the ray with the altitude, note alt_cam < 0
    
    # 4. Align the ray using the scale in IMU/body frame.
    ray_imubodyFrame *= scale  # scale the ray in IMU/body frame
    
    # 5. Target in body frame: t_cam2body + ray_imubodyframe
    target_imubodyFrame = calibration.t_cam2body + ray_imubodyFrame  # target in IMU/body frame
    
    # 6. Target in NED frame: GPS_in_NED + rotation@target_imubodyframe
    target_nedFrame = translation + (rotation @ target_imubodyFrame.reshape(-1, 1)).flatten()  # target in NED frame
    
    return target_nedFrame
    
    
if __name__ == '__main__':
    camera = Camera.load_config("camchain.yaml")
    logger.info(camera)
    
    ros_drone = np.load('logs/dr1_dect_pose2.npy', allow_pickle=True).tolist()
    
    calibration = UAVCalibration()
    target_coords = []
    for ros_read in tqdm(ros_drone):
        translation = ros_read['translation']
        rotation = ros_read['R_imu']
        detections = ros_read['detections']
        pixel_coords = []
        for det in detections:
            pixel_coords.append([
                det[0] + det[2] // 2,  # x center
                det[1] + det[3] // 2  # y center
            ])
            
        pixel_coords = np.array(pixel_coords).reshape(-1, 2)
        
        for pixel_coord in pixel_coords:
            target_nedFrame = pixel_localization(
                camera,
                pixel_coord,
                translation,
                rotation,
                calibration
            )
            target_coords.append(target_nedFrame)
            
    np.save("logs/with_ts2.npy", target_coords)
    
            