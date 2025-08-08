import numpy as np
import os
from loguru import logger
from tqdm import tqdm
from camera import Camera
from converter import LLtoUTM, UTMtoLL
from calibration import UAVCalibration


class Tracker:
    def __init__(self, camera: Camera, calibration: UAVCalibration):
        self.camera = camera
        self.calibration = calibration


    def pixel2NED(
            self,
            pixel_coord: np.ndarray,
            translation: np.ndarray,
            rotation: np.ndarray,
    ):
        """Run pixel localization from the UAV data. The translation and output is in NED frame.
        
        Assume body frame aligns with IMU frame in orientation.
        Pipeline: 
            1. Get ray in camera frame from pixel coordinates.
            2. Transform ray to IMU/body frame and NED frame.
            3. Get the altitude of the camera and get the scale.
            4. Align the ray using the scale in IMU/body frame.
            5. Target in body frame: t_cam2body + ray_imubodyframe
            6. Target in NED frame: GPS_in_NED + rotation@target_imubodyframe
        
        Args:
            pixel_coord (np.ndarray): Pixel coordinates in the image.
            translation (np.ndarray): Translation vector from the UAV pose, [northing, easting, altitude] in NED.
            rotation (np.ndarray): Rotation matrix from the UAV pose, IMU2NED.
        """
        pixel_coord = pixel_coord.reshape(-1, 2)  # Ensure pixel_coord is 2D
        ray_cameraFrame = self.camera.reproject(pixel_coord, fix_distortion=True)  # Reproject to get ray in camera frame
        
        ray_imubodyFrame = (self.calibration.R_ic @ ray_cameraFrame.reshape(-1, 1)).flatten()  # ray in IMU/body frame
        ray_nedFrame = (rotation @ ray_imubodyFrame.reshape(-1, 1)).flatten()  # ray in NED frame
        
        alt_cam = self.calibration.get_alt_cam(translation[-1], rotation)  # altitude of the camera in NED frame
        scale =  - alt_cam / ray_nedFrame[-1]  # scale factor to align the ray with the altitude, note alt_cam < 0
        
        ray_imubodyFrame *= scale  # scale the ray in IMU/body frame

        target_imubodyFrame = self.calibration.t_cam2body + ray_imubodyFrame  # target in IMU/body frame
        
        target_nedFrame = translation + (rotation @ target_imubodyFrame.reshape(-1, 1)).flatten()  # target in NED frame
        
        return target_nedFrame


    def pixel2GPS(
            self,
            pixel_coord: np.ndarray,
            translation: np.ndarray,
            rotation: np.ndarray,
            zone: str
    ):
        """Run pixel localization from the UAV data. The translation is in NED frame.
        
        Args:
            pixel_coord (np.ndarray): Pixel coordinates in the image.
            translation (np.ndarray): Translation vector from the UAV pose, [northing, easting, altitude] in NED.
            rotation (np.ndarray): Rotation matrix from the UAV pose, IMU2NED.
            zone (str): zone for UTM to LL.
        
        Return:
            lat (float): latitude
            long (float): longitude
        """
        ned_coord = self.pixel2NED(pixel_coord, translation, rotation)
        lat, long = UTMtoLL(23, ned_coord[0], ned_coord[1], zone)
        return lat, long










if __name__=='__main__':
    def extract_lat_lon_from_file(file_path):
        lat, lon = None, None
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('latitude:'):
                    lat = float(line.strip().split(':')[1])
                elif line.strip().startswith('longitude:'):
                    lon = float(line.strip().split(':')[1])
        return [lat, lon] if lat is not None and lon is not None else None

    def read_all_casualty_coords(directory):
        coords = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                lat_lon = extract_lat_lon_from_file(file_path)
                if lat_lon:
                    coords.append(lat_lon)
        return coords

    # Example usage:
    directory_path = '/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/GT'
    casualty_gps = np.array(read_all_casualty_coords(directory_path))
    casualty_coords = np.array([LLtoUTM(23, lat, lon) for lat, lon in casualty_gps])[:, 1:3].astype(np.float32)  # Extracting easting and northing
###################


    camera = Camera.load_config("camchain.yaml")
    logger.info(camera)
    
    ros_drone = np.load('logs/dr1_dect_pose2.npy', allow_pickle=True).tolist()
    
    calibration = UAVCalibration()
    

    target_coords = []
    
    theta = np.radians(-43)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    records = []
    tracker = Tracker(camera, calibration)
    for ros_read in tqdm(ros_drone):
        translation = ros_read['translation']
        # translation = np.array(translation) + np.array([0, 0, offset])
        rotation = ros_read['R_imu']
        rotation = Rz @ rotation # Align with NED frame
        detections = ros_read['detections']
        pixel_coords = []
        for det in detections:
            pixel_coords.append([
                (det[0] + det[2]) // 2,  # x center
                (det[1] + det[3]) // 2  # y center
            ])
            
        pixel_coords = np.array(pixel_coords).reshape(-1, 2)
        
        localization_this_frame = []
        for pixel_coord in pixel_coords:
            target_nedFrame = tracker.pixel2NED(
                pixel_coord,
                translation,
                rotation
            )
            target_coords.append(target_nedFrame)
            localization_this_frame.append(target_nedFrame)
            
        records.append({
            'translation': translation,
            'localization': np.array(localization_this_frame, dtype=np.float32)
        })
    np.save("logs/with_ts4.npy", target_coords)
    
    
    # save_heatmap_frames(
    #     records,
    #     save_path= '/mnt/UNENCRYPTED/ruichend/results/heatmap_frames'
    # )
    # save_video(
    #     output_vid='/mnt/UNENCRYPTED/ruichend/results/heatmap.mp4',
    #     src_path='/mnt/UNENCRYPTED/ruichend/results/heatmap_frames',
    #     fps=60,
    #     compress=True
    # )

    
            