"""Localization of the AprilTag."""
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
import shutil
from loguru import logger
from tqdm import tqdm


from camera import Camera
from tracking_utils import box_center, set_device
from rosbag_utils import bundeled_data_from_bag, image_stream, camera_config
from viz_utils import save_video
from converter import LLtoUTM, UTMtoLL
from apriltag_utils import AprilTagDetector

import argparse

if __name__ == '__main__':

    j = 3
    #############################################################################
    # bag_pth = '/mnt/ENCRYPTED/workshop2/20250311/course-1/dione/course_1.bag'
    bag_pth = f'/mnt/UNENCRYPTED/ruichend/seq/seq{j}/seq_{j}.bag'
    save_frames_to = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_april'
    save_vid = False
    vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_april.mp4'
    #############################################################################


    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')

    # data_bundel = bundeled_data_from_bag(bag_pth, frame_topic, pose_topic)

    logger.info('Done.')

    device = set_device()
    

    logger.info("Loading Apriltag Detector...")
    model = AprilTagDetector()
        
        
    cam = Camera.load_config('camchain.yaml')
    logger.info(cam)


    box_color = (255, 0, 0)  # Red
    thickness = 5  # Thicker box
    
    logger.info(f"Saving frames to `{save_frames_to}`.")
    if not os.path.isdir(save_frames_to):
        logger.info(f"`{save_frames_to}` does not exist, making...")
        os.mkdir(save_frames_to)



    i = 0
    coords = []
    # Ric = np.array([
    #     [0,  -1,  0],
    #     [-1, 0,  0],
    #     [0,  0, -1]
    # ])
    
    # Ry_180 = np.array([
    #     [-1,  0,  0],
    #     [0, 1,  0],
    #     [0,  0, -1]
    # ])
    

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
    t_imu2body = R_bc @ (-t_body2imu_camera).reshape(-1, 1)

    logger.info(f"t_body2imu: {t_body2imu}")
    
    t_cam2imu = np.array([-0.01592941, 0.01279475, 0.05199873])
    
    T_bi = np.identity(4)
    R_bi = R_bc.T @ R_ci.T
    T_bi[:3, :3] = R_bi
    T_bi[:3, 3] = t_body2imu    
    
    
    for ts, frame, translation, R_wi, zone in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=True, 
        use_pose_rot = False, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    ):
        R_wc =  R_wi @ R_ci.T
        # R_wc = R_dw.T @ R_dc
        result = model(frame)
        
        # bias = R_wi.T @ t_body2imu.reshape(-1, 1)
        # bias = np.array([0, 0, 0])
        
        
        R_wb = R_wi @ R_bi.T
        imu_coord_world = translation + (R_wb @ t_imu2body.reshape(-1, 1)).flatten()
        translation = imu_coord_world

        for img_coord in result:
            img_coord = img_coord.reshape(-1, 2)
            
            print(f"Image Coord: {img_coord.flatten()}")
            
            
            # long_uav, lat_uav, alt_uav = translation
            # zone, easting, northing = LLtoUTM(23, lat_uav, long_uav)
            northing, easting, alt_uav = translation
            ray_cam = (cam.reproject(img_coord)).reshape(-1, 1)
            ray_world = R_wc @ (ray_cam)
            
            # bias = R_iw.T @ t_cam2imu.reshape(-1, 1)
            # bias = np.array([0, 0, 0])
            
            
            ray_world = ray_world.flatten()
            s = -(translation[2] / ray_world[2])
            
            world_coord = np.array([northing, easting, 0]).flatten() + s * ray_world

            print(f"s: {s}")
            print(f"Ray Cam: {ray_cam.flatten()}")
            print(f"Ray World: {ray_world.flatten()}")
            # print(f"bias: {bias.flatten()}")
            _, gt_e, gt_n = LLtoUTM(23, 39.9411551, -75.1987216)
            print(f"Northing: {[northing, easting, alt_uav]}")
            print(f"World Coord: {[world_coord[0], world_coord[1], world_coord[2]]}")
            print(f"GT: {[gt_n, gt_e, -24.927169916112156]}", end='\n\n')
            
            offset = np.array([northing, easting, alt_uav]) - np.array([gt_n, gt_e, 0])

            print(f"offset: {np.round(offset, 2)}")
            print(f"offset + s * ray_world: {np.round(offset + s * ray_world, 2)}")
            print("======================================================")


            

            lat, long = UTMtoLL(23, world_coord[0], world_coord[1], zone)

            coords.append([long, lat, world_coord[-1]])
            label = f"({(world_coord[0] - gt_n):.2f}, {(world_coord[1] - gt_e):.2f}, {world_coord[2]:.2f})"

            img_coord = tuple(img_coord[0].astype(int))
            cv2.circle(frame, img_coord, color = box_color, radius=thickness)
            # Prepare label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame,
                        (img_coord[0], img_coord[1] - text_h - 10),
                        (img_coord[0] + text_w, img_coord[1]),
                        box_color, -1)
            # Draw text label on top
            cv2.putText(frame, label, (img_coord[0], img_coord[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i += 1
        if save_vid:
            cv2.imwrite(os.path.join(save_frames_to, f'frame_{i:06d}.jpg'), frame)

    coords = np.array(coords)
    np.save('logs/seq3_april.npy', coords)
    if save_vid:
        save_video(vid_pth, save_frames_to)