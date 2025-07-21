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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Localization from the UAV")
    parser.add_argument('--save_vid', action='store_true', help='Save video from the frames.')
    parser.add_argument('--loc', action='store_true', help='Run localization.')
    args = parser.parse_args()

    j = 3
    #############################################################################
    # bag_pth = '/mnt/ENCRYPTED/workshop2/20250311/course-1/dione/course_1.bag'
    bag_pth = f'/mnt/UNENCRYPTED/ruichend/seq/seq{j}/seq_{j}.bag'
    save_frames_to = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_frames'
    if args.save_vid:
        if args.loc:
            vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_loc.mp4'
        else:
            vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_dect.mp4'
    #############################################################################

    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')

    # data_bundel = bundeled_data_from_bag(bag_pth, frame_topic, pose_topic)

    logger.info('Done.')

    device = set_device()
    

    logger.info("Loading YOLO model...")
    ckpt = 'checkpoints/11x_ft.pt'
    model = YOLO(ckpt)
    model.to(device)
        
        
    cam = Camera.load_config('camchain.yaml')
    logger.info(cam)


    box_color = (255, 0, 0)  # Red
    thickness = 5  # Thicker box
    
    logger.info(f"Saving frames to `{save_frames_to}`.")
    if os.path.isdir(save_frames_to):
        shutil.rmtree(save_frames_to)
        logger.info(f"Removed previous frames: {save_frames_to}")
    os.mkdir(save_frames_to)

    if os.path.isfile(vid_pth):
        os.remove(vid_pth)


    i = 0
    
    t_cam2body = np.array([1.297, -1.282, -0.150])
    t_body2cam = np.array([1.30, 0.15, -1.30])
    R_bc = np.array([
        [0.995,  0,  -0.099],
        [0.099, 0,  0.995],
        [0,  1, 0]
    ])
    
    T_ic = np.array([
        [ 0.99961803,  0.01171022,  0.02503332, -0.01592941],
        [-0.01195821,  0.99988067,  0.00977995,  0.01279475],
        [-0.0249158 , -0.01007557,  0.99963878,  0.05199873],
        [ 0.        ,  0.        ,  0.        ,  1.        ],
    ])
    R_ic = T_ic[:3, :3]
    t_cam2imu = T_ic[:3, 3]
    t_cam2body = np.array([-0.015, -0.15, 0.05])
    
    def rotation_matrix_y(theta_deg):
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return np.array([
            [cos_t, 0, sin_t],
            [0,     1, 0    ],
            [-sin_t,0, cos_t]
        ])
        
    def rotation_matrix_x(theta_deg):
        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        return np.array([
            [1, 0, 0],
            [0, cos_t, -sin_t],
            [0, sin_t, cos_t]
        ])
    R_id = rotation_matrix_x(-15)
    t_imu2body = t_cam2body - (R_id.T @ t_cam2imu.reshape(-1, 1)).flatten()
    
    for ts, frame, translation, R_wi, zone in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=True, 
        use_pose_rot = False, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    ):
        R_wc = R_wi @ R_ic
        R_wd = R_wi @ R_id
        R_cd = R_ic.T @ R_id
        # R_wc = R_dw.T @ R_dc
        result = model(frame, verbose=False)[0]
        if i == 0:
            starting_coord = translation.copy()

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            img_coord = box_center(box, center=False)

            if args.loc:
                
                img_coord = img_coord.reshape(-1, 2)        
                # long_uav, lat_uav, alt_uav = translation
                # zone, easting, northing = LLtoUTM(23, lat_uav, long_uav)
                northing, easting, alt_uav = translation  # NED
                ray_cam = (cam.reproject(img_coord)).reshape(-1, 1)
                ray_body =(R_cd.T @ (ray_cam)).flatten() # ray in body frame
                ray_world = (R_wc @ (ray_cam)).flatten()  # ray in world frame
                
                
                cam_loc = translation + (R_wd @ t_cam2body.reshape(-1, 1)).flatten()  # camera: northing, easting, altitude
                alt_cam = cam_loc[-1]
                s = -(alt_cam)/ray_world[-1]
                
                target2body = s * ray_body + t_cam2body
                target2body_world = (R_wd @ target2body.reshape(-1, 1)).flatten()
                
                world_coord = translation + target2body_world

                label = f"({np.round(world_coord - starting_coord, 2)})"

            else:
                print("Detection mode enabled.")
                conf = box.conf.item()
                label = f"Person {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), box_color, thickness)

            # Prepare label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame,
                        (xyxy[0], xyxy[1] - text_h - 10),
                        (xyxy[0] + text_w, xyxy[1]),
                        box_color, -1)

            # Draw text label on top
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i += 1
        # Always save the frame, even if no person is found
        cv2.imwrite(os.path.join(save_frames_to, f'frame_{i:06d}.jpg'), frame)

    
    if args.save_vid:
        save_video(vid_pth, save_frames_to)