"""Not updated yet, debug AprilTag first."""

import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
from loguru import logger
from tqdm import tqdm


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

    j = 2
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
    gps_topic = 'mavros/global_position/raw/fix'
    logger.info('Loding data from rosbag...')

    # data_bundel = bundeled_data_from_bag(bag_pth, frame_topic, pose_topic)

    logger.info('Done.')

    device = set_device()
    

    logger.info("Loading YOLO model...")
    ckpt = 'checkpoints/11s_ft.pt'
    model = YOLO(ckpt)
    model.to(device)
        
        
    cam = Camera.load_config('camchain.yaml')
    logger.info(cam)


    box_color = (255, 0, 0)  # Red
    thickness = 5  # Thicker box
    
    logger.info(f"Saving frames to `{save_frames_to}`.")
    if not os.path.isdir(save_frames_to):
        logger.info(f"`{save_frames_to}` does not exist, making...")
        os.mkdir(save_frames_to)



    i = 0
    for ts, frame, translation, R_wc in image_stream(bag_pth, frame_topic, pose_topic, args.loc, gps_topic):

        result = model(frame, verbose=False)[0]

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            img_coord = box_center(box, center=False)

            if args.loc:
                print("Localization mode enabled.")
                ray_world = R_wc @ cam.reproject(img_coord).reshape(-1, 1)
                ray_world = ray_world.flatten()
                s = np.abs(translation[2] / ray_world[2])
                
                world_coord = translation + s * ray_world

                label = f"({world_coord[0]:.2f}, {world_coord[1]:.2f}, {world_coord[2]:.2f})"

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