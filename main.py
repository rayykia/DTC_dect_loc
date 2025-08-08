
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
from viz_utils import save_video, save_heatmap_frames
from apriltag_utils import AprilTagDetector
import argparse

from localization import Tracker
from calibration import UAVCalibration
from converter import LLtoUTM, UTMtoLL


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


directory_path = '/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/GT'
casualty_gps = np.array(read_all_casualty_coords(directory_path))
casualty_coords = np.array([LLtoUTM(23, lat, lon) for lat, lon in casualty_gps])[:, 1:3].astype(np.float32)  # Extracting easting and northing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Localization from the UAV")
    parser.add_argument('--save_vid', action='store_true', help='Save video from the frames.')
    parser.add_argument('--loc', action='store_true', help='Run localization.')
    args = parser.parse_args()

    #############################################################################
    bag_pth = "/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/dry_run_1.bag"
    save_frames_to = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1'
    if args.save_vid:
        if args.loc:
            vid_pth = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_loc.mp4'
        else:
            vid_pth = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_dect.mp4'
    #############################################################################


    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')

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

    if args.save_vid and os.path.isfile(vid_pth):
            os.remove(vid_pth)



    calib = UAVCalibration()
    i = 0
    confidence_threshold = 0.67
    logger.info(f"YOLO confidence threshold: {confidence_threshold}")
    
    coords = []
    detections = []
    
    ### manual calibration ###
    theta = np.radians(-43)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    ########################
    records = []
    if args.loc:
        logger.info("Starting localization...")
    else:
        logger.info("Detection running...")


    tracker = Tracker(cam, calib)
    for ts, frame, translation, R_wi, zone in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=args.loc, 
        use_pose_rot = False, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    ):
        
        
        result = model.predict(frame, verbose=False, conf=confidence_threshold)[0]
        
        if args.loc:
            R_wi = Rz @ R_wi
            if i == 0:
                starting_coord = translation.copy()
        localization_this_frame = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            pixel_coord = box_center(box, center=False)

            if args.loc:
                # world_coord = pixel_localization(
                #     cam,
                #     pixel_coord,
                #     translation,
                #     R_wi,
                #     calib
                # )
                world_coord = tracker.pixel2NED(
                    pixel_coord, translation, R_wi
                )
                coords.append(world_coord)
                
                localization_this_frame.append(world_coord)
                

                label = f"({np.round(world_coord - starting_coord, 2)})"

            else:
                conf = box.conf.item()
                label = f"Person {conf:.2f}"
                # Log bounding box: frame_id, x1, y1, x2, y2, confidence
                # detections.append([i, xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

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

        records.append({
            'translation': translation,
            'localization': np.array(localization_this_frame, dtype=np.float32)
        })
        if args.save_vid:
            # Always save the frame, even if no person is found
            cv2.imwrite(os.path.join(save_frames_to, f'frame_{i:06d}.jpg'), frame)
    
    np.save('logs/use_alt.npy', np.array(coords))
    if args.save_vid:
        save_heatmap_frames(
            records,
            save_path= '/mnt/UNENCRYPTED/ruichend/results/heatmap_frames',
            gt = casualty_coords
        )
        save_video(
            output_vid='/mnt/UNENCRYPTED/ruichend/results/heatmap.mp4',
            src_path='/mnt/UNENCRYPTED/ruichend/results/heatmap_frames',
            fps=60,
            compress=False
        )
        # save_video(vid_pth, save_frames_to)