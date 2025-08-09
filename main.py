import numpy as np
import cv2
import os
from loguru import logger
from tqdm import tqdm
import shutil
import argparse

from utils.rosbag_utils import image_stream
from utils.viz_utils import save_video
from utils.converter import LLtoUTM
from detector import PersonDetector


def extract_lat_lon_from_file(file_path):
    """Extract latitude and longitude from GT files."""
    lat, lon = None, None
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('latitude:'):
                lat = float(line.strip().split(':')[1])
            elif line.strip().startswith('longitude:'):
                lon = float(line.strip().split(':')[1])
    return [lat, lon] if lat is not None and lon is not None else None


def read_all_casualty_coords(directory):
    """Read all casualty coordinates from GT directory."""
    coords = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            lat_lon = extract_lat_lon_from_file(file_path)
            if lat_lon:
                coords.append(lat_lon)
    return coords


def main():
    parser = argparse.ArgumentParser(description="Localization from the UAV using PersonDetector")
    parser.add_argument('--save_vid', action='store_true', help='Save video from the frames.')
    parser.add_argument('--loc', action='store_true', help='Run localization.')
    parser.add_argument('--save_crops', action='store_true', help='Save individual detection crops.')
    args = parser.parse_args()

    # Load ground truth data
    directory_path = '/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/GT'
    casualty_gps = np.array(read_all_casualty_coords(directory_path))
    casualty_coords = np.array([LLtoUTM(23, lat, lon) for lat, lon in casualty_gps])[:, 1:3].astype(np.float32)
    
    # Configuration paths
    bag_pth = "/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/dry_run_1.bag"
    save_frames_to = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_detector'
    crops_save_to = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_crops'
    
    if args.save_vid:
        if args.loc:
            vid_pth = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_detector_loc.mp4'
        else:
            vid_pth = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_detector_det.mp4'

    # ROS topics
    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    
    logger.info('Loading data from rosbag...')

    # Initialize PersonDetector
    logger.info("Initializing PersonDetector...")
    detector = PersonDetector(
        model_path='checkpoints/11x_ft.pt',
        camera_config='camchain.yaml',
        confidence_threshold=0.67,
        distance_threshold=2.5,
        clustering_method='centroid',
        device='cuda',
        manual_rotation_deg=-43.0
    )
    
    # Setup output directories
    logger.info(f"Saving frames to `{save_frames_to}`.")
    if os.path.isdir(save_frames_to):
        shutil.rmtree(save_frames_to)
        logger.info(f"Removed previous frames: {save_frames_to}")
    os.mkdir(save_frames_to)
    
    if args.save_crops:
        logger.info(f"Saving crops to `{crops_save_to}`.")
        if os.path.isdir(crops_save_to):
            shutil.rmtree(crops_save_to)
            logger.info(f"Removed previous crops: {crops_save_to}")
        os.mkdir(crops_save_to)

    if args.save_vid and os.path.isfile(vid_pth):
        os.remove(vid_pth)

    # Data collection
    all_coords = []
    all_crops = []
    records = []
    
    if args.loc:
        logger.info("Starting localization...")
    else:
        logger.info("Detection running...")

    frame_count = 0
    total_detections = 0
    
    # Process each frame using PersonDetector
    for ts, frame, translation, R_wi, zone in tqdm(image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=args.loc, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    )):
        
        # Process frame with PersonDetector
        result = detector.process_frame(
            frame=frame,
            translation=translation,
            rotation=R_wi,
            enable_localization=args.loc,
            draw_annotations=True,
            return_full_frame=True,
            toGPS=False
        )
        
        # Collect data
        num_detections = len(result.crops)
        total_detections += num_detections
        
        if num_detections > 0:
            logger.debug(f"Frame {frame_count}: {num_detections} detections")
            
            # Save individual crops if requested
            if args.save_crops:
                for i, crop in enumerate(result.crops):
                    crop_filename = f"frame_{frame_count:06d}_det_{i:02d}_id_{result.cluster_ids[i]}.jpg"
                    crop_path = os.path.join(crops_save_to, crop_filename)
                    cv2.imwrite(crop_path, crop)
            
            # Collect coordinates for analysis
            for coord in result.localizations:
                if coord is not None:
                    all_coords.append(coord)
            
            all_crops.extend(result.crops)
        
        # Record frame data for heatmap generation
        records.append({
            'translation': translation,
            'localization': np.array(result.localizations if result.localizations else [], dtype=np.float32)
        })
        
        # Save annotated frame
        if args.save_vid:
            # Use the annotated frame from detector if available, otherwise use original
            output_frame = result.frame if result.frame is not None else frame
            cv2.imwrite(os.path.join(save_frames_to, f'frame_{frame_count:06d}.jpg'), output_frame)
        
        frame_count += 1

    # Save results
    logger.info(f"Processed {frame_count} frames with {total_detections} total detections")
    logger.info(f"Found {len(all_coords)} localized detections")
    logger.info(f"Saved {len(all_crops)} crops")
    
    # Get cluster information
    cluster_info = detector.get_cluster_info()
    logger.info(f"Final clustering: {len(cluster_info)} clusters")
    for cluster_id, info in cluster_info.items():
        logger.info(f"Cluster {cluster_id}: {info['num_points']} points, centroid: {info['centroid']}")
    
    # Save coordinate data
    os.makedirs('logs', exist_ok=True)
    np.save('logs/detector_coords.npy', np.array(all_coords))
    
    # Generate video if requested
    if args.save_vid:
        logger.info(f"Generating video: {vid_pth}")
        save_video(vid_pth, save_frames_to)
        logger.info("Video generation complete")


if __name__ == '__main__':
    main()