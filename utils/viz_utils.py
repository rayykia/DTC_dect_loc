"""Save video from a diectory of frames and compress its size if needed."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
from time import time
import ffmpeg
from loguru import logger
import glob

from typing import List, Optional


def compress_vid(f, out_pth = None):
    logger.info(f'Compressing video...')
    
    if out_pth is None:
        dirname = os.path.dirname(f)
        basename = os.path.basename(f)
        name, ext = os.path.splitext(basename)
        compressed_output = os.path.join(dirname, f"compressed_{name}{ext}")
    else:
        compressed_output = out_pth

    crf = '28'
    ffmpeg.input(f).output(compressed_output, vcodec='libx264', crf=crf).run()
    logger.info(f'Compressed video saved to `{compressed_output}`.')
    os.remove(f)


def save_video(
        output_vid: str, 
        src_path: str,
        compress: bool = True,
        fps = 30
):
    """Save video from the referred frames.

    Args:
        f (str): name of the saved video.
        src_path (str): source directory.
        compress (bool): compress the output video
    """
    if compress:
        temp_path = '/mnt/UNENCRYPTED/ruichend/results/temp_vid.mp4'
    else:
        temp_path = output_vid

    logger.info(f"Loading frames from `{src_path}`.")
    image_files = sorted(
        glob.glob(os.path.join(src_path, '*.png')) +
        glob.glob(os.path.join(src_path, '*.jpg'))
    )

    # Read first image to get dimensions
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    scale = 0.5  # shrink to 50% size
    new_width, new_height = int(width * scale), int(height * scale)

    video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (new_width, new_height))

    logger.info(f"Writing video to `{temp_path}`.")
    for file in tqdm(image_files):
        frame = cv2.imread(file)
        if frame is not None:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            video_writer.write(resized_frame)

    video_writer.release()

    if compress:
        compress_vid(temp_path, output_vid)
   
   
   
def save_heatmap_frames(
    record: List[dict],
    save_path: str,
    gt: Optional[np.ndarray] = None
):
    """Save heatmaps with UAV trajectory and detected frames.
    
    Args:
        record (List[dict]): List of dictionaries containing 'translation', and 'localization'.
        save_path (str): Path to save the heatmaps.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_trajectory = []
    all_localization = []
    for data in record:
        all_trajectory.append(data['translation'][:2])  # northing, easting
        for loc in data['localization']:
            all_localization.append(loc.flatten()[:2])  # northing, easting
    all_trajectory = np.array(all_trajectory, dtype=np.float32)
    all_localization = np.array(all_localization, dtype=np.float32)
    
    east_min = min(np.min(all_trajectory[:, 1]), np.min(all_localization[:, 1])) - 5
    east_max = max(np.max(all_trajectory[:, 1]), np.max(all_localization[:, 1])) + 5
    north_min = min(np.min(all_trajectory[:, 0]), np.min(all_localization[:, 0])) - 5
    north_max = max(np.max(all_trajectory[:, 0]), np.max(all_localization[:, 0])) + 5
    hist_range = [[east_min, east_max], [north_min, north_max]]
    
    bins = 30
    
    hist = np.zeros((bins, bins))
    
    curr_trajectory = []
    curr_localization = []
    mark = False
    for data in tqdm(record):
        plt.figure(figsize=(6, 6), dpi=300)
        
        curr_trajectory.append(data['translation'][:2])
        for loc in data['localization']:
            mark = True
            curr_localization.append(loc.flatten()[:2])
        curr_trajectory = np.array(curr_trajectory, dtype=np.float32)
        curr_localization = np.array(curr_localization, dtype=np.float32)
        
        # update and plot histogram
        if curr_localization != []:
            hist, _, _ = np.histogram2d(
                curr_localization[:, 1], 
                curr_localization[:, 0], 
                bins=bins, 
                range=hist_range
            )
        plt.imshow(
            hist.T, 
            extent=[east_min, east_max, north_min, north_max], 
            origin='lower', 
            interpolation='nearest',
            cmap="coolwarm"
        )
        
        # plot trajectory
        plt.plot(curr_trajectory[:, 1], curr_trajectory[:, 0], color='w', linewidth=2, label='UAV Trajectory')
        
        # plot ground truth
        if gt is not None:
            plt.scatter(gt[:, 0], gt[:, 1], marker='+', color='r', label='GT')
        
        # mark localization for this frame
        if mark:
            localization_this_frame = data['localization']
            plt.scatter(localization_this_frame[:, 1], localization_this_frame[:, 0], color='k', label='Localization', marker='x')
            mark = False
            
        plt.title(f'Detection Heatmap - Frame {len(curr_trajectory)}')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}/{len(curr_trajectory):06d}.png")
        plt.close()
        curr_trajectory = curr_trajectory.tolist()
        curr_localization = curr_localization.tolist()
        
        
    

if __name__ == "__main__":
#     save_video("loc.mp4", "coorded_imgs")
#     save_video('/mnt/extra_dtc/ruichend/results/dry_run_1_dect.mp4', '/mnt/extra-dtc/ruichend/results/dry_run_1_april')
#     compress_vid('/mnt/UNENCRYPTED/ruichend/results/temp_vid.mp4', '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_dect30.mp4')
    save_video(
            output_vid='/mnt/UNENCRYPTED/ruichend/results/dry_run_1_loc.mp4',
            src_path='/mnt/UNENCRYPTED/ruichend/results/dry_run_1',
            fps=30,
            compress=True
        )
    
    
