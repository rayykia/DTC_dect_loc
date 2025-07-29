"""Save video from a diectory of frames and compress its size if needed."""

import cv2
import os
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO
from time import time
import ffmpeg
from loguru import logger


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
        compress: bool = True
):
    """Save video from the referred frames.

    Args:
        f (str): name of the saved video.
        src_path (str): source directory.
        compress (bool): compress the output video
    """
    temp_path = '/mnt/UNENCRYPTED/ruichend/results/temp_vid.mp4'
    
    fps = 30

    logger.info(f"Loading frames from `{src_path}`.")
    image_files = sorted(glob(os.path.join(src_path, '*.jpg')))

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
        

if __name__ == "__main__":
    # save_video("loc.mp4", "coorded_imgs")
    # save_video('/mnt/extra_dtc/ruichend/results/dry_run_1_dect.mp4', '/mnt/extra-dtc/ruichend/results/dry_run_1_april')
    compress_vid('/mnt/UNENCRYPTED/ruichend/results/temp_vid.mp4', '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_dect30.mp4')