import cv2
import apriltag
from rosbag_utils import image_stream
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stderr
import numpy as np
from viz_utils import save_video



def suppress_stderr():
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved

def restore_stderr(saved):
    sys.stderr.flush()
    os.dup2(saved, 2)
    os.close(saved)



class AprilTagDetector:
    def __init__(self):
        self.detector = apriltag.Detector()


    def detect(self, frame):
        """
        Detect AprilTags in a grayscale image.
        
        Args:
            gray_frame (np.ndarray): Grayscale image frame.
        Returns:
            list: List of detected AprilTag results.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        saved = suppress_stderr()
        try:
            results = self.detector.detect(gray_frame)
        finally:
            restore_stderr(saved)
        return results


    def __call__(self, frame):
        """
        Get the centers of detected AprilTags in a frame.
        
        Args:
            frame (np.ndarray): Image frame with detected AprilTags.
        Returns:
            list: List of centers of detected AprilTags.
        """
        results = self.detect(frame)
        centers = []
        for detection in results:
            center = detection.center
            centers.append(center)
        return centers