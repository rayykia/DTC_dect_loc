from __future__ import annotations
import re
import cv2
import yaml
import numpy as np
from typing import List, Union, Optional


class Camera:
    """Camera model.

    Initialization:
    >>> camera = Camera.load_camera()  # load intrinsics from yaml
    >>> camera = Camera(fx, fy, cx, cy)  # initialize directly
    """
    def __init__(
            self, 
            fx: float, fy: float, cx: float, cy: float, 
            w: float, h: float, 
            distortion_coef: Optional[Union[np.ndarray, List[float]]] = None, 
            distortion_model: Optional[str] = None
    ):
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        self.width, self.height = w, h
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy

        if distortion_coef is not None:
            self.dist_coef = np.array(distortion_coef)
        else: 
            self.dist_coef = None

        self.dist_model = distortion_model


    @staticmethod
    def load_config(
        path: str = "camchain.yaml"
    ) -> Camera:
        # from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        with open(path, "r") as f:
            cfg = yaml.load(f, Loader=loader)

        cam = cfg['cam0']
        fx, fy, cx, cy = cam['intrinsics']
        distortion_coef = np.array(cam['distortion_coeffs'])
        distortion_model = cam['distortion_model']
        w, h = cam['resolution']

        return Camera(fx, fy, cx, cy, w, h, distortion_coef, distortion_model)
    
    @staticmethod
    def from_rosbag(msg):
        """If camera intrinsics are stored in a rosbag message."""
        distortion_coef = msg.D
        distortion_model = msg.distortion_model
        fx, cx, fy, cy = msg.K[0], msg.K[2], msg.K[4], msg.K[5]
        w = msg.width
        h = msg.height

        return Camera(fx, fy, cx, cy, w, h, distortion_coef, distortion_model)

    def project(
        self,
        camera_coords: Union[np.ndarray, List]
    ):
        """
        Project the coordinates from the camera frame to the image pixel frame.
        
        Args:
            camera_coords (np.ndarray or List): each row is [x, y, z] (N, 3)
        Returns:
            np.ndarray: each row is [u, v] (N, 2)
        """
        camera_coords = np.array(camera_coords)
        if camera_coords.ndim == 1:
            camera_coords = camera_coords[None, :]
        assert camera_coords.ndim == 2 and camera_coords.shape[1] == 3, f"Input must be of shape (N, 3), but got {camera_coords.shape}"
        assert (camera_coords[:, -1] > 0).all(), f"Z coordinate must be positive for projection, got {camera_coords}"

        # Normalize the coordinates
        normalized = camera_coords / camera_coords[:, -1].reshape(-1, 1)  # divide by z to get normalized coordinates

        # Apply intrinsic parameters
        pixel_coords = normalized @ self.K.T  # (3, N)
        pixel_coords = pixel_coords[:, :2]

        return pixel_coords
    
    
    def reproject(
            self, 
            pixel_coords: Union[np.ndarray, List],
            fix_distortion: bool = True
    ):
        """
        Reproject from image pixel frame to normalized camera frame.
        
        Args:
            pixel_coords (np.ndarray or List): pixel coordinates with origin at top-left corner
        Returns:
            np.ndarray: each row is [x, y, 1] (N, 3)
        """
        pixel_coords = np.array(pixel_coords)
        assert pixel_coords.ndim == 2 and pixel_coords.shape[1] == 2, f"Input must be of shape (N, 2), but got {pixel_coords.shape}"

        if self.dist_coef is not None and fix_distortion:
            # undistort when given the distortion coefficients
            pixel_coords = pixel_coords.reshape(-1, 1, 2).astype(np.float32)
            undistorted = cv2.undistortPoints(pixel_coords, self.K, self.dist_coef)
            rays = np.concatenate([undistorted[:, 0], np.ones((len(undistorted), 1))], axis=1)
        else:
            # pinhole model
            x = (pixel_coords[:, 0] - self.cx) / self.fx
            y = (pixel_coords[:, 1] - self.cy) / self.fy
            rays = np.stack([x, y, np.ones_like(x)], axis=1)

        return rays
    
    def __repr__(self):
        """Return all camera info."""
        return (f"Camera(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
                f"width={self.width}, height={self.height}, "
                f"distortion_model={self.dist_model!r}, "
                f"distortion_coef={self.dist_coef})")



if __name__ == "__main__":
    # Example usage
    cam = Camera.load_config("camchain.yaml")
    print(cam)
    
    pixel_coords = np.array([[152, 252]])
    rays = cam.reproject(pixel_coords)
    print("Reprojected rays:\n", rays)
    
    # Load from ROS message (example, not executable here)
    # msg = ...  # Assume this is a ROS camera info message
    # cam_from_msg = Camera.from_rosbag(msg)
    # print(cam_from_msg)