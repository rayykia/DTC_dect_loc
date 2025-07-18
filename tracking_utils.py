import numpy as np
from ultralytics.engine.results import Boxes
from scipy.spatial.transform import Rotation as R
from typing import List, Union
import torch
from loguru import logger


def set_device(device: str = 'cuda'):
    if device == 'cuda':
        if torch.cuda.is_available():
            logger.info("CUDA is availabe, using CUDA.")
            device = torch.device('cuda')
        else:
            logger.info("CUDA is not available, using CPU.")
            device = torch.device('cpu')
        return device
    elif device == 'cpu':
        device = torch.device('cpu')
        return device
    else:
        raise ValueError(f"`should` be `cuda` or `cpu`, but got {device} instead.")


def box_center(
        boxes: Boxes, 
        center: bool = True
) -> np.ndarray:
    """Get the center of the bounding boxes.
    
    Args:
        boxes (Boxes): Bounding boxes from the YOLO model.
        center (bool): If True, the origin is the image center, else top-left corner.
    Returns:
        np.ndarray: Array of shape (N, 2) containing the centers of the boxes.
    """
    box_centers = boxes.xywh[:, :2]
    
    if not center:
        box_centers = box_centers.cpu().numpy()
    else:
        H, W = boxes.orig_shape[:2]
        box_centers = box_centers.cpu().numpy() - np.array([W / 2, H / 2])

    return box_centers.astype(int)


def quaternions_to_SO3(
    quaternions: Union[List, np.ndarray]
):
    """
    Convert an array of N quaternions (x, y, z, w) to N 3x3 rotation matrices.
    
    Args:
        quaternions (List or np.ndarray): shape (N, 4)
    Returns:
        np.ndarray: rotation matrices shape (N, 3, 3)
    """
    quaternions = np.array(quaternions)
    rotations = R.from_quat(quaternions)

    return rotations.as_matrix()


def to_SE3(
        rot: np.ndarray,
        translation: np.ndarray,
        from_quat: bool = True
):
    """
    Convert rotation and translation into homogeneous transforamation matrices.
    
    Args:
        rot (np.ndarray): rotation matrices or quaternions
        translation (np.ndarray): translation vectors
        from_quat (bool): set True if `rot` are quaternions
    Returns:
        np.ndarray: rotation matrices shape (N, 3, 3)
    """
    if from_quat:
        rot = quaternions_to_SO3(rot)

    T = np.empty((rot.shape[0], 4, 4))
    T[:, :3, :3] = rot
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.

    return T


def dummy_SO3(
        rotation = None
) -> np.ndarray:
    """Dummy rotation matrix for the UAV camera facing downward.
    If rotation is not set, a random yaw angle is generated.
    `rotation`: [-pi, pi]
    
    Args:
        rotation (float or None): Yaw angle in radians (rotation about Z_world).
    Returns:
        np.ndarray: 3x3 homogeneous transformation matrix (R_wc).
    """
    if rotation is None:
        rotation = np.random.uniform(-np.pi, np.pi)

    # Yaw rotation around Z axis (world up)
    R_yaw = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation),  np.cos(rotation), 0],
        [0,                0,                1]
    ])

    R_downward = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    R_wc = R_yaw @ R_downward

    return R_wc




def dummy_SE3(
        rotation=None, 
        translation=None
) -> np.ndarray:
    """Dummy homogenerous transformation matrix for the UAV camera facing downward.
    If parameters are not set, random values are generated.
    `rotation`: [-pi, pi]
    `translation`: [-10, 10] for x and y, [5, 20] for z.
    
    Args:
        rotation (float or None): Yaw angle in radians (rotation about Z_world).
        translation (np.ndarray or None): tramslation of the UAV.
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix (T_wc).
    """
    
    # Random translation if not provided
    if translation is None:
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(5, 20)  # height above ground
        translation = np.array([x, y, z])
    else:
        translation = np.array(translation)
    
    T_wc = np.eye(4)
    T_wc[:3, :3] = dummy_SO3(rotation)
    T_wc[:3, 3] = translation

    return T_wc
