"""Loading data from the rosbag."""

import rosbag
import cv2
import numpy as np
from tqdm import tqdm
from tracking_utils import quaternions_to_SO3, to_SE3
import bisect
from typing import Tuple, Optional
from loguru import logger

from converter import LLtoUTM, UTMtoLL



def read_gps(
        bag: rosbag.Bag,
        gps_topic: str
):
    """Read the UAV GPS data from rosbag topic.

    Args:
        bag (rosbag.Bag): the rosbag to extract data from
        pose_topic (str): the topic to load
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - (N,), timestamps
            - (N, 3), GPS data: longitude, latitude, altitude
    """
    gps_timestamps, gps_ll = [], []

    for _, msg, t in bag.read_messages(topics=[gps_topic]):
        gps_timestamps.append(t.to_sec())
        gps_ll.append(
            [msg.longitude, msg.latitude, msg.altitude]
        )

    assert len(gps_ll) != 0, f"No GPS data found in topic `{gps_topic}`."
    
    gps_timestamps = np.array(gps_timestamps)
    gps_ll = np.array(gps_ll)
    
    # z0 = gps_ll[0, -1]
    # gps_ll = gps_ll - np.array([0, 0, z0])

    return gps_timestamps, gps_ll


def read_imu(
        bag: rosbag.Bag,
        imu_topic: str
):
    """Read the UAV IMU data from rosbag topic.

    Args:
        bag (rosbag.Bag): the rosbag to extract data from
        pose_topic (str): the topic to load
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - (N,), timestamps
            - (N, 4), quaternions (x, y, z, w)
    """
    imu_timestamps, rot_quat = [], []

    for _, msg, t in bag.read_messages(topics=[imu_topic]):
        imu_timestamps.append(t.to_sec())
        rot_quat.append(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        )

    assert len(rot_quat) != 0, f"No GPS data found in topic `{imu_topic}`."
    
    imu_timestamps = np.array(imu_timestamps)
    rot_quat = np.array(rot_quat)


    return imu_timestamps, rot_quat


def read_pose(
        bag: rosbag.Bag,
        pose_topic: str,
):
    """Read the UAV pose from rosbag topic.

    Args:
        bag (rosbag.Bag): the rosbag to extract data from
        pose_topic (str): the topic to load
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - (N,), timestamps
            - (N, 3), translation vectors
            - (N, 4), quaternions (x, y, z, w)
    """
    pose_timestamps, trans, rot = [], [], []

    for _, msg, t in bag.read_messages(topics=[pose_topic]):
        pose_timestamps.append(t.to_sec())
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        trans.append([
            x, y, z
        ])
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        rot.append([
            qx, qy, qz, qw
        ])


    assert len(trans) != 0, f"No pose data found in topic `{pose_topic}`."
    
    pose_timestamps = np.array(pose_timestamps)
    
    trans = np.array(trans)
    z0 = trans[0, -1]
    trans = trans - np.array([0, 0, z0])
    rot = np.array(rot)

    return pose_timestamps, trans, rot



def find_nearest(ts_list, target_ts):
    """Given the a timestamp (target_ts), find the closet one in a list of timestamps (ts_list)."""
    idx = bisect.bisect_left(ts_list, target_ts)
    
    if idx == 0:
        return 0
    if idx == len(ts_list):
        return -1

    before = ts_list[idx - 1]
    after = ts_list[idx]

    if abs(before - target_ts) < abs(after - target_ts):
        return idx - 1
    else:
        return idx

def camera_config(f: str):
    logger.info("Extracting camera info.")
    with rosbag.Bag(f) as bag:
        for _, msg, _ in bag.read_messages(topics=['/camera/camera_info']):
            camera_info = msg
            break
    return camera_info



def image_stream(
        f: str, 
        img_topic: str, 
        pose_topic: str, 
        loc: bool,
        use_pose_rot: bool = False,
        gps_topic: Optional[str] = None,
        imu_topic: Optional[str] = None,
):
    """Image generator for rosbag.

    Args:
        f (str): path to the rosbag
        img_topic (str): topic of the images
        pose_topic (str): topic of the poses
        loc (bool): If True, also stream pose data with images
        gps_topic (str): topic of the GPS data, if set, use GPS data for translation instead of pose data
        imu_topic (str): topic of the IMU data, if set, use IMU data for rotation instead of pose data
    """
    bag = rosbag.Bag(f)

    if loc:
        logger.info("Loading pose...")

        if not use_pose_rot:
            logger.info("Loading GPS...")
            gps_timestamps, gps_ll = read_gps(bag, gps_topic)

            translation = []
            zone = None
            for i, ll in enumerate(gps_ll):
                zone, easting, northing = LLtoUTM(23, ll[1], ll[0])
                translation.append([northing, easting, -ll[-1]])
            translation = np.array(translation)

            pose_timestamps, rotation = read_imu(bag, imu_topic)

            T_ci = np.array([
            [ 0.99961803, -0.01195821, -0.0249158,   0.01737192],
            [ 0.01171022, 0.99988067, -0.01007557, -0.01208277],
            [ 0.02503332,  0.00977995,  0.99963878, -0.05170631],
            [ 0.,          0.,          0.,          1.        ]
            ])
            R_ci = T_ci[:3, :3]
            rotation = quaternions_to_SO3(rotation)
            # rotation = [r.T @ R_ci.T for r in rotation]

        else:
            
            logger.info("Loading GPS...")
            gps_timestamps, gps_ll = read_gps(bag, gps_topic)

            translation = []
            zone = None
            for i, ll in enumerate(gps_ll):
                zone, easting, northing = LLtoUTM(23, ll[1], ll[0])
                translation.append([northing, easting, -ll[-1]])
            translation = np.array(translation)
            
            pose_timestamps, _, rotation = read_pose(bag, pose_topic)
            rotation = quaternions_to_SO3(rotation)

        


        logger.info("Localization Running...")
        for _, msg, t in tqdm(bag.read_messages(topics=[img_topic])):
            ts = t.to_sec()
            
            img_np = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            time_idx = find_nearest(pose_timestamps, ts)
            rot = rotation[time_idx]
            
            
            time_idx = find_nearest(gps_timestamps, ts)
            trans = translation[time_idx]

            yield ts, image, trans, rot, zone

    else:
        logger.info("Detection Running...")
        for _, msg, t in tqdm(bag.read_messages(topics=[img_topic])):
            ts = t.to_sec()
            
            img_np = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            trans = None
            rot = None
            zone = None

            yield ts, image, trans, rot, zone



def bundeled_data_from_bag(
        f: str, 
        img_topic: str, 
        pose_topic: str, 
):
    """Read the data: frames and poses.

    Args:
        f (str): path to the rosbag
        img_topic (str): topic of the images
        pose_topic (str): topic of the poses
    """
    bag = rosbag.Bag(f)
    data_bundle = []

    logger.info("Loading pose...")
    pose_timestamps, translation, rotation = read_pose(bag, pose_topic)
    rotation = quaternions_to_SO3(rotation)

    T_ci = np.array([
        [ 0.99961803, -0.01195821, -0.0249158,   0.01737192],
        [ 0.01171022, 0.99988067, -0.01007557, -0.01208277],
        [ 0.02503332,  0.00977995,  0.99963878, -0.05170631],
        [ 0.,          0.,          0.,          1.        ]
    ])

    R_ci = T_ci[:3, :3]
    t_ci = T_ci[:3, 3]

    rotation = rotation @ R_ci.T
    translation += t_ci

    logger.info("Aligning frames...")
    for topic, msg, t in tqdm(bag.read_messages(topics=[img_topic])):
        ts = t.to_sec()

        # ###############################
        # # use 1000 frames for debugging
        # if ts < 1741733505.4956772:
        #     continue
        # if ts > 1741733755.5114589:
        #     break
        # ###############################
        
        img_np = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        time_idx = find_nearest(pose_timestamps, ts)
        trans = translation[time_idx]
        rot = rotation[time_idx]

        # Extract values
        data_bundle.append({
            'timestamp': ts,
            'image': image,
            'trans': trans,
            'rot': rot
        })

    return data_bundle

if __name__ == "__main__":
    bag_pth = '/mnt/UNENCRYPTED/ruichend/seq/seq3/seq_3.bag'
    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'

    # data_boundle = bundeled_data_from_bag(bag_pth, frame_topic,pose_topic)
    # print(data_boundle[2500]['timestamp'])
    # print(data_boundle[3500]['timestamp'])
    # print(data_boundle[4000]['timestamp'])
    # print(data_boundle[10000]['timestamp'])
    
    ts, trans, rot = read_pose(rosbag.Bag(bag_pth), pose_topic)
    print(len(ts))
    print(len(trans))
    print(len(rot))
    