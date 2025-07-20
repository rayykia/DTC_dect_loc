#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
from cv_bridge import CvBridge
from pyproj import Proj
from tf.transformations import quaternion_matrix

class HumanGPSTracker:
    def __init__(self):
        rospy.init_node('human_gps_detector')

        # YOLO model
        self.model = YOLO('yolov8x.pt')  
        self.bridge = CvBridge()

        # Camera intrinsics
        self.K = np.array([
            [1465.7934, 0, 1037.3987],
            [0, 1465.1953, 754.1169],
            [0, 0, 1]
        ])
        self.K_inv = np.linalg.inv(self.K)

        # Camera-to-IMU rotation 
        self.R_cam_imu = np.array([
            [ 0.99961803,  0.01171022,  0.02503332],
            [-0.01195821,  0.99988067,  0.00977995],
            [-0.0249158 , -0.01007557,  0.99963878]
        ])

        # GPS projection
        self.proj_utm = Proj(proj='utm', zone=33, ellps='WGS84')

        # Latest drone pose
        self.last_pose = None  # {'t': np.array([x, y, z]), 'R': 3x3, 'lat':..., 'lon':..., 'alt':...}

        # Subscribers
        rospy.Subscriber('/camera/image_color/compressed', CompressedImage, self.image_callback)
        rospy.Subscriber('/ekf/pose', PoseStamped, self.pose_callback)

    def pose_callback(self, msg):
        # Position in UTM (meters)
        t = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # Orientation as rotation matrix
        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        R_world_imu = quaternion_matrix(quat)[:3, :3]  # 3x3 rotation

        self.last_pose = {'t': t, 'R_world_imu': R_world_imu}

    def image_callback(self, msg):
        if self.last_pose is None:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = self.model(img, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] != "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                gps_coord = self.pixel_to_gps(u, v, self.last_pose)
                if gps_coord:
                    lat, lon = gps_coord
                    print(f"[Human] lat: {lat:.6f}, lon: {lon:.6f}")

    def pixel_to_gps(self, u, v, pose):
        pixel = np.array([u, v, 1])
        ray_cam = self.K_inv @ pixel
        ray_world = pose['R_world_imu'] @ self.R_cam_imu @ ray_cam

        print(f"[INFO] ray_world: {ray_world}, drone height: {pose['t'][2]:.2f}")

        if ray_world[2] == 0:
            print("[INFO] Ray Z is zero, can't project.")
            return None

        s = pose['t'][2] / ray_world[2]

        if s < 0:
            print("s < 0, not projecting backwards")
            return None

        ground_pos = pose['t'] + s * ray_world
        x_ground, y_ground = ground_pos[0], ground_pos[1]
        lon, lat = self.proj_utm(x_ground, y_ground, inverse=True)
        return lat, lon



if __name__ == '__main__':
    try:
        tracker = HumanGPSTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass