#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu, MagneticField
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from tf.transformations import (
    euler_from_quaternion, quaternion_from_euler, quaternion_matrix
)
from pyproj import Proj
from filterpy.kalman import ExtendedKalmanFilter
from tf.transformations import quaternion_multiply, quaternion_from_euler


class PythonRobotLocalizationEKF:
    def __init__(self):
        rospy.init_node('python_robot_localization_ekf')

        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.ekf = ExtendedKalmanFilter(dim_x=9, dim_z=6)
        self.ekf.x = np.zeros(9)
        self.ekf.P *= 1.0
        self.ekf.R *= 0.5
        self.ekf.Q = np.diag([
            0.1, 0.1, 0.1,  # pos
            0.05, 0.05, 0.05,  # vel
            0.01, 0.01, 0.02   # orientation
        ])

        self.prev_time = rospy.Time.now()

        # Coordinate converters
        self.proj_utm = Proj(proj='utm', zone=33, ellps='WGS84')
        self.proj_wgs84 = Proj(proj='latlong', datum='WGS84')

        # IMU to GPS offset in IMU/body frame (in meters)
        self.T_gps_imu = np.array([-1.30, 0.15, 0.0])

        # Publishers
        self.pose_pub = rospy.Publisher('/ekf/pose', PoseStamped, queue_size=10)
        self.gps_pose_pub = rospy.Publisher('/ekf/pose_gps', NavSatFix, queue_size=10)
        self.rel_gps_pose_pub = rospy.Publisher('/ekf/pose_rel_gps', NavSatFix, queue_size=10)

        # Subscribers
        rospy.Subscriber('/mavros/global_position/raw/fix', NavSatFix, self.gps_callback)
        rospy.Subscriber('/mavros/global_position/rel_alt', Float64, self.alt_callback)
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.abs_alt_callback)
        rospy.Subscriber('/imu/imu', Imu, self.imu_callback)
        rospy.Subscriber('/imu/magnetic_field', MagneticField, self.mag_callback)

        # Sensor values
        self.rel_alt = 0.0
        self.abs_alt = 0.0
        self.mag_yaw = None
        self.latest_gps_covariance = [0.0]*9
        self.latest_gps_cov_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

    def gps_callback(self, msg):
        x_gps, y_gps = self.proj_utm(msg.longitude, msg.latitude)
        z_gps = self.rel_alt  # EKF still uses rel_alt for Z

        # Orientation for transforming offset
        q = quaternion_from_euler(self.ekf.x[6], self.ekf.x[7], self.ekf.x[8])
        R = quaternion_matrix(q)[:3, :3]

        # Apply IMU→GPS offset to get IMU position
        T_world = R @ self.T_gps_imu
        x = x_gps - T_world[0]
        y = y_gps - T_world[1]
        z = z_gps - T_world[2]

        yaw = self.mag_yaw if self.mag_yaw is not None else self.ekf.x[8]
        z_meas = np.array([x, y, z, 0, 0, yaw])
        self.update_measurement(z_meas)

        # Cache GPS covariance
        self.latest_gps_covariance = list(msg.position_covariance)
        self.latest_gps_cov_type = msg.position_covariance_type

    def alt_callback(self, msg):
        self.rel_alt = msg.data

    def abs_alt_callback(self, msg):
        self.abs_alt = msg.altitude

    def mag_callback(self, msg):
        mx = msg.magnetic_field.x
        my = msg.magnetic_field.y
        self.mag_yaw = np.arctan2(my, mx)

    

    def imu_callback(self, msg):
        now = rospy.Time.now()
        dt = (now - self.prev_time).to_sec()
        self.prev_time = now
        dt = max(dt, 0.01)

        self.ekf.F = np.eye(9)
        for i in range(3):
            self.ekf.F[i, i+3] = dt
        self.ekf.predict()

        # Original IMU orientation
        q_orig = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        # Fixed transform: rotate 180° around Z (flip X/Y)
        q_correction = quaternion_from_euler(0, 0, np.pi)
        q_fixed = quaternion_multiply(q_correction, q_orig)

        roll, pitch, yaw = euler_from_quaternion(q_fixed)
        self.ekf.x[6] = roll
        self.ekf.x[7] = pitch
        if self.mag_yaw is not None:
            self.ekf.x[8] = self.mag_yaw

        self.publish_pose()


    def update_measurement(self, z):
        def hx(x):
            return np.array([x[0], x[1], x[2], 0, 0, x[8]])

        def H_jacobian(x):
            H = np.zeros((6, 9))
            H[0, 0] = 1
            H[1, 1] = 1
            H[2, 2] = 1
            H[5, 8] = 1
            return H

        self.ekf.update(z, H_jacobian, hx)

    def publish_pose(self):
        # Publish pose in UTM/ENU
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = self.ekf.x[0]
        pose.pose.position.y = self.ekf.x[1]
        pose.pose.position.z = self.ekf.x[2]

        q = quaternion_from_euler(self.ekf.x[6], self.ekf.x[7], self.ekf.x[8])
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        self.pose_pub.publish(pose)

        # Convert UTM -> lat/lon
        lon, lat = self.proj_utm(self.ekf.x[0], self.ekf.x[1], inverse=True)

        # Absolute altitude version (correct)
        gps_msg = NavSatFix()
        gps_msg.header.stamp = rospy.Time.now()
        gps_msg.header.frame_id = "gps"
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = self.abs_alt
        gps_msg.status.status = 0
        gps_msg.status.service = 1
        gps_msg.position_covariance = self.latest_gps_covariance
        gps_msg.position_covariance_type = self.latest_gps_cov_type
        self.gps_pose_pub.publish(gps_msg)

        # Relative altitude version
        rel_gps_msg = NavSatFix()
        rel_gps_msg.header.stamp = rospy.Time.now()
        rel_gps_msg.header.frame_id = "gps"
        rel_gps_msg.latitude = lat
        rel_gps_msg.longitude = lon
        rel_gps_msg.altitude = self.rel_alt
        rel_gps_msg.status.status = 0
        rel_gps_msg.status.service = 1
        rel_gps_msg.position_covariance = self.latest_gps_covariance
        rel_gps_msg.position_covariance_type = self.latest_gps_cov_type
        self.rel_gps_pose_pub.publish(rel_gps_msg)


if __name__ == '__main__':
    try:
        ekf = PythonRobotLocalizationEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass