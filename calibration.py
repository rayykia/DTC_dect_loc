import numpy as np

class UAVCalibration:
    """Calibration parameters for the UAV.
    
    Body frame is in the same orientation as  the IMU frame but with an offset.
    """
    def __init__(self):
        self.T_ci = np.array([
            [ 0.011106298412152327,  0.9999324199187616,  0.0034359468839849595, 0.036802732375442404],
            [-0.999832733821092,  0.01115499451474039,  -0.014493808237225339,  -0.008332238900780303],
            [-0.014531156713131006 , -0.0032743996068676073, 0.9998890557415823,  -0.08775357009176091],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
        ])
        self.R_ci = self.T_ci[:3, :3]
        self.R_ic = self.R_ci.T  # Rotation: camera to IMU
        self.t_imu2cam = self.T_ci[:3, 3]  # Translation from IMU to camera in camera frame
        
        self.t_body2imu = np.array([-0.0389588892, 0, -0.2796108098])  # body to IMU in IMU frame
        self.t_imu2body = -self.t_body2imu  # IMU to body in body frame

        self.t_body2cam = (self.T_ci @ np.hstack((self.t_body2imu, [1])).reshape(-1, 1)).flatten()[:3]  # body to camera in camera frame
        self.t_body2cam_imu = (self.R_ic @ self.t_body2cam.reshape(-1, 1)).flatten()  # body to camera in IMU frame

        self.t_cam2body = -self.t_body2cam_imu
        
    def get_alt_cam(self, alt_body, R_wi):
        """
        Get the altitude of the camera in the world frame.

        Args:
            alt_body: Altitude in body frame.
            R_wi: Rotation from world to IMU frame.
        
        Return: 
            alt_cam (float): Altitude of the camera in the NED frame.
        """
        alt_offset = (R_wi @ self.t_body2cam_imu.reshape(-1, 1)).flatten()[-1]
        assert alt_offset <= 0, "Drone upside down?"
        alt_cam = alt_body - (R_wi @ self.t_body2cam_imu.reshape(-1, 1)).flatten()[-1]
        return alt_cam