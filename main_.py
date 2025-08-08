import numpy as np
import torch
from ultralytics import YOLO
import cv2
import os
from loguru import logger
from tqdm import tqdm
import shutil


from camera import Camera
from tracking_utils import box_center, set_device
from rosbag_utils import bundeled_data_from_bag, image_stream, camera_config
from viz_utils import save_video
from apriltag_utils import AprilTagDetector
import argparse




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
        # self.t_imu2body = np.array([11.0, 0.0, 26.0])  # IMU to body in body frame
        self.t_body2cam = (self.T_ci @ np.hstack((self.t_body2imu, [1])).reshape(-1, 1)).flatten()[:3]  # body to camera in camera frame
        self.t_body2cam_imu = (self.R_ic @ self.t_body2cam.reshape(-1, 1)).flatten()  # body to camera in IMU frame
        # self.t_cam2body = -(self.R_ic @ self.t_imu2cam.reshape(-1, 1)).flatten()  # camera to body in IMU/body frame
        self.t_cam2body = -self.t_body2cam_imu
        
    def get_alt_cam(self, alt_body, R_wi):
        """
        Get the altitude of the camera in the world frame.
        :param alt_body: Altitude in body frame.
        :param R_wi: Rotation from world to IMU frame.
        :return: Altitude of the camera in the world frame.
        """
        alt_offset = (R_wi @ self.t_body2cam_imu.reshape(-1, 1)).flatten()[-1]
        assert alt_offset <= 0, "Drone upside down?"
        alt_cam = alt_body - (R_wi @ self.t_body2cam_imu.reshape(-1, 1)).flatten()[-1]
        return alt_cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Localization from the UAV")
    parser.add_argument('--save_vid', action='store_true', help='Save video from the frames.')
    parser.add_argument('--loc', action='store_true', help='Run localization.')
    args = parser.parse_args()

    j = 3
    #############################################################################
    # bag_pth = '/mnt/ENCRYPTED/workshop2/20250311/course-1/dione/course_1.bag'
    # bag_pth = f'/mnt/UNENCRYPTED/ruichend/seq/seq{j}/seq_{j}.bag'
    # save_frames_to = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_frames'
    # if args.save_vid:
    #     if args.loc:
    #         vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_loc.mp4'
    #     else:
    #         vid_pth = f'/mnt/UNENCRYPTED/ruichend/results/seq{j}_dect.mp4'
    #############################################################################
    bag_pth = "/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/dry_run_1.bag"
    save_frames_to = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_april'
    if args.save_vid:
        if args.loc:
            vid_pth = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_loc.mp4'
        else:
            vid_pth = '/mnt/UNENCRYPTED/ruichend/results/dry_run_1_dect.mp4'
    #############################################################################


    pose_topic = '/mavros/local_position/pose'
    frame_topic = '/camera/image_color/compressed'
    gps_topic = '/mavros/global_position/raw/fix'
    imu_topic = '/imu/imu'
    logger.info('Loding data from rosbag...')

    # data_bundel = bundeled_data_from_bag(bag_pth, frame_topic, pose_topic)

    logger.info('Done.')

    device = set_device()
    

    logger.info("Loading YOLO model...")
    ckpt = 'checkpoints/11x_ft.pt'
    model = YOLO(ckpt)
    model.to(device)
        
        
    cam = Camera.load_config('camchain.yaml')
    logger.info(cam)


    box_color = (255, 0, 0)  # Red
    thickness = 5  # Thicker box
    
    logger.info(f"Saving frames to `{save_frames_to}`.")
    if os.path.isdir(save_frames_to):
        shutil.rmtree(save_frames_to)
        logger.info(f"Removed previous frames: {save_frames_to}")
    os.mkdir(save_frames_to)

    if args.save_vid and os.path.isfile(vid_pth):
            os.remove(vid_pth)



    calib = UAVCalibration()
    i = 0
    confidence_threshold = 0.67
    logger.info(f"YOLO confidence threshold: {confidence_threshold}")
    
    coords = []
    detections = []
    for ts, frame, translation, R_wi, zone in image_stream(
        bag_pth, 
        frame_topic, 
        pose_topic, 
        loc=args.loc, 
        use_pose_rot = False, 
        gps_topic=gps_topic, 
        imu_topic=imu_topic
    ):
        
        
        result = model.predict(frame, verbose=False, conf=confidence_threshold)[0]
        
        if args.loc:
            R_wc = R_wi @ calib.R_ic
            # R_wd = R_wi @ R_id
            # R_cd = R_ic.T @ R_id
            # R_wc = R_dw.T @ R_dc
            if i == 0:
                starting_coord = translation.copy()

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            img_coord = box_center(box, center=False)

            if args.loc:
                
                
                
                img_coord = img_coord.reshape(-1, 2)        
                # long_uav, lat_uav, alt_uav = translation
                # zone, easting, northing = LLtoUTM(23, lat_uav, long_uav)
                northing, easting, alt_uav = translation  # NED
                ray_cam = (cam.reproject(img_coord)).reshape(-1, 1)
                ray_body =(calib.R_ic @ (ray_cam)).flatten() # ray in body frame
                ray_world = (R_wc @ (ray_cam)).flatten()  # ray in world frame
                
                
                alt_cam = calib.get_alt_cam(alt_uav, R_wi)
                s = -(alt_cam)/ray_world[-1]
                
                
                
                target2body = s * ray_body + calib.t_cam2body
                target2body_world = (R_wi @ target2body.reshape(-1, 1)).flatten()
                
                world_coord = translation + target2body_world
                
                # print(f"Alt_uav: {alt_uav}, Alt_cam: {alt_cam}")
                # print(f"s: {s}")
                # print(f"Ray Cam: {ray_cam.flatten()}")
                # print(f"Ray World: {ray_world.flatten()}")
                # print(f"Trans: {[northing, easting, alt_uav]}")
                # print(f"World Coord: {[world_coord[0], world_coord[1], world_coord[2]]}")
                # print("======================================================")
                
                
                coords.append(world_coord)
                
                
                # label_coord = world_coord - starting_coord

                label = f"({np.round(world_coord - starting_coord, 2)})"

            else:
                conf = box.conf.item()
                label = f"Person {conf:.2f}"
                # Log bounding box: frame_id, x1, y1, x2, y2, confidence
                # detections.append([i, xyxy[0], xyxy[1], xyxy[2], xyxy[3]])

            # Draw bounding box
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), box_color, thickness)
            
            # Draw img_coord
            for pt in img_coord:
                cv2.circle(frame, pt, radius=4, color=(0, 255, 0), thickness=-1) 

            # Prepare label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame,
                        (xyxy[0], xyxy[1] - text_h - 10),
                        (xyxy[0] + text_w, xyxy[1]),
                        box_color, -1)

            # Draw text label on top
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i += 1

        if args.save_vid:
            # Always save the frame, even if no person is found
            cv2.imwrite(os.path.join(save_frames_to, f'frame_{i:06d}.jpg'), frame)
    
    # np.save('logs/dry_run_1_detections.npy', np.array(detections, dtype=np.int32))
    # np.save('logs/dry_run_1_half.npy', coords)
    if args.save_vid:
        save_video(vid_pth, save_frames_to)