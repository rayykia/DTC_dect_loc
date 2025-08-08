import numpy as np
import os
from loguru import logger
from tqdm import tqdm
from camera import Camera
from converter import LLtoUTM, UTMtoLL
from calibration import UAVCalibration



class Clusterer:
    """Online clustering for 3D coordinate points.
    Assigns points to existing clusters or creates new ones based on distance threshold.
    
    notes: This is only for static object for now.
    """
    
    def __init__(
        self, 
        distance_threshold: float = 2.5, 
        method: str = 'centroid'
    ):
        """Initialize the online clusterer.
        
        Args:
            distance_threshold (float): Maximum distance for a point to be considered part of an existing cluster.
            method (str):
                'centroid' - compare to cluster centroid
                'nearest' - compare to nearest point in cluster
                'average' - compare to average distance to all points in cluster
        """
        self.distance_threshold = distance_threshold
        self.method = method
        self.clusters = {}
        self.cluster_centroids = {}
        self.next_cluster_id = 0
        
        
    def add_point(self, point: np.ndarray) -> int:
        """Add a new point and return its cluster assignment.
        
        Args:
            point (np.ndarray): 3D coordinate point (shape: (3,))
            
        Returns:
            int : cluster ID assigned to this point
        """
        point = np.array(point).reshape(-1)
        
        if len(self.clusters) == 0:
            # First point, create first cluster
            cluster_id = self._create_new_cluster(point)
        else:
            # Find nearest cluster
            cluster_id = self._find_nearest_cluster(point)
            
            if cluster_id is None:
                # No cluster within threshold, create new one
                cluster_id = self._create_new_cluster(point)
            else:
                # Add to existing cluster
                self._add_to_cluster(cluster_id, point)
                
        return cluster_id
    
    
    def add_batch(self, points: np.ndarray) -> List[int]:
        """Add multiple points at once.
            
        Args:
            points (np.ndarray): Array of 3D points (shape: (N, 3))
            
        Returns:
            List[int] : cluster IDs for each point
        """
        cluster_ids = []
        for point in points:
            cluster_ids.append(self.add_point(point))
        return cluster_ids
    
    
    def _find_nearest_cluster(self, point: np.ndarray) -> Optional[int]:
        """Find the nearest cluster within threshold distance.
        """
        min_distance = float('inf')
        nearest_cluster = None
        
        for cluster_id in self.clusters:
            distance = self._compute_distance_to_cluster(point, cluster_id)
            
            if distance < min_distance and distance <= self.distance_threshold:
                min_distance = distance
                nearest_cluster = cluster_id
                
        return nearest_cluster
    
    def _compute_distance_to_cluster(self, point: np.ndarray, cluster_id: int) -> float:
        """Compute distance from point to cluster based on selected method.
        """
        if self.method == 'centroid':
            # Distance to cluster centroid
            return np.linalg.norm(point - self.cluster_centroids[cluster_id])
            
        elif self.method == 'nearest':
            # Distance to nearest point in cluster
            cluster_points = np.array(self.clusters[cluster_id])
            distances = np.linalg.norm(cluster_points - point, axis=1)
            return np.min(distances)
            
        elif self.method == 'average':
            # Average distance to all points in cluster
            cluster_points = np.array(self.clusters[cluster_id])
            distances = np.linalg.norm(cluster_points - point, axis=1)
            return np.mean(distances)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _create_new_cluster(self, point: np.ndarray) -> int:
        """Create a new cluster with the given point.
        """
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        self.clusters[cluster_id] = [point]
        self.cluster_centroids[cluster_id] = point.copy()
        
        return cluster_id
    
    def _add_to_cluster(self, cluster_id: int, point: np.ndarray):
        """Add point to existing cluster and update centroid.
        """
        self.clusters[cluster_id].append(point)
        
        # Update centroid incrementally
        n = len(self.clusters[cluster_id])
        old_centroid = self.cluster_centroids[cluster_id]
        self.cluster_centroids[cluster_id] = (old_centroid * (n-1) + point) / n
    
    def get_cluster_points(self, cluster_id: int) -> np.ndarray:
        """Get all points in a specific cluster.
        """
        if cluster_id in self.clusters:
            return np.array(self.clusters[cluster_id])
        else:
            return np.array([])
    
    def get_all_clusters(self) -> dict:
        """Get all clusters as a dictionary.
        """
        return {cid: np.array(points) for cid, points in self.clusters.items()}
    
    def get_cluster_info(self) -> dict:
        """Get information about all clusters.
        """
        info = {}
        for cluster_id in self.clusters:
            points = np.array(self.clusters[cluster_id])
            info[cluster_id] = {
                'num_points': len(points),
                'centroid': self.cluster_centroids[cluster_id],
                'min': points.min(axis=0),
                'max': points.max(axis=0),
                'std': points.std(axis=0) if len(points) > 1 else np.zeros(3)
            }
        return info
    
    def merge_clusters(self, cluster_id1: int, cluster_id2: int):
        """Merge two clusters into one.
        """
        if cluster_id1 not in self.clusters or cluster_id2 not in self.clusters:
            raise ValueError("Invalid cluster IDs")
            
        # Merge points from cluster2 into cluster1
        self.clusters[cluster_id1].extend(self.clusters[cluster_id2])
        
        # Update centroid
        points = np.array(self.clusters[cluster_id1])
        self.cluster_centroids[cluster_id1] = points.mean(axis=0)
        
        # Remove cluster2
        del self.clusters[cluster_id2]
        del self.cluster_centroids[cluster_id2]
        




class Tracker:
    def __init__(
        self, 
        camera: Camera, 
        calibration: UAVCalibration, 
        distance_threshold: float = 2.5, 
        method: str = 'centroid'
    ):
        """Initialize the Tracker.
        
        Args:
            camera (Camera): Camera object, initialized with the intrinsics.
            calibration (UAVValibration): Calibration of the UAV paltform.
            distance_threshold (float): Maximum distance for a point to be considered part of an existing cluster, i.e. same object.
            method (str):
                'centroid' - compare to cluster centroid
                'nearest' - compare to nearest point in cluster
                'average' - compare to average distance to all points in cluster
        """
        self.camera = camera
        self.calibration = calibration
        self.clusterer = Clusterer(distance_threshold, method)


    def pixel2NED(
            self,
            pixel_coord: np.ndarray,
            translation: np.ndarray,
            rotation: np.ndarray,
    ):
        """Run pixel localization from the UAV data. The translation and output is in NED frame.
        
        Assume body frame aligns with IMU frame in orientation.
        Pipeline: 
            1. Get ray in camera frame from pixel coordinates.
            2. Transform ray to IMU/body frame and NED frame.
            3. Get the altitude of the camera and get the scale.
            4. Align the ray using the scale in IMU/body frame.
            5. Target in body frame: t_cam2body + ray_imubodyframe
            6. Target in NED frame: GPS_in_NED + rotation@target_imubodyframe
        
        Args:
            pixel_coord (np.ndarray): Pixel coordinates in the image.
            translation (np.ndarray): Translation vector from the UAV pose, [northing, easting, altitude] in NED.
            rotation (np.ndarray): Rotation matrix from the UAV pose, IMU2NED.
        """
        pixel_coord = pixel_coord.reshape(-1, 2)  # Ensure pixel_coord is 2D
        ray_cameraFrame = self.camera.reproject(pixel_coord, fix_distortion=True)  # Reproject to get ray in camera frame
        
        ray_imubodyFrame = (self.calibration.R_ic @ ray_cameraFrame.reshape(-1, 1)).flatten()  # ray in IMU/body frame
        ray_nedFrame = (rotation @ ray_imubodyFrame.reshape(-1, 1)).flatten()  # ray in NED frame
        
        alt_cam = self.calibration.get_alt_cam(translation[-1], rotation)  # altitude of the camera in NED frame
        scale =  - alt_cam / ray_nedFrame[-1]  # scale factor to align the ray with the altitude, note alt_cam < 0
        
        ray_imubodyFrame *= scale  # scale the ray in IMU/body frame

        target_imubodyFrame = self.calibration.t_cam2body + ray_imubodyFrame  # target in IMU/body frame
        
        target_nedFrame = translation + (rotation @ target_imubodyFrame.reshape(-1, 1)).flatten()  # target in NED frame
        
        return target_nedFrame


    def pixel2GPS(
            self,
            pixel_coord: np.ndarray,
            translation: np.ndarray,
            rotation: np.ndarray,
            zone: str
    ):
        """Run pixel localization from the UAV data. The translation is in NED frame.
        
        Args:
            pixel_coord (np.ndarray): Pixel coordinates in the image.
            translation (np.ndarray): Translation vector from the UAV pose, [northing, easting, altitude] in NED.
            rotation (np.ndarray): Rotation matrix from the UAV pose, IMU2NED.
            zone (str): zone for UTM to LL.
        
        Return:
            lat (float): latitude
            long (float): longitude
        """
        ned_coord = self.pixel2NED(pixel_coord, translation, rotation)
        lat, long = UTMtoLL(23, ned_coord[0], ned_coord[1], zone)
        return lat, long
    
    
    def track_pixels(
        self
    ):
        """Track the give detection(s).
        
        Args:
            pixel_coord (np.ndarray): Array of detection in the pixel frame.
        
        Returns:
            coord (np.ndarray): The detected objects' coordinate in NED frame.
            
        """










if __name__=='__main__':
    def extract_lat_lon_from_file(file_path):
        lat, lon = None, None
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip().startswith('latitude:'):
                    lat = float(line.strip().split(':')[1])
                elif line.strip().startswith('longitude:'):
                    lon = float(line.strip().split(':')[1])
        return [lat, lon] if lat is not None and lon is not None else None

    def read_all_casualty_coords(directory):
        coords = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                lat_lon = extract_lat_lon_from_file(file_path)
                if lat_lon:
                    coords.append(lat_lon)
        return coords

    # Example usage:
    directory_path = '/mnt/UNENCRYPTED/ruichend/seq/dry_run_1/GT'
    casualty_gps = np.array(read_all_casualty_coords(directory_path))
    casualty_coords = np.array([LLtoUTM(23, lat, lon) for lat, lon in casualty_gps])[:, 1:3].astype(np.float32)  # Extracting easting and northing
###################


    camera = Camera.load_config("camchain.yaml")
    logger.info(camera)
    
    ros_drone = np.load('logs/dr1_dect_pose2.npy', allow_pickle=True).tolist()
    
    calibration = UAVCalibration()
    

    target_coords = []
    
    theta = np.radians(-43)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    records = []
    tracker = Tracker(camera, calibration)
    for ros_read in tqdm(ros_drone):
        translation = ros_read['translation']
        # translation = np.array(translation) + np.array([0, 0, offset])
        rotation = ros_read['R_imu']
        rotation = Rz @ rotation # Align with NED frame
        detections = ros_read['detections']
        pixel_coords = []
        for det in detections:
            pixel_coords.append([
                (det[0] + det[2]) // 2,  # x center
                (det[1] + det[3]) // 2  # y center
            ])
            
        pixel_coords = np.array(pixel_coords).reshape(-1, 2)
        
        localization_this_frame = []
        for pixel_coord in pixel_coords:
            target_nedFrame = tracker.pixel2NED(
                pixel_coord,
                translation,
                rotation
            )
            target_coords.append(target_nedFrame)
            localization_this_frame.append(target_nedFrame)
            
        records.append({
            'translation': translation,
            'localization': np.array(localization_this_frame, dtype=np.float32)
        })
    np.save("logs/with_ts4.npy", target_coords)
    
    
    # save_heatmap_frames(
    #     records,
    #     save_path= '/mnt/UNENCRYPTED/ruichend/results/heatmap_frames'
    # )
    # save_video(
    #     output_vid='/mnt/UNENCRYPTED/ruichend/results/heatmap.mp4',
    #     src_path='/mnt/UNENCRYPTED/ruichend/results/heatmap_frames',
    #     fps=60,
    #     compress=True
    # )

    
            