import numpy as np
import torch
from ultralytics import YOLO
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from utils.camera import Camera
from utils.tracking_utils import box_center, set_device
from localization import Tracker
from utils.calibration import UAVCalibration
from utils.converter import LLtoUTM


@dataclass
class DetectionResult:
    """Result from processing a single frame."""
    crops: List[np.ndarray]  # bounding box images
    localizations: List[np.ndarray]  # GPS coordinates
    cluster_ids: List[int]  # IDs for the corresponding detections
    frame: Optional[np.ndarray] = None  # annotated frame / original frame


class PersonDetector:
    """Wraps the detection and localization pipeline."""
    
    def __init__(
        self,
        model_path: str = 'checkpoints/11x_ft.pt',
        camera_config: str = 'camchain.yaml',
        confidence_threshold: float = 0.67,
        distance_threshold: float = 2.5,
        clustering_method: str = 'centroid',
        device: str = 'cuda',
        manual_rotation_deg: float = -43.0
    ):
        """Initialize the detector with all necessary components.
        
        Args:
            model_path (str): Path to YOLO model checkpoint.
            camera_config (str): Path to camera configuration YAML.
            confidence_threshold (float): YOLO detection confidence threshold.
            distance_threshold (float): Clustering distance threshold for localization.
            clustering_method (str): Clustering method ('centroid', 'nearest', 'average').
            device (str): Device for model inference ('cuda' or 'cpu').
            manual_rotation_deg (float): Manual add yaw to IMU orientation in degree.
        """
        # Set device
        self.device = set_device(device)
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        
        # Load camera and calibration
        self.cam = Camera.load_config(camera_config)
        self.calib = UAVCalibration()
        
        # Initialize tracker for localization
        self.tracker = Tracker(
            self.cam, 
            self.calib, 
            distance_threshold=distance_threshold, 
            method=clustering_method
        )
        
        # Manual calibration rotation
        theta = np.radians(manual_rotation_deg)
        self.Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        
        # Visualization settings
        self.box_color = (255, 0, 0)  # Red
        self.thickness = 5
        
        # State for localization
        self.starting_coord = None
        self.frame_count = 0
        
        self.zone = None
    
    def process_frame(
        self,
        frame: np.ndarray,
        translation: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        enable_localization: bool = True,
        draw_annotations: bool = True,
        return_full_frame: bool = False,
        toGPS: bool = False
    ) -> DetectionResult:
        """Process a single frame for detection and tracking (localization + ID).
        
        Args:
            frame (np.ndarray): Input image frame
            translation (np.ndarray): UAV translation vector in NED frame
            rotation (np.ndarray): UAV rotation matrix (IMU to NED)
            enable_localization (bool): Whether to perform localization
            draw_annotations (bool): Whether to draw bounding boxes and labels on frame
            return_full_frame (bool): Whether to return the full annotated frame
            toGPS (bool): Wnetner output GPS coordinates instead of NED coordinates.
            
        Returns:
            DetectionResult containing crops, localizations, and cluster IDs
        """
        result = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)[0]
        
        # if localization
        if enable_localization and translation is not None and rotation is not None:
            rotation = self.Rz @ rotation
            if self.starting_coord is None:
                self.starting_coord = translation.copy()
            if self.zone is None:
                self.zone, _, _ = LLtoUTM(23, self.starting_coord[1], self.starting_coord[0])
        
        crops = []
        localizations = []
        cluster_ids = []
        annotated_frame = frame.copy() if draw_annotations or return_full_frame else None
        
        # for each detection
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            pixel_coord = box_center(box, center=False)
            confidence = box.conf.item()
            
            # Extract crop from bounding box
            x1, y1, x2, y2 = xyxy
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            crop = frame[y1:y2, x1:x2].copy()
            crops.append(crop)
            
            if enable_localization and translation is not None and rotation is not None:
                # Perform localization
                world_coord, cluster_id = self.tracker.track(
                    pixel_coord, translation, rotation, toGPS, self.zone
                )
                
                localizations.append(world_coord)
                cluster_ids.append(cluster_id)
                
                if draw_annotations and annotated_frame is not None:
                    label = f"({np.round(world_coord - self.starting_coord, 2)}, ID: {cluster_id})"
            else:
                # No localization - append None/default values
                localizations.append(None)
                cluster_ids.append(-1)  # -1 indicates no clustering
                
                if draw_annotations and annotated_frame is not None:
                    label = f"Person {confidence:.2f}"
            
            # Draw annotations on frame if requested
            if draw_annotations and annotated_frame is not None:
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.box_color, self.thickness)
                
                # Prepare label background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame,
                            (x1, y1 - text_h - 10),
                            (x1 + text_w, y1),
                            self.box_color, -1)
                
                # Draw text label
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        self.frame_count += 1
        
        return DetectionResult(
            crops=crops,
            localizations=localizations,
            cluster_ids=cluster_ids,
            frame=annotated_frame if return_full_frame else None
        )
    
    def reset_state(self):
        """Reset detector state (for processing new sequences)."""
        self.starting_coord = None
        self.frame_count = 0
        # Note: Tracker clustering state is preserved unless explicitly reset
    
    def reset_clustering(self):
        """Reset the clustering state in the tracker."""
        self.tracker = Tracker(
            self.cam, 
            self.calib, 
            distance_threshold=self.tracker.clusterer.distance_threshold, 
            method=self.tracker.clusterer.method
        )
    
    def get_cluster_info(self) -> Dict:
        """Get information about current clusters."""
        return self.tracker.clusterer.get_cluster_info()



if __name__ == "__main__":
    # Example usage
    detector = PersonDetector()
    
    # Process a single frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.process_frame(dummy_frame, enable_localization=False)
    
    print(f"Detected {len(result.crops)} objects")
    print(f"Got {len(result.localizations)} localizations")
    print(f"Cluster IDs: {result.cluster_ids}")
    
    # Example for localization
    dummy_translation = np.array([0, 0, -10])  # NED coordinates
    dummy_rotation = np.eye(3)  # Identity rotation matrix
    
    result_loc = detector.process_frame(
        dummy_frame, 
        translation=dummy_translation,
        rotation=dummy_rotation,
        enable_localization=True,
        return_full_frame=True
    )
    
    for i, (crop, loc, cluster_id) in enumerate(zip(result_loc.crops, result_loc.localizations, result_loc.cluster_ids)):
        print(f"Detection {i}: crop_shape={crop.shape}, world_coord={loc}, cluster_id={cluster_id}")