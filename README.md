# Localization and Detection Modules

This README covers the usage of `localization.py` and `detector.py` modules for UAV-based object detection and geolocation.

## Overview

The system provides two main components:

-   **`localization.py`**: Core localization engine with online clustering and coordinate transformations
-   **`detector.py`**: High-level detection wrapper that combines YOLO detection with localization

## Module: `localization.py`

### Clusterer Class

Online clustering for 3D coordinate points. Assigns points to existing clusters or creates new ones based on distance threshold.

#### Basic Usage

```python
from localization import Clusterer

# Initialize clusterer
clusterer = Clusterer(
    distance_threshold=2.5,  # Maximum distance for cluster assignment
    method='centroid'        # 'centroid', 'nearest', or 'average'
)

# Add single point
point = np.array([1.0, 2.0, -5.0])  # NED coordinates
cluster_id = clusterer.add_point(point)

# Add multiple points
points = np.array([[1.1, 2.1, -5.1], [10.0, 15.0, -3.0]])
cluster_ids = clusterer.add_batch(points)

# Get cluster information
info = clusterer.get_cluster_info()
for cluster_id, details in info.items():
    print(f"Cluster {cluster_id}: {details['num_points']} points, centroid: {details['centroid']}")
```

#### Clustering Methods

-   **`centroid`**: Compare distance to cluster centroid (fastest)
-   **`nearest`**: Compare distance to nearest point in cluster (most conservative)
-   **`average`**: Compare average distance to all points in cluster (most balanced)

### Tracker Class

Main localization class that combines detection with coordinate transformation.

#### Initialization

```python
from localization import Tracker
from utils.camera import Camera
from utils.calibration import UAVCalibration

# Load camera and calibration
cam = Camera.load_config('camchain.yaml')
calib = UAVCalibration()

# Initialize tracker
tracker = Tracker(
    camera=cam,
    calibration=calib,
    distance_threshold=2.5,  # Clustering threshold in meters
    method='centroid'        # Clustering method
)
```

#### Core Methods

**`pixel2NED(pixel_coord, translation, rotation)`**

Transform pixel coordinates to NED world coordinates.

```python
# Input data
pixel_coord = np.array([320, 240])  # Image pixel coordinates
translation = np.array([100, 50, -10])  # UAV position in NED
rotation = np.eye(3)  # IMU to NED rotation matrix

# Get world coordinates
ned_coord = tracker.pixel2NED(pixel_coord, translation, rotation)
print(f"NED coordinates: {ned_coord}")  # [northing, easting, down]
```

**`pixel2GPS(pixel_coord, translation, rotation, zone)`**

Transform pixel coordinates to GPS latitude/longitude.

```python
zone = "17T"  # UTM zone
gps_coord = tracker.pixel2GPS(pixel_coord, translation, rotation, zone)
print(f"GPS coordinates: {gps_coord}")  # [latitude, longitude]
```

**`track(pixel_coord, translation, rotation, toGPS=False, zone=None)`**

Perform detection tracking with clustering.

```python
# Track detection and assign cluster ID
world_coord, cluster_id = tracker.track(
    pixel_coord=np.array([320, 240]),
    translation=translation,
    rotation=rotation,
    toGPS=True,      # Return GPS coordinates
    zone="17T"       # UTM zone for GPS conversion
)

print(f"Location: {world_coord}, Cluster ID: {cluster_id}")
```

### Coordinate Systems

The tracker handles multiple coordinate transformations:

1. **Pixel Frame**: Image coordinates (origin at top-left)
2. **Camera Frame**: 3D camera coordinates with optical axis as Z
3. **IMU/Body Frame**: UAV body-fixed coordinates
4. **NED Frame**: North-East-Down world coordinates
5. **GPS Frame**: Latitude/Longitude coordinates

**Transformation Pipeline**: `Pixel → Camera → IMU/Body → NED → GPS`

## Module: `detector.py`

### PersonDetector Class

High-level wrapper that combines YOLO detection with localization and clustering.

#### Initialization

```python
from detector import PersonDetector

# Initialize with default settings
detector = PersonDetector()

# Custom configuration
detector = PersonDetector(
    model_path='checkpoints/11x_ft.pt',     # YOLO model path
    camera_config='camchain.yaml',          # Camera configuration
    confidence_threshold=0.67,              # Detection confidence threshold
    distance_threshold=2.5,                 # Clustering distance threshold
    clustering_method='centroid',           # Clustering method
    device='cuda',                          # Inference device
    manual_rotation_deg=-43.0               # Manual yaw calibration
)
```

#### Core Method: `process_frame`

Process a single frame for detection and localization.

```python
# Basic detection only
result = detector.process_frame(
    frame=image,                    # Input image (np.ndarray)
    enable_localization=False       # Disable localization
)

# Full localization pipeline
result = detector.process_frame(
    frame=image,
    translation=translation,        # UAV position in NED
    rotation=rotation,              # IMU to NED rotation matrix
    enable_localization=True,       # Enable localization
    draw_annotations=True,          # Draw bounding boxes
    return_full_frame=True,         # Return annotated frame
    toGPS=True                      # Output GPS coordinates
)
```

#### DetectionResult Structure

The `process_frame` method returns a `DetectionResult` with:

```python
@dataclass
class DetectionResult:
    """Result from processing a single frame."""
    crops: List[np.ndarray]  # bounding box images
    localizations: List[np.ndarray]  # GPS coordinates
    cluster_ids: List[int]  # IDs for the corresponding detections
    frame: Optional[np.ndarray] = None  # annotated frame / original frame
```

#### Example Usage

```python
# Process frame
result = detector.process_frame(
    frame=frame,
    translation=uav_position,
    rotation=uav_orientation,
    enable_localization=True,
    toGPS=True
)

# Access results
for i, (crop, location, cluster_id) in enumerate(zip(
    result.crops,
    result.localizations,
    result.cluster_ids
)):
    print(f"Detection {i}:")
    print(f"  Crop shape: {crop.shape}")
    print(f"  GPS location: {location}")
    print(f"  Cluster ID: {cluster_id}")

    # Save crop
    cv2.imwrite(f"detection_{i}_cluster_{cluster_id}.jpg", crop)
```

#### State Management

```python
# Reset detector state (for new sequences)
detector.reset_state()

# Reset clustering (start fresh tracking)
detector.reset_clustering()

# Get cluster information
cluster_info = detector.get_cluster_info()
for cluster_id, info in cluster_info.items():
    print(f"Cluster {cluster_id}: {info['num_points']} detections")
```

## Complete Workflow Example

```python
import numpy as np
import cv2
from detector import PersonDetector
from utils.rosbag_utils import image_stream

# Initialize detector
detector = PersonDetector(
    confidence_threshold=0.7,
    distance_threshold=3.0,
    clustering_method='centroid'
)

# Process ROS bag data
for ts, frame, translation, rotation, zone in image_stream(
    bag_path="data.bag",
    frame_topic="/camera/image_raw",
    pose_topic="/mavros/local_position/pose",
    loc=True
):

    # Process frame
    result = detector.process_frame(
        frame=frame,
        translation=translation,
        rotation=rotation,
        enable_localization=True,
        toGPS=True
    )

    # Handle detections
    if len(result.crops) > 0:
        print(f"Frame {ts}: Found {len(result.crops)} detections")

        for i, (crop, gps_coord, cluster_id) in enumerate(zip(
            result.crops, result.localizations, result.cluster_ids
        )):
            # Save crop with metadata
            filename = f"detection_t{ts:.3f}_c{cluster_id}_n{i}.jpg"
            cv2.imwrite(filename, crop)

            # Log GPS coordinates
            print(f"  Detection {i}: GPS {gps_coord}, Cluster {cluster_id}")

# Final clustering report
cluster_info = detector.get_cluster_info()
print(f"\nFinal Results: {len(cluster_info)} unique objects tracked")
```

## Configuration Files

### Camera Configuration (`camchain.yaml`)

```yaml
cam0:
    intrinsics: [fx, fy, cx, cy]
    distortion_coeffs: [k1, k2, p1, p2, k3]
    distortion_model: "radtan"
    resolution: [width, height]
```

### Required Dependencies

-   NumPy
-   OpenCV
-   PyTorch
-   Ultralytics YOLO
-   SciPy
-   Loguru
-   TQDM

## Performance Notes

-   **GPU Acceleration**: Use CUDA for faster YOLO inference
-   **Memory Management**: Crops are stored in memory; consider batch processing for large datasets
-   **Clustering Efficiency**: 'centroid' method is fastest for real-time applications
-   **Coordinate Precision**: GPS coordinates have ~1-2 meter accuracy depending on altitude and calibration quality
