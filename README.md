# Localization Module

## Overview

`localization.py` provides tools for **localizing objects in 3D space** from UAV imagery and GPS/IMU data.  
It includes:

-   **Online clustering** of detected 3D points for static object tracking.
-   **Camera-to-world coordinate transformations** using calibration data.
-   **Integration with GPS/UTM coordinate conversions** for georeferencing detections.

This module is intended for UAV-based inspection, mapping, and object localization workflows.

---

## Features

-   **Clusterer**: Online, incremental clustering of 3D points.
    -   Distance-based assignment with configurable method (`centroid`, `nearest`, `average`).
    -   Supports single or batch point insertion.
-   **Geolocation**: Project image detections into world coordinates.
-   **Camera Model Integration**: Works with the `Camera` and `UAVCalibration` classes.
-   **GPS & UTM Conversion**: Uses `converter.py` to transform between latitude/longitude and UTM coordinates.
-   **Logging & Progress Tracking** via `loguru` and `tqdm`.

---

## Example Usage

When got data from ROS:

```python
cam = Camera.load_config('camchain.yaml')  # Load camera intrinsics.
calib = UAVCalibration()  # Load UAV calibration.
tracker = Tracker(cam, calib, distance_threshold=2.5, method='centroid')

# for a given pixel coordinate
ned_coordinate, cluster_id = tracker.track(
    pixel_coordinate,
    translation,    # translation of UAV in NED frame
    rotation,       # rotation transform the IMU/camera frame to NED frame
)
```
