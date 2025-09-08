# Bachelor_rad-CV-for-tracking

# Using computer vision to track objects in real time

This repository provides a computer vision pipeline for **real-time object tracking** in video feeds. While originally developed for analyzing football matches, the core functionalities can be adapted for various tracking applications, with specialized features for tennis court analysis. The process involves four main steps:

## 1 - Object Detection

This step utilizes **YOLOv5 PyTorch Hub inference** with a pre-trained **YOLOv5l** model to identify and locate objects within video frames.

<div align="center">
<img src="./readme_photos/det.png" alt="Detection" width="500"/>
</div>

For more details on YOLOv5, visit: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## 2 - Object Tracking

Our tracking module is built upon the **Deep SORT algorithm**, providing robust and persistent object tracking across frames.

The Deep SORT implementation is adapted from: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

## 3 - Getting tracking dataframe

After step 3, the tracking data of the video footage is generated and can be used to extract various statistics and insights.

<div align="center">
<img src="./readme_photos/init_dataframe.png" alt="Initial Tracking Data" width="500"/>
</div>

## 4 - Court Coordinate Mapping and Distance Analysis

The system includes advanced court analysis capabilities specifically designed for tennis videos:

- **Edge Detection and Homography**: Automatic detection of court boundaries using edge detection algorithms and creation of homography matrices for perspective transformation
- **Coordinate Mapping**: Transformation of pixel coordinates from video footage to real tennis court coordinates
- **Distance Calculation**: Computation of distances traveled by players and ball in real-world units (meters)
- **Movement Analysis**: Detailed analysis of player movement patterns and ball trajectory

<div align="center">
<img src="./readme_photos/debug_frame_100.jpg" alt="Court Edge Detection" width="500"/>
</div>

All measurements and analysis results are automatically saved to CSV files for further processing and analysis.

## 5 - Resolution and ROI Optimization

The system supports advanced optimization techniques to improve both speed and accuracy:

- **Resolution Optimization**: Automatically tests multiple video resolutions to find the best trade-off between processing speed (FPS) and detection/tracking accuracy.
- **ROI (Region of Interest) Optimization**: Focuses detection and tracking on a dynamically selected region of the frame (e.g., around a specific player), significantly reducing computation and improving real-time performance.

### Resolution Optimization

The resolution optimizer will automatically evaluate different downscaled versions of your video, measuring performance and accuracy at each step. This helps you select the optimal resolution for your hardware and use case.

**Example usage:**

```python
from Time_Optimization.resolution_optimizer import ResolutionPerformanceOptimizer

optimizer = ResolutionPerformanceOptimizer(target_fps=30.0, min_detection_threshold=2)

results = optimizer.optimize_resolution(
    video_path="./test_videos/melbourne2.mp4",
    out_name="tennis_test",
    teams_colors=['white', 'white', 'blue', 'blue', 'black', 'yellow'],
    ball_only=True,
    step_factor=0.8,
    min_width=320,
    test_percentage=90.0,              # Test 90% of resolutions
    prefer_higher_resolution=True      # Focus on higher quality
)

# Generate both reports
optimizer.generate_performance_report(save_path="./Out/optimization_report.txt")
optimizer.plot_performance_analysis(save_path="./Out/performance_plots.png")
optimizer.plot_comprehensive_timing_analysis(save_path="./Out/timing_analysis.png")
```

<div align="center">
<img src="./readme_photos/timing_analysis.png" alt="Resolution Optimization Results" width="500"/>
</div>

The optimizer will save a detailed report and performance plots in the `Out/` directory.

---

### ROI (Region of Interest) Optimization

ROI optimization restricts detection and tracking to a region around a specific player, greatly improving speed while maintaining accuracy for the target. The ROI is dynamically updated using Kalman filter predictions and adapts its size based on the player's movement and bounding box size.

**Example usage:**

```python
from Detect_and_Track.roi_main import run_roi_experiment

success = run_roi_experiment(
    video_path="./test_videos/melbourne2.mp4",
    experiment_name="tennis_test_roi",
    teams_colors=['white', 'white', 'red', 'red', 'black', 'yellow'],
    target_player_id=2,
    ball_only=False,
    save_roi_images=True,      # Enable ROI image saving
    roi_save_interval=15       # Save ROI image every 15 frames
)
```

After running, ROI images will be saved in the `Out/tennis_test_roi_roi_images/` directory. 

<div align="center">
<img src="./readme_photos/ROI-collage.jpg" alt="ROI Collage Example" width="500"/>
</div>

*Example: Collage of 9 ROI images from different frames, showing how the ROI adapts to the player's position.*

---


---

# Installation

To get started with this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aljaljak2/Bachelor_rad-CV-for-tracking
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Bachelor_rad-CV-for-tracking/
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

# Usage

This repository offers functionalities for both detection and tracking.

### Generating Tracking Dataframe

The primary use case is to generate a dataframe containing tracking data from a video feed.

```python
from Detect_and_Track.get_init_data import get_init_data

# Call get_init_data function with arguments:
# video path, a name for the output video and the initial dataframe (with unmapped coordinates relative to TV video),
# and a list of two teams colors as described in create_df.py file, and whether or not to save only the frames with the ball detected in them.
# Note: For general object tracking, 'teams_colors' and 'ball_only' can be adapted or disregarded based on your specific objects.
get_init_data(path, out_name, teams_colors, ball_only)

# The output video with detected objects tracked, and the initial dataframe will be saved in the 'Out/' directory.
# This initial dataframe contains coordinates relative to the original video.
```

### Object Detection and Tracking Only

To generate a video with detected and tracked objects, each with a unique ID:


```python
from Detect_and_Track.get_tracks import get_video_tracks
from Detect_and_Track.create_tracking_boxes_video import create_tracking_boxes_video

# Call get_video_tracks with the path of your video footage.
# This function will save a clean video (without zoomed-in frames) in the "Out/" folder with the chosen name.
# For more information, refer to the 'Detect_and_Track/get_tracks' file.
get_video_tracks(video_path, output_video_name)

# To create a video with tracking boxes around the objects:
# This function will save a new video in the "Out/" folder with detected and tracked objects, each assigned a unique ID.
create_tracking_boxes_video(video_path, output_video_name)
```
## Tennis Court Analysis with Coordinate Mapping

For tennis videos, you can now perform advanced court analysis with multiple processing options:

### Main Processing Pipeline (Average Corners Method)

This pipeline calculates average corners from multiple frames to provide stable and robust coordinate mapping:

```python
from tennis_court_tracker import main_video_processing_pipeline

# Process tennis video with average corner detection and coordinate mapping
df_mapped, player_distances, ball_results, average_corners, frame_info = main_video_processing_pipeline(
   video_path="path/to/tennis_video.mp4",
   data_csv_path="path/to/tracking_data.csv",
   sample_interval=1.0,  # Sample every 1 second
   max_frames=30         # Use up to 30 frames for corner detection
)


# The function will automatically:
# 1. Detect tennis court corners from multiple frames
# 2. Calculate average corners for stable mapping
# 3. Create homography matrix for coordinate transformation
# 4. Map pixel coordinates to real court coordinates
# 5. Calculate distances traveled by players and ball
# 6. Save all results to multiple CSV files with detailed analysis

```

### Dynamic Processing Pipeline

This advanced pipeline dynamically calculates distances by detecting corners at specified intervals throughout the video, providing more accurate tracking for videos with camera movement:

```python
from tennis_court_tracker import main_video_processing_pipeline_dynamic

# Process tennis video with dynamic corner detection and coordinate mapping
df_mapped, player_distances, ball_results, average_corners, frame_info = main_video_processing_pipeline_dynamic(
    video_path="path/to/tennis_video.mp4",
    data_csv_path="path/to/tracking_data.csv",
    sample_interval=1.0,  # Sample every 1 second
    max_frames=30         # Use up to 30 frames for analysis
)

# This dynamic approach:
# 1. Detects court corners for each individual frame
# 2. Updates homography matrix dynamically as needed
# 3. Provides more accurate mapping for videos with camera motion
# 4. Automatically handles perspective changes throughout the video
```

### Single Frame Corner Detection
For testing and visualization purposes, you can detect and visualize corners on a single frame:

```python
import cv2
from tennis_court_tracker.corner_detection import CornerDetector

# Load a single frame for corner detection
frame_path = "./test_videos/frame_for_detection.png"
frame = cv2.imread(frame_path)

# Initialize corner detector
corner_detector = CornerDetector()

# Detect corners on the frame
corners = corner_detector.detect_court_corners(frame, debug=True)

# Draw detected corners on the frame
frame_with_corners = corner_detector.draw_corners_on_frame(frame, corners)

# Save the result
cv2.imwrite("./Out/corners_visualization.png", frame_with_corners)
```

