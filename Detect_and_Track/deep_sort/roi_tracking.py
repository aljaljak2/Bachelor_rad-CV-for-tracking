# roi_tracking.py - Enhanced ROI tracking implementation with fair comparison

import os
import cv2
import numpy as np
import time
import csv
from .utils import read_class_names
from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from .tracker import Tracker
from .generate_detections import create_box_encoder

YOLO_COCO_CLASSES = "./Detect_and_Track/model_data/coco/coco.names"

def trackingXl5_ROI_single_player(Yolo_model, ball_model, video_path, target_player_id=1,
                                  save_roi_images=False, roi_save_interval=15,
                                 experiment_name="roi_experiment"):
    '''
    Enhanced ROI tracking that optimizes both detection and tracking steps.
    Tracks only one specific player using Kalman filter predictions with full ROI optimization.
    Ball detection and tracking are EXCLUDED for fair comparison with standard approach.

    Parameters
    ----------
    Yolo_model : pytorch model
        pytorch YoloV5l model.
    ball_model : pytorch model
        pytorch YoloV5l model to detect the ball specifically (not used but kept for compatibility).
    video_path : string
        the path of the directory of the processed video.
    target_player_id : int
        ID of the player to track with ROI (default: 1)
    save_roi_images : bool
        Whether to save ROI images during tracking (default: False)
    roi_save_interval : int
        Save ROI image every N frames (default: 15)
    experiment_name : str
        Name for the experiment (used in ROI image saving)

    Return
    ----------
    frames : list
        list of frames of the video with objects tracked.
    tboxes : list
        list of every object tracked in every frame.
    fps : int
        number of frames per second used in processing
    '''

    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # Initialize deep sort object
    model_filename = './Detect_and_Track/model_data/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    if video_path:
        vid = cv2.VideoCapture(video_path)

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f'fps of the input video = {fps}')
    print('Please wait ... \n')
    NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    frames = []
    tboxes = []

    # ROI tracking variables
    target_track = None
    roi_active = False
    roi_frames_used = 0
    first_frame_processed = False
    # For adaptive ROI
    next_roi_coords = None  # ROI to use for the next frame
    prev_center = None  # For movement-based margin

    # Initialize timing variables
    frame_count = 0
    timing_data = []

    # Ensure output directory exists
    os.makedirs('./Out', exist_ok=True)

    # Create ROI images directory if needed
    if save_roi_images:
        roi_images_dir = f'./Out/{experiment_name}_roi_images'
        os.makedirs(roi_images_dir, exist_ok=True)

    # Create CSV filename
    csv_filename = f'./Out/roi_detailed_frame_timing_{int(time.time())}.csv'

    while True:
        # Start timing for this frame
        frame_start_time = time.time()

        _, frame = vid.read()
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.resize(original_frame, (1280, 720))
            frames.append(original_frame)
        except:
            break

        frame_count += 1
        print(f"Frame {frame_count} - Processing started at: {frame_start_time:.6f}")

        # Detection timing
        detection_start = time.time()

        # Use ROI from previous frame's prediction
        if roi_active and next_roi_coords is not None:
            x1, y1, x2, y2 = next_roi_coords
            roi_frame = original_frame[y1:y2, x1:x2]

            # Save ROI image if requested
            if save_roi_images and frame_count % roi_save_interval == 0:
                roi_image_path = f'{roi_images_dir}/frame_{frame_count:06d}_roi.jpg'
                cv2.imwrite(roi_image_path, cv2.cvtColor(roi_frame, cv2.COLOR_RGB2BGR))

            # Run detection on ROI
            results = Yolo_model(roi_frame)
            pred_bbox = results.xyxy[0].tolist()

            # Adjust coordinates back to full frame
            adjusted_bboxes = []
            for box in pred_bbox:
                adjusted_box = [
                    box[0] + x1,  # x1
                    box[1] + y1,  # y1
                    box[2] + x1,  # x2
                    box[3] + y1,  # y2
                    box[4],       # confidence
                    box[5]        # class
                ]
                adjusted_bboxes.append(adjusted_box)

            pred_bbox = adjusted_bboxes
            bboxes = [np.array(box) for box in pred_bbox]
            roi_frames_used += 1

            print(f"Frame {frame_count} - Using ROI: ({x1},{y1}) to ({x2},{y2})")

        else:
            # First frame or when target is lost - process full frame
            results = Yolo_model(original_frame)
            pred_bbox = results.xyxy[0].tolist()
            bboxes = [np.array(box) for box in pred_bbox]
            print(f"Frame {frame_count} - Processing full frame")

        # Extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            boxes.append([bbox[0].astype(int), bbox[1].astype(int),
                         bbox[2].astype(int)-bbox[0].astype(int),
                         bbox[3].astype(int)-bbox[1].astype(int)])
            scores.append(bbox[4])
            names.append(NUM_CLASS[int(bbox[5])])

        detection_end = time.time()

        # Tracking timing
        tracking_start = time.time()

        # ROI-optimized feature extraction
        if roi_active and next_roi_coords is not None and len(boxes) > 0:
            x1, y1, x2, y2 = next_roi_coords
            roi_frame_for_features = original_frame[y1:y2, x1:x2]

            # Adjust box coordinates relative to ROI for feature extraction
            roi_boxes = []
            valid_indices = []
            for i, box in enumerate(boxes):
                roi_x = max(0, box[0] - x1)
                roi_y = max(0, box[1] - y1)
                roi_w = min(x2 - x1 - roi_x, box[2])
                roi_h = min(y2 - y1 - roi_y, box[3])
                if roi_w > 10 and roi_h > 10:
                    roi_boxes.append([roi_x, roi_y, roi_w, roi_h])
                    valid_indices.append(i)
                else:
                    roi_boxes.append([1, 1, 10, 10]) # Fallback for tiny boxes, ensures encoder doesn't error
                    valid_indices.append(i)
            try:
                if len(roi_boxes) > 0:
                    features = np.array(encoder(roi_frame_for_features, roi_boxes))
                    print(f"Frame {frame_count} - ROI feature extraction: {len(roi_boxes)} boxes")
                else:
                    features = np.array([])
            except Exception as e:
                print(f"Frame {frame_count} - ROI feature extraction failed: {e}")
                features = np.array(encoder(original_frame, boxes))
                print(f"Frame {frame_count} - Fallback to full frame feature extraction")
        else:
            features = np.array(encoder(original_frame, boxes))

        # Filter out ball detections (same as standard approach for fair comparison)
        non_ball_detections_indices = [i for i, name in enumerate(names) if name != 'sports ball']
        filtered_boxes = [boxes[i] for i in non_ball_detections_indices]
        filtered_scores = [scores[i] for i in non_ball_detections_indices]
        filtered_names = [names[i] for i in non_ball_detections_indices]
        filtered_features = [features[i] for i in non_ball_detections_indices]

        # Create detections (excluding ball)
        detections = [Detection(bbox, score, class_name, feature)
                     for bbox, score, class_name, feature in zip(filtered_boxes, filtered_scores, filtered_names, filtered_features)]

        # ROI-optimized tracking prediction and update
        tracker.predict()

        # Filter detections for ROI tracking optimization
        if roi_active and target_track is not None and next_roi_coords is not None:
            relevant_detections = []
            x1, y1, x2, y2 = next_roi_coords
            for detection in detections:
                det_bbox = detection.to_tlbr()
                overlap_x1 = max(det_bbox[0], x1)
                overlap_y1 = max(det_bbox[1], y1)
                overlap_x2 = min(det_bbox[2], x2)
                overlap_y2 = min(det_bbox[3], y2)
                overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
                det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                # Only include detections with significant overlap (excluding ball explicitly)
                if overlap_area > 0.2 * det_area:
                    relevant_detections.append(detection)
            tracker.update(relevant_detections)
            print(f"Frame {frame_count} - ROI tracking: {len(relevant_detections)}/{len(detections)} detections processed (excluding ball)")
        else:
            tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        current_target_found = False

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_class()
            tracking_id = track.track_id
            index = key_list[val_list.index(class_name)]

            # Exclude ball (class index 32) from being added to tracked_bboxes
            if index == 32:
                continue

            # Check if this is our target player
            if tracking_id == target_player_id and not first_frame_processed:
                target_track = track
                roi_active = True
                first_frame_processed = True
                current_target_found = True
                print(f"Frame {frame_count} - Target player {target_player_id} locked for ROI tracking")
            elif roi_active and track == target_track:
                current_target_found = True

            # Only track the target player (same as standard approach for fair comparison)
            if target_track is not None and track == target_track:
                tracked_bboxes.append(bbox.tolist() + [tracking_id, index])

        # After processing tracks, set up ROI for next frame using Kalman prediction
        if roi_active and target_track is not None:
            predicted_bbox = target_track.to_tlbr()
            obj_width = predicted_bbox[2] - predicted_bbox[0]
            obj_height = predicted_bbox[3] - predicted_bbox[1]
            obj_size = max(obj_width, obj_height)
            center_x = (predicted_bbox[0] + predicted_bbox[2]) / 2
            center_y = (predicted_bbox[1] + predicted_bbox[3]) / 2
            # Calculate movement (Euclidean distance) from previous center
            if prev_center is not None:
                movement = np.linalg.norm(np.array([center_x, center_y]) - np.array(prev_center))
            else:
                movement = 0
            # Adaptive margins based on object size and movement
            min_margin = int(0.2 * obj_size + 0.5 * movement)
            max_margin = int(1.0 * obj_size + 1.0 * movement)
            # Use a margin between min and max (e.g., average)
            margin = int(0.5 * (min_margin + max_margin))
            x1 = max(0, int(predicted_bbox[0] - margin))
            y1 = max(0, int(predicted_bbox[1] - margin))
            x2 = min(1280, int(predicted_bbox[2] + margin))
            y2 = min(720, int(predicted_bbox[3] + margin))
            next_roi_coords = (x1, y1, x2, y2)
            prev_center = (center_x, center_y)
        else:
            next_roi_coords = None
            prev_center = None

        # If target is lost, disable ROI and fall back to full frame
        if roi_active and not current_target_found:
            print(f"Frame {frame_count} - Target player lost, switching back to full frame")
            roi_active = False
            target_track = None
            next_roi_coords = None

        # Ball detection is EXCLUDED (same as standard approach for fair comparison)
        # Removed the entire ball detection block here for fair comparison

        tboxes.append([[round(bb) for bb in tracked_bbox] for tracked_bbox in tracked_bboxes])

        tracking_end = time.time()

        # End timing for this frame
        frame_end_time = time.time()
        total_processing_time = frame_end_time - frame_start_time
        detection_time = detection_end - detection_start
        tracking_time = tracking_end - tracking_start
        complete_detection_tracking_time = detection_time + tracking_time

        print(f"Frame {frame_count} - Completed at: {frame_end_time:.6f}")
        print(f"Frame {frame_count} - Total time: {total_processing_time:.6f}s "
              f"(Detection: {detection_time:.6f}s, Tracking: {tracking_time:.6f}s, "
              f"Detection+Tracking: {complete_detection_tracking_time:.6f}s)")

        # Store timing data with ROI optimization indicator
        timing_data.append([frame_count, frame_start_time, frame_end_time,
                           detection_time, tracking_time, complete_detection_tracking_time,
                           total_processing_time, roi_active])

    # Write timing data to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame_Number', 'Start_Time (unix_timestamp)', 'End_Time (unix_timestamp)',
                        'Detection_Time (s)', 'Tracking_Time (s)', 'Complete_Detection_Tracking_Time (s)',
                        'Total_Processing_Time (s)', 'ROI_Active'])
        writer.writerows(timing_data)

    print(f"\nDetailed timing data saved to: {csv_filename}")
    print(f'Tracked {len(frames)} frames')
    print(f'ROI was used for {roi_frames_used} frames out of {len(frames)} total frames')

    return frames, tboxes, fps


def compare_tracking_approaches_fair(video_path, out_name, teams_colors, ball_only=False, 
                                   target_player_id=1, save_roi_images=True, roi_save_interval=15):
    """
    FAIR comparison between standard tracking approach and ROI tracking approach.
    Both approaches now track only one player and exclude ball detection.
    
    Parameters
    ----------
    video_path : str
        Path to the video file
    out_name : str
        Output name prefix for files
    teams_colors : list
        List of team colors for classification
    ball_only : bool
        Whether to keep only frames with ball detected (kept for compatibility)
    target_player_id : int
        ID of the player to track
    save_roi_images : bool
        Whether to save ROI images during tracking
    roi_save_interval : int
        Save ROI image every N frames
        
    Returns
    -------
    dict
        Comparison results with timing and performance metrics
    """
    
    from .standard_one_player import standard_trackingXl5_one_player
    from .tracks_cleaning import clean_tracks
    from .tracking_video import make_tracking_video
    from ..init_dataframe.create_df import creatInitDataFrame
    from ..yoloV5.load_models import yoloV5l
    
    print("="*60)
    print("FAIR TRACKING APPROACHES COMPARISON")
    print("Both approaches track only ONE player (no ball detection)")
    print("="*60)
    
    # Load models once
    print("Loading YOLO models...")
    modelv5l, ball_modelv5l = yoloV5l()
    
    # ========================================
    # Standard Approach (One Player)
    # ========================================
    print("\n" + "="*50)
    print("RUNNING STANDARD TRACKING APPROACH (ONE PLAYER)")
    print("="*50)
    
    std_start_time = time.time()
    
    # Run standard tracking (one player only)
    std_iframes, std_itboxes, std_fps = standard_trackingXl5_one_player(
        modelv5l, ball_modelv5l, video_path, target_player_id, 
        experiment_name=f"{out_name}_standard"
    )
    
    std_total_time = time.time() - std_start_time
    
    # Clean and process standard results
    std_ziframes, std_zitboxes = clean_tracks(std_iframes, std_itboxes, ball_only)
    std_init_df = creatInitDataFrame(std_zitboxes, std_ziframes, teams_colors)
    
    # Save standard results
    std_init_df.to_csv(f'./Out/{out_name}_standard_one_player_init_df.csv', index=False)
    make_tracking_video(std_ziframes, std_zitboxes, f'./Out/{out_name}_standard_one_player_out.mp4', std_fps, draw=False)
    make_tracking_video(std_ziframes, std_zitboxes, f'./Out/{out_name}_standard_one_player_tracked.mp4', std_fps, draw=True)
    
    print(f"Standard approach (one player) completed in {std_total_time:.2f} seconds")
    print(f"Processed {len(std_iframes)} frames, {len(std_ziframes)} after cleaning")
    
    # ========================================
    # ROI Approach (One Player)
    # ========================================
    print("\n" + "="*40)
    print("RUNNING ROI TRACKING APPROACH (ONE PLAYER)")
    print("="*40)
    
    roi_start_time = time.time()
    
    # Run ROI tracking (one player only)
    roi_iframes, roi_itboxes, roi_fps = trackingXl5_ROI_single_player(
        modelv5l, ball_modelv5l, video_path, target_player_id, 
        save_roi_images=save_roi_images, roi_save_interval=roi_save_interval,
        experiment_name=f"{out_name}_roi"
    )
    
    roi_total_time = time.time() - roi_start_time
    
    # Clean and process ROI results
    roi_ziframes, roi_zitboxes = clean_tracks(roi_iframes, roi_itboxes, ball_only)
    roi_init_df = creatInitDataFrame(roi_zitboxes, roi_ziframes, teams_colors)
    
    # Save ROI results
    roi_init_df.to_csv(f'./Out/{out_name}_roi_one_player_init_df.csv', index=False)
    make_tracking_video(roi_ziframes, roi_zitboxes, f'./Out/{out_name}_roi_one_player_out.mp4', roi_fps, draw=False)
    make_tracking_video(roi_ziframes, roi_zitboxes, f'./Out/{out_name}_roi_one_player_tracked.mp4', roi_fps, draw=True)
    
    print(f"ROI approach (one player) completed in {roi_total_time:.2f} seconds")
    print(f"Processed {len(roi_iframes)} frames, {len(roi_ziframes)} after cleaning")
    
    # ========================================
    # FAIR Performance Analysis
    # ========================================
    print("\n" + "="*40)
    print("FAIR PERFORMANCE COMPARISON ANALYSIS")
    print("="*40)
    
    # Calculate performance metrics
    std_avg_per_frame = std_total_time / len(std_iframes) if len(std_iframes) > 0 else 0
    roi_avg_per_frame = roi_total_time / len(roi_iframes) if len(roi_iframes) > 0 else 0
    
    time_improvement = std_total_time - roi_total_time
    time_improvement_pct = (time_improvement / std_total_time) * 100 if std_total_time > 0 else 0
    speedup_factor = std_total_time / roi_total_time if roi_total_time > 0 else 0
    
    # Count ROI frames used (from CSV if available)
    roi_frames_used = 0
    try:
        csv_files = [f for f in os.listdir('./Out') if f.startswith('roi_detailed_frame_timing')]
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join('./Out', x)))
            import pandas as pd
            timing_df = pd.read_csv(f'./Out/{latest_csv}')
            roi_frames_used = timing_df['ROI_Active'].sum()
    except:
        pass
    
    # Create results dictionary
    results = {
        'standard_approach_one_player': {
            'total_time': std_total_time,
            'frames_processed': len(std_iframes),
            'clean_frames': len(std_ziframes),
            'avg_total_per_frame': std_avg_per_frame,
            'fps': std_fps
        },
        'roi_approach_one_player': {
            'total_time': roi_total_time,
            'frames_processed': len(roi_iframes),
            'clean_frames': len(roi_ziframes),
            'avg_total_per_frame': roi_avg_per_frame,
            'fps': roi_fps,
            'roi_frames_used': roi_frames_used
        },
        'improvements': {
            'time_saved': time_improvement,
            'total_time_improvement_pct': time_improvement_pct,
            'overall_speedup': speedup_factor,
            'avg_per_frame_improvement': std_avg_per_frame - roi_avg_per_frame
        }
    }
    
    # Print summary
    print(f"Standard Approach (One Player):")
    print(f"  - Total time: {std_total_time:.2f}s")
    print(f"  - Average per frame: {std_avg_per_frame:.6f}s")
    print(f"  - Frames processed: {len(std_iframes)}")
    print(f"  - Clean frames: {len(std_ziframes)}")
    
    print(f"\nROI Approach (One Player):")
    print(f"  - Total time: {roi_total_time:.2f}s")
    print(f"  - Average per frame: {roi_avg_per_frame:.6f}s")
    print(f"  - Frames processed: {len(roi_iframes)}")
    print(f"  - Clean frames: {len(roi_ziframes)}")
    print(f"  - ROI frames used: {roi_frames_used}/{len(roi_iframes)}")
    
    print(f"\nFair Performance Improvements:")
    print(f"  - Time saved: {time_improvement:.2f}s")
    print(f"  - Improvement: {time_improvement_pct:.2f}%")
    print(f"  - Overall speedup: {speedup_factor:.2f}x")
    
    # Save detailed report
    report_content = f"""
FAIR TRACKING APPROACHES COMPARISON REPORT
==========================================

Video: {video_path}
Target Player ID: {target_player_id}
Ball Only Mode: {ball_only}

COMPARISON SCOPE:
- Both approaches track ONLY ONE player (target_player_id={target_player_id})
- Ball detection and tracking EXCLUDED from both approaches
- Fair comparison focusing on detection and tracking optimization

STANDARD APPROACH RESULTS (ONE PLAYER):
---------------------------------------
Total Processing Time: {std_total_time:.2f} seconds
Average Time per Frame: {std_avg_per_frame:.6f} seconds
Frames Processed: {len(std_iframes)}
Clean Frames: {len(std_ziframes)}
FPS: {std_fps}
Detection Method: Full frame processing (1280x720)
Feature Extraction: Full frame

ROI APPROACH RESULTS (ONE PLAYER):
----------------------------------
Total Processing Time: {roi_total_time:.2f} seconds
Average Time per Frame: {roi_avg_per_frame:.6f} seconds
Frames Processed: {len(roi_iframes)}
Clean Frames: {len(roi_ziframes)}
ROI Frames Used: {roi_frames_used}/{len(roi_iframes)} ({(roi_frames_used/len(roi_iframes))*100:.1f}%)
FPS: {roi_fps}
Detection Method: ROI-based processing (adaptive region size)
Feature Extraction: ROI-optimized

FAIR PERFORMANCE IMPROVEMENTS:
------------------------------
Time Saved: {time_improvement:.2f} seconds
Performance Improvement: {time_improvement_pct:.2f}%
Overall Speedup: {speedup_factor:.2f}x
Per-frame Improvement: {std_avg_per_frame - roi_avg_per_frame:.6f} seconds

OPTIMIZATION ANALYSIS:
---------------------
The performance improvement is TRUE optimization comparison because:
1. Both approaches track the same number of objects (1 player)
2. Both approaches exclude ball detection 
3. ROI approach reduces computational load through:
   - Smaller detection area ({roi_frames_used}/{len(roi_iframes)} frames used ROI)
   - Optimized feature extraction on smaller regions
   - Targeted tracking updates

FILES GENERATED:
---------------
Standard Approach (One Player):
- {out_name}_standard_one_player_init_df.csv
- {out_name}_standard_one_player_out.mp4
- {out_name}_standard_one_player_tracked.mp4

ROI Approach (One Player):
- {out_name}_roi_one_player_init_df.csv
- {out_name}_roi_one_player_out.mp4
- {out_name}_roi_one_player_tracked.mp4

Analysis generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f'./Out/{out_name}_fair_tracking_comparison_report.txt', 'w') as f:
        f.write(report_content)
    
    print(f"\nFair comparison report saved to: ./Out/{out_name}_fair_tracking_comparison_report.txt")
    print("="*60)
    print("This is now a FAIR comparison - both approaches track only one player!")
    print("="*60)
    
    return results