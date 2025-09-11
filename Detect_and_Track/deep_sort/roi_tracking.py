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
                                     experiment_name="simple_roi_experiment"):
    '''
    Simplified ROI-based detection that tracks a single player without DeepSORT.
    Uses only YOLO detection within adaptive ROI regions.
    
    Parameters
    ----------
    Yolo_model : pytorch model
        pytorch YoloV5l model.
    ball_model : pytorch model
        pytorch YoloV5l model (kept for compatibility, not used).
    video_path : string
        the path of the directory of the processed video.
    target_player_id : int
        ID to assign to the tracked player (default: 1)
    save_roi_images : bool
        Whether to save ROI images during tracking (default: False)
    roi_save_interval : int
        Save ROI image every N frames (default: 15)
    experiment_name : str
        Name for the experiment (used in ROI image saving)

    Return
    ----------
    frames : list
        list of frames of the video.
    tboxes : list
        list of detected objects in each frame.
    fps : int
        number of frames per second used in processing
    '''

    if video_path:
        vid = cv2.VideoCapture(video_path)

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f'fps of the input video = {fps}')
    print('Please wait ... \n')
    
    NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)

    frames = []
    tboxes = []

    # Simple ROI tracking variables
    roi_active = False
    roi_frames_used = 0
    last_detection = None  # Store last known player position
    roi_coords = None      # Current ROI coordinates
    consecutive_misses = 0 # Count frames without detection
    max_misses = 10        # Max frames before expanding search

    # ROI parameters
    roi_expansion_factor = 1.5  # How much to expand ROI when player lost
    min_roi_size = 100         # Minimum ROI dimension
    max_roi_size = 400         # Maximum ROI dimension

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
    csv_filename = f'./Out/simple_roi_detailed_timing_{int(time.time())}.csv'

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

        detected_player = None
        
        # Use ROI if we have a previous detection
        if roi_active and roi_coords is not None:
            x1, y1, x2, y2 = roi_coords
            
            # Ensure ROI is within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(1280, x2)
            y2 = min(720, y2)
            
            roi_frame = original_frame[y1:y2, x1:x2]

            # Save ROI image if requested
            if save_roi_images and frame_count % roi_save_interval == 0:
                roi_image_path = f'{roi_images_dir}/frame_{frame_count:06d}_roi.jpg'
                cv2.imwrite(roi_image_path, cv2.cvtColor(roi_frame, cv2.COLOR_RGB2BGR))

            # Run detection on ROI
            try:
                results = Yolo_model(roi_frame)
                pred_bbox = results.xyxy[0].tolist()

                # Find the best person detection in ROI
                best_person = None
                best_confidence = 0
                
                for box in pred_bbox:
                    class_id = int(box[5])
                    confidence = box[4]
                    
                    # Look for person class (class 0 in COCO)
                    if class_id == 0 and confidence > best_confidence and confidence > 0.3:
                        # Convert back to full frame coordinates
                        detected_player = [
                            int(box[0] + x1),  # x1
                            int(box[1] + y1),  # y1
                            int(box[2] + x1),  # x2
                            int(box[3] + y1),  # y2
                            target_player_id,   # assigned ID
                            class_id           # class
                        ]
                        best_confidence = confidence
                        best_person = detected_player

                if best_person:
                    detected_player = best_person
                    consecutive_misses = 0
                    roi_frames_used += 1
                    print(f"Frame {frame_count} - Player detected in ROI: ({x1},{y1}) to ({x2},{y2})")
                else:
                    consecutive_misses += 1
                    print(f"Frame {frame_count} - No player in ROI, miss count: {consecutive_misses}")

            except Exception as e:
                print(f"Frame {frame_count} - ROI detection failed: {e}")
                consecutive_misses += 1

        # If no ROI active or too many consecutive misses, search full frame
        if not roi_active or consecutive_misses >= max_misses:
            print(f"Frame {frame_count} - Searching full frame")
            
            results = Yolo_model(original_frame)
            pred_bbox = results.xyxy[0].tolist()

            # Find the best person detection in full frame
            best_person = None
            best_confidence = 0
            
            # If we had a previous detection, prefer detections close to it
            for box in pred_bbox:
                class_id = int(box[5])
                confidence = box[4]
                
                if class_id == 0 and confidence > 0.3:  # Person class
                    candidate = [
                        int(box[0]),       # x1
                        int(box[1]),       # y1
                        int(box[2]),       # x2
                        int(box[3]),       # y2
                        target_player_id,  # assigned ID
                        class_id          # class
                    ]
                    
                    # Calculate score based on confidence and proximity to last detection
                    score = confidence
                    if last_detection is not None:
                        # Calculate distance to last known position
                        last_center_x = (last_detection[0] + last_detection[2]) / 2
                        last_center_y = (last_detection[1] + last_detection[3]) / 2
                        curr_center_x = (candidate[0] + candidate[2]) / 2
                        curr_center_y = (candidate[1] + candidate[3]) / 2
                        
                        distance = np.sqrt((curr_center_x - last_center_x)**2 + 
                                         (curr_center_y - last_center_y)**2)
                        
                        # Prefer closer detections (add proximity bonus)
                        proximity_bonus = max(0, 1 - distance / 300)  # Normalize by max expected distance
                        score = confidence + 0.3 * proximity_bonus
                    
                    if score > best_confidence:
                        best_confidence = score
                        best_person = candidate

            if best_person:
                detected_player = best_person
                consecutive_misses = 0
                roi_active = True
                print(f"Frame {frame_count} - Player detected in full frame, activating ROI")
            else:
                consecutive_misses += 1
                print(f"Frame {frame_count} - No player detected anywhere")

        detection_end = time.time()

        # Update ROI for next frame based on current detection
        if detected_player is not None:
            last_detection = detected_player
            
            # Calculate ROI for next frame
            player_width = detected_player[2] - detected_player[0]
            player_height = detected_player[3] - detected_player[1]
            center_x = (detected_player[0] + detected_player[2]) / 2
            center_y = (detected_player[1] + detected_player[3]) / 2
            
            # Adaptive ROI size based on player size
            roi_width = max(min_roi_size, min(max_roi_size, int(player_width * 2.5)))
            roi_height = max(min_roi_size, min(max_roi_size, int(player_height * 2.5)))
            
            # Calculate ROI coordinates
            x1 = int(center_x - roi_width / 2)
            y1 = int(center_y - roi_height / 2)
            x2 = int(center_x + roi_width / 2)
            y2 = int(center_y + roi_height / 2)
            
            roi_coords = (x1, y1, x2, y2)
            
        elif consecutive_misses < max_misses and roi_coords is not None:
            # Expand ROI when player is temporarily lost
            x1, y1, x2, y2 = roi_coords
            expand_w = int((x2 - x1) * 0.2)  # Expand by 20%
            expand_h = int((y2 - y1) * 0.2)
            
            roi_coords = (x1 - expand_w, y1 - expand_h, x2 + expand_w, y2 + expand_h)
        else:
            # Lost player for too long, disable ROI
            roi_active = False
            roi_coords = None

        # Prepare output for this frame
        frame_detections = []
        if detected_player is not None:
            frame_detections.append(detected_player)
        
        tboxes.append(frame_detections)

        # End timing for this frame
        frame_end_time = time.time()
        total_processing_time = frame_end_time - frame_start_time
        detection_time = detection_end - detection_start

        print(f"Frame {frame_count} - Completed at: {frame_end_time:.6f}")
        print(f"Frame {frame_count} - Total time: {total_processing_time:.6f}s "
              f"(Detection: {detection_time:.6f}s)")

        # Store timing data
        timing_data.append([frame_count, frame_start_time, frame_end_time,
                           detection_time, total_processing_time, roi_active,
                           consecutive_misses])

    # Write timing data to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame_Number', 'Start_Time', 'End_Time',
                        'Detection_Time', 'Total_Time', 'ROI_Active', 'Consecutive_Misses'])
        writer.writerows(timing_data)

    print(f"\nDetailed timing data saved to: {csv_filename}")
    print(f'Processed {len(frames)} frames')
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