# standard_one_player.py - Standard tracking implementation for single player

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

def standard_trackingXl5_one_player(Yolo_model, ball_model, video_path, target_player_id=1,
                                   experiment_name="standard_one_player"):
    '''
    Standard tracking approach that tracks only one specific player (no ball tracking).
    This is for fair comparison with ROI approach.
    Processes full frames but only tracks the target player.

    Parameters
    ----------
    Yolo_model : pytorch model
        pytorch YoloV5l model.
    ball_model : pytorch model
        pytorch YoloV5l model to detect the ball specifically (not used but kept for compatibility).
    video_path : string
        the path of the directory of the processed video.
    target_player_id : int
        ID of the player to track (default: 1)
    experiment_name : str
        Name for the experiment (used in timing file naming)

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

    # Standard tracking variables
    target_track = None
    first_frame_processed = False

    # Initialize timing variables
    frame_count = 0
    timing_data = []

    # Ensure output directory exists
    os.makedirs('./Out', exist_ok=True)

    # Create CSV filename
    csv_filename = f'./Out/standard_one_player_detailed_frame_timing_{int(time.time())}.csv'

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

        # Detection timing - ALWAYS process full frame
        detection_start = time.time()
        
        results = Yolo_model(original_frame)
        pred_bbox = results.xyxy[0].tolist()
        bboxes = [np.array(box) for box in pred_bbox]

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

        # ALWAYS use full frame for feature extraction (standard approach)
        features = np.array(encoder(original_frame, boxes))

        # Filter out ball detections (same as ROI approach for fair comparison)
        non_ball_detections_indices = [i for i, name in enumerate(names) if name != 'sports ball']
        filtered_boxes = [boxes[i] for i in non_ball_detections_indices]
        filtered_scores = [scores[i] for i in non_ball_detections_indices]
        filtered_names = [names[i] for i in non_ball_detections_indices]
        filtered_features = [features[i] for i in non_ball_detections_indices]

        # Create detections (excluding ball)
        detections = [Detection(bbox, score, class_name, feature)
                     for bbox, score, class_name, feature in zip(filtered_boxes, filtered_scores, filtered_names, filtered_features)]

        # Standard tracking prediction and update - process ALL detections
        tracker.predict()
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
                first_frame_processed = True
                current_target_found = True
                print(f"Frame {frame_count} - Target player {target_player_id} locked for standard tracking")
            elif target_track is not None and track == target_track:
                current_target_found = True

            # Only track the target player (same as ROI for fair comparison)
            if target_track is not None and track == target_track:
                tracked_bboxes.append(bbox.tolist() + [tracking_id, index])

        # If target is lost, reset target tracking
        if target_track is not None and not current_target_found:
            print(f"Frame {frame_count} - Target player lost in standard tracking")
            target_track = None
            first_frame_processed = False

        # Ball detection is EXCLUDED (same as ROI approach)
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

        # Store timing data
        timing_data.append([frame_count, frame_start_time, frame_end_time,
                           detection_time, tracking_time, complete_detection_tracking_time,
                           total_processing_time])

    # Write timing data to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame_Number', 'Start_Time (unix_timestamp)', 'End_Time (unix_timestamp)',
                        'Detection_Time (s)', 'Tracking_Time (s)', 'Complete_Detection_Tracking_Time (s)',
                        'Total_Processing_Time (s)'])
        writer.writerows(timing_data)

    print(f"\nDetailed timing data saved to: {csv_filename}")
    print(f'Tracked {len(frames)} frames')

    return frames, tboxes, fps