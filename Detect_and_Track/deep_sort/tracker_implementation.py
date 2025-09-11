import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import time
import csv
from .utils import read_class_names

from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from .tracker import Tracker
from .generate_detections import create_box_encoder
# import nn_matching
# import generate_detections as gdet

YOLO_COCO_CLASSES = "./Detect_and_Track/model_data/coco/coco.names"

def trackingXl5(Yolo_model, ball_model, video_path):
    '''
    this functions is to track objects in a video by:
    1 - reading the video frames
    2 - detecting the objects in the frame
    3 - tracking objects frame by frame by the ID of each object.
    
    Parameters
    ----------
    Yolo_model : pytorch model
        pytorch YoloV5l model.  
    ball_model : pytorch model
        pytorch YoloV5l model to detect the ball specifically.  
    video_path : string
        the path of the directory of the processed video.
     
    Return 
    ----------
    frames : list
        list of frames of the video with objects tracked. 
    tboxes :list
        list of every object tracked in every frame.      
         object:[y1, x1, y2, x2, class of the object, id of the object]
         where y1, x1, y2, x2 are the coordinates of the box around the object
    fps: int
        numder of frames per second used in processing
    '''

    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = './Detect_and_Track/model_data/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
   
    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f'fps of the input video = {fps}')
    print('Please wait ... \n')
    NUM_CLASS = read_class_names(YOLO_COCO_CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
 
    frames = []
    tboxes = []
    
    # Initialize timing variables
    frame_count = 0
    timing_data = []
    
    # Ensure output directory exists
    os.makedirs('./Out', exist_ok=True)
    
    # Create CSV filename
    csv_filename = f'./Out/detailed_frame_timing_{int(time.time())}.csv'
    
    while True:
        # Start timing for this frame
        frame_start_time = time.time()
        
        _, frame = vid.read()
        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            source_frame = original_frame
            original_frame = cv2.resize(original_frame, (1280,720)) 
            frames.append(original_frame) 
        except:
            break
        
        frame_count += 1
        print(f"Frame {frame_count} - Processing started at: {frame_start_time:.6f}")
        
        # Detection timing
        detection_start = time.time()
        # Determine the smaller dimension of the current frame
        frame_height, frame_width = source_frame.shape[:2]
        imgsz = min(frame_width, frame_height)
        # Pass imgsz/size to YOLO model if supported
        try:
            results = Yolo_model(original_frame, size=imgsz)
        except TypeError:
            # Fallback for models that do not accept 'size' argument
            for i in range(3):
                print("MODEL DO NOT ACCEPT THIS PARAMETER FOR SIZE\n")
            results = Yolo_model(original_frame)
        pred_bbox = results.xyxy[0].tolist()
        bboxes = [np.array(box) for box in pred_bbox] 
        
        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes: 
            boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
            scores.append(bbox[4])
            names.append(NUM_CLASS[int(bbox[5])])
        
        detection_end = time.time()
        
        # Tracking timing
        tracking_start = time.time()
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
        
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 

            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name
            # give id 0 to the ball
            if index == 32:
                tracked_bboxes.append(bbox.tolist() + [0, index]) # Structure data, that we could use it with our draw_bbox function
            else:
                tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        #detect the ball only
        if 32 not in [b[-1] for b in tracked_bboxes]:  
            ball_results = ball_model(original_frame).xyxy[0].tolist()           
            if len(ball_results) > 0:
                ball_pred_bbox = ball_results[0]
                ball = [round(ball_pred_bbox[0]), round(ball_pred_bbox[1]), round(ball_pred_bbox[2]), round(ball_pred_bbox[3]), 0, 32]
                tracked_bboxes.append(ball)
                
        tboxes.append([[round(bb) for bb in tracked_bbox] for tracked_bbox in tracked_bboxes])
        
        tracking_end = time.time()
        
        # End timing for this frame
        frame_end_time = time.time()
        total_processing_time = frame_end_time - frame_start_time
        detection_time = detection_end - detection_start
        tracking_time = tracking_end - tracking_start
        complete_detection_tracking_time = detection_time + tracking_time 
        
        print(f"Frame {frame_count} - Completed at: {frame_end_time:.6f}")
        print(f"Frame {frame_count} - Total time: {total_processing_time:.6f}s (Detection: {detection_time:.6f}s, Tracking: {tracking_time:.6f}s, Detection+Tracking: {complete_detection_tracking_time:.6f}s)")

        
        # Store timing data
        timing_data.append([frame_count, frame_start_time, frame_end_time, detection_time, tracking_time, complete_detection_tracking_time, total_processing_time])

    
    # Write timing data to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame_Number', 'Start_Time (unix_timestamp)', 'End_Time (unix_timestamp)', 'Detection_Time (s)', 'Tracking_Time (s)', 'Complete_Detection_Tracking_Time (s)', 'Total_Processing_Time (s)'])
        writer.writerows(timing_data)
    
    print(f"\nDetailed timing data saved to: {csv_filename}")
    print(f'tracked {len(frames)} frames')

    return frames, tboxes, fps