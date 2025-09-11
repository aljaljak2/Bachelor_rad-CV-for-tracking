import pandas as pd
from .init_dataframe.create_df import creatInitDataFrame
from .get_tracks import get_video_tracks
from .deep_sort.tracker_implementation import trackingXl5
from .deep_sort.tracks_cleaning import clean_tracks
from .deep_sort.tracking_video import make_tracking_video
from .yoloV5.load_models import yoloV5l


def create_tracking_video_and_init_data(video_path, out_name, teams_colors, ball_only=True):
    '''
    Combined function to process video once and create both tracking video and init CSV.
    
    Parameters
    ----------
    video_path : string
        the path of the directory of the processed video.
    out_name : string
        the name to save outputs with.
    teams_colors : list of strings
        list of the colors to classify the teams.
    ball_only : boolean
        whether or not to save only the frames with the ball detected in them.
        
    Returns
    ----------
    init_df : pandas dataframe
        the initial unmapped dataframe.
    ziframes : list
        list of cleaned frames.
    zitboxes : list
        list of cleaned tracking boxes.
    '''
    
    from Detect_and_Track.deep_sort.tracker_implementation import trackingXl5
    from Detect_and_Track.deep_sort.tracks_cleaning import clean_tracks
    from Detect_and_Track.deep_sort.tracking_video import make_tracking_video
    from Detect_and_Track.yoloV5.load_models import yoloV5l
    from Detect_and_Track.init_dataframe.create_df import creatInitDataFrame
    import os
    
    # Ensure output directory exists
    os.makedirs('./Out', exist_ok=True)
    
    import cv2
    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid.release()
    imgsz = min(width, height)
    print("Loading YOLO models...")
    modelv5l, ball_modelv5l = yoloV5l(imgsz=imgsz)
    print("IMGSZ:", modelv5l.imgsz)
    
    print("Processing video with tracking and timing...")
    # This is where all the timing measurements happen
    iframes, itboxes, fps = trackingXl5(modelv5l, ball_modelv5l, video_path)
    
    print("Cleaning tracks...")
    ziframes, zitboxes = clean_tracks(iframes, itboxes, ball_only)
    
    print("Creating initial dataframe...")
    # Create the init CSV
    init_df = creatInitDataFrame(zitboxes, ziframes, teams_colors)
    init_df.to_csv(f'./Out/{out_name}_init_df.csv', index=False)
    print(f"Init dataframe saved to: ./Out/{out_name}_init_df.csv")
    
    print("Creating tracking videos...")
    # Create clean video (without bounding boxes)
    make_tracking_video(ziframes, zitboxes, f'./Out/{out_name}_out.mp4', fps, draw=False)
    print(f"Clean video saved to: ./Out/{out_name}_out.mp4")
    
    # Create tracking video (with bounding boxes)
    make_tracking_video(ziframes, zitboxes, f'./Out/{out_name}_out_tracked.mp4', fps, draw=True)
    print(f"Tracking video saved to: ./Out/{out_name}_out_tracked.mp4")
    
    print("All outputs created successfully!")
    return init_df, ziframes, zitboxes