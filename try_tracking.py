#from Distance_measurement.tennis_court_tracker import  *
from Distance_measurement.main_court_tracker import *

# --- PARAMETERS ---
video_path = "./test_videos/nadal-verdasco.mp4"
out_name = "nadal-verdasco"
init_df_path = f"./Out/{out_name}_init_df.csv"
'''
# --- 1. Get initial data (creates a dataframe, usually saved as CSV) ---
# (Uncomment if you want to regenerate the initial dataframe)
from Detect_and_Track.get_init_data import get_init_data
from Detect_and_Track.get_tracks import get_video_tracks
from Detect_and_Track.create_tracking_boxes_video import create_tracking_boxes_video
teams_colors = ["red", "blue"]
ball_only = False
get_init_data(video_path, out_name, teams_colors, ball_only)

get_video_tracks(video_path, out_name)
create_tracking_boxes_video(video_path, out_name)
'''
'''
# Standard pipeline (uses average corners)
results = main_video_processing_pipeline(
    video_path=video_path,  
    data_csv_path=init_df_path
)
'''


# Dynamic pipeline (updates homography per frame)
results = main_video_processing_pipeline_dynamic(
    video_path=video_path,      
    data_csv_path=init_df_path
)


'''
frame_path="./test_videos/sinner_dimitrov.png"

frame = cv2.imread(frame_path)
tracker = TennisCourtTracker()
corners = tracker.detect_court_corners(frame)
tracker._draw_corners_on_frame(frame, corners)
cv2.imwrite("./Out/SINNER.png", frame)
'''

