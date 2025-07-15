from Distance_measurement.tennis_court_tracker import main_video_processing_pipeline
from Distance_measurement.tracker_based_on_first_frame import main_first_frame_processing_pipeline

from Distance_measurement.tracker_custom_frame import main_processing_pipeline
from Distance_measurement.tennis_court_tracker import *
'''
# --- PARAMETERS ---
video_path = "./test_videos/hurkach.mp4"
out_name = "hurkach"
init_df_path = f"./Out/{out_name}_init_df.csv"

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

# --- 2. Call the TennisCourtTracker pipeline ---


results = main_video_processing_pipeline(
    video_path=video_path,  
    data_csv_path=init_df_path
    )


'''
frame_path="./test_videos/himg.png"

frame = cv2.imread(frame_path)
tracker = TennisCourtTracker()
corners = tracker.detect_court_corners(frame)
tracker._draw_corners_on_frame(frame, corners)
cv2.imwrite("./Out/himg_corners.png", frame)


