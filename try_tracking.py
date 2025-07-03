from Detect_and_Track.get_init_data import get_init_data
from Detect_and_Track.get_tracks import get_video_tracks
from Detect_and_Track.create_tracking_boxes_video import create_tracking_boxes_video
import pandas as pd

# 1. Detect and track players/ball, save video and initial dataframe
video_path = "./test_videos/rafa-dimitar.mp4"
out_name = "psv-arsenal"
teams_colors = ["red", "blue"]  # or your actual team colors
ball_only = False

# Step 1: Get initial data (creates a dataframe, usually saved as CSV)
get_init_data(video_path, out_name, teams_colors, ball_only)

# Step 2: Get tracking video and bounding boxes (creates a video and returns frames, boxes)
tracked_frames, bboxes = get_video_tracks(video_path, out_name, ball_only=ball_only)

# Step 3: (Optional) Create a video with bounding boxes drawn
create_tracking_boxes_video(video_path, out_name)

