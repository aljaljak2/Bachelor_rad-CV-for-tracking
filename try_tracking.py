from Detect_and_Track.get_init_data import get_init_data
from Detect_and_Track.get_tracks import get_video_tracks
from Detect_and_Track.create_tracking_boxes_video import create_tracking_boxes_video
import pandas as pd
import cv2
from Detect_and_Track.court_detection import detect_court_corners_simple as detect_court_corners
from Distance_measurement.pixel_to_court_mapping import compute_homography, map_and_save_dataframe_pixels_to_court
from Distance_measurement.distance_calculation import calculate_player_distances, calculate_ball_distance
from Distance_measurement.court_corners_average import get_average_court_corners

# --- PARAMETERS ---
video_path = "./test_videos/nadal-verdasco.mp4"
out_name = "nadal_verdasco"
teams_colors = ["red", "blue"]  # or your actual team colors
ball_only = False
init_df_path = f"./Out/{out_name}_init_df.csv"
mapped_df_path = f"./Out/{out_name}_mapped_df.csv"

# --- 1. Get initial data (creates a dataframe, usually saved as CSV) ---
init_df = get_init_data(video_path, out_name, teams_colors, ball_only)

# --- 2. Get tracking video and bounding boxes (creates a video and returns frames, boxes) ---
tracked_frames, bboxes = get_video_tracks(video_path, out_name, ball_only=ball_only)

# --- 3. (Optional) Create a video with bounding boxes drawn ---
create_tracking_boxes_video(video_path, out_name)

# --- 4. Detect court corners in every 30th frame and use average ---
avg_pixel_corners = get_average_court_corners(tracked_frames, every_n=30, debug=False)
print("Average pixel corners:", avg_pixel_corners)

# --- 5. Define real-world court corners for doubles tennis (meters) ---
# Standard doubles tennis court: 23.77m x 10.97m (length x width)
court_length = 23.77
court_width = 10.97
# Order: top-left, top-right, bottom-right, bottom-left (match your pixel_corners order)
real_corners = [(0, 0), (court_width, 0), (court_width, court_length), (0, court_length)]

# --- 6. Compute homography and map dataframe ---
H = compute_homography(avg_pixel_corners, real_corners)
df = pd.read_csv(init_df_path)
df_mapped = map_and_save_dataframe_pixels_to_court(df, H, mapped_df_path)
print(f"Mapped dataframe saved to {mapped_df_path}")

# --- 7. Calculate distances ---
player_distances = calculate_player_distances(df_mapped, x_col='X_mapped', y_col='Y_mapped')
ball_distance = calculate_ball_distance(df_mapped, x_col='X_mapped', y_col='Y_mapped')
print("Player distances (meters):\n", player_distances)
print(f"Ball total distance (meters): {ball_distance:.2f}")

