from Distance_measurement.tennis_court_tracker import main_processing_pipeline

# --- PARAMETERS ---
video_path = "./test_videos/nadal-verdasco.mp4"
out_name = "nadal_verdasco"
init_df_path = f"./Out/{out_name}_init_df.csv"

# --- 1. Get initial data (creates a dataframe, usually saved as CSV) ---
# (Uncomment if you want to regenerate the initial dataframe)
# from Detect_and_Track.get_init_data import get_init_data
# teams_colors = ["red", "blue"]
# ball_only = False
# get_init_data(video_path, out_name, teams_colors, ball_only)

# --- 2. Call the TennisCourtTracker pipeline ---
# Use a frame from your video and the initial dataframe as input
frame_path = './test_videos/frame_for_detection5.png'  # or extract a frame from your video
main_processing_pipeline(frame_path, init_df_path)

