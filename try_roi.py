from Detect_and_Track.roi_main import *

video_path="./test_videos/melbourne2.mp4"
experiment_name="tennis_test"


success = run_fair_roi_experiment(
    video_path="./test_videos/melbourne2.mp4",
    experiment_name="tennis_test",
    teams_colors=['white', 'white', 'red', 'red', 'black', 'yellow'],
    target_player_id=2,
    ball_only=False,
    save_roi_images=True,      # Enable ROI image saving
    roi_save_interval=15         # Save every 30 frames
)

'''
success1 = run_roi_only_experiment_fair(video_path, experiment_name, teams_colors=None, 
                           target_player_id=1, ball_only=False, save_roi_images=True, 
                           roi_save_interval=15)
'''
'''
from Detect_and_Track.roi_main import quick_fair_roi_test

# Fer poređenje
success = quick_fair_roi_test("./test_videos/melbourne2.mp4", "melbourne_fair")
'''