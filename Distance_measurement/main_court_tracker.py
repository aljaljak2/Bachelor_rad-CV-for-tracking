"""
Main Tennis Court Tracker Module - Refactored
"""
import pandas as pd
import os
import json
from .corner_detection import CornerDetector
from .coordinate_mapper import CoordinateMapper
from .player_distance_calculator import PlayerDistanceCalculator
from .ball_distance_calculator import BallDistanceCalculator
from .homography_manager import HomographyManager


class TennisCourtTracker:
    def __init__(self):
        # Initialize all modules
        self.corner_detector = CornerDetector()
        self.coordinate_mapper = CoordinateMapper()
        self.player_calculator = PlayerDistanceCalculator()
        self.ball_calculator = BallDistanceCalculator()

    # Corner Detection Methods (delegated to CornerDetector)
    def detect_corners_from_video(self, video_path, sample_interval=1.0, max_frames=30, debug=False):
        """Detect court corners from multiple frames in a video"""
        return self.corner_detector.detect_corners_from_video(video_path, sample_interval, max_frames, debug)

    def detect_court_corners(self, frame, debug=False):
        """Detect court corners from a single frame"""
        return self.corner_detector.detect_court_corners(frame, debug)

    def visualize_average_corners(self, frame, average_corners, output_path="average_corners_visualization.jpg", show_image=True):
        """Visualize average corners on a representative frame"""
        return self.corner_detector.visualize_average_corners(frame, average_corners, output_path, show_image)

    def draw_corners_on_frame(self, frame, corners):
        """Draw detected corners on frame"""
        return self.corner_detector.draw_corners_on_frame(frame, corners)

    def get_ordered_corners(self, corners):
        """Order corners in a consistent manner"""
        return self.corner_detector.get_ordered_corners(corners)

    # Coordinate Mapping Methods (delegated to CoordinateMapper)
    def compute_homography(self, pixel_corners):
        """Compute homography matrix to map pixel coordinates to court coordinates"""
        return self.coordinate_mapper.compute_homography(pixel_corners)

    def validate_homography(self, H, pixel_corners):
        """Validate the computed homography matrix"""
        return self.coordinate_mapper.validate_homography(H, pixel_corners)

    def map_pixels_to_court(self, df, H, x_col='X', y_col='Y'):
        """Map pixel coordinates to court coordinates using homography"""
        return self.coordinate_mapper.map_pixels_to_court(df, H, x_col, y_col)

    def debug_coordinates(self, df, x_col='X_court', y_col='Y_court', sample_size=10):
        """Debug coordinate mapping"""
        return self.coordinate_mapper.debug_coordinates(df, x_col, y_col, sample_size)

    # Player Distance Methods (delegated to PlayerDistanceCalculator)
    def calculate_player_distances_improved(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                          class_col='Class', frame_col='frame', fps=30.0):
        """Calculate improved player distances with temporal awareness"""
        return self.player_calculator.calculate_player_distances_improved(df, id_col, x_col, y_col, class_col, frame_col, fps)

    def analyze_player_movement_patterns(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                       class_col='Class', frame_col='frame'):
        """Analyze detailed movement patterns for each player"""
        return self.player_calculator.analyze_player_movement_patterns(df, id_col, x_col, y_col, class_col, frame_col)

    def validate_player_measurements(self, player_results, rally_duration_seconds=None):
        """Validate player measurements against realistic tennis movement patterns"""
        return self.player_calculator.validate_player_measurements(player_results, rally_duration_seconds)

    # Ball Distance Methods (delegated to BallDistanceCalculator)
    def calculate_ball_distance_improved(self, df, x_col='X_court', y_col='Y_court', 
                                        class_col='Class', frame_col='frame', fps=30.0):
        """Calculate improved ball distance with temporal awareness"""
        return self.ball_calculator.calculate_ball_distance_improved(df, x_col, y_col, class_col, frame_col, fps)

    def calculate_ball_distance_with_trajectory_estimation(self, df, x_col='X_court', y_col='Y_court',
                                                          class_col='Class', frame_col='frame', fps=30.0):
        """Calculate ball distance with parabolic trajectory estimation"""
        return self.ball_calculator.calculate_ball_distance_with_trajectory_estimation(df, x_col, y_col, class_col, frame_col, fps)

    def validate_ball_measurements(self, ball_results, rally_duration_seconds=None):
        """Validate ball measurements against realistic tennis expectations"""
        return self.ball_calculator.validate_ball_measurements(ball_results, rally_duration_seconds)

    def analyze_ball_trajectory(self, df, x_col='X_court', y_col='Y_court', 
                               class_col='Class', frame_col='frame'):
        """Analyze ball trajectory patterns"""
        return self.ball_calculator.analyze_ball_trajectory(df, x_col, y_col, class_col, frame_col)

    # Legacy method for backward compatibility
    def _draw_corners_on_frame(self, frame, corners):
        """Legacy method - redirects to draw_corners_on_frame"""
        return self.draw_corners_on_frame(frame, corners)


def main_video_processing_pipeline(video_path, data_csv_path, sample_interval=1.0, max_frames=30):
    """
    Main processing pipeline for video-based corner detection
    
    Args:
        video_path: Path to video file or URL
        data_csv_path: Path to CSV file with tracking data
        sample_interval: Interval in seconds between frame samples
        max_frames: Maximum number of frames to process
        
    Returns:
        Tuple of (df_mapped, player_distances, ball_results, average_corners, frame_info)
    """
    tracker = TennisCourtTracker()
    
    print("=== STEP 1: Detecting Court Corners from Video ===")
    try:
        average_corners, all_frame_corners, frame_info, representative_frame = tracker.detect_corners_from_video(
            video_path, sample_interval=sample_interval, max_frames=max_frames, debug=True
        )
        
        if average_corners is None:
            print("Error: Could not detect corners from video")
            return None, None, None, None, None
        
        print(f"Successfully calculated average corners from {len(all_frame_corners)} frames")
        
        # Print summary statistics
        valid_frames = sum(1 for info in frame_info if info['valid'])
        print(f"Frame processing summary: {valid_frames}/{len(frame_info)} frames had valid corners")
        
        # Optional: Visualize the average corners on the representative frame
        if representative_frame is not None:
            print("=== STEP 1.5: Visualizing Average Corners ===")
            # Extract directory and filename from data_csv_path to match the Out folder structure
            csv_dir = os.path.dirname(data_csv_path)
            csv_filename = os.path.basename(data_csv_path)
            image_filename = csv_filename.replace('.csv', '_average_corners.jpg')
            image_output_path = os.path.join(csv_dir, image_filename)
            
            tracker.visualize_average_corners(
                representative_frame, 
                average_corners, 
                output_path=image_output_path,
                show_image=False
            )
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None, None, None, None
    
    print("=== STEP 2: Computing Homography with Average Corners ===")
    try:
        H = tracker.compute_homography(average_corners)
        print("Homography matrix computed successfully using average corners")
        is_valid = tracker.validate_homography(H, average_corners)
        if not is_valid:
            print("WARNING: Homography validation failed")
    except Exception as e:
        print(f"Error computing homography: {e}")
        return None, None, None, None, None
    
    print("=== STEP 3: Loading and Processing Tracking Data ===")
    df = pd.read_csv(data_csv_path)
    print(f"Loaded {len(df)} tracking points")
    df_mapped = tracker.map_pixels_to_court(df, H)
    tracker.debug_coordinates(df_mapped)
    
    print("=== STEP 4: Calculating Distances ===")
    player_distances = tracker.calculate_player_distances_improved(df_mapped)
    print("Player distances:")
    print(player_distances)
    
    ball_results = tracker.calculate_ball_distance_improved(df_mapped, fps=30.0)
    tracker.validate_ball_measurements(ball_results)
    
    # Save results
    output_path = data_csv_path.replace('.csv', '_with_video_court_coords.csv')
    df_mapped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_path}")
    
    # Save corner analysis results
    corner_analysis_path = data_csv_path.replace('.csv', '_corner_analysis.csv')
    corner_df = pd.DataFrame(frame_info)
    corner_df.to_csv(corner_analysis_path, index=False, encoding='utf-8')
    print(f"Corner analysis saved to: {corner_analysis_path}")
    
    return df_mapped, player_distances, ball_results, average_corners, frame_info


def main_video_processing_pipeline_dynamic(video_path, data_csv_path, sample_interval=1.0, max_frames=30):
    """
    Main processing pipeline with dynamic homography updates
    
    Args:
        video_path: Path to video file or URL
        data_csv_path: Path to CSV file with tracking data
        sample_interval: Interval in seconds between frame samples
        max_frames: Maximum number of frames to process
        
    Returns:
        Tuple of (df_mapped, player_distances, ball_results, average_corners, frame_info)
    """
    tracker = TennisCourtTracker()

    print("=== STEP 1: Processing Video with Dynamic Homography ===")
    try:
        # Get frame analysis (keep existing corner detection)
        average_corners, all_frame_corners, frame_info, representative_frame = tracker.detect_corners_from_video(
            video_path, sample_interval=sample_interval, max_frames=max_frames, debug=True
        )

        if average_corners is None:
            print("Error: Could not detect corners from video")
            return None, None, None, None, None

    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None, None, None, None

    print("=== STEP 2: Setting up Dynamic Homography Processing ===")
    homography_manager = HomographyManager(tracker)

    # Initialize with first valid frame
    initial_homography = None
    for info in frame_info:
        if info['valid']:
            initial_homography = homography_manager.update_homography_if_needed(
                info['corners'], info['frame_number']
            )
            break

    if initial_homography is None:
        print("Error: Could not initialize homography")
        return None, None, None, None, None

    print("=== STEP 3: Loading and Processing Tracking Data with Dynamic Homography ===")
    df = pd.read_csv(data_csv_path)
    print(f"Loaded {len(df)} tracking points")

    # Process data with dynamic homography updates
    df_mapped = df.copy()
    df_mapped['X_court'] = 0.0  # Initialize court coordinates
    df_mapped['Y_court'] = 0.0
    current_homography = initial_homography

    # Group by frame and apply appropriate homography
    unique_frames = sorted(df['frame'].unique())
    homography_updates = 0
    
    for frame_num in unique_frames:
        frame_data = df[df['frame'] == frame_num]

        # Check if we have corner data for this frame
        frame_corners = None
        for info in frame_info:
            if info['frame_number'] == frame_num and info['valid']:
                frame_corners = info['corners']
                break

        # Update homography if needed
        if frame_corners is not None:
            updated_homography = homography_manager.update_homography_if_needed(
                frame_corners, frame_num
            )
            if updated_homography is not current_homography:
                current_homography = updated_homography
                homography_updates += 1

        # Apply current homography to this frame's data
        if current_homography is not None:
            frame_mapped = tracker.map_pixels_to_court(frame_data, current_homography)
            df_mapped.loc[df['frame'] == frame_num, ['X_court', 'Y_court']] = frame_mapped[['X_court', 'Y_court']]

    print(f"[DEBUG] Applied {homography_updates} homography updates across {len(unique_frames)} frames")
    tracker.debug_coordinates(df_mapped)

    print("=== STEP 4: Calculating Distances with Dynamic Mapping ===")
    player_distances = tracker.calculate_player_distances_improved(df_mapped)
    print("Player distances:")
    print(player_distances)
    
    ball_results = tracker.calculate_ball_distance_improved(df_mapped, fps=30.0)
    tracker.validate_ball_measurements(ball_results)
    
    # Save results
    output_path = data_csv_path.replace('.csv', '_with_dynamic_court_coords.csv')
    df_mapped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_path}")
    
    # Save corner analysis results
    corner_analysis_path = data_csv_path.replace('.csv', '_corner_analysis.csv')
    corner_df = pd.DataFrame(frame_info)
    corner_df.to_csv(corner_analysis_path, index=False, encoding='utf-8')
    print(f"Corner analysis saved to: {corner_analysis_path}")
    
    # Save player movement analysis
    player_analysis_path = data_csv_path.replace('.csv', '_player_movement_analysis.csv')
    player_distances.to_csv(player_analysis_path, index=False, encoding='utf-8')
    print(f"Player movement analysis saved to: {player_analysis_path}")
    
    # Save ball movement analysis
    ball_analysis_path = data_csv_path.replace('.csv', '_ball_movement_analysis.csv')
    ball_df = pd.DataFrame([ball_results])  # Convert dict to DataFrame
    ball_df.to_csv(ball_analysis_path, index=False, encoding='utf-8')
    print(f"Ball movement analysis saved to: {ball_analysis_path}")
    
    # Save detailed movement patterns (optional)
    movement_patterns = tracker.analyze_player_movement_patterns(df_mapped)
    if movement_patterns:
        patterns_path = data_csv_path.replace('.csv', '_movement_patterns.json')
        with open(patterns_path, 'w') as f:
            json.dump(movement_patterns, f, indent=2)
        print(f"Movement patterns saved to: {patterns_path}")
    
    # Save homography statistics
    homography_stats = homography_manager.get_homography_stats()
    homography_stats_path = data_csv_path.replace('.csv', '_homography_stats.json')
    with open(homography_stats_path, 'w') as f:
        # Convert numpy arrays and types to JSON serializable format
        stats_serializable = {}
        for key, value in homography_stats.items():
            if key == 'current_corners' and value is not None:
                # Convert corners to regular Python lists/floats
                stats_serializable[key] = [[float(corner[0]), float(corner[1])] for corner in value]
            elif key == 'update_frames':
                # Convert numpy integers to regular Python integers
                stats_serializable[key] = [int(frame) for frame in value]
            elif key == 'total_updates':
                # Convert numpy integer to regular Python integer
                stats_serializable[key] = int(value)
            elif key == 'threshold':
                # Convert to regular Python float
                stats_serializable[key] = float(value)
            else:
                stats_serializable[key] = value
        json.dump(stats_serializable, f, indent=2)
    print(f"Homography statistics saved to: {homography_stats_path}")
    
    return df_mapped, player_distances, ball_results, average_corners, frame_info