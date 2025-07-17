"""
Enhanced Tennis Court Tracker with Improved Coordinate Mapping - FIXED VERSION
Copy this entire code to replace your enhanced_tennis_court_tracker.py
"""
import pandas as pd
import os
import json
import numpy as np

# You'll need to make sure these imports match your actual file structure
try:
    from .corner_detection import CornerDetector
    from .improved_coordinate_mapper import ImprovedCoordinateMapper
    from .player_distance_calculator import PlayerDistanceCalculator
    from .ball_distance_calculator import BallDistanceCalculator
    from .homography_manager import HomographyManager
except ImportError:
    # If relative imports fail, try absolute imports
    from corner_detection import CornerDetector
    from improved_coordinate_mapper import ImprovedCoordinateMapper
    from player_distance_calculator import PlayerDistanceCalculator
    from ball_distance_calculator import BallDistanceCalculator
    from homography_manager import HomographyManager


class EnhancedTennisCourtTracker:
    def __init__(self):
        # Initialize all modules with improved coordinate mapper
        self.corner_detector = CornerDetector()
        self.coordinate_mapper = ImprovedCoordinateMapper()
        self.player_calculator = PlayerDistanceCalculator()
        self.ball_calculator = BallDistanceCalculator()

    # Corner Detection Methods
    def detect_corners_from_video(self, video_path, sample_interval=1.0, max_frames=30, debug=False):
        return self.corner_detector.detect_corners_from_video(video_path, sample_interval, max_frames, debug)

    def detect_court_corners(self, frame, debug=False):
        return self.corner_detector.detect_court_corners(frame, debug)

    def visualize_average_corners(self, frame, average_corners, output_path="average_corners_visualization.jpg", show_image=True):
        return self.corner_detector.visualize_average_corners(frame, average_corners, output_path, show_image)

    # Enhanced Coordinate Mapping Methods
    def compute_homography(self, pixel_corners):
        try:
            H, ordered_corners = self.coordinate_mapper.compute_homography(pixel_corners)
            return H, ordered_corners
        except Exception as e:
            print(f"Enhanced homography computation failed: {e}")
            # Fallback to basic computation if needed
            try:
                H = self.coordinate_mapper.compute_homography_basic(pixel_corners)
                return H, None
            except:
                # If that also fails, use the original method
                return self.coordinate_mapper.compute_homography(pixel_corners)

    def validate_homography(self, H, pixel_corners):
        return self.coordinate_mapper.validate_homography(H, pixel_corners)

    def map_pixels_to_court(self, df, H, x_col='X', y_col='Y'):
        # Use the improved mapping method
        df_mapped = self.coordinate_mapper.map_pixels_to_court(df, H, x_col, y_col)
        
        # Apply court constraints and add semantic information
        df_constrained = self.coordinate_mapper.apply_court_constraints(df_mapped)
        df_with_zones = self.coordinate_mapper.add_court_zones(df_constrained)
        df_final = self.coordinate_mapper.get_distance_to_lines(df_with_zones)
        
        return df_final

    def debug_coordinates(self, df, x_col='X_court', y_col='Y_court', sample_size=10):
        return self.coordinate_mapper.debug_coordinates(df, sample_size)

    def diagnose_mapping_quality(self, df, pixel_corners):
        print("\n=== MAPPING QUALITY DIAGNOSIS ===")
        
        # Check coordinate ranges
        x_coords = df['X_court'].dropna()
        y_coords = df['Y_court'].dropna()
        
        issues = []
        
        # Range checks
        if x_coords.max() - x_coords.min() > 50:
            issues.append("X coordinate range too large - possible homography error")
        if y_coords.max() - y_coords.min() > 50:
            issues.append("Y coordinate range too large - possible homography error")
        
        # Boundary checks
        court_width = self.coordinate_mapper.court_width
        court_length = self.coordinate_mapper.court_length
        
        x_outliers = ((x_coords < -5) | (x_coords > court_width + 5)).sum()
        y_outliers = ((y_coords < -5) | (y_coords > court_length + 5)).sum()
        
        outlier_percentage = (x_outliers + y_outliers) / (len(x_coords) + len(y_coords)) * 100
        
        if outlier_percentage > 20:
            issues.append(f"High outlier percentage ({outlier_percentage:.1f}%) - check corner detection")
        
        if issues:
            print("[WARNING] Issues detected:")
            for issue in issues:
                print(f"  - {issue}")
            
            print("\n[SUGGESTION] Suggestions:")
            print("  1. Verify corner detection accuracy")
            print("  2. Check corner ordering (should be: BL, BR, TR, TL)")
            print("  3. Consider using frame-by-frame corner detection")
            print("  4. Apply coordinate constraints to filter outliers")
        else:
            print("[SUCCESS] Mapping quality looks good!")
        
        return issues

    # Player Distance Methods
    def calculate_player_distances_improved(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                          class_col='Class', frame_col='frame', fps=30.0):
        return self.player_calculator.calculate_player_distances_improved(df, id_col, x_col, y_col, class_col, frame_col, fps)

    def analyze_player_movement_patterns(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                       class_col='Class', frame_col='frame'):
        return self.player_calculator.analyze_player_movement_patterns(df, id_col, x_col, y_col, class_col, frame_col)

    def validate_player_measurements(self, player_results, rally_duration_seconds=None):
        return self.player_calculator.validate_player_measurements(player_results, rally_duration_seconds)

    # Ball Distance Methods
    def calculate_ball_distance_improved(self, df, x_col='X_court', y_col='Y_court', 
                                        class_col='Class', frame_col='frame', fps=30.0):
        return self.ball_calculator.calculate_ball_distance_improved(df, x_col, y_col, class_col, frame_col, fps)

    def calculate_ball_distance_with_trajectory_estimation(self, df, x_col='X_court', y_col='Y_court',
                                                          class_col='Class', frame_col='frame', fps=30.0):
        return self.ball_calculator.calculate_ball_distance_with_trajectory_estimation(df, x_col, y_col, class_col, frame_col, fps)

    def validate_ball_measurements(self, ball_results, rally_duration_seconds=None):
        return self.ball_calculator.validate_ball_measurements(ball_results, rally_duration_seconds)

    def analyze_ball_trajectory(self, df, x_col='X_court', y_col='Y_court', 
                               class_col='Class', frame_col='frame'):
        return self.ball_calculator.analyze_ball_trajectory(df, x_col, y_col, class_col, frame_col)


def enhanced_video_processing_pipeline(video_path, data_csv_path, sample_interval=1.0, max_frames=30, use_corner_diagnostics=True):
    """
    Enhanced processing pipeline with improved coordinate mapping and diagnostics
    """
    tracker = EnhancedTennisCourtTracker()
    
    print("=== ENHANCED TENNIS COURT PROCESSING PIPELINE ===")
    print("=== STEP 1: Detecting Court Corners from Video ===")
    
    try:
        average_corners, all_frame_corners, frame_info, representative_frame = tracker.detect_corners_from_video(
            video_path, sample_interval=sample_interval, max_frames=max_frames, debug=True
        )
        
        if average_corners is None:
            print("[ERROR] Error: Could not detect corners from video")
            return None, None, None, None, None, None
        
        print(f"[SUCCESS] Successfully calculated average corners from {len(all_frame_corners)} frames")
        
        # Print corner analysis
        valid_frames = sum(1 for info in frame_info if info['valid'])
        print(f"[INFO] Frame processing: {valid_frames}/{len(frame_info)} frames had valid corners")
        
        # Visualize corners
        if representative_frame is not None:
            print("=== STEP 1.5: Visualizing Average Corners ===")
            csv_dir = os.path.dirname(data_csv_path)
            csv_filename = os.path.basename(data_csv_path)
            image_filename = csv_filename.replace('.csv', '_enhanced_corners.jpg')
            image_output_path = os.path.join(csv_dir, image_filename)
            
            tracker.visualize_average_corners(
                representative_frame, 
                average_corners, 
                output_path=image_output_path,
                show_image=False
            )
            print(f"[INFO] Corner visualization saved to: {image_output_path}")
        
    except Exception as e:
        print(f"[ERROR] Error processing video: {e}")
        return None, None, None, None, None, None
    
    print("=== STEP 2: Computing Enhanced Homography ===")
    try:
        # Use enhanced homography computation
        result = tracker.compute_homography(average_corners)
        if isinstance(result, tuple):
            H, ordered_corners = result
            print("[SUCCESS] Enhanced homography computed with corner ordering")
        else:
            H = result
            ordered_corners = None
            print("[SUCCESS] Basic homography computed")
        
        # Validate homography
        is_valid = tracker.validate_homography(H, average_corners)
        if not is_valid:
            print("[WARNING] WARNING: Homography validation failed - results may be unrealistic")
        else:
            print("[SUCCESS] Homography validation passed")
            
    except Exception as e:
        print(f"[ERROR] Error computing homography: {e}")
        return None, None, None, None, None, None
    
    print("=== STEP 3: Loading and Processing Tracking Data ===")
    df = pd.read_csv(data_csv_path)
    print(f"[INFO] Loaded {len(df)} tracking points")
    
    # Apply enhanced mapping
    df_mapped = tracker.map_pixels_to_court(df, H)
    print("[SUCCESS] Applied enhanced coordinate mapping with court constraints")
    
    # Debug and diagnose
    if use_corner_diagnostics:
        print("=== STEP 3.5: Diagnostic Analysis ===")
        tracker.debug_coordinates(df_mapped)
        mapping_issues = tracker.diagnose_mapping_quality(df_mapped, average_corners)
    else:
        mapping_issues = []
    
    print("=== STEP 4: Calculating Enhanced Distance Metrics ===")
    
    # Use appropriate coordinate columns based on what's available
    x_col = 'X_court_clamped' if 'X_court_clamped' in df_mapped.columns else 'X_court'
    y_col = 'Y_court_clamped' if 'Y_court_clamped' in df_mapped.columns else 'Y_court'
    
    player_distances = tracker.calculate_player_distances_improved(
        df_mapped, x_col=x_col, y_col=y_col
    )
    print("[INFO] Player distances calculated:")
    print(player_distances)
    
    ball_results = tracker.calculate_ball_distance_improved(
        df_mapped, x_col=x_col, y_col=y_col, fps=30.0
    )
    print("[INFO] Ball trajectory analyzed:")
    tracker.validate_ball_measurements(ball_results)
    
    print("=== STEP 5: Saving Enhanced Results ===")
    
    # Save main results
    output_path = data_csv_path.replace('.csv', '_enhanced_court_coords.csv')
    df_mapped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[SAVE] Enhanced results saved to: {output_path}")
    
    # Save diagnostics if enabled
    if use_corner_diagnostics:
        diagnostics = {
            'mapping_issues': mapping_issues,
            'coordinate_ranges': {
                'x_min': float(df_mapped['X_court'].min()),
                'x_max': float(df_mapped['X_court'].max()),
                'y_min': float(df_mapped['Y_court'].min()),
                'y_max': float(df_mapped['Y_court'].max()),
            },
            'outlier_count': int(df_mapped['is_outlier'].sum()) if 'is_outlier' in df_mapped.columns else 0,
            'total_points': len(df_mapped),
            'homography_valid': is_valid
        }
        
        diagnostics_path = data_csv_path.replace('.csv', '_mapping_diagnostics.json')
        with open(diagnostics_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"[SAVE] Diagnostics saved to: {diagnostics_path}")
    else:
        diagnostics = None
    
    # Save additional analysis files
    corner_analysis_path = data_csv_path.replace('.csv', '_corner_analysis.csv')
    corner_df = pd.DataFrame(frame_info)
    corner_df.to_csv(corner_analysis_path, index=False, encoding='utf-8')
    
    player_analysis_path = data_csv_path.replace('.csv', '_player_analysis.csv')
    player_distances.to_csv(player_analysis_path, index=False, encoding='utf-8')
    
    ball_analysis_path = data_csv_path.replace('.csv', '_ball_analysis.csv')
    ball_df = pd.DataFrame([ball_results])
    ball_df.to_csv(ball_analysis_path, index=False, encoding='utf-8')
    
    print("[SUCCESS] All analysis files saved")
    print("\n[COMPLETE] Enhanced Processing Complete!")
    print(f"   [INFO] Main results: {output_path}")
    print(f"   [INFO] Use columns: {x_col}, {y_col}")
    print(f"   [INFO] Court zones available: {'court_zone' in df_mapped.columns}")
    print(f"   [INFO] Outliers detected: {'is_outlier' in df_mapped.columns}")
    
    return df_mapped, player_distances, ball_results, average_corners, frame_info, diagnostics


def process_tennis_video(video_path, data_csv_path, method='enhanced', **kwargs):
    """
    Convenience function to process tennis video with the best available method
    """
    if method == 'dynamic':
        # For now, just use the enhanced method since dynamic has more complexity
        print("[INFO] Dynamic method not fully implemented, using enhanced method")
        return enhanced_video_processing_pipeline(video_path, data_csv_path, **kwargs)
    else:
        return enhanced_video_processing_pipeline(video_path, data_csv_path, **kwargs)