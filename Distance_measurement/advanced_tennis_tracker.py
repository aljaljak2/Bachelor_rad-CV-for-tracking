"""
Advanced Tennis Tracker with Improved Distance Calculation and Outlier Handling
"""
import pandas as pd
import numpy as np
import cv2
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class AdvancedTennisTracker:
    def __init__(self, court_width=10.97, court_length=23.77):
        self.court_width = court_width
        self.court_length = court_length
        
        # Define realistic boundaries with larger buffer for initial filtering
        self.extended_buffer = 10.0  # meters
        self.tight_buffer = 3.0     # meters for final filtering
        
        self.court_lines = {
            'baseline_near': 0,
            'service_line_near': 6.40,
            'net': court_length / 2,
            'service_line_far': court_length - 6.40,
            'baseline_far': court_length,
            'center_line': court_width / 2
        }

    def detect_and_fix_corner_issues(self, corners_data):
        """
        Detect and fix corner detection issues that lead to poor homography
        """
        print("\n=== CORNER ISSUE DETECTION ===")
        
        if len(corners_data) < 4:
            print("[ERROR] Not enough corners detected")
            return None
        
        # Analyze corner stability across frames
        corner_std = np.std(corners_data, axis=0)
        print(f"Corner stability (std dev): {corner_std}")
        
        # Check for suspicious perfect validation (0.000m errors)
        # This often indicates corner detection picked the same points repeatedly
        max_std = np.max(corner_std)
        if max_std < 1.0:  # Very low variation
            print("[WARNING] Corners show very low variation - may be detection artifacts")
            print("Consider manual corner specification or different detection parameters")
        
        # Use median corners instead of mean for robustness
        median_corners = np.median(corners_data, axis=0)
        print(f"Using median corners: {median_corners}")
        
        return median_corners

    def robust_outlier_detection(self, df):
        """
        Advanced outlier detection using multiple methods
        """
        print("\n=== ADVANCED OUTLIER DETECTION ===")
        
        # Method 1: Statistical outliers (Z-score)
        z_scores_x = np.abs(stats.zscore(df['X_court']))
        z_scores_y = np.abs(stats.zscore(df['Y_court']))
        statistical_outliers = (z_scores_x > 3) | (z_scores_y > 3)
        
        # Method 2: IQR-based outliers
        Q1_x, Q3_x = df['X_court'].quantile([0.25, 0.75])
        Q1_y, Q3_y = df['Y_court'].quantile([0.25, 0.75])
        IQR_x, IQR_y = Q3_x - Q1_x, Q3_y - Q1_y
        
        iqr_outliers = (
            (df['X_court'] < Q1_x - 1.5 * IQR_x) | 
            (df['X_court'] > Q3_x + 1.5 * IQR_x) |
            (df['Y_court'] < Q1_y - 1.5 * IQR_y) | 
            (df['Y_court'] > Q3_y + 1.5 * IQR_y)
        )
        
        # Method 3: Court boundary outliers
        court_outliers = (
            (df['X_court'] < -self.extended_buffer) | 
            (df['X_court'] > self.court_width + self.extended_buffer) |
            (df['Y_court'] < -self.extended_buffer) | 
            (df['Y_court'] > self.court_length + self.extended_buffer)
        )
        
        # Method 4: DBSCAN clustering to find spatial outliers
        coordinates = df[['X_court', 'Y_court']].values
        scaler = StandardScaler()
        coordinates_scaled = scaler.fit_transform(coordinates)
        
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(coordinates_scaled)
        cluster_outliers = clustering.labels_ == -1
        
        # Combine methods
        combined_outliers = statistical_outliers | iqr_outliers | court_outliers | cluster_outliers
        
        print(f"Statistical outliers: {statistical_outliers.sum()} ({statistical_outliers.sum()/len(df)*100:.1f}%)")
        print(f"IQR outliers: {iqr_outliers.sum()} ({iqr_outliers.sum()/len(df)*100:.1f}%)")
        print(f"Court boundary outliers: {court_outliers.sum()} ({court_outliers.sum()/len(df)*100:.1f}%)")
        print(f"Clustering outliers: {cluster_outliers.sum()} ({cluster_outliers.sum()/len(df)*100:.1f}%)")
        print(f"Combined outliers: {combined_outliers.sum()} ({combined_outliers.sum()/len(df)*100:.1f}%)")
        
        return combined_outliers

    def interpolate_missing_positions(self, df, id_col='ID'):
        """
        Interpolate missing positions for continuous tracking
        """
        print("\n=== POSITION INTERPOLATION ===")
        
        df_interpolated = df.copy()
        
        for obj_id in df['ID'].unique():
            obj_data = df[df[id_col] == obj_id].sort_values('frame')
            
            if len(obj_data) < 2:
                continue
            
            # Create frame range for this object
            frame_min, frame_max = obj_data['frame'].min(), obj_data['frame'].max()
            all_frames = pd.DataFrame({'frame': range(frame_min, frame_max + 1)})
            
            # Merge with object data
            obj_complete = pd.merge(all_frames, obj_data, on='frame', how='left')
            
            # Interpolate missing positions
            obj_complete['X_court_interp'] = obj_complete['X_court'].interpolate(method='linear')
            obj_complete['Y_court_interp'] = obj_complete['Y_court'].interpolate(method='linear')
            
            # Fill other columns
            obj_complete['ID'] = obj_complete['ID'].fillna(obj_id)
            obj_complete['Class'] = obj_complete['Class'].fillna(method='ffill').fillna(method='bfill')
            
            # Only keep interpolated frames that were missing
            missing_frames = obj_complete[obj_complete['X_court'].isna() & obj_complete['X_court_interp'].notna()]
            
            if len(missing_frames) > 0:
                print(f"Object {obj_id}: Interpolated {len(missing_frames)} missing positions")
                
                # Add interpolated data back to main dataframe
                for _, row in missing_frames.iterrows():
                    new_row = {
                        'frame': row['frame'],
                        'ID': obj_id,
                        'X_court': row['X_court_interp'],
                        'Y_court': row['Y_court_interp'],
                        'Class': row['Class'],
                        'interpolated': True
                    }
                    df_interpolated = pd.concat([df_interpolated, pd.DataFrame([new_row])], ignore_index=True)
        
        return df_interpolated.sort_values(['frame', 'ID'])

    def calculate_corrected_distances(self, df, fps=25.0, id_col='ID', class_col='Class'):
        """
        Calculate distances with outlier filtering and interpolation
        """
        print("\n=== CORRECTED DISTANCE CALCULATION ===")
        
        # Step 1: Remove outliers
        outliers = self.robust_outlier_detection(df)
        df_clean = df[~outliers].copy()
        
        print(f"Removed {outliers.sum()} outliers, {len(df_clean)} points remaining")
        
        # Step 2: Interpolate missing positions
        df_interpolated = self.interpolate_missing_positions(df_clean, id_col)
        
        # Step 3: Calculate distances for each object
        results = []
        
        for obj_id in df_interpolated[id_col].unique():
            obj_data = df_interpolated[df_interpolated[id_col] == obj_id].sort_values('frame')
            
            if len(obj_data) < 2:
                continue
            
            # Calculate frame-to-frame distances
            positions = obj_data[['X_court', 'Y_court']].values
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            
            # Calculate time intervals (handle variable frame rates)
            frame_intervals = np.diff(obj_data['frame'].values) / fps
            
            # Calculate speeds (with outlier filtering)
            speeds = distances / frame_intervals  # m/s
            speeds_kmh = speeds * 3.6  # km/h
            
            # Filter unrealistic speeds (> 50 km/h for players, > 200 km/h for ball)
            obj_class = obj_data[class_col].iloc[0] if class_col in obj_data.columns else 'Unknown'
            max_speed = 200 if obj_class == 'Ball' else 50
            
            realistic_speeds = speeds_kmh[speeds_kmh <= max_speed]
            realistic_distances = distances[speeds_kmh <= max_speed]
            
            # Calculate statistics
            total_distance = realistic_distances.sum()
            avg_speed = realistic_speeds.mean() if len(realistic_speeds) > 0 else 0
            max_speed_actual = realistic_speeds.max() if len(realistic_speeds) > 0 else 0
            
            # Calculate time span
            time_span = (obj_data['frame'].max() - obj_data['frame'].min()) / fps
            
            # Calculate court coverage (area of bounding box)
            x_range = obj_data['X_court'].max() - obj_data['X_court'].min()
            y_range = obj_data['Y_court'].max() - obj_data['Y_court'].min()
            court_coverage = x_range * y_range
            
            results.append({
                'ID': obj_id,
                'Class': obj_class,
                'TotalDistance': total_distance,
                'AverageSpeed': avg_speed,
                'MaxSpeed': max_speed_actual,
                'TimeSpan': time_span,
                'CourtCoverage': court_coverage,
                'DataPoints': len(obj_data),
                'InterpolatedPoints': obj_data.get('interpolated', False).sum(),
                'OutlierRate': outliers[df[id_col] == obj_id].sum() / len(df[df[id_col] == obj_id]) * 100
            })
        
        return pd.DataFrame(results), df_clean, df_interpolated

    def validate_and_suggest_improvements(self, results_df, original_df):
        """
        Validate results and suggest improvements
        """
        print("\n=== VALIDATION AND SUGGESTIONS ===")
        
        issues = []
        suggestions = []
        
        # Check for unrealistic distances
        player_results = results_df[results_df['Class'] == 'Player']
        
        if len(player_results) > 0:
            avg_distance = player_results['TotalDistance'].mean()
            max_distance = player_results['TotalDistance'].max()
            
            print(f"Player distance statistics:")
            print(f"  Average total distance: {avg_distance:.2f}m")
            print(f"  Maximum total distance: {max_distance:.2f}m")
            print(f"  Average time span: {player_results['TimeSpan'].mean():.2f}s")
            
            # Check if distances are too low (common issue)
            if avg_distance < 5.0:
                issues.append("Player distances appear too low")
                suggestions.append("1. Check homography computation - corners may be incorrectly detected")
                suggestions.append("2. Verify frame rate setting (currently assuming 25 fps)")
                suggestions.append("3. Consider manual corner specification")
            
            # Check outlier rates
            high_outlier_objects = results_df[results_df['OutlierRate'] > 30]
            if len(high_outlier_objects) > 0:
                issues.append(f"{len(high_outlier_objects)} objects have >30% outlier rate")
                suggestions.append("4. Improve corner detection stability")
                suggestions.append("5. Use dynamic homography updates")
        
        # Check coordinate ranges
        x_range = original_df['X_court'].max() - original_df['X_court'].min()
        y_range = original_df['Y_court'].max() - original_df['Y_court'].min()
        
        if x_range > 30 or y_range > 40:
            issues.append("Coordinate ranges are too large")
            suggestions.append("6. Recalibrate court corner detection")
            suggestions.append("7. Check for camera movement or distortion")
        
        if issues:
            print("\n[WARNING] Issues detected:")
            for issue in issues:
                print(f"  - {issue}")
            print("\n[SUGGESTIONS] Improvement suggestions:")
            for suggestion in suggestions:
                print(f"  {suggestion}")
        else:
            print("\n[SUCCESS] Results look reasonable!")
        
        return issues, suggestions

    def manual_corner_correction(self, image_path=None):
        """
        Provide manual corner specification when automatic detection fails
        """
        print("\n=== MANUAL CORNER CORRECTION ===")
        print("Based on your court image, try these manually specified corners:")
        print("(Update these coordinates based on your specific court image)")
        
        # These are example coordinates - you should update based on your court image
        manual_corners = [
            (136.9, 615.0),   # Bottom-left baseline
            (1223.5, 621.0),  # Bottom-right baseline
            (890.6, 107.1),   # Top-right baseline  
            (375.4, 227.4)    # Top-left baseline
        ]
        
        print("Suggested manual corners:")
        labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        for corner, label in zip(manual_corners, labels):
            print(f"  {label}: {corner}")
        
        return manual_corners

    def recompute_homography_with_manual_corners(self, manual_corners):
        """
        Recompute homography using manually specified corners
        """
        print("\n=== RECOMPUTING HOMOGRAPHY ===")
        
        # Real court corners (standard)
        real_corners = np.array([
            [0, 0],                          # Bottom-left
            [self.court_width, 0],           # Bottom-right
            [self.court_width, self.court_length],  # Top-right
            [0, self.court_length]           # Top-left
        ], dtype=np.float32)
        
        pixel_corners = np.array(manual_corners, dtype=np.float32)
        
        # Compute homography
        H, status = cv2.findHomography(pixel_corners, real_corners, cv2.RANSAC, 5.0)
        
        if H is None:
            print("[ERROR] Failed to compute homography with manual corners")
            return None
        
        # Validate
        transformed = cv2.perspectiveTransform(pixel_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(transformed - real_corners, axis=1)
        
        print("Manual corner validation:")
        for i, (label, error) in enumerate(zip(["BL", "BR", "TR", "TL"], errors)):
            print(f"  {label}: {error:.3f}m error")
        
        max_error = np.max(errors)
        if max_error < 0.5:
            print(f"[SUCCESS] Manual homography looks good (max error: {max_error:.3f}m)")
        else:
            print(f"[WARNING] Manual homography has high error (max error: {max_error:.3f}m)")
        
        return H


def process_with_advanced_tracking(data_csv_path, fps=25.0, use_manual_corners=False):
    """
    Process tennis tracking data with advanced outlier handling and distance correction
    """
    print("=== ADVANCED TENNIS TRACKING PROCESSING ===")
    
    # Load data
    df = pd.read_csv(data_csv_path)
    print(f"Loaded {len(df)} tracking points")
    
    # Initialize advanced tracker
    tracker = AdvancedTennisTracker()
    
    # Process with advanced methods
    results_df, df_clean, df_interpolated = tracker.calculate_corrected_distances(df, fps=fps)
    
    # Validate and get suggestions
    issues, suggestions = tracker.validate_and_suggest_improvements(results_df, df)
    
    # If manual corner correction is needed
    if use_manual_corners or len(issues) > 0:
        print("\n" + "="*60)
        print("MANUAL CORNER CORRECTION RECOMMENDED")
        print("="*60)
        
        manual_corners = tracker.manual_corner_correction()
        
        # You can uncomment and modify this section to apply manual correction:
        # H_manual = tracker.recompute_homography_with_manual_corners(manual_corners)
        # if H_manual is not None:
        #     # Reapply mapping with manual homography
        #     # ... (implement remapping logic)
        #     pass
    
    # Save results
    output_path = data_csv_path.replace('.csv', '_advanced_corrected.csv')
    df_interpolated.to_csv(output_path, index=False)
    
    results_path = data_csv_path.replace('.csv', '_advanced_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n[SAVE] Advanced results saved to: {output_path}")
    print(f"[SAVE] Distance analysis saved to: {results_path}")
    
    return results_df, df_clean, df_interpolated, issues, suggestions


# Usage example
def fix_distance_calculation(csv_path):
    """
    Main function to fix distance calculation issues
    """
    # Process with advanced tracking
    results, clean_df, interpolated_df, issues, suggestions = process_with_advanced_tracking(
        csv_path, 
        fps=25.0,  # Adjust based on your video frame rate
        use_manual_corners=True  # Set to True if automatic corner detection is poor
    )
    
    print("\n=== FINAL RESULTS ===")
    print("Player movement analysis:")
    player_results = results[results['Class'] == 'Player']
    
    for _, player in player_results.iterrows():
        if player['TotalDistance'] > 0.1:  # Only show players with significant movement
            print(f"\nPlayer {player['ID']}:")
            print(f"  Total distance: {player['TotalDistance']:.2f}m")
            print(f"  Average speed: {player['AverageSpeed']:.1f} km/h")
            print(f"  Max speed: {player['MaxSpeed']:.1f} km/h")
            print(f"  Time span: {player['TimeSpan']:.1f}s")
            print(f"  Outlier rate: {player['OutlierRate']:.1f}%")
    
    return results, clean_df, interpolated_df