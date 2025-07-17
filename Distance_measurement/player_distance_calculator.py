
"""
Player Distance Calculator Module for Tennis Court Tracking
"""
import numpy as np
import pandas as pd
from collections import Counter


class PlayerDistanceCalculator:
    def __init__(self):
        # Tennis court dimensions in meters (doubles)
        self.court_width = 10.97
        self.court_length = 23.77

    def calculate_player_distances_improved(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                      class_col='Class', frame_col='frame', fps=30.0):
        """
        Improved player distance calculation with temporal awareness and movement analysis
    
        Args:
            df: DataFrame with tracking data
            id_col: Column name for player ID
            x_col, y_col: Column names for court coordinates
            class_col: Column name for object classification
            frame_col: Column name for frame numbers
            fps: Video frame rate (frames per second)
    
        Returns:
            DataFrame with comprehensive player movement statistics
        """
        players = df[df[class_col] == 'Player'].copy()
    
        if players.empty:
            return pd.DataFrame(columns=[
                'ID', 'TotalDistance', 'AverageSpeed', 'MaxSpeed', 'FrameCount', 
                'ValidMovements', 'FilteredMovements', 'DetectionGaps', 'MovementSegments',
                'CourtCoverage', 'TimeActive'
            ])
    
        results = []
    
        for pid, group in players.groupby(id_col):
            # Sort by frame number
            group = group.sort_values(frame_col).reset_index(drop=True)
            coords = group[[x_col, y_col]].values
            frames = group[frame_col].values
        
            if len(coords) < 2:
                results.append({
                    'ID': pid,
                    'TotalDistance': 0.0,
                    'AverageSpeed': 0.0,
                    'MaxSpeed': 0.0,
                    'FrameCount': len(coords),
                    'ValidMovements': 0,
                    'FilteredMovements': 0,
                    'DetectionGaps': 0,
                    'MovementSegments': 0,
                    'CourtCoverage': 0.0,
                    'TimeActive': 0.0
                })
                continue
        
            # Calculate frame differences and time intervals
            frame_diffs = np.diff(frames)
            time_intervals = frame_diffs / fps
        
            # Calculate coordinate differences and distances
            coord_diffs = np.diff(coords, axis=0)
            frame_distances = np.linalg.norm(coord_diffs, axis=1)
        
            # Calculate instantaneous speeds (m/s)
            speeds = np.divide(frame_distances, time_intervals,
                            out=np.zeros_like(frame_distances),
                            where=time_intervals != 0)
        
            # Filter unrealistic movements
            max_realistic_speed = 12.0  # m/s (top tennis players can reach ~11 m/s)
            min_movement_threshold = 0.05  # m (ignore tiny movements due to detection noise)
        
            # Create validity mask
            valid_mask = (
                (speeds > 0) & 
                (speeds <= max_realistic_speed) & 
                (frame_distances >= min_movement_threshold)
            )
        
            # Handle detection gaps
            detection_gaps = np.sum(frame_diffs > 1)
        
            # Calculate movement segments (continuous tracking periods)
            movement_segments = self._calculate_movement_segments(frames)
        
            # Calculate total distance with gap handling
            total_distance = 0.0
            valid_movements = 0
        
            for i in range(len(frame_distances)):
                if valid_mask[i]:
                    if frame_diffs[i] == 1:
                        # Consecutive frames - use direct distance
                        total_distance += frame_distances[i]
                        valid_movements += 1
                    elif frame_diffs[i] <= 3:  # Small gap - estimate movement
                        # For small gaps, use linear interpolation
                        estimated_distance = frame_distances[i] * (frame_diffs[i] / frame_diffs[i])
                        total_distance += estimated_distance
                        valid_movements += 1
                    else:
                        # Large gap - player likely stationary or off-court
                        # Don't count this movement
                        pass
        
            # Calculate speeds for valid movements only
            valid_speeds = speeds[valid_mask]
            average_speed = np.mean(valid_speeds) if len(valid_speeds) > 0 else 0.0
            max_speed = np.max(valid_speeds) if len(valid_speeds) > 0 else 0.0
        
            # Calculate court coverage (area covered by player)
            court_coverage = self._calculate_court_coverage(coords)
        
            # Calculate active time (time with valid detections)
            time_active = (frames[-1] - frames[0]) / fps if len(frames) > 1 else 0.0
        
            # Count filtered movements
            filtered_movements = np.sum(~valid_mask)
        
            results.append({
                'ID': pid,
                'TotalDistance': total_distance,
                'AverageSpeed': average_speed,
                'MaxSpeed': max_speed,
                'AverageSpeedKmh': average_speed * 3.6,
                'MaxSpeedKmh': max_speed * 3.6,
                'FrameCount': len(coords),
                'ValidMovements': valid_movements,
                'FilteredMovements': filtered_movements,
                'DetectionGaps': detection_gaps,
                'MovementSegments': movement_segments,
                'CourtCoverage': court_coverage,
                'TimeActive': time_active,
                'DistancePerMinute': (total_distance / time_active * 60) if time_active > 0 else 0.0
            })
    
        result_df = pd.DataFrame(results)
    
        # Print analysis
        print("\n=== PLAYER MOVEMENT ANALYSIS ===")
        for _, row in result_df.iterrows():
            print(f"\nPlayer {row['ID']}:")
            print(f"  Total distance: {row['TotalDistance']:.1f}m")
            print(f"  Average speed: {row['AverageSpeedKmh']:.1f} km/h")
            print(f"  Max speed: {row['MaxSpeedKmh']:.1f} km/h")
            print(f"  Court coverage: {row['CourtCoverage']:.1f}m^2")
            print(f"  Active time: {row['TimeActive']:.1f}s")
            print(f"  Distance per minute: {row['DistancePerMinute']:.1f}m/min")
            print(f"  Detection gaps: {row['DetectionGaps']}")
            print(f"  Movement segments: {row['MovementSegments']}")
    
        return result_df

    def analyze_player_movement_patterns(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                   class_col='Class', frame_col='frame'):
        """
        Analyze detailed movement patterns for each player
        
        Args:
            df: DataFrame with tracking data
            id_col: Column name for player ID
            x_col, y_col: Column names for court coordinates
            class_col: Column name for object classification
            frame_col: Column name for frame numbers
            
        Returns:
            Dictionary with movement pattern analysis for each player
        """
        players = df[df[class_col] == 'Player'].copy()
    
        if players.empty:
            return {}
    
        analysis = {}
    
        for pid, group in players.groupby(id_col):
            group = group.sort_values(frame_col)
            coords = group[[x_col, y_col]].values
        
            if len(coords) < 10:  # Need sufficient data points
                continue
        
            # Calculate movement vectors
            movements = np.diff(coords, axis=0)
            movement_magnitudes = np.linalg.norm(movements, axis=1)
        
            # Analyze movement directions
            movement_angles = np.arctan2(movements[:, 1], movements[:, 0])
        
            # Calculate acceleration (change in speed)
            speed_changes = np.diff(movement_magnitudes)
        
            # Identify different movement types
            stationary_threshold = 0.1  # m
            walking_threshold = 1.0     # m/frame
            running_threshold = 3.0     # m/frame
        
            stationary_frames = np.sum(movement_magnitudes < stationary_threshold)
            walking_frames = np.sum((movement_magnitudes >= stationary_threshold) & 
                               (movement_magnitudes < walking_threshold))
            running_frames = np.sum(movement_magnitudes >= walking_threshold)
        
            analysis[pid] = {
                'total_frames': int(len(coords)),
                'stationary_frames': int(stationary_frames),
                'walking_frames': int(walking_frames),
                'running_frames': int(running_frames),
                'dominant_direction': self._get_dominant_direction(movement_angles),
                'movement_variability': float(np.std(movement_magnitudes)),
                'avg_acceleration': float(np.mean(np.abs(speed_changes))),
                'position_heat_map': self._create_position_heatmap(coords)
            }
    
        return analysis

    def validate_player_measurements(self, player_results, rally_duration_seconds=None):
        """
        Validate player measurements against realistic tennis movement patterns
        
        Args:
            player_results: DataFrame with player movement statistics
            rally_duration_seconds: Duration of the rally in seconds (optional)
        """
        print("\n=== PLAYER MEASUREMENT VALIDATION ===")
    
        for _, player in player_results.iterrows():
            pid = player['ID']
            total_distance = player['TotalDistance']
            avg_speed = player['AverageSpeedKmh']
            max_speed = player['MaxSpeedKmh']
        
            print(f"\nPlayer {pid}:")
            print(f"  Total distance: {total_distance:.1f}m")
        
            # Validate against realistic tennis movement
            if rally_duration_seconds and rally_duration_seconds > 0:
                distance_per_second = total_distance / rally_duration_seconds
                print(f"  Distance per second: {distance_per_second:.1f}m/s")
            
                if 0.5 <= distance_per_second <= 8.0:
                    print("  [OK] Distance per second is realistic")
                else:
                    print("  [W] Distance per second seems unrealistic")
        
            # Speed validation
            if 0 <= avg_speed <= 25:  # km/h
                print(f"  [OK] Average speed ({avg_speed:.1f} km/h) is realistic")
            else:
                print(f"  [W] Average speed ({avg_speed:.1f} km/h) seems unrealistic")
        
            if 0 <= max_speed <= 45:  # km/h
                print(f"  [OK] Max speed ({max_speed:.1f} km/h) is realistic")
            else:
                print(f"  [W] Max speed ({max_speed:.1f} km/h) seems unrealistic")
        
            # Court coverage validation
            court_coverage = player['CourtCoverage']
            total_court_area = self.court_width * self.court_length  # ~260 m²
            coverage_percentage = (court_coverage / total_court_area) * 100
        
            print(f"  Court coverage: {court_coverage:.1f}m^2 ({coverage_percentage:.1f}% of court)")
        
            if 1 <= coverage_percentage <= 50:
                print("  [OK] Court coverage is realistic")
            else:
                print("  ⚠ Court coverage seems unrealistic")

    def _calculate_movement_segments(self, frames):
        """
        Calculate the number of continuous movement segments
        
        Args:
            frames: Array of frame numbers
            
        Returns:
            Number of movement segments
        """
        if len(frames) <= 1:
            return 0
    
        segments = 1
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 1:  # Gap detected
                segments += 1
    
        return segments

    def _calculate_court_coverage(self, coords):
        """
        Calculate the area covered by player movement (simplified as convex hull area)
        
        Args:
            coords: Array of coordinate positions
            
        Returns:
            Area covered in square meters
        """
        if len(coords) < 3:
            return 0.0
    
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            return hull.volume  # In 2D, volume is actually area
        except:
            # Fallback: use bounding box area
            x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
            y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
            return x_range * y_range

    def _get_dominant_direction(self, angles):
        """
        Determine the dominant movement direction
        
        Args:
            angles: Array of movement angles in radians
            
        Returns:
            String representing the dominant direction
        """
        # Convert to degrees and categorize
        angles_deg = np.degrees(angles)
    
        # Categorize into 8 directions
        directions = []
        for angle in angles_deg:
            if -22.5 <= angle < 22.5:
                directions.append('E')
            elif 22.5 <= angle < 67.5:
                directions.append('NE')
            elif 67.5 <= angle < 112.5:
                directions.append('N')
            elif 112.5 <= angle < 157.5:
                directions.append('NW')
            elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                directions.append('W')
            elif -157.5 <= angle < -112.5:
                directions.append('SW')
            elif -112.5 <= angle < -67.5:
                directions.append('S')
            elif -67.5 <= angle < -22.5:
                directions.append('SE')
    
        # Find most common direction
        if directions:
            return Counter(directions).most_common(1)[0][0]
        return 'Unknown'

    def _create_position_heatmap(self, coords):
        """
        Create a simplified position heatmap (JSON-safe)
        
        Args:
            coords: Array of coordinate positions
            
        Returns:
            Dictionary with heatmap information
        """
        # Divide court into grid and count positions
        x_bins = np.linspace(0, self.court_width, 11)  # 10 divisions
        y_bins = np.linspace(0, self.court_length, 24)  # 23 divisions

        heatmap, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_bins, y_bins])

        # Find the most occupied cell
        max_cell = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        max_cell = tuple(int(i) for i in max_cell)  # Convert to native ints

        position_spread = [float(x) for x in np.std(coords, axis=0)]
        center_of_activity = [float(x) for x in np.mean(coords, axis=0)]

        return {
            'most_occupied_region': max_cell,
            'position_spread': position_spread,
            'center_of_activity': center_of_activity
        }
