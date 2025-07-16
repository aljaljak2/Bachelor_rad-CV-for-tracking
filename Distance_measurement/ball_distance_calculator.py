"""
Ball Distance Calculator Module for Tennis Court Tracking
"""
import numpy as np
import pandas as pd


class BallDistanceCalculator:
    def __init__(self):
        # Tennis court dimensions in meters (doubles)
        self.court_width = 10.97
        self.court_length = 23.77

    def calculate_ball_distance_improved(self, df, x_col='X_court', y_col='Y_court', 
                                        class_col='Class', frame_col='frame', fps=30.0):
        """
        Improved ball distance calculation with temporal awareness and trajectory estimation

        Args:
            df: DataFrame with tracking data
            x_col, y_col: Column names for court coordinates
            class_col: Column name for object classification
            frame_col: Column name for frame numbers
            fps: Video frame rate (frames per second)

        Returns:
            dict: Contains total distance, average speed, max speed, and trajectory info
        """
        ball_data = df[df[class_col] == 'Ball'].copy()

        if ball_data.empty or len(ball_data) < 2:
            return {
                'total_distance': 0.0,
                'average_speed': 0.0,
                'max_speed': 0.0,
                'trajectory_segments': 0,
                'detection_gaps': 0,
                'valid_points': 0
            }

        # Sort by frame number
        ball_data = ball_data.sort_values(frame_col).reset_index(drop=True)

        # Calculate time intervals between detections
        frame_diffs = np.diff(ball_data[frame_col].values)
        time_intervals = frame_diffs / fps  # Convert to seconds

        # Get coordinate differences
        coords = ball_data[[x_col, y_col]].values
        coord_diffs = np.diff(coords, axis=0)
        frame_distances = np.linalg.norm(coord_diffs, axis=1)

        # Calculate instantaneous speeds (m/s)
        speeds = np.divide(frame_distances, time_intervals,
                           out=np.zeros_like(frame_distances),
                           where=time_intervals != 0)

        # Filter out unrealistic speeds
        max_realistic_speed = 70.0  # m/s
        valid_speed_mask = (speeds > 0) & (speeds <= max_realistic_speed)

        # Detect gaps in detection (where frame difference > 1)
        detection_gaps = np.sum(frame_diffs > 1)

        # Handle gaps by estimating trajectory
        total_distance = 0.0
        trajectory_segments = 0

        for i in range(len(frame_distances)):
            if valid_speed_mask[i]:
                if frame_diffs[i] == 1:
                    # Consecutive frames - use direct distance
                    total_distance += frame_distances[i]
                else:
                    # Gap in detection - estimate trajectory
                    gap_frames = frame_diffs[i]
                    gap_time = time_intervals[i]

                    # Simple linear interpolation for gaps
                    estimated_distance = frame_distances[i]
                    total_distance += estimated_distance

                    print(f"INFO: Gap detected {gap_frames} frames, estimated distance {estimated_distance:.2f}m")

                trajectory_segments += 1

        # Calculate statistics
        valid_speeds = speeds[valid_speed_mask]
        average_speed = np.mean(valid_speeds) if len(valid_speeds) > 0 else 0.0
        max_speed = np.max(valid_speeds) if len(valid_speeds) > 0 else 0.0

        # Convert speeds to km/h for reporting
        average_speed_kmh = average_speed * 3.6
        max_speed_kmh = max_speed * 3.6

        print(f"BALL ANALYSIS: Total distance {total_distance:.2f}m")
        print(f"BALL ANALYSIS: Average speed {average_speed_kmh:.1f} km/h")
        print(f"BALL ANALYSIS: Max speed {max_speed_kmh:.1f} km/h")
        print(f"BALL ANALYSIS: Detection gaps {detection_gaps}")
        print(f"BALL ANALYSIS: Valid trajectory segments {trajectory_segments}")

        return {
            'total_distance': total_distance,
            'average_speed': average_speed,
            'max_speed': max_speed,
            'average_speed_kmh': average_speed_kmh,
            'max_speed_kmh': max_speed_kmh,
            'trajectory_segments': trajectory_segments,
            'detection_gaps': detection_gaps,
            'valid_points': len(ball_data),
            'filtered_unrealistic': np.sum(~valid_speed_mask)
        }

    def calculate_ball_distance_with_trajectory_estimation(self, df, x_col='X_court', y_col='Y_court',
                                                          class_col='Class', frame_col='frame', fps=30.0):
        """
        Advanced ball distance calculation with parabolic trajectory estimation
        
        Args:
            df: DataFrame with tracking data
            x_col, y_col: Column names for court coordinates
            class_col: Column name for object classification
            frame_col: Column name for frame numbers
            fps: Video frame rate (frames per second)
            
        Returns:
            dict: Contains total distance and trajectory estimation info
        """
        ball_data = df[df[class_col] == 'Ball'].copy()

        if ball_data.empty or len(ball_data) < 3:
            return self.calculate_ball_distance_improved(df, x_col, y_col, class_col, frame_col, fps)

        ball_data = ball_data.sort_values(frame_col).reset_index(drop=True)

        total_distance = 0.0
        coords = ball_data[[x_col, y_col]].values
        frames = ball_data[frame_col].values

        i = 0
        while i < len(coords) - 1:
            current_frame = frames[i]
            next_frame = frames[i + 1]

            if next_frame - current_frame == 1:
                # Consecutive frames - direct distance
                distance = np.linalg.norm(coords[i + 1] - coords[i])
                total_distance += distance
                i += 1
            else:
                # Gap in detection - try to estimate trajectory
                gap_size = next_frame - current_frame

                if gap_size <= 5 and i + 1 < len(coords):  # Only estimate for small gaps
                    # Use parabolic trajectory estimation
                    p1 = coords[i]
                    p2 = coords[i + 1]

                    # Estimate trajectory length (accounting for parabolic path)
                    straight_distance = np.linalg.norm(p2 - p1)

                    # Approximate parabolic path as 1.1-1.3 times straight distance
                    trajectory_factor = 1.2  # Conservative estimate
                    estimated_distance = straight_distance * trajectory_factor

                    total_distance += estimated_distance

                    print(f"TRAJECTORY: Gap of {gap_size} frames, estimated distance {estimated_distance:.2f}m")
                else:
                    # Large gap - use direct distance
                    distance = np.linalg.norm(coords[i + 1] - coords[i])
                    total_distance += distance

                    print(f"LARGE GAP: {gap_size} frames, using direct distance {distance:.2f}m")

                i += 1

        return {
            'total_distance': total_distance,
            'method': 'trajectory_estimation',
            'gaps_estimated': True
        }

    def validate_ball_measurements(self, ball_results, rally_duration_seconds=None):
        """
        Validate ball measurements against realistic tennis expectations
        
        Args:
            ball_results: Dictionary with ball movement statistics
            rally_duration_seconds: Duration of the rally in seconds (optional)
            
        Returns:
            The original ball_results dictionary
        """
        print("\n=== BALL MEASUREMENT VALIDATION ===")

        total_distance = ball_results['total_distance']
        avg_speed = ball_results.get('average_speed_kmh', 0)
        max_speed = ball_results.get('max_speed_kmh', 0)

        # Tennis ball physics validation
        print(f"Total ball distance: {total_distance:.2f}m")

        if rally_duration_seconds:
            rally_average_speed = (total_distance / rally_duration_seconds) * 3.6
            print(f"Rally average speed: {rally_average_speed:.1f} km/h")

            # Realistic ranges for tennis
            if 10 <= rally_average_speed <= 150:
                print("[OK] Rally average speed is realistic")
            else:
                print("[W] Rally average speed seems unrealistic")

        if avg_speed > 0:
            print(f"Ball average speed: {avg_speed:.1f} km/h")
            if 20 <= avg_speed <= 200:
                print("[OK] Average speed is realistic")
            else:
                print("[W] Average speed seems unrealistic")

        if max_speed > 0:
            print(f"Ball max speed: {max_speed:.1f} km/h")
            if 50 <= max_speed <= 250:
                print("[OK] Max speed is realistic")
            else:
                print("[W] Max speed seems unrealistic")

        # Distance validation
        if total_distance > 0:
            if 50 <= total_distance <= 2000:  # Typical rally distances
                print("[OK] Total distance is realistic")
            else:
                print("[W] Total distance seems unrealistic")

        return ball_results

    def analyze_ball_trajectory(self, df, x_col='X_court', y_col='Y_court', 
                               class_col='Class', frame_col='frame'):
        """
        Analyze ball trajectory patterns
        
        Args:
            df: DataFrame with tracking data
            x_col, y_col: Column names for court coordinates
            class_col: Column name for object classification
            frame_col: Column name for frame numbers
            
        Returns:
            Dictionary with trajectory analysis
        """
        ball_data = df[df[class_col] == 'Ball'].copy()

        if ball_data.empty or len(ball_data) < 3:
            return {}

        ball_data = ball_data.sort_values(frame_col).reset_index(drop=True)
        coords = ball_data[[x_col, y_col]].values

        # Calculate velocity vectors
        velocities = np.diff(coords, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)

        # Calculate acceleration vectors
        accelerations = np.diff(velocities, axis=0)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

        # Analyze direction changes
        velocity_angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        angle_changes = np.diff(velocity_angles)

        # Identify bounces (sudden direction changes)
        angle_change_threshold = np.pi / 4  # 45 degrees
        potential_bounces = np.where(np.abs(angle_changes) > angle_change_threshold)[0]

        # Calculate trajectory statistics
        analysis = {
            'total_points': len(coords),
            'average_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
            'speed_variability': float(np.std(speeds)),
            'average_acceleration': float(np.mean(acceleration_magnitudes)),
            'max_acceleration': float(np.max(acceleration_magnitudes)),
            'potential_bounces': len(potential_bounces),
            'trajectory_smoothness': float(1.0 / (1.0 + np.std(angle_changes))),
            'court_crossings': self._count_court_crossings(coords)
        }

        return analysis

    def _count_court_crossings(self, coords):
        """
        Count how many times the ball crosses the court
        
        Args:
            coords: Array of ball coordinates
            
        Returns:
            Number of court crossings
        """
        if len(coords) < 2:
            return 0

        # Check crossings of the net (middle of the court)
        net_y = self.court_length / 2
        crossings = 0

        for i in range(len(coords) - 1):
            y1, y2 = coords[i][1], coords[i + 1][1]
            
            # Check if ball crossed the net line
            if (y1 < net_y < y2) or (y1 > net_y > y2):
                crossings += 1

        return crossings