"""
Improved Coordinate Mapping Module for Tennis Court Tracking
"""
import cv2
import numpy as np
import pandas as pd


class ImprovedCoordinateMapper:
    def __init__(self):
        # Tennis court dimensions in meters (doubles court)
        self.court_width = 10.97  # Side to side
        self.court_length = 23.77  # Baseline to baseline
        
        # Define court coordinate system with origin at bottom-left baseline
        # This matches tennis conventions where baselines are at y=0 and y=23.77
        self.real_corners = np.array([
            [0, 0],                          # Bottom-left (BL)
            [self.court_width, 0],           # Bottom-right (BR) 
            [self.court_width, self.court_length],  # Top-right (TR)
            [0, self.court_length]           # Top-left (TL)
        ], dtype=np.float32)
        
        # Define key court lines for validation
        self.court_lines = {
            'baseline_near': 0,
            'service_line_near': 6.40,
            'net': self.court_length / 2,  # 11.885m
            'service_line_far': self.court_length - 6.40,
            'baseline_far': self.court_length,
            'singles_left': 1.37,
            'singles_right': self.court_width - 1.37,
            'center_line': self.court_width / 2
        }

    def order_corners_by_position(self, corners):
        """
        Order corners based on their position in the image
        Expected order: [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
        """
        if len(corners) != 4:
            raise ValueError("Exactly 4 corners required")
        
        corners = np.array(corners)
        
        # Sort by y-coordinate (bottom to top in image coordinates)
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        
        # Bottom two points (smaller y values in image)
        bottom_points = sorted_by_y[:2]
        bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
        bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
        
        # Top two points (larger y values in image)
        top_points = sorted_by_y[2:]
        top_left = top_points[np.argmin(top_points[:, 0])]
        top_right = top_points[np.argmax(top_points[:, 0])]
        
        return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

    def compute_homography(self, pixel_corners):
        """
        Compute homography matrix with improved corner ordering
        
        Args:
            pixel_corners: List of 4 corner points in pixel coordinates
            
        Returns:
            Homography matrix H
        """
        if len(pixel_corners) != 4:
            raise ValueError("Exactly 4 corner points required")
        
        # Order corners consistently
        ordered_pixel_corners = self.order_corners_by_position(pixel_corners)
        
        # Compute homography
        H, status = cv2.findHomography(
            ordered_pixel_corners, 
            self.real_corners, 
            cv2.RANSAC, 
            5.0
        )
        
        if H is None:
            raise ValueError("Could not compute homography matrix")
        
        return H, ordered_pixel_corners

    def validate_homography(self, H, pixel_corners):
        """
        Validate homography and provide detailed feedback
        """
        ordered_corners = self.order_corners_by_position(pixel_corners)
        
        # Transform pixel corners to court coordinates
        transformed = cv2.perspectiveTransform(
            ordered_corners.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        
        # Calculate errors
        errors = np.linalg.norm(transformed - self.real_corners, axis=1)
        
        print("=== Homography Validation ===")
        corner_names = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        
        for i, (name, pixel, expected, actual, error) in enumerate(
            zip(corner_names, ordered_corners, self.real_corners, transformed, errors)
        ):
            print(f"{name}: Pixel {pixel} -> Expected {expected} -> Actual {actual} (Error: {error:.3f}m)")
        
        max_error = np.max(errors)
        avg_error = np.mean(errors)
        
        print(f"\nError Statistics:")
        print(f"Max error: {max_error:.3f}m")
        print(f"Average error: {avg_error:.3f}m")
        
        is_valid = max_error < 1.0  # Allow up to 1m error
        print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid

    def map_pixels_to_court(self, df, H, x_col='X', y_col='Y'):
        """
        Map pixel coordinates to court coordinates
        """
        points = df[[x_col, y_col]].values.astype(np.float32)
        
        # Transform points using OpenCV (more robust)
        transformed = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['X_court'] = transformed[:, 0]
        result_df['Y_court'] = transformed[:, 1]
        
        return result_df

    def apply_court_constraints(self, df, buffer=2.0):
        """
        Apply realistic constraints to court coordinates
        
        Args:
            df: DataFrame with X_court and Y_court columns
            buffer: Buffer zone around court in meters
            
        Returns:
            DataFrame with constrained coordinates and outlier flags
        """
        result_df = df.copy()
        
        # Define valid ranges with buffer
        x_min, x_max = -buffer, self.court_width + buffer
        y_min, y_max = -buffer, self.court_length + buffer
        
        # Flag outliers
        result_df['is_outlier'] = (
            (result_df['X_court'] < x_min) | 
            (result_df['X_court'] > x_max) |
            (result_df['Y_court'] < y_min) | 
            (result_df['Y_court'] > y_max)
        )
        
        # Option 1: Clamp coordinates to valid range
        result_df['X_court_clamped'] = np.clip(
            result_df['X_court'], x_min, x_max
        )
        result_df['Y_court_clamped'] = np.clip(
            result_df['Y_court'], y_min, y_max
        )
        
        return result_df

    def add_court_zones(self, df):
        """
        Add semantic court zone information
        """
        result_df = df.copy()
        
        # Determine court zones
        conditions = [
            (result_df['Y_court'] <= self.court_lines['service_line_near']),
            (result_df['Y_court'] <= self.court_lines['net']),
            (result_df['Y_court'] <= self.court_lines['service_line_far']),
            (result_df['Y_court'] <= self.court_lines['baseline_far'])
        ]
        
        choices = ['Near Service Box', 'Near Court', 'Far Court', 'Far Service Box']
        
        result_df['court_zone'] = np.select(conditions, choices, default='Out of Bounds')
        
        # Add side information
        result_df['court_side'] = np.where(
            result_df['X_court'] < self.court_lines['center_line'], 
            'Left', 'Right'
        )
        
        return result_df

    def get_distance_to_lines(self, df):
        """
        Calculate distances to key court lines
        """
        result_df = df.copy()
        
        # Distance to baselines
        result_df['dist_to_near_baseline'] = np.abs(result_df['Y_court'] - self.court_lines['baseline_near'])
        result_df['dist_to_far_baseline'] = np.abs(result_df['Y_court'] - self.court_lines['baseline_far'])
        
        # Distance to net
        result_df['dist_to_net'] = np.abs(result_df['Y_court'] - self.court_lines['net'])
        
        # Distance to sidelines
        result_df['dist_to_left_sideline'] = np.abs(result_df['X_court'] - 0)
        result_df['dist_to_right_sideline'] = np.abs(result_df['X_court'] - self.court_width)
        
        # Distance to center line
        result_df['dist_to_center'] = np.abs(result_df['X_court'] - self.court_lines['center_line'])
        
        return result_df

    def debug_coordinates(self, df, sample_size=10):
        """
        Enhanced debugging with court-specific analysis
        """
        print("=== Tennis Court Coordinate Analysis ===")
        print(f"Court dimensions: {self.court_width}m × {self.court_length}m")
        print(f"Total data points: {len(df)}")
        
        # Basic statistics
        print(f"\nCoordinate Ranges:")
        print(f"X_court: {df['X_court'].min():.2f} to {df['X_court'].max():.2f}m")
        print(f"Y_court: {df['Y_court'].min():.2f} to {df['Y_court'].max():.2f}m")
        
        # Check for outliers
        x_outliers = df[(df['X_court'] < -2) | (df['X_court'] > self.court_width + 2)]
        y_outliers = df[(df['Y_court'] < -2) | (df['Y_court'] > self.court_length + 2)]
        
        print(f"\nOutlier Analysis:")
        print(f"X outliers: {len(x_outliers)} ({len(x_outliers)/len(df)*100:.1f}%)")
        print(f"Y outliers: {len(y_outliers)} ({len(y_outliers)/len(df)*100:.1f}%)")
        
        # Sample coordinates by class
        if 'Class' in df.columns:
            print(f"\nSample coordinates by class:")
            for class_name in df['Class'].unique():
                class_data = df[df['Class'] == class_name]
                print(f"\n{class_name} ({len(class_data)} points):")
                sample = class_data[['frame', 'X_court', 'Y_court']].head(5)
                print(sample.to_string(index=False))
        
        # Check coordinate distribution
        print(f"\nCoordinate Distribution:")
        print(f"Points in near court (Y < {self.court_lines['net']:.1f}): {len(df[df['Y_court'] < self.court_lines['net']])}")
        print(f"Points in far court (Y > {self.court_lines['net']:.1f}): {len(df[df['Y_court'] > self.court_lines['net']])}")
        print(f"Points on left side (X < {self.court_lines['center_line']:.1f}): {len(df[df['X_court'] < self.court_lines['center_line']])}")
        print(f"Points on right side (X > {self.court_lines['center_line']:.1f}): {len(df[df['X_court'] > self.court_lines['center_line']])}")

    def create_court_visualization_data(self, df):
        """
        Create data for court visualization
        """
        # Court boundaries
        court_bounds = {
            'outer_bounds': [
                [0, 0], [self.court_width, 0], 
                [self.court_width, self.court_length], [0, self.court_length], [0, 0]
            ],
            'service_boxes': [
                # Near service boxes
                [[0, 0], [self.court_width/2, 0], [self.court_width/2, self.court_lines['service_line_near']], [0, self.court_lines['service_line_near']], [0, 0]],
                [[self.court_width/2, 0], [self.court_width, 0], [self.court_width, self.court_lines['service_line_near']], [self.court_width/2, self.court_lines['service_line_near']], [self.court_width/2, 0]],
                # Far service boxes
                [[0, self.court_lines['service_line_far']], [self.court_width/2, self.court_lines['service_line_far']], [self.court_width/2, self.court_length], [0, self.court_length], [0, self.court_lines['service_line_far']]],
                [[self.court_width/2, self.court_lines['service_line_far']], [self.court_width, self.court_lines['service_line_far']], [self.court_width, self.court_length], [self.court_width/2, self.court_length], [self.court_width/2, self.court_lines['service_line_far']]]
            ],
            'net_line': [[0, self.court_lines['net']], [self.court_width, self.court_lines['net']]],
            'center_line': [[self.court_width/2, 0], [self.court_width/2, self.court_length]]
        }
        
        return court_bounds


# Example usage function
def process_tennis_data(csv_file, pixel_corners):
    """
    Complete pipeline for processing tennis tracking data
    
    Args:
        csv_file: Path to CSV file with tracking data
        pixel_corners: List of 4 corner points in pixel coordinates
                      Order: [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
    
    Returns:
        Processed DataFrame with court coordinates
    """
    # Initialize mapper
    mapper = ImprovedCoordinateMapper()
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Compute homography
    H, ordered_corners = mapper.compute_homography(pixel_corners)
    
    # Validate homography
    if not mapper.validate_homography(H, pixel_corners):
        print("WARNING: Homography validation failed!")
    
    # Map coordinates
    df_mapped = mapper.map_pixels_to_court(df, H)
    
    # Apply constraints and add analysis
    df_constrained = mapper.apply_court_constraints(df_mapped)
    df_with_zones = mapper.add_court_zones(df_constrained)
    df_final = mapper.get_distance_to_lines(df_with_zones)
    
    # Debug analysis
    mapper.debug_coordinates(df_final)
    
    return df_final, H, mapper