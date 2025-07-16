"""
Coordinate Mapping Module for Tennis Court Tracking
"""
import cv2
import numpy as np
import pandas as pd


class CoordinateMapper:
    def __init__(self):
        # Tennis court dimensions in meters (doubles)
        self.court_width = 10.97
        self.court_length = 23.77
        self.real_corners = [
            (0, 0),
            (self.court_width, 0),
            (self.court_width, self.court_length),
            (0, self.court_length)
        ]

    def get_ordered_corners(self, corners):
        """Order corners in a consistent manner"""
        if len(corners) != 4:
            return corners
        corners = sorted(corners, key=lambda p: p[1])
        top_points = sorted(corners[:2], key=lambda p: p[0])
        bottom_points = sorted(corners[2:], key=lambda p: p[0])
        return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]

    def compute_homography(self, pixel_corners):
        """
        Compute homography matrix to map pixel coordinates to court coordinates
        
        Args:
            pixel_corners: List of 4 corner points in pixel coordinates
            
        Returns:
            Homography matrix H
        """
        if len(pixel_corners) != 4:
            raise ValueError("Exactly 4 corner points required")
        
        ordered_corners = self.get_ordered_corners(pixel_corners)
        pixel_pts = np.array(ordered_corners, dtype=np.float32)
        real_pts = np.array(self.real_corners, dtype=np.float32)
        
        H, status = cv2.findHomography(pixel_pts, real_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            raise ValueError("Could not compute homography matrix")
        
        return H

    def validate_homography(self, H, pixel_corners):
        """
        Validate the computed homography matrix
        
        Args:
            H: Homography matrix
            pixel_corners: Original pixel corners used to compute H
            
        Returns:
            bool: True if homography is valid
        """
        ordered_corners = self.get_ordered_corners(pixel_corners)
        pixel_pts = np.array(ordered_corners, dtype=np.float32)
        
        # Transform pixel corners to court coordinates
        pixel_pts_h = np.concatenate([pixel_pts, np.ones((4, 1))], axis=1)
        mapped_pts = (H @ pixel_pts_h.T).T
        mapped_pts = mapped_pts / mapped_pts[:, 2:3]
        
        # Compare with expected real corners
        real_pts = np.array(self.real_corners, dtype=np.float32)
        errors = np.linalg.norm(mapped_pts[:, :2] - real_pts, axis=1)
        
        print("Homography validation:")
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i, (name, pixel, real, mapped, error) in enumerate(zip(corner_names, ordered_corners, self.real_corners, mapped_pts[:, :2], errors)):
            print(f"{name}: Pixel {pixel} -> Expected {real} -> Mapped {mapped} (Error: {error:.3f}m)")
        
        return np.all(errors < 1.0)

    def map_pixels_to_court(self, df, H, x_col='X', y_col='Y'):
        """
        Map pixel coordinates to court coordinates using homography
        
        Args:
            df: DataFrame with pixel coordinates
            H: Homography matrix
            x_col: Column name for X pixel coordinates
            y_col: Column name for Y pixel coordinates
            
        Returns:
            DataFrame with added court coordinates
        """
        points = df[[x_col, y_col]].values.astype(np.float32)
        
        # Add homogeneous coordinate
        points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        
        # Apply homography transformation
        mapped = (H @ points_h.T).T
        mapped = mapped / mapped[:, 2:3]  # Normalize by homogeneous coordinate
        
        # Create new DataFrame with court coordinates
        df_mapped = df.copy()
        df_mapped['X_court'] = mapped[:, 0]
        df_mapped['Y_court'] = mapped[:, 1]
        
        return df_mapped

    def map_single_point(self, point, H):
        """
        Map a single pixel point to court coordinates
        
        Args:
            point: Tuple (x, y) in pixel coordinates
            H: Homography matrix
            
        Returns:
            Tuple (x_court, y_court) in court coordinates
        """
        point_h = np.array([point[0], point[1], 1.0], dtype=np.float32)
        mapped = H @ point_h
        mapped = mapped / mapped[2]  # Normalize
        
        return (mapped[0], mapped[1])

    def map_court_to_pixels(self, df, H, x_col='X_court', y_col='Y_court'):
        """
        Map court coordinates back to pixel coordinates (inverse transformation)
        
        Args:
            df: DataFrame with court coordinates
            H: Homography matrix
            x_col: Column name for X court coordinates
            y_col: Column name for Y court coordinates
            
        Returns:
            DataFrame with added pixel coordinates
        """
        # Compute inverse homography
        H_inv = np.linalg.inv(H)
        
        points = df[[x_col, y_col]].values.astype(np.float32)
        
        # Add homogeneous coordinate
        points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        
        # Apply inverse homography transformation
        mapped = (H_inv @ points_h.T).T
        mapped = mapped / mapped[:, 2:3]  # Normalize by homogeneous coordinate
        
        # Create new DataFrame with pixel coordinates
        df_mapped = df.copy()
        df_mapped['X_pixel'] = mapped[:, 0]
        df_mapped['Y_pixel'] = mapped[:, 1]
        
        return df_mapped

    def debug_coordinates(self, df, x_col='X_court', y_col='Y_court', sample_size=10):
        """
        Debug coordinate mapping by checking ranges and displaying samples
        
        Args:
            df: DataFrame with court coordinates
            x_col: Column name for X court coordinates
            y_col: Column name for Y court coordinates
            sample_size: Number of sample coordinates to display
        """
        print("Tennis Court Coordinate Statistics:")
        print(f"X_court range: {df[x_col].min():.3f} to {df[x_col].max():.3f} meters")
        print(f"Y_court range: {df[y_col].min():.3f} to {df[y_col].max():.3f} meters")
        print(f"Expected court dimensions: {self.court_width}m x {self.court_length}m")
        print(f"\nSample of court coordinates:")
        print(df[['frame', x_col, y_col, 'Class']].head(sample_size))
        
        # Check for points outside court boundaries
        x_outside = df[(df[x_col] < -2) | (df[x_col] > self.court_width + 2)]
        y_outside = df[(df[y_col] < -2) | (df[y_col] > self.court_length + 2)]
        
        if not x_outside.empty:
            print(f"\nWARNING: {len(x_outside)} points outside court X boundaries")
        if not y_outside.empty:
            print(f"WARNING: {len(y_outside)} points outside court Y boundaries")
        
        # Check for unrealistic coordinate ranges
        if df[x_col].max() - df[x_col].min() > 50:
            print("WARNING: X coordinate range seems too large")
        if df[y_col].max() - df[y_col].min() > 50:
            print("WARNING: Y coordinate range seems too large")