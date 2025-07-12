import numpy as np
import pandas as pd
import cv2

def compute_homography(pixel_corners, real_corners):
    """
    Compute homography matrix from pixel to real-world court coordinates.
    
    IMPORTANT: Corner ordering must match between pixel_corners and real_corners!
    Recommended order: [top-left, top-right, bottom-right, bottom-left]
    
    pixel_corners: list of 4 (x, y) tuples in image (pixel) space
    real_corners: list of 4 (x, y) tuples in real court space (meters)
    Returns: 3x3 homography matrix
    """
    pixel_pts = np.array(pixel_corners, dtype=np.float32)
    real_pts = np.array(real_corners, dtype=np.float32)
    
    # Add validation
    if len(pixel_pts) != 4 or len(real_pts) != 4:
        raise ValueError("Exactly 4 corner points required")
    
    H, status = cv2.findHomography(pixel_pts, real_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("Could not compute homography matrix")
    
    return H

def validate_homography(H, pixel_corners, real_corners, tolerance=1.0):
    """
    Validate homography by checking if known corner points map correctly
    """
    pixel_pts = np.array(pixel_corners, dtype=np.float32)
    
    # Transform pixel corners using homography
    pixel_pts_h = np.concatenate([pixel_pts, np.ones((4, 1))], axis=1)
    mapped_pts = (H @ pixel_pts_h.T).T
    mapped_pts = mapped_pts / mapped_pts[:, 2:3]
    
    # Compare with expected real corners
    real_pts = np.array(real_corners, dtype=np.float32)
    errors = np.linalg.norm(mapped_pts[:, :2] - real_pts, axis=1)
    
    print("Homography validation:")
    for i, (pixel, real, mapped, error) in enumerate(zip(pixel_corners, real_corners, mapped_pts[:, :2], errors)):
        print(f"Corner {i}: Pixel {pixel} -> Expected {real} -> Mapped {mapped} (Error: {error:.3f})")
    
    return np.all(errors < tolerance)

def map_dataframe_pixels_to_court(df, H, x_col='X', y_col='Y'):
    """
    Map all (X, Y) pixel positions in dataframe to real court coordinates using homography H.
    Returns a new dataframe with X_mapped and Y_mapped columns.
    """
    points = df[[x_col, y_col]].values.astype(np.float32)
    
    # Add homogeneous coordinate
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    
    # Apply homography
    mapped = (H @ points_h.T).T
    
    # Convert from homogeneous coordinates
    mapped = mapped / mapped[:, 2:3]
    
    # Create new dataframe
    df_mapped = df.copy()
    df_mapped['X_mapped'] = mapped[:, 0]
    df_mapped['Y_mapped'] = mapped[:, 1]
    
    return df_mapped


def map_and_save_dataframe_pixels_to_court(df, H, out_path, x_col='X', y_col='Y'):
    """
    Map all (X, Y) pixel positions in dataframe to real court coordinates using homography H.
    Save the new dataframe (with X_mapped, Y_mapped) to the specified CSV file.
    Returns the new dataframe.
    """
    df_mapped = map_dataframe_pixels_to_court(df, H, x_col, y_col)
    df_mapped.to_csv(out_path, index=False)
    return df_mapped

# Example usage:
# pixel_corners = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
# real_corners = [(0, 0), (court_width, 0), (court_width, court_length), (0, court_length)]
# H = compute_homography(pixel_corners, real_corners)
# df = map_dataframe_pixels_to_court(df, H)
