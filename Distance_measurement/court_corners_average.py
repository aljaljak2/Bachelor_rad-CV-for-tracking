import numpy as np
from Detect_and_Track.court_detection import detect_court_corners_simple as detect_court_corners

def get_average_court_corners(frames, every_n=30, debug=False):
    """
    Detect court corners every 'every_n' frames, then return the average positions for each corner.
    frames: list or iterable of frames (images)
    every_n: interval for detection (default 30)
    Returns: list of 4 (x, y) tuples (averaged corners)
    """
    all_corners = []
    for i, frame in enumerate(frames):
        if i % every_n == 0:
            corners = detect_court_corners(frame, debug=debug)
            if corners and len(corners) == 4:
                all_corners.append(corners)
    if not all_corners:
        raise ValueError("No valid corners detected in any frame.")
    # Convert to numpy array: shape (num_samples, 4, 2)
    all_corners_np = np.array(all_corners, dtype=np.float32)
    avg_corners = np.mean(all_corners_np, axis=0)
    avg_corners = [tuple(map(float, pt)) for pt in avg_corners]
    return avg_corners

# Example usage:
# avg_corners = get_average_court_corners(frames, every_n=30)
