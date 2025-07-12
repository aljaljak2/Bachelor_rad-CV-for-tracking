import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression


def detect_court_corners_simple(frame, debug=False):
    """
    Improved approach: Find the four corners by tracing the same line segments
    that contain the bottom corners to find the corresponding top corners.
    """
    print("[DEBUG] Converting to grayscale...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("[DEBUG] Applying Gaussian blur...")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    print("[DEBUG] Performing Canny edge detection...")
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    print("[DEBUG] Running Hough Line Transform...")
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=50)

    debug_img = frame.copy()
    h, w = frame.shape[:2]
    
    if lines is None:
        print("[DEBUG] No lines detected.")
        return []

    print(f"[DEBUG] Number of lines detected: {len(lines)}")
    
    # Draw all detected lines in green
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Find potential sideline segments
    sideline_segments = find_sideline_segments(lines, w, h)
    
    if len(sideline_segments) < 2:
        print("[DEBUG] Not enough sideline segments found.")
        return []
    
    # Group segments into left and right sidelines
    left_sideline, right_sideline = group_sideline_segments(sideline_segments, w, debug_img)
    
    if not left_sideline or not right_sideline:
        print("[DEBUG] Could not identify both sidelines.")
        return []
    
    # Find corners by tracing the same lines that contain bottom corners
    corners = find_corners_by_line_tracing(left_sideline, right_sideline, debug_img, h, w)
    
    if debug:
        print("[DEBUG] Displaying image with detected sidelines and corners...")
        cv2.imshow('Improved Corner Detection', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"[DEBUG] Found {len(corners)} corners: {corners}")
    return corners


def find_sideline_segments(lines, img_width, img_height):
    """
    Find line segments that could be part of tennis court sidelines.
    """
    sideline_segments = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line properties
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle from horizontal
        if abs(dx) < 1e-6:
            angle = 90.0
        else:
            angle = abs(np.arctan(dy / dx) * 180 / np.pi)
        
        # Calculate slope for line extension
        if abs(dx) < 1e-6:
            slope = float('inf')
        else:
            slope = dy / dx
        
        # Filter for sideline candidates:
        # 1. Reasonably long lines
        # 2. Not perfectly horizontal (angle > 30 degrees)
        # 3. Not perfectly vertical (angle < 85 degrees) to account for perspective
        min_length = max(80, min(img_width, img_height) * 0.15)
        
        if length > min_length and 30 < angle < 85:
            # Calculate center point and average x-coordinate
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            sideline_segments.append({
                'line': (x1, y1, x2, y2),
                'length': length,
                'angle': angle,
                'slope': slope,
                'center_x': center_x,
                'center_y': center_y,
                'top_y': min(y1, y2),
                'bottom_y': max(y1, y2),
                'top_point': (x1, y1) if y1 < y2 else (x2, y2),
                'bottom_point': (x1, y1) if y1 > y2 else (x2, y2)
            })
    
    print(f"[DEBUG] Found {len(sideline_segments)} potential sideline segments")
    return sideline_segments


def group_sideline_segments(segments, img_width, debug_img):
    """
    Group segments into left and right sidelines using clustering.
    """
    if len(segments) < 2:
        return [], []
    
    # Extract x-coordinates for clustering
    x_coords = np.array([[seg['center_x']] for seg in segments])
    
    # Use DBSCAN to cluster segments by x-coordinate
    clustering = DBSCAN(eps=img_width * 0.1, min_samples=1).fit(x_coords)
    labels = clustering.labels_
    
    # Group segments by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(segments[i])
    
    # Find the two main clusters (left and right sidelines)
    cluster_info = []
    for label, cluster_segments in clusters.items():
        if len(cluster_segments) >= 1:  # At least 1 segment
            avg_x = np.mean([seg['center_x'] for seg in cluster_segments])
            total_length = sum([seg['length'] for seg in cluster_segments])
            cluster_info.append((label, avg_x, total_length, cluster_segments))
    
    # Sort by total length (descending) and take the two longest
    cluster_info.sort(key=lambda x: x[2], reverse=True)
    
    if len(cluster_info) < 2:
        print("[DEBUG] Could not find two distinct sideline clusters.")
        return [], []
    
    # Assign left and right based on x-coordinate
    cluster1 = cluster_info[0]
    cluster2 = cluster_info[1]
    
    if cluster1[1] < cluster2[1]:  # cluster1 is more to the left
        left_sideline = cluster1[3]
        right_sideline = cluster2[3]
    else:
        left_sideline = cluster2[3]
        right_sideline = cluster1[3]
    
    # Draw clustered segments
    colors = [(255, 0, 0), (0, 0, 255)]  # Blue for left, Red for right
    for i, sideline in enumerate([left_sideline, right_sideline]):
        color = colors[i]
        for seg in sideline:
            x1, y1, x2, y2 = seg['line']
            cv2.line(debug_img, (x1, y1), (x2, y2), color, 3)
    
    print(f"[DEBUG] Left sideline: {len(left_sideline)} segments, Right sideline: {len(right_sideline)} segments")
    return left_sideline, right_sideline


def find_corners_by_line_tracing(left_sideline, right_sideline, debug_img, img_height, img_width):
    """
    Find the four corners by:
    1. Finding the bottommost points for each sideline
    2. Identifying which line segment contains each bottom corner
    3. Extending that same line upward to find the corresponding top corner
    """
    corners = []
    
    # Process left sideline
    if left_sideline:
        left_bottom, left_top = find_corner_pair(left_sideline, img_height, img_width, is_left=True)
        
        if left_bottom and left_top:
            corners.extend([left_top, left_bottom])
            
            # Draw corners and labels
            cv2.circle(debug_img, left_top, 12, (0, 255, 0), -1)
            cv2.circle(debug_img, left_bottom, 12, (0, 255, 0), -1)
            cv2.putText(debug_img, "TL", (left_top[0]-20, left_top[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_img, "BL", (left_bottom[0]-20, left_bottom[1]+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw the extended line
            cv2.line(debug_img, left_top, left_bottom, (0, 255, 255), 2)
            
            print(f"[DEBUG] Left sideline corners: Top={left_top}, Bottom={left_bottom}")
    
    # Process right sideline
    if right_sideline:
        right_bottom, right_top = find_corner_pair(right_sideline, img_height, img_width, is_left=False)
        
        if right_bottom and right_top:
            corners.extend([right_top, right_bottom])
            
            # Draw corners and labels
            cv2.circle(debug_img, right_top, 12, (0, 255, 0), -1)
            cv2.circle(debug_img, right_bottom, 12, (0, 255, 0), -1)
            cv2.putText(debug_img, "TR", (right_top[0]+15, right_top[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_img, "BR", (right_bottom[0]+15, right_bottom[1]+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw the extended line
            cv2.line(debug_img, right_top, right_bottom, (0, 255, 255), 2)
            
            print(f"[DEBUG] Right sideline corners: Top={right_top}, Bottom={right_bottom}")
    
    return corners


def find_corner_pair(sideline_segments, img_height, img_width, is_left=True):
    """
    Find the bottom corner and corresponding top corner for a sideline.
    
    Algorithm:
    1. Find the segment that contains the bottommost point
    2. Find all segments that are collinear with this segment
    3. Among collinear segments, find the one with the topmost point
    """
    if not sideline_segments:
        return None, None
    
    # Find the segment that contains the bottommost point
    bottom_point = None
    bottom_y = -1
    bottom_segment = None
    
    for seg in sideline_segments:
        x1, y1, x2, y2 = seg['line']
        
        # Check both endpoints
        for point in [(x1, y1), (x2, y2)]:
            if point[1] > bottom_y:
                bottom_y = point[1]
                bottom_point = point
                bottom_segment = seg
    
    if bottom_segment is None:
        return None, None
    
    # Find all segments that are collinear with the bottom segment
    collinear_segments = find_collinear_segments(bottom_segment, sideline_segments)
    
    # Among collinear segments, find the topmost point
    top_point = find_topmost_point_from_collinear_segments(collinear_segments)
    
    return bottom_point, top_point


def find_collinear_segments(reference_segment, all_segments):
    """
    Find all segments that are collinear (on the same line) with the reference segment.
    """
    collinear_segments = [reference_segment]
    
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_segment['line']
    ref_slope = reference_segment['slope']
    
    # Calculate reference line parameters
    # Line equation: ax + by + c = 0
    # For line through (x1,y1) and (x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
    a = ref_y2 - ref_y1
    b = ref_x1 - ref_x2
    c = (ref_x2 - ref_x1) * ref_y1 - (ref_y2 - ref_y1) * ref_x1
    
    # Normalize to avoid floating point issues
    norm = np.sqrt(a*a + b*b)
    if norm > 0:
        a, b, c = a/norm, b/norm, c/norm
    
    for seg in all_segments:
        if seg == reference_segment:
            continue
            
        x1, y1, x2, y2 = seg['line']
        
        # Check if both endpoints of this segment lie on the reference line
        # Distance from point (x,y) to line ax + by + c = 0 is |ax + by + c|
        dist1 = abs(a * x1 + b * y1 + c)
        dist2 = abs(a * x2 + b * y2 + c)
        
        # If both endpoints are close to the line (within tolerance), consider it collinear
        tolerance = 10.0  # pixels
        if dist1 < tolerance and dist2 < tolerance:
            collinear_segments.append(seg)
    
    print(f"[DEBUG] Found {len(collinear_segments)} collinear segments")
    return collinear_segments


def find_topmost_point_from_collinear_segments(collinear_segments):
    """
    Find the topmost point among all collinear segments.
    """
    if not collinear_segments:
        return None
    
    topmost_point = None
    topmost_y = float('inf')
    
    for seg in collinear_segments:
        x1, y1, x2, y2 = seg['line']
        
        # Check both endpoints
        for point in [(x1, y1), (x2, y2)]:
            if point[1] < topmost_y:  # Lower y value means higher on screen
                topmost_y = point[1]
                topmost_point = point
    
    return topmost_point


def extend_line_to_find_top_corner(segment, bottom_point, img_height, img_width):
    """
    This function is kept for compatibility but not used in the new approach.
    """
    x1, y1, x2, y2 = segment['line']
    
    # Simply return the topmost point of this specific segment
    if y1 < y2:
        return (x1, y1)
    else:
        return (x2, y2)


def get_ordered_corners(corners):
    """
    Return corners in a consistent order: [top-left, top-right, bottom-right, bottom-left]
    """
    if len(corners) != 4:
        return corners
    
    # Sort by y-coordinate to separate top and bottom
    corners = sorted(corners, key=lambda p: p[1])
    
    # Top two points
    top_points = corners[:2]
    bottom_points = corners[2:]
    
    # Sort by x-coordinate
    top_points = sorted(top_points, key=lambda p: p[0])
    bottom_points = sorted(bottom_points, key=lambda p: p[0])
    
    # Return in order: top-left, top-right, bottom-right, bottom-left
    return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]


# Example usage function
def process_frame(frame):
    """
    Process a single frame to detect tennis court corners.
    """
    corners = detect_court_corners_simple(frame, debug=True)
    
    if len(corners) == 4:
        ordered_corners = get_ordered_corners(corners)
        print(f"[RESULT] Court corners found:")
        print(f"  Top-left: {ordered_corners[0]}")
        print(f"  Top-right: {ordered_corners[1]}")
        print(f"  Bottom-right: {ordered_corners[2]}")
        print(f"  Bottom-left: {ordered_corners[3]}")
        return ordered_corners
    else:
        print(f"[RESULT] Could not find all 4 corners. Found {len(corners)} corners.")
        return corners


# For testing with an image file
def test_with_image(image_path):
    """
    Test the corner detection with an image file.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    corners = process_frame(frame)
    return corners