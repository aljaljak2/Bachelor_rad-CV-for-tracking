import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import urllib.request
import os
from collections import defaultdict

class TennisCourtTrackerV2:
    """
    Version 2 of Tennis Court Tracker - focuses on corner detection from the first frame only
    """
    
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

    def download_video(self, url, output_path="downloaded_video_v2.mp4"):
        """Download video from URL if it's a web URL"""
        try:
            if url.startswith(('http://', 'https://')):
                print(f"[DEBUG] Downloading video from: {url}")
                urllib.request.urlretrieve(url, output_path)
                print(f"[DEBUG] Video downloaded to: {output_path}")
                return output_path
            else:
                # Local file path
                if os.path.exists(url):
                    return url
                else:
                    raise FileNotFoundError(f"Local video file not found: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to download video: {e}")
            raise

    def detect_corners_from_first_frame(self, video_path, debug=False):
        """
        Detect court corners from the first frame of the video
        
        Args:
            video_path: Path to video file or URL
            debug: Whether to show debug information and save debug images
        
        Returns:
            tuple: (corners, first_frame) or (None, None) if detection fails
        """
        # Download video if it's a URL
        if video_path.startswith(('http://', 'https://')):
            video_path = self.download_video(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[DEBUG] Video info: {total_frames} frames, {fps:.2f} FPS")
        print(f"[DEBUG] Processing first frame for corner detection...")
        
        # Read the first frame
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            print("[ERROR] Could not read first frame")
            return None, None
        
        print(f"[DEBUG] First frame shape: {first_frame.shape}")
        
        # Detect corners in the first frame
        corners = self.detect_court_corners(first_frame, debug=debug)
        
        if len(corners) == 4:
            print(f"[DEBUG] Successfully detected 4 corners in first frame")
            
            # Save debug image if requested
            if debug:
                debug_frame = first_frame.copy()
                self._draw_corners_on_frame(debug_frame, corners)
                debug_path = 'first_frame_corners_debug.jpg'
                cv2.imwrite(debug_path, debug_frame)
                print(f"[DEBUG] Debug image saved to: {debug_path}")
        else:
            print(f"[ERROR] Found {len(corners)} corners instead of 4 in first frame")
            if debug:
                debug_frame = first_frame.copy()
                if corners:
                    self._draw_corners_on_frame(debug_frame, corners)
                debug_path = 'first_frame_corners_failed.jpg'
                cv2.imwrite(debug_path, debug_frame)
                print(f"[DEBUG] Failed detection image saved to: {debug_path}")
        
        # Clean up downloaded video if it was a URL
        if video_path.startswith('downloaded_video_v2'):
            try:
                os.remove(video_path)
                print(f"[DEBUG] Cleaned up downloaded video: {video_path}")
            except:
                pass
        
        return corners, first_frame

    def detect_court_corners(self, frame, debug=False):
        """Enhanced corner detection method for single frame"""
        if debug:
            print("[DEBUG] Starting corner detection...")
            print(f"[DEBUG] Frame dimensions: {frame.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with optimized parameters
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        if debug:
            cv2.imwrite('debug_edges.jpg', edges)
            print("[DEBUG] Edge detection completed, saved to debug_edges.jpg")
        
        # Hough Line Transform with adjusted parameters
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                               threshold=80, 
                               minLineLength=100, 
                               maxLineGap=50)
        
        if lines is None:
            print("[DEBUG] No lines detected")
            return []
        
        print(f"[DEBUG] Detected {len(lines)} lines")
        
        h, w = frame.shape[:2]
        
        # Find sideline segments
        sideline_segments = self._find_sideline_segments(lines, w, h)
        print(f"[DEBUG] Found {len(sideline_segments)} sideline segments")
        
        if len(sideline_segments) < 2:
            print("[DEBUG] Not enough sideline segments found")
            return []
        
        # Group sideline segments into left and right
        left_sideline, right_sideline = self._group_sideline_segments(sideline_segments, w, frame.copy())
        
        if not left_sideline or not right_sideline:
            print("[DEBUG] Could not identify both left and right sidelines")
            return []
        
        print(f"[DEBUG] Left sideline: {len(left_sideline)} segments")
        print(f"[DEBUG] Right sideline: {len(right_sideline)} segments")
        
        # Find corners by line tracing
        corners = self._find_corners_by_line_tracing(left_sideline, right_sideline, frame.copy(), h, w)
        
        print(f"[DEBUG] Found {len(corners)} corners")
        
        return corners

    def visualize_corners(self, frame, corners, output_path="first_frame_corners_visualization.jpg", show_image=True):
        """
        Visualize detected corners on the first frame
        
        Args:
            frame: The frame to draw on
            corners: List of corner coordinates
            output_path: Path to save the visualization
            show_image: Whether to display the image using cv2.imshow
        """
        if frame is None:
            print("[ERROR] No frame provided for visualization")
            return
        
        if not corners:
            print("[ERROR] No corners provided for visualization")
            return
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw the corners
        self._draw_corners_on_frame(vis_frame, corners)
        
        # Add title specific to first frame detection
        cv2.putText(vis_frame, 'FIRST FRAME CORNER DETECTION', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add corner coordinates as text
        if len(corners) == 4:
            ordered_corners = self.get_ordered_corners(corners)
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            for i, (corner, label) in enumerate(zip(ordered_corners, corner_labels)):
                coord_text = f'{label}: ({corner[0]:.1f}, {corner[1]:.1f})'
                cv2.putText(vis_frame, coord_text, (10, 150 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the visualization
        try:
            cv2.imwrite(output_path, vis_frame)
            print(f"[DEBUG] Corner visualization saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save visualization: {e}")
        
        # Display the image if requested
        if show_image:
            try:
                cv2.imshow('First Frame Tennis Court Corners', vis_frame)
                print("[DEBUG] Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[WARNING] Could not display image: {e}")
                print("Image saved to file instead.")

    # Import existing methods from the original class
    def _draw_corners_on_frame(self, frame, corners):
        """Draw detected corners on the frame with labels"""
        if len(corners) != 4:
            # If we don't have exactly 4 corners, just draw them as red circles
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red filled circle
                cv2.putText(frame, f'C{i+1}', (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Order corners and draw them with proper labels
            ordered_corners = self.get_ordered_corners(corners)
            corner_labels = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Cyan
            
            for i, (corner, label, color) in enumerate(zip(ordered_corners, corner_labels, colors)):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(frame, (x, y), 8, color, -1)  # Filled circle
                cv2.circle(frame, (x, y), 12, color, 2)   # Outer circle
                cv2.putText(frame, label, (x + 15, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            # Draw lines connecting the corners to show the court outline
            for i in range(4):
                pt1 = (int(ordered_corners[i][0]), int(ordered_corners[i][1]))
                pt2 = (int(ordered_corners[(i+1)%4][0]), int(ordered_corners[(i+1)%4][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)  # White lines
        
        # Add title
        cv2.putText(frame, 'Tennis Court Corners Detection V2', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add corner count info
        cv2.putText(frame, f'Corners found: {len(corners)}/4', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _find_sideline_segments(self, lines, img_width, img_height):
        """Find potential sideline segments from detected lines"""
        sideline_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # Calculate angle
            if abs(dx) < 1e-6:
                angle = 90.0
            else:
                angle = abs(np.arctan(dy / dx) * 180 / np.pi)
            
            # Calculate slope
            if abs(dx) < 1e-6:
                slope = float('inf')
            else:
                slope = dy / dx
            
            # Filter for sideline-like segments
            min_length = max(80, min(img_width, img_height) * 0.15)
            if length > min_length and 30 < angle < 85:
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
        return sideline_segments

    def _group_sideline_segments(self, segments, img_width, debug_img):
        """Group sideline segments into left and right sides"""
        if len(segments) < 2:
            return [], []
        
        # Use DBSCAN clustering to group segments by x-coordinate
        x_coords = np.array([[seg['center_x']] for seg in segments])
        clustering = DBSCAN(eps=img_width * 0.1, min_samples=1).fit(x_coords)
        labels = clustering.labels_
        
        # Group segments by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(segments[i])
        
        # Select the two largest clusters
        cluster_info = []
        for label, cluster_segments in clusters.items():
            if len(cluster_segments) >= 1:
                avg_x = np.mean([seg['center_x'] for seg in cluster_segments])
                total_length = sum([seg['length'] for seg in cluster_segments])
                cluster_info.append((label, avg_x, total_length, cluster_segments))
        
        # Sort by total length (descending)
        cluster_info.sort(key=lambda x: x[2], reverse=True)
        
        if len(cluster_info) < 2:
            return [], []
        
        # Select the two best clusters
        cluster1 = cluster_info[0]
        cluster2 = cluster_info[1]
        
        # Determine which is left and which is right
        if cluster1[1] < cluster2[1]:
            left_sideline = cluster1[3]
            right_sideline = cluster2[3]
        else:
            left_sideline = cluster2[3]
            right_sideline = cluster1[3]
        
        return left_sideline, right_sideline

    def _find_corners_by_line_tracing(self, left_sideline, right_sideline, debug_img, img_height, img_width):
        """Find corners by tracing along the sidelines"""
        corners = []
        
        # Find corners for left sideline
        if left_sideline:
            left_bottom, left_top = self._find_corner_pair(left_sideline, img_height, img_width, is_left=True)
            if left_bottom and left_top:
                corners.extend([left_top, left_bottom])
        
        # Find corners for right sideline
        if right_sideline:
            right_bottom, right_top = self._find_corner_pair(right_sideline, img_height, img_width, is_left=False)
            if right_bottom and right_top:
                corners.extend([right_top, right_bottom])
        
        return corners

    def _find_corner_pair(self, sideline_segments, img_height, img_width, is_left=True):
        """Find a pair of corners (top and bottom) for a sideline"""
        if not sideline_segments:
            return None, None
        
        # Find the bottommost point
        bottom_point = None
        bottom_y = -1
        bottom_segment = None
        
        for seg in sideline_segments:
            x1, y1, x2, y2 = seg['line']
            for point in [(x1, y1), (x2, y2)]:
                if point[1] > bottom_y:
                    bottom_y = point[1]
                    bottom_point = point
                    bottom_segment = seg
        
        if bottom_segment is None:
            return None, None
        
        # Find collinear segments
        collinear_segments = self._find_collinear_segments(bottom_segment, sideline_segments)
        
        # Find the topmost point from collinear segments
        top_point = self._find_topmost_point_from_collinear_segments(collinear_segments)
        
        return bottom_point, top_point

    def _find_collinear_segments(self, reference_segment, all_segments):
        """Find segments that are collinear with the reference segment"""
        collinear_segments = [reference_segment]
        
        ref_x1, ref_y1, ref_x2, ref_y2 = reference_segment['line']
        
        # Calculate line equation coefficients (ax + by + c = 0)
        a = ref_y2 - ref_y1
        b = ref_x1 - ref_x2
        c = (ref_x2 - ref_x1) * ref_y1 - (ref_y2 - ref_y1) * ref_x1
        
        # Normalize
        norm = np.sqrt(a*a + b*b)
        if norm > 0:
            a, b, c = a/norm, b/norm, c/norm
        
        # Check each segment for collinearity
        for seg in all_segments:
            if seg == reference_segment:
                continue
            
            x1, y1, x2, y2 = seg['line']
            
            # Calculate distance from both endpoints to the reference line
            dist1 = abs(a * x1 + b * y1 + c)
            dist2 = abs(a * x2 + b * y2 + c)
            
            tolerance = 10.0  # pixels
            if dist1 < tolerance and dist2 < tolerance:
                collinear_segments.append(seg)
        
        return collinear_segments

    def _find_topmost_point_from_collinear_segments(self, collinear_segments):
        """Find the topmost point from a list of collinear segments"""
        if not collinear_segments:
            return None
        
        topmost_point = None
        topmost_y = float('inf')
        
        for seg in collinear_segments:
            x1, y1, x2, y2 = seg['line']
            for point in [(x1, y1), (x2, y2)]:
                if point[1] < topmost_y:
                    topmost_y = point[1]
                    topmost_point = point
        
        return topmost_point

    def get_ordered_corners(self, corners):
        """Order corners as top-left, top-right, bottom-right, bottom-left"""
        if len(corners) != 4:
            return corners
        
        # Sort by y-coordinate (top to bottom)
        corners = sorted(corners, key=lambda p: p[1])
        
        # Split into top and bottom pairs
        top_points = sorted(corners[:2], key=lambda p: p[0])  # Sort by x
        bottom_points = sorted(corners[2:], key=lambda p: p[0])  # Sort by x
        
        # Return as: top-left, top-right, bottom-right, bottom-left
        return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]

    def compute_homography(self, pixel_corners):
        """Compute homography matrix from pixel corners to real court coordinates"""
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
        """Validate the computed homography matrix"""
        ordered_corners = self.get_ordered_corners(pixel_corners)
        pixel_pts = np.array(ordered_corners, dtype=np.float32)
        
        # Transform pixel points to real coordinates
        pixel_pts_h = np.concatenate([pixel_pts, np.ones((4, 1))], axis=1)
        mapped_pts = (H @ pixel_pts_h.T).T
        mapped_pts = mapped_pts / mapped_pts[:, 2:3]
        
        # Calculate errors
        real_pts = np.array(self.real_corners, dtype=np.float32)
        errors = np.linalg.norm(mapped_pts[:, :2] - real_pts, axis=1)
        
        print("Homography validation:")
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i, (name, pixel, real, mapped, error) in enumerate(zip(corner_names, ordered_corners, self.real_corners, mapped_pts[:, :2], errors)):
            print(f"{name}: Pixel {pixel} -> Expected {real} -> Mapped {mapped} (Error: {error:.3f}m)")
        
        return np.all(errors < 1.0)

    def map_pixels_to_court(self, df, H, x_col='X', y_col='Y'):
        """Map pixel coordinates to court coordinates using homography"""
        points = df[[x_col, y_col]].values.astype(np.float32)
        points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        
        # Apply homography transformation
        mapped = (H @ points_h.T).T
        mapped = mapped / mapped[:, 2:3]
        
        # Add court coordinates to dataframe
        df_mapped = df.copy()
        df_mapped['X_court'] = mapped[:, 0]
        df_mapped['Y_court'] = mapped[:, 1]
        
        return df_mapped

    def calculate_player_distances(self, df, id_col='ID', x_col='X_court', y_col='Y_court', class_col='Class', frame_col='frame'):
        """Calculate distances traveled by players"""
        players = df[df[class_col] == 'Player'].copy()
        
        if players.empty:
            return pd.DataFrame(columns=['ID', 'TotalDistance', 'FrameCount', 'FilteredFrames'])
        
        distances = []
        for pid, group in players.groupby(id_col):
            group = group.sort_values(frame_col)
            coords = group[[x_col, y_col]].values
            
            if len(coords) < 2:
                distances.append({'ID': pid, 'TotalDistance': 0.0, 'FrameCount': len(coords), 'FilteredFrames': 0})
                continue
            
            # Calculate frame-to-frame distances
            diffs = np.diff(coords, axis=0)
            frame_distances = np.linalg.norm(diffs, axis=1)
            
            # Filter out unrealistic movements
            max_speed_per_frame = 0.5  # meters per frame
            valid_mask = frame_distances <= max_speed_per_frame
            valid_distances = frame_distances[valid_mask]
            
            total_distance = np.sum(valid_distances)
            filtered_count = np.sum(~valid_mask)
            
            distances.append({
                'ID': pid,
                'TotalDistance': total_distance,
                'FrameCount': len(coords),
                'FilteredFrames': filtered_count
            })
        
        return pd.DataFrame(distances)

    def calculate_ball_distance_improved(self, df, x_col='X_court', y_col='Y_court', class_col='Class', frame_col='frame', fps=30.0):
        """Calculate ball distance with improved trajectory estimation"""
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

                    print(f"[INFO] Gap detected: {gap_frames} frames, estimated distance: {estimated_distance:.2f}m")

                trajectory_segments += 1

        # Calculate statistics
        valid_speeds = speeds[valid_speed_mask]
        average_speed = np.mean(valid_speeds) if len(valid_speeds) > 0 else 0.0
        max_speed = np.max(valid_speeds) if len(valid_speeds) > 0 else 0.0

        # Convert speeds to km/h for reporting
        average_speed_kmh = average_speed * 3.6
        max_speed_kmh = max_speed * 3.6

        print(f"[BALL ANALYSIS] Total distance: {total_distance:.2f}m")
        print(f"[BALL ANALYSIS] Average speed: {average_speed_kmh:.1f} km/h")
        print(f"[BALL ANALYSIS] Max speed: {max_speed_kmh:.1f} km/h")
        print(f"[BALL ANALYSIS] Detection gaps: {detection_gaps}")
        print(f"[BALL ANALYSIS] Valid trajectory segments: {trajectory_segments}")

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

    def debug_coordinates(self, df, x_col='X_court', y_col='Y_court', sample_size=10):
        """Debug coordinate transformation results"""
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
        
        if df[x_col].max() - df[x_col].min() > 50:
            print("WARNING: X coordinate range seems too large")
        if df[y_col].max() - df[y_col].min() > 50:
            print("WARNING: Y coordinate range seems too large")


def main_first_frame_processing_pipeline(video_path, data_csv_path, debug=False):
    """
    Main processing pipeline for first frame corner detection
    
    Args:
        video_path: Path to video file or URL
        data_csv_path: Path to CSV file with tracking data
        debug: Whether to show debug information and save debug images
    """
    tracker = TennisCourtTrackerV2()
    
    print("=== STEP 1: Detecting Court Corners from First Frame ===")
    try:
        corners, first_frame = tracker.detect_corners_from_first_frame(
            video_path, debug=debug
        )
        
        if corners is None or len(corners) != 4:
            print("Error: Could not detect 4 corners from first frame")
            if corners:
                print(f"Only found {len(corners)} corners")
            return None
        
        print(f"Successfully detected 4 corners from first frame")
        
        # Print corner coordinates for verification
        ordered_corners = tracker.get_ordered_corners(corners)
        corner_labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
        print("Detected corners:")
        for label, corner in zip(corner_labels, ordered_corners):
            print(f"  {label}: ({corner[0]:.1f}, {corner[1]:.1f})")
        
    except Exception as e:
        print(f"Error processing first frame: {e}")
        return None
    
    print("=== STEP 2: Visualizing Detected Corners ===")
    try:
        # Extract directory and filename from data_csv_path to match the Out folder structure
        import os
        csv_dir = os.path.dirname(data_csv_path)  # Gets './Out'
        csv_filename = os.path.basename(data_csv_path)  # Gets 'nadal_verdasco_init_df.csv'
        image_filename = csv_filename.replace('.csv', '_first_frame_corners.jpg')
        image_output_path = os.path.join(csv_dir, image_filename)
        
        tracker.visualize_corners(
            first_frame, 
            corners, 
            output_path=image_output_path,
            show_image=False  # Set to True if you want to display the image
        )
        
    except Exception as e:
        print(f"Warning: Could not visualize corners: {e}")
    
    print("=== STEP 3: Computing Homography with Detected Corners ===")
    try:
        H = tracker.compute_homography(corners)
        print("Homography matrix computed successfully from first frame corners")
        is_valid = tracker.validate_homography(H, corners)
        if not is_valid:
            print("WARNING: Homography validation failed - results may be inaccurate")
        else:
            print("Homography validation passed")
    except Exception as e:
        print(f"Error computing homography: {e}")
        return None
    
    print("=== STEP 4: Loading and Processing Tracking Data ===")
    try:
        df = pd.read_csv(data_csv_path)
        print(f"Loaded {len(df)} tracking points from CSV")
        
        # Map pixel coordinates to court coordinates
        df_mapped = tracker.map_pixels_to_court(df, H)
        print("Successfully mapped pixel coordinates to court coordinates")
        
        # Debug coordinate transformation
        tracker.debug_coordinates(df_mapped)
        
    except Exception as e:
        print(f"Error loading or processing tracking data: {e}")
        return None
    
    print("=== STEP 5: Calculating Player Distances ===")
    try:
        player_distances = tracker.calculate_player_distances(df_mapped)
        print("Player distance analysis:")
        if not player_distances.empty:
            for _, row in player_distances.iterrows():
                print(f"  Player {row['ID']}: {row['TotalDistance']:.2f}m total distance "
                      f"({row['FrameCount']} frames, {row['FilteredFrames']} filtered)")
        else:
            print("  No player data found")
        
    except Exception as e:
        print(f"Error calculating player distances: {e}")
        player_distances = pd.DataFrame()
    
    print("=== STEP 6: Calculating Ball Distance and Speed ===")
    try:
        ball_results = tracker.calculate_ball_distance_improved(df_mapped, fps=30.0)
        
        print("Ball trajectory analysis:")
        print(f"  Total distance: {ball_results['total_distance']:.2f}m")
        print(f"  Average speed: {ball_results['average_speed_kmh']:.1f} km/h")
        print(f"  Max speed: {ball_results['max_speed_kmh']:.1f} km/h")
        print(f"  Valid trajectory segments: {ball_results['trajectory_segments']}")
        print(f"  Detection gaps: {ball_results['detection_gaps']}")
        print(f"  Valid points: {ball_results['valid_points']}")
        
    except Exception as e:
        print(f"Error calculating ball distance: {e}")
        ball_results = {'total_distance': 0.0, 'average_speed_kmh': 0.0, 'max_speed_kmh': 0.0}
    
    print("=== STEP 7: Saving Results ===")
    try:
        # Save the mapped coordinates
        output_path = data_csv_path.replace('.csv', '_with_first_frame_court_coords.csv')
        df_mapped.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Mapped coordinates saved to: {output_path}")
        
        # Save corner detection results
        corner_results_path = data_csv_path.replace('.csv', '_first_frame_corners.csv')
        corner_df = pd.DataFrame({
            'corner_label': ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left'],
            'x_pixel': [c[0] for c in tracker.get_ordered_corners(corners)],
            'y_pixel': [c[1] for c in tracker.get_ordered_corners(corners)],
            'x_court': [c[0] for c in tracker.real_corners],
            'y_court': [c[1] for c in tracker.real_corners]
        })
        corner_df.to_csv(corner_results_path, index=False, encoding='utf-8')
        print(f"Corner detection results saved to: {corner_results_path}")
        
        # Save analysis summary
        summary_path = data_csv_path.replace('.csv', '_analysis_summary.csv')
        summary_data = []
        
        # Add player data
        for _, row in player_distances.iterrows():
            summary_data.append({
                'type': 'player',
                'id': row['ID'],
                'total_distance_m': row['TotalDistance'],
                'frame_count': row['FrameCount'],
                'filtered_frames': row['FilteredFrames'],
                'average_speed_kmh': 0,
                'max_speed_kmh': 0
            })
        
        # Add ball data
        summary_data.append({
            'type': 'ball',
            'id': 'ball',
            'total_distance_m': ball_results['total_distance'],
            'frame_count': ball_results['valid_points'],
            'filtered_frames': ball_results.get('filtered_unrealistic', 0),
            'average_speed_kmh': ball_results['average_speed_kmh'],
            'max_speed_kmh': ball_results['max_speed_kmh']
        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"Analysis summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Corner detection method: First frame only")
    print(f"Corners detected: {len(corners)}/4")
    print(f"Tracking points processed: {len(df_mapped)}")
    print(f"Players analyzed: {len(player_distances)}")
    print(f"Ball distance: {ball_results['total_distance']:.2f}m")
    
    return df_mapped, player_distances, ball_results, corners, first_frame
