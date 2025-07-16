import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import urllib.request
import os
from collections import defaultdict
from .homography_manager import HomographyManager

class TennisCourtTracker:
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

    def download_video(self, url, output_path="downloaded_video.mp4"):
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

    def detect_corners_from_video(self, video_path, sample_interval=1.0, max_frames=30, debug=False):
        """
        Detect court corners from multiple frames in a video
        
        Args:
            video_path: Path to video file or URL
            sample_interval: Interval in seconds between frame samples
            max_frames: Maximum number of frames to process
            debug: Whether to show debug information
        
        Returns:
            tuple: (average_corners, all_frame_corners, frame_info, representative_frame)
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
        duration = total_frames / fps
        
        print(f"[DEBUG] Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Calculate frame sampling
        frame_interval = int(fps * sample_interval)
        frames_to_process = min(max_frames, int(duration / sample_interval))
        
        print(f"[DEBUG] Will process {frames_to_process} frames, sampling every {frame_interval} frames ({sample_interval}s)")
        
        all_frame_corners = []
        frame_info = []
        processed_count = 0
        representative_frame = None
        best_frame_score = 0
        
        for i in range(frames_to_process):
            frame_number = int(i * frame_interval)  # Ensure it's a Python int
            timestamp = frame_number / fps
            
            # Seek to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                print(f"[WARNING] Could not read frame {frame_number}")
                continue
            
            print(f"[DEBUG] Processing frame {frame_number} at {timestamp:.2f}s")
            
            # Detect corners in this frame
            corners = self.detect_court_corners(frame, debug=False)
            
            frame_data = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'corners_found': len(corners),
                'corners': corners
            }
            
            if len(corners) == 4:
                all_frame_corners.append(corners)
                frame_data['valid'] = True
                processed_count += 1
                print(f"[DEBUG] Frame {frame_number}: Found 4 corners [OK]")
                
                # Select representative frame (middle of the sequence with good corners)
                if processed_count == max(1, len(all_frame_corners) // 2):
                    representative_frame = frame.copy()
                    print(f"[DEBUG] Selected frame {frame_number} as representative frame")
                    
            else:
                frame_data['valid'] = False
                print(f"[DEBUG] Frame {frame_number}: Found {len(corners)} corners [FAILED]")
            
            frame_info.append(frame_data)
            
            # Save debug image for first few frames if debug is enabled
            if debug and i < 5:
                debug_frame = frame.copy()
                if corners:
                    self._draw_corners_on_frame(debug_frame, corners)
                output_path = f'debug_frame_{frame_number}.jpg'
                cv2.imwrite(output_path, debug_frame)
                print(f"[DEBUG] Saved debug frame to: {output_path}")
        
        cap.release()
        
        print(f"[DEBUG] Successfully processed {processed_count}/{frames_to_process} frames")
        
        if processed_count == 0:
            print("[ERROR] No valid corners found in any frame")
            return None, all_frame_corners, frame_info, None
        
        # Calculate average corners
        average_corners = self._calculate_average_corners(all_frame_corners)
        
        # If no representative frame was selected, use the first valid frame
        if representative_frame is None and all_frame_corners:
            print("[DEBUG] No representative frame selected, using first valid frame")
            # Get the first valid frame
            cap = cv2.VideoCapture(video_path)
            for info in frame_info:
                if info['valid']:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, info['frame_number'])
                    ret, representative_frame = cap.read()
                    if ret:
                        break
            cap.release()
        
        # Clean up downloaded video if it was a URL
        if video_path.startswith('downloaded_video'):
            try:
                os.remove(video_path)
                print(f"[DEBUG] Cleaned up downloaded video: {video_path}")
            except:
                pass
        
        return average_corners, all_frame_corners, frame_info, representative_frame

    def _calculate_average_corners(self, all_frame_corners):
        """Calculate average corner positions from multiple frames"""
        if not all_frame_corners:
            return []
        
        print(f"[DEBUG] Calculating average corners from {len(all_frame_corners)} valid frames")
        
        # Group corners by their likely position (using simple clustering)
        corner_groups = [[] for _ in range(4)]
        
        for frame_corners in all_frame_corners:
            # Order corners consistently for this frame
            ordered_corners = self.get_ordered_corners(frame_corners)
            
            # Add to respective groups
            for i, corner in enumerate(ordered_corners):
                corner_groups[i].append(corner)
        
        # Calculate average for each corner group
        average_corners = []
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        
        for i, (group, name) in enumerate(zip(corner_groups, corner_names)):
            if group:
                # Convert to regular Python floats to avoid numpy compatibility issues
                x_coords = [float(corner[0]) for corner in group]
                y_coords = [float(corner[1]) for corner in group]
                
                # Use numpy for statistics instead of statistics module
                avg_x = np.mean(x_coords)
                avg_y = np.mean(y_coords)
                std_x = np.std(x_coords) if len(x_coords) > 1 else 0
                std_y = np.std(y_coords) if len(y_coords) > 1 else 0
                
                average_corners.append((avg_x, avg_y))
                print(f"[DEBUG] {name}: ({avg_x:.1f}, {avg_y:.1f}) +/- ({std_x:.1f}, {std_y:.1f})")
        
        return average_corners

    def visualize_average_corners(self, frame, average_corners, output_path="average_corners_visualization.jpg", show_image=True):
        """
        Visualize average corners on a representative frame
        
        Args:
            frame: The frame to draw on
            average_corners: List of average corner coordinates
            output_path: Path to save the visualization
            show_image: Whether to display the image using cv2.imshow
        """
        if frame is None:
            print("[ERROR] No frame provided for visualization")
            return
        
        if not average_corners or len(average_corners) != 4:
            print(f"[ERROR] Invalid average corners: expected 4, got {len(average_corners) if average_corners else 0}")
            return
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw the average corners
        self._draw_corners_on_frame(vis_frame, average_corners)
        
        # Add additional information
        cv2.putText(vis_frame, 'AVERAGE CORNERS FROM MULTIPLE FRAMES', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add corner coordinates as text
        ordered_corners = self.get_ordered_corners(average_corners)
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        for i, (corner, label) in enumerate(zip(ordered_corners, corner_labels)):
            coord_text = f'{label}: ({corner[0]:.1f}, {corner[1]:.1f})'
            cv2.putText(vis_frame, coord_text, (10, 150 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save the visualization
        try:
            cv2.imwrite(output_path, vis_frame)
            print(f"[DEBUG] Average corners visualization saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save visualization: {e}")
        
        # Display the image if requested
        if show_image:
            try:
                cv2.imshow('Average Tennis Court Corners', vis_frame)
                print("[DEBUG] Press any key to close the image window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[WARNING] Could not display image: {e}")
                print("Image saved to file instead.")

    def detect_court_corners(self, frame, debug=False):
        """Original corner detection method for single frame"""
        if debug:
            print("[DEBUG] Converting to grayscale...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=50)
        
        h, w = frame.shape[:2]
        if lines is None:
            return []
        
        sideline_segments = self._find_sideline_segments(lines, w, h)
        if len(sideline_segments) < 2:
            return []
        
        left_sideline, right_sideline = self._group_sideline_segments(sideline_segments, w, frame.copy())
        if not left_sideline or not right_sideline:
            return []
        
        corners = self._find_corners_by_line_tracing(left_sideline, right_sideline, frame.copy(), h, w)
        
        return corners

    def _detect_significant_corner_change(self, current_corners, previous_corners, threshold=15.0):
        """
        Detect if corner positions have changed significantly

        Args:
            current_corners: Current frame corners
            previous_corners: Previous frame corners  
            threshold: Pixel distance threshold for significant change

        Returns:
            bool: True if significant change detected
        """
        if not current_corners or not previous_corners or len(current_corners) != 4 or len(previous_corners) != 4:
            return True

        current_ordered = self.get_ordered_corners(current_corners)
        previous_ordered = self.get_ordered_corners(previous_corners)

        for curr, prev in zip(current_ordered, previous_ordered):
            distance = np.linalg.norm(np.array(curr) - np.array(prev))
            if distance > threshold:
                return True

        return False

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
        cv2.putText(frame, 'Tennis Court Corners Detection', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add corner count info
        cv2.putText(frame, f'Corners found: {len(corners)}/4', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _find_sideline_segments(self, lines, img_width, img_height):
        sideline_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if abs(dx) < 1e-6:
                angle = 90.0
            else:
                angle = abs(np.arctan(dy / dx) * 180 / np.pi)
            if abs(dx) < 1e-6:
                slope = float('inf')
            else:
                slope = dy / dx
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
        if len(segments) < 2:
            return [], []
        x_coords = np.array([[seg['center_x']] for seg in segments])
        clustering = DBSCAN(eps=img_width * 0.1, min_samples=1).fit(x_coords)
        labels = clustering.labels_
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(segments[i])
        cluster_info = []
        for label, cluster_segments in clusters.items():
            if len(cluster_segments) >= 1:
                avg_x = np.mean([seg['center_x'] for seg in cluster_segments])
                total_length = sum([seg['length'] for seg in cluster_segments])
                cluster_info.append((label, avg_x, total_length, cluster_segments))
        cluster_info.sort(key=lambda x: x[2], reverse=True)
        if len(cluster_info) < 2:
            return [], []
        cluster1 = cluster_info[0]
        cluster2 = cluster_info[1]
        if cluster1[1] < cluster2[1]:
            left_sideline = cluster1[3]
            right_sideline = cluster2[3]
        else:
            left_sideline = cluster2[3]
            right_sideline = cluster1[3]
        return left_sideline, right_sideline

    def _find_corners_by_line_tracing(self, left_sideline, right_sideline, debug_img, img_height, img_width):
        corners = []
        if left_sideline:
            left_bottom, left_top = self._find_corner_pair(left_sideline, img_height, img_width, is_left=True)
            if left_bottom and left_top:
                corners.extend([left_top, left_bottom])
        if right_sideline:
            right_bottom, right_top = self._find_corner_pair(right_sideline, img_height, img_width, is_left=False)
            if right_bottom and right_top:
                corners.extend([right_top, right_bottom])
        return corners

    def _find_corner_pair(self, sideline_segments, img_height, img_width, is_left=True):
        if not sideline_segments:
            return None, None
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
        collinear_segments = self._find_collinear_segments(bottom_segment, sideline_segments)
        top_point = self._find_topmost_point_from_collinear_segments(collinear_segments)
        return bottom_point, top_point

    def _find_collinear_segments(self, reference_segment, all_segments):
        collinear_segments = [reference_segment]
        ref_x1, ref_y1, ref_x2, ref_y2 = reference_segment['line']
        a = ref_y2 - ref_y1
        b = ref_x1 - ref_x2
        c = (ref_x2 - ref_x1) * ref_y1 - (ref_y2 - ref_y1) * ref_x1
        norm = np.sqrt(a*a + b*b)
        if norm > 0:
            a, b, c = a/norm, b/norm, c/norm
        for seg in all_segments:
            if seg == reference_segment:
                continue
            x1, y1, x2, y2 = seg['line']
            dist1 = abs(a * x1 + b * y1 + c)
            dist2 = abs(a * x2 + b * y2 + c)
            tolerance = 10.0
            if dist1 < tolerance and dist2 < tolerance:
                collinear_segments.append(seg)
        return collinear_segments

    def _find_topmost_point_from_collinear_segments(self, collinear_segments):
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
        if len(corners) != 4:
            return corners
        corners = sorted(corners, key=lambda p: p[1])
        top_points = sorted(corners[:2], key=lambda p: p[0])
        bottom_points = sorted(corners[2:], key=lambda p: p[0])
        return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]

    def compute_homography(self, pixel_corners):
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
        ordered_corners = self.get_ordered_corners(pixel_corners)
        pixel_pts = np.array(ordered_corners, dtype=np.float32)
        pixel_pts_h = np.concatenate([pixel_pts, np.ones((4, 1))], axis=1)
        mapped_pts = (H @ pixel_pts_h.T).T
        mapped_pts = mapped_pts / mapped_pts[:, 2:3]
        real_pts = np.array(self.real_corners, dtype=np.float32)
        errors = np.linalg.norm(mapped_pts[:, :2] - real_pts, axis=1)
        print("Homography validation:")
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i, (name, pixel, real, mapped, error) in enumerate(zip(corner_names, ordered_corners, self.real_corners, mapped_pts[:, :2], errors)):
            print(f"{name}: Pixel {pixel} -> Expected {real} -> Mapped {mapped} (Error: {error:.3f}m)")
        return np.all(errors < 1.0)

    def map_pixels_to_court(self, df, H, x_col='X', y_col='Y'):
        points = df[[x_col, y_col]].values.astype(np.float32)
        points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        mapped = (H @ points_h.T).T
        mapped = mapped / mapped[:, 2:3]
        df_mapped = df.copy()
        df_mapped['X_court'] = mapped[:, 0]
        df_mapped['Y_court'] = mapped[:, 1]
        return df_mapped
    '''
    def calculate_player_distances(self, df, id_col='ID', x_col='X_court', y_col='Y_court', class_col='Class', frame_col='frame'):
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
            diffs = np.diff(coords, axis=0)
            frame_distances = np.linalg.norm(diffs, axis=1)
            max_speed_per_frame = 0.5
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
    '''

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

    def _calculate_movement_segments(self, frames):
        """
        Calculate the number of continuous movement segments
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

    def validate_player_measurements(self, player_results, rally_duration_seconds=None):
        """
        Validate player measurements against realistic tennis movement patterns
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
                    print(" [W] Distance per second seems unrealistic")
        
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

    def analyze_player_movement_patterns(self, df, id_col='ID', x_col='X_court', y_col='Y_court', 
                                   class_col='Class', frame_col='frame'):
        """
        Analyze detailed movement patterns for each player
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

    def _get_dominant_direction(self, angles):
        """
        Determine the dominant movement direction
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
            from collections import Counter
            return Counter(directions).most_common(1)[0][0]
        return 'Unknown'

    def _create_position_heatmap(self, coords):
        """
        Create a simplified position heatmap (JSON-safe)
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

    def calculate_ball_distance_improved(self, df, x_col='X_court', y_col='Y_court', class_col='Class', frame_col='frame', fps=30.0):
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

                    print(f"INFO Gap detected {gap_frames} frames, estimated distance {estimated_distance:.2f}m")

                trajectory_segments += 1

        # Calculate statistics
        valid_speeds = speeds[valid_speed_mask]
        average_speed = np.mean(valid_speeds) if len(valid_speeds) > 0 else 0.0
        max_speed = np.max(valid_speeds) if len(valid_speeds) > 0 else 0.0

        # Convert speeds to km/h for reporting
        average_speed_kmh = average_speed * 3.6
        max_speed_kmh = max_speed * 3.6

        print(f"BALL ANALYSIS Total distance {total_distance:.2f}m")
        print(f"BALL ANALYSIS Average speed {average_speed_kmh:.1f} km/h")
        print(f"BALL ANALYSIS Max speed {max_speed_kmh:.1f} km/h")
        print(f"BALL ANALYSIS Detection gaps {detection_gaps}")
        print(f"BALL ANALYSIS Valid trajectory segments {trajectory_segments}")

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

                    print(f"TRAJECTORY Gap of {gap_size} frames, estimated distance {estimated_distance:.2f}m")
                else:
                    # Large gap - use direct distance
                    distance = np.linalg.norm(coords[i + 1] - coords[i])
                    total_distance += distance

                    print(f"LARGE GAP {gap_size} frames, using direct distance {distance:.2f}m")

                i += 1

        return {
            'total_distance': total_distance,
            'method': 'trajectory_estimation',
            'gaps_estimated': True
        }

    def validate_ball_measurements(self, ball_results, rally_duration_seconds=None):
        """
        Validate ball measurements against realistic tennis expectations
        """
        print("\n=== BALL MEASUREMENT VALIDATION ===")

        total_distance = ball_results['total_distance']
        avg_speed = ball_results.get('average_speed_kmh', 0)
        max_speed = ball_results.get('max_speed_kmh', 0)

        # Tennis ball physics validation
        print(f"Total ball distance {total_distance:.2f}m")

        if rally_duration_seconds:
            rally_average_speed = (total_distance / rally_duration_seconds) * 3.6
            print(f"Rally average speed {rally_average_speed:.1f} km/h")

            # Realistic ranges for tennis
            if 10 <= rally_average_speed <= 150:
                print("Rally average speed is realistic")
            else:
                print("Rally average speed seems unrealistic")

        if avg_speed > 0:
            print(f"Ball average speed {avg_speed:.1f} km/h")
            if 20 <= avg_speed <= 200:
                print("Average speed is realistic")
            else:
                print("Average speed seems unrealistic")

        if max_speed > 0:
            print(f"Ball max speed {max_speed:.1f} km/h")
            if 50 <= max_speed <= 250:
                print("Max speed is realistic")
            else:
                print("Max speed seems unrealistic")

        # Distance validation
        if total_distance > 0:
            if 50 <= total_distance <= 2000:  # Typical rally distances
                print("Total distance is realistic")
            else:
                print("Total distance seems unrealistic")

        return ball_results


    def debug_coordinates(self, df, x_col='X_court', y_col='Y_court', sample_size=10):
        print("Tennis Court Coordinate Statistics:")
        print(f"X_court range: {df[x_col].min():.3f} to {df[x_col].max():.3f} meters")
        print(f"Y_court range: {df[y_col].min():.3f} to {df[y_col].max():.3f} meters")
        print(f"Expected court dimensions: {self.court_width}m x {self.court_length}m")
        print(f"\nSample of court coordinates:")
        print(df[['frame', x_col, y_col, 'Class']].head(sample_size))
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


def main_video_processing_pipeline(video_path, data_csv_path, sample_interval=1.0, max_frames=30):
    """
    Main processing pipeline for video-based corner detection
    
    Args:
        video_path: Path to video file or URL
        data_csv_path: Path to CSV file with tracking data
        sample_interval: Interval in seconds between frame samples
        max_frames: Maximum number of frames to process
    """
    tracker = TennisCourtTracker()
    
    print("=== STEP 1: Detecting Court Corners from Video ===")
    try:
        # Fixed: Now unpacking 4 values instead of 3
        average_corners, all_frame_corners, frame_info, representative_frame = tracker.detect_corners_from_video(
            video_path, sample_interval=sample_interval, max_frames=max_frames, debug=True
        )
        
        if average_corners is None:
            print("Error: Could not detect corners from video")
            return
        
        print(f"Successfully calculated average corners from {len(all_frame_corners)} frames")
        
        # Print summary statistics
        valid_frames = sum(1 for info in frame_info if info['valid'])
        print(f"Frame processing summary: {valid_frames}/{len(frame_info)} frames had valid corners")
        
        # Optional: Visualize the average corners on the representative frame
        if representative_frame is not None:
            print("=== STEP 1.5: Visualizing Average Corners ===")
            # Extract directory and filename from data_csv_path to match the Out folder structure
            import os
            csv_dir = os.path.dirname(data_csv_path)  # Gets './Out'
            csv_filename = os.path.basename(data_csv_path)  # Gets 'nadal_verdasco_init_df.csv'
            image_filename = csv_filename.replace('.csv', '_average_corners.jpg')
            image_output_path = os.path.join(csv_dir, image_filename)
            
            tracker.visualize_average_corners(
                representative_frame, 
                average_corners, 
                output_path=image_output_path,
                show_image=False  # Set to True if you want to display the image
            )
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return
    
    print("=== STEP 2: Computing Homography with Average Corners ===")
    try:
        H = tracker.compute_homography(average_corners)
        print("Homography matrix computed successfully using average corners")
        is_valid = tracker.validate_homography(H, average_corners)
        if not is_valid:
            print("WARNING: Homography validation failed")
    except Exception as e:
        print(f"Error computing homography: {e}")
        return
    
    print("=== STEP 3: Loading and Processing Tracking Data ===")
    df = pd.read_csv(data_csv_path)
    print(f"Loaded {len(df)} tracking points")
    df_mapped = tracker.map_pixels_to_court(df, H)
    tracker.debug_coordinates(df_mapped)
    
    print("=== STEP 4: Calculating Distances ===")
    player_distances = tracker.calculate_player_distances_improved(df_mapped)
    print("Player distances:")
    print(player_distances)
    
    ball_results = tracker.calculate_ball_distance_improved(df_mapped, fps=30.0)
    tracker.validate_ball_measurements(ball_results)
    #print(f"\nBall total distance: {ball_distance:.2f} meters")
    
    # Save results
    output_path = data_csv_path.replace('.csv', '_with_video_court_coords.csv')
    df_mapped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_path}")
    
    # Save corner analysis results
    corner_analysis_path = data_csv_path.replace('.csv', '_corner_analysis.csv')
    corner_df = pd.DataFrame(frame_info)
    corner_df.to_csv(corner_analysis_path, index=False, encoding='utf-8')
    print(f"Corner analysis saved to: {corner_analysis_path}")
    
    return df_mapped, player_distances, ball_results, average_corners, frame_info

def main_video_processing_pipeline_dynamic(video_path, data_csv_path, sample_interval=1.0, max_frames=30):
    """
    Main processing pipeline with dynamic homography updates
    """
    tracker = TennisCourtTracker()

    print("=== STEP 1: Processing Video with Dynamic Homography ===")
    try:
        # Get frame analysis (keep existing corner detection)
        average_corners, all_frame_corners, frame_info, representative_frame = tracker.detect_corners_from_video(
            video_path, sample_interval=sample_interval, max_frames=max_frames, debug=True
        )

        if average_corners is None:
            print("Error: Could not detect corners from video")
            return

    except Exception as e:
        print(f"Error processing video: {e}")
        return

    print("=== STEP 2: Setting up Dynamic Homography Processing ===")
    homography_manager = HomographyManager(tracker)

    # Initialize with first valid frame
    for info in frame_info:
        if info['valid']:
            initial_homography = homography_manager.update_homography_if_needed(
                info['corners'], info['frame_number']
            )
            break

    print("=== STEP 3: Loading and Processing Tracking Data with Dynamic Homography ===")
    df = pd.read_csv(data_csv_path)
    print(f"Loaded {len(df)} tracking points")

    # Process data with dynamic homography updates
    df_mapped = df.copy()
    current_homography = initial_homography

    # Group by frame and apply appropriate homography
    for frame_num in sorted(df['frame'].unique()):
        frame_data = df[df['frame'] == frame_num]

        # Check if we have corner data for this frame
        frame_corners = None
        for info in frame_info:
            if info['frame_number'] == frame_num and info['valid']:
                frame_corners = info['corners']
                break

        # Update homography if needed
        if frame_corners is not None:
            current_homography = homography_manager.update_homography_if_needed(
                frame_corners, frame_num
            )

        # Apply current homography to this frame's data
        if current_homography is not None:
            frame_mapped = tracker.map_pixels_to_court(frame_data, current_homography)
            df_mapped.loc[df['frame'] == frame_num, ['X_court', 'Y_court']] = frame_mapped[['X_court', 'Y_court']]

    tracker.debug_coordinates(df_mapped)

    print("=== STEP 4: Calculating Distances ===")
    player_distances = tracker.calculate_player_distances_improved(df_mapped)
    print("Player distances:")
    print(player_distances)
    
    ball_results = tracker.calculate_ball_distance_improved(df_mapped, fps=30.0)
    tracker.validate_ball_measurements(ball_results)
    
    # Save results (existing saves)
    output_path = data_csv_path.replace('.csv', '_with_video_court_coords.csv')
    df_mapped.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_path}")
    
    # Save corner analysis results (existing)
    corner_analysis_path = data_csv_path.replace('.csv', '_corner_analysis.csv')
    corner_df = pd.DataFrame(frame_info)
    corner_df.to_csv(corner_analysis_path, index=False, encoding='utf-8')
    print(f"Corner analysis saved to: {corner_analysis_path}")
    
    # NEW: Save player movement analysis
    player_analysis_path = data_csv_path.replace('.csv', '_player_movement_analysis.csv')
    player_distances.to_csv(player_analysis_path, index=False, encoding='utf-8')
    print(f"Player movement analysis saved to: {player_analysis_path}")
    
    # NEW: Save ball movement analysis
    ball_analysis_path = data_csv_path.replace('.csv', '_ball_movement_analysis.csv')
    ball_df = pd.DataFrame([ball_results])  # Convert dict to DataFrame
    ball_df.to_csv(ball_analysis_path, index=False, encoding='utf-8')
    print(f"Ball movement analysis saved to: {ball_analysis_path}")
    
    # NEW: Save detailed movement patterns (optional)
    movement_patterns = tracker.analyze_player_movement_patterns(df_mapped)
    if movement_patterns:
        patterns_path = data_csv_path.replace('.csv', '_movement_patterns.json')
        import json
        with open(patterns_path, 'w') as f:
            json.dump(movement_patterns, f, indent=2)
        print(f"Movement patterns saved to: {patterns_path}")
    
    return df_mapped, player_distances, ball_results, average_corners, frame_info