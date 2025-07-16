"""
Corner Detection Module for Tennis Court Tracking
"""
import cv2
import numpy as np
import urllib.request
import os
from sklearn.cluster import DBSCAN


class CornerDetector:
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
                    self.draw_corners_on_frame(debug_frame, corners)
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

    def get_ordered_corners(self, corners):
        """Order corners in a consistent manner"""
        if len(corners) != 4:
            return corners
        corners = sorted(corners, key=lambda p: p[1])
        top_points = sorted(corners[:2], key=lambda p: p[0])
        bottom_points = sorted(corners[2:], key=lambda p: p[0])
        return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]

    def visualize_average_corners(self, frame, average_corners, output_path="average_corners_visualization.jpg", show_image=True):
        """
        Visualize average corners on a representative frame
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
        self.draw_corners_on_frame(vis_frame, average_corners)
        
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

    def draw_corners_on_frame(self, frame, corners):
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

    def _detect_significant_corner_change(self, current_corners, previous_corners, threshold=15.0):
        """
        Detect if corner positions have changed significantly
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

    def _find_sideline_segments(self, lines, img_width, img_height):
        """Find line segments that could be sidelines"""
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
        """Group sideline segments into left and right sidelines"""
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
        """Find corners by tracing sideline segments"""
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
        """Find top and bottom corner pair for a sideline"""
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
        """Find segments that are collinear with the reference segment"""
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
        """Find the topmost point from collinear segments"""
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