import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import os

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

    def detect_court_corners(self, frame, debug=False):
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
        sideline_segments = self._find_sideline_segments(lines, w, h)
        if len(sideline_segments) < 2:
            print("[DEBUG] Not enough sideline segments found.")
            return []
        left_sideline, right_sideline = self._group_sideline_segments(sideline_segments, w, debug_img)
        if not left_sideline or not right_sideline:
            print("[DEBUG] Could not identify both sidelines.")
            return []
        corners = self._find_corners_by_line_tracing(left_sideline, right_sideline, debug_img, h, w)
        if debug:
            print("[DEBUG] Displaying image with detected sidelines and corners...")
            cv2.imshow('Tennis Court Detection', debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"[DEBUG] Found {len(corners)} corners: {corners}")
        return corners

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
        print(f"[DEBUG] Found {len(sideline_segments)} potential sideline segments")
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

    def calculate_ball_distance(self, df, x_col='X_court', y_col='Y_court', class_col='Class', frame_col='frame'):
        ball = df[df[class_col] == 'Ball'].copy()
        if ball.empty or len(ball) < 2:
            return 0.0
        ball = ball.sort_values(frame_col)
        coords = ball[[x_col, y_col]].values
        diffs = np.diff(coords, axis=0)
        frame_distances = np.linalg.norm(diffs, axis=1)
        max_ball_speed_per_frame = 2.0
        valid_distances = frame_distances[frame_distances <= max_ball_speed_per_frame]
        return np.sum(valid_distances)

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

    def draw_and_save_corners(self, frame, corners, out_path):
        """
        Draw detected corners on the frame and save the result to the specified path.
        """
        if frame is None or not corners or len(corners) != 4:
            print("[ERROR] Cannot draw corners: invalid frame or corners.")
            return
        # Order corners for consistent labeling
        ordered_corners = self.get_ordered_corners(corners)
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        vis_frame = frame.copy()
        for i, (corner, label, color) in enumerate(zip(ordered_corners, corner_labels, colors)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis_frame, (x, y), 8, color, -1)
            cv2.circle(vis_frame, (x, y), 12, color, 2)
            cv2.putText(vis_frame, label, (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Draw lines connecting the corners
        for i in range(4):
            pt1 = (int(ordered_corners[i][0]), int(ordered_corners[i][1]))
            pt2 = (int(ordered_corners[(i+1)%4][0]), int(ordered_corners[(i+1)%4][1]))
            cv2.line(vis_frame, pt1, pt2, (255, 255, 255), 2)
        # Add title
        cv2.putText(vis_frame, 'Tennis Court Corners Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Save to Out folder
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis_frame)
        print(f"[DEBUG] Saved corners visualization to: {out_path}")

def main_processing_pipeline(frame_path, data_csv_path):
    tracker = TennisCourtTracker()
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Could not load image from {frame_path}")
        return
    print("=== STEP 1: Detecting Court Corners ===")
    corners = tracker.detect_court_corners(frame, debug=True)
    if len(corners) != 4:
        print(f"Error: Could not detect all 4 corners. Found {len(corners)}")
        return
    # Draw and save corners visualization
    out_dir = './Out'
    out_img_path = os.path.join(out_dir, os.path.basename(frame_path).replace('.png', '_corners.jpg').replace('.jpg', '_corners.jpg'))
    tracker.draw_and_save_corners(frame, corners, out_img_path)
    print(f"[INFO] Corners visualization saved to {out_img_path}")
    print("=== STEP 2: Computing Homography ===")
    try:
        H = tracker.compute_homography(corners)
        print("Homography matrix computed successfully")
        is_valid = tracker.validate_homography(H, corners)
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
    player_distances = tracker.calculate_player_distances(df_mapped)
    print("Player distances:")
    print(player_distances)
    ball_distance = tracker.calculate_ball_distance(df_mapped)
    print(f"\nBall total distance: {ball_distance:.2f} meters")
    output_path = data_csv_path.replace('.csv', '_with_court_coords.csv')
    df_mapped.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    return df_mapped, player_distances, ball_distance