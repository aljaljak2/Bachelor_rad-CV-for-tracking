"""
Homography Manager Module for Dynamic Tennis Court Tracking
"""


class HomographyManager:
    def __init__(self, tennis_tracker):
        """
        Initialize HomographyManager
        
        Args:
            tennis_tracker: Instance of TennisCourtTracker
        """
        self.tracker = tennis_tracker
        self.current_homography = None
        self.current_corners = None
        self.homography_history = []
        self.corner_change_threshold = 15.0
        
    def update_homography_if_needed(self, frame_corners, frame_number):
        """
        Update homography if corners have changed significantly
        
        Args:
            frame_corners: List of 4 corner points for current frame
            frame_number: Current frame number
            
        Returns:
            Current homography matrix (updated or existing)
        """
        if not frame_corners or len(frame_corners) != 4:
            return self.current_homography
        
        # Check if this is first frame or significant change
        if (self.current_corners is None or 
            self._detect_significant_corner_change(frame_corners, self.current_corners, self.corner_change_threshold)):
            
            try:
                new_homography = self.tracker.compute_homography(frame_corners)
                is_valid = self.tracker.validate_homography(new_homography, frame_corners)
                
                if is_valid:
                    self.current_homography = new_homography
                    self.current_corners = frame_corners
                    self.homography_history.append({
                        'frame': frame_number,
                        'homography': new_homography,
                        'corners': frame_corners.copy()
                    })
                    print(f"[DEBUG] Updated homography at frame {frame_number}")
                    return new_homography
                else:
                    print(f"[WARNING] Invalid homography at frame {frame_number}, keeping previous")
                    
            except Exception as e:
                print(f"[ERROR] Failed to compute homography at frame {frame_number}: {e}")
        
        return self.current_homography

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

        current_ordered = self.tracker.get_ordered_corners(current_corners)
        previous_ordered = self.tracker.get_ordered_corners(previous_corners)

        import numpy as np
        for curr, prev in zip(current_ordered, previous_ordered):
            distance = np.linalg.norm(np.array(curr) - np.array(prev))
            if distance > threshold:
                return True

        return False

    def get_homography_stats(self):
        """
        Get statistics about homography updates
        
        Returns:
            Dictionary with homography statistics
        """
        return {
            'total_updates': len(self.homography_history),
            'current_corners': self.current_corners,
            'threshold': self.corner_change_threshold,
            'update_frames': [h['frame'] for h in self.homography_history]
        }

    def set_corner_change_threshold(self, threshold):
        """
        Set the threshold for detecting significant corner changes
        
        Args:
            threshold: New threshold value in pixels
        """
        self.corner_change_threshold = threshold
        print(f"[DEBUG] Corner change threshold set to {threshold} pixels")