class HomographyManager:
    def __init__(self, tennis_tracker):
        self.tracker = tennis_tracker
        self.current_homography = None
        self.current_corners = None
        self.homography_history = []
        self.corner_change_threshold = 15.0
        
    def update_homography_if_needed(self, frame_corners, frame_number):
        """Update homography if corners have changed significantly"""
        if not frame_corners or len(frame_corners) != 4:
            return self.current_homography
        
        # Check if this is first frame or significant change
        if (self.current_corners is None or 
            self.tracker._detect_significant_corner_change(frame_corners, self.current_corners, self.corner_change_threshold)):
            
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