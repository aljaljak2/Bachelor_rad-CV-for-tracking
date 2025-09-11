import os
import cv2
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import glob

class ResolutionPerformanceOptimizer:
    """
    Module for testing tennis court tracking at multiple resolutions to find optimal 
    performance vs accuracy balance for real-time processing.
    """
    
    def __init__(self, target_fps: float = 30.0, min_detection_threshold: int = 2):
        """
        Initialize the optimizer.
        
        Parameters
        ----------
        target_fps : float
            Target FPS for real-time processing (default: 30.0)
        min_detection_threshold : int
            Minimum number of detections per frame to consider valid (default: 2)
        """
        self.target_fps = target_fps
        self.min_detection_threshold = min_detection_threshold
        self.results = []
        
    def generate_resolution_list(self, original_width: int, original_height: int, 
                                step_factor: float = 0.8, min_width: int = 320) -> List[Tuple[int, int]]:
        """
        Generate list of resolutions to test, starting from original and scaling down.
        
        Parameters
        ----------
        original_width : int
            Original video width
        original_height : int  
            Original video height
        step_factor : float
            Factor to scale down resolution each step (default: 0.8 = 20% reduction)
        min_width : int
            Minimum width to test (default: 320)
            
        Returns
        -------
        List[Tuple[int, int]]
            List of (width, height) tuples to test
        """
        resolutions = []
        current_width = original_width
        current_height = original_height
        
        while current_width >= min_width:
            # Ensure even dimensions for video encoding compatibility
            width = int(current_width // 2) * 2
            height = int(current_height // 2) * 2
            resolutions.append((width, height))
            
            current_width *= step_factor
            current_height *= step_factor
            
        return resolutions
    
    def interactive_resolution_selection(self, video_path: str, step_factor: float = 0.8, 
                                        min_width: int = 320) -> List[Tuple[int, int]]:
        """
        Interactive resolution selection allowing user to choose which resolutions to test.
        
        Parameters
        ----------
        video_path : str
            Path to input video
        step_factor : float
            Resolution reduction factor between options
        min_width : int
            Minimum width to generate
            
        Returns
        -------
        List[Tuple[int, int]]
            List of selected resolutions to test
        """
        # Get original video dimensions
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"\nOriginal video resolution: {original_width}x{original_height}")
        
        # Generate all possible resolutions
        all_resolutions = self.generate_resolution_list(
            original_width, original_height, step_factor, min_width
        )
        
        print(f"\nAvailable resolutions to test:")
        print("=" * 50)
        for i, (width, height) in enumerate(all_resolutions, 1):
            # Calculate percentage of original
            percentage = (width * height) / (original_width * original_height) * 100
            print(f"{i:2d}. {width:4d}x{height:4d} ({percentage:5.1f}% of original)")
        
        print("\nResolution Selection Options:")
        print("1. Enter specific numbers (e.g., '1 3 5' to test resolutions 1, 3, and 5)")
        print("2. Enter 'all' to test all resolutions")
        print("3. Enter 'range X-Y' to test resolutions from X to Y (e.g., 'range 1-5')")
        print("4. Enter 'top N' to test the N highest resolutions (e.g., 'top 3')")
        print("5. Enter 'bottom N' to test the N lowest resolutions (e.g., 'bottom 3')")
        print("6. Enter 'every N' to test every Nth resolution (e.g., 'every 2')")
        
        while True:
            try:
                user_input = input("\nEnter your selection: ").strip().lower()
                
                if user_input == 'all':
                    selected_resolutions = all_resolutions
                    print(f"Selected all {len(selected_resolutions)} resolutions for testing.")
                    break
                
                elif user_input.startswith('range '):
                    # Parse range input (e.g., "range 1-5")
                    range_part = user_input.replace('range ', '')
                    if '-' in range_part:
                        start_str, end_str = range_part.split('-')
                        start_idx = int(start_str) - 1  # Convert to 0-based indexing
                        end_idx = int(end_str)  # End is exclusive in slicing
                        
                        if 0 <= start_idx < len(all_resolutions) and 0 < end_idx <= len(all_resolutions):
                            selected_resolutions = all_resolutions[start_idx:end_idx]
                            print(f"Selected resolutions {start_str} to {end_str} ({len(selected_resolutions)} resolutions).")
                            break
                        else:
                            print(f"Invalid range. Please use numbers between 1 and {len(all_resolutions)}.")
                    else:
                        print("Invalid range format. Use 'range X-Y' (e.g., 'range 1-5').")
                
                elif user_input.startswith('top '):
                    # Take top N resolutions (highest resolution)
                    n = int(user_input.replace('top ', ''))
                    if 1 <= n <= len(all_resolutions):
                        selected_resolutions = all_resolutions[:n]
                        print(f"Selected top {n} highest resolutions.")
                        break
                    else:
                        print(f"Invalid number. Please use a number between 1 and {len(all_resolutions)}.")
                
                elif user_input.startswith('bottom '):
                    # Take bottom N resolutions (lowest resolution)
                    n = int(user_input.replace('bottom ', ''))
                    if 1 <= n <= len(all_resolutions):
                        selected_resolutions = all_resolutions[-n:]
                        print(f"Selected bottom {n} lowest resolutions.")
                        break
                    else:
                        print(f"Invalid number. Please use a number between 1 and {len(all_resolutions)}.")
                
                elif user_input.startswith('every '):
                    # Take every Nth resolution
                    n = int(user_input.replace('every ', ''))
                    if n >= 1:
                        selected_resolutions = all_resolutions[::n]
                        print(f"Selected every {n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} resolution ({len(selected_resolutions)} resolutions).")
                        break
                    else:
                        print("Invalid number. Please use a number >= 1.")
                
                else:
                    # Parse individual numbers (e.g., "1 3 5")
                    indices = [int(x.strip()) for x in user_input.split()]
                    
                    # Validate indices
                    invalid_indices = [i for i in indices if i < 1 or i > len(all_resolutions)]
                    if invalid_indices:
                        print(f"Invalid indices: {invalid_indices}. Please use numbers between 1 and {len(all_resolutions)}.")
                        continue
                    
                    # Convert to 0-based indexing and get resolutions
                    selected_indices = [i - 1 for i in indices]
                    selected_resolutions = [all_resolutions[i] for i in selected_indices]
                    
                    print(f"Selected {len(selected_resolutions)} resolutions: {indices}")
                    break
                    
            except ValueError:
                print("Invalid input. Please enter numbers, 'all', or use the specified formats.")
            except Exception as e:
                print(f"Error parsing input: {e}")
        
        # Display selected resolutions
        print(f"\nSelected resolutions for testing:")
        print("-" * 40)
        for i, (width, height) in enumerate(selected_resolutions, 1):
            percentage = (width * height) / (original_width * original_height) * 100
            print(f"{i:2d}. {width:4d}x{height:4d} ({percentage:5.1f}% of original)")
        
        # Confirm selection
        while True:
            confirm = input(f"\nProceed with testing these {len(selected_resolutions)} resolutions? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                break
            elif confirm in ['n', 'no']:
                print("Selection cancelled. Please run again to make a new selection.")
                return []
            else:
                print("Please enter 'y' or 'n'.")
        
        return selected_resolutions

    def create_resized_video(self, input_video_path: str, output_video_path: str, 
                           target_resolution: Tuple[int, int]) -> str:
        """
        Create a resized version of the input video.
        
        Parameters
        ----------
        input_video_path : str
            Path to original video
        output_video_path : str
            Path for resized video output
        target_resolution : Tuple[int, int]
            Target (width, height) for resizing
            
        Returns
        -------
        str
            Path to created resized video
        """
        cap = cv2.VideoCapture(input_video_path)
        
        # Get original video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer for resized video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, target_resolution)
        
        print(f"Creating resized video: {target_resolution[0]}x{target_resolution[1]}")
        print(f"Processing {frame_count} frames...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            resized_frame = cv2.resize(frame, target_resolution)
            out.write(resized_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        out.release()
        
        print(f"Resized video saved: {output_video_path}")
        return output_video_path
    
    def analyze_timing_csv(self, csv_path: str) -> Dict:
        """
        Analyze the timing CSV file to extract performance metrics.
        
        Parameters
        ----------
        csv_path : str
            Path to the detailed_frame_timing CSV file
            
        Returns
        -------
        Dict
            Dictionary containing performance metrics
        """
        if not os.path.exists(csv_path):
            print(f"Warning: Timing CSV not found: {csv_path}")
            return {
                'avg_total_time': float('inf'),
                'avg_detection_time': float('inf'),
                'avg_tracking_time': float('inf'),
                'avg_fps': 0.0,
                'frame_count': 0,
                'real_time_percentage': 0.0
            }
        
        try:
            df = pd.read_csv(csv_path)
            
            # Calculate metrics
            avg_total_time = df['Total_Processing_Time (s)'].mean()
            avg_detection_time = df['Detection_Time (s)'].mean()
            avg_tracking_time = df['Tracking_Time (s)'].mean()
            frame_count = len(df)
            
            # Calculate average FPS (1 / avg_total_time)
            avg_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0.0
            
            # Calculate what percentage of real-time this achieves
            real_time_percentage = (avg_fps / self.target_fps) * 100 if self.target_fps > 0 else 0.0
            
            return {
                'avg_total_time': avg_total_time,
                'avg_detection_time': avg_detection_time,
                'avg_tracking_time': avg_tracking_time,
                'avg_fps': avg_fps,
                'frame_count': frame_count,
                'real_time_percentage': real_time_percentage,
                'csv_path': csv_path
            }
            
        except Exception as e:
            print(f"Error analyzing timing CSV {csv_path}: {e}")
            return {
                'avg_total_time': float('inf'),
                'avg_detection_time': float('inf'),
                'avg_tracking_time': float('inf'),
                'avg_fps': 0.0,
                'frame_count': 0,
                'real_time_percentage': 0.0
            }
    
    def count_detections_in_results(self, ziframes: List, zitboxes: List) -> Dict:
        """
        Analyze detection quality from the tracking results.
        
        Parameters
        ----------
        ziframes : List
            List of processed frames
        zitboxes : List
            List of tracking boxes per frame
            
        Returns
        -------
        Dict
            Dictionary containing detection quality metrics
        """
        if not zitboxes:
            return {
                'avg_detections_per_frame': 0.0,
                'avg_player_detections': 0.0,
                'avg_ball_detections': 0.0,
                'frames_with_ball': 0,
                'ball_detection_rate': 0.0,
                'total_frames': len(ziframes)
            }
        
        total_detections = []
        player_detections = []
        ball_detections = []
        frames_with_ball = 0
        
        for frame_boxes in zitboxes:
            frame_total = len(frame_boxes)
            frame_players = len([box for box in frame_boxes if box[-1] != 32])  # Not ball
            frame_balls = len([box for box in frame_boxes if box[-1] == 32])    # Ball
            
            total_detections.append(frame_total)
            player_detections.append(frame_players)
            ball_detections.append(frame_balls)
            
            if frame_balls > 0:
                frames_with_ball += 1
        
        return {
            'avg_detections_per_frame': np.mean(total_detections) if total_detections else 0.0,
            'avg_player_detections': np.mean(player_detections) if player_detections else 0.0,
            'avg_ball_detections': np.mean(ball_detections) if ball_detections else 0.0,
            'frames_with_ball': frames_with_ball,
            'ball_detection_rate': (frames_with_ball / len(zitboxes)) * 100 if zitboxes else 0.0,
            'total_frames': len(ziframes)
        }
    
    def run_resolution_test(self, video_path: str, resolution: Tuple[int, int], 
                           out_name: str, teams_colors: List[str], 
                           ball_only: bool = True) -> Dict:
        """
        Run tracking test at a specific resolution.
        
        Parameters
        ----------
        video_path : str
            Path to video file
        resolution : Tuple[int, int]
            Target resolution (width, height)  
        out_name : str
            Output name prefix
        teams_colors : List[str]
            Team colors for classification
        ball_only : bool
            Whether to keep only frames with ball
            
        Returns
        -------
        Dict
            Combined results from timing and detection analysis
        """
        # Import the tracking function from the correct location
        import sys
        import os
        
        # Add parent directory to path to access Detect_and_Track folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            from Detect_and_Track.create_tracking_video_and_init_data import create_tracking_video_and_init_data
        except ImportError as e:
            print(f"Import error: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Python path: {sys.path}")
            print(f"Looking for: Detect_and_Track/create_tracking_video_and_init_data.py")
            raise
        
        print(f"\n{'='*60}")
        print(f"Testing resolution: {resolution[0]}x{resolution[1]}")
        print(f"{'='*60}")
        
        # Create resized video if needed
        original_cap = cv2.VideoCapture(video_path)
        original_width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_cap.release()
        
        test_video_path = video_path
        if (original_width, original_height) != resolution:
            # Create resized video
            resized_video_name = f"temp_resized_{resolution[0]}x{resolution[1]}_{int(time.time())}.mp4"
            resized_video_path = f"./Out/{resized_video_name}"
            test_video_path = self.create_resized_video(video_path, resized_video_path, resolution)
        
        try:
            # Run the tracking
            start_time = time.time()
            init_df, ziframes, zitboxes = create_tracking_video_and_init_data(
                test_video_path, f"{out_name}_{resolution[0]}x{resolution[1]}", 
                teams_colors, ball_only
            )
            end_time = time.time()
            
            # Find the timing CSV file
            timing_csv_pattern = f"./Out/detailed_frame_timing_*.csv"
            timing_files = glob.glob(timing_csv_pattern)
            latest_timing_file = max(timing_files, key=os.path.getctime) if timing_files else None
            
            # Analyze timing performance
            timing_metrics = self.analyze_timing_csv(latest_timing_file) if latest_timing_file else {}
            
            # Analyze detection quality
            detection_metrics = self.count_detections_in_results(ziframes, zitboxes)
            
            # Calculate overall processing time
            total_processing_time = end_time - start_time
            
            # Combine results
            result = {
                'resolution': resolution,
                'resolution_str': f"{resolution[0]}x{resolution[1]}",
                'total_processing_time': total_processing_time,
                'video_path': test_video_path,
                **timing_metrics,
                **detection_metrics
            }
            
            # Clean up temporary resized video
            if test_video_path != video_path and os.path.exists(test_video_path):
                os.remove(test_video_path)
                print(f"Cleaned up temporary video: {test_video_path}")
            
            return result
            
        except Exception as e:
            print(f"Error testing resolution {resolution}: {e}")
            # Clean up on error
            if test_video_path != video_path and os.path.exists(test_video_path):
                os.remove(test_video_path)
            return {
                'resolution': resolution,
                'resolution_str': f"{resolution[0]}x{resolution[1]}",
                'error': str(e),
                'avg_fps': 0.0,
                'real_time_percentage': 0.0
            }
    
    def optimize_resolution_interactive(self, video_path: str, out_name: str, teams_colors: List[str],
                                       ball_only: bool = True, step_factor: float = 0.8, 
                                       min_width: int = 320) -> Dict:
        """
        Run optimization with interactive resolution selection.
        
        Parameters
        ----------
        video_path : str
            Path to input video
        out_name : str
            Output name prefix
        teams_colors : List[str]
            Team colors for classification
        ball_only : bool
            Whether to keep only frames with ball
        step_factor : float
            Resolution reduction factor between options
        min_width : int
            Minimum width to generate
            
        Returns
        -------
        Dict
            Optimization results with recommendations
        """
        print("INTERACTIVE RESOLUTION OPTIMIZATION")
        print("=" * 50)
        
        # Interactive resolution selection
        selected_resolutions = self.interactive_resolution_selection(
            video_path, step_factor, min_width
        )
        
        if not selected_resolutions:
            return {"error": "No resolutions selected"}
        
        print(f"\nStarting optimization with {len(selected_resolutions)} selected resolutions...")
        
        # Test each selected resolution
        results = []
        for i, resolution in enumerate(selected_resolutions, 1):
            print(f"\nProgress: {i}/{len(selected_resolutions)}")
            result = self.run_resolution_test(
                video_path, resolution, out_name, teams_colors, ball_only
            )
            results.append(result)
            
            # Print summary for this resolution
            if 'error' not in result:
                print(f"\nResults for {result['resolution_str']}:")
                print(f"  Average FPS: {result.get('avg_fps', 0):.2f}")
                print(f"  Real-time %: {result.get('real_time_percentage', 0):.1f}%")
                print(f"  Avg detections/frame: {result.get('avg_detections_per_frame', 0):.1f}")
                print(f"  Ball detection rate: {result.get('ball_detection_rate', 0):.1f}%")
            else:
                print(f"\nError for {result['resolution_str']}: {result['error']}")
        
        self.results = results
        
        # Analyze results and make recommendations
        return self.analyze_optimization_results()

    # [Include all other methods from the original class - analyze_optimization_results, 
    # generate_performance_report, plot_performance_analysis, etc.]
    # ... (keeping the rest of the methods unchanged for brevity)
    
    def extract_detailed_timing_statistics(self, csv_path: str) -> Dict:
        """
        Extract detailed timing statistics from CSV file including median and standard deviation.
        
        Parameters
        ----------
        csv_path : str
            Path to the timing CSV file
            
        Returns
        -------
        Dict
            Detailed timing statistics
        """
        if not os.path.exists(csv_path):
            return {
                'total_time_stats': {'mean': 0, 'median': 0, 'std': 0},
                'detection_time_stats': {'mean': 0, 'median': 0, 'std': 0},
                'tracking_time_stats': {'mean': 0, 'median': 0, 'std': 0},
                'frame_count': 0,
                'total_processing_time': 0,
                'total_detection_time': 0,
                'total_tracking_time': 0
            }
        
        try:
            df = pd.read_csv(csv_path)
            
            # Per-frame statistics
            total_times = df['Total_Processing_Time (s)'].values
            detection_times = df['Detection_Time (s)'].values
            tracking_times = df['Tracking_Time (s)'].values
            
            # Calculate statistics
            total_time_stats = {
                'mean': np.mean(total_times),
                'median': np.median(total_times),
                'std': np.std(total_times)
            }
            
            detection_time_stats = {
                'mean': np.mean(detection_times),
                'median': np.median(detection_times),
                'std': np.std(detection_times)
            }
            
            tracking_time_stats = {
                'mean': np.mean(tracking_times),
                'median': np.median(tracking_times),
                'std': np.std(tracking_times)
            }
            
            # Total times
            frame_count = len(df)
            total_processing_time = np.sum(total_times)
            total_detection_time = np.sum(detection_times)
            total_tracking_time = np.sum(tracking_times)
            
            return {
                'total_time_stats': total_time_stats,
                'detection_time_stats': detection_time_stats,
                'tracking_time_stats': tracking_time_stats,
                'frame_count': frame_count,
                'total_processing_time': total_processing_time,
                'total_detection_time': total_detection_time,
                'total_tracking_time': total_tracking_time
            }
            
        except Exception as e:
            print(f"Error extracting detailed statistics from {csv_path}: {e}")
            return {
                'total_time_stats': {'mean': 0, 'median': 0, 'std': 0},
                'detection_time_stats': {'mean': 0, 'median': 0, 'std': 0},
                'tracking_time_stats': {'mean': 0, 'median': 0, 'std': 0},
                'frame_count': 0,
                'total_processing_time': 0,
                'total_detection_time': 0,
                'total_tracking_time': 0
            }

    def analyze_optimization_results(self) -> Dict:
        """
        Analyze the optimization results and provide recommendations.
        
        Returns
        -------
        Dict
            Analysis results with recommendations
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter out error results
        valid_results = [r for r in self.results if 'error' not in r and r.get('avg_fps', 0) > 0]
        
        if not valid_results:
            return {"error": "No valid results found"}
        
        # Find best options based on different criteria
        best_fps = max(valid_results, key=lambda x: x.get('avg_fps', 0))
        best_real_time = max(valid_results, key=lambda x: x.get('real_time_percentage', 0))
        best_detections = max(valid_results, key=lambda x: x.get('avg_detections_per_frame', 0))
        
        # Find resolutions that achieve real-time or close to it
        real_time_candidates = [
            r for r in valid_results 
            if r.get('real_time_percentage', 0) >= 80  # Within 80% of target FPS
        ]
        
        # Find balanced option (good FPS with reasonable detection quality)
        balanced_candidates = [
            r for r in valid_results 
            if (r.get('real_time_percentage', 0) >= 60 and 
                r.get('avg_detections_per_frame', 0) >= self.min_detection_threshold)
        ]
        
        recommended = None
        if real_time_candidates:
            # Choose the highest quality among real-time candidates
            recommended = max(real_time_candidates, key=lambda x: x.get('avg_detections_per_frame', 0))
        elif balanced_candidates:
            # Choose the fastest among balanced candidates
            recommended = max(balanced_candidates, key=lambda x: x.get('avg_fps', 0))
        else:
            # Fall back to best FPS
            recommended = best_fps
        
        return {
            'total_resolutions_tested': len(self.results),
            'valid_results_count': len(valid_results),
            'target_fps': self.target_fps,
            
            'best_fps': best_fps,
            'best_real_time_percentage': best_real_time,
            'best_detection_quality': best_detections,
            
            'real_time_candidates': real_time_candidates,
            'balanced_candidates': balanced_candidates,
            
            'recommended_resolution': recommended,
            'all_results': valid_results
        }
    def extract_comprehensive_timing_data(self) -> Dict:
        """
        Extract comprehensive timing data from all tested resolutions.
        
        Returns
        -------
        Dict
            Comprehensive timing analysis
        """
        timing_analysis = {
            'resolutions': [],
            'resolution_strings': [],
            'avg_time_per_frame': [],
            'total_processing_time': [],
            'avg_detection_time': [],
            'avg_tracking_time': [],
            'total_detection_time': [],
            'total_tracking_time': [],
            'frame_counts': [],
            'estimated_fps': []
        }
        
        for result in self.results:
            if 'error' in result:
                continue
                
            resolution_str = result.get('resolution_str', 'Unknown')
            resolution = result.get('resolution', (0, 0))
            
            # Extract timing data
            avg_total_time = result.get('avg_total_time', 0)
            avg_detection_time = result.get('avg_detection_time', 0)
            avg_tracking_time = result.get('avg_tracking_time', 0)
            frame_count = result.get('frame_count', 0)
            
            # Calculate totals
            total_detection = avg_detection_time * frame_count if frame_count > 0 else 0
            total_tracking = avg_tracking_time * frame_count if frame_count > 0 else 0
            total_processing = avg_total_time * frame_count if frame_count > 0 else 0
            
            # Calculate FPS
            fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
            
            timing_analysis['resolutions'].append(resolution)
            timing_analysis['resolution_strings'].append(resolution_str)
            timing_analysis['avg_time_per_frame'].append(avg_total_time)
            timing_analysis['total_processing_time'].append(total_processing)
            timing_analysis['avg_detection_time'].append(avg_detection_time)
            timing_analysis['avg_tracking_time'].append(avg_tracking_time)
            timing_analysis['total_detection_time'].append(total_detection)
            timing_analysis['total_tracking_time'].append(total_tracking)
            timing_analysis['frame_counts'].append(frame_count)
            timing_analysis['estimated_fps'].append(fps)
        
        return timing_analysis
    
    def analyze_optimization_results(self) -> Dict:
        """
        Analyze the optimization results and provide recommendations.
        
        Returns
        -------
        Dict
            Analysis results with recommendations
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Filter out error results
        valid_results = [r for r in self.results if 'error' not in r and r.get('avg_fps', 0) > 0]
        
        if not valid_results:
            return {"error": "No valid results found"}
        
        # Find best options based on different criteria
        best_fps = max(valid_results, key=lambda x: x.get('avg_fps', 0))
        best_real_time = max(valid_results, key=lambda x: x.get('real_time_percentage', 0))
        best_detections = max(valid_results, key=lambda x: x.get('avg_detections_per_frame', 0))
        
        # Find resolutions that achieve real-time or close to it
        real_time_candidates = [
            r for r in valid_results 
            if r.get('real_time_percentage', 0) >= 80  # Within 80% of target FPS
        ]
        
        # Find balanced option (good FPS with reasonable detection quality)
        balanced_candidates = [
            r for r in valid_results 
            if (r.get('real_time_percentage', 0) >= 60 and 
                r.get('avg_detections_per_frame', 0) >= self.min_detection_threshold)
        ]
        
        recommended = None
        if real_time_candidates:
            # Choose the highest quality among real-time candidates
            recommended = max(real_time_candidates, key=lambda x: x.get('avg_detections_per_frame', 0))
        elif balanced_candidates:
            # Choose the fastest among balanced candidates
            recommended = max(balanced_candidates, key=lambda x: x.get('avg_fps', 0))
        else:
            # Fall back to best FPS
            recommended = best_fps
        
        return {
            'total_resolutions_tested': len(self.results),
            'valid_results_count': len(valid_results),
            'target_fps': self.target_fps,
            
            'best_fps': best_fps,
            'best_real_time_percentage': best_real_time,
            'best_detection_quality': best_detections,
            
            'real_time_candidates': real_time_candidates,
            'balanced_candidates': balanced_candidates,
            
            'recommended_resolution': recommended,
            'all_results': valid_results
        }
        
    def generate_performance_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a detailed performance report.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save the report (if None, prints to console)

        Returns
        -------
        str
            The generated report
        """
        analysis = self.analyze_optimization_results()

        if 'error' in analysis:
            return f"Error generating report: {analysis['error']}"

        report = []
        report.append("="*80)
        report.append("TENNIS COURT TRACKING - RESOLUTION OPTIMIZATION REPORT")
        report.append("="*80)
        report.append(f"Target FPS for real-time processing: {self.target_fps}")
        report.append(f"Total resolutions tested: {analysis['total_resolutions_tested']}")
        report.append(f"Valid results: {analysis['valid_results_count']}")
        report.append("")

        # Recommended resolution
        if analysis['recommended_resolution']:
            rec = analysis['recommended_resolution']
            report.append("RECOMMENDED RESOLUTION:")
            report.append("-" * 30)
            report.append(f"Resolution: {rec['resolution_str']}")
            report.append(f"Average FPS: {rec.get('avg_fps', 0):.2f}")
            report.append(f"Real-time percentage: {rec.get('real_time_percentage', 0):.1f}%")
            report.append(f"Average detections per frame: {rec.get('avg_detections_per_frame', 0):.1f}")
            report.append(f"Ball detection rate: {rec.get('ball_detection_rate', 0):.1f}%")
            report.append("")

        # Real-time candidates
        if analysis['real_time_candidates']:
            report.append("REAL-TIME CAPABLE RESOLUTIONS:")
            report.append("-" * 40)
            for candidate in analysis['real_time_candidates']:
                report.append(f"  {candidate['resolution_str']} - {candidate.get('avg_fps', 0):.2f} FPS")
            report.append("")

        # Detailed per-resolution analysis
        report.append("DETAILED RESOLUTION ANALYSIS:")
        report.append("="*80)
        report.append("")

        for i, result in enumerate(self.results):
            if 'error' in result:
                report.append(f"RESOLUTION {i+1}: {result['resolution_str']} - ERROR")
                report.append("-" * 60)
                report.append(f"Error: {result['error']}")
                report.append("")
                continue

            # Get detailed timing statistics
            csv_path = result.get('csv_path')
            detailed_stats = self.extract_detailed_timing_statistics(csv_path) if csv_path else None

            resolution_str = result.get('resolution_str', 'Unknown')
            report.append(f"RESOLUTION {i+1}: {resolution_str}")
            report.append("-" * 60)

            # Basic performance metrics
            fps = result.get('avg_fps', 0)
            real_time_pct = result.get('real_time_percentage', 0)
            detections = result.get('avg_detections_per_frame', 0)
            ball_rate = result.get('ball_detection_rate', 0)

            report.append(f"Performance Summary:")
            report.append(f"  - Average FPS: {fps:.2f}")
            report.append(f"  - Real-time capability: {real_time_pct:.1f}%")
            report.append(f"  - Average detections per frame: {detections:.1f}")
            report.append(f"  - Ball detection rate: {ball_rate:.1f}%")
            report.append("")

            # Detailed timing analysis
            if detailed_stats:
                frame_count = detailed_stats['frame_count']

                report.append(f"Frame Count: {frame_count}")
                report.append("")

                # Per-frame timing statistics
                report.append("PER-FRAME TIMING STATISTICS:")
                report.append("  Total Processing Time per Frame:")
                report.append(f"    - Mean:     {detailed_stats['total_time_stats']['mean']:.4f} seconds")
                report.append(f"    - Median:   {detailed_stats['total_time_stats']['median']:.4f} seconds")
                report.append(f"    - Std Dev:  {detailed_stats['total_time_stats']['std']:.4f} seconds")
                report.append("")

                report.append("  Detection Time per Frame:")
                report.append(f"    - Mean:     {detailed_stats['detection_time_stats']['mean']:.4f} seconds")
                report.append(f"    - Median:   {detailed_stats['detection_time_stats']['median']:.4f} seconds")
                report.append(f"    - Std Dev:  {detailed_stats['detection_time_stats']['std']:.4f} seconds")
                report.append("")

                report.append("  Tracking Time per Frame:")
                report.append(f"    - Mean:     {detailed_stats['tracking_time_stats']['mean']:.4f} seconds")
                report.append(f"    - Median:   {detailed_stats['tracking_time_stats']['median']:.4f} seconds")
                report.append(f"    - Std Dev:  {detailed_stats['tracking_time_stats']['std']:.4f} seconds")
                report.append("")

                # Total timing for all frames
                report.append("TOTAL PROCESSING TIME FOR ALL FRAMES:")
                report.append(f"  - Total Processing Time:  {detailed_stats['total_processing_time']:.2f} seconds")
                report.append(f"  - Total Detection Time:   {detailed_stats['total_detection_time']:.2f} seconds")
                report.append(f"  - Total Tracking Time:    {detailed_stats['total_tracking_time']:.2f} seconds")
                report.append("")

                # Percentage breakdown
                if detailed_stats['total_processing_time'] > 0:
                    det_pct = (detailed_stats['total_detection_time'] / detailed_stats['total_processing_time']) * 100
                    track_pct = (detailed_stats['total_tracking_time'] / detailed_stats['total_processing_time']) * 100
                    other_pct = 100 - det_pct - track_pct

                    report.append("TIME DISTRIBUTION:")
                    report.append(f"  - Detection:  {det_pct:.1f}%")
                    report.append(f"  - Tracking:   {track_pct:.1f}%")
                    report.append(f"  - Other:      {other_pct:.1f}%")
                    report.append("")
            else:
                # Fallback to basic timing info
                avg_total_time = result.get('avg_total_time', 0)
                avg_detection_time = result.get('avg_detection_time', 0)
                avg_tracking_time = result.get('avg_tracking_time', 0)
                frame_count = result.get('frame_count', 0)

                report.append(f"Frame Count: {frame_count}")
                report.append("")
                report.append("BASIC TIMING INFORMATION:")
                report.append(f"  - Average total time per frame: {avg_total_time:.4f} seconds")
                report.append(f"  - Average detection time per frame: {avg_detection_time:.4f} seconds")
                report.append(f"  - Average tracking time per frame: {avg_tracking_time:.4f} seconds")

                if frame_count > 0:
                    total_proc = avg_total_time * frame_count
                    total_det = avg_detection_time * frame_count
                    total_track = avg_tracking_time * frame_count

                    report.append("")
                    report.append(f"  - Total processing time: {total_proc:.2f} seconds")
                    report.append(f"  - Total detection time: {total_det:.2f} seconds")
                    report.append(f"  - Total tracking time: {total_track:.2f} seconds")
                report.append("")

            report.append("="*60)
            report.append("")

        # Add comprehensive timing analysis
        timing_data = self.extract_comprehensive_timing_data()
        if timing_data['resolutions']:
            report.append("COMPREHENSIVE TIMING ANALYSIS:")
            report.append("=" * 50)

            # Summary statistics
            total_resolutions = len(timing_data['resolutions'])
            avg_fps_all = np.mean(timing_data['estimated_fps']) if timing_data['estimated_fps'] else 0
            fastest_idx = np.argmax(timing_data['estimated_fps']) if timing_data['estimated_fps'] else 0
            slowest_idx = np.argmin(timing_data['estimated_fps']) if timing_data['estimated_fps'] else 0

            report.append(f"Total resolutions tested: {total_resolutions}")
            report.append(f"Average FPS across all resolutions: {avg_fps_all:.2f}")
            report.append(f"Fastest resolution: {timing_data['resolution_strings'][fastest_idx]} ({timing_data['estimated_fps'][fastest_idx]:.2f} FPS)")
            report.append(f"Slowest resolution: {timing_data['resolution_strings'][slowest_idx]} ({timing_data['estimated_fps'][slowest_idx]:.2f} FPS)")
            report.append("")

            # Detailed timing table
            report.append("DETAILED TIMING BREAKDOWN:")
            report.append("-" * 80)
            report.append(f"{'Resolution':<12} {'Frames':<8} {'Avg/Frame(s)':<12} {'Total(s)':<10} {'Det/Frame(s)':<12} {'Track/Frame(s)':<13} {'FPS':<8}")
            report.append("-" * 80)

            for i in range(len(timing_data['resolution_strings'])):
                res_str = timing_data['resolution_strings'][i]
                frames = timing_data['frame_counts'][i]
                avg_frame = timing_data['avg_time_per_frame'][i]
                total_time = timing_data['total_processing_time'][i]
                avg_det = timing_data['avg_detection_time'][i]
                avg_track = timing_data['avg_tracking_time'][i]
                fps = timing_data['estimated_fps'][i]

                report.append(f"{res_str:<12} {frames:<8} {avg_frame:<12.4f} {total_time:<10.2f} {avg_det:<12.4f} {avg_track:<13.4f} {fps:<8.2f}")

            report.append("")

            # Total time breakdown
            report.append("TOTAL TIME BREAKDOWN BY COMPONENT:")
            report.append("-" * 50)
            report.append(f"{'Resolution':<12} {'Total Detect(s)':<15} {'Total Track(s)':<14} {'Total Process(s)':<15}")
            report.append("-" * 50)

            for i in range(len(timing_data['resolution_strings'])):
                res_str = timing_data['resolution_strings'][i]
                total_det = timing_data['total_detection_time'][i]
                total_track = timing_data['total_tracking_time'][i]
                total_proc = timing_data['total_processing_time'][i]

                report.append(f"{res_str:<12} {total_det:<15.2f} {total_track:<14.2f} {total_proc:<15.2f}")

            report.append("")

            # Performance insights
            report.append("PERFORMANCE INSIGHTS:")
            report.append("-" * 30)

            if len(timing_data['estimated_fps']) >= 2:
                fps_improvement = timing_data['estimated_fps'][-1] / timing_data['estimated_fps'][0] if timing_data['estimated_fps'][0] > 0 else 0
                report.append(f"FPS improvement (lowest vs highest res): {fps_improvement:.2f}x")

                # Calculate detection vs tracking time ratios
                avg_det_ratio = np.mean([timing_data['avg_detection_time'][i] / timing_data['avg_time_per_frame'][i] 
                                       for i in range(len(timing_data['avg_time_per_frame'])) 
                                       if timing_data['avg_time_per_frame'][i] > 0]) * 100
                avg_track_ratio = np.mean([timing_data['avg_tracking_time'][i] / timing_data['avg_time_per_frame'][i] 
                                         for i in range(len(timing_data['avg_time_per_frame'])) 
                                         if timing_data['avg_time_per_frame'][i] > 0]) * 100

                report.append(f"Average time spent on detection: {avg_det_ratio:.1f}%")
                report.append(f"Average time spent on tracking: {avg_track_ratio:.1f}%")

            report.append("")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        else:
            print(report_text)

        return report_text

    def plot_performance_analysis(self, save_path: Optional[str] = None):
        """
        Create visualization plots of the performance analysis.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save the plot (if None, displays interactively)
        """
        analysis = self.analyze_optimization_results()

        if 'error' in analysis or not analysis['all_results']:
            print("No valid results to plot")
            return

        results = analysis['all_results']

        # Extract data for plotting
        resolutions = [r['resolution_str'] for r in results]
        fps_values = [r.get('avg_fps', 0) for r in results]
        detection_values = [r.get('avg_detections_per_frame', 0) for r in results]
        ball_detection_rates = [r.get('ball_detection_rate', 0) for r in results]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tennis Court Tracking - Resolution Performance Analysis', fontsize=16)

        # Plot 1: FPS vs Resolution
        ax1.bar(resolutions, fps_values, color='skyblue', alpha=0.7)
        ax1.axhline(y=self.target_fps, color='red', linestyle='--', label=f'Target FPS ({self.target_fps})')
        ax1.set_title('Average FPS by Resolution')
        ax1.set_ylabel('FPS')
        ax1.set_xlabel('Resolution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Detection Quality vs Resolution  
        ax2.bar(resolutions, detection_values, color='lightgreen', alpha=0.7)
        ax2.axhline(y=self.min_detection_threshold, color='orange', linestyle='--', 
                   label=f'Min Threshold ({self.min_detection_threshold})')
        ax2.set_title('Average Detections per Frame')
        ax2.set_ylabel('Detections per Frame')
        ax2.set_xlabel('Resolution')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Ball Detection Rate
        ax3.bar(resolutions, ball_detection_rates, color='coral', alpha=0.7)
        ax3.set_title('Ball Detection Rate (%)')
        ax3.set_ylabel('Ball Detection Rate (%)')
        ax3.set_xlabel('Resolution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Plot 4: FPS vs Detection Quality (scatter)
        ax4.scatter(detection_values, fps_values, s=100, alpha=0.7, c='purple')
        for i, res in enumerate(resolutions):
            ax4.annotate(res, (detection_values[i], fps_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.axhline(y=self.target_fps, color='red', linestyle='--', alpha=0.7)
        ax4.axvline(x=self.min_detection_threshold, color='orange', linestyle='--', alpha=0.7)
        ax4.set_title('FPS vs Detection Quality')
        ax4.set_xlabel('Average Detections per Frame')
        ax4.set_ylabel('FPS')
        ax4.grid(True, alpha=0.3)

        # Highlight recommended resolution if available
        if analysis['recommended_resolution']:
            rec = analysis['recommended_resolution']
            rec_res = rec['resolution_str']
            if rec_res in resolutions:
                idx = resolutions.index(rec_res)
                # Highlight in all plots
                ax1.bar(rec_res, fps_values[idx], color='gold', alpha=0.9, label='Recommended')
                ax2.bar(rec_res, detection_values[idx], color='gold', alpha=0.9)
                ax3.bar(rec_res, ball_detection_rates[idx], color='gold', alpha=0.9)
                ax1.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {save_path}")
        else:
            plt.show()

    def plot_comprehensive_timing_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive timing analysis plots.

        Parameters
        ----------
        save_path : Optional[str]
            Path to save the plot (if None, displays interactively)
        """
        timing_data = self.extract_comprehensive_timing_data()

        if not timing_data['resolutions']:
            print("No timing data to plot")
            return

        # Create comprehensive timing plots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Comprehensive Timing Analysis - Tennis Court Tracking', fontsize=16, fontweight='bold')

        resolutions = timing_data['resolution_strings']

        # Plot 1: Average Time per Frame
        ax1.bar(resolutions, timing_data['avg_time_per_frame'], color='lightblue', alpha=0.7)
        ax1.set_title('Average Processing Time per Frame')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Resolution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(timing_data['avg_time_per_frame']):
            ax1.text(i, v + max(timing_data['avg_time_per_frame']) * 0.01, f'{v:.3f}s', 
                    ha='center', va='bottom', fontsize=8)

        # Plot 2: FPS Comparison
        ax2.bar(resolutions, timing_data['estimated_fps'], color='lightgreen', alpha=0.7)
        ax2.axhline(y=self.target_fps, color='red', linestyle='--', label=f'Target FPS ({self.target_fps})')
        ax2.set_title('Estimated FPS by Resolution')
        ax2.set_ylabel('FPS')
        ax2.set_xlabel('Resolution')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(timing_data['estimated_fps']):
            ax2.text(i, v + max(timing_data['estimated_fps']) * 0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontsize=8)

        # Plot 3: Detection vs Tracking Time (Stacked Bar)
        ax3.bar(resolutions, timing_data['avg_detection_time'], label='Detection Time', 
                color='royalblue', alpha=0.7)
        ax3.bar(resolutions, timing_data['avg_tracking_time'], 
                bottom=timing_data['avg_detection_time'], label='Tracking Time', 
                color='orange', alpha=0.7)
        ax3.set_title('Detection vs Tracking Time per Frame')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('Resolution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Total Processing Time
        ax4.bar(resolutions, timing_data['total_processing_time'], color='gold', alpha=0.7)
        ax4.set_title('Total Processing Time for All Frames')
        ax4.set_ylabel('Total Time (seconds)')
        ax4.set_xlabel('Resolution')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for i, v in enumerate(timing_data['total_processing_time']):
            ax4.text(i, v + max(timing_data['total_processing_time']) * 0.01, f'{v:.1f}s', 
                    ha='center', va='bottom', fontsize=8)

        # Plot 5: Component Time Distribution (Pie charts for highest and lowest resolution)
        if len(timing_data['resolutions']) >= 2:
            # Highest resolution (first)
            det_time_high = timing_data['total_detection_time'][0]
            track_time_high = timing_data['total_tracking_time'][0]
            other_time_high = timing_data['total_processing_time'][0] - det_time_high - track_time_high

            sizes_high = [det_time_high, track_time_high, max(0, other_time_high)]
            labels = ['Detection', 'Tracking', 'Other']
            colors = ['royalblue', 'orange', 'lightgray']

            # Remove zero or negative values
            sizes_high = [max(0, s) for s in sizes_high]
            valid_data_high = [(s, l, c) for s, l, c in zip(sizes_high, labels, colors) if s > 0]
            if valid_data_high:
                sizes_high, labels_high, colors_high = zip(*valid_data_high)
                ax5.pie(sizes_high, labels=labels_high, colors=colors_high, autopct='%1.1f%%', startangle=90)
            ax5.set_title(f'Time Distribution\n{resolutions[0]} (Highest Res)')

            # Lowest resolution (last)
            det_time_low = timing_data['total_detection_time'][-1]
            track_time_low = timing_data['total_tracking_time'][-1]
            other_time_low = timing_data['total_processing_time'][-1] - det_time_low - track_time_low

            sizes_low = [det_time_low, track_time_low, max(0, other_time_low)]
            sizes_low = [max(0, s) for s in sizes_low]
            valid_data_low = [(s, l, c) for s, l, c in zip(sizes_low, labels, colors) if s > 0]
            if valid_data_low:
                sizes_low, labels_low, colors_low = zip(*valid_data_low)
                ax6.pie(sizes_low, labels=labels_low, colors=colors_low, autopct='%1.1f%%', startangle=90)
            ax6.set_title(f'Time Distribution\n{resolutions[-1]} (Lowest Res)')
        else:
            ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax5.transAxes)
            ax6.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax6.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive timing plots saved to: {save_path}")
        else:
            plt.show()
