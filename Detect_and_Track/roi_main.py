# roi_main.py - Enhanced main functions for FAIR ROI tracking experiments

import os
import sys
import logging
import time
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration for ROI experiments.
    
    Parameters
    ----------
    log_level : int
        Logging level (default: logging.INFO)
        
    Returns
    -------
    str
        Path to the log file
    """
    log_filename = f'./Out/roi_experiment_{int(time.time())}.log'
    
    # Create Out directory if it doesn't exist
    os.makedirs('./Out', exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename


def validate_experiment_setup(video_path, teams_colors=None):
    """
    Validate that all required components are available for the experiment.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    teams_colors : list, optional
        Team colors list
        
    Returns
    -------
    tuple
        (is_valid: bool, issues: list)
    """
    
    issues = []
    
    # Check video file
    if not os.path.exists(video_path):
        issues.append(f"Video file not found: {video_path}")
    else:
        # Check if file is readable
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                issues.append(f"Cannot open video file: {video_path}")
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    issues.append(f"Video appears to have no frames: {video_path}")
            cap.release()
        except Exception as e:
            issues.append(f"Error checking video file: {str(e)}")
    
    # Check required directories and files
    required_dirs = ['./Detect_and_Track', './Out']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Required directory not found: {dir_path}")
    
    required_files = [
        './Detect_and_Track/model_data/mars-small128.pb',
        './Detect_and_Track/model_data/coco/coco.names'
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            issues.append(f"Required model file not found: {file_path}")
    
    # Check team colors format
    if teams_colors is not None:
        if not isinstance(teams_colors, list) or len(teams_colors) != 6:
            issues.append("teams_colors must be a list of 6 color names")
    
    # Try importing required modules
    try:
        import torch
        import cv2
        import numpy as np
        import pandas as pd
    except ImportError as e:
        issues.append(f"Required Python package not available: {str(e)}")
    
    return len(issues) == 0, issues


def run_fair_roi_experiment(video_path, experiment_name, teams_colors=None, 
                           target_player_id=1, ball_only=True, save_roi_images=True, 
                           roi_save_interval=15):
    """
    Main function to run FAIR ROI tracking experiment and comparison.
    Both approaches now track only ONE player for fair comparison.
    
    Parameters
    ----------
    video_path : str
        Path to the video file to process
    experiment_name : str
        Name for this experiment (used in output files)
    teams_colors : list, optional
        List of team colors for classification. If None, uses default colors.
    target_player_id : int, optional
        ID of the player to track (default: 1)
    ball_only : bool, optional
        Whether to keep only frames with ball detected (default: True)
    save_roi_images : bool, optional
        Whether to save ROI images during tracking (default: True)
    roi_save_interval : int, optional
        Save ROI image every N frames (default: 15)
        
    Returns
    -------
    bool
        True if experiment completed successfully, False otherwise
    """
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("="*60)
        logger.info("STARTING FAIR ROI TRACKING EXPERIMENT")
        logger.info("Both approaches track ONLY ONE player (no ball detection)")
        logger.info("="*60)
        logger.info(f"Video path: {video_path}")
        logger.info(f"Experiment name: {experiment_name}")
        logger.info(f"Target player ID: {target_player_id}")
        logger.info(f"Ball only mode: {ball_only}")
        logger.info(f"Save ROI images: {save_roi_images}")
        logger.info(f"ROI save interval: {roi_save_interval}")
        
        # Validate setup
        is_valid, issues = validate_experiment_setup(video_path, teams_colors)
        if not is_valid:
            logger.error("Setup validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
            
        # Set default team colors if not provided
        if teams_colors is None:
            teams_colors = ['white', 'white', 'red', 'red', 'black', 'yellow']
            logger.info("Using default team colors: ['white', 'white', 'red', 'red', 'black', 'yellow']")
        else:
            logger.info(f"Using provided team colors: {teams_colors}")
        
        # Import required modules
        try:
            from Detect_and_Track.deep_sort.roi_tracking import compare_tracking_approaches_fair
            logger.info("Successfully imported FAIR ROI tracking modules")
        except ImportError as e:
            logger.error(f"Failed to import ROI tracking modules: {e}")
            return False
        
        # Run the FAIR comparison experiment
        logger.info("Starting FAIR tracking approaches comparison...")
        start_time = time.time()
        
        comparison_results = compare_tracking_approaches_fair(
            video_path=video_path,
            out_name=experiment_name,
            teams_colors=teams_colors,
            ball_only=ball_only,
            target_player_id=target_player_id,
            save_roi_images=save_roi_images,
            roi_save_interval=roi_save_interval
        )
        
        experiment_duration = time.time() - start_time
        
        if comparison_results:
            logger.info("="*60)
            logger.info("FAIR EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total experiment duration: {experiment_duration:.2f} seconds")
            
            # Log key results
            std_results = comparison_results['standard_approach_one_player']
            roi_results = comparison_results['roi_approach_one_player']
            improvements = comparison_results['improvements']
            
            logger.info("SUMMARY OF FAIR RESULTS:")
            logger.info(f"Standard approach (1 player) - Total time: {std_results['total_time']:.2f}s, "
                       f"Avg per frame: {std_results['avg_total_per_frame']:.6f}s")
            logger.info(f"ROI approach (1 player) - Total time: {roi_results['total_time']:.2f}s, "
                       f"Avg per frame: {roi_results['avg_total_per_frame']:.6f}s")
            logger.info(f"Performance improvement: {improvements['total_time_improvement_pct']:.2f}%")
            logger.info(f"Overall speedup: {improvements['overall_speedup']:.2f}x")
            logger.info(f"ROI frames used: {roi_results['roi_frames_used']}/{roi_results['frames_processed']} "
                       f"({(roi_results['roi_frames_used']/roi_results['frames_processed'])*100:.1f}%)")
            
            # List generated files
            logger.info("\nGENERATED FILES:")
            output_files = [
                f"./Out/{experiment_name}_standard_one_player_init_df.csv",
                f"./Out/{experiment_name}_roi_one_player_init_df.csv", 
                f"./Out/{experiment_name}_standard_one_player_out.mp4",
                f"./Out/{experiment_name}_standard_one_player_tracked.mp4",
                f"./Out/{experiment_name}_roi_one_player_out.mp4",
                f"./Out/{experiment_name}_roi_one_player_tracked.mp4",
                f"./Out/{experiment_name}_fair_tracking_comparison_report.txt"
            ]
            
            for file_path in output_files:
                if os.path.exists(file_path):
                    logger.info(f"[OK] {file_path}")
                else:
                    logger.warning(f"[W] {file_path} (not found)")
            
            # Check ROI images
            if save_roi_images:
                roi_images_dir = f"./Out/{experiment_name}_roi_roi_images"
                if os.path.exists(roi_images_dir):
                    roi_image_count = len([f for f in os.listdir(roi_images_dir) if f.endswith('.jpg')])
                    logger.info(f"[OK] ROI images: {roi_image_count} images in {roi_images_dir}")
                else:
                    logger.warning(f"[W] ROI images directory not found: {roi_images_dir}")
            
            logger.info(f"\nExperiment log saved to: {log_file}")
            logger.info("="*60)
            logger.info("FAIR COMPARISON: Both approaches tracked ONLY ONE player!")
            logger.info("Performance difference is due to ROI optimization, not workload difference.")
            logger.info("="*60)
            return True
            
        else:
            logger.error("Comparison function returned None - experiment failed")
            return False
            
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        logger.exception("Full error traceback:")
        return False


def run_roi_only_experiment_fair(video_path, experiment_name, teams_colors=None, 
                                target_player_id=1, ball_only=True, save_roi_images=True, 
                                roi_save_interval=15):
    """
    Run only ROI tracking for single player (without comparison to standard approach).
    Useful for quick testing of ROI parameters on single player tracking.
    
    Parameters
    ----------
    video_path : str
        Path to the video file to process
    experiment_name : str
        Name for this experiment
    teams_colors : list, optional
        List of team colors for classification
    target_player_id : int, optional
        ID of the player to track with ROI (default: 1)
    ball_only : bool, optional
        Whether to keep only frames with ball detected (default: True)
    save_roi_images : bool, optional
        Whether to save ROI images during tracking (default: True)
    roi_save_interval : int, optional
        Save ROI image every N frames (default: 15)
        
    Returns
    -------
    tuple
        (success: bool, results: dict or None)
    """
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Running ROI-only experiment (single player, no ball)...")
        
        # Set default team colors if not provided
        if teams_colors is None:
            teams_colors = ['white', 'white', 'red', 'red', 'black', 'yellow']
        
        # Import required modules
        from Detect_and_Track.deep_sort.roi_tracking import trackingXl5_ROI_single_player
        from Detect_and_Track.deep_sort.tracks_cleaning import clean_tracks
        from Detect_and_Track.deep_sort.tracking_video import make_tracking_video
        from Detect_and_Track.init_dataframe.create_df import creatInitDataFrame
        from Detect_and_Track.yoloV5.load_models import yoloV5l
        
        # Load models
        logger.info("Loading YOLO models...")
        modelv5l, ball_modelv5l = yoloV5l()
        
        # Run ROI tracking (single player)
        logger.info("Running ROI tracking (single player, no ball)...")
        start_time = time.time()
        iframes_roi, itboxes_roi, fps_roi = trackingXl5_ROI_single_player(
            modelv5l, ball_modelv5l, video_path, target_player_id, 
            save_roi_images=save_roi_images, roi_save_interval=roi_save_interval,
            experiment_name=experiment_name
        )
        roi_total_time = time.time() - start_time
        
        # Clean and process results
        logger.info("Cleaning tracks and creating outputs...")
        ziframes_roi, zitboxes_roi = clean_tracks(iframes_roi, itboxes_roi, ball_only)
        init_df_roi = creatInitDataFrame(zitboxes_roi, ziframes_roi, teams_colors)
        
        # Save outputs
        init_df_roi.to_csv(f'./Out/{experiment_name}_roi_only_single_player_init_df.csv', index=False)
        make_tracking_video(ziframes_roi, zitboxes_roi, f'./Out/{experiment_name}_roi_only_single_player_out.mp4', fps_roi, draw=False)
        make_tracking_video(ziframes_roi, zitboxes_roi, f'./Out/{experiment_name}_roi_only_single_player_tracked.mp4', fps_roi, draw=True)
        
        results = {
            'total_time': roi_total_time,
            'frames_processed': len(iframes_roi),
            'clean_frames': len(ziframes_roi),
            'fps': fps_roi,
            'avg_time_per_frame': roi_total_time / len(iframes_roi) if len(iframes_roi) > 0 else 0,
            'target_player_id': target_player_id
        }
        
        logger.info(f"ROI-only experiment (single player) completed in {roi_total_time:.2f} seconds")
        logger.info(f"Processed {len(iframes_roi)} frames, {len(ziframes_roi)} after cleaning")
        logger.info(f"Tracked player ID: {target_player_id}")
        
        return True, results
        
    except Exception as e:
        logger.error(f"ROI-only experiment failed: {str(e)}")
        return False, None


def run_standard_only_experiment(video_path, experiment_name, teams_colors=None, 
                                target_player_id=1, ball_only=True):
    """
    Run only standard tracking for single player (for testing purposes).
    
    Parameters
    ----------
    video_path : str
        Path to the video file to process
    experiment_name : str
        Name for this experiment
    teams_colors : list, optional
        List of team colors for classification
    target_player_id : int, optional
        ID of the player to track (default: 1)
    ball_only : bool, optional
        Whether to keep only frames with ball detected (default: True)
        
    Returns
    -------
    tuple
        (success: bool, results: dict or None)
    """
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Running standard-only experiment (single player, no ball)...")
        
        # Set default team colors if not provided
        if teams_colors is None:
            teams_colors = ['white', 'white', 'red', 'red', 'black', 'yellow']
        
        # Import required modules
        from Detect_and_Track.deep_sort.standard_one_player import standard_trackingXl5_one_player
        from Detect_and_Track.deep_sort.tracks_cleaning import clean_tracks
        from Detect_and_Track.deep_sort.tracking_video import make_tracking_video
        from Detect_and_Track.init_dataframe.create_df import creatInitDataFrame
        from Detect_and_Track.yoloV5.load_models import yoloV5l
        
        # Load models
        logger.info("Loading YOLO models...")
        modelv5l, ball_modelv5l = yoloV5l()
        
        # Run standard tracking (single player)
        logger.info("Running standard tracking (single player, no ball)...")
        start_time = time.time()
        iframes_std, itboxes_std, fps_std = standard_trackingXl5_one_player(
            modelv5l, ball_modelv5l, video_path, target_player_id,
            experiment_name=experiment_name
        )
        std_total_time = time.time() - start_time
        
        # Clean and process results
        logger.info("Cleaning tracks and creating outputs...")
        ziframes_std, zitboxes_std = clean_tracks(iframes_std, itboxes_std, ball_only)
        init_df_std = creatInitDataFrame(zitboxes_std, ziframes_std, teams_colors)
        
        # Save outputs
        init_df_std.to_csv(f'./Out/{experiment_name}_standard_only_single_player_init_df.csv', index=False)
        make_tracking_video(ziframes_std, zitboxes_std, f'./Out/{experiment_name}_standard_only_single_player_out.mp4', fps_std, draw=False)
        make_tracking_video(ziframes_std, zitboxes_std, f'./Out/{experiment_name}_standard_only_single_player_tracked.mp4', fps_std, draw=True)
        
        results = {
            'total_time': std_total_time,
            'frames_processed': len(iframes_std),
            'clean_frames': len(ziframes_std),
            'fps': fps_std,
            'avg_time_per_frame': std_total_time / len(iframes_std) if len(iframes_std) > 0 else 0,
            'target_player_id': target_player_id
        }
        
        logger.info(f"Standard-only experiment (single player) completed in {std_total_time:.2f} seconds")
        logger.info(f"Processed {len(iframes_std)} frames, {len(ziframes_std)} after cleaning")
        logger.info(f"Tracked player ID: {target_player_id}")
        
        return True, results
        
    except Exception as e:
        logger.error(f"Standard-only experiment failed: {str(e)}")
        return False, None


def quick_fair_roi_test(video_path, experiment_name="fair_test"):
    """
    Quick test function for immediate usage with fair comparison.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    experiment_name : str, optional
        Name for experiment outputs
        
    Returns
    -------
    bool
        Success status
    """
    
    print("Starting quick FAIR ROI test (both approaches track one player)...")
    
    # Validate setup
    is_valid, issues = validate_experiment_setup(video_path)
    if not is_valid:
        print("Setup validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Run fair experiment
    success = run_fair_roi_experiment(
        video_path=video_path,
        experiment_name=experiment_name,
        teams_colors=None,  # Use defaults
        target_player_id=1,
        ball_only=True,
        save_roi_images=True,
        roi_save_interval=10
    )
    
    if success:
        print(f"Quick FAIR test completed successfully!")
        print(f"Check ./Out directory for results with prefix '{experiment_name}'")
        print("Both approaches tracked only ONE player for fair comparison.")
    else:
        print("Quick fair test failed. Check logs for details.")
    
    return success


def run_parameter_sweep_fair(video_path, base_experiment_name, target_player_ids=[1, 2, 3]):
    """
    Run FAIR ROI experiments with different target player IDs to find optimal parameters.
    Both approaches track only one player each.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    base_experiment_name : str
        Base name for experiments
    target_player_ids : list
        List of player IDs to test
        
    Returns
    -------
    dict
        Results for each parameter combination
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting FAIR parameter sweep for player IDs: {target_player_ids}")
    logger.info("Each test compares standard vs ROI for ONE player only")
    
    results = {}
    
    for player_id in target_player_ids:
        experiment_name = f"{base_experiment_name}_fair_player_{player_id}"
        logger.info(f"\nTesting FAIR comparison with target player ID: {player_id}")
        
        success = run_fair_roi_experiment(
            video_path=video_path,
            experiment_name=experiment_name,
            target_player_id=player_id,
            ball_only=True,
            save_roi_images=False,  # Save space during sweep
            roi_save_interval=30
        )
        
        results[player_id] = {
            'experiment_name': experiment_name,
            'success': success,
            'comparison_type': 'fair_one_player'
        }
        
        if success:
            logger.info(f"FAIR Player ID {player_id} test completed successfully")
        else:
            logger.warning(f"FAIR Player ID {player_id} test failed")
    
    logger.info("FAIR parameter sweep completed")
    return results


# Main execution functions for easy importing
def main_fair_roi_experiment(video_path, experiment_name="fair_roi_test", target_player_id=1):
    """
    Main function for running a complete FAIR ROI experiment.
    Both approaches track only one player.
    """
    return run_fair_roi_experiment(
        video_path=video_path,
        experiment_name=experiment_name,
        target_player_id=target_player_id,
        ball_only=True,
        save_roi_images=True,
        roi_save_interval=15
    )


def main_roi_only_fair(video_path, experiment_name="roi_only_fair_test", target_player_id=1):
    """
    Main function for running ROI-only tracking (single player).
    """
    setup_logging()
    success, results = run_roi_only_experiment_fair(
        video_path=video_path,
        experiment_name=experiment_name,
        target_player_id=target_player_id,
        ball_only=True,
        save_roi_images=True,
        roi_save_interval=15
    )
    return success, results


def main_standard_only_fair(video_path, experiment_name="standard_only_fair_test", target_player_id=1):
    """
    Main function for running standard-only tracking (single player).
    """
    setup_logging()
    success, results = run_standard_only_experiment(
        video_path=video_path,
        experiment_name=experiment_name,
        target_player_id=target_player_id,
        ball_only=True
    )
    return success, results