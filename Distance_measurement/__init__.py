"""
Distance Measurement Module for Tennis Court Tracking

This module provides comprehensive tennis court tracking capabilities including:
- Corner detection from video frames
- Coordinate mapping using homography
- Player distance and movement analysis
- Ball trajectory and distance calculation
- Dynamic homography management
"""

from .corner_detection import CornerDetector
from .coordinate_mapper import CoordinateMapper
from .player_distance_calculator import PlayerDistanceCalculator
from .ball_distance_calculator import BallDistanceCalculator
from .homography_manager import HomographyManager
from .tennis_court_tracker import (
    TennisCourtTracker, 
    main_video_processing_pipeline,
    main_video_processing_pipeline_dynamic
)

__all__ = [
    'CornerDetector',
    'CoordinateMapper', 
    'PlayerDistanceCalculator',
    'BallDistanceCalculator',
    'HomographyManager',
    'TennisCourtTracker',
    'main_video_processing_pipeline',
    'main_video_processing_pipeline_dynamic'
]

__version__ = "1.0.0"
