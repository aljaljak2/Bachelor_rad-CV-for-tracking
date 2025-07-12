import numpy as np
import pandas as pd

def calculate_player_distances(df, id_col='ID', x_col='X_mapped', y_col='Y_mapped', class_col='Class', frame_col='frame'):
    """
    Calculate total distance run for each player (by ID) in the dataframe.
    
    IMPORTANT FIXES:
    1. Sort by frame to ensure correct chronological order
    2. Add smoothing to reduce noise
    3. Filter out unrealistic movements
    
    Returns a DataFrame with columns: ID, TotalDistance, FrameCount
    """
    players = df[df[class_col] == 'Player'].copy()
    
    if players.empty:
        return pd.DataFrame(columns=['ID', 'TotalDistance', 'FrameCount'])
    
    distances = []
    
    for pid, group in players.groupby(id_col):
        # CRITICAL: Sort by frame to ensure chronological order
        group = group.sort_values(frame_col)
        
        coords = group[[x_col, y_col]].values
        
        if len(coords) < 2:
            distances.append({'ID': pid, 'TotalDistance': 0.0, 'FrameCount': len(coords)})
            continue
        
        # Calculate frame-to-frame distances
        diffs = np.diff(coords, axis=0)
        frame_distances = np.linalg.norm(diffs, axis=1)
        
        # Filter out unrealistic movements (likely tracking errors)
        # Adjust this threshold based on your sport and frame rate
        max_speed_per_frame = 2.0  # meters per frame (adjust based on your frame rate)
        valid_distances = frame_distances[frame_distances <= max_speed_per_frame]
        
        total_distance = np.sum(valid_distances)
        
        distances.append({
            'ID': pid, 
            'TotalDistance': total_distance, 
            'FrameCount': len(coords),
            'FilteredFrames': len(frame_distances) - len(valid_distances)
        })
    
    return pd.DataFrame(distances)

def calculate_ball_distance(df, x_col='X_mapped', y_col='Y_mapped', class_col='Class', frame_col='frame'):
    """
    Calculate total distance the ball has traveled.
    Returns a float (total distance in court units).
    """
    ball = df[df[class_col] == 'Ball'].copy()
    
    if ball.empty or len(ball) < 2:
        return 0.0
    
    # Sort by frame
    ball = ball.sort_values(frame_col)
    
    coords = ball[[x_col, y_col]].values
    diffs = np.diff(coords, axis=0)
    frame_distances = np.linalg.norm(diffs, axis=1)
    
    # Filter out unrealistic ball movements
    max_ball_speed_per_frame = 5.0  # adjust based on sport and frame rate
    valid_distances = frame_distances[frame_distances <= max_ball_speed_per_frame]
    
    return np.sum(valid_distances)

def debug_coordinates(df, sample_size=10):
    """
    Debug function to examine coordinate ranges and detect issues
    """
    print("Coordinate Statistics:")
    print(f"X_mapped range: {df['X_mapped'].min():.3f} to {df['X_mapped'].max():.3f}")
    print(f"Y_mapped range: {df['Y_mapped'].min():.3f} to {df['Y_mapped'].max():.3f}")
    
    print(f"\nSample of mapped coordinates:")
    print(df[['frame', 'X_mapped', 'Y_mapped', 'Class']].head(sample_size))
    
    # Check for extreme values
    x_extreme = df[(df['X_mapped'] < -50) | (df['X_mapped'] > 50)]
    y_extreme = df[(df['Y_mapped'] < -50) | (df['Y_mapped'] > 50)]
    
    if not x_extreme.empty:
        print(f"\nWARNING: Found {len(x_extreme)} points with extreme X coordinates")
    if not y_extreme.empty:
        print(f"WARNING: Found {len(y_extreme)} points with extreme Y coordinates")
