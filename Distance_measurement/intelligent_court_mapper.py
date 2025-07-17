"""
Intelligent Court Coordinate Mapper
This class remaps all coordinates (including negative and out-of-bounds) to realistic court positions
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import cv2


class IntelligentCourtMapper:
    def __init__(self, court_width=10.97, court_length=23.77):
        self.court_width = court_width
        self.court_length = court_length
        
        # Define court zones for intelligent mapping
        self.court_zones = {
            'baseline_near': (0, 6.40),
            'service_area_near': (6.40, 11.885),
            'service_area_far': (11.885, 17.37),
            'baseline_far': (17.37, 23.77)
        }
        
        # Expected coordinate distribution based on tennis gameplay
        self.expected_distribution = {
            'baseline_zones': 0.4,  # 40% of play near baselines
            'service_zones': 0.3,   # 30% in service areas
            'net_zone': 0.2,        # 20% near net
            'out_of_bounds': 0.1    # 10% slightly out of bounds
        }

    def analyze_coordinate_distribution(self, df):
        """
        Analyze the current coordinate distribution to understand mapping issues
        """
        print("=== COORDINATE DISTRIBUTION ANALYSIS ===")
        
        x_coords = df['X_court'].values
        y_coords = df['Y_court'].values
        
        # Basic statistics
        x_stats = {
            'min': np.min(x_coords),
            'max': np.max(x_coords),
            'mean': np.mean(x_coords),
            'std': np.std(x_coords),
            'range': np.max(x_coords) - np.min(x_coords)
        }
        
        y_stats = {
            'min': np.min(y_coords),
            'max': np.max(y_coords),
            'mean': np.mean(y_coords),
            'std': np.std(y_coords),
            'range': np.max(y_coords) - np.min(y_coords)
        }
        
        print(f"X coordinates: min={x_stats['min']:.2f}, max={x_stats['max']:.2f}, "
              f"mean={x_stats['mean']:.2f}, std={x_stats['std']:.2f}")
        print(f"Y coordinates: min={y_stats['min']:.2f}, max={y_stats['max']:.2f}, "
              f"mean={y_stats['mean']:.2f}, std={y_stats['std']:.2f}")
        
        # Identify coordinate clusters for intelligent remapping
        valid_mask = (
            (x_coords > -100) & (x_coords < 100) &  # Remove extreme outliers
            (y_coords > -100) & (y_coords < 100)
        )
        
        if valid_mask.sum() > 10:
            coords = np.column_stack([x_coords[valid_mask], y_coords[valid_mask]])
            
            # Use clustering to identify main court area
            n_clusters = min(5, len(coords) // 20)  # Adaptive number of clusters
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(coords)
                
                print(f"\nIdentified {n_clusters} coordinate clusters:")
                for i in range(n_clusters):
                    cluster_coords = coords[clusters == i]
                    cluster_center = np.mean(cluster_coords, axis=0)
                    cluster_size = len(cluster_coords)
                    print(f"  Cluster {i}: center=({cluster_center[0]:.2f}, {cluster_center[1]:.2f}), "
                          f"size={cluster_size} points")
        
        return x_stats, y_stats

    def create_intelligent_mapping_function(self, df):
        """
        Create an intelligent mapping function that transforms coordinates to realistic court positions
        """
        print("\n=== CREATING INTELLIGENT MAPPING FUNCTION ===")
        
        x_coords = df['X_court'].values
        y_coords = df['Y_court'].values
        
        # Method 1: Linear rescaling based on data distribution
        def linear_rescale_mapping(x, y):
            # Find the main data cluster (remove extreme outliers)
            x_clean = x[(x > np.percentile(x, 5)) & (x < np.percentile(x, 95))]
            y_clean = y[(y > np.percentile(y, 5)) & (y < np.percentile(y, 95))]
            
            # Scale to court dimensions with some buffer
            x_scaler = MinMaxScaler(feature_range=(-2, self.court_width + 2))
            y_scaler = MinMaxScaler(feature_range=(-3, self.court_length + 3))
            
            x_scaled = x_scaler.fit_transform(x.reshape(-1, 1)).flatten()
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            return x_scaled, y_scaled
        
        # Method 2: Percentile-based mapping for robust scaling
        def percentile_mapping(x, y):
            # Use percentiles to handle outliers
            x_p5, x_p95 = np.percentile(x, [5, 95])
            y_p5, y_p95 = np.percentile(y, [5, 95])
            
            # Map percentile range to court dimensions
            x_mapped = np.interp(x, [x_p5, x_p95], [0, self.court_width])
            y_mapped = np.interp(y, [y_p5, y_p95], [0, self.court_length])
            
            return x_mapped, y_mapped
        
        # Method 3: Distribution-aware mapping
        def distribution_aware_mapping(x, y):
            # Analyze data distribution and map to expected tennis distribution
            
            # Find main playing area (central 80% of data)
            x_sorted = np.sort(x)
            y_sorted = np.sort(y)
            
            x_10, x_90 = np.percentile(x_sorted, [10, 90])
            y_10, y_90 = np.percentile(y_sorted, [10, 90])
            
            # Map core playing area to court with realistic margins
            x_core_mapped = np.interp(x, [x_10, x_90], [1, self.court_width - 1])
            y_core_mapped = np.interp(y, [y_10, y_90], [2, self.court_length - 2])
            
            # Handle outliers by extending mapping
            x_mapped = np.where(
                x < x_10,
                np.interp(x, [np.min(x), x_10], [-3, 1]),
                np.where(
                    x > x_90,
                    np.interp(x, [x_90, np.max(x)], [self.court_width - 1, self.court_width + 3]),
                    x_core_mapped
                )
            )
            
            y_mapped = np.where(
                y < y_10,
                np.interp(y, [np.min(y), y_10], [-4, 2]),
                np.where(
                    y > y_90,
                    np.interp(y, [y_90, np.max(y)], [self.court_length - 2, self.court_length + 4]),
                    y_core_mapped
                )
            )
            
            return x_mapped, y_mapped
        
        # Method 4: Physics-aware mapping (considers tennis movement patterns)
        def physics_aware_mapping(x, y):
            # Group points by object/player for trajectory-aware mapping
            if 'ID' in df.columns:
                x_mapped = np.zeros_like(x)
                y_mapped = np.zeros_like(y)
                
                for obj_id in df['ID'].unique():
                    obj_mask = df['ID'] == obj_id
                    obj_x = x[obj_mask]
                    obj_y = y[obj_mask]
                    
                    if len(obj_x) > 1:
                        # For each object, maintain relative movement patterns
                        # but scale to court dimensions
                        
                        # Calculate movement vectors
                        dx = np.diff(obj_x)
                        dy = np.diff(obj_y)
                        
                        # Scale movements to realistic court speeds
                        # Typical tennis movement: 0.1-2.0 meters per frame
                        movement_magnitudes = np.sqrt(dx**2 + dy**2)
                        
                        if len(movement_magnitudes) > 0:
                            # Scale extreme movements
                            reasonable_movement = np.percentile(movement_magnitudes, 90)
                            if reasonable_movement > 5:  # If movements are too large
                                scale_factor = 2.0 / reasonable_movement
                                dx *= scale_factor
                                dy *= scale_factor
                        
                        # Reconstruct positions starting from court center
                        start_x = self.court_width / 2
                        start_y = self.court_length / 2
                        
                        obj_x_new = np.cumsum(np.concatenate([[start_x], dx]))
                        obj_y_new = np.cumsum(np.concatenate([[start_y], dy]))
                        
                        x_mapped[obj_mask] = obj_x_new
                        y_mapped[obj_mask] = obj_y_new
                    else:
                        # Single point - place in court center
                        x_mapped[obj_mask] = self.court_width / 2
                        y_mapped[obj_mask] = self.court_length / 2
                
                return x_mapped, y_mapped
            else:
                # Fallback to distribution-aware mapping
                return distribution_aware_mapping(x, y)
        
        # Test all methods and choose the best one
        methods = {
            'linear_rescale': linear_rescale_mapping,
            'percentile': percentile_mapping,
            'distribution_aware': distribution_aware_mapping,
            'physics_aware': physics_aware_mapping
        }
        
        best_method = None
        best_score = float('inf')
        
        print("Testing mapping methods:")
        
        for method_name, method_func in methods.items():
            try:
                x_test, y_test = method_func(x_coords, y_coords)
                
                # Score based on how well it fits court dimensions
                x_range = np.max(x_test) - np.min(x_test)
                y_range = np.max(y_test) - np.min(y_test)
                
                # Penalties for being too far from expected court dimensions
                x_penalty = abs(x_range - self.court_width) / self.court_width
                y_penalty = abs(y_range - self.court_length) / self.court_length
                
                # Penalty for too many extreme outliers
                x_outliers = np.sum((x_test < -5) | (x_test > self.court_width + 5))
                y_outliers = np.sum((y_test < -5) | (y_test > self.court_length + 5))
                outlier_penalty = (x_outliers + y_outliers) / len(x_test)
                
                total_score = x_penalty + y_penalty + outlier_penalty
                
                print(f"  {method_name}: score={total_score:.3f}, "
                      f"x_range={x_range:.2f}, y_range={y_range:.2f}, "
                      f"outliers={outlier_penalty:.3f}")
                
                if total_score < best_score:
                    best_score = total_score
                    best_method = method_func
                    best_method_name = method_name
                    
            except Exception as e:
                print(f"  {method_name}: failed ({e})")
        
        if best_method:
            print(f"\nSelected method: {best_method_name} (score: {best_score:.3f})")
        
        return best_method, best_method_name

    def apply_intelligent_mapping(self, df):
        """
        Apply intelligent coordinate mapping to the dataframe
        """
        print("\n=== APPLYING INTELLIGENT MAPPING ===")
        
        # Analyze current distribution
        x_stats, y_stats = self.analyze_coordinate_distribution(df)
        
        # Create mapping function
        mapping_func, method_name = self.create_intelligent_mapping_function(df)
        
        if mapping_func is None:
            print("[ERROR] Could not create mapping function")
            return df
        
        # Apply mapping
        x_mapped, y_mapped = mapping_func(df['X_court'].values, df['Y_court'].values)
        
        # Create result dataframe
        df_mapped = df.copy()
        df_mapped['X_court_intelligent'] = x_mapped
        df_mapped['Y_court_intelligent'] = y_mapped
        
        # Add quality metrics
        df_mapped['in_court_bounds'] = (
            (x_mapped >= -3) & (x_mapped <= self.court_width + 3) &
            (y_mapped >= -4) & (y_mapped <= self.court_length + 4)
        )
        
        df_mapped['in_strict_bounds'] = (
            (x_mapped >= 0) & (x_mapped <= self.court_width) &
            (y_mapped >= 0) & (y_mapped <= self.court_length)
        )
        
        # Apply final smoothing and constraints
        x_final, y_final = self.apply_final_constraints(x_mapped, y_mapped, df_mapped)
        df_mapped['X_court_final'] = x_final
        df_mapped['Y_court_final'] = y_final
        
        # Add court zones
        df_mapped = self.add_court_zone_information(df_mapped)
        
        # Print results
        print(f"\nMapping results using {method_name}:")
        print(f"Original X range: {x_stats['min']:.2f} to {x_stats['max']:.2f}")
        print(f"Mapped X range:   {np.min(x_final):.2f} to {np.max(x_final):.2f}")
        print(f"Original Y range: {y_stats['min']:.2f} to {y_stats['max']:.2f}")
        print(f"Mapped Y range:   {np.min(y_final):.2f} to {np.max(y_final):.2f}")
        
        in_bounds = df_mapped['in_court_bounds'].sum()
        strictly_in_bounds = df_mapped['in_strict_bounds'].sum()
        print(f"Points in court bounds: {in_bounds}/{len(df_mapped)} ({in_bounds/len(df_mapped)*100:.1f}%)")
        print(f"Points strictly in court: {strictly_in_bounds}/{len(df_mapped)} ({strictly_in_bounds/len(df_mapped)*100:.1f}%)")
        
        return df_mapped

    def apply_final_constraints(self, x_mapped, y_mapped, df_mapped):
        """
        Apply final smoothing and physical constraints
        """
        x_final = x_mapped.copy()
        y_final = y_mapped.copy()
        
        # Smooth trajectories for each object
        if 'ID' in df_mapped.columns:
            for obj_id in df_mapped['ID'].unique():
                obj_mask = df_mapped['ID'] == obj_id
                obj_indices = np.where(obj_mask)[0]
                
                if len(obj_indices) > 3:  # Need at least 4 points for smoothing
                    obj_x = x_final[obj_mask]
                    obj_y = y_final[obj_mask]
                    obj_frames = df_mapped.loc[obj_mask, 'frame'].values
                    
                    # Sort by frame
                    sort_idx = np.argsort(obj_frames)
                    obj_x_sorted = obj_x[sort_idx]
                    obj_y_sorted = obj_y[sort_idx]
                    obj_frames_sorted = obj_frames[sort_idx]
                    
                    # Apply light smoothing (reduce jitter while preserving movement)
                    try:
                        # Use interpolation to smooth the trajectory
                        f_x = interpolate.UnivariateSpline(obj_frames_sorted, obj_x_sorted, s=0.5)
                        f_y = interpolate.UnivariateSpline(obj_frames_sorted, obj_y_sorted, s=0.5)
                        
                        obj_x_smooth = f_x(obj_frames_sorted)
                        obj_y_smooth = f_y(obj_frames_sorted)
                        
                        # Update final coordinates
                        x_final[obj_indices[sort_idx]] = obj_x_smooth
                        y_final[obj_indices[sort_idx]] = obj_y_smooth
                        
                    except Exception:
                        # If smoothing fails, use simple moving average
                        window = min(3, len(obj_x_sorted) // 2)
                        if window >= 1:
                            obj_x_smooth = np.convolve(obj_x_sorted, np.ones(window)/window, mode='same')
                            obj_y_smooth = np.convolve(obj_y_sorted, np.ones(window)/window, mode='same')
                            
                            x_final[obj_indices[sort_idx]] = obj_x_smooth
                            y_final[obj_indices[sort_idx]] = obj_y_smooth
        
        # Apply soft constraints (allow some out-of-bounds but penalize extreme values)
        buffer_soft = 4.0
        x_final = np.clip(x_final, -buffer_soft, self.court_width + buffer_soft)
        y_final = np.clip(y_final, -buffer_soft, self.court_length + buffer_soft)
        
        return x_final, y_final

    def add_court_zone_information(self, df_mapped):
        """
        Add semantic court zone information
        """
        df_result = df_mapped.copy()
        
        # Determine court zones based on Y coordinate
        y_coords = df_result['Y_court_final']
        
        conditions = [
            y_coords < 6.40,                           # Near baseline
            (y_coords >= 6.40) & (y_coords < 11.885), # Near service area
            (y_coords >= 11.885) & (y_coords < 17.37), # Far service area
            (y_coords >= 17.37) & (y_coords <= 23.77), # Far baseline
        ]
        
        choices = ['Near Baseline', 'Near Service', 'Far Service', 'Far Baseline']
        
        df_result['court_zone'] = np.select(conditions, choices, default='Out of Bounds')
        
        # Add side information
        x_coords = df_result['X_court_final']
        df_result['court_side'] = np.where(x_coords < self.court_width/2, 'Left', 'Right')
        
        return df_result

    def calculate_realistic_distances(self, df_mapped, fps=25.0):
        """
        Calculate distances using the intelligently mapped coordinates
        """
        print("\n=== CALCULATING REALISTIC DISTANCES ===")
        
        results = []
        
        for obj_id in df_mapped['ID'].unique():
            obj_data = df_mapped[df_mapped['ID'] == obj_id].sort_values('frame')
            
            if len(obj_data) < 2:
                continue
            
            # Use final mapped coordinates
            positions = obj_data[['X_court_final', 'Y_court_final']].values
            
            # Calculate frame-to-frame distances
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            
            # Calculate time intervals
            frame_intervals = np.diff(obj_data['frame'].values) / fps
            
            # Calculate speeds
            speeds_ms = distances / frame_intervals
            speeds_kmh = speeds_ms * 3.6
            
            # Apply realistic speed limits
            obj_class = obj_data['Class'].iloc[0] if 'Class' in obj_data.columns else 'Unknown'
            if obj_class == 'Ball':
                max_speed = 200  # km/h
                typical_speed = 80
            else:
                max_speed = 45   # km/h for players
                typical_speed = 15
            
            # Filter unrealistic speeds but be more lenient
            realistic_mask = speeds_kmh <= max_speed
            realistic_distances = distances[realistic_mask]
            realistic_speeds = speeds_kmh[realistic_mask]
            
            # Calculate statistics
            total_distance = realistic_distances.sum()
            avg_speed = realistic_speeds.mean() if len(realistic_speeds) > 0 else 0
            max_speed_actual = realistic_speeds.max() if len(realistic_speeds) > 0 else 0
            
            # Time span and coverage
            time_span = (obj_data['frame'].max() - obj_data['frame'].min()) / fps
            
            x_coverage = obj_data['X_court_final'].max() - obj_data['X_court_final'].min()
            y_coverage = obj_data['Y_court_final'].max() - obj_data['Y_court_final'].min()
            court_coverage = x_coverage * y_coverage
            
            # Quality metrics
            in_bounds_ratio = obj_data['in_court_bounds'].mean()
            
            results.append({
                'ID': obj_id,
                'Class': obj_class,
                'TotalDistance': total_distance,
                'AverageSpeed': avg_speed,
                'MaxSpeed': max_speed_actual,
                'TimeSpan': time_span,
                'CourtCoverage': court_coverage,
                'InBoundsRatio': in_bounds_ratio,
                'DataPoints': len(obj_data),
                'ValidMovements': len(realistic_distances)
            })
        
        return pd.DataFrame(results)


def intelligent_court_mapping_pipeline(csv_path, fps=25.0):
    """
    Complete pipeline for intelligent court coordinate mapping
    """
    print("=== INTELLIGENT COURT MAPPING PIPELINE ===")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tracking points")
    
    # Initialize intelligent mapper
    mapper = IntelligentCourtMapper()
    
    # Apply intelligent mapping
    df_mapped = mapper.apply_intelligent_mapping(df)
    
    # Calculate realistic distances
    distance_results = mapper.calculate_realistic_distances(df_mapped, fps)
    
    # Display results
    print("\n=== INTELLIGENT MAPPING RESULTS ===")
    active_objects = distance_results[distance_results['TotalDistance'] > 0.5]
    
    if len(active_objects) > 0:
        print("Active objects with significant movement:")
        for _, obj in active_objects.iterrows():
            print(f"\n{obj['Class']} {obj['ID']}:")
            print(f"  Total distance: {obj['TotalDistance']:.2f}m")
            print(f"  Average speed: {obj['AverageSpeed']:.1f} km/h")
            print(f"  Max speed: {obj['MaxSpeed']:.1f} km/h")
            print(f"  Time span: {obj['TimeSpan']:.1f}s")
            print(f"  Court coverage: {obj['CourtCoverage']:.2f}m²")
            print(f"  In bounds: {obj['InBoundsRatio']*100:.1f}%")
    
    # Save results
    output_path = csv_path.replace('.csv', '_intelligent_mapped.csv')
    df_mapped.to_csv(output_path, index=False)
    
    results_path = csv_path.replace('.csv', '_intelligent_distances.csv')
    distance_results.to_csv(results_path, index=False)
    
    print(f"\n[SAVE] Mapped data saved to: {output_path}")
    print(f"[SAVE] Distance results saved to: {results_path}")
    
    return df_mapped, distance_results


# Usage example
if __name__ == "__main__":
    csv_path = "./Out/match_point_init_df_with_video_court_coords.csv"
    df_mapped, results = intelligent_court_mapping_pipeline(csv_path, fps=25.12)