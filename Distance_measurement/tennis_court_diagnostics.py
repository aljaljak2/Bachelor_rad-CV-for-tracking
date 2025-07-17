"""
Tennis Court Mapping Diagnostics and Correction Tool
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection


class TennisCourtDiagnostics:
    def __init__(self):
        self.court_width = 10.97  # meters
        self.court_length = 23.77  # meters
        
        # Define standard court coordinate system
        # Origin at bottom-left baseline, Y increases toward far baseline
        self.standard_corners = np.array([
            [0, 0],                          # Bottom-left baseline
            [self.court_width, 0],           # Bottom-right baseline  
            [self.court_width, self.court_length],  # Top-right baseline
            [0, self.court_length]           # Top-left baseline
        ], dtype=np.float32)
        
        # Key court lines for validation
        self.court_lines = {
            'baseline_near': 0,
            'service_line_near': 6.40,
            'net': 11.885,  # Court length / 2
            'service_line_far': 17.37,  # 23.77 - 6.40
            'baseline_far': 23.77,
            'center_line': 5.485,  # Court width / 2
            'singles_sideline_left': 1.37,
            'singles_sideline_right': 9.60,  # 10.97 - 1.37
            'doubles_sideline_left': 0,
            'doubles_sideline_right': 10.97
        }

    def analyze_corner_configuration(self, pixel_corners):
        """
        Analyze and suggest corrections for corner configuration
        """
        print("=== Corner Configuration Analysis ===")
        
        if len(pixel_corners) != 4:
            print(f"❌ Expected 4 corners, got {len(pixel_corners)}")
            return False
        
        corners = np.array(pixel_corners)
        print(f"Input corners: {corners}")
        
        # Check if corners form a reasonable quadrilateral
        centroid = np.mean(corners, axis=0)
        print(f"Centroid: {centroid}")
        
        # Calculate angles and distances
        for i, corner in enumerate(corners):
            next_corner = corners[(i + 1) % 4]
            distance = np.linalg.norm(next_corner - corner)
            print(f"Side {i}-{(i+1)%4}: distance = {distance:.1f} pixels")
        
        # Suggest proper ordering based on image coordinates
        # In image coordinates: (0,0) is top-left, Y increases downward
        ordered_corners = self.order_corners_for_tennis_court(corners)
        print(f"\nSuggested corner order:")
        labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        for i, (corner, label) in enumerate(zip(ordered_corners, labels)):
            print(f"{label}: {corner}")
        
        return ordered_corners

    def order_corners_for_tennis_court(self, corners):
        """
        Order corners specifically for tennis court mapping
        """
        corners = np.array(corners)
        
        # Sort by y-coordinate (top to bottom in image coordinates)
        y_sorted = corners[np.argsort(corners[:, 1])]
        
        # Top two corners (smaller y values - closer to net in tennis view)
        top_two = y_sorted[:2]
        top_left = top_two[np.argmin(top_two[:, 0])]  # Leftmost of top two
        top_right = top_two[np.argmax(top_two[:, 0])]  # Rightmost of top two
        
        # Bottom two corners (larger y values - baseline in tennis view)
        bottom_two = y_sorted[2:]
        bottom_left = bottom_two[np.argmin(bottom_two[:, 0])]  # Leftmost of bottom two
        bottom_right = bottom_two[np.argmax(bottom_two[:, 0])]  # Rightmost of bottom two
        
        # Return in order: BL, BR, TR, TL (matches standard_corners)
        return np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float32)

    def compute_robust_homography(self, pixel_corners):
        """
        Compute homography with multiple validation steps
        """
        print("\n=== Homography Computation ===")
        
        # Order corners properly
        ordered_corners = self.order_corners_for_tennis_court(pixel_corners)
        print(f"Ordered pixel corners: {ordered_corners}")
        print(f"Target court corners: {self.standard_corners}")
        
        # Compute homography
        H, mask = cv2.findHomography(
            ordered_corners, 
            self.standard_corners,
            cv2.RANSAC,
            5.0
        )
        
        if H is None:
            print("❌ Failed to compute homography")
            return None, None
        
        print("✅ Homography computed successfully")
        
        # Validate by transforming corners back
        transformed_corners = cv2.perspectiveTransform(
            ordered_corners.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        
        print("\nValidation - Corner transformation:")
        labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        max_error = 0
        
        for i, (label, expected, actual) in enumerate(zip(labels, self.standard_corners, transformed_corners)):
            error = np.linalg.norm(actual - expected)
            max_error = max(max_error, error)
            print(f"{label}: Expected {expected} → Got {actual} (Error: {error:.3f}m)")
        
        if max_error > 1.0:
            print(f"⚠️  High transformation error: {max_error:.3f}m")
        else:
            print(f"✅ Transformation error acceptable: {max_error:.3f}m")
        
        return H, ordered_corners

    def diagnose_coordinate_issues(self, df):
        """
        Diagnose issues with current coordinate mapping
        """
        print("\n=== Coordinate Mapping Diagnosis ===")
        
        x_coords = df['X_court'].dropna()
        y_coords = df['Y_court'].dropna()
        
        print(f"Current coordinate ranges:")
        print(f"X: {x_coords.min():.3f} to {x_coords.max():.3f} (range: {x_coords.max() - x_coords.min():.3f})")
        print(f"Y: {y_coords.min():.3f} to {y_coords.max():.3f} (range: {y_coords.max() - y_coords.min():.3f})")
        
        # Expected ranges
        print(f"\nExpected coordinate ranges:")
        print(f"X: 0 to {self.court_width} (range: {self.court_width})")
        print(f"Y: 0 to {self.court_length} (range: {self.court_length})")
        
        # Identify issues
        issues = []
        
        if x_coords.min() < -5:
            issues.append(f"Large negative X values (min: {x_coords.min():.3f})")
        if y_coords.min() < -5:
            issues.append(f"Large negative Y values (min: {y_coords.min():.3f})")
        if x_coords.max() > self.court_width + 5:
            issues.append(f"X values too large (max: {x_coords.max():.3f})")
        if y_coords.max() > self.court_length + 5:
            issues.append(f"Y values too large (max: {y_coords.max():.3f})")
        if (x_coords.max() - x_coords.min()) > 50:
            issues.append(f"X range too large ({x_coords.max() - x_coords.min():.3f})")
        if (y_coords.max() - y_coords.min()) > 50:
            issues.append(f"Y range too large ({y_coords.max() - y_coords.min():.3f})")
        
        if issues:
            print("\n❌ Issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ Coordinates look reasonable")
        
        return issues

    def correct_mapping(self, df, new_homography):
        """
        Apply corrected homography mapping
        """
        print("\n=== Applying Corrected Mapping ===")
        
        # Extract pixel coordinates
        pixel_coords = df[['X', 'Y']].values.astype(np.float32)
        
        # Apply homography transformation
        court_coords = cv2.perspectiveTransform(
            pixel_coords.reshape(-1, 1, 2), new_homography
        ).reshape(-1, 2)
        
        # Create corrected dataframe
        df_corrected = df.copy()
        df_corrected['X_court_corrected'] = court_coords[:, 0]
        df_corrected['Y_court_corrected'] = court_coords[:, 1]
        
        # Add outlier detection
        x_outliers = (court_coords[:, 0] < -2) | (court_coords[:, 0] > self.court_width + 2)
        y_outliers = (court_coords[:, 1] < -2) | (court_coords[:, 1] > self.court_length + 2)
        df_corrected['is_outlier'] = x_outliers | y_outliers
        
        # Statistics
        x_corr = df_corrected['X_court_corrected']
        y_corr = df_corrected['Y_court_corrected']
        
        print(f"Corrected coordinate ranges:")
        print(f"X: {x_corr.min():.3f} to {x_corr.max():.3f}")
        print(f"Y: {y_corr.min():.3f} to {y_corr.max():.3f}")
        
        outlier_count = df_corrected['is_outlier'].sum()
        print(f"Outliers detected: {outlier_count} ({outlier_count/len(df_corrected)*100:.1f}%)")
        
        return df_corrected

    def visualize_court_mapping(self, df, title="Tennis Court Mapping"):
        """
        Create visualization of court mapping
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original coordinates
        ax1.scatter(df['X_court'], df['Y_court'], c='red', alpha=0.6, s=20)
        ax1.set_title('Original Mapping')
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Corrected coordinates (if available)
        if 'X_court_corrected' in df.columns:
            # Plot court boundaries
            court_rect = Rectangle((0, 0), self.court_width, self.court_length, 
                                 linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.3)
            ax2.add_patch(court_rect)
            
            # Plot court lines
            # Net line
            ax2.plot([0, self.court_width], [self.court_lines['net'], self.court_lines['net']], 
                    'k-', linewidth=2, label='Net')
            
            # Service lines
            ax2.plot([0, self.court_width], 
                    [self.court_lines['service_line_near'], self.court_lines['service_line_near']], 
                    'b--', linewidth=1, label='Service Lines')
            ax2.plot([0, self.court_width], 
                    [self.court_lines['service_line_far'], self.court_lines['service_line_far']], 
                    'b--', linewidth=1)
            
            # Center line
            ax2.plot([self.court_lines['center_line'], self.court_lines['center_line']], 
                    [0, self.court_length], 'g--', linewidth=1, label='Center Line')
            
            # Singles sidelines
            ax2.plot([self.court_lines['singles_sideline_left'], self.court_lines['singles_sideline_left']], 
                    [0, self.court_length], 'r:', linewidth=1, label='Singles Lines')
            ax2.plot([self.court_lines['singles_sideline_right'], self.court_lines['singles_sideline_right']], 
                    [0, self.court_length], 'r:', linewidth=1)
            
            # Plot points
            non_outliers = df[~df['is_outlier']] if 'is_outlier' in df.columns else df
            outliers = df[df['is_outlier']] if 'is_outlier' in df.columns else pd.DataFrame()
            
            if len(non_outliers) > 0:
                ax2.scatter(non_outliers['X_court_corrected'], non_outliers['Y_court_corrected'], 
                           c='blue', alpha=0.6, s=20, label='Valid Points')
            
            if len(outliers) > 0:
                ax2.scatter(outliers['X_court_corrected'], outliers['Y_court_corrected'], 
                           c='red', alpha=0.8, s=30, marker='x', label='Outliers')
            
            ax2.set_title('Corrected Mapping')
            ax2.set_xlabel('X (meters)')
            ax2.set_ylabel('Y (meters)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
            ax2.set_xlim(-2, self.court_width + 2)
            ax2.set_ylim(-2, self.court_length + 2)
        
        plt.tight_layout()
        plt.show()

    def generate_correction_report(self, df_original, df_corrected):
        """
        Generate a comprehensive correction report
        """
        print("\n" + "="*60)
        print("TENNIS COURT MAPPING CORRECTION REPORT")
        print("="*60)
        
        # Before/After comparison
        print("\n1. COORDINATE RANGE COMPARISON")
        print("-" * 40)
        
        orig_x = df_original['X_court']
        orig_y = df_original['Y_court']
        corr_x = df_corrected['X_court_corrected']
        corr_y = df_corrected['Y_court_corrected']
        
        print(f"{'Metric':<20} {'Original':<15} {'Corrected':<15} {'Expected':<15}")
        print("-" * 65)
        print(f"{'X Range':<20} {orig_x.max()-orig_x.min():<15.2f} {corr_x.max()-corr_x.min():<15.2f} {self.court_width:<15.2f}")
        print(f"{'Y Range':<20} {orig_y.max()-orig_y.min():<15.2f} {corr_y.max()-corr_y.min():<15.2f} {self.court_length:<15.2f}")
        print(f"{'X Min':<20} {orig_x.min():<15.2f} {corr_x.min():<15.2f} {'~0':<15}")
        print(f"{'X Max':<20} {orig_x.max():<15.2f} {corr_x.max():<15.2f} {self.court_width:<15.2f}")
        print(f"{'Y Min':<20} {orig_y.min():<15.2f} {corr_y.min():<15.2f} {'~0':<15}")
        print(f"{'Y Max':<20} {orig_y.max():<15.2f} {corr_y.max():<15.2f} {self.court_length:<15.2f}")
        
        # Outlier analysis
        print("\n2. OUTLIER ANALYSIS")
        print("-" * 40)
        
        if 'is_outlier' in df_corrected.columns:
            outlier_count = df_corrected['is_outlier'].sum()
            total_count = len(df_corrected)
            outlier_percentage = (outlier_count / total_count) * 100
            
            print(f"Total points: {total_count}")
            print(f"Outliers: {outlier_count} ({outlier_percentage:.1f}%)")
            print(f"Valid points: {total_count - outlier_count} ({100 - outlier_percentage:.1f}%)")
        
        # Class-wise analysis
        print("\n3. CLASS-WISE ANALYSIS")
        print("-" * 40)
        
        if 'Class' in df_corrected.columns:
            for class_name in df_corrected['Class'].unique():
                class_data = df_corrected[df_corrected['Class'] == class_name]
                class_outliers = class_data['is_outlier'].sum() if 'is_outlier' in class_data.columns else 0
                class_valid = len(class_data) - class_outliers
                
                print(f"\n{class_name}:")
                print(f"  Total points: {len(class_data)}")
                print(f"  Valid points: {class_valid}")
                print(f"  Outliers: {class_outliers}")
                print(f"  X range: {class_data['X_court_corrected'].min():.2f} to {class_data['X_court_corrected'].max():.2f}")
                print(f"  Y range: {class_data['Y_court_corrected'].min():.2f} to {class_data['Y_court_corrected'].max():.2f}")
        
        # Recommendations
        print("\n4. RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        if outlier_count > total_count * 0.1:
            recommendations.append("High outlier percentage - consider reviewing corner detection")
        
        if corr_x.min() < -1 or corr_x.max() > self.court_width + 1:
            recommendations.append("X coordinates still outside expected range - check corner ordering")
        
        if corr_y.min() < -1 or corr_y.max() > self.court_length + 1:
            recommendations.append("Y coordinates still outside expected range - verify court orientation")
        
        if abs((corr_x.max() - corr_x.min()) - self.court_width) > 2:
            recommendations.append("X range doesn't match court width - homography may be distorted")
        
        if abs((corr_y.max() - corr_y.min()) - self.court_length) > 2:
            recommendations.append("Y range doesn't match court length - homography may be distorted")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ Mapping appears to be working correctly!")
        
        return df_corrected


def fix_tennis_mapping(csv_file, pixel_corners):
    """
    Complete pipeline to diagnose and fix tennis court mapping issues
    
    Args:
        csv_file: Path to CSV file with tracking data
        pixel_corners: List of 4 corner points from your image
                      Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        Corrected DataFrame with proper court coordinates
    """
    print("🎾 TENNIS COURT MAPPING CORRECTION PIPELINE")
    print("="*50)
    
    # Initialize diagnostics
    diagnostics = TennisCourtDiagnostics()
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} tracking points")
    
    # Step 1: Analyze corner configuration
    ordered_corners = diagnostics.analyze_corner_configuration(pixel_corners)
    if ordered_corners is None:
        print("❌ Failed to analyze corners")
        return None
    
    # Step 2: Diagnose current mapping issues
    issues = diagnostics.diagnose_coordinate_issues(df)
    
    # Step 3: Compute corrected homography
    H_corrected, final_corners = diagnostics.compute_robust_homography(pixel_corners)
    if H_corrected is None:
        print("❌ Failed to compute corrected homography")
        return None
    
    # Step 4: Apply corrected mapping
    df_corrected = diagnostics.correct_mapping(df, H_corrected)
    
    # Step 5: Generate report
    df_final = diagnostics.generate_correction_report(df, df_corrected)
    
    # Step 6: Visualize results (optional - requires matplotlib)
    # diagnostics.visualize_court_mapping(df_final, "Corrected Tennis Court Mapping")
    
    print("\n✅ Mapping correction completed!")
    print("Use the 'X_court_corrected' and 'Y_court_corrected' columns for analysis")
    
    return df_final, H_corrected, final_corners


# Example usage based on your image
def example_usage():
    """
    Example of how to use the correction pipeline with your data
    """
    
    # Your corner coordinates from the image (you'll need to provide these)
    # Based on your image, these should be the pixel coordinates of the court corners
    # Order them as: Bottom-Left, Bottom-Right, Top-Right, Top-Left
    pixel_corners = [
        (436.9, 615.0),  # Bottom-Left (BL)
        (1223.5, 621.0), # Bottom-Right (BR)  
        (890.5, 107.1),  # Top-Right (TR)
        (375.4, 227.4)   # Top-Left (TL)
    ]
    
    # Run the correction pipeline
    df_corrected, homography, corners = fix_tennis_mapping(
        'match_point_init_df_with_video_court_coords.csv', 
        pixel_corners
    )
    
    return df_corrected, homography, corners