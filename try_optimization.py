from Time_Optimization.resolution_optimizer import ResolutionPerformanceOptimizer

# Initialize optimizer
optimizer = ResolutionPerformanceOptimizer(target_fps=30.0, min_detection_threshold=2)

# Run interactive optimization - this will show you all available resolutions
# and let you choose which ones to test
results = optimizer.optimize_resolution_interactive(
    video_path="./test_videos/melbourne2.mp4",
    out_name="IMGSZ_test",
    teams_colors=['white', 'white', 'blue', 'blue', 'black', 'yellow'],
    ball_only=True,
    step_factor=0.8,  # 20% reduction each step
    min_width=320     # Minimum width to generate
)

# Generate reports and plots
if results and 'error' not in results:
    print("\n" + "="*60)
    print("GENERATING REPORTS AND VISUALIZATIONS")
    print("="*60)
    
    # Generate comprehensive report
    optimizer.generate_performance_report(save_path="./Out/interactive_optimization_report.txt")
    
    # Create performance plots
    optimizer.plot_performance_analysis(save_path="./Out/interactive_performance_plots.png")
    
    # Create timing analysis plots
    optimizer.plot_comprehensive_timing_analysis(save_path="./Out/interactive_timing_analysis.png")
    
    # Print final recommendation
    if results.get('recommended_resolution'):
        rec = results['recommended_resolution']
        print(f"\n FINAL RECOMMENDATION: {rec['resolution_str']}")
        print(f"   Expected FPS: {rec.get('avg_fps', 0):.2f}")
        print(f"   Real-time capability: {rec.get('real_time_percentage', 0):.1f}%")
        print(f"   Average detections per frame: {rec.get('avg_detections_per_frame', 0):.1f}")
        print(f"   Ball detection rate: {rec.get('ball_detection_rate', 0):.1f}%")
    
    print(f"\nTesting completed! Check the ./Out/ folder for detailed reports and plots.")
else:
    print("Optimization was cancelled or encountered an error.")