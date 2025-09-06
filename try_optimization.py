from Time_Optimization.resolution_optimizer import ResolutionPerformanceOptimizer

optimizer = ResolutionPerformanceOptimizer(target_fps=30.0, min_detection_threshold=2)

results = optimizer.optimize_resolution(
    video_path="./test_videos/melbourne2.mp4",
    out_name="tennis_test",
    teams_colors=['white', 'white', 'blue', 'blue', 'black', 'yellow'],
    ball_only=True,
    step_factor=0.8,
    min_width=320,
    test_percentage=30.0,              # Test 60% of resolutions
    prefer_higher_resolution=True      # Focus on higher quality
)

# Generate both reports
optimizer.generate_performance_report(save_path="./Out/optimization_report.txt")
optimizer.plot_performance_analysis(save_path="./Out/performance_plots.png")
optimizer.plot_comprehensive_timing_analysis(save_path="./Out/timing_analysis.png")