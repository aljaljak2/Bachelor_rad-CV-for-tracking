import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add your project imports here
try:
    from Distance_measurement.main_court_tracker import *
    from Distance_measurement.corner_detection import *
    from Detect_and_Track.get_init_data import get_init_data
    from Detect_and_Track.get_tracks import get_video_tracks
    from Detect_and_Track.create_tracking_boxes_video import create_tracking_boxes_video
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")

class TennisTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Court Tracking System v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Application state
        self.video_path = None
        self.output_name = ""
        self.init_df_path = ""
        self.results = None
        self.processing = False
        
        # Create main layout
        self.setup_styles()
        self.create_menu()
        self.create_main_layout()
        
    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Section.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        style.configure('Process.TButton', font=('Arial', 10, 'bold'))
        
    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Video", command=self.select_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Corner Detection Test", command=self.open_corner_test)
        tools_menu.add_command(label="View Results", command=self.view_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_layout(self):
        """Create the main application layout"""
        # Create main container with padding
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_container, text="Tennis Court Tracking System", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        self.create_control_panel(main_container)
        
        # Right panel - Preview and Results
        self.create_preview_panel(main_container)
        
        # Bottom panel - Progress and Logs
        self.create_progress_panel(main_container)
        
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.configure(width=400)
        
        # Video Input Section
        ttk.Label(control_frame, text="Video Input", style='Section.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Video selection
        ttk.Button(control_frame, text="📁 Select Video", command=self.select_video, width=20).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.video_label = ttk.Label(control_frame, text="No video selected", foreground='gray')
        self.video_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Output name
        ttk.Label(control_frame, text="Output Name:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.output_name_var = tk.StringVar(value="tennis_analysis")
        ttk.Entry(control_frame, textvariable=self.output_name_var, width=25).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Team Colors Section
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        ttk.Label(control_frame, text="Team Configuration", style='Section.TLabel').grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Team colors
        ttk.Label(control_frame, text="Team 1 Color:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.team1_color = ttk.Combobox(control_frame, values=["red", "blue", "white", "black", "green", "yellow"], width=12)
        self.team1_color.set("red")
        self.team1_color.grid(row=5, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(control_frame, text="Team 2 Color:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.team2_color = ttk.Combobox(control_frame, values=["red", "blue", "white", "black", "green", "yellow"], width=12)
        self.team2_color.set("blue")
        self.team2_color.grid(row=6, column=1, sticky=tk.W, padx=(10, 0))
        
        # Ball only option
        self.ball_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Track ball only", variable=self.ball_only_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Processing Section
        ttk.Separator(control_frame, orient='horizontal').grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        ttk.Label(control_frame, text="Processing Pipeline", style='Section.TLabel').grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Step 1: Object Detection & Tracking
        step1_frame = ttk.Frame(control_frame)
        step1_frame.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(step1_frame, text="1. Object Detection & Tracking").grid(row=0, column=0, sticky=tk.W)
        self.tracking_button = ttk.Button(step1_frame, text="Start Tracking", command=self.run_tracking, style='Process.TButton')
        self.tracking_button.grid(row=0, column=1, padx=(10, 0))
        self.tracking_status = ttk.Label(step1_frame, text="⏸ Ready", foreground='gray')
        self.tracking_status.grid(row=0, column=2, padx=(10, 0))
        
        # Step 2: Court Analysis
        step2_frame = ttk.Frame(control_frame)
        step2_frame.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(step2_frame, text="2. Court Analysis").grid(row=0, column=0, sticky=tk.W)
        
        # Pipeline selection
        self.pipeline_var = tk.StringVar(value="standard")
        ttk.Radiobutton(step2_frame, text="Standard", variable=self.pipeline_var, value="standard").grid(row=1, column=0, sticky=tk.W, padx=(20, 0))
        ttk.Radiobutton(step2_frame, text="Dynamic", variable=self.pipeline_var, value="dynamic").grid(row=1, column=1, sticky=tk.W)
        
        self.analysis_button = ttk.Button(step2_frame, text="Start Analysis", command=self.run_analysis, style='Process.TButton')
        self.analysis_button.grid(row=0, column=1, padx=(10, 0))
        self.analysis_status = ttk.Label(step2_frame, text="⏸ Ready", foreground='gray')
        self.analysis_status.grid(row=0, column=2, padx=(10, 0))
        
        # Quick Actions
        ttk.Separator(control_frame, orient='horizontal').grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        ttk.Label(control_frame, text="Quick Actions", style='Section.TLabel').grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=14, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(action_frame, text="🎬 Create Tracking Video", command=self.create_tracking_video).grid(row=0, column=0, pady=2, sticky=tk.W)
        ttk.Button(action_frame, text="📊 View Statistics", command=self.view_stats).grid(row=1, column=0, pady=2, sticky=tk.W)
        ttk.Button(action_frame, text="📁 Open Output Folder", command=self.open_output_folder).grid(row=2, column=0, pady=2, sticky=tk.W)
        
    def create_preview_panel(self, parent):
        """Create the right preview panel"""
        preview_frame = ttk.LabelFrame(parent, text="Preview & Results", padding="10")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        
        # Video preview
        self.preview_label = ttk.Label(preview_frame, text="Video preview will appear here\nSelect a video to get started", 
                                     anchor='center', background='#ecf0f1', relief='sunken')
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.preview_label.configure(width=50)
        
        # Results notebook
        self.results_notebook = ttk.Notebook(preview_frame)
        self.results_notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.stats_frame, text="Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(self.stats_frame, wrap=tk.WORD, height=15)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.stats_frame.columnconfigure(0, weight=1)
        self.stats_frame.rowconfigure(0, weight=1)
        
        # Files tab
        self.files_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.files_frame, text="Output Files")
        
        self.files_tree = ttk.Treeview(self.files_frame, columns=('Size', 'Modified'), show='tree headings')
        self.files_tree.heading('#0', text='File Name')
        self.files_tree.heading('Size', text='Size')
        self.files_tree.heading('Modified', text='Modified')
        self.files_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Files scrollbar
        files_scrollbar = ttk.Scrollbar(self.files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        files_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.files_tree.configure(yscrollcommand=files_scrollbar.set)
        
        self.files_frame.columnconfigure(0, weight=1)
        self.files_frame.rowconfigure(0, weight=1)
        
    def create_progress_panel(self, parent):
        """Create the bottom progress panel"""
        progress_frame = ttk.LabelFrame(parent, text="Progress & Logs", padding="10")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to start processing...")
        self.progress_label.grid(row=0, column=1, padx=(10, 0))
        
        # Log output
        self.log_text = scrolledtext.ScrolledText(progress_frame, wrap=tk.WORD, height=8)
        self.log_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def select_video(self):
        """Handle video file selection"""
        file_path = filedialog.askopenfilename(
            title="Select Tennis Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.video_label.config(text=f"📹 {filename}", foreground='black')
            
            # Auto-generate output name from filename
            base_name = os.path.splitext(filename)[0]
            self.output_name_var.set(base_name)
            
            # Load video preview
            self.load_video_preview()
            
            self.log_message(f"Video selected: {filename}")
    
    def load_video_preview(self):
        """Load and display video preview"""
        if not self.video_path:
            return
            
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize frame for preview
                height, width = frame.shape[:2]
                max_width, max_height = 400, 250
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to RGB and then to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo  # Keep a reference
                
        except Exception as e:
            self.log_message(f"Error loading video preview: {e}", "ERROR")
    
    def run_tracking(self):
        """Run object detection and tracking in a separate thread"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return
            
        if self.processing:
            messagebox.showinfo("Info", "Processing is already running.")
            return
            
        # Start tracking in a separate thread
        thread = threading.Thread(target=self._tracking_worker)
        thread.daemon = True
        thread.start()
    
    def _tracking_worker(self):
        """Worker thread for tracking process"""
        try:
            self.processing = True
            self.update_ui_state()
            
            self.output_name = self.output_name_var.get() or "tennis_analysis"
            teams_colors = [self.team1_color.get(), self.team2_color.get()]
            ball_only = self.ball_only_var.get()
            
            self.progress_var.set(0)
            self.update_progress("Starting object detection and tracking...", 10)
            
            # Step 1: Generate initial data
            self.log_message("Starting object detection and tracking...")
            get_init_data(self.video_path, self.output_name, teams_colors, ball_only)
            self.update_progress("Object detection completed", 30)
            
            # Step 2: Get tracks
            self.log_message("Processing object tracks...")
            get_video_tracks(self.video_path, self.output_name)
            self.update_progress("Object tracking completed", 60)
            
            # Step 3: Create tracking video
            self.log_message("Creating tracking visualization video...")
            create_tracking_boxes_video(self.video_path, self.output_name)
            self.update_progress("Tracking video created", 100)
            
            self.init_df_path = f"./Out/{self.output_name}_init_df.csv"
            
            self.log_message("Object detection and tracking completed successfully!", "SUCCESS")
            self.root.after(0, lambda: self.tracking_status.config(text="✅ Complete", foreground='green'))
            self.root.after(0, self.refresh_file_list)
            
        except Exception as e:
            error_msg = f"Error in tracking process: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.root.after(0, lambda: self.tracking_status.config(text="❌ Error", foreground='red'))
        finally:
            self.processing = False
            self.root.after(0, self.update_ui_state)
    
    def run_analysis(self):
        """Run court analysis pipeline"""
        if not self.init_df_path or not os.path.exists(self.init_df_path):
            messagebox.showwarning("Warning", "Please run object tracking first.")
            return
            
        if self.processing:
            messagebox.showinfo("Info", "Processing is already running.")
            return
            
        # Start analysis in a separate thread
        thread = threading.Thread(target=self._analysis_worker)
        thread.daemon = True
        thread.start()
    
    def _analysis_worker(self):
        """Worker thread for analysis process"""
        try:
            self.processing = True
            self.update_ui_state()
            
            self.progress_var.set(0)
            self.update_progress("Starting court analysis...", 10)
            
            if self.pipeline_var.get() == "dynamic":
                self.log_message("Running dynamic homography pipeline...")
                self.update_progress("Processing with dynamic homography...", 30)
                results = main_video_processing_pipeline_dynamic(
                    video_path=self.video_path,
                    data_csv_path=self.init_df_path
                )
            else:
                self.log_message("Running standard homography pipeline...")
                self.update_progress("Processing with standard homography...", 30)
                results = main_video_processing_pipeline(
                    video_path=self.video_path,
                    data_csv_path=self.init_df_path
                )
            
            self.update_progress("Analysis completed", 100)
            self.results = results
            
            self.log_message("Court analysis completed successfully!", "SUCCESS")
            self.root.after(0, lambda: self.analysis_status.config(text="✅ Complete", foreground='green'))
            self.root.after(0, self.display_results)
            self.root.after(0, self.refresh_file_list)
            
        except Exception as e:
            error_msg = f"Error in analysis process: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.root.after(0, lambda: self.analysis_status.config(text="❌ Error", foreground='red'))
        finally:
            self.processing = False
            self.root.after(0, self.update_ui_state)
    
    def create_tracking_video(self):
        """Create tracking video with bounding boxes"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return
            
        try:
            self.log_message("Creating tracking video...")
            create_tracking_boxes_video(self.video_path, self.output_name or "tracking_video")
            self.log_message("Tracking video created successfully!", "SUCCESS")
            self.refresh_file_list()
        except Exception as e:
            self.log_message(f"Error creating tracking video: {e}", "ERROR")
    
    def view_stats(self):
        """Display statistics in the results panel"""
        if not self.results:
            messagebox.showinfo("Info", "No analysis results available. Please run court analysis first.")
            return
            
        self.display_results()
        self.results_notebook.select(0)  # Switch to statistics tab
    
    def display_results(self):
        """Display analysis results in the statistics tab"""
        if not self.results:
            return
            
        try:
            df_mapped, player_distances, ball_results, average_corners, frame_info = self.results
            
            stats_text = "=== TENNIS COURT ANALYSIS RESULTS ===\n\n"
            
            # Player Statistics
            stats_text += "PLAYER MOVEMENT ANALYSIS:\n"
            stats_text += "-" * 40 + "\n"
            
            if not player_distances.empty:
                for _, player in player_distances.iterrows():
                    stats_text += f"\nPlayer {player['ID']}:\n"
                    stats_text += f"  Total Distance: {player['TotalDistance']:.1f}m\n"
                    stats_text += f"  Average Speed: {player.get('AverageSpeedKmh', 0):.1f} km/h\n"
                    stats_text += f"  Max Speed: {player.get('MaxSpeedKmh', 0):.1f} km/h\n"
                    stats_text += f"  Active Time: {player.get('TimeActive', 0):.1f}s\n"
                    stats_text += f"  Court Coverage: {player.get('CourtCoverage', 0):.1f}m²\n"
            else:
                stats_text += "No player movement data available.\n"
            
            # Ball Statistics
            stats_text += "\n\nBALL MOVEMENT ANALYSIS:\n"
            stats_text += "-" * 40 + "\n"
            if ball_results:
                stats_text += f"Total Distance: {ball_results.get('total_distance', 0):.1f}m\n"
                stats_text += f"Average Speed: {ball_results.get('average_speed_kmh', 0):.1f} km/h\n"
                stats_text += f"Max Speed: {ball_results.get('max_speed_kmh', 0):.1f} km/h\n"
                stats_text += f"Detection Gaps: {ball_results.get('detection_gaps', 0)}\n"
                stats_text += f"Valid Points: {ball_results.get('valid_points', 0)}\n"
            else:
                stats_text += "No ball movement data available.\n"
            
            # Corner Detection Statistics
            stats_text += "\n\nCORNER DETECTION ANALYSIS:\n"
            stats_text += "-" * 40 + "\n"
            if frame_info:
                valid_frames = sum(1 for info in frame_info if info.get('valid', False))
                total_frames = len(frame_info)
                stats_text += f"Frames Processed: {total_frames}\n"
                stats_text += f"Valid Corner Detections: {valid_frames}\n"
                stats_text += f"Success Rate: {(valid_frames/total_frames*100):.1f}%\n"
            
            if average_corners:
                stats_text += f"\nAverage Corner Positions:\n"
                corner_labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
                for i, (corner, label) in enumerate(zip(average_corners, corner_labels)):
                    stats_text += f"  {label}: ({corner[0]:.1f}, {corner[1]:.1f})\n"
            
            # Data Statistics
            stats_text += "\n\nDATA STATISTICS:\n"
            stats_text += "-" * 40 + "\n"
            stats_text += f"Total Tracking Points: {len(df_mapped)}\n"
            stats_text += f"Unique Objects: {df_mapped['ID'].nunique()}\n"
            stats_text += f"Frame Range: {df_mapped['frame'].min()} - {df_mapped['frame'].max()}\n"
            
            # Update the text widget
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
        except Exception as e:
            self.log_message(f"Error displaying results: {e}", "ERROR")
    
    def refresh_file_list(self):
        """Refresh the output files list"""
        try:
            # Clear existing items
            for item in self.files_tree.get_children():
                self.files_tree.delete(item)
            
            # Add files from Output directory
            output_dir = Path("./Out")
            if output_dir.exists():
                for file_path in output_dir.iterdir():
                    if file_path.is_file():
                        try:
                            size = file_path.stat().st_size
                            modified = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                            size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                            
                            self.files_tree.insert('', tk.END, text=file_path.name, 
                                                 values=(size_str, modified))
                        except Exception:
                            continue
                            
        except Exception as e:
            self.log_message(f"Error refreshing file list: {e}", "ERROR")
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        output_dir = Path("./Out")
        if output_dir.exists():
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                os.system(f"open {output_dir}")
            else:
                os.system(f"xdg-open {output_dir}")
        else:
            messagebox.showinfo("Info", "Output folder does not exist yet.")
    
    def open_corner_test(self):
        """Open corner detection test window"""
        CornerTestWindow(self.root)
    
    def view_results(self):
        """Open results viewer window"""
        if self.results:
            ResultsViewerWindow(self.root, self.results)
        else:
            messagebox.showinfo("Info", "No results available. Please run analysis first.")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Tennis Court Tracking System v1.0

A computer vision application for analyzing tennis gameplay using:
• YOLO object detection
• DeepSORT tracking algorithms  
• Homography-based coordinate mapping
• Player and ball movement analysis

Developed for tennis performance analysis and statistics generation."""
        
        messagebox.showinfo("About", about_text)
    
    def update_progress(self, message, value):
        """Update progress bar and label"""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(text=message))
    
    def log_message(self, message, level="INFO"):
        """Add message to log output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_color = {
            "INFO": "black",
            "SUCCESS": "green", 
            "ERROR": "red",
            "WARNING": "orange"
        }.get(level, "black")
        
        def update_log():
            self.log_text.insert(tk.END, f"[{timestamp}] {level}: {message}\n")
            self.log_text.see(tk.END)
            
        self.root.after(0, update_log)
    
    def update_ui_state(self):
        """Update UI state based on processing status"""
        state = "disabled" if self.processing else "normal"
        self.tracking_button.config(state=state)
        self.analysis_button.config(state=state)


class CornerTestWindow:
    """Window for testing corner detection on single frames"""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Corner Detection Test")
        self.window.geometry("800x600")
        self.window.transient(parent)
        
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create widgets for corner test window"""
        # Control frame
        control_frame = ttk.Frame(self.window, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Detect Corners", command=self.detect_corners).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Save Result", command=self.save_result).grid(row=0, column=2, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Select an image to start")
        self.status_label.grid(row=0, column=3, padx=(20, 0))
        
        # Image display frame
        image_frame = ttk.Frame(self.window, padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(image_frame, text="Image will appear here", 
                                   anchor='center', background='#f0f0f0', relief='sunken')
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
    
    def select_image(self):
        """Select image for corner detection"""
        file_path = filedialog.askopenfilename(
            title="Select Tennis Court Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.load_image()
            self.status_label.config(text="Image loaded - ready for corner detection")
    
    def load_image(self):
        """Load and display the selected image"""
        try:
            # Load with OpenCV
            self.original_image = cv2.imread(self.image_path)
            
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = image_rgb.shape[:2]
            max_width, max_height = 700, 400
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
    
    def detect_corners(self):
        """Detect and display corners on the image"""
        if not self.original_image is not None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        
        try:
            # Import corner detector
            from Distance_measurement.corner_detection import CornerDetector
            
            cd = CornerDetector()
            corners = cd.detect_court_corners(self.original_image.copy(), debug=False)
            
            # Draw corners on image
            result_image = self.original_image.copy()
            cd.draw_corners_on_frame(result_image, corners)
            
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = image_rgb.shape[:2]
            max_width, max_height = 700, 400
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            self.processed_image = result_image
            self.status_label.config(text=f"Found {len(corners)} corners")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting corners: {e}")
    
    def save_result(self):
        """Save the processed image with corners"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Corner Detection Result",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {e}")


class ResultsViewerWindow:
    """Window for viewing detailed analysis results"""
    
    def __init__(self, parent, results):
        self.window = tk.Toplevel(parent)
        self.window.title("Analysis Results Viewer")
        self.window.geometry("1000x700")
        self.window.transient(parent)
        
        self.results = results
        self.create_widgets()
        self.load_results()
    
    def create_widgets(self):
        """Create widgets for results viewer"""
        # Create notebook for different result views
        self.notebook = ttk.Notebook(self.window, padding="10")
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Player data tab
        self.player_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.player_frame, text="Player Data")
        
        # Ball data tab
        self.ball_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ball_frame, text="Ball Data")
        
        # Corner data tab
        self.corner_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.corner_frame, text="Corner Analysis")
        
        # Raw data tab
        self.raw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_frame, text="Raw Data")
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
    
    def load_results(self):
        """Load and display results in appropriate tabs"""
        try:
            df_mapped, player_distances, ball_results, average_corners, frame_info = self.results
            
            # Player data
            self.create_player_view(player_distances)
            
            # Ball data
            self.create_ball_view(ball_results)
            
            # Corner data
            self.create_corner_view(average_corners, frame_info)
            
            # Raw data
            self.create_raw_view(df_mapped)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading results: {e}")
    
    def create_player_view(self, player_distances):
        """Create player data view"""
        if player_distances.empty:
            ttk.Label(self.player_frame, text="No player data available").pack(pady=20)
            return
        
        # Create treeview for player data
        columns = list(player_distances.columns)
        tree = ttk.Treeview(self.player_frame, columns=columns[1:], show='tree headings')
        
        # Configure columns
        tree.heading('#0', text='Player ID')
        tree.column('#0', width=100)
        
        for col in columns[1:]:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Insert data
        for _, row in player_distances.iterrows():
            values = [f"{val:.2f}" if isinstance(val, (int, float)) else str(val) 
                     for val in row.iloc[1:]]
            tree.insert('', tk.END, text=str(row.iloc[0]), values=values)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(self.player_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(self.player_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.player_frame.columnconfigure(0, weight=1)
        self.player_frame.rowconfigure(0, weight=1)
    
    def create_ball_view(self, ball_results):
        """Create ball data view"""
        if not ball_results:
            ttk.Label(self.ball_frame, text="No ball data available").pack(pady=20)
            return
        
        # Create text widget for ball data
        text_widget = scrolledtext.ScrolledText(self.ball_frame, wrap=tk.WORD)
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Format ball data
        ball_text = "BALL MOVEMENT ANALYSIS\n" + "="*50 + "\n\n"
        for key, value in ball_results.items():
            if isinstance(value, (int, float)):
                ball_text += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                ball_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        text_widget.insert(1.0, ball_text)
        text_widget.config(state='disabled')
        
        self.ball_frame.columnconfigure(0, weight=1)
        self.ball_frame.rowconfigure(0, weight=1)
    
    def create_corner_view(self, average_corners, frame_info):
        """Create corner analysis view"""
        # Create text widget for corner data
        text_widget = scrolledtext.ScrolledText(self.corner_frame, wrap=tk.WORD)
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        corner_text = "CORNER DETECTION ANALYSIS\n" + "="*50 + "\n\n"
        
        # Average corners
        if average_corners:
            corner_text += "Average Corner Positions:\n"
            corner_labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
            for i, (corner, label) in enumerate(zip(average_corners, corner_labels)):
                corner_text += f"  {label}: ({corner[0]:.1f}, {corner[1]:.1f})\n"
            corner_text += "\n"
        
        # Frame analysis
        if frame_info:
            valid_frames = sum(1 for info in frame_info if info.get('valid', False))
            total_frames = len(frame_info)
            corner_text += f"Frame Analysis:\n"
            corner_text += f"  Total Frames Processed: {total_frames}\n"
            corner_text += f"  Valid Corner Detections: {valid_frames}\n"
            corner_text += f"  Success Rate: {(valid_frames/total_frames*100):.1f}%\n\n"
            
            corner_text += "Frame-by-Frame Results:\n"
            for info in frame_info[:10]:  # Show first 10 frames
                status = "✅" if info.get('valid', False) else "❌"
                corner_text += f"  Frame {info.get('frame_number', 'N/A')}: {status} "
                corner_text += f"({info.get('corners_found', 0)} corners found)\n"
            
            if len(frame_info) > 10:
                corner_text += f"  ... and {len(frame_info) - 10} more frames\n"
        
        text_widget.insert(1.0, corner_text)
        text_widget.config(state='disabled')
        
        self.corner_frame.columnconfigure(0, weight=1)
        self.corner_frame.rowconfigure(0, weight=1)
    
    def create_raw_view(self, df_mapped):
        """Create raw data view"""
        if df_mapped.empty:
            ttk.Label(self.raw_frame, text="No tracking data available").pack(pady=20)
            return
        
        # Create treeview for raw data (show first 1000 rows)
        display_df = df_mapped.head(1000)
        columns = list(display_df.columns)
        
        tree = ttk.Treeview(self.raw_frame, columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Insert data
        for _, row in display_df.iterrows():
            values = [f"{val:.3f}" if isinstance(val, float) else str(val) for val in row]
            tree.insert('', tk.END, values=values)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(self.raw_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(self.raw_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Info label
        info_label = ttk.Label(self.raw_frame, 
                              text=f"Showing first 1000 of {len(df_mapped)} total tracking points")
        info_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Grid layout
        tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.raw_frame.columnconfigure(0, weight=1)
        self.raw_frame.rowconfigure(1, weight=1)


def main():
    """Main application entry point"""
    # Create output directory if it doesn't exist
    os.makedirs("./Out", exist_ok=True)
    
    # Create and run the application
    root = tk.Tk()
    app = TennisTrackingApp(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()