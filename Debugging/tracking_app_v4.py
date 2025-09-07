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
import subprocess
import webbrowser

# Add parent directory to path for project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add your project imports here
try:
    from Distance_measurement.main_court_tracker import *
    from Distance_measurement.corner_detection import *
    from Detect_and_Track.create_tracking_video_and_init_data import create_tracking_video_and_init_data
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")

class VideoPlayer:
    """Simple video player widget"""
    
    def __init__(self, parent, width=640, height=360):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Video state
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.frame_delay = int(1000 / self.fps)
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create video player widgets"""
        # Main frame
        self.frame = ttk.Frame(self.parent)
        
        # Video display
        self.video_label = ttk.Label(self.frame, text="No video loaded", 
                                   anchor='center', background='black', foreground='white')
        self.video_label.grid(row=0, column=0, columnspan=4, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Controls frame
        controls_frame = ttk.Frame(self.frame)
        controls_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
        # Control buttons
        self.play_button = ttk.Button(controls_frame, text="▶ Play", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=2)
        
        ttk.Button(controls_frame, text="⏹ Stop", command=self.stop_video).grid(row=0, column=1, padx=2)
        ttk.Button(controls_frame, text="⏮ Start", command=self.goto_start).grid(row=0, column=2, padx=2)
        ttk.Button(controls_frame, text="⏭ End", command=self.goto_end).grid(row=0, column=3, padx=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                    variable=self.progress_var, command=self.seek_video)
        self.progress_bar.grid(row=0, column=4, sticky=(tk.W, tk.E), padx=(10, 10))
        
        # Time labels
        self.time_label = ttk.Label(controls_frame, text="00:00 / 00:00")
        self.time_label.grid(row=0, column=5, padx=2)
        
        # Configure column weights
        controls_frame.columnconfigure(4, weight=1)
        
    def load_video(self, video_path):
        """Load a video file"""
        try:
            if self.cap:
                self.cap.release()
                
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise ValueError("Could not open video file")
                
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.frame_delay = int(1000 / self.fps)
            self.current_frame = 0
            
            # Load first frame
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.update_time_display()
                
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {e}")
            return False
    
    def display_frame(self, frame):
        """Display a frame in the video label"""
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            if width > self.width or height > self.height:
                scale = min(self.width/width, self.height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to RGB and then to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def toggle_play(self):
        """Toggle play/pause"""
        if not self.cap:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="⏸ Pause")
            self.play_video()
        else:
            self.play_button.config(text="▶ Play")
    
    def play_video(self):
        """Play video frames"""
        if not self.is_playing or not self.cap:
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            self.display_frame(frame)
            self.update_progress()
            self.update_time_display()
            
            # Schedule next frame
            self.parent.after(self.frame_delay, self.play_video)
        else:
            # End of video
            self.is_playing = False
            self.play_button.config(text="▶ Play")
    
    def stop_video(self):
        """Stop video and go to beginning"""
        self.is_playing = False
        self.play_button.config(text="▶ Play")
        self.goto_start()
    
    def goto_start(self):
        """Go to start of video"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset position
        self.update_progress()
        self.update_time_display()
    
    def goto_end(self):
        """Go to end of video"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames - 1)
        self.current_frame = self.total_frames - 1
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        self.update_progress()
        self.update_time_display()
    
    def seek_video(self, value):
        """Seek to specific position"""
        if not self.cap:
            return
            
        # Calculate frame number from percentage
        frame_number = int((float(value) / 100) * self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Reset position
        self.update_time_display()
    
    def update_progress(self):
        """Update progress bar"""
        if self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.progress_var.set(progress)
    
    def update_time_display(self):
        """Update time display"""
        if self.total_frames > 0 and self.fps > 0:
            current_seconds = self.current_frame / self.fps
            total_seconds = self.total_frames / self.fps
            
            current_time = f"{int(current_seconds//60):02d}:{int(current_seconds%60):02d}"
            total_time = f"{int(total_seconds//60):02d}:{int(total_seconds%60):02d}"
            
            self.time_label.config(text=f"{current_time} / {total_time}")
    
    def get_frame(self):
        """Get the video player frame widget"""
        return self.frame
    
    def destroy(self):
        """Clean up video player"""
        if self.cap:
            self.cap.release()


class FileViewerWindow:
    """Window for viewing files in-app"""
    
    def __init__(self, parent, file_path):
        self.window = tk.Toplevel(parent)
        self.window.title(f"File Viewer - {os.path.basename(file_path)}")
        self.window.geometry("900x600")
        self.window.transient(parent)
        
        self.file_path = file_path
        self.create_widgets()
        self.load_file()
    
    def create_widgets(self):
        """Create file viewer widgets"""
        # Toolbar
        toolbar = ttk.Frame(self.window)
        toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(toolbar, text=f"File: {os.path.basename(self.file_path)}").grid(row=0, column=0, padx=5)
        ttk.Button(toolbar, text="Open in External App", command=self.open_external).grid(row=0, column=1, padx=5)
        ttk.Button(toolbar, text="Open Folder", command=self.open_folder).grid(row=0, column=2, padx=5)
        
        # Content area
        self.content_frame = ttk.Frame(self.window, padding="10")
        self.content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
    
    def load_file(self):
        """Load and display file content"""
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.load_csv()
            elif file_ext == '.json':
                self.load_json()
            elif file_ext in ['.txt', '.log']:
                self.load_text()
            elif file_ext in ['.mp4', '.avi', '.mov']:
                self.load_video()
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                self.load_image()
            else:
                ttk.Label(self.content_frame, text=f"File type {file_ext} not supported for preview").pack(pady=20)
                
        except Exception as e:
            ttk.Label(self.content_frame, text=f"Error loading file: {e}").pack(pady=20)
    
    def load_csv(self):
        """Load CSV file"""
        df = pd.read_csv(self.file_path)
        
        # Create treeview
        columns = list(df.columns)
        tree = ttk.Treeview(self.content_frame, columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Insert data (first 1000 rows)
        display_df = df.head(1000)
        for _, row in display_df.iterrows():
            values = [f"{val:.3f}" if isinstance(val, float) else str(val) for val in row]
            tree.insert('', tk.END, values=values)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(self.content_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(self.content_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Info label
        info_label = ttk.Label(self.content_frame, 
                              text=f"Showing first 1000 of {len(df)} rows, {len(df.columns)} columns")
        info_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Grid layout
        tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(1, weight=1)
    
    def load_json(self):
        """Load JSON file"""
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        
        text_widget = scrolledtext.ScrolledText(self.content_frame, wrap=tk.WORD)
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
        text_widget.insert(1.0, formatted_json)
        text_widget.config(state='disabled')
    
    def load_text(self):
        """Load text file"""
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        text_widget = scrolledtext.ScrolledText(self.content_frame, wrap=tk.WORD)
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        text_widget.insert(1.0, content)
        text_widget.config(state='disabled')
    
    def load_video(self):
        """Load video file"""
        # Create video player
        self.video_player = VideoPlayer(self.content_frame, width=800, height=450)
        self.video_player.get_frame().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Load video
        self.video_player.load_video(self.file_path)
    
    def load_image(self):
        """Load image file"""
        image = cv2.imread(self.file_path)
        if image is not None:
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if too large
            height, width = image_rgb.shape[:2]
            max_width, max_height = 800, 600
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            image_label = ttk.Label(self.content_frame, image=photo)
            image_label.image = photo  # Keep reference
            image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def open_external(self):
        """Open file in external application"""
        try:
            if sys.platform == "win32":
                os.startfile(self.file_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", self.file_path])
            else:
                subprocess.run(["xdg-open", self.file_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")
    
    def open_folder(self):
        """Open folder containing the file"""
        try:
            folder_path = os.path.dirname(os.path.abspath(self.file_path))
            if sys.platform == "win32":
                subprocess.run(f'explorer /select,"{os.path.abspath(self.file_path)}"', shell=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", os.path.abspath(self.file_path)])
            else:
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")


class TennisTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tennis Court Tracking System v1.0")
        self.root.geometry("1200x540")
        self.root.configure(bg='#f0f0f0')
        
        # Application state
        self.video_path = None
        self.output_name = ""
        self.init_df_path = ""
        self.init_df = None
        self.ziframes = None
        self.zitboxes = None
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
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Play Original Video", command=self.play_original_video)
        view_menu.add_command(label="Play Clean Video", command=self.play_clean_video)
        view_menu.add_command(label="Play Tracking Video", command=self.play_tracking_video)
        view_menu.add_separator()
        view_menu.add_command(label="View Results", command=self.view_results)
        view_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Corner Detection Test", command=self.open_corner_test)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_layout(self):
        """Create the main application layout"""
        # Create main container with padding
        main_container = ttk.Frame(self.root, padding="3")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(1, weight=2)  # Give more weight to main content
        main_container.rowconfigure(2, weight=0)  # Fixed height for progress panel
        
        # Title
        title_label = ttk.Label(main_container, text="Tennis Court Tracking System", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))
        
        # Left panel - Controls
        self.create_control_panel(main_container)
        
        # Right panel - Video Preview and Results
        self.create_preview_panel(main_container)
        
        # Bottom panel - Progress and Logs
        self.create_progress_panel(main_container)
        
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        control_frame.configure(width=350)
        
        # Video Input Section
        ttk.Label(control_frame, text="Video Input", style='Section.TLabel').grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Video selection
        ttk.Button(control_frame, text="📁 Select Video", command=self.select_video, width=20).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.video_label = ttk.Label(control_frame, text="No video selected", foreground='gray')
        self.video_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Video playback buttons
        video_controls = ttk.Frame(control_frame)
        video_controls.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Button(video_controls, text="▶ Original", command=self.play_original_video).grid(row=0, column=0, padx=(0, 3))
        ttk.Button(video_controls, text="🎬 Clean", command=self.play_clean_video).grid(row=0, column=1, padx=3)
        ttk.Button(video_controls, text="📊 Tracking", command=self.play_tracking_video).grid(row=0, column=2, padx=3)
        
        # Output name
        ttk.Label(control_frame, text="Output Name:").grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        self.output_name_var = tk.StringVar(value="tennis_analysis")
        ttk.Entry(control_frame, textvariable=self.output_name_var, width=25).grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
        # Team Colors Section
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        ttk.Label(control_frame, text="Team Configuration", style='Section.TLabel').grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Team colors
        ttk.Label(control_frame, text="Team 1 Color:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.team1_color = ttk.Combobox(control_frame, values=["red", "blue", "white", "black", "green", "yellow"], width=12)
        self.team1_color.set("red")
        self.team1_color.grid(row=6, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(control_frame, text="Team 2 Color:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.team2_color = ttk.Combobox(control_frame, values=["red", "blue", "white", "black", "green", "yellow"], width=12)
        self.team2_color.set("blue")
        self.team2_color.grid(row=7, column=1, sticky=tk.W, padx=(10, 0))
        
        # Ball only option
        self.ball_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Track ball only", variable=self.ball_only_var).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Processing Section
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        ttk.Label(control_frame, text="Processing Pipeline", style='Section.TLabel').grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Step 1: Object Detection & Tracking (Combined)
        step1_frame = ttk.Frame(control_frame)
        step1_frame.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(step1_frame, text="1. Complete Object Tracking").grid(row=0, column=0, sticky=tk.W)
        self.tracking_button = ttk.Button(step1_frame, text="Start Processing", command=self.run_complete_tracking, style='Process.TButton')
        self.tracking_button.grid(row=0, column=1, padx=(10, 0))
        self.tracking_status = ttk.Label(step1_frame, text="⏸ Ready", foreground='gray')
        self.tracking_status.grid(row=0, column=2, padx=(10, 0))
        
        # Step 2: Court Analysis
        step2_frame = ttk.Frame(control_frame)
        step2_frame.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(step2_frame, text="2. Court Analysis").grid(row=0, column=0, sticky=tk.W)
        
        # Pipeline selection
        self.pipeline_var = tk.StringVar(value="standard")
        ttk.Radiobutton(step2_frame, text="Standard", variable=self.pipeline_var, value="standard").grid(row=1, column=0, sticky=tk.W, padx=(20, 0))
        ttk.Radiobutton(step2_frame, text="Dynamic", variable=self.pipeline_var, value="dynamic").grid(row=1, column=1, sticky=tk.W)
        
        self.analysis_button = ttk.Button(step2_frame, text="Start Analysis", command=self.run_analysis, style='Process.TButton')
        self.analysis_button.grid(row=0, column=1, padx=(10, 0))
        self.analysis_status = ttk.Label(step2_frame, text="⏸ Ready", foreground='gray')
        self.analysis_status.grid(row=0, column=2, padx=(10, 0))
        
    def create_preview_panel(self, parent):
        """Create the right preview panel"""
        preview_frame = ttk.LabelFrame(parent, text="Video Preview & Results", padding="5")
        preview_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)  # Results area gets remaining space
        
        # Video player
        self.video_player = VideoPlayer(preview_frame, width=480, height=180)
        self.video_player.get_frame().grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        # Results notebook
        self.results_notebook = ttk.Notebook(preview_frame)
        self.results_notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.stats_frame, text="Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(self.stats_frame, wrap=tk.WORD, height=3)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3, pady=3)
        self.stats_frame.columnconfigure(0, weight=1)
        self.stats_frame.rowconfigure(0, weight=1)
        
        # Output Files tab
        self.files_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.files_frame, text="Output Files")
        
        # Files controls
        files_controls = ttk.Frame(self.files_frame)
        files_controls.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(files_controls, text="Output folder contents:").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(files_controls, text="🔄 Refresh", command=self.refresh_output_files).grid(row=0, column=1, padx=(10, 0))
        ttk.Button(files_controls, text="📁 Open Folder", command=self.open_output_folder).grid(row=0, column=2, padx=(5, 0))
        
        # Files tree
        self.files_tree = ttk.Treeview(self.files_frame, columns=('Size', 'Modified', 'Type'), show='tree headings')
        self.files_tree.heading('#0', text='File Name')
        self.files_tree.heading('Size', text='Size')
        self.files_tree.heading('Modified', text='Modified')
        self.files_tree.heading('Type', text='Type')
        self.files_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3, pady=3)
        
        # Files scrollbar
        files_scrollbar = ttk.Scrollbar(self.files_frame, orient=tk.VERTICAL, command=self.files_tree.yview)
        files_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.files_tree.configure(yscrollcommand=files_scrollbar.set)
        
        # Double-click to open file
        self.files_tree.bind('<Double-1>', self.on_file_double_click)
        
        # Right-click context menu
        self.create_file_context_menu()
        
        self.files_frame.columnconfigure(0, weight=1)
        self.files_frame.rowconfigure(1, weight=1)
        
        # Load initial file list
        self.refresh_output_files()
        
    def create_file_context_menu(self):
        """Create context menu for files tree"""
        self.file_context_menu = tk.Menu(self.root, tearoff=0)
        self.file_context_menu.add_command(label="Open in App", command=self.open_file_in_app)
        self.file_context_menu.add_command(label="Open in External App", command=self.open_file_external)
        self.file_context_menu.add_separator()
        self.file_context_menu.add_command(label="Show in Folder", command=self.show_file_in_folder)
        self.file_context_menu.add_separator()
        self.file_context_menu.add_command(label="Play Video (if video)", command=self.play_selected_video)
        
        self.files_tree.bind('<Button-3>', self.show_file_context_menu)  # Right-click
        
    def show_file_context_menu(self, event):
        """Show context menu for files"""
        item = self.files_tree.selection()[0] if self.files_tree.selection() else None
        if item:
            self.file_context_menu.post(event.x_root, event.y_root)
        
    def create_progress_panel(self, parent):
        """Create the bottom progress panel"""
        progress_frame = ttk.LabelFrame(parent, text="Progress & Logs", padding="3")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(3, 0))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to start processing...")
        self.progress_label.grid(row=0, column=1, padx=(10, 0))
        
        # Log output
        self.log_text = scrolledtext.ScrolledText(progress_frame, wrap=tk.WORD, height=2)
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
            
            # Load video in player
            self.video_player.load_video(file_path)
            
            self.log_message(f"Video selected: {filename}")
            
            # Check if there are existing analysis files for this video
            self.check_existing_analysis_files()
    
    def check_existing_analysis_files(self):
        """Check if analysis files exist for current video and enable statistics if available"""
        if not self.video_path:
            return
            
        output_name = self.output_name_var.get()
        if not output_name:
            return
            
        # Check for key analysis files
        init_df_file = f"./Out/{output_name}_init_df.csv"
        analysis_files = [
            f"./Out/{output_name}_with_video_court_coords.csv",
            f"./Out/{output_name}_with_dynamic_court_coords.csv"
        ]
        
        # Update init_df_path if file exists
        if os.path.exists(init_df_file):
            self.init_df_path = init_df_file
            self.tracking_status.config(text="✅ Complete", foreground='green')
            self.log_message(f"Found existing tracking data: {os.path.basename(init_df_file)}")
        
        # Check for analysis results
        for analysis_file in analysis_files:
            if os.path.exists(analysis_file):
                self.analysis_status.config(text="✅ Complete", foreground='green')
                self.log_message(f"Found existing analysis: {os.path.basename(analysis_file)}")
                
                # Try to load and display existing results
                try:
                    self.load_existing_analysis_results(analysis_file)
                except Exception as e:
                    self.log_message(f"Could not load existing analysis: {e}", "WARNING")
                break
        
        # Refresh file list to show all available files
        self.refresh_output_files()
    
    def load_existing_analysis_results(self, analysis_file):
        """Load existing analysis results for display"""
        try:
            # Load the analysis CSV
            df_mapped = pd.read_csv(analysis_file)
            
            # Try to generate basic statistics from the loaded data
            player_ids = df_mapped['ID'].unique()
            player_distances = pd.DataFrame()
            
            # Calculate basic player statistics
            for player_id in player_ids:
                if player_id == 'ball':  # Skip ball for player stats
                    continue
                    
                player_data = df_mapped[df_mapped['ID'] == player_id]
                if len(player_data) < 2:
                    continue
                    
                # Calculate total distance (simplified)
                coords = player_data[['court_x', 'court_y']].dropna()
                if len(coords) > 1:
                    distances = []
                    for i in range(1, len(coords)):
                        prev_x, prev_y = coords.iloc[i-1]
                        curr_x, curr_y = coords.iloc[i]
                        dist = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
                        distances.append(dist)
                    
                    total_distance = sum(distances)
                    player_distances = pd.concat([player_distances, pd.DataFrame({
                        'ID': [player_id],
                        'TotalDistance': [total_distance],
                        'DataPoints': [len(player_data)]
                    })], ignore_index=True)
            
            # Create simplified results tuple
            self.results = (df_mapped, player_distances, {}, None, None)
            self.display_results()
            
        except Exception as e:
            self.log_message(f"Error loading existing results: {e}", "ERROR")
    
    def play_original_video(self):
        """Play original video in video player"""
        if self.video_path:
            self.video_player.load_video(self.video_path)
            self.log_message("Playing original video")
        else:
            messagebox.showwarning("Warning", "Please select a video file first.")
    
    def play_clean_video(self):
        """Play clean video if available"""
        output_name = self.output_name_var.get()
        if output_name:
            clean_video_path = f"./Out/{output_name}_out.mp4"
            
            if os.path.exists(clean_video_path):
                self.video_player.load_video(clean_video_path)
                self.log_message(f"Playing clean video: {os.path.basename(clean_video_path)}")
                return
            
        messagebox.showinfo("Info", "Clean video not available. Please run tracking first.")
    
    def play_tracking_video(self):
        """Play tracking video if available"""
        output_name = self.output_name_var.get()
        if output_name:
            tracking_video_path = f"./Out/{output_name}_out_tracked.mp4"
            
            if os.path.exists(tracking_video_path):
                self.video_player.load_video(tracking_video_path)
                self.log_message(f"Playing tracking video: {os.path.basename(tracking_video_path)}")
                return
            
        messagebox.showinfo("Info", "Tracking video not available. Please run tracking first.")
    
    def play_selected_video(self):
        """Play selected video from files tree"""
        selected_item = self.files_tree.selection()
        if selected_item:
            filename = self.files_tree.item(selected_item[0])['text']
            file_path = os.path.join("./Out", filename)
            
            # Check if it's a video file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            if any(file_path.lower().endswith(ext) for ext in video_extensions):
                self.video_player.load_video(file_path)
                self.log_message(f"Playing video: {filename}")
            else:
                messagebox.showinfo("Info", "Selected file is not a video.")
    
    def on_file_double_click(self, event):
        """Handle double-click on file"""
        selected_item = self.files_tree.selection()
        if selected_item:
            filename = self.files_tree.item(selected_item[0])['text']
            file_path = os.path.join("./Out", filename)
            self.open_file_viewer(file_path)
    
    def open_file_in_app(self):
        """Open selected file in app viewer"""
        selected_item = self.files_tree.selection()
        if selected_item:
            filename = self.files_tree.item(selected_item[0])['text']
            file_path = os.path.join("./Out", filename)
            self.open_file_viewer(file_path)
    
    def open_file_external(self):
        """Open selected file in external application"""
        selected_item = self.files_tree.selection()
        if selected_item:
            filename = self.files_tree.item(selected_item[0])['text']
            file_path = os.path.join("./Out", filename)
            try:
                if sys.platform == "win32":
                    os.startfile(file_path)
                elif sys.platform == "darwin":
                    subprocess.run(["open", file_path])
                else:
                    subprocess.run(["xdg-open", file_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
    
    def show_file_in_folder(self):
        """Show selected file in folder"""
        selected_item = self.files_tree.selection()
        if selected_item:
            filename = self.files_tree.item(selected_item[0])['text']
            file_path = os.path.abspath(os.path.join("./Out", filename))
            try:
                if sys.platform == "win32":
                    subprocess.run(f'explorer /select,"{file_path}"', shell=True)
                elif sys.platform == "darwin":
                    subprocess.run(["open", "-R", file_path])
                else:
                    folder_path = os.path.dirname(file_path)
                    subprocess.run(["xdg-open", folder_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not show file in folder: {e}")
    
    def open_file_viewer(self, file_path):
        """Open file in internal viewer"""
        if os.path.exists(file_path):
            FileViewerWindow(self.root, file_path)
        else:
            messagebox.showerror("Error", "File not found.")
    
    def refresh_output_files(self):
        """Refresh the output files list"""
        try:
            # Clear existing items
            for item in self.files_tree.get_children():
                self.files_tree.delete(item)
            
            # Check output directory
            output_dir = Path("./Out")
            if not output_dir.exists():
                self.log_message("Output directory does not exist yet")
                return
            
            file_count = 0
            # Add all files from output directory
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    try:
                        stat = os.stat(file_path)
                        size = stat.st_size
                        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                        size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                        
                        # Determine file type
                        ext = file_path.suffix.lower()
                        file_type = {
                            '.csv': 'Data',
                            '.mp4': 'Video', '.avi': 'Video', '.mov': 'Video',
                            '.png': 'Image', '.jpg': 'Image', '.jpeg': 'Image',
                            '.json': 'Config',
                            '.txt': 'Text', '.log': 'Log'
                        }.get(ext, 'Other')
                        
                        filename = file_path.name
                        self.files_tree.insert('', tk.END, text=filename, 
                                             values=(size_str, modified, file_type))
                        file_count += 1
                    except Exception:
                        continue
            
            # Log the number of files found
            self.log_message(f"Output files refreshed: {file_count} files found")
                        
        except Exception as e:
            self.log_message(f"Error refreshing file list: {e}", "ERROR")
    
    def run_complete_tracking(self):
        """Run complete object detection, tracking, and video creation in a separate thread"""
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first.")
            return
            
        if self.processing:
            messagebox.showinfo("Info", "Processing is already running.")
            return
            
        # Start tracking in a separate thread
        thread = threading.Thread(target=self._complete_tracking_worker)
        thread.daemon = True
        thread.start()
    
    def _complete_tracking_worker(self):
        """Worker thread for complete tracking process"""
        try:
            self.processing = True
            self.update_ui_state()
            
            self.output_name = self.output_name_var.get() or "tennis_analysis"
            teams_colors = [self.team1_color.get(), self.team2_color.get()]
            ball_only = self.ball_only_var.get()
            
            self.progress_var.set(0)
            self.update_progress("Starting complete tracking process...", 10)
            
            # Run the combined function
            self.log_message("Starting complete object detection, tracking, and video creation...")
            self.init_df, self.ziframes, self.zitboxes = create_tracking_video_and_init_data(
                self.video_path, self.output_name, teams_colors, ball_only
            )
            
            self.update_progress("Complete tracking process finished", 100)
            
            self.init_df_path = f"./Out/{self.output_name}_init_df.csv"
            
            self.log_message("Complete tracking process completed successfully!", "SUCCESS")
            self.root.after(0, lambda: self.tracking_status.config(text="✅ Complete", foreground='green'))
            self.root.after(0, self.refresh_output_files)
            
        except Exception as e:
            error_msg = f"Error in complete tracking process: {str(e)}"
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
            self.root.after(0, self.refresh_output_files)
            
        except Exception as e:
            error_msg = f"Error in analysis process: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.root.after(0, lambda: self.analysis_status.config(text="❌ Error", foreground='red'))
        finally:
            self.processing = False
            self.root.after(0, self.update_ui_state)
    
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
                    if 'AverageSpeedKmh' in player:
                        stats_text += f"  Average Speed: {player['AverageSpeedKmh']:.1f} km/h\n"
                    if 'MaxSpeedKmh' in player:
                        stats_text += f"  Max Speed: {player['MaxSpeedKmh']:.1f} km/h\n"
                    if 'TimeActive' in player:
                        stats_text += f"  Active Time: {player['TimeActive']:.1f}s\n"
                    if 'CourtCoverage' in player:
                        stats_text += f"  Court Coverage: {player['CourtCoverage']:.1f}m²\n"
                    if 'DataPoints' in player:
                        stats_text += f"  Data Points: {player['DataPoints']}\n"
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
            if frame_info:
                stats_text += "\n\nCORNER DETECTION ANALYSIS:\n"
                stats_text += "-" * 40 + "\n"
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
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        output_dir = os.path.abspath("./Out")
        if os.path.exists(output_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(output_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", output_dir])
                else:
                    subprocess.run(["xdg-open", output_dir])
            except Exception as e:
                self.log_message(f"Error opening output folder: {e}", "ERROR")
        else:
            messagebox.showinfo("Info", "Output folder does not exist yet.")
            # Create the folder if it doesn't exist
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log_message("Created output folder: ./Out")
            except Exception as e:
                self.log_message(f"Error creating output folder: {e}", "ERROR")
    
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

Features:
• Video playback with controls
• Output file management
• In-app file viewing
• Real-time progress tracking
• Combined processing pipeline

Developed for tennis performance analysis and statistics generation."""
        
        messagebox.showinfo("About", about_text)
    
    def update_progress(self, message, value):
        """Update progress bar and label"""
        self.root.after(0, lambda: self.progress_var.set(value))
        self.root.after(0, lambda: self.progress_label.config(text=message))
    
    def log_message(self, message, level="INFO"):
        """Add message to log output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
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
        if self.original_image is None:
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