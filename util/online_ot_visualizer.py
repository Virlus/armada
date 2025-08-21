import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import os
from typing import Optional, Tuple, List
import time


class OnlineOTVisualizer:
    """
    Real-time OT cost visualization plugin that creates a video showing:
    - Left side: Side camera image at each timestep
    - Right side: Growing OT cost curve with threshold line
    """
    
    def __init__(self, 
                 output_dir: str,
                 episode_idx: int,
                 fps: int = 10,
                 img_width: int = 640,
                 img_height: int = 480,
                 plot_width: int = 640,
                 plot_height: int = 480):
        """
        Initialize the online OT visualizer
        
        Args:
            output_dir: Directory to save the video
            episode_idx: Episode index for filename
            fps: Video frame rate
            img_width/img_height: Dimensions for side camera image
            plot_width/plot_height: Dimensions for OT plot
        """
        self.output_dir = output_dir
        self.episode_idx = episode_idx
        self.fps = fps
        self.img_width = img_width
        self.img_height = img_height
        self.plot_width = plot_width
        self.plot_height = plot_height
        
        # Video setup
        self.total_width = img_width + plot_width
        self.total_height = max(img_height, plot_height)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Data storage
        self.timesteps = []
        self.ot_costs = []
        self.cumulative_ot_costs = []
        self.side_images = []
        self.ot_threshold = None
        
        # Video writer (will be initialized when first frame is added)
        self.video_writer = None
        self.is_recording = False
        
        # Matplotlib setup for plot rendering
        plt.style.use('default')
        self.fig = None
        self.ax = None
        
    def start_episode(self, ot_threshold: Optional[float] = None):
        """Start recording a new episode"""
        self.ot_threshold = ot_threshold
        self.timesteps = []
        self.ot_costs = []
        self.cumulative_ot_costs = []
        self.side_images = []
        self.is_recording = True
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(self.plot_width/100, self.plot_height/100), dpi=100)
        self.fig.patch.set_facecolor('white')
        
    def add_step(self, 
                 timestep: int, 
                 ot_cost: float, 
                 side_image: np.ndarray, 
                 cumulative_ot_cost: Optional[float] = None):
        """
        Add a new timestep to the visualization
        
        Args:
            timestep: Current timestep
            ot_cost: OT cost for this timestep
            side_image: Side camera image (H, W, 3) in RGB format
            cumulative_ot_cost: Cumulative OT cost up to this timestep
        """
        if not self.is_recording:
            return
            
        self.timesteps.append(timestep)
        self.ot_costs.append(ot_cost)
        
        if cumulative_ot_cost is not None:
            self.cumulative_ot_costs.append(cumulative_ot_cost)
        else:
            # Calculate cumulative cost
            if len(self.cumulative_ot_costs) == 0:
                self.cumulative_ot_costs.append(ot_cost)
            else:
                self.cumulative_ot_costs.append(self.cumulative_ot_costs[-1] + ot_cost)
        
        # Store side image (expects RGB format, will be converted to BGR for video output)
        if side_image.dtype != np.uint8:
            side_image = (side_image * 255).astype(np.uint8)
        if len(side_image.shape) == 3 and side_image.shape[2] == 3:
            self.side_images.append(side_image)
        else:
            # Handle grayscale or other formats
            if len(side_image.shape) == 2:
                side_image = np.stack([side_image] * 3, axis=2)
            self.side_images.append(side_image)
        
        # Generate and save frame
        self._generate_frame()
    
    def _generate_frame(self):
        """Generate a single frame combining side image and OT plot"""
        # Create the OT cost plot
        plot_img = self._create_ot_plot()
        
        # Get the latest side image
        side_img = self.side_images[-1]
        
        # Resize images to target dimensions
        side_img_resized = cv2.resize(side_img, (self.img_width, self.img_height))
        plot_img_resized = cv2.resize(plot_img, (self.plot_width, self.plot_height))
        
        # Combine images horizontally
        combined_frame = np.hstack([side_img_resized, plot_img_resized])
        
        # Ensure the frame has the correct total dimensions
        if combined_frame.shape[0] != self.total_height:
            combined_frame = cv2.resize(combined_frame, (self.total_width, self.total_height))
        
        # Initialize video writer on first frame
        if self.video_writer is None:
            video_path = os.path.join(self.output_dir, f"ot_visualization_episode_{self.episode_idx}.mp4")
            self.video_writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, 
                                              (self.total_width, self.total_height))
        
        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        self.video_writer.write(combined_frame_bgr)
    
    def _create_ot_plot(self) -> np.ndarray:
        """Create the OT cost plot as an image"""
        # Clear the plot
        self.ax.clear()
        
        # Plot cumulative OT cost curve
        if len(self.timesteps) > 0:
            self.ax.plot(self.timesteps, self.cumulative_ot_costs, 
                        'b-', linewidth=2, label='Cumulative OT Cost')
            
            # Add threshold line if available
            if self.ot_threshold is not None:
                self.ax.axhline(y=self.ot_threshold, color='r', linestyle='--', 
                              linewidth=2, label=f'Threshold ({self.ot_threshold:.3f})')
            
            # Highlight failure point if threshold is exceeded
            if self.ot_threshold is not None and len(self.cumulative_ot_costs) > 0:
                if self.cumulative_ot_costs[-1] > self.ot_threshold:
                    self.ax.scatter([self.timesteps[-1]], [self.cumulative_ot_costs[-1]], 
                                  color='red', s=100, zorder=5, label='Failure Detected')
        
        # Formatting
        self.ax.set_xlabel('Timestep', fontsize=12)
        self.ax.set_ylabel('Cumulative OT Cost', fontsize=12)
        self.ax.set_title('Online OT Cost Monitoring', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left')
        
        # Set axis limits
        if len(self.timesteps) > 0:
            self.ax.set_xlim(0, max(self.timesteps[-1] + 5, 10))
            max_cost = max(self.cumulative_ot_costs) if self.cumulative_ot_costs else 1
            if self.ot_threshold is not None:
                max_cost = max(max_cost, self.ot_threshold * 1.2)
            self.ax.set_ylim(0, max_cost * 1.1)
        
        # Convert matplotlib figure to image
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Convert to numpy array
        plot_img = np.frombuffer(raw_data, dtype=np.uint8)
        plot_img = plot_img.reshape((size[1], size[0], 3))
        
        return plot_img
    
    def end_episode(self, success: bool = True, failure_reason: Optional[str] = None):
        """
        End the current episode and save the video
        
        Args:
            success: Whether the episode was successful
            failure_reason: Reason for failure if not successful
        """
        if not self.is_recording:
            return
        
        # Add final annotation to the plot if failed
        if not success and failure_reason:
            if len(self.timesteps) > 0:
                # Add failure annotation
                self.ax.annotate(f'FAILED: {failure_reason}', 
                               xy=(self.timesteps[-1], self.cumulative_ot_costs[-1]),
                               xytext=(self.timesteps[-1] - 5, self.cumulative_ot_costs[-1] + 0.1),
                               fontsize=12, color='red', fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='red'))
                
                # Generate final frame with annotation
                self._generate_frame()
        
        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
            video_path = os.path.join(self.output_dir, f"ot_visualization_episode_{self.episode_idx}.mp4")
            print(f"Saved OT visualization video: {video_path}")
        
        # Close matplotlib figure
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        self.is_recording = False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.video_writer is not None:
            self.video_writer.release()
        if self.fig is not None:
            plt.close(self.fig)


class OTVisualizationModule:
    """
    Integration module for OT visualization that can be plugged into the failure detection system
    """
    
    def __init__(self, enable_visualization: bool = False):
        self.enable_visualization = enable_visualization
        self.visualizer = None
        self.current_episode_idx = None
        
    def initialize(self, output_dir: str, fps: int = 10):
        """Initialize the visualization module"""
        if not self.enable_visualization:
            return
            
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)
    
    def start_episode(self, episode_idx: int, ot_threshold: Optional[float] = None):
        """Start visualization for a new episode"""
        if not self.enable_visualization:
            return
            
        self.current_episode_idx = episode_idx
        self.visualizer = OnlineOTVisualizer(
            output_dir=self.output_dir,
            episode_idx=episode_idx,
            fps=self.fps
        )
        self.visualizer.start_episode(ot_threshold=ot_threshold)
    
    def add_step(self, timestep: int, ot_cost: float, side_image: np.ndarray, 
                 cumulative_ot_cost: Optional[float] = None):
        """Add a step to the visualization"""
        if not self.enable_visualization or self.visualizer is None:
            return
            
        self.visualizer.add_step(timestep, ot_cost, side_image, cumulative_ot_cost)
    
    def end_episode(self, success: bool = True, failure_reason: Optional[str] = None):
        """End the current episode visualization"""
        if not self.enable_visualization or self.visualizer is None:
            return
            
        self.visualizer.end_episode(success=success, failure_reason=failure_reason)
        self.visualizer = None
    
    def cleanup(self):
        """Cleanup visualization resources"""
        if self.visualizer is not None:
            self.visualizer.cleanup()
            self.visualizer = None
