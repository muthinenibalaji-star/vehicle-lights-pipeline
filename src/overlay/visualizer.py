"""
Overlay visualizer for debugging and demos.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class OverlayVisualizer:
    """Visualizer for live/MP4 overlay output."""
    
    def __init__(self, config: dict, class_names: List[str]):
        """Initialize visualizer."""
        self.config = config
        self.class_names = class_names
        self.enabled = config.get('enabled', False)
        self.mode = config.get('mode', 'live')
        
        if not self.enabled:
            return
        
        # Visualization parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.get('font_scale', 0.6)
        self.thickness = config.get('thickness', 2)
        
        # Video writer for MP4 mode
        self.video_writer = None
        if self.mode in ['mp4', 'both']:
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.video_path = output_dir / f'overlay_{timestamp}.mp4'
            
            logger.info(f"Overlay video will be saved to: {self.video_path}")
        
        # Color map for states
        self.state_colors = {
            'ON': (0, 255, 0),      # Green
            'OFF': (128, 128, 128), # Gray
            'BLINKING': (0, 165, 255)  # Orange
        }
        
        logger.info(f"Visualizer initialized (mode: {self.mode})")
    
    def draw(self, frame: np.ndarray, tracks: List[Dict], fps: float):
        """Draw overlay on frame."""
        if not self.enabled:
            return
        
        overlay = frame.copy()
        
        # Draw tracks
        for track in tracks:
            self._draw_track(overlay, track)
        
        # Draw FPS
        cv2.putText(
            overlay,
            f'FPS: {fps:.1f}',
            (10, 30),
            self.font,
            self.font_scale,
            (0, 255, 0),
            self.thickness
        )
        
        # Show live
        if self.mode in ['live', 'both']:
            cv2.imshow('Vehicle Lights', overlay)
            cv2.waitKey(1)
        
        # Write to video
        if self.mode in ['mp4', 'both']:
            if self.video_writer is None:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    str(self.video_path),
                    fourcc,
                    30.0,
                    (w, h)
                )
            
            self.video_writer.write(overlay)
    
    def _draw_track(self, frame: np.ndarray, track: Dict):
        """Draw a single track."""
        x, y, w, h = [int(v) for v in track['bbox']]
        
        # Get color based on state
        state = track.get('state', 'OFF')
        color = self.state_colors.get(state, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.thickness)
        
        # Draw label
        label = f"ID:{track['track_id']} {track['class']}"
        if self.config.get('show_state', True):
            label += f" {state}"
        if self.config.get('show_confidence', True):
            label += f" {track['confidence']:.2f}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.thickness
        )
    
    def close(self):
        """Close visualizer and save video."""
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Overlay video saved: {self.video_path}")
        
        if self.mode in ['live', 'both']:
            cv2.destroyAllWindows()
