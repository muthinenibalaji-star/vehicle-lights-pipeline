"""
Session logger for vehicle lights pipeline.
Outputs single JSON array per session.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from loguru import logger


class SessionLogger:
    """Logger that creates single JSON array per session."""
    
    def __init__(self, config: dict, camera_settings: dict):
        """Initialize session logger."""
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session data
        self.session_data = []
        
        # Add session header
        session_start = datetime.now().isoformat() + 'Z'
        self.session_data.append({
            'session_start': session_start,
            'camera_settings': camera_settings
        })
        
        # Output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_file = self.output_dir / f'session_{timestamp}.json'
        
        logger.info(f"Session logger initialized: {self.output_file}")
    
    def log_frame(self, frame_id: int, timestamp: float, tracks: List[Dict]):
        """
        Log a single frame.
        
        Args:
            frame_id: Frame number
            timestamp: Timestamp in seconds since session start
            tracks: List of tracked objects with state
        """
        frame_data = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'detections': []
        }
        
        for track in tracks:
            detection = {
                'track_id': track['track_id'],
                'class': track['class'],
                'bbox': track['bbox'],  # Already in xywh format
                'confidence': track['confidence'],
                'state': track.get('state', 'UNKNOWN'),
                'state_conf': track.get('state_conf', 0.0)
            }
            frame_data['detections'].append(detection)
        
        self.session_data.append(frame_data)
    
    def close(self):
        """Save session data to file."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            logger.info(f"Session log saved: {self.output_file}")
            logger.info(f"Total frames logged: {len(self.session_data) - 1}")
        except Exception as e:
            logger.error(f"Failed to save session log: {e}")
