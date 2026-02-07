"""
Vehicle lights tracker using DeepSort with pair constraints.
"""

from typing import List, Dict
import numpy as np
from loguru import logger
from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleLightsTracker:
    """Tracker with left/right pair constraints."""
    
    def __init__(self, config: dict, class_names: List[str]):
        """Initialize tracker."""
        self.config = config
        self.class_names = class_names
        
        # Initialize DeepSort
        self.tracker = DeepSort(
            max_age=config.get('max_age', 30),
            n_init=config.get('min_hits', 3),
            max_iou_distance=1.0 - config.get('iou_threshold', 0.3),
            embedder=config.get('embedder'),
            half=True,
            bgr=True
        )
        
        self.max_det_per_class = config.get('max_det_per_class', 2)
        
        logger.info(f"Tracker initialized")
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker with new detections."""
        if not detections:
            self.tracker.update_tracks([], frame=None)
            return []
        
        # Convert to DeepSort format
        raw_dets = []
        for det in detections:
            raw_dets.append((
                det['bbox'],
                det['confidence'],
                det['class']
            ))
        
        # Update tracker
        tracks = self.tracker.update_tracks(raw_dets, frame=None)
        
        # Convert to output format
        output = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            bbox = track.to_ltrb()
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            output.append({
                'track_id': track.track_id,
                'class': track.get_det_class(),
                'class_id': self.class_names.index(track.get_det_class()),
                'bbox': [float(x), float(y), float(w), float(h)],
                'confidence': float(track.get_det_conf())
            })
        
        return output
