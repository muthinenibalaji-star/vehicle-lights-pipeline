"""
ON/OFF/BLINKING state estimator for vehicle lights.
"""

import numpy as np
import cv2
from typing import Tuple, Dict
from collections import defaultdict, deque
from loguru import logger


class StateEstimator:
    """Estimate ON/OFF/BLINKING states for vehicle lights."""
    
    def __init__(self, config: dict):
        """Initialize state estimator."""
        self.config = config
        
        # ON/OFF parameters
        self.bg_ring_ratio = config['on_off'].get('background_ring_ratio', 1.5)
        self.eps = config['on_off'].get('eps', 1e-6)
        self.baseline_frames = config['on_off'].get('baseline_update_frames', 30)
        self.hysteresis = config['on_off'].get('hysteresis_margin', 0.15)
        
        # BLINKING parameters
        self.window_sec = config['blinking'].get('window_seconds', 2.0)
        self.min_toggles = config['blinking'].get('min_toggles', 2)
        self.min_on_frames = config['blinking'].get('min_on_frames', 5)
        self.min_off_frames = config['blinking'].get('min_off_frames', 5)
        self.warmup_frames = config['blinking'].get('warmup_frames', 30)
        
        # Track state history
        self.track_states = defaultdict(lambda: {
            'history': deque(maxlen=int(self.window_sec * 30)),  # Assume 30 FPS
            'baseline_off': None,
            'stable_off_count': 0,
            'frame_count': 0
        })
        
        logger.info("State estimator initialized")
    
    def estimate(self, track_id: int, bbox: list, frame: np.ndarray) -> Tuple[str, float]:
        """
        Estimate state for a tracked light.
        
        Args:
            track_id: Track ID
            bbox: Bounding box [x, y, w, h]
            frame: Frame image (BGR)
        
        Returns:
            (state, confidence) where state is 'ON', 'OFF', or 'BLINKING'
        """
        state_info = self.track_states[track_id]
        state_info['frame_count'] += 1
        
        # Extract ROI
        x, y, w, h = [int(v) for v in bbox]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 <= x or y2 <= y:
            return 'OFF', 0.0
        
        roi = frame[y:y2, x:x2]
        
        # Compute brightness (ON/OFF detection)
        is_on, on_conf = self._detect_on_off(roi, bbox, frame, state_info)
        
        # Store in history
        state_info['history'].append(is_on)
        
        # Check for blinking
        if state_info['frame_count'] > self.warmup_frames:
            is_blinking, blink_conf = self._detect_blinking(state_info['history'])
            
            if is_blinking:
                return 'BLINKING', blink_conf
        
        # Return ON/OFF
        return 'ON' if is_on else 'OFF', on_conf
    
    def _detect_on_off(self, roi: np.ndarray, bbox: list, frame: np.ndarray, 
                       state_info: dict) -> Tuple[bool, float]:
        """Detect if light is ON or OFF using contrast."""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        Y_roi = np.mean(gray)
        
        # Get background ring
        x, y, w, h = [int(v) for v in bbox]
        ring_w = int(w * self.bg_ring_ratio)
        ring_h = int(h * self.bg_ring_ratio)
        
        x_bg = max(0, x - (ring_w - w) // 2)
        y_bg = max(0, y - (ring_h - h) // 2)
        x2_bg = min(frame.shape[1], x_bg + ring_w)
        y2_bg = min(frame.shape[0], y_bg + ring_h)
        
        bg_roi = frame[y_bg:y2_bg, x_bg:x2_bg]
        gray_bg = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        Y_bg = np.mean(gray_bg)
        
        # Compute contrast score
        contrast = (Y_roi - Y_bg) / (Y_bg + self.eps)
        
        # Update baseline if stable OFF
        if state_info['baseline_off'] is None:
            state_info['baseline_off'] = contrast
            state_info['stable_off_count'] = 0
        
        # Simple threshold with hysteresis
        threshold = 0.3  # Will be improved with baseline
        
        if contrast > threshold + self.hysteresis:
            is_on = True
            confidence = min(1.0, contrast / (threshold + self.hysteresis))
        elif contrast < threshold - self.hysteresis:
            is_on = False
            confidence = 0.8
            state_info['stable_off_count'] += 1
        else:
            # In hysteresis zone - keep previous state
            is_on = len(state_info['history']) > 0 and state_info['history'][-1]
            confidence = 0.5
        
        return is_on, confidence
    
    def _detect_blinking(self, history: deque) -> Tuple[bool, float]:
        """Detect if light is blinking based on history."""
        if len(history) < self.min_toggles * 2:
            return False, 0.0
        
        # Count toggles
        toggles = 0
        prev = None
        on_streak = 0
        off_streak = 0
        
        for state in history:
            if prev is not None and state != prev:
                toggles += 1
            
            if state:
                on_streak += 1
                off_streak = 0
            else:
                off_streak += 1
                on_streak = 0
            
            prev = state
        
        # Check if meets blinking criteria
        if toggles >= self.min_toggles:
            confidence = min(1.0, toggles / (self.min_toggles * 2))
            return True, confidence
        
        return False, 0.0
