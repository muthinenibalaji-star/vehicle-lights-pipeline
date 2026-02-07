"""
Camera capture module with fixed exposure and white balance.
"""

import cv2
import numpy as np
from typing import Optional, Union
from loguru import logger


class CameraCapture:
    """Camera capture with fixed settings for reproducibility."""
    
    def __init__(self, config: dict):
        """
        Initialize camera capture.
        
        Args:
            config: Camera configuration dictionary
        """
        self.config = config
        self.source = config['source']
        self.resolution = tuple(config['resolution'])
        self.fps = config['fps']
        self.buffer_size = config.get('buffer_size', 1)
        
        # Initialize capture
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.source}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set buffer size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Set exposure if fixed
        if config['exposure'] == 'fixed':
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
            self.cap.set(cv2.CAP_PROP_EXPOSURE, config.get('exposure_value', -5))
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto mode
        
        # Set white balance if fixed
        if config['white_balance'] == 'fixed':
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Manual WB
            if 'white_balance_temp' in config:
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config['white_balance_temp'])
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto WB
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera opened: {self.source}")
        logger.info(f"Resolution: {actual_width}x{actual_height} (requested: {self.resolution[0]}x{self.resolution[1]})")
        logger.info(f"FPS: {actual_fps:.1f} (requested: {self.fps})")
        logger.info(f"Exposure mode: {config['exposure']}")
        logger.info(f"White balance mode: {config['white_balance']}")
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.
        
        Returns:
            Frame as numpy array (H, W, 3) in BGR format, or None if failed
        """
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame")
            return None
        
        return frame
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()
