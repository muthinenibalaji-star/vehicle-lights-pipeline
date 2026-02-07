"""
Main pipeline orchestrator for vehicle lights detection system.

Threading model:
- Thread A: Capture - Get newest frame, drop old
- Thread B: Inference - Detection loop
- Thread C: Postprocess - Tracking + State + Logging
"""

import time
import threading
from queue import Queue, Empty
from typing import Dict, Optional
from pathlib import Path

import numpy as np
from loguru import logger

from src.capture.camera import CameraCapture
from src.detection.detector import RTMDetDetector
from src.tracking.tracker import VehicleLightsTracker
from src.state.estimator import StateEstimator
from src.logging.session_logger import SessionLogger
from src.overlay.visualizer import OverlayVisualizer
from src.utils.performance import PerformanceMonitor


class VehicleLightsPipeline:
    """Main pipeline for vehicle lights detection and state estimation."""
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.running = False
        
        # Create output directories
        Path(config['logging']['output_dir']).mkdir(parents=True, exist_ok=True)
        if config['overlay']['enabled']:
            Path(config['overlay']['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.capture = CameraCapture(config['camera'])
        self.detector = RTMDetDetector(config['detection'], config['classes'])
        self.tracker = VehicleLightsTracker(config['tracking'], config['classes'])
        self.state_estimator = StateEstimator(config['state'])
        self.session_logger = SessionLogger(
            config['logging'],
            camera_settings=self._get_camera_settings()
        )
        
        # Optional overlay
        self.visualizer = None
        if config['overlay']['enabled']:
            self.visualizer = OverlayVisualizer(config['overlay'], config['classes'])
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(
            enabled=config['performance'].get('profile', False)
        )
        
        # Thread-safe queues
        self.frame_queue = Queue(maxsize=2)  # Capture -> Inference
        self.detection_queue = Queue(maxsize=2)  # Inference -> Postprocess
        
        # Threads
        self.capture_thread = None
        self.inference_thread = None
        self.postprocess_thread = None
        
        # Frame counter
        self.frame_id = 0
        self.start_time = None
        
        logger.info("Pipeline initialization complete")
    
    def _get_camera_settings(self) -> Dict:
        """Get camera settings for logging."""
        return {
            'resolution': self.config['camera']['resolution'],
            'fps': self.config['camera']['fps'],
            'exposure': self.config['camera']['exposure'],
            'white_balance': self.config['camera']['white_balance'],
        }
    
    def _capture_loop(self):
        """Thread A: Capture frames and put newest in queue."""
        logger.info("Capture thread started")
        
        while self.running:
            try:
                # Capture frame
                frame = self.capture.read()
                if frame is None:
                    logger.warning("No frame captured, retrying...")
                    time.sleep(0.01)
                    continue
                
                # Put in queue (non-blocking, drop old frames)
                try:
                    # Clear old frames
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            break
                    
                    # Put newest frame
                    self.frame_queue.put_nowait(frame)
                    
                except Exception as e:
                    logger.debug(f"Frame queue full, dropping frame: {e}")
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)
        
        logger.info("Capture thread stopped")
    
    def _inference_loop(self):
        """Thread B: Run detection inference."""
        logger.info("Inference thread started")
        
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Run detection
                with self.perf_monitor.measure('inference'):
                    detections = self.detector.detect(frame)
                
                # Put in queue
                try:
                    # Clear old detections
                    while not self.detection_queue.empty():
                        try:
                            self.detection_queue.get_nowait()
                        except Empty:
                            break
                    
                    # Put newest detection
                    self.detection_queue.put_nowait({
                        'frame': frame,
                        'detections': detections,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    logger.debug(f"Detection queue full: {e}")
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                time.sleep(0.1)
        
        logger.info("Inference thread stopped")
    
    def _postprocess_loop(self):
        """Thread C: Tracking, state estimation, and logging."""
        logger.info("Postprocess thread started")
        
        while self.running:
            try:
                # Get detection from queue (blocking with timeout)
                try:
                    data = self.detection_queue.get(timeout=0.1)
                except Empty:
                    # Log empty frame if configured
                    if self.config['logging']['log_every_frame']:
                        timestamp = time.time() - self.start_time if self.start_time else 0.0
                        self.session_logger.log_frame(self.frame_id, timestamp, [])
                        self.frame_id += 1
                    continue
                
                frame = data['frame']
                detections = data['detections']
                
                # Calculate timestamp
                if self.start_time is None:
                    self.start_time = time.time()
                    timestamp = 0.0
                else:
                    timestamp = time.time() - self.start_time
                
                # Tracking
                with self.perf_monitor.measure('tracking'):
                    tracks = self.tracker.update(detections)
                
                # State estimation
                with self.perf_monitor.measure('state'):
                    for track in tracks:
                        state, conf = self.state_estimator.estimate(
                            track['track_id'],
                            track['bbox'],
                            frame
                        )
                        track['state'] = state
                        track['state_conf'] = conf
                
                # Logging
                with self.perf_monitor.measure('logging'):
                    self.session_logger.log_frame(self.frame_id, timestamp, tracks)
                
                # Visualization
                if self.visualizer:
                    with self.perf_monitor.measure('overlay'):
                        fps = self.perf_monitor.get_fps()
                        self.visualizer.draw(frame, tracks, fps)
                
                # Performance monitoring
                self.perf_monitor.tick()
                
                # Print stats every 30 frames
                if self.frame_id % 30 == 0:
                    fps = self.perf_monitor.get_fps()
                    stats = self.perf_monitor.get_stats()
                    logger.info(f"Frame {self.frame_id} | FPS: {fps:.1f} | {stats}")
                
                self.frame_id += 1
                
            except Exception as e:
                logger.error(f"Postprocess error: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Postprocess thread stopped")
    
    def run(self):
        """Run the pipeline."""
        logger.info("Starting pipeline threads...")
        
        self.running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.postprocess_thread = threading.Thread(target=self._postprocess_loop, daemon=True)
        
        self.capture_thread.start()
        self.inference_thread.start()
        self.postprocess_thread.start()
        
        logger.info("All threads started. Press Ctrl+C to stop.")
        
        # Wait for threads
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
            self.running = False
        
        # Wait for threads to finish
        self.capture_thread.join(timeout=2.0)
        self.inference_thread.join(timeout=2.0)
        self.postprocess_thread.join(timeout=2.0)
        
        logger.info("Pipeline stopped")
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        self.running = False
        
        if self.capture:
            self.capture.release()
        
        if self.session_logger:
            self.session_logger.close()
        
        if self.visualizer:
            self.visualizer.close()
        
        if self.perf_monitor:
            final_stats = self.perf_monitor.get_detailed_stats()
            logger.info(f"Final performance stats:\n{final_stats}")
        
        logger.info("Cleanup complete")
