"""
RTMDet-m detector for vehicle lights using MMDetection.
"""

import torch
import numpy as np
import cv2
from typing import List, Dict
from pathlib import Path
from loguru import logger

from mmdet.apis import DetInferencer


class RTMDetDetector:
    """RTMDet-m detector wrapper."""
    
    def __init__(self, config: dict, class_names: List[str]):
        """
        Initialize detector.
        
        Args:
            config: Detection configuration dictionary
            class_names: List of class names in exact order
        """
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Load model
        model_config = Path(config['model_config'])
        checkpoint = Path(config['checkpoint'])
        
        if not model_config.exists():
            raise FileNotFoundError(f"Model config not found: {model_config}")
        
        if not checkpoint.exists():
            logger.warning(f"Checkpoint not found: {checkpoint}")
            logger.warning("Using pretrained COCO weights. Fine-tune for best results.")
            checkpoint = None
        
        # Initialize inferencer
        device = config.get('device', 'cuda:0')
        
        logger.info(f"Loading RTMDet model on {device}...")
        
        try:
            self.inferencer = DetInferencer(
                model=str(model_config),
                weights=str(checkpoint) if checkpoint else None,
                device=device
            )
            logger.info("RTMDet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Detection parameters
        self.conf_threshold = config.get('conf_threshold', 0.35)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.inference_resolution = tuple(config.get('inference_resolution', [1920, 1080]))
        self.fp16 = config.get('fp16', True)
        
        logger.info(f"Detection threshold: {self.conf_threshold}")
        logger.info(f"NMS threshold: {self.nms_threshold}")
        logger.info(f"Inference resolution: {self.inference_resolution}")
        logger.info(f"FP16 enabled: {self.fp16}")
        
        # Warm-up
        logger.info("Warming up model...")
        dummy_img = np.zeros((*self.inference_resolution[::-1], 3), dtype=np.uint8)
        self.detect(dummy_img)
        logger.info("Model warm-up complete")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a frame.
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
        
        Returns:
            List of detections, each with:
                - class: class name
                - class_id: class index
                - bbox: [x, y, w, h] in top-left format
                - confidence: detection confidence
        """
        # Resize if needed
        if frame.shape[:2][::-1] != self.inference_resolution:
            frame = cv2.resize(frame, self.inference_resolution)
        
        # Run inference with autocast for FP16
        with torch.cuda.amp.autocast(enabled=self.fp16):
            result = self.inferencer(
                frame,
                return_datasamples=True,
                no_save_vis=True
            )
        
        # Parse results
        detections = []
        
        if 'predictions' in result and len(result['predictions']) > 0:
            pred = result['predictions'][0]
            
            if hasattr(pred, 'pred_instances'):
                instances = pred.pred_instances
                
                bboxes = instances.bboxes.cpu().numpy()  # xyxy format
                scores = instances.scores.cpu().numpy()
                labels = instances.labels.cpu().numpy()
                
                for bbox, score, label in zip(bboxes, scores, labels):
                    if score < self.conf_threshold:
                        continue
                    
                    # Convert xyxy to xywh (top-left)
                    x1, y1, x2, y2 = bbox
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    detections.append({
                        'class': self.class_names[int(label)],
                        'class_id': int(label),
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'confidence': float(score)
                    })
        
        return detections
