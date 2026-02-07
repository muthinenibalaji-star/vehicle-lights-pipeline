# API Documentation

## Pipeline API

### VehicleLightsPipeline

Main pipeline class that orchestrates all components.

```python
from src.pipeline import VehicleLightsPipeline

config = {
    'camera': {...},
    'detection': {...},
    'tracking': {...},
    'state': {...},
    'logging': {...},
    'overlay': {...}
}

pipeline = VehicleLightsPipeline(config)
pipeline.run()  # Blocks until Ctrl+C
pipeline.cleanup()
```

## Component APIs

### CameraCapture

Handles camera capture with fixed settings.

```python
from src.capture import CameraCapture

config = {
    'source': 0,  # or path to video
    'resolution': [1920, 1080],
    'fps': 30,
    'exposure': 'fixed',
    'exposure_value': -5,
    'white_balance': 'fixed',
    'white_balance_temp': 4600
}

camera = CameraCapture(config)
frame = camera.read()  # Returns np.ndarray or None
camera.release()
```

### RTMDetDetector

RTMDet-m detector wrapper.

```python
from src.detection import RTMDetDetector

config = {
    'model_config': 'configs/rtmdet_m_vehicle_lights.py',
    'checkpoint': 'models/rtmdet_m_vehicle_lights.pth',
    'device': 'cuda:0',
    'conf_threshold': 0.35,
    'fp16': True
}

class_names = ['front_headlight_left', ...]

detector = RTMDetDetector(config, class_names)
detections = detector.detect(frame)
```

**Returns**: List of detections:
```python
[
    {
        'class': 'front_headlight_left',
        'class_id': 0,
        'bbox': [x, y, w, h],  # xywh top-left
        'confidence': 0.95
    }
]
```

### VehicleLightsTracker

DeepSort-based tracker.

```python
from src.tracking import VehicleLightsTracker

config = {
    'max_age': 30,
    'min_hits': 3,
    'iou_threshold': 0.3,
    'embedder': None
}

tracker = VehicleLightsTracker(config, class_names)
tracks = tracker.update(detections)
```

**Returns**: List of tracks:
```python
[
    {
        'track_id': 1,
        'class': 'front_headlight_left',
        'class_id': 0,
        'bbox': [x, y, w, h],
        'confidence': 0.95
    }
]
```

### StateEstimator

ON/OFF/BLINKING state estimator.

```python
from src.state import StateEstimator

config = {
    'on_off': {
        'background_ring_ratio': 1.5,
        'hysteresis_margin': 0.15
    },
    'blinking': {
        'window_seconds': 2.0,
        'min_toggles': 2,
        'warmup_frames': 30
    }
}

estimator = StateEstimator(config)
state, confidence = estimator.estimate(track_id, bbox, frame)
```

**Returns**: Tuple of (state, confidence):
- state: 'ON', 'OFF', or 'BLINKING'
- confidence: float in [0.0, 1.0]

### SessionLogger

Logs to single JSON array per session.

```python
from src.logging import SessionLogger

config = {
    'output_dir': 'outputs/logs',
    'log_every_frame': True
}

camera_settings = {
    'resolution': [1920, 1080],
    'fps': 30,
    'exposure': 'fixed',
    'white_balance': 'fixed'
}

logger = SessionLogger(config, camera_settings)
logger.log_frame(frame_id, timestamp, tracks)
logger.close()  # Saves JSON file
```

### OverlayVisualizer

Optional visualization.

```python
from src.overlay import OverlayVisualizer

config = {
    'enabled': True,
    'mode': 'live',  # or 'mp4' or 'both'
    'output_dir': 'outputs/videos',
    'show_bbox': True,
    'show_state': True,
    'font_scale': 0.6
}

visualizer = OverlayVisualizer(config, class_names)
visualizer.draw(frame, tracks, fps)
visualizer.close()
```

## Output Format

### Session JSON Structure

```json
[
  {
    "session_start": "2025-02-06T10:30:00.000Z",
    "camera_settings": {
      "resolution": [1920, 1080],
      "fps": 30,
      "exposure": "fixed",
      "white_balance": "fixed"
    }
  },
  {
    "frame_id": 0,
    "timestamp": 0.0,
    "detections": [
      {
        "track_id": 1,
        "class": "front_headlight_left",
        "bbox": [100, 200, 50, 30],
        "confidence": 0.95,
        "state": "ON",
        "state_conf": 0.92
      }
    ]
  },
  {
    "frame_id": 1,
    "timestamp": 0.033,
    "detections": []
  }
]
```

## Utilities

### Performance Monitor

```python
from src.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor(enabled=True)

with monitor.measure('inference'):
    # Your code here
    pass

monitor.tick()  # Record frame time
fps = monitor.get_fps()
stats = monitor.get_stats()  # "inference:25.3ms | tracking:5.1ms"
```

## Constants

### Class Names (Exact Order)

```python
CLASSES = [
    'front_headlight_left',
    'front_headlight_right',
    'front_indicator_left',
    'front_indicator_right',
    'front_all_weather_left',
    'front_all_weather_right',
    'rear_brake_left',
    'rear_brake_right',
    'rear_indicator_left',
    'rear_indicator_right',
    'rear_tailgate_left',
    'rear_tailgate_right',
]
```

### State Values

```python
STATES = ['ON', 'OFF', 'BLINKING', 'UNKNOWN']
```

## Threading Model

The pipeline uses 3 threads:

1. **Capture Thread**: Captures frames, drops old ones
2. **Inference Thread**: Runs detection
3. **Postprocess Thread**: Tracking + State + Logging

Queues:
- `frame_queue`: Capture → Inference (maxsize=2)
- `detection_queue`: Inference → Postprocess (maxsize=2)

This ensures bounded latency (newest frame wins) over processing every frame.

## Example: Standalone Detection

```python
import cv2
from src.detection import RTMDetDetector

# Load config
config = {
    'model_config': 'configs/rtmdet_m_vehicle_lights.py',
    'checkpoint': 'models/rtmdet_m_vehicle_lights.pth',
    'device': 'cuda:0',
    'conf_threshold': 0.35
}

classes = ['front_headlight_left', ...]  # All 12 classes

# Initialize detector
detector = RTMDetDetector(config, classes)

# Load image
frame = cv2.imread('test.jpg')

# Run detection
detections = detector.detect(frame)

# Print results
for det in detections:
    print(f"{det['class']}: {det['confidence']:.2f} @ {det['bbox']}")
```

## Example: Custom State Estimator

```python
from src.state import StateEstimator
import cv2

config = {
    'on_off': {
        'background_ring_ratio': 1.5,
        'hysteresis_margin': 0.15
    },
    'blinking': {
        'window_seconds': 2.0,
        'min_toggles': 2
    }
}

estimator = StateEstimator(config)

# For each tracked light
for track in tracks:
    track_id = track['track_id']
    bbox = track['bbox']
    
    # Estimate state
    state, conf = estimator.estimate(track_id, bbox, frame)
    
    print(f"Track {track_id}: {state} (conf: {conf:.2f})")
```
