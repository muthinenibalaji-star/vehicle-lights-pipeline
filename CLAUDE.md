# CLAUDE.md - Project Context for AI Assistants

This document provides comprehensive context for understanding the Vehicle Lights Detection Pipeline project.

## Project Overview

Real-time vehicle lights detection, tracking, and state estimation system using RTMDet-m (MMDetection) for automotive applications. Detects 12 types of vehicle lights and estimates their states (ON/OFF/BLINKING).

## Tech Stack

| Component | Technology |
|-----------|------------|
| Detection | RTMDet-m via MMDetection |
| Tracking | DeepSort |
| Framework | PyTorch + CUDA 11.8 |
| Python | 3.10 (required) |
| Camera | OpenCV |

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Camera    │───▶│  Detector   │───▶│    Tracker      │
│  (Thread A) │    │  (Thread B) │    │   (Thread C)    │
└─────────────┘    └─────────────┘    └─────────────────┘
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │ State Estimator│
                                      │ (ON/OFF/BLINK) │
                                      └───────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                       ┌──────────┐                    ┌──────────┐
                       │  Logger  │                    │ Visualizer│
                       │  (JSON)  │                    │ (Optional)│
                       └──────────┘                    └──────────┘
```

## Directory Structure

```
vehicle-lights-pipeline/
├── main.py                 # Entry point
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Pinned dependencies (Python 3.10)
├── configs/
│   ├── default.yaml        # Runtime config
│   └── rtmdet_m_vehicle_lights.py  # MMDetection training config
├── src/
│   ├── pipeline.py         # Main orchestrator (3 threads)
│   ├── capture/camera.py   # Camera capture
│   ├── detection/detector.py    # RTMDet wrapper
│   ├── tracking/tracker.py      # DeepSort wrapper
│   ├── state/estimator.py       # ON/OFF/BLINKING detection
│   ├── overlay/visualizer.py    # Debug visualization
│   └── logging/session_logger.py # JSON output
├── scripts/
│   ├── validate_dataset.py      # Dataset validation
│   └── create_sample_annotation.py # COCO format example
├── data/vehicle_lights/    # Dataset (COCO format)
├── models/                 # Model checkpoints
└── docs/                   # Documentation
```

## 12 Vehicle Light Classes (Exact Order)

```python
CLASSES = [
    'front_headlight_left',     # 0
    'front_headlight_right',    # 1
    'front_indicator_left',     # 2
    'front_indicator_right',    # 3
    'front_all_weather_left',   # 4
    'front_all_weather_right',  # 5
    'rear_brake_left',          # 6
    'rear_brake_right',         # 7
    'rear_indicator_left',      # 8
    'rear_indicator_right',     # 9
    'rear_tailgate_left',       # 10
    'rear_tailgate_right',      # 11
]
```

> **CRITICAL**: Category IDs in COCO annotations MUST match these indices (0-11).

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, parses args, creates pipeline |
| `src/pipeline.py` | Thread orchestration, queue management |
| `src/detection/detector.py` | RTMDet inference via MMDetection DetInferencer |
| `src/tracking/tracker.py` | DeepSort tracking with pair constraints |
| `src/state/estimator.py` | Contrast-based ON/OFF + toggle-based BLINKING |
| `configs/rtmdet_m_vehicle_lights.py` | MMDetection training config (300 epochs) |
| `configs/default.yaml` | Runtime settings (thresholds, camera, overlay) |

## Data Flow

1. **Capture** (Thread A): `cv2.VideoCapture` → frame queue
2. **Inference** (Thread B): `DetInferencer` → detection queue
3. **Postprocess** (Thread C): Tracking → State estimation → JSON logging

## Output Format (JSON)

```json
[
  {"session_start": "2025-02-06T10:30:00Z", "camera_settings": {...}},
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
  }
]
```

## Common Commands

```bash
# Run pipeline
python main.py --config configs/default.yaml --overlay live

# Train model
python train.py configs/rtmdet_m_vehicle_lights.py --amp

# Evaluate model
python evaluate.py configs/rtmdet_m_vehicle_lights.py work_dirs/.../epoch_300.pth

# Validate dataset
python scripts/validate_dataset.py data/vehicle_lights/annotations/train.json

# Run tests
pytest tests/ -v
```

## Dependencies Installation Order

```bash
# 1. PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. MMDetection stack
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet

# 3. Other dependencies
pip install -r requirements.txt
```

## Key Configuration Parameters

### Detection (`configs/default.yaml`)
- `conf_threshold`: 0.35 (detection confidence)
- `nms_threshold`: 0.45 (NMS IoU threshold)
- `fp16`: true (half-precision for speed)

### State Estimation
- `background_ring_ratio`: 1.5 (for contrast calculation)
- `min_toggles`: 2 (minimum to detect BLINKING)
- `warmup_frames`: 30 (skip early blink detection)

### Tracking
- `max_age`: 30 (frames before deleting lost track)
- `min_hits`: 3 (confirmations before track is valid)

## State Estimation Logic

**ON/OFF Detection:**
- Extract ROI brightness (Y channel)
- Compare to background ring brightness
- Contrast = (Y_roi - Y_bg) / Y_bg
- Threshold with hysteresis margin

**BLINKING Detection:**
- Track ON/OFF history over 2-second window
- Count state toggles
- If toggles >= 2, classify as BLINKING

## Performance Targets

- **FPS**: ~30 @ 1920×1080
- **Latency**: <35ms per frame
- **GPU**: NVIDIA RTX A5000 or similar

## Known Patterns

1. **Transfer Learning**: Start from COCO pretrained weights (`load_from` in config)
2. **Pair Constraints**: Left/right lights enforced via `max_det_per_class: 2`
3. **Thread Isolation**: Capture/Inference/Postprocess run independently
4. **Queue Drop Policy**: Old frames/detections dropped to maintain real-time

## Testing

- `tests/test_bbox.py`: Bbox format conversions
- `tests/test_output.py`: JSON output contract validation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low FPS | Enable FP16, reduce resolution |
| No detections | Lower `conf_threshold` to 0.25 |
| OOM | Reduce `batch_size` in training config |
| Import error | Check virtual env, reinstall mmdet |
