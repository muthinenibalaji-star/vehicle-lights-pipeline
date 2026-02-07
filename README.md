# Vehicle Lights Detection + State Estimation Pipeline

Real-time vehicle lights detection, tracking, and state estimation (ON/OFF/BLINKING) using Claude 4.5 architecture.

## Quick Start (One-Click Deployment)

```bash
# Clone the repository
git clone <your-repo-url>
cd vehicle-lights-pipeline

# Run the setup script (installs dependencies, downloads model, sets up config)
./setup.sh

# Run the pipeline
python main.py --config configs/default.yaml
```

## System Requirements

- **GPU**: NVIDIA RTX A5000 (or similar)
- **CUDA**: 11.8+
- **Python**: 3.10
- **OS**: Ubuntu 20.04+ / Linux

## Project Structure

```
vehicle-lights-pipeline/
├── setup.sh                    # One-click setup script
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── configs/                    # Configuration files
│   ├── default.yaml           # Default runtime config
│   └── training.py            # MMDetection training config
├── src/                       # Source code
│   ├── capture/               # Camera capture module
│   ├── detection/             # RTMDet-m detection
│   ├── tracking/              # DeepSort tracking
│   ├── state/                 # ON/OFF/BLINKING estimation
│   ├── logging/               # JSON logging
│   ├── overlay/               # Visualization (optional)
│   └── utils/                 # Utilities
├── data/                      # Data directory
│   └── vehicle_lights/        # Dataset structure
│       ├── train/
│       ├── val/
│       ├── test/
│       └── annotations/
├── models/                    # Model checkpoints
│   └── rtmdet_m_vehicle_lights.pth
├── tests/                     # Unit tests
│   ├── test_bbox.py
│   ├── test_output.py
│   └── test_parser.py
├── scripts/                   # Utility scripts
│   ├── validate_dataset.py
│   ├── visualize_samples.py
│   └── evaluate_offline.py
├── outputs/                   # Output directory
│   ├── logs/                  # JSON logs
│   └── videos/                # Overlay videos (if enabled)
└── docs/                      # Documentation
    ├── API.md
    ├── TRAINING.md
    └── TROUBLESHOOTING.md
```

## Configuration

Edit `configs/default.yaml` to customize:

- Camera settings (resolution, FPS, exposure)
- Detection thresholds
- Tracking parameters
- State estimation windows
- Overlay settings (on/off, live/mp4/both)

## Output Format

Single JSON array per session:

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
  }
]
```

## Class List (Exact Order)

```
front_headlight_left
front_headlight_right
front_indicator_left
front_indicator_right
front_all_weather_left
front_all_weather_right
rear_brake_left
rear_brake_right
rear_indicator_left
rear_indicator_right
rear_tailgate_left
rear_tailgate_right
```

## Performance Targets

- **FPS**: ~30 FPS end-to-end
- **Resolution**: 1920×1080
- **Latency**: <35ms per frame

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_bbox.py -v

# Check output contract
python scripts/validate_output.py outputs/logs/session_*.json
```

## Training (Optional)

See `docs/TRAINING.md` for fine-tuning instructions.

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues and solutions.

## License

Proprietary - Balaji's Team
