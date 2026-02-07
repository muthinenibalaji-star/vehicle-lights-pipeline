# Deployment Guide

## Quick Deployment (One Command)

```bash
cd vehicle-lights-pipeline
./setup.sh
```

This will:
1. ✓ Check system requirements (GPU, Python)
2. ✓ Create virtual environment
3. ✓ Install all dependencies
4. ✓ Download pretrained model
5. ✓ Create directory structure
6. ✓ Run tests

## Manual Deployment (Step-by-Step)

### 1. System Requirements

**Hardware:**
- GPU: NVIDIA RTX A5000 (or similar)
- RAM: 16GB+
- Storage: 10GB+

**Software:**
- OS: Ubuntu 20.04+ / Linux (Windows via WSL2)
- CUDA: 11.8+
- Python: 3.10 (recommended)
- NVIDIA Driver: Latest

Verify GPU:
```bash
nvidia-smi
```

### 2. Clone Repository

```bash
git clone <your-repo-url>
cd vehicle-lights-pipeline
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

#### a) Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### b) Install MMDetection

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

#### c) Install Other Dependencies

All dependencies are pinned for Python 3.10 compatibility:

```bash
pip install -r requirements.txt
```

**Pinned versions include:**
| Package | Version |
|---------|--------|
| numpy | 1.24.3 |
| opencv-python | 4.8.1.78 |
| scipy | 1.11.4 |
| pandas | 2.0.3 |
| loguru | 0.7.2 |

### 5. Download Model

Option A - Use pretrained COCO model (quick start):
```bash
wget -P models/ https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
mv models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth models/rtmdet_m_8xb32-300e_coco.pth
```

Option B - Use your fine-tuned model:
```bash
# Copy your trained model
cp /path/to/your/model.pth models/rtmdet_m_vehicle_lights.pth
```

### 6. Configure

Edit `configs/default.yaml`:

```yaml
camera:
  source: 0  # Your camera device
  resolution: [1920, 1080]

detection:
  checkpoint: models/rtmdet_m_vehicle_lights.pth  # Or rtmdet_m_8xb32-300e_coco.pth
```

### 7. Test Installation

```bash
# Run unit tests
pytest tests/ -v

# Test GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); ret, _ = cap.read(); print(f'Camera: {ret}'); cap.release()"
```

### 8. Run Pipeline

```bash
# Basic run
python main.py

# With custom config
python main.py --config configs/default.yaml

# With overlay
python main.py --overlay live

# Debug mode
python main.py --debug
```

## Docker Deployment (Optional)

### Build Image

Create `Dockerfile`:
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install MMDetection
RUN pip install -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.0" && \
    mim install mmdet

# Copy application
COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t vehicle-lights-pipeline .
docker run --gpus all -v $(pwd)/outputs:/app/outputs vehicle-lights-pipeline
```

## Production Deployment

### Systemd Service

Create `/etc/systemd/system/vehicle-lights.service`:

```ini
[Unit]
Description=Vehicle Lights Detection Pipeline
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/vehicle-lights-pipeline
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python main.py --config configs/default.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vehicle-lights
sudo systemctl start vehicle-lights
sudo systemctl status vehicle-lights
```

View logs:
```bash
sudo journalctl -u vehicle-lights -f
```

### Monitoring

Monitor performance:
```bash
# GPU usage
nvidia-smi -l 1

# Pipeline logs
tail -f outputs/logs/pipeline_*.log

# System resources
htop
```

## Verification

### Check Output

```bash
# List session logs
ls -lh outputs/logs/

# Validate output format
python -c "
import json
with open('outputs/logs/session_*.json') as f:
    data = json.load(f)
    print(f'Frames: {len(data)-1}')
    print(f'First frame: {data[1]}')
"
```

### Performance Check

Target: ~30 FPS

```bash
# Run with profiling
python main.py --profile

# Check FPS in logs
grep "FPS:" outputs/logs/pipeline_*.log | tail -20
```

## Troubleshooting

If you encounter issues, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

Common issues:
1. **Low FPS**: Enable FP16, reduce resolution
2. **No detections**: Lower confidence threshold
3. **Camera not found**: Check device permissions
4. **CUDA error**: Verify GPU driver and CUDA installation

## Next Steps

1. **Fine-tune model**: See [TRAINING.md](docs/TRAINING.md)
2. **Adjust thresholds**: Edit `configs/default.yaml`
3. **Monitor output**: Check `outputs/logs/`
4. **Review API**: See [API.md](docs/API.md)

## Support

- Check logs: `outputs/logs/pipeline_*.log`
- Run tests: `pytest tests/ -v`
- Enable debug: `python main.py --debug`
