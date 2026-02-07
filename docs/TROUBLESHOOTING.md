# Troubleshooting Guide

## Installation Issues

### CUDA Not Found

**Symptom**: `RuntimeError: CUDA not available`

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### MMDetection Installation Fails

**Symptom**: `ERROR: Could not find a version that satisfies mmdet`

**Solution**:
```bash
# Use mim instead of pip
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## Runtime Issues

### Low FPS (<20 FPS)

**Symptom**: Pipeline runs slower than 30 FPS target

**Solutions**:
1. Enable FP16 in config:
   ```yaml
   detection:
     fp16: true
   ```

2. Reduce inference resolution:
   ```yaml
   detection:
     inference_resolution: [1600, 900]
   ```

3. Disable overlay:
   ```yaml
   overlay:
     enabled: false
   ```

4. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   ```

### Camera Not Opening

**Symptom**: `Failed to open camera: 0`

**Solutions**:
1. Check camera permissions:
   ```bash
   ls -l /dev/video*
   sudo usermod -a -G video $USER
   ```

2. Try different camera index:
   ```bash
   python main.py --source 1  # or 2, 3, etc.
   ```

3. Check with v4l2:
   ```bash
   v4l2-ctl --list-devices
   ```

### Memory Errors

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size (should already be 1)
2. Lower inference resolution
3. Close other GPU applications
4. Check memory usage:
   ```bash
   nvidia-smi
   ```

## Detection Issues

### No Detections

**Symptom**: Empty detection lists every frame

**Solutions**:
1. Lower confidence threshold:
   ```yaml
   detection:
     conf_threshold: 0.25  # from 0.35
   ```

2. Check if model is loaded:
   ```bash
   ls -lh models/rtmdet_m_vehicle_lights.pth
   ```

3. Verify class names in config match model training

### Wrong Detections

**Symptom**: Detections on wrong objects

**Solutions**:
1. Fine-tune model on your specific dataset
2. Increase confidence threshold
3. Check NMS threshold:
   ```yaml
   detection:
     nms_threshold: 0.45
   ```

## Tracking Issues

### Track ID Churn

**Symptom**: Track IDs change frequently for same object

**Solutions**:
1. Increase max_age:
   ```yaml
   tracking:
     max_age: 50  # from 30
   ```

2. Lower IOU threshold:
   ```yaml
   tracking:
     iou_threshold: 0.2  # from 0.3
   ```

### Missing Tracks

**Symptom**: Objects detected but not tracked

**Solutions**:
1. Lower min_hits:
   ```yaml
   tracking:
     min_hits: 2  # from 3
   ```

## State Estimation Issues

### All Lights Show OFF

**Symptom**: State estimator marks all lights as OFF

**Solutions**:
1. Check brightness threshold
2. Adjust contrast parameters:
   ```yaml
   state:
     on_off:
       hysteresis_margin: 0.10  # from 0.15
   ```

### False BLINKING Detections

**Symptom**: Stable lights marked as BLINKING

**Solutions**:
1. Increase toggle requirements:
   ```yaml
   state:
     blinking:
       min_toggles: 3  # from 2
       min_on_frames: 7  # from 5
   ```

2. Increase warmup period:
   ```yaml
   state:
     blinking:
       warmup_frames: 60  # from 30
   ```

## Output Issues

### JSON File Empty or Corrupted

**Symptom**: Output JSON file is empty or invalid

**Solutions**:
1. Check if pipeline ran to completion
2. Verify output directory permissions:
   ```bash
   ls -ld outputs/logs
   chmod 755 outputs/logs
   ```

3. Check logs for errors:
   ```bash
   tail -n 100 outputs/logs/pipeline_*.log
   ```

### Bbox Format Wrong

**Symptom**: Bounding boxes don't match expected xywh format

**Solutions**:
1. Run bbox tests:
   ```bash
   pytest tests/test_bbox.py -v
   ```

2. Verify in code that detector returns xywh (top-left)

## Performance Debugging

### Enable Profiling

```bash
python main.py --config configs/default.yaml --profile
```

This will show timing for each module:
- `inference`: Detection time
- `tracking`: Tracking time
- `state`: State estimation time
- `logging`: File write time
- `overlay`: Visualization time

### Check Individual Components

Test detector only:
```python
from src.detection.detector import RTMDetDetector
import cv2

config = {...}
detector = RTMDetDetector(config, class_names)
frame = cv2.imread('test.jpg')
dets = detector.detect(frame)
```

## Getting Help

1. Check logs: `outputs/logs/pipeline_*.log`
2. Run with debug mode: `python main.py --debug`
3. Run tests: `pytest tests/ -v`
4. Check GPU: `nvidia-smi`
5. Verify config: `cat configs/default.yaml`

## Common Error Messages

### `AttributeError: 'DetInferencer' object has no attribute 'model'`
- **Cause**: MMDetection version mismatch
- **Fix**: Reinstall MMDetection with mim

### `FileNotFoundError: [Errno 2] No such file or directory: 'models/rtmdet_m_vehicle_lights.pth'`
- **Cause**: Model checkpoint not found
- **Fix**: Download or train model first

### `ValueError: could not broadcast input array from shape (1080,1920,3) into shape (1920,1080,3)`
- **Cause**: Resolution mismatch
- **Fix**: Check camera resolution in config matches actual camera

### `RuntimeError: DeepSort requires filterpy`
- **Cause**: Missing dependency
- **Fix**: `pip install filterpy --break-system-packages`
