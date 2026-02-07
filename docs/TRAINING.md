# Training Guide - RTMDet on Custom Vehicle Lights Dataset

This guide shows you how to train RTMDet-m on your custom vehicle lights dataset using transfer learning from COCO pretrained weights.

## Prerequisites

- ✅ Virtual environment with all dependencies installed
- ✅ CUDA-enabled GPU (NVIDIA RTX A5000 or similar)
- ✅ Dataset in COCO format

---

## Step 1: Prepare Your Dataset

### Directory Structure

Organize your dataset as follows:

```
data/vehicle_lights/
├── train/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── val/
│   ├── img_100.jpg
│   └── ...
├── test/
│   ├── img_200.jpg
│   └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### COCO Annotation Format

Each JSON file must follow COCO format with **exactly 12 categories** in this order:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 200, 50, 30],
      "area": 1500,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "front_headlight_left"},
    {"id": 1, "name": "front_headlight_right"},
    {"id": 2, "name": "front_indicator_left"},
    {"id": 3, "name": "front_indicator_right"},
    {"id": 4, "name": "front_all_weather_left"},
    {"id": 5, "name": "front_all_weather_right"},
    {"id": 6, "name": "rear_brake_left"},
    {"id": 7, "name": "rear_brake_right"},
    {"id": 8, "name": "rear_indicator_left"},
    {"id": 9, "name": "rear_indicator_right"},
    {"id": 10, "name": "rear_tailgate_left"},
    {"id": 11, "name": "rear_tailgate_right"}
  ]
}
```

> [!IMPORTANT]
> **Category IDs must match indices exactly** (0-11). The order must match the config file.

### Bbox Format

Bboxes are in **COCO format**: `[x, y, width, height]` where `(x, y)` is the top-left corner.

---

## Step 2: Validate Your Dataset

Before training, validate your annotations:

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Validate each split
python scripts/validate_dataset.py data/vehicle_lights/annotations/train.json
python scripts/validate_dataset.py data/vehicle_lights/annotations/val.json
python scripts/validate_dataset.py data/vehicle_lights/annotations/test.json
```

The validator checks:
- ✅ Category IDs match expected order
- ✅ All 12 categories present
- ✅ Image references are valid
- ✅ Bbox formats are correct

---

## Step 3: Download Pretrained Weights

Download RTMDet-m COCO pretrained weights for transfer learning:

```bash
mkdir -p models
wget -P models/ https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth

# Rename for convenience
mv models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth models/rtmdet_m_8xb32-300e_coco.pth
```

The config file (`configs/rtmdet_m_vehicle_lights.py`) is already set to load these weights:

```python
load_from = 'models/rtmdet_m_8xb32-300e_coco.pth'
```

---

## Step 4: Configure Training

The training config is at [`configs/rtmdet_m_vehicle_lights.py`](file:///e:/vehicle-lights-pipeline/vehicle-lights-pipeline/configs/rtmdet_m_vehicle_lights.py).

### Key Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_epochs` | 300 | Total training epochs |
| `batch_size` | 8 | Batch size per GPU |
| `learning_rate` | 0.004 | Initial learning rate |
| `num_classes` | 12 | Vehicle lights classes |
| `conf_threshold` | 0.35 | Detection confidence threshold |
| `nms_threshold` | 0.45 | NMS IoU threshold |

### Adjust for Your GPU

If you have limited GPU memory, reduce batch size:

```python
# In configs/rtmdet_m_vehicle_lights.py
train_dataloader = dict(
    batch_size=4,  # Reduce from 8 to 4
    ...
)
```

---

## Step 5: Start Training

### Single GPU Training

```bash
python train.py configs/rtmdet_m_vehicle_lights.py
```

### Multi-GPU Training (Recommended)

```bash
# Using 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 \
    train.py configs/rtmdet_m_vehicle_lights.py --launcher pytorch
```

### With Mixed Precision (Faster)

```bash
python train.py configs/rtmdet_m_vehicle_lights.py --amp
```

### Resume from Checkpoint

```bash
python train.py configs/rtmdet_m_vehicle_lights.py \
    --resume work_dirs/rtmdet_m_vehicle_lights/latest.pth
```

---

## Step 6: Monitor Training

### View Logs

```bash
# Real-time logs
tail -f work_dirs/rtmdet_m_vehicle_lights/*.log

# Check latest metrics
grep "mAP" work_dirs/rtmdet_m_vehicle_lights/*.log | tail -20
```

### TensorBoard (Optional)

If you enable TensorBoard in the config:

```bash
tensorboard --logdir work_dirs/rtmdet_m_vehicle_lights
```

### Training Progress

You should see output like:

```
Epoch [10/300] [50/100]  lr: 3.9e-03, loss: 1.234, loss_cls: 0.456, loss_bbox: 0.778
Epoch [10/300] [100/100]  lr: 3.9e-03, loss: 1.123, loss_cls: 0.423, loss_bbox: 0.700
Evaluating bbox...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.678
```

---

## Step 7: Evaluate Model

After training completes, evaluate on the validation set:

```bash
python evaluate.py \
    configs/rtmdet_m_vehicle_lights.py \
    work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth
```

### Visualize Predictions

```bash
python evaluate.py \
    configs/rtmdet_m_vehicle_lights.py \
    work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth \
    --show-dir outputs/visualizations
```

This saves annotated images to `outputs/visualizations/`.

---

## Step 8: Export Trained Model

Copy the best checkpoint to the models directory:

```bash
# Use the best epoch (check logs for highest mAP)
cp work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth models/rtmdet_m_vehicle_lights.pth
```

Update `configs/default.yaml` to use your trained model:

```yaml
detection:
  checkpoint: models/rtmdet_m_vehicle_lights.pth
```

---

## Step 9: Test in Pipeline

Run the full pipeline with your trained model:

```bash
python main.py --config configs/default.yaml --overlay live
```

---

## Training Tips

### 1. Class Balance

Ensure balanced representation of all 12 classes:

```bash
# Check class distribution
python -c "
import json
with open('data/vehicle_lights/annotations/train.json') as f:
    data = json.load(f)
    from collections import Counter
    cats = Counter(ann['category_id'] for ann in data['annotations'])
    for cat_id, count in sorted(cats.items()):
        print(f'Class {cat_id}: {count} instances')
"
```

### 2. Data Augmentation

The config includes:
- Random horizontal flip (50%)
- Photometric distortion (brightness, contrast, saturation)
- Resize to 1920x1080

**Avoid** aggressive crops that cut off lights.

### 3. Lighting Conditions

Include diverse lighting:
- ☀️ Daytime
- 🌙 Nighttime
- 🌆 Twilight/dusk
- 🌧️ Rain/fog

### 4. OFF State Detection

Ensure you have examples of lights in **OFF state** to avoid false positives.

### 5. Validation Split

If using video frames, use **time-based split** (not random) to avoid data leakage.

---

## Hyperparameter Tuning

### Learning Rate

If loss plateaus early:

```python
# Increase learning rate
optimizer=dict(type='AdamW', lr=0.008, weight_decay=0.05)
```

If training is unstable:

```python
# Decrease learning rate
optimizer=dict(type='AdamW', lr=0.002, weight_decay=0.05)
```

### Batch Size

Larger batch = more stable gradients:

```python
train_dataloader = dict(batch_size=16, ...)  # If you have enough GPU memory
```

### Epochs

Monitor validation mAP. If still improving at epoch 300:

```python
max_epochs = 500
```

---

## Troubleshooting

### Low mAP

1. **Check dataset quality**: Run validation script
2. **Increase training time**: Try 500 epochs
3. **Adjust thresholds**: Lower `conf_threshold` to 0.25
4. **Check class balance**: Ensure all classes well-represented

### Out of Memory

1. **Reduce batch size**: Set to 4 or 2
2. **Enable gradient checkpointing**: Add to config
3. **Use FP16**: Add `--amp` flag

### No Improvement

1. **Verify pretrained weights loaded**: Check logs for "load checkpoint"
2. **Check learning rate**: May be too high or too low
3. **Inspect data**: Visualize training samples

### Slow Training

1. **Enable AMP**: Use `--amp` flag
2. **Increase workers**: Set `num_workers=8` in dataloader
3. **Use SSD**: Move dataset to SSD instead of HDD

---

## Next Steps

After successful training:

1. ✅ Run full pipeline: `python main.py --overlay live`
2. ✅ Test on real camera feed
3. ✅ Monitor FPS and adjust inference settings
4. ✅ Fine-tune state estimation thresholds in `configs/default.yaml`

For deployment, see [DEPLOYMENT.md](file:///e:/vehicle-lights-pipeline/vehicle-lights-pipeline/DEPLOYMENT.md).
