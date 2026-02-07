# Training Guide

## Dataset Preparation

### 1. COCO Format Structure

Your dataset should be organized as:

```
data/vehicle_lights/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image100.jpg
│   └── ...
├── test/
│   ├── image200.jpg
│   └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### 2. Annotation Format

Each annotation JSON must follow COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
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
    ...
  ]
}
```

**CRITICAL**: Category IDs must match the exact order in the playbook.

### 3. Validate Dataset

Before training, validate your dataset:

```bash
python scripts/validate_dataset.py data/vehicle_lights/annotations/train.json
python scripts/validate_dataset.py data/vehicle_lights/annotations/val.json
```

## Training

### 1. Single GPU Training

```bash
python tools/train.py configs/rtmdet_m_vehicle_lights.py
```

### 2. Multi-GPU Training

```bash
bash tools/dist_train.sh configs/rtmdet_m_vehicle_lights.py 4  # 4 GPUs
```

### 3. Resume Training

```bash
python tools/train.py configs/rtmdet_m_vehicle_lights.py --resume work_dirs/rtmdet_m_vehicle_lights/latest.pth
```

## Evaluation

```bash
python tools/test.py \
    configs/rtmdet_m_vehicle_lights.py \
    work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth \
    --show-dir outputs/visualizations
```

## Export Model

After training, copy the best checkpoint:

```bash
cp work_dirs/rtmdet_m_vehicle_lights/epoch_300.pth models/rtmdet_m_vehicle_lights.pth
```

## Tips

1. **Class Balance**: Ensure balanced representation of all 12 classes
2. **Augmentation**: Avoid aggressive crops that cut off lights
3. **OFF-lens Recall**: Test on completely OFF scenarios
4. **Lighting Conditions**: Include day/night/twilight examples
5. **Validation**: Use time-based split, not random, if using video frames

## Hyperparameters

Adjust in `configs/rtmdet_m_vehicle_lights.py`:

- `max_epochs`: Default 300
- `learning_rate`: Default 0.004
- `batch_size`: Default 8 (adjust for GPU memory)
- `conf_threshold`: Detection threshold (default 0.35)

## Monitoring

View training logs:

```bash
tail -f work_dirs/rtmdet_m_vehicle_lights/*.log
```

Use TensorBoard (if enabled):

```bash
tensorboard --logdir work_dirs/rtmdet_m_vehicle_lights
```
