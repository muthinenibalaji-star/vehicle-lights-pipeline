#!/usr/bin/env python3
"""
Quick start example: Prepare a sample annotation file.

This creates a minimal COCO format annotation file to help you understand
the required structure.
"""

import json
from pathlib import Path


def create_sample_annotation():
    """Create a sample COCO annotation file."""
    
    annotation = {
        "images": [
            {
                "id": 1,
                "file_name": "sample_001.jpg",
                "width": 1920,
                "height": 1080
            },
            {
                "id": 2,
                "file_name": "sample_002.jpg",
                "width": 1920,
                "height": 1080
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,  # front_headlight_left
                "bbox": [450, 520, 65, 45],  # [x, y, width, height]
                "area": 2925,
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,  # front_headlight_right
                "bbox": [1405, 518, 68, 47],
                "area": 3196,
                "iscrowd": 0
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,  # front_indicator_left
                "bbox": [380, 540, 35, 25],
                "area": 875,
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
    
    return annotation


def main():
    # Create sample annotation
    sample = create_sample_annotation()
    
    # Save to file
    output_file = Path("data/vehicle_lights/annotations/sample.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(sample, f, indent=2)
    
    print(f"✓ Sample annotation created: {output_file}")
    print("\nThis file shows the required COCO format structure.")
    print("Use this as a template for your own annotations.")
    print("\nValidate it with:")
    print(f"  python scripts/validate_dataset.py {output_file}")


if __name__ == '__main__':
    main()
