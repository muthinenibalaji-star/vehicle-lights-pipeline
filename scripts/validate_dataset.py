#!/usr/bin/env python3
"""
Validate COCO dataset for vehicle lights.

Checks:
- Category IDs match class order
- Annotations reference valid images
- Bbox formats are valid
"""

import json
import argparse
from pathlib import Path
from loguru import logger


EXPECTED_CLASSES = [
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


def validate_dataset(annotation_file: Path):
    """Validate COCO annotation file."""
    logger.info(f"Validating {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    errors = []
    warnings = []
    
    # Check categories
    if 'categories' not in data:
        errors.append("Missing 'categories' field")
        return errors, warnings
    
    categories = data['categories']
    
    # Check category count
    if len(categories) != len(EXPECTED_CLASSES):
        errors.append(f"Expected {len(EXPECTED_CLASSES)} categories, found {len(categories)}")
    
    # Check category order and IDs
    for i, expected_name in enumerate(EXPECTED_CLASSES):
        if i >= len(categories):
            errors.append(f"Missing category: {expected_name}")
            continue
        
        cat = categories[i]
        
        # Check ID matches index
        if cat['id'] != i:
            errors.append(f"Category '{cat['name']}' has ID {cat['id']}, expected {i}")
        
        # Check name matches
        if cat['name'] != expected_name:
            errors.append(f"Category {i}: expected '{expected_name}', found '{cat['name']}'")
    
    # Check images
    if 'images' not in data:
        errors.append("Missing 'images' field")
    else:
        image_ids = {img['id'] for img in data['images']}
        logger.info(f"Found {len(image_ids)} images")
    
    # Check annotations
    if 'annotations' not in data:
        errors.append("Missing 'annotations' field")
    else:
        annotations = data['annotations']
        logger.info(f"Found {len(annotations)} annotations")
        
        # Validate annotations
        for ann in annotations:
            # Check image reference
            if 'image_id' in ann and ann['image_id'] not in image_ids:
                warnings.append(f"Annotation {ann['id']} references non-existent image {ann['image_id']}")
            
            # Check category
            if 'category_id' in ann:
                if ann['category_id'] < 0 or ann['category_id'] >= len(EXPECTED_CLASSES):
                    errors.append(f"Annotation {ann['id']} has invalid category_id {ann['category_id']}")
            
            # Check bbox format
            if 'bbox' in ann:
                bbox = ann['bbox']
                if len(bbox) != 4:
                    errors.append(f"Annotation {ann['id']} has invalid bbox length: {len(bbox)}")
                elif any(v < 0 for v in bbox):
                    warnings.append(f"Annotation {ann['id']} has negative bbox values")
    
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description='Validate COCO dataset')
    parser.add_argument('annotation_file', type=Path, help='Path to annotation JSON file')
    args = parser.parse_args()
    
    if not args.annotation_file.exists():
        logger.error(f"File not found: {args.annotation_file}")
        return 1
    
    errors, warnings = validate_dataset(args.annotation_file)
    
    # Print results
    if errors:
        logger.error(f"Found {len(errors)} errors:")
        for err in errors:
            logger.error(f"  - {err}")
    
    if warnings:
        logger.warning(f"Found {len(warnings)} warnings:")
        for warn in warnings:
            logger.warning(f"  - {warn}")
    
    if not errors and not warnings:
        logger.success("Dataset validation passed!")
        return 0
    elif not errors:
        logger.info("Dataset validation passed with warnings")
        return 0
    else:
        logger.error("Dataset validation failed")
        return 1


if __name__ == '__main__':
    exit(main())
