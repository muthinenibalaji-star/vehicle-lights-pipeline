#!/usr/bin/env python3
"""
Vehicle Lights Detection Pipeline - Main Entry Point

Real-time vehicle lights detection, tracking, and state estimation.
Target: ~30 FPS @ 1920x1080 on NVIDIA RTX A5000
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

import yaml
import torch

from src.capture.camera import CameraCapture
from src.detection.detector import RTMDetDetector
from src.tracking.tracker import VehicleLightsTracker
from src.state.estimator import StateEstimator
from src.logging.session_logger import SessionLogger
from src.overlay.visualizer import OverlayVisualizer
from src.pipeline import VehicleLightsPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Vehicle Lights Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Video source (overrides config): 0 for webcam, or path to video file'
    )
    
    parser.add_argument(
        '--overlay',
        type=str,
        choices=['live', 'mp4', 'both', 'off'],
        help='Overlay mode (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for logs/videos (overrides config)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config with command line arguments."""
    if args.source is not None:
        config['camera']['source'] = args.source
        logger.info(f"Overriding camera source: {args.source}")
    
    if args.overlay is not None:
        config['overlay']['enabled'] = (args.overlay != 'off')
        if args.overlay != 'off':
            config['overlay']['mode'] = args.overlay
        logger.info(f"Overriding overlay mode: {args.overlay}")
    
    if args.output_dir is not None:
        config['logging']['output_dir'] = args.output_dir
        config['overlay']['output_dir'] = args.output_dir
        logger.info(f"Overriding output directory: {args.output_dir}")
    
    if args.profile:
        config['performance']['profile'] = True
        logger.info("Performance profiling enabled")
    
    return config


def check_gpu():
    """Check GPU availability and print info."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available. GPU required for real-time inference.")
        sys.exit(1)
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    logger.info(f"GPU detected: {gpu_name}")
    logger.info(f"GPU memory: {gpu_memory:.2f} GB")
    
    return device


def setup_logging(debug: bool = False):
    """Configure logging."""
    logger.remove()
    
    log_level = "DEBUG" if debug else "INFO"
    
    # Console logging
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File logging
    logger.add(
        "outputs/logs/pipeline_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="100 MB"
    )


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    logger.info("=" * 60)
    logger.info("Vehicle Lights Detection Pipeline")
    logger.info("=" * 60)
    
    # Check GPU
    check_gpu()
    
    # Load and override config
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Create pipeline
    try:
        pipeline = VehicleLightsPipeline(config)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        logger.info("Starting pipeline...")
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pipeline.cleanup()
        logger.info("Pipeline shutdown complete")


if __name__ == '__main__':
    main()
