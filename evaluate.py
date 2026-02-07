#!/usr/bin/env python3
"""
Evaluation script for RTMDet-m on vehicle lights dataset.

Evaluates a trained model checkpoint on the validation/test set.
"""

import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import DetInferencer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RTMDet on vehicle lights')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save evaluation results'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='show prediction results'
    )
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='append',
        help='override some settings in the used config'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Merge CLI arguments
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Load checkpoint
    runner.load_checkpoint(args.checkpoint)

    # Run evaluation
    runner.test()


if __name__ == '__main__':
    main()
