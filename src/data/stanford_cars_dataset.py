#!/usr/bin/env python3
"""
Stanford Cars Dataset Preparation Script

This script processes the Stanford Cars Dataset by:
1. Creating the directory structure for train/test/validation splits
2. Copying files from raw dataset into processed structure
3. Creating a validation split from training data
"""

import os
import random
import shutil
import argparse
from typing import Dict, List, Tuple

# Default paths
DEFAULT_BASE_PATH = './../data/raw/Stanford Cars'
DEFAULT_PROCESSED_PATH = './../data/processed/Stanford Cars'
DEFAULT_VAL_RATIO = 0.2

def setup_validation_structure(raw_test_path: str, processed_val_path: str) -> None:
    """
    Create validation directory structure based on test folders.
    
    Args:
        raw_test_path: Path to raw test data directory
        processed_val_path: Path where validation folders will be created
    """
    test_items = os.listdir(raw_test_path)
    test_subfolders = [item for item in test_items 
                       if os.path.isdir(os.path.join(raw_test_path, item))]

    for subfolder in test_subfolders:
        destination_subfolder_path = os.path.join(processed_val_path, subfolder)
        os.makedirs(destination_subfolder_path, exist_ok=True)
        print(f"Created directory: {destination_subfolder_path}")

    print("Validation folder structure created successfully.")

def copy_initial_data(raw_train_path: str, raw_test_path: str, 
                     processed_train_path: str, processed_test_path: str) -> None:
    """
    Copy raw train and test data to processed locations.
    
    Args:
        raw_train_path: Source path for training data
        raw_test_path: Source path for test data
        processed_train_path: Destination path for training data
        processed_test_path: Destination path for test data
    """
    shutil.copytree(raw_train_path, processed_train_path, dirs_exist_ok=True)
    shutil.copytree(raw_test_path, processed_test_path, dirs_exist_ok=True)
    print("Data copied to processed directories successfully.")

def create_validation_split(processed_dir: str, val_ratio: float = 0.2) -> Tuple[int, int]:
    """
    Create validation split by moving images from training set.
    
    Args:
        processed_dir: Base directory containing train/test/validation splits
        val_ratio: Proportion of training images to move to validation

    Returns:
        Tuple containing (number of images moved, number of skipped classes)
    """
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'validation')
    os.makedirs(val_dir, exist_ok=True)

    total_moved = 0
    skipped_classes = 0

    for class_name in os.listdir(train_dir):
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)

        if not os.path.isdir(class_train_dir):
            continue

        image_files = os.listdir(class_train_dir)
        if len(image_files) < 2:
            skipped_classes += 1
            continue

        random.shuffle(image_files)
        val_count = int(len(image_files) * val_ratio)
        val_count = min(max(1, val_count), len(image_files) - 1)
        selected_images = image_files[:val_count]

        os.makedirs(class_val_dir, exist_ok=True)

        for fname in selected_images:
            src = os.path.join(class_train_dir, fname)
            dst = os.path.join(class_val_dir, fname)
            shutil.move(src, dst)
            total_moved += 1

    return total_moved, skipped_classes

def process_dataset(base_path: str, processed_path: str, val_ratio: float) -> None:
    """
    Main function to process the Stanford Cars dataset.
    
    Args:
        base_path: Path to raw dataset
        processed_path: Path where processed dataset will be created
        val_ratio: Proportion of training data to use for validation
    """
    # Setup paths
    raw_train_path = os.path.join(base_path, 'train')
    raw_test_path = os.path.join(base_path, 'test')
    processed_train_path = os.path.join(processed_path, 'train')
    processed_test_path = os.path.join(processed_path, 'test')
    processed_val_path = os.path.join(processed_path, 'validation')

    # Create directory structure and copy data
    setup_validation_structure(raw_test_path, processed_val_path)
    copy_initial_data(raw_train_path, raw_test_path, 
                     processed_train_path, processed_test_path)

    # Create validation split
    total_moved, skipped_classes = create_validation_split(processed_path, val_ratio)
    print(f"{total_moved} images were moved to 'validation'.")
    print(f"{skipped_classes} classes were skipped due to having less than 2 images.")

def main():
    """Parse arguments and process the dataset."""
    parser = argparse.ArgumentParser(description='Process Stanford Cars Dataset')
    parser.add_argument('--base-path', default=DEFAULT_BASE_PATH,
                      help='Path to raw dataset')
    parser.add_argument('--processed-path', default=DEFAULT_PROCESSED_PATH,
                      help='Path where processed dataset will be created')
    parser.add_argument('--val-ratio', type=float, default=DEFAULT_VAL_RATIO,
                      help='Proportion of training data to use for validation')
    
    args = parser.parse_args()
    
    process_dataset(args.base_path, args.processed_path, args.val_ratio)

if __name__ == '__main__':
    main()