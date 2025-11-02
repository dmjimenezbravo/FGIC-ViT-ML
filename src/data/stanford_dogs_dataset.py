#!/usr/bin/env python3
"""
Stanford Dogs Dataset Preparation Script

This script processes the Stanford Dogs Dataset by:
1. Organizing images into train/test splits based on provided .mat files
2. Creating a validation set from the training data
"""

import os
import random
import shutil
import argparse
from scipy.io import loadmat
from typing import List, Dict, Optional

# Default paths
DEFAULT_JPG_FOLDER = "./../data/raw/Stanford Dogs/Images"
DEFAULT_OUTPUT_FOLDER = "./../data/processed/Stanford Dogs"
DEFAULT_TRAIN_LIST_PATH = "./../data/raw/Stanford Dogs/train_list.mat"
DEFAULT_TEST_LIST_PATH = "./../data/raw/Stanford Dogs/test_list.mat"

def create_directory_structure(output_folder: str, jpg_folder: str) -> None:
    """
    Create the initial directory structure for the dataset.

    Args:
        output_folder: Base directory for the processed dataset
        jpg_folder: Source directory containing original images
    """
    splits = ['train', 'test', 'validation']
    
    # Create split directories
    for split in splits:
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)

    # Create class subdirectories for each split
    class_names = sorted(os.listdir(jpg_folder))
    for class_name in class_names:
        for split in splits:
            os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)

def load_split_data(train_list_path: str, test_list_path: str) -> Dict[str, List[str]]:
    """
    Load train and test split information from .mat files.

    Args:
        train_list_path: Path to training split .mat file
        test_list_path: Path to test split .mat file

    Returns:
        Dictionary containing train and test file lists
    """
    train_data_mat = loadmat(train_list_path)["file_list"]
    test_data_mat = loadmat(test_list_path)["file_list"]

    train_data = [train_data_mat[i][0][0] for i in range(train_data_mat.shape[0])]
    test_data = [test_data_mat[i][0][0] for i in range(test_data_mat.shape[0])]

    return {
        "train": train_data,
        "test": test_data
    }

def organize_images(jpg_folder: str, output_folder: str, split_data: Dict[str, List[str]]) -> None:
    """
    Move images to their respective split folders according to the split data.

    Args:
        jpg_folder: Source directory containing original images
        output_folder: Base directory for the processed dataset
        split_data: Dictionary containing train and test file lists
    """
    for split, data_list in split_data.items():
        for rel_path in data_list:
            class_folder = rel_path.split('/')[0]
            filename = os.path.basename(rel_path)

            src_path = os.path.join(jpg_folder, class_folder, filename)
            dest_path = os.path.join(output_folder, split, class_folder, filename)

            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
            else:
                print(f"Image not found: {src_path}")

def create_validation_split(dataset_dir: str, val_ratio: float = 0.2) -> tuple[int, int]:
    """
    Create validation split from training data.

    Args:
        dataset_dir: Base directory of the dataset
        val_ratio: Proportion of training data to use for validation

    Returns:
        Tuple of (number of images moved, number of skipped classes)
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'validation')
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

def process_dataset(jpg_folder: str, output_folder: str, train_list_path: str, 
                   test_list_path: str, val_ratio: float = 0.2) -> None:
    """
    Main function to process the Stanford Dogs dataset.

    Args:
        jpg_folder: Directory containing all original images
        output_folder: Directory where processed dataset will be created
        train_list_path: Path to training split .mat file
        test_list_path: Path to test split .mat file
        val_ratio: Proportion of training data to use for validation
    """
    if os.path.exists(output_folder):
        print("Dataset already processed.")
        return

    # Create directory structure
    create_directory_structure(output_folder, jpg_folder)

    # Load and organize split data
    split_data = load_split_data(train_list_path, test_list_path)
    organize_images(jpg_folder, output_folder, split_data)

    # Create validation split
    total_moved, skipped_classes = create_validation_split(output_folder, val_ratio)
    print(f"{total_moved} images were moved to 'validation'.")
    print(f"{skipped_classes} classes were skipped due to having less than 2 images.")

def main():
    parser = argparse.ArgumentParser(description="Process Stanford Dogs Dataset")
    parser.add_argument("--jpg-folder", default=DEFAULT_JPG_FOLDER, help="Directory containing all original images")
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER, help="Directory where processed dataset will be created")
    parser.add_argument("--train-list", default=DEFAULT_TRAIN_LIST_PATH, help="Path to training split .mat file")
    parser.add_argument("--test-list", default=DEFAULT_TEST_LIST_PATH, help="Path to test split .mat file")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proportion of training data to use for validation")

    args = parser.parse_args()

    process_dataset(
        jpg_folder=args.jpg_folder,
        output_folder=args.output_folder,
        train_list_path=args.train_list,
        test_list_path=args.test_list,
        val_ratio=args.val_ratio
    )

if __name__ == "__main__":
    main()