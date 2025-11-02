#!/usr/bin/env python3
"""
CUB-200-2011 Dataset Preparation Script

This script processes the Caltech-UCSD Birds-200-2011 Dataset by:
1. Reading metadata files (classes, image labels, splits)
2. Creating directory structure for train/test/validation splits
3. Organizing images into appropriate splits
4. Creating validation split from training data
"""

import os
import random
import shutil
import argparse
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt

# Default paths
DEFAULT_JPG_FOLDER = "./data/raw/CUB-200-2011/images"
DEFAULT_OUTPUT_FOLDER = "./data/processed/CUB-200-2011"
DEFAULT_LBL_TXT_PATH = "./data/raw/CUB-200-2011/image_class_labels.txt"
DEFAULT_SPLIT_TXT_PATH = "./data/raw/CUB-200-2011/train_test_split.txt"
DEFAULT_CLASS_NAME_PATH = "./data/raw/CUB-200-2011/classes.txt"
DEFAULT_IMG_NAME_PATH = "./data/raw/CUB-200-2011/images.txt"
DEFAULT_VAL_RATIO = 0.2

def create_directory_structure(output_folder: str) -> None:
    """
    Create initial directory structure for dataset splits.
    
    Args:
        output_folder: Base path for processed dataset
    """
    splits = ['train', 'test', 'validation']
    for split in splits:
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)

def read_class_names(class_name_path: str) -> Dict[int, str]:
    """
    Read class names and IDs from file.
    
    Args:
        class_name_path: Path to classes.txt file
    
    Returns:
        Dictionary mapping class IDs to class names
    """
    class_names = {}
    with open(class_name_path, 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ', 1)
            class_names[int(class_id)] = class_name
    return class_names

def create_class_directories(output_folder: str, class_names: Dict[int, str]) -> None:
    """
    Create class subdirectories in each split.
    
    Args:
        output_folder: Base path for processed dataset
        class_names: Dictionary of class IDs to names
    """
    splits = ['train', 'test', 'validation']
    for class_name in class_names.values():
        for split in splits:
            os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)

def read_image_paths(img_name_path: str) -> Dict[int, str]:
    """
    Read image paths from file.
    
    Args:
        img_name_path: Path to images.txt file
    
    Returns:
        Dictionary mapping image IDs to relative paths
    """
    img_id_to_path = {}
    with open(img_name_path, 'r') as f:
        for line in f:
            img_id, img_rel_path = line.strip().split(' ', 1)
            img_id_to_path[int(img_id)] = img_rel_path
    return img_id_to_path

def read_image_classes(lbl_txt_path: str) -> Dict[int, int]:
    """
    Read image class assignments from file.
    
    Args:
        lbl_txt_path: Path to image_class_labels.txt file
    
    Returns:
        Dictionary mapping image IDs to class IDs
    """
    img_id_to_class = {}
    with open(lbl_txt_path, 'r') as f:
        for line in f:
            img_id, class_id = map(int, line.strip().split())
            img_id_to_class[img_id] = class_id
    return img_id_to_class

def read_split_assignments(split_txt_path: str) -> Dict[int, str]:
    """
    Read train/test split assignments from file.
    
    Args:
        split_txt_path: Path to train_test_split.txt file
    
    Returns:
        Dictionary mapping image IDs to split names ('train' or 'test')
    """
    img_id_to_split = {}
    with open(split_txt_path, 'r') as f:
        for line in f:
            img_id, is_train = map(int, line.strip().split())
            img_id_to_split[img_id] = 'train' if is_train == 1 else 'test'
    return img_id_to_split

def organize_images(jpg_folder: str, output_folder: str, 
                   img_id_to_path: Dict[int, str],
                   img_id_to_class: Dict[int, int],
                   img_id_to_split: Dict[int, str],
                   class_names: Dict[int, str]) -> None:
    """
    Organize images into appropriate splits based on metadata.
    
    Args:
        jpg_folder: Source directory containing original images
        output_folder: Base directory for processed dataset
        img_id_to_path: Dictionary mapping image IDs to relative paths
        img_id_to_class: Dictionary mapping image IDs to class IDs
        img_id_to_split: Dictionary mapping image IDs to split names
        class_names: Dictionary mapping class IDs to class names
    """
    for img_id, rel_path in img_id_to_path.items():
        img_src = os.path.join(jpg_folder, rel_path)
        if not os.path.isfile(img_src):
            print(f"Image not found: {img_src}")
            continue
            
        class_id = img_id_to_class[img_id]
        split = img_id_to_split[img_id]
        class_name = class_names[class_id]
        
        dest_path = os.path.join(output_folder, split, class_name)
        os.makedirs(dest_path, exist_ok=True)
        shutil.move(img_src, os.path.join(dest_path, os.path.basename(rel_path)))

def create_validation_split(dataset_dir: str, val_ratio: float = 0.2) -> Tuple[int, int]:
    """
    Create validation split from training data.

    Args:
        dataset_dir: Base directory containing the splits
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

def process_dataset(jpg_folder: str, output_folder: str, lbl_txt_path: str,
                   split_txt_path: str, class_name_path: str, img_name_path: str,
                   val_ratio: float = 0.2) -> None:
    """
    Main function to process the CUB-200-2011 dataset.

    Args:
        jpg_folder: Directory containing source images
        output_folder: Directory for processed dataset
        lbl_txt_path: Path to image_class_labels.txt
        split_txt_path: Path to train_test_split.txt
        class_name_path: Path to classes.txt
        img_name_path: Path to images.txt
        val_ratio: Proportion of training data to use for validation
    """
    if os.path.exists(output_folder):
        print("Dataset already processed.")
        return

    # Create initial structure
    create_directory_structure(output_folder)
    
    # Read metadata
    class_names = read_class_names(class_name_path)
    img_id_to_path = read_image_paths(img_name_path)
    img_id_to_class = read_image_classes(lbl_txt_path)
    img_id_to_split = read_split_assignments(split_txt_path)
    
    # Create class directories
    create_class_directories(output_folder, class_names)
    
    # Organize images into splits
    organize_images(jpg_folder, output_folder, img_id_to_path, 
                   img_id_to_class, img_id_to_split, class_names)
    
    # Create validation split
    total_moved, skipped = create_validation_split(output_folder, val_ratio)
    print(f"{total_moved} images were moved to 'validation'.")
    print(f"{skipped} classes were skipped due to having less than 2 images.")

def main():
    """Parse arguments and process the dataset."""
    parser = argparse.ArgumentParser(description='Process CUB-200-2011 Dataset')
    
    parser.add_argument('--jpg-folder', default=DEFAULT_JPG_FOLDER,
                      help='Directory containing source images')
    parser.add_argument('--output-folder', default=DEFAULT_OUTPUT_FOLDER,
                      help='Directory for processed dataset')
    parser.add_argument('--lbl-txt-path', default=DEFAULT_LBL_TXT_PATH,
                      help='Path to image_class_labels.txt')
    parser.add_argument('--split-txt-path', default=DEFAULT_SPLIT_TXT_PATH,
                      help='Path to train_test_split.txt')
    parser.add_argument('--class-name-path', default=DEFAULT_CLASS_NAME_PATH,
                      help='Path to classes.txt')
    parser.add_argument('--img-name-path', default=DEFAULT_IMG_NAME_PATH,
                      help='Path to images.txt')
    parser.add_argument('--val-ratio', type=float, default=DEFAULT_VAL_RATIO,
                      help='Proportion of training data to use for validation')
    
    args = parser.parse_args()
    
    process_dataset(
        jpg_folder=args.jpg_folder,
        output_folder=args.output_folder,
        lbl_txt_path=args.lbl_txt_path,
        split_txt_path=args.split_txt_path,
        class_name_path=args.class_name_path,
        img_name_path=args.img_name_path,
        val_ratio=args.val_ratio
    )

if __name__ == '__main__':
    main()