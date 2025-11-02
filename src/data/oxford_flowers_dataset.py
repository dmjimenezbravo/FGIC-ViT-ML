#!/usr/bin/env python3
"""
Oxford 102 Flowers Dataset Preparation Script

This script processes the Oxford 102 Flowers Dataset by:
1. Reading class names and creating directory structure
2. Loading image labels and split information
3. Organizing images into train/test/validation splits
4. Optionally swapping train and test sets
"""

import os
import shutil
import argparse
from typing import Dict, List, Tuple
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Default paths
DEFAULT_JPG_FOLDER = "./../data/raw/Oxford 102 Flower/jpg"
DEFAULT_OUTPUT_FOLDER = "./../data/processed/Oxford 102 Flower"
DEFAULT_LBL_MAT_PATH = "./../data/raw/Oxford 102 Flower/imagelabels.mat"
DEFAULT_SPLIT_MAT_PATH = "./../data/raw/Oxford 102 Flower/setid.mat"
DEFAULT_CLASS_NAME_PATH = "./../data/raw/Oxford 102 Flower/Oxford-102_Flower_dataset_labels.txt"

def read_class_names(class_name_path: str) -> List[str]:
    """
    Read class names from text file.
    
    Args:
        class_name_path: Path to the text file containing class names
        
    Returns:
        List of class names
    """
    with open(class_name_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def create_directory_structure(output_folder: str, class_names: List[str]) -> None:
    """
    Create directory structure for dataset splits.
    
    Args:
        output_folder: Base path for processed dataset
        class_names: List of class names to create directories for
    """
    splits = ['train', 'test', 'validation']
    
    # Create split directories
    for split in splits:
        split_path = os.path.join(output_folder, split)
        os.makedirs(split_path, exist_ok=True)
        
        # Create class subdirectories
        for class_name in class_names:
            os.makedirs(os.path.join(split_path, class_name), exist_ok=True)

def load_split_data(lbl_mat_path: str, split_mat_path: str) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Load image labels and split information.
    
    Args:
        lbl_mat_path: Path to labels .mat file
        split_mat_path: Path to split assignments .mat file
        
    Returns:
        Tuple of (image labels, split assignments)
    """
    lbl_data = loadmat(lbl_mat_path)["labels"].flatten()
    split_data = loadmat(split_mat_path)
    
    split_sets = {
        "train": split_data["trnid"].flatten(),
        "test": split_data["tstid"].flatten(),
        "validation": split_data["valid"].flatten()
    }
    
    return lbl_data, split_sets

def organize_images(jpg_folder: str, output_folder: str, class_names: List[str],
                   lbl_data: List[int], split_sets: Dict[str, List[int]]) -> None:
    """
    Organize images into train/test/validation splits.
    
    Args:
        jpg_folder: Source directory containing original images
        output_folder: Base directory for processed dataset
        class_names: List of class names
        lbl_data: Image labels
        split_sets: Dictionary mapping split names to image IDs
    """
    img_files = sorted(os.listdir(jpg_folder))
    img_id_to_file = {
        int(fname.split("_")[1].split(".")[0]): fname
        for fname in img_files
    }

    for split, img_ids in split_sets.items():
        for img_id in img_ids:
            img_filename = img_id_to_file.get(img_id)
            if img_filename:
                img_path = os.path.join(jpg_folder, img_filename)
                class_idx = lbl_data[img_id - 1] - 1
                class_name = class_names[class_idx]
                dest_path = os.path.join(output_folder, split, class_name, img_filename)
                shutil.copy(img_path, dest_path)

def swap_train_test(base_path: str) -> None:
    """
    Swap train and test directories.
    
    Args:
        base_path: Base directory containing train and test folders
    """
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    temp_path = os.path.join(base_path, "temp_test")
    
    os.rename(train_path, temp_path)
    os.rename(test_path, train_path)
    os.rename(temp_path, test_path)
    print("Train and test folders swapped successfully.")

def process_dataset(jpg_folder: str, output_folder: str, lbl_mat_path: str,
                   split_mat_path: str, class_name_path: str, swap_splits: bool = True) -> None:
    """
    Main function to process the Oxford 102 Flowers dataset.
    
    Args:
        jpg_folder: Directory containing source images
        output_folder: Directory for processed dataset
        lbl_mat_path: Path to labels .mat file
        split_mat_path: Path to split assignments .mat file
        class_name_path: Path to class names text file
        swap_splits: Whether to swap train and test splits after processing
    """
    if os.path.exists(output_folder):
        print("Dataset already processed.")
        return

    # Read class names and create directory structure
    class_names = read_class_names(class_name_path)
    create_directory_structure(output_folder, class_names)
    
    # Load split information
    lbl_data, split_sets = load_split_data(lbl_mat_path, split_mat_path)
    
    # Organize images into splits
    organize_images(jpg_folder, output_folder, class_names, lbl_data, split_sets)
    
    # Optionally swap train and test splits
    if swap_splits:
        swap_train_test(output_folder)

def main():
    """Parse arguments and process the dataset."""
    parser = argparse.ArgumentParser(description='Process Oxford 102 Flowers Dataset')
    
    parser.add_argument('--jpg-folder', default=DEFAULT_JPG_FOLDER, help='Directory containing source images')
    parser.add_argument('--output-folder', default=DEFAULT_OUTPUT_FOLDER, help='Directory for processed dataset')
    parser.add_argument('--lbl-mat-path', default=DEFAULT_LBL_MAT_PATH, help='Path to labels .mat file')
    parser.add_argument('--split-mat-path', default=DEFAULT_SPLIT_MAT_PATH, help='Path to split assignments .mat file')
    parser.add_argument('--class-name-path', default=DEFAULT_CLASS_NAME_PATH, help='Path to class names text file')
    parser.add_argument('--swap-splits', action='store_true', help='Swap train and test splits after processing')
    
    args = parser.parse_args()
    
    process_dataset(
        jpg_folder=args.jpg_folder,
        output_folder=args.output_folder,
        lbl_mat_path=args.lbl_mat_path,
        split_mat_path=args.split_mat_path,
        class_name_path=args.class_name_path,
        swap_splits=args.swap_splits
    )

if __name__ == '__main__':
    main()