#!/usr/bin/env python3
"""
NABirds Dataset Preparation Script

This script processes the NABirds Dataset by:
1. Reading metadata files (classes, image labels, splits)
2. Creating directory structure for train/test/validation splits
3. Organizing images into appropriate splits
4. Creating validation split from training data
5. Cleaning empty folders and consolidating nested directories
"""

import os
import random
import shutil
import argparse
from typing import Dict, List, Tuple, Set

# Default paths
DEFAULT_JPG_FOLDER = "./../data/raw/NABirds/images"
DEFAULT_OUTPUT_FOLDER = "./../data/processed/NABirds/"
DEFAULT_LBL_TXT_PATH = "./../data/raw/NABirds/image_class_labels.txt"
DEFAULT_SPLIT_TXT_PATH = "./../data/raw/NABirds/train_test_split.txt"
DEFAULT_CLASS_NAME_PATH = "./../data/raw/NABirds/classes.txt"
DEFAULT_IMG_NAME_PATH = "./../data/raw/NABirds/images.txt"
DEFAULT_VAL_RATIO = 0.2

# Global constants
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

def create_directory_structure(output_folder: str) -> None:
    """
    Create initial directory structure for dataset splits.
    
    Args:
        output_folder: Base path for processed dataset
    """
    splits = ['train', 'test', 'validation']
    for split in splits:
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)

def read_class_names(class_name_path: str) -> Dict[str, str]:
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
            class_names[class_id] = class_name
    return class_names

def create_class_directories(output_folder: str, class_names: Dict[str, str]) -> None:
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

def read_image_paths(img_name_path: str) -> Dict[str, str]:
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
            img_id_to_path[img_id] = img_rel_path
    return img_id_to_path

def read_image_classes(lbl_txt_path: str) -> Dict[str, str]:
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
            img_id, class_id = line.strip().split()
            img_id_to_class[img_id] = class_id
    return img_id_to_class

def read_split_assignments(split_txt_path: str) -> Dict[str, List[str]]:
    """
    Read train/test split assignments from file.
    
    Args:
        split_txt_path: Path to train_test_split.txt file
    
    Returns:
        Dictionary mapping split names to lists of image IDs
    """
    train_ids = []
    test_ids = []
    with open(split_txt_path, 'r') as f:
        for line in f:
            img_id, is_train_str = line.strip().split()
            if int(is_train_str) == 1:
                train_ids.append(img_id)
            else:
                test_ids.append(img_id)
    return {'train': train_ids, 'test': test_ids}

def organize_images(jpg_folder: str, output_folder: str, 
                   img_id_to_path: Dict[str, str],
                   img_id_to_class: Dict[str, str],
                   class_names: Dict[str, str],
                   split_mapping: Dict[str, List[str]]) -> None:
    """
    Organize images into appropriate splits based on metadata.
    
    Args:
        jpg_folder: Source directory containing original images
        output_folder: Base directory for processed dataset
        img_id_to_path: Dictionary mapping image IDs to relative paths
        img_id_to_class: Dictionary mapping image IDs to class IDs
        class_names: Dictionary mapping class IDs to class names
        split_mapping: Dictionary mapping split names to image ID lists
    """
    for split, img_ids in split_mapping.items():
        for img_id in img_ids:
            rel_path = img_id_to_path.get(img_id)
            if rel_path is None:
                print(f"Image ID not found in images.txt: {img_id}")
                continue

            class_id = img_id_to_class.get(img_id)
            if class_id is None:
                print(f"Image ID not found in image_class_labels.txt: {img_id}")
                continue

            class_name = class_names.get(class_id)
            if class_name is None:
                print(f"Class ID not found in classes.txt: {class_id}")
                continue

            src_path = os.path.join(jpg_folder, rel_path)
            dest_path = os.path.join(output_folder, split, class_name, os.path.basename(rel_path))

            if os.path.isfile(src_path):
                shutil.move(src_path, dest_path)
            else:
                print(f"Image not found: {src_path}")

def folder_has_images(folder_path: str) -> bool:
    """
    Check whether a folder (or any of its subfolders) contains image files.

    Args:
        folder_path: Path to the folder to search recursively

    Returns:
        True if at least one file with an image extension is found
    """
    for root, _, files in os.walk(folder_path):
        if any(file.lower().endswith(IMAGE_EXTENSIONS) for file in files):
            return True
    return False

def clean_empty_folders(split_dir: str) -> List[str]:
    """
    Remove folders that contain no image files and record their paths.

    Args:
        split_dir: Path to the root directory to scan

    Returns:
        List of paths to deleted folders
    """
    deleted_folders = []
    for root, dirs, _ in os.walk(split_dir, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not folder_has_images(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    deleted_folders.append(dir_path)
                except Exception as e:
                    print(f"Could not delete {dir_path}: {e}")
    return deleted_folders

def identify_small_classes(train_dir: str, min_images: int = 2) -> List[str]:
    """
    Identify classes with fewer than min_images images.

    Args:
        train_dir: Path to training directory
        min_images: Minimum number of images required

    Returns:
        List of class names with insufficient images
    """
    skipped_class_names = []
    for class_name in os.listdir(train_dir):
        class_train_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_train_dir):
            continue

        image_paths = []
        for root, _, files in os.walk(class_train_dir):
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, f))

        if len(image_paths) < min_images:
            skipped_class_names.append(class_name)
    
    return skipped_class_names

def consolidate_nested_folders(folder_path: str, initial_path: str = None, 
                             accumulated_name: str = "") -> None:
    """
    Consolidate nested folder structure by moving images to flattened directories.

    Args:
        folder_path: Current folder being processed
        initial_path: Root path where consolidated folders will be created
        accumulated_name: Accumulated name for the consolidated folder
    """
    if initial_path is None:
        initial_path = folder_path

    elements = os.listdir(folder_path)
    folders = [e for e in elements if os.path.isdir(os.path.join(folder_path, e))]
    files = [e for e in elements if os.path.isfile(os.path.join(folder_path, e))]

    if files:
        if accumulated_name:
            final_name = accumulated_name
        else:
            print(f"Folder '{os.path.basename(folder_path)}' already contains images. No action taken.")
            return

        new_path = os.path.join(initial_path, final_name)
        os.makedirs(new_path, exist_ok=True)

        print(f"Moving files from '{folder_path}' to '{new_path}'")
        for file in files:
            source = os.path.join(folder_path, file)
            destination = os.path.join(new_path, file)
            shutil.move(source, destination)

        try:
            os.rmdir(folder_path)
            print(f"Empty folder deleted: {folder_path}")
        except OSError:
            pass

    elif folders:
        for sub in folders:
            new_accumulated_name = f"{accumulated_name}-{sub}" if accumulated_name else sub
            consolidate_nested_folders(
                os.path.join(folder_path, sub),
                initial_path,
                new_accumulated_name
            )
            try:
                os.rmdir(os.path.join(folder_path, sub))
            except OSError:
                pass

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

        image_paths = []
        for root, _, files in os.walk(class_train_dir):
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, f))

        if len(image_paths) < 2:
            skipped_classes += 1
            continue

        random.shuffle(image_paths)
        val_count = min(max(1, int(len(image_paths) * val_ratio)), 
                       len(image_paths) - 1)
        selected_images = image_paths[:val_count]

        for src in selected_images:
            relative_path = os.path.relpath(src, class_train_dir)
            dst = os.path.join(class_val_dir, relative_path)
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
            total_moved += 1

    return total_moved, skipped_classes

def process_dataset(jpg_folder: str, output_folder: str, lbl_txt_path: str,
                   split_txt_path: str, class_name_path: str, img_name_path: str,
                   val_ratio: float = 0.2, cleanup: bool = True) -> None:
    """
    Main function to process the NABirds dataset.

    Args:
        jpg_folder: Directory containing source images
        output_folder: Directory for processed dataset
        lbl_txt_path: Path to image_class_labels.txt
        split_txt_path: Path to train_test_split.txt
        class_name_path: Path to classes.txt
        img_name_path: Path to images.txt
        val_ratio: Proportion of training data to use for validation
        cleanup: Whether to clean empty folders and consolidate nested directories
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
    split_mapping = read_split_assignments(split_txt_path)
    
    # Create class directories
    create_class_directories(output_folder, class_names)
    
    # Organize images into splits
    organize_images(jpg_folder, output_folder, img_id_to_path, 
                   img_id_to_class, class_names, split_mapping)
    
    # Create validation split
    total_moved, skipped = create_validation_split(output_folder, val_ratio)
    print(f"{total_moved} images were moved to 'validation'.")
    print(f"{skipped} classes were skipped due to having less than 2 images.")
    
    if cleanup:
        # Clean empty folders
        for split in ['train', 'test', 'validation']:
            split_dir = os.path.join(output_folder, split)
            deleted = clean_empty_folders(split_dir)
            print(f"\nCleaned {len(deleted)} empty folders in {split} split")
            
        # Identify problematic classes
        small_classes = identify_small_classes(os.path.join(output_folder, 'train'))
        if small_classes:
            print("\nClasses with insufficient images:")
            for name in small_classes:
                print(f" - {name}")

def main():
    """Parse arguments and process the dataset."""
    parser = argparse.ArgumentParser(description='Process NABirds Dataset')
    
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
    parser.add_argument('--no-cleanup', action='store_true',
                      help='Skip cleaning empty folders and checking for small classes')
    
    args = parser.parse_args()
    
    process_dataset(
        jpg_folder=args.jpg_folder,
        output_folder=args.output_folder,
        lbl_txt_path=args.lbl_txt_path,
        split_txt_path=args.split_txt_path,
        class_name_path=args.class_name_path,
        img_name_path=args.img_name_path,
        val_ratio=args.val_ratio,
        cleanup=not args.no_cleanup
    )

if __name__ == '__main__':
    main()
