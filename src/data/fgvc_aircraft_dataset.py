#!/usr/bin/env python3
"""
FGVC Aircraft Dataset Preparation Script

This script processes the FGVC Aircraft Dataset by:
1. Reading class names from variants.txt
2. Creating directory structure for train/test/validation splits
3. Reading split assignments from individual files
4. Organizing images into appropriate splits
"""

import os
import shutil
import argparse
from typing import Dict, List, Tuple

# Default paths
DEFAULT_JPG_FOLDER = "./data/raw/FGVC-Aircraft/fgvc-aircraft-2013b/data/images"
DEFAULT_OUTPUT_FOLDER = "./data/processed/FGVC-Aircraft"
DEFAULT_CLASS_NAME_PATH = "./data/raw/FGVC-Aircraft/fgvc-aircraft-2013b/data/variants.txt"
DEFAULT_SPLIT_FILES = {
    "train": "./data/raw/FGVC-Aircraft/fgvc-aircraft-2013b/data/images_variant_train.txt",
    "validation": "./data/raw/FGVC-Aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt",
    "test": "./data/raw/FGVC-Aircraft/fgvc-aircraft-2013b/data/images_variant_test.txt"
}

def create_directory_structure(output_folder: str, splits: List[str]) -> None:
    """
    Create the necessary directory structure for the dataset.
    
    Args:
        output_folder: Path to the main output directory
        splits: List of split names (e.g., ['train', 'validation', 'test'])
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for split in splits:
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)

def read_class_names(class_name_path: str) -> List[str]:
    """
    Read class names from the variants file.
    
    Args:
        class_name_path: Path to the variants.txt file
    
    Returns:
        List of class names
    """
    with open(class_name_path, 'r') as f:
        return [line.strip() for line in f]

def create_class_directories(output_folder: str, splits: List[str], class_names: List[str]) -> None:
    """
    Create directories for each class within each split.
    
    Args:
        output_folder: Path to the main output directory
        splits: List of split names
        class_names: List of class names
    """
    for split in splits:
        for class_name in class_names:
            os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)

def organize_images(jpg_folder: str, output_folder: str, split_files: Dict[str, str], move: bool = False) -> None:
    """
    Organize images into their respective split and class directories.
    
    Args:
        jpg_folder: Path to the folder containing all images
        output_folder: Path to the main output directory
        split_files: Dictionary mapping split names to their file paths
        move: If True, move files instead of copying them
    """
    for split, split_path in split_files.items():
        with open(split_path, 'r') as f:
            for line in f:
                image_name, class_name = line.strip().split(' ', 1)
                img_src = os.path.join(jpg_folder, image_name + '.jpg')
                dest_dir = os.path.join(output_folder, split, class_name)

                if not os.path.isfile(img_src):
                    print(f"Image not found: {img_src}")
                    continue

                os.makedirs(dest_dir, exist_ok=True)
                if move:
                    shutil.move(img_src, os.path.join(dest_dir, os.path.basename(img_src)))
                else:
                    shutil.copy2(img_src, dest_dir)

def process_fgvc_aircraft_dataset(
    jpg_folder: str = DEFAULT_JPG_FOLDER,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    class_name_path: str = DEFAULT_CLASS_NAME_PATH,
    split_files: Dict[str, str] = DEFAULT_SPLIT_FILES,
    move: bool = False
) -> None:
    """
    Process the FGVC Aircraft dataset by organizing images into train/validation/test splits.
    
    Args:
        jpg_folder: Path to the folder containing all images
        output_folder: Path to the main output directory
        class_name_path: Path to the variants.txt file
        split_files: Dictionary mapping split names to their file paths
        move: If True, move files instead of copying them
    """
    if os.path.exists(output_folder):
        print("Output directory already exists. Images might have been already extracted.")
        return

    splits = ["train", "validation", "test"]
    
    # Create directory structure
    create_directory_structure(output_folder, splits)
    
    # Read class names and create class directories
    class_names = read_class_names(class_name_path)
    create_class_directories(output_folder, splits, class_names)
    
    # Organize images into splits
    organize_images(jpg_folder, output_folder, split_files, move)
    print("Dataset processing completed successfully.")

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process FGVC Aircraft dataset into train/validation/test splits")
    parser.add_argument("--jpg-folder", default=DEFAULT_JPG_FOLDER,
                      help="Directory containing the image files")
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER,
                      help="Directory where processed dataset will be saved")
    parser.add_argument("--class-name-path", default=DEFAULT_CLASS_NAME_PATH,
                      help="Path to the variants.txt file")
    parser.add_argument("--move", action="store_true",
                      help="Move files instead of copying them")
    
    args = parser.parse_args()
    
    # Use default split files - these are typically fixed for the dataset
    process_fgvc_aircraft_dataset(
        jpg_folder=args.jpg_folder,
        output_folder=args.output_folder,
        class_name_path=args.class_name_path,
        split_files=DEFAULT_SPLIT_FILES,
        move=args.move
    )

if __name__ == "__main__":
    main()