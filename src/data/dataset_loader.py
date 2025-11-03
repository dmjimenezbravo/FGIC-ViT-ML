"""
Dataset loading utilities for processing image datasets with standardized directory structure.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

def get_image_paths_and_labels(base_path_to_data_folder: str) -> List[Dict[str, str]]:
    """
    Get image paths and their corresponding labels from a directory structure.
    
    Args:
        base_path_to_data_folder (str): Base path to the dataset directory
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing image paths and labels
    """
    image_data = []
    if not os.path.exists(base_path_to_data_folder):
        print(f"Warning: Directory does not exist, skipping: {base_path_to_data_folder}")
        return []
        
    for root, _, files in os.walk(base_path_to_data_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_image_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_image_path, base_path_to_data_folder)
                parts = rel_path.split(os.sep)
                label = os.path.join(*parts[:-1]) if len(parts) >= 2 else "unknown"
                image_data.append({'image_path': full_image_path, 'label': label})
                
    return image_data

def load_datasets(base_datasets_dir: str, datasets_config: List[Dict]) -> Dict:
    """
    Load multiple datasets according to their configuration.
    
    Args:
        base_datasets_dir (str): Base directory containing all datasets
        datasets_config (List[Dict]): List of dataset configurations
        
    Returns:
        Dict: Dictionary containing dataset information
    """
    all_datasets_info = {}
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        base_folder = dataset_config['base_folder_in_datasets_dir']
        subsets_map = dataset_config['subsets_map']
        
        current_dataset_root = os.path.join(base_datasets_dir, base_folder)
            
        if not os.path.exists(current_dataset_root):
            print(f"Dataset root not found: {current_dataset_root}")
            continue
            
        all_datasets_info[dataset_name] = {}
        for standard_subset_name, actual_folder_name in subsets_map.items():
            subset_path = os.path.join(current_dataset_root, actual_folder_name)
            images_info = get_image_paths_and_labels(subset_path)
            all_datasets_info[dataset_name][standard_subset_name] = images_info
            print(f"{dataset_name} - {standard_subset_name}: {len(images_info)} images")
            
    return all_datasets_info