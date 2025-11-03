"""
Extract embeddings from images using various ViT models.

This script can be run directly to extract embeddings from datasets:
    python extract_embeddings.py --datasets_dir path/to/datasets --output_dir path/to/output
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from pathlib import Path

# Add root directory to path for importing project modules
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from src.vit_embeddings.utils_vits import (
    get_device,
    load_vit_model,
    get_embedding_output,
    check_model_requirements,
    get_model_configs,
    get_model_transforms
)
from src.data.dataset_loader import load_datasets

def extract_embeddings_batch(
    image_list: List[Dict],
    model_name: str,
    batch_size: int = 64,
    device: Optional[torch.device] = None
) -> List[Dict]:
    """
    Extract embeddings from a batch of images.
    
    Args:
        image_list (List[Dict]): List of image information dictionaries
        model_name (str): Name of the model to use
        batch_size (int): Batch size for processing
        device (torch.device, optional): Device to run extraction on
        
    Returns:
        List[Dict]: List of dictionaries containing embeddings and metadata
    """
    # Check model requirements
    requirements_met, message = check_model_requirements(model_name)
    if not requirements_met:
        raise ImportError(f"Model requirements not met: {message}")
    
    # Configure device and model
    if device is None:
        device = get_device()
    
    model = load_vit_model(model_name, device)
    transform = get_model_transforms(model_name)
    
    extracted = []
    if not image_list:
        return extracted

    for i in tqdm(range(0, len(image_list), batch_size), desc=f"{model_name} embeddings"):
        batch = image_list[i:i+batch_size]
        imgs, paths, labels = [], [], []

        for item in batch:
            try:
                img = Image.open(item['image_path']).convert('RGB')
                if model_name == 'SigLIPv2':
                    proc = transform(images=img, return_tensors='pt')
                    imgs.append(proc['pixel_values'])
                else:
                    imgs.append(transform(img).unsqueeze(0))
                paths.append(item['image_path'])
                labels.append(item['label'])
            except Exception as e:
                print(f"Error loading {item['image_path']}: {e}")
                continue

        if not imgs:
            continue

        batch_tensor = torch.cat(imgs, dim=0).to(device)
        emb = get_embedding_output(model, batch_tensor, model_name)
        emb_np = emb.cpu().numpy()

        for j, p in enumerate(paths):
            extracted.append({
                'image_path': p,
                'label': labels[j],
                'embedding': emb_np[j]
            })
            
    return extracted

def save_embeddings(
    data: List[Dict],
    output_dir: str,
    model_name: str,
    dataset_name: str,
    subset: str
) -> None:
    """
    Save extracted embeddings to disk.
    
    Args:
        data (List[Dict]): List of dictionaries containing embeddings
        output_dir (str): Base directory to save embeddings
        model_name (str): Name of the model used
        dataset_name (str): Name of the dataset
        subset (str): Subset name (train/test/validation)
    """
    if not data:
        return
        
    model_out = os.path.join(output_dir, model_name)
    dataset_out_dir = os.path.join(model_out, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # Split metadata and embeddings
    df_meta = pd.DataFrame(
        [{'image_path': it['image_path'], 'label': it['label']} for it in data]
    )
    emb_arr = np.array([it['embedding'] for it in data])
    
    # Create DataFrame of embeddings with column names
    df_emb = pd.DataFrame(
        emb_arr,
        columns=[f"dim_{i}" for i in range(emb_arr.shape[1])]
    )
    
    # Concatenate metadata and embeddings
    df = pd.concat([df_meta, df_emb], axis=1)

    # Save as parquet
    fname = f"{subset}_embeddings.parquet"
    output_path = os.path.join(dataset_out_dir, fname)
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path} — shape {df.shape}")



def get_datasets_config():
    """
    Get the configuration for all datasets to process.
    Returns a list of dictionaries containing dataset configurations.
    """
    return [
        {
            'name': 'NABirds',
            'base_folder_in_datasets_dir': 'NABirds',
            'subsets_map': {
                'train': 'train',
                'test': 'test',
                'validation': 'validation'
            }
        },
        {
            'name': 'CUB_200_2011',
            'base_folder_in_datasets_dir': 'CUB_200_2011',
            'subsets_map': {
                'train': 'train',
                'test': 'test',
                'validation': 'validation'
            }
        },
        {
            'name': 'FGVC-Aircraft',
            'base_folder_in_datasets_dir': 'FGVC-Aircraft',
            'subsets_map': {
                'train': 'train',
                'test': 'test',
                'validation': 'validation'
            }
        },
        {
            'name': 'Oxford 102 Flower',
            'base_folder_in_datasets_dir': 'Oxford 102 flower',
            'subsets_map': {
                'train': 'train',
                'test': 'test',
                'validation': 'validation'
            }
        },
        {
            'name': 'Stanford Cars',
            'base_folder_in_datasets_dir': 'Stanford Cars',
            'subsets_map': {
                'train': 'train',
                'test': 'test',
                'validation': 'validation'
            }
        },
        {
            'name': 'Stanford Dogs',
            'base_folder_in_datasets_dir': 'Stanford Dogs',
            'subsets_map': {
                'train': 'train',
                'test': 'test',
                'validation': 'validation'
            }
        }
    ]

def parse_arguments():
    """
    Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract embeddings from image datasets using ViT models'
    )
    
    parser.add_argument('--datasets_dir', type=str, required=True,
                      help='Base directory containing all datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory where embeddings will be saved')
    parser.add_argument('--models', nargs='+', type=str,
                      default=['FRANCA', 'DINOv2', 'CLIP', 'SigLIPv2'],
                      help='List of models to use for extraction')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for processing')
    parser.add_argument('--datasets', nargs='+', type=str,
                      help='Specific datasets to process. If not provided, all datasets will be processed')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.datasets_dir):
        raise ValueError(f"Datasets directory not found: {args.datasets_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def filter_datasets_config(datasets_config: List[Dict], selected_datasets: Optional[List[str]] = None) -> List[Dict]:
    """
    Filter dataset configurations based on selected datasets.
    
    Args:
        datasets_config (List[Dict]): Complete list of dataset configurations
        selected_datasets (List[str], optional): Names of datasets to include
        
    Returns:
        List[Dict]: Filtered configurations
    """
    if not selected_datasets:
        return datasets_config
        
    filtered_config = [cfg for cfg in datasets_config if cfg['name'] in selected_datasets]
    if not filtered_config:
        raise ValueError(f"No valid datasets found among: {selected_datasets}")
        
    return filtered_config

def process_dataset_subset(
    images_info: List[Dict],
    model_name: str,
    dataset_name: str,
    subset_name: str,
    output_dir: str,
    batch_size: int
) -> None:
    """
    Process a specific dataset subset.
    
    Args:
        images_info (List[Dict]): Information about images to process
        model_name (str): Name of the model to use
        dataset_name (str): Name of the dataset
        subset_name (str): Name of the subset (train/test/validation)
        output_dir (str): Output directory
        batch_size (int): Batch size
    """
    if not images_info:
        print(f"Skipping empty subset: {subset_name}")
        return
        
    print(f"Extracting embeddings for {subset_name} split "
          f"({len(images_info)} images)")
    
    # Extract embeddings
    embeddings = extract_embeddings_batch(
        images_info,
        model_name,
        batch_size=batch_size
    )
    
    if embeddings:
        # Save embeddings
        save_embeddings(
            embeddings,
            output_dir,
            model_name,
            dataset_name,
            subset_name
        )
    else:
        print(f"No embeddings extracted for {subset_name}")

def process_model(
    model_name: str,
    all_datasets_info: Dict,
    output_dir: str,
    batch_size: int
) -> None:
    """
    Process all datasets for a specific model.
    
    Args:
        model_name (str): Name of the model to use
        all_datasets_info (Dict): Information about all datasets
        output_dir (str): Output directory
        batch_size (int): Batch size
    """
    print(f"\n=== Processing model: {model_name} ===")
    
    try:
        # Verify model requirements
        requirements_met, message = check_model_requirements(model_name)
        if not requirements_met:
            print(f"Skipping {model_name}: {message}")
            return

        # Process each dataset
        for dataset_name, subsets in all_datasets_info.items():
            print(f"\nProcessing {dataset_name}...")
            
            for subset_name, images_info in subsets.items():
                process_dataset_subset(
                    images_info,
                    model_name,
                    dataset_name,
                    subset_name,
                    output_dir,
                    batch_size
                )
                    
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")

def main():
    """
    Main function for embedding extraction.
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Load and filter dataset configuration
        datasets_config = get_datasets_config()
        datasets_config = filter_datasets_config(datasets_config, args.datasets)

        # Load dataset information
        print("\nLoading datasets information...")
        all_datasets_info = load_datasets(args.datasets_dir, datasets_config)

        # Process each model
        for model_name in args.models:
            process_model(
                model_name,
                all_datasets_info,
                args.output_dir,
                args.batch_size
            )
        
        print("\nExtraction completed!")
        
    except Exception as e:
        print(f"\nError during extraction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()