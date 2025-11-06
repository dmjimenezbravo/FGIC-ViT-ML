#!/usr/bin/env python3
"""
Script for training and evaluating machine learning models on various embeddings and datasets.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from src.ml_classifiers.train_classifiers import (
    setup_training_data,
    train_model,
    evaluate_model
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate ML models on embeddings'
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='data/embeddings',
        help='Base directory containing embeddings'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        choices=['CLIP', 'FRANCA', 'DINOv2', 'SigLIPv2'],
        help='Specific models to evaluate (default: all)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        type=str,
        help='Specific datasets to evaluate (default: all)'
    )
    parser.add_argument(
        '--balance_data',
        action='store_true',
        help='Whether to apply SMOTE for data balancing'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip combinations that already have results'
    )
    
    return parser.parse_args()

def load_embeddings(embeddings_dir: str) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """
    Load all embedding parquet files into a dictionary.
    
    Args:
        embeddings_dir: Base directory containing embeddings
        
    Returns:
        Dictionary mapping (model, dataset, split) to DataFrame
    """
    dataframes = {}
    embeddings_path = Path(embeddings_dir)
    
    for model_dir in embeddings_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            for parquet_file in dataset_dir.glob('*_embeddings.parquet'):
                split_name = parquet_file.stem.replace('_embeddings', '')
                key = (model_dir.name, dataset_dir.name, split_name)
                dataframes[key] = pd.read_parquet(parquet_file)
                
    return dataframes

def get_available_combinations(
    dataframes: Dict[Tuple[str, str, str], pd.DataFrame],
    selected_models: List[str] = None,
    selected_datasets: List[str] = None
) -> List[Tuple[str, str]]:
    """
    Get all available model-dataset combinations with complete splits.
    
    Args:
        dataframes: Dictionary of loaded DataFrames
        selected_models: Optional list of models to filter by
        selected_datasets: Optional list of datasets to filter by
        
    Returns:
        List of (model_name, dataset_name) tuples
    """
    models = {key[0] for key in dataframes.keys()}
    datasets = {key[1] for key in dataframes.keys()}
    
    if selected_models:
        models = models.intersection(selected_models)
    if selected_datasets:
        datasets = datasets.intersection(selected_datasets)
    
    combinations = []
    required_splits = ['train', 'validation', 'test']
    
    for model in sorted(models):
        for dataset in sorted(datasets):
            if all((model, dataset, split) in dataframes for split in required_splits):
                combinations.append((model, dataset))
                
    return combinations

def main():
    """Main execution function."""
    args = parse_args()
    
    print("Loading embeddings...")
    dataframes = load_embeddings(args.embeddings_dir)
    
    print("Finding available combinations...")
    combinations = get_available_combinations(dataframes, args.models, args.datasets)
    
    if not combinations:
        print("No valid model-dataset combinations found!")
        return
        
    print(f"Found {len(combinations)} valid combinations")
    
    for model_name, dataset_name in combinations:
        print(f"\nProcessing {model_name} - {dataset_name}")
        
        # Create results directory
        results_subdir = os.path.join(args.results_dir, model_name, dataset_name)
        if args.skip_existing and os.path.exists(results_subdir):
            print("Results exist, skipping...")
            continue
            
        # Get data splits
        df_train = dataframes[(model_name, dataset_name, 'train')]
        df_val = dataframes[(model_name, dataset_name, 'validation')]
        df_test = dataframes[(model_name, dataset_name, 'test')]
        
        # Prepare data
        train_data, val_data, test_data = setup_training_data(df_train, df_val, df_test)
        
        print("Training models...")
        best_model, results = train_model(
            train_data,
            val_data,
            model_name=model_name,
            dataset_name=dataset_name,
            balance_data=args.balance_data,
            results_dir=args.results_dir
        )
        
        print("Evaluating on test set...")
        eval_results = evaluate_model(
            best_model,
            test_data,
            model_name=model_name,
            dataset_name=dataset_name,
            results_dir=args.results_dir
        )
        
        print(f"Results saved in {results_subdir}")

if __name__ == "__main__":
    main()