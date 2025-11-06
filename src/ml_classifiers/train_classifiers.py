"""
Training and evaluation of machine learning classifiers using PyCaret, with LaTeX results export.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from pycaret.classification import *
from imblearn.over_sampling import SMOTE

def create_results_dir(
    model_name: str,
    dataset_name: str,
    base_results_dir: str = 'results'
) -> str:
    """
    Create directory structure for storing results.
    
    Args:
        model_name (str): Name of the embeddings model (e.g., 'CLIP', 'FRANCA')
        dataset_name (str): Name of the dataset
        base_results_dir (str): Base directory for results
        
    Returns:
        str: Path to the results directory
    """
    results_path = Path(base_results_dir) / model_name / dataset_name
    results_path.mkdir(parents=True, exist_ok=True)
    return str(results_path)

def save_latex_table(
    df: pd.DataFrame,
    output_path: str,
    caption: str,
    label: str
) -> None:
    """
    Save DataFrame as LaTeX table.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the LaTeX file
        caption (str): Table caption
        label (str): Table label for references
    """
    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        escape=True
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)

def setup_training_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare dataframes for training.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_val (pd.DataFrame): Validation data
        df_test (pd.DataFrame): Test data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Processed dataframes
    """
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True)
    )

def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    balance_data: bool = False,
    results_dir: Optional[str] = None,
    **setup_params
) -> Tuple[Any, pd.DataFrame]:
    """
    Train a model using PyCaret with optional data balancing and results export.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_val (pd.DataFrame): Validation data
        model_name (str): Name of the embeddings model
        dataset_name (str): Name of the dataset
        balance_data (bool): Whether to apply SMOTE balancing
        results_dir (str, optional): Base directory for results
        **setup_params: Additional parameters for setup
        
    Returns:
        Tuple[Any, pd.DataFrame]: Best model and results table
    """
    setup_params = {
        'data': df_train,
        'test_data': df_val,
        'target': 'label',
        'session_id': 123,
        'index': False,
        **setup_params
    }
    
    if balance_data:
        setup_params.update({
            'fix_imbalance': True,
            'fix_imbalance_method': SMOTE(k_neighbors=2)
        })
    
    setup(**setup_params)
    
    best_model = compare_models(
        include=['dummy', 'lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm',
                 'mlp', 'ridge', 'rf', 'qda', 'ada', 'lda', 'et'],
        cross_validation=False
    )
    
    results = pull()
    
    # Save results if directory is provided
    if results_dir:
        results_path = create_results_dir(model_name, dataset_name, results_dir)
        
        # Save model comparison results
        comparison_path = os.path.join(results_path, 'model_comparison.tex')
        save_latex_table(
            results,
            comparison_path,
            f"Model Comparison for {dataset_name} using {model_name} embeddings",
            f"tab:model_comparison_{model_name}_{dataset_name}"
        )
    
    return best_model, results

def evaluate_model(
    model: Any,
    df_test: pd.DataFrame,
    model_name: str,
    dataset_name: str,
    results_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate a trained model on test data and save results.
    
    Args:
        model: Trained PyCaret model
        df_test (pd.DataFrame): Test data
        model_name (str): Name of the embeddings model
        dataset_name (str): Name of the dataset
        results_dir (str, optional): Base directory for results
        
    Returns:
        Dict: Dictionary containing predictions and metrics
    """
    predictions = predict_model(model, data=df_test)
    metrics = predictions.drop(['label', 'prediction_label', 'prediction_score'], axis=1)
    
    if results_dir:
        results_path = create_results_dir(model_name, dataset_name, results_dir)
        
        # Save test metrics
        metrics_path = os.path.join(results_path, 'test_metrics.tex')
        save_latex_table(
            pd.DataFrame([metrics.iloc[0]]).T.reset_index(),
            metrics_path,
            f"Test Metrics for {dataset_name} using {model_name} embeddings",
            f"tab:test_metrics_{model_name}_{dataset_name}"
        )
        
        # Save confusion matrix
        plot_model(model, plot='confusion_matrix', save=os.path.join(results_path, 'confusion_matrix.png'))
        
    return {
        'predictions': predictions,
        'metrics': metrics,
        'model': model
    }