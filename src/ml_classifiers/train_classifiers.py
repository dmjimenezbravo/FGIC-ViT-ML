"""
Training and evaluation of machine learning classifiers using PyCaret.
"""
import pandas as pd
from typing import Dict, Tuple, Any
from pycaret.classification import *
from imblearn.over_sampling import SMOTE

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
    balance_data: bool = False,
    **setup_params
) -> Tuple[Any, pd.DataFrame]:
    """
    Train a model using PyCaret with optional data balancing.
    
    Args:
        df_train (pd.DataFrame): Training data
        df_val (pd.DataFrame): Validation data
        balance_data (bool): Whether to apply SMOTE balancing
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
    return best_model, results

def evaluate_model(model: Any, df_test: pd.DataFrame) -> Dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained PyCaret model
        df_test (pd.DataFrame): Test data
        
    Returns:
        Dict: Dictionary containing predictions and metrics
    """
    predictions = predict_model(model, data=df_test)
    return {
        'predictions': predictions,
        'model': model
    }