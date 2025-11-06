"""
Machine Learning Classifiers Package

This package provides functionality for training and evaluating machine learning
classifiers on embeddings extracted from Vision Transformer models. The package
includes tools for:

- Training classifiers using PyCaret
- Evaluating model performance
- Handling data imbalance with SMOTE
- Exporting results in LaTeX format
"""

from .train_classifiers import (
    setup_training_data,
    train_model,
    evaluate_model
)

from .train_evaluate_models import (
    load_embeddings,
    get_available_combinations
)

__all__ = [
    # Training and evaluation core functions
    'setup_training_data',
    'train_model',
    'evaluate_model',
    
    # Data loading and management
    'load_embeddings',
    'get_available_combinations'
]