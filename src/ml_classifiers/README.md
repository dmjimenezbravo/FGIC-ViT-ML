# Machine Learning Classifiers Module

This directory contains modules for training and evaluating machine learning classifiers using embeddings extracted from Vision Transformer models. The modules provide standardized functionality for model training, evaluation, and results export.

## Directory Structure

```
ml_classifiers/
├── __init__.py              # Package initialization and exports
├── train_classifiers.py     # Core training and evaluation functions
└── train_evaluate_models.py # Command-line script for batch processing
```

## Module Usage

### As Python Modules

#### Basic Import
```python
from src.ml_classifiers import (
    setup_training_data,
    train_model,
    evaluate_model,
    load_embeddings,
    get_available_combinations
)
```

#### Individual Module Usage Examples

##### Training and Evaluating a Single Model
```python
from src.ml_classifiers import setup_training_data, train_model, evaluate_model

# Prepare data
train_data, val_data, test_data = setup_training_data(df_train, df_val, df_test)

# Train model
best_model, results = train_model(
    train_data,
    val_data,
    model_name="CLIP",
    dataset_name="CUB-200-2011",
    balance_data=True,
    results_dir="results"
)

# Evaluate on test set
eval_results = evaluate_model(
    best_model,
    test_data,
    model_name="CLIP",
    dataset_name="CUB-200-2011",
    results_dir="results"
)
```

##### Processing Multiple Models and Datasets
```python
from src.ml_classifiers import load_embeddings, get_available_combinations

# Load all embeddings
dataframes = load_embeddings("data/embeddings")

# Get available combinations
combinations = get_available_combinations(
    dataframes,
    selected_models=["CLIP", "FRANCA"],
    selected_datasets=["CUB-200-2011", "Stanford Cars"]
)

# Process each combination
for model_name, dataset_name in combinations:
    # Get data splits
    df_train = dataframes[(model_name, dataset_name, 'train')]
    df_val = dataframes[(model_name, dataset_name, 'validation')]
    df_test = dataframes[(model_name, dataset_name, 'test')]
    
    # Train and evaluate
    train_data, val_data, test_data = setup_training_data(df_train, df_val, df_test)
    best_model, results = train_model(train_data, val_data, model_name, dataset_name)
    eval_results = evaluate_model(best_model, test_data, model_name, dataset_name)
```

### As Command Line Script

The module includes a command-line script for batch processing of multiple model-dataset combinations:

```bash
python src/ml_classifiers/train_evaluate_models.py \
    --embeddings_dir data/embeddings \
    --results_dir results \
    --models CLIP FRANCA DINOv2 SigLIPv2 \
    --balance_data \
    --skip_existing
```

#### Command Line Arguments

- `--embeddings_dir`: Base directory containing embeddings (default: 'data/embeddings')
- `--results_dir`: Directory to save results (default: 'results')
- `--models`: Specific models to evaluate (optional)
- `--datasets`: Specific datasets to evaluate (optional)
- `--balance_data`: Apply SMOTE for data balancing
- `--skip_existing`: Skip combinations that already have results

## Output Format

Results are saved in a structured directory format:

```
results/
└── {model_name}/
    └── {dataset_name}/
        ├── model_comparison.tex  # Model comparison results
        ├── test_metrics.tex      # Test set evaluation metrics
        └── confusion_matrix.png  # Confusion matrix visualization
```

Each `.tex` file contains:
- `model_comparison.tex`: Table comparing performance of all tried models
- `test_metrics.tex`: Detailed metrics for the best model on test set