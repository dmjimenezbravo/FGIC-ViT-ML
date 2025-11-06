# Source Code Directory

This directory contains the source code for the FGIC-ViT-ML project. The code is organized into modules that handle different aspects of the fine-grained image classification pipeline.

## Directory Structure

```
src/
├── data/                  # Dataset handling and preprocessing
│   ├── __init__.py
│   ├── cub200_dataset.py         # CUB-200-2011 dataset utilities
│   ├── dataset_loader.py         # Common dataset loading functionality
│   ├── fgvc_aircraft_dataset.py  # FGVC Aircraft dataset utilities
│   ├── nabirds_dataset.py        # NABirds dataset utilities
│   ├── oxford_flowers_dataset.py # Oxford 102 Flowers dataset utilities
│   ├── stanford_cars_dataset.py  # Stanford Cars dataset utilities
│   ├── stanford_dogs_dataset.py  # Stanford Dogs dataset utilities
│   └── utils_visualization.py    # Visualization utilities
├── ml_classifiers/       # Machine learning model training and evaluation
│   ├── __init__.py
│   ├── metrics.py              # Performance metrics calculation
│   ├── train_classifiers.py    # Core ML training functionality
│   └── train_evaluate_models.py # Training pipeline and evaluation
└── vit_embeddings/      # Vision Transformer embeddings extraction
    ├── __init__.py
    ├── extract_embeddings.py   # Main embeddings extraction pipeline
    └── utils_vits.py          # Vision Transformer utilities

```

## Module Descriptions

### Data Module (`data/`)

- **Dataset Loader** (`dataset_loader.py`):
  - Common dataset loading functionality
  - Data preprocessing pipelines
  - Split management (train/val/test)

- **Dataset Modules**: Each dataset has its dedicated module (`*_dataset.py`) that provides:
  - Data loading and preprocessing functions
  - Class mapping and label handling
  - Dataset-specific utilities
  - Data augmentation when applicable

- **Visualization Utilities** (`utils_visualization.py`):
  - Functions for visualizing dataset samples
  - Class distribution plots
  - Results visualization helpers

### Machine Learning Module (`ml_classifiers/`)

- **Training Core** (`train_classifiers.py`):
  - Implementation of ML training pipelines
  - Model selection and hyperparameter tuning
  - Cross-validation utilities
  - Performance metric calculations

- **Evaluation Pipeline** (`train_evaluate_models.py`):
  - End-to-end training workflow
  - Model evaluation functions
  - Results export in LaTeX format
  - Performance analysis utilities

- **Metrics** (`metrics.py`):
  - Implementation of evaluation metrics
  - Performance measurement utilities
  - Statistical analysis functions
  - Results validation tools

### Vision Transformer Embeddings Module (`vit_embeddings/`)

- **Embeddings Extraction** (`extract_embeddings.py`):
  - Main pipeline for feature extraction from images
  - Support for multiple Vision Transformer models
  - Batch processing and data management
  - Embeddings storage in parquet format

- **ViT Utilities** (`utils_vits.py`):
  - Common functions for Vision Transformer models
  - Model loading and initialization
  - Image preprocessing pipelines
  - Feature extraction helpers
  - Support for CLIP, DINOv2, Franca, and SigLIPv2 models

## Usage

The modules in this directory are primarily used by the notebooks in the `notebooks/` directory. The typical workflow involves:

1. Using the data modules to load and preprocess datasets
2. Extracting embeddings using Vision Transformer models
3. Training ML models using the ml_classifiers module
4. Evaluating and analyzing results

## Dependencies

The code relies on several key libraries:
- numpy and pandas for data handling
- scikit-learn for ML algorithms
- PyTorch for deep learning (if applicable)
- matplotlib and seaborn for visualization

Make sure to install all required dependencies before using these modules.