# Notebooks Directory

This directory contains Jupyter notebooks for the FGIC-ViT-ML project. The notebooks are designed to be executed in sequence, as each one builds upon the results of the previous ones.

## Execution Order

1. **`1_datasets_exploration_visualization.ipynb`**
   - Purpose: Initial exploration and visualization of the fine-grained image classification datasets
   - Visualizes sample images from each dataset
   - Analyzes class distributions and dataset statistics
   - Helps understand the characteristics of each dataset

2. **`2_embeddings_extraction.ipynb`**
   - Purpose: Extract embeddings from various Vision Transformer models
   - Processes images through different ViT models (CLIP, DINOv2, Franca, SigLIPv2)
   - Saves embeddings in parquet format in the `data/embeddings` directory
   - Creates train/validation/test splits for each dataset

3. **`3_train_evaluation.ipynb`**
   - Purpose: Train and evaluate machine learning models using the extracted embeddings
   - Loads preprocessed embeddings from parquet files
   - Trains various ML models using the extracted features
   - Performs comprehensive evaluation and analysis
   - Generates visualization of results and performance metrics

## Dependencies

Each notebook uses functions from the `src` directory modules. Make sure you have installed all required dependencies and have the proper directory structure set up before running the notebooks.

## Results

- Embeddings are saved in `data/embeddings/[model_name]/[dataset_name]/`
- Training results and evaluations are saved in `results/[model_name]/[dataset_name]/`
- Visualizations and plots are generated within the notebooks

## Notes

- Notebooks should be run in the specified order to ensure all required data is available
- Each notebook includes markdown cells with detailed explanations and code documentation
- Make sure to check the output paths and configurations in each notebook before running