# ViT Embeddings Module

This directory contains modules for extracting and managing embeddings from various Vision Transformer (ViT) models. The modules provide functionality to process images through different ViT architectures and extract their embeddings in a standardized way.

## Directory Structure

```
vit_embeddings/
├── __init__.py            # Package initialization and exports
├── extract_embeddings.py  # Main script for embedding extraction
└── utils_vits.py         # ViT models utilities and configurations
```

## Supported Models

The module currently supports the following Vision Transformer models:

- **FRANCA**: FRANCA ViT-B/14 model (768-dim embeddings)
- **DINOv2**: DINOv2 ViT-B/14 model (768-dim embeddings)
- **CLIP**: OpenAI CLIP ViT-B/32 model (512-dim embeddings)
- **SigLIPv2**: Google SigLIP v2 Base model (768-dim embeddings)

## Module Usage

### As Python Module

#### Basic Import
```python
from src.vit_embeddings import (
    extract_embeddings_batch,
    save_embeddings,
    process_dataset_subset,
    process_model,
    load_vit_model,
    get_model_transforms
)
```

#### Example Usage

##### Extract Embeddings from Images
```python
from src.vit_embeddings import extract_embeddings_batch, save_embeddings

# Prepare image information
image_info = [
    {'image_path': 'path/to/image1.jpg', 'label': 'class1'},
    {'image_path': 'path/to/image2.jpg', 'label': 'class2'},
]

# Extract embeddings
embeddings = extract_embeddings_batch(
    image_list=image_info,
    model_name='FRANCA',  # or 'DINOv2', 'CLIP', 'SigLIPv2'
    batch_size=64
)

# Save embeddings
save_embeddings(
    data=embeddings,
    output_dir='path/to/output',
    model_name='FRANCA',
    dataset_name='my_dataset',
    subset='train'
)
```

##### Process Complete Dataset
```python
from src.vit_embeddings import process_model

# Dataset information dictionary
datasets_info = {
    'dataset_name': {
        'train': train_images_info,
        'test': test_images_info,
        'validation': val_images_info
    }
}

# Process dataset with specific model
process_model(
    model_name='FRANCA',
    all_datasets_info=datasets_info,
    output_dir='path/to/output',
    batch_size=64
)
```

### As Command Line Script

The module can also be used directly from the command line:

```bash
python extract_embeddings.py \
    --datasets_dir path/to/datasets \
    --output_dir path/to/output \
    --models FRANCA DINOv2 CLIP SigLIPv2 \
    --batch_size 64
```

#### Command Line Arguments

- `--datasets_dir`: Base directory containing all datasets
- `--output_dir`: Directory where embeddings will be saved
- `--models`: List of models to use (default: all supported models)
- `--batch_size`: Batch size for processing (default: 64)
- `--datasets`: Optional list of specific datasets to process

## Output Format

Embeddings are saved in parquet format with the following structure:

```
output_dir/
└── model_name/
    └── dataset_name/
        ├── train_embeddings.parquet
        ├── test_embeddings.parquet
        └── validation_embeddings.parquet
```

Each parquet file contains:
- `image_path`: Path to the original image
- `label`: Class label of the image
- `dim_0` to `dim_N`: Embedding dimensions (N depends on the model)
