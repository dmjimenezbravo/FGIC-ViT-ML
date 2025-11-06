# Embeddings Directory

This directory contains the embeddings extracted from various Vision Transformer (ViT) models for each dataset. The embeddings are stored in parquet format for efficient storage and loading.

## Directory Structure

```
embeddings/
├── CLIP/                    # OpenAI CLIP embeddings
│   ├── CUB-200-2011/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── FGVC-Aircraft/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── NABirds/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── Oxford 102 Flowers/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── Stanford Cars/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── Stanford Dogs/
│       ├── train/
│       ├── validation/
│       └── test/
├── DINOv2/                 # Facebook's DINOv2 embeddings
│   └── ... (same structure as CLIP)
├── Franca/                 # Franca model embeddings
│   └── ... (same structure as CLIP)
└── SigLIPv2/              # SigLIP v2 embeddings
    └── ... (same structure as CLIP)
```

## Generating Embeddings

To generate the embeddings, use the functions from `src/vit_embeddings`:

```python
from src.vit_embeddings import (
    extract_embeddings_batch,
    save_embeddings,
    process_model
)

# Example: Extract embeddings for a specific model and dataset
model_name = "CLIP"  # Options: CLIP, DINOv2, Franca, SigLIPv2
dataset_name = "CUB-200-2011"

# Process the model and dataset combination
process_model(
    model_name=model_name,
    dataset_name=dataset_name,
    data_dir="data/processed",
    output_dir="data/embeddings"
)
```

Alternatively, you can use the extraction script directly:

```bash
# From project root
python -m src.vit_embeddings.extract_embeddings \
    --model CLIP \
    --dataset CUB-200-2011 \
    --data_dir data/processed \
    --output_dir data/embeddings
```

## Loading Embeddings

To load the extracted embeddings:

```python
import pandas as pd

def load_embeddings(model_name, dataset_name, split):
    """Load embeddings for a specific model, dataset, and split."""
    path = f"data/embeddings/{model_name}/{dataset_name}/{split}"
    return pd.read_parquet(path)

# Example: Load training embeddings for CLIP on CUB-200-2011
train_embeddings = load_embeddings("CLIP", "CUB-200-2011", "train")
val_embeddings = load_embeddings("CLIP", "CUB-200-2011", "validation")
test_embeddings = load_embeddings("CLIP", "CUB-200-2011", "test")
```

## Available Models

1. **CLIP**
   - OpenAI's Contrastive Language-Image Pre-training model
   - 512-dimensional embeddings
   - ViT-B/32 architecture

2. **DINOv2**
   - Facebook's self-supervised vision transformer
   - 768-dimensional embeddings
   - ViT-B/14 architecture

3. **Franca**
   - Cross-modal vision-language model
   - 512-dimensional embeddings
   - Based on ViT architecture

4. **SigLIPv2**
   - Improved vision-language model
   - 512-dimensional embeddings
   - Enhanced CLIP-like architecture

## Embedding Format

Each parquet file contains:
- Feature vectors (embeddings)
- Class labels
- Image paths
- Additional metadata

## Notes

- Embeddings are normalized by default
- All embeddings are stored in float32 format
- Each split maintains the same structure as the processed dataset
- Parquet files include data compression for efficient storage
- Feature dimensions are consistent within each model

## Troubleshooting

If you encounter issues:
1. Ensure all required models are installed
2. Verify processed datasets exist and are complete
3. Check available disk space (embeddings can be large)
4. Monitor GPU memory usage during extraction
5. Look for error logs during the extraction process