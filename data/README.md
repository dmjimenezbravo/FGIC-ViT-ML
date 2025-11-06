# Data Directory

This directory contains the structure for organizing datasets and their embeddings in the FGIC-ViT-ML project. Due to storage constraints, neither the raw datasets nor their processed versions are included in this repository. Users must download and process the datasets following the instructions provided in each subdirectory.

## Directory Organization

The data is organized into three main subdirectories:

1. `raw/`: Contains the original downloaded datasets
   - Not included in repository
   - See `raw/README.md` for download and setup instructions

2. `processed/`: Contains the preprocessed datasets
   - Not included in repository
   - Created by running processing scripts
   - See `processed/README.md` for processing instructions

3. `embeddings/`: Contains the extracted embeddings
   - Generated using Vision Transformer models
   - See `embeddings/README.md` for extraction instructions

## Getting Started

1. First, follow the instructions in `raw/README.md` to download and set up the datasets
2. Then, process the datasets following `processed/README.md`
3. Finally, generate embeddings as explained in `embeddings/README.md`

## Directory Structure

```
data/
├── embeddings/              # Generated embeddings from Vision Transformers
│   ├── CLIP/               # Embeddings from CLIP model
│   │   ├── CUB-200-2011/
│   │   ├── FGVC-Aircraft/
│   │   ├── NABirds/
│   │   ├── Oxford 102 Flowers/
│   │   ├── Stanford Cars/
│   │   └── Stanford Dogs/
│   ├── DINOv2/            # Embeddings from DINOv2 model
│   │   └── ...
│   ├── Franca/            # Embeddings from Franca model
│   │   └── ...
│   └── SigLIPv2/          # Embeddings from SigLIPv2 model
│       └── ...
└── processed/              # Processed datasets (not included in repository)
│   └── README.md
└── raw/              # Original datasets (not included in repository)
    └── README.md
```

## Datasets

1. **CUB-200-2011**
   - 200 bird species
   - 11,788 images

2. **FGVC-Aircraft**
   - 100 aircraft variants
   - 10,000 images

3. **NABirds**
   - 555 bird species
   - 48,562 images

4. **Oxford 102 Flowers**
   - 102 flower categories
   - 8,189 images

5. **Stanford Cars**
   - 196 car models
   - 16,185 images

6. **Stanford Dogs**
   - 120 dog breeds
   - 20,580 images

## Embeddings

The `embeddings/` directory contains feature vectors extracted from different Vision Transformer models:

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training
- **DINOv2**: Facebook's self-supervised vision transformer
- **Franca**: Cross-modal vision-language model
- **SigLIPv2**: Improved vision-language model

Each embedding is stored in parquet format with the following structure:
- `train/`: Training set embeddings
- `validation/`: Validation set embeddings
- `test/`: Test set embeddings

## Storage Requirements

Approximate storage requirements for each stage:

1. **Raw Datasets**:
   - CUB-200-2011: ~1.1 GB
   - FGVC Aircraft: ~2.5 GB
   - NABirds: ~25 GB
   - Oxford 102 Flowers: ~330 MB
   - Stanford Cars: ~1.9 GB
   - Stanford Dogs: ~775 MB
   - Total: ~32 GB

2. **Processed Datasets**: ~32 GB (similar to raw)

Make sure you have sufficient storage space available before starting the data processing pipeline.

## Important Notes

1. **Data Not Included**: Due to size constraints and licensing considerations:
   - Raw datasets must be downloaded manually
   - Processed datasets must be generated locally
   - Follow the README in each subdirectory for instructions

2. **Directory Structure**: 
   - Maintain the exact directory structure as shown
   - Use the same naming conventions
   - Don't modify the directory hierarchy

3. **Processing Order**:
   - Download raw datasets first
   - Run processing scripts
   - Generate embeddings
   - Follow the instructions in each subdirectory's README

4. **Verification**:
   - Use the provided notebooks to verify data at each stage
   - Check the expected file counts and structures
   - Validate embeddings before training models