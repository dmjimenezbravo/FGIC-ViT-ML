# Processed Datasets Directory

This directory contains the processed versions of all datasets used in the FGIC-ViT-ML project. The data here is created by running the processing scripts from the `src/data/` directory on the raw datasets.

## Directory Structure

```
processed/
├── CUB-200-2011/
│   ├── train/             # Training split
│   ├── validation/        # Validation split
│   └── test/             # Test split
├── FGVC-Aircraft/
│   ├── train/
│   ├── validation/
│   └── test/
├── NABirds/
│   ├── train/
│   ├── validation/
│   └── test/
├── Oxford 102 Flowers/
│   ├── train/
│   ├── validation/
│   └── test/
├── Stanford Cars/
│   ├── train/
│   ├── validation/
│   └── test/
└── Stanford Dogs/
    ├── train/
    ├── validation/
    └── test/
```

## Dataset Processing

To generate this processed structure, you need to run the processing scripts for each dataset. These scripts handle:
- Image organization and validation
- Train/validation/test splitting
- Class mapping creation
- Metadata generation

### Processing Scripts

```python
# Import processing functions
from src.data import (
    process_cub200_dataset,
    process_aircraft_dataset,
    process_dogs_dataset,
    process_cars_dataset,
    process_flowers_dataset,
    process_nabirds_dataset
)

# Process each dataset
process_cub200_dataset()    # Processes CUB-200-2011
process_aircraft_dataset()   # Processes FGVC-Aircraft
process_dogs_dataset()      # Processes Stanford Dogs
process_cars_dataset()      # Processes Stanford Cars
process_flowers_dataset()   # Processes Oxford 102 Flowers
process_nabirds_dataset()   # Processes NABirds
```

## Dataset Contents

Each processed dataset directory contains:

Each dataset directory contains three split directories:

1. **Training Split** (`train/`):
   - Training data samples

2. **Validation Split** (`validation/`):
   - Validation data samples

3. **Test Split** (`test/`):
   - Test data samples

Each split contains the preprocessed data ready for model training and evaluation. The original images and metadata remain in the raw dataset directory.

## Verification

After processing, you can verify the datasets:

1. Check that all directories are created with proper structure
2. Verify the number of images matches the expected counts
3. Ensure all class mappings are complete
4. Validate split ratios in metadata
5. Run the dataset exploration notebook for visualization

## Notes

- All relative paths in metadata files use forward slashes (/) for consistency
- Image files maintain their original format (typically JPEG or PNG)
- Processing preserves original image quality

## Troubleshooting

If you encounter issues:
1. Ensure raw datasets are correctly extracted
2. Check file permissions on both raw and processed directories
3. Verify enough disk space is available
4. Look for error logs in the processing output
5. Make sure all required Python dependencies are installed