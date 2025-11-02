# Data Processing Modules

This directory contains modules for processing and preparing various fine-grained image classification datasets. Each module provides standardized functionality to organize datasets into a consistent structure with train/test/validation splits.

## Directory Structure

```
data/
├── __init__.py           # Package initialization and exports
├── utils_visualization.py # Visualization utilities
├── cub200_dataset.py     # CUB-200-2011 dataset processor
├── fgvc_aircraft_dataset.py  # FGVC Aircraft dataset processor
├── nabirds_dataset.py    # NABirds dataset processor
├── oxford_flowers_dataset.py # Oxford 102 Flowers dataset processor
├── stanford_cars_dataset.py  # Stanford Cars dataset processor
└── stanford_dogs_dataset.py  # Stanford Dogs dataset processor
```

## Module Usage

### As Python Modules

#### Basic Import
```python
from src.data import (
    process_cub200_dataset,
    process_aircraft_dataset,
    process_dogs_dataset,
    process_cars_dataset,
    process_flowers_dataset,
    process_nabirds_dataset
)
```

#### Individual Module Usage Examples

Each module can be used either as a Python import or as a command-line script.

##### CUB-200-2011 Dataset
```python
from src.data.cub200_dataset import process_dataset

process_dataset(
    jpg_folder="path/to/images",
    output_folder="path/to/output",
    lbl_txt_path="path/to/labels.txt",
    split_txt_path="path/to/split.txt",
    class_name_path="path/to/classes.txt",
    img_name_path="path/to/images.txt",
    val_ratio=0.2
)
```

##### FGVC Aircraft Dataset
```python
from src.data.fgvc_aircraft_dataset import process_dataset

process_dataset(
    jpg_folder="path/to/images",
    output_folder="path/to/output",
    class_name_path="path/to/variants.txt",
    move=True  # Set to True to move files instead of copying
)
```

##### Stanford Dogs Dataset
```python
from src.data.stanford_dogs_dataset import process_dataset

process_dataset(
    jpg_folder="path/to/Images",
    output_folder="path/to/output",
    train_list_path="path/to/train_list.mat",
    test_list_path="path/to/test_list.mat",
    val_ratio=0.2
)
```

##### Stanford Cars Dataset
```python
from src.data.stanford_cars_dataset import process_dataset

process_dataset(
    base_path="path/to/raw/data",
    processed_path="path/to/output",
    val_ratio=0.2
)
```

##### Oxford 102 Flowers Dataset
```python
from src.data.oxford_flowers_dataset import process_dataset

process_dataset(
    jpg_folder="path/to/jpg",
    output_folder="path/to/output",
    lbl_mat_path="path/to/imagelabels.mat",
    split_mat_path="path/to/setid.mat",
    class_name_path="path/to/labels.txt",
    swap_splits=True
)
```

##### NABirds Dataset
```python
from src.data.nabirds_dataset import process_dataset

process_dataset(
    jpg_folder="path/to/images",
    output_folder="path/to/output",
    lbl_txt_path="path/to/image_class_labels.txt",
    split_txt_path="path/to/train_test_split.txt",
    class_name_path="path/to/classes.txt",
    img_name_path="path/to/images.txt",
    val_ratio=0.2,
    cleanup=True
)
```

### As Command-Line Scripts

Each module can also be run directly from the command line:

#### CUB-200-2011
```bash
python cub200_dataset.py \
    --jpg-folder "/path/to/images" \
    --output-folder "/path/to/output" \
    --lbl-txt-path "path/to/image_class_labels.txt" \
    --split-txt-path "/path/to/train_test_split.txt" \
    --class-name-path "/path/to/classes.txt" \
    --img-name-path "/path/to/images.txt" \
    --val-ratio 0.2
```

#### FGVC Aircraft
```bash
python fgvc_aircraft_dataset.py \
    --jpg-folder "/path/to/images" \
    --output-folder "/path/to/output" \
    --class-name-path "/path/to/variants.txt" \
    --move
```

#### Stanford Dogs
```bash
python stanford_dogs_dataset.py \
    --jpg-folder "/path/to/images" \
    --output-folder "/path/to/output" \
    --train-list "/path/to/train.mat" \
    --test-list "/path/to/test.mat" \
    --val-ratio 0.2
```

#### Stanford Cars
```bash
python stanford_cars_dataset.py \
    --base-path "/path/to/raw/data" \
    --processed-path "/path/to/output" \
    --val-ratio 0.2
```

#### Oxford 102 Flowers
```bash
python oxford_flowers_dataset.py \
    --jpg-folder "/path/to/images" \
    --output-folder "/path/to/output" \
    --lbl-mat-path "/path/to/label.mat" \
    --split-mat-path "/path/to/split.mat" \
    --class-name-path "/path/to/classes.txt" \
    --swap-splits
```

#### NABirds
```bash
python nabirds_dataset.py \
    --jpg-folder "/path/to/images" \
    --output-folder "/path/to/output" \
    --lbl-txt-path "path/to/image_class_labels.txt" \
    --split-txt-path "/path/to/train_test_split.txt" \
    --class-name-path "/path/to/classes.txt" \
    --img-name-path "/path/to/images.txt" \
    --val-ratio 0.2 \
    --cleanup
```

## Output Directory Structure

All dataset processors create a standardized output structure:

```
output_folder/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_2/
│       ├── image3.jpg
│       └── image4.jpg
├── test/
│   ├── class_1/
│   └── class_2/
└── validation/
    ├── class_1/
    └── class_2/
```

## Notes

- All paths can be absolute or relative
- Default values assume a specific directory structure (see individual module documentation)
- All processors handle the creation of output directories
- Validation split is created from training data (when not provided by dataset)
- Each module includes error handling and progress reporting