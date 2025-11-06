# Raw Datasets Directory

This directory should contain the downloaded and extracted datasets used in the FGIC-ViT-ML project. Please follow the instructions below to set up each dataset correctly.

## Dataset Downloads and Setup

### 1. CUB-200-2011
- **Download**: [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- **Files needed**: 
  - `CUB_200_2011.tgz` (Main dataset)
- **Setup**:
```bash
# Extract the dataset
tar -xzf CUB_200_2011.tgz
# Should create a directory with images and annotations
# Rename the directory to match the required format
mv CUB_200_2011 "CUB-200-2011"
```

### 2. FGVC Aircraft
- **Download**: [FGVC Aircraft](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)
- **Files needed**:
  - `fgvc-aircraft-2013b.tar.gz` (Main dataset)
- **Setup**:
```bash
# Extract the dataset
tar -xzf fgvc-aircraft-2013b.tar.gz
# Should create a fgvc-aircraft-2013b directory
# Rename the directory to match the required format
mv fgvc-aircraft-2013b "FGVC-Aircraft"
```

### 3. NABirds
- **Download**: [NABirds](https://visipedia.github.io/datasets.html)
- **Files needed**:
  - `nabirds.tar.gz` (Main dataset)
- **Setup**:
```bash
# Extract the dataset
tar -xzf nabirds.tar.gz
# Should create a nabirds directory
```

### 4. Oxford 102 Flowers
- **Download**: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Files needed**:
  - `102flowers.tgz` (Images)
  - `imagelabels.mat` (Labels)
- **Setup**:
```bash
# Extract the images
tar -xzf 102flowers.tgz
# Will create a jpg directory with all images
```

### 5. Stanford Cars
- **Download**: [Stanford Cars](https://github.com/cyizhuo/Stanford_Cars_dataset)
- **Files needed**:
  - `cars_train.tgz` (Training images)
  - `cars_test.tgz` (Test images)
  - `car_devkit.tgz` (Development kit with annotations)
- **Setup**:
```bash
# Extract all archives
tar -xzf cars_train.tgz
tar -xzf cars_test.tgz
tar -xzf car_devkit.tgz
```

### 6. Stanford Dogs
- **Download**: [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- **Files needed**:
  - `images.tar` (All images)
  - `annotation.tar` (Annotations)
- **Setup**:
```bash
# Extract both archives
tar -xf images.tar
tar -xf annotation.tar
```

## Directory Structure After Extraction

After extracting all datasets, your `raw/` directory should look like this:

```
raw/
├── CUB-200-2011/
│   ├── images/
│   └── annotations/
├── FGVC-Aircraft/
│   ├── data/
│   └── ...
├── NABirds/
│   ├── images/
│   └── ...
├── Oxford 102 Flowers/
│   ├── jpg/
│   └── imagelabels.mat
├── Stanford Cars/
│   ├── cars_train/
│   ├── cars_test/
│   └── devkit/
└── Stanford Dogs/
    ├── Images/
    └── Annotation/
```

## Important Notes

1. **Storage Space**: Make sure you have enough storage space available. The combined datasets require approximately:
   - CUB-200-2011: ~1.1 GB
   - FGVC Aircraft: ~2.5 GB
   - NABirds: ~25 GB
   - Oxford 102 Flowers: ~330 MB
   - Stanford Cars: ~1.9 GB
   - Stanford Dogs: ~775 MB

2. **Directory Names**: Keep the directory names exactly as shown above, including spaces and hyphens. The data loading scripts expect these specific names to match the embeddings directory structure.

3. **File Permissions**: Ensure that the extracted files have the correct read permissions for the scripts to access them.

4. **Processing and Verification**: 
   1. After extraction, you need to process each dataset using the corresponding script from `src/data/`:
      ```python
      # Process each dataset individually
      from src.data import (
          process_cub200_dataset,
          process_aircraft_dataset,
          process_dogs_dataset,
          process_cars_dataset,
          process_flowers_dataset,
          process_nabirds_dataset
      )

      # Run processing for each dataset
      process_cub200_dataset()
      process_aircraft_dataset()
      process_dogs_dataset()
      process_cars_dataset()
      process_flowers_dataset()
      process_nabirds_dataset()
      ```
   2. This will create a processed version of each dataset in the `data/processed/` directory with the following structure:
      ```
      processed/
      ├── CUB-200-2011/
      ├── FGVC-Aircraft/
      ├── NABirds/
      ├── Oxford 102 Flowers/
      ├── Stanford Cars/
      └── Stanford Dogs/
      ```
   3. Each processed dataset directory will contain:
      - Organized image files
      - Train/validation/test splits
      - Class mappings and metadata
   4. Finally, you can run the dataset exploration notebook from the `notebooks/` directory to verify that all datasets are correctly processed and set up.

## Troubleshooting

If you encounter any issues:
1. Verify that all archives were completely downloaded
2. Check that the extraction was successful and complete
3. Ensure directory names match exactly
4. Verify file permissions
5. Make sure all annotation files are present