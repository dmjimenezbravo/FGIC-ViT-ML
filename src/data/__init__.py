"""
Fine-Grained Image Classification Dataset Processing Package

This package provides utilities for processing and preparing various fine-grained
image classification datasets, including:

- CUB-200-2011 (Caltech-UCSD Birds)
- FGVC Aircraft
- Stanford Dogs
- Stanford Cars
- Oxford 102 Flowers
- NABirds
"""

from .cub200_dataset import process_dataset as process_cub200_dataset
from .fgvc_aircraft_dataset import process_dataset as process_aircraft_dataset
from .stanford_dogs_dataset import process_dataset as process_dogs_dataset
from .stanford_cars_dataset import process_dataset as process_cars_dataset
from .oxford_flowers_dataset import process_dataset as process_flowers_dataset
from .nabirds_dataset import process_dataset as process_nabirds_dataset
from .utils_visualization import (
    count_elements_in_subdirs,
    plot_directory_sizes,
    count_images,
    plot_image_counts
)

__all__ = [
    # Dataset processors
    'process_cub200_dataset',
    'process_aircraft_dataset',
    'process_dogs_dataset',
    'process_cars_dataset',
    'process_flowers_dataset',
    'process_nabirds_dataset',
    
    # Visualization utilities
    'count_elements_in_subdirs',
    'plot_directory_sizes',
    'count_images',
    'plot_image_counts'
]
