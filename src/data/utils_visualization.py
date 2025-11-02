"""
Auto-generated module utils_visualization.py
Extracted from notebook: 1_datasets_exploration_preparation.ipynb
"""

import os
import tarfile
import shutil
import scipy.io
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import zipfile



# ---------- Extracted code blocks ----------

def count_elements_in_subdirs(directory, image_extensions=None):
  """
  Count image files recursively inside a directory.
  
  :param directory: str. Path to the top-level directory to scan; the function walks all subdirectories.
  :param image_extensions: tuple | list | None. Iterable of (case-insensitive) extensions to consider.
    If None, defaults to ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp').
  :return: int. Total number of image files found.
  """
  if image_extensions is None:
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
  else:
    image_extensions = tuple(ext.lower() for ext in image_extensions)
  total = 0

  for root, _, files in os.walk(directory):
    for file in files:
      if file.lower().endswith(image_extensions):
        total += 1
  
  return total

def plot_directory_sizes(dir1, dir2, dir3, labels=("Directory 1", "Directory 2", "Directory 3", "Total")):
  """
  Generate a bar chart showing the number of elements in each directory.

  :param dir1: str. Path to the first directory.
  :param dir2: str. Path to the second directory.
  :param dir3: str. Path to the third directory.
  :param labels: tuple. Tuple of labels for the bars (default ('Directory 1', 'Directory 2', 'Directory 3', 'Total')).
  :return: None. Displays a matplotlib figure.
  """
  size = [count_elements_in_subdirs(dir1), count_elements_in_subdirs(dir2), count_elements_in_subdirs(dir3)]
  total_size = sum(size)
  size.append(total_size)

  plt.figure(figsize=(8, 5))
  plt.bar(labels, size, color=['blue', 'green', 'red', 'purple'])
  plt.xlabel('Directories')
  plt.ylabel('Number of elements')
  plt.title('Number of elements in each directory')
  plt.grid(axis='y', linestyle='--', alpha=0.7)

  for i, v in enumerate(size):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)

  plt.show()

# ---- block separator ----
def count_images(directory, image_extensions=None):
    """
    Count image files per class folder under a top-level directory.

    :param directory: str. Top-level directory containing class subfolders; the function walks subdirectories recursively.
    :param image_extensions: tuple | list | None. Iterable of (case-insensitive) extensions to consider.
        If None, defaults to ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp').
    :return: dict. Mapping from class (folder) name to number of image files found.
    """
    if image_extensions is None:
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    else:
        image_extensions = tuple(ext.lower() for ext in image_extensions)

    counts = {}
    
    for dirpath, dirnames, filenames in os.walk(directory):
        image_files = [f for f in filenames if f.lower().endswith(image_extensions)]
        if image_files:
            class_name = os.path.basename(dirpath)
            counts[class_name] = len(image_files)
    
    return counts

def plot_image_counts(dataset_name, train_counts, test_counts, validation_counts):
    """
    Plot number of images per class for train/test/validation splits.

    :param dataset_name: str. Name of the dataset (used for the plot title).
    :param train_counts: dict. Mapping from class (folder) name to number of images in the training split.
    :param test_counts: dict. Mapping from class (folder) name to number of images in the test split.
    :param validation_counts: dict. Mapping from class (folder) name to number of images in the validation split.
    :return: None. Displays a matplotlib horizontal grouped bar chart showing counts per class for each split.
    """
    all_directories = sorted(set(train_counts.keys()).union(test_counts.keys()).union(validation_counts.keys()))
    if not all_directories:
        print("No image directories found in the specified paths.")
        return

    train_values = [train_counts.get(dir, 0) for dir in all_directories]
    test_values = [test_counts.get(dir, 0) for dir in all_directories]
    validation_values = [validation_counts.get(dir, 0) for dir in all_directories]

    y = np.arange(len(all_directories))
    height = 0.25

    fig, ax = plt.subplots(figsize=(12, max(5, len(all_directories) * 0.3)))

    ax.barh(y - height, train_values, height, label='Train')
    ax.barh(y, test_values, height, label='Test')
    ax.barh(y + height, validation_values, height, label='Validation')
    ax.set_ylabel('Classes')
    ax.set_xlabel('Number of images per class')
    ax.set_title(f'"{dataset_name}" classes distribution')
    ax.set_yticks(y)
    ax.set_yticklabels(all_directories)
    ax.set_ylim(y[0] - height, y[-1] + height)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, 251, 10))
    ax.set_xlim(0, 250)
    ax.legend()

    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    
    plt.show()