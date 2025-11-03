"""
ViT Embeddings package for extracting embeddings from images using various Vision Transformer models.
Provides functionality for loading models, processing images and extracting embeddings in batches.
"""

from .extract_embeddings import (
    extract_embeddings_batch,
    save_embeddings,
    process_dataset_subset,
    process_model
)

from .utils_vits import (
    get_device,
    load_vit_model,
    get_embedding_output,
    check_model_requirements,
    get_model_configs,
    get_model_transforms
)

__all__ = [
    # From extract_embeddings
    'extract_embeddings_batch',
    'save_embeddings',
    'process_dataset_subset',
    'process_model',
    
    # From vit_utils
    'get_device',
    'load_vit_model',
    'get_embedding_output',
    'check_model_requirements',
    'get_model_configs',
    'get_model_transforms'
]