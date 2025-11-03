"""
Utility functions for working with Vision Transformer models.
"""
import os
import torch
import clip
from typing import Dict, Any, Optional, Tuple, Union
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

def get_device() -> torch.device:
    """
    Get the appropriate device for model execution.
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration for all supported models.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of model configurations
    """
    return {
        'FRANCA': {
            'name': 'FRANCA',
            'description': 'FRANCA ViT-B/14 model',
            'embedding_dim': 768,
            'model_id': 'franca_vitb14',
            'repo': 'valeoai/Franca',
            'image_size': 224,
            'preprocessing': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'resize_size': 256,
                'crop_size': 224,
                'interpolation': 'bicubic'
            }
        },
        'DINOv2': {
            'name': 'DINOv2',
            'description': 'DINOv2 ViT-B/14 model',
            'embedding_dim': 768,
            'model_id': 'dinov2_vitb14',
            'repo': 'facebookresearch/dinov2',
            'image_size': 224,
            'preprocessing': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'resize_size': 256,
                'crop_size': 224,
                'interpolation': 'bicubic'
            }
        },
        'CLIP': {
            'name': 'CLIP',
            'description': 'OpenAI CLIP ViT-B/32 model',
            'embedding_dim': 512,
            'model_id': 'ViT-B/32',
            'image_size': 224,
            'preprocessing': 'clip_default'  # CLIP uses its own preprocessing
        },
        'SigLIPv2': {
            'name': 'SigLIPv2',
            'description': 'Google SigLIP v2 Base model',
            'embedding_dim': 768,
            'model_id': 'google/siglip2-base-patch16-256',
            'image_size': 256,
            'preprocessing': 'siglip_processor'  # SigLIP uses AutoProcessor
        }
    }

def get_model_transforms(model_name: str) -> Union[transforms.Compose, AutoProcessor]:
    """
    Get the appropriate transforms for each model type.
    
    Args:
        model_name (str): Name of the model ('CLIP', 'FRANCA', 'DINOv2', 'SigLIPv2')
        
    Returns:
        Union[transforms.Compose, AutoProcessor]: The appropriate transformation pipeline
    """
    configs = get_model_configs()
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}")
        
    config = configs[model_name]
    preprocessing = config['preprocessing']
    
    if model_name == "CLIP":
        return clip.load(config['model_id'])[1]  # Returns the transform
    
    elif model_name == "SigLIPv2":
        return AutoProcessor.from_pretrained(config['model_id'])
    
    else:  # FRANCA and DINOv2
        interpolation = (transforms.InterpolationMode.BICUBIC 
                       if preprocessing['interpolation'] == 'bicubic'
                       else transforms.InterpolationMode.BILINEAR)
                       
        return transforms.Compose([
            transforms.Resize(preprocessing['resize_size'], interpolation),
            transforms.CenterCrop(preprocessing['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing['mean'],
                std=preprocessing['std']
            ),
        ])

def load_vit_model(model_name: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load a specific Vision Transformer model.
    
    Args:
        model_name (str): Name of the model to load
        device (torch.device, optional): Device to load the model on
        
    Returns:
        torch.nn.Module: The loaded model
    """
    if device is None:
        device = get_device()
        
    configs = get_model_configs()
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}")
        
    config = configs[model_name]
    
    if model_name == "FRANCA":
        return torch.hub.load(
            config['repo'],
            config['model_id'],
            trust_repo=True
        ).eval().to(device)
    
    elif model_name == "DINOv2":
        return torch.hub.load(
            config['repo'],
            config['model_id']
        ).eval().to(device)
    
    elif model_name == "CLIP":
        return clip.load(config['model_id'], device=device)[0].eval()
    
    elif model_name == "SigLIPv2":
        return AutoModelForZeroShotImageClassification.from_pretrained(
            config['model_id']
        ).eval().to(device)

def get_embedding_output(
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,
    model_name: str
) -> torch.Tensor:
    """
    Extract embeddings from a batch of images using the specified model.
    
    Args:
        model (torch.nn.Module): The ViT model
        batch_tensor (torch.Tensor): Batch of processed images
        model_name (str): Name of the model being used
        
    Returns:
        torch.Tensor: Extracted embeddings
    """
    with torch.no_grad():
        if model_name == 'CLIP':
            return model.encode_image(batch_tensor)
        elif model_name == 'SigLIPv2':
            return model.get_image_features(batch_tensor)
        else:
            feats = model.forward_features(batch_tensor) if hasattr(model, 'forward_features') else model(batch_tensor)
            if isinstance(feats, torch.Tensor):
                return feats
            elif isinstance(feats, dict) and 'x_norm_clstoken' in feats:
                return feats['x_norm_clstoken']
            else:
                return list(feats.values())[0]

def check_model_requirements(model_name: str) -> Tuple[bool, str]:
    """
    Check if all requirements for a specific model are met.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        Tuple[bool, str]: (requirements_met, message)
    """
    try:
        if model_name == "CLIP":
            import clip
            return True, "CLIP requirements met"
        elif model_name == "SigLIPv2":
            from transformers import AutoModelForZeroShotImageClassification
            return True, "SigLIPv2 requirements met"
        elif model_name in ["FRANCA", "DINOv2"]:
            import torch
            import torchvision
            return True, f"{model_name} requirements met"
        else:
            return False, f"Unknown model: {model_name}"
    except ImportError as e:
        return False, f"Missing requirement for {model_name}: {str(e)}"

def get_model_embedding_dim(model_name: str) -> int:
    """
    Get the embedding dimension for a specific model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        int: Embedding dimension
    """
    configs = get_model_configs()
    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}")
    return configs[model_name]['embedding_dim']