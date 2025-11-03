"""
Metrics calculation and visualization utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, Any

def calculate_metrics(predictions: pd.DataFrame, actual_col: str = 'label') -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions
        actual_col (str): Name of the column containing actual labels
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    pred_col = 'prediction_label' if 'prediction_label' in predictions.columns else 'Label'
    
    y_true = predictions[actual_col]
    y_pred = predictions[pred_col]
    
    # Let PyCaret handle the metrics calculation
    metrics = {}  # You can add specific metric calculations here
    return metrics

def plot_confusion_matrix(
    predictions: pd.DataFrame,
    actual_col: str = 'label',
    figsize: tuple = (20, 20),
    top_n_classes: int = 20
) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions
        actual_col (str): Name of the column containing actual labels
        figsize (tuple): Figure size
        top_n_classes (int): Number of top classes to show
    """
    pred_col = 'prediction_label' if 'prediction_label' in predictions.columns else 'Label'
    
    y_true = predictions[actual_col]
    y_pred = predictions[pred_col]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # If there are too many classes, show only top-N most frequent
    n_clases = len(np.unique(y_true))
    if n_clases > top_n_classes:
        clases_frecuentes = y_true.value_counts().nlargest(top_n_classes).index
        mask = [cls in clases_frecuentes for cls in np.unique(y_true)]
        
        cm_plot = cm_norm[mask][:, mask]
        labels_plot = np.unique(y_true)[mask]
    else:
        cm_plot = cm_norm
        labels_plot = np.unique(y_true)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm_plot, annot=False, cmap="Blues", 
                xticklabels=labels_plot, yticklabels=labels_plot)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()