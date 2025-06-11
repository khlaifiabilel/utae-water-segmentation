"""
Utility functions for data handling
Author: Bilel Khlaifi
Date: 2025-06-11
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_class_weights(masks: List[np.ndarray], num_classes: int = 2) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        masks: List of mask arrays
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    # Count pixels for each class
    class_counts = np.zeros(num_classes)
    
    for mask in masks:
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                class_counts[cls] += count
    
    # Calculate weights (inverse frequency)
    total_pixels = np.sum(class_counts)
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * num_classes
    
    return torch.tensor(class_weights, dtype=torch.float32)

def create_train_val_split(data_list: List, train_ratio: float = 0.8, 
                          val_ratio: float = 0.1, seed: int = 42) -> Tuple[List, List, List]:
    """
    Split data into train, validation, and test sets
    
    Args:
        data_list: List of data samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list))
    
    train_size = int(len(data_list) * train_ratio)
    val_size = int(len(data_list) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    return train_data, val_data, test_data

def plot_data_distribution(data_dir: Path, save_dir: Optional[Path] = None):
    """
    Plot data distribution statistics
    
    Args:
        data_dir: Directory containing processed data
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = data_dir / "plots"
    save_dir.mkdir(exist_ok=True)
    
    # Load statistics
    stats_file = data_dir / "dataset_statistics.yaml"
    if not stats_file.exists():
        print(f"Statistics file not found: {stats_file}")
        return
    
    import yaml
    with open(stats_file, 'r') as f:
        stats = yaml.safe_load(f)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # S1 statistics
    if 's1' in stats:
        s1_stats = stats['s1']
        axes[0, 0].bar(s1_stats.keys(), s1_stats.values())
        axes[0, 0].set_title('Sentinel-1 Statistics')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # S2 statistics
    if 's2' in stats:
        s2_stats = stats['s2']
        axes[0, 1].bar(s2_stats.keys(), s2_stats.values())
        axes[0, 1].set_title('Sentinel-2 Statistics')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / "data_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Data distribution plots saved to: {save_dir}")

def verify_data_integrity(data_dir: Path) -> bool:
    """
    Verify the integrity of processed data
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        True if data is valid, False otherwise
    """
    required_files = ['dataset_statistics.yaml', 'data_analysis.yaml']
    required_dirs = ['train', 'validation', 'test']
    
    # Check required files
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"Missing required file: {file_path}")
            return False
    
    # Check data directories
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            print(f"Missing data directory: {dir_path}")
            continue
        
        # Check if directory has processed data
        processed_file = dir_path / f"{dir_name}_processed.pt"
        if not processed_file.exists():
            print(f"Missing processed data file: {processed_file}")
            return False
    
    print("Data integrity check passed!")
    return True