"""
PyTorch Dataset classes for IBM Granite UKI flood detection data
Author: Bilel Khlaifi
Date: 2025-06-11
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloodDetectionDataset(Dataset):
    """Dataset class for flood detection using UTAE model"""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 transform: Optional[A.Compose] = None,
                 config_path: str = "config/training_config.yaml",
                 load_to_memory: bool = False):
        """
        Initialize flood detection dataset
        
        Args:
            data_dir: Directory containing processed data
            split: Data split ('train', 'validation', 'test')
            transform: Albumentations transform pipeline
            config_path: Path to configuration file
            load_to_memory: Whether to load all data to memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load processed data
        self.samples = self._load_data()
        
        # Load to memory if specified
        if self.load_to_memory:
            logger.info(f"Loading {len(self.samples)} samples to memory...")
            self.memory_samples = []
            for i in tqdm(range(len(self.samples))):
                self.memory_samples.append(self._load_sample(i))
        
        logger.info(f"Initialized {split} dataset with {len(self.samples)} samples")
    
    def _load_data(self) -> List[Dict]:
        """Load processed data samples"""
        split_file = self.data_dir / self.split / f"{self.split}_processed.pt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {split_file}")
        
        samples = torch.load(split_file)
        logger.info(f"Loaded {len(samples)} samples from {split_file}")
        
        return samples
    
    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample and convert to tensors"""
        if self.load_to_memory:
            sample = self.memory_samples[idx]
        else:
            sample = self.samples[idx]
        
        # Convert to tensors
        processed_sample = {}
        
        # S1 data
        if 's1_data' in sample:
            s1_data = sample['s1_data']
            if not isinstance(s1_data, torch.Tensor):
                s1_data = torch.from_numpy(s1_data).float()
            processed_sample['s1_data'] = s1_data
        
        # S2 data
        if 's2_data' in sample:
            s2_data = sample['s2_data']
            if not isinstance(s2_data, torch.Tensor):
                s2_data = torch.from_numpy(s2_data).float()
            processed_sample['s2_data'] = s2_data
        
        # Mask
        if 'mask' in sample:
            mask = sample['mask']
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long()
            processed_sample['mask'] = mask
        
        # Metadata
        processed_sample['timestamp'] = torch.tensor(sample.get('timestamp', 0)).long()
        processed_sample['location'] = sample.get('location', 'unknown')
        
        return processed_sample
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self._load_sample(idx)
        
        # Apply transformations if specified
        if self.transform is not None:
            sample = self._apply_transforms(sample)
        
        return sample
    
    def _apply_transforms(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation transforms"""
        # Convert tensors to numpy for albumentations
        transform_data = {}
        
        # Combine S1 and S2 data for joint transformation
        images = []
        if 's1_data' in sample:
            s1 = sample['s1_data'].numpy()
            if len(s1.shape) == 3:  # (C, H, W)
                s1 = s1.transpose(1, 2, 0)  # (H, W, C)
            images.append(s1)
        
        if 's2_data' in sample:
            s2 = sample['s2_data'].numpy()
            if len(s2.shape) == 3:  # (C, H, W)
                s2 = s2.transpose(1, 2, 0)  # (H, W, C)
            images.append(s2)
        
        # Combine all channels
        if images:
            combined_image = np.concatenate(images, axis=2)  # (H, W, total_channels)
            transform_data['image'] = combined_image
        
        # Add mask
        if 'mask' in sample:
            mask = sample['mask'].numpy()
            if len(mask.shape) == 3:
                mask = mask.squeeze()  # Remove channel dimension if present
            transform_data['mask'] = mask
        
        # Apply transforms
        if 'image' in transform_data and 'mask' in transform_data:
            transformed = self.transform(image=transform_data['image'], mask=transform_data['mask'])
            
            # Split back into S1 and S2
            transformed_image = transformed['image']
            if len(transformed_image.shape) == 3:  # (H, W, C)
                transformed_image = transformed_image.transpose(2, 0, 1)  # (C, H, W)
            
            # Split channels back
            s1_channels = self.config['model']['s1_channels']
            s2_channels = self.config['model']['s2_channels']
            
            channel_idx = 0
            if 's1_data' in sample:
                sample['s1_data'] = torch.from_numpy(
                    transformed_image[channel_idx:channel_idx + s1_channels]
                ).float()
                channel_idx += s1_channels
            
            if 's2_data' in sample:
                sample['s2_data'] = torch.from_numpy(
                    transformed_image[channel_idx:channel_idx + s2_channels]
                ).float()
            
            # Update mask
            sample['mask'] = torch.from_numpy(transformed['mask']).long()
        
        return sample
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in the dataset"""
        class_counts = {}
        
        logger.info("Computing class distribution...")
        for i in tqdm(range(len(self))):
            sample = self._load_sample(i)
            mask = sample['mask'].numpy()
            
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                cls = int(cls)
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += int(count)
        
        return class_counts

def get_data_transforms(config: Dict, split: str) -> A.Compose:
    """
    Get data augmentation transforms based on configuration
    
    Args:
        config: Configuration dictionary
        split: Data split ('train', 'validation', 'test')
        
    Returns:
        Albumentations transform pipeline
    """
    if split == 'train':
        # Training transforms with augmentation
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=0, std=1),  # Already normalized, just ensure format
        ])
    else:
        # Validation/test transforms (no augmentation)
        transforms = A.Compose([
            A.Normalize(mean=0, std=1),
        ])
    
    return transforms

def create_data_loaders(config: Dict, 
                       data_dir: Union[str, Path],
                       batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        config: Configuration dictionary
        data_dir: Directory containing processed data
        batch_size: Batch size (if None, uses config)
        num_workers: Number of workers (if None, uses config)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    batch_size = batch_size or config['data']['batch_size']
    num_workers = num_workers or config['data']['num_workers']
    
    # Create datasets
    train_dataset = FloodDetectionDataset(
        data_dir=data_dir,
        split='train',
        transform=get_data_transforms(config, 'train'),
        load_to_memory=config['data'].get('load_to_memory', False)
    )
    
    val_dataset = FloodDetectionDataset(
        data_dir=data_dir,
        split='validation',
        transform=get_data_transforms(config, 'validation'),
        load_to_memory=config['data'].get('load_to_memory', False)
    )
    
    test_dataset = FloodDetectionDataset(
        data_dir=data_dir,
        split='test',
        transform=get_data_transforms(config, 'test'),
        load_to_memory=config['data'].get('load_to_memory', False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def test_dataset():
    """Test the dataset implementation"""
    from pathlib import Path
    
    # Test with dummy data
    config = {
        'model': {'s1_channels': 2, 's2_channels': 6},
        'data': {'batch_size': 4, 'num_workers': 2, 'load_to_memory': False}
    }
    
    data_dir = Path("data/processed")
    
    if data_dir.exists():
        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                config, data_dir, batch_size=2
            )
            
            # Test one batch
            for batch in train_loader:
                print("Batch keys:", batch.keys())
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
                break
            
            print("Dataset test passed!")
            
        except Exception as e:
            print(f"Dataset test failed: {e}")
    else:
        print(f"Data directory not found: {data_dir}")

if __name__ == "__main__":
    test_dataset()