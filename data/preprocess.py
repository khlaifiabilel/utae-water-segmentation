"""
Preprocessing pipeline for IBM Granite UKI flood detection dataset
Author: Bilel Khlaifi
Date: 2025-06-11
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import yaml
import argparse
from tqdm import tqdm
import rasterio
from datasets import load_dataset
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FloodDataPreprocessor:
    """Preprocess IBM Granite flood detection data for UTAE model"""
    
    def __init__(self, 
                 config_path: str = "config/training_config.yaml",
                 data_dir: str = "data/processed",
                 target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize preprocessor
        
        Args:
            config_path: Path to training configuration file
            data_dir: Directory to store processed data
            target_size: Target image size (H, W)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract model configuration
        self.s1_channels = self.config['model']['s1_channels']
        self.s2_channels = self.config['model']['s2_channels']
        
        logger.info(f"Initialized preprocessor with target size: {target_size}")
        logger.info(f"S1 channels: {self.s1_channels}, S2 channels: {self.s2_channels}")
    
    def load_raw_dataset(self) -> Dict:
        """Load the raw dataset from HuggingFace"""
        dataset_name = self.config['data']['dataset_name']
        logger.info(f"Loading dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        return dataset
    
    def analyze_data_structure(self, dataset) -> Dict:
        """Analyze the structure of the dataset"""
        logger.info("Analyzing dataset structure...")
        
        analysis = {}
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            if len(split_data) == 0:
                continue
                
            sample = split_data[0]
            split_analysis = {
                'num_samples': len(split_data),
                'features': {}
            }
            
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    split_analysis['features'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype) if hasattr(value, 'dtype') else 'unknown',
                        'min': float(np.min(value)) if np.isreal(value).all() else 'non-numeric',
                        'max': float(np.max(value)) if np.isreal(value).all() else 'non-numeric',
                        'mean': float(np.mean(value)) if np.isreal(value).all() else 'non-numeric'
                    }
                else:
                    split_analysis['features'][key] = {
                        'type': str(type(value)),
                        'value_example': str(value)[:100]  # First 100 chars
                    }
            
            analysis[split_name] = split_analysis
            
            # Log key information
            logger.info(f"\n{split_name.upper()} split:")
            logger.info(f"  Samples: {split_analysis['num_samples']}")
            for feat_name, feat_info in split_analysis['features'].items():
                if 'shape' in feat_info:
                    logger.info(f"  {feat_name}: {feat_info['shape']} ({feat_info['dtype']})")
        
        # Save analysis
        analysis_file = self.data_dir / "data_analysis.yaml"
        with open(analysis_file, 'w') as f:
            yaml.dump(analysis, f, default_flow_style=False, indent=2)
        
        return analysis
    
    def identify_data_channels(self, sample: Dict) -> Dict[str, str]:
        """
        Identify which keys correspond to S1, S2, and mask data
        
        Args:
            sample: A sample from the dataset
            
        Returns:
            Dictionary mapping data types to key names
        """
        channel_mapping = {}
        
        # Common key patterns for different data types
        s1_patterns = ['sentinel1', 's1', 'sar', 'radar']
        s2_patterns = ['sentinel2', 's2', 'optical', 'msi']
        mask_patterns = ['mask', 'label', 'flood', 'water', 'target', 'gt']
        
        for key in sample.keys():
            key_lower = key.lower()
            
            # Check for S1 data
            if any(pattern in key_lower for pattern in s1_patterns):
                channel_mapping['s1_key'] = key
            
            # Check for S2 data
            elif any(pattern in key_lower for pattern in s2_patterns):
                channel_mapping['s2_key'] = key
            
            # Check for mask data
            elif any(pattern in key_lower for pattern in mask_patterns):
                channel_mapping['mask_key'] = key
        
        logger.info(f"Identified data channels: {channel_mapping}")
        return channel_mapping
    
    def normalize_sentinel1(self, s1_data: np.ndarray) -> np.ndarray:
        """
        Normalize Sentinel-1 SAR data
        
        Args:
            s1_data: Raw S1 data array
            
        Returns:
            Normalized S1 data
        """
        # Convert to dB if not already (assuming linear scale input)
        s1_data = np.clip(s1_data, a_min=1e-6, a_max=None)  # Avoid log(0)
        s1_db = 10 * np.log10(s1_data)
        
        # Normalize to [-1, 1] range (typical SAR values: -30 to 5 dB)
        s1_normalized = np.clip((s1_db + 30) / 35, -1, 1)
        
        return s1_normalized.astype(np.float32)
    
    def normalize_sentinel2(self, s2_data: np.ndarray) -> np.ndarray:
        """
        Normalize Sentinel-2 optical data
        
        Args:
            s2_data: Raw S2 data array
            
        Returns:
            Normalized S2 data
        """
        # Normalize to [0, 1] range (assuming 0-10000 scale)
        s2_normalized = np.clip(s2_data / 10000.0, 0, 1)
        
        return s2_normalized.astype(np.float32)
    
    def resize_data(self, data: np.ndarray, target_size: Tuple[int, int], 
                   mode: str = 'bilinear') -> np.ndarray:
        """
        Resize spatial data to target size
        
        Args:
            data: Input data array (C, H, W) or (T, C, H, W)
            target_size: Target (H, W)
            mode: Interpolation mode ('bilinear' or 'nearest')
            
        Returns:
            Resized data array
        """
        data_tensor = torch.from_numpy(data).float()
        
        # Handle different input dimensions
        if len(data_tensor.shape) == 3:  # (C, H, W)
            data_tensor = data_tensor.unsqueeze(0)  # (1, C, H, W)
            squeeze_needed = True
        elif len(data_tensor.shape) == 4:  # (T, C, H, W)
            batch_size = data_tensor.shape[0]
            data_tensor = data_tensor.view(-1, *data_tensor.shape[2:])  # (T*C, H, W)
            data_tensor = data_tensor.unsqueeze(1)  # (T*C, 1, H, W)
            squeeze_needed = False
        else:
            raise ValueError(f"Unsupported data shape: {data_tensor.shape}")
        
        # Resize
        resized = F.interpolate(
            data_tensor, 
            size=target_size, 
            mode=mode, 
            align_corners=False if mode == 'bilinear' else None
        )
        
        if squeeze_needed:
            resized = resized.squeeze(0)  # Back to (C, H, W)
        else:
            # Reshape back to (T, C, H, W)
            resized = resized.squeeze(1)  # (T*C, H, W)
            resized = resized.view(batch_size, -1, *target_size)  # (T, C, H, W)
        
        return resized.numpy()
    
    def process_sample(self, sample: Dict, channel_mapping: Dict) -> Dict:
        """
        Process a single sample
        
        Args:
            sample: Raw sample from dataset
            channel_mapping: Mapping of data types to keys
            
        Returns:
            Processed sample
        """
        processed = {}
        
        # Process S1 data
        if 's1_key' in channel_mapping:
            s1_data = np.array(sample[channel_mapping['s1_key']])
            s1_normalized = self.normalize_sentinel1(s1_data)
            s1_resized = self.resize_data(s1_normalized, self.target_size, mode='bilinear')
            processed['s1_data'] = s1_resized
        
        # Process S2 data
        if 's2_key' in channel_mapping:
            s2_data = np.array(sample[channel_mapping['s2_key']])
            s2_normalized = self.normalize_sentinel2(s2_data)
            s2_resized = self.resize_data(s2_normalized, self.target_size, mode='bilinear')
            processed['s2_data'] = s2_resized
        
        # Process mask data
        if 'mask_key' in channel_mapping:
            mask_data = np.array(sample[channel_mapping['mask_key']])
            mask_resized = self.resize_data(mask_data[None, ...], self.target_size, mode='nearest')[0]
            processed['mask'] = mask_resized.astype(np.int64)
        
        # Add metadata
        processed['timestamp'] = sample.get('timestamp', 0)
        processed['location'] = sample.get('location', 'unknown')
        
        return processed
    
    def create_statistics(self, dataset, channel_mapping: Dict, num_samples: int = 100):
        """
        Compute dataset statistics for normalization
        
        Args:
            dataset: Raw dataset
            channel_mapping: Channel mapping
            num_samples: Number of samples to use for statistics
        """
        logger.info(f"Computing dataset statistics from {num_samples} samples...")
        
        s1_values = []
        s2_values = []
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            n_samples = min(num_samples, len(split_data))
            
            for i in tqdm(range(n_samples), desc=f"Processing {split_name}"):
                sample = split_data[i]
                
                if 's1_key' in channel_mapping:
                    s1_data = np.array(sample[channel_mapping['s1_key']])
                    s1_values.extend(s1_data.flatten())
                
                if 's2_key' in channel_mapping:
                    s2_data = np.array(sample[channel_mapping['s2_key']])
                    s2_values.extend(s2_data.flatten())
        
        # Compute statistics
        stats = {}
        
        if s1_values:
            s1_array = np.array(s1_values)
            stats['s1'] = {
                'mean': float(np.mean(s1_array)),
                'std': float(np.std(s1_array)),
                'min': float(np.min(s1_array)),
                'max': float(np.max(s1_array)),
                'percentile_1': float(np.percentile(s1_array, 1)),
                'percentile_99': float(np.percentile(s1_array, 99))
            }
        
        if s2_values:
            s2_array = np.array(s2_values)
            stats['s2'] = {
                'mean': float(np.mean(s2_array)),
                'std': float(np.std(s2_array)),
                'min': float(np.min(s2_array)),
                'max': float(np.max(s2_array)),
                'percentile_1': float(np.percentile(s2_array, 1)),
                'percentile_99': float(np.percentile(s2_array, 99))
            }
        
        # Save statistics
        stats_file = self.data_dir / "dataset_statistics.yaml"
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, indent=2)
        
        logger.info(f"Dataset statistics saved to: {stats_file}")
        return stats
    
    def process_dataset(self, dataset, max_samples_per_split: Optional[int] = None):
        """
        Process the entire dataset
        
        Args:
            dataset: Raw dataset
            max_samples_per_split: Maximum samples to process per split (for testing)
        """
        logger.info("Starting dataset preprocessing...")
        
        # Analyze data structure
        analysis = self.analyze_data_structure(dataset)
        
        # Identify data channels from first sample
        first_sample = dataset[list(dataset.keys())[0]][0]
        channel_mapping = self.identify_data_channels(first_sample)
        
        # Compute statistics
        self.create_statistics(dataset, channel_mapping)
        
        # Process each split
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            
            if len(split_data) == 0:
                logger.warning(f"Empty split: {split_name}")
                continue
            
            # Limit samples if specified
            n_samples = len(split_data)
            if max_samples_per_split:
                n_samples = min(max_samples_per_split, n_samples)
            
            logger.info(f"Processing {split_name} split: {n_samples} samples")
            
            # Create output directory for split
            split_dir = self.data_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Process samples
            processed_samples = []
            
            for i in tqdm(range(n_samples), desc=f"Processing {split_name}"):
                try:
                    sample = split_data[i]
                    processed_sample = self.process_sample(sample, channel_mapping)
                    processed_samples.append(processed_sample)
                    
                    # Save individual sample for debugging
                    if i < 5:  # Save first 5 samples
                        sample_file = split_dir / f"sample_{i}.pt"
                        torch.save(processed_sample, sample_file)
                
                except Exception as e:
                    logger.error(f"Error processing sample {i} in {split_name}: {str(e)}")
                    continue
            
            # Save processed split
            split_file = split_dir / f"{split_name}_processed.pt"
            torch.save(processed_samples, split_file)
            logger.info(f"Saved {len(processed_samples)} processed samples to {split_file}")
        
        logger.info("Dataset preprocessing completed!")
    
    def visualize_samples(self, num_samples: int = 3):
        """Create visualization of processed samples"""
        logger.info(f"Creating visualizations for {num_samples} samples...")
        
        viz_dir = self.data_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Load processed data
        for split_name in ['train', 'validation', 'test']:
            split_dir = self.data_dir / split_name
            
            if not split_dir.exists():
                continue
            
            # Load samples
            for i in range(min(num_samples, 5)):
                sample_file = split_dir / f"sample_{i}.pt"
                
                if not sample_file.exists():
                    continue
                
                sample = torch.load(sample_file)
                
                # Create visualization
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f'{split_name} - Sample {i}')
                
                # S1 data (if available)
                if 's1_data' in sample:
                    s1 = sample['s1_data']
                    if len(s1.shape) >= 3:
                        axes[0, 0].imshow(s1[0], cmap='gray')
                        axes[0, 0].set_title('S1 - VV')
                        if s1.shape[0] > 1:
                            axes[0, 1].imshow(s1[1], cmap='gray')
                            axes[0, 1].set_title('S1 - VH')
                
                # S2 data (if available)
                if 's2_data' in sample:
                    s2 = sample['s2_data']
                    if len(s2.shape) >= 3 and s2.shape[0] >= 3:
                        # RGB composite (assuming bands 2,1,0 are R,G,B)
                        rgb = np.stack([s2[2], s2[1], s2[0]], axis=2)
                        rgb = np.clip(rgb, 0, 1)
                        axes[0, 2].imshow(rgb)
                        axes[0, 2].set_title('S2 - RGB')
                
                # Mask
                if 'mask' in sample:
                    mask = sample['mask']
                    axes[1, 0].imshow(mask, cmap='Blues')
                    axes[1, 0].set_title('Water Mask')
                
                # Remove empty subplots
                for ax in axes.flat:
                    if not ax.has_data():
                        ax.remove()
                
                plt.tight_layout()
                plt.savefig(viz_dir / f'{split_name}_sample_{i}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to: {viz_dir}")

def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description="Preprocess IBM Granite UKI flood detection dataset")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory to store processed data")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                       help="Target image size (height width)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per split (for testing)")
    parser.add_argument("--visualize", action="store_true",
                       help="Create sample visualizations")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = FloodDataPreprocessor(
        config_path=args.config,
        data_dir=args.data_dir,
        target_size=tuple(args.target_size)
    )
    
    # Load raw dataset
    dataset = preprocessor.load_raw_dataset()
    
    # Process dataset
    preprocessor.process_dataset(dataset, max_samples_per_split=args.max_samples)
    
    # Create visualizations if requested
    if args.visualize:
        preprocessor.visualize_samples()
    
    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()