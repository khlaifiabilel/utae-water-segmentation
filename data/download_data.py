"""
Download script for IBM Granite Geospatial UKI Flood Detection Dataset
Author: Bilel Khlaifi
Date: 2025-06-11
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from datasets import load_dataset
import torch
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and prepare IBM Granite UKI flood detection dataset"""
    
    def __init__(self, 
                 dataset_name: str = "ibm-granite/granite-geospatial-uki-flooddetection",
                 cache_dir: Optional[str] = None,
                 data_dir: str = "data/raw"):
        """
        Initialize dataset downloader
        
        Args:
            dataset_name: HuggingFace dataset identifier
            cache_dir: Directory to cache downloaded files
            data_dir: Directory to store processed data
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> Dict[str, Any]:
        """
        Download the dataset from HuggingFace
        
        Returns:
            Dictionary containing train, validation, and test splits
        """
        logger.info(f"Downloading dataset: {self.dataset_name}")
        
        try:
            # Download the dataset
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            logger.info(f"Dataset downloaded successfully!")
            logger.info(f"Dataset structure: {dataset}")
            
            # Print dataset information
            self._print_dataset_info(dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def _print_dataset_info(self, dataset):
        """Print detailed information about the dataset"""
        logger.info("="*50)
        logger.info("DATASET INFORMATION")
        logger.info("="*50)
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            logger.info(f"\n{split_name.upper()} SPLIT:")
            logger.info(f"  Number of samples: {len(split_data)}")
            
            if len(split_data) > 0:
                sample = split_data[0]
                logger.info(f"  Sample keys: {list(sample.keys())}")
                
                # Print shape information for each key
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        logger.info(f"    {key}: {value.shape}")
                    elif isinstance(value, (list, tuple)):
                        logger.info(f"    {key}: length {len(value)}")
                    else:
                        logger.info(f"    {key}: {type(value)}")
    
    def save_dataset_info(self, dataset, output_file: str = "dataset_info.yaml"):
        """
        Save dataset information to a YAML file
        
        Args:
            dataset: The downloaded dataset
            output_file: Output file name
        """
        info_file = self.data_dir / output_file
        
        dataset_info = {
            'dataset_name': self.dataset_name,
            'download_date': '2025-06-11',
            'splits': {}
        }
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            split_info = {
                'num_samples': len(split_data),
                'features': {}
            }
            
            if len(split_data) > 0:
                sample = split_data[0]
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        split_info['features'][key] = {
                            'type': str(type(value).__name__),
                            'shape': list(value.shape) if hasattr(value, 'shape') else None,
                            'dtype': str(value.dtype) if hasattr(value, 'dtype') else None
                        }
                    else:
                        split_info['features'][key] = {
                            'type': str(type(value).__name__),
                            'shape': None,
                            'dtype': None
                        }
            
            dataset_info['splits'][split_name] = split_info
        
        with open(info_file, 'w') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, indent=2)
        
        logger.info(f"Dataset info saved to: {info_file}")
    
    def extract_sample_data(self, dataset, num_samples: int = 5):
        """
        Extract a few samples for inspection
        
        Args:
            dataset: The downloaded dataset
            num_samples: Number of samples to extract
        """
        samples_dir = self.data_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            split_samples_dir = samples_dir / split_name
            split_samples_dir.mkdir(exist_ok=True)
            
            logger.info(f"Extracting {num_samples} samples from {split_name} split...")
            
            for i in range(min(num_samples, len(split_data))):
                sample = split_data[i]
                sample_file = split_samples_dir / f"sample_{i}.pt"
                
                # Convert to tensors if needed and save
                sample_tensors = {}
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        sample_tensors[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
                    else:
                        sample_tensors[key] = value
                
                torch.save(sample_tensors, sample_file)
                logger.info(f"  Saved sample {i} to {sample_file}")

def main():
    """Main function to download and prepare the dataset"""
    parser = argparse.ArgumentParser(description="Download IBM Granite UKI flood detection dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw", 
                       help="Directory to store the dataset")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Cache directory for HuggingFace datasets")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of sample files to extract for inspection")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(
        cache_dir=args.cache_dir,
        data_dir=args.data_dir
    )
    
    # Download dataset
    dataset = downloader.download_dataset()
    
    # Save dataset information
    downloader.save_dataset_info(dataset)
    
    # Extract sample data for inspection
    downloader.extract_sample_data(dataset, num_samples=args.num_samples)
    
    logger.info("Dataset download and preparation completed!")
    logger.info(f"Data stored in: {args.data_dir}")

if __name__ == "__main__":
    main()