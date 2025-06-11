import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import datetime
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from models.utae_water_segmentation import create_water_segmentation_model
from dataset import IBMGraniteDataset, IBMGraniteDataset
from utils.losses import get_loss_function
from utils.metrics import evaluate_model
import config

def train_model(cfg):
    """
    Train the UTAE model for water segmentation
    Args:
        cfg: Configuration dictionary
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = Path(cfg['TRAINING_CONFIG']['save_dir'])
    log_dir = Path(cfg['TRAINING_CONFIG']['log_dir'])
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(save_dir / f"config_{run_id}.json", 'w') as f:
        json.dump(cfg, f, indent=4)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IBMGraniteDataset(
        data_dir=cfg['DATASET_CONFIG']['data_dir'],
        split='train',
        img_size=cfg['DATASET_CONFIG']['img_size'],
        temporal_length=cfg['DATASET_CONFIG']['temporal_length'],
        augmentations=True
    )
    
    val_dataset = IBMGraniteDataset(
        data_dir=cfg['DATASET_CONFIG']['data_dir'],
        split='val',
        img_size=cfg['DATASET_CONFIG']['img_size'],
        temporal_length=cfg['DATASET_CONFIG']['temporal_length'],
        augmentations=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['DATASET_CONFIG']['batch_size'],
        shuffle=True,
        num_workers=cfg['DATASET_CONFIG']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['DATASET_CONFIG']['batch_size'],
        shuffle=False,
        num_workers=cfg['DATASET_CONFIG']['num_workers'],
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = create_water_segmentation_model(
        temporal_length=cfg['DATASET_CONFIG']['temporal_length']
    )
    model = model.to(device)
    
    # Loss function
    criterion = get_loss_function(cfg['LOSS_CONFIG'])
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['TRAINING_CONFIG']['learning_rate'],
        weight_decay=cfg['TRAINING_CONFIG']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler_type = cfg['TRAINING_CONFIG'].get('lr_scheduler', 'reduce_on_plateau')
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg['TRAINING_CONFIG']['lr_scheduler_params']['factor'],
            patience=cfg['TRAINING_CONFIG']['lr_scheduler_params']['patience'],
            verbose=True
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['TRAINING_CONFIG']['num_epochs'],
            eta_min=1e-6
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['TRAINING_CONFIG']['lr_scheduler_params'].get('step_size', 30),
            gamma=cfg['TRAINING_CONFIG']['lr_scheduler_params'].get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir / run_id)
    
    # Training variables
    best_val_loss = float('inf')
    best_mean_iou = 0.0
    early_stopping_counter = 0
    
    # Training loop
    print("Starting training...")
    for epoch in range(cfg['TRAINING_CONFIG']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['TRAINING_CONFIG']['num_epochs']} [Train]")
        for batch in train_bar:
            # Get data
            inputs = batch['image'].to(device)  # [B, T, C, H, W]
            targets = batch['mask'].to(device)  # [B, H, W]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)  # [B, num_classes, H, W]
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            batch_loss = loss.item()
            train_loss += batch_loss * inputs.size(0)
            
            # Update progress bar
            train_bar.set_postfix(loss=batch_loss)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['TRAINING_CONFIG']['num_epochs']} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                # Get data
                inputs = batch['image'].to(device)
                targets = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                batch_loss = loss.item()
                val_loss += batch_loss * inputs.size(0)
                
                # Update progress bar
                val_bar.set_postfix(loss=batch_loss)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataset)
        
        # Evaluate model
        metrics = evaluate_model(model, val_loader, device)
        
        # Update learning rate
        if scheduler_type == 'reduce_on_plateau':
            scheduler.step(avg_val_loss)
        elif scheduler and scheduler_type != 'reduce_on_plateau':
            scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IoU/water', metrics['iou'][1], epoch)  # Water class IoU
        writer.add_scalar('IoU/mean', metrics['mean_iou'], epoch)
        writer.add_scalar('F1/water', metrics['f1'][1], epoch)    # Water class F1
        writer.add_scalar('F1/mean', metrics['mean_f1'], epoch)
        writer.add_scalar('Accuracy', metrics['accuracy'], epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{cfg['TRAINING_CONFIG']['num_epochs']}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Mean IoU: {metrics['mean_iou']:.4f}, "
              f"Water IoU: {metrics['iou'][1]:.4f}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'metrics': metrics,
            }, save_dir / f"best_model_loss_{run_id}.pth")
            print(f"Saved new best model (loss) at epoch {epoch+1}")
            early_stopping_counter = 0
        
        # Save best model based on mean IoU
        if metrics['mean_iou'] > best_mean_iou:
            best_mean_iou = metrics['mean_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'metrics': metrics,
            }, save_dir / f"best_model_iou_{run_id}.pth")
            print(f"Saved new best model (IoU) at epoch {epoch+1}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if cfg['TRAINING_CONFIG']['early_stopping'] and \
           early_stopping_counter >= cfg['TRAINING_CONFIG']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'metrics': metrics,
    }, save_dir / f"final_model_{run_id}.pth")
    
    writer.close()
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}, "
          f"Best mean IoU: {best_mean_iou:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train UTAE Water Segmentation model")
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to config file (optional, uses default if not provided)')
    args = parser.parse_args()
    
    # Load configuration
    cfg = {
        'DATASET_CONFIG': config.DATASET_CONFIG,
        'MODEL_CONFIG': config.MODEL_CONFIG,
        'TRAINING_CONFIG': config.TRAINING_CONFIG,
        'LOSS_CONFIG': config.LOSS_CONFIG,
        'AUGMENTATION_CONFIG': config.AUGMENTATION_CONFIG,
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            user_cfg = json.load(f)
            # Update default config with user config
            for key in user_cfg:
                if key in cfg:
                    cfg[key].update(user_cfg[key])
    
    train_model(cfg)

if __name__ == "__main__":
    main()