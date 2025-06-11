import numpy as np
import torch
import torch.nn.functional as F

def accuracy(outputs, targets, ignore_index=-1):
    """
    Computes pixel accuracy
    Args:
        outputs: Model output after softmax, [B, C, H, W]
        targets: Ground truth labels, [B, H, W]
        ignore_index: Index to ignore from evaluation
    Returns:
        Pixel accuracy
    """
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        if ignore_index >= 0:
            valid_mask = (targets != ignore_index)
            correct = (predicted == targets) & valid_mask
            total = valid_mask.sum().item()
        else:
            correct = (predicted == targets)
            total = targets.numel()
        
        correct_pixels = correct.sum().item()
        return correct_pixels / (total + 1e-8)  # Avoid division by zero

def iou(outputs, targets, num_classes=2, ignore_index=-1):
    """
    Computes Intersection over Union (IoU), also known as Jaccard Index
    Args:
        outputs: Model output after softmax, [B, C, H, W]
        targets: Ground truth labels, [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore from evaluation
    Returns:
        List of IoU for each class, Mean IoU
    """
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        class_iou = []
        
        for cls in range(num_classes):
            if ignore_index >= 0:
                valid_mask = (targets != ignore_index)
                pred_cls = (predicted == cls) & valid_mask
                target_cls = (targets == cls) & valid_mask
            else:
                pred_cls = (predicted == cls)
                target_cls = (targets == cls)
            
            intersection = (pred_cls & target_cls).sum().float().item()
            union = (pred_cls | target_cls).sum().float().item()
            
            iou_value = intersection / (union + 1e-8)  # Avoid division by zero
            class_iou.append(iou_value)
        
        mean_iou = sum(class_iou) / len(class_iou)
        return class_iou, mean_iou

def f1_score(outputs, targets, num_classes=2, ignore_index=-1):
    """
    Computes F1 score (harmonic mean of precision and recall)
    Args:
        outputs: Model output after softmax, [B, C, H, W]
        targets: Ground truth labels, [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore from evaluation
    Returns:
        List of F1 scores for each class, Mean F1
    """
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        class_f1 = []
        
        for cls in range(num_classes):
            if ignore_index >= 0:
                valid_mask = (targets != ignore_index)
                pred_cls = (predicted == cls) & valid_mask
                target_cls = (targets == cls) & valid_mask
            else:
                pred_cls = (predicted == cls)
                target_cls = (targets == cls)
            
            # True positives, false positives, false negatives
            tp = (pred_cls & target_cls).sum().float().item()
            fp = (pred_cls & ~target_cls).sum().float().item()
            fn = (~pred_cls & target_cls).sum().float().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            class_f1.append(f1)
        
        mean_f1 = sum(class_f1) / len(class_f1)
        return class_f1, mean_f1

def evaluate_model(model, data_loader, device, metrics=None):
    """
    Evaluate model performance on dataset
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        device: Device to use for evaluation
        metrics: List of metrics to compute, from ['accuracy', 'iou', 'f1']
    Returns:
        Dictionary of results
    """
    if metrics is None:
        metrics = ['accuracy', 'iou', 'f1']
        
    model.eval()
    results = {
        'accuracy': 0.0,
        'iou': [0.0, 0.0],  # Assuming 2 classes (no-water, water)
        'mean_iou': 0.0,
        'f1': [0.0, 0.0],  # Assuming 2 classes (no-water, water)
        'mean_f1': 0.0
    }
    
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image'].to(device)
            targets = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Compute metrics
            if 'accuracy' in metrics:
                batch_accuracy = accuracy(probs, targets)
                results['accuracy'] += batch_accuracy * batch_size
            
            if 'iou' in metrics:
                batch_iou, batch_mean_iou = iou(probs, targets)
                results['iou'][0] += batch_iou[0] * batch_size
                results['iou'][1] += batch_iou[1] * batch_size
                results['mean_iou'] += batch_mean_iou * batch_size
            
            if 'f1' in metrics:
                batch_f1, batch_mean_f1 = f1_score(probs, targets)
                results['f1'][0] += batch_f1[0] * batch_size
                results['f1'][1] += batch_f1[1] * batch_size
                results['mean_f1'] += batch_mean_f1 * batch_size
    
    # Normalize by total samples
    for key in results:
        if key in ['iou', 'f1']:
            results[key][0] /= total_samples
            results[key][1] /= total_samples
        else:
            results[key] /= total_samples
    
    return results