import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice loss for water segmentation
    Optimized for binary segmentation (water/no-water)
    """
    def __init__(self, smooth=1.0, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output before softmax, [B, C, H, W]
            targets: Ground truth labels, [B, H, W]
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        B, C, H, W = logits.shape
        targets_one_hot = torch.zeros_like(logits)
        
        # Ignore pixels with ignore_index
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1)
            targets_one_hot.scatter_(1, targets.unsqueeze(1).clamp(0).long() * valid_mask.long(), 1)
        else:
            targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        
        # Flatten
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)
        
        # Compute Dice coefficient for each class
        intersection = torch.sum(probs_flat * targets_flat, dim=2)
        cardinality = torch.sum(probs_flat + targets_flat, dim=2)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Average Dice over classes (typically just water/no-water)
        return 1 - dice_score.mean()

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in water segmentation
    Focuses more on hard-to-classify examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output before softmax, [B, C, H, W]
            targets: Ground truth labels, [B, H, W]
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # One-hot encode targets
        B, C, H, W = logits.shape
        targets_one_hot = torch.zeros_like(logits)
        
        # Create mask for valid pixels
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1)
            targets_one_hot.scatter_(1, targets.unsqueeze(1).clamp(0).long() * valid_mask.long(), 1)
            valid_mask = valid_mask.expand(-1, C, -1, -1)
        else:
            targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
            valid_mask = torch.ones_like(targets_one_hot)
        
        # Compute focal weight
        pt = (targets_one_hot * probs).sum(1)  # Get probability of ground truth class
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = targets_one_hot * self.alpha + (1 - targets_one_hot) * (1 - self.alpha)
        
        # Compute loss
        loss = -alpha_weight * focal_weight.unsqueeze(1) * targets_one_hot * log_probs
        valid_pixels = valid_mask.sum() + 1e-6  # Avoid division by zero
        
        return loss.sum() / valid_pixels

class ComboLoss(nn.Module):
    """
    Combination of Cross-Entropy and Dice loss for water segmentation
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None, ignore_index=-1):
        super(ComboLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights) if class_weights else None,
                                      ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model output before softmax, [B, C, H, W]
            targets: Ground truth labels, [B, H, W]
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def get_loss_function(loss_config):
    """
    Factory function to create loss function based on configuration
    """
    loss_type = loss_config.get('loss_type', 'cross_entropy')
    class_weights = loss_config.get('class_weights', None)
    ignore_index = loss_config.get('ignore_index', -1)
    
    if loss_type == 'cross_entropy':
        if class_weights:
            return nn.CrossEntropyLoss(weight=torch.tensor(class_weights), 
                                       ignore_index=ignore_index)
        else:
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == 'dice':
        return DiceLoss(ignore_index=ignore_index)
    elif loss_type == 'focal':
        return FocalLoss(ignore_index=ignore_index)
    elif loss_type == 'combo':
        return ComboLoss(class_weights=class_weights, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")