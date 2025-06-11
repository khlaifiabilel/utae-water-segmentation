import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class EncoderBlock(nn.Module):
    """Encoder block with downsampling"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.use_attention = use_attention
        
        if use_attention:
            self.attention = SpatialAttention(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_attention:
            x = self.attention(x)
        skip = x
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    """Decoder block with upsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Ensure skip connection has the same dimensions as upsampled feature
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TemporalAttention(nn.Module):
    """Temporal attention module for focusing on relevant time steps"""
    def __init__(self, channels, n_head=8, d_k=None):
        super().__init__()
        if d_k is None:
            d_k = channels // n_head
        
        self.n_head = n_head
        self.d_k = d_k
        
        # Query, key, value projections
        self.q_proj = nn.Linear(channels, n_head * d_k)
        self.k_proj = nn.Linear(channels, n_head * d_k)
        self.v_proj = nn.Linear(channels, n_head * d_k)
        
        # Output projection
        self.out_proj = nn.Linear(n_head * d_k, channels)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C, H, W]
               B = batch size, T = time steps, C = channels, H = height, W = width
        Returns:
            Tensor with same shape after applying temporal attention
        """
        B, T, C, H, W = x.shape
        
        # Reshape for attention computation
        x_flat = rearrange(x, 'b t c h w -> (b h w) t c')
        
        # Project queries, keys, values
        q = self.q_proj(x_flat).view(-1, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x_flat).view(-1, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x_flat).view(-1, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(-1, T, self.n_head * self.d_k)
        out = self.out_proj(out)
        
        # Reshape back to original format
        out = rearrange(out, '(b h w) t c -> b t c h w', b=B, h=H, w=W)
        
        return out

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant regions"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels//8, 1, kernel_size=1)
        
    def forward(self, x):
        # Generate attention map
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)
        
        # Apply attention
        return x * attn

class UTAE_WaterSegmentation(nn.Module):
    """U-shaped Temporal Attention-based network adapted for water segmentation"""
    def __init__(
        self, 
        input_dim, 
        n_classes=2, 
        encoder_widths=[64, 128, 256, 512],
        decoder_widths=[512, 256, 128, 64],
        out_conv=[32, 32],
        temporal_attention=True,
        spatial_attention=True,
        attention_head_dims=64,
        n_head=8
    ):
        super().__init__()
        
        self.temporal_attention = temporal_attention
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.out_conv_widths = out_conv
        
        # Input convolution to handle S1 & S2 bands
        self.inc = nn.Sequential(
            nn.Conv2d(input_dim, encoder_widths[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_widths[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks
        self.enc_blocks = nn.ModuleList()
        for i in range(len(encoder_widths)-1):
            self.enc_blocks.append(
                EncoderBlock(
                    encoder_widths[i], 
                    encoder_widths[i+1],
                    use_attention=spatial_attention
                )
            )
        
        # Temporal attention module
        if temporal_attention:
            self.temporal_att = TemporalAttention(
                channels=encoder_widths[-1],
                n_head=n_head,
                d_k=attention_head_dims
            )
        
        # Decoder blocks
        self.dec_blocks = nn.ModuleList()
        for i in range(len(decoder_widths)-1):
            self.dec_blocks.append(
                DecoderBlock(
                    decoder_widths[i] + encoder_widths[-i-2], 
                    decoder_widths[i+1]
                )
            )
        
        # Output convolutions
        self.out_convs = nn.ModuleList()
        prev_width = decoder_widths[-1]
        for width in self.out_conv_widths:
            self.out_convs.append(ConvBlock(prev_width, width))
            prev_width = width
        
        # Final classification layer
        self.final_conv = nn.Conv2d(prev_width, n_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape [B, T, C, H, W]
               B = batch size, T = time steps, C = input_dim
        Returns:
            Segmentation mask of shape [B, n_classes, H, W]
        """
        B, T, C, H, W = x.shape
        
        # Process each time step independently through encoder
        enc_features = []
        for t in range(T):
            x_t = x[:, t]  # [B, C, H, W]
            
            # Initial convolution
            x_t = self.inc(x_t)
            
            # Encoder path
            skip_connections = []
            for enc_block in self.enc_blocks:
                x_t, skip = enc_block(x_t)
                skip_connections.append(skip)
            
            enc_features.append(x_t)
        
        # Stack features from all time steps
        x = torch.stack(enc_features, dim=1)  # [B, T, C, H, W]
        
        # Apply temporal attention if enabled
        if self.temporal_attention:
            x = self.temporal_att(x)
        
        # Take features from the last time step for decoding 
        # (or implement more sophisticated temporal fusion here)
        x = x[:, -1]  # [B, C, H, W]
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder path
        for i, dec_block in enumerate(self.dec_blocks):
            x = dec_block(x, skip_connections[i])
        
        # Output convolutions
        for conv in self.out_convs:
            x = conv(x)
        
        # Final classification
        x = self.final_conv(x)
        
        return x

# Define model with IBM dataset specifics
def create_water_segmentation_model(temporal_length=1):
    """
    Create a UTAE model configured for IBM Granite Geospatial UKI Flood Detection dataset
    
    Args:
        temporal_length: Number of time steps in input data (default: 1)
        
    Returns:
        UTAE_WaterSegmentation model
    """
    # IBM Granite dataset specifics:
    # - Sentinel-1: 2 bands (VV, VH)
    # - Sentinel-2: 13 bands (including RGB, NIR, etc.)
    input_dim = 15  # 2 (S1) + 13 (S2)
    
    model = UTAE_WaterSegmentation(
        input_dim=input_dim,
        n_classes=2,  # Binary: water/no water
        encoder_widths=[64, 128, 256, 512],
        decoder_widths=[512, 256, 128, 64],
        out_conv=[32, 32],
        temporal_attention=(temporal_length > 1),  # Only use temporal attention with multiple time steps
        spatial_attention=True,
        n_head=8
    )
    
    return model