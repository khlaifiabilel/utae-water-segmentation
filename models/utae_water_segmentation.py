"""Compact U-shaped temporal attention model for water segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // 8)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class EncoderBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, use_attention: bool = False
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvBlock(input_channels, output_channels),
            ConvBlock(output_channels, output_channels),
        ]
        if use_attention:
            layers.append(SpatialAttention(output_channels))
        self.layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.layers(inputs)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Upsample input to output width, then fuse an explicitly sized skip."""

    def __init__(
        self, input_channels: int, skip_channels: int, output_channels: int
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.layers = nn.Sequential(
            ConvBlock(output_channels + skip_channels, output_channels),
            ConvBlock(output_channels, output_channels),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(
                inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.layers(torch.cat((inputs, skip), dim=1))


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, n_head: int = 8, d_k: int | None = None) -> None:
        super().__init__()
        if n_head <= 0:
            raise ValueError("n_head must be positive")
        d_k = max(1, channels // n_head) if d_k is None else d_k
        if d_k <= 0:
            raise ValueError("attention head dimension must be positive")
        self.n_head = n_head
        self.d_k = d_k
        projected_channels = n_head * d_k
        self.q_proj = nn.Linear(channels, projected_channels)
        self.k_proj = nn.Linear(channels, projected_channels)
        self.v_proj = nn.Linear(channels, projected_channels)
        self.out_proj = nn.Linear(projected_channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, time, _, height, width = inputs.shape
        flattened = rearrange(inputs, "b t c h w -> (b h w) t c")
        shape = (-1, time, self.n_head, self.d_k)
        query = self.q_proj(flattened).view(shape).transpose(1, 2)
        key = self.k_proj(flattened).view(shape).transpose(1, 2)
        value = self.v_proj(flattened).view(shape).transpose(1, 2)
        weights = F.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / self.d_k**0.5, dim=-1
        )
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = self.out_proj(output.view(-1, time, self.n_head * self.d_k))
        return rearrange(output, "(b h w) t c -> b t c h w", b=batch, h=height, w=width)


class UTAE_WaterSegmentation(nn.Module):
    """U-shaped encoder-decoder with optional deepest-feature temporal attention."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        encoder_widths: Sequence[int] = (64, 128, 256, 512),
        out_conv: Sequence[int] = (32, 32),
        temporal_attention: bool = True,
        spatial_attention: bool = True,
        attention_head_dims: int | None = None,
        n_head: int = 8,
    ) -> None:
        super().__init__()
        widths = tuple(encoder_widths)
        if input_dim <= 0 or n_classes <= 0:
            raise ValueError("input_dim and n_classes must be positive")
        if len(widths) < 2 or any(width <= 0 for width in widths):
            raise ValueError("encoder_widths must contain at least two positive widths")
        self.input_dim = input_dim
        self.temporal_attention = temporal_attention
        self.inc = ConvBlock(input_dim, widths[0])
        self.enc_blocks = nn.ModuleList(
            EncoderBlock(widths[index], widths[index + 1], spatial_attention)
            for index in range(len(widths) - 1)
        )
        if temporal_attention:
            self.temporal_att = TemporalAttention(
                widths[-1], n_head=n_head, d_k=attention_head_dims
            )

        decoder_specs = zip(
            reversed(widths[1:]),
            reversed(widths[1:]),
            reversed(widths[:-1]),
            strict=True,
        )
        self.dec_blocks = nn.ModuleList(
            DecoderBlock(input_width, skip_width, output_width)
            for input_width, skip_width, output_width in decoder_specs
        )
        output_layers: list[nn.Module] = []
        previous_width = widths[0]
        for output_width in out_conv:
            if output_width <= 0:
                raise ValueError("out_conv widths must be positive")
            output_layers.append(ConvBlock(previous_width, output_width))
            previous_width = output_width
        self.out_convs = nn.Sequential(*output_layers)
        self.final_conv = nn.Conv2d(previous_width, n_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                f"Expected input shape [B,T,C,H,W], got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] == 0:
            raise ValueError("Input must contain at least one time step")
        if inputs.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input channels, got {inputs.shape[2]}"
            )

        encoded_steps: list[torch.Tensor] = []
        last_skips: list[torch.Tensor] = []
        for time_index in range(inputs.shape[1]):
            features = self.inc(inputs[:, time_index])
            skips: list[torch.Tensor] = []
            for encoder in self.enc_blocks:
                features, skip = encoder(features)
                skips.append(skip)
            encoded_steps.append(features)
            if time_index == inputs.shape[1] - 1:
                last_skips = skips

        features = torch.stack(encoded_steps, dim=1)
        if self.temporal_attention:
            features = self.temporal_att(features)
        decoded = features[:, -1]
        for decoder, skip in zip(self.dec_blocks, reversed(last_skips), strict=True):
            decoded = decoder(decoded, skip)
        return self.final_conv(self.out_convs(decoded))


def create_water_segmentation_model(
    input_dim: int,
    temporal_length: int = 1,
    n_classes: int = 2,
    encoder_widths: Sequence[int] = (64, 128, 256, 512),
    out_conv: Sequence[int] = (32, 32),
    temporal_attention: bool | None = None,
    spatial_attention: bool = True,
    attention_head_dims: int | None = None,
    n_head: int = 8,
) -> UTAE_WaterSegmentation:
    """Create a model for an explicitly configured input channel count."""
    if temporal_length <= 0:
        raise ValueError("temporal_length must be positive")
    use_temporal_attention = (
        temporal_length > 1 if temporal_attention is None else temporal_attention
    )
    return UTAE_WaterSegmentation(
        input_dim=input_dim,
        n_classes=n_classes,
        encoder_widths=encoder_widths,
        out_conv=out_conv,
        temporal_attention=use_temporal_attention,
        spatial_attention=spatial_attention,
        attention_head_dims=attention_head_dims,
        n_head=n_head,
    )
