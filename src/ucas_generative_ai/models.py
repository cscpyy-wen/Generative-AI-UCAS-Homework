"""The standard, gated, and residual-dropout binary PixelCNN models."""
import torch
from torch import nn
from torch.nn import functional as F

class MaskedConv2d(nn.Conv2d):
    """Raster-order masked convolution for single-channel image generation."""

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in ('A', 'B'):
            raise ValueError('mask_type must be A or B')
        (kh, kw) = self.kernel_size
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError('The spatial kernel sizes must be odd')
        if self.stride != (1, 1) or self.padding_mode != 'zeros':
            raise ValueError('Use stride=1 and zero padding to preserve raster causality')
        expected_padding = tuple((d * (k // 2) for (d, k) in zip(self.dilation, self.kernel_size)))
        if self.padding != expected_padding:
            raise ValueError('Use centered same padding')
        self.mask_type = mask_type
        mask = torch.ones_like(self.weight)
        mask[:, :, kh // 2 + 1:, :] = 0
        mask[:, :, kh // 2, kw // 2 + (mask_type == 'B'):] = 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ChannelLayerNorm(nn.Module):
    """Normalize channels at each (batch, row, column), never across pixels."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class CausalResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.conv = MaskedConv2d('B', channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv(F.relu(self.norm(x)))

class PixelCNN(nn.Module):

    def __init__(self, channels=64, n_blocks=12):
        super().__init__()
        self.input_conv = MaskedConv2d('A', 1, channels, 7, padding=3)
        self.blocks = nn.Sequential(*[CausalResidualBlock(channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(ChannelLayerNorm(channels), nn.ReLU(), MaskedConv2d('B', channels, channels, 1), nn.ReLU(), MaskedConv2d('B', channels, 1, 1))

    def logits(self, x):
        """Unnormalized conditional logits for numerically stable training."""
        return self.head(self.blocks(self.input_conv(x)))

    def forward(self, x):
        """Conditional Bernoulli probabilities, as used by the course template."""
        return torch.sigmoid(self.logits(x))

class VerticalConv2d(MaskedConv2d):
    """A excludes the current row; B includes the entire hidden-feature row."""

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(mask_type, *args, **kwargs)
        center = self.kernel_size[0] // 2
        self.mask.zero_()
        self.mask[:, :, :center + (mask_type == 'B'), :] = 1

def gated_activation(value):
    (signal, gate) = value.chunk(2, dim=1)
    return torch.tanh(signal) * torch.sigmoid(gate)

class GatedCausalBlock(nn.Module):

    def __init__(self, channels, dilation):
        super().__init__()
        self.vertical_norm = ChannelLayerNorm(channels)
        self.horizontal_norm = ChannelLayerNorm(channels)
        self.vertical_conv = VerticalConv2d('B', channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.horizontal_conv = MaskedConv2d('B', channels, 2 * channels, (1, 3), padding=(0, dilation), dilation=(1, dilation))
        self.vertical_to_horizontal = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.horizontal_projection = nn.Conv2d(channels, channels, 1)

    def forward(self, vertical, horizontal):
        vertical_pre = self.vertical_conv(self.vertical_norm(vertical))
        horizontal_pre = self.horizontal_conv(self.horizontal_norm(horizontal))
        horizontal_pre = horizontal_pre + self.vertical_to_horizontal(vertical_pre)
        next_vertical = gated_activation(vertical_pre)
        next_horizontal = horizontal + self.horizontal_projection(gated_activation(horizontal_pre))
        return (next_vertical, next_horizontal)

class GatedPixelCNN(nn.Module):

    def __init__(self, channels=64, n_blocks=12, dilations=None):
        super().__init__()
        self.dilations = tuple(dilations or [1, 2, 4] * 4)
        if len(self.dilations) != n_blocks or any((d < 1 for d in self.dilations)):
            raise ValueError('Provide one positive dilation for each block')
        self.vertical_input = VerticalConv2d('A', 1, 2 * channels, 7, padding=3)
        self.horizontal_input = MaskedConv2d('A', 1, 2 * channels, (1, 7), padding=(0, 3))
        self.input_vertical_to_horizontal = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.blocks = nn.ModuleList([GatedCausalBlock(channels, d) for d in self.dilations])
        self.head = nn.Sequential(ChannelLayerNorm(channels), nn.ReLU(), MaskedConv2d('B', channels, channels, 1), nn.ReLU(), MaskedConv2d('B', channels, 1, 1))

    def logits(self, x):
        vertical_pre = self.vertical_input(x)
        horizontal_pre = self.horizontal_input(x)
        horizontal_pre = horizontal_pre + self.input_vertical_to_horizontal(vertical_pre)
        vertical = gated_activation(vertical_pre)
        horizontal = gated_activation(horizontal_pre)
        for block in self.blocks:
            (vertical, horizontal) = block(vertical, horizontal)
        return self.head(horizontal)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

class RegularizedGatedPixelCNN(GatedPixelCNN):

    def __init__(self, channels=64, n_blocks=12, dilations=None, dropout_p=0.2):
        super().__init__(channels, n_blocks, dilations)
        if not 0 <= dropout_p < 1:
            raise ValueError('dropout_p must be in [0, 1)')
        self.dropout_p = float(dropout_p)
        for block in self.blocks:
            block.horizontal_dropout = nn.Dropout(self.dropout_p)

    @staticmethod
    def _gate(value):
        (signal, gate) = value.chunk(2, dim=1)
        return torch.tanh(signal) * torch.sigmoid(gate)

    def logits(self, x):
        vertical_pre = self.vertical_input(x)
        horizontal_pre = self.horizontal_input(x)
        horizontal_pre = horizontal_pre + self.input_vertical_to_horizontal(vertical_pre)
        vertical = self._gate(vertical_pre)
        horizontal = self._gate(horizontal_pre)
        for block in self.blocks:
            vertical_pre = block.vertical_conv(block.vertical_norm(vertical))
            horizontal_pre = block.horizontal_conv(block.horizontal_norm(horizontal))
            horizontal_pre = horizontal_pre + block.vertical_to_horizontal(vertical_pre)
            vertical = self._gate(vertical_pre)
            residual = block.horizontal_dropout(self._gate(horizontal_pre))
            horizontal = horizontal + block.horizontal_projection(residual)
        return self.head(horizontal)
