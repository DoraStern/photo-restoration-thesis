"""
Method 3: pure transformer-based regression restoration -- a windowed
self-attention architecture (Swin Transformer-style, following SwinIR),
applied to damaged photo restoration the way MDTNet applies it
specifically to old photos.

Architecturally distinct from your other two methods on purpose:
  - No adversarial training (unlike Method 1's translation network)
  - No iterative generative sampling (unlike Method 2's diffusion stage)
  - Just self-attention layers trained with a plain reconstruction loss --
    a single deterministic forward pass, nothing else

This is DELIBERATELY scaled down from the full SwinIR/MDTNet configs used
in their papers (which use embed_dim=180, 6 groups of 6 blocks each --
heavy, multi-GPU-scale models). Here: embed_dim=60, 4 groups of 4 blocks,
6 attention heads. Same architectural family and mechanism, thesis-scale
compute budget. Worth stating explicitly as a scope decision, same as the
U-Net-instead-of-SwinIR choice made for DiffBIR's Stage 1.

Unlike a U-Net, this network does NOT downsample -- windowed attention
operates at the input's full resolution throughout, using shifted windows
across blocks to let information flow between windows (this is literally
Swin's whole trick: local attention within a window is cheap, and shifting
the window grid between blocks lets far-apart pixels influence each other
over several layers without ever computing full-image attention, which
would be too expensive).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Inverse of window_partition: (num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention restricted to a local window, with a
    learnable relative position bias (standard Swin design -- lets the
    model learn "how much should a pixel attend to its neighbor 3 steps
    to the left" independent of where in the image that pair occurs)."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (num_windows*B, N, C) where N = window_size * window_size
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """One transformer block: (optionally shifted) windowed attention +
    residual, then an MLP + residual. Blocks alternate between
    shift_size=0 (regular windows) and shift_size=window_size//2 (shifted
    windows) -- that alternation is what lets information cross window
    boundaries across the depth of the network."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def _compute_attn_mask(self, H: int, W: int, device):
        """Builds the mask that prevents attention across the artificial
        boundary created by cyclically shifting the window grid -- without
        this, shifted windows would let pixels 'wrap around' the image
        edges and attend to unrelated content on the opposite side."""
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size for given H, W"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._compute_attn_mask(H, W, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: a stack of SwinTransformerBlocks
    (alternating regular/shifted windows) at one resolution, followed by a
    conv layer, wrapped in a residual connection around the whole group.
    This is SwinIR's mid-level building block -- several RSTBs stacked in
    sequence form the network's "deep feature extraction" stage."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C)
        shortcut = x
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Recomputes this block's activations during backward instead
                # of storing them -- trades some extra compute time for a
                # large memory reduction, since attention matrices at
                # image_size=256 are the dominant memory cost (see the
                # OutOfMemoryError this was added to fix). use_reentrant=False
                # is the current recommended checkpoint mode.
                x = checkpoint(block, x, H, W, use_reentrant=False)
            else:
                x = block(x, H, W)
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return shortcut + x


class TransformerRestorationNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, embed_dim: int = 60,
                 depths=(4, 4, 4, 4), num_heads: int = 6, window_size: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim

        # Shallow feature extraction -- a single conv, same role as the
        # first conv in a U-Net, just without any downsampling
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        # Deep feature extraction: a sequence of RSTB groups, all operating
        # at the SAME (full) spatial resolution -- no downsample/upsample
        # anywhere in this network, unlike the U-Net used for Method 2
        self.layers = nn.ModuleList([
            RSTB(dim=embed_dim, depth=d, num_heads=num_heads, window_size=window_size,
                 use_checkpoint=use_checkpoint)
            for d in depths
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def _pad_to_window_multiple(self, x: torch.Tensor):
        """Windowed attention requires H and W to be divisible by
        window_size. Reflect-pads up to the next multiple, and returns the
        original size so the output can be cropped back down."""
        _, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, H, W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, orig_H, orig_W = self._pad_to_window_multiple(x)

        shallow_feat = self.conv_first(x)

        B, C, H, W = shallow_feat.shape
        feat = shallow_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for layer in self.layers:
            feat = layer(feat, H, W)
        feat = self.norm(feat)

        feat = feat.transpose(1, 2).view(B, C, H, W)
        feat = self.conv_after_body(feat)

        # Global residual: the network predicts a correction on top of the
        # shallow features, rather than reconstructing from nothing --
        # same principle as skip connections in the U-Net method, applied
        # once at the whole-network level instead of per-layer
        feat = feat + shallow_feat

        out = self.conv_last(feat)
        out = torch.tanh(out)  # match the dataset's [-1, 1] normalization

        return out[:, :, :orig_H, :orig_W]


def transformer_regression_loss(pred: torch.Tensor, target: torch.Tensor, l1_weight: float = 1.0):
    """Pure L1 regression loss -- no adversarial term, no diffusion
    sampling. This is the defining characteristic of this method: a single
    deterministic forward pass trained to minimize pixel-wise error."""
    l1 = torch.nn.functional.l1_loss(pred, target)
    return l1_weight * l1, l1
