"""
rgdn_model.py
========================================================================
Sigma-aware RGDN denoiser for BL-LCILW / PnP-ADMM SISR.

Design notes:
  * BatchNorm-free; convolutions use bias=True.
  * Gradient-sigma attention uses detached, clamped gradient normalization
    to reduce local Jacobian amplification.
  * Default capacity is 64 features / 8 blocks.
  * Sigma conditioning is clipped to [0, 0.5], including inside attention.
    This behavior requires a checkpoint trained/fine-tuned with this policy.

Important: removing BatchNorm and stabilizing attention improves empirical
PnP-ADMM stability, but it does not mathematically prove non-expansiveness.
Validate the complete reconstruction and backward maps for each experiment.
This file does not provide a contractivity certificate.

Author: Abdellah Jarmouni
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientSigmaGuidedAttention(nn.Module):
    """Attention guided by image gradients and the denoising level sigma.

    The max-gradient normalizer is detached and lower-bounded: letting
    gradients flow through the denominator can amplify tiny perturbations
    and produce a locally explosive derivative.
    """

    def __init__(self, channels: int, grad_norm_floor: float = 0.05):
        super().__init__()
        self.grad_norm_floor = float(grad_norm_floor)

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        )

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        reduced = max(channels // 8, 8)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

        self.fusion = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")

        gray = x.mean(dim=1, keepdim=True)

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)

        # Detach and floor the normalizer to avoid a locally explosive
        # derivative through max-gradient normalization.
        denom = grad_mag.amax(dim=(2, 3), keepdim=True).detach()
        denom = denom.clamp_min(self.grad_norm_floor)
        grad_mag = (grad_mag / denom).clamp(0.0, 1.0)

        if sigma_map.shape[-2:] != x.shape[-2:]:
            sigma_map = F.interpolate(
                sigma_map,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        sigma_norm = sigma_map.clamp(0.0, 0.5) / 0.5

        channel_att = self.channel_attention(x)
        spatial_input = torch.cat([grad_mag, sigma_norm], dim=1)
        spatial_att = self.spatial_attention(spatial_input)

        out = x * channel_att * spatial_att
        return self.fusion(out) + x


class RGDNBlock(nn.Module):
    """Residual block with gradient-and-sigma-guided attention (No BatchNorm)."""

    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.use_attention = use_attention
        self.attention = GradientSigmaGuidedAttention(channels) if use_attention else None

        # Small learnable residual scale reduces early expansiveness.
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.use_attention:
            out = self.attention(out, sigma_map)

        return identity + self.scale * out


class RGDN(nn.Module):
    """Regularized Gradient Denoising Network with sigma-aware attention.

    Defaults are intentionally 64 features / 8 blocks for a safer PnP prior.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        use_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.num_features = int(num_features)
        self.num_blocks = int(num_blocks)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels + 1, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.body = nn.ModuleList(
            [RGDNBlock(num_features, use_attention=use_attention) for _ in range(num_blocks)]
        )

        self.middle = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True),
        )

        self.skip_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _prepare_sigma(v: torch.Tensor, sigma_map) -> torch.Tensor:
        B, _, H, W = v.shape

        if isinstance(sigma_map, (float, int)):
            sigma_tensor = torch.full(
                (B, 1, H, W),
                float(sigma_map),
                device=v.device,
                dtype=v.dtype,
            )
        elif isinstance(sigma_map, torch.Tensor):
            sigma_map = sigma_map.to(device=v.device, dtype=v.dtype)

            if sigma_map.dim() == 0:
                sigma_tensor = torch.full(
                    (B, 1, H, W),
                    float(sigma_map.item()),
                    device=v.device,
                    dtype=v.dtype,
                )
            elif sigma_map.dim() == 1:
                if sigma_map.numel() == 1:
                    sigma_tensor = sigma_map.view(1, 1, 1, 1).expand(B, 1, H, W)
                elif sigma_map.numel() == B:
                    sigma_tensor = sigma_map.view(B, 1, 1, 1).expand(B, 1, H, W)
                else:
                    raise ValueError(f"1D sigma_map must have 1 or B elements, got {sigma_map.numel()}")
            elif sigma_map.dim() == 3:
                sigma_tensor = sigma_map.unsqueeze(1)
            elif sigma_map.dim() == 4:
                sigma_tensor = sigma_map
            else:
                raise ValueError(f"Unsupported sigma_map shape: {tuple(sigma_map.shape)}")

            if sigma_tensor.shape[-2:] != (H, W):
                sigma_tensor = F.interpolate(
                    sigma_tensor,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            if sigma_tensor.shape[0] == 1 and B > 1:
                sigma_tensor = sigma_tensor.expand(B, -1, -1, -1)
        else:
            raise TypeError(f"sigma_map must be float/int/tensor, got {type(sigma_map)}")

        return sigma_tensor.clamp(0.0, 0.5)

    def forward(self, v: torch.Tensor, sigma_map) -> torch.Tensor:
        single = False
        if v.dim() == 3:
            v = v.unsqueeze(0)
            single = True
        if v.dim() != 4:
            raise ValueError(f"Expected v shape [B,C,H,W] or [C,H,W], got {tuple(v.shape)}")

        sigma_tensor = self._prepare_sigma(v, sigma_map)

        inp = torch.cat([v, sigma_tensor], dim=1)
        feat = self.head(inp)
        skip = feat

        for block in self.body:
            feat = block(feat, sigma_tensor)

        feat = self.middle(feat)
        feat = feat + self.skip_scale * skip

        predicted_noise = self.tail(feat)
        output = torch.clamp(v - predicted_noise, 0.0, 1.0)

        if single:
            output = output.squeeze(0)
        return output


# ============================================================
# Loading helpers
# ============================================================
def strip_compile_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_rgdn_arch_from_state(state_dict):
    state_dict = strip_compile_prefix(state_dict)
    num_features = 64
    if "head.0.weight" in state_dict:
        num_features = int(state_dict["head.0.weight"].shape[0])

    block_ids = set()
    pat = re.compile(r"^body\.(\d+)\.")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            block_ids.add(int(m.group(1)))
    num_blocks = (max(block_ids) + 1) if block_ids else 8
    return num_features, num_blocks


def load_rgdn(model_path: str, device: str = "cuda") -> RGDN:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        args = checkpoint.get("args", None)
        if args is not None:
            num_features = int(getattr(args, "num_features", 64))
            num_blocks = int(getattr(args, "num_blocks", 8))
        else:
            num_features, num_blocks = infer_rgdn_arch_from_state(state)
    else:
        state = checkpoint
        num_features, num_blocks = infer_rgdn_arch_from_state(state)

    state = strip_compile_prefix(state)
    model = RGDN(
        in_channels=3,
        num_features=num_features,
        num_blocks=num_blocks,
        use_attention=True,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model
