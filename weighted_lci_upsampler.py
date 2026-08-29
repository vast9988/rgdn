"""
weighted_lci_upsampler.py
========================================================================
2-D Tensor-Product Learning-Weighted Lagrange-Chebyshev Interpolation
(LCI) upsampling operator U_w, per Eq. (14) and Eq. (19) of "Bilevel
Learning-Weighted Lagrange-Chebyshev Interpolation for SISR".

Block assembly uses overlap-add with a Hann window (hop = n // 2) so
adjacent block polynomials blend smoothly instead of producing seams.
The LR signal is padded by a full block on each side (reflect, with a
replicate fallback for very short signals), block starts tile the whole
padded signal with a guaranteed final flush block, the overlap-add sum
is normalised by the accumulated window mass (positive floor), and the
margin is cropped off. Borders are therefore produced by the operator
itself; no bicubic splice is involved anywhere in the output.

Author: Abdellah Jarmouni
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def chebyshev_nodes(n: int) -> torch.Tensor:
    """
    Returns n Chebyshev nodes of the first kind in [-1, 1].
    Eq. (2) in the paper.
    """
    k = torch.arange(1, n + 1, dtype=torch.float32)
    return -torch.cos((2 * k - 1) * math.pi / (2 * n))


def all_lagrange_bases(t: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """
    Computes all Lagrange basis polynomials L_k(t) for given nodes.
    Eq. (3) in the paper.
    """
    n = len(nodes)
    L = torch.ones((len(t), n), device=t.device, dtype=t.dtype)
    for k in range(n):
        for j in range(n):
            if j != k:
                L[:, k] *= (t - nodes[j]) / (nodes[k] - nodes[j])
    return L


def _hann_window(length: int, device: torch.device) -> torch.Tensor:
    """
    Raised-cosine (Hann) window of given length.
    Values are in [0, 1], smooth at both ends, peak at centre.
    Used to blend overlapping block outputs in the HR domain.
    """
    n = torch.arange(length, dtype=torch.float32, device=device)
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / (length - 1)))


class GlobalWeightedLCI2D(nn.Module):
    """
    Implements the 2-D upsampling operator U_w = U_w^(y) ⊗ U_w^(x)
    using a block-wise Tensor Product with overlap-add smoothing to
    eliminate seam artifacts at block boundaries. Borders are produced
    by the operator itself (no bicubic splice).
    """

    def __init__(self, n_nodes: int = 8, scale_factor: int = 4,
                 w_min: float = 0.01):
        super().__init__()
        self.n_nodes = n_nodes
        self.scale   = scale_factor
        self.w_min   = w_min

        # Learnable parameters theta (initialised to 0 -> w_k = 1 for all k)
        self.theta = nn.Parameter(torch.zeros(n_nodes))

        # LR Chebyshev nodes: t_k
        self.register_buffer('nodes', chebyshev_nodes(n_nodes))

        # HR evaluation coordinates: Chebyshev nodes of order (scale * n_nodes)
        i = torch.arange(1, scale_factor * n_nodes + 1, dtype=torch.float32)
        hr_positions = -torch.cos(
            (2 * i - 1) * math.pi / (2 * scale_factor * n_nodes))
        self.register_buffer('hr_positions', hr_positions)

        # Precompute Lagrange basis matrix at HR positions: [scale*n_nodes, n_nodes]
        L = all_lagrange_bases(hr_positions, self.nodes)
        self.register_buffer('L_base', L)

        # Hann window in HR domain for overlap-add blending
        hr_block_len = scale_factor * n_nodes   # e.g. 4 * 8 = 32
        self.register_buffer(
            'hr_window',
            _hann_window(hr_block_len, torch.device('cpu')))

    def get_weights(self) -> torch.Tensor:
        """Eq. (8): softmax reparameterisation."""
        n, w_min = self.n_nodes, self.w_min
        s = F.softmax(self.theta, dim=0)
        return w_min + (n - n * w_min) * s

    def _upsample_1d(self, signal_1d: torch.Tensor,
                     w: torch.Tensor) -> torch.Tensor:
        """
        Block-wise 1-D weighted LCI upsampling with overlap-add smoothing.

        Steps:
          1. Pad the LR signal by a FULL block (n) on each side so the true
             borders end up inside the fully-overlapped region.
          2. Extract OVERLAPPING blocks (hop = n // 2) tiling the whole padded
             signal, including a final block flush with the padded end.
          3. Upsample each block via the einsum.
          4. Apply the Hann window to each upsampled block in the HR domain.
          5. Overlap-add all windowed HR blocks into the output buffer.
          6. Normalise by the accumulated window mass (with a positive floor).
          7. Crop the full-block margin back off.
        """
        B_flat, N = signal_1d.shape
        n         = self.n_nodes
        s         = self.scale
        hr_block  = s * n           # HR samples per block (e.g. 32)
        hop_lr    = max(1, n // 2)  # LR hop between blocks (e.g. 4)

        # ---- Step 1: pad LR signal by a full block each side ----
        # A full-block margin guarantees every kept output pixel (after the
        # final crop) is covered by fully-supported, overlapping blocks.
        pad_left  = n
        # Right pad: a full block, plus enough to align N to the hop grid.
        pad_right = n + ((hop_lr - (N % hop_lr)) % hop_lr)

        # reflect mode requires pad < signal length; fall back to replicate
        # for very short signals (keeps the operator well-defined everywhere).
        if pad_left < N and pad_right < N:
            mode = 'reflect'
        else:
            mode = 'replicate'
        signal_padded = F.pad(signal_1d, (pad_left, pad_right), mode=mode)
        N_padded      = signal_padded.shape[1]

        # ---- Step 2: dense overlapping block starts over the WHOLE signal ----
        last_start = N_padded - n
        start_list = list(range(0, last_start + 1, hop_lr))
        if not start_list:
            start_list = [0]
        if start_list[-1] != last_start:
            start_list.append(last_start)   # guarantee right-edge coverage
        starts = torch.tensor(start_list, device=signal_1d.device,
                              dtype=torch.long)
        num_blocks = starts.shape[0]

        idx     = starts.unsqueeze(1) + torch.arange(
                      n, device=signal_1d.device)        # [num_blocks, n]
        f_nodes = signal_padded[:, idx]                  # [B_flat, num_blocks, n]

        # ---- Step 3: upsample each block ----
        hr_blocks = torch.einsum(
            'k, bmk, tk -> bmt', w, f_nodes, self.L_base)  # [B, blocks, hr_block]

        # ---- Step 4: apply Hann window in HR domain ----
        win = self.hr_window.to(signal_1d.device)          # [hr_block]
        hr_blocks_windowed = hr_blocks * win.view(1, 1, -1)

        # ---- Step 5: overlap-add into output buffer ----
        hr_out_len = N_padded * s
        output = torch.zeros(B_flat, hr_out_len,
                             device=signal_1d.device, dtype=signal_1d.dtype)
        norm   = torch.zeros(B_flat, hr_out_len,
                             device=signal_1d.device, dtype=signal_1d.dtype)

        win_row = win.view(1, -1)
        for m in range(num_blocks):
            hr_start = int(starts[m].item()) * s
            hr_end   = hr_start + hr_block
            # By construction hr_end <= hr_out_len for every block, so no block
            # is ever dropped and the edges stay fully covered.
            output[:, hr_start:hr_end] += hr_blocks_windowed[:, m, :]
            norm[:,   hr_start:hr_end] += win_row

        # ---- Step 6: normalise (positive floor; coverage keeps norm ~constant) ----
        output = output / norm.clamp_min(1e-6)

        # ---- Step 7: crop the full-block margin back off ----
        hr_offset = pad_left * s
        output    = output[:, hr_offset: hr_offset + N * s]

        return output

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Eq (19): 2-D separable upsampling U_w = U_w^(y) ⊗ U_w^(x).
        Borders are produced by the operator itself — no bicubic splice.
        """
        w          = self.get_weights()
        B, C, H, W = y.shape

        # ---- U_w^(x): Horizontal upsampling ----
        y_flat_x  = y.reshape(B * C * H, W)
        hr_x_flat = self._upsample_1d(y_flat_x, w)
        hr_x      = hr_x_flat.reshape(B, C, H, W * self.scale)

        # ---- U_w^(y): Vertical upsampling ----
        hr_x_transposed = hr_x.transpose(2, 3).contiguous()
        y_flat_y        = hr_x_transposed.reshape(
            B * C * (W * self.scale), H)
        hr_y_flat = self._upsample_1d(y_flat_y, w)
        hr_y      = hr_y_flat.reshape(
            B, C, W * self.scale, H * self.scale)

        x_hat = hr_y.transpose(2, 3).contiguous()
        return x_hat


class PatchWeightNetwork(nn.Module):
    """
    Implements the 3-layer MLP Phi_xi from Section 2.2.7.
    Maps an LR patch directly to space-variant Chebyshev parameters.
    """
    def __init__(self, in_channels=3, n_nodes=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_nodes, kernel_size=1)
        )

        # Initialize the final layer to zero.
        # This guarantees the network starts by outputting theta=0 (uniform w_k=1),
        # making the first iteration perfectly mathematically equivalent to standard LCI.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, y_lr):
        return self.mlp(y_lr)


class PatchWeightedLCI2D(nn.Module):
    """
    Spatially adaptive 2-D upsampling operator.
    Applies the dense weight map generated by PatchWeightNetwork correctly
    across the separable horizontal and vertical tensor product passes.
    """
    def __init__(self, n_nodes: int = 8, scale_factor: int = 4, w_min: float = 0.01):
        super().__init__()
        self.n_nodes = n_nodes
        self.scale   = scale_factor
        self.w_min   = w_min

        # The global theta array is replaced by the network
        self.weight_net = PatchWeightNetwork(in_channels=3, n_nodes=n_nodes)

        self.register_buffer('nodes', chebyshev_nodes(n_nodes))
        i = torch.arange(1, scale_factor * n_nodes + 1, dtype=torch.float32)
        hr_positions = -torch.cos((2 * i - 1) * math.pi / (2 * scale_factor * n_nodes))
        self.register_buffer('hr_positions', hr_positions)
        self.register_buffer('L_base', all_lagrange_bases(hr_positions, self.nodes))

        hr_block_len = scale_factor * n_nodes
        self.register_buffer('hr_window', _hann_window(hr_block_len, torch.device('cpu')))

    def _upsample_1d_spatial(self, signal_1d: torch.Tensor, w_1d: torch.Tensor) -> torch.Tensor:
        """1-D upsampling where weights vary per-pixel. w_1d shape: [B_flat, N, n_nodes]"""
        B_flat, N = signal_1d.shape
        n = self.n_nodes; s = self.scale; hr_block = s * n; hop_lr = max(1, n // 2)

        pad_left  = n
        pad_right = n + ((hop_lr - (N % hop_lr)) % hop_lr)
        mode = 'reflect' if (pad_left < N and pad_right < N) else 'replicate'

        # Pad signal and weights identically
        signal_padded = F.pad(signal_1d, (pad_left, pad_right), mode=mode)
        w_padded = F.pad(w_1d.permute(0, 2, 1), (pad_left, pad_right), mode='replicate').permute(0, 2, 1)
        N_padded = signal_padded.shape[1]

        last_start = N_padded - n
        start_list = list(range(0, last_start + 1, hop_lr))
        if not start_list: start_list = [0]
        if start_list[-1] != last_start: start_list.append(last_start)
        starts = torch.tensor(start_list, device=signal_1d.device, dtype=torch.long)
        num_blocks = starts.shape[0]

        idx = starts.unsqueeze(1) + torch.arange(n, device=signal_1d.device)
        f_nodes = signal_padded[:, idx]

        # Extract the specific weights assigned to the center of each sliding block
        centers = starts + n // 2
        w_blocks = w_padded[:, centers, :]

        hr_blocks = torch.einsum('bmk, bmk, tk -> bmt', w_blocks, f_nodes, self.L_base)

        win = self.hr_window.to(signal_1d.device)
        hr_blocks_windowed = hr_blocks * win.view(1, 1, -1)

        hr_out_len = N_padded * s
        output = torch.zeros(B_flat, hr_out_len, device=signal_1d.device, dtype=signal_1d.dtype)
        norm   = torch.zeros(B_flat, hr_out_len, device=signal_1d.device, dtype=signal_1d.dtype)

        win_row = win.view(1, -1)
        for m in range(num_blocks):
            hr_start = int(starts[m].item()) * s
            hr_end   = hr_start + hr_block
            output[:, hr_start:hr_end] += hr_blocks_windowed[:, m, :]
            norm[:, hr_start:hr_end] += win_row

        output = output / norm.clamp_min(1e-6)
        return output[:, pad_left * s : pad_left * s + N * s]

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape

        # 1. Generate the spatial weight map
        theta_map = self.weight_net(y)
        s_map = F.softmax(theta_map, dim=1)
        w_map = self.w_min + (self.n_nodes - self.n_nodes * self.w_min) * s_map # [B, n, H, W]

        # 2. Horizontal pass (requires W-dimension weights)
        y_flat_x = y.reshape(B * C * H, W)
        w_x = w_map.permute(0, 2, 3, 1) # [B, H, W, n]
        w_1d_x = w_x.unsqueeze(1).expand(B, C, H, W, self.n_nodes).reshape(B * C * H, W, self.n_nodes)

        hr_x_flat = self._upsample_1d_spatial(y_flat_x, w_1d_x)
        hr_x = hr_x_flat.reshape(B, C, H, W * self.scale)

        # 3. Vertical pass (requires weights upsampled horizontally to match new width)
        hr_x_transposed = hr_x.transpose(2, 3).contiguous()
        y_flat_y = hr_x_transposed.reshape(B * C * (W * self.scale), H)

        w_map_up = F.interpolate(w_map, size=(H, W * self.scale), mode='bilinear', align_corners=False)
        w_y = w_map_up.permute(0, 3, 2, 1) # [B, W*s, H, n]
        w_1d_y = w_y.unsqueeze(1).expand(B, C, W * self.scale, H, self.n_nodes).reshape(B * C * (W * self.scale), H, self.n_nodes)

        hr_y_flat = self._upsample_1d_spatial(y_flat_y, w_1d_y)
        hr_y = hr_y_flat.reshape(B, C, W * self.scale, H * self.scale)

        return hr_y.transpose(2, 3).contiguous()

    def get_w_map(self, y: torch.Tensor) -> torch.Tensor:
        """Helper to extract the final generated weight map for logging."""
        with torch.no_grad():
            theta_map = self.weight_net(y)
            return self.w_min + (self.n_nodes - self.n_nodes * self.w_min) * F.softmax(theta_map, dim=1)


class AlphaNet(nn.Module):
    """Predicts the spatial regularisation map alpha from the LR image."""

    def __init__(self, in_channels=3, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # Initialize final layer near zero so it starts predicting a flat baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lr_img):
        raw = self.net(lr_img)
        # Upsample the alpha map to High-Resolution space
        hr_raw = F.interpolate(
            raw,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )

        # Sigmoid bounds the output; the affine map keeps alpha in [0.001, 0.025],
        # which is strictly positive (no zero-division in the denoiser) and
        # guarantees (alpha / rho) never exceeds 0.5, preventing dead gradients.
        # At initialization (sigmoid(0) = 0.5) it outputs 0.013, i.e. an initial
        # sigma to RGDN of 0.013 / 0.05 = 0.26 (moderate denoising).
        return torch.sigmoid(hr_raw) * 0.024 + 0.001
