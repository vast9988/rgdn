"""
psnr_luminance.py
========================================================================
Standalone, importable MATLAB-convention Y-channel PSNR -- the SISR-literature
metric (matches the paper's Table 4 numbers).

Y uses the MATLAB rgb2ycbcr studio-range coefficients
    Y = (65.481 R + 128.553 G + 24.966 B) / 255 + 16   (R,G,B in [0,1]) -> [16,235]
PSNR is computed on that Y with peak = 255 and a `crop_border`-pixel shave.

This is the single PSNR metric used consistently across the whole pipeline
(training, evaluation, and inference scripts in this repository).

Usage
-----
    from psnr_luminance import calculate_benchmark_psnr, rgb_to_y_channel

    p = calculate_benchmark_psnr(sr, hr, crop_border=4)   # sr, hr: [B,C,H,W] in [0,1]

Accepts torch tensors ([B,C,H,W] or [C,H,W], values in [0,1]) or numpy arrays
([H,W,C] or [B,H,W,C], uint8 0-255 or float 0-1) -- inputs are normalized
internally.

Author: Abdellah Jarmouni
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                       # torch optional; numpy path still works
    _HAS_TORCH = False


# ------------------------------------------------------------
# input normalization -> torch [B,C,H,W] float in [0,1]
# ------------------------------------------------------------

def _to_bchw01(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        t = x.detach().float()
        if t.dim() == 3:                  # [C,H,W]
            t = t.unsqueeze(0)
        if t.max() > 2.0:                 # looks like 0-255
            t = t / 255.0
        return t
    arr = np.asarray(x).astype(np.float32)
    if arr.ndim == 3:                     # [H,W,C]
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):   # [B,H,W,C] -> [B,C,H,W]
        arr = arr.transpose(0, 3, 1, 2)
    if arr.max() > 2.0:
        arr = arr / 255.0
    if _HAS_TORCH:
        return torch.from_numpy(arr)
    return arr                            # numpy [B,C,H,W]


# ------------------------------------------------------------
# Y channel (MATLAB rgb2ycbcr, studio range, [0,255] scale)
# ------------------------------------------------------------

def rgb_to_y_channel(img):
    """img -> Y in [0,255] scale (range ~[16,235]). Accepts the formats above."""
    t = _to_bchw01(img)
    im = (t.clamp(0.0, 1.0) if _HAS_TORCH and isinstance(t, torch.Tensor)
          else np.clip(t, 0.0, 1.0)) * 255.0
    return (65.481 * im[:, 0:1] + 128.553 * im[:, 1:2]
            + 24.966 * im[:, 2:3]) / 255.0 + 16.0


# ------------------------------------------------------------
# PSNR
# ------------------------------------------------------------

def calculate_benchmark_psnr(img1, img2, crop_border=4):
    """
    MATLAB-Y PSNR (peak=255) with a `crop_border`-pixel shave.

    img1, img2 : [B,C,H,W]/[C,H,W] torch in [0,1], or [H,W,C]/[B,H,W,C] numpy
                 (uint8 0-255 or float 0-1). Same spatial size.
    Returns a Python float (dB); inf if the images are identical.
    """
    y1 = rgb_to_y_channel(img1)
    y2 = rgb_to_y_channel(img2)
    if crop_border > 0:
        y1 = y1[:, :, crop_border:-crop_border, crop_border:-crop_border]
        y2 = y2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if _HAS_TORCH and isinstance(y1, torch.Tensor):
        mse = float(torch.mean((y1 - y2) ** 2))
    else:
        mse = float(np.mean((np.asarray(y1) - np.asarray(y2)) ** 2))
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


# convenient alias
psnr = calculate_benchmark_psnr


if __name__ == '__main__':
    # quick self-test
    if _HAS_TORCH:
        a = torch.rand(1, 3, 64, 64)
        b = (a + 0.04 * torch.randn_like(a)).clamp(0, 1)
        print("torch  :", round(calculate_benchmark_psnr(a, b, 4), 4), "dB")
        print("identical image -> inf:", calculate_benchmark_psnr(a, a, 4))
    a_np = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    b_np = np.clip(a_np.astype(np.float32) + 8, 0, 255).astype(np.uint8)
    print("numpy  :", round(calculate_benchmark_psnr(a_np, b_np, 4), 4), "dB")
