"""
infer_algo2_inspect.py
========================================================================
Inference + w/alpha inspection for BL-LCILW Algorithm 2.

Same boundary-safe pipeline as infer_algo2.py:

    HR -> reflect pad -> degrade -> ADMM/PnP -> crop HR pad -> metric/save

Additionally prints per-image statistics for the learned weight map w
and the alpha map. Inspection statistics are reported on the valid
cropped region, not the artificial reflected margin.

RGDN loading is robust: the architecture (e.g. 64/8 or 128/12) is
inferred from the checkpoint state_dict when possible, and
torch.compile '_orig_mod.' prefixes are stripped if present.

Author: Abdellah Jarmouni
"""

import os
import glob
import argparse
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from weighted_lci_upsampler import PatchWeightedLCI2D, AlphaNet
from rgdn_model import RGDN
from psnr_luminance import calculate_benchmark_psnr


# ============================================================
# Math Helpers: Folded-Frequency FFT Solver
# ============================================================

def psf2otf(psf: torch.Tensor, shape: tuple) -> torch.Tensor:
    kH, kW = psf.shape[2:]
    psf_padded = F.pad(psf, (0, shape[1] - kW, 0, shape[0] - kH))
    psf_padded = torch.roll(psf_padded, shifts=(-(kH // 2), -(kW // 2)), dims=(2, 3))
    return torch.fft.fft2(psf_padded)


def get_effective_otf(kernel_base: torch.Tensor, scale: int, img_shape: tuple) -> torch.Tensor:
    B, C, H, W = img_shape
    otf_H = psf2otf(kernel_base, (H, W))
    box_psf = torch.zeros(1, 1, H, W, device=kernel_base.device, dtype=kernel_base.dtype)
    for i in range(scale):
        for j in range(scale):
            box_psf[0, 0, -i % H, -j % W] = 1.0 / (scale ** 2)
    return otf_H * torch.fft.fft2(box_psf)


def adjoint_AT(y: torch.Tensor, kernel_base: torch.Tensor, scale: int) -> torch.Tensor:
    C = y.shape[1]
    pad = kernel_base.shape[-1] // 2
    kernel = kernel_base.repeat(C, 1, 1, 1).to(y.device)
    up = F.interpolate(y, scale_factor=scale, mode="nearest") / (scale ** 2)
    return F.conv2d(up, kernel, padding=pad, groups=C)


def fft_solve_balanced(otf, scale, mu, rho, rhs):
    B, C, H, W = rhs.shape
    lam = mu + rho
    V = torch.fft.fft2(rhs)
    F_bar = torch.conj(otf)
    h, w = H // scale, W // scale
    if H % scale != 0 or W % scale != 0:
        raise ValueError(f"HR size {(H, W)} must be divisible by scale={scale}.")
    num_fold = (otf * V).reshape(B, C, scale, h, scale, w).mean(dim=(2, 4))
    den_fold = (torch.abs(otf) ** 2).reshape(1, 1, scale, h, scale, w).mean(dim=(2, 4))
    inv_fold = num_fold / (den_fold + lam)
    inv_up = (inv_fold.unsqueeze(2).unsqueeze(4)
              .expand(B, C, scale, h, scale, w).reshape(B, C, H, W))
    X = (V - F_bar * inv_up) / lam
    return torch.fft.ifft2(X).real


# ============================================================
# Degradation and Model Loading
# ============================================================

def degrade(hr_t: torch.Tensor, kernel_base: torch.Tensor, scale: int) -> torch.Tensor:
    C = hr_t.shape[1]
    kernel = kernel_base.repeat(C, 1, 1, 1).to(hr_t.device)
    blur = F.conv2d(hr_t, kernel, padding=kernel_base.shape[-1] // 2, groups=C)
    return F.avg_pool2d(blur, scale, scale).clamp(0, 1)


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


def build_rgdn_from_checkpoint_dict(ckpt_obj, device):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state = ckpt_obj["model_state_dict"]
        args = ckpt_obj.get("args", None)
        if args is not None:
            num_features = int(getattr(args, "num_features", 64))
            num_blocks = int(getattr(args, "num_blocks", 8))
        else:
            num_features, num_blocks = infer_rgdn_arch_from_state(state)
    else:
        state = ckpt_obj
        num_features, num_blocks = infer_rgdn_arch_from_state(state)
    state = strip_compile_prefix(state)
    model = RGDN(in_channels=3, num_features=num_features, num_blocks=num_blocks, use_attention=True).to(device)
    model.load_state_dict(state)
    return model


def load_models(ckpt_path: str, scale: int, device: torch.device, rgdn_path: str | None = None):
    net_W = PatchWeightedLCI2D(n_nodes=8, scale_factor=scale).to(device)
    net_A = AlphaNet(in_channels=3, scale_factor=scale).to(device)

    hp = {"mu": 0.01, "rho": 0.05, "scale": scale, "L": 15}
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ck, dict) and "net_W" in ck:
        net_W.load_state_dict(ck["net_W"])
        net_A.load_state_dict(ck["net_Alpha"])
        net_R = build_rgdn_from_checkpoint_dict(ck["net_RGDN"], device)
        hp.update(ck.get("hparams", {}))
    else:
        net_W.load_state_dict(ck)
        alpha_path = ckpt_path.replace("_w_final", "_alpha_final")
        net_A.load_state_dict(torch.load(alpha_path, map_location=device, weights_only=False))
        if rgdn_path is None:
            raise ValueError("Separate WeightNet checkpoint requires --rgdn path for RGDN weights.")
        rg = torch.load(rgdn_path, map_location=device, weights_only=False)
        net_R = build_rgdn_from_checkpoint_dict(rg, device)

    for model in (net_W, net_A, net_R):
        model.eval()
    return net_W, net_A, net_R, hp


def make_denoiser(net_R: nn.Module, r: float):
    if r >= 1.0:
        return lambda v, s: net_R(v, s)
    return lambda v, s: v + r * (net_R(v, s) - v)


# ============================================================
# Core Inference
# ============================================================

@torch.no_grad()
def super_resolve(y, net_W, net_A, denoiser, mu, rho, scale, L, kernel_base):
    U_w_y = net_W(y)
    alpha_map = net_A(y)

    B, C, h_lr, w_lr = y.shape
    otf = get_effective_otf(kernel_base, scale, (B, C, h_lr * scale, w_lr * scale))
    ATy = adjoint_AT(y, kernel_base, scale)

    x = U_w_y
    z = x.clone()
    u = torch.zeros_like(x)
    for _ in range(L):
        rhs = ATy + mu * U_w_y + rho * (z - u)
        x = fft_solve_balanced(otf, scale, mu, rho, rhs)

        z = denoiser(x + u, alpha_map / rho)

        u = u + x - z

    return x.clamp(0, 1), U_w_y.clamp(0, 1), alpha_map


def make_gaussian_kernel(device: torch.device) -> torch.Tensor:
    coords = torch.arange(5, dtype=torch.float32, device=device) - 2
    g1d = torch.exp(-(coords ** 2) / 2.0)
    g2d = g1d.outer(g1d)
    return (g2d / g2d.sum()).view(1, 1, 5, 5).to(device)


def match_to_hr(a, hr):
    h = min(a.shape[2], hr.shape[2])
    w = min(a.shape[3], hr.shape[3])
    return a[:, :, :h, :w], hr[:, :, :h, :w]


def crop_hr_pad(x, pad):
    if pad <= 0:
        return x
    return x[:, :, pad:-pad, pad:-pad]


def crop_lr_pad(x, lr_pad):
    if lr_pad <= 0:
        return x
    return x[:, :, lr_pad:-lr_pad, lr_pad:-lr_pad]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="bl_lcilw_algo2_full.pth")
    ap.add_argument("--rgdn", default="best_model_rgdn.pth")
    ap.add_argument("--test_dir", default="./Set5")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--save_dir", default="./sr_outputs")
    ap.add_argument("--override_L", type=int, default=None, help="Override ADMM iterations from checkpoint.")
    ap.add_argument("--pad", type=int, default=8,
                    help="HR reflect padding before degradation; cropped before PSNR/saving. Use 0 to disable.")
    ap.add_argument("--denoiser_relax", type=float, default=0.5,
                    help="Averaged-operator relaxation; ignored if checkpoint stores denoiser_relax.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    net_W, net_A, net_R, hp = load_models(args.ckpt, args.scale, device, args.rgdn)
    mu = float(hp["mu"])
    rho = float(hp["rho"])
    scale = int(hp["scale"])
    L = int(hp["L"] if args.override_L is None else args.override_L)

    pad = int(args.pad)
    if pad < 0:
        raise ValueError("--pad must be >= 0")
    if pad > 0 and pad % scale != 0:
        raise ValueError(f"--pad is in HR pixels and should be divisible by scale={scale}. Got pad={pad}.")
    lr_pad = pad // scale if pad > 0 else 0

    if "denoiser_relax" in hp:
        relax = float(hp["denoiser_relax"])
        src = "checkpoint"
    else:
        relax = float(args.denoiser_relax)
        src = "CLI (--denoiser_relax)"
    denoiser = make_denoiser(net_R, relax)

    print(f"Loaded models | mu={mu} rho={rho} scale={scale} L={L} | "
          f"denoiser_relax={relax} (from {src}) | HR pad={pad}")
    if "denoiser_relax" not in hp:
        print("  [!] checkpoint did not store training relaxation; ensure --denoiser_relax matches training.")

    kernel_base = make_gaussian_kernel(device)
    paths = sorted(glob.glob(os.path.join(args.test_dir, "*.*")))
    if not paths:
        print(f"No images found in {args.test_dir}")
        return

    tot_sr = tot_uw = tot_bic = 0.0
    for p in paths:
        hr = np.array(Image.open(p).convert("RGB"), np.float32) / 255.0
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, h0, w0 = hr_t.shape
        hr_t = hr_t[:, :, :(h0 // scale) * scale, :(w0 // scale) * scale]
        hr_eval = hr_t

        # Boundary fix: pad HR before degradation.
        hr_input = F.pad(hr_t, (pad, pad, pad, pad), mode="reflect") if pad > 0 else hr_t
        y = degrade(hr_input, kernel_base, scale)

        # Inspection on the actual padded inference input, but report valid-region stats.
        w_map = net_W.get_w_map(y)
        w_valid = crop_lr_pad(w_map, lr_pad)
        alpha_inspect = net_A(y)
        alpha_valid = crop_hr_pad(alpha_inspect, pad)
        sigma_valid = alpha_valid / rho

        print(f"\n--- Stats for {os.path.basename(p)} ---")
        print(f"w_map padded shape : {list(w_map.shape)} [Batch, Nodes, LR_H, LR_W]")
        print(f"w_map valid stats  : min={w_valid.min().item():.4f}, max={w_valid.max().item():.4f}, mean={w_valid.mean().item():.4f}")
        print(f"w valid at (0,0)   : {w_valid[0, :, 0, 0].cpu().numpy().round(4)}")
        print(f"alpha padded shape : {list(alpha_inspect.shape)} [Batch, 1, HR_H, HR_W]")
        print(f"alpha valid stats  : min={alpha_valid.min().item():.5f}, max={alpha_valid.max().item():.5f}, mean={alpha_valid.mean().item():.5f}")
        print(f"Effective σ valid  : min={sigma_valid.min().item():.4f}, max={sigma_valid.max().item():.4f}")

        sr_pad, uw_pad, alpha_pad = super_resolve(y, net_W, net_A, denoiser, mu, rho, scale, L, kernel_base)
        bic_pad = F.interpolate(y, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)

        sr = crop_hr_pad(sr_pad, pad)
        uw = crop_hr_pad(uw_pad, pad)
        bic = crop_hr_pad(bic_pad, pad)
        alpha_map = crop_hr_pad(alpha_pad, pad)

        def psnr(a):
            aa, hh = match_to_hr(a, hr_eval)
            return float(calculate_benchmark_psnr(aa, hh, crop_border=scale))

        ps, pu, pb = psnr(sr), psnr(uw), psnr(bic)
        tot_sr += ps
        tot_uw += pu
        tot_bic += pb

        name = os.path.splitext(os.path.basename(p))[0]
        out = (sr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(out).save(os.path.join(args.save_dir, f"{name}_x{scale}_SR.png"))
        plt.imsave(os.path.join(args.save_dir, f"{name}_alpha_heatmap.png"),
                   alpha_map[0, 0].cpu().numpy(), cmap="jet")

        sigma_map = alpha_map / rho
        print(f"  {name:12s}  bicubic {pb:5.2f} | U_w {pu:5.2f} | ADMM-SR {ps:5.2f} dB"
              f"   (SR mean={float(sr.mean()):.3f}, alpha mean={float(alpha_map.mean()):.5f}, "
              f"sigma mean={float(sigma_map.mean()):.5f})")

    n = len(paths)
    print(f"\n  AVERAGE      bicubic {tot_bic/n:5.2f} | U_w {tot_uw/n:5.2f} | ADMM-SR {tot_sr/n:5.2f} dB")
    print(f"  SR images & Alpha heatmaps written to {args.save_dir}/")


if __name__ == "__main__":
    main()
