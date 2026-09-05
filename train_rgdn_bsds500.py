"""Fresh legacy-RGDN training with sigma conditioning in [0, 0.5].

Derived from the user's train_rgdn_bsds500(2).py. Same default architecture,
70/30 noise mixture, image/gradient/identity losses and Jacobian regularizer.
Validation uses fixed crops/noise and separate low-sigma/original-suite scores.
The empirical Jacobian penalty is NOT a contractivity certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rgdn_model import RGDN, GradientSigmaGuidedAttention

LOW_SIGMAS = (0.001, 0.005, 0.01, 0.03, 0.05)
ORIGINAL_SIGMAS = tuple(s / 255.0 for s in (15, 25, 50, 100, 127))
SIGMA_POLICY = {"name": "legacy_sigma_zero_floor_v1", "minimum": 0.0,
                "maximum": 0.5, "attention_minimum": 0.0,
                "attention_maximum": 0.5, "identity_sigma": 0.001}


def split_paths(root_dir, split):
    """Require an explicit split; never recursively fall back to all BSDS500."""
    root = Path(root_dir)
    candidates = (root / "images" / split, root / split,
                  root / "BSDS500" / "images" / split)
    for folder in candidates:
        if folder.is_dir():
            paths = sorted(p.resolve() for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if paths:
                return paths
    raise FileNotFoundError(f"No images in explicit {split!r} split under {root}.")


def read_crop(path, patch_size, rng, center=False):
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    p = patch_size
    if h >= p and w >= p:
        i = (h - p) // 2 if center else rng.randint(0, h - p)
        j = (w - p) // 2 if center else rng.randint(0, w - p)
        return img[i:i+p, j:j+p].copy()
    # Preserve the original trainer's resize behavior for undersized images.
    from skimage.transform import resize
    return resize(img, (p, p), anti_aliasing=True).astype(np.float32)


class BSDS500Dataset(Dataset):
    def __init__(self, paths, patch_size=64, samples_per_image=50):
        self.image_paths = paths
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image

    def __len__(self):
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx):
        clean = read_crop(self.image_paths[idx % len(self.image_paths)],
                          self.patch_size, random)
        if random.random() > 0.5:
            clean = np.flip(clean, axis=1).copy()
        if random.random() > 0.5:
            clean = np.flip(clean, axis=0).copy()
        if random.random() > 0.5:
            clean = np.rot90(clean, k=random.randint(1, 3)).copy()
        clean = torch.from_numpy(clean).permute(2, 0, 1).float()
        sigma_val = (random.uniform(0.0, 0.15) if random.random() < 0.70
                     else random.uniform(0.15, 0.50))
        p = self.patch_size
        if random.random() > 0.5:
            sigma_map = torch.full((1, p, p), sigma_val, dtype=torch.float32)
        else:
            grid_size = max(1, p // 8)
            low_res_sigma = (torch.randn(1, 1, grid_size, grid_size) * 0.05
                             + sigma_val).clamp(0.0, 0.5)
            sigma_map = F.interpolate(low_res_sigma, size=(p, p), mode="bicubic",
                                      align_corners=False).squeeze(0).clamp(0.0, 0.5)
        noisy = (clean + torch.randn_like(clean) * sigma_map).clamp(0.0, 1.0)
        return noisy, clean, sigma_map


class FixedValidationDataset(Dataset):
    """One center plus reproducible random crops; paired noise across sigmas.

    Local RNGs make samples independent of training RNG, epoch and worker count.
    Each image has the same number of crops and therefore equal metric weight.
    """
    def __init__(self, paths, patch_size=64, patches_per_image=5, seed=4321):
        self.image_paths = paths
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.seed = seed

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        image_idx, crop_idx = divmod(idx, self.patches_per_image)
        sample_seed = self.seed + 1000003 * image_idx + crop_idx
        clean = read_crop(self.image_paths[image_idx], self.patch_size,
                          random.Random(sample_seed), center=(crop_idx == 0))
        clean = torch.from_numpy(clean).permute(2, 0, 1).contiguous().float()
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(sample_seed))
        return clean, noise


def seed_worker(_):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RGDNLoss(nn.Module):
    def __init__(self, gradient_weight: float = 0.1, identity_weight: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.gradient_weight = gradient_weight
        self.identity_weight = identity_weight

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _gradients(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, identity_pred: torch.Tensor = None):
        loss_img = self.l1(pred, target)
        loss_grad = self.l1(self._gradients(pred), self._gradients(target))

        loss_id = torch.tensor(0.0, device=pred.device)
        if identity_pred is not None:
            loss_id = self.l1(identity_pred, target)

        total = loss_img + (self.gradient_weight * loss_grad) + (self.identity_weight * loss_id)
        return total, loss_img, loss_grad, loss_id


# ============================================================
# LOCAL JACOBIAN PENALTY — FP32, outside AMP
# ============================================================
def local_jacobian_penalty_fp32(model, noisy, sigma, eps: float, margin: float, detach_base: bool = False):
    """Finite-difference local Lipschitz penalty computed in full FP32.

    Why FP32 matters:
      The input perturbation has L2 norm eps, so its per-pixel amplitude is tiny.
      Computing this inside AMP/FP16 can make eta noisy and can train the wrong
      signal. For PnP stability, keep this estimate in FP32.

    The estimate is:
        eta = ||D(x + eps*d) - D(x)||_2 / eps,
    where ||d||_2 = 1 for each sample.

    This is a training regularizer, not a mathematical proof of non-expansiveness.
    """
    noisy32 = noisy.float()
    sigma32 = sigma.float()

    d = torch.randn_like(noisy32)
    d = d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
    noisy_pert = (noisy32 + eps * d).clamp(0.0, 1.0)

    # Disable autocast explicitly so both forward passes are full FP32.
    with torch.autocast(device_type=noisy.device.type, enabled=False):
        base = model(noisy32, sigma32)
        if detach_base:
            base = base.detach()
        pert = model(noisy_pert, sigma32)
        eta = (pert - base).flatten(1).norm(dim=1) / eps
        penalty = torch.relu(eta - margin).pow(2).mean()

    return penalty, eta.detach().mean()





@torch.no_grad()
def validate(model, val_loader, device):
    """Mean per-crop RGB PSNR; fixed samples, FP32, no global noise sampling.

    Unlike the old mean of batch-MSE PSNRs, every crop has equal weight.
    Inputs are reproducible; GPU kernels need not be bitwise deterministic.
    """
    was_training = model.training
    model.eval()
    levels = LOW_SIGMAS + ORIGINAL_SIGMAS
    totals = [0.0] * len(levels)
    count = 0
    identity_total = 0.0
    try:
        for clean, noise in val_loader:
            clean = clean.to(device).float()
            noise = noise.to(device).float()
            for i, sigma in enumerate(levels):
                noisy = (clean + sigma * noise).clamp(0.0, 1.0)
                prediction = model(noisy, sigma)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Non-finite validation output at sigma={sigma}")
                mse = (prediction - clean).square().flatten(1).mean(1)
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
                totals[i] += psnr.double().sum().item()
            identity = model(clean, SIGMA_POLICY["identity_sigma"])
            if not torch.isfinite(identity).all():
                raise FloatingPointError("Non-finite clean-input validation output")
            identity_total += (identity-clean).square().flatten(1).mean(1).double().sum().item()
            count += clean.shape[0]
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Empty validation dataset")
    means = [v/count for v in totals]
    result = {
        "num_crops": count,
        "metric": "mean_per_crop_RGB_PSNR_peak1_FP32",
        "per_sigma": [{"sigma": level, "psnr_db": score}
                      for level, score in zip(levels, means)],
        "low_sigma_mean_psnr_db": sum(means[:len(LOW_SIGMAS)])/len(LOW_SIGMAS),
        "original_sigma_mean_psnr_db": sum(means[len(LOW_SIGMAS):])/len(ORIGINAL_SIGMAS),
        "clean_input_mse_at_sigma_0p001": identity_total/count,
    }
    for row in result["per_sigma"]:
        print(f"  sigma={row['sigma']:.6f}: PSNR={row['psnr_db']:.4f} dB")
    print(f"  Low-sigma mean={result['low_sigma_mean_psnr_db']:.4f} dB | "
          f"original-suite mean={result['original_sigma_mean_psnr_db']:.4f} dB | "
          f"clean-input MSE={identity_total/count:.6e}")
    return result


def atomic_save(payload, path):
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def verify_sigma_policy(device):
    # Small startup check, independent of learned image reconstruction quality.
    v = torch.zeros(1, 3, 2, 2, device=device)
    sigma = torch.tensor([0.001, 0.005, 0.03, 0.6], device=device).reshape(1, 1, 2, 2)
    sigma.requires_grad_(True)
    used = RGDN._prepare_sigma(v, sigma)
    expected = torch.tensor([0.001, 0.005, 0.03, 0.5], device=device).reshape_as(sigma)
    if not torch.allclose(used, expected):
        raise RuntimeError("Wrong model imported: expected sigma clamp [0, 0.5]")
    grad, = torch.autograd.grad(used.sum(), sigma)
    if not torch.equal(grad, torch.tensor([1., 1., 1., 0.], device=device).reshape_as(sigma)):
        raise RuntimeError("Sigma-conditioning derivative check failed")
    # Check the attention path too, without consuming the model-initialization RNG.
    with torch.random.fork_rng(devices=[]):
        attention = GradientSigmaGuidedAttention(8)
        attention_sigma = torch.full((1, 1, 8, 8), 0.005, requires_grad=True)
        observed = []
        handle = attention.spatial_attention[0].register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0][:, 1:2]))
        try:
            attention(torch.zeros(1, 8, 8, 8), attention_sigma)
        finally:
            handle.remove()
        if not torch.allclose(observed[0], torch.full_like(attention_sigma, 0.01)):
            raise RuntimeError("Attention still has an incompatible sigma floor")
        attention_grad, = torch.autograd.grad(observed[0].sum(), attention_sigma)
        if not torch.allclose(attention_grad, torch.full_like(attention_sigma, 2.0)):
            raise RuntimeError("Attention sigma-conditioning derivative check failed")
    print("Sigma input AND attention: [0, 0.5]; sub-0.01 conditioning gradients PASS")


def train_rgdn(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or cuda")
    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Fresh run requires an empty/new save directory: {save_dir}")
    train_paths = split_paths(args.data_root, args.train_split)
    val_paths = split_paths(args.data_root, args.val_split)
    if set(train_paths) & set(val_paths):
        raise ValueError("Training and validation splits overlap")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_sigma_policy(device)
    model = RGDN(in_channels=3, num_features=args.num_features,
                 num_blocks=args.num_blocks, use_attention=True).to(device)
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    train_dataset = BSDS500Dataset(train_paths, args.patch_size)
    val_dataset = FixedValidationDataset(val_paths, args.patch_size,
                                         args.val_patches_per_image, args.val_seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(args.val_seed))
    if len(train_loader) == 0:
        raise ValueError("No complete training batch; reduce batch_size")
    criterion = RGDNLoss(args.gradient_weight, args.identity_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=4096.,
                                  growth_interval=4000)
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args), "sigma_policy": SIGMA_POLICY,
        "model_sha256": hashlib.sha256(Path(__file__).with_name("rgdn_model.py").read_bytes()).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "train_images": [str(p) for p in train_paths],
        "validation_images": [str(p) for p in val_paths],
        "low_sigma_levels": LOW_SIGMAS, "original_sigma_levels": ORIGINAL_SIGMAS,
        "validation": "fixed per-index crops/noise; FP32; mean per-crop RGB PSNR",
    }
    (save_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    print(f"Device={device} | AMP={amp_enabled} | parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"Images: train={len(train_paths)}, val={len(val_paths)} | "
          f"training batches={len(train_loader)} | validation crops={len(val_dataset)}")
    print("Fresh initialization | legacy RGDN | sigma [0, 0.5] | no contraction certificate")
    best_low = best_original = -math.inf
    global_step = total_skips = consecutive_skips = 0
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(6, dtype=np.float64)
        updates = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for noisy, clean, sigma in pbar:
            noisy, clean, sigma = [x.to(device, non_blocking=True).float()
                                   for x in (noisy, clean, sigma)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                denoised = model(noisy, sigma)
                identity = model(clean, torch.full_like(sigma, 0.001))
                loss, image_loss, gradient_loss, identity_loss = criterion(denoised, clean, identity)
            if args.jacobian_weight > 0 and global_step % args.jacobian_every == 0:
                penalty, eta = local_jacobian_penalty_fp32(
                    model, noisy, sigma, args.jacobian_eps, args.jacobian_margin,
                    args.jacobian_detach_base)
                loss = loss + args.jacobian_weight * penalty
            else:
                penalty = eta = loss.new_zeros(())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite forward loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grads_finite = all(torch.isfinite(p.grad).all().item()
                               for p in model.parameters() if p.grad is not None)
            global_step += 1
            if not grads_finite:
                if not amp_enabled:
                    raise FloatingPointError("Non-finite gradient without AMP")
                old_scale = scaler.get_scale()
                # unscale_ recorded nonfinite gradients; step skips the update.
                scaler.step(optimizer)
                scaler.update()
                total_skips += 1
                consecutive_skips += 1
                print(f"AMP overflow: skipped update, scale {old_scale}->{scaler.get_scale()}, total={total_skips}")
                if not scaler.get_scale() < old_scale or consecutive_skips > 8 or total_skips > 100:
                    raise FloatingPointError("AMP overflow recovery limit exceeded")
                continue
            consecutive_skips = 0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            totals += [x.item() for x in (loss, image_loss, gradient_loss, identity_loss, penalty, eta)]
            pbar.set_postfix(L1=f"{image_loss.item():.4f}", Id=f"{identity_loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if updates == 0:
            raise FloatingPointError("No successful optimizer updates in epoch")
        scheduler.step()
        means = totals / updates
        print(f"Epoch {epoch+1}: loss={means[0]:.6f} L1={means[1]:.6f} "
              f"gradient={means[2]:.6f} identity={means[3]:.6f} "
              f"jacobian_penalty={means[4]:.6f} eta_diagnostic={means[5]:.4f} "
              f"updates={updates} total_amp_skips={total_skips}")
        validation = None
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            validation = validate(model, val_loader, device)
        payload = {
            "epoch": epoch, "epoch_number": epoch+1,
            "model_state_dict": raw_model.state_dict(), "args": args,
            "rgdn_arch": "legacy_rgdn_v1", "sigma_policy": SIGMA_POLICY,
            "validation": validation, "total_amp_skips": total_skips,
            "metadata": metadata,
        }
        if validation is not None:
            low = validation["low_sigma_mean_psnr_db"]
            original = validation["original_sigma_mean_psnr_db"]
            if low > best_low:
                best_low = low
                selected = dict(payload, selection_metric="low_sigma_mean_psnr_db", best_avg_psnr=low)
                atomic_save(selected, save_dir / "best_model_rgdn_lowsigma.pth")
                # Compatibility filename intentionally aliases low-sigma selection.
                atomic_save(selected, save_dir / "best_model_rgdn.pth")
                print("Saved best_model_rgdn_lowsigma.pth and best_model_rgdn.pth (low-sigma selection)")
            if original > best_original:
                best_original = original
                atomic_save(dict(payload, selection_metric="original_sigma_mean_psnr_db",
                                 best_avg_psnr=original), save_dir / "best_model_rgdn_originalsigma.pth")
                print("Saved best_model_rgdn_originalsigma.pth")
        # Every epoch leaves an inference checkpoint, even between validation dates.
        atomic_save(payload, save_dir / "last_model_rgdn.pth")
        if (epoch+1) % args.save_every == 0 or epoch+1 == args.epochs:
            atomic_save(payload, save_dir / f"checkpoint_epoch_{epoch+1}.pth")
        record = {"epoch": epoch+1, "successful_updates": updates,
                  "total_amp_skips": total_skips, "training_loss_mean": means.tolist(),
                  "validation": validation}
        with (save_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
    print(f"Done: best low-sigma={best_low:.4f} dB; best original-suite={best_original:.4f} dB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="./data/BSDS500")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default="val")
    p.add_argument("--save_dir", default="./rgdn_checkpoints_sigma0_clean")
    p.add_argument("--device", default=None)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--num_features", type=int, default=64)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val_seed", type=int, default=4321)
    p.add_argument("--val_patches_per_image", type=int, default=5)
    p.add_argument("--gradient_weight", type=float, default=0.1)
    p.add_argument("--identity_weight", type=float, default=0.1)
    p.add_argument("--jacobian_weight", type=float, default=3e-4)
    p.add_argument("--jacobian_eps", type=float, default=0.003)
    p.add_argument("--jacobian_margin", type=float, default=0.9)
    p.add_argument("--jacobian_every", type=int, default=1)
    p.add_argument("--jacobian_detach_base", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_amp", action="store_true", help="Disable default CUDA AMP")
    args = p.parse_args()
    for name in ("epochs", "batch_size", "patch_size", "num_features", "num_blocks",
                 "val_every", "save_every", "val_patches_per_image", "jacobian_every"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.patch_size < 8 or args.num_workers < 0 or not args.train_split or not args.val_split:
        p.error("patch_size must be >=8, num_workers >=0 and split names nonempty")
    for name in ("lr", "jacobian_eps"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name} must be finite and positive")
    for name in ("weight_decay", "grad_clip", "gradient_weight", "identity_weight",
                 "jacobian_weight", "jacobian_margin"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"--{name} must be finite and nonnegative")
    if not 0 <= args.seed < 2**32 or not 0 <= args.val_seed < 2**32:
        p.error("seeds must be in [0, 2**32)")
    train_rgdn(args)


if __name__ == "__main__":
    main()
