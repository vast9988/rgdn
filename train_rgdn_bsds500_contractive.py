"""Train and verify the certified contractive RGDN on BSDS500.

Unlike the legacy trainer, contractivity is not encouraged by a one-direction
finite-difference penalty.  It is guaranteed by ``ContractiveRGDN`` for every
checkpoint.  This script additionally runs deterministic pairwise and local
Jacobian stress tests to catch implementation, serialization, or numerical
errors before a checkpoint can be selected as ``best``.

The default selection metric emphasizes sigma values 15--30/255 because that
contains the observed BL-LCILW PnP operating region (roughly 0.06--0.12 when
rho=0.05), while validation still reports the conventional wider noise range.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rgdn_model_contractive import ContractiveRGDN, load_contractive_rgdn


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def discover_images(root: Path, split: str) -> List[Path]:
    candidates = (
        root / "images" / split,
        root / split,
        root / "BSDS500" / "images" / split,
    )
    for directory in candidates:
        if directory.is_dir():
            images: List[Path] = []
            for pattern in IMAGE_EXTENSIONS:
                images.extend(directory.rglob(pattern))
                images.extend(directory.rglob(pattern.upper()))
            unique = sorted({path.resolve() for path in images})
            if unique:
                return unique
    raise FileNotFoundError(
        f"no images found for split={split!r} below {root}; expected images/{split}"
    )


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resize_minimum(clean: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = clean.shape
    if height >= size and width >= size:
        return clean
    scale = max(size / max(height, 1), size / max(width, 1))
    new_height = max(size, int(math.ceil(height * scale)))
    new_width = max(size, int(math.ceil(width * scale)))
    return F.interpolate(
        clean.unsqueeze(0),
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0.0, 1.0)


def augment_patch(clean: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        clean = clean.flip(-1)
    if random.random() < 0.5:
        clean = clean.flip(-2)
    rotations = random.randrange(4)
    if rotations:
        clean = torch.rot90(clean, rotations, dims=(-2, -1))
    return clean.contiguous()


class BSDS500NoiseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        patch_size: int,
        patches_per_image: int,
        operational_sigma: Tuple[float, float],
        sigma_min: float,
        sigma_max: float,
        spatial_sigma_probability: float,
    ):
        self.paths = discover_images(root, split)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.operational_sigma = operational_sigma
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.spatial_sigma_probability = float(spatial_sigma_probability)
        if self.patch_size <= 0 or self.patches_per_image <= 0:
            raise ValueError("patch_size and patches_per_image must be positive")

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def _sample_sigma_value(self) -> float:
        # Concentrate training where AlphaNet/rho actually drives the PnP prior,
        # but retain low- and high-noise coverage.
        draw = random.random()
        op_low, op_high = self.operational_sigma
        if draw < 0.65:
            return random.uniform(op_low, op_high)
        if draw < 0.85:
            return random.uniform(self.sigma_min, op_low)
        return random.uniform(op_high, self.sigma_max)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = resize_minimum(load_rgb(self.paths[index % len(self.paths)]), self.patch_size)
        _, height, width = clean.shape
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        clean = clean[:, top : top + self.patch_size, left : left + self.patch_size]
        clean = augment_patch(clean)

        sigma_value = self._sample_sigma_value()
        if random.random() < self.spatial_sigma_probability:
            grid = max(2, self.patch_size // 16)
            low_resolution = torch.randn(1, 1, grid, grid) * 0.015 + sigma_value
            sigma = F.interpolate(
                low_resolution,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(self.sigma_min, self.sigma_max)
        else:
            sigma = torch.full(
                (1, self.patch_size, self.patch_size), sigma_value, dtype=torch.float32
            )
        noisy = (clean + torch.randn_like(clean) * sigma).clamp(0.0, 1.0)
        return noisy, clean, sigma


class BSDS500ValidationDataset(Dataset):
    """One deterministic center patch per validation image."""

    def __init__(self, root: Path, split: str, patch_size: int, maximum: int = 0):
        paths = discover_images(root, split)
        self.paths = paths[:maximum] if maximum > 0 else paths
        self.patch_size = int(patch_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        clean = resize_minimum(load_rgb(self.paths[index]), self.patch_size)
        _, height, width = clean.shape
        top = (height - self.patch_size) // 2
        left = (width - self.patch_size) // 2
        clean = clean[:, top : top + self.patch_size, left : left + self.patch_size]
        return clean.contiguous(), self.paths[index].name


def image_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    channels = x.shape[1]
    sobel_x = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    sobel_y = sobel_x.T.contiguous()
    gx = F.conv2d(x, sobel_x[None, None].expand(channels, 1, 3, 3), padding=1, groups=channels)
    gy = F.conv2d(x, sobel_y[None, None].expand(channels, 1, 3, 3), padding=1, groups=channels)
    return gx, gy


def parse_integer_list(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 or item > 255 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated integers in [1,255]")
    return result


@torch.no_grad()
def validate_psnr(
    model: ContractiveRGDN,
    loader: DataLoader,
    device: torch.device,
    sigma_levels: Sequence[int],
    selection_levels: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    model.eval()
    per_sigma: Dict[str, float] = {}
    for sigma_integer in sigma_levels:
        sigma = sigma_integer / 255.0
        image_psnrs: List[float] = []
        for batch_index, (clean_cpu, _) in enumerate(loader):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 100_003 * sigma_integer + batch_index)
            noise = torch.randn(clean_cpu.shape, generator=generator, dtype=clean_cpu.dtype)
            noisy = (clean_cpu + sigma * noise).clamp(0.0, 1.0).to(device)
            clean = clean_cpu.to(device)
            prediction = model(noisy, sigma)
            mse = (prediction - clean).square().flatten(1).mean(1).clamp_min(1.0e-12)
            image_psnrs.extend((-10.0 * torch.log10(mse)).cpu().tolist())
        per_sigma[str(sigma_integer)] = float(np.mean(image_psnrs))

    selection_missing = sorted(set(selection_levels) - set(sigma_levels))
    if selection_missing:
        raise ValueError(f"selection sigma levels were not validated: {selection_missing}")
    selection = float(np.mean([per_sigma[str(level)] for level in selection_levels]))
    overall = float(np.mean(list(per_sigma.values())))
    return {
        "psnr_db_by_sigma_255": per_sigma,
        "selection_sigmas_255": list(selection_levels),
        "selection_mean_psnr_db": selection,
        "all_sigma_mean_psnr_db": overall,
    }


@torch.no_grad()
def pairwise_stress_test(
    model: ContractiveRGDN,
    device: torch.device,
    trials: int,
    patch_size: int,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ratios: List[float] = []
    output_min = float("inf")
    output_max = -float("inf")
    for trial in range(trials):
        # Include the wider transient range possible for x+u in ADMM.
        v = 1.5 * torch.rand(1, 3, patch_size, patch_size, generator=generator) - 0.25
        vp = 1.5 * torch.rand(1, 3, patch_size, patch_size, generator=generator) - 0.25
        sigma_value = 0.001 + 0.499 * torch.rand(1, generator=generator).item()
        if trial % 2:
            sigma = torch.full((1, 1, patch_size, patch_size), sigma_value)
        else:
            grid = max(2, patch_size // 8)
            sigma = torch.rand(1, 1, grid, grid, generator=generator) * 0.499 + 0.001
        v, vp, sigma = v.to(device), vp.to(device), sigma.to(device)
        dv, dvp = model(v, sigma), model(vp, sigma)
        denominator = (v - vp).norm().clamp_min(1.0e-12)
        ratios.append(float(((dv - dvp).norm() / denominator).cpu()))
        output_min = min(output_min, float(dv.min().cpu()), float(dvp.min().cpu()))
        output_max = max(output_max, float(dv.max().cpu()), float(dvp.max().cpu()))
    return {
        "maximum_pairwise_ratio": max(ratios),
        "mean_pairwise_ratio": float(np.mean(ratios)),
        "output_minimum": output_min,
        "output_maximum": output_max,
    }


def local_jacobian_power_estimate(
    model: ContractiveRGDN,
    device: torch.device,
    patch_size: int,
    iterations: int,
    seed: int,
    sigma: float = 0.08,
) -> float:
    """Diagnostic power iteration for one local Jacobian (not the proof)."""
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.rand(1, 3, patch_size, patch_size, generator=generator).to(device)
    direction = torch.randn(x.shape, generator=generator).to(device)
    direction = direction / direction.norm().clamp_min(1.0e-12)
    estimate = 0.0
    for _ in range(iterations):
        _, jvp = torch.autograd.functional.jvp(
            lambda value: model(value, sigma),
            x,
            direction,
            create_graph=False,
            strict=False,
        )
        estimate = float(jvp.norm().detach().cpu())
        if estimate <= 1.0e-12:
            return 0.0
        output_direction = (jvp / jvp.norm().clamp_min(1.0e-12)).detach()
        x_graph = x.detach().requires_grad_(True)
        output = model(x_graph, sigma)
        vjp = torch.autograd.grad(
            outputs=output,
            inputs=x_graph,
            grad_outputs=output_direction,
            retain_graph=False,
            create_graph=False,
        )[0]
        norm = vjp.norm().detach()
        if float(norm.cpu()) <= 1.0e-12:
            return 0.0
        direction = (vjp / norm).detach()
    return estimate


def verify_model(
    model: ContractiveRGDN,
    device: torch.device,
    pairwise_trials: int,
    test_patch_size: int,
    power_iterations: int,
    tolerance: float,
    seed: int,
) -> Dict[str, Any]:
    certificate = model.certificate()
    pairwise = pairwise_stress_test(
        model, device, pairwise_trials, test_patch_size, seed + 17
    )
    jacobian = local_jacobian_power_estimate(
        model, device, test_patch_size, power_iterations, seed + 31
    )
    eta = float(certificate["input_lipschitz_upper_bound"])
    gates = {
        "analytic_strict_contraction": bool(certificate["strict_contraction"]),
        "analytic_layers_nonexpansive": bool(
            certificate["all_certified_layers_nonexpansive"]
        ),
        "pairwise_ratio_at_most_eta_plus_tolerance": bool(
            pairwise["maximum_pairwise_ratio"] <= eta + tolerance
        ),
        "jacobian_estimate_at_most_eta_plus_tolerance": bool(
            jacobian <= eta + tolerance
        ),
        "bounded_output_observed": bool(
            pairwise["output_minimum"] >= -tolerance
            and pairwise["output_maximum"] <= 1.0 + tolerance
        ),
    }
    return {
        "certificate": certificate,
        "pairwise_stress": pairwise,
        "local_jacobian_power_estimate": jacobian,
        "numeric_tolerance": tolerance,
        "gates": gates,
        "passed": all(gates.values()),
        "note": "numeric checks are diagnostics; the guarantee is structural",
    }


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_checkpoint(
    model: ContractiveRGDN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    validation: Dict[str, Any],
    verification: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "checkpoint_format": "contractive_rgdn_v1",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": model.model_config(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation": validation,
        "contractivity_verification": verification,
        "certificate": model.certificate(),
        "training_args": vars(args),
    }


def print_verification(report: Dict[str, Any]) -> None:
    certificate = report["certificate"]
    stress = report["pairwise_stress"]
    print(
        "Contractivity | "
        f"certified eta={certificate['input_lipschitz_upper_bound']:.6f} | "
        f"averaged alpha={certificate['valid_averaged_alpha']:.6f} | "
        f"pairwise max={stress['maximum_pairwise_ratio']:.6f} | "
        f"Jacobian~={report['local_jacobian_power_estimate']:.6f} | "
        f"PASS={report['passed']}"
    )


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed, deterministic=not args.nondeterministic)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = ContractiveRGDN(
        in_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        eta=args.eta,
        gradient_coeff=args.gradient_coeff,
        anchor=args.anchor,
        residual_mix_init=args.residual_mix_init,
        output_mix_init=args.output_mix_init,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    ).to(device)
    certificate = model.certificate()
    print(json.dumps(certificate, indent=2))
    if args.jacobian_weight != 0.0:
        print(
            "WARNING: --jacobian_weight/--jacobian_eps are deprecated and ignored. "
            "This architecture enforces the bound structurally."
        )

    root = Path(args.data_root)
    train_dataset = BSDS500NoiseDataset(
        root=root,
        split=args.train_split,
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        operational_sigma=(args.operational_sigma_low, args.operational_sigma_high),
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        spatial_sigma_probability=args.spatial_sigma_probability,
    )
    validation_dataset = BSDS500ValidationDataset(
        root=root,
        split=args.val_split,
        patch_size=args.val_patch_size,
        maximum=args.max_val_images,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"Device={device} | train images={len(train_dataset.paths)} | "
        f"train samples/epoch={len(train_dataset)} | "
        f"validation images={len(validation_dataset)} | "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # Compatibility with older PyTorch.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_directory = Path(args.save_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    initial_verification = verify_model(
        model,
        device,
        args.pairwise_trials,
        args.test_patch_size,
        args.power_iters,
        args.contractivity_tolerance,
        args.seed,
    )
    print_verification(initial_verification)
    if not initial_verification["passed"]:
        raise RuntimeError("initial structural/numerical contractivity verification failed")

    best_score = -float("inf")
    best_path = output_directory / "best_model_rgdn_contractive.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "mse": 0.0, "gradient": 0.0, "identity": 0.0}
        batches = 0
        progress = tqdm(train_loader, desc=f"RGDN {epoch:03d}/{args.epochs:03d}")
        for noisy, clean, sigma in progress:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            sigma = sigma.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(noisy, sigma)
                mse = F.mse_loss(prediction, clean)
                gradient_loss = prediction.new_zeros(())
                if args.gradient_weight > 0.0:
                    pred_gx, pred_gy = image_gradients(prediction)
                    clean_gx, clean_gy = image_gradients(clean)
                    gradient_loss = F.l1_loss(pred_gx, clean_gx) + F.l1_loss(
                        pred_gy, clean_gy
                    )
                identity_loss = prediction.new_zeros(())
                if args.identity_weight > 0.0:
                    identity = model(clean, args.sigma_min)
                    identity_loss = F.mse_loss(identity, clean)
                loss = (
                    mse
                    + args.gradient_weight * gradient_loss
                    + args.identity_weight * identity_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite RGDN training gradient")
            scaler.step(optimizer)
            scaler.update()
            batches += 1
            sums["total"] += float(loss.detach())
            sums["mse"] += float(mse.detach())
            sums["gradient"] += float(gradient_loss.detach())
            sums["identity"] += float(identity_loss.detach())
            progress.set_postfix(
                mse=f"{float(mse.detach()):.5f}",
                grad_norm=f"{float(gradient_norm):.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        scheduler.step()
        means = {key: value / max(batches, 1) for key, value in sums.items()}
        print(f"Epoch {epoch}: training={json.dumps(means, sort_keys=True)}")

        should_validate = epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs
        if should_validate:
            psnr = validate_psnr(
                model,
                validation_loader,
                device,
                args.validation_sigmas,
                args.selection_sigmas,
                args.seed,
            )
            verification = verify_model(
                model,
                device,
                args.pairwise_trials,
                args.test_patch_size,
                args.power_iters,
                args.contractivity_tolerance,
                args.seed + epoch,
            )
            print(f"Validation PSNR: {json.dumps(psnr, indent=2)}")
            print_verification(verification)
            payload = make_checkpoint(
                model, optimizer, scheduler, args, epoch, psnr, verification
            )
            atomic_torch_save(payload, output_directory / "last_model_rgdn_contractive.pth")
            score = float(psnr["selection_mean_psnr_db"])
            if verification["passed"] and score > best_score:
                best_score = score
                atomic_torch_save(payload, best_path)
                # Compatibility alias for scripts whose default is best_model_rgdn.pth.
                atomic_torch_save(payload, output_directory / "best_model_rgdn.pth")
                print(f"New verified best: {best_path} ({best_score:.4f} dB)")

        if epoch % args.save_every == 0:
            periodic_verification = verify_model(
                model,
                device,
                args.pairwise_trials,
                args.test_patch_size,
                args.power_iters,
                args.contractivity_tolerance,
                args.seed + 10_000 + epoch,
            )
            periodic_payload = make_checkpoint(
                model,
                optimizer,
                scheduler,
                args,
                epoch,
                {"not_run_this_epoch": not should_validate},
                periodic_verification,
            )
            atomic_torch_save(
                periodic_payload, output_directory / f"checkpoint_epoch_{epoch:03d}.pth"
            )

    if not best_path.is_file():
        raise RuntimeError("training finished without a verified best checkpoint")
    print(f"Training complete | best operational PSNR={best_score:.4f} dB | {best_path}")
    return best_path


def verify_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, checkpoint = load_contractive_rgdn(args.verify_only, map_location=device)
    model = model.to(device).eval()
    report = verify_model(
        model,
        device,
        args.pairwise_trials,
        args.test_patch_size,
        args.power_iters,
        args.contractivity_tolerance,
        args.seed,
    )
    print(json.dumps(report, indent=2))
    saved = checkpoint.get("certificate")
    if saved is not None:
        print("Saved certificate:")
        print(json.dumps(saved, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an analytically certified bounded contractive RGDN"
    )
    parser.add_argument("--data_root", default="./data/BSDS500")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--save_dir", default="./rgdn_contractive_checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--verify_only", default=None, metavar="CHECKPOINT")

    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.99)
    parser.add_argument("--gradient_coeff", type=float, default=0.20)
    parser.add_argument("--anchor", type=float, default=0.50)
    parser.add_argument("--residual_mix_init", type=float, default=0.10)
    parser.add_argument("--output_mix_init", type=float, default=0.25)
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--sigma_max", type=float, default=0.50)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--patches_per_image", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient_weight", type=float, default=0.0)
    parser.add_argument("--identity_weight", type=float, default=0.0)
    parser.add_argument("--spatial_sigma_probability", type=float, default=0.50)
    parser.add_argument("--operational_sigma_low", type=float, default=0.05)
    parser.add_argument("--operational_sigma_high", type=float, default=0.13)

    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--val_patch_size", type=int, default=128)
    parser.add_argument("--max_val_images", type=int, default=100)
    parser.add_argument(
        "--validation_sigmas",
        type=parse_integer_list,
        default=parse_integer_list("15,20,25,30,50,100,127"),
    )
    parser.add_argument(
        "--selection_sigmas",
        type=parse_integer_list,
        default=parse_integer_list("15,20,25,30"),
    )
    parser.add_argument("--pairwise_trials", type=int, default=16)
    parser.add_argument("--test_patch_size", type=int, default=32)
    parser.add_argument("--power_iters", type=int, default=12)
    parser.add_argument("--contractivity_tolerance", type=float, default=2.0e-3)

    # Accepted only to make migration from the legacy command explicit.
    parser.add_argument("--jacobian_weight", type=float, default=0.0)
    parser.add_argument("--jacobian_eps", type=float, default=0.003)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
    if args.val_every <= 0 or args.save_every <= 0:
        raise ValueError("val_every and save_every must be positive")
    if args.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if not 0.0 <= args.spatial_sigma_probability <= 1.0:
        raise ValueError("spatial_sigma_probability must lie in [0,1]")
    if not args.sigma_min < args.operational_sigma_low < args.operational_sigma_high < args.sigma_max:
        raise ValueError(
            "require sigma_min < operational_sigma_low < operational_sigma_high < sigma_max"
        )
    if args.pairwise_trials <= 0 or args.power_iters <= 0 or args.test_patch_size <= 0:
        raise ValueError("contractivity diagnostic sizes/counts must be positive")
    if args.contractivity_tolerance < 0.0:
        raise ValueError("contractivity_tolerance must be nonnegative")


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    if args.verify_only:
        seed_everything(args.seed, deterministic=not args.nondeterministic)
        verify_checkpoint(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
