"""BL-LCILW Algorithm 2: calibrated SGD, MSE-only Moreau smoothing.

This controlled variant keeps the validated reconstruction machinery from
``train_algo2_bilevel_xonly_paper_fixedk8.py`` while changing the requested
upper-level experiment:

* the sole upper objective is valid-region reconstruction MSE;
* its Moreau--Yosida envelope is evaluated in exact closed form;
* WeightNet and AlphaNet use plain mini-batch gradient descent, with no Adam,
  momentum, weight decay, or adaptive optimizer state;
* epsilon follows Algorithm 2's gradient-norm-triggered geometric decay;
* every outer iteration performs the paper's Gauss--Seidel upper update:
  WeightNet first, then AlphaNet using the updated weights, followed by exactly
  one real three-state PnP-ADMM step;
* a simultaneous/Jacobi update remains available only as an explicit ablation;
* the validated x-only fixed-K=8 Neumann hypergradient and fail-closed numerical
  checks are retained;
* ``--rgdn_arch contractive`` loads the analytically certified bounded
  contraction from ``rgdn_model_contractive.py`` without changing the legacy
  checkpoint path used by the original experiment.

For J(x)=1/2||P(x-x_gt)||_2^2, with P selecting non-padding pixels,

    prox_{epsilon J}(x) = x                                  outside P,
    prox_{epsilon J}(x) = (x + epsilon*x_gt)/(1+epsilon)      inside P,
    J_epsilon(x)        = J(x)/(1+epsilon),
    grad J_epsilon(x)   = P(x-x_gt)/(1+epsilon).

This is a literal Moreau envelope, not a sampled approximation. Since MSE is
already smooth, the envelope changes its scale but not its minimizer or gradient
direction. It is retained for Algorithm 2's continuation schedule; improvement
must come from the bilevel/ADMM optimization, not from claiming that this
quadratic envelope changes the reconstruction optimum.

The optimized variables are network parameters theta=(theta_w,theta_alpha).
The adaptive gate measures their combined implicit hypergradient after the
lower-level step. ``--adaptive_norm rms`` (default) uses
||grad_theta J_epsilon||_2/sqrt(dim(theta)); ``--adaptive_norm l2`` uses the
literal unnormalised Euclidean norm. Both values are logged.

Reference: N. Parikh and S. Boyd, "Proximal Algorithms," Foundations and
Trends in Optimization, 2014, Sec. 3.1.

Author: Abdellah Jarmouni
"""

import os
import math
import argparse
import logging
import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.fft
from tqdm import tqdm


PAPER_ADMM_OUTER_ITERS = 12
PAPER_NEUMANN_K = 8
ALPHANET_ALPHA_MIN = 1.0e-3
ALPHANET_ALPHA_MAX = 2.5e-2
DEFAULT_SIGMA_INIT = 0.12
DEFAULT_NEUMANN_DIAGNOSTIC_TOL = 1.0e-2
DEFAULT_NEUMANN_CATASTROPHIC_LIMIT = 100.0
DEFAULT_MOREAU_EPSILON = 1.0e-1
DEFAULT_EPSILON_GAMMA = 8.0e-1
DEFAULT_EPSILON_SIGMA = 5.0e-1
DEFAULT_EPSILON_TOL = 1.0e-4
DEFAULT_LR_W = 3.0e-6
DEFAULT_LR_ALPHA = 1.0e-5
DEFAULT_GRAD_CLIP_W = 100.0
DEFAULT_GRAD_CLIP_ALPHA = 100.0

try:
    from weighted_lci_upsampler import (
        PatchWeightedLCI2D,
        AlphaNet,
    )
    from rgdn_model import RGDN as LegacyRGDN
    from rgdn_model_contractive import ContractiveRGDN
    from train_global_lci import BSDS500SRDataset
    _PROJECT_IMPORT_ERROR = None
except ImportError as exc:  # Allows the dependency-free --self_test to run.
    _PROJECT_IMPORT_ERROR = exc


def build_rgdn_from_checkpoint(rgdn_arch, checkpoint_path, device):
    """Construct and load the requested RGDN without architecture guessing."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if rgdn_arch == 'contractive':
        if not isinstance(checkpoint, dict):
            raise TypeError('contractive RGDN checkpoint must be a dictionary')
        if checkpoint.get('checkpoint_format') != 'contractive_rgdn_v1':
            raise ValueError(
                'expected checkpoint_format=contractive_rgdn_v1 for '
                f'--rgdn_arch contractive, got {checkpoint.get("checkpoint_format")!r}'
            )
        model_config = checkpoint.get('model_config')
        if not isinstance(model_config, dict):
            raise KeyError(
                'contractive checkpoint is missing model_config; refusing to guess'
            )
        model = ContractiveRGDN(**model_config).to(device)
        state = checkpoint.get('model_state_dict')
        if state is None:
            raise KeyError('contractive checkpoint is missing model_state_dict')
        model.load_state_dict(state, strict=True)
        certificate = model.certificate()
        if not certificate['strict_contraction']:
            raise RuntimeError(f'RGDN certificate is not contractive: {certificate}')
        if not certificate['all_certified_layers_nonexpansive']:
            raise RuntimeError(f'RGDN layer certificate failed: {certificate}')
        metadata = {
            'architecture': model.architecture_name,
            'checkpoint_format': checkpoint['checkpoint_format'],
            'model_config': model.model_config(),
            'certificate': certificate,
        }
        return model, metadata

    if rgdn_arch != 'legacy':
        raise ValueError(f'unknown RGDN architecture {rgdn_arch!r}')
    model = LegacyRGDN(
        in_channels=3, num_features=64, num_blocks=8, use_attention=True
    ).to(device)
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    return model, {
        'architecture': 'legacy_rgdn',
        'checkpoint_format': checkpoint.get('checkpoint_format', 'legacy')
        if isinstance(checkpoint, dict) else 'legacy_raw_state',
        'model_config': {
            'in_channels': 3,
            'num_features': 64,
            'num_blocks': 8,
            'use_attention': True,
        },
        'certificate': None,
    }


@dataclass
class AdaptiveMoreauState:
    """Persistent state for Algorithm 2's triggered epsilon continuation.

    With ``norm_mode='rms'``, the gate is equivalently written in raw L2 form

        ||g||_2 < sigma * sqrt(parameter_count) * gamma * epsilon.

    This avoids making the decision depend only on the number of neural
    parameters. ``norm_mode='l2'`` implements the unscaled printed criterion.
    """

    epsilon: float
    gamma: float
    sigma: float
    epsilon_tol: float
    parameter_count: int
    norm_mode: str = 'rms'
    reductions: int = 0
    converged: bool = False

    def __post_init__(self):
        self.epsilon = float(self.epsilon)
        self.gamma = float(self.gamma)
        self.sigma = float(self.sigma)
        self.epsilon_tol = float(self.epsilon_tol)
        self.parameter_count = int(self.parameter_count)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError('epsilon_0 must be positive and finite')
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma < 1.0:
            raise ValueError('epsilon_gamma must lie strictly between 0 and 1')
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError('epsilon_sigma must be positive and finite')
        if not math.isfinite(self.epsilon_tol) or self.epsilon_tol <= 0.0:
            raise ValueError('epsilon_tol must be positive and finite')
        if self.parameter_count <= 0:
            raise ValueError('parameter_count must be positive')
        if self.norm_mode not in {'rms', 'l2'}:
            raise ValueError("adaptive_norm must be either 'rms' or 'l2'")

    def metric(self, raw_l2_norm):
        raw_l2_norm = float(raw_l2_norm)
        if not math.isfinite(raw_l2_norm) or raw_l2_norm < 0.0:
            raise FloatingPointError('adaptive gradient norm is invalid')
        if self.norm_mode == 'rms':
            return raw_l2_norm / math.sqrt(self.parameter_count)
        return raw_l2_norm

    def update(self, raw_l2_norm):
        """Apply Algorithm 2 lines 13--19 and return gate diagnostics."""
        epsilon_before = self.epsilon
        metric_value = self.metric(raw_l2_norm)
        threshold = self.sigma * self.gamma * epsilon_before
        triggered = bool(not self.converged and metric_value < threshold)
        if triggered:
            self.epsilon = self.gamma * epsilon_before
            self.reductions += 1
        if self.sigma * self.epsilon < self.epsilon_tol:
            self.converged = True
        return {
            'epsilon_before': epsilon_before,
            'epsilon_after': self.epsilon,
            'raw_l2_norm': float(raw_l2_norm),
            'metric_norm': metric_value,
            'threshold': threshold,
            'triggered': triggered,
            'reductions': self.reductions,
            'converged': self.converged,
        }

    def state_dict(self):
        return {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'sigma': self.sigma,
            'epsilon_tol': self.epsilon_tol,
            'parameter_count': self.parameter_count,
            'norm_mode': self.norm_mode,
            'reductions': self.reductions,
            'converged': self.converged,
        }


def _require_project_dependencies():
    if _PROJECT_IMPORT_ERROR is not None:
        raise ImportError(
            'Training requires weighted_lci_upsampler.py, rgdn_model.py, and '
            'train_global_lci.py in the project directory.'
        ) from _PROJECT_IMPORT_ERROR


# ============================================================
# Live file logging
# ============================================================

def setup_live_file_logger(out_prefix, log_dir='logs'):
    """Create a per-run file logger that flushes every record to disk."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{out_prefix}_{timestamp}.log')

    logger = logging.getLogger('bl_lcilw_training')
    logger.setLevel(logging.INFO)
    for existing_handler in logger.handlers:
        existing_handler.close()
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(
        log_path,
        mode='w',
        encoding='utf-8',
        delay=False,
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(handler)

    print(f'Live training log: {log_path}')
    return logger, log_path


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    """Seed model initialization and CPU/CUDA random-number generators."""
    seed = int(seed)
    if not 0 <= seed < 2 ** 32:
        raise ValueError(f'seed must be in [0, 2**32), got {seed}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ReflectionPad1d has no deterministic CUDA backward implementation.
    # Keep deterministic implementations enabled where available, but warn and
    # continue for this known unsupported operation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_dataloader_worker(_worker_id):
    """Seed Python/NumPy inside each DataLoader worker from PyTorch's seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def initialize_alphanet_constant_sigma(net_alpha, rho, sigma_init):
    """Initialize the current AlphaNet to the paper's explicit alpha_0.

    The project AlphaNet maps ``raw`` to

        alpha = 0.001 + 0.024 * sigmoid(raw).

    Zeroing the final convolution weights and setting its bias to the inverse
    logit therefore gives an exactly constant, spatially neutral initial map.
    The network remains fully trainable after initialization.
    """
    rho = float(rho)
    sigma_init = float(sigma_init)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError(f'rho must be positive and finite, got {rho}')
    if not math.isfinite(sigma_init) or sigma_init <= 0.0:
        raise ValueError(
            f'sigma_init must be positive and finite, got {sigma_init}'
        )

    alpha_init = rho * sigma_init
    if not ALPHANET_ALPHA_MIN < alpha_init < ALPHANET_ALPHA_MAX:
        sigma_low = ALPHANET_ALPHA_MIN / rho
        sigma_high = ALPHANET_ALPHA_MAX / rho
        raise ValueError(
            'Requested sigma_init is outside AlphaNet\'s open representable '
            f'range ({sigma_low:.6g}, {sigma_high:.6g}) for rho={rho:.6g}; '
            f'got sigma_init={sigma_init:.6g} (alpha_init={alpha_init:.6g}).'
        )

    if not hasattr(net_alpha, 'net') or len(net_alpha.net) == 0:
        raise TypeError(
            'AlphaNet must expose its sequential predictor as .net so the '
            'paper alpha_0 initialization can be applied safely.'
        )
    final_layer = net_alpha.net[-1]
    if not isinstance(final_layer, nn.Conv2d) or final_layer.out_channels != 1:
        raise TypeError(
            'Expected AlphaNet.net[-1] to be a one-output nn.Conv2d; got '
            f'{type(final_layer).__name__}.'
        )

    probability = (
        (alpha_init - ALPHANET_ALPHA_MIN)
        / (ALPHANET_ALPHA_MAX - ALPHANET_ALPHA_MIN)
    )
    raw_bias = math.log(probability / (1.0 - probability))
    with torch.no_grad():
        final_layer.weight.zero_()
        if final_layer.bias is None:
            raise TypeError('AlphaNet final convolution must have a bias.')
        final_layer.bias.fill_(raw_bias)

    return {
        'sigma_init': sigma_init,
        'alpha_init': alpha_init,
        'sigmoid_probability': probability,
        'raw_bias': raw_bias,
    }


# ============================================================
# Lower-level math helpers (verified correct)
# ============================================================

def forward_A(x, kernel_base, scale):
    C = x.shape[1]
    pad = kernel_base.shape[-1] // 2

    kernel = kernel_base.repeat(C, 1, 1, 1).to(
        device=x.device,
        dtype=x.dtype
    )

    x_pad = F.pad(
        x,
        (pad, pad, pad, pad),
        mode='circular'
    )

    blurred = F.conv2d(
        x_pad,
        kernel,
        padding=0,
        groups=C
    )

    return F.avg_pool2d(
        blurred,
        kernel_size=scale,
        stride=scale
    )

def adjoint_AT(y, kernel_base, scale):
    C = y.shape[1]
    pad = kernel_base.shape[-1] // 2

    kernel = kernel_base.repeat(C, 1, 1, 1).to(
        device=y.device,
        dtype=y.dtype
    )

    up = F.interpolate(
        y,
        scale_factor=scale,
        mode='nearest'
    ) / (scale ** 2)

    up_pad = F.pad(
        up,
        (pad, pad, pad, pad),
        mode='circular'
    )

    return F.conv2d(
        up_pad,
        kernel,
        padding=0,
        groups=C
    )

def psf2otf(psf, shape):
    kH, kW = psf.shape[2:]
    psf_padded = F.pad(psf, (0, shape[1] - kW, 0, shape[0] - kH))
    psf_padded = torch.roll(psf_padded, shifts=(-(kH // 2), -(kW // 2)), dims=(2, 3))
    return torch.fft.fft2(psf_padded)


def get_effective_otf(kernel_base, scale, img_shape):
    B, C, H, W = img_shape
    otf_H = psf2otf(kernel_base, (H, W))
    box_psf = torch.zeros(1, 1, H, W, device=kernel_base.device, dtype=kernel_base.dtype)
    for i in range(scale):
        for j in range(scale):
            box_psf[0, 0, -i % H, -j % W] = 1.0 / (scale ** 2)
    return otf_H * torch.fft.fft2(box_psf)


def fft_solve_balanced(otf, scale, mu, rho, rhs):
    """Closed-form x-update via frequency folding. Solves
    (H^T D^T D H + (mu+rho) I) x = rhs."""
    B, C, H, W = rhs.shape
    lam = mu + rho
    V = torch.fft.fft2(rhs)
    F_bar = torch.conj(otf)
    h, w = H // scale, W // scale
    num_fold = (otf * V).reshape(B, C, scale, h, scale, w).mean(dim=(2, 4))
    den_fold = (torch.abs(otf) ** 2).reshape(1, 1, scale, h, scale, w).mean(dim=(2, 4))
    inv_fold = num_fold / (den_fold + lam)
    inv_up = (inv_fold.unsqueeze(2).unsqueeze(4)
              .expand(B, C, scale, h, scale, w).reshape(B, C, H, W))
    X = (V - F_bar * inv_up) / lam
    return torch.fft.ifft2(X).real


# ============================================================
# Implicit gradient at the CURRENT iterate (truncated Neumann, eta=1)
# ============================================================


def crop_valid_region(x, valid_pad):
    """Return the non-padded HR region. valid_pad is measured in HR pixels."""
    if valid_pad <= 0:
        return x
    return x[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]


def make_valid_upper_grad(x_l, x_gt, valid_pad):
    """Gradient of raw J = 1/2||x-x_gt||^2 on the valid center region.

    The padded reflect margin is used only as a boundary buffer. It must not
    contribute to the bilevel gradient; otherwise the optimizer learns from
    artificial boundary pixels.
    """
    if valid_pad <= 0:
        return x_l - x_gt
    g_x = torch.zeros_like(x_l)
    g_x[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad] = (
        x_l[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]
        - x_gt[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]
    )
    return g_x


def valid_upper_loss_value(x_l, x_gt, valid_pad):
    """Raw MSE objective (half squared L2 sum) on the valid center region."""
    xv = crop_valid_region(x_l, valid_pad)
    gv = crop_valid_region(x_gt, valid_pad)
    return 0.5 * ((xv - gv) ** 2).sum().item()


def _validate_moreau_epsilon(epsilon):
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f'Moreau epsilon must be positive and finite, got {epsilon}')
    return epsilon


def moreau_mse_prox(x_l, x_gt, valid_pad, epsilon):
    """Exact prox_{epsilon J}(x_l) for valid-region half-squared MSE.

    The loss is independent of the reflect-padding margin, so the proximal
    point equals ``x_l`` there. This function is primarily a mathematical and
    self-test primitive; training uses the equivalent closed-form gradient.
    """
    epsilon = _validate_moreau_epsilon(epsilon)
    if x_l.shape != x_gt.shape:
        raise ValueError(
            f'Moreau prox tensors must have the same shape, got '
            f'{tuple(x_l.shape)} and {tuple(x_gt.shape)}'
        )
    prox = x_l.clone()
    if valid_pad <= 0:
        return (x_l + epsilon * x_gt) / (1.0 + epsilon)
    prox[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad] = (
        x_l[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]
        + epsilon * x_gt[:, :, valid_pad:-valid_pad, valid_pad:-valid_pad]
    ) / (1.0 + epsilon)
    return prox


def make_valid_moreau_upper_grad(x_l, x_gt, valid_pad, epsilon):
    """Exact gradient of the Moreau envelope of valid-region MSE."""
    epsilon = _validate_moreau_epsilon(epsilon)
    return make_valid_upper_grad(x_l, x_gt, valid_pad) / (1.0 + epsilon)


def valid_moreau_upper_loss_value(x_l, x_gt, valid_pad, epsilon):
    """Exact Moreau-envelope value J_epsilon=J/(1+epsilon)."""
    epsilon = _validate_moreau_epsilon(epsilon)
    return valid_upper_loss_value(x_l, x_gt, valid_pad) / (1.0 + epsilon)


def benchmark_psnr_y(x_l, x_gt, valid_pad=0, border=4):
    """Mean MATLAB-style luminance PSNR for observational training logs.

    The reflect margin is removed first.  A scale-width benchmark border is
    then shaved, matching the Set5/Set14 PSNR-Y convention.  This function is
    diagnostic only and never participates in autograd.
    """
    with torch.no_grad():
        pred = crop_valid_region(x_l.detach(), valid_pad).clamp(0.0, 1.0)
        target = crop_valid_region(x_gt.detach(), valid_pad).clamp(0.0, 1.0)
        if pred.shape != target.shape:
            raise ValueError(
                f'PSNR tensors must have identical shape, got '
                f'{tuple(pred.shape)} and {tuple(target.shape)}'
            )

        if pred.shape[1] == 3:
            coefficients = pred.new_tensor([65.481, 128.553, 24.966])
            pred_y = (
                16.0 + (pred * coefficients.view(1, 3, 1, 1)).sum(dim=1)
            ) / 255.0
            target_y = (
                16.0 + (target * coefficients.view(1, 3, 1, 1)).sum(dim=1)
            ) / 255.0
        elif pred.shape[1] == 1:
            pred_y = pred[:, 0]
            target_y = target[:, 0]
        else:
            raise ValueError(
                'benchmark_psnr_y expects one or three channels, got '
                f'{pred.shape[1]}'
            )

        border = int(border)
        if border < 0:
            raise ValueError(f'PSNR border must be non-negative, got {border}')
        if border > 0:
            if pred_y.shape[-2] <= 2 * border or pred_y.shape[-1] <= 2 * border:
                raise ValueError(
                    f'PSNR image is too small for border={border}: '
                    f'{tuple(pred_y.shape[-2:])}'
                )
            pred_y = pred_y[:, border:-border, border:-border]
            target_y = target_y[:, border:-border, border:-border]

        mse = (pred_y - target_y).square().mean(dim=(-2, -1))
        tiny = torch.finfo(mse.dtype).tiny
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(tiny))
        return float(psnr.mean())


def _tensor_list_l2_norm(tensors):
    """Euclidean norm over a list/tuple of tensors, ignoring None entries."""
    squared = [torch.sum(t.detach() ** 2) for t in tensors if t is not None]
    if not squared:
        return 0.0
    return math.sqrt(float(torch.stack(squared).sum()))


def _parameter_grad_l2_norm(params):
    """Euclidean norm of the gradients currently stored in parameters."""
    return _tensor_list_l2_norm([p.grad for p in params])


def _parameter_grads_finite(params):
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in params
    )


def _state_tuple_l2_norm(tensors):
    """Device-resident Euclidean norm for an ADMM-state tensor tuple."""
    return torch.sqrt(torch.stack([
        torch.sum(t.detach() ** 2) for t in tensors
    ]).sum())


def _constrain_weight_logits(net_W, theta):
    """Map unconstrained WeightNet logits to admissible Chebyshev weights."""
    n = net_W.n_nodes
    return (
        net_W.w_min
        + (n - n * net_W.w_min)
        * F.softmax(theta, dim=1)
    )


def _weight_map(net_W, y):
    """Return the constrained spatial Chebyshev-weight map produced by net_W."""
    return _constrain_weight_logits(net_W, net_W.weight_net(y))


def x_only_equilibrium_components(
    x,
    y,
    U_w_y,
    alpha_map,
    denoiser,
    mu,
    rho,
    forward_op,
    adjoint_op,
):
    """Return u(x,w), T(x;w,alpha), and G(x,w,alpha)=x-T.

    This is the single shared definition used by training and the self-test.
    It eliminates z and u from the equilibrium equations, but it does not
    change the real three-state PnP-ADMM forward iteration.
    """
    if rho <= 0:
        raise ValueError(f'rho must be positive, got {rho}')
    data_gradient = adjoint_op(forward_op(x) - y)
    u_from_x = -(data_gradient + mu * (x - U_w_y)) / rho
    fixed_point = denoiser(x + u_from_x, alpha_map / rho)
    root = x - fixed_point
    return u_from_x, fixed_point, root


def x_only_vjp(fixed_point, x, vector):
    """Compute J_T(x)^T vector without constructing the Jacobian."""
    value = torch.autograd.grad(
        fixed_point,
        x,
        grad_outputs=vector,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if value is None:
        value = torch.zeros_like(x)
    return value.detach()


def _gmres_x_only(operator, rhs, tolerance, max_iters, restart):
    """Restarted matrix-free GMRES for a tensor-valued linear system."""
    if tolerance <= 0 or max_iters <= 0 or restart <= 0:
        raise ValueError('GMRES tolerance, max_iters, and restart must be positive')
    restart = min(int(restart), int(max_iters))
    rhs_flat = rhs.detach().reshape(-1)
    rhs_norm = torch.linalg.vector_norm(rhs_flat)
    rhs_norm_value = float(rhs_norm)
    tiny = torch.finfo(rhs.dtype).eps

    def apply_flat(vector):
        return operator(vector.reshape_as(rhs)).reshape(-1).detach()

    solution = torch.zeros_like(rhs_flat)
    history = []
    total_iterations = 0
    converged = rhs_norm_value == 0.0
    if converged:
        return solution.reshape_as(rhs), {
            'iterations': 0,
            'converged': True,
            'residual_abs': 0.0,
            'residual_rel': 0.0,
            'history': [0.0],
        }

    while total_iterations < max_iters:
        residual = (rhs_flat - apply_flat(solution)).detach()
        beta = torch.linalg.vector_norm(residual)
        relative = float(beta) / (rhs_norm_value + tiny)
        history.append(relative)
        if not math.isfinite(relative):
            break
        if relative <= tolerance:
            converged = True
            break

        cycle_size = min(restart, max_iters - total_iterations)
        basis = [residual / beta]
        hessenberg = torch.zeros(
            cycle_size + 1,
            cycle_size,
            device=rhs.device,
            dtype=rhs.dtype,
        )
        candidate = solution
        breakdown = False

        for column in range(cycle_size):
            value = apply_flat(basis[column])

            # Two modified Gram-Schmidt passes reduce loss of orthogonality for
            # the non-normal adjoint operators encountered in PnP systems.
            for _ in range(2):
                for row in range(column + 1):
                    coefficient = torch.dot(basis[row], value)
                    hessenberg[row, column] += coefficient
                    value = value - coefficient * basis[row]

            next_norm = torch.linalg.vector_norm(value)
            hessenberg[column + 1, column] = next_norm
            if float(next_norm) > 10.0 * tiny:
                basis.append(value / next_norm)
            else:
                breakdown = True

            small_h = hessenberg[:column + 2, :column + 1]
            small_rhs = torch.zeros(
                column + 2,
                device=rhs.device,
                dtype=rhs.dtype,
            )
            small_rhs[0] = beta
            coefficients = torch.linalg.lstsq(
                small_h,
                small_rhs.unsqueeze(1),
            ).solution[:, 0]
            candidate = solution.clone()
            for index in range(column + 1):
                candidate = candidate + coefficients[index] * basis[index]

            estimate = torch.linalg.vector_norm(
                small_rhs - small_h @ coefficients
            )
            estimated_relative = float(estimate) / (rhs_norm_value + tiny)
            history.append(estimated_relative)
            total_iterations += 1
            if (
                not math.isfinite(estimated_relative)
                or estimated_relative <= tolerance
                or breakdown
                or total_iterations >= max_iters
            ):
                break

        solution = candidate.detach()
        actual = torch.linalg.vector_norm(rhs_flat - apply_flat(solution))
        actual_relative = float(actual) / (rhs_norm_value + tiny)
        history.append(actual_relative)
        if math.isfinite(actual_relative) and actual_relative <= tolerance:
            converged = True
            break
        if breakdown or not math.isfinite(actual_relative):
            break

    final_residual = torch.linalg.vector_norm(rhs_flat - apply_flat(solution))
    final_abs = float(final_residual)
    final_rel = final_abs / (rhs_norm_value + tiny)
    return solution.reshape_as(rhs).detach(), {
        'iterations': total_iterations,
        'converged': bool(converged and final_rel <= tolerance),
        'residual_abs': final_abs,
        'residual_rel': final_rel,
        'history': history,
    }


def _neumann_x_only(
    fixed_point,
    x,
    gradient_seed,
    min_iters,
    max_iters,
    tolerance,
):
    """Adaptive/fixed truncated Neumann solve of (I-J_T^T)p=g."""
    p = gradient_seed.detach().clone()
    gradient_norm = torch.linalg.vector_norm(gradient_seed)
    gradient_norm_value = float(gradient_norm)
    previous_term_norm = gradient_norm
    tiny = torch.finfo(x.dtype).eps
    term_norms = []
    term_ratios = []
    residual_history = []
    converged = False
    finite = True
    residual_abs = float('inf')
    residual_rel = float('inf')
    residual_to_last_term = float('nan')
    k_used = 0

    while True:
        jtp = x_only_vjp(fixed_point, x, p)
        residual = gradient_seed - p + jtp
        residual_norm = torch.linalg.vector_norm(residual)
        residual_abs = float(residual_norm)
        residual_rel = residual_abs / (gradient_norm_value + tiny)
        residual_history.append(residual_rel)
        finite = math.isfinite(residual_abs) and math.isfinite(residual_rel)
        if k_used > 0:
            residual_to_last_term = float(
                residual_norm / (previous_term_norm + tiny)
            )
        if k_used >= min_iters and finite and residual_rel <= tolerance:
            converged = True
        if converged or k_used >= max_iters or not finite:
            break

        p_new = (gradient_seed + jtp).detach()
        term = p_new - p
        term_norm = torch.linalg.vector_norm(term)
        term_norms.append(float(term_norm))
        term_ratios.append(float(term_norm / (previous_term_norm + tiny)))
        previous_term_norm = term_norm
        p = p_new
        k_used += 1

    diagnostics = {
        'iterations': k_used,
        'converged': converged,
        'finite': finite,
        'residual_abs': residual_abs,
        'residual_rel': residual_rel,
        'residual_history': residual_history,
        'last_term': term_norms[-1] if term_norms else 0.0,
        'last_ratio': term_ratios[-1] if term_ratios else 0.0,
        'max_ratio': max(term_ratios) if term_ratios else 0.0,
        'max_term_relative_to_seed': (
            max(term_norms) / (gradient_norm_value + tiny)
            if term_norms else 0.0
        ),
        'adjoint_relative_to_seed': float(
            torch.linalg.vector_norm(p) / (gradient_norm + tiny)
        ),
        'residual_to_last_term': residual_to_last_term,
    }
    return p.detach(), diagnostics


def implicit_grads_split_xonly(
    x_l,
    x_gt,
    y,
    net_W,
    net_Alpha,
    denoiser,
    kernel_base,
    mu,
    rho,
    scale,
    neumann_iters=8,
    adaptive_neumann=False,
    neumann_min_iters=4,
    neumann_max_iters=64,
    neumann_tol=1e-2,
    implicit_solver='neumann',
    gmres_tol=1e-4,
    gmres_max_iters=80,
    gmres_restart=20,
    neumann_catastrophic_limit=float('inf'),
    moreau_epsilon=DEFAULT_MOREAU_EPSILON,
    eta=1.0,
    valid_pad=0,
    forward_op=None,
    adjoint_op=None,
):
    """Return x-only implicit gradients of the Moreau-smoothed MSE.

    ``implicit_solver`` selects ``neumann`` (paper-style), ``gmres``
    (validation/reference), or ``neumann_gmres`` (GMRES only when Neumann fails).
    No direct WeightNet or AlphaNet penalty is present: MSE is the complete
    upper objective in this experiment.
    """
    if not math.isclose(float(eta), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f'x-only Neumann must be undamped (eta=1), got {eta}')
    if implicit_solver not in {'neumann', 'gmres', 'neumann_gmres'}:
        raise ValueError(f'unknown implicit_solver={implicit_solver!r}')
    if neumann_iters < 0 or neumann_min_iters < 0:
        raise ValueError('Neumann iteration counts must be non-negative')
    if neumann_max_iters < neumann_min_iters:
        raise ValueError('neumann_max_iters must be >= neumann_min_iters')
    if not math.isfinite(float(neumann_tol)) or neumann_tol <= 0:
        raise ValueError('neumann_tol must be positive and finite')
    if neumann_catastrophic_limit <= 1.0:
        raise ValueError(
            'neumann_catastrophic_limit must be greater than one, got '
            f'{neumann_catastrophic_limit}'
        )
    moreau_epsilon = _validate_moreau_epsilon(moreau_epsilon)
    if implicit_solver in {'gmres', 'neumann_gmres'}:
        if not math.isfinite(float(gmres_tol)) or gmres_tol <= 0:
            raise ValueError('gmres_tol must be positive and finite')
        if gmres_max_iters <= 0 or gmres_restart <= 0:
            raise ValueError(
                'gmres_max_iters and gmres_restart must be positive'
            )

    if forward_op is None:
        if kernel_base is None:
            raise ValueError('kernel_base is required when forward_op is absent')
        forward_op = lambda value: forward_A(value, kernel_base, scale)
    if adjoint_op is None:
        if kernel_base is None:
            raise ValueError('kernel_base is required when adjoint_op is absent')
        adjoint_op = lambda value: adjoint_AT(value, kernel_base, scale)

    x_in = x_l.detach().requires_grad_(True)
    U_w_y = net_W(y)
    alpha_map = net_Alpha(y)
    _, fixed_point, root = x_only_equilibrium_components(
        x_in,
        y,
        U_w_y,
        alpha_map,
        denoiser,
        mu,
        rho,
        forward_op,
        adjoint_op,
    )
    gradient_seed = make_valid_moreau_upper_grad(
        x_in.detach(),
        x_gt,
        valid_pad,
        moreau_epsilon,
    ).detach()

    def adjoint_operator(vector):
        # (I-J_T^T)v, the transpose Jacobian of G=x-T.
        return vector - x_only_vjp(fixed_point, x_in, vector)

    adaptive_min = neumann_min_iters if adaptive_neumann else neumann_iters
    adaptive_max = neumann_max_iters if adaptive_neumann else neumann_iters
    neumann_solution = None
    neumann_diagnostics = {
        'iterations': 0,
        'converged': False,
        'finite': True,
        'residual_abs': float('nan'),
        'residual_rel': float('nan'),
        'residual_history': [],
        'last_term': 0.0,
        'last_ratio': 0.0,
        'max_ratio': 0.0,
        'max_term_relative_to_seed': 0.0,
        'adjoint_relative_to_seed': 0.0,
        'residual_to_last_term': float('nan'),
    }
    if implicit_solver in {'neumann', 'neumann_gmres'}:
        neumann_solution, neumann_diagnostics = _neumann_x_only(
            fixed_point,
            x_in,
            gradient_seed,
            adaptive_min,
            adaptive_max,
            neumann_tol,
        )

    run_gmres = (
        implicit_solver == 'gmres'
        or (
            implicit_solver == 'neumann_gmres'
            and not neumann_diagnostics['converged']
        )
    )
    gmres_solution = None
    gmres_diagnostics = {
        'iterations': 0,
        'converged': False,
        'residual_abs': float('nan'),
        'residual_rel': float('nan'),
        'history': [],
    }
    if run_gmres:
        gmres_solution, gmres_diagnostics = _gmres_x_only(
            adjoint_operator,
            gradient_seed,
            gmres_tol,
            gmres_max_iters,
            gmres_restart,
        )
        if not gmres_diagnostics['converged']:
            raise RuntimeError(
                'x-only GMRES failed to meet its requested residual tolerance: '
                f"relative residual={gmres_diagnostics['residual_rel']:.6e}, "
                f'tolerance={gmres_tol:.6e}, '
                f"iterations={gmres_diagnostics['iterations']}"
            )

    if run_gmres:
        adjoint = gmres_solution
        selected_method = (
            'gmres' if implicit_solver == 'gmres' else 'gmres_fallback'
        )
        selected_diagnostics = gmres_diagnostics
    else:
        adjoint = neumann_solution
        selected_method = 'adaptive_neumann' if adaptive_neumann else 'fixed_neumann'
        selected_diagnostics = {
            'iterations': neumann_diagnostics['iterations'],
            'converged': neumann_diagnostics['converged'],
            'residual_abs': neumann_diagnostics['residual_abs'],
            'residual_rel': neumann_diagnostics['residual_rel'],
        }

    if (
        selected_method in {'adaptive_neumann', 'fixed_neumann'}
        and not neumann_diagnostics['finite']
    ):
        raise FloatingPointError(
            'x-only Neumann produced a non-finite adjoint residual; refusing '
            'to apply this upper-level update.'
        )
    if selected_method in {'adaptive_neumann', 'fixed_neumann'}:
        catastrophic_measure = max(
            float(neumann_diagnostics['max_ratio']),
            float(neumann_diagnostics['max_term_relative_to_seed']),
            float(neumann_diagnostics['adjoint_relative_to_seed']),
        )
        if (
            not math.isfinite(catastrophic_measure)
            or catastrophic_measure > float(neumann_catastrophic_limit)
        ):
            raise FloatingPointError(
                'x-only fixed Neumann powers grew catastrophically: '
                f'measure={catastrophic_measure:.6e}, '
                f'limit={float(neumann_catastrophic_limit):.6e}. '
                'The optimizer step was not reached.'
            )
    if adjoint is None or not bool(torch.isfinite(adjoint).all()):
        raise FloatingPointError('x-only implicit adjoint contains non-finite values')

    w_params = [parameter for parameter in net_W.parameters() if parameter.requires_grad]
    a_params = [parameter for parameter in net_Alpha.parameters() if parameter.requires_grad]
    all_grads = torch.autograd.grad(
        fixed_point,
        w_params + a_params,
        grad_outputs=adjoint,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    n_w = len(w_params)
    w_grads, a_grads = all_grads[:n_w], all_grads[n_w:]
    if not all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in all_grads
    ):
        raise FloatingPointError(
            'x-only implicit parameter gradient contains non-finite values; '
            'refusing to apply this upper-level update.'
        )

    tiny = torch.finfo(x_in.dtype).eps
    root_abs = float(torch.linalg.vector_norm(root.detach()))
    root_rel = root_abs / (float(torch.linalg.vector_norm(x_in.detach())) + tiny)
    diagnostics = {
        'implicit_state': 'x_only',
        'moreau_epsilon': moreau_epsilon,
        'solver_requested': implicit_solver,
        'solver_method': selected_method,
        'solver_iterations': int(selected_diagnostics['iterations']),
        'solver_converged': bool(selected_diagnostics['converged']),
        'solver_residual_abs': float(selected_diagnostics['residual_abs']),
        'solver_residual_rel': float(selected_diagnostics['residual_rel']),
        'x_root_abs': root_abs,
        'x_root_rel': root_rel,
        'neumann_last_term': neumann_diagnostics['last_term'],
        'neumann_last_ratio': neumann_diagnostics['last_ratio'],
        'neumann_max_ratio': neumann_diagnostics['max_ratio'],
        'neumann_max_term_relative_to_seed': (
            neumann_diagnostics['max_term_relative_to_seed']
        ),
        'neumann_adjoint_relative_to_seed': (
            neumann_diagnostics['adjoint_relative_to_seed']
        ),
        'neumann_catastrophic_limit': float(neumann_catastrophic_limit),
        'neumann_k_used': int(neumann_diagnostics['iterations']),
        'neumann_adjoint_residual_abs': neumann_diagnostics['residual_abs'],
        'neumann_adjoint_residual_rel': neumann_diagnostics['residual_rel'],
        'adjoint_residual_abs': float(selected_diagnostics['residual_abs']),
        'adjoint_residual_rel': float(selected_diagnostics['residual_rel']),
        'adjoint_residual_to_last_term': neumann_diagnostics['residual_to_last_term'],
        'neumann_converged': bool(neumann_diagnostics['converged']),
        'neumann_residual_finite': bool(neumann_diagnostics['finite']),
        'neumann_tol': float(neumann_tol),
        'neumann_mode': (
            'adaptive' if adaptive_neumann else 'fixed'
        ) if implicit_solver != 'gmres' else 'skipped',
        'gmres_used': bool(run_gmres),
        'gmres_iterations': int(gmres_diagnostics['iterations']),
        'gmres_converged': bool(gmres_diagnostics['converged']),
        'gmres_residual_rel': float(gmres_diagnostics['residual_rel']),
    }
    return (w_params, w_grads), (a_params, a_grads), diagnostics


def _new_neumann_counter():
    """Accumulator for fixed x-only Neumann training diagnostics."""
    return {
        'total': 0,
        'converged': 0,
        'neumann_skipped': 0,
        'nonfinite': 0,
        'solver_converged': 0,
        'solver_nonfinite': 0,
        'gmres_used': 0,
        'gmres_converged': 0,
        'gmres_iters_sum': 0,
        'k_sum': 0,
        'k_min': None,
        'k_max': 0,
        'residual_sum': 0.0,
        'residual_max': 0.0,
        'solver_residual_sum': 0.0,
        'solver_residual_max': 0.0,
        'root_rel_sum': 0.0,
        'root_rel_max': 0.0,
    }


def _update_neumann_counter(counter, diagnostics):
    """Add one implicit-state evaluation to a convergence counter."""
    k_used = int(diagnostics['neumann_k_used'])
    residual = float(diagnostics['neumann_adjoint_residual_rel'])
    finite = bool(diagnostics['neumann_residual_finite']) and math.isfinite(residual)
    skipped = diagnostics['neumann_mode'] == 'skipped'
    solver_residual = float(diagnostics['solver_residual_rel'])
    solver_finite = math.isfinite(solver_residual)
    root_rel = float(diagnostics['x_root_rel'])

    counter['total'] += 1
    counter['converged'] += int(bool(diagnostics['neumann_converged']))
    counter['neumann_skipped'] += int(skipped)
    counter['nonfinite'] += int(not skipped and not finite)
    counter['solver_converged'] += int(bool(diagnostics['solver_converged']))
    counter['solver_nonfinite'] += int(not solver_finite)
    counter['gmres_used'] += int(bool(diagnostics['gmres_used']))
    counter['gmres_converged'] += int(bool(diagnostics['gmres_converged']))
    counter['gmres_iters_sum'] += int(diagnostics['gmres_iterations'])
    counter['k_sum'] += k_used
    if not skipped:
        counter['k_min'] = (
            k_used if counter['k_min'] is None else min(counter['k_min'], k_used)
        )
        counter['k_max'] = max(counter['k_max'], k_used)
    if finite:
        counter['residual_sum'] += residual
        counter['residual_max'] = max(counter['residual_max'], residual)
    if solver_finite:
        counter['solver_residual_sum'] += solver_residual
        counter['solver_residual_max'] = max(
            counter['solver_residual_max'], solver_residual
        )
    if math.isfinite(root_rel):
        counter['root_rel_sum'] += root_rel
        counter['root_rel_max'] = max(counter['root_rel_max'], root_rel)


def _merge_neumann_counter(destination, source):
    """Merge a batch/epoch counter into a longer-lived counter."""
    if source['total'] == 0:
        return
    destination['total'] += source['total']
    destination['converged'] += source['converged']
    for key in (
        'neumann_skipped',
        'nonfinite',
        'solver_converged',
        'solver_nonfinite',
        'gmres_used',
        'gmres_converged',
        'gmres_iters_sum',
    ):
        destination[key] += source[key]
    destination['k_sum'] += source['k_sum']
    if source['k_min'] is not None:
        destination['k_min'] = (
            source['k_min']
            if destination['k_min'] is None
            else min(destination['k_min'], source['k_min'])
        )
    destination['k_max'] = max(destination['k_max'], source['k_max'])
    destination['residual_sum'] += source['residual_sum']
    destination['residual_max'] = max(
        destination['residual_max'], source['residual_max']
    )
    destination['solver_residual_sum'] += source['solver_residual_sum']
    destination['solver_residual_max'] = max(
        destination['solver_residual_max'], source['solver_residual_max']
    )
    destination['root_rel_sum'] += source['root_rel_sum']
    destination['root_rel_max'] = max(
        destination['root_rel_max'], source['root_rel_max']
    )


def _neumann_counter_values(counter):
    """Return safe rate/mean values for one convergence counter."""
    total = counter['total']
    attempted = total - counter['neumann_skipped']
    finite_total = attempted - counter['nonfinite']
    solver_finite_total = total - counter['solver_nonfinite']
    return {
        'rate': counter['converged'] / max(attempted, 1),
        'solver_rate': counter['solver_converged'] / max(total, 1),
        'mean_k': counter['k_sum'] / max(attempted, 1),
        'mean_residual': counter['residual_sum'] / max(finite_total, 1),
        'mean_solver_residual': (
            counter['solver_residual_sum'] / max(solver_finite_total, 1)
        ),
        'mean_root_rel': counter['root_rel_sum'] / max(total, 1),
        'mean_gmres_iters': (
            counter['gmres_iters_sum'] / max(counter['gmres_used'], 1)
        ),
    }


def _new_neumann_summary():
    """Separate network counters plus unique implicit-state evaluations."""
    return {
        'w': _new_neumann_counter(),
        'alpha': _new_neumann_counter(),
        'states': _new_neumann_counter(),
    }


def _merge_neumann_summary(destination, source):
    for key in ('w', 'alpha', 'states'):
        _merge_neumann_counter(destination[key], source[key])


def _log_neumann_summary(logger, scope, summary, neumann_tol, **indices):
    """Write fixed-K residual/root statistics; tolerance is diagnostic only."""
    prefix = ' '.join(f'{key}={value}' for key, value in indices.items())
    if prefix:
        prefix += ' '

    state = summary['states']
    w_counter = summary['w']
    alpha_counter = summary['alpha']
    state_values = _neumann_counter_values(state)
    logger.info(
        '%sxonly_implicit_%s_summary '
        'tol=%.6e '
        'states_neumann_converged=%d states_neumann_attempted=%d '
        'states_neumann_skipped=%d states_total=%d states_neumann_rate=%.6f '
        'states_solver_converged=%d states_solver_rate=%.6f '
        'w_converged=%d w_total=%d alpha_converged=%d alpha_total=%d '
        'k_min=%d k_mean=%.3f k_max=%d '
        'neumann_rel_mean=%.6e neumann_rel_max=%.6e nonfinite=%d '
        'solver_rel_mean=%.6e solver_rel_max=%.6e '
        'x_root_rel_mean=%.6e x_root_rel_max=%.6e',
        prefix,
        scope,
        neumann_tol,
        state['converged'],
        state['total'] - state['neumann_skipped'],
        state['neumann_skipped'],
        state['total'],
        state_values['rate'],
        state['solver_converged'],
        state_values['solver_rate'],
        w_counter['converged'],
        w_counter['total'],
        alpha_counter['converged'],
        alpha_counter['total'],
        state['k_min'] if state['k_min'] is not None else 0,
        state_values['mean_k'],
        state['k_max'],
        state_values['mean_residual'],
        state['residual_max'],
        state['nonfinite'],
        state_values['mean_solver_residual'],
        state['solver_residual_max'],
        state_values['mean_root_rel'],
        state['root_rel_max'],
    )


def _assign_grads(params, grads):
    for p, g in zip(params, grads):
        p.grad = g if g is not None else None


# ============================================================
# Project-independent x-only mathematical self-test
# ============================================================

def _relative_tensor_error(actual, reference):
    """Return ||actual-reference|| / (||reference|| + machine epsilon)."""
    tiny = torch.finfo(reference.dtype).eps
    relative_error = (
        torch.linalg.vector_norm(actual - reference)
        / (torch.linalg.vector_norm(reference) + tiny)
    )
    return relative_error.detach().item()


def run_self_test():
    """Validate the x-only root, VJPs, solvers, fallback, and IFT sign.

    This test needs PyTorch but not the BL-LCILW project modules or checkpoints.
    It uses a small affine equilibrium whose exact root and Jacobian are known,
    then compares the implemented implicit gradients with centered finite
    differences of the solved equilibrium loss.
    """
    print('Running project-independent x-only implicit-gradient self-test...')
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dtype = torch.float64
        device = torch.device('cpu')
        test_moreau_epsilon = 0.25

        # Verify the closed-form Moreau prox, value, gradient, and adaptive gate.
        moreau_x = torch.tensor(
            [[[[0.2, -0.1, 0.4, 0.3],
               [0.0, 0.5, -0.2, 0.1],
               [0.7, -0.4, 0.2, 0.6],
               [0.1, 0.3, -0.5, 0.8]]]],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        moreau_target = torch.zeros_like(moreau_x)
        moreau_prox = moreau_mse_prox(
            moreau_x,
            moreau_target,
            valid_pad=1,
            epsilon=test_moreau_epsilon,
        )
        prox_objective = (
            0.5 * torch.sum(crop_valid_region(
                moreau_prox - moreau_target, 1
            ) ** 2)
            + 0.5 / test_moreau_epsilon
            * torch.sum((moreau_x - moreau_prox) ** 2)
        )
        closed_envelope = (
            0.5 / (1.0 + test_moreau_epsilon)
            * torch.sum(crop_valid_region(
                moreau_x - moreau_target, 1
            ) ** 2)
        )
        autograd_moreau_grad = torch.autograd.grad(
            closed_envelope,
            moreau_x,
        )[0]
        exact_moreau_grad = make_valid_moreau_upper_grad(
            moreau_x.detach(),
            moreau_target,
            valid_pad=1,
            epsilon=test_moreau_epsilon,
        )
        moreau_value_error = abs(
            float(prox_objective.detach() - closed_envelope.detach())
        )
        moreau_grad_error = _relative_tensor_error(
            autograd_moreau_grad,
            exact_moreau_grad,
        )
        if moreau_value_error > 1e-14 or moreau_grad_error > 1e-14:
            raise AssertionError(
                'closed-form Moreau MSE test failed: '
                f'value_error={moreau_value_error:.6e}, '
                f'gradient_error={moreau_grad_error:.6e}'
            )

        gate_test = AdaptiveMoreauState(
            epsilon=0.1,
            gamma=0.8,
            sigma=0.5,
            epsilon_tol=1e-4,
            parameter_count=4,
            norm_mode='rms',
        )
        no_decay = gate_test.update(raw_l2_norm=0.2)
        decay = gate_test.update(raw_l2_norm=0.01)
        if no_decay['triggered'] or not decay['triggered']:
            raise AssertionError('adaptive Moreau gradient-norm gate test failed')
        if not math.isclose(gate_test.epsilon, 0.08, rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError('adaptive Moreau geometric decay test failed')

        # Validate the explicit paper alpha_0 initialization independently of
        # the project AlphaNet import.
        class ToyBoundedAlphaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(3, 1, kernel_size=1))

            def forward(self, value):
                raw = self.net(value)
                return (
                    ALPHANET_ALPHA_MIN
                    + (ALPHANET_ALPHA_MAX - ALPHANET_ALPHA_MIN)
                    * torch.sigmoid(raw)
                )

        bounded_alpha = ToyBoundedAlphaNet().to(device=device, dtype=dtype)
        alpha_init_meta = initialize_alphanet_constant_sigma(
            bounded_alpha,
            rho=0.05,
            sigma_init=DEFAULT_SIGMA_INIT,
        )
        alpha_probe = bounded_alpha(torch.randn(
            2, 3, 4, 5, dtype=dtype, device=device
        ))
        alpha_init_error = abs(
            alpha_probe.detach().mean().item() / 0.05 - DEFAULT_SIGMA_INIT
        ) / DEFAULT_SIGMA_INIT
        if alpha_init_error > 1e-12:
            raise AssertionError(
                'AlphaNet constant-sigma initialization failed: '
                f'relative error={alpha_init_error:.6e}, '
                f'metadata={alpha_init_meta}'
            )

        # A deliberately expansive map must be classified as catastrophic at
        # fixed K=8, while an ordinary residual above 1e-2 remains allowable.
        expansive_x = torch.ones(1, dtype=dtype, device=device).requires_grad_(True)
        _, expansive_diag = _neumann_x_only(
            fixed_point=2.0 * expansive_x,
            x=expansive_x,
            gradient_seed=torch.ones_like(expansive_x),
            min_iters=PAPER_NEUMANN_K,
            max_iters=PAPER_NEUMANN_K,
            tolerance=DEFAULT_NEUMANN_DIAGNOSTIC_TOL,
        )
        expansive_measure = max(
            expansive_diag['max_ratio'],
            expansive_diag['max_term_relative_to_seed'],
            expansive_diag['adjoint_relative_to_seed'],
        )
        if expansive_measure <= DEFAULT_NEUMANN_CATASTROPHIC_LIMIT:
            raise AssertionError(
                'fixed-K catastrophic-growth guard self-test failed: '
                f'measure={expansive_measure:.6e}'
            )

        # First validate restarted GMRES independently on a nonsymmetric system.
        linear_matrix = torch.tensor(
            [
                [4.0, 1.0, -0.2],
                [0.3, 3.0, 0.5],
                [0.0, -0.4, 2.0],
            ],
            dtype=dtype,
            device=device,
        )
        linear_rhs = torch.tensor(
            [1.0, -2.0, 0.5],
            dtype=dtype,
            device=device,
        )
        gmres_solution, gmres_linear_diag = _gmres_x_only(
            lambda value: linear_matrix @ value,
            linear_rhs,
            tolerance=1e-12,
            max_iters=20,
            restart=3,
        )
        linear_exact = torch.linalg.solve(linear_matrix, linear_rhs)
        gmres_linear_error = _relative_tensor_error(
            gmres_solution,
            linear_exact,
        )
        if not gmres_linear_diag['converged'] or gmres_linear_error > 1e-10:
            raise AssertionError(
                'GMRES linear-system test failed: '
                f'converged={gmres_linear_diag["converged"]}, '
                f'relative error={gmres_linear_error:.6e}'
            )

        forward_matrix = torch.tensor(
            [
                [0.20, 0.03, 0.00],
                [0.00, 0.15, 0.02],
                [0.01, 0.00, 0.18],
            ],
            dtype=dtype,
            device=device,
        )
        denoiser_matrix = torch.tensor(
            [
                [0.30, 0.05, 0.00],
                [0.00, 0.25, 0.04],
                [0.02, 0.00, 0.20],
            ],
            dtype=dtype,
            device=device,
        )
        sigma_matrix = torch.tensor(
            [
                [0.05, 0.01, 0.00],
                [0.00, 0.04, 0.01],
                [0.01, 0.00, 0.03],
            ],
            dtype=dtype,
            device=device,
        )
        denoiser_bias = torch.tensor(
            [0.03, -0.02, 0.01],
            dtype=dtype,
            device=device,
        )
        upsample_base = torch.tensor(
            [0.25, -0.10, 0.30],
            dtype=dtype,
            device=device,
        )
        upsample_parameter_matrix = torch.tensor(
            [
                [0.20, -0.05],
                [0.04, 0.12],
                [-0.08, 0.10],
            ],
            dtype=dtype,
            device=device,
        )
        alpha_base = torch.tensor(
            [0.04, 0.05, 0.03],
            dtype=dtype,
            device=device,
        )
        alpha_parameter_matrix = torch.tensor(
            [
                [0.03, 0.01],
                [0.01, -0.02],
                [0.02, 0.04],
            ],
            dtype=dtype,
            device=device,
        )
        observation = torch.tensor(
            [0.12, -0.07, 0.20],
            dtype=dtype,
            device=device,
        )
        target = torch.tensor(
            [0.18, -0.04, 0.26],
            dtype=dtype,
            device=device,
        )
        mu = 0.08
        rho = 1.0

        class ToyWeightNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = nn.Parameter(torch.tensor(
                    [0.12, -0.08],
                    dtype=dtype,
                    device=device,
                ))

            def forward(self, _observation):
                return (
                    upsample_base
                    + upsample_parameter_matrix @ self.theta
                )

        class ToyAlphaNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = nn.Parameter(torch.tensor(
                    [0.02, 0.03],
                    dtype=dtype,
                    device=device,
                ))

            def forward(self, _observation):
                return alpha_base + alpha_parameter_matrix @ self.theta

        class ToyDenoiser(nn.Module):
            def forward(self, value, sigma):
                return (
                    denoiser_matrix @ value
                    + sigma_matrix @ sigma
                    + denoiser_bias
                )

        net_w = ToyWeightNet()
        net_alpha = ToyAlphaNet()
        denoiser = ToyDenoiser()
        forward_op = lambda value: forward_matrix @ value
        adjoint_op = lambda value: forward_matrix.T @ value

        identity = torch.eye(3, dtype=dtype, device=device)
        x_input_matrix = identity - (
            forward_matrix.T @ forward_matrix + mu * identity
        ) / rho
        exact_jacobian_t = denoiser_matrix @ x_input_matrix

        with torch.no_grad():
            upsampled = net_w(observation)
            alpha_map = net_alpha(observation)
            zero = torch.zeros(3, dtype=dtype, device=device)
            _, affine_offset, _ = x_only_equilibrium_components(
                zero,
                observation,
                upsampled,
                alpha_map,
                denoiser,
                mu,
                rho,
                forward_op,
                adjoint_op,
            )
            x_star = torch.linalg.solve(
                identity - exact_jacobian_t,
                affine_offset,
            )

        # Check the shared u/T/G definition and the J_T^T v implementation.
        x_probe = x_star.detach().requires_grad_(True)
        upsampled = net_w(observation)
        alpha_map = net_alpha(observation)
        u_from_x, fixed_point, root = x_only_equilibrium_components(
            x_probe,
            observation,
            upsampled,
            alpha_map,
            denoiser,
            mu,
            rho,
            forward_op,
            adjoint_op,
        )
        manual_u = -(
            forward_matrix.T @ (forward_matrix @ x_probe - observation)
            + mu * (x_probe - upsampled)
        ) / rho
        manual_t = denoiser(x_probe + manual_u, alpha_map / rho)
        component_error = max(
            _relative_tensor_error(u_from_x, manual_u),
            _relative_tensor_error(fixed_point, manual_t),
            _relative_tensor_error(root, x_probe - manual_t),
        )
        root_relative = float(
            torch.linalg.vector_norm(root.detach())
            / (
                torch.linalg.vector_norm(x_probe.detach())
                + torch.finfo(dtype).eps
            )
        )
        test_vector = torch.tensor(
            [0.4, -0.3, 0.2],
            dtype=dtype,
            device=device,
        )
        vjp_actual = x_only_vjp(fixed_point, x_probe, test_vector)
        vjp_exact = exact_jacobian_t.T @ test_vector
        vjp_error = _relative_tensor_error(vjp_actual, vjp_exact)
        if component_error > 1e-12 or root_relative > 1e-12 or vjp_error > 1e-12:
            raise AssertionError(
                'x-only component/VJP test failed: '
                f'component={component_error:.6e}, '
                f'root={root_relative:.6e}, vjp={vjp_error:.6e}'
            )

        common_arguments = dict(
            x_l=x_star,
            x_gt=target,
            y=observation,
            net_W=net_w,
            net_Alpha=net_alpha,
            denoiser=denoiser,
            kernel_base=None,
            mu=mu,
            rho=rho,
            scale=1,
            neumann_iters=8,
            adaptive_neumann=True,
            neumann_min_iters=0,
            neumann_max_iters=80,
            neumann_tol=1e-12,
            gmres_tol=1e-12,
            gmres_max_iters=20,
            gmres_restart=3,
            moreau_epsilon=test_moreau_epsilon,
            valid_pad=0,
            forward_op=forward_op,
            adjoint_op=adjoint_op,
        )
        (w_params_n, w_grads_n), (a_params_n, a_grads_n), neumann_diag = (
            implicit_grads_split_xonly(
                **common_arguments,
                implicit_solver='neumann',
            )
        )
        diagnostic_only_arguments = dict(common_arguments)
        diagnostic_only_arguments.update(
            neumann_iters=0,
            adaptive_neumann=False,
            neumann_min_iters=0,
            neumann_max_iters=0,
            neumann_tol=1e-20,
        )
        (_, _), (_, _), diagnostic_only_diag = implicit_grads_split_xonly(
            **diagnostic_only_arguments,
            implicit_solver='neumann',
            neumann_catastrophic_limit=DEFAULT_NEUMANN_CATASTROPHIC_LIMIT,
        )
        if (
            diagnostic_only_diag['neumann_converged']
            or not diagnostic_only_diag['neumann_residual_finite']
        ):
            raise AssertionError(
                'fixed-Neumann residual must remain diagnostic-only for a '
                'finite, non-catastrophic truncation'
            )
        (_, w_grads_g), (_, a_grads_g), gmres_diag = (
            implicit_grads_split_xonly(
                **common_arguments,
                implicit_solver='gmres',
            )
        )
        fallback_arguments = dict(common_arguments)
        fallback_arguments.update(
            neumann_iters=0,
            neumann_min_iters=0,
            neumann_max_iters=0,
        )
        (_, w_grads_f), (_, a_grads_f), fallback_diag = (
            implicit_grads_split_xonly(
                **fallback_arguments,
                implicit_solver='neumann_gmres',
            )
        )

        if not neumann_diag['neumann_converged']:
            raise AssertionError(
                'adaptive Neumann did not converge in the affine self-test: '
                f"residual={neumann_diag['neumann_adjoint_residual_rel']:.6e}"
            )
        if not gmres_diag['gmres_converged']:
            raise AssertionError('GMRES reference did not converge in the self-test')
        if (
            fallback_diag['solver_method'] != 'gmres_fallback'
            or not fallback_diag['gmres_converged']
        ):
            raise AssertionError(
                'Neumann-to-GMRES fallback was not selected correctly'
            )

        def flatten_gradients(weight_gradients, alpha_gradients):
            if any(value is None for value in weight_gradients + alpha_gradients):
                raise AssertionError('toy parameters unexpectedly received None gradients')
            return torch.cat([
                *(value.reshape(-1) for value in weight_gradients),
                *(value.reshape(-1) for value in alpha_gradients),
            ])

        implicit_neumann = flatten_gradients(w_grads_n, a_grads_n)
        implicit_gmres = flatten_gradients(w_grads_g, a_grads_g)
        implicit_fallback = flatten_gradients(w_grads_f, a_grads_f)
        neumann_gmres_error = _relative_tensor_error(
            implicit_neumann,
            implicit_gmres,
        )
        fallback_gmres_error = _relative_tensor_error(
            implicit_fallback,
            implicit_gmres,
        )

        def equilibrium_loss(weight_values, alpha_values):
            upsampled_value = (
                upsample_base
                + upsample_parameter_matrix @ weight_values
            )
            alpha_value = alpha_base + alpha_parameter_matrix @ alpha_values
            zero_value = torch.zeros(3, dtype=dtype, device=device)
            _, offset_value, _ = x_only_equilibrium_components(
                zero_value,
                observation,
                upsampled_value,
                alpha_value,
                denoiser,
                mu,
                rho,
                forward_op,
                adjoint_op,
            )
            equilibrium = torch.linalg.solve(
                identity - exact_jacobian_t,
                offset_value,
            )
            return (
                0.5 / (1.0 + test_moreau_epsilon)
                * torch.sum((equilibrium - target) ** 2)
            )

        weight_values = w_params_n[0].detach().clone()
        alpha_values = a_params_n[0].detach().clone()
        finite_difference = torch.zeros(4, dtype=dtype, device=device)
        step = 1e-6
        for index in range(weight_values.numel()):
            direction = torch.zeros_like(weight_values)
            direction[index] = step
            plus = equilibrium_loss(weight_values + direction, alpha_values)
            minus = equilibrium_loss(weight_values - direction, alpha_values)
            finite_difference[index] = (plus - minus) / (2.0 * step)
        offset = weight_values.numel()
        for index in range(alpha_values.numel()):
            direction = torch.zeros_like(alpha_values)
            direction[index] = step
            plus = equilibrium_loss(weight_values, alpha_values + direction)
            minus = equilibrium_loss(weight_values, alpha_values - direction)
            finite_difference[offset + index] = (plus - minus) / (2.0 * step)

        finite_difference_error = _relative_tensor_error(
            implicit_gmres,
            finite_difference,
        )
        if (
            neumann_gmres_error > 1e-9
            or fallback_gmres_error > 1e-10
            or finite_difference_error > 1e-6
        ):
            raise AssertionError(
                'implicit-gradient validation failed: '
                f'Neumann-vs-GMRES={neumann_gmres_error:.6e}, '
                f'fallback-vs-GMRES={fallback_gmres_error:.6e}, '
                f'GMRES-vs-finite-difference={finite_difference_error:.6e}'
            )

        print(
            'Self-test passed | '
            f'GMRES linear error={gmres_linear_error:.3e} | '
            f'component error={component_error:.3e} | '
            f'VJP error={vjp_error:.3e} | '
            f'root residual={root_relative:.3e} | '
            f'alpha-init error={alpha_init_error:.3e} | '
            f'Moreau value error={moreau_value_error:.3e} | '
            f'Moreau grad error={moreau_grad_error:.3e} | '
            'adaptive-epsilon gate=PASS | '
            'catastrophic guard=PASS | '
            'residual gate=DIAGNOSTIC_ONLY | '
            f'Neumann K={neumann_diag["neumann_k_used"]} | '
            f'Neumann-vs-GMRES={neumann_gmres_error:.3e} | '
            f'IFT-vs-FD={finite_difference_error:.3e}'
        )
    finally:
        torch.set_default_dtype(original_dtype)

# ============================================================
# Algorithm 2 (Gauss--Seidel upper update by default), per batch
# ============================================================

def run_algorithm2_on_image(
    y, x_gt, net_W, net_Alpha, denoiser, opt_W, opt_Alpha,
    otf, ATy, kernel_base, mu, rho, scale, L_iterations, neumann_iters,
    moreau_state,
    neumann_tol=DEFAULT_NEUMANN_DIAGNOSTIC_TOL,
    neumann_catastrophic_limit=DEFAULT_NEUMANN_CATASTROPHIC_LIMIT,
    grad_clip_w=DEFAULT_GRAD_CLIP_W,
    grad_clip_alpha=DEFAULT_GRAD_CLIP_ALPHA,
    update_mode='sequential',
    valid_pad=0,
    logger=None, epoch_idx=0, batch_idx=0, log_every_outer=1,
):
    if log_every_outer <= 0:
        raise ValueError(f'log_every_outer must be positive, got {log_every_outer}')
    if grad_clip_w is not None and grad_clip_w <= 0:
        raise ValueError(
            f'grad_clip_w must be positive or None, got {grad_clip_w}'
        )
    if grad_clip_alpha is not None and grad_clip_alpha <= 0:
        raise ValueError(
            'grad_clip_alpha must be positive or None, got '
            f'{grad_clip_alpha}'
        )
    if update_mode not in {'simultaneous', 'sequential'}:
        raise ValueError(
            "update_mode must be 'simultaneous' or 'sequential', got "
            f'{update_mode!r}'
        )
    if not isinstance(moreau_state, AdaptiveMoreauState):
        raise TypeError('moreau_state must be an AdaptiveMoreauState instance')

    # line 1-2: initialise x^0 = U_{w^0} y, z^0 = x^0, u^0 = 0
    with torch.no_grad():
        x_l = net_W(y)
    z_l = x_l.clone()
    u_l = torch.zeros_like(x_l)
    batch_neumann_summary = _new_neumann_summary()

    for l in range(L_iterations):
        epsilon_l = moreau_state.epsilon

        # ---- lines 6-7: upper update ----
        # Algorithm 2 lines 6--7 are Gauss--Seidel: alpha^{l+1} is evaluated
        # with w^{l+1}. The simultaneous path is retained only as an ablation.
        (wP, wG), (aP, aG), w_implicit_diag = implicit_grads_split_xonly(
            x_l, x_gt, y, net_W, net_Alpha, denoiser,
            kernel_base, mu, rho, scale,
            neumann_iters=neumann_iters,
            adaptive_neumann=False,
            neumann_min_iters=neumann_iters,
            neumann_max_iters=neumann_iters,
            neumann_tol=neumann_tol,
            implicit_solver='neumann',
            neumann_catastrophic_limit=neumann_catastrophic_limit,
            moreau_epsilon=epsilon_l,
            valid_pad=valid_pad,
        )
        _update_neumann_counter(batch_neumann_summary['w'], w_implicit_diag)
        _update_neumann_counter(batch_neumann_summary['states'], w_implicit_diag)
        w_recon_grad_norm = _tensor_list_l2_norm(wG)

        opt_W.zero_grad(set_to_none=True)
        _assign_grads(wP, wG)
        w_preclip_grad_norm = _parameter_grad_l2_norm(wP)
        if not _parameter_grads_finite(wP) or not math.isfinite(w_preclip_grad_norm):
            raise FloatingPointError(
                'Combined WeightNet gradient is non-finite; optimizer step rejected.'
            )
        w_was_clipped = bool(
            grad_clip_w is not None and w_preclip_grad_norm > grad_clip_w
        )
        if grad_clip_w is not None:
            torch.nn.utils.clip_grad_norm_(wP, grad_clip_w)
        w_postclip_grad_norm = _parameter_grad_l2_norm(wP)

        if update_mode == 'sequential':
            opt_W.step()  # w^{l+1}, then recompute alpha at the new weight.
            (_, _), (aP, aG), alpha_implicit_diag = implicit_grads_split_xonly(
                x_l, x_gt, y, net_W, net_Alpha, denoiser,
                kernel_base, mu, rho, scale,
                neumann_iters=neumann_iters,
                adaptive_neumann=False,
                neumann_min_iters=neumann_iters,
                neumann_max_iters=neumann_iters,
                neumann_tol=neumann_tol,
                implicit_solver='neumann',
                neumann_catastrophic_limit=neumann_catastrophic_limit,
                moreau_epsilon=epsilon_l,
                valid_pad=valid_pad,
            )
            _update_neumann_counter(
                batch_neumann_summary['states'], alpha_implicit_diag
            )
        else:
            # Both gradients were evaluated at (x^l, w^l, alpha^l).
            alpha_implicit_diag = w_implicit_diag
            _update_neumann_counter(
                batch_neumann_summary['states'], alpha_implicit_diag
            )

        _update_neumann_counter(batch_neumann_summary['alpha'], alpha_implicit_diag)
        alpha_recon_grad_norm = _tensor_list_l2_norm(aG)
        opt_Alpha.zero_grad(set_to_none=True)
        _assign_grads(aP, aG)
        alpha_preclip_grad_norm = _parameter_grad_l2_norm(aP)
        if (
            not _parameter_grads_finite(aP)
            or not math.isfinite(alpha_preclip_grad_norm)
        ):
            raise FloatingPointError(
                'AlphaNet gradient is non-finite; optimizer step rejected.'
            )
        alpha_was_clipped = bool(
            grad_clip_alpha is not None
            and alpha_preclip_grad_norm > grad_clip_alpha
        )
        if grad_clip_alpha is not None:
            torch.nn.utils.clip_grad_norm_(aP, grad_clip_alpha)
        alpha_postclip_grad_norm = _parameter_grad_l2_norm(aP)

        if update_mode == 'simultaneous':
            opt_W.step()
        opt_Alpha.step()

        applied_implicit_grad_norm = math.sqrt(
            w_recon_grad_norm ** 2 + alpha_recon_grad_norm ** 2
        )

        should_log = (
            (l + 1) % log_every_outer == 0
            or l == 0
            or l + 1 == L_iterations
        )

        # ---- lines 9-11: one PnP-ADMM step with w^{l+1}, alpha^{l+1} ----
        x_prev = x_l
        z_prev = z_l
        u_prev = u_l

        # Capture the exact WeightNet logits used by the real net_W(y) forward.
        # A hook avoids an additional diagnostic forward, which could otherwise
        # alter BatchNorm-style running buffers in training mode.
        captured_weight_logits = {}
        weight_hook = None
        if logger is not None and should_log:
            def _capture_weight_logits(_module, _inputs, output):
                captured_weight_logits['theta'] = output.detach()

            weight_hook = net_W.weight_net.register_forward_hook(
                _capture_weight_logits
            )

        with torch.no_grad():
            try:
                U_w_y = net_W(y)
            finally:
                if weight_hook is not None:
                    weight_hook.remove()
            alpha_map = net_Alpha(y)
            rhs = ATy + mu * U_w_y + rho * (z_l - u_l)
            x_l = fft_solve_balanced(otf, scale, mu, rho, rhs)   # line 9
            z_l = denoiser(x_l + u_l, alpha_map / rho)           # line 10
            u_l = u_l + x_l - z_l                                # line 11

        # ---- lines 12-19: evaluate stationarity at the NEW state, then adapt
        # epsilon. This is deliberately recomputed at
        # (x^{l+1}, w^{l+1}, alpha^{l+1}); using the gradients applied above
        # would test the stale pre-ADMM state and would not implement line 13.
        (_, gate_w_grads), (_, gate_alpha_grads), gate_implicit_diag = (
            implicit_grads_split_xonly(
                x_l,
                x_gt,
                y,
                net_W,
                net_Alpha,
                denoiser,
                kernel_base,
                mu,
                rho,
                scale,
                neumann_iters=neumann_iters,
                adaptive_neumann=False,
                neumann_min_iters=neumann_iters,
                neumann_max_iters=neumann_iters,
                neumann_tol=neumann_tol,
                implicit_solver='neumann',
                neumann_catastrophic_limit=neumann_catastrophic_limit,
                moreau_epsilon=epsilon_l,
                valid_pad=valid_pad,
            )
        )
        adaptive_grad_l2 = math.sqrt(
            _tensor_list_l2_norm(gate_w_grads) ** 2
            + _tensor_list_l2_norm(gate_alpha_grads) ** 2
        )
        adaptive_info = moreau_state.update(adaptive_grad_l2)

        # Convergence diagnostics only; they do not alter the fixed-L baseline.
        with torch.no_grad():
            tiny = 1e-12
            primal_abs = torch.linalg.vector_norm(x_l - z_l)
            primal_res = (
                primal_abs
                / (torch.linalg.vector_norm(x_l) + tiny)
            )

            # Classical consensus-ADMM dual residual for x - z = 0:
            # s^{l+1} = rho (z^{l+1} - z^l).  The absolute norm is the
            # classical residual; dual_rel is an additional scale-free view.
            dual_abs = rho * torch.linalg.vector_norm(z_l - z_prev)
            dual_res = (
                dual_abs
                / (rho * torch.linalg.vector_norm(z_prev) + tiny)
            )

            state_change = torch.sqrt(
                torch.sum((x_l - x_prev) ** 2)
                + torch.sum((z_l - z_prev) ** 2)
                + torch.sum((u_l - u_prev) ** 2)
            )
            state_reference = torch.sqrt(
                torch.sum(x_prev ** 2)
                + torch.sum(z_prev ** 2)
                + torch.sum(u_prev ** 2)
            )
            state_res = state_change / (state_reference + tiny)
            psnr_y = benchmark_psnr_y(
                x_l,
                x_gt,
                valid_pad=valid_pad,
                border=scale,
            )
            raw_mse_objective = valid_upper_loss_value(
                x_l,
                x_gt,
                valid_pad,
            )
            moreau_objective = valid_moreau_upper_loss_value(
                x_l,
                x_gt,
                valid_pad,
                epsilon_l,
            )

        if logger is not None and should_log:
            with torch.no_grad():
                # w lives on the LR grid; alpha lives on the HR grid.
                if 'theta' not in captured_weight_logits:
                    raise RuntimeError(
                        'WeightNet diagnostic hook did not capture logits.'
                    )
                w_map = _constrain_weight_logits(
                    net_W,
                    captured_weight_logits['theta'],
                )
                w_valid_pad = valid_pad // scale if valid_pad > 0 else 0
                w_valid = crop_valid_region(w_map, w_valid_pad)
                alpha_valid = crop_valid_region(alpha_map, valid_pad)
                sigma_valid = alpha_valid / rho

                w_min = w_valid.min().item()
                w_mean = w_valid.mean().item()
                w_max = w_valid.max().item()
                alpha_min = alpha_valid.min().item()
                alpha_mean = alpha_valid.mean().item()
                alpha_max = alpha_valid.max().item()
                sigma_min = sigma_valid.min().item()
                sigma_mean = sigma_valid.mean().item()
                sigma_max = sigma_valid.max().item()

            logger.info(
                'epoch=%03d batch=%05d outer=%02d/%02d '
                'primal=%.6e state=%.6e psnr_y=%.6f '
                'implicit_grad=%.6e upper_mse=%.8e upper_moreau=%.8e '
                'primal_abs=%.6e dual_abs=%.6e dual_rel=%.6e',
                epoch_idx,
                batch_idx,
                l + 1,
                L_iterations,
                primal_res.item(),
                state_res.item(),
                psnr_y,
                applied_implicit_grad_norm,
                raw_mse_objective,
                moreau_objective,
                primal_abs.item(),
                dual_abs.item(),
                dual_res.item(),
            )
            logger.info(
                'epoch=%03d batch=%05d outer=%02d/%02d gradient_norms '
                'w_recon=%.6e w_preclip=%.6e w_postclip=%.6e '
                'w_clipped=%d alpha_recon=%.6e '
                'alpha_preclip=%.6e alpha_postclip=%.6e alpha_clipped=%d',
                epoch_idx,
                batch_idx,
                l + 1,
                L_iterations,
                w_recon_grad_norm,
                w_preclip_grad_norm,
                w_postclip_grad_norm,
                int(w_was_clipped),
                alpha_recon_grad_norm,
                alpha_preclip_grad_norm,
                alpha_postclip_grad_norm,
                int(alpha_was_clipped),
            )
            logger.info(
                'epoch=%03d batch=%05d outer=%02d/%02d adaptive_moreau '
                'epsilon_before=%.8e epsilon_after=%.8e gamma=%.8e '
                'sigma=%.8e norm_mode=%s grad_l2=%.8e grad_metric=%.8e '
                'threshold=%.8e triggered=%d reductions=%d converged=%d '
                'gate_neumann_k=%d gate_neumann_residual_rel=%.8e',
                epoch_idx,
                batch_idx,
                l + 1,
                L_iterations,
                adaptive_info['epsilon_before'],
                adaptive_info['epsilon_after'],
                moreau_state.gamma,
                moreau_state.sigma,
                moreau_state.norm_mode,
                adaptive_info['raw_l2_norm'],
                adaptive_info['metric_norm'],
                adaptive_info['threshold'],
                int(adaptive_info['triggered']),
                adaptive_info['reductions'],
                int(adaptive_info['converged']),
                gate_implicit_diag['neumann_k_used'],
                gate_implicit_diag['neumann_adjoint_residual_rel'],
            )
            logger.info(
                'epoch=%03d batch=%05d outer=%02d/%02d xonly_solver '
                'w_method=%s w_iters=%d w_converged=%d '
                'w_adjoint_abs=%.6e w_adjoint_rel=%.6e w_root_rel=%.6e '
                'alpha_method=%s alpha_iters=%d alpha_converged=%d '
                'alpha_adjoint_abs=%.6e alpha_adjoint_rel=%.6e '
                'alpha_root_rel=%.6e',
                epoch_idx,
                batch_idx,
                l + 1,
                L_iterations,
                w_implicit_diag['solver_method'],
                w_implicit_diag['solver_iterations'],
                int(w_implicit_diag['solver_converged']),
                w_implicit_diag['solver_residual_abs'],
                w_implicit_diag['solver_residual_rel'],
                w_implicit_diag['x_root_rel'],
                alpha_implicit_diag['solver_method'],
                alpha_implicit_diag['solver_iterations'],
                int(alpha_implicit_diag['solver_converged']),
                alpha_implicit_diag['solver_residual_abs'],
                alpha_implicit_diag['solver_residual_rel'],
                alpha_implicit_diag['x_root_rel'],
            )
            logger.info(
                'epoch=%03d batch=%05d outer=%02d/%02d neumann '
                'mode=%s tol=%.6e '
                'w_k=%d w_residual_abs=%.6e w_residual_rel=%.6e '
                'w_converged=%d w_finite=%d w_last_term=%.6e '
                'w_last_ratio=%.6e w_next_ratio=%.6e w_max_ratio=%.6e '
                'w_max_term_rel=%.6e w_adjoint_rel_seed=%.6e '
                'alpha_k=%d alpha_residual_abs=%.6e alpha_residual_rel=%.6e '
                'alpha_converged=%d alpha_finite=%d alpha_last_term=%.6e '
                'alpha_last_ratio=%.6e alpha_next_ratio=%.6e '
                'alpha_max_ratio=%.6e alpha_max_term_rel=%.6e '
                'alpha_adjoint_rel_seed=%.6e catastrophic_limit=%.6e',
                epoch_idx,
                batch_idx,
                l + 1,
                L_iterations,
                w_implicit_diag['neumann_mode'],
                neumann_tol,
                w_implicit_diag['neumann_k_used'],
                w_implicit_diag['neumann_adjoint_residual_abs'],
                w_implicit_diag['neumann_adjoint_residual_rel'],
                int(w_implicit_diag['neumann_converged']),
                int(w_implicit_diag['neumann_residual_finite']),
                w_implicit_diag['neumann_last_term'],
                w_implicit_diag['neumann_last_ratio'],
                w_implicit_diag['adjoint_residual_to_last_term'],
                w_implicit_diag['neumann_max_ratio'],
                w_implicit_diag['neumann_max_term_relative_to_seed'],
                w_implicit_diag['neumann_adjoint_relative_to_seed'],
                alpha_implicit_diag['neumann_k_used'],
                alpha_implicit_diag['neumann_adjoint_residual_abs'],
                alpha_implicit_diag['neumann_adjoint_residual_rel'],
                int(alpha_implicit_diag['neumann_converged']),
                int(alpha_implicit_diag['neumann_residual_finite']),
                alpha_implicit_diag['neumann_last_term'],
                alpha_implicit_diag['neumann_last_ratio'],
                alpha_implicit_diag['adjoint_residual_to_last_term'],
                alpha_implicit_diag['neumann_max_ratio'],
                alpha_implicit_diag['neumann_max_term_relative_to_seed'],
                alpha_implicit_diag['neumann_adjoint_relative_to_seed'],
                neumann_catastrophic_limit,
            )
            logger.info(
                'epoch=%03d batch=%05d outer=%02d/%02d map_stats '
                'w_min=%.6e w_mean=%.6e w_max=%.6e '
                'alpha_min=%.6e alpha_mean=%.6e alpha_max=%.6e '
                'sigma_min=%.6e sigma_mean=%.6e sigma_max=%.6e',
                epoch_idx,
                batch_idx,
                l + 1,
                L_iterations,
                w_min,
                w_mean,
                w_max,
                alpha_min,
                alpha_mean,
                alpha_max,
                sigma_min,
                sigma_mean,
                sigma_max,
            )

        # Algorithm 2 line 18 is a genuine termination test. Because epsilon
        # is global across mini-batches, the driver also stops the remaining
        # batches/epochs after this batch returns.
        if moreau_state.converged:
            if logger is not None:
                logger.info(
                    'epoch=%03d batch=%05d outer=%02d/%02d '
                    'adaptive_smoothing_stop sigma_epsilon=%.8e '
                    'epsilon_tol=%.8e',
                    epoch_idx,
                    batch_idx,
                    l + 1,
                    L_iterations,
                    moreau_state.sigma * moreau_state.epsilon,
                    moreau_state.epsilon_tol,
                )
            break

    raw_J = valid_upper_loss_value(x_l, x_gt, valid_pad)
    moreau_J = valid_moreau_upper_loss_value(
        x_l,
        x_gt,
        valid_pad,
        moreau_state.epsilon,
    )
    if logger is not None:
        _log_neumann_summary(
            logger,
            'batch',
            batch_neumann_summary,
            neumann_tol,
            epoch=f'{epoch_idx:03d}',
            batch=f'{batch_idx:05d}',
        )
    return x_l, moreau_J, raw_J, batch_neumann_summary


# ============================================================
# Training driver (loops Algorithm 2 over the dataset)
# ============================================================

def train_bilevel_algorithm2(
    dataloader,
    device='cuda',
    epochs=10,
    L_iterations=PAPER_ADMM_OUTER_ITERS,
    neumann_iters=PAPER_NEUMANN_K,
    neumann_tol=DEFAULT_NEUMANN_DIAGNOSTIC_TOL,
    neumann_catastrophic_limit=DEFAULT_NEUMANN_CATASTROPHIC_LIMIT,
    sigma_init=DEFAULT_SIGMA_INIT,
    lr_w=DEFAULT_LR_W,
    lr_alpha=DEFAULT_LR_ALPHA,
    mu=0.01,
    rho=0.05,
    scale=4,
    grad_clip_w=DEFAULT_GRAD_CLIP_W,
    grad_clip_alpha=DEFAULT_GRAD_CLIP_ALPHA,
    update_mode='sequential',
    denoiser_relax=1.0,
    epsilon_0=DEFAULT_MOREAU_EPSILON,
    epsilon_gamma=DEFAULT_EPSILON_GAMMA,
    epsilon_sigma=DEFAULT_EPSILON_SIGMA,
    epsilon_tol=DEFAULT_EPSILON_TOL,
    adaptive_norm='rms',
    train_pad=8,
    w_init='phi_xi_spatial_weights.pth',
    rgdn_init='best_model_rgdn.pth',
    rgdn_arch='legacy',
    out_prefix='bl_lcilw_algo2_xonly_moreau_gd_adaptive',
    log_dir='logs',
    log_every_outer=1,
    seed=1234,
):
    _require_project_dependencies()
    seed_everything(seed)
    print('Initializing Algorithm 2: MSE-only Moreau + adaptive epsilon + calibrated SGD...')
    train_pad = int(train_pad)
    if rho <= 0:
        raise ValueError(f'rho must be positive, got {rho}')
    if not math.isfinite(float(lr_w)) or lr_w <= 0.0:
        raise ValueError(f'lr_w must be positive and finite, got {lr_w}')
    if not math.isfinite(float(lr_alpha)) or lr_alpha <= 0.0:
        raise ValueError(f'lr_alpha must be positive and finite, got {lr_alpha}')
    if grad_clip_w is not None and (
        not math.isfinite(float(grad_clip_w)) or grad_clip_w <= 0.0
    ):
        raise ValueError(
            'grad_clip_w must be positive and finite or None, got '
            f'{grad_clip_w}'
        )
    if grad_clip_alpha is not None and (
        not math.isfinite(float(grad_clip_alpha)) or grad_clip_alpha <= 0.0
    ):
        raise ValueError(
            'grad_clip_alpha must be positive and finite or None, got '
            f'{grad_clip_alpha}'
        )
    if update_mode not in {'simultaneous', 'sequential'}:
        raise ValueError(
            "update_mode must be 'simultaneous' or 'sequential', got "
            f'{update_mode!r}'
        )
    if rgdn_arch not in {'legacy', 'contractive'}:
        raise ValueError(
            "rgdn_arch must be 'legacy' or 'contractive', got "
            f'{rgdn_arch!r}'
        )
    if scale <= 0:
        raise ValueError(f'scale must be positive, got {scale}')
    if L_iterations != PAPER_ADMM_OUTER_ITERS:
        raise ValueError(
            'This controlled Algorithm 2 variant fixes L=12; got '
            f'L_iterations={L_iterations}.'
        )
    if neumann_iters != PAPER_NEUMANN_K:
        raise ValueError(
            'This controlled Algorithm 2 variant fixes Neumann K=8; got '
            f'neumann_iters={neumann_iters}.'
        )
    if not math.isfinite(float(neumann_tol)) or neumann_tol <= 0:
        raise ValueError(
            f'neumann diagnostic tolerance must be positive, got {neumann_tol}'
        )
    if not math.isfinite(float(neumann_catastrophic_limit)):
        raise ValueError('neumann_catastrophic_limit must be finite')
    if neumann_catastrophic_limit <= 1.0:
        raise ValueError(
            'neumann_catastrophic_limit must be greater than one, got '
            f'{neumann_catastrophic_limit}'
        )
    if not 0.0 < float(denoiser_relax) <= 1.0:
        raise ValueError(
            'denoiser_relax must lie in (0, 1], got '
            f'{denoiser_relax}'
        )
    if log_every_outer <= 0:
        raise ValueError(f'log_every_outer must be positive, got {log_every_outer}')
    if train_pad < 0:
        raise ValueError("train_pad must be >= 0")
    if train_pad > 0 and train_pad % scale != 0:
        raise ValueError(
            f"train_pad is in HR pixels and must be divisible by scale={scale}. "
            f"Got {train_pad}."
        )

    logger, log_path = setup_live_file_logger(out_prefix, log_dir=log_dir)
    logger.info('training_start')
    logger.info(
        'config epochs=%d L=%d implicit_state=x_only implicit_solver=fixed_neumann '
        'neumann_fixed_k=%d neumann_diagnostic_tol=%.6e '
        'neumann_catastrophic_limit=%.6e sigma_init=%.6e '
        'optimizer=sgd momentum=0 weight_decay=0 lr_w=%.6e lr_alpha=%.6e '
        'mu=%.6g rho=%.6g upper_objective=mse_only '
        'smoothing=moreau_mse epsilon_0=%.6g epsilon_gamma=%.6g '
        'epsilon_sigma=%.6g epsilon_tol=%.6g adaptive_norm=%s '
        'denoiser_relax=%.6g rgdn_arch=%s '
        'grad_clip_w=%s grad_clip_alpha=%s train_pad=%d '
        'log_every_outer=%d update_mode=%s '
        'diagnostics=xonly_root_and_adjoint_residual '
        'seed=%d deterministic=warn_only',
        epochs,
        L_iterations,
        neumann_iters,
        neumann_tol,
        neumann_catastrophic_limit,
        sigma_init,
        lr_w,
        lr_alpha,
        mu,
        rho,
        epsilon_0,
        epsilon_gamma,
        epsilon_sigma,
        epsilon_tol,
        adaptive_norm,
        denoiser_relax,
        rgdn_arch,
        str(grad_clip_w),
        str(grad_clip_alpha),
        train_pad,
        log_every_outer,
        update_mode,
        seed,
    )

    net_W = PatchWeightedLCI2D(n_nodes=8, scale_factor=scale).to(device)
    net_Alpha = AlphaNet(in_channels=3, scale_factor=scale).to(device)

    alpha_initialization = initialize_alphanet_constant_sigma(
        net_Alpha,
        rho=rho,
        sigma_init=sigma_init,
    )
    logger.info(
        'alphanet_initialized mode=constant_sigma sigma_init=%.8e '
        'alpha_init=%.8e raw_bias=%.8e fully_trainable=1',
        alpha_initialization['sigma_init'],
        alpha_initialization['alpha_init'],
        alpha_initialization['raw_bias'],
    )

    # Load WeightNet
    try:
        net_W.load_state_dict(torch.load(w_init, map_location=device, weights_only=False))
        print("Loaded WeightNet checkpoint.")
        logger.info('weightnet_checkpoint_loaded path=%s', w_init)
    except Exception as exc:
        print(f"WeightNet load failed: {exc}")
        print("Initializing WeightNet from scratch.")
        logger.warning('weightnet_checkpoint_failed path=%s error=%s', w_init, exc)

    # Construct and load the selected RGDN.  Contractive checkpoints carry the
    # exact architecture configuration and an analytic certificate; guessing
    # from tensor shapes is intentionally forbidden.
    try:
        net_RGDN, rgdn_metadata = build_rgdn_from_checkpoint(
            rgdn_arch, rgdn_init, device
        )
        print(f"Loaded {rgdn_metadata['architecture']} checkpoint.")
        logger.info(
            'rgdn_checkpoint_loaded path=%s architecture=%s format=%s',
            rgdn_init,
            rgdn_metadata['architecture'],
            rgdn_metadata['checkpoint_format'],
        )
        if rgdn_metadata['certificate'] is not None:
            certificate = rgdn_metadata['certificate']
            relaxed_eta = net_RGDN.relaxed_contraction_factor(denoiser_relax)
            logger.info(
                'rgdn_certificate fixed_sigma_eta=%.8e averaged_alpha=%.8e '
                'bounded_range=[0,1] denoiser_relax=%.8e relaxed_eta=%.8e',
                certificate['input_lipschitz_upper_bound'],
                certificate['valid_averaged_alpha'],
                denoiser_relax,
                relaxed_eta,
            )
    except Exception as exc:
        logger.error('rgdn_checkpoint_failed path=%s error=%s', rgdn_init, exc)
        raise RuntimeError(
            "RGDN checkpoint loading failed; refusing to freeze it."
        ) from exc

    # Freeze only after successful loading
    net_RGDN.eval()
    for p in net_RGDN.parameters():
        p.requires_grad_(False)

    if denoiser_relax >= 1.0:
        denoiser = lambda v, sigma_: net_RGDN(v, sigma_)
    else:
        r = float(denoiser_relax)
        denoiser = lambda v, sigma_: v + r * (net_RGDN(v, sigma_) - v)

    # Plain mini-batch gradient descent. Explicit zero values make it impossible
    # to silently acquire Adam-like state, momentum, or regularization.
    opt_W = optim.SGD(
        net_W.parameters(),
        lr=lr_w,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        foreach=False,
    )
    opt_Alpha = optim.SGD(
        net_Alpha.parameters(),
        lr=lr_alpha,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        foreach=False,
    )

    upper_parameter_count = sum(
        parameter.numel()
        for module in (net_W, net_Alpha)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    moreau_state = AdaptiveMoreauState(
        epsilon=epsilon_0,
        gamma=epsilon_gamma,
        sigma=epsilon_sigma,
        epsilon_tol=epsilon_tol,
        parameter_count=upper_parameter_count,
        norm_mode=adaptive_norm,
    )
    logger.info(
        'adaptive_moreau_initialized parameter_count=%d epsilon=%.8e '
        'gamma=%.8e sigma=%.8e epsilon_tol=%.8e norm_mode=%s',
        upper_parameter_count,
        moreau_state.epsilon,
        moreau_state.gamma,
        moreau_state.sigma,
        moreau_state.epsilon_tol,
        moreau_state.norm_mode,
    )

    coords = torch.arange(5, dtype=torch.float32, device=device) - 2
    g1d = torch.exp(-coords ** 2 / 2.0)
    g2d = g1d.outer(g1d)
    kernel_base = (g2d / g2d.sum()).view(1, 1, 5, 5).to(device)

    print(
        f'[cfg] optimizer=plain SGD | lr_w={lr_w:g} lr_alpha={lr_alpha:g} '
        f'| upper=MSE only | Moreau epsilon_0={epsilon_0:g} '
        f'gamma={epsilon_gamma:g} sigma={epsilon_sigma:g} '
        f'norm={adaptive_norm} | update_mode={update_mode} '
        f'| denoiser_relax={denoiser_relax:g} '
        f'| HR train_pad={train_pad}'
    )
    print(
        f'[cfg] x-only fixed Neumann K={neumann_iters} | residual '
        f'tolerance={neumann_tol:g} is diagnostic only | catastrophic '
        f'limit={neumann_catastrophic_limit:g} | sigma_init={sigma_init:g} '
        f'| grad_clip_w={grad_clip_w} grad_clip_alpha={grad_clip_alpha}'
    )

    training_neumann_summary = _new_neumann_summary()
    alpha_initialization_verified = False
    best_mean_raw_mse = float('inf')
    for epoch in range(epochs):
        net_W.train(); net_Alpha.train()
        epoch_moreau_loss, epoch_raw_mse, nb = 0.0, 0.0, 0
        epoch_neumann_summary = _new_neumann_summary()
        for batch_idx, (y_unused, x_gt) in enumerate(
            tqdm(dataloader, desc=f"Bilevel Epoch {epoch+1}"),
            start=1,
        ):
            # Boundary-safe training path:
            #   HR -> reflect pad -> degrade -> ADMM/PnP -> loss only on valid center.
            # The LR returned by the dataset is intentionally ignored because it was
            # generated from the unpadded HR crop.
            x_gt = x_gt.to(device)

            # Keep the HR crop divisible by the scale, then add an HR reflect margin.
            _, _, H0, W0 = x_gt.shape
            H = (H0 // scale) * scale
            W = (W0 // scale) * scale
            x_gt = x_gt[:, :, :H, :W]

            if train_pad > 0:
                x_gt_train = F.pad(x_gt, (train_pad, train_pad, train_pad, train_pad), mode='reflect')
            else:
                x_gt_train = x_gt

            # Generate the LR observation from the padded HR target, matching inference.
            y = forward_A(x_gt_train, kernel_base, scale)

            if not alpha_initialization_verified:
                with torch.no_grad():
                    alpha_probe = net_Alpha(y)
                    observed_alpha = float(alpha_probe.mean())
                    observed_sigma = observed_alpha / rho
                    expected_sigma = alpha_initialization['sigma_init']
                    relative_error = abs(observed_sigma - expected_sigma) / expected_sigma
                if relative_error > 1.0e-5:
                    raise RuntimeError(
                        'AlphaNet sigma initialization verification failed: '
                        f'expected={expected_sigma:.8e}, '
                        f'observed={observed_sigma:.8e}, '
                        f'relative_error={relative_error:.3e}.'
                    )
                logger.info(
                    'alphanet_initialization_verified observed_alpha=%.8e '
                    'observed_sigma=%.8e relative_error=%.3e',
                    observed_alpha,
                    observed_sigma,
                    relative_error,
                )
                print(
                    '[check] AlphaNet initialization verified | '
                    f'alpha_mean={observed_alpha:.6e} | '
                    f'sigma_mean={observed_sigma:.6e}'
                )
                alpha_initialization_verified = True

            B, C, h_lr, w_lr = y.shape
            otf = get_effective_otf(kernel_base, scale, (B, C, h_lr * scale, w_lr * scale))
            ATy = adjoint_AT(y, kernel_base, scale)

            _, J_moreau, J_raw, batch_neumann_summary = run_algorithm2_on_image(
                y=y,
                x_gt=x_gt_train,
                net_W=net_W,
                net_Alpha=net_Alpha,
                denoiser=denoiser,
                opt_W=opt_W,
                opt_Alpha=opt_Alpha,
                otf=otf,
                ATy=ATy,
                kernel_base=kernel_base,
                mu=mu,
                rho=rho,
                scale=scale,
                L_iterations=L_iterations,
                neumann_iters=neumann_iters,
                moreau_state=moreau_state,
                neumann_tol=neumann_tol,
                neumann_catastrophic_limit=neumann_catastrophic_limit,
                grad_clip_w=grad_clip_w,
                grad_clip_alpha=grad_clip_alpha,
                update_mode=update_mode,
                valid_pad=train_pad,
                logger=logger,
                epoch_idx=epoch + 1,
                batch_idx=batch_idx,
                log_every_outer=log_every_outer,
            )
            epoch_moreau_loss += J_moreau
            epoch_raw_mse += J_raw
            nb += 1
            _merge_neumann_summary(epoch_neumann_summary, batch_neumann_summary)
            logger.info(
                'epoch=%03d batch=%05d batch_complete '
                'upper_moreau=%.8e upper_mse=%.8e epsilon=%.8e',
                epoch + 1,
                batch_idx,
                J_moreau,
                J_raw,
                moreau_state.epsilon,
            )
            if moreau_state.converged:
                logger.info(
                    'epoch=%03d training_loop_stop_requested batch=%05d',
                    epoch + 1,
                    batch_idx,
                )
                break

        mean_epoch_moreau = epoch_moreau_loss / max(nb, 1)
        mean_epoch_raw_mse = epoch_raw_mse / max(nb, 1)
        print(
            f'Epoch {epoch+1} complete | mean Moreau J='
            f'{mean_epoch_moreau:.6e} | mean raw MSE objective='
            f'{mean_epoch_raw_mse:.6e} | epsilon={moreau_state.epsilon:.6e}'
        )
        logger.info(
            'epoch=%03d epoch_complete mean_upper_moreau=%.8e '
            'mean_upper_mse=%.8e epsilon=%.8e reductions=%d batches=%d',
            epoch + 1,
            mean_epoch_moreau,
            mean_epoch_raw_mse,
            moreau_state.epsilon,
            moreau_state.reductions,
            nb,
        )
        _log_neumann_summary(
            logger,
            'epoch',
            epoch_neumann_summary,
            neumann_tol,
            epoch=f'{epoch + 1:03d}',
        )
        _merge_neumann_summary(training_neumann_summary, epoch_neumann_summary)

        checkpoint = {
            'net_W': net_W.state_dict(),
            'net_Alpha': net_Alpha.state_dict(),
            'net_RGDN': net_RGDN.state_dict(),
            'rgdn_checkpoint_format': rgdn_metadata['checkpoint_format'],
            'rgdn_model_config': rgdn_metadata['model_config'],
            'rgdn_certificate': rgdn_metadata['certificate'],
            'optimizer_W': opt_W.state_dict(),
            'optimizer_Alpha': opt_Alpha.state_dict(),
            'epoch': epoch + 1,
            'mean_upper_moreau': mean_epoch_moreau,
            'mean_upper_mse': mean_epoch_raw_mse,
            'adaptive_moreau': moreau_state.state_dict(),
            'hparams': {'mu': mu, 'rho': rho, 'scale': scale, 'L': L_iterations,
                        'neumann_iters': neumann_iters,
                        'neumann_mode': 'fixed',
                        'neumann_diagnostic_tol': neumann_tol,
                        'neumann_catastrophic_limit': neumann_catastrophic_limit,
                        'implicit_state': 'x_only',
                        'implicit_solver': 'fixed_neumann',
                        'sigma_init': sigma_init,
                        'alpha_init': alpha_initialization['alpha_init'],
                        'upper_objective': 'mse_only',
                        'smoothing': 'moreau_envelope_mse',
                        'epsilon_0': epsilon_0,
                        'epsilon_gamma': epsilon_gamma,
                        'epsilon_sigma': epsilon_sigma,
                        'epsilon_tol': epsilon_tol,
                        'adaptive_norm': adaptive_norm,
                        'optimizer': 'sgd',
                        'momentum': 0.0,
                        'weight_decay': 0.0,
                        'lr_w': lr_w,
                        'lr_alpha': lr_alpha,
                        'rgdn_arch': rgdn_arch,
                        'rgdn_checkpoint_format': rgdn_metadata['checkpoint_format'],
                        'rgdn_model_config': rgdn_metadata['model_config'],
                        'rgdn_certificate': rgdn_metadata['certificate'],
                        'denoiser_relax': denoiser_relax, 'train_pad': train_pad,
                        'sequential': update_mode == 'sequential',
                        'update_mode': update_mode,
                        'diagnostics': 'xonly_root_and_adjoint_residual',
                        'seed': seed,
                        'deterministic': 'warn_only'},
        }
        torch.save(checkpoint, f'{out_prefix}_full.pth')
        torch.save(net_W.state_dict(), f'{out_prefix}_w_final.pth')
        torch.save(net_Alpha.state_dict(), f'{out_prefix}_alpha_final.pth')
        is_best = mean_epoch_raw_mse < best_mean_raw_mse
        if is_best:
            best_mean_raw_mse = mean_epoch_raw_mse
            torch.save(checkpoint, f'{out_prefix}_best_full.pth')
            torch.save(net_W.state_dict(), f'{out_prefix}_w_best.pth')
            torch.save(net_Alpha.state_dict(), f'{out_prefix}_alpha_best.pth')
        logger.info(
            'epoch=%03d checkpoints_saved prefix=%s is_best=%d '
            'best_mean_upper_mse=%.8e',
            epoch + 1,
            out_prefix,
            int(is_best),
            best_mean_raw_mse,
        )
        if moreau_state.converged:
            break

    _log_neumann_summary(
        logger,
        'training',
        training_neumann_summary,
        neumann_tol,
    )
    logger.info(
        'training_complete log_path=%s epsilon_final=%.8e reductions=%d '
        'smoothing_converged=%d best_mean_upper_mse=%.8e',
        log_path,
        moreau_state.epsilon,
        moreau_state.reductions,
        int(moreau_state.converged),
        best_mean_raw_mse,
    )
    for handler in logger.handlers:
        handler.flush()
    return net_W, net_Alpha, net_RGDN, moreau_state


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(
        description=(
            'BL-LCILW Algorithm 2 ablation: MSE-only exact Moreau envelope, '
            'gradient-norm-triggered epsilon decay, plain gradient descent, '
            'one ADMM step per outer iteration, and fixed K=8 x-only '
            'truncated Neumann differentiation.'
        )
    )
    ap.add_argument(
        '--self_test',
        action='store_true',
        help=(
            'run the project-independent x-only mathematical self-test and '
            'exit without loading data or checkpoints'
        ),
    )
    ap.add_argument('--data_root', default='./data/BSDS500')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--hr_patch', type=int, default=128)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--admm_iters', type=int, default=PAPER_ADMM_OUTER_ITERS,
                    help='paper outer-iteration count; this baseline requires 12')
    ap.add_argument(
        '--neumann_iters',
        type=int,
        default=PAPER_NEUMANN_K,
        help='fixed paper-style Neumann truncation; this baseline requires 8',
    )
    ap.add_argument(
        '--neumann_tol',
        type=float,
        default=DEFAULT_NEUMANN_DIAGNOSTIC_TOL,
        help=(
            'diagnostic relative adjoint-residual threshold (default: 1e-2); '
            'it is logged and never used to reject an otherwise finite K=8 gradient'
        ),
    )
    ap.add_argument(
        '--neumann_catastrophic_limit',
        type=float,
        default=DEFAULT_NEUMANN_CATASTROPHIC_LIMIT,
        help=(
            'fail-closed limit for a Neumann term ratio, maximum term relative '
            'to the seed, or adjoint relative to the seed (default: 100)'
        ),
    )
    ap.add_argument(
        '--sigma_init',
        type=float,
        default=DEFAULT_SIGMA_INIT,
        help=(
            'explicit initial mean RGDN sigma produced by AlphaNet '
            '(default: 0.12; alpha_0=rho*sigma_init)'
        ),
    )
    ap.add_argument('--mu', type=float, default=0.01)
    ap.add_argument('--rho', type=float, default=0.05)
    ap.add_argument('--lr_w', type=float, default=DEFAULT_LR_W,
                    help='WeightNet calibrated-SGD step size (default: 3e-6)')
    ap.add_argument('--lr_alpha', type=float, default=DEFAULT_LR_ALPHA,
                    help='AlphaNet calibrated-SGD step size (default: 1e-5)')
    ap.add_argument(
        '--grad_clip_w',
        type=float,
        default=DEFAULT_GRAD_CLIP_W,
        help='WeightNet global-norm clip; use 0 to disable (default: 100)',
    )
    ap.add_argument(
        '--grad_clip_alpha',
        type=float,
        default=DEFAULT_GRAD_CLIP_ALPHA,
        help='AlphaNet global-norm clip; use 0 to disable (default: 100)',
    )
    ap.add_argument(
        '--update_mode',
        choices=('simultaneous', 'sequential'),
        default='sequential',
        help='paper Gauss--Seidel upper step or simultaneous/Jacobi ablation',
    )
    ap.add_argument('--epsilon_0', type=float, default=DEFAULT_MOREAU_EPSILON,
                    help='initial Moreau-envelope parameter (default: 0.1)')
    ap.add_argument('--epsilon_gamma', type=float, default=DEFAULT_EPSILON_GAMMA,
                    help='triggered geometric decay factor in (0,1) (default: 0.8)')
    ap.add_argument('--epsilon_sigma', type=float, default=DEFAULT_EPSILON_SIGMA,
                    help='gradient-norm threshold constant (default: 0.5)')
    ap.add_argument('--epsilon_tol', type=float, default=DEFAULT_EPSILON_TOL,
                    help='smoothing convergence tolerance (default: 1e-4)')
    ap.add_argument(
        '--adaptive_norm',
        choices=('rms', 'l2'),
        default='rms',
        help=(
            'gradient norm for the epsilon trigger: dimension-normalized RMS '
            '(recommended) or literal raw L2'
        ),
    )
    ap.add_argument('--denoiser_relax', type=float, default=1.0, 
                    help='<1 uses v+r*(RGDN(v)-v); for certified RGDN its bound is 1-r+r*eta < 1')
    ap.add_argument('--pad', type=int, default=8,
                    help='HR reflect padding before degradation during training; loss/gradient use center crop. Use 0 to disable.')
    ap.add_argument('--log_dir', default='logs',
                    help='directory for the live per-run training log')
    ap.add_argument('--log_every_outer', type=int, default=1,
                    help='write ADMM diagnostics every N outer iterations')
    ap.add_argument('--seed', type=int, default=1234,
                    help='fixed random seed for reproducible training')
    ap.add_argument('--w_init', default='phi_xi_spatial_weights.pth',
                    help='initial WeightNet state-dict checkpoint')
    ap.add_argument('--rgdn_init', default='best_model_rgdn.pth',
                    help='frozen RGDN training checkpoint')
    ap.add_argument(
        '--rgdn_arch',
        choices=('legacy', 'contractive'),
        default='legacy',
        help=(
            'RGDN checkpoint architecture; use contractive with checkpoints '
            'from train_rgdn_bsds500_contractive.py'
        ),
    )
    ap.add_argument(
        '--out_prefix',
        default='bl_lcilw_algo2_xonly_moreau_gd_adaptive',
        help='checkpoint and log prefix for this MSE/Moreau/GD ablation',
    )
    args = ap.parse_args()

    if args.grad_clip_w == 0.0:
        args.grad_clip_w = None
    if args.grad_clip_alpha == 0.0:
        args.grad_clip_alpha = None

    if args.self_test:
        run_self_test()
        raise SystemExit(0)

    _require_project_dependencies()

    seed_everything(args.seed)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading BSDS500 dataset...")
    train_dataset = BSDS500SRDataset(
        root_dir=args.data_root, split='train', hr_patch=args.hr_patch,
        scale=args.scale, samples_per_image=10, augment=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=seed_dataloader_worker,
        generator=data_generator)

    print('\nStarting MSE-only Moreau-smoothed Algorithm 2 with plain GD...')
    train_bilevel_algorithm2(
        dataloader=train_loader, device=device, epochs=args.epochs,
        L_iterations=args.admm_iters, neumann_iters=args.neumann_iters,
        neumann_tol=args.neumann_tol,
        neumann_catastrophic_limit=args.neumann_catastrophic_limit,
        sigma_init=args.sigma_init,
        lr_w=args.lr_w, lr_alpha=args.lr_alpha, mu=args.mu, rho=args.rho, scale=args.scale,
        grad_clip_w=args.grad_clip_w,
        grad_clip_alpha=args.grad_clip_alpha,
        update_mode=args.update_mode,
        denoiser_relax=args.denoiser_relax, train_pad=args.pad,
        epsilon_0=args.epsilon_0,
        epsilon_gamma=args.epsilon_gamma,
        epsilon_sigma=args.epsilon_sigma,
        epsilon_tol=args.epsilon_tol,
        adaptive_norm=args.adaptive_norm,
        w_init=args.w_init, rgdn_init=args.rgdn_init,
        rgdn_arch=args.rgdn_arch,
        out_prefix=args.out_prefix,
        log_dir=args.log_dir, log_every_outer=args.log_every_outer,
        seed=args.seed)
    print(f"\nDone. Checkpoints saved with prefix {args.out_prefix}.")
