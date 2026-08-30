"""Certified contractive RGDN for BL-LCILW / PnP-ADMM.

The guarantee implemented here is the one used by the BL-LCILW convergence
assumption.  For every *fixed* sigma map and every pair of equal-sized inputs,

    ||D_sigma(v) - D_sigma(v')||_2 <= eta ||v - v'||_2,   0 < eta < 1.

The output is also projected to [0, 1], so the operator has bounded range.
Every eta-contraction is alpha-averaged for any
alpha >= (1 + eta) / 2; the smallest value established by this construction
is reported by :meth:`ContractiveRGDN.certificate`.

This is a structural certificate, not a sampled Jacobian regularizer:

* the gradient lift is nonexpansive (normalized Sobel filters);
* pointwise orthogonal channel mixing has norm one -- either a FIXED
  deterministic DCT matrix (v1 behavior, ``learned_mixing=False``) or a
  LEARNED matrix kept exactly orthogonal throughout training by PyTorch's
  orthogonal parametrization (``learned_mixing=True``); an orthogonal matrix
  satisfies ||Qx|| = ||x|| for every possible learned value, so training can
  never break the bound;
* learned depthwise filters are normalized to l1 norm at most one (Young's
  convolution inequality, valid for ANY odd ``kernel_size``);
* GroupSort, sigmoid gates, and clipping are nonexpansive in the image input;
* every residual connection is a convex combination;
* the final affine contraction has slope eta < 1.

Sigma-dependent gates and biases depend only on sigma.  Consequently they do
not change the Lipschitz bound with respect to v when sigma is held fixed, as
it is in one PnP denoiser call.  No claim is made for pairs with different
sigma maps.

Backward compatibility: the constructor defaults (``kernel_size=3``,
``learned_mixing=False``) reproduce the original architecture exactly, so
existing contractive_rgdn_v1 checkpoints load unchanged.  New checkpoints
store the two extra keys in ``model_config`` and are reconstructed exactly by
:func:`load_contractive_rgdn`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrScalar = Union[torch.Tensor, float, int]


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(probability / (1.0 - probability))


def _dct_orthogonal(channels: int, phase: int = 0) -> torch.Tensor:
    """Return a deterministic orthogonal matrix with varied signed ordering."""
    n = torch.arange(channels, dtype=torch.float64)
    k = torch.arange(channels, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos(math.pi * (n + 0.5) * k / channels)
    matrix[0].mul_(math.sqrt(1.0 / channels))
    if channels > 1:
        matrix[1:].mul_(math.sqrt(2.0 / channels))

    # Row/column permutations and sign flips preserve orthogonality and give
    # different blocks distinct, deterministic channel mixing.
    matrix = torch.roll(matrix, shifts=(phase * 3) % channels, dims=0)
    matrix = torch.roll(matrix, shifts=(phase * 5) % channels, dims=1)
    signs = torch.where(
        (torch.arange(channels) + phase) % 2 == 0,
        torch.ones(channels, dtype=torch.float64),
        -torch.ones(channels, dtype=torch.float64),
    )
    matrix = signs[:, None] * matrix
    return matrix.float()


class FixedOrthogonalMix(nn.Module):
    """A fixed 1x1 orthogonal convolution (Euclidean operator norm one)."""

    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("orthogonal mixing matrix must be square")
        self.channels = int(matrix.shape[0])
        error = torch.linalg.matrix_norm(
            matrix.double().T @ matrix.double()
            - torch.eye(self.channels, dtype=torch.float64),
            ord=2,
        ).item()
        if error > 1.0e-5:
            raise ValueError(f"mixing matrix is not orthogonal: error={error:.3e}")
        self.register_buffer("matrix", matrix[:, :, None, None], persistent=False)

    def orthogonality_error(self) -> float:
        with torch.no_grad():
            w = self.matrix[:, :, 0, 0].double()
            eye = torch.eye(self.channels, dtype=torch.float64, device=w.device)
            return float(torch.linalg.matrix_norm(w.T @ w - eye, ord=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.matrix.to(dtype=x.dtype))


class LearnedOrthogonalMix(nn.Module):
    """A LEARNED 1x1 orthogonal convolution (Euclidean operator norm one).

    The weight is kept exactly orthogonal at every training step by
    ``torch.nn.utils.parametrizations.orthogonal`` (matrix-exponential
    trivialization), so ||Qx||_2 = ||x||_2 holds for every value the
    optimizer can ever produce.  This is the certified replacement for a
    dense learned channel-mixing layer: C^2 trainable parameters per mix,
    spectral norm exactly one by construction.
    """

    def __init__(self, channels: int, init: torch.Tensor | None = None):
        super().__init__()
        self.channels = int(channels)
        self.linear = nn.Linear(channels, channels, bias=False)
        torch.nn.utils.parametrizations.orthogonal(self.linear, "weight")
        if init is not None:
            if init.shape != (channels, channels):
                raise ValueError("init matrix has the wrong shape")
            try:
                with torch.no_grad():
                    self.linear.weight = init.float()
            except (RuntimeError, NotImplementedError, ValueError):
                # Assignment through the parametrization is unsupported on
                # some torch versions; the default random orthogonal
                # initialization is equally certified.
                pass

    def orthogonality_error(self) -> float:
        with torch.no_grad():
            w = self.linear.weight.double()
            eye = torch.eye(self.channels, dtype=torch.float64, device=w.device)
            return float(torch.linalg.matrix_norm(w.T @ w - eye, ord=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the orthogonal weight in full FP32 even under autocast so
        # the matrix exponential keeps orthogonality to float32 precision;
        # only the final cast follows the activation dtype (exactly the same
        # cast the fixed DCT path performs).
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight = self.linear.weight
        return F.conv2d(
            x, weight.to(dtype=x.dtype).view(self.channels, self.channels, 1, 1)
        )


class CertifiedDepthwiseConv2d(nn.Module):
    """Zero-padded depthwise convolution with induced l2 norm at most target.

    For each channel, Young's convolution inequality gives

        ||k * x||_2 <= ||k||_1 ||x||_2.

    Zero padding is a contraction, so normalizing every channel's kernel l1
    norm to at most ``target`` is valid for every spatial image size and for
    every odd ``kernel_size``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        target: float = 1.0,
        identity_init: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        if not 0.0 < float(target) <= 1.0:
            raise ValueError("target must lie in (0, 1]")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        self.target = float(target)
        raw = torch.zeros(channels, 1, kernel_size, kernel_size)
        if identity_init:
            raw[:, 0, self.padding, self.padding] = self.target
        else:
            nn.init.normal_(raw, mean=0.0, std=0.02)
        self.raw_weight = nn.Parameter(raw)

    def normalized_weight(self) -> torch.Tensor:
        l1 = self.raw_weight.abs().sum(dim=(1, 2, 3), keepdim=True)
        scale = torch.clamp(self.target / (l1 + 1.0e-12), max=1.0)
        return self.raw_weight * scale

    def certified_norms(self) -> torch.Tensor:
        return self.normalized_weight().detach().abs().sum(dim=(1, 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.normalized_weight(),
            padding=self.padding,
            groups=self.channels,
        )


def group_sort_2(x: torch.Tensor) -> torch.Tensor:
    """Pairwise GroupSort activation; a Euclidean-norm-preserving sorting map."""
    if x.shape[1] % 2:
        raise ValueError("GroupSort-2 requires an even channel count")
    batch, channels, height, width = x.shape
    paired = x.reshape(batch, channels // 2, 2, height, width)
    return paired.sort(dim=2).values.reshape(batch, channels, height, width)


class SigmaConditionedNonexpansiveBlock(nn.Module):
    """A sigma-conditioned nonexpansive block with a convex residual update."""

    def __init__(
        self,
        channels: int,
        phase: int,
        residual_mix_init: float = 0.10,
        gate_floor: float = 0.25,
        kernel_size: int = 3,
        learned_mixing: bool = False,
    ):
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even for GroupSort-2")
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("gate_floor must lie in [0, 1)")
        self.channels = int(channels)
        self.gate_floor = float(gate_floor)

        q = _dct_orthogonal(channels, phase=phase)
        if learned_mixing:
            self.mix_in = LearnedOrthogonalMix(channels, init=q)
            self.mix_out = LearnedOrthogonalMix(channels, init=q.T.contiguous())
        else:
            self.mix_in = FixedOrthogonalMix(q)
            self.mix_out = FixedOrthogonalMix(q.T.contiguous())
        self.spatial_1 = CertifiedDepthwiseConv2d(channels, kernel_size=kernel_size)
        self.spatial_2 = CertifiedDepthwiseConv2d(channels, kernel_size=kernel_size)

        # [gate_1, gate_2, bias_1, bias_2].  This subnetwork only sees sigma,
        # hence it cannot increase the fixed-sigma Lipschitz constant in v.
        self.sigma_conditioner = nn.Conv2d(1, 4 * channels, 1, bias=True)
        nn.init.zeros_(self.sigma_conditioner.weight)
        nn.init.zeros_(self.sigma_conditioner.bias)
        with torch.no_grad():
            self.sigma_conditioner.bias[: 2 * channels].fill_(6.0)

        self.raw_residual_mix = nn.Parameter(
            torch.tensor(_logit(residual_mix_init), dtype=torch.float32)
        )

    def _condition(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        raw = self.sigma_conditioner(sigma)
        gate_1, gate_2, bias_1, bias_2 = raw.chunk(4, dim=1)
        gate_1 = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(gate_1)
        gate_2 = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(gate_2)
        # A bounded translation is useful numerically and has zero derivative
        # with respect to the image input at fixed sigma.
        bias_1 = 0.25 * torch.tanh(bias_1)
        bias_2 = 0.25 * torch.tanh(bias_2)
        return gate_1, gate_2, bias_1, bias_2

    def forward(self, h: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        gate_1, gate_2, bias_1, bias_2 = self._condition(sigma)
        branch = self.mix_in(h)
        branch = gate_1 * self.spatial_1(branch) + bias_1
        branch = group_sort_2(branch)
        branch = self.mix_out(branch)
        branch = gate_2 * self.spatial_2(branch) + bias_2
        mix = torch.sigmoid(self.raw_residual_mix)
        return (1.0 - mix) * h + mix * branch


class ContractiveRGDN(nn.Module):
    """Bounded, fixed-sigma eta-contractive gradient-guided denoiser.

    ``use_attention`` is accepted only for constructor compatibility with the
    legacy RGDN call site.  Unsafe input-dependent attention is deliberately
    not used; normalized gradient features replace it.

    ``kernel_size`` (odd) sets the certified depthwise kernel size and
    ``learned_mixing`` swaps the fixed DCT channel mixing for learned exactly-
    orthogonal matrices.  Both default to the original v1 behavior.
    """

    architecture_name = "contractive_rgdn_v1"

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        use_attention: bool = True,
        eta: float = 0.99,
        gradient_coeff: float = 0.20,
        anchor: float = 0.50,
        residual_mix_init: float = 0.10,
        output_mix_init: float = 0.25,
        sigma_min: float = 1.0e-3,
        sigma_max: float = 0.50,
        kernel_size: int = 3,
        learned_mixing: bool = False,
    ):
        super().__init__()
        del use_attention
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_features < 3 * in_channels or num_features % 2:
            raise ValueError(
                "num_features must be even and at least 3*in_channels "
                f"(got {num_features} for {in_channels} channels)"
            )
        if not 0.0 < float(eta) < 1.0:
            raise ValueError("eta must lie strictly between zero and one")
        if not 0.0 <= float(gradient_coeff) < 1.0 / math.sqrt(2.0):
            raise ValueError("gradient_coeff must lie in [0, 1/sqrt(2))")
        if not 0.0 <= float(anchor) <= 1.0:
            raise ValueError("anchor must lie in [0, 1]")
        if not 0.0 <= float(residual_mix_init) <= 1.0:
            raise ValueError("residual_mix_init must lie in [0, 1]")
        if not 0.0 <= float(output_mix_init) <= 1.0:
            raise ValueError("output_mix_init must lie in [0, 1]")
        if not 0.0 <= float(sigma_min) < float(sigma_max):
            raise ValueError("require 0 <= sigma_min < sigma_max")
        if int(kernel_size) < 1 or int(kernel_size) % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")

        self.in_channels = int(in_channels)
        self.num_features = int(num_features)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.learned_mixing = bool(learned_mixing)
        self.gradient_coeff = float(gradient_coeff)
        self.base_coeff = math.sqrt(1.0 - 2.0 * self.gradient_coeff ** 2)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.register_buffer("eta_tensor", torch.tensor(float(eta)))
        self.register_buffer("anchor_tensor", torch.tensor(float(anchor)))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.T.contiguous()
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=False)
        self.register_buffer("sobel_y", sobel_y[None, None], persistent=False)

        # An additive sigma embedding cannot alter the fixed-sigma image bound.
        self.sigma_lift = nn.Conv2d(1, num_features, 1, bias=True)
        nn.init.zeros_(self.sigma_lift.weight)
        nn.init.zeros_(self.sigma_lift.bias)

        self.body = nn.ModuleList(
            [
                SigmaConditionedNonexpansiveBlock(
                    num_features,
                    phase=index + 1,
                    residual_mix_init=residual_mix_init,
                    kernel_size=self.kernel_size,
                    learned_mixing=self.learned_mixing,
                )
                for index in range(num_blocks)
            ]
        )
        self.tail_spatial = CertifiedDepthwiseConv2d(
            in_channels, kernel_size=self.kernel_size
        )
        self.tail_conditioner = nn.Conv2d(1, 2 * in_channels, 1, bias=True)
        nn.init.zeros_(self.tail_conditioner.weight)
        nn.init.zeros_(self.tail_conditioner.bias)
        with torch.no_grad():
            self.tail_conditioner.bias[:in_channels].fill_(6.0)
        self.raw_output_mix = nn.Parameter(
            torch.tensor(_logit(output_mix_init), dtype=torch.float32)
        )

    @property
    def eta(self) -> float:
        return float(self.eta_tensor.detach().cpu().item())

    @property
    def anchor(self) -> float:
        return float(self.anchor_tensor.detach().cpu().item())

    def model_config(self) -> Dict[str, Any]:
        first_mix = float(torch.sigmoid(self.body[0].raw_residual_mix).detach()) if self.body else 0.0
        return {
            "in_channels": self.in_channels,
            "num_features": self.num_features,
            "num_blocks": self.num_blocks,
            "eta": self.eta,
            "gradient_coeff": self.gradient_coeff,
            "anchor": self.anchor,
            # Initialization values are irrelevant after state_dict loading,
            # but valid values make reconstruction explicit and portable.
            "residual_mix_init": first_mix,
            "output_mix_init": float(torch.sigmoid(self.raw_output_mix).detach()),
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "kernel_size": self.kernel_size,
            "learned_mixing": self.learned_mixing,
        }

    def _prepare_sigma(self, v: torch.Tensor, sigma_map: TensorOrScalar) -> torch.Tensor:
        batch, _, height, width = v.shape
        if isinstance(sigma_map, (float, int)):
            sigma = torch.full(
                (batch, 1, height, width),
                float(sigma_map),
                dtype=v.dtype,
                device=v.device,
            )
        elif torch.is_tensor(sigma_map):
            sigma = sigma_map.to(dtype=v.dtype, device=v.device)
            if sigma.ndim == 0:
                sigma = sigma.reshape(1, 1, 1, 1)
            elif sigma.ndim == 1:
                if sigma.numel() not in (1, batch):
                    raise ValueError("1D sigma must have one value or one value per batch item")
                sigma = sigma.reshape(-1, 1, 1, 1)
            elif sigma.ndim == 2:
                sigma = sigma[None, None]
            elif sigma.ndim == 3:
                sigma = sigma[:, None]
            elif sigma.ndim != 4:
                raise ValueError(f"unsupported sigma shape {tuple(sigma.shape)}")
            if sigma.shape[1] != 1:
                raise ValueError("sigma_map must have exactly one channel")
            if sigma.shape[0] == 1 and batch > 1:
                sigma = sigma.expand(batch, -1, -1, -1)
            elif sigma.shape[0] != batch:
                raise ValueError("sigma batch dimension is incompatible with v")
            if sigma.shape[-2:] != (height, width):
                sigma = F.interpolate(
                    sigma,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            raise TypeError(f"sigma_map must be a scalar or tensor, got {type(sigma_map)!r}")
        return sigma.clamp(self.sigma_min, self.sigma_max)

    def _gradient_lift(self, v: torch.Tensor) -> torch.Tensor:
        kernels_x = self.sobel_x.to(dtype=v.dtype).expand(self.in_channels, 1, 3, 3)
        kernels_y = self.sobel_y.to(dtype=v.dtype).expand(self.in_channels, 1, 3, 3)
        gx = F.conv2d(v, kernels_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(v, kernels_y, padding=1, groups=self.in_channels)
        lifted = torch.cat(
            [
                self.base_coeff * v,
                self.gradient_coeff * gx,
                self.gradient_coeff * gy,
            ],
            dim=1,
        )
        missing = self.num_features - lifted.shape[1]
        if missing:
            lifted = torch.cat(
                [
                    lifted,
                    lifted.new_zeros(
                        lifted.shape[0], missing, lifted.shape[2], lifted.shape[3]
                    ),
                ],
                dim=1,
            )
        return lifted

    def forward(self, v: torch.Tensor, sigma_map: TensorOrScalar) -> torch.Tensor:
        single = v.ndim == 3
        if single:
            v = v.unsqueeze(0)
        if v.ndim != 4 or v.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B,{self.in_channels},H,W] or [{self.in_channels},H,W], "
                f"got {tuple(v.shape)}"
            )
        sigma = self._prepare_sigma(v, sigma_map)

        h = self._gradient_lift(v) + self.sigma_lift(sigma)
        for block in self.body:
            h = block(h, sigma)

        branch = h[:, : self.in_channels]
        raw_tail = self.tail_conditioner(sigma)
        gate, bias = raw_tail.chunk(2, dim=1)
        gate = 0.25 + 0.75 * torch.sigmoid(gate)
        bias = 0.25 * torch.tanh(bias)
        branch = torch.clamp(gate * self.tail_spatial(branch) + bias, 0.0, 1.0)

        output_mix = torch.sigmoid(self.raw_output_mix)
        nonexpansive = (1.0 - output_mix) * v + output_mix * branch
        eta = self.eta_tensor.to(dtype=v.dtype)
        anchor = self.anchor_tensor.to(dtype=v.dtype)
        output = torch.clamp(anchor + eta * (nonexpansive - anchor), 0.0, 1.0)
        return output.squeeze(0) if single else output

    def predicted_noise(self, v: torch.Tensor, sigma_map: TensorOrScalar) -> torch.Tensor:
        """Return the residual v-D_sigma(v), for legacy residual interpretation."""
        return v - self(v, sigma_map)

    @torch.no_grad()
    def certificate(self) -> Dict[str, Any]:
        layer_norms = []
        mixing_errors = []
        for block in self.body:
            layer_norms.extend(block.spatial_1.certified_norms().cpu().tolist())
            layer_norms.extend(block.spatial_2.certified_norms().cpu().tolist())
            mixing_errors.append(block.mix_in.orthogonality_error())
            mixing_errors.append(block.mix_out.orthogonality_error())
        layer_norms.extend(self.tail_spatial.certified_norms().cpu().tolist())
        maximum = max(layer_norms) if layer_norms else 0.0
        max_mix_error = max(mixing_errors) if mixing_errors else 0.0
        eta = self.eta
        return {
            "architecture": self.architecture_name,
            "scope": "input v at fixed sigma_map",
            "bounded_range": [0.0, 1.0],
            "input_lipschitz_upper_bound": eta,
            "strict_contraction": bool(0.0 < eta < 1.0),
            "valid_averaged_alpha": (1.0 + eta) / 2.0,
            "gradient_lift_upper_bound": math.sqrt(
                self.base_coeff ** 2 + 2.0 * self.gradient_coeff ** 2
            ),
            "maximum_normalized_depthwise_l1": maximum,
            "kernel_size": self.kernel_size,
            "mixing": (
                "learned_orthogonal" if self.learned_mixing
                else "fixed_orthogonal_dct"
            ),
            "maximum_mixing_orthogonality_error": max_mix_error,
            "all_certified_layers_nonexpansive": bool(
                maximum <= 1.0 + 1.0e-6 and max_mix_error <= 1.0e-4
            ),
        }

    def relaxed_contraction_factor(self, relaxation: float) -> float:
        """Bound for (1-r)I + rD when 0 < r <= 1."""
        relaxation = float(relaxation)
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")
        return (1.0 - relaxation) + relaxation * self.eta


def load_contractive_rgdn(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[ContractiveRGDN, Mapping[str, Any]]:
    """Load a certified RGDN checkpoint and reconstruct its exact architecture."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("RGDN checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict", checkpoint)
    config = dict(checkpoint.get("model_config", {}))
    if not config:
        raise KeyError(
            "contractive RGDN checkpoint is missing model_config; refusing to guess"
        )
    allowed = {
        "in_channels",
        "num_features",
        "num_blocks",
        "eta",
        "gradient_coeff",
        "anchor",
        "residual_mix_init",
        "output_mix_init",
        "sigma_min",
        "sigma_max",
        "kernel_size",
        "learned_mixing",
    }
    model = ContractiveRGDN(**{key: value for key, value in config.items() if key in allowed})
    model.load_state_dict(state, strict=strict)
    certificate = model.certificate()
    if not certificate["strict_contraction"] or not certificate[
        "all_certified_layers_nonexpansive"
    ]:
        raise RuntimeError(f"invalid contractivity certificate: {certificate}")
    return model, checkpoint


# Explicit alias for code that expects the generic class name after selecting
# the contractive architecture module.
RGDN = ContractiveRGDN


__all__ = [
    "CertifiedDepthwiseConv2d",
    "ContractiveRGDN",
    "FixedOrthogonalMix",
    "LearnedOrthogonalMix",
    "RGDN",
    "load_contractive_rgdn",
]
