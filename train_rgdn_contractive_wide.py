"""
train_rgdn_contractive_wide.py
========================================================================
Launcher for the WIDE certified contractive RGDN: learned orthogonal
channel mixing + kernel-5 depthwise + 160 features / 12 blocks.

Reuses your validated trainer (train_rgdn_bsds500_contractive.py) verbatim
-- datasets, AMP overflow policy, verification gates, checkpoint format,
resume -- and only injects the two new architecture options into the model
constructor. Everything saved is standard contractive_rgdn_v1 format with
model_config carrying kernel_size / learned_mixing, so every loader in the
project reconstructs it exactly.

Requires the UPDATED rgdn_model_contractive.py in the same directory.

Usage: identical to the original trainer plus two flags:
    --kernel_size 5 --learned_mixing
"""

import functools
import json

import torch

import train_rgdn_bsds500_contractive as T
from rgdn_model_contractive import ContractiveRGDN


def main() -> None:
    parser = T.build_parser()
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="odd certified depthwise kernel size")
    parser.add_argument("--learned_mixing", action="store_true",
                        help="learned exactly-orthogonal 1x1 channel mixing")
    args = parser.parse_args()
    T.validate_arguments(args)

    # Inject the new architecture options into every construction the
    # trainer performs; checkpoint loading paths use model_config and are
    # unaffected.
    T.ContractiveRGDN = functools.partial(
        ContractiveRGDN,
        kernel_size=args.kernel_size,
        learned_mixing=args.learned_mixing,
    )

    probe = T.ContractiveRGDN(
        in_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
        eta=args.eta,
    )
    total = sum(p.numel() for p in probe.parameters())
    print(f"[wide] features={args.num_features} blocks={args.num_blocks} "
          f"kernel={args.kernel_size} learned_mixing={args.learned_mixing} "
          f"| parameters={total:,}")
    print(json.dumps(probe.certificate(), indent=2))
    del probe

    if args.verify_only:
        T.seed_everything(args.seed, deterministic=not args.nondeterministic)
        T.verify_checkpoint(args)
    else:
        T.train(args)


if __name__ == "__main__":
    main()
