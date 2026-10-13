"""Pick the vqvae/ code that matches a checkpoint, before importing it.

The CS-3 port changed the encoder readout (queries seeded with the encoder
mean), so pre-port GB10 checkpoints (E3, C1, ...) produce different codes under
main's vqvae/ even though their weights can be renamed to load. They must run
on the pre-port code, kept as a worktree of origin/gb10-scaleup:

    git worktree add --detach ../nubun_preport origin/gb10-scaleup

Usage, at the top of a script and BEFORE any `import vqvae...`:

    import vqvae_select
    vqvae_select.select_from_argv()       # reads --checkpoint, selects vqvae/
    from vqvae.model import VQVAE
    ...
    ckpt = vqvae_select.load_checkpoint(args.checkpoint)   # cached, no reload

Pre-port checkpoints are recognised by nn.MultiheadAttention's packed
in_proj_weight, which the port replaced with separate q/k/v projections.
"""
import os
import sys

import torch

PREPORT_DIR = os.environ.get(
    "NUBUN_PREPORT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nubun_preport"))


def is_preport(state_dict: dict) -> bool:
    return any(k.endswith("in_proj_weight") for k in state_dict)


_cache: dict = {}


def load_checkpoint(path: str) -> dict:
    if path in _cache:
        return _cache[path]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _cache[path] = ckpt
    if is_preport(ckpt["model_state"]):
        root = os.path.abspath(PREPORT_DIR)
        if not os.path.isdir(os.path.join(root, "vqvae")):
            raise SystemExit(f"{path} is a pre-port checkpoint; needs pre-port code at {root} "
                             "(git worktree add --detach ../nubun_preport origin/gb10-scaleup)")
        if any(m == "vqvae" or m.startswith("vqvae.") for m in sys.modules):
            raise RuntimeError("vqvae was imported before load_checkpoint(); call it first")
        sys.path.insert(0, root)
        print(f"pre-port checkpoint: using vqvae/ from {root}")
    return ckpt


def select_from_argv(flag: str = "--checkpoint") -> None:
    """Select vqvae/ from the script's --checkpoint argument (no-op if absent)."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            load_checkpoint(argv[i + 1])
        elif a.startswith(flag + "="):
            load_checkpoint(a.split("=", 1)[1])
