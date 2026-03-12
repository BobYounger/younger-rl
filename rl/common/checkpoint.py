from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save model state together with optional optimizer and metadata."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step) if step is not None else None,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra is not None:
        payload["extra"] = dict(extra)

    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model state and optionally restore optimizer state."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model"], strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Optional[Path]:
    """Return the most recently modified checkpoint in a directory."""
    candidates = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
