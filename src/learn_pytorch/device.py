"""Device selection helpers for PyTorch experiments."""

from __future__ import annotations

from typing import Literal

import torch

DevicePreference = Literal["auto", "cuda", "mps", "cpu"]


def get_device(preference: DevicePreference = "auto") -> torch.device:
    """Return a PyTorch device using cuda, then mps, then cpu by default.

    Args:
        preference: Use "auto" for best available device, or request a
            specific backend. Requested unavailable backends fall back to CPU.
    """
    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if preference == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps") if has_mps else torch.device("cpu")

    if preference != "auto":
        raise ValueError(f"Unknown device preference: {preference!r}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return torch.device("mps")

    return torch.device("cpu")
