import os
from typing import cast

import torch

DEBUG_FINITE = os.getenv("BGB_DEBUG_FINITE", "0") == "1"


def assert_finite(tag: str, x: torch.Tensor) -> None:
    if DEBUG_FINITE and not torch.isfinite(x).all():
        # Use getattr to handle torch.nanmin/nanmax for mypy
        nanmin = cast(float, torch.nanmin(x).item())  # type: ignore[attr-defined]
        nanmax = cast(float, torch.nanmax(x).item())  # type: ignore[attr-defined]
        mu = float(torch.nanmean(torch.nan_to_num(x)))
        print(
            f"[FINITE-FAIL] {tag}: min={nanmin:.3e} max={nanmax:.3e} mean={mu:.3e} shape={x.shape}",
            flush=True,
        )
        raise FloatingPointError(f"Non-finite at {tag}")
