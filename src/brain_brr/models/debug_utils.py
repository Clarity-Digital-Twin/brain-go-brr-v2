import os

import torch

DEBUG_FINITE = os.getenv("BGB_DEBUG_FINITE", "0") == "1"


def assert_finite(tag: str, x: torch.Tensor) -> None:
    if DEBUG_FINITE and not torch.isfinite(x).all():
        mn = float(torch.nanmin(x))
        mx = float(torch.nanmax(x))
        mu = float(torch.nanmean(torch.nan_to_num(x)))
        print(f"[FINITE-FAIL] {tag}: min={mn:.3e} max={mx:.3e} mean={mu:.3e} shape={x.shape}", flush=True)
        raise FloatingPointError(f"Non-finite at {tag}")