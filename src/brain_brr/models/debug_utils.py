import torch

from src.brain_brr.utils.env import env

# Enable finite assertions by default in smoke tests and when nan-debugging
DEBUG_FINITE = env.debug_finite() or env.smoke_test() or env.nan_debug()


def assert_finite(tag: str, x: torch.Tensor) -> None:
    if DEBUG_FINITE and not torch.isfinite(x).all():
        # Handle both old and new PyTorch versions
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
        xmin = float(x_clean.min().item())
        xmax = float(x_clean.max().item())
        mu = float(x_clean.mean().item())
        print(
            f"[FINITE-FAIL] {tag}: min={xmin:.3e} max={xmax:.3e} mean={mu:.3e} shape={x.shape}",
            flush=True,
        )
        raise FloatingPointError(f"Non-finite at {tag}")
