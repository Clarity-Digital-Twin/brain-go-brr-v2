import torch

from src.brain_brr.utils.env import env

# Enable finite assertions by default in smoke tests and when nan-debugging
DEBUG_FINITE = env.debug_finite() or env.smoke_test() or env.nan_debug()


def assert_finite(tag: str, x: torch.Tensor, raise_on_fail: bool = True) -> bool:
    """Assert tensor contains only finite values with detailed debugging.

    Args:
        tag: Identifier for this check location
        x: Tensor to check
        raise_on_fail: Whether to raise exception on failure

    Returns:
        True if all values are finite, False otherwise
    """
    if not DEBUG_FINITE:
        return True

    if torch.isfinite(x).all():
        return True

    # Detailed analysis when non-finite values found
    nan_count = torch.isnan(x).sum().item()
    inf_count = torch.isinf(x).sum().item()
    total = x.numel()

    # Handle both old and new PyTorch versions
    x_clean = torch.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    xmin = float(x_clean.min().item())
    xmax = float(x_clean.max().item())
    mu = float(x_clean.mean().item())
    std = float(x_clean.std().item())

    print(
        f"[NaN-CHECK FAIL] {tag}:\n"
        f"  Shape: {x.shape}, Total: {total}\n"
        f"  NaN: {nan_count} ({100*nan_count/total:.2f}%)\n"
        f"  Inf: {inf_count} ({100*inf_count/total:.2f}%)\n"
        f"  Clean stats: min={xmin:.3e}, max={xmax:.3e}, mean={mu:.3e}, std={std:.3e}",
        flush=True,
    )

    if raise_on_fail:
        raise FloatingPointError(f"Non-finite values at {tag}: {nan_count} NaN, {inf_count} Inf")
    return False


def clamp_and_check(tag: str, x: torch.Tensor, min_val: float = -10.0, max_val: float = 10.0) -> torch.Tensor:
    """Clamp tensor values and check for NaN/Inf with optional replacement.

    Args:
        tag: Identifier for this location
        x: Tensor to process
        min_val: Minimum clamp value
        max_val: Maximum clamp value

    Returns:
        Clamped and cleaned tensor
    """
    # First check and report
    had_nonfinite = not assert_finite(f"{tag}_pre_clamp", x, raise_on_fail=False)

    # Replace NaN/Inf
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=max_val, neginf=min_val)
        if DEBUG_FINITE:
            print(f"[NaN-CHECK] {tag}: Replaced non-finite values", flush=True)

    # Clamp to range
    x = torch.clamp(x, min=min_val, max=max_val)

    # Final check
    assert_finite(f"{tag}_post_clamp", x)

    return x


def check_gradients(model: torch.nn.Module, max_grad_norm: float = 100.0) -> dict[str, float]:
    """Check gradients for NaN/Inf and compute statistics.

    Args:
        model: Model to check gradients for
        max_grad_norm: Maximum expected gradient norm

    Returns:
        Dictionary with gradient statistics
    """
    grad_stats = {}
    problematic_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad

            # Check for non-finite
            if not torch.isfinite(grad).all():
                nan_count = torch.isnan(grad).sum().item()
                inf_count = torch.isinf(grad).sum().item()
                problematic_params.append(f"{name}: {nan_count} NaN, {inf_count} Inf")

            # Compute norm
            grad_norm = grad.norm().item()
            grad_stats[name] = grad_norm

            # Check for exploding gradients
            if grad_norm > max_grad_norm:
                problematic_params.append(f"{name}: norm={grad_norm:.2e} > {max_grad_norm}")

    if problematic_params and DEBUG_FINITE:
        print(f"[GRAD-CHECK] Problematic gradients found:")
        for issue in problematic_params:
            print(f"  {issue}")
        print(flush=True)

    return grad_stats
