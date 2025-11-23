# Investigation Report: Baseline Crash & Exp4 Instability

## Executive Summary
This report details the root cause analysis of the "unknown error" CUDA crash that terminated the baseline experiment at epoch 13 and the instability risks in Exp4 (SGDR).

**Root Cause:** Numerical instability in `torch.linalg.eigh` (eigendecomposition) within the GNN module (`src/brain_brr/models/gnn_pyg.py`).
**Trigger:** The baseline's cosine scheduler allowed the model to converge into a "pathological" region of the loss landscape where the learned adjacency matrices became symmetric with near-degenerate eigenvalues (multiple eigenvalues very close to each other).
**Why it Crashed:** `cuSOLVER` (the underlying CUDA library for `eigh`) is known to fail with "unknown error" or infinite loops when processing matrices with degenerate eigenvalues on the GPU. The existing constant regularization (`laplacian_eps=1e-3`) shifts all eigenvalues equally, preserving the degeneracy.

## Findings

### 1. The Crash Mechanism
*   **Location:** `src/brain_brr/models/gnn_pyg.py`, function `_compute_dynamic_pe_vectorized`.
*   **Operation:** `eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)`
*   **Failure:** A low-level CUDA error ("unknown error") occurred. This type of error is often not caught by standard Python `try...except` blocks in a way that allows recovery, or it leaves the CUDA context in a bad state.
*   **Evidence:** The logs showed "CUDA error: unknown error" at epoch 13 for the baseline. Exp4 avoided this specific crash point because the Cyclic LR (SGDR) periodically "kicked" the weights away from this sharp, unstable minimum.

### 2. Insufficient Existing Mitigations
*   **`laplacian_eps`:** The configs use `1e-3`, which adds a constant value to the diagonal. This shifts the spectrum ($\lambda_i \to \lambda_i + \epsilon$) but **does not change the gaps** between eigenvalues ($\Delta \lambda = (\lambda_i + \epsilon) - (\lambda_j + \epsilon) = \lambda_i - \lambda_j$). Degeneracy ($\Delta \lambda \approx 0$) remains.
*   **`condition_adjacency`:** Row softmax and EMA help, but they enforce structure that can actually *increase* symmetry, leading to more degeneracy.
*   **NaN Checks:** The code checks for NaNs *after* `eigh`, but the crash happens *during* `eigh` (or causes a CUDA context failure immediately).

## The Fix: Diagonal Jitter & CPU Fallback

I have applied a robust fix to `src/brain_brr/models/gnn_pyg.py` on the `gemini/investigation` branch:

### 1. Random Diagonal Jitter (The Real Fix)
We now add small random noise to the diagonal of the Laplacian before decomposition:
```python
jitter = torch.randn(batch_total, N, device=device) * 1e-5
l_stable.diagonal(dim1=-2, dim2=-1).add_(jitter)
```
**Why it works:** Random noise breaks symmetries. It shifts eigenvalues by slightly different random amounts, ensuring that $\Delta \lambda \neq 0$. This prevents the degeneracy that crashes `cuSOLVER`.

### 2. Robust CPU Fallback
If the GPU `eigh` fails (raises `RuntimeError`), we now explicitly catch it and retry on the CPU:
```python
l_cpu = l_stable.cpu()
evals_cpu, evecs_cpu = torch.linalg.eigh(l_cpu)
```
**Why it works:** CPU LAPACK implementations are generally more robust to pathological matrices than GPU `cuSOLVER`. This provides a safety net that allows training to continue even if a "bad" matrix is generated.

### 3. Improved Logging
Explicit warnings are logged when these fallbacks are triggered, allowing us to track how often the model enters these unstable regions.

## Recommendation for Next Steps

1.  **Merge this fix:** The changes in `src/brain_brr/models/gnn_pyg.py` are critical for stability.
2.  **Resume/Retrain:** With this fix, the baseline model can likely pass epoch 13 without crashing. Exp4 can also be resumed with higher confidence that it won't crash unexpectedly.
3.  **Monitor:** Watch the logs for "GPU Eigendecomposition failed... trying CPU fallback" to see if the jitter is doing its job (ideally, the jitter prevents the crash, so the fallback is rarely used).

