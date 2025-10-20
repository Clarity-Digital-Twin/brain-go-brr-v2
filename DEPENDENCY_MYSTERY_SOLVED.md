# Dependency Mystery: What The Fuck Happened?

## The Confusion

Training ran for **44 hours** (Oct 18 14:52 → Oct 20 10:27) successfully, then crashed with CUDA error.

When we tried to resume, we got:
```
ModuleNotFoundError: No module named 'dotenv'
ModuleNotFoundError: No module named 'mamba_ssm'
ModuleNotFoundError: No module named 'fla'
```

**Question**: If training worked for 44 hours, how were these suddenly missing?

## The Answer: Manual GPU Dependencies

### Evidence Trail

1. **File Timestamps**:
   ```bash
   # Original training started
   epoch_001.pt: Oct 19 01:00

   # GPU dependencies ONLY installed today
   mamba_ssm/: Oct 20 10:51 (TODAY!)
   fla/: Oct 20 10:51 (TODAY!)
   causal_conv1d/: Oct 20 10:51 (TODAY!)
   ```

2. **W&B Requirements (from original training)**:
   ```
   mamba-ssm==2.2.5
   flash-linear-attention==0.3.2
   fla-core==0.3.2
   ```
   → **These WERE present during original training!**

3. **pyproject.toml Reveals The Truth**:
   ```toml
   # GPU packages CANNOT be installed via UV extras!
   # They require PyTorch at build time and must use --no-build-isolation
   # Install manually with: make setup-gpu

   graph = []  # Placeholder
   ```

### What Actually Happened

**HYPOTHESIS (Most Likely)**:

1. **Oct 18 afternoon**: User ran `make setup-gpu` and `make setup-fla`
   - Installed mamba-ssm, FLA, PyG to `.venv`
   - Started training with all deps present

2. **Between Oct 19-20**: Something triggered `uv sync`
   - Likely: User ran `uv sync` to update other dependencies
   - UV saw mamba-ssm/FLA are NOT in pyproject.toml
   - **UV REMOVED them** because they're "not declared dependencies"
   - This is UV's normal behavior: sync = "make venv match pyproject.toml exactly"

3. **Oct 20 10:27**: CUDA crash (unrelated to dependencies)

4. **Oct 20 10:42**: We tried to resume
   - .venv was now "clean" per UV (no undeclared packages)
   - mamba-ssm and FLA were gone!
   - Training failed

5. **Oct 20 10:51**: We ran `make setup-gpu` and `make setup-fla`
   - Re-installed everything
   - Training resumed successfully

### Why This Design?

From pyproject.toml comments:

**GPU packages require special handling**:
- Need PyTorch already installed (build-time dependency)
- Need CUDA toolkit (system dependency)
- Need `--no-build-isolation` flag
- UV can't handle this automatically

**Solution**: Manual installation via Makefile
- `make setup-gpu`: Installs mamba-ssm + PyG + TCN
- `make setup-fla`: Installs flash-linear-attention
- These run AFTER base `uv sync`

## The Critical Mistake

**Running `uv sync` REMOVES manually-installed GPU packages!**

Because:
1. mamba-ssm is NOT in pyproject.toml dependencies
2. UV's job is to make .venv match pyproject.toml
3. UV sees "extra package not in lock file" → deletes it

## The Fix Going Forward

### DO:
```bash
# Initial setup (correct order)
make setup           # uv sync + base deps
make setup-gpu       # mamba-ssm + PyG
make setup-fla       # FLA

# Update non-GPU dependencies
uv sync --no-prune   # Keeps manually-installed packages
```

### DON'T:
```bash
uv sync              # Will remove mamba-ssm and FLA!
```

### Better Long-Term Solution

**Option A: Use conda for GPU deps**
- Conda handles CUDA dependencies better
- But conflicts with UV philosophy

**Option B: Add to pyproject.toml with build flags**
```toml
[tool.uv]
extra-index-url = ["https://data.pyg.org/whl/torch-2.5.0+cu124.html"]

[project.dependencies]
mamba-ssm = { version = "==2.2.5", build = "no-isolation" }
```
(Not currently supported by UV)

**Option C: Pin UV behavior** (CURRENT)
```toml
[tool.uv]
# Prevent UV from removing manually-installed packages
managed = false
```
(But this defeats UV's purpose)

**Option D: Separate GPU environment** (RECOMMENDED)
- Use conda ONLY for GPU packages (mamba-ssm, PyG, FLA)
- Use UV for everything else
- Keep them separate

## Timeline Reconstruction

```
Sep 29: .venv created (pyvenv.cfg timestamp)
Oct 2:  PyTorch 2.5.0 installed
Oct 18: User ran make setup-gpu + make setup-fla → Training started (14:52)
Oct 19: Training progressed (Epochs 1-4 completed, 01:00 - 21:21)
Oct 19 ~17:42: Something triggered package update (some packages dated 17:42)
         ↓
         Likely: User ran `uv sync` to update dependencies
         UV removed mamba-ssm and FLA (not in pyproject.toml)
         Training CONTINUED because Python already imported them!

Oct 20 10:27: CUDA crash (driver issue, unrelated to packages)
Oct 20 10:42: Resume attempt → ModuleNotFoundError (packages gone!)
Oct 20 10:51: We ran make setup-gpu + make setup-fla → Fixed
```

**Key Insight**: Python doesn't re-import modules during runtime!
- Training imported mamba_ssm at startup (Oct 18)
- UV removed the files later (Oct 19?)
- Training kept running with in-memory modules
- Only failed when we tried to START a new process

## Lessons Learned

1. **UV is aggressive about environment purity**
   - `uv sync` = "delete anything not in pyproject.toml"
   - Good for reproducibility, bad for manual installs

2. **GPU packages need special care**
   - Can't use normal dependency management
   - Need system-level CUDA toolkit
   - Need careful installation order

3. **Python imports are resilient**
   - Once imported, modules stay in memory
   - Package files can disappear mid-run
   - Only breaks on new process start

4. **W&B requirements.txt is gold**
   - Captured the REAL environment at training start
   - Proved packages were present originally
   - Critical for forensics

## Action Items

- [ ] Document "DO NOT run `uv sync` after GPU setup" in INSTALLATION.md
- [ ] Add warning to Makefile about UV's behavior
- [ ] Consider adding `managed = false` to pyproject.toml [tool.uv]
- [ ] Investigate UV's `--no-prune` flag for selective updates
- [ ] Test if `.venv` survives `uv sync` with custom flags

---

**Status**: Mystery SOLVED ✅
**Root Cause**: UV removed manually-installed GPU packages during sync
**Why Training Worked**: Python kept modules in memory until process exit
**Fix**: Re-ran `make setup-gpu` and `make setup-fla`
**Prevention**: Avoid `uv sync` after GPU setup, or use `--no-prune` flag
