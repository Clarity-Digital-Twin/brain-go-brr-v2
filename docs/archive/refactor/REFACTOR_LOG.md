# Refactor Log - Zero Hallucination Guarantee

## Strategy: Each file migration follows EXACT steps

1. **COPY** file to new location (preserve original)
2. **ADD** compatibility shim to old location
3. **TEST** both import paths work
4. **COMMIT** only after tests pass

## Completed Migrations

### ✅ constants.py (21:26)
```bash
# Step 1: Copy
cp src/experiment/constants.py src/brain_brr/constants.py

# Step 2: Add shim to old file
# Added deprecation warning + import from new location

# Step 3: Test
python -c "from src.experiment.constants import CHANNEL_NAMES_10_20"  # Works with warning
python -c "from src.brain_brr.constants import CHANNEL_NAMES_10_20"   # Works clean

# Step 4: Commit
git commit -m "refactor: migrate constants.py"
```

### 🔄 events.py (In Progress)
```bash
# Step 1: Copy (DONE)
cp src/experiment/events.py src/brain_brr/events/events.py

# Step 2: Add shim (TODO)
# Need to update src/experiment/events.py with deprecation

# Step 3: Test (TODO)
# Will verify both imports work

# Step 4: Commit (TODO)
```

## Verification Commands

```bash
# Check what's changed
git status --short

# Compare files
diff -u src/experiment/FILE.py src/brain_brr/MODULE/FILE.py

# Test imports
python -c "from old.path import Thing; from new.path import Thing"

# Run tests
pytest tests/test_smoke.py -v
```

## Safety Guarantees

1. **Original files untouched** - Only add import shims
2. **Git tracks everything** - Can revert any change
3. **Tests must pass** - No commit without green tests
4. **Deprecation warnings** - Users know to update imports
5. **Zero functionality change** - Pure reorganization

## Next Files to Migrate

- [ ] events.py → brain_brr/events/
- [ ] export.py → brain_brr/events/export.py
- [ ] postprocess.py → brain_brr/post/
- [ ] models.py → brain_brr/models/ (split into 5)
- [ ] data.py → brain_brr/data/ (split into 4)
- [ ] pipeline.py → brain_brr/train/ (split into 6)

Each migration is ATOMIC - if it fails, we stop and fix.