#!/usr/bin/env python3
"""Apply PR #708 fix to mamba-ssm Triton kernels.

This patches mamba-ssm source OR installed package to fix the int32 pointer overflow
bug that causes XID 31 crashes on A100 with large batch sizes.

PR #708: https://github.com/state-spaces/mamba/pull/708
Author: younesbelkada (HuggingFace)
Issue: https://github.com/state-spaces/mamba/issues/503

The fix: Cast all `tl.program_id()` calls to `tl.int64` to prevent pointer overflow
when computing memory addresses for large batches (e.g., batch=64, d_model=512).
"""

import argparse
import sys
from pathlib import Path


def patch_triton_files(triton_dir: Path) -> int:
    """Patch Triton kernel files in the given directory.

    Returns:
        Number of patches applied
    """
    files_to_patch = [
        triton_dir / "ssd_chunk_scan.py",
        triton_dir / "ssd_chunk_state.py",
        triton_dir / "ssd_state_passing.py",
        triton_dir / "ssd_combined.py",
    ]

    print(f"\n📝 Patching {len(files_to_patch)} Triton kernel files...")

    total_changes = 0

    for file_path in files_to_patch:
        if not file_path.exists():
            print(f"⚠️  {file_path.name}: Not found (skipping)")
            continue

        content = file_path.read_text()
        original_content = content

        content = content.replace(
            "tl.program_id(axis=0)",
            "tl.program_id(axis=0).to(tl.int64)"
        )
        content = content.replace(
            "tl.program_id(axis=1)",
            "tl.program_id(axis=1).to(tl.int64)"
        )
        content = content.replace(
            "tl.program_id(axis=2)",
            "tl.program_id(axis=2).to(tl.int64)"
        )

        bytes_added = len(content) - len(original_content)
        occurrences = bytes_added // 15

        if bytes_added == 0:
            print(f"⚠️  {file_path.name}: No changes (already patched)")
        else:
            file_path.write_text(content)
            print(f"✅ {file_path.name}: Patched ~{occurrences} tl.program_id() calls")
            total_changes += occurrences

    return total_changes


def apply_pr708_fix(source_dir: Path = None):
    """Apply PR #708 int64 pointer casting fix to Triton kernels.

    Args:
        source_dir: If provided, patch source directory. Otherwise patch installed package.
    """

    print("=" * 70)
    print("🔧 Applying PR #708 Fix to mamba-ssm")
    print("=" * 70)

    if source_dir:
        print(f"\n📦 Patching SOURCE directory: {source_dir}")
        triton_dir = source_dir / "mamba_ssm" / "ops" / "triton"

        if not triton_dir.exists():
            print(f"❌ Triton directory not found: {triton_dir}")
            return False

        print(f"✓ Triton kernels:  {triton_dir}")

    else:
        print(f"\n📦 Patching INSTALLED package")
        try:
            import mamba_ssm
            mamba_dir = Path(mamba_ssm.__file__).parent
            triton_dir = mamba_dir / "ops" / "triton"
            print(f"✓ Found mamba_ssm: {mamba_dir}")
            print(f"✓ Triton kernels:  {triton_dir}")
        except ImportError as e:
            print(f"❌ mamba_ssm not installed: {e}")
            return False

        if not triton_dir.exists():
            print(f"❌ Triton directory not found: {triton_dir}")
            return False

    total_changes = patch_triton_files(triton_dir)

    if total_changes == 0:
        print("\n⚠️  WARNING: No changes applied! Kernel files may already be patched.")
        print("    This is OK if running patch script multiple times.")
    else:
        print(f"\n✅ Successfully patched ~{total_changes} pointer operations!")

    if not source_dir:
        print("\n🧪 Verifying patched mamba_ssm imports...")
        try:
            modules_to_reload = [
                'mamba_ssm.ops.triton.ssd_chunk_scan',
                'mamba_ssm.ops.triton.ssd_chunk_state',
                'mamba_ssm.ops.triton.ssd_state_passing',
                'mamba_ssm.ops.triton.ssd_combined',
            ]
            for mod in modules_to_reload:
                if mod in sys.modules:
                    del sys.modules[mod]

            from mamba_ssm import Mamba2
            print("✅ mamba_ssm reimports successfully with PR #708 fix!")

        except ImportError as e:
            print(f"❌ Failed to reimport mamba_ssm: {e}")
            return False

    print("\n" + "=" * 70)
    print("✅ PR #708 APPLICATION COMPLETE")
    print("=" * 70)
    print("\nWhat was patched:")
    print("  • All tl.program_id(axis=*) calls now cast to int64")
    print("  • Prevents pointer overflow with large batches (e.g., 64x512)")
    print("  • Fixes XID 31 MMU Faults on A100/A6000/H100")
    print("\nContext:")
    print("  • PR #708: https://github.com/state-spaces/mamba/pull/708")
    print("  • This is a TEMPORARY fix until PR #708 merges upstream")
    print("  • Remove this patch once mamba-ssm releases version with PR #708")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply PR #708 fix to mamba-ssm")
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to mamba-ssm source directory (if patching before build)"
    )
    args = parser.parse_args()

    success = apply_pr708_fix(source_dir=args.source)
    sys.exit(0 if success else 1)