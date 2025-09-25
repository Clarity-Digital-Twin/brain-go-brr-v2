"""TUSZ official split handling - PATIENT-DISJOINT splits.

This module ensures we use TUSZ's official train/dev/eval splits correctly:
- train/ (579 patients) for training
- dev/ (53 patients) for validation and hyperparameter tuning
- eval/ (43 patients) for final testing ONLY

NO PATIENT LEAKAGE ALLOWED!
"""

import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def extract_patient_id(filepath: Path) -> str:
    """Extract patient ID from TUSZ filepath.

    TUSZ format: /path/to/split/PATIENT_ID/SESSION_ID/PATIENT_SESSION_TOKEN.edf
    Example: train/aaaaagxr/s018_t000/aaaaagxr_s018_t000.edf
    """
    # Patient ID is the first part of the filename before underscore
    return filepath.stem.split("_")[0]


def get_tusz_official_splits(
    data_root: Path,
    split: str = "train",
    verbose: bool = True
) -> Tuple[List[Path], List[Path], Set[str]]:
    """Get TUSZ official train/dev/eval splits.

    Args:
        data_root: Root directory containing train/, dev/, eval/ subdirectories
        split: Which split to get ("train", "dev", or "eval")
        verbose: Print split statistics

    Returns:
        Tuple of (edf_files, csv_files, patient_ids)
    """
    if split not in ["train", "dev", "eval"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'eval'")

    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Find all EDF files
    edf_files = sorted(split_dir.glob("**/*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {split_dir}")

    # Pair with CSV annotation files
    csv_files = [f.with_suffix(".csv") for f in edf_files]

    # Extract unique patient IDs
    patient_ids = set()
    for edf_file in edf_files:
        patient_id = extract_patient_id(edf_file)
        patient_ids.add(patient_id)

    if verbose:
        n_with_csv = sum(1 for f in csv_files if f.exists())
        print(f"\n[TUSZ {split.upper()} Split]")
        print(f"  Patients: {len(patient_ids)}")
        print(f"  EDF files: {len(edf_files)}")
        print(f"  CSV annotations: {n_with_csv}/{len(csv_files)}")
        print(f"  Example patients: {sorted(patient_ids)[:5]}")

    return edf_files, csv_files, patient_ids


def validate_patient_disjointness(
    train_patients: Set[str],
    dev_patients: Set[str],
    eval_patients: Set[str] = None
) -> None:
    """Validate that patient sets are completely disjoint.

    CRITICAL: This prevents patient leakage between splits!

    Args:
        train_patients: Set of patient IDs in training
        dev_patients: Set of patient IDs in validation/dev
        eval_patients: Optional set of patient IDs in eval

    Raises:
        ValueError: If any patient appears in multiple splits
    """
    # Check train vs dev
    train_dev_overlap = train_patients & dev_patients
    if train_dev_overlap:
        raise ValueError(
            f"PATIENT LEAKAGE DETECTED! {len(train_dev_overlap)} patients in both train and dev:\n"
            f"Examples: {sorted(train_dev_overlap)[:10]}\n"
            f"This invalidates all validation metrics!"
        )

    # Check eval if provided
    if eval_patients:
        train_eval_overlap = train_patients & eval_patients
        if train_eval_overlap:
            raise ValueError(
                f"PATIENT LEAKAGE! {len(train_eval_overlap)} patients in both train and eval:\n"
                f"Examples: {sorted(train_eval_overlap)[:10]}"
            )

        dev_eval_overlap = dev_patients & eval_patients
        if dev_eval_overlap:
            raise ValueError(
                f"PATIENT LEAKAGE! {len(dev_eval_overlap)} patients in both dev and eval:\n"
                f"Examples: {sorted(dev_eval_overlap)[:10]}"
            )

    print("\n✅ PATIENT DISJOINTNESS VALIDATED - No leakage detected!")
    print(f"   Train: {len(train_patients)} patients")
    print(f"   Dev: {len(dev_patients)} patients")
    if eval_patients:
        print(f"   Eval: {len(eval_patients)} patients")


def load_tusz_for_training(
    data_root: Path,
    use_eval: bool = False,
    verbose: bool = True
) -> Dict[str, Tuple[List[Path], List[Path]]]:
    """Load TUSZ with official splits for training.

    Standard protocol:
    - Train on train/
    - Validate on dev/
    - Test on eval/ (only if use_eval=True, for final evaluation)

    Args:
        data_root: Root directory with train/, dev/, eval/ subdirs
        use_eval: Whether to load eval set (ONLY for final testing!)
        verbose: Print statistics

    Returns:
        Dictionary with "train", "dev" (and optionally "eval") splits,
        each containing (edf_files, csv_files)
    """
    splits = {}
    all_patients = {}

    # Load train and dev (always needed)
    for split_name in ["train", "dev"]:
        edf_files, csv_files, patient_ids = get_tusz_official_splits(
            data_root, split_name, verbose=verbose
        )
        splits[split_name] = (edf_files, csv_files)
        all_patients[split_name] = patient_ids

    # Optionally load eval (ONLY for final testing)
    if use_eval:
        warnings.warn(
            "⚠️  Loading EVAL set! This should ONLY be done for final testing!\n"
            "   Do NOT use eval for hyperparameter tuning or model selection!",
            stacklevel=2
        )
        edf_files, csv_files, patient_ids = get_tusz_official_splits(
            data_root, "eval", verbose=verbose
        )
        splits["eval"] = (edf_files, csv_files)
        all_patients["eval"] = patient_ids

    # CRITICAL: Validate patient disjointness
    validate_patient_disjointness(
        all_patients["train"],
        all_patients["dev"],
        all_patients.get("eval")
    )

    if verbose:
        print("\n" + "="*60)
        print("TUSZ OFFICIAL SPLITS LOADED SUCCESSFULLY")
        print("Protocol: Train on train/ | Validate on dev/ | Test on eval/")
        print("="*60)

    return splits


# For backward compatibility, provide a function that mimics the old interface
# but uses proper splits
def get_train_val_splits(
    data_root: Path,
    validation_split: float = None,  # IGNORED - we use official dev
    split_seed: int = None,  # IGNORED - we use official splits
    verbose: bool = True
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Get train/val splits using TUSZ official splits.

    NOTE: validation_split and split_seed are IGNORED.
    We use the official TUSZ train/ and dev/ splits.

    Returns:
        (train_edf_files, train_csv_files, val_edf_files, val_csv_files)
    """
    if validation_split is not None or split_seed is not None:
        warnings.warn(
            "validation_split and split_seed are IGNORED!\n"
            "Using official TUSZ train/ and dev/ splits for patient disjointness.",
            stacklevel=2
        )

    splits = load_tusz_for_training(data_root, use_eval=False, verbose=verbose)

    train_edf, train_csv = splits["train"]
    val_edf, val_csv = splits["dev"]  # Use dev as validation

    return train_edf, train_csv, val_edf, val_csv