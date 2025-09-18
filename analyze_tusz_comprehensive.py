#!/usr/bin/env python3
"""Comprehensive TUSZ dataset analysis - understand EVERYTHING about channels and montages."""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
import mne
import sys
from typing import Dict, List, Any

# Suppress MNE warnings
mne.set_log_level('ERROR')

# Our required 19 channels
REQUIRED_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

def analyze_edf_file(edf_path: Path) -> Dict[str, Any]:
    """Analyze a single EDF file for channel information."""
    result = {
        'file': str(edf_path),
        'subset': edf_path.parts[-6],  # dev/train/eval
        'patient': edf_path.parts[-4],
        'session': edf_path.parts[-3],
        'montage': edf_path.parts[-2],
        'filename': edf_path.name,
        'error': None
    }

    try:
        # Load file header only (fast)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        # Basic info
        result['n_channels'] = len(raw.ch_names)
        result['sfreq'] = raw.info['sfreq']
        result['duration'] = raw.n_times / raw.info['sfreq']
        result['channels_raw'] = raw.ch_names

        # Check for our required channels (with canonicalization)
        channels_upper = [ch.upper() for ch in raw.ch_names]
        found_required = []
        missing_required = []

        for req in REQUIRED_CHANNELS:
            # Check various possible formats
            found = False
            for ch_raw, ch_upper in zip(raw.ch_names, channels_upper):
                # Check if required channel is in the raw name
                if req.upper() in ch_upper:
                    # More specific: check it's not part of another channel
                    # e.g., "FP1" should match "EEG FP1-REF" but not "FP10"
                    parts = ch_upper.replace('EEG', '').replace('-', ' ').replace('_', ' ').split()
                    if req.upper() in parts:
                        found_required.append(f"{req} -> {ch_raw}")
                        found = True
                        break

            if not found:
                missing_required.append(req)

        result['found_required'] = found_required
        result['missing_required'] = missing_required
        result['has_all_19'] = len(missing_required) == 0

        # Special focus on Fz, Cz, Pz (midline channels)
        midline_info = {}
        for midline in ['FZ', 'CZ', 'PZ']:
            matches = [ch for ch in raw.ch_names if midline in ch.upper()]
            midline_info[midline] = matches
        result['midline_channels'] = midline_info

        # Detect reference type from channel names
        if any('-REF' in ch for ch in raw.ch_names):
            result['reference'] = 'REF'
        elif any('-LE' in ch for ch in raw.ch_names):
            result['reference'] = 'LE'
        elif any('-AR' in ch for ch in raw.ch_names):
            result['reference'] = 'AR'
        else:
            result['reference'] = 'UNKNOWN'

    except Exception as e:
        result['error'] = str(e)

    return result

def main():
    """Run comprehensive analysis on all TUSZ data."""

    # Find all EDF files
    data_root = Path('data/tusz/edf')
    all_results = []

    for subset in ['dev', 'train', 'eval']:
        subset_path = data_root / subset
        if not subset_path.exists():
            print(f"⚠️  {subset} directory not found, skipping...")
            continue

        edf_files = list(subset_path.glob('**/*.edf'))
        print(f"\n📊 Analyzing {subset}: {len(edf_files)} files")

        # Analyze each file
        for i, edf_file in enumerate(edf_files):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(edf_files)} ({i*100/len(edf_files):.1f}%)")

            result = analyze_edf_file(edf_file)
            all_results.append(result)

            # Report problems immediately
            if result['missing_required'] and 'Fz' in result['missing_required']:
                print(f"  ⚠️ Missing Fz/Pz: {edf_file.name} in {result['montage']}")

    # Save raw results to CSV
    csv_file = 'tusz_channel_analysis.csv'
    print(f"\n💾 Saving raw results to {csv_file}")

    with open(csv_file, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    # Generate statistics
    print("\n📈 Generating statistics...")
    stats = generate_statistics(all_results)

    # Save statistics to JSON
    json_file = 'tusz_channel_statistics.json'
    with open(json_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"💾 Statistics saved to {json_file}")

    # Generate human-readable report
    generate_report(all_results, stats)

    print("\n✅ Analysis complete!")
    print(f"   - Raw data: {csv_file}")
    print(f"   - Statistics: {json_file}")
    print(f"   - Report: tusz_channel_report.md")

def generate_statistics(results: List[Dict]) -> Dict:
    """Generate statistics from analysis results."""
    stats = {
        'total_files': len(results),
        'by_subset': Counter(r['subset'] for r in results),
        'by_montage': Counter(r['montage'] for r in results),
        'by_reference': Counter(r['reference'] for r in results if not r.get('error')),
        'files_with_all_19': sum(1 for r in results if r.get('has_all_19')),
        'files_with_errors': sum(1 for r in results if r.get('error')),
    }

    # Missing channels frequency
    missing_freq = Counter()
    for r in results:
        if r.get('missing_required'):
            for ch in r['missing_required']:
                missing_freq[ch] += 1

    stats['missing_channel_frequency'] = dict(missing_freq)
    stats['percent_with_all_19'] = (stats['files_with_all_19'] / stats['total_files'] * 100) if stats['total_files'] > 0 else 0

    # Montage-specific stats
    montage_stats = {}
    for montage in stats['by_montage'].keys():
        montage_results = [r for r in results if r['montage'] == montage]
        montage_stats[montage] = {
            'total': len(montage_results),
            'with_all_19': sum(1 for r in montage_results if r.get('has_all_19')),
            'common_missing': Counter(ch for r in montage_results for ch in r.get('missing_required', []))
        }
    stats['montage_specific'] = montage_stats

    return stats

def generate_report(results: List[Dict], stats: Dict):
    """Generate human-readable markdown report."""

    with open('tusz_channel_report.md', 'w') as f:
        f.write("# TUSZ Channel Analysis Report\n\n")
        f.write(f"**Total Files Analyzed**: {stats['total_files']}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Files with all 19 required channels: {stats['files_with_all_19']} ({stats['percent_with_all_19']:.1f}%)\n")
        f.write(f"- Files with errors: {stats['files_with_errors']}\n\n")

        f.write("## By Dataset\n\n")
        for subset, count in stats['by_subset'].items():
            f.write(f"- {subset}: {count} files\n")

        f.write("\n## By Montage\n\n")
        for montage, mstats in stats['montage_specific'].items():
            percent = (mstats['with_all_19'] / mstats['total'] * 100) if mstats['total'] > 0 else 0
            f.write(f"### {montage}\n")
            f.write(f"- Total: {mstats['total']} files\n")
            f.write(f"- With all 19: {mstats['with_all_19']} ({percent:.1f}%)\n")
            if mstats['common_missing']:
                f.write(f"- Common missing: {dict(mstats['common_missing'].most_common(5))}\n")
            f.write("\n")

        f.write("## Most Frequently Missing Channels\n\n")
        for ch, count in sorted(stats['missing_channel_frequency'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {ch}: {count} files\n")

        # Problem files
        f.write("\n## Problem Files (Missing Fz or Pz)\n\n")
        problem_files = [r for r in results if r.get('missing_required') and ('Fz' in r['missing_required'] or 'Pz' in r['missing_required'])]
        for pf in problem_files[:10]:  # First 10
            f.write(f"- {pf['file']}\n")
            f.write(f"  - Montage: {pf['montage']}\n")
            f.write(f"  - Missing: {pf['missing_required']}\n")
            f.write(f"  - Midline channels found: {pf.get('midline_channels', {})}\n\n")

if __name__ == "__main__":
    main()