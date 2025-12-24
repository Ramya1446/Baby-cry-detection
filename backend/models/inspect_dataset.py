"""
Dataset Inspection Tool
Helps identify data quality issues causing low accuracy
"""

import os
import librosa
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def inspect_dataset(dataset_path="./data/raw/baby-cry-detection"):
    """Inspect dataset for quality issues"""
    
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    
    stats = defaultdict(lambda: {
        'count': 0,
        'durations': [],
        'sample_rates': [],
        'energies': [],
        'silent_files': [],
        'short_files': [],
        'errors': []
    })
    
    print("\n📁 Scanning files...\n")
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️  Category folder not found: {category}")
            continue
        
        files = [f for f in os.listdir(category_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
        
        print(f"Checking {category}: {len(files)} files")
        
        for filename in files:
            filepath = os.path.join(category_path, filename)
            
            try:
                # Load audio
                audio, sr = librosa.load(filepath, sr=None, duration=5.0)
                
                duration = len(audio) / sr
                energy = np.sqrt(np.mean(audio**2))
                
                stats[category]['count'] += 1
                stats[category]['durations'].append(duration)
                stats[category]['sample_rates'].append(sr)
                stats[category]['energies'].append(energy)
                
                # Check for silent files
                if energy < 0.001:
                    stats[category]['silent_files'].append(filename)
                
                # Check for very short files
                if duration < 0.5:
                    stats[category]['short_files'].append(filename)
                
            except Exception as e:
                stats[category]['errors'].append((filename, str(e)))
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n📊 File Counts:")
    for category in categories:
        count = stats[category]['count']
        print(f"  {category:15} {count:3} files")
    
    print("\n⚠️  ISSUES DETECTED:")
    
    issues_found = False
    
    for category in categories:
        cat_stats = stats[category]
        
        # Silent files
        if cat_stats['silent_files']:
            issues_found = True
            print(f"\n  {category} - SILENT FILES ({len(cat_stats['silent_files'])}):")
            for f in cat_stats['silent_files'][:5]:
                print(f"    - {f}")
            if len(cat_stats['silent_files']) > 5:
                print(f"    ... and {len(cat_stats['silent_files']) - 5} more")
        
        # Short files
        if cat_stats['short_files']:
            issues_found = True
            print(f"\n  {category} - SHORT FILES ({len(cat_stats['short_files'])}):")
            for f in cat_stats['short_files'][:5]:
                print(f"    - {f}")
            if len(cat_stats['short_files']) > 5:
                print(f"    ... and {len(cat_stats['short_files']) - 5} more")
        
        # Errors
        if cat_stats['errors']:
            issues_found = True
            print(f"\n  {category} - CORRUPTED FILES ({len(cat_stats['errors'])}):")
            for f, err in cat_stats['errors'][:3]:
                print(f"    - {f}: {err}")
            if len(cat_stats['errors']) > 3:
                print(f"    ... and {len(cat_stats['errors']) - 3} more")
    
    if not issues_found:
        print("  ✅ No major issues detected!")
    
    print("\n📈 STATISTICS:")
    
    for category in categories:
        cat_stats = stats[category]
        
        if cat_stats['durations']:
            print(f"\n  {category}:")
            print(f"    Avg duration: {np.mean(cat_stats['durations']):.2f}s")
            print(f"    Avg energy: {np.mean(cat_stats['energies']):.4f}")
            print(f"    Sample rates: {set(cat_stats['sample_rates'])}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Check balance
    counts = [stats[cat]['count'] for cat in categories]
    min_count = min(counts)
    max_count = max(counts)
    
    if max_count - min_count > 20:
        print("\n⚠️  CLASS IMBALANCE DETECTED")
        print(f"  Min: {min_count}, Max: {max_count}, Diff: {max_count - min_count}")
        print("  Recommendation: Balance classes to ~120 samples each")
    else:
        print("\n✅ Classes are balanced")
    
    # Check silent files
    total_silent = sum(len(stats[cat]['silent_files']) for cat in categories)
    if total_silent > 0:
        print(f"\n⚠️  {total_silent} SILENT/LOW ENERGY FILES")
        print("  Recommendation: Remove or replace these files")
        print("  They might be corrupted or contain no actual cry sounds")
    
    # Check short files
    total_short = sum(len(stats[cat]['short_files']) for cat in categories)
    if total_short > 0:
        print(f"\n⚠️  {total_short} VERY SHORT FILES")
        print("  Recommendation: These may not have enough information")
    
    # Check errors
    total_errors = sum(len(stats[cat]['errors']) for cat in categories)
    if total_errors > 0:
        print(f"\n❌ {total_errors} CORRUPTED FILES")
        print("  Recommendation: Remove these files immediately")
    
    # General recommendations
    print("\n💡 GENERAL RECOMMENDATIONS:")
    print("  1. Each category should have 120-150 high-quality samples")
    print("  2. Recordings should be 2-5 seconds long")
    print("  3. Cries should be clear and audible")
    print("  4. Remove background noise/music")
    print("  5. Ensure correct labeling (wrong labels = bad model)")
    
    print("\n" + "=" * 70)
    
    # Check for possible mislabeling
    print("\n🔍 CHECKING FOR POSSIBLE MISLABELING...")
    
    # Compare energy distributions
    energy_by_cat = {cat: stats[cat]['energies'] for cat in categories if stats[cat]['energies']}
    
    if len(energy_by_cat) >= 2:
        print("  Energy distribution comparison:")
        for cat in categories:
            if stats[cat]['energies']:
                avg_energy = np.mean(stats[cat]['energies'])
                std_energy = np.std(stats[cat]['energies'])
                print(f"    {cat:15} Avg: {avg_energy:.4f} ± {std_energy:.4f}")
        
        print("\n  ✓ If one category has significantly different energy,")
        print("    it might contain different types of audio")
    
    print("\n" + "=" * 70)
    
    # Summary
    print("\n📋 SUMMARY:")
    
    total_files = sum(stats[cat]['count'] for cat in categories)
    total_issues = total_silent + total_short + total_errors
    
    print(f"  Total files: {total_files}")
    print(f"  Total issues: {total_issues}")
    
    if total_issues > total_files * 0.1:
        print(f"\n  ❌ CRITICAL: {total_issues/total_files*100:.1f}% of files have issues!")
        print("     This is likely causing the low accuracy.")
        print("     Clean your dataset before training.")
    elif total_issues > 0:
        print(f"\n  ⚠️  WARNING: {total_issues} problematic files found")
        print("     Consider cleaning these before training")
    else:
        print("\n  ✅ Dataset looks healthy!")
        print("     If accuracy is still low, check:")
        print("       - Are labels correct?")
        print("       - Do files actually contain baby cries?")
        print("       - Is there too much background noise?")
    
    print("\n" + "=" * 70)
    
    return stats

def create_cleanup_script(stats):
    """Generate a script to remove problematic files"""
    
    print("\n" + "=" * 70)
    print("GENERATING CLEANUP SCRIPT")
    print("=" * 70)
    
    cleanup_commands = []
    
    for category in stats.keys():
        cat_stats = stats[category]
        
        # Collect all problematic files
        problem_files = (
            cat_stats['silent_files'] + 
            cat_stats['short_files'] + 
            [f for f, _ in cat_stats['errors']]
        )
        
        if problem_files:
            for filename in problem_files:
                # Windows PowerShell command
                cleanup_commands.append(
                    f'Remove-Item "./data/raw/baby-cry-detection/{category}/{filename}" -ErrorAction SilentlyContinue'
                )
    
    if cleanup_commands:
        script_path = "cleanup_dataset.ps1"
        with open(script_path, 'w') as f:
            f.write("# Auto-generated cleanup script\n")
            f.write("# Removes problematic audio files\n\n")
            for cmd in cleanup_commands:
                f.write(cmd + "\n")
        
        print(f"\n✓ Cleanup script saved: {script_path}")
        print(f"  Total files to remove: {len(cleanup_commands)}")
        print("\n  To run: .\\cleanup_dataset.ps1")
        print("  Or manually delete the files listed above")
    else:
        print("\n✓ No problematic files to remove")

if __name__ == "__main__":
    stats = inspect_dataset()
    
    if stats:
        user_input = input("\n\nGenerate cleanup script? (y/n): ").strip().lower()
        if user_input == 'y':
            create_cleanup_script(stats)
    
    print("\n✅ Inspection complete!")