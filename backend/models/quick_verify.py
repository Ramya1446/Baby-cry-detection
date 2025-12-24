"""
Quick verification script to check if dataset is ready
"""

import os

def quick_verify(dataset_path="./data/raw/baby-cry-detection"):
    """Quick check if dataset is ready for training"""
    
    print("=" * 70)
    print("🍼 QUICK DATASET VERIFICATION")
    print("=" * 70)
    
    print(f"\n📂 Dataset path: {dataset_path}\n")
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset path does not exist!")
        return False
    
    categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    audio_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    
    print("📊 Checking categories:\n")
    print("-" * 70)
    
    results = {}
    total_files = 0
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        
        if os.path.exists(category_path):
            # Count audio files
            audio_files = [f for f in os.listdir(category_path)
                          if f.lower().endswith(audio_extensions)]
            
            count = len(audio_files)
            results[category] = count
            total_files += count
            
            # Visual bar
            bar = '█' * (count // 10) if count > 0 else ''
            status = '✓' if count > 0 else '⚠️'
            
            print(f"{status} {category:15} {count:4} files  {bar}")
            
            # Show first file as example
            if count > 0:
                example_file = audio_files[0]
                file_size = os.path.getsize(os.path.join(category_path, example_file)) / 1024
                print(f"   Example: {example_file[:50]} ({file_size:.1f} KB)")
        else:
            print(f"❌ {category:15}    - (folder missing)")
            results[category] = 0
    
    print("-" * 70)
    print(f"   {'TOTAL':15} {total_files:4} files")
    print()
    
    # Check balance
    if total_files > 0:
        counts = list(results.values())
        min_count = min(counts)
        max_count = max(counts)
        
        if min_count > 0:
            imbalance = max_count / min_count
            
            print("⚖️  Balance check:")
            print(f"   Min: {min_count} | Max: {max_count} | Ratio: {imbalance:.2f}x")
            
            if imbalance <= 1.5:
                print("   ✅ Dataset is well balanced!")
            elif imbalance <= 2.0:
                print("   ⚠️  Slight imbalance (acceptable)")
            else:
                print("   ⚠️  Significant imbalance - consider balancing")
    
    print("\n" + "=" * 70)
    
    # Final verdict
    if total_files == 0:
        print("❌ FAIL: No audio files found!")
        print("\n💡 Solutions:")
        print("   • Make sure audio files are IN the category folders")
        print("   • Check if files have correct extensions (.wav, .mp3, etc.)")
        print("   • Verify you extracted/copied the files correctly")
        return False
    
    elif total_files < 100:
        print(f"⚠️  WARNING: Only {total_files} files found")
        print("   This might be too small for good training")
        print("   Recommend at least 100-200 files per category")
        return True
    
    else:
        print(f"✅ READY! Found {total_files} audio files")
        print("\n🚀 NEXT STEPS:")
        print("   1. Delete old preprocessed data:")
        print("      Remove-Item ./data/processed -Recurse -Force")
        print("      (or: rm -rf ./data/processed)")
        print()
        print("   2. Delete old models:")
        print("      Remove-Item ./saved_models/*.keras")
        print("      (or: rm ./saved_models/*.keras)")
        print()
        print("   3. Run training:")
        print("      python train_baby_cry_FIXED.py")
        print()
        print("=" * 70)
        return True

if __name__ == "__main__":
    quick_verify()