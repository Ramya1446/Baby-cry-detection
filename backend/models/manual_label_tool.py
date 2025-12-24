"""
Manual Audio Labeling Tool
===========================
Interactive tool to label your real-life baby cry recordings
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import json
from datetime import datetime

class AudioLabeler:
    def __init__(self):
        self.categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
        self.labeled_data = []
        self.output_dir = "./data/manual_labeled/"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        for cat in self.categories:
            os.makedirs(os.path.join(self.output_dir, cat), exist_ok=True)
    
    def play_audio(self, filepath):
        """Play audio file"""
        try:
            audio = AudioSegment.from_file(filepath)
            print(f"  🔊 Playing audio... (duration: {len(audio)/1000:.1f}s)")
            play(audio)
        except Exception as e:
            print(f"  ⚠️ Couldn't play audio: {e}")
            print("  You can manually open the file to listen")
    
    def show_audio_info(self, filepath):
        """Display audio characteristics"""
        try:
            audio, sr = librosa.load(filepath, sr=None)
            duration = len(audio) / sr
            energy = np.sqrt(np.mean(audio**2))
            
            print(f"\n  📊 Audio Info:")
            print(f"    Duration: {duration:.2f}s")
            print(f"    Sample Rate: {sr} Hz")
            print(f"    Energy: {energy:.4f}")
            print(f"    Samples: {len(audio)}")
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            if len(pitch_values) > 0:
                valid = pitch_values[pitch_values > 0]
                if len(valid) > 0:
                    print(f"    Avg Pitch: {np.mean(valid):.1f} Hz")
        except Exception as e:
            print(f"  ⚠️ Error analyzing audio: {e}")
    
    def label_audio(self, filepath):
        """Label a single audio file"""
        print("\n" + "=" * 70)
        print(f"📁 File: {os.path.basename(filepath)}")
        print("=" * 70)
        
        self.show_audio_info(filepath)
        
        print("\n🎵 Playing audio...")
        self.play_audio(filepath)
        
        print("\n🏷️  Available Categories:")
        for i, cat in enumerate(self.categories, 1):
            print(f"  {i}. {cat.replace('_', ' ').title()}")
        print(f"  6. Skip this file")
        print(f"  7. Play again")
        print(f"  0. Exit labeling")
        
        while True:
            choice = input("\nYour choice (1-7): ").strip()
            
            if choice == '0':
                return 'exit'
            elif choice == '6':
                print("  ⏭️  Skipped")
                return 'skip'
            elif choice == '7':
                print("\n  🔊 Replaying...")
                self.play_audio(filepath)
                continue
            elif choice in ['1', '2', '3', '4', '5']:
                idx = int(choice) - 1
                selected_category = self.categories[idx]
                
                # Confirm
                confirm = input(f"\n  Confirm label as '{selected_category}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_category
                else:
                    print("  Cancelled, choose again")
            else:
                print("  ❌ Invalid choice, try again")
    
    def copy_labeled_file(self, source_path, category, index):
        """Copy file to labeled directory"""
        ext = os.path.splitext(source_path)[1]
        new_filename = f"manual_{index:03d}{ext}"
        dest_path = os.path.join(self.output_dir, category, new_filename)
        
        # Copy file
        audio, sr = librosa.load(source_path, sr=None)
        sf.write(dest_path, audio, sr)
        
        return dest_path
    
    def process_directory(self, input_dir):
        """Process all audio files in a directory"""
        print("\n" + "=" * 70)
        print("🎤 MANUAL AUDIO LABELING")
        print("=" * 70)
        
        # Find audio files
        audio_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a')):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"\n❌ No audio files found in: {input_dir}")
            return
        
        print(f"\n✓ Found {len(audio_files)} audio files")
        print("\n📝 Instructions:")
        print("  - Listen to each audio carefully")
        print("  - Select the correct category")
        print("  - You can replay if needed")
        print("  - Press Enter to start...\n")
        input()
        
        labeled_count = 0
        skipped_count = 0
        
        for i, filepath in enumerate(audio_files, 1):
            print(f"\n\n[{i}/{len(audio_files)}]")
            
            result = self.label_audio(filepath)
            
            if result == 'exit':
                print("\n  👋 Exiting labeling process...")
                break
            elif result == 'skip':
                skipped_count += 1
                continue
            else:
                # Copy to labeled directory
                dest_path = self.copy_labeled_file(filepath, result, labeled_count + 1)
                
                # Save metadata
                self.labeled_data.append({
                    'original_file': filepath,
                    'labeled_file': dest_path,
                    'category': result,
                    'timestamp': datetime.now().isoformat()
                })
                
                labeled_count += 1
                print(f"  ✅ Labeled as '{result}' and saved!")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'labeling_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.labeled_data, f, indent=2)
        
        print("\n" + "=" * 70)
        print("LABELING SUMMARY")
        print("=" * 70)
        print(f"\n  ✅ Labeled: {labeled_count} files")
        print(f"  ⏭️  Skipped: {skipped_count} files")
        print(f"\n  📁 Labeled files saved to: {self.output_dir}")
        print(f"  📄 Metadata saved to: {metadata_path}")
        
        # Show distribution
        if self.labeled_data:
            print("\n  Distribution:")
            from collections import Counter
            dist = Counter([item['category'] for item in self.labeled_data])
            for cat, count in dist.items():
                print(f"    {cat:15} {count} files")
        
        print("\n" + "=" * 70)

def main():
    print("\n" + "=" * 70)
    print("🎤 MANUAL AUDIO LABELING TOOL")
    print("=" * 70)
    
    print("\n📂 Where are your audio files?")
    print("  Enter the directory path (or press Enter for current directory)")
    
    input_dir = input("\nDirectory path: ").strip()
    
    if not input_dir:
        input_dir = "."
    
    if not os.path.exists(input_dir):
        print(f"\n❌ Directory not found: {input_dir}")
        return
    
    labeler = AudioLabeler()
    labeler.process_directory(input_dir)
    
    print("\n✅ Done! You can now use these labeled files for fine-tuning.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()