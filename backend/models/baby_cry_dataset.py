import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pickle
import warnings
from collections import Counter
import json
warnings.filterwarnings('ignore')

class BabyCryDatasetProcessor:
    def __init__(self, dataset_path, target_sr=22050, duration=3.0):
        """
        Dataset processor for baby-cry-detection dataset from Kaggle
        
        Args:
            dataset_path: Path to the baby-cry-detection dataset
            target_sr: Target sampling rate
            duration: Fixed duration for audio clips
        """
        self.dataset_path = dataset_path
        self.target_sr = target_sr
        self.duration = duration
        self.max_pad_len = int(target_sr * duration)
        self.label_encoder = LabelEncoder()
        
        # Define cry categories based on the dataset
        # This dataset typically has categories like: belly_pain, burping, discomfort, hungry, tired
        self.cry_categories = {}
        self.category_description = {}
        
    def discover_dataset_structure(self):
        """
        Automatically discover the dataset structure and categories
        
        Returns:
            categories: Dictionary mapping category names to indices
        """
        print("Discovering dataset structure...")
        
        categories_found = set()
        file_count = 0
        sample_files = []
        
        # Walk through dataset to find categories
        for root, dirs, files in os.walk(self.dataset_path):
            # Check directory names for categories
            for dir_name in dirs:
                dir_lower = dir_name.lower().replace('_', ' ').replace('-', ' ')
                categories_found.add(dir_name)
            
            # Check audio files
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    file_count += 1
                    if len(sample_files) < 5:
                        sample_files.append(os.path.join(root, file))
        
        print(f"Found {file_count} audio files")
        print(f"Discovered categories: {sorted(categories_found)}")
        print(f"Sample files: {[os.path.basename(f) for f in sample_files[:3]]}")
        
        # Create category mapping
        if categories_found:
            self.cry_categories = {cat: idx for idx, cat in enumerate(sorted(categories_found))}
        else:
            # Default categories if structure is different
            self.cry_categories = {
                'belly_pain': 0,
                'burping': 1,
                'discomfort': 2,
                'hungry': 3,
                'tired': 4
            }
        
        # Category descriptions for recommendations
        self.category_description = {
            'belly_pain': 'Baby is experiencing stomach/digestive discomfort',
            'burping': 'Baby needs to burp or has gas',
            'discomfort': 'Baby is uncomfortable (temperature, position, etc.)',
            'hungry': 'Baby needs feeding',
            'tired': 'Baby is sleepy and needs rest',
            'pain': 'Baby is in pain',
            'attention': 'Baby wants attention or interaction',
            'diaper': 'Baby needs diaper change'
        }
        
        return self.cry_categories
    
    def extract_label_from_path(self, file_path):
        """
        Extract cry category label from file path
        
        Args:
            file_path: Path to audio file
            
        Returns:
            label: Cry category index or None
        """
        file_path_lower = file_path.lower()
        file_name = os.path.basename(file_path_lower)
        dir_name = os.path.basename(os.path.dirname(file_path_lower))
        
        # Check each category
        for category, label in self.cry_categories.items():
            category_clean = category.lower().replace('_', ' ').replace('-', ' ')
            
            # Check in directory name
            if category.lower() in dir_name or category_clean in dir_name:
                return label
            
            # Check in file name
            if category.lower() in file_name or category_clean in file_name:
                return label
            
            # Check variations
            if category == 'hungry' and ('hunger' in file_name or 'feed' in file_name):
                return label
            if category == 'tired' and ('sleep' in file_name or 'sleepy' in file_name):
                return label
            if category == 'belly_pain' and ('pain' in file_name or 'stomach' in file_name):
                return label
            if category == 'burping' and ('burp' in file_name or 'gas' in file_name):
                return label
        
        return None
    
    def load_audio_file(self, file_path):
        """
        Load and preprocess audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            audio: Preprocessed audio array or None
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, duration=self.duration)
            audio = librosa.util.normalize(audio)
            
            if len(audio) < self.max_pad_len:
                audio = np.pad(audio, (0, self.max_pad_len - len(audio)), mode='constant')
            else:
                audio = audio[:self.max_pad_len]
                
            return audio
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            return None
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features with deltas"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return features
    
    def load_dataset(self, use_spectrogram=True, balance_classes=True):
        """
        Load and process the entire dataset
        
        Args:
            use_spectrogram: Use mel-spectrogram (True) or MFCC (False)
            balance_classes: Balance dataset classes
            
        Returns:
            X: Feature arrays
            y: Labels
            metadata: Dataset metadata
        """
        # First discover the dataset structure
        self.discover_dataset_structure()
        
        print(f"\nLoading dataset from: {self.dataset_path}")
        print(f"Categories: {list(self.cry_categories.keys())}")
        
        X = []
        y = []
        metadata = []
        skipped_files = 0
        
        # Process all audio files
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    file_path = os.path.join(root, file)
                    
                    # Extract label
                    label = self.extract_label_from_path(file_path)
                    if label is None:
                        skipped_files += 1
                        continue
                    
                    # Load and process audio
                    audio = self.load_audio_file(file_path)
                    if audio is None:
                        skipped_files += 1
                        continue
                    
                    # Extract features
                    try:
                        if use_spectrogram:
                            features = self.extract_mel_spectrogram(audio)
                        else:
                            features = self.extract_mfcc_features(audio)
                        
                        X.append(features)
                        y.append(label)
                        
                        metadata.append({
                            'file_path': file_path,
                            'file_name': file,
                            'label': label,
                            'category': list(self.cry_categories.keys())[label]
                        })
                        
                        if len(X) % 100 == 0:
                            print(f"Processed {len(X)} files...")
                            
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        skipped_files += 1
                        continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nDataset loaded:")
        print(f"  Total samples: {len(X)}")
        print(f"  Skipped files: {skipped_files}")
        print(f"  Feature shape: {X.shape[1:] if len(X) > 0 else 'No data'}")
        
        # Show label distribution
        if len(y) > 0:
            label_counts = Counter(y)
            print("\nLabel distribution:")
            for label, count in sorted(label_counts.items()):
                category_name = list(self.cry_categories.keys())[label]
                print(f"  {label} ({category_name}): {count} samples")
        
        # Balance classes if requested
        if balance_classes and len(X) > 0:
            X, y, metadata = self.balance_dataset(X, y, metadata)
        
        return X, y, metadata
    
    def balance_dataset(self, X, y, metadata):
        """Balance dataset by oversampling minority classes"""
        from sklearn.utils import resample
        
        print("\nBalancing dataset...")
        
        label_counts = Counter(y)
        max_samples = max(label_counts.values())
        target_size = min(max_samples, 500)  # Cap at 500 samples per class
        
        X_balanced = []
        y_balanced = []
        metadata_balanced = []
        
        for label in range(len(self.cry_categories)):
            label_indices = np.where(y == label)[0]
            
            if len(label_indices) == 0:
                continue
            
            X_label = X[label_indices]
            y_label = y[label_indices]
            metadata_label = [metadata[i] for i in label_indices]
            
            # Resample to target size
            if len(X_label) < target_size:
                X_resampled, y_resampled, metadata_resampled = resample(
                    X_label, y_label, metadata_label,
                    n_samples=target_size,
                    random_state=42
                )
            else:
                X_resampled, y_resampled, metadata_resampled = X_label[:target_size], y_label[:target_size], metadata_label[:target_size]
            
            X_balanced.extend(X_resampled)
            y_balanced.extend(y_resampled)
            metadata_balanced.extend(metadata_resampled)
        
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)
        
        print(f"Balanced dataset: {len(X_balanced)} samples")
        
        label_counts = Counter(y_balanced)
        print("Balanced label distribution:")
        for label, count in sorted(label_counts.items()):
            category_name = list(self.cry_categories.keys())[label]
            print(f"  {label} ({category_name}): {count} samples")
        
        return X_balanced, y_balanced, metadata_balanced
    
    def prepare_data_for_training(self, X, y, test_size=0.2, val_size=0.1):
        """Prepare data for training"""
        if len(X) == 0:
            raise ValueError("No data to process")
        
        # One-hot encode labels
        y_categorical = to_categorical(y, num_classes=len(self.cry_categories))
        
        # Add channel dimension for CNN
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"\nData splits:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_processed_data(self, X, y, metadata, save_path):
        """Save processed data and metadata"""
        os.makedirs(save_path, exist_ok=True)
        
        np.save(os.path.join(save_path, 'features.npy'), X)
        np.save(os.path.join(save_path, 'labels.npy'), y)
        
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(os.path.join(save_path, 'categories.json'), 'w') as f:
            json.dump(self.cry_categories, f, indent=2)
        
        with open(os.path.join(save_path, 'category_descriptions.json'), 'w') as f:
            json.dump(self.category_description, f, indent=2)
        
        print(f"Data saved to {save_path}")
    
    def load_processed_data(self, save_path):
        """Load preprocessed data"""
        X = np.load(os.path.join(save_path, 'features.npy'))
        y = np.load(os.path.join(save_path, 'labels.npy'))
        
        with open(os.path.join(save_path, 'categories.json'), 'r') as f:
            self.cry_categories = json.load(f)
        
        if os.path.exists(os.path.join(save_path, 'category_descriptions.json')):
            with open(os.path.join(save_path, 'category_descriptions.json'), 'r') as f:
                self.category_description = json.load(f)
        
        return X, y
    
    def visualize_dataset(self, metadata, X=None):
        """Visualize dataset statistics"""
        if not metadata:
            return
        
        df = pd.DataFrame(metadata)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Label distribution
        label_counts = df['category'].value_counts()
        axes[0].bar(range(len(label_counts)), label_counts.values)
        axes[0].set_xticks(range(len(label_counts)))
        axes[0].set_xticklabels(label_counts.index, rotation=45, ha='right')
        axes[0].set_title('Samples per Category')
        axes[0].set_ylabel('Count')
        
        # Sample spectrogram if available
        if X is not None and len(X) > 0:
            sample_idx = 0
            sample = X[sample_idx]
            if len(sample.shape) == 3:
                sample = sample[:, :, 0]
            
            im = axes[1].imshow(sample, aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title(f'Sample Spectrogram: {metadata[sample_idx]["category"]}')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Frequency')
            plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    processor = BabyCryDatasetProcessor("./data/raw/baby-cry-detection")
    
    # Load and process dataset
    X, y, metadata = processor.load_dataset(
        use_spectrogram=True,
        balance_classes=True
    )
    
    if len(X) > 0:
        # Visualize
        processor.visualize_dataset(metadata, X)
        
        # Prepare for training
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_data_for_training(X, y)
        
        # Save
        processor.save_processed_data(X, y, metadata, "./data/processed/")
        
        print("\nDataset processing complete!")
    else:
        print("No data was processed!")