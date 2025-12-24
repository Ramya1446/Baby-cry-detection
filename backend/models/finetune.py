"""
Fine-tune Model with Manual Labeled Data
=========================================
Uses your real-life baby cry recordings to improve the model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import librosa
import json
from sklearn.model_selection import train_test_split

class FineTuner:
    def __init__(self, model_path, manual_data_dir="./data/manual_labeled/"):
        self.model_path = model_path
        self.manual_data_dir = manual_data_dir
        self.categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
        self.target_sr = 22050
        self.duration = 3.0
        self.model = None
    
    def load_base_model(self):
        """Load the pre-trained model"""
        print("\n📦 Loading base model...")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("   Train the base model first!")
            return False
        
        try:
            # Load without compile to avoid custom loss issues
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            # Recompile for fine-tuning
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Very low LR
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✓ Model loaded: {os.path.basename(self.model_path)}")
            print(f"  Parameters: {self.model.count_params():,}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_audio(self, filepath):
        """Preprocess audio file to match model input"""
        try:
            audio, sr = librosa.load(filepath, sr=self.target_sr, duration=self.duration)
            audio = librosa.util.normalize(audio)
            
            max_len = int(self.target_sr * self.duration)
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
            else:
                audio = audio[:max_len]
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.target_sr, n_mels=128, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
        except Exception as e:
            print(f"  ⚠️ Error processing {os.path.basename(filepath)}: {e}")
            return None
    
    def load_manual_data(self):
        """Load manually labeled audio files"""
        print("\n📂 Loading manual labeled data...")
        
        if not os.path.exists(self.manual_data_dir):
            print(f"❌ Manual data directory not found: {self.manual_data_dir}")
            return None, None
        
        X_list = []
        y_list = []
        file_count = {cat: 0 for cat in self.categories}
        
        for cat_idx, category in enumerate(self.categories):
            cat_dir = os.path.join(self.manual_data_dir, category)
            
            if not os.path.exists(cat_dir):
                continue
            
            audio_files = [f for f in os.listdir(cat_dir) 
                          if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            
            print(f"  {category:15} {len(audio_files)} files")
            
            for audio_file in audio_files:
                filepath = os.path.join(cat_dir, audio_file)
                features = self.preprocess_audio(filepath)
                
                if features is not None:
                    X_list.append(features)
                    y_list.append(cat_idx)
                    file_count[category] += 1
        
        if not X_list:
            print("\n❌ No manual data found!")
            return None, None
        
        X = np.array(X_list)
        X = np.expand_dims(X, axis=-1)  # Add channel dimension
        y = np.array(y_list)
        
        print(f"\n✓ Loaded {len(X)} manual samples")
        print(f"  Feature shape: {X.shape[1:]}")
        
        return X, y
    
    def augment_data(self, X, y, augmentation_factor=10):
        """Augment the manual data to create more training samples"""
        print(f"\n🔄 Augmenting data (factor: {augmentation_factor})...")
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.15,
            horizontal_flip=False,
            fill_mode='constant'
        )
        
        X_aug = []
        y_aug = []
        
        # Add original data
        for i in range(len(X)):
            X_aug.append(X[i])
            y_aug.append(y[i])
        
        # Generate augmented versions
        for i in range(len(X)):
            for _ in range(augmentation_factor - 1):
                aug_img = datagen.random_transform(X[i])
                X_aug.append(aug_img)
                y_aug.append(y[i])
        
        X_aug = np.array(X_aug)
        y_aug = np.array(y_aug)
        
        print(f"✓ Augmented to {len(X_aug)} samples")
        
        return X_aug, y_aug
    
    def fine_tune(self, X, y, epochs=50, validation_split=0.2):
        """Fine-tune the model on manual data"""
        print("\n" + "=" * 70)
        print("FINE-TUNING MODEL")
        print("=" * 70)
        
        # Convert labels to one-hot
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(self.categories))
        
        # Split data
        if len(X) > 5:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_onehot, test_size=validation_split, stratify=y, random_state=42
            )
            print(f"\n  Training samples: {len(X_train)}")
            print(f"  Validation samples: {len(X_val)}")
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X, y_onehot
            print(f"\n  Training samples: {len(X_train)} (no validation split due to small size)")
            validation_data = None
        
        # Freeze early layers (optional - helps prevent overfitting)
        print("\n  Freezing early layers...")
        for layer in self.model.layers[:-5]:  # Freeze all but last 5 layers
            layer.trainable = False
        
        trainable_count = sum([1 for layer in self.model.layers if layer.trainable])
        print(f"  Trainable layers: {trainable_count}/{len(self.model.layers)}")
        
        # Recompile after freezing
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss' if validation_data is None else 'val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                './saved_models/finetuned_baby_cry_model.keras',
                monitor='loss' if validation_data is None else 'val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\n  Training for {epochs} epochs...")
        print("  (This will be quick with small dataset)")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(8, len(X_train)),
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Fine-tuning complete!")
        
        return history
    
    def evaluate_on_manual_data(self, X, y):
        """Test the fine-tuned model"""
        print("\n" + "=" * 70)
        print("EVALUATION ON MANUAL DATA")
        print("=" * 70)
        
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(self.categories))
        
        # Evaluate
        loss, acc = self.model.evaluate(X, y_onehot, verbose=0)
        print(f"\n  Overall Accuracy: {acc:.1%}")
        
        # Per-sample predictions
        predictions = self.model.predict(X, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        print("\n  Per-sample results:")
        correct = 0
        for i in range(len(X)):
            true_cat = self.categories[y[i]]
            pred_cat = self.categories[pred_classes[i]]
            confidence = predictions[i, pred_classes[i]]
            
            match = "✓" if true_cat == pred_cat else "✗"
            if match == "✓":
                correct += 1
            
            print(f"    {i+1}. {match} True: {true_cat:12} → Pred: {pred_cat:12} ({confidence:.1%})")
        
        print(f"\n  Accuracy: {correct}/{len(X)} = {correct/len(X):.1%}")

def main():
    print("=" * 70)
    print("FINE-TUNE WITH MANUAL DATA")
    print("=" * 70)
    
    # Check if manual data exists
    manual_dir = "./data/manual_labeled/"
    if not os.path.exists(manual_dir):
        print(f"\n❌ Manual data not found!")
        print("\n  Steps:")
        print("    1. Run: python manual_label_tool.py")
        print("    2. Label your 4-5 real audio files")
        print("    3. Then run this script again")
        return
    
    # Find base model
    model_paths = [
        "./saved_models/best_baby_cry_model.keras",
        "./saved_models/final_baby_cry_model.keras"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("\n❌ No base model found!")
        print("   Train the base model first: python train_baby_cry_FINAL_FIX.py")
        return
    
    # Initialize fine-tuner
    finetuner = FineTuner(model_path, manual_dir)
    
    # Load base model
    if not finetuner.load_base_model():
        return
    
    # Load manual data
    X_manual, y_manual = finetuner.load_manual_data()
    
    if X_manual is None:
        return
    
    # Ask about augmentation
    print("\n" + "=" * 70)
    print("DATA AUGMENTATION")
    print("=" * 70)
    
    if len(X_manual) < 20:
        print(f"\n⚠️  You have only {len(X_manual)} samples.")
        print("   Data augmentation is HIGHLY recommended!")
        
        aug_choice = input("\n  Augment data? (y/n, default: y): ").strip().lower()
        
        if aug_choice != 'n':
            aug_factor = input("  Augmentation factor (default: 10): ").strip()
            aug_factor = int(aug_factor) if aug_factor.isdigit() else 10
            
            X_manual, y_manual = finetuner.augment_data(X_manual, y_manual, aug_factor)
    
    # Fine-tune
    epochs = input("\n  Number of epochs (default: 50): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 50
    
    history = finetuner.fine_tune(X_manual, y_manual, epochs=epochs)
    
    # Evaluate
    finetuner.evaluate_on_manual_data(X_manual, y_manual)
    
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print("\n✅ Fine-tuned model saved: ./saved_models/finetuned_baby_cry_model.keras")
    print("\n  Next steps:")
    print("    1. Update app.py to use 'finetuned_baby_cry_model.keras'")
    print("    2. Test with your real audio files")
    print("    3. If accuracy is good, this model is ready to use!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()