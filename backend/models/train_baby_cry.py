"""
BABY CRY CLASSIFICATION - BIAS FIX V3
======================================
Fixes persistent class bias issues
"""

import os
import numpy as np
from baby_cry_dataset import BabyCryDatasetProcessor
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import json
from sklearn.utils.class_weight import compute_class_weight

def setup_directories():
    """Set up necessary directories"""
    directories = ["./data/raw/", "./data/processed/", "./saved_models/", "./logs/"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def build_model(input_shape, num_classes, dropout_rate=0.5):
    """
    Simpler, more stable architecture
    """
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    
    # Output - CRITICAL: Use 'glorot_uniform' for balanced initialization
    outputs = layers.Dense(num_classes, activation='softmax', 
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

class BalancedBatchGenerator(tf.keras.utils.Sequence):
    """
    FIXED: Properly balanced batch generator
    """
    def __init__(self, X, y, batch_size=25, augmentation=True, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Get labels
        self.y_labels = np.argmax(y, axis=1)
        self.num_classes = y.shape[1]
        
        # CRITICAL: Ensure batch_size is divisible by num_classes
        if batch_size % self.num_classes != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by num_classes ({self.num_classes})")
        
        self.samples_per_class = batch_size // self.num_classes
        
        # Split indices by class
        self.class_indices = [np.where(self.y_labels == i)[0] for i in range(self.num_classes)]
        
        # FIXED: Calculate steps to ensure ALL classes get equal epochs
        self.steps = max([len(idx) for idx in self.class_indices]) // self.samples_per_class
        
        print(f"  Batch Generator Info:")
        print(f"    Samples per class per batch: {self.samples_per_class}")
        print(f"    Steps per epoch: {self.steps}")
        for i in range(self.num_classes):
            print(f"    Class {i}: {len(self.class_indices[i])} samples")
        
        # Augmentation
        if self.augmentation:
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=8,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='constant',
                cval=0.0
            )
        
        self.on_epoch_end()
    
    def __len__(self):
        return self.steps
    
    def on_epoch_end(self):
        """Shuffle at epoch end"""
        if self.shuffle:
            for i in range(self.num_classes):
                np.random.shuffle(self.class_indices[i])
    
    def __getitem__(self, index):
        """Generate one batch - FIXED"""
        batch_X = []
        batch_y = []
        
        # Sample from each class
        for class_idx in range(self.num_classes):
            indices = self.class_indices[class_idx]
            
            # FIXED: Use modulo to cycle through all samples
            for i in range(self.samples_per_class):
                sample_idx = (index * self.samples_per_class + i) % len(indices)
                idx = indices[sample_idx]
                batch_X.append(self.X[idx])
                batch_y.append(self.y[idx])
        
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        
        # Shuffle within batch
        shuffle_idx = np.random.permutation(len(batch_X))
        batch_X = batch_X[shuffle_idx]
        batch_y = batch_y[shuffle_idx]
        
        # Augmentation
        if self.augmentation:
            augmented = [self.datagen.random_transform(img) for img in batch_X]
            batch_X = np.array(augmented)
        
        return batch_X, batch_y

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("Baby Cry Detection - BIAS FIX V3")
    print("=" * 70)
    
    setup_directories()
    
    # CRITICAL: Reset random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    CONFIG = {
        'target_sr': 22050,
        'duration': 3.0,
        'use_spectrogram': True,
        'balance_classes': True,
        'test_size': 0.15,
        'val_size': 0.15,
        'epochs': 100,
        'batch_size': 25,  # 5 classes * 5 samples = 25
        'learning_rate': 0.0001,  # Lower learning rate
        'patience': 20,
        'dropout_rate': 0.5,
    }
    
    print("\n🔧 Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    dataset_path = "./data/raw/baby-cry-detection"
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found at: {dataset_path}")
        return
    
    print("\n" + "=" * 70)
    print("Step 1: Loading Dataset")
    print("=" * 70)
    
    processor = BabyCryDatasetProcessor(
        dataset_path=dataset_path,
        target_sr=CONFIG['target_sr'],
        duration=CONFIG['duration']
    )
    
    # FORCE REPROCESS to ensure clean data
    print("\n⚠️  FORCING DATASET REPROCESS (clean start)")
    print("Processing raw dataset...")
    
    X, y, metadata = processor.load_dataset(
        use_spectrogram=CONFIG['use_spectrogram'],
        balance_classes=CONFIG['balance_classes']
    )
    
    if len(X) == 0:
        print("\n❌ No data was processed!")
        return
    
    processor.save_processed_data(X, y, metadata, "./data/processed/")
    
    print(f"\n✓ Dataset: {len(X)} samples, shape: {X.shape[1:]}")
    
    # CRITICAL: Verify class distribution
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    class_counts = {}
    for cls, count in zip(unique, counts):
        cat_name = list(processor.cry_categories.keys())[cls]
        class_counts[cls] = count
        print(f"  {cat_name:15} {count:3} samples ({count/len(y)*100:.1f}%)")
    
    # Check if truly balanced
    min_count = min(counts)
    max_count = max(counts)
    if max_count - min_count > 20:
        print(f"\n⚠️  WARNING: Class imbalance detected!")
        print(f"  Min: {min_count}, Max: {max_count}, Diff: {max_count - min_count}")
    else:
        print(f"\n✓ Classes are balanced (diff: {max_count - min_count})")
    
    print("\n" + "=" * 70)
    print("Step 2: Preparing Training Data")
    print("=" * 70)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_data_for_training(
        X, y,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )
    
    print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Verify training distribution
    train_labels = np.argmax(y_train, axis=1)
    print("\nTraining set distribution:")
    for cls in range(len(processor.cry_categories)):
        count = np.sum(train_labels == cls)
        cat_name = list(processor.cry_categories.keys())[cls]
        print(f"  {cat_name:15} {count:3} samples")
    
    print("\n" + "=" * 70)
    print("Step 3: Building Model")
    print("=" * 70)
    
    input_shape = X_train.shape[1:]
    num_classes = len(processor.cry_categories)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=CONFIG['dropout_rate']
    )
    
    # CRITICAL: Compute class weights even for balanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"\nClass weights:")
    for cls, weight in class_weight_dict.items():
        cat_name = list(processor.cry_categories.keys())[cls]
        print(f"  {cat_name:15} {weight:.3f}")
    
    # Compile with standard loss
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=CONFIG['learning_rate'],
        clipnorm=1.0  # Gradient clipping
    )
    
    # Use label smoothing to prevent overconfidence
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    print(f"\nModel parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            './saved_models/best_baby_cry_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n" + "=" * 70)
    print("Step 4: Training Model")
    print("=" * 70)
    
    train_gen = BalancedBatchGenerator(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        augmentation=True,
        shuffle=True
    )
    
    val_gen = BalancedBatchGenerator(
        X_val, y_val,
        batch_size=CONFIG['batch_size'],
        augmentation=False,
        shuffle=False
    )
    
    history = model.fit(
        train_gen,
        epochs=CONFIG['epochs'],
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weight_dict,  # CRITICAL: Use class weights
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("Step 5: Evaluating Model")
    print("=" * 70)
    
    # Load best model
    best_model = tf.keras.models.load_model('./saved_models/best_baby_cry_model.keras')
    
    # Evaluate
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    
    # Per-class analysis
    y_pred = best_model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n" + "=" * 70)
    print("Per-Class Performance")
    print("=" * 70)
    
    category_names = list(processor.cry_categories.keys())
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Header
    print("\n" + " " * 15, end="")
    for cat in category_names:
        print(f"{cat[:10]:>12}", end="")
    print()
    
    # Matrix
    for i, cat in enumerate(category_names):
        print(f"{cat[:15]:15}", end="")
        for j in range(len(category_names)):
            print(f"{cm[i,j]:12}", end="")
        print()
    
    print("\nPer-class metrics:")
    for cls_idx, cat_name in enumerate(category_names):
        cls_mask = y_true_classes == cls_idx
        if cls_mask.sum() > 0:
            cls_acc = (y_pred_classes[cls_mask] == cls_idx).sum() / cls_mask.sum()
            
            pred_mask = y_pred_classes == cls_idx
            if pred_mask.sum() > 0:
                cls_prec = (y_true_classes[pred_mask] == cls_idx).sum() / pred_mask.sum()
            else:
                cls_prec = 0.0
            
            cls_rec = cls_acc
            cls_f1 = 2 * cls_prec * cls_rec / (cls_prec + cls_rec + 1e-7)
            
            print(f"  {cat_name:15} Acc: {cls_acc:.1%}  Prec: {cls_prec:.1%}  Rec: {cls_rec:.1%}  F1: {cls_f1:.1%}")
    
    # CRITICAL: Check for bias
    print("\n" + "=" * 70)
    print("BIAS CHECK")
    print("=" * 70)
    
    pred_dist = np.bincount(y_pred_classes, minlength=num_classes)
    print("\nPrediction distribution:")
    for cls_idx, cat_name in enumerate(category_names):
        percentage = pred_dist[cls_idx] / len(y_pred_classes) * 100
        print(f"  {cat_name:15} {pred_dist[cls_idx]:3} predictions ({percentage:.1f}%)")
    
    # Check if any class is over-predicted
    expected_percentage = 100 / num_classes
    bias_detected = False
    for cls_idx, cat_name in enumerate(category_names):
        percentage = pred_dist[cls_idx] / len(y_pred_classes) * 100
        if percentage > expected_percentage * 1.5:  # 50% more than expected
            print(f"\n⚠️  WARNING: {cat_name} is over-predicted ({percentage:.1f}% vs expected {expected_percentage:.1f}%)")
            bias_detected = True
    
    if not bias_detected:
        print("\n✅ No significant prediction bias detected!")
    
    # Save final model
    best_model.save('./saved_models/final_baby_cry_model.keras')
    
    # Save results
    results = {
        'config': CONFIG,
        'results': {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss)
        },
        'dataset': {
            'categories': processor.cry_categories,
            'total_samples': int(len(X)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test))
        },
        'class_distribution': {
            cat_name: int(class_counts[cls_idx]) 
            for cls_idx, cat_name in enumerate(category_names)
        }
    }
    
    with open('./saved_models/baby_cry_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Test Accuracy: {test_acc:.2%}")
    
    if test_acc >= 0.65:
        print(f"✅ GOOD! Model achieved {test_acc:.1%} accuracy.")
    elif test_acc >= 0.50:
        print(f"✓ FAIR. Model achieved {test_acc:.1%} accuracy.")
    else:
        print(f"⚠️  Model needs improvement: {test_acc:.1%} accuracy.")
    
    print("\n💡 Key improvements:")
    print("  ✓ Properly balanced batch generation")
    print("  ✓ Class weights applied")
    print("  ✓ Gradient clipping")
    print("  ✓ Stable initialization")
    print("  ✓ Lower learning rate")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU available: {len(gpus)} device(s)\n")
        except RuntimeError as e:
            print(f"GPU error: {e}\n")
    else:
        print("Training on CPU\n")
    
    main()