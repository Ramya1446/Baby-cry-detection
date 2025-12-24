"""
BABY CRY CLASSIFICATION - FINAL FIX
====================================
Complete rewrite with proven approach
"""

import os
import numpy as np
from baby_cry_dataset import BabyCryDatasetProcessor
import tensorflow as tf
from tensorflow.keras import layers, models
import json

def setup_directories():
    directories = ["./data/raw/", "./data/processed/", "./saved_models/", "./logs/"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def build_simple_cnn(input_shape, num_classes):
    """
    Simple but effective CNN - no fancy tricks
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Global pooling + dense
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output - CRITICAL: Initialize bias to prevent initial class preference
        layers.Dense(num_classes, activation='softmax',
                    kernel_initializer='glorot_uniform',
                    bias_initializer=tf.keras.initializers.Constant(-np.log(num_classes - 1)))
    ])
    
    return model

def main():
    print("=" * 70)
    print("Baby Cry Detection - FINAL FIX")
    print("=" * 70)
    
    setup_directories()
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    CONFIG = {
        'target_sr': 22050,
        'duration': 3.0,
        'use_spectrogram': True,
        'balance_classes': True,
        'test_size': 0.15,
        'val_size': 0.15,
        'epochs': 200,  # More epochs
        'batch_size': 32,  # Standard batch size
        'learning_rate': 0.001,  # Higher initial LR
        'dropout_rate': 0.4,
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
    
    # Force reprocess
    print("Processing dataset...")
    X, y, metadata = processor.load_dataset(
        use_spectrogram=CONFIG['use_spectrogram'],
        balance_classes=CONFIG['balance_classes']
    )
    
    if len(X) == 0:
        print("\n❌ No data was processed!")
        return
    
    processor.save_processed_data(X, y, metadata, "./data/processed/")
    print(f"\n✓ Dataset: {len(X)} samples, shape: {X.shape[1:]}")
    
    # Verify balance
    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        cat_name = list(processor.cry_categories.keys())[cls]
        print(f"  {cat_name:15} {count:3} samples ({count/len(y)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Step 2: Preparing Training Data")
    print("=" * 70)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_data_for_training(
        X, y,
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size']
    )
    
    print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # CRITICAL: Verify training distribution
    train_labels = np.argmax(y_train, axis=1)
    val_labels = np.argmax(y_val, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    
    print("\nTraining distribution:")
    for cls in range(len(processor.cry_categories)):
        train_count = np.sum(train_labels == cls)
        val_count = np.sum(val_labels == cls)
        test_count = np.sum(test_labels == cls)
        cat_name = list(processor.cry_categories.keys())[cls]
        print(f"  {cat_name:12} Train: {train_count:3} Val: {val_count:2} Test: {test_count:2}")
    
    print("\n" + "=" * 70)
    print("Step 3: Building Model")
    print("=" * 70)
    
    input_shape = X_train.shape[1:]
    num_classes = len(processor.cry_categories)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = build_simple_cnn(input_shape, num_classes)
    
    # CRITICAL: Use standard categorical crossentropy with label smoothing
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    print(f"\nModel parameters: {model.count_params():,}")
    model.summary()
    
    # Callbacks
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            './saved_models/best_baby_cry_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Add custom callback to monitor per-class accuracy
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"\n  Epoch {epoch+1}: Train Acc={logs['accuracy']:.3f}, Val Acc={logs['val_accuracy']:.3f}")
        )
    ]
    
    print("\n" + "=" * 70)
    print("Step 4: Training Model (Standard Approach)")
    print("=" * 70)
    
    # Use standard ImageDataGenerator - simpler and proven
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='constant'
    )
    
    # CRITICAL: Shuffle data before each epoch
    print("\n⚠️ Training with standard data augmentation")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Steps per epoch: {len(X_train) // CONFIG['batch_size']}")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=CONFIG['batch_size'], shuffle=True),
        epochs=CONFIG['epochs'],
        steps_per_epoch=len(X_train) // CONFIG['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("Step 5: Evaluating Model")
    print("=" * 70)
    
    # Load best model
    best_model = tf.keras.models.load_model('./saved_models/best_baby_cry_model.keras')
    
    # Evaluate
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test Accuracy: {test_acc:.2%}")
    
    # Detailed predictions
    y_pred = best_model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n" + "=" * 70)
    print("Per-Class Performance")
    print("=" * 70)
    
    category_names = list(processor.cry_categories.keys())
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    print("\nConfusion Matrix:")
    print("\n" + " " * 15, end="")
    for cat in category_names:
        print(f"{cat[:10]:>12}", end="")
    print()
    
    for i, cat in enumerate(category_names):
        print(f"{cat[:15]:15}", end="")
        for j in range(len(category_names)):
            print(f"{cm[i,j]:12}", end="")
        print()
    
    # Per-class metrics
    print("\nPer-class accuracy:")
    class_accuracies = []
    for cls_idx, cat_name in enumerate(category_names):
        cls_mask = y_true_classes == cls_idx
        if cls_mask.sum() > 0:
            cls_correct = (y_pred_classes[cls_mask] == cls_idx).sum()
            cls_total = cls_mask.sum()
            cls_acc = cls_correct / cls_total
            class_accuracies.append(cls_acc)
            
            bar = "█" * int(cls_acc * 20)
            print(f"  {cat_name:12} {cls_correct:2}/{cls_total:2} = {cls_acc:.1%} {bar}")
    
    # BIAS CHECK
    print("\n" + "=" * 70)
    print("BIAS CHECK")
    print("=" * 70)
    
    pred_dist = np.bincount(y_pred_classes, minlength=num_classes)
    print("\nPrediction distribution:")
    
    expected_per_class = len(y_pred_classes) / num_classes
    bias_detected = False
    
    for cls_idx, cat_name in enumerate(category_names):
        count = pred_dist[cls_idx]
        percentage = count / len(y_pred_classes) * 100
        expected_pct = 100 / num_classes
        
        bar = "█" * int(percentage / 5)
        status = ""
        
        if percentage > expected_pct * 1.5:
            status = " ⚠️ OVER-PREDICTED"
            bias_detected = True
        elif percentage < expected_pct * 0.5:
            status = " ⚠️ UNDER-PREDICTED"
            bias_detected = True
        
        print(f"  {cat_name:12} {count:3} ({percentage:5.1f}%) {bar}{status}")
    
    if not bias_detected:
        print("\n✅ No significant prediction bias!")
    
    # Check if balanced
    min_acc = min(class_accuracies)
    max_acc = max(class_accuracies)
    balance_score = min_acc / max_acc if max_acc > 0 else 0
    
    print(f"\n📊 Balance Score: {balance_score:.2f}")
    print(f"  (Min class: {min_acc:.1%}, Max class: {max_acc:.1%})")
    
    if balance_score > 0.7:
        print("  ✅ Well balanced across classes!")
    elif balance_score > 0.5:
        print("  ✓ Reasonably balanced")
    else:
        print("  ⚠️ Some class imbalance detected")
    
    # Save final model
    best_model.save('./saved_models/final_baby_cry_model.keras')
    
    # Save results
    results = {
        'config': CONFIG,
        'results': {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'per_class_accuracy': {
                cat_name: float(class_accuracies[i])
                for i, cat_name in enumerate(category_names)
            },
            'balance_score': float(balance_score)
        },
        'dataset': {
            'categories': processor.cry_categories,
            'total_samples': int(len(X)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test))
        }
    }
    
    with open('./saved_models/baby_cry_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Accuracy: {test_acc:.2%}")
    print(f"📊 Balance Score: {balance_score:.2f}")
    
    if test_acc >= 0.60 and balance_score >= 0.6:
        print("\n✅ SUCCESS! Model is working well and balanced!")
    elif test_acc >= 0.50:
        print("\n✓ Model is functional but could be better")
    else:
        print("\n⚠️ Model needs improvement")
        print("\nPossible issues:")
        print("  - Classes might be too similar")
        print("  - Need more diverse training data")
        print("  - Consider feature engineering")
    
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