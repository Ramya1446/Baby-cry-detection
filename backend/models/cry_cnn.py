import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class InfantCryCNN:
    def __init__(self, input_shape, num_classes=6):
        """
        Initialize the CNN model for infant cry classification
        
        Args:
            input_shape: Shape of input features (height, width, channels)
            num_classes: Number of cry categories
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model_v1(self, dropout_rate=0.3, l2_reg=0.001):
        """
        Build CNN Model Version 1 - Basic Architecture
        Suitable for mel-spectrogram features
        
        Args:
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
        """
        self.model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape,
                         kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=l2(l2_reg)),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def build_model_v2(self, dropout_rate=0.4, l2_reg=0.001):
        """
        Build CNN Model Version 2 - Advanced Architecture with Residual-like connections
        Better for complex audio patterns
        
        Args:
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # First Conv Block
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(l2_reg))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Second Conv Block with skip connection
        shortcut = x
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection (adjust dimensions if needed)
        if shortcut.shape[-1] != 64:
            shortcut = layers.Conv2D(64, (1, 1), padding='same')(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Third Conv Block
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Fourth Conv Block with Attention
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        
        # Simple attention mechanism
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense Layers with residual connection
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        dense_shortcut = x
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        
        # Dense skip connection
        dense_shortcut = layers.Dense(256)(dense_shortcut)
        x = layers.Add()([x, dense_shortcut])
        x = layers.Activation('relu')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        return self.model
    
    def build_lightweight_model(self, dropout_rate=0.2):
        """
        Build a lightweight CNN model for mobile deployment
        Optimized for speed and smaller model size
        
        Args:
            dropout_rate: Dropout rate for regularization
        """
        self.model = models.Sequential([
            # Depthwise Separable Conv Block 1
            layers.SeparableConv2D(32, (3, 3), activation='relu', 
                                 input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Depthwise Separable Conv Block 2
            layers.SeparableConv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            
            # Depthwise Separable Conv Block 3
            layers.SeparableConv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Compact Dense Layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type
        """
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
        """
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_cry_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data Augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip audio spectrograms
            fill_mode='constant'
        )
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        # Evaluate model
        test_loss, test_acc, test_prec, test_rec = self.model.evaluate(X_test, y_test, verbose=0)
        f1_score = 2 * (test_prec * test_rec) / (test_prec + test_rec + 1e-8)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test F1-Score: {f1_score:.4f}")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Get unique classes actually present in the test set
        unique_classes = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
        
        # Default cry labels for all possible classes
        all_cry_labels = ['Hungry', 'Sleepy', 'Uncomfortable', 'Pain', 'Attention', 'Diaper']
        
        # If we have more or fewer classes, adjust the labels
        if self.num_classes != len(all_cry_labels):
            all_cry_labels = [f'Class_{i}' for i in range(self.num_classes)]
        
        # Only use labels for classes present in test set
        cry_labels = [all_cry_labels[i] for i in unique_classes if i < len(all_cry_labels)]
        
        # Classification report with only present classes
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                   labels=unique_classes,
                                   target_names=cry_labels,
                                   zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes, labels=unique_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=cry_labels, yticklabels=cry_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return test_acc, test_prec, test_rec, f1_score
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history found. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_cry_reason(self, audio_features):
        """
        Predict cry reason from audio features
        
        Args:
            audio_features: Extracted features from audio
            
        Returns:
            prediction: Predicted cry category
            confidence: Confidence scores
        """
        # Ensure correct shape
        if len(audio_features.shape) == 2:
            audio_features = np.expand_dims(audio_features, axis=0)  # Add batch dimension
        if len(audio_features.shape) == 3:
            audio_features = np.expand_dims(audio_features, axis=-1)  # Add channel dimension
        
        # Predict
        predictions = self.model.predict(audio_features)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0]
        
        cry_labels = ['Hungry', 'Sleepy', 'Uncomfortable', 'Pain', 'Attention', 'Diaper']
        predicted_reason = cry_labels[predicted_class]
        
        return predicted_reason, confidence
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath: Path to load the model from
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def get_model_summary(self):
        """
        Get model summary
        """
        if self.model is None:
            print("Model not built yet.")
            return
        
        self.model.summary()
        
        # Calculate model size
        param_count = self.model.count_params()
        print(f"\nTotal parameters: {param_count:,}")
        
        # Estimate model size in MB (rough estimation)
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        print(f"Estimated model size: {model_size_mb:.2f} MB")

# Example usage
if __name__ == "__main__":
    # Example for mel-spectrogram input shape
    input_shape = (128, 130, 1)  # (n_mels, time_frames, channels)
    
    # Initialize model
    cry_model = InfantCryCNN(input_shape=input_shape, num_classes=6)
    
    # Build model (choose one)
    model = cry_model.build_model_v2()
    cry_model.compile_model(learning_rate=0.001)
    
    # Print model summary
    cry_model.get_model_summary()