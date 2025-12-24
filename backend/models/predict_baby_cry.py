import os
import numpy as np
import librosa
import tensorflow as tf
import json
from datetime import datetime

class BabyCryPredictor:
    def __init__(self, model_path, categories_path="./data/processed/categories.json"):
        """
        Initialize baby cry predictor
        
        Args:
            model_path: Path to trained model (.keras or .h5)
            categories_path: Path to categories JSON file
        """
        self.model_path = model_path
        self.model = None
        
        # Audio parameters
        self.target_sr = 22050
        self.duration = 3.0
        self.max_pad_len = int(self.target_sr * self.duration)
        
        # Load categories
        if os.path.exists(categories_path):
            with open(categories_path, 'r') as f:
                self.cry_categories = json.load(f)
            self.category_names = list(self.cry_categories.keys())
        else:
            # Default categories
            self.category_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
            self.cry_categories = {name: idx for idx, name in enumerate(self.category_names)}
        
        # Load descriptions if available
        descriptions_path = "./data/processed/category_descriptions.json"
        if os.path.exists(descriptions_path):
            with open(descriptions_path, 'r') as f:
                self.category_descriptions = json.load(f)
        else:
            self.category_descriptions = {}
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from: {self.model_path}")
            print(f"Categories: {self.category_names}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio_file_path):
        """Preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=self.target_sr, duration=self.duration)
            audio = librosa.util.normalize(audio)
            
            # Pad or trim
            if len(audio) < self.max_pad_len:
                audio = np.pad(audio, (0, self.max_pad_len - len(audio)), mode='constant')
            else:
                audio = audio[:self.max_pad_len]
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.target_sr, n_mels=128, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Add dimensions
            features = np.expand_dims(mel_spec_db, axis=0)
            features = np.expand_dims(features, axis=-1)
            
            return features, audio
            
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return None, None
    
    def analyze_audio(self, audio):
        """Analyze audio characteristics"""
        try:
            characteristics = {}
            
            # Energy
            rms = np.sqrt(np.mean(audio**2))
            characteristics['energy'] = float(rms)
            characteristics['energy_level'] = 'high' if rms > 0.1 else 'medium' if rms > 0.05 else 'low'
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.target_sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            if len(pitch_values) > 0:
                valid = pitch_values[pitch_values > 0]
                if len(valid) > 0:
                    characteristics['pitch_mean'] = float(np.mean(valid))
                    characteristics['pitch_std'] = float(np.std(valid))
                    
                    # Classify pitch
                    if characteristics['pitch_mean'] > 400:
                        characteristics['pitch_level'] = 'very_high'
                    elif characteristics['pitch_mean'] > 300:
                        characteristics['pitch_level'] = 'high'
                    else:
                        characteristics['pitch_level'] = 'normal'
                else:
                    characteristics['pitch_mean'] = 0
                    characteristics['pitch_std'] = 0
                    characteristics['pitch_level'] = 'unknown'
            
            # Duration
            characteristics['duration'] = len(audio) / self.target_sr
            
            return characteristics
            
        except Exception as e:
            print(f"Warning: Could not analyze audio: {e}")
            return {}
    
    def predict(self, audio_file_path, confidence_threshold=0.5):
        """
        Predict cry category from audio file
        
        Args:
            audio_file_path: Path to audio file
            confidence_threshold: Minimum confidence for reliable prediction
            
        Returns:
            result: Dictionary with prediction results
        """
        # Preprocess
        features, audio = self.preprocess_audio(audio_file_path)
        if features is None:
            return {"error": "Failed to preprocess audio"}
        
        # Analyze audio
        audio_analysis = self.analyze_audio(audio)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        # Get category name
        predicted_category = self.category_names[predicted_idx]
        
        # Confidence scores for all categories
        confidence_scores = {
            name: float(predictions[0][idx])
            for idx, name in enumerate(self.category_names)
        }
        
        # Build result
        result = {
            "file": os.path.basename(audio_file_path),
            "predicted_category": predicted_category,
            "confidence": confidence,
            "is_reliable": confidence >= confidence_threshold,
            "confidence_scores": confidence_scores,
            "audio_analysis": audio_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add description if available
        if predicted_category in self.category_descriptions:
            result["description"] = self.category_descriptions[predicted_category]
        
        return result
    
    def get_recommendations(self, predicted_category, confidence, audio_analysis=None):
        """Get care recommendations based on prediction"""
        
        recommendations = {
            'hungry': [
                "Check when baby was last fed (typically every 2-3 hours for newborns)",
                "Look for hunger cues: rooting reflex, sucking motions, hands to mouth",
                "Offer breast or bottle feeding",
                "Ensure proper latch if breastfeeding",
                "Burp baby during and after feeding"
            ],
            'tired': [
                "Check wake windows appropriate for baby's age",
                "Create a calm, quiet sleep environment",
                "Dim the lights and reduce stimulation",
                "Try gentle rocking or swaying motion",
                "Consider swaddling for younger babies",
                "Use white noise or soft lullabies"
            ],
            'belly_pain': [
                "⚠️ Try gentle tummy massage in circular motions",
                "Hold baby upright or try 'bicycle legs' exercise",
                "Check for signs of gas or constipation",
                "Consider burping if recently fed",
                "Warm compress on tummy may help",
                "If pain persists or worsens, contact pediatrician"
            ],
            'burping': [
                "Hold baby upright against your shoulder",
                "Gently pat or rub baby's back",
                "Try different positions: sitting up, over your lap",
                "Take breaks during feeding to burp",
                "Keep baby upright for 15-20 minutes after feeding",
                "Consider anti-gas drops if recommended by doctor"
            ],
            'discomfort': [
                "Check room temperature (68-72°F / 20-22°C is ideal)",
                "Examine clothing for tags, tightness, or irritation",
                "Check diaper and change if needed",
                "Look for any physical discomfort sources",
                "Try different holding positions",
                "Provide skin-to-skin contact for comfort"
            ],
            'pain': [
                "⚠️ Check for fever with thermometer",
                "Examine baby carefully for any signs of injury",
                "Look for trapped hair or tight clothing",
                "Check extremities and fingers/toes",
                "If crying persists or baby seems ill, contact pediatrician",
                "Document symptoms to share with doctor"
            ],
            'attention': [
                "Provide social interaction - talk, sing, make eye contact",
                "Gentle play appropriate for baby's age",
                "Skin-to-skin contact or cuddling",
                "Change scenery - go to different room or outside",
                "Show interesting objects or toys",
                "Remember: babies need interaction and connection"
            ],
            'diaper': [
                "Check and change diaper",
                "Clean thoroughly with wipes or warm water",
                "Apply diaper cream to prevent/treat rash",
                "Ensure diaper isn't too tight",
                "Allow some diaper-free time for air circulation",
                "Check for signs of diaper rash or irritation"
            ]
        }
        
        base_recs = recommendations.get(predicted_category, [
            "Monitor baby's needs and comfort level",
            "Try general soothing techniques",
            "Consult with pediatrician if concerns persist"
        ])
        
        # Add confidence note
        if confidence < 0.6:
            base_recs.insert(0, "⚠️ Low confidence - try multiple approaches and observe baby's response")
        elif confidence > 0.8:
            base_recs.insert(0, "✓ High confidence prediction - try these specific approaches first")
        
        # Add audio analysis insights
        if audio_analysis:
            energy_level = audio_analysis.get('energy_level', 'unknown')
            pitch_level = audio_analysis.get('pitch_level', 'unknown')
            
            if energy_level == 'high' and pitch_level in ['high', 'very_high']:
                base_recs.append("⚡ High-intensity cry detected - respond promptly to address baby's urgent need")
            elif energy_level == 'low':
                base_recs.append("💤 Low-energy cry - baby may be tired or winding down")
        
        return base_recs

def main():
    """Main prediction interface"""
    print("=" * 60)
    print("Baby Cry Detection - Prediction System")
    print("=" * 60)
    
    # Check for model
    model_path = "./saved_models/best_baby_cry_model.keras"
    if not os.path.exists(model_path):
        model_path = "./saved_models/final_baby_cry_model.keras"
    
    if not os.path.exists(model_path):
        print("ERROR: No trained model found!")
        print("Please train the model first by running: python train_baby_cry.py")
        return
    
    # Initialize predictor
    try:
        predictor = BabyCryPredictor(model_path)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return
    
    print("\nOptions:")
    print("1. Analyze single audio file")
    print("2. Test with sample from dataset")
    print("3. Batch analysis")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == "1":
        audio_file = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            return
        
        print("\nAnalyzing...")
        result = predictor.predict(audio_file)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Display results
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\nFile: {result['file']}")
        print(f"Predicted Category: {result['predicted_category'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Reliable: {'✓ Yes' if result['is_reliable'] else '✗ Low confidence'}")
        
        if 'description' in result:
            print(f"\nDescription: {result['description']}")
        
        print("\nConfidence Scores:")
        for category, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 30)
            print(f"  {category:15} {score:5.1%} {bar}")
        
        if result['audio_analysis']:
            print("\nAudio Analysis:")
            analysis = result['audio_analysis']
            print(f"  Energy Level: {analysis.get('energy_level', 'unknown')}")
            print(f"  Pitch Level: {analysis.get('pitch_level', 'unknown')}")
            if 'pitch_mean' in analysis and analysis['pitch_mean'] > 0:
                print(f"  Mean Pitch: {analysis['pitch_mean']:.0f} Hz")
        
        print("\n" + "=" * 60)
        print("RECOMMENDED ACTIONS")
        print("=" * 60)
        recommendations = predictor.get_recommendations(
            result['predicted_category'],
            result['confidence'],
            result['audio_analysis']
        )
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
    elif choice == "2":
        # Test with sample
        sample_dir = "./data/raw/baby-cry-detection"
        if not os.path.exists(sample_dir):
            print("Dataset not found. Please download first.")
            return
        
        # Find a sample file
        for root, dirs, files in os.walk(sample_dir):
            audio_files = [f for f in files if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            if audio_files:
                sample_file = os.path.join(root, audio_files[0])
                print(f"\nTesting with: {sample_file}")
                
                result = predictor.predict(sample_file)
                print(f"\nPredicted: {result['predicted_category']}")
                print(f"Confidence: {result['confidence']:.1%}")
                break
        
    elif choice == "3":
        folder = input("Enter folder path with audio files: ").strip()
        if not os.path.exists(folder):
            print("Folder not found")
            return
        
        print("\nProcessing files...")
        results = []
        
        for file in os.listdir(folder):
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                file_path = os.path.join(folder, file)
                result = predictor.predict(file_path)
                if "error" not in result:
                    results.append(result)
                    print(f"  {file}: {result['predicted_category']} ({result['confidence']:.1%})")
        
        print(f"\nProcessed {len(results)} files")
        
        # Save results
        output_file = "batch_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()