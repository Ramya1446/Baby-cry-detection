import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import tempfile

# Page config
st.set_page_config(
    page_title="Baby Cry Detection",
    page_icon="🍼",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class BabyCryPredictor:
    def __init__(self, model_path, categories_path=None):
        self.model_path = model_path
        self.model = None
        self.target_sr = 22050
        self.duration = 3.0
        self.max_pad_len = int(self.target_sr * self.duration)
        
        # Load categories
        if categories_path and os.path.exists(categories_path):
            with open(categories_path, 'r') as f:
                self.cry_categories = json.load(f)
            self.category_names = list(self.cry_categories.keys())
        else:
            self.category_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
        
        # Category descriptions
        self.descriptions = {
            'belly_pain': 'Baby is experiencing stomach or digestive discomfort',
            'burping': 'Baby needs to burp or has gas',
            'discomfort': 'Baby is uncomfortable (temperature, position, etc.)',
            'hungry': 'Baby needs feeding',
            'tired': 'Baby is sleepy and needs rest'
        }
        
        self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        try:
            # FIXED: Load model with compile=False to avoid custom loss issues
            _self.model = tf.keras.models.load_model(_self.model_path, compile=False)
            
            # Recompile with standard loss
            _self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            st.success(f"✓ Model loaded: {os.path.basename(_self.model_path)}")
            return _self.model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def preprocess_audio(self, audio_file):
        try:
            audio, sr = librosa.load(audio_file, sr=self.target_sr, duration=self.duration)
            audio = librosa.util.normalize(audio)
            
            if len(audio) < self.max_pad_len:
                audio = np.pad(audio, (0, self.max_pad_len - len(audio)), mode='constant')
            else:
                audio = audio[:self.max_pad_len]
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.target_sr, n_mels=128, fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            features = np.expand_dims(mel_spec_db, axis=0)
            features = np.expand_dims(features, axis=-1)
            
            return features, audio
        except Exception as e:
            st.error(f"Error preprocessing audio: {e}")
            return None, None
    
    def analyze_audio(self, audio):
        try:
            characteristics = {}
            
            rms = np.sqrt(np.mean(audio**2))
            characteristics['energy'] = float(rms)
            characteristics['energy_level'] = 'High' if rms > 0.1 else 'Medium' if rms > 0.05 else 'Low'
            
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.target_sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            if len(pitch_values) > 0:
                valid = pitch_values[pitch_values > 0]
                if len(valid) > 0:
                    characteristics['pitch_mean'] = float(np.mean(valid))
                    characteristics['pitch_level'] = 'Very High' if characteristics['pitch_mean'] > 400 else 'High' if characteristics['pitch_mean'] > 300 else 'Normal'
                else:
                    characteristics['pitch_mean'] = 0
                    characteristics['pitch_level'] = 'Unknown'
            else:
                characteristics['pitch_mean'] = 0
                characteristics['pitch_level'] = 'Unknown'
            
            return characteristics
        except:
            return {}
    
    def predict(self, audio_file):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        features, audio = self.preprocess_audio(audio_file)
        if features is None:
            return {"error": "Failed to preprocess audio"}
        
        audio_analysis = self.analyze_audio(audio)
        
        predictions = self.model.predict(features, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        predicted_category = self.category_names[predicted_idx]
        
        confidence_scores = {
            name: float(predictions[idx])
            for idx, name in enumerate(self.category_names)
        }
        
        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "confidence_scores": confidence_scores,
            "audio_analysis": audio_analysis,
            "description": self.descriptions.get(predicted_category, "")
        }
    
    def get_recommendations(self, category):
        recommendations = {
            'hungry': [
                "Check when baby was last fed (typically every 2-3 hours for newborns)",
                "Look for hunger cues: rooting reflex, sucking motions, hands to mouth",
                "Offer breast or bottle feeding",
                "Ensure proper latch if breastfeeding",
                "Burp baby during and after feeding"
            ],
            'tired': [
                "Check age-appropriate wake windows",
                "Create a calm, quiet sleep environment",
                "Dim the lights and reduce stimulation",
                "Try gentle rocking or swaying motion",
                "Consider swaddling for younger babies",
                "Use white noise machine"
            ],
            'belly_pain': [
                "⚠️ Try gentle tummy massage in circular motions (clockwise)",
                "Hold baby upright or try 'bicycle legs' exercise",
                "Check for signs of gas or constipation",
                "Warm compress on tummy may help",
                "Avoid overfeeding",
                "If pain persists or worsens, contact pediatrician"
            ],
            'burping': [
                "Hold baby upright against your shoulder",
                "Gently pat or rub baby's back",
                "Try different positions: sitting up, over your lap",
                "Take breaks during feeding to burp",
                "Keep baby upright for 15-20 minutes after feeding"
            ],
            'discomfort': [
                "Check room temperature (68-72°F / 20-22°C is ideal)",
                "Examine clothing for tags, tightness, or irritation",
                "Check diaper and change if needed",
                "Try different holding positions",
                "Provide skin-to-skin contact",
                "Ensure baby is not too hot or cold"
            ]
        }
        return recommendations.get(category, ["Monitor baby's needs carefully", "Try general soothing techniques", "Contact pediatrician if concerned"])

@st.cache_resource
def load_predictor():
    model_paths = [
        "./saved_models/best_baby_cry_model.keras",
        "./saved_models/final_baby_cry_model.keras",
        "C:/Users/rramy/Music/babybloom-aid/backend/saved_models/best_baby_cry_model.keras",
        "C:/Users/rramy/Music/babybloom-aid/backend/saved_models/final_baby_cry_model.keras"
    ]
    
    categories_path = "./data/processed/categories.json"
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            return BabyCryPredictor(model_path, categories_path)
    
    return None

def create_confidence_chart(confidence_scores):
    categories = list(confidence_scores.keys())
    values = [confidence_scores[cat] * 100 for cat in categories]
    
    formatted_categories = [cat.replace('_', ' ').title() for cat in categories]
    
    colors = ['#28a745' if v >= 30 else '#ffc107' if v >= 15 else '#dc3545' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=formatted_categories,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Category",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🍼 Baby Cry Detection System</h1>', unsafe_allow_html=True)
    st.markdown("Upload a baby cry audio file to identify the reason and get care recommendations")
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("❌ Model not found!")
        st.info("""
        Please train the model first:
        1. Delete old data: `Remove-Item ./data/processed -Recurse -Force`
        2. Delete old models: `Remove-Item ./saved_models/*.keras -Force`
        3. Run: `python train_baby_cry_FIXED_V3.py`
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This system uses deep learning to classify baby cries into 5 categories:
        
        - 🤕 **Belly Pain** - Digestive discomfort
        - 💨 **Burping** - Needs to burp/has gas
        - 😣 **Discomfort** - General discomfort
        - 🍼 **Hungry** - Needs feeding
        - 😴 **Tired** - Needs sleep
        
        Upload WAV, MP3, OGG, or FLAC files.
        """)
        
        st.header("📊 Model Info")
        results_path = "./saved_models/baby_cry_training_results.json"
        
        # FIXED: Handle missing or malformed results file
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Check if results has the expected structure
                if 'results' in results and 'test_accuracy' in results['results']:
                    st.metric("Model Accuracy", f"{results['results']['test_accuracy']:.1%}")
                
                # Handle both old and new result formats
                if 'dataset' in results and 'categories' in results['dataset']:
                    num_categories = len(results['dataset']['categories'])
                    st.metric("Categories", num_categories)
                else:
                    st.metric("Categories", "5")
            except Exception as e:
                st.warning(f"Could not load training results: {e}")
                st.metric("Categories", "5")
        else:
            st.metric("Categories", "5")
            
        st.warning("⚠️ This tool assists parents but does NOT replace professional medical advice. Always consult your pediatrician for health concerns.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload a baby cry recording (3-5 seconds recommended)"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Analyze Cry", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio... Please wait."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        result = predictor.predict(tmp_path)
                        
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.session_state['result'] = result
                            st.success("✓ Analysis complete!")
                    finally:
                        os.unlink(tmp_path)
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            
            # Prediction box
            category_display = result['predicted_category'].replace('_', ' ').title()
            emoji_map = {
                'Belly Pain': '🤕',
                'Burping': '💨',
                'Discomfort': '😣',
                'Hungry': '🍼',
                'Tired': '😴'
            }
            emoji = emoji_map.get(category_display, '👶')
            
            st.markdown(f"""
            <div class="prediction-box">
                <h1>{emoji}</h1>
                <h2>{category_display}</h2>
                <p style="font-size: 1.3rem; font-weight: bold;">Confidence: {result['confidence']:.1%}</p>
                <p style="font-size: 0.95rem; margin-top: 10px;">{result['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reliability indicator
            if result['confidence'] > 0.5:
                st.success(f"🟢 **High Reliability** - Strong prediction")
            elif result['confidence'] > 0.3:
                st.warning(f"🟡 **Medium Reliability** - Consider multiple factors")
            else:
                st.error(f"🟠 **Low Reliability** - Try different audio or consult other indicators")
            
            # Audio analysis
            if result['audio_analysis']:
                st.subheader("🎵 Audio Characteristics")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    energy_level = result['audio_analysis'].get('energy_level', 'Unknown')
                    energy_emoji = '🔊' if energy_level == 'High' else '🔉' if energy_level == 'Medium' else '🔈'
                    st.metric(f"{energy_emoji} Energy Level", energy_level)
                
                with col_b:
                    pitch_level = result['audio_analysis'].get('pitch_level', 'Unknown')
                    pitch_emoji = '📈' if pitch_level in ['Very High', 'High'] else '📊'
                    st.metric(f"{pitch_emoji} Pitch Level", pitch_level)
        else:
            st.info("👆 Upload an audio file and click 'Analyze Cry' to see results")
    
    # Confidence chart and recommendations
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        st.divider()
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("📈 Confidence Breakdown")
            fig = create_confidence_chart(result['confidence_scores'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 3 predictions
            sorted_scores = sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True)
            st.write("**Top 3 Predictions:**")
            for i, (cat, score) in enumerate(sorted_scores[:3], 1):
                st.write(f"{i}. {cat.replace('_', ' ').title()}: {score:.1%}")
        
        with col4:
            st.subheader("💡 Recommended Actions")
            recommendations = predictor.get_recommendations(result['predicted_category'])
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="recommendation-box">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
        
        # Download results
        st.divider()
        
        col5, col6, col7 = st.columns([1, 1, 1])
        
        with col5:
            report = {
                "timestamp": datetime.now().isoformat(),
                "predicted_category": result['predicted_category'],
                "confidence": result['confidence'],
                "confidence_scores": result['confidence_scores'],
                "audio_analysis": result['audio_analysis'],
                "recommendations": recommendations
            }
            
            st.download_button(
                label="📥 Download Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"cry_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col6:
            if st.button("🔄 Analyze Another", use_container_width=True):
                del st.session_state['result']
                st.rerun()
        
        with col7:
            st.write("")  # Spacer
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Important Reminders:</strong></p>
        <p>• This is an AI-assisted tool, not a medical diagnosis<br>
        • Always trust your parental instincts<br>
        • Contact your pediatrician for persistent concerns or emergencies<br>
        • Every baby is unique - use this as one of many tools</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()