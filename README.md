# Baby Cry Emotion Detection System

An audio-based machine learning system designed to detect and classify baby crying sounds into five emotional states: **belly pain, hunger, burping, discomfort, and tiredness**. The system helps caregivers and healthcare providers understand infant needs by analyzing cry patterns from audio input.

---

## Project Overview

Infants communicate primarily through crying, making it difficult for caregivers to identify the underlying reason. This project aims to bridge that gap by using audio signal processing and machine learning techniques to classify baby cries into meaningful emotional categories.

The system analyzes cry audio features and predicts the most likely emotional state, enabling timely and appropriate caregiver responses.

---

## Cry Emotion Classes

The model classifies baby cries into the following five categories:

1. **Belly Pain** – Cry patterns associated with abdominal discomfort  
2. **Hungry** – Rhythmic and intense crying indicating hunger  
3. **Burping** – Crying due to trapped gas  
4. **Discomfort** – Crying caused by wet diapers, temperature, or irritation  
5. **Tired** – Cry patterns indicating fatigue or sleepiness  

---

## Key Features

- Audio-based baby cry classification  
- Multi-class emotion detection (5 categories)  
- Feature extraction using signal processing techniques  
- Machine learning / deep learning-based prediction  
- Scalable for real-time or recorded audio input  

---

## System Workflow

1. Audio input is captured or uploaded  
2. Audio preprocessing and normalization  
3. Feature extraction from cry signals  
4. Classification using trained ML/DL model  
5. Predicted baby emotion is displayed  

---

## Technologies Used

### Programming Language
- Python

### Audio Processing
- Librosa
- NumPy
- SciPy

### Feature Extraction
- MFCC (Mel-Frequency Cepstral Coefficients)  
- Chroma Features  
- Spectral Centroid  
- Zero Crossing Rate  
- RMS Energy  

### Machine Learning / Deep Learning
- scikit-learn (traditional ML models)  
- TensorFlow / Keras (CNN / LSTM models)  

### Visualization & Evaluation
- Matplotlib
- Seaborn

---

## Model Training

- Audio samples are labeled with one of the five emotional states  
- Features are extracted and stored as numerical representations  
- The dataset is split into training and testing sets  
- A multi-class classification model is trained  
- Model performance is evaluated using accuracy and confusion matrix  

---
## Output

The system outputs one of the following labels:

- Belly Pain
- Hungry
- Burping
- Discomfort
- Tired

---
## Use cases

- Smart baby monitoring systems
- Infant healthcare assistance
- IoT-based baby surveillance devices
- Neonatal care support tools
- Assistive technology for new parents

---
## Screenshots
<img width="1826" height="842" alt="image" src="https://github.com/user-attachments/assets/6c5fe15e-2c05-44ff-bb2d-2913aa980e25" />

## Installation

```bash
pip install librosa numpy scipy scikit-learn tensorflow matplotlib seaborn
