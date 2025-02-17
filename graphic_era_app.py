import streamlit as st
import numpy as np
import joblib
from data_processing import load_and_preprocess_data, preprocess_input

# Load pre-trained models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Streamlit App Title
st.title("🧠 Parkinson's Disease Detection")
st.write("Enter voice-related features below to check for Parkinson's Disease.")

# Input fields for all features (assuming 22 features in the dataset)
feature_names = [
    "MDVP:Fo (Hz)", "MDVP:Fhi (Hz)", "MDVP:Flo (Hz)", "MDVP:Jitter (%)", "MDVP:Jitter (Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer (dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

# Collect input features
features = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0, step=0.001)
    features.append(value)

# Convert to NumPy array
features = np.array([features])

# Model selection
model_choice = st.radio("Select a Model:", ("SVM", "Random Forest"))

# Prediction Button
if st.button("Predict Parkinson’s Disease"):
    try:
        # Preprocess input features (apply same scaling as during training)
        features = preprocess_input(features)
        
        # Load selected model
        model = svm_model if model_choice == "SVM" else rf_model
        
        # Predict
        prediction = model.predict(features)
        
        # Display result
        result = "Positive for Parkinson’s Disease" if prediction[0] == 1 else "Negative for Parkinson’s Disease"
        st.subheader(f"Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
