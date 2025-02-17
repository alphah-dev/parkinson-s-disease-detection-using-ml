import streamlit as st
import numpy as np
import joblib
from data_processing import load_and_preprocess_data, preprocess_input

svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

st.title("🧠 Parkinson's Disease Detection")
st.write("Enter voice-related features below to check for Parkinson's Disease.")


feature_names = [
    "MDVP:Fo (Hz)", "MDVP:Fhi (Hz)", "MDVP:Flo (Hz)", "MDVP:Jitter (%)", "MDVP:Jitter (Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer (dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

features = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0, step=0.001)
    features.append(value)

features = np.array([features])

model_choice = st.radio("Select a Model:", ("SVM", "Random Forest"))

if st.button("Predict Parkinson’s Disease"):
    try:
        features = preprocess_input(features)
        
        model = svm_model if model_choice == "SVM" else rf_model
        
        
        prediction = model.predict(features)
        
        result = "Positive for Parkinson’s Disease" if prediction[0] == 1 else "Negative for Parkinson’s Disease"
        st.subheader(f"Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
