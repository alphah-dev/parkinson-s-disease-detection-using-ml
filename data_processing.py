import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def load_and_preprocess_data():
    # Load dataset (replace 'parkinsons.csv' with your actual dataset file)
    data = pd.read_csv("parkinsons.csv")

    # Ensure the dataset has the correct structure
    if data.isnull().values.any():
        raise ValueError("Dataset contains missing values. Please clean the data.")

    # Assuming the last column is the target (adjust if needed)
    X = data.iloc[:, :-1]   
    y = data.iloc[:, -1]   

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler  

# Preprocess a single input sample for prediction
def preprocess_input(features):
    _, _, _, _, scaler = load_and_preprocess_data()  
    return scaler.transform(np.array(features).reshape(1, -1))
