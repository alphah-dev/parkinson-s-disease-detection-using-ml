from model import train_and_evaluate, train_random_forest

def main():
    print("Training Support Vector Machine (SVM)...")
    best_svm = train_and_evaluate()
    print("SVM training complete.\n")
    
    print("Training Random Forest Classifier...")
    best_rf = train_random_forest()
    print("Random Forest training complete.\n")

    print("Model training completed successfully.")

if __name__ == "__main__":
    main()