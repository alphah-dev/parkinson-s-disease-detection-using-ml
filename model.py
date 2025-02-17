from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from data_processing import load_and_preprocess_data

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'kernel': ['linear', 'rbf']
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized SVM Accuracy: {accuracy:.2f}")
    print("Best SVM Parameters:", grid_search.best_params_)
    
    return best_svm

def train_random_forest():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Optimized Random Forest Accuracy: {accuracy:.2f}")
    print("Best Random Forest Parameters:", grid_search.best_params_)

    return best_rf
