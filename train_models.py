import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV  # Added for warning fix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Import our data preprocessing functions from Phase 1
from preprocess import preprocess_diabetes, preprocess_heart, preprocess_kidney

def train_and_evaluate(X_train, X_test, y_train, y_test, disease_name):
    print(f"\n🤖 Training Models for: {disease_name.upper()}...")
    
    # Initialize the three algorithms - Updated SVM to use CalibratedClassifierCV
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_accuracy = 0
    best_model_name = ""
    best_model_object = None
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': f"{acc*100:.2f}%",
            'Precision': f"{prec*100:.2f}%",
            'Recall': f"{rec*100:.2f}%",
            'F1-Score': f"{f1*100:.2f}%"
        })
        
        # Track the absolute best model based on accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_object = model
            
    # Print comparison table cleanly in terminal
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Serialize and save the absolute winner
    model_filename = f"model_{disease_name.lower()}.pkl"
    joblib.dump(best_model_object, model_filename)
    print(f"🌟 WINNER: {best_model_name} ({best_accuracy*100:.2f}% Acc) saved as '{model_filename}'!")

if __name__ == "__main__":
    # 1. Diabetes
    X_train, X_test, y_train, y_test = preprocess_diabetes()
    train_and_evaluate(X_train, X_test, y_train, y_test, 'diabetes')
    
    # 2. Heart Disease
    X_train, X_test, y_train, y_test = preprocess_heart()
    train_and_evaluate(X_train, X_test, y_train, y_test, 'heart')
    
    # 3. Kidney Disease
    X_train, X_test, y_train, y_test = preprocess_kidney()
    train_and_evaluate(X_train, X_test, y_train, y_test, 'kidney')
    
    print("\n🚀 Phase 2 complete! All optimal models trained clean and saved to your workspace.")