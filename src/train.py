import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.pkl')

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Trains a model and returns evaluation metrics."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Confusion_Matrix': confusion_matrix(y_test, y_pred)
    }
    
    if y_prob is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_prob)
    else:
        metrics['ROC-AUC'] = 'N/A'
        
    print(f"Results for {model_name}:")
    for k, v in metrics.items():
        if k != 'Confusion_Matrix' and k != 'Model':
            print(f"  {k}: {v:.4f}")
            
    print("  Confusion Matrix:")
    print(metrics['Confusion_Matrix'])
    
    return model, metrics

if __name__ == "__main__":
    print("--- Day 2: Model Development ---")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}. Run data_prep.py first.")
        
    data = joblib.load(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    # Model 1: Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_model, lr_metrics = train_and_evaluate(lr, X_train, X_test, y_train, y_test, "Logistic Regression")
    
    # Model 2: Random Forest (Light Version - 50 trees, limited depth)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model, rf_metrics = train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest (Light)")
    
    # Save base models for now (Day 3 will optimize and overwrite them)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'lr_base.pkl'))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_base.pkl'))
    
    print("\nBase models trained and saved to models/ directory.")
    print("Day 2 completed successfully.")
