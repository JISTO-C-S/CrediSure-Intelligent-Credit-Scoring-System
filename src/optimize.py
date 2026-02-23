import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.pkl')

def plot_feature_importance(importances, feature_names, title, filename):
    """Plots and saves the top 10 feature importances."""
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and take top 10
    # For Logistic Regression, we often look at the absolute magnitude for importance
    importance_df['Abs_Importance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_DIR, filename))
    plt.close()
    print(f"Saved feature importance plot to {filename}")
    return importance_df

if __name__ == "__main__":
    print("--- Day 3: Model Optimization & Interpretation ---")
    
    data = joblib.load(PROCESSED_DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    feature_names = data['feature_names']
    
    # --- Logistic Regression Optimization ---
    print("\nTuning Logistic Regression...")
    param_grid = {'C': [0.1, 1, 10]}
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
    
    # Save optimized model
    joblib.dump(best_lr, os.path.join(MODELS_DIR, 'lr_optimized.pkl'))
    
    # Extract Feature Importances (Coefficients)
    print("\nExtracting Feature Importances (Logistic Regression)...")
    lr_importances = best_lr.coef_[0]
    lr_top_features = plot_feature_importance(
        lr_importances, 
        feature_names, 
        'Top 10 Feature Importances (Logistic Regression Coefficients)', 
        'lr_feature_importance.png'
    )
    
    # --- Random Forest Interpretation ---
    print("\nExtracting Feature Importances (Random Forest)...")
    # Load the base RF model we trained yesterday
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_base.pkl'))
    rf_importances = rf_model.feature_importances_
    
    rf_top_features = plot_feature_importance(
        rf_importances, 
        feature_names, 
        'Top 10 Feature Importances (Random Forest)', 
        'rf_feature_importance.png'
    )
    
    # We will choose the Optimized LR model as our final model because it's simpler and 
    # coefficients provide clear positive/negative directional impact for the rejection explanation system.
    # However, we'll keep both saved.
    joblib.dump(best_lr, os.path.join(MODELS_DIR, 'credit_model.pkl'))
    print("\nSaved credit_model.pkl (Optimized Logistic Regression) as the final model for Day 4.")
    print("Day 3 completed successfully.")
