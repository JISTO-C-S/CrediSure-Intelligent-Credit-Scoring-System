import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SRC_DATA_FILE = os.path.join(BASE_DIR, 'statlog+german+credit+data', 'german.data')
OUT_CSV_FILE = os.path.join(DATA_DIR, 'german_credit.csv')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks')

# Column names based on german.doc
COLUMNS = [
    'checking_status', 'duration_months', 'credit_history', 'purpose', 
    'credit_amount', 'savings_status', 'employment_duration', 
    'installment_rate', 'personal_status_sex', 'other_debtors', 
    'residence_duration', 'property', 'age', 'other_installment_plans', 
    'housing', 'existing_credits', 'job', 'num_dependents', 
    'telephone', 'foreign_worker', 'target'
]

def load_data(filepath, columns):
    """Loads the raw german.data space-separated file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source data file not found at {filepath}")
    
    # The file consists of space-separated values
    df = pd.read_csv(filepath, sep=' ', names=columns)
    
    # Target variable: 1 = Good, 2 = Bad (from german.doc)
    # Convert it to 0 = Good, 1 = Bad for standard binary classification
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    return df

def perform_eda(df, out_dir):
    """Generates basic EDA plots and saves them."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # 1. Distribution of Target Variable
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='target', palette='Set2')
    plt.title('Distribution of Credit Risk (0=Good, 1=Bad)')
    plt.savefig(os.path.join(out_dir, 'target_distribution.png'))
    plt.close()

    # 2. Credit Amount Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['credit_amount'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Credit Amount')
    plt.savefig(os.path.join(out_dir, 'credit_amount_distribution.png'))
    plt.close()
    
    # 3. Correlation Matrix (Numerical features only)
    plt.figure(figsize=(10, 8))
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(os.path.join(out_dir, 'correlation_matrix.png'))
    plt.close()
    
    print(f"EDA plots saved to {out_dir}")

def preprocess_data(df):
    """Handles missing values, encodes categories, and scales numericals."""
    print("Starting data preprocessing...")
    
    # German credit dataset technically doesn't have traditional missing values (they are categorized)
    # But let's add a robust check just in case
    if df.isnull().sum().any():
        print("Handling missing values...")
        df.fillna(df.median(numeric_only=True), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    # 1. Encode Categorical Variables (Label Encoding for tree-based + log reg compatibility in this simple setup)
    # For a robust production model, OneHotEncoding might be preferred for LogReg, 
    # but the project plan focuses on a simple, light implementation. We will use LabelEncoder.
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # 2. Train-Test Split (80-20)
    # It is crucial to split BEFORE scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale Numerical Features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("Preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler, label_encoders, numerical_cols, categorical_cols

if __name__ == "__main__":
    print("--- Day 1: Data Preparation ---")
    
    # 1. Load Data
    df = load_data(SRC_DATA_FILE, COLUMNS)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Save a clean CSV version for raw inspection later if needed
    df.to_csv(OUT_CSV_FILE, index=False)
    print(f"Saved raw CSV with headers to {OUT_CSV_FILE}")
    
    # 2. EDA
    perform_eda(df, NOTEBOOK_DIR)
    
    # 3. Preprocess & Split
    X_train, X_test, y_train, y_test, scaler, label_encoders, num_cols, cat_cols = preprocess_data(df)
    
    # We will save the preprocessed data using joblib so Day 2 can pick it up easily
    import joblib
    
    processed_data_path = os.path.join(DATA_DIR, 'processed_data.pkl')
    data_dict = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'label_encoders': label_encoders,
        'numerical_cols': num_cols, 'categorical_cols': cat_cols,
        'feature_names': X_train.columns.tolist()
    }
    joblib.dump(data_dict, processed_data_path)
    print(f"Saved processed data and artifacts to {processed_data_path}")
    print("Day 1 completed successfully.")
