"""
Data Preprocessing Module for Customer Churn Prediction

This module handles:
- Loading and cleaning the dataset
- Handling missing values
- Encoding categorical variables
- Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def clean_data(df):
    """
    Clean the dataset:
    - Convert TotalCharges to numeric
    - Handle missing values
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()
    
    # Convert TotalCharges to numeric, replacing empty strings with NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Check for missing values
    missing_count = df_clean['TotalCharges'].isna().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values in TotalCharges")
        # Fill missing values with 0 (new customers with no charges yet)
        df_clean['TotalCharges'].fillna(0, inplace=True)
        print("Missing values filled with 0")
    
    # Drop customerID as it's not useful for modeling
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
    
    return df_clean


def encode_categorical_variables(df, target_col='Churn'):
    """
    Encode categorical variables using Label Encoding and One-Hot Encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with categorical variables
    target_col : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X_encoded, y_encoded, feature_names, label_encoders_dict)
        - X_encoded: Encoded features
        - y_encoded: Encoded target
        - feature_names: List of feature names after encoding
        - label_encoders_dict: Dictionary of label encoders for inverse transform
    """
    df_encoded = df.copy()
    
    # Separate features and target
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    # Identify categorical and numerical columns
    # Include both 'object' and pandas 'category' dtypes to ensure binned labels
    # like TenureGroup are properly one-hot encoded instead of passed through.
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode target variable using LabelEncoder
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y)
    
    # Store label encoders for categorical columns that need label encoding
    label_encoders = {}
    
    # Apply One-Hot Encoding to categorical variables
    # Use ColumnTransformer for cleaner code
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    X_encoded = preprocessor.fit_transform(X)
    
    # Get feature names after encoding
    feature_names = numerical_cols.copy()
    for col in categorical_cols:
        categories = preprocessor.named_transformers_['cat'].categories_[categorical_cols.index(col)]
        for cat in categories[1:]:  # Skip first category (drop='first')
            feature_names.append(f"{col}_{cat}")
    
    print(f"Total features after encoding: {len(feature_names)}")
    
    # Store preprocessor for later use
    label_encoders['preprocessor'] = preprocessor
    label_encoders['target'] = label_encoder_y
    
    return X_encoded, y_encoded, feature_names, label_encoders


def get_data_summary(df):
    """
    Get summary statistics of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Get value counts for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return summary


def prepare_data(file_path, target_col='Churn'):
    """
    Complete data preparation pipeline.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    target_col : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X, y, feature_names, encoders, df_original)
    """
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Get summary
    summary = get_data_summary(df_clean)
    print("\n=== Dataset Summary ===")
    print(f"Shape: {summary['shape']}")
    print(f"\nMissing Values:\n{summary['missing_values']}")
    
    # Encode variables
    X, y, feature_names, encoders = encode_categorical_variables(df_clean, target_col)
    
    return X, y, feature_names, encoders, df_clean


if __name__ == "__main__":
    # Test the preprocessing pipeline
    file_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X, y, feature_names, encoders, df = prepare_data(file_path)
    print(f"\nPreprocessed data shape: X={X.shape}, y={y.shape}")

