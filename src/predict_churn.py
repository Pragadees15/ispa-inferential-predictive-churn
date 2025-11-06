"""
Churn Prediction Script

This script accepts new customer data and returns churn prediction.
Can be used as a standalone script or imported as a module.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from preprocess import prepare_data, encode_categorical_variables


def load_model_and_encoders(model_path="../models/churn_model.joblib"):
    """
    Load the trained model and encoders.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    dict
        Dictionary containing model and encoders
    """
    try:
        model_data = joblib.load(model_path)
        return model_data
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        print("Please train the model first using train.py")
        sys.exit(1)


def prepare_customer_data(customer_data, encoders, feature_names):
    """
    Prepare customer data for prediction (same preprocessing as training).
    
    Parameters:
    -----------
    customer_data : pd.DataFrame or dict
        Customer data (single or multiple customers)
    encoders : dict
        Dictionary containing preprocessor and target encoder
    feature_names : list
        List of feature names
        
    Returns:
    --------
    np.array
        Preprocessed feature matrix
    """
    # Convert dict to DataFrame if needed
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    # Clean TotalCharges if present
    if 'TotalCharges' in customer_data.columns:
        customer_data['TotalCharges'] = pd.to_numeric(
            customer_data['TotalCharges'], errors='coerce'
        )
        customer_data['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID if present
    if 'customerID' in customer_data.columns:
        customer_data = customer_data.drop('customerID', axis=1)
    
    # Use the same preprocessor from training
    preprocessor = encoders['preprocessor']
    X_encoded = preprocessor.transform(customer_data)
    
    return X_encoded


def predict_churn(customer_data, model_path="../models/churn_model.joblib", return_proba=True):
    """
    Predict churn for new customer data.
    
    Parameters:
    -----------
    customer_data : pd.DataFrame or dict
        Customer data (single or multiple customers)
    model_path : str
        Path to the saved model
    return_proba : bool
        Whether to return probability scores
        
    Returns:
    --------
    dict or pd.DataFrame
        Predictions with probabilities
    """
    # Load model and encoders
    model_data = load_model_and_encoders(model_path)
    model = model_data['model']
    encoders = model_data.get('encoders', {})
    feature_names = model_data.get('feature_names', [])
    
    # Prepare customer data
    X = prepare_customer_data(customer_data, encoders, feature_names)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] if return_proba else None
    
    # Convert predictions back to original labels
    target_encoder = encoders.get('target')
    if target_encoder:
        predictions_labels = target_encoder.inverse_transform(predictions)
    else:
        predictions_labels = ['Yes' if p == 1 else 'No' for p in predictions]
    
    # Create results DataFrame
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    results = customer_data.copy()
    results['Churn_Prediction'] = predictions_labels
    results['Churn_Probability'] = probabilities if return_proba else None
    results['Churn_Risk'] = pd.cut(
        probabilities if return_proba else [0.5] * len(predictions),
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    ) if return_proba else None
    
    return results


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict customer churn')
    parser.add_argument('--data', type=str, help='Path to CSV file with customer data')
    parser.add_argument('--model', type=str, default='../models/churn_model.joblib',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    
    args = parser.parse_args()
    
    if args.data:
        # Load data from file
        customer_data = pd.read_csv(args.data)
        results = predict_churn(customer_data, args.model)
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print("\n=== Churn Predictions ===")
            print(results.to_string(index=False))
    else:
        # Example usage with sample data
        print("No input data provided. Using example customer data...")
        example_customer = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 99.65,
            'TotalCharges': 1189.8
        }
        
        results = predict_churn(example_customer, args.model)
        print("\n=== Example Prediction ===")
        print(results[['Churn_Prediction', 'Churn_Probability', 'Churn_Risk']].to_string(index=False))


if __name__ == "__main__":
    main()

