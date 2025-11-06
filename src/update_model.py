"""
Model Update Script

This script retrains the model when new data is provided.
It can be used to update the model with new training data.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from preprocess import prepare_data
from train import train_models, save_model
from evaluate import evaluate_models


def update_model(new_data_path, model_path="../models/churn_model.joblib", 
                 retrain_from_scratch=True, save_path=None):
    """
    Update/retrain the model with new data.
    
    Parameters:
    -----------
    new_data_path : str
        Path to new training data CSV file
    model_path : str
        Path to existing model (if retrain_from_scratch=False)
    retrain_from_scratch : bool
        If True, train from scratch. If False, try to update existing model.
    save_path : str, optional
        Path to save the updated model. If None, overwrites model_path.
        
    Returns:
    --------
    dict
        Dictionary containing updated model and evaluation results
    """
    print("="*60)
    print("MODEL UPDATE PROCESS")
    print("="*60)
    
    # Load and prepare new data
    print(f"\nLoading new data from {new_data_path}...")
    X, y, feature_names, encoders, df = prepare_data(new_data_path)
    
    # Train models
    print("\nTraining models with new data...")
    results = train_models(X, y, feature_names)
    
    # Evaluate models
    print("\nEvaluating updated models...")
    comparison_df, gains_lift_lr, gains_lift_tree = evaluate_models(
        results['lr_model'], results['chaid_model'],
        results['X_test'], results['y_test'], feature_names
    )
    
    # Determine best model based on ROC-AUC
    if results['lr_model'] and results['chaid_model']:
        lr_proba = results['lr_model'].predict_proba(results['X_test'])[:, 1]
        tree_proba = results['chaid_model'].predict_proba(results['X_test'])[:, 1]
        
        from sklearn.metrics import roc_auc_score
        lr_auc = roc_auc_score(results['y_test'], lr_proba)
        tree_auc = roc_auc_score(results['y_test'], tree_proba)
        
        if lr_auc >= tree_auc:
            best_model = results['lr_model']
            best_model_name = 'Logistic Regression'
            print(f"\nBest model: {best_model_name} (AUC: {lr_auc:.4f})")
        else:
            best_model = results['chaid_model']
            best_model_name = 'CHAID Decision Tree'
            print(f"\nBest model: {best_model_name} (AUC: {tree_auc:.4f})")
    else:
        best_model = results['lr_model']  # Default to LR
        best_model_name = 'Logistic Regression'
    
    # Save updated model
    if save_path is None:
        save_path = model_path
    
    # Prepare model data with encoders and metadata
    model_data = {
        'model': best_model,
        'encoders': encoders,
        'feature_names': feature_names,
        'model_name': best_model_name,
        'metadata': {
            'training_samples': len(results['y_train']),
            'test_samples': len(results['y_test']),
            'features': len(feature_names)
        }
    }
    
    # Also save business rules if CHAID model
    if best_model_name == 'CHAID Decision Tree':
        model_data['tree_rules'] = results.get('tree_rules', '')
        model_data['business_rules'] = results.get('business_rules', [])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model_data, save_path)
    print(f"\nUpdated model saved to {save_path}")
    
    # Also save both models separately
    joblib.dump({
        'model': results['lr_model'],
        'encoders': encoders,
        'feature_names': feature_names,
        'model_name': 'Logistic Regression'
    }, save_path.replace('churn_model', 'lr_model'))
    
    joblib.dump({
        'model': results['chaid_model'],
        'encoders': encoders,
        'feature_names': feature_names,
        'model_name': 'CHAID Decision Tree',
        'tree_rules': results.get('tree_rules', ''),
        'business_rules': results.get('business_rules', [])
    }, save_path.replace('churn_model', 'chaid_model'))
    
    return {
        'model': best_model,
        'results': results,
        'comparison': comparison_df,
        'gains_lift_lr': gains_lift_lr,
        'gains_lift_tree': gains_lift_tree
    }


def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Update/retrain churn prediction model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to new training data CSV file')
    parser.add_argument('--model', type=str, default='../models/churn_model.joblib',
                       help='Path to save updated model')
    parser.add_argument('--output', type=str,
                       help='Alternative path to save updated model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        sys.exit(1)
    
    update_model(args.data, args.model, save_path=args.output)


if __name__ == "__main__":
    main()

