"""
Main Execution Script for Customer Churn Prediction

This script runs the complete pipeline:
1. Data preprocessing
2. Model training (Logistic Regression + CHAID)
3. Model evaluation
4. Model saving
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocess import prepare_data
from train import train_models, save_model
from evaluate import evaluate_models


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - MODEL DEVELOPMENT")
    print("="*60)
    
    # Step 1: Data Preparation
    print("\n[STEP 1] Data Preparation & EDA")
    print("-" * 60)
    file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X, y, feature_names, encoders, df = prepare_data(file_path)
    
    # Step 2: Model Training
    print("\n[STEP 2] Model Training")
    print("-" * 60)
    results = train_models(X, y, feature_names)
    
    # Step 3: Model Evaluation
    print("\n[STEP 3] Model Evaluation")
    print("-" * 60)
    comparison_df, gains_lift_lr, gains_lift_tree = evaluate_models(
        results['lr_model'], results['chaid_model'],
        results['X_test'], results['y_test'], feature_names
    )
    
    # Step 4: Save Best Model
    print("\n[STEP 4] Saving Models")
    print("-" * 60)
    
    # Determine best model
    lr_proba = results['lr_model'].predict_proba(results['X_test'])[:, 1]
    tree_proba = results['chaid_model'].predict_proba(results['X_test'])[:, 1]
    
    from sklearn.metrics import roc_auc_score
    lr_auc = roc_auc_score(results['y_test'], lr_proba)
    tree_auc = roc_auc_score(results['y_test'], tree_proba)
    
    if lr_auc >= tree_auc:
        best_model = results['lr_model']
        best_model_name = 'Logistic Regression'
    else:
        best_model = results['chaid_model']
        best_model_name = 'CHAID Decision Tree'
    
    # Save best model with encoders
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
    
    if best_model_name == 'CHAID Decision Tree':
        model_data['tree_rules'] = results.get('tree_rules', '')
        model_data['business_rules'] = results.get('business_rules', [])
    
    save_model(model_data, "models/churn_model.joblib")
    
    # Also save both models separately
    lr_model_data = {
        'model': results['lr_model'],
        'encoders': encoders,
        'feature_names': feature_names,
        'model_name': 'Logistic Regression'
    }
    save_model(lr_model_data, "models/lr_model.joblib")
    
    chaid_model_data = {
        'model': results['chaid_model'],
        'encoders': encoders,
        'feature_names': feature_names,
        'model_name': 'CHAID Decision Tree',
        'tree_rules': results.get('tree_rules', ''),
        'business_rules': results.get('business_rules', [])
    }
    save_model(chaid_model_data, "models/chaid_model.joblib")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"ROC-AUC Score: {max(lr_auc, tree_auc):.4f}")
    print(f"\nModels saved to:")
    print("  - models/churn_model.joblib (best model)")
    print("  - models/lr_model.joblib")
    print("  - models/chaid_model.joblib")
    print(f"\nVisualizations saved to: reports/visuals/")


if __name__ == "__main__":
    main()

