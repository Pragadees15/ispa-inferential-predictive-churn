"""
Model Evaluation Module for Customer Churn Prediction

This module handles:
- Model performance metrics (Accuracy, Precision, Recall, F1-score)
- ROC Curve and AUC score
- Gains Chart
- Lift Chart
- Model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import os
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive performance metrics.
    
    Parameters:
    -----------
    y_true : np.array
        True labels
    y_pred : np.array
        Predicted labels
    y_pred_proba : np.array, optional
        Predicted probabilities (for ROC-AUC)
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics['roc_auc'] = None
    
    return metrics


def plot_roc_curve(y_true, y_pred_proba_lr, y_pred_proba_tree, model_names=None, save_path=None):
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    y_true : np.array
        True labels
    y_pred_proba_lr : np.array
        Predicted probabilities from Logistic Regression
    y_pred_proba_tree : np.array
        Predicted probabilities from Decision Tree
    model_names : list, optional
        Names of the models
    save_path : str, optional
        Path to save the plot
    """
    if model_names is None:
        model_names = ['Logistic Regression', 'CHAID Decision Tree']
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_pred_proba_lr)
    auc_lr = roc_auc_score(y_true, y_pred_proba_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'{model_names[0]} (AUC = {auc_lr:.3f})', linewidth=2)
    
    # Plot ROC curve for Decision Tree
    fpr_tree, tpr_tree, _ = roc_curve(y_true, y_pred_proba_tree)
    auc_tree = roc_auc_score(y_true, y_pred_proba_tree)
    plt.plot(fpr_tree, tpr_tree, label=f'{model_names[1]} (AUC = {auc_tree:.3f})', linewidth=2)
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def calculate_gains_lift(y_true, y_pred_proba, n_bins=10):
    """
    Calculate Gains and Lift metrics.
    
    Parameters:
    -----------
    y_true : np.array
        True labels
    y_pred_proba : np.array
        Predicted probabilities
    n_bins : int
        Number of bins for decile analysis
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing gains and lift metrics
    """
    # Create DataFrame with predictions and actuals
    df = pd.DataFrame({
        'actual': y_true,
        'prob': y_pred_proba
    })
    
    # Sort by predicted probability (descending)
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    # Create deciles
    df['decile'] = pd.qcut(df.index, q=n_bins, labels=False, duplicates='drop') + 1
    
    # Calculate metrics per decile
    gains_lift = df.groupby('decile').agg({
        'actual': ['count', 'sum', 'mean'],
        'prob': 'mean'
    }).reset_index()
    
    gains_lift.columns = ['decile', 'total_customers', 'churn_customers', 'churn_rate', 'avg_prob']
    
    # Calculate cumulative metrics
    gains_lift['cumulative_customers'] = gains_lift['total_customers'].cumsum()
    gains_lift['cumulative_churn'] = gains_lift['churn_customers'].cumsum()
    gains_lift['cumulative_churn_rate'] = gains_lift['cumulative_churn'] / gains_lift['cumulative_customers']
    
    # Calculate gains (% of total churners captured)
    total_churners = df['actual'].sum()
    gains_lift['gain'] = (gains_lift['cumulative_churn'] / total_churners) * 100
    
    # Calculate lift (how many times better than random)
    overall_churn_rate = df['actual'].mean()
    gains_lift['lift'] = gains_lift['churn_rate'] / overall_churn_rate
    
    # Calculate cumulative lift
    gains_lift['cumulative_lift'] = gains_lift['cumulative_churn_rate'] / overall_churn_rate
    
    return gains_lift


def plot_gains_chart(gains_lift_lr, gains_lift_tree, model_names=None, save_path=None):
    """
    Plot Gains Chart for multiple models.
    
    Parameters:
    -----------
    gains_lift_lr : pd.DataFrame
        Gains/Lift metrics for Logistic Regression
    gains_lift_tree : pd.DataFrame
        Gains/Lift metrics for Decision Tree
    model_names : list, optional
        Names of the models
    save_path : str, optional
        Path to save the plot
    """
    if model_names is None:
        model_names = ['Logistic Regression', 'CHAID Decision Tree']
    
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative gains
    plt.plot(gains_lift_lr['decile'], gains_lift_lr['gain'], 
             marker='o', label=model_names[0], linewidth=2, markersize=8)
    plt.plot(gains_lift_tree['decile'], gains_lift_tree['gain'], 
             marker='s', label=model_names[1], linewidth=2, markersize=8)
    
    # Plot baseline (random model)
    baseline_gain = [i * 10 for i in range(1, 11)]
    plt.plot(range(1, 11), baseline_gain, 'k--', label='Baseline (Random)', linewidth=1)
    
    plt.xlabel('Decile', fontsize=12)
    plt.ylabel('Cumulative Gain (%)', fontsize=12)
    plt.title('Gains Chart - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 11))
    plt.ylim([0, 100])
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gains chart saved to {save_path}")
    
    plt.show()


def plot_lift_chart(gains_lift_lr, gains_lift_tree, model_names=None, save_path=None):
    """
    Plot Lift Chart for multiple models.
    
    Parameters:
    -----------
    gains_lift_lr : pd.DataFrame
        Gains/Lift metrics for Logistic Regression
    gains_lift_tree : pd.DataFrame
        Gains/Lift metrics for Decision Tree
    model_names : list, optional
        Names of the models
    save_path : str, optional
        Path to save the plot
    """
    if model_names is None:
        model_names = ['Logistic Regression', 'CHAID Decision Tree']
    
    plt.figure(figsize=(12, 8))
    
    # Plot lift
    plt.plot(gains_lift_lr['decile'], gains_lift_lr['lift'], 
             marker='o', label=model_names[0], linewidth=2, markersize=8)
    plt.plot(gains_lift_tree['decile'], gains_lift_tree['lift'], 
             marker='s', label=model_names[1], linewidth=2, markersize=8)
    
    # Plot baseline (lift = 1)
    plt.axhline(y=1, color='k', linestyle='--', label='Baseline (Lift = 1)', linewidth=1)
    
    plt.xlabel('Decile', fontsize=12)
    plt.ylabel('Lift', fontsize=12)
    plt.title('Lift Chart - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 11))
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Lift chart saved to {save_path}")
    
    plt.show()


def evaluate_models(lr_model, chaid_model, X_test, y_test, feature_names, save_dir="reports/visuals"):
    """
    Comprehensive model evaluation.
    
    Parameters:
    -----------
    lr_model : sklearn model
        Trained Logistic Regression model
    chaid_model : sklearn model
        Trained CHAID Decision Tree model
    X_test : np.array
        Test features
    y_test : np.array
        Test labels
    feature_names : list
        List of feature names
    save_dir : str
        Directory to save visualizations
        
    Returns:
    --------
    pd.DataFrame
        Comparison table of model metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    y_pred_tree = chaid_model.predict(X_test)
    y_pred_proba_tree = chaid_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics_lr = calculate_metrics(y_test, y_pred_lr, y_pred_proba_lr)
    metrics_tree = calculate_metrics(y_test, y_pred_tree, y_pred_proba_tree)
    
    # Calculate gains and lift
    gains_lift_lr = calculate_gains_lift(y_test, y_pred_proba_lr)
    gains_lift_tree = calculate_gains_lift(y_test, y_pred_proba_tree)
    
    # Get lift at top decile
    lift_lr = gains_lift_lr.iloc[0]['lift']
    lift_tree = gains_lift_tree.iloc[0]['lift']
    
    # Print metrics
    print("\n=== Logistic Regression Metrics ===")
    print(f"Accuracy:  {metrics_lr['accuracy']:.4f}")
    print(f"Precision: {metrics_lr['precision']:.4f}")
    print(f"Recall:    {metrics_lr['recall']:.4f}")
    print(f"F1-Score:  {metrics_lr['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics_lr['roc_auc']:.4f}")
    print(f"Lift (Top Decile): {lift_lr:.2f}")
    
    print("\n=== CHAID Decision Tree Metrics ===")
    print(f"Accuracy:  {metrics_tree['accuracy']:.4f}")
    print(f"Precision: {metrics_tree['precision']:.4f}")
    print(f"Recall:    {metrics_tree['recall']:.4f}")
    print(f"F1-Score:  {metrics_tree['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics_tree['roc_auc']:.4f}")
    print(f"Lift (Top Decile): {lift_tree:.2f}")
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'CHAID Decision Tree'],
        'Accuracy': [metrics_lr['accuracy'], metrics_tree['accuracy']],
        'ROC-AUC': [metrics_lr['roc_auc'], metrics_tree['roc_auc']],
        'Lift': [lift_lr, lift_tree],
        'Best Use Case': [
            'General purpose, interpretable coefficients',
            'Rule-based decisions, high interpretability'
        ]
    })
    
    print("\n=== Model Comparison Table ===")
    print(comparison_df.to_string(index=False))
    
    # Plot ROC curves
    plot_roc_curve(y_test, y_pred_proba_lr, y_pred_proba_tree,
                   save_path=os.path.join(save_dir, 'roc_curves.png'))
    
    # Plot Gains chart
    plot_gains_chart(gains_lift_lr, gains_lift_tree,
                     save_path=os.path.join(save_dir, 'gains_chart.png'))
    
    # Plot Lift chart
    plot_lift_chart(gains_lift_lr, gains_lift_tree,
                    save_path=os.path.join(save_dir, 'lift_chart.png'))
    
    # Print detailed gains/lift table
    print("\n=== Gains and Lift Analysis (Top 3 Deciles) ===")
    print("\nLogistic Regression:")
    print(gains_lift_lr[['decile', 'total_customers', 'churn_customers', 
                         'gain', 'lift']].head(3).to_string(index=False))
    
    print("\nCHAID Decision Tree:")
    print(gains_lift_tree[['decile', 'total_customers', 'churn_customers', 
                           'gain', 'lift']].head(3).to_string(index=False))
    
    return comparison_df, gains_lift_lr, gains_lift_tree


if __name__ == "__main__":
    # Test evaluation pipeline
    from preprocess import prepare_data
    from train import train_models
    
    file_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X, y, feature_names, encoders, df = prepare_data(file_path)
    
    results = train_models(X, y, feature_names)
    
    comparison_df, gains_lift_lr, gains_lift_tree = evaluate_models(
        results['lr_model'], results['chaid_model'],
        results['X_test'], results['y_test'], feature_names
    )

