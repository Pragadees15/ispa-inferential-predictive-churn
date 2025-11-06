"""
Model Training Module for Customer Churn Prediction

This module handles:
- Training Logistic Regression model
- Training CHAID (Decision Tree) model
- Model comparison and selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    y : np.array
        Target vector
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training target
    random_state : int
        Random seed
        
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Trained model
    """
    print("\n=== Training Logistic Regression ===")
    model = LogisticRegression(random_state=random_state, max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return model


def train_chaid_tree(X_train, y_train, max_depth=5, min_samples_split=50, random_state=42):
    """
    Train CHAID-style Decision Tree model.
    
    Note: CHAID is a specific algorithm, but we use sklearn's DecisionTreeClassifier
    with appropriate parameters to create interpretable decision rules.
    
    Parameters:
    -----------
    X_train : np.array
        Training features
    y_train : np.array
        Training target
    max_depth : int
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split a node
    random_state : int
        Random seed
        
    Returns:
    --------
    sklearn.tree.DecisionTreeClassifier
        Trained decision tree model
    """
    print("\n=== Training CHAID Decision Tree ===")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=20,
        random_state=random_state,
        criterion='gini'
    )
    model.fit(X_train, y_train)
    print("CHAID Decision Tree training completed.")
    return model


def extract_decision_rules(tree_model, feature_names):
    """
    Extract readable decision rules from the tree model.
    
    Parameters:
    -----------
    tree_model : sklearn.tree.DecisionTreeClassifier
        Trained decision tree
    feature_names : list
        List of feature names
        
    Returns:
    --------
    str
        Text representation of decision rules
    """
    tree_rules = export_text(tree_model, feature_names=feature_names, max_depth=10)
    return tree_rules


def interpret_tree_rules(tree_model, feature_names, X_train, y_train):
    """
    Extract business-interpretable rules from the decision tree.
    
    Parameters:
    -----------
    tree_model : sklearn.tree.DecisionTreeClassifier
        Trained decision tree
    feature_names : list
        List of feature names
    X_train : np.array
        Training features
    y_train : np.array
        Training target
        
    Returns:
    --------
    list
        List of interpretable business rules
    """
    rules = []
    
    # Get tree structure
    tree = tree_model.tree_
    
    def get_path(node_id, path=""):
        """Recursively extract paths from tree."""
        if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
            samples = tree.n_node_samples[node_id]
            values = tree.value[node_id][0]
            churn_prob = values[1] / (values[0] + values[1]) if (values[0] + values[1]) > 0 else 0
            
            if churn_prob > 0.5:  # High churn probability
                rules.append({
                    'path': path,
                    'samples': samples,
                    'churn_probability': churn_prob,
                    'churn_count': int(values[1]),
                    'no_churn_count': int(values[0])
                })
        else:
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            
            # Left child (condition true)
            left_path = f"{path} AND {feature} <= {threshold:.2f}" if path else f"{feature} <= {threshold:.2f}"
            get_path(tree.children_left[node_id], left_path)
            
            # Right child (condition false)
            right_path = f"{path} AND {feature} > {threshold:.2f}" if path else f"{feature} > {threshold:.2f}"
            get_path(tree.children_right[node_id], right_path)
    
    get_path(0)
    
    # Sort by churn probability
    rules.sort(key=lambda x: x['churn_probability'], reverse=True)
    
    return rules


def train_models(X, y, feature_names, test_size=0.2, random_state=42):
    """
    Train both Logistic Regression and CHAID models.
    
    Parameters:
    -----------
    X : np.array
        Feature matrix
    y : np.array
        Target vector
    feature_names : list
        List of feature names
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary containing models, test data, and metadata
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, random_state)
    
    # Train CHAID Tree
    chaid_model = train_chaid_tree(X_train, y_train, max_depth=5, random_state=random_state)
    
    # Extract decision rules
    tree_rules = extract_decision_rules(chaid_model, feature_names)
    business_rules = interpret_tree_rules(chaid_model, feature_names, X_train, y_train)
    
    # Print top business rules
    print("\n=== Top Business Rules from CHAID Tree ===")
    for i, rule in enumerate(business_rules[:5], 1):
        print(f"\nRule {i}:")
        print(f"  Condition: {rule['path']}")
        print(f"  Churn Probability: {rule['churn_probability']:.2%}")
        print(f"  Samples: {rule['samples']}")
        print(f"  Churn Count: {rule['churn_count']}, No Churn: {rule['no_churn_count']}")
    
    return {
        'lr_model': lr_model,
        'chaid_model': chaid_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'tree_rules': tree_rules,
        'business_rules': business_rules
    }


def save_model(model, filepath, metadata=None):
    """
    Save trained model using joblib.
    
    Parameters:
    -----------
    model : sklearn model or dict
        Trained model to save, or dictionary containing model and metadata
    filepath : str
        Path to save the model
    metadata : dict, optional
        Additional metadata to save with the model (if model is not a dict)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # If model is already a dictionary, save it directly
    if isinstance(model, dict):
        joblib.dump(model, filepath)
    else:
        # Otherwise, create a dictionary with model and metadata
        save_dict = {
            'model': model,
            'metadata': metadata or {}
        }
        joblib.dump(save_dict, filepath)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load saved model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    dict
        Dictionary containing model and metadata
    """
    return joblib.load(filepath)


if __name__ == "__main__":
    # Test training pipeline
    from preprocess import prepare_data
    
    file_path = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X, y, feature_names, encoders, df = prepare_data(file_path)
    
    results = train_models(X, y, feature_names)
    
    # Save models
    save_model(results['lr_model'], "../models/lr_model.joblib")
    save_model(results['chaid_model'], "../models/chaid_model.joblib", 
               {'tree_rules': results['tree_rules'], 'business_rules': results['business_rules']})

