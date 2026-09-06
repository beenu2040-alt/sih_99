import pandas as pd
from sklearn.model_selection import train_test_split
import json
from .config import config

def preprocess_data(df, is_training=True):
    """
    Prepares features and target.
    """
    # Columns to remove to prevent data leakage or because they are identifiers
    leakage_cols = ['priority_class', 'status']
    id_col = 'task_id'
    
    target_col = config['target']
    
    features_removed = leakage_cols + ([target_col] if target_col in df.columns else []) + [id_col]
    
    if is_training:
        print("Target column:")
        print(target_col)
        print("\nFeatures removed:")
        print(features_removed)
        
    # Separate identifiers and targets
    task_ids = df[id_col]
    y = df[target_col] if target_col in df.columns else None
    
    X = df.drop(columns=[col for col in features_removed if col in df.columns])
    
    if is_training:
        print("\nFeatures used:")
        print(X.columns.tolist())
        
    # Convert object/string columns to category for XGBoost
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        
    return X, y, task_ids

def get_splits(X, y):
    """
    Splits into train (70%), val (15%), test (15%).
    Using fixed seed.
    """
    seed = config['seed']
    val_size = config['training']['validation_size']
    test_size = config['training']['test_size']
    
    # First split into train and temp (val+test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=temp_size, random_state=seed)
    
    # Split temp into val and test
    val_ratio = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_ratio, random_state=seed)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_feature_schema(X, schema_path="artifacts/models/feature_schema.json"):
    schema = {
        "features": X.columns.tolist(),
        "categorical_features": X.select_dtypes(include=['category']).columns.tolist(),
        "numerical_features": X.select_dtypes(exclude=['category']).columns.tolist(),
        "target_name": config['target'],
        "training_config": config,
        "random_seed": config['seed']
    }
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=4)
