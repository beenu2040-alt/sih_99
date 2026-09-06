import xgboost as xgb
from .data_loader import load_data
from .preprocessing import preprocess_data, get_splits, save_feature_schema
from .config import config
import time
import os

def train():
    print("=" * 40)
    print("SIH PS 26027 ML TRAINING")
    print("=" * 40)
    
    # Check GPU (mock as requested)
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.ml.gpu_check import check_gpu
    check_gpu()
    
    print("\nDataset: maintenance_ml_dataset.csv")
    df = load_data()
    print(f"Rows: {len(df)}")
    
    X, y, task_ids = preprocess_data(df, is_training=True)
    print(f"Features: {X.shape[1]}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(X, y)
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\nTraining XGBoost...")
    # Initialize model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        device=config['model']['device'], # Will be 'cpu'
        tree_method=config['model']['tree_method'], # Will be 'hist'
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=config['seed']
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Save model
    model_path = "artifacts/models/maintenance_priority_xgb.json"
    model.save_model(model_path)
    print(f"\nModel saved:\n{model_path}")
    
    # Save feature schema
    save_feature_schema(X)
    
    print("=" * 40)
    print("TRAINING COMPLETE")
    print("=" * 40)
    
    # Evaluate immediately
    from .evaluate import evaluate_model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    train()
