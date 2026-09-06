import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import os

def evaluate_model(model=None, X_test=None, y_test=None):
    print("\nEvaluation:")
    
    if model is None or X_test is None or y_test is None:
        from .data_loader import load_data
        from .preprocessing import preprocess_data, get_splits
        df = load_data()
        X, y, _ = preprocess_data(df, is_training=False)
        _, _, X_test, _, _, y_test = get_splits(X, y)
        
        model = xgb.XGBRegressor()
        model.load_model("artifacts/models/maintenance_priority_xgb.json")
        
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    
    with open("artifacts/metrics/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Priority Score")
    plt.ylabel("Predicted Priority Score")
    plt.title("Actual vs Predicted Priority Score")
    plt.savefig("artifacts/plots/actual_vs_predicted.png")
    plt.close()
    
    # Plot Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.savefig("artifacts/plots/residuals.png")
    plt.close()
    
    # Feature Importance
    importance = model.feature_importances_
    features = X_test.columns
    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances")
    plt.savefig("artifacts/plots/feature_importance.png")
    plt.close()
    
    feat_imp.to_csv("artifacts/feature_importance.csv", index=False)

if __name__ == "__main__":
    evaluate_model()
