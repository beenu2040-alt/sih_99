import xgboost as xgb
import pandas as pd
from .preprocessing import preprocess_data
import os

class PriorityPredictor:
    def __init__(self, model_path="artifacts/models/maintenance_priority_xgb.json"):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
    def score_to_class(self, score):
        # Domain logic based on the existing dataset distribution
        # > 80: CRITICAL
        # > 60: HIGH
        # > 40: MEDIUM
        # <= 40: LOW
        if score > 80:
            return "CRITICAL"
        elif score > 60:
            return "HIGH"
        elif score > 40:
            return "MEDIUM"
        else:
            return "LOW"
            
    def predict_priority(self, df):
        """
        Predicts priority score and class for given dataframe.
        """
        X, _, task_ids = preprocess_data(df, is_training=False)
        
        scores = self.model.predict(X)
        
        results = pd.DataFrame({
            'task_id': task_ids,
            'predicted_priority_score': scores
        })
        
        results['predicted_priority_class'] = results['predicted_priority_score'].apply(self.score_to_class)
        
        return results

def predict_priority(data):
    predictor = PriorityPredictor()
    return predictor.predict_priority(data)
