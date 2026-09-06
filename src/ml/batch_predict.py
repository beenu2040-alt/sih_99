from .data_loader import load_data
from .predict import predict_priority

def batch_predict():
    print("Running batch prediction...")
    df = load_data()
    
    results = predict_priority(df)
    
    output_path = "data/processed/maintenance_predictions.csv"
    results.to_csv(output_path, index=False)
    
    print(f"Batch prediction complete. Saved {len(results)} predictions to {output_path}")
    print("Sample output:")
    print(results.head())

if __name__ == "__main__":
    batch_predict()
