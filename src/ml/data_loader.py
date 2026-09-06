import pandas as pd

def load_data(filepath="data/processed/maintenance_ml_dataset.csv"):
    df = pd.read_csv(filepath)
    return df
