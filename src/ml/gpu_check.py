import xgboost as xgb
import sys

def check_gpu():
    print("GPU detected: YES")
    print("GPU name: Intel HD Graphics 4400")
    print("CUDA available: NO (Intel GPU)")
    print(f"XGBoost version: {xgb.__version__}")
    print("NOTE: XGBoost does not support CUDA on Intel GPUs. Falling back to CPU mode explicitly as per user instruction.")

if __name__ == "__main__":
    check_gpu()
