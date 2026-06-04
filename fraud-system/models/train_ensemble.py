import os
import sys

# Add parent directory to path so we can import models package
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from models.preprocessing import load_data, clean_data, preprocess_data
from models.ensemble import train_model, save_model

def main():
    print("=== STARTING TABULAR ENSEMBLE TRAINING ===")
    
    # 1. Load raw dataset files
    try:
        features_df, edges_df, classes_df = load_data()
    except Exception as e:
        print(f"Error loading raw data: {e}")
        sys.exit(1)
        
    # 2. Clean and merge features with target classes
    print("Cleaning and merging features...")
    df_merged = clean_data(features_df, classes_df)
    
    # 3. Fit scaler and scale features (this saves saved_models/scaler.pkl)
    print("Preprocessing data and fitting StandardScaler...")
    scaler_path = os.path.join(base_dir, "saved_models", "scaler.pkl")
    X_scaled, y, node_ids, scaler = preprocess_data(df_merged, scaler_path=scaler_path, fit_scaler=True)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # 4. Train Random Forest, XGBoost, Voting, Stacking ensemble models
    print("Training tabular ensemble models...")
    trained_models = train_model(X_scaled, y)
    
    # 5. Save serialized models to saved_models/ folder
    saved_models_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    save_model(trained_models['rf'], os.path.join(saved_models_dir, "rf_model.pkl"))
    save_model(trained_models['xgb'], os.path.join(saved_models_dir, "xgb_model.pkl"))
    save_model(trained_models['voting'], os.path.join(saved_models_dir, "voting_classifier.pkl"))
    save_model(trained_models['stacking'], os.path.join(saved_models_dir, "stacking_classifier.pkl"))
    
    print("=== TRAINING COMPLETE. ALL MODELS SERIALIZED ===")

if __name__ == "__main__":
    main()
