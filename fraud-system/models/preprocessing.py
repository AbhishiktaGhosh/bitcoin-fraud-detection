import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(features_path=None, edgelist_path=None, classes_path=None):
    """
    Loads features, edgelist, and classes DataFrames.
    If paths are not provided, it attempts to find them in the final_year _project directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(base_dir)
    
    # Default paths
    if features_path is None:
        features_path = os.path.join(root_dir, "final_year _project", "elliptic_txs_features.csv")
    if edgelist_path is None:
        edgelist_path = os.path.join(root_dir, "final_year _project", "elliptic_txs_edgelist.csv")
    if classes_path is None:
        classes_path = os.path.join(root_dir, "final_year _project", "elliptic_txs_classes.csv")
        
    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path, header=None)
    
    print(f"Loading edgelist from {edgelist_path}...")
    edges_df = pd.read_csv(edgelist_path)
    
    print(f"Loading classes from {classes_path}...")
    classes_df = pd.read_csv(classes_path)
    
    return features_df, edges_df, classes_df

def clean_data(features_df, classes_df):
    """
    Renames columns, merges features with classes, fills missing values,
    and returns features and labels.
    """
    # Define columns: id, time_step, and 165 features
    features_df = features_df.copy()
    features_df.columns = ['id', 'time_step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
    
    # Merge features with classes
    df_merged = pd.merge(features_df, classes_df, left_on='id', right_on='txId', how='left')
    df_merged['class'] = df_merged['class'].fillna('unknown')
    
    return df_merged

def preprocess_data(df_merged, scaler_path=None, fit_scaler=False):
    """
    Preprocesses the data: filters out unknown classes, encodes target labels,
    scales transaction and aggregate features.
    
    Class mapping:
    - '1' (illicit/fraud) -> 1
    - '2' (licit/legitimate) -> 0
    """
    # Labeled data only
    df_labeled = df_merged[df_merged['class'] != 'unknown'].copy()
    
    feature_cols = [col for col in df_merged.columns if col.startswith('trans_feat_') or col.startswith('agg_feat_')]
    X = df_labeled[feature_cols].copy()
    
    # Map target label
    # '1' is fraud/illicit, '2' is normal/licit
    y = (df_labeled['class'] == '1').astype(int)
    
    # Set up scaler
    if scaler_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scaler_path = os.path.join(base_dir, "saved_models", "scaler.pkl")
        
    scaler = None
    if not fit_scaler and os.path.exists(scaler_path):
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
    else:
        print("Fitting and saving a new StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_labeled.index)
    
    return X_scaled_df, y, df_labeled['id'].values, scaler
