import os
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def train_model(X, y):
    """
    Trains tabular machine learning models on features X and labels y.
    Returns a dictionary of trained models:
    - 'rf': Random Forest
    - 'xgb': XGBoost
    - 'voting': Voting Classifier (RF + XGBoost)
    - 'stacking': Stacking Classifier (RF + XGBoost with Logistic Regression meta-classifier)
    """
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    print("Training XGBoost Classifier...")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb.fit(X, y)
    
    print("Training Voting Classifier...")
    voting = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft',
        n_jobs=-1
    )
    voting.fit(X, y)
    
    print("Training Stacking Classifier...")
    stacking = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        final_estimator=LogisticRegression(),
        n_jobs=-1
    )
    stacking.fit(X, y)
    
    return {
        'rf': rf,
        'xgb': xgb,
        'voting': voting,
        'stacking': stacking
    }

def save_model(model, filepath):
    """
    Serializes a model to the specified filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        joblib.dump(model, f)
    print(f"Saved model to {filepath}")

def load_model(filepath):
    """
    Loads a serialized model from the specified filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, 'rb') as f:
        model = joblib.load(f)
    return model
