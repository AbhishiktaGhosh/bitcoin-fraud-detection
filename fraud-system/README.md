# GraphGuard - Production Fraud Detection Terminal

This repository contains the production-grade, self-contained implementation of the GraphGuard Bitcoin Fraud Detection system. All components are isolated inside the `fraud-system/` directory.

## Project Structure

```
fraud-system/
│
├── notebooks/                     # Experimental notebooks
│   ├── GAT.ipynb
│   ├── GCN.ipynb
│   ├── ...
│
├── models/                        # Core algorithmic modules
│   ├── __init__.py
│   ├── preprocessing.py           # Data loaders and cleaning
│   ├── features.py                # Tabular & NetworkX graph features
│   ├── ensemble.py                # ML models (RF, XGBoost, Voting, Stacking)
│   ├── graph.py                   # Directed NetworkX graph construction
│   ├── louvain.py                 # Louvain community detection
│   └── fraud_type.py              # Structural pattern classification
│
├── pipelines/
│   ├── __init__.py
│   └── inference_pipeline.py      # Orchestrations and ONNX GNN predictions
│
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI service endpoints
│
├── dashboard/
│   └── app.py                     # Streamlit dashboard UI
│
├── data/                          # Data folder (final_probs.npy, etc.)
│
├── saved_models/                  # Pickle models and scaler
│   ├── scaler.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── voting_classifier.pkl
│   ├── stacking_classifier.pkl
│   └── gnn_checkpoints/           # ONNX GNN checkpoints
│
├── requirements.txt               # Dependencies
└── .env                           # Configurations
```

## Setup Instructions

### 1. Virtual Environment & Dependencies
Create a virtual environment and install the required libraries:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Tabular Ensemble Models
Fit the feature scaler and train the tabular machine learning models (Random Forest, XGBoost, Stacking, Voting) using:
```bash
python models/train_ensemble.py
```
This script will serialize the models into the `saved_models/` folder.

### 3. Launch FastAPI Backend
Run the backend web service (Uvicorn):
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
You can inspect the interactive documentation at http://127.0.0.1:8000/docs.

### 4. Launch Streamlit Dashboard
Run the frontend analyst UI using:
```bash
streamlit run dashboard/app.py
```
Access the terminal in your browser at http://localhost:8501.

## API Documentation

- **`GET /health`**: Health status of the model graph and loaded GNN models.
- **`POST /predict`**: Score a transaction node (by graph `node_id`) or custom feature vector.
- **`POST /cluster`**: Retrieve Louvain partitions and risk statistics for all communities.
- **`GET /cluster-graph/{cluster_id}`**: Get nodes and edge coordinates of a cluster for graph rendering.
- **`GET /model-info`**: Metadata of GNN sessions and ensemble weights.
