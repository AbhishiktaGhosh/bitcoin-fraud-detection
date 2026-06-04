import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import onnxruntime as ort

# Add parent directory to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from models.preprocessing import load_data, clean_data, preprocess_data
from models.graph import build_graph
from models.ensemble import load_model
from models.louvain import detect_communities, get_cluster_assignments
from models.fraud_type import classify_fraud_type

class InferencePipeline:
    def __init__(self):
        self.base_dir = base_dir
        self.saved_models_dir = os.path.join(self.base_dir, "saved_models")
        self.gnn_dir = os.path.join(self.saved_models_dir, "gnn_checkpoints")
        
        # Load scaler
        scaler_path = os.path.join(self.saved_models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = load_model(scaler_path)
            print("Scaler loaded successfully.")
        else:
            self.scaler = None
            print("WARNING: Scaler not found. Run training script first.")
            
        # Load tabular models
        self.models = {}
        for m_name in ['rf', 'xgb', 'voting', 'stacking']:
            p = os.path.join(self.saved_models_dir, f"{m_name}_model.pkl" if m_name not in ['voting', 'stacking'] else f"{m_name}_classifier.pkl")
            if os.path.exists(p):
                self.models[m_name] = load_model(p)
                print(f"Tabular model '{m_name}' loaded successfully.")
            else:
                print(f"WARNING: Tabular model '{m_name}' not found.")
                
        # Load GNN ONNX models
        self.gnn_sessions = {}
        gnn_names = ['gat', 'gcn', 'gin', 'graphsage', 'gtn', 'mpnn']
        for name in gnn_names:
            onnx_path = os.path.join(self.gnn_dir, f"{name}_model.onnx")
            if os.path.exists(onnx_path):
                try:
                    self.gnn_sessions[name] = ort.InferenceSession(onnx_path)
                    print(f"GNN model '{name}' loaded successfully in ONNX Runtime.")
                except Exception as e:
                    print(f"Error loading GNN '{name}': {e}")
            else:
                print(f"WARNING: GNN model '{name}' not found at {onnx_path}.")
                
        # Load data and construct graph on startup
        self.features_df = None
        self.edges_df = None
        self.classes_df = None
        self.G = None
        self.node_probs = {}
        self.partition = {}
        self.cluster_stats = {}
        self._load_and_initialize_graph()
        
    def _load_and_initialize_graph(self):
        """Loads data and constructs graph & community partitions on startup."""
        try:
            self.features_df, self.edges_df, self.classes_df = load_data()
            self.df_merged = clean_data(self.features_df, self.classes_df)
            self.G = build_graph(self.edges_df, self.features_df)
            
            # Map node IDs to index for GNN predictions
            # The original 'id' column in features
            self.tx_ids = self.features_df.iloc[:, 0].values
            self.node_to_idx = {tx: i for i, tx in enumerate(self.tx_ids)}
            
            # Precompute node probabilities (using pre-saved final_probs.npy or fallback to tabular ensemble)
            final_probs_path = os.path.join(self.base_dir, "data", "final_probs.npy")
            if not os.path.exists(final_probs_path) and os.path.exists(os.path.join(os.path.dirname(self.base_dir), "final_probs.npy")):
                # Copy from E:\Programs\bitcoin-fraud-detection\final_probs.npy
                import shutil
                os.makedirs(os.path.join(self.base_dir, "data"), exist_ok=True)
                shutil.copy(os.path.join(os.path.dirname(self.base_dir), "final_probs.npy"), final_probs_path)
                
            if os.path.exists(final_probs_path):
                print(f"Loading precomputed GNN probabilities from {final_probs_path}...")
                probs = np.load(final_probs_path)
                for tx_id, idx in self.node_to_idx.items():
                    if idx < len(probs):
                        self.node_probs[tx_id] = float(probs[idx])
            else:
                print("No precomputed GNN probabilities found. Running GNN batch prediction...")
                self._run_batch_gnn_prediction()
                
            # Run Louvain partitioning on startup
            if len(self.node_probs) > 0:
                self.partition, self.cluster_stats = detect_communities(self.G, self.node_probs)
                print(f"Detected {len(self.cluster_stats)} communities.")
        except Exception as e:
            print(f"Error initializing graph / pipeline: {e}")
            
    def _run_batch_gnn_prediction(self):
        """Computes predictions for all nodes in the graph using the 6 GNN ONNX models."""
        if len(self.gnn_sessions) == 0:
            print("No GNN models loaded. Cannot run GNN predictions.")
            return
            
        print("Preparing features for batch GNN inference...")
        # Convert features to numpy float32 matrix
        x_data = self.features_df.iloc[:, 2:].values.astype(np.float32)
        
        # Build edge index
        edges_src = self.edges_df.iloc[:, 0].map(self.node_to_idx)
        edges_dst = self.edges_df.iloc[:, 1].map(self.node_to_idx)
        valid = edges_src.notna() & edges_dst.notna()
        src = edges_src[valid].astype(np.int64).values
        dst = edges_dst[valid].astype(np.int64).values
        edge_index_data = np.stack([src, dst], axis=0)
        
        # Run inference on all models
        p_sum = np.zeros((len(x_data), 2), dtype=np.float32)
        weights_sum = 0.0
        
        for name, session in self.gnn_sessions.items():
            print(f"Running GNN batch inference for '{name}'...")
            try:
                outputs = session.run(None, {"x": x_data, "edge_index": edge_index_data})
                out_raw = outputs[0]
                
                # Apply softmax row-wise
                exp_out = np.exp(out_raw - np.max(out_raw, axis=1, keepdims=True))
                probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
                
                # Check for 3-class models (GTN, MPNN) and slice to get licit/illicit
                if probs.shape[1] == 3:
                    probs = probs[:, 1:]
                    
                p_sum += probs
                weights_sum += 1.0
            except Exception as e:
                print(f"Failed running batch GNN '{name}': {e}")
                
        if weights_sum > 0:
            ensemble_prob = p_sum / weights_sum
            # Under target mapping: class 0 is illicit/fraud, class 1 is licit/legitimate
            # So probability of fraud is ensemble_prob[:, 0]
            fraud_probs = ensemble_prob[:, 0]
            for tx_id, idx in self.node_to_idx.items():
                if idx < len(fraud_probs):
                    self.node_probs[tx_id] = float(fraud_probs[idx])
            # Save the npy file for next startup
            os.makedirs(os.path.join(self.base_dir, "data"), exist_ok=True)
            np.save(os.path.join(self.base_dir, "data", "final_probs.npy"), fraud_probs)
            print("Batch GNN predictions computed and saved to final_probs.npy")
            
    def predict_single_node(self, node_id):
        """
        Runs prediction for a single node ID.
        Extracts its local neighborhood (ego graph), runs predictions,
        locates its community, and classifies its fraud type.
        """
        if self.G is None or node_id not in self.G:
            raise ValueError(f"Transaction ID {node_id} does not exist in the graph.")
            
        # Get precomputed fraud score (or compute dynamically on ego graph if not available)
        score = self.node_probs.get(node_id, 0.0)
        label = int(score > 0.5)
        
        # Get community information
        cluster_id = self.partition.get(node_id, -1)
        cluster_stats = self.cluster_stats.get(cluster_id, {"size": 1, "mean": 0.0, "max": 0.0, "p90": 0.0})
        
        # Extract cluster nodes for structural pattern analysis
        clusters = get_cluster_assignments(self.partition)
        community_nodes = clusters.get(cluster_id, [node_id])
        
        # Classify fraud type
        fraud_type, metrics = classify_fraud_type(self.G, community_nodes)
        
        # Extract local neighborhood (ego graph) nodes (radius 2)
        ego_nodes = list(nx.ego_graph(self.G, node_id, radius=2).nodes())
        ego_subgraph = self.G.subgraph(ego_nodes)
        
        # Retrieve neighbor details
        neighbors = []
        for n in self.G.neighbors(node_id):
            neighbors.append({
                "id": int(n),
                "score": float(self.node_probs.get(n, 0.0)),
                "label": "Fraudulent" if self.node_probs.get(n, 0.0) > 0.5 else "Legitimate",
                "relationship": "Outgoing"
            })
        for n in self.G.predecessors(node_id):
            neighbors.append({
                "id": int(n),
                "score": float(self.node_probs.get(n, 0.0)),
                "label": "Fraudulent" if self.node_probs.get(n, 0.0) > 0.5 else "Legitimate",
                "relationship": "Incoming"
            })
            
        # Retrieve features for this node
        feat_row = self.df_merged[self.df_merged['id'] == node_id]
        feature_profile = {}
        time_step = 1
        if not feat_row.empty:
            time_step = int(feat_row['time_step'].values[0])
            for col in feat_row.columns:
                if col.startswith('trans_feat_') or col.startswith('agg_feat_'):
                    feature_profile[col] = float(feat_row[col].values[0])
                    
        explanation = f"Transaction ID {node_id} is located inside cluster {cluster_id} " \
                      f"with a structural pattern resembling a '{fraud_type}'. " \
                      f"The average risk score in this cluster is {cluster_stats['mean']:.2%}, " \
                      f"and the transaction itself carries a {score:.2%} fraud probability."
                      
        return {
            "node_id": node_id,
            "fraud_score": score,
            "label": "Fraudulent" if label == 1 else "Legitimate",
            "cluster_id": cluster_id,
            "cluster_stats": cluster_stats,
            "fraud_type": fraud_type,
            "metrics": metrics,
            "time_step": time_step,
            "neighbors": neighbors,
            "feature_profile": feature_profile,
            "explanation": explanation
        }
        
    def predict_custom_input(self, features_dict, model_name='voting'):
        """
        Predicts fraud score and label on custom transaction input features.
        
        Parameters:
        - features_dict: dict containing the 165 transaction features.
        - model_name: str ('rf', 'xgb', 'voting', 'stacking')
        """
        if self.scaler is None:
            raise RuntimeError("Scaler is not initialized. Please train models first.")
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not loaded.")
            
        # Ensure all 165 features are present
        feature_cols = [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
        input_data = []
        for col in feature_cols:
            input_data.append(float(features_dict.get(col, 0.0)))
            
        # Scale and predict
        df_input = pd.DataFrame([input_data], columns=feature_cols)
        X_scaled = self.scaler.transform(df_input)
        
        model = self.models[model_name]
        prob = float(model.predict_proba(X_scaled)[0, 1])
        label_pred = int(model.predict(X_scaled)[0])
        
        return {
            "fraud_score": prob,
            "label": "Fraudulent" if label_pred == 1 else "Legitimate",
            "model_used": model_name
        }
