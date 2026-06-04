import pandas as pd
import numpy as np
import networkx as nx

def build_features(df_merged):
    """
    Extracts tabular features from the merged DataFrame.
    """
    feature_cols = [col for col in df_merged.columns if col.startswith('trans_feat_') or col.startswith('agg_feat_')]
    return df_merged[['id', 'time_step'] + feature_cols]

def create_graph_features(G, node_ids=None):
    """
    Computes graph-derived features for each node in node_ids.
    Features:
    - degree: total degree
    - in_degree: number of incoming edges
    - out_degree: number of outgoing edges
    - clustering_coeff: local clustering coefficient
    """
    if node_ids is None:
        node_ids = list(G.nodes())
        
    # Degrees
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    degrees = dict(G.degree())
    
    # Local clustering coefficient (for directed or undirected)
    # Convert to undirected for standard clustering coefficient
    UG = G.to_undirected()
    clustering = nx.clustering(UG)
    
    features_list = []
    for node in node_ids:
        features_list.append({
            "id": node,
            "degree": degrees.get(node, 0),
            "in_degree": in_degrees.get(node, 0),
            "out_degree": out_degrees.get(node, 0),
            "clustering_coef": clustering.get(node, 0.0)
        })
        
    return pd.DataFrame(features_list)
