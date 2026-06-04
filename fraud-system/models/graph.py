import pandas as pd
import networkx as nx

def build_graph(edges_df, features_df):
    """
    Constructs a directed NetworkX graph G where nodes represent actual transaction IDs.
    
    Parameters:
    - edges_df: DataFrame with columns ['txId1', 'txId2']
    - features_df: DataFrame where first column or 'id' column contains transaction IDs.
    
    Returns:
    - G: nx.DiGraph
    """
    print("Building NetworkX directed graph...")
    G = nx.DiGraph()
    
    # Get all node IDs
    if 'id' in features_df.columns:
        node_ids = features_df['id'].values
    else:
        node_ids = features_df.iloc[:, 0].values
        
    G.add_nodes_from(node_ids)
    
    # Identify valid nodes set for fast lookup
    node_set = set(node_ids)
    
    # Extract edges
    valid_edges = []
    for u, v in zip(edges_df['txId1'], edges_df['txId2']):
        if u in node_set and v in node_set:
            valid_edges.append((u, v))
            
    G.add_edges_from(valid_edges)
    print(f"Graph construction complete. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def create_edges(edges_df, node_to_idx):
    """
    Helper to map transaction edge IDs to integer indices for GNN models.
    """
    edges_src = edges_df['txId1'].map(node_to_idx)
    edges_dst = edges_df['txId2'].map(node_to_idx)
    
    valid = edges_src.notna() & edges_dst.notna()
    src = edges_src[valid].astype(int).values
    dst = edges_dst[valid].astype(int).values
    
    return src, dst
