from collections import defaultdict
import numpy as np
import community.community_louvain as community_louvain

def detect_communities(G, node_probs):
    """
    Applies Louvain community detection to an undirected copy of G.
    Computes statistics for each community based on node_probs.
    
    Parameters:
    - G: nx.DiGraph or nx.Graph
    - node_probs: dict mapping node IDs (transaction IDs) to fraud probabilities
    
    Returns:
    - partition: dict mapping node IDs to cluster IDs
    - cluster_stats: dict mapping cluster IDs to stat dictionaries:
      - 'size': number of nodes in cluster
      - 'mean': average fraud probability
      - 'max': maximum fraud probability
      - 'p90': 90th percentile of fraud probability
    """
    print("Converting graph to undirected for Louvain community detection...")
    UG = G.to_undirected()
    
    print("Running Louvain partitioning...")
    partition = community_louvain.best_partition(UG)
    
    # Group nodes by cluster
    clusters = get_cluster_assignments(partition)
    
    # Calculate stats
    cluster_stats = {}
    for cid, nodes in clusters.items():
        # Get scores for nodes in the cluster
        scores = [node_probs[n] for n in nodes if n in node_probs]
        if len(scores) == 0:
            continue
        scores = np.array(scores)
        cluster_stats[cid] = {
            "size": len(nodes),
            "mean": float(np.mean(scores)),
            "max": float(np.max(scores)),
            "p90": float(np.percentile(scores, 90))
        }
        
    return partition, cluster_stats

def get_cluster_assignments(partition):
    """
    Groups node IDs by cluster ID based on the partition.
    
    Returns:
    - clusters: dict mapping cluster IDs to lists of node IDs
    """
    clusters = defaultdict(list)
    for node, cid in partition.items():
        clusters[cid].append(node)
    return dict(clusters)
