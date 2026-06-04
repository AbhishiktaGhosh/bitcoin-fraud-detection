import networkx as nx
import numpy as np

def classify_fraud_type(G, community_nodes):
    """
    Computes structural metrics for a set of community nodes and classifies
    the subgraph pattern as a 'fraud_ring', 'hub_spoke', or 'chain/mixed'.
    
    Parameters:
    - G: nx.DiGraph or nx.Graph (the full graph)
    - community_nodes: list of node IDs in the community
    
    Returns:
    - pattern: str ('fraud_ring', 'hub_spoke', 'chain/mixed')
    - metrics: dict containing calculated metrics:
      - 'avg_degree': average node degree in community
      - 'density': subgraph density
      - 'clustering': average clustering coefficient
    """
    if len(community_nodes) <= 1:
        return "chain/mixed", {"avg_degree": 0.0, "density": 0.0, "clustering": 0.0}
        
    subgraph = G.subgraph(community_nodes)
    
    # Calculate metrics
    degrees = [d for _, d in subgraph.degree()]
    avg_degree = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
    density = float(nx.density(subgraph))
    
    # Clustering coefficient (convert to undirected copy)
    UG_sub = subgraph.to_undirected()
    clustering = float(nx.average_clustering(UG_sub))
    
    # Static Default Thresholds derived from Elliptic dataset percentiles
    dens_thr = 0.08
    clust_thr = 0.15
    deg_thr = 3.0
    
    if density > dens_thr or clustering > clust_thr:
        pattern = "fraud_ring"
    elif avg_degree > deg_thr:
        pattern = "hub_spoke"
    else:
        pattern = "chain/mixed"
        
    return pattern, {
        "avg_degree": avg_degree,
        "density": density,
        "clustering": clustering
    }

def classify_batch_communities(G, clusters_dict):
    """
    Classifies a batch of clusters using adaptive thresholds calculated 
    from percentiles of the metrics across all clusters.
    
    Parameters:
    - G: nx.Graph or nx.DiGraph
    - clusters_dict: dict mapping cluster IDs to lists of node IDs
    
    Returns:
    - cluster_patterns: dict mapping cluster IDs to pattern strings
    - cluster_metrics: dict mapping cluster IDs to metric dicts
    """
    deg_list = []
    dens_list = []
    clust_list = []
    
    cluster_metrics = {}
    
    for cid, nodes in clusters_dict.items():
        if len(nodes) <= 1:
            cluster_metrics[cid] = (0.0, 0.0, 0.0)
            continue
            
        subgraph = G.subgraph(nodes)
        deg = np.mean([d for _, d in subgraph.degree()])
        density = nx.density(subgraph)
        clustering = nx.average_clustering(subgraph.to_undirected())
        
        deg_list.append(deg)
        dens_list.append(density)
        clust_list.append(clustering)
        
        cluster_metrics[cid] = (deg, density, clustering)
        
    # Handle empty lists gracefully
    if len(deg_list) == 0:
        deg_thr, dens_thr, clust_thr = 3.0, 0.08, 0.15
    else:
        deg_thr = np.percentile(deg_list, 80)
        dens_thr = np.percentile(dens_list, 75)
        clust_thr = np.percentile(clust_list, 70)
        
    cluster_patterns = {}
    metrics_summary = {}
    for cid, metrics in cluster_metrics.items():
        deg, density, clustering = metrics
        
        if len(clusters_dict[cid]) <= 1:
            pattern = "chain/mixed"
        elif density > dens_thr or clustering > clust_thr:
            pattern = "fraud_ring"
        elif deg > deg_thr:
            pattern = "hub_spoke"
        else:
            pattern = "chain/mixed"
            
        cluster_patterns[cid] = pattern
        metrics_summary[cid] = {
            "avg_degree": deg,
            "density": density,
            "clustering": clustering
        }
        
    return cluster_patterns, metrics_summary
