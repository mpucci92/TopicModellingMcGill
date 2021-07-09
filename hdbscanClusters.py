import hdbscan

def topicClusters(cluster_size, metric,cluster_selection_method,cluster_selection_epsilon,embeddings):
    cluster = hdbscan.HDBSCAN(min_cluster_size=cluster_size, # 5,euclidean,eom,umap_embeddings
                              metric=metric,
                              cluster_selection_method=cluster_selection_method,
                              cluster_selection_epsilon=cluster_selection_epsilon).fit(embeddings)
    return cluster