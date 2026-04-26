import numpy as np

def silhouette_score(X, labels):
    n = X.shape[0]
    
    # pairwise distance matrix
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    unique_labels = np.unique(labels)
    clusters = {lab: np.where(labels == lab)[0] for lab in unique_labels}
    
    silhouettes = []
    
    for i in range(n):
        label = labels[i]
        same_cluster = clusters[label]
        
        # a(i): intra-cluster distance
        if len(same_cluster) > 1:
            a = np.mean([dist[i][j] for j in same_cluster if j != i])
        else:
            silhouettes.append(0)
            continue
        
        # b(i): nearest other cluster
        b = float("inf")
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster = clusters[other_label]
            avg_dist = np.mean([dist[i][j] for j in other_cluster])
            b = min(b, avg_dist)
        
        s = (b - a) / max(a, b)
        silhouettes.append(s)
    
    return float(np.mean(silhouettes))