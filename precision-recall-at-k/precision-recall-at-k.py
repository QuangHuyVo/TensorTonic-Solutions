import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    recommended = np.asarray(recommended)
    relevant = np.asarray(relevant)

    top_k = recommended[:k]

    top_k_set = set(top_k)
    relevant_set = set(relevant)

    hits = top_k_set & relevant_set

    precision = len(hits) / k if k > 0 else 0
    recall = len(hits) / len(relevant_set) if len(relevant_set) > 0 else 0

    return [precision, recall]
    