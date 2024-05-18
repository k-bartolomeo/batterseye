import numpy as np


def get_labels_and_weights(embeddings, labels, idx, all_idxs, k, distance):
    if distance == 'euclidean':
        def distance_fn(a, b):
            return np.linalg.norm(a - b, axis=1)
    elif distance == 'cosine':
        def distance_fn(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        raise ValueError("Argument `kind` must be one of {euclidean, cosine}")
    
    neighbor_idxs = all_idxs[all_idxs != idx]
    idx_map = dict(zip(range(neighbor_idxs.shape[0]), neighbor_idxs))
    distances = distance_fn(embeddings[neighbor_idxs], embeddings[idx])
    
    neighbors = np.argpartition(distances, k)[:k]
    neighbor_labels = labels[[idx_map[neighbor] for neighbor in neighbors]]
    unique_labels = np.unique(neighbor_labels)
    neighbor_label_idxs = [
        np.argwhere(neighbor_labels == x).flatten() for x in unique_labels
    ]
    
    inverse_distances = 1 / distances[neighbors]
    weights = inverse_distances / inverse_distances.sum()
    aggregated = [weights[idx].sum() for idxs in neighbor_label_idxs]
    return unique_labels, aggregated


def get_img_label(embeddings, labels, idx, all_idxs, k, distance, return_prob=False):
    labels, weights = get_labels_and_weights(
        embeddings=embeddings,
        labels=labels,
        idx=idx,
        all_idxs=all_idxs,
        k=k,
        distance=distance
    )

    label = labels[np.argmax(weights)]
    if return_prob:
        return label, np.max(weights)
    return label, None


def get_top_k_hit(embeddings, labels, idx, all_idxs, k, top_k, distance):
    labels, weights = get_labels_and_weights(
        embeddings=embeddings,
        labels=labels,
        idx=idx,
        all_idxs=all_idxs,
        k=k,
        distance=distance
    )
    top_k = min(len(weights), top_k)
    top_k_idxs = np.argpartition(weights, -top_k)[-top_k:]
    top_k_labels = labels[top_k_idxs]
    top_k_hit = labels[idx] in top_k_labels
    return top_k_hit
