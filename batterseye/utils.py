import numpy as np

from tqdm import tqdm
from .neighbors import get_img_label, get_top_k_hit


def get_predictions(embeddings, labels, k, distance, return_prob):
    predicted_labels = []
    predicted_probs = []
    idxs = range(embeddings.shape[0])

    for idx in tqdm(idxs):
        label, prob = get_img_label(
            embeddings=embeddings,
            labels=labels,
            idx=idx,
            all_idxs=idxs,
            k=k,
            distance=distance,
            return_prob=return_prob
        )
        predicted_labels.append(label)
        predicted_probs.append(prob)

    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs) if return_prob else None
    
    return predicted_labels, predicted_probs


def get_acc(predicted_labels, labels):
    return (predicted_labels == labels).sum() / len(labels)    


def get_top_k_acc(embeddings, labels, k, top_k, distance):
    results = []
    idxs = range(embeddings.shape[0])

    for idx in tqdm(idxs):
        result = get_top_k_hit(
            embeddings=embeddings,
            labels=labels,
            idx=idx, 
            all_idxs=idxs, 
            k=k, 
            top_k=top_k, 
            distance=distance
        )
        results.append(result)

    acc = sum(results) / len(results)
    return acc


def evaluate(embeddings, labels, k, distance, return_prob, top_k):
    predictions, probs = get_predictions(
        embeddings=embeddings,
        labels=labels,
        k=k,
        distance=distance,
        return_prob=return_prob
    )
    acc = get_acc(predictions, labels)
    top_k_acc = (
        get_top_k_acc(
            embeddings=embeddings,
            labels=labels,
            k=k,
            top_k=top_k,
            distance=distance
        )
        if top_k is not None
        else None
    )

    return acc, top_k_acc, predictions, probs
