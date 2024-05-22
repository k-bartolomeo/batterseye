import numpy.typing as npt

import pandas as pd
import tensorflow as tf

from tqdm import tqdm


def get_condition(subset: pd.DataFrame, label: int, mode: str, k: int) -> pd.Series:
    """Gets evaluation condition for given parameters"""
    if mode == 'atleast':
        cond = subset.other_label.isin([label]).sum() >= k
    elif mode == 'exact':
        cond = subset.other_label.isin([label]).sum() == k
    else:
        raise ValueError(
            "Value for argument `mode` must be one of {'atleast', 'exact'}"
        )
    return cond


def get_results(
    df: pd.DataFrame, k: int, mode: str, top_k: int
) -> tuple[int, int, float, list[tuple]]:
    """Calculates model accuracy with given parameters"""
    correct = 0
    chances = 0
    misses = []

    for label in tqdm(df.true_label.unique()):
        subset = df[df.true_label == label].sort_values(by='probability')
        cond = get_condition(subset=subset, label=label, mode=mode, k=k)
        if cond:
            chances += 1
            other_lbl = subset.other_label.iloc[:top_k].tolist()
            if label in other_lbl:
                correct += 1
            else:
                misses.append((other_lbl, label))

    acc = correct / chances
    return correct, chances, acc, misses


def evaluate(
    predictions: tf.Tensor | npt.NDArray, 
    labels: list, 
    mode: str = 'atleast',
    k: int = 1,
    top_k: int = 1
) -> tuple[int, int, float, list[tuple]]:
    """Evaluates Siamese network's classification predictions

    Builds DataFrame of predictions, test image's true label, and
    comparison image's label. Predictions are probabilities that
    two images are similar. Gets number of correct predictions, 
    number of chances, accuracy of predictions, and pairs of player
    IDs where prediction was incorrect. Chances defined as opportunity
    for model to make correct prediction, i.e. the model was given
    at least `k` pairs of images where the comparison image had the
    same label as the test image's true label.

    Parameters
    ----------
    predictions
        Array of probabilities denoting whether images 
        in pairs are different.
    labels
        List of tuples of comparison image labels and test images'
        true labels.
    mode
        Method to use for including test labels in accuracy 
        calculations. If set to 'atleast', then at least `k` comparison
        images with same label needed for test label to be included
        in accuracy calculation. If set to 'exact', then only test 
        labels where number of comparison images with same label is `k`
        are included in accuracy calculation.
    k
        Minimum or exact number of comparison images with matching 
        label for test label to be included in accuracy calculation.
    top_k
        K to use in top-k accuracy calculation. Default set to 1
        such that if no value is provided, function defaults to
        returning standard accuracy.

    Returns
    -------
    Four-item tuple. First item is number of correct predictions. 
    Second item is number of chances under given parameters. Third
    item is either standard prediction accuracy or top-k prediction
    accuracy. Fourth item is list of pairs of player IDs for 
    incorrect predictions.

    Example
    -------
    >>> k = 1
    >>> top_k = 1
    >>> predictions
    [0.994, 0.032, 0.645, 0.122]
    >>> labels
    [(2, 1), (1, 1), (3, 2), (4, 4)]
    >>> correct, chances, acc, misses = evaluate(
    ...     predictions=predictions,
    ...     labels=labels,
    ...     mode='atleast',
    ...     k=k,
    ...     top_k=top_k
    ... )
    >>> correct, chances, acc
    2, 2, 1.00
    >>> misses
    []

    For the above example, there are 3 distinct true labels - 1, 2, 
    and 4. Only the labels 1 and 4 are included in the calculation
    because for label 2, there is pairing in the dataset where the
    model could have made the correct prediction.
    """
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    df = pd.DataFrame({
        'probability': predictions,
        'other_label': [x[0] for x in labels],
        'true_label': [x[1] for x in labels]
    })

    correct, chances, accuracy, misses = get_results(df=df, k=k, mode=mode, top_k=top_k)
    return correct, chances, accuracy, misses
