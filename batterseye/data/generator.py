import math
import random

from typing import Generator

import tensorflow as tf


class PairGenerator:
    """Image pair generator for Siamese Network

    Creates image pair generator that can then be used to construct
    TensorFlow dataset. Generator necessary because overhead needed
    to load all image pairs into memory as NumPy causes memory error.
    Instead, separate lists of image pairs and labels are passed to
    generator, which then passes them to TensorFlow dataset during
    training and evaluation.

    Attributes
    ----------
    pairs : list[tuple[tf.Tensor, tf.Tensor]]
        List of image pairs of type tf.Tensor.
    labels : list[int] | list[tuple[int, int]]
        List of binary values indicating whether pair of images
        are of same player or different players, or list of pairs
        of player IDs.
    training : bool
        Boolean indicating whether the data should be shuffled
        for model training.
    batch_size : int | None
        Optional batch size used for constructing TensorFlow dataset.

    Parameters
    ----------
    pairs
        List of image pairs of type tf.Tensor.
    labels
        List of binary values indicating whether pair of images
        are of same player or different players, or list of pairs
        of player IDs.
    training
        Boolean indicating whether the data should be shuffled
        for model training.
    batch_size
        Optional batch size used for constructing TensorFlow dataset.
    """
    def __init__(
        self,
        pairs: list[tuple[tf.Tensor, tf.Tensor]],
        labels: list[int] | list[tuple[int, int]],
        training: bool = False,
        batch_size: int | None = None,
    ) -> None:
        self.pairs = pairs
        self.labels = labels
        self.training = training
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Gets number of image pairs"""
        return len(self.pairs)
    
    def __call__(self) -> Generator:
        """Produces image pairs and their label"""
        pairs = list(zip(self.pairs, self.labels))
        if self.training:
            random.shuffle(pairs)
        for i in range(len(pairs)):
            imgs = pairs[i][0]
            label = pairs[i][1]
            yield {'x1': imgs[0], 'x2': imgs[1]}, label

    @property
    def steps(self) -> int:
        """Gets number of steps in single pass of data"""
        if self.batch_size is None:
            return len(self)
        return math.ceil(len(self) / self.batch_size)
